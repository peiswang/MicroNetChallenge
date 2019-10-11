import torch
import torch.nn as nn

import numpy as np
from collections import OrderedDict
from quantization.quantize import Quantization

from .utils import *

__all__ = ['mixnet_s_final']

class Conv2D_INT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2D_INT, self).__init__()

        self.base = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                                                            dilation=dilation, groups=groups, bias=False)
        self.scale = nn.Parameter(torch.ones(out_channels), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)

    def forward(self, x):
        # print('Conv2D_INT x : ', x.type())
        # if 'Half' in self.bias.type():
        #     x = x.half()
        y = self.base(x)
        # print('Conv2D_INT y : ', y.type())
        # print('Conv2D_INT weight: ', self.base.weight.type())
        # print('Conv2D_INT scale : ', self.scale.type())
        # print('Conv2D_INT bias : ', self.bias.type())
        y *= self.scale.view(1,-1,1,1)
        y += self.bias.view(1,-1,1,1)

        return y

class FC_INT(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(FC_INT, self).__init__()

        self.base = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.scale = nn.Parameter(torch.ones(out_features), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=True)

    def forward(self, x):
        # if 'Half' in self.bias.type():
        #     x = x.half()
        y = self.base(x)
        y *= self.scale.view(1,-1)
        y += self.bias.view(1,-1)
        return y


def split_layer(total_channels, num_groups):
    split = [int(np.ceil(total_channels / num_groups)) for _ in range(num_groups)]
    split[num_groups - 1] += total_channels - sum(split)
    return split


def round_filters(filters, multiplier=1.0, divisor=8, min_depth=None):
    multiplier = multiplier
    divisor = divisor
    min_depth = min_depth
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return new_filters


class DepthwiseConv2D(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, bias=False):
        super(DepthwiseConv2D, self).__init__()
        padding = (kernel_size - 1) // 2

        self.depthwise_conv = nn.Sequential(Quantization(), Conv2D_INT(in_channels, in_channels, kernel_size=kernel_size, padding=padding, stride=stride, groups=in_channels, bias=bias))

    def forward(self, x):
        out = self.depthwise_conv(x)
        return out

class GroupConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, n_chunks=1, bias=False):
        super(GroupConv2D, self).__init__()
        self.n_chunks = n_chunks
        self.split_in_channels = split_layer(in_channels, n_chunks)
        split_out_channels = split_layer(out_channels, n_chunks)

        if n_chunks == 1:
            self.group_conv = nn.Sequential(Quantization(), Conv2D_INT(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        else:
            self.group_layers = nn.ModuleList()
            for idx in range(n_chunks):
                self.group_layers.append(nn.Sequential(Quantization(), Conv2D_INT(self.split_in_channels[idx], split_out_channels[idx], kernel_size=kernel_size, bias=bias)))

    def forward(self, x):
        if self.n_chunks == 1:
            return self.group_conv(x)
        else:
            # ??Ignore FLOPs for Split & Concat Op??
            split = torch.split(x, self.split_in_channels, dim=1)
            out = torch.cat([layer(s) for layer, s in zip(self.group_layers, split)], dim=1)
            return out


class MDConv(nn.Module):
    def __init__(self, out_channels, n_chunks, stride=1):
        super(MDConv, self).__init__()
        self.n_chunks = n_chunks
        self.split_out_channels = split_layer(out_channels, n_chunks)

        self.layers = nn.ModuleList()
        for idx in range(self.n_chunks):
            kernel_size = 2 * idx + 3
            self.layers.append(DepthwiseConv2D(self.split_out_channels[idx], kernel_size=kernel_size, stride=stride))

    def forward(self, x):
        split = torch.split(x, self.split_out_channels, dim=1)
        out = torch.cat([layer(s) for layer, s in zip(self.layers, split)], dim=1)
        return out



class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, out_channels, swish):
        super(SqueezeExcitation, self).__init__()
        self.activation = Swish() if swish else nn.ReLU()
        self.se_reduce = nn.Sequential(
            GroupConv2D(in_channels, out_channels, bias=True),
            self.activation
        )

        self.se_expand = nn.Sequential(
            GroupConv2D(out_channels, in_channels, bias=True),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()
        self.pointproduct = PointProduct()

    def forward(self, x):
        #se_tensor = torch.mean(x, dim=[2, 3], keepdim=True)
        se_tensor = self.avg_pool(x)
        out = self.se_expand(self.se_reduce(se_tensor))
        # Additional Code to count this part if FLOPs
        # out = torch.sigmoid(out) * x
        out = self.pointproduct(self.sigmoid(out), x)

        return out


class MixBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_chunks, stride, expand_ratio, se_ratio, swish, expand_ksize, project_ksize, drop_connect_rate=0.0):
        super(MixBlock, self).__init__()
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._has_se = (se_ratio is not None) and (se_ratio > 0) and (se_ratio <= 1)
        self.activation = Swish() if swish else nn.ReLU()
        self.pointadd = PointAdd()

        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                GroupConv2D(in_channels, in_channels * expand_ratio, n_chunks=expand_ksize),
                self.activation
            )

            self.mdconv = nn.Sequential(
                MDConv(in_channels * expand_ratio, n_chunks=n_chunks, stride=stride),
                self.activation
            )

            if self._has_se:
                num_reduced_filters = max(1, int(in_channels * se_ratio))
                self.squeeze_excitation = SqueezeExcitation(in_channels * expand_ratio, num_reduced_filters, swish)

                self.project_conv = nn.Sequential(
                    GroupConv2D(in_channels * expand_ratio, out_channels, n_chunks=project_ksize),
                )
            else:
                self.project_conv = nn.Sequential(
                    GroupConv2D(in_channels * expand_ratio, out_channels, n_chunks=project_ksize),
                )
        else:
            self.mdconv = nn.Sequential(
                MDConv(in_channels, n_chunks=n_chunks, stride=stride),
                self.activation
            )

            if self._has_se:
                num_reduced_filters = max(1, int(in_channels * se_ratio))
                self.squeeze_excitation = SqueezeExcitation(in_channels, num_reduced_filters, swish)

                self.project_conv = nn.Sequential(
                    GroupConv2D(in_channels, out_channels, n_chunks=project_ksize),
                )
            else:
                self.project_conv = nn.Sequential(
                    GroupConv2D(in_channels * expand_ratio, out_channels, n_chunks=project_ksize),
                )

    def forward(self, x):
        if self.expand_ratio != 1:
            out = self.expand_conv(x)
            out = self.mdconv(out)

            if self._has_se:
                out = self.squeeze_excitation(out)
                out = self.project_conv(out)
            else:
                out = self.project_conv(out)
        else:
            out = self.mdconv(x)
            if self._has_se:
                out = self.squeeze_excitation(out)
                out = self.project_conv(out)
            else:
                out = self.project_conv(out)

        if self.stride == 1 and self.in_channels == self.out_channels:
            out = self.pointadd(out, x)
        return out


class MixNet(nn.Module):
    def __init__(self, stem, head, last_out_channels, block_args, dropout=0.2, num_classes=1000):
        super(MixNet, self).__init__()

        self.conv = nn.Sequential(
            GroupConv2D(3, stem, kernel_size=3, stride=2, padding=1, bias=False),
            # GroupConv2D(3, stem, kernel_size=3, stride=2, padding=0, bias=False),
            nn.ReLU(),
        )

        layers = []
        for in_channels, out_channels, n_chunks, stride, expand_ratio, se_ratio, swish, expand_ksize, project_ksize, dropconnect in block_args:
            layers.append(MixBlock(in_channels, out_channels, n_chunks, stride, expand_ratio, se_ratio, swish, expand_ksize, project_ksize, dropconnect))

        self.layers = nn.Sequential(*layers)

        self.head_conv = nn.Sequential(
            GroupConv2D(last_out_channels, head, kernel_size=1, bias=False),
            nn.ReLU(),
        )

        #self.avg_pool2d = nn.AvgPool2d(4)
        self.adapt_avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            Quantization(),
            FC_INT(head, num_classes),
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.layers(out)
        out = self.head_conv(out)

        #out = self.avg_pool2d(out)
        out = self.adapt_avg_pool2d(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters

def init_model(model, pretrained):
    state_dict = torch.load(pretrained)
    new_state_dict = OrderedDict()
    for key_ori, key_pre in zip(model.state_dict().keys(), state_dict.keys()):
        new_state_dict[key_ori] = state_dict[key_pre]
    model.load_state_dict(new_state_dict)


def mixnet_s_final(pretrained=False, num_classes=1000, multiplier=1.0, divisor=8, dropout=0.2, dropconnect=0.0, min_depth=None, bit_width=8):
    small = [
        # in_channels, out_channels, n_chunks, stride, expqand_ratio, se_ratio, swish, expand_ksize, project_ksize:
        [16, 16, 1, 1, 1, None, False, 1, 1, dropconnect],
        [16, 24, 1, 2, 6, None, False, 2, 2, dropconnect],
        [24, 24, 1, 1, 3, None, False, 2, 2, dropconnect],
        [24, 40, 3, 2, 6, 0.5, True, 1, 1, dropconnect],
        [40, 40, 2, 1, 6, 0.5, True, 2, 2, dropconnect],

        [40, 40, 2, 1, 6, 0.5, True, 2, 2, dropconnect],
        [40, 40, 2, 1, 6, 0.5, True, 2, 2, dropconnect],
        [40, 80, 3, 2, 6, 0.25, True, 1, 2, dropconnect],
        [80, 80, 2, 1, 6, 0.25, True, 1, 2, dropconnect],
        [80, 80, 2, 1, 6, 0.25, True, 1, 2, dropconnect],

        [80, 120, 3, 1, 6, 0.5, True, 2, 2, dropconnect],
        [120, 120, 4, 1, 3, 0.5, True, 2, 2, dropconnect],
        [120, 120, 4, 1, 3, 0.5, True, 2, 2, dropconnect],
        [120, 200, 5, 2, 6, 0.5, True, 1, 1, dropconnect],
        [200, 200, 4, 1, 6, 0.5, True, 1, 2, dropconnect],

        [200, 200, 4, 1, 6, 0.5, True, 1, 2, dropconnect]
    ]

    stem = round_filters(16, multiplier)
    last_out_channels = round_filters(200, multiplier)
    head = round_filters(1536, multiplier)

    model = MixNet(stem=stem, head=head, last_out_channels=last_out_channels, block_args=small, num_classes=num_classes, dropout=dropout)
    
    if pretrained:
        init_model(model, pretrained)
    return model

