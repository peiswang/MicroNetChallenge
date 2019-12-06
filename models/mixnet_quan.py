import torch
import torch.nn as nn

import numpy as np
from collections import OrderedDict
from quantization.quantize import Quantization

__all__ = ['mixnet_m_quan', 'mixnet_l_quan', 'mixnet_s_quan']

class AvgPool(nn.Module):
    def __init__(self):
        return super(AvgPool, self).__init__()

    def forward(slef, x):
        assert(x.dim() == 4)
        N, C, H, W = x.shape
        y = x.view(N*C, -1).mean(dim=1).view(N, C, 1, 1)

        return y


class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        #return x * torch.nn.functional.sigmoid(x)
        return x * torch.sigmoid(x)


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


def drop_connect(inputs, training=False, drop_connect_rate=0.):
    """Apply drop connect."""
    if not training:
        return inputs

    keep_prob = 1 - drop_connect_rate
    random_tensor = keep_prob + torch.rand(
        (inputs.size()[0], 1, 1, 1), dtype=inputs.dtype, device=inputs.device)
    random_tensor.floor_()  # binarize
    output = inputs.div(keep_prob) * random_tensor
    return output


class DepthwiseConv2D(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, bias=False):
        super(DepthwiseConv2D, self).__init__()
        padding = (kernel_size - 1) // 2

        self.depthwise_conv = nn.Sequential(Quantization(), nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, stride=stride, groups=in_channels, bias=bias))

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
            self.group_conv = nn.Sequential(Quantization(), nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        else:
            self.group_layers = nn.ModuleList()
            for idx in range(n_chunks):
                self.group_layers.append(nn.Sequential(Quantization(), nn.Conv2d(self.split_in_channels[idx], split_out_channels[idx], kernel_size=kernel_size, bias=bias)))

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
            #nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
            #nn.BatchNorm2d(out_channels),
            self.activation
        )

        self.se_expand = nn.Sequential(
            GroupConv2D(out_channels, in_channels, bias=True),
            #nn.Conv2d(out_channels, in_channels, kernel_size=1, bias=True),
            #nn.BatchNorm2d(in_channels),
        )
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_pool = AvgPool()

    def forward(self, x):
        #se_tensor = torch.mean(x, dim=[2, 3], keepdim=True)
        se_tensor = self.avg_pool(x)
        out = self.se_expand(self.se_reduce(se_tensor))
        # Additional Code to count this part if FLOPs
        out = torch.sigmoid(out) * x

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
        self.drop_connect_rate = drop_connect_rate

        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                GroupConv2D(in_channels, in_channels * expand_ratio, n_chunks=expand_ksize),
                nn.BatchNorm2d(in_channels * expand_ratio),
                self.activation
            )

            self.mdconv = nn.Sequential(
                MDConv(in_channels * expand_ratio, n_chunks=n_chunks, stride=stride),
                nn.BatchNorm2d(in_channels * expand_ratio),
                self.activation
            )

            if self._has_se:
                num_reduced_filters = max(1, int(in_channels * se_ratio))
                self.squeeze_excitation = SqueezeExcitation(in_channels * expand_ratio, num_reduced_filters, swish)

                self.project_conv = nn.Sequential(
                    GroupConv2D(in_channels * expand_ratio, out_channels, n_chunks=project_ksize),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.project_conv = nn.Sequential(
                    GroupConv2D(in_channels * expand_ratio, out_channels, n_chunks=project_ksize),
                    nn.BatchNorm2d(out_channels),
                )
        else:
            self.mdconv = nn.Sequential(
                MDConv(in_channels, n_chunks=n_chunks, stride=stride),
                nn.BatchNorm2d(in_channels),
                self.activation
            )

            if self._has_se:
                num_reduced_filters = max(1, int(in_channels * se_ratio))
                self.squeeze_excitation = SqueezeExcitation(in_channels, num_reduced_filters, swish)

                self.project_conv = nn.Sequential(
                    GroupConv2D(in_channels, out_channels, n_chunks=project_ksize),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.project_conv = nn.Sequential(
                    GroupConv2D(in_channels * expand_ratio, out_channels, n_chunks=project_ksize),
                    nn.BatchNorm2d(out_channels),
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
            if self.drop_connect_rate > 0.:
                out = drop_connect(out, self.training, self.drop_connect_rate)
            out = out + x
        return out


class MixNet(nn.Module):
    def __init__(self, stem, head, last_out_channels, block_args, dropout=0.2, num_classes=1000):
        super(MixNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=stem, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem),
            nn.ReLU(),
        )

        layers = []
        for in_channels, out_channels, n_chunks, stride, expand_ratio, se_ratio, swish, expand_ksize, project_ksize, dropconnect in block_args:
            layers.append(MixBlock(in_channels, out_channels, n_chunks, stride, expand_ratio, se_ratio, swish, expand_ksize, project_ksize, dropconnect))

        self.layers = nn.Sequential(*layers)

        self.head_conv = nn.Sequential(
            #nn.Conv2d(last_out_channels, head, kernel_size=1, bias=False),
            GroupConv2D(last_out_channels, head, kernel_size=1, bias=False),
            nn.BatchNorm2d(head),
            nn.ReLU(),
        )

        #self.avg_pool2d = nn.AvgPool2d(4)
        # self.adapt_avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.adapt_avg_pool2d = AvgPool()

        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            Quantization(),
            nn.Linear(head, num_classes),
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


def mixnet_s_quan(pretrained=False, num_classes=1000, multiplier=1.0, divisor=8, dropout=0.2, dropconnect=0.0, min_depth=None, bit_width=8):
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


def mixnet_m_quan(pretrained=False, num_classes=1000, multiplier=1.0, divisor=8, dropout=0.2, dropconnect=0.0, min_depth=None, bit_width=8):
    medium = [
        [24, 24, 1, 1, 1, None, False, 1, 1, dropconnect],
        [24, 32, 3, 2, 6, None, False, 2, 2, dropconnect],
        [32, 32, 1, 1, 3, None, False, 2, 2, dropconnect],
        [32, 40, 4, 2, 6, 0.5, True, 1, 1, dropconnect],
        [40, 40, 2, 1, 6, 0.5, True, 2, 2, dropconnect],

        [40, 40, 2, 1, 6, 0.5, True, 2, 2, dropconnect],
        [40, 40, 2, 1, 6, 0.5, True, 2, 2, dropconnect],
        [40, 80, 3, 2, 6, 0.25, True, 1, 1, dropconnect],
        [80, 80, 4, 1, 6, 0.25, True, 2, 2, dropconnect],
        [80, 80, 4, 1, 6, 0.25, True, 2, 2, dropconnect],

        [80, 80, 4, 1, 6, 0.25, True, 2, 2, dropconnect],
        [80, 120, 1, 1, 6, 0.5, True, 1, 1, dropconnect],
        [120, 120, 4, 1, 3, 0.5, True, 2, 2, dropconnect],
        [120, 120, 4, 1, 3, 0.5, True, 2, 2, dropconnect],
        [120, 120, 4, 1, 3, 0.5, True, 2, 2, dropconnect],

        [120, 200, 4, 2, 6, 0.5, True, 1, 1, dropconnect],
        [200, 200, 4, 1, 6, 0.5, True, 1, 2, dropconnect],
        [200, 200, 4, 1, 6, 0.5, True, 1, 2, dropconnect],
        [200, 200, 4, 1, 6, 0.5, True, 1, 2, dropconnect]
    ]
    for line in medium:
        line[0] = round_filters(line[0], multiplier)
        line[1] = round_filters(line[1], multiplier)

    stem = round_filters(24, multiplier)
    last_out_channels = round_filters(200, multiplier)
    head = round_filters(1536, multiplier=1.0)

    model = MixNet(stem=stem, head=head, last_out_channels=last_out_channels, block_args=medium, num_classes=num_classes, dropout=dropout)
    
    if pretrained:
        init_model(model, pretrained)
    return model

def mixnet_l_quan(pretrained=False, num_classes=1000, dropout=0.2, dropconnect=0.0, bit_width=8):
    return mixnet_m_quan(pretrained, num_classes=num_classes, multiplier=1.3, dropout=dropout, dropconnect=dropconnect, bit_width=bit_width)
