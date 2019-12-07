import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..quantize import QuantConv2d

__all__ = ['densenet']


from torch.autograd import Variable

class Bottleneck(nn.Module):
    def __init__(self, inplanes, expansion=4, growthRate=12, dropRate=0):
        super(Bottleneck, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = QuantConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = QuantConv2d(planes, growthRate, kernel_size=3, 
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate
#        self.attention = AttentionLayer(planes, planes)
#        self.with_attention = with_attention

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
#        if self.with_attention:
#            out = self.attention(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, expansion=1, growthRate=12, dropRate=0):
        super(BasicBlock, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = QuantConv2d(inplanes, growthRate, kernel_size=3, 
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out


class Transition(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = QuantConv2d(inplanes, outplanes, kernel_size=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):

    def __init__(self, depth=22, block=Bottleneck, 
        dropRate=0, num_classes=10, growthRate=12, compressionRate=2,
        init = 'Default', prune_fc = False):
        super(DenseNet, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) / 3 if block == BasicBlock else (depth - 4) // 6

        self.growthRate = growthRate
        self.dropRate = dropRate

        # self.inplanes is a global variable used across multiple
        # helper functions
        self.inplanes = growthRate * 2 
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_denseblock(block, n)
        self.trans1 = self._make_transition(compressionRate)
        self.dense2 = self._make_denseblock(block, n)
        self.trans2 = self._make_transition(compressionRate)
        self.dense3 = self._make_denseblock(block, n)
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.inplanes, num_classes)

        self.prune_fc = prune_fc

        # Weight initialization
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        '''
        if init == 'XavierUniform':
            print('Initialize with xavier uniform')
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()
                else:
                    if hasattr(m, 'weight') and m.weight is not None:
                        m.weight.data.fill_(1)
                    if hasattr(m, 'bias') and m.bias is not None:
                        m.bias.data.zero_()
        elif init == 'XavierNormal':
            print('Initialize with xavier normal')
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()
                else:
                    if hasattr(m, 'weight') and m.weight is not None:
                        m.weight.data.fill_(1)
                    if hasattr(m, 'bias') and m.bias is not None:
                        m.bias.data.zero_()
        elif init == 'Default':
            print('Initialize with default initializer')
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    kdim = 1
                    if isinstance(m, nn.Conv2d):
                        kdim = m.kernel_size[0] * m.kernel_size[1]
                        num_outputs = m.out_channels
                    else:
                        num_outputs = m.out_features
                    n = kdim * num_outputs
                    m.weight.data.normal_(0, math.sqrt(2./n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                else:
                    if hasattr(m, 'weight'):
                        m.weight.data.fill_(1)
                    if hasattr(m, 'bias'):
                        m.bias.data.zero_()
        '''
    def _make_denseblock(self, block, blocks):
        layers = []
        for i in range(blocks):
            # Currently we fix the expansion ratio as the default value
            layers.append(block(self.inplanes, growthRate=self.growthRate, dropRate=self.dropRate))
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return Transition(inplanes, outplanes)


    def forward(self, x):
#        if self.prune_fc:
#          with torch.no_grad():
        x = self.conv1(x)
        x = self.trans1(self.dense1(x)) 
        x = self.trans2(self.dense2(x)) 
        x = self.dense3(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)


        return x


def densenet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return DenseNet(**kwargs)
