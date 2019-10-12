import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['densenet_final', 'Quant', 'Scale']

from torch.autograd import Variable

class Quant(nn.Module):
  def __init__(self):
    super(Quant, self).__init__()

  def forward(self, x):
    return x.round().clamp(0, 15)

class Scale(nn.Module):
    def __init__(self, channels):
        super(Scale, self).__init__()
        self.weight = nn.Parameter(torch.ones(channels), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(channels), requires_grad=True)

    def forward(self, x):
        if len(x.size()) == 2:
            out = x*self.weight.view(1,-1) + self.bias.view(1,-1)
        else:
            out = x*self.weight.view(1,-1,1,1) + self.bias.view(1,-1,1,1)
        return out

class Bottleneck(nn.Module):
    def __init__(self, inplanes, expansion=4, growthRate=12, dropRate=0):
        super(Bottleneck, self).__init__()
        planes = expansion * growthRate
        self.scale1 = Scale(inplanes)
        self.scale2 = Scale(planes)
        self.scale3 = Scale(growthRate)

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, growthRate, kernel_size=3, padding=1, bias=False)
        self.quant = Quant()

    def forward(self, x):
        out = self.scale1(x)
        out = self.quant(out)
        out = self.conv1(out)
        out = self.scale2(out)
        out = self.quant(out)
        out = self.conv2(out)
        out = self.scale3(out)

        out = torch.cat((x, out), 1)

        return out


class Transition(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()
        self.scale1 = Scale(inplanes)
        self.scale2 = Scale(outplanes)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False)
        self.avg_pool2d = nn.AvgPool2d(2)
        self.quant = Quant()

    def forward(self, x):
        out = self.scale1(x)
        out = self.quant(out)
        out = self.conv1(out)
        out = self.avg_pool2d(out)
        out = self.scale2(out)
        return out


class DenseNet(nn.Module):
    def __init__(self, depth=22, block=Bottleneck, 
        dropRate=0, num_classes=10, growthRate=12, compressionRate=2,
        init = 'Default'):
        super(DenseNet, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
#        n = (depth - 4) / 3 if block == BasicBlock else (depth - 4) // 6
        n = (depth - 4) // 6
        self.growthRate = growthRate
        self.dropRate = dropRate
        # self.inplanes is a global variable used across multiple
        # helper functions
        self.inplanes = growthRate * 2
        self.scale1 = Scale(self.inplanes)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_denseblock(block, n)
        self.trans1 = self._make_transition(compressionRate)
        self.dense2 = self._make_denseblock(block, n)
        self.trans2 = self._make_transition(compressionRate)
        self.dense3 = self._make_denseblock(block, n)
        self.scale2 = Scale(self.inplanes)
        self.scale3 = Scale(num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.inplanes, num_classes, bias=False)
        self.quant = Quant()
        
        # Weight initialization
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
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
        x = self.conv1(x)
        x = self.scale1(x)
        x = self.trans1(self.dense1(x)) 
        x = self.trans2(self.dense2(x)) 
        x = self.dense3(x)
        x = self.scale2(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.scale3(x)

        return x


def densenet_final(**kwargs):
    """
    Constructs a ResNet model.
    """
    return DenseNet(**kwargs)
