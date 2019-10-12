import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['densenet']


from torch.autograd import Variable

class Quant(nn.Module):
  def __init__(self, bitwidth):
    super(Quant, self).__init__()
    self.low = 0
    self.high = 2 ** bitwidth - 1

  def forward(self, x):
    return x.round().clamp(self.low, self.high)

class Scale(nn.Module):
  def __init__(self, inplanes):
    super(Scale, self).__init__()
    self.weight = nn.Parameter(torch.rand(1, inplanes, 1))
    self.bias = nn.Parameter(torch.rand(1, inplanes, 1))

  def forward(self, x):
#    self.half()
    y = x.view(x.size(0), x.size(1), -1)
    return (self.weight * y + self.bias).reshape(x.shape)

class Bottleneck(nn.Module):
    def __init__(self, inplanes, expansion=4, growthRate=12, dropRate=0, bitwidth = 4):
        super(Bottleneck, self).__init__()
        planes = expansion * growthRate
        self.scale1 = Scale(inplanes)
        self.quant = Quant(bitwidth)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.scale2 = Scale(planes)
        self.conv2 = nn.Conv2d(planes, growthRate, kernel_size=3, 
                               padding=1, bias=False)
        self.scale3 = Scale(growthRate)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate
        self.half = False
#        self.attention = AttentionLayer(planes, planes)
#        self.with_attention = with_attention

    def set_half(self):
      self.half = True
      self.scale1.half()
      self.quant.half()
      self.scale2.half()
      self.scale3.half()
      self.relu.half()

    def forward(self, x):
        out = self.scale1(x)
#        out = self.relu(out)
        out = self.quant(out)
        out = out.float()
        out = self.conv1(out)
        if self.half:
          out = out.half()
        out = self.scale2(out)
#        out = self.relu(out)
        out = self.quant(out)
        out = out.float()
        out = self.conv2(out)
        if self.half:
          out = out.half()
        out = self.scale3(out)

        out = torch.cat((x, out), 1)

        return out


class Transition(nn.Module):
    def __init__(self, inplanes, outplanes, bitwidth = 4):
        super(Transition, self).__init__()
        self.scale1 = Scale(inplanes)
        self.quant1 = Quant(bitwidth)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1,
                               bias=False)
        self.scale2 = Scale(outplanes)
        self.half = False
#        self.relu = nn.ReLU(inplace=True)

    def set_half(self):
      self.half = True
      self.scale1.half()
      self.quant1.half()
      self.scale2.half()

    def forward(self, x):
        out = self.scale1(x)
#        out = self.relu(out)
        out = self.quant1(out)
        out = out.float()
        out = self.conv1(out)
        if self.half:
          out = out.half()
        out = F.avg_pool2d(out, 2)
        out = self.scale2(out)

        return out


class DenseNet(nn.Module):

    def __init__(self, depth=22, block=Bottleneck, 
        dropRate=0, num_classes=10, growthRate=12, compressionRate=2,
        init = 'Default', bitwidth = 4):
        super(DenseNet, self).__init__()
        self.half = False

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
#        n = (depth - 4) / 3 if block == BasicBlock else (depth - 4) // 6
        n = (depth - 4) // 6
        self.growthRate = growthRate
        self.dropRate = dropRate

        # self.inplanes is a global variable used across multiple
        # helper functions
        self.inplanes = growthRate * 2 
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False)
        self.scale1 = Scale(self.inplanes)
        self.dense1 = self._make_denseblock(block, n, bitwidth)
        self.trans1 = self._make_transition(compressionRate, bitwidth)
        self.dense2 = self._make_denseblock(block, n, bitwidth)
        self.trans2 = self._make_transition(compressionRate, bitwidth)
        self.dense3 = self._make_denseblock(block, n, bitwidth)
        self.scale2 = Scale(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.inplanes, num_classes, bias = False)
        self.scale3 = Scale(num_classes)
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
    def _make_denseblock(self, block, blocks, bitwidth):
        layers = []
        for i in range(blocks):
            # Currently we fix the expansion ratio as the default value
            layers.append(block(self.inplanes, growthRate=self.growthRate, dropRate=self.dropRate, bitwidth = bitwidth))
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate, bitwidth):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return Transition(inplanes, outplanes, bitwidth = bitwidth)

    def set_half(self):
      self.half = True
      for module in self.modules():
        if isinstance(module, (Bottleneck, Transition)):
          module.set_half()
        elif isinstance(module, (Scale, Quant, nn.ReLU, nn.AvgPool2d)):
          module.half()
      '''
      self.scale1.half()
      self.dense1.set_half()
      self.trans1.set_half()
      self.dense2.set_half()
      self.trans2.set_half()
      self.dense3.set_half()
      self.scale2.half()
      self.relu.half()
      self.avgpool.half()
      self.scale3.half()
      '''

    def forward(self, x):
#        x = x * 512
#        self.conv1.weight.data[:] = self.conv1.weight.data * self.scale1.weight.data.view(24, 1, 1, 1)
#        print(self.conv1.weight.data)
#        print(self.scale1.weight.data)
#        x = (x * 255).half()
        x = self.conv1(x)
        
        if self.half:    
          x = x.half()
#        print(x.max(), x.mean(), x.min())
        x = self.scale1(x)
#        x = x / 255
#        print(x.max(), x.mean(), x.min())

        x = self.trans1(self.dense1(x)) 
#        print(x.max(), x.mean(), x.min())
        x = self.trans2(self.dense2(x)) 
        x = self.dense3(x)
        x = self.scale2(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = x.float()
        x = self.fc(x)
        if self.half:
          x = x.half()
        x = self.scale3(x)

#        print('#################################################')

        return x


def densenet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return DenseNet(**kwargs)
