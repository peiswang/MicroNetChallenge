import torch
import torch.nn as nn

class LabelSmoothLoss(nn.Module):
  def __init__(self, eps = 0.1, reduction = 'mean'):
    super().__init__()
    self.eps = eps
    self.log_softmax = nn.LogSoftmax(dim = 1)
    self.reduction = reduction
    
  def _label_smooth(self, target, num_classes):
    e = self.eps / (num_classes - 1)
    label = torch.ones(len(target), num_classes) * e
#    print('batch size: %d'%(len(target)))
    for i in range(len(target)):
      _label = target[i].item()
      label[i][_label] = 1 - self.eps
      
    return label.to(target.device)
      
    
  def forward(self, x, target):
    num_classes = x.shape[1]
#    print("class number: %d"%(num_classes))
    label = self._label_smooth(target, num_classes)
    x = self.log_softmax(x)
    loss = torch.sum(- x * label, dim = 1)
    if self.reduction == 'none':
      return loss
    elif self.reduction == 'sum':
      return torch.sum(loss)
    elif self.reduction == 'mean':
      return torch.mean(loss)
    else:
      raise ValueError('Unrecognized option, expect reduction to be one of none, mean, sum')
        