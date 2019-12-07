import torch
from torch import nn

class HPLoss(object):
  def __init__(self, alpha = 0.1):
    self.ce = nn.CrossEntropyLoss()
    self.alpha = alpha

  def __call__(self, x, target):
    output = x.softmax(dim = 1)
    return self.alpha * (output * output.log()).sum(dim = 1).mean() + self.ce(x, target)
