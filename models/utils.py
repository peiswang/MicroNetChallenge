import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Swish', 'PointProduct', 'PointAdd']


class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * torch.sigmoid(x)

class PointAdd(nn.Module):
    """
    Wrapper for pointwise "+"
    """
    def __init__(self):
        return super(PointAdd, self).__init__()

    def forward(self, x1, x2):      
        return x1 + x2

class PointProduct(nn.Module):
    """
    Wrapper for pointwise "+"
    """
    def __init__(self):
        return super(PointProduct, self).__init__()

    def forward(self, x1, x2):      
        return x1 * x2
