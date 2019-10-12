import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Swish', 'PointProduct', 'PointAdd', 'AvgPool']


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

class AvgPool(nn.Module):
    """
    Average pooling layers. This is to make sure all operations (including accumulation) are conducted in FP16.
    """
    def __init__(self):
        return super(AvgPool, self).__init__()

    def forward(slef, x):
        assert(x.dim() == 4)
        N, C, H, W = x.shape
        y = x.view(N*C, -1).mean(dim=1).view(N, C, 1, 1)

        return y

