from __future__ import division

import math

class WarmupLR(object):
  def __init__(self, batch_size, num_samples, base_lr, target_lr, warm_up_epoch):
    self.batch_size = batch_size
    self.num_samples = num_samples
    self.base_lr = base_lr
    self.target_lr = target_lr
    self.warm_up_epoch = warm_up_epoch
    self.batch_idx = 0
    self.iterations_per_epoch = math.ceil(num_samples / batch_size)
    self.warm_up_iterations = warm_up_epoch * self.iterations_per_epoch
    
  def _calculate_lr(self):
    if self.batch_idx < self.warm_up_iterations:
      return self.base_lr + (self.lr - self.base_lr) / warm_up_iterations * (batch_idx + 1)
    else: 
      return self.lr
      
  @property
  def lr(self):
    return self._calculate_lr()
    
  def __call__(self, optimizer):
    lr = self._calculate_lr()
    self.batch_idx += 1
    for param_group in optimizer.param_groups:
      param_group['lr'] = lr
