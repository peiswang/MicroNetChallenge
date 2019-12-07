from __future__ import division

import math

class WarmupCosineLR(object):
  def __init__(self, batch_size, total_samples, epoch, base_lr, target_lr, warm_up_epoch = 5, cur_epoch = 0):
    self.batch_size = batch_size
    self.total_samples = total_samples
    self.epoch = epoch
    self.cur_epoch = cur_epoch
    self.warm_up_epoch = warm_up_epoch
    self.iterations_per_epoch = math.ceil(total_samples / batch_size)
    self.iterations = epoch * self.iterations_per_epoch
    self.warm_up_iterations = warm_up_epoch * self.iterations_per_epoch
    self.batch_idx = cur_epoch * self.iterations_per_epoch
    self.base_lr = base_lr
    self.target_lr = target_lr
    self.coies = [math.cos((i - self.warm_up_iterations) * math.pi / (self.iterations - self.warm_up_iterations)) for i in range(self.iterations)]
    
  def _calculate_lr(self):
    if self.batch_idx < self.warm_up_iterations:
      return self.base_lr + (self.target_lr - self.base_lr) / self.warm_up_iterations * self.batch_idx
    else:
      return 0.5 * (1 + self.coies[self.batch_idx]) * self.target_lr
    
  @property
  def lr(self):
    return self._calculate_lr()
    
  def __call__(self, optimizer):
    lr = self._calculate_lr()
    for param_group in optimizer.param_groups:
      param_group['lr'] = lr
    self.batch_idx += 1
