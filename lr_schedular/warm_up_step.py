from __future__ import division

import math

class WarmupStepLR(object):
  def __init__(self, batch_size, num_samples, base_lr, target_lr, warm_up_epoch, epochs, gamma, decay_epochs):
    self.batch_size = batch_size
    self.num_samples = num_samples
    self.base_lr = base_lr
    self.target_lr = target_lr
    self.warm_up_epoch = warm_up_epoch
    self.epochs = epochs
    self.gamma = gamma
    self.decay_epochs = decay_epochs
    self.iterations_per_epoch = math.ceil(num_samples / batch_size)
    self.warm_up_iterations = warm_up_epoch * self.iterations_per_epoch
    self.decay_iterations = [batch * self.iterations_per_epoch for batch in decay_epochs]
    self.batch_idx = 0
    
  @property
  def lr(self):
    if self.batch_idx < self.warm_up_iterations:
      return self.base_lr + (self.target_lr - self.base_lr) / self.warm_up_iterations * (self.batch_idx + 1)
    else:
      lr = self.target_lr
      for iteration in self.decay_iterations:
        if self.batch_idx >= iteration:
          lr *= self.gamma
        else:
          break
      return lr
      
  def __call__(self, optimizer):
    lr = self.lr
    self.batch_idx += 1
    for param_group in optimizer.param_groups:
      param_group['lr'] = lr
      