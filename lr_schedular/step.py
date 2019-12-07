from __future__ import division

import math

class StepLR(object):
  def __init__(self, batch_size, num_samples, init_lr, gamma, decay_epochs):
    self.batch_size = batch_size
    self.num_samples = num_samples
    self.init_lr = init_lr
    self.gamma = gamma
    self.decay_epochs = decay_epochs
    self.batch_idx = 0
    self.iterations_per_epoch = math.ceil(num_samples / batch_size)
    self.decay_iterations = [n * self.iterations_per_epoch for n in decay_epochs]
    
  @property
  def lr(self):
    lr = self.init_lr
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
    