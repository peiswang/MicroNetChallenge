import numpy as np

import transform as my_transform

import augmentation_transform

from torchvision import transforms
        
class AutoAug(object):
  def __init__(self, num = 2):
    self.good_policies = my_transform.good_policies
    self.num = num
    
  def __call__(self, data):
    x = data
    for i in range(self.num):
      epoch_policy = self.good_policies[np.random.choice(len(self.good_policies))]
      x = augmentation_transform.apply_policy(epoch_policy, data)
    x = augmentation_transform.zero_pad_and_crop(x, 4)
    x = augmentation_transform.random_flip(x)
    x = augmentation_transform.cutout_numpy(x)
    return x
    
