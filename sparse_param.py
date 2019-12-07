from __future__ import division

import math
import time

import torch

from torch import nn

class TArrayIndexMaxHeap(object):
  def __init__(self, params, capacity):
    self.params = params
    self.capacity = capacity
    self.indicies = []
    
  def param_at(self, index):
    return abs(self.params[index[0]].flatten()[index[1]])
    
  @property
  def extract_max(self):
    if len(self.indicies) == 0:
      raise ValueError('Length of indicies must be lager than one')
    index = self.indicies[0]
    return self.param_at(index)
    
  def insert(self, index):
    if len(self.indicies) < self.capacity:
      self.indicies.append(index)
      self._heap_up()
    else:
      param_at_index = self.param_at(index)
      if param_at_index < self.extract_max:
        self.indicies[0] = index
        self._heap_down()
    
  def _heap_up(self):
    i = len(self.indicies) - 1
    while i > 0:
      p = (i - 1) // 2
      param_i = self.param_at(self.indicies[i])
      param_p = self.param_at(self.indicies[p])
      if param_i <= param_p:
        break
      index = self.indicies[i]
      self.indicies[i] = self.indicies[p]
      self.indicies[p] = index
      i = p
  
  def _heap_down(self):
    i = 0
    while i <= (len(self.indicies) - 1) // 2:
      left = 2 * i + 1
      right = 2 * i + 2
      param_i = self.param_at(self.indicies[i])
      param_left = self.param_at(self.indicies[left])
      if left >= len(self.indicies):
        break
      if right >= len(self.indicies):
        if param_i >= param_left:
          break
        index = self.indicies[i]
        self.indicies[i] = self.indicies[left]
        self.indicies[left] = index
        i = left
      else:
        param_right = self.param_at(self.indicies[right])
        if param_i >= param_left and param_i >= param_right:
          break
        else:
          j = left if param_left > param_right else right
          index = self.indicies[i]
          self.indicies[i] = self.indicies[j]
          self.indicies[j] = index
          i = j
        

class SparseParam(object):
  def __init__(self, masked_params, masks = None, params = None, mask_policy = 'local', mask_grad = False):
    self.masked_params = masked_params
    self.masks = masks
    self.params = params
    device = masked_params[0].device
    for masked_param in masked_params:
      if device != masked_param.device:
        raise ValueError('All the masked parameters must be on the same gpus.')
    if masks is not None:
      self.masks_initialized = True
      for i in range(len(masks)):
        self.masks[i] = masks[i].to(device)
    else:
      self.masks_initialized = False
    if params is not None:
      self.params_initialized = True
      for i in range(len(params)):
        self.params[i] = nn.Parameter(params[i].to(device))
    else:
      self.params_initialized = False
    self.mask_policy = mask_policy
    self.mask_grad = mask_grad
    
  def _init_masks_and_params(self):
    if not self.masks_initialized:
      self.masks = [torch.ones_like(param) for param in self.masked_params]
      self.masks_initialized = True
    if not self.params_initialized:
      params = [torch.empty_like(param, requires_grad = True) for param in self.masked_params]
      for param, masked_param in zip(params, self.masked_params):
        param[:] = masked_param[:]
      self.params = [nn.Parameter(param) for param in params]
      self.params_initialized = True
      
  def _update_mask_local(self, rate):
    for param, mask in zip(self.params, self.masks):
      numel = param.numel()
      num_prune = math.ceil(numel * rate)
      argsort = param.abs().flatten().argsort(descending = False)
      mask.flatten()[argsort[0:num_prune]] = 0
      mask.flatten()[argsort[num_prune + 1:numel]] = 1
  ''' 
  def _update_mask_global(self, rate):
    numel = 0
    for param in self.params:
      numel += param.numel()
    
    num_prune = math.ceil(numel * rate)
    
    heap = TArrayIndexMaxHeap(self.params, num_prune)
    
    sort_s = time.time()
    for i, param in enumerate(self.params):
      numel = param.numel()
      for n in range(numel):
        heap.insert((i, n))
    sort_e = time.time()
    print('sort time: %f'%(sort_e - sort_s))
    
    ones_s = time.time()
    for mask in self.masks:
      mask.fill_(1)
    ones_e = time.time()
    print('ones time: %f'%(ones_e - ones_s))
      
    zeros_s = time.time()
    for index in heap.indicies:
      self.masks[index[0]].flatten()[index[1]] = 0
    zeros_e = time.time()
    print('zeros time: %f'%(zeros_e - zeros_s))
  '''
  '''
  def _update_mask_global(self, rate):
    numel = 0
    for param in self.params:
      numel += param.numel()
    num_prune = math.ceil(numel * rate)
    num_remain = numel - num_prune
    
    for mask in self.masks:
      mask.zero_()
      
    sort_s = time.time()
    argsorts = [param.abs().argsort(descending = True) for param in self.params]
    sort_e = time.time()
    print('sort time: %f'%(sort_e - sort_s))
    max_indices = [0] * len(argsorts)
    
    one_s = time.time()
    cur_remain = 0
    while cur_remain < num_remain:
      max_idx = 0
      max_param = -1
      for i, param in enumerate(self.params):
        if max_indices[i] < param.numel() and abs(param.flatten()[max_indices[i]]) > max_param:
          max_idx = i
          max_param = abs(param.flatten()[max_indices[i]])
      self.masks[max_idx].flatten()[max_indices[max_idx]] = 1
      max_indices[max_idx] += 1
      cur_remain += 1
      
    one_e = time.time()
    print('ones time: %f'%(one_e - one_s))
  '''
  
  def _update_mask_global(self, rate):
    numel = 0
    for param in self.params:
      numel += param.numel()
      
    num_prune = math.ceil(numel * rate)
    num_remain = numel - num_prune
    
    params = torch.cat([param.flatten() for param in self.params])
    
    topk, _ = params.abs().topk(num_remain)
    
    for param, mask in zip(self.params, self.masks):
      mask[:] = param.abs().__gt__(topk[-1])
    
      
  def update_mask(self, rate):
    self._init_masks_and_params()
    if self.mask_policy == 'local':
      self._update_mask_local(rate)
    elif self.mask_policy == 'global':
      self._update_mask_global(rate)
    else:
      raise ValueError('Unrecognized mask update policy: %s, '
                       'can be one of: local, global'%(self.mask_policy))
    '''                   
    numel = 0
    num_prune = 0
    for i, mask in enumerate(self.masks):
      num = mask.numel()
      zeros = num - mask.flatten().sum().item()
      print('parameter %d:\n\tsparsity: %f'%(i, zeros / num))
      numel += num
      num_prune += zeros
    print('total sparsity: %f\n'%(num_prune / numel))
    '''
    
    self.update_params()
  
  def update_params(self):
    self._init_masks_and_params()
    with torch.no_grad():
      for i in range(len(self.params)):
        self.masked_params[i][:] = self.params[i] * self.masks[i]
      
  def attach_gradients(self):
    self._init_masks_and_params()
    for i in range(len(self.params)):
#      if self.params[i].grad is None:
      self.params[i].grad = self.masked_params[i].grad
      if self.mask_grad:
        self.params[i].grad *= self.masks[i]
#      print(self.params[i].grad, self.masked_params[i].grad)
  
  
