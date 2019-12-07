import torch
import torch.nn as nn

class Round(torch.autograd.Function):
  @staticmethod
  def forward(self, x):
    output = x.round()
    return output.float()

  @staticmethod
  def backward(self, g_output):
    return g_output

class ActQuant(nn.Module):
  def __init__(self):
    super(ActQuant, self).__init__()
    self.update_alpha = False
    self.register_buffer('alpha', torch.tensor(1).float())
    self.register_buffer('bitwidth', torch.tensor(1).int())
    self.register_buffer('alpha_mean', torch.tensor(1).float())

    self.update_alpha = False
    self.update_iter = 0
    self.round = Round.apply 

  def __setattr__(self, key, value):
    if key in ('alpha', 'alpha_mean', 'bitwidth'):
      device = self.__getattr__(key).device
    if key == 'alpha':
      if isinstance(value, torch.Tensor):
        del self._buffers['alpha']
        self._buffers['alpha'] = value.to(device)
      elif isinstance(value, int) or isinstance(value, float):
        del self._buffers['alpha']
        self._buffers['alpha'] = torch.tensor(value).float().to(device)
      else:
        raise TypeError
    elif key == 'bitwidth':
      if isinstance(value, torch.Tensor):
        del self._buffers['bitwidth']
        self._buffers['bitwidth'] = value.to(device)
      elif isinstance(value, int):
        del self._buffers['bitwidth']
        self._buffers['bitwidth'] = torch.tensor(value).int().to(device)
      else:
        raise TypeError
    elif key == 'alpha_mean':
      if isinstance(value, torch.Tensor):
        del self._buffers['alpha_mean']
        self._buffers['alpha_mean'] = value.to(device)
      elif isinstance(value, int) or isinstance(value, float):
        del self._buffers['alpha_mean']
        self._buffers['alpha_mean'] = torch.tensor(value).float().to(device)
      else:
        raise TypeError
    else:
      nn.Module.__setattr__(self, key, value)
  
  def forward(self, x):
    if self.update_alpha:
      self.alpha_mean = x.min()

    if self.alpha_mean.item() < 0:
      # int quant
      if self.alpha_mean.item() < -1.0:
        bitwidth = self.bitwidth.item()
      else:
        bitwidth = min(8, self.bitwidth.item() + 1)
      lower_bound = - 2 ** (bitwidth - 1)
      upper_bound = - lower_bound - 1

    else:
      bitwidth = self.bitwidth.item()
      lower_bound = 0
      upper_bound = 2 ** bitwidth - 1

    if self.update_alpha:
      # update alpha
      x_flatten = x.flatten()
      alpha = x.abs().max() / upper_bound
      B = torch.clamp(self.round(x_flatten / alpha), lower_bound, upper_bound)
      best_alpha = alpha
      err = x_flatten - alpha * B
      best_loss = err.dot(err).item() 
      for i in range(self.update_iter):
        alpha = x_flatten.dot(B) / B.dot(B)
        B = torch.clamp(self.round(x_flatten / alpha), lower_bound, upper_bound)
        err = x_flatten - alpha * B
        loss = err.dot(err).item()
        if loss < best_loss:
          best_alpha = alpha
          best_loss = loss

      alpha = best_alpha
      B = torch.clamp(self.round(x_flatten / alpha), lower_bound, upper_bound)
      new_x = B * alpha
      lower_bound_ = new_x.min()
      upper_bound_ = new_x.max()
      x_flatten = x.clamp(lower_bound_, upper_bound_).flatten()
      B = torch.clamp(self.round(x_flatten / alpha), lower_bound, upper_bound)
      best_alpha = alpha
      err = x_flatten - alpha * B
      best_loss = err.dot(err).item()
      for i in range(self.update_iter):
        alpha = x_flatten.dot(B) / B.dot(B)
        B = torch.clamp(self.round(x_flatten / alpha), lower_bound, upper_bound)
        err = x_flatten - alpha * B
        loss = err.dot(err).item()
        if loss < best_loss:
          best_alpha = alpha
          best_loss = loss

      self.alpha = best_alpha
      return x
    else:
      B = torch.clamp(self.round(x / self.alpha), lower_bound, upper_bound) 
      return B * self.alpha

class QuantConv2d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, bias = False):
    super(QuantConv2d, self).__init__()
    self.act = ActQuant()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = bias)

  def forward(self, x):
    x = self.act(x)
    x = self.conv(x)
    return x

def set_quant_bitwidth(model, bitwidth):
  # get the number of act quantization layers in this model
  num_quant = 0
  for m in model.modules():
    if isinstance(m, ActQuant):
      num_quant += 1
  
  if isinstance(bitwidth, int):
    bitwidth = [bitwidth] * num_quant

  if len(bitwidth) != num_quant:
    raise ValueError('The bitwidth must be a single number, or a list of '
                     'integers with length of the number of '
                     'act quantization layers in the model.')

  if isinstance(bitwidth, dict):
    for name, module in model.named_modules():
      if not name in bitwidth:
        raise ValueError('Missed value: %s'%(name))
      elif not isinstance(module, ActQuant):
        raise ValueError('The module named %s is not a ActQuant module.'%(name))
      elif not isinstance(bitwidth[name], int):
        raise TypeError
      else:
        module.bitwidth = bitwidth[name]
    return

  j = 0
  for i, m in enumerate(model.modules()):
    if isinstance(m, ActQuant):
      m.bitwidth = bitwidth[j]
      j += 1


def set_quant_iter(model, iterations):
  # get the number of act quantization layers in this model
  num_quant = 0
  for m in model.modules():
    if isinstance(m, ActQuant):
      num_quant += 1
  
  if isinstance(iterations, int):
    iterations = [iterations] * num_quant

  if len(iterations) != num_quant:
    raise ValueError('The bitwidth must be a single number, or a list of '
                     'integers with length of the number of '
                     'act quantization layers in the model.')

  if isinstance(iterations, dict):
    for name, module in model.named_modules():
      if not name in iterations:
        raise ValueError('Missed value: %s'%(name))
      elif not isinstance(module, ActQuant):
        raise ValueError('The module named %s is not a ActQuant module.'%(name))
      elif not isinstance(iterations[name], int):
        raise TypeError
      else:
        module.iterations = iterations[name]
    return

  j = 0
  for i, m in enumerate(model.modules()):
    if isinstance(m, ActQuant):
      m.iterations = iterations[j]
      j += 1


def set_quant_upalpha(model, upalpha):
  for m in model.modules():
    if isinstance(m, ActQuant):
      m.update_alpha = upalpha

import time

def upalpha_one_batch(model, dataloader, iterations, bitwidth):
  set_quant_bitwidth(model, bitwidth)
  set_quant_iter(model, iterations)
  set_quant_upalpha(model, True)
  model.train()
  print('Updating alpha for act quantization layers...')
  start = time.time()
  for data, _ in dataloader:
    with torch.no_grad():
      model(data)
    break
  end = time.time()

def upalpha_epochs(model, dataloader, epochs, bitwidth):
  set_quant_bitwidth(model, bitwidth)
  set_quant_iter(model, 1)
  set_quant_upalpha(model, True)
  model.train()
  print('Updating alpha for act quantization layers...')
  for epoch in range(epochs):
    print('Epoch %d | %d'%(epoch, epochs))
    start = time.time()
    for data, _ in dataloader:
      with torch.no_grad():
        model(data)

    end = time.time()
    print('Running time: %.2f min'%((end - start) / 60))
   
import math

def upbn(model, dataloader, epochs):
  set_quant_upalpha(model, False) 
  model.train()
  for epoch in range(epochs):
    for data, _ in dataloader:
      with torch.no_grad():
        model(data)

