import torch
from torch import nn

class SparseQuantParam(object):
  def __init__(self, params_q, params, masks, alphas = None):
    self.params_q = params_q
    self._device = self.params_q[0].device
    self.dims = [param.size(0) for param in params_q]
    if params is not None:
      assert(len(params_q) == len(params))
      self.params = [nn.Parameter(param.to(self._device)) for param in params]
    else:
      self.params = [nn.Parameter(param_q.clone()) for param_q in params_q]

    if masks is not None:
      assert(len(masks) == len(params_q))
      self.masks = [mask.to(self._device) for mask in masks]
    else:
      self.masks = [torch.ones_like(param) for param in params_q]

    if alphas is not None:
      assert(len(alphas) == len(params_q))
      self.alphas = [alpha.to(self._device) for alpha in alphas]
    else:
      self.alphas = [torch.empty(param.size(0),1, device = self._device) for param in params_q]

  def update_params(self, bitwidth, update_alpha = True, training = True):
     clip = 2 ** (bitwidth - 1) 
     for param, param_q, mask, dim, alpha in zip(self.params, self.params_q, self.masks, self.dims, self.alphas):
       param_ft = (param * mask).view(dim, -1)
       if not update_alpha:
         a = alpha.to(param_ft.device)
         wb = torch.clamp(torch.round(param_ft / a), -clip, clip - 1)
         wb[a.flatten() == 0, :] = 0
         param_q[:] = (a * wb).reshape(param_q.shape)
         continue
       a = param_ft.abs().max(dim = 1)[0].view(dim, 1) / clip
       wb = torch.clamp(torch.round(param_ft / a), -clip, clip - 1)
       wb[a.flatten() == 0, :] = 0
       for t in range(100):
         a = ((param_ft * wb).sum(dim = 1) / (wb * wb).sum(dim = 1)).view(dim, 1)
         wb = torch.clamp(torch.round(param_ft / a), -clip, clip - 1)
         wb[a.flatten() == 0, :] = 0
#         err = param_ft - a * wb
#         print('Loss:%f'%((err * err).mean().item()))
#       print('\n\n\n')
       alpha[:] = a
       param_q[:] = (a * wb).reshape(param.shape)
       param_q.requires_grad = True

  def attach_gradients(self):
    for param, param_q in zip(self.params, self.params_q):
      param.grad = param_q.grad
