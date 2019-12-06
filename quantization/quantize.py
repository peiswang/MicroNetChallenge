import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantizationConfig:
    quan_type = None
    signed = None
    method = None
    bit_width = None
    scale = None
    max_val = None
    min_val = None
    def __init__(self, quan_type):
        assert(isinstance(quan_type, str))
        quan_type = quan_type.lower()
        self.quan_type = quan_type
        if quan_type.startswith('int'):
            self.signed = True
            self.method = 'uniform'
            bit_width = quan_type[3:]
            assert(bit_width.isdigit())
            self.bit_width = int(bit_width)
            self.max_val = (1 << (self.bit_width - 1)) - 1
            self.min_val = - self.max_val
        elif quan_type.startswith('uint'):
            self.signed = False
            self.method = 'uniform'
            bit_width = quan_type[4:]
            assert(bit_width.isdigit())
            self.bit_width = int(bit_width)
            self.max_val = (1 << self.bit_width) - 1
            self.min_val = 0
        elif quan_type.startswith('log'):
            self.signed = True
            self.method = 'log'
            bit_width = quan_type[3:]
            assert(bit_width.isdigit())
            self.bit_width = int(bit_width)
            self.max_val = 1 << ((1 << (self.bit_width-1)) - 2)
            self.min_val = - self.max_val
        elif quan_type.startswith('ulog'):
            self.signed = False
            self.method = 'log'
            bit_width = quan_type[4:]
            assert(bit_width.isdigit())
            self.bit_width = int(bit_width)
            self.max_val = 1 << ((1 << self.bit_width) - 2)
            self.min_val = 0
        elif quan_type == 'float32':
            pass
        else:
            assert(False)
        

class WeightQuantizer:
    def __init__(self, model, quan_type, layer_type=(nn.Conv2d, nn.Linear), filterout_fn=None):
        self.num_quan_layers = 0
        self.target_modules = []
        self.target_params= []
        self.saved_tensor_params = []
        self.scales = []
        self.sparsity = []
        self.multiscales = []
        self.quan_cfgs = []
        index = -1
        for m_name, m in model.named_modules():
            if isinstance(m, layer_type):
                index += 1
                if filterout_fn is not None:
                    if filterout_fn(index, m_name, m):
                        continue
                self.target_modules.append(m)
                self.target_params.append(m.weight)
                self.saved_tensor_params.append(m.weight.data.clone())
                self.num_quan_layers += 1
                self.scales.append(None)
                self.sparsity.append(0)

                if hasattr(m, 'groups') and m.groups > 1:
                    self.multiscales.append(True)
                else:
                    self.multiscales.append(False)

                if isinstance(quan_type, str):
                    layer_quan_type = quan_type
                else:
                    layer_quan_type = quan_type[index]
                self.quan_cfgs.append(QuantizationConfig(layer_quan_type))
                m.bit_width = self.quan_cfgs[index].bit_width


    def save_scales(self, f):
        torch.save(self.scales, f)

    def load_scales(self, f):
        self.scales = torch.load(f)

    def set_scales(self, scales):
        self.scales = scales

    def init_quantization(self, init_scales_path=None):
        init_scales = None
        if init_scales_path is not None:
            init_scales = torch.load(init_scales_path)
        for index in range(self.num_quan_layers):
            quan_cfg = self.quan_cfgs[index]
            if quan_cfg.quan_type == 'float32':
                self.scales[index] = 1.0
                print('layer', index, 'not quantized!')
                continue

            if init_scales is None or isinstance(init_scales[index], float) and init_scales[index] == 1.0:
                s = self.target_params[index].data.size()
                if self.multiscales[index]:
                    w = self.target_params[index].data.view(s[0], -1)
                else:
                    w = self.target_params[index].data.view(1, -1)

                alpha = w.abs().max(dim=1)[0] / quan_cfg.max_val
                alpha_old = alpha * 1.1
                count = 0
                while((alpha-alpha_old).norm()>1e-9):
                    q = self.quantize(w, alpha, quan_cfg)
                    alpha_old = alpha
                    alpha = (w*q).sum(dim=1) / (q*q).sum(dim=1)
                    count += 1
                self.scales[index] = alpha
                w.view(s)
                print(count)
            else:
                print('using init scales!')
                self.scales[index] = init_scales[index]

    def quantize(self, w, alpha, quan_cfg):
        q = w / alpha.unsqueeze(1)
        if quan_cfg.method == 'log':
            q_sign = q.sign()
            q_abs = q.abs()
            q_0_idx = q_abs<=0.5
            q_thresh = (q_abs/1.5).log2() + 1
            q_thresh[q_thresh<0] = 0
            q = 2**(q_thresh.floor())
            q = q * q_sign
            q[q_0_idx] = 0
            q.clamp_(quan_cfg.min_val, quan_cfg.max_val)
        elif quan_cfg.method == 'uniform':
            q.round_().clamp_(quan_cfg.min_val, quan_cfg.max_val)
        return q

    def quantization(self):
        self.clampConvParams()
        self.save_params()
        self.quantizeConvParams()

    def real_quantization(self):
        for index in range(self.num_quan_layers):
            alpha = self.scales[index]
            s = self.target_params[index].data.size()
            if self.multiscales[index]:
                w = self.target_params[index].data.view(s[0], -1)
            else:
                w = self.target_params[index].data.view(1, -1)
            q = self.quantize(w, alpha, self.quan_cfgs[index])
            self.target_params[index].data.copy_(q.view(s))

    def clampConvParams(self):
        for index in range(self.num_quan_layers):
            quan_cfg = self.quan_cfgs[index]
            if quan_cfg.quan_type == 'float32':
                continue

            s = self.target_params[index].data.size()
            if self.multiscales[index]:
                w = self.target_params[index].data.view(s[0], -1)
            else:
                w = self.target_params[index].data.view(1, -1)
            alpha = self.scales[index].unsqueeze(1)
            q = w / alpha
            q.clamp_(self.quan_cfgs[index].min_val, self.quan_cfgs[index].max_val)
            self.target_params[index].data.copy_((q*alpha).view(s))

    def save_params(self):
        for index in range(self.num_quan_layers):
            self.saved_tensor_params[index].copy_(self.target_params[index].data)

    def quantizeConvParams(self):
        for index in range(self.num_quan_layers):
            quan_cfg = self.quan_cfgs[index]
            if quan_cfg.quan_type == 'float32':
                continue

            alpha = self.scales[index]
            s = self.target_params[index].data.size()
            if self.multiscales[index]:
                w = self.target_params[index].data.view(s[0], -1)
            else:
                w = self.target_params[index].data.view(1, -1)
            q = self.quantize(w, alpha, self.quan_cfgs[index])
            self.target_params[index].data.copy_((q*alpha.unsqueeze(1)).view(s))

    def clip_grad(self, clip_val=0.01):
        # global_grad = 0
        for index in range(self.num_quan_layers):
            self.target_modules[index].weight.grad.data[self.target_modules[index].weight.grad.data>clip_val] =clip_val
            self.target_modules[index].weight.grad.data[self.target_modules[index].weight.grad.data<-clip_val] = -clip_val
            # max_grad = self.target_modules[index].grad.data.view(-1).abs().max().item()
            # print(max_grad)
            # if global_grad < max_grad:
            #     global_grad = max_grad
        # print('global_max_grad:', global_grad)
        # input('continue')

    def restore(self):
        for index in range(self.num_quan_layers):
            self.target_params[index].data.copy_(self.saved_tensor_params[index])

    def quantization_onelayer(self, index):
        self.saved_tensor_params[index].copy_(self.target_params[index].data)
        alpha = self.scales[index]
        s = self.target_params[index].data.size()
        if self.multiscales[index]:
            w = self.target_params[index].data.view(s[0], -1)
        else:
            w = self.target_params[index].data.view(1, -1)
        q = self.quantize(w, alpha, self.quan_cfgs[index])
        self.target_params[index].data.copy_((q*alpha.unsqueeze(1)).view(s))

    def get_sparsity(self):
        n_elements = 0
        n_nonzeros = 0
        for index in range(self.num_quan_layers):
            s = (self.target_params[index].data.view(-1).abs()>0)
            n_elements += s.numel()
            n_nonzeros += s.sum().item()
            s = s.sum().item()/s.numel()
            # print(s)
            self.sparsity[index] = s
        return 1.0 * (n_elements - n_nonzeros) / n_elements
        

class Quantization(nn.Module):
    def __init__(self):
        super(Quantization, self).__init__()
        self.bit_width = None
        self.scale = None
        self.out_scale = None
        self.enabled = False
        self.signed = None

    def forward(self, input):
        y = input
        if self.enabled:
            y = torch.clamp(torch.round(input/self.scale), self.min_val, self.max_val) * self.out_scale
        return y

    def enable_quantization(self):
        self.enabled = True

    def half(self):
        self.scale = torch.Tensor([self.scale]).cuda().half() 
        # print(self.scale.type())
        self.out_scale = torch.Tensor([self.out_scale]).cuda().half() 

    def set_bitwidth(self, bit_width):
        self.bit_width = bit_width

    def set_scale(self, scale):
        self.scale = scale
        self.out_scale = scale

    def set_in_scale(self, scale):
        self.scale = scale

    def set_out_scale(self, scale):
        self.out_scale = scale

    def set_sign(self, signed):
        self.signed = signed
        if self.signed:
            self.max_val = (1 << (self.bit_width - 1)) - 1
            self.min_val = - self.max_val
        else:
            self.max_val = (1 << self.bit_width) - 1
            self.min_val = 0

    def set_quantization_parameters(self, signed=None, bit_width=None, scale=None):
        assert(bit_width is not None and scale is not None)
        self.set_bitwidth(bit_width)
        self.set_sign(signed)
        self.set_scale(scale)
        
