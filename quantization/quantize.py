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
        elif quan_type == 'float32':
            pass
        else:
            assert(False)
        

class WeightQuantizer:
    def __init__(self, model, quan_type, layer_type=(nn.Conv2d, nn.Linear)):
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

    def quantize(self, w, alpha, quan_cfg):
        q = w / alpha.unsqueeze(1)
        if quan_cfg.method == 'uniform':
            q.round_().clamp_(quan_cfg.min_val, quan_cfg.max_val)
        return q

    def quantization(self):
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


    def restore(self):
        for index in range(self.num_quan_layers):
            self.target_params[index].data.copy_(self.saved_tensor_params[index])

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
        
