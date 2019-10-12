import os
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

from quantization.models.densenet_final import *

def get_density(w):
    n_nonzeros = (w.abs()>0).sum().item()
    return n_nonzeros * 1.0 / w.numel()

def get_nonzero_number(w):
    n_nonzeros = (w.abs()>0).sum().item()
    return n_nonzeros

def addition_tree(base, length):
    count = 0
    base_i = base
    while(length>1):
        count_i = length // 2
        count += count_i * base_i
        length = math.ceil(length/2)
        base_i +=1
    return count

def count_conv_linear_adder_tree(module, input, output):
    # using adder tree 
    input = input[0]
    output = output[0]
    if output.dim() == 3:
        C, H, W = output.shape
    else:
        C = output.shape[0]
        H = W = 1

    ####### params     
    weight = module.weight.data
    w = weight.view(-1)
    w_element = w.numel()
    w_nonzero = (w.abs()>0).sum().item()
    module.total_params += (w_element + w_nonzero * module.bit_width) / 32.

    # multiplication
    multiply_bit_width = max(module.bit_width, module.act_bit_width)
    k_element = w_element / C
    density = get_density(w)
    multiply_ops = density * H * W * C * k_element * multiply_bit_width / 32.
    # addition
    if module.act_bit_width == 32:
        addition_ops = density * H * W * C * (k_element - 1) * 32. / 32.
    else:
        base_bit_width = module.bit_width + module.act_bit_width
        if module.input_signed:
            base_bit_width -= 1
        addition_ops = 0
        for kernel_idx in range(C):
            w_k = weight[kernel_idx].view(-1)
            n_el = get_nonzero_number(w_k)
            addition_ops += addition_tree(base_bit_width, n_el) * H * W
        addition_ops /= 32.

    module.total_ops += multiply_ops + addition_ops

def count_conv_linear_fixed_bitwidth(module, input, output):
    # using max bitwidth to accumulate 
    input = input[0]
    output = output[0]
    if output.dim() == 3:
        C, H, W = output.shape
    else:
        C = output.shape[0]
        H = W = 1

    ####### params     
    weight = module.weight.data
    w = weight.view(-1)
    w_element = w.numel()
    w_nonzero = (w.abs()>0).sum().item()
    module.total_params += (w_element + w_nonzero * module.bit_width) / 32.

    # multiplication
    multiply_bit_width = max(module.bit_width, module.act_bit_width)
    k_element = w_element / C
    density = get_density(w)
    multiply_ops = density * H * W * C * k_element * multiply_bit_width / 32.
    # addition
    addition_ops = 0
    if module.act_bit_width == 32:
        addition_ops = density * H * W * C * (k_element - 1) * 32. / 32.
    else:
        act_uint_max = 2**module.act_bit_width - 1
        weights_int_max = 2**(module.act_bit_width-1) - 1
        max_add_val = act_uint_max*weights_int_max*k_element
        max_bitwidth = math.floor(math.log(max_add_val, 2)) + 1
        for kernel_idx in range(C):
            w_k = weight[kernel_idx].view(-1)
            n_el = get_nonzero_number(w_k)
            addition_ops += max_bitwidth * (n_el - 1) * H * W
    addition_ops /= 32.
    module.total_ops += multiply_ops + addition_ops

def count_conv_linear_FP32(module, input, output):
    # using FP32 to accumulate
    input = input[0]
    output = output[0]
    if output.dim() == 3:
        C, H, W = output.shape
    else:
        C = output.shape[0]
        H = W = 1

    ####### params     
    weight = module.weight.data
    w = weight.view(-1)
    w_element = w.numel()
    w_nonzero = (w.abs()>0).sum().item()
    module.total_params += (w_element + w_nonzero * module.bit_width) / 32.

    # multiplication
    multiply_bit_width = max(module.bit_width, module.act_bit_width)
    k_element = w_element / C
    density = get_density(w)
    multiply_ops = density * H * W * C * k_element * multiply_bit_width / 32.
    # addition
    addition_ops = density * H * W * C * (k_element - 1) * 32. / 32.
    module.total_ops += multiply_ops + addition_ops

def count_act_quant(module, input, output):
    module.total_params += 0.0
    module.total_ops += 0.0

def count_bn2d(module, input, output):
    # calculated in the previous conv layer
    module.total_params += 0
    module.total_ops += 0

def count_relu(module, input, output):
    #count flops for single input
    input = input[0]
    output = output[0]

    num_elements = output.numel()

    total_ops = 1 * num_elements / 2.
    module.total_ops += total_ops
    module.total_params += 0.


def count_avgpool2d_FP16(module, input, output):
    # count flops for single input
    # using FP16 to accumulate 
    input = input[0]
    output = output[0]

    add_per_output = torch.prod(torch.Tensor([module.kernel_size])) - 1
    mul_per_out = 1.
    MAC_per_out = add_per_output.item() + mul_per_out
    num_elements = output.numel()
    total_ops = MAC_per_out * num_elements / 2.

    module.total_ops += total_ops
    module.total_params += 0.

def count_avgpool2d_FP32(module, input, output):
    # count flops for single input
    # using FP32 to accumulate 
    input = input[0]
    output = output[0]

    add_per_output = torch.prod(torch.Tensor([module.kernel_size])) - 1
    mul_per_out = 1.
    num_elements = output.numel()
    total_ops = add_per_output.item() * num_elements + num_elements * mul_per_out / 2.

    module.total_ops += total_ops
    module.total_params += 0.


def count_scale_FP16(module, input, output):
    # using FP16 to accumulate 
    input = input[0]
    num_elements = input.numel()
    C = input.size(0)

    if module.bias[0].item() == 0:
        scale_op = num_elements * 1.0 / 2.0
        param = C * 1.0 / 2.0 # FP16
    else:
        scale_op = num_elements * 2.0 / 2.0
        param = C * 2.0 / 2.0 # FP16

    module.total_params += param
    module.total_ops += scale_op

def count_scale_FP32(module, input, output):
    # using FP32 to accumulate 
    input = input[0]
    num_elements = input.numel()
    C = input.size(0)

    if module.bias[0].item() == 0:
        scale_op = num_elements * 1.0 / 2.0
        param = C * 1.0 / 2.0 # FP16
    else:
        scale_op = num_elements / 2.0 + num_elements # FP16 scale + FP32 add bias
        param = C * 2.0 / 2.0 # FP16

    module.total_params += param
    module.total_ops += scale_op

'''
Using adder tree in convolution accumulation
Using FP16 in all the other operations
'''

__hook_fn_dict0__ = {
    nn.Conv2d: count_conv_linear_adder_tree,    
    nn.ReLU: count_relu,
    Quant: count_act_quant,
    Scale: count_scale_FP16,
    nn.AvgPool2d: count_avgpool2d_FP16,
    nn.Linear: count_conv_linear_FP32,
}

'''
Using fixed bitwidth (without overflow) to accumulate convolution intermediate results
Using FP16 in all the other operations
'''

__hook_fn_dict1__ = {
    nn.Conv2d: count_conv_linear_fixed_bitwidth,    
    nn.ReLU: count_relu,
    Quant: count_act_quant,
    Scale: count_scale_FP16,  
    nn.AvgPool2d: count_avgpool2d_FP16,  
    nn.Linear: count_conv_linear_FP32,
}

'''
Using FP32 to accumulate intermediate results in convolution, average pooling and bias term
Using FP16 in all the other operations
'''

__hook_fn_dict2__ = {
    nn.Conv2d: count_conv_linear_FP32,    
    nn.ReLU: count_relu,
    Quant: count_act_quant,
    Scale: count_scale_FP32,
    nn.AvgPool2d: count_avgpool2d_FP32,
    nn.Linear: count_conv_linear_FP32,
}

'''
Using fixed bitwidth (without overflow) to accumulate convolution intermediate results
Using FP32 to accumulate intermediate results in average pooling and bias term
Using FP16 in all the other operations
'''

__hook_fn_dict3__ = {
    nn.Conv2d: count_conv_linear_fixed_bitwidth,    
    nn.ReLU: count_relu,
    Quant: count_act_quant,
    Scale: count_scale_FP32,
    nn.AvgPool2d: count_avgpool2d_FP32,
    nn.Linear: count_conv_linear_FP32,
}

'''
Using adder tree in convolution accumulation
Using FP32 to accumulate intermediate results in average pooling and bias term
Using FP16 in all the other operations
'''

__hook_fn_dict4__ = {
    nn.Conv2d: count_conv_linear_adder_tree,    
    nn.ReLU: count_relu,
    Quant: count_act_quant,
    Scale: count_scale_FP32,
    nn.AvgPool2d: count_avgpool2d_FP32,
    nn.Linear: count_conv_linear_FP32,
}

'''
Using adder tree in convolution accumulation
Using FP32 to accumulate intermediate results in average pooling
Using FP16 in all the other operations
'''

__hook_fn_dict5__ = {
    nn.Conv2d: count_conv_linear_adder_tree,    
    nn.ReLU: count_relu,
    Quant: count_act_quant,
    Scale: count_scale_FP16,
    nn.AvgPool2d: count_avgpool2d_FP32,
    nn.Linear: count_conv_linear_FP32,
}

'''
Using fixed bitwidth (without overflow) to accumulate convolution intermediate results
Using FP32 to accumulate intermediate results in average pooling
Using FP16 in all the other operations
'''

__hook_fn_dict6__ = {
    nn.Conv2d: count_conv_linear_fixed_bitwidth,    
    nn.ReLU: count_relu,
    Quant: count_act_quant,
    Scale: count_scale_FP16,
    nn.AvgPool2d: count_avgpool2d_FP32,
    nn.Linear: count_conv_linear_FP32,
}