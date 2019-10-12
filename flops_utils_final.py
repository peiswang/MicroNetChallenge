import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import math
from models.utils import *
from quantization.quantize import Quantization

def get_density(w):
    n_nonzeros = (w.abs()>0).sum().item()
    return n_nonzeros * 1.0 / w.numel()

def get_nonzero_number(w):
    n_nonzeros = (w.abs()>0).sum().item()
    return n_nonzeros

def adder_tree(base, length):
    count = 0
    base_i = base
    while(length>1):
        count_i = length // 2
        count += count_i * base_i
        length = math.ceil(length/2)
        base_i +=1
    return count, base_i

def count_conv_linear_adder_tree(module, input, output):
    input = input[0]
    output = output[0]
    # print(output.dim())
    if output.dim() == 3:
        C, H, W = output.shape
    else:
        C = output.shape[0]
        H = W = 1

    ####### Params
    weight = module.weight.data
    w = weight.view(-1)
    w_element = w.numel()
    w_nonzero = (w.abs()>0).sum().item()
    # scale terms for scale and bias in Conv2D_INT/FC_INT (FP16)
    scale_param = (2 * C + 1) / 2.
    module.total_params = (w_element + w_nonzero * module.bit_width) / 32. + scale_param

    ####### FLOPs
    # multiplication
    multiply_bit_width = max(module.bit_width, module.act_bit_width)
    k_element = w_element / C
    density = get_density(w)
    multiply_ops = density * H * W * C * k_element * multiply_bit_width / 32.

    # addition
    base_bit_width = module.bit_width + module.act_bit_width
    if module.input_signed:
        base_bit_width -= 1
    addition_ops = 0
    for kernel_idx in range(C):
        w_k = weight[kernel_idx].view(-1)
        n_el = get_nonzero_number(w_k)
        addition_bits, max_bit = adder_tree(base_bit_width, n_el)
        addition_ops += addition_bits * H * W
    # print(max_bit)
    addition_ops /= 32.

    # scale and bias in FP16 format
    scale_bias_ops = C * H * W * 2 * 16. / 32.

    module.total_ops += multiply_ops + addition_ops + scale_bias_ops

def count_conv_linear_adder_int16(module, input, output):
    input = input[0]
    output = output[0]
    # print(output.dim())
    if output.dim() == 3:
        C, H, W = output.shape
    else:
        C = output.shape[0]
        H = W = 1

    ####### Params
    weight = module.weight.data
    w = weight.view(-1)
    w_element = w.numel()
    w_nonzero = (w.abs()>0).sum().item()
    # scale terms for scale and bias in Conv2D_INT/FC_INT (FP16)
    scale_param = (2 * C + 1) / 2.
    module.total_params = (w_element + w_nonzero * module.bit_width) / 32. + scale_param

    ####### FLOPs
    # multiplication
    multiply_bit_width = max(module.bit_width, module.act_bit_width)
    k_element = w_element / C
    density = get_density(w)
    multiply_ops = density * H * W * C * k_element * multiply_bit_width / 32.

    # addition
    base_bit_width = module.bit_width + module.act_bit_width
    if module.input_signed:
        base_bit_width -= 1
    addition_ops = 0

    max_n_el = 0
    for kernel_idx in range(C):
        w_k = weight[kernel_idx].view(-1)
        n_el = get_nonzero_number(w_k)
        if n_el > max_n_el:
            max_n_el = n_el
    _, max_bit = adder_tree(base_bit_width, max_n_el)
    max_bit = min(max_bit, 16)

    for kernel_idx in range(C):
        w_k = weight[kernel_idx].view(-1)
        n_el = get_nonzero_number(w_k)
        addition_ops += max_bit * (n_el - 1) * H * W

    addition_ops /= 32.

    # scale and bias in FP16 format
    scale_bias_ops = C * H * W * 2 * 16. / 32.

    module.total_ops += multiply_ops + addition_ops + scale_bias_ops

def count_conv_linear_adder_int(module, input, output):
    input = input[0]
    output = output[0]
    # print(output.dim())
    if output.dim() == 3:
        C, H, W = output.shape
    else:
        C = output.shape[0]
        H = W = 1

    ####### Params
    weight = module.weight.data
    w = weight.view(-1)
    w_element = w.numel()
    w_nonzero = (w.abs()>0).sum().item()
    # scale terms for scale and bias in Conv2D_INT/FC_INT (FP16)
    scale_param = (2 * C + 1) / 2.
    module.total_params = (w_element + w_nonzero * module.bit_width) / 32. + scale_param

    ####### FLOPs
    # multiplication
    multiply_bit_width = max(module.bit_width, module.act_bit_width)
    k_element = w_element / C
    density = get_density(w)
    multiply_ops = density * H * W * C * k_element * multiply_bit_width / 32.

    # addition
    base_bit_width = module.bit_width + module.act_bit_width
    if module.input_signed:
        base_bit_width -= 1
    addition_ops = 0

    max_n_el = 0
    for kernel_idx in range(C):
        w_k = weight[kernel_idx].view(-1)
        n_el = get_nonzero_number(w_k)
        if n_el > max_n_el:
            max_n_el = n_el
    _, max_bit = adder_tree(base_bit_width, max_n_el)

    for kernel_idx in range(C):
        w_k = weight[kernel_idx].view(-1)
        n_el = get_nonzero_number(w_k)
        addition_ops += max_bit * (n_el - 1) * H * W

    addition_ops /= 32.

    # scale and bias in FP16 format
    scale_bias_ops = C * H * W * 2 * 16. / 32.

    module.total_ops += multiply_ops + addition_ops + scale_bias_ops

def count_conv_linear_adder_fp16(module, input, output):
    input = input[0]
    output = output[0]
    # print(output.dim())
    if output.dim() == 3:
        C, H, W = output.shape
    else:
        C = output.shape[0]
        H = W = 1

    ####### Params
    weight = module.weight.data
    w = weight.view(-1)
    w_element = w.numel()
    w_nonzero = (w.abs()>0).sum().item()
    # scale terms for scale and bias in Conv2D_INT/FC_INT (FP16)
    scale_param = (2 * C + 1) / 2.
    module.total_params = (w_element + w_nonzero * module.bit_width) / 32. + scale_param

    ####### FLOPs
    # multiplication
    multiply_bit_width = max(module.bit_width, module.act_bit_width)
    k_element = w_element / C
    density = get_density(w)
    multiply_ops = density * H * W * C * k_element * multiply_bit_width / 32.
    addition_ops = density * H * W * C * (k_element - 1) * 16. / 32.

    # scale and bias in FP16 format
    scale_bias_ops = C * H * W * 2 * 16. / 32.

    module.total_ops += multiply_ops + addition_ops + scale_bias_ops

def count_conv_linear_adder_fp32(module, input, output):
    input = input[0]
    output = output[0]
    # print(output.dim())
    if output.dim() == 3:
        C, H, W = output.shape
    else:
        C = output.shape[0]
        H = W = 1

    ####### Params
    weight = module.weight.data
    w = weight.view(-1)
    w_element = w.numel()
    w_nonzero = (w.abs()>0).sum().item()
    # scale terms for scale and bias in Conv2D_INT/FC_INT (FP16)
    scale_param = (2 * C + 1) / 2.
    module.total_params = (w_element + w_nonzero * module.bit_width) / 32. + scale_param

    ####### FLOPs
    # multiplication
    multiply_bit_width = max(module.bit_width, module.act_bit_width)
    k_element = w_element / C
    density = get_density(w)
    multiply_ops = density * H * W * C * k_element * multiply_bit_width / 32.
    addition_ops = density * H * W * C * (k_element - 1) * 32. / 32.

    # scale and bias in FP16 format
    scale_bias_ops = C * H * W * 2 * 16. / 32.

    module.total_ops += multiply_ops + addition_ops + scale_bias_ops

def count_quantization(module, input, output):
    input = input[0]
    num_elements = input.numel()
    alpha = module.scale # min
    if alpha == 1.0:
        # print(alpha)
        # omit .div(alpha), FP32 to UINT
        quan_op = 0.0 # num_elements * 3.0 / 2.0
        param = 0.0 # FP16
    else:
        # FP32 to INT, only count .div(alpha)
        quan_op = num_elements * 1.0 / 2.0 # num_elements * 4.0 / 2.0
        param = 1.0 / 2.0 # FP16

    module.total_params = param
    module.total_ops += quan_op

def count_relu(module, input, output):
    input = input[0]
    output = output[0]

    num_elements = output.numel()
    total_ops = 1 * num_elements / 2.

    module.total_ops += total_ops
    module.total_params = 0.

def count_avgpool2d(module, input, output):
    #count flops for single input
    input = input[0]
    output = output[0]

    add_per_output = torch.prod(torch.Tensor([module.kernel_size])) - 1
    mul_per_out = 1
    MAC_per_out = add_per_output + mul_per_out
    num_elements = output.numel()
    total_ops = MAC_per_out * num_elements / 2.

    module.total_ops += total_ops
    module.total_params = 0


def count_adap_avgpool2d(module, input, output):
    #count flops for single input
    input = input[0]
    output = output[0]

    input_h = input.size(1)
    input_w = input.size(2)
    if isinstance(module.output_size, tuple):
        output_h = module.output_size[0]
        output_w = module.output_size[1]
    else:
        output_h = output_w = module.output_size
    kh = (input_h + output_h - 1) // output_h
    kw = (input_w + output_w - 1) // output_w

    add_per_output = kh * kw - 1
    mul_per_out = 1
    MAC_per_out = add_per_output + mul_per_out

    num_elements = output.numel()
    total_ops = MAC_per_out * num_elements / 2.

    module.total_ops += total_ops
    module.total_params = 0


def count_sigmoid(module, input, output):
    #count flops for single input
    input = input[0]
    output = output[0]

    num_elements = output.numel()
    total_ops = 3 * num_elements / 2.0
    module.total_ops += total_ops
    module.total_params = 0

def count_swish(module, input, output):
    #count flops for single input
    input = input[0]
    output = output[0]

    num_elements = output.numel()
    total_ops = 4 * num_elements / 2.0
    module.total_ops += total_ops
    module.total_params = 0

def count_pointproduct(module, input, output):
    #count flops for single input
    x1 = input[0][0]
    x2 = input[1][0]
    output = output[0]

    total_ops = max(x1.numel(), x2.numel()) / 2.0
    module.total_ops += total_ops
    module.total_params = 0
    

def count_pointadd(module, input, output):
    #count flops for single input
    x1 = input[0][0]
    x2 = input[1][0]
    output = output[0]
    
    total_ops = max(x1.numel(), x2.numel()) / 2.0
    module.total_ops += total_ops
    module.total_params = 0

__hook_fn_dict_adder_fp32__ = {
    nn.Conv2d: count_conv_linear_adder_fp32,    
    nn.ReLU: count_relu,
    nn.Sigmoid: count_sigmoid,
    Swish: count_swish,
    PointProduct: count_pointproduct,
    PointAdd: count_pointadd,    
    nn.AvgPool2d: count_avgpool2d,  
    nn.AdaptiveAvgPool2d: count_adap_avgpool2d,
    nn.Linear: count_conv_linear_adder_fp32,
    Quantization: count_quantization,
}

__hook_fn_dict_adder_fp16__ = {
    nn.Conv2d: count_conv_linear_adder_fp16,    
    nn.ReLU: count_relu,
    nn.Sigmoid: count_sigmoid,
    Swish: count_swish,
    PointProduct: count_pointproduct,
    PointAdd: count_pointadd,    
    nn.AvgPool2d: count_avgpool2d,  
    nn.AdaptiveAvgPool2d: count_adap_avgpool2d,
    nn.Linear: count_conv_linear_adder_fp16,
    Quantization: count_quantization,
}

__hook_fn_dict_adder_int__ = {
    nn.Conv2d: count_conv_linear_adder_int,    
    nn.ReLU: count_relu,
    nn.Sigmoid: count_sigmoid,
    Swish: count_swish,
    PointProduct: count_pointproduct,
    PointAdd: count_pointadd,    
    nn.AvgPool2d: count_avgpool2d,  
    nn.AdaptiveAvgPool2d: count_adap_avgpool2d,
    nn.Linear: count_conv_linear_adder_int,
    Quantization: count_quantization,
}

__hook_fn_dict_adder_int16__ = {
    nn.Conv2d: count_conv_linear_adder_int16,    
    nn.ReLU: count_relu,
    nn.Sigmoid: count_sigmoid,
    Swish: count_swish,
    PointProduct: count_pointproduct,
    PointAdd: count_pointadd,    
    nn.AvgPool2d: count_avgpool2d,  
    nn.AdaptiveAvgPool2d: count_adap_avgpool2d,
    nn.Linear: count_conv_linear_adder_int16,
    Quantization: count_quantization,
}

__hook_fn_dict_adder_tree__ = {
    nn.Conv2d: count_conv_linear_adder_tree,    
    nn.ReLU: count_relu,
    nn.Sigmoid: count_sigmoid,
    Swish: count_swish,
    PointProduct: count_pointproduct,
    PointAdd: count_pointadd,    
    nn.AvgPool2d: count_avgpool2d,  
    nn.AdaptiveAvgPool2d: count_adap_avgpool2d,
    nn.Linear: count_conv_linear_adder_tree,
    Quantization: count_quantization,
}
