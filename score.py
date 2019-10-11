import argparse
import os
import random
import shutil
import time
import warnings
import sys
from datetime import datetime
import pickle
import math
import numpy as np

import torch
import torch.nn as nn

import flops_utils_final as flops_utils

from quantization.quantize import WeightQuantizer, Quantization

#import torchvision.models as models
import models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='mixnet_s_final',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--pretrained', default='./pretrained/mixnet_s_prune_quan_final.pth', type=str, metavar='PATH', 
                    help='path to pre-trained model')
parser.add_argument('--scales', default='./pretrained/scales.npy', type=str, metavar='PATH',
                    help='path to the pre-calculated scales (.npy file)')
parser.add_argument('--adder', default='tree', type=str, metavar='TYPE',
                    help='adder type of convolutions (tree | int | int16 | fp16 | fp32)')

###
# (1) tree: adder-tree, use minimal bit-width as needed.
# (2) int: using integers for accumulation. Bit-width is selected to make assure no overflow.
# (3) int16: based on (2), if bit-width > 16, then FP16 for accumulation.
# (4) FP16: FP16 for accumulation.
# (5) FP32: FP32 for accumulation.


signed = [None, False, False, True, True, False, False, False, 
          True, True, False, False, False, True, True, True, True, 
          True, True, True, True, True, True, True, True, True, True, 
          True, True, True, True, True, True, True, True, True, True, 
          True, True, True, True, True, True, True, True, True, True, 
          True, True, True, True, True, True, True, True, True, True, 
          True, True, True, True, True, True, True, True, True, True, 
          True, True, True, True, True, True, True, True, True, True, 
          True, True, True, True, True, True, True, True, True, True, 
          True, True, True, True, True, True, True, True, True, True, 
          True, True, True, True, True, True, True, True, True, True, 
          True, True, True, True, True, True, True, True, True, True, 
          True, True, True, True, True, True, False]

weight_bitwidth = [7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 5, 5, 5, 5, 5,
                   5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 5, 5, 5, 5, 5, 
                   7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 
                   5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 
                   5, 4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 4, 4, 5, 5, 5, 5, 3, 3, 4, 4, 3, 3]

hook_dict = None

def main():
    args = parser.parse_args()

    ## student model
    model = models.__dict__[args.arch](pretrained=False)

    # Data loading code
    input_size = 224

    if args.pretrained:
        state_dict = torch.load(args.pretrained, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict, strict=False)
    else:
        assert(False)

    print("adding 7-bit activation quantization op (first layer 8-bit)")
    scales = np.load(args.scales)
    scales[0] = 1.0
    idx = 0
    for m in model.modules():
        if isinstance(m, Quantization):
            m.set_quantization_parameters(signed[idx], 7, scales[idx])
            m.set_out_scale(1.0)
            idx += 1

    print("adding weight quantization op")
    weight_quan_type = ['int5'] * len(signed)
    for idx in range(1, 8):
        weight_quan_type[idx] = 'int7'
    weight_quan_type[0] = 'int7'
    weight_quan_type[19] = 'int7'
    weight_quan_type[44] = 'int7'
    weight_quan_type[50] = 'int7'
    for idx in range(-6, 0):
        weight_quan_type[idx] = 'int4'
    for idx in range(-15, -10):
        weight_quan_type[idx] = 'int4'
    for idx in range(-23, -19):
        weight_quan_type[idx] = 'int4'
    weight_quan_type[-29] = 'int4'

    weight_quan_type[-1] = 'int3'
    weight_quan_type[-2] = 'int3'
    weight_quan_type[-5] = 'int3'
    weight_quan_type[-6] = 'int3'

    weight_quantizer = WeightQuantizer(model, weight_quan_type)

    # set activation quantization bits
    idx = 0
    for m_name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if signed[idx] is None:
                m.input_signed = False
                m.act_bit_width = 8
            elif signed[idx]:
                m.input_signed = True
                m.act_bit_width = 7
            else:
                m.input_signed = False
                m.act_bit_width = 7
            idx += 1

    # print('Param sparsity:', weight_quantizer.get_sparsity())
    print('Calculating score using', args.adder)
    global hook_dict
    if args.adder == 'tree':
        hook_dict = flops_utils.__hook_fn_dict_adder_tree__
    elif args.adder == 'int':
        hook_dict = flops_utils.__hook_fn_dict_adder_int__
    elif args.adder == 'int16':
        hook_dict = flops_utils.__hook_fn_dict_adder_int16__
    elif args.adder == 'fp16':
        hook_dict = flops_utils.__hook_fn_dict_adder_fp16__
    elif args.adder == 'fp32':
        hook_dict = flops_utils.__hook_fn_dict_adder_fp32__
    else:
        print(args.adder)
        assert(False)

    total_ops, total_params = model_count(model=model)
    print('#OPs:', total_ops)
    print('#Params:', total_params)
    op_score = total_ops / 1000000 / 1170.0
    param_score = total_params / 1000000 / 6.9
    print('OP score:', op_score)
    print('Param score:', param_score)
    print('Final score:', op_score+param_score)

def add_hook(model):
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue
        module.name = name
        # module.register_buffer('total_ops', torch.zeros(1))
        module.total_ops = 0
        module.total_params = 0
        module_type = type(module)
        fn = None
        if module_type in hook_dict:
            fn = hook_dict[module_type]
        else:
            print("WARNING: We have not implemented counting method for ",
                module_type, ", which will be ignored")

        if fn is not None:
            handler = module.register_forward_hook(fn)

def model_count(model, input_channel=3, input_size=224):

    model.eval()
    add_hook(model=model)

    input = torch.randn(1, input_channel, input_size, input_size)
    with torch.no_grad():
        model(input)

    total_ops = 0
    total_params = 0
    for module in model.modules():
        # skip for non-leaf module
        if len(list(module.children())) > 0:  
            continue
        total_ops += module.total_ops
        total_params += module.total_params

    return total_ops, total_params

if __name__ == '__main__':
    main()
