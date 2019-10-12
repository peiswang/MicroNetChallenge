import argparse
import os
import random
import shutil
import time
import warnings
import sys
from datetime import datetime
from collections import OrderedDict
import pickle
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from collections import OrderedDict

from PIL import Image
import quantization
from quantization import *
import flops_utils

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--score-type', default = 0, type = int,
                    help = 'Please refer to GitHub')

def main():
    args = parser.parse_args()

    ## student model
    new_model = quantization.densenet_final.densenet_final(depth = 100, num_classes = 100, growthRate = 12, compressionRate = 2)

    num_param = 0
    for m_name, m in new_model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            num_param += 1

    act_bitwidth = [4]*num_param
    act_bitwidth[0] = 32
    act_bitwidth[-1] = 32

    weights_bitwidth = [4]*num_param
    weights_bitwidth[-1] = 6

    # Data loading code
    if args.resume:
        state_dict = torch.load(args.resume)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'scale' in k:
                v = v.flatten()
            new_state_dict[k] = v 
        new_model.load_state_dict(new_state_dict, strict=True)

    idx = 0
    for m_name, m in new_model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.act_bit_width = act_bitwidth[idx]
            m.bit_width = weights_bitwidth[idx]
            m.input_signed = False
            idx += 1

    total_ops, total_params = model_count(model=new_model, C=3, H=32, W=32, score_type=args.score_type)
    print('#OPs:', total_ops)
    print('#Params:', total_params)
    op_score = total_ops / 1000000000 / 10.49
    param_score = total_params / 1000000 / 36.5
    print('OP score:', op_score)
    print('Param score:', param_score)
    print('Final score:', op_score+param_score)

def add_hook(model, score_type=0):
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue
        module.name = name
        # module.register_buffer('total_ops', torch.zeros(1))
        module.total_ops = 0
        module.total_params = 0
        module_type = type(module)
        fn = None
        if module_type in flops_utils.__hook_fn_dict0__:
            if score_type == 0:
                fn = flops_utils.__hook_fn_dict0__[module_type]
            elif score_type == 1:
                fn = flops_utils.__hook_fn_dict1__[module_type]
            elif score_type == 2:
                fn = flops_utils.__hook_fn_dict2__[module_type]
            elif score_type == 3:
                fn = flops_utils.__hook_fn_dict3__[module_type]
            elif score_type == 4:
                fn = flops_utils.__hook_fn_dict4__[module_type]
            elif score_type == 5:
                fn = flops_utils.__hook_fn_dict5__[module_type]
            elif score_type == 6:
                fn = flops_utils.__hook_fn_dict6__[module_type]
            else:
                assert(score_type<7)
        else:
            print("WARNING: We have not implemented counting method for ",
                module_type, ", which will be ignored")

        if fn is not None:
            handler = module.register_forward_hook(fn)

def model_count(model, C=3, H=32, W=32, score_type=0):

    model.eval()
    add_hook(model=model, score_type=score_type)

    for module in model.modules():
        module.total_ops = 0.0
        module.total_params = 0.0 

    input = torch.randn(1, C, H, W)
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
