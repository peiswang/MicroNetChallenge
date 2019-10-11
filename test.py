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

from quantization.quantize import WeightQuantizer

from PIL import Image

import models as models
from quantization.quantize import Quantization

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d','--data', metavar='DIR', default='./data',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='mixnet_s_final',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', default='./pretrained/mixnet_s_prune_quan_final.pth', type=str, metavar='PATH', 
                    help='path to pre-trained model')
parser.add_argument('--act-bit-width', default=7, type=int,
                    help='activation quantization bit-width')
parser.add_argument('--scales', default='./pretrained/scales.npy', type=str, metavar='PATH',
                    help='path to the pre-calculated scales (.npy file)')

quan_modules = []

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

def main():
    args = parser.parse_args()

    cudnn.benchmark = True

    ## 
    model = models.__dict__[args.arch](pretrained=False)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    input_size = 224

    print("adding quan op ('{}bit')...".format(args.act_bit_width))
    scales = np.load(args.scales)
    idx = 0
    for m in model.modules():
        if isinstance(m, Quantization):
            m.set_quantization_parameters(signed[idx], args.act_bit_width, scales[idx])
            m.set_out_scale(1.0)
            quan_modules.append(m)
            idx += 1

    # Data loading code
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0., 0., 0.],
                                     std=[1./255., 1./255., 1./255.])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            #transforms.Resize(256),
            transforms.Resize(int(input_size/0.875), interpolation=Image.BICUBIC), # == 256
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.pretrained:
        state_dict = torch.load(args.pretrained, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict, strict=False)
    else:
        assert(False)

    # enable feature map quantization
    for index, q_module in enumerate(quan_modules):
        if q_module.signed is not None:
            q_module.enable_quantization()
            q_module.half()

    validate(val_loader, model, criterion, args)

def validate(val_loader, model, criterion, args, weight_quantizer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    model.half()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = torch.round(input).float()
            input = input.cuda().half()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.print_freq is not None and i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    print(sys.argv)
    main()
