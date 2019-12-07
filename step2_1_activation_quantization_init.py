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

from extensions import refinery_loss
from sparse_util import PruneOp

from PIL import Image

#import torchvision.models as models
import models as models
from quantization.quantize import Quantization

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d','--data', metavar='DIR', default='./data',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='mixnet_s_quan',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_policy', default='cosine',
                    help='lr policy')
parser.add_argument('--warmup-epochs', default=0, type=int, metavar='N',
                    help='number of warmup epochs')
parser.add_argument('--warmup-lr-multiplier', default=0.1, type=float, metavar='W',
                    help='warmup lr multiplier')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1-4)',
                    dest='weight_decay')
parser.add_argument('--dropout', default=0.0, type=float,
                    help='dropout ratio (default: 0.2)')
parser.add_argument('--dropconnect', default=0.0, type=float,
                    help='dropconnect ratio (default: 0.2)')
parser.add_argument('--power', default=1.0, type=float,
                    metavar='P', help='power for poly learning-rate decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH', 
                    help='path to pre-trained model')
parser.add_argument('--act-bit-width', default=8, type=int,
                    help='activation quantization bit-width')
parser.add_argument('--scales', default='', type=str, metavar='PATH',
                    help='path to the pre-calculated scales (.npy file)')
parser.add_argument('--masks', default='', type=str, metavar='PATH', 
                    help='path to masks file')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0
prune_op= None

feat = None
quan_modules = []

def hook(module, input, output):
    global feat
    feat = output.data.cpu().numpy()

target_sparsity = np.asarray([0.3, 0, 0.4, # stem, layer0
                        0.5, 0.2, 0, 0.3, 0.2, # layer1
                        0.5, 0.3, 0.3, 0.5, 0.2, # layer2
                        0.5, 0, 0.3, 0.4, 0.8, 0.8, 0.2, # layer3
                        0.5, 0.5, 0.4, 0.5, 0.6, 0.7, 0.5, 0.5, # layer4
                        0.6, 0.5, 0.4, 0.6, 0.4, 0.5, 0.5, 0.5, # layer5
                        0.5, 0.5, 0.2, 0.4, 0.5, 0.7, 0.6, 0.5, # layer6
                        0.4, 0.2, 0.4, 0.5, 0.8, 0.8, 0.2, 0.2, # layer7
                        0.6, 0.5, 0.6, 0.7, 0.7, 0.5, 0.5, # layer8
                        0.6, 0.4, 0.6, 0.6, 0.5, 0.5, 0.5, # layer9
                        0.4, 0.4, 0.3, 0.5, 0.6, 0.5, 0.6, 0.3, 0.3, # layer10
                        0.6, 0.5, 0.4, 0.5, 0.6, 0.8, 0.5, 0.6, 0.6, 0.5, # layer11
                        0.4, 0.3, 0.3, 0.5, 0.6, 0.6, 0.7, 0.6, 0.5, 0.5, # layer12
                        0.5, 0.2, 0.4, 0.6, 0.6, 0.7, 0.4, 0.4, 0.5, # layer13
                        0.5, 0.5, 0.6, 0.6, 0.7, 0.65, 0.45, 0.5, 0.5, # layer14
                        0.5, 0.3, 0.5, 0.6, 0.6, 0.7, 0.5, 0.5, 0.5, # layer15
                        0.6, 0.7])


signed = [None, False, False, True, True, False, False, False, True, True, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False]

def main():
    args = parser.parse_args()

    args.results_dir = './checkpoint'
    args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    args.save_path = save_path

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))


    ## student model
    model = models.__dict__[args.arch](pretrained=False, dropout=args.dropout, dropconnect=args.dropconnect, bit_width=args.act_bit_width)

    idx = 0
    for m in model.modules():
        if isinstance(m, Quantization):
            m.set_bitwidth(args.act_bit_width)
            m.set_sign(signed[idx])
            quan_modules.append(m)
            idx += 1

    model.cuda()
    # model = torch.nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    input_size = 224

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(input_size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    args.epoch_size = len(train_dataset) // args.batch_size

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(int(input_size/0.875), interpolation=Image.BICUBIC), # == 256
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    global prune_op
    prune_op = PruneOp(model, target_sparsity)

    if args.pretrained:
        checkpoint = torch.load(args.pretrained)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            prune_op.set_masks(checkpoint['masks'])
        else:
            # model.load_state_dict(checkpoint)
            state_dict = checkpoint
            new_state_dict = OrderedDict()
            for key_ori, key_pre in zip(model.state_dict().keys(), state_dict.keys()):
                new_state_dict[key_ori] = state_dict[key_pre]
            model.load_state_dict(new_state_dict)
        prune_op.init_pruning()

    print(args)
    print('Param sparsit:', prune_op.get_sparsity())
    prune_op.mask_params()

    # print('pretrained validation')
    # validate(val_loader, model, criterion, args)
    if args.scales:
        print("inferring sign for quantization ('{}bit')...".format(args.act_bit_width))
        scales = np.load(args.scales)
        # enable feature map quantization
        for index, q_module in enumerate(quan_modules):
            q_module.set_scale(scales[index])
            if signed[index] is not None:
                q_module.enable_quantization()
    else:
        print("quantizing ('{}bit')...".format(args.act_bit_width))
        quantize(train_dataset, model, args)

    print('validate...')
    validate(val_loader, model, criterion, args)


def quantize(train_dataset, model, args):

    model.eval()

    def get_safelen(x):
        # Assuming more than 1/10 values are valid (i.e., positive values for unsigned quantization).
        # For each batch, we sample 10^(k-1) values (k=floor(ln(feat_len))).
        x = x / 10
        y = 1
        while(x>=10):
            x = x / 10
            y = y * 10
        return int(y)

    # act_sta_len = 3000000
    act_sta_len = 2000000
    # act_sta_len = 100000
    feat_buf = np.zeros(act_sta_len)

    scales = np.zeros(len(quan_modules))

    with torch.no_grad():
        for index, q_module in enumerate(quan_modules):
            batch_iterator = iter(torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                             num_workers=args.workers, pin_memory=True))
            images, targets = next(batch_iterator)
            images = images.cuda()
            targets = targets.cuda()

            #### ADD HANDLE ####
            handle = q_module.register_forward_hook(hook)

            model(images)
            # handle.remove()

            #global feat
            feat_len = feat.size
            per_batch = min(get_safelen(feat_len), 100000)
            n_batches = int(act_sta_len / per_batch)

            failed = True
            while(failed):
                failed = False
                print('Extracting features for ', n_batches, ' batches...')
                for batch_idx in range(0, n_batches):
                    if batch_idx % args.epoch_size == 0:
                        batch_iterator = iter(torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                                         num_workers=args.workers, pin_memory=True))
                        
                    images, targets = next(batch_iterator)
                    images = images.cuda()
                    targets = targets.cuda()
                    # forward
                    model(images)

                    #global feat
                    feat_tmp = np.abs(feat).reshape(-1)
                    if feat_tmp.size < per_batch:
                        per_batch = int(per_batch / 10)
                        n_batches = int(n_batches * 10)
                        failed = True
                        break
                    np.random.shuffle(feat_tmp)
                    feat_buf[batch_idx*per_batch:(batch_idx+1)*per_batch] = feat_tmp[0:per_batch]

                if(not failed):
                    print('Init quantization... ')
                    scales[index], _signed = q_module.init_quantization(feat_buf)
                    print(scales[index])
                    np.save(os.path.join(args.save_path, args.arch + '_scales_act_' + str(args.act_bit_width) + 'bit.npy'), scales)
                    #q_module.set_scale(scales[index])
            #### REMOVE HANDLE ####
            handle.remove()

    np.save(os.path.join(args.save_path, args.arch + '_scales_act_' + str(args.act_bit_width) + 'bit.npy'), scales)
    np.save(args.arch + '_scales_act_' + str(args.act_bit_width) + 'bit.npy', scales)
    # enable feature map quantization
    for index, q_module in enumerate(quan_modules):
        q_module.set_scale(scales[index])


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    # prune_op.pruning()
    # print('Sparsit:', prune_op.get_sparsity())

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

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

        # prune_op.restore()


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
