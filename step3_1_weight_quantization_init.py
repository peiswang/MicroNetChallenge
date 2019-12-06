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
from quantization.quantize import WeightQuantizer

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
parser.add_argument('--teacher', metavar='ARCH', default='mixnet_l',
                    choices=model_names,
                    help='model architecture: ' +' | '.join(model_names) +
                        ' (default: mixnet_s)')
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
parser.add_argument('--weight-scales', default='', type=str, metavar='PATH',
                    help='path to the pre-calculated scales (.npy file)')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH', 
                    help='path to pre-trained model')
parser.add_argument('--act-bit-width', default=8, type=int,
                    help='activation quantization bit-width')
parser.add_argument('--scales', default='', type=str, metavar='PATH',
                    help='path to the pre-calculated scales (.npy file)')
parser.add_argument('--masks', default='', type=str, metavar='PATH', 
                    help='path to masks file')
parser.add_argument('--pretrained-teacher', default='', type=str, metavar='PATH', 
                    help='path to pre-trained model')
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

quan_modules = []

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

weight_quan_type = ['int5'] * target_sparsity.shape[0]
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

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    ## teacher model
    teacher_model = models.__dict__[args.teacher](pretrained=args.pretrained_teacher)
    teacher_model = torch.nn.DataParallel(teacher_model).cuda()

    for param in teacher_model.parameters():
        param.requires_grad = False

    ## student model
    model = models.__dict__[args.arch](pretrained=False, dropout=args.dropout, dropconnect=args.dropconnect)

    print("adding quan op ('{}bit')...".format(args.act_bit_width))
    scales = np.load(args.scales)
    idx = 0
    for m in model.modules():
        if isinstance(m, Quantization):
            m.set_quantization_parameters(signed[idx], args.act_bit_width, scales[idx])
            quan_modules.append(m)
            idx += 1

    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_refinery = refinery_loss.RefineryLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    if 'alexnet' in args.arch:
        input_size = 227
    else:
        input_size = 224

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            #transforms.RandomResizedCrop(input_size),
            transforms.RandomResizedCrop(input_size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    args.epoch_size = len(train_dataset) // args.batch_size

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

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

    global prune_op
    prune_op = PruneOp(model, target_sparsity)
    # prune_op.init_pruning()

    if args.pretrained:
        state_dict = torch.load(args.pretrained)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            prune_op.set_masks(state_dict['masks'])

        new_state_dict = OrderedDict()
        for key_ori, key_pre in zip(model.state_dict().keys(), state_dict.keys()):
            new_state_dict[key_ori] = state_dict[key_pre]
        model.load_state_dict(new_state_dict)

        prune_op.init_pruning()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            prune_op.set_masks(checkpoint['masks'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    print(args)
    print('Param sparsit:', prune_op.get_sparsity())

    prune_op.mask_params()

    # enable feature map quantization
    for index, q_module in enumerate(quan_modules):
        if q_module.signed is not None:
            q_module.enable_quantization()

    print("add weight quantizing op (int4.5*)...")
    weight_quantizer = WeightQuantizer(model, weight_quan_type)
    if args.weight_scales:
        weight_quantizer.load_scales(args.weight_scales)
    else:
        weight_quantizer.init_quantization()
        weight_quantizer.save_scales(os.path.join('./', args.arch + '_weight_5_4bit.pt'))
        weight_quantizer.save_scales(os.path.join(args.save_path, args.arch + '_weight_5_4bit.pt'))

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    print('upbn...')
    upbn(train_loader, model, args, 300, weight_quantizer)
    validate(val_loader, model, criterion, args, weight_quantizer)

    # save model after upbn
    state_dict = model.state_dict()
    new_state_dict = OrderedDict()
    best_path = os.path.join(args.save_path, 'mixnet_prune_quan_act_' + str(args.act_bit_width) + 'bit_weight_5_4bit_upbn.pth')
    for key in state_dict.keys():
        if 'module.' in key:
            new_state_dict[key.replace('module.', '')] = state_dict[key].cpu()
        else:
            new_state_dict[key] = state_dict[key].cpu()
    torch.save(new_state_dict, best_path)


def train(train_loader, teacher_model, model, criterion, optimizer, epoch, args, weight_quantizer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_kd = AverageMeter()
    losses_gt = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    teacher_model.train()
    model.train()

    end = time.time()

    if args.lr_policy == 'step':
        local_lr = adjust_learning_rate(optimizer, epoch, args)
    elif args.lr_policy == 'epoch_poly':
        local_lr = adjust_learning_rate_epoch_poly(optimizer, epoch, args)
        
    for i, (input, target) in enumerate(train_loader):
        global_iter = epoch * args.epoch_size + i
        if args.lr_policy == 'iter_poly':
            local_lr = adjust_learning_rate_poly(optimizer, global_iter, args)
        elif args.lr_policy == 'cosine':
            local_lr = adjust_learning_rate_cosine(optimizer, global_iter, args)
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        update_p = 1.0 / (1.0 + 0.0001 * global_iter)

        if weight_quantizer:
            weight_quantizer.quantization()

        # compute output
        with torch.no_grad():
            teacher_output = teacher_model(input)
        teacher_labels = F.softmax(teacher_output, dim=1)
        student_output = model(input)
        kd_loss = criterion((student_output, teacher_labels), target)
        kd_loss = kd_loss[0]
        loss = kd_loss

        # measure accuracy and record loss
        acc1, acc5 = accuracy(student_output, target, topk=(1, 5))
        losses_kd.update(kd_loss.item(), input.size(0))
        # losses_gt.update(lb_loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # prune_op.restore()
        prune_op.mask_grad()
        if weight_quantizer:
            # weight_quantizer.clip_grad()
            weight_quantizer.restore()

        optimizer.step()

        torch.cuda.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            # print(prune_op.get_sparsity())
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss-kd {loss_kd.val:.4f} ({loss_kd.avg:.4f})\t'
                  'Loss-gt {loss_gt.val:.4f} ({loss_gt.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'LR: {lr: .8f}'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss_kd=losses_kd, loss_gt=losses_gt, top1=top1, top5=top5, lr=local_lr))

def validate(val_loader, model, criterion, args, weight_quantizer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    if weight_quantizer:
        weight_quantizer.quantization()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
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
    if weight_quantizer:
        weight_quantizer.restore()

    return top1.avg


def upbn(train_loader, model, args, maxiter=300, weight_quantizer=None):
    # switch to train mode
    model.train()

    if weight_quantizer:
        weight_quantizer.quantization()

    with torch.no_grad():
        for i, (input, target) in enumerate(train_loader):
            if i == maxiter:
                break
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            #target = target.cuda(args.gpu, non_blocking=True)
            # compute output
            output = model(input)

    if weight_quantizer:
        weight_quantizer.restore()


def save_checkpoint(state, is_best, path='./', filename='checkpoint'):
    saved_path = os.path.join(path, filename+'.pth.tar')
    torch.save(state, saved_path)

    if is_best:
        state_dict = state['state_dict']
        new_state_dict = OrderedDict()
        best_path = os.path.join(path, 'model_best.pth')
        for key in state_dict.keys():
            if 'module.' in key:
                new_state_dict[key.replace('module.', '')] = state_dict[key].cpu()
            else:
                new_state_dict[key] = state_dict[key].cpu()
        torch.save(new_state_dict, best_path)


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


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
    
def adjust_learning_rate_epoch_poly(optimizer, epoch, args):
    """Sets epoch poly learning rate"""
    lr = args.lr * ((1 - epoch * 1.0 / args.epochs) ** args.power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_learning_rate_poly(optimizer, global_iter, args):
    """Sets iter poly learning rate"""
    lr = args.lr * ((1 - global_iter * 1.0 / (args.epochs * args.epoch_size)) ** args.power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_learning_rate_cosine(optimizer, global_iter, args):
    warmup_lr = args.lr * args.warmup_lr_multiplier
    max_iter = args.epochs * args.epoch_size
    warmup_iter = args.warmup_epochs * args.epoch_size
    if global_iter < warmup_iter:
        slope = (args.lr - warmup_lr) / warmup_iter
        lr = slope * global_iter + warmup_lr
    else:
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * (global_iter - warmup_iter) / (max_iter - warmup_iter)))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

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
