'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import autoaug_dataset as datasets
import models.cifar as models
from models.attention_best import AttentionLayer

import loss

from lr_schedular.warm_up_cos import WarmupCosineLR
from lr_schedular.warm_up import WarmupLR
from lr_schedular.warm_up_step import WarmupStepLR
from lr_schedular.step import StepLR

from utils import AverageMeter, accuracy, mkdir_p


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--augnum', default = 2, type = int,
                    help = 'number of data augmentation to be aplied on dataset')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--base-lr', default = 0.01, type = float,
                     help = 'base learning rate')
parser.add_argument('--lr-schedular', default = 'WarmupStep', type = str,
                     help = 'the learning rate schedular')
parser.add_argument('--warm-up-epochs', default = 5, type = int,
                     help = 'epochs for learning rate warming up')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--loss', default = 'CrossEntropy', type = str,
                     help = 'the loss function, can be one of CrossEntropy, LabelSmooth')
parser.add_argument('--eps', default = 0.1, type = float,
                     help = 'epsilon for label smooth loss')
parser.add_argument('--no-bias-decay', default = 'True', type = str,
                    help = 'whether or not eliminite bias decay')
parser.add_argument('--init', default = 'XavierUniform', type = str,
                    help = 'initialization of parameters, can be one of XavierUniform, XavierNormal, Default')
parser.add_argument('--attention-weight-decay', default = 4e-5, type = float,
                    help = 'weight decay for attention layers')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = 66666 
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

lr_schedular = None

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.enables = True


def main():
    global best_acc
    global lr_schedular
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)



    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100


    trainset = dataloader(root='../../cifar', train=True, scale = 255, augnum = args.augnum)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='../../cifar', train=False, scale = 255)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    
    num_samples = len(trainset)
    
#    lr_schedular = WarmupCosineLR(args.train_batch, num_samples, args.epochs, args.base_lr, args.lr, args.warm_up_epochs)
    if args.lr_schedular == 'WarmupStep':
      lr_schedular = WarmupStepLR(args.train_batch, num_samples, args.base_lr, args.lr, args.warm_up_epochs, args.epochs, args.gamma, args.schedule)
    elif args.lr_schedular == 'WarmupCosine':
      lr_schedular = WarmupCosineLR(args.train_batch, num_samples, args.epochs, args.base_lr, args.lr, args.warm_up_epochs)
    elif args.lr_schedular == 'Step':
      lr_schedular = StepLR(args.train_batch, num_samples, args.lr, args.gamma, args.schedule)
    else:
      print('Unrecognized learning rate schedular:%s'%(args.lr_schedular))
    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                    init = args.init
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
#    criterion = nn.CrossEntropyLoss()
#    criterion = loss.LabelSmoothLoss()
    if args.loss == 'CrossEntropy':
      criterion = nn.CrossEntropyLoss()
    elif args.loss == 'LabelSmooth':
      criterion = loss.LabelSmoothLoss(args.eps)
    elif args.loss == 'HPLoss':
      criterion = loss.HPLoss(args.eps)
    else:
      print('Unrecognized loss: %s'(args.loss))
#    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
#    if args.no_bias_decay == 'True':
#      print('No bias decay')
#    params = split_model_params(model)
#    else:
#      print('With bias decay')
#      params = model.parameters()
    optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_schedular.batch_idx = start_epoch * lr_schedular.iterations_per_epoch
#        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
#    else:
#        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
#        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


#    for param_group in optimizer.param_groups:
#      param_group['lr'] = args.lr

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
#        adjust_learning_rate(optimizer, epoch)
        #lr = optimizer.param_groups[0]['lr']

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr_schedular.lr))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)

        print("Epoch %d, loss: %f(%f), accuracy: %f(%f)"%(epoch + 1, train_loss, test_loss, train_acc, test_acc))

        # append logger file
#        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

#    logger.close()
#    logger.plot()
#    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    global lr_schedular
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    dtime = 0
    forward_time = 0
    backward_time = 0
    t = 0

#    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        lr_schedular(optimizer)
        
#        print('Iteration: %d, lr: %f'%(lr_schedular.batch_idx, optimizer.param_groups[0]['lr']))
        # measure data loading time
        start = time.time()
        dtime += start - end
        data_time.update(time.time() - end)
#        if batch_idx % 100 == 99:
#            print('Iteration: %d, load data time: %f s'%(batch_idx + 1, dtime))
#            dtime = 0

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        forward_start = time.time()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        forward_end = time.time()
        forward_time += forward_end - forward_start

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
#        print(loss.data)
#        losses.update(loss.data[0], inputs.size(0))
        losses.update(loss.item(), inputs.size(0))
#        top1.update(prec1[0], inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
#        top5.update(prec5[0], inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        backward_start = time.time()
        loss.backward()
        backward_end = time.time()
        backward_time += backward_end - backward_start
        optimizer.step()

        # measure elapsed time
        batch_time.update(t)
        end = time.time()
        t += end - start

        if (batch_idx % 100 == 99):
            print("IterationL %d, data time: %fs, forward time: %f s, backward time: %f s, total time: %f s, top1 accuracy: %f, loss: %f"
                   %(batch_idx + 1, dtime, forward_time, backward_time, t, top1.avg, losses.avg))
#            print("Iteration: %d, running time: %f s, top1 accuracy: %f, loss: %f"%(batch_idx + 1, t, top1.avg, losses.avg))
            t = 0
            forward_time = 0
            dtime = 0
            backward_time = 0

        # plot progress
#        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
#                    batch=batch_idx + 1,
#                    size=len(trainloader),
#                    data=data_time.avg,
#                    bt=batch_time.avg,
#                    total=bar.elapsed_td,
#                    eta=bar.eta_td,
#                    loss=losses.avg,
#                    top1=top1.avg,
#                    top5=top5.avg,
#                    )
#        bar.next()
#    bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
#    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
#        forward_start = time.time()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
#        forward_end = time.time()
#        forward_time += forward_end - forward_start

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
#        losses.update(loss.data[0], inputs.size(0))
#        top1.update(prec1[0], inputs.size(0))
#        top5.update(prec5[0], inputs.size(0))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        # plot progress
#        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
#                    batch=batch_idx + 1,
#                    size=len(testloader),
#                    data=data_time.avg,
#                    bt=batch_time.avg,
#                    total=bar.elapsed_td,
#                    eta=bar.eta_td,
#                    loss=losses.avg,
#                    top1=top1.avg,
#                    top5=top5.avg,
#                    )
#$        bar.next()
#    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']
            
def split_model_params(model):
  attention_decay_params = []
  decay_params = []
  no_decay_params = []
  for n in model.modules():
    if isinstance(n, nn.Conv2d) or isinstance(n, nn.Linear):
      decay_params.append(n.weight)
      if n.bias is not None:
        no_decay_params.append(n.bias)
    elif isinstance(n, AttentionLayer):
      attention_decay_params.append(n.w0)
      attention_decay_params.append(n.w1)
      attention_decay_params.append(n.w2)
      attention_decay_params.append(n.bias0)
      attention_decay_params.append(n.bias1)
      attention_decay_params.append(n.bias2)
    else:
      if hasattr(n, 'weight') and n.weight is not None:
        no_decay_params.append(n.weight)
      if hasattr(n, 'bias') and n.bias is not None:
        no_decay_params.append(n.bias)
  assert len(list(model.parameters())) == len(decay_params) + len(no_decay_params) + len(attention_decay_params)
  return [
    {'params': decay_params},
    {'params': no_decay_params,
     'weight_decay': 0 if args.no_bias_decay == 'True' else args.weight_decay},
    {'params': attention_decay_params,
     'weight_decay': args.attention_weight_decay}
  ]
      

if __name__ == '__main__':
    main()
