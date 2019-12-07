import torch
import torch.nn as nn
import loss
import quantization
from quantization import *
from models import cifar
from torch.utils import data
import sparse_param

import argparse
import random
from lr_schedular.warm_up_cos import WarmupCosineLR
import autoaug_dataset

import time
import os

parser = argparse.ArgumentParser()

# Optimization arguments
parser.add_argument('--lr', default = 0.1, type = float,
                    help = 'The learning rate for training')
parser.add_argument('--start-lr', default = 0.001, type = float,
                    help = 'The start up learning rate for lr warmup')
parser.add_argument('--weight-decay', default = 1e-4, type = float,
                    help = 'The weight decay for training')
parser.add_argument('--momentum', default = 0.9, type = float,
                    help = 'The momentum for training')
parser.add_argument('--warm-up-epochs', default = 5, type = int,
                    help = 'The epochs to apply lr warmup')
parser.add_argument('--epochs', default = 300, type = int,
                    help = 'The total epochs for training')
parser.add_argument('--train-batch', default = 64, type = int,
                    help = 'Training batch size')
parser.add_argument('--test-batch', default = 64, type = int,
                    help = 'Testing batch size')
parser.add_argument('--loss', default = 'ce:0', type = str,
                    help = 'Loss function')

# Teacher student arguments
parser.add_argument('--teacher', default = '', type = str,
                    help = 'The teacher model')
parser.add_argument('--teacher-id', default = 0, type = str,
                    help = 'The gpu ids to run the teacher model on')
parser.add_argument('--student-id', default = 0, type = str,
                    help = 'The gou ids to run the student model on')
parser.add_argument('--output-id', default = 0, type = int,
                    help = 'The gpu id to store the results on')

# Resume and save arguments
parser.add_argument('--resume', default = None, type = str,
                    help = 'Resume from model')
parser.add_argument('--resume-optimizer-state', default = 'True', type = str,
                    help = '')
parser.add_argument('--checkpoint', default = 'checkpoint', type = str,
                    help = '')

# Pruning arguments
parser.add_argument('--update-interval', default = 50, type = int,
                    help = '')
parser.add_argument('--update-stop-iter', default = 5000, type = int,
                    help = '')
parser.add_argument('--rate', default = 0, type = float,
                    help = 'Sparse rate')

# Quantization arguments
parser.add_argument('--act-bitwidth', default = 8, type = int,
                    help = 'Bitwidth for activation quantization')
parser.add_argument('--weight-bitwidth', default = 8, type = int,
                    help = 'Bitwidth for weight quantization')

args = parser.parse_args()


random_seed = 88888
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

iteration = 0
def main():
  global iteration
  teacher_ids = [int(id) for id in args.teacher_id.split(',')]
  student_ids = [int(id) for id in args.student_id.split(',')]

  # Prepare dataset
  train_set = autoaug_dataset.CIFAR100(root = '../../cifar', train = True, scale = 255)
  test_set = autoaug_dataset.CIFAR100(root = '../../cifar', train = False, scale = 255)
  trainloader = data.DataLoader(train_set, batch_size = args.train_batch, shuffle = True, num_workers = 4)
  testloader = data.DataLoader(test_set, batch_size = args.test_batch, shuffle = False, num_workers = 4)
  
  # Prepare student and teacher models
  student_model = quantization.densenet.densenet(depth = 100, num_classes = 100, growthRate = 12, compressionRate = 2, prune_fc = True)
  teacher_model = cifar.densenet(depth = 172, num_classes = 100, growthRate = 30, compressionRate = 2)
  student_model = nn.DataParallel(student_model, device_ids = student_ids, output_device = args.output_id)
  teacher_model = nn.DataParallel(teacher_model, device_ids = student_ids, output_device = args.output_id)
  # Before applying parallel computation, models must be fetched on device_ids[0] first
  student_model.to(torch.device(student_ids[0]))
  teacher_model.to(torch.device(teacher_ids[0]))
  # Load teacher model from teacher checkpoint
  teacher_checkpoint = torch.load(args.teacher)
  teacher_model.load_state_dict(teacher_checkpoint['state_dict']) 

  params = None
  masks = None
  alphas = None
  # Resume from last training
  start_epoch = 0
  best_acc = 0
  if args.resume is not None:
    student_checkpoint = torch.load(args.resume)
    student_model.load_state_dict(student_checkpoint['state_dict'])
#    load_state_dict(student_model, student_checkpoint['state_dict'])
    if args.resume_optimizer_state == 'True':
      start_epoch = student_checkpoint['epoch']
    if 'params' in student_checkpoint:
      print('Resuming parameters...')
      params = student_checkpoint['params']
    if 'masks' in student_checkpoint:
      print('Resuming masks...')
      masks = student_checkpoint['masks']
    if 'alphas' in student_checkpoint:
      print('Resuming alphas...')
      alphas = student_checkpoint['alphas']

  assert(masks is not None and params is not None and alphas is not None)

  ########
  acc = test(net, test_loader)
  print('Original acc:%.2f'%(acc * 100))

  best_acc = acc
  checkpoint = args.resume + '.upbn'
  torch.save({
    'state_dict': net.state_dict(),
    'masks': masks,
    'params': params,
    'alphas': alphas
  }, checkpoint)

  for epoch in range(10):
    quantization.upbn(net, train_loader, 1)
    acc = test(net, test_loader)
    if acc > best_acc:
      best_acc = acc
      torch.save({
        'state_dict': net.state_dict(),
        'masks': masks,
        'params': params,
        'alphas': alphas
      }, checkpoint)
  ##########

  student_checkpoint = torch.load(args.resume + '.upbn')
  student_model.load_state_dict(student_checkpoint['state_dict'])
#   load_state_dict(student_model, student_checkpoint['state_dict'])
  if args.resume_optimizer_state == 'True':
    start_epoch = student_checkpoint['epoch']
  if 'params' in student_checkpoint:
    print('Resuming parameters...')
    params = student_checkpoint['params']
  if 'masks' in student_checkpoint:
    print('Resuming masks...')
    masks = student_checkpoint['masks']
  if 'alphas' in student_checkpoint:
    print('Resuming alphas...')
    alphas = student_checkpoint['alphas']

  best_acc = 0

  # Prepare optimizer for fc weight pruning
  fc_weights, fc_biases = split_params(student_model)
  sp = sparse_param.SparseParam(fc_weights, mask_policy = 'global') 
  sp.update_mask(0)
  sp.update_params()

  # alpha will be fixed during training after updated
  set_quant_upalpha(student_model, False)

  
 
  optimizer = torch.optim.SGD(sp.params + fc_biases, lr = args.lr, weight_decay = args.weight_decay, momentum = args.momentum)
  lrs = WarmupCosineLR(args.train_batch, len(train_set), args.epochs, args.start_lr, args.lr, args.warm_up_epochs, start_epoch)

  loss_func = loss.get_loss(args.loss)

  if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)

  s_acc = test(student_model, testloader)
  t_acc = test(teacher_model, testloader)
  print('Teacher acc: %.2f, student acc: %.2f'%(t_acc * 100, s_acc * 100))

  iteration = start_epoch * len(trainloader)
  for epoch in range(start_epoch, args.epochs):
    print('Epoch %d | %d, learning rate: %f'%(epoch+1, args.epochs, lrs.lr))
    start = time.time()
    train_acc = train(student_model, teacher_model, trainloader, optimizer, loss_func, lrs, sp)
    test_acc = test(student_model, testloader)
    end = time.time()
    if test_acc > best_acc:
      best_acc = test_acc
      torch.save({
        'epoch': epoch,
        'state_dict':student_model.state_dict(),
        'params': params,
        'masks': masks,
        'alphas': alphas,
        'fc_params': sp.params,
        'fc_masks': sp.masks
      }, os.path.join(args.checkpoint, 'model_best.pth.tar'))
    print('Running time: %.2fmin, acc: %.2f(%.2f), best acc: %.2f'%((end - start) / 60, train_acc * 100, test_acc * 100, best_acc * 100))

def train(student_model, teacher_model, dataloader, optimizer, loss_func, lrs, sp):
  global iteration
  acc_sum = torch.zeros(1, device = torch.device(args.output_id)).int()
  n = 0
  student_model.eval()
  teacher_model.train()
  for batch_idx, (data, label) in enumerate(dataloader):
    lrs(optimizer)
    if (iteration < args.update_stop_iter and args.update_interval > 0 and (iteration + 1) % args.update_interval == 0) or iteration == 0:
      sp.update_mask(args.rate)
    iteration += 1
    with torch.no_grad():
      sp.update_params()

    student_out = student_model(data).softmax(dim = 1)
    with torch.no_grad():
      teacher_out = teacher_model(data).softmax(dim = 1)
    l = loss_func(student_out, teacher_out)
    optimizer.zero_grad()
    l.backward()
    sp.attach_gradients()
    optimizer.step()
    
    label = label.to(torch.device(args.output_id))
    acc_sum += (student_out.argmax(dim = 1).float() == label.float()).sum().int()
    n += data.size(0)
#    print(acc_sum.item() / n, ' ', l.item())

  return acc_sum.item() / n
    
    


def test(student_model, dataloader):
  acc_sum = torch.zeros(1, device = torch.device(args.output_id)).int()
  n = 0
  student_model.eval()
  for data, label in dataloader:
    with torch.no_grad():
      out = student_model(data)
    label = label.to(torch.device(args.output_id))
    acc_sum += (out.argmax(dim = 1).float() == label.float()).sum().int()
    n += data.size(0)
  return acc_sum.item() / n

def split_params(model):
  fc_weights = [] 
  fc_biases = []
  for m in model.modules():
    if isinstance(m, nn.Linear):
      fc_weights.append(m.weight)
      if m.bias is not None:
        fc_biases.append(m.bias)

  return fc_weights, fc_biases

if __name__ == '__main__':
  main()

