import torch
import torch.nn as nn
import quantization
from quantization import *
from torch.utils import data

import argparse
import autoaug_dataset

import time
import os

parser = argparse.ArgumentParser()

parser.add_argument('--student-id', default = '0', type = str,
                    help = 'The gou ids to run the student model on')
parser.add_argument('--output-id', default = 0, type = int,
                    help = 'The gpu id to store the results on')
parser.add_argument('--test-batch', default = 64, type = int)
parser.add_argument('--resume', default = None, type = str,
                    help = 'Resume from model')
parser.add_argument('--half', default = 'False', type = str)
parser.add_argument('--data-root',  default = '.', type = str,
                   help = 'The root directory of CIFAR100 dataset.')

args = parser.parse_args()


def main():
  student_ids = [int(id) for id in args.student_id.split(',')]

  # Prepare dataset
  test_set = autoaug_dataset.CIFAR100(root = args.data_root, train = False, scale = 255)
  
  testloader = data.DataLoader(test_set, batch_size = args.test_batch, shuffle = False, num_workers = 4)

  # Prepare student and teacher models
  student_model = quantization.densenet_merged.densenet(depth = 100, num_classes = 100, growthRate = 12, compressionRate = 2, bitwidth = 4)
  student_model = nn.DataParallel(student_model, device_ids = student_ids, output_device = args.output_id)
  # Before applying parallel computation, models must be fetched on device_ids[0] first
  student_model.to(torch.device(student_ids[0]))

  state_dict = torch.load(args.resume)


  names = [name for name in state_dict.keys()]
  for name in names:
    value = state_dict[name]
    del state_dict[name]
    state_dict['module.' + name] = value

  student_model.load_state_dict(state_dict, strict = True)

  if args.half == 'True':
    student_model.module.set_half()

  s_acc = test(student_model, testloader)
  print('student acc: %.2f'%(s_acc * 100))
    
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

  print(acc_sum.item(), n)
  return acc_sum.item() / n

if __name__ == '__main__':
  main()

