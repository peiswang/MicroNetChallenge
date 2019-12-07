import argparse
import quantization
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--resume', default = None, type = str,
                    help = 'The parameter list file')

args = parser.parse_args()

def main():
  net = quantization.densenet_merged.densenet(depth = 100, num_classes = 100)

  # Load the parameter list
  params_list = torch.load(args.resume)

  print(len(list(net.parameters())), len(params_list))

  # Fetch parameter list into net
  assert(len(list(net.parameters())) == len(params_list))

  for net_param, param in zip(net.parameters(), params_list):
    net_param.data[:] = param.reshape(net_param.shape).cpu()

  torch.save(net.state_dict(), args.resume)

if __name__ == '__main__':
  main()
