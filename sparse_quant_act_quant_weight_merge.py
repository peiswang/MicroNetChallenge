import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--resume', default = None, type = str,
                    help = 'The original model to be merged.')
parser.add_argument('--checkpoint', default = None, type = str,
                    help = 'The file to save the merged model, default to be $resume$.merged')
parser.add_argument('--weight-bitwidth', default = 8, type = int,
                    help = 'The bitwidth of weight.')
parser.add_argument('--gpu-id', default = 0, type = int,
                    help = 'The device to run on')
parser.add_argument('--depth', default = 100, type = int,
                    help = 'The depth of the model')

args = parser.parse_args()

def main():
  assert(args.resume is not None)
  checkpoint = torch.load(args.resume)

  if args.checkpoint is None:
    args.checkpoint = args.resume + '.merged'
    
  assert('state_dict' in checkpoint)
  state_dict = checkpoint['state_dict']
  for name, value in state_dict.items():
    state_dict[name] = value.cuda(args.gpu_id)

  assert('params' in checkpoint)
  params = checkpoint['params']
  for i, param in enumerate(params):
    params[i] = param.cuda(args.gpu_id)

  assert('masks' in checkpoint)
  masks = checkpoint['masks']
  for i, mask in enumerate(masks):
    masks[i] = mask.cuda(args.gpu_id)

  assert('alphas' in checkpoint)
  alphas = checkpoint['alphas']
  for i, alpha in enumerate(alphas):
    alphas[i] = alpha.cuda(args.gpu_id)

  assert('fc_params' in checkpoint)
  fc_params = checkpoint['fc_params']
  fc_params = [param.cuda(args.gpu_id) for param in fc_params]

  assert('fc_masks' in checkpoint)
  fc_masks = checkpoint['fc_masks']
  fc_masks = [mask.cuda(args.gpu_id) for mask in fc_masks]

  assert('fc_alphas' in checkpoint)
  fc_alphas = checkpoint['fc_alphas']
  fc_alphas = [alpha.cuda(args.gpu_id) for alpha in fc_alphas]

  n = (args.depth - 4) // 6

  bn_means = []
  bn_vars = []
  bn_weights = []
  bn_biases = []
  
  quant_act_alphas = []
  quant_act_bitwidths = []
  quant_act_alpha_means = []

  fc_weights = []
  fc_biases = []

  for name, value in state_dict.items():
    if 'bn' in name and name.endswith('running_mean'):
      bn_means.append(value)
    if 'bn' in name and name.endswith('running_var'):
      bn_vars.append(value)
    if 'bn' in name and name.endswith('weight'):
      bn_weights.append(value)
    if 'bn' in name and name.endswith('bias'):
      bn_biases.append(value)
    if 'act' in name and name.endswith('alpha'):
      quant_act_alphas.append(value)
    if 'act' in name and name.endswith('bitwidth'):
      quant_act_bitwidths.append(value)
    if 'act' in name and name.endswith('alpha_mean'):
      quant_act_alpha_means.append(value)
    if 'fc' in name and name.endswith('weight'):
      fc_weights.append(value)
    if 'fc' in name and name.endswith('bias'):
      fc_biases.append(value)

#  print(len(bn_means), len(quant_act_alphas), len(params))

  for alpha in alphas:
    if (alpha < 0).sum().item() > 0:
      print(alpha)

  new_params_list = []

  # the first convolutional layer
  new_params_list.append(_quantize(params[0], masks[0], alphas[0], args.weight_bitwidth))
  new_params_list.append(alphas[0])
  new_params_list.append(torch.zeros_like(alphas[0])) 

  def merge_block(block_id, layer_id):
    idx = block_id * (2 * n + 1) + layer_id * 2
    bn1_weight = bn_weights[idx].flatten()
    bn1_bias = bn_biases[idx].flatten()
    bn1_mean = bn_means[idx].flatten()
    bn1_var = bn_vars[idx].flatten()
    
    conv1_weight = params[idx + 1]
    conv1_mask = masks[idx + 1]
    conv1_alpha = alphas[idx + 1].flatten()

    quant1_alpha = quant_act_alphas[idx].flatten()
    quant1_bitwidth = quant_act_bitwidths[idx].flatten()
    quant1_alpha_mean = quant_act_alpha_means[idx].flatten()


    #########################
    bn2_weight = bn_weights[idx + 1].flatten()
    bn2_bias = bn_biases[idx + 1].flatten()
    bn2_mean = bn_means[idx + 1].flatten()
    bn2_var = bn_vars[idx + 1].flatten()
    
    conv2_weight = params[idx + 2]
    conv2_mask = masks[idx + 2]
    conv2_alpha = alphas[idx + 2].flatten()

    quant2_alpha = quant_act_alphas[idx + 1].flatten()
    quant2_bitwidth = quant_act_bitwidths[idx + 1].flatten()
    quant2_alpha_mean = quant_act_alpha_means[idx + 1].flatten()

    # merge the first bn layer and the first quant layer
    # to a single scale layer
    bn1_var_sqrt = (bn1_var + 1e-5).sqrt()
    scale1 = bn1_weight / quant1_alpha / bn1_var_sqrt
    bias1 = - bn1_weight * bn1_mean / quant1_alpha / bn1_var_sqrt + bn1_bias / quant1_alpha
    new_params_list.append(scale1)
    new_params_list.append(bias1)
#    new_params_list.append(quant1_bitwidth)
#    new_params_list.append(quant1_alpha_mean)

    # the first convolution layer
    new_params_list.append(_quantize(conv1_weight, conv1_mask, conv1_alpha, args.weight_bitwidth))

    # merge the second bn layer
    alpha1 = quant1_alpha * conv1_alpha
    alpha2 = quant2_alpha
    bn2_var_sqrt = (bn2_var + 1e-5).sqrt()
    scale2 = bn2_weight * alpha1 / alpha2 / bn2_var_sqrt
    bias2 = - bn2_weight * bn2_mean / alpha2 / bn2_var_sqrt + bn2_bias / alpha2
    new_params_list.append(scale2)
    new_params_list.append(bias2)
#    new_params_list.append(quant2_bitwidth)
#    new_params_list.append(quant2_alpha_mean)

    # the second convolution layer
    new_params_list.append(_quantize(conv2_weight, conv2_mask, conv2_alpha, args.weight_bitwidth))

    # each convolution layer should be followed by a scale layer
    # in order to dequantize the output feature maps
    new_params_list.append(quant2_alpha * conv2_alpha)
    new_params_list.append(torch.zeros_like(conv2_alpha))

  def merge_transition(block_id):
    idx = block_id * (2 * n + 1) + 2 * n
    bn_weight = bn_weights[idx].flatten()
    bn_bias = bn_biases[idx].flatten()
    bn_mean = bn_means[idx].flatten()
    bn_var = bn_vars[idx].flatten()
    
    conv_weight = params[idx + 1]
    conv_mask = masks[idx + 1]
    conv_alpha = alphas[idx + 1]

    quant_alpha = quant_act_alphas[idx].flatten()
    quant_bitwidth = quant_act_bitwidths[idx].flatten()
    quant_alpha_mean = quant_act_alpha_means[idx].flatten()   

    # merge the bn layer and the quantization layer into a single scale layer
    bn_var_sqrt = (bn_var + 1e-5).sqrt()
    scale = bn_weight / quant_alpha / bn_var_sqrt
    bias = - bn_weight * bn_mean / quant_alpha / bn_var_sqrt + bn_bias / quant_alpha  
    new_params_list.append(scale)
    new_params_list.append(bias)

    conv_weight_q = _quantize(conv_weight, conv_mask, conv_alpha, args.weight_bitwidth)

    alpha = (quant_alpha * conv_alpha).flatten()
    for i in range(alpha.numel()):
      if alpha[i].item() < 0:
        conv_alpha[i] = - conv_alpha[i]
        conv_weight_q[i,:] = - conv_weight[i,:]
   

    # convolution layer
    new_params_list.append(conv_weight_q)

    # each convolution layer should be followed by a scale layer
    new_params_list.append(quant_alpha * conv_alpha)
    new_params_list.append(torch.zeros_like(conv_alpha)) 

  # densenet blocks
  for block_id in range(3):
    for layer_id in range(n):
      merge_block(block_id, layer_id)
    if block_id != 2:
      merge_transition(block_id)

  # the last fc layer
  bn_var = bn_vars[-1]
  bn_mean = bn_means[-1]
  bn_weight = bn_weights[-1]
  bn_bias = bn_biases[-1]
#  fc_weight = fc_weights[0]
  fc_bias = fc_biases[0]

  fc_param = fc_params[0]
  fc_mask = fc_masks[0]
  fc_alpha = fc_alphas[0]

  # scale layer merged from bn 
  bn_var_sqrt = (bn_var + 1e-5).sqrt()
  scale = bn_weight / bn_var_sqrt
  bias = - bn_weight * bn_mean / bn_var_sqrt + bn_bias
  new_params_list.append(scale)
  new_params_list.append(bias)

  # quantized fc layer
  new_params_list.append(_quantize(fc_param, fc_mask, fc_alpha, 6))
#  new_params_list.append(fc_bias)

  # the scale layer
  new_params_list.append(fc_alpha)
  new_params_list.append(fc_bias)

  torch.save(new_params_list, args.checkpoint)


def _quantize(param, mask, alpha, bitwidth):
  lower_bound = -(2 ** (bitwidth - 1))
  upper_bound = -lower_bound - 1
  alpha = alpha.view(alpha.numel(), 1)
  return ((param * mask).view(param.size(0), -1) / alpha).round().clamp(lower_bound, upper_bound).reshape(param.shape)

if __name__ == '__main__':
  main()
