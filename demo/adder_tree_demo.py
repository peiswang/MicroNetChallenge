import torch
from torch.nn import functional as F
import numpy as np
import math
import time

def Add_Tree_GEMM(A, B, conv_param):
    M = A.shape[0]
    N = B.shape[1]
    K = A.shape[1]

    A = torch.from_numpy(A)
    B = torch.from_numpy(B) # weights

    act_signed = conv_param['act_signed']
    act_bitwidth = conv_param['act_bitwidth']
    weight_bitwidth = conv_param['weight_bitwidth']

    bitwidth_init = act_bitwidth + weight_bitwidth

    if act_signed:
        bitwidth_init -= 1

    C = np.zeros((M,N)).astype(np.float32)

    assert(K==B.shape[0]) # M*K x K*N = M*N

    for m in range(M):
        for n in range(N):
            a = A[m,:]
            b = B[:,n]
            c = a*b
            c_tmp = c[b!=0.0]
            bitwidth = bitwidth_init

            while c_tmp.nelement() > 1:
                upper_bound = 2**(bitwidth-1)-1
                lower_bound = -1*upper_bound-1

                c_holder = -1*c_tmp
                c_tmp.clamp_(lower_bound, upper_bound)
                assert((c_holder+c_tmp).max().item()==0)

                num_c = c_tmp.nelement()
                n_c_div2 = num_c - num_c%2
                c_div2 = c_tmp[0:n_c_div2].contiguous().view(-1,2)
                c_div2 = torch.sum(c_div2, 1)
                num_c_new = num_c//2 + num_c%2

                c_new = torch.zeros(num_c_new).to(c_tmp.device)
                c_new[0:num_c//2] = c_div2
                if num_c%2 == 1:
                    c_new[-1] = c_tmp[-1]

                bitwidth += 1
                c_tmp = c_new

            if c_tmp.nelement() == 0:
                C[m,n] = 0
            else:
                upper_bound = 2**(bitwidth-1)-1
                lower_bound = -1*upper_bound-1

                c_holder = -1*c_tmp
                c_tmp.clamp_(lower_bound, upper_bound)
                assert((c_holder+c_tmp).max().item()==0)
                C[m,n] = c_tmp.item()
    return C

def im2col(x, hh, ww, stride):

    """
    Args:
      x: image matrix to be translated into columns, (C, H, W)
      hh: filter height
      ww: filter width
      stride: stride
    Returns:
      col: (new_h*new_w,hh*ww*C) matrix, each column is a cube that will convolve with a filter
            new_h = (H-hh) // stride + 1, new_w = (W-ww) // stride + 1
    """

    c, h, w = x.shape
    new_h = (h-hh) // stride + 1
    new_w = (w-ww) // stride + 1
    col = np.zeros([new_h*new_w,c*hh*ww]).astype(np.float32)

    for i in range(new_h):
       for j in range(new_w):
           patch = x[...,i*stride:i*stride+hh,j*stride:j*stride+ww]
           col[i*new_w+j,:] = np.reshape(patch,-1)
    return col

def col2im(mul, h_prime, w_prime, C):
    """
      Args:
      mul: (h_prime*w_prime*w,F) matrix, each col should be reshaped to C*h_prime*w_prime when C>0, or h_prime*w_prime when C = 0
      h_prime: reshaped filter height
      w_prime: reshaped filter width
      C: reshaped filter channel, if 0, reshape the filter to 2D, Otherwise reshape it to 3D
    Returns:
      if C == 0: (F,h_prime,w_prime) matrix
      Otherwise: (F,C,h_prime,w_prime) matrix
    """
    F = mul.shape[1]
    if(C == 1):
        out = np.zeros([F,h_prime,w_prime]).astype(np.float32)
        for i in range(F):
            col = mul[:,i]
            out[i,:,:] = np.reshape(col,(h_prime,w_prime))
    else:
        out = np.zeros([F,C,h_prime,w_prime]).astype(np.float32)
        for i in range(F):
            col = mul[:,i]
            out[i,:,:] = np.reshape(col,(C,h_prime,w_prime))

    return out

def conv_forward_naive(x, w, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    pad_num = conv_param['padding']
    stride = conv_param['stride']
    groups = conv_param['groups']
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H_prime = (H+2*pad_num-HH) // stride + 1
    W_prime = (W+2*pad_num-WW) // stride + 1
    out = np.zeros([N, F, H_prime, W_prime]).astype(np.float32)
    # im2col
    if groups == 1:
        for im_num in range(N):
            im = x[im_num,:,:,:]
            im_pad = np.pad(im,((0,0), (pad_num,pad_num), (pad_num,pad_num)), 'constant')
            im_col = im2col(im_pad, HH, WW, stride)
            filter_col = np.reshape(w,(F,-1))
            mul = Add_Tree_GEMM(im_col, filter_col.T, conv_param) # im_col.dot(filter_col.T)
            out[im_num,:,:,:] = col2im(mul,H_prime,W_prime,1)
    else:
        assert(groups == F and F == C)
        for im_num in range(N):
            im = x[im_num,:,:,:]
            im_pad = np.pad(im,((0,0), (pad_num,pad_num), (pad_num,pad_num)), 'constant')
            for group_id in range(groups):
                im_pad_i = im_pad[group_id,:,:]
                im_pad_i = np.expand_dims(im_pad_i, axis=0)
                im_col_i = im2col(im_pad_i, HH, WW, stride)
                w_i = w[group_id]
                filter_col_i = np.reshape(w_i,(1,-1))
                mul_i = Add_Tree_GEMM(im_col_i, filter_col_i.T, conv_param) # im_col.dot(filter_col.T)
                out[im_num,group_id,:,:] = col2im(mul_i, H_prime, W_prime, 1)
    return out

def test_conv3x3_uint():
    scaling_factor = 0.05
    scaling_factor = 0.01

    w = torch.randn(128, 256, 3, 3)
    x = torch.rand(1, 256, 17, 17)

    alpha_w = torch.max(w.abs()) / 15
    alpha_x = torch.max(x.abs()) / 127

    w = torch.clamp(torch.round(w/alpha_w), -15, 15).float()
    x = torch.clamp(torch.round(x/alpha_x), 0, 127).float()

    out_GT = F.conv2d(x, w, stride=1, padding=1, groups=1)

    w = w.cpu().numpy()
    x = x.cpu().numpy()

    w = w.astype(np.float32)
    x = x.astype(np.float32)

    conv_param = dict()
    conv_param['padding'] = 1
    conv_param['stride'] = 1
    conv_param['groups'] = 1
    conv_param['act_signed'] = False
    conv_param['act_bitwidth'] = 7
    conv_param['weight_bitwidth'] = 5

    out = conv_forward_naive(x, w, conv_param)

    out = torch.from_numpy(out).cuda()

    print('diff : ', torch.max((out.float()-out_GT.cuda()).abs()))

def test_conv3x3_int():
    scaling_factor = 0.05
    scaling_factor = 0.01

    w = torch.randn(128, 256, 3, 3)
    x = torch.randn(1, 256, 17, 17)

    alpha_w = torch.max(w.abs()) / 15
    alpha_x = torch.max(x.abs()) / 63

    w = torch.clamp(torch.round(w/alpha_w), -15, 15).float()
    x = torch.clamp(torch.round(x/alpha_x), -63, 63).float()

    out_GT = F.conv2d(x, w, stride=1, padding=1, groups=1)

    w = w.cpu().numpy()
    x = x.cpu().numpy()

    w = w.astype(np.float32)
    x = x.astype(np.float32)

    conv_param = dict()
    conv_param['padding'] = 1
    conv_param['stride'] = 1
    conv_param['groups'] = 1
    conv_param['act_signed'] = True
    conv_param['act_bitwidth'] = 7
    conv_param['weight_bitwidth'] = 5

    out = conv_forward_naive(x, w, conv_param)

    out = torch.from_numpy(out).cuda()

    print('diff : ', torch.max((out.float()-out_GT.cuda()).abs()))

def test_conv1x1_int():
    scaling_factor = 0.05
    scaling_factor = 0.01

    w = torch.randn(128, 256, 1, 1)
    x = torch.randn(1, 256, 17, 17)

    alpha_w = torch.max(w.abs()) / 15
    alpha_x = torch.max(x.abs()) / 63

    w = torch.clamp(torch.round(w/alpha_w), -15, 15).float()
    x = torch.clamp(torch.round(x/alpha_x), -63, 63).float()

    out_GT = F.conv2d(x, w, stride=1, padding=1, groups=1)

    w = w.cpu().numpy()
    x = x.cpu().numpy()

    w = w.astype(np.float32)
    x = x.astype(np.float32)

    conv_param = dict()
    conv_param['padding'] = 1
    conv_param['stride'] = 1
    conv_param['groups'] = 1
    conv_param['act_signed'] = True
    conv_param['act_bitwidth'] = 7
    conv_param['weight_bitwidth'] = 5

    out = conv_forward_naive(x, w, conv_param)

    out = torch.from_numpy(out).cuda()

    print('diff : ', torch.max((out.float()-out_GT.cuda()).abs()))

def test_conv3x3_depthwise_int():
    scaling_factor = 0.05
    scaling_factor = 0.01

    w = torch.randn(256, 1, 3, 3)
    x = torch.randn(1, 256, 17, 17)

    alpha_w = torch.max(w.abs()) / 15
    alpha_x = torch.max(x.abs()) / 63

    w = torch.clamp(torch.round(w/alpha_w), -15, 15).float()
    x = torch.clamp(torch.round(x/alpha_x), -63, 63).float()

    out_GT = F.conv2d(x, w, stride=1, padding=0, groups=256)

    w = w.cpu().numpy()
    x = x.cpu().numpy()

    w = w.astype(np.float32)
    x = x.astype(np.float32)

    conv_param = dict()
    conv_param['padding'] = 0
    conv_param['stride'] = 1
    conv_param['groups'] = 256
    conv_param['act_signed'] = True
    conv_param['act_bitwidth'] = 7
    conv_param['weight_bitwidth'] = 5

    out = conv_forward_naive(x, w, conv_param)

    out = torch.from_numpy(out).cuda()

    print('diff : ', torch.max((out.float()-out_GT.cuda()).abs()))


test_conv3x3_depthwise_int()
test_conv1x1_int()
test_conv3x3_int()
test_conv3x3_uint()
