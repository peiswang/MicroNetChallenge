# Score Description Page

## Declaration
We assume the input of each layer in the shape of `C x H x W` where `C` is the channel number of the input feature map, 
`H` is the height of the input feature map and `W`
is the width of the input feature map. We define the param in convolution layers in the shape of `n x c x h x w` where `n` is the kernel 
number, i.e., the output channel number 
of the output feature map, `c` in the input channel number, `h` is the height of each kernel and `w` is the width of each kernel.

## Convolution and inner-product layer :
1. Op :

  * First convolution layer
  * Accumulation
    * Adder Tree
    * Fixed-length integer accumulation
    * FP32
2. Param : 
  


## Scale layer :
1. Op :
  
2. Param :
  
  
## ReLU layer :
* Op :
  * Type : FP16 (owing to `model.half()` and `x.half()`)
  * Count : [Link](https://github.com/wps712/MicroNetChallenge/blob/fa136d792419d236c28137bdea48f498dda49fad/flops_utils_final.py#L260)
* Param :
  * no param

## Pooling layer :
* Op :
  * Type : FP16 (owing to `model.half()` and `x.half()`)
  * Count : [Link](https://github.com/wps712/MicroNetChallenge/blob/fa136d792419d236c28137bdea48f498dda49fad/flops_utils_final.py#L270)
* Param :
  * no param
  
## Quantization layer :
* Op :
  * Type : FP16 (owing to `model.half()` and `x.half()`)
  * Count : [Link](https://github.com/wps712/MicroNetChallenge/blob/fa136d792419d236c28137bdea48f498dda49fad/flops_utils_final.py#L243)
* Param :
  * Type : FP16 (owing to `model.half()`)
  * If alpha equals to one, then there is no param in this quantization layer, otherwise, the param number is `1`.
  
## Sigmoid layer :
* Op :
  * Type : FP16 (owing to `model.half()` and `x.half()`)
  * Count : [Link](https://github.com/wps712/MicroNetChallenge/blob/fa136d792419d236c28137bdea48f498dda49fad/flops_utils_final.py#L311)
* Param :
  * no param
  
## Swish layer :
* Op :
  * Type : FP16 (owing to `model.half()` and `x.half()`)
  * Count : [Link](https://github.com/wps712/MicroNetChallenge/blob/fa136d792419d236c28137bdea48f498dda49fad/flops_utils_final.py#L321)
* Param :
  * no param
  
## Point Add layer :
* Op :
  * Type : FP16 (owing to `model.half()` and `x.half()`)
  * Count : [Link](https://github.com/wps712/MicroNetChallenge/blob/fa136d792419d236c28137bdea48f498dda49fad/flops_utils_final.py#L342)
* Param :
  * no param
  
## Point Dot layer :
* Op :
  * Type : FP16 (owing to `model.half()` and `x.half()`)
  * Count : [Link](https://github.com/wps712/MicroNetChallenge/blob/fa136d792419d236c28137bdea48f498dda49fad/flops_utils_final.py#L331)
* Param :
  * no param
