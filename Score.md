# Score Page

### Model description :
We first prune the full-precision, and then quantize weights and activations to low-bit representation according to the sensitivity.
The first layer is already UINT8.

### How to test score ?

```shell
CUDA_VISIBLE_DEVICES=0 python score.py --pretrained pretrained/mixnet_s_prune_quan_final.pth --scales pretrained/scales.npy --adder tree
```

### Args :
* --pretrained: `string`, the path to merged pruned quantized model
* --scales: `string`, the path to activation quantization scales (.npy file)
* --adder : (tree | int | fp16 | fp32) ,

  * tree: Using adder tree in convolution accumulation and FP16 in all the other operations.
  * int: Using fixed-length integers for accumulation. Bit-width for accumulator is selected to make assure no overflow. Other operations are conducted using FP16.
  * int16: Based on `int`, if bit-width > 16, then FP16 for accumulation.
  * fp16: Using FP16 for accumulation in convolution and inner-product layers.
  * fp32: Using FP32 for accumulation in convolution and inner-product layers.

### Our scores : 
adder-type | op score | param score | final score
------------ | ------------- | ----------- | -----------------
tree | **0.075749** | **0.049394** | **0.125144**
int | 0.092668 | 0.049394 | 0.142063
int16 | 0.087681 | 0.049394 | 0.137076
fp16 | 0.088973 | 0.049394 | 0.138368
fp32 | 0.139955 | 0.049394 | 0.189349
