# Score Page

### Model description :
We first prune the full-precision model to 75% (in total) sparsity, and quantize weights and activations to 4-bit.
Then, we prune 50% weights in the fully-connected layer and quantize the remaining weights in FC to 6-bit.
We still keep the input activations of the first and the last layer in float32. 

### How to test score ?

```shell
python score.py --resume ./model/densenet-100.pth.tar --score-type 0
```

### Args :
* --resume : `string`, the path to merged pruned quantized model
* --score-type :  `int`,

  * 0: Using adder tree in convolution accumulation and FP16 in all the other operations
  * 1: Using [fixed-length](https://github.com/wps712/MicroNetChallenge/blob/e858966bd82a150abec52033237a9c48c97fbb62/flops_utils.py#L98) (without overflow) integers to accumulate convolution intermediate results and FP16 in all the other operations
  * 2: Using FP32 to accumulate intermediate results in convolution, average pooling and bias term, and FP16 in all the other operations
  * 3: Using [fixed-length](https://github.com/wps712/MicroNetChallenge/blob/e858966bd82a150abec52033237a9c48c97fbb62/flops_utils.py#L98) (without overflow) integers to accumulate convolution intermediate results; FP32 to accumulate intermediate results in average pooling and bias term; FP16 in all the other operations
  * 4: Using adder tree in convolution accumulation; FP32 to accumulate intermediate results in average pooling and bias term; FP16 in all the other operations
  * 5: Using adder tree in convolution accumulation; FP32 to accumulate intermediate results in average pooling; FP16 in all the other operations
  * 6: Using [fixed-length](https://github.com/wps712/MicroNetChallenge/blob/e858966bd82a150abec52033237a9c48c97fbb62/flops_utils.py#L98) (without overflow) integers to accumulate convolution intermediate results; FP32 to accumulate intermediate results in average pooling; FP16 in all the other operations

### Our scores : 
score-type | op score | param score | final score
------------ | ------------- | ----------- | -----------------
0 | **0.002805** | **0.001365** | **0.004169**
1 | 0.003869 | 0.001365 | 0.005234
2 | 0.007179 | 0.001365 | 0.008544
3 | 0.004075 | 0.001365 | 0.005440
4 | 0.003010 | 0.001365 | 0.004375
5 | 0.002807 | 0.001365 | 0.004171
6 | 0.003871 | 0.001365 | 0.005236
