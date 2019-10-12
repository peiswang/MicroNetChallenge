# Test Page

### How to test accuracy ?

```shell
CUDA_VISIBLE_DEVICES=0 python test.py --pretrained pretrained/mixnet_s_prune_quan_final.pth --scales pretrained/scales.npy
```

### Args :
* --pretrained: `string`, the path to merged pruned quantized model
* --scales: `string`, the path to activation quantization scales (.npy file)
* others : as default

### Our accuracy : 
Top-1 : **75.034%**
Top-5 : **92.218%**
