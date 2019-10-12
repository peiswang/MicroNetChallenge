# Test Page

## How to test accuracy ?
```shell
python test.py --resume ./model/densenet-100.pth.tar 
```


## Args :
* --resume : `string`, the path to merged pruned quantized model
* --half : `string`, can be one of 'True' or 'False', default is 'False', whether to run the network with half precision

## Our accuracy : 
|   float32   |  float16   |
|-------------|------------|
|  **80.39%** |  **80.17** |
