# Test Page

## How to test accuracy ?
```shell
python test.py --resume ./model/densenet-100.pth.tar --data-root ../../cifar
```

## Args :
* --resume : `string`, the path to merged pruned quantized model
* --half : `string`, can be either 'True' or 'False', default is 'False', whether to run the network with half precision
* --data-root : `string`, the root directory of cifar100 dataset. Please insure that you have already downloaded the cifar100
                dataset and placed in the data-root directory. If not, please download it with torchvision.

## Our accuracy : 
|   float32   |  float16    |
|-------------|-------------|
|  **80.39%** |  **80.17%** |

If you cannot get this accuracy with our model, please check your 
torchvision version. Because the definition of ToTensor transform of 
older version of torchvision is different from that of newer version 
of torchvision.
