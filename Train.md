# Claim

* The main codes are modified from [pytorch-classification](https://github.com/bearpaw/pytorch-classification)
* The codes for data augmentation are modified from [AutoAugment](https://github.com/tensorflow/models/tree/master/research/autoaugment)

# Data Augmentation
Inspired by [AutoAugment](https://arxiv.org/pdf/1805.09501.pdf), we use the similar data augmentation policy.
In AutoAugment, a policy contains multiple subpolicies, each subpolicy contains 2 tuples of 
<O, P, L>, each tuple says that: the corresponding operation O is applied 
on the input data with probability of P and 'magnitude' of L. For different operations, 'magnitude' has 
different meanings, for eaxmple, for Rotation operation, the 'magnitude' denotes the digree of rotation.
For each input, it first selects a subpolicy randomly in multiple subpolicies, each subpolicy is applied 
on the input data with corresponding probability and magnitude in sequence.

AutoAugment can observably improve the network performance, however, the policy they used on cifar100 
was searched on cifar10 dataset, which is not allowed to use in this challenge. So we use a simplified 
policy. Our policy consists of 14 tuples of <O, P, L>. For each input, we randomly select two tuples and 
apply these two operations with corresponding probability and magnitude in sequence, the policy we 
use can be seen at [transform.py](https://github.com/wps712/MicroNet/blob/cifar100/transform.py)

# Training

## train the teacher net from scratch
```
python cifar.py -a densenet --dataset cifar100 --depth 172 --growthRate 30 --train-batch 64 --epochs 300 --schedule 100 200 250 --wd 1e-4 --gamma 0.5 --lr 0.1 --checkpoint densenet-172/checkpoint --gpu-id '4,5,6,7' --no-bias-decay 'False' --lr-schedular 'WarmupCosine' --warm-up-epochs 5 --base-lr 0.001
``` 

## train the student net from scratch
```
python cifar.py -a densenet --dataset cifar100 --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 100 200 250 --wd 1e-4 --gamma 0.5 --lr 0.1 --checkpoint checkpoint/warmup_cosine/augv1 --gpu-id 0 --no-bias-decay 'False' --lr-schedular 'WarmupCosine' --loss 'CrossEntropy' --warm-up-epochs 5 --base-lr 0.001
```

## prune the weights of convolution layers, train with teacher-student
```
python train_sparse_teacher_student.py --resume-optimizer-state False --resume checkpoint/warmup_cosine/model_best.pth.tar --student-id 4 --teacher-id 4 --output-id 4 --loss kldiv:1 --checkpoint sparse/checkpoint/75/teacher_student/kldiv/1 --teacher densenet-190/checkpoint/model_best.pth.tar --rate 0.75 --lr 0.001 --warm-up-epochs 0
``` 

## quantize the activation of convolution layers, train with teacher-student
```
python train_sparse_quant_act_teacher_student.py --lr 0.001  --warm-up-epochs 0 --loss kldiv:1 --teacher densenet-190/checkpoint/model_best.pth.tar --teacher-id 0 --student-id 0 --output-id 0 --resume model_best.pth.tar --checkpoint sparse_quant/a4 --update-stop-iter 0 --resume-optimizer-state False --act-bitwidth 4
```

## quantize the weight of convolution layers, train with teacher-student
```
python train_sparse_quant_act_quant_weight_teacher_student.py --lr 0.001 --warm-up-epochs 0 --loss kldiv:1 --teacher densenet-190/checkpoint/model_best.pth.tar --teacher-id 0 --student-id 0 --output-id 0 --resume sparse_quant/a4/model_best.pth.tar.cpu --resume-optimizer-state False --checkpoint sparse_quant/a4/w4 --weight-bitwidth 4
```

## update parameters of bn layers
```
python update_bn.py --resume sparse_quant/a4/w4/model_best.pth.tar --weight-bitwidth 4
```

## prune the weight of last fc layer, train with teacher-student
```

```
