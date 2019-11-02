# Codebase

* The codes for training are mainly based on [pytorch-classification](https://github.com/bearpaw/pytorch-classification)
* The codes for data augmentation are based on [AutoAugment](https://github.com/tensorflow/models/tree/master/research/autoaugment)

# Method

## Data Augmentation
Inspired by [AutoAugment](https://arxiv.org/pdf/1805.09501.pdf), we use the similar data augmentation policy.
In AutoAugment, a policy contains multiple subpolicies, each subpolicy contains 2 tuples of 
<O, P, L>, each tuple says that: the corresponding operation O is applied 
on the input data with probability of P and 'magnitude' of L. For different operations, 'magnitude' has 
different meanings, for eaxmple, for Rotation operation, the 'magnitude' denotes the digree of rotation.
For each input, it first selects a subpolicy randomly in multiple subpolicies, each of 2 operations in 
the selected subpolicies is applied on the input data with corresponding probability and magnitude in sequence.

AutoAugment can observably improve the network performance, however, the policy they used on cifar100 
was searched on cifar10 dataset, which is not allowed to use in this challenge. So we use a simplified 
policy. Our policy consists of 14 tuples of <O, P, L>. For each input, we randomly select two tuples and 
apply these two operations with corresponding probability and magnitude in sequence, the policy we 
use can be seen at [transform.py](https://github.com/wps712/MicroNetChallenge/blob/cifar100/transform.py)

## Training schedular
Our training schedular can be decomposed into 8 steps:
* train the teacher net (densenet with depth of 172 and growthrate of 30) from scratch (300 epochs)
* train the student net (densenet with depth of 100 and growthrate of 12) from scratch (300epochs)
* prune the convolution layers of student net with sparsity of 75%, train with teacher-student (300 epochs)
* quantize the activations of convolution layers of student net to 4bit, exclude the first convolution layer, train with teacher-student (300 epochs)
* quantize the weights of convolution layers of student net to 4bit, train with teacher-student (300 epochs)
* update means and vars for BN layers, to do that, simply set the student net to train mode, and run network forward on train-set for 
  10 epochs
* prune the weight of fc layer with sparsity of 50%, train with teacher-student (50 epochs, previous layers fixed)
* quantize the weight of fc layer to 4bit, train with teacher-student (50 epochs, previous layers fixed)

## Optimizing 
### Learning rate
For step 1 and 2, we use 5 epochs to warmup the learning rate from 0.001 to 0.1, and use consine learning 
rate for the ramaining epochs. For other steps for network compression, we use cosine learning rate without 
warmup. For those steps for compression of convolution layers, we set the initial learning rate to 0.001. 
For those steps for compression of fc layer, we set the initial learning rate to 0.0001

### weight-decay and momentum
We set the weight decay to 1e-4, and momentum to 0.9

### Loss
For step 1 and 2, we use the cross entropy loss.
For those steps training with teacher-student, we use the kl divergence loss
