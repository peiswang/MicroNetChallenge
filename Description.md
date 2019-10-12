# Score Description Page

We calculate the score with the same method as [ScoreDescription](https://github.com/wps712/MicroNetChallenge/blob/master/Description.md).
Except that in cifar100 track, the activations of first convolutional layer and last fc layer are not quantized, each operation in these layers is counted as 32-bit operation.
