# Train Page


* Prune the convolution layers of student net. We find some layers are more sensitive to pruning, thus these layers are pruned with low sparsity.
* Prune the fc layers.
* Conduct activation quantization for all layers. We use 7-bit activation quantization (first layer 8-bit). Then finetune the quantized model.
* Conduct weight quantization for all layers. Large layer are quantized with lower bitwidth. Then finetune the quantized model.
