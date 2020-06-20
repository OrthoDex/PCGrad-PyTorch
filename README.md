# PCGrad-PyTorch
PyTorch implementation for "Gradient Surgery for Multi-Task Learning" https://arxiv.org/abs/2001.06782

This is currently a Work in Progress! Please see [the issues tab](https://github.com/OrthoDex/PCGrad-PyTorch/issues) for pending tasks.

For the Tensorflow implementation by the Paper authors, please see https://github.com/tianheyu927/PCGrad

Usage: 

```
"""
grad_list is a list of lists
structured as :
[
task1 gradients: [], 
task1 gradients: [], 
...
taskn gradients:[]
]
"""


grad_list = pc_grad_update(grad_list)
```

Currently validated on Multi MNIST: https://github.com/OrthoDex/MultiObjectiveOptimization/tree/feature/pc-grad
