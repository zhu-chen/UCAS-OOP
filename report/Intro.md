# Pytorch简析

这是笔者于中国科学院大学2024秋季学期《面向对象程序设计》课程的课程报告，主要针对Pytorch中的torch.nn模块进行简析。

# Pytorch简介

## 什么是Pytorch?

PyTorch是一个开源的Python机器学习库，基于Torch库，底层由C++实现，应用于人工智能领域，如计算机视觉和自然语言处理。

## Pytorch的特点

1. 动态计算图：PyTorch采用动态计算图（Dynamic Computational Graph），即在每次前向传播时都会动态创建计算图。这使得调试和开发更加灵活和直观，因为可以在运行时修改计算图。

2. 强大的GPU加速

3. 丰富的库和工具和易于使用的API

同时，Pytorch可分为前后端两个部分，前端是与用户直接交互的python API，后端是框架内部实现的部分，包括Autograd，部分核心的算法使用C++实现。本文主要关注Pytorch中用python实现的部分。

# pytorch.nn模块简析

## 什么是torch.nn模块

`torch.nn`是 PyTorch 中专门用于构建和训练神经网络的模块,提供了一系列用于定义和操作神经网络的类和函数，使得构建复杂的神经网络变得更加简便和高效。

如果我们不使用`torch.nn`模块,手动搭建神经网络，我们需要自己定义每一层的激活函数、损失函数、运算过程等

```python

#定义激活函数
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)   
#定义神经网络运算过程，其中@表示点乘
def model(xb):
    return log_softmax(xb @ weights + bias)
#定义代价函数
def nll(input, target):
    return -input[range(target.shape[0]), target].mean()
loss_func = nll
```

而使用`torch.nn.functional`模块，我们就可以直接调用现成的函数，如下所示：

```python
import torch.nn.functional as F
loss_func = F.cross_entropy
def model(xb):
    return xb @ weights + bias
```
