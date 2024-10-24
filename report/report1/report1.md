# 功能分析与建模

## 神经网络简介

写在最前面:本文**不是**人工智能相关课程的作业，不会对神经网络的原理进行详细的介绍。相关介绍只是为了说明想要构建一个神经网络要做些什么，并借此说明torch.nn模块的功能。不过，即使不了解神经网络的原理，也应该不会对后面的内容产生太大的困扰。

概括的说，如果想要实现一个神经网络，需要做以下几件事:

1. 定义神经网络的结构，即神经网络的层数、每一层的神经元个数等。这些层包括输入层、隐藏层和输出层。

2. 定义训练过程，包括前向传播、损失函数、反向传播和参数更新等。

这些足够进行接下来的说明。

## 主要功能

### 一个简单的例子

既然与神经网络有关，那么torch.nn模块的最主要的功能，就是构建出一个神经网络模型。

下面是一个简单的例子，展示了如何使用torch.nn模块构建一个简单的神经网络模型并进行训练。

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的前馈神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        # 调用父类的构造函数
        super(SimpleNN, self).__init__()
        # 定义神经网络的结构
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 利用定义好的神经网络结构,定义前向传播过程
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 创建模型实例
model = SimpleNN()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters(), lr=0.01) #不属于torch.nn模块

# 示例输入
input_data = _________ # 784维向量
output = model(input_data) # 前向传播
loss = criterion(output, torch.tensor([1])) 

# 反向传播和优化
loss.backward()
optimizer.step()
```

当然，遗憾的是，这个例子本身是过于简单和不完整的，同时也没有数据集，无法实际进行训练。而要想找一个完整的例子，所需要的代码量过于庞大，不适合放在这里。不过，这个例子足以说明torch.nn模块的主要功能。

## 需求建模

根据以上的例子，我们可以建立需求模型

```
[用例名称]

构建一个简单的神经网络模型

[场景]

Who:调用者，被构建出的模型，模型的各层，激活函数，损失函数，优化器(不属于torch.nn模块,忽略)
Where:内存
When:训练时(运行时)

[用例描述]

1. 调用者创建一个神经网络模型
    1. 调用者定义模型的结构，即添加各层的信息(种类，维度)
    2. 调用者选择激活函数
2. 调用者定义损失函数和优化器
3. 调用者输入数据
4. 调用者调用模型进行前向传播，得到输出，损失函数根据输出进行计算得到损失
5. 根据损失进行反向传播，优化器更新参数

[用例价值]

得到一个训练好的神经网络模型

[约束和限制]

输入数据的维度必须与模型的输入层维度相同
```

寻找其中的动词和名词

```
名词:神经网络模型，模型的各层，激活函数，损失函数，优化器(忽略)，输入数据，输出数据,参数
动词:创建，添加，输入，前向传播，计算，反向传播，更新(忽略)
```

输入和输出实际上是一个数组，或者更专业的说法是pytorch的张量`Tensor`,不属于本次讨论的范围，也没必要抽象成类。因此我们得到应该抽象出来的类和方法:

```
[类]:神经网络模型
[方法]:创建，添加，前向传播
[属性]:结构（即各层的信息）

[类]:神经网络的层
[方法]:添加,计算(前向传播)
[属性]:种类*，维度,参数

[类]:激活函数
[方法]:计算(前向传播)
[属性]:种类*

[类]:损失函数
[方法]:计算(反向传播)
[属性]:种类*

注：对于不同的种类的层和函数，实际上是创建了不同的对象，调用了不同的方法，但考虑到本文并不需要关注它们之间的具体区别，因此将它们抽象成一个类。实际上，一般把神经网络的层与函数统称为计算功能类。
```

## 从简单用例模拟torch.nn运行过程

接下来让我们看一看torch.nn库是怎么创建并训练这一模型的。需要注意的是神经网络模型本身较为复杂，且完整过程本身不是torch.nn一个库就可以完成的，我们只关注torch.nn库中用python实现的部分。

### 网络的创建

首先，这个网络类的创建是由实例代码，即调用者自己实现的simpleNN类，它继承自`nn.Module`，并调用了父类的构造函数。

`torch.nn`是torch.nn模块中的一个基类，所有神经网络模型都应该继承自它.

```python
class Module:

    training: bool
    _parameters: Dict[str, Optional[Parameter]]
    ...# 省略了一些属性

    def __init__(self, *args, **kwargs) -> None:
        """Initialize internal Module state, shared by both nn.Module and ScriptModule."""
        torch._C._log_api_usage_once("python.nn_module")
        # 调用私有API，记录log
        
        # Backward compatibility: no args used to be allowed when call_super_init=False
        if self.call_super_init is False and bool(kwargs):
            raise TypeError(
                f"{type(self).__name__}.__init__() got an unexpected keyword argument '{next(iter(kwargs))}'"
                ""
            )

        if self.call_super_init is False and bool(args):
            raise TypeError(
                f"{type(self).__name__}.__init__() takes 1 positional argument but {len(args) + 1} were"
                " given"
            )
        # 调用call_super_init并检查参数是否正确

        #使用特殊的赋值方法以提高性能

        super().__setattr__("training", True)
        super().__setattr__("_parameters", {})
        ... # 省略了一些对属性的赋值操作


        if self.call_super_init:
            super().__init__(*args, **kwargs)
        # 调用父类的构造函数
```

`Module`类隐式继承了`object`类，与java中的`Object`类类似，是所有类的基类。因此不需要再向上溯源进行分析了。

### 定义模型的结构

接下来，模型调用了`nn.Linear`这一构造函数，定义了一个全连接层。

```python
self.fc1 = nn.Linear(784, 128)
```

该部分对应的类是`Linear`类

```python
class Linear(Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
    ...
```

`linear`类继承自`Module`类，构造函数先调用了父类的构造函数，然后根据传输的输入输出维度，创建了一个`Linear`对象，并利用`Parameter`类的构造函数初始化了`weight`和`bias`两个参数。

进入`Parameter`类

```python
class Parameter(torch.Tensor, metaclass=_ParameterMeta):

    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torch.empty(0)
        if type(data) is torch.Tensor or type(data) is Parameter:
            # For ease of BC maintenance, keep this path for standard Tensor.
            # Eventually (tm), we should change the behavior for standard Tensor to match.
            return torch.Tensor._make_subclass(cls, data, requires_grad)

        # Path for custom tensors: set a flag on the instance to indicate parameter-ness.
        t = data.detach().requires_grad_(requires_grad)
        if type(t) is not type(data):
            raise RuntimeError(
                ...# 省略了一些错误处理
            )
        t._is_param = True
        return t
    ...
    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)
    ...
```

`Parameter`类继承自`torch.Tensor`类，并使用`_ParameterMeta`作为元类。更具体的分析将在后续的报告中进行。


在之后的前向传播过程中，调用了位于`torch.nn.functional`中的`linear`函数，这一方法的实现如下

```python
    linear = _add_docstr(
    torch._C._nn.linear,
    r"""
linear(input, weight, bias=None) -> Tensor

Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

This operation supports 2-D :attr:`weight` with :ref:`sparse layout<sparse-docs>`

{sparse_beta_warning}

This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.

Shape:

    - Input: :math:`(*, in\_features)` where `*` means any number of
      additional dimensions, including none
    - Weight: :math:`(out\_features, in\_features)` or :math:`(in\_features)`
    - Bias: :math:`(out\_features)` or :math:`()`
    - Output: :math:`(*, out\_features)` or :math:`(*)`, based on the shape of the weight
""".format(
        **sparse_support_notes
    ),
)
```

可以看到，处于性能上的考虑，`linear`函数实际上是调用了`torch._C._nn.linear`这一C++函数，这一函数的具体实现不在本文的讨论范围内。值得注意的是，这一方法并不属于任何类，而是直接定义在`torch.nn.functional`模块中。对此具体的思考将在后续的报告中进行。

其他例如`ReLU`激活函数等计算功能类的实现也是类似的，都是继承自`module`这一基类，核心的方法都为构造函数和一个`forward`(或`backward`)方法，一个构造函数不再赘述。事实上神经网络的隐层种类还有很多，比如卷积层、池化层等，全部说明是不现实的。不过，这些层之间的区别主要在于具体在一个神经网络中功能的区别，而这些我们并不需要关心。

至此，对实例代码执行流程的分析已经结束。

## 总结

完成了这一流程后，我最大的感受是Pytorch作为一个高性能的深度学习框架，对性能的追求是非常明显的。无论是基类`Module`中特殊的赋值方法，还是`torch.nn.functional`中直接调用C++函数，都体现了对性能的极致追求。然而，这也在一定程度上增加了代码的复杂度，使得代码的阅读和理解变得更加困难。

同时，仅从面向对象的角度来看，Pytorch的设计很难称得上是顶尖，基类`module`有接近上百种方法，几乎是个上帝类。

另外，对于大量的计算功能类，其唯一的区别就是具体的计算方法不同，激活函数相关的类就是这一现象的重灾区。比如下面这些类：

```python
class ReLU(Module):
    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str
```

```python
class SiLU(Module):
    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.silu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str
```

```python
class SELU(Module):
    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.selu(input, self.inplace)

    def extra_repr(self) -> str:
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str
```

相似的的XXLU不止列出的这几个，这些类完全可以通过一个类来实现，通过传入不同的参数来调用不同的计算方法，或者抽象出一个类，让这些类继承自它。这样的设计不仅可以减少代码量，也可以减少代码的复杂度，提高代码的可读性。