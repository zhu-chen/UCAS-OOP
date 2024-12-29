# 高级设计意图分析

## 组合模式

```
组合模式（Composite Pattern），也被称为部分-整体模式、合成模式或对象树，是一种结构型设计模式。这种模式将一组具有相似功能的对象视为一个单一的对象，使得客户可以以统一的方式处理单个对象和组合对象。
```

torch.nn中最为重要的类是Module类，它是所有神经网络模型的基类。这一类的设计就是一个典型的组合模式的例子。它将所有的神经网络模型视为一个整体，而这个整体又由许多的子模型组成。这样的设计使得我们可以以统一的方式处理单个模型和组合模型。

在pytorch中，神经网络模型中的一层是一个Module对象，而整个模型也是一个Module对象。这样的设计使得我们可以将一个模型的各个部分视为一个整体，而这个整体又可以作为另一个模型的一部分。这样的设计使得我们可以很方便的构建一个复杂的模型，并且可以用统一的方式处理这个模型的训练、推理等过程，而不需要关心这个模型的具体结构。

尽管在之前的报告中对Module这个类的“上帝类”状况有所吐槽，但这些主要是从面向对象的代码设计角度。但当我们实际使用时，很快就能体会到这种组合模式设计的便利之处，对于某一种层，我们都可以不用关心实现上的细节，想要了解一种模型，只需要看看它的forward方法即可。

## 观察者模式

经典的观察者模式的定义是：

```
观察者模式定义了一种一对多的依赖关系，让多个观察者对象同时监听某一个主题对象，在状态上发生变化时，会通知所有观察者对象，让它们能主动更新自己
```

pytorch实现了一种类似于观察者模式的机制，即hook机制。hook意为钩子，在实际应用中，hook机制允许用户在不修改原始代码的情况下，在特定位置插入这样的“钩子”，以实现自定义的功能。

```
hook机制允许用户在特定事件发生时插入自定义代码。这种机制提供了极大的灵活性，使用户可以在不修改原始代码的情况下扩展或修改模型的行为。
```

尽管这与经典的观察者模式有所不同，但如果我们将“特定事件发生”视为一种“状态发生变化”，那么可以将这hook机制视为观察者模式的一种实现。

同时，由于pytorch本质是一个库，观察者模式中的观察者对象，具体的实现实际上是由用户自己定义，包括同步/异步等属于经典观察者模式的特性，pytorch并没有做出实现，而是将这一部分交给了用户。因而hook机制与典型的观察者模式有一些微小的区别

下面是使用hook机制的一个例子：

```python
def forward_hook(module, input, output):
    ……#do something
model = MyModule()
hook_handle = model.register_forward_hook(forward_hook)
# Forward pass
x = torch.randn(1, 10)
output = model(x)
# Remove the hook
hook_handle.remove()
```

在这个例子中，我们定义了一个forward_hook函数，然后将通过调用Module类中register_forward_hook方法这个函数注册到model的forward_hook中。这样，当model进行前向传播时，forward_hook函数就会被调用。这样的设计使得我们可以在模型的前向传播过程中插入自定义的代码，而不需要修改模型的定义。

在pytorch中，对hook机制的实现如下（以forward_hook为例）：

Module类中的register_forward_hook方法实现了hook的注册：

```python
def register_forward_hook(
        self,
        hook: Union[
            Callable[[T, Tuple[Any, ...], Any], Optional[Any]],
            Callable[[T, Tuple[Any, ...], Dict[str, Any], Any], Optional[Any]],
        ],
        *,
        ……    ) -> RemovableHandle:
        handle = RemovableHandle(……)
        self._forward_hooks[handle.id] = hook
        if with_kwargs:
            self._forward_hooks_with_kwargs[handle.id] = True
        if always_call:
            self._forward_hooks_always_called[handle.id] = True
        if prepend:
            self._forward_hooks.move_to_end(handle.id, last=False)  # type: ignore[attr-defined]
        return handle
```

当我们调用model.register_forward_hook进行注册时，实际上是将hook函数添加到了model的_forward_hooks中，并返回了一个RemovableHandle对象。这个对象可以用来在需要的时候移除这个hook函数。

在实际进行前向传播时,pytorch会先判定是否有hook函数需要调用，如果有的话，就会调用这个hook函数。

```python
    def _call_impl(self, *args, **kwargs):
        forward_call = (self._slow_forward if torch._C._get_tracing_state() else self.forward)
        # If we don't have any hooks, we want to skip the rest of the logic in
        # this function, and just call forward.
        if not (self._backward_hooks or self._backward_pre_hooks…….):
            return forward_call(*args, **kwargs)        result = None
        called_always_called_hooks = set()
        def inner():
            nonlocal result, args, kwargs
            ……            
            if _global_forward_pre_hooks or self._forward_pre_hooks:
                for hook_id, hook in (
                    *_global_forward_pre_hooks.items(),
                    *self._forward_pre_hooks.items(),
                ):
                ……#遍历hookid并执行
```

