# 1 torch.autograd.Function 基本逻辑

当你调用自定义 torch.autograd.Function 的 apply 方法（前向传播）时，autograd.Function 会通过 _backward_cls **生成反向传播类**，并构建计算图节点（PyNode）。

反向传播时（调用 loss.backward()）：
先执行 _backward_cls 中的核心梯度计算逻辑（即用户定义的 backward 方法）；
计算完梯度后，会遍历 Function 实例中 _backward_hooks 容器（由 _HookMixin 维护）中的所有钩子函数；
每个钩子函数会接收梯度相关参数（如 grad_outputs 输出梯度、grad_inputs 输入梯度），允许修改梯度后返回；
最终返回修改后的梯度，继续向上游传播。

## 1.1 torch.autograd.Function

```python
class Function(_SingleLevelFunction):
    r"""Base class to create custom `autograd.Function`.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            f"{self.__class__} should not be instantiated. Methods on autograd functions"
            "are all static, so you should invoke them on the class itself. "
            "Instantiating an autograd function will raise an "
            "error in a future version of PyTorch.",
            DeprecationWarning,
            stacklevel=2,
        )

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "Legacy autograd function with non-static forward method is deprecated. "
            "Please use new-style autograd function with static forward method. "
            "(Example: https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function)"
        )

    """
    Bool that specifies if PyTorch should attempt to autogenerate
    :func:`torch.vmap` support for this autograd.Function.
    """
    generate_vmap_rule = False

    @staticmethod
    def vmap(info, in_dims, *args):
        r"""Define the behavior for this autograd.Function underneath :func:`torch.vmap`.
        """
        raise NotImplementedError(
            "To use autograd.Function with vmap, you must either override the "
            "vmap staticmethod or set generate_vmap_rule=True."
        )

    @classmethod
    def apply(cls, *args, **kwargs):
        def bind_default_args(func, *args, **kwargs):
            signature = inspect.signature(func)
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()

            return bound_args.args

        is_setup_ctx_defined = _is_setup_context_defined(cls.setup_context)
        if is_setup_ctx_defined:
            args = bind_default_args(cls.forward, *args, **kwargs)

        if not torch._C._are_functorch_transforms_active():
            # See NOTE: [functorch vjp and autograd interaction]
            args = _functorch.utils.unwrap_dead_wrappers(args)
            return super().apply(*args, **kwargs)  # type: ignore[misc]

        if not is_setup_ctx_defined:
            raise RuntimeError(
                "In order to use an autograd.Function with functorch transforms "
                "(vmap, grad, jvp, jacrev, ...), it must override the setup_context "
                "staticmethod. For more details, please see "
                "https://pytorch.org/docs/main/notes/extending.func.html"
            )

        return custom_function_call(cls, *args, **kwargs)

    @staticmethod
    def _compiled_autograd_key(ctx):
        return (ctx._autograd_function_id,)
```

## 1.2 torch.autograd._SingleLevelFunction

```python
class _SingleLevelFunction(
    _C._FunctionBase, FunctionCtx, _HookMixin, metaclass=FunctionMeta
):
    @staticmethod
    def forward(*args: Any, **kwargs: Any) -> Any:
        r"""Define the forward of the custom autograd Function.
        """
        raise NotImplementedError(
            "You must implement the forward function for custom autograd.Function."
        )

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> Any:
        r"""There are two ways to define the forward pass of an autograd.Function.

        Either:

        1. Override forward with the signature ``forward(ctx, *args, **kwargs)``.
           ``setup_context`` is not overridden. Setting up the ctx for backward
           happens inside the ``forward``.
        2. Override forward with the signature ``forward(*args, **kwargs)`` and
           override ``setup_context``. Setting up the ctx for backward happens
           inside ``setup_context`` (as opposed to inside the ``forward``)

        See :meth:`torch.autograd.Function.forward` and :ref:`extending-autograd` for more details.
        """
        raise NotImplementedError("setup_context is not implemented.")

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        r"""Define a formula for differentiating the operation with backward mode automatic differentiation.

        """
        raise NotImplementedError(
            "You must implement either the backward or vjp method for "
            "your custom autograd.Function to use it with backward "
            "mode AD."
        )

    # vjp and backward are alias of each other
    vjp = backward

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        r"""Define a formula for differentiating the operation with forward mode automatic differentiation.
        """
        raise NotImplementedError(
            "You must implement the jvp function for custom "
            "autograd.Function to use it with forward mode AD."
        )
```

## 1.3 FunctionMeta 子类 --> 创建 Backward 类 : _backward_cls

Function的元类（metaclass），用于控制torch.autograd.Function子类的创建过程, 使得用户在自定义
torch.autograd.Function子类时，自动为子类添加**_backward_cls属性**，使得子类可以自动完成反向传播相关的初始化工作。

**_backward_cls** 是一个类，对应原函数的微分版本(反向传播使用)，该类继承自 BackwardCFunction, 与原前向传播类强关联，是autograd反向计算的核心载体。


```python
class FunctionMeta(type):
    """Function的元类（metaclass），用于控制autograd.Function子类的创建过程。

    元类的核心作用是在定义自定义autograd.Function子类时，自动完成反向传播相关的初始化工作，
    无需用户手动编写反向传播类的框架代码。

    自动为目标类（autograd.Function子类）添加以下关键属性：
        _backward_cls: 动态生成的反向传播类（对应原函数的微分版本），
                      继承自BackwardCFunction，与原前向传播类强关联，
                      是autograd反向计算的核心载体。
    """

    def __init__(cls, name, bases, attrs):
        """元类的构造方法，在定义autograd.Function子类时自动调用。

        参数说明：
            cls: 正在被创建的autograd.Function子类（目标类本身）
            name: 目标类的类名（字符串）
            bases: 目标类的父类元组（通常包含autograd.Function）
            attrs: 目标类的属性字典（包含类变量、方法等定义）
        """
        # 1. 动态生成反向传播类（Backward Class）
        # 类名规则：原类名 + "Backward"（例如原类是MyFunction，反向类就是MyFunctionBackward）
        # 父类：BackwardCFunction（PyTorch底层提供的反向传播基类，封装了autograd核心逻辑）
        # 类属性：_forward_cls = cls → 建立反向类与原前向类的双向关联，便于反向传播时获取前向信息
        backward_fn = type(
            name + "Backward",  # 动态生成的反向类名称
            (BackwardCFunction,),  # 继承自底层反向传播基类
            {"_forward_cls": cls}  # 存储原前向类的引用，供反向计算时使用
        )

        # 2. 为反向类分配唯一的autograd函数ID
        # AUTOGRAD_FUNCTION_COUNTER是全局计数器，通过next()获取下一个唯一ID，
        # 用于PyTorch内部区分不同的autograd函数，确保计算图追踪和梯度传播的正确性
        backward_fn._autograd_function_id = next(AUTOGRAD_FUNCTION_COUNTER)  # type: ignore[attr-defined]

        # 3. 初始化反向类的延迟加载模块属性（默认为None）
        # _bw_module用于存储反向传播逻辑的延迟加载模块（针对惰性加载场景，如动态导入反向逻辑）
        backward_fn._bw_module = None  # type: ignore[attr-defined]

        # 4. 若原前向类定义了_lazy_backward_info（延迟反向传播信息），则传递给反向类
        # _lazy_backward_info通常包含反向传播模块的路径、名称等信息，
        # 此处将其bw_module属性赋值给反向类，支持反向逻辑的延迟加载（避免初始化时冗余导入）
        if getattr(cls, "_lazy_backward_info", None):
            backward_fn._bw_module = cls._lazy_backward_info.bw_module  # type: ignore[attr-defined]

        # 5. 将动态生成的反向类绑定到原前向类的_backward_cls属性
        # 这样用户定义的autograd.Function子类就可以通过self._backward_cls访问反向类，
        # PyTorch内部在构建计算图时也会通过该属性关联前向-反向逻辑
        cls._backward_cls = backward_fn

        # 6. 调用父类（type）的构造方法，完成目标类的常规初始化
        # 确保元类不会阻断Python类创建的默认流程
        super().__init__(name, bases, attrs)
```

- 注释：
> type 是 python 中的元类(metaclass)，负责创建类:
> 类本身是 type 的实例，但类对象并不是type 的实例；
> 而object 是所有类的基类，所有类都直接或间接继承自object.

- cls._backward_cls 都继承自 BackwardCFunction, 其内部的 apply 方法会调用Function.backward 函数<br>
- 从前向获取backward 并执行：backward_fn = self._forward_cls.backward<br>

```python
class BackwardCFunction(_C._FunctionBase, FunctionCtx, _HookMixin):
    r"""
    This class is used for internal autograd work. Do not use.
    """

    def apply(self, *args):
        r"""
        Apply method used when executing this Node during the backward
        """
        # _forward_cls is defined by derived class
        # The user should define either backward or vjp but never both.
        backward_fn = self._forward_cls.backward  # type: ignore[attr-defined]
        vjp_fn = self._forward_cls.vjp  # type: ignore[attr-defined]
        if backward_fn is not Function.backward and vjp_fn is not Function.vjp:
            raise RuntimeError(
                "Implementing both 'backward' and 'vjp' for a custom "
                "Function is not allowed. You should only implement one "
                "of them."
            )
        user_fn = vjp_fn if vjp_fn is not Function.vjp else backward_fn
        return user_fn(self, *args)

    def apply_jvp(self, *args):
        r"""
        Apply method used when executing forward mode AD during the forward
        """
        # _forward_cls is defined by derived class
        return self._forward_cls.jvp(self, *args)  # type: ignore[attr-defined]

    def _compiled_autograd_key(self):
        return self._forward_cls._compiled_autograd_key(self)  # type: ignore[attr-defined]
```

- **注意 _backward_cls 类继承了 _C._FunctionBase, 而 FunctionBase c++ 属性来自THPFunction_properties， 里面大部分都从 c++ 的 THPFunction 类获取，THPFunction 也就是 ctx 的来源。从_backward_cls 中通过get 就获取了ctx 类。**

```c++
static struct PyGetSetDef THPFunction_properties[] = {
    {"saved_tensors",
     (getter)THPFunction_saved_tensors,
     nullptr,
     nullptr,
     nullptr},
    {"saved_variables",
     (getter)THPFunction_saved_variables,
     nullptr,
     nullptr,
     nullptr},
    {"_raw_saved_tensors",
     (getter)THPFunction_raw_saved_tensors,
     nullptr,
     nullptr,
     nullptr},
    {"next_functions",
     (getter)THPFunction_next_functions,
     nullptr,
     nullptr,
     nullptr},
    {"to_save",
     &getObject<&THPFunction::to_save>,
     &setObject<&THPFunction::to_save>,
     nullptr,
     nullptr},
    {"non_differentiable",
     &getObject<&THPFunction::non_differentiable>,
     &setObject<&THPFunction::non_differentiable>,
     nullptr,
     nullptr},
    {"dirty_tensors",
     &getObject<&THPFunction::dirty_tensors>,
     &setObject<&THPFunction::dirty_tensors>,
     nullptr,
     nullptr},
    {"saved_for_forward",
     &getObject<&THPFunction::saved_for_forward>,
     &setObject<&THPFunction::saved_for_forward>,
     nullptr,
     nullptr},
    {"needs_input_grad",
     &getObject<&THPFunction::needs_input_grad>,
     &setObject<&THPFunction::needs_input_grad>,
     nullptr,
     nullptr},
    {"requires_grad", getRequiresGrad, nullptr, nullptr, nullptr},
    {"metadata", (getter)THPFunction_metadata, nullptr, nullptr, nullptr},
    {"_input_metadata",
     (getter)THPFunction_input_metadata,
     nullptr,
     nullptr,
     nullptr},
    {"materialize_grads",
     nullptr,
     (setter)THPFunction_set_materialize_grads,
     nullptr,
     nullptr},
    {"_materialize_non_diff_grads",
     (getter)THPFunction_get_materialize_non_diff_grads,
     (setter)THPFunction_set_materialize_non_diff_grads,
     nullptr,
     nullptr},
    {"_compiled_autograd_backward_state",
     (getter)THPFunction_get_compiled_autograd_backward_state,
     (setter)THPFunction_set_compiled_autograd_backward_state,
     nullptr,
     nullptr},
    {nullptr}};
```



## 1.4 context python 和 c++ 中分别的定义

- python 中

```python
class FunctionCtx:
    def save_for_backward(self, *tensors: torch.Tensor):
        r"""Save given tensors for a future call to :func:`~Function.backward`.

        ``save_for_backward`` should be called at most once, in either the
        :func:`setup_context` or :func:`forward` methods, and only with tensors.
        """
        # 张量还没有真正被"保存"，而是暂存在 to_save 属性中等待处理
        # 处理后的tensor 才是 self.saved_tensors
        self.to_save = tensors

    def save_for_forward(self, *tensors: torch.Tensor):
        r"""Save given tensors for a future call to :func:`~Function.jvp`.

        ``save_for_forward`` should be called at most once, in either the
        :func:`setup_context` or :func:`forward` methods, and all arguments
        should be tensors.

        In :func:`jvp`, saved objects can be accessed through the :attr:`saved_tensors`
        attribute.
        """
        for tensor in tensors:
            assert isinstance(tensor, torch.Tensor) or tensor is None, (
                "save_for_forward expects all arguments to be tensors; you should "
                "save non-tensors as attributes on ctx."
            )

        self.saved_for_forward = tensors

    def mark_dirty(self, *args: torch.Tensor):
        r"""Mark given tensors as modified in an in-place operation.

        This should be called at most once, in either the :func:`setup_context`
        or :func:`forward` methods, and all arguments should be inputs.

        Every tensor that's been modified in-place in a call to :func:`forward`
        should be given to this function, to ensure correctness of our checks.
        It doesn't matter whether the function is called before or after
        modification.
        """
        self.dirty_tensors = args

    @deprecated(
        "`mark_shared_storage` is deprecated. "
        "Tensors with shared storages are automatically tracked. "
        "Note that calls to `set_()` are not tracked",
        category=FutureWarning,
    )
    def mark_shared_storage(self, *pairs):
        pass

    def mark_non_differentiable(self, *args: torch.Tensor):
        r"""Mark outputs as non-differentiable.

        This should be called at most once, in either the :func:`setup_context`
        or :func:`forward` methods, and all arguments should be tensor outputs.

        This will mark outputs as not requiring gradients, increasing the
        efficiency of backward computation. You still need to accept a gradient
        for each output in :meth:`~Function.backward`, but it's always going to
        be a zero tensor with the same shape as the shape of a corresponding
        output.
        """
        self.non_differentiable = args

    def set_materialize_grads(self, value: bool):
        r"""Set whether to materialize grad tensors. Default is ``True``.

        This should be called only from either the :func:`setup_context` or
        :func:`forward` methods.

        If ``True``, undefined grad tensors will be expanded to tensors full of zeros
        prior to calling the :func:`backward` and :func:`jvp` methods.
        """
        self.materialize_grads = value
```

- c++ 中

```c++
struct THPFunction {
  PyObject_HEAD

  PyObject* needs_input_grad;

  // Python tuple of tensors whose variables we should save.  Set
  // by Python with 'save_for_backward'.  If nullptr, no tensors were
  // saved.
  PyObject* to_save;
  // Python tuple of tensors which are not differentiable.  Set by
  // Python with 'mark_non_differentiable'.  If nullptr, no tensors were
  // non-differentiable.
  PyObject* non_differentiable;
  // Python tuple of tensors which had inplace updates in the forward()
  // pass.  Set by Python with 'mark_dirty'.  If nullptr, no tensors were
  // modified inplace.
  PyObject* dirty_tensors;

  // boolean indicating whether to materialize undefined output grad tensors
  // into tensors full of zeros. Set by Python with 'set_materialize_grads'.
  // Default is true.
  bool materialize_grads;

  // boolean indicating whether to materialize output grad tensors
  // corresponding to non-differentiable outputs. Normally, someone would
  // already get this behavior by switching off materialize_grads,
  // but there are certain use cases where that is not feasible:
  // https://github.com/pytorch/pytorch/pull/98659#pullrequestreview-1376822560
  bool materialize_non_diff_grads;

  PyObject* compiled_autograd_backward_state;
  std::vector<c10::SymInt> compiled_autograd_symints;

  std::vector<torch::autograd::VariableInfo> output_info;
  std::vector<torch::autograd::VariableInfo> input_info;
  std::vector<torch::autograd::SavedVariable> saved_variables;
  // For each input, true if the input is a THPVariable
  std::vector<bool> is_variable_input;
  char has_freed_buffers;

  PyObject* saved_for_forward;
  // The actual PyNode (in the autograd graph) that this data was
  // saved for.  This field may be NULL (because a user can construct
  // a THPFunction directly from Python), but when this field is non-NULL,
  // it is guaranteed that cdata.lock()->obj == this
  //
  // In most ordinary use, this field should always be non-NULL; e.g.,
  // when we allocate a THPFunction because we are running Node.apply,
  // after constructing a THPFunction, we immediately allocate a PyNode
  // for it.  We can't enforce this directly in the constructor of
  // THPFunction though, because there's no way to keep it live long enough
  // to save an owning reference to PyNode into the grad_fn of a Variable.
  std::weak_ptr<torch::autograd::PyNode> cdata;
};
```

## 1.5 _HookMixin ： 梯度钩子函数的管理

```python
class _HookMixin:
    """反向传播钩子（backward hook）的复用混合类（Mixin Class）。

    混合类（Mixin）的设计目的：不单独实例化，而是被其他类继承，
    提供通用的“反向钩子注册逻辑”，避免代码重复（比如autograd.Function相关类会继承该Mixin）。

    核心功能：标准化反向钩子的注册流程，包括钩子容器初始化、可移除句柄创建、钩子存储，
    同时支持后续通过句柄安全移除钩子，确保反向传播过程中钩子的有序执行和资源释放。
    """

    @staticmethod
    def _register_hook(backward_hooks, hook):
        """注册反向传播钩子函数到钩子容器中。

        反向钩子的作用：在autograd反向传播过程中（计算梯度时）插入自定义逻辑，
        比如修改梯度、打印梯度信息、记录梯度数据等。

        参数说明：
            backward_hooks: 存储反向钩子的容器（OrderedDict类型或None）
                - 若为None，会自动初始化一个OrderedDict（保持钩子注册顺序，确保执行顺序可预测）
                - 若已存在，则直接复用该容器追加新钩子
            hook: 要注册的反向钩子函数（需符合PyTorch反向钩子的签名要求，
                  通常接收grad_outputs等参数，返回修改后的梯度）

        返回值：
            tuple: (更新后的backward_hooks容器, 钩子的可移除句柄RemovableHandle)
                - 更新后的容器：包含新注册的钩子
                - 可移除句柄：用于后续通过handle.remove()安全删除该钩子，避免内存泄漏或无效钩子残留
        """
        # 若钩子容器未初始化（为None），则创建OrderedDict作为容器
        # 选择OrderedDict的原因：保持钩子的“注册顺序”，确保反向传播时钩子按注册顺序执行
        if backward_hooks is None:
            backward_hooks = OrderedDict()

        # 创建钩子的“可移除句柄”：RemovableHandle是PyTorch提供的钩子管理工具
        # 句柄会关联到钩子容器backward_hooks，后续通过handle.remove()可直接删除对应的钩子
        # 句柄的id属性是唯一标识，用于在容器中定位钩子
        handle = hooks.RemovableHandle(backward_hooks)

        # 将钩子函数存入容器：key为句柄的唯一id，value为钩子函数
        # 用id作为key便于通过句柄快速查找和删除对应的钩子
        backward_hooks[handle.id] = hook

        # 返回更新后的钩子容器和句柄：容器供类保存，句柄返回给用户用于移除钩子
        return backward_hooks, handle
```

- **触发时机: 通过backward 计算梯度后, example:** <br>

```python
import torch
from torch.autograd import Function

# 自定义 autograd.Function 子类（自动继承 _HookMixin 的能力）
class MyFunction(Function):
    @staticmethod
    def forward(ctx, x):
        # 前向计算：y = x²
        ctx.save_for_backward(x)  # 保存中间变量供反向使用
        return x ** 2

    @staticmethod
    def backward(ctx, grad_output):
        # 反向计算：dy/dx = 2x * grad_output
        x, = ctx.saved_tensors
        return 2 * x * grad_output

# 1. 创建自定义 Function 实例
func = MyFunction()

# 2. 注册反向钩子（底层调用 _HookMixin._register_hook）
# 钩子功能：打印输入梯度，并对梯度乘以0.5（梯度缩放）
def my_hook(grad_inputs):
    print(f"钩子触发：原始梯度 = {grad_inputs}")
    return tuple(g * 0.5 for g in grad_inputs)  # 修改梯度

handle = func.register_backward_hook(my_hook)

# 3. 前向+反向传播
x = torch.tensor([3.0], requires_grad=True)
y = func.apply(x)
y.backward()  # 触发反向传播，同时执行钩子

# 输出结果：
# 钩子触发：原始梯度 = (tensor([6.]),)  # 原始反向计算结果：2*3*1=6
# x.grad = tensor([3.])  # 钩子修改后的梯度：6 * 0.5 = 3

# 4. 移除钩子（通过 _HookMixin 返回的句柄）
handle.remove()

# 再次反向传播（钩子不再执行）
x.grad.zero_()
y = func.apply(x)
y.backward()
print(x.grad)  # 输出：tensor([6.])  # 无钩子修改，恢复原始梯度
```

## 1.6 torch._C._FunctionBase

```python
# */python3.10/dist-packages/torch/_C/__init__.pyi
# c++ Defined in torch/csrc/autograd/python_function.cpp
class _FunctionBase:
    saved_tensors: Tuple[Tensor]
    _raw_saved_tensors: Tuple[Any]
    next_functions: Tuple[Tuple[Any, _int], ...]
    needs_input_grad: Tuple[_bool]
    metadata: dict
    _materialize_non_diff_grads: _bool
    # skip adding type hints for the fields that have wrappers defined
    # in torch/autograd/function.py
```

- 核心函数

```c++
static struct PyMethodDef THPFunction_methods[] = {
    {(char*)"name", THPFunction_name, METH_NOARGS, nullptr},
    {(char*)"_sequence_nr", THPFunction_sequence_nr, METH_NOARGS, nullptr},
    {(char*)"_set_sequence_nr", THPFunction_set_sequence_nr, METH_O, nullptr},
    {(char*)"maybe_clear_saved_tensors",
     THPFunction_maybe_clear_saved_tensors,
     METH_NOARGS,
     nullptr},
    {(char*)"apply", THPFunction_apply, METH_CLASS | METH_VARARGS, nullptr},
    {(char*)"_register_hook_dict",
     THPFunction__register_hook_dict,
     METH_O,
     nullptr},
    {(char*)"register_hook", THPFunction_register_hook, METH_O, nullptr},
    {(char*)"register_prehook", THPFunction_register_prehook, METH_O, nullptr},
    {(char*)"_get_compiled_autograd_symints",
     THPFunction_get_compiled_autograd_symints,
     METH_NOARGS,
     nullptr},
    {nullptr}};
```

- THPFunction_apply

```c++
// PyTorch中autograd.Function类的核心C++执行入口函数
// 负责调度自定义autograd函数的前向传播（forward），处理输入输出、构建计算图节点
// 参数说明：
//   cls: Python层定义的autograd.Function子类（PyTypeObject指针）
//   inputs: 传入forward方法的输入参数（Python元组类型）
// 返回值：forward执行结果经过处理后的Python对象（计算图追踪/梯度关联后的输出）
PyObject* THPFunction_apply(PyObject* cls, PyObject* inputs) {
  // PyTorch错误处理宏：捕获C++层TH（Torch Core）相关异常，转换为Python异常向上传播
  // 与末尾END_HANDLE_TH_ERRORS成对使用，确保错误路径的资源正确释放
  HANDLE_TH_ERRORS

  // 保存当前autograd操作序列ID（在被自动递增前）
  // sequence_number用于标记计算图中操作的执行顺序，peek()仅读取当前值不触发递增
  auto seq_id = at::sequence_number::peek();
  // 解包Python输入参数：将inputs元组转换为C++层可处理的结构化数据
  // 模板参数false表示非inplace模式（不修改原始输入）
  // 返回值为pair：first是解包后的输入数据（含变量、参数元组等），second是输入属性标志
  auto info_pair = unpack_input<false>(inputs);
  // 引用解包后的输入数据容器（UnpackedInput包含输入变量列表、Python参数元组等）
  UnpackedInput& unpacked_input = info_pair.first;
  // 引用输入属性标志（InputFlags包含是否可执行、梯度需求、计算图边信息等）
  InputFlags& input_info = info_pair.second;

  // 记录函数调用（用于性能分析、Profiler追踪）
  // 时机：所有输入解码完成后，上下文（ctx）分配前（避免上下文初始化干扰性能统计）
  // 参数：1. 函数类名（从PyTypeObject中获取）；2. 追踪用输入数据；3. 保存的操作序列ID
  RECORD_FUNCTION(
      ((PyTypeObject*)cls)->tp_name,  // 对应的autograd.Function子类名称
      unpacked_input.record_function_inputs,  // 供Profiler使用的输入数据
      seq_id);  // 操作序列ID（确保追踪结果与计算图顺序一致）

  // 获取functorch（函数式编程接口）的线程局部存储（TLS）访问器
  const auto& functorch_tls = at::functorch::functorchTLSAccessor();
  if (functorch_tls) {
    // 注释说明：functorch对autograd.Function的支持在Python层实现
    // 若C++层执行到此处，仅允许_SingleLevelFunction类型（单层自动微分）
    // 该检查用于调试：若出现不支持的类型，直接抛出明确错误（避免静默失效）
    functorch_tls->checkSupportsSingleLevelAutogradFunction();
  }

  // 从autograd.Function子类中获取"_backward_cls"属性（反向传播类）
  // THPObjectPtr：PyTorch封装的Python对象智能指针，自动管理引用计数（避免内存泄漏）
  THPObjectPtr backward_cls(PyObject_GetAttrString(cls, "_backward_cls"));
  if (!backward_cls)  // 若获取反向传播类失败（如未定义），返回nullptr表示错误
    return nullptr;

  // 调用_backward_cls的无参构造函数，创建上下文对象（ctx）
  // PyObject_CallFunctionObjArgs：Python C API，调用对象的构造函数（参数为nullptr表示无参）
  THPObjectPtr ctx_obj(PyObject_CallFunctionObjArgs(backward_cls, nullptr));
  if (!ctx_obj)  // 上下文对象创建失败，返回错误
    return nullptr;

  // 将Python层的ctx_obj转换为C++层THPFunction指针（THPFunction是ctx的底层实现）
  THPFunction* ctx = (THPFunction*)ctx_obj.get();

  // 创建计算图节点（PyNode）：封装反向传播所需的上下文和元数据
  // std::shared_ptr管理PyNode生命周期，deleteNode为自定义删除器（确保节点资源正确释放）
  // 转移ctx_obj的所有权到PyNode（避免重复管理引用计数）
  auto cdata = std::shared_ptr<PyNode>(new PyNode(std::move(ctx_obj)), deleteNode);
  // 将计算图节点关联到ctx，供后续forward/backward访问
  ctx->cdata = cdata;

  // 若处于追踪模式（如torch.jit.trace），记录输入变量对应的计算图节点
  // _trace_pre_record：在追踪时将输入节点加入计算图，确保反向传播可追踪
  auto* node = _trace_pre_record(cls, inputs, unpacked_input.input_vars);

  // 初始化反向传播函数及上下文（ctx）的核心参数
  bool is_executable = input_info.is_executable;  // 标记输入是否可执行（影响反向传播计算逻辑）
  cdata->set_next_edges(std::move(input_info.next_edges));  // 设置计算图的下一条边（构建反向传播链路）
  // 转移输入梯度需求标记（哪些输入需要计算梯度），release()释放unique_ptr所有权到ctx
  ctx->needs_input_grad = input_info.needs_input_grad.release();
  // 转移输入变量标记（记录每个输入是否为可微分Variable）
  ctx->is_variable_input = std::move(input_info.is_variable_input);

  // 检查autograd.Function子类是否重写了setup_context静态方法
  // 关键说明：重写后forward方法不接收ctx参数，需通过setup_context关联ctx与输入/输出
  // 未重写则forward方法第一个参数为ctx，直接在forward中操作ctx

  // 从子类中获取setup_context方法（Python对象）
  auto cls_setup_context = THPObjectPtr(PyObject_GetAttrString(cls, "setup_context"));
  if (!cls_setup_context)  // 获取setup_context失败（如未定义），返回错误
    return nullptr;

  // 获取autograd.Function基类的setup_context方法（作为判断是否重写的基准）
  auto orig_setup_context = get_base_setup_context();
  if (!orig_setup_context)  // 获取基类方法失败，返回错误
    return nullptr;

  // 判断子类是否重写了setup_context：比较方法对象指针（不同则表示重写）
  auto overridden_setup_context = cls_setup_context.get() != orig_setup_context;

  // 获取输入参数元组的长度（即传入forward的参数个数）
  auto num_args = PyTuple_GET_SIZE(inputs);

  // 声明forward方法的输出结果（Python对象智能指针，自动管理引用计数）
  THPObjectPtr output;
  {
    // 作用域：临时禁用梯度计算模式（AutoGradMode(false)）
    // 原因：前向传播仅需构建计算图、记录依赖关系，无需实时计算梯度
    AutoGradMode grad_mode(false);
    // 临时禁用前向梯度（forward AD）模式（仅在反向传播时启用前向AD计算）
    at::AutoFwGradMode fw_grad_mode(false);

    // 从autograd.Function子类中获取forward方法（Python对象）
    THPObjectPtr forward_fn(PyObject_GetAttrString(cls, "forward"));
    if (!forward_fn)  // 获取forward方法失败，返回错误
      return nullptr;

    // 分支1：子类重写了setup_context（forward不接收ctx参数）
    if (overridden_setup_context) {
      // 1. 调用forward方法：传入解包后的输入参数元组（无ctx）
      output = PyObject_CallObject(forward_fn, unpacked_input.input_tuple);
      if (!output)  // forward执行失败，返回错误
        return nullptr;

      // 2. 构造setup_context的参数元组：(ctx, 原始输入, forward输出)
      // setup_context作用：将ctx与输入/输出关联，为反向传播保存必要信息
      auto ctx_input_output_tuple = make_ctx_input_output_tuple(ctx, unpacked_input, output);
      if (!ctx_input_output_tuple)  // 参数元组构造失败，返回错误
        return nullptr;

      // 3. 再次获取setup_context方法（确保指针有效性）
      THPObjectPtr setup_context_fn(PyObject_GetAttrString(cls, "setup_context"));
      // 4. 调用setup_context方法，完成ctx与输入/输出的绑定
      auto result = PyObject_CallObject(setup_context_fn, ctx_input_output_tuple);
      if (!result)  // setup_context执行失败，返回错误
        return nullptr;
    } else {
      // 分支2：子类未重写setup_context（forward接收ctx作为第一个参数）
      // 1. 构造forward的参数元组：(ctx, *解包后的输入参数)
      auto ctx_input_tuple = make_ctx_input_tuple(ctx, unpacked_input, num_args);
      if (!ctx_input_tuple)  // 参数元组构造失败，返回错误
        return nullptr;

      // 2. 调用forward方法：传入包含ctx的参数元组
      output = PyObject_CallObject(forward_fn, ctx_input_tuple);
    }

    // 检查forward输出是否有效（为空表示执行失败）
    if (!output)
      return nullptr;
  }  // 作用域结束：AutoGradMode和AutoFwGradMode自动恢复原状态

  // 处理forward输出：关联计算图、设置梯度追踪、处理追踪模式等
  // 参数说明：
  //   cls: autograd.Function子类 | cdata: 计算图节点 | ctx: 上下文对象
  //   unpacked_input: 解包后的输入 | inputs: 原始输入元组 | output: forward输出（所有权转移）
  //   is_executable: 输入是否可执行 | node: 追踪模式下的输入节点 | overridden_setup_context: 是否重写setup_context
  return process_outputs(
      cls,
      cdata,
      ctx,
      unpacked_input,
      inputs,
      std::move(output),  // 转移output所有权，避免二次拷贝
      is_executable,
      node,
      overridden_setup_context);

  // 错误处理宏结束：与HANDLE_TH_ERRORS成对，捕获所有未处理的异常并转换为Python异常
  END_HANDLE_TH_ERRORS
}
```

- 输出绑定计算图节点

```c++
// 处理autograd.Function前向传播（forward）的输出结果，核心职责是：
// 1. 标准化输出格式（统一转为元组）；2. 为输出张量绑定计算图节点/梯度函数；
// 3. 记录反向传播所需的输入元数据；4. 保存反向依赖的变量；5. 处理JIT追踪模式；
// 6. 清理临时资源，返回最终的Python层输出。
// 返回值：处理后的Python侧输出对象（单个元素或元组，与forward返回格式一致）
PyObject* process_outputs(
    PyObject* op_obj,                     // 对应的autograd.Function类对象（Python侧）
    const std::shared_ptr<PyNode>& cdata, // 计算图节点（PyNode）的智能指针，存储反向传播核心信息
    THPFunction* grad_fn,                 // 维护前向/反向的状态和元数据
    const UnpackedInput& unpacked,        // 解包后的输入数据,包含输入变量列表等
    PyObject* inputs,                     // 原始Python侧输入参数元组
    // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
    THPObjectPtr&& raw_output, // forward执行后的原始输出（THPObjectPtr右值引用，转移所有权）
    bool is_executable,        // 输入是否可执行（决定是否需要记录反向传播元数据）
    torch::jit::Node* node,    // JIT追踪模式下的计算图节点（torch::jit::Node）， nullptr表示非追踪模式
    // 是否重写了setup_context方法（影响张量保存逻辑）
    bool overridden_setup_context) {
  // 确保输出是元组格式：若raw_output不是元组，自动包装为单元素元组
  // 返回值unpack_output标记是否需要后续解包（即原始输出不是元组，仅为单个元素）
  bool unpack_output = ensure_tuple(raw_output);

  // 获取输出元组的元素个数（统一格式后，直接通过元组API获取长度）
  auto num_outputs = PyTuple_GET_SIZE(raw_output.get());

  // 创建新的输出元组容器，用于存储“绑定计算图后的最终输出”
  THPObjectPtr outputs(PyTuple_New(num_outputs));
  if (!outputs)  // 元组创建失败，抛出Python错误（由上层错误处理宏捕获）
    throw python_error();

  // 清空计算图节点的输入元数据缓存，避免前一次执行的残留数据干扰
  cdata->clear_input_metadata(); // PyNode

  // 若输入可执行（is_executable=true），记录输入变量的元数据（类型、设备、尺寸等）
  // 这些元数据将在反向传播时用于校验输入输出兼容性、优化梯度计算
  if (is_executable) {
    grad_fn->input_info.clear();  // 清空旧的输入元数据
    grad_fn->input_info.reserve(unpacked.input_vars.size());  // 预分配空间，优化性能
    // 遍历所有解包后的输入变量，为每个变量记录元数据并存入input_info
    for (auto& var : unpacked.input_vars) {
      grad_fn->input_info.emplace_back(var);
    }
  }

  // 存储“重写setup_context时需要保存的张量”（通过集合去重，避免重复保存）
  std::unordered_set<at::TensorImpl*> to_save_if_setup_context{};
  // 存储最终需要保存的张量列表（供反向传播使用）
  std::vector<std::optional<at::Tensor>> tensors_to_save{};
  // 收集需要保存的张量：根据setup_context是否重写、是否可执行等条件，筛选反向传播依赖的张量
  _get_tensors_to_save(
      grad_fn,                  // 上下文对象，从中获取待保存的张量信息
      to_save_if_setup_context, // 去重集合，避免重复保存同一张量
      tensors_to_save,          // 输出：最终待保存的张量列表
      overridden_setup_context, // 是否重写setup_context（影响筛选逻辑）
      is_executable);           // 是否可执行（非可执行时无需保存）

  // 判断当前操作是否为inplace操作：通过grad_fn的dirty_tensors是否存在（非空）判断
  bool is_inplace = static_cast<bool>(grad_fn->dirty_tensors);
  // 核心步骤：包装输出张量，为其绑定计算图节点和梯度函数，使其具备autograd追踪能力
  // 作用：1. 为输出张量设置grad_fn（指向当前反向传播函数）；2. 关联计算图节点cdata；
  // 3. 处理inplace操作的梯度传播；4. 过滤无需微分的输出
  _wrap_outputs(
      cdata,                    // 计算图节点，与输出绑定
      grad_fn,                  // 梯度上下文对象
      unpacked.input_vars,      // 输入变量列表（用于关联输入输出的依赖关系）
      raw_output,               // forward原始输出（待包装）
      outputs,                  // 输出容器（存储包装后的最终输出）
      is_executable,            // 是否可执行（影响梯度绑定逻辑）
      to_save_if_setup_context); // 需要保存的张量集合（用于校验）

  // JIT追踪模式后处理：将输出记录到追踪节点node中，更新计算图的输入输出关联
  // 仅当node非nullptr（即处于追踪模式）时有效
  _trace_post_record(
      node,          // JIT追踪节点
      op_obj,        // autograd.Function类对象
      unpacked.input_vars,  // 输入变量列表
      outputs,       // 包装后的输出元组
      is_inplace,    // 是否为inplace操作（追踪时需特殊标记）
      unpack_output); // 是否需要解包（影响追踪输出格式记录）

  // 关键注意点：创建SavedVariables（反向传播依赖的保存变量）必须在输出包装之后
  // 原因：输出张量需要先通过_wrap_outputs设置好grad_fn/fw_grad（前向梯度），
  // 才能被正确保存为SavedVariables，确保反向传播时梯度可追溯
  if (is_executable) {
    // 保存需要的变量到计算图节点cdata中，供反向传播时读取
    _save_variables(tensors_to_save, cdata, grad_fn);
  } else {
    // 非可执行模式：清理grad_fn中无需保留的属性，避免内存泄漏
    Py_CLEAR(grad_fn->to_save);          // 清理待保存变量列表
    Py_CLEAR(grad_fn->non_differentiable); // 清理不可微分输出标记
  }

  // 清理前向传播中临时保存的“前向专用数据”，释放资源
  Py_CLEAR(grad_fn->saved_for_forward);

  // 根据unpack_output标记决定输出格式：恢复forward的原始返回格式
  if (unpack_output) {
    // 若原始输出是单个元素（被ensure_tuple包装为单元素元组），则解包返回单个元素
    PyObject* output = PyTuple_GET_ITEM(outputs.get(), 0);
    Py_INCREF(output); // 增加引用计数：返回的对象需由Python层管理生命周期
    return output;
  }

  // 若原始输出是元组，则直接返回包装后的元组（释放THPObjectPtr所有权）
  return outputs.release();
}
```

核心点在于_wrap_outputs:

**给定原始输出张量的 Python 元组（raw_output），将其中每个张量包装为可微分变量（Variable）后存入另一个 Python 元组（outputs）的对应位置；若输出需要计算梯度（requires grad），则将梯度函数（self，即当前反向传播函数）关联到该变量上。若生成这些输出张量的操作为原地操作（inplace），则需要处理大量复杂逻辑 —— 输入张量到可微分变量的映射表（t2var）用于检测该原地操作是否发生，脏张量集合（dirty_inputs，即受原地操作修改的输入张量）则用于确定该场景下的处理逻辑，该方法执行完成后，映射表 t2var 会扩展新增输出张量与其对应可微分变量的映射关系。**

# 2 recompute 核心逻辑
