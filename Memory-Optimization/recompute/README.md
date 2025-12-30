**从底像上分析下具体逻辑** <br>

# 1 最终解决策略

**最终靠前向和反向时注册的tensor 钩子函数来解决** <br>

```python
class saved_tensors_hooks:
    """Context-manager that sets a pair of pack / unpack hooks for saved tensors.

    Use this context-manager to define how intermediary results of an operation
    should be packed before saving, and unpacked on retrieval.

    In that context, the ``pack_hook`` function will be called everytime an
    operation saves a tensor for backward (this includes intermediary results
    saved using
    :func:`~torch.autograd.function._ContextMethodMixin.save_for_backward` but
    also those recorded by a PyTorch-defined operation). The output of
    ``pack_hook`` is then stored in the computation graph instead of the
    original tensor.

    The ``unpack_hook`` is called when the saved tensor needs to be accessed,
    namely when executing :func:`torch.Tensor.backward()` or
    :func:`torch.autograd.grad()`. It takes as argument the *packed* object
    returned by ``pack_hook`` and should return a tensor which has the same
    content as the original tensor (passed as input to the corresponding
    ``pack_hook``).

    The hooks should have the following signatures:

        pack_hook(tensor: Tensor) -> Any

        unpack_hook(Any) -> Tensor

    where the return value of ``pack_hook`` is a valid input to ``unpack_hook``.

    In general, you want ``unpack_hook(pack_hook(t))`` to be equal to ``t`` in terms
    of value, size, dtype and device.

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> def pack_hook(x):
        ...     print("Packing", x)
        ...     return x
        >>>
        >>> def unpack_hook(x):
        ...     print("Unpacking", x)
        ...     return x
        >>>
        >>> a = torch.ones(5, requires_grad=True)
        >>> b = torch.ones(5, requires_grad=True) * 2
        >>> with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        ...     y = a * b
        Packing tensor([1., 1., 1., 1., 1.], requires_grad=True)
        Packing tensor([2., 2., 2., 2., 2.], grad_fn=<MulBackward0>)
        >>> y.sum().backward()
        Unpacking tensor([1., 1., 1., 1., 1.], requires_grad=True)
        Unpacking tensor([2., 2., 2., 2., 2.], grad_fn=<MulBackward0>)

    .. warning ::
        Performing an inplace operation on the input to either hooks may lead
        to undefined behavior.

    .. warning ::
        Only one pair of hooks is allowed at a time. When recursively nesting this
        context-manager, only the inner-most pair of hooks will be applied.
    """

    def __init__(
        self,
        pack_hook: Callable[[torch.Tensor], Any],
        unpack_hook: Callable[[Any], torch.Tensor],
    ) -> None:
        self.pack_hook = pack_hook
        self.unpack_hook = unpack_hook

    def __enter__(self) -> None:
        torch._C._autograd._push_saved_tensors_default_hooks(
            self.pack_hook, self.unpack_hook
        )

    def __exit__(self, *args: object) -> None:
        torch._C._autograd._pop_saved_tensors_default_hooks()
```

# 2. _checkpoint_hook 提供中间具体钩子函数

```python
class _checkpoint_hook(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, frame):
        def pack_hook(x):
            # See Rule 4 above
            holder = _Holder()
            frame.weak_holders.append(weakref.ref(holder))
            # Save metadata to detect non-determinism
            if frame.metadata_fn is not None:
                with torch.no_grad():
                    frame.x_metadatas.append(frame.metadata_fn(x))
            return holder

        def unpack_hook(holder):
            gid = torch._C._current_graph_task_id()
            if gid == -1:
                # generate a temporary id if we trigger unpack outside of a backward call
                gid = int(uuid.uuid4())

            if not frame.is_recomputed[gid]:
                ctx = frame.input_saver.grad_fn
                args = ctx.get_args(ctx.saved_tensors)

                try:
                    with _recomputation_hook(
                        weakref.ref(frame), gid
                    ), torch.autograd.enable_grad():
                        frame.recompute_fn(*args)
                except _StopRecomputationError:
                    pass
                frame.is_recomputed[gid] = True
                frame.check_recomputed_tensors_match(gid)

            _internal_assert(gid in holder.handles)

            if holder.handles[gid] is None:
                raise CheckpointError(
                    "torch.utils.checkpoint: Unpack is being triggered for a tensor that was already "
                    "unpacked once. If you are calling ctx.saved_tensors in backward, make sure to do "
                    "so only once. Otherwise please open an issue with details on your use case."
                )
            _internal_assert(holder.handles[gid] in frame.recomputed[gid])
            ret = frame.recomputed[gid][holder.handles[gid]]
            holder.handles[gid] = None
            return ret

        if frame.unpack_error_cb is not None:
            def unpack_hook_with_error_cb(holder):
                try:
                    return unpack_hook(holder)
                except CheckpointError as e:
                    frame.unpack_error_cb(e)
            super().__init__(pack_hook, unpack_hook_with_error_cb)
        else:
            super().__init__(pack_hook, unpack_hook)
```

# 3 重计算时 _recomputation_hook 来处理 no_grad

```python
class _recomputation_hook(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, target_frame_ref: ReferenceType, gid: int):
        def pack_hook(x):
            x = x.detach() if x.requires_grad else x
            target_frame = target_frame_ref()
            assert target_frame is not None  # appease mypy
            recomp_idx = target_frame.recomp_counter[gid]
            target_frame.recomp_counter[gid] += 1

            if recomp_idx >= len(target_frame.weak_holders):
                assert not target_frame.early_stop
                if not target_frame.forward_completed:
                    # We run into this case when early stop is not enabled and do
                    # grad within checkpoint.
                    # We need to set this flag, so we don't error out later when
                    # we check if the number of tensors saved during forward and
                    # recomputation match.
                    target_frame.ignore_saved_mismatch = True
                    return x
                raise CheckpointError(
                    "torch.utils.checkpoint: trying to save more tensors during "
                    "recomputation than during the original forward pass."
                )

            holder = target_frame.weak_holders[recomp_idx]()

            # This holder may have been cleared because someone may have called
            # backward within forward. If so, we don't need to save.
            if holder is not None:
                _internal_assert(holder.handles.get(gid, None) is None)
                holder.handles[gid] = _Handle()
                target_frame.recomputed[gid][holder.handles[gid]] = x

            if target_frame.early_stop and target_frame.recomp_counter[gid] == len(
                target_frame.weak_holders
            ):
                raise _StopRecomputationError
            # See Rule 6: [ retain_graph is True ] above
            return x

        def unpack_hook(x):
            # See Rule 6: [ retain_graph is True ] above for an example of when
            # the graph created during recomputation could be backwarded.
            return x

        super().__init__(pack_hook, unpack_hook)
```

# 4 _checkpoint_without_reentrant_generator 中启动 recomputation 上下文

**通过yeild 而非 return 来维持 recompute 三个上下文**

- _checkpoint_hook 处理 recompute_fn 的计算
- forward_context 处理 forward 环境 : _CachingTorchDispatchMode(TorchDispatchMode)
- recompute_context 处理 recompute 环境：_CachedTorchDispatchMode(TorchDispatchMode)

```python
def _checkpoint_without_reentrant_generator(
    fn,
    preserve_rng_state=True,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    *args,
    **kwargs
):
    """Checkpointing without reentrant autograd.

    Args:
        fn: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        preserve_rng_state(bool, optional):  Omit stashing and restoring
            the RNG state during each checkpoint.
            Default: ``True``
        context_fn(Callable, optional): A callable returning a tuple of two
            context managers. The function and its recomputation will be run
            under the first and second context managers respectively.
        determinism_check(str, optional): A string specifying the determinism
            check to perform. By default it is set to ``"default"`` which
            compares the shapes, dtypes, and devices of the recomputed tensors
            against those the saved tensors. To turn off this check, specify
            ``"none"``. Currently these are the only two supported values.
            Please open an issue if you would like to see more determinism
            checks.
        debug(bool, optional): If ``True``, error messages will also include
            a trace of the operators ran during the original forward computation
            as well as the recomputation.
        *args: Arguments to pass in to the given ``function``.
        **kwargs: Keyword arguments to pass into the given ``function``.
    """
    unpack_error_cb = None

    if _checkpoint_debug_enabled if _checkpoint_debug_enabled is not None else debug:
        if context_fn != noop_context_fn:
            raise ValueError(
                "debug=True is incompatible with non-default context_fn"
            )
        context_fn, unpack_error_cb = _get_debug_context_and_cb()

    if determinism_check in _allowed_determinism_checks_to_fns:
        metadata_fn = _allowed_determinism_checks_to_fns[determinism_check]
    else:
        raise ValueError(
            f"determinism_check should be one of {list(_allowed_determinism_checks_to_fns.keys())}, "
            f"but got {determinism_check}"
        )

    device_type = _infer_device_type(*args)
    device_module = _get_device_module(device_type)
    forward_context, recompute_context = context_fn()
    if _is_compiling(fn, args, kwargs) and context_fn != noop_context_fn:
        assert (
            isinstance(forward_context, TorchDispatchMode) and
            isinstance(recompute_context, TorchDispatchMode)
        ), \
            "In torch.compile mode, `context_fn` arg passed to `torch.utils.checkpoint` " + \
            "must generate a tuple of two `TorchDispatchMode`s."
    # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
    device_autocast_kwargs, cpu_autocast_kwargs = _get_autocast_kwargs(device_type=device_type)

    if preserve_rng_state:
        fwd_cpu_state = torch.get_rng_state()
        # Don't eagerly initialize the cuda context by accident.
        # (If the user intends that the context is initialized later, within their
        # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
        # we have no way to anticipate this will happen before we run the function.
        # If they do so, we raise an error.)
        had_device_in_fwd = False
        if getattr(device_module, "_initialized", False):
            had_device_in_fwd = True
            fwd_devices, fwd_device_states = get_device_states(*args)

    def recompute_fn(*inputs):
        kwargs, *args = inputs
        # This will be called later during recomputation. This wrapping enables
        # the necessary global state to be captured.
        rng_devices = []
        if preserve_rng_state and had_device_in_fwd:
            rng_devices = fwd_devices
        with torch.random.fork_rng(
            devices=rng_devices, enabled=preserve_rng_state, device_type=device_type
        ):
            if preserve_rng_state:
                torch.set_rng_state(fwd_cpu_state)
                if had_device_in_fwd:
                    set_device_states(fwd_devices, fwd_device_states, device_type=device_type)

            device_autocast_ctx = torch.amp.autocast(
                device_type=device_type, **device_autocast_kwargs
            ) if torch.amp.is_autocast_available(device_type) else contextlib.nullcontext()
            with device_autocast_ctx, torch.amp.autocast("cpu", **cpu_autocast_kwargs), recompute_context:  # type: ignore[attr-defined]
                fn(*args, **kwargs)

    new_frame = _CheckpointFrame(
        recompute_fn,
        _enable_checkpoint_early_stop,
        unpack_error_cb,
        metadata_fn
    )
    dummy = torch.empty((0,), requires_grad=True)
    new_frame.input_saver = _NoopSaveInputs.apply(dummy, kwargs, *args)

    # When ambient grad_mode is False
    if new_frame.input_saver.grad_fn is None:
        yield
        return

    with _checkpoint_hook(new_frame), forward_context:
        yield
    new_frame.forward_completed = True

    if getattr(device_module, "_initialized", False) and \
       preserve_rng_state and not had_device_in_fwd:  # type: ignore[possibly-undefined]
        # Device was not initialized before running the forward, so we didn't
        # stash the device state.
        raise RuntimeError(
            "PyTorch's device state was initialized in the forward pass "
            "of a Checkpoint, which is not allowed. Please open an issue "
            "if you need this feature."
        )

    return
```

# context_fn 设置接口

```python
def create_selective_checkpoint_contexts(policy_fn_or_list, allow_cache_entry_mutation=False):
    """
    Helper to avoid recomputing certain ops during activation checkpointing.

    Use this with `torch.utils.checkpoint.checkpoint` to control which
    operations are recomputed during the backward pass.

    Args:
        policy_fn_or_list (Callable or List):
          - If a policy function is provided, it should accept a
            :class:`SelectiveCheckpointContext`, the :class:`OpOverload`, args and
            kwargs to the op, and return a :class:`CheckpointPolicy` enum value
            indicating whether the execution of the op should be recomputed or not.
          - If a list of operations is provided, it is equivalent to a policy
            returning `CheckpointPolicy.MUST_SAVE` for the specified
            operations and `CheckpointPolicy.PREFER_RECOMPUTE` for all other
            operations.
        allow_cache_entry_mutation (bool, optional): By default, an error is
            raised if any tensors cached by selective activation checkpoint are
            mutated in order to ensure correctness. If set to `True`, this check
            is disabled.
    Returns:
        A tuple of two context managers.

    Example:
        >>> # xdoctest: +REQUIRES(LINUX)
        >>> import functools
        >>>
        >>> x = torch.rand(10, 10, requires_grad=True)
        >>> y = torch.rand(10, 10, requires_grad=True)
        >>>
        >>> ops_to_save = [
        >>>    torch.ops.aten.mm.default,
        >>> ]
        >>>
        >>> def policy_fn(ctx, op, *args, **kwargs):
        >>>    if op in ops_to_save:
        >>>        return CheckpointPolicy.MUST_SAVE
        >>>    else:
        >>>        return CheckpointPolicy.PREFER_RECOMPUTE
        >>>
        >>> context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)
        >>>
        >>> # or equivalently
        >>> context_fn = functools.partial(create_selective_checkpoint_contexts, ops_to_save)
        >>>
        >>> def fn(x, y):
        >>>     return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y
        >>>
        >>> out = torch.utils.checkpoint.checkpoint(
        >>>     fn, x, y,
        >>>     use_reentrant=False,
        >>>     context_fn=context_fn,
        >>> )
    """
    # NB: If grad_mode is disabled, checkpoint would not run forward under
    #     context_fn anyway, so proceed as usual.
    if isinstance(policy_fn_or_list, list):
        for op in policy_fn_or_list:
            if not isinstance(op, torch._ops.OpOverload):
                _extra_msg = (
                    "Please update the OpOverloadPacket to a specific OpOverload."
                    "For example, if you have `torch.ops.aten.mm`, change it to `torch.ops.aten.mm.default`."
                ) if isinstance(op, torch._ops.OpOverloadPacket) else ""
                raise ValueError(
                    f"Expected op in `op_list` to be an OpOverload but got: {op} "
                    f"of type {type(op)}. {_extra_msg}"
                )

        def policy_fn(ctx, op, *args, **kwargs):
            if op in policy_fn_or_list:
                return CheckpointPolicy.MUST_SAVE
            else:
                return CheckpointPolicy.PREFER_RECOMPUTE
    elif callable(policy_fn_or_list):
        policy_fn = policy_fn_or_list
    else:
        raise TypeError("policy_fn_or_list must be either a function or a list of ops.")

    storage: Dict[Any, List[Any]] = defaultdict(list)
    return (
        _CachingTorchDispatchMode(policy_fn, storage),
        _CachedTorchDispatchMode(policy_fn, storage, allow_cache_entry_mutation),
    )
```

# 6 最外层用户调用

TorchDynamo 不会深入到 utils.checkpoint 函数内部执行。其执行流程如下：
- TorchDynamo 会尝试将 utils.checkpoint 封装为一个高阶算子（HigherOrderOp），在此过程中会试探性检查前向传播函数是否可以安全地被追踪（trace）。
- 若检查通过，则由 Dynamo 生成的 FX 计算图会包含这个被封装的高阶算子。因此，TorchDynamo 不会再深入到 utils.checkpoint 函数内部进行处理。
- 若检查不通过，TorchDynamo 会通过中断计算图的方式回退到即时执行模式（eager mode）。此时，下文的禁用封装器会确保：在 utils.checkpoint 内部代码所生成的执行帧中，TorchDynamo 不会被再次触发。

```python
# TorchDynamo does not step inside utils.checkpoint function.  The flow
# looks likes this
#  1) TorchDynamo tries to wrap utils.checkpoint in a HigherOrderOp by
#     speculatively checking if the forward function is safe to trace.
#  2) If yes, then Dynamo-generated Fx graph has the wrapped higher
#     order op. As a result, TorchDynamo does not look inside utils.checkpoint.
#  3) If not, then TorchDynamo falls back to eager by performing a graph
#     break. And here, the following disable wrapper ensures that
#     TorchDynamo does not trigger again on the frames created by
#     utils.checkpoint innards.
@torch._disable_dynamo
def checkpoint(
    function,
    *args,
    use_reentrant: Optional[bool] = None,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    **kwargs
):
    r"""Checkpoint a model or part of the model.

    Activation checkpointing is a technique that trades compute for memory.
    Instead of keeping tensors needed for backward alive until they are used in
    gradient computation during backward, forward computation in checkpointed
    regions omits saving tensors for backward and recomputes them during the
    backward pass. Activation checkpointing can be applied to any part of a
    model.

    There are currently two checkpointing implementations available, determined
    by the :attr:`use_reentrant` parameter. It is recommended that you use
    ``use_reentrant=False``. Please refer the note below for a discussion of
    their differences.

    .. warning::

        If the :attr:`function` invocation during the backward pass differs
        from the forward pass, e.g., due to a global variable, the checkpointed
        version may not be equivalent, potentially causing an
        error being raised or leading to silently incorrect gradients.

    .. warning::

        The ``use_reentrant`` parameter should be passed explicitly. In version
        2.4 we will raise an exception if ``use_reentrant`` is not passed.
        If you are using the ``use_reentrant=True`` variant, please refer to the
        note below for important considerations and potential limitations.

    .. note::

        The reentrant variant of checkpoint (``use_reentrant=True``) and
        the non-reentrant variant of checkpoint (``use_reentrant=False``)
        differ in the following ways:

        * Non-reentrant checkpoint stops recomputation as soon as all needed
          intermediate activations have been recomputed. This feature is enabled
          by default, but can be disabled with :func:`set_checkpoint_early_stop`.
          Reentrant checkpoint always recomputes :attr:`function` in its
          entirety during the backward pass.

        * The reentrant variant does not record the autograd graph during the
          forward pass, as it runs with the forward pass under
          :func:`torch.no_grad`. The non-reentrant version does record the
          autograd graph, allowing one to perform backward on the graph within
          checkpointed regions.

        * The reentrant checkpoint only supports the
          :func:`torch.autograd.backward` API for the backward pass without its
          `inputs` argument, while the non-reentrant version supports all ways
          of performing the backward pass.

        * At least one input and output must have ``requires_grad=True`` for the
          reentrant variant. If this condition is unmet, the checkpointed part
          of the model will not have gradients. The non-reentrant version does
          not have this requirement.

        * The reentrant version does not consider tensors in nested structures
          (e.g., custom objects, lists, dicts, etc) as participating in
          autograd, while the non-reentrant version does.

        * The reentrant checkpoint does not support checkpointed regions with
          detached tensors from the computational graph, whereas the
          non-reentrant version does. For the reentrant variant, if the
          checkpointed segment contains tensors detached using ``detach()`` or
          with :func:`torch.no_grad`, the backward pass will raise an error.
          This is because ``checkpoint`` makes all the outputs require gradients
          and this causes issues when a tensor is defined to have no gradient in
          the model. To avoid this, detach the tensors outside of the
          ``checkpoint`` function.

    Args:
        function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        preserve_rng_state(bool, optional):  Omit stashing and restoring
            the RNG state during each checkpoint. Note that under torch.compile,
            this flag doesn't take effect and we always preserve RNG state.
            Default: ``True``
        use_reentrant(bool):
            specify whether to use the activation checkpoint variant that
            requires reentrant autograd. This parameter should be passed
            explicitly. In version 2.5 we will raise an exception if
            ``use_reentrant`` is not passed. If ``use_reentrant=False``,
            ``checkpoint`` will use an implementation that does not require
            reentrant autograd. This allows ``checkpoint`` to support additional
            functionality, such as working as expected with
            ``torch.autograd.grad`` and support for keyword arguments input into
            the checkpointed function.
        context_fn(Callable, optional): A callable returning a tuple of two
            context managers. The function and its recomputation will be run
            under the first and second context managers respectively.
            This argument is only supported if ``use_reentrant=False``.
        determinism_check(str, optional): A string specifying the determinism
            check to perform. By default it is set to ``"default"`` which
            compares the shapes, dtypes, and devices of the recomputed tensors
            against those the saved tensors. To turn off this check, specify
            ``"none"``. Currently these are the only two supported values.
            Please open an issue if you would like to see more determinism
            checks. This argument is only supported if ``use_reentrant=False``,
            if ``use_reentrant=True``, the determinism check is always disabled.
        debug(bool, optional): If ``True``, error messages will also include
            a trace of the operators ran during the original forward computation
            as well as the recomputation. This argument is only supported if
            ``use_reentrant=False``.
        args: tuple containing inputs to the :attr:`function`

    Returns:
        Output of running :attr:`function` on :attr:`*args`
    """
    if use_reentrant is None:
        warnings.warn(
            "torch.utils.checkpoint: the use_reentrant parameter should be "
            "passed explicitly. In version 2.5 we will raise an exception "
            "if use_reentrant is not passed. use_reentrant=False is "
            "recommended, but if you need to preserve the current default "
            "behavior, you can pass use_reentrant=True. Refer to docs for more "
            "details on the differences between the two variants.",
            stacklevel=2
        )
        use_reentrant = True

    # Hack to mix *args with **kwargs in a python 2.7-compliant way
    preserve = kwargs.pop("preserve_rng_state", True)
    if kwargs and use_reentrant:
        raise ValueError(
            "Unexpected keyword arguments: " + ",".join(arg for arg in kwargs)
        )

    if use_reentrant:
        if context_fn is not noop_context_fn or debug is not False:
            raise ValueError(
                "Passing `context_fn` or `debug` is only supported when "
                "use_reentrant=False."
            )
        return CheckpointFunction.apply(function, preserve, *args)
    else:
        gen = _checkpoint_without_reentrant_generator(
            function, preserve, context_fn, determinism_check, debug, *args, **kwargs
        )
        # Runs pre-forward logic
        next(gen)
        ret = function(*args, **kwargs)
        # Runs post-forward logic
        try:
            next(gen)
        except StopIteration:
            return ret
```
