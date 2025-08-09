# 1 Pytorch DDP 理论
- [论文链接](https://arxiv.org/pdf/2006.15704)

# 2 pytorch DDP 代码实现
- https://github.com/pytorch/examples/tree/main/distributed/ddp

启动脚本
```
# https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu_torchrun.py

torchrun --standalone --nnodes=1 --nproc-per-node=2 multigpu_torchrun.py 4 1 --batch_size 4
```

# 3 pytorch DDP 代码解析
- [https://github.com/pytorch/pytorch/blob/main/torch/nn/parallel/distributed.py](https://github1s.com/pytorch/pytorch/blob/main/torch/nn/parallel/distributed.py#L326)

## 3.1 DistributedDataParallel 属性和方法
```python
class DistributedDataParallel(Module, Joinable)
    def __init__(...):
        self.logger
        self.process_group
        self.device_mesh
        self._delay_all_reduce_params
        self.parameters_to_ignore
        self._module_parameters
        self.is_multi_device_module
        self.device_type
        self.device_ids
        self.output_device
        self.static_graph
        self.dim
        self.module
        self.device
        self.broadcast_buffers
        self.find_unused_parameters
        self.require_backward_grad_sync
        self.require_forward_grad_sync
        self.gradient_as_bucket_view
        self.mixed_precision
        self.broadcast_bucket_size
        self.bucket_bytes_cap_default
        self.bucket_bytes_cap
        self.use_side_stream_for_tensor_copies
        self._delay_grad_buffer
        self._delay_grad_views
        self._delay_all_reduce_all_params
        self._comm_hooks
        self._mp_stream
        self._submodule_to_event
        self._has_rebuilt_buckets
        self._lazy_init_ran
        self._accum_grad_hooks
        self._use_python_reducer
        self._force_to_disable_cpp_reducer
        self._ddp_sink_clone

    def _register_accum_grad_hook(...):
    def _delayed_all_reduce_hook(...):
    def _register_delay_all_reduce_hook(...):
    def _setup_in_backward_optimizers(...):
    def _fire_reducer_autograd_hook(...):
    def _root_copy_hook(...):
    def _module_wait_for_copy_hook(...):
    def _log_and_throw(...):
    def _ddp_init_helper(...):
    def __getstate__(...):
    def __setstate__(...):
    def _build_params_for_reducer(...):
    def _assign_modules_buffers(...):
    def _build_debug_param_to_name_mapping(...):
    def _get_parameters(...):
    def _check_default_group(...):
    def no_sync(...):
    def _get_active_ddp_module(...):
    def _inside_ddp_forward(...):
    def _run_ddp_forward(...):
    def _clear_grad_buffer(...):
    def _lazy_init(...):
    def _should_disable_cpp_reducer(...):
    def _pre_forward(...):
    def _post_forward(...):
    def forward(...):
    def scatter(...):
    def to_kwargs(...):
    def gather(...):
    def train(...):
    def _check_global_requires_backward_grad_sync(...):
    def _check_and_sync_module_buffers(...):
    def _sync_final_model(...):
    def _match_all_reduce_for_bwd_pass(...):
    def _match_unused_params_allreduce(...):
    def join(...):
    def join_hook(...):
    def join_device(...):
    def join_process_group(...):
    def _register_buffer_comm_hook(...):
    def register_comm_hook(...):
    def _register_builtin_comm_hook(...):
    def _register_fused_optim(...):
    def _distributed_broadcast_coalesced(...):
    def _check_sync_bufs_post_fwd(...):
    def _check_sync_bufs_pre_fwd(...):
    def will_sync_module_buffers(...):
    def _find_common_rank(...):
    def _sync_buffers(...):
    def _sync_module_buffers(...):
    def _default_broadcast_coalesced(...):
    def _passing_sync_batchnorm_handle(...):
    def _check_comm_hook(...):
    def _distributed_rank(...):
    def _get_data_parallel_params(...):
    def _set_params_and_buffers_to_ignore_for_model(...):
    def _get_ddp_logging_data(...):
    def _set_ddp_runtime_logging_sample_rate(...):
    def _set_static_graph(...):
    def _remove_autograd_hooks(...):
    def _check_reducer_finalized(...):
    def _set_sparse_metadata(...):
    def _update_process_group(...):
    def _set_ddp_sink_clone(...):
```

## 3.2 self.process_group 的初始化
- process_group is not None and device_mesh is not None : RuntimeError
- process_group is None and device_mesh is None : self.process_group = _get_default_group() 
- process_group is not None and device_mesh is None : self.process_group = process_group
- process_group is None and device_mesh is Not None : # 这里针对 DDP + TP(用到了DTensor) 的情况
```python
  #这个函数 _pre_dp_module_transform 的作用是在将一个已经应用了张量并行（Tensor Parallelism, TP）的模块包装到数据并行（Data Parallelism, DP）时，
  # 确保张量并行和数据并行之间的兼容性。具体来说，它处理分布式张量（DTensor）和本地张量之间的转换，
  # 以避免 DistributedDataParallel（DDP）对 DTensor 进行特殊处理，并确保梯度能够正确传播。
  if device_mesh.ndim != 1:
      raise RuntimeError(
          f"Only 1D device mesh is supported, but got {device_mesh}."
      )
  self.device_mesh = device_mesh
  self.process_group = device_mesh.get_group(mesh_dim=0)
  from torch.distributed.device_mesh import _mesh_resources

  root_mesh = _mesh_resources.get_root_mesh(device_mesh)
  # if a root mesh is not the same as device_mesh,
  # meaning the device_mesh is sliced out from the root mesh.
  if root_mesh != device_mesh:
      # TODO: This is a temporary work around to enable DDP + TP.
      # We should do the logic in DDP so that the 2D implementation is
      # sound and the state_dict works out of the box.
      # This has to be done before check UninitializedParameter.
      from torch.distributed.tensor.parallel.ddp import (
          _pre_dp_module_transform,
      )

      _pre_dp_module_transform(module)
```

**_pre_dp_module_tranform 的实现** <br>
```python
# 启用在使用 DDP 时 PyTorch 中张量并行（TP）和数据并行（DP）之间的组合性。
# 1. 我们需要在使用数据并行 API 包装模块之前，将 DTensor 类型的参数转换为本地张量。
# 2. 然后，我们注册两个钩子：一个用于在前向传播前将本地张量转换回 DTensor，
# 3. 另一个用于在前向传播后将 DTensor 转换回本地张量。
# 如此，我们在前向传播时按照DTesor来计算，前向传播后转换为local tensor，反向传播再转为DTensor ？？？ 这里可能有点疑问.
# 通过这种方式集成，我们可以避免 DDP 对 DTensor 参数进行任何特殊处理，并确保 DTensor 的梯度能够正确传播回 DP，例如 DDP 的梯度桶。

def _pre_dp_module_transform(module: nn.Module):
    """
    Enable the composability between Tensor Parallelism (TP) and Data
    Parallelism(DP) in PyTorch when using DDP. We need to convert Parameters which
    are DTensors to local tensors before wrapping with data parallelism API.
    We then register two hooks, one for converting local tensors back to DTensor
    preforward and one to convert DTensors back to tensors after Forward. By
    integrating this way, we avoid any special handling of DTensor parameters by DDP
    and get DTensor's gradients propagated back to DP, e.g. gradient buckets of DDP.

    For now, this API only works with ``DistributedDataParallel``. It will later support
    other DP methods such as FSDP.

    Args:
        module (:class:`nn.Module`):
            Module which has been applied TP on.

    Example::
        >>> # xdoctest: +SKIP("distributed")
        >>> from torch.distributed.tensor.parallel import parallelize_module, PairwiseParallel
        >>> from torch.nn.parallel import DistributedDataParallel as DDP
        >>> from torch.distributed.tensor.parallel.ddp import pre_dp_module_transform
        >>>
        >>> # Define the module.
        >>> m = module(...)
        >>> parallelize_module(m, PairwiseParallel())
        >>> m = pre_dp_module_transform(m)
        >>> m = DDP(m)
        >>>
    """

    _localize_dtensor(module, None, None) # DTensor --> Local Tensor
    # TODO: To add test cases and ensure that it works for nested modules
    module.register_forward_pre_hook(_reconstruct_dtensor) # Local Tensor --> DTensor
    module.register_forward_hook(_localize_dtensor) # DTensor --> LocalTensor
```

## 3.3 DDP 中的 Reducer
- [reducer.cpp](https://github1s.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/reducer.hpp#L44)

**作用：** <br>
- 将参数分桶以进行规约。
- 重置分桶状态(第二个step)。
- 注册梯度钩子。
- 记录构造时的 DDP 日志数据。
- 将 DDP 的句柄传递给 SyncBatchNorm 层。

reducer 中允许自定义通信算子:

- 外部通信函数时： register_comm_hook(distributed.py) --> register_comm_hook(reducer.hpp)
- 使用pytorch自带通信函数，指定type就行：_register_builtin_comm_hook(distributed.py) --> register_builtin_comm_hook(reducer.hpp)


## 3.4 _register_fused_optim
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在 DDP 中注册优化器以在参数的梯度规约完成后立即对其进行优化。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;将优化器注册到 DDP 中，使得某个参数的优化将**在该参数的梯度规约完成后立即运行**，而不是等待所有参数的梯度规约完成。这可以根据工作负载的不同带来训练速度的提升，因为优化器可以在其他参数的梯度规约仍在进行时就开始运行。此外，这种方法还有潜力减少训练过程中的峰值内存消耗，因为它只需要逐个加载单个参数的优化器状态，而不需要一次性加载所有参数的优化器状态。<br>

**register_fused_optim : 将DDP 注册到optimizer中**
- [register_fused_optim](https://github1s.com/pytorch/pytorch/blob/main/torch/nn/parallel/distributed.py#L2049)

**_setup_in_backward_optimizers进行backward 和 optimizer的overlap** <br>
- [register_fused_optim](https://github1s.com/pytorch/pytorch/blob/main/torch/nn/parallel/distributed.py#L1018)




