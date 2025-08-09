# 代码地址
Megatron-LM 中用于做异构参数更新的模块

- [代码地址](https://github1s.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py#L14)

# 1 参数组的拆分

1. 将self.param_groups 中多个参数组里的参数全部flatten 到 params = [] 内；
2. 计算总的参数个数：params_total_numel = sum([param.numel() for param in params])
3. 计算出gpu 和 cpu 分别的参数个数：gpu_params_total_numel 和 cpu_params_total_numel；
4. offload_fraction 只是表示gpu params 的占比；
5. offload_threshold 表示的是卸载的阈值， gpu 上的元素个数要低于这个值才会停止；

**遍历param_groups，得到这么几个重要的类属性：**

- self.cpu_param_groups: cpu 上的参数组（可能和原来参数组的顺序不一致，但只是"param"改了，里面的超参数没改）；
- self.gpu_param_groups: gpu 上的参数组（可能和原来参数组的顺序不一致，但只是"param"改了，里面的超参数没改）；

- self.gpu_params_map_cpu_copy : {orig_gpu_param : cpu_param} # 需要进行D2H的转化
- self.cpu_copys_map_gpu_param : {cpu_param : orig_gpu_param} # 原来就在cpu上的param不会在此，但会在cpu_param_groups中；

- self.param_to_fp32_param : {origin_param bf16 ：高精度 fp32 param} 的映射，如果传进啦的就是fp32**则不需要**；
- self.fp32_param_to_orig_param : {fp32_param : origin_param}

- self.param_to_inner_param = {orig_param : inner_param} # inner_param 可能是gpu转过来的，也可能经过type cast, 也可能什么都没动
- self.inner_param_to_orig_param = {inner_param : orig_param}

- **self.cpu_copy_map_grad = {inner_param : inner_param_grad}, 包含从gpu 到cpu 的copy**


# 2 创建sub optimizer
## 2.1 if overlap_cpu_optimizer_d2h_h2d

- self.cpu_param_groups 中每个param 创建一个cpu_optimizers;
- sel.cpu_optimizers = self.build_cpu_optimizer_list(cpu_optimizer_cls, self.cpu_param_groups)

```python
    @staticmethod
    def build_cpu_optimizer_list(cpu_optimizer_cls, cpu_param_groups):
        """Build several cpu optimizers to enable overlap. Currently we naively
        assign each parameter to an individual optimizer.

        Args:
            cpu_optimizer_cls (Type[torch.optim.Optimizer]): A torch optimizer class
            cpu_param_groups (List[Dict[str, Any]]): The CPU parameter groups
        """
        cpu_optimizers = []

        if len(cpu_param_groups) == 0:
            return cpu_optimizers

        for group in cpu_param_groups:
            group_defaults = group.copy()
            params = group_defaults.pop("params")
            if isinstance(params, torch.Tensor):
                params = [params]
            for param in params:
                _cpu_param_group = group_defaults.copy()
                _cpu_param_group["params"] = [param]
                cpu_optimizers.append(cpu_optimizer_cls([_cpu_param_group]))
        return cpu_optimizers
```

## 2.2 if not overlap_cpu_optimizer_d2h_h2d

- 单独的一个cpu_optimizers
self.cpu_optimizers = [self.cpu_optimizer_cls(self.cpu_param_groups)]

## 2.3 创建gpu optimizer
```python
if len(self.gpu_param_groups) > 0:
    self.gpu_optimizer = self.gpu_optimizer_cls(self.gpu_param_groups)
else:
    self.gpu_optimizer = None
```

## 2.4 grad copy gpu to cpu
- 隐式D2H:  self.cpu_copy_map_grad[param].data.copy_(grad, non_blocking=True)
- **每个cpu_optimzier 的 D2H 操作都记录一个event : self._cpu_optimizer_map_data_event[optimizer] = self._d2h_stream.record_event()**
- 在 self._d2h_stream 里进行下述操作.

```python
def _set_sub_optimizer_grads(self):
    if self.param_update_in_fp32:
        for param in self.param_to_fp32_param:
            if param in self.gpu_params_map_cpu_copy:
                # Skip if the param is offloaded to CPU, it should be handled
                # in the following part.
                continue
            fp32_param = self.param_to_fp32_param[param]
            grad = getattr(param, "decoupled_grad", param.grad)
            if grad is not None:
                fp32_param.grad = grad.to(fp32_param.dtype)
                fp32_param.requires_grad = True
            else:
                fp32_param.requires_grad = False

    # Sync the grads from GPU to CPU.
    for optimizer in self.cpu_optimizers:
        for param in _param_generator(optimizer):
            gpu_param = self.cpu_copys_map_gpu_param[param] # mtn : 一定有gpu_param
            grad = getattr(gpu_param, "decoupled_grad", gpu_param.grad) # use_precision_aware_optimizer
            if grad is None:
                param.requires_grad = False
                continue

            param.requires_grad = False
            if param not in self.cpu_copy_map_grad:
                self.cpu_copy_map_grad[param] = torch.empty(
                    param.shape, dtype=param.dtype, pin_memory=self.pin_cpu_grads, device="cpu"
                )
                param.grad = self.cpu_copy_map_grad[param]

            self.cpu_copy_map_grad[param].data.copy_(grad, non_blocking=True) # grad 从 gpu copy到cpu
        self._cpu_optimizer_map_data_event[optimizer] = self._d2h_stream.record_event()
```

## 2.5 param copy back to gpu
- self._d2h_stream 上所有event的等待： self._d2h_stream.record_event().wait(torch.cuda.current_stream())

```python
def _register_param_copy_back_gpu_hook(self):
    def param_copy_back_gpu_hook_closure():
        def param_copy_back_gpu_hook(optimizer, args, kwargs):
            self._h2d_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._h2d_stream):
                for param in _param_generator(optimizer):
                    gpu_param = self.cpu_copys_map_gpu_param[param]
                    gpu_param.data.copy_(param.data, non_blocking=True)
            self._d2h_stream.record_event().wait(torch.cuda.current_stream())

        return param_copy_back_gpu_hook

    def fp32_param_copy_back_gpu_hook_closure():
        def fp32_param_copy_back_gpu_hook(optimizer, args, kwargs):
            for group in self.param_groups:
                for param in group["params"]:
                    if param in self.gpu_params_map_cpu_copy:
                        # Skip if the param is offloaded to GPU, it has been
                        # copied back in the previous hook.
                        continue

                    if param in self.param_to_fp32_param:
                        fp32_param = self.param_to_fp32_param[param]
                        param.data.copy_(fp32_param.data)

        return fp32_param_copy_back_gpu_hook

    for optimizer in self.sub_optimizers:
        if optimizer is not self.gpu_optimizer:
            optimizer.register_step_post_hook(param_copy_back_gpu_hook_closure())
        elif self.param_update_in_fp32:
            optimizer.register_step_post_hook(fp32_param_copy_back_gpu_hook_closure())
```

## 2.6 根据sub_optimzier 的 state 更新 self.state
```python
def _sync_sub_optimizers_state_to_hdo(self):
    """
    Update HDO state attribute to sub-optimizers.
    """

    # optimizer.state:
    # {
    #    torch.nn.Parameter: {
    #        str: Any,
    #    },
    #    ...
    # }
    new_state = defaultdict(dict)
    for optimizer in self.sub_optimizers:
        for param in optimizer.state:
            orig_param = self.inner_param_to_orig_param[param]
            new_state[orig_param] = optimizer.state[param]
            if self.param_update_in_fp32:
                new_state[orig_param]["master_param"] = param
    self.state = new_state
```

## 2.7 _sync_hdo_param_groups_to_sub_optimizers
- 找到原来的param group, 想办法利用原来param group的超参数，而 "params" 用现在的；

```python
def _sync_hdo_param_groups_to_sub_optimizers(self):
    """Sync HDO new param_groups attribute (e.g. lr, wd, etc.) to sub-optimizers."""
    param_in_param_group_index = {} # param --> innder_param --> (group_id, param_id)
    for i, group in enumerate(self.param_groups):
        for p_id, param in enumerate(group["params"]):
            inner_param = self.param_to_inner_param[param]
            param_in_param_group_index[inner_param] = (i, p_id)

    for optimizer in self.sub_optimizers:
        new_param_groups = []
        for group in optimizer.param_groups:
            new_group = group.copy()
            # After sync-up the sub-optimizer last update, we need to sync-up the
            # HDO new param_groups attributes to the sub-optimizer.
            assert len(group["params"]) > 0, "param_groups should not be empty"
            group_id, _ = param_in_param_group_index[group["params"][0]]
            update_group_attrs = self.param_groups[group_id].copy()
            del update_group_attrs["params"]
            new_group.update(update_group_attrs)

            new_param_groups.append(new_group)
        optimizer.param_groups = new_param_groups
```
