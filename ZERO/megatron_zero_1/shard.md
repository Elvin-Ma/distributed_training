# 0 buffer 的建立

- param_and_grad_dtype_to_params = {(param_dtype, grad_dtype) : [param1, param2, ...]}
- param_and_grad_dtype_to_indices = {(torch.bfloat16, torch.float32): [0, 3], (torch.uint8, torch.float32): [1, 2]}

```python
        def _allocate_buffers_for_parameters(
            input_params, data_parallel_group, gradient_scaling_factor
        ):
            param_and_grad_dtype_to_params = {}
            param_and_grad_dtype_to_offsets = {}
            param_and_grad_dtype_to_indices = {}

            # Group parameters by their gradient type.
            for param in input_params:
                assert param.requires_grad

                param_dtype = param.dtype
                if is_float8tensor(param):
                    # Currently TE's Float8Tensor is a wrapper of torch.Tensor. It has a "fake"
                    # dtype (usually a higher precision dtype such as bfloat16), but its actual
                    # data is stored in the form of a torch uint8 tensor within the Float8Tensor's
                    # ".data" attribute. Therefore, when creating the param buffer for fp8 params,
                    # it is necessary to use torch.uint8, not the "fake" dtype got from
                    # "param.dtype".
                    param_dtype = torch.uint8
                grad_dtype = torch.float if self.ddp_config.grad_reduce_in_fp32 else param.dtype

                params = param_and_grad_dtype_to_params.get((param_dtype, grad_dtype), [])
                params.append(param)
                param_and_grad_dtype_to_params[(param_dtype, grad_dtype)] = params

                # Get the index of each param among the params with same dtype, if a param is fp8,
                # use its "fake" high precision dtype to find which params have same dtype with it.
                # For example:
                #     Case 1:
                #         params = [p1(bf16), p2(bf16), p3(bf16), p4(bf16)]
                #         param_and_grad_dtype_to_indices = {
                #             (torch.bfloat16, torch.float32): [0, 1, 2, 3],
                #         }
                #     Case 2:
                #         params = [p1(bf16), p2(fp8), p3(fp8), p4(bf16)]
                #         param_and_grad_dtype_to_indices = {
                #             (torch.bfloat16, torch.float32): [0, 3],
                #             (torch.uint8, torch.float32): [1, 2],
                #         }
                # We need these indices to load a non-native-fp8 checkpoint in native-fp8 mode.
                offset = param_and_grad_dtype_to_offsets.get((param.dtype, grad_dtype), 0)
                param_and_grad_dtype_to_offsets[(param.dtype, grad_dtype)] = offset + 1
                indices = param_and_grad_dtype_to_indices.get((param_dtype, grad_dtype), [])
                indices.append(offset)
                param_and_grad_dtype_to_indices[(param_dtype, grad_dtype)] = indices

            if not config.calculate_per_token_loss:
                target_gradient_scaling_factor = 1.0 / parallel_state.get_data_parallel_world_size(
                    with_context_parallel=True
                )
                if self.ddp_config.average_in_collective:
                    if self.ddp_config.num_distributed_optimizer_instances == 1:
                        # Collective is averaging gradients in collective with data_parallel_group.
                        assert (
                            gradient_scaling_factor
                            / torch.distributed.get_world_size(group=data_parallel_group)
                            == target_gradient_scaling_factor
                        )
                    else:
                        # For non-expert parameters, gradient_scaling_factor is 1.
                        # For expert parameters, gradient_scaling_factor is edp_size/dp_size.
                        assert (gradient_scaling_factor == 1) or (
                            gradient_scaling_factor
                            == (
                                parallel_state.get_expert_data_parallel_world_size()
                                / parallel_state.get_data_parallel_world_size(
                                    with_context_parallel=True
                                )
                            )
                        )
                else:
                    assert gradient_scaling_factor == target_gradient_scaling_factor

            # Allocate the grad buffers and map the grads.
            buffers = []
            for (param_dtype, grad_dtype), params in param_and_grad_dtype_to_params.items():
                buffers.append(
                    _ParamAndGradBuffer(
                        self.ddp_config,
                        param_dtype,
                        grad_dtype,
                        params,
                        data_parallel_group,
                        self.bucket_size,
                        param_to_name,
                        gradient_scaling_factor,
                        param_and_grad_dtype_to_indices[(param_dtype, grad_dtype)],
                    )
                )
```

# 1 _ParamAndGradBuffer

## 1.1 pad 相关

param_start_index 需要 128 字节对齐，用bfloat16 的话这里就是 64 个元素对齐.

全局内存访问以128字节为基本事务单位（NVIDIA文档[1]和MT-GPU架构均遵循此设计）。若数据起始地址未对齐到128字节边界，会导致：

- 需要额外内存事务
- 降低内存吞吐量（可能下降30%+）
- 引发bank conflicts（多核访问冲突）

```python
def _pad_start_of_param_if_needed(param_start_index: int) -> int:
    """
    Pads start index of param if using distributed optimizer (to ensure "good" alignment).
    """
    if self.ddp_config.use_distributed_optimizer:
        # Ensure that params start at 128-byte aligned addresses (64 values
        # since params are >= 16-bit precision).
        return _pad(param_start_index, 64)
    return param_start_index
```

**bucket_end_index** padding

```python
    def _pad_end_of_bucket_if_needed(bucket_end_index: int) -> int:
        """
        Pads end index of bucket if using distributed optimizer (to ensure uniform sharding).
        """
        if self.ddp_config.use_distributed_optimizer:
            # Workaround for TE bug causing cuBLAS to pick an incompatible algorithm.
            # This also helps cuBLAS pick more efficient algorithms for GEMMs.
            # We now ensure that all buckets start at a memory address that is 256-byte
            # aligned (128 values since params and grads use >= 16-bit precision).
            if self.ddp_config.pad_buckets_for_high_nccl_busbw:
                # Make sure the bucket size is divisible by a large power of 2 (2^16) to
                # ensure NCCL collectives have high bus bandwidth at large DP counts,
                # since NCCL message size (which for ring algorithms is bucket_size /
                # dp_size) apparently needs to be divisible by a power of 2 for high busbw.
                bucket_size_divisor = math.lcm(self.data_parallel_world_size, 128, 2**16)
            else:
                bucket_size_divisor = math.lcm(self.data_parallel_world_size, 128) # 这里可能会改
            return _pad(bucket_end_index, bucket_size_divisor)
        return bucket_end_index
```

## 1.2 统计 bucket 和 每个param 的范围

**param 不会被bucket 切分**
param_start_index : param 在 **buffer** 上的起始index
bucket_start_index: bucket 在 **buffer** 上的起始index
per_bucket_numel_unpadded: 每个bucket的未padding的元素个数

目的得到两个对应的映射关系：<br>
self.param_index_map = {param: (param_start_index, param_end_index, bucket_id)}
self.bucket_indices = [(bucket_start_index, bucket_end_index)]

- PP stage 0 : 按照 bucket_size 来分配bucket
- shared embedding weight: 需要单独分配bucket
- PP stage > 0: 把整个stage 里所有params 放到同一个bucket


```python
    # Param -> location in buffer mapping (used in dist. optimizer).
    self.param_index_map = {param: (param_start_index, param_end_index, bucket_id)}
    self.bucket_indices = [(bucket_start_index, bucket_end_index)]

    for param in params[::-1]:
            # Iterate through parameters in reverse order to roughly follow backprop order.

            this_numel = param.data.nelement()
            # 64 元素对齐，128 bytes 对齐.
            param_start_index = _pad_start_of_param_if_needed(param_start_index)

            # Create bucket with collected parameters if current param needs its own bucket.
            if _does_param_require_new_bucket(param):
                # We are creating a bucket for the already accumulated parameters, whose params
                # end at the current param_start_index.
                if self.ddp_config.use_distributed_optimizer:
                    # Make sure new bucket is appropriately padded.
                    if param_start_index % self.data_parallel_world_size != 0:
                        # 确保之前的bucket 进行了正确的padding, 这里一般不会出问题
                        param_start_index = _pad_end_of_bucket_if_needed(param_start_index)

                # 如果bucket_params不为空，则结束之前bucket 并重新开始
                if len(bucket_params) > 0:
                    bucket_end_index = _update_bucket_metadata(param_start_index)

            param_end_index = param_start_index + this_numel
            self.param_index_map[param] = (param_start_index, param_end_index, bucket_id)
            bucket_params.add(param)

            # pp rank 0 或 shared_embedding 会创建多个bucket
            # If we have enough elements already or the current param is part of the shared
            # embedding layer and needs a separate bucket, form a new bucket.
            if (
                bucket_size is not None and (param_end_index - bucket_start_index) >= bucket_size
            ) or _does_param_require_new_bucket(param):
                bucket_end_index = _update_bucket_metadata(param_end_index)
                param_start_index = bucket_end_index
            else:
                # pp rank > 0 一个buffer 生成一个bucket
                param_start_index = param_end_index

        # 最后一个bucket_params 也得结束
        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:
            bucket_end_index = _update_bucket_metadata(param_end_index)
```

## 1.3 创建 buffer
- self.numel : 是按照 padding 后的个数统计的，每个bucket 都会有padding;
- self.numel_unpadded : 排除掉padding的总元素个数;
- 分布式优化器下 self.numel 需要被 data_parallel_world_size 整除.
- param 的 dtype : bfloat16
- grad_data 的 dtype : float32

```python
    # Next, create underlying storage for buffer (with numel elements that includes
    # padding as necessary).
    self.numel = bucket_end_index
    self.numel_unpadded = sum(per_bucket_numel_unpadded)
    assert self.numel_unpadded <= self.numel
    if self.ddp_config.use_distributed_optimizer:
        assert self.numel % self.data_parallel_world_size == 0
    else:
        assert self.numel == self.numel_unpadded

    self.param_data = None
    # Only re-map param tensors if using distributed optimizer.
    if self.ddp_config.use_distributed_optimizer:
        self.param_data = torch.zeros(
            self.numel,
            dtype=self.param_dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )
    self.grad_data = torch.zeros(
        self.numel,
        dtype=self.grad_dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )
```

## 1.4 copy data 并创建 bucket

```python
for param in params[::-1]:
    param_start_index, param_end_index, bucket_id = self.param_index_map[param]

    # Assign param.data to appropriate segment of self.param_data.
    if self.param_data is not None:
        old_param_data = param.data
        new_param_data = self._get(
            param.data.shape, param_start_index, buffer_type=BufferType.PARAM
        )
        if is_float8tensor(param):
            param._data = new_param_data
        else:
            param.data = new_param_data
        assert old_param_data._base is None
        # Copy tensor values (from initialization or checkpoint).
        param.data.detach().copy_(old_param_data)
        del old_param_data

    param.main_grad = self._get(
        param.data.shape, param_start_index, buffer_type=BufferType.GRAD
    ) # param.grad vs param.main_grad

    # bucket_id 更新时会到这里
    if bucket_id != cur_bucket_id:
        # 根据现在的param_start_index, 计算前一个bucket的bucket_end_index
        bucket_end_index = _pad_end_of_bucket_if_needed(param_start_index)
        self.buckets.append(
            self._new_bucket(
                bucket_params=bucket_params,
                start_index=bucket_start_index,
                end_index=bucket_end_index,
                numel_unpadded=per_bucket_numel_unpadded[cur_bucket_id],
                bucket_id=cur_bucket_id,
            )
        )
        bucket_start_index = bucket_end_index
        bucket_params = []
        assert cur_bucket_id + 1 == len(self.buckets)
        assert bucket_id == cur_bucket_id + 1
        cur_bucket_id = bucket_id
    bucket_params.append(param)
```

## 1.5 创建bucket 的过程

```python
    def _new_bucket(
        self,
        bucket_params: List[torch.nn.Parameter],
        start_index: int,
        end_index: int,
        numel_unpadded: int,
        bucket_id: int,
    ) -> _ParamAndGradBucket:
        """
        Helper function that creates a new bucket. Also updates param->bucket mapping.
        """

        # Assert that indices are correctly padded (if needed), and that bucket
        # position is same as originally computed.
        if self.ddp_config.use_distributed_optimizer:
            assert start_index % self.data_parallel_world_size == 0
            assert end_index % self.data_parallel_world_size == 0
        assert (start_index, end_index) == self.bucket_indices[bucket_id]

        # Get appropriate view into global _ParamAndGradBuffer.
        bucketed_param_data = None
        if self.param_data is not None:
            bucketed_param_data = self._get(
                torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.PARAM
            )
        bucketed_grad_data = self._get(
            torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.GRAD
        )
        bucket = _ParamAndGradBucket(
            params=bucket_params,
            param_data=bucketed_param_data,
            grad_data=bucketed_grad_data,
            offset=start_index,
            numel_unpadded=numel_unpadded,
            gradient_scaling_factor=self.gradient_scaling_factor,
            bucket_id=bucket_id,
        )
        for bucket_param in bucket_params:
            assert bucket_param not in self.param_to_bucket
            self.param_to_bucket[bucket_param] = bucket

        return bucket
```

# 2  从 _ParamAndGradBuffer(buffers) 里得到 _ParamAndGradBucketGroup

**buffers 里的核心时一个个的 buckets, 将不同buffer 里的buckets 重新组合得到 _ParamAndGradBucketGroup.**

**bucket_groups** 里的参数是从模型后到模型前排列的.

**self.param_to_bucket_group[param] = bucket_group**

在某些场景下，我们希望将来自不同缓冲区buffer（按(param_dtype, grad_dtype) 划分）的桶（buckets）归为一组(_ParamAndGradBucketGroup)，以便对它们的通信进行聚合.

例如，当模型中同时存在 fp8 权重和 bf16 偏置，且启用了虚拟流水线并行时(vpp)，每个模型块都会有一个 fp8 桶和一个 bf16 桶，这会使通信核的数量翻倍。此外，由于使用了 CUDA_DEVICE_MAX_CONNECTIONS=1，多个连续的通信将阻碍通信核与计算核的重叠执行.

分组策略如下：

- 如果 force_single_bucket_group 为 True，将所有缓冲区中的所有桶放入单个桶组中。
- 如果 force_single_bucket_group 为 False，当输入缓冲区中**没有 fp8 缓冲区时，每个桶组仅包含一个桶**。
- 如果 force_single_bucket_group 为 False，当**使用 fp8 参数时**，将所有`非 fp8 桶合并到最后一个 fp8 桶组中`。
  - 由于非 fp8 参数（通常是各层的**偏置**）相对较小，它们很可能被分组到单个非 fp8 桶中。
  - fp8 桶从模型的末尾开始，即第一个桶对应模型的末尾，而`最后一个桶对应模型的开头`。
  - 如果我们将非 fp8 桶与第一个 fp8 桶合并，就无法在模型末尾的反向传播完成后启动 reduce-scatter 来同步梯度。这是因为我们需要等待来自开头层的非 fp8 参数获得它们的梯度。
  - 将非 fp8 桶与最后一个 fp8 桶合并有助于避免此问题。

```python
def partition_buckets(
    buffers: List[_ParamAndGradBuffer], force_single_bucket_group: bool = False
) -> List[_ParamAndGradBucketGroup]:

    if len(buffers) == 0:
        return []

    dtype_to_buffer_map = {}
    for buffer in buffers:
        dtype = buffer.param_dtype
        # Make sure that the param_dtype of any two buffers is different.
        assert dtype not in dtype_to_buffer_map
        dtype_to_buffer_map[dtype] = buffer

    # Case 1: Put all buckets into a single bucket group if force_single_bucket_group is True.
    if force_single_bucket_group:
        buckets = []
        ddp_config = buffers[0].ddp_config
        data_parallel_group = buffers[0].data_parallel_group
        data_parallel_world_size = buffers[0].data_parallel_world_size
        for buffer in buffers:
            assert ddp_config == buffer.ddp_config
            assert data_parallel_group == buffer.data_parallel_group
            assert data_parallel_world_size == buffer.data_parallel_world_size
            buckets.extend(buffer.buckets)

        bucket_group = _ParamAndGradBucketGroup(
            buckets, ddp_config, data_parallel_group, data_parallel_world_size
        )
        return [bucket_group]

    if torch.uint8 not in dtype_to_buffer_map:
        # Case 2: When there is no fp8 buffer in the input buffers, let each bucket group have
        #         only one bucket.
        bucket_groups = []
        for buffer in buffers:
            for bucket in buffer.buckets:
                bucket_groups.append(
                    _ParamAndGradBucketGroup(
                        [bucket],
                        buffer.ddp_config,
                        buffer.data_parallel_group,
                        buffer.data_parallel_world_size,
                    )
                )
        return bucket_groups
    else:
        # Case 3: When using fp8 params, merge all non-fp8 buckets into the last fp8 bucket group.
        non_fp8_buckets = []
        for buffer in buffers:
            if buffer.param_dtype != torch.uint8:
                for bucket in buffer.buckets:
                    non_fp8_buckets.append(bucket)

        bucket_groups = []
        fp8_buffer = dtype_to_buffer_map[torch.uint8]
        for bucket in fp8_buffer.buckets:
            if len(bucket_groups) == len(fp8_buffer.buckets) - 1:
                # The last bucket group.
                group_buckets = [bucket] + non_fp8_buckets
            else:
                # The first N-1 bucket groups.
                group_buckets = [bucket]
            bucket_groups.append(
                _ParamAndGradBucketGroup(
                    group_buckets,
                    buffer.ddp_config,
                    buffer.data_parallel_group,
                    buffer.data_parallel_world_size,
                )
            )
        return bucket_groups
```

# 3 反向时的梯度累计

param.main_grad 就是之前建立的 buffer, 是一个高精度的grad 会一直存在于整个训练过程中。低精度的grad 用完后会释放。

因此这里是个典型的 ZERO-1 场景.

```python
    for param in self.module.parameters():
        if param.requires_grad:
            # Expand so we get access to grad_fn.
            param_tmp = param.expand_as(param)
            # Get the gradient accumulator function.
            grad_acc = param_tmp.grad_fn.next_functions[0][0]
            grad_acc.register_hook(self._make_backward_post_hook(param))
            self.grad_accs.append(grad_acc)

    def _make_backward_post_hook(self, param: torch.nn.Parameter):
        """
        Creates a backward post-hook to dispatch an all-reduce / reduce-scatter when
        ready (i.e., when all grads in a bucket have been computed in all microbatches
        in a batch).
        """

        def hook(*unused):
            if is_graph_capturing():
                return

            if param in self.param_to_bucket_group:
                assert param.requires_grad
                if self.ddp_config.overlap_grad_reduce:
                    assert (
                        param.grad is not None
                    ), 'param.grad being None is not safe when overlap_grad_reduce is True'
                if param.grad is not None and (
                    not param.grad_added_to_main_grad or getattr(param, 'zero_out_wgrad', False)
                ):
                    param.main_grad.add_(param.grad.data)
                param.grad = None

                if self.ddp_config.overlap_grad_reduce:
                    self.param_to_bucket_group[param].register_grad_ready(param)

        return hook
```

# 4 Optimizer shard param

每个buffer 里有两个重要tensor : self.param_data and self.grad_data.

## 4.1 构建从参数引用到梯度缓冲区分片范围的映射
该方法会构建从参数引用到**梯度缓冲区分片范围**的映射，且此映射特定于每个数据并行（DP，Data-Parallel）进程组等级（rank）所 “拥有” 的参数集合。每个梯度缓冲区（会被填充为数据并行全局大小（DP-world-size）的整数倍）在`概念上(conceptually)`会被划分为与`DP-world-size`数量相等的连续区域，其中每个 DP 进程组等级 “拥有” 一个连续区域。此处的 “拥有” 意味着该 DP 进程组等级负责对梯度的相关子集执行归约（reduce）操作，并更新参数的相关子集。

需要注意的是，梯度缓冲区的这种概念性划分`不考虑参数边界`，因此默认**每个创建的范围都指向完整参数的一个分片（或子集）**。最易于理解的方式是：对于所有 “模型到主进程”（model-to-main）和 “主进程到模型”（main-to-model）的操作，每个 DP 进程组等级仅对梯度缓冲区的视图（view）执行操作（即归约、聚集）。

该方法会创建四种范围：<br>
- 参数在**整个梯度缓冲区中的范围**（即全局索引，world index）;
- 参数在**对应梯度桶（grad bucket）缓冲区中的范围**;
- 参数在 DP 进程组等级**本地rank 切片下的梯度缓冲区**中的范围;
- 参数在其自身内部的范围（即**参数自身的分片**，shard）.

提取当前DP rank 的 param. 如果param 在当前shard 中会得到如下map, 所有buffer 的map 被打包到一个 self.gbuf_ranges 变量里: <br>

```python
    param_range_map = {}
    param_range_map[param] = {
        "gbuf_world": param_world_range,
        "gbuf_world_in_bucket": param_world_range_in_bucket,
        "gbuf_local": param_local_range,
        "param": sub_param_range,
    }
    data = {"param_map": param_range_map}

    self.gbuf_ranges.append(data)
```

- 相关代码

```python
    # 获取全局buffer
    self.buffers = list(itertools.chain(*per_model_buffers.values()))
    # 从DDP 拿到的buffers
    self.per_model_buffers = per_model_buffers
    # 通信组
    self.data_parallel_group = data_parallel_group
    self.data_parallel_group_gloo = data_parallel_group_gloo

    # {buffer 索引 : 模型索引} 的 映射
    self.gbuf_idx_to_model_idx_map = {}
    # {model index : bucket group} 的 映射
    self.per_model_bucket_groups = {}


    self.gbuf_ranges = []
    self.per_bucket_numel = [] # 每个bucket的元素个数
    self.per_bucket_numel_unpadded = [] # 每个bucket的元素个数, 不包含padding

    for buffer in self.buffers:
        self.per_bucket_numel.append(
            {
                (buffer.param_dtype, buffer.grad_dtype): [
                    bucket.grad_data.numel() for bucket in buffer.buckets
                ]
            }
        )
        self.per_bucket_numel_unpadded.append(
            {
                (buffer.param_dtype, buffer.grad_dtype): [
                    bucket.numel_unpadded for bucket in buffer.buckets
                ]
            }
        )
        self.gbuf_ranges.append(self._build_gbuf_range_map(buffer))


    # 针对每个数据并行（DP，Data-Parallel）进程组等级（rank）：
    # 确定其在参数与梯度缓冲区（param_and_grad_buffer）中的分片范围（shard ranges）。
    # 每个 DP 等级会保存所有其他 DP 等级的范围信息：
    # 此举旨在为归约散射（reduce-scatter）和全聚集（all-gather）操作创建参数（args）
    @classmethod
    def _build_model_gbuf_range(cls, param_and_grad_buffer: _ParamAndGradBuffer, bucket_index: int):
        """
        Build mapping between params and their grad buffers.

        This method does the initial setup for the method above. This setup
        includes determining the shard ranges into the param_and_grad_buffer
        for each data-parallel (DP) rank. Each DP rank keeps range info for
        all other DP ranks, for the purpose of creating args for
        reduce-scatter and all-gather.
        """

        data_parallel_rank = torch.distributed.get_rank(param_and_grad_buffer.data_parallel_group)
        data_parallel_world_size = param_and_grad_buffer.data_parallel_group.size()

        bucket = param_and_grad_buffer.buckets[bucket_index]
        gbuf_size = bucket.grad_data.numel() # 包含padding 的 elements

        # 确保gbuf_size 必须可以被 data_parallel_world_size 整除
        # 这一点在bucket 划分时就已经确定了，但如果data_parallel_world_size 发生变化， 这里就会被check 住
        assert (
            gbuf_size % data_parallel_world_size == 0
        ), f"Each bucket's buffer size should be divisible by {data_parallel_world_size}"
        # 计算出每个DP rank 的 gbuf_range_size
        max_gbuf_range_size = gbuf_size // data_parallel_world_size

        # 每个bucket 在所有DP rank 的 range
        # All world ranges (i.e., across all data parallel ranks).
        gbuf_world_all_ranges = []
        for r in range(data_parallel_world_size):
            # Compute start of chunk in this bucket.
            gbuf_world_start = r * max_gbuf_range_size
            gbuf_world_end = min(gbuf_size, gbuf_world_start + max_gbuf_range_size)
            # Add bucket's offset in grad buffer. 计算出每个rank 在 buffer 里的range
            gbuf_world_range = Range(
                gbuf_world_start + bucket.offset, gbuf_world_end + bucket.offset
            )
            gbuf_world_all_ranges.append(gbuf_world_range)

        # 提取出当前rank 在 buffer 里的range
        gbuf_world_range = gbuf_world_all_ranges[data_parallel_rank]

        # 构建每个param 的range
        # param_index_map 为每个param 在buffer 里的起始位置
        #
        param_range_map = cls._build_model_gbuf_param_range_map(
            param_and_grad_buffer.param_index_map, gbuf_world_range, bucket.offset
        )

        # Group into dict.
        data = {"param_map": param_range_map}

        return data

    @classmethod
    def _build_model_gbuf_param_range_map(
        cls,
        param_world_index_map: Dict[torch.nn.Parameter, Tuple],
        gbuf_world_range: Range,
        bucket_offset: int,
    ):
        # Param range map.
        param_range_map = {}
        for param, param_world_indexes in param_world_index_map.items():

            # Param range.
            param_world_start, param_world_end, _ = param_world_indexes
            param_local_start = max(0, param_world_start - gbuf_world_range.start)
            param_local_end = min(gbuf_world_range.size, param_world_end - gbuf_world_range.start)

            # Add param, if within local gbuf range.
            if param_local_end > param_local_start:
                param_local_range = Range(param_local_start, param_local_end)
                param_world_range = param_local_range.normalize(
                    param_local_start + gbuf_world_range.start
                )
                param_world_range_in_bucket = Range(
                    param_world_range.start - bucket_offset, param_world_range.end - bucket_offset
                )
                sub_param_start = max(0, gbuf_world_range.start - param_world_start)
                sub_param_range = param_local_range.normalize(sub_param_start)
                param_range_map[param] = {
                    "gbuf_world": param_world_range,
                    "gbuf_world_in_bucket": param_world_range_in_bucket,
                    "gbuf_local": param_local_range,
                    "param": sub_param_range,
                }

        return param_range_map
```

## 4.2 param 到 gbuf + bucket 的映射

self.model_param_gbuf_map = {param : (gbuf_index, dtype, bucket_index)}

```python
    @classmethod
    def _build_model_param_gbuf_map(
        cls, gbuf_ranges: List[Dict]
    ) -> Dict[torch.nn.Parameter, Tuple]:
        """
        Create a reverse of the gbuf_ranges, for referencing in opposite direction.
        """
        param_gbuf_map = {}
        for gbuf_index, gbuf_range_map in enumerate(gbuf_ranges):
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_map.items():
                for bucket_index, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):
                    for param, _ in gbuf_range_map["param_map"].items():
                        assert param not in param_gbuf_map, (
                            "Param should not be in param_gbuf_map; each param only belongs "
                            "to a single bucket."
                        )
                        param_gbuf_map[param] = (gbuf_index, dtype, bucket_index)
        return param_gbuf_map
```

## 4.3 optimizer ranges 的获取

**提取 Local DP rank 的 param**
local_param_group_map = {} # {param: (group_index, param_index)}
group_ranges = [{params: []}, {params: []}, ...]

```python
    @classmethod
    def _build_optimizer_group_ranges(cls, param_groups: List[Dict], gbuf_ranges: List[Dict]):
        """
        Create optimizer groups.

        Given the set of parameter shard ranges that are owned by the current
        data-parallel (DP) rank, gather the set of parameters that will be
        used (in the method below) to create the current DP's optimizer
        groups.
        """

        # 参数组映射（Param group map）
        # 全局参数组映射（World param group map）
        # - 存储所有参数的 {模型参数（model_parameter）: 组索引（group_index）} 映射关系。
        # 用途: 构建当前 DP 进程组等级（this DP rank）的参数本地映射（local mapping）。
        world_param_group_map = {}
        for group_index, group in enumerate(param_groups):
            for param in group["params"]:
                assert param.requires_grad
                world_param_group_map[param] = group_index

        # 优化器组范围与参数 - 组映射
        # - 构建两组映射关系：
        #   一组是 “参数到其所属优化器组的索引及在组内顺序” 的映射。
        #   另一组是 “优化器组到其包含的参数” 的映射;
        # 其中，优化器组的索引和参数在组内的顺序对于 checkpoint（检查点）的保存与加载尤为重要。
        local_param_group_map = {} # {param: (group_index, param_index)}
        # [{params: []}, {params: []}, ...]}, {}, ...]
        group_ranges = [{"params": []} for _ in param_groups]
        for gbuf_range_map in gbuf_ranges:
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_map.items():
                for gbuf_range_map in gbuf_range_map_for_all_buckets:
                    for param in gbuf_range_map["param_map"]:
                        group_index = world_param_group_map[param]
                        group_range = group_ranges[group_index]
                        group_range["params"].append(param)
                        local_param_group_map[param] = (group_index, len(group_range["params"]) - 1)

        # Squeeze zero-size group ranges.
        # 查找并保存当前DP rank 中param 原来的param_group
        for group_index, group_range in enumerate(group_ranges):
            group_range["orig_group"] = param_groups[group_index]
            group_range["orig_group_idx"] = param_groups[group_index]

        return local_param_group_map, group_ranges
```

## 4.4 提取local param 并 构建 main_param

### 4.4.1 添加 main_param
```python
    # Add main_param field to each parameter. We will use this fp32 copy to compute
    # the param norm.
    # For parameters with optimizer state on this rank, None will be overwritten by
    # the corresponding sharded main_param tensor.
    for param_group in self.optimizer.param_groups:
        # For all the parameters in this group.
        for param in param_group['params']:
            if param.requires_grad:
                # fp32 copy only needed for 16-bit parameters.
                if param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:
                    param.main_param = None
                    param.main_param_sharded = True
```

### 4.4.2 构建 main_param 和 相应各参数版本

- model_float16_groups = [] # 当前 DP rand 的 原始 bf16 参数
- model_fp32_groups = []    # 当前 DP rank 的 原始 fp32 参数
- shard_float16_groups = [] # 当前 DP rank 的 原始 bf16 参数的切片
- shard_fp32_groups = []    # 当前 DP rank 的 原始 fp32 参数的切片
- shard_fp32_from_float16_groups = [] # 当前 DP rank 的 bf16 参数切片的高精度版本 即: main_param

```python
    if model_param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:
        # Generate sharded model param.
        shard_model_param = model_param.detach().view(-1)[
            param_range.start : param_range.end
        ]
        # 高精度main_param 的获取
        shard_main_param = shard_model_param.clone().float()
        model_param.main_param = shard_main_param
        model_param.main_param_sharded = True

        # Add to group.
        model_float16_params_this_group.append(model_param)
        shard_float16_params_this_group.append(shard_model_param)
        shard_fp32_from_float16_params_this_group.append(shard_main_param)

    # fp32 params.
    elif model_param.type() == 'torch.cuda.FloatTensor':
        shard_model_param = model_param.view(-1)[param_range.start : param_range.end]
        model_fp32_params_this_group.append(model_param)
        shard_fp32_params_this_group.append(shard_model_param)
        tensor_parallel.copy_tensor_model_parallel_attributes(
            shard_model_param, model_param
        )
        if hasattr(model_param, 'shared'):
            shard_model_param.shared = model_param.shared
```

### 4.4.3 替换原始param_groups

- 将原始参数组中的参数换为切片后的参数

```python
group_range["orig_group"]["params"] = [
    *shard_fp32_params_this_group,
    *shard_float16_params_this_group,
]
```

- optimizer 的参数组替换: 只替换params, 其它超参没有替换

```python
    self.optimizer.param_groups = [g["orig_group"] for g in self.opt_group_ranges]
    self.optimizer.load_state_dict(self.optimizer.state_dict())
```

## 4.5 相关数据的搬迁

### 4.5.1 main_grad 需要切片到 shard_main_grad 并交给 optimizer 里的 shard_main_param

```python
    def _copy_model_grads_to_main_grads(self):
        """
        Copy model grads to main grads.

        Since this step follows a reduce-scatter through the DDP's grad
        buffer, this method is responsible for copying the updated grads
        from the grad buffer to the main shard's grad field.
        """
        if self.is_stub_optimizer:
            return

        if self.ddp_config.use_custom_fsdp:
            return

        # Utility method for copying group grads.
        def copy_group_grads(model_groups, shard_main_groups):
            for model_group, shard_main_group in zip(model_groups, shard_main_groups):
                for model_param, shard_main_param in zip(model_group, shard_main_group):

                    param_range_map = self._get_model_param_range_map(model_param)
                    param_range = param_range_map["param"]
                    assert param_range.size == shard_main_param.nelement()

                    model_grad = model_param.main_grad
                    shard_model_grad = model_grad.view(-1)[param_range.start : param_range.end]
                    if self.config.use_precision_aware_optimizer:
                        # Pytorch requires a param and its' grad to be the same dtype, but we want
                        # their types to be different in precision-aware optimizer. So we use
                        # ".decoupled_grad" to replace ".grad".
                        # Note that this requires corresponding modifications in the optimizer (Let
                        # the optimizer read gradients from ".decoupled_grad" instead of ".grad").
                        shard_main_param.decoupled_grad = shard_model_grad
                    else:
                        shard_main_param.grad = shard_model_grad.float()

        # Copy model groups to shard groups.
        if self.config.use_precision_aware_optimizer:
            copy_group_grads(self.model_float16_groups, self.shard_float16_groups)
            copy_group_grads(self.model_fp32_groups, self.shard_fp32_groups)
        else:
            copy_group_grads(self.model_float16_groups, self.shard_fp32_from_float16_groups)
            copy_group_grads(self.model_fp32_groups, self.shard_fp32_groups)
```

### 4.5.2 参数更新结束后，需要将 shard_main_param 的梯度更新到 model_param 中

```python
    def _copy_main_params_to_model_params(self):
        """
        Copy main params to model params.

        Since this step is followed by an all-gather through the DDP's grad
        buffer, this method is responsible for copying the updated params
        from the main shards into the correct position in the grad buffer.
        """
        if self.is_stub_optimizer:
            return

        if self.ddp_config.use_custom_fsdp:
            for model_chunk in self.model_chunks:
                model_chunk.param_and_grad_buffer.copy_main_weights_to_model_weights()
            return

        # Utility method for copying group params.
        def copy_group_params(shard_main_groups, model_groups):
            for shard_main_group, model_group in zip(shard_main_groups, model_groups):
                for shard_main_param, model_param in zip(shard_main_group, model_group):

                    param_range_map = self._get_model_param_range_map(model_param)
                    world_range = param_range_map["gbuf_world_in_bucket"]

                    assert world_range.size == shard_main_param.nelement()

                    gbuf_index, _, bucket_id = self.model_param_gbuf_map[model_param]
                    model_param_buffer = self.buffers[gbuf_index].buckets[bucket_id].param_data

                    shard_model_param = model_param_buffer.view(-1)[
                        world_range.start : world_range.end
                    ]

                    if is_float8tensor(model_param):
                        # 1. When "--fp8-param-gather" is disabled, the main param is first cast to
                        #    BF16/FP16, and then cast to FP8, so the amax_history is calculated
                        #    using BF16/FP16 param.
                        # 2. When "--fp8-param-gather" is enabled, we can cast the FP32 main param
                        #    to FP8 directly, which results in slightly different results with
                        #    higher speed. In theory, this does not affect convergence.
                        # TODO: The following code maintains the logic of the point-1 above. It can
                        # be deleted if it is not necessary.
                        shard_main_param = shard_main_param.to(model_param.dtype)

                        quantize_param_fragment(
                            shard_main_param, out=shard_model_param, param=model_param
                        )
                    else:
                        shard_model_param.data.copy_(shard_main_param)

        # When using precision-aware optimizer, main params are held by self.optimizer. It will also
        # do the work of copying data from main params to model params.
        if self.config.use_precision_aware_optimizer:
            return

        # Copy shard groups to model groups.
        copy_group_params(self.shard_fp32_from_float16_groups, self.model_float16_groups)
        copy_group_params(self.shard_fp32_groups, self.model_fp32_groups)
```
