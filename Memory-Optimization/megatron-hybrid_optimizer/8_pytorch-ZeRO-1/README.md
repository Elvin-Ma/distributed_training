# 1 接口

```
class
torch.distributed.optim.ZeroRedundancyOptimizer(params, optimizer_class, process_group=None, parameters_as_bucket_view=False, overlap_with_ddp=False, **defaults)[source][source]

    包装一个任意的 optim.Optimizer 并将其状态在组内的各个 rank 之间进行分片。

    分片的方式遵循 ZeRO 的描述。

    每个 rank 中的本地优化器实例仅负责更新大约 1 / world_size 的参数，因此只需要保留 1 / world_size 的优化器状态。在本地更新参数后，每个 rank 会将其参数广播给所有其他 peer，以保持所有模型副本的状态一致。ZeroRedundancyOptimizer 可以与 torch.nn.parallel.DistributedDataParallel 结合使用，以减少每个 rank 的峰值内存消耗。

    ZeroRedundancyOptimizer 使用一种基于排序的贪心算法来在每个 rank 上打包一定数量的参数。每个参数属于一个单一的 rank，不会在多个 rank 之间分割。这种分区是任意的，可能与参数的注册顺序或使用顺序不一致。

    Parameters

        params (Iterable) – an Iterable of torch.Tensor s or dict s giving all parameters, which will be sharded across ranks.
    Keyword Arguments

            optimizer_class (torch.nn.Optimizer) – the class of the local optimizer.

            process_group (ProcessGroup, optional) – torch.distributed ProcessGroup (default: dist.group.WORLD initialized by torch.distributed.init_process_group()).

            parameters_as_bucket_view (bool, optional) – if True, parameters are packed into buckets to speed up communication, and param.data fields point to bucket views at different offsets; if False, each individual parameter is communicated separately, and each params.data stays intact (default: False).

            overlap_with_ddp (bool, optional) – if True, step() is overlapped with DistributedDataParallel ‘s gradient synchronization; this requires (1) either a functional optimizer for the optimizer_class argument or one with a functional equivalent and (2) registering a DDP communication hook constructed from one of the functions in ddp_zero_hook.py; parameters are packed into buckets matching those in DistributedDataParallel, meaning that the parameters_as_bucket_view argument is ignored. If False, step() runs disjointly after the backward pass (per normal). (default: False)

            **defaults – any trailing arguments, which are forwarded to the local optimizer.
```

# 2 代码实现

```python
import torch.nn as nn
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
model = nn.Sequential(*[nn.Linear(2000, 2000).to(rank) for _ in range(20)])
ddp = DDP(model, device_ids=[rank])
opt = ZeroRedundancyOptimizer(
    ddp.parameters(),
    optimizer_class=torch.optim.Adam,
    lr=0.01
)
ddp(inputs).sum().backward()
opt.step()
```

# 3 改进

```python
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    # 初始化分布式环境
    dist.init_process_group(backend="nccl")

    # 获取当前进程的 rank 和 world_size
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 设置当前设备
    torch.cuda.set_device(rank)

    # 定义模型
    model = nn.Sequential(*[nn.Linear(2000, 2000).to(rank) for _ in range(20)])

    # 使用 DistributedDataParallel 包装模型
    ddp_model = DDP(model, device_ids=[rank])

    # 定义 ZeroRedundancyOptimizer
    opt = ZeroRedundancyOptimizer(
        ddp_model.parameters(),
        optimizer_class=torch.optim.Adam,
        lr=0.01
    )

    # 模拟输入数据
    inputs = torch.randn(32, 2000).to(rank)

    # 前向传播
    outputs = ddp_model(inputs)

    # 计算损失并反向传播
    loss = outputs.sum()
    loss.backward()

    # 更新参数
    opt.step()

    # 清理分布式环境
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

# 4 执行

```bash
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12345 train.py
```