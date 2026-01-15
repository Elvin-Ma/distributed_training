# 1 Distributed Checkpoint - torch.distributed.checkpoint

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;分布式检查点（Distributed Checkpoint，DCP）支持**在多个进程（rank）上并行加载和保存模型**。它具备**加载时的重分片（resharding）功能**，这使得模型能够`以一种集群拓扑结构保存，而以另一种集群拓扑结构加载`。

DCP 与 torch.save 和 torch.load 存在一些显著差异：

- DCP 会在每个检查点生成`多个文件`，每个进程(对应每个rank)`至少生成一个文件`。

- DCP 采用原地（in place）操作方式，这意味着**模型需要先分配好自身的数据存储空间**，而 DCP 会直接使用这些已分配的存储空间，而非另行分配。


# 2 How DCP works
torch.distributed.checkpoint() 支持在多个进程（rank）上并行地保存和加载模型。你可以使用这个模块在任意数量的进程上并行地进行保存操作，然后在加载时**重新分片（re-shard）以适应不同的集群拓扑结构**。

此外，通过使用 torch.distributed.checkpoint.state_dict() 中的模块，分布式检查点（DCP）提供了在分布式环境中优雅处理 state_dict 生成和加载(loading)的支持。这包括**管理模型和优化器之间的完全限定名（Fully Qualified Name，FQN）映射**，以及为 PyTorch 提供的并行性设置默认参数。

分布式检查点（DCP）与 torch.save() 和 torch.load() 存在几个显著的不同之处：

- DCP 在每个检查点会生成多个文件，每个进程（rank）**至少生成一个文件**。<br>
- DCP 采用**原地操作方式**，这意味着**模型需要先分配好自身的数据存储空间**，而 DCP 会直接使用这些已分配的存储空间，而不是另行分配。<br>
- DCP `对有状态（Stateful）对象提供了特殊处理`（这些对象在 torch.distributed.checkpoint.stateful 中有正式定义），如果定义了 state_dict 和 load_state_dict 方法，DCP 会自动调用它们。<br>

**重分片的代码执行核心链路** <br>
函数通过 “本地计划（算需求）→全局计划（协调整）→分发计划（定分工）→执行计划（做重分片）” 的四步流程，将 “分布式策略改变” 的重分片需求落地：

- 所有 rank 先基于自身视角计算 “需要什么分片”（local_step）；
- 主 rank 汇总所有需求，统一分配 “谁读原分片、谁接收拆分后的数据”（global_step）；
- 计划分发到每个 rank（reduce_scatter）；
- 各 rank 按计划读取、通信、拆分 / 合并，完成重分片（read_data）。

整个过程中，planner是重分片的核心（封装了所有拓扑适配和分片计算逻辑），distW是分布式通信的载体，storage_reader是存储交互的接口 —— 三者配合实现了 “任意分布式策略改变” 下的重分片加载。

# 3 核心过程: _load_state_dict

```sh
重分片核心逻辑（local_step）：以 8-rank 中的rank=0为例，create_local_plan()的内部计算过程：
从元数据中读取：张量W全局形状[1024,4096]、保存时按dim=1拆分（4 分片，每片[1024,1024]）；
计算新拓扑下的分片规则：8-rank 仍按dim=1拆分，每片[1024,512]；
确定当前 rank（0）的分片范围：dim=1的0~512列；
映射到保存时的分片：该范围落在原 4-rank 的S0（0~1024 列）的前 512 列；
生成local_plan：记录 “需要读取 S0 的 0~512 列→拼接为本地分片 T0”。
所有 8 个 rank 都会执行这个逻辑，生成各自的local_plan（比如 rank1 需要 S0 的 512~1024 列，rank2 需要 S1 的 0~512 列，依此类推）。
```

```python
def _load_state_dict(
    state_dict: dict[str, Any],
    storage_reader: StorageReader,
    process_group: Optional[dist.ProcessGroup] = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: Optional[LoadPlanner] = None,
) -> None:

    distW = _DistWrapper(process_group, not no_dist, coordinator_rank)
    if planner is None:
        planner = DefaultLoadPlanner()

    ckpt_kwargs = {}
    if (ckpt_id := getattr(storage_reader, "checkpoint_id", None)) is not None:
        ckpt_kwargs["checkpoint_id"] = ckpt_id
        ckpt_kwargs["process_group"] = distW.group

    @_dcp_method_logger(**ckpt_kwargs)
    def local_step():
        assert planner is not None
        # 步骤1：读取全局元数据（所有rank都读，确保拿到保存时的拓扑/张量信息）
        # 元数据包含：保存时4-rank的拆分策略、张量全局形状（如W: [1024,4096]）
        metadata = storage_reader.read_metadata()
        # 步骤2：规划器初始化——绑定“当前state_dict+元数据+是否是主rank”
        # 核心：planner解析元数据，得到“保存时的分布式策略”和“张量全局结构”
        # 同时，planner感知当前拓扑（8-rank），为“策略转换”做准备
        planner.set_up_planner(state_dict, metadata, distW.is_coordinator)
        # 步骤3：存储读取器初始化——对接元数据，知道分片的存储路径
        storage_reader.set_up_storage_reader(metadata, distW.is_coordinator)
        # 步骤4：生成本地初步加载计划（重分片的第一步计算）
        # 核心逻辑：planner根据“当前rank ID+8-rank拓扑”，计算自己需要的分片
        # 比如rank 0（8-rank）会计算：需要原4-rank中S0的前512列（新分片T0）
        local_plan = planner.create_local_plan()
        # 步骤5：存储层优化本地计划（比如合并相同路径的读取请求）
        local_plan = storage_reader.prepare_local_plan(local_plan)
        return local_plan

    @_dcp_method_logger(**ckpt_kwargs)
    def global_step(all_local_plans):
        assert planner is not None
        # 步骤1：主rank聚合所有rank的local_plan，生成全局加载计划
        # 核心：解决“分片冲突/重复读取”，统一分配原分片的读取权
        # 比如：原S0需要被rank0和rank1读取，全局计划会指定“由rank0读取S0，再拆分给rank1”
        all_local_plans = planner.create_global_plan(all_local_plans)
        # 步骤2：存储层优化全局计划（比如合并跨rank的相同分片读取请求）
        all_local_plans = storage_reader.prepare_global_plan(all_local_plans)
        return all_local_plans

    central_plan: LoadPlan = distW.reduce_scatter("plan", local_step, global_step)

    @_dcp_method_logger(**ckpt_kwargs)
    def read_data():
        assert planner is not None
        # 步骤1：规划器最终确认计划——结合central_plan，生成“数据读取+通信”的具体指令
        # 比如：rank0的指令是“从存储读取S0→按dim=1拆分→保留前512列”
        final_local_plan = planner.finish_plan(central_plan)
        # 步骤2：按计划读取数据（核心：重分片的实际执行）
        # 内部逻辑：
        # - 读取阶段：负责读取的rank（如rank0）从存储加载原分片S0；
        # - 通信阶段：按计划拆分/发送分片（rank0发送T1给rank1）；
        # - 拼接阶段：接收方（如rank1）拼接收到的子分片，得到本地分片；
        all_reads = storage_reader.read_data(final_local_plan, planner)

        # 步骤3：等待所有读取/通信操作完成
        all_reads.wait()
        return None

    # 所有rank同步执行read_data，对read_data 的结果(可能触发exception)做all_gather 确保通信完成.
    _ = distW.all_gather("read", read_data)
```

# 4 read_data

```python
def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
    """
    根据加载计划从存储中读取数据，按文件聚合请求并处理，最终将数据交付给加载规划器

    参数:
        plan: LoadPlan - 加载计划，包含所有需要读取的ReadItem项
        planner: LoadPlanner - 加载规划器，负责解析/加载/提交字节流/张量数据

    返回:
        Future[None] - 异步风格的返回值（实际同步完成，仅兼容异步接口）
    """
    # 1. 按文件路径聚合读取请求：避免重复打开同一文件，提升IO效率
    per_file: dict[str, list[ReadItem]] = {}
    for read_item in plan.items:
        # 获取当前读取项对应的存储元信息
        item_md: _StorageInfo = self.storage_data[read_item.storage_index]
        # 提取文件相对路径作为分组key
        path = item_md.relative_path
        # 按路径分组：不存在则创建空列表，再追加当前读取项
        per_file.setdefault(path, []).append(read_item)

    # 2. 遍历每个文件的所有读取请求，批量处理
    for relative_path, reqs in per_file.items():
        # 拼接基础路径 + 文件相对路径，得到文件完整路径
        new_path = self.fs.concat_path(self.path, relative_path)
        # 以二进制只读模式打开文件流（with自动管理流的关闭）
        with self.fs.create_stream(new_path, "rb") as stream:
            # TODO 优化点：按偏移量排序请求 + 缓存读取结果，减少重复IO
            for req in reqs:
                # 获取当前请求对应的存储元信息
                item_md = self.storage_data[req.storage_index]
                # 根据存储元信息对文件流进行切片（提取请求的指定片段）
                file_slice = self._slice_file(stream, item_md)
                # 对切片后的文件流应用转换（解码/格式转换等）
                # 兼容旧版本：旧实现可能无transform_descriptors字段，兜底为空元组
                transform_from = self.transforms.transform_load_stream(
                    req,
                    item_md.transform_descriptors or (),
                    file_slice,
                )

                # 3. 分支1：处理字节流类型的读取请求
                if req.type == LoadItemType.BYTE_IO:
                    # 读取转换后流的所有字节，封装为可seek的BytesIO（重置指针到开头）
                    read_bytes = io.BytesIO(transform_from.read(-1))
                    read_bytes.seek(0)
                    # 将字节流交付给规划器处理
                    planner.load_bytes(req, read_bytes)
                # 4. 分支2：处理张量类型的读取请求
                else:
                    # torch.load要求输入可seek，若流不可seek则封装为BytesIO
                    if transform_from.seekable():
                        seekable = transform_from
                    else:
                        seekable = io.BytesIO(transform_from.read(-1))
                        seekable.seek(0)

                    # 加载张量：
                    # - map_location="cpu"：强制加载到CPU（避免设备不匹配）
                    # - weights_only=True：仅加载权重（安全，防止恶意代码执行）
                    tensor = cast(
                        Tensor,
                        torch.load(
                            seekable,
                            map_location="cpu",
                            weights_only=True,
                        ),
                    )
                    # 根据存储偏移和长度对张量进行切片（提取指定维度/范围）
                    tensor = narrow_tensor_by_index(
                        tensor, req.storage_offsets, req.lengths
                    )
                    # 解析目标张量并脱离计算图（避免梯度关联）
                    target_tensor = planner.resolve_tensor(req).detach()

                    # 断言校验：目标张量与读取张量尺寸必须一致（防止数据错位）
                    assert target_tensor.size() == tensor.size(), (
                        f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                    )
                    # 原地复制张量数据（避免重新分配内存，提升效率）
                    target_tensor.copy_(tensor)
                    # 将处理后的张量提交给规划器
                    planner.commit_tensor(req, target_tensor)

    # 5. 返回已完成的Future（兼容异步接口，无实际异步操作）
    fut: Future = Future()
    fut.set_result(None)
    return fut
```

# 5 example

```python
import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn as nn

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

CHECKPOINT_DIR = "checkpoint"


class AppState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict
        }

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"]
        )

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(16, 16)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 8)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355 "

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def run_fsdp_checkpoint_save_example(rank, world_size):
    print(f"Running basic FSDP checkpoint saving example on rank {rank}.")
    setup(rank, world_size)

    # create a model and move it to GPU with id rank
    model = ToyModel().to(rank)
    model = FSDP(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    optimizer.zero_grad()
    model(torch.rand(8, 16, device="cuda")).sum().backward()
    optimizer.step()

    state_dict = { "app": AppState(model, optimizer) }
    dcp.save(state_dict, checkpoint_id=CHECKPOINT_DIR)

    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Running fsdp checkpoint example on {world_size} devices.")
    mp.spawn(
        run_fsdp_checkpoint_save_example,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
```

运行结果：
![](https://docs.pytorch.org/tutorials/_images/distributed_checkpoint_generated_files.png)

# 4 load example

在保存之后，我们创建一个相同的由全分片数据并行（Fully Sharded Data Parallel，FSDP）包裹的模型，并将存储中保存的状态字典（state dict）加载到该模型中。你可以在相同的世界大小（world size）或不同的世界大小下进行加载。

请注意，在加载之前，你需要调用 model.state_dict()，并将其传递给 DCP 的 load_state_dict() API。这与 torch.load() 有根本的不同，因为 torch.load() 只需要在加载前提供检查点的路径即可。我们需要在加载前获取状态字典的原因是：

- DCP **使用来自模型状态字典的预分配存储空间来从检查点目录加载数据**。在加载过程中，传入的状态字典会被**原地更新**。

- DCP 在加载前需要模型的分片信息，以支持**重新分片（resharding）**。

```python
import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
import torch.multiprocessing as mp
import torch.nn as nn

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

CHECKPOINT_DIR = "checkpoint"


class AppState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict
        }

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"]
        )

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(16, 16)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 8)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355 "

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def run_fsdp_checkpoint_load_example(rank, world_size):
    print(f"Running basic FSDP checkpoint loading example on rank {rank}.")
    setup(rank, world_size)

    # create a model and move it to GPU with id rank
    model = ToyModel().to(rank)
    model = FSDP(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    state_dict = { "app": AppState(model, optimizer)}
    dcp.load(
        state_dict=state_dict,
        checkpoint_id=CHECKPOINT_DIR,
    )

    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Running fsdp checkpoint example on {world_size} devices.")
    mp.spawn(
        run_fsdp_checkpoint_load_example,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
```

# 5 src code analysis
- 调用方式

```python
torch.distributed.checkpoint.save()
```

- save func stack <br>

![alt text](image.png)


# 参考链接
- [torch.doc](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html#module-torch.distributed.checkpoint)
