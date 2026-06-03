# PyTorch SymmetricMemory 系统笔记

> 基于仓库当前 `v2.9.0` 分支源码，所有引用路径均可点。
> 目标读者：熟悉 PyTorch 分布式与 CUDA、想把 SymmetricMemory 吃透甚至自己加 backend 的工程师。

---

## 目录

1. [它是什么 / 解决什么](#1-它是什么--解决什么)
2. [Python 端用法与案例](#2-python-端用法与案例)
3. [核心对象模型](#3-核心对象模型)
4. [前端 API 全景](#4-前端-api-全景)
5. [后端实现分类与选型](#5-后端实现分类与选型)
6. [一次完整的前→后端调用链：`one_shot_all_reduce`](#6-一次完整的前后端调用链one_shot_all_reduce)
7. [Inductor / torch.compile 集成](#7-inductor--torchcompile-集成)
8. [signal_pad 同步协议](#8-signal_pad-同步协议)
9. [局限、平台 gating 与已知坑](#9-局限平台-gating-与已知坑)
10. [参考文件清单](#10-参考文件清单)

---

## 1. 它是什么 / 解决什么

`SymmetricMemory` 是 PyTorch 为"**组内每张卡都持有相同大小的内存 + 彼此 P2P 可访问 + 附带轻量同步区**"这一抽象搭的 C++ 接口，核心声明在
[SymmetricMemory.hpp](torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp)。

它解决的核心痛点：

- **绕开 NCCL 黑盒**：NCCL 集合通讯把 buffer + 算法 + 调度都打包了；SymmetricMemory 把"**远端指针**"暴露出来，让用户自己写通讯内核（one-shot all-reduce、multimem、async-TP 等），便于和计算 fuse。
- **直接 peer load/store**：所有 rank 拿到彼此 buffer 的 device 指针，可以直接 `buffers[peer][i]` 读写，配合 NVLink/NVSwitch/GDR 获得远高于 `ncclAllReduce` 小消息下的带宽/延迟。
- **内置 signal_pad 同步**：每块内存旁边挂一小片 `signal_pad`，用于 CAS 信号语义的跨 rank 同步，写算法不用再借道 NCCL。
- **持久分配 + 编译器友好**：支持 `alloc_id` 的持久化申请，Inductor 能在 compile 时把某张张量"钉"到对称内存里，和图里的通讯节点做 lowering。

头文件注释原文（[SymmetricMemory.hpp:8](torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp#L8)）把设计意图说得很清：
> Signal pads are P2P-accessible memory regions designated for synchronization… users may utilize signal pads for their own synchronization logic, provided that the signal pads remain zero-filled following successful synchronization.

和常见替代品的差异：

| 维度 | `ncclAllReduce` | `SymmetricMemory` |
|---|---|---|
| 编程模型 | 黑盒集合 | 暴露 peer 指针，用户/Inductor 自己写 kernel |
| 与计算融合 | 困难 | 容易（手写一个 CUDA kernel 就能边算边 all-reduce） |
| 小消息延迟 | 受 NCCL protocol 开销 | 直接 load-store，通常更低 |
| 生命周期 | 每次 call 独立 | 支持 persistent alloc，compile 时可预规划 |
| 同步 | 内部 stream | 用户级 `signal_pad` + barrier/put_signal/wait_signal |

---

## 2. Python 端用法与案例

所有 Python 入口都在 [torch/distributed/_symmetric_memory/__init__.py](torch/distributed/_symmetric_memory/__init__.py)（1795 行）。

### 2.1 最小例子：分配 → rendezvous → peer 读写

```python
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

dist.init_process_group("nccl")
symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)

# 组内每个 rank 分配相同 shape/dtype 的张量
t = symm_mem.empty(1024, dtype=torch.float32, device="cuda")

# 集合 rendezvous：阻塞到所有 rank 都到齐，返回 handle
hdl = symm_mem.rendezvous(t, group=dist.group.WORLD)

# 通过 hdl 拿到任意 peer 的 buffer（零拷贝视图）
for peer in range(hdl.world_size):
    peer_view = hdl.get_buffer(peer, (1024,), torch.float32)
    # peer_view 指向 peer rank 的同名 tensor，可以直接 load/store

# 一轮典型的"写-同步-读"
t.fill_(float(hdl.rank))
hdl.barrier(channel=0, timeout_ms=5000)   # 保证 peer 写入可见
peer_view = hdl.get_buffer((hdl.rank + 1) % hdl.world_size, (1024,), torch.float32)
print(peer_view[0])   # == (rank+1) % world_size
```

测试参考：[test/distributed/test_symmetric_memory.py](test/distributed/test_symmetric_memory.py)。

### 2.2 直接调用高层集合原语

对称内存被作为新的集合原语注册进 dispatcher：

```python
# sum all-reduce，用 one-shot 算法（一次 load 所有 peer，本地归约）
out = torch.ops.symm_mem.one_shot_all_reduce(t, "sum", dist.group.WORLD.group_name)

# H100 multimem 硬件规约（multimem.ld_reduce，带宽 ~2x）
torch.ops.symm_mem.multimem_all_reduce_(t, "sum", dist.group.WORLD.group_name)

# two-shot：适合中/大消息
torch.ops.symm_mem.two_shot_all_reduce_(t, "sum", dist.group.WORLD.group_name)
```

算子 schema 定义在 [SymmetricMemory.cpp:453-515](torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.cpp#L453-L515)。

### 2.3 和 `torch.compile` 一起使用

只要 `enable_symm_mem_for_group` 已开启 + 输入大小在阈值内，Inductor 自动把 `dist.all_reduce(...)` lower 成 `one_shot_all_reduce`：

```python
@torch.compile
def f(x):
    dist.all_reduce(x)            # 编译后被替换为 one_shot_all_reduce
    return x * 2
```

判定逻辑见 [torch/_inductor/comm_lowering.py:147-167](torch/_inductor/comm_lowering.py#L147-L167)（见 §7）。

---

## 3. 核心对象模型

### 3.1 两个抽象基类

[SymmetricMemory.hpp:38](torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp#L38)

```cpp
class TORCH_API SymmetricMemory : public c10::intrusive_ptr_target {
 public:
  virtual std::vector<void*> get_buffer_ptrs()      = 0;
  virtual std::vector<void*> get_signal_pad_ptrs()  = 0;
  virtual void** get_buffer_ptrs_dev()              = 0;  // 设备侧数组，kernel 直接用
  virtual void** get_signal_pad_ptrs_dev()          = 0;
  virtual size_t get_buffer_size()                  = 0;
  virtual size_t get_signal_pad_size()              = 0;
  virtual bool   has_multicast_support()            = 0;
  virtual void*  get_multicast_ptr()                = 0;
  virtual int    get_rank()                         = 0;
  virtual int    get_world_size()                   = 0;

  virtual void barrier(int channel, size_t timeout_ms)                    = 0;
  virtual void put_signal(int dst_rank, int channel, size_t timeout_ms)   = 0;
  virtual void wait_signal(int src_rank, int channel, size_t timeout_ms)  = 0;

  // 把 peer buffer 包成 Tensor（零拷贝视图）
  virtual at::Tensor get_buffer(int rank, c10::IntArrayRef sizes,
                                c10::ScalarType dtype, int64_t storage_offset) = 0;
};
```

[SymmetricMemory.hpp:100](torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp#L100)

```cpp
class SymmetricMemoryAllocator : public c10::intrusive_ptr_target {
 public:
  virtual void* alloc(size_t size, int device_idx,
                      const std::optional<std::string>& group_name) = 0;
  virtual void  free(void* ptr)                                     = 0;
  virtual c10::intrusive_ptr<SymmetricMemory> rendezvous(
      void* ptr, const std::optional<std::string>& group_name)      = 0;
};
```

**两件事分离**：`alloc/free` 只管本 rank 的物理内存；`rendezvous()` 才把组内所有 rank 的指针交换齐、返回可用的 `SymmetricMemory`。

### 3.2 signal_pad 布局

常量定义在 [CUDASymmetricMemoryTypes.hpp:10-15](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryTypes.hpp#L10-L15)：

```cpp
constexpr int symm_max_nblocks        = 32;   // 最多 32 个 CUDA block 参与同步
constexpr int max_cuda_p2p_domain_size = 72;  // 对齐 NVL72 域
constexpr size_t signal_pad_size = 32 * 72 * sizeof(uint32_t);  // = 9216 bytes
```

每个 rank 的分配实际布局：

```
┌──────────────────────────────────┬──────────────────────┐
│        user buffer (size)        │    signal_pad (~9KB) │
└──────────────────────────────────┴──────────────────────┘
 ptr                              ptr + round_up(size,16)
```

signal_pad 里的一个 `uint32_t` 地址由三元组决定：
```
addr = signal_pads[target_rank] + blockIdx.x * world_size + source_rank
```
即 **(block_channel, source, target)** → 一个独立 uint32 槽。保证跨 block、跨 peer 对互不冲突。

### 3.3 指针交换（rendezvous）

统一的 Store-based 交换工具 `StoreExchange::all_gather<T>` 在 [CUDASymmetricMemoryUtils.hpp:46](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp#L46)：每个 rank 把序列化后的 payload `set` 进 TCPStore，再 `wait + get` 其他 rank 的。不同 backend 交换的载荷不同：

- **CUDA**：`cuIpc` POSIX FD 或 `CU_MEM_HANDLE_TYPE_FABRIC` handle
- **NVSHMEM**：仅交换基址，peer 指针由 `nvshmem_ptr()` 计算
- **NCCL**：`ncclWindow_t`

---

## 4. 前端 API 全景

### 4.1 Python 顶层 API（`torch.distributed._symmetric_memory`）

| API | 位置 | 作用 |
|---|---|---|
| `enable_symm_mem_for_group(group_name)` | [\_\_init\_\_.py:24](torch/distributed/_symmetric_memory/__init__.py#L24) | 为指定 PG 开启 symm mem，建立 side-channel store |
| `is_symm_mem_enabled_for_group(group_name)` | [\_\_init\_\_.py:76](torch/distributed/_symmetric_memory/__init__.py#L76) | 查询 |
| `get_symm_mem_workspace(group_name, min_size)` | [\_\_init\_\_.py:91](torch/distributed/_symmetric_memory/__init__.py#L91) | 返回一块组内共享的工作区（可被多个算子复用） |
| `empty(*size, dtype, device)` | [\_\_init\_\_.py:1671](torch/distributed/_symmetric_memory/__init__.py#L1671) | 对称分配；语义类似 `torch.empty`，但要求组内 size/shape 一致 |
| `rendezvous(tensor, group)` | [\_\_init\_\_.py:1714](torch/distributed/_symmetric_memory/__init__.py#L1714) | 集合调用，返回 `_SymmetricMemory` handle |
| `is_nvshmem_available()` | [\_\_init\_\_.py:1743](torch/distributed/_symmetric_memory/__init__.py#L1743) | |
| `set_backend(name)` / `get_backend(device)` | [\_\_init\_\_.py:1759](torch/distributed/_symmetric_memory/__init__.py#L1759) | 手动切后端 |

### 4.2 pybind 暴露的 `_SymmetricMemory`

在 [torch/csrc/distributed/c10d/init.cpp](torch/csrc/distributed/c10d/init.cpp)（搜索 `_SymmetricMemory`）里把 C++ 对象绑到 Python。handle 对象上的主要方法：`rank`、`world_size`、`buffer_size`、`get_buffer(rank, sizes, dtype, storage_offset=0)`、`get_signal_pad(...)`、`barrier(channel, timeout_ms)`、`put_signal(...)`、`wait_signal(...)`。

### 4.3 算子层（Dispatcher）

[SymmetricMemory.cpp:453-515](torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.cpp#L453-L515) 注册了一组高层集合算子：

```
symm_mem::one_shot_all_reduce(Tensor, str reduce_op, str group_name) -> Tensor
symm_mem::one_shot_all_reduce_out(..., Tensor(a!) out) -> Tensor(a!)
symm_mem::one_shot_all_reduce_copy(Tensor symm_buffer, Tensor local_input, ...) -> Tensor
symm_mem::two_shot_all_reduce_(Tensor(a!), str, str) -> Tensor(a!)
symm_mem::two_shot_all_reduce_out(...)
symm_mem::multimem_all_reduce_(Tensor(a!), str, str) -> Tensor(a!)
symm_mem::multimem_one_shot_all_reduce(Tensor, str, str) -> Tensor
symm_mem::multimem_one_shot_all_reduce_out(..., Tensor(a!)) -> Tensor(a!)
symm_mem::reduce_scatter_out(Tensor(a!) input, str group_name, bool split_last_dim, Tensor(b!) output) -> Tensor(b!)
```

Meta kernel（shape/dtype 推理，供 compile/fx 使用）见 [SymmetricMemory.cpp:434, 443, 513](torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.cpp#L434)。

---

## 5. 后端实现分类与选型

### 5.1 总览

| Backend | 文件 | 适用场景 | 多节点 | 多播 | 状态 |
|---|---|---|---|---|---|
| **CUDA**   | [CUDASymmetricMemory.cu](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.cu) (838L) | 单节点/NVL72 最佳延迟 | 需 fabric handle（CUDA 12.3+） | ✓ (SM≥90) | 完整 |
| **NVSHMEM**| [NVSHMEMSymmetricMemory.cu](torch/csrc/distributed/c10d/symm_mem/NVSHMEMSymmetricMemory.cu) (449L) | 多机大规模 | ✓ | 依赖 NVSHMEM | 完整 |
| **NCCL**   | [NCCLSymmetricMemory.cu](torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.cu) (293L) | 复用 NCCL 的 window | ✓ | ✗ | 部分（barrier/signal 为 TODO，见 L96-L106） |

选后端顺序：环境变量 `TORCH_SYMMMEM=CUDA|NVSHMEM|NCCL` 直接指定；否则查一个可用性表。选择逻辑位于 [CUDASymmetricMemoryUtils.cpp](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.cpp)。

### 5.2 CUDA backend（核心）

分配路径（[CUDASymmetricMemory.cu:378](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.cu#L378)）：

1. `cuMemCreate_()` 建立物理 allocation，handle type 在
   POSIX FD 与 FABRIC 之间按 CUDA 版本/硬件挑（[L363-L369](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.cu#L363-L369)）。
2. 保留 VA 后 `cuMemMap_()` 到本地（[CUDASymmetricMemoryUtils.hpp:108-112](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp#L108-L112)）。
3. Rendezvous 阶段 `cuMemExportToShareableHandle_()` 导出句柄（[L629-L634](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.cu#L629-L634)），经 `IpcChannel`（Unix domain socket）或 TCPStore 分发，peer 侧 `cuMemImportFromShareableHandle_() + cuMemMap_()` 拿到可访问 VA（[L688-L708](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.cu#L688-L708)）。
4. 可选：`cuMulticastCreate_()`（[L528](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.cu#L528)）+ `cuMulticastBindMem_()`（[L585-L588](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.cu#L585-L588)）建 multicast 对象，后续 `multimem.ld_reduce`/`multimem.st` 硬件规约。
5. P2P enable 由 `CudaDMAConnectivity` 检测（[CudaDMAConnectivity.cpp](torch/csrc/distributed/c10d/symm_mem/CudaDMAConnectivity.cpp)）。

硬件/版本前提：CUDA 12.0+ 驱动 API、SM ≥ 80（intra-node 约束见 [intra_node_comm.cu:25-31](torch/csrc/distributed/c10d/symm_mem/intra_node_comm.cu#L25-L31)）、multicast 需 SM ≥ 90 + CUDA 12.3+。

### 5.3 NVSHMEM backend

[NVSHMEMSymmetricMemory.cu:95](torch/csrc/distributed/c10d/symm_mem/NVSHMEMSymmetricMemory.cu#L95) 用 `nvshmem_malloc()` 分配 buffer 与 signal_pad。peer 指针靠 [L85-L92](torch/csrc/distributed/c10d/symm_mem/NVSHMEMSymmetricMemory.cu#L85-L92) 的 `nvshmem_ptr(base, peer_global_rank)` 直接算出（若 peer 不可 P2P 则返回 `nullptr`，跨网络改走 nvshmem device API）。`world_within_cuda_p2p_` 标记该组是否全在一个 P2P 域内。

注册入口：[L430-L446](torch/csrc/distributed/c10d/symm_mem/NVSHMEMSymmetricMemory.cu#L430-L446)。Triton 优化 kernel 见 [torch/distributed/_symmetric_memory/_nvshmem_triton.py](torch/distributed/_symmetric_memory/_nvshmem_triton.py)，collective 增强见 [nvshmem_extension.cu](torch/csrc/distributed/c10d/symm_mem/nvshmem_extension.cu)、team 管理 [nvshmem_team_manager.hpp](torch/csrc/distributed/c10d/symm_mem/nvshmem_team_manager.hpp)。

### 5.4 NCCL backend

[NCCLSymmetricMemory.cu:163, L232](torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.cu#L163) 用 `ncclMemAlloc()` 分配，再 `ncclCommWindowRegister(comm, ptr, size, &handle, NCCL_WIN_COLL_SYMMETRIC)` 注册成 window（[L221-L241](torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.cu#L221-L241)）。需要 NCCL ≥ 2.27.1（[L5-L6](torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.cu#L5-L6)）。

> ⚠️ `barrier/put_signal/wait_signal` 目前是 TODO（[L96-L106](torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.cu#L96-L106)），`has_multicast_support() == false`。仓库里的设计讨论：[nccl_symm.md](nccl_symm.md)、[nccl_symm_cc.md](nccl_symm_cc.md)、[nccl_register_vs_window_register.md](nccl_register_vs_window_register.md)。

### 5.5 如何新增一个 backend？

按照 §3 两个基类的接口各写一个子类即可：
1. `alloc/free/rendezvous` 负责物理分配 + 指针交换 + 构造子类 `SymmetricMemory` 实例；
2. 子类实现 `get_*_ptrs{,_dev}`、`barrier/put_signal/wait_signal`、`get_buffer/get_signal_pad`；
3. 通过静态变量在加载时 `register_allocator(device_type, allocator)`（`TORCH_SYMMMEM` env 决定默认后端）；
4. 若要接入现有算子，在自己的 backend cu 里写 `TORCH_LIBRARY_IMPL(symm_mem, <DeviceType>, m)`，把 `one_shot_all_reduce` 等 impl 指向自己的 kernel。

---

## 6. 一次完整的前→后端调用链：`one_shot_all_reduce`

以 `torch.ops.symm_mem.one_shot_all_reduce(t, "sum", group_name)` 为例：

**① Python dispatcher 入口**
`torch.ops.symm_mem.one_shot_all_reduce` → schema：[SymmetricMemory.cpp:463](torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.cpp#L463)

**② Meta kernel**（torch.compile 走 FakeTensor 传播）
[SymmetricMemory.cpp:434](torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.cpp#L434) `one_shot_all_reduce_meta`，注册于 [L513](torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.cpp#L513)。

**③ CUDA dispatcher impl**
[CUDASymmetricMemoryOps.cu:1189-1218](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryOps.cu#L1189-L1218):
```cpp
TORCH_LIBRARY_IMPL(symm_mem, CUDA, m) {
  m.impl("one_shot_all_reduce", ::one_shot_all_reduce);
  // ...
}
```

**④ C++ 适配层**
[CUDASymmetricMemoryOps.cu:551](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryOps.cu#L551)
```cpp
at::Tensor one_shot_all_reduce(const at::Tensor& input,
                               std::string reduce_op,
                               std::string group_name) {
  auto out = at::empty_like(input);
  return one_shot_all_reduce_out(input, reduce_op, group_name, out);
}
```

**⑤ 主实现**
[CUDASymmetricMemoryOps.cu:447-530](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryOps.cu#L447-L530) `one_shot_all_reduce_out_impl`：
1. `rendezvous(input, group_name)` 拿到 `SymmetricMemory` handle（L457）。
2. 对齐校验（L466-470）——通常要求 16B 对齐，决定 vectorized 路径。
3. `init_elementwise_launch_config()` 算 grid/block（L474-483）。
4. `AT_DISPATCH_*` 按 dtype + alignment 选 kernel 模板（L486-510）并 launch。

**⑥ CUDA kernel**
[CUDASymmetricMemoryOps.cu:389-445](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryOps.cu#L389-L445) `one_shot_all_reduce_kernel<scalar_t, k_alignment>`：
```
barrier-in (signal_pads)               // 等所有 rank 都可读了
for i in [tid..numel, step=grid*block]:
    acc = 0
    for peer in 0..world_size:
        acc += buffers[peer][i]        // P2P load，硬件走 NVLink/fabric
    output[i] = acc
barrier-out (signal_pads)              // 通知 peer 我这轮用完了
```

**⑦ signal_pad 同步原语**
barrier 的底层是 [CUDASymmetricMemory-inl.h:111-155](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.h#L111-L155) 的 `sync_remote_blocks<hasPrev, hasSubseq>`（详见 §8）。

**⑧ 同步到 stream**
整条链路默认复用 caller 的当前 CUDA stream，kernel 返回即 enqueue 完成；外层同步语义由用户自己控制（`torch.cuda.synchronize` / event）。

---

## 7. Inductor / torch.compile 集成

### 7.1 Lowering 决策

[torch/_inductor/comm_lowering.py:147-167](torch/_inductor/comm_lowering.py#L147-L167)

```python
def _should_lower_as_one_shot_all_reduce(inp, reduce_op, group_name):
    from torch.distributed._symmetric_memory import is_symm_mem_enabled_for_group
    inp_size = inp.get_numel() * inp.get_dtype().itemsize
    return (
        config._collective.auto_select
        and is_symm_mem_enabled_for_group(group_name)
        and can_realize_as_comm_buffer(inp, ir.CommBufferType.SYMM_MEM)
        and reduce_op in ("sum",)
        and inp_size <= config._collective.one_shot_all_reduce_threshold_bytes
    )
```

四个条件：①组开启 symm mem；②输入可以被 realize 到 symm mem buffer；③`sum`；④大小在阈值内（默认适合 one-shot 的小~中消息）。

### 7.2 Realize 成通信 buffer

Inductor IR 侧新增了 `CommBufferType.SYMM_MEM`（搜索 [torch/_inductor/ir.py](torch/_inductor/ir.py) 的 `CommBufferType`）。命中条件时 Inductor 把该张量在编译期就钉到对称内存（调用 `empty_strided_p2p`），使 rendezvous 只发生一次，compile 图里直接跑 `one_shot_all_reduce`。

### 7.3 async-TP 等更激进的重写

`torch/_inductor/fx_passes/micro_pipeline_tp.py` 里有配套 pass，把"matmul + all_reduce"改写成 pipelined 版本，依赖对称内存做 producer/consumer 的跨 rank 写入。

---

## 8. signal_pad 同步协议

### 8.1 单槽语义

[CUDASymmetricMemory-inl.h:33-109](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.h#L33-L109)：
- 每个槽是 `uint32_t`，**约定 0 = 空闲、1 = 有信号**，使用前后都必须回到 0。
- `put_signal`：CAS(0→1) 直到成功；把"我完成了"的信号写入 peer。
- `wait_signal`：CAS(1→0) 直到成功；读到 peer 的信号并立刻清零。
- 两种内存序参数（`std::memory_order_{release,acquire,relaxed}`）在 CAS 上被翻译成 CUDA `system_scope` 的 atomic，保证跨 GPU 的可见性。

### 8.2 跨 block 同步：`sync_remote_blocks`

[CUDASymmetricMemory-inl.h:111-155](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.h#L111-L155) 三种使用模式：

| 模板参数 `<hasPrev, hasSubseq>` | put 语义 | wait 语义 | 典型用途 |
|---|---|---|---|
| `<false,true>` | relaxed | acquire | 进入阶段前的 barrier-in |
| `<true,true>`  | release | acquire | 阶段之间的完整 barrier |
| `<true,false>` | release | relaxed | 阶段结束后的 barrier-out |

thread 0..world_size-1 里每个线程负责"和一个 peer 对信号"。整块之间靠 `__syncthreads()` 再收拢。槽位编址：`addr = signal_pads[target] + blockIdx.x * world_size + self_rank`，所以最多 32 个 block × world_size 个 peer 互不踩。

### 8.3 Host barrier

[CUDASymmetricMemory.cu:166-201](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.cu#L166-L201) 的 `barrier_kernel`：每个线程对一个 peer 做 `try_put_signal<release>` + `try_wait_signal<acquire>`；任意一侧超时就 `trap()`（fatal）。这是 `symm_mem_hdl.barrier(channel, timeout_ms)` 的落地。

### 8.4 使用约定

- signal_pad 是组内共享资源：不同算子用不同 `channel`（blockIdx.x 映射）避免冲突；
- **算子必须把用到的槽恢复成 0**，否则下次使用会死锁，这也是 §1 里那条头文件注释强调的不变量。

---

## 9. 局限、平台 gating 与已知坑

### 9.1 构建开关（CMake）

[caffe2/CMakeLists.txt:583-590](caffe2/CMakeLists.txt#L583-L590) 附近：
- 基本功能：`PYTORCH_C10_DRIVER_API_SUPPORTED=1`；
- NVSHMEM：`USE_NVSHMEM=1`（链接 `libnvshmem`）；
- NCCL backend：`USE_C10D_NCCL=1` 且 NCCL ≥ 2.27.1。

### 9.2 运行时环境变量

（[CUDASymmetricMemoryUtils.cpp:21-42](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.cpp#L21-L42) 与 [env.hpp](torch/csrc/distributed/c10d/symm_mem/env.hpp)）

| 变量 | 作用 |
|---|---|
| `TORCH_SYMMMEM` | 选定默认 backend（`CUDA`/`NVSHMEM`/`NCCL`） |
| `TORCH_SYMM_MEM_DISABLE_MULTICAST=1` | 即使硬件支持也禁用 multicast |
| `TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES=1` | 允许同一物理 device 出现多次（开发/测试用） |
| `TORCH_SYMMMEM_NBLOCKS=N` | 覆盖 `symm_max_nblocks`（32） |

### 9.3 硬件/版本矩阵

- CUDA ≥ 12.0 方可用 driver API；
- 多播：CUDA ≥ 12.3 且 SM ≥ 90（H100/B100/B200）；
- Fabric handle：CUDA ≥ 12.3，且平台启用了 NVLink SHARP / UAFS；
- 默认 intra-node 快路径要求 SM ≥ 80（[intra_node_comm.cu:25-31](torch/csrc/distributed/c10d/symm_mem/intra_node_comm.cu#L25-L31)）。

### 9.4 Rendezvous 约束

[CUDASymmetricMemory.cu:450-476](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.cu#L450-L476) 的 `validate_rendezvous_requests`：
- 所有 rank 的 `size` 与 `device_type` 必须一致；
- 默认不允许同一物理 device 被不同 rank 重复注册（除非开关放行）。

### 9.5 规模上限

- [CUDASymmetricMemoryTypes.hpp:8-10](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryTypes.hpp#L8-L10)：`max_cuda_p2p_domain_size = 72`（对齐 NVL72），`symm_max_nblocks = 32`。
- 单次集合参与 rank > 72 或同时跑 > 32 个 block 的 symm 内核会在注册/校验处失败；超大规模应走 NVSHMEM 多机路径。

### 9.6 多节点

| Backend | intra-node | multi-node |
|---|---|---|
| CUDA | ✓ | CUDA 12.3+ fabric handle 下可行 |
| NVSHMEM | ✓ | ✓（首选） |
| NCCL | ✓ | ✓，但高级同步原语未完成 |

### 9.7 持久分配的生命周期

[SymmetricMemory.hpp:163-172](torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp#L163-L172)：
> If an alloc_id is supplied, empty_strided_p2p will perform persistent allocation... For safety, if a previous persistent allocation is still active, persistent allocations with the same alloc_id will fail.

Inductor 在编译期会复用同一 alloc_id，**Python 侧必须等上一个 tensor 真正析构**，否则二次 compile 会报重复注册。

### 9.8 常见踩坑

- **signal_pad 不清零**：算子 kernel 崩溃或早退时没把槽恢复到 0，下一次调用直接死锁。
- **异 stream 访问 peer buffer**：peer 指针只是普通 device ptr，没有内置 stream 依赖；跨流访问需要自己 record/wait event。
- **不同 rank 申请大小不同**：`rendezvous` 阶段 abort，错误信息定位在 `validate_rendezvous_requests`。
- **torch.compile 认为可 lower 但运行时 PG 未开启 symm mem**：会 fallback 到普通 NCCL；可 `TORCH_LOGS=inductor` 看决策。

---

## 10. 参考文件清单

### 核心抽象
- [torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp](torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp) — 两个抽象基类、op schema 声明
- [torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.cpp](torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.cpp) — dispatcher 注册、meta kernel
- [torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryTypes.hpp](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryTypes.hpp) — 常量（`symm_max_nblocks`、`signal_pad_size`）
- [torch/csrc/distributed/c10d/symm_mem/env.hpp](torch/csrc/distributed/c10d/symm_mem/env.hpp) — 环境变量

### CUDA backend
- [CUDASymmetricMemory.hpp](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.hpp)
- [CUDASymmetricMemory.cu](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.cu) — 分配、rendezvous、multicast、barrier
- [CUDASymmetricMemory-inl.h](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.h) — CAS / sync_remote_blocks / multimem
- [CUDASymmetricMemoryOps.cu](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryOps.cu) — `one_shot_all_reduce` 等 kernel
- [CUDASymmetricMemoryUtils.hpp](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp) / [CUDASymmetricMemoryUtils.cpp](torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.cpp) — StoreExchange、IpcChannel、后端选择
- [CudaDMAConnectivity.cpp](torch/csrc/distributed/c10d/symm_mem/CudaDMAConnectivity.cpp) — P2P 连接探测
- [cuda_mem_pool.cpp](torch/csrc/distributed/c10d/symm_mem/cuda_mem_pool.cpp) — 和 caching allocator 的接入
- [intra_node_comm.{hpp,cpp,cu}](torch/csrc/distributed/c10d/symm_mem/intra_node_comm.hpp) — intra-node 集合 fast path

### NVSHMEM backend
- [NVSHMEMSymmetricMemory.cu](torch/csrc/distributed/c10d/symm_mem/NVSHMEMSymmetricMemory.cu)
- [nvshmem_extension.cu](torch/csrc/distributed/c10d/symm_mem/nvshmem_extension.cu) / [nvshmem_extension.cuh](torch/csrc/distributed/c10d/symm_mem/nvshmem_extension.cuh)
- [nvshmem_team_manager.hpp](torch/csrc/distributed/c10d/symm_mem/nvshmem_team_manager.hpp)
- [torch/distributed/_symmetric_memory/_nvshmem_triton.py](torch/distributed/_symmetric_memory/_nvshmem_triton.py)

### NCCL backend
- [NCCLSymmetricMemory.cu](torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.cu)
- 设计笔记：[nccl_symm.md](nccl_symm.md)、[nccl_symm_cc.md](nccl_symm_cc.md)、[nccl_register_vs_window_register.md](nccl_register_vs_window_register.md)

### Python / 绑定
- [torch/distributed/_symmetric_memory/__init__.py](torch/distributed/_symmetric_memory/__init__.py)
- [torch/csrc/distributed/c10d/init.cpp](torch/csrc/distributed/c10d/init.cpp) — `_SymmetricMemory` pybind

### Inductor / torch.compile
- [torch/_inductor/comm_lowering.py](torch/_inductor/comm_lowering.py)
- [torch/_inductor/ir.py](torch/_inductor/ir.py)（`CommBufferType.SYMM_MEM`）
- [torch/_inductor/fx_passes/micro_pipeline_tp.py](torch/_inductor/fx_passes/micro_pipeline_tp.py)

### 测试
- [test/distributed/test_symmetric_memory.py](test/distributed/test_symmetric_memory.py)
- [test/distributed/test_nccl.py](test/distributed/test_nccl.py)
- [test/distributed/test_nvshmem.py](test/distributed/test_nvshmem.py)
- [test/distributed/test_nvshmem_triton.py](test/distributed/test_nvshmem_triton.py)
- [test/distributed/tensor/parallel/test_micro_pipeline_tp.py](test/distributed/tensor/parallel/test_micro_pipeline_tp.py)

### 构建
- [caffe2/CMakeLists.txt](caffe2/CMakeLists.txt)（搜索 `USE_NVSHMEM` / `symm_mem`）

---

**一句话总结**：`SymmetricMemory` = "组内每张卡都持有对称 buffer + signal_pad，对外暴露 peer 指针" 的抽象；CUDA/NVSHMEM/NCCL 三个 backend 在 rendezvous 机制上分别对应 IPC/fabric handle、`nvshmem_ptr`、`ncclCommWindowRegister`；上层集合原语（`one_shot_all_reduce` 等）都是"barrier-in → peer load/store → barrier-out"的小 CUDA kernel，由 Inductor 在合适大小下自动 lower 调用。
