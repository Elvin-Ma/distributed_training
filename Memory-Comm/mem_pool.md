# MemPool

`MemPool` 是 PyTorch CUDA 内存管理中的**私有内存池**抽象。

CUDA 默认的 `CUDACachingAllocator` 管理一个全局内存池，所有 tensor 共享。`MemPool` 允许用户或 `CUDAGraph` 创建独立的内存池，实现：

- **内存隔离**：不同任务的内存分配互不干扰
- **CUDA Graph 支持**：CUDAGraph 需要在 capture 期间将所有分配限定到同一个池，replay 时才能复用相同的内存地址
- **OOM 策略控制**：`use_on_oom` — OOM 时回退到此池；`no_split` — 禁止内存块分割

```c++
// c10/core/CachingDeviceAllocator.h
using CaptureId_t  = unsigned long long;
using MempoolId_t  = std::pair<CaptureId_t, CaptureId_t>;
//                              ^first            ^second
//                              uuid_ (CUDAGraph自动创建)   uid_ (用户主动创建)


struct TORCH_CUDA_CPP_API MemPool {
  MemPool(
      c10::cuda::CUDACachingAllocator::CUDAAllocator* allocator = nullptr,
      bool is_user_created = true,
      bool use_on_oom = false,
      bool no_split = false);
  MemPool(const MemPool&) = delete;
  MemPool(MemPool&&) = default;
  MemPool& operator=(const MemPool&) = delete;
  MemPool& operator=(MemPool&&) = default;
  ~MemPool();

  MempoolId_t id();
  c10::cuda::CUDACachingAllocator::CUDAAllocator* allocator();
  int use_count();
  c10::DeviceIndex device();
  static MempoolId_t graph_pool_handle(bool is_user_created = true);

 private:
  static std::atomic<CaptureId_t> uid_;
  static std::atomic<CaptureId_t> uuid_;
  c10::cuda::CUDACachingAllocator::CUDAAllocator* allocator_;
  bool is_user_created_;
  MempoolId_t id_;
  c10::DeviceIndex device_;
};
```

# 2 相关数据结构解释

## 2.1 `MempoolId_t` 的结构

`MempoolId_t` 是一个 pair，**两个字段只有一个非零**，用哪个字段来区分池的创建来源：

| 场景 | `id_.first` (uuid) | `id_.second` (uid) |
|---|---|---|
| 用户主动创建（`is_user_created=true`） | `0` | `uid_++` |
| CUDAGraph 自动创建（`is_user_created=false`） | `uuid_++` | `0` |
| 未传任何池（哨兵值） | `0` | `0` |

- `MempoolId_t` 的作用：DeviceCachingAllocator 创建的池会注册到 `CUDACachingAllocator::pools_` 中，以 `MempoolId_t` 为 key。

```c++
class DeviceCachingAllocator {
   // Private pools for CUDA graphs
  ska::flat_hash_map<MempoolId_t, std::unique_ptr<PrivatePool>, MempoolIdHash>
      graph_pools;
  // Pools no longer referenced by any graph. Their BlockPools are eligible for
  // free_blocks. Can't be a vector or deque because we might erase entries in
  // any order. Could be an std::list, but we don't care much, access and
  // insert/erase are rare.
  ska::flat_hash_map<MempoolId_t, PrivatePool*, MempoolIdHash>
      graph_pools_freeable;
}
```

## 2.2 `uid_` vs `uuid_` 的原理

两者都是**类级静态原子计数器**，每次创建新池时自增，保证全局唯一性：

```cpp
if (is_user_created_) {
    id_ = {0, uid_++};   // second 字段存 uid
} else {
    id_ = {uuid_++, 0};  // first 字段存 uuid
}
```

**为什么从 1 开始而不是 0？** 因为 `{0, 0}` 被用作哨兵值，表示"没有传入任何池"。从 1 开始确保任何真实池的 id 都不会与哨兵值冲突。

# 3 使用逻辑

## 3.1 生命周期

```sh
构造: createOrIncrefPool(device_, id_)  → 在 allocator 里注册/引用计数+1
共享: CUDAGraph 持有同一个池时 use_count() 会 > 1
析构: releasePool(device_, id_)         → 引用计数-1
      emptyCache(id_)                   → 释放该池的 GPU 内存
```

## 3.2 对应关系
CUDAGraph、MemPool、PrivatePool 的对应关系:

```sh
MemPool (上层C++/Python对象)
    │  1:1
    ▼
PrivatePool (allocator内部数据结构)
    │  1:N
    ▼
CUDAGraph (可共享同一个 pool) # 共享显存可以减少碎片和峰值占用
```

CUDAGraph::capture_begin 时有两条路径：
- 路径 A：不传 pool（默认，每个 graph 独占一个 pool）;
- 路径 B：传入已有的 pool.id()（多个 graph 共享同一 pool）, 有两种方式：
```sh
# 方式1: 从另一个 graph 的 pool id 共享
g2.capture_begin(pool=g1.pool())

# 方式2: 用 graph_pool_handle 预先创建共享句柄
handle = torch.cuda.graph_pool_handle()
g1.capture_begin(pool=handle)
g2.capture_begin(pool=handle)
```
- cudagraph.reset 时，会释放所有 graph 引用的 pool，同时销毁trace 的graph

```sh
用户层:
  MemPool A ──────────────────────────────────────────── MemPool B
     │ id={0,1}                                            │ id={0,2}
     │                                                     │
allocator层 (graph_pools 哈希表):
  PrivatePool{id={0,1}}                             PrivatePool{id={0,2}}
  use_count=3                                       use_count=1
     │ ↑ 共享                                              │
     ├── CUDAGraph G1 (capture_begin(pool=A.id()))        │
     ├── CUDAGraph G2 (capture_begin(pool=A.id()))        │
     └── CUDAGraph G3 (capture_begin(pool=A.id()))        │
                                                          └── CUDAGraph G4 (独占)
```

## 3.3 什么时候会用 MemPool 呢？

```sh
beginAllocateToPool(pool_id, tid_filter)   ← 开启"重定向窗口"
at::empty({size}, options)                 ← 这次 malloc 被路由到 memPool_
endAllocateToPool(pool_id)                 ← 关闭"重定向窗口"
```

