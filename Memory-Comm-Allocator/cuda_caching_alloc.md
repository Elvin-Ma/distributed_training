# CUDA Caching Allocator 实现原理与配置详解

## 1. 概述

PyTorch 的 CUDA Caching Allocator（`c10/cuda/CUDACachingAllocator.cpp`）是一个高性能的 GPU 内存缓存分配器。其核心思想是：**避免频繁调用昂贵的 `cudaMalloc`/`cudaFree`，通过缓存已释放的内存块并在后续请求中复用，从而大幅降低分配开销。**

## 2. UML 类图

```
┌─────────────────────────────────────────────────────────────────────┐
│                      NativeCachingAllocator                         │
│                      «extends CUDAAllocator»                        │
├─────────────────────────────────────────────────────────────────────┤
│ - enable_: bool                                                     │
│ - mutex: AlignedMutex[67]              // 67 路分片锁                │
│ - allocated_blocks: flat_hash_map<void*, Block*>[67]                │
│ - clock_converter: ApproximateClockToUnixTimeConverter              │
│ - record_history: bool                                              │
│ - annotation_buffer: RingBuffer<AnnotationEntry>                    │
├─────────────────────────────────────────────────────────────────────┤
│ + device_allocator: vector<unique_ptr<DeviceCachingAllocator>>      │
│ + get_allocated_block(ptr, remove): Block*                          │
│ + malloc(ptr, device, size, stream)                                 │
│ + free(ptr)                                                         │
│ + emptyCache()                                                      │
│ + getDeviceStats(device): DeviceStats                               │
│ + snapshot(): SnapshotInfo                                          │
└─────────────────────┬───────────────────────────────────────────────┘
                      │ 1..N (每个 GPU 一个)
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     DeviceCachingAllocator                          │
├─────────────────────────────────────────────────────────────────────┤
│ - mutex: recursive_mutex                                            │
│ - stats: DeviceStats                                                │
│ - device_id: DeviceIndex                                            │
│ - device_prop: cudaDeviceProp                                       │
│ - total_allocated_memory: size_t                                    │
│ - allowed_memory_maximum: optional<size_t>                          │
│ - record_history: bool                                              │
│ - record_context_: RecordContext                                    │
│ - captures_underway: vector<pair<MempoolId_t, function>>            │
│ - deferred_blocks: flat_hash_map<Block*, vector<cudaGraphNode_t>>   │
│ - oom_observers_: vector<OutOfMemoryObserver>                       │
├─────────────────────────────────────────────────────────────────────┤
│ - large_blocks: BlockPool                      ──────┐              │
│ - small_blocks: BlockPool                      ──────┤              │
│ - active_blocks: flat_hash_set<Block*>         ──────┤ 持有         │
│ - expandable_segments_: vector<ExpandableSegment*> ──┤              │
│ - graph_pools: flat_hash_map<MempoolId_t,      ──────┤              │
│     unique_ptr<PrivatePool>>                         │              │
│ - graph_pools_freeable: flat_hash_map<...,     ──────┤              │
│     PrivatePool*>                                    │              │
│ - cuda_events: flat_hash_map<CUDAStream,       ──────┘              │
│     deque<pair<Event, Block*>>>                                     │
│ - alloc_buffer: RingBuffer<TraceEntry>                              │
├─────────────────────────────────────────────────────────────────────┤
│ + malloc(size, stream, requested_size, ctx): Block*                 │
│ + free(block: Block*)                                               │
│ + setMemoryFraction(fraction)                                       │
│ + emptyCache()                                                      │
│ + recordHistory(enabled, context_recorder, ...)                     │
│ + snapshot(): vector<SegmentInfo>                                   │
│ + beginAllocateToPool(mempool_id, filter)                           │
│ + endAllocateToPool(mempool_id)                                     │
│ + releasePool(mempool_id)                                           │
│ - get_free_block(params: AllocParams): bool                         │
│ - alloc_block(params: AllocParams, ...): bool                       │
│ - alloc_found_block(params: AllocParams): Block*                    │
│ - free_block(block: Block*, context)                                │
│ - try_merge_blocks(block, prev, pool): size_t                       │
│ - should_split(block, size): bool                                   │
│ - release_cached_blocks(context): bool                              │
│ - garbage_collect_cached_blocks(context): bool                      │
│ - process_events(context)                                           │
│ - insert_events(block: Block*)                                      │
└──────────┬──────────────────────┬──────────────────┬────────────────┘
           │                      │                  │
    ┌──────▼──────┐    ┌──────────▼────────┐  ┌──────▼──────────────┐
    │  BlockPool  │    │  PrivatePool      │  │ ExpandableSegment   │
    ├─────────────┤    ├───────────────────┤  ├─────────────────────┤
    │ blocks:     │    │ id: MempoolId_t   │  │ - device_: DeviceIdx│
    │  set<Block*>│    │ use_count: int    │  │ - stream_: optional │
    │  (按size排序)│    │ cudaMalloc_count: │  │   <cudaStream_t>    │
    │ unmapped:   │    │   int             │  │ - segment_size_:    │
    │  set<Block*>│    │ allocator_:       │  │   size_t            │
    │  (按addr排序)│    │  CUDAAllocator*   │  │ - ptr_: CUdeviceptr │
    │ is_small:   │    ├───────────────────┤  │ - max_handles_:     │
    │   bool      │    │ large_blocks:     │  │   size_t            │
    │ owner_      │◄───│   BlockPool ──────┤  │ - mapped_size_:     │
    │  PrivatePool│    │ small_blocks:     │  │   size_t            │
    │ get_free_   │    │   BlockPool ──────┤  │ - handles_: vector  │
    │  blocks_    │    └───────────────────┘  │   <optional<        │
    │  call_count │                           │   CUmemGenericAlloc │
    ├─────────────┤                           │   ationHandle>>     │
    │ insert_into │                           │ - peers_: vector    │
    │  _blocks()  │                           │   <DeviceIndex>     │
    │ owner_      │                           ├─────────────────────┤
    │  MempoolId()│                           │ + map(range):       │
    └──────┬──────┘                           │   SegmentRange      │
           │ 0..N                             │ + unmap(ptr, size): │
           ▼                                  │   bool              │
┌───────────────────────────────────┐         │ + ptr(): char*      │
│              Block                │         │ + size(): size_t    │
├───────────────────────────────────┤         └──────────┬──────────┘
│ device: DeviceIndex               │                    │
│ stream: cudaStream_t              │                    │
│ stream_uses: stream_set           │         ┌──────────▼──────────┐
│ size: size_t                      │         │   SegmentRange      │
│ requested_size: size_t            │         ├─────────────────────┤
│ pool: BlockPool*            ──────┤────►    │ ptr: char*          │
│ ptr: void*                        │         │ size: size_t        │
│ allocated: bool                   │         └─────────────────────┘
│ mapped: bool                      │
│ event_count: int                  │
│ gc_count_base: int64_t            │
│ context_when_allocated:           │
│   shared_ptr<GatheredContext>     │
│ context_when_segment_allocated:   │
│   shared_ptr<GatheredContext>     │
│ expandable_segment_:              │
│   ExpandableSegment*         ─────┤────► ExpandableSegment
├───────────────────────────────────┤
│ prev: Block*   ◄──┐  ┌──► next:   │   Block 双向链表
│                   │  │   Block*   │   (分裂产生的相邻块)
│ gc_count(): size_t│  │            │
│ is_split(): bool  └──┘            │
│ splice(before, after)             │
└───────────────────────────────────┘

┌───────────────────────────────────┐  ┌──────────────────────────────┐
│           EventPool               │  │       AllocParams            │
├───────────────────────────────────┤  ├──────────────────────────────┤
│ - pools_: vector<PerDevicePool>   │  │ search_key: Block            │
├───────────────────────────────────┤  │ pool: BlockPool*             │
│ + get(device): Event              │  │ alloc_size: size_t           │
│ + empty_cache()                   │  │ is_expandable_segments_      │
├───────────────────────────────────┤  │ active: bool                 │
│ «inner» PerDevicePool             │  │ block: Block*                │
│   mutex_: mutex                   │  │ stat_types: StatTypes        │
│   event_pool_: vector<cudaEvent_t>│  │ err: cudaError_t             │
└───────────────────────────────────┘  ├──────────────────────────────┤
                                       │ + device(): DeviceIndex      │
┌───────────────────────────────────┐  │ + stream(): cudaStream_t     │
│        RingBuffer<T>              │  │ + size(): size_t             │
├───────────────────────────────────┤  └──────────────────────────────┘
│ - alloc_trace_max_entries_: size_t│
│ - alloc_trace_lock: mutex         │  ┌──────────────────────────────┐
│ - alloc_trace_next: size_t        │  │  BlockState (checkpoint)     │
│ - alloc_trace: vector<T>*         │  ├──────────────────────────────┤
├───────────────────────────────────┤  │ device: DeviceIndex          │
│ + setMaxEntries(size)             │  │ stream: cudaStream_t         │
│ + insertEntries(entry)            │  │ stream_uses: stream_set      │
│ + getEntries(result)              │  │ size: size_t                 │
│ + clear()                         │  │ ptr: void*                   │
└───────────────────────────────────┘  │ allocated: bool              │
                                       │ gc_count_base: int64_t       │
┌───────────────────────────────────┐  └──────────────────────────────┘
│  PrivatePoolState                 │
│  «extends AllocatorState»         │  ┌──────────────────────────────┐
├───────────────────────────────────┤  │  SegmentState                │
│ owner_id: MempoolId_t             │  ├──────────────────────────────┤
│ segments: vector<SegmentState>  ──┤─►│ blocks: vector<BlockState>   │
└───────────────────────────────────┘  │ is_small: bool               │
                                       └──────────────────────────────┘
```

### 关键关系说明

```
NativeCachingAllocator ──1:N──► DeviceCachingAllocator    (每 GPU 一个)
DeviceCachingAllocator ──1:2──► BlockPool                 (large + small)
DeviceCachingAllocator ──1:N──► PrivatePool               (CUDA Graph 私有池)
DeviceCachingAllocator ──1:N──► ExpandableSegment          (可扩展虚拟内存段)
PrivatePool            ──1:2──► BlockPool                 (独立的 large + small)
BlockPool              ──1:N──► Block                     (空闲块集合)
Block                  ──1:1──► BlockPool                 (所属池，反向引用)
Block                  ◄─prev/next─► Block                (分裂链表，双向)
Block                  ──0:1──► ExpandableSegment          (可选，所属可扩展段)
BlockPool              ──0:1──► PrivatePool               (可选，所属私有池)
EventPool              ──1:N──► PerDevicePool             (每设备的 event 缓存)
```

## 3. 核心数据结构

### 2.1 Block（内存块）

```
struct Block {
    DeviceIndex device;          // GPU 设备号
    cudaStream_t stream;         // 关联的 CUDA stream
    stream_set stream_uses;      // 使用过该 block 的 stream 集合
    size_t size;                 // 实际块大小（可能大于请求大小）
    size_t requested_size;       // 用户请求的大小
    BlockPool* pool;             // 所属内存池
    void* ptr;                   // GPU 内存地址
    bool allocated;              // 是否正在使用
    bool mapped;                 // 虚拟地址是否映射了物理内存（用于 expandable segments）
    Block* prev / next;          // 分裂产生的相邻块链表
    int event_count;             // 未完成的跨 stream CUDA event 数量
    int64_t gc_count_base;       // 用于 GC 年龄计算
    ExpandableSegment* expandable_segment_;  // 所属可扩展段
};
```

Block 是分配器的基本单元。大块可以被**分裂（split）**为小块，相邻的空闲块可以被**合并（merge）**。

### 2.2 BlockPool（内存池）

```
struct BlockPool {
    std::set<Block*, Comparison> blocks;    // 空闲块集合，按 (stream, size, addr) 排序
    std::set<Block*, Comparison> unmapped;  // 未映射块集合，按地址排序
    bool is_small;                          // 是否为小块池
    PrivatePool* owner_PrivatePool;         // 所属私有池（CUDA Graph 用）
};
```

分配器维护两个主池：
- **Small Pool**：用于 ≤ 1MB 的分配
- **Large Pool**：用于 > 1MB 的分配

### 2.3 DeviceCachingAllocator（设备级分配器）

每个 GPU 设备一个实例，管理该设备上的所有内存分配。持有 `large_blocks`、`small_blocks` 两个 BlockPool，以及 `active_blocks` 集合追踪所有活跃分配。

### 2.4 PrivatePool（私有池）

为 CUDA Graph 提供隔离的内存池。Graph 捕获期间的分配从私有池中获取，确保 replay 时地址有效。

## 3. 内存分配流程

```
malloc(size, stream)
│
├─ 1. process_events()          // 处理跨 stream 的 event 完成通知
│
├─ 2. round_size(size)          // 对齐请求大小
│     ├─ 最小 512 字节 (kMinBlockSize)
│     ├─ 支持 power-of-2 分区对齐
│     └─ 否则对齐到 512 字节的倍数
│
├─ 3. get_pool(size)            // 选择内存池
│     ├─ size ≤ 1MB → small_blocks
│     └─ size > 1MB → large_blocks
│
├─ 4. get_free_block()          // Best-Fit 搜索空闲块
│     ├─ 优先同 stream 的块（避免同步）
│     ├─ 按大小排序，选最小满足要求的块
│     └─ expandable segment 可动态扩展
│
├─ 5. [未找到] trigger_free_memory_callbacks()  // 触发外部释放回调
├─ 6. [未找到] garbage_collect_cached_blocks()  // GC 回收
├─ 7. [未找到] alloc_block()    // 分配新 GPU 内存
│     ├─ 尝试 expandable segment 扩展
│     └─ 回退到 cudaMalloc
│
├─ 8. [cudaMalloc 失败] release_available_cached_blocks()  // 释放大块缓存
├─ 9. [仍然失败] release_cached_blocks()                    // 清空全部缓存
│
└─ 10. alloc_found_block()      // 可能分裂块，返回给用户
```

### 分配大小策略

| 请求大小 | 实际分配大小 | 所在池 |
|---------|------------|-------|
| ≤ 1MB | 2MB 缓冲区 (kSmallBuffer) | Small Pool |
| 1MB ~ 10MB | 20MB 缓冲区 (kLargeBuffer) | Large Pool |
| ≥ 10MB | 向上对齐到 2MB 的倍数 | Large Pool |

## 4. 内存释放与块合并

```
free(block)
│
├─ 标记 block.allocated = false
│
├─ 跨 stream 使用？
│   ├─ CUDA Graph 捕获中 → 延迟释放，记录 free marker
│   └─ 普通情况 → insert_events() 在所有使用 stream 上记录 cudaEvent
│
└─ free_block()（当所有 event 完成后）
    ├─ try_merge_blocks() 与前后相邻块合并
    │   └─ 条件：空闲 + 同 stream + event_count==0 + 相同 mapped 状态
    ├─ 返回块到 pool.blocks 集合
    └─ 更新统计信息
```

### 块分裂条件 (should_split)

- **Small Pool**：剩余部分 ≥ 512 字节时分裂
- **Large Pool**：剩余部分 > 1MB **且** 分配大小 < `max_split_size` 时分裂
- **Expandable Segments**：剩余部分 ≥ 512 字节时始终分裂
- **超大块**（≥ `max_split_size`）：**不分裂**，减少碎片化。但仍会满足差距在 1MB 以内的请求

## 5. 跨 Stream 内存复用

**问题**：Block 在 stream A 上分配，在 stream B 上使用后释放，不能立即复用，因为 stream B 可能还未完成。

**解决方案 — 基于 Event 的同步**：

1. `recordStream(block, streamB)`：标记 block 被 streamB 使用
2. 释放时，`insert_events()` 在 streamB 上记录 cudaEvent
3. `process_events()` 定期检查 event 状态，完成后才真正释放 block

## 6. Expandable Segments（可扩展段）

### 问题背景

传统模式下，每次 `cudaMalloc` 分配固定大小的段。当 batch size 微变（如 N → N+1），已有段无法完全复用，产生大量不可回收的内存碎片。

### 解决方案

利用 CUDA Driver API（`cuMemCreate`/`cuMemAddressReserve`/`cuMemMap`）将虚拟地址分配与物理内存映射分离：

1. **预留巨大虚拟地址空间**（约 1.125 × GPU 总显存，256TiB 地址空间足够）
2. **按需映射物理内存**：Small Pool 以 2MB 为页，Large Pool 以 20MB 为页
3. **OOM 时可回收**：`unmap()` 释放空闲页的物理内存归还给 CUDA

### 优势

- 多次分配可拼接到同一段中，减少碎片
- Batch size 变化时，新分配自然追加到已有段的末尾
- OOM 时可通过 unmap 回收空闲物理页

### 限制

- 初始分配速度略慢（毫秒级）
- 不支持 CUDA tensor 的 IPC（跨进程共享）
- `cudaDeviceEnablePeerAccess` 不适用于 `cuMemMap` 分配的内存，需用 allocator 提供的 `enablePeerAccess` 方法

## 7. 垃圾回收（GC）

**触发条件**：当已使用内存达到 `garbage_collection_threshold` 比例时触发。

**算法**：
1. 计算所有可释放块（非分裂块）的平均年龄
2. 优先释放年龄 > 平均值的块
3. 持续释放直到回收量 ≥ 目标（多出的已分配内存量）
4. 年龄由 `gc_count = pool.get_free_blocks_call_count - block.gc_count_base` 计算

## 8. CUDA Graph 支持

### 私有池生命周期

1. `beginAllocateToPool(mempool_id)`：创建/引用计数 +1，所有分配路由到私有池
2. Graph 捕获期间，分配/释放在私有池内正常进行
3. `endAllocateToPool()`：捕获结束
4. `releasePool()`：引用计数 -1；为 0 时池变为可释放状态
5. `emptyCache()`：释放可释放池中的块

### 捕获期间的跨 Stream 延迟释放

- 带跨 stream 使用的块存入 `deferred_blocks`，不立即释放
- 捕获结束后调用 `insert_events_deferred_until_no_capture()` 处理

## 9. 配置选项

通过环境变量 `PYTORCH_CUDA_ALLOC_CONF` 配置，格式为逗号分隔的 `key:value` 对。

### 核心配置

| 配置项 | 默认值 | 说明 |
|-------|-------|------|
| `max_split_size` | 无限大 | 超过此大小的块不允许分裂，减少碎片。单位 MB |
| `roundup_power2_divisions` | 0 | 按 2 的幂次分区对齐请求大小，减少碎片 |
| `max_non_split_rounding_size` | - | 非分裂块的最大对齐大小 |
| `garbage_collection_threshold` | 0.0（禁用） | 触发 GC 的已用内存比例 (0.0~1.0) |
| `expandable_segments` | False | 启用可扩展段（虚拟内存映射方式） |

### 性能调优配置

| 配置项 | 默认值 | 说明 |
|-------|-------|------|
| `release_lock_on_cudamalloc` | False | cudaMalloc 期间释放分配器锁，减少锁竞争 |
| `per_process_memory_fraction` | 1.0 | 限制进程可用 GPU 内存比例 (0.0~1.0) |
| `graph_capture_record_stream_reuse` | False | Graph 感知的内存复用优化 |

### Pinned Memory 配置

| 配置项 | 默认值 | 说明 |
|-------|-------|------|
| `pinned_use_cuda_host_register` | False | 使用 cudaHostRegister 注册 pinned memory |
| `pinned_num_register_threads` | 1 | 注册线程数（最大 128） |
| `pinned_reserve_segment_size_mb` | 0 | 预留 pinned memory 段大小 |

### 配置示例

```bash
# 限制最大分裂块大小为 512MB，启用可扩展段，GC 阈值 80%
export PYTORCH_CUDA_ALLOC_CONF=max_split_size:512,expandable_segments:True,garbage_collection_threshold:0.8

# 多进程共享 GPU，每进程限制 50% 显存
export PYTORCH_CUDA_ALLOC_CONF=per_process_memory_fraction:0.5

# 减少碎片：按 2 的幂次对齐
export PYTORCH_CUDA_ALLOC_CONF=roundup_power2_divisions:4
```

## 10. 锁与并发

### 设备级锁

每个设备一个 `std::recursive_mutex`，保护分配器所有状态。

### allocated_blocks 分片

`NativeCachingAllocator` 对 `allocated_blocks` 哈希表使用 **67 路分片**锁（`twang_mix64(ptr) % 67`），大幅降低分配查找时的锁竞争。

### release_lock_on_cudamalloc

开启后，在执行 `cudaMalloc` 时临时释放分配器锁，避免阻塞其他线程的分配操作。

## 11. 内存快照与追踪

分配器支持记录详细的内存追踪信息用于调试：

**追踪事件类型**：
- `ALLOC` / `FREE_REQUESTED` / `FREE_COMPLETED`：用户侧分配/释放
- `SEGMENT_ALLOC` / `SEGMENT_FREE`：cudaMalloc/cudaFree 级操作
- `SEGMENT_MAP`：expandable segment 的内存映射
- `OOM`：内存不足错误

**上下文记录级别**（`RecordContext`）：
- `NEVER`：不记录调用栈
- `STATE`：仅记录活跃分配的调用栈
- `ALLOC`：+ 历史分配记录
- `ALL`：+ 释放记录

使用 `torch.cuda.memory._snapshot()` 可获取当前内存状态快照。

## 12. 常见 OOM 场景与调优建议

| 场景 | 建议配置 |
|------|---------|
| Batch size 微变导致碎片 | `expandable_segments:True` |
| 大块内存被分裂后无法合并 | `max_split_size:512`（限制分裂上限） |
| 频繁分配/释放产生碎片 | `garbage_collection_threshold:0.8` |
| 多进程抢占 GPU 内存 | `per_process_memory_fraction:0.5` |
| 小块分配碎片化 | `roundup_power2_divisions:4` |
| cudaMalloc 阻塞其他线程 | `release_lock_on_cudamalloc:True` |

## 13. 架构总结

```
NativeCachingAllocator（全局单例）
├── DeviceCachingAllocator[0]（GPU 0）
│   ├── small_blocks (BlockPool, ≤1MB)
│   ├── large_blocks (BlockPool, >1MB)
│   ├── active_blocks（活跃分配追踪）
│   ├── expandable_segments_[]（可扩展段列表）
│   ├── graph_pools（CUDA Graph 私有池）
│   │   └── PrivatePool
│   │       ├── small_blocks
│   │       └── large_blocks
│   └── cuda_events（event 池，用于跨 stream 同步）
├── DeviceCachingAllocator[1]（GPU 1）
│   └── ...
└── allocated_blocks（67 路分片哈希表）
```
