# 1 普通跨卡访问

1.cudaIpcGetMemHandle 拿到 IPC handle；
1.通过 CPU 通道把 handle 发给对端；
1.对端 cudaIpcOpenMemHandle 映射到自己进程；
1.得到一个与本地地址完全无关的 remote 指针。

- 这种方式的缺点:<br>
```
慢（每次新 buffer 都要握手）
不规整（每个 rank 看到的对端地址都不一样，kernel 内必须查表）
无法被 NCCL 内部用作 NVLS / one-shot allreduce 的零拷贝路径
```

- 解决思路
只要 NCCL / driver 在底层用 CUDA VMM (cuMemAddressReserve + cuMemCreate + cuMemMap) 把 4 块物理显存映**射到同一段虚拟地址区间(的不同部分?)**，并打开 P2P，那么 GPU 上的一个 kernel 就可以用完全相同的指针访问 4 个 rank 的 buffer——只需在指针上加一个 **"rank stride"** 偏移，或者通过 multicast 地址写一次广播到所有 rank。

# 2 跨卡访问数据-改进版本

## 2.1 CUDA Virtual Memory Management (VMM)
- cuMemAddressReserve(size) 在每个进程预留同一段 VA 区间（rank 间通过 allgather 协商起始地址，或用 fixed address）;
- cuMemCreate 申请物理显存（generic allocation handle）;
- cuMemMap 把物理 handle 映射到上面预留的 VA 区间;
- cuMemSetAccess : 给本地 GPU + 所有 peer GPU 授予 RW 权限.


## 2.2 Fabric / Multicast Handle (Hopper+)
H100/Grace-Hopper 引入 NVLink SHARP multicast：

cuMulticastCreate 创建一个 multicast object，**物理上对应交换机里的一个 multicast group；**
每个 rank cuMulticastBindMem 把自己的物理段绑进去；
然后 cuMulticastAddDevice 把所有 GPU 都加进来；
最后 cuMemMap 把这个 multicast handle 映射成一个虚拟地址 mc_ptr;
得到 mc_ptr 后，对它做一次 store，就会被交换机硬件复制到所有 rank 的物理副本——这就是 NVLS one-shot all-reduce 的硬件基础。Reduce 是在交换机里完成的，而不是在某个 GPU 上.

## 2.3 NCCL user-buffer registration
ncclCommRegister(comm, ptr, size, &handle) 把这段对称 buffer 告诉 NCCL，NCCL 内部就会知道："**这块内存在所有 rank 上是对称的**，我可以直接走 NVLS / IB GDR / NVLink P2P 的 zero-copy 算法，而不需要先把数据拷到我自己的 staging buffer。"

关键点：传给 registerMemPool(..., symm=true) 的 MemPool 必须满足"所有 rank 同步、同顺序、**同大小**"分配，否则注册会失败或产生 UB。

# 3 集合通信硬件路径全景

```sh
消息大小 / 硬件拓扑 / buffer注册状态
         │
         ├─── 节点内 NVLink Switch (Hopper+) ──→ NVLS (in-switch reduce)
         ├─── 节点间 IB + SHARP Switch ────────→ IB SHARP (in-network reduce)
         ├─── NVLink P2P (无Switch) ───────────→ Ring/Tree + Copy Engine 或 SM Kernel
         ├─── PCIe 拓扑 ──────────────────────→ Ring + SM Kernel (最慢)
         └─── 混合多节点 ──────────────────────→ 节点内NVLS + 节点间IB SHARP (两级)
```

## 3.1 SM Kernel (Ring AllReduce) — 通用基线

**触发条件**
- 无 NVLink Switch / 无 SHARP / buffer 未注册
- 小消息走 Tree 算法，大消息走 Ring

**原理**
```
Ring: Scatter-Reduce → AllGather，共 2(N-1) 轮
每一步：GPU_i 读 GPU_{i-1} 的数据，做 reduce，写到自己的 buffer
全程由 CUDA Kernel 在 SM 上完成
```

| 项 | 说明 |
|---|---|
| 延迟 | O(2α(N-1)) + O(2β·S) |
| SM 占用 | 高，抢占计算资源 |
| 带宽效率 | 大消息接近理论峰值 |
| 适用范围 | 所有硬件，兜底方案 |

## 3.2 Copy Engine (CE) 路径 — 解放 SM

**触发条件**
- 节点内 NVLink P2P 可用（`cuMemSetAccess` 授权了 peer GPU）
- NCCL 把 AllReduce 拆成 **ReduceScatter (kernel) + AllGather (CE)**
- 或 pipeline 中 memcpy 阶段走 CE，reduce 阶段走 SM

**原理**
```
GPU 有独立 DMA Copy Engine，不占 SM
AllGather 纯搬数据，无计算 → 交给 CE
ReduceScatter 需要加法    → SM Kernel
两者可以 overlap（双 stream）
```

**优势**
- SM 利用率下降 ~30-50%（实测依 workload 而异）
- 适合 gradient AllReduce 与 forward compute overlap 的场景
- NCCL `NCCL_ALGO=RING` + `NCCL_PROTO=LL128` 会选此路径

## 3.3 NVLS (NVLink Switch SHARP, Hopper+) — 交换机内 Reduce

**触发条件**
- H100 NVLink Switch 存在（`cuMulticastCreate` 成功）
- `ncclCommRegister` 注册了对称 buffer（所有 rank 同地址、同大小）
- NCCL 探测到 `NVLS` 算法可用

**完整硬件路径**
```
1. cuMulticastCreate   → 交换机分配 multicast group
2. cuMulticastBindMem  → 每个 rank 把自己的物理段 bind 进去
3. cuMulticastAddDevice→ 所有 GPU 加入 group
4. cuMemMap            → 把 multicast handle 映射成虚拟地址 mc_ptr

AllReduce 执行时：
  每个 rank: store 自己的 tile → mc_ptr
               ↓
  NVLink Switch: 硬件把 N 份数据做 elementwise add（float/bf16）
               ↓
  结果 multicast 回所有 rank 的物理副本
  ──── 全程无 GPU SM 参与 ────
```

| 对比维度 | Ring Kernel | NVLS |
|---|---|---|
| 通信轮数 | 2(N-1) | **1** |
| SM 消耗 | 高 | **近零** |
| 延迟(小消息) | ~几十 μs | **~几 μs** |
| 带宽上限 | NVLink/2 (scatter+gather) | **接近全 NVLink 带宽** |
| Reduce 发生在 | GPU SM | **交换机 ASIC** |

**关键约束**
- buffer 必须"对称"：所有 rank `cuMemCreate` 大小相同、顺序一致
- `ncclCommRegister` 失败 → 自动退回 Ring Kernel（无报错，仅性能下降）
- 仅 fp32 / bf16 / fp16 的 sum reduce 支持硬件加速

## 3.4 IB SHARP — 跨节点在网计算

**触发条件**
- InfiniBand 网络，交换机支持 SHARP v2+
- `NCCL_SHARP_ENABLE=1`，或 NCCL 自动探测
- 消息大小在 SHARP 阈值内（通常 < 1MB，可调）

**原理**
```
传统 AllReduce（树形）：
  Leaf GPU → IB → Aggregator GPU（做reduce）→ IB → 广播
  网络流量 = 2 × message_size × log(N)

IB SHARP：
  Leaf GPU → IB → SHARP Switch（片上做 reduce）→ IB → 广播
  每一跳数据在交换机内部聚合，流量 = message_size（不随 N 增长！）
```

**适用场景**
- 多机 AllReduce，尤其 gradient 同步
- 节点数越多收益越大（N=64 时延迟节省 ~6x）

## 3.5 两级混合路径（生产主流）

```
节点内：NVLS (NVLink Switch)
  ↕  8 GPU → 节点内 AllReduce 完成，得到 partial result

节点间：IB SHARP 或 Ring-over-IB
  ↕  各节点 partial result → 跨节点 AllReduce

节点内：NVLS broadcast 回各 GPU
```

NCCL 称之为 `NVLS_TREE` 算法，自动检测拓扑后启用。

## 3.6 决策流程

```
ncclAllReduce 调用
       │
       ├─ 节点内 NVLink Switch + 对称注册buffer?
       │         YES → NVLS (或 NVLS_TREE 多机)
       │         NO  ↓
       ├─ IB SHARP 可用 + 消息 < threshold?
       │         YES → SHARP Tree
       │         NO  ↓
       ├─ NVLink P2P 可用?
       │         YES → Ring + CE (AllGather) + Kernel (ReduceScatter)
       │         NO  ↓
       └─ PCIe only → Ring + SM Kernel (全走 SM)
```

**实践建议**
- 注册对称 buffer 是激活 NVLS 的关键：`ncclCommRegister` 要在所有 rank 同步调用，且 pooled allocation 必须 `symm=true`
- 验证路径：`NCCL_DEBUG=INFO` 日志里会打印 `Algorithm: NVLS` / `SHARP` / `RING`
- 小消息 (<32KB) 用 NVLS 收益最大；大消息 (>100MB) Ring 带宽效率已接近硬件上限，NVLS 优势缩小
- `torch.distributed` 的 `gradient_as_bucket_view=True` + 合理 bucket size，可以让 CE AllGather 与 backward kernel overlap，隐藏通信延迟

## 3.7 哪个走 Copy Engine？

这俩 API 本身**都不绑定**到某个具体搬运引擎。实际跑时：

| 场景 | 主要引擎 |
|---|---|
| `ncclCommRegister` 普通 user buffer，**节点内 NVLink、小–中等消息** | **SM (ld/st / multimem)** 为主 |
| `ncclCommRegister` 普通 user buffer，**节点内 NVLink、大消息** 或 **跨 PCIe P2P** | **Copy Engine (CE)** 常被用于大块搬运，SM 省下来做计算 |
| `ncclCommRegister` 普通 user buffer，**跨节点 IB/RoCE** | **NIC DMA**（RDMA），不经 CE |
| `ncclCommWindowRegister` symmetric window，all-reduce / all-gather 等 | **SM + NVLS multicast**（`multimem.*` 指令），几乎**纯 SM**，不走 CE |

# 4 symme-mem 对称内存的真实建立流程

假设 4 个 rank（同节点 4×H100），要分配一块大小 S 的对称 buffer。

## 步骤 1：每个 rank 预留同一段 VA

CUdeviceptr va;
cuMemAddressReserve(&va, S, /*align*/0, /*addr*/0, 0);
第一次预留出来的 va 各 rank 可能不同。要让它们一致，常见做法：

方案 A：rank0 先 reserve，把 va 通过 allgather 广播给其他 rank；其他 rank 用 cuMemAddressReserve(&va, S, 0, va_from_rank0, 0) 指定起始地址去 reserve。
方案 B：所有 rank 各自 reserve 一大段"地址池"，然后在池内用相同 offset 切出同址子段。
方案 C：直接用 NCCL/driver 内置的 fixed-VA 协商（NCCL 2.19+ 在 symmetric allocator 内部就是这么做的）。
无论哪种，结果是 4 个 rank 上都拿到了数值相同的 va，但此时这段 VA 还没映射任何物理内存，访问会 segfault。

## 步骤 2：每个 rank 创建自己的物理段

CUmemGenericAllocationHandle h_local;
cuMemCreate(&h_local, S, &prop, 0);   // 在本地 GPU 上分配 S 字节
这是本地操作，每个 rank 各自做，得到的是自己 GPU 上的物理显存。

## 步骤 3：导出 / 交换 handle

int fd_local;
cuMemExportToShareableHandle(&fd_local, h_local, POSIX_FILE_DESCRIPTOR, 0);
// 通过 socket/MPI/NCCL bootstrap 把 fd 发给其他 rank
// 同时也接收其他 rank 发过来的 fd
交换之后，每个 rank 手里有 4 个 handle：1 个本地的 + 3 个远端导入的。


CUmemGenericAllocationHandle h[4];
h[my_rank] = h_local;
for (int r : other_ranks) {
    cuMemImportFromShareableHandle(&h[r], fd_from_r, POSIX_FILE_DESCRIPTOR);
}
注意：h[r] 对当前进程而言只是一个 handle 数字，还不能访问——必须 map 到 VA 上。

## 步骤 4：每个 rank 各自做 4 次 cuMemMap
这是问题的关键。对称内存通常不是把所有 rank 的物理段都映射到"同一个 VA"——那样会冲突——而是把它们映射到一段"VA 数组"里：


// 4 个 rank 的物理段映射到连续的 4 段 VA
for (int r = 0; r < world_size; r++) {
    cuMemMap(va + r * S, S, 0, h[r], 0);
    cuMemSetAccess(va + r * S, S, &accessDesc, 1);  // 允许本 GPU 读写
}
也就是说，每个 rank 在自己的进程里都把 4 段物理显存映射到 [va, va+4S) 这段连续 VA 里：


所有 rank 的虚拟地址布局完全相同：
  [va + 0*S, va + 1*S)  -> rank0 的物理显存
  [va + 1*S, va + 2*S)  -> rank1 的物理显存
  [va + 2*S, va + 3*S)  -> rank2 的物理显存
  [va + 3*S, va + 4*S)  -> rank3 的物理显存
每个 rank 上这 4 段映射都是它自己 cuMemMap 出来的，不是 rank0 映射一次别人就能用。

## 步骤 5：打开 P2P 访问
cuMemSetAccess 时把 peer device 也加进去，这样本 GPU 通过 NVLink 访问那段 VA 时，硬件会路由到 peer GPU 的物理显存上：


CUmemAccessDesc desc[4];
for (int r = 0; r < 4; r++) {
    desc[r].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc[r].location.id   = r;       // peer GPU id
    desc[r].flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
}
cuMemSetAccess(va, 4*S, desc, 4);
现在在任意 rank 上：


float* base = (float*)va;
base[0 * N + i]   // 物理上落到 rank0 的 HBM（如果不是本地，走 NVLink P2P load）
base[1 * N + i]   // 物理上落到 rank1 的 HBM
base[2 * N + i]   // 物理上落到 rank2 的 HBM
base[3 * N + i]   // 物理上落到 rank3 的 HBM
所有 rank 上同一表达式访问的物理位置是同一个——这就是"对称"的语义。
