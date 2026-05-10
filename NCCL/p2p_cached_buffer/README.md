# 通信总体流程
```sh
graph TD
    A[用户调用 ncclAllReduce] --> B[enqueue.cc 准备操作]
    B --> C{确定协议}
    C -->|Simple| D[chunkSize = buffSizes/SIMPLE/NCCL_STEPS]
    C -->|LL/LL128| E[使用协议特定粒度]
    D --> F[计算分块数量 nchunks]
    E --> F
    F --> G[创建 workEntry]
    G --> H[提交到 workFifo]
    H --> I[proxy 线程处理]
    I --> J[CUDA 内核执行]
    J --> K[使用 buffSizes 确定传输块]
    K --> L[完成传输]
```

# 1 NCCL 内部 P2P 缓冲区的核心作用

## 1.1 硬件传输适配：数据对齐与格式兼容
GPU 间 P2P 传输（NVLink/PCIe）有严格的硬件要求：比如数据地址需按 **64B/128B** 对齐、数据块大小需满足最小传输粒度。用户传入的 sendbuffer/recvbuffer 中的数据可能不满足这些要求（比如碎片化小数据、地址未对齐），NCCL 会先将数据拷贝到内部缓冲区，做对齐 / 格式转换后再传输，避免硬件传输失败或性能损耗。<br>

## 1.2 提升传输效率：碎片化数据聚合
当通信数据量小且分散时（比如多次小批量传输），频繁的 P2P 调用会产生大量开销。NCCL 会先把这些碎片化数据聚合到内部缓冲区，凑成 “大批次” 后再通过 P2P 链路传输，最大化利用 NVLink/PCIe 的带宽，减少传输次数。<br>

## 1.3 异步通信解耦：避免阻塞用户线程
NCCL 的 P2P 通信是异步执行的，内部缓冲区可以暂存待发送的用户数据，或暂存已接收的待处理数据。这样用户线程无需等待 P2P 传输完成，就能继续操作 sendbuffer/recvbuffer（比如填充新数据），实现 “数据拷贝” 和 “硬件传输” 的解耦，提升整体并行性。

## 1.4. 容错与重试：降低传输失败的影响
P2P 传输可能因硬件瞬态问题失败，此时 NCCL 可直接从内部缓冲区重新发起传输，无需重新从用户的 sendbuffer 读取数据（避免重复拷贝用户数据，减少性能损失）。

## 1.5. 拓扑适配：优化不同链路的传输特性
NVLink（高带宽、低延迟）和 PCIe（带宽较低）的传输特性不同，NCCL 会为每对 P2P 链路分配适配其特性的缓冲区（比如 **NVLink 链路的缓冲区更大，传输粒度更粗**），最大化利用不同链路的性能。

# 2 传输buffer
## 2.1 什么时候 sendbuffer 可以直接传输？
只有同时满足以下所有条件，NCCL 才会直接从你的 sendbuffer 发起 P2P 传输（零拷贝）：<br>

- 地址对齐：sendbuffer 的起始地址必须满足硬件传输要求（如 NVLink 要求 64B/128B 对齐，PCIe 要求 64B 对齐）；
- 数据粒度：传输的数据块大小是硬件最小传输单元的整数倍（如 `NVLink 最小传输单元为 128B`，不能传输 100B 这样的非整数倍数据）；
- 存储位置：sendbuffer 必须位于 GPU 的 HBM（高带宽内存）中，且不是 CPU 内存、页锁定内存（Pinned Memory）或 GPU 显存的非连续区域；
- 通信类型：仅适用于简单的点对点通信（如 ncclSend/ncclRecv），复杂集合通信（如 AllReduce、Broadcast）几乎不会直接传输（因为需要数据聚合 / 拆分）；
- 拓扑匹配：P2P 链路为直连（如 GPU 间有 NVLink 直连），无需中转节点。

```sh
用户sendbuffer（GPU0 HBM，64B对齐，128B数据）
↓（NCCL直接发起DMA传输）
用户recvbuffer（GPU1 HBM）
```

## 2.2 什么时候 sendbuffer 需要拷贝到内部缓冲区？

这是绝大多数实际场景（尤其是新手开发）的情况，只要不满足上述任一条件，就会走间接传输：<br>

- 用户数据不满足对齐 / 粒度要求：比如 sendbuffer 地址未对齐、传输小碎片数据（如 32B）；
- 使用集合通信：AllReduce、AllGather、Broadcast 等操作需要先聚合多个 rank 的数据，或拆分数据到不同链路，必须借助内部缓冲区；
- 非直连拓扑：GPU 间无直连链路（如 GPU0→GPU2 需经 GPU1 中转），中转节点需要内部缓冲区暂存数据；
- 异步通信需求：NCCL 为了让用户线程提前释放 sendbuffer（比如用户要继续往 sendbuffer 写新数据），会先把数据拷贝到内部缓冲区，再异步传输；
- 容错机制开启：NCCL 开启传输重试 / 校验时，需要内部缓冲区缓存数据，避免重复读取用户 sendbuffer。

```sh
用户sendbuffer（GPU0，地址未对齐，50B数据）
↓（NCCL拷贝到内部缓冲区，做64B对齐+补全为128B）
NCCL内部P2P缓冲区（GPU0）
↓（P2P DMA传输）
NCCL内部P2P缓冲区（GPU1）
↓（NCCL裁剪数据为50B，拷贝到目标地址）
用户recvbuffer（GPU1）
```

## 2.3 直接传输sendbuffer 时 内部缓冲区还有用吗?
答案: 是依然有核心作用，只是不再承担 “用户数据中转” 的核心角色，转而发挥控制、容错、调度等辅助性功能，并不会完全闲置。

1. 控制面 / 元数据的传输与缓存;
-直接传输仅针对数据面（从你的 sendbuffer 直接传往 recvbuffer）；
- 控制面数据（比如传输的数据长度、校验和、DMA 指令、错误码、传输完成标志等）依然会暂存到内部缓冲区，再通过 P2P 链路传输。

```sh
举个例子：
GPU0 向 GPU1 直接传输 1MB 数据时，会先把 “传输长度 1MB+CRC 校验码 + 目标地址” 这些元数据写入内部缓冲区，
GPU1 从缓冲区读取这些元数据后，才会启动 DMA 从 GPU0 的 sendbuffer 直接拉取数据。
```

2. 异步通信的状态缓存与解耦

NCCL 的直接传输是异步 DMA 操作，缓冲区会承担 “状态中转站” 的角色：

- 当你调用ncclSend发起直接传输后，NCCL 会立即把传输的 ```“异步任务状态”（如 DMA 是否启动、传输进度、是否完成）写入缓冲区```，你的用户线程无需等待传输完成即可返回；
- 后续你调用ncclGroupStart/End或cudaStreamSynchronize时，NCCL 会从缓冲区读取这些状态，判断传输是否完成，避免直接查询硬件寄存器（效率低）或占用用户内存（易冲突）.

3. 容错与校验的兜底保障

即使是直接传输，NCCL 依然会通过缓冲区实现轻量级容错：

- 传输前：NCCL 会在缓冲区预留少量空间，存储用户数据的 CRC 校验和（从 sendbuffer 计算后写入缓冲区）；
- 传输后：GPU1 接收完数据后，会从缓冲区读取校验和，与接收数据的 CRC 对比，若校验失败，可直接从缓冲区读取 “重传指令”，重新发起直接传输（无需重新从你的 sendbuffer 计算校验和，减少开销）；
- 硬件瞬态错误（如 NVLink 偶发丢包）时，缓冲区会`缓存 “重传上下文”`，快速重试直接传输，避免整个通信任务失败。

4. 多流 / 多请求的调度与限流

如果你的程序同时发起多个直接传输请求（比如多个 CUDA 流的 P2P 通信、多组 rank 对的并行传输），缓冲区会作为 “调度队列”：<br>

- NCCL 会把多个直接传输请求的 “优先级、链路负载” 等信息写入缓冲区，动态排序后依次发起 DMA 传输，避免多条 P2P 链路同时抢占 NVLink/PCIe 总线（导致拥塞）；
- 缓冲区还会限制同时进行的直接传输数量（比如 **NVLink 链路最多同时跑 8 个直接传输**），防止硬件资源耗尽，保障整体稳定性。

5. 拓扑与链路状态的维护
缓冲区会缓存每对 P2P 链路的实时状态（如 NVLink 带宽、延迟、负载）：
- 直接传输过程中，NCCL 会从缓冲区读取链路状态，动态调整 DMA 的传输速率（比如链路负载高时降低速率）；
- 若链路突发故障（如 NVLink 临时断开），NCCL 会从缓冲区读取预存的 “备用拓扑路径”，快速切换传输方式（比如从直接传输转为中转传输），而无需重新扫描拓扑（耗时）。

## 2.4 大量P2P连接时会占用太多缓冲区吗？

答: 不会！NCCL 通过多层核心优化，让缓冲区开销远低于 “连接数 × 单缓冲区大小”，实际显存占用可控且与连接数无线性关系。

NCCL 不会为每个 P2P 连接分配独立的固定缓冲区，而是创建一个共享的缓冲区池：

- 所有主节点与非主节点的 P2P 连接，按需从池中申请缓冲区（仅在传输数据时占用），传输完成后立即归还；
- 缓冲区池的大小仅需 “2~4 倍单 Chunk 大小” 即可支撑所有连接的`交替传输`，而非 “连接数 × 单 Chunk 大小”。

**example** <br>

```sh
假设主节点 GPU0 与 7 个非主节点（GPU1~GPU7）做 Gather，每个节点需传输 8MB 数据：
所有非主节点将 8MB 数据拆分为 8 个 1MB 的 Chunk（硬件最优粒度）；
NCCL 为 GPU0 创建 2MB 的缓冲区池（2 个 1MB Chunk 大小）；
GPU0 从池里申请 1MB 缓冲区，接收 GPU1 的 Chunk1 → 传输完成后归还缓冲区 → 再申请接收 GPU2 的 Chunk1 → 循环至所有节点的 Chunk1 传输完成；
同时，GPU0 将接收的 Chunk 合并写入用户指定的 recvbuffer，释放缓冲区空间；
重复步骤 3~4，直到所有节点的 8 个 Chunk 都传输完成；
✅ 全程缓冲区池仅占用 2MB 显存，远低于 “7 个连接 ×1MB=7MB” 的理论值。
```

# 3 缓冲区buffer大小

## 3.1 初始代码设置
```c++
// NCCL_STEPS (通常为 8): 允许多个块并行传输
#define DEFAULT_LL_BUFFSIZE (NCCL_LL_LINES_PER_THREAD*NCCL_LL_MAX_NTHREADS*NCCL_STEPS*sizeof(union ncclLLFifoLine))
#define DEFAULT_LL128_BUFFSIZE
(NCCL_LL128_ELEMS_PER_THREAD*NCCL_LL128_MAX_NTHREADS*NCCL_STEPS*sizeof(uint64_t))
#define DEFAULT_BUFFSIZE (1 << 22) /* 4MiB */
NCCL_PARAM(BuffSize, "BUFFSIZE", -2);
NCCL_PARAM(LlBuffSize, "LL_BUFFSIZE", -2);
NCCL_PARAM(Ll128BuffSize, "LL128_BUFFSIZE", -2);

NCCL_PARAM(P2pNetChunkSize, "P2P_NET_CHUNKSIZE", (1 << 17)); /* 128 kB */
NCCL_PARAM(P2pPciChunkSize, "P2P_PCI_CHUNKSIZE", (1 << 17)); /* 128 kB */
NCCL_PARAM(P2pNvlChunkSize, "P2P_NVL_CHUNKSIZE", (1 << 19)); /* 512 kB */

static ncclResult_t computeBuffSizes(struct ncclComm* comm) {
  int64_t envs[NCCL_NUM_PROTOCOLS] = { ncclParamLlBuffSize(), ncclParamLl128BuffSize(), ncclParamBuffSize() };
  int defaults[NCCL_NUM_PROTOCOLS] = { DEFAULT_LL_BUFFSIZE, DEFAULT_LL128_BUFFSIZE, DEFAULT_BUFFSIZE };

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    comm->buffSizes[p] = envs[p] != -2 ? envs[p] : defaults[p];
  }
}
```

## 3.2 协议选择
- NCCL_PROTO_LL
- NCCL_PROTO_LL128
- NCCL_PROTO_SIMPLE

| 协议 | 适用场景 | 特点 |
| --- | --- | --- |
| NCCL_PROTO_LL | 小消息传输 | 低延迟，适合小数据量通信 |
| NCCL_PROTO_LL128 | 中等消息传输 | 平衡延迟和带宽，适合中等数据量 |
| NCCL_PROTO_SIMPLE | 大消息传输 | 高吞吐量，适合大数据量通信 |

- LL 协议: 默认 1MB，最大限制 2MB        --> 超小消息，低延迟
- LL128 协议: 默认 4.7MB，最大限制 16MB  --> 中等消息，Hopper 架构优化
- SIMPLE 协议: 默认 4MB，最大限制 128MB  --> 大消息，高吞吐量

P2P ChunkSize: 点对点通信中每次传输的数据块大小，根据连接类型不同而不同;

**chunkSize 与 Simple 协议直接相关， 决定了大消息如何被分割传输，是重要的性能调优参数**

- 网络连接: 128KB
- PCIe 连接: 128KB
- NVLink 连接: 512KB

协议选择和 ChunkSize 设置应根据实际应用场景、消息大小和硬件架构进行优化，NCCL 的默认值在大多数情况下已经提供了良好的性能。

**实验表明，NVLink 在 512KB-1MB 块大小时达到峰值吞吐，超过 4MB 后吞吐量开始明显下降**

|传输类型|	数据面传输路径|	控制面传输路径|
| --- | --- | --- |
|满足直接传输条件|	直接从sendbuffer→recvbuffer|	通过P2P缓冲区|
|不满足直接传输条件|	sendbuffer→缓冲区→recvbuffer|	通过P2P缓冲区|

# 4 缓冲区buffer 动态调整

| 场景 | buffSizes[NCCL_PROTO_SIMPLE] | 实际 chunkSize | p2pChunkSize |
| --- | --- | --- | --- |
| 默认值 | 4MB | 512KB (4MB/8) | 512KB (NVLink) |
| 增大到 8MB | 8MB | 1MB (8MB/8) | 1MB (NVLink) |
| 增大到 16MB | 16MB | 2MB (16MB/8) | 2MB (NVLink) |

## 4.1 p2pChunkSize 重计算:
**p2pChunkSize = min(连接类型默认值, buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS)**
