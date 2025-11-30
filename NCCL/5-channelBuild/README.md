# 1 channel 原理及基本数据结构

## 1.1 channel
通过ncclTopoComputePaths 我们可以计算出GPU和NIC节点到其他任意节点的`最优路径了`，本节看下**nccl中channel的搜索过程**。

**nccl中channel:** 的概念表示一个通信路径，为了更好的利用带宽和网卡，以及同一块数据可以通过多个channel`并发通信`，另外后续可以看到`一个channel对应了一个GPU SM`，所以基于这些原因，nccl会使用多channel，搜索的过程就是搜索出来一组channel。

**单 Channel 的局限性:**

想象一下只有一条车道的高速公路，所有车辆（数据包）都必须按顺序在这条车道上行驶。即使这条车道速度很快（高带宽），它的**最终总吞吐量**也是有上限的。在 NCCL 的上下文中：

- 一个 Channel 可以理解为一套独立的、并行工作的硬件资源序列，包括执行上下文、命令队列、网络连接等。
- 如果只使用一个 Channel，意味着一个巨大的 All-Reduce 操作只能在一个序列中执行，`无法充分利用 GPU 和网络接口的并行能力`。它可能无法“喂饱”高带宽的硬件，导致通信时间成为训练的瓶颈。
- 即使 Ring 算法很高效，它也只能在一个固定的路径上（例如，GPU0 -> GPU1 -> GPU2 ...）顺序传输数据。
- 虽然 NVLink 是点对点全连接的，但 Ring 算法在一个时刻只使用系统一部分链路。


**多 Channel 的解决方案：**
多 Channel 的本质是 “分而治之” 和 “并行处理”。它将一个大的通信操作（如 All-Reduce）**拆分成多个小块**，并让这些小块在不同的、并行的通路上`同时进行传输和计算`。

- 核心思想： 通过增加“车道”数量来提升`整体“交通吞吐量”`。
- 独立的执行流：每个 Channel 有自己的操作序列，可以独立地在 GPU 流处理器、网络引擎上调度和执行。
- 独立的硬件资源：可能会映射到不同的 NVLink 链路(也可能相同)、不同的网络连接（如 InfiniBand 端口）、不同的 RDMA 引擎等。
- 当启用多 Channel 时，NCCL 会创建多个独立的通信上下文。
- 多channel 的优势：
  - 更高的带宽利用率：并行化使得多个硬件链路同时保持繁忙，更接近硬件的理论峰值带宽。
  - 隐藏延迟：当一个 Channel 的某个操作在等待网络数据时，另一个 Channel 可能正在执行计算或数据传输，实现了流水线并行，提升了整体效率。
  - 更好的扩展性：随着 GPU 数量和模型规模的增大，可以通过增加 Channel 数量来维持高效的通信效率。

**channel 划分流程**
- 在执行集合通信操作（如 All-Reduce）时，NCCL 会将需要通信的大数据缓冲区（Buffer）在逻辑上平均分割成 N 个等份，其中 **N 是 Channel 的数量**；
- channel-process-gpus-kernel 对应关系:
  - 每个进程（每张卡）在这个 Ring 中都有一个固定的前任（Prev）和后继（Next）;
  - 每个 channel 对应一个 CUDA Block，运行在独立的 SM（Streaming Multiprocessor）上，避免线程竞争;
  - H100 的 72 个 SM 可同时支持数十个 channel 并行工作;
  - NCCL 通过动态调整 channel 数量（如环境变量NCCL_NCHANNELS_PER_NET_PEER），确保负载均匀分布在不同链路和 SM 上;
  - 一个进程在 Channel 0 的 Ring 中可能是 GPU0 -> GPU1 -> GPU2 ... 的一部分;
  - 同时，同一个进程在 Channel 1 的 Ring 中可能是 GPU0 -> GPU3 -> GPU6 ... 的一部分;
  - 每个 Ring 都在不同的物理通道上并行传输，这就是多 Channel 能够提升带宽的原因;
  - Ring-All-Reduce、Ring-All-Gather、Ring-Reduce-Scatter 等基于 Ring 的算法;
  - 对于 Tree-Based 算法，一个 Channel 可能对应一个逻辑 Tree;
  - 在极特殊情况下，NCCL 可能会在单个 Channel 内使用混合策略，但这是优化细节，不影响基本架构;
- channel 和 ring 对应关系:
  - 每个 channel 对应一个独立的 ring，一个 channel 内部维护完整的环形拓扑结构；
  - 在 NCCL 的 Ring-Based 集合通信中，**一个 Channel 确实严格对应一个逻辑 Ring**。
    - 1个 Channel = 1个完整的逻辑 Ring
    - 这个 Ring 包含了通信组中所有 rank（进程/GPU），每个 rank 在 Ring 中都有明确的前任（prev）和后继（next）
    - 多个 Channel = 多个并行的 Ring，它们同时处理数据的不同部分
  - 另外，NCCL 会竭尽所能地将不同的逻辑 Ring 映射到`不同的、不重叠`的物理链路上，以实现真正的并行。
  - 逻辑 Ring 映射到物理链路时，NCCL 会尽量**避免**将同一个 Channel 映射到同一个物理链路上。
  - 如果每个节点有多个 InfiniBand 网卡（比如2个），NCCL 会尽可能地将不同的 Channel 映射到不同的网卡上。Channel 0 的数据走网卡0，Channel 1 的数据走网卡1，从而聚合两个网卡的带宽。
- channel 和 pcie
  - 若 NVLink channel 已满，NCCL 可通过 PCIe 新增 1-2 个 channel，进一步提升吞吐量
- 动态拓扑感知与算法选择
  - 在单机 8 卡场景中，NCCL 优先使用 Ring 算法，每个 GPU 通过 NVLink channel 形成环形拓扑，实现高效的 AllReduce;
  - 跨节点场景中，Hierarchical 算法将`节点内通信（NVLink）与节点间通信（RoCE）分层处理，减少跨网络流量`;

example:
```sh
# channel 与 链路
NCCL 首先会为 Channel 0 构建一个逻辑 Ring，例如 GPU0 -> GPU1 -> GPU2 -> GPU3 -> GPU4 -> GPU5 -> GPU6 -> GPU7
Channel 1 的 Ring 可能是 GPU0 -> GPU3 -> GPU6 -> GPU1 -> GPU4 -> GPU7 -> GPU2 -> GPU5
当 Channel 0 的 Ring 在传输数据时，数据从 GPU0 到 GPU1，会占用连接 GPU0 和 NVSwitch 的某几条 NVLink 链路，以及 NVSwitch 到 GPU1 的某几条链路。
当 Channel 1 的 Ring 同时传输数据时，数据从 GPU0 到 GPU3，会占用连接 GPU0 和 NVSwitch 的另外几条 NVLink 链路，以及 NVSwitch 到 GPU3 的链路。
H100 有 18 条 NVLink 链路，这是一个非常宽的“接口”。单个 Channel 的 Ring 通信无法占满所有链路。多 Channel 的设计使得来自不同 Ring 的数据流可以通过 GPU 和 NVSwitch 之间的不同物理链路同时进行传输。

单 Channel：构建 1 个 Ring，比如 0->1->2->3->4->5->6->7->0

4 Channels：NCCL 会构建 4 个独立的 Ring：
Ring 0: 0->1->2->3->4->5->6->7->0 （处理数据块 0）
Ring 1: 0->3->6->1->4->7->2->5->0 （处理数据块 1）
Ring 2: 0->2->4->6->1->3->5->7->0 （处理数据块 2）
Ring 3: 0->4->1->5->2->6->3->7->0 （处理数据块 3）

# channel 与网卡
对于跨节点通信（InfiniBand）：
这个原理同样适用。如果每个节点有多个 InfiniBand 网卡（比如2个），NCCL 会尽可能地将不同的 Channel 映射到不同的网卡上。Channel 0 的数据走网卡0，Channel 1 的数据走网卡1，从而聚合两个网卡的带宽。
```

**channel/ring 经验个数**

| 训练规模               | 典型 Ring 数量 | 说明                     |
|------------------------|----------------|--------------------------|
| 单节点（8 GPU）        | 2-4个          | 主要优化 NVLink 利用率   |
| 中等规模（8-32节点）   | 4-8个          | 平衡节点内和节点间带宽   |
| 大规模（32-256节点）   | 8-16个         | 最大化跨节点带宽利用率   |
| 超大规模（256+节点）   | 16-32个        | 极端情况可能更多         |

## 1.2 nccl 中 channel 相关的数据结构

```c++
struct ncclRing {
  // Shortcuts for userRanks[1] and userRanks[n-1]
  int prev;
  int next;

  // Maps an internal nccl index to user-specified rank order. This is necessary
  // since we need to know how the user expects data to be ordered across
  // devices. Ordered from current device.
  int* userRanks;

  int index; // This rank's index in the ring
};

// The root of each tree only has one node down (+1 intra-node).
#define NCCL_MAX_TREE_ARITY_TOP 2
// Nodes inside the binary tree can have to two nodes down (+1 intra-node).
#define NCCL_MAX_TREE_ARITY 3
struct ncclTree {
  int depth;
  int up;
  int down[NCCL_MAX_TREE_ARITY];
};

#define NCCL_MAX_DIRECT_ARITY 7
struct ncclDirect {
  int depth;
  int out;
  int nHeads;   // Number of parallel N<->1<->net operations we'll do in parallel; size of up/down
  int headRank; // Index in 0..nHeads-1 I am the head rank of. -1 if I'm not a head rank (no local NIC)
  int shift;    // Shuffling of send/recv for scatter/gather operations, basically localRank%nHeads
  // The heads[...] are guaranteed to be in rotated order start with self:
  //   headRank, (headRank+1)%nHeads, (headRank+2)%nHeads, ...
  int heads[NCCL_MAX_DIRECT_ARITY+1];
  int up[NCCL_MAX_DIRECT_ARITY];
  int down[NCCL_MAX_DIRECT_ARITY];
};

// # 假设有6个GPU的NVLS拓扑
//         GPU0 (headRank=0, treeUp=-1)
//         /    \
//     GPU1      GPU2 (treeDown[0,1])
//     /  \        \
//  GPU3  GPU4    GPU5

// # 对应字段值示例：
// GPU0: treeDown = [1, 2, -1], treeUp = -1
// GPU1: treeDown = [3, 4, -1], treeUp = 0
// GPU2: treeDown = [5, -1, -1], treeUp = 0
// GPU3: treeUp = 1, treeDown = [-1, -1, -1]
// 限制 up 数组的最大长度（32），表示单个节点在 NVLS 模式下最多可并行向上（汇聚）通信的目标节点数量。
#define NCCL_MAX_NVLS_ARITY 32
// 限制 treeDown 数组的最大长度（3），表示 NVLS 树形结构中单个节点最多可拥有的子节点数量（即树的 “度” 为 3）。
#define NCCL_MAX_NVLS_TREE_ARITY 3
struct ncclNvls {
  int out;      // 输出方向的目标节点（rank）
  // nHeads=1（单组并行操作）；
  int nHeads;   // Number of parallel N<->1<->net operations we'll do in parallel; size of up/down
  int headRank; // Index in 0..nHeads-1 I am the head rank of. -1 if I'm not a head rank (no local NIC)
  int up[NCCL_MAX_NVLS_ARITY]; // 向上（汇聚）通信的目标节点列表
  int down;                    // 向下（分发）通信的目标节点
  int treeUp;                  // 树形结构中的父节点（上一级节点）
  int treeDown[NCCL_MAX_NVLS_TREE_ARITY]; // 树形结构中的子节点列表（下一级节点）
};

struct ncclChannel {
  struct ncclChannelPeer** peers;
  struct ncclDevChannelPeer** devPeers;
  /* devPeer pointer array used for host side access */
  struct ncclDevChannelPeer** devPeersHostPtr;
  struct ncclRing ring;
  int* devRingUserRanks;
  struct ncclTree tree;

  struct ncclTree collnetChain;
  struct ncclDirect collnetDirect;

  struct ncclNvls nvls;

  int id; // index of this channel
  uint32_t workFifoProduced; // +1 successor of last used work fifo byte

  /* comm split sharable resources */
  struct ncclChannelPeer* collnetPeers;
  struct ncclDevChannelPeer* collnetDevPeers;
  struct ncclChannelPeer* nvlsPeers;
  struct ncclDevChannelPeer* nvlsDevPeers;
};
```

# 2 channel 搜索

**一个channel 有点类似于一个线程的概念**

nccl中channel的概念表示一个通信路径，为了更好的利用带宽和网卡，以及同一块数据可以通过多个channel并发通信，另外后续可以看到一个channel对应了一个GPU SM，所以基于这些原因，nccl会使用多channel，搜索的过程就是搜索出来一组channel。

## 2.1 单机8卡简单情况 : ncclTopoSearchInit

单机的情况下会在ncclTopoTrimSystem函数里删除网卡，因此我们先看下单机八卡这种简化的情况，最后再看下多机引入网卡之后的情况。

ncclTopoSearchInit就是初始化**system->maxWidth**，如果是单机单卡的情况，那么maxWidth设置为**LOC_WIDTH**，否则就遍历每个GPU节点，查看到其他所有GPU节点或者网卡最大带宽(maxBw) 和 总最大带宽(totalBw).

```c++

// 计算 “单通道最大带宽” 的核心工具函数: 每个 Channel 对应一条物理链路
// Initialize system->maxBw. This is the per-channel (i.e. per-SM) max bw.
// 针对指定的 GPU 节点，查询它到目标类型节点（如其他 GPU 或网络设备）的所有链路中，
// 单条链路能提供的最大带宽，为后续通信通道（Channel）的带宽配置提供基准。
static float getMaxBw(struct ncclTopoSystem* system, struct ncclTopoNode* gpu, int type) {
  float maxBw = 0.0;
  for (int i=0; i<system->nodes[type].count; i++) {
    struct ncclTopoLinkList* path = gpu->paths[type]+i;
    // path->bw：该链路路径的带宽（单条链路的理论 / 实测带宽，如 NVLink 单链路 50GB/s，PCIe Gen5 单链路 32GB/s）。
    float bw = path->bw;
    if (path->count == 0) continue; // 该路径包含的物理链路数量（若为 0，说明该路径无效，跳过）。
    // 从所有有效链路（path->count > 0）中，筛选出带宽最大的值，
    // 作为当前 GPU 节点到type类型节点的 “单通道最大带宽”。
    // 将该值作为对应 Channel 的带宽上限，确保数据分片时不超过硬件能力
    maxBw = std::max(maxBw, bw);
  }
  return maxBw;
}

static float getTotalBw(struct ncclTopoSystem* system, struct ncclTopoNode* gpu) {
  float nvlinkBw = 0.0, pciBw = 0.0;
  for (int l=0; l<gpu->nlinks; l++) {
    struct ncclTopoLink* link = gpu->links+l;
    // NVLink：多链路独立并行，总带宽 = 各链路带宽之和
    if (link->type == LINK_NVL) nvlinkBw += link->bw; // 一个GPU 可以连接多个NVLink 链路
    if (link->type == LINK_PCI) pciBw = link->bw;     // 一个GPU 可以连接一个PCIe 链路 x 16 lanes
  }
  return std::max(pciBw, nvlinkBw);
}

// ncclTopoSearchInit就是初始化system->maxWidth，如果是单机单卡的情况，那么maxWidth设置为LOC_WIDTH，
// 否则就遍历每个GPU节点，查看到其他所有GPU节点或者网卡最大带宽 和 总最大带宽。
ncclResult_t ncclTopoSearchInit(struct ncclTopoSystem* system) {
  // 初始化最大带宽和总带宽为0.0（初始值）
  system->maxBw = 0.0;
  system->totalBw = 0.0;

  // 获取"网络节点"（NET类型节点，如InfiniBand/RoCE网卡）的数量
  int inter = system->nodes[NET].count;

  // 特殊场景处理：无网络节点（inter==0）且仅1个GPU节点（单GPU场景）
  if (inter == 0 && system->nodes[GPU].count == 1) {
    // 单GPU本地通信，最大带宽和总带宽均设为本地带宽（LOC_BW，如GPU内部显存访问带宽）
    system->maxBw = LOC_BW;
    system->totalBw = LOC_BW;
    return ncclSuccess;
  }

  // 遍历所有GPU节点，计算并更新系统的最大带宽和总带宽
  for (int g=0; g<system->nodes[GPU].count; g++) {
    // 获取当前遍历的GPU节点（ncclTopoNode结构体存储节点属性，如连接关系、带宽等）
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes + g;

    // 计算当前GPU节点的最大带宽，并更新系统全局最大带宽
    // 若存在网络节点（inter!=0），则以网络链路为基准；否则以GPU间链路（如NVLink/PCIe）为基准
    system->maxBw = std::max(system->maxBw, getMaxBw(system, gpu, inter ? NET : GPU));

    // 计算当前GPU节点的总可用带宽（聚合所有链路的带宽），并更新系统全局总带宽
    system->totalBw = std::max(system->totalBw, getTotalBw(system, gpu));
  }

  return ncclSuccess;
}
```

## 2.2 ncclTopoComputeCommCPU

**功能:** 为通信实例（ncclComm）初始化 CPU 相关信息（架构和厂商），基于拓扑中的 CPU 节点属性，用于后续通信优化（如 CPU 参与的数据中转适配）。

```c++
ncclResult_t ncclTopoComputeCommCPU(struct ncclComm* comm) {
  // 前提假设：系统中至少存在1个CPU节点，且所有CPU的架构和厂商相同（简化适配逻辑）
  const struct ncclTopoNodeSet* cpus = &comm->topo->nodes[CPU];  // 获取拓扑中所有CPU节点的集合

  // 将通信实例的CPU架构设置为第一个CPU节点的架构（如x86_64、ARM）
  comm->cpuArch = cpus->nodes[0].cpu.arch;
  // 将通信实例的CPU厂商设置为第一个CPU节点的厂商（如Intel、AMD）
  comm->cpuVendor = cpus->nodes[0].cpu.vendor;

  return ncclSuccess;
}
```

## 2.3 channel 搜索: ncclTopoGraph 的建立

### 2.3.1 channel 搜索整体逻辑: ncclTopoCompute

主流程: pass 变量用于区分不同的搜索阶段：
- pass=1: 尝试各种参数组合寻找可行解；
- pass=2: 在找到可行解的基础上，尝试优化带宽等参数；

目标是搜索出来尽可能多，带宽尽可能大的一系列channel，本质就是暴力搜索，先设置一系列的条件搜答案，如果搜不出来则降低条件继续搜。

> 节点内 example :
> 此时没有NET节点，所以crossNic为0，然后初始化graph，首先设置最高的条件，
> 限制节点内部只能使用不超过PATH_NVL路径，节点间只能使用不超过PATH_PIX的路径，
> 然后通过system-maxWidth设置speedIntra和speedInter，接着执行ncclTopoSearchRec搜索出一个答案存储到tmpGraph中；
> 如果: 此时就是最优的结果，channel数等于maxChannel，并且speedInter也等于maxWidth，则直接退出；
> 否则: 逐步降低条件，比如将sameChannel设置为0，允许channel之间不一样;
> 调大typeIntra和typeInter；允许crossNic；调小speedInter和speedIntra；
> 然后：开始搜索channel，对于ringGraph来说其实就是搜索出来一系列的环: ；
> 每个rank对应这个环的一个节点，记录了环的prev和next;
> 这里是一个回溯的过程，执行一次 ncclTopoSearchRec 就会得到一个环;
> 执行一次ncclTopoSearchTryGpu看选择出来的下一个点能不能到达;
> 执行一次ncclTopoSearchRecGpu用来找下一个GPU;

主流程: pass 变量用于区分不同的搜索阶段：
- pass=1: 尝试各种参数组合寻找可行解
- pass=2: 在找到可行解的基础上，尝试优化带宽等参数

**什么时候会启动 goto sarch:** <br>
- 当尝试不同通道配置时
- 当调整通信模式时（如从平衡树改为普通树）
- 当调整路径类型时
- 当降低带宽要求时
- 在第二轮优化时提高带宽

```c++
ncclResult_t ncclTopoCompute(ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  int ngpus = system->nodes[GPU].count;
  int crossNic = (system->nodes[NET].count > 1) &&
	 (graph->pattern == NCCL_TOPO_PATTERN_RING ||
          graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE ||
          graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) ? ncclParamCrossNic() : 0;
  graph->crossNic = crossNic == 1 ? 1 : 0;
  graph->bwIntra = graph->bwInter = 0;
  graph->latencyInter = 0;
  int minTypeIntra = PATH_LOC, minTypeInter = PATH_PIX;
  int maxTypeIntra = PATH_SYS, maxTypeInter = PATH_SYS;
  if (ngpus > 1) {
    NCCLCHECK(ncclTopoGetGpuMinPath(system, GPU, &minTypeIntra));
    NCCLCHECK(ncclTopoGetGpuMaxPath(system, GPU, &maxTypeIntra));
  }
  if (system->nodes[NET].count > 0) {
    NCCLCHECK(ncclTopoGetGpuMinPath(system, NET, &minTypeInter));
    NCCLCHECK(ncclTopoGetGpuMaxPath(system, NET, &maxTypeInter));
    maxTypeIntra = maxTypeInter;
  }

  graph->typeIntra = minTypeIntra;
  graph->typeInter = minTypeInter;
  graph->nChannels = 0;
  int trySameChannels = graph->pattern == NCCL_TOPO_PATTERN_NVLS ? 0 : 1;
  graph->sameChannels = trySameChannels;

  int cpuArch, cpuVendor, cpuModel;
  NCCLCHECK(ncclTopoCpuType(system, &cpuArch, &cpuVendor, &cpuModel));

  const char* str = ncclGetEnv("NCCL_GRAPH_FILE");
  if (str) {
    INFO(NCCL_ENV, "NCCL_GRAPH_FILE set by environment to %s", str);
    struct ncclXml* xml;
    NCCLCHECK(xmlAlloc(&xml, NCCL_GRAPH_XML_MAX_NODES));
    NCCLCHECK(ncclTopoGetXmlGraphFromFile(str, xml));
    int nChannels;
    NCCLCHECK(ncclTopoGetGraphFromXml(xml->nodes, system, graph, &nChannels));
    INFO(NCCL_GRAPH, "Search %d : %d channels loaded from XML graph", graph->id, nChannels);
    free(xml);
    if (graph->nChannels > 0) return ncclSuccess;
  }

  int ccMin;
  NCCLCHECK(ncclTopoGetCompCap(system, &ccMin, NULL));
  if (graph->pattern == NCCL_TOPO_PATTERN_NVLS && (system->nodes[NVS].count == 0 || ccMin < 90)) return ncclSuccess;
  // NVLS and COLLNET_DIRECT search must have ngpus heads at most.
  if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) graph->maxChannels = std::min(NCCL_MAX_NVLS_ARITY, system->nodes[GPU].count);
  if (graph->pattern == NCCL_TOPO_PATTERN_COLLNET_DIRECT) graph->maxChannels = std::min(NCCL_MAX_DIRECT_ARITY+1, system->nodes[GPU].count);

  if (ngpus == 1) if (graph->pattern != NCCL_TOPO_PATTERN_RING) graph->pattern = NCCL_TOPO_PATTERN_TREE;

  if (system->nodes[NET].count == 0 && graph->pattern == NCCL_TOPO_PATTERN_NVLS) {
    // Force intra-node NVLS algorithm to pull evenly from all GPUs.
    graph->minChannels = graph->maxChannels;
  }

  int splitNvLink;
  NCCLCHECK(ncclTopoSplitNvLink(system, &splitNvLink));
  if (graph->pattern == NCCL_TOPO_PATTERN_RING && splitNvLink) {
    // We have two sockets with NVLink and a slower link in between (typically QPI).
    // Tree is likely going to work better but it needs at least 2 channels.
    // Since Tree needs to have the same number of channels as Ring, also force Ring to use 2 channels.
    if (graph->maxChannels >= 2 && graph->minChannels == 1) graph->minChannels = 2;
  }

  struct ncclTopoGraph tmpGraph;
  memcpy(&tmpGraph, graph, sizeof(struct ncclTopoGraph));

  // First try crossnic, then decrease bw and finally increase bwIntra.
  int nspeeds = 0;
  float* speedArray = NULL;
  if (system->nodes[NET].count == 0) {
    nspeeds = ccMin >= 100 ? NSPEEDSINTRA_SM100 : (ccMin >= 90 ? NSPEEDSINTRA_SM90 : NSPEEDSINTRA);
    speedArray = ccMin >= 100 ? sm100SpeedArrayIntra : (ccMin >= 90 ? sm90SpeedArrayIntra : speedArrayIntra);
  } else {
    nspeeds = ccMin >= 100 ? NSPEEDSINTER_SM100 : (ccMin >= 90 ? NSPEEDSINTER_SM90 : NSPEEDSINTER);
    speedArray = ccMin >= 100 ? sm100SpeedArrayInter : (ccMin >= 90 ? sm90SpeedArrayInter : speedArrayInter);
  }
  int pass = 1;
  int speedIndex = 0;
  float maxBw = system->maxBw;
  float totalBw = system->totalBw;
  if (ngpus > 1 && graph->pattern != NCCL_TOPO_PATTERN_RING) totalBw *= ngpus*1.0/(ngpus-1);
  while ((speedArray[speedIndex] > maxBw || speedArray[speedIndex]*graph->minChannels > totalBw) && speedIndex < nspeeds-1) speedIndex++;
  tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[speedIndex];
  int64_t globalTimeout = NCCL_SEARCH_GLOBAL_TIMEOUT;

search:
  int time = tmpGraph.sameChannels ? NCCL_SEARCH_TIMEOUT_SAMECHANNELS :
    tmpGraph.pattern == NCCL_TOPO_PATTERN_TREE ? NCCL_SEARCH_TIMEOUT_TREE : NCCL_SEARCH_TIMEOUT;
  tmpGraph.nChannels = 0;
  globalTimeout -= time;

  NCCLCHECK(ncclTopoSearchRec(system, &tmpGraph, graph, &time));
#if 0
  // ... just print the graph
#endif
  // Optimal solution, stop here
  if (time == -1) goto done;
  if (graph->nChannels*graph->bwInter >= system->totalBw) goto done;

  if (pass == 1) {
    // First pass, we don't have a solution yet ; try other options

    // Try having different channels (except when going through AMD CPUs)
    if (tmpGraph.sameChannels == 1 &&
        !(cpuArch == NCCL_TOPO_CPU_ARCH_X86 && cpuVendor == NCCL_TOPO_CPU_VENDOR_AMD && tmpGraph.typeIntra == PATH_SYS)) {
      tmpGraph.sameChannels = 0;
      goto search;
    }
    tmpGraph.sameChannels = trySameChannels;

    if (time != -1) globalTimeout += time;
    else globalTimeout = NCCL_SEARCH_GLOBAL_TIMEOUT;
    if (globalTimeout < 0 && graph->nChannels) goto done;

    // Try a simpler tree
    if (ccMin >= 90 && tmpGraph.pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) {
      tmpGraph.pattern = NCCL_TOPO_PATTERN_TREE;
      goto search;
    }
    tmpGraph.pattern = graph->pattern;

    int maxIntra = system->nodes[NET].count > 0 ? tmpGraph.typeInter : maxTypeIntra;
    if (tmpGraph.typeIntra < maxIntra && (graph->nChannels == 0 || tmpGraph.typeIntra < graph->typeIntra)) {
      tmpGraph.typeIntra += 1;
      if (tmpGraph.typeIntra < PATH_DIS) goto search;
    }
    tmpGraph.typeIntra = minTypeIntra;

    if (system->nodes[NET].count > 0 && tmpGraph.typeInter < maxTypeInter && (graph->nChannels == 0 || tmpGraph.typeInter < graph->typeInter || tmpGraph.typeInter < PATH_PXN)) {
      tmpGraph.typeInter += 1;
      if (tmpGraph.typeInter < PATH_DIS) goto search;
    }
    tmpGraph.typeInter = minTypeInter;

    if (crossNic == 2 && tmpGraph.crossNic == 0
        && (graph->pattern == NCCL_TOPO_PATTERN_RING || graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE)) {
      // Try again with crossNic if permitted
      tmpGraph.crossNic = 2;
      goto search;
    }
    tmpGraph.crossNic = crossNic == 1 ? 1 : 0;

    // Decrease bw until we find a solution
    if ((speedIndex < nspeeds-1) && (graph->nChannels == 0 || (speedArray[speedIndex+1]/graph->bwInter > .49))) {
      tmpGraph.bwInter = tmpGraph.bwIntra = speedArray[++speedIndex];
      goto search;
    }
    speedIndex = 0;
    while (speedArray[speedIndex] > maxBw && speedIndex < nspeeds-1) speedIndex++;
    tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[speedIndex];

  }

done:
  // We have a solution. Start from that solution and move to pass 2.
  if (pass == 1) {
    time = -1;
    NCCLCHECK(ncclTopoDupChannels(graph, ccMin, ngpus));
    memcpy(&tmpGraph, graph, sizeof(tmpGraph));
    speedIndex = 0;
    while (speedArray[speedIndex] > graph->bwInter && speedIndex < nspeeds-1) speedIndex++;
    tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[speedIndex];
    tmpGraph.minChannels = graph->nChannels;
    pass = 2;
  }

  if (pass == 2) {
    // See if we can increase bw
    if (time != 0 && speedIndex > 0) {
      if (graph->pattern == NCCL_TOPO_PATTERN_RING) {
        // increase bw for Ring
        tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[--speedIndex];
        goto search;
      } else if (graph->pattern == NCCL_TOPO_PATTERN_NVLS && tmpGraph.bwInter == graph->bwInter && tmpGraph.bwInter < tmpGraph.bwIntra*2) {
        tmpGraph.minChannels = tmpGraph.maxChannels = graph->nChannels;
        tmpGraph.bwInter = speedArray[--speedIndex];
        goto search;
      } else if (tmpGraph.bwIntra == graph->bwIntra && tmpGraph.bwIntra < tmpGraph.bwInter*2) {
        // increase bwIntra for trees (2 nodes or collnet)
        tmpGraph.bwIntra = speedArray[--speedIndex];
        goto search;
      }
    }
    time = -1;
    memcpy(&tmpGraph, graph, sizeof(tmpGraph));
  }

  if (graph->nChannels == 0 && graph->collNet == 0 && graph->pattern != NCCL_TOPO_PATTERN_NVLS) {
    INFO(NCCL_GRAPH, "Could not find a path for pattern %d, falling back to simple order", graph->pattern);
    for (int i=0; i<ngpus; i++) graph->intra[i] = system->nodes[GPU].nodes[i].gpu.rank;
    graph->inter[0] = graph->inter[1] = 0;
    graph->bwIntra = graph->bwInter = 0.1;
    graph->typeIntra = graph->typeInter = PATH_SYS;
    graph->nChannels = 1;
  }
  return ncclSuccess;
}
```

### 2.3.2 ncclTopoGraph 数据结构
- ncclTopoGraph 用于记录channel 搜索的结果，ncclTopoGraph 几种模式：

```c++
#define NCCL_TOPO_PATTERN_BALANCED_TREE 1   // Spread NIC traffic between two GPUs (Tree parent + one child on first GPU, second child on second GPU)
#define NCCL_TOPO_PATTERN_SPLIT_TREE 2      // Spread NIC traffic between two GPUs (Tree parent on first GPU, tree children on the second GPU)
#define NCCL_TOPO_PATTERN_TREE 3            // All NIC traffic going to/from the same GPU
#define NCCL_TOPO_PATTERN_RING 4            // Ring
#define NCCL_TOPO_PATTERN_NVLS 5            // NVLS+SHARP and NVLS+Tree
#define NCCL_TOPO_PATTERN_COLLNET_DIRECT 6  // Collnet Direct
```

**不同channel的连接放置到intra 里**

```c++
// 定义NCCL（NVIDIA集体通信库）中的网络拓扑结构描述结构体
// 用于存储拓扑相关的配置、属性及计算结果，辅助NCCL选择最优通信策略（如环形、树形等）
struct ncclTopoGraph {
  // 输入/输出参数（既可能作为初始配置，也可能作为计算后结果）
  int id; // 0=环形(ring)，1=树形(tree)，2=collnet(集体通信网络)，3=nvls(NVLink交换机)，4=collnetDirect(直接模式collnet)
  int pattern; // 通信模式标识（如对应allreduce、broadcast等集体操作类型或拓扑变体）
  int crossNic; // 是否跨网卡标识（0=不跨网卡，1=跨网卡）
  int collNet; // collnet模式启用标识（0=禁用，1=启用）
  int minChannels; // 最小通信通道数（通道为并行通信逻辑路径，此为通道数下限）
  int maxChannels; // 最大通信通道数（通道数上限，实际通道数在此范围内计算）

  // 输出参数（根据输入配置和硬件拓扑计算得到的结果）
  int nChannels; // 实际使用的通信通道数（由min/maxChannels及拓扑能力共同决定）
  float bwIntra; // 节点内单个channel带宽
  float bwInter; // 节点间单个channel带宽
  float latencyInter; // 外部通信延迟（如跨节点间，单位通常为微秒）
  int typeIntra; // 内部连接类型（标识节点内硬件连接方式，如NVLink、PCIe等）
  int typeInter; // 外部连接类型（标识节点间网络连接方式，如InfiniBand、Ethernet等）
  int sameChannels; // 内外通信是否使用相同通道配置（0=不同，1=相同）
  int nHops; // 跨节点通信跳数（经过的中间设备数量，跳数越少延迟通常越低）
  // 内部连接信息数组：存储各通道在最大节点数（NCCL_TOPO_MAX_NODES）下的节点连接关系（如节点ID、索引等）
  int intra[MAXCHANNELS*NCCL_TOPO_MAX_NODES]; // 节点内每个channel路径
  // 外部连接信息数组：存储各通道的跨节点连接关系（*2通常用于分别存储源节点和目标节点信息）
  int64_t inter[MAXCHANNELS*2]; // 节点间每个channel路径
};
```

### 2.3.3 回溯算法

回溯算法的核心思想是：尝试所有可能的选择，当发现当前选择无法得到正确解时，回退到上一步尝试其他选择。这种"试错"的思想使得回溯算法能够系统地搜索所有可能的解空间。

```python
def backtrack(路径, 选择列表):
    if 满足结束条件:
        结果.append(路径)
        return

    for 选择 in 选择列表:
        if 不满足约束条件:
            continue

        # 做选择
        路径.append(选择)
        更新状态

        # 进入下一层决策树
        backtrack(路径, 选择列表)

        # 撤销选择
        路径.pop()
        恢复状态
```

example:
```python
# 给定一个没有重复数字的序列，返回其所有可能的全排列。
def permute(nums): # nums: [1, 2, 3]
    def backtrack(path, used):
        # 如果当前路径长度等于数组长度，说明找到一个排列
        if len(path) == len(nums):
            result.append(path[:])
            return

        for i in range(len(nums)):
            # 跳过已经使用过的元素
            if used[i]:
                continue

            # 做选择 : 假设运行到第二个元素
            used[i] = True #  used: [True, True, False]
            path.append(nums[i]) # 尝试 path 2: [1 --> 2 --> ?] :

            # 进入下一层决策树
            backtrack(path, used)

            # 撤销选择
            path.pop() # 1 --> 2 --> ？ 撤销 2 尝试 3 : 1 --> 3 --> ?
            used[i] = False # 对应位置重置为未使用

    result = []
    used = [False] * len(nums)
    backtrack([], used) // used: [False, False, False]
    return result

# 测试
print(permute([1, 2, 3]))
# 输出: [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
```


### 2.3.4 具体搜索逻辑: ncclTopoSearchRec

- ncclTopoSearchRec
- ncclTopoSearchRecNet
- ncclTopoSearchTryGpu
- ncclTopoSearchRecGpu
- ncclTopoReplayGetGpu

```c++
// 调用处:
NCCLCHECK(ncclTopoSearchRec(system, &tmpGraph, graph, &time));

// 实现处: ncclTopoSearchTryGpu 里会递归调用 ncclTopoSearchRec
// 通过递归搜索尝试不同的GPU排列来寻找最优的通信通道
// graph : 输出参数，存储找到的ring拓扑, saveGraph: 保存最佳ring配置
ncclResult_t ncclTopoSearchRec(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int* time) {
  int backToNet, backToFirstRank;
  // Net 连接以及什么时候需要闭环
  // 这里的pattern通常就是ring模式，决定了搜索策略。
  NCCLCHECK(ncclTopoSearchParams(system, graph->pattern, &backToNet, &backToFirstRank));
  if (system->nodes[NET].count) {
    // Start from NET
    ncclTopoSearchRecNet(system, graph, saveGraph, backToNet, backToFirstRank, time);
  } else {
    // Intra-node only. 节点内通信
    if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) {
      // NVLS only : 直接尝试指定的GPU
      // time:
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, -1, -1, graph->nChannels));
      return ncclSuccess;
    } else if (graph->nChannels == 0) { // 第一个channel
      // Try PCI order first
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_PCI, time, -1, -1, 0));
    } else {
      // Also try to replay previous channel
      int g;
      // 第一个channel 采用replay 模式
      // 通过ncclTopoReplayGetGpu获取上一个环对应step的gpu, 存入g 中
      // -1 : // 获取上一个环的第一个GPU作为起始点
      NCCLCHECK(ncclTopoReplayGetGpu(system, graph, -1, &g));
      // -1, -1： replay 模式下复用上一通道的配置，直接从已知的有效 GPU 开始，而不需要从特定的起始节点建立连接
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_REPLAY, time, -1, -1, g));
    }

    // 尝试其它的可能
    if (graph->sameChannels == 0 || graph->nChannels == 0) {
      // Finally, try all other possibilities unless we are forced to use the same channels
      for (int g=0; g<system->nodes[GPU].count; g++) {
        NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, -1, -1, g));
      }
    }
  }
  return ncclSuccess;
}
```

### 2.3.5 具体搜索逻辑: ncclTopoSearchTryGpu

- system - 拓扑系统信息，包含所有GPU、网络等节点的连接关系
- graph - 当前正在构建的通信图结构
- saveGraph - 用于保存找到的最佳通信图
- step - 当前在构建环或树中的第几步
- backToNet - 指示在哪一步需要连接回网络接口(-1表示不需要)
- backToFirstRank - 指示在哪一步需要回到环的起始节点(-1表示不需要闭环)
- forcedOrder - 强制的搜索顺序模式：
  - 0: 正常搜索
  - FORCED_ORDER_PCI: 按PCI总线顺序搜索
  - FORCED_ORDER_REPLAY: 重放之前通道的顺序
- time - 搜索时间限制的计数器指针
- type - 起始节点类型(-1表示无特定起始节点)
- index - 起始节点的索引(-1 表示无起始节点索引)
- g - 目标GPU的索引

**总思路： 递归执行ncclTopoSearchRecGpu， 直到 NGPUS 执行完毕**

- 第一个环是[0,1,2,3,4,5,6,7]

```c++
// 这是一个典型的回溯算法实现，用于尝试所有可能的GPU连接组合
// 这是一个典型的回溯算法实现，用于尝试所有可能的GPU连接组合
// step: 当前搜索步骤或深度，表示正在处理第几个GPU节点。
// forcedOrder: 强制排序标志，控制GPU搜索顺序的策略: FORCED_ORDER_PCI(1): 使用PCIe顺序;
// type: 起始节点类型，表示从哪种类型的节点开始连接（如GPU、NET、NVS等），-1表示无起始节点。
ncclResult_t ncclTopoSearchTryGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int step, int backToNet, int backToFirstRank, int forcedOrder, int *time, int type, int index, int g) {
  // 创建一个标志位，用于标记当前通道正在使用哪个GPU
  // 每个通道使用不同的bit位来标记，避免冲突
  // 0:1, 1:2, 2:4
  const uint64_t flag = 1ULL<<(graph->nChannels);
  struct ncclTopoNode* gpu;
  // 尝试从指定节点(type,index)到目标GPU(g)建立连接
  // 第三个参数1表示正向建立连接(占用带宽)
  // type=-1, index=-1 时，表示没有起始节点，不考虑从哪个节点开始连接, 直接返回g
  NCCLCHECK(ncclTopoFollowPath(system, graph, type, index, GPU, g, 1, &gpu));
  if (gpu) { // 如果连接建立成功
    gpu->used ^= flag; // 标记该GPU在对应channel为已使用(通过异或操作切换bit位), 一个通道对应一个位
    // 功能：递归地构建GPU通信路径，是NCCL拓扑搜索算法的核心实现
    NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, step, backToNet, backToFirstRank, forcedOrder, time));
    gpu->used ^= flag; // 回溯：取消标记该GPU为已使用
    // 回溯：撤销之前建立的连接路径，释放占用的带宽
    // 第三个参数-1表示反向操作(释放带宽)
    NCCLCHECK(ncclTopoFollowPath(system, graph, type, index, GPU, g, -1, &gpu));
  }
  return ncclSuccess;
}
```

### 2.3.6 具体搜索逻辑: ncclTopoSearchRecGpu
```c++
ncclResult_t ncclTopoSearchRecGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, struct ncclTopoNode* gpu, int step, int backToNet, int backToFirstRank, int forcedOrder, int *time) {
  if ((*time) <= 0) return ncclSuccess;
  (*time)--;

  int ngpus = system->nodes[GPU].count;
  // channel 查到底了，判断是否可用
  if (step == ngpus) { // step 走到底了
    // Determine whether we found a better solution or not
    int copy = 0;
    graph->nChannels++; // 进入下一个channel
    // 比较当前图与保存的最佳图，决定是否更新
    NCCLCHECK(ncclTopoCompareGraphs(system, graph, saveGraph, &copy));
    if (copy) {
      // 找到更好的解决方案，更新保存的图
      memcpy(saveGraph, graph, sizeof(struct ncclTopoGraph));
      if (graph->nChannels == graph->maxChannels) *time = -1; // 所有channel 均已找到，返回
    }
    // 继续递归搜索下一个环/channel: channel
    if (graph->nChannels < graph->maxChannels) {
      NCCLCHECK(ncclTopoSearchRec(system, graph, saveGraph, time));
    }
    // 这里回溯，因为要找到所有可能解，从中选择最优
    graph->nChannels--;
    return ncclSuccess;
  }
  // ============== 没走到底时 =================
  // 第n个channel 的第 step 步 = 选中的gpu rank
  // 记录当前GPU到intra数组：先将对应卡节点写入到graph->intra的对应位置
  graph->intra[graph->nChannels*ngpus+step] = gpu->gpu.rank;
  // 指针算术规则: C/C++标准规定，同一数组中两个指针相减等于它们之间的元素个数
  // 类型安全: 编译器会自动根据指针类型(ncclTopoNode*)来计算正确的字节偏移
  int g = gpu - system->nodes[GPU].nodes; // 当前GPU索引
  int nets[NCCL_TOPO_MAX_NODES]; //
  // 情况1：需要返回到网卡(树状拓扑的中间节点)
  if (step == backToNet) {
    // first get back to NIC
    if (system->nodes[NET].count) {
      int startNetIndex;
      NCCLCHECK(getNetIndex(system, graph->inter[graph->nChannels*2], &startNetIndex));
      struct ncclTopoNode* startNet = system->nodes[NET].nodes+startNetIndex;
      int netCount;
      NCCLCHECK(ncclTopoSelectNets(system, graph->typeInter, g, nets, &netCount));
      for (int i=0; i<netCount; i++) {
        int n = nets[i];
        struct ncclTopoNode* net = system->nodes[NET].nodes+n;
        if (graph->pattern == NCCL_TOPO_PATTERN_TREE && net->id != startNet->id) continue; // Trees are symmetric
        if (graph->pattern == NCCL_TOPO_PATTERN_RING && graph->crossNic == 2) {
          if (graph->nChannels & 1 && net->id != graph->inter[(graph->nChannels-1)*2]) continue;
        } else {
          if (graph->crossNic == 0 && (net->net.asic != startNet->net.asic || net->net.port != startNet->net.port)) continue;
        }

        // Balanced Tree : count half of the bandwidth on first two GPUs
        int nextBackToNet = -1;
        float bwInterSave = graph->bwInter;
        if (graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) {
          // Count half of the bandwidth on each of the first two GPUs
          if (step == 0) nextBackToNet = 1;
          else if (net->id != graph->inter[graph->nChannels*2+1]) continue;
          graph->bwInter /= 2;
        }

        NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, NET, n, 1, &net));
        graph->bwInter = bwInterSave;
        if (net) {
          graph->inter[graph->nChannels*2+1] = net->id;
          NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, step, nextBackToNet, backToFirstRank, forcedOrder, time));

          if (graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) graph->bwInter /= 2;
          NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, NET, n, -1, &net));
          graph->bwInter = bwInterSave;
        }
      }
    }
  } else if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) { // 情况2：NVLS模式特殊处理
    NCCLCHECK(ncclTopoSearchTryNvls(system, graph, saveGraph, g, ngpus, time));
  } else if (graph->pattern == NCCL_TOPO_PATTERN_COLLNET_DIRECT) { // 情况3：Collnet直接模式特殊处理
    NCCLCHECK(ncclTopoSearchTryCollnetDirect(system, graph, saveGraph, g, ngpus, time));
  } else if (step < system->nodes[GPU].count-1) { // 情况4：选择下一个GPU
    // Go to next GPU
    int next[NCCL_TOPO_MAX_NODES]; // next数组表示候选的GPU节点
    int count;
    // 根据强制排序策略选择下一个GPU
    if (forcedOrder == FORCED_ORDER_PCI) { // Try the PCI order
      next[0] = step+1; // PCI order 时直接选择下一个节点
      count = 1;        // 跳数为1
    } else if (forcedOrder == FORCED_ORDER_REPLAY) { // Try last channel order
      NCCLCHECK(ncclTopoReplayGetGpu(system, graph, step, next));
      count = 1;
    } else { // Normal search
      NCCLCHECK(ncclTopoSearchNextGpuSort(system, graph, gpu, next, &count, backToNet == -1 ? 0 : backToNet == step+1 ? 1 : -1 ));
    }
    for (int i=0; i<count; i++) { // 递归回溯算法：使用深度优先搜索探索所有可能的GPU连接路径
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, step+1, backToNet, backToFirstRank, forcedOrder, time, GPU, g, next[i]));
    }
  } else if (step == backToFirstRank) {
    // Find first GPU and loop back to it
    int p; // 第一个GPU节点的索引
    NCCLCHECK(getGpuIndex(system, graph->intra[graph->nChannels*ngpus], &p));
    struct ncclTopoNode* firstGpu;
    NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, GPU, p, 1, &firstGpu));
    // 判断到0是否可达
    if (firstGpu) {
      // 如果可达的话继续递归执行ncclTopoSearchRecGpu，此时step == ngpus，即搜索到了一个环
      // 那会将现有的graph去更新最优的saveGraph: 在下面函数中进行
      NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, firstGpu, step+1, backToNet, -1, forcedOrder, time));
      // 回溯的关键步骤，它撤销了之前建立的连接，释放了占用的资源，使得算法可以尝试其他可能的路径组合。
      NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, GPU, p, -1, &firstGpu)); // 回溯
    }
  } else { // 当 step >= ngpus-1 但又不满足其他特定条件时
    // Next path
    // 这个调用会立即进入step == ngpus的基本情况，完成当前通道的评估和比较，然后通过ncclTopoSearchRec继续搜索更多的通道。
    NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, ngpus, -1, -1, forcedOrder, time));
  }
  return ncclSuccess;
}
```

### 2.3.7 获取下一个节点: ncclTopoFollowPath

- ncclTopoFollowPath : 这个函数就是判断能否从type1的index1节点到达type2的index2节点
- 起点的时候type1为-1，因此直接将node设置为type2的index2就返回;
- 如果index1和index2的类型都是GPU那么speed就设置为graph->speedIntra;
- mult是函数的入参，表示需要在path上加还是减去speed，向下搜环的时候需要在path上减去speed，当回溯回去的时候需要将speed加回去;
- 判断path的type是否大于之前设置的type，即graph->typeIntra，大于的话说明不可达，然后通过followPath将path上的边全都减去speed，如果有边剩下的带宽不够speed，那么通过rewind加回去;
- followPath : 将path上的边全都减去speed，如果有边剩下的带宽不够speed，那么通过rewind加回去，此时路径不可达；如果足够的话，则设置node为index2;
-

```c++
// Try to go from node type1/index1 to no type2/index2. mult indicates whether we are counting the bandwidth (1) or undoing (-1).
static ncclResult_t ncclTopoFollowPath(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int type1, int index1, int type2, int index2, float mult, struct ncclTopoNode** node) {
  // First handle easy cases
  *node = system->nodes[type2].nodes+index2; // 给node 赋值
  if (type1 == -1) return ncclSuccess;       // 如果起点是-1，则直接将终点赋给node并返回成功
  struct ncclTopoNode* node1 = system->nodes[type1].nodes+index1;
  struct ncclTopoLinkList* path = node1->paths[type2]+index2;     // 正向路径
  struct ncclTopoNode* node2 = system->nodes[type2].nodes+index2;
  struct ncclTopoLinkList* revPath = node2->paths[type1]+index1;  // reverse path 反向路径

  if (path == NULL) {
    WARN("No path computed to go from %s/%d to %s/%d", topoNodeTypeStr[type1], index1, topoNodeTypeStr[type2], index2);
    return ncclInternalError;
  }

  // Now check link type
  *node = NULL;
  int intra = (type1 == GPU || type1 == NVS) && (type2 == GPU || type2 == NVS);
  float bw = intra ? graph->bwIntra : graph->bwInter;
  int type = intra ? graph->typeIntra : graph->typeInter;

  if (path->type >= PATH_DIS) return ncclSuccess; // Disconnect 直接返回
  if (mult == 1 && (path->type > type)) return ncclSuccess;
  if (mult == 1 && (graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE ||
        graph->pattern == NCCL_TOPO_PATTERN_TREE ||
        graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) &&
      (revPath->type > type)) return ncclSuccess;

  bw *= mult; // 调整带宽的正负，来达到是 + 还是 - 的目的

  // Check there is enough bandwidth on paths.
  int step = 0;
  NCCLCHECK(followPath(path, node1, path->count, bw, &step));
  if (step < path->count) goto rewind;       // 执行到的step小于path->count，说明有连接带宽不够

  // Enough bandwidth : return destination node.
  graph->nHops += mult*path->count;          // 跳数
  *node = system->nodes[type2].nodes+index2; // 最终目的： 结点赋值
  return ncclSuccess;

rewind: // 带宽不足，回退
  // Not enough bandwidth : rewind and exit.
  NCCLCHECK(followPath(path, node1, step, -bw, &step));
  return ncclSuccess;
}


// This is unfortunately needed since manipulating floats often results in rounding errors.
#define SUB_ROUND(a, b) (a = roundf((a-b)*1000)/1000)

static ncclResult_t followPath(struct ncclTopoLinkList* path, struct ncclTopoNode* start, int maxSteps, float bw, int* steps) {
  float pciBw = bw;
  for (int step=0; step<path->count; step++) {
    struct ncclTopoNode* node = path->list[step]->remNode; // 远端结点
    if (node->type == CPU) {
      // Account for P2P inefficiency through Intel CPU RC
      // 计算通过Intel CPU RC（Root Complex）时的PCIe带宽开销
      if (path->type == PATH_PHB && start->type == GPU &&
          node->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 &&
          node->cpu.vendor == NCCL_TOPO_CPU_VENDOR_INTEL) {
        pciBw = INTEL_P2P_OVERHEAD(bw);
      }
    }
  }

  struct ncclTopoNode* node = start;
  for (int step=0; step<maxSteps; step++) {
    struct ncclTopoLink* link = path->list[step];
    struct ncclTopoLink* revLink = NULL; // 反向链路指针
    float fwBw = link->type == LINK_PCI ? pciBw : bw; // bw : 外面传过来的带宽
    float revBw = 0;
    if (link->remNode->type == GPU && link->remNode->gpu.cudaCompCap < 80 && start->type != GPU) {
      if (revLink == NULL) NCCLCHECK(findRevLink(node, link->remNode, link->type, &revLink));
      revBw += fwBw/8;
    }
    if (link->remNode->type == CPU && link->remNode->cpu.arch == NCCL_TOPO_CPU_ARCH_POWER && link->type == LINK_NVL) {
      if (revLink == NULL) NCCLCHECK(findRevLink(node, link->remNode, link->type, &revLink));
      revBw += fwBw;
    }
    // 带宽不足直接返回
    // Coverity thinks that revLink could be NULL below.  However, we access it only if revBw is non-0, and the
    // logic of the code is that revBw can become non-0 only if revLink is non-NULL (see the "if" statement right above).
    // coverity[var_deref_op]
    if (link->bw < fwBw || (revBw && revLink->bw < revBw)) { *steps = step; return ncclSuccess; }
    // 从链路带宽中减去使用的带宽 ： 多个通信通道（channels）可能共享同一条物理链路
    SUB_ROUND(link->bw, fwBw);
    if (revBw) SUB_ROUND(revLink->bw, revBw);
    node = link->remNode;
  }
  *steps = maxSteps;
  return ncclSuccess;
}
```

### 2.3.8 何时更新graph 呢？: ncclTopoCompareGraphs

```c++
// 比较两个拓扑图，决定是否将当前图复制为参考图
ncclResult_t ncclTopoCompareGraphs(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* refGraph, int* copy) {
  // 1. 如果当前图的通道数小于最小要求通道数，直接返回成功（不替换）
  if (graph->nChannels < graph->minChannels) return ncclSuccess;

  // 针对NVLS拓扑模式的特殊处理
  if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) {
    // NVLS通道对应GPU从NVLS拉取数据，所以越多越好

    // 如果当前图通道数多于参考图且不超过GPU数量，则标记为需要复制
    if (graph->nChannels > refGraph->nChannels && graph->nChannels <= system->nodes[GPU].count) *copy = 1;

    // 如果当前图的总带宽（通道数×互联带宽）大于参考图，也标记为需要复制
    if (graph->nChannels*graph->bwInter > refGraph->nChannels*refGraph->bwInter) *copy = 1;

    return ncclSuccess;
  }

  // 2. 尝试获得更好的带宽性能
  // 如果当前图的总内部带宽大于参考图，标记为需要复制
  if (graph->nChannels*graph->bwIntra > refGraph->nChannels*refGraph->bwIntra) {
    *copy = 1;
    return ncclSuccess;
  }

  // 如果当前图的总内部带宽小于参考图，直接返回（不替换）
  if (graph->nChannels*graph->bwIntra < refGraph->nChannels*refGraph->bwIntra) return ncclSuccess;

  // 3. 在带宽相同的情况下，选择跳数更少的路径
  // 如果拓扑模式相同、跨网卡设置相同，且当前图跳数更少，则标记为需要复制
  if (graph->pattern == refGraph->pattern && graph->crossNic == refGraph->crossNic && graph->nHops < refGraph->nHops) *copy = 1;

  return ncclSuccess;
}
```

**如前文所述，如果搜索出来的结果没有达到条件，就开始逐步降低条件继续搜索，接下来的过程比较类似，就不再赘述了.**

## 2.4 多机下的搜索

然后看下多机场景下，比如两机十六卡场景，这个时候有网卡，所以ncclTopoSearchParams设置参数为：<br>

- backToFirstRank = -1; # 不返回 第一个结点
- backToNet = 7; # 级别7

两机16卡下, 网卡和channel 的关系:

- 在环形拓扑（NCCL_TOPO_PATTERN_RING）中，每个通道（channel）通常需要两个网卡连接点；

### 2.4.1 ncclTopoSearchRec 的 ncclTopoSearchRecNet 分支
ncclTopoSearchRec直接执行**ncclTopoSearchRecNet**.

```C++
ncclResult_t ncclTopoSearchRec(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int* time) {
  int backToNet, backToFirstRank;
  NCCLCHECK(ncclTopoSearchParams(system, graph->pattern, &backToNet, &backToFirstRank));
  if (system->nodes[NET].count) {
    // Start from NET
    ncclTopoSearchRecNet(system, graph, saveGraph, backToNet, backToFirstRank, time);
  }
  ...
}
```

ncclTopoSearchRecNet 会搜索出来一个答案，这里会**遍历每个网卡**，尝试用每个网卡作为起点搜索环，首先是网卡0，`将0写入到inter中第一个channel中`，然后将`网卡0的带宽减去speedInter`，maxChannel减去1，然后后边过程和上述很像，会通过ncclTopoSearchTryGpu搜索出一个环。

```c++
ncclResult_t ncclTopoSearchRecNet(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int backToNet, int backToFirstRank, int* time) {
  const int bw = graph->bwInter;
  int nets[NCCL_TOPO_MAX_NODES];
  int netCount;
  int graphFound = 0;
  NCCLCHECK(ncclTopoSelectNets(system, graph->typeInter, -1, nets, &netCount));
  for (int i=0; i<netCount; i++) {
    if ((graph->pattern == NCCL_TOPO_PATTERN_NVLS || graph->pattern == NCCL_TOPO_PATTERN_COLLNET_DIRECT) && graphFound) break;
    int n = nets[(graph->nChannels+i)%netCount];
    struct ncclTopoNode* net = system->nodes[NET].nodes+n;
    if (graph->collNet && net->net.collSupport == 0) continue;
    if (net->net.bw < bw) continue;
    if (graph->pattern == NCCL_TOPO_PATTERN_RING && graph->crossNic == 2
        && (graph->nChannels & 1) && net->id != graph->inter[(graph->nChannels-1)*2+1]) continue;

    graph->inter[graph->nChannels*2] = net->id;
    graph->latencyInter = net->net.latency;

    for (int i=0; i<system->nodes[NET].count; i++) {
      if ((system->nodes[NET].nodes[i].net.asic == net->net.asic) &&
          (system->nodes[NET].nodes[i].net.port == net->net.port)) {
        system->nodes[NET].nodes[i].net.bw -= bw;
      }
    }

    if (graph->pattern == NCCL_TOPO_PATTERN_NVLS || graph->pattern == NCCL_TOPO_PATTERN_COLLNET_DIRECT) {
      // NVLS search only tries to find NIC:GPU combinations to compute the heads.
      if (graph->nChannels < netCount) {
        int gpu = net->net.localGpu;
        if (gpu != -1) {
          int duplicate = 0;
          // check whether there is duplicate head when one GPU connects with multiple NICs
          for (int gc = 0; gc < graph->nChannels; gc++) {
            if (graph->intra[gc * system->nodes[GPU].count] == system->nodes[GPU].nodes[gpu].gpu.rank) {
              duplicate = 1;
              break;
            }
          }
          if (!duplicate) {
            NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, NET, n, gpu));
            graphFound = 1;
          }
        }
      }
    } else {
      if (graph->nChannels > 0 && graph->sameChannels == 1) {
        // Try to replay the last channel
        int g;
        NCCLCHECK(ncclTopoReplayGetGpu(system, graph, -1, &g));
        NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_REPLAY, time, NET, n, g));
      } else {
        if (graph->nChannels == 0 && system->nodes[NVS].count == 0) {
          // Always try the PCI order first to set a reference, but don't count in the timeout nor let it run for long
          int t = 1 << 10;
          NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_PCI, &t, NET, n, 0));
          if (t == -1) *time = -1;
        }

        // Then try the most local GPUs
        int localGpu = net->net.localGpu;
        if (localGpu != -1) {
          NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, NET, n, localGpu));
        }
        int localGpus[NCCL_TOPO_MAX_NODES], localGpuCount, pathType;
        NCCLCHECK(ncclTopoGetLocal(system, NET, n, GPU, localGpus, &localGpuCount, &pathType));
        // if no GPUs are connected, skip this net
        if (pathType == PATH_DIS) continue;
        for (int g = 0; g < localGpuCount; ++g) {
          if (localGpus[g] == localGpu) continue; // We already tried this one
          NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, NET, n, localGpus[g]));
        }
      }
    }

    for (int i=0; i<system->nodes[NET].count; i++) {
      if ((system->nodes[NET].nodes[i].net.asic == net->net.asic) &&
          (system->nodes[NET].nodes[i].net.port == net->net.port)) {
        system->nodes[NET].nodes[i].net.bw += bw;
      }
    }
  }
  return ncclSuccess;
}
```

ncclTopoSearchTryGpu还是会调用ncclTopoSearchRecGpu，当没有遍历完所有GPU节点时，仍然通过递归执行ncclTopoSearchRecGpu来填充graph->intra，最后遍历所有GPU之后step等于7，即**backToNet，这里首先拿出来起始网卡，即网卡0**，如果搜索参数支持crossNic的话就选一个合法的网卡即可，如果不支持的话就判断网卡0是否合法，合法的话将网卡0填充到graph->inter，一个环就搜索完成了。

### 2.4.2 ncclTopoSearchRecGpu 对 if (step == backToNet)  的处理
```c++
ncclResult_t ncclTopoSearchRecGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, struct ncclTopoNode* gpu, int step, int backToNet, int backToFirstRank, int forcedOrder, int *time) {
  if ((*time) <= 0) return ncclSuccess;
  (*time)--;

  int ngpus = system->nodes[GPU].count;
  if (step == ngpus) { // step 走到底了
    // Determine whether we found a better solution or not
    int copy = 0;
    graph->nChannels++; // 进入下一个channel
    // 比较当前图与保存的最佳图，决定是否更新
    NCCLCHECK(ncclTopoCompareGraphs(system, graph, saveGraph, &copy));
    if (copy) {
      // 找到更好的解决方案，更新保存的图
      memcpy(saveGraph, graph, sizeof(struct ncclTopoGraph));
      if (graph->nChannels == graph->maxChannels) *time = -1; // 所有channel 均已找到，返回
    }
    if (graph->nChannels < graph->maxChannels) {
      NCCLCHECK(ncclTopoSearchRec(system, graph, saveGraph, time));
    }
    // 通道回溯，不影响其它通道的正常选择
    graph->nChannels--;
    return ncclSuccess;
  }
  // ============== 没走到底时 =================
  // 第n个channel 的第 step 步 = 选中的gpu rank
  // 记录当前GPU到intra数组
  graph->intra[graph->nChannels*ngpus+step] = gpu->gpu.rank;
  // 指针算术规则: C/C++标准规定，同一数组中两个指针相减等于它们之间的元素个数
  // 类型安全: 编译器会自动根据指针类型(ncclTopoNode*)来计算正确的字节偏移
  int g = gpu - system->nodes[GPU].nodes; // 当前GPU索引
  int nets[NCCL_TOPO_MAX_NODES]; //
  // 情况1：需要返回到网卡(树状拓扑的中间节点)
  // 遍历所有GPU之后step等于7 时
  if (step == backToNet) {
    // first get back to NIC
    if (system->nodes[NET].count) {
      int startNetIndex;
      NCCLCHECK(getNetIndex(system, graph->inter[graph->nChannels*2], &startNetIndex));
      struct ncclTopoNode* startNet = system->nodes[NET].nodes+startNetIndex;
      int netCount;
      NCCLCHECK(ncclTopoSelectNets(system, graph->typeInter, g, nets, &netCount));
      for (int i=0; i<netCount; i++) { // 变量所有net
        int n = nets[i];
        struct ncclTopoNode* net = system->nodes[NET].nodes+n;
        if (graph->pattern == NCCL_TOPO_PATTERN_TREE && net->id != startNet->id) continue; // Trees are symmetric
        if (graph->pattern == NCCL_TOPO_PATTERN_RING && graph->crossNic == 2) {
          if (graph->nChannels & 1 && net->id != graph->inter[(graph->nChannels-1)*2]) continue;
        } else {
          if (graph->crossNic == 0 && (net->net.asic != startNet->net.asic || net->net.port != startNet->net.port)) continue;
        }

        // Balanced Tree : count half of the bandwidth on first two GPUs
        int nextBackToNet = -1;
        float bwInterSave = graph->bwInter;
        if (graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) {
          // Count half of the bandwidth on each of the first two GPUs
          if (step == 0) nextBackToNet = 1;
          else if (net->id != graph->inter[graph->nChannels*2+1]) continue;
          graph->bwInter /= 2;
        }

        NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, NET, n, 1, &net));
        graph->bwInter = bwInterSave;
        if (net) {            // 合法的话将网卡0填充到graph->inter，一个环就搜索完成了
          graph->inter[graph->nChannels*2+1] = net->id;
          // 往下递归
          NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, step, nextBackToNet, backToFirstRank, forcedOrder, time));

          if (graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) graph->bwInter /= 2;
          NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, NET, n, -1, &net)); // 回溯
          graph->bwInter = bwInterSave;
        }
      }
    }
  } else if {
    ...
  }
  return ncclSuccess;
}
```

### 2.4.3 回到ncclTopoSearchRecNet
- 接下来会尝试复制刚刚搜索出来的环，当搜索出一个答案后，回到第一次ncclTopoSearchRecNet;<br>
- 接下来会尝试从离网卡0最近的GPU开始搜索，而不是从GPU0开始，假设为GPUn;<br>
- 这里会先判断GPUn到PCIe switch的双向带宽是否还有空闲，如果有空闲的话才从GPUn开始搜索;<br>
- 但是和这里的注释表述不太相符，注释的意思是说不会将一个GPU既用来发送，又用来接收（说这种情况会影响带宽，这一点比较疑惑）。<br>

```c++
/**
 * 递归搜索网络拓扑 - 在网络层级进行拓扑搜索
 *
 * 这个函数是NCCL拓扑发现算法的核心部分，负责在网络设备层级搜索最优的通信路径。
 * 它通过递归尝试不同的网络设备和GPU组合，寻找满足带宽和延迟要求的通信环。
 *
 * @param system     拓扑系统，包含所有节点（GPU、CPU、网络等）的完整信息
 * @param graph      当前正在构建的拓扑图，包含通信环的配置信息
 * @param saveGraph  用于保存找到的最佳拓扑图的引用
 * @param backToNet  指示是否需要回到网络节点（用于环的闭合）
 * @param backToFirstRank 是否需要回到第一个rank完成环
 * @param time       超时控制指针，防止搜索时间过长
 *
 * @return ncclSuccess 成功，其他值为错误代码
 */
ncclResult_t ncclTopoSearchRecNet(
    struct ncclTopoSystem* system,    // 完整的拓扑系统
    struct ncclTopoGraph* graph,      // 当前搜索的拓扑图
    struct ncclTopoGraph* saveGraph,  // 保存最佳拓扑图
    int backToNet,                    // 回溯到网络节点标志
    int backToFirstRank,              // 回溯到第一个rank标志
    int* time)                        // 超时控制
{
  // 获取当前所需的节点间带宽
  const int bw = graph->bwInter;

  // 存储候选网络设备的数组和计数
  int nets[NCCL_TOPO_MAX_NODES];
  int netCount;
  int graphFound = 0;  // 标记是否找到有效拓扑图

  // 步骤1: 选择符合条件的网络设备
  // 根据通信类型过滤可用的网络设备
  NCCLCHECK(ncclTopoSelectNets(system, graph->typeInter, -1, nets, &netCount));

  // 步骤2: 遍历所有候选网络设备
  for (int i = 0; i < netCount; i++) {
    // 对于NVLS或CollNet Direct模式，如果已经找到图就提前退出
    if ((graph->pattern == NCCL_TOPO_PATTERN_NVLS ||
         graph->pattern == NCCL_TOPO_PATTERN_COLLNET_DIRECT) && graphFound) {
      break;
    }

    // 选择网络设备（使用轮询策略避免总是从同一个设备开始）
    int n = nets[(graph->nChannels + i) % netCount];
    struct ncclTopoNode* net = system->nodes[NET].nodes + n; // 选择 i 对应的net

    // 过滤条件1: 如果启用CollNet但设备不支持，跳过
    if (graph->collNet && net->net.collSupport == 0) continue;

    // 过滤条件2: 网络带宽不足，跳过
    if (net->net.bw < bw) continue;

    // 过滤条件3: 在Ring模式且强制交替网卡策略下，检查网卡交替约束
    if (graph->pattern == NCCL_TOPO_PATTERN_RING && graph->crossNic == 2
        && (graph->nChannels & 1) && net->id != graph->inter[(graph->nChannels - 1) * 2 + 1]) {
      continue;
    }

    // 步骤3: 将当前网络设备加入拓扑图
    graph->inter[graph->nChannels * 2] = net->id;  // 记录网络设备ID
    graph->latencyInter = net->net.latency;        // 记录网络延迟

    // 步骤4: 临时占用网络带宽资源（模拟实际使用）
    // 减少相关网络端口的可用带宽
    for (int i = 0; i < system->nodes[NET].count; i++) {
      if ((system->nodes[NET].nodes[i].net.asic == net->net.asic) &&
          (system->nodes[NET].nodes[i].net.port == net->net.port)) {
        system->nodes[NET].nodes[i].net.bw -= bw;
      }
    }

    // 步骤5: 根据不同的通信模式采用不同的搜索策略
    if (graph->pattern == NCCL_TOPO_PATTERN_NVLS ||
        graph->pattern == NCCL_TOPO_PATTERN_COLLNET_DIRECT) {
      /* NVLS 或 CollNet Direct 模式的特殊处理 */
      // 这些模式只需要找到NIC:GPU组合来计算heads
      if (graph->nChannels < netCount) {
        int gpu = net->net.localGpu;  // 获取与网络设备直连的GPU

        if (gpu != -1) {  // 如果存在直连GPU
          int duplicate = 0;

          // 检查是否存在重复的head GPU
          // 当一个GPU连接到多个NIC时，避免重复使用同一个GPU作为head
          for (int gc = 0; gc < graph->nChannels; gc++) {
            if (graph->intra[gc * system->nodes[GPU].count] ==
                system->nodes[GPU].nodes[gpu].gpu.rank) {
              duplicate = 1;
              break;
            }
          }

          // 如果没有重复，尝试使用这个GPU组合
          if (!duplicate) {
            NCCLCHECK(ncclTopoSearchTryGpu(
                system, graph, saveGraph,
                0, backToNet, backToFirstRank,
                0, time, NET, n, gpu));
            graphFound = 1;  // 标记已找到有效拓扑
          }
        }
      }
    } else {
      /* 其他通信模式（如Ring、Tree等）的处理 */
      if (graph->nChannels > 0 && graph->sameChannels == 1) {
        // 情况A: 重用上一个通道的GPU顺序（通道一致性）
        int g;
        NCCLCHECK(ncclTopoReplayGetGpu(system, graph, -1, &g));
        NCCLCHECK(ncclTopoSearchTryGpu(
            system, graph, saveGraph,
            0, backToNet, backToFirstRank,
            FORCED_ORDER_REPLAY, time, NET, n, g));
      } else {
        // 情况B: 新通道的GPU选择

        // 第一个通道的特殊处理：尝试PCI顺序作为参考
        if (graph->nChannels == 0 && system->nodes[NVS].count == 0) {
          // 使用PCI顺序尝试，但限制时间（不占用太多搜索时间）
          int t = 1 << 10;  // 设置较短的时间限制
          NCCLCHECK(ncclTopoSearchTryGpu(
              system, graph, saveGraph,
              0, backToNet, backToFirstRank,
              FORCED_ORDER_PCI, &t, NET, n, 0));
          if (t == -1) *time = -1;  // 如果超时，传播超时状态
        }

        // 策略1: 首先尝试最本地的GPU（延迟最低）
        int localGpu = net->net.localGpu;
        if (localGpu != -1) {
          NCCLCHECK(ncclTopoSearchTryGpu(
              system, graph, saveGraph,
              0, backToNet, backToFirstRank,
              0, time, NET, n, localGpu));
        }

        // 策略2: 尝试其他本地连接的GPU
        int localGpus[NCCL_TOPO_MAX_NODES], localGpuCount, pathType;
        NCCLCHECK(ncclTopoGetLocal(
            system, NET, n, GPU,
            localGpus, &localGpuCount, &pathType));

        // 如果没有GPU连接，跳过这个网络设备
        if (pathType == PATH_DIS) continue;

        // 遍历所有本地GPU（除了已经尝试过的直连GPU）
        for (int g = 0; g < localGpuCount; ++g) {
          if (localGpus[g] == localGpu) continue;  // 跳过已尝试的直连GPU
          NCCLCHECK(ncclTopoSearchTryGpu(          // 回溯算法，暴力尝试所有可能解
              system, graph, saveGraph,
              0, backToNet, backToFirstRank,
              0, time, NET, n, localGpus[g]));
        }
      }
    }

    // 步骤6: 恢复网络带宽资源（回溯）
    // 搜索完成后，恢复网络端口的可用带宽
    for (int i = 0; i < system->nodes[NET].count; i++) {
      if ((system->nodes[NET].nodes[i].net.asic == net->net.asic) &&
          (system->nodes[NET].nodes[i].net.port == net->net.port)) {
        system->nodes[NET].nodes[i].net.bw += bw;
      }
    }
  }

  return ncclSuccess;
}
```

# 参考

- [参考连接](https://zhuanlan.zhihu.com/p/653440728)