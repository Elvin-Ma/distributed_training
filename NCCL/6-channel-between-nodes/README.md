# 当前局面介绍

上节中完成了单机内部的channel搜索，仍然以ringGraph为例的话，相当于在单台机器内部搜索出来了一系列的环，接下来需要将机器之间的环连接起来。<br>

example: <br>

假设两机十六卡的情况下第一台机器的一个ring为：<br>

```shell
graph->intra: GPU/0 GPU/7 GPU/6 GPU/3 GPU/2 GPU/5 GPU/4 GPU/1
graph->inter: NET/0 NET/0
```

第二个机器对应的ring为：

```shell
graph->intra: GPU/10 GPU/9 GPU/8 GPU/13 GPU/12 GPU/15 GPU/14 GPU/11
graph->inter: NET/0 NET/0
```

# 2 相关数据结构

## 2.1 ncclGraphInfo

ncclGraphInfo记录了环的信息，包含：

```c++
struct ncclGraphInfo {
    int sameChannels;
    float speedIntra; // 内部带宽
    float speedInter; // Inter 带宽
    int typeIntra;    // 内部类型
  };
```

## 2.2 allGather3Data 聚合channel 信息

```c++
  struct {
    int cudaCompCap;                // 计算能力
    int fullCudaCompCap;            // 完整计算能力
    int nChannels;                  // channel数量
    struct ncclGraphInfo tree;
    struct ncclGraphInfo ring;
    struct ncclGraphInfo collNet;
    struct ncclTopoRanks topoRanks; // 拓扑信息
  } *allGather3Data;


  NCCLCHECK(ncclCalloc(&allGather3Data, nranks)); // 每个rank 都要一份allGather3Data
  allGather3Data[rank].cudaCompCap = ncclCudaCompCap();
  allGather3Data[rank].nChannels = comm->nChannels = treeGraph.nChannels = ringGraph.nChannels =
    std::min(treeGraph.nChannels, ringGraph.nChannels);
  ...
  allGather3Data[rank].ring.sameChannels = ringGraph.sameChannels;
  allGather3Data[rank].ring.speedIntra = ringGraph.speedIntra;
  allGather3Data[rank].ring.speedInter = ringGraph.speedInter;
  allGather3Data[rank].ring.typeIntra = ringGraph.typeIntra;
```

## 2.3 ncclTopoRanks

```c++
​struct ncclTopoRanks {
  int ringRecv[MAXCHANNELS];  // 环的静态拓扑结构
  int ringSend[MAXCHANNELS];  // 环的静态拓扑结构
  int ringPrev[MAXCHANNELS];   // 当前rank在ring中的prev：实际的数据流向
  int ringNext[MAXCHANNELS];   // 当前rank在ring中的next：实际的数据流向
  int treeUpRecv[MAXCHANNELS];
  int treeUpSend[MAXCHANNELS];
  int treeDnRecv[MAXCHANNELS];
  int treeDnSend[MAXCHANNELS];
};
```

然后开始设置ncclTopoRanks，获取当前rank在ring中的prev和next，其中第一个rank的prev和最后一个rank的next为-1，如rank6的prev为7，next为3；
获取当前ring的ringRecv和ringSend(**即ring的第一个节点和最后一个节点**)，最后将搜索到的环复制了一遍，这里在官方issue中看到相关解释是为了进一步的并行以充分利用带宽。<br>

## 2.4 ncclChannel

成员简单介绍（这里可能是之前版本）
- 其中collectives保存了用户向nccl提交的通信操作，比如ncclSend，ncclRecv等都会向collectives里加一项;
- ncclColl则保存了这些操作对应的参数；
- collectives是一个环形队列，所以collStart指向了开始位置，collCount表示队列中操作数量；
- FifoHead和FifoTail用于协调kernel产出数据和NET发送数据，其实就是生产者消费者;
- ncclPeer保存了通信相关的信息，后续再具体介绍.

```c++
// NCCL通信通道结构体
struct ncclChannel {
  // 对等节点信息（主机端）
  struct ncclChannelPeer** peers;           // 指向所有对等节点信息的指针数组（主机端）
  struct ncclDevChannelPeer** devPeers;     // 指向设备端对等节点信息的指针数组（设备端）
  /* devPeer pointer array used for host side access */
  struct ncclDevChannelPeer** devPeersHostPtr;  // 设备端对等节点信息的主机端访问指针

  // 通信拓扑结构
  struct ncclRing ring;                     // 环形通信拓扑
  int* devRingUserRanks;                    // 设备端的用户rank数组
  struct ncclTree tree;                     // 树形通信拓扑

  // CollNet（集合网络）专用通信结构
  struct ncclTree collnetChain;             // CollNet链式拓扑
  struct ncclDirect collnetDirect;          // CollNet直接通信结构

  // NVLink SHARP（NVLS）相关结构
  struct ncclNvls nvls;                     // NVLink SHARP通信结构

  // 通道标识信息
  int id;                                   // 通道ID，这个通道的索引
  uint32_t workFifoProduced;                // 工作FIFO生产指针，最后使用的工作FIFO字节的后继位置

  /* comm split sharable resources */
  // 通信分割共享资源
  struct ncclChannelPeer* collnetPeers;     // CollNet对等节点数组
  struct ncclDevChannelPeer* collnetDevPeers; // CollNet设备端对等节点数组
  struct ncclChannelPeer* nvlsPeers;        // NVLS对等节点数组
  struct ncclDevChannelPeer* nvlsDevPeers;  // NVLS设备端对等节点数组
};
```

# 3 跨节点channel 连接流程

## 3.1 ncclTopoRreset 预填

- 在 /root/projects/nccl/src/init.cc:initTransportsRank 中被调用

```c++
ncclResult_t ncclTopoPreset(struct ncclComm* comm,
    struct ncclTopoGraph* treeGraph, struct ncclTopoGraph* ringGraph, struct ncclTopoGraph* collNetGraph,
    struct ncclTopoRanks* topoRanks) {
  int rank = comm->rank;             // 当前进程在全局中的rank
  int localRanks = comm->localRanks; // 当前节点上的本地rank数量
  int nChannels = comm->nChannels;   // 通信通道数量

  // 遍历所有通信通道
  for (int c=0; c<nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c; // 获取通道
    channel->ring.prev = channel->ring.next = -1;   // 初始化环形通道的前后节点为-1（无效值）
    ...

    // 获取当前通道的拓扑信息数组指针
    // ringIntra: 环形拓扑在当前节点内的rank排列
    // treeIntra: 树形拓扑在当前节点内的rank排列
    // collNetIntra: CollNet拓扑在当前节点内的rank排列
    int* ringIntra = ringGraph->intra+c*localRanks;
    int* treeIntra = treeGraph->intra+c*localRanks;
    int* collNetIntra = collNetGraph->intra+c*localRanks;

    // 遍历本地所有rank，找到当前rank在拓扑中的位置
    for (int i=0; i<localRanks; i++) {
      // 处理环形拓扑, 仅处理当前rank
      if (ringIntra[i] == rank) {
        // 设置环形拓扑的接收和发送目标：
        // ringRecv: 从环的第一个rank接收数据
        // ringSend: 向环的最后一个rank发送数据
        topoRanks->ringRecv[c] = ringIntra[0];            // ring 的第一个节点
        topoRanks->ringSend[c] = ringIntra[localRanks-1]; // ring 的最后一个节点

        // 设置当前通道的环形前后邻居：
        // prev: 如果是第一个节点则没有前驱，否则取前一个rank
        // next: 如果是最后一个节点则没有后继，否则取后一个rank
        channel->ring.prev = (i == 0) ? -1 : ringIntra[i-1];
        channel->ring.next = (i == localRanks-1) ? -1 : ringIntra[i+1];
      }
      ...
    }

    // 将通道中的环形拓扑信息保存到topoRanks结构中
    topoRanks->ringPrev[c] = channel->ring.prev;
    topoRanks->ringNext[c] = channel->ring.next;
  }

  // 通道复制：将前nChannels个通道的数据复制到后nChannels个通道
  // 这样做的目的是为了支持双向通信或其他通信模式
  struct ncclChannel* channel0 = comm->channels;        // 指向第一个通道
  struct ncclChannel* channel1 = channel0+nChannels;    // 指向第二个通道块的首地址
  memcpy(channel1, channel0, nChannels*sizeof(struct ncclChannel)); // nChannels 一起copy

  return ncclSuccess;
}
```

## 3.2 allGather3Data

- 在 /root/projects/nccl/src/init.cc:initTransportsRank 中被调用

```c++
NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allGather3Data, sizeof(*allGather3Data)), ret, fail);
```

计算出当前rank所在的node保存在comm->node，以及每个node的第一个rank保存在nodesFirstRank，因此例子中：

```shell
nodesFirstRank[0]: 0
nodesFirstRank[1]: 10
```

## 3.3 ncclTopoPostset

然后开始将每个机器的环首尾相连组成大环。

```c++
​ncclResult_t ncclTopoPostset(struct ncclComm* comm, int* firstRanks, struct ncclTopoRanks** allTopoRanks, int* rings) {
  // Gather data from all ranks
  int *ringRecv, *ringSend, *ringPrev, *ringNext, *treeUpRecv, *treeUpSend, *treeDnRecv,*treeDnSend;
  int nranks = comm->nRanks;
  int nChannels = comm->nChannels;
  NCCLCHECK(ncclCalloc(&ringRecv, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&ringSend, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&ringPrev, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&ringNext, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&treeUpRecv, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&treeUpSend, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&treeDnRecv, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&treeDnSend, nranks*MAXCHANNELS));
  for (int i=0; i<nranks; i++) {
    for (int c=0; c<nChannels;c++) {
      ringRecv[c*nranks+i] = allTopoRanks[i]->ringRecv[c];
      ringSend[c*nranks+i] = allTopoRanks[i]->ringSend[c];
      ringPrev[c*nranks+i] = allTopoRanks[i]->ringPrev[c];
      ringNext[c*nranks+i] = allTopoRanks[i]->ringNext[c];
      treeUpRecv[c*nranks+i] = allTopoRanks[i]->treeUpRecv[c];
      treeUpSend[c*nranks+i] = allTopoRanks[i]->treeUpSend[c];
      treeDnRecv[c*nranks+i] = allTopoRanks[i]->treeDnRecv[c];
      treeDnSend[c*nranks+i] = allTopoRanks[i]->treeDnSend[c];
    }
  }

  // Connect rings and trees. This should also duplicate the channels.
  NCCLCHECK(connectRings(comm, ringRecv, ringSend, ringPrev, ringNext, firstRanks));
  NCCLCHECK(connectTrees(comm, treeUpRecv, treeUpSend, treeDnRecv, treeDnSend, firstRanks));

  // Duplicate ringPrev/ringNext for ncclBuildRing
  memcpy(ringPrev+nChannels*nranks, ringPrev, nChannels*nranks*sizeof(int));
  memcpy(ringNext+nChannels*nranks, ringNext, nChannels*nranks*sizeof(int));

  // Duplication should be complete now
  nChannels = comm->nChannels = std::min(MAXCHANNELS,nChannels*2);

  // Honor NCCL_MIN_NRINGS/NCCL_MAX_NRINGS.
  // We permit combining max, then min, to only use the first channels, then duplicate them.
  nChannels = comm->nChannels = std::min((int)ncclMaxNchannels(), nChannels);
  int c;
  for (c=nChannels; c<ncclMinNchannels(); c++) {
    memcpy(ringPrev+c*nranks, ringPrev+(c-nChannels)*nranks, nranks*sizeof(int));
    memcpy(ringNext+c*nranks, ringNext+(c-nChannels)*nranks, nranks*sizeof(int));
    memcpy(comm->channels+c, comm->channels+c-nChannels, sizeof(struct ncclChannel));
  }
  nChannels = comm->nChannels = c;

  // Create rings array and check all is fine
  NCCLCHECK(ncclBuildRings(nChannels, rings, comm->rank, comm->nRanks, ringPrev, ringNext));

  free(ringRecv);
  free(ringSend);
  free(ringPrev);
  free(ringNext);
  free(treeUpRecv);
  free(treeUpSend);
  free(treeDnRecv);
  free(treeDnSend);

  return ncclSuccess;
}
```

这里将所有channel的prev，next，send，recv信息打平到数组中，例如recv[0]表示第一个ring中rank0的recv是哪个rank，然后开始计算**当前机器第一个rank的prev和最后一个rank的next**。

## 3.4 connectRings

```c++
/**
 * 连接各个节点形成通信环拓扑结构
 *
 * @param comm NCCL通信器
 * @param ringRecv 接收环数组，记录每个rank的接收关系
 * @param ringSend 发送环数组，记录每个rank的发送关系
 * @param ringPrev 前驱节点数组
 * @param ringNext 后继节点数组
 * @param firstRanks 每个节点的第一个rank数组
 * @return ncclResult_t 执行结果
 */
static ncclResult_t connectRings(struct ncclComm* comm, int* ringRecv, int* ringSend, int* ringPrev, int* ringNext, int* firstRanks) {
  int nChannels = comm->nChannels;  // 通道数量
  int nNodes = comm->nNodes;        // 节点数量

  // 遍历所有通道
  for (int c=0; c<nChannels; c++) {
    // 计算当前通道在数组中的偏移量（二维数组展平为一维）
    int* recv = ringRecv + c * comm->nRanks;
    int* send = ringSend + c * comm->nRanks;
    int* prev = ringPrev + c * comm->nRanks;
    int* next = ringNext + c * comm->nRanks;

    // 获取当前通道的两个通道结构（可能是双向通信）
    struct ncclChannel* channel0 = comm->channels + c;   // 获取正向channel
    struct ncclChannel* channel1 = channel0 + nChannels; // 获取反向channel

    // 遍历所有节点
    for (int n=0; n<nNodes; n++) {
      // 计算当前节点的接收rank和上一个节点的发送rank
      int recvRank = recv[firstRanks[n]];                       // 当前节点n的接收rank
      int prevSendRank = send[firstRanks[(n-1+nNodes)%nNodes]]; // 上一个节点的发送rank（环形计算）

      // 设置前驱关系：当前接收rank的前驱是上一个节点的发送rank
      prev[recvRank] = prevSendRank;

      // 如果当前进程就是接收rank，更新通道的前驱信息
      if (comm->rank == recvRank) {
        channel0->ring.prev = prevSendRank;
        channel1->ring.prev = prevSendRank;
      }

      // 计算当前节点的发送rank和下一个节点的接收rank
      int sendRank = send[firstRanks[n]];                   // 当前节点n的发送rank
      int nextRecvRank = recv[firstRanks[(n+1)%nNodes]];    // 下一个节点的接收rank

      // 设置后继关系：当前发送rank的后继是下一个节点的接收rank
      next[sendRank] = nextRecvRank;

      // 如果当前进程就是发送rank，更新通道的后继信息
      if (comm->rank == sendRank) {
        channel0->ring.next = nextRecvRank;
        channel1->ring.next = nextRecvRank;
      }
    }

    // 记录调试信息：显示环的拓扑结构
    TRACE(NCCL_GRAPH, "Ring %d : %d -> %d -> %d", c, channel0->ring.prev, comm->rank, channel0->ring.next);
    TRACE(NCCL_GRAPH, "Ring %d : %d -> %d -> %d", c+nChannels, channel1->ring.prev, comm->rank, channel1->ring.next);
  }

  return ncclSuccess;
}
```

如上所示，当前机器recv rank的prev就是前一个机器的send rank，当前机器send rank的next就是下一个机器的recv rank。然后执行ncclBuildRings按照大环的顺序依次记录rank到rings。

## 3.5 ncclBuildRings

```c++
/**
 * 构建通信环拓扑结构
 *
 * @param nrings 环的数量
 * @param rings 输出参数，存储构建好的环结构（二维数组展平）
 * @param rank 当前进程的rank
 * @param nranks 总的进程数量
 * @param prev 前驱节点数组（未在函数中使用，可能是接口兼容性保留）
 * @param next 后继节点数组，定义每个rank的下一个节点
 * @return ncclResult_t 执行结果
 */
ncclResult_t ncclBuildRings(int nrings, int* rings, int rank, int nranks, int* prev, int* next) {
  // 遍历所有环
  for (int r=0; r<nrings; r++) {
    char prefix[30];  // 用于调试日志的前缀字符串

    // 从当前rank开始遍历环
    int current = rank;  // 当前遍历的rank，从自身开始

    // 遍历整个环，构建环的完整路径
    for (int i=0; i<nranks; i++) {
      // 将当前rank存入环数组中
      // rings是二维数组展平：rings[环索引][位置] = rank
      rings[r * nranks + i] = current;

      // 移动到环中的下一个rank
      // next也是二维数组：next[环索引][当前rank] = 下一个rank
      current = next[r * nranks + current];
    }

    // ... （这里可能有其他代码，如调试输出等）

    // 验证环的完整性：检查环中是否包含所有rank
    for (int i=0; i<nranks; i++) {
      int found = 0;  // 标记是否找到rank i

      // 在当前环中搜索rank i
      for (int j=0; j<nranks; j++) {
        if (rings[r * nranks + j] == i) {
          found = 1;  // 找到rank i
          break;
        }
      }

      // 如果环中缺少某个rank，报错返回
      if (found == 0) {
        WARN("Error : ring %d does not contain rank %d", r, i);
        return ncclInternalError;
      }
    }
  }

  return ncclSuccess;
}
```

example:

```shell
GPU/6 GPU/3 GPU/2 GPU/5 GPU/4 GPU/1 GPU/10 GPU/9 GPU/8 GPU/13 GPU/12 GPU/15 GPU/14 GPU/11 GPU/0 GPU/7
```

到此就完成了机器之间大环建立，**每个rank都知道自己的上一个和下一个rank是谁**，`那么就可以建立实际的通信链路了`。

# 4 buffer 申请

接下来每个rank都要**为通信分配一些内存**。

## 4.1 亲和性设置

为了提高性能，这里会在分配buffer之前设置cpu亲和性，使得分配的内存尽量是当前numa本地的。

```c++
// 获取指定rank的CPU亲和性设置
ncclResult_t ncclTopoGetCpuAffinity(struct ncclTopoSystem* system, int rank, cpu_set_t* affinity) {
  struct ncclTopoNode* cpu = NULL, *gpu = NULL;
  int gpuIndex, cpuIndex;

  // 1. 根据rank获取GPU索引和对应的CPU索引
  NCCLCHECK(ncclTopoRankToIndex(system, rank, &gpuIndex, /*showWarn=*/true)); // 获取GPU索引
  NCCLCHECK(ncclGetLocalCpu(system, gpuIndex, &cpuIndex)); // 获取CPU索引

  // 获取对应的GPU和CPU节点
  gpu = system->nodes[GPU].nodes+gpuIndex;
  cpu = system->nodes[CPU].nodes+cpuIndex;

  // 2. 查询当前进程的CPU亲和性设置
  cpu_set_t mask;
  SYSCHECK(sched_getaffinity(0, sizeof(cpu_set_t), &mask), "sched_getaffinity");

  // 3. 获取与GPU相近的CPU的亲和性设置
  // 找到当前rank对应的cpu节点之后，可以获取到该cpu对应的core，即cpuMask;
  cpu_set_t cpuMask = cpu->cpu.affinity;

  // 4. 计算最终的亲和性设置
  cpu_set_t finalMask;
  if (ncclParamIgnoreCpuAffinity())
    // 如果设置了忽略CPU亲和性，直接使用GPU相关的CPU亲和性
    finalMask = cpuMask;
  else
    // 否则取当前进程亲和性与GPU相关CPU亲和性的交集
    // 获取当前进程对应的亲和性，即mask，默认会取cpuMask和mask的交集finalMask;
    CPU_AND(&finalMask, &mask, &cpuMask);

  // 5. 将结果复制到输出参数
  memcpy(affinity, &finalMask, sizeof(cpu_set_t));

  // 6. 生成详细的调试信息
  char msg[1024] = "";
  snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), "Affinity for GPU %d is ", gpu->gpu.dev);
  if (CPU_COUNT(&finalMask)) {
    (void)ncclCpusetToRangeStr(&finalMask, msg + strlen(msg), sizeof(msg) - strlen(msg));
  } else {
    snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), "empty, ignoring");
  }
  snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), ". (GPU affinity = ");
  (void)ncclCpusetToRangeStr(&cpuMask, msg + strlen(msg), sizeof(msg) - strlen(msg));
  if (!ncclParamIgnoreCpuAffinity()) {
    snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), " ; CPU affinity = ");
    (void)ncclCpusetToRangeStr(&mask, msg + strlen(msg), sizeof(msg) - strlen(msg));
  }
  snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), ").");
  INFO(NCCL_INIT, "%s: %s", __func__, msg);

  return ncclSuccess;
}
```

1. 找到当前rank对应的cpu节点之后，可以获取到该cpu对应的core，即cpuMask;
2. 然后获取当前进程对应的亲和性，即mask，默认会取cpuMask和mask的交集finalMask;
3. 如果交集不为空的话，会将finalMask设置给当前进程。

# 5 channel 设置

## 5.1 initChannel

**最终目的：** 让指定通道具备完整的通信上下文（`节点信息、拓扑配置、设备资源`），能够独立承担 NCCL 集合通信的并行数据传输任务，支撑多通道并发通信以提升整体通信效率。

- 先检查通道是否已初始化（通过channel->id != -1判断），避免重复操作；若未初始化，则设置通道 ID、初始化工作队列生产计数等基础属性，标记通道进入可用状态。
- 节点通信信息（Peers）的分配与关联， 节点通信信息（Peers）的分配与关联：
  - 优先复用 / 分配共享资源（sharedRes）中的全局节点信息，同时为当前通道分配私有节点数组，通过映射全局父 rank 关联共享资源，并原子递增引用计数（保障多通信器共享资源的生命周期）；
  - 分配设备端私有节点数组（devPeers）及 host 端访问指针（devPeersHostPtr），通过 CUDA 流异步复制共享资源的设备端节点地址，确保 GPU 能直接访问通信节点信息。
- 环形通信依赖资源初始化： 分配并初始化环形拓扑通信所需的 rank 列表（host 端ring.userRanks和设备端devRingUserRanks），为环形数据传输（NCCL 核心传输模式）提供 rank 顺序支持，同时将设备端内存注册到通信器释放队列，绑定资源生命周期。
- 设备操作同步与资源释放： 过强顺序 CUDA 流（ncclStrongStream）执行设备端内存分配、地址复制等异步操作，最终同步流确保所有设备端操作完成，保证通道初始化后资源状态一致、可直接用于通信。


```c++
/**
 * @brief 初始化NCCL通信器（comm）中的指定通道（channel）
 *
 * 该函数负责初始化通信器中特定ID的通道，包括分配通道所需的节点信息（peers）、设备端节点信息（devPeers）
 * 以及环形通信相关的rank列表，同时管理共享资源的引用计数和设备内存的生命周期。
 * 通道初始化完成后，可用于NCCL集合通信操作的数据传输。
 *
 * @param comm NCCL通信器指针，包含整个通信域的配置和共享资源
 * @param channelId 要初始化的通道ID（通信器中可能包含多个并行通道）
 * @return ncclResult_t NCCL状态码，ncclSuccess表示初始化成功，其他码表示错误
 */
ncclResult_t initChannel(struct ncclComm* comm, int channelId) {
  // 获取通信器中对应ID的通道结构体（通道是通信器的子集，用于并行数据传输）
  struct ncclChannel* channel = &comm->channels[channelId];

  // 检查通道是否已初始化（id != -1表示已初始化），避免重复初始化
  if (channel->id != -1) return ncclSuccess;

  // 通信器中的总rank数（参与集合通信的进程/设备总数）
  int nRanks = comm->nRanks;
  // 本地NVLS（NVIDIA Virtual Local Switch）相关的rank数（本地节点内的虚拟rank）
  int nvlsRanks = comm->localRanks;
  // 通道的总节点数：普通rank数 + Collnet（跨节点网络通信根节点） + NVLS本地节点
  int nPeers = nRanks + 1 /* Collnet根节点（网络通信入口） */ + nvlsRanks /* NVLS本地节点 */;

  // 设置通道ID（标记通道已关联到目标ID）
  channel->id = channelId;
  // 工作队列生产计数初始化（用于同步通道的工作任务生产/消费，初始为0表示无已生产任务）
  channel->workFifoProduced = 0;

  // 获取通信器的共享资源（多个通信器可能共享的公共资源，如全局节点信息）
  struct ncclSharedResources* sharedRes = comm->sharedRes;
  cudaStream_t deviceStream; // 用于设备端操作（内存分配、复制）的CUDA流

  /**
   * 申请强顺序的CUDA流（强流保证操作的执行顺序）
   * - ncclCudaGraphNone(): 不使用CUDA Graph模式（直接执行流操作）
   * - &sharedRes->deviceStream: 共享资源中的设备流（复用共享流避免重复创建）
   * - concurrent=false: 非并发模式（流操作串行执行，保证初始化顺序）
   * - &deviceStream: 输出参数，获取到的可用CUDA流
   */
  NCCLCHECK(ncclStrongStreamAcquire(ncclCudaGraphNone(), &sharedRes->deviceStream, /*concurrent=*/false, &deviceStream));

  // 初始化通道的节点信息数组（host端，存储每个节点的通道通信信息）
  if (channel->peers == NULL) {
    // 注释说明：nRanks+1的额外空间用于Collnet根节点（网络通信节点）
    // 共享资源的内存需用ncclCalloc分配（避免与单个通信器绑定，支持多通信器共享）
    if (sharedRes->peers[channelId] == NULL) {
      // 为共享资源分配该通道的节点信息数组（大小为共享资源的全局rank数tpNRanks）
      NCCLCHECK(ncclCalloc(sharedRes->peers + channelId, sharedRes->tpNRanks));
    }

    // 为当前通道分配host端节点信息数组（使用通信器的永久内存栈，生命周期与通信器一致）
    channel->peers = ncclMemoryStackAlloc<struct ncclChannelPeer*>(&comm->memPermanent, nPeers);

    // 为每个rank绑定共享资源中的节点信息，并递增引用计数（共享资源生命周期管理）
    for (int r = 0; r < nRanks; r++) {
      // 映射当前通信器的rank到全局父rank（topParentRanks），关联共享资源中的节点信息
      channel->peers[r] = comm->sharedRes->peers[channelId] + comm->topParentRanks[r];
      // 原子递增引用计数（多通信器共享时，保证资源在最后一个使用者释放后才销毁）
      ncclAtomicRefCountIncrement(&channel->peers[r]->refCount);
    }
  }

  // 初始化通道的设备端节点信息数组（device端，供GPU直接访问节点通信信息）
  if (channel->devPeers == NULL) {
    // 若共享资源中该通道的设备端节点信息未分配，则异步分配（使用设备流）
    if (sharedRes->devPeers[channelId] == NULL) {
      NCCLCHECK(ncclCudaCallocAsync(sharedRes->devPeers + channelId, sharedRes->tpNRanks, deviceStream));
    }

    /* 注释说明：channel->devPeers是当前通道私有资源，不与其他通信器共享，
     * 因此注册到comm的CUDA释放列表，后续调用commFree()时自动释放
     */
    // 为当前通道分配设备端节点信息数组（异步分配，使用设备流）
    NCCLCHECK(ncclCudaCallocAsync(&channel->devPeers, nPeers, deviceStream));
    // 将设备端内存注册到通信器的释放队列（生命周期绑定通信器）
    ncclCommPushCudaFree(comm, channel->devPeers);

    // 分配host端指针，用于host访问设备端的节点信息结构体
    NCCLCHECK(ncclCalloc(&channel->devPeersHostPtr, nPeers));

    // 复制共享资源中设备端节点信息的地址到当前通道的设备端数组
    for (int r = 0; r < nRanks; r++) {
      // 获取共享资源中当前rank对应的设备端节点信息地址（转换为uintptr_t便于复制）
      uintptr_t addr = (uintptr_t)(comm->sharedRes->devPeers[channelId] + comm->topParentRanks[r]);
      // 异步复制地址到设备端数组（使用之前申请的设备流，保证与其他设备操作顺序）
      NCCLCHECK(ncclCudaMemcpyAsync((uintptr_t*)(channel->devPeers + r), (uintptr_t*)&addr, 1, deviceStream));
      // Host端保存设备端节点信息的地址（供host代码访问GPU上的节点信息）
      channel->devPeersHostPtr[r] = (struct ncclDevChannelPeer*)addr;
    }
  }

  // 初始化环形通信的用户rank列表（host端，存储环形拓扑中的rank顺序）
  channel->ring.userRanks = ncclMemoryStackAlloc<int>(&comm->memPermanent, nRanks);
  // 初始化设备端环形通信rank列表（异步分配，注册到通信器释放队列）
  NCCLCHECK(ncclCudaCallocAsync(&channel->devRingUserRanks, nRanks, deviceStream));
  ncclCommPushCudaFree(comm, channel->devRingUserRanks);

  /* 关键同步：确保之前的设备端操作（内存分配、地址复制）已完成
   * 因为后续通道使用时依赖devPeers等设备端资源的有效性
   */
  // 释放强顺序CUDA流（归还到共享资源）
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &sharedRes->deviceStream, /*concurrent=*/false));
  // 同步共享设备流，等待所有异步操作执行完毕
  NCCLCHECK(ncclStrongStreamSynchronize(&sharedRes->deviceStream));

  // 通道初始化成功
  return ncclSuccess;
}
```

## 5.2 setupChannel

从当前rank为起点，将环写到userRanks。

```c++
/**
 * @brief 配置NCCL通道的环形拓扑结构（依赖initChannel初始化基础资源）
 *
 * 该函数是通道初始化的后续配置步骤，在initChannel完成基础资源分配后，
 * 基于全局环形rank列表（ringRanks），为当前rank个性化配置环形拓扑信息：
 * 1. 计算当前rank在环形中的相对索引（相对于rank 0的位置）；
 * 2. 生成以当前rank为起点的环形rank序列，便于后续环形通信时快速定位上下游节点。
 * NCCL的核心集合通信（如allreduce）依赖环形拓扑实现高效数据传输，此函数是环形通信的关键配置步骤。
 *
 * @param comm 已初始化的NCCL通信器指针（需确保通道已通过initChannel完成基础初始化）
 * @param channelId 要配置的通道ID（需与initChannel的目标通道一致）
 * @param rank 当前进程/设备的本地rank（在当前通信器中的标识）
 * @param nranks 通信器中的总rank数（环形拓扑的节点总数）
 * @param ringRanks 全局环形拓扑的rank列表（预定义的环形节点顺序，所有rank共享同一列表）
 * @return ncclResult_t NCCL状态码，ncclSuccess表示配置成功，其他码表示错误
 */
static ncclResult_t setupChannel(struct ncclComm* comm, int channelId, int rank, int nranks, int* ringRanks) {
  // 打印初始化日志（TRACE是NCCL的日志宏，记录当前rank和总rank数，用于调试）
  TRACE(NCCL_INIT, "rank %d nranks %d", rank, nranks);

  // 第一步：确保通道已完成基础初始化（调用之前分析的initChannel函数，分配peers、devPeers等资源）
  // 若通道未初始化则触发初始化，已初始化则直接返回成功（initChannel有幂等性）
  NCCLCHECK(initChannel(comm, channelId));

  // 获取当前通道的环形拓扑结构体（环形拓扑是NCCL数据传输的核心拓扑模式）
  struct ncclRing* ring = &comm->channels[channelId].ring;

  // 查找两个关键索引：
  // ixZero：全局环形列表（ringRanks）中rank 0的位置（环形拓扑的逻辑起点）
  // ixRank：全局环形列表（ringRanks）中当前rank的位置
  int ixZero = 0, ixRank = 0;
  for (int i = 0; i < nranks; i++) {
    if (ringRanks[i] == 0) ixZero = i;   // 定位rank 0在全局环形中的索引
    if (ringRanks[i] == rank) ixRank = i;// 定位当前rank在全局环形中的索引
  }

  /**
   * 计算当前rank在环形中的相对索引（ring->index）
   * 逻辑：以rank 0为环形起点，计算当前rank相对于起点的“环形距离”
   * - (ixRank - ixZero)：当前rank与rank 0的索引差（可能为负）
   * - +nranks：避免负数（确保计算结果非负）
   * - %nranks：确保索引落在[0, nranks-1]范围内（符合环形拓扑的循环特性）
   * 示例：nranks=4，ixZero=1（rank0在全局列表索引1），ixRank=3（当前rank在全局列表索引3）
   * 计算：(3-1 +4) %4 = 2 → 当前rank在环形中的相对索引为2
   */
  ring->index = (ixRank - ixZero + nranks) % nranks;

  /**
   * 生成以当前rank为起点的环形rank序列（ring->userRanks）
   * 逻辑：将全局环形列表（ringRanks）按“当前rank的位置”重新排序，使序列起点为当前rank
   * 目的：后续通信时，可通过索引直接定位上下游节点（如index+1=下一个节点，index-1=上一个节点）
   * 示例：全局ringRanks=[2,0,3,1]，当前rank=3（ixRank=2），nranks=4
   * 循环i=0→3：
   * i=0：(0+2)%4=2 → ringRanks[2]=3（当前rank自身）
   * i=1：(1+2)%4=3 → ringRanks[3]=1（下一个节点）
   * i=2：(2+2)%4=0 → ringRanks[0]=2（下下个节点）
   * i=3：(3+2)%4=1 → ringRanks[1]=0（上一个节点）
   * 最终userRanks=[3,1,2,0]（以当前rank为起点的环形序列）
   */
  for (int i = 0; i < nranks; i++) {
    ring->userRanks[i] = ringRanks[(i + ixRank) % nranks];
  }

  // 环形拓扑配置完成
  return ncclSuccess;
}
```

# 后续

**然后执行ncclTransportP2pSetup建立当前rank和prev，next的通信链路。**


# 参考

- [链接](https://zhuanlan.zhihu.com/p/658868934)