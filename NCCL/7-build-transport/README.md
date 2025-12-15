# 0 背景

上节以ringGraph为例介绍了机器间channel的连接过程，现在**环里每个rank都知道了从哪个rank接收数据以及将数据发送给哪个rank**，本节具体介绍下P2P和rdma NET场景下数据通信链路的建立过程。

nccl通过**ncclTransportP2pSetup**完成数据通信链路的建立，还是以上节两机十六卡的环为例：

- 第一台机器的环
```sh
graph->intra: GPU/0 GPU/7 GPU/6 GPU/3 GPU/2 GPU/5 GPU/4 GPU/1
graph->inter: NET/0 NET/0
```

- 第二台机器的环
```sh
graph->intra: GPU/10 GPU/9 GPU/8 GPU/13 GPU/12 GPU/15 GPU/14 GPU/11
graph->inter: NET/0 NET/0
```

# 1 相关数据结构

## 1.1 ncclChannelPeer
首先介绍一下ncclPeer，ncclPeer保存了两个connector，对于rank 10，send负责和rank 9通信，recv负责和rank 1通信。后续为了方便表述，假设rank 10叫接收端，rank 1叫发送端。

```c++
struct ncclChannelPeer {
  struct ncclConnector send[NCCL_MAX_CONNS]; // send负责和rank 9通信
  struct ncclConnector recv[NCCL_MAX_CONNS]; // recv负责和rank 1通信
  int refCount;
};
```

## 1.2 ncclConnector
```c++
struct ncclConnector {
  int connected; // 是否已链接
  int hasSeen;   // 该连接器是否已经建立连接
  int p2pOnly;   // 是否仅支持点对点通信，非0时只允许 P2P 通信
  struct ncclProxyConnector proxyConn;     // 包含了代理连接的信息: rank, 进程信息, 代理进度回调函数
  struct ncclTransportComm* transportComm; // 定义了传输层的方法和操作
  void* transportResources;                // 指向与特定传输相关的资源数据
  struct ncclConnInfo conn;                // 存储实际的连接信息
};
```

## 1.3 ncclConnInfo
ncclConnInfo记录了通信过程上下文信息，本节只需要关注buffs，即通信过程中的buffer，buffs实际位于transportResources，这里只是指针指过去。

```c++
struct ncclConnInfo {
  // 环形缓冲区数组（每个元素对应一种 NCCL 协议）：里面每一项就是一个环形通信缓冲区
  // NCCL_NUM_PROTOCOLS：NCCL 支持的通信协议数量（如 SIMPLE（基础）、LL（低延迟）、PROTO（通用）等）
  // 核心规则：Local for recv, remote for send（接收时指向本地缓冲区，发送时指向对端缓冲区）
  // 本质：数据传输的载体，NCCL 基于 GPU P2P/RDMA 直接读写对端缓冲区，无需中间拷贝
  char *buffs[NCCL_NUM_PROTOCOLS];
  // 内存句柄数组（与 buffs 一一对应）
  // 用途：管理缓冲区的内存访问凭证（如 CUDA IPC 句柄、RDMA 内存注册句柄、NCCL 内部内存管理句柄）
  // 作用：跨设备 / 进程访问缓冲区的 “钥匙”（如 GPU 直连时，需通过句柄注册内存以启用 P2P 访问）
  void* mhandles[NCCL_NUM_PROTOCOLS];
  // - 核心规则：Local for recv, remote for send；
  // - 生产者 - 消费者模型的 “已写入 / 可读取” 边界：
  // 接收时：本地指针，标记对端已写入、本端可读取的buffs[i] 里的位置；
  // 发送时：远程指针，本端写完数据后原子更新对端 tail，告知对端可读取。
  uint64_t *tail;
  // 核心规则：Local for send, remote for recv
  // 生产者 - 消费者模型的 “可写入 / 已读取” 边界
  // 发送时：本地指针，标记本端可写入的位置（避免覆盖未读取数据）；
  // 接收时：远程指针，本端读完数据后原子更新对端 head，告知对端可继续写入
  // head/tail 配合实现无锁环形缓冲区，是 NCCL 低开销通信的核心。
  uint64_t *head;

  int flags;          // 是否启用低延迟模式、P2P 访问、调试模式、跨节点通信等（每一bit对应一个功能）
  int shared;         // 标记缓冲区是否被多通信连接复用（如多进程 / 线程通信时，共享缓冲区减少内存开销）
  int stepSize;       // 仅针对 SIMPLE 协议（基础协议，适配小数据量）: 每次读写缓冲区的字节数 / 元素数，控制数据传输粒度
  // GPU 直连时，需知道对端内存的虚拟 / 物理地址，通过该字段完成指针交换，实现直接访问
  void **ptrExchange; // 直连模式下，交换双方的核心指针（如 buffs/head/tail)
  // NCCL 集体通信（如 AllReduce）的归约操作（sum/max 等）需预缩放数据，该字段交换缩放因子指针，确保双方参数一致
  uint64_t* redOpArgExchange;

  // GPU 无法直连时（如跨节点、无 P2P 权限），通过 CPU Proxy 中转数据；
  // connFifo 是 GPU 与 Proxy 之间的异步 FIFO 队列，缓存待传输的数据 / 指令
  struct ncclConnFifo* connFifo; // 用于 GPU 和代理(proxy)之间的通信

  uint64_t step;      // 记录当前连接在集体通信流程中的步骤（如 scatter → reduce → gather）
  uint64_t llLastCleaning; // 针对 LL（低延迟）协议：LL 协议复用缓冲区，需定期清理已读取的旧数据
  // 关联通信使用的底层网络设备，调用 ncclNet 库 API 实现跨节点 / 设备传输
  // NCCL 抽象的网络设备句柄（屏蔽 NVLink/InfiniBand/Ethernet 差异）
  ncclNetDeviceHandle_t netDeviceHandle;
}
```

> 在 NCCL 中，代理(proxy) 指的是运行在 CPU 上的辅助线程或进程，负责处理**GPU 无法直接完成的通信任务**：<br>
> 工作方式: GPU <---> 代理(proxy) <---> 网络硬件 <---> 远程节点;
> 网络协议处理：处理 TCP/IP、InfiniBand 等复杂网络协议;
> 内存管理：管理主机内存和设备内存之间的数据传输;
> 异步操作：执行非阻塞的通信操作，提高整体性能;
> 多进程协调：在多进程环境中协调通信;
> example1:在使用 RDMA 网络时，代理处理复杂的连接建立和内存注册
> example2:在多播或广播操作中，代理可能负责数据包的复制和转发
> example3:当需要临时缓冲数据时，代理管理主机端的缓冲区

- connFifo 使用场景
```sh
场景一：发送数据
GPU 准备好数据
GPU 向 connFifo 写入发送请求
Proxy 从 connFifo 读取请求
Proxy 使用网络 API 发送数据
Proxy 通知 GPU 操作完成
场景二：接收数据
Proxy 接收到网络数据
Proxy 将接收信息写入 connFifo
GPU 轮询检查 connFifo
GPU 处理接收到的数据
```

## 1.4 ncclConnFifo

ncclConnFifo结构体主要用于 GPU 和代理(proxy)之间的通信，GPU 内核将通信请求放入此结构体并通过 connFifo 变量传递给代理处理。

```c++
struct ncclConnFifo {
  int mode;      // 通信模式：NCCL_MODE_NORMAL(正常模式)、NCCL_MODE_OFFSET(偏移模式)、NCCL_MODE_PTR(指针模式)
  int offset;    // 数据在缓冲区中的偏移量（字节为单位）
  ssize_t size;  // 要传输的数据大小（字节为单位）
  void* ptr;     // 数据缓冲区的指针（在指针模式下使用）
};
```

## 1.5 ncclTransportComm

ncclConnector的 **ncclTransportComm** 定义了一系列的通信相关的函数指针，用户可以自己实现这些接口，`ncclTransport 定义了send和recv两个ncclTransportComm`，本节会介绍下P2P和NET两个ncclTransport。

```c++
struct ncclTransportComm {
  // 初始化和设置连接
  ncclResult_t (*setup)(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo*, struct ncclPeerInfo*, struct ncclConnect*, struct ncclConnector*, int channelId, int connIndex);
  // 建立连接
  ncclResult_t (*connect)(struct ncclComm* comm, struct ncclConnect*, int nranks, int rank, struct ncclConnector*);
  // 释放连接资源
  ncclResult_t (*free)(struct ncclConnector*);
  // 初始化代理共享资源
  ncclResult_t (*proxySharedInit)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, int nChannels);
  // 设置代理连接
  ncclResult_t (*proxySetup)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done);
  // 连接代理
  ncclResult_t (*proxyConnect)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done);
  // 释放代理资源
  ncclResult_t (*proxyFree)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState);
  // 代理进度处理（执行实际的通信操作）
  ncclResult_t (*proxyProgress)(struct ncclProxyState* proxyState, struct ncclProxyArgs*);
  // 注册缓冲区到代理
  ncclResult_t (*proxyRegister)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done);
  // 从代理注销缓冲区
  ncclResult_t (*proxyDeregister)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, int* done);
};
```

## 1.6 ncclTransport

**ncclTransport** 是 NCCL 中用于定义不同类型传输方式的抽象接口结构体。它的主要作用可以总结为以下几个方面：
- 统一传输接口：为各种传输方式（P2P、SHM、NET、COLLNET等）提供统一的接口规范，使得上层代码可以以一致的方式调用不同的传输机制
- 传输能力检测：通过 canConnect 函数判断两个节点之间是否可以建立特定类型的连接，支持运行时动态选择最优传输路径
- 双向通信支持：分别定义发送(send)和接收(recv)两个方向的传输接口，每个方向都包含完整的连接生命周期管理函数
- 插件化架构：支持多种传输方式的插件化实现，易于扩展新的传输机制（如新增的 PROFILER 传输）
-  核心功能包括：连接管理：建立、配置和释放连接；代理交互：与 CPU 代理进行通信协调；资源管理：注册和注销缓冲区内存；数据传输：执行实际的数据发送和接收操作

```c++
struct ncclTransport {
  const char name[8];  // 传输方式的名称（如 "P2P"、"SHM"、"NET"等）

  // 检查两个节点是否可以通过此传输方式连接
  ncclResult_t (*canConnect)(int*, struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo*, struct ncclPeerInfo*);

  // 发送传输通信接口
  struct ncclTransportComm send;

  // 接收传输通信接口
  struct ncclTransportComm recv;
};

// 通过net 创建传输方式 :
// */nccl/src/transport/net.cc 中
struct ncclTransport netTransport = {
  "NET",
  canConnect,
  // ncclTransportComm 对应的函数
  { sendSetup, sendConnect, sendFree, proxySharedInit, sendProxySetup, sendProxyConnect, sendProxyFree, sendProxyProgress, sendProxyRegBuffer, sendProxyDeregBuffer },
  // ncclTransportComm 对应的函数
  { recvSetup, recvConnect, recvFree, proxySharedInit, recvProxySetup, recvProxyConnect, recvProxyFree, recvProxyProgress, recvProxyRegBuffer, recvProxyDeregBuffer }
};
```

# 2 ncclTransportP2pSetup

ncclTransportP2pSetup 函数的主要作用是建立 NCCL 通信中的点对点（P2P）连接。以下是其主要功能：
- 初始化 P2P 连接：为指定的连接索引 (connIndex) 建立所有通道（channels）与对应节点之间的点对点连接，处理发送和接收两个方向的连接；
- 连接管理机制：使用环形连接方式，每个 rank 依次与其他 ranks 建立连接；支持批量处理多个连接，通过 maxPeers 参数控制每轮处理的连接数，使用 connectSend 和 connectRecv 位图标记需要建立的连接；
- 传输层选择：调用 selectTransport 模板函数自动选择合适的传输方式（如 P2P、共享内存、网络等）根据拓扑结构和节点信息确定最佳传输路径；
- 数据交换与同步：使用 bootstrap 通信机制交换连接信息，实现异步连接建立过程，支持连接进度检查，完成主机和设备端连接信息的同步复制；
- 资源管理：管理连接数据缓冲区内存分配和释放，协调 CUDA 流资源的使用，最终进行全局同步确保所有 rank 完成连接建立.

## 2.1 回顾相关数据结构
```c++
struct ncclTopoGraph {
  // Input / output
  int id; // ring : 0, tree : 1, collnet : 2, nvls : 3, collnetDirect : 4
  int pattern;
  int crossNic;
  int collNet;
  int minChannels;
  int maxChannels;
  // Output
  int nChannels;
  float bwIntra;
  float bwInter;
  float latencyInter;
  int typeIntra;
  int typeInter;
  int sameChannels;
  int nHops;
  int intra[MAXCHANNELS*NCCL_TOPO_MAX_NODES];
  int64_t inter[MAXCHANNELS*2];
};
```

> **ncclConnect**
> ncclConnect 是 NCCL 中专用于通信连接建立阶段的 **"握手数据载体"**，是节点 / 设备间**交换连接参数**的核心轻量级结构体，
> 本质是一个固定大小的通用**字节缓冲区**，用于序列化传输连接初始化所需的所有元信息;
> 通信第一步握手阶段的 “数据信封”————将所有握手参数(如缓冲区地址、内存句柄、协议配置、同步变量等)序列化到字节数组中，通过底层网络（NVLink/InfiniBand/Ethernet）;传输，避免直接传输复杂结构体导致的跨架构 / 对齐问题;
> CONNECT_SIZE: 通常为几百字节～几 KB;
> 作用: 序列化存储连接握手所需的所有元信息（见下文），传输后由对端反序列化解析;
> 真实存储： 存储了建立连接必需的参数，主要为 ncclConnInfo 里的参数.
> 为何不直接传输ncclConnInfo ? 答: GPU/CPU、x86/ARM 等不同架构的结构体对齐规则不同，直接传输 ncclConnInfo 这类复杂结构体会导致内存错乱；字节数组是 “原始载体”，序列化 / 反序列化时可统一处理对齐、大小端问题;
> 反序列化：对端接收后，从 data 中解析出所有参数，填充到本地的 ncclConnInfo 结构体
> ncclConnect 完成使命后被释放，后续通信完全依赖 ncclConnInfo 这个运行时上下文.

```c++
// 作为不同传输方式（P2P、SHM、NET等）之间交换连接信息ncclConnInfo的数据结构
// 每种传输方式（如 P2P、共享内存、网络）都可以将自己的连接信息序列化到这个结构中
struct ncclConnect {
  char data[CONNECT_SIZE];
};
```

## 2.2 ncclTransportP2pSetup函数解释

此函数是 NCCL 中点对点（P2P）传输层的核心初始化入口，负责为当前 Rank 与集群中**其他所有 Rank**建立可靠的 P2P 通信连接：从资源分配、传输协议选择，到跨 Rank 握手参数交换，最终完成连接信息（ncclConnInfo）的 GPU 同步，为后续集体通信（AllReduce/AllGather 等）提供可用的 P2P 传输通道。

- 作用：为 comm（当前通信器）中所有 Rank 建立 P2P 传输连接，覆盖 “传输协议选择 → 连接参数交换 → 连接状态确认 → GPU 内存同步” 全流程；
- 输入：通信器 comm、拓扑图 graph、连接索引 connIndex（多连接场景下的索引）；
- 输出：ncclResult_t 状态码，标识连接建立是否成功。

**核心3点:** <br>

1. **传输协议选择**（selectTransport）: 根据设备拓扑（如 NVLink 是否可用）、网络类型（IB/Ethernet）选择最优传输层. 该函数会将传输层的关键参数（如缓冲区地址、内存句柄）序列化到 ncclConnect 结构体的 data 数组中，为后续握手做准备.
- 优先选 NVLink（GPU 直连，低延迟高带宽）；
- 其次选 InfiniBand（跨节点高速网络）；
- 最后选 Ethernet（通用网络）；

2. **握手参数交换**（bootstrapSend/Recv）： 通过 NCCL 底层的 bootstrap 机制，与目标 Peer 交换 ncclConnect 握手数据（即上阶段初始化的传输层参数）。
- bootstrapSend/Recv：NCCL 封装的引导通信接口，基于底层网络（如 MPI/TCP）实现跨 Rank 的小数据传输，专门用于交换连接参数；
- 核心逻辑：当前 Rank 先发送自己的 ncclConnect 数据（传输层参数）给目标 Peer，再接收目标 Peer 回传的 ncclConnect 数据，完成参数交换。


3. **真正连接**
- 连接的信息存储在 ncclConnector 结构体中, ncclConnInfo 也是其中的成员;
- conn->transportComm->connect : 传输层的最终连接接口，基于交换后的 ncclConnect 数据初始化 ncclConnInfo（如缓冲区地址、head/tail 指针）等；
- cudaMemcpyAsync：将主机端的 ncclConnInfo 拷贝到 GPU 显存，因为后续 NCCL 集体通信的核函数运行在 GPU 上，需要直接访问连接信息；
- 标记确保所有通道都完成连接后才进入下一轮；

```c++
// NCCL点对点（P2P）传输层连接建立与初始化核心函数
// 参数：
//   comm: NCCL通信器，包含rank信息、通道配置等
//   graph: 系统硬件拓扑图，用于选择最优传输路径
//          当 graph 参数为 NULL 时，表示这是通用的点对点连接建立过程，而不是针对特定拓扑图的连接
//          适用于任意两个 GPU 之间的直接连接
//   connIndex: 连接索引（区分同一通道的不同传输路径）
//              在同一个逻辑通道（Channel）内，为不同的物理传输路径（或传输实例）分配唯一索引;
//              从而支持多路径通信、链路容错或带宽聚合，确保 NCCL 能灵活利用硬件提供的所有可用传输资源
// 返回：ncclSuccess表示成功，其他值为错误码
ncclResult_t ncclTransportP2pSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, int connIndex) {
  // 传输设置过程中使用的流；用于P2P预连接和CUDA Graph
  ncclResult_t ret = ncclSuccess;          // 函数返回结果，初始化为成功
  // maxPeers 个一级指针
  // 每个peer 2 * MAXCHANNELS * sizeof(struct ncclConnect) 的大小
  // 2 表示前面是 recvData 后面是 recvData 都存在 data 每个指针里
  // ncclConnect 是对
  struct ncclConnect** data;               // 存储所有Peer的ncclConnect数据 recvData、sendData 都存储于此
  struct ncclConnect** recvData = NULL;    // 接收连接的ncclConnect指针
  struct ncclConnect** sendData = NULL;    // 发送连接的ncclConnect指针
  int done = 0;                            // 已完成连接的peer计数
  int maxPeers = ncclParamConnectRoundMaxPeers();  // 单次批量处理的最大peer数（避免并发过载）

  ...

  // 分配data内存：maxPeers个struct ncclConnect*指针
  NCCLCHECK(ncclCalloc(&data, maxPeers));
  // 分配recvData内存：maxPeers个指针，失败则跳转到fail标签
  NCCLCHECKGOTO(ncclCalloc(&recvData, maxPeers), ret, fail);
  // 分配sendData内存：maxPeers个指针，失败则跳转到fail标签
  NCCLCHECKGOTO(ncclCalloc(&sendData, maxPeers), ret, fail);

  ...

  // 首次初始化：遍历除自身外的所有rank（i=0为自身，故从1开始）
  for (int i=1; i<comm->nRanks; i++) {
    // 生成引导通信标签：结合i（步长）和graph id（拓扑标识）
    // 底层引导通信（bootstrap）的唯一标签，确保不同 Rank / 拓扑的连接参数不串流
    int bootstrapTag = (i<<8) + (graph ? graph->id+1 : 0);
    // 计算接收端peer：当前rank - i 后取模（环形拓扑中的前向peer
    // 0 : 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1
    // 1: 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2
    // 2: 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3
    int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;
    // 计算发送端peer：当前rank + i 后取模（环形拓扑中的后向peer）:
    // 0 : 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    // 1: 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0
    // 3: 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
    int sendPeer = (comm->rank + i) % comm->nRanks;
    // 当前rank与recvPeer的recv通道掩码（bit位表示通道是否启用）
    // NCCL 会把通信带宽拆分为多个独立的 Channel（比如 8/16/32 个），
    // 每个 Channel 对应一套独立的 ncclConnInfo（连接信息）和传输队列：
    // 掩码的核心价值：快速筛选需要建连的 Channel，避免无差别遍历所有 Channel，提升建连效率
    uint64_t recvMask = comm->connectRecv[recvPeer]; // 21 (0b10101) --> 0/2/4 通道
    // 当前rank与sendPeer的send通道掩码（bit位表示通道是否启用）// 10 (0b01010) --> 1/3/5 通道
    // 接收（recv）和发送（send）完全可以复用同一个 Channel —— 这是 NCCL 的默认设计，
    // 也是 NVLink/IB 等全双工高速链路的核心使用方式；同一个逻辑 Channel 不仅能承载双向通信，
    // 还是 NCCL 优化带宽利用率、减少资源开销的关键策略。
    uint64_t sendMask = comm->connectSend[sendPeer];

    // data[i]存储与指定send/recv peer的所有send/recv连接信息
    // 数据按通道数打包：前N个条目为recvData，后M个为sendData
    // 每个data条目不一定有相同数量的send/recv连接
    int p = i-(done+1);  // 计算data数组的索引（相对done的偏移）
    // 若存在需要连接的recv/send通道，则分配或清空data[p]内存
    if (recvMask || sendMask) {
      if (data[p] == NULL)
        // 分配2*MAXCHANNELS个ncclConnect结构体（预留足够通道空间），失败跳转
        NCCLCHECKGOTO(ncclCalloc(data + p, 2 * MAXCHANNELS), ret, fail);
      else
        // 若已分配则清零，避免旧数据干扰
        memset(data[p], 0, 2 * MAXCHANNELS * sizeof(struct ncclConnect)); // 接收+发送通道的总大小
    }

    //
    // 从data 中提取 recv 对应的指针, data[i] 中先存储recv连接，再存储send连接, 每个channel 都要建立连接
    recvData[p] = data[p];  // recvData[p]指向data[p]起始（recv连接起始位置）
    int sendChannels = 0, recvChannels = 0;  // 统计send/recv通道数量
    int type;  // 存储传输类型（如NVLink/PCIe）

    // 遍历所有通道，处理recv连接 --> 为接收通道选择传输层（模板参数0表示接收）
    for (int c=0; c<MAXCHANNELS; c++) {
      // 若当前通道c在recvMask中启用
      if (recvMask & (1UL<<c)) {
        // 选择recv方向（模板参数0）的传输类型，初始化ncclConnect，失败跳转;
        // selectTransport核心逻辑是：根据设备拓扑（如 NVLink 是否可用）、网络类型（IB/Ethernet）选择最优传输层;
        // selectTransport 会将传输层的关键参数（如缓冲区地址、内存句柄）序列化到 ncclConnect 结构体的 data 数组中，为后续握手做准备;
        NCCLCHECKGOTO(selectTransport<0>(comm, graph, recvData[p]+recvChannels++, c, recvPeer, connIndex, &type), ret, fail);
      }
    }

    // 从data 中提取 send 对应的指针, sendData[p]指向recv连接后的位置
    sendData[p] = recvData[p]+recvChannels;

    // 遍历所有通道，处理send连接 --> 为发送通道选择传输层（模板参数1表示发送）
    for (int c=0; c<MAXCHANNELS; c++) {
      // 若当前通道c在sendMask中启用
      if (sendMask & (1UL<<c)) {
        // 选择send方向（模板参数1）的传输类型，初始化ncclConnect，失败跳转
        NCCLCHECKGOTO(selectTransport<1>(comm, graph, sendData[p]+sendChannels++, c, sendPeer, connIndex, &type), ret, fail);
      }
    }

    // 开始 bootstrap 交换信息
    // 特殊情况：sendPeer与recvPeer为同一节点/设备（如环形拓扑闭环）
    if (sendPeer == recvPeer) {
      // 若存在需要连接的通道
      if (recvChannels+sendChannels) {
        // 发送当前节点的连接信息给peer（包含recv+send所有通道）
        NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, recvPeer, bootstrapTag, data[p], sizeof(struct ncclConnect)*(recvChannels+sendChannels)), ret, fail);
        // 接收peer的连接信息（同步数据）
        NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, recvPeer, bootstrapTag, data[p], sizeof(struct ncclConnect)*(recvChannels+sendChannels)), ret, fail);
        // 调整sendData/recvData指向（因send/recv为同一peer，交换位置）
        sendData[p] = data[p];
        recvData[p] = data[p]+sendChannels;
      }
    } else {
      // 普通情况：sendPeer与recvPeer不同
      if (recvChannels)
        // 发送recv通道信息给recvPeer
        NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, recvPeer, bootstrapTag, recvData[p], sizeof(struct ncclConnect)*recvChannels), ret, fail);
      if (sendChannels)
        // 发送send通道信息给sendPeer
        NCCLCHECKGOTO(bootstrapSend(comm->bootstrap, sendPeer, bootstrapTag, sendData[p], sizeof(struct ncclConnect)*sendChannels), ret, fail);
      if (sendChannels)
        // 接收sendPeer的send通道信息
        NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, sendPeer, bootstrapTag, sendData[p], sizeof(struct ncclConnect)*sendChannels), ret, fail);
      if (recvChannels)
        // 接收recvPeer的recv通道信息
        NCCLCHECKGOTO(bootstrapRecv(comm->bootstrap, recvPeer, bootstrapTag, recvData[p], sizeof(struct ncclConnect)*recvChannels), ret, fail);
    }

    // 若达到单次最大处理peer数，或遍历到最后一个rank，则开始建立实际连接
    if (i-done == maxPeers || i == comm->nRanks-1) {
      // 循环直到所有通道与所有rank完成连接
      bool allChannelsConnected;
      allChannelsConnected = false;
      while (!allChannelsConnected) {
        allChannelsConnected = true;  // 假设所有通道已连接，后续验证
        // 遍历当前批次的所有peer（done+1到i）
        for (int j=done+1; j<=i; j++) {
          // 重新计算当前j对应的recvPeer和sendPeer
          int recvPeer = (comm->rank - j + comm->nRanks) % comm->nRanks;
          int sendPeer = (comm->rank + j) % comm->nRanks;
          uint64_t recvMask = comm->connectRecv[recvPeer];
          uint64_t sendMask = comm->connectSend[sendPeer];

          int p = j-(done+1);  // 计算data数组索引
          int sendDataOffset = 0;  // send通道偏移量（遍历通道时计数）
          int recvDataOffset = 0;  // recv通道偏移量（遍历通道时计数）

          // 遍历所有通道
          for (int c=0; c<MAXCHANNELS; c++) {
            // 处理send连接
            if (sendMask & (1UL<<c)) {
              // 获取通道c中sendPeer对应的send连接器（connIndex为当前连接索引）
              struct ncclConnector* conn = comm->channels[c].peers[sendPeer]->send + connIndex;
              // 若连接器未完成连接
              if (conn->connected == 0) {
                // 调用传输层的connect函数完成连接，失败跳转
                NCCLCHECKGOTO(conn->transportComm->connect(comm, sendData[p] + sendDataOffset, 1, comm->rank, conn), ret, fail);
                if (ret == ncclSuccess) {
                  conn->connected = 1;  // 标记连接完成
                  // 异步将host端conn信息拷贝到device端（供GPU访问）
                  CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[sendPeer]->send[connIndex], &conn->conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream), ret, fail);
                } else if (ret == ncclInProgress) {
                  // 连接未完成，标记需要继续循环
                  allChannelsConnected = false;
                }
              }
              sendDataOffset++;  // 移动到下一个send通道
            }

            // 处理recv连接
            if (recvMask & (1UL<<c)) {
              // 获取通道c中recvPeer对应的recv连接器（connIndex为当前连接索引）
              struct ncclConnector* conn = comm->channels[c].peers[recvPeer]->recv + connIndex;
              // 若连接器未完成连接
              if (conn->connected == 0) {
                // 调用传输层的connect函数完成连接，失败跳转
                NCCLCHECKGOTO(conn->transportComm->connect(comm, recvData[p] + recvDataOffset, 1, comm->rank, conn), ret, fail);
                if (ret == ncclSuccess) {
                  conn->connected = 1;  // 标记连接完成
                  // 异步将host端conn信息拷贝到device端（供GPU访问）
                  CUDACHECKGOTO(cudaMemcpyAsync(&comm->channels[c].devPeersHostPtr[recvPeer]->recv[connIndex], &conn->conn, sizeof(struct ncclConnInfo), cudaMemcpyHostToDevice, hostStream), ret, fail);
                } else if (ret == ncclInProgress) {
                  // 连接未完成，标记需要继续循环
                  allChannelsConnected = false;
                }
              }
              recvDataOffset++;  // 移动到下一个recv通道
            }
          }
        }

        // 打印连接进度（仅rank0且开启报告时）

        ...
      }

      done = i;  // 更新已完成的peer计数
    }
  }

  // 统计总耗时并打印日志
  ...


  // 同步所有rank：防止部分rank提前销毁连接导致资源错误
  ...

  TIME_PRINT("P2P Setup/Connect");  // 打印P2P建立全过程的时间统计

exit:  // 资源释放出口（正常结束或错误跳转）
  // 释放data数组中每个元素的内存
  ...

  // 同步deviceStream与hostStream（确保异步操作完成）
  // 释放hostStream资源
  // 释放deviceStream资源
  ...
  return ret;  // 返回函数结果

fail:  // 错误处理入口
  goto exit;  // 跳转到资源释放出口
}
```

# 3 传输层的选择 : selectTransport and setup

## 3.1 selectTransport

这里的 **setup** 对应ncclTransportComm 里的setup 函数, 不同的传输层，setup 函数实现不同。

```c++
struct ncclTransport* ncclTransports[NTRANSPORTS+1] = {
  &p2pTransport,
  &shmTransport,
  &netTransport,
  &collNetTransport,
  &profilerTransport // Not really used for transport, only to create proxy ops polling on profiler counters.
};

/**
 * @brief 为指定的Peer连接选择合适的传输层（Transport）并完成初始化
 *
 * NCCL支持多种传输层（如NVLink、PCIe、TCP等），该函数会遍历所有可用传输层，
 * 找到第一个能在当前两个Rank之间建立连接的传输层，完成连接初始化后返回；
 * 若所有传输层都不可用，则返回系统错误。
 *
 * @param comm        NCCL通信器上下文，包含全局通信配置、Rank信息等
 * @param graph       拓扑图结构体，存储设备/节点间的拓扑关系（如NVLink拓扑、PCIe拓扑）
 * @param connect     连接上下文，存储连接的基础配置（如地址、端口、硬件路径等）
 * @param channelId   通道ID，NCCL一个通信器会划分多个通道并行通信，每个通道独立管理连接
 * @param peer        目标Peer的Rank编号（对端Rank）
 * @param connIndex   连接索引，单个Peer可能存在多个并行连接，通过索引区分
 * @param transportType 输出参数：返回选中的传输层类型索引（NTRANSPORTS枚举值）；若为NULL则不返回
 *
 * @return ncclResult_t 成功返回ncclSuccess，无可用传输层返回ncclSystemError
 *
 * @note 原代码中"type"变量未定义，推测为笔误（应为发送/接收类型标识，1=SEND，其他=RECV）
 */
static ncclResult_t selectTransport(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclConnect* connect, int channelId, int peer, int connIndex, int* transportType) {
  // 获取本地Rank的Peer信息（包含Rank编号、总线ID、设备ID等）
  struct ncclPeerInfo* myInfo = comm->peerInfo + comm->rank;
  // 获取目标Peer（对端Rank）的Peer信息
  struct ncclPeerInfo* peerInfo = comm->peerInfo + peer;

  // 根据传输类型（SEND/RECV）获取对应的连接器（Connector）
  // Connector是NCCL管理单个连接的核心结构体，包含传输层上下文、连接状态等
  // 【注】原代码中"type"变量未定义，推测为函数隐式参数（1=发送连接，其他=接收连接）
  struct ncclConnector* connector = (type == 1) ?
                                    comm->channels[channelId].peers[peer]->send + connIndex :  // 发送连接的连接器
                                    comm->channels[channelId].peers[peer]->recv + connIndex;    // 接收连接的连接器

  // 遍历所有可用传输层（NTRANSPORTS为传输层总数，如NVLink/PCIe/TCP等）
  for (int t = 0; t < NTRANSPORTS; t++) {
    // 获取当前遍历的传输层实例（包含该传输层的能力检测、连接建立、数据收发等接口）
    struct ncclTransport *transport = ncclTransports[t];
    // 根据传输类型（SEND/RECV）获取对应的传输层通信上下文
    struct ncclTransportComm* transportComm = (type == 1) ? &transport->send : &transport->recv;

    int ret = 0;  // 标记当前传输层是否支持该连接（1=支持，0=不支持）
    // 检查当前传输层是否能在本地Rank和目标Peer之间建立连接
    // 核心逻辑：基于拓扑信息判断传输层硬件路径是否可用（如NVLink是否直连、PCIe是否可达）
    NCCLCHECK(transport->canConnect(&ret, comm, graph, myInfo, peerInfo));

    // 若当前传输层支持该连接，则完成初始化并返回
    if (ret) {
      // 将连接器绑定到选中的传输层通信上下文
      connector->transportComm = transportComm;
      // 执行传输层的连接初始化（如分配硬件资源、绑定地址、建立会话等）
      // 这里的setup 对应ncclTransportComm 里的setup 函数, 不同的传输层，setup 函数实现不同
      NCCLCHECK(transportComm->setup(comm, graph, myInfo, peerInfo, connect, connector, channelId, connIndex));
      // 若需要返回传输层类型，则将当前索引赋值给输出参数
      if (transportType) *transportType = t;
      // 传输层选择并初始化成功，返回成功状态
      return ncclSuccess;
    }
  }

  // 所有传输层都不支持该连接，打印警告日志（包含Rank和总线ID便于定位拓扑问题）
  WARN("No transport found for rank %d[%lx] -> rank %d[%lx]", myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId);
  // 返回系统错误（无可用传输层）
  return ncclSystemError;
}
```

## 3.2 ncclTranportComm::setup 函数

**每个传输层都有各自的setup 函数**

该函数是 NCCL网络传输层（NET Transport） 中发送连接的核心初始化函数，核心作用是：为本地 Rank 到目标 Peer 的发送连接**确定网络设备、GPU Direct RDMA（GDR）启用状态、代理 Rank 等关键参数**，`通过代理通信完成连接配置，并填充连接信息结构体，为后续网络传输（如跨节点通信）的发送操作奠定基础`。<br>

- GPU Direct RDMA（GDR）: ncclTopoCheckGdr会检测硬件（GPU / 网卡）是否支持 GDR、驱动是否配置正确，最终决定是否启用;
- 代理 Rank（proxyRank）: 跨节点通信时，NCCL 会**选择节点内一个 Rank 作为 “网络代理”**，负责该节点所有 Rank 的网络数据转发（减少网卡资源占用）；若proxyRank == myInfo->rank，说明是同节点通信，无需代理;
  - 集群中每个节点通常只配备 1~2 张高速网卡（如 InfiniBand、NVIDIA BlueField、RoCE 网卡），但节点内的 GPU 数量远多于网卡数（比如 1 个节点 8 张 GPU/Rank，却只有 1 张 IB 网卡）;
  -
- 共享缓冲区（shared buffers）NCCL 为减少内存分配 / 释放开销，对非首连接（connIndex≠0）复用缓冲区，shared标志控制该行为，默认启用（1）。
- NetPXNNCCL 的跨节点网络 P2P 优化特性，针对多节点集群优化网络路由，仅在跨节点通信（proxyRank≠本地Rank）时启用。

**代理的作用** <br>
- 集群中每个节点通常只配备 1~2 张高速网卡（如 InfiniBand、NVIDIA BlueField、RoCE 网卡），但节点内的 GPU 数量远多于网卡数（比如 1 个节点 8 张 GPU/Rank，却只有 1 张 IB 网卡）;
- **代理 Rank 的核心逻辑**：NCCL 在每个节点内选举 1 个（或少数）Rank 作为网络代理，节点内所有 Rank 的跨节点通信流量，都先通过节点内高速通路（NVLink/PCIe） 汇聚到代理 Rank，再由代理 Rank 统一通过网卡与其他节点通信；反向接收时，代理 Rank 先通过网卡接收数据，再通过节点内高速通路分发到目标 Rank；

```c++
/**
 * @brief 网络传输层（NET）发送连接的初始化配置函数
 *
 * 为本地Rank到目标Peer的发送连接完成核心参数配置：
 * 1. 确定使用的网络设备（如网卡）、网络ID及代理Rank（跨节点通信时需代理转发）；
 * 2. 检测并配置GPU Direct RDMA（GDR）是否启用（减少CPU-GPU数据拷贝）；
 * 3. 建立与代理Rank的连接，同步连接配置参数；
 * 4. 填充连接信息结构体（供后续数据发送使用）；
 * 5. 打印连接初始化日志（包含通道、Rank、网络设备、GDR状态等关键信息）。
 *
 * @param comm        NCCL通信器上下文，包含全局配置、拓扑父节点Rank、网络参数等
 * @param graph       拓扑图结构体（存储设备/节点间的网络拓扑关系，可为NULL）
 * @param myInfo      本地Rank的Peer信息（Rank编号、NVML设备ID、节点ID等）
 * @param peerInfo    目标Peer的Rank信息（对端Rank的核心属性）
 * @param connectInfo 输出参数：填充后的连接信息结构体（存储代理Rank、GDR状态等）
 * @param send        发送连接器结构体（绑定当前连接的配置、代理连接等）
 * @param channelId   通道ID（NCCL多通道并行通信，每个通道独立配置）
 * @param connIndex   连接索引（单个Peer的多个并行发送连接，通过索引区分）
 *
 * @return ncclResult_t 成功返回ncclSuccess，失败返回对应NCCL错误码
 */
/* Determine if we will use this transport for this peer and return connect
 * information for this peer */
static ncclResult_t sendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  // 初始化连接设置请求结构体，存储本次连接配置的所有参数（网络设备、GDR、共享缓冲区等）
  struct setupReq req = { 0 };

  /* 配置共享缓冲区标志（shared）：减少内存分配开销，复用缓冲区 */
  // 逻辑说明：
  // - 若存在拓扑图（graph非空）或为第一个连接（connIndex=0）：不使用共享缓冲区（shared=0）
  // - 否则：读取ncclParamNetSharedBuffers参数，若参数不为-2则用参数值，默认1（启用共享）
  send->conn.shared = req.shared = graph || connIndex == 0 ? 0 : ncclParamNetSharedBuffers() != -2 ? ncclParamNetSharedBuffers() : 1;

  // 将通道ID和连接索引填充到请求结构体，供代理通信时标识当前连接
  req.channelId = channelId;
  req.connIndex = connIndex;

  int proxyRank;    // 代理Rank：跨节点通信时，本地节点的代理Rank（负责转发网络数据）
  int64_t netId;    // 网络设备ID：标识使用的物理网卡（如PCIe地址对应的网卡）
  /* 获取网络传输的核心参数：
   * 1. netId：当前连接使用的网络设备ID；
   * 2. req.netDev：网络设备索引（对应ncclNet接口的设备编号）；
   * 3. proxyRank：本次发送连接的代理Rank（本地Rank则无代理，跨节点则为节点内代理）；
   * 核心逻辑：基于拓扑信息选择最优网卡，确定是否需要代理转发
   */
  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, peerInfo->rank, &netId, &req.netDev, &proxyRank));

  /* 检查并配置GPU Direct RDMA（GDR）：
   * GDR允许GPU直接访问网卡DMA缓冲区，无需CPU中转，提升跨节点通信性能；
   * 入参说明：1表示检测发送方向的GDR，&req.useGdr返回是否启用GDR（1=启用，0=禁用）
   */
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->rank, netId, 1, &req.useGdr));

  // 若启用GDR，给发送连接器标记“直接访问网卡”标志（NCCL_DIRECT_NIC）
  send->conn.flags |= req.useGdr ? NCCL_DIRECT_NIC : 0;

  /* 全局GDR状态同步：
   * 若当前连接（第一个连接，connIndex=0）禁用GDR，则整个通信器禁用GDR（保证一致性）
   */
  if (!req.useGdr && connIndex == 0) comm->useGdr = 0;

  /* 启用NetPXN标志：
   * NetPXN是NCCL跨节点网络P2P特性，若代理Rank非本地Rank（跨节点通信），则启用该特性
   */
  if (proxyRank != myInfo->rank && connIndex == 0) comm->useNetPXN = true;

  /* 建立与代理Rank的连接：
   * TRANSPORT_NET：指定传输层类型为网络层；1表示发送方向；
   * &send->proxyConn：填充代理连接结构体，绑定与代理Rank的通信通道
   */
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_NET, 1, proxyRank, &send->proxyConn));

  /* 填充拓扑父节点Rank信息（用于跨节点/多进程通信的层级管理）：
   * - tpLocalRank：本地节点内的拓扑父Rank；
   * - tpRank：本地Rank对应的全局拓扑父Rank；
   * - tpRemoteRank：目标Peer对应的全局拓扑父Rank；
   */
  req.tpLocalRank = comm->topParentLocalRanks[comm->localRank];
  req.tpRank = comm->topParentRanks[myInfo->rank];
  req.tpRemoteRank = comm->topParentRanks[peerInfo->rank];

  /* 向代理Rank发送同步配置请求（阻塞调用）：
   * ncclProxyMsgSetup：代理消息类型（连接设置）；
   * &req：发送的配置参数；sizeof(req)：参数长度；
   * NULL/0：无返回数据；
   * 核心作用：同步连接参数到代理Rank，完成代理侧的连接初始化
   */
  NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgSetup, &req, sizeof(req), NULL, 0));

  /* 打印连接初始化日志（区分是否有代理），便于调试拓扑/网络问题 */
  if (proxyRank == myInfo->rank) {
    // 无代理（本地Rank即为代理，同节点通信）：打印通道、Rank、网络设备、GDR、共享缓冲区状态
    INFO(NCCL_INIT|NCCL_NET,"Channel %02d/%d : %d[%d] -> %d[%d] [send] via NET/%s/%d%s%s%s", channelId, connIndex, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, comm->ncclNet->name, req.netDev,
        req.useGdr ? "/GDRDMA" : "", req.useGdr==ncclTopoGdrModePci ? "(PCI)" : "",
        req.shared ? "/Shared" : "");
  } else {
    // 有代理（跨节点通信）：额外打印代理Rank
    INFO(NCCL_INIT|NCCL_NET,"Channel %02d/%d : %d[%d] -> %d[%d] [send] via NET/%s/%d(%d)%s%s%s", channelId, connIndex, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, comm->ncclNet->name, req.netDev,
        proxyRank,
        req.useGdr ? "/GDRDMA" : "", req.useGdr==ncclTopoGdrModePci ? "(PCI)" : "",
        req.shared ? "/Shared" : "");
  }

  /* 填充连接信息结构体（供上层使用）：
   * 1. 首字节存储代理Rank的拓扑父Rank（标识代理的全局层级）；
   * 2. 偏移ncclNetHandle_t长度的位置存储GDR启用状态（useGdr）；
   * 注：ncclNetHandle_t是网络句柄类型，此处用于对齐内存布局
   */
  *((int*)connectInfo) = comm->topParentRanks[proxyRank];
  memcpy((uint8_t*)connectInfo + sizeof(ncclNetHandle_t), &req.useGdr, sizeof(int));

  // 发送连接初始化配置完成，返回成功
  return ncclSuccess;
}
```



