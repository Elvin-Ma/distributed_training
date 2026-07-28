# 一、Simple 协议的同步模型:生产者-消费者环形 FIFO

## 1. 整体结构

Simple 协议下,每一条连接(sender → receiver)本质是一个**深度为 `NCCL_STEPS`(8)个 step 的环形 FIFO**,由三块内存构成:

- **数据 buffer**(`buffs[NCCL_PROTO_SIMPLE]`):环形数据区,分成 8 个 step slot;
- **tail**(64 位计数器):生产者(sender)已经写完了多少个 step;
- **head**(64 位计数器):消费者(receiver)已经消费完了多少个 step。

不变式:`head <= step <= tail`,FIFO 有空位的条件是 `tail - head < NCCL_STEPS`。

> 注意: 一个 rank 在一个 ring/channel 上通常就是两条有向连接;
>
> 常规 P2P write 路径时:
> buffs[NCCL_PROTO_SIMPLE] 通常在 **receiver** 侧的 ncclRecvMem 后面分配;
> sender 的 send->conn.buffs[SIMPLE] 只是映射到对端 receiver 的这块 buffer;
>
> 常规 P2P read 路径时:
> Simple buffer 会放在 sender 侧的 ncclSendMem 后面，receiver 远程读;

## 2. head/tail 如何做到"对端共享"——连接建立时的指针交叉

这是在 **host 侧建连时**(`transport/p2p.cc`)完成的。每一侧各自分配一块结构:

```cpp
struct ncclConnInfo {
  // Regular comm mechanism
  char *buffs[NCCL_NUM_PROTOCOLS]; // Local for recv, remote for send
  void* mhandles[NCCL_NUM_PROTOCOLS];
  uint64_t *tail;     // Local for recv, remote for send
  uint64_t *head;     // Local for send, remote for recv

  int flags;          // Direct communication / other flags
  int shared;         // Buffers are shared
  int stepSize;       // Step size for the SIMPLE buffer
  void **ptrExchange; // Pointer exchange for direct communication
  uint64_t* redOpArgExchange; // PreOp scaler exchange for direct pull case

  struct ncclConnFifo* connFifo; // Used for GPU - Proxy communication

  uint64_t step;      // Keep where we are
  uint64_t llLastCleaning;
  ncclNetDeviceHandle_t netDeviceHandle;
};

struct ncclSendMem {
  union {
    struct {
      uint64_t head;
      char pad1[CACHE_LINE_SIZE-sizeof(uint64_t)];
      void* ptrExchange;
      uint64_t redOpArgExchange[2];
      char pad2[CACHE_LINE_SIZE-sizeof(void*)-2*sizeof(uint64_t)];
      int offsFifo[NCCL_STEPS];
    };
    char pad3[MEM_ALIGN];
  };
};

struct ncclRecvMem {
  union {
    struct {
      uint64_t tail;
      char pad1[CACHE_LINE_SIZE-sizeof(uint64_t)];
      struct ncclConnFifo connFifo[NCCL_STEPS];
      int flush; // For GDRCopy-based flush
    };
    char pad4[MEM_ALIGN];
  };
};
```

然后通过 **CUDA IPC / cuMemMap(NVLink/PCIe P2P)把对端的这块显存映射进本进程的地址空间**,再做指针交叉:

```cpp
// p2p.cc 建连时(示意)
// sender 视角:
send->conn.head = &sendMem->head;        // 本地内存,自己轮询
send->conn.tail = &remoteRecvMem->tail;  // 对端(receiver)内存,IPC 映射后远程写

// receiver 视角:
recv->conn.tail = &recvMem->tail;        // 本地内存,自己轮询
recv->conn.head = &remoteSendMem->head;  // 对端(sender)内存,远程写
```

所以"共享"不是什么魔法:**head 物理上只有一份,存在 sender 的显存里;tail 物理上也只有一份,存在 receiver 的显存里**。双方通过 P2P 映射都拿到了指向这两个位置的设备指针。

设计上遵循一条经典原则:**自旋读永远读本地显存,写通知永远写远端显存**。这样轮询不产生跨 NVLink 流量,只有真正状态变化时才有一次远程 store:

```
sender:  poll 本地的 head   (receiver 远程写过来)
         store 远端的 tail  (通知 receiver 数据就绪)
receiver: poll 本地的 tail
          store 远端的 head (通知 sender slot 已腾出)
```

## 3. 设备侧代码:角色划分、waitPeer、postPeer

`prims_simple.h` 里把线程按角色分工,每条连接的等待/通知只由**一个线程**负责:

```cpp
// 构造函数中按 tid 分配角色
flags |= RoleWaitRecv / RoleWaitSend / RolePostRecv / RolePostSend / ...;
```

每个负责同步的线程持有两个成员:

```cpp
uint64_t* connStepPtr;   // 指向 head 或 tail(见上面的交叉)
uint64_t  connStepCache; // 对 counter 的本地缓存,减少 volatile load
uint64_t  step;          // 本 primitive 自己推进到的 step 号
```

在 `loadSendConn` / `loadRecvConn` 里绑定:

```cpp
// 发送方向
if (flags & RoleWaitSend) connStepPtr = conn->head;  // 等空位:轮询 head
if (flags & RolePostSend) connStepPtr = conn->tail;  // 发通知:写 tail
// 接收方向
if (flags & RoleWaitRecv) connStepPtr = conn->tail;  // 等数据:轮询 tail
if (flags & RolePostRecv) connStepPtr = conn->head;  // 释放 slot:写 head
```

### waitPeer:自旋等待

```cpp
// prims_simple.h(简化)
template <int DirectRecv, int DirectSend, int Recv, int Send, ...>
__device__ __forceinline__ void waitPeer(...) {
  const bool isSendNotRecv = (Send && Recv) ? (flags & RoleWaitSend) : Send;
  if ((flags & (Recv*RoleWaitRecv | Send*RoleWaitSend)) && ...) {
    int spins = 0;
    // sender: 等 head + NCCL_STEPS >= step + StepPerSlice(FIFO 有空位)
    // receiver: 等 tail            >= step + StepPerSlice(数据已到)
    while (connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) < step + StepPerSlice) {
      connStepCache = loadStepValue(connStepPtr);   // ld.relaxed / ld.volatile
      if (checkAbort(spins)) break;
    }
  }
  ...
  // 等到之后,把本 step 对应的 buffer 地址填进共享内存的 srcs/dsts 数组
  ptrs[index] = connEltsFifo + (step % NCCL_STEPS) * stepSize;  // 常规路径
  step += StepPerSlice;
}
```

注意两点:

- 同一个 `connStepPtr`,sender 和 receiver 比较条件不同:sender 加了 `NCCL_STEPS` 的偏移,即"head 落后我不超过 8 个 step 就还有空位"。
- `connStepCache` 是关键优化:计数器是单调递增的,一旦缓存值满足条件,后续 slice 可以不再去读显存。

### postPeer:通知对端

```cpp
template<int Recv, int Send>
inline __device__ void postPeer(bool dataStored) {
  if (flags & (Recv*RolePostRecv | Send*RolePostSend)) {
    step += StepPerSlice;
    if (Send && (flags & RolePostSend) && dataStored) {
      fence_acq_rel_sys();   // membar.sys:确保数据写入先于 tail 更新对对端可见
    }
    st_relaxed_sys_global(connStepPtr, step);  // st.relaxed.sys 写远端计数器
  }
}
```

**内存序是 Simple 协议的核心开销所在**:

- 发送侧 post tail 前必须有 `fence.acq_rel.sys`(系统级栅栏),保证对端 GPU 观察到 tail 更新时,数据一定已经可见。这个 fence 很贵,所以 Simple 用大 chunk 摊薄它——这正是它小消息延迟差的根源(对比 LL:16B 原子写天然保证数据和 flag 一起到,无需 fence)。
- 接收侧 post head 前不需要 fence(它只是归还 slot,没有数据伴随),所以是纯 relaxed store。

另外,在 `genericOp` 的主循环里,waitPeer 之后、真正搬数据之前有一次 `barrier()`/`subBarrier()`,把"同步线程等到了"这个事实广播给组内所有搬数据的线程;搬完后再 barrier 一次,才允许 post 线程去更新计数器。

### 一次 send/recv 的完整时间线

```
sender 侧线程组                          receiver 侧线程组
──────────────────                       ──────────────────
RoleWaitSend: 自旋读本地 head
  (head+8 >= step+1 → 有空位)
barrier ─────────────────┐
所有线程 reduceCopy:     │
  userbuf → FIFO slot    │
barrier ─────────────────┘
RolePostSend:
  fence.acq_rel.sys
  st.relaxed.sys 远端 tail = step ─────→ RoleWaitRecv: 自旋读本地 tail
                                           (tail >= step+1 → 数据到了)
                                         barrier
                                         所有线程 reduceCopy:
                                           FIFO slot → userbuf(或 reduce)
                                         barrier
        head = step ←──────────────────  RolePostRecv:
  (sender 下一轮 waitSend 看到空位)         st.relaxed.sys 远端 head = step
```

---

# 二、direct 的含义与代码体现

## 1. 语义

上面的常规路径每跳数据要走两次拷贝:`发送方 userbuf → FIFO → 接收方 userbuf`。**direct 的意思是:当两个 GPU 之间 P2P 可直达、且能拿到对端用户缓冲区的地址时,跳过中间 FIFO,直接对对端 user buffer 读/写**,省掉一次拷贝和一半的 buffer 带宽占用。同步仍然靠 head/tail(数据不经过 FIFO,但 step 计数照常推进,起纯信号量作用)。

它有两个方向:

- **DirectWrite**:sender 直接把结果写进 receiver 的目标缓冲区;
- **DirectRead**:receiver 直接从 sender 的缓冲区读。

这就是为什么 `directSend(inpIx, outIx, nelem)` 要传**两个 offset**——`outIx` 就是数据在对端用户缓冲区里的偏移;非 direct 的 `send` 只需要一个。

## 2. 对端用户指针从哪来:ptrExchange / buffer 注册

FIFO 地址建连时就固定了,但用户缓冲区每次调用都不同,所以需要**运行时交换指针**。两条路:

**(a) ptrExchange(p2p 场景)**:建连时在 `ncclSendMem` 里留了一个 `void* ptrExchange` 槽,双方都映射了它。kernel 启动时,接收方角色线程把自己的 buffer 地址写进去,发送方读出来:

```cpp
// prims_simple.h 构造函数中(简化)
if (flags & RolePostRecv) {
  *connPtrsFifoPtr = (void*)recvbuff;   // 把我的接收缓冲区地址暴露给对端
}
if (flags & RoleWaitSend) {
  // 自旋等对端写入
  while ((directBuff = *(void* volatile*)connPtrExchange) == nullptr) { ... }
}
```

**(b) 用户缓冲区注册(NVLink Sharp / 新版 user buffer registration)**:host 侧提前把 sendbuff/recvbuff 注册并映射到所有 peer,`work->regUsed` 置位,设备端直接用预先算好的对端地址,免去 kernel 内握手。

## 3. waitPeer 里的指针选择——direct 的核心分支

direct 与否最终体现在:**waitPeer 往 `srcs/dsts` 数组里填的是 FIFO slot 地址,还是对端 user buffer 地址**:

```cpp
// waitPeer 内(简化自 prims_simple.h)
void **ptrs = isSendNotRecv ? (ncclShmem.groups[group].dsts + Dst)
                            : (ncclShmem.groups[group].srcs + Src);
if (isSendNotRecv && DirectSend) {
  if (flags & DirectWrite) {
    ptrs[index] = directBuff + dstIx * sizeof(T);   // 直接写对端 recvbuff!
  } else if (flags & DirectRead) {
    ptrs[index] = nullptr;                          // 什么都不写,对端会来读
  } else {
    ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*stepSize; // 退回 FIFO
  }
} else if (!isSendNotRecv && DirectRecv) {
  if (flags & DirectRead) {
    ptrs[index] = directBuff + srcIx * sizeof(T);   // 直接读对端 sendbuff
  } else if (flags & DirectWrite) {
    ptrs[index] = directBuff + dstIx * sizeof(T);   // 对端已直接写到我这,从本地目标区"收"
  } else {
    ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*stepSize;
  }
}
```

`DirectWrite/DirectRead` 这两个 flag 是构造时从连接属性判定的:

```cpp
// loadRecvConn / loadSendConn(简化)
if (conn->flags & (NCCL_DIRECT_WRITE | NCCL_P2P_WRITE)) flags |= DirectWrite;
if (conn->flags & (NCCL_DIRECT_READ  | NCCL_P2P_READ )) flags |= DirectRead;
```

## 4. genericOp 里跳过拷贝

指针换掉之后还有最后一步:如果 direct 使得**源地址和目的地址相同**(例如 DirectWrite 场景下,receiver 的"接收源"就是自己的目标缓冲区,数据早被对端写好了),那连 `reduceCopy` 都直接跳过:

```cpp
// genericOp 内(简化)
if (DirectRecv && ncclShmem.groups[group].srcs[0] == ncclShmem.groups[group].dsts[0]) {
  // 数据已经在最终位置,无需搬运;只推进同步计数
  if (Send) { /* 仅在需要继续转发时做拷贝 */ }
} else {
  reduceCopy<...>(tid, nworkers, ..., srcs, dsts, sliceSize);
}
```

此时这一个 step 完全退化为**纯同步操作**:waitPeer 等计数器、postPeer 推计数器,数据零拷贝。

## 5. 回到你贴的 runRing

以 `prims.directRecvReduceDirectSend(offset, offset, nelem)` 为例,在 8 卡 NVLink 全 direct 的理想情况下:

- **recv 端**:不从 FIFO 读——上一跳邻居已经把它的贡献直接写进了我 recvbuff 的 `offset` 处(或我直接读它的 buffer);
- **reduce**:与我本地 sendbuff 的 `offset` 处做归约;
- **send 端**:结果不写 FIFO——直接写进下一跳邻居 recvbuff 的 `offset` 处;
- head/tail 照常一等一推,只承担流控职责。

整条 ring 的数据流变成 GPU 用户缓冲区之间的直接 NVLink store,中转 buffer 完全旁路。而如果某一跳不满足 direct 条件(比如跨网络、未注册),`DirectWrite/DirectRead` flag 不置位,同样这行代码自动退回 FIFO 路径——这就是接口统一叫 `directXxx`、而是否真 direct 由 flags 在运行时决定的设计意图。