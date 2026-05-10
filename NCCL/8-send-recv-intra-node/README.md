# 1 pytorch ProcessGroupNCCL 中ncclSend/ncclRecv 的调用

```c++
  auto ret = pointToPoint(
      tensor,
      [&](at::Tensor& input,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream,
          int dst) {
        auto ncclDataType = getNcclDataType(input.scalar_type());
        return ncclSend(
            input.data_ptr(),
            input.numel(),
            ncclDataType,
            dst,
            comm,
            stream.stream());
      },
      dstRank,
      OpType::SEND,
      c10::str("nccl:send ", rank_, "->", dstRank).c_str());

    auto ret = pointToPoint(
      tensor,
      [&](at::Tensor& output,
          ncclComm_t comm,
          at::cuda::CUDAStream& stream,
          int src) {
        auto ncclDataType = getNcclDataType(output.scalar_type());
        return ncclRecv(
            output.data_ptr(),
            output.numel(),
            ncclDataType,
            src,
            comm,
            stream.stream());
      },
      srcRank,
      OpType::RECV,
      c10::str("nccl:recv ", rank_, "<-", srcRank).c_str());
```

# 2 nccl 侧入口: ncclSend/ncclRecv

## 2.1 ncclSend/ncclRecv
- */pytorch/third_party/nccl/nccl/src/collectives.cc

```c++
ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  NvtxParamsSendRecv payload{count * ncclTypeSize(datatype), peer};
  NVTX3_FUNC_WITH_PARAMS(Send, SendRecvSchema, payload)

  struct ncclInfo info = { ncclFuncSend, "Send",
    NULL, (void*)sendbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };
  ncclResult_t ret;
  NCCLCHECK(ncclGroupStart());
  NCCLCHECKGOTO(ncclEnqueueCheck(&info), ret, exit);
exit:
  NCCLCHECK(ncclGroupEnd());
  return ret;
}

ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  NvtxParamsSendRecv payload{count * ncclTypeSize(datatype), peer};
  NVTX3_FUNC_WITH_PARAMS(Recv, SendRecvSchema, payload)

  struct ncclInfo info = { ncclFuncRecv, "Recv",
    NULL, recvbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };
  ncclResult_t ret;
  NCCLCHECK(ncclGroupStart());
  NCCLCHECKGOTO(ncclEnqueueCheck(&info), ret, exit);
exit:
  NCCLCHECK(ncclGroupEnd());
  return ret;
}
```

## 2.2 ncclGroupStart and ncclGroupEnd

ncclGroupStart只是对ncclGroupMode加一，ncclGroupMode非0表示处于Group操作中，GroupStart和GroupEnd间的操作不会阻塞，最后通过GroupEnd一次性提交操作。<br>

- 嵌套调用只是增加计数，不改变队列记录机制;
- 嵌套结构，最后一层才执行;
- 嵌套结构允许这些不同层次的代码互不干扰地使用 Group 机制；
- 嵌套结构的核心原因: 每层嵌套间有新的通信操作，也要等到最后一个GroupEnd 才执行，可以将最前GroupStart 和 最后 GroupEnd 间所有操作都收集到一个队列中，最后统一执行。避免遇到一个GroupEnd 就停止的情况。

```c++
ncclGroupStart(); // depth=1
  ncclGroupStart(); // depth=2
    ncclSend(...); // 仅记录，不执行
    ncclRecv(...); // 仅记录，不执行
  ncclGroupEnd(); // depth=1，仍不执行
  ncclSend(...); // 仅记录，不执行
ncclGroupEnd(); // depth=0，此时提交所有3个操作
```

```c++
NCCL_API(ncclResult_t, ncclGroupStart);
ncclResult_t ncclGroupStart() {
  ncclResult_t ret = ncclSuccess;
  NVTX3_FUNC_RANGE_IN(nccl_domain);

  NCCLCHECK(ncclGroupStartInternal());
  TRACE_CALL("ncclGroupStart()");
  return ret;
}

NCCL_API(ncclResult_t, ncclGroupEnd);
ncclResult_t ncclGroupEnd() {
  ncclResult_t ret = ncclSuccess;
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  NCCLCHECKGOTO(ncclGroupEndInternal(), ret, exit);
  TRACE_CALL("ncclGroupEnd()");
exit:
  return ret;
}

inline ncclResult_t ncclGroupStartInternal() {
  // 跟踪当前有多少层嵌套的 ncclGroupStart/ncclGroupEnd 调用
  // ncclGroupDepth > 0：表示当前处于分组模式，后续的集体通信操作会被收集而不是立即执行;
  // 当 ncclGroupDepth == 0：表示不在分组模式，集体通信操作会立即执行;
  // 即使内层调用了 ncclGroupEnd，操作也不会执行
  ncclGroupDepth++;
  return ncclSuccess;
}
```

## 2.3 ncclEnqueueCheck
ncclEnqueueCheck 是 NCCL 核心函数之一，作用是在将集合通信操作（如 allreduce、broadcast 等）入队执行前，完成全量的合法性检查、环境准备、日志记录和任务追加，并处理错误和资源恢复。

- **函数核心流程** <br>

1.启动 NCCL 组操作 → 2. 检查通信器有效性 → 3. 检查操作参数合法性 → 4. （可选）切换 CUDA 设备做指针检查 → 5. 记录操作日志 → 6. 追加任务到通信器队列 → 7. 恢复 CUDA 设备、结束组操作 → 8. 检查异步错误并返回。

```c++
/**
 * @brief NCCL集合通信操作入队前的核心检查与准备函数
 * @param info 指向ncclInfo结构体的指针，包含集合操作的所有元信息（通信器、缓冲区、参数、流等）
 * @return ncclResult_t NCCL错误码（ncclSuccess表示成功，其他为具体错误）
 */
ncclResult_t ncclEnqueueCheck(struct ncclInfo* info) {
  // 1. 启动NCCL内部组操作（NCCL组操作支持批量提交多个集合操作）
  // NCCLCHECK：检查NCCL调用返回值，失败则直接抛出/处理错误
  NCCLCHECK(ncclGroupStartInternal());

  // 初始化返回值为“成功”
  ncclResult_t ret = ncclSuccess;
  // 保存原始CUDA设备ID，用于后续恢复（初始值-1表示未切换过设备）
  int devOld = -1;

  // 2. 检查通信器（comm）的基本有效性
  // NCCLCHECKGOTO：带跳转的错误检查宏，失败则设置ret为错误码，跳转到fail标签
  NCCLCHECKGOTO(CommCheck(info->comm, info->opName, "comm"), ret, fail);

  // 检查通信器是否处于“可通信”状态（如是否完成初始化、无未处理错误等）
  NCCLCHECKGOTO(ncclCommEnsureReady(info->comm), ret, fail);

  // 3. 如果通信器开启了指针检查（checkPointers），切换到通信器绑定的CUDA设备
  if (info->comm->checkPointers) {
    // 获取当前活跃的CUDA设备ID并保存
    CUDACHECKGOTO(cudaGetDevice(&devOld), ret, fail);
    // 切换到通信器对应的CUDA设备（指针检查需要在设备上下文内执行）
    CUDACHECKGOTO(cudaSetDevice(info->comm->cudaDev), ret, fail);
  }

  // 4. 检查集合操作参数的合法性（如缓冲区指针、count、数据类型、root节点等）
  NCCLCHECKGOTO(ArgsCheck(info), ret, fail);

  // 5. 打印集合操作的核心日志（INFO级别，NCCL_COLL表示集合通信分类）
  // 日志内容：操作名、操作计数、收发缓冲区、数据量、数据类型、操作类型、根节点、通信器、rank数、CUDA流
  INFO(NCCL_COLL,"%s: opCount %lx sendbuff %p recvbuff %p count %zi datatype %d op %d root %d comm %p [nranks=%d] stream %p",
        info->opName, info->comm->opCount, info->sendbuff, info->recvbuff, info->count,
        info->datatype, info->op, info->root, info->comm, info->comm->nRanks, info->stream);

  // 打印调用追踪日志（更细粒度，用于调试/性能分析）
  TRACE_CALL("nccl%s(%" PRIx64 ",%" PRIx64 ",%zi,%d,%d,%d,%p,%p)",
        info->opName,
        reinterpret_cast<int64_t>(info->sendbuff),  // 发送缓冲区指针（转64位十六进制）
        reinterpret_cast<int64_t>(info->recvbuff),  // 接收缓冲区指针
        info->count, info->datatype, info->op, info->root, info->comm, info->stream);

  // 6. 将当前集合操作任务追加到通信器的任务队列中（准备后续执行）
  NCCLCHECKGOTO(taskAppend(info->comm, info), ret, fail);

// 退出标签：无论成功/失败，最终都会执行此处的资源恢复逻辑
exit:
  // 恢复之前的CUDA设备（如果切换过设备）
  if (devOld != -1) CUDACHECK(cudaSetDevice(devOld));
  // 检查组操作的错误状态，更新返回值ret
  ncclGroupErrCheck(ret);
  // 结束NCCL内部组操作（若组深度为1，会触发组内所有操作的执行）
  NCCLCHECK(ncclGroupEndInternal());

  /*
   * 关键注释：如果组深度为1，ncclGroupEndInternal()会触发组操作执行，通信器状态可能变化
   * 因此需要在此处检查异步错误（非阻塞模式下）
   */
  if (info->comm && !info->comm->config.blocking) {
    // 获取通信器的异步错误，更新返回值ret
    NCCLCHECK(ncclCommGetAsyncError(info->comm, &ret));
  };
  // 返回最终的错误码
  return ret;

// 错误处理标签：所有检查/操作失败时跳转到此处
fail:
  // 若通信器有效且为非阻塞模式，设置通信器的异步错误状态
  if (info->comm && !info->comm->config.blocking) {
    (void) ncclCommSetAsyncError(info->comm, ret); // (void)忽略返回值，避免编译警告
  }
  // 跳转到exit标签执行资源恢复
  goto exit;
}
```

## 2.4 taskAppend

**taskAppend** 是 NCCL 内部核心函数，负责将 ncclInfo 描述的操作（P2P/SendRecv 或集合通信）封装为任务结构体，并追加到通信器的任务队列中；同时处理「单 rank 通信器」「流一致性检查」「P2P 通道预连接」等特殊逻辑。

**核心特性：** <br>
- 区分 P2P（Send/Recv）和集合通信（AllReduce/Broadcast 等）两种任务类型，分别`入队到不同队列`；
- 单 rank 通信器的集合操作直接转为 ncclMemcpyAsync 执行，无需入队；
- 管理通信器用到的 CUDA 流列表，保证组内流的「CUDA 图捕获一致性」；
- 从通信器的「作用域内存（memScoped）」分配任务内存，组结束后自动释放。

```c++
/**
 * @brief 将ncclInfo转换为任务结构体，追加到通信器的任务队列中
 * @note 特殊逻辑：单rank通信器的集合操作直接用ncclMemcpyAsync执行，无需入队
 * @param comm 目标NCCL通信器
 * @param info 包含操作元信息的ncclInfo结构体指针
 * @return ncclResult_t NCCL错误码（ncclSuccess为成功）
 */
static ncclResult_t taskAppend(struct ncclComm* comm, struct ncclInfo* info) {
  // 1. 获取通信器的任务管理核心结构体（包含所有任务队列、流列表、统计信息）
  ncclTasks *tasks = &comm->tasks;

  // 特殊情况1：数据量为0，且不是Send/Recv操作 → 无需处理，直接返回成功
  ...

  // 2. P2P操作分支（Send/Recv）
  if (info->coll == ncclFuncSend || info->coll == ncclFuncRecv) {
    // 目标peer节点（Send/Recv的对端rank，由info->root指定）
    int peer = info->root; // root : peer for p2p operations
    // 计算操作的总字节数（count × 数据类型大小）
    ssize_t nBytes = info->count*ncclTypeSize(info->datatype);
    // 标记当前操作是Send（true）还是Recv（false）
    bool isSendNotRecv = info->coll == ncclFuncSend;

    // 必须先将通信器加入线程局部的组上下文，才能从comm->memScoped分配内存
    ncclGroupCommJoin(info->comm);
    // 从通信器的作用域内存中分配P2P任务结构体（组结束后自动释放）
    struct ncclTaskP2p* p2p = ncclMemoryStackAlloc<struct ncclTaskP2p>(&comm->memScoped);
    // 填充P2P任务参数：缓冲区地址（Recv用recvbuff，Send实际也复用该字段）
    p2p->buff = (void*)info->recvbuff;
    // 任务的总字节数
    p2p->bytes = nBytes;
    // 分片ID（初始为0，P2P大消息会分片传输）
    p2p->chunk = 0;

    // 将P2P任务入队到对端peer的Send/Recv队列
    ncclIntruQueueEnqueue(
      // 选择队列：Send→peer的sendQueue，Recv→peer的recvQueue
      isSendNotRecv ? &tasks->peers[peer].sendQueue : &tasks->peers[peer].recvQueue,
      p2p);
    // 更新P2P任务计数
    tasks->nTasksP2p += 1;

    // 3. 预连接通道（仅当当前rank≠对端peer时需要）
    if (comm->rank != peer) {
      // 计算当前P2P操作对应的通道基准ID
      int channelBaseId;
      NCCLCHECK(ncclChannelComputeBase(comm, peer, info->coll, &channelBaseId));
      // 检查该peer的Send/Recv通道是否已标记为“已处理”
      if (!(isSendNotRecv ? tasks->peers[peer].sendSeen : tasks->peers[peer].recvSeen)) {
        // 标记为已处理，避免重复预连接
        (isSendNotRecv ? tasks->peers[peer].sendSeen : tasks->peers[peer].recvSeen) = true;
        // 遍历通信器配置的“每peer通道数”，逐个检查通道连接状态
        for (int c=0; c < comm->p2pnChannelsPerPeer; c++) {
          int channelId;
          // 从基准ID计算具体的通道ID
          NCCLCHECK(ncclChannelComputeFromBase(comm, channelBaseId, c, &channelId));
          if (isSendNotRecv) {
            // Send通道：检查是否未连接（P2P仅用connector 1）
            if (comm->channels[channelId].peers[peer]->send[1].connected == 0) {
              // 标记需要连接的Send通道，触发预连接
              comm->connectSend[peer] |= (1UL<<channelId);
              ncclGroupCommPreconnect(comm);
            }
          } else {
            // Recv通道：检查是否未连接
            if (comm->channels[channelId].peers[peer]->recv[1].connected == 0) {
              // 标记需要连接的Recv通道，触发预连接
              comm->connectRecv[peer] |= (1UL<<channelId);
              ncclGroupCommPreconnect(comm);
            }
          }
        }
      }
    }
  }
  // 4. 集合通信分支（非Send/Recv：AllReduce/Broadcast/Reduce等）
  else {
    // 拷贝归约操作的完整状态到info→opFull（避免op handle在ncclGroupEnd前被销毁）
    NCCLCHECK(hostToDevRedOp(&info->opFull, info->op, info->datatype, comm));

    // 特殊情况2：单rank通信器 → 无需网络通信，直接执行设备内memcpy
    if (comm->nRanks == 1) {
      // 启动单rank的“伪集合操作”（本质是sendbuff→recvbuff的memcpy）
      NCCLCHECK(ncclLaunchOneRank(info->recvbuff, info->sendbuff, info->count, info->opFull, info->datatype, info->stream));
      return ncclSuccess;
    }
    // 多rank情况：封装为任务入队
    else {
      // 先将通信器加入线程局部组上下文，保证内存分配安全
      ncclGroupCommJoin(info->comm);
      // 从作用域内存分配ncclInfo副本（避免原info被外部修改）
      struct ncclInfo* t = ncclMemoryStackAlloc<struct ncclInfo>(&comm->memScoped);
      // 初始化任务的默认参数：通道数、线程数、算法/协议、是否用户调优
      info->nChannels = 0;
      info->nThreads = 0;
      info->algorithm = NCCL_ALGO_UNDEF;  // 未指定算法（后续由NCCL自动选择）
      info->protocol = NCCL_PROTO_UNDEF;  // 未指定协议（后续自动选择）
      info->userTuned = false;            // 未启用用户自定义调优
      // 拷贝原始info的所有字段到新分配的t中
      memcpy(t, info, sizeof(struct ncclInfo));
      // 将集合通信任务入队到collQueue（带排序，按collCmp规则保证执行顺序）
      ncclIntruQueueSortEnqueue(&tasks->collQueue, t, collCmp);
      // 更新集合通信任务的总字节数统计
      tasks->workBytesTotal += info->count * ncclTypeSize(info->datatype);
      // 更新集合通信任务计数
      tasks->nTasksColl += 1;
    }
  }

  // 5. CUDA流管理：保证组内流的一致性，记录所有用到的流
  // 触发条件：当前流≠最近使用的流，或流列表为空
  if (info->stream != tasks->streamRecent || tasks->streams == nullptr) {
    // 更新“最近使用的流”为当前流
    tasks->streamRecent = info->stream;
    // 遍历流列表，检查当前流是否已存在
    struct ncclCudaStreamList* l = tasks->streams;
    while (true) {
      // 流列表遍历到末尾 → 当前流是新流，需要添加
      if (l == nullptr) {
        // 获取当前流的CUDA图捕获状态（是否被某个CUDA Graph捕获）
        struct ncclCudaGraph graph;
        NCCLCHECK(ncclCudaGetCapturingGraph(&graph, info->stream));
        // 校验：同一组内的流必须“全未捕获”或“全被同一图捕获”（否则报错）
        if (tasks->streams != nullptr && !ncclCudaGraphSame(tasks->capturingGraph, graph)) {
          WARN("Streams given to a communicator within a NCCL group must either be all uncaptured or all captured by the same graph.");
          return ncclInvalidUsage;
        }
        // 更新通信器的CUDA图捕获状态（结构体赋值）
        tasks->capturingGraph = graph;
        // 分配新的流列表节点，加入链表头部
        l = ncclMemoryStackAlloc<struct ncclCudaStreamList>(&comm->memScoped);
        l->stream = info->stream;    // 保存当前流
        l->next = tasks->streams;    // 指向原链表头部
        tasks->streams = l;          // 更新链表头部为新节点
        break;
      }
      // 找到已存在的流 → 无需添加，退出循环
      if (l->stream == info->stream)
        break;
      // 继续遍历下一个节点
      l = l->next;
    }
  }
  // 所有逻辑完成，返回成功
  return ncclSuccess;
}
```

# 3 ncclGroupEndInternal

```c++
用户调用 ncclAllReduce() 等通信原语
  ↓
操作被加入 comm->tasks 队列
  ↓
ncclGroupEnd() 触发执行
  ↓
doLaunches() 开始处理
  ↓
ncclLaunchPrepare() 为每个操作创建 kernel plan
  ↓
[关键点] kernelFn 在此阶段被设置
  ↓
ncclLaunchKernel() 使用 kernelFn 启动 CUDA 内核
```

## 3.1 ncclGroupEndInternal 调度 groupLaunch

这是 NCCL 通信组（Group）机制的**最终收尾与执行入口**，负责：

- 校验通信组的调用合法性（避免无组上下文时调用）；
- 处理组嵌套深度，`仅当嵌套深度归 0 时触发实际执行`；
- 区分「阻塞 / 非阻塞」模式，批量启动组内所有通信算子（ncclComm 对应的任务）；
- 处理组内错误、管理异步任务线程、清理异常场景下的资源。

简单说：ncclGroupStart() 是「开始组定义」，ncclGroupCommJoin() 是「把算子加入组」，ncclGroupEndInternal() 是「结束组定义并真正执行所有算子」。

```c++
ncclResult_t ncclGroupEndInternal() {
  ncclResult_t ret = ncclSuccess;

  // 【步骤1：校验组调用合法性】
  if (ncclGroupDepth == 0) {
    WARN("ncclGroupEnd: not in a group call."); // 无组上下文时调用，抛警告
    ret = ncclInvalidUsage; // 返回「非法使用」错误
    goto exit; // 跳转到出口
  }

  // 【步骤2：处理组嵌套深度】
  // ncclGroupDepth：组嵌套深度（支持 ncclGroupStart() 嵌套调用）
  // 只有深度减到 0 时，才执行后续的组启动逻辑；否则仅减少深度，直接退出
  if ((--ncclGroupDepth) > 0) goto exit;

  // 【步骤3：检查组内累积错误】
  // ncclGroupError：组内所有 comm 加入时累积的错误（如 comm 初始化失败）
  if ((ret = ncclGroupError) != ncclSuccess) goto fail; // 有错误则跳转到失败清理逻辑

  // 【步骤4：判断是否有待执行的组任务】
  // 三个条件满足其一即有任务：
  // 1. ncclGroupCommHead != nullptr：组内有已加入的 comm 链表（核心通信算子）
  // 2. !ncclIntruQueueEmpty(&ncclAsyncJobs)：有异步待执行的 P2P/集合通信任务
  // 3. ncclGroupCommPreconnectHead != nullptr：有预连接（preconnect）的 comm 任务（优化连接建立）
  if (ncclGroupCommHead != nullptr || !ncclIntruQueueEmpty(&ncclAsyncJobs) || ncclGroupCommPreconnectHead != nullptr) {
    // 【步骤5：初始化组任务核心结构体 ncclGroupJobMain】
    // 把组的关键上下文（comm链表、异步任务、错误标记、阻塞模式等）绑定到任务结构体
    ncclGroupJobMain.groupCommHeadPtr = &ncclGroupCommHead; // 组内 comm 链表头指针
    ncclGroupJobMain.groupCommPreconnectHeadPtr = &ncclGroupCommPreconnectHead; // 预连接 comm 链表头
    ncclGroupJobMain.groupErrorPtr = &ncclGroupError; // 组错误标记指针
    ncclGroupJobMain.asyncJobsPtr = &ncclAsyncJobs; // 异步任务队列指针
    ncclGroupJobMain.abortFlagPtr = &ncclGroupJobAbortFlag; // 任务终止标记指针
    ncclGroupJobMain.groupBlockingPtr = &ncclGroupBlocking; // 组阻塞模式指针
    ncclGroupJobMain.initialized = true; // 标记任务结构体已初始化
    ncclGroupJobMainPtr = &ncclGroupJobMain; // 全局指向该任务结构体

    // 【步骤6：校验阻塞模式合法性】
    // ncclGroupBlocking：组阻塞模式（0=非阻塞，1=阻塞），由 ncclGroupCommJoin 统一设置
    assert(ncclGroupBlocking == 0 || ncclGroupBlocking == 1); // 断言保证模式合法

    // 【步骤7：分支1：非阻塞（nonblocking）组执行】
    if (ncclGroupBlocking == 0 && (ncclGroupCommPreconnectHead != nullptr || !ncclIntruQueueEmpty(&ncclAsyncJobs))) {
      /* nonblocking group */
      // 先标记所有异步任务的 comm 状态为「执行中（ncclInProgress）」
      if (!ncclIntruQueueEmpty(&ncclAsyncJobs)) {
        ncclAsyncJob* job = ncclIntruQueueHead(&ncclAsyncJobs);
        do {
          // 为异步任务的 comm 设置「执行中」错误码（异步场景下正常状态）
          NCCLCHECKGOTO(ncclCommSetAsyncError(job->comm, ncclInProgress), ret, fail);
          job->comm->groupJob = ncclGroupJobMainPtr; // 绑定 comm 到当前组任务
          job = job->next;
        } while (job);
      }

      // 标记组内所有 comm 状态为「执行中」，并绑定到当前组任务
      if (ncclGroupCommHead) {
        ncclComm_t comm = ncclGroupCommHead;
        do {
          NCCLCHECKGOTO(ncclCommSetAsyncError(comm, ncclInProgress), ret, fail);
          /* link group job to communicators. */
          comm->groupJob = ncclGroupJobMainPtr; // 关联 comm 与组任务
          comm = comm->groupNext;
        } while (comm);
      }

      // 启动异步线程执行组任务
      // ncclAsyncJobMain：异步任务主函数，内部调用 groupLaunch 执行所有算子
      // pthread_create：创建线程，非阻塞返回（主线程不等待任务完成）
      ncclGroupJobMainPtr->base.func = groupLaunch; // 绑定组执行函数为 groupLaunch
      SYSCHECKGOTO(pthread_create(&ncclGroupJobMainPtr->base.thread, NULL, ncclAsyncJobMain, (void*)&ncclGroupJobMainPtr->base), ret, fail);
      ret = ncclInProgress; // 返回「执行中」状态（非阻塞特性）
    } else {
      // 【步骤8：分支2：阻塞（blocking）组执行】
      /* blocking group */
      // 直接调用 groupLaunch 执行所有算子，主线程阻塞直到所有任务完成
      NCCLCHECKGOTO(groupLaunch(&ncclGroupJobMainPtr->base), ret, fail);
      // 执行完成后重置组任务状态（清理上下文）
      groupResetJobState(ncclGroupJobMainPtr);
    }
  }

// 【出口标签】：正常退出（无错误/仅嵌套深度未归0）
exit:
  return ret;

// 【失败标签】：执行过程中出错，清理组资源
fail:
  // 清理组内的 comm 链表、预连接任务、异步任务，重置错误/阻塞模式/终止标记
  groupCleanup(&ncclGroupCommHead, &ncclGroupCommPreconnectHead, &ncclAsyncJobs, &ncclGroupError, &ncclGroupBlocking, &ncclGroupJobAbortFlag, ret);
  goto exit; // 清理后跳转到出口返回
}
```

- 过程

```sh
调用 ncclGroupEndInternal()
  ↓
校验：是否在组上下文内？→ 否 → 返回非法使用错误
  ↓
减少组嵌套深度 → 深度>0 → 直接返回（嵌套未结束）
  ↓
检查组内是否有累积错误 → 有 → 清理资源并返回错误
  ↓
判断组内是否有待执行任务 → 无 → 直接返回成功
  ↓
初始化组任务结构体，绑定所有上下文
  ↓
根据阻塞模式分支执行：
  ├─ 非阻塞 → 创建线程异步执行 groupLaunch，返回「执行中」
  └─ 阻塞 → 主线程同步执行 groupLaunch，完成后重置状态
  ↓
正常返回 / 出错则清理资源后返回
```

## 3.2 子线程启动: 执行 groupLaunch

```c++
SYSCHECKGOTO(pthread_create(&ncclGroupJobMainPtr->base.thread, NULL, ncclAsyncJobMain, (void*)&ncclGroupJobMainPtr->base), ret, fail);
```

- 线程中运行的是 ncclAsyncJobMain 这个函数, 最终执行的是里面的job->func 函数， func 就是 groupLaunch

- func 从 ncclAsyncJob 获取，同时 ncclAsyncJob 也是 func 的参数

```c++
void* ncclAsyncJobMain(void* arg) {
  struct ncclAsyncJob* job = (struct ncclAsyncJob*)arg;
  job->result = job->func(job);
  if (job->result != ncclSuccess) {
    INFO(NCCL_INIT,"%s:%d -> %d [Async thread]", __FILE__, __LINE__, job->result);
  }
  __atomic_store_n(&job->state, ncclGroupJobDone, __ATOMIC_RELEASE);
  return arg;
}
```

- 参数从ncclGroupJob.base中获取

```c++
struct ncclGroupJob {
  struct ncclAsyncJob base;
  struct ncclComm **groupCommHeadPtr;
  struct ncclComm **groupCommPreconnectHeadPtr;
  ncclResult_t *groupErrorPtr;
  volatile bool *abortFlagPtr;
  int *groupBlockingPtr;
  struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> *asyncJobsPtr;
  bool initialized;
};

struct ncclAsyncJob {
  struct ncclAsyncJob* next;
  pthread_t thread;
  ncclResult_t result;
  ncclResult_t(*func)(struct ncclAsyncJob*);
  void(*undo)(struct ncclAsyncJob*);
  void(*destructor)(void*);
  ncclGroupJobState_t state;
  volatile uint32_t *abortFlag; /* point to comm abortFlag */
  volatile uint32_t *childAbortFlag; /* point to child abortFlag */
  ncclComm_t comm;
};
```

**func 从 ncclAsyncJob 获取，同时 ncclAsyncJob 也是 func 的参数**

## 3.3 groupLanch 到 doLaunch

- 中间有个重要的 : ncclTransportP2pSetup 过程.

- groupLaunch: 是 NCCL 中真正负责批量调度 / 启动组内所有通信 Kernel 的函数（核心逻辑）；

- groupLaunch 的参数是 **ncclAsyncJob**;

- **groupLaunch 的核心职责(Kernel 调度的关键)**
  - groupLaunch 是 NCCL 中分组异步任务（GroupJob）的核心启动函数，负责：
  - 预处理 “预连接（Preconnect）” 通信任务并加入异步队列；
  - 为异步任务队列中的每个任务创建线程执行；
  - 监控所有异步任务的状态（完成 / 失败 / 终止），回收线程资源；
  - 执行分组通信的核心启动逻辑；
  - 清理任务资源、恢复 CUDA 设备上下文，并处理错误回滚.

```c++
/**
 * @brief NCCL分组异步任务的核心启动函数
 * @param job_ 通用异步任务指针，实际指向struct ncclGroupJob类型
 * @return ncclResult_t NCCL错误码（ncclSuccess表示成功，其他为具体错误）
 */
static ncclResult_t groupLaunch(struct ncclAsyncJob *job_) {
  // 保存当前CUDA设备ID，函数结束后恢复
  int savedDev;
  // 初始化返回值为成功
  ncclResult_t ret = ncclSuccess;
  // 标记所有异步任务是否完成
  bool jobsDone = false;
  // 标记是否因任务错误需要终止
  bool errorJobAbortFlag = false;
  // 将通用异步任务指针强转为分组任务（GroupJob）指针（核心类型转换）
  // ncclGroupJob 的第一个成员必然是 ncclAsyncJob 类型的基类字段，
  // 调用方保证传入的 job_ 本质就是 ncclGroupJob 实例
  // (void*)&ncclGroupJobMainPtr->base 是 ncclAsyncJob, 可以直接对指针强转变为ncclGroupJob
  struct ncclGroupJob *gjob = (struct ncclGroupJob*) job_;

  // 从GroupJob中提取核心上下文（均为指针解引用，获取实际的通信/队列/终止标志）
  // 分组通信链表头（主通信链，管理GPU间核心通信）
  struct ncclComm *groupCommHeadMain = *gjob->groupCommHeadPtr;
  // 分组预连接通信链表头（提前建立通信连接，减少通信延迟）
  struct ncclComm *groupCommPreconnectHeadMain = *gjob->groupCommPreconnectHeadPtr;
  // 异步任务队列（侵入式队列，存储待执行的异步任务）
  struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> *asyncJobsMain = gjob->asyncJobsPtr;
  // 分组终止标志（volatile保证多线程可见，原子操作修改）
  volatile bool *groupAbortFlag = gjob->abortFlagPtr;

  // ========== 步骤1：保存当前CUDA设备上下文 ==========
  // CUDACHECKGOTO：NCCL封装的CUDA调用错误检查宏，失败则赋值ret并跳转到fail标签
  // 保存当前激活的CUDA设备ID到savedDev，后续恢复
  CUDACHECKGOTO(cudaGetDevice(&savedDev), ret, fail);

  // ========== 步骤2：处理预连接（Preconnect）任务 ==========
  // 预连接的作用：提前建立GPU间通信连接，避免通信时动态建立连接的延迟
  if (groupCommPreconnectHeadMain != nullptr) {
    struct ncclComm* comm = groupCommPreconnectHeadMain;
    // 遍历预连接通信链表
    do {
      // 为预连接任务分配内存
      struct ncclPreconnectJob* job;
      // NCCLCHECKGOTO：NCCL内存分配/内部调用错误检查宏，失败跳转fail
      NCCLCHECKGOTO(ncclCalloc(&job, 1), ret, fail);

      // 初始化预连接任务的核心字段（继承自ncclAsyncJob基类）
      job->base.func = ncclPreconnectFunc;    // 任务执行函数：预连接核心逻辑
      job->base.undo = nullptr;               // 无回滚函数
      job->base.destructor = free;            // 任务销毁函数：释放内存
      job->base.state = ncclGroupJobRunning;  // 任务状态：运行中
      job->base.abortFlag = comm->abortFlag;  // 绑定当前通信的终止标志
      job->comm = comm;                       // 绑定当前通信对象

      // 将预连接任务加入异步任务队列（等待后续启动线程执行）
      ncclIntruQueueEnqueue(asyncJobsMain, &job->base);

      // 安全遍历链表：先保存下一个节点，再标记当前节点已处理（避免重复处理）
      struct ncclComm* next = comm->preconnectNext;
      comm->preconnectNext = reinterpret_cast<struct ncclComm*>(0x1); // 标记为已处理
      comm = next;
    } while (comm != nullptr);
  }

  // ========== 步骤3：启动异步任务队列中的所有任务（创建线程执行） ==========
  if (!ncclIntruQueueEmpty(asyncJobsMain)) {
    struct ncclAsyncJob* job = ncclIntruQueueHead(asyncJobsMain);
    // 遍历异步任务队列，为每个任务创建POSIX线程
    do {
      // SYSCHECKGOTO：系统调用错误检查宏，失败跳转fail
      // 创建线程执行ncclAsyncJobMain（异步任务核心执行函数），线程ID存入job->thread
      SYSCHECKGOTO(pthread_create(&job->thread, nullptr, ncclAsyncJobMain, job), ret, fail);
      job = job->next; // 下一个任务
    } while (job != nullptr);

    // ========== 步骤4：监控所有异步任务的状态（等待完成/处理错误/终止） ==========
    do {
      jobsDone = true; // 先假设所有任务完成，遍历后修正
      job = ncclIntruQueueHead(asyncJobsMain);
      do {
        // 原子加载任务状态（ACQUIRE内存序：保证后续读操作可见任务状态的修改）
        ncclGroupJobState_t state = __atomic_load_n(&job->state, __ATOMIC_ACQUIRE);

        // 状态1：任务仍在运行 → 标记未完成
        if (state == ncclGroupJobRunning) {
          jobsDone = false;
        }
        // 状态2：任务已完成 → 回收线程、检查任务结果
        else if (state == ncclGroupJobDone) {
          // 等待线程退出（回收资源），失败则打印警告并标记系统错误
          if (pthread_join(job->thread, nullptr) != 0) {
            WARN("Error waiting for pthread_join : %s", strerror(errno));
            ret = ncclSystemError;
          }
          // 标记线程已回收（避免重复join）
          job->state = ncclGroupJobJoined;
          // 若任务执行失败且全局返回值仍为成功，则更新返回值并标记需要终止
          if (job->result != ncclSuccess && ret == ncclSuccess) {
            ret = job->result;
            errorJobAbortFlag = true;
          }
        }
        // 状态3：其他（安全检查）→ 必须是已回收状态，否则断言失败
        else {
          assert(state == ncclGroupJobJoined);
        }

        // ========== 处理终止信号：转发分组终止/错误终止标志 ==========
        // 若分组被终止 或 任务执行错误需要终止 → 触发任务的终止标志
        if (__atomic_load_n(groupAbortFlag, __ATOMIC_RELAXED) || errorJobAbortFlag == true) {
          // 原子设置当前任务的终止标志（RELAXED内存序：仅保证值修改，无内存屏障）
          __atomic_store_n(job->abortFlag, 1, __ATOMIC_RELAXED);
          // 若有子任务终止标志，也同步设置
          if (job->childAbortFlag) __atomic_store_n(job->childAbortFlag, 1, __ATOMIC_RELAXED);
        }

        job = job->next; // 下一个任务
      } while (job != nullptr);

      // 若任务未完成，休眠1微秒（让出CPU，避免忙等），让预连接线程有时间执行
      if (jobsDone == false) usleep(1);
    } while (jobsDone == false); // 循环直到所有任务完成

    // 若异步任务执行失败，跳转到fail标签处理错误
    if (ret != ncclSuccess) goto fail;
  }

  // ========== 步骤5：执行分组通信的核心启动逻辑 ==========
  // 若分组通信链表非空，调用doLaunches执行实际的GPU通信逻辑（如allreduce/allgather等）
  if (groupCommHeadMain != nullptr) {
    NCCLCHECKGOTO(doLaunches(groupCommHeadMain), ret, fail);
  }

  // ========== 步骤6：清理异步任务队列 ==========
  // 遍历并销毁所有异步任务（释放内存、设置异步错误状态）
  while (!ncclIntruQueueEmpty(asyncJobsMain)) {
    struct ncclAsyncJob* job = ncclIntruQueueDequeue(asyncJobsMain);
    // 若任务绑定了通信对象且非阻塞模式，设置通信的异步错误状态
    if (job->comm && !job->comm->config.blocking)
      (void) ncclCommSetAsyncError(job->comm, ret);
    // 调用任务的销毁函数（如free）释放内存
    if (job->destructor) job->destructor((void*)job);
  }

  // ========== 步骤7：清理分组通信链表 ==========
  // 遍历通信链表，让每个通信对象退出分组，并设置异步错误状态
  while (groupCommHeadMain != nullptr) {
    struct ncclComm* comm = groupCommHeadMain;
    struct ncclComm* next = comm->groupNext;
    // 让通信对象离开分组（清理分组关联的状态）
    (void) ncclGroupCommLeave(comm);
    // 非阻塞模式下，设置通信的异步错误状态
    if (!comm->config.blocking) {
      (void) ncclCommSetAsyncError(comm, ret);
    }
    groupCommHeadMain = next;
  }

  // ========== 步骤8：恢复CUDA设备上下文 ==========
  // 无需错误跳转，仅恢复设备（失败不影响核心逻辑）
  CUDACHECK(cudaSetDevice(savedDev));

// ========== 正常退出点 ==========
exit:
  return ret;

// ========== 错误处理点 ==========
fail:
  // 分组清理：释放所有关联的通信/队列/终止标志资源，处理错误回滚
  groupCleanup(gjob->groupCommHeadPtr, gjob->groupCommPreconnectHeadPtr, gjob->asyncJobsPtr, gjob->groupErrorPtr, gjob->groupBlockingPtr, gjob->abortFlagPtr, ret);
  // 跳转到正常退出点返回错误码
  goto exit;
}
```

## 3.4 ncclPreconnectFunc : 预连接核心逻辑

- pthread_create 时执行: **ncclTransportP2pSetup**

```c++
 ncclResult_t ncclPreconnectFunc(struct ncclAsyncJob* job_) {
  struct ncclPreconnectJob* job = (struct ncclPreconnectJob*)job_;
  struct ncclComm* comm = job->comm;
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  if (CPU_COUNT(&comm->cpuAffinity)) sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);
  NCCLCHECK(ncclTransportP2pSetup(comm, NULL, 1));
  return ncclSuccess;
}
```

## 3.5 do_launch 到 ncclLaunchKernel

- **ncclLaunchPrepare 完成预连接**

- **ncclLaunchKernel(comm, plan)**

```c++
static ncclResult_t doLaunches(struct ncclComm* head) {
  ncclResult_t result = ncclSuccess;
  struct ncclComm* cliqueComm0 = head->intraComm0;
  struct ncclComm* cliqueHead = head;
  struct ncclComm* cliqueNextHead;
  bool useBarrier = ncclParamLaunchMode == ncclLaunchModeGroup;
  // This outer loop iterates over cliques of comms which are siblings of the
  // same global entity. We calculate a clique as all comms which have the same
  // `intraComm0` value.
  do {
    struct ncclComm* comm = cliqueHead;
    bool capturingYes = false, capturingNo = false;
    do {
      (ncclCudaGraphValid(comm->tasks.capturingGraph) ? capturingYes : capturingNo) = true;
      CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), result, failure);
      NCCLCHECKGOTO(ncclLaunchPrepare(comm), result, failure);
      if (useBarrier) ncclCommIntraBarrierIn(comm, 1);
      comm = comm->groupNext;
    } while (comm != nullptr && comm->intraComm0 == cliqueComm0);
    cliqueNextHead = comm;

    if (capturingYes && capturingNo) {
      // We have entered barriers but are aborting without leaving them. Thus
      // these comms are permanently trashed. We need a good mechanism for
      // tracking and reporting that.
      WARN("Either none or all communicators in a ncclGroup() can be CUDA graph captured.");
      result = ncclInvalidUsage;
      goto failure;
    }

    while (true) { // Iterate rounds of launches for clique.
      bool moreRounds = false;
      comm = cliqueHead;
      do { // Iterate clique members.
        struct ncclComm* next = comm->groupNext;
        if (useBarrier) {
          // Barrier reduction result tells us if this was the final round.
          moreRounds = 0 != ncclCommIntraBarrierOut(comm);
        } else {
          moreRounds |= comm->unlaunchedPlansHead != nullptr;
        }
        if (moreRounds) {
          // Pop next unlaunched kernel
          struct ncclKernelPlan* plan = comm->unlaunchedPlansHead;
          if (plan != nullptr) {
            comm->unlaunchedPlansHead = plan->next;
            CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), result, failure);
            NCCLCHECKGOTO(ncclLaunchKernelBefore_NoUncapturedCuda(comm, plan), result, failure);
            NCCLCHECKGOTO(ncclLaunchKernel(comm, plan), result, failure);
          }
          // Barrier reduction input indicates if we require further rounds.
          if (useBarrier) ncclCommIntraBarrierIn(comm, comm->unlaunchedPlansHead != nullptr ? 1 : 0);
          if (plan != nullptr) {
            NCCLCHECKGOTO(ncclLaunchKernelAfter_NoCuda(comm, plan), result, failure);
          }
        } else { // Final round.
          CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), result, failure);
          NCCLCHECKGOTO(ncclLaunchFinish(comm), result, failure);
        }
        comm = next;
      } while (comm != cliqueNextHead);
      if (!moreRounds) break;
    }
    cliqueHead = cliqueNextHead;
  } while (cliqueHead != nullptr);
failure:
  return result;
}
```

## 3.6 从 ncclLaunchKernel 到 cudaLaunchKernel

```c++
ncclResult_t ncclLaunchKernel(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  struct ncclTasks* tasks = &comm->tasks;
  void *fn = plan->kernelFn;
  cudaStream_t launchStream = tasks->streams->stream;
  dim3 grid = {(unsigned)plan->channelCount, 1, 1};
  dim3 block = {(unsigned)plan->threadPerBlock, 1, 1};
  size_t smem = ncclShmemDynamicSize(comm->cudaArch);
  void *args[3] = {&comm->devComm, &plan->channelMask, &plan->workHead};

  #if CUDART_VERSION >= 11080
  int driverVersion;
  NCCLCHECK(ncclCudaDriverVersion(&driverVersion));
  if (driverVersion >= 11080) {
    int compCap = comm->compCap;
    unsigned int clusterSize = (compCap == 90) ? comm->config.cgaClusterSize : 0;

    cudaLaunchConfig_t launchConfig = {0};
    cudaLaunchAttribute launchAttrs[3];
    int attrs = 0;
    /* Cooperative Group Array (CGA)
     * On sm90 and later we have an extra level of hierarchy where we
     * can group together several blocks within the Grid, called
     * Thread Block Clusters.
     * Clusters enable multiple thread blocks running concurrently
     * across multiple SMs to synchronize and collaboratively fetch
     * and exchange data. A cluster of blocks are guaranteed to be
     * concurrently scheduled onto a group of SMs.
     * The maximum value is 8 and it must be divisible into the grid dimensions
     */
    if (clusterSize) {
      // Grid dimension must be divisible by clusterSize
      if (grid.x % clusterSize) clusterSize = 1;
      launchAttrs[attrs].id = cudaLaunchAttributeClusterDimension;
      launchAttrs[attrs++].val.clusterDim = {clusterSize, 1, 1};
      launchAttrs[attrs].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
      launchAttrs[attrs++].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicySpread;
    }
    #if CUDART_VERSION >= 12000
    if (compCap >= 90 && driverVersion >= 12000) {
      // Set the NCCL Mem Sync domain on CUDA 12.0 and later (sm90)
      launchAttrs[attrs].id = cudaLaunchAttributeMemSyncDomain;
      launchAttrs[attrs++].val.memSyncDomain = (cudaLaunchMemSyncDomain) ncclParamMemSyncDomain();
    }
    #endif
    launchConfig.gridDim = grid;
    launchConfig.blockDim = block;
    launchConfig.dynamicSmemBytes = smem;
    launchConfig.attrs = launchAttrs;
    launchConfig.numAttrs = attrs;
    launchConfig.stream = launchStream;

    CUDACHECK(cudaLaunchKernelExC(&launchConfig, fn, args));
    return ncclSuccess;
  }
  #endif
  // Standard kernel launch
  CUDACHECK(cudaLaunchKernel(fn, grid, block, args, smem, launchStream));
  return ncclSuccess;
}
```

# 4 kernel select : 从 ncclLaunchPrepare 开始

## 4.1 ncclLaunchPrepare

为待调度的集合通信/点对点任务构建：<br>
- 内核执行计划；
- 分配资源；
- 建立CUDA流依赖

核心执行流程：<br>
1. 资源清理与内存准备：先轮询回调函数释放内存池资源，再压入内存栈用于分配任务结构体；
2. 内核计划（KernelPlan）构建：循环为待执行任务（coll/P2P）分配KernelPlan，优先调度集合通信任务（避免rank间逻辑分歧），直到任务队列清空；
3. 流依赖构建：建立deviceStream/hostStream/userStream之间的依赖关系，保证任务执行顺序的正确性；
4. 持久化处理：针对CUDA Graph持久化场景，处理host任务启动、持久化引用计数及析构函数注册；
5. 异常处理：若过程中出错，回滚内存栈分配的资源。


```c++
/**
 * @brief NCCL通信任务执行前的核心准备函数：为待调度的集合通信/点对点任务构建内核执行计划、分配资源、建立CUDA流依赖
 *
 * 该函数是NCCL将通信任务（coll/P2P）提交到GPU执行前的关键前置步骤，核心职责是把待执行的任务队列拆分为可执行的内核计划（KernelPlan），
 * 处理CUDA Graph持久化逻辑、分配内存资源、构建CUDA流之间的依赖关系，确保任务能按正确顺序、高效且安全地在GPU上执行。
 *
 * @param[in,out] comm NCCL通信上下文指针，包含通信组的所有状态（任务队列、内存池、CUDA流、持久化配置等），函数会修改该结构体的任务/计划相关字段
 * @return ncclResult_t 操作返回码：ncclSuccess表示准备成功；其他值（如ncclInvalidUsage）表示失败（如队列深度不足）
 *
 * @details 核心执行流程：
 * 1. 资源清理与内存准备：先轮询回调函数释放内存池资源，再压入内存栈用于分配任务结构体；
 * 2. 内核计划（KernelPlan）构建：循环为待执行任务（coll/P2P）分配KernelPlan，优先调度集合通信任务（避免rank间逻辑分歧），直到任务队列清空；
 * 3. 流依赖构建：建立deviceStream/hostStream/userStream之间的依赖关系，保证任务执行顺序的正确性；
 * 4. 持久化处理：针对CUDA Graph持久化场景，处理host任务启动、持久化引用计数及析构函数注册；
 * 5. 异常处理：若过程中出错，回滚内存栈分配的资源。
 *
 * @note 关键设计点说明：
 * - persistent（持久化）：由ncclCudaGraphValid判断是否处于CUDA Graph捕获模式，持久化模式下KernelPlan无任务数限制，非持久化模式限制为队列深度的1/2；
 * - 任务调度优先级：优先排空集合通信（coll）任务，再处理点对点（P2P）任务——避免先处理P2P导致不同rank的内核切割点不一致，引发通道选择逻辑分歧；
 * - 流依赖逻辑：通过两级扇入扇出（fan-in/fan-out）构建流依赖，满足ncclStrongStreamWaitStream对强流的要求，保证任务执行顺序；
 * - 队列深度检查：若任务无法填入当前KernelPlan，触发队列深度过小的告警，避免无限循环。
 *
 * @warning 该函数是NCCL内部核心函数，外部不应直接调用；需保证comm上下文已完成初始化，且任务队列（tasks->nTasksColl/nTasksP2p）状态合法。
 */
ncclResult_t ncclLaunchPrepare(struct ncclComm* comm) {
  // 初始化返回码为成功
  ncclResult_t result = ncclSuccess;
  // 取出通信上下文中的待执行任务队列
  struct ncclTasks* tasks = &comm->tasks;
  // 判断是否处于CUDA Graph捕获模式（持久化任务模式）
  bool persistent = ncclCudaGraphValid(tasks->capturingGraph);
  // 统计本次构建的内核计划数
  int nPlans = 0;

  // Poll for callbacks sent to us from other threads. Typically these free
  // resources from to our memory pools.
  // 轮询其他线程发送的回调函数（主要用于释放内存池资源），非阻塞模式
  NCCLCHECK(ncclCommPollCallbacks(comm, /*waitSome=*/false));

  // We already have one frame present which holds all of our tasks (which we
  // are about to schedule). Now push an additional frame for allocating
  // work structs (see appendWorkElem() variants all use scoped allocation).
  // 压入新的内存栈帧，用于分配任务结构体（appendWorkElem等函数依赖该作用域分配）
  ncclMemoryStackPush(&comm->memScoped);

  // 若存在待执行的集合通信或点对点任务，则开始构建内核计划
  if (tasks->nTasksColl + tasks->nTasksP2p != 0) {
    do {
      // 从内存池分配内核计划结构体（持久化内存域）
      struct ncclKernelPlan* plan = ncclMemoryPoolAlloc<struct ncclKernelPlan>(&comm->memPool_ncclKernelPlan, &comm->memPermanent);
      // 将新计划加入通信上下文的计划队列
      ncclIntruQueueEnqueue(&comm->planQueue, plan);
      nPlans += 1;
      // 关联计划与通信上下文
      plan->comm = comm;
      // 设置计划的回收函数
      plan->reclaimer.fn = reclaimPlan;
      // 标记计划是否为持久化模式（CUDA Graph）
      plan->persistent = persistent;

      // Non-persistent kernels fill up at most half of our fifo per kernel.
      // 非持久化模式下，每个内核最多使用FIFO队列的一半深度（避免队列溢出）；持久化模式无限制
      int nWorkBudget = plan->persistent ? INT_MAX : comm->workFifoDepth/2;
      int nWorkBudgetOld = nWorkBudget;

      // Drain coll tasks first. This is essential since we partition tasks based
      // on the work budget and p2p work isn't collective. If we were to drain p2p
      // first, the place where we cut the kernel could vary by rank which would
      // cause the "shortest channel first" channel picker to have divergent results.
      // 优先调度集合通信任务（核心：避免不同rank的内核切割点不一致，导致通道选择逻辑分歧）
      if (tasks->nTasksColl != 0) {
        NCCLCHECKGOTO(scheduleCollTasksToPlan(comm, plan, &nWorkBudget), result, failure);
      }
      // And only drain p2p tasks once colls are depleted.
      // 仅当集合通信任务排空后，再调度点对点任务
      if (tasks->nTasksColl == 0 && tasks->nTasksP2p != 0) {
        NCCLCHECKGOTO(scheduleP2pTasksToPlan(comm, plan, &nWorkBudget), result, failure);
      }
      // 若本次循环未调度任何任务（预算未消耗），说明队列深度过小，触发告警并返回错误
      if (nWorkBudget == nWorkBudgetOld) {
        // We weren't able to fit any tasks into our budget which means now we're
        // stuck in an infinite loop. We defer this check until here, instead of
        // doing it in comm init, to permit testing with insanely shallow queues
        // for cases where that's expected to still work (e.g. few channels).
        WARN("'NCCL_WORK_FIFO_DEPTH=%d' is too small. Minimum value is %d", comm->workFifoDepth, 2*MAXCHANNELS);
        result = ncclInvalidUsage;
        goto failure;
      }
      // 完成当前内核计划的初始化
      finishPlan(plan);
    } while (tasks->nTasksColl + tasks->nTasksP2p != 0); // 直到所有待执行任务都被分配到计划中

    // 取出计划队列的头节点，标记为待启动的计划头
    struct ncclKernelPlan* planHead = ncclIntruQueueHead(&comm->planQueue);
    comm->unlaunchedPlansHead = planHead;

    // Semantically we want these dependencies for the kernels launched:
    //   1. Launch host task on hostStream.
    //   2. Launch kernel, depends on all of {deviceStream, hostStream, userStream[i]...}
    //   3. {deviceStream, userStream[i]...} depend on kernel.
    // We achieve this by:
    //   1. userStream[0] waits on deviceStream
    //   2. deviceStream waits on each of userStream[1...]
    //   3. host task launch on hostStream
    //   4. userStream[0] waits on hostStream
    //   5. kernel launch on userStream[0]
    //   6. deviceStream waits on userStream[0]
    //   7. userStream[1...] each waits on deviceStream
    // The two-level fan-in fan-out is because ncclStrongStreamWaitStream() requires
    // at least one of the two streams to be strong-stream.
    // 核心：构建CUDA流依赖关系，保证任务执行顺序的正确性（两级扇入扇出适配强流要求）
    cudaStream_t launchStream = tasks->streams->stream;
    // 获取设备流（deviceStream）的强流句柄
    NCCLCHECKGOTO(ncclStrongStreamAcquire(tasks->capturingGraph, &comm->sharedRes->deviceStream), result, failure);

    // Create dependency for device stream on user streams. First from extra user
    // streams to deviceStream. Then deviceStream to first user stream.
    // 构建设备流与用户流的依赖：先让deviceStream等待所有额外用户流，再让主用户流等待deviceStream
    for (struct ncclCudaStreamList* l=tasks->streams->next; l != nullptr; l = l->next) {
      NCCLCHECKGOTO(ncclStrongStreamWaitStream(tasks->capturingGraph, &comm->sharedRes->deviceStream, l->stream), result, failure);
    }
    NCCLCHECKGOTO(ncclStrongStreamWaitStream(tasks->capturingGraph, launchStream, &comm->sharedRes->deviceStream), result, failure);

    // 处理需要启动host任务的场景（持久化/阻塞启动/存在代理操作），host任务用于推送代理参数（高性能成本，仅必要时启用）
    if (persistent || comm->persistentRefs != 0 || ncclCudaLaunchBlocking) {
      bool acquired = false;
      // 遍历所有计划，检查是否有代理操作需要host任务处理
      for (struct ncclKernelPlan* plan=planHead; plan != nullptr; plan = plan->next) {
        if (plan->hasProxyOps) {
          if (!acquired) {
            acquired = true;
            // 获取hostStream的强流句柄
            NCCLCHECKGOTO(ncclStrongStreamAcquire(tasks->capturingGraph, &comm->sharedRes->hostStream), result, failure);
          }
          // 在hostStream上启动host任务，回调函数处理计划参数
          NCCLCHECKGOTO(ncclStrongStreamLaunchHost(tasks->capturingGraph, &comm->sharedRes->hostStream, hostStreamPlanCallback, plan), result, failure);
        }
      }
      if (acquired) {
        // 让待启动的内核依赖于hostStream的任务完成
        NCCLCHECKGOTO(ncclStrongStreamWaitStream(tasks->capturingGraph, launchStream, &comm->sharedRes->hostStream), result, failure);
        // 释放hostStream的强流句柄
        NCCLCHECKGOTO(ncclStrongStreamRelease(tasks->capturingGraph, &comm->sharedRes->hostStream), result, failure);
      }
    }

    // 持久化模式：更新持久化引用计数，并为CUDA Graph注册析构函数（回收计划资源）
    if (persistent) {
      comm->persistentRefs += nPlans;
      NCCLCHECKGOTO(ncclCudaGraphAddDestructor(tasks->capturingGraph, persistentDestructor, (void*)planHead), result, failure);
    }
  }

  // 异常处理分支：回滚内存栈，释放本次分配的任务结构体
  if (false) {
  failure:
    ncclMemoryStackPop(&comm->memScoped); // deallocate ncclWork's
  }
  // 返回最终执行结果
  return result;
}
```

## 4.2 scheduleP2pTasksToPlan

**plan->kernelFn = ncclDevKernelForFunc[ncclDevFuncId_P2p()];**

```c++
/**
 * @brief 将NCCL通信上下文中的点对点（P2P）任务调度到指定内核计划（KernelPlan）中，完成任务分块、通道适配与预算控制
 *
 * 该函数是NCCL P2P通信任务执行的核心调度逻辑，负责将待执行的P2P发送/接收任务按通道数、数据分块规则拆分，填充到内核计划中；
 * 适配跨节点/节点内的不同分块策略，校验自通信任务的合法性，同时严格控制内核计划的工作预算（避免超限）。
 *
 * @param[in,out] comm NCCL通信上下文指针，包含P2P任务队列、通道配置、节点数、分块大小等核心配置
 * @param[in,out] plan 待填充的内核计划（KernelPlan），函数会将拆分后的P2P任务添加到该计划中，同时更新内核函数/线程配置
 * @param[in,out] nWorkBudget 输入输出参数：内核计划的剩余工作预算（可容纳的任务数）；每添加一个P2P任务会消耗1个预算，预算不足时提前返回
 * @return ncclResult_t 操作返回码：
 *         - ncclSuccess：调度成功（所有P2P任务填充完成，或预算不足时提前返回）；
 *         - ncclInternalError：自通信任务的发送/接收计划未对齐；
 *         - ncclInvalidUsage：自通信任务仅有发送/接收，无匹配的另一端。
 *
 * @details 核心执行流程：
 * 1. 初始化配置：提取P2P任务关键参数（节点数、发送/接收顺序、通道数），配置内核函数和线程块大小；
 * 2. 通道数适配：根据总通道数限制，调整每个对等节点的最小/最大通道数（确保不超限）；
 * 3. 遍历P2P任务：按预定义的sendOrder/recvOrder遍历对等节点，处理每个节点的发送/接收任务；
 * 4. 自通信校验：检查自rank的发送/接收任务是否匹配，避免单边任务；
 * 5. 数据分块计算：根据跨节点/节点内场景计算分块大小，零大小同步任务编码为-1；
 * 6. 任务填充：循环拆分数据为适配通道的块，添加到内核计划，消耗预算；完成后移除任务队列中的已执行任务；
 * 7. 预算控制：若剩余预算不足，立即返回（保证内核计划不超限）。
 *
 * @note 关键设计点说明：
 * - 分块策略：跨节点场景分块更小（stepSize/2），节点内更大（stepSize/8），平衡延迟与带宽；
 * - 通道数调整：从最大通道数向下调整，确保总通道数（nChannelsMin*nRanks）不超过comm->p2pnChannels；
 * - 任务融合（fuseOk）：每NCCL_MAX_WORK_ELEMENTS_P2P/2个任务重置融合标记，控制任务融合粒度；
 * - 零大小任务：编码为-1（同步作用），添加到计划时还原为0，不传输数据但保证同步；
 * - 预算检查：每次添加任务前检查剩余预算，不足时立即返回（而非报错），保证内核计划合法性。
 */
static ncclResult_t scheduleP2pTasksToPlan(
    struct ncclComm* comm, struct ncclKernelPlan* plan, int* nWorkBudget
  ) {
  // 提取通信上下文中的待执行任务结构体
  struct ncclTasks* tasks = &comm->tasks;
  // 通信组的总rank数
  int nRanks = comm->nRanks;
  // 对等节点的P2P任务队列（发送/接收）
  struct ncclTasks::Peer* peers = tasks->peers;
  // P2P发送任务的执行顺序（预定义的rank顺序）
  int const *sendOrder = tasks->p2pSendOrder;
  // P2P接收任务的执行顺序（预定义的rank顺序）
  int const *recvOrder = tasks->p2pRecvOrder;

  // 配置内核计划的线程块大小：不小于NCCL_MAX_NTHREADS（保证并行度）
  plan->threadPerBlock = std::max(plan->threadPerBlock, NCCL_MAX_NTHREADS);
  // 若内核未特殊化，配置P2P对应的设备内核函数
  if (!plan->kernelSpecialized) {
    plan->kernelFn = ncclDevKernelForFunc[ncclDevFuncId_P2p()];
    plan->kernelSpecialized = ncclDevKernelForFuncIsSpecialized[ncclDevFuncId_P2p()];
  }

  // Compute how much to split operations
  // Natural step size matching buffer steps.
  // 计算P2P任务的基础分块大小（与缓冲区步长匹配）
  ssize_t stepSize = comm->p2pChunkSize;
  // Try to use all channels
  // 初始最大通道数：每个对等节点的P2P通道数上限
  int nChannelsMax = comm->p2pnChannelsPerPeer;
  int nChannelsMin = nChannelsMax;
  // Try to use all channels, but one channel per operation.
  // 调整最小通道数：确保总通道数（nChannelsMin*nRanks）不超过集群总P2P通道数，且不小于1
  while (nChannelsMin*nRanks > comm->p2pnChannels && nChannelsMin > 1) nChannelsMin /= 2;

  // 任务融合标记：控制P2P任务的融合粒度（减少内核启动开销）
  bool fuseOk = false;
  // We can perform 8 send/recv per round per CTA. Make sure we jump between fused blocks at node boundaries.
  // 循环处理所有待执行的P2P任务，直到任务队列为空
  while (tasks->nTasksP2p != 0) {
    // 按预定义的P2P任务顺序（p2pOrderSteps）遍历每个对等节点
    for (int i=0; i < tasks->p2pOrderSteps; i++) {
      // 当前轮次的发送/接收目标rank
      int sendPeer = sendOrder[i];
      int recvPeer = recvOrder[i];
      // 获取对应对等节点的发送/接收任务队列头节点
      struct ncclTaskP2p* send = sendPeer != -1 ? ncclIntruQueueHead(&peers[sendPeer].sendQueue) : NULL;
      struct ncclTaskP2p* recv = recvPeer != -1 ? ncclIntruQueueHead(&peers[recvPeer].recvQueue) : NULL;

      // 自通信（发送/接收目标为自身rank）的合法性校验
      if (sendPeer == comm->rank) {
        // 自通信时发送/接收目标必须同为自身，否则计划未对齐，返回内部错误
        if (recvPeer != comm->rank) {
          WARN("Sendrecv plan not aligned for self");
          return ncclInternalError;
        }
        // 自通信仅有发送无接收，参数非法
        if (send && recv == nullptr) {
          WARN("Trying to send to self without a matching recv");
          return ncclInvalidUsage;
        }
        // 自通信仅有接收无发送，参数非法
        if (send == nullptr && recv) {
          WARN("Trying to recv to self without a matching send");
          return ncclInvalidUsage;
        }
      }

      // 若当前轮次有发送或接收任务需要处理
      if (send != nullptr || recv != nullptr) {
        // 提取发送/接收的缓冲区指针、数据大小
        char* recvPtr = recv ? (char*)recv->buff : nullptr;
        char* sendPtr = send ? (char*)send->buff : nullptr;
        ssize_t recvBytes = recv ? recv->bytes : 0;
        ssize_t sendBytes = send ? send->bytes : 0;

        // 计算分块的最小/最大尺寸：跨节点（nNodes>1）分块更小，平衡延迟；节点内分块更大，利用带宽
        ssize_t minSize = comm->nNodes > 1 ? stepSize/2 : stepSize/8;
        ssize_t maxSize = comm->nNodes > 1 ? stepSize : stepSize*32;

        // 根据通道数、最小/最大尺寸，计算当前接收/发送任务的最大分块大小
        ssize_t recvChunkBytesMax = calcP2pChunkSize(recvBytes, nChannelsMin, nChannelsMax, minSize, maxSize);
        ssize_t sendChunkBytesMax = calcP2pChunkSize(sendBytes, nChannelsMin, nChannelsMax, minSize, maxSize);

        // Zero size send/recv are syncs, encode here with -1.
        // 零大小的发送/接收任务是同步任务，编码为-1（便于后续处理）
        recvBytes = recv && recvBytes == 0 ? -1 : recvBytes;
        sendBytes = send && sendBytes == 0 ? -1 : sendBytes;

        // Advance to current chunk. Syncs will always have chunk=0 so no effect on the -1.
        // 偏移到当前分块的起始位置（同步任务chunk=0，不影响-1编码）
        if (recv) recvPtr   += recv->chunk*recvChunkBytesMax;
        if (recv) recvBytes -= recv->chunk*recvChunkBytesMax;
        if (send) sendPtr   += send->chunk*sendChunkBytesMax;
        if (send) sendBytes -= send->chunk*sendChunkBytesMax;

        // 循环拆分数据为适配的分块，直到当前发送/接收任务完成
        do {
          // 每NCCL_MAX_WORK_ELEMENTS_P2P/2个任务重置融合标记，控制融合粒度
          if ((i % (NCCL_MAX_WORK_ELEMENTS_P2P/2)) == 0) fuseOk = false;
          // 计算当前分块大小（同步任务的-1会被保留，后续处理为0）
          ssize_t recvChunkBytes = std::min(recvBytes, recvChunkBytesMax); // -1 preserved
          ssize_t sendChunkBytes = std::min(sendBytes, sendChunkBytesMax);

          // 处理接收任务分块
          if (recvChunkBytes != 0) {
            // 同步任务（-1）还原为0（仅同步，无数据传输）
            if (recvChunkBytes == -1) recvChunkBytes = 0;
            // 检查剩余预算，不足则提前返回（保证内核计划不超限）
            if (*nWorkBudget < 1) return ncclSuccess; // ensure room in budget
            // 将接收任务分块添加到内核计划，消耗1个预算，标记任务可融合
            NCCLCHECK(addP2pToPlan(comm, plan, nWorkBudget, /*isSendNotRecv=*/false, recvPeer, recv->chunk, recvPtr, recvChunkBytes, fuseOk));
            fuseOk = true;
            // 偏移缓冲区指针、剩余字节数，更新当前分块索引
            recvPtr += recvChunkBytes;
            recvBytes -= recvChunkBytes;
            recv->chunk += 1;
            // 若当前接收任务完成，从队列移除并减少待执行P2P任务数
            if (recvBytes <= 0) {
              recvBytes = 0; // in case still -1
              ncclIntruQueueDequeue(&peers[recvPeer].recvQueue);
              tasks->nTasksP2p -= 1;
            }
          }

          // 处理发送任务分块（逻辑与接收一致）
          if (sendChunkBytes != 0) {
            if (sendChunkBytes == -1) sendChunkBytes = 0;
            if (*nWorkBudget < 1) return ncclSuccess; // ensure room in budget
            NCCLCHECK(addP2pToPlan(comm, plan, nWorkBudget, /*isSendNotRecv=*/true, sendPeer, send->chunk, sendPtr, sendChunkBytes, fuseOk));
            fuseOk = true;
            sendPtr += sendChunkBytes;
            sendBytes -= sendChunkBytes;
            send->chunk += 1;
            if (sendBytes <= 0) {
              sendBytes = 0; // in case still -1
              ncclIntruQueueDequeue(&peers[sendPeer].sendQueue);
              tasks->nTasksP2p -= 1;
            }
          }
        } while (sendBytes != 0 || recvBytes != 0); // 直到当前发送/接收任务分块完成
      }
    }
  }
  // 所有P2P任务调度完成，返回成功
  return ncclSuccess;
}
```

## 4.3 ncclDevKernelForFunc ： 从 primary_funcs 到 ncclDevKernelForFunc

- kernel 从这来 **primary_funcs**

```py
  # Forward declarations of kernels.
  for kfn in kernel_funcs:
    cudart, _ = required_cuda(*kfn)
    sym = paste("_", "ncclDevKernel", *kfn)
    if cudart != 0: out("#if CUDART_VERSION >= %d\n" % cudart)
    out("__global__ void %s(struct ncclDevComm*, uint64_t, struct ncclWork*);\n" % sym)
    if cudart != 0: out("#endif\n")
  out("\n")

  # List of all kernel function pointers.
  out("extern int const ncclDevKernelCount = %d;\n" % len(kernel_funcs))
  out("extern void* const ncclDevKernelList[] = {\n")
  index = 0
  for kfn in kernel_funcs:
    cudart, _ = required_cuda(*kfn)
    sym = paste("_", "ncclDevKernel", *kfn)
    if cudart != 0: out("#if CUDART_VERSION >= %d\n" % cudart)
    out("/*%4d*/ (void*)%s,\n" % (index, sym));
    if cudart != 0: out("#else\n" "/*%4d*/ nullptr,\n" "#endif\n" % index)
    index += 1
  out("nullptr};\n")
  out("\n")

  # Maps primary id to kernel function pointer.
  out("extern void* const ncclDevKernelForFunc[] = {\n")
  index = 0
  for fn in primary_funcs:
    kfn = best_kernel(*fn)
    sym = paste("_", "ncclDevKernel", *kfn)
    cudart, _ = required_cuda(*kfn)
    if cudart != 0: out("#if CUDART_VERSION >= %d\n" % cudart)
    out("/*%4d*/ (void*)%s,\n" % (index, sym))
    if cudart != 0: out("#else\n" "/*%4d*/ nullptr,\n" "#endif\n" % index)
    index += 1
  out("nullptr};\n")
  out("\n")

  # Does the prior map use an explicitly specialized kernel.
  out("extern bool const ncclDevKernelForFuncIsSpecialized[] = {\n")
  index = 0
  for fn in primary_funcs:
    kfn = best_kernel(*fn)
    specialized = "1" if fn == kfn else "0"
    out("/*%4d*/ %s,\n" % (index, specialized))
    index += 1
  out("0};\n")
```

## 4.4 primary_funcs 的形成

**形成kernel的开始**

```python
algos_of_coll = {
  "AllGather":     ["RING","COLLNET_DIRECT","NVLS"],
  "AllReduce":     all_algos,
  "Broadcast":     ["RING"],
  "Reduce":        ["RING"],
  "ReduceScatter": ["RING","COLLNET_DIRECT","NVLS"],
  "SendRecv":      [None]
}

coll_camel_to_lower = {
  "AllGather":     "all_gather",
  "AllReduce":     "all_reduce",
  "Broadcast":     "broadcast",
  "Reduce":        "reduce",
  "ReduceScatter": "reduce_scatter",
  "SendRecv":      "sendrecv"
}
coll_lower_to_camel = {coll_camel_to_lower[x]: x for x in coll_camel_to_lower}
```

# 5 kernel 函数具体调度

5.1 sendrecv.cu

**此代码自动生成: build/obj/device/gensrc/sendrecv.cu**

- __global__ ncclDevKernel_SendRecv 声明;
- __device__ ncclDevFunc_SendRecv 声明;

```c++
#include "common.h"
#include "sendrecv.h"

// - 本质：生成一个可独立启动的 CUDA 设备内核（device kernel），这是能通过<<<grid, block>>>语法直接在 GPU 上启动的函数。
DEFINE_ncclDevKernel(SendRecv, ncclFuncSendRecv, FuncCopy, int8_t, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, 669)
// - 本质：生成一个仅能在设备端调用的函数（device function），不能直接启动，只能被其他 Kernel 或 Device Function 调用。
DEFINE_ncclDevFunc(SendRecv, ncclFuncSendRecv, FuncCopy, int8_t, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE)
```

## 5.2 ncclDevKernel_SendRecv 用户kernel 的定义

- **DEFINE_ncclDevKernel：定义可直接启动的 CUDA Kernel**

```c++
// ncclDevKernel_SendRecv
#define DEFINE_ncclDevKernel(suffix, coll, redop, ty, algo, proto, specializedFnId) \
  __global__ void ncclDevKernel_##suffix(ncclDevKernelArgs4K NCCL_GRID_CONSTANT const args4K) { \
    ncclKernelMain<specializedFnId, RunWorkBatch<coll, ty, redop<ty>, algo, proto>>(&args4K.args); \
  }
```

- ncclKernelMain 中查找并执行func 函数

```c++
template<int SpecializedFnId, typename SpecializedRunWorkBatch>
__device__ __forceinline__ void ncclKernelMain(struct ncclDevKernelArgs const* args) {
  ...

  while (ncclShmem.aborted == 0) {
    profiler(START);
    if (0 <= SpecializedFnId && ncclShmem.funcId == (unsigned)SpecializedFnId) {
      SpecializedRunWorkBatch().run();
    } else {
      ncclDevFuncTable[ncclShmem.funcId]();
    }

    if (ncclShmem.nextBatchIx == -1) break;
    int batchIx = ncclShmem.nextBatchIx;
    __syncthreads();
    profiler(STOP);
    loadWorkBatchToShmem(tid, tn, args, batchIx);
    __syncthreads();
  }

  ...
}
```

- ncclDevFuncTable : **前面的号码就是specializedFnId**

```c++
// build/obj/device/gensrc/device_table.cu
__device__ ncclDevFuncPtr_t const ncclDevFuncTable[] = {
  ...
  /* 669*/ ncclDevFunc_SendRecv,
  nullptr
};
```

**查询得到ncclDevFunc_SendRecv 并执行**

## 5.3 ncclDevFunc_SendRecv 的定义

- 声明

```c++
DEFINE_ncclDevFunc(SendRecv, ncclFuncSendRecv, FuncCopy, int8_t, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE)
```

- 定义

```c++
// ncclDevFunc_SendRecv
#define DEFINE_ncclDevFunc(suffix, coll, redop, ty, algo, proto) \
  __device__ void ncclDevFunc_##suffix() { \
    RunWorkBatch<coll, ty, redop<ty>, algo, proto>().run(); \
  }
```

- RunWorkBatch nccl 针对SendRecv 有特化的实现

## 5.4 RunWorkBatch<ncclFuncSendRecv， ...> Specialized 的实现

**ncclDevKernelForFuncIsSpecialized**
- 核心逻辑：同样接收 Func 特征参数，但不返回 Kernel 实例，`只检查注册表中该参数组合对应的 Kernel 是否是 “特化版”（而非通用版）`—— 比如是否为int8_t单独定制，而非用一个兼容int8_t/int32_t/float的通用 Kernel。

```c++
// Specialized here for non-P2p (Coll and CollReg)
template<ncclFunc_t Fn, typename T, typename RedOp, int Algo, int Proto>
struct RunWorkBatch {
  // This __forceinline__ is necessary. The compiler was inserting a function call
  // here from the LL ncclKernel.
  __device__ __forceinline__ void run() {
    int tid = threadIdx.x;
    int tn = blockDim.x;

    if (RedOpArg<RedOp>::ArgUsed) {
      int nWorks = ncclShmem.nWorks;
      for (int w=tid; w < nWorks; w += tn) {
        struct ncclDevWorkColl* work = (ncclDevWorkColl*)(ncclShmem.workStorage + w*ncclShmem.workSize);
        if (work->redOpArgIsPtr) {
          work->redOpArg = RedOpArg<RedOp>::loadArg(reinterpret_cast<void*>(work->redOpArg));
        }
      }
      __syncthreads();
    }

    #pragma unroll 1
    for (int w=0; w < ncclShmem.nWorks; w++) {
      struct ncclDevWorkColl* work = (struct ncclDevWorkColl*)(ncclShmem.workStorage + w*ncclShmem.workSize);
      if (w != 0) {
        struct ncclDevWorkColl* workPrev = (struct ncclDevWorkColl*)(ncclShmem.workStorage + (w-1)*ncclShmem.workSize);
        if (work->nWarps != workPrev->nWarps) __syncthreads();
      }
      int subtn = work->nWarps*WARP_SIZE;
      // Coverity reports a possible thread divergence due to not all threads participating in the collective.
      // However, the code ensures that the participation is on a per-warp basis.
      // coverity[device_thread_diverged:FALSE]
      if (tid < subtn) RunWorkColl<Fn, T, RedOp, Algo, Proto>().run(tid, subtn, work);
    }
  }
};
```

## 5.4 一般实现

**ncclDevKernelForFunc** <br>
- 核心逻辑：NCCL 内部维护了一个全局的 DevKernel 注册表（由DEFINE_ncclDevKernel生成的内核都会注册到这里），该接口接收Func标识（如ncclFuncSendRecv）、数据类型（如int8_t）、算法（如NCCL_ALGO_RING）、协议（如NCCL_PROTO_SIMPLE）等参数，在注册表中匹配并返回对应的ncclDevKernel对象指针。

**ncclDevKernelForFunc 的使用场景:** <br>
- 核心场景：需要实际启动 / 调用 Kernel的阶段，是 “执行层” 接口。

```c++
template<typename T, typename RedOp>
struct RunWorkBatch<ncclFuncSendRecv, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  static_assert(sizeof(T)==1, "SendRecv only works on single byte types T.");

  template<typename Proto>
  __device__ void runSend(int tid, int tn, int group, struct ncclDevWorkP2p* work) {
    size_t bytes = work->sendBytes;
    bool useLargeChunk = (work->sendIpcReg && ncclShmem.comm.isAllNvlink) || work->sendNetReg;
    int chunkSize = useLargeChunk ? NCCL_MAX_NET_SIZE : u32fp8Decode(work->sendChunkSize_u32fp8);
    int stepSize = useLargeChunk ? NCCL_MAX_NET_SIZE : ncclShmem.comm.p2pChunkSize;
    Primitives<T, RedOp, FanAsymmetric<0, 1>, 1, Proto, 1>
      prims(tid, tn, nullptr, &work->sendRank, work->sendAddr, nullptr,
            /*redOpArg(ignored)=*/0, group, 1, 1, nullptr, work, stepSize);
    size_t cursor = 0;
    do {
      int n = min(size_t(chunkSize), bytes-cursor);
      prims.directSend(cursor, cursor, n);
      cursor += n;
    } while (cursor < bytes);
  }

  template<typename Proto>
  __device__ void runRecv(int tid, int tn, int group, struct ncclDevWorkP2p* work) {
    size_t bytes = work->recvBytes;
    bool useLargeChunk = (work->recvIpcReg && ncclShmem.comm.isAllNvlink) || work->recvNetReg;
    int chunkSize = useLargeChunk ? NCCL_MAX_NET_SIZE : u32fp8Decode(work->recvChunkSize_u32fp8);
    int stepSize = useLargeChunk ? NCCL_MAX_NET_SIZE : ncclShmem.comm.p2pChunkSize;
    Primitives<T, RedOp, FanAsymmetric<1, 0>, 1, Proto, 1>
      prims(tid, tn, &work->recvRank, nullptr, nullptr, work->recvAddr,
            /*redOpArg(ignored)=*/0, group, 1, 1, nullptr, work, stepSize);
    size_t cursor = 0;
    do {
      int n = min(size_t(chunkSize), bytes-cursor);
      prims.directRecv(cursor, n);
      cursor += n;
    } while (cursor < bytes);
  }

  __device__ __forceinline__ void run() {
    const int tid = threadIdx.x;
    const int tn = blockDim.x;
    const int wid = tid/WARP_SIZE;
    const int nWarps = tn/WARP_SIZE;
    const int lane = tid%WARP_SIZE;

    struct Shared {
      uint32_t workSendMask; // bitmasks of which work indices have send/recv
      uint32_t workRecvMask;
    };
    Shared* shared = (Shared*)ncclScratchForWarp(0);

    struct ncclDevWorkP2p* works = (ncclDevWorkP2p*)ncclShmem.workStorage;
    int nWorks = ncclShmem.nWorks;

    if (wid == 0) {
      // Modify the memory range of each work[] to reflect this channel's
      // partition of the work. Since integer divides are very heavy it's
      // best to do them all in one warp.
      int workIx = lane%16;
      int isSend = lane < 16 ? 0 : 1;
      bool hasWork = false;
      if (workIx < nWorks) {
        struct ncclDevWorkP2p* work = &works[workIx];
        size_t bytes = isSend ? work->sendBytes : work->recvBytes;
        int nParts = isSend ? work->nSendChannels : work->nRecvChannels;
        int part = ncclP2pChannelToPart(work->nP2pChannels, work->channelBase, ncclShmem.channelId);
        hasWork = (part < nParts);
        if (nParts != 0) {
          size_t partBeg, partEnd;
          ncclP2pPartBounds(nParts, part, bytes, &partBeg, &partEnd);
          (isSend ? work->sendAddr : work->recvAddr) = (char*)(isSend ? work->sendAddr : work->recvAddr) + partBeg;
          (isSend ? work->sendBytes : work->recvBytes) = partEnd - partBeg;
        }
      }
      // Coverity reports a possible thread divergence due to not all threads participating in the collective.
      // However, the code ensures that the participation is on a per-warp basis.
      // coverity[device_thread_diverged:FALSE]
      uint32_t mask = __ballot_sync(~0u, hasWork);
      if (lane == 0) {
        shared->workSendMask = mask>>16;
        shared->workRecvMask = mask & 0xffff;
      }
    }

    // The fastest way to compute a warp uniform division x/y in [0,32) is to
    // use each lane to guess a solution and count the ones that don't exceed
    // the numerator:
    //   __popc(__ballot_sync(~0u, y*(lane+1) <= x))
    // That takes 1/3 the time of standard division and about 3/4 the time of
    // approximate floating point division:
    //   __float2int_rd(__fdividef(float(x),float(y))).

    // nWarpPerWork = nWarps/nWorks
    int nWarpPerWork = __popc(__ballot_sync(~0u, nWorks*(lane+1) <= nWarps));
    int nRecvWarpPerWork = nWarpPerWork<=4 ? nWarpPerWork/2 : (nWarpPerWork-1)/2;
    int nSendWarpPerWork = nWarpPerWork<=4 ? nRecvWarpPerWork : nRecvWarpPerWork+1;
    // This might reduce nWarpPerWork which is probably desirable. It is better
    // to have a balanced number of reading and writing threads even if that
    // leaves warps unused.
    nWarpPerWork = nSendWarpPerWork + nRecvWarpPerWork;
    // The work index this warp belongs to: workIx = wid/nWarpPerWork
    int workIx = __popc(__ballot_sync(~0u, (lane+1)*nWarpPerWork <= wid));

    __syncthreads(); // Wait for works[] and shared->* to be updated by warp=0

    uint32_t workSendMask = shared->workSendMask;
    uint32_t workRecvMask = shared->workRecvMask;

    __syncthreads(); // release scratch space used by shared->*
    if (nWorks <= workIx) return;

    // Thread range for whole work (send & recv combined)
    int subtid = tid - workIx*nWarpPerWork*WARP_SIZE;
    int subtn = nWarpPerWork*WARP_SIZE;

    // A send primtive of sufficient size requires 2 cuda barrier ids.
    constexpr int nSendWarpsForExtraGroup = NCCL_SIMPLE_EXTRA_GROUP_IF_NTHREADS_GE/WARP_SIZE;
    // Count up all group ids used below this workIx:
    int group, extra;
    // Each recv gets one group id:
    group = __popc(workRecvMask & ((1<<workIx)-1));
    // Sends accompanying recvs get one and maybe an extra:
    extra = (nSendWarpPerWork >= nSendWarpsForExtraGroup) ? 1 : 0;
    group += __popc((workSendMask & workRecvMask) & ((1<<workIx)-1))*(1+extra);
    // Sends without recvs use more warps so compute extra accordingly:
    extra = (nWarpPerWork >= nSendWarpsForExtraGroup) ? 1 : 0;
    group += __popc((workSendMask & ~workRecvMask) & ((1<<workIx)-1))*(1+extra);

    struct ncclDevWorkP2p* work = &works[workIx];
    bool hasSend = 1 & (workSendMask>>workIx);
    bool hasRecv = 1 & (workRecvMask>>workIx);
    bool isCopy = work->sendRank == ncclShmem.comm.rank;
    bool isSend = !hasRecv || (hasSend && subtid < nSendWarpPerWork*WARP_SIZE);

    if (!isCopy && hasSend && hasRecv) {
      // Translate thread ids to reflect just this send or recv as opposed to whole work.
      if (isSend) {
        subtn = nSendWarpPerWork*WARP_SIZE;
      } else {
        subtid -= nSendWarpPerWork*WARP_SIZE;
        subtn = nRecvWarpPerWork*WARP_SIZE;
        group += 1 + (nSendWarpPerWork >= nSendWarpsForExtraGroup ? 1 : 0);
      }
    }

    if (isCopy) {
      reduceCopy<COLL_UNROLL, RedOp, T, 0,1,1, 0,1,1, /*PreOpSrcs=*/0>
        (subtid, subtn, 0, nullptr, false, 1, &work->sendAddr, 1, &work->recvAddr, (ssize_t)work->sendBytes);
    } else if (isSend) {
      if (work->sendProtoLL) {
        runSend<ProtoLL>(subtid, subtn, group, work);
      } else {
        runSend<ProtoSimple<1,1>>(subtid, subtn, group, work);
      }
    } else {
      if (work->recvProtoLL) {
        runRecv<ProtoLL>(subtid, subtn, group, work);
      } else {
        runRecv<ProtoSimple<1,1>>(subtid, subtn, group, work);
      }
    }
  }
};
```

# 6 Primitives 里完成具体操作

## 6.1 directSend

- send: 适用于常规的点对点通信场景
- directSend: 适用于需要高性能、低延迟的场景，特别是在支持直接内存访问(DMA)的硬件上，可以减少数据拷贝次数

```c++
  __device__ __forceinline__ void directSend(intptr_t inpIx, intptr_t outIx, int eltN) {
    genericOp<0, 1, 0, 1, Input, -1>(inpIx, outIx, eltN, false);
  }
```

## 6.2 directRecv

```c++
  __device__ __forceinline__ void directRecv(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<1, 0, 1, 0, -1, Output>(outIx, outIx, eltN, postOp);
  }
```

## 6.3 genericOp

```c++
  // send : 0, 1, 0, 1, Input, -1
  // recv: 1, 0, 1, 0, -1, Output
  template <int DirectRecv1, int DirectSend1, int Recv, int Send, int SrcBuf, int DstBuf>
  __device__ __forceinline__ void genericOp(
      intptr_t srcIx, intptr_t dstIx, int nelem, bool postOp
    ) {
    constexpr int DirectRecv = 1 && Direct && DirectRecv1;
    constexpr int DirectSend = 1 && Direct && DirectSend1;
    constexpr int Src = SrcBuf != -1;
    constexpr int Dst = DstBuf != -1;

    nelem = nelem < 0 ? 0 : nelem;
    int sliceSize = stepSize*StepPerSlice;
    sliceSize = max(divUp(nelem, 16*SlicePerChunk)*16, sliceSize/32);
    int slice = 0;
    int offset = 0;

    if (tid < nworkers && offset < nelem && !isNetOffload) {
      // Worker-only loop for non-empty slices. Non-workers and empty slices are
      // processed in the loop following this if block. The benefit of splitting
      // the loop like this is we pull two branches out of the critical path.
      // Using "number of branch insns (taken or not) encountered dynamically"
      // as the performance metric, then:
      //   perf_orig = 2*numslices
      //   perf_new = 2+numslices
      // So the new code and old code behave the same for numslices=2, and for
      // numslices>2 the new code is superior. And note that in the case
      // numslices=1, the loop is trivially unrollable (single iteration) so we
      // don't incur that that tail branch and we still have perf_new=2.
      //
      // ORIGINAL CODE:
      //   unrolled for(slices) {
      //     if(worker) { // This branch removed
      //       wait();
      //       subBarrier();
      //       if(slice not empty) // This branch removed
      //         ReduceCopyMulti();
      //     }
      //     barrier();
      //     post();
      //   } // Since we no longer unroll, new branch added here
      #if __CUDA_ARCH__ < 700
        // Above doesn't matter on older hardware.
        #pragma unroll SlicePerChunk
      #else
        #pragma unroll 1
      #endif
      do {
        sliceSize = sliceSize < nelem-offset ? sliceSize : nelem-offset;
        if (tid == 0) {
          T* userInput = (T*)ncclShmem.groups[group].userInput;
          T* userOutput = (T*)ncclShmem.groups[group].userOutput;
          if (Src) ncclShmem.groups[group].srcs[0] = (SrcBuf==Input ? userInput : userOutput) + srcIx + offset;
          if (Dst) ncclShmem.groups[group].dsts[0] = (DstBuf==Input ? userInput : userOutput) + dstIx + offset;
        }
        waitPeer<DirectRecv, DirectSend, Recv, Send, Src, Dst>(srcIx, dstIx, offset, sliceSize);
        subBarrier();
        /* if user abort the kernel, we don't need to actually perform copy/reduce; just set size
         * to 0 to avoid unnecessary workload. */
        int workSize = ncclShmem.aborted ? 0 : sliceSize;
        if (flags & AnyNetDeviceUnpack) {
          ncclNetDeviceUnpack<Recv>(tid, tidInBlock, nworkers, group, ncclShmem.groups[group].devicePlugin.unpack.unpackNetDeviceIndexMask, Src, workSize);
          // Sync here to make sure all workers are reading from the updated srcs)
          subBarrier();
        }

        if (DirectRecv && ncclShmem.groups[group].srcs[0] == ncclShmem.groups[group].dsts[0]
            /* NVLS can have srcs[0] == dsts[0], but we cannot enter this "if branch",
             * so we need to check whether MultimemSrcs and MultimemDsts are 0. */
            && MultimemSrcs == 0 && MultimemDsts == 0 && !Src) {
          // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
          if (Send && Dst && ncclShmem.groups[group].srcs[0] != ncclShmem.groups[group].dsts[1]) {
            reduceCopy<Unroll, RedOp, T, 0, 1, 1, 0, 1, MaxSend, /*PreOpSrcs*/0>
              (tid, nworkers, /*redArg*/0, /*preOpArgs*/nullptr, /*postOp*/false,
               1, ncclShmem.groups[group].srcs,
               fan.nsend(), ncclShmem.groups[group].dsts+1,
               workSize);
          }
        } else if (DirectSend && !DirectRecv && SrcBuf != Input && ncclShmem.groups[group].dsts[Dst] == nullptr) {
          // For broadcast in CollNet to do empty send
          reduceCopy<Unroll, RedOp, T, 0, 1, 1, 0, 1, 1, /*PreOpSrcs*/0>
            (tid, nworkers, ncclShmem.redOpArgs[0],  nullptr, postOp,
             Recv, ncclShmem.groups[group].srcs,
             Dst, ncclShmem.groups[group].dsts,
             workSize);
        } else if (ncclShmem.groups[group].srcs[0] && ncclShmem.groups[group].dsts[0]) {
          constexpr int PreOpSrcs = SrcBuf != Input ? 0 :
                                    DirectRecv*MaxRecv == NCCL_MAX_DIRECT_ARITY ? (1+NCCL_MAX_DIRECT_ARITY) : 1;
          if (Send && Dst && ncclShmem.groups[group].dsts[1] == nullptr) {
            // this case should only be directCopySend() with registered buffers and send to net peer
            reduceCopy<Unroll, RedOp, T,
              0, Recv + Src, Recv * MaxRecv + Src,
              0, 1, 1, PreOpSrcs>
              (tid, nworkers, ncclShmem.redOpArgs[0], ncclShmem.redOpArgs, postOp,
                Recv * fan.nrecv() + Src, ncclShmem.groups[group].srcs,
                1, ncclShmem.groups[group].dsts,
                workSize);
          } else {
            reduceCopy<Unroll, RedOp, T,
              MultimemSrcs, Recv + Src, Recv * MaxRecv + Src,
              MultimemDsts, Send + Dst, Send * MaxSend + Dst, PreOpSrcs>
              (tid, nworkers, ncclShmem.redOpArgs[0], ncclShmem.redOpArgs, postOp,
                Recv * fan.nrecv() + Src, ncclShmem.groups[group].srcs,
                Send * fan.nsend() + Dst, ncclShmem.groups[group].dsts,
                workSize);
          }
        } else {
          // we will come here when calling prims.directSend with net peer,
          // in this case, ncclShmem.groups[group].dsts[0] == NULL, so we
          // skip data flush.
          workSize = 0;
        }
        barrier(); // This barrier has a counterpart in following loop
        postPeer<Recv, Send>(0 < workSize);
        offset += sliceSize;
        slice += 1;
        // Yes, for some template arguments this code will be unreachable.  That's fine.
        // coverity[dead_error_line]
      } while (slice < SlicePerChunk && offset < nelem);
    }

    // Non-workers come straight here. Workers too but only once the remaining
    // slices are all empty. Since empty slices are the uncommon case, and
    // worker perf is the limiter, perf-wise this loop is effectively unentered,
    // hence just a single branch insn.
    #pragma unroll 1
    while (slice < SlicePerChunk) {
      sliceSize = sliceSize < nelem-offset ? sliceSize : nelem-offset;
      { // Only workers could have Wait roles so we know the slice must be empty
        // since we've exited the loop above.
        waitPeer<DirectRecv, DirectSend, Recv, Send, Src, Dst>(0, 0, 0, sliceSize);
      }
      barrier(); // Has couterpart in preceding worker-only loop.
      int workSize = ncclShmem.aborted ? 0 : sliceSize;
      postPeer<Recv, Send>(0 < workSize);
      offset += sliceSize;
      slice += 1;
    }
  }
```