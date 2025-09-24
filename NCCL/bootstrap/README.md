# 1 主进程生成ncclUniqueId 时会发起ncclSocket 服务

如[ncclUniqueID](../ncclUniqueID/README.md) 所述:

**主进程创建完成ncclUniqueId 后，还会创建一个ncclSocket 监听线程，该线程会监听来自其他进程的连接请求。**

**bootstrapRoot** 是核心函数，用于多个ranks bootstrap 沟通过程。

```c++
ncclResult_t bootstrapGetUniqueId(struct ncclBootstrapHandle* handle) {
  memset(handle, 0, sizeof(ncclBootstrapHandle));

  const char* env = ncclGetEnv("NCCL_COMM_ID");
  if (env) {
    INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", env);
    if (ncclSocketGetAddrFromString(&handle->addr, env) != ncclSuccess) {
      WARN("Invalid NCCL_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
      return ncclInvalidArgument;
    }
    handle->magic = NCCL_MAGIC;
  } else {
    NCCLCHECK(getRandomData(&handle->magic, sizeof(handle->magic)));
    memcpy(&handle->addr, &bootstrapNetIfAddr, sizeof(union ncclSocketAddress));
    NCCLCHECK(bootstrapCreateRoot(handle, false));
  }

  return ncclSuccess;
}

ncclResult_t bootstrapCreateRoot(struct ncclBootstrapHandle* handle, bool idFromEnv) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSocket* listenSock = NULL;  // 用于监听客户端连接的套接字
  struct bootstrapRootArgs* args = NULL; // 传递给线程的参数结构体
  pthread_t thread;                      // 线程标识符，用于管理线程生命周期

  NCCLCHECK(ncclCalloc(&listenSock, 1));
  NCCLCHECKGOTO(ncclSocketInit(listenSock, &handle->addr, handle->magic, ncclSocketTypeBootstrap, NULL, 0), ret, fail);
  NCCLCHECKGOTO(ncclSocketListen(listenSock), ret, fail);
  NCCLCHECKGOTO(ncclSocketGetAddr(listenSock, &handle->addr), ret, fail);

  NCCLCHECKGOTO(ncclCalloc(&args, 1), ret, fail);
  args->listenSock = listenSock;
  args->magic = handle->magic;
  PTHREADCHECKGOTO(pthread_create(&thread, NULL, bootstrapRoot, (void*)args), "pthread_create", ret, fail);
  ncclSetThreadName(thread, "NCCL BootstrapR");
  PTHREADCHECKGOTO(pthread_detach(thread), "pthread_detach", ret, fail); // will not be pthread_join()'d
exit:
  return ret;
fail:
  if (listenSock) free(listenSock);
  if (args) free(args);
  goto exit;
}
```

- 创建 **ncclComm_t**

```c++
  static std::shared_ptr<NCCLComm> create(
      int numRanks,
      int rank,
      ncclUniqueId commId,
      at::DeviceIndex deviceIndex,
      ncclConfig_t& config) {
    at::cuda::OptionalCUDAGuard gpuGuard(deviceIndex);
    auto comm = std::make_shared<NCCLComm>();
    comm->nonBlocking_ = config.blocking == 0;
    LOG(INFO) << "Rank " << rank << ": creating NCCL communicator with mode: "
              << (comm->nonBlocking_ ? "nonblocking" : "blocking");
    C10D_NCCL_CHECK_NONBLOCKING(
        ncclCommInitRankConfig(
            &(comm->ncclComm_), numRanks, commId, rank, &config),
        std::nullopt);
    comm->ncclId_ = commId;
    comm->rank_ = rank;
    comm->deviceIndex_ = deviceIndex;
    // Under blocking mode, comm is initialized immediately after NCCL init
    // returns; Under nonblocking mode, we check whether comm is initialized the
    // *next* time ncclComm_ is accessed.
    comm->initialized_ = !comm->nonBlocking_;
    return comm;
  }
```

# 2 client 端是如何连接 server 端的呢？

创建ncclComm_t 时，最核心的函数是: **ncclCommInitRankConfig** <br>

NCCL_COMM_ID 环境变量存在并且rank = 0 时，会调用bootstrapCreateRoot重试开启server端监听线程。

```c++
static ncclResult_t ncclCommInitRankDev(ncclComm_t* newcomm, int nranks, int nId, ncclUniqueId* commId, int myrank, int cudaDev, ncclConfig_t *config, const char funcName[]) {
  ...

  commIdEnv = ncclGetEnv("NCCL_COMM_ID");
  if (commIdEnv && myrank == 0) {
    INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", commIdEnv);
    if (nId > 1) {
      INFO(NCCL_INIT | NCCL_ENV, "NCCL_COMM_ID cannot be used with more than one ncclUniqueId");
      job->nId = 1;
    }
    // start the bootstrap root before bootstrapping, use only the first handle
    NCCLCHECKGOTO(bootstrapCreateRoot((struct ncclBootstrapHandle*)&job->commId[0], true), res, fail);
  }
  launchedJob = true;
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, ncclCommInitRankFunc, NULL, ncclCommInitJobFree, comm), res, fail);

exit:
  return ncclGroupErrCheck(res);
fail:
  if (job && !launchedJob) ncclCommInitJobFree(job);
  if (comm) {
    free(comm->abortFlag);
    if (comm->abortFlagDev) (void)ncclCudaHostFree((void*)comm->abortFlagDev);
    free(comm->abortFlagRefCount);
    free(comm);
  }
  if (newcomm) *newcomm = NULL;
  goto exit;
}
```

**ncclCommInitRankFunc** 会完成重要的初始化任务, 初始化通信器（communicator）的核心函数.

```c++
static ncclResult_t ncclCommInitRankFunc(struct ncclAsyncJob* job_) {
  struct ncclCommInitRankAsyncJob* job = (struct ncclCommInitRankAsyncJob*)job_;
  ncclComm_t comm = job->comm;
  ncclResult_t res = ncclSuccess;
  int archMajor, archMinor;
  size_t maxLocalSizeBytes = 0;
  int cudaDev = job->cudaDev;
  int* parentRanks = NULL;
  int cudaArch;
  int maxSharedMem = 0;
  double sum_timers = 0;
  uint64_t timers[TIMERS_INIT_COUNT] = {0};
  unsigned long long commIdHash;

  timers[TIMER_INIT_TOTAL] = clockNano();
  CUDACHECKGOTO(cudaSetDevice(cudaDev), res, fail);
  CUDACHECKGOTO(cudaDeviceGetAttribute(&maxSharedMem, cudaDevAttrMaxSharedMemoryPerBlockOptin, cudaDev), res, fail);
  CUDACHECKGOTO(cudaDeviceGetAttribute(&archMajor, cudaDevAttrComputeCapabilityMajor, cudaDev), res, fail);
  CUDACHECKGOTO(cudaDeviceGetAttribute(&archMinor, cudaDevAttrComputeCapabilityMinor, cudaDev), res, fail);
  cudaArch = 100*archMajor + 10*archMinor;

  timers[TIMER_INIT_KERNELS] = clockNano();
  NCCLCHECK(ncclInitKernelsForDevice(cudaArch, maxSharedMem, &maxLocalSizeBytes));
  // Set the maximum kernel stack size of all kernels to avoid
  // a CUDA memory reconfig on load (c.f. NVSHMEM issue)
  if (maxLocalSizeBytes > 0 && ncclParamSetStackSize() == 1) {
    TRACE(NCCL_INIT, "Setting cudaLimitStackSize to %zu", maxLocalSizeBytes);
    CUDACHECKIGNORE(cudaDeviceSetLimit(cudaLimitStackSize, maxLocalSizeBytes));
  }
  timers[TIMER_INIT_KERNELS] = clockNano() - timers[TIMER_INIT_KERNELS];

  if (job->parent) {
    NCCLCHECKGOTO(ncclCalloc(&parentRanks, job->parent->nRanks), res, fail);
    if (job->excludeRanksCount) {
      NCCLCHECKGOTO(getParentRanks(job->parent->nRanks, job->parent->rank, job->excludeRanksList, job->excludeRanksCount, &job->nranks, &job->myrank, parentRanks), res, fail);
    } else {
      NCCLCHECKGOTO(commGetSplitInfo(comm, job->parent, job->color, job->key, &job->nranks, &job->myrank, parentRanks), res, fail);
      // Negative color does not create a new comm object. We needed to take part in the allgather, but we're done now.
      if (job->color == NCCL_SPLIT_NOCOLOR) goto exit;
    }
    // child hash obtained from (parent hash, split count, color)
    uint64_t hacc[2] = {1, 1};
    eatHash(hacc, &job->parent->commHash);
    eatHash(hacc, &job->splitCount);
    eatHash(hacc, &job->color);
    comm->commHash = digestHash(hacc);
    timers[TIMER_INIT_ALLOC] = clockNano();
    NCCLCHECKGOTO(commAlloc(comm, job->parent, job->nranks, job->myrank), res, fail);
    timers[TIMER_INIT_ALLOC] = clockNano() - timers[TIMER_INIT_ALLOC];
    INFO(NCCL_INIT, "%s comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx parent %p splitCount %d color %d key %d- Init START", job->funcName,
         comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId, job->parent, job->splitCount, job->color, job->key);
    timers[TIMER_INIT_BOOTSTRAP] = clockNano();
    NCCLCHECKGOTO(bootstrapSplit(comm->commHash, comm, job->parent, job->color, job->key, parentRanks), res, fail);
    timers[TIMER_INIT_BOOTSTRAP] = clockNano() - timers[TIMER_INIT_BOOTSTRAP];
    // debug info, no commId was used
    commIdHash = 0;
  } else {
    // obtain a unique hash using the first commId
    comm->commHash = commIdHash = getHash(job->commId->internal, NCCL_UNIQUE_ID_BYTES);
    timers[TIMER_INIT_ALLOC] = clockNano();
    NCCLCHECKGOTO(commAlloc(comm, NULL, job->nranks, job->myrank), res, fail);
    timers[TIMER_INIT_ALLOC] = clockNano() - timers[TIMER_INIT_ALLOC];
    INFO(NCCL_INIT, "%s comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx commId 0x%llx - Init START", job->funcName,
         comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId, commIdHash);
    timers[TIMER_INIT_BOOTSTRAP] = clockNano();
    NCCLCHECKGOTO(bootstrapInit(job->nId, (struct ncclBootstrapHandle*)job->commId, comm), res, fail);
    timers[TIMER_INIT_BOOTSTRAP] = clockNano() - timers[TIMER_INIT_BOOTSTRAP];
  }
  comm->cudaArch = cudaArch;

  NCCLCHECKGOTO(initTransportsRank(comm, job->parent, timers), res, fail);

  // update communicator state
  comm->initState = ncclSuccess;
  timers[TIMER_INIT_TOTAL] = clockNano() - timers[TIMER_INIT_TOTAL];

  // Trace this call for replay tool
  if (job->parent) {
    /* unlink child abort flag. */
    __atomic_store_n(&job->parent->childAbortFlag, NULL, __ATOMIC_RELEASE);
    TRACE_CALL("ncclCommSplit(%p, %d, %d, %p, %d, %d)", job->parent, job->color, job->key, comm, comm->rank, comm->nRanks);
    INFO(NCCL_INIT, "%s comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx parent %p splitCount %d color %d key %d - Init COMPLETE", job->funcName,
         comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId, job->parent, job->splitCount, job->color, job->key);
  } else {
    // the name for the replay tool is ncclCommInitRank for all the variations
    TRACE_CALL("ncclCommInitRank(%p, %d, 0x%llx, %d, %d)", comm, comm->nRanks, commIdHash, comm->rank, comm->cudaDev);
    INFO(NCCL_INIT, "%s comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx commId 0x%llx - Init COMPLETE", job->funcName,
         comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId, commIdHash);
  }
  sum_timers = 0.0;
  for (int it = 1; it < TIMERS_INIT_COUNT; ++it)
    sum_timers += (timers[it] / 1e9);
  INFO(NCCL_INIT | NCCL_PROFILE,
       "Init timings - %s: rank %d nranks %d total %.2f (kernels %.2f, alloc %.2f, bootstrap %.2f, allgathers %.2f, topo %.2f, graphs %.2f, "
       "connections %.2f, rest %.2f)",
       job->funcName, comm->rank, comm->nRanks,
       timers[TIMER_INIT_TOTAL] / 1e9, timers[TIMER_INIT_KERNELS] / 1e9, timers[TIMER_INIT_ALLOC] / 1e9,
       timers[TIMER_INIT_BOOTSTRAP] / 1e9, timers[TIMER_INIT_ALLGATHER] / 1e9, timers[TIMER_INIT_TOPO] / 1e9,
       timers[TIMER_INIT_GRAPHS] / 1e9, timers[TIMER_INIT_CONNECT] / 1e9, timers[TIMER_INIT_TOTAL] / 1e9 - sum_timers);
exit:
  if (job->newcomm) {
    /* assign it to user pointer. */
    __atomic_store_n(job->newcomm, comm, __ATOMIC_RELEASE);
  }
  free(parentRanks);
  return res;
fail:
  comm->initState = res;
  goto exit;
}
```

**bootstrapInit 是个核心过程** <br>

bootstrapInit 是 NCCL 中负责**初始化通信组引导(strap)过程**的核心函数，其作用是**建立节点间的初始通信网络**。 <br>

```c++

ncclResult_t bootstrapInit(int nHandles, void* handles, struct ncclComm* comm) {
  ncclResult_t result = ncclSuccess;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  // char nextPeerHandle[NCCL_NET_HANDLE_MAXSIZE];
  struct bootstrapState* state;
  struct ncclSocket* proxySocket;
  struct ncclSocket sock, listenSockRoot;
  struct extInfo info = {0};
  union ringConnectInfo nextPeer;
  bool performRasAddRanks = true;
  struct rasRankInit* rasRanks = nullptr;

  uint64_t timers[BOOTSTRAP_INIT_TIME_N] = {0};

  // 分配并初始化 bootstrapState 结构体，存储通信组的元数据。
  NCCLCHECK(ncclCalloc(&state, 1));
  state->rank = rank;
  state->nranks = nranks;
  state->cudaDev = comm->cudaDev;
  state->abortFlag = comm->abortFlag;
  state->net = comm->ncclNet;
  comm->bootstrap = state;
  comm->magic = state->magic = BOOTSTRAP_HANDLE(handles, 0)->magic; // state and comm magic set to the first magic ID

  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d", rank, nranks);

  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_TOTAL]);
  // fill up the info
  info.nranks = nranks;
  info.nroots = nHandles;
  // get the ring connection info
  memset(&nextPeer, 0, sizeof(union ringConnectInfo));
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_CREATE]);
  if (ncclParamBootstrapNetEnable()) {
    // Create net interface for other ranks to contact me (all gather)
    NCCLCHECK(netGetDevice(rank, comm, &STATE_LISTEN(state, net.dev)));
    NCCLCHECK(state->net->listen(comm->netContext, STATE_LISTEN(state, net.dev), STATE_LISTEN(state, net.handle), &STATE_LISTEN(state, net.comm)));
    memcpy(info.connectInfo.handle, STATE_LISTEN(state, net.handle), NCCL_NET_HANDLE_MAXSIZE);
  } else {
    // create socket for ring neightbor to contact mee
    NCCLCHECK(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, socket), &info.connectInfo.addr, ncclSocketTypeBootstrap));
  }
  // Create socket for root to contact me using the root's magic
  int curr_root = rootIdFromRank(rank, nranks, nHandles);
  NCCLCHECK(createListenSocket(comm, BOOTSTRAP_HANDLE(handles, curr_root)->magic, &listenSockRoot, &info.listenRootAddress, ncclSocketTypeBootstrap));
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_CREATE]);

  // stagger connection times to avoid an overload of the root
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_DELAY]);
  int nRankRoot = nRankFromRoot(curr_root, nranks, nHandles);
  if (nRankRoot > ncclParamStaggerThreshold()) {
    // for socket the message rate in microsec
    double msg_rate = ncclParamStaggerRate() / 1.0e6;
    long musec = localIdFromRoot(rank, curr_root, nranks, nHandles) / msg_rate;
    struct timespec tv;
    long c_1e6 = 1e6;
    tv.tv_sec = musec / c_1e6;
    tv.tv_nsec = 1e3 * (musec % c_1e6);
    TRACE(NCCL_BOOTSTRAP, "rank %d delaying connection to root by %ld microsec", rank, musec);
    (void)nanosleep(&tv, NULL);
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_DELAY]);

  // send info on my listening socket to root
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_SEND]);
  // send contact info to my own root
  info.rank = rank;
  info.iroot = curr_root;
  NCCLCHECK(sendToRoot(BOOTSTRAP_HANDLE(handles, curr_root), comm, &info));
  // if needed, send the connection info to the previous root
  if (nHandles > 1 && isFirstFromRoot(rank, curr_root, nranks, nHandles)) {
    int prev_rank = BOOTSTRAP_PID(rank - 1, nranks);
    int prev_root = rootIdFromRank(prev_rank, nranks, nHandles);
    info.rank = prev_rank + 1; // my rank as seen by the previous root
    info.iroot = prev_root;
    NCCLCHECK(sendToRoot(BOOTSTRAP_HANDLE(handles, prev_root), comm, &info));
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_SEND]);

  // get info on my "next" rank in the bootstrap ring from root
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_RECV]);
  NCCLCHECK(ncclSocketInit(&sock));
  NCCLCHECK(ncclSocketAccept(&sock, &listenSockRoot));
  NCCLCHECK(socketRecv(&sock, &nextPeer, sizeof(nextPeer)));
  NCCLCHECK(ncclSocketClose(&sock));
  NCCLCHECK(ncclSocketClose(&listenSockRoot));
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_RECV]);

  // accept and connect the ring network
  if (ncclParamBootstrapNetEnable()) {
    NCCLCHECK(netRingConnect(comm->netContext, state->net, &state->listen, nextPeer.handle,
                             &STATE_RING(state, net.sendComm), &STATE_RING(state, net.sendDevHandle),
                             &STATE_RING(state, net.recvComm), &STATE_RING(state, net.recvDevHandle), state->abortFlag));
  } else {
    NCCLCHECK(socketRingConnect(&nextPeer.addr, &STATE_RING(state, socket.send), &STATE_LISTEN(state, socket), &STATE_RING(state, socket.recv), comm->magic, state->abortFlag));
  }

  // AllGather all listen handlers
  // in case of failure, those resources will be free'd when calling bootstrapDestroy, so we can return immediatly
  NCCLCHECK(ncclCalloc(&state->peerProxyAddresses, nranks));
  NCCLCHECK(ncclCalloc(&proxySocket, 1));
  NCCLCHECKGOTO(createListenSocket(comm, comm->magic, proxySocket, state->peerProxyAddresses + rank, ncclSocketTypeProxy), result, fail);

  NCCLCHECKGOTO(ncclCalloc(&state->peerProxyAddressesUDS, nranks), result, fail);
  NCCLCHECKGOTO(getUDS(state->peerProxyAddressesUDS + rank), result, fail);

  // create a socket for others to reach out (P2P)
  union ncclSocketAddress peerSocketAddress;
  NCCLCHECKGOTO(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, peerSocket), &peerSocketAddress, ncclSocketTypeBootstrap), result, fail);
  NCCLCHECKGOTO(ncclCalloc(&state->peerP2pAddresses, nranks), result, fail);
  memcpy(state->peerP2pAddresses + rank, &peerSocketAddress, sizeof(union ncclSocketAddress));

  // Initialize RAS
  if (ncclParamRasEnable() == 1) {
    // The RAS thread will take care of freeing the memory allocated below.
    NCCLCHECK(ncclCalloc(&rasRanks, nranks));
    memcpy(&rasRanks[rank].addr, &bootstrapNetIfAddr, sizeof(rasRanks[rank].addr));
    rasRanks[rank].pid = getpid();
    rasRanks[rank].cudaDev = comm->cudaDev;
    rasRanks[rank].nvmlDev = comm->nvmlDev;
    rasRanks[rank].hostHash = getHostHash();
    rasRanks[rank].pidHash = getPidHash();
    if (ncclRasCommInit(comm, rasRanks+rank) != ncclSuccess) {
      INFO(NCCL_INIT|NCCL_RAS, "Continuing in spite of a RAS initialization error");
      // We should still participate in the ringAllInfo below as the peers will be waiting for us.
      // Just make sure that the address is clearly invalid...
      memset(rasRanks+rank, '\0', sizeof(*rasRanks));
      performRasAddRanks = false;
    }
  }

  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_RING]);
  NCCLCHECKGOTO(ringAllInfo(comm, state, state->peerP2pAddresses, state->peerProxyAddresses, state->peerProxyAddressesUDS, rasRanks), result, fail);
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_RING]);

  // Create the service proxy and get the UDS
  NCCLCHECKGOTO(ncclProxyInit(comm, proxySocket, state->peerProxyAddresses, state->peerProxyAddressesUDS), result, fail);

  if (ncclParamRasEnable() == 1 && performRasAddRanks) {
    if (ncclRasAddRanks(rasRanks, nranks) != ncclSuccess)
      INFO(NCCL_INIT|NCCL_RAS, "Continuing in spite of a RAS initialization error");
  }

  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_TOTAL]);
  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d - DONE", rank, nranks);
  INFO(NCCL_BOOTSTRAP | NCCL_PROFILE, "Bootstrap timings total %f (create %f, send %f, recv %f, ring %f, delay %f)", timers[BOOTSTRAP_INIT_TIME_TOTAL] / 1e9,
       timers[BOOTSTRAP_INIT_TIME_CREATE] / 1e9,
       timers[BOOTSTRAP_INIT_TIME_SEND] / 1e9,
       timers[BOOTSTRAP_INIT_TIME_RECV] / 1e9,
       timers[BOOTSTRAP_INIT_TIME_RING] / 1e9,
       timers[BOOTSTRAP_INIT_TIME_DELAY] / 1e9);
exit:
  return result;
fail:
  free(proxySocket);
  goto exit;
}
```


