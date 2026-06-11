# 1 集群进程相互依赖性

## 1.1 节点内
- 在普通 torchrun 集群训练里，如果某个 worker 进程异常退出，torchrun 的策略是：这个 worker group 失败。

- 同一台机器上的其他 worker，会被本机 agent 主动杀掉。

其他机器上的 worker，通常不是由这台机器的 agent 直接杀掉，而是因为通信断开、collective 报错、store/rendezvous 状态变化等，随后各自节点上的 agent 也会监控到失败，然后停止本机 worker。也就是说是“每个节点的 agent 管自己本地的进程”。

## 1.2 节点间 ： 其他节点的 ranks 会很快感知通信断开

如果某个 rank 进程真的退出了，其他节点上的 ranks 正在跑 collective，比如 allreduce，底层通信通常会很快发现 peer 断开、socket close、NCCL communicator abort、store 连接异常等。

这类情况不是“rank 卡住不响应”，而是**连接明确断了**，所以`不需要`等 init_process_group(timeout=...) 里的 ProcessGroup timeout。

## 1.3 timeout 与 network error
- 重点: timeout 时经常发生在退出进程通信操作已经下发，但kernel 还没执行; 此时对应进程支持到通信kernel 时不会报网络错误.

| 现象 | 更可能 |
| --- | --- |
| 某 rank 日志里更早有 exception / OOM / killed / abort | network / remote-exit |
| 所有 rank 进程都还在，但卡住很久 | timeout |
| 一个 rank 提前退出，其他 rank 随后报错 | network / remote-exit |
| rank 间 if 分支不同、通信次数不同 | timeout |
| collective 顺序不一致 | timeout |
| 节点/网卡明确断开、连接 reset | network error |
| 网络不返回错误只是无进展 | timeout |
| 某 rank 卡在 dataloader/checkpoint/CPU 逻辑 | timeout |
| 某 rank 被 kill -9 | network / remote-exit |

# 2 进程建立与运行

```sh
root  32233   995  1 19:25 pts/2  00:00:12 /usr/bin/python /usr/local/bin/torchrun --standalone --nproc_per_node=2 test_allreduce.py
root  32372 32233  0 19:25 ?      00:00:04 /usr/bin/python -u test_allreduce.py
root  32373 32233  0 19:25 ?      00:00:04 /usr/bin/python -u test_allreduce.py
```

## 2.1 进程启动

- 每台机器上都会有一个 agent 进程
- 每个 agent 只负责启动自己机器上的 local workers
- 多个 agent 通过 rendezvous/store 协调成一个集群
- 示意图：
```sh
node0:
  LocalElasticAgent
    -> 启动本机 rank 0, rank 1, ...

node1:
  LocalElasticAgent
    -> 启动本机 rank 2, rank 3, ...

node2:
  LocalElasticAgent
    -> 启动本机 rank 4, rank 5, ...

master/rendezvous 节点：
  负责协调大家怎么组成集群

每台机器上的 agent：
  负责启动自己机器上的 worker 子进程
```

- 具体的两个启动类:
```py
class SimpleElasticAgent(ElasticAgent): #控制子进程的启动和多进程的循环监控.
class LocalElasticAgent(SimpleElasticAgent): #实际实现进程的启动.
```

- torchrun 的 agent 进程会起两个thread

```sh
Thread 32233 (idle): "MainThread"
    _invoke_run (torch/distributed/elastic/agent/server/api.py:881)
    run (torch/distributed/elastic/agent/server/api.py:717)
    wrapper (torch/distributed/elastic/metrics/api.py:138)
    launch_agent (torch/distributed/launcher/api.py:284)
    __call__ (torch/distributed/launcher/api.py:156)
    run (torch/distributed/run.py:927)
    main (torch/distributed/run.py:936)
    wrapper (torch/distributed/elastic/multiprocessing/errors/__init__.py:357)
    <module> (torchrun:6)
Thread 32370 (idle): "RendezvousKeepAliveTimer_0"
    wait (threading.py:324)
    wait (threading.py:607)
    _run (torch/distributed/elastic/rendezvous/utils.py:277)
    run (threading.py:953)
    _bootstrap_inner (threading.py:1016)
    _bootstrap (threading.py:973)
```

# 3 Agent子进程的启动和监控：在 MainThread 中完成

- LocalElasticAgent: _start_workers 启动子进程

## 3.1 启动子进程

```py
    @prof
    def _start_workers(self, worker_group: WorkerGroup) -> dict[int, Any]:
        spec = worker_group.spec
        store = worker_group.store
        assert store is not None
        restart_count = spec.max_restarts - self._remaining_restarts

        use_agent_store: bool = spec.rdzv_handler.use_agent_store
        logger.info("use_agent_store: %s", use_agent_store)

        args: dict[int, tuple] = {}
        envs: dict[int, dict[str, str]] = {}
        log_line_prefixes: Optional[dict[int, str]] = (
            {} if self._log_line_prefix_template else None
        )
        for worker in worker_group.workers:
            local_rank = worker.local_rank
            worker_env = {
                "LOCAL_RANK": str(local_rank),
                "RANK": str(worker.global_rank),
                "GROUP_RANK": str(worker_group.group_rank),
                "ROLE_RANK": str(worker.role_rank),
                "ROLE_NAME": spec.role,
                "LOCAL_WORLD_SIZE": str(spec.local_world_size),
                "WORLD_SIZE": str(worker.world_size),
                "GROUP_WORLD_SIZE": str(worker_group.group_world_size),
                "ROLE_WORLD_SIZE": str(worker.role_world_size),
                "MASTER_ADDR": worker_group.master_addr,
                "MASTER_PORT": str(worker_group.master_port),
                "TORCHELASTIC_RESTART_COUNT": str(restart_count),
                "TORCHELASTIC_MAX_RESTARTS": str(spec.max_restarts),
                "TORCHELASTIC_RUN_ID": spec.rdzv_handler.get_run_id(),
                "TORCHELASTIC_USE_AGENT_STORE": str(use_agent_store),
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": os.getenv(
                    "TORCH_NCCL_ASYNC_ERROR_HANDLING", str(1)
                ),
            }
            if "OMP_NUM_THREADS" in os.environ:
                worker_env["OMP_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]

            if self._log_line_prefix_template:
                log_line_prefix = Template(
                    self._log_line_prefix_template
                ).safe_substitute(
                    role_name=spec.role,
                    rank=worker.global_rank,
                    local_rank=local_rank,
                )
                log_line_prefixes[local_rank] = log_line_prefix

            envs[local_rank] = worker_env
            worker_args = list(spec.args)
            worker_args = macros.substitute(worker_args, str(local_rank))
            args[local_rank] = tuple(worker_args)

        self._setup_local_watchdog(envs=envs)
        self._setup_healthcheck()

        assert spec.entrypoint is not None
        assert self._logs_specs is not None
        self._pcontext = start_processes(
            name=spec.role,
            entrypoint=spec.entrypoint,
            args=args,
            envs=envs,
            logs_specs=self._logs_specs,
            log_line_prefixes=log_line_prefixes,
            start_method=self._start_method,
            numa_options=spec.numa_options,
        )

        return self._pcontext.pids()
```

## 3.2 启动和监控子进程代码
- SimpleElasticAgent: _invoke_run begin process and moniter

```py
    def _invoke_run(self, role: str = DEFAULT_ROLE) -> RunResult:
        # NOTE: currently only works for a single role

        spec = self._worker_group.spec
        role = spec.role

        self._initialize_workers(self._worker_group)
        monitor_interval = spec.monitor_interval
        rdzv_handler = spec.rdzv_handler

        while True:
            assert self._worker_group.state != WorkerState.INIT
            # 为了避免循环一直高速空转，占满 CPU。monitor_interval 就是监控间隔，比如每隔 0.1 秒、1 秒或几秒检查一次 worker 状态。
            time.sleep(monitor_interval)
            # self.workers = [Worker(local_rank=i) for i in range(self.spec.local_world_size)]
            # self._worker_group : 属性 self.workers

            run_result = self._monitor_workers(self._worker_group)
            state = run_result.state
            self._worker_group.state = state

            if state == WorkerState.SUCCEEDED:
                return run_result
            elif state in {WorkerState.UNHEALTHY, WorkerState.FAILED}:
                if self._remaining_restarts > 0:
                    self._restart_workers(self._worker_group)
                else:
                    self._stop_workers(self._worker_group)
                    self._worker_group.state = WorkerState.FAILED
                    return run_result
            elif state == WorkerState.HEALTHY:
                self._restart_workers(self._worker_group)
            else:
                raise Exception(  # noqa: TRY002
                    f"[{role}] Worker group in {state.name} state"
                )
```

## 3.3 monitor_workers 检查是否有子进程异常

```sh
worker 异常
 -> record() 捕获异常
 -> 写 error.json
 -> 父进程发现 exitcode 非 0 / multiprocessing exception
 -> 创建 ProcessFailure(error_file=...)
 -> ProcessFailure 读取 error.json
 -> 放进 RunProcsResult.failures
```

## 3.4 底层进程管理与控制逻辑
- _poll 中会查询每个进程的状态值，并返回给父进程；
- _close 中会尝试所有进程退出；

```python
class SubprocessContext(PContext):
    """``PContext`` holding worker processes invoked as a binary."""

    def __init__(
        self,
        name: str,
        entrypoint: str,
        args: dict[int, tuple],
        envs: dict[int, dict[str, str]],
        logs_specs: LogsSpecs,
        log_line_prefixes: Optional[dict[int, str]] = None,
        numa_options: Optional[NumaOptions] = None,
    ):
        super().__init__(
            name,
            entrypoint,
            args,
            envs,
            logs_specs,
            log_line_prefixes,
        )

        # state vector; _vdone[local_rank] -> is local_rank finished or not
        self._running_local_ranks: set[int] = set(range(self.nprocs))
        self._failures: dict[int, ProcessFailure] = {}
        self.subprocess_handlers: dict[int, SubprocessHandler] = {}
        self._numa_options: Optional[NumaOptions] = numa_options

    def _start(self):
        if self.subprocess_handlers:
            raise ValueError(
                "The subprocess handlers already initialized. Most likely the start method got called twice."
            )
        self.subprocess_handlers = {
            local_rank: get_subprocess_handler(
                entrypoint=self.entrypoint,  # type: ignore[arg-type] # entrypoint is always a str
                args=self.args[local_rank],
                env=self.envs[local_rank],
                stdout=self.stdouts[local_rank],
                stderr=self.stderrs[local_rank],
                local_rank_id=local_rank,
                numa_options=self._numa_options,
            )
            for local_rank in range(self.nprocs)
        }

    def _poll(self) -> Optional[RunProcsResult]:
        done_local_ranks = set()
        for local_rank in self._running_local_ranks:
            handler = self.subprocess_handlers[local_rank]
            exitcode = handler.proc.poll()
            if exitcode is not None:
                done_local_ranks.add(local_rank) # 标记已经有结果的进程
                if exitcode != 0:  # failed or signaled
                    self._failures[local_rank] = ProcessFailure(
                        local_rank=local_rank,
                        pid=handler.proc.pid,
                        exitcode=exitcode,
                        error_file=self.error_files[local_rank],
                    )
                # else: --> succeeded; nothing to do

        self._running_local_ranks.difference_update(done_local_ranks) # 剩余的进程

        # if ALL procs are finished or ANY have failed
        if not self._running_local_ranks or self._failures:
            self.close()  # terminate all running procs
            # 所有进程的结果都打包到这里
            result = RunProcsResult(
                failures=self._failures,
                stdouts=self.stdouts,
                stderrs=self.stderrs,
            )
            if result.is_failed():
                first_failure = min(result.failures.values(), key=lambda f: f.timestamp)
                logger.error(
                    "failed (exitcode: %s) local_rank: %s (pid: %s) of binary: %s",
                    first_failure.exitcode,
                    first_failure.local_rank,
                    first_failure.pid,
                    self.entrypoint,
                )
            else:
                # Populate return with dummy values. This provides consistency with MultiprocessingHandler
                result.return_values = dict.fromkeys(range(self.nprocs))

            return result
        else:  # there are no failures and procs still running
            return None

    def pids(self) -> dict[int, int]:
        return {
            local_rank: sh.proc.pid
            for local_rank, sh in self.subprocess_handlers.items()
        }

    def _close(self, death_sig: signal.Signals, timeout: int = 30) -> None:
        if not self.subprocess_handlers:
            return
        for handler in self.subprocess_handlers.values():
            if handler.proc.poll() is None:
                logger.warning(
                    "Sending process %s closing signal %s",
                    handler.proc.pid,
                    death_sig.name,
                )
                handler.close(death_sig=death_sig)
        end = time.monotonic() + timeout
        for handler in self.subprocess_handlers.values():
            time_to_wait = end - time.monotonic()
            if time_to_wait <= 0:
                break
            try:
                handler.proc.wait(time_to_wait)
            except subprocess.TimeoutExpired:
                # Ignore the timeout expired exception, since
                # the child process will be forcefully terminated via SIGKILL
                pass
        for handler in self.subprocess_handlers.values():
            if handler.proc.poll() is None:
                logger.warning(
                    "Unable to shutdown process %s via %s, forcefully exiting via %s",
                    handler.proc.pid,
                    death_sig,
                    _get_kill_signal(),
                )
                handler.close(death_sig=_get_kill_signal())
                handler.proc.wait()
```

# 4. Agent 监控线程的工作原理

- SimpleElasticAgent 里 初始化进程时会调用 _rendezvous();
- _rendezvous()调用 rdzv_handler.next_rendezvous()； # rdzv_handler 类型是 DynamicRendezvousHandler
- next_rendezvous() 在 rendezvous 成功加入之后调用 _start_heartbeats()；_start_heartbeats() 创建后台线程；后台线程之后周期性调用 _keep_alive()。

```sh
Thread 32370 (idle): "RendezvousKeepAliveTimer_0"
    wait (threading.py:324)
    wait (threading.py:607)
    _run (torch/distributed/elastic/rendezvous/utils.py:277)
    run (threading.py:953)
    _bootstrap_inner (threading.py:1016)
    _bootstrap (threading.py:973)
```

## 4.1 rendezvous 子线程的启动
```sh
ElasticAgent._invoke_run()
  -> ElasticAgent._initialize_workers()
    -> ElasticAgent._rendezvous(worker_group)
      -> spec.rdzv_handler.next_rendezvous()
        -> DynamicRendezvousHandler.next_rendezvous()
          -> self._stop_heartbeats()
          -> self._op_executor.run(_RendezvousExitOp)
          -> self._op_executor.run(_RendezvousJoinOp)
          -> self._start_heartbeats()
            -> _PeriodicTimer(...)
            -> timer.set_name("RendezvousKeepAliveTimer_0")
            -> timer.start()
              -> Thread(target=_PeriodicTimer._run, ...).start()
```

重要步骤：
1. DynamicRendezvousHandler 通过 _start_heartbeats 函数借助_PeriodicTimer 来启动子线程执行 **自己 _keep_alive_weak 方法**;
2. DynamicRendezvousHandler 将 _keep_alive_weak 赋值给 _PeriodicTimer.ctx.function;
3. _PeriodicTimer.start() 启动线程，并调用 _PeriodicTimer._run();
4. _PeriodicTimer._run() 中调用 ctx.function(*ctx.args, **ctx.kwargs) --> 需要传入 DynamicRendezvousHandler 作为  weak_self 参数;
5. ctx.function 就是 staticmethod DynamicRendezvousHandler._keep_alive_weak;


## 4.2 rendezvous 子线程里的函数调用栈

```sh
# 调用栈
_keep_alive_weak()
  -> _keep_alive()
  -> _op_executor.run(_RendezvousKeepAliveOp)
  -> sync()

# 功能栈
等 keep_alive_interval
  -> _keep_alive()
  -> run(_RendezvousKeepAliveOp)
  -> sync() 拉取最新共享 state
  -> _RendezvousKeepAliveOp 判断是否该心跳
  -> 如果需要心跳：更新 last_heartbeats[self_node]
  -> mark_dirty()
  -> 下一轮 run while 开头 sync()
  -> set_state / compare_set 写回共享 state
```

## 4.3 周期性调用过程

- dynamic rendezvous 的心跳监控循环;
- 心跳线程的周期驱动循环启动;

```py
class _PeriodicTimer:
    @staticmethod
    def _run(ctx) -> None:
        while not ctx.stop_event.wait(ctx.interval):
            ctx.function(*ctx.args, **ctx.kwargs)
```

# 5 rdzv-backend 的选择

## 5.1 torch 中几种不同的store
```sh
默认 torchrun                    → StaticTCPRendezvous
torchrun --rdzv-backend=static   → StaticTCPRendezvous
torchrun --rdzv-backend=c10d     → DynamicRendezvousHandler
torchrun --standalone            → DynamicRendezvousHandler + c10d backend
torchrun --rdzv-backend=etcd-v2  → DynamicRendezvousHandler + etcd backend
```

| 启动方式 | rdzv_backend | RendezvousHandler | 底层 Store | 典型用途 |
| --- | --- | --- | --- | --- |
| 默认 `torchrun ...` | `static` | `StaticTCPRendezvous` | `TCPStore + PrefixStore` | 普通单机/多机，兼容老式 `--master-addr/--master-port` |
| `torchrun --rdzv-backend=static ...` | `static` | `StaticTCPRendezvous` | `TCPStore + PrefixStore` | 显式使用 static rendezvous |
| `torchrun --rdzv-backend=c10d ...` | `c10d` | `DynamicRendezvousHandler` | 默认 `TCPStore`，也可配 `FileStore` | 动态 rendezvous，不依赖外部 etcd |
| `torchrun --standalone ...` | `c10d` | `DynamicRendezvousHandler` | `TCPStore`，端口自动选择 | 单机快速启动，自动生成 `rdzv_id` 和 endpoint |
| `torchrun --rdzv-backend=etcd-v2 ...` | `etcd-v2` | `DynamicRendezvousHandler` | etcd-backed Store | 多机弹性训练，需要外部 etcd |
| `torchrun --rdzv-backend=etcd ...` | `etcd` | `EtcdRendezvousHandler` | `EtcdStore` | 旧 etcd rendezvous 路径，偏 legacy |

> 默认情况下 --rdzv-backend=static, 但--standalone 时用的是 c10d

- static rendezvous 后端: 适合固定节点训练
```sh
启动前必须知道所有节点数量和每个节点的 node_rank。
rank/world_size 是静态计算出来的。
节点数量不能动态变化。
没有 rendezvous 心跳线程。
没有 wait_list，num_nodes_waiting() 永远是 0。
有节点挂了，只能由 agent 检测本地 worker 失败并按 max_restarts 重启；不会因为新节点加入而重新 rendezvous。
更接近传统分布式启动方式。
```

- c10d rendezvous 后端: 适合 elastic / 动态 rendezvous
```sh
使用 DynamicRendezvousHandler。
支持 min_nodes:max_nodes，例如 --nnodes=2:4。
节点 rank/world_size 由 rendezvous 过程动态分配。
rendezvous 完成后有心跳线程：RendezvousKeepAliveTimer_x。
能发现 rendezvous 成员超时失活。
支持新节点进入等待队列。
agent 主循环会检查 num_nodes_waiting()，如果有新节点等待，会重启 worker group 并重新 rendezvous。
可以配合 max_restarts 做弹性恢复。
```


## 5.2 agent 中启动store

- 重点: Agent 中的Store world_size 是节点的个数，而非进程的个数

| 位置 | Store 用途 | world_size 含义 |
| --- | --- | --- |
| `StaticTCPRendezvous.next_rendezvous()` | Agent rendezvous / elastic control plane | 节点数，也就是 Agent 数 |
| `init_process_group(env://)` | 训练进程初始化 ProcessGroup | worker 进程总数 |
| worker 环境变量 `WORLD_SIZE` | 传给训练脚本 | worker 进程总数 |
| worker 环境变量 `GROUP_WORLD_SIZE` | Agent group size | 节点数 |

```py
class StaticTCPRendezvous(RendezvousHandler):
    def next_rendezvous(self) -> RendezvousInfo:
        logger.info("Creating TCPStore as the c10d::Store implementation")
        is_master = self.rank == 0
        if not self._store:
            self._store = TCPStore(  # type: ignore[call-arg]
                self.master_addr,
                self.master_port,
                self.world_size,
                is_master,
                self.timeout,
                multi_tenant=True,
            )
        store = PrefixStore(self.run_id, self._store)
        # TCPStore server instance is used by trainer code
        bootstrap_store_info = RendezvousStoreInfo(self.master_addr, self.master_port)
        return RendezvousInfo(
            store,
            self.rank,
            self.world_size,
            bootstrap_store_info,
        )
```

## 5.3 子进程中的store

- TORCHELASTIC_USE_AGENT_STORE = True (默认) 时，子进程 init_process_group 直接用agent store;
```py
TCPStore(
    host_name=hostname,
    port=port,
    world_size=world_size,
    is_master=False,
    timeout=timeout,
)
```

# 6 DynamicRendezvousHandler 信息共享机制

- 心跳逻辑

```sh
utils.py 里的 _PeriodicTimer._run 是心跳线程的周期驱动循环；
dynamic_rendezvous.py 里的 _keep_alive / _RendezvousKeepAliveOp 是具体心跳逻辑。
```

- 监控线程调用逻辑

```sh
_PeriodicTimer._run()
  -> _keep_alive_weak()
  -> _keep_alive()
  -> _op_executor.run(_RendezvousKeepAliveOp)
  -> _state_holder.sync()
  -> 如果需要心跳，更新 state.last_heartbeats[self_node]
  -> mark_dirty()
  -> 下一轮 sync 写回 Store
```

## 6.1 从 ElasticAgent 到 DynamicRendezvousHandler

```sh
class LocalElasticAgent(SimpleElasticAgent)._worker_group : class WorkerGroup
class WorkerGroup.spec : class WorkerSpec
class WorkerSpec.rdzv_handler : class RendezvousHandler --> DynamicRendezvousHandler
```

## 6.2 基本数据结构
- 共享的信息 state
```sh
class _RendezvousState:
    round: int                                 # 第几次 rendezvous
    complete: bool                             # 当前 rendezvous 是否完成，完成的条件是 participants 的数量达到 min_nodes，并且在 deadline 之前
    deadline: Optional[datetime]               # 如果 complete 是 False，那么 deadline 是 rendezvous 期望完成的时间点；如果 complete 是 True，那么 deadline 是 None
    closed: bool                               # 共享状态close : run_id 的 rendezvous 是否关闭，关闭的条件是调用了 set_closed() 或 shutdown()，一旦关闭就不再接受新的节点加入
    participants: dict[_NodeDesc, int]         # 正式成员
    wait_list: set[_NodeDesc]                  # 等待下一轮加入的节点，会触发 worker group re-rendezvous
    redundancy_list: set[_NodeDesc]            # 冗余节点，可以在下一轮加入而不会触发 re-rendezvous
    last_heartbeats: dict[_NodeDesc, datetime] # 包含 participants、wait_list 和 redundancy_list 中每个节点的最后一次心跳时间
```

- _BackendRendezvousStateHolder 来管理state

```py
class _BackendRendezvousStateHolder(_RendezvousStateHolder):
    def __init__(
        self,
        backend: RendezvousBackend,
        settings: RendezvousSettings,
        cache_duration: int = 1,
    ) -> None:
        self._backend = backend
        self._state = _RendezvousState()
        self._settings = settings
        self._cache_duration = cache_duration
        self._token = None
        self._dirty = False
        self._last_sync_time = -1
        self._dead_nodes = []

    def sync(self) -> Optional[bool]:

    def _sanitize(self) -> None:

```

- 由 C10dRendezvousBackend 来负责 store 信息共享

```sh
class C10dRendezvousBackend(RendezvousBackend):
    """Represents a C10d-backed rendezvous backend.

    Args:
        store:
            The :py:class:`torch.distributed.Store` instance to use to
            communicate with the C10d store.
        run_id:
            The run id of the rendezvous.
    """

    # See the explanation in the __init__ method.
    _NULL_SENTINEL = "Y2FuaW1hZGFt"

    _store: Store
    _key: str

    def __init__(self, store: Store, run_id: str) -> None:
        if not run_id:
            raise ValueError("The run id must be a non-empty string.")

        self._store = store

        self._key = "torch.rendezvous." + run_id

        # The read operation of a store blocks the caller until the specified
        # key becomes available. This behavior makes it tricky to use a store
        # as a regular key-value dictionary.
        #
        # As a workaround we initially set a sentinel value as the rendezvous
        # state. Whenever this value gets returned we treat it as a None.
        self._call_store("compare_set", self._key, "", self._NULL_SENTINEL)
```


## 6.3 周期性sync
```python
def sync(self) -> Optional[bool]:
        """See base class."""
        state_bits: Optional[bytes] = None

        token = None

        has_set: Optional[bool]

        if self._dirty:
            has_set = False

            state_bits = pickle.dumps(self._state)

            set_response = self._backend.set_state(state_bits, self._token)
            if set_response is not None:
                state_bits, token, has_set = set_response
        else:
            has_set = None

            if self._cache_duration > 0:
                # Avoid overloading the backend if we are asked to retrieve the
                # state repeatedly. Try to serve the cached state.
                if self._last_sync_time >= max(
                    time.monotonic() - self._cache_duration, 0
                ):
                    return None

            get_response = self._backend.get_state()
            if get_response is not None:
                state_bits, token = get_response

        if state_bits is not None:
            try:
                self._state = pickle.loads(state_bits)
            except pickle.PickleError as exc:
                raise RendezvousStateError(
                    "The rendezvous state is corrupt. See inner exception for details."
                ) from exc
        else:
            self._state = _RendezvousState()

        if has_set and self._dead_nodes and logger.isEnabledFor(logging.DEBUG):
            node_list = ", ".join(f"'{dead_node}'" for dead_node in self._dead_nodes)

            msg = (
                f"As part of the sync operation the node(s) {node_list} have been removed from the "
                f"rendezvous '{self._settings.run_id}' since they had no heartbeat."
            )
            self._record(message=msg)
            logger.debug(msg)

        self._token = token

        self._dirty = False

        self._last_sync_time = time.monotonic()

        self._sanitize()

        return has_set
```

## 6.4 CAS : compare and set

```py
    base64_state: bytes = self._call_store(
        "compare_set", self._key, token, base64_state_str
    )
```

- state --> token 格式转化

```sh
1. state_bits = pickle.dumps(self._state) # 转成pickle二进制格式
2. b64_token :str = b64encode(state_bits) # base64 编码 --> 转成 base64 文本形式（还是bytes）但内容已变成base64字节文本序列 --> eg ：b64encode(b"hello")＝b"aGVsbG8="
3. local_token : = b64_token.decode()     # bytes -> str 的字符解码 : b"aGVsbG8=".decode() = "aGVsbG8="
```

- CAS 语义

```sh
语法结构: compare_set(key, expected_value, desired_value)

如果 remote Store[key] 当前值 == expected_value:
    Store[key] = desired_value
否则:
    返回 remote 当前已有的 state 对应的token
    Store[key] 保持不变, 但 expected_value 会替换为 self._token = token


最后返回 remote Store[key] 当前值 : state_bits, token
```

# 7 state 的动态调整过程

## DynamicRendezvousHandler 通过 _op_executor 来调整

```py
class DynamicRendezvousHandler(RendezvousHandler):
    self._op_executor = _DistributedRendezvousOpExecutor(
        self._this_node, self._state_holder, self._settings
    )
```

## _DistributedRendezvousOpExecutor 中 run 来调整

> 调整后必须 mark_dirty

```py
class _DistributedRendezvousOpExecutor(_RendezvousOpExecutor):
    def run(
        self,
        state_handler: Callable[[_RendezvousContext, float], _Action],
        deadline: float,
        update_deadline: Optional[Callable[[timedelta], float]] = None,
    ) -> None:
        """See base class."""
        action = None
        while action != _Action.FINISH:
            # Reads or writes the latest rendezvous state shared by all nodes in
            # the rendezvous. Note that our local changes might get overridden
            # by another node if that node synced its changes before us.
            has_set = self._state_holder.sync()
            if has_set is not None:
                if has_set:
                    msg = (
                        f"The node '{self._node}' has successfully synced its local changes with "
                        f"other nodes in the rendezvous '{self._settings.run_id}'."
                    )
                else:
                    msg = (
                        f"The node '{self._node}' has a stale state and failed to sync its local "
                        f"changes with other nodes in the rendezvous '{self._settings.run_id}'."
                    )

                self._record(message=msg)
                logger.debug(msg)

            self._state = self._state_holder.state

            ctx = _RendezvousContext(self._node, self._state, self._settings)

            # Determine the next action to take based on the current state of
            # the rendezvous.
            action = state_handler(ctx, deadline)

            if action == _Action.FINISH:
                continue

            if action == _Action.ERROR_CLOSED:
                raise RendezvousClosedError

            if action == _Action.ERROR_TIMEOUT:
                raise RendezvousTimeoutError

            if action == _Action.SYNC:
                # Delay the execution by one second to avoid overloading the
                # backend if we are asked to poll for state changes.
                _delay(seconds=1)
            else:
                if action == _Action.KEEP_ALIVE:
                    self._keep_alive()
                elif action == _Action.ADD_TO_PARTICIPANTS:
                    self._add_to_participants()
                elif action == _Action.ADD_TO_WAIT_LIST:
                    self._add_to_wait_list()
                elif action == _Action.ADD_TO_REDUNDANCY_LIST:
                    self._add_to_redundancy_list()
                elif action == _Action.REMOVE_FROM_PARTICIPANTS:
                    self._remove_from_participants()
                elif action == _Action.REMOVE_FROM_WAIT_LIST:
                    self._remove_from_wait_list()
                elif action == _Action.REMOVE_FROM_REDUNDANCY_LIST:
                    self._remove_from_redundancy_list()
                    # update deadline since the node may participate in rendezvous process
                    if update_deadline:
                        deadline = update_deadline(self._settings.timeout.join)
                elif action == _Action.MARK_RENDEZVOUS_COMPLETE:
                    self._mark_rendezvous_complete()
                elif action == _Action.MARK_RENDEZVOUS_CLOSED:
                    self._mark_rendezvous_closed()

                # Attempt to sync our changes back to other nodes.
                self._state_holder.mark_dirty()
```

# 8 几个相关周期

## 8.1. Agent 主循环检测周期：默认 0.1s

```py
# /torch/distributed/elastic/agent/server/api.py
class SimpleElasticAgent(ElasticAgent):
    def _invoke_run(self)
        # agent 大约每 0.1s 检查一次 worker 状态
        monitor_interval = spec.monitor_interval

        while True:
            time.sleep(monitor_interval)
            run_result = self._monitor_workers(self._worker_group)
```

## 8.2 Rendezvous state backend 同步/读取周期：默认最多约 1s 一次
- self._cache_duration = 1.0 s;
- 让本地 handler 持有一份短暂缓存的 rendezvous state，避免 backend(TCPStore) 频繁同步;
- 目的: 避免 agent 高频调用 num_nodes_participating() / num_nodes_waiting() 时把 rendezvous backend 打爆;
- 结果：因此虽然 agent -> monitor_interval 默认 0.1s 检查一次，但实际从 backend 拉取最新 rdzv state 通常是 1s 级别.

```py
    # torch/distributed/elastic/rendezvous/dynamic_rendezvous.py
class _BackendRendezvousStateHolder:
    def sync(self) -> Optional[bool]:
        if self._last_sync_time >= max(
            time.monotonic() - self._cache_duration, 0
        ):
            return None
```

## 8.3. Rendezvous 心跳周期：默认 5s；死亡清理阈值默认约 15s

- 心跳更新的本质是写当前节点的 last_heartbeats[node] 时间戳;
- 如果一个进程出问题了，他就写不进去最新时间戳，就会认为该进程挂了，从而触发 rdzv 死亡检测.

```py
class DynamicRendezvousHandler(RendezvousHandler):
    keep_alive_interval: int = 5,
    keep_alive_max_attempt: int = 3,

# 心跳更新的本质是写当前节点的 last_heartbeats[node] 时间戳 --> 如果一个进程出问题了，他就写不进去最新时间戳，就会认为该进程挂了，从而触发 rdzv 死亡检测。
class _DistributedRendezvousOpExecutor
    def _keep_alive(self) -> None:
        msg = (
            f"The node '{self._node}' updated its keep-alive heartbeat time for the rendezvous "
            f"'{self._settings.run_id}'. Pending sync."
        )
        self._record(message=msg)
        logger.debug(msg)

        self._state.last_heartbeats[self._node] = datetime.now(timezone.utc)
```

- 死亡判定窗口：默认 15s
```py
class _BackendRendezvousStateHolder(_RendezvousStateHolder):
    def _sanitize(self) -> None:
        state = self._state

        expire_time = datetime.now(timezone.utc) - (
            self._settings.keep_alive_interval * self._settings.keep_alive_max_attempt
        )

        # Filter out the dead nodes.
        self._dead_nodes = [
            node
            for node, last_heartbeat in state.last_heartbeats.items()
            if last_heartbeat < expire_time
        ]
```

# 9 rank assign

##　8.1 两种方式 : fast path & slow path
```sh
_assign_worker_ranks 的目标是给本地 workers 分配 global_rank、role_rank、world_size、role_world_size。

Fast Path:
  假设所有节点完全同构。
  不访问 Store。
  直接用 group_rank 和 local_rank 算。
  启动最快，但适用范围窄。

Slow Path:
  默认路径。
  每个 agent 把 role/local_world_size 写入 Store。
  rank0 汇总所有 agent 信息，计算 rank，再写回 Store。
  支持异构 role 和异构 worker 数，但大规模下有额外 Store 协调成本。
```

两种路径对比

| 项目 | Fast Path | Slow Path |
| --- | --- | --- |
| 默认启用 | 否 | 是 |
| 触发条件 | `TORCH_ELASTIC_WORKER_IDENTICAL=1` | 默认 |
| 是否访问 Store | 不访问 | 访问 |
| 是否需要 rank0 汇总 | 不需要 | 需要 |
| 支持不同 role | 不支持 | 支持 |
| 支持不同 local_world_size | 不支持 | 支持 |
| 复杂度 | `O(1)` | `O(n)` |
| 大规模启动开销 | 很低 | 随 agent 数增加 |
| 适用场景 | 同构单 role 训练 | 通用 elastic 场景 |

## 9.2 切换方式

TORCH_ELASTIC_WORKER_IDENTICAL=1 时会启动 Fast Path，适合这种情况：

```sh
所有节点 GPU 数一致
每个节点启动相同数量 worker
所有 worker 都是同一个 role
典型 DDP / FSDP / DeepSpeed 单 role 训练
大规模集群，例如千卡、万卡
频繁 elastic restart，希望减少 Store 协调开销
```

## 9.3 slow patch 耗时估计

- 1250 个 agent 时，需要：
```sh
1250 次 store.set(role_info)
rank0 multi_get 1250 个 role_info
rank0 计算 1250 个 agent 的 rank
rank0 multi_set 1250 个 assigned_ranks
1250 个 agent 各自 store.get(assigned_ranks)
```

实际多出来的时间大概可能是:

```sh
理想 TCPStore / 网络稳定：
  几百毫秒 ~ 1 秒左右

普通大规模环境：
  1 ~ 3 秒比较可能

Store 忙、网络抖动、rank0 压力大：
  数秒甚至更高
```

# 10 torchrun 日志控制
- --redirects=3
- --tee=3

```sh
torchrun_args=(
  --nnodes="${NNODES}"
  --nproc_per_node="${NPROC_PER_NODE}"
  --node_rank="${NODE_RANK}"
  --rdzv_backend="${RDZV_BACKEND}"
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}"
  --rdzv_id="${RDZV_ID}"
)

if [[ -n "${TORCHRUN_LOG_DIR}" ]]; then
  mkdir -p "${TORCHRUN_LOG_DIR}"
  torchrun_args+=(
    --log_dir="${TORCHRUN_LOG_DIR}"
    # 0: 不重定向。stdout/stderr 默认直接走控制台;
    # 1: 重定向 stdout 到文件(仅写stdout.log);
    # 2: 重定向 stderr 到文件(仅写stderr.log);
    # 3: 重定向 stdout 和 stderr 到文件(stdout 和 stderr 分别写入两个文件);
    --redirects=3
    # 0: 不 tee, 日志如果被 redirect 到文件，就只写文件，不实时打印回终端;
    # 1: tee stdout。stdout 会写到 stdout.log，同时实时打印到终端;
    # 2: tee stderr。stderr 会写到 stderr.log，同时实时打印到终端;
    # 3: tee stdout 和 stderr。stdout 和 stderr 会写到两个文件，同时实时打印到终端;
    --tee=1
  )
fi
```
