# 1 torch-elastic 背景
TE 最重要的是 Agent 和 Rendezvous 这两个概念。<br>
- Agent是运行在单节点上的独立后台进程，可以认为是 worker manager 或者 process supervisor，其负责启动worker，监控 worker 运行，捕获woker异常，通过 rendezvous 实现 worker 间的相互发现，当有成员变动时候负责基于 rendezvous 进行变更同步。
为了实现弹性训练，需要有一个节点/进程之间彼此发现的机制。<br>
- rendezvous就是这个发现机制或者说同步组件。当系统启动或者成员变更时候，所有worker会（重新）集合（rendezvous）以建立一个新的进程组。<br>

## 1.1 功能分离
TE 是围绕在 Rendezvous 基础之上的多个elastic agent构成，这是一种功能分离，让我们对比一下看看。<br>

**Agent 偏重具体节点上的逻辑。** <br>
- Agent 负责具体业务逻辑相关操作，比如启动进程执行用户程序，监控用户程序运行情况，如果有异常就通知 Rendezvous。<br>
- Agent 是一个 worker manager，负责启动/管理 workers 进程，组成一个 worker group，监控 workers 运行状态，捕获失效 workers，如果有故障/新加入worker，则重启 worker group。<br>
- Agent负责维护 WORLD_SIZE 以及 RANK 信息。用户不需要再手动提供，Agent会自动处理这些。<br>
- Agent 是具体节点上的后台进程，是独立个体。Agent自己无法实现整体上的弹性训练，所以需要一个机制来完成 worker 之间的相互发现，变更同步等等（WORLD_SIZE 和 RANK 这些信息其实也需要多个节点同步才能确定），这就是下面的 Rendezvous 概念。<br>

**Rendezvous 负责集群逻辑，保证节点之间对于""有哪些节点参与训练"达成强一致共识.** <br>
- 每一个 Agent 内部包括一个 **Rendezvous handler**，这些 handler 总体上构成了一个 **Rendezvous 集群**，从而构成了一个 **Agent 集群**。<br>
- Rendezvous 完成之后，会创建一个**共享键值存储(shared key-value store)**, 这个store实现了一个torch.distributed.Store API。此**存储仅由已完成Rendezvous的成员共享**, 它旨在让Torch Distributed Elastic在**初始化作业**过程之中**交换控制和数据信息**。<br>
- Rendezvous 负责**在每个agent之上维护当前 group 所有相关信息**。每个 agent 之上有一个 rendezvous，**它们会互相通信，总体维护一套信息**，这些信息存储在上面提到的Store 之中。<br>
- Rendezvous 负责集群逻辑相关，比如**新加入节点，移除节点，分配rank**等等。<br>

## 1.2 Rendezvous 概述
在 Torch Distributed Elastic 上下文之中，我们使用 rendezvous 这个术语来特指一个特定功能：一个结合了**对等发现(peer discovery)** 的分布式同步（distributed synchronization）原语。<br>

Rendezvous 被Torch Distributed Elastic用来收集一个训练job的参与者（节点），这样，参与者们可以**商议**得到参与者列表和每个参与者的**角色**，也可以对训练**何时开始/恢复做出一致的集体决定**。<br>

**Rendezvous 把功能分割解耦，业务逻辑被抽象成为一系列算子**，比如 _RendevzousJoinOp。而 Rendezvous 内部维护了一套**状态机**，由算子决定下一步操作。比如 _RendezvousOpExecutor 来执行各种算子，依据**算子结果得到下一步应该执行的 Action**，从而对本身进行操作。<br>

比如在 _DistributedRendezvousOpExecutor 之中，如果发现了当前 action 是 ADD_TO_WAIT_LIST，会执行 _add_to_wait_list，进而调用 self._state.wait_list.add(self._node). <br>

```python
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

# 2 Agent 总体逻辑
## 2.1 功能
Elastic agent 是 torchelastic 的控制台（control plane），**Elastic agent是一个独立进程，负责启动和管理底层 worker 进程**，代理具体负责：<br>
- 与PyTorch原生分布式协同工作：使每个worker都能获得所有需要的信息，以便成功调用**torch.distributed.init_process_group()**; <br>
- 容错：监控每个worker，当出现错误或者异常时能及时**终止所有worker并重启它们**; <br>
- 弹性：对成员更改作出反应，并**使用新的成员来重启所有workers**; <br>

## 2.2 工作基础
Torchelastic agent 和 用户 worker 依据故障切换契约来工作：<br>
- TE（torchelastic）希望用户worker以**5分钟为误差**完成工作。<br>
- 设计**DDP**应用程序时，**最好让所有worker都失败，而不只是一个worker失败**。<br>
- TE不会在代理之间同步重启次数(重启次数谁管谁的)。<br>
- TE **re-rendezvous不会减少重启次数(不会减少剩余重启次数)**。<br>
- 当单个代理完成其工作（成功或失败）时，它将**关闭rendezvous**。如果其他代理仍有worker在工作，他们将被**终止**。<br>
- 基于上述情况，如果至少有一个代理完成了任务，则缩容(scale down)**不起作用**。<br>
- 当代理检测到Scale up时，它不会减少 "max_restarts" (属于正常restart, 不会降低重启次数)。<br>
- Torchelast agent 之间通过etcd或者类似后端来保持协同工作。<br>

*注释：etcd是一个高可用的分布式键值存储系统，主要用于存储配置信息、服务发现、协调以及其他需要高度可用性的场景。* <br>

## 2.3 Agent UML 图
![agent uml](images/torchrun-agent.png)

## 2.4 agent 有多种配置方式

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;单的agent部署在每个节点上，并与本地进程协同工作。更高级的agent可以远程启动和管理workers。Agent可以做到彻底的去中心化，与其他agents（管理同一个job的workers）进行沟通协调做出一个集体性决策，决策是基于其管理的 workers 情况来完成。对于如何配置，源码中也给出了示例，如果在GPU上启动训练一个拥有 8 个 trainer（每GPU一个trainer）的 job，我们可以做如下可能的配置: <br>

```python
1. Use 8 x single GPU instances, place an agent per instance, managing 1 worker per agent.
2. Use 4 x double GPU instances, place an agent per instance, managing 2 workers per agent.
3. Use 2 x quad GPU instances, place an agent per instance, managing 4 workers per agent.
4. Use 1 x 8 GPU instance, place an agent per instance, managing 8 workers per agent.
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;基类ElasticAgent 是一个 Abstract Class，真正运行的代理都需要由此派生。从 ElasticAgent 的注释可知，代理进程负责管理一个或多个worker process。工作进程被**假定为常规分布式PyTorch脚本**。当worker进程由代理创建时，代理Agent将为worker进程提供必要的信息，以便正确初始化torch进程组。部署时，**精确的拓扑**和 agent-to-worker 比率取决于代理的具体实现和用户作业放置偏好。<br>

```python
class ElasticAgent(abc.ABC):
    """
    Agent process responsible for managing one or more worker processes.
    The worker processes are assumed to be regular distributed PyTorch scripts.
    When the worker process is created by the agent, the agent provides the
    necessary information for the worker processes to properly initialize
    a torch process group.

    The exact deployment topology and ratio of agent-to-worker is dependent
    on the specific implementation of the agent and the user's job placement
    preferences. 

    Usage
    ::

     group_result = agent.run()
      if group_result.is_failed():
        # workers failed
        failure = group_result.failures[0]
        log.exception(f"worker 0 failed with exit code : {failure.exit_code}")
      else:
        return group_result.return_values[0] # return rank 0's results

    """

    @abc.abstractmethod
    def run(self, role: str = DEFAULT_ROLE) -> RunResult:
        """
        Runs the agent, retrying the worker group on failures up to
        ``max_restarts``.

        Returns:
            The result of the execution, containing the return values or
            failure details for each worker mapped by the worker's global rank.

        Raises:
            Exception - any other failures NOT related to worker process
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_worker_group(self, role: str = DEFAULT_ROLE) -> WorkerGroup:
        """
        Returns:
            The ``WorkerGroup`` for the given ``role``.
            Note that the worker group is a mutable object and hence in a
            multi-threaded/process environment it may change state.
            Implementors are encouraged (but not required) to return
            a defensive read-only copy.
        """
        raise NotImplementedError()
```

- SimpleElasticAgent 实现了基类的部分函数，其目的是为了方便扩展新代理的实现。
- LocalElasticAgent 派生了SimpleElasticAgent ，是目前弹性训练最终使用的代理，主要用于在本地进行操作，负责管理单机上所有的worker进程。

# 3 Worker
## 3.1. Worker 类代表了一个worker实例，我们上文介绍了WorkerSpec，Worker 就是依据 WorkerSpec 构建出来的，其重点成员变量如下：<br>

- id（任意）：唯一标识一个worker，具体是由ElasticAgent的特定实现来解释，对于本地代理，它可以是worker的pid（int），对于远程代理，它可以被编码为``host:port（string）`。<br>
- local_rank ：worker的local rank。<br>
- global_rank：worker的global rank。<br>
- role_rank：具有相同角色的所有worker的rank。<br>
- world_size：全局worker数量。<br>
- role_world_size：具有相同角色的worker数量。<br>

## 3.2. WorkerGroup 代表了一个工作组，作为一个整体来管理多个 workers，进行批量处理. <br>

## 3.3. 在SimpleElasticAgent 初始化之中，会建立一个 WorkerGroup. <br>
```python
class SimpleElasticAgent(ElasticAgent):
    """
    An ``ElasticAgent`` that manages workers (``WorkerGroup``)
    for a single ``WorkerSpec`` (e.g. one particular type of worker role).
    """

    def __init__(self, spec: WorkerSpec, exit_barrier_timeout: float = 300):
        self._worker_group = WorkerGroup(spec)
        self._remaining_restarts = self._worker_group.spec.max_restarts
        self._store = None
        self._exit_barrier_timeout = exit_barrier_timeout
        self._total_execution_time = 0
```

## 3.4. WorkerState 表示 WorkerGroup的状态。工作组中的所有工作人员作为一个整体来维护/更改状态。如果工作组中的一个worker失败，则整个工作组被认为是失败：<br>

- UNKNOWN-代理**丢失了**对工作组状态的跟踪，无法恢复

- INIT-创建的工作组对象**尚未启动**

- HEALTHY-worker健康运行

- UNHEALTHY-worker在运行但是不健康

- STOPPED-代理停止（中断）worker

- SUCCEEDED-worker已完成运行(exit数值为0)

- FAILED-worker未能成功完成（exit数值不等于0)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;WorkerGroup从初始的INIT状态开始，然后进入"HEALTHY"或"UNHEALTHY"状态，最后到达终端"SUCCEEDED"或"FAILED"状态。工作组可以被代理打断并且临时置于"STOPPED"状态。处于"已停止"状态的工作进程可以在不久的将来被**调度重启**，被设置为已停止的状态的例子为：<br>
- 观察到工作组故障|不健康 <br>
- 检测到成员更改 <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;当**工作组上的操作（启动、停止、rdzv、重试等）失败**，并导致操作部分应用于工作组时，状态将为"**未知**"。这通常发生在**状态改变期间发生异常**，而且异常未捕获/未处理的情况下。当工作组处于"未知"状态，Agent**不会恢复工作组**，因此**最好终止作业**，并且**由job manager重试节点**。<br>

# 4 SimpleElasticAgent
## 4.1 总体运行
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SimpleElasticAgent 是 Agent 的实现类之一。此抽象是为了方便扩展新的 agent 实现。从后面可知，目前内置的 **LocalElasticAgent 负责管理单机上的所有 worker 进程**，如果用户希望只用**一个代理**就管理多机上所有的 worker，而不仅仅是本机 worker，那么可以通过扩展 SimpleElasticAgent 来实现一个自定义 Agent。<br>

SimpleElasticAgent 主循环 _invoke_run 是核心逻辑（这里默认代理和worker在同一个机器之上），其中做如下操作：<br>

1. 使用 self._initialize_workers(self._worker_group) 完成初始化工作，比如来启动 worker，为每个worker 分配 rank 等等。<br>
2. 然后进入 while True 循环，在循环之中通过 _monitor_workers 定期轮训用户程序运行情况，得到 worker 进程运行结果，然后依据情况进行不同处理。<br>
- 如果程序正常结束，则返回。<br>
- 如果程序出错，则重试，如果重试次数达到，结束workers。<br>
- 如果节点成员关系有变化，比如scale up就会有新的节点在waiting，这时候就重启所有workers。<br>

```python
def _invoke_run(self, role: str = DEFAULT_ROLE) -> RunResult:
    # NOTE: currently only works for a single role

    spec = self._worker_group.spec
    role = spec.role

    logger.info(
        "[%s] starting workers for entrypoint: %s", role, spec.get_entrypoint_name()
    )

    self._initialize_workers(self._worker_group)
    monitor_interval = spec.monitor_interval
    rdzv_handler = spec.rdzv_handler

    while True:
        assert self._worker_group.state != WorkerState.INIT
        time.sleep(monitor_interval)
        run_result = self._monitor_workers(self._worker_group)
        state = run_result.state
        self._worker_group.state = state

        put_metric(f"workers.{role}.remaining_restarts", self._remaining_restarts)
        put_metric(f"workers.{role}.{state.name.lower()}", 1)

        if state == WorkerState.SUCCEEDED:
            logger.info(
                "[%s] worker group successfully finished."
                " Waiting %s seconds for other agents to finish.",
                role,
                self._exit_barrier_timeout,
            )
            self._exit_barrier()
            return run_result
        elif state in {WorkerState.UNHEALTHY, WorkerState.FAILED}:
            if self._remaining_restarts > 0:
                logger.info(
                    "[%s] Worker group %s. "
                    "%s/%s attempts left;"
                    " will restart worker group",
                    role,
                    state.name,
                    self._remaining_restarts,
                    spec.max_restarts,
                )
                self._remaining_restarts -= 1
                self._restart_workers(self._worker_group)
            else:
                self._stop_workers(self._worker_group)
                self._worker_group.state = WorkerState.FAILED
                return run_result
        elif state == WorkerState.HEALTHY:
            # membership changes do not count as retries
            num_nodes_waiting = rdzv_handler.num_nodes_waiting()
            group_rank = self._worker_group.group_rank
            if num_nodes_waiting > 0:
                logger.info(
                    "[%s] Detected %s "
                    "new nodes from group_rank=%s; "
                    "will restart worker group",
                    role,
                    num_nodes_waiting,
                    group_rank,
                )
                self._restart_workers(self._worker_group)
        else:
            raise Exception(  # noqa: TRY002
                f"[{role}] Worker group in {state.name} state"
            )
```

## 4.2 初始化workers
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;代理主循环之中，首先使用 self._initialize_workers(self._worker_group) 来启动 worker。在 _initialize_workers之中：<br>
- 首先使用 self._rendezvous(worker_group) 进行节点之间的**同步共识操作以及rank处理**等等。<br>
- 其次调用 **_start_workers 启动 workers**。这里的 **_start_workers** 是虚函数，**需要派生类实现**。<br>

```python
    @prof
    def _initialize_workers(self, worker_group: WorkerGroup) -> None:
        r"""
        Starts a fresh set of workers for the worker_group.
        Essentially a rendezvous followed by a start_workers.

        The caller should first call ``_stop_workers()`` to stop running workers
        prior to calling this method.

        Optimistically sets the state of the worker group that
        just started as ``HEALTHY`` and delegates the actual monitoring
        of state to ``_monitor_workers()`` method
        """
        role = worker_group.spec.role

        # TODO after stopping workers, wait at least monitor_interval*2 for
        # workers on different nodes to fail on a collective op before waiting
        # on the rdzv barrier, this way we ensure that nodes enter rdzv
        # at around the same time and reduce false positive rdzv timeout errors
        self._rendezvous(worker_group) # 同步共识操作 

        worker_ids = self._start_workers(worker_group) # 启动worker
        for local_rank, w_id in worker_ids.items():
            worker = worker_group.workers[local_rank]
            worker.id = w_id

        worker_group.state = WorkerState.HEALTHY
```

## 4.3  _rendezvous
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们首先看看 **_rendezvous** , 其做如下操作：<br>
- 调用 **next_rendezvous()** 来处理成员关系**变化**，其会返回 world size，store等。<br>
- 会把 **store 配置到 workgroup** 之中，后续 worker 之间就可以**通过这个kvstore进行沟通**。<br>
- 调用 **_assign_worker_ranks 会生成 worker**，并且为 worker 建立 ranks，返回的 workers 都赋值在代理的 worker_group.workers 之中。<br>

```python
@prof
def _rendezvous(self, worker_group: WorkerGroup) -> None:
    r"""
    Runs rendezvous for the workers specified by worker spec.
    Assigns workers a new global rank and world size.
    Updates the rendezvous store for the worker group.
    """

    spec = worker_group.spec

    # 处理成员关系变化，注意，这里得到的是 group rank!
    store, group_rank, group_world_size = spec.rdzv_handler.next_rendezvous()
    self._store = store # store被设置到 Agent之中，store可以被认为是远端KV存储

    # 依据 group rank 为 worker 建立 ranks
    workers = self._assign_worker_ranks(store, group_rank, group_world_size, spec)
    worker_group.workers = workers
    worker_group.store = store
    worker_group.group_rank = group_rank
    worker_group.group_world_size = group_world_size

    if group_rank == 0:
        self._set_master_addr_port(store, spec.master_addr, spec.master_port)
    master_addr, master_port = self._get_master_addr_port(store)
    restart_count = spec.max_restarts - self._remaining_restarts
```

## 4.4 next_rendezvous
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Elastic 调用 rdzv_handler.next_rendezvous() 来处理成员关系变化，目的是启动下一轮 rendezvous 操作（因为本worker已经启动，需要加入集群）。<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;注意，next_rendezvous 是 RendezvousHandler 的内部函数。这一函数调用会被阻塞，直到 worker 的数量达到了要求。在 worker 被初始化，或者重启的时候，这一函数都会被调用。当函数返回时，不同的 worker group 会以返回中的 rank 作为唯一的标示。其内部逻辑是：<br>
- 先使用_RendezvousExitOp让该node退出。<br>
- 然后再使用_RendezvousJoinOp把该node重新加入。<br>
- 最后启动心跳，返回world size，store等。<br>

```python
def next_rendezvous(self) -> RendezvousInfo:
    """See base class."""
    msg = (
        f"The node '{self._this_node}' attempts to join the next round of the rendezvous "
        f"'{self._settings.run_id}'."
    )
    self._record(message=msg)
    logger.info(msg)

    try:
        self._stop_heartbeats()

        # Delay the execution for a small random amount of time if this is our
        # first run. This will slightly skew the rendezvous attempts across the
        # nodes and reduce the load on the backend.
        if self._state_holder.state.round == 0:
            _delay(seconds=(0, 0.3))

        exit_op = _RendezvousExitOp()
        join_op = _RendezvousJoinOp()

        deadline = self._get_deadline(self._settings.timeout.join)
        self._op_executor.run(exit_op, deadline)
        self._op_executor.run(join_op, deadline, self._get_deadline)

        self._start_heartbeats()

        rank, world_size = self._get_world()
        store = self._get_store()

    except Exception as e:
        self._record(
            message=f"{type(e).__name__}: {str(e)}",
            node_state=NodeState.FAILED,
        )
        raise
```

## 4.5 为worker 分配ranks
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;接着是调用 _assign_worker_ranks 为 worker 建立 ranks。分配 rank 算法如下：<br>
- 每个代理将其配置（组排名、组全局大小、工作线程数）写入公共存储。<br>
- rank0的代理从存储中读取所有角色信息，并确定每个代理的工作线程排名。<br>
- 确定global rank：工作线程的global rank是通过计算其前面所有工作线程的本地全局大小（local_world_size）的累加和来得出的。为了提高效率，每个工作线程被分配一个基础全局排名，使其工作线程的范围在[基础全局排名, 基础全局排名 + 本地全局大小)之间。<br>
- 确定role rank：role rank的确定方法与第3点中的算法相同，不同之处在于rank是根据role名称来计算的。<br>
- rank0的agent将分配好的排名写入存储。<br>
- 每个代理从存储中读取分配好的rank。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;然后生成 workers，把 worker 都赋值在 worker_group.workers 之中。<br>

```python
def _assign_worker_ranks(
    self, store, group_rank: int, group_world_size: int, spec: WorkerSpec
) -> List[Worker]:
    """Determine proper ranks for worker processes.

    The rank assignment is done according to the following algorithm:

    1. Each agent writes its configuration(group_rank, group_world_size
        , num_workers) to the common store.
    2. The rank 0 agent reads all the role_info from the store and
        determines each agents worker ranks.
    3. Determine the global rank: the global rank of the workers is computed
        by cumulative sum of the local_world_size for all workers in front of it.
        For efficiency reasons each worker is assigned a base global rank
        such that it's workers are in the range [base_global_rank,
        base_global_rank + local_world_size).
    4. Determine the role rank: The role rank is determined using the algorithms
        in the point 3 with the exception that the ranks are calculated with
        respect to the role name.
    5. The rank 0 agent writes the assigned ranks to the store.
    6. Each agent reads the assigned ranks from the store.

    Time complexity: each worker O(1), rank0 O(n), overall O(n)
    """

    ROLE_INFO_PREFIX = "torchelastic/role_info/"
    ASSIGNED_RANKS_PREFIX = "torchelastic/assigned_ranks/"

    agent_role_info = _RoleInstanceInfo(
        spec.role, group_rank, spec.local_world_size
    )
    store.set(f"{ROLE_INFO_PREFIX}{group_rank}", agent_role_info.serialize())

    # tcp store is collocated with rank 0 so we can use it to do extra compute to reduce overall # of operations.
    # TCP存储与排名为0的节点在空间或逻辑上是相邻或共同配置的, 这意味着它们之间的数据访问或通信可能会更加高效。
    if group_rank == 0:
        role_infos_bytes = store.multi_get(
            [f"torchelastic/role_info/{i}" for i in range(group_world_size)]
        )
        role_infos = [
            _RoleInstanceInfo.deserialize(info_bytes)
            for info_bytes in role_infos_bytes
        ]

        role_sizes = defaultdict(lambda: 0)
        global_size = 0
        for role_info in role_infos:
            role_sizes[role_info.role] += role_info.local_world_size
            global_size += role_info.local_world_size

        base_global_rank = 0
        role_ranks = defaultdict(lambda: 0)

        keys = []
        values = []
        for i, role_info in enumerate(role_infos):
            keys.append(f"{ASSIGNED_RANKS_PREFIX}{i}")
            values.append(
                json.dumps(
                    [
                        base_global_rank,
                        global_size,
                        role_ranks[role_info.role],
                        role_sizes[role_info.role],
                    ]
                )
            )

            base_global_rank += role_info.local_world_size
            role_ranks[role_info.role] += role_info.local_world_size

        store.multi_set(keys, values)

    # get will block until the data is available in the store.
    (
        base_global_rank,
        global_world_size,
        base_role_rank,
        role_world_size,
    ) = json.loads(store.get(f"{ASSIGNED_RANKS_PREFIX}{group_rank}"))

    workers = []
    for local_rank in range(spec.local_world_size):
        worker = Worker(
            local_rank=local_rank,
            global_rank=base_global_rank + local_rank,
            role_rank=base_role_rank + local_rank,
            world_size=global_world_size,
            role_world_size=role_world_size,
        )
        workers.append(worker)
    return workers
```

## 4.6 启动 workers 进程
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;派生类的 _start_workers 来启动 worker 进程，因此基类这里没有实现, 在LocalElasticAgent里有实现。<br>

**注意：这里会为envs[local_rank] 设定具体的 worker_env 环境变量，并在此环境变量下启动 start_processes** <br>

```python
@prof
def _start_workers(self, worker_group: WorkerGroup) -> Dict[int, Any]:
    spec = worker_group.spec
    store = worker_group.store
    assert store is not None
    restart_count = spec.max_restarts - self._remaining_restarts

    use_agent_store: bool = spec.rdzv_handler.use_agent_store
    logger.info("use_agent_store: %s", use_agent_store)

    args: Dict[int, Tuple] = {}
    envs: Dict[int, Dict[str, str]] = {}
    log_line_prefixes: Optional[Dict[int, str]] = (
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
    )

    return self._pcontext.pids()
```

**到目前为止逻辑如下：** <br>
```python
+--------------------------------------------------+
| LocalElasticAgent                                |         _initialize_workers
|                                                  |                 +
|                                                  |                 |
|                                                  |                 |
|   +----------------------+                       |                 v
|   |WorkerGroup           |                       |         _rendezvous(worker_group)
|   |                      |                       |                 +
|   |     spec             |                       |                 |
|   |                      |                       |                 | 1
|   |     group_world_size |                       |                 v
|   |                      |                       |        rdzv_handler.next_rendezvous()
|   |     store            |                       |                 +
|   |                      |    +----------------+ |                 |
|   |     group_rank       |    | Worker0(rank 0)| |               2 | ranks
|   |                      |    | Worker1(rank 1)| |  Workers        v
|   |     workers  +----------> | ...            | | <----+ _assign_worker_ranks
|   |                      |    | Workern(rank n)| |    3
|   +----------------------+    +----------------+ |
|                                                  |
+--------------------------------------------------+
```

## 4.7 _assign_worker_ranks 内部细节
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Agent使用_RoleInstanceInfo与其他agent交换信息。该信息用于确定本agent workers的rank。<br>

```python
class _RoleInstanceInfo:
    """The class is used by the agent to exchange the information with other agents.

    The information is used to determine the rank of the workers that agent
    manages in heterogeneous environments, where different agents can have
    different number of workers.
    """

    __slots__ = ["role", "rank", "local_world_size"]

    def __init__(self, role: str, rank: int, local_world_size: int):
        r"""Initialize the agent class instance.

        Args:
            role (str): user-defined role for the workers with this spec
            rank (int): the rank of the agent
            local_world_size (int): number of local workers to run
        """
        self.role = role
        self.rank = rank
        self.local_world_size = local_world_size
```

接下来，在各个代理之间同步，得到角色的总体信息。每个代理将其配置（group_rank, group_world_size , num_workers）写入公共存储。这里就是使用之前 Rendezvous 返回的 store 来进行信息共享。<br>

```python
agent_role_info = _RoleInstanceInfo(
    spec.role, group_rank, spec.local_world_size
)

# 每个agent 都将 agent_role_info 写到公共存储store
store.set(f"{ROLE_INFO_PREFIX}{group_rank}", agent_role_info.serialize())

# tcp store is collocated with rank 0 so we can use it to do extra compute to reduce overall # of operations.
# tcp存储与秩0共置(最接近)，因此我们可以利用它进行额外计算以减少总操作数。
if group_rank == 0:
    role_infos_bytes = store.multi_get(
        [f"torchelastic/role_info/{i}" for i in range(group_world_size)]
    )
    role_infos = [
        _RoleInstanceInfo.deserialize(info_bytes)
        for info_bytes in role_infos_bytes
    ]

    role_sizes = defaultdict(lambda: 0)
    global_size = 0
    for role_info in role_infos:
        role_sizes[role_info.role] += role_info.local_world_size
        global_size += role_info.local_world_size

    base_global_rank = 0
    role_ranks = defaultdict(lambda: 0)

    keys = []
    values = []
    for i, role_info in enumerate(role_infos):
        keys.append(f"{ASSIGNED_RANKS_PREFIX}{i}")
        values.append(
            json.dumps(
                [
                    base_global_rank,
                    global_size,
                    role_ranks[role_info.role],
                    role_sizes[role_info.role],
                ]
            )
        )

        base_global_rank += role_info.local_world_size
        role_ranks[role_info.role] += role_info.local_world_size

    store.multi_set(keys, values)
```

**至此总的逻辑为：** <br>
1. 调用 rdzv_handler.next_rendezvous() 来和其它节点进行同步，获得信息。<br>
2. 获得信息中的store（可以认为就是远端的KV存储），group_world_size，group_rank 传给 Agent。<br>
3. ranks 等信息传给 _assign_worker_ranks方法。<br>
4. _assign_worker_ranks 之中，调用 store.set、store.multi_get、store.multi_set在各个代理之间同步，得到角色的总体信息。每个代理将其配置（group_rank, group_world_size , num_workers）写入公共KV存储。<br>
5. 依据 role infos 来确定全局rank：当前代理的global rank是 本代理 的 group_rank 在infos数组的偏移量（offset）。偏移量的计算方法是，排名低于group_rank的所有代理的local_world之和。<br>
6. 使用各种信息建立一系列的 Workers。<br>
7. Workers 被复制给 Agent 的 WorkerGroup 之中。<br>

```python
                                                              _initialize_workers
                                                                      +
                                                                      |
                                                                      |
                                                                      v
                                                              _rendezvous(worker_group)
                                                                      +
+----------------------------------------------+                      |
| LocalElasticAgent                            |                      | 1
|                                              |   2                  v
|                                         +--------------+  rdzv_handler.next_rendezvous()
| +--------------------+                  |    |                      +
| | WorkerGroup        |                  |    |                      |
| |                    |                  |    |                    3 | ranks
| |                    |                  |    |                      v
| |  spec              |                  |    |       +--------------+------------------+
| |                    |                  |    |       | _assign_worker_ranks            |
| |                    |                  |    |       |                                 |
| |  store   <----------------------------+    |       |                        4        |
| |                    |                  |    |       | role_infos = store.set          |
| |                    |                  |    |       |               +                 |
| |  group_world_size<--------------------+    |       |               | 5               |
| |                    |                  |    |       |               |                 |
| |                    |                  |    |       |               v                 |
| |  group_rank <-------------------------+    |       |           store.multi_get       |
| |                    |                       |       |    store.multi_set(keys, values)|
| |                    |   +----------------+  |       |               +                 |
| |  workers  +----------->+ Worker0(rank 0)|  |       |               |                 |
| |                    |   | Worker1(rank 1)|  |       |               | 6               |
| |                    |   | ...            |  |Workers|               v                 |
| |                    |   | Workern(rank n)+<------------+ new Worker(local_rank,       |
| +--------------------+   +----------------+  |    7  |               global_rank,      |
|                                              |       |               role_rank,        |
+----------------------------------------------+       |               world_size,       |
                                                       |               role_world_size)  |
                                                       |                                 |
                                                       +---------------------------------+
```

## 4.8 barrier
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;实际上，几乎不可能保证DDP的所有 worker 都能保证同时结束，所以因此TE提供了一个finalization barrier，这个barrier的作用是对worker finalization 实施等待超时（5分钟）。<br>

```python
def _exit_barrier(self):
    """
    Wait for ``exit_barrier_timeout`` seconds for all agents to finish
    executing their local workers (either successfully or not). This
    acts as a safety guard against user scripts that terminate at different
    times. This barrier keeps the agent process alive until all workers finish.
    """
    start = time.time()
    try:
        store_util.barrier(
            self._store,
            self._worker_group.group_rank,
            self._worker_group.group_world_size,
            key_prefix=_TERMINAL_STATE_SYNC_ID,
            barrier_timeout=self._exit_barrier_timeout,
        )
    except Exception:
        log.exception(
            f"Error waiting on exit barrier. Elapsed: {time.time() - start} seconds"
        )
```

# 5 LocalElasticAgent
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;LocalElasticAgent 是弹性训练最终使用的代理，主要用于在本地进行操作，负责管理单机上所有的worker进程，其派生自 SimpleElasticAgent。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;此agent在**每个主机之上部署**，并配置为生成n个工作进程。当使用GPU时，**n是主机上可用的GPU数量**。本地agent**不会**与部署在其他主机上的其他本地agent通信，即使worker可以在主机间通信。Worker id被解释为本地进程。**agent作为把本机所有worker process作为一个整体启动和停止**。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;传递给worker的函数和参数必须与python multiprocessing兼容。要将multiprocessing数据结构传递给worker，用户可以在与指定的start_method相同的多处理multiprocessing中创建数据结构，并将其作为函数参数传递。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;exit_barrier_timeout用来**指定等待其他agent完成的时间量(以秒为单位)**。这起到了一个安全网的作用，可以处理worker在不同时间完成的情况，以防止agent将提前完成的worker视为scale-down事件。**强烈建议用户代码确保worker以同步方式终止，而不是依赖于exit_barrier_timeout**。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;SimpleElasticAgent 主要是提供给了初始化和总体运行方式，但是遗留了一些抽象函数没有被实现，比如_start_workers，_stop_workers，_monitor_workers，_shutdown。LocalElasticAgent 就补齐了这些函数。<br>

# 6 整体运行流程

```python
def trainer(args) -> str:
    return "do train"

def main():
    start_method="spawn"
    shared_queue= multiprocessing.get_context(start_method).Queue()
    spec = WorkerSpec(
                role="trainer",
                local_world_size=nproc_per_process,
                entrypoint=trainer,
                args=("foobar",),
                ...<OTHER_PARAMS...>)
    agent = LocalElasticAgent(spec, start_method)
    results = agent.run()

    if results.is_failed():
        print("trainer failed")
    else:
        print(f"rank 0 return value: {results.return_values[0]}")
        # prints -> rank 0 return value: do train
```

# 7 _initialize_workers 里 _start_workers 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_start_workers 方法会调用 start_processes 来启动 worker 进程，默认_start_method 是 "spawn"。也就是启动了多个进程，并行执行用户程序。同时这些进程的运行结果会被监控。start_processes 参数之中，entrypoint和args 是用户命令和参数，entrypoint可以是函数或者字符串。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_start_workers 把 start_processes 方法启动多进程的结果保存在 **_pcontext ：PContext** 之中，**后续就用 _pcontext 来继续控制**，比如结束 worker 就是直接调用 _pcontext 的 close方法。<br>

```python
def _start_workers(self, worker_group: WorkerGroup) -> Dict[int, Any]:
    spec = worker_group.spec
    store = worker_group.store
    assert store is not None
    restart_count = spec.max_restarts - self._remaining_restarts

    use_agent_store: bool = spec.rdzv_handler.use_agent_store
    logger.info("use_agent_store: %s", use_agent_store)

    args: Dict[int, Tuple] = {}
    envs: Dict[int, Dict[str, str]] = {}
    log_line_prefixes: Optional[Dict[int, str]] = (
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
    )

    return self._pcontext.pids()
```

# 8 monitor ： 监控
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;运行之后，TE 会调用 _monitor_workers 对workers进行监控。之前把启动多线程的结果保存在 _pcontext 之中，现在就**用 _pcontext 对运行情况进行监控**。<br>

```python
@prof
def _monitor_workers(self, worker_group: WorkerGroup) -> RunResult:
    role = worker_group.spec.role
    worker_pids = {w.id for w in worker_group.workers}
    assert self._pcontext is not None
    pc_pids = set(self._pcontext.pids().values())
    if worker_pids != pc_pids:
        logger.error(
            "[%s] worker pids do not match process_context pids."
            " Expected: %s, actual: %s",
            role,
            worker_pids,
            pc_pids,
        )
        return RunResult(state=WorkerState.UNKNOWN)

    result = self._pcontext.wait(0)
    if result:
        if result.is_failed():
            # map local rank failure to global rank
            worker_failures = {}
            for local_rank, failure in result.failures.items():
                worker = worker_group.workers[local_rank]
                worker_failures[worker.global_rank] = failure
            return RunResult(
                state=WorkerState.FAILED,
                failures=worker_failures,
            )
        else:
            # copy ret_val_queue into a map with a global ranks
            workers_ret_vals = {}
            for local_rank, ret_val in result.return_values.items():
                worker = worker_group.workers[local_rank]
                workers_ret_vals[worker.global_rank] = ret_val
            return RunResult(
                state=WorkerState.SUCCEEDED,
                return_values=workers_ret_vals,
            )
    else:
        return RunResult(state=WorkerState.HEALTHY)
```

# 9 目前，整体逻辑梳理
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;因为启动和监控涉及到系统整体运行逻辑，需要和 rendezvous 一起才能更好理解，所以我们把这部分的分析推迟，等到 Rendezvous 之后再来做整体分析。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;目前总体逻辑如下：<br>
- 调用 rdzv_handler.next_rendezvous() 来和其他节点进行同步，获得信息。<br>
- 获得信息中的store（可以认为就是远端的KV存储），group_world_size，group_rank 传给 Agent。<br>
- ranks 等信息传给 _assign_worker_ranks方法。<br>
- _assign_worker_ranks 之中，调用 _share_and_gather 在各个代理之间同步，得到角色的总体信息。每个代理将其配置（group_rank, group_world_size , num_workers）写入公共KV存储。<br>
- 依据 role infos 来确定全局rank：当前代理的global rank是 本代理 的 group_rank 在infos数组的偏移量（offset）。偏移量的计算方法是，排名低于- -- group_rank的所有代理的local_world之和。<br>
- 使用各种信息建立一系列的 Workers。<br>
- Workers 被复制给 Agent 的 WorkerGroup 之中。<br>
- 使用 _start_workers 来启动 worker 进程。<br>
- 把 worker 进程 id 赋值给 Agent 的 worker.id 之中，这样以后就可以用 worker.id 来操作进程。<br>
- 使用 _monitor_workers 监控 worker 进程。<br>
- 使用 _exit_barrier 来等待 worker 进程结束。<br>

```python
                                                              _initialize_workers
                                                                      +
                                                                      |
                                                                      |
                                                                      v
                                                              _rendezvous(worker_group)
                                                                      +
+----------------------------------------------+                      |
| LocalElasticAgent                            |                      | 1
|                                              |   2                  v
|                                         +--------------+  rdzv_handler.next_rendezvous()
| +--------------------+                  |    |                      +
| | WorkerGroup        |                  |    |                      |
| |                    |                  |    |                    3 | ranks
| |                    |                  |    |                      v
| |  spec              |                  |    |       +--------------+------------------+
| |                    |                  |    |       | _assign_worker_ranks            |
| |                    |                  |    |       |                                 |
| |  store   <----------------------------+    |       |                        4        |
| |                    |                  |    |       | role_infos = store.set          |
| |                    |                  |    |       |               +          store) |
| |  group_world_size<--------------------+    |       |               | 5               |
| |                    |                  |    |       |               |                 |
| |                    |                  |    |       |               v                 |
| |  group_rank <-------------------------+    |       |          store.multi_set        |
| |                    |                       |       |          store.multi_get        |
| |                    |   +----------------+  |       |               +                 |
| |  workers  +----------->+ Worker0(rank 0)|  |       |               |                 |
| |                    |   | Worker1(rank 1)|  |       |               | 6               |
| |                    |   | ...            |  |Workers|               v                 |
| |                    |   | Workern(rank n)+<------------+ new Worker(local_rank,       |
| +--------------------+   +---------+------+  |    7  |               global_rank,      |
|                                    ^         |       |               role_rank,        |
|                                    |         |       |               world_size,       |
|                                    |         |       |               role_world_size)  |
+----------------------------------------------+       |                                 |
                                     |                 +---------------+-----------------+
                                     |                                 |
                                     |                                 | 8
                                     |              9                  v
                                     +-----------------------+   _start_workers
                                                                       +
                                                                       | 10
                                                                       |
                                                                       v
                                                       +---------------+--------------+
                                                       | state = _monitor_workers     |
                                                  +--> |                              +-->
                                                  |    +---------------+--------------+  |
                                                  |                    |                 |
                                                  <--------------------------------------+
                                                     LOOP  Every 30S   |
                                                                       | 11
                                                                       v
                                                                    _exit_barrier
```

