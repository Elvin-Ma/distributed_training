# PGNCCL 支持 ACE 吗？
- NCCL 新的 zero-CTA / Copy Engine collective offload 路径，要求 buffer 是 symmetric memory 的，需要用 registered window 注册。

# Copy Engine 必须用 symmetric-memory 吗？
1. Copy engine 这个硬件本身不要求 symmetric memory。普通 cudaMemcpyAsync/P2P copy **也可能走 copy engine**。
2. **传统 NCCL collective**不要求 symmetric memory。常规 ncclAllReduce / allGather 等一般用 NCCL kernel 占 SM 做搬运/规约，不需要用户 buffer 对称注册。
3. NCCL 2.27+ / 2.28+ 里宣传的 Copy Engine collectives、zero-CTA 通信 offload**要求使用 symmetric memory window**。[NVIDIA 文档](https://developer.nvidia.com/blog/fusing-communication-and-compute-with-new-device-api-and-copy-engine-collectives-in-nvidia-nccl-2-28/)说 device API relies on symmetric memory；NVIDIA blog 也明确说新 API 要用 symmetric memory windows。PyTorch 文档里也写到：NCCL collective 在 symmetric memory tensors + zero-CTA policy 下，数据移动会 offload 到 GPU copy engines。


# 传统模式会用 Copy Engine 吗？
一般来说：传统 NCCL collective 不会用 copy engine 来执行 collective 主数据路径，它通常是 launch NCCL CUDA kernels，占用 SM 来做 copy / reduce / reduce-copy。

细分一下：

- 非规约型 collective，比如部分 broadcast / allgather / sendrecv，逻辑上只是搬数据，但传统 NCCL 仍通常通过 NCCL kernel 搬，不是像 cudaMemcpyAsync 那样直接交给 CE。
- 规约型 collective，比如 allreduce / reduce_scatter，本来就需要做 reduce，传统实现更自然是用 SM 执行 load、reduce、store。
- NCCL 内部有 “Copy” primitive，但这个 “Copy” 是 NCCL kernel 里的通信原语名，不等价于 GPU hardware copy engine。
- 真正的 CE collective/offload 是新路径，需要 symmetric memory/window registration + zero-CTA 之类配置。NVIDIA 文档也把 “zero-CTA optimization supports using the Copy Engine” 放在
buffer registration / symmetric memory 语境下。

# NCCL Copy Engine 的重要配置
- NCCL_CTA_POLICY_* 控制的是 NCCL communicator 在执行 collective 时**愿意占用多少 GPU CTA/SM 资源**，核心目标是在“通信吞吐”和“给计算 kernel 留资源”之间取舍。

**NCCL_CTA_POLICY_DEFAULT**
  - 默认策略。
  - NCCL 自己根据 topology、message size、collective 类型等选择 CTA 使用量。
  - 目标通常是 通信性能/整体性能最大化。
  - 传统 NCCL collective 大多会 launch communication kernel，占用一定 SM/CTA。
  - 适合普通训练、benchmark 前的默认选择。

**NCCL_CTA_POLICY_ZERO**
  - 激进地要求 NCCL：只要能不用 CTA，就尽量不用 CTA。
  - 目标是把通信从 SM 上挪走，给计算 kernel 最大限度保留 CTA/SM 资源。
  - 在满足条件时，NCCL 可以走 zero-CTA optimization，也就是用 GPU Copy Engine 做通信。
  - 可能牺牲单独看 NCCL collective 的吞吐/latency，因为它优先保留计算资源，而不是让通信最快。

  但 ZERO 不是魔法开关。它只是告诉 NCCL “能 zero-CTA 就 zero-CTA”。真正走 CE/zero-CTA 还需要条件，例如 NCCL 文档里列的：
  - CUDA driver >= 12.5
  - collective 在单个 NVL / MNNVL domain 内，不能走 IB/ROCE 网络
  - buffer 通过 NCCL window 做了 **symmetric registration**
  - **communicator 配了 NCCL_CTA_POLICY_ZERO**
  - collective 类型受支持，目前文档列的是 **AlltoAll、AllGather、Scatter、Gather**

  **环境变量：NCCL_CTA_POLICY**
  - NCCL_CTA_POLICY是 `环境变量版`的 CTA policy 配置，用来给 NCCL communicator 指定 CTA 使用策略。它会影响 NCCL collective 是否倾向于占用 SM/CTA，或者尽量走 zero-CTA / Copy Engine 路径。
    - 默认策略(NCCL 自己决定 CTA 使用，偏默认性能): export NCCL_CTA_POLICY=0 / export NCCL_CTA_POLICY=DEFAULT
    - efficiency 策略(更偏效率/资源节省，减少不必要 CTA 使用): export NCCL_CTA_POLICY=1 / export NCCL_CTA_POLICY=EFFICIENCY
    - zero-CTA 策略(要求 NCCL 尽量使用 zero-CTA 路径): export NCCL_CTA_POLICY=2 / export NCCL_CTA_POLICY=ZERO
