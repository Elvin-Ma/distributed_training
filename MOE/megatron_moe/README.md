# 1 MOE 概述
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MoE（混合专家）是在Megatron-Core框架中实现的一种强大的大型语言模型（LLM）架构，旨在提高大型语言模型的效率和可扩展性。它利用专家并行性，允许多个专家分布在不同的工作节点上，**每个工作节点处理不同的训练样本批次**。这种方法显著提高了计算吞吐量，使模型能够实现高性能指标，例如在H100上使用BF16训练8个70亿参数模型时达到47%的MFU（模型实际使用的浮点运算能力占硬件平台理论最大计算能力的比例）。<br>

**MoE的关键特性：** <br>

- 并行性技术：MoE结合了多种并行策略，包括专家并行性、数据并行性、张量并行性、序列并行性、管道并行性和上下文并行性。这种组合使得能够有效处理更大的模型变体。

- 路由和负载均衡：系统采用先进的路由机制，如Top-K路由器，并利用负载均衡算法来优化专家之间的令牌（token）分配。

- 性能优化：诸如GroupedGEMM和FP8训练等技术提高了MoE模型的效率，特别是在涉及多个专家时。

- Token分发机制：MoE支持无丢弃和令牌丢弃两种策略，以有效管理专家之间的令牌分配。

# 2 Megatron Core MoE Key Features
- [Megatron-MoE](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md)

## 2.1 与其他并行模式结合性
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Megatron-Core 提供了丰富的并行映射，将专家并行与张量并行、数据并行、序列并行和管道并行相结合。这使得 Mixtral 8X7B bf16 训练在 MCore v0.9 版本下能够达到 468 TFLOPS 的性能。<br>

**并行性** <br>
- 专家并行（Expert Parallelism）: <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;一种针对混合专家（MoE）模型的特定并行方法，其中**专家被划分到不同的工作节点上，每个工作节点处理不同的训练样本批次，**每个工作节点为每个 MoE 层处理**一个或多个专家**。<br>

- 3D 并行性：数据并行（Data Parallelism）、张量并行（Tensor Parallelism）、管道并行（Pipeline Parallelism）<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;注：当在使用 MoE 的同时启用专家并行和张量并行时，**必须启用序列并行**。<br>

- 上下文并行（Context Parallelism）<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;将序列维度进行划分，以支持长上下文训练。<br>

- 更丰富的并行映射：专家并行可以与数据并行/张量并行/管道并行/上下文并行相结合，以处理更大的 MoE 变体。<br>
- 全面的分布式优化器支持。<br>


## 2.2 路由器与负载均衡

- 路由器类型：<br>
- - Top-K 多层感知器（MLP）路由器 <br>

- 负载均衡算法：<br>
- -  Sinkhorn（S-BASE）<br>
- -  辅助损失/负载均衡损失 <br>
- -  无辅助损失的负载均衡策略 <br>


- 性能优化
- - 当本地专家数量大于1时使用GroupedGEMM
- - 支持的数据类型：bf16
- - 针对更大规模混合专家（MoE）模型的性能提升
- - 为MoE启用--tp-comm-overlap
- - 支持FP8训练

- Token分发机制
- - 无丢弃/无令牌丢弃
- - 令牌丢弃，可选择是否填充至容量。


- 易用性
- - Mixtral模型的检查点转换器，详见示例。
- - 分布式检查点存储
- - 逐层日志记录
- - 升级支持
- - 细粒度升级

- 即将推出的功能
- - 大规模混合专家（MoE）训练的新型并行机制
- - GroupedGEMM支持FP8格式
- - Token permutation/Unpermutation 融合
- - TopK路由器融合
- - MoE层频率


# 3 Performance Best Practice
## 3.1 Parallel Mapping
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;为了找到一个良好的并行映射方法，以帮助你实现新模型的高吞吐量，有一些通用规则可以帮到你。以下是每种并行策略在不同方面的特性概述。<br>

| Parallel Strategy | Peak Activation Memory          | Weight Memory  | Optimizer states                  | Communication (Per-Layer) |
|:-----------------:|:-------------------------------:|:--------------:|:---------------------------------:|:-------------------------:|
| TP                | 1/N (with SP on)                | 1/N            | 1/N                               |        High               |
| EP                | 1                               | 1/N in MoELayer| 1/N                               |       Medium              |
| PP                | 1 (>1 with virtual pipeline)    | 1/N            | 1/N                               |       Medium              |
| CP                | 1/N                             | 1              | 1/N (with distributed optimizer)  |       Medium              |
| DP                | 1                               | 1              | 1/N (with distributed optimizer)  |        Low                |

1. 尽量减小模型并行化规模。
- 对于大型语言模型，通常需要采用模型并行化来防止内存溢出（OOM），但这会带来通信开销并影响性能。<br>
- 使用分布式优化器时，主权重和优化器状态将在所有数据并行（DP）节点间进行分片，且通信开销较小。因此，在训练过程中如果GPU内存充足，应尽量减小模型并行化规模，并增大数据并行化规模。<br>
2. 确保专家并行（EP）和张量并行（TP）的通信在NVLink域内进行。
- EP和TP的通信应尽量保持在NVLink域内，因为这两者都是**通信密集型操作**。
- 如果模型过大，需要跨多个节点进行扩展，首先考虑在TP和EP之前使用管道并行（PP）。详见第3点。
3. 使用管道并行来进一步扩展模型规模。
- 当PP规模（PP_size）大于等于2时，启用虚拟管道并行（VPP）来减少PP气泡，通过设置每个虚拟管道阶段的层数（num_layers_per_virtual_pipeline_stage）来实现。
- VPP规模调优：vpp_size的合法值是num_layers/pp_size的所有公约数。例如，若num_layers=24，pp_size=4，则vpp_size可选{1, 2, 3, 6}。vpp_size越大，管道气泡越小，但每个PP阶段之间的点对点（P2P）通信次数越多。经验上，选择一个中间值往往能取得最佳平衡。vpp_size=num_layers / pp_size / num_layers_per_virtual_pipeline_stage。
4. 在可能的情况下，专家层优先选择专家并行（EP）而非张量并行（TP）：
- TP比EP节省更多内存，但EP能实现更高的GEMM效率和更低的通信开销。
- 如果EP规模增加到与专家数量相同，则可以省略专家计算中的本地token permutation/un-permutation.
- 简化混合专家（MoE）层的计算图，便于实现潜在的通信-计算重叠。
- 在实际应用中，对于8x7B模型，EP8TP1优于EP4TP2。
5. 对于长上下文训练，启用上下文并行（CP）。
- CP的效率很大程度上取决于其通信是否能与计算重叠。
- 经验上，当序列长度大于等于8K时，使用CP。

## 3.2 MoE 并行折叠
MoE 并行折叠将 MoE（混合专家）相关的并行组与密集（Dense）组分离。

传统的 MoE 并行组通过使用具有默认顺序（tp-cp-ep-dp-pp）的5维并行组生成器与密集组交织在一起。在 MoE 中，**EP（专家并行）组是注意力（Attention）中 DP（数据并行）的一个子组**。

通过 MoE 并行折叠，我们为注意力使用了一个具有 tp-cp-dp-pp 顺序的并行组生成器，而为 MoE 使用了另一个具有 tp-ep-dp-pp 顺序的并行组生成器。在 MoE 中，EPxTP 组是注意力中 **DPxCPxTP 的一个子组**。

通过设置 --expert-tensor-parallel-size，我们可以为 MoE 设置特定的 TP（张量并行）规模。<br>

## 3.3 MoE 并行折叠的优势
1. 默认情况下，CP（上下文并行）和 EP（专家并行）组被折叠在一起，这样：<br>
- 它减少了启用 CP 和 EP 所需的最小 GPU 数量。例如，传统方式下（CP=8，EP=8）至少需要 64 个 GPU，而现在只需 8 个 GPU。<br>
- CP 和 EP 的通信都可以放在 NVLink 域内进行。<br>

2. 我们可以为注意力（Attention）部分和 MoE（混合专家）部分设置不同的 TP（张量并行）规模。<br>
对于 MoE，EP 通常比 TP 更高效。但在传统方式下，仅使用 EP 可能会导致大多数模型出现内存溢出（OOM）。<br>
通过 MoE 并行折叠，我们可以为注意力部分启用 TP，并为 MoE 模型设置 TP=1，这通常能获得更好的 MFU（可能是指某种性能或利用率指标）。<br>

