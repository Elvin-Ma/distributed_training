# State of torch.compile for training (August 2025)

- [link](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/)
- [知乎翻译](https://zhuanlan.zhihu.com/p/1945653415607776563?share_code=1bUfdb56tD8CG&utm_psn=1945829432687329523)

本文的目的是集中梳理截至 2025 年 8 月，torch.compile 在训练领域的应用现状。文中内容都是其他渠道提及的信息，只是我们很少将所有相关内容整合在同一篇文档中。本文档的目标读者是正在评估将 torch.compile 用于**大规模训练任务的团队**。

# 1 base info

首先，我们来介绍基础信息。torch.compile（又称 PT2）是一款面向 PyTorch 即时执行（eager）程序的编译器，可同时用于推理和训练工作负载。相比即时执行代码，它通常能带来 1.5 至 2 倍的速度提升；此外，torch.compile 还支**持针对内存（例如自动激活检查点）和分布式通信（例如异步张量并行）进行全局优化**。

**torch.compile 有哪些功能？**
torch.compile 最核心的功能是提供了一个装饰器，你可以将其附加到函数上，从而对该函数进行编译：

```python
@torch.compile()
def f(x, y):
    ...
```

以下是 compile（编译功能）的一些非功能性特性，了解这些特性十分重要：

1. Just-in-time compilation:<br>
实际上，我们并不会提前编译函数，而是要等到函数首次被调用时才进行编译，并且在编译完成前，执行(execution)过程会处于阻塞状态。无论是本地缓存还是**远程缓存**，都能让你在**重新运行模型时省去编译开销**。（对于推理场景，可借助 **AOTInductor 实现提前编译（Ahead-of-time compilation）**；而`针对训练场景的提前编译功能，目前仍在开发中`。）

2. 与即时执行模式（Eager）的组合性:<br>
PyTorch 最初的成功源于其即时执行模式（Eager mode）极高的可定制性，而 torch.compile 则致力于保留这一特性。你可以根据需求，将该函数设置为训练循环中任意大小的组成部分；经过编译的函数**可与自动求导（autograd）、分布式数据并行（DDP）、完全共享数据并行（FSDP）以及其他 PyTorch 子系统协同工作**。（不过这种组合性有时并非完美，例如在以下场景中：双重反向传播（暂不支持）、张量子类Tensor subclass（需要子类提供特定支持）、自动求导（`对从编译区域返回的中间变量求导无法实现`）。）若某一代码区域无法通过编译，你可使用 torch.compiler.disable() 完全禁用编译功能，退回到即时执行模式。

3. 梯度更新会延迟到编译区域结束时执行。<br>
这一现象的原因是，PyTorch 即时执行模式下的自动求导（eager autograd）不支持从大型反向传播节点中增量式地流式传输梯度。（该问题可通过使用编译式自动求导（compiled autograd）解决，但这要求你的整个反向传播过程都具备可编译性。）

> 注意是梯度更新而不是权重更新

4. 计算图可能会重新编译。<br>
为确保始终生成**无控制流**的直线型计算图，我们会主动针对函数中使用的所有`非张量参数 / 全局变量`进行特化处理。若这些参数 / 全局变量**发生变化，计算图将重新编译**。（可通过设置 torch._dynamo.config.error_on_recompile = True 禁止重新编译。）

5. 默认静态，可重新编译以支持动态形状 <br>
我们会主动(aggresively)将所有尺寸**特化为静态尺寸**。不过，若发现某一尺寸会随时间变化，`在首次重新编译时，我们将尝试生成一个可处理动态形状的单一编译区域`。但我们无法保证一定能为包含动态形状的模型完成编译。（你可使用 [mark_dynamic](https://docs.pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html) 强制将输入形状设为动态；若我们对某一尺寸进行了静态特化，你可使用 mark_unbacked 触发错误提示。）

6. 计算图中断会透明地跳过不可捕获代码 <br>
默认情况下，若编译器遇到无法处理的代码行，会触发计算图中断，对该行代码禁用编译，但仍会尝试对其前后的代码区域进行编译。（可通过设置 fullgraph=True 禁止此行为。）

7. 默认情况下，函数调用会被内联（inlined），循环会被展开（unrolled） <br>
如果你的模型中包含多个 Transformer 块副本，编译时间会随 Transformer 块的数量**成比例增加**。（你可以通过 “区域编译（regional compilation）” 来减少编译时间，即只编译 Transformer 块，而非编译整个模型。）

8. 与 PyTorch 即时执行模式（eager）并非按位等效 <br>
与 PyTorch 即时eager mode的最大差异在于：当 float16/bfloat16（半精度浮点）操作被融合时，我们**不会插入多余的降精度 / 升精度转换**（down/up-conversions）。（可通过设置torch._inductor.config.emulate_precision_casts = True禁用此行为；你也可以重写即时执行模式的代码，使其在更高精度下执行操作，同时需知晓 torch.compile 会对该代码进行优化。XLA（加速线性代数）也有类似配置xla_allow_excess_precision，JAX 框架默认启用该配置。）

不过，我们也可能会做出替换操作的决策（例如替换矩阵乘法（matmul）的实现）；此外，由于编译过程中不可避免的归约顺序（reduction ordering）差异，也可能导致细微的结果偏差。为帮助诊断这类问题，我们支持将计算图捕获前端（graph capture frontend）与编译器后端（compiler backend）分开进行消融实验（ablation）。

9. 分布式集合通信（Distributed collectives）与 DTensor 可编译，但默认未优化<br>
我们**能够捕获 c10d 集合通信操作**以及**处理 DTensor 的程序**，但默认情况下不会对集合通信操作应用优化。（目前存在可启用的实验性优化，但这部分仍在积极开发中。）通常而言，我们无法对高度优化的分布式框架代码进行追踪（以实现编译优化）。

# 2 State of advanced parallelism
在大规模训练任务中，torch.compile 面临着激烈的竞争，主要来自三类方案：

- PyTorch 原生分布式框架：这类框架采用即时执行模式（eager mode），所有优化均通过手动实现（例如 **megatron**）；
- 自定义 “编译器” 栈：这类方案复用了 PyTorch 的追踪机制（如 **symbolic_trace** 和 **make_fx**），但所需的编译过程（passes）需手动实现；
- JAX 框架：该框架始终优先基于 XLA（加速线性代数）构建，在编译驱动的并行化技术方面已领先数年。

以下是我们目前在高级并行化方面的进展（重点与 JAX 进行对比）: <br>

## 2.1 DTensor：用于表示分片张量的 “全局张量” 抽象 <br>
DTensor 是一种张量子类，可用于表示在 **SPMD（单程序多数据）**设备网格（device mesh）上进行分片的张量。DTensor 的`形状对应原始完整张量的全局形状`，但它会**根据 “布局”（placement）仅在本地存储数据的一个分片**。以下是一些重要细节：<br>

- 分片布局（Shard placements）<br>
与 JAX 的布局不同，DTensor 的布局以 “设备网格（device mesh）” 为核心导向；也就是说，通常你需要**指定一个包含设备网格维度大小的布局列表**，其中Shard(i)表示张量的第 i 个维度会被分片。这与 JAX 的设计思路恰好相反 ——JAX 的布局以 “张量” 为核心导向。

例如，给定一个维度为["dp"（数据并行）, "tp"（张量并行）]的二维设备网格：

在 DTensor 中，布局为[Replicate, Shard(0)]（或使用命名**设备网格轴**时表示为{"dp": Replicate, "tp": Shard(0)}）的张量，对应的 JAX 布局为P("tp", None)。

采用这种设计的原因在于，**DTensor 支持Partial（部分）布局** —— 该布局表示设备网格上某一轴存在`待执行的归约（reduction）操作`。Partial布局在矩阵乘法（matrix multiplies）中极为常见，且它并不与任何特定的张量轴相关联，因此用 “以设备网格为导向” 的形式来表示会更便捷。

但这种设计也存在权衡：以设备网格为导向的布局无法直接支持对分片顺序的指定。例如，若想对一个一维张量先在tp（张量并行）维度分片、再在dp（数据并行）维度分片，在 JAX 中可表示为P(("tp", "dp"),)；但在 DTensor 中，这种顺序无法与[Shard(0), Shard(0)]区分开 —— **实际上，DTensor 始终强制采用 “从左到右” 的分片顺序**。

目前已有提案计划扩展 DTensor 的分片规格，以支持对分片顺序的设置，从而达到与 JAX 同等的表达能力，但该提案尚未落地实现。

- DTensor 支持直接求导：<br>
我们会在包含 DTensor 的程序上**直接运行自动求导**（而非将 DTensor 程序 “解语法糖” 转换为使用常规张量（regular Tensors）的程序后再求导）。这一设计确保了原函数（primal）与其对应的切向量（tangent）可以采用不同的分片策略，这一点与 JAX 实现了功能对等（parity）。

- Tensor 的 Python 子类 <br>
与 JAX 不同，**DTensor 是独立于 Tensor 的一个子类**。不过，Tensor 与 DTensor 的互操作性良好：我们`完全可以将一个 Tensor 看作是在所有维度上均已复制（replicated）的 DTensor`。

DTensor 基于 Python 实现，这一设计使其易于修改和调试，但也带来了相当多的开销（例如，FSDP2 不会直接将梯度累积到 DTensor 中 —— 因为当模型存在数千个参数时，在 DTensor 上执行 detach（分离）和 add（加法）操作会成为性能瓶颈）。

尽管存在上述开销，DTensor 在设计时仍以实现良好的即时执行（eager）性能为目标，并会对分片传播（sharding propagation）的结果进行大量缓存。因此，在快速路径（fastpath，指常见、高效的执行流程）中，DTensor 只需查询应执行的重分布（redistribute）操作，随后直接调度至本地的即时执行操作即可。但这种缓存策略也意味着，对于具有动态形状（dynamic shapes）的工作负载，开销可能会非常大 —— 因为缓存需要所有输入形状都完全匹配才能生效。

- 编译（Compilation）<br>
DTensor 可通过 torch.compile 进行编译，**编译过程会将其 “解语法糖”（desugar）**为底层的集合通信操作（collectives），并消除即时执行模式（eager mode）下 DTensor 带来的所有开销（即便未执行其他任何优化）。不过，编译场景下`对具有动态形状（dynamic shapes）的 DTensor 支持尚不完善`，具体可参考问题链接：http://github.com/pytorch/pytorch/issues/159635（我们认为目前这并非任何关键用例的核心路径，因此由一名资历相对较浅的工程师在逐步推进该问题的解决）。

- 贪心式传播（Greedy propagation）: 即output 是如何分片的？<br>
由于 **DTensor 必须在即时执行模式（eager mode）下工作**，它仅实现了贪心式分片传播 —— 即在每一次即时执行(eager)操作中，都会贪心选择能`使该操作的集合通信开销（collective costs）最小化的**输出分片方式`。目前，借助类编译器框架（compiler-like framework）来支持`分片反向传播`的工作仍在推进中。

- 算子覆盖范围（Operator Coverage）: 分片也需要传播
**DTensor 的运行需要为算子（operations）配置分片传播规则（sharding propagation rules）**。若某一算子的`分片传播规则未实现，DTensor 会直接报错`，而非触发低效的全收集（allgather）操作，以复制（replication）模式运行该算子。

目前我们尚未实现对所有算子的分片传播规则覆盖，但像 Llama3 这类 Transformer 模型所需的关键算子均已覆盖（[分片规则定义参见此链接](https://github.com/pytorch/pytorch/tree/main/torch/distributed/tensor/_ops)）。对于用户自定义算子（user defined operators），你可以自行编写自定义分片规则。

- 不规则分片（Jagged sharding）
我们目前**不支持 “不规则分片” 这一概念 —— 而该概念是实现 “具有非均衡路由的专家并行（expert parallelism with imbalanced routing）” 所必需的**。不过，我们认为现有分片规则基本可复用，以支持此类需求。由于动态性（dynamism）仅会体现在不规则分片的本地张量中，因此`不规则分片不会受前文 “编译” 部分提及的动态形状（dynamic shapes）问题影响`。

- 生态系统（Ecosystem）
我们致力于将 DTensor 打造为分片张量（sharded tensors）的标准表示形式，目前 DTensor 已与检查点（checkpointing）、FSDP2、SimpleFSDP、AutoParallel、torchtitan 等工具 / 框架实现集成。

## 2.2 函数式集合通信（Functional collectives）
若你不希望使用 DTensor，我们还支持 “函数式集合通信”(—— **这是集合通信操作的非原地修改（non-mutating）版本**)，无需依赖 DTensor，即可用于以编译器友好的方式手动实现 SPMD（单程序多数据）操作。（实际上，若你使用传统集合通信 API 并对其进行编译，我们会自动将这些 API `静默转换为函数式集合通信`，以适配编译器处理流程。）

在编译后，函数式集合通信不一定会强制分配输出缓冲区（output buffer），因为它们可支持原地重写（re-inplaced）。需要重点注意的是，函数式集合通信目前不支持自动求导（autograd），相关讨论可参考：https://discuss.pytorch.org/t/supporting-autograd-for-collectives/219430

> 注释：
> 传统集合通信（问题所在）： 传统的操作如 dist.all_reduce(tensor) 是原地操作。它直接修改输入 tensor 的值，而**不返回一个新的张量**。这种“副作用”对于编译器来说很难分析和优化，因为**编译器更喜欢纯函数（输入决定输出，不改变输入）**。
> 函数式集合通信（解决方案）： 函数式版本的行为像一个数学函数：它接收输入张量，返回一个全新的输出张量，而不改变输入张量。

## 2.3 计算图捕获（Graph Capture）
目前有**两种**应用尤为广泛的计算图捕获机制，可用于在模型代码之外单独执行分布式优化。所有计算图捕获机制都会生成 FX 图 —— 这是一种无控制流的简单 Python 基本块中间表示（IR），对于图中可能出现的实际算子集合完全不设限制。

- 符号追踪（Symbolic_trace）
符号追踪是最初的计算图捕获机制，尽管存在局限性，但应用仍十分广泛。它完全通过 Python 算子重载（operator overloading）实现，能将所有可重载的操作完整捕获并记录到计算图中。

我们认为这在很大程度上属于传统流水线（legacy pipeline）—— 因为它无法追踪包含 “基于形状的条件判断” 的代码，且最终生成的计算图中，中间值的形状（shapes）和数据类型（dtypes）相关的有用元数据（metadata）完全缺失。例如，用于实现流水线并行（pipeline parallelism）的传统工具栈 PiPPY，就是基于符号追踪这一计算图捕获机制构建的。

- **make_fx/torch.export**（算子捕获工具 / 模型导出工具）
这种计算图捕获机制的工作原理是：向程序中传入（虚拟的）张量（fake tensors），并记录下 ATen 算子的执行过程。它包含多种不同变体，例如：是采用类似 JAX jit 的 Python 追踪方式，还是采用类似 Dynamo 的复杂字节码分析方式；同样，你还可以提取不同层级的中间表示（IR）（如分发前、分发后；此外，算子既可以被分解，也可以作为单个单元保留）。

我们在编译器并行化方面的工作正是基于这种捕获机制构建的，但理论上，你完全可以在该中间表示（IR）的基础上自行编写计算图处理流程（graph pass）。不过在实际操作中，若缺乏 PyTorch 相关专业知识，这一过程会颇具难度，原因如下：

要将追踪得到的计算图完整集成到 PyTorch 的自动求导（autograd）系统中，使其能与其他代码协同工作，从通用性角度实现这一点非常复杂；
在编译的不同阶段所能获取的具体算子集合并无文档说明，且实际上与 Inductor 底层编译栈（Inductor lowering stack）高度绑定；同时，关于如何在自定义处理流程（pass）执行前**防止算子被分解**，相关文档也十分欠缺。

## 2.3 默认情况下并非 SPMD 编译器
torch.compile 在默认情况下，并不会假设所编译的程序采用 SPMD（单程序多数据）模式，这意味着它不会执行诸如 “丢弃未使用的集合通信操作” 这类行为（你可通过配置标志修改此行为）。此外，torch.compile 的默认使用模式是**在所有节点上并行编译**，这就要求开发者必须格外注意，确保编译器的每个实例都能以完全相同的方式编译（若仅有一个进程组序号重新编译，或不同编译器实例做出不同决策，都可能导致 NCCL 超时）。

我们最终的设想是 “编译程序一次，再将其分发到所有节点”，但由于该功能目前尚未实现，开发者解决此问题的通用方案通常分为两类：

消除进程组序号（rank）产生差异化行为的所有根源，例如，不允许编译器在制定编译决策时，参考动态输入的实际大小；
向编译器中**引入额外的集合通信操作**，以传递那些 “必须在所有进程组序号间保持一致” 的编译决策。

## 2.4 我们对高级并行化未来的愿景
在开发中的 SimpleFSDP（简易完全分片数据并行）与 AutoParallel（自动并行）技术引领下，我们对高级并行化未来的愿景是：用户只需编写单节点程序，**在代码中通过数学形式表达其想要实现的计算逻辑即可**。之后，这些单节点程序会通过两个步骤转换为高效的分布式程序：

第一步，以基础直接的方式（即仅**明确所有中间张量应采用的分片方式**）将集合通信操作（collectives）插入计算图；
第二步，**对集合通信操作进行优化**，以处理调度层面的需求（如数据预取（pre-fetching）和**数据分桶（bucketing）**）。

AutoParallel 设定了一个类 GSPMD（通用单程序多数据）风格的目标：为程序**自动确定 “足够优” 的分片策略** —— 理论上它应能自动识别并实现数据并行、张量并行，甚至专家并行（！）；而 SimpleFSDP 的目标则更聚焦：仅按照 FSDP 框架要求的模式插入集合通信操作，随后通过编写 FSDP 专属的优化流程（pass），以达到与 FSDP2 相当的性能。

编写领域特定优化是非常常见的做法：例如，异步张量并行（async tensor parallelism）的实现方式也是一个优化流程 —— 该流程会**先检测计算图中的张量并行（TP）模式，再将其重写为异步张量并行操作**。

与 JAX 的发展路径不同（JAX 最初从一个高度通用的求解器起步，之后不得不逐步增加更多手动干预手段），**PyTorch 最初的所有分布式模式均完全通过手动编写实现**；直到最近，我们才开始`添加更多自动化机制，作为 “全手动实现” 之外的替代方案`。

# 3 优化现状(State of optimization)
torch.compile 会执行多项优化，以下是其中一些尤为重要、需重点了解的优化内容：<br>

- **Inductor:** Inductor torch.compile 的后端组件，负责为 PyTorch 程序生成 Triton 内核（Triton kernels）**。它对 PyTorch 算子集的覆盖度极高，`能够对逐点运算（pointwise operations）和归约运算（reductions）进行融合` —— 包括反向传播（backwards）过程中通常会出现的运算模式。此外，它还能`将逐点运算融合到矩阵乘法（matmuls）中`，并**对不同的矩阵乘法后端（包括 cuBlas、cutlass 和 Triton）进行自动调优**，从而为任意给定的运算规模`选择最优后端`。

当人们提及 torch.compile 提升程序运行速度时，**通常所指的就是 Inductor 的作用**；不过，torch.compile `并非必须与 Inductor 搭配使用`：例如，你**可以仅使用 AOTAutograd（提前自动求导，Ahead-of-Time Autograd），而跳过 Inductor 编译环节**。

- **CUDA 图（CUDA graphs）**
Inductor 内置了对模型 CUDA 图（CUDA graphing）的支持。与手动应用 CUDA 图相比，我们提供的 CUDA 图支持能保证**更高的可靠性**（例如，可避免 `遗漏复制所有输入缓冲区` 和 `在 CUDA 图区域内执行 CPU 计算` 等手动操作易出现的问题）。

**torch.compile 的 CUDA 图功能通常与 Inductor 搭配使用**，但我们也提供了仅支持即时执行模式（eager mode）的 CUDA 图集成方案（`该方案的实践验证相对较少`）。

- **自动激活检查点（Automatic Activation Checkpointing）**
借助 torch.compile，我们能够对**内存与计算之间的权衡关系**进行全局优化 —— 这一效果**远优于**即时执行模式（eager mode）下 PyTorch 所支持的激活检查点 API（该类 API 要求用户手动指定需要或不需要进行检查点存储的内容）。不过，已有部分开发者反馈，**调整自动激活检查点（AC）的超参数过程可能会非常繁琐**；我们也在其中发现了一些漏洞。

- **FP8 优化（FP8 Optimizations）**
在**传统编译领域，一项重大的成功案例是为一种自定义 FP8 格式（custom FP8 flavor）添加了支持**。借助 torch.compile，开发者`无需为该自定义 FP8 变体手动编写内核代码（manual kernels）`。目前，这项优化已被合并到 `torchao`（PyTorch 高级优化库）的主代码库中（upstreamed）。

- **灵活注意力机制（Flex Attention）**
灵活注意力机制的应用范围持续扩大，在开源社区（OSS）中已有 632 个下游代码仓库用户（2025 年 1 月时仅为 125 个）。在 Llama 系列模型中，该机制已被用于实现分块注意力（**chunked attention**）、文档掩码（document masking）以及上下文并行（context parallelism）功能。尽管有时会有开发者反馈其存在轻微的**数值差异**，但灵活注意力机制仍是一款非常出色的研究工具。

- **Helion**

[Helion](https://github.com/pytorch/helion) 是一个正积极开发的项目，计划于今年 10 月推出测试版（beta）。该项目提供了一个**更高级别的接口**，用于编写 Triton 内核（Triton kernels），**其语法风格与编写 PyTorch 即时执行（eager）代码完全一致**。

Helion 高度依赖`自动调优（autotuning）技术，通过探索内核可能的结构选择空间，来找到最优的内核实现方案`。目前该项目尚未准备好投入生产环境（not production ready），但值得关注的是，它`即将正式发布`。

- **编译时间现状(state of compile time)**
**torch.compile 是一款即时编译器（just-in-time compiler）**，因此在默认配置下，编译过程会在你的 GPU 集群上进行 —— 这会导致 GPU **无法同时用于执行其他有用任务！** 通常情况下，大多数极端的编译耗时问题都源于**重复编译（往往由动态形状导致，但有时也存在其他原因）**。在 Transformer 模型中，可通过仅编译 Transformer 块（编译一次即可，无需为模型中的每个 Transformer 块重复编译 N 次）来缩短编译时间。

我们认为，对于大规模训练任务而言，**缓存并非理想的长期解决方案**，因此一直在研发 “预编译（precompile）” 功能以填补这一空白。预编译的核心逻辑很简单：将编译过程改为 “提前执行” 的流程，通过预编译生成一个二进制文件；之后在训练脚本中，你可直接运行该二进制文件来加载已编译完成的模型。

**预编译生成的产物基于我们的 ABI 稳定接口（为 AOTInductor 开发）构建**，该接口支持同一二进制文件适配多个 PyTorch 版本 —— 即便 PyTorch 库本身并未提供跨版本的 ABI 兼容性。

# 4 如何入门？
对于希望将 torch.compile 用于大规模训练的用户，我们观察到最典型的做法是：复刻（fork）**torchtitan** 仓库，并以该代码库作为你训练工具栈的基础。torchtitan 展示了 PyTorch 的原生功能（包括 torch.compile）—— 实际上，它向你演示了如何将 PyTorch 中的各项功能协同使用，从而实现大规模训练。在此基础上，你可以替换掉那些你有明确自定义需求的组件，同时保留无需修改的部分即可。

