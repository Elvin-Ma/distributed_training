# Muon Optimizer

本文介绍 Muon 的核心原理，以及它与 FSDP、FSDP2 组合时的通信与工程实现。全文分为三个章节：

1. Muon Optimizer：算法原理、参数分组与单机用法；
2. FSDP + Muon：分片场景下的通信原因与方案选择；
3. FSDP2 + Muon：基于 DTensor 的完整实践与实现原理。

> 说明：分布式优化器 API 仍在演进。落地前请固定 PyTorch 与第三方库版本，并用小模型完成数值与 checkpoint 验证。

---

## 1.1 Muon 是什么

**Muon** 全称 **MomentUm Orthogonalized by Newton–Schulz**，主要用于神经网络隐藏层的二维权重矩阵。

它先像 SGD Momentum 一样计算更新方向，再通过 Newton–Schulz 迭代对更新矩阵做近似正交化，使不同奇异方向的更新尺度更加均衡。

Muon 通常不会负责模型中的全部参数。常见做法是：

- 隐藏层二维权重矩阵使用 Muon；
- embedding、输出层、bias、归一化参数和标量参数使用 AdamW。

## 1.2 核心算法

设权重矩阵与当前梯度分别为：

\[
W\in\mathbb{R}^{m\times n},\qquad G_t=\nabla_W L_t
\]

### 1.2.1 计算动量

Muon 首先计算动量：

\[
B_t=\mu B_{t-1}+G_t
\]

使用 Nesterov 动量时，更新输入可写为：

\[
\widetilde{B}_t=G_t+\mu B_t
\]

常见起点是 `momentum=0.95`。

### 1.2.2 正交化更新矩阵

若对更新输入做奇异值分解：

\[
\widetilde{B}_t=U\Sigma V^\top
\]

Muon 希望得到：

\[
O_t=UV^\top
\]

这会把非零奇异值近似调整为 1，只保留更新矩阵的方向结构。对非方阵，更准确的说法是半正交化。

最终更新为：

\[
W_{t+1}=(1-\eta\lambda)W_t-\eta O_t
\]

其中 \(\eta\) 是学习率，\(\lambda\) 是 decoupled weight decay 系数。实际实现还可能根据矩阵形状调整学习率。

> Muon 正交化的是**更新量**，不是模型权重本身。

## 1.3 为什么使用 Newton–Schulz

直接通过 SVD 计算 \(UV^\top\) 成本较高。Muon 使用以矩阵乘法为主的 Newton–Schulz 迭代近似该结果：

\[
X_{k+1}=aX_k+b(X_kX_k^\top)X_k+c(X_kX_k^\top)^2X_k
\]

经典 Muon 系数为：

\[
(a,b,c)=(3.4445,-4.7750,2.0315)
\]

通常迭代 5 次。该过程适合 GPU 矩阵乘法，也常使用 `bfloat16` 执行。

简化后的教学实现如下：

```python
def newton_schulz(update, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315

    x = update.to(torch.bfloat16)
    x = x / x.norm().clamp_min(eps)

    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T

    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x

    return x.T if transposed else x
```

该函数用于说明算法，不包含生产实现中的参数分组、形状缩放、混合精度和分布式通信优化。

## 1.4 Muon 与 AdamW 的区别

| 对比项 | AdamW | Muon |
|---|---|---|
| 更新方式 | 逐元素自适应缩放 | 矩阵级近似正交化 |
| 主要状态 | 一阶矩和二阶矩 | 主要是动量 |
| 适用参数 | 几乎全部参数 | 主要是隐藏层二维矩阵 |
| 额外计算 | 逐元素运算 | 多次矩阵乘法 |
| 几何结构 | 基本不使用矩阵结构 | 显式使用矩阵结构 |
| 常见用法 | 单独使用 | 与 AdamW 混合使用 |

可以把 Muon 简化理解为：

```text
SGD / Nesterov Momentum
        +
更新矩阵的近似正交化
```

## 1.5 参数分组

推荐交给 Muon 的参数：

- Transformer block 中的线性层权重；
- Attention 的 Q、K、V、O 投影；
- MLP 的 up、down、gate projection；
- 普通 MLP 隐藏层二维权重；
- 经实现明确支持后展平的卷积核。

通常继续交给 AdamW 的参数：

- bias；
- LayerNorm、RMSNorm 参数；
- embedding；
- LM head、classifier head 等输出层；
- 标量和一维参数。

不要只用 `param.ndim == 2` 选择 Muon 参数，因为 embedding 和输出层通常也是二维矩阵。应结合模块类型或参数名显式排除。

## 1.6 PyTorch 单机示例

PyTorch 提供了 [`torch.optim.Muon`](https://docs.pytorch.org/docs/main/generated/torch.optim.Muon.html)。使用前应确认安装版本包含该接口。

```python
import torch


muon_params = [
    param
    for param in model.hidden_layers.parameters()
    if param.requires_grad and param.ndim == 2
]
muon_param_ids = {id(param) for param in muon_params}

adamw_params = [
    param
    for param in model.parameters()
    if param.requires_grad and id(param) not in muon_param_ids
]

muon_optimizer = torch.optim.Muon(
    muon_params,
    lr=0.02,
    momentum=0.95,
    nesterov=True,
    weight_decay=0.01,
    ns_steps=5,
)

adamw_optimizer = torch.optim.AdamW(
    adamw_params,
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.01,
)

for inputs, targets in dataloader:
    muon_optimizer.zero_grad(set_to_none=True)
    adamw_optimizer.zero_grad(set_to_none=True)

    loss = loss_fn(model(inputs), targets)
    loss.backward()

    muon_optimizer.step()
    adamw_optimizer.step()
```

## 1.7 超参数起点

常见起点：

```python
momentum = 0.95
nesterov = True
ns_steps = 5
```

Muon 的学习率经常明显高于 AdamW，例如：

```text
Muon:  2e-2
AdamW: 3e-4
```

这只是调参起点，不是固定比例。不同实现的更新尺度调整方式不同，迁移配置时应同时检查：

- Muon 学习率；
- AdamW 学习率；
- weight decay；
- warmup 与学习率调度；
- `adjust_lr_fn` 或同类形状缩放选项；
- 参数是否按预期分组。

## 1.8 优势与限制

优势：

- 显式利用权重矩阵的结构；
- 在部分语言模型与视觉任务中可提高样本效率；
- 相比 AdamW，处理矩阵时通常只需保存动量状态；
- Newton–Schulz 主要由适合 GPU 的矩阵乘法组成。

限制：

- 不能自然覆盖所有参数类型；
- 通常需要与 AdamW 等优化器混合使用；
- 参数分组、学习率尺度和分布式实现更复杂；
- 小矩阵或小 batch 场景下，额外矩阵乘法可能不划算；
- 在 FSDP/ZeRO-3 中，整矩阵正交化通常需要额外通信；
- 不能假定它在所有任务上都优于充分调参的 AdamW。

---

# Muon with comm ?

## 2.1 为什么 Muon 在 FSDP 中需要额外通信

FSDP 会把参数、梯度和优化器状态切分到多个 rank。AdamW 与 SGD 的更新主要是逐元素操作，因此每个 rank 可以直接更新自己的 shard。

Muon 不同。它需要对逻辑上的完整矩阵执行：

\[
G=U\Sigma V^\top\longrightarrow UV^\top
\]

设完整梯度矩阵按行切分为：

\[
G=
\begin{bmatrix}
G_0\\G_1\\\vdots\\G_{P-1}
\end{bmatrix}
\]

一般而言：

\[
\operatorname{Muon}(G_i)
\neq
\operatorname{shard}_i(\operatorname{Muon}(G))
\]

因此，仅在每个本地 shard 上独立运行 Muon，会得到 block-wise/local Muon，而不是整矩阵 Muon。

## 2.2 不同并行方式的通信需求

| 并行方式 | Muon 的额外通信 |
|---|---|
| 单卡 | 不需要 |
| DDP，所有 rank 保存完整参数 | 梯度同步后通常不需要额外通信 |
| FSDP/ZeRO-3，矩阵被切分 | 整矩阵 Muon 通常需要额外通信 |
| FSDP-aware Muon | 通过重分布组装矩阵，计算后再分发更新 |
| 每个 shard 独立运行 Muon | 可不通信，但算法不再等价于整矩阵 Muon |

典型的分布式 Muon 流程如下：

```text
FSDP backward
    ↓
每个 rank 持有 gradient shard
    ↓
重分布：storage layout → compute layout
    ↓
负责计算的 rank 持有完整矩阵或完整 block
    ↓
本地执行全部 Newton–Schulz 迭代
    ↓
重分布：compute layout → storage layout
    ↓
每个 rank 更新自己的 parameter shard
```

通信主要发生在 Newton–Schulz 迭代的前后，而不是每次迭代之间。

## 2.3 为什么 FSDP1 不能直接套用普通 Muon

FSDP1 的 optimizer 可见参数可能是一维 shard。即使启用 `use_orig_params=True`，参数在 sharded 状态下仍可能：

- 只包含原参数的一部分；
- 在某些 rank 上为空；
- 不保留原始二维矩阵布局。

而 `torch.optim.Muon` 要求参数与梯度是二维矩阵，也不会自动为 FSDP shard 组装完整矩阵。因此，不应假定下面的代码能够得到正确的整矩阵 Muon：

```python
model = FSDP(model, use_orig_params=True)
optimizer = torch.optim.Muon(model.parameters())
```

如果必须继续使用 FSDP1，应选择明确声明支持该分片形式的优化器实现；否则优先迁移到 FSDP2。

## 2.4 local Muon 为什么不等价

Newton–Schulz 的核心计算包含：

\[
XX^\top
\]

当 \(X\) 按行切分时，完整 Gram matrix 包含跨 shard 的交叉项：

\[
X_iX_j^\top,\qquad i\neq j
\]

本地计算 \(X_iX_i^\top\) 会丢失这些交叉项。因此 local Muon 可能带来：

- world size 改变时优化器行为变化；
- 每个 shard 独立归一化奇异值；
- 更新尺度与整矩阵 Muon 不一致；
- checkpoint 在不同并行规模间恢复时行为不一致。

除非实现明确给出 block-wise/local Muon 的数学定义与适用范围，否则不要把一维 shard 随意 reshape 后传给普通 Muon。

## 2.5 方案选择

### 2.5.1 模型可以使用 DDP

优先使用：

```text
DDP gradient All-Reduce
+ 每卡独立计算相同的 Muon 更新
```

实现最简单，也不需要额外的 Muon 矩阵重分布。

### 2.5.2 模型必须使用全分片

优先使用：

```text
FSDP2 + DTensor-aware distributed Muon
```

不要直接把普通 `torch.optim.Muon` 套在 FSDP 参数上。

### 2.5.3 跨节点通信昂贵

可依次评估：

1. 使用 HSDP，把 Muon 通信限制在较小的 shard mesh；
2. 批量打包多个矩阵的 All-to-All；
3. 对 fused QKV 按 Q/K/V 或 head/block 分别正交化；
4. 使用 Dion2/Dion3 等低秩通信方案。

低秩方案可以减少通信和计算，但不与完整 Muon 严格等价。

---

# FSDP2 + Muon

## 3.1 FSDP2 的参数状态

FSDP2 使用 [`torch.distributed.fsdp.fully_shard`](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)，并把参数转换为 DTensor。典型流程是：

```text
Forward:
    parameter All-Gather
    → forward compute
    → 释放完整参数

Backward:
    parameter All-Gather
    → backward compute
    → gradient Reduce-Scatter
```

backward 完成后：

- 参数是 sharded DTensor；
- 梯度是 sharded DTensor；
- optimizer 在分片后的 DTensor 参数上执行；
- 每个 rank 通常只持有本地 shard。

例如，8 路 FSDP 按第 0 维切分：

```text
global parameter: [4096, 4096]
local shard:      [ 512, 4096]
```

## 3.2 推荐实现

可选实现主要分为两类：

| 实现 | 适用场景 | 特点 |
|---|---|---|
| Microsoft `dion.Muon` | 独立训练代码接入 | 支持现代 PyTorch、FSDP2/DTensor、批量通信 |
| TorchTitan `DistMuon` | TorchTitan/FlexShard 训练栈 | 显式规划 storage/compute layout，并流水化重分布 |

PyTorch 核心的 `torch.optim.Muon` 是单 Tensor 优化器，不包含专门的 FSDP2 矩阵重分布方案。

## 3.3 使用 `dion.Muon`

### 3.3.1 安装与版本要求

`dion` 面向现代 PyTorch 的 DTensor/FSDP2。安装方式：

```bash
pip install git+https://github.com/microsoft/dion.git
```

建议在项目中固定 commit 或 release，而不是长期跟随 `main`：

```bash
pip install "dion @ git+https://github.com/microsoft/dion.git@<commit>"
```

### 3.3.2 初始化进程组与 DeviceMesh

```python
import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from dion import Muon


dist.init_process_group("nccl")

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

world_size = dist.get_world_size()
mesh = init_device_mesh(
    "cuda",
    mesh_shape=(world_size,),
    mesh_dim_names=("fsdp",),
)
```

不要用全局 rank 代替 `LOCAL_RANK` 选择设备；多节点环境中两者不同。

### 3.3.3 应用 FSDP2

Transformer 建议自底向上逐层应用 `fully_shard()`：

```python
model = MyTransformer().cuda()

for block in model.layers:
    fully_shard(block, mesh=mesh)

fully_shard(model, mesh=mesh)
```

必须在 `fully_shard()` 之后创建 optimizer，使 optimizer 接收到转换后的 DTensor 参数。

### 3.3.4 参数分组

优先根据模块类型分组。无法方便访问模块类型时，可以用参数名作为经过审核的回退方案：

```python
muon_params = []
adamw_params = []

excluded_names = ("embed", "embedding", "lm_head", "output", "norm")

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue

    use_muon = (
        param.ndim == 2
        and not any(token in name.lower() for token in excluded_names)
    )

    if use_muon:
        muon_params.append(param)
    else:
        adamw_params.append(param)
```

初始化后应输出并检查两组参数名，确认 embedding、unembedding、norm 和 bias 没有误入 Muon 组。

### 3.3.5 创建混合优化器

`dion.Muon` 可以通过参数组分别运行 Muon 与 AdamW：

```python
optimizer = Muon(
    [
        {
            "params": muon_params,
            "algorithm": "muon",
            "lr": 0.02,
        },
        {
            "params": adamw_params,
            "algorithm": "adamw",
            "lr": 3e-4,
            "weight_decay": 0.01,
        },
    ],
    distributed_mesh=mesh,
    mu=0.95,
    nesterov=False,
    weight_decay=0.01,
    adjust_lr="spectral_norm",
)
```

参数名与默认值可能随版本变化；应以固定版本的 [`dion.Muon` 源码](https://github.com/microsoft/dion/blob/main/dion/muon.py) 为准。

### 3.3.6 训练循环

```python
for batch in dataloader:
    optimizer.zero_grad(set_to_none=True)

    output = model(batch["input_ids"])
    loss = loss_fn(output, batch["labels"])

    loss.backward()
    optimizer.step()
```

启动示例：

```bash
torchrun --standalone --nproc_per_node=8 train.py
```

多节点启动时，应按集群环境补充 rendezvous 配置。

## 3.4 HSDP 配置

HSDP 使用二维 DeviceMesh：一个维度负责复制，另一个维度负责分片。

例如 8 卡配置为 2 路复制 × 4 路分片：

```python
mesh = init_device_mesh(
    "cuda",
    mesh_shape=(2, 4),
    mesh_dim_names=("replicate", "shard"),
)
```

FSDP2 使用完整 mesh：

```python
for block in model.layers:
    fully_shard(block, mesh=mesh)

fully_shard(model, mesh=mesh)
```

Muon 只接收发生参数分片的子 mesh：

```python
optimizer = Muon(
    param_groups,
    distributed_mesh=mesh["shard"],
    lr=0.02,
    mu=0.95,
)
```

这样可以把 Muon 的矩阵重分布限制在 shard group 内。复制维度的梯度同步仍由并行策略负责。

## 3.5 TorchTitan `DistMuon` 的实现原理

TorchTitan 的 `DistMuon` 把参数布局分成两类。

### 3.5.1 Storage layout

FSDP2 长期保存参数、梯度和 momentum 的布局：

```text
parameter: DTensor Shard(...)
gradient:  DTensor Shard(...)
momentum:  DTensor Shard(...)
```

该布局用于节省显存，并与 FSDP2 Reduce-Scatter 的结果对齐。

### 3.5.2 Compute layout

执行 Muon 正交化时使用的布局，可以是：

- 某个 rank 持有完整矩阵；
- 不同 rank 分别持有完整 block；
- storage layout 已满足本地计算要求，无需重分布。

### 3.5.3 Optimizer step

一次 optimizer step 可概括为：

```text
FSDP2 backward
    ↓
sharded DTensor gradient
    ↓
在本地 storage shard 上更新 momentum
    ↓
打包后的 storage → compute All-to-All
    ↓
在负责的 rank 上执行本地 Newton–Schulz
    ↓
打包后的 compute → storage All-to-All
    ↓
更新本地 parameter shard
```

动量更新是逐元素操作，因此可以在本地 shard 上完成：

\[
B_{t,i}=\mu B_{t-1,i}+G_{t,i}
\]

momentum 无需长期保存成完整矩阵。

### 3.5.4 为什么使用 All-to-All

简单的 All-Gather 方案会让每个 rank 都持有完整矩阵，并重复运行 Newton–Schulz。All-to-All 可以同时完成矩阵重组与计算任务分配。

四卡示例：

```text
FSDP storage layout:
rank 0: A0 B0 C0 D0
rank 1: A1 B1 C1 D1
rank 2: A2 B2 C2 D2
rank 3: A3 B3 C3 D3

storage → compute All-to-All:
rank 0: [A0 A1 A2 A3] = full A
rank 1: [B0 B1 B2 B3] = full B
rank 2: [C0 C1 C2 C3] = full C
rank 3: [D0 D1 D2 D3] = full D
```

每个完整矩阵只正交化一次。计算结束后，再通过反向 All-to-All 把更新拆回各 rank 的 storage shard。

### 3.5.5 通信次数

对需要重分布的 bucket，典型流程是：

```text
1 次 storage → compute All-to-All
N 次本地 Newton–Schulz 迭代
1 次 compute → storage All-to-All
```

因此通信次数不随 `ns_steps` 线性增长：

\[
\text{communication}\not\propto\text{ns\_steps}
\]

若 storage layout 已经满足 compute layout，则该参数可以跳过额外通信。

### 3.5.6 通信与计算重叠

高效实现通常使用：

- 多参数打包；
- bucket 化通信；
- 双缓冲；
- 独立通信 stream；
- CUDA Event 管理跨 stream 依赖；
- 当前 bucket 计算时预取下一个 bucket。

这些优化不会改变 Muon 的数学更新，但会显著影响 optimizer step 的 wall-clock 时间。

## 3.6 Fused QKV 与 block 处理

对 fused QKV 参数，不一定要把整个张量视为一个大矩阵。可以按 Q/K/V 或 attention head 划分 block，再分别执行 Muon。

例如：

```text
physical tensor: [num_heads * head_dim, hidden_dim]
logical view:    [num_heads, head_dim, hidden_dim]
```

合理的 block 划分可以：

- 保持每个逻辑矩阵独立正交化；
- 让 shard 边界与 block 边界对齐；
- 减少整块重组的通信；
- 使用 batched BF16 kernel 处理多个小矩阵。

具体配置方式取决于所用实现。`dion.Muon` 支持通过参数组的 `split_sizes` 描述 fused 矩阵的分块；TorchTitan 则通过 compute layout 描述 block 的拥有关系。

## 3.7 性能与正确性检查

### 3.7.1 最小正确性检查

在扩展到大模型前，至少验证：

1. 单卡 Muon 与分布式 Muon 在小矩阵上的更新误差；
2. 2 卡与 4 卡运行时 loss 曲线是否一致；
3. 参数组中没有遗漏或重复参数；
4. embedding、输出层、norm 与 bias 使用预期算法；
5. checkpoint 保存、恢复后 optimizer state 完整；
6. 更改 world size 恢复 checkpoint 的行为符合预期；
7. 无梯度参数和空 shard 能被正确处理。

### 3.7.2 性能测量

应分别测量：

- forward 时间；
- backward 时间；
- optimizer step 时间；
- 峰值显存；
- Muon 重分布通信量；
- 通信与计算重叠比例；
- 不同 bucket 大小的影响。

不要只比较单步耗时。优化器的最终价值应结合达到目标 loss 所需的 token、step 和总训练时间评估。

### 3.7.3 常见问题

| 现象 | 优先检查 |
|---|---|
| `param` 或 `grad` 不是二维 | 参数分组是否混入 bias、norm 或一维 shard |
| 多卡结果与单卡差异很大 | 是否错误地在每个 shard 上独立运行 Muon |
| optimizer step 很慢 | 是否逐矩阵通信、bucket 太小或未重叠通信与计算 |
| 显存突然升高 | 是否在每个 rank 上 All-Gather 了所有完整矩阵 |
| HSDP 卡住 | optimizer 是否传入了错误的 mesh 或 process group |
| embedding 训练异常 | embedding/unembedding 是否误入 Muon 参数组 |
| checkpoint 无法恢复 | optimizer state 与 DeviceMesh/world size 是否兼容 |

## 3.8 参考资料与源码入口

- [Muon 原始介绍](https://kellerjordan.github.io/posts/muon/)
- [PyTorch `torch.optim.Muon`](https://docs.pytorch.org/docs/main/generated/torch.optim.Muon.html)
- [PyTorch FSDP2 `fully_shard`](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)
- [Microsoft Dion](https://github.com/microsoft/dion)
- [`dion.Muon` 源码](https://github.com/microsoft/dion/blob/main/dion/muon.py)
- [TorchTitan](https://github.com/pytorch/torchtitan)
- [TorchTitan `DistMuon`](https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/flex_shard/dist_muon.py)
- [TorchTitan optimizer reshard runtime](https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/flex_shard/_optimizer_reshard_runtime.py)
- [TorchTitan optimizer reshard schedule](https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/flex_shard/_optimizer_reshard_schedule.py)

## 3.9 总结

- Muon 适合隐藏层二维权重矩阵，其他参数通常继续使用 AdamW；
- FSDP shard 上的 local Muon 通常不等价于整矩阵 Muon；
- FSDP2-aware Muon 会在 storage layout 与 compute layout 之间重分布；
- Newton–Schulz 迭代在负责计算的 rank 上本地执行，迭代之间通常不通信；
- 打包 All-to-All、bucket、双缓冲和通信/计算重叠是高效实现的关键；
- 大规模训练前应固定版本，并完成数值、性能与 checkpoint 验证。
