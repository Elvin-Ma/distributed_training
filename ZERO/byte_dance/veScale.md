# veScale: 一致且高效的Eager模式SPMD分布式训练系统

> **论文**: veScale: Consistent and Efficient Tensor Programming with Eager-Mode SPMD
> **机构**: ByteDance Seed
> **arXiv**: 2509.07003v1

---

## 📋 目录

1. [核心问题与解决方案](#核心问题与解决方案)
2. [系统架构概览](#系统架构概览)
3. [关键技术详解](#关键技术详解)
4. [使用教程](#使用教程)
5. [性能优势](#性能优势)

---

## 🎯 核心问题与解决方案

### 现有系统的两大挑战

```mermaid
graph TD
    A[分布式LLM训练] --> B[挑战1: 一致性问题]
    A --> C[挑战2: 性能瓶颈]

    B --> B1[PyTorch DTensor结果不一致]
    B --> B2[分布式RNG与单设备不匹配]
    B --> B3[不同并行度结果不同]

    C --> C1[DTensor CPU开销高58%]
    C --> C2[通信效率低]
    C --> C3[GPU空闲时间长]

    style B fill:#ff6b6b
    style C fill:#ff6b6b
```

### veScale的三大解决方案

```mermaid
graph LR
    A[veScale] --> B[易用API]
    A --> C[一致性保证]
    A --> D[高性能运行时]

    B --> B1[零模型代码修改]
    B --> B2[Plan API描述并行]
    B --> B3[节省78.4%开发量]

    C --> C1[分布式RNG算法]
    C --> C2[单设备语义保证]
    C --> C3[位级精确匹配]

    D --> D1[DTensor开销降95%]
    D --> D2[通信融合优化]
    D --> D3[2.2×端到端加速]

    style A fill:#4ecdc4
    style B fill:#95e1d3
    style C fill:#95e1d3
    style D fill:#95e1d3
```

---

## 🏗️ 系统架构概览

### 整体流程图

```mermaid
flowchart TB
    subgraph Input["输入层"]
        M[PyTorch原生模型<br/>无需修改]
        P[并行化Plan<br/>Plan API定义]
    end

    subgraph Transform["转换层"]
        PI[Plan IR生成<br/>无需编译模型]
        T1[正则表达式匹配]
        T2[Hook注册]
        T3[DTensor转换]
    end

    subgraph Runtime["运行时"]
        R1[一致性分布式RNG<br/>虚拟线程映射]
        R2[高效DTensor Dispatch<br/>规则匹配+缓存+C++]
        R3[优化通信<br/>梯度分组+N维融合]
    end

    subgraph Output["输出"]
        O[并行化模型<br/>单设备语义+高性能]
    end

    M --> PI
    P --> PI
    PI --> T1
    T1 --> T2
    T2 --> T3
    T3 --> R1
    R1 --> R2
    R2 --> R3
    R3 --> O

    style Input fill:#e3f2fd
    style Transform fill:#fff9c4
    style Runtime fill:#c8e6c9
    style Output fill:#f8bbd0
```

### 核心组件关系

```mermaid
graph TB
    subgraph API["用户接口"]
        PA[Plan API<br/>plan.shard]
        MA[Main API<br/>parallelize]
        DA[Debug API<br/>breakpoint]
    end

    subgraph Core["核心引擎"]
        PIR[Plan IR<br/>中间表示]
        DT[DTensor<br/>分布式张量]
        RNG[Distributed RNG<br/>分布式随机数]
    end

    subgraph Opt["性能优化"]
        D1[Static Eager模式]
        D2[Dispatch优化]
        C1[通信融合]
    end

    PA --> PIR
    MA --> PIR
    PIR --> DT
    DT --> RNG
    DT --> D1
    D1 --> D2
    DT --> C1
    DA -.调试.-> DT

    style API fill:#bbdefb
    style Core fill:#c5cae9
    style Opt fill:#c8e6c9
```

---

## 🔧 关键技术详解

### 1. 分布式RNG算法：虚拟线程映射

#### 问题对比

```mermaid
graph LR
    subgraph Wrong["❌ 错误方法"]
        W1[Seed-based<br/>不同设备不同seed]
        W2[Offset-based<br/>不同设备不同offset]
        W3[结果: 统计不相关]
    end

    subgraph Right["✅ veScale方法"]
        R1[Thread-based<br/>虚拟线程映射]
        R2[全局虚拟GPU]
        R3[结果: 位级匹配]
    end

    style Wrong fill:#ffcdd2
    style Right fill:#c8e6c9
```

#### 虚拟线程算法流程

```mermaid
sequenceDiagram
    participant T as 本地张量
    participant M as 映射算法
    participant V as 虚拟GPU
    participant G as RNG生成器

    T->>M: 本地索引 i
    M->>M: 映射到全局索引 j
    M->>V: 计算虚拟线程 j mod Θ
    M->>V: 计算虚拟偏移 ⌊j/Θ⌋
    V->>G: (seed, virtual_thread, virtual_offset)
    G->>T: 生成随机值

    Note over T,G: 保证分布式结果=单设备结果
```

#### 关键代码逻辑

```python
# 算法伪代码
for i in range(local_tensor_size):
    # 1. 映射本地索引到全局索引
    global_index = map_local_to_global(i, tensor_coord, global_shape)

    # 2. 虚拟化线程和偏移
    virtual_thread = global_index % global_thread_count
    virtual_offset = global_index // global_thread_count

    # 3. 生成随机值（全局视角）
    random_value = curand(global_seed, virtual_thread, virtual_offset)
    local_tensor[i] = random_value
```

### 2. DTensor性能优化：三层递进

```mermaid
graph TD
    V[Vanilla DTensor<br/>580μs/op<br/>100%开销] --> O1
    O1[规则匹配绕过<br/>550μs/op<br/>93%开销] --> O2
    O2[Sharding缓存<br/>68μs/op<br/>12%开销] --> O3
    O3[C++核心<br/>35μs/op<br/>5%开销] --> O4
    O4[Static Eager<br/>0μs/op<br/>0%开销]

    style V fill:#ef5350
    style O1 fill:#ff7043
    style O2 fill:#ffa726
    style O3 fill:#66bb6a
    style O4 fill:#26a69a
```

#### 优化技术对比

| 优化层级 | 技术 | 开销降低 | 适用场景 |
|---------|------|---------|---------|
| **Level 1** | 规则匹配绕过 | 7% | 已知输出元数据的算子 |
| **Level 2** | Sharding缓存 | 81% | 重复模块结构（LLM） |
| **Level 3** | C++ Core | 7% | 关键路径计算 |
| **Level 4** | Static Eager | 5% | 元数据静态的运行时 |

### 3. 通信优化：N维融合

#### 传统方法 vs veScale

```mermaid
graph TB
    subgraph Traditional["传统方法：N次通信"]
        T1[维度1: SP<br/>AllReduce] --> T2[维度2: DP<br/>AllReduce]
        T2 --> T3[...<br/>AllReduce]
        T3 --> T4[维度N<br/>AllReduce]
        TN[通信代价: 2SB·N]
    end

    subgraph VeScale["veScale：1次融合通信"]
        V1[融合所有维度<br/>Flatten Mesh] --> V2[单次AllReduce]
        VN[通信代价: 2SB·1]
    end

    T4 --> TN
    V2 --> VN

    style Traditional fill:#ffcdd2
    style VeScale fill:#c8e6c9
```

#### 融合算法步骤

```mermaid
flowchart LR
    A[梯度DTensor] --> B{找Partial维度}
    B --> C[维度融合<br/>Flatten Mesh]
    C --> D[单次AllReduce]
    D --> E[恢复原始Mesh]
    E --> F[Placement→Replicate]

    style D fill:#4caf50
```

---

## 📚 使用教程

### Step 1: 定义并行策略（Plan API）

```python
from vescale import VescalePlan, Shard, Replicate

plan = VescalePlan()

# ========== Tensor Parallel ==========
# fc1输入复制，权重按列切分
plan.shard("blk\d+.fc1.<in>", Replicate, mesh="TP")
plan.shard("blk\d+.fc1.weight", Shard(1), mesh="TP")

# fc2权重按行切分
plan.shard("blk\d+.fc2.weight", Shard(0), mesh="TP")

# ========== Sequence Parallel ==========
# LayerNorm输入按序列维度切分
plan.shard("blk\d+.ln1.<in>", Shard(1), mesh="TP")
plan.shard("blk\d+.ln1.weight", Replicate, mesh="TP")

# ========== ZeRO-3 Parallel ==========
# 初始化时切分，运行时复制
plan.shard("blk\d+.\w+.weight", Shard(0), mesh="DP", phase="INIT")
plan.shard("blk\d+.\w+.weight", Replicate, mesh="DP", phase="RUN")
```

### Step 2: 应用并行化

```python
import torch
import vescale

# 原始单设备模型（无需修改！）
model = YourModel()

# 一行代码并行化
model = vescale.parallelize(model, plan)

# 优化器自动并行化
optim = torch.optim.Adam(model.parameters())

# 训练循环（单设备写法！）
for batch in data_loader:
    loss = model(batch)
    loss.backward()
    optim.step()

    # 可交互式调试
    vescale.breakpoint()  # 分布式环境下的断点调试
```

### 核心API对比

```mermaid
graph LR
    subgraph Megatron["Megatron-LM（126行改动）"]
        M1[ColumnParallelLinear]
        M2[RowParallelLinear]
        M3[DistributedOptimizer]
        M4[手动AllReduce]
    end

    subgraph VeScale["veScale（29行改动）"]
        V1[plan.shard]
        V2[vescale.parallelize]
        V3[自动优化器并行]
        V4[自动通信]
    end

    style Megatron fill:#ffcdd2
    style VeScale fill:#c8e6c9
```

### 高级特性：Static Eager模式

```python
# 场景：MoE等包含大量轻量算子的模型

# 描述动态元数据
plan.redistribute("matmul.<in>", src=Shard(0), dst=Replicate)
plan.annotate("weight.grad", Partial)
plan.annotate("dropout.<in>", Shard(0))

# 运行时零开销
# - 直接本地张量执行
# - 无DTensor dispatch
# - 通过Plan IR预描述的Hook实现通信
```

---

## 📊 性能优势

### 端到端性能对比

```mermaid
graph TB
    subgraph Models["测试模型"]
        M1[LLaMA-3-70B<br/>1.8× 加速]
        M2[Mixtral-8×7B<br/>2.2× 加速]
        M3[LI-DiT-10B<br/>1.8× 加速]
    end

    subgraph Baselines["对比基线"]
        B1[vs Megatron-LM]
        B2[vs TorchTitan]
        B3[vs Megatron-DeepSpeed]
    end

    M1 & M2 & M3 --> B1 & B2 & B3

    style Models fill:#81c784
```

### 开发效率对比

| 系统 | LLaMA-3 | Mixtral | LI-DiT |
|------|---------|---------|--------|
| **Megatron-LM** | 126行 | 162行 | 82行 |
| **TorchTitan** | 63行 | 100行 | 95行 |
| **veScale** | **29行** | **38行** | **38行** |
| **节省比例** | **77%** | **76%** | **78%** |

### 一致性验证

```mermaid
graph LR
    A[单设备训练] --> B{Loss曲线}
    C[veScale 2-8卡] --> B
    D[TorchTitan 2-8卡] --> E{Loss曲线<br/>偏差0.5+}
    F[Megatron-LM 2-8卡] --> E

    B --> G[✅ 位级匹配<br/>误差<6e-5]
    E --> H[❌ 四个数量级差异]

    style G fill:#c8e6c9
    style H fill:#ffcdd2
```

---

## 🎓 关键概念速查

### SPMD (Single Program Multiple Data)

```mermaid
graph LR
    A[单一模型定义] --> B[自动分片到多设备]
    B --> C[每个设备执行相同代码]
    C --> D[操作不同数据分片]
```

### DTensor工作流

```mermaid
sequenceDiagram
    participant U as 用户代码
    participant D as DTensor
    participant R as Redistribution
    participant S as Sharding Propagation
    participant L as Local Tensor

    U->>D: 算子调用
    D->>R: ①检查并重分布
    R->>S: ②推断输出元数据
    S->>L: ③执行本地计算
    L->>D: ④封装为DTensor
    D->>U: 返回结果
```

### Plan IR示例

```
# 用户Plan API
plan.shard("blk1.fc1.<in>", Replicate, mesh="TP")

# 转换为Plan IR
blk1.fc1.forward_pre:<in>:0:redist(->R,TP)
│         │           │    │      │   └─ Mesh名称
│         │           │    │      └───── 目标Placement
│         │           │    └──────────── 操作类型
│         │           └───────────────── 张量索引
│         └───────────────────────────── Hook类型
└─────────────────────────────────────── 路径
```

---

## 🚀 最佳实践

### 1. 选择合适的并行策略

```mermaid
graph TD
    A{模型大小} -->|<10B| B[TP + DP]
    A -->|10-100B| C[TP + SP + DP + ZeRO]
    A -->|>100B| D[TP + SP + DP + ZeRO + PP]

    E{模型类型} -->|Dense| F[标准TP/SP]
    E -->|MoE| G[Static Eager模式]
    E -->|长序列| H[SP + Context Parallel]
```

### 2. Plan API编写技巧

```python
# ✅ 推荐：使用正则表达式批量匹配
plan.shard("blk\d+.attn.qkv.weight", Shard(0), mesh="TP")

# ❌ 避免：逐个手动指定
# plan.shard("blk1.attn.qkv.weight", ...)
# plan.shard("blk2.attn.qkv.weight", ...)

# ✅ 推荐：复用Plan Zoo
from vescale.plan_zoo import llama3_4d_plan
plan = llama3_4d_plan(tp=4, sp=2, dp=8)
```

### 3. 调试策略

```python
# 1. 开启交互式断点
vescale.breakpoint()  # 类似pdb但支持分布式

# 2. 验证一致性
# - 先单设备训练，记录loss曲线
# - 再分布式训练，对比loss曲线
# - veScale保证完全一致（误差<6e-5）

# 3. 性能分析
# - 检查GPU利用率
# - 分析通信时间占比
# - 使用Static Eager降低CPU开销
```

---

## 📖 总结

### veScale核心价值

```mermaid
mindmap
  root((veScale))
    易用性
      零模型改动
      声明式Plan API
      自动优化器并行
      交互式调试
    一致性
      单设备语义
      位级精确匹配
      跨规模可复现
      分布式RNG算法
    高性能
      5.21× DTensor加速
      N维通信融合
      Static Eager模式
      2.2× 端到端加速
    生产就绪
      已部署内产
      支持多种模型
      开源计划
      PyTorch原生
```

### 适用场景

| 场景 | 推荐理由 |
|------|---------|
| **LLM预训练** | 4D并行+ZeRO-3，高效且一致 |
| **LLM微调** | 零代码改动，快速迁移 |
| **MoE模型** | Static Eager解决轻量算子开销 |
| **研究实验** | 一致性保证可复现性 |
| **生产部署** | 久经考验的内部实战系统 |

### 与其他系统对比

|  | veScale | TorchTitan | Megatron-LM |
|--|---------|------------|-------------|
| **编程模式** | Eager SPMD | Eager SPMD | Eager 耦合 |
| **代码改动** | ✅ 0行模型 | ⚠️ 6行模型 | ❌ 126行模型 |
| **一致性** | ✅ 位级匹配 | ❌ 0.5+误差 | ❌ 0.5+误差 |
| **性能** | ✅ 2.2× | ⚪ 基线 | ⚠️ 缺ZeRO-3 |
| **易用性** | ✅ Plan API | ⚠️ 复杂Plan | ❌ 侵入式 |

---

## 🔗 资源链接

- **项目地址**: https://github.com/volcengine/veScale
- **论文**: arXiv:2509.07003v1
- **文档**: 即将发布
- **上游合作**: 与TorchTitan团队协作中

---

## 📝 引用

```bibtex
@article{li2025vescale,
  title={veScale: Consistent and Efficient Tensor Programming with Eager-Mode SPMD},
  author={Li, Youjie and Wan, Cheng and Lin, Zhiqi and others},
  journal={arXiv preprint arXiv:2509.07003},
  year={2025}
}
```

---

**更新日期**: 2026-03-06
**文档版本**: v1.0
