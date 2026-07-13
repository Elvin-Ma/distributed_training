# 1. ParoQuant 论文
- [ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference](https://arxiv.org/abs/2511.10645)

- channel 的概念:
[out_features, in_features] : 这里的某一行或某一列，也常被叫做一个 channel，具体取决于论文讨论的是 input channel 还是 output channel。

- outlier(异常值) 就是数值明显比大多数值大很多的元素;

- Pairwise Rotation = 每次拿两个 hidden channel 做二维旋转
```sh
  目的 = 分散 outlier，降低量化误差
  优势 = 比 full rotation 便宜，容易融合进推理 kernel
```

- ParoQuant 不是重新训练完整模型，而是 PTQ 风格的后训练量化。流程大概是：

```sh
1. 取 calibration data
2. 逐层优化 rotation angle theta 和 scaling alpha
3. 目标是让量化后的 decoder layer 输出接近原始 FP16 layer 输出
4. 再做一点 QAT-like 微调，优化权重和量化参数
5. 导出 PARO checkpoint
6. 推理时用专门 kernel 执行 rotation + scaling + INT4 matmul
```

- ParoQuant 的逻辑是：<br>

```sh
LLM INT4 量化误差大
  -> 主要因为 outlier 和 group dynamic range 太宽
  -> 用 channel-wise scaling 拉平 channel 幅度
  -> 用 pairwise Givens rotation 分散/压制 outlier
  -> 得到更好量化的权重 W'
  -> 用低开销 kernel 在推理时执行
  -> reasoning 长输出时误差积累更小
```

> 它的关键贡献不是“发现旋转能降 outlier”，而是把旋转做成了 足够便宜、可优化、可并行、能融合进推理 kernel 的形式。

# 2.  Qlinear marlin kernel 和 Qlinear形式

## QLinear A16W4 形式

- QLinear 的执行逻辑通常是：权重量化存储，计算时按需反量化，然后和 activation 做矩阵乘。它主要解决的是  Linear 层权重太大、显存带宽压力太高、推理吞吐受限 的问题。
- 典型 weight-only QLinear：

```sh
  输入 activation: FP16 / BF16
  权重存储: INT4 / INT8
  scale: FP16 / FP32
  zero-point: 可选
  累加通常是 FP16 或 FP32，依 kernel 而定

  逻辑上:
  W_fp ≈ (W_q - zp) * scale
  Y = X_fp @ W_fp + bias
```

| 项目 | 格式 |
| --- | --- |
| activation | FP16 / BF16 |
| weight 存储 | INT4 / INT8 |
| dequant 后 weight | FP16 / BF16 |
| 乘法输入 | FP16 / BF16 activation × FP16 / BF16 weight |
| accumulate | FP16 或 FP32，很多高性能 kernel 会用 FP32 accumulate 或 tensor core mixed precision |
| output | FP16 / BF16 |

- 真实实现里一般不会先完整生成 W_fp，而是在 GEMM kernel 中边解包、边反量化、边乘加;

- 核心不是让数学乘法本身变成 INT4 更快，而是解决三个系统瓶颈：
1. 降低显存占用
    - FP16 权重：每个权重 2 bytes
    - INT4 权重：每个权重 0.5 byte
    - 理论上权重体积降到 1/4，外加 scale/zero-point 开销。
2. 降低显存带宽压力
    - LLM 推理时 Linear 层大量时间花在从显存读权重。
    - 权重变小后，同样带宽下能喂给 GPU 更多数据。
3. 提高推理吞吐
    - 在 decode 阶段，batch/token 较小时经常是 memory-bound。
    - INT4 weight-only 可以显著减少权重读取，带来实际加速。

- 它不主要解决什么？
  - 不主要解决 activation 显存。
  - 不一定降低 KV cache 显存。
  - 不一定让 attention 本身更快。
  - 不保证训练加速，QLinear 主要用于推理。
  - 不等于无损压缩，量化会有精度损失。

## W8A8 / 全 INT8 GEMM

这才是更典型的“INT8 推理加速”：

activation: INT8
weight: INT8
accumulate: INT32
output: FP16 / BF16 / INT8

计算形式类似：

X_fp ≈ scale_x * (X_int8 - zp_x)
W_fp ≈ scale_w * (W_int8 - zp_w)

Y_fp ≈ scale_x * scale_w * sum(X_int8 * W_int8)

矩阵乘核心是：

INT8 × INT8 -> INT32 accumulate

然后再根据 output scale 转回 FP16/BF16 或继续量化成 INT8。

这类方案能利用 GPU / CPU / NPU 的 INT8 dot product 指令，理论上算力效率更高。


## QLinear 对比

| 类型 | activation | weight | accumulate | 主要收益 |
| --- | --- | --- | --- | --- |
| INT8 weight-only | FP16/BF16 | INT8 | FP16/FP32 或 mixed | 降权重显存和带宽 |
| W8A8 INT8 | INT8 | INT8 | INT32 | 降带宽并利用 INT8 算力 |
| INT4 weight-only | FP16/BF16 | INT4 | FP16/FP32 或 mixed | 更强压缩权重 |

- 有专门的 INT4 矩阵乘硬件和 kernel，但 LLM 推理里常说的 INT4 加速，多数不是纯 INT4×INT4，而是 INT4 权重存储 + FP16/BF16 activation 的 mixed matmul。
- “纯 INT4 GEMM”在 LLM 推理里不是最常见的主路径，因为 LLM activation 通常仍是 FP16/BF16；
- LLM 常见的 INT4 weight-only matmul.

## QLinear Marlin kernel
- MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models
- arXiv: https://arxiv.org/abs/2408.11743
- PDF: https://arxiv.org/pdf/2408.11743
- GitHub 实现: https://github.com/IST-DASLab/marlin
- Hugging Face paper page: https://huggingface.co/papers/2408.11743

- Marlin Kernel 是一种专门用于低比特量化矩阵乘法的高性能 CUDA kernel，常见于 LLM 推理中的 QLinear / Linear 层加速，尤其是 INT4 weight-only quantization。
- 是执行 QLinear 的一种 CUDA 后端。
- Marlin 主要针对低比特权重量化，尤其是 INT4 weight-only GEMM，做了高度优化。
- 它会要求权重按 Marlin 需要的格式预打包/重排。
- 重点是推理时更快，尤其在 NVIDIA GPU 上利用 tensor cores、内存访问优化等。

- Marlin Kernel 核心思路：
1. weight-only quantization
    - activation 仍然是 FP16/BF16
    - 只量化权重
    - 精度损失比 activation+weight 全量化更小
2. 权重预打包
    - INT4 权重会按照 Marlin kernel 需要的 layout 重新排列
    - 不是简单的 [out_features, in_features]
    - 这样 GPU 线程读取时更连续、更适合 tensor core / warp-level 计算
3. 融合 dequant + GEMM
    - 不先把 INT4 权重完整还原成 FP16
    - 而是在计算过程中边解包、边乘 scale、边参与矩阵乘
    - 避免生成巨大的中间 FP16 权重矩阵
4. 优化显存带宽
    - LLM 推理中很多时候瓶颈不是纯算力，而是权重读取带宽
    - INT4 权重更小，Marlin 又减少额外读写，所以能明显降低 memory traffic
5. tile / warp 级优化
    - 把矩阵乘分成适合 GPU 的 tile
    - 尽量提高 shared memory、register、tensor core 的利用率
    - 减少线程间等待和非连续访存

# 3. PPL 是什么指标？

PPL 是 Perplexity，困惑度，是语言模型评估常用指标。

它衡量模型对一段文本的预测“有多困惑”。越低表示模型越能准确预测下一个 token，语言建模能力越好。

形式上，如果模型对真实 token 序列的平均 negative log likelihood 是：

loss = - average(log p(x_t | x_<t))

那么：

PPL = exp(loss)

直观理解：

PPL = 模型在每个位置平均有多少个“等概率候选 token”

例如：

PPL = 5.663

可以粗略理解为：模型每预测一个 token 时，平均困惑程度相当于在约 5.663 个候选里选择。

你的结果：

W4A16:   5.663
W4AINT8: 5.772
W4AFP8:  5.739

说明相对 W4A16：

W4AINT8 PPL 增加 0.109
W4AFP8  PPL 增加 0.076

# 4 SmoothQuant
SmoothQuant 是一种 LLM 后训练量化 PTQ 方法(不用再训练)，主要目标是让大模型可以稳定做 W8A8 量化：

Weight: INT8
Activation: INT8

它出自论文：

SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models
作者：Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han
会议：ICML 2023
链接：

- Paper: https://proceedings.mlr.press/v202/xiao23c.html
- arXiv: https://arxiv.org/abs/2211.10438
- GitHub: https://github.com/mit-han-lab/smoothquant

核心问题

LLM 里 activation 经常有很大的 outlier。直接把 activation 量化成 INT8，会导致 scale 被少数极大值拉大，普
通值的量化分辨率变差，精度明显下降。

SmoothQuant 的核心思路是：

> activation 难量化，weight 相对容易量化；所以把 activation 的 outlier 压到 weight 里。

对 Linear：

Y = XW

引入一个按 channel 的缩放因子 s：

Y = (X / s) * (sW)

数学结果不变，因为：

(X / s) * (sW) = XW

但数值分布变了：

X / s  -> activation outlier 被平滑，更容易 INT8 量化
sW     -> weight 被相应放大，但 weight 通常更容易承受量化

实际中 s 通常由 activation 和 weight 的 channel-wise 最大值决定，并有一个超参数 alpha 控制“迁移多少量化
难度”：

s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)

直观上：

- alpha 越大：更多地压 activation，更多难度转移到 weight；
- alpha 越小：activation 平滑少一些，weight 压力也小一些。

一句话总结:

SmoothQuant 是一种**不训练模型**的量化前处理方法：通过等价的 per-channel scale 变换，把 activation outlier
平滑掉，并把尺度吸收到 weight 中，从而让 LLM 的 activation 和 weight 都能更稳定地做 INT8 量化。

# 5 GPTQ
GPTQ 是一种 后训练权重量化 PTQ 方法，主要用于把大语言模型的权重压到 INT4/INT3/INT2 等低 bit，同时尽量保持模型输出接近原始 FP16/BF16 模型。

论文：

GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
作者：Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh
会议：ICLR 2023
链接：

- arXiv: https://arxiv.org/abs/2210.17323
- GitHub: https://github.com/IST-DASLab/gptq

它解决什么问题

普通 round-to-nearest 量化是：

W_fp -> W_q

每个权重独立四舍五入到最近的量化值。但问题是：单个权重误差不一定重要，真正重要的是这一层的输出误差：

X W_fp ≈ X W_q

其中 X 是校准数据跑出来的 activation。GPTQ 的目标就是：让量化后的 Linear 层输出尽量接近原始层输出。

核心原理

GPTQ 基于二阶近似。对某一层 Linear：

Y = XW

量化目标可以写成：

min ||XW - XW_q||^2

等价地，它会利用输入 activation 的二阶统计量：

H = X^T X

这里的 H 可以理解为 Hessian 的近似，用来衡量不同权重通道对输出误差的影响。

量化时，GPTQ 不是把所有权重独立 round 完就结束，而是：

1. 选一个权重列或 block 进行量化；
2. 计算这一步引入的量化误差；
3. 根据 Hessian 近似，把误差补偿到后续还没量化的权重上；
4. 继续量化下一列或下一个 block。

直观上：

当前权重量化错了一点
-> 不直接接受这个误差
-> 用后面还未量化的权重来补偿它
-> 使最终 XW_q 更接近 XW

和简单 INT4 量化的区别

简单量化：

W_q = round(W / scale)

只看权重本身。

GPTQ：

W_q = quantize(W, X)

会看校准数据 X，关注该层输出误差，并用二阶信息做误差补偿。

常见用法

GPTQ 最常见是：

W4A16

也就是：

weight: INT4
activation: FP16/BF16

推理时通常是：

Y = X_fp16 @ dequant(W_int4)

GPTQ 本身主要负责“怎么得到高质量的 INT4 权重”，不是某个特定推理 kernel。实际推理可以用 ExLlama、Marlin、
Triton、CUDA kernel 等执行。

一句话总结：GPTQ 是一种面向 LLM 的后训练权重量化算法，用校准数据和 Hessian 近似，在逐步量化权重的同时补
偿误差，从而让低 bit 权重下的层输出尽量接近原模型。


