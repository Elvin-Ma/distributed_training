# 1. recompute

```python
import torch
import torch.nn  as nn
from torch.utils.checkpoint  import checkpoint
import logging

# 配置 logging，包含文件名、函数名和行号
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s'
)

logger = logging.getLogger(__name__)

class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, nhead)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model)
        )
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)

    def forward(self, x):
        # 修正形状后的自注意力
        x_attn = x.transpose(0,  1)
        attn_output, _ = self.self_attn(x_attn,  x_attn, x_attn)
        attn_output = attn_output.transpose(0,  1)
        x = self.norm1(x  + attn_output)

        # 前馈网络部分
        ffn_output = self.ffn(x)
        x = self.norm2(x  + ffn_output)
        return x

class CheckpointedDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder  = DecoderLayer()

    def forward(self, x):
        # 启用非重入式检查点
        return checkpoint(self.decoder,  x, use_reentrant=False)

# 训练配置
device = torch.device('cuda')
model = CheckpointedDecoder().to(device)
optimizer = torch.optim.Adam(model.parameters(),  lr=1e-4)
criterion = nn.MSELoss()

# 数据生成（建议实际数据应设置 requires_grad=False）
x = torch.randn(32,  64, 512, device=device)
target = torch.randn(32,  64, 512, device=device)

# 训练循环
for epoch in range(100):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    logger.info(f'Epoch {epoch}, Loss: {loss.item():.4f}')
```

## 1.1 use_reentrant = True

**作用：**

使用传统的可重入模式 实现梯度检查点。在反向传播时，通过 重新执行前向计算 来重建中间激活值。

**特点：**

- 兼容性更好：支持更广泛的自定义操作（如复杂控制流）。

- 显存优化较弱：保存部分中间状态，显存节省不如非可重入模式。

**潜在问题：**

- 对 inplace 操作敏感（可能导致梯度错误）。

- 无法正确处理某些复杂的计算图结构。

## 1.2 use_reentrant = False
当设置 use_reentrant=False 时，梯度检查点（Gradient Checkpointing）会启用 非可重入模式（Non-Reentrant Mode）。这种模式的核心原理是通过 静态计算图分析 和 更高效的激活值重建机制 来实现显存优化，以下是其详细工作原理：<br>

当设置 use_reentrant=False 时，梯度检查点（Gradient Checkpointing）会启用 非可重入模式（Non-Reentrant Mode）。这种模式的核心原理是通过**静态计算图分析** 和 更高效的激活值重建机制 来实现显存优化，以下是其详细工作原理：

**原理：**<br>

通过 按需重建（On-demand Recomputation） 和 缓存管理 实现高效激活值生成：

**前向传播：**

- 仅保存检查点区域的 输入张量 和 元数据（如计算图结构）。

- 不保存任何中间激活值。

**反向传播：**

- 根据元数据和输入张量，重新运行检查点区域的前向计算。

- 动态缓存中间结果：仅保留反向传播当前步骤所需的激活值，计算后立即释放。

- 显存优化：通过细粒度的缓存管理，显存占用从 O(N) 降低到 O(1)（N 为层数）。

**关键技术实现**<br>

- PyTorch 在非可重入模式下使用以下技术优化性能：

- 计算图切分（Graph Partitioning）：将被检查点的代码块视为一个子图，独立分析其输入/输出依赖。

- 自动微分引擎优化：在反向传播时，仅重新计算子图范围内的激活值，而非整个前向过程。

- 内存池复用：通过内存预分配和复用，避免频繁申请/释放显存。

# 2 offload

## 2.1 offload simple

```python
import torch
import torch.nn  as nn

# 自定义OffloadFunction类
class OffloadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # 保存输入到CPU，并保留原始输入用于计算图
        ctx.save_for_backward(input.detach().cpu())
        return input  # 保持前向传播在GPU执行

    @staticmethod
    def backward(ctx, grad_output):
        # 从CPU加载输入数据到GPU并启用梯度
        input_cpu, = ctx.saved_tensors
        input_gpu = input_cpu.cuda().requires_grad_()

        # 重新执行前向计算以构建局部计算图
        with torch.enable_grad():
            # 此处模拟实际模型的前向计算（例如Decoder层）
            output = input_gpu * 2  # 示例计算（替换为实际模型操作）
            output.backward(grad_output)

        return input_gpu.grad   # 返回梯度到上一层

# 定义包含Offload的Decoder模型
class OffloadDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1  = nn.Linear(512, 512)
        self.layer2  = nn.Linear(512, 512)

    def forward(self, x):
        # 第一层正常计算
        x = self.layer1(x)

        # 对第二层应用OffloadFunction
        x = OffloadFunction.apply(x)

        # 继续后续计算（示例）
        x = self.layer2(x)
        return x

# 训练流程
def train():
    model = OffloadDecoder().cuda()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    # 模拟数据
    inputs = torch.randn(32,  512).cuda()
    targets = torch.randn(32,  512).cuda()

    # 训练循环
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
```

## 2.2 Megatron-LM optimizer cpu offloading

- [code addr](https://github.com/NVIDIA/Megatron-LM/blob/main/tests/unit_tests/test_optimizer_cpu_offloading.py)

# 3 gradient accumulate
```python
# 训练循环
for epoch in range(10):
    model.train()
    optimizer.zero_grad()

    for step, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.cuda(),  labels.cuda()

        # 混合精度前向传播
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels) / gradient_accumulate_steps  # 损失归一化

        # 梯度累积反向传播
        loss.backward()

        # 累积到指定步数后更新参数
        if (step + 1) % gradient_accumulate_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            print(f'Epoch {epoch} Step {step}: Loss {loss.item()*gradient_accumulate_steps:.4f}')
```
