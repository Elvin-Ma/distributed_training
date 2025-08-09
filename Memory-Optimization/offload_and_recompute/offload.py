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