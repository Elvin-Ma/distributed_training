import torch
import torch.nn  as nn
from torch.utils.checkpoint  import checkpoint
import logging
torch.cuda.memory_allocated

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