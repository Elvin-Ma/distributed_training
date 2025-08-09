import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

# 配置参数
class Config:
    vocab_size = 10000
    d_model = 512
    nhead = 8
    num_layers = 6
    dim_feedforward = 2048
    max_seq_len = 512
    batch_size = 64
    epochs = 10
    lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformer Decoder模型
class TransformerDecoderModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)
        self.fc = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, x, memory):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embed(pos)

        tgt_mask = nn.Transformer().generate_square_subsequent_mask(seq_len).to(x.device)
        x = self.decoder(x, memory, tgt_mask=tgt_mask)
        return self.fc(x)

# 虚拟数据集
class DummyDataset(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randint(0, Config.vocab_size, (size, Config.max_seq_len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]

# 异步数据预取器
class DataPrefetcher:
    def __init__(self, loader):
        self.loader = loader
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.iter)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            self.next_batch = [t.to(Config.device, non_blocking=True)
                               for t in self.next_batch]

    def __iter__(self):
        self.iter = iter(self.loader)
        self.preload()
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

# 训练函数
def train():
    # 初始化
    config = Config()
    torch.backends.cuda.matmul.allow_tf32 = True  # 启用TF32加速

    # 数据加载
    train_set = DummyDataset()
    loader = DataLoader(train_set,
                       batch_size=config.batch_size,
                       shuffle=True,
                       num_workers=4,
                       pin_memory=True,
                       persistent_workers=True)

    # 模型初始化
    model = TransformerDecoderModel(config)
    model = model.to(config.device)

    # 使用CPU Offload（需要PyTorch >= 1.10）
    # if torch.cuda.is_available():
    #     model = DDP(model, device_ids=[config.device])

    # 混合精度训练
    # scaler = GradScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    # 训练循环
    for epoch in range(config.epochs):
        prefetcher = DataPrefetcher(loader)
        batch_idx = 0

        for src, tgt in prefetcher:
            optimizer.zero_grad(set_to_none=True)  # 更节省内存

            # with autocast():
            memory = torch.randn(src.size(0), config.max_seq_len, config.d_model).to(config.device)
            output = model(src, memory)
            loss = F.cross_entropy(output.view(-1, config.vocab_size),
                                    tgt.view(-1),
                                    ignore_index=0)

            # 梯度缩放和异步反向传播
            # scaler.scale(loss).backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 参数更新
            # scaler.step(optimizer)
            # scaler.update()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
            batch_idx += 1

if __name__ == "__main__":
    train()