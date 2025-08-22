import torch
import torch.distributed as dist
import os

def run(rank, world_size):
    # 设置设备（重要：NCCL需要GPU）
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # 初始化进程组
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank
    )

    # 创建张量（注意：必须位于当前进程的GPU上）
    if rank == 0:
        data = torch.tensor([1.0, 2.0, 3.0], device=device)
        # 非阻塞发送到进程1
        req = dist.isend(data, dst=1)
        print(f"Rank {rank}: 开始发送数据: {data.cpu()}")
        req.wait()  # 等待发送完成
        print(f"Rank {rank}: 发送完成")
    else:
        recv_data = torch.zeros(3, device=device)
        # 非阻塞接收
        req = dist.irecv(recv_data, src=0)
        print(f"Rank {rank}: 等待接收数据...")
        req.wait()  # 等待接收完成
        print(f"Rank {rank}: 收到数据: {recv_data.cpu()}")

    # 清理进程组
    dist.destroy_process_group()

if __name__ == "__main__":
    # 获取环境变量
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    run(rank, world_size)