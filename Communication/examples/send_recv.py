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

def run_send_recv(rank, world_size):
    # 设置设备（重要：NCCL需要GPU）
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # 初始化进程组
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank
    )

    send_tensor = torch.arange(2, dtype=torch.float32, device=device) + 2 * rank
    recv_tensor = torch.zeros(2, dtype=torch.float32, device=device)

    dst_rank = (rank + 1) % world_size
    src_rank = (rank - 1) % world_size

    print(f"===========rank: {rank} dst_rank: {dst_rank} src_rank: {src_rank}")

    # if rank == 0:
    #     work1 = dist.isend(send_tensor, dst=dst_rank)
    #     work2 = dist.irecv(recv_tensor, src=src_rank)
    # else:
    #     work2 = dist.irecv(recv_tensor, src=src_rank)
    #     work1 = dist.isend(send_tensor, dst=dst_rank)
    # work1.wait()
    # work2.wait()

    send_op = dist.P2POp(dist.isend, send_tensor, dst_rank)
    recv_op = dist.P2POp(dist.irecv, recv_tensor, src_rank)
    reqs = dist.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()

    # works = []
    # works.append(dist.isend(send_tensor, dst=dst_rank))
    # works.append(dist.irecv(recv_tensor, src=src_rank))

    # for work in works:
    #     work.wait()

    print(f"============rank: {rank} send_tensor: {send_tensor} recv_tensor: {recv_tensor}")



if __name__ == "__main__":
    # 获取环境变量
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    if True:
        import os, debugpy
        base_port=5001
        rank=int(os.getenv("LOCAL_RANK"))
        debugpy.listen(("0.0.0.0", base_port + rank))
        print("Waiting for debugger to attach...", os.getpid())
        debugpy.wait_for_client()


    # run(rank, world_size)
    run_send_recv(rank, world_size)
    print(f"run send_recv done !!!")
