import os
import time

import torch
import torch.distributed as dist


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    group_rank = int(os.environ.get("GROUP_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="cuda", device_id=device)
    dist.barrier(device_ids=[local_rank])

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    tensor = torch.tensor([rank + 1], dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    expected = world_size * (world_size + 1) / 2
    assert tensor.item() == expected, (
        f"rank {rank}: expected {expected}, got {tensor.item()}"
    )

    print(
        f"rank={rank} local_rank={local_rank} "
        f"world_size={world_size} allreduce_sum={tensor.item()}"
    )

    for i in range(1000000):
        time.sleep(5)
        print(f"=========== run step: {i} rank : {rank}", flush=True)
        # dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
        # torch.cuda.synchronize()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
