import os

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
device = torch.device("cuda", rank)
torch.cuda.set_device(device)

ctx = torch.cuda.green_contexts.GreenContext.create(num_sms=40, device_id=rank)
ctx.set_context()
try:
    x = torch.ones(1024, device=device, dtype=torch.bfloat16)
    dist.all_reduce(x, async_op=False)
finally:
    ctx.pop_context()

dist.destroy_process_group()

print(f"Rank {rank} allreduce test passed.")
