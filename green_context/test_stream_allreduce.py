import os
import torch
import time
import torch.distributed as dist
from torch.cuda.green_contexts import GreenContext

local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)
dist.init_process_group("nccl", device_id=device)
rank, world = dist.get_rank(), dist.get_world_size()

a = torch.randn([1, 1024, 1024], device=device)


ctx = GreenContext.create(num_sms=1, device_id=local_rank)
stream = ctx.Stream()
ctx.set_context()

ctx_2 = GreenContext.create(num_sms=1, device_id=local_rank)
stream_2 = ctx_2.Stream()


with torch.cuda.stream(stream):
    for i in range(5):
        dist.all_reduce(a, async_op=False)


    torch.cuda.synchronize()
    start_time = time.time()

    for i in range(20):
        dist.all_reduce(a, async_op=False)

    torch.cuda.synchronize()
    end_time = time.time()

print(f"============= allreduce total time: {end_time - start_time}")
# ctx.pop_context()
