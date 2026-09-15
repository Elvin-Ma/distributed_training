import os
from isort import stream
import torch
import time
import torch.distributed as dist
from torch.cuda.green_contexts import GreenContext

local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)
dist.init_process_group("nccl", device_id=device)
rank, world = dist.get_rank(), dist.get_world_size()

a = torch.randn([128, 1024, 1024], device=device)

b = torch.randn([200, 2048, 2048], device=device)
c = torch.randn([2048, 2048], device=device)

comm_sm = 100
kernel_sm = 108 - comm_sm

ctx_comm = GreenContext.create(num_sms=comm_sm, device_id=local_rank)
ctx_compute = GreenContext.create(num_sms=kernel_sm, device_id=local_rank)
stream_comm = ctx_comm.Stream()
stream_compute = ctx_compute.Stream()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start_comm = torch.cuda.Event(enable_timing=True)
end_comm = torch.cuda.Event(enable_timing=True)

with torch.cuda.stream(stream_comm):
    for i in range(5):
        dist.all_reduce(a, async_op=False)

    start_comm.record(stream_comm)
    for i in range(20):
        dist.all_reduce(a, async_op=False)
    end_comm.record(stream_comm)



with torch.cuda.stream(stream_compute):
    for i in range(5):
        output = torch.matmul(b, c)

    start.record(stream_compute)
    for i in range(20):
        output = torch.matmul(b, c)
    end.record(stream_compute)

end_comm.synchronize()
ms_comm = start_comm.elapsed_time(end_comm)
print(f"=============rank{rank} all_reduce total time: {ms_comm} ms")


# Event timing is queried on the CPU, so wait until the GPU reaches `end`.
end.synchronize()
ms = start.elapsed_time(end)
print(f"=============rank{rank} matmul total time: {ms} ms")


# Ensure the communication stream has also completed before the process exits.
stream_comm.synchronize()
stream_compute.synchronize()
# ctx.pop_context()
