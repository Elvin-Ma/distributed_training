import os
from pathlib import Path
import torch
import contextlib
import time
import torch.distributed as dist
from torch.cuda.green_contexts import GreenContext

local_rank = 0
torch.cuda.set_device(local_rank)
device = torch.cuda.current_device()

a = torch.randn(1024, 1024, device=device)
b = torch.randn(1024, 1024, device=device)


ctx = GreenContext.create(num_sms=5, device_id=local_rank)
stream = ctx.Stream()


with torch.cuda.stream(stream):
    for i in range(5):
        output = torch.matmul(a, b)

    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(20):
        output = torch.matmul(a, b)
    torch.cuda.synchronize()
    end_time = time.time()


print(f"============= matmul total time: {end_time - start_time}")
