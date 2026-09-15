import os
from pathlib import Path
import torch
import contextlib
import torch.distributed as dist
from torch.cuda.green_contexts import GreenContext

@contextlib.contextmanager
def maybe_enable_profiling(enable: bool, trace_dir: str = "./profile_traces"):
    if not enable:
        yield None
        return
    rank = dist.get_rank() if dist.is_initialized() else 0
    trace_path = Path(trace_dir)
    trace_path.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        acc_events=True,
        with_stack=True,
    ) as prof:
        yield prof
    prof.export_chrome_trace(str(trace_path / f"rank{rank}_trace.json"))

local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)
dist.init_process_group("nccl", device_id=device)
rank, world = dist.get_rank(), dist.get_world_size()
assert world >= 2

ctx = GreenContext.create(num_sms=1, device_id=local_rank)
cases = [("primary", torch.cuda.Stream()), ("green(req=1)", ctx.Stream())]
x = torch.zeros(1024 * 1024**2, device=device, dtype=torch.float32)  # 4GB
torch.cuda.synchronize()
iters = 20
with maybe_enable_profiling(True, trace_dir="./profile_traces") as prof:
    for name, stream in cases:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            x.fill_(float(rank == 0))
            dist.all_reduce(x, async_op=False)
            assert (x == 1).all().item()  # 正确性检查
            x.zero_()                   # 避免反复 SUM 溢出
            for _ in range(10):
                dist.all_reduce(x, async_op=False)
            start.record()              # 提前初始化计时事件
            end.record()
        end.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            start.record()
            for _ in range(iters):
                dist.all_reduce(x, async_op=False)
            end.record()
        end.synchronize()
        ms = torch.tensor([start.elapsed_time(end) / iters], device=device)
        dist.all_reduce(ms, op=dist.ReduceOp.MAX, async_op=False)
        ms = ms.item()
        gbps = x.numel() * x.element_size() / (ms * 1e6)
        if rank == 0:
            print(f"{name}: {ms * 1000:.2f} us/op, algbw={gbps:.2f} GB/s")

dist.destroy_process_group()
