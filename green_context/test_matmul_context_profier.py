import torch
import contextlib
import os
from datetime import datetime
from pathlib import Path
import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler


def test_green_context():
    a = torch.randn(1000, 1000, device='cuda')
    b = torch.randn(1000, 1000, device='cuda')
    result = torch.matmul(a, b)

    ctx = torch.cuda.green_contexts.GreenContext.create(
        num_sms=20,
        device_id=0,
    )

    ctx.set_context()

    try:
        a = torch.randn(1000, 1000, device='cuda')
        b = torch.randn(1000, 1000, device='cuda')
        result = torch.matmul(a, b)
    finally:
        ctx.pop_context()

    ctx_2 = torch.cuda.green_contexts.GreenContext.create(
        num_sms=20,
        device_id=0,
    )

    ctx_2.set_context()

    try:
        a = torch.randn(1000, 1000, device='cuda')
        b = torch.randn(1000, 1000, device='cuda')
        result = torch.matmul(a, b)
    finally:
        ctx_2.pop_context()


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

def test_green_context_with_profiling():
    with maybe_enable_profiling(True, trace_dir="./profile_traces") as prof:
        test_green_context()

if __name__  == "__main__":

    test_green_context_with_profiling()
    print("Green context test passed.")