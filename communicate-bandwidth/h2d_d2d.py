import torch
#import torch_cuda
import time
# import utils
# from utils import maybe_enable_profiling


def d2d_throughput(cnt):
    data0 = [torch.randn(1024, 1024, device="cuda") for i in range(cnt)]
    data1 = [torch.randn(1024, 1024, device="cuda") for i in range(cnt)]
    stream1 = torch.cuda.Stream()

    torch.cuda.synchronize()
    with torch.cuda.stream(stream1):
        warmup = [data1[i].copy_(data0[i]) for i in range(cnt)]
        stream1.synchronize()
        start_time = time.time()
        output = [data1[i].copy_(data0[i], non_blocking=True) for i in range(cnt)]
        stream1.synchronize()
        end_time = time.time()
        print(f"D2D Throughput: {cnt * 4 /1024 / (end_time - start_time)} GB/s")

def h2d_throughput(cnt):
    data0 = [torch.randn(1024, 1024, device="cpu", pin_memory=True) for i in range(cnt)]
    data1 = [torch.randn(1024, 1024, device="cuda") for i in range(cnt)]

    warmup = [data1[i].copy_(data0[i]) for i in range(cnt)]

    torch.cuda.synchronize()
    start_time = time.time()
    output = [data1[i].copy_(data0[i], non_blocking=True) for i in range(cnt)]
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"H2D Throughput: {cnt * 4 /1024 / (end_time - start_time)} GB/s")


def h2d_d2d_throughput(cnt):
    data0 = [torch.randn(1024, 1024, device="cuda") for i in range(cnt)]
    data1 = [torch.randn(1024, 1024, device="cuda") for i in range(cnt)]
    data2 = [torch.randn(1024, 1024, 113, device="cuda") for i in range(cnt)]
    data3 = [torch.randn(1024, 1024, 113, device="cpu", pin_memory=True) for i in range(cnt)]

    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    warmup = [data1[i].copy_(data0[i]) for i in range(cnt)] # d2d
    warmup = [data2[i].copy_(data3[i]) for i in range(cnt)] # h2d

    d2d_start_event = torch.cuda.Event(enable_timing=True)
    d2d_end_event = torch.cuda.Event(enable_timing=True)

    h2d_start_event = torch.cuda.Event(enable_timing=True)
    h2d_end_event = torch.cuda.Event(enable_timing=True)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./mtn_test_h2d_d2d_overlap_huge"),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:

        for step in range(cnt):
            prof.step()

            with torch.cuda.stream(stream1):
                d2d_start_event.record(stream1)
                output = [data0[i].copy_(data1[i], non_blocking=True) for i in range(cnt)]
                d2d_end_event.record(stream1)


            with torch.cuda.stream(stream2):
                h2d_start_event.record(stream2)
                output = [data2[i].copy_(data3[i], non_blocking=True) for i in range(cnt)]
                h2d_end_event.record(stream2)

            stream1.synchronize()
            stream2.synchronize()

            d2d_time = d2d_start_event.elapsed_time(d2d_end_event) / 1000  # 秒
            h2d_time = h2d_start_event.elapsed_time(h2d_end_event) / 1000  # 秒

            print(f"D2D time : {d2d_time}")
            print(f"H2D time : {h2d_time}")
            print(f"D2D Throughput: {4 * cnt * data0[0].numel() / 1024/1024/1024 / (d2d_time)} GB/s")
            print(f"H2D Throughput: {4 * cnt * data2[0].numel() / 1024/1024/1024 / (h2d_time)} GB/s")

if __name__ == "__main__":
    # d2d_throughput(10)
    # h2d_throughput(10)
    h2d_d2d_throughput(10)

    print(f"run h2d_d2d.py successfully !!!")
