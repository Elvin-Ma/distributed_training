import os
import time
import torch
import datetime
import threading
import torch.distributed as dist
import torch.multiprocessing as mp

def all_reduce_with_thread_timeout(tensor, timeout_seconds=3):

    def _all_reduce(pg = None):
        try:
            work = dist.all_reduce(tensor, group=pg, async_op=True)
            work.wait()
        except Exception as e:
            print(f"=========rank 0 all_reduce thread catch exception: {e}========")

    print(f"=========rank 0 start all_reduce========")
    thread = threading.Thread(target=_all_reduce)
    thread.daemon = True
    thread.start()
    print(f"=========rank 0 all_reduce thread start ========")
    thread.join(timeout=timeout_seconds)
    print(f"=========rank 0 all_reduce thread join ========")

    if thread.is_alive():
        # 线程仍在运行，说明超时
        # raise RuntimeError(f"All_reduce 操作在 {timeout_seconds} 秒后超时")
        print(f"=========rank 0 all_reduce timeout and start new process group ========")
        group = dist.new_group([0], backend='nccl')
        thread = threading.Thread(target=_all_reduce, args=(group,))
        thread.daemon = True
        thread.start()
        print(f"=========rank 0 all_reduce thread start in new process group ========")
        thread.join(timeout=timeout_seconds)
        print(f"=========rank 0 all_reduce thread join in new process group ========")

        return tensor

    return tensor

if __name__ == "__main__":
    torch.distributed.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    tensor = torch.randn(100, device=device)

    if rank == 0:
        try:
            result = all_reduce_with_thread_timeout(tensor, 10)
            print(f"====== final result: {tensor}")
        except RuntimeError as e:
            print(f"rank 0捕获到超时异常: {e}")
    else:
        print(f"=======rank 1 : run finished !!!")
