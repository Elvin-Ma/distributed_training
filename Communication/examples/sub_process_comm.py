import os
import time
import torch
import logging
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import ProcessGroupNCCL, ProcessGroup

logging.basicConfig(
    format="%(asctime)s - pid: %(process)d - %(filename)s : %(lineno)d - %(funcName)s  - %(levelname)s - %(message)s",
    level=logging.INFO
)


def worker(tensor, pid):
    print(f"Process {pid}: tensor device: {tensor.device}")
    print(f"Process {pid}: tensor data: {tensor}")
    tensor[pid] = pid * 10

def update_tensor_on_sub_process():
    mp.set_start_method('spawn', force=True)

    tensor = torch.tensor([1, 2, 3, 4, 5]).cuda()
    print(f"Main before: {tensor}")

    p = mp.Process(target=worker, args=(tensor, 0))
    p.start()
    p.join()

    print(f"Main after: {tensor}")


def _create_pg(store, rank: int, world_size: int):
    opts = ProcessGroupNCCL.Options()
    opts.config.blocking = False

    pg = ProcessGroup(store, rank, world_size)
    pg._set_default_backend(ProcessGroup.BackendType.NCCL)

    backend_class = ProcessGroupNCCL(store, rank, world_size, opts)
    backend_class._set_sequence_number_for_group()
    backend_class.eager_connect_single_device(
        torch.device(torch.accelerator.current_device_index())
    )
    pg._register_backend(
        torch.device("cuda"), ProcessGroup.BackendType.NCCL, backend_class
    )
    return pg

def _worker(
    q: mp.Queue,
    store_addr: str,
    port: int,
    rank: int,
    world_size: int,
) -> None:
    logging.info(f"rank {rank}: _worker start !!!")
    torch.cuda.set_device(rank)

    logging.info(f"rank {rank}: getting tensor from queue start ...")
    tensor = q.get()
    logging.info(f"rank {rank}: getting tensor from queue done !!!")
    logging.info(f"rank {rank}: tensor ptr {tensor.data_ptr()}")

    store = dist.TCPStore(store_addr, port, world_size, rank == 0)

    prefix_store = dist.PrefixStore("nccl_patched", store)

    logging.info(f"rank {rank}: store created successfully !!!")

    pg_1 = _create_pg(store, rank, world_size)
    # pg_2 = _create_pg(prefix_store, rank, world_size)

    logging.info(f"rank {rank}: pg created successfully !!!")
    logging.info(f"rank {rank}: pg.size() {pg_1.size()}")

    for i in range(2):
        logging.info(f"rank {rank}: iteration {i}")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=pg_1)
        logging.info(f"rank {rank}: tensor {tensor}")
        # dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=pg_2)
        # logging.info(f"rank {rank}: tensor {tensor}")

    logging.info(f"rank {rank}: all_reduce done !!!")

def sub_process_allreduce(store_addr: str, port: int):
    rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))

    logging.info(f"rank {rank}: sub_process_allreduce start !!!")

    device = torch.device('cuda', rank)
    torch.cuda.set_device(rank)

    tensor = torch.zeros(1024, device=device) + rank
    tensor = tensor.share_memory_()
    logging.info(f"rank {rank}: tensor ptr {tensor.data_ptr()}")

    ctx = mp.get_context("spawn")

    q = ctx.Queue()
    q.put(tensor)

    p = ctx.Process(
        target=_worker,
        args=(
            q,
            store_addr,
            port,
            rank,
            world_size,
        ),
        daemon=False,
    )
    p.start()
    p.join()

    torch.cuda.synchronize()
    logging.info(f"rank {rank}: tensor {tensor}")
    logging.info(f"rank {rank}: run successfully !!!")


# torchrun --nproc_per_node=2 --standalone shared_memory.py
if __name__ == "__main__":
    # update_tensor_on_sub_process()
    sub_process_allreduce("127.0.0.1", 29502)
    print(f"run shared_memory.py successfully !!!")