#torchrun --nproc-per-node 4 --standalone test.py

import logging
import torch
import time
import random
from datetime import timedelta
import torch.distributed as dist

logging.basicConfig(
    format="%(asctime)s - pid: %(process)d - %(filename)s : %(lineno)d - %(funcName)s  - %(levelname)s - %(message)s",
    level=logging.INFO
)

def timeout(test_hang = False):
    '''
    通信组pg_rank=0 时 : 会hang; 其他进程在时触发timeout;
    RuntimeError: [1] is setting up NCCL communicator and retrieving NcclUniqueId from [0]
    via c10d key-value store by key '0', but store->get('0') got error:
    wait timeout after 10000ms, keys: //worker/attempt_0/default_pg/0//1//cuda//0
    '''
    dist.init_process_group(backend="mccl", timeout=timedelta(seconds=10))
    rank = dist.get_rank()

    """模拟某个进程卡住导致的 barrier 超时"""
    torch.cuda.set_device(rank)

    logging.info(f"Rank {rank} started")

    data = torch.randn(4,5).cuda()

    pg = dist.new_group(ranks=[2,3], backend="mccl")

    ranks = [2 if test_hang else 3]

    if rank in ranks:
        logging.info(f"Rank {rank} start all_reduce")
        dist.all_reduce(data, group=pg)
        logging.info(f"Rank {rank} all_reduce done")
    else:
        time.sleep(10000)

    logging.info(f"Rank {rank} run finish.")

if __name__ == "__main__":
    timeout()

