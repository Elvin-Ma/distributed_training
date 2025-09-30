#!/bin/bash

# os.environ['NCCL_BLOCKING_WAIT'] = '1'
# os.environ['NCCL_TIMEOUT'] = '6'  # 设置超时时间为60秒

export NCCL_TIMEOUT=3
# export NCCL_DEBUG=INFO
export NCCL_BLOCKING_WAIT=1

# torchrun --nnodes=1 --nproc_per_node=2 send_recv.py
torchrun --nproc_per_node=2 --standalone sub_process_comm.py
