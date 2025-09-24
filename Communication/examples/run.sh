#!/bin/bash

torchrun --nnodes=1 --nproc_per_node=2 send_recv.py