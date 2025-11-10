#!/bin/bash

nvcc cutlass_gemm_example.cu -o gemm_test \
  -I/root/projects/cuda-and-gpu/GroupedGemm/cutlass/include \
  -I/root/projects/cuda-and-gpu/GroupedGemm/cutlass/tools/util/include \
  -arch=sm_80  # 根据 GPU 架构调整（如 A100 用 sm_80，RTX 3090 用 sm_86）
