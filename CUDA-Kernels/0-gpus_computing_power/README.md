# 1 Cuda Install
## 1.1 cuda下载
- [cuda版本入口](https://developer.nvidia.com/cuda-toolkit-archive)
- [各平台选择](https://developer.nvidia.com/cuda-12-1-0-download-archive)

## 1.2 cuda 安装
- 安装指令
```ptyhon
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run
```

## 1.3 环境变量设置
- PATH includes /usr/local/cuda-12.1/bin
- LD_LIBRARY_PATH includes /usr/local/cuda-12.1/lib64, or, add /usr/local/cuda-12.1/lib64 to /etc/ld.so.conf and run ldconfig as root

## 1.4 卸载
- To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-12.1/bin


# 2 CUDA_COMPUTE_CAPABILITY
以下是CUDA计算能力（Compute Capability）与NVIDIA GPU架构的对应关系，以及代表性GPU型号的总结：<br>

| 架构名称 | 计算能力 | 代表GPU型号 | 说明 |
| :--: | :--: | :--: | :--: |
| Tesla | 1.0 – 1.3 | GeForce 8800 GTX, Tesla C870 | 初代CUDA架构，支持基础并行计算。 |
| Fermi | 2.0 – 2.1 | GeForce GTX 480, Tesla C2050 | 引入全局内存缓存和ECC支持。 |
| Kepler | 3.0 – 3.7 | GeForce GTX 680, Tesla K80 | 动态并行和Hyper-Q技术，提升能效。 |
| Maxwell | 5.0 – 5.3 | GeForce GTX 750 Ti, GTX 980 | 优化能效，引入统一内存技术。 |
| Pascal | 6.0 – 6.2 | Tesla P100, GTX 1080 | 支持NVLink和HBM2显存，适用于深度学习。 |
| Volta | 7.0 – 7.2 | Tesla V100, Titan V | 引入Tensor Core，专为AI和高性能计算优化。 |
| Turing | 7.5 | GeForce RTX 2080, Tesla T4 | 支持光线追踪和DLSS，RT Core首次亮相。 |
| Ampere | 8.0, 8.6, 8.7 | A100 (8.0), RTX 3080 (8.6), A30 (8.7) | 第三代Tensor Core，支持稀疏计算和多实例GPU。 |
| Ada Lovelace | 8.9 | GeForce RTX 4090, RTX 6000 Ada | 改进光线追踪和AI性能，第四代Tensor Core。 |
| Hopper | 9.0 | H100, H200 | 面向数据中心的下一代架构，支持Transformer引擎和更高吞吐量。 |

- [nvidia compute capability](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-8-x)


# 3 算力对比

| 计算类型 | 精度 | A100 算力 | H100 算力 | 提升倍数 | 关键说明 |
|----------|------|-----------|-----------|----------|----------|
| 峰值理论算力 | - | - | - | - | - |
| CUDA Core | FP64 | 9.7 TFLOPS | 30.6 TFLOPS | ~3.2x | H100 FP64 CUDA Core 数量大增 |
| CUDA Core | FP32 | 19.5 TFLOPS | 60.9 TFLOPS | ~3.1x | 传统科学计算与图形 |
| CUDA Core | FP16 | 39 TFLOPS | 121.8 TFLOPS | ~3.1x | - |
| Tensor Core | FP64 | 不适用 | 67.0 TFLOPS | N/A | H100 新增 FP64 Tensor Core |
| Tensor Core | TF32 | 156 TFLOPS | ~989 TFLOPS | ~6.3x | AI 训练的主力精度 |
| Tensor Core | FP16 | 312 TFLOPS | ~1.98 PetaFLOPS | ~6.3x | AI 训练与推理 |
| Tensor Core | FP8 | 不适用 | ~3.95 PetaFLOPS | N/A | Hopper 新增，AI 推理与训练的未来 |
| Tensor Core | INT8 | 624 TOPS | ~3.96 PetaOPS | ~6.3x | AI 推理 |


# 4 Vector Core and Tensor Core

**这是两者最本质的不同** <br>
- Vector Core 执行的是单指令多数据流（SIMD） 操作;
- 而Tensor Core执行的是单指令多矩阵乘（SIMMD） 操作.

## 4.1 Tensor Core
Tensor Core是NVIDIA从Volta架构开始引入的专用硬件单元，旨在极高效地执行小型、高精度的矩阵乘累加运算。

- 操作对象：是小型的稠密矩阵块（例如4x4, 8x8, 16x16）。
- 操作类型：核心操作是 D = A * B + C，其中A, B, C, D都是矩阵。这是一个**融合乘加（FMA） 操作**，但是在整个矩阵层面进行的。
- 并行粒度：它在一个时钟周期内直接计算出一个小的结果矩阵，而不是分别计算各个元素。例如，一个Tensor Core操作可以一次性完成一个4x4矩阵乘另一个4x4矩阵，再加到一个4x4矩阵上，总共64个FMA操作。

**如何调用Tensor Core 呢？**
TensorCore 不是自动触发的，使用Tensor Core需要显式地调用特定的API、内建函数或库， 有以下几种方式来调用: <br>

- CUDA C++：使用 wmma (Warp Matrix Multiply Accumulate) 命名空间下的API。
- cuBLAS / cuDNN：使用这些库中提供的特定API（如 cublasGemmEx）并指定计算类型（如 CUDA_R_16F, CUDA_R_32F）。
- PTX汇编：直接使用Tensor Core相关的指令（如 mma.sync.aligned.m16n8k4）。

**Tensor Core 与 Warp : Tensor Core的操作是以Warp为单位的协作操作**

- 协作性：一个Warp（32个线程）需要协同工作来共同**提供输入矩阵A和B的数据**，并共同接收输出矩阵D的结果。线程之间**不再是独立的**，它们共享任务。

- 数据分布：输入矩阵A和B的元素被分布在Warp内所有32个线程的寄存器中。同样，结果矩阵D也被分布在所有线程的寄存器中。

- 同步性：Tensor Core指令（如 mma.sync）是隐式同步的。这意味着Warp中的所有线程必须一起到达这个指令点，然后硬件会执行这个矩阵运算，最后所有线程一起得到结果。


# 5 GPU specification
- [A100 specification](https://github.com/user-attachments/assets/1f2ae1ac-0e85-484b-8256-99d74af725a0)

- [H100 specification](https://github.com/user-attachments/assets/6f2fa572-24a3-4a23-99cd-467d21599d20)

- [A100 vs H100 in HPC](https://github.com/user-attachments/assets/b6e04fb6-ba86-4c91-924b-bde396175871)

- [A100 vs H100 in AI](https://github.com/user-attachments/assets/96f8f80f-6ace-457e-957f-db9a967b2313)

- [A100 白皮书](https://images.nvidia.cn/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)

- [H100 白皮书](https://resources.nvidia.com/en-us-tensor-core)

- [Tesla V100 白皮书](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)
