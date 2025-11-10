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

# 3 GPU specification
- [A100 specification](https://github.com/user-attachments/assets/1f2ae1ac-0e85-484b-8256-99d74af725a0)

- [H100 specification](https://github.com/user-attachments/assets/6f2fa572-24a3-4a23-99cd-467d21599d20)

- [A100 vs H100 in HPC](https://github.com/user-attachments/assets/b6e04fb6-ba86-4c91-924b-bde396175871)

- [A100 vs H100 in AI](https://github.com/user-attachments/assets/96f8f80f-6ace-457e-957f-db9a967b2313)

- [A100 白皮书](https://images.nvidia.cn/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)

- [H100 白皮书](https://resources.nvidia.com/en-us-tensor-core)

- [Tesla V100 白皮书](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)


