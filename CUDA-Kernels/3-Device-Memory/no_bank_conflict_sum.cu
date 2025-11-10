/*
  A100(Amphere 架构): nvcc -arch=sm_80 -o a.out no_bank_conflict_sum.cu
*/

#include <iostream>
#include <sys/time.h>
#include <stdio.h>
#include <cuda_runtime.h>  // CUDA运行时API

// 定义每个block里thread的个数
const int N = 1024;
#define ThreadsPerBlock 256

double get_walltime() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}

// 折半规约内核
__global__ void halfWarpReductionKernel(const int *a, int *r) {
    extern __shared__ int cache[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIndex = threadIdx.x;

    // 将数据从全局内存加载到共享内存
    if (tid < N) {
        cache[cacheIndex] = a[tid];
    } else {
        cache[cacheIndex] = 0; // 初始化为0，防止未定义行为
    }
    __syncthreads();

    // 折半规约
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (cacheIndex < stride) {
            cache[cacheIndex] += cache[cacheIndex + stride];
        }
        __syncthreads();
    }

    // 将每个block的结果写回全局内存
    if (cacheIndex == 0) {
        r[blockIdx.x] = cache[0];
    }
}

int main() {
    // 定义数组大小
    const int numBlocks = (N + ThreadsPerBlock - 1) / ThreadsPerBlock;

    // 主机端数据
    int *h_a = new int[N];
    int *h_r = new int[numBlocks];

    // 初始化数据
    int golden = 0;
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        golden += i;
    }

    // 设备端数据
    int *d_a, *d_r;
    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_r, numBlocks * sizeof(int));

    // 复制数据到设备
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);

    // 启动核函数
    dim3 dimBlock(ThreadsPerBlock);
    dim3 dimGrid(numBlocks);

    // warm up
    halfWarpReductionKernel<<<dimGrid, dimBlock, ThreadsPerBlock * sizeof(int)>>>(d_a, d_r);

    int kernel_repeat = 20;
    float kernel_time = 0.0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < kernel_repeat; i++) {
        halfWarpReductionKernel<<<dimGrid, dimBlock, ThreadsPerBlock * sizeof(int)>>>(d_a, d_r);
    }

    // 检查内核错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernel_time, start, stop);

    printf("kernel time: %.4f ms\n", kernel_time / kernel_repeat);

    // 复制结果到主机
    cudaMemcpy(h_r, d_r, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果
    for (int i = 0; i < numBlocks; i++) {
        std::cout << "Block " << i << " result: " << h_r[i] << std::endl;
    }

    int result = 0;
    for (int i = 0; i < numBlocks; i++) {
        result += h_r[i];
    }

    std::cout << "Result: " << result << std::endl;
    std::cout << "Golden result: " << golden << std::endl;

    // 释放内存
    cudaFree(d_a);
    cudaFree(d_r);
    delete[] h_a;
    delete[] h_r;

    return 0;
}