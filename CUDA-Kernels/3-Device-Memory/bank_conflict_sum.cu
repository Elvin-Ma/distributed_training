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

// 非连续的归约求和：Bank Conflict
__global__ void BC_addKernel(const int *a, int *r) {
    // 每个block里thread的个数
    __shared__ int cache[ThreadsPerBlock];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int cacheIndex = threadIdx.x;

    // copy data to shared memory from global memory
    if (tid < N) {
        cache[cacheIndex] = a[tid];
    } else {
        cache[cacheIndex] = 0;
    }
    __syncthreads();

    // add these data using reduce
    for (int i = 1; i < blockDim.x; i *= 2) {
        // 由近到远的累加, 两个相邻threadIdx.x 在shared memory 的索引相差index
        int index = 2 * i * cacheIndex;
        if (index < blockDim.x) {
            // 相邻线程index > 1 且为偶数，必定会Bank Conflict
            cache[index] += cache[index + i];
        }
        __syncthreads();
    }

    // copy the result of reduce to global memory
    if (cacheIndex == 0) {
        // 每个block 都往外copy
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
    BC_addKernel<<<dimGrid, dimBlock, ThreadsPerBlock * sizeof(int)>>>(d_a, d_r);

    int kernel_repeat = 20;
    float kernel_time = 0.0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < kernel_repeat; i++) {
        BC_addKernel<<<dimGrid, dimBlock, ThreadsPerBlock * sizeof(int)>>>(d_a, d_r);
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