#include <stdio.h>
#include <cuda_runtime.h>

// 定义检查CUDA调用错误的宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// CUDA核函数
__global__ static void test(int lid) {
    int tid = threadIdx.x;
    int value = tid;
    // 使用__shfl_sync函数进行数据交换
    int ret = __shfl_sync(0xFFFFFFFF, value, 5, 32);
    // 输出每个线程在数据交换前后的值, 只能在 __global__ 函数中使用
    // 如果在__device__函数中使用, 会出现编译错误
    printf("thread id = %d, before shuffle value = %d, after shuffle value=%d\n", tid, value, ret);
}

int main() {
    // 启动CUDA核函数
    test<<<1, 32>>>(0);
    // 检查核函数启动是否出错
    CUDA_CHECK(cudaGetLastError());
    // 同步设备
    CUDA_CHECK(cudaDeviceSynchronize());
    // 重置CUDA设备
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}