#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>

template <typename T, int warpSize = 32>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int mask = warpSize/2; mask > 0; mask >>= 1) {
        T shfl_val = __shfl_xor_sync(0xffffffff, val, mask, warpSize);
        val += shfl_val;
    }
    return val;
}

template <typename T>
__global__ void reduce_sum_kernel(T* dst, const T* src, int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    T val = (tid < n) ? src[tid] : static_cast<T>(0);

    val = warp_reduce_sum(val);

    if ((threadIdx.x & (warpSize-1)) == 0) {
        atomicAdd(dst, val);
    }
}

template <typename T>
void launch_reduce_sum(T* d_dst, const T* d_src, int n, int block_size=1024) {
    int grid_size = (n + block_size - 1) / block_size;
    reduce_sum_kernel<<<grid_size, block_size>>>(d_dst, d_src, n);
    cudaDeviceSynchronize();
}

int main() {
    int n = 1 << 20; // 1 million elements
    std::vector<int> h_data(n);
    for (int i = 0; i < n; ++i) h_data[i] = i % 100; // Generate test data

    int* d_src, *d_dst;
    cudaMalloc(&d_src, n * sizeof(int));
    cudaMalloc(&d_dst, sizeof(int));
    cudaMemcpy(d_src, h_data.data(),  n * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize destination
    cudaMemset(d_dst, 0, sizeof(int));

    launch_reduce_sum<int>(d_dst, d_src, n);

    int h_result;
    cudaMemcpy(&h_result, d_dst, sizeof(int), cudaMemcpyDeviceToHost);

    // Verification
    int expected = 0;
    for (int i = 0; i < n; ++i) expected += h_data[i];
    std::cout << "CUDA Result: " << h_result << std::endl;
    std::cout << "Expected:    " << expected << std::endl;
    std::cout << "Correct: " << (h_result == expected ? "Yes" : "No") << std::endl;

    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
}