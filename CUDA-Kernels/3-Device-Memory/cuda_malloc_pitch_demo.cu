#include <cstdio>
#include <cuda_runtime.h>

// 核函数：对二维浮点数组的每个元素加 1
__global__ void kernel(float* devData, size_t pitch, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height && col < width) {
        // 使用 pitch 计算行地址（转换为字节后偏移）
        float* rowPtr = (float*)((char*)devData + row * pitch);
        rowPtr[col] += 1.0f;
    }
}

int main() {
    const int width = 5;  // 列数（元素数量）
    const int height = 3; // 行数
    size_t byteWidth = width * sizeof(float); // 每行请求的字节宽度

    // 主机内存分配并初始化
    float hostData[height][width] = {
        {1, 2, 3, 4, 5},
        {6, 7, 8, 9, 10},
        {11, 12, 13, 14, 15}
    };

    // 设备内存分配（使用 cudaMallocPitch）
    float* devData;
    size_t pitch;
    cudaMallocPitch((void**)&devData, &pitch, byteWidth, height);

    // 将数据从主机拷贝到设备（使用 pitch）
    cudaMemcpy2D(
        devData, pitch,                   // 目标地址和行宽
        hostData, byteWidth,              // 源地址和行宽
        byteWidth, height,                // 每行复制的字节数和行数
        cudaMemcpyHostToDevice
    );

    // 启动核函数
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    kernel<<<grid, block>>>(devData, pitch, width, height);

    // 将结果拷贝回主机
    cudaMemcpy2D(
        hostData, byteWidth,              // 目标地址和行宽
        devData, pitch,                   // 源地址和行宽
        byteWidth, height,                // 每行复制的字节数和行数
        cudaMemcpyDeviceToHost
    );

    // 打印结果
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.1f ", hostData[i][j]);
        }
        printf("\n");
    }

    cudaFree(devData);
    return 0;
}
