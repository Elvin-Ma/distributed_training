#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>

// 挂钟时间
double get_walltime() { 
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}

// 数据对比
void compare(float *hostC, float *serialC, int M, int N) {
    float error = 0;
    bool tmp = true;
    for (int i = 0; i < M * N; i++) {
        error = fmax(error, fabs(hostC[i] - serialC[i]));
        if (error > 1e-5) {
            tmp = false;
            printf("error:hostC[%d] = %.3f, serialC[%d] = %.3f\n", i, hostC[i], i, serialC[i]);
            break;
        }
    }
    if (tmp) {
        printf("GPU output all right\n");
    }
}

void matmul_cpu(float *hostA, float *hostB, float *hostC, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float tmp = 0;
            for (int s = 0; s < K; s++) {
                tmp += hostA[i * K + s] * hostB[s * N + j];
            }
            hostC[i * N + j] = tmp;
        }
    }
}

__global__ void matmul_kernel_v1(float *dA, float *dB, float *dC, int M, int K, int N) {
    int row = threadIdx.x + blockIdx.x * blockDim.x; // 当前block 的线程在全局矩阵中的行索引
    int col = threadIdx.y + blockIdx.y * blockDim.y; // 当前block的线程在全局矩阵中的列索引
    float tmp = 0;
    if (row < M && col < N) {
        for (int s = 0; s < K; s++) {
            tmp += dA[row * K + s] * dB[s * N + col];
        }
        dC[row * N + col] = tmp;
    }
}

void matmul_cuda_v1(float *hostA, float *hostB, float *hostC, int M, int K, int N) {
    double st, ela;
    st = get_walltime(); // start time 挂钟时间

    float *dA, *dB, *dC;
    cudaMalloc((void **)&dA, M * K * sizeof(float)); // 显存申请 
    cudaMalloc((void **)&dB, N * K * sizeof(float));
    cudaMalloc((void **)&dC, M * N * sizeof(float));

    cudaMemcpy(dA, hostA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hostB, N * K * sizeof(float), cudaMemcpyHostToDevice);

    int BLOCK_DIM_x = 32;
    int BLOCK_DIM_y = 32;
    int num_blocks_x = (M + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
    int num_blocks_y = (N + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_blocks_x, num_blocks_y, 1);
    int repeat = 20;
    matmul_kernel_v1<<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N); // warmup

    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < repeat; i++) {
        matmul_kernel_v1<<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time

    cudaMemcpy(hostC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    ela = get_walltime() - st; // 20次的总时间: 包含H2D和Free的时间
    printf("M-K-N: %d-%d-%d\n", M, K, N);
    printf("GPU use time: %.4f second\n", ela);
    printf("kernel time: %.4f second, %.4f ms\n", ker_time / (repeat * 1000.), ker_time / repeat);
    printf("grid dim: %d, %d, %d\n", grid_dim.x, grid_dim.y, grid_dim.z);
    printf("block dim: %d, %d, %d\n", block_dim.x, block_dim.y, block_dim.z);
}

int main() {
    float *hostA, *hostB, *hostC, *cpuC;
    int M = 1024;
    int K = 1024;
    int N = 1024;

    hostA = (float *)malloc(M * K * sizeof(float));
    hostB = (float *)malloc(N * K * sizeof(float));
    hostC = (float *)malloc(M * N * sizeof(float));
    cpuC = (float *)malloc(M * N * sizeof(float));

    for (int i = 0; i < M * K; i++) {
        hostA[i] = i % 3;
    }

    for (int i = 0; i < N * K; i++) {
        hostB[i] = i % 3;
    }

    matmul_cuda_v1(hostA, hostB, hostC, M, K, N);

    double st, ela;
    st = get_walltime();
    matmul_cpu(hostA, hostB, cpuC, M, K, N);
    ela = get_walltime() - st;
    compare(hostC, cpuC, M, N);
    printf("CPU time:%.2f second\n", ela);
    free(hostA);
    free(hostB);
    free(hostC);
    free(cpuC);
    return 0;
}