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

template <int BLOCK_DIM>
__global__ void matmul_kernel_v2(float *dA, float *dB, float *dC, int M, int K, int N) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;

    float tmp = 0.0f;
    __shared__ float SA[BLOCK_DIM][BLOCK_DIM];
    __shared__ float SB[BLOCK_DIM][BLOCK_DIM];
    int width = (K + BLOCK_DIM - 1) / BLOCK_DIM;

    // 两层for循环
    for (int ph = 0; ph < width; ph++) {
        if (row < M && threadIdx.y + ph * BLOCK_DIM < K) {
            SA[threadIdx.x][threadIdx.y] = dA[row * K + threadIdx.y + ph * BLOCK_DIM];
        } else {
            SA[threadIdx.x][threadIdx.y] = 0.0f;
        }

        if (threadIdx.x + ph * BLOCK_DIM < K && col < N) {
            SB[threadIdx.x][threadIdx.y] = dB[(threadIdx.x + ph * BLOCK_DIM) * N + col];
        } else {
            SB[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        // 只是一个block中的数据不用重复从HBM读取, 直接从Share Memory 获取即可
        for (int s = 0; s < BLOCK_DIM; s++) {
            tmp += SA[threadIdx.x][s] * SB[s][threadIdx.y];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        dC[row * N + col] = tmp;
    }
}

void matmul_cuda_v2(float *hostA, float *hostB, float *hostC, int M, int K, int N) {
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
    matmul_kernel_v2<32><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N); // warmup

    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < repeat; i++) {
        matmul_kernel_v2<32><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
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

    matmul_cuda_v2(hostA, hostB, hostC, M, K, N);

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