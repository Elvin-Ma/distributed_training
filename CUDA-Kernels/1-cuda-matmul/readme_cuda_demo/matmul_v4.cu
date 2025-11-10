#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>

const int TM = 4;                 // 一个线程处理4行元素
const int TN = 4;                 // 一个线程处理4列元素
const int BLOCK_DIM_x = 32;       // BlockDim.x = 32
const int BLOCK_DIM_y = 32;       // BlockDim.y = 32
const int BM = TM * BLOCK_DIM_x;  // 一个block处理的总行数 : 128
const int BN = TN * BLOCK_DIM_y;  // 一个block处理的总列数 : 128
const int BK = 8;                 // 一个Blcok处理的K : 8

double get_walltime() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}

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

template <int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_kernel_v4(float *dA, float *dB, float *dC, int M, int K, int N) {
    __shared__ float SA[BM * BK];                     // share memory 大小同上
    __shared__ float SB[BK * BN];                     // share memory 大小同上

    int indA = TM * (blockIdx.x * blockDim.x);        // C中一个block 起始行号
    int indB = TN * (blockIdx.y * blockDim.y);        // C中一个block 起始列号
    int width = (K + BK - 1) / BK;                    // K方向可以分成几分

    float tmp[TM * TN] = {0.0f};                      // 一个线程处理的寄存器

    // 此处时关键改进 ：将 32 * 32 = 1024 个线程，重新构造为128 * 8 和 8 * 128, 分别用于加载dA和dB的数据
    int tid = threadIdx.x + threadIdx.y * blockDim.x; // 一个block 处理的tid 号 ：0 ~ 1023
    int smem_a_m = tid % 128;                         // 一个tid 在SA中的行号, 结合indA 可得在A中的行号
    int smem_a_k = tid / 128;                         // 一个tid 在SA中的列号, 0~8
    int smem_b_k = tid % 8;                           // 一个tid 在SB中的行号
    int smem_b_n = tid / 8;                           // 一个tid 在SB中的列号
    for (int ph = 0; ph < width; ph++) {
        if (indA + smem_a_m < M && smem_a_k + ph * BK < K)  {            
            SA[smem_a_m * BK + smem_a_k] = dA[(indA + smem_a_m) * K + smem_a_k + ph * BK]; // 加载A的元素到SA
        } else {
            SA[smem_a_m * BK + smem_a_k] = 0.0f;
        }

        if (indB + smem_b_n < N && smem_b_k + ph * BK < K) {
            SB[smem_b_k * BN + smem_b_n] = dB[(smem_b_k + ph * BK) * N + indB + smem_b_n]; // 加载B的元素到SB
        } else {
            SB[smem_b_k * BN + smem_b_n] = 0.0f;
        }

        __syncthreads(); // 进行同步，确保数据加载完成

        for (int index_q = 0; index_q < TM; index_q++) {
            for (int index_v = 0; index_v < TN; index_v++) {
                int reg_c_m = threadIdx.x * TM + index_q; // 在SA中的行
                int reg_c_n = threadIdx.y * TN + index_v; // 在SB中的列
                for (int index_k = 0; index_k < BK; index_k++) {
                    tmp[index_q * TN + index_v] += SA[reg_c_m * BK + index_k] * SB[index_k * BN + reg_c_n]; 
                }
            }
        }
        __syncthreads();
    }

    for (int index_q = 0; index_q < TM; index_q++) {
        for (int index_v = 0; index_v < TN; index_v++) {
            int reg_c_m = threadIdx.x * TM + index_q;
            int reg_c_n = threadIdx.y * TN + index_v;
            if (indA + index_q < M && indB + index_v < N) {
                dC[(indA + reg_c_m) * N + indB + reg_c_n] = tmp[index_q * TN + index_v]; // 结果写回到dC
            }
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void matmul_kernel_v4_pro(float *dA, float *dB, float *dC, int M, int K, int N) {
    __shared__ float SA[BM][BK];  // share memory 128 * 8 
    __shared__ float SB[BK][BN];  // share memory 8 * 128 

    int indA = TM * (blockIdx.x * blockDim.x); // C中一个block 起始行号
    int indB = TN * (blockIdx.y * blockDim.y); // C中一个block 起始列号
    int width = (K + BK - 1) / BK;             // K方向可以分成几分

    float tmp[TM][TN] = {0.0f};                       // 一个线程处理的寄存器
    int tid = threadIdx.x + threadIdx.y * blockDim.x; // 一个block 处理的tid 号 ：0 ~ 1023
    int smem_a_m = tid % 128;                         // 一个tid 在SA中的行号, 结合indA 可得在A中的行号
    int smem_a_k = tid / 128;                         // 一个tid 在SA中的列号, 0~8
    int smem_b_k = tid % 8;                           // 一个tid 在SB中的行号
    int smem_b_n = tid / 8;                           // 一个tid 在SB中的列号
    for (int ph = 0; ph < width; ph++) {
        if (indA + smem_a_m < M && smem_a_k + ph * BK < K)  {
           // SA[smem_a_m * BK + smem_a_k] = dA[(indA + smem_a_m) * K + smem_a_k + ph * BK]; // 加载A的元素到SA
           SA[smem_a_m][smem_a_k] = dA[(indA + smem_a_m)*K + ph*BK + smem_a_k];
        } else {
            SA[smem_a_m][smem_a_k] = 0.0f;
        }

        if (indB + smem_b_n < N && smem_b_k + ph * BK < K) {
            // SB[smem_b_k * BN + smem_b_n] = dB[(smem_b_k + ph * BK) * N + indB + smem_b_n]; // 加载B的元素到SB
            SB[smem_b_k][smem_b_n] = dB[(smem_b_k + ph * BK) * N + indB + smem_b_n]; // 加载B的元素到SB
        } else {
            // SB[smem_b_k * BN + smem_b_n] = 0.0f;
            SB[smem_b_k][smem_b_n] = 0.0f;
        }

        __syncthreads(); // 进行同步，确保数据加载完成

        for (int index_q = 0; index_q < TM; index_q++) {
            for (int index_v = 0; index_v < TN; index_v++) {
                int reg_c_m = threadIdx.x * TM + index_q; // 在SA中的行
                int reg_c_n = threadIdx.y * TN + index_v; // 在SB中的列
                for (int index_k = 0; index_k < BK; index_k++) {
                    // tmp[index_q * TN + index_v] += SA[reg_c_m * BK + index_k] * SB[index_k * BN + reg_c_n]; 
                    tmp[index_q][index_v] += SA[reg_c_m][index_k] * SB[index_k][reg_c_n];
                }
            }
        }
        __syncthreads();
    }

    for (int index_q = 0; index_q < TM; index_q++) {
        for (int index_v = 0; index_v < TN; index_v++) {
            int reg_c_m = threadIdx.x * TM + index_q;
            int reg_c_n = threadIdx.y * TN + index_v;
            if (indA + index_q < M && indB + index_v < N) {
                // dC[(indA + reg_c_m) * N + indB + reg_c_n] = tmp[index_q * TN + index_v]; // 结果写回到dC
                dC[(indA + reg_c_m) * N + indB + reg_c_n] = tmp[index_q][index_v]; // 结果写回到dC
            }
        }
    }
}

void matmul_cuda_v4(float *hostA, float *hostB, float *hostC, int M, int K, int N) {
    double st, ela;
    st = get_walltime();

    float *dA, *dB, *dC;
    cudaMalloc((void **)&dA, M * K * sizeof(float));
    cudaMalloc((void **)&dB, N * K * sizeof(float));
    cudaMalloc((void **)&dC, M * N * sizeof(float));

    cudaMemcpy(dA, hostA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hostB, N * K * sizeof(float), cudaMemcpyHostToDevice);

    int num_blocks_x = (M + BM - 1) / BM;
    int num_blocks_y = (N + BN - 1) / BN;
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_blocks_x, num_blocks_y, 1);
    int repeat = 20;

    matmul_kernel_v4_pro<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    
    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < repeat; i++) {       
        matmul_kernel_v4_pro<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);      
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        // Possibly: exit(-1) if program cannot continue....
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
    ela = get_walltime() - st;
    printf("M-K-N: %d-%d-%d\n", M, K, N);
    printf("GPU use time: %.4f second\n", ela);
    printf("kernel time: %.4f second, %.4f ms\n", ker_time / (repeat * 1000.), ker_time / repeat);
    printf("grid dim: %d, %d, %d\n", grid_dim.x, grid_dim.y, grid_dim.z);
    printf("block dim: %d, %d, %d\n", block_dim.x, block_dim.y, block_dim.z);
}

int main() {
    float *hostA, *hostB, *hostC, *serialC;
    int M = 1024;
    int K = 1024;
    int N = 1024;

    hostA = (float *)malloc(M * K * sizeof(float));
    hostB = (float *)malloc(N * K * sizeof(float));
    hostC = (float *)malloc(M * N * sizeof(float));
    serialC = (float *)malloc(M * N * sizeof(float));

    for (int i = 0; i < M * K; i++) {
        hostA[i] = i % 3;
    }
    for (int i = 0; i < N * K; i++)
    {
        hostB[i] = i % 3;
    }

    matmul_cuda_v4(hostA, hostB, hostC, M, K, N);

    double st, ela;
    st = get_walltime();
    matmul_cpu(hostA, hostB, serialC, M, K, N);
    ela = get_walltime() - st;
    printf("CPU time:%.2f second\n", ela);
    compare(hostC, serialC, M, N);
    free(hostA);
    free(hostB);
    free(hostC);
    free(serialC);
    return 0;
} 