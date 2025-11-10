#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>

const int TM = 4;                 // 一个线程处理4行元素
const int TN = 4;                 // 一个线程处理4列元素
const int BLOCK_DIM_x = 32;       // BlockDim.x = 32
const int BLOCK_DIM_y = 32;       // BlockDim.y = 32
const int BM = TM * BLOCK_DIM_x;  // 一个block处理的总行数 : 128
const int BN = TN * BLOCK_DIM_y;  // 一个block处理的总列数 : 128
const int BK = 8;                 // 一个Blcok处理的
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

template <int BM, int BN, int BK, int TM, int TN> // 128, 128, 8, 4, 4
__global__ void matmul_kernel_v3(float *dA, float *dB, float *dC, int M, int K, int N) {
    __shared__ float SA[BM * BK];                            // 一个block可以处理的元素个数: 128 * 8
    __shared__ float SB[BK * BN];                            // 一个block可以处理的元素个数：8 * 128
    int indA = TM * (threadIdx.x + blockIdx.x * blockDim.x); // 当前线程处理的是A中的第几行，也是结果C的第几行
    int indB = TN * (threadIdx.y + blockIdx.y * blockDim.y); // 当前线程处理的是B中的第几列，也是结果C的第几列
    int width = (K + BK - 1) / BK;                           // K 方向的份数
    float tmp[TM * TN] = {0.0f};                             // 寄存器变量，一个线程需要的寄存器

    for (int ph = 0; ph < width; ph++) {
        // 加载矩阵A的元素到shared memory
        for (int index_q = 0; index_q < TM; index_q++) {     // 遍历TM 加载数据
            for (int index_k = 0; index_k < BK; index_k++) { // 变量BK 加载数据
                if (indA + index_q < M && index_k + ph * BK < K) {
                    SA[(threadIdx.x * TM + index_q) * BK + index_k] = dA[(indA + index_q) * K + index_k + ph * BK]; // 逐元素加载数据到share memory，所有的threadIdx
  时进行
                } else {
                    SA[(threadIdx.x * TM + index_q) * BK + index_k] = 0.0f;
                }
            }
        }

        __syncthreads();

        // 加载矩阵B中的元素到 shared memory
        for (int index_v = 0; index_v < TN; index_v++) {
            for (int index_k = 0; index_k < BK; index_k++) {
                if (indB + index_v < N && index_k + ph * BK < K) {
                    SB[index_k * BN + threadIdx.y * TN + index_v] = dB[(index_k + ph * BK) * N + indB + index_v]; // 逐元素加载数据到SB
                } else {
                    SB[index_k * BN + threadIdx.y * TN + index_v] = 0.0f;
                }
            }
        }

        __syncthreads();

        for (int index_q = 0; index_q < TM; index_q++) {
            for (int index_v = 0; index_v < TN; index_v++) {
                for (int index_k = 0; index_k < BK; index_k++) {
                    // 计算一个线程中tmp的每个元素的值，就是按照刚才数据加载的方式来进行的
                    tmp[index_q * TN + index_v] += SA[(threadIdx.x * TM + index_q) * BK + index_k] * SB[index_k * BN + threadIdx.y * TN + index_v];
                }
            }
        }

        __syncthreads();
    }

    for (int index_q = 0; index_q < TM; index_q++) {
        for (int index_v = 0; index_v < TN; index_v++) {
            if (indA + index_q < M && indB + index_v < N) {
                dC[(indA + index_q) * N + indB + index_v] = tmp[index_q * TN + index_v]; // 将tmp数据写到output : dC, indA, indB 是output 的全局坐标

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

    matmul_cuda_v3(hostA, hostB, hostC, M, K, N);

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