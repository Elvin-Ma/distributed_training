#include <mma.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <sys/time.h>
#include <stdio.h>
#define max_function(a, b) ((a) > (b) ? (a) : (b))
double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}

/*
  nvcuda 是一个与 CUDA 设备运行时 相关的命名空间，
  通常用于支持 动态并行（Dynamic Parallelism） 和 设备端 launch 配置 等功能。
  它提供了一些底层接口来在 CUDA 内核中启动子内核（nested launch），
  适用于高级 CUDA 编程场景。

  nvcuda::launchDevice<KernelFn>(...) ： 用于从设备端调用另一个CUDA 内核。
  nvcuda::getRuntimeClassInstance() ：获取当前CUDA运行时类实例，用于动态并行内部机制;
  nvcuda::grid_group ：提供对当前网络(grid)的控制和同步功能；
  nvcuda::memcpyAsync/nvcuda::memzeroAsync: 在设备端异步执行内存拷贝或清零操作（需要统一虚拟寻址或托管内存支持）
  nvcuda::malloc/nvcuda::free : 在设备端动态分配/释放内存；
  nvcuda::printf : 可以在设备端使用，支持嵌套内核调试输出
  nvcuda::device_prinf : 另一种设备端打印方式，用于调试嵌套内核
 **/

using namespace nvcuda;
const int WMMA_M = 8;
const int WMMA_N = 8;
const int WMMA_K = 4;
const int warpSize = 32;

__global__ void col_wmma_ker(int M, int N, int K, double *dA, double *dB, double *dC, double alpha, double beta)
{
    int lda = M; // A=[M,K],索引(x,y) = x * K + y，列优先原则索引(x,y) = y * M + x
    int ldb = K;
    int ldc = M;

    int warpX = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpY = (blockIdx.y * blockDim.y + threadIdx.y);
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, double, wmma::col_major> left_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, double, wmma::col_major> right_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, double> c_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, double> mid_frag;

    // Initialize the output to zero
    wmma::fill_fragment(mid_frag, 0.0f);
    int aRow = warpX * WMMA_M; //(aRow, aCol),x方向数多少个WMMA_M
    int bCol = warpY * WMMA_N;
    for (int i = 0; i < K; i += WMMA_K)
    {

        int aCol = i;
        int bRow = i;
        if (aRow < M && aCol < K && bRow < K && bCol < N)
        {
            // 读取A,B矩阵里面子矩阵的元素
            wmma::load_matrix_sync(left_frag, dA + aRow + aCol * lda, lda);
            wmma::load_matrix_sync(right_frag, dB + bRow + bCol * ldb, ldb);
            // 子矩阵做乘法
            wmma::mma_sync(mid_frag, left_frag, right_frag, mid_frag);
        }
    }
    int cRow = warpX * WMMA_M;
    int cCol = warpY * WMMA_N;
    if (cRow < M && cCol < N)
    {
        wmma::load_matrix_sync(c_frag, dC + cRow + cCol * ldc, ldc, wmma::mem_col_major);
#pragma unroll                                        // 负责覆盖默认的循环展开，可以删除
        for (int i = 0; i < c_frag.num_elements; i++) // 使用c_frag.num_elements可以访问frag里面的元素
        {
            c_frag.x[i] = alpha * mid_frag.x[i] + beta * c_frag.x[i];
        }

        // Store the output
        wmma::store_matrix_sync(dC + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
    }
}

template <typename T>
__global__ void kernelMat(int M, int N, int K, T *dA, T *dB, T *dC)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N)
    {
        T sum = 0;
        for (int index = 0; index < K; index++)
        {
            sum += dA[row * K + index] * dB[index * N + col];
        }
        dC[row * N + col] = sum;
    }
}
template <typename Tout, typename Tin>
__global__ void convertType(Tout *out, Tin *in, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size)
    {
        out[idx] = in[idx];
    }
}
__global__ void transpose(int M, int N, double *input, double *output)
{
    int outputIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (outputIdx < M * N)
    {
        int inputIdx = (outputIdx % N) * M + outputIdx / N;
        output[outputIdx] = input[inputIdx];
    }
}
void Mat(int M, int N, int K, double *hA, double *hB, double *hC, double alpha, double beta)
{
    dim3 grid_dim;
    dim3 block_dim;
    int BLOCK_DIM_x = 128;
    int BLOCK_DIM_y = 4;
    int num_block_x = (M + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
    int num_block_y = (N + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
    double *dA, *dB;
    double *dC;
    cudaMalloc((void **)&dA, M * K * sizeof(double));
    cudaMalloc((void **)&dB, K * N * sizeof(double));
    cudaMalloc((void **)&dC, M * N * sizeof(double));
    cudaMemcpy(dA, hA, M * K * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, K * N * sizeof(double), cudaMemcpyHostToDevice);
    //-------------------------------------------
    double st, ela;
    double *wmmaC;
    wmmaC = (double *)malloc(M * N * sizeof(double));
    st = get_walltime();
    block_dim.x = BLOCK_DIM_x;
    block_dim.y = BLOCK_DIM_y;
    block_dim.z = 1;
    grid_dim.x = (M + (WMMA_M * block_dim.x / warpSize - 1)) / (WMMA_M * block_dim.x / warpSize);
    grid_dim.y = (N + WMMA_N * block_dim.y - 1) / (WMMA_N * block_dim.y);
    grid_dim.z = 1;
    col_wmma_ker<<<grid_dim, block_dim>>>(M, N, K, dA, dB, dC, alpha, beta);
    ela = get_walltime() - st;

    cudaMemcpy(wmmaC, dC, M * N * sizeof(double), cudaMemcpyDeviceToHost);
    //-------------------------------------------
    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    grid_dim.x = num_block_x;
    grid_dim.y = num_block_y;
    grid_dim.z = 1;

    block_dim.x = BLOCK_DIM_x;
    block_dim.y = BLOCK_DIM_y;
    block_dim.z = 1;
    kernelMat<double><<<grid_dim, block_dim>>>(M, N, K, dA, dB, dC); // kernel计算矩阵乘法
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must double ker_time

    cudaMemcpy(hC, dC, M * N * sizeof(double), cudaMemcpyDeviceToHost);
    //-------------------------------------------

    double err = 0;
    for (int i = 0; i < M * N; i++)
    {
        err = max_function(err, fabs(wmmaC[i] - hC[i]));
    }
    //-------------------------------------------
    double *cublasC;
    cublasC = (double *)malloc(M * N * sizeof(double));
    cublasHandle_t handle; // cublas句柄
    cublasCreate(&handle); // 初始化句柄
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
    cudaMemcpy(cublasC, dC, M * N * sizeof(double), cudaMemcpyDeviceToHost);
    double err_dgemm = 0;
    for (int i = 0; i < M * N; i++)
    {
        // err = max_function(err, fabs(wmmaC[i]));
        err_dgemm = max_function(err_dgemm, fabs(cublasC[i] - hC[i]));
    }

    //----------------------
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(wmmaC);
    free(cublasC);

    printf("kernel time:%.4f, col wmma error:%.4e, use time:%.4f, Dgemm error:%.4e\n", ker_time / 1000., err, ela, err_dgemm);
}

int main()
{
    int N = 16384;
    int K = 1024;
    int M = 2048;
    double alpha = 1.0f;
    double beta = 0.0f;
    double *hA, *hB;
    double *hC;
    hA = (double *)malloc(M * K * sizeof(double));
    hB = (double *)malloc(K * N * sizeof(double));
    hC = (double *)malloc(M * N * sizeof(double));
    for (int i = 0; i < M * K; i++)
    {
        hA[i] = (i % 10) * 1e-1;
    }
    for (int i = 0; i < K * N; i++)
    {
        hB[i] = (i % 10) * 1e-1;
    }
    Mat(M, N, K, hA, hB, hC, alpha, beta);
    free(hA);
    free(hB);
    free(hC);

    return 0;
}