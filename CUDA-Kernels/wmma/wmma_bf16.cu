// nvcc -arch=sm_80 wmma_bf16.cu -o wmma_bf16 -std=c++11
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_bf16.h>  // BF16 类型支持（必需）
#include <iostream>
#include <vector>
#include <cmath>

using namespace nvcuda;

// ==================== 配置参数 ====================
#define TEST_SMALL_MATRIX 1  // 1=小矩阵（16x16x16，快速验证）；0=大矩阵（1024x1024x1024）

#if TEST_SMALL_MATRIX
const int M = 16;        // A/C 的行数（=WMMA_M，无边界）
const int K = 16;        // A的列数/B的行数（=WMMA_K，无边界）
const int N = 16;        // B/C 的列数（=WMMA_N，无边界）
#else
const int M = 1024;      // 大矩阵尺寸（可被16整除，满足对齐）
const int K = 1024;
const int N = 1024;
#endif

const int WMMA_M = 16;   // WMMA tile 行维度（固定16）
const int WMMA_N = 16;   // WMMA tile 列维度（固定16）
const int WMMA_K = 16;   // WMMA tile 公共维度（固定16）

// 线程块尺寸：16x16=256线程（8 warp），适配WMMA warp级并行
const int BLOCK_DIM_X = 16;
const int BLOCK_DIM_Y = 16;

// ==================== 设备端 WMMA BF16 核函数 ====================
__global__ void wmma_bf16_matmul_kernel(
    const __nv_bfloat16* __restrict__ A,  // 输入矩阵A (M×K), BF16, 列优先
    const __nv_bfloat16* __restrict__ B,  // 输入矩阵B (K×N), BF16, 列优先
    float* __restrict__ C                 // 输出矩阵C (M×N), FP32, 列优先（累积精度）
) {
    // 计算当前线程块负责的 tile 基地址（元素级，无threadIdx偏移，保证对齐）
    const int tile_row = blockIdx.y * WMMA_M;  // tile在A/C中的起始行
    const int tile_col = blockIdx.x * WMMA_N;  // tile在B/C中的起始列

    // 定义WMMA Fragment（数据类型改为__nv_bfloat16，存储格式保持col_major）
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;  // 累积器仍用FP32

    // 初始化累积器为0
    wmma::fill_fragment(c_frag, 0.0f);

    // 循环拆分K维度，计算tile乘法-累积
    for (int k = 0; k < K; k += WMMA_K) {
        // 加载A的BF16 tile（地址逻辑与FP16一致，BF16也是2字节）
        wmma::load_matrix_sync(
            a_frag,
            A + tile_row + k * M,  // tile基地址（元素级）
            M                      // ld=A的行数（列优先核心参数）
        );

        // 加载B的BF16 tile
        wmma::load_matrix_sync(
            b_frag,
            B + k + tile_col * K,  // tile基地址（元素级）
            K                      // ld=B的行数（列优先核心参数）
        );

        // 核心运算：C_frag += A_frag × B_frag（BF16 Tensor Core加速）
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 存储FP32结果到全局内存C
    wmma::store_matrix_sync(
        C + tile_row + tile_col * M,  // tile基地址（元素级）
        c_frag,
        M,                            // ld=C的行数
        wmma::mem_col_major            // 输出列优先存储
    );
}

// ==================== 主机端辅助函数 ====================
// 检查CUDA错误
#define CHECK_CUDA_ERR(err) \
    do { \
        cudaError_t e = err; \
        if (e != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(e) << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

// CPU端BF16矩阵乘法（参考结果计算）
void cpu_bf16_matmul(
    const std::vector<__nv_bfloat16>& A,
    const std::vector<__nv_bfloat16>& B,
    std::vector<float>& C
) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                // BF16转float计算（避免CPU端BF16运算精度损失）
                sum += __bfloat162float(A[m + k * M]) * __bfloat162float(B[k + n * K]);
            }
            C[m + n * M] = sum;
        }
    }
}

// 验证结果（BF16精度略高于FP16，误差阈值可保持不变）
bool verify_result(
    const std::vector<float>& cpu_C,
    const std::vector<float>& gpu_C,
    float eps = TEST_SMALL_MATRIX ? 1e-6f : 1e-3f
) {
    for (int i = 0; i < M * N; ++i) {
        if (fabs(cpu_C[i] - gpu_C[i]) > eps) {
            std::cerr << "Result mismatch at index " << i
                      << ": CPU=" << cpu_C[i] << ", GPU=" << gpu_C[i] << std::endl;
            return false;
        }
    }
    std::cout << "\n✅ Result verification passed! All elements match within " << eps << std::endl;
    return true;
}

// ==================== 主函数 ====================
int main() {
    // 1. 检查CUDA架构（BF16需Compute Capability ≥8.0）
    int dev_id = 0;
    CHECK_CUDA_ERR(cudaSetDevice(dev_id));
    cudaDeviceProp prop;
    CHECK_CUDA_ERR(cudaGetDeviceProperties(&prop, dev_id));
    std::cout << "========================================" << std::endl;
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;

    if (prop.major < 8) {
        std::cerr << "❌ Error: BF16 WMMA requires Compute Capability ≥8.0 (Ampere/Hopper)" << std::endl;
        std::cerr << "Supported GPUs: A100, RTX 30/40 series, H100, etc." << std::endl;
        return 1;
    }

    // 2. 主机端数据初始化（BF16类型）
    std::vector<__nv_bfloat16> h_A(M * K);
    std::vector<__nv_bfloat16> h_B(K * N);
    std::vector<float> h_cpu_C(M * N, 0.0f);
    std::vector<float> h_gpu_C(M * N, 0.0f);

#if TEST_SMALL_MATRIX
    // 小矩阵测试：全1初始化（预期结果全16.0f）
    std::cout << "========================================" << std::endl;
    std::cout << "Test Mode: Small Matrix (16x16x16), All 1.0f (BF16)" << std::endl;
    for (int i = 0; i < M * K; ++i) h_A[i] = __float2bfloat16(1.0f);  // float→BF16
    for (int i = 0; i < K * N; ++i) h_B[i] = __float2bfloat16(1.0f);
#else
    // 大矩阵测试：随机初始化（0~1，float转BF16）
    std::cout << "========================================" << std::endl;
    std::cout << "Test Mode: Large Matrix (" << M << "x" << K << "x" << N << "), Random (BF16)" << std::endl;
    srand(time(nullptr));
    for (int i = 0; i < M * K; ++i) h_A[i] = __float2bfloat16(static_cast<float>(rand()) / RAND_MAX);
    for (int i = 0; i < K * N; ++i) h_B[i] = __float2bfloat16(static_cast<float>(rand()) / RAND_MAX);
#endif

    // 3. CPU计算参考结果（BF16转float计算）
    std::cout << "========================================" << std::endl;
    std::cout << "Computing CPU reference result (BF16→float)..." << std::endl;
    cpu_bf16_matmul(h_A, h_B, h_cpu_C);
#if TEST_SMALL_MATRIX
    std::cout << "CPU Result (all=16.0f): " << h_cpu_C[0] << ", " << h_cpu_C[10] << std::endl;
#endif

    // 4. 设备端内存分配与数据拷贝（BF16类型）
    __nv_bfloat16 *d_A, *d_B;
    float *d_C;
    CHECK_CUDA_ERR(cudaMalloc(&d_A, M * K * sizeof(__nv_bfloat16)));  // BF16=2字节
    CHECK_CUDA_ERR(cudaMalloc(&d_B, K * N * sizeof(__nv_bfloat16)));
    CHECK_CUDA_ERR(cudaMalloc(&d_C, M * N * sizeof(float)));

    CHECK_CUDA_ERR(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    // 5. 核函数配置（与FP16版本一致，无需修改）
    dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y);  // 16x16线程块（256线程）
    dim3 grid_dim(N / WMMA_N, M / WMMA_M);     // 网格：(N/16)×(M/16)
    std::cout << "========================================" << std::endl;
    std::cout << "Launching WMMA BF16 Kernel..." << std::endl;
    std::cout << "Grid Dim: (" << grid_dim.x << ", " << grid_dim.y << ")" << std::endl;
    std::cout << "Block Dim: (" << block_dim.x << ", " << block_dim.y << ")" << std::endl;

    // 计时
    cudaEvent_t start, stop;
    CHECK_CUDA_ERR(cudaEventCreate(&start));
    CHECK_CUDA_ERR(cudaEventCreate(&stop));
    CHECK_CUDA_ERR(cudaEventRecord(start));

    // 启动BF16核函数
    wmma_bf16_matmul_kernel<<<grid_dim, block_dim>>>(d_A, d_B, d_C);
    CHECK_CUDA_ERR(cudaGetLastError());  // 检查核函数启动错误
    CHECK_CUDA_ERR(cudaDeviceSynchronize());  // 等待核函数完成

    // 停止计时
    CHECK_CUDA_ERR(cudaEventRecord(stop));
    CHECK_CUDA_ERR(cudaEventSynchronize(stop));
    float elapsed_ms;
    CHECK_CUDA_ERR(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::cout << "Kernel Execution Time: " << elapsed_ms << " ms" << std::endl;

    // 6. 结果拷贝与验证
    CHECK_CUDA_ERR(cudaMemcpy(h_gpu_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
#if TEST_SMALL_MATRIX
    std::cout << "GPU Result (all=16.0f): " << h_gpu_C[0] << ", " << h_gpu_C[10] << std::endl;
#endif
    verify_result(h_cpu_C, h_gpu_C);

    // 7. 计算算力（BF16与FP16算力一致，因都是2字节/元素，Tensor Core吞吐量相同）
    float flops = 2.0f * M * N * K;  // 2*M*N*K FLOPs（乘加对）
    float gflops = flops / (elapsed_ms * 1e6f);  // GFLOPS = 1e9 FLOPs / 1e3 ms
    std::cout << "========================================" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS (BF16)" << std::endl;
    std::cout << "========================================" << std::endl;

    // 8. 资源释放
    CHECK_CUDA_ERR(cudaFree(d_A));
    CHECK_CUDA_ERR(cudaFree(d_B));
    CHECK_CUDA_ERR(cudaFree(d_C));
    CHECK_CUDA_ERR(cudaEventDestroy(start));
    CHECK_CUDA_ERR(cudaEventDestroy(stop));
    CHECK_CUDA_ERR(cudaDeviceReset());

    return 0;
}