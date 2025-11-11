// nvcc -arch=sm_80 wmma_fp16.cu -o wmma_fp16 -std=c++11
#include <cuda_runtime.h>
#include <mma.h>
#include <iostream>
#include <vector>
#include <cmath>

using namespace nvcuda;

// ==================== 配置参数 ====================
#define TEST_SMALL_MATRIX 1  // 1=小矩阵（16x16x16，快速验证）；0=大矩阵（1024x1024x1024）

#if TEST_SMALL_MATRIX
const int M = 16;        // A/C 的行数（=WMMA_M，无边界，确保对齐）
const int K = 16;        // A的列数/B的行数（=WMMA_K，无边界，确保对齐）
const int N = 16;        // B/C 的列数（=WMMA_N，无边界，确保对齐）
#else
const int M = 1024;      // 大矩阵尺寸（可被16整除，满足对齐要求）
const int K = 1024;
const int N = 1024;
#endif

const int WMMA_M = 16;   // WMMA tile 行维度（固定16，warp级接口要求）
const int WMMA_N = 16;   // WMMA tile 列维度（固定16）
const int WMMA_K = 16;   // WMMA tile 公共维度（固定16）

// 线程块尺寸：必须是 32线程（1 warp）或 64线程（2 warp），因WMMA是warp级接口
// 16x16=256线程（8 warp），每个warp处理1个tile（16x16），效率最高
const int BLOCK_DIM_X = 16;
const int BLOCK_DIM_Y = 16;

// ==================== 设备端 WMMA 核函数（彻底修复对齐问题）====================
__global__ void wmma_matmul_kernel(
    const half* __restrict__ A,  // 输入矩阵A (M×K), FP16, 列优先（元素级）
    const half* __restrict__ B,  // 输入矩阵B (K×N), FP16, 列优先（元素级）
    float* __restrict__ C        // 输出矩阵C (M×N), FP32, 列优先（元素级）
) {
    // 1. 计算当前线程块负责的 TILE 基地址（元素级，无threadIdx偏移！）
    // tile_row：当前tile在A/C中的起始行（元素级，必须是WMMA_M的倍数，确保对齐）
    const int tile_row = blockIdx.y * WMMA_M;
    // tile_col：当前tile在B/C中的起始列（元素级，必须是WMMA_N的倍数，确保对齐）
    const int tile_col = blockIdx.x * WMMA_N;

    // 2. 定义WMMA fragment（严格匹配列优先，对齐由fragment自动保证）
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // 3. 初始化累积器为0
    wmma::fill_fragment(c_frag, 0.0f);

    // 4. 循环拆分K维度，计算tile乘法-累积（每个迭代处理一个K方向的tile）
    for (int k = 0; k < K; k += WMMA_K) {
        // 4.1 加载A的tile（核心修正：地址仅含tile基地址，无threadIdx偏移）
        // A的tile基地址（元素级）= 行偏移（tile_row） + 列偏移（k） × A的总行数（M）
        // 字节地址 = 基地址（字节） + 元素偏移 × sizeof(half) → 自动满足32字节对齐
        wmma::load_matrix_sync(
            a_frag,                                  // fragment引用（warp级，自动分配元素）
            A + tile_row + k * M,                    // 正确tile基地址（元素级，无threadIdx）
            M                                       // ld=A的行数（元素级，列优先核心参数）
        );

        // 4.2 加载B的tile（核心修正：地址仅含tile基地址，无threadIdx偏移）
        // B的tile基地址（元素级）= 行偏移（k） + 列偏移（tile_col） × B的总行数（K）
        wmma::load_matrix_sync(
            b_frag,                                  // fragment引用
            B + k + tile_col * K,                    // 正确tile基地址（元素级，无threadIdx）
            K                                       // ld=B的行数（元素级，列优先核心参数）
        );

        // 4.3 核心运算：C_frag += A_frag × B_frag（warp级并行，Tensor Core加速）
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 4.4 存储结果到C（核心修正：地址仅含tile基地址，无threadIdx偏移）
    // C的tile基地址（元素级）= 行偏移（tile_row） + 列偏移（tile_col） × C的总行数（M）
    wmma::store_matrix_sync(
        C + tile_row + tile_col * M,                // 正确tile基地址（元素级，无threadIdx）
        c_frag,                                     // 累积器fragment
        M,                                           // ld=C的行数（元素级）
        wmma::mem_col_major                          // 输出存储格式（列优先）
    );
}

// ==================== 主机端辅助函数（无修改，确保正确）====================
#define CHECK_CUDA_ERR(err) \
    do { \
        cudaError_t e = err; \
        if (e != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(e) << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

// CPU端矩阵乘法（列优先，与GPU完全一致）
void cpu_matmul(
    const std::vector<half>& A,
    const std::vector<half>& B,
    std::vector<float>& C
) {
    for (int m = 0; m < M; ++m) {  // C的行（A的行）
        for (int n = 0; n < N; ++n) {  // C的列（B的列）
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {  // 公共维度K
                // 列优先索引：A[m + k*M]（第k列第m行），B[k + n*K]（第n列第k行）
                sum += __half2float(A[m + k * M]) * __half2float(B[k + n * K]);
            }
            C[m + n * M] = sum;  // C的列优先索引（第n列第m行）
        }
    }
}

// 验证结果（根据精度调整误差阈值）
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

// ==================== 主函数（无修改）====================
int main() {
    // 1. 检查CUDA架构（必须支持WMMA：Compute Capability ≥7.0）
    int dev_id = 0;
    CHECK_CUDA_ERR(cudaSetDevice(dev_id));
    cudaDeviceProp prop;
    CHECK_CUDA_ERR(cudaGetDeviceProperties(&prop, dev_id));
    std::cout << "========================================" << std::endl;
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    if (prop.major < 7) {
        std::cerr << "❌ Error: WMMA requires Compute Capability ≥7.0 (Volta/Turing/Ampere/Hopper)" << std::endl;
        return 1;
    }

    // 2. 初始化主机端数据（FP16）
    std::vector<half> h_A(M * K);
    std::vector<half> h_B(K * N);
    std::vector<float> h_cpu_C(M * N, 0.0f);
    std::vector<float> h_gpu_C(M * N, 0.0f);

#if TEST_SMALL_MATRIX
    // 小矩阵测试：全1初始化（预期结果全16.0f，快速验证）
    std::cout << "========================================" << std::endl;
    std::cout << "Test Mode: Small Matrix (16x16x16), All 1.0f" << std::endl;
    for (int i = 0; i < M * K; ++i) h_A[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; ++i) h_B[i] = __float2half(1.0f);
#else
    // 大矩阵测试：随机初始化（0~1）
    std::cout << "========================================" << std::endl;
    std::cout << "Test Mode: Large Matrix (" << M << "x" << K << "x" << N << "), Random" << std::endl;
    srand(time(nullptr));
    for (int i = 0; i < M * K; ++i) h_A[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    for (int i = 0; i < K * N; ++i) h_B[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
#endif

    // 3. CPU计算参考结果
    std::cout << "========================================" << std::endl;
    std::cout << "Computing CPU reference result..." << std::endl;
    cpu_matmul(h_A, h_B, h_cpu_C);
#if TEST_SMALL_MATRIX
    std::cout << "CPU Result (all=16.0f): " << h_cpu_C[0] << ", " << h_cpu_C[10] << std::endl;
#endif

    // 4. 设备端内存分配与数据拷贝（cudaMalloc确保基地址对齐）
    half *d_A, *d_B;
    float *d_C;
    CHECK_CUDA_ERR(cudaMalloc(&d_A, M * K * sizeof(half)));  // 基地址默认256字节对齐
    CHECK_CUDA_ERR(cudaMalloc(&d_B, K * N * sizeof(half)));
    CHECK_CUDA_ERR(cudaMalloc(&d_C, M * N * sizeof(float)));

    CHECK_CUDA_ERR(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));

    // 5. 核函数配置（正确，覆盖所有tile）
    dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y);  // 16x16线程块（256线程=8 warp，效率最高）
    dim3 grid_dim(N / WMMA_N, M / WMMA_M);     // 网格：(N/16)×(M/16)（每个线程块处理1个tile）
    std::cout << "========================================" << std::endl;
    std::cout << "Launching WMMA Kernel..." << std::endl;
    std::cout << "Grid Dim: (" << grid_dim.x << ", " << grid_dim.y << ")" << std::endl;
    std::cout << "Block Dim: (" << block_dim.x << ", " << block_dim.y << ")" << std::endl;

    // 计时
    cudaEvent_t start, stop;
    CHECK_CUDA_ERR(cudaEventCreate(&start));
    CHECK_CUDA_ERR(cudaEventCreate(&stop));
    CHECK_CUDA_ERR(cudaEventRecord(start));

    // 启动核函数
    wmma_matmul_kernel<<<grid_dim, block_dim>>>(d_A, d_B, d_C);
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

    // 7. 计算算力（GFLOPS）
    float flops = 2.0f * M * N * K;  // FP16矩阵乘法：2*M*N*K FLOPs（每个乘加对算2次操作）
    float gflops = flops / (elapsed_ms * 1e6f);  // GFLOPS = 1e9 FLOPs / 1e3 ms = 1e6 FLOPs/ms
    std::cout << "========================================" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
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