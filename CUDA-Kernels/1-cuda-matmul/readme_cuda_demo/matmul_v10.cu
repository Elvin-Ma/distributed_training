#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>

// #include "cuda_matmul.h"

#include <cublas_v2.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

typedef float scalar_t;

// 文章作者实现的V10版本：基于CUDA的SGEMM（单精度矩阵乘法）优化实现
// 核心公式：C = α·A·B + β·C（A:M×K, B:K×N, C:M×N）
// 优化策略：Warp Tiling分块 + 共享内存缓存 + 寄存器缓存 + float4批量访存
namespace V10_Author {
    // ===================== 基础常量定义 =====================
    const int WARPSIZE = 32;          // CUDA线程束（Warp）固定大小（32线程/束）
    const uint NUM_THREADS = 128;     // 每个线程块的总线程数（128=4个Warp）

    // ===================== SGEMM核函数 =====================
    // __launch_bounds__：编译器优化提示，指定线程块最大线程数，优化寄存器分配
    __global__ void __launch_bounds__(NUM_THREADS)
    sgemmWarptiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
        // -------------------- 分块大小硬编码（核心参数） --------------------
        // BM/BN/BK：线程块级共享内存缓存块大小（处理64×128的C块，每次加载64×8的A、8×128的B）
        // WM/WN：单个Warp负责计算的子块大小；WNITER：N维度子Warp分块步数
        // TM/TN：单个线程负责的最小计算子块大小（4×4）
        const uint BN = 128, BM = 64, BK = 8;
        const uint WN = 64, WM = 32, WNITER = 2;
        const uint TN = 4, TM = 4;

        // -------------------- 线程/束定位计算 --------------------
        // 1. 线程块在Grid中的位置（对应C矩阵块的行列索引）
        const uint cRow = blockIdx.y, cCol = blockIdx.x;
        // 2. 当前线程所属Warp在线程块内的行列索引（线程块内2×2的Warp布局）
        const uint warpIdx = threadIdx.x / WARPSIZE;
        const uint warpCol = warpIdx % (BN / WN), warpRow = warpIdx / (BN / WN);
        // 3. 子Warp分块大小（编译期常量，确保数值固定）
        constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER); // M维度子Warp步数
        constexpr uint WSUBM = WM / WMITER, WSUBN = WN / WNITER;           // 子Warp分块尺寸
        // 4. 当前线程在所属Warp内的行列索引（Warp内8×4的线程布局）
        const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
        const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
        const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

        // -------------------- 内存空间分配 --------------------
        __shared__ float As[BM * BK];  // 共享内存：缓存A矩阵的BM×BK块（转置存储以适配合并访问）
        __shared__ float Bs[BK * BN];  // 共享内存：缓存B矩阵的BK×BN块
        // 线程寄存器：缓存中间计算结果（避免重复访存，提升计算效率）
        float threadResults[WMITER * TM * WNITER * TN] = {0.0}; // 最终结果缓存
        float regM[WMITER * TM] = {0.0};                       // A数据寄存器缓存
        float regN[WNITER * TN] = {0.0};                       // B数据寄存器缓存

        // -------------------- 矩阵指针偏移 --------------------
        // 定位当前线程块处理的A/B/C矩阵块起始位置（行优先存储）
        A += cRow * BM * K;                          // A块起始：当前线程块行 × 行大小
        B += cCol * BN;                              // B块起始：当前线程块列
        C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN; // Warp负责的C子块起始

        // -------------------- 共享内存加载索引计算 --------------------
        // 基于float4（128bit）批量加载4个float，最大化内存带宽
        const uint innerRowA = threadIdx.x / (BK / 4), innerColA = threadIdx.x % (BK / 4);
        constexpr uint rowStrideA = (NUM_THREADS * 4) / BK; // A加载行步长
        const uint innerRowB = threadIdx.x / (BN / 4), innerColB = threadIdx.x % (BN / 4);
        constexpr uint rowStrideB = NUM_THREADS / (BN / 4); // B加载行步长

        // -------------------- 核心循环：K维度分块计算 --------------------
        // 遍历K维度所有块，逐块计算点积（矩阵乘法核心是K维度的乘积累加）
        for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
            // 1. 加载A矩阵到共享内存（转置存储，适配GPU合并访问规则）
            for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
                float4 tmp = reinterpret_cast<float4 *>(&A[(innerRowA + offset) * K + innerColA * 4])[0];
                As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
                As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
                As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
                As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
            }
            // 2. 加载B矩阵到共享内存（float4批量加载，无需转置）
            for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
                reinterpret_cast<float4 *>(&Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
                    reinterpret_cast<float4 *>(&B[(innerRowB + offset) * N + innerColB * 4])[0];
            }
            __syncthreads(); // 同步：确保所有线程完成共享内存加载

            // 3. 遍历当前BK块，寄存器内完成乘积累加
            for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
                // 加载共享内存数据到寄存器（最快存储层级）
                for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                    for (uint i = 0; i < TM; ++i) {
                        regM[wSubRowIdx * TM + i] = As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + i];
                    }
                }
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    for (uint i = 0; i < TN; ++i) {
                        regN[wSubColIdx * TN + i] = Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + i];
                    }
                }
                // Warp级矩阵乘法：寄存器内乘积累加
                for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                                threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) + (wSubColIdx * TN) + resIdxN] +=
                                    regM[wSubRowIdx * TM + resIdxM] * regN[wSubColIdx * TN + resIdxN];
                            }
                        }
                    }
                }
            }
            // 更新A/B指针：处理下一个K块
            A += BK;     // A右移BK列
            B += BK * N; // B下移BK行
            __syncthreads();
        }

        // -------------------- 结果写回：合并α/β系数，写回C矩阵 --------------------
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN; // 当前子Warp的C块
                for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
                    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                        // float4批量加载原始C值 → 计算SGEMM最终值 → 批量写回
                        float4 tmp = reinterpret_cast<float4 *>(&C_interim[(threadRowInWarp * TM + resIdxM) * N + threadColInWarp * TN + resIdxN])[0];
                        const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) + wSubColIdx * TN + resIdxN;
                        tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
                        tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
                        tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
                        tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
                        reinterpret_cast<float4 *>(&C_interim[(threadRowInWarp * TM + resIdxM) * N + threadColInWarp * TN + resIdxN])[0] = tmp;
                    }
                }
            }
        }
    }
}

void MatmulCoreV10_Author(const scalar_t *a, const scalar_t *b, scalar_t *out, uint32_t M, uint32_t N, uint32_t P) {
    dim3 grid(std::ceil(static_cast<double>(P) / (128)), std::ceil(static_cast<double>(M) / 64));
    dim3 block(128);
    V10_Author::sgemmWarptiling<<<grid, block>>>(M, P, N, 1, (float*)a, (float*)b, 0, out);
}