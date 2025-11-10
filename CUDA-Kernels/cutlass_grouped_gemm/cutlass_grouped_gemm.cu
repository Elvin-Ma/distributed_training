#include <iostream>
#include <vector>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

// 1. 定义核心配置参数
using Element = cutlass::half_t;           // 使用FP16数据类型
using Layout = cutlass::layout::RowMajor;  // 行优先布局
constexpr int kStages = 3;                // 流水线级数
constexpr int kAlignmentA = 8;            // 128-bit对齐 (128/16=8 elements)
constexpr int kAlignmentB = 8;            // 与kAlignmentA相同

// 2. 正确定义MMA（显式包含对齐参数）
using Mma = cutlass::gemm::threadblock::DefaultMma<
    Element, Layout, kAlignmentA,          // A矩阵：元素类型、布局、对齐
    Element, Layout, kAlignmentB,          // B矩阵：元素类型、布局、对齐
    Element, Layout,                       // C矩阵配置
    cutlass::arch::Sm80,                   // 架构标签
    cutlass::gemm::GemmShape<128, 128, 32>,// 线程块形状
    cutlass::gemm::GemmShape<64, 64, 32>,  // Warp形状
    cutlass::gemm::GemmShape<16, 8, 8>,    // 指令形状
    cutlass::arch::OpClassTensorOp,        // 运算类
    kStages,                              // 流水线级数
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle, // Swizzle
    cutlass::arch::OpMultiplyAdd          // 乘加操作符
>;

// 3. 定义Epilogue（保持与之前一致）
using Epilogue = cutlass::epilogue::threadblock::DefaultEpilogue<
    cutlass::gemm::GemmShape<128, 128, 32>,
    typename Mma::Operator,
    1,
    Element,
    cutlass::epilogue::thread::LinearCombination<Element, 8, float, float>
>;

// 4. 定义线程块调度策略
using ThreadblockSwizzle =
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

// 5. 实例化GemmGrouped模板（严格匹配头文件定义）
using GemmGroupedKernel = cutlass::gemm::kernel::GemmGrouped<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    cutlass::gemm::GroupScheduleMode::kDeviceOnly, // 设备端调度
    false                                          // 非转置模式
>;

// 6. 定义设备级GEMM操作
using GroupedGemm = cutlass::gemm::device::GemmGrouped<GemmGroupedKernel>;

// CUDA错误检查宏
#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(code)
                << " at " << file << ":" << line << std::endl;
        exit(code);
    }
}

int main() {
    const int group_count = 3;
    std::vector<GroupedGemm::Arguments> args_list;

    // 初始化参数（显式处理对齐）
    for(int i = 0; i < group_count; ++i) {
        int m = 128 * (i + 1);
        int n = 256;
        int k = 64;

        // 确保维度满足对齐要求
        assert(k % kAlignmentA == 0 && "K维度需要满足A矩阵对齐要求");
        assert(k % kAlignmentB == 0 && "K维度需要满足B矩阵对齐要求");

        args_list.emplace_back(
            cutlass::gemm::GemmCoord{m, n, k},
            nullptr, // ptr_A（需对齐分配）
            nullptr, // ptr_B
            nullptr, // ptr_C
            nullptr, // ptr_D
            {1.0f, 0.0f}, // alpha, beta
            k,      // lda
            n,      // ldb
            n,      // ldc
            n,      // ldd
            1       // batch_count
        );
    }

    // 分配对齐的设备内存
    std::vector<Element*> d_A(group_count), d_B(group_count), d_D(group_count);
    for(int i = 0; i < group_count; ++i) {
        auto& args = args_list[i];

        // A矩阵内存分配（对齐）
        size_t size_A = args.problem_size.m() * args.problem_size.k();
        size_A = (size_A + kAlignmentA - 1) / kAlignmentA * kAlignmentA; // 对齐调整
        cudaErrorCheck(cudaMalloc(&d_A[i], size_A * sizeof(Element)));

        // B矩阵内存分配（对齐）
        size_t size_B = args.problem_size.k() * args.problem_size.n();
        size_B = (size_B + kAlignmentB - 1) / kAlignmentB * kAlignmentB;
        cudaErrorCheck(cudaMalloc(&d_B[i], size_B * sizeof(Element)));

        // D矩阵分配（常规）
        size_t size_D = args.problem_size.m() * args.problem_size.n();
        cudaErrorCheck(cudaMalloc(&d_D[i], size_D * sizeof(Element)));

        args.ptr_A = d_A[i];
        args.ptr_B = d_B[i];
        args.ptr_D = d_D[i];
    }

    // 初始化并执行GEMM（与之前相同）
    GroupedGemm gemm_op;
    size_t workspace_size = GroupedGemm::get_workspace_size(args_list);
    void* workspace;
    cudaErrorCheck(cudaMalloc(&workspace, workspace_size));

    cutlass::Status status = gemm_op.initialize(args_list, workspace);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "初始化失败: "
                << cutlass::cutlassGetStatusString(status) << std::endl;
        return -1;
    }

    status = gemm_op.run();
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "执行失败: "
                << cutlass::cutlassGetStatusString(status) << std::endl;
        return -1;
    }

    // 清理资源
    for(auto ptr : d_A) cudaFree(ptr);
    for(auto ptr : d_B) cudaFree(ptr);
    for(auto ptr : d_D) cudaFree(ptr);
    cudaFree(workspace);

    std::cout << "Grouped GEMM成功执行!" << std::endl;
    return 0;
}