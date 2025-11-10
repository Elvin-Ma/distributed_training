#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cuda_runtime.h>

// 定义半精度数据类型
using half_t = cutlass::half_t;

// 定义 GEMM 模板
using Gemm = cutlass::gemm::device::Gemm<
    half_t,
    cutlass::layout::RowMajor,
    half_t,
    cutlass::layout::RowMajor,
    half_t,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80
>;

int main() {
    const int M = 1024, N = 512, K = 256;
    half_t alpha(1.0f), beta(0.0f);

    // 创建主机端张量（HostTensor）
    cutlass::HostTensor<half_t, cutlass::layout::RowMajor> A_host({M, K});
    cutlass::HostTensor<half_t, cutlass::layout::RowMajor> B_host({K, N});
    cutlass::HostTensor<half_t, cutlass::layout::RowMajor> C_host({M, N});

    // 手动填充随机数据（替代 randomize()）
    // 步骤 1: 获取主机端数据指针
    half_t* A_ptr = A_host.host_data();
    half_t* B_ptr = B_host.host_data();
    half_t* C_ptr = C_host.host_data();

    // 步骤 2: 生成随机数（范围 [-1, 1]）
    for (int i = 0; i < M * K; ++i) {
        A_ptr[i] = half_t((rand() / float(RAND_MAX) * 2.0f) - 1.0f);
    }
    for (int i = 0; i < K * N; ++i) {
        B_ptr[i] = half_t((rand() / float(RAND_MAX) * 2.0f) - 1.0f);
    }
    for (int i = 0; i < M * N; ++i) {
        C_ptr[i] = half_t((rand() / float(RAND_MAX) * 2.0f) - 1.0f);
    }

    // 步骤 3: 将数据同步到设备端
    A_host.sync_device();
    B_host.sync_device();
    C_host.sync_device();

    // 创建 GEMM 算子实例
    Gemm gemm_op;
    cutlass::Status status;

    // 构造参数
    Gemm::Arguments args(
        {M, N, K},
        {A_host.device_data(), K},  // 使用 device_data() 获取设备指针
        {B_host.device_data(), N},
        {C_host.device_data(), N},
        {C_host.device_data(), N},
        {alpha, beta}
    );

    // 运行 GEMM
    status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM 失败! 错误代码: " << int(status) << std::endl;
        return -1;
    }

    // 将结果拷贝回主机
    C_host.sync_host();

    // 验证结果（简单检查）
    bool has_error = false;
    for (int i = 0; i < M * N; ++i) {
        if (isnan(float(C_host.host_data()[i]))) {
            has_error = true;
            break;
        }
    }

    if (has_error) {
        std::cerr << "结果包含 NaN/Inf!" << std::endl;
    } else {
        std::cout << "GEMM 成功完成!" << std::endl;
    }

    return 0;
}

