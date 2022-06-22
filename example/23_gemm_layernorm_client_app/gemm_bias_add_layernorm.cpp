#include <iomanip>
#include <vector>
#include <type_traits>
#include <iostream>

#include "config.hpp"
#include "tensor_layout.hpp"
#include "element_wise_operation.hpp"
#include "device_gemm_reduce.hpp"
#include "device_elementwise.hpp"
#include "reduction_operator.hpp"

struct DeviceMem
{
    DeviceMem() = delete;
    DeviceMem(std::size_t mem_size);
    void* GetDeviceBuffer();
    std::size_t GetBufferSize();
    void ToDevice(const void* p);
    void FromDevice(void* p);
    void SetZero();
    template <typename T>
    void SetValue(T x);
    ~DeviceMem();

    void* mpDeviceBuf;
    std::size_t mMemSize;
};

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_gemm_instance {

using DeviceGemmBiasAddMeanSquareMeanPtr = ck::tensor_operation::device::DeviceGemmReducePtr<1, 2>;

void add_device_gemm_bias_add_mean_squaremean_xdl_cshuffle_f16_f16_f16_f16_f16_f32_f32_mk_kn_mn_instances(
    std::vector<DeviceGemmBiasAddMeanSquareMeanPtr>&);

void add_device_gemm_bias_add_mean_squaremean_xdl_cshuffle_f16_f16_f16_f16_f16_f32_f32_mk_nk_mn_instances(
    std::vector<DeviceGemmBiasAddMeanSquareMeanPtr>&);

void add_device_gemm_bias_add_mean_squaremean_xdl_cshuffle_f16_f16_f16_f16_f16_f32_f32_km_kn_mn_instances(
    std::vector<DeviceGemmBiasAddMeanSquareMeanPtr>&);

void add_device_gemm_bias_add_mean_squaremean_xdl_cshuffle_f16_f16_f16_f16_f16_f32_f32_km_nk_mn_instances(
    std::vector<DeviceGemmBiasAddMeanSquareMeanPtr>&);

} // namespace device_gemm_instance

using Normalize          = ck::tensor_operation::element_wise::Normalize;
using DeviceNormalizePtr = ck::tensor_operation::device::DeviceElementwisePtr<5, 1, 2, Normalize>;

void add_device_normalize_from_mean_squaremean_f16_f32_f32_f16_f16_instances(
    std::vector<DeviceNormalizePtr>& instances);

} // namespace device
} // namespace tensor_operation
} // namespace ck

template <typename gemm_reduce,
          typename normalize,
          typename ALayout,
          typename BLayout,
          typename CLayout>
void GetDeviceOp(std::vector<gemm_reduce>& gemm_reduce_ptrs, std::vector<normalize>& normalize_ptrs)
{
    if(std::is_same<ALayout, ck::tensor_layout::gemm::RowMajor>::value &&
       std::is_same<BLayout, ck::tensor_layout::gemm::RowMajor>::value &&
       std::is_same<CLayout, ck::tensor_layout::gemm::RowMajor>::value)
    {
        ck::tensor_operation::device::device_gemm_instance::
            add_device_gemm_bias_add_mean_squaremean_xdl_cshuffle_f16_f16_f16_f16_f16_f32_f32_mk_kn_mn_instances(
                gemm_reduce_ptrs);
    }
    else if(std::is_same<ALayout, ck::tensor_layout::gemm::RowMajor>::value &&
            std::is_same<BLayout, ck::tensor_layout::gemm::ColumnMajor>::value &&
            std::is_same<CLayout, ck::tensor_layout::gemm::RowMajor>::value)
    {
        ck::tensor_operation::device::device_gemm_instance::
            add_device_gemm_bias_add_mean_squaremean_xdl_cshuffle_f16_f16_f16_f16_f16_f32_f32_mk_nk_mn_instances(
                gemm_reduce_ptrs);
    }
    else if(std::is_same<ALayout, ck::tensor_layout::gemm::ColumnMajor>::value &&
            std::is_same<BLayout, ck::tensor_layout::gemm::RowMajor>::value &&
            std::is_same<CLayout, ck::tensor_layout::gemm::RowMajor>::value)
    {
        ck::tensor_operation::device::device_gemm_instance::
            add_device_gemm_bias_add_mean_squaremean_xdl_cshuffle_f16_f16_f16_f16_f16_f32_f32_km_kn_mn_instances(
                gemm_reduce_ptrs);
    }
    else if(std::is_same<ALayout, ck::tensor_layout::gemm::ColumnMajor>::value &&
            std::is_same<BLayout, ck::tensor_layout::gemm::ColumnMajor>::value &&
            std::is_same<CLayout, ck::tensor_layout::gemm::RowMajor>::value)
    {
        ck::tensor_operation::device::device_gemm_instance::
            add_device_gemm_bias_add_mean_squaremean_xdl_cshuffle_f16_f16_f16_f16_f16_f32_f32_km_nk_mn_instances(
                gemm_reduce_ptrs);
    }

    ck::tensor_operation::device::
        add_device_normalize_from_mean_squaremean_f16_f32_f32_f16_f16_instances(normalize_ptrs);
}

template <typename gemm_reduce_op_ptr>
bool RunDeviceGemmMeanSquareMean(gemm_reduce_op_ptr& p_op,
                                 const void* p_a,
                                 const void* p_b,
                                 const void* p_bias,
                                 const void* p_d0,
                                 void* p_c,
                                 void* p_mean,
                                 void* p_square_mean,
                                 int M,
                                 int N,
                                 int K,
                                 int StrideA,
                                 int StrideB,
                                 int StrideC,
                                 int StrideD0,
                                 bool time_kernel)
{
    using PassThrough          = ck::tensor_operation::element_wise::PassThrough;
    using UnaryDivElementOp    = ck::tensor_operation::element_wise::UnaryDivide;
    using UnarySquareElementOp = ck::tensor_operation::element_wise::UnarySquare;

    auto passOp   = PassThrough{};
    auto squareOp = UnarySquareElementOp{};
    auto divOp    = UnaryDivElementOp{N};

    auto argument_ptr =
        p_op->MakeArgumentPointer(p_a,
                                  p_b,
                                  p_bias,
                                  {p_d0},
                                  p_c,
                                  {p_mean, p_square_mean},
                                  M,
                                  N,
                                  K,
                                  StrideA,
                                  StrideB,
                                  StrideC,
                                  {StrideD0},
                                  {&passOp, &passOp, &passOp}, // functor for a, b, c
                                  {&passOp},                   // functor for d0
                                  {&passOp, &squareOp},        // functor for inputs of reduction
                                  {&divOp, &divOp});           // functor for outputs of reduction

    if(p_op->IsSupportedArgument(argument_ptr.get()))
    {
        auto invoker_ptr = p_op->MakeInvokerPointer();

        // If we evaluate running time of gemm_reduce. The output may wrong.
        // Because we need to initialize the reduction tensor before runing the kernel.
        // However we run kernel many times for time_kernel = trie without reinitialize the out
        // of reduction tensor.
        float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

        if(time_kernel)
            std::cout << "Gemm + reduce Perf: " << std::setw(10) << ave_time << " ms" << std::endl;

        return true;
    }

    return false;
}

template <typename normalize_op_ptr>
bool RunDeviceNormalize2D(normalize_op_ptr& p_op,
                          const void* p_x,
                          const void* p_mean,
                          const void* p_square_mean,
                          const void* p_gamma,
                          const void* p_beta,
                          void* p_y,
                          int M,
                          int N,
                          int StrideX,
                          bool time_kernel)
{
    std::array<const void*, 5> input = {p_x, p_mean, p_square_mean, p_gamma, p_beta};
    std::array<void*, 1> output      = {p_y};
    auto normalize_functor           = ck::tensor_operation::device::Normalize{};

    auto argument_ptr = p_op->MakeArgumentPointer(input,
                                                  output,
                                                  {M, N},
                                                  {{StrideX, 1}, {1, 0}, {1, 0}, {0, 1}, {0, 1}},
                                                  {{StrideX, 1}},
                                                  ck::tensor_operation::device::Normalize{});

    if(p_op->IsSupportedArgument(argument_ptr.get()))
    {
        auto invoker_ptr = p_op->MakeInvokerPointer();
        float ave_time   = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

        if(time_kernel)
            std::cout << "Normalize Perf: " << std::setw(10) << ave_time << " ms" << std::endl;

        return true;
    }

    return false;
}

int main()
{
    ck::index_t M = 1024;
    ck::index_t N = 1024;
    ck::index_t K = 1024;

    ck::index_t StrideA  = 1024;
    ck::index_t StrideB  = 1024;
    ck::index_t StrideC  = 1024;
    ck::index_t StrideD0 = 1024;

    using F16 = ck::half_t;
    using F32 = float;

    using ADataType            = F16;
    using BDataType            = F16;
    using BiasDataType         = F32;
    using CDataType            = F16;
    using D0DataType           = F16;
    using ReduceDataType       = F32;
    using GammaDataType        = F16;
    using BetaDataType         = F16;
    using LayerNormOutDataType = F16;

    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    std::vector<
        ck::tensor_operation::device::device_gemm_instance::DeviceGemmBiasAddMeanSquareMeanPtr>
        gemm_reduce_ptrs;

    std::vector<ck::tensor_operation::device::DeviceNormalizePtr> normalize_ptrs;

    GetDeviceOp<
        ck::tensor_operation::device::device_gemm_instance::DeviceGemmBiasAddMeanSquareMeanPtr,
        ck::tensor_operation::device::DeviceNormalizePtr,
        ALayout,
        BLayout,
        CLayout>(gemm_reduce_ptrs, normalize_ptrs);

    std::cout << "found " << gemm_reduce_ptrs.size()
              << " gemm_reduceMean_reduceSquareMean instances" << std::endl;

    std::cout << "found " << normalize_ptrs.size() << " normalize instances" << std::endl;

    auto f_matrix_space_size =
        [](std::size_t nRow, std::size_t nCol, std::size_t stride, auto layout) {
            using Layout = decltype(layout);

            if(std::is_same<Layout, ck::tensor_layout::gemm::RowMajor>::value)
            {
                return (nRow - 1) * stride + nCol;
            }
            else
            {
                return (nCol - 1) * stride + nRow;
            }
        };

    DeviceMem a_device_buf(sizeof(ADataType) * f_matrix_space_size(M, K, StrideA, ALayout{}));
    DeviceMem b_device_buf(sizeof(BDataType) * f_matrix_space_size(K, N, StrideB, BLayout{}));
    DeviceMem bias_device_buf(sizeof(BiasDataType) * N);
    DeviceMem c_device_buf(sizeof(CDataType) * f_matrix_space_size(M, N, StrideC, CLayout{}));
    DeviceMem d0_device_buf(sizeof(D0DataType) * f_matrix_space_size(M, N, StrideD0, CLayout{}));
    DeviceMem reduceMean_device_buf(sizeof(ReduceDataType) * M);
    DeviceMem reduceMeanSquare_device_buf(sizeof(ReduceDataType) * M);
    DeviceMem gamma_device_buf(sizeof(GammaDataType) * N);
    DeviceMem beta_device_buf(sizeof(BetaDataType) * N);
    DeviceMem layerNorm_device_buf(sizeof(LayerNormOutDataType) * M * N);

    bool b_time_kernel           = true;
    bool b_only_run_first_kernel = true;

    // layernorm => (1) + (2)
    // (1). c = gemm(a, b), reduce_mean(c), reduce_square_mean(c)
    // (2). normalize(c, mean, square_mean, gamma, beta)
    for(auto& gemm_reduce_ptr : gemm_reduce_ptrs)
    {
        // run first available kernel
        if(RunDeviceGemmMeanSquareMean(gemm_reduce_ptr,
                                       a_device_buf.GetDeviceBuffer(),
                                       b_device_buf.GetDeviceBuffer(),
                                       bias_device_buf.GetDeviceBuffer(),
                                       d0_device_buf.GetDeviceBuffer(),
                                       c_device_buf.GetDeviceBuffer(),
                                       reduceMean_device_buf.GetDeviceBuffer(),
                                       reduceMeanSquare_device_buf.GetDeviceBuffer(),
                                       M,
                                       N,
                                       K,
                                       StrideA,
                                       StrideB,
                                       StrideC,
                                       StrideD0,
                                       b_time_kernel))
        {
            if(b_only_run_first_kernel)
                break;
        }
        else
        {
            std::cout << gemm_reduce_ptr->GetTypeString() << " does not support this problem"
                      << std::endl;
        }
    }

    for(auto& normalize_ptr : normalize_ptrs)
    {
        if(RunDeviceNormalize2D(normalize_ptr,
                                c_device_buf.GetDeviceBuffer(),
                                reduceMean_device_buf.GetDeviceBuffer(),
                                reduceMeanSquare_device_buf.GetDeviceBuffer(),
                                gamma_device_buf.GetDeviceBuffer(),
                                beta_device_buf.GetDeviceBuffer(),
                                layerNorm_device_buf.GetDeviceBuffer(),
                                M,
                                N,
                                StrideC,
                                b_time_kernel))
        {
            if(b_only_run_first_kernel)
                break;
        }
        else
        {
            std::cout << normalize_ptr->GetTypeString() << " does not support this problem"
                      << std::endl;
        }
    }
}