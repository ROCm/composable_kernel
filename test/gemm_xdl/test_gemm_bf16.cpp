#include <algorithm>
#include <cstdlib>
#include <half.hpp>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

#include "gemm_util.hpp"
#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "host_gemm.hpp"
#include "device_tensor.hpp"
#include "device_gemm_xdl.hpp"
#include "device_gemm_xdl_c_shuffle.hpp"
#include "element_wise_operation.hpp"
#include "reference_gemm.hpp"
#include "gemm_specialization.hpp"
#include "test_util.hpp"

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using DeviceGemmPtr_ =
    ck::tensor_operation::device::DeviceGemmPtr<ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough>;

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_gemm_instance {
void add_device_gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_nk_mn_instances(std::vector<DeviceGemmPtr_>&);
}
} // namespace device
} // namespace tensor_operation
} // namespace ck

namespace {

using BF16 = ck::bhalf_t;

using ADataType   = BF16;
using BDataType   = BF16;
using CDataType   = BF16;
using AccDataType = float;

using ALayout = ck::tensor_layout::gemm::RowMajor;
using BLayout = ck::tensor_layout::gemm::ColumnMajor;
using CLayout = ck::tensor_layout::gemm::RowMajor;

auto PrepareGemmTensor(const ck::gemm_util::GemmParams& params)
{
    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                            std::vector<std::size_t>({stride, 1}));
            }
            else
            {
                return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                            std::vector<std::size_t>({1, stride}));
            }
        };

    // use fp32 host kernel to verify bf16 device kernel
    Tensor<ADataType> a_m_k_bf16(
        f_host_tensor_descriptor(params.M, params.K, params.StrideA, ALayout{}));
    Tensor<BDataType> b_k_n_bf16(
        f_host_tensor_descriptor(params.K, params.N, params.StrideB, BLayout{}));
    Tensor<CDataType> c_m_n_device_bf16(
        f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));

    Tensor<float> a_m_k_fp32(
        f_host_tensor_descriptor(params.M, params.K, params.StrideA, ALayout{}));
    Tensor<float> b_k_n_fp32(
        f_host_tensor_descriptor(params.K, params.N, params.StrideB, BLayout{}));
    Tensor<float> c_m_n_host_fp32(
        f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));
    Tensor<float> c_m_n_device_fp32(
        f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));

    a_m_k_bf16.GenerateTensorValue(GeneratorTensor_3<ADataType>{-0.5, 0.5});
    b_k_n_bf16.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});

    bf16_to_f32_(a_m_k_bf16, a_m_k_fp32);
    bf16_to_f32_(b_k_n_bf16, b_k_n_fp32);

    return std::make_tuple(a_m_k_bf16,
                           b_k_n_bf16,
                           c_m_n_device_bf16,
                           a_m_k_fp32,
                           b_k_n_fp32,
                           c_m_n_host_fp32,
                           c_m_n_device_fp32);
}

bool TestGemm(DeviceGemmPtr_& gemmPtr)
{
    // Arrange
    ck::gemm_util::GemmParams params;
    params.M       = 1024;
    params.N       = 1024;
    params.K       = 1024;
    params.StrideA = 1024;
    params.StrideB = 1024;
    params.StrideC = 1024;

    auto host_tensors                = PrepareGemmTensor(params);
    const Tensor<ADataType>& a_bf16  = std::get<0>(host_tensors);
    const Tensor<BDataType>& b_bf16  = std::get<1>(host_tensors);
    Tensor<CDataType>& c_device_bf16 = std::get<2>(host_tensors);
    Tensor<float>& a_fp32            = std::get<3>(host_tensors);
    Tensor<float>& b_fp32            = std::get<4>(host_tensors);
    Tensor<float>& c_host_fp32       = std::get<5>(host_tensors);
    Tensor<float>& c_device_fp32     = std::get<6>(host_tensors);

    auto a_element_op = PassThrough{};
    auto b_element_op = PassThrough{};
    auto c_element_op = PassThrough{};

    // use fp32 host kernel to verify bf16 device kernel
    using ReferenceGemmInstance = ck::tensor_operation::host::
        ReferenceGemm<float, float, float, PassThrough, PassThrough, PassThrough>;
    ck::gemm_util::RunHostGEMM<ReferenceGemmInstance>(
        a_fp32, b_fp32, c_host_fp32, a_element_op, b_element_op, c_element_op);

    // Act
    ck::gemm_util::RunDeviceGEMM(
        gemmPtr, params, a_bf16, b_bf16, c_device_bf16, a_element_op, b_element_op, c_element_op);

    bf16_to_f32_(c_device_bf16, c_device_fp32);

    // Assert
    bool res = test_util::check_err(
        c_device_fp32.mData, c_host_fp32.mData, "Error: incorrect results!", 1e-2f, 1e-3f);

    std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;

    return res;
}

} // anonymous namespace

int main()
{
    std::vector<DeviceGemmPtr_> gemmPtrs;
    ck::tensor_operation::device::device_gemm_instance::
        add_device_gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_nk_mn_instances(gemmPtrs);

    bool res = true;

    for(auto& gemmPtr : gemmPtrs)
    {
        res &= TestGemm(gemmPtr);
    }

    std::cout << "TestGemm ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;
}
