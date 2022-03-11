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
void add_device_gemm_xdl_f16_f16_f16_mk_nk_mn_instances(std::vector<DeviceGemmPtr_>&);
}
} // namespace device
} // namespace tensor_operation
} // namespace ck

namespace {

using ADataType   = ck::half_t;
using BDataType   = ck::half_t;
using CDataType   = ck::half_t;
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

    Tensor<ADataType> a_m_k(
        f_host_tensor_descriptor(params.M, params.K, params.StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(
        f_host_tensor_descriptor(params.K, params.N, params.StrideB, BLayout{}));
    Tensor<CDataType> c_m_n_host_result(
        f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));
    Tensor<CDataType> c_m_n_device_result(
        f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));

    a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{-0.5, 0.5});
    b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});

    return std::make_tuple(a_m_k, b_k_n, c_m_n_host_result, c_m_n_device_result);
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

    auto host_tensors           = PrepareGemmTensor(params);
    const Tensor<ADataType>& a  = std::get<0>(host_tensors);
    const Tensor<BDataType>& b  = std::get<1>(host_tensors);
    Tensor<CDataType>& c_host   = std::get<2>(host_tensors);
    Tensor<CDataType>& c_device = std::get<3>(host_tensors);

    auto a_element_op = PassThrough{};
    auto b_element_op = PassThrough{};
    auto c_element_op = PassThrough{};

    using ReferenceGemmInstance = ck::tensor_operation::host::
        ReferenceGemm<ADataType, BDataType, CDataType, PassThrough, PassThrough, PassThrough>;
    ck::gemm_util::RunHostGEMM<ReferenceGemmInstance>(
        a, b, c_host, a_element_op, b_element_op, c_element_op);

    // Act
    ck::gemm_util::RunDeviceGEMM(
        gemmPtr, params, a, b, c_device, a_element_op, b_element_op, c_element_op);

    // Assert
    bool res = test_util::check_err(
        c_device.mData, c_host.mData, "Error: incorrect results!", 1e-3f, 1e-3f);

    std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;

    return res;
}

} // anonymous namespace

int main()
{
    std::vector<DeviceGemmPtr_> gemmPtrs;
    ck::tensor_operation::device::device_gemm_instance::
        add_device_gemm_xdl_f16_f16_f16_mk_nk_mn_instances(gemmPtrs);

    bool res = true;

    for(auto& gemmPtr : gemmPtrs)
    {
        res &= TestGemm(gemmPtr);
    }

    std::cout << "TestGemm ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;
}
