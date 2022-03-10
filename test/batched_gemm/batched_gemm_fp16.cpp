#include <half.hpp>
#include <tuple>
#include <vector>

#include "batched_gemm_util.hpp"
#include "reference_batched_gemm.hpp"
#include "config.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "device_tensor.hpp"
#include "device_batched_gemm_xdl.hpp"
#include "element_wise_operation.hpp"
#include "test_util.hpp"

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using DeviceBatchedGemmPtr =
    ck::tensor_operation::device::DeviceGemmPtr<ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough>;

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_batched_gemm_instance {
void add_device_batched_gemm_xdl_f16_f16_f16_gmk_gnk_gmn_instances(
    std::vector<DeviceBatchedGemmPtr>& instances);
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

auto PrepareGemmTensor(const std::size_t batch_count,
                       const ck::batched_gemm_util::GemmParams& params)
{
    auto f_host_tensor_descriptor =
        [batch_count](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor(std::vector<std::size_t>({batch_count, row, col}),
                                            std::vector<std::size_t>({row * stride, stride, 1}));
            }
            else
            {
                return HostTensorDescriptor(std::vector<std::size_t>({batch_count, row, col}),
                                            std::vector<std::size_t>({col * stride, 1, stride}));
            }
        };

    Tensor<ADataType> a_g_m_k(
        f_host_tensor_descriptor(params.M, params.K, params.StrideA, ALayout{}));
    Tensor<BDataType> b_g_k_n(
        f_host_tensor_descriptor(params.K, params.N, params.StrideB, BLayout{}));
    Tensor<CDataType> c_g_m_n_host_result(
        f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));
    Tensor<CDataType> c_g_m_n_device_result(
        f_host_tensor_descriptor(params.M, params.N, params.StrideC, CLayout{}));

    a_g_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{-0.5, 0.5});
    b_g_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});

    return std::make_tuple(a_g_m_k, b_g_k_n, c_g_m_n_host_result, c_g_m_n_device_result);
}

bool TestBatchedGemm(const std::size_t batch_count, DeviceBatchedGemmPtr& gemmPtr)
{
    // Arrange
    ck::batched_gemm_util::GemmParams params;
    params.M       = 1024;
    params.N       = 1024;
    params.K       = 1024;
    params.StrideA = 1024;
    params.StrideB = 1024;
    params.StrideC = 1024;

    auto host_tensors           = PrepareGemmTensor(batch_count, params);
    const Tensor<ADataType>& a  = std::get<0>(host_tensors);
    const Tensor<BDataType>& b  = std::get<1>(host_tensors);
    Tensor<CDataType>& c_host   = std::get<2>(host_tensors);
    Tensor<CDataType>& c_device = std::get<3>(host_tensors);

    auto a_element_op = PassThrough{};
    auto b_element_op = PassThrough{};
    auto c_element_op = PassThrough{};

    using ReferenceBatchedGemmInstance =
        ck::tensor_operation::host::ReferenceBatchedGemm<ADataType,
                                                         BDataType,
                                                         CDataType,
                                                         PassThrough,
                                                         PassThrough,
                                                         PassThrough>;
    ck::batched_gemm_util::RunHostBatchedGemm<ReferenceBatchedGemmInstance>(
        a, b, c_host, a_element_op, b_element_op, c_element_op);

    // Act
    ck::batched_gemm_util::RunDeviceBatchedGemm(
        gemmPtr, params, a, b, c_device, a_element_op, b_element_op, c_element_op);

    // Assert
    // bool res = test_util::check_err(
    // c_device.mData, c_host.mData, "Error: incorrect results!", 1e-5f, 1e-4f);
    bool res = check_error(c_device, c_host) < 0.007815f;

    std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;

    return res;
}
} // namespace

int main()
{
    std::vector<DeviceBatchedGemmPtr> batched_gemm_ptrs;
    ck::tensor_operation::device::device_batched_gemm_instance::
        add_device_batched_gemm_xdl_f16_f16_f16_gmk_gnk_gmn_instances(batched_gemm_ptrs);

    bool res = true;

    const std::size_t batch_count = 4;
    for(auto& gemmPtr : batched_gemm_ptrs)
    {
        res &= TestBatchedGemm(batch_count, gemmPtr);
    }

    std::cout << "TestGemm ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;
}
