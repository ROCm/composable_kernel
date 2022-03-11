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
void add_device_gemm_xdl_f32_f32_f32_mk_nk_mn_instances(std::vector<DeviceGemmPtr_>&);
}
} // namespace device
} // namespace tensor_operation
} // namespace ck

namespace {

using ADataType   = float;
using BDataType   = float;
using CDataType   = float;
using AccDataType = float;

using ALayout = ck::tensor_layout::gemm::RowMajor;
using BLayout = ck::tensor_layout::gemm::ColumnMajor;
using CLayout = ck::tensor_layout::gemm::RowMajor;

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

    auto host_tensors = ck::gemm_util::
        PrepareGemmTensor<ADataType, BDataType, CDataType, ALayout, BLayout, CLayout>{}(params);

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
        c_device.mData, c_host.mData, "Error: incorrect results!", 1e-5f, 1e-4f);

    std::cout << (res ? "SUCCESS" : "FAILURE") << std::endl;

    return res;
}

} // anonymous namespace

int main()
{
    std::vector<DeviceGemmPtr_> gemmPtrs;
    ck::tensor_operation::device::device_gemm_instance::
        add_device_gemm_xdl_f32_f32_f32_mk_nk_mn_instances(gemmPtrs);

    bool res = true;

    for(auto& gemmPtr : gemmPtrs)
    {
        res &= TestGemm(gemmPtr);
    }

    std::cout << "TestGemm ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;
}
