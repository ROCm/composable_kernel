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

using DeviceGemmNoOpPtr =
    ck::tensor_operation::device::DeviceGemmPtr<ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough>;

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_gemm_instance {
void add_device_gemm_xdl_c_shuffle_bf16_bf16_bf16_km_kn_mn_instances(
    std::vector<DeviceGemmNoOpPtr>&);
void add_device_gemm_xdl_c_shuffle_bf16_bf16_bf16_km_nk_mn_instances(
    std::vector<DeviceGemmNoOpPtr>&);
void add_device_gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_nk_mn_instances(
    std::vector<DeviceGemmNoOpPtr>&);
void add_device_gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_kn_mn_instances(
    std::vector<DeviceGemmNoOpPtr>&);
} // namespace device_gemm_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

int main()
{
    using RowMajor    = ck::tensor_layout::gemm::RowMajor;
    using ColumnMajor = ck::tensor_layout::gemm::ColumnMajor;

    bool res = true;
    std::vector<DeviceGemmNoOpPtr> gemmPtrs;

    ck::tensor_operation::device::device_gemm_instance::
        add_device_gemm_xdl_c_shuffle_bf16_bf16_bf16_km_kn_mn_instances(gemmPtrs);

    for(auto& gemmPtr : gemmPtrs)
    {
        res &= ck::gemm_util::TestGemmBF16<DeviceGemmNoOpPtr,
                                           ColumnMajor,
                                           RowMajor,
                                           RowMajor,
                                           PassThrough,
                                           PassThrough,
                                           PassThrough>{}(gemmPtr);
    }

    gemmPtrs.clear();
    ck::tensor_operation::device::device_gemm_instance::
        add_device_gemm_xdl_c_shuffle_bf16_bf16_bf16_km_nk_mn_instances(gemmPtrs);

    for(auto& gemmPtr : gemmPtrs)
    {
        res &= ck::gemm_util::TestGemmBF16<DeviceGemmNoOpPtr,
                                           ColumnMajor,
                                           ColumnMajor,
                                           RowMajor,
                                           PassThrough,
                                           PassThrough,
                                           PassThrough>{}(gemmPtr);
    }

    gemmPtrs.clear();
    ck::tensor_operation::device::device_gemm_instance::
        add_device_gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_kn_mn_instances(gemmPtrs);

    for(auto& gemmPtr : gemmPtrs)
    {
        res &= ck::gemm_util::TestGemmBF16<DeviceGemmNoOpPtr,
                                           RowMajor,
                                           RowMajor,
                                           RowMajor,
                                           PassThrough,
                                           PassThrough,
                                           PassThrough>{}(gemmPtr);
    }

    gemmPtrs.clear();
    ck::tensor_operation::device::device_gemm_instance::
        add_device_gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_nk_mn_instances(gemmPtrs);

    for(auto& gemmPtr : gemmPtrs)
    {
        res &= ck::gemm_util::TestGemmBF16<DeviceGemmNoOpPtr,
                                           RowMajor,
                                           ColumnMajor,
                                           RowMajor,
                                           PassThrough,
                                           PassThrough,
                                           PassThrough>{}(gemmPtr);
    }

    std::cout << "TestGemm ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;
    return res ? 0 : 1;
}
