#include <algorithm>
#include <cstdlib>
#include <half.hpp>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

#include "cgemm_util.hpp"
#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "device_tensor.hpp"
#include "device_cgemm_4gemm_xdl_cshuffle.hpp"
#include "element_wise_operation.hpp"
#include "reference_cgemm.hpp"
#include "gemm_specialization.hpp"

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using DeviceCGemmNoOpPtr =
    ck::tensor_operation::device::DeviceGemmPtr<ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough>;

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_cgemm_instance {
void add_device_cgemm_4gemm_xdl_c_shuffle_bf16_bf16_bf16_km_kn_mn_instances(
    std::vector<DeviceCGemmNoOpPtr>&);
void add_device_cgemm_4gemm_xdl_c_shuffle_bf16_bf16_bf16_km_nk_mn_instances(
    std::vector<DeviceCGemmNoOpPtr>&);
void add_device_cgemm_4gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_nk_mn_instances(
    std::vector<DeviceCGemmNoOpPtr>&);
void add_device_cgemm_4gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_kn_mn_instances(
    std::vector<DeviceCGemmNoOpPtr>&);
} // namespace device_cgemm_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

int main()
{
    using RowMajor    = ck::tensor_layout::gemm::RowMajor;
    using ColumnMajor = ck::tensor_layout::gemm::ColumnMajor;

    bool res = true;
    std::vector<DeviceCGemmNoOpPtr> gemmPtrs;

    ck::tensor_operation::device::device_gemm_instance::
        add_device_cgemm_4gemm_xdl_c_shuffle_bf16_bf16_bf16_km_kn_mn_instances(cgemmPtrs);

    for(auto& cgemmPtr : cgemmPtrs)
    {
        res &= ck::cgemm_util::TestCGemmBF16<DeviceCGemmNoOpPtr,
                                             ColumnMajor,
                                             RowMajor,
                                             RowMajor,
                                             PassThrough,
                                             PassThrough,
                                             PassThrough>{}(cgemmPtr);
    }

    cgemmPtrs.clear();
    ck::tensor_operation::device::device_cgemm_instance::
        add_device_cgemm_4gemm_xdl_c_shuffle_bf16_bf16_bf16_km_nk_mn_instances(cgemmPtrs);

    for(auto& cgemmPtr : cgemmPtrs)
    {
        res &= ck::cgemm_util::TestCGemmBF16<DeviceCGemmNoOpPtr,
                                             ColumnMajor,
                                             ColumnMajor,
                                             RowMajor,
                                             PassThrough,
                                             PassThrough,
                                             PassThrough>{}(gemmPtr);
    }

    cgemmPtrs.clear();
    ck::tensor_operation::device::device_cgemm_instance::
        add_device_cgemm_4gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_kn_mn_instances(cgemmPtrs);

    for(auto& cgemmPtr : cgemmPtrs)
    {
        res &= ck::cgemm_util::TestCGemmBF16<DeviceCGemmNoOpPtr,
                                             RowMajor,
                                             RowMajor,
                                             RowMajor,
                                             PassThrough,
                                             PassThrough,
                                             PassThrough>{}(cgemmPtr);
    }

    cgemmPtrs.clear();
    ck::tensor_operation::device::device_cgemm_instance::
        add_device_cgemm_4gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_nk_mn_instances(cgemmPtrs);

    for(auto& cgemmPtr : cgemmPtrs)
    {
        res &= ck::cgemm_util::TestCGemmBF16<DeviceCGemmNoOpPtr,
                                             RowMajor,
                                             ColumnMajor,
                                             RowMajor,
                                             PassThrough,
                                             PassThrough,
                                             PassThrough>{}(cgemmPtr);
    }

    std::cout << "TestCGemm ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;
    return res ? 0 : 1;
}
