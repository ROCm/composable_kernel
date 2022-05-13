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
#include "device_tensor.hpp"
#include "device_cgemm_4gemm_xdl_cshuffle.hpp"
#include "element_wise_operation.hpp"
#include "gemm_specialization.hpp"

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using DeviceCGemmNoOpPtr =
    ck::tensor_operation::device::DevicecgemmPtr<ck::tensor_operation::element_wise::PassThrough,
                                                 ck::tensor_operation::element_wise::PassThrough,
                                                 ck::tensor_operation::element_wise::PassThrough>;

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_cgemm_instance {
void add_device_cgemm_4gemm_xdl_c_shuffle_f16_f16_f16_km_kn_mn_instances(
    std::vector<DeviceCGemmNoOpPtr>&);
void add_device_cgemm_4gemm_xdl_c_shuffle_f16_f16_f16_km_nk_mn_instances(
    std::vector<DeviceCGemmNoOpPtr>&);
void add_device_cgemm_4gemm_xdl_c_shuffle_f16_f16_f16_mk_nk_mn_instances(
    std::vector<DeviceCGemmNoOpPtr>&);
void add_device_cgemm_4gemm_xdl_c_shuffle_f16_f16_f16_mk_kn_mn_instances(
    std::vector<DeviceCGemmNoOpPtr>&);
} // namespace device_cgemm_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

int main()
{
    using ADataType = ck::half_t;
    using BDataType = ck::half_t;
    using CDataType = ck::half_t;

    using RowMajor    = ck::tensor_layout::gemm::RowMajor;
    using ColumnMajor = ck::tensor_layout::gemm::ColumnMajor;

    bool res = true;
    std::vector<DeviceCGemmNoOpPtr> cgemmPtrs;
    ck::tensor_operation::device::device_cgemm_instance::
        add_device_cgemm_4gemm_xdl_f16_f16_f16_km_kn_mn_instances(cgemmPtrs);
    ck::tensor_operation::device::device_cgemm_instance::
        add_device_cgemm_4gemm_xdl_splitk_f16_f16_f16_km_kn_mn_instances(cgemmPtrs);
    ck::tensor_operation::device::device_cgemm_instance::
        add_device_cgemm_4gemm_xdl_c_shuffle_f16_f16_f16_km_kn_mn_instances(cgemmPtrs);

    for(auto& cgemmPtr : cgemmPtrs)
    {
        res &= ck::cgemm_util::TestCGemm<DeviceCGemmNoOpPtr,
                                         ADataType,
                                         BDataType,
                                         CDataType,
                                         ColumnMajor,
                                         RowMajor,
                                         RowMajor,
                                         PassThrough,
                                         PassThrough,
                                         PassThrough>{}(cgemmPtr);
    }

    cgemmPtrs.clear();
    ck::tensor_operation::device::device_cgemm_instance::
        add_device_cgemm_4gemm_xdl_f16_f16_f16_km_nk_mn_instances(cgemmPtrs);
    ck::tensor_operation::device::device_cgemm_instance::
        add_device_cgemm_4gemm_xdl_splitk_f16_f16_f16_km_nk_mn_instances(cgemmPtrs);
    ck::tensor_operation::device::device_cgemm_instance::
        add_device_cgemm_4gemm_xdl_c_shuffle_f16_f16_f16_km_nk_mn_instances(cgemmPtrs);

    for(auto& cgemmPtr : cgemmPtrs)
    {
        res &= ck::cgemm_util::TestCGemm<DeviceCGemmNoOpPtr,
                                         ADataType,
                                         BDataType,
                                         CDataType,
                                         ColumnMajor,
                                         ColumnMajor,
                                         RowMajor,
                                         PassThrough,
                                         PassThrough,
                                         PassThrough>{}(cgemmPtr);
    }

    cgemmPtrs.clear();
    ck::tensor_operation::device::device_cgemm_instance::
        add_device_cgemm_4gemm_xdl_f16_f16_f16_mk_kn_mn_instances(cgemmPtrs);
    ck::tensor_operation::device::device_cgemm_instance::
        add_device_cgemm_4gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_instances(cgemmPtrs);
    ck::tensor_operation::device::device_cgemm_instance::
        add_device_cgemm_4gemm_xdl_c_shuffle_f16_f16_f16_mk_kn_mn_instances(cgemmPtrs);

    for(auto& cgemmPtr : cgemmPtrs)
    {
        res &= ck::cgemm_util::TestCGemm<DeviceCGemmNoOpPtr,
                                         ADataType,
                                         BDataType,
                                         CDataType,
                                         RowMajor,
                                         RowMajor,
                                         RowMajor,
                                         PassThrough,
                                         PassThrough,
                                         PassThrough>{}(cgemmPtr);
    }

    cgemmPtrs.clear();
    ck::tensor_operation::device::device_cgemm_instance::
        add_device_cgemm_4gemm_xdl_f16_f16_f16_mk_nk_mn_instances(cgemmPtrs);
    ck::tensor_operation::device::device_cgemm_instance::
        add_device_cgemm_4gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_instances(cgemmPtrs);
    ck::tensor_operation::device::device_cgemm_instance::
        add_device_cgemm_4gemm_xdl_c_shuffle_f16_f16_f16_mk_nk_mn_instances(cgemmPtrs);
    ck::tensor_operation::device::device_cgemm_instance::
        add_device_cgemm_4gemm_xdl_c_shuffle_2_stage_f16_f16_f16_mk_nk_mn_instances(cgemmPtrs);

    for(auto& cgemmPtr : cgemmPtrs)
    {
        res &= ck::cgemm_util::TestCGemm<DeviceCGemmNoOpPtr,
                                         ADataType,
                                         BDataType,
                                         CDataType,
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
