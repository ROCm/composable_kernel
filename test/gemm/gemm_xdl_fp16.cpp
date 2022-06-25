#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/host_tensor/device_memory.hpp"
#include "ck/library/host_tensor/host_tensor.hpp"
#include "ck/library/host_tensor/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"

#include "test/gemm/gemm_util.hpp"

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using DeviceGemmNoOpPtr =
    ck::tensor_operation::device::DeviceGemmPtr<ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough>;

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_gemm_instance {
void add_device_gemm_xdl_f16_f16_f16_km_kn_mn_instances(std::vector<DeviceGemmNoOpPtr>&);
void add_device_gemm_xdl_f16_f16_f16_km_nk_mn_instances(std::vector<DeviceGemmNoOpPtr>&);
void add_device_gemm_xdl_f16_f16_f16_mk_nk_mn_instances(std::vector<DeviceGemmNoOpPtr>&);
void add_device_gemm_xdl_f16_f16_f16_mk_kn_mn_instances(std::vector<DeviceGemmNoOpPtr>&);

void add_device_gemm_xdl_splitk_f16_f16_f16_km_kn_mn_instances(std::vector<DeviceGemmNoOpPtr>&);
void add_device_gemm_xdl_splitk_f16_f16_f16_km_nk_mn_instances(std::vector<DeviceGemmNoOpPtr>&);
void add_device_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_instances(std::vector<DeviceGemmNoOpPtr>&);
void add_device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_instances(std::vector<DeviceGemmNoOpPtr>&);

void add_device_gemm_xdl_c_shuffle_f16_f16_f16_km_kn_mn_instances(std::vector<DeviceGemmNoOpPtr>&);
void add_device_gemm_xdl_c_shuffle_f16_f16_f16_km_nk_mn_instances(std::vector<DeviceGemmNoOpPtr>&);
void add_device_gemm_xdl_c_shuffle_f16_f16_f16_mk_nk_mn_instances(std::vector<DeviceGemmNoOpPtr>&);
void add_device_gemm_xdl_c_shuffle_f16_f16_f16_mk_kn_mn_instances(std::vector<DeviceGemmNoOpPtr>&);

void add_device_gemm_xdl_c_shuffle_2_stage_f16_f16_f16_mk_nk_mn_instances(
    std::vector<DeviceGemmNoOpPtr>&);
} // namespace device_gemm_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

int main()
{
    using ADataType   = ck::half_t;
    using BDataType   = ck::half_t;
    using CDataType   = ck::half_t;
    using AccDataType = float;

    using RowMajor    = ck::tensor_layout::gemm::RowMajor;
    using ColumnMajor = ck::tensor_layout::gemm::ColumnMajor;

    bool res = true;
    std::vector<DeviceGemmNoOpPtr> gemmPtrs;
    ck::tensor_operation::device::device_gemm_instance::
        add_device_gemm_xdl_f16_f16_f16_km_kn_mn_instances(gemmPtrs);
    ck::tensor_operation::device::device_gemm_instance::
        add_device_gemm_xdl_splitk_f16_f16_f16_km_kn_mn_instances(gemmPtrs);
    ck::tensor_operation::device::device_gemm_instance::
        add_device_gemm_xdl_c_shuffle_f16_f16_f16_km_kn_mn_instances(gemmPtrs);

    for(auto& gemmPtr : gemmPtrs)
    {
        res &= ck::gemm_util::TestGemm<DeviceGemmNoOpPtr,
                                       ADataType,
                                       BDataType,
                                       CDataType,
                                       AccDataType,
                                       ColumnMajor,
                                       RowMajor,
                                       RowMajor,
                                       PassThrough,
                                       PassThrough,
                                       PassThrough>{}(gemmPtr);
    }

    gemmPtrs.clear();
    ck::tensor_operation::device::device_gemm_instance::
        add_device_gemm_xdl_f16_f16_f16_km_nk_mn_instances(gemmPtrs);
    ck::tensor_operation::device::device_gemm_instance::
        add_device_gemm_xdl_splitk_f16_f16_f16_km_nk_mn_instances(gemmPtrs);
    ck::tensor_operation::device::device_gemm_instance::
        add_device_gemm_xdl_c_shuffle_f16_f16_f16_km_nk_mn_instances(gemmPtrs);

    for(auto& gemmPtr : gemmPtrs)
    {
        res &= ck::gemm_util::TestGemm<DeviceGemmNoOpPtr,
                                       ADataType,
                                       BDataType,
                                       CDataType,
                                       AccDataType,
                                       ColumnMajor,
                                       ColumnMajor,
                                       RowMajor,
                                       PassThrough,
                                       PassThrough,
                                       PassThrough>{}(gemmPtr);
    }

    gemmPtrs.clear();
    ck::tensor_operation::device::device_gemm_instance::
        add_device_gemm_xdl_f16_f16_f16_mk_kn_mn_instances(gemmPtrs);
    ck::tensor_operation::device::device_gemm_instance::
        add_device_gemm_xdl_splitk_f16_f16_f16_mk_kn_mn_instances(gemmPtrs);
    ck::tensor_operation::device::device_gemm_instance::
        add_device_gemm_xdl_c_shuffle_f16_f16_f16_mk_kn_mn_instances(gemmPtrs);

    for(auto& gemmPtr : gemmPtrs)
    {
        res &= ck::gemm_util::TestGemm<DeviceGemmNoOpPtr,
                                       ADataType,
                                       BDataType,
                                       CDataType,
                                       AccDataType,
                                       RowMajor,
                                       RowMajor,
                                       RowMajor,
                                       PassThrough,
                                       PassThrough,
                                       PassThrough>{}(gemmPtr);
    }

    gemmPtrs.clear();
    ck::tensor_operation::device::device_gemm_instance::
        add_device_gemm_xdl_f16_f16_f16_mk_nk_mn_instances(gemmPtrs);
    ck::tensor_operation::device::device_gemm_instance::
        add_device_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_instances(gemmPtrs);
    ck::tensor_operation::device::device_gemm_instance::
        add_device_gemm_xdl_c_shuffle_f16_f16_f16_mk_nk_mn_instances(gemmPtrs);
    ck::tensor_operation::device::device_gemm_instance::
        add_device_gemm_xdl_c_shuffle_2_stage_f16_f16_f16_mk_nk_mn_instances(gemmPtrs);

    for(auto& gemmPtr : gemmPtrs)
    {
        res &= ck::gemm_util::TestGemm<DeviceGemmNoOpPtr,
                                       ADataType,
                                       BDataType,
                                       CDataType,
                                       AccDataType,
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
