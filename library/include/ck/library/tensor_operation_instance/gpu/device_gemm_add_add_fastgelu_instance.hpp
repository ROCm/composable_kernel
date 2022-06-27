// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_gemm_instance {

using DeviceGemmAddAddFastGeluPtr = ck::tensor_operation::device::DeviceGemmMultipleDPtr<
    2,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::AddAddFastGelu>;

void add_device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_mk_kn_mn_instances(
    std::vector<DeviceGemmAddAddFastGeluPtr>&);
void add_device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_mk_nk_mn_instances(
    std::vector<DeviceGemmAddAddFastGeluPtr>&);
void add_device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_km_kn_mn_instances(
    std::vector<DeviceGemmAddAddFastGeluPtr>&);
void add_device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_km_nk_mn_instances(
    std::vector<DeviceGemmAddAddFastGeluPtr>&);

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename D0DataType,
          typename D1DataType,
          typename EDataType,
          typename ALayout,
          typename BLayout,
          typename D0Layout,
          typename D1Layout,
          typename ELayout>
auto get_device_gemm_add_add_fastgelu_instances()
{
    std::vector<DeviceGemmAddAddFastGeluPtr> op_ptrs;

    if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                 is_same_v<EDataType, half_t>)
    {
        if constexpr(is_same_v<ALayout, tensor_layout::gemm::RowMajor> &&
                     is_same_v<BLayout, tensor_layout::gemm::RowMajor> &&
                     is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
        {
            ck::tensor_operation::device::device_gemm_instance::
                add_device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_mk_kn_mn_instances(
                    op_ptrs);
        }
        else if constexpr(is_same_v<ALayout, tensor_layout::gemm::RowMajor> &&
                          is_same_v<BLayout, tensor_layout::gemm::ColumnMajor> &&
                          is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
        {
            ck::tensor_operation::device::device_gemm_instance::
                add_device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_mk_nk_mn_instances(
                    op_ptrs);
        }
        else if constexpr(is_same_v<ALayout, tensor_layout::gemm::ColumnMajor> &&
                          is_same_v<BLayout, tensor_layout::gemm::RowMajor> &&
                          is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
        {
            ck::tensor_operation::device::device_gemm_instance::
                add_device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_km_kn_mn_instances(
                    op_ptrs);
        }
        else if constexpr(is_same_v<ALayout, tensor_layout::gemm::ColumnMajor> &&
                          is_same_v<BLayout, tensor_layout::gemm::ColumnMajor> &&
                          is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
        {
            ck::tensor_operation::device::device_gemm_instance::
                add_device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_km_nk_mn_instances(
                    op_ptrs);
        }
    }

    return op_ptrs;
}

} // namespace device_gemm_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
