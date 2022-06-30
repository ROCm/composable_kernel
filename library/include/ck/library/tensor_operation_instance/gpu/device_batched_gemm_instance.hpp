// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using DeviceBatchedGemmNoOpPtr = ck::tensor_operation::device::DeviceBatchedGemmPtr<
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>;

void add_device_batched_gemm_xdl_bf16_bf16_bf16_gmk_gkn_gmn_instances(
    std::vector<DeviceBatchedGemmNoOpPtr>&);
void add_device_batched_gemm_xdl_bf16_bf16_bf16_gmk_gnk_gmn_instances(
    std::vector<DeviceBatchedGemmNoOpPtr>&);
void add_device_batched_gemm_xdl_bf16_bf16_bf16_gkm_gkn_gmn_instances(
    std::vector<DeviceBatchedGemmNoOpPtr>&);
void add_device_batched_gemm_xdl_bf16_bf16_bf16_gkm_gnk_gmn_instances(
    std::vector<DeviceBatchedGemmNoOpPtr>&);
void add_device_batched_gemm_xdl_f16_f16_f16_gmk_gkn_gmn_instances(
    std::vector<DeviceBatchedGemmNoOpPtr>&);
void add_device_batched_gemm_xdl_f16_f16_f16_gmk_gnk_gmn_instances(
    std::vector<DeviceBatchedGemmNoOpPtr>&);
void add_device_batched_gemm_xdl_f16_f16_f16_gkm_gkn_gmn_instances(
    std::vector<DeviceBatchedGemmNoOpPtr>&);
void add_device_batched_gemm_xdl_f16_f16_f16_gkm_gnk_gmn_instances(
    std::vector<DeviceBatchedGemmNoOpPtr>&);
void add_device_batched_gemm_xdl_f32_f32_f32_gmk_gkn_gmn_instances(
    std::vector<DeviceBatchedGemmNoOpPtr>&);
void add_device_batched_gemm_xdl_f32_f32_f32_gmk_gnk_gmn_instances(
    std::vector<DeviceBatchedGemmNoOpPtr>&);
void add_device_batched_gemm_xdl_f32_f32_f32_gkm_gkn_gmn_instances(
    std::vector<DeviceBatchedGemmNoOpPtr>&);
void add_device_batched_gemm_xdl_f32_f32_f32_gkm_gnk_gmn_instances(
    std::vector<DeviceBatchedGemmNoOpPtr>&);
void add_device_batched_gemm_xdl_int8_int8_int8_gmk_gkn_gmn_instances(
    std::vector<DeviceBatchedGemmNoOpPtr>&);
void add_device_batched_gemm_xdl_int8_int8_int8_gmk_gnk_gmn_instances(
    std::vector<DeviceBatchedGemmNoOpPtr>&);
void add_device_batched_gemm_xdl_int8_int8_int8_gkm_gkn_gmn_instances(
    std::vector<DeviceBatchedGemmNoOpPtr>&);
void add_device_batched_gemm_xdl_int8_int8_int8_gkm_gnk_gmn_instances(
    std::vector<DeviceBatchedGemmNoOpPtr>&);

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
auto get_device_batched_gemm_instances()
{
    std::vector<DeviceBatchedGemmNoOpPtr> op_ptrs;

    if constexpr(is_same<ADataType, float>::value && is_same<BDataType, float>::value &&
                 is_same<CDataType, float>::value)
    {
        if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value &&
                     is_same<BLayout, tensor_layout::gemm::RowMajor>::value &&
                     is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::instance::
                add_device_batched_gemm_xdl_f32_f32_f32_gmk_gkn_gmn_instances(op_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::instance::
                add_device_batched_gemm_xdl_f32_f32_f32_gmk_gnk_gmn_instances(op_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::instance::
                add_device_batched_gemm_xdl_f32_f32_f32_gkm_gkn_gmn_instances(op_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::instance::
                add_device_batched_gemm_xdl_f32_f32_f32_gkm_gnk_gmn_instances(op_ptrs);
        }
    }
    else if constexpr(is_same<ADataType, half_t>::value && is_same<BDataType, half_t>::value &&
                      is_same<CDataType, half_t>::value)
    {
        if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value &&
                     is_same<BLayout, tensor_layout::gemm::RowMajor>::value &&
                     is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::instance::
                add_device_batched_gemm_xdl_f16_f16_f16_gmk_gkn_gmn_instances(op_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::instance::
                add_device_batched_gemm_xdl_f16_f16_f16_gmk_gnk_gmn_instances(op_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::instance::
                add_device_batched_gemm_xdl_f16_f16_f16_gkm_gkn_gmn_instances(op_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::instance::
                add_device_batched_gemm_xdl_f16_f16_f16_gkm_gnk_gmn_instances(op_ptrs);
        }
    }
    else if constexpr(is_same<ADataType, bhalf_t>::value && is_same<BDataType, bhalf_t>::value &&
                      is_same<CDataType, bhalf_t>::value)
    {
        if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value &&
                     is_same<BLayout, tensor_layout::gemm::RowMajor>::value &&
                     is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::instance::
                add_device_batched_gemm_xdl_bf16_bf16_bf16_gmk_gkn_gmn_instances(op_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::instance::
                add_device_batched_gemm_xdl_bf16_bf16_bf16_gmk_gnk_gmn_instances(op_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::instance::
                add_device_batched_gemm_xdl_bf16_bf16_bf16_gkm_gkn_gmn_instances(op_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::instance::
                add_device_batched_gemm_xdl_bf16_bf16_bf16_gkm_gnk_gmn_instances(op_ptrs);
        }
    }
    else if constexpr(is_same<ADataType, int8_t>::value && is_same<BDataType, int8_t>::value &&
                      is_same<CDataType, int8_t>::value)
    {
        if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value &&
                     is_same<BLayout, tensor_layout::gemm::RowMajor>::value &&
                     is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::instance::
                add_device_batched_gemm_xdl_int8_int8_int8_gmk_gkn_gmn_instances(op_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::instance::
                add_device_batched_gemm_xdl_int8_int8_int8_gmk_gnk_gmn_instances(op_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::instance::
                add_device_batched_gemm_xdl_int8_int8_int8_gkm_gkn_gmn_instances(op_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::instance::
                add_device_batched_gemm_xdl_int8_int8_int8_gkm_gnk_gmn_instances(op_ptrs);
        }
    }

    return op_ptrs;
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
