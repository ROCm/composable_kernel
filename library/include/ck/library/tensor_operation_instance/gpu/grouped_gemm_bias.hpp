// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_gemm_fixed_nk.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_wmma_fixed_nk.hpp"
#include "ck/utility/type.hpp"

#include <memory>
#include <vector>

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// fp16_output
void add_device_grouped_gemm_xdl_fixed_nk_bias_f16_f16_f16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmFixedNK<Row,
                                                         Row,
                                                         Row_Tuple,
                                                         Row,
                                                         F16,
                                                         F16,
                                                         F32_Tuple,
                                                         F16,
                                                         PassThrough,
                                                         PassThrough,
                                                         Add>>>& instances);

void add_device_grouped_gemm_xdl_fixed_nk_bias_f16_f16_f16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmFixedNK<Row,
                                                         Col,
                                                         Row_Tuple,
                                                         Row,
                                                         F16,
                                                         F16,
                                                         F32_Tuple,
                                                         F16,
                                                         PassThrough,
                                                         PassThrough,
                                                         Add>>>& instances);

// fp32_output
void add_device_grouped_gemm_xdl_fixed_nk_bias_f16_f16_f32_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmFixedNK<Row,
                                                         Row,
                                                         Row_Tuple,
                                                         Row,
                                                         F16,
                                                         F16,
                                                         F32_Tuple,
                                                         F32,
                                                         PassThrough,
                                                         PassThrough,
                                                         Add>>>& instances);

void add_device_grouped_gemm_xdl_fixed_nk_bias_f16_f16_f32_mk_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmFixedNK<Row,
                                                         Col,
                                                         Row_Tuple,
                                                         Row,
                                                         F16,
                                                         F16,
                                                         F32_Tuple,
                                                         F32,
                                                         PassThrough,
                                                         PassThrough,
                                                         Add>>>& instances);

void add_device_grouped_gemm_wmma_fixed_nk_bias_f16_f16_f16_mk_kn_mn_instances(
    std::vector<
        std::unique_ptr<DeviceGroupedGemmFixedNK<Row,
                                                 Row,
                                                 Row_Tuple,
                                                 Row,
                                                 F16,
                                                 F16,
                                                 F16_Tuple,
                                                 F16,
                                                 PassThrough,
                                                 PassThrough,
                                                 ck::tensor_operation::element_wise::SplitKAdd>>>&
        instances);

void add_device_grouped_gemm_wmma_fixed_nk_bias_f16_f16_f16_mk_nk_mn_instances(
    std::vector<
        std::unique_ptr<DeviceGroupedGemmFixedNK<Row,
                                                 Col,
                                                 Row_Tuple,
                                                 Row,
                                                 F16,
                                                 F16,
                                                 F16_Tuple,
                                                 F16,
                                                 PassThrough,
                                                 PassThrough,
                                                 ck::tensor_operation::element_wise::SplitKAdd>>>&
        instances);

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGroupedGemmFixedNK<ALayout,
                                                           BLayout,
                                                           DsLayout,
                                                           ELayout,
                                                           ADataType,
                                                           BDataType,
                                                           DsDataType,
                                                           EDataType,
                                                           PassThrough,
                                                           PassThrough,
                                                           Add>>
{
    using DeviceOp = DeviceGroupedGemmFixedNK<ALayout,
                                              BLayout,
                                              DsLayout,
                                              ELayout,
                                              ADataType,
                                              BDataType,
                                              DsDataType,
                                              EDataType,
                                              PassThrough,
                                              PassThrough,
                                              Add>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        // fp16_output
        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                     is_same_v<EDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<ELayout, Row>)
            {
#if defined(CK_USE_XDL)
                add_device_grouped_gemm_xdl_fixed_nk_bias_f16_f16_f16_mk_kn_mn_instances(op_ptrs);
#endif
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<ELayout, Row>)
            {
#if defined(CK_USE_XDL)
                add_device_grouped_gemm_xdl_fixed_nk_bias_f16_f16_f16_mk_nk_mn_instances(op_ptrs);
#endif
            }
        }

        // fp32_output
        else if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                          is_same_v<EDataType, float>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<ELayout, Row>)
            {
#ifdef CK_USE_XDL
                add_device_grouped_gemm_xdl_fixed_nk_bias_f16_f16_f32_mk_kn_mn_instances(op_ptrs);
#endif
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<ELayout, Row>)
            {
#ifdef CK_USE_XDL
                add_device_grouped_gemm_xdl_fixed_nk_bias_f16_f16_f32_mk_nk_mn_instances(op_ptrs);
#endif
            }
        }
        return op_ptrs;
    }
};

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceGroupedGemmFixedNK<
    ALayout,
    BLayout,
    DsLayout,
    ELayout,
    ADataType,
    BDataType,
    DsDataType,
    EDataType,
    PassThrough,
    PassThrough,
    ck::tensor_operation::element_wise::SplitKAdd>>
{
    using DeviceOp = DeviceGroupedGemmFixedNK<ALayout,
                                              BLayout,
                                              DsLayout,
                                              ELayout,
                                              ADataType,
                                              BDataType,
                                              DsDataType,
                                              EDataType,
                                              PassThrough,
                                              PassThrough,
                                              ck::tensor_operation::element_wise::SplitKAdd>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        // fp16_output
        if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
                     is_same_v<EDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
                         is_same_v<ELayout, Row>)
            {
#if defined(CK_USE_WMMA)
                add_device_grouped_gemm_wmma_fixed_nk_bias_f16_f16_f16_mk_kn_mn_instances(op_ptrs);
#endif
            }
            else if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                              is_same_v<ELayout, Row>)
            {
#if defined(CK_USE_WMMA)
                add_device_grouped_gemm_wmma_fixed_nk_bias_f16_f16_f16_mk_nk_mn_instances(op_ptrs);
#endif
            }
        }
        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
