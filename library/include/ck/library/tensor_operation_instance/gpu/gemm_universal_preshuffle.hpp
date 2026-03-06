// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_v2.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/utility/amd_ck_fp8.hpp"
#include "ck/utility/data_type.hpp"
#if defined(CK_USE_XDL) || defined(CK_USE_WMMA)
#include "gemm_universal_preshuffle.inc"
#endif

#include <memory>
#include <vector>

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
struct DeviceOperationInstanceFactory<
    // ck::tensor_operation::device::DeviceGemmV2BPreshuffle<ALayout,
    ck::tensor_operation::device::DeviceGemmV2BPreshuffle<
        ALayout,
        BLayout,
        CLayout,
        ADataType,
        BDataType,
        CDataType,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough>>
{
    using DeviceOp = DeviceGemmV2BPreshuffle<ALayout,
                                             BLayout,
                                             CLayout,
                                             ADataType,
                                             BDataType,
                                             CDataType,
                                             ck::tensor_operation::element_wise::PassThrough,
                                             ck::tensor_operation::element_wise::PassThrough,
                                             ck::tensor_operation::element_wise::PassThrough>;

    static auto GetInstances()
    {
#if defined(CK_USE_XDL) || defined(CK_USE_WMMA)
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
#if(defined(CK_ENABLE_BF16) && defined(CK_ENABLE_FP8))
        if constexpr(is_same_v<ADataType, f8_t> && is_same_v<BDataType, f8_t> &&
                     is_same_v<CDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<CLayout, Row>)
            {
#ifdef CK_USE_XDL
                add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma32x32_mn_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma32x32_mn_compute_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma_mn_p1_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma_mn_p2_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma_mn_p3_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma_mn_p4_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma_mn_p5_instances(
                    op_ptrs);
                add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma_mn_compute_instances_p1(
                    op_ptrs);
                add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma_mn_compute_instances_p2(
                    op_ptrs);
                add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma16x16_nk_mn_comp_default_instances_part5(
                    op_ptrs);
                add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma16x16_nk_mn_comp_default_instances_part6(
                    op_ptrs);
                add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma16x16_nk_mn_comp_default_instances_part4(
                    op_ptrs);
                add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma16x16_nk_mn_comp_default_instances_part3(
                    op_ptrs);
                add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma16x16_nk_mn_comp_default_instances_part2(
                    op_ptrs);
                add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma16x16_nk_mn_comp_default_instances_part1(
                    op_ptrs);
#endif
#ifdef CK_USE_WMMA
                add_device_gemm_universal_preshuffle_wmma_f8_f8_bf16_mk_wmma_mn_default_instances_p1(
                    op_ptrs);
                add_device_gemm_universal_preshuffle_wmma_f8_f8_bf16_mk_wmma_mn_default_instances_p2(
                    op_ptrs);
                add_device_gemm_universal_preshuffle_wmma_f8_f8_bf16_mk_wmma_mn_default_instances_p3(
                    op_ptrs);
                add_device_gemm_universal_preshuffle_wmma_f8_f8_bf16_mk_wmma_mn_default_instances_p4(
                    op_ptrs);
#endif
            }
        }
#endif
#if(defined(CK_ENABLE_FP16) && defined(CK_ENABLE_FP8))
        if constexpr(is_same_v<ADataType, f8_t> && is_same_v<BDataType, f8_t> &&
                     is_same_v<CDataType, half_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<CLayout, Row>)
            {
#ifdef CK_USE_XDL
                add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_compute_default_instances_p1(
                    op_ptrs);
                add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_compute_default_instances_p2(
                    op_ptrs);
                add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p1_default_instances(
                    op_ptrs);
                add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p2_default_instances(
                    op_ptrs);
                add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p3_default_instances(
                    op_ptrs);
                add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p4_default_instances(
                    op_ptrs);
                add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p5_default_instances(
                    op_ptrs);
                add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p1_default_instances_v2(
                    op_ptrs);
                add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p2_default_instances_v2(
                    op_ptrs);
                add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p3_default_instances_v2(
                    op_ptrs);
                add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p4_default_instances_v2(
                    op_ptrs);
                add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma_mn_p5_default_instances_v2(
                    op_ptrs);
                add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma16x16_mn_compute_default_instances_p1(
                    op_ptrs);
                add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma16x16_mn_compute_default_instances_p2(
                    op_ptrs);
                add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma16x16_mn_compute_default_instances_p3(
                    op_ptrs);
                add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma16x16_mn_compute_default_instances_p4(
                    op_ptrs);
                add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma16x16_mn_compute_default_instances_p5(
                    op_ptrs);
                add_device_gemm_universal_preshuffle_xdl_f8_f8_f16_mk_mfma16x16_mn_compute_default_instances_p6(
                    op_ptrs);
#endif
#ifdef CK_USE_WMMA
                add_device_gemm_universal_preshuffle_wmma_f8_f8_f16_mk_wmma_mn_default_instances_p1(
                    op_ptrs);
                add_device_gemm_universal_preshuffle_wmma_f8_f8_f16_mk_wmma_mn_default_instances_p2(
                    op_ptrs);
                add_device_gemm_universal_preshuffle_wmma_f8_f8_f16_mk_wmma_mn_default_instances_p3(
                    op_ptrs);
                add_device_gemm_universal_preshuffle_wmma_f8_f8_f16_mk_wmma_mn_default_instances_p4(
                    op_ptrs);
#endif
            }
        }
#endif
#endif // CK_USE_XDL || CK_USE_WMMA

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
