// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_v2.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

#ifdef CK_USE_WMMA
#include "gemm_universal_wmma.inc"
#endif
#ifdef CK_USE_XDL
#include "gemm_universal_xdl.inc"
#endif

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
#ifdef CK_USE_XDL
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
#if(defined(CK_ENABLE_BF16) && defined(CK_ENABLE_FP8))
        if constexpr(is_same_v<ADataType, f8_t> && is_same_v<BDataType, f8_t> &&
                     is_same_v<CDataType, bhalf_t>)
        {
            if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
                         is_same_v<CLayout, Row>)
            {
                add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma16x16_nk_mn_comp_default_instances_part5(
                    op_ptrs);
                add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma16x16_nk_mn_comp_default_instances_part6(
                    op_ptrs);
                add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma16x16_nk_mn_comp_default_instances_part4(
                    op_ptrs);
                add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_mfma16x16_nk_mn_comp_default_instances_part3(
                    op_ptrs);
                add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_nk_mn_comp_default_instances_part2(
                    op_ptrs);
                add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_nk_mn_comp_default_instances_part1(
                    op_ptrs);
                // add_device_gemm_xdl_universal_preshuffle_f8_f8_bf16_mk_nk_mn_comp_kpadding_instances(
                //     op_ptrs);
            }
        }
#endif
// #ifdef CK_ENABLE_FP16
//         if constexpr(is_same_v<ADataType, half_t> && is_same_v<BDataType, half_t> &&
//                      is_same_v<CDataType, half_t>)
//         {

//             if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Col> &&
//                          is_same_v<CLayout, Row>)
//             {
//                 add_device_gemm_xdl_universal_preshuffle_f16_f16_f16_mk_nk_mn_comp_default_instances(
//                     op_ptrs);
//                 add_device_gemm_xdl_universal_preshuffle_f16_f16_f16_mk_nk_mn_comp_kpadding_instances(
//                     op_ptrs);
//             }
//         }
// #endif
#endif // CK_USE_XDL

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
