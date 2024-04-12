// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_weight_multiple_d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#ifdef CK_USE_XDL
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv3d_bwd_weight_xdl_bilinear_ndhwgc_gkzyxc_ndhwgk_bf16_f32_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeightMultipleD<3,
                                                                    NDHWGC,
                                                                    GKZYXC,
                                                                    NDHWGK,
                                                                    Tuple<GKZYXC>,
                                                                    BF16,
                                                                    F32,
                                                                    BF16,
                                                                    Tuple<F32>,
                                                                    PassThrough,
                                                                    Bilinear,
                                                                    PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_bwd_weight_xdl_bilinear_ndhwgc_gkzyxc_ndhwgk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeightMultipleD<3,
                                                                    NDHWGC,
                                                                    GKZYXC,
                                                                    NDHWGK,
                                                                    Tuple<GKZYXC>,
                                                                    F16,
                                                                    F16,
                                                                    F16,
                                                                    Tuple<F16>,
                                                                    PassThrough,
                                                                    Bilinear,
                                                                    PassThrough>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv3d_bwd_weight_xdl_bilinear_ndhwgc_gkzyxc_ndhwgk_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeightMultipleD<3,
                                                                    NDHWGC,
                                                                    GKZYXC,
                                                                    NDHWGK,
                                                                    Tuple<GKZYXC>,
                                                                    F32,
                                                                    F32,
                                                                    F32,
                                                                    Tuple<F32>,
                                                                    PassThrough,
                                                                    Bilinear,
                                                                    PassThrough>>>& instances);
#endif
#if defined CK_ENABLE_FP16 && defined CK_ENABLE_FP8 && defined CK_ENABLE_BF8
void add_device_grouped_conv3d_bwd_weight_xdl_bilinear_ndhwgc_gkzyxc_ndhwgk_f16_comp_bf8_f8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdWeightMultipleD<3,
                                                                    NDHWGC,
                                                                    GKZYXC,
                                                                    NDHWGK,
                                                                    Tuple<GKZYXC>,
                                                                    F16,
                                                                    F16,
                                                                    F16,
                                                                    Tuple<F16>,
                                                                    PassThrough,
                                                                    Bilinear,
                                                                    PassThrough,
                                                                    BF8,
                                                                    F8>>>& instances);
#endif
#endif

template <ck::index_t NumDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename DsLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename DsDataType,
          typename ComputeTypeA,
          typename ComputeTypeB>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGroupedConvBwdWeightMultipleD<
        NumDimSpatial,
        InLayout,
        WeiLayout,
        OutLayout,
        DsLayout,
        InDataType,
        WeiDataType,
        OutDataType,
        DsDataType,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::Bilinear,
        ck::tensor_operation::element_wise::PassThrough,
        ComputeTypeA,
        ComputeTypeB>>
{
    using DeviceOp =
        DeviceGroupedConvBwdWeightMultipleD<NumDimSpatial,
                                            InLayout,
                                            WeiLayout,
                                            OutLayout,
                                            DsLayout,
                                            InDataType,
                                            WeiDataType,
                                            OutDataType,
                                            DsDataType,
                                            ck::tensor_operation::element_wise::PassThrough,
                                            ck::tensor_operation::element_wise::Bilinear,
                                            ck::tensor_operation::element_wise::PassThrough,
                                            ComputeTypeA,
                                            ComputeTypeB>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

#ifdef CK_USE_XDL
        if constexpr(NumDimSpatial == 3)
        {
            if constexpr(is_same_v<InLayout, NDHWGC> && is_same_v<WeiLayout, GKZYXC> &&
                         is_same_v<OutLayout, NDHWGK>)
            {
#ifdef CK_ENABLE_FP32
                if constexpr(is_same_v<InDataType, float> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, float> && is_same_v<ComputeTypeA, float> &&
                             is_same_v<ComputeTypeB, float>)
                {
                    add_device_grouped_conv3d_bwd_weight_xdl_bilinear_ndhwgc_gkzyxc_ndhwgk_f32_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_FP16
                if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                             is_same_v<OutDataType, half_t> && is_same_v<ComputeTypeA, half_t> &&
                             is_same_v<ComputeTypeB, half_t>)
                {
                    add_device_grouped_conv3d_bwd_weight_xdl_bilinear_ndhwgc_gkzyxc_ndhwgk_f16_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                if constexpr(is_same_v<InDataType, ck::bhalf_t> && is_same_v<WeiDataType, float> &&
                             is_same_v<OutDataType, ck::bhalf_t> &&
                             is_same_v<ComputeTypeA, ck::bhalf_t> &&
                             is_same_v<ComputeTypeB, ck::bhalf_t>)
                {
                    add_device_grouped_conv3d_bwd_weight_xdl_bilinear_ndhwgc_gkzyxc_ndhwgk_bf16_f32_bf16_instances(
                        op_ptrs);
                }
#endif
#if defined CK_ENABLE_FP16 && defined CK_ENABLE_FP8 && defined CK_ENABLE_BF8
                if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
                             is_same_v<OutDataType, half_t> && is_same_v<ComputeTypeA, bf8_t> &&
                             is_same_v<ComputeTypeB, f8_t>)
                {
                    add_device_grouped_conv3d_bwd_weight_xdl_bilinear_ndhwgc_gkzyxc_ndhwgk_f16_comp_bf8_f8_instances(
                        op_ptrs);
                }
#endif
            }
        }
#endif
        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
