// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_data_multiple_d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

#ifdef CK_ENABLE_FP16
void add_device_grouped_conv3d_bwd_data_xdl_bilinear_ndhwgk_gkzyxc_ndhwgc_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Tuple<NDHWGC>,
                                                                  NDHWGC,
                                                                  F16,
                                                                  F16,
                                                                  Tuple<F16>,
                                                                  F16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  Bilinear>>>& instances);
#endif
#ifdef CK_ENABLE_FP32
void add_device_grouped_conv3d_bwd_data_xdl_bilinear_ndhwgk_gkzyxc_ndhwgc_f32_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Tuple<NDHWGC>,
                                                                  NDHWGC,
                                                                  F32,
                                                                  F32,
                                                                  Tuple<F32>,
                                                                  F32,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  Bilinear>>>& instances);
#endif
#ifdef CK_ENABLE_BF16
void add_device_grouped_conv3d_bwd_data_xdl_bilinear_ndhwgk_gkzyxc_ndhwgc_bf16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<3,
                                                                  NDHWGK,
                                                                  GKZYXC,
                                                                  Tuple<NDHWGC>,
                                                                  NDHWGC,
                                                                  BF16,
                                                                  BF16,
                                                                  Tuple<BF16>,
                                                                  BF16,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  Bilinear>>>& instances);
#endif
template <ck::index_t NumDimSpatial,
          typename OutLayout,
          typename WeiLayout,
          typename InLayout,
          typename OutDataType,
          typename WeiDataType,
          typename InDataType,
          typename ComputeTypeA,
          typename ComputeTypeB>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD<
        NumDimSpatial,
        OutLayout,
        WeiLayout,
        Tuple<InLayout>,
        InLayout,
        OutDataType,
        WeiDataType,
        Tuple<InDataType>,
        InDataType,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::Bilinear,
        ComputeTypeA,
        ComputeTypeB>>
{
    using DeviceOp =
        DeviceGroupedConvBwdDataMultipleD<NumDimSpatial,
                                          OutLayout,
                                          WeiLayout,
                                          Tuple<InLayout>,
                                          InLayout,
                                          OutDataType,
                                          WeiDataType,
                                          Tuple<InDataType>,
                                          InDataType,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::PassThrough,
                                          ck::tensor_operation::element_wise::Bilinear,
                                          ComputeTypeA,
                                          ComputeTypeB>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
        if constexpr(NumDimSpatial == 3)
        {
            if constexpr(is_same_v<InLayout, NDHWGC> && is_same_v<WeiLayout, GKZYXC> &&
                         is_same_v<OutLayout, NDHWGK>)
            {
#ifdef CK_ENABLE_FP16
                if constexpr(is_same_v<InDataType, F16> && is_same_v<WeiDataType, F16> &&
                             is_same_v<OutDataType, F16> && is_same_v<ComputeTypeA, F16> &&
                             is_same_v<ComputeTypeB, F16>)
                {
                    add_device_grouped_conv3d_bwd_data_xdl_bilinear_ndhwgk_gkzyxc_ndhwgc_f16_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_FP32
                else if constexpr(is_same_v<InDataType, F32> && is_same_v<WeiDataType, F32> &&
                                  is_same_v<OutDataType, F32> && is_same_v<ComputeTypeA, F32> &&
                                  is_same_v<ComputeTypeB, F32>)
                {
                    add_device_grouped_conv3d_bwd_data_xdl_bilinear_ndhwgk_gkzyxc_ndhwgc_f32_instances(
                        op_ptrs);
                }
#endif
#ifdef CK_ENABLE_BF16
                else if constexpr(is_same_v<InDataType, BF16> && is_same_v<WeiDataType, BF16> &&
                                  is_same_v<OutDataType, BF16> && is_same_v<ComputeTypeA, BF16> &&
                                  is_same_v<ComputeTypeB, BF16>)
                {
                    add_device_grouped_conv3d_bwd_data_xdl_bilinear_ndhwgk_gkzyxc_ndhwgc_bf16_instances(
                        op_ptrs);
                }
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
