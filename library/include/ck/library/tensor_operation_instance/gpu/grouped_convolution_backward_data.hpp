// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_data.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// conv2d backward data
void add_device_grouped_conv2d_bwd_data_xdl_gnhwc_gkyxc_gnhwk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdData<2,
                                                         GNHWC,
                                                         GKYXC,
                                                         GNHWK,
                                                         F16,
                                                         F16,
                                                         F16,
                                                         PassThrough,
                                                         PassThrough,
                                                         PassThrough>>>& instances);

// conv3d backward data
void add_device_grouped_conv3d_bwd_data_xdl_gnhwc_gkyxc_gnhwk_f16_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvBwdData<3,
                                                         GNDHWC,
                                                         GKZYXC,
                                                         GNDHWK,
                                                         F16,
                                                         F16,
                                                         F16,
                                                         PassThrough,
                                                         PassThrough,
                                                         PassThrough>>>& instances);

template <ck::index_t NumDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType>
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceGroupedConvBwdData<
    NumDimSpatial,
    InLayout,
    WeiLayout,
    OutLayout,
    InDataType,
    WeiDataType,
    OutDataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>>
{
    using DeviceOp = DeviceGroupedConvBwdData<NumDimSpatial,
                                              InLayout,
                                              WeiLayout,
                                              OutLayout,
                                              InDataType,
                                              WeiDataType,
                                              OutDataType,
                                              ck::tensor_operation::element_wise::PassThrough,
                                              ck::tensor_operation::element_wise::PassThrough,
                                              ck::tensor_operation::element_wise::PassThrough>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        if constexpr(NumDimSpatial == 2 && is_same_v<InLayout, GNHWC> &&
                     is_same_v<WeiLayout, GKYXC> && is_same_v<OutLayout, GNHWK>)
        {
            if constexpr(is_same_v<InDataType, F16> && is_same_v<WeiDataType, F16> &&
                         is_same_v<OutDataType, F16>)
            {
                add_device_grouped_conv2d_bwd_data_xdl_gnhwc_gkyxc_gnhwk_f16_instances(op_ptrs);
            }
        }
        else if constexpr(NumDimSpatial == 3 && is_same_v<InLayout, GNDHWC> &&
                          is_same_v<WeiLayout, GKZYXC> && is_same_v<OutLayout, GNDHWK>)
        {
            if constexpr(is_same_v<InDataType, F16> && is_same_v<WeiDataType, F16> &&
                         is_same_v<OutDataType, F16>)
            {
                add_device_grouped_conv3d_bwd_data_xdl_gndhwc_gkzyxc_gndhwk_f16_instances(op_ptrs);
            }
        }

        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
