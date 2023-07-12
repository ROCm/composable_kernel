// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <array>

#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <ck::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation>
struct DeviceGroupedConvBwdWeight : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in,
                        void* p_wei,
                        const void* p_out,
                        const ck::index_t G,
                        const ck::index_t N,
                        const ck::index_t K,
                        const ck::index_t C,
                        const std::array<ck::index_t, NDimSpatial>& input_spatial_lengths,
                        const std::array<ck::index_t, NDimSpatial>& filter_spatial_lengths,
                        const std::array<ck::index_t, NDimSpatial>& output_spatial_lengths,
                        const std::array<ck::index_t, NDimSpatial + 3>& input_strides,
                        const std::array<ck::index_t, NDimSpatial + 3>& output_strides,
                        const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
                        const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
                        const std::array<ck::index_t, NDimSpatial>& input_left_pads,
                        const std::array<ck::index_t, NDimSpatial>& input_right_pads,
                        InElementwiseOperation in_element_op,
                        WeiElementwiseOperation wei_element_op,
                        OutElementwiseOperation out_element_op,
                        ck::index_t split_k) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
