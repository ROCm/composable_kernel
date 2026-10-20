// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Compatibility header for the ROCm 7.0.2.x series.
//
// composablekernel-devel shipped this path in ROCm 7.0.2.1 compute build 50.
// The CK develop import (#3773) renamed the file to
// device_grouped_conv_bwd_weight_explicit.hpp and the class from
// DeviceGroupedConvBwdWeight_Explicit_Xdl to DeviceGroupedConvBwdWeight_Explicit.
// The template parameter list is identical, so the old spelling stays available
// here for the rest of 7.0.2.x. New code should use the new path and name.

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_explicit.hpp"

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
          typename OutElementwiseOperation,
          typename DeviceGemmV3Op>
using DeviceGroupedConvBwdWeight_Explicit_Xdl =
    DeviceGroupedConvBwdWeight_Explicit<NDimSpatial,
                                        InLayout,
                                        WeiLayout,
                                        OutLayout,
                                        InDataType,
                                        WeiDataType,
                                        OutDataType,
                                        InElementwiseOperation,
                                        WeiElementwiseOperation,
                                        OutElementwiseOperation,
                                        DeviceGemmV3Op>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
