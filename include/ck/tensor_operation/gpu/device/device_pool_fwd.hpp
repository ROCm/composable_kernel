// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <array>

#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/utility/reduction_enums.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <index_t NDimSpatial, ReduceTensorOp ReduceOpId>
struct DevicePoolFwd : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* in_dev,
                        void* out_dev,
                        void* out_indices_dev,
                        ck::index_t N,
                        ck::index_t C,
                        std::array<ck::index_t, NDimSpatial> input_spatial_lengths,
                        std::array<ck::index_t, NDimSpatial> window_spatial_lengths,
                        std::array<ck::index_t, NDimSpatial> output_spatial_lengths,
                        std::array<ck::index_t, NDimSpatial> window_strides,
                        std::array<ck::index_t, NDimSpatial> input_left_pads,
                        std::array<ck::index_t, NDimSpatial> input_right_pads) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
