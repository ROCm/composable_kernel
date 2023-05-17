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

template <index_t InOutRank,
          index_t WindowRank,
          typename InDataType,
          typename OutDataType,
          typename IndexDataType,
          ReduceTensorOp ReduceOpId,
          bool OuputIndex>
struct DevicePoolFwd : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in_dev,
                        void* p_out_dev,
                        void* p_out_indices_dev,
                        std::array<ck::index_t, InOutRank> input_stride,
                        std::array<ck::index_t, InOutRank> output_stride,
                        std::array<ck::index_t, InOutRank> indices_stride,
                        std::array<ck::index_t, InOutRank> input_lengths,
                        std::array<ck::index_t, WindowRank> window_lengths,
                        std::array<ck::index_t, InOutRank> output_lengths,
                        std::array<ck::index_t, WindowRank> window_strides,
                        std::array<ck::index_t, WindowRank> input_left_pads,
                        std::array<ck::index_t, WindowRank> input_right_pads,
                        std::array<ck::index_t, WindowRank> pooling_dims) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
