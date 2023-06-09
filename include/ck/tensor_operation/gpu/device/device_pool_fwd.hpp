// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>

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
          bool OutputIndex>
struct DevicePoolFwd : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in_dev,
                        void* p_out_dev,
                        void* p_out_indices_dev,
                        std::vector<ck::index_t> input_lengths,
                        std::vector<ck::index_t> window_lengths,
                        std::vector<ck::index_t> output_lengths,
                        std::vector<ck::index_t> input_stride,
                        std::vector<ck::index_t> output_stride,
                        std::vector<ck::index_t> indices_stride,
                        std::vector<ck::index_t> window_strides,
                        std::vector<ck::index_t> input_left_pads,
                        std::vector<ck::index_t> input_right_pads,
                        std::vector<ck::index_t> pooling_dims) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
