// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include <iostream>
#include <vector>

#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <ck::index_t NumInputTensor,
          ck::index_t NumOutputTensor,
          index_t NDim,
          typename ElementwiseFunctor>
struct DeviceElementwise : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::array<const void*, NumInputTensor> p_inputs,
                        std::array<void*, NumOutputTensor> p_outputs,
                        std::vector<index_t> lengths,
                        std::vector<std::vector<index_t>> input_strides,
                        std::vector<std::vector<index_t>> output_strides,
                        ElementwiseFunctor functor) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <ck::index_t NumInputTensor,
          ck::index_t NumOutputTensor,
          index_t NDim,
          typename ElementwiseFunctor>
using DeviceElementwisePtr =
    std::unique_ptr<DeviceElementwise<NumInputTensor, NumOutputTensor, NDim, ElementwiseFunctor>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
