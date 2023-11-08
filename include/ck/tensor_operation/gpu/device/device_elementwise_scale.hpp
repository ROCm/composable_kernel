// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <memory>
#include <array>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename InDataTypeTuple,
          typename OutDataTypeTuple,
          typename ElementwiseOperation,
          typename UnaryOperation,
          typename Scale,
          index_t NumDim>
struct DeviceElementwise : public BaseOperator
{
    static constexpr int NumInput  = InDataTypeTuple::Size();
    static constexpr int NumOutput = OutDataTypeTuple::Size();

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::array<index_t, NumDim> lengths,
                        const std::array<std::array<index_t, NumDim>, NumInput> inStridesArray,
                        const std::array<std::array<index_t, NumDim>, NumOutput> outStridesArray,
                        const std::array<const void*, NumInput> in_dev_buffers,
                        const std::array<void*, NumOutput> out_dev_buffers,
                        ElementwiseOperation elementwise_op,
                        UnaryOperation unary_op,
                        Scale scale_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
}; // namespace device

template <typename InDataTypeTuple,
          typename OutDataTypeTuple,
          typename ElementwiseOperation,
          typename UnaryOperation,
          typename Scale,
          index_t NumDim>
using DeviceElementwisePtr = std::unique_ptr<DeviceElementwise<InDataTypeTuple,
                                                               OutDataTypeTuple,
                                                               ElementwiseOperation,
                                                               UnaryOperation,
                                                               Scale,
                                                               NumDim>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
