// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>
#include <memory>
#include <array>
#include <iostream>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/utility/reduction_enums.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <index_t NumReduction,
          typename InElementwiseOperationTuple,
          typename AccElementwiseOperationTuple>
struct DeviceMultipleReduce : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::vector<index_t> inLengths,
                        const std::vector<index_t> inStrides,
                        const std::vector<index_t> outLengths,
                        const std::vector<index_t> outStrides,
                        const std::vector<int> reduceDims,
                        const std::array<float, NumReduction> alpha_values,
                        const std::array<float, NumReduction> beta_values,
                        const void* in_dev,
                        const std::array<void*, NumReduction> out_dev_buffers,
                        const InElementwiseOperationTuple in_elementwise_op_tuple,
                        const AccElementwiseOperationTuple acc_elementwise_op_tuple) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <index_t NumReduction,
          typename InElementwiseOperationTuple,
          typename AccElementwiseOperationTuple>
using DeviceMultipleReducePtr = std::unique_ptr<
    DeviceMultipleReduce<NumReduction, InElementwiseOperationTuple, AccElementwiseOperationTuple>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
