// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>
#include <memory>
#include <iostream>

#include "ck/utility/common_header.hpp"
#include "ck/utility/reduction_enums.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename InElementwiseOperation, typename AccElementwiseOperation>
struct DeviceReduce : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::vector<index_t> inLengths,
                        const std::vector<index_t> inStrides,
                        const std::vector<index_t> outLengths,
                        const std::vector<index_t> outStrides,
                        const std::vector<int> reduceDims,
                        float alpha,
                        float beta,
                        const void* in_dev,
                        const void* in_index_dev,
                        void* out_dev,
                        void* out_index_dev,
                        const InElementwiseOperation in_elementwise_op,
                        const AccElementwiseOperation acc_elementwise_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename InElementwiseOperation, typename AccElementwiseOperation>
using DeviceReducePtr =
    std::unique_ptr<DeviceReduce<InElementwiseOperation, AccElementwiseOperation>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
