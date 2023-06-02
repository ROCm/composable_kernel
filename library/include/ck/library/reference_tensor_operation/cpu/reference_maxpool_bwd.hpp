// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>

#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"
#include "ck/utility/reduction_functions_accumulate.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

namespace ck {
namespace tensor_operation {
namespace host {
using namespace std;

template <typename DOutDataType,
          typename IndexDataType,
          typename DInDataType,
          typename ElementwiseOperation>
struct ReferenceMaxPoolBwd : public device::BaseOperator
{
    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(const Tensor<DOutDataType>& dout,
                 const Tensor<IndexDataType>& indices,
                 Tensor<DInDataType>& din,
                 ElementwiseOperation elementwise_op)
            : dout_(dout), indices_(indices), din_(din), elementwise_op_(elementwise_op)
        {
        }

        const Tensor<DOutDataType>& dout_;
        const Tensor<IndexDataType>& indices_;
        Tensor<DInDataType>& din_;
        ElementwiseOperation elementwise_op_;
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        float Run(const Argument& arg)
        {
            int din_length  = arg.din_.GetElementSpaceSize();
            int dout_length = arg.dout_.GetElementSpaceSize();

            for(int i = 0; i < dout_length; ++i)
            {
                int index = arg.indices_.mData[i];
                if(index >= 0 && index < din_length)
                    arg.din_.mData[index] += arg.dout_.mData[i];
            }
            return 0;
        }

        float Run(const device::BaseArgument* p_arg,
                  const StreamConfig& /* stream_config */ = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg));
        }
    };

    bool IsSupportedArgument(const device::BaseArgument*) override { return true; }

    static auto MakeArgument(const Tensor<DOutDataType>& dout,
                             const Tensor<IndexDataType>& indices,
                             Tensor<DInDataType>& din,
                             ElementwiseOperation elementwise_op)
    {
        return Argument{dout, indices, din, elementwise_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    virtual std::unique_ptr<device::BaseInvoker> MakeInvokerPointer()
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "ReferenceMaxPoolBwd"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
