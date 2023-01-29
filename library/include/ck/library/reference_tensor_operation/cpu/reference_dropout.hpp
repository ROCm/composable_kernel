// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>

#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

namespace ck {
namespace tensor_operation {
namespace host {

template <typename RefDataType, typename InDataType, typename OutDataType>
struct ReferenceDropout : public device::BaseOperator
{
    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(const Tensor<RefDataType>& ref,
                 const Tensor<InDataType>& in,
                 Tensor<OutDataType>& out,
                 RefDataType p_dropout)
            : ref_(ref), in_(in), out_(out), p_dropout_(p_dropout)
        {
        }
        const Tensor<RefDataType>& ref_;
        const Tensor<InDataType>& in_;
        Tensor<OutDataType>& out_;
        RefDataType p_dropout_;
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        float Run(const Argument& arg)
        {
            arg.out_.ForEach([&](auto& self, auto idx) {
                self(idx) = arg.ref_(idx) < arg.p_dropout_ ? arg.in_(idx) : 0;
            });
            return 0;
        }

        float Run(const device::BaseArgument* p_arg,
                  const StreamConfig& /* stream_config */ = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg));
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    bool IsSupportedArgument(const device::BaseArgument*) override { return true; }

    static auto MakeArgument(const Tensor<RefDataType>& ref,
                             const Tensor<InDataType>& in,
                             Tensor<OutDataType>& out,
                             RefDataType p_dropout)
    {
        return Argument{ref, in, out, p_dropout};
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
        str << "ReferenceDropout"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
