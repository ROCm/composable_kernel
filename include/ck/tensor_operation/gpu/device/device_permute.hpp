// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <array>
#include <cmath>
#include <memory>
#include <type_traits>

#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <index_t NumDim, typename InDataType, typename OutDataType, typename ElementwiseOperation>
struct DevicePermute : BaseOperator
{
    using Lengths = std::array<index_t, NumDim>;
    using Strides = Lengths;

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const Lengths inLengths,
                        const Strides inStrides,
                        const Lengths outLengths,
                        const Strides outStrides,
                        const void* in_dev_buffer,
                        void* out_dev_buffer,
                        ElementwiseOperation elementwise_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <index_t NumDim,
          typename InDataType,
          typename OutDataType,
          typename ElementwiseOperation,
          typename DerivedDeviceOperator>
struct DevicePermuteCRTP : DevicePermute<NumDim, InDataType, OutDataType, ElementwiseOperation>
{
    private:
    using BaseType = DevicePermute<NumDim, InDataType, OutDataType, ElementwiseOperation>;

    public:
    // override methods inherited from 'BaseOperator'
    bool IsSupportedArgument(const BaseArgument* arg) override final
    {
        const auto* const argument =
            dynamic_cast<const typename DerivedDeviceOperator::Argument*>(arg);
        if(!argument)
        {
            return false;
        }

        return DerivedDeviceOperator::IsSupportedArgument(*argument);
    }

    // override methods inherited from 'DevicePermute'
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const typename BaseType::Lengths inLengths,
                        const typename BaseType::Strides inStrides,
                        const typename BaseType::Lengths outLengths,
                        const typename BaseType::Strides outStrides,
                        const void* in_dev_buffer,
                        void* out_dev_buffer,
                        ElementwiseOperation elementwise_op) override final
    {
        return std::make_unique<typename DerivedDeviceOperator::Argument>(inLengths,
                                                                          inStrides,
                                                                          outLengths,
                                                                          outStrides,
                                                                          in_dev_buffer,
                                                                          out_dev_buffer,
                                                                          elementwise_op);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override final
    {
        return std::make_unique<typename DerivedDeviceOperator::Invoker>();
    };

    // generate other utility methods
    template <typename... Args>
    static auto MakeArgument(Args&&... args)
    {
        static_assert(std::is_constructible_v<typename DerivedDeviceOperator::Argument, Args...>);

        return typename DerivedDeviceOperator::Argument{std::forward<Args>(args)...};
    }

    static auto MakeInvoker() noexcept(
        std::is_nothrow_default_constructible_v<typename DerivedDeviceOperator::Invoker>)
    {
        static_assert(std::is_default_constructible_v<typename DerivedDeviceOperator::Invoker>);

        return typename DerivedDeviceOperator::Invoker{};
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
