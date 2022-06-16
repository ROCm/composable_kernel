#pragma once
#include <iostream>
#include <vector>
#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ElementwiseFunctor>
struct DeviceElementwise : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_input_tuple,
                        void* p_output_tuple,
                        std::vector<index_t> lengths,
                        std::vector<std::vector<index_t>> input_strides,
                        std::vector<std::vector<index_t>> output_strides,
                        ElementwiseFunctor functor) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename ElementwiseFunctor>
using DeviceElementwisePtr = std::unique_ptr<DeviceElementwise<ElementwiseFunctor>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
