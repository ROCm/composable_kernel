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
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        void* p_c,
                        const std::vector<int>& shape_a,
                        const std::vector<int>& stride_a,
                        const std::vector<int>& shape_b,
                        const std::vector<int>& stride_b,
                        ElementwiseFunctor functor) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
