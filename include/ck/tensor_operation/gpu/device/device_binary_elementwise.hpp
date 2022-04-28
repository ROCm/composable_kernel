#pragma once
#include <iostream>
#include <vector>

#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ElementwiseFunctor>
struct DeviceBinaryElementwise : public BaseOperator
{

    virtual std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                              const void* p_b,
                                                              void* p_c,
                                                              std::vector<int> shape_a,
                                                              std::vector<int> stride_a,
                                                              std::vector<int> shape_b,
                                                              std::vector<int> stride_b,
                                                              ElementwiseFunctor functor,
                                                              index_t threadPerBlock) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
