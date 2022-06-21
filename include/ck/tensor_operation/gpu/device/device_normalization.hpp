#pragma once
#include <iostream>
#include <vector>

#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

struct DeviceNormalization : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument> MakeArgumentPointer(const std::vector<index_t> inLengths,
                                                              const std::vector<index_t> inStrides,
                                                              const std::vector<int> reduceDims,
                                                              void* alpha,
                                                              void* beta,
                                                              const void* in_dev,
                                                              void* out_dev) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;

    virtual index_t GetRank() const = 0;

    virtual index_t GetNumReduceDim() const = 0;
};

using DeviceNormalizationPtr = std::unique_ptr<DeviceNormalization>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
