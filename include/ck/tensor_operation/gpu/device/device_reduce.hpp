#ifndef DEVICE_REDUCE_HPP
#define DEVICE_REDUCE_HPP

#include <vector>
#include <memory>
#include <iostream>

#include "common_header.hpp"
#include "device_base.hpp"
#include "reduction_enums.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename InElementwiseOperation, typename AccElementwiseOperation>
struct DeviceReduce : public BaseOperator
{
    virtual long_index_t GetWorkspaceSizeInBytes(const std::vector<int> inLengths,
                                                 const std::vector<int> reduceDims)
    {
        (void)inLengths;
        (void)reduceDims;

        return (0);
    };

    virtual bool HasFurtherCall() { return (false); };

    virtual std::vector<int> GetWorkspace2dLengths(const BaseArgument* argPtr)
    {
        (void)argPtr;
        return (std::vector<int>{0, 0});
    };

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::vector<int> inLengths,
                        const std::vector<int> inStrides,
                        const std::vector<int> outLengths,
                        const std::vector<int> outStrides,
                        const std::vector<int> reduceDims,
                        float alpha,
                        float beta,
                        const void* in_dev,
                        void* out_dev,
                        void* out_indices_dev,
                        void* workspace_dev,
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
#endif
