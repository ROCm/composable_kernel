#ifndef DEVICE_BASE_HPP
#define DEVICE_BASE_HPP

namespace ck {
namespace tensor_operation {
namespace device {

struct BaseArgument
{
    BaseArgument()                    = default;
    BaseArgument(const BaseArgument&) = default;
    BaseArgument& operator=(const BaseArgument&) = default;

    virtual ~BaseArgument() {}
};

struct BaseInvoker
{
    BaseInvoker()                   = default;
    BaseInvoker(const BaseInvoker&) = default;
    BaseInvoker& operator=(const BaseInvoker&) = default;

    virtual float Run(const BaseArgument*, int = 1) = 0;

    virtual ~BaseInvoker() {}
};

struct BaseOperator
{
    BaseOperator()                    = default;
    BaseOperator(const BaseOperator&) = default;
    BaseOperator& operator=(const BaseOperator&) = default;

    virtual bool IsSupportedArgument(const BaseArgument*) = 0;

    virtual ~BaseOperator() {}
};

struct BaseGpuOperator
{
    BaseGpuOperator()                       = default;
    BaseGpuOperator(const BaseGpuOperator&) = default;
    BaseGpuOperator& operator=(const BaseGpuOperator&) = default;

    virtual ~BaseGpuOperator() {}
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
