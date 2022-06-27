#ifndef DEVICE_BATCHNORM_INFER_HPP
#define DEVICE_BATCHNORM_INFER_HPP

#include <vector>
#include <memory>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

struct DeviceBatchNormInfer : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::vector<index_t> xyLengths,
                        const std::vector<index_t> xStrides,
                        const std::vector<index_t> yStrides,
                        const std::vector<index_t> bnScaleBiasMeanVarLengths,
                        const std::vector<index_t> bnScaleBiasMeanVarStrides,
                        const void* p_x,
                        const void* bnScale,
                        const void* bnBias,
                        double epsilon,
                        const void* estimatedMean,
                        const void* estimatedInvVariance,
                        void* p_y) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

using DeviceBatchNormInferPtr = std::unique_ptr<DeviceBatchNormInfer>;

} // namespace device
} // namespace tensor_operation
} // namespace ck

#endif
