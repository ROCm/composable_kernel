#ifndef DEVICE_BATCHNORM_BACKWARD_HPP
#define DEVICE_BATCHNORM_BACKWARD_HPP

#include <vector>
#include <memory>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

struct DeviceBatchNormBwd : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::vector<ck::index_t> xyLengths,
                        const std::vector<ck::index_t> xStrides,
                        const std::vector<ck::index_t> dyStrides,
                        const std::vector<ck::index_t> dxStrides,
                        const std::vector<ck::index_t> bnScaleBiasDiffLengths,
                        const std::vector<ck::index_t> bnScaleBiasDiffStrides,
                        const void* p_x,
                        const void* p_dy,
                        const void* bnScale,
                        const void* savedMean,
                        const void* savedInvVariance,
                        double epsilon,
                        void* p_dx,
                        void* resultBnScaleDiff,
                        void* resultBnBiasDiff) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

using DeviceBatchNormBwdPtr = std::unique_ptr<DeviceBatchNormBwd>;

} // namespace device
} // namespace tensor_operation
} // namespace ck

#endif
