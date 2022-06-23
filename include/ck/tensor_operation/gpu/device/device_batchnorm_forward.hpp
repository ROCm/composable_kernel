#ifndef DEVICE_BNORM_FWD_HPP
#define DEVICE_BNORM_FWD_HPP

#include <vector>
#include <memory>
#include <iostream>

#include "common_header.hpp"
#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

struct DeviceBatchNormFwd : public BaseOperator
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
                        void* p_y,
                        double exponentialAverageFactor,
                        void* resultRunningMean,
                        void* resultRunningVariance,
                        double epsilon,
                        void* resultSaveMean,
                        void* resultSaveInvVariance) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

using DeviceBatchNormFwdPtr = std::unique_ptr<DeviceBatchNormFwd>;

} // namespace device
} // namespace tensor_operation
} // namespace ck

#endif
