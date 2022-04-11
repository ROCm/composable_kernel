#ifndef DEVICE_BATCHNORM_FWD_HPP
#define DEVICE_BATCHNORM_FWD_HPP

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
    virtual long_index_t GetWorkspaceSizeInBytes(index_t c, bool resultSave)
    {
        (void)c;
        (void)resultSave;

        return (0);
    };

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::vector<index_t> inLengths,
                        const std::vector<index_t> inStrides,
                        const std::vector<index_t> outLengths,
                        const std::vector<index_t> outStrides,
                        const std::vector<index_t> bnScaleBiasMeanVarLengths,
                        const std::vector<index_t> bnScaleBiasMeanVarStrides,
                        float alpha,
                        float beta,
                        const void* in_dev,
                        void* out_dev,
                        void* workspace_dev,
                        const void* bnScale,
                        const void* bnBias,
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
