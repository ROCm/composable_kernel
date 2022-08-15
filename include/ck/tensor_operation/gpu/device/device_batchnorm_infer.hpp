// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <array>
#include <memory>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <index_t Rank, index_t NumBatchNormReduceDim>
struct DeviceBatchNormInfer : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const std::array<index_t, Rank> xyLengths,
        const std::array<index_t, Rank> xStrides,
        const std::array<index_t, Rank> yStrides,
        const std::array<index_t, Rank - NumBatchNormReduceDim> bnScaleBiasMeanVarLengths,
        const std::array<index_t, Rank - NumBatchNormReduceDim> bnScaleBiasMeanVarStrides,
        const void* p_x,
        const void* bnScale,
        const void* bnBias,
        double epsilon,
        const void* estimatedMean,
        const void* estimatedInvVariance,
        void* p_y) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <index_t Rank, index_t NumBatchNormReduceDim>
using DeviceBatchNormInferPtr = std::unique_ptr<DeviceBatchNormInfer<Rank, NumBatchNormReduceDim>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
