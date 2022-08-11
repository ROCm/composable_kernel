// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <vector>

#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

struct DeviceNormalization : public BaseOperator
{
    // inLengths: input tensor extent(s) from high to low dimension
    // inStrides: input tensor stride(s) from high to low dimension
    // reduceDims: the dimension(s) the normalization operation is applied
    // alpha: typeless pointer in host memory storing the alpha scaling value of type AccDataType
    // beta: typeless pointer in host memory storing the beta scaling value of type AccDataType
    // in_dev: typeless const pointer in device memory storing the input tensor
    // out_dev: typeless pointer in device memory storing the output tensor
    virtual std::unique_ptr<BaseArgument> MakeArgumentPointer(const std::vector<index_t> inLengths,
                                                              const std::vector<index_t> inStrides,
                                                              const std::vector<int> reduceDims,
                                                              const void* alpha,
                                                              const void* beta,
                                                              const void* in_dev,
                                                              void* out_dev) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;

    virtual index_t GetRank() const = 0;

    virtual index_t GetNumReduceDim() const = 0;
};

using DeviceNormalizationPtr = std::unique_ptr<DeviceNormalization>;

template <typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename AccDataType,
          typename YDataType,
          typename AccElementwiseOperation,
          index_t Rank,
          index_t NumReduceDim>
struct DeviceNormalization2 : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::vector<index_t> lengths,
                        const std::vector<index_t> xStrides,
                        const std::vector<index_t> gammaStrides,
                        const std::vector<index_t> betaStrides,
                        const std::vector<index_t> reduceDims,
                        AccDataType epsilon,
                        const void* p_x,
                        const void* p_gamma,
                        const void* p_beta,
                        void* p_y,
                        AccElementwiseOperation acc_elementwise_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename AccDataType,
          typename YDataType,
          typename AccElementwiseOperation,
          index_t Rank,
          index_t NumReduceDim>
using DeviceNormalization2Ptr = std::unique_ptr<DeviceNormalization2<XDataType,
                                                                     GammaDataType,
                                                                     BetaDataType,
                                                                     AccDataType,
                                                                     YDataType,
                                                                     AccElementwiseOperation,
                                                                     Rank,
                                                                     NumReduceDim>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
