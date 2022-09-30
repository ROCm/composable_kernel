// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <vector>

#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename InDataTypeTuple,
          typename CDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename AccDataType,
          typename YDataType,
          typename ElementwiseOperation,
          typename AccElementwiseOperation,
          index_t Rank,
          index_t NumReduceDim>
struct DeviceElementwiseLayernorm : public BaseOperator
{
    static constexpr int NumInput = InDataTypeTuple::Size();

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::vector<index_t> lengths,
                        const std::array<std::vector<index_t>, NumInput> inStridesArray,
                        const std::vector<index_t> cStrides,
                        const std::vector<index_t> gammaStrides,
                        const std::vector<index_t> betaStrides,
                        const std::vector<index_t> yStrides,
                        const std::vector<index_t> reduceDims,
                        AccDataType epsilon,
                        const std::array<const void*, NumInput> in_dev_buffers,
                        void* p_c,
                        const void* p_gamma,
                        const void* p_beta,
                        void* p_y,
                        ElementwiseOperation elementwise_op,
                        AccElementwiseOperation acc_elementwise_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename InDataTypeTuple,
          typename CDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename AccDataType,
          typename YDataType,
          typename ElementwiseOperation,
          typename AccElementwiseOperation,
          index_t Rank,
          index_t NumReduceDim>
using DeviceElementwiseLayernormPtr =
    std::unique_ptr<DeviceElementwiseLayernorm<InDataTypeTuple,
                                               CDataType,
                                               GammaDataType,
                                               BetaDataType,
                                               AccDataType,
                                               YDataType,
                                               ElementwiseOperation,
                                               AccElementwiseOperation,
                                               Rank,
                                               NumReduceDim>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
