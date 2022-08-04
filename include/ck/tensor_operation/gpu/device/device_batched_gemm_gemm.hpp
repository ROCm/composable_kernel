// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <vector>

#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ALayout,
          typename B0Layout,
          typename B1Layout,
          typename CLayout,
          typename ADataType,
          typename B0DataType,
          typename B1DataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename B0ElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CElementwiseOperation>
struct DeviceBatchedGemmGemm : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b0,
                        const void* p_b1,
                        void* p_c,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t O,
                        ck::index_t Batch,
                        ck::index_t StrideA,
                        ck::index_t StrideB0,
                        ck::index_t StrideB1,
                        ck::index_t StrideC,
                        ck::index_t BatchStrideA,
                        ck::index_t BatchStrideB0,
                        ck::index_t BatchStrideB1,
                        ck::index_t BatchStrideC,
                        AElementwiseOperation a_element_op,
                        B0ElementwiseOperation b0_element_op,
                        B1ElementwiseOperation b1_element_op,
                        CElementwiseOperation c_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename ALayout,
          typename B0Layout,
          typename B1Layout,
          typename CLayout,
          typename ADataType,
          typename B0DataType,
          typename B1DataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename B0ElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CElementwiseOperation>
using DeviceBatchedGemmPtr = std::unique_ptr<DeviceBatchedGemmGemm<ALayout,
                                                                   B0Layout,
                                                                   B1Layout,
                                                                   CLayout,
                                                                   ADataType,
                                                                   B0DataType,
                                                                   B1DataType,
                                                                   CDataType,
                                                                   AElementwiseOperation,
                                                                   B0ElementwiseOperation,
                                                                   B1ElementwiseOperation,
                                                                   CElementwiseOperation>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
