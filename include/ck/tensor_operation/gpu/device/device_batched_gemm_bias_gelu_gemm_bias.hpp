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
          typename D0Layout,
          typename B1Layout,
          typename CLayout,
          typename D1sLayout,
          typename ADataType,
          typename B0DataType,
          typename D0DataType,
          typename B1DataType,
          typename CDataType,
          typename D1sDataType,
          typename AElementwiseOperation,
          typename B0ElementwiseOperation,
          typename Acc0ElementwiseOperation,
          typename D0ElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CElementwiseOperation,
          typename D1ElementwiseOperation>
struct DeviceBatchedGemmBiasGeluGemmBias : public BaseOperator
{
    static constexpr index_t NumDTensor = D1sDataType::Size();

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b0,
                        const void* p_d0,
                        const void* p_b1,
                        void* p_c,
                        std::array<const void*, NumDTensor> p_ds,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t O,
                        ck::index_t Batch,
                        ck::index_t StrideA,
                        ck::index_t StrideB0,
                        ck::index_t StrideD0,
                        ck::index_t StrideB1,
                        ck::index_t StrideC,
                        std::array<ck::index_t, NumDTensor> StrideD1s,
                        ck::index_t BatchStrideA,
                        ck::index_t BatchStrideB0,
                        ck::index_t BatchStrideD0,
                        ck::index_t BatchStrideB1,
                        ck::index_t BatchStrideC,
                        std::array<ck::index_t, NumDTensor> BatchStrideD1s,
                        AElementwiseOperation a_element_op,
                        B0ElementwiseOperation b0_element_op,
                        Acc0ElementwiseOperation acc0_element_op,
                        D0ElementwiseOperation d0_element_op,
                        B1ElementwiseOperation b1_element_op,
                        CElementwiseOperation c_element_op,
                        D1ElementwiseOperation d1_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
