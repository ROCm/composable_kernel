// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include <iostream>
#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename DxsInElementwiseOperation,
          typename DxsReduceAccElementwiseOperation>
struct DeviceBatchedGemmReduce : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        void* p_c,
                        void* p_dxs,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t StrideA,
                        ck::index_t StrideB,
                        ck::index_t StrideC,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CElementwiseOperation c_element_op,
                        DxsInElementwiseOperation dxs_in_element_op,
                        DxsReduceAccElementwiseOperation dxs_out_element_op,
                        ck::index_t Batch) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename DxsInElementwiseOperation,
          typename DxsReduceAccElementwiseOperation>
using DeviceBatchedGemmReducePtr =
    std::unique_ptr<DeviceBatchedGemmReduce<AElementwiseOperation,
                                            BElementwiseOperation,
                                            CElementwiseOperation,
                                            DxsInElementwiseOperation,
                                            DxsReduceAccElementwiseOperation>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
