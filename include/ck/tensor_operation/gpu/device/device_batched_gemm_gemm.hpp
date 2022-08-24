// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// Batched GEMM + GEMM
//   input  : A0[Batch, M, K]
//   input  : B0[Batch, K, N]
//   input  : B1[Batch, N, O]
//   output : C1[Batch, M, O]
//
//   C0 = a0_op(A0) * b0_op(B0)
//   C1 = c0_op(C0) * b1_op(B1)
template <typename A0Layout,
          typename B0Layout,
          typename B1Layout,
          typename C1Layout,
          typename A0DataType,
          typename B0DataType,
          typename B1DataType,
          typename C1DataType,
          typename A0ElementwiseOperation,
          typename B0ElementwiseOperation,
          typename C0ElementwiseOperation,
          typename B1ElementwiseOperation,
          typename C1ElementwiseOperation>
struct DeviceBatchedGemmGemm : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a0,
                        const void* p_b0,
                        const void* p_b1,
                        void* p_c1,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t O,
                        ck::index_t Batch,
                        ck::index_t StrideA0,
                        ck::index_t StrideB0,
                        ck::index_t StrideB1,
                        ck::index_t StrideC1,
                        ck::index_t BatchStrideA0,
                        ck::index_t BatchStrideB0,
                        ck::index_t BatchStrideB1,
                        ck::index_t BatchStrideC1,
                        A0ElementwiseOperation a0_element_op,
                        B0ElementwiseOperation b0_element_op,
                        C0ElementwiseOperation c0_element_op,
                        B1ElementwiseOperation b1_element_op,
                        C1ElementwiseOperation c1_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
