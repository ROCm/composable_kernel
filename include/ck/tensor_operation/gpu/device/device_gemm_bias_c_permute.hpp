// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <array>

#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

struct GemmBiasCPermuteDesc
{
    ck::index_t M0_, M1_, M2, N0_, N1_;
    ck::index_t stride_M0_, stride_M1_, stride_M2_, stride_N0_, stride_N1_;
};

// input : A[M, K], B[K, N],
// input : D[M, N], ...
// output : E[M, N]
// C = a_op(A) * b_op(B)
// E = cde_op(C, D)
template <typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct DeviceGemmBiasCPermute : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        const void* p_d,
                        void* p_e,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t StrideA,
                        ck::index_t StrideB,
                        ck::index_t StrideD,
                        ck::index_t StrideE,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
using DeviceGemmBiasCPermutePtr = std::unique_ptr<
    DeviceGemmBiasCPermute<AElementwiseOperation, BElementwiseOperation, CElementwiseOperation>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
