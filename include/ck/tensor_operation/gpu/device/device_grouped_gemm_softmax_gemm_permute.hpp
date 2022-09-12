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
          typename CPermuteNumDims_G_M_Gemm1N, // Sequence<>
          typename ADataType,
          typename B0DataType,
          typename B1DataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename B0ElementwiseOperation,
          typename Acc0ElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CElementwiseOperation>
struct DeviceGroupedGemmSoftmaxGemmPermute : public BaseOperator
{
    struct ProblemDesc
    {
        // Overall problem shape
        index_t M;
        index_t N;
        index_t K;
        index_t O;
        index_t Batch;

        // Stride for A/B0/B1; layout determined by template args
        index_t StrideA;
        index_t StrideB0;
        index_t StrideB1;
        index_t BatchStrideA;
        index_t BatchStrideB0;
        index_t BatchStrideB1;

        // Lengths and strides for output C
        std::vector<index_t> c_gs_ms_os_lengths;
        std::vector<index_t> c_gs_ms_os_strides;
    };

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::vector<const void*> p_a_vec,
                        std::vector<const void*> p_b0_vec,
                        std::vector<const void*> p_b1_vec,
                        std::vector<void*> p_c_vec,
                        std::vector<ProblemDesc> problem_desc_vec,
                        AElementwiseOperation a_element_op,
                        B0ElementwiseOperation b0_element_op,
                        Acc0ElementwiseOperation acc0_element_op,
                        B1ElementwiseOperation b1_element_op,
                        CElementwiseOperation c_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
