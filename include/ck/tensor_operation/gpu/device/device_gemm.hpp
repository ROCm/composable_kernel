// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <vector>

#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

struct GemmShape
{
    ck::index_t M, N, K;
    ck::index_t StrideA, StrideB, StrideC;
};

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct DeviceGemm : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        void* p_c,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t StrideA,
                        ck::index_t StrideB,
                        ck::index_t StrideC,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CElementwiseOperation c_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
using DeviceGemmPtr = std::unique_ptr<DeviceGemm<ALayout,
                                                 BLayout,
                                                 CLayout,
                                                 ADataType,
                                                 BDataType,
                                                 CDataType,
                                                 AElementwiseOperation,
                                                 BElementwiseOperation,
                                                 CElementwiseOperation>>;

template <typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct DeviceGroupedGemm : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument> MakeArgumentPointer(std::vector<const void*>& p_a,
                                                              std::vector<const void*>& p_b,
                                                              std::vector<void*>& p_c,
                                                              std::vector<GemmShape>& gemm_shapes,
                                                              AElementwiseOperation a_element_op,
                                                              BElementwiseOperation b_element_op,
                                                              CElementwiseOperation c_element_op,
                                                              ck::index_t KBatch = 1) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
using DeviceGroupedGemmPtr = std::unique_ptr<
    DeviceGroupedGemm<AElementwiseOperation, BElementwiseOperation, CElementwiseOperation>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
