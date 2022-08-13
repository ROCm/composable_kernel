// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>

#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// FIXME: DeviceGemmReduce type need to well define the problem
template <typename ALayout,
          typename BLayout,
          typename DELayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename RsDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename QsElementwiseOperation,
          typename RsElementwiseOperation>
struct DeviceGemmMultipleDMultipleR : public BaseOperator
{
    static constexpr index_t NumDTensor = DsDataType::Size();
    static constexpr index_t NumRTensor = RsDataType::Size();

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        std::array<const void*, NumDTensor> p_ds,
                        void* p_e,
                        std::array<void*, NumRTensor> p_rs,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t StrideA,
                        ck::index_t StrideB,
                        std::array<ck::index_t, NumDTensor> StrideDs,
                        ck::index_t StrideE,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op,
                        QsElementwiseOperation qs_element_op,
                        RsElementwiseOperation rs_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename ALayout,
          typename BLayout,
          typename DELayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename RsDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename QsElementwiseOperation,
          typename RsElementwiseOperation>
using DeviceGemmMultipleDMultipleRPtr =
    std::unique_ptr<DeviceGemmMultipleDMultipleR<ALayout,
                                                 BLayout,
                                                 DELayout,
                                                 ADataType,
                                                 BDataType,
                                                 DsDataType,
                                                 EDataType,
                                                 RsDataType,
                                                 AElementwiseOperation,
                                                 BElementwiseOperation,
                                                 CDEElementwiseOperation,
                                                 QsElementwiseOperation,
                                                 RsElementwiseOperation>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
