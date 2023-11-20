// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <vector>

#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

struct GemmMultiABDDesc
{
    ck::index_t M_, N_, K_;

    std::vector<ck::index_t> stride_As_;
    std::vector<ck::index_t> stride_Bs_;
    std::vector<ck::index_t> stride_Ds_;

    ck::index_t stride_C_;
};

template <typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename ELayout,
          typename AsDataType,
          typename BsDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct DeviceGroupedGemmMultiABD : public BaseOperator
{
    static constexpr index_t NumATensor = AsDataType::Size();
    static constexpr index_t NumBTensor = BsDataType::Size();
    static constexpr index_t NumDTensor = DsDataType::Size();

    static_assert(AsLayout::Size() == AsDataType::Size(), "wrong! inconsistent NumATensor");
    static_assert(BsLayout::Size() == BsDataType::Size(), "wrong! inconsistent NumBTensor");
    static_assert(DsLayout::Size() == DsDataType::Size(), "wrong! inconsistent NumDTensor");

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::vector<std::array<const void*, NumATensor>>& p_as,
                        std::vector<std::array<const void*, NumBTensor>>& p_bs,
                        std::vector<std::array<const void*, NumDTensor>>& p_ds,
                        std::vector<void*>& p_e,
                        std::vector<GemmMultiABDDesc>& gemm_desc,
                        AElementwiseOperation a_element_op = AElementwiseOperation{},
                        BElementwiseOperation b_element_op = BElementwiseOperation{},
                        CElementwiseOperation c_element_op = CElementwiseOperation{}) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;

    virtual void SetElementwiseOps(BaseArgument* p_arg,
                                   AElementwiseOperation a_element_op,
                                   BElementwiseOperation b_element_op,
                                   CElementwiseOperation cde_element_op) const = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
