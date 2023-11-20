// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <array>

#include "device_grouped_gemm_multi_abd.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <index_t NumATensor = 1, index_t NumBTensor = 1, index_t NumDTensor = 0>
struct GroupedGemmMultiABDKernelArgument
{
    std::array<const void*, NumATensor> p_as_grid;
    std::array<const void*, NumBTensor> p_bs_grid;
    std::array<const void*, NumDTensor> p_ds_grid;
    void* p_e_grid;

    index_t M;
    index_t N;
    index_t K;

    std::array<index_t, NumATensor> StrideAs;
    std::array<index_t, NumBTensor> StrideBs;
    std::array<index_t, NumDTensor> StrideDs;
    index_t StrideE;
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
struct DeviceGroupedGemmMultiABDFixedNK : DeviceGroupedGemmMultiABD<AsLayout,
                                                                    BsLayout,
                                                                    DsLayout,
                                                                    ELayout,
                                                                    AsDataType,
                                                                    BsDataType,
                                                                    DsDataType,
                                                                    EDataType,
                                                                    AElementwiseOperation,
                                                                    BElementwiseOperation,
                                                                    CElementwiseOperation>
{
    virtual void SetDeviceKernelArgs(BaseArgument* p_arg, const void* kernel_args) const = 0;
    virtual size_t GetDeviceKernelArgSize(const BaseArgument* p_arg) const               = 0;
    virtual void SetKBatch(BaseArgument* p_arg, index_t k_batch) const                   = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
