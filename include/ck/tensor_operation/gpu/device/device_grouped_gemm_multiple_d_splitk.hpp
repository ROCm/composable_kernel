// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <array>
#include <iostream>
#include <vector>
#include <sstream>

#include "ck/tensor_operation/gpu/device/device_grouped_gemm_multiple_d.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct DeviceGroupedGemmMultipleDSplitK : public DeviceGroupedGemmMultipleD<ALayout,
                                                                            BLayout,
                                                                            DsLayout,
                                                                            ELayout,
                                                                            ADataType,
                                                                            BDataType,
                                                                            DsDataType,
                                                                            EDataType,
                                                                            AElementwiseOperation,
                                                                            BElementwiseOperation,
                                                                            CDEElementwiseOperation>
{
    //----------------------------------------------------------------------------------------------
    /// @brief      Sets the k batch size.
    ///
    /// @param      p_arg   Pointer to the Argument we're going to change.
    /// @param[in]  kbatch  The kbatch value.
    ///
    virtual void SetKBatchSize(BaseArgument* p_arg, index_t kbatch) const = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
