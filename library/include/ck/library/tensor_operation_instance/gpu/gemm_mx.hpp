// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>
#include <memory>
#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3_mx.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

template <typename ADataType,
          typename AScaleDataType,
          typename BDataType,
          typename BScaleDataType,
          typename CDataType,
          index_t ScaleBlockSize,
          typename ALayout,
          typename BLayout,
          typename CLayout>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGemmMX<ALayout,
                                               BLayout,
                                               CLayout,
                                               ADataType,
                                               AScaleDataType,
                                               BDataType,
                                               BScaleDataType,
                                               CDataType,
                                               ScaleBlockSize,
                                               ck::tensor_operation::element_wise::PassThrough,
                                               ck::tensor_operation::element_wise::PassThrough,
                                               ck::tensor_operation::element_wise::PassThrough>>
{
    using DeviceOp = DeviceGemmMX<ALayout,
                                  BLayout,
                                  CLayout,
                                  ADataType,
                                  AScaleDataType,
                                  BDataType,
                                  BScaleDataType,
                                  CDataType,
                                  ScaleBlockSize,
                                  ck::tensor_operation::element_wise::PassThrough,
                                  ck::tensor_operation::element_wise::PassThrough,
                                  ck::tensor_operation::element_wise::PassThrough>;

    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;

        // TODO: Add instances for all the combinations of ADataType, BDataType, CDataType, etc.
        return op_ptrs;
    }
};

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
