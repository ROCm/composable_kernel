// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_permute.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"
#include "ck/library/tensor_operation_instance/gpu/batched_gemm_softmax_gemm_permute/batched_gemm_softmax_gemm_permute.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

template <index_t NumDimG,
          index_t NumDimM,
          index_t NumDimN,
          index_t NumDimK,
          index_t NumDimO,
          typename ADataType,
          typename B0DataType,
          typename B1DataType,
          typename CDataType,
          typename Acc0BiasDataType,
          typename Acc1BiasDataType,
          typename AElementwiseOperation,
          typename B0ElementwiseOperation,
          typename C0DEElementwiseOperation,
          typename B1ElementwiseOperation,
          typename C1DEElementwiseOperation,
          MaskingSpecialization MaskingSpec>
struct DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceBatchedGemmSoftmaxGemmPermute<NumDimG,
                                                                      NumDimM,
                                                                      NumDimN,
                                                                      NumDimK,
                                                                      NumDimO,
                                                                      ADataType,
                                                                      B0DataType,
                                                                      B1DataType,
                                                                      CDataType,
                                                                      Acc0BiasDataType,
                                                                      Acc1BiasDataType,
                                                                      AElementwiseOperation,
                                                                      B0ElementwiseOperation,
                                                                      C0DEElementwiseOperation,
                                                                      B1ElementwiseOperation,
                                                                      C1DEElementwiseOperation,
                                                                      MaskingSpec>>
{
    using DeviceOp = DeviceBatchedGemmSoftmaxGemmPermute<NumDimG,
                                                         NumDimM,
                                                         NumDimN,
                                                         NumDimK,
                                                         NumDimO,
                                                         ADataType,
                                                         B0DataType,
                                                         B1DataType,
                                                         CDataType,
                                                         Acc0BiasDataType,
                                                         Acc1BiasDataType,
                                                         AElementwiseOperation,
                                                         B0ElementwiseOperation,
                                                         C0DEElementwiseOperation,
                                                         B1ElementwiseOperation,
                                                         C1DEElementwiseOperation,
                                                         MaskingSpec>;
    static auto GetInstances()
    {
        std::vector<std::unique_ptr<DeviceOp>> op_ptrs;
        if(ck::get_device_name() == "gfx908")
        {
            DeviceOperationInstanceCreator<ArchitectureEnum::Gfx908,
                                           DeviceOp>::add_device_instances(op_ptrs);
        }
        else if(ck::get_device_name() == "gfx90a")
        {
            DeviceOperationInstanceCreator<ArchitectureEnum::Gfx90a,
                                           DeviceOp>::add_device_instances(op_ptrs);
        }
        else if(ck::get_device_name() == "gfx940")
        {
            DeviceOperationInstanceCreator<ArchitectureEnum::Gfx940,
                                           DeviceOp>::add_device_instances(op_ptrs);
        }
        return op_ptrs;
    }
};
} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
