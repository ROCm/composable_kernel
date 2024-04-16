// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_abd_xdl_cshuffle.hpp"

#include "device_gemm_xdl_multi_abd_bf16_i8_bf16_km_kn_mn_common.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_xdl_multi_abd_bf16_i8_bf16_km_kn_mn_bias_gelu_v1_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleABD<AsLayout,
                                                      BsLayout,
                                                      ck::Tuple<D0Layout>,
                                                      ELayout,
                                                      AsDataType,
                                                      BsDataType,
                                                      ck::Tuple<D0DataType>,
                                                      EDataType,
                                                      AElementOp,
                                                      BElementOp,
                                                      AddFastGelu>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_xdl_multi_abd_bf16_i8_bf16_km_kn_mn_instances<ck::Tuple<D0Layout>,
                                                                  ck::Tuple<D0DataType>,
                                                                  AddFastGelu,
                                                                  GemmMNKPadding,
                                                                  PipelineVersion::v1,
                                                                  LoopScheduler::Default>{});
}

void add_device_gemm_xdl_multi_abd_bf16_i8_bf16_km_kn_mn_bias_v1_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleABD<AsLayout,
                                                      BsLayout,
                                                      ck::Tuple<D0Layout>,
                                                      ELayout,
                                                      AsDataType,
                                                      BsDataType,
                                                      ck::Tuple<D0DataType>,
                                                      EDataType,
                                                      AElementOp,
                                                      BElementOp,
                                                      Add>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_xdl_multi_abd_bf16_i8_bf16_km_kn_mn_instances<ck::Tuple<D0Layout>,
                                                                  ck::Tuple<D0DataType>,
                                                                  Add,
                                                                  GemmMNKPadding,
                                                                  PipelineVersion::v1,
                                                                  LoopScheduler::Default>{});
}

void add_device_gemm_xdl_multi_abd_bf16_i8_bf16_km_kn_mn_v1_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleABD<AsLayout,
                                                      BsLayout,
                                                      ck::Tuple<>,
                                                      ELayout,
                                                      AsDataType,
                                                      BsDataType,
                                                      ck::Tuple<>,
                                                      EDataType,
                                                      AElementOp,
                                                      BElementOp,
                                                      PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_xdl_multi_abd_bf16_i8_bf16_km_kn_mn_instances<ck::Tuple<>,
                                                                  ck::Tuple<>,
                                                                  PassThrough,
                                                                  GemmMNKPadding,
                                                                  PipelineVersion::v1,
                                                                  LoopScheduler::Default>{});
}

void add_device_gemm_xdl_multi_abd_bf16_i8_bf16_km_kn_mn_gelu_v1_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleABD<AsLayout,
                                                      BsLayout,
                                                      ck::Tuple<>,
                                                      ELayout,
                                                      AsDataType,
                                                      BsDataType,
                                                      ck::Tuple<>,
                                                      EDataType,
                                                      AElementOp,
                                                      BElementOp,
                                                      FastGelu>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_xdl_multi_abd_bf16_i8_bf16_km_kn_mn_instances<ck::Tuple<>,
                                                                  ck::Tuple<>,
                                                                  FastGelu,
                                                                  GemmMNKPadding,
                                                                  PipelineVersion::v1,
                                                                  LoopScheduler::Default>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
