// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_abd_xdl_cshuffle.hpp"

#include "device_gemm_xdl_multi_abd_bf16_i8_bf16_mk_kn_mn_common.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using CDEElementOp = AddFastGelu;

void add_device_gemm_xdl_multi_abd_bf16_i8_bf16_mk_kn_mn_bias_gelu_v1_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleABD<AsLayout,
                                                      BsLayout,
                                                      DsLayout,
                                                      ELayout,
                                                      AsDataType,
                                                      BsDataType,
                                                      DsDataType,
                                                      EDataType,
                                                      AElementOp,
                                                      BElementOp,
                                                      CDEElementOp>>>& instances)
{
#if 0
    add_device_operation_instances(
        instances,
        device_gemm_xdl_multi_abd_bf16_i8_bf16_mk_kn_mn_instances<CDEElementOp,
                                                                  GemmDefault,
                                                                  PipelineVersion::v1,
                                                                  LoopScheduler::Default>{});

    add_device_operation_instances(
        instances,
        device_gemm_xdl_multi_abd_bf16_i8_bf16_mk_kn_mn_instances<CDEElementOp,
                                                                  GemmMNPadding,
                                                                  PipelineVersion::v1,
                                                                  LoopScheduler::Default>{});
#endif

    add_device_operation_instances(
        instances,
        device_gemm_xdl_multi_abd_bf16_i8_bf16_mk_kn_mn_instances<CDEElementOp,
                                                                  GemmMNKPadding,
                                                                  PipelineVersion::v1,
                                                                  LoopScheduler::Default>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
