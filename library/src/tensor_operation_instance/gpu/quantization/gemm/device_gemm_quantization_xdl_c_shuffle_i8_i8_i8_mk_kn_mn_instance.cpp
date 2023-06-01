// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "device_gemm_quantization_xdl_c_shuffle_i8_i8_i8_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

// Layout(A, B, C) = [Row, Row, Row]
void add_device_gemm_quantization_xdl_c_shuffle_i8_i8_i8_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleD<Row,
                                                    Row,
                                                    Empty_Tuple,
                                                    Row,
                                                    int8_t,
                                                    int8_t,
                                                    Empty_Tuple,
                                                    int8_t,
                                                    PassThrough,
                                                    PassThrough,
                                                    Mul_Clamp>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_quantization_xdl_c_shuffle_i8_i8_i8_mk_kn_mn_instances<Mul_Clamp,
                                                                           LoopScheduler::Default,
                                                                           PipelineVersion::v1>{});
#if CK_EXPERIMENTAL_INTER_WAVE_INSTANCES
    add_device_operation_instances(
        instances,
        device_gemm_quantization_xdl_c_shuffle_i8_i8_i8_mk_kn_mn_instances<Mul_Clamp,
                                                                           LoopScheduler::Interwave,
                                                                           PipelineVersion::v1>{});
#endif
#if CK_EXPERIMENTAL_PIPELINE_V2_INSTANCES
    add_device_operation_instances(
        instances,
        device_gemm_quantization_xdl_c_shuffle_i8_i8_i8_mk_kn_mn_instances<Mul_Clamp,
                                                                           LoopScheduler::Default,
                                                                           PipelineVersion::v2>{});
#endif
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
