// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "device_gemm_multiply_multiply_wp_xdl_bf16_bf16_bf16_mk_nk_mn.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_gemm_multiply_multiply_weight_preshuffle_xdl_bf16_bf16_bf16_mk_mfma_mn_p2_instances(
    std::vector<std::unique_ptr<DeviceGemmMultipleDSplitKBPreShuffle<Row,
                                                                     Col,
                                                                     Tuple<Row, Col>,
                                                                     Row,
                                                                     BF16,
                                                                     BF16,
                                                                     Tuple<F32, F32>,
                                                                     BF16,
                                                                     PassThrough,
                                                                     PassThrough,
                                                                     MultiplyMultiply>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_gemm_multiply_multiply_weight_preshuffle_xdl_bf16_bf16_bf16_mk_mfma_mn_p2_instances<
            v1,
            GemmDefault>{});

    add_device_operation_instances(
        instances,
        device_gemm_multiply_multiply_weight_preshuffle_xdl_bf16_bf16_bf16_mk_mfma_mn_p2_instances<
            v2,
            GemmDefault>{});

    add_device_operation_instances(
        instances,
        device_gemm_multiply_multiply_weight_preshuffle_xdl_bf16_bf16_bf16_mk_mfma_mn_p2_instances<
            v1,
            GemmNPadding>{});

    add_device_operation_instances(
        instances,
        device_gemm_multiply_multiply_weight_preshuffle_xdl_bf16_bf16_bf16_mk_mfma_mn_p2_instances<
            v2,
            GemmNPadding>{});

    add_device_operation_instances(
        instances,
        device_gemm_multiply_multiply_weight_preshuffle_xdl_bf16_bf16_bf16_mk_mfma_mn_p2_instances<
            v1,
            GemmKPadding>{});

    add_device_operation_instances(
        instances,
        device_gemm_multiply_multiply_weight_preshuffle_xdl_bf16_bf16_bf16_mk_mfma_mn_p2_instances<
            v2,
            GemmKPadding>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
