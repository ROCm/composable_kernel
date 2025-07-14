// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "device_batched_gemm_b_scale_xdl_f16_i4_f16_mk_nk_mn.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
void add_device_batched_gemm_b_scale_xdl_f16_i4_f16_mk_nk_mn_mem_v2_default_instances(
    std::vector<std::unique_ptr<DeviceBatchedGemmV2BScale<Row,
                                                          Col,
                                                          Row,
                                                          F16,
                                                          I4,
                                                          F16,
                                                          F16,
                                                          1,
                                                          128,
                                                          PassThrough,
                                                          PassThrough,
                                                          PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_batched_gemm_b_scale_xdl_f16_i4_f16_mk_nk_mn_mem_instances<Intrawave,
                                                                          GemmDefault>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
