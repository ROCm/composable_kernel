// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "device_grouped_gemm_xdl_tile_loop_multiply_bf16_i8_bf16_mk_kn_mn.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_gemm_xdl_tile_loop_multiply_bias_bf16_i8_bf16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmTileLoop<Row,
                                                          Row,
                                                          ck::Tuple<Row, Row>,
                                                          Row,
                                                          BF16,
                                                          I8,
                                                          ck::Tuple<BF16, BF16>,
                                                          BF16,
                                                          PassThrough,
                                                          PassThrough,
                                                          MultiplyAdd>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_grouped_gemm_xdl_tile_loop_bf16_i8_bf16_mk_kn_mn_comp_instances<
            ck::Tuple<Row, Row>,
            ck::Tuple<BF16, BF16>,
            MultiplyAdd>{});
    add_device_operation_instances(
        instances,
        device_grouped_gemm_xdl_tile_loop_bf16_i8_bf16_mk_kn_mn_mem_instances<ck::Tuple<Row, Row>,
                                                                              ck::Tuple<BF16, BF16>,
                                                                              MultiplyAdd>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
