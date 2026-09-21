// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_gemm_tile_loop/device_grouped_gemm_tile_loop_multiply_wmma_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using DsDataType   = ck::Tuple<BF16, BF16>;
using DsLayout     = ck::Tuple<Row, Row>;
using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = MultiplyAdd;

void add_device_grouped_gemm_wmma_tile_loop_multiply_bias_bf16_i8_bf16_mk_kn_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmTileLoop<Row,
                                                          Row,
                                                          DsLayout,
                                                          Row,
                                                          BF16,
                                                          I8,
                                                          DsDataType,
                                                          BF16,
                                                          AElementOp,
                                                          BElementOp,
                                                          CDEElementOp>>>& instances)
{

    add_device_grouped_gemm_tile_loop_multiply_wmma_instances<
        BF16,
        I8,
        DsDataType,
        Row,
        Row,
        DsLayout,
        device_grouped_gemm_tile_loop_multiply_wmma_mk_kn_mn_instances>(instances);
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
