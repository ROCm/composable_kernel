// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile_profiler/tile_grouped_conv_instance_factory.hpp"
#include "ck_tile_profiler/tile_grouped_conv_bwd_data_bf16_instances_4.hpp"

namespace ck_tile {
namespace ops {

void add_grouped_conv2d_bwd_data_bf16_instances_4(
    std::vector<std::unique_ptr<DeviceOp2DBF16>>& instances)
{
    add_device_operation_instances(
        instances, tile_grouped_conv_bwd_data_bf16_instances_4<2, NHWGC, GKYXC, NHWGK>{});
}

} // namespace ops
} // namespace ck_tile
