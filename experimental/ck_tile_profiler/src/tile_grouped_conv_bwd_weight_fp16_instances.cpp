// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile_profiler/tile_grouped_conv_instance_factory.hpp"
#include "ck_tile_profiler/tile_grouped_conv_bwd_weight_fp16_instances.hpp"

namespace ck_tile {
namespace ops {

void add_grouped_conv2d_bwd_weight_f16_instances(
    std::vector<std::unique_ptr<DeviceOp2DF16>>& instances)
{
    add_device_operation_instances(
        instances, tile_grouped_conv_bwd_weight_f16_instances<2, NHWGC, GKYXC, NHWGK>{});
}

} // namespace ops
} // namespace ck_tile
