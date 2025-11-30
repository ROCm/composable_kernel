// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/library/tensor_operation_instance/gpu/tile_grouped_conv_instance_factory.hpp"
#include "ck_tile/library/tensor_operation_instance/gpu/tile_grouped_conv_bwd_data_bf16_instances_3.hpp"

namespace ck_tile {
namespace ops {

void add_grouped_conv2d_bwd_data_bf16_instances_3(std::vector<std::unique_ptr<DeviceOp2DBF16>>& instances)
{
    add_device_operation_instances(instances,
                                   tile_grouped_conv_bwd_data_bf16_instances_3<
                                       2,
                                       NHWGC,
                                       GKYXC,
                                       NHWGK>{});
}

} // namespace ops
} // namespace ck_tile
