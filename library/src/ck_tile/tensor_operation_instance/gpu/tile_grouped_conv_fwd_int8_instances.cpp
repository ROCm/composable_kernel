// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/library/tensor_operation_instance/gpu/tile_grouped_conv_instance_factory.hpp"
#include "ck_tile/library/tensor_operation_instance/gpu/tile_grouped_conv_fwd_int8_instances.hpp"

namespace ck_tile {
namespace ops {

void add_grouped_conv2d_fwd_int8_instances(std::vector<std::unique_ptr<DeviceOpFwd2DINT8>>& instances)
{
    add_device_operation_instances(instances,
                                   tile_grouped_conv_fwd_int8_instances<
                                       2,
                                       NHWGC,
                                       GKYXC,
                                       NHWGK>{});
}

} // namespace ops
} // namespace ck_tile
