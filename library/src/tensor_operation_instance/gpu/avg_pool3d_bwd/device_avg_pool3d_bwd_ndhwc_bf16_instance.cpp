// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "avg_pool3d_bwd_ndhwc_instance_common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_avgpool_bwd_ndhwc_bf16_instances(
    std::vector<std::unique_ptr<DeviceAvgPoolBwd<3, BF16, BF16, NDHWC, NDHWC>>>& instances)
{
    add_device_operation_instances(instances, device_avgpool_bwd_ndhwc_bf16_instances{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
