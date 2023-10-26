// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "max_pool_bwd_instance_common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_maxpool_bwd_f16_instances(
    std::vector<std::unique_ptr<DeviceMaxPoolBwd<F16, I32, F16>>>& instances)
{
    add_device_operation_instances(instances, device_maxpool_bwd_instances<F16, I32, F16>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
