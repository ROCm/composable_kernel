// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "pool_fwd_instance_common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

static constexpr auto ReduceOpId = ck::ReduceTensorOp::MAX;

void add_device_pool3d_fwd_ndhwc_f8_instances(
    std::vector<std::unique_ptr<DevicePoolFwd<5, 3, F8, F8, I32, NDHWC, NDHWC, ReduceOpId, false>>>&
        instances)
{
    add_device_operation_instances(
        instances, device_pool3d_fwd_ndhwc_instances<F8, F8, I32, F8, ReduceOpId, false>{});
}

void add_device_pool3d_fwd_ndhwc_index_f8_instances(
    std::vector<std::unique_ptr<DevicePoolFwd<5, 3, F8, F8, I32, NDHWC, NDHWC, ReduceOpId, true>>>&
        instances)
{
    add_device_operation_instances(
        instances, device_pool3d_fwd_ndhwc_instances<F8, F8, I32, F8, ReduceOpId, true>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
