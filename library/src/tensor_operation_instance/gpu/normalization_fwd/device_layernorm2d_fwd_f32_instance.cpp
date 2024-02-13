// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "normalization_fwd_instance_common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using Pass = ck::tensor_operation::element_wise::PassThrough;

void add_device_normalization_fwd_rank_2_1_f32_instances(
    std::vector<std::unique_ptr<DeviceNormalizationFwd<F32, F32, F32, F32, F32, Pass, 2, 1>>>&
        instances)
{
    add_device_operation_instances(instances,
                                   device_normalization_f32_generic_instance<Pass, 2, 1>{});
    add_device_operation_instances(instances, device_normalization_f32_instances<Pass, 2, 1>{});
    add_device_operation_instances(instances,
                                   device_normalization_splitk_f32_instances<Pass, 2, 1>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
