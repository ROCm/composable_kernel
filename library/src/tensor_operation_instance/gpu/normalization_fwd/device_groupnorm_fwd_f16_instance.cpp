// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "normalization_fwd_instance_common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using Pass = ck::tensor_operation::element_wise::PassThrough;

void add_device_normalization_fwd_rank_5_3_f16_instances(
    std::vector<std::unique_ptr<DeviceNormalizationFwd<F16, F16, F16, F16, F32, Pass, 5, 3>>>&
        instances)
{
    add_device_operation_instances(instances,
                                   device_normalization_f16_generic_instance<Pass, 5, 3>{});
    add_device_operation_instances(instances, device_normalization_f16_instances<Pass, 5, 3>{});
    add_device_operation_instances(instances,
                                   device_normalization_splitk_f16_instances<Pass, 5, 3>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
