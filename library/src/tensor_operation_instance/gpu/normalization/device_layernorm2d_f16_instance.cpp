// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "normalization_instance_common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using Pass = ck::tensor_operation::element_wise::PassThrough;

void add_device_normalization_rank_2_1_f16_instances(
    std::vector<std::unique_ptr<DeviceNormalization<F16, F16, F16, F32, F16, Pass, 2, 1>>>&
        instances)
{
    add_device_operation_instances(instances, device_normalization_f16_instances<Pass, 2, 1>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
