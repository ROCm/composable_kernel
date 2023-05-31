// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <vector>

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"
#include "ck/library/tensor_operation_instance/gpu/softmax/device_softmax_f32_f32_instance_rank4_reduce2.hpp"
#include "ck/library/tensor_operation_instance/gpu/softmax/device_softmax_f32_f32_instance_type.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_softmax_f32_f32_rank4_reduce2_instances(
    std::vector<DeviceSoftmaxPtr<F32, F32, F32, PassThrough, PassThrough, 4, 2>>& instances)
{
    add_device_operation_instances(instances, device_softmax_f32_f32_instances<4, 2>{});
}

void get_device_softmax_f32_f32_rank4_reduce2_generic_instance(
    DeviceSoftmaxPtr<F32, F32, F32, PassThrough, PassThrough, 4, 2>& instance)
{
    get_first_device_operation_instance(instance, device_softmax_f32_f32_instances<4, 2>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
