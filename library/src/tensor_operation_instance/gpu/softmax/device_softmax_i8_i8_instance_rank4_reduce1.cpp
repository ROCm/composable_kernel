// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <vector>

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/device_operation_instance_factory.hpp"
#include "ck/library/tensor_operation_instance/gpu/softmax/device_softmax_i8_i8_instance_rank4_reduce1.hpp"
#include "ck/library/tensor_operation_instance/gpu/softmax/device_softmax_i8_i8_instance_type.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_softmax_i8_i8_rank4_reduce1_instances(
    std::vector<DeviceSoftmaxPtr<I8, F32, I8, PassThrough, PassThrough, 4, 1>>& instances)
{
    add_device_operation_instances(instances, device_softmax_i8_i8_instances<4, 1>{});
}

void get_device_softmax_i8_i8_rank4_reduce1_generic_instance(
    DeviceSoftmaxPtr<I8, F32, I8, PassThrough, PassThrough, 4, 1>& instance)
{
    get_first_device_operation_instance(instance, device_softmax_i8_i8_instances<4, 1>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
