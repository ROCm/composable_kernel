// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>

#include "ck/tensor_operation/gpu/device/device_softmax.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_softmax_f32_f32_rank3_reduce3_instances(
    std::vector<DeviceSoftmaxPtr<float,
                                 float,
                                 float,
                                 element_wise::PassThrough,
                                 element_wise::PassThrough,
                                 3>>& instances);

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
