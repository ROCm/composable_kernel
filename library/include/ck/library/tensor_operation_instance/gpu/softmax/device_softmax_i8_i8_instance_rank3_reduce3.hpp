// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>

#include "ck/tensor_operation/gpu/device/device_softmax.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/utility/data_type.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_softmax_i8_i8_rank3_reduce3_instances(
    std::vector<DeviceSoftmaxPtr<int8_t,
                                 float,
                                 int8_t,
                                 element_wise::PassThrough,
                                 element_wise::PassThrough,
                                 3>>& instances);

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
