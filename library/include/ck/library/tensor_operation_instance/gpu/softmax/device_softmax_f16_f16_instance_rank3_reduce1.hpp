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

void add_device_softmax_f16_f16_rank3_reduce1_instances(
    std::vector<DeviceSoftmaxPtr<half_t,
                                 float,
                                 half_t,
                                 element_wise::PassThrough,
                                 element_wise::PassThrough,
                                 3>>& instances);

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
