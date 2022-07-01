// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {

enum struct ConvolutionBackwardWeightSpecialization
{
    Default,
    Filter1x1Stride1Pad0,
    Filter1x1Pad0,
    OddC,
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
