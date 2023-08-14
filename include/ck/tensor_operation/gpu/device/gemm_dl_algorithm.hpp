// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {

enum struct GemmDlAlgorithm
{
    Default, // Uses DOT vector instructions
    Dpp8,    // Uses DOT vector instructions with DPP8 SEL modifier to reduce data loads from LDS
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
