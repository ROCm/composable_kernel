// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

namespace ck_tile {
namespace ops {

using NHWGC  = ck_tile::tensor_layout::convolution::NHWGC;
using GKYXC  = ck_tile::tensor_layout::convolution::GKYXC;
using NHWGK  = ck_tile::tensor_layout::convolution::NHWGK;

using PassThrough = ck_tile::element_wise::PassThrough;

} // namespace ops
} // namespace ck_tile
