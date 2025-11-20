// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "multi_reduce2d_kernel.hpp"
namespace ck_tile {
template <typename Problem_, typename Policy_ = Reduce2dDefaultPolicy>
using MultiReduceMultiblock = MultiReduce2d<Problem_, Policy_, true>;

} // namespace ck_tile
