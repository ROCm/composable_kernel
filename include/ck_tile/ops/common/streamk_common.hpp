// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {
enum StreamKReductionStrategy : uint32_t
{
    Atomic    = 0u,
    Reduction = 1u
};
} // namespace ck_tile
