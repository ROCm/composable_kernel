// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {
enum StreamKReductionStrategy : uint32_t
{
    Atomic = 0u,
    Linear = 1u,
    Tree   = 2u
};
} // namespace ck_tile
