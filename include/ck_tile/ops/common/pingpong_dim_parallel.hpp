// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck_tile {
enum struct warp_parallelism_type : std::uint16_t
{
    NO_WARP_PARALLELISM = 0,
    M_DIMENSION_PARALLELISM,
    N_DIMENSION_PARALLELISM,
    K_DIMENSION_PARALLELISM
};
} // namespace ck_tile
