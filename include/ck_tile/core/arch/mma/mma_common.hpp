// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "../config.hpp"

namespace ck_tile::core::arch {
// This is a meta-tag that will indicate whether an instruction is supported
// TODO: Should we use class NoneSuch for this purpose?

// Helper function to convert from fragment vectors to native vector types for built-ins, (if
// required!)
template <typename T>
CK_TILE_DEVICE inline auto to_native_vector(T const& vec) -> T const&
{
    return vec;
}

} // namespace ck_tile::core::arch
