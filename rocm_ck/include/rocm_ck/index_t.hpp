// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Role: types - index_t, long_index_t. No runtime, no CK deps.

#pragma once

#include <cstdint>

namespace rocm_ck {

// Matches ck_tile::index_t without pulling in CK Tile headers.
using index_t = std::int32_t;

// batch_stride * nhead can exceed int32. Matches ck_tile::long_index_t.
using long_index_t = std::int64_t;

} // namespace rocm_ck
