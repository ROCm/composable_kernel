// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/container/tuple_array.hpp"

namespace ck_tile {

#if CK_TILE_RBUFFER_DEFAULT == CK_TILE_RBUFFER_USE_TUPLE
template <typename T, index_t N>
using rbuffer = tuple_array<T, N>;
#else
template <typename T, index_t N>
using rbuffer = array<T, N>
#endif

} // namespace ck_tile
