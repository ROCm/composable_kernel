// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/core/numeric/integer.hpp"

namespace ck_tile {

#if CK_TILE_STATICALLY_INDEXED_ARRAY_DEFAULT == CK_TILE_STATICALLY_INDEXED_ARRAY_USE_TUPLE

template <typename T, index_t N>
using statically_indexed_array = tuple_array<T, N>;

#else

// consider mark this struct as deprecated
template <typename T, index_t N>
using statically_indexed_array = array<T, N>;

#endif

// consider always use ck_tile::array for this purpose
} // namespace ck_tile
