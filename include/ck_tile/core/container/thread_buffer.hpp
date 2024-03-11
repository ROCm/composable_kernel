// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/container/tuple.hpp"

namespace ck_tile {

#if CK_TILE_THREAD_BUFFER_DEFAULT == CK_TILE_THREAD_BUFFER_USE_TUPLE
template <typename T, index_t N>
using thread_buffer = tuple_array<T, N>;

template <typename... Ts>
CK_TILE_HOST_DEVICE constexpr auto make_thread_buffer(Ts&&... ts)
{
    return make_tuple(ts...);
}
#else
template <typename T, index_t N>
using thread_buffer = array<T, N>;

template <typename... Ts>
CK_TILE_HOST_DEVICE constexpr auto make_thread_buffer(Ts&&... ts)
{
    return make_array(ts...);
}
#endif

} // namespace ck_tile
