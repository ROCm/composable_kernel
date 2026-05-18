// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"

namespace ck_tile {
// amd_wave_read_first_lane is the SGPR function from AMD GPU device to load 1 or a series of the
// memory to the SGPR registers.
__device__ inline uint32_t amd_wave_read_first_lane(uint16_t v)
{
    return __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(v));
}

__device__ inline uint32_t amd_wave_read_first_lane(uint8_t v)
{
    return __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(v));
}

__device__ inline uint32_t amd_wave_read_first_lane(uint32_t value)
{
    return __builtin_amdgcn_readfirstlane(value);
}

__device__ inline int32_t amd_wave_read_first_lane(int32_t value)
{
    return __builtin_amdgcn_readfirstlane(value);
}

template <typename Object, std::enable_if_t<std::is_trivially_copyable_v<Object>, int> = 0>
__device__ inline auto amd_wave_read_first_lane(const Object& obj)
{
    constexpr size_t ObjectSize = sizeof(Object);
    constexpr size_t SGPR_size  = 4;
    constexpr size_t NumFull    = ObjectSize / SGPR_size;
    constexpr size_t Tail       = ObjectSize % SGPR_size;

    const unsigned char* src = reinterpret_cast<const unsigned char*>(&obj);
    alignas(Object) unsigned char dst[ObjectSize];

    static_for<0, NumFull, 1>{}([&](auto Ic) {
        constexpr size_t offset = Ic * SGPR_size;
        uint32_t read_src;
        __builtin_memcpy(&read_src, src + offset, SGPR_size);
        read_src = __builtin_amdgcn_readfirstlane(read_src);
        __builtin_memcpy(dst + offset, &read_src, SGPR_size);
    });

    if constexpr(Tail != 0)
    {
        constexpr size_t offset = NumFull * SGPR_size;
        uint32_t tail_loc       = 0;
        __builtin_memcpy(&tail_loc, src + offset, Tail);
        tail_loc = __builtin_amdgcn_readfirstlane(tail_loc);
        __builtin_memcpy(dst + offset, &tail_loc, Tail);
    }
    return bit_cast<Object>(dst);
}

// Overload for host to return the same value
template <typename T>
__host__ inline T amd_wave_read_first_lane(T v)
{
    return v;
}

} // namespace ck_tile
