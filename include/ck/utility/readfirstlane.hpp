// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/functional2.hpp"
#include "ck/utility/math.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ck {
namespace detail {

template <unsigned Size>
struct get_unsigned_int;

template <>
struct get_unsigned_int<1>
{
    using type = uint8_t;
};

template <>
struct get_unsigned_int<2>
{
    using type = uint16_t;
};

template <>
struct get_unsigned_int<4>
{
    using type = uint32_t;
};

template <unsigned Size>
using get_unsigned_int_t = typename get_unsigned_int<Size>::type;

} // namespace detail

__device__ inline int32_t amd_wave_read_first_lane(int32_t value)
{
    return __builtin_amdgcn_readfirstlane(value);
}

template <
    typename Object,
    typename = std::enable_if_t<std::is_class_v<Object> && std::is_trivially_copyable_v<Object>>>
__device__ auto amd_wave_read_first_lane(const Object& obj)
{
    using Size                = unsigned;
    constexpr Size SgprSize   = 4;
    constexpr Size ObjectSize = sizeof(Object);

    using Sgpr = detail::get_unsigned_int_t<SgprSize>;

    alignas(Object) std::byte to_obj[ObjectSize];

    auto* const from_obj = reinterpret_cast<const std::byte*>(&obj);

    constexpr Size RemainedSize             = ObjectSize % SgprSize;
    constexpr Size CompleteSgprCopyBoundary = ObjectSize - RemainedSize;
    for(Size offset = 0; offset < CompleteSgprCopyBoundary; offset += SgprSize)
    {
        *reinterpret_cast<Sgpr*>(to_obj + offset) =
            amd_wave_read_first_lane(*reinterpret_cast<const Sgpr*>(from_obj + offset));
    }

    if constexpr(0 < RemainedSize)
    {
        using Carrier = detail::get_unsigned_int_t<RemainedSize>;

        *reinterpret_cast<Carrier>(to_obj + CompleteSgprCopyBoundary) =
            amd_wave_read_first_lane(*reinterpret_cast<const Carrier*>(from_obj + CompleteSgprCopyBoundary));
    }

    /// NOTE: Implicitly start object lifetime. It's better to use std::start_lifetime_at() in this
    /// scenario
    return *reinterpret_cast<Object*>(to_obj);
}

} // namespace ck
