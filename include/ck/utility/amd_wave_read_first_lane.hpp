// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/functional2.hpp"
#include "ck/utility/math.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ck {
namespace detail {

template <unsigned SizeInBytes>
struct get_carrier;

template <>
struct get_carrier<1>
{
    using type = uint8_t;
};

template <>
struct get_carrier<2>
{
    using type = uint16_t;
};

template <>
struct get_carrier<3>
{
    using type = class carrier
    {
        using value_type = uint32_t;

        std::array<std::byte, 3> bytes;
        static_assert(sizeof(bytes) <= sizeof(value_type));

        public:
        inline carrier(value_type value) noexcept
        {
            auto const from = reinterpret_cast<const std::byte*>(&value);
            std::copy_n(from, bytes.size(), bytes.data());
        }

        // method to trigger template substitution failure
        inline carrier& operator=(const carrier& other) noexcept
        {
            std::copy_n(other.bytes.data(), bytes.size(), bytes.data());
            return *this;
        }

        inline operator value_type() const noexcept
        {
            value_type value{};

            auto const to = reinterpret_cast<std::byte*>(&value);
            std::copy_n(bytes.data(), bytes.size(), to);

            return value;
        }
    };
};
static_assert(sizeof(get_carrier<3>::type) == 3);

template <>
struct get_carrier<4>
{
    using type = uint32_t;
};

template <unsigned Size>
using get_carrier_t = typename get_carrier<Size>::type;

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

    auto* const from_obj = reinterpret_cast<const std::byte*>(&obj);
    alignas(Object) std::byte to_obj[ObjectSize];

    constexpr Size RemainedSize             = ObjectSize % SgprSize;
    constexpr Size CompleteSgprCopyBoundary = ObjectSize - RemainedSize;
    for(Size offset = 0; offset < CompleteSgprCopyBoundary; offset += SgprSize)
    {
        using Sgpr = detail::get_carrier_t<SgprSize>;

        *reinterpret_cast<Sgpr*>(to_obj + offset) =
            amd_wave_read_first_lane(*reinterpret_cast<const Sgpr*>(from_obj + offset));
    }

    if constexpr(0 < RemainedSize)
    {
        using Carrier = detail::get_carrier_t<RemainedSize>;

        *reinterpret_cast<Carrier*>(to_obj + CompleteSgprCopyBoundary) = amd_wave_read_first_lane(
            *reinterpret_cast<const Carrier*>(from_obj + CompleteSgprCopyBoundary));
    }

    /// NOTE: Implicitly start object lifetime. It's better to use std::start_lifetime_at() in this
    /// scenario
    return *reinterpret_cast<Object*>(to_obj);
}

} // namespace ck
