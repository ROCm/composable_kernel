// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "../config.hpp"

namespace ck_tile::core::arch
{
    // Supporting utilities for SFINAE
    // TODO: Move to core utilities?
    template <bool Cond, typename TrueT, typename FalseT>
    struct conditional
    {
        using type = TrueT;
    };

    template <typename TrueT, typename FalseT>
    struct conditional<false, TrueT, FalseT>
    {
        using type = FalseT;
    };

    template <bool Cond, typename TrueT, typename FalseT>
    using conditional_t = typename conditional<Cond, TrueT, FalseT>::type;

    // Utility to check if a value is contained in a list of values at compile time
    template <typename T, T Val, T... Vals>
    struct contains_value : public conditional_t<((Val == Vals) || ...), true_type, false_type>
    {
        static_assert(sizeof...(Vals) >= 1u, "Value list must be >= 1");
    };

    template <typename T, T Val, T... Vals>
    static constexpr bool contains_value_v = contains_value<T, Val, Vals...>::value;

    template <bool B, typename T = void>
    struct enable_if
    {
    };

    template <typename T>
    struct enable_if<true, T>
    {
        using type = T;
    };

    template <bool B, typename T = void>
    using enable_if_t = typename enable_if<B, T>::type;

    // Enabler for targets.
    // Given a TargetId, enable if it exists in the TargetIds list
    template <uint32_t TargetId, uint32_t... TargetIds>
    using enable_target_id_t = enable_if_t<contains_number_v<uint32_t, TargetId, TargetIds...>>;


    // This is a meta-tag that will indicate whether an instruction is supported
    // TODO: Should we use class NoneSuch for this purpose?
    struct Unsupported;

    // Helper function to convert from fragment vectors to native vector types for built-ins, (if required!)
    template<typename T>
    CK_TILE_DEVICE inline auto to_native_vector(T const& vec) -> T const&
    {
        return vec;
    }
    
} // namespace ck_tile::core::arch
