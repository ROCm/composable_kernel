// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Role: types - FixedString. No runtime, no CK deps.
//
// A compile-time string for use in template parameters (NTTPs).
//
// C++20 requires template parameters to be "structural types" - loosely, types
// that are trivially comparable and don't contain pointers or references.
// std::string and std::string_view fail this requirement (internal pointer).
//
// FixedString stores the string inline in a char array, making it structural:
//
//   template <PhysicalTensor PT>  // PhysicalTensor contains FixedString<16>
//   void dispatch() { ... }
//
// When to use FixedString vs std::string_view:
//   - FixedString: the type must be structural (template parameters).
//   - string_view: consteval-only types that never become template parameters
//     (e.g., ResolvedTensor - see resolved_tensor.hpp).
//
// The capacity is a template parameter so each use site documents its limit:
//   FixedString<16> name("bias");   // tensor names: 15 chars max

#pragma once

#include <cstddef>
#include <string_view>

namespace rocm_ck {

template <std::size_t MaxLen>
struct FixedString
{
    char data[MaxLen]{};
    int len = 0;

    constexpr FixedString() = default;

    constexpr FixedString(std::string_view sv) : len(static_cast<int>(sv.size()))
    {
        if(sv.size() > MaxLen - 1)
            throw "FixedString: input exceeds capacity";
        for(int i = 0; i < len; ++i)
            data[i] = sv[i];
    }

    constexpr bool operator==(std::string_view sv) const
    {
        if(len != static_cast<int>(sv.size()))
            return false;
        for(int i = 0; i < len; ++i)
            if(data[i] != sv[i])
                return false;
        return true;
    }

    // Required: the string_view overload above suppresses the implicit == from <=>.
    constexpr bool operator==(const FixedString&) const  = default;
    constexpr auto operator<=>(const FixedString&) const = default;
};

} // namespace rocm_ck
