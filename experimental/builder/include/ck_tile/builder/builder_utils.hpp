// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/sequence.hpp"
#include "ck_tile/builder/types.hpp"

namespace ck_tile::builder {

// Convert a static array to a sequence
// Usage example:
// static constexpr std::vector arr {1, 2, 3};
// using seq = to_sequence_v<arr>; // seq is ck::Sequence<1, 2, 3>
template <typename T, const T& Arr>
struct to_sequence_t
{
    private:
    template <std::size_t... Is>
    static auto get_sequence_type(std::index_sequence<Is...>) -> ck::Sequence<Arr[Is]...>;

    // Helper method to handler the unusual .Size() method name in ck::Array.
    static constexpr auto get_size(const auto& arr)
    {
        if constexpr(requires { arr.size(); })
        {
            return arr.size();
        }
        else
        {
            return arr.Size();
        }
    }

    public:
    using value = decltype(get_sequence_type(std::make_index_sequence<get_size(Arr)>{}));
};

template <auto& Arr>
using to_sequence_v = typename to_sequence_t<std::remove_cvref_t<decltype(Arr)>, Arr>::value;

// Wrapper function to make constexpr strings a structural type for NTTP.
template <size_t N>
struct StringLiteral
{
    char data[N];
    constexpr StringLiteral(const char (&str)[N])
    {
        for(size_t i = 0; i < N; ++i)
            data[i] = str[i];
    }

    constexpr bool operator==(const StringLiteral<N>& other) const
    {
        for(size_t i = 0; i < N; ++i)
        {
            if(data[i] != other.data[i])
            {
                return false;
            }
        }
        return true;
    }
};

// This is a C++17 deduction guide. It allows the compiler to automatically
// deduce the template argument `N` for `StringLiteral` from a string literal
// constructor argument. For example, you can write `StringLiteral s{"foo"};`
// instead of `StringLiteral<4> s{"foo"};`.
template <size_t N>
StringLiteral(const char (&)[N]) -> StringLiteral<N>;

// Helper to provide a readable error for unsupported enum values.
// The compiler will print the name of this struct in the error message, so
// the name of the enum value will appear instead of just its integer value.
template <auto T>
struct UnsupportedEnumValue
{
};

// Helper functions to convert enums to strings
constexpr std::string_view ConvDirectionToString(ConvDirection dir)
{
    switch(dir)
    {
    case ConvDirection::FORWARD: return "Forward";
    case ConvDirection::BACKWARD_DATA: return "Backward Data";
    case ConvDirection::BACKWARD_WEIGHT: return "Backward Weight";
    default: return "Unknown";
    }
}

constexpr std::string_view DataTypeToString(DataType dt)
{
    switch(dt)
    {
    case DataType::FP16: return "FP16";
    case DataType::FP32: return "FP32";
    case DataType::BF16: return "BF16";
    case DataType::FP8: return "FP8";
    case DataType::I8: return "I8";
    case DataType::U8: return "U8";
    default: return "Unknown";
    }
}

constexpr std::string_view LayoutToString(GroupConvLayout1D layout)
{
    switch(layout)
    {
    case GroupConvLayout1D::GNWC_GKXC_GNWK: return "GNWC_GKXC_GNWK";
    case GroupConvLayout1D::NWGC_GKXC_NWGK: return "NWGC_GKXC_NWGK";
    case GroupConvLayout1D::NGCW_GKXC_NGKW: return "NGCW_GKXC_NGKW";
    case GroupConvLayout1D::NGCW_GKCX_NGKW: return "NGCW_GKCX_NGKW";
    default: return "Unknown";
    }
}

constexpr std::string_view LayoutToString(GroupConvLayout2D layout)
{
    switch(layout)
    {
    case GroupConvLayout2D::GNHWC_GKYXC_GNHWK: return "GNHWC_GKYXC_GNHWK";
    case GroupConvLayout2D::NHWGC_GKYXC_NHWGK: return "NHWGC_GKYXC_NHWGK";
    case GroupConvLayout2D::NGCHW_GKYXC_NGKHW: return "NGCHW_GKYXC_NGKHW";
    case GroupConvLayout2D::NGCHW_GKCYX_NGKHW: return "NGCHW_GKCYX_NGKHW";
    default: return "Unknown";
    }
}

constexpr std::string_view LayoutToString(GroupConvLayout3D layout)
{
    switch(layout)
    {
    case GroupConvLayout3D::GNDHWC_GKZYXC_GNDHWK: return "GNDHWC_GKZYXC_GNDHWK";
    case GroupConvLayout3D::NDHWGC_GKZYXC_NDHWGK: return "NDHWGC_GKZYXC_NDHWGK";
    case GroupConvLayout3D::NGCDHW_GKZYXC_NGKDHW: return "NGCDHW_GKZYXC_NGKDHW";
    case GroupConvLayout3D::NGCDHW_GKCZYX_NGKDHW: return "NGCDHW_GKCZYX_NGKDHW";
    default: return "Unknown";
    }
}

} // namespace ck_tile::builder
