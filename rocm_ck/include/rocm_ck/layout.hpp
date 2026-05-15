// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Role: types — Layout enum, constexpr/consteval helpers. No runtime, no CK deps.

#pragma once

#include "rocm_ck/platform.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace rocm_ck {

// Auto is a resolve-time placeholder — Signature::resolve() replaces it with
// the concrete layout from the operator slot. It never reaches the kernel.
enum class Layout : uint8_t
{
    Row,
    Col,
    Auto
};

constexpr const char* layoutName(Layout layout)
{
    switch(layout)
    {
    case Layout::Row: return "Row";
    case Layout::Col: return "Col";
    case Layout::Auto: return "Auto";
    }
    ROCM_CK_UNREACHABLE();
}

constexpr bool isValidLayoutForRank(Layout layout, int rank)
{
    switch(layout)
    {
    case Layout::Row: return rank == 2;
    case Layout::Col: return rank == 2;
    case Layout::Auto: return false;
    }
    ROCM_CK_UNREACHABLE();
}

template <typename T, std::size_t N>
constexpr T leadingDimStride(Layout layout, const std::array<T, N>& strides)
{
    switch(layout)
    {
    case Layout::Row: return strides[0];
    case Layout::Col: return strides[1];
    case Layout::Auto: throw "leadingDimStride requires Row or Col layout";
    }
    ROCM_CK_UNREACHABLE();
}

constexpr std::array<int, 2> layoutStrides(Layout layout, int rows, int cols)
{
    switch(layout)
    {
    case Layout::Row: return {cols, 1};
    case Layout::Col: return {1, rows};
    case Layout::Auto: throw "layoutStrides requires Row or Col layout";
    }
    ROCM_CK_UNREACHABLE();
}

} // namespace rocm_ck
