// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/pk_f6.hpp"
#include "ck_tile/core/numeric/pk_fp4.hpp"

namespace ck_tile::core::arch::mma {
namespace scale::detail {

// Utility for converting the datatype of the A or B input matrix in a scale intrinsics to the
// appropriate datatype flag. Note that this is not the same as the flag indicating the scale
// datatype, see ScaleDataTypeToEnum.
template <typename T>
inline constexpr int32_t ScaleDataTypeToFlag_v = [] {
    // sizeof(T) trick to only trigger the static assert for unsupported datatypes.
    static_assert(sizeof(T) == 0, "Unsupported scale data type");
    return -1;
}();
template <>
inline constexpr int32_t ScaleDataTypeToFlag_v<fp8_t> = 0; // e4m3
template <>
inline constexpr int32_t ScaleDataTypeToFlag_v<bf8_t> = 1; // e5m2
template <>
inline constexpr int32_t ScaleDataTypeToFlag_v<pk_fp6x16_t> = 2; // e2m3
template <>
inline constexpr int32_t ScaleDataTypeToFlag_v<pk_bf6x16_t> = 3; // e3m2
template <>
inline constexpr int32_t ScaleDataTypeToFlag_v<pk_fp4_t> = 4; // e2m1

} // namespace scale::detail
} // namespace ck_tile::core::arch::mma
