// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/pk_f6.hpp"
#include "ck_tile/core/numeric/pk_fp4.hpp"

#include <cstdint>

namespace ck_tile::core::arch::mma {

/**
 * @brief Maps a data type to its hardware matrix format code used by f8f6f4 builtins.
 */
template <typename T>
struct PackedDataTypeToFlag;

template <>
struct PackedDataTypeToFlag<fp8_t> // e4m3
{
    static constexpr int32_t value = 0;
};

template <>
struct PackedDataTypeToFlag<bf8_t> // e5m2
{
    static constexpr int32_t value = 1;
};

template <>
struct PackedDataTypeToFlag<pk_fp6x16_t> // e2m3
{
    static constexpr int32_t value = 2;
};

template <>
struct PackedDataTypeToFlag<pk_bf6x16_t> // e3m2
{
    static constexpr int32_t value = 3;
};

template <>
struct PackedDataTypeToFlag<pk_fp4_t> // e2m1
{
    static constexpr int32_t value = 4;
};

template <typename T>
inline constexpr int32_t PackedDataTypeToFlag_v = PackedDataTypeToFlag<T>::value;

} // namespace ck_tile::core::arch::mma
