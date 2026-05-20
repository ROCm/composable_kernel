// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/pk_fp4.hpp"
#include "ck_tile/core/numeric/pk_f6.hpp"

#include <cstdint>
#include <stdio.h>
#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER
#include <concepts>
#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

namespace ck_tile::core::arch::mma {

namespace scale::detail {

template <typename T>
struct ScaleDataTypeToFlag;

template <>
struct ScaleDataTypeToFlag<fp8_t> // e4m3 (4 exponent bits 3 mantissa bits)
{
    static constexpr int32_t value = 0;
};

template <>
struct ScaleDataTypeToFlag<bf8_t> // e5m2
{
    static constexpr int32_t value = 1;
};

template <>
struct ScaleDataTypeToFlag<pk_fp6x16_t> // e2m3
{
    static constexpr int32_t value = 2;
};

template <>
struct ScaleDataTypeToFlag<pk_bf6x16_t> // e3m2
{
    static constexpr int32_t value = 3;
};

template <>
struct ScaleDataTypeToFlag<pk_fp4_t> // e2m1
{
    static constexpr int32_t value = 4;
};

template <typename T>
inline constexpr int32_t ScaleDataTypeToFlag_v = ScaleDataTypeToFlag<T>::value;

#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

/**
 * @concept ScaleMfmaDataTypeToFlag
 * @brief  Expresses the interface of required members for each DataTypeToFlag type on Gfx9
 */
template <typename DataTypeToFlag>
concept ScaleMfmaDataTypeToFlag = requires(DataTypeToFlag dataTypeToFlag) {
    // Flag members for scale MFMA instructions
    { DataTypeToFlag::value } -> std::convertible_to<int32_t>;
};

#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

} // namespace scale::detail

struct DefaultScaleMfmaCtrlFlags
{
    static constexpr int32_t OPSEL_A = 0;
    static constexpr int32_t OPSEL_B = 0;
};

CK_TILE_HOST_DEVICE void print_flags(DefaultScaleMfmaCtrlFlags const& ctrlFlags)
{
    printf("CtrlFlags      OPSEL_A / OPSEL_B        : %d / %d\n",
           ctrlFlags.OPSEL_A,
           ctrlFlags.OPSEL_B);
}

#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

/**
 * @concept ScaleMfmaCtrlFlags
 * @brief  Expresses the interface of required members for each CtrlFlags type on Gfx9
 */
template <typename CtrlFlags>
concept ScaleMfmaCtrlFlags = requires(CtrlFlags ctrlFlags) {
    // Flag members for scale MFMA instructions
    { CtrlFlags::OPSEL_A } -> std::convertible_to<int32_t>;
    { CtrlFlags::OPSEL_B } -> std::convertible_to<int32_t>;
};

#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

} // namespace ck_tile::core::arch::mma
