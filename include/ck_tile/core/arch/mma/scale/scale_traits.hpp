// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/pk_fp4.hpp"
// #include "ck_tile/core/numeric/pk_fp6.hpp"

#include <cstdint>
#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER
#include <concepts>
#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

namespace ck_tile::core::arch::mma {

namespace scale::detail {

template <typename T>
struct ScaleDataTypeToFlag;

template <>
struct ScaleDataTypeToFlag<fp8_t> // e4m3
{
    static constexpr std::int32_t value = 0;
};

template <>
struct ScaleDataTypeToFlag<bf8_t> // e5m2
{
    static constexpr std::int32_t value = 1;
};

// template <>
// struct ScaleDataTypeToFlag<pk_fp6_t<1>> // e2m3
// {
//     static constexpr std::int32_t value = 2;
// };

// template <>
// struct ScaleDataTypeToFlag<bf6_t> // e3m2
// {
//     static constexpr std::int32_t value = 3;
// };

template <>
struct ScaleDataTypeToFlag<pk_fp4_t> // e2m1
{
    static constexpr std::int32_t value = 4;
};

#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

/**
 * @concept ScaleMfmaDataTypeToFlag
 * @brief  Expresses the interface of required members for each DataTypeToFlag type on Gfx9
 */
template <typename DataTypeToFlag>
concept ScaleMfmaDataTypeToFlag = requires(DataTypeToFlag dataTypeToFlag) {
    // Flag members for scale MFMA instructions
    { DataTypeToFlag::value } -> std::convertible_to<int>;
};

#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

template <typename T>
inline constexpr std::int32_t ScaleDataTypeToFlag_v = ScaleDataTypeToFlag<T>::value;

} // namespace scale::detail

struct DefaultScaleMfmaCtrlFlags
{
    static constexpr std::int32_t OPSEL_A = 0;
    static constexpr std::int32_t OPSEL_B = 0;
};

#if CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

/**
 * @concept ScaleMfmaCtrlFlags
 * @brief  Expresses the interface of required members for each CtrlFlags type on Gfx9
 */
template <typename CtrlFlags>
concept ScaleMfmaCtrlFlags = requires(CtrlFlags ctrlFlags) {
    // Flag members for scale MFMA instructions
    { CtrlFlags::OPSEL_A } -> std::convertible_to<int>;
    { CtrlFlags::OPSEL_B } -> std::convertible_to<int>;
};

#endif // CK_TILE_CONCEPTS && CK_TILE_CONCEPTS_HEADER

} // namespace ck_tile::core::arch::mma
