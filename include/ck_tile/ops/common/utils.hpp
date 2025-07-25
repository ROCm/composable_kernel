// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <string>

#include "ck_tile/core.hpp"

namespace ck_tile {

// clang-format off
template <typename T> struct typeToStr;
template <> struct typeToStr<float> { static constexpr const char * name = "fp32"; };
template <> struct typeToStr<fp16_t> { static constexpr const char * name = "fp16"; };
template <> struct typeToStr<bf16_t> { static constexpr const char * name = "bf16"; };
template <> struct typeToStr<fp8_t> { static constexpr const char * name = "fp8"; };
template <> struct typeToStr<bf8_t> { static constexpr const char * name = "bf8"; };
template <> struct typeToStr<int8_t> { static constexpr const char * name = "int8"; };
template <> struct typeToStr<pk_int4_t> { static constexpr const char * name = "pk_int4"; };
// clang-format on

template <typename ADataType_, typename BDataType_>
std::string gemm_prec_str()
{
    std::string base_str = std::string(typeToStr<ADataType_>::name);
    if(!std::is_same_v<ADataType_, BDataType_>)
    {
        base_str += "_" + std::string(typeToStr<BDataType_>::name);
    }
    return base_str;
}

/**
 * @brief Calculate optimal vector size for thread memory loads in tensor operations.
 *
 * @tparam DataType     The tensor data type.
 * @tparam Layout       The tensor layout (RowMajor/ColumnMajor).
 * @tparam Dim0PerBlock The first dimension of the tile.
 * @tparam Dim1PerBlock The second dimension of the tile.
 * @tparam BlockSize    The thread block size.
 * @return Optimal vector size (elements per vector load) for each thread.
 */
template <typename DataType,
          typename Layout,
          index_t Dim0PerBlock,
          index_t Dim1PerBlock,
          index_t BlockSize>
CK_TILE_HOST_DEVICE static constexpr auto GetThreadVectorLoadSize()
{
    // For Dim0 × Dim1 tensor dimensions
    constexpr index_t elements_per_thread = Dim0PerBlock * Dim1PerBlock / BlockSize;
    constexpr index_t PackedSize = ck_tile::numeric_traits<remove_cvref_t<DataType>>::PackedSize;

    // XPerTile depends on layout (contiguous dimension)
    constexpr index_t XPerTile =
        std::is_same_v<Layout, tensor_layout::gemm::RowMajor> ? Dim1PerBlock : Dim0PerBlock;

    if constexpr(XPerTile % (PackedSize * 32 / sizeof(DataType)) == 0 &&
                 elements_per_thread % (PackedSize * 32 / sizeof(DataType)) == 0 && PackedSize == 2)
    {
        return (PackedSize * 32 / sizeof(DataType));
    }
    else if constexpr(XPerTile % (PackedSize * 16 / sizeof(DataType)) == 0 &&
                      elements_per_thread % (PackedSize * 16 / sizeof(DataType)) == 0)
    {
        return (PackedSize * 16 / sizeof(DataType));
    }
    else if constexpr(XPerTile % (PackedSize * 8 / sizeof(DataType)) == 0 &&
                      elements_per_thread % (PackedSize * 8 / sizeof(DataType)) == 0)
    {
        return (PackedSize * 8 / sizeof(DataType));
    }
    else if constexpr(sizeof(DataType) >= PackedSize * 4 &&
                      XPerTile % (PackedSize * 4 / sizeof(DataType)) == 0 &&
                      elements_per_thread % (PackedSize * 4 / sizeof(DataType)) == 0)
    {
        return (PackedSize * 4 / sizeof(DataType));
    }
    else if constexpr(sizeof(DataType) >= PackedSize * 2 &&
                      XPerTile % (PackedSize * 2 / sizeof(DataType)) == 0 &&
                      elements_per_thread % (PackedSize * 2 / sizeof(DataType)) == 0)
    {
        return (PackedSize * 2 / sizeof(DataType));
    }
    else
    {
        return PackedSize;
    }
}

} // namespace ck_tile
