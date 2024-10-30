// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

#define VECTOR_LOAD_SIZE 16

namespace ck_tile {

template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename BlockGemmShape_,
          typename TileGemmTraits_>
struct GemmPipelineProblem
{
    using ADataType      = remove_cvref_t<ADataType_>;
    using BDataType      = remove_cvref_t<BDataType_>;
    using CDataType      = remove_cvref_t<CDataType_>;
    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;
    using GemmTraits     = remove_cvref_t<TileGemmTraits_>;

    static constexpr index_t kBlockSize = BlockGemmShape::NumWarps * get_warp_size();
    static constexpr bool kPadM         = GemmTraits::kPadM;
    static constexpr bool kPadN         = GemmTraits::kPadN;
    static constexpr bool kPadK         = GemmTraits::kPadK;

    using LayoutA = remove_cvref_t<typename GemmTraits::LayoutA>;
    using LayoutB = remove_cvref_t<typename GemmTraits::LayoutB>;
    using LayoutC = remove_cvref_t<typename GemmTraits::LayoutC>;

    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentA()
    {
        if constexpr(std::is_same_v<LayoutA, ck_tile::tensor_layout::gemm::ColumnMajor>)
        {
            constexpr index_t pixels_per_thread =
                BlockGemmShape::kM * BlockGemmShape::kK / kBlockSize;
            return pixels_per_thread < VECTOR_LOAD_SIZE / sizeof(ADataType)
                       ? pixels_per_thread
                       : VECTOR_LOAD_SIZE / sizeof(ADataType);
        }
        else
        {
            return VECTOR_LOAD_SIZE / sizeof(ADataType);
        }
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentB()
    {
        if constexpr(std::is_same_v<LayoutB, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            constexpr index_t pixels_per_thread =
                BlockGemmShape::kN * BlockGemmShape::kK / kBlockSize;
            return pixels_per_thread < VECTOR_LOAD_SIZE / sizeof(BDataType)
                       ? pixels_per_thread
                       : VECTOR_LOAD_SIZE / sizeof(BDataType);
        }
        else
        {
            return VECTOR_LOAD_SIZE / sizeof(BDataType);
        }
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentC()
    {
        if constexpr(std::is_same_v<LayoutC, ck_tile::tensor_layout::gemm::ColumnMajor>)
        {
            constexpr index_t N1 = kBlockSize / get_warp_size();
            constexpr index_t N2 = min(BlockGemmShape::kN / N1, get_warp_size());
            constexpr index_t M0 = get_warp_size() / N2;
            constexpr index_t M1 = BlockGemmShape::kM / M0;

            return min(M1, static_cast<index_t>(VECTOR_LOAD_SIZE / sizeof(CDataType)));
        }
        else
        {
            constexpr index_t M1 = kBlockSize / get_warp_size();
            constexpr index_t M2 = min(BlockGemmShape::kM / M1, get_warp_size());
            constexpr index_t N0 = get_warp_size() / M2;
            constexpr index_t N1 = BlockGemmShape::kN / N0;

            return min(N1, static_cast<index_t>(VECTOR_LOAD_SIZE / sizeof(CDataType)));
        }
    }

    static constexpr index_t AlignmentA = []() {
        if constexpr(std::is_same_v<LayoutA, tensor_layout::gemm::RowMajor>)
        {
            return kPadK ? 1 : GetAlignmentA();
        }
        else
        {
            return kPadM ? 1 : GetAlignmentA();
        }
    }();
    static constexpr index_t AlignmentB = []() {
        if constexpr(std::is_same_v<LayoutB, tensor_layout::gemm::RowMajor>)
        {
            return kPadN ? 1 : GetAlignmentB();
        }
        else
        {
            return kPadK ? 1 : GetAlignmentB();
        }
    }();
    static constexpr index_t AlignmentC = []() {
        if constexpr(std::is_same_v<LayoutC, tensor_layout::gemm::RowMajor>)
        {
            return kPadN ? 1 : GetAlignmentC();
        }
        else
        {
            return kPadM ? 1 : GetAlignmentC();
        }
    }();
};

} // namespace ck_tile
