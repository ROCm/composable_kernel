// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <bool kPadM_,
          bool kPadN_,
          bool kPadK_,
          typename BlockGemmShape_,
          typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename ALayout_,
          typename BLayout_,
          typename CLayout_>
struct TileGemmTraits
{
    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;
    
    static constexpr bool kPadM = kPadM_;
    static constexpr bool kPadN = kPadN_;
    static constexpr bool kPadK = kPadK_;

    static constexpr int _VectorSize = 16;

    static constexpr index_t kBlockSize = BlockGemmShape::NumWarps * get_warp_size();

    using ALayout = ALayout_;
    using BLayout = BLayout_;
    using CLayout = CLayout_;

    using ADataType      = remove_cvref_t<ADataType_>;
    using BDataType      = remove_cvref_t<BDataType_>;
    using CDataType      = remove_cvref_t<CDataType_>;

    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentA()
    {
        if constexpr(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::ColumnMajor>)
        {
            constexpr index_t pixels_per_thread =
                BlockGemmShape::kM * BlockGemmShape::kK / kBlockSize;
            return pixels_per_thread < _VectorSize / sizeof(ADataType)
                       ? pixels_per_thread
                       : _VectorSize / sizeof(ADataType);
        }
        else
        {
            return _VectorSize / sizeof(ADataType);
        }
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentB()
    {
        if constexpr(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            constexpr index_t pixels_per_thread =
                BlockGemmShape::kN * BlockGemmShape::kK / kBlockSize;
            return pixels_per_thread < _VectorSize / sizeof(BDataType)
                       ? pixels_per_thread
                       : _VectorSize / sizeof(BDataType);
        }
        else
        {
            return _VectorSize / sizeof(BDataType);
        }
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentC()
    {
        if constexpr(std::is_same_v<CLayout, ck_tile::tensor_layout::gemm::ColumnMajor>)
        {
            constexpr index_t N1 = kBlockSize / get_warp_size();
            constexpr index_t N2 = min(BlockGemmShape::kN / N1, get_warp_size());
            constexpr index_t M0 = get_warp_size() / N2;
            constexpr index_t M1 = BlockGemmShape::kM / M0;

            return min(M1, static_cast<index_t>(_VectorSize / sizeof(CDataType)));
        }
        else
        {
            constexpr index_t M1 = kBlockSize / get_warp_size();
            constexpr index_t M2 = min(BlockGemmShape::kM / M1, get_warp_size());
            constexpr index_t N0 = get_warp_size() / M2;
            constexpr index_t N1 = BlockGemmShape::kN / N0;

            return min(N1, static_cast<index_t>(_VectorSize / sizeof(CDataType)));
        }
    }

    static constexpr index_t VectorSizeA = []() {
        if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
        {
            return kPadK ? 1 : GetAlignmentA();
        }
        else
        {
            return kPadM ? 1 : GetAlignmentA();
        }
    }();
    static constexpr index_t VectorSizeB = []() {
        if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>)
        {
            return kPadN ? 1 : GetAlignmentB();
        }
        else
        {
            return kPadK ? 1 : GetAlignmentB();
        }
    }();
    static constexpr index_t VectorSizeC = []() {
        if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
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
