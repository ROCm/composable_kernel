// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/mhc/pipeline/mhc_gemm_shape.hpp"

namespace ck_tile {

// MHC Problem V5: Optimized for large C values with split-K
// Adaptive M tile size based on batch size for optimal performance
template <typename XDataType_,
          typename ComputeDataType_,
          typename YDataType_,
          index_t MTile_ = 16> // Default M=16 for small/medium batches
struct MHCProblemV5
{
    using XDataType       = remove_cvref_t<XDataType_>;
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;
    using YDataType       = remove_cvref_t<YDataType_>;

    using PhiDataType = XDataType;

    // BlockGemm compatibility
    using ADataType = XDataType;
    using BDataType = PhiDataType;
    using CDataType = ComputeDataType;

    static constexpr index_t kMTile = MTile_; // Adaptive M tile size

    // Adaptive tile configuration with K-loop optimization
    // M: Adaptive (16 or 64) based on batch size
    // N: 32 (fits output_dim=24 perfectly)
    // K: 128 (allows more K-tiles per block)
    using BlockGemmShape = TileGemmShape<sequence<MTile_, 32, 128>,  // BlockTile
                                         sequence<1, 1, 1>,          // BlockWarps: 1 warp
                                         sequence<MTile_, 32, 128>>; // WarpTile

    static constexpr index_t VectorSizeA = 4;
    static constexpr index_t VectorSizeB = 4;

    // 1 warp × 64 threads/warp = 64 threads
    using BlockShape = Generic2dBlockShape<sequence<1, 64>, sequence<1, 64>, sequence<1, 1>>;

    using ALayout = ck_tile::tensor_layout::gemm::RowMajor;
    using BLayout = ck_tile::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck_tile::tensor_layout::gemm::RowMajor;

    using AsDataTypeTuple = tuple<ADataType>;
    using BsDataTypeTuple = tuple<BDataType>;
    using AsLayoutTuple   = tuple<ALayout>;
    using BsLayoutTuple   = tuple<BLayout>;

    using AElementWise = identity;
    using BElementWise = identity;

    static constexpr bool TransposeC = false;
    static constexpr bool kPadM      = true;
    static constexpr bool kPadN      = true;
    static constexpr bool kPadK      = true;
    static constexpr bool Preshuffle = false;

    static constexpr auto Scheduler        = GemmPipelineScheduler::Intrawave;
    static constexpr index_t NumWaveGroups = 1;

    static constexpr index_t VectorLoadSize = 16;
    static constexpr index_t kBlockSize     = BlockShape::BlockSize;

    static constexpr bool DoubleSmemBuffer      = true;
    static constexpr bool UseStructuredSparsity = false;
    static constexpr bool FixedVectorSize       = false;

    struct Traits
    {
        static constexpr bool UsePersistentKernel = false;
    };

    CK_TILE_HOST static const std::string GetName() { return "MHCProblemV5"; }

    // Tile distribution for loading X: Adaptive_M × 128
    // M: Adaptive (16 or 64)
    // K: 128 = 1×1×4×16 (4 threads × 32 vector for better coalescing)
    CK_TILE_HOST_DEVICE static constexpr auto MakeXLoadTileDistribution()
    {
        using namespace ck_tile;

        constexpr index_t m_grid = MTile_ / 16;

        using XTileDistEncoding =
            tile_distribution_encoding<sequence<>,                       // R: No replication
                                       tuple<sequence<m_grid, 1, 16, 1>, // H0 (M): adaptive
                                             sequence<2, 1, 4, 16>>, // H1 (K): 128 = 2×1×4×16
                                       tuple<sequence<1, 2>, sequence<1, 2>>, // P→RH major
                                       tuple<sequence<1, 1>, sequence<2, 2>>, // P→RH minor
                                       sequence<1, 1, 2, 2>,                  // Y→RH major
                                       sequence<0, 3, 0, 3>>;                 // Y→RH minor

        return make_static_tile_distribution(XTileDistEncoding{});
    }

    // Tile distribution for loading Phi: 32 × 128
    // N: 32 = 1×1×16×2 (16 threads × 2 vector, fits output_dim=24)
    // K: 128 = 2×1×4×16 (matches X distribution)
    CK_TILE_HOST_DEVICE static constexpr auto MakePhiLoadTileDistribution()
    {
        using namespace ck_tile;

        using PhiTileDistEncoding =
            tile_distribution_encoding<sequence<>,                   // R: No replication
                                       tuple<sequence<1, 1, 16, 2>,  // H0 (N): 32 = 1×1×16×2
                                             sequence<2, 1, 4, 16>>, // H1 (K): 128 = 2×1×4×16
                                       tuple<sequence<1, 2>, sequence<1, 2>>, // P→RH major
                                       tuple<sequence<1, 1>, sequence<2, 2>>, // P→RH minor
                                       sequence<1, 1, 2, 2>,                  // Y→RH major
                                       sequence<0, 3, 0, 3>>;                 // Y→RH minor

        return make_static_tile_distribution(PhiTileDistEncoding{});
    }
};

} // namespace ck_tile
