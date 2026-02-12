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

    // Adaptive tile configuration
    // M=16 (default): Optimal for small/medium batches (B < 4096)
    // M=64: Optimal for large batches (B >= 4096)
    // N=32, K=128: Fixed for all configurations
    using BlockGemmShape = TileGemmShape<sequence<MTile_, 32, 128>,  // BlockTile: Adaptive M
                                         sequence<1, 1, 1>,          // BlockWarps: 1 warp
                                         sequence<MTile_, 32, 128>>; // WarpTile: matches BlockTile

    static constexpr index_t VectorSizeA = 4;
    static constexpr index_t VectorSizeB = 4;

    // 1 warp × 64 threads/warp = 64 threads (same as V4)
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

    // Adaptive tile distribution for loading X (input matrix)
    // X is [Batch, nC] row-major, we load kM×kK tiles
    // For M=16: H0 (M): [grid=1, warp=1, thread=16, vector=1] = 16
    // For M=64: H0 (M): [grid=4, warp=1, thread=16, vector=1] = 64
    // H1 (K): [grid=2, warp=1, thread=4, vector=16] = 128 (same for all)
    CK_TILE_HOST_DEVICE static constexpr auto MakeXLoadTileDistribution()
    {
        using namespace ck_tile;

        constexpr index_t m_grid = MTile_ / 16; // M=16 → grid=1, M=64 → grid=4

        using XTileDistEncoding = tile_distribution_encoding<
            sequence<>,                            // R: No replication
            tuple<sequence<m_grid, 1, 16, 1>,      // H0 (M): adaptive grid based on MTile_
                  sequence<2, 1, 4, 16>>,          // H1 (K): grid=2, warp=1, thread=4, vector=16
            tuple<sequence<1, 2>, sequence<1, 2>>, // P→RH major: warp arrangement
            tuple<sequence<1, 1>, sequence<2, 2>>, // P→RH minor: thread arrangement
            sequence<1, 1, 2, 2>,                  // Y→RH major: data layout
            sequence<0, 3, 0, 3>>;                 // Y→RH minor: vectorization

        return make_static_tile_distribution(XTileDistEncoding{});
    }

    // Tile distribution for loading Phi (weight matrix)
    // Phi is [output_dim, nC] row-major, we load kN×kK tiles (32×128)
    // H0 (N): [grid=1, warp=1, thread=16, vector=2] = 32
    // H1 (K): [grid=2, warp=1, thread=4, vector=16] = 128
    CK_TILE_HOST_DEVICE static constexpr auto MakePhiLoadTileDistribution()
    {
        using namespace ck_tile;

        using PhiTileDistEncoding = tile_distribution_encoding<
            sequence<>,                            // R: No replication
            tuple<sequence<1, 1, 16, 2>,           // H0 (N): grid=1, warp=1, thread=16, vector=2
                  sequence<2, 1, 4, 16>>,          // H1 (K): grid=2, warp=1, thread=4, vector=16
            tuple<sequence<1, 2>, sequence<1, 2>>, // P→RH major: warp arrangement
            tuple<sequence<1, 1>, sequence<2, 2>>, // P→RH minor: thread arrangement
            sequence<1, 1, 2, 2>,                  // Y→RH major: data layout
            sequence<0, 3, 0, 3>>;                 // Y→RH minor: vectorization

        return make_static_tile_distribution(PhiTileDistEncoding{});
    }
};

} // namespace ck_tile
