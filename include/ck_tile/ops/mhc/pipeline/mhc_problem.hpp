// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/mhc/pipeline/mhc_gemm_shape.hpp"

namespace ck_tile {

template <typename XDataType_, typename ComputeDataType_, typename YDataType_, typename BlockShape_>
struct MHCProblem
{
    using XDataType       = remove_cvref_t<XDataType_>;
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;
    using YDataType       = remove_cvref_t<YDataType_>;
    using BlockShape      = remove_cvref_t<BlockShape_>;

    // PhiDataType is the same as XDataType for the weight matrix
    using PhiDataType = XDataType;

    // BlockGemm compatibility - map our types to BlockGemm's expected types
    using ADataType = XDataType;       // Input matrix A
    using BDataType = PhiDataType;     // Weight matrix B (phi)
    using CDataType = ComputeDataType; // Output/accumulator matrix C

    // BlockGemmShape with kM, kN, kK members for BlockGemm
    // Using 16x16x16 warp tiles with 1x1 warp layout for 16x16 block
    // Minimal tile size to maximize block count: (1024/16) × (24/16) = 64 × 2 = 128 blocks
    // This provides 8x better parallelism than original (128 blocks vs 16 blocks)
    // Testing if overhead from many small blocks becomes a problem
    using BlockGemmShape =
        TileGemmShape<sequence<16, 16, 16>,  // BlockTile (M, N, K) - minimal tiles for max blocks
                      sequence<1, 1, 1>,     // BlockWarps (1 warp per block)
                      sequence<16, 16, 16>>; // WarpTile (16x16x16 is supported by MFMA)

    // Layout types for BlockGemm
    using ALayout = ck_tile::tensor_layout::gemm::RowMajor; // x is row-major [B, nC]
    using BLayout =
        ck_tile::tensor_layout::gemm::ColumnMajor; // phi treated as column-major for V1 pipeline
    using CLayout = ck_tile::tensor_layout::gemm::RowMajor; // output is row-major

    // For GEMM pipeline compatibility
    using AsDataTypeTuple = tuple<ADataType>;
    using BsDataTypeTuple = tuple<BDataType>;
    using AsLayoutTuple   = tuple<ALayout>;
    using BsLayoutTuple   = tuple<BLayout>;

    using AElementWise = identity;
    using BElementWise = identity;

    static constexpr bool TransposeC = false;
    static constexpr bool kPadM      = true; // Enable padding to help with boundary conditions
    static constexpr bool kPadN      = true; // Enable padding
    static constexpr bool kPadK      = true; // Enable padding
    static constexpr bool Preshuffle = false;

    static constexpr auto Scheduler        = GemmPipelineScheduler::Intrawave;
    static constexpr index_t NumWaveGroups = 1;

    static constexpr index_t VectorLoadSize = 16;
    static constexpr index_t VectorSizeA    = 4;
    static constexpr index_t VectorSizeB    = 4;

    // kBlockSize for BlockGemm compatibility
    static constexpr index_t kBlockSize = BlockShape::BlockSize;

    // Additional traits required by v3 pipeline
    static constexpr bool DoubleSmemBuffer      = true; // Enable double buffering for multi-block
    static constexpr bool UseStructuredSparsity = false;
    static constexpr bool FixedVectorSize       = false;

    struct Traits
    {
        static constexpr bool UsePersistentKernel = false;
    };

    CK_TILE_HOST static const std::string GetName() { return "MHCProblem"; }

    // Helper to derive Generic2dBlockShape from BlockGemmShape
    // This ensures BlockShape parameters match our tile sizes
    using DerivedBlockShape =
        Generic2dBlockShape<sequence<BlockGemmShape::kM, BlockGemmShape::kK>, // BlockTile [M, K]
                            sequence<BlockGemmShape::kM / VectorSizeA, // ThreadPerBlock [M, K]
                                     BlockGemmShape::kK / VectorSizeA>,
                            sequence<1, VectorSizeA>>; // Vector [1, VectorSizeA]

    // Tile distribution for loading X (input matrix) from global memory
    // X is [Batch, nC] row-major, we load kM×kK tiles
    // Use BlockShape parameters to ensure consistency
    CK_TILE_HOST_DEVICE static constexpr auto MakeXLoadTileDistribution()
    {
        using namespace ck_tile;
        using S = BlockShape;

        // For a 2D tile [M, K], we need to define distribution for both dimensions
        // Using BlockShape's Repeat, WarpPerBlock, ThreadPerWarp, Vector parameters
        using XTileDistEncoding =
            tile_distribution_encoding<sequence<>, // R: No replication
                                       tuple<sequence<S::Repeat_M,
                                                      S::WarpPerBlock_M,
                                                      S::ThreadPerWarp_M,
                                                      S::Vector_M>, // H0 (M/Batch dimension)
                                             sequence<S::Repeat_N,
                                                      S::WarpPerBlock_N,
                                                      S::ThreadPerWarp_N,
                                                      S::Vector_N>>, // H1 (K/nC dimension) - using
                                                                     // N params for K
                                       tuple<sequence<1, 2>>,        // P→RH major
                                       tuple<sequence<2, 2>>,        // P→RH minor
                                       sequence<2, 2, 1, 1>,         // Y→RH major
                                       sequence<0, 3, 0, 3>>;        // Y→RH minor

        return make_static_tile_distribution(XTileDistEncoding{});
    }

    // Tile distribution for loading Phi (weight matrix) from global memory
    // Phi is [output_dim, nC] row-major, we load kN×kK tiles
    CK_TILE_HOST_DEVICE static constexpr auto MakePhiLoadTileDistribution()
    {
        using namespace ck_tile;
        using S = BlockShape;

        // For Phi [N, K] tile
        using PhiTileDistEncoding =
            tile_distribution_encoding<sequence<>,                 // R: No replication
                                       tuple<sequence<S::Repeat_N, // H0 (N/output_dim dimension)
                                                      S::WarpPerBlock_N,
                                                      S::ThreadPerWarp_N,
                                                      S::Vector_N>,
                                             sequence<S::Repeat_N, // H1 (K/nC dimension)
                                                      S::WarpPerBlock_N,
                                                      S::ThreadPerWarp_N,
                                                      S::Vector_N>>,
                                       tuple<sequence<1, 2>>, // P→RH major
                                       tuple<sequence<2, 2>>, // P→RH minor
                                       sequence<2, 2, 1, 1>,  // Y→RH major
                                       sequence<0, 3, 0, 3>>; // Y→RH minor

        return make_static_tile_distribution(PhiTileDistEncoding{});
    }
};

} // namespace ck_tile
