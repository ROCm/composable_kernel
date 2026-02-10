// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/mhc/pipeline/mhc_gemm_shape.hpp"

namespace ck_tile {

// MHC Problem V4: Simplified version that derives BlockShape from BlockGemmShape
// No need to manually specify BlockShape - it's automatically derived
template <typename XDataType_, typename ComputeDataType_, typename YDataType_>
struct MHCProblemV4
{
    using XDataType       = remove_cvref_t<XDataType_>;
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;
    using YDataType       = remove_cvref_t<YDataType_>;

    // PhiDataType is the same as XDataType for the weight matrix
    using PhiDataType = XDataType;

    // BlockGemm compatibility - map our types to BlockGemm's expected types
    using ADataType = XDataType;       // Input matrix A
    using BDataType = PhiDataType;     // Weight matrix B (phi)
    using CDataType = ComputeDataType; // Output/accumulator matrix C

    // BlockGemmShape with kM, kN, kK members for BlockGemm
    // Using 16x16x16 tiles with 1 warp per block
    using BlockGemmShape = TileGemmShape<sequence<16, 16, 16>,  // BlockTile (M, N, K)
                                         sequence<1, 1, 1>,     // BlockWarps (1 warp per block)
                                         sequence<16, 16, 16>>; // WarpTile (16x16x16 MFMA)

    // Vector sizes for loading
    static constexpr index_t VectorSizeA = 4;
    static constexpr index_t VectorSizeB = 4;

    // Derive BlockShape from BlockGemmShape
    // Match V3's approach: use a simple 1×64 configuration for 1 warp
    // This ensures proper tile distribution for load_tile/store_tile operations
    using BlockShape =
        Generic2dBlockShape<sequence<1, 64>, // BlockTile [1, 64] - simple layout for 1 warp
                            sequence<1, 64>, // ThreadPerBlock [1, 64] = 64 threads (1 warp)
                            sequence<1, 1>>; // Vector [1, 1] - no vectorization in BlockShape

    // Layout types for BlockGemm
    using ALayout = ck_tile::tensor_layout::gemm::RowMajor;
    using BLayout = ck_tile::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck_tile::tensor_layout::gemm::RowMajor;

    // For GEMM pipeline compatibility
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

    // kBlockSize derived from BlockShape
    static constexpr index_t kBlockSize = BlockShape::BlockSize;

    // Additional traits
    static constexpr bool DoubleSmemBuffer      = true;
    static constexpr bool UseStructuredSparsity = false;
    static constexpr bool FixedVectorSize       = false;

    struct Traits
    {
        static constexpr bool UsePersistentKernel = false;
    };

    CK_TILE_HOST static const std::string GetName() { return "MHCProblemV4"; }

    // Tile distribution for loading X (input matrix) from global memory
    // X is [Batch, nC] row-major, we load kM×kK tiles (16×16)
    // For a 16×16 tile with 64 threads (1 warp):
    // M: 1 repeat × 1 warp × 16 threads × 1 vector = 16
    // K: 1 repeat × 1 warp × 4 threads × 4 vector = 16
    // Total threads: 1 warp × (16×4) = 64 threads ✓
    CK_TILE_HOST_DEVICE static constexpr auto MakeXLoadTileDistribution()
    {
        using namespace ck_tile;

        // H0 (M dimension): [repeat=1, warp=1, thread=16, vector=1] = 16
        // H1 (K dimension): [repeat=1, warp=1, thread=4, vector=4] = 16
        // P→RH: Warp layout = 1 warp in M × 1 warp in K
        //       Thread layout = 16 threads in M × 4 threads in K = 64 threads/warp
        // Y→RH: Access order = M_repeat → M_vector → K_repeat → K_vector (vectorized)
        using XTileDistEncoding = tile_distribution_encoding<
            sequence<>,                            // R: No replication
            tuple<sequence<1, 1, 16, 1>,           // H0 (M): repeat=1, warp=1, thread=16, vector=1
                  sequence<1, 1, 4, 4>>,           // H1 (K): repeat=1, warp=1, thread=4, vector=4
            tuple<sequence<1, 2>, sequence<1, 2>>, // P→RH major
            tuple<sequence<1, 1>, sequence<2, 2>>, // P→RH minor
            sequence<1, 1, 2, 2>,                  // Y→RH major
            sequence<0, 3, 0, 3>>;                 // Y→RH minor

        return make_static_tile_distribution(XTileDistEncoding{});
    }

    // Tile distribution for loading Phi (weight matrix) from global memory
    // Phi is [output_dim, nC] row-major, we load kN×kK tiles (16×16)
    CK_TILE_HOST_DEVICE static constexpr auto MakePhiLoadTileDistribution()
    {
        using namespace ck_tile;

        // Same distribution as X for 16×16 tiles
        using PhiTileDistEncoding = tile_distribution_encoding<
            sequence<>,                            // R: No replication
            tuple<sequence<1, 1, 16, 1>,           // H0 (N): repeat=1, warp=1, thread=16, vector=1
                  sequence<1, 1, 4, 4>>,           // H1 (K): repeat=1, warp=1, thread=4, vector=4
            tuple<sequence<1, 2>, sequence<1, 2>>, // P→RH major
            tuple<sequence<1, 1>, sequence<2, 2>>, // P→RH minor
            sequence<1, 1, 2, 2>,                  // Y→RH major
            sequence<0, 3, 0, 3>>;                 // Y→RH minor

        return make_static_tile_distribution(PhiTileDistEncoding{});
    }
};

} // namespace ck_tile
