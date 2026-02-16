// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/mhc/pipeline/mhc_gemm_shape.hpp"

namespace ck_tile {

// MHC Problem V5: Optimized for large C values with split-K
// 2-warp configuration in M dimension for improved MFMA utilization
template <typename XDataType_,
          typename ComputeDataType_,
          typename YDataType_,
          index_t MTile_ = 16> // Template parameter (not used in 2-warp config)
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

    static constexpr index_t kMTile = MTile_; // Use template parameter

    // Adaptive warp configuration based on MTile
    // MTile=16: 1-warp config (16×32×128, 1 warp, 16×32×128 per warp)
    // MTile=64: 2-warp config (64×32×128, 2 warps in M, 32×32×128 per warp)
    // MTile=128: 4-warp config (128×32×64, 4 warps in M, 32×32×64 per warp) - reduced K to fit LDS
    using BlockGemmShape = std::conditional_t<
        MTile_ == 16,
        TileGemmShape<sequence<16, 32, 128>,  // BlockTile for M=16
                      sequence<1, 1, 1>,      // BlockWarps: 1 warp
                      sequence<16, 32, 128>>, // WarpTile
        std::conditional_t<MTile_ == 64,
                           TileGemmShape<sequence<64, 32, 128>,  // BlockTile for M=64
                                         sequence<2, 1, 1>,      // BlockWarps: 2 in M
                                         sequence<32, 32, 128>>, // WarpTile
                           TileGemmShape<sequence<128, 32, 64>,  // BlockTile for M=128, K=64 to fit
                                                                 // LDS
                                         sequence<4, 1, 1>,      // BlockWarps: 4 in M
                                         sequence<32, 32, 64>>   // WarpTile
                           >>;

    static constexpr index_t VectorSizeA = 4;
    static constexpr index_t VectorSizeB = 4;

    // Adaptive block size based on MTile
    // MTile=16: 1 warp × 64 threads = 64 threads
    // MTile=64: 2 warps × 64 threads = 128 threads
    // MTile=128: 4 warps × 64 threads = 256 threads
    using BlockShape = std::conditional_t<
        MTile_ == 16,
        Generic2dBlockShape<sequence<1, 64>, sequence<1, 64>, sequence<1, 1>>, // 64 threads
        std::conditional_t<
            MTile_ == 64,
            Generic2dBlockShape<sequence<1, 128>, sequence<1, 128>, sequence<1, 1>>, // 128 threads
            Generic2dBlockShape<sequence<1, 256>, sequence<1, 256>, sequence<1, 1>>  // 256 threads
            >>;

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

    CK_TILE_HOST static const std::string GetName()
    {
        return MTile_ == 16   ? "MHCProblemV5_1Warp_M16"
               : MTile_ == 64 ? "MHCProblemV5_2Warp_M64"
                              : "MHCProblemV5_4Warp_M128";
    }

    // X tile distribution - adaptive based on MTile
    CK_TILE_HOST_DEVICE static constexpr auto MakeXLoadTileDistribution()
    {
        using namespace ck_tile;

        if constexpr(MTile_ == 16)
        {
            // M=16: 1 warp, 64 threads
            // M: 16 = 1×1×16×1 (1 repeat × 1 warp × 16 threads × 1 vector)
            // K: 128 = 1×1×4×32 (1 repeat × 1 warp × 4 threads × 32 vector)
            using XTileDistEncoding =
                tile_distribution_encoding<sequence<>,                   // R: No replication
                                           tuple<sequence<1, 1, 16, 1>,  // H0 (M): 16 = 1×1×16×1
                                                 sequence<1, 1, 4, 32>>, // H1 (K): 128 = 1×1×4×32
                                           tuple<sequence<1, 2>, sequence<1, 2>>, // P→RH major
                                           tuple<sequence<1, 1>, sequence<2, 2>>, // P→RH minor
                                           sequence<1, 1, 2, 2>,                  // Y→RH major
                                           sequence<0, 3, 0, 3>>;                 // Y→RH minor

            return make_static_tile_distribution(XTileDistEncoding{});
        }
        else if constexpr(MTile_ == 64)
        {
            // M=64: 2 warps, 128 threads
            // M: 64 = 1×2×8×4 (1 repeat × 2 warps × 8 threads × 4 vector)
            // K: 128 = 1×1×8×16 (1 repeat × 1 warp × 8 threads × 16 vector)
            // Thread layout per warp: 8×8 = 64 threads
            using XTileDistEncoding =
                tile_distribution_encoding<sequence<>,                   // R: No replication
                                           tuple<sequence<1, 2, 8, 4>,   // H0 (M): 64 = 1×2×8×4
                                                 sequence<1, 1, 8, 16>>, // H1 (K): 128 = 1×1×8×16
                                           tuple<sequence<1, 2>, sequence<1, 2>>, // P→RH major
                                           tuple<sequence<1, 1>, sequence<2, 2>>, // P→RH minor
                                           sequence<1, 1, 2, 2>,                  // Y→RH major
                                           sequence<0, 3, 0, 3>>;                 // Y→RH minor

            return make_static_tile_distribution(XTileDistEncoding{});
        }
        else // MTile_ == 128
        {
            // M=128: 4 warps, 256 threads
            // M: 128 = 1×4×16×2 (1 repeat × 4 warps × 16 threads × 2 vector)
            // K: 64 = 1×1×4×16 (1 repeat × 1 warp × 4 threads × 16 vector) - reduced to fit LDS
            using XTileDistEncoding =
                tile_distribution_encoding<sequence<>,                  // R: No replication
                                           tuple<sequence<1, 4, 16, 2>, // H0 (M): 128 = 1×4×16×2
                                                 sequence<1, 1, 4, 16>>, // H1 (K): 64 = 1×1×4×16
                                           tuple<sequence<1, 2>, sequence<1, 2>>, // P→RH major
                                           tuple<sequence<1, 1>, sequence<2, 2>>, // P→RH minor
                                           sequence<1, 1, 2, 2>,                  // Y→RH major
                                           sequence<0, 3, 0, 3>>;                 // Y→RH minor

            return make_static_tile_distribution(XTileDistEncoding{});
        }
    }

    // Phi tile distribution - adaptive based on MTile (K dimension changes for M=128)
    CK_TILE_HOST_DEVICE static constexpr auto MakePhiLoadTileDistribution()
    {
        using namespace ck_tile;

        if constexpr(MTile_ == 128)
        {
            // N: 32 = 1×1×16×2 (1 repeat × 1 warp × 16 threads × 2 vector)
            // K: 64 = 1×1×4×16 (1 repeat × 1 warp × 4 threads × 16 vector) - reduced for M=128
            using PhiTileDistEncoding =
                tile_distribution_encoding<sequence<>,                   // R: No replication
                                           tuple<sequence<1, 1, 16, 2>,  // H0 (N): 32 = 1×1×16×2
                                                 sequence<1, 1, 4, 16>>, // H1 (K): 64 = 1×1×4×16
                                           tuple<sequence<1, 2>, sequence<1, 2>>, // P→RH major
                                           tuple<sequence<1, 1>, sequence<2, 2>>, // P→RH minor
                                           sequence<1, 1, 2, 2>,                  // Y→RH major
                                           sequence<0, 3, 0, 3>>;                 // Y→RH minor

            return make_static_tile_distribution(PhiTileDistEncoding{});
        }
        else if constexpr(MTile_ == 64)
        {
            // N: 32 = 1×1×8×4 (1 repeat × 1 warp × 8 threads × 4 vector)
            // K: 128 = 1×1×8×16 (1 repeat × 1 warp × 8 threads × 16 vector)
            // Thread layout: 8×8 = 64 threads (matches X distribution)
            using PhiTileDistEncoding =
                tile_distribution_encoding<sequence<>,                   // R: No replication
                                           tuple<sequence<1, 1, 8, 4>,   // H0 (N): 32 = 1×1×8×4
                                                 sequence<1, 1, 8, 16>>, // H1 (K): 128 = 1×1×8×16
                                           tuple<sequence<1, 2>, sequence<1, 2>>, // P→RH major
                                           tuple<sequence<1, 1>, sequence<2, 2>>, // P→RH minor
                                           sequence<1, 1, 2, 2>,                  // Y→RH major
                                           sequence<0, 3, 0, 3>>;                 // Y→RH minor

            return make_static_tile_distribution(PhiTileDistEncoding{});
        }
        else // MTile_ == 16
        {
            // N: 32 = 1×1×16×2 (1 repeat × 1 warp × 16 threads × 2 vector)
            // K: 128 = 1×1×4×32 (1 repeat × 1 warp × 4 threads × 32 vector)
            using PhiTileDistEncoding =
                tile_distribution_encoding<sequence<>,                   // R: No replication
                                           tuple<sequence<1, 1, 16, 2>,  // H0 (N): 32 = 1×1×16×2
                                                 sequence<1, 1, 4, 32>>, // H1 (K): 128 = 1×1×4×32
                                           tuple<sequence<1, 2>, sequence<1, 2>>, // P→RH major
                                           tuple<sequence<1, 1>, sequence<2, 2>>, // P→RH minor
                                           sequence<1, 1, 2, 2>,                  // Y→RH major
                                           sequence<0, 3, 0, 3>>;                 // Y→RH minor

            return make_static_tile_distribution(PhiTileDistEncoding{});
        }
    }
};

} // namespace ck_tile
