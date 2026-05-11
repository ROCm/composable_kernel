// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace tutorial_10 {

using namespace ck_tile;

// ============================================================================
// COPY DISTRIBUTIONS
// ============================================================================
// Optimized for memory bandwidth: coalesced global access
// - sequence<1>: NO replication (all 256 threads have unique data)
// - Thread-based hierarchical partitioning: M0/M1/M2 or N0/N1/N2
// - Vector width: K1 = 16 bytes / sizeof(DataType) = 8 for half_t
// - Perfect coalescing: consecutive threads access consecutive addresses

template<typename DataType, index_t kBlockSize, index_t kWaveSize, index_t kKPerBlock, index_t kMPerBlock>
CK_TILE_HOST_DEVICE static constexpr auto MakeACopyDistribution()
{
    // Vector width calculation for 16-byte loads
    constexpr index_t K1 = 16 / sizeof(DataType);  // 8 for half_t
    constexpr index_t K0 = kKPerBlock / K1;         // 32 / 8 = 4
    constexpr index_t M2 = kWaveSize / K0;          // 64 / 4 = 16
    constexpr index_t M1 = kBlockSize / kWaveSize;  // 256 / 64 = 4
    constexpr index_t M0 = kMPerBlock / (M2 * M1);  // 64 / (16 * 4) = 1

    return make_static_tile_distribution(
        tile_distribution_encoding<
            sequence<1>,                                    // NO replication!
            tuple<sequence<M0, M1, M2>, sequence<K0, K1>>, // Thread partitioning
            tuple<sequence<1>, sequence<1, 2>>,            // Ps_to_Hs
            tuple<sequence<1>, sequence<2, 0>>,            // Ps_in_Hs
            sequence<1, 2>,                                 // Ys_to_Hs
            sequence<0, 1>                                  // Ys_in_Hs
        >{});
}

template<typename DataType, index_t kBlockSize, index_t kWaveSize, index_t kKPerBlock, index_t kNPerBlock>
CK_TILE_HOST_DEVICE static constexpr auto MakeBCopyDistribution()
{
    // B is K×N, so vector width applies to N dimension (innermost/contiguous)
    constexpr index_t N1 = 16 / sizeof(DataType);  // 8 for half_t
    constexpr index_t N0 = kNPerBlock / N1;         // 64 / 8 = 8
    constexpr index_t K2 = kWaveSize / N0;          // 64 / 8 = 8
    constexpr index_t K1 = kBlockSize / kWaveSize;  // 256 / 64 = 4
    constexpr index_t K0 = kKPerBlock / (K2 * K1);  // 32 / (8 * 4) = 1

    return make_static_tile_distribution(
        tile_distribution_encoding<
            sequence<1>,                                    // NO replication!
            tuple<sequence<K0, K1, K2>, sequence<N0, N1>>, // Thread partitioning (K, N)
            tuple<sequence<1>, sequence<1, 2>>,            // Ps_to_Hs
            tuple<sequence<1>, sequence<2, 0>>,            // Ps_in_Hs
            sequence<1, 2>,                                 // Ys_to_Hs
            sequence<0, 1>                                  // Ys_in_Hs
        >{});
}

// ============================================================================
// GEMM DISTRIBUTIONS
// ============================================================================
// Optimized for MFMA compute: warp-based with replication
// - sequence<2>: Replication for MFMA (each warp has full data)
// - Warp-level partitioning for M16N16K16 MFMA
// - Each warp gets complete K dimension for computation

template<index_t MWarp, index_t kWaveSize, index_t kWarpM, index_t kWarpK, index_t kMPerBlock, index_t kKPerBlock>
CK_TILE_HOST_DEVICE static constexpr auto MakeAGemmDistribution()
{
    constexpr index_t MIterPerWarp = 2;
    constexpr index_t KIterPerWarp = 2;

    using AWarpDstr = tile_distribution_encoding<
        sequence<2>,
        tuple<sequence<MWarp, MIterPerWarp, 1, kWarpM>,
              sequence<KIterPerWarp, 1, kWarpK>>,
        tuple<sequence<1, 0>, sequence<2, 1>>,
        tuple<sequence<0, 1>, sequence<2, 0>>,
        sequence<2, 3>,
        sequence<2, 3>>;

    return make_static_tile_distribution(AWarpDstr{});
}

template<index_t NWarp, index_t kWaveSize, index_t kWarpN, index_t kWarpK, index_t kNPerBlock, index_t kKPerBlock>
CK_TILE_HOST_DEVICE static constexpr auto MakeBGemmDistribution()
{
    constexpr index_t NIterPerWarp = 2;
    constexpr index_t KIterPerWarp = 2;

    using BWarpDstr = tile_distribution_encoding<
        sequence<2>,
        tuple<sequence<KIterPerWarp, 1, kWarpK>,
              sequence<NWarp, NIterPerWarp, 1, kWarpN>>,
        tuple<sequence<2, 1>, sequence<1, 0>>,
        tuple<sequence<2, 0>, sequence<0, 1>>,
        sequence<2, 3>,
        sequence<2, 3>>;

    return make_static_tile_distribution(BWarpDstr{});
}

} // namespace tutorial_10
