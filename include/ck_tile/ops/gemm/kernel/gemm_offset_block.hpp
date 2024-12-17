// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {
template <typename TilePartitioner_>
struct OffsettedBlockToCTileMap
{
    using tile_partitioner_type = TilePartitioner_;

    __host__ __device__ OffsettedBlockToCTileMap(ck_tile::index_t B2CkTileMap,
                                                 ck_tile::index_t M,
                                                 ck_tile::index_t N)
        : B2CkTileMap_{B2CkTileMap}, M_{M}, N_{N}
    {
    }

    __host__ __device__ constexpr auto CalculateBottomIndex(const ck_tile::index_t idx_top) const
    {
        ck_tile::index_t block_1d_id = idx_top;

        const auto M0 = ck_tile::integer_divide_ceil(M_, tile_partitioner_type::MPerBlock);
        const auto N0 = ck_tile::integer_divide_ceil(N_, tile_partitioner_type::NPerBlock);

        block_1d_id = block_1d_id % (M0 * N0);

        block_1d_id = block_1d_id % (M0 * N0);

        ck_tile::index_t idx_N0 = block_1d_id % N0;
        ck_tile::index_t idx_M0 = block_1d_id / N0;

        const auto M01_adapt = (idx_M0 < M0 - M0 % B2CkTileMap_) ? B2CkTileMap_ : M0 % B2CkTileMap_;

        ck_tile::index_t idx_M00          = idx_M0 / B2CkTileMap_;
        ck_tile::index_t idx_M01          = idx_M0 % B2CkTileMap_;
        ck_tile::index_t idx_N0_M01_local = idx_N0 + idx_M01 * N0;

        return make_tuple(idx_N0_M01_local % M01_adapt + idx_M00 * B2CkTileMap_,
                          idx_N0_M01_local / M01_adapt);
    }

    ck_tile::index_t B2CkTileMap_;
    ck_tile::index_t M_;
    ck_tile::index_t N_;
};
} // namespace ck_tile
