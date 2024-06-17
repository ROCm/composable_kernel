// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename BlockFmhaShape_>
struct FmhaBwdTilePartitioner
{
    using BlockFmhaShape = ck_tile::remove_cvref_t<BlockFmhaShape_>;

    static constexpr ck_tile::index_t kN0 = BlockFmhaShape::kN0;

    CK_TILE_HOST static constexpr auto
    GridSize(ck_tile::index_t batch_size_, ck_tile::index_t nhead_, ck_tile::index_t seqlen_k_)
    {
        // TODO: this may need tuning
        return dim3(ck_tile::integer_divide_ceil(seqlen_k_, kN0), nhead_, batch_size_);
    }

    CK_TILE_DEVICE auto operator()(ck_tile::index_t /*seqlen_k*/)
    {
        const index_t i_block = blockIdx.x;
        const index_t i_nhead = blockIdx.y;
        const index_t i_batch = blockIdx.z;

        return ck_tile::make_tuple(i_block, i_nhead, i_batch);
    }
};

template <ck_tile::index_t kBlockSize>
struct FmhaBwdOGradDotOTilePartitioner
{
    CK_TILE_HOST static constexpr auto
    GridSize(ck_tile::index_t batch_size_, ck_tile::index_t nhead_, ck_tile::index_t seqlen_q_)
    {
        // TODO: this may need tuning
        return dim3(ck_tile::integer_divide_ceil(seqlen_q_, kBlockSize), nhead_, batch_size_);
    }

    CK_TILE_DEVICE auto operator()(ck_tile::index_t /*seqlen_q*/)
    {
        const index_t i_block = blockIdx.x;
        const index_t i_nhead = blockIdx.y;
        const index_t i_batch = blockIdx.z;

        return ck_tile::make_tuple(i_block, i_nhead, i_batch);
    }
};

} // namespace ck_tile
