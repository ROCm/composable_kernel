// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <index_t Headdim>
static CK_TILE_HOST_DEVICE constexpr index_t ceil_to_qualified_tile_length()
{
    if constexpr(Headdim == 48)
        return 48;
    else if constexpr(Headdim == 96)
        return 128;
    else if constexpr(Headdim == 160)
        return 256;
    else if constexpr(Headdim == 192)
        return 192;
    else if constexpr(is_power_of_two_integer(Headdim))
        return Headdim;
    else
        static_assert(Headdim == 0,
                      "only Headdim of 48, 96, 160, 192 and power-of-two is supported");
};

template <typename BlockTile_, // sequence<...
          typename Gemm0BlockWarps_,
          typename Gemm0WarpTile_,
          typename Gemm1BlockWarps_,
          typename Gemm1WarpTile_,
          bool IsVLayoutRowMajor_>
struct TileUnifiedAttentionShape
{
    using BlockTile       = remove_cvref_t<BlockTile_>;
    using Gemm0BlockWarps = remove_cvref_t<Gemm0BlockWarps_>;
    using Gemm0WarpTile   = remove_cvref_t<Gemm0WarpTile_>;
    using Gemm1BlockWarps = remove_cvref_t<Gemm1BlockWarps_>;
    using Gemm1WarpTile   = remove_cvref_t<Gemm1WarpTile_>;

    static constexpr index_t NumGemm0Warps =
        reduce_on_sequence(Gemm0BlockWarps{}, multiplies{}, number<1>{});
    static constexpr index_t NumGemm1Warps =
        reduce_on_sequence(Gemm1BlockWarps{}, multiplies{}, number<1>{});
    static_assert(NumGemm1Warps % NumGemm0Warps == 0);

    static constexpr index_t NumWarps = max(NumGemm0Warps, NumGemm1Warps);

    static constexpr index_t kBlockM = BlockTile::at(
        number<0>{}); // tile size along the flattened batch dimension (: num_queries_per_kv * BS)
    static constexpr index_t kBlockQ = BlockTile::at(
        number<1>{}); // tile size along the flattened batch dimension (: num_queries_per_kv * BS)
    // static constexpr index_t kBlockM = BlockTile::at(number<1>{}); // tile size along q seqlen *
    // num_queries_per_kv (q_head//kv_head)
    static constexpr index_t kPageBlockSize =
        BlockTile::at(number<2>{});                                 //  BLOCK size for K seqlen
    static constexpr index_t kHeadDim = BlockTile::at(number<3>{}); //  BLOCK size for K seqlen

    //     BlockTile::at(number<5>{}); // total length of K0, used for pipeline that need load Q at
    //                                 // once (or repeately load Q as a whole tile)
    // static_assert(kQKHeaddim % kK0 == 0, "kQKHeaddim should be divisible by kK0");

    static constexpr index_t kHeadDimPadded = ceil_to_qualified_tile_length<kHeadDim>();

    // v, rowmajor : seqlen*hdim, colmajor : hdim*seqlen
    static constexpr bool IsVLayoutRowMajor = IsVLayoutRowMajor_;
    using VLayout                           = std::conditional_t<IsVLayoutRowMajor,
                                                                 ck_tile::tensor_layout::gemm::RowMajor,
                                                                 ck_tile::tensor_layout::gemm::ColumnMajor>;
};
} // namespace ck_tile
