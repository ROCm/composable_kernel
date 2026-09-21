// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <index_t Headdim>
static CK_TILE_HOST_DEVICE constexpr index_t ceil_to_qualified_tile_length()
{
    if constexpr(Headdim == 48)
        return 48;
    else if constexpr(Headdim == 80)
        return 96;
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
struct TileSageAttnShape
{
    using BlockTile       = remove_cvref_t<BlockTile_>;
    using Gemm0BlockWarps = remove_cvref_t<Gemm0BlockWarps_>;
    using Gemm0WarpTile   = remove_cvref_t<Gemm0WarpTile_>;
    using Gemm1BlockWarps = remove_cvref_t<Gemm1BlockWarps_>;
    using Gemm1WarpTile   = remove_cvref_t<Gemm1WarpTile_>;

    static constexpr index_t NumGemm0Warps =
        reduce_on_sequence(Gemm0BlockWarps{}, multiplies<>{}, number<1>{});
    static constexpr index_t NumGemm1Warps =
        reduce_on_sequence(Gemm1BlockWarps{}, multiplies<>{}, number<1>{});
    static_assert(NumGemm1Warps % NumGemm0Warps == 0);

    static constexpr index_t NumWarps = max(NumGemm0Warps, NumGemm1Warps);

    static constexpr index_t kM0 = BlockTile::at(number<0>{}); // tile size along q seqlen
    static constexpr index_t kN0 = BlockTile::at(number<1>{}); // tile size along k seqlen
    static constexpr index_t kK0 = BlockTile::at(number<2>{}); // tile size along qk gemm unroll
    static constexpr index_t kN1 = BlockTile::at(number<3>{}); // tile size along v head_dim
    static constexpr index_t kK1 = BlockTile::at(number<4>{}); // tile size along kv gemm unroll
    static constexpr index_t kQKHeaddim =
        BlockTile::at(number<5>{}); // total length of K0, used for pipeline that need load Q at
                                    // once (or repeately load Q as a whole tile)
    static_assert(kQKHeaddim % kK0 == 0, "kQKHeaddim should be divisible by kK0");

    static constexpr index_t kSubQKHeaddim = ceil_to_qualified_tile_length<kQKHeaddim>();

    // v, rowmajor : seqlen*hdim, colmajor : hdim*seqlen
    static constexpr bool IsVLayoutRowMajor = IsVLayoutRowMajor_;
    using VLayout                           = std::conditional_t<IsVLayoutRowMajor,
                                                                 ck_tile::tensor_layout::gemm::RowMajor,
                                                                 ck_tile::tensor_layout::gemm::ColumnMajor>;
};

} // namespace ck_tile
