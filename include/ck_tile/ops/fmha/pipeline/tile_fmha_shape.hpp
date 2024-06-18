// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename BlockTile_, // sequence<...
          typename Gemm0BlockWarps_,
          typename Gemm0WarpTile_,
          typename Gemm1BlockWarps_,
          typename Gemm1WarpTile_,
          bool IsVLayoutRowMajor_>
struct TileFmhaShape
{
    using BlockTile       = remove_cvref_t<BlockTile_>;
    using Gemm0BlockWarps = remove_cvref_t<Gemm0BlockWarps_>;
    using Gemm0WarpTile   = remove_cvref_t<Gemm0WarpTile_>;
    using Gemm1BlockWarps = remove_cvref_t<Gemm1BlockWarps_>;
    using Gemm1WarpTile   = remove_cvref_t<Gemm1WarpTile_>;

    static constexpr index_t NumWarps =
        reduce_on_sequence(Gemm0BlockWarps{}, multiplies{}, number<1>{});

    static_assert(NumWarps == reduce_on_sequence(Gemm1BlockWarps{}, multiplies{}, number<1>{}));

    static constexpr index_t kM0 = BlockTile::at(number<0>{}); // tile size along q seqlen
    static constexpr index_t kN0 = BlockTile::at(number<1>{}); // tile size along k seqlen
    static constexpr index_t kK0 = BlockTile::at(number<2>{}); // tile size along qk gemm unroll
    static constexpr index_t kN1 = BlockTile::at(number<3>{}); // tile size along v head_dim
    static constexpr index_t kK1 = BlockTile::at(number<4>{}); // tile size along kv gemm unroll
    static constexpr index_t kK0BlockLength =
        BlockTile::at(number<5>{}); // total length of K0, used for pipeline that need load Q at
                                    // once (or repeately load Q as a whole tile)
    static_assert(kK0BlockLength % kK0 == 0, "kK0BlockLength should be divisible by kK0");

    // v, rowmajor : seqlen*hdim, colmajor : hdim*seqlen
    static constexpr bool IsVLayoutRowMajor = IsVLayoutRowMajor_;
    using VLayout                           = std::conditional_t<IsVLayoutRowMajor,
                                       ck_tile::tensor_layout::gemm::RowMajor,
                                       ck_tile::tensor_layout::gemm::ColumnMajor>;
};

template <typename BlockTile_, // sequence<...
          typename Gemm0BlockWarps_,
          typename Gemm0WarpTile_,
          typename Gemm1BlockWarps_,
          typename Gemm1WarpTile_,
          typename Gemm2BlockWarps_,
          typename Gemm2WarpTile_,
          typename Gemm3BlockWarps_,
          typename Gemm3WarpTile_,
          typename Gemm4BlockWarps_,
          typename Gemm4WarpTile_>
struct TileFmhaBwdShape
{
    using BlockTile       = remove_cvref_t<BlockTile_>;
    using Gemm0BlockWarps = remove_cvref_t<Gemm0BlockWarps_>;
    using Gemm0WarpTile   = remove_cvref_t<Gemm0WarpTile_>;
    using Gemm1BlockWarps = remove_cvref_t<Gemm1BlockWarps_>;
    using Gemm1WarpTile   = remove_cvref_t<Gemm1WarpTile_>;
    using Gemm2BlockWarps = remove_cvref_t<Gemm2BlockWarps_>;
    using Gemm2WarpTile   = remove_cvref_t<Gemm2WarpTile_>;
    using Gemm3BlockWarps = remove_cvref_t<Gemm3BlockWarps_>;
    using Gemm3WarpTile   = remove_cvref_t<Gemm3WarpTile_>;
    using Gemm4BlockWarps = remove_cvref_t<Gemm4BlockWarps_>;
    using Gemm4WarpTile   = remove_cvref_t<Gemm4WarpTile_>;

    static constexpr index_t NumWarps =
        reduce_on_sequence(Gemm0BlockWarps{}, multiplies{}, number<1>{});

    static_assert(NumWarps == reduce_on_sequence(Gemm1BlockWarps{}, multiplies{}, number<1>{}) &&
                  NumWarps == reduce_on_sequence(Gemm4BlockWarps{}, multiplies{}, number<1>{}));

    static constexpr index_t kM0 = BlockTile::at(number<0>{}); // tile size along q seqlen
    static constexpr index_t kN0 = BlockTile::at(number<1>{}); // tile size along k seqlen
    static constexpr index_t kK0 =
        BlockTile::at(number<2>{}); // tile size along gemm0(Q@K^T) unroll
    static constexpr index_t kK1 =
        BlockTile::at(number<3>{}); // tile size along gemm1(P^T@dO) unroll
    static constexpr index_t kK2 =
        BlockTile::at(number<4>{}); // tile size along gemm2(dO@V^T) unroll
    static constexpr index_t kK3 =
        BlockTile::at(number<5>{}); // tile size along gemm3(dS^T@Q) unroll
    static constexpr index_t kK4 = BlockTile::at(number<6>{}); // tile size along gemm4(dS@K) unroll
    static constexpr index_t kQKHeaddim =
        BlockTile::at(number<7>{}); // Q & K headdim, used for pipeline that need load Q/Q^T or
                                    // K/K^T at once
    static constexpr index_t kVHeaddim = BlockTile::at(number<8>{}); // V headdim, used for pipeline
                                                                     // that need load V at once
};

} // namespace ck_tile
