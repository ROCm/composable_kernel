// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

/*
tensors:
1. act  (A): input feature map
2. gate (G): B matrix for first gemm, output will do activation(Silu)
3. up   (U): B matrix for first gemm
4. down (D): B matrix for second gemm
                                                                  N_d
                                                                 /   \
                                                                 +----------+ |
                                                                 |   Down   | |
                                                                 x----------x |
                       hidden                hidden          K_d |          | |
                        N_g            N_u                       x----------x |
             |   +------x-----x------+------x-----x------+       |          | |
    dim      |   | Gate |     |      | Up   |     |      |       |          | |
  contiguous |   |      |     |      |      |     |      |       |          | |
             |   |      |     |      |      |     |      |       |          | |
             v   +------x-----x------+------x-----x------+       +----------+ V
      K_a               |     |             |     |                    | contiguous
     /  \               v     v             v     v                    |
    +---------+  +------x-----x------+------x-----x------+             |
M_a |    A    |  |      |     |      |      |     |      |             |
    +---------+  +------x-----x------+------x-----x------+             |
    -------->             |                    |                       |
    contiguous            |                    V                       V
                          |                 x-----x              +----------+
                          +------------->   |  Y  |  --------->  |  Out(O)  |
                            SILU            x-----x              +----------+
                                             K_y = N_g = N_u          dim
*/
template <typename BlockTile_, // sequence<M_a, N_g, N_sub0, K_a, N_d
          typename Gemm0BlockWarps_,
          typename Gemm0WarpTile_,
          typename Gemm1BlockWarps_,
          typename Gemm1WarpTile_>
struct FusedMoeTileShape
{
    using BlockTile       = remove_cvref_t<BlockTile_>;
    using Gemm0BlockWarps = remove_cvref_t<Gemm0BlockWarps_>;
    using Gemm0WarpTile   = remove_cvref_t<Gemm0WarpTile_>;
    using Gemm1BlockWarps = remove_cvref_t<Gemm1BlockWarps_>;
    using Gemm1WarpTile   = remove_cvref_t<Gemm1WarpTile_>;

    static constexpr index_t NumWarps =
        reduce_on_sequence(Gemm0BlockWarps{}, multiplies{}, number<1>{});

    static_assert(NumWarps == reduce_on_sequence(Gemm1BlockWarps{}, multiplies{}, number<1>{}));

    static constexpr index_t kM_a = BlockTile::at(number<0>{});
    static constexpr index_t kN_g = BlockTile::at(number<1>{});
    static constexpr index_t kN_u = BlockTile::at(number<1>{});
    // e.g. N_g = 256, n_sub_gu=128, then we split blockN of G/U into 2 parts to loopover
    // this can help B matrix direct-to-register using too much vgpr issue
    static constexpr index_t kN_sub0 = BlockTile::at(number<2>{});
    static constexpr index_t kK_a    = BlockTile::at(number<3>{});
    static constexpr index_t kN_d    = BlockTile::at(number<4>{});
    // static_assert(kN_g == kN_u);
    static constexpr index_t kK_y = kN_g;

    static constexpr index_t kBlockNSub_0   = kN_sub0; // allow partial
    static constexpr index_t kBlockM_0      = kM_a;
    static constexpr index_t kBlockN_0      = kN_g; // note N will x2 in real pipeline for gemm-0
    static constexpr index_t kBlockK_0      = kK_a;
    static constexpr index_t kWarpM_0       = Gemm0WarpTile::at(number<0>{});
    static constexpr index_t kWarpN_0       = Gemm0WarpTile::at(number<1>{});
    static constexpr index_t kWarpK_0       = Gemm0WarpTile::at(number<2>{});
    static constexpr index_t kBlockWarpsM_0 = Gemm0BlockWarps::at(numner<0>{});
    static constexpr index_t kBlockWarpsN_0 = Gemm0BlockWarps::at(numner<1>{});
    static constexpr index_t kBlockWarpsK_0 = Gemm0BlockWarps::at(numner<2>{});
    static constexpr index_t kSubBlockM_0   = kWarpM_0 * kBlockWarpsM_0;
    static constexpr index_t kSubBlockN_0   = kWarpN_0 * kBlockWarpsN_0;
    static constexpr index_t kSubBlockK_0   = kWarpK_0 * kBlockWarpsK_0;
    static_assert(kBlockM_0 % kSubBlockM_0 == 0);
    static_assert(kBlockN_0 % kSubBlockN_0 == 0);
    static_assert(kBlockK_0 % kSubBlockK_0 == 0);
    static constexpr index_t kWarpRepeatM_0 = kBlockM_0 / kSubBlockM_0;
    static constexpr index_t kWarpRepeatN_0 = kBlockN_0 / kSubBlockN_0;
    static constexpr index_t kWarpRepeatK_0 = kBlockK_0 / kSubBlockK_0;

    static constexpr index_t kBlockKSub_1   = kBlockNSub_0;
    static constexpr index_t kBlockM_1      = kM_a;
    static constexpr index_t kBlockN_1      = kN_d;
    static constexpr index_t kBlockK_1      = kN_g;
    static constexpr index_t kWarpM_1       = Gemm1WarpTile::at(number<0>{});
    static constexpr index_t kWarpN_1       = Gemm1WarpTile::at(number<1>{});
    static constexpr index_t kWarpK_1       = Gemm1WarpTile::at(number<2>{});
    static constexpr index_t kBlockWarpsM_1 = Gemm1BlockWarps::at(numner<0>{});
    static constexpr index_t kBlockWarpsN_1 = Gemm1BlockWarps::at(numner<1>{});
    static constexpr index_t kBlockWarpsK_1 = Gemm1BlockWarps::at(numner<2>{});
    static constexpr index_t kSubBlockM_1   = kWarpM_1 * kBlockWarpsM_1;
    static constexpr index_t kSubBlockN_1   = kWarpN_1 * kBlockWarpsN_1;
    static constexpr index_t kSubBlockK_1   = kWarpK_1 * kBlockWarpsK_1;
    static_assert(kBlockM_1 % kSubBlockM_1 == 0);
    static_assert(kBlockN_1 % kSubBlockN_1 == 0);
    static_assert(kBlockK_1 % kSubBlockK_1 == 0);
    static constexpr index_t kWarpRepeatM_1 = kBlockM_1 / kSubBlockM_1;
    static constexpr index_t kWarpRepeatN_1 = kBlockN_1 / kSubBlockN_1;
    static constexpr index_t kWarpRepeatK_1 = kBlockK_1 / kSubBlockK_1;

    // d, rowmajor : hidden*dim, colmajor : dim*hidden (vLLM use this layout)
    // static constexpr bool IsDLayoutRowMajor = IsDLayoutRowMajor_;
    // using DLayout                           = std::conditional_t<IsDLayoutRowMajor,
    //                                    ck_tile::tensor_layout::gemm::RowMajor,
    //                                    ck_tile::tensor_layout::gemm::ColumnMajor>;
};

} // namespace ck_tile
