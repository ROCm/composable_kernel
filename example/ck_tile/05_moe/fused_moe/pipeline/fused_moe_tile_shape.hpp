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
template <typename BlockTile_, // sequence<M_a, N_g, N_u, K_a, N_d
          typename Gemm0BlockWarps_,
          typename Gemm0WarpTile_,
          typename Gemm1BlockWarps_,
          typename Gemm1WarpTile_,
          bool IsDLayoutRowMajor_>
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
    static constexpr index_t kN_u = BlockTile::at(number<2>{});
    static constexpr index_t kK_a = BlockTile::at(number<3>{});
    static constexpr index_t kN_d = BlockTile::at(number<4>{});
    static_assert(kN_g == kN_u);
    static constexpr index_t kK_y = kN_g;

    static constexpr index_t kM_0 = kM_a;
    static constexpr index_t kN_0 = kN_g; // note N will x2
    static constexpr index_t kK_0 = kK_a;

    static constexpr index_t kM_1 = kM_0;
    static constexpr index_t kN_1 = kN_d;
    static constexpr index_t kK_1 = kN_g;

    // d, rowmajor : hidden*dim, colmajor : dim*hidden (vLLM use this layout)
    static constexpr bool IsDLayoutRowMajor = IsDLayoutRowMajor_;
    using DLayout                           = std::conditional_t<IsDLayoutRowMajor,
                                       ck_tile::tensor_layout::gemm::RowMajor,
                                       ck_tile::tensor_layout::gemm::ColumnMajor>;
};

} // namespace ck_tile
