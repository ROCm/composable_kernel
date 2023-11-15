// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

namespace ck {
namespace tile_program {

template <typename BlockTile_, // Sequence<...
          typename Gemm0BlockWarps_,
          typename Gemm0WarpTile_,
          typename Gemm1BlockWarps_,
          typename Gemm1WarpTile_,
          typename VLayout_ = ck::tensor_layout::gemm::RowMajor>
struct TileFmhaShape
{
    using BlockTile       = remove_cvref_t<BlockTile_>;
    using Gemm0BlockWarps = remove_cvref_t<Gemm0BlockWarps_>;
    using Gemm0WarpTile   = remove_cvref_t<Gemm0WarpTile_>;
    using Gemm1BlockWarps = remove_cvref_t<Gemm1BlockWarps_>;
    using Gemm1WarpTile   = remove_cvref_t<Gemm1WarpTile_>;

    static constexpr index_t kM0 = BlockTile::At(Number<0>{}); // tile size along q seqlen
    static constexpr index_t kN0 = BlockTile::At(Number<1>{}); // tile size along k seqlen
    static constexpr index_t kK0 = BlockTile::At(Number<2>{}); // tile size along qk gemm unroll
    static constexpr index_t kN1 = BlockTile::At(Number<3>{}); // tile size along v head_dim
    static constexpr index_t kK1 = BlockTile::At(Number<4>{}); // tile size along kv gemm unroll
    static constexpr index_t kK0BlockLength =
        BlockTile::At(Number<5>{}); // total length of K0, used for pipeline that need load Q at
                                    // once (or repeately load Q as a whole tile)

    using VLayout = remove_cvref_t<VLayout_>; // rowmajor : seqlen*hdim, colmajor : hdim*seqlen
};

} // namespace tile_program
} // namespace ck
