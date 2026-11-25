// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

enum struct GemmLoopOrder
{
    KMN,
    MNK,
};

template <typename AType_,
          typename BType_,
          typename CType_,
          typename BlockWarps_,
          typename WarpGemm_,
          GemmLoopOrder BlockGemmLoopOrder_ = GemmLoopOrder::KMN,
          bool NWarpMajor_                  = false>
struct BlockGemmARegBRegCRegV2CustomPolicy
{
    using AType = remove_cvref_t<AType_>;
    using BType = remove_cvref_t<BType_>;
    using CType = remove_cvref_t<CType_>;

    using BlockWarps = remove_cvref_t<BlockWarps_>;

    static constexpr index_t kMWarps = BlockWarps::at(number<0>{});
    static constexpr index_t kNWarps = BlockWarps::at(number<1>{});
    // warps on K: each takes its own K range, and C holds every warp's partial sum
    // (replicated over the warps; the caller reduces them)
    static constexpr index_t kKWarps = BlockWarps::at(number<2>{});
    // warps on N outermost: warp w takes the contiguous columns [w, w + 1) * N / NWarp,
    // instead of every NWarp-th warp tile
    static constexpr bool kNWarpMajor = NWarpMajor_;

    using WarpGemm = remove_cvref_t<WarpGemm_>;

    static constexpr auto BlockGemmLoopOrder = BlockGemmLoopOrder_;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        return make_tuple(WarpGemm{}, kMWarps, kNWarps);
    }
};

} // namespace ck_tile
