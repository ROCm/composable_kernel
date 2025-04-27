// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp"

namespace ck_tile {
struct MoeGemmPipelineAgBgCrPolicy : public UniversalFlatmmPipelineAgBgCrPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGlobalTileDistribution_C()
    {
        using S_       = remove_cvref_t<typename Problem::BlockShape>;
        using WarpGemm = remove_cvref_t<typename Problem::W>;
        // using CDataType = typename WarpGemm::CDataType;

        constexpr auto c_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<S_::Repeat_M1, S_::WarpPerBlock_M1>,
                                             sequence<S_::Repeat_N1, S_::WarpPerBlock_N1>>,
                                       tuple<sequence<1, 2>>,
                                       tuple<sequence<1, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WarpGemm::CWarpDstrEncoding{});
        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);
        return c_block_dstr;
    }
};
} // namespace ck_tile


