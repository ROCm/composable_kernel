// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"

namespace ck_tile {
// Default policy for GemmPipelineAGmemBGmemCregComputeV4, except the block gemm method, it shares
// the same vector size implementation, SmemSize, Global memory tile distiribution as the
// UniversalGemm Pipeline Policy.
// Default policy class should not be templated, put template on
// member functions instead.
struct GemmPipelineAgBgCrCompV4DefaultPolicy
    : public UniversalGemmBasePolicy<GemmPipelineAgBgCrCompV4DefaultPolicy>
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        // using AccDataType     = float;
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;

        constexpr bool single_load_tr_length =
            (DS_READ_TR_SIZE() / sizeof(typename Problem::ComputeDataType)) ==
            (WarpTile::at(I1) * WarpTile::at(I2) / get_warp_size());
        constexpr auto wg_attr_num_access =
            ((is_a_load_tr<Problem> || is_b_load_tr<Problem>) && !single_load_tr_length)
                ? WGAttrNumAccessEnum::Double
                : WGAttrNumAccessEnum::Single;

        using WarpGemm        = WarpGemmMfmaDispatcher<typename Problem::ADataType,
                                                       typename Problem::BDataType,
                                                       typename Problem::CDataType, // AccDataType
                                                       WarpTile::at(I0),
                                                       WarpTile::at(I1),
                                                       WarpTile::at(I2),
                                                       Problem::TransposeC,
                                                       false,
                                                       false,
                                                       wg_attr_num_access>;
        using BlockGemmPolicy = BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::ADataType,
                                                                    typename Problem::BDataType,
                                                                    typename Problem::CDataType,
                                                                    BlockWarps,
                                                                    WarpGemm>;

        return BlockGemmARegBRegCRegV1<Problem, BlockGemmPolicy>{};
    }
};
} // namespace ck_tile
