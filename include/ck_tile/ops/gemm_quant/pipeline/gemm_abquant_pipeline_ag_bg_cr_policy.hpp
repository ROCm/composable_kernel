// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_policy.hpp"
#include "gemm_group_quant_utils.hpp"

namespace ck_tile {

struct GemmABQuantPipelineAgBgCrDefaultPolicy
    : public UniversalGemmBasePolicy<GemmABQuantPipelineAgBgCrDefaultPolicy>
{
    using Base = UniversalGemmBasePolicy<GemmABQuantPipelineAgBgCrDefaultPolicy>;
    using Base::I0;
    using Base::I1;
    using Base::I2;

    template <typename Problem>
    using LdsADataType = typename Problem::AComputeDataType;

    template <typename Problem>
    using LdsBDataType = typename Problem::BComputeDataType;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeAQ()
    {
        return GemmAQuantPipelineAgBgCrDefaultPolicy::GetVectorSizeAQ<Problem>();
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeAQDramTileDistribution()
    {
        return GemmAQuantPipelineAgBgCrDefaultPolicy::MakeAQDramTileDistribution<Problem>();
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeBQ()
    {
        return GemmBQuantPipelineAgBgCrDefaultPolicy::GetVectorSizeBQ<Problem>();
    }
    template <typename Problem, bool CompactPreshuffledN = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBQDramTileDistribution()
    {
        return GemmBQuantPipelineAgBgCrDefaultPolicy::
            MakeBQDramTileDistribution<Problem, CompactPreshuffledN>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;

        static_assert(Problem::BQuantGroupSize::kK % WarpTile::at(I2) == 0,
                      "KPerWarpGemm must be a multiple of QuantGroupSize::kK!");

#if defined(__gfx125__)
        // WMMA uses the native K0/K1 lane decomposition. The MFMA transpose-load
        // selector below assumes at most four accesses and yields Invalid for
        // gfx1250's 16x16x128 fp8/bf8 tile (64 elements per lane, 8 per access).
        constexpr auto wg_attr_num_access = WGAttrNumAccessEnum::Default;
#else
        constexpr index_t vector_size =
            DS_READ_TR_SIZE() / sizeof(typename Problem::AComputeDataType);
        constexpr index_t thread_elements = WarpTile::at(I1) * WarpTile::at(I2) / get_warp_size();
        constexpr auto wg_attr_num_access =
            !(Base::is_a_load_tr<Problem> || Base::is_b_load_tr<Problem>)
                ? WGAttrNumAccessEnum::Single
            : vector_size == thread_elements     ? WGAttrNumAccessEnum::Single
            : vector_size * 2 == thread_elements ? WGAttrNumAccessEnum::Double
            : vector_size * 4 == thread_elements ? WGAttrNumAccessEnum::Quad
                                                 : WGAttrNumAccessEnum::Invalid;
#endif

        using WarpGemm = WarpGemmDispatcher<typename Problem::AComputeDataType,
                                            typename Problem::BComputeDataType,
                                            typename Problem::CDataType,
                                            WarpTile::at(I0),
                                            WarpTile::at(I1),
                                            WarpTile::at(I2),
                                            Problem::TransposeC,
                                            false, // SwizzleA
                                            false, // UseStructuredSparsity
                                            wg_attr_num_access>;
        static_assert(std::is_same_v<typename Problem::AComputeDataType, fp8_t> ||
                      std::is_same_v<typename Problem::AComputeDataType, bf8_t>);
        static_assert(std::is_same_v<typename Problem::BComputeDataType, fp8_t> ||
                      std::is_same_v<typename Problem::BComputeDataType, bf8_t>);
        static_assert(std::is_same_v<typename Problem::CDataType, float>);

        using BlockGemmPolicy = BlockGemmASmemBSmemCRegV1CustomPolicy<typename Problem::ADataType,
                                                                      typename Problem::BDataType,
                                                                      typename Problem::CDataType,
                                                                      BlockWarps,
                                                                      WarpGemm>;
        return ABQuantBlockUniversalGemmAsBsCr<Problem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
