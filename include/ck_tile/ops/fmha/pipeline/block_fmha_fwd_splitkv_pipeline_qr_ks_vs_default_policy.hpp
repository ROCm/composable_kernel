// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1.hpp"

namespace ck_tile {

// This pipeline is qkv all located in LDS
struct BlockFmhaFwdSplitKVPipelineQRKSVSDefaultPolicy
    : BlockFmhaPipelineQXKSVSCustomPolicy</* QLoadOnce = */ true,
                                          /* AsyncCopyK = */ false,
                                          /* AsyncCopyV = */ false,
                                          /* NumPrefetchK = */ 1,
                                          /* NumPrefetchV = */ 1>
{
    using BasePolicy = BlockFmhaPipelineQXKSVSCustomPolicy</* QLoadOnce = */ true,
                                                           /* AsyncCopyK = */ false,
                                                           /* AsyncCopyV = */ false,
                                                           /* NumPrefetchK = */ 1,
                                                           /* NumPrefetchV = */ 1>;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentOacc()
    {
        using OaccDataType = remove_cvref_t<typename Problem::OaccDataType>;

        return static_cast<index_t>(16 / sizeof(OaccDataType));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackP()
    {
        // TODO: this is for 3d layout
        using PDataType = remove_cvref_t<typename Problem::PDataType>;
        return static_cast<index_t>(16 / sizeof(PDataType));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakePLdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPack     = GetSmemKPackP<Problem>();

        constexpr auto k_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kKPerBlock / kKPack>{}, number<kMPerBlock>{}, number<kKPack>{}),
            make_tuple(number<(kMPerBlock + 1) * kKPack>{}, number<kKPack>{}, number<1>{}),
            number<8>{},
            number<1>{});

        constexpr auto k_lds_block_desc = transform_tensor_descriptor(
            k_lds_block_desc_0,
            make_tuple(
                make_pass_through_transform(number<kMPerBlock>{}),
                make_merge_transform(make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return k_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeMLdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;

        constexpr auto k_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kMPerBlock>{}), make_tuple(number<1>{}), number<8>{}, number<1>{});

        return k_lds_block_desc_0;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeP()
    {
        return MakePLdsBlockDescriptor<Problem>().get_element_space_size() *
               sizeof(typename Problem::PDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeM()
    {
        return MakeMLdsBlockDescriptor<Problem>().get_element_space_size() *
               sizeof(typename Problem::SMPLComputeDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return BasePolicy::template GetSmemSizeKV<Problem>() + GetSmemSizeP<Problem>() +
               GetSmemSizeM<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetKVBlockGemm()
    {
        using GemmProblem =
            BlockGemmProblem<typename Problem::PDataType,
                             typename Problem::VDataType,
                             typename Problem::OaccDataType,
                             Problem::kNumGemm1Warps * get_warp_size(),
                             TileGemmShape<sequence<Problem::BlockFmhaShape::kM0,
                                                    Problem::BlockFmhaShape::kN1,
                                                    Problem::BlockFmhaShape::kK1>,
                                           typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                                           typename Problem::BlockFmhaShape::Gemm1WarpTile>>;

        auto warp_gemm = [&]() {
            if constexpr(std::is_same_v<typename Problem::KDataType, fp8_t> &&
                         std::is_same_v<typename Problem::VDataType, fp8_t> &&
                         std::is_same_v<typename Problem::OaccDataType, float>)
            {
                return WarpGemmMfmaFp8Fp8F32M32N32K16SwizzleBTransposedCDistribution<>{};
                // return
                // WarpGemmImpl<WarpGemmAtrributeMfmaTransposedCDistribution_SwizzleB<
                //         WarpGemmAttributeMfmaImpl_f32_32x32x16_f8_base<typename
                //         Problem::PDataType, typename Problem::VDataType>>>{};
            }
            else
            {
                return WarpGemmMfmaDispatcher<
                    typename Problem::PDataType,
                    typename Problem::VDataType,
                    typename Problem::OaccDataType,
                    Problem::BlockFmhaShape::Gemm1WarpTile::at(number<0>{}),
                    Problem::BlockFmhaShape::Gemm1WarpTile::at(number<1>{}),
                    Problem::BlockFmhaShape::Gemm1WarpTile::at(number<2>{}),
                    false>{};
            }
        }();

        using WarpGemm = remove_cvref_t<decltype(warp_gemm)>;

        using BlockGemmPolicy =
            BlockGemmASmemBSmemCRegV1CustomPolicy<typename Problem::PDataType,
                                                  typename Problem::VDataType,
                                                  typename Problem::OaccDataType,
                                                  typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                                                  WarpGemm>;
        return BlockGemmASmemBSmemCRegV1<GemmProblem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
