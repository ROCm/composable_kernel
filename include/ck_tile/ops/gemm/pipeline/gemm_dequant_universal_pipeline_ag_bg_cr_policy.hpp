// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"

namespace ck_tile {

// UniversalGemm Policy
struct UniversalGemmDequantPipelineAgBgCrPolicy
    : public UniversalGemmBasePolicy<UniversalGemmPipelineAgBgCrPolicy>
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        using ADataType = remove_cvref_t<typename Problem::ADataType>;

        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t KPack     = GetSmemPackA<Problem>();

        constexpr auto DataTypeSize = sizeof(ADataType);
        constexpr auto MLdsLayer =
            (32 * 4 / KPerBlock / DataTypeSize) < 1 ? 1 : (32 * 4 / KPerBlock / DataTypeSize);

        constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<KPerBlock / KPack * MLdsLayer>{},
                       number<MPerBlock / MLdsLayer>{},
                       number<KPack>{}),
            make_tuple(number<KPack>{}, number<KPerBlock * MLdsLayer>{}, number<1>{}),
            number<KPack>{},
            number<1>{});

        constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
            a_lds_block_desc_0,
            make_tuple(make_xor_transform(make_tuple(number<MPerBlock / MLdsLayer>{},
                                                     number<KPerBlock / KPack * MLdsLayer>{})),
                       make_pass_through_transform(number<KPack>{})),
            make_tuple(sequence<1, 0>{}, sequence<2>{}),
            make_tuple(sequence<1, 0>{}, sequence<2>{}));

        constexpr auto a_lds_block_desc_xk0_mnldslayer_mn_xk1 = transform_tensor_descriptor(
            a_lds_block_desc_permuted,
            make_tuple(make_unmerge_transform(
                           make_tuple(number<KPerBlock / KPack>{}, number<MLdsLayer>{})),
                       make_pass_through_transform(number<MPerBlock / MLdsLayer>{}),
                       make_pass_through_transform(number<KPack>{})),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
            make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

        constexpr auto a_lds_block_desc = transform_tensor_descriptor(
            a_lds_block_desc_xk0_mnldslayer_mn_xk1,
            make_tuple(make_merge_transform_v3_division_mod(
                           make_tuple(number<MPerBlock / MLdsLayer>{}, number<MLdsLayer>{})),
                       make_merge_transform_v3_division_mod(
                           make_tuple(number<KPerBlock / KPack>{}, number<KPack>{}))),
            make_tuple(sequence<1, 2>{}, sequence<0, 3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return a_lds_block_desc;
    }

    /**
     * @brief Create LDS block descriptor for B tensor.
     *
     * @tparam Problem  Gemm pipeline problem.
     * @return B tensor LDS block descriptor.
     */
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        // using BLayout   = remove_cvref_t<typename Problem::BLayout>;
        using BDataType = remove_cvref_t<typename Problem::BDataType>;

        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

#if 1
        // if constexpr(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::ColumnMajor>)
        {
            constexpr index_t KPack     = GetSmemPackB<Problem>();
            constexpr auto BK0          = number<KPerBlock / KPack>{};
            constexpr auto DataTypeSize = sizeof(BDataType);
            constexpr auto NLdsLayer =
                (32 * 4 / KPerBlock / DataTypeSize) < 1 ? 1 : (32 * 4 / KPerBlock / DataTypeSize);

            constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(
                    BK0 * number<NLdsLayer>{}, number<NPerBlock / NLdsLayer>{}, number<KPack>{}),
                make_tuple(number<KPack>{}, number<KPerBlock * NLdsLayer>{}, number<1>{}),
                number<KPack>{},
                number<1>{});

            constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
                b_lds_block_desc_0,
                make_tuple(make_xor_transform(make_tuple(number<NPerBlock / NLdsLayer>{},
                                                         BK0 * number<NLdsLayer>{})),
                           make_pass_through_transform(number<KPack>{})),
                make_tuple(sequence<1, 0>{}, sequence<2>{}),
                make_tuple(sequence<1, 0>{}, sequence<2>{}));

            constexpr auto b_lds_block_desc_bk0_nldslayer_n_bk1 = transform_tensor_descriptor(
                b_lds_block_desc_permuted,
                make_tuple(make_unmerge_transform(make_tuple(BK0, number<NLdsLayer>{})),
                           make_pass_through_transform(number<NPerBlock / NLdsLayer>{}),
                           make_pass_through_transform(number<KPack>{})),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

            constexpr auto b_lds_block_desc = transform_tensor_descriptor(
                b_lds_block_desc_bk0_nldslayer_n_bk1,
                make_tuple(make_merge_transform_v3_division_mod(
                               make_tuple(number<NPerBlock / NLdsLayer>{}, number<NLdsLayer>{})),
                           make_merge_transform_v3_division_mod(make_tuple(BK0, number<KPack>{}))),
                make_tuple(sequence<1, 2>{}, sequence<0, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
            return b_lds_block_desc;
        }
#else
        else // B is Row Major
        {
            constexpr index_t BlockSize   = Problem::kBlockSize;
            constexpr index_t VecLoadSize = GetVectorSizeB<Problem>();
            using TileEncodingPattern     = TileDistributionEncodingPattern2D<BlockSize,
                                                                          KPerBlock,
                                                                          NPerBlock,
                                                                          VecLoadSize,
                                                                          BTileAccessPattern>;

            constexpr auto BK0 = number<TileEncodingPattern::X1>{};
            constexpr auto BK1 = number<TileEncodingPattern::Y0>{};
            // constexpr auto N0 = BBlockTransferThreadClusterLengths_BK0_N_BK1{}.At(I1);
            constexpr auto N0 = TileEncodingPattern::X0;
            constexpr auto N1 = NPerBlock / N0;

            using WarpTile         = typename Problem::BlockGemmShape::WarpTile;
            constexpr auto NPerXdl = number<WarpTile::at(I1)>{};

            // constexpr auto KThreadWrite     =
            // BBlockTransferThreadClusterLengths_BK0_N_BK1{}.At(I0);
            constexpr auto KThreadWrite     = TileEncodingPattern::Y2;
            constexpr auto K0PerThreadWrite = BK0 / KThreadWrite;
            constexpr auto KThreadRead      = 64 / NPerXdl;
            constexpr auto K0PerThreadRead  = BK0 / KThreadRead;

            constexpr auto kfold =
                (BK1 * N0 * sizeof(BDataType) > 128) ? 1 : 128 / (BK1 * N0 * sizeof(BDataType));
            constexpr auto KThreadReadPerm =
                (kfold * K0PerThreadWrite / K0PerThreadRead) > 1
                    ? KThreadRead / (kfold * K0PerThreadWrite / K0PerThreadRead)
                    : KThreadRead;

            // 1<=npair<=n0
            constexpr auto npair = (BK1 * NPerXdl * sizeof(BDataType) > 128)
                                       ? 1
                                       : ((128 / (BK1 * NPerXdl * sizeof(BDataType))) > N0
                                              ? N0
                                              : 128 / (BK1 * NPerXdl * sizeof(BDataType)));

            constexpr auto b_lds_block_desc = make_naive_tensor_descriptor_packed(
                make_tuple(number<KThreadWrite / kfold / KThreadReadPerm>{},
                           number<K0PerThreadWrite>{},
                           number<KThreadReadPerm * N1>{},
                           number<kfold * N0 / npair>{},
                           number<npair>{},
                           BK1));

            constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
                b_lds_block_desc,
                make_tuple(
                    make_pass_through_transform(number<KThreadWrite / kfold / KThreadReadPerm>{}),
                    make_pass_through_transform(number<K0PerThreadWrite>{}),
                    make_xor_transform(
                        make_tuple(number<KThreadReadPerm * N1>{}, number<kfold * N0 / npair>{})),
                    make_pass_through_transform(number<npair>{}),
                    make_pass_through_transform(BK1)),
                make_tuple(
                    sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}),
                make_tuple(
                    sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}));

            constexpr auto b_lds_block_desc_unmerged = transform_tensor_descriptor(
                b_lds_block_desc_permuted,
                make_tuple(
                    make_pass_through_transform(number<KThreadWrite / kfold / KThreadReadPerm>{}),
                    make_pass_through_transform(number<K0PerThreadWrite>{}),
                    make_unmerge_transform(make_tuple(number<KThreadReadPerm>{}, number<N1>{})),
                    make_unmerge_transform(make_tuple(number<kfold>{}, number<N0 / npair>{})),
                    make_pass_through_transform(number<npair>{}),
                    make_pass_through_transform(BK1)),
                make_tuple(sequence<0>{},
                           sequence<1>{},
                           sequence<2>{},
                           sequence<3>{},
                           sequence<4>{},
                           sequence<5>{}),
                make_tuple(sequence<1>{},
                           sequence<2>{},
                           sequence<0, 3>{},
                           sequence<4, 5>{},
                           sequence<6>{},
                           sequence<7>{}));

            // constexpr auto b_lds_block_desc_bk0_n_bk1 = transform_tensor_descriptor(
            //     b_lds_block_desc_unmerged,
            //     make_tuple(make_merge_transform_v3_division_mod(
            //                    make_tuple(number<KThreadReadPerm>{},
            //                               number<KThreadWrite / kfold / KThreadReadPerm>{},
            //                               number<kfold>{},
            //                               number<K0PerThreadWrite>{})),
            //                make_merge_transform_v3_division_mod(
            //                    make_tuple(number<N0 / npair>{}, number<npair>{}, number<N1>{})),
            //                make_pass_through_transform(BK1)),
            //     make_tuple(sequence<0, 1, 4, 2>{}, sequence<5, 6, 3>{}, sequence<7>{}),
            //     make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

            constexpr auto b_lds_block_desc_kn = transform_tensor_descriptor(
                b_lds_block_desc_unmerged,
                make_tuple(make_merge_transform_v3_division_mod(
                               make_tuple(number<KThreadReadPerm>{},
                                          number<KThreadWrite / kfold / KThreadReadPerm>{},
                                          number<kfold>{},
                                          number<K0PerThreadWrite>{},
                                          BK1)),
                           make_merge_transform_v3_division_mod(
                               make_tuple(number<N0 / npair>{}, number<npair>{}, number<N1>{}))),
                make_tuple(sequence<0, 1, 4, 2, 7>{}, sequence<5, 6, 3>{}),
                make_tuple(sequence<1>{}, sequence<0>{}));

            // return b_lds_block_desc_bk0_n_bk1;
            return b_lds_block_desc_kn;

            // constexpr auto b_lds_block_desc_bk0_n_bk1 = make_naive_tensor_descriptor(
            //     make_tuple(BK0, number<NPerBlock>{}, number<KPack>{}),
            //     make_tuple(number<KPack>{}, number<KPerBlock>{}, number<1>{}),
            //     number<KPack>{},
            //     number<1>{});

            // constexpr auto b_lds_block_desc = transform_tensor_descriptor(
            //     b_lds_block_desc_bk0_n_bk1,
            //     make_tuple(make_pass_through_transform(number<NPerBlock>{}),
            //                make_merge_transform_v3_division_mod(make_tuple(BK0,
            //                number<KPack>{}))),
            //     make_tuple(sequence<1>{}, sequence<0, 2>{}),
            //     make_tuple(sequence<0>{}, sequence<1>{}));

            // return b_lds_block_desc;
        }
#endif
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using BlockWarps      = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile        = typename Problem::BlockGemmShape::WarpTile;
        using WarpGemm        = WarpGemmMfmaDispatcher<typename Problem::ComputeDataType,
                                                typename Problem::ComputeDataType,
                                                typename Problem::CDataType,
                                                WarpTile::at(I0),
                                                WarpTile::at(I1),
                                                WarpTile::at(I2),
                                                Problem::TransposeC>;
        using BlockGemmPolicy = BlockGemmASmemBSmemCRegV1CustomPolicy<typename Problem::ADataType,
                                                                      typename Problem::BDataType,
                                                                      typename Problem::CDataType,
                                                                      BlockWarps,
                                                                      WarpGemm>;
        return BlockUniversalGemmDequantAsBsCr<Problem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
