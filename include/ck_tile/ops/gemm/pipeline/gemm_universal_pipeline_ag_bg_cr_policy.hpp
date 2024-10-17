// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"

namespace ck_tile {

// UniversalGemm Policy
template <typename LayoutA_, typename LayoutB_, typename LayoutC_>
struct UniversalGemmPipelineAgBgCrPolicy
{
    using LayoutA = remove_cvref_t<LayoutA_>;
    using LayoutB = remove_cvref_t<LayoutB_>;
    using LayoutC = remove_cvref_t<LayoutC_>;

    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};

    static constexpr bool TransposeC = true;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        using WarpGemm = WarpGemmMfmaDispatcher<typename Problem::ADataType,
                                                typename Problem::BDataType,
                                                typename Problem::CDataType,
                                                Problem::BlockGemmShape::WarpTile::at(I0),
                                                Problem::BlockGemmShape::WarpTile::at(I1),
                                                Problem::BlockGemmShape::WarpTile::at(I2),
                                                TransposeC>;

        using ADataType = remove_cvref_t<typename Problem::ADataType>;

        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t K1        = WarpGemm::kK;
        constexpr index_t K0        = KPerBlock / K1;

        if constexpr(std::is_same<tensor_layout::gemm::RowMajor, LayoutA>::value)
        {
            constexpr auto MLdsLayer        = 32 * 4 / KPerBlock / sizeof(ADataType) < 1
                                                  ? 1
                                                  : 32 * 4 / KPerBlock / sizeof(ADataType);
            constexpr auto a_lds_block_desc = make_naive_tensor_descriptor(
                make_tuple(K0 * number<MLdsLayer>{}, number<MPerBlock / MLdsLayer>{}, K1),
                make_tuple(K1, number<KPerBlock * MLdsLayer>{}, I1));

            constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
                a_lds_block_desc,
                make_tuple(make_xor_transform(make_tuple(number<MPerBlock / MLdsLayer>{},
                                                         number<K0 * MLdsLayer>{})),
                           make_pass_through_transform(K1)),
                make_tuple(sequence<1, 0>{}, sequence<2>{}),
                make_tuple(sequence<1, 0>{}, sequence<2>{}));

            constexpr auto a_lds_block_desc_ak0_kMLdsLayer_m_ak1 = transform_tensor_descriptor(
                a_lds_block_desc_permuted,
                make_tuple(make_unmerge_transform(make_tuple(K0, number<MLdsLayer>{})),
                           make_pass_through_transform(number<MPerBlock / MLdsLayer>{}),
                           make_pass_through_transform(K1)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

            constexpr auto a_lds_block_desc_m_k = transform_tensor_descriptor(
                a_lds_block_desc_ak0_kMLdsLayer_m_ak1,
                make_tuple(make_merge_transform_v3_division_mod(make_tuple(K0, K1)),
                           make_merge_transform_v3_division_mod(
                               make_tuple(number<MPerBlock / MLdsLayer>{}, number<MLdsLayer>{}))),
                make_tuple(sequence<0, 3>{}, sequence<1, 2>{}),
                make_tuple(sequence<1>{}, sequence<0>{}));

            return a_lds_block_desc_m_k;
        }
        else // ColumnMajor A
        {
            // kfold and mpair dimension is not always required.
            // more dimension in merge_transform increase the difficulty of generating immarg offset
            // for compiler.
            constexpr auto M0 = get_warp_size() * Problem::BlockGemmShape::BlockWarps::at(I0);
            constexpr auto M1 = MPerBlock / M0;

            constexpr auto KThreadWrite     = Problem::kBlockSize / M0;
            constexpr auto K0PerThreadWrite = K0 / KThreadWrite;
            constexpr auto KThreadRead      = 64 / WarpGemm::kM;
            constexpr auto K0PerThreadRead  = K0 / KThreadRead;

            constexpr auto kfold =
                (K1 * M0 * sizeof(ADataType) > 128) ? 1 : 128 / (K1 * M0 * sizeof(ADataType));
            constexpr auto KThreadReadPerm =
                (kfold * K0PerThreadWrite / K0PerThreadRead) > 1
                    ? KThreadRead / (kfold * K0PerThreadWrite / K0PerThreadRead)
                    : KThreadRead;

            // 1<=mpair<=kN0
            constexpr auto mpair = (K1 * WarpGemm::kM * sizeof(ADataType) > 128)
                                       ? 1
                                       : ((128 / (K1 * WarpGemm::kM * sizeof(ADataType))) > M0
                                              ? M0
                                              : 128 / (K1 * WarpGemm::kM * sizeof(ADataType)));

            constexpr auto a_lds_block_desc = make_naive_tensor_descriptor_packed(
                make_tuple(number<KThreadWrite / kfold / KThreadReadPerm>{},
                           number<K0PerThreadWrite>{},
                           number<KThreadReadPerm * M1>{},
                           number<kfold * M0 / mpair>{},
                           number<mpair>{},
                           K1));

            constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
                a_lds_block_desc,
                make_tuple(
                    make_pass_through_transform(number<KThreadWrite / kfold / KThreadReadPerm>{}),
                    make_pass_through_transform(number<K0PerThreadWrite>{}),
                    make_xor_transform(
                        make_tuple(number<KThreadReadPerm * M1>{}, number<kfold * M0 / mpair>{})),
                    make_pass_through_transform(number<mpair>{}),
                    make_pass_through_transform(K1)),
                make_tuple(
                    sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}),
                make_tuple(
                    sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}));

            constexpr auto a_lds_block_desc_unmerged = transform_tensor_descriptor(
                a_lds_block_desc_permuted,
                make_tuple(
                    make_pass_through_transform(number<KThreadWrite / kfold / KThreadReadPerm>{}),
                    make_pass_through_transform(number<K0PerThreadWrite>{}),
                    make_unmerge_transform(make_tuple(number<KThreadReadPerm>{}, number<M1>{})),
                    make_unmerge_transform(make_tuple(number<kfold>{}, number<M0 / mpair>{})),
                    make_pass_through_transform(number<mpair>{}),
                    make_pass_through_transform(K1)),
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

            constexpr auto a_lds_block_desc_m_k = transform_tensor_descriptor(
                a_lds_block_desc_unmerged,
                make_tuple(make_merge_transform_v3_division_mod(
                               make_tuple(number<KThreadReadPerm>{},
                                          number<KThreadWrite / kfold / KThreadReadPerm>{},
                                          number<kfold>{},
                                          number<K0PerThreadWrite>{},
                                          K1)),
                           make_merge_transform_v3_division_mod(
                               make_tuple(number<M0 / mpair>{}, number<mpair>{}, number<M1>{}))),
                make_tuple(sequence<0, 1, 4, 2, 7>{}, sequence<5, 6, 3>{}),
                make_tuple(sequence<1>{}, sequence<0>{}));

            return a_lds_block_desc_m_k;
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        using WarpGemm = WarpGemmMfmaDispatcher<typename Problem::ADataType,
                                                typename Problem::BDataType,
                                                typename Problem::CDataType,
                                                Problem::BlockGemmShape::WarpTile::at(I0),
                                                Problem::BlockGemmShape::WarpTile::at(I1),
                                                Problem::BlockGemmShape::WarpTile::at(I2),
                                                TransposeC>;

        using BDataType = remove_cvref_t<typename Problem::BDataType>;

        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t K1 = WarpGemm::kK;
        constexpr index_t K0 = KPerBlock / K1;

        if constexpr(std::is_same<tensor_layout::gemm::ColumnMajor, LayoutB>::value)
        {
            // NLdsLayer * K0 as logical Bank
            constexpr auto NLdsLayer = 32 * 4 / KPerBlock / sizeof(BDataType) < 1
                                           ? 1
                                           : 32 * 4 / KPerBlock / sizeof(BDataType);
            ;
            constexpr auto b_lds_block_desc = make_naive_tensor_descriptor(
                make_tuple(K0 * number<NLdsLayer>{}, number<NPerBlock / NLdsLayer>{}, K1),
                make_tuple(K1, number<KPerBlock * NLdsLayer>{}, I1));

            constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
                b_lds_block_desc,
                make_tuple(make_xor_transform(make_tuple(number<NPerBlock / NLdsLayer>{},
                                                         number<K0 * NLdsLayer>{})),
                           make_pass_through_transform(K1)),
                make_tuple(sequence<1, 0>{}, sequence<2>{}),
                make_tuple(sequence<1, 0>{}, sequence<2>{}));

            constexpr auto b_lds_block_desc_bk0_kNLdsLayer_n_bk1 = transform_tensor_descriptor(
                b_lds_block_desc_permuted,
                make_tuple(make_unmerge_transform(make_tuple(K0, number<NLdsLayer>{})),
                           make_pass_through_transform(number<NPerBlock / NLdsLayer>{}),
                           make_pass_through_transform(K1)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

            constexpr auto b_lds_block_desc_n_k = transform_tensor_descriptor(
                b_lds_block_desc_bk0_kNLdsLayer_n_bk1,
                make_tuple(make_merge_transform_v3_division_mod(make_tuple(K0, K1)),
                           make_merge_transform_v3_division_mod(
                               make_tuple(number<NPerBlock / NLdsLayer>{}, number<NLdsLayer>{}))),
                make_tuple(sequence<0, 3>{}, sequence<1, 2>{}),
                make_tuple(sequence<1>{}, sequence<0>{}));

            return b_lds_block_desc_n_k;
        }
        else // RowMajor B
        {
            constexpr auto N0 = get_warp_size() * Problem::BlockGemmShape::BlockWarps::at(I1);
            constexpr auto N1 = NPerBlock / N0;

            constexpr auto KThreadWrite     = Problem::kBlockSize / N0;
            constexpr auto K0PerThreadWrite = K0 / KThreadWrite;
            constexpr auto KThreadRead      = 64 / WarpGemm::kN;
            constexpr auto K0PerThreadRead  = K0 / KThreadRead;

            constexpr auto kfold =
                (K1 * N0 * sizeof(BDataType) > 128) ? 1 : 128 / (K1 * N0 * sizeof(BDataType));
            constexpr auto KThreadReadPerm =
                (kfold * K0PerThreadWrite / K0PerThreadRead) > 1
                    ? KThreadRead / (kfold * K0PerThreadWrite / K0PerThreadRead)
                    : KThreadRead;

            // 1<=npair<=kN0
            constexpr auto npair = (K1 * WarpGemm::kN * sizeof(BDataType) > 128)
                                       ? 1
                                       : ((128 / (K1 * WarpGemm::kN * sizeof(BDataType))) > N0
                                              ? N0
                                              : 128 / (K1 * WarpGemm::kN * sizeof(BDataType)));

            constexpr auto b_lds_block_desc = make_naive_tensor_descriptor_packed(
                make_tuple(number<KThreadWrite / kfold / KThreadReadPerm>{},
                           number<K0PerThreadWrite>{},
                           number<KThreadReadPerm * N1>{},
                           number<kfold * N0 / npair>{},
                           number<npair>{},
                           K1));

            constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
                b_lds_block_desc,
                make_tuple(
                    make_pass_through_transform(number<KThreadWrite / kfold / KThreadReadPerm>{}),
                    make_pass_through_transform(number<K0PerThreadWrite>{}),
                    make_xor_transform(
                        make_tuple(number<KThreadReadPerm * N1>{}, number<kfold * N0 / npair>{})),
                    make_pass_through_transform(number<npair>{}),
                    make_pass_through_transform(K1)),
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
                    make_pass_through_transform(K1)),
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

            constexpr auto b_lds_block_desc_n_k = transform_tensor_descriptor(
                b_lds_block_desc_unmerged,
                make_tuple(make_merge_transform_v3_division_mod(
                               make_tuple(number<KThreadReadPerm>{},
                                          number<KThreadWrite / kfold / KThreadReadPerm>{},
                                          number<kfold>{},
                                          number<K0PerThreadWrite>{},
                                          K1)),
                           make_merge_transform_v3_division_mod(
                               make_tuple(number<N0 / npair>{}, number<npair>{}, number<N1>{}))),
                make_tuple(sequence<0, 1, 4, 2, 7>{}, sequence<5, 6, 3>{}),
                make_tuple(sequence<1>{}, sequence<0>{}));

            return b_lds_block_desc_n_k;
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeA()
    {
        constexpr index_t smem_size_a = sizeof(typename Problem::ADataType) *
                                        MakeALdsBlockDescriptor<Problem>().get_element_space_size();
        return smem_size_a;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeB()
    {
        constexpr index_t smem_size_b = sizeof(typename Problem::BDataType) *
                                        MakeBLdsBlockDescriptor<Problem>().get_element_space_size();
        return smem_size_b;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        constexpr index_t smem_size_a = GetSmemSizeA<Problem>();
        constexpr index_t smem_size_b = GetSmemSizeB<Problem>();
        index_t smem_size             = 0;
        smem_size += smem_size_a + smem_size_b;

        return smem_size;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeADramTileDistribution()
    {
        using WarpGemm = WarpGemmMfmaDispatcher<typename Problem::ADataType,
                                                typename Problem::BDataType,
                                                typename Problem::CDataType,
                                                Problem::BlockGemmShape::WarpTile::at(I0),
                                                Problem::BlockGemmShape::WarpTile::at(I1),
                                                Problem::BlockGemmShape::WarpTile::at(I2),
                                                TransposeC>;

        constexpr index_t BlockSize = Problem::kBlockSize;

        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t K1 = WarpGemm::kK;
        constexpr index_t K0 = KPerBlock / K1;
        constexpr index_t M2 = get_warp_size() / K0;

        constexpr index_t M1 = BlockSize / get_warp_size();
        static_assert(M2 != 0, "M2 is zero, which will lead to a division by zero error.");
        static_assert(M1 != 0, "M1 is zero, which will lead to a division by zero error.");
        constexpr index_t M0 = MPerBlock / (M2 * M1);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBDramTileDistribution()
    {
        using WarpGemm = WarpGemmMfmaDispatcher<typename Problem::ADataType,
                                                typename Problem::BDataType,
                                                typename Problem::CDataType,
                                                Problem::BlockGemmShape::WarpTile::at(I0),
                                                Problem::BlockGemmShape::WarpTile::at(I1),
                                                Problem::BlockGemmShape::WarpTile::at(I2),
                                                TransposeC>;

        constexpr index_t BlockSize = Problem::kBlockSize;

        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t K1 = WarpGemm::kK;
        constexpr index_t K0 = KPerBlock / K1;
        constexpr index_t N2 = get_warp_size() / K0;

        constexpr index_t N1 = BlockSize / get_warp_size();
        static_assert(N2 != 0, "M2 is zero, which will lead to a division by zero error.");
        static_assert(N1 != 0, "M1 is zero, which will lead to a division by zero error.");
        constexpr index_t N0 = NPerBlock / (N2 * N1);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using AccDataType     = float;
        using BlockWarps      = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile        = typename Problem::BlockGemmShape::WarpTile;
        using WarpGemm        = WarpGemmMfmaDispatcher<typename Problem::ADataType,
                                                typename Problem::BDataType,
                                                AccDataType,
                                                WarpTile::at(I0),
                                                WarpTile::at(I1),
                                                WarpTile::at(I2),
                                                TransposeC>;
        using BlockGemmPolicy = BlockGemmASmemBSmemCRegV1CustomPolicy<typename Problem::ADataType,
                                                                      typename Problem::BDataType,
                                                                      typename Problem::CDataType,
                                                                      BlockWarps,
                                                                      WarpGemm>;
        return BlockGemmASmemBSmemCRegV1<Problem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
