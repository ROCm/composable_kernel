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

        static_assert(Problem::BlockGemmShape::BlockWarps::at(I2) == 1,
                      "Assume there is only 1 warp among K dimension");

        using ADataType = remove_cvref_t<typename Problem::ADataType>;

        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t KPerWarp     = WarpGemm::kK;
        constexpr index_t KIterPerWarp = KPerBlock / KPerWarp;
        static_assert(KPerBlock == KIterPerWarp * KPerWarp);

        if constexpr(std::is_same<tensor_layout::gemm::RowMajor, LayoutA>::value)
        {
            constexpr auto MLdsLayer        = 32 * 4 / KPerBlock / sizeof(ADataType) < 1
                                                  ? 1
                                                  : 32 * 4 / KPerBlock / sizeof(ADataType);
            constexpr auto a_lds_block_desc = make_naive_tensor_descriptor(
                make_tuple(
                    KIterPerWarp * number<MLdsLayer>{}, number<MPerBlock / MLdsLayer>{}, KPerWarp),
                make_tuple(KPerWarp, number<KPerBlock * MLdsLayer>{}, I1));

            constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
                a_lds_block_desc,
                make_tuple(make_xor_transform(make_tuple(number<MPerBlock / MLdsLayer>{},
                                                         number<KIterPerWarp * MLdsLayer>{})),
                           make_pass_through_transform(KPerWarp)),
                make_tuple(sequence<1, 0>{}, sequence<2>{}),
                make_tuple(sequence<1, 0>{}, sequence<2>{}));

            constexpr auto a_lds_block_desc_ak0_kMLdsLayer_m_ak1 = transform_tensor_descriptor(
                a_lds_block_desc_permuted,
                make_tuple(make_unmerge_transform(make_tuple(KIterPerWarp, number<MLdsLayer>{})),
                           make_pass_through_transform(number<MPerBlock / MLdsLayer>{}),
                           make_pass_through_transform(KPerWarp)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

            constexpr auto a_lds_block_desc_m_k = transform_tensor_descriptor(
                a_lds_block_desc_ak0_kMLdsLayer_m_ak1,
                make_tuple(make_merge_transform_v3_division_mod(make_tuple(KIterPerWarp, KPerWarp)),
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
            constexpr auto MThreads = get_warp_size() * Problem::BlockGemmShape::BlockWarps::at(I0);
            static_assert(MThreads <= MPerBlock,
                          "Make sure GEMM M tile (BlockTile[0]) is greater than or equal to "
                          "(get_warp_size() * BlockWarps[0])");

            constexpr auto MPerThread = MPerBlock / MThreads;
            static_assert(MPerBlock == MThreads * MPerThread);

            constexpr auto KPerThreadForWrite  = Problem::kBlockSize / MThreads;
            constexpr auto K0PerThreadForWrite = KIterPerWarp / KPerThreadForWrite;
            constexpr auto KPerThreadForRead   = get_warp_size() / WarpGemm::kM;
            constexpr auto K0PerThreadForRead  = KIterPerWarp / KPerThreadForRead;

            static_assert(KPerThreadForRead <= KIterPerWarp,
                          "GEMM M warp tile size (WarpTile[0]) is too small");

            static_assert(KIterPerWarp == K0PerThreadForWrite * KPerThreadForWrite);
            static_assert(KIterPerWarp == K0PerThreadForRead * KPerThreadForRead);

            // # bytes per 32 LDS banks: 32 * 4 bytes
            constexpr auto BankLength = 128;

            constexpr auto kfold = (KPerWarp * MThreads * sizeof(ADataType) > BankLength)
                                       ? 1
                                       : BankLength / (KPerWarp * MThreads * sizeof(ADataType));
            constexpr auto KPerThreadForReadPerm =
                (kfold * K0PerThreadForWrite / K0PerThreadForRead) > 1
                    ? KPerThreadForRead / (kfold * K0PerThreadForWrite / K0PerThreadForRead)
                    : KPerThreadForRead;

            // 1<=mpair<=kN0
            constexpr auto mpair =
                (KPerWarp * WarpGemm::kM * sizeof(ADataType) > BankLength)
                    ? 1
                    : ((BankLength / (KPerWarp * WarpGemm::kM * sizeof(ADataType))) > MThreads
                           ? MThreads
                           : BankLength / (KPerWarp * WarpGemm::kM * sizeof(ADataType)));

            constexpr auto a_lds_block_desc = make_naive_tensor_descriptor_packed(
                make_tuple(number<KPerThreadForWrite / kfold / KPerThreadForReadPerm>{},
                           number<K0PerThreadForWrite>{},
                           number<KPerThreadForReadPerm * MPerThread>{},
                           number<kfold * MThreads / mpair>{},
                           number<mpair>{},
                           KPerWarp));

            constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
                a_lds_block_desc,
                make_tuple(
                    make_pass_through_transform(
                        number<KPerThreadForWrite / kfold / KPerThreadForReadPerm>{}),
                    make_pass_through_transform(number<K0PerThreadForWrite>{}),
                    make_xor_transform(make_tuple(number<KPerThreadForReadPerm * MPerThread>{},
                                                  number<kfold * MThreads / mpair>{})),
                    make_pass_through_transform(number<mpair>{}),
                    make_pass_through_transform(KPerWarp)),
                make_tuple(
                    sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}),
                make_tuple(
                    sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}));

            constexpr auto a_lds_block_desc_unmerged = transform_tensor_descriptor(
                a_lds_block_desc_permuted,
                make_tuple(
                    make_pass_through_transform(
                        number<KPerThreadForWrite / kfold / KPerThreadForReadPerm>{}),
                    make_pass_through_transform(number<K0PerThreadForWrite>{}),
                    make_unmerge_transform(
                        make_tuple(number<KPerThreadForReadPerm>{}, number<MPerThread>{})),
                    make_unmerge_transform(make_tuple(number<kfold>{}, number<MThreads / mpair>{})),
                    make_pass_through_transform(number<mpair>{}),
                    make_pass_through_transform(KPerWarp)),
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
                make_tuple(make_merge_transform_v3_division_mod(make_tuple(
                               number<KPerThreadForReadPerm>{},
                               number<KPerThreadForWrite / kfold / KPerThreadForReadPerm>{},
                               number<kfold>{},
                               number<K0PerThreadForWrite>{},
                               KPerWarp)),
                           make_merge_transform_v3_division_mod(make_tuple(
                               number<MThreads / mpair>{}, number<mpair>{}, number<MPerThread>{}))),
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

        static_assert(Problem::BlockGemmShape::BlockWarps::at(I2) == 1,
                      "Assume there is only 1 warp among K dimension");

        using BDataType = remove_cvref_t<typename Problem::BDataType>;

        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t KPerWarp     = WarpGemm::kK;
        constexpr index_t KIterPerWarp = KPerBlock / KPerWarp;
        static_assert(KPerBlock == KIterPerWarp * KPerWarp);

        if constexpr(std::is_same<tensor_layout::gemm::ColumnMajor, LayoutB>::value)
        {
            // NLdsLayer * KIterPerWarp as logical Bank
            constexpr auto NLdsLayer = 32 * 4 / KPerBlock / sizeof(BDataType) < 1
                                           ? 1
                                           : 32 * 4 / KPerBlock / sizeof(BDataType);
            ;
            constexpr auto b_lds_block_desc = make_naive_tensor_descriptor(
                make_tuple(
                    KIterPerWarp * number<NLdsLayer>{}, number<NPerBlock / NLdsLayer>{}, KPerWarp),
                make_tuple(KPerWarp, number<KPerBlock * NLdsLayer>{}, I1));

            constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
                b_lds_block_desc,
                make_tuple(make_xor_transform(make_tuple(number<NPerBlock / NLdsLayer>{},
                                                         number<KIterPerWarp * NLdsLayer>{})),
                           make_pass_through_transform(KPerWarp)),
                make_tuple(sequence<1, 0>{}, sequence<2>{}),
                make_tuple(sequence<1, 0>{}, sequence<2>{}));

            constexpr auto b_lds_block_desc_bk0_kNLdsLayer_n_bk1 = transform_tensor_descriptor(
                b_lds_block_desc_permuted,
                make_tuple(make_unmerge_transform(make_tuple(KIterPerWarp, number<NLdsLayer>{})),
                           make_pass_through_transform(number<NPerBlock / NLdsLayer>{}),
                           make_pass_through_transform(KPerWarp)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

            constexpr auto b_lds_block_desc_n_k = transform_tensor_descriptor(
                b_lds_block_desc_bk0_kNLdsLayer_n_bk1,
                make_tuple(make_merge_transform_v3_division_mod(make_tuple(KIterPerWarp, KPerWarp)),
                           make_merge_transform_v3_division_mod(
                               make_tuple(number<NPerBlock / NLdsLayer>{}, number<NLdsLayer>{}))),
                make_tuple(sequence<0, 3>{}, sequence<1, 2>{}),
                make_tuple(sequence<1>{}, sequence<0>{}));

            return b_lds_block_desc_n_k;
        }
        else // RowMajor B
        {
            constexpr auto NThreads = get_warp_size() * Problem::BlockGemmShape::BlockWarps::at(I1);
            static_assert(NThreads <= NPerBlock,
                          "Make sure GEMM N tile (BlockTile[1]) is greater than or equal to "
                          "(get_warp_size() * BlockWarps[1])");

            constexpr auto NPerThread = NPerBlock / NThreads;
            static_assert(NPerBlock == NThreads * NPerThread);

            constexpr auto KPerThreadForWrite  = Problem::kBlockSize / NThreads;
            constexpr auto K0PerThreadForWrite = KIterPerWarp / KPerThreadForWrite;
            constexpr auto KPerThreadForRead   = get_warp_size() / WarpGemm::kN;
            constexpr auto K0PerThreadForRead  = KIterPerWarp / KPerThreadForRead;

            static_assert(KPerThreadForRead <= KIterPerWarp,
                          "GEMM N warp tile size (WarpTile[1]) is too small");

            static_assert(KIterPerWarp == K0PerThreadForWrite * KPerThreadForWrite);
            static_assert(KIterPerWarp == K0PerThreadForRead * KPerThreadForRead);

            // # bytes per 32 LDS banks: 32 * 4 bytes
            constexpr auto BankLength = 128;

            constexpr auto kfold = (KPerWarp * NThreads * sizeof(BDataType) > BankLength)
                                       ? 1
                                       : BankLength / (KPerWarp * NThreads * sizeof(BDataType));
            constexpr auto KPerThreadForReadPerm =
                (kfold * K0PerThreadForWrite / K0PerThreadForRead) > 1
                    ? KPerThreadForRead / (kfold * K0PerThreadForWrite / K0PerThreadForRead)
                    : KPerThreadForRead;

            // 1<=npair<=kN0
            constexpr auto npair =
                (KPerWarp * WarpGemm::kN * sizeof(BDataType) > BankLength)
                    ? 1
                    : ((BankLength / (KPerWarp * WarpGemm::kN * sizeof(BDataType))) > NThreads
                           ? NThreads
                           : BankLength / (KPerWarp * WarpGemm::kN * sizeof(BDataType)));

            constexpr auto b_lds_block_desc = make_naive_tensor_descriptor_packed(
                make_tuple(number<KPerThreadForWrite / kfold / KPerThreadForReadPerm>{},
                           number<K0PerThreadForWrite>{},
                           number<KPerThreadForReadPerm * NPerThread>{},
                           number<kfold * NThreads / npair>{},
                           number<npair>{},
                           KPerWarp));

            constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
                b_lds_block_desc,
                make_tuple(
                    make_pass_through_transform(
                        number<KPerThreadForWrite / kfold / KPerThreadForReadPerm>{}),
                    make_pass_through_transform(number<K0PerThreadForWrite>{}),
                    make_xor_transform(make_tuple(number<KPerThreadForReadPerm * NPerThread>{},
                                                  number<kfold * NThreads / npair>{})),
                    make_pass_through_transform(number<npair>{}),
                    make_pass_through_transform(KPerWarp)),
                make_tuple(
                    sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}),
                make_tuple(
                    sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}));

            constexpr auto b_lds_block_desc_unmerged = transform_tensor_descriptor(
                b_lds_block_desc_permuted,
                make_tuple(
                    make_pass_through_transform(
                        number<KPerThreadForWrite / kfold / KPerThreadForReadPerm>{}),
                    make_pass_through_transform(number<K0PerThreadForWrite>{}),
                    make_unmerge_transform(
                        make_tuple(number<KPerThreadForReadPerm>{}, number<NPerThread>{})),
                    make_unmerge_transform(make_tuple(number<kfold>{}, number<NThreads / npair>{})),
                    make_pass_through_transform(number<npair>{}),
                    make_pass_through_transform(KPerWarp)),
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
                make_tuple(make_merge_transform_v3_division_mod(make_tuple(
                               number<KPerThreadForReadPerm>{},
                               number<KPerThreadForWrite / kfold / KPerThreadForReadPerm>{},
                               number<kfold>{},
                               number<K0PerThreadForWrite>{},
                               KPerWarp)),
                           make_merge_transform_v3_division_mod(make_tuple(
                               number<NThreads / npair>{}, number<npair>{}, number<NPerThread>{}))),
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

        static_assert(Problem::BlockGemmShape::BlockWarps::at(I2) == 1,
                      "Assume there is only 1 warp among K dimension");

        constexpr index_t BlockSize = Problem::kBlockSize;

        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t KPerWarp     = WarpGemm::kK;
        constexpr index_t KIterPerWarp = KPerBlock / KPerWarp;
        static_assert(KPerBlock == KIterPerWarp * KPerWarp);

        constexpr index_t MThreadPerWarp = get_warp_size() / KIterPerWarp;
        constexpr index_t NumWarps       = BlockSize / get_warp_size();
        constexpr index_t MPerThread     = MPerBlock / (MThreadPerWarp * NumWarps);
        static_assert(MPerBlock == MPerThread * NumWarps * MThreadPerWarp);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<MPerThread, NumWarps, MThreadPerWarp>,
                                             sequence<KIterPerWarp, KPerWarp>>,
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

        static_assert(Problem::BlockGemmShape::BlockWarps::at(I2) == 1,
                      "Assume there is only 1 warp among K dimension");

        constexpr index_t BlockSize = Problem::kBlockSize;

        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t KPerWarp     = WarpGemm::kK;
        constexpr index_t KIterPerWarp = KPerBlock / KPerWarp;
        static_assert(KPerBlock == KIterPerWarp * KPerWarp);

        constexpr index_t NThreadPerWarp = get_warp_size() / KIterPerWarp;
        constexpr index_t NumWarps       = BlockSize / get_warp_size();
        constexpr index_t NPerThread     = NPerBlock / (NThreadPerWarp * NumWarps);
        static_assert(NPerBlock == NPerThread * NumWarps * NThreadPerWarp);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<NPerThread, NumWarps, NThreadPerWarp>,
                                             sequence<KIterPerWarp, KPerWarp>>,
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
