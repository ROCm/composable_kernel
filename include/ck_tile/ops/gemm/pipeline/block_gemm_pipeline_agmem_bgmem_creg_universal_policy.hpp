// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"

namespace ck_tile {

// UniversalGemm Policy
template <typename LayoutA_, typename LayoutB_, typename LayoutC_>
struct BlockGemmPipelineAGmemBGmemCRegUniversaltPolicy
{
    using LayoutA = remove_cvref_t<LayoutA_>;
    using LayoutB = remove_cvref_t<LayoutB_>;
    using LayoutC = remove_cvref_t<LayoutC_>;

    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        using WarpGemm = WarpGemmMfmaDispatcher<typename Problem::ADataType,
                                                typename Problem::BDataType,
                                                typename Problem::CDataType,
                                                Problem::BlockGemmShape::WarpTile::at(I0),
                                                Problem::BlockGemmShape::WarpTile::at(I1),
                                                Problem::BlockGemmShape::WarpTile::at(I2),
                                                true>;

        using ADataType = remove_cvref_t<typename Problem::ADataType>;

        using namespace ck_tile;

        constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t kK1        = WarpGemm::kK;
        constexpr index_t kK0        = kKPerBlock / kK1;

        if constexpr(std::is_same<tensor_layout::gemm::RowMajor, LayoutA>::value)
        {
            constexpr auto kMLdsLayer       = 32 * 4 / kKPerBlock / sizeof(ADataType) < 1
                                                  ? 1
                                                  : 32 * 4 / kKPerBlock / sizeof(ADataType);
            constexpr auto a_lds_block_desc = make_naive_tensor_descriptor(
                make_tuple(kK0 * number<kMLdsLayer>{}, number<kMPerBlock / kMLdsLayer>{}, kK1),
                make_tuple(kK1, number<kKPerBlock * kMLdsLayer>{}, I1));

            constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
                a_lds_block_desc,
                make_tuple(make_xor_transform(make_tuple(number<kMPerBlock / kMLdsLayer>{},
                                                         number<kK0 * kMLdsLayer>{})),
                           make_pass_through_transform(kK1)),
                make_tuple(sequence<1, 0>{}, sequence<2>{}),
                make_tuple(sequence<1, 0>{}, sequence<2>{}));

            constexpr auto a_lds_block_desc_ak0_kMLdsLayer_m_ak1 = transform_tensor_descriptor(
                a_lds_block_desc_permuted,
                make_tuple(make_unmerge_transform(make_tuple(kK0, number<kMLdsLayer>{})),
                           make_pass_through_transform(number<kMPerBlock / kMLdsLayer>{}),
                           make_pass_through_transform(kK1)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

            constexpr auto a_lds_block_desc_m_k = transform_tensor_descriptor(
                a_lds_block_desc_ak0_kMLdsLayer_m_ak1,
                make_tuple(make_merge_transform_v3_division_mod(make_tuple(kK0, kK1)),
                           make_merge_transform_v3_division_mod(make_tuple(
                               number<kMPerBlock / kMLdsLayer>{}, number<kMLdsLayer>{}))),
                make_tuple(sequence<0, 3>{}, sequence<1, 2>{}),
                make_tuple(sequence<1>{}, sequence<0>{}));

            return a_lds_block_desc_m_k;
        }
        else // ColumnMajor A
        {
            // kfold and mpair dimension is not always required.
            // more dimension in merge_transform increase the difficulty of generating immarg offset
            // for compiler.
            constexpr auto kM0 = get_warp_size() * Problem::BlockGemmShape::BlockWarps::at(I0);
            constexpr auto kM1 = kMPerBlock / kM0;

            constexpr auto kKThreadWrite     = Problem::kBlockSize / kM0;
            constexpr auto kK0PerThreadWrite = kK0 / kKThreadWrite;
            constexpr auto kKThreadRead      = 64 / WarpGemm::kM;
            constexpr auto kK0PerThreadRead  = kK0 / kKThreadRead;

            constexpr auto kfold =
                (kK1 * kM0 * sizeof(ADataType) > 128) ? 1 : 128 / (kK1 * kM0 * sizeof(ADataType));
            constexpr auto kKThreadReadPerm =
                (kfold * kK0PerThreadWrite / kK0PerThreadRead) > 1
                    ? kKThreadRead / (kfold * kK0PerThreadWrite / kK0PerThreadRead)
                    : kKThreadRead;

            // 1<=mpair<=kN0
            constexpr auto mpair = (kK1 * WarpGemm::kM * sizeof(ADataType) > 128)
                                       ? 1
                                       : ((128 / (kK1 * WarpGemm::kM * sizeof(ADataType))) > kM0
                                              ? kM0
                                              : 128 / (kK1 * WarpGemm::kM * sizeof(ADataType)));

            constexpr auto a_lds_block_desc = make_naive_tensor_descriptor_packed(
                make_tuple(number<kKThreadWrite / kfold / kKThreadReadPerm>{},
                           number<kK0PerThreadWrite>{},
                           number<kKThreadReadPerm * kM1>{},
                           number<kfold * kM0 / mpair>{},
                           number<mpair>{},
                           kK1));

            constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
                a_lds_block_desc,
                make_tuple(
                    make_pass_through_transform(number<kKThreadWrite / kfold / kKThreadReadPerm>{}),
                    make_pass_through_transform(number<kK0PerThreadWrite>{}),
                    make_xor_transform(make_tuple(number<kKThreadReadPerm * kM1>{},
                                                  number<kfold * kM0 / mpair>{})),
                    make_pass_through_transform(number<mpair>{}),
                    make_pass_through_transform(kK1)),
                make_tuple(
                    sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}),
                make_tuple(
                    sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}));

            constexpr auto a_lds_block_desc_unmerged = transform_tensor_descriptor(
                a_lds_block_desc_permuted,
                make_tuple(
                    make_pass_through_transform(number<kKThreadWrite / kfold / kKThreadReadPerm>{}),
                    make_pass_through_transform(number<kK0PerThreadWrite>{}),
                    make_unmerge_transform(make_tuple(number<kKThreadReadPerm>{}, number<kM1>{})),
                    make_unmerge_transform(make_tuple(number<kfold>{}, number<kM0 / mpair>{})),
                    make_pass_through_transform(number<mpair>{}),
                    make_pass_through_transform(kK1)),
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
                               make_tuple(number<kKThreadReadPerm>{},
                                          number<kKThreadWrite / kfold / kKThreadReadPerm>{},
                                          number<kfold>{},
                                          number<kK0PerThreadWrite>{},
                                          kK1)),
                           make_merge_transform_v3_division_mod(
                               make_tuple(number<kM0 / mpair>{}, number<mpair>{}, number<kM1>{}))),
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
                                                true>;

        using BDataType = remove_cvref_t<typename Problem::BDataType>;

        using namespace ck_tile;

        constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t kK1 = WarpGemm::kK;
        constexpr index_t kK0 = kKPerBlock / kK1;

        if constexpr(std::is_same<tensor_layout::gemm::ColumnMajor, LayoutB>::value)
        {
            // kNLdsLayer * K0 as logical Bank
            constexpr auto kNLdsLayer = 32 * 4 / kKPerBlock / sizeof(BDataType) < 1
                                            ? 1
                                            : 32 * 4 / kKPerBlock / sizeof(BDataType);
            ;
            constexpr auto b_lds_block_desc = make_naive_tensor_descriptor(
                make_tuple(kK0 * number<kNLdsLayer>{}, number<kNPerBlock / kNLdsLayer>{}, kK1),
                make_tuple(kK1, number<kKPerBlock * kNLdsLayer>{}, I1));

            constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
                b_lds_block_desc,
                make_tuple(make_xor_transform(make_tuple(number<kNPerBlock / kNLdsLayer>{},
                                                         number<kK0 * kNLdsLayer>{})),
                           make_pass_through_transform(kK1)),
                make_tuple(sequence<1, 0>{}, sequence<2>{}),
                make_tuple(sequence<1, 0>{}, sequence<2>{}));

            constexpr auto b_lds_block_desc_bk0_kNLdsLayer_n_bk1 = transform_tensor_descriptor(
                b_lds_block_desc_permuted,
                make_tuple(make_unmerge_transform(make_tuple(kK0, number<kNLdsLayer>{})),
                           make_pass_through_transform(number<kNPerBlock / kNLdsLayer>{}),
                           make_pass_through_transform(kK1)),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

            constexpr auto b_lds_block_desc_n_k = transform_tensor_descriptor(
                b_lds_block_desc_bk0_kNLdsLayer_n_bk1,
                make_tuple(make_merge_transform_v3_division_mod(make_tuple(kK0, kK1)),
                           make_merge_transform_v3_division_mod(make_tuple(
                               number<kNPerBlock / kNLdsLayer>{}, number<kNLdsLayer>{}))),
                make_tuple(sequence<0, 3>{}, sequence<1, 2>{}),
                make_tuple(sequence<1>{}, sequence<0>{}));

            return b_lds_block_desc_n_k;
        }
        else // RowMajor B
        {
            constexpr auto kN0 = get_warp_size() * Problem::BlockGemmShape::BlockWarps::at(I1);
            constexpr auto kN1 = kNPerBlock / kN0;

            constexpr auto kKThreadWrite     = Problem::kBlockSize / kN0;
            constexpr auto kK0PerThreadWrite = kK0 / kKThreadWrite;
            constexpr auto kKThreadRead      = 64 / WarpGemm::kN;
            constexpr auto kK0PerThreadRead  = kK0 / kKThreadRead;

            constexpr auto kfold =
                (kK1 * kN0 * sizeof(BDataType) > 128) ? 1 : 128 / (kK1 * kN0 * sizeof(BDataType));
            constexpr auto kKThreadReadPerm =
                (kfold * kK0PerThreadWrite / kK0PerThreadRead) > 1
                    ? kKThreadRead / (kfold * kK0PerThreadWrite / kK0PerThreadRead)
                    : kKThreadRead;

            // 1<=npair<=kN0
            constexpr auto npair = (kK1 * WarpGemm::kN * sizeof(BDataType) > 128)
                                       ? 1
                                       : ((128 / (kK1 * WarpGemm::kN * sizeof(BDataType))) > kN0
                                              ? kN0
                                              : 128 / (kK1 * WarpGemm::kN * sizeof(BDataType)));

            constexpr auto b_lds_block_desc = make_naive_tensor_descriptor_packed(
                make_tuple(number<kKThreadWrite / kfold / kKThreadReadPerm>{},
                           number<kK0PerThreadWrite>{},
                           number<kKThreadReadPerm * kN1>{},
                           number<kfold * kN0 / npair>{},
                           number<npair>{},
                           kK1));

            constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
                b_lds_block_desc,
                make_tuple(
                    make_pass_through_transform(number<kKThreadWrite / kfold / kKThreadReadPerm>{}),
                    make_pass_through_transform(number<kK0PerThreadWrite>{}),
                    make_xor_transform(make_tuple(number<kKThreadReadPerm * kN1>{},
                                                  number<kfold * kN0 / npair>{})),
                    make_pass_through_transform(number<npair>{}),
                    make_pass_through_transform(kK1)),
                make_tuple(
                    sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}),
                make_tuple(
                    sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}));

            constexpr auto b_lds_block_desc_unmerged = transform_tensor_descriptor(
                b_lds_block_desc_permuted,
                make_tuple(
                    make_pass_through_transform(number<kKThreadWrite / kfold / kKThreadReadPerm>{}),
                    make_pass_through_transform(number<kK0PerThreadWrite>{}),
                    make_unmerge_transform(make_tuple(number<kKThreadReadPerm>{}, number<kN1>{})),
                    make_unmerge_transform(make_tuple(number<kfold>{}, number<kN0 / npair>{})),
                    make_pass_through_transform(number<npair>{}),
                    make_pass_through_transform(kK1)),
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
                               make_tuple(number<kKThreadReadPerm>{},
                                          number<kKThreadWrite / kfold / kKThreadReadPerm>{},
                                          number<kfold>{},
                                          number<kK0PerThreadWrite>{})),
                           make_merge_transform_v3_division_mod(
                               make_tuple(number<kN0 / npair>{}, number<npair>{}, number<kN1>{})),
                           make_pass_through_transform(kK1)),
                make_tuple(sequence<0, 1, 4, 2, 7>{}, sequence<5, 6, 3>{}),
                make_tuple(sequence<1>{}, sequence<0>{}));

            return b_lds_block_desc_n_k;
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeA()
    {
        constexpr index_t smem_size_a = sizeof(typename Problem::ADataType) *
                                        MakeALdsBlockDescriptor<Problem>().get_element_space_size();
        return smem_size_a;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeB()
    {
        constexpr index_t smem_size_b = sizeof(typename Problem::BDataType) *
                                        MakeBLdsBlockDescriptor<Problem>().get_element_space_size();
        return smem_size_b;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
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
                                                true>;

        constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr auto kM0 = get_warp_size() * Problem::BlockGemmShape::BlockWarps::at(I0);
        constexpr auto kM1 = kMPerBlock / kM0;

        constexpr index_t kK1 = WarpGemm::kK;
        constexpr index_t kK0 = kKPerBlock / kK1;

        static_assert(kM1 != 0, "M1 is zero, which will lead to a division by zero error.");

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<kM0, kM1>, sequence<kK0, kK1>>,
                                       tuple<sequence<1>, sequence<2>>,
                                       tuple<sequence<0>, sequence<1>>,
                                       sequence<1, 2>,
                                       sequence<1, 0>>{});
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
                                                true>;

        constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr auto kN0 = get_warp_size() * Problem::BlockGemmShape::BlockWarps::at(I1);
        constexpr auto kN1 = kNPerBlock / kN0;

        constexpr index_t kK1 = WarpGemm::kK;
        constexpr index_t kK0 = kKPerBlock / kK1;

        static_assert(kN1 != 0, "M1 is zero, which will lead to a division by zero error.");

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<kN0, kN1>, sequence<kK0, kK1>>,
                                       tuple<sequence<1>, sequence<2>>,
                                       tuple<sequence<0>, sequence<1>>,
                                       sequence<1, 2>,
                                       sequence<1, 0>>{});
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
                                                true>;
        using BlockGemmPolicy = BlockGemmASmemBSmemCRegV1CustomPolicy<typename Problem::ADataType,
                                                                      typename Problem::BDataType,
                                                                      typename Problem::CDataType,
                                                                      BlockWarps,
                                                                      WarpGemm>;
        return BlockGemmASmemBSmemCRegV1<Problem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
