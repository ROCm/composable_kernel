// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"

namespace ck_tile {

// UniversalGemm Policy
struct UniversalGemmPipelineAgBgCrPolicy
{

    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};

    static constexpr bool TransposeC = true;

    template <typename Problem, typename DataType, index_t MNPerBlock>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorLoadSize()
    {
        constexpr index_t BlockSize           = Problem::kBlockSize;
        constexpr index_t KPerBlock           = Problem::BlockGemmShape::kK;
        constexpr index_t elements_per_thread = MNPerBlock * KPerBlock / BlockSize;

        if constexpr(elements_per_thread % (16 / sizeof(DataType)) == 0)
        {
            return (16 / sizeof(DataType));
        }
        else if constexpr(elements_per_thread % (8 / sizeof(DataType)) == 0)
        {
            return (8 / sizeof(DataType));
        }
        else if constexpr(elements_per_thread % (4 / sizeof(DataType)) == 0 &&
                          sizeof(DataType) >= 4)
        {
            return (4 / sizeof(DataType));
        }
        else if constexpr(elements_per_thread % (2 / sizeof(DataType)) == 0 &&
                          sizeof(DataType) >= 2)
        {
            return (2 / sizeof(DataType));
        }
        else
        {
            return 1;
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {

        using ADataType = remove_cvref_t<typename Problem::ADataType>;

        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t KPack     = GetVectorLoadSize<Problem, ADataType, MPerBlock>();

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

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {

        using BDataType = remove_cvref_t<typename Problem::BDataType>;

        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t KPack     = GetVectorLoadSize<Problem, BDataType, NPerBlock>();

        constexpr auto DataTypeSize = sizeof(BDataType);
        constexpr auto NLdsLayer =
            (32 * 4 / KPerBlock / DataTypeSize) < 1 ? 1 : (32 * 4 / KPerBlock / DataTypeSize);

        constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<KPerBlock / KPack * NLdsLayer>{},
                       number<NPerBlock / NLdsLayer>{},
                       number<KPack>{}),
            make_tuple(number<KPack>{}, number<KPerBlock * NLdsLayer>{}, number<1>{}),
            number<KPack>{},
            number<1>{});

        constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
            b_lds_block_desc_0,
            make_tuple(make_xor_transform(make_tuple(number<NPerBlock / NLdsLayer>{},
                                                     number<KPerBlock / KPack * NLdsLayer>{})),
                       make_pass_through_transform(number<KPack>{})),
            make_tuple(sequence<1, 0>{}, sequence<2>{}),
            make_tuple(sequence<1, 0>{}, sequence<2>{}));

        constexpr auto b_lds_block_desc_xk0_mnldslayer_mn_xk1 = transform_tensor_descriptor(
            b_lds_block_desc_permuted,
            make_tuple(make_unmerge_transform(
                           make_tuple(number<KPerBlock / KPack>{}, number<NLdsLayer>{})),
                       make_pass_through_transform(number<NPerBlock / NLdsLayer>{}),
                       make_pass_through_transform(number<KPack>{})),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
            make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

        constexpr auto b_lds_block_desc = transform_tensor_descriptor(
            b_lds_block_desc_xk0_mnldslayer_mn_xk1,
            make_tuple(make_merge_transform_v3_division_mod(
                           make_tuple(number<NPerBlock / NLdsLayer>{}, number<NLdsLayer>{})),
                       make_merge_transform_v3_division_mod(
                           make_tuple(number<KPerBlock / KPack>{}, number<KPack>{}))),
            make_tuple(sequence<1, 2>{}, sequence<0, 3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
        return b_lds_block_desc;
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
        using ADataType = remove_cvref_t<typename Problem::ADataType>;
        using ALayout   = remove_cvref_t<typename Problem::ALayout>;

        constexpr index_t BlockSize = Problem::kBlockSize;

        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        if constexpr(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::ColumnMajor>)
        {
            constexpr index_t M1           = Problem::VectorLoadSize / sizeof(ADataType);
            constexpr index_t M0           = MPerBlock / M1;
            constexpr index_t total_pixels = MPerBlock * KPerBlock / BlockSize;
            static_assert(total_pixels % M1 == 0);
            constexpr index_t K3    = total_pixels / M1;
            constexpr index_t KPack = GetVectorLoadSize<Problem, ADataType, MPerBlock>();
            static_assert(KPack % K3 == 0);
            constexpr index_t K2 = KPack / K3;
            if constexpr(get_warp_size() % (K2 * M0) == 0)
            {
                constexpr index_t K1 = get_warp_size() / (K2 * M0);
                constexpr index_t K0 = BlockSize / get_warp_size();
                static_assert(KPerBlock == K0 * K1 * K2 * K3);
                return make_static_tile_distribution(
                    tile_distribution_encoding<sequence<1>,
                                               tuple<sequence<M0, M1>, sequence<K0, K1, K2, K3>>,
                                               tuple<sequence<2>, sequence<2, 1, 2>>,
                                               tuple<sequence<0>, sequence<1, 0, 2>>,
                                               sequence<2, 1>,
                                               sequence<3, 1>>{});
            }
            else
            {
                constexpr index_t K1   = (K2 * M0) / get_warp_size();
                constexpr index_t K2_m = K2 / K1;
                constexpr index_t K0   = BlockSize / get_warp_size() / K1;
                static_assert(KPerBlock == K0 * K1 * K2_m * K3);
                return make_static_tile_distribution(
                    tile_distribution_encoding<sequence<1>,
                                               tuple<sequence<M0, M1>, sequence<K0, K1, K2_m, K3>>,
                                               tuple<sequence<2, 2>, sequence<1, 2>>,
                                               tuple<sequence<0, 1>, sequence<0, 2>>,
                                               sequence<2, 1>,
                                               sequence<3, 1>>{});
            }
        }
        else
        {
            constexpr index_t K1 = Problem::VectorLoadSize / sizeof(ADataType);
            constexpr index_t K0 = KPerBlock / K1;
            constexpr index_t M2 = get_warp_size() / K0;
            if constexpr(get_warp_size() % (M2 * K0) == 0)
            {
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
            else
            {
                constexpr index_t M0 = BlockSize / get_warp_size();
                constexpr index_t M1 = MPerBlock / (M2 * M0);
                return make_static_tile_distribution(
                    tile_distribution_encoding<sequence<1>,
                                               tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                                               tuple<sequence<1>, sequence<1, 2>>,
                                               tuple<sequence<0>, sequence<2, 0>>,
                                               sequence<1, 2>,
                                               sequence<1, 1>>{});
            }
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBDramTileDistribution()
    {
        using BDataType = remove_cvref_t<typename Problem::BDataType>;
        using BLayout   = remove_cvref_t<typename Problem::BLayout>;

        constexpr index_t BlockSize = Problem::kBlockSize;

        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        if constexpr(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            constexpr index_t N1           = Problem::VectorLoadSize / sizeof(BDataType);
            constexpr index_t N0           = NPerBlock / N1;
            constexpr index_t total_pixels = NPerBlock * KPerBlock / BlockSize;
            static_assert(total_pixels % N1 == 0);
            constexpr index_t K3    = total_pixels / N1;
            constexpr index_t KPack = GetVectorLoadSize<Problem, BDataType, NPerBlock>();
            static_assert(KPack % K3 == 0);
            constexpr index_t K2 = KPack / K3;
            if constexpr(get_warp_size() % (K2 * N0) == 0)
            {
                constexpr index_t K1 = get_warp_size() / (K2 * N0);
                constexpr index_t K0 = BlockSize / get_warp_size();
                static_assert(KPerBlock == K0 * K1 * K2 * K3);
                return make_static_tile_distribution(
                    tile_distribution_encoding<sequence<1>,
                                               tuple<sequence<N0, N1>, sequence<K0, K1, K2, K3>>,
                                               tuple<sequence<2>, sequence<2, 1, 2>>,
                                               tuple<sequence<0>, sequence<1, 0, 2>>,
                                               sequence<2, 1>,
                                               sequence<3, 1>>{});
            }
            else
            {
                constexpr index_t K1   = (K2 * N0) / get_warp_size();
                constexpr index_t K2_m = K2 / K1;
                constexpr index_t K0   = BlockSize / get_warp_size() / K1;
                static_assert(KPerBlock == K0 * K1 * K2_m * K3);
                return make_static_tile_distribution(
                    tile_distribution_encoding<sequence<1>,
                                               tuple<sequence<N0, N1>, sequence<K0, K1, K2_m, K3>>,
                                               tuple<sequence<2, 2>, sequence<1, 2>>,
                                               tuple<sequence<0, 1>, sequence<0, 2>>,
                                               sequence<2, 1>,
                                               sequence<3, 1>>{});
            }
        }
        else
        {

            constexpr index_t K1 = Problem::VectorLoadSize / sizeof(BDataType);
            constexpr index_t K0 = KPerBlock / K1;
            constexpr index_t N2 = get_warp_size() / K0;
            // coalesce reading for each blocks
            if constexpr(get_warp_size() % (N2 * K0) == 0)
            {
                constexpr index_t N1 = BlockSize / get_warp_size();
                static_assert(N2 != 0, "N2 is zero, which will lead to a division by zero error.");
                static_assert(N1 != 0, "N1 is zero, which will lead to a division by zero error.");
                constexpr index_t N0 = NPerBlock / (N2 * N1);

                return make_static_tile_distribution(
                    tile_distribution_encoding<sequence<1>,
                                               tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                               tuple<sequence<1>, sequence<1, 2>>,
                                               tuple<sequence<1>, sequence<2, 0>>,
                                               sequence<1, 2>,
                                               sequence<0, 1>>{});
            }
            // coalesce reading for each warps
            else
            {
                constexpr index_t N0 = BlockSize / get_warp_size();
                constexpr index_t N1 = NPerBlock / (N2 * N0);

                return make_static_tile_distribution(
                    tile_distribution_encoding<sequence<1>,
                                               tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                               tuple<sequence<1>, sequence<1, 2>>,
                                               tuple<sequence<0>, sequence<2, 0>>,
                                               sequence<1, 2>,
                                               sequence<1, 1>>{});
            }
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledARegBlockDescriptor()
    {
        using ALayout   = remove_cvref_t<typename Problem::ALayout>;
        using ADataType = remove_cvref_t<typename Problem::ADataType>;
        static_assert(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::ColumnMajor>);
        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t M1           = Problem::VectorLoadSize / sizeof(ADataType);
        constexpr index_t M0           = MPerBlock / M1;
        constexpr index_t total_pixels = MPerBlock * KPerBlock / BlockSize;
        static_assert(total_pixels % M1 == 0);
        constexpr index_t K3     = total_pixels / M1;
        constexpr index_t kKPack = GetVectorLoadSize<Problem, ADataType, MPerBlock>();
        static_assert(kKPack % K3 == 0);
        constexpr index_t K2 = kKPack / K3; // TODO: this dimention could be outside single wave
        constexpr index_t warp_size = get_warp_size();
        if constexpr(warp_size % (K2 * M0) == 0)
        {
            constexpr index_t K1 = warp_size / (K2 * M0);
            constexpr index_t K0 = BlockSize / warp_size;

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<M0, M1>, sequence<K0, K1, K2, K3>>,
                                           tuple<sequence<2>, sequence<2, 1, 2>>,
                                           tuple<sequence<0>, sequence<1, 0, 2>>,
                                           sequence<1, 2>,
                                           sequence<1, 3>>{});
        }
        else
        {
            constexpr index_t K1   = (K2 * M0) / get_warp_size();
            constexpr index_t K2_m = K2 / K1;
            constexpr index_t K0   = BlockSize / get_warp_size() / K1;
            static_assert(KPerBlock == K0 * K1 * K2_m * K3);
            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<M0, M1>, sequence<K0, K1, K2_m, K3>>,
                                           tuple<sequence<2, 2>, sequence<1, 2>>,
                                           tuple<sequence<0, 1>, sequence<0, 2>>,
                                           sequence<1, 2>,
                                           sequence<1, 3>>{});
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledBRegBlockDescriptor()
    {
        using BLayout   = remove_cvref_t<typename Problem::BLayout>;
        using BDataType = remove_cvref_t<typename Problem::BDataType>;
        static_assert(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>);
        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t N1           = Problem::VectorLoadSize / sizeof(BDataType);
        constexpr index_t N0           = NPerBlock / N1;
        constexpr index_t total_pixels = NPerBlock * KPerBlock / BlockSize;
        static_assert(total_pixels % N1 == 0);
        constexpr index_t K3     = total_pixels / N1;
        constexpr index_t kKPack = GetVectorLoadSize<Problem, BDataType, NPerBlock>();
        static_assert(kKPack % K3 == 0);
        constexpr index_t K2 = kKPack / K3; // TODO: this dimention could be outside single wave
        constexpr index_t warp_size = get_warp_size();
        if constexpr(warp_size % (K2 * N0) == 0)
        {
            constexpr index_t K1 = warp_size / (K2 * N0);
            constexpr index_t K0 = BlockSize / warp_size;

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<N0, N1>, sequence<K0, K1, K2, K3>>,
                                           tuple<sequence<2>, sequence<2, 1, 2>>,
                                           tuple<sequence<0>, sequence<1, 0, 2>>,
                                           sequence<1, 2>,
                                           sequence<1, 3>>{});
        }
        else
        {
            constexpr index_t K1   = (K2 * N0) / get_warp_size();
            constexpr index_t K2_m = K2 / K1;
            constexpr index_t K0   = BlockSize / get_warp_size() / K1;
            static_assert(KPerBlock == K0 * K1 * K2_m * K3);
            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<N0, N1>, sequence<K0, K1, K2_m, K3>>,
                                           tuple<sequence<2, 2>, sequence<1, 2>>,
                                           tuple<sequence<0, 1>, sequence<0, 2>>,
                                           sequence<1, 2>,
                                           sequence<1, 3>>{});
        }
    }

    CK_TILE_HOST_DEVICE static constexpr auto IsTransposeC() { return TransposeC; }

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
