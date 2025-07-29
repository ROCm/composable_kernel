// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"

namespace ck_tile {

struct UniversalWeightPreshufflePipelineAgBgCrPolicy
{
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};

    // 3d + padding
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        using namespace ck_tile;

        constexpr index_t MPerXdl = Problem::BlockGemmShape::WarpTile::at(I0);
        constexpr index_t NPerXdl = Problem::BlockGemmShape::WarpTile::at(I1);
        if constexpr(MPerXdl == 16 && NPerXdl == 16)
        {
            /*reduce transform layers,compare with old ck*/
            constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
            constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
            constexpr index_t KPack     = GetSmemPackA<Problem>();

            constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<KPerBlock / KPack>{}, number<MPerBlock>{}, number<KPack>{}),
                make_tuple(number<KPack>{}, number<KPerBlock>{}, number<1>{}),
                number<KPack>{},
                number<1>{});

            constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
                a_lds_block_desc_0,
                make_tuple(make_xor_transform(
                               make_tuple(number<MPerBlock>{}, number<KPerBlock / KPack>{})),
                           make_pass_through_transform(number<KPack>{})),
                make_tuple(sequence<1, 0>{}, sequence<2>{}),
                make_tuple(sequence<1, 0>{}, sequence<2>{}));

            constexpr auto a_lds_block_desc = transform_tensor_descriptor(
                a_lds_block_desc_permuted,
                make_tuple(make_pass_through_transform(number<MPerBlock>{}),
                           make_merge_transform_v3_division_mod(
                               make_tuple(number<KPerBlock / KPack>{}, number<KPack>{}))),
                make_tuple(sequence<1>{}, sequence<0, 2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return a_lds_block_desc;
        }
        else
        {
            constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
            constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;
            constexpr index_t kKPack     = GetSmemPackA<Problem>();

            constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<kKPerBlock / kKPack>{}, number<kMPerBlock>{}, number<kKPack>{}),
                make_tuple(number<(kMPerBlock + 1) * kKPack>{}, number<kKPack>{}, number<1>{}),
                number<kKPack>{},
                number<1>{});

            constexpr auto a_lds_block_desc = transform_tensor_descriptor(
                a_lds_block_desc_0,
                make_tuple(make_pass_through_transform(kMPerBlock),
                           make_merge_transform(make_tuple(kKPerBlock / kKPack, kKPack))),
                make_tuple(sequence<1>{}, sequence<0, 2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return a_lds_block_desc;
        }
/*xor*/
#if 0
        constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t kKPack     = GetSmemPackA<Problem>();
        using ADataType = remove_cvref_t<typename Problem::ADataType>;

        constexpr auto DataTypeSize = sizeof(ADataType);
        constexpr auto MLdsLayer =
            (32 * 4 / kKPerBlock / DataTypeSize) < 1 ? 1 : (32 * 4 / kKPerBlock / DataTypeSize);

        constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kKPerBlock / kKPack * MLdsLayer>{},
                    number<kMPerBlock / MLdsLayer>{},
                    number<kKPack>{}),
            make_tuple(number<kKPack>{}, number<kKPerBlock * MLdsLayer>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
            a_lds_block_desc_0,
            make_tuple(make_xor_transform(make_tuple(number<kMPerBlock / MLdsLayer>{},
                                                    number<kKPerBlock / kKPack * MLdsLayer>{})),
                    make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<1, 0>{}, sequence<2>{}),
            make_tuple(sequence<1, 0>{}, sequence<2>{}));

        constexpr auto a_lds_block_desc_xk0_mnldslayer_mn_xk1 = transform_tensor_descriptor(
            a_lds_block_desc_permuted,
            make_tuple(make_unmerge_transform(
                        make_tuple(number<MLdsLayer>{}, number<kKPerBlock / kKPack>{})),
                        make_pass_through_transform(number<kMPerBlock / MLdsLayer>{}),
                        make_pass_through_transform(number<kKPack>{})),
            make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
            make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

        constexpr auto a_lds_block_desc = transform_tensor_descriptor(
            a_lds_block_desc_xk0_mnldslayer_mn_xk1,
            make_tuple(make_merge_transform(
                        make_tuple(number<kMPerBlock / MLdsLayer>{}, number<MLdsLayer>{})),
                        make_merge_transform(
                        make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
            make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
        return a_lds_block_desc;
#endif
    }

    /**
     * @brief Get the maximum global memory vector load size.
     *
     * @tparam Problem      The UniversalGemmPipelineProblem object.
     * @tparam DataType     The tensor data type we're considering.
     * @tparam MNPerBlock   The MPerBlock or NPerBlock value depending on tensor (A/B).
     * @tparam XPerTile     The contiguous Tile dimension size.
     * @return Maximum DRAM vector load size.
     */
    template <typename Problem, typename DataType, index_t MNPerBlock, index_t XPerTile>
    CK_TILE_HOST_DEVICE static constexpr auto GetGlobalVectorLoadSize()
    {
        constexpr index_t BlockSize           = Problem::kBlockSize;
        constexpr index_t KPerBlock           = Problem::BlockGemmShape::kK;
        constexpr index_t elements_per_thread = MNPerBlock * KPerBlock / BlockSize;
        constexpr index_t PackedSize =
            ck_tile::numeric_traits<remove_cvref_t<DataType>>::PackedSize;

        // Assume DataType is even!
        if constexpr(XPerTile % (PackedSize * 32 / sizeof(DataType)) == 0 &&
                     elements_per_thread % (PackedSize * 32 / sizeof(DataType)) == 0 &&
                     PackedSize == 2)
        {
            return (PackedSize * 32 / sizeof(DataType));
        }
        else if constexpr(XPerTile % (PackedSize * 16 / sizeof(DataType)) == 0 &&
                          elements_per_thread % (PackedSize * 16 / sizeof(DataType)) == 0)
        {
            return (PackedSize * 16 / sizeof(DataType));
        }
        else if constexpr(XPerTile % (PackedSize * 8 / sizeof(DataType)) == 0 &&
                          elements_per_thread % (PackedSize * 8 / sizeof(DataType)) == 0)
        {
            return (PackedSize * 8 / sizeof(DataType));
        }
        else if constexpr(sizeof(DataType) >= PackedSize * 4 &&
                          XPerTile % (PackedSize * 4 / sizeof(DataType)) == 0 &&
                          elements_per_thread % (PackedSize * 4 / sizeof(DataType)) == 0)
        {
            return (PackedSize * 4 / sizeof(DataType));
        }
        else if constexpr(sizeof(DataType) >= PackedSize * 2 &&
                          XPerTile % (PackedSize * 2 / sizeof(DataType)) == 0 &&
                          elements_per_thread % (PackedSize * 2 / sizeof(DataType)) == 0)
        {
            return (PackedSize * 2 / sizeof(DataType));
        }
        else
        {
            return PackedSize;
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeA()
    {
        using ALayout               = remove_cvref_t<typename Problem::ALayout>;
        using ADataType             = remove_cvref_t<typename Problem::ADataType>;
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        if constexpr(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            return GetGlobalVectorLoadSize<Problem, ADataType, MPerBlock, KPerBlock>();
        }
        else
        {
            return GetGlobalVectorLoadSize<Problem, ADataType, MPerBlock, MPerBlock>();
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeB()
    {
        using BLayout               = remove_cvref_t<typename Problem::BLayout>;
        using BDataType             = remove_cvref_t<typename Problem::BDataType>;
        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        if constexpr(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            return GetGlobalVectorLoadSize<Problem, BDataType, NPerBlock, NPerBlock>();
        }
        else
        {
            return GetGlobalVectorLoadSize<Problem, BDataType, NPerBlock, KPerBlock>();
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
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        constexpr index_t smem_size_a = GetSmemSizeA<Problem>();

        return smem_size_a;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemPackA()
    {
        return Problem::VectorLoadSize / sizeof(typename Problem::ADataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetKBPerLoad()
    {
        using TileShape = typename Problem::BlockGemmShape;
        if constexpr(TileShape::WarpTile::at(I1) == 32)
        {
            return TileShape::WarpTile::at(I2) / 2;
        }
        else
        {
            static_assert(TileShape::WarpTile::at(I1) == 16);
            return TileShape::WarpTile::at(I2) / 4;
        }
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
            constexpr index_t KPack = GetSmemPackA<Problem>();
            static_assert(KPack % K3 == 0);
            constexpr index_t K2 = KPack / K3;
            if constexpr(get_warp_size() >= (K2 * M0))
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
            // coalesce reading for each blocks
            if constexpr(get_warp_size() % (M2 * K0) == 0)
            {
                constexpr index_t M1 = BlockSize / get_warp_size();
                static_assert(M2 != 0, "M2 is zero, which will lead to a division by zero error.");
                static_assert(M1 != 0, "M1 is zero, which will lead to a division by zero error.");
                constexpr index_t M0 = MPerBlock / (M2 * M1);
                static_assert(M0 * M1 * M2 == MPerBlock,
                              "Incorrect M0, M2, M1 configuration! "
                              "M0, M1, M2 must cover whole MPerBlock!");

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
                static_assert(M0 * M1 * M2 == MPerBlock,
                              "Incorrect M0, M1, M2 configuration! "
                              "M0, M1, M2 must cover whole MPerBlock!");
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
    CK_TILE_HOST_DEVICE static constexpr auto MakeBFlatDramTileDistribution()
    {
        using TileShape = typename Problem::BlockGemmShape;

        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t WaveSize  = get_warp_size();
        constexpr index_t WaveNum   = BlockSize / WaveSize;

        constexpr index_t KBPerLoad   = GetKBPerLoad<Problem>();
        constexpr index_t KThdPerWave = WaveSize; // threads cnt in K dim
        constexpr index_t KWavePerBlk = 1;
        constexpr index_t KRepeat     = 1;
        static_assert(TileShape::flatKPerWarp == KThdPerWave * KBPerLoad, "wrong");

        constexpr index_t NBPerLoad   = 1;
        constexpr index_t NThdPerWave = 1;
        constexpr index_t NWavePerBlk = TileShape::BlockWarps::at(number<1>{}); // N_Warp
        constexpr index_t NRepeat     = 1;

        constexpr index_t WaveRepeat = WaveNum / TileShape::flatNPerWarp;

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<WaveRepeat>,                                          // ?
                tuple<sequence<NRepeat, NWavePerBlk, NThdPerWave, NBPerLoad>,  // second direction
                      sequence<KRepeat, KWavePerBlk, KThdPerWave, KBPerLoad>>, // first  direction
                // wave in blk,     // thd in wave
                // <M, K>           // <M, K>
                tuple<sequence<0, 1, 2>, sequence<1, 2>>, // which direction
                tuple<sequence<0, 1, 1>, sequence<2, 2>>, // which index
                // <repeat, vec_load>
                sequence<1, 1, 2, 2>,
                sequence<0, 3, 0, 3>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledARegBlockDistribution()
    {
        using ALayout   = remove_cvref_t<typename Problem::ALayout>;
        using ADataType = remove_cvref_t<typename Problem::ADataType>;
        static_assert(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::ColumnMajor>);
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t kKPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t M1           = Problem::VectorLoadSize / sizeof(ADataType);
        constexpr index_t M0           = kMPerBlock / M1;
        constexpr index_t total_pixels = kMPerBlock * kKPerBlock / kBlockSize;
        static_assert(total_pixels % M1 == 0);
        constexpr index_t K3     = total_pixels / M1;
        constexpr index_t kKPack = GetSmemPackA<Problem>();
        static_assert(kKPack % K3 == 0);
        constexpr index_t K2 = kKPack / K3; // TODO: this dimention could be outside single wave
        constexpr index_t warp_size = get_warp_size();
        if constexpr(warp_size >= (K2 * M0))
        {
            constexpr index_t K1 = warp_size / (K2 * M0);
            constexpr index_t K0 = kBlockSize / warp_size;

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
            constexpr index_t K0   = kBlockSize / get_warp_size() / K1;
            static_assert(kKPerBlock == K0 * K1 * K2_m * K3);
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
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockWeightPreshuffle()
    {
        // using AccDataType = float;
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;
        using WarpGemm   = WarpGemmMfmaDispatcher<typename Problem::ADataType,
                                                  typename Problem::BDataType,
                                                  typename Problem::CDataType,
                                                  WarpTile::at(I0),
                                                  WarpTile::at(I1),
                                                  WarpTile::at(I2),
                                                  Problem::TransposeC>;

        using BlockWeightPreshufflePolicy =
            BlockWeightPreshuffleASmemBSmemCRegV1CustomPolicy<typename Problem::ADataType,
                                                              typename Problem::BDataType,
                                                              typename Problem::CDataType,
                                                              BlockWarps,
                                                              WarpGemm>;
        return BlockWeightPreshuffleASmemBSmemCRegV1<Problem, BlockWeightPreshufflePolicy>{};
    }
};

} // namespace ck_tile
