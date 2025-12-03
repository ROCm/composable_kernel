// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp"

namespace ck_tile {

struct MXFlatmmPipelineAgBgCrPolicy : UniversalFlatmmPipelineAgBgCrPolicy
{
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};

    static constexpr index_t kDramLoadPackBytes = 128;

    static constexpr int MXdlPack = 2;
    static constexpr int NXdlPack = 2;
    static constexpr int KXdlPack = 2;

    template <typename Problem>
    static inline constexpr auto wg_attr_num_access =
        std::is_same_v<remove_cvref_t<typename Problem::ADataType>, pk_fp4_t>
            ? WGAttrNumAccessEnum::Single
            : WGAttrNumAccessEnum::Double;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockFlatmm()
    {
        using ADataType = remove_cvref_t<typename Problem::ADataType>;
        using BDataType = remove_cvref_t<typename Problem::BDataType>;
        static_assert(
            sizeof(ADataType) * numeric_traits<BDataType>::PackedSize ==
                sizeof(BDataType) * numeric_traits<ADataType>::PackedSize,
            "sizeof(ADataType) / APackedSize must be equal to sizeof(BDataType) / BPackedSize!");
        using BlockWarps        = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile          = typename Problem::BlockGemmShape::WarpTile;
        using WarpGemm          = WarpGemmDispatcher< //
            ADataType,
            BDataType,
            typename Problem::CDataType,
            WarpTile::at(I0),
            WarpTile::at(I1),
            WarpTile::at(I2),
            Problem::TransposeC,
            false,
            false,
            wg_attr_num_access<Problem>>;
        using BlockFlatmmPolicy = BlockFlatmmASmemBSmemCRegV1CustomPolicy< //
            ADataType,
            BDataType,
            typename Problem::CDataType,
            BlockWarps,
            WarpGemm>;
        return BlockFlatmmASmemBSmemCRegV1<Problem, BlockFlatmmPolicy>{};
    }

    template <typename Problem, typename TensorView>
    CK_TILE_DEVICE static constexpr auto
    MakeMX_AAsyncLoadDramDescriptor(const TensorView& naive_view)
    {
        using ADataType           = remove_cvref_t<typename Problem::ADataType>;
        using ALayout             = remove_cvref_t<typename Problem::ALayout>;
        constexpr index_t MPerXdl = Problem::BlockGemmShape::WarpTile::at(I0);
        constexpr index_t NPerXdl = Problem::BlockGemmShape::WarpTile::at(I1);
        static_assert(MPerXdl == 16 && NPerXdl == 16);
        static_assert(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>);

        const auto& naive_desc = naive_view.get_tensor_descriptor();
        constexpr auto ndims   = remove_cvref_t<decltype(naive_desc)>::get_num_of_dimension();
        static_assert(ndims == 2, "only support 2D tensor");
        const auto rows = naive_desc.get_length(number<0>{});
        const auto cols = naive_desc.get_length(number<1>{});

        constexpr index_t APackedSize = numeric_traits<ADataType>::PackedSize;
        constexpr index_t K2          = GetSmemPackA<Problem>() * APackedSize; // f4=32; f8=16
        constexpr index_t K1          = kDramLoadPackBytes * APackedSize / K2; // 8
        const index_t K0              = cols / (K1 * K2);
        const auto col_lens           = make_tuple(K0, number<K1>{}, number<K2>{});

        constexpr index_t M1 = 4; // so that we can use imm offset to load lds
        const index_t M0     = rows / M1;
        const auto row_lens  = make_tuple(M0, number<M1>{});

        const auto desc_0 =
            make_naive_tensor_descriptor_packed(container_concat(row_lens, col_lens));
        const auto desc_1 = transform_tensor_descriptor(
            desc_0,
            make_tuple(make_pass_through_transform(M0),
                       make_xor_transform(make_tuple(number<M1>{}, number<K1>{})),
                       make_pass_through_transform(K0),
                       make_pass_through_transform(number<K2>{})),
            make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}, sequence<4>{}),
            make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}, sequence<4>{}));
        const auto desc = transform_tensor_descriptor( //
            desc_1,
            make_tuple(make_merge_transform_v3_division_mod(row_lens),
                       make_merge_transform_v3_division_mod(col_lens)),
            make_tuple(sequence<0, 1>{}, sequence<2, 3, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
        // printf("A async load dram desc %d x %d: \n", desc.get_length(I0), desc.get_length(I1));

        return tensor_view<typename TensorView::buffer_view,
                           remove_cvref_t<decltype(desc)>,
                           TensorView::DstInMemOp>{naive_view.buf_, desc};
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeMX_ADramTileDistribution()
    {

        using ADataType = remove_cvref_t<typename Problem::ADataType>;
        using ALayout   = remove_cvref_t<typename Problem::ALayout>;
        static_assert(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>);

        constexpr index_t BlockSize   = Problem::kBlockSize;
        constexpr index_t MPerBlock   = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock   = Problem::BlockGemmShape::kK;
        constexpr index_t APackedSize = numeric_traits<ADataType>::PackedSize;

        constexpr index_t K2 = GetSmemPackA<Problem>() * APackedSize; // f4=32; f8=16
        constexpr index_t K1 = kDramLoadPackBytes * APackedSize / K2; // 8
        constexpr index_t K0 = KPerBlock / (K1 * K2);                 // KPerBlock/256

        constexpr index_t M2 = get_warp_size() / K1;        // 8
        constexpr index_t M1 = BlockSize / get_warp_size(); // 4
        constexpr index_t M0 = MPerBlock / (M2 * M1);
        static_assert(M0 * M1 * M2 == MPerBlock, "M0, M1, M2 must cover whole MPerBlock!");
        static_assert(K0 * K1 * K2 == KPerBlock, "K0, K1, K2 must cover whole KPerBlock!");

        return make_static_tile_distribution(
            tile_distribution_encoding< //
                sequence<1>,
                tuple<sequence<M0, M1, M2>, sequence<K0, K1, K2>>, // ?,4,8 1,8,32 or 2,8,16
                tuple<sequence<1>, sequence<1, 2>>,                // M1 M2,K1
                tuple<sequence<1>, sequence<2, 1>>,
                sequence<1, 2, 2>, // M0,K0,K2
                sequence<0, 0, 2>>{});
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeMX_ALdsBlockDescriptor()
    {
        using ADataType           = remove_cvref_t<typename Problem::ADataType>;
        using ALayout             = remove_cvref_t<typename Problem::ALayout>;
        constexpr index_t MPerXdl = Problem::BlockGemmShape::WarpTile::at(I0);
        constexpr index_t NPerXdl = Problem::BlockGemmShape::WarpTile::at(I1);
        static_assert(MPerXdl == 16 && NPerXdl == 16);
        static_assert(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>);

        /*reduce transform layers,compare with old ck*/
        constexpr index_t MPerBlock   = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock   = Problem::BlockGemmShape::kK;
        constexpr index_t APackedSize = numeric_traits<ADataType>::PackedSize;
        constexpr index_t K2          = GetSmemPackA<Problem>() * APackedSize; // f4=32; f8=16
        constexpr index_t K1          = kDramLoadPackBytes * APackedSize / K2; // 8
        constexpr index_t K0          = KPerBlock / (K1 * K2);                 // KPerBlock/256
        static_assert(K0 * K1 * K2 == KPerBlock, "K0, K1, K2 must cover whole KPerBlock!");

        constexpr index_t M3 = 4; // so that we can use imm offset to load lds
        constexpr index_t M2 = get_warp_size() / K1 / M3;  // 2
        constexpr index_t M1 = MPerXdl / (M2 * M3);        // 2
        constexpr index_t M0 = MPerBlock / (M1 * M2 * M3); // MPerBlock/16
        static_assert(M0 * M1 * M2 * M3 == MPerBlock, "M0, M1, M2, M3 must cover whole MPerBlock!");

        constexpr index_t Pad = 4 * K2; // 4 * 32

        constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor( //
            make_tuple(number<M0>{},
                       number<M1>{},
                       number<K0>{},
                       number<M2>{},
                       number<M3>{},
                       number<K1>{},
                       number<K2>{}),
            make_tuple(number<M1*(K0 * (M2 * M3 * K1 * K2) + (K0 - 1) * Pad)>{},
                       number<K0*(M2 * M3 * K1 * K2) + (K0 - 1) * Pad>{},
                       number<M2 * M3 * K1 * K2 + Pad>{},
                       number<M3 * K1 * K2>{},
                       number<K1 * K2>{},
                       number<K2>{},
                       number<1>{}),
            number<K2>{},
            number<1>{});

        constexpr auto a_lds_block_desc_1 = transform_tensor_descriptor(
            a_lds_block_desc_0,
            make_tuple(make_pass_through_transform(M0),
                       make_pass_through_transform(M1),
                       make_pass_through_transform(K0),
                       make_pass_through_transform(M2),
                       make_xor_transform(make_tuple(number<M3>{}, number<K1>{})),
                       make_pass_through_transform(number<K2>{})),
            make_tuple(sequence<0>{},
                       sequence<1>{},
                       sequence<2>{},
                       sequence<3>{},
                       sequence<4, 5>{},
                       sequence<6>{}),
            make_tuple(sequence<0>{},
                       sequence<1>{},
                       sequence<2>{},
                       sequence<3>{},
                       sequence<4, 5>{},
                       sequence<6>{}));
        constexpr auto a_lds_block_desc = transform_tensor_descriptor(
            a_lds_block_desc_1,
            make_tuple(make_merge_transform_v3_division_mod(
                           make_tuple(number<M0>{}, number<M1>{}, number<M2>{}, number<M3>{})),
                       make_merge_transform_v3_division_mod(
                           make_tuple(number<K0>{}, number<K1>{}, number<K2>{}))),
            make_tuple(sequence<0, 1, 3, 4>{}, sequence<2, 5, 6>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        // return a_lds_block_desc_permuted;
        return a_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeMX_ALDS_TileDistribution()
    {
        using TileShape = typename Problem::BlockGemmShape;

        static_assert(TileShape::WarpTile::at(I1) == 16, "requires XDL_N == 16");
        static_assert(TileShape::BlockWarps::at(I0) == 1, "requires Wave_M == 1");

        constexpr int M_warps = TileShape::BlockWarps::at(number<0>{});
        constexpr int N_warps = TileShape::BlockWarps::at(number<1>{});
        constexpr int M_Lane  = TileShape::WarpTile::at(I0); // 16

        constexpr int K_Lane = 64 / M_Lane; // 4

        constexpr int K_Thread         = TileShape::WarpTile::at(I2) / K_Lane; // 32
        constexpr index_t num_access_v = static_cast<index_t>(wg_attr_num_access<Problem>);
        constexpr int K1               = K_Thread / num_access_v; // 16

        return make_static_tile_distribution(
            std::conditional_t<
                num_access_v == 1,
                tile_distribution_encoding<
                    sequence<N_warps>,
                    tuple<sequence<M_warps, MXdlPack, M_Lane>, sequence<K_Lane, K1>>,
                    tuple<sequence<1, 0>, sequence<2, 1>>,
                    tuple<sequence<0, 0>, sequence<0, 2>>,
                    sequence<2>,
                    sequence<1>>,
                tile_distribution_encoding< //
                    sequence<N_warps>,
                    tuple<sequence<M_warps, MXdlPack, M_Lane>, sequence<num_access_v, K_Lane, K1>>,
                    tuple<sequence<1, 0>, sequence<2, 1>>,
                    tuple<sequence<0, 0>, sequence<1, 2>>,
                    sequence<2, 2>,
                    sequence<0, 2>>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeMX_BFlatDramTileDistribution()
    {
        using TileShape = typename Problem::BlockGemmShape;

        static_assert(TileShape::WarpTile::at(I1) == 16, "only for XDL_N == 16");

        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t WaveSize  = get_warp_size();
        constexpr index_t WaveNum   = BlockSize / WaveSize;

        constexpr index_t K1          = WaveSize; // threads cnt in K dim
        constexpr index_t KWavePerBlk = 1;
        constexpr index_t K0          = KWavePerBlk;

        constexpr index_t NWavePerBlk = TileShape::BlockWarps::at(number<1>{}); // N_Warp

        constexpr index_t WaveRepeat   = WaveNum / TileShape::flatNPerWarp;
        constexpr index_t kKPerThread  = 32;
        constexpr index_t num_access_v = static_cast<index_t>(wg_attr_num_access<Problem>);
        constexpr index_t K2           = kKPerThread / num_access_v;

        return make_static_tile_distribution(
            std::conditional_t< //
                num_access_v == 1,
                tile_distribution_encoding< //
                    sequence<WaveRepeat>,
                    tuple<sequence<NWavePerBlk, NXdlPack>, // 4 2
                          sequence<K0, K1, K2>>,           // 1 64 32
                    tuple<sequence<0, 1, 2>, sequence<2>>,
                    tuple<sequence<0, 0, 0>, sequence<1>>,
                    sequence<2>,
                    sequence<2>>,
                tile_distribution_encoding< //
                    sequence<WaveRepeat>,
                    tuple<sequence<NWavePerBlk, NXdlPack>,     // 4 2
                          sequence<num_access_v, K0, K1, K2>>, // 2 1 64 16
                    tuple<sequence<0, 1, 2>, sequence<2>>,
                    tuple<sequence<0, 0, 1>, sequence<2>>,
                    sequence<2, 2>,
                    sequence<0, 3>>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeMX_ScaleA_DramTileDistribution()
    {
        using TileShape = typename Problem::BlockGemmShape; // ck_tile::TileFlatmmShape

        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t WaveSize  = get_warp_size();
        constexpr index_t WaveNum   = BlockSize / WaveSize;

        constexpr index_t kMPerBlock = TileShape::BlockTile::at(I0);

        constexpr index_t M_Warps = TileShape::BlockWarps::at(I0);
        constexpr index_t N_Warps = TileShape::BlockWarps::at(I1);

        static_assert(WaveNum == M_Warps * N_Warps, "Block warps do not match block size");

        constexpr index_t M_Lanes = TileShape::WarpTile::at(I0);
        constexpr index_t K_Lanes = 64 / M_Lanes;

        // Y dimension (M) decomposition
        constexpr index_t Y2 = M_Lanes;
        constexpr index_t Y1 = M_Warps;
        constexpr index_t Y0 = kMPerBlock / (MXdlPack * Y1 * Y2);

        // X dimension (K) decomposition
        constexpr index_t X0 = K_Lanes;
        constexpr index_t X1 = 1; // packed 2x2 E8M0 data into 1 int32_t for load

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<N_Warps>, // repeat N_warps
                                       tuple<sequence<Y0, Y1, Y2>, sequence<X0, X1>>,
                                       tuple<sequence<1, 0>, sequence<2, 1>>,
                                       tuple<sequence<1, 0>, sequence<0, 2>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeMX_ScaleB_DramTileDistribution()
    {
        using TileShape = typename Problem::BlockGemmShape; // ck_tile::TileFlatmmShape

        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t WaveSize  = get_warp_size();
        constexpr index_t WaveNum   = BlockSize / WaveSize;

        constexpr index_t kNPerBlock = TileShape::BlockTile::at(I1);

        constexpr index_t M_Warps = TileShape::BlockWarps::at(I0);
        constexpr index_t N_Warps = TileShape::BlockWarps::at(I1);

        static_assert(WaveNum == M_Warps * N_Warps, "Block warps do not match block size");

        constexpr index_t N_Lanes = TileShape::WarpTile::at(I1);
        constexpr index_t K_Lanes = 64 / N_Lanes;

        // Y dimension (M) decomposition
        constexpr index_t Y2 = N_Lanes;
        constexpr index_t Y1 = N_Warps;
        constexpr index_t Y0 = kNPerBlock / (NXdlPack * Y1 * Y2);

        // X dimension (K) decomposition
        constexpr index_t X0 = K_Lanes;
        constexpr index_t X1 = 1; // packed 2x2 E8M0 data into 1 int32_t for load

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<M_Warps>, // ?
                                       tuple<sequence<Y0, Y1, Y2>, sequence<X0, X1>>,
                                       tuple<sequence<0, 1>, sequence<2, 1>>,
                                       tuple<sequence<0, 1>, sequence<0, 2>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeMX_ScaleA_FlatDramTileDistribution()
    {
        using TileShape = typename Problem::BlockGemmShape;

        constexpr index_t M_Warp      = TileShape::BlockWarps::at(number<0>{});
        constexpr index_t K_Lane      = 64 / TileShape::WarpTile::at(I0);
        constexpr index_t M_Lane      = TileShape::WarpTile::at(I0);
        constexpr index_t N_Wrap      = TileShape::BlockWarps::at(number<1>{});
        constexpr index_t MWavePerBlk = M_Warp;

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<N_Wrap>,                      // ?
                                       tuple<sequence<MWavePerBlk, M_Lane>,   // second direction
                                             sequence<K_Lane, 1>>,            // first direction
                                       tuple<sequence<1, 0>, sequence<2, 1>>, // which direction
                                       tuple<sequence<0, 0>, sequence<0, 1>>, // which index
                                       // <repeat, vec_load>
                                       sequence<2>,
                                       sequence<1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeMX_ScaleB_FlatDramTileDistribution()
    {
        using TileShape = typename Problem::BlockGemmShape;

        constexpr index_t N_Warp      = TileShape::BlockWarps::at(number<1>{});
        constexpr index_t K_Lane      = 64 / TileShape::WarpTile::at(I1);
        constexpr index_t N_Lane      = TileShape::WarpTile::at(I1);
        constexpr index_t M_Wrap      = TileShape::BlockWarps::at(number<0>{});
        constexpr index_t NWavePerBlk = N_Warp;

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<M_Wrap>,                      // ?
                                       tuple<sequence<NWavePerBlk, N_Lane>,   // second direction
                                             sequence<K_Lane, 1>>,            // first direction
                                       tuple<sequence<0, 1>, sequence<2, 1>>, // which direction
                                       tuple<sequence<0, 0>, sequence<0, 1>>, // which index
                                       // <repeat, vec_load>
                                       sequence<2>,
                                       sequence<1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeA()
    {
        using ADataType               = remove_cvref_t<typename Problem::ADataType>;
        constexpr index_t APackedSize = numeric_traits<ADataType>::PackedSize;
        return sizeof(ADataType) * MakeMX_ALdsBlockDescriptor<Problem>().get_element_space_size() /
               APackedSize;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return GetSmemSizeA<Problem>();
    }
};

} // namespace ck_tile
