// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp"

namespace ck_tile {

namespace detail {
template <typename Problem>
struct MXFlatmmPipelineAgBgCrPolicy : UniversalFlatmmPipelineAgBgCrPolicy
{
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};

    static constexpr index_t kDramLoadPackBytes = 128;
    static constexpr index_t DWORDx4            = 16;
    static constexpr index_t DWORDx3            = 12;

    static constexpr int MXdlPack = Problem::MXdlPack;
    static constexpr int NXdlPack = Problem::NXdlPack;
    static constexpr int KXdlPack = Problem::KXdlPack;

    private:
    using ADataType                      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType                      = remove_cvref_t<typename Problem::BDataType>;
    static constexpr index_t APackedSize = numeric_traits<ADataType>::PackedSize;
    static constexpr index_t BPackedSize = numeric_traits<BDataType>::PackedSize;

    using ALayout = remove_cvref_t<typename Problem::ALayout>;
    static_assert(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>);

    using TileShape                    = typename Problem::BlockGemmShape;
    using BlockWarps                   = typename TileShape::BlockWarps;
    static constexpr index_t BlockSize = Problem::kBlockSize;
    static constexpr index_t WaveSize  = get_warp_size();
    static constexpr index_t WaveNum   = BlockSize / WaveSize;

    static constexpr index_t MPerBlock = TileShape::kM;
    static constexpr index_t NPerBlock = TileShape::kN;
    static constexpr index_t KPerBlock = TileShape::kK;
    static constexpr index_t MWarps    = BlockWarps::at(I0);
    static constexpr index_t NWarps    = BlockWarps::at(I1);
    static_assert(WaveNum == MWarps * NWarps, "Block warps do not match block size");

    static constexpr index_t MPerXdl = TileShape::WarpTile::at(I0);
    static constexpr index_t NPerXdl = TileShape::WarpTile::at(I1);
    static constexpr index_t KPerXdl = TileShape::WarpTile::at(I2);
    static_assert(MPerXdl == 16 && NPerXdl == 16);
    static constexpr index_t K_Lane   = get_warp_size() / 16;
    static constexpr index_t K_Thread = KPerXdl / K_Lane;

    public:
    static constexpr index_t AK1 = DWORDx4 * APackedSize;
    static constexpr index_t BK1 = DWORDx4 * BPackedSize;

    CK_TILE_HOST_DEVICE static constexpr auto GetBlockFlatmm()
    {
        using WarpGemm          = WarpGemmDispatcher< //
            ADataType,
            BDataType,
            typename Problem::CDataType,
            MPerXdl,
            NPerXdl,
            KPerXdl,
            Problem::TransposeC>;
        using BlockFlatmmPolicy = BlockFlatmmASmemBSmemCRegV1CustomPolicy< //
            ADataType,
            BDataType,
            typename Problem::CDataType,
            BlockWarps,
            WarpGemm>;
        return BlockFlatmmASmemBSmemCRegV1<Problem, BlockFlatmmPolicy>{};
    }

    CK_TILE_DEVICE static constexpr auto MakeMX_ABytesDramTileDistribution()
    {
        constexpr index_t K2 = std::is_same_v<ADataType, pk_fp6x16_t> ? DWORDx3 : DWORDx4;
        constexpr index_t K1 = kDramLoadPackBytes / DWORDx4; // fp8/fp6/fp4 K1 equal to 8
        constexpr index_t K0 =
            KPerBlock / APackedSize * sizeof(ADataType) / (K1 * K2); // KPerBlock/256/packsize

        constexpr index_t M2 = WaveSize / K1;
        constexpr index_t M1 = BlockSize / WaveSize;
        constexpr index_t M0 = MPerBlock / (M2 * M1);
        static_assert(M0 * M1 * M2 == MPerBlock, "M0, M1, M2 must cover whole MPerBlock!");
        static_assert(K0 * K1 * K2 == KPerBlock / APackedSize * sizeof(ADataType),
                      "K0, K1, K2 must cover whole KPerBlock!");

        return make_static_tile_distribution(
            tile_distribution_encoding< //
                sequence<1>,
                tuple<sequence<M0, M1, M2>, sequence<K0, K1, K2>>, // ?,4,8 1,8,32 or 2,8,16
                tuple<sequence<1>, sequence<1, 2>>,                // M1 M2,K1
                tuple<sequence<1>, sequence<2, 1>>,
                sequence<1, 2, 2>, // M0,K0,K2
                sequence<0, 0, 2>>{});
    }

    template <typename WindowTmp>
    CK_TILE_DEVICE static constexpr auto
    MakeMX_AAsyncLoadBytesDramWindow(const WindowTmp& window_tmp)
    {
        constexpr auto ndims = std::decay_t<decltype(window_tmp)>::get_num_of_dimension();
        static_assert(ndims == 2, "only support 2D tensor");
        auto&& tensor_view_tmp  = window_tmp.get_bottom_tensor_view();
        const auto [rows, cols] = tensor_view_tmp.get_tensor_descriptor().get_lengths();

        constexpr index_t K2 = std::is_same_v<ADataType, pk_fp6x16_t> ? DWORDx3 : DWORDx4;
        constexpr index_t K1 = kDramLoadPackBytes / DWORDx4; // fp8/fp6/fp4 K1 equal to 8
        const index_t K0     = cols / (K1 * K2 / sizeof(ADataType) * APackedSize);
        const auto col_lens  = make_tuple(K0, number<K1>{}, number<K2>{});

        constexpr index_t M1 = 4; // so that we can use imm offset to load lds
        const index_t M0     = integer_divide_ceil(rows, M1);
        const auto row_lens  = make_tuple(M0, number<M1>{});

        const auto d0 = make_naive_tensor_descriptor_packed(container_concat(row_lens, col_lens));
        const auto desc_0 = decltype(d0)( // set correct size (without padding)
            d0.get_transforms(),
            tensor_view_tmp.get_tensor_descriptor().get_element_space_size());
#ifdef __gfx950__
        //  In gfx950, direct_to_lds is used, so, xor is applied to dram read and lds read. (xor in
        //  lds write is ignored).
        //  In gfx1250, we use async load from DRAM, so, if we want to use lds, we need to use xor
        //  in lds write and lds read.

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
#else
        // gfx1250 branch without XOR transform
        const auto desc =
            transform_tensor_descriptor(desc_0,
                                        make_tuple(make_merge_transform_v3_division_mod(row_lens),
                                                   make_merge_transform_v3_division_mod(col_lens)),
                                        make_tuple(sequence<0, 1>{}, sequence<2, 3, 4>{}),
                                        make_tuple(sequence<0>{}, sequence<1>{}));
#endif

        auto&& byte_ptr = reinterpret_cast<const uint8_t*>(&(tensor_view_tmp.get_buffer_view()(0)));
        auto&& byte_tensor_view = make_tensor_view<address_space_enum::global>(byte_ptr, desc);

        auto&& origin_tmp       = window_tmp.get_window_origin();
        constexpr index_t test1 = APackedSize / sizeof(ADataType);
        return make_tile_window(byte_tensor_view,
                                make_tuple(number<MPerBlock>{}, number<KPerBlock / test1>{}),
                                {origin_tmp[0], origin_tmp[1] / test1},
                                MakeMX_ABytesDramTileDistribution());
    }

    CK_TILE_DEVICE static constexpr auto MakeMX_ALdsBytesBlockDescriptor()
    {
        constexpr index_t K2 = std::is_same_v<ADataType, pk_fp6x16_t> ? DWORDx3 : AK1 / APackedSize;
        constexpr index_t K2_Pad = 16;
        constexpr index_t K1     = kDramLoadPackBytes / DWORDx4; // 8
        constexpr index_t K0     = std::is_same_v<ADataType, pk_fp6x16_t>
                                       ? KPerBlock / (K1 * K2 / sizeof(ADataType) * APackedSize)
                                       : KPerBlock / (K1 * AK1); // KPerBlock/256
        static_assert(K0 * K1 * K2 / sizeof(ADataType) * APackedSize == KPerBlock,
                      "K0, K1, K2 must cover whole KPerBlock!");

        constexpr index_t M3 = 4; // so that we can use imm offset to load lds
        constexpr index_t M2 = WaveSize / K1 / M3;
        constexpr index_t M1 = MPerXdl / (M2 * M3);
        constexpr index_t M0 = MPerBlock / (M1 * M2 * M3); // MPerBlock/16
        static_assert(M0 * M1 * M2 * M3 == MPerBlock, "M0, M1, M2, M3 must cover whole MPerBlock!");

        constexpr index_t Pad = 4 * K2; // 4 dwords

        constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<M0>{},
                       number<K0>{},
                       number<M1>{},
                       number<M2>{},
                       number<M3>{},
                       number<K1>{},
                       number<K2>{}),
            make_tuple(number<K0*(M1 * (M2 * M3 * K1 * K2_Pad) + (M1 - 1) * Pad)>{},
                       number<M1*(M2 * M3 * K1 * K2_Pad) + (M1 - 1) * Pad>{},
                       number<M2 * M3 * K1 * K2_Pad + Pad>{},
                       number<M3 * K1 * K2_Pad>{},
                       number<K1 * K2_Pad>{},
                       number<K2_Pad>{},
                       number<1>{}),
            number<K2>{},
            number<1>{});

        constexpr auto a_lds_block_desc_1 = transform_tensor_descriptor(
            a_lds_block_desc_0,
            make_tuple(make_pass_through_transform(M0),
                       make_pass_through_transform(K0),
                       make_pass_through_transform(M1),
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
            make_tuple(sequence<0, 2, 3, 4>{}, sequence<1, 5, 6>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        // return a_lds_block_desc_permuted;
        return a_lds_block_desc;
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeMX_ALDSBytes_TileDistribution()
    {
        static_assert(BlockWarps::at(I0) == 1, "requires Wave_M == 1");

        if constexpr(std::is_same_v<ADataType, pk_fp4_t>)
        {
#if defined(__gfx1250__)
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<NWarps>,
                    tuple<sequence<MWarps, MXdlPack, MPerXdl>,
                          // gfx950:    32  / 32     4      32  / 2      == 64
                          // gfx1250:   64  / 32     2      32  / 2      == 64
                          sequence<K_Thread / AK1, K_Lane, AK1 / APackedSize>>,
                    tuple<sequence<1, 0>, sequence<2, 1>>,
                    tuple<sequence<0, 0>, sequence<1, 2>>,
                    sequence<2, 2>,
                    sequence<0, 2>>{});
#else

            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<NWarps>,
                    // gfx950:                                             4     32  / 2 == 64
                    // gfx1250:                                            2     32  / 2 == 32
                    tuple<sequence<MWarps, MXdlPack, MPerXdl>, sequence<K_Lane, AK1 / APackedSize>>,
                    tuple<sequence<1, 0>, sequence<2, 1>>,
                    tuple<sequence<0, 0>, sequence<0, 2>>,
                    sequence<2>,
                    sequence<1>>{});
#endif
        }
        else if constexpr(std::is_same_v<ADataType, fp8_t>)
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<NWarps>,
                    tuple<sequence<MWarps, MXdlPack, MPerXdl>,
                          // gfx950:    32  / 16     4      16  / 1      == 128
                          // gfx1250:   64  / 16     2      16  / 1      == 128
                          sequence<K_Thread / AK1, K_Lane, AK1 / APackedSize>>,
                    tuple<sequence<1, 0>, sequence<2, 1>>,
                    tuple<sequence<0, 0>, sequence<1, 2>>,
                    sequence<2, 2>,
                    sequence<0, 2>>{});
        else if constexpr(std::is_same_v<ADataType, pk_fp6x16_t>)
        {
#if defined(__gfx1250__)
            // gfx1250: 4-element K-tuple <K_Lane, block-parity, sub-slot, DWORDx3>
            //          = <2, 2, 2, 12>.
            // Warp lane (P1) spans block-parity x MPerXdl = 2 x 16 = 32.
            // Y dims (per-thread) span K_Lane x sub-slot x DWORDx3 = 48 bytes.
            // Result: thread 0 -> K={0..31, 64..95}, thread 16 -> K={32..63, 96..127},
            // matching the B-side layout so A.K and B.K align in the WMMA.

            // Register Mapping for 16x128 for FP6:
            //                      K_Lane1       K_Lane2
            // Size              |   BLOCK_M  |   BLOCK_M   |
            // M                 | 0  ...  15 |  0  ...  15 |
            // Thread Id         | 0  ...  15 | 16  ...  31 |
            // Register Element  |------------|-------------|
            // Reg 0 - 5         |    K0-K31  |    K32-K63  | - block-parity1
            // Reg 6 - 11        |    K64-K95 |    K96-K127 | - block-parity2
            // K0-K15 = subslot1
            // K16-K31 = subslot2
            // sizeof(subslot1) = sizeof(subslot2) = sizeof(DWORDx3) = 12 bytes

            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<NWarps>,
                    tuple<sequence<MWarps, MXdlPack, MPerXdl>,
                          // gfx1250: 2 2 2 12 = 96
                          sequence<K_Lane, KPerXdl / (K_Lane * APackedSize) / 2, 2, DWORDx3>>,
                    tuple<sequence<1, 0>, sequence<2, 1>>,
                    tuple<sequence<0, 0>, sequence<1, 2>>,
                    sequence<2, 2, 2>,
                    sequence<0, 2, 3>>{});
#else
            return make_static_tile_distribution(
                tile_distribution_encoding< //
                    sequence<NWarps>,
                    tuple<sequence<MWarps, MXdlPack, MPerXdl>,
                          // gfx950: 4       128   /  (4     *    16)          12 = 96
                          sequence<K_Lane, KPerXdl / (K_Lane * APackedSize), DWORDx3>>,
                    tuple<sequence<1, 0>, sequence<2, 1>>,
                    tuple<sequence<0, 0>, sequence<0, 2>>,
                    sequence<2, 2>,
                    sequence<1, 2>>{});
#endif
        }
        else
            static_assert(false, "unsupported datatype");
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeMX_BFlatBytesDramTileDistribution()
    {
        constexpr index_t K1          = WaveSize; // threads cnt in K dim
        constexpr index_t KWavePerBlk = 1;
        constexpr index_t K0          = KWavePerBlk;

        constexpr index_t WaveRepeat = WaveNum / TileShape::flatNPerWarp;

        if constexpr(std::is_same_v<BDataType, pk_fp4_t>)
        {
#if defined(__gfx1250__)
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    //           1
                    sequence<WaveRepeat>,
                    //                4        2
                    tuple<sequence<NWarps, NXdlPack>,
                          //           64   / 32   1   32      16   = 64*16
                          sequence<K_Thread / BK1, K0, K1, BK1 / BPackedSize>>,
                    tuple<sequence<0, 1, 2>, sequence<2>>,
                    tuple<sequence<0, 0, 1>, sequence<2>>,
                    sequence<2, 2>,
                    sequence<0, 3>>{});

#else
            return make_static_tile_distribution(tile_distribution_encoding<
                                                 // gfx1250: 1
                                                 sequence<WaveRepeat>,
                                                 // gfx1250:      4        2
                                                 tuple<sequence<NWarps, NXdlPack>,
                                                       // gfx950:1  64  32  / 2 = 64*16
                                                       // gfx1250:1 32  32  / 2 = 32*16
                                                       sequence<K0, K1, BK1 / BPackedSize>>,
                                                 tuple<sequence<0, 1, 2>, sequence<2>>,
                                                 tuple<sequence<0, 0, 0>, sequence<1>>,
                                                 sequence<2>,
                                                 sequence<2>>{});
#endif
        }
        else if constexpr(std::is_same_v<BDataType, fp8_t>)
        {
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    // gfx950/1250:  1
                    sequence<WaveRepeat>,
                    // gfx950/1250:   4        2
                    tuple<sequence<NWarps, NXdlPack>,
                          // gfx950:   32   / 16   1   64    16 = 128*16
                          // gfx1250:  64   / 16   1   32    16 = 128*16
                          sequence<K_Thread / BK1, K0, K1, BK1 / BPackedSize>>,
                    tuple<sequence<0, 1, 2>, sequence<2>>,
                    tuple<sequence<0, 0, 1>, sequence<2>>,
                    sequence<2, 2>,
                    sequence<0, 3>>{});
        }
        else if constexpr(std::is_same_v<ADataType, pk_fp6x16_t>)
        {
#if defined(__gfx1250__)
            // Use a 5-element K-tuple <2, 1, 32, 2, 12> so the warp lane (P1) can
            // span (K0=1) x (K1=32) = 32 cleanly, and per-thread Y dims cover the
            // three remaining K-tuple positions (0, 3, 4) sized <2, 2, 12> = 48 B.
            //   K-tuple: <K_Thread*sizeof/(DWORDx3*BPackedSize)/K_Lane,  // 2 (macro)
            //             K0,                                            // 1
            //             K1,                                            // 32
            //             K_Lane,                                        // 2 (sub)
            //             DWORDx3>                                       // 12 (bytes)
            //   Ps2RHssMajor = <<0,1,2>, <2,2>>      P0=warp spans (R,N,K0); P1=lane spans (K1, K0)
            //   Ps2RHssMinor = <<0,0,1>, <1,2>>      P1 minors -> K0(=1) * K1(=32) = 32 lanes
            //   Ys2RHsMajor  = <2, 2, 2>             three Y dims all in K-tuple
            //   Ys2RHsMinor  = <0, 3, 4>             at positions 0, 3, 4 = <2, 2, 12> = 48 bytes
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<WaveRepeat>,
                    tuple<sequence<NWarps, NXdlPack>,
                          // gfx1250: 2 1 32 2 12
                          sequence<K_Thread * sizeof(BDataType) / (DWORDx3 * BPackedSize) / K_Lane,
                                   K0,
                                   K1,
                                   K_Lane,
                                   DWORDx3>>,
                    tuple<sequence<0, 1, 2>, sequence<2, 2>>,
                    tuple<sequence<0, 0, 1>, sequence<1, 2>>,
                    sequence<2, 2, 2>,
                    sequence<0, 3, 4>>{});
#else
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<WaveRepeat>,
                    tuple<sequence<NWarps, NXdlPack>,
                          // gfx950:  1 64 2 12 = 128 * 12
                          sequence<K0,
                                   K1,
                                   K_Thread * sizeof(BDataType) / (DWORDx3 * BPackedSize),
                                   DWORDx3>>,
                    tuple<sequence<0, 1, 2>, sequence<2>>,
                    tuple<sequence<0, 0, 0>, sequence<1>>,
                    sequence<2, 2>,
                    sequence<2, 3>>{});
#endif
        }
        else
            static_assert(false, "unsupported datatype");
    }

    template <typename WindowTmp>
    CK_TILE_HOST_DEVICE static constexpr auto
    MakeMX_BFlatBytesDramWindow(const WindowTmp& window_tmp)
    {
        constexpr auto M_Warp_Tile  = Problem::BlockGemmShape::WarpTile::at(I1);
        constexpr auto flatNPerWarp = Problem::BlockGemmShape::flatNPerWarp;
        constexpr auto flatKPerWarp = Problem::BlockGemmShape::flatKPerWarp;

        static_assert(std::decay_t<decltype(window_tmp)>::get_num_of_dimension() == 2);
        auto&& tensor_view_tmp          = window_tmp.get_bottom_tensor_view();
        const auto [flat_n, flat_k]     = tensor_view_tmp.get_tensor_descriptor().get_lengths();
        constexpr auto flat_k_per_block = KPerBlock * M_Warp_Tile;
        auto&& byte_tensor_desc         = transform_tensor_descriptor(
            make_naive_tensor_descriptor_packed(
                make_tuple(flat_n,
                           flat_k / flat_k_per_block,
                           number<flat_k_per_block / BPackedSize * sizeof(BDataType)>{})),
            make_tuple(make_pass_through_transform(flat_n),
                       make_merge_transform_v3_division_mod(make_tuple(
                           flat_k / flat_k_per_block,
                           number<flat_k_per_block / BPackedSize * sizeof(BDataType)>{}))),
            make_tuple(sequence<0>{}, sequence<1, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
        auto&& byte_ptr = reinterpret_cast<const uint8_t*>(&(tensor_view_tmp.get_buffer_view()(0)));
        auto&& byte_tensor_view =
            make_tensor_view<address_space_enum::global>(byte_ptr, byte_tensor_desc);
        auto&& origin_tmp = window_tmp.get_window_origin();
        auto origin_n     = origin_tmp[0];
        auto origin_k     = static_cast<int>(origin_tmp[1] * sizeof(BDataType) / BPackedSize);
        return make_tile_window(
            byte_tensor_view,
            make_tuple(number<flatNPerWarp>{},
                       number<flatKPerWarp * sizeof(BDataType) / BPackedSize>{}),
            {origin_n, origin_k},
            MakeMX_BFlatBytesDramTileDistribution());
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeMX_ScaleA_DramTileDistribution()
    {
        constexpr index_t M_Lanes = TileShape::WarpTile::at(I0);
        constexpr index_t K_Lanes = get_warp_size() / M_Lanes;

        // Y dimension (M) decomposition
        constexpr index_t Y2 = M_Lanes;
        constexpr index_t Y1 = MWarps;
        constexpr index_t Y0 = MPerBlock / (MXdlPack * Y1 * Y2);

        // X dimension (K) decomposition
        constexpr index_t X0 = K_Lanes;
        constexpr index_t X1 = 1; // packed 2x2 E8M0 data into 1 int32_t for load

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<NWarps>, // repeat NWarps
                                       tuple<sequence<Y0, Y1, Y2>, sequence<X0, X1>>,
                                       tuple<sequence<1, 0>, sequence<2, 1>>,
                                       tuple<sequence<1, 0>, sequence<0, 2>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeMX_ScaleB_DramTileDistribution()
    {
        constexpr index_t N_Lanes = TileShape::WarpTile::at(I1);
        constexpr index_t K_Lanes = get_warp_size() / N_Lanes;

        // Y dimension (M) decomposition
        constexpr index_t Y2 = N_Lanes;
        constexpr index_t Y1 = NWarps;
        constexpr index_t Y0 = NPerBlock / (NXdlPack * Y1 * Y2);

        // X dimension (K) decomposition
        constexpr index_t X0 = K_Lanes;
        constexpr index_t X1 = 1; // packed 2x2 E8M0 data into 1 int32_t for load

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<MWarps>, // ?
                                       tuple<sequence<Y0, Y1, Y2>, sequence<X0, X1>>,
                                       tuple<sequence<0, 1>, sequence<2, 1>>,
                                       tuple<sequence<0, 1>, sequence<0, 2>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    CK_TILE_DEVICE static constexpr auto Make_Wave32_MX_ScaleA_DramTileDistribution()
    {
        constexpr index_t kMPerBlock = TileShape::kM;
        constexpr index_t kKPerBlock = TileShape::kK;

        constexpr index_t BlockScaleSize = Problem::BlockScaleSize;
        constexpr index_t ScalePack      = 4; // 4 scale values per packed int32_t

        constexpr index_t MIterPerWarp = kMPerBlock / MWarps / MPerXdl;

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<NWarps>,
                                       tuple<sequence<MIterPerWarp, MWarps, get_warp_size()>,
                                             sequence<kKPerBlock / BlockScaleSize / ScalePack, 1>>,
                                       tuple<sequence<1, 0>, sequence<1>>,
                                       tuple<sequence<1, 0>, sequence<2>>,
                                       sequence<1, 2, 2>,
                                       sequence<0, 0, 1>>{});
    }

    CK_TILE_DEVICE static constexpr auto Make_Wave32_MX_ScaleB_DramTileDistribution()
    {

        constexpr index_t kNPerBlock = TileShape::kN;
        constexpr index_t kKPerBlock = TileShape::kK;

        constexpr index_t BlockScaleSize = Problem::BlockScaleSize;
        constexpr index_t ScalePack      = 4; // 4 scale values per packed int32_t

        constexpr index_t NIterPerWarp = kNPerBlock / NWarps / NPerXdl;

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<MWarps>,
                                       tuple<sequence<NIterPerWarp, NWarps, get_warp_size()>,
                                             sequence<kKPerBlock / BlockScaleSize / ScalePack, 1>>,
                                       tuple<sequence<0, 1>, sequence<1>>,
                                       tuple<sequence<0, 1>, sequence<2>>,
                                       sequence<1, 2, 2>,
                                       sequence<0, 0, 1>>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeMX_ScaleA_FlatDramTileDistribution()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<NWarps>,                      // ?
                                       tuple<sequence<MWarps, MPerXdl>,       // second direction
                                             sequence<K_Lane, 1>>,            // first direction
                                       tuple<sequence<1, 0>, sequence<2, 1>>, // which direction
                                       tuple<sequence<0, 0>, sequence<0, 1>>, // which index
                                       // <repeat, vec_load>
                                       sequence<2>,
                                       sequence<1>>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeMX_ScaleB_FlatDramTileDistribution()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<MWarps>,                      // ?
                                       tuple<sequence<NWarps, NPerXdl>,       // second direction
                                             sequence<K_Lane, 1>>,            // first direction
                                       tuple<sequence<0, 1>, sequence<2, 1>>, // which direction
                                       tuple<sequence<0, 0>, sequence<0, 1>>, // which index
                                       // <repeat, vec_load>
                                       sequence<2>,
                                       sequence<1>>{});
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeA()
    {
        if constexpr(!std::is_same_v<ADataType, pk_fp6x16_t>)
        {
            return sizeof(ADataType) * MakeMX_ALdsBytesBlockDescriptor().get_element_space_size();
        }
        else
        {
            return MakeMX_ALdsBytesBlockDescriptor().get_element_space_size();
        }
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return GetSmemSizeA(); }
};
} // namespace detail

struct MXFlatmmPipelineAgBgCrPolicy
{

#define FORWARD_METHOD_(method)                                                                    \
    template <typename Problem, typename... Args>                                                  \
    CK_TILE_HOST_DEVICE static constexpr auto method(Args&&... args)                               \
    {                                                                                              \
        return detail::MXFlatmmPipelineAgBgCrPolicy<Problem>::method(std::forward<Args>(args)...); \
    }

    FORWARD_METHOD_(GetBlockFlatmm);
    FORWARD_METHOD_(MakeMX_AAsyncLoadBytesDramWindow);
    FORWARD_METHOD_(MakeMX_ABytesDramTileDistribution);
    FORWARD_METHOD_(MakeMX_ALdsBytesBlockDescriptor);
    FORWARD_METHOD_(MakeMX_ALDSBytes_TileDistribution);
    FORWARD_METHOD_(MakeMX_BFlatBytesDramTileDistribution);
    FORWARD_METHOD_(MakeMX_BFlatBytesDramWindow);
    FORWARD_METHOD_(MakeMX_ScaleA_DramTileDistribution);
    FORWARD_METHOD_(MakeMX_ScaleB_DramTileDistribution);
    FORWARD_METHOD_(Make_Wave32_MX_ScaleA_DramTileDistribution);
    FORWARD_METHOD_(Make_Wave32_MX_ScaleB_DramTileDistribution);
    FORWARD_METHOD_(MakeMX_ScaleA_FlatDramTileDistribution);
    FORWARD_METHOD_(MakeMX_ScaleB_FlatDramTileDistribution);
    FORWARD_METHOD_(GetSmemSizeA);
    FORWARD_METHOD_(GetSmemSize);

#undef FORWARD_METHOD_
};

} // namespace ck_tile
