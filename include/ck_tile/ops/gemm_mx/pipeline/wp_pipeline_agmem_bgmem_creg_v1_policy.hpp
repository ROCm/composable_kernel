// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm_mx/block/block_mx_asmem_breg_creg.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"

namespace ck_tile {

namespace detail {
template <typename Problem>
struct MXGemmPipelineAgBgCrPolicy : UniversalGemmPipelineAgBgCrPolicy
{
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};

    static constexpr index_t kDramLoadPackBytes = 128;
    static constexpr index_t DWORDx4            = 16;

    static constexpr int MXdlPack = 2;
    static constexpr int NXdlPack = 2;
    static constexpr int KXdlPack = 2;

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

    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using WarpGemm        = WarpGemmDispatcher<ADataType,
                                                   BDataType,
                                                   typename Problem::CDataType,
                                                   MPerXdl,
                                                   NPerXdl,
                                                   KPerXdl,
                                                   Problem::TransposeC>;
        using BlockGemmPolicy = BlockGemmASmemBSmemCRegV1CustomPolicy<ADataType,
                                                                      BDataType,
                                                                      typename Problem::CDataType,
                                                                      BlockWarps,
                                                                      WarpGemm>;
        return BlockMXGemmASmemBRegCReg<Problem, BlockGemmPolicy, MXdlPack, NXdlPack, KXdlPack>{};
    }

    CK_TILE_DEVICE static constexpr auto MakeMX_ABytesDramTileDistribution()
    {
        constexpr index_t K2 = DWORDx4;
        constexpr index_t K1 = kDramLoadPackBytes / DWORDx4;
        constexpr index_t K0 = KPerBlock / APackedSize * sizeof(ADataType) / (K1 * K2);

        constexpr index_t M2 = WaveSize / K1;
        constexpr index_t M1 = BlockSize / WaveSize;
        constexpr index_t M0 = MPerBlock / (M2 * M1);
        static_assert(M0 * M1 * M2 == MPerBlock, "M0, M1, M2 must cover whole MPerBlock!");
        static_assert(K0 * K1 * K2 == KPerBlock / APackedSize * sizeof(ADataType),
                      "K0, K1, K2 must cover whole KPerBlock!");

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<M0, M1, M2>, sequence<K0, K1, K2>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 1>>,
                                       sequence<1, 2, 2>,
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

        constexpr index_t K2 = DWORDx4;
        constexpr index_t K1 = kDramLoadPackBytes / DWORDx4;
        const index_t K0     = cols / (K1 * K2 / sizeof(ADataType) * APackedSize);
        const auto col_lens  = make_tuple(K0, number<K1>{}, number<K2>{});

        constexpr index_t M1 = 4;
        const index_t M0     = integer_divide_ceil(rows, M1);
        const auto row_lens  = make_tuple(M0, number<M1>{});

        const auto d0 = make_naive_tensor_descriptor_packed(container_concat(row_lens, col_lens));
        const auto desc_0 = decltype(d0)(
            d0.get_transforms(), tensor_view_tmp.get_tensor_descriptor().get_element_space_size());
        const auto desc_1 = transform_tensor_descriptor(
            desc_0,
            make_tuple(make_pass_through_transform(M0),
                       make_xor_transform(make_tuple(number<M1>{}, number<K1>{})),
                       make_pass_through_transform(K0),
                       make_pass_through_transform(number<K2>{})),
            make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}, sequence<4>{}),
            make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}, sequence<4>{}));
        const auto desc =
            transform_tensor_descriptor(desc_1,
                                        make_tuple(make_merge_transform_v3_division_mod(row_lens),
                                                   make_merge_transform_v3_division_mod(col_lens)),
                                        make_tuple(sequence<0, 1>{}, sequence<2, 3, 4>{}),
                                        make_tuple(sequence<0>{}, sequence<1>{}));

        auto&& byte_ptr = reinterpret_cast<const uint8_t*>(&(tensor_view_tmp.get_buffer_view()(0)));
        auto&& byte_tensor_view = make_tensor_view<address_space_enum::global>(byte_ptr, desc);

        auto&& origin_tmp = window_tmp.get_window_origin();
        return make_tile_window(
            byte_tensor_view,
            make_tuple(number<MPerBlock>{}, number<KPerBlock / APackedSize * sizeof(ADataType)>{}),
            {origin_tmp[0], origin_tmp[1] / APackedSize * static_cast<index_t>(sizeof(ADataType))},
            MakeMX_ABytesDramTileDistribution());
    }

    CK_TILE_DEVICE static constexpr auto MakeMX_ALdsBytesBlockDescriptor()
    {
        constexpr index_t K2     = AK1 / APackedSize;
        constexpr index_t K2_Pad = 16;
        constexpr index_t K1     = kDramLoadPackBytes / DWORDx4;
        constexpr index_t K0     = KPerBlock * sizeof(ADataType) / (K1 * AK1);
        static_assert(K0 >= 1,
                      "KPerBlock is too small for the selected ADataType and tile dimensions");
        static_assert(K0 * K1 * K2 / sizeof(ADataType) * APackedSize == KPerBlock,
                      "K0, K1, K2 must cover whole KPerBlock!");

        constexpr index_t M3 = 4;
        constexpr index_t M2 = WaveSize / K1 / M3;
        constexpr index_t M1 = MPerXdl / (M2 * M3);
        constexpr index_t M0 = MPerBlock / (M1 * M2 * M3);
        static_assert(M0 * M1 * M2 * M3 == MPerBlock, "M0, M1, M2, M3 must cover whole MPerBlock!");

        constexpr index_t Pad = 4 * K2;

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

        return a_lds_block_desc;
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeMX_BFlatBytesDramTileDistribution()
    {
        constexpr index_t K1          = WaveSize;
        constexpr index_t KWavePerBlk = 1;
        constexpr index_t K0          = KWavePerBlk;

        constexpr index_t WaveRepeat = WaveNum / TileShape::flatNPerWarp;

        if constexpr(std::is_same_v<BDataType, pk_fp4_t>)
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<WaveRepeat>,
                    tuple<sequence<NWarps, NXdlPack>, sequence<K0, K1, BK1 / BPackedSize>>,
                    tuple<sequence<0, 1, 2>, sequence<2>>,
                    tuple<sequence<0, 0, 0>, sequence<1>>,
                    sequence<2>,
                    sequence<2>>{});
        else if constexpr(std::is_same_v<BDataType, fp8_t>)
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<WaveRepeat>,
                    tuple<sequence<NWarps, NXdlPack>,
                          sequence<K_Thread / BK1, K0, K1, BK1 / BPackedSize>>,
                    tuple<sequence<0, 1, 2>, sequence<2>>,
                    tuple<sequence<0, 0, 1>, sequence<2>>,
                    sequence<2, 2>,
                    sequence<0, 3>>{});
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
        constexpr index_t MRepeat = MPerBlock / (MWarps * MPerXdl);
        static_assert(MWarps == 1,
                      "Current preshuffle ScaleA distribution assumes a single warp along M.");
        static_assert(MRepeat % MXdlPack == 0,
                      "ScaleA distribution requires MRepeat to be divisible by MXdlPack.");

        constexpr index_t M_Lanes = TileShape::WarpTile::at(I0);
        constexpr index_t K_Lanes = 64 / M_Lanes;

        constexpr index_t Y2 = M_Lanes;
        constexpr index_t Y1 = MWarps;
        constexpr index_t Y0 = MPerBlock / (MXdlPack * Y1 * Y2);

        constexpr index_t X0 = K_Lanes;
        constexpr index_t X1 = 1;

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<NWarps>,
                                       tuple<sequence<Y0, Y1, Y2>, sequence<X0, X1>>,
                                       tuple<sequence<1, 0>, sequence<2, 1>>,
                                       tuple<sequence<1, 0>, sequence<0, 2>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeMX_ScaleB_DramTileDistribution()
    {
        constexpr index_t NRepeat = NPerBlock / (NWarps * NPerXdl);
        static_assert(NRepeat % NXdlPack == 0,
                      "ScaleB distribution requires NRepeat to be divisible by NXdlPack.");

        constexpr index_t N_Lanes = TileShape::WarpTile::at(I1);
        constexpr index_t K_Lanes = 64 / N_Lanes;

        constexpr index_t Y2 = N_Lanes;
        constexpr index_t Y1 = NWarps;
        constexpr index_t Y0 = NPerBlock / (NXdlPack * Y1 * Y2);

        constexpr index_t X0 = K_Lanes;
        constexpr index_t X1 = 1;

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<MWarps>,
                                       tuple<sequence<Y0, Y1, Y2>, sequence<X0, X1>>,
                                       tuple<sequence<0, 1>, sequence<2, 1>>,
                                       tuple<sequence<0, 1>, sequence<0, 2>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    // Scale A follows the preshuffled-B path rather than the standard packed MX GEMM scale
    // path, so it uses the flat K view that matches the B-flat iteration order.
    CK_TILE_HOST_DEVICE static constexpr auto MakeMX_ScaleA_FlatDramTileDistribution()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<NWarps>,
                                       tuple<sequence<MWarps, MPerXdl>, sequence<K_Lane, 1>>,
                                       tuple<sequence<1, 0>, sequence<2, 1>>,
                                       tuple<sequence<0, 0>, sequence<0, 1>>,
                                       sequence<2>,
                                       sequence<1>>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeMX_ScaleB_FlatDramTileDistribution()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<MWarps>,
                                       tuple<sequence<NWarps, NPerXdl>, sequence<K_Lane, 1>>,
                                       tuple<sequence<0, 1>, sequence<2, 1>>,
                                       tuple<sequence<0, 0>, sequence<0, 1>>,
                                       sequence<2>,
                                       sequence<1>>{});
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeA()
    {
        return sizeof(ADataType) * MakeMX_ALdsBytesBlockDescriptor().get_element_space_size();
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return GetSmemSizeA(); }
};
} // namespace detail

struct MXGemmPipelineAgBgCrPolicy
{
#define FORWARD_METHOD_(method)                                                                  \
    template <typename Problem, typename... Args>                                                \
    CK_TILE_HOST_DEVICE static constexpr auto method(Args&&... args)                             \
    {                                                                                            \
        return detail::MXGemmPipelineAgBgCrPolicy<Problem>::method(std::forward<Args>(args)...); \
    }

    FORWARD_METHOD_(GetBlockGemm);
    FORWARD_METHOD_(MakeMX_AAsyncLoadBytesDramWindow);
    FORWARD_METHOD_(MakeMX_ABytesDramTileDistribution);
    FORWARD_METHOD_(MakeMX_ALdsBytesBlockDescriptor);
    FORWARD_METHOD_(MakeMX_BFlatBytesDramTileDistribution);
    FORWARD_METHOD_(MakeMX_BFlatBytesDramWindow);
    FORWARD_METHOD_(MakeMX_ScaleA_DramTileDistribution);
    FORWARD_METHOD_(MakeMX_ScaleB_DramTileDistribution);
    FORWARD_METHOD_(MakeMX_ScaleA_FlatDramTileDistribution);
    FORWARD_METHOD_(MakeMX_ScaleB_FlatDramTileDistribution);
    FORWARD_METHOD_(GetSmemSizeA);
    FORWARD_METHOD_(GetSmemSize);

#undef FORWARD_METHOD_

    // A is always RowMajor and B is preshuffled: no transpose-load needed.
    template <typename Problem>
    static constexpr bool is_a_load_tr = false;

    template <typename Problem>
    static constexpr bool is_b_load_tr = false;
};

} // namespace ck_tile
