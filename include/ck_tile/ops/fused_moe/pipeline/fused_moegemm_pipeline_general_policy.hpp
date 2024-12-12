// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fused_moe/pipeline/fused_moegemm_traits.hpp"
#include "ck_tile/ops/flatmm.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_problem.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_breg_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_breg_creg_v1.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v2.hpp"

namespace ck_tile {

struct FusedMoeGemmPipelineGeneralPolicy
{
    CK_TILE_HOST_DEVICE static constexpr index_t GetAsyncCopyDwords()
    {
        // TODO: always 1 dword
        return 1;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignment_A()
    {
        // using async
        constexpr index_t copy_bytes = 4 * GetAsyncCopyDwords();
        constexpr index_t data_bytes = sizeof(typename Problem::ADataType);
        static_assert(copy_bytes % data_bytes == 0);
        return copy_bytes / data_bytes;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignment_G()
    {
        constexpr index_t copy_bytes = [&]() { return 16; }();
        constexpr index_t data_bytes = sizeof(typename Problem::GDataType);
        static_assert(copy_bytes % data_bytes == 0);
        return copy_bytes / data_bytes;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignment_D()
    {
        constexpr index_t copy_bytes = [&]() { return 16; }();
        constexpr index_t data_bytes = sizeof(typename Problem::DDataType);
        static_assert(copy_bytes % data_bytes == 0);
        return copy_bytes / data_bytes;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignment_O()
    {
        if constexpr(Problem::Traits::OAtomic == 1)
        {
            // pack fp16/bf16 atomic
            static_assert(sizeof(typename Problem::ODataType) == 2);
            return 2;
        }
        else if constexpr(Problem::Traits::OAtomic == 2)
        {
            // fp32 atomic
            return 1;
        }
        else
        {
            return 16 / sizeof(typename Problem::ODataType);
        }
    }

    template <typename DataType_>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPack()
    {
        // TODO: this is for 3d layout
        return 16 / sizeof(remove_cvref_t<DataType_>);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPack_A()
    {
        return GetSmemKPack<typename Problem::ADataType>();
    }

    // used for bridge LDS shuffle
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPack_Y()
    {
        // TODO: this should match mfma layout
        return 16 / sizeof(typename Problem::YDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize_A()
    {
        constexpr auto a_lds_desc = MakeLdsBlockDesc_A<Problem>();
        return a_lds_desc.get_element_space_size();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize_Bridge()
    {
        constexpr auto bridge_lds_desc = MakeBridgeLdsBlockDesc<Problem>();
        return bridge_lds_desc.get_element_space_size();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        constexpr index_t a_lds      = GetSmemSize_A<Problem>();
        constexpr index_t bridge_lds = GetSmemSize_Bridge<Problem>();
        return max(a_lds, bridge_lds);
    }

    template <index_t MPerBlock, index_t KPerBlock, index_t NumWarps, index_t Alignment>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGlobalTileDistribution_SimpleMxK()
    {
        constexpr index_t K_vec = Alignment;
        constexpr index_t K_rem = KPerBlock / K_vec;

        if constexpr(get_warp_size() < K_rem)
        {
            static_assert(K_rem % get_warp_size() == 0);
            constexpr index_t K_lan = get_warp_size(); // lane within same wave is along gemm-k
            constexpr index_t K_wav = K_rem / get_warp_size();
            static_assert(K_wav <= NumWarps, "not not support thread has repeat along K yet");
            constexpr index_t M_wav = NumWarps / K_wav;
            static_assert(MPerBlock % M_wav == 0, "this tile size is too small please check");
            constexpr index_t M_rep = MPerBlock / M_wav;

            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<1>,
                    tuple<sequence<M_rep, M_wav>, sequence<K_wav, K_lan, K_vec>>,
                    tuple<sequence<1, 2>, sequence<2>>,
                    tuple<sequence<1, 0>, sequence<1>>,
                    sequence<1, 2>,
                    sequence<0, 2>>{});
        }
        else
        {
            constexpr index_t K_lan = K_rem;
            constexpr index_t M_lan = get_warp_size() / K_lan;
            constexpr index_t M_wav = NumWarps;
            static_assert(MPerBlock % (M_lan * M_wav) == 0,
                          "this tile size is too small please check");
            constexpr index_t M_rep = MPerBlock / (M_lan * M_wav);
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<1>,
                    tuple<sequence<M_rep, M_wav, M_lan>, sequence<K_lan, K_vec>>,
                    tuple<sequence<1>, sequence<1, 2>>,
                    tuple<sequence<1>, sequence<2, 0>>,
                    sequence<1, 2>,
                    sequence<0, 1>>{});
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGlobalTileDistribution_A()
    {
        constexpr index_t Block_M_   = Problem::BlockShape::Block_M0;
        constexpr index_t Block_K_   = Problem::BlockShape::Block_K0;
        constexpr index_t NumWarps_  = Problem::BlockShape::NumWarps;
        constexpr index_t Alignment_ = GetAlignment_A<Problem>();
        return MakeGlobalTileDistribution_SimpleMxK<Block_M_, Block_K_, NumWarps_, Alignment_>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGlobalTileDistribution_G()
    {
        using WG = decltype(GetWarpGemm0<Problem>());
        using S_ = typename Problem::BlockShape;
        static_assert(S_::WarpPerBlock_N0 == 4);
        constexpr auto g_outer_dstr_enc = tile_distribution_encoding<
            sequence<S_::WarpPerBlock_M0>,
            tuple<sequence<S_::Repeat_N0, S_::WarpPerBlock_N0>, sequence<S_::Repeat_K0>>,
            tuple<sequence<0, 1>>,
            tuple<sequence<0, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};
        constexpr auto g_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            g_outer_dstr_enc, typename WG::BWarpDstrEncoding{});

        return make_static_tile_distribution(g_block_dstr_encode);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGlobalTileDistribution_O()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<1, 2, 16>, sequence<4, 8>>,
                                       tuple<sequence<0, 1>, sequence<1, 2>>,
                                       tuple<sequence<0, 0>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<1, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm0()
    {
        using S_          = typename Problem::BlockShape;
        using GemmProblem = BlockGemmProblem<typename Problem::ADataType,
                                             typename Problem::GDataType,
                                             typename Problem::AccDataType,
                                             S_::BlockSize,
                                             TileGemmShape<typename S_::BlockTile_0,
                                                           typename S_::WarpPerBlock_0,
                                                           typename S_::WarpTile_0>>;

        constexpr auto warp_gemm = GetWarpGemm0<Problem>();
        using BlockGemmPolicy =
            BlockGemmASmemBSmemCRegV1CustomPolicy<typename Problem::ADataType,
                                                  // using BlockGemmPolicy =
                                                  // BlockGemmASmemBRegCRegV1CustomPolicy<typename
                                                  // Problem::ADataType,
                                                  typename Problem::GDataType,
                                                  typename Problem::AccDataType,
                                                  typename S_::WarpPerBlock_0,
                                                  decltype(warp_gemm)>;

        return BlockGemmASmemBSmemCRegV1<GemmProblem, BlockGemmPolicy>{};
        // return BlockGemmASmemBRegCRegV1<GemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm1()
    {
        using S_          = typename Problem::BlockShape;
        using GemmProblem = BlockGemmProblem<typename Problem::YDataType,
                                             typename Problem::DDataType,
                                             typename Problem::AccDataType,
                                             S_::BlockSize,
                                             TileGemmShape<typename S_::BlockTile_1,
                                                           typename S_::WarpPerBlock_1,
                                                           typename S_::WarpTile_1>>;

        constexpr auto warp_gemm = GetWarpGemm1<Problem>();
        using BlockGemmPolicy    = BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::ADataType,
                                                                    typename Problem::GDataType,
                                                                    typename Problem::AccDataType,
                                                                    typename S_::WarpPerBlock_1,
                                                                    decltype(warp_gemm)>;

        return BlockGemmARegBRegCRegV2<GemmProblem, BlockGemmPolicy>{};
    }

    // this is used as A matrix for 2nd gemm
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeYTileDistribution()
    {
        using S_       = remove_cvref_t<typename Problem::BlockShape>;
        using WarpGemm = remove_cvref_t<decltype(GetWarpGemm1<Problem>())>;

        constexpr auto y_outer_dstr_enc =
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<S_::Repeat_M1, S_::WarpPerBlock_M1>,
                                             sequence<S_::WarpPerBlock_K1, S_::Repeat_K1>>,
                                       tuple<sequence<1, 2>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{};

        constexpr auto y_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            y_outer_dstr_enc, typename WarpGemm::AWarpDstrEncoding{});
        constexpr auto y_block_dstr = make_static_tile_distribution(y_block_dstr_encode);
        return y_block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGlobalTileDistribution_D()
    {
        using S_       = remove_cvref_t<typename Problem::BlockShape>;
        using WarpGemm = remove_cvref_t<decltype(GetWarpGemm1<Problem>())>;

        constexpr auto d_outer_dstr_enc =
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<S_::Repeat_N1, S_::WarpPerBlock_N1>,
                                             sequence<S_::WarpPerBlock_K1, S_::Repeat_K1>>,
                                       tuple<sequence<1, 2>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{};

        constexpr auto d_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            d_outer_dstr_enc, typename WarpGemm::BWarpDstrEncoding{});
        constexpr auto d_block_dstr = make_static_tile_distribution(d_block_dstr_encode);
        return d_block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsBlockDesc_A()
    {
        constexpr index_t Block_M = Problem::BlockShape::Block_M0;
        constexpr index_t Block_K = Problem::BlockShape::Block_K0;
        constexpr index_t kK1     = GetSmemKPack_A<Problem>(); // LDS
        constexpr index_t kK0     = Block_K / kK1;

        static_assert(Block_K % kK1 == 0);

        constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kK0>{}, number<Block_M>{}, number<kK1>{}),
            make_tuple(number<Block_M * kK1>{}, number<kK1>{}, number<1>{}),
            number<8>{},
            number<1>{});

        constexpr auto a_lds_block_desc = transform_tensor_descriptor(
            a_lds_block_desc_0,
            make_tuple(make_pass_through_transform(number<Block_M>{}),
                       make_merge_transform(make_tuple(number<kK0>{}, number<kK1>{}))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        // constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
        //     make_tuple(number<Block_M>{}, number<Block_K>{}),
        //     make_tuple(number<Block_K>{},  number<1>{}),
        //     number<8>{},
        //     number<1>{});

        return a_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsBlockDesc_G()
    {
        constexpr index_t Block_N = Problem::BlockShape::Block_N0;
        constexpr index_t Block_K = Problem::BlockShape::Block_K0;
        constexpr index_t kK1     = GetSmemKPack_A<Problem>(); // LDS
        constexpr index_t kK0     = Block_K / kK1;

        static_assert(Block_K % kK1 == 0);

        constexpr auto d_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kK0>{}, number<Block_N>{}, number<kK1>{}),
            make_tuple(number<Block_N * kK1>{}, number<kK1>{}, number<1>{}),
            number<8>{},
            number<1>{});

        constexpr auto d_lds_block_desc = transform_tensor_descriptor(
            d_lds_block_desc_0,
            make_tuple(make_pass_through_transform(number<Block_N>{}),
                       make_merge_transform(make_tuple(number<kK0>{}, number<kK1>{}))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        // constexpr auto d_lds_block_desc = make_naive_tensor_descriptor(
        //     make_tuple(number<Block_N>{}, number<Block_K>{}),
        //     make_tuple(number<Block_K>{},  number<1>{}),
        //     number<8>{},
        //     number<1>{});

        return d_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBridgeLdsBlockDesc()
    {
        constexpr index_t Block_M = Problem::BlockShape::Block_M0;
        constexpr index_t Block_N = Problem::BlockShape::Block_N0;

        constexpr index_t KVector = GetSmemKPack_Y<Problem>();

        constexpr auto desc =
            make_naive_tensor_descriptor(make_tuple(number<Block_M>{}, number<Block_N>{}),
                                         make_tuple(number<Block_N>{}, number<1>{}),
                                         number<KVector>{},
                                         number<1>{});
        return desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemm0()
    {
        using S_ = typename Problem::BlockShape;
        // A is vgpr, B is agpr. But since we transposed, so also need swap this
        // TODO: this is ugly
        constexpr auto wg_ctrl = WGAttrCtlEnum::Raw_avv;
        // TODO: ugly
        if constexpr(std::is_same_v<typename Problem::ADataType, ck_tile::fp16_t> &&
                     std::is_same_v<typename Problem::GDataType, ck_tile::fp16_t> &&
                     S_::Warp_M0 == 32 && S_::Warp_N0 == 32 && S_::Warp_K0 == 8)
        {
            return WarpGemmImpl<WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution_SwizzleB<
                WarpGemmAttributeMfmaImplF16F16F32M32N32K8<wg_ctrl>,
                1>>{};
        }
        else if constexpr(std::is_same_v<typename Problem::ADataType, ck_tile::bf16_t> &&
                          std::is_same_v<typename Problem::GDataType, ck_tile::bf16_t> &&
                          S_::Warp_M0 == 32 && S_::Warp_N0 == 32 && S_::Warp_K0 == 8)
        {
            return WarpGemmImpl<WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution_SwizzleB<
                WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8<wg_ctrl>,
                1>>{};
        }
        else if constexpr(std::is_same_v<typename Problem::ADataType, ck_tile::bf16_t> &&
                          std::is_same_v<typename Problem::GDataType, ck_tile::bf16_t> &&
                          S_::Warp_M0 == 32 && S_::Warp_N0 == 32 && S_::Warp_K0 == 16)
        {
            return WarpGemmImpl<WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution_SwizzleB<
                WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8<wg_ctrl>,
                2>>{};
        }
        else if constexpr(std::is_same_v<typename Problem::ADataType, ck_tile::int8_t> &&
                          std::is_same_v<typename Problem::GDataType, ck_tile::int8_t> &&
                          S_::Warp_M0 == 32 && S_::Warp_N0 == 32 && S_::Warp_K0 == 32)
        {
            return WarpGemmImpl<WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution_SwizzleB<
                WarpGemmAttributeMfmaImpl_i32_32x32x16_i8<wg_ctrl>,
                2>>{};
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemm1()
    {
        using S_               = typename Problem::BlockShape;
        constexpr auto wg_ctrl = WGAttrCtlEnum::Raw_avv;
        // TODO: ugly
        if constexpr(std::is_same_v<typename Problem::YDataType, ck_tile::fp16_t> &&
                     std::is_same_v<typename Problem::DDataType, ck_tile::fp16_t> &&
                     S_::Warp_M0 == 32 && S_::Warp_N0 == 32 && S_::Warp_K0 == 8)
        {
            return WarpGemmImpl<WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution_SwizzleB<
                WarpGemmAttributeMfmaImplF16F16F32M32N32K8<wg_ctrl>,
                1>>{};
        }
        else if constexpr(std::is_same_v<typename Problem::YDataType, ck_tile::bf16_t> &&
                          std::is_same_v<typename Problem::DDataType, ck_tile::bf16_t> &&
                          S_::Warp_M0 == 32 && S_::Warp_N0 == 32 && S_::Warp_K0 == 8)
        {
            return WarpGemmImpl<WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution_SwizzleB<
                WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8<wg_ctrl>,
                1>>{};
        }
        else if constexpr(std::is_same_v<typename Problem::YDataType, ck_tile::bf16_t> &&
                          std::is_same_v<typename Problem::DDataType, ck_tile::bf16_t> &&
                          S_::Warp_M0 == 32 && S_::Warp_N0 == 32 && S_::Warp_K0 == 16)
        {
            return WarpGemmImpl<WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution_SwizzleB<
                WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8<wg_ctrl>,
                2>>{};
        }
        else if constexpr(std::is_same_v<typename Problem::YDataType, ck_tile::int8_t> &&
                          std::is_same_v<typename Problem::DDataType, ck_tile::int8_t> &&
                          S_::Warp_M0 == 32 && S_::Warp_N0 == 32 && S_::Warp_K0 == 32)
        {
            return WarpGemmImpl<WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution_SwizzleB<
                WarpGemmAttributeMfmaImpl_i32_32x32x16_i8<wg_ctrl>,
                2>>{};
        }
    }
};
} // namespace ck_tile
