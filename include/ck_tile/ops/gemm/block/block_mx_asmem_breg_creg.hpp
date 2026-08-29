// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename Problem_,
          typename BlockPolicy_,
          index_t MXdlPack_,
          index_t NXdlPack_,
          index_t KXdlPack_>
struct BlockMXGemmASmemBRegCReg
{
    using Problem            = remove_cvref_t<Problem_>;
    using BlockPolicy        = remove_cvref_t<BlockPolicy_>;
    using ADataType          = remove_cvref_t<typename Problem::ADataType>;
    using CDataType          = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape     = remove_cvref_t<typename Problem::BlockGemmShape>;
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    static constexpr auto config = BlockPolicy::template GetWarpGemmMWarpNWarp<Problem>();
    using WarpGemm               = remove_cvref_t<decltype(config.template at<0>())>;

    static constexpr index_t MWarp = config.template at<1>();
    static constexpr index_t NWarp = config.template at<2>();

    static constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WarpGemm::kM);
    static constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WarpGemm::kN);
    static constexpr index_t KIterPerWarp = KPerBlock / WarpGemm::kK;

    static constexpr index_t MXdlPack      = MXdlPack_;
    static constexpr index_t NXdlPack      = NXdlPack_;
    static constexpr index_t KXdlPack      = KXdlPack_;
    static constexpr index_t APackedSize   = numeric_traits<ADataType>::PackedSize;
    static constexpr index_t DsReadPreload = 4;
    static constexpr index_t m_preload     = (MIterPerWarp * KIterPerWarp >= DsReadPreload)
                                                 ? DsReadPreload
                                                 : MIterPerWarp * KIterPerWarp;

    static constexpr index_t MPackIterPerWarp = MIterPerWarp / MXdlPack;
    static constexpr index_t NPackIterPerWarp = NIterPerWarp / NXdlPack;
    static constexpr index_t KPackIterPerWarp = KIterPerWarp / KXdlPack;

    using AWarpTensor = typename WarpGemm::AWarpTensor;
    statically_indexed_array<AWarpTensor, m_preload> preloaded_a_warp_tensor;

    CK_TILE_HOST_DEVICE static constexpr auto MakeABlockTileDistribution()
    {
        constexpr index_t K_Lane   = get_warp_size() / 16;
        constexpr index_t K_Thread = WarpGemm::kK / K_Lane;
        constexpr index_t AK1      = 16 * APackedSize;

        static_assert(BlockGemmShape::WarpTile::at(I0) == 16 &&
                      BlockGemmShape::WarpTile::at(I1) == 16);
        static_assert(BlockGemmShape::BlockWarps::at(I0) == 1, "requires Wave_M == 1");

        if constexpr(std::is_same_v<ADataType, pk_fp4_t>)
            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<NWarp>,
                                           tuple<sequence<MWarp, MXdlPack, WarpGemm::kM>,
                                                 sequence<K_Lane, AK1 / APackedSize>>,
                                           tuple<sequence<1, 0>, sequence<2, 1>>,
                                           tuple<sequence<0, 0>, sequence<0, 2>>,
                                           sequence<2>,
                                           sequence<1>>{});
        else if constexpr(std::is_same_v<ADataType, fp8_t>)
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<NWarp>,
                    tuple<sequence<MWarp, MXdlPack, WarpGemm::kM>,
                          sequence<K_Thread / AK1, K_Lane, AK1 / APackedSize>>,
                    tuple<sequence<1, 0>, sequence<2, 1>>,
                    tuple<sequence<0, 0>, sequence<1, 2>>,
                    sequence<2, 2>,
                    sequence<0, 2>>{});
        else
            static_assert(false, "unsupported datatype");
    }

    template <typename AWarpWindow>
    CK_TILE_DEVICE auto MakeALoadWindows(const AWarpWindow& a_warp_window) const
    {
        return generate_tuple(
            [&](auto kIter) {
                return generate_tuple(
                    [&](auto mIter) {
                        return make_tile_window(
                            a_warp_window.get_bottom_tensor_view(),
                            a_warp_window.get_window_lengths(),
                            {mIter * WarpGemm::kM,
                             kIter * WarpGemm::kK * sizeof(ADataType) / APackedSize},
                            a_warp_window.get_tile_distribution());
                    },
                    number<MXdlPack>{});
            },
            number<m_preload / MXdlPack>{});
    }

    template <typename ALoadWindows>
    CK_TILE_DEVICE void LocalPrefetch(const ALoadWindows& a_load_windows)
    {
        static_for<0, m_preload, 1>{}([&](auto loadIter) {
            constexpr auto mIter = loadIter % MXdlPack;
            constexpr auto kIter = loadIter / MXdlPack;

            load_tile(preloaded_a_warp_tensor(loadIter),
                      a_load_windows[number<kIter>{}][number<mIter>{}]);
        });
    }

    CK_TILE_DEVICE static constexpr auto MakeCBlockTile()
    {
        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WarpGemm::CWarpDstrEncoding{});

        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);

        auto c_block_tensor = make_static_distributed_tensor<CDataType>(c_block_dstr);
        return c_block_tensor;
    }

    template <typename CWarpTensors,
              typename BWarpTensors,
              typename ScaleATileTensors,
              typename ScaleBTileTensors,
              typename AWarpWindow>
    CK_TILE_DEVICE void operator()(CWarpTensors& c_warp_tensors,
                                   const BWarpTensors& b_warp_tensors,
                                   const ScaleATileTensors& scale_a_tile_tensors,
                                   const ScaleBTileTensors& scale_b_tile_tensors,
                                   const AWarpWindow& a_warp_window)
    {
        static_for_product<number<KPackIterPerWarp>,
                           number<MPackIterPerWarp>,
                           number<NPackIterPerWarp>,
                           number<KXdlPack>,
                           number<MXdlPack>,
                           number<NXdlPack>>{}([&](auto ikpack,
                                                   auto impack,
                                                   auto inpack,
                                                   auto ikxdl,
                                                   auto imxdl,
                                                   auto inxdl) {
            constexpr auto m_iter    = impack * MXdlPack + imxdl;
            constexpr auto n_iter    = inpack * NXdlPack + inxdl;
            constexpr auto k_iter    = ikpack * KXdlPack + ikxdl;
            constexpr auto APackIter = ikxdl * MXdlPack + imxdl;

            WarpGemm{}.template operator()<OpSelA<APackIter>, OpSelB<ikxdl * NXdlPack + inxdl>>(
                c_warp_tensors(number<m_iter>{})(number<n_iter>{}),
                preloaded_a_warp_tensor(number<APackIter>{}),
                bit_cast<typename WarpGemm::BWarpTensor>(
                    b_warp_tensors(number<n_iter>{})(number<k_iter>{})),
                scale_a_tile_tensors(impack)(ikpack).get_thread_buffer()[0],
                scale_b_tile_tensors(inpack)(ikpack).get_thread_buffer()[0]);

            constexpr auto addr = m_iter % 2 + k_iter * 2 + m_iter / 2 * 4 + m_preload;
            if constexpr(addr < (KIterPerWarp * MIterPerWarp) && (n_iter == NIterPerWarp - 1))
            {
                constexpr auto AmIter = addr % 2 + addr / 4 * 2;
                constexpr auto AkIter = addr / 2 % 2;
                preloaded_a_warp_tensor(number<APackIter>{}) =
                    bit_cast<AWarpTensor>(load_tile_with_offset(
                        a_warp_window,
                        tuple<number<AmIter * WarpGemm::kM>,
                              number<sizeof(ADataType) * AkIter * WarpGemm::kK / APackedSize>>{}));
            }
        });
    }
};

} // namespace ck_tile
