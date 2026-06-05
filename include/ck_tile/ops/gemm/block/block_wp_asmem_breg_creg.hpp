// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/block/block_wp_asmem_bsmem_creg_v1_custom_policy.hpp"

namespace ck_tile {

// A is block window on shared memory
// B is block window on register
// C is block distributed tensor
template <typename Problem_, typename BlockPolicy_>
struct BlockWeightPreshuffleASmemBRegCReg
{
    using Problem        = remove_cvref_t<Problem_>;
    using BlockPolicy    = remove_cvref_t<BlockPolicy_>;
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    static constexpr auto I0   = number<0>();
    static constexpr auto I1   = number<1>();
    static constexpr auto I2   = number<2>();
    static constexpr auto idxM = I0;
    static constexpr auto idxN = I1;
    static constexpr auto idxK = I2;
    using BlockTile            = remove_cvref_t<typename BlockGemmShape::BlockTile>;
    using BlockWarps           = remove_cvref_t<typename BlockGemmShape::BlockWarps>;
    using WarpTile             = remove_cvref_t<typename BlockGemmShape::WarpTile>;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr auto config = BlockPolicy::template GetWarpGemmMWarpNWarp<Problem>();
    using WarpGemm               = remove_cvref_t<decltype(config.template at<0>())>;

    static constexpr index_t MWarp = config.template at<1>();
    static constexpr index_t NWarp = config.template at<2>();

    static constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WarpGemm::kM);
    static constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WarpGemm::kN);
    static constexpr index_t KIterPerWarp = KPerBlock / WarpGemm::kK;

    static constexpr index_t MPerBlockPerIter = MWarp * WarpGemm::kM;
    static constexpr index_t KPerBlockPerIter = WarpGemm::kK;

    static constexpr index_t DsReadPreload = 2; // default 2, preload 2 ds read

    static constexpr index_t m_preload = (MIterPerWarp * KIterPerWarp >= DsReadPreload)
                                             ? DsReadPreload
                                             : MIterPerWarp * KIterPerWarp;

    static constexpr index_t WarpTileK      = WarpTile::at(I2);
    static constexpr index_t K1             = Problem::VectorLoadSize / sizeof(ADataType);
    static constexpr index_t WaveSize       = get_warp_size();
    static constexpr index_t Bload_num_perK = NPerBlock * WarpTileK / NWarp / K1 / WaveSize;
    // Total number of buffer_load instructions for B (controls s_waitcnt vmcnt for async case)
    static constexpr index_t Bload_total_num = Bload_num_perK * KIterPerWarp;

    using AWarpTensor = typename WarpGemm::AWarpTensor;
    statically_indexed_array<AWarpTensor, m_preload> preloaded_a_warp_tensor;

    CK_TILE_DEVICE static constexpr auto MakeABlockDistributionEncode()
    {
        constexpr auto a_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<1, MWarp>, sequence<1>>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};
#ifdef __gfx125__
        constexpr index_t MPerXdl = WarpTile::at(I0);
        constexpr index_t KPerXdl = WarpTile::at(I2);

        constexpr index_t MWarpBlock = MPerXdl / 16;

        constexpr index_t KLane      = get_warp_size() / 16;
        constexpr index_t KPerThread = KPerXdl / KLane;

        constexpr index_t PackedSize    = numeric_traits<typename Problem::ADataType>::PackedSize;
        constexpr index_t MaxVecSize    = 16 / sizeof(ADataType) * PackedSize;
        constexpr index_t KItemsPerLoad = min(MaxVecSize, KPerThread);
        constexpr index_t KFragment     = KPerThread / KItemsPerLoad;
        constexpr auto a_block_inner_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MWarpBlock, 16>, sequence<KFragment, KLane, KItemsPerLoad>>,
            tuple<sequence<2, 1>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2, 2>,
            sequence<0, 0, 2>>{};

        constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encoding, a_block_inner_dstr_encoding);
        return a_block_dstr_encode;
#else
        constexpr auto a_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            a_block_outer_dstr_encoding, typename WarpGemm::AWarpDstrEncoding{});
        return a_block_dstr_encode;
#endif
    }

    template <typename SmemBlockWindow>
    CK_TILE_DEVICE auto MakeALoadWindows(SmemBlockWindow& a_block_window) const
    {
        constexpr auto a_load_dstr = make_static_tile_distribution(MakeABlockDistributionEncode());

        // create MIterPerWarp x KIterPerWarp window
        return generate_tuple(
            [&](auto kIter) {
                return generate_tuple(
                    [&](auto mIter) {
                        return make_tile_window(
                            get_slice_tile(
                                a_block_window,
                                sequence<mIter * MPerBlockPerIter, kIter * KPerBlockPerIter>{},
                                sequence<(mIter + 1) * MPerBlockPerIter,
                                         (kIter + 1) * KPerBlockPerIter>{}),
                            a_load_dstr);
                    },
                    number<MIterPerWarp>{});
            },
            number<KIterPerWarp>{});
    }

    template <typename ALoadWindows>
    CK_TILE_DEVICE void LocalPrefetch(const ALoadWindows& a_load_windows)
    {

        static_for<0, m_preload, 1>{}([&](auto loadIter) {
            constexpr auto mIter = loadIter % MIterPerWarp;
            constexpr auto kIter = loadIter / MIterPerWarp;

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

    // C += A * B
    template <typename CBlockTensor,
              typename ALoadWindows,
              typename BFlatBlockTensor,
              typename BFlatDistribution>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const ALoadWindows& a_load_windows,
                                   BFlatBlockTensor& b_block_tensor,
                                   const BFlatDistribution&)
    {
        constexpr auto MIter_2nd_last = (MIterPerWarp >= 2) ? MIterPerWarp - 2 : MIterPerWarp - 1;

        using CWarpDstr   = typename WarpGemm::CWarpDstr;
        using CWarpTensor = typename WarpGemm::CWarpTensor;

        using BWarpTensor = typename WarpGemm::BWarpTensor;

        constexpr auto b_block_y_lengths =
            to_sequence(BFlatDistribution{}.get_ys_to_d_descriptor().get_lengths());

        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

        constexpr auto b_block_y_index_zeros =
            uniform_sequence_gen_t<BFlatDistribution::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        static_ford<sequence<KIterPerWarp, MIterPerWarp>>{}([&](auto km) {
            constexpr auto kIter     = number<km[number<0>{}]>{};
            constexpr auto mIter     = number<km[number<1>{}]>{};
            constexpr auto AwarpIter = (kIter * MIterPerWarp + mIter) % m_preload;
            static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                // read C warp tensor from C block tensor
                BWarpTensor b_warp_tensor;
                CWarpTensor c_warp_tensor;

                b_warp_tensor.get_thread_buffer() = b_block_tensor.get_y_sliced_thread_data(
                    merge_sequences(
                        sequence<nIter, kIter>{},
                        typename sequence_split<decltype(b_block_y_index_zeros), 2>::right_type{}),
                    merge_sequences(
                        sequence<1, 1>{},
                        typename sequence_split<decltype(b_block_y_lengths), 2>::right_type{}));

                c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                    merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                // warp GEMM
                WarpGemm{}(
                    c_warp_tensor, preloaded_a_warp_tensor(number<AwarpIter>{}), b_warp_tensor);

                // write C warp tensor into C block tensor
                c_block_tensor.set_y_sliced_thread_data(
                    merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                    c_warp_tensor.get_thread_buffer());
            });
            // preload next A from lds
            if constexpr((kIter * MIterPerWarp + mIter) < (KIterPerWarp * MIterPerWarp - m_preload))
            {
                constexpr auto AmIter = (mIter + m_preload) % MIterPerWarp;
                constexpr auto AkIter = (kIter + (mIter + m_preload) / MIterPerWarp);

                load_tile(preloaded_a_warp_tensor(number<AwarpIter>{}),
                          a_load_windows[number<AkIter>{}][number<AmIter>{}]);
            }

            // barrier
            if constexpr((kIter == KIterPerWarp - 1) && (mIter == MIter_2nd_last))
            {
                if constexpr(Problem::Async)
                {
                    // vmcnt is equal to buffer_load instructions for B
                    // it means that we wait only for buffer load instructions for A
                    // because those are the one using async load to lds
                    s_waitcnt<Bload_total_num>();
                }
                block_sync_lds();
            }
        });
    }

    // C += A * B with scale value
    template <typename CBlockTensor,
              typename ALoadWindows,
              typename BFlatBlockTensor,
              typename BFlatDistribution,
              typename AScaleBlockTensor,
              typename BScaleBlockTensor>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const ALoadWindows& a_load_windows,
                                   BFlatBlockTensor& b_block_tensor,
                                   const BFlatDistribution&,
                                   AScaleBlockTensor& a_scale_tensor,
                                   BScaleBlockTensor& b_scale_tensor)
    {
        constexpr auto MIter_2nd_last = (MIterPerWarp >= 2) ? MIterPerWarp - 2 : MIterPerWarp - 1;

        using CWarpDstr   = typename WarpGemm::CWarpDstr;
        using CWarpTensor = typename WarpGemm::CWarpTensor;

        using BWarpTensor = typename WarpGemm::BWarpTensor;

        constexpr auto b_block_y_lengths =
            to_sequence(BFlatDistribution{}.get_ys_to_d_descriptor().get_lengths());

        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());

        constexpr auto b_block_y_index_zeros =
            uniform_sequence_gen_t<BFlatDistribution::NDimY, 0>{};
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        static_ford<sequence<KIterPerWarp, MIterPerWarp>>{}([&](auto km) {
            constexpr auto kIter     = number<km[number<0>{}]>{};
            constexpr auto mIter     = number<km[number<1>{}]>{};
            constexpr auto AwarpIter = (kIter * MIterPerWarp + mIter) % m_preload;

            index_t scale_a = a_scale_tensor.get_y_sliced_thread_data(sequence<mIter, kIter, 0>{},
                                                                      sequence<1, 1, 1>{})[0];
            static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                // read C warp tensor from C block tensor
                BWarpTensor b_warp_tensor;
                CWarpTensor c_warp_tensor;

                index_t scale_b = b_scale_tensor.get_y_sliced_thread_data(
                    sequence<nIter, kIter, 0>{}, sequence<1, 1, 1>{})[0];

                b_warp_tensor.get_thread_buffer() = b_block_tensor.get_y_sliced_thread_data(
                    merge_sequences(
                        sequence<nIter, kIter>{},
                        typename sequence_split<decltype(b_block_y_index_zeros), 2>::right_type{}),
                    merge_sequences(
                        sequence<1, 1>{},
                        typename sequence_split<decltype(b_block_y_lengths), 2>::right_type{}));

                c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                    merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                // warp GEMM
                WarpGemm{}(c_warp_tensor,
                           preloaded_a_warp_tensor(number<AwarpIter>{}),
                           b_warp_tensor,
                           scale_a,
                           scale_b);

                // write C warp tensor into C block tensor
                c_block_tensor.set_y_sliced_thread_data(
                    merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                    c_warp_tensor.get_thread_buffer());
            });
            // preload next A from lds
            if constexpr((kIter * MIterPerWarp + mIter) < (KIterPerWarp * MIterPerWarp - m_preload))
            {
                constexpr auto AmIter = (mIter + m_preload) % MIterPerWarp;
                constexpr auto AkIter = (kIter + (mIter + m_preload) / MIterPerWarp);

                load_tile(preloaded_a_warp_tensor(number<AwarpIter>{}),
                          a_load_windows[number<AkIter>{}][number<AmIter>{}]);
            }

            // barrier
            if constexpr((kIter == KIterPerWarp - 1) && (mIter == MIter_2nd_last))
            {
                block_sync_lds();
            }
            // preload next A from lds
            if constexpr((kIter * MIterPerWarp + mIter) < (KIterPerWarp * MIterPerWarp - m_preload))
            {
                constexpr auto AmIter = (mIter + m_preload) % MIterPerWarp;
                constexpr auto AkIter = (kIter + (mIter + m_preload) / MIterPerWarp);

                load_tile(preloaded_a_warp_tensor(number<AwarpIter>{}),
                          a_load_windows[number<AkIter>{}][number<AmIter>{}]);
            }

            // barrier
            if constexpr((kIter == KIterPerWarp - 1) && (mIter == MIter_2nd_last))
            {
                block_sync_lds();
            }
        });
    }
};

} // namespace ck_tile
