// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_default_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"

namespace ck_tile {

// A is block window on shared memory
// B is block window on shared memory
// C is block distributed tensor
template <typename Problem_, typename Policy_ = BlockGemmASmemBSmemCRegV1DefaultPolicy>
struct BlockUniversalGemmAsBsCr
{
    private:
    // TODO: This should be in Policy - UniversalGemmPolicyBase ?
    template <typename PipelineProblem_, typename GemmPolicy_>
    struct GemmTraits_
    {
        using Problem        = remove_cvref_t<PipelineProblem_>;
        using Policy         = remove_cvref_t<GemmPolicy_>;
        using ADataType      = remove_cvref_t<typename Problem::ADataType>;
        using BDataType      = remove_cvref_t<typename Problem::BDataType>;
        using CDataType      = remove_cvref_t<typename Problem::CDataType>;
        using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

        static constexpr index_t kBlockSize = Problem::kBlockSize;
        static constexpr auto Scheduler     = Problem::Scheduler;

        static constexpr index_t MPerBlock = BlockGemmShape::kM;
        static constexpr index_t NPerBlock = BlockGemmShape::kN;
        static constexpr index_t KPerBlock = BlockGemmShape::kK;

        static constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WarpGemm = remove_cvref_t<decltype(config.template at<0>())>;

        static constexpr index_t MWarp = config.template at<1>();
        static constexpr index_t NWarp = config.template at<2>();

        static constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WarpGemm::kM);
        static constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WarpGemm::kN);
        static constexpr index_t KIterPerWarp = KPerBlock / WarpGemm::kK;

        static constexpr index_t MPerBlockPerIter = MPerBlock / MIterPerWarp;
        static constexpr index_t NPerBlockPerIter = NPerBlock / NIterPerWarp;
        static constexpr index_t KPerBlockPerIter = KPerBlock / KIterPerWarp;

        using AWarpTileDistr = remove_cvref_t<decltype(make_static_tile_distribution(
            typename WarpGemm::AWarpDstrEncoding{}))>;
        using BWarpTileDistr = remove_cvref_t<decltype(make_static_tile_distribution(
            typename WarpGemm::BWarpDstrEncoding{}))>;

        using AWarpTile =
            remove_cvref_t<decltype(make_static_distributed_tensor<ADataType>(AWarpTileDistr{}))>;
        using BWarpTile =
            remove_cvref_t<decltype(make_static_distributed_tensor<BDataType>(BWarpTileDistr{}))>;
    };

    public:
    using Traits = GemmTraits_<Problem_, Policy_>;

    using ADataType = remove_cvref_t<typename Traits::ADataType>;
    using BDataType = remove_cvref_t<typename Traits::BDataType>;
    using CDataType = remove_cvref_t<typename Traits::CDataType>;

    using AWarpTile = remove_cvref_t<typename Traits::AWarpTile>;
    using BWarpTile = remove_cvref_t<typename Traits::BWarpTile>;
    using WarpGemm  = remove_cvref_t<typename Traits::WarpGemm>;

    static constexpr index_t KIterPerWarp = Traits::KIterPerWarp;
    static constexpr index_t MIterPerWarp = Traits::MIterPerWarp;
    static constexpr index_t NIterPerWarp = Traits::NIterPerWarp;

    static constexpr index_t MPerBlock = Traits::MPerBlock;
    static constexpr index_t NPerBlock = Traits::NPerBlock;
    static constexpr index_t KPerBlock = Traits::KPerBlock;

    static constexpr index_t MPerBlockPerIter = Traits::MPerBlockPerIter;
    static constexpr index_t NPerBlockPerIter = Traits::NPerBlockPerIter;
    static constexpr index_t KPerBlockPerIter = Traits::KPerBlockPerIter;

    static constexpr index_t MWarp = Traits::MWarp;
    static constexpr index_t NWarp = Traits::NWarp;

    static constexpr auto Scheduler = Traits::Scheduler;

    statically_indexed_array<statically_indexed_array<AWarpTile, KIterPerWarp>, MIterPerWarp>
        a_warp_tiles_;

    statically_indexed_array<statically_indexed_array<BWarpTile, KIterPerWarp>, NIterPerWarp>
        b_warp_tiles_;

    template <typename ASmemBlockWindow, typename BSmemBlockWindow>
    CK_TILE_DEVICE void LocalPrefetch(const ASmemBlockWindow& a_block_window,
                                      const BSmemBlockWindow& b_block_window)
    {
        static_assert(MPerBlock == ASmemBlockWindow{}.get_window_lengths()[number<0>{}] &&
                          NPerBlock == BSmemBlockWindow{}.get_window_lengths()[number<0>{}] &&
                          KPerBlock == ASmemBlockWindow{}.get_window_lengths()[number<1>{}],
                      "BlockUniversalGemmAsBsCr: MPerBlock, NPerBlock, KPerBlock defined in "
                      " BlockGemmShape are different from A/B block smem windows apropriate dims!");

        static_assert(std::is_same_v<ADataType, typename ASmemBlockWindow::DataType> &&
                          std::is_same_v<BDataType, typename BSmemBlockWindow::DataType>,
                      "wrong!");

        const index_t iMWarp = get_warp_id() / NWarp;
        const index_t iNWarp = get_warp_id() % NWarp;

        // construct A-warp-window
        // TODO: Can I construct this tile window and set later it's origin ???? move tile window
        // ???
        // TODO: create AWarpWindow type - compile tile
        //       will need ASmemBlockWindow_ - which is also compile time!! MPerBlock x KPerBlock
        // Would need to pass A/BSmem tensor view with nullptr, to constructor!
        //
        // This order of loads is fixed, since Smem descriptors are fixed!
        // We should have all below warp window available at compile-time!
        // --> static load coordinates
        // Pass it through BlockGemmASmemBSmemCRegV1CustomPolicy ??

        auto a_warp_window_tmp = make_tile_window(
            a_block_window.get_bottom_tensor_view(),
            make_tuple(number<WarpGemm::kM>{}, number<WarpGemm::kK>{}),
            a_block_window.get_window_origin() + multi_index<2>{iMWarp * WarpGemm::kM, 0},
            make_static_tile_distribution(typename WarpGemm::AWarpDstrEncoding{}));

        statically_indexed_array<
            statically_indexed_array<decltype(a_warp_window_tmp), KIterPerWarp>,
            MIterPerWarp>
            a_warp_windows;

        // construct B-warp-window
        auto b_warp_window_tmp = make_tile_window(
            b_block_window.get_bottom_tensor_view(),
            make_tuple(number<WarpGemm::kN>{}, number<WarpGemm::kK>{}),
            b_block_window.get_window_origin() + multi_index<2>{iNWarp * WarpGemm::kN, 0},
            make_static_tile_distribution(typename WarpGemm::BWarpDstrEncoding{}));

        statically_indexed_array<
            statically_indexed_array<decltype(b_warp_window_tmp), KIterPerWarp>,
            NIterPerWarp>
            b_warp_windows;

        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                a_warp_windows(mIter)(kIter) = a_warp_window_tmp;

                move_tile_window(a_warp_windows(mIter)(kIter),
                                 {mIter * MPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });

        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                b_warp_windows(nIter)(kIter) = b_warp_window_tmp;

                move_tile_window(b_warp_windows(nIter)(kIter),
                                 {nIter * NPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });

        static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                // read A warp tensor from A block window
                load_tile(a_warp_tiles_(mIter)(kIter), a_warp_windows(mIter)(kIter));

                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    // read B warp tensor from B Block window
                    load_tile(b_warp_tiles_(nIter)(kIter), b_warp_windows(nIter)(kIter));
                });
            });
        });
    }

    private:
    template <GemmPipelineScheduler Scheduler, typename GemmTraits>
    struct BlockGemmImpl
    {
    };

    template <typename GemmTraits>
    struct BlockGemmImpl<GemmPipelineScheduler::Intrawave, GemmTraits>
    {

        // C += A * B
        template <typename CBlockTensor, typename AWarpTiles, typename BWarpTiles>
        CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                       const AWarpTiles& a_warp_tiles,
                                       const BWarpTiles& b_warp_tiles) const
        {
            static_assert(
                std::is_same_v<typename GemmTraits::CDataType, typename CBlockTensor::DataType>,
                "wrong!");

            using CWarpDstr   = typename GemmTraits::WarpGemm::CWarpDstr;
            using CWarpTensor = typename GemmTraits::WarpGemm::CWarpTensor;

            constexpr auto c_warp_y_lengths =
                to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
            constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

            // hot loop:
            static_for<0, GemmTraits::KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, GemmTraits::MIterPerWarp, 1>{}([&](auto mIter) {
                    static_for<0, GemmTraits::NIterPerWarp, 1>{}([&](auto nIter) {
                        // read C warp tensor from C block tensor-

                        // TODO: Universal GEMM allocates whole c_thread_buff
                        // StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                        //                         AccDataType,
                        //                         MRepeat * NRepeat,
                        //                         xdlops_gemm.GetRegSizePerXdlops(),
                        //                         true>
                        //     c_thread_buf_;
                        CWarpTensor c_warp_tensor;

                        c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                        // warp GEMM
                        typename GemmTraits::WarpGemm{}(
                            c_warp_tensor, a_warp_tiles[mIter][kIter], b_warp_tiles[nIter][kIter]);

                        // write C warp tensor into C block tensor
                        c_block_tensor.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());
                    });
                });
            });
        }
    };

    template <typename GemmTraits>
    struct BlockGemmImpl<GemmPipelineScheduler::Interwave, GemmTraits>
    {
    };

    public:
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
        auto c_block_tensor         = make_static_distributed_tensor<CDataType>(c_block_dstr);

        return c_block_tensor;
    }

    // C += A * B
    template <typename CBlockTensor, typename ASmemBlockWindow, typename BSmemBlockWindow>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   [[maybe_unused]] const ASmemBlockWindow& a_block_window,
                                   [[maybe_unused]] const BSmemBlockWindow& b_block_window) const
    {
        BlockGemmImpl<Scheduler, Traits>{}.template operator()(
            c_block_tensor, a_warp_tiles_, b_warp_tiles_);
    }

    // C = A * B
    template <typename ASmemBlockWindow, typename BSmemBlockWindow>
    CK_TILE_DEVICE auto operator()([[maybe_unused]] const ASmemBlockWindow& a_block_window,
                                   [[maybe_unused]] const BSmemBlockWindow& b_block_window) const
    {
        auto c_block_tensor = MakeCBlockTile();
        BlockGemmImpl<Scheduler, Traits>{}.template operator()(
            c_block_tensor, a_warp_tiles_, b_warp_tiles_);
        return c_block_tensor;
    }
};

} // namespace ck_tile
