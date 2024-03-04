// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/store_tile.hpp"
#include "ck/tile_program/warp_tile/warp_gemm.hpp"
#include "ck/utility/philox_rand.hpp"

namespace ck {
namespace tile_program {
namespace block {

struct BlockDropout
{
    __host__ __device__ BlockDropout(const long_index_t global_idx_,
                                     const float p_dropout_rescale_,
                                     const uint8_t p_undrop_in_uint8_t_,
                                     const index_t total_n_len_,
                                     const unsigned long long seed,
                                     const unsigned long long offset,
                                     const bool is_store_randval_)
        : global_idx(global_idx_),
          p_dropout_rescale(p_dropout_rescale_),
          p_undrop_in_uint8_t(p_undrop_in_uint8_t_),
          total_n_len(total_n_len_),
          ph(seed, 0, offset),
          is_store_randval(is_store_randval_)
    {
    }

    template <typename BlockGemm, typename RandValDramBlockWindowTmp>
    __host__ __device__ static constexpr auto
    MakeRandvalDramWindow(RandValDramBlockWindowTmp& randval_dram_block_window_tmp,
                          index_t seqlen_k_start)
    {
        using Problem        = remove_cvref_t<typename BlockGemm::Problem>;
        constexpr index_t kM = Problem::BlockGemmShape::kM;

        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template At<0>())>;

        const auto block_origin  = randval_dram_block_window_tmp.GetWindowOrigin();
        auto randval_dram_window = make_tile_window(
            randval_dram_block_window_tmp.GetBottomTensorView(),
            make_tuple(Number<kM>{}, Number<WG::kN>{}),
            {block_origin.At(Number<0>{}), seqlen_k_start}, // M/N
            BlockDropout::template MakeRandValSramPartTileDistribution<BlockGemm>());

        return randval_dram_window;
    }

    template <typename BlockGemm>
    __host__ __device__ static constexpr auto MakeRandValLdsBlockDescriptor()
    {
        using Problem                = remove_cvref_t<typename BlockGemm::Problem>;
        constexpr index_t kMPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t kNPerBlock = Problem::BlockGemmShape::kN;

        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template At<0>())>;

        constexpr index_t kNPerStep = WG::kN;
        constexpr index_t kN1       = 8;
        constexpr index_t kN0       = kNPerStep / kN1;
        static_assert(kNPerBlock % kNPerStep == 0,
                      "kNPerStep must be evenly divided by kNPerBlock");

        constexpr auto randval_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<kN0>{}, Number<kMPerBlock>{}, Number<kN1>{}),
            make_tuple(Number<(kMPerBlock + 1) * kN1>{}, Number<kN1>{}, Number<1>{}),
            Number<kN1>{},
            Number<1>{});

        constexpr auto randval_lds_block_desc = transform_tensor_descriptor(
            randval_lds_block_desc_0,
            make_tuple(make_pass_through_transform(Number<kMPerBlock>{}),
                       make_merge_transform(make_tuple(Number<kN0>{}, Number<kN1>{}))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return randval_lds_block_desc;
    }

    template <typename BlockGemm>
    __host__ __device__ static constexpr auto MakeRandValSramTileDistribution()
    {
        using Problem               = remove_cvref_t<typename BlockGemm::Problem>;
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;

        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template At<0>())>;

        constexpr index_t kNPerStep = WG::kN;
        constexpr index_t kN1       = 1;
        constexpr index_t kN0       = kNPerStep / kN1;

        constexpr index_t kM3 = 8;
        constexpr index_t kM2 = get_warp_size() / kN0;
        constexpr index_t kM1 = 2;
        constexpr index_t kM0 = MPerBlock / (kM1 * kM2 * kM3);

        static_assert(NPerBlock % kNPerStep == 0, "kNPerStep must be evenly divided by NPerBlock");

        // Construct Drop-Block-Tensor
        constexpr auto randval_block_dstr_encoding =
            StaticTileDistributionEncoding<Sequence<>,
                                           Tuple<Sequence<kM0, kM1, kM2, kM3>, Sequence<kN0, kN1>>,
                                           Tuple<Sequence<1>, Sequence<1, 2>>,
                                           Tuple<Sequence<0>, Sequence<2, 0>>,
                                           Sequence<1, 1, 2>,
                                           Sequence<1, 3, 1>>{};

        constexpr auto randval_block_dstr =
            make_static_tile_distribution(randval_block_dstr_encoding);

        return randval_block_dstr;
    }

    template <typename BlockGemm>
    __host__ __device__ static constexpr auto MakeRandValSramPartTileDistribution()
    {
        constexpr auto config =
            BlockGemm::Policy::template GetWarpGemmMWarpNWarp<typename BlockGemm::Problem>();

        using WG                = remove_cvref_t<decltype(config.template At<0>())>;
        constexpr index_t MWarp = config.template At<1>();
        constexpr index_t NWarp = config.template At<2>();

        constexpr index_t MIterPerWarp = BlockGemm::BlockGemmShape::kM / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = 1; // only a part s_acc distribution

        constexpr auto randval_block_outer_part_dstr_encoding = StaticTileDistributionEncoding<
            Sequence<>,
            Tuple<Sequence<MIterPerWarp, MWarp>, Sequence<NIterPerWarp, NWarp>>,
            Tuple<Sequence<1, 2>>,
            Tuple<Sequence<1, 1>>,
            Sequence<1, 2>,
            Sequence<0, 0>>{};

        constexpr auto randval_block_part_dstr_encode =
            detail::make_embed_tile_distribution_encoding(randval_block_outer_part_dstr_encoding,
                                                          typename WG::CWarpDstrEncoding{});

        return make_static_tile_distribution(randval_block_part_dstr_encode);
    }

    template <typename BlockGemm,
              typename RandValOutputDataType,
              typename PComputeWindow,
              typename RandValDramWindow>
    __host__ __device__ void Run(void* randval_ptr,
                                 const index_t start_n0_idx,
                                 PComputeWindow& p_compute,
                                 RandValDramWindow& randval_dram_window) const
    {
        using Problem               = remove_cvref_t<typename BlockGemm::Problem>;
        using BlockGemmShape        = remove_cvref_t<typename BlockGemm::BlockGemmShape>;
        static constexpr index_t kN = BlockGemmShape::kN;

        // Block GEMM
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG              = remove_cvref_t<decltype(config.template At<0>())>;

        // rand val tile in LDS
        auto randval_lds = make_tensor_view<AddressSpaceEnum::Lds>(
            reinterpret_cast<uint8_t*>(randval_ptr), MakeRandValLdsBlockDescriptor<BlockGemm>());

        auto randval_lds_window = make_tile_window(
            randval_lds, MakeRandValLdsBlockDescriptor<BlockGemm>().GetLengths(), {0, 0});

        // register distribute
        auto randval_distr_generated =
            make_static_distributed_tensor<uint8_t>(MakeRandValSramTileDistribution<BlockGemm>());

        auto randval_lds_read_window =
            make_tile_window(randval_lds_window.GetBottomTensorView(),
                             randval_lds_window.GetWindowLengths(),
                             randval_lds_window.GetWindowOrigin(),
                             MakeRandValSramPartTileDistribution<BlockGemm>());

        constexpr index_t kNPerStep = WG::kN;
        static_for<0, kN / kNPerStep, 1>{}([&](auto i_n0) {
            auto warp_id  = get_warp_id();
            auto lane_idx = get_lane_id();
            auto idx_n    = lane_idx % kNPerStep + i_n0 * kNPerStep + start_n0_idx;
            auto idx_m    = (lane_idx / kNPerStep) * 8 + warp_id * WG::kM;
            const uint64_t element_global_1d_id = idx_m * total_n_len + idx_n + global_idx;

            // generate random number
            uint8_t tmp[16];
            ph.get_random_16x8(tmp, element_global_1d_id);

            static_assert(randval_distr_generated.kThreadElementSpaceSize == 16, "Wrong!");
            constexpr auto randval_distr_generated_spans =
                decltype(randval_distr_generated)::GetDistributedSpans();
            int i_random_idx = 0;
            sweep_tile_span(randval_distr_generated_spans[Number<0>{}], [&](auto idx0) {
                sweep_tile_span(randval_distr_generated_spans[Number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx           = make_tuple(idx0, idx1);
                    randval_distr_generated(i_j_idx) = type_convert<uint8_t>(tmp[i_random_idx++]);
                });
            });
            // save to LDS
            store_tile(randval_lds_window, randval_distr_generated);
            block_sync_lds(); // wait data save to LDS
            // read form LDS to register
            auto randval_for_dropping = load_tile(randval_lds_read_window);
            constexpr auto randval_for_dropping_spans =
                decltype(randval_for_dropping)::GetDistributedSpans();
            sweep_tile_span(randval_for_dropping_spans[Number<0>{}], [&](auto idx0) {
                sweep_tile_span(randval_for_dropping_spans[Number<1>{}], [&](auto idx1) {
                    constexpr auto second_ =
                        TileDistributedIndex<i_n0, idx1.impl_.At(1), idx1.impl_.At(2)>{};
                    constexpr auto p_idx = make_tuple(idx0, second_);
                    constexpr auto d_idx = make_tuple(idx0, idx1);
                    p_compute(p_idx)     = randval_for_dropping[d_idx] <= p_undrop_in_uint8_t
                                               ? p_compute[p_idx] * p_dropout_rescale
                                               : float(0);
                });
            });
            // save to Global
            if(is_store_randval)
            {
                const auto randval_store = cast_tile<RandValOutputDataType>(randval_for_dropping);
                store_tile(randval_dram_window, randval_store);
                __builtin_amdgcn_sched_barrier(0);
                move_tile_window(randval_dram_window, {0, WG::kN});
            }
        });
    }

    long_index_t global_idx;
    float p_dropout_rescale;
    uint8_t p_undrop_in_uint8_t;
    index_t total_n_len;
    ck::philox ph;
    bool is_store_randval = false;
};

} // namespace block
} // namespace tile_program
} // namespace ck
