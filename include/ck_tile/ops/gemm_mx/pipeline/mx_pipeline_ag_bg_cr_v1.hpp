// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm_mx/pipeline/mx_pipeline_ag_bg_cr_v1_policy.hpp"

namespace ck_tile {

template <typename Problem, typename PipelinePolicy = MXGemmPipelineAgBgCrPolicy<Problem>>
struct MXGemmPipelineAgBgCrV1
{
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    using ComputeType = ADataType;
    static_assert(sizeof(ADataType) >= sizeof(BDataType));

    using ALayout = remove_cvref_t<typename Problem::ALayout>;
    using BLayout = remove_cvref_t<typename Problem::BLayout>;
    using CLayout = remove_cvref_t<typename Problem::CLayout>;

    using AsDataType = ck_tile::tuple<ADataType>;
    using BsDataType = ck_tile::tuple<BDataType>;
    using AsLayout   = ck_tile::tuple<ALayout>;
    using BsLayout   = ck_tile::tuple<BLayout>;
    using AElementWise = element_wise::PassThrough;
    using BElementWise = element_wise::PassThrough;

    static constexpr index_t APackedSize = numeric_traits<ADataType>::PackedSize;
    static constexpr index_t BPackedSize = numeric_traits<BDataType>::PackedSize;

    using BlockFlatmm =
        remove_cvref_t<decltype(PipelinePolicy::GetBlockFlatmm())>;

    static constexpr auto config =
        BlockFlatmm::BlockPolicy::template GetWarpGemmMWarpNWarp<Problem>();

    using WG = remove_cvref_t<decltype(config.template at<0>())>;

    static constexpr index_t BlockSize = Problem::kBlockSize;
    static constexpr index_t WaveSize  = get_warp_size();
    static constexpr index_t NumWaveGroups = BlockSize / WaveSize;
    static constexpr bool UsePersistentKernel = true;

    static constexpr index_t kMPerBlock = BlockGemmShape::kM;
    static constexpr index_t kNPerBlock = BlockGemmShape::kN;
    static constexpr index_t kKPerBlock = BlockGemmShape::kK;

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr index_t MWarp = config.template at<1>();
    static constexpr index_t NWarp = config.template at<2>();

    static constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WG::kM);
    static constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WG::kN);
    static constexpr index_t KIterPerWarp = kKPerBlock / WG::kK;

    static constexpr index_t MXdlPack          = Problem::MXdlPack;
    static constexpr index_t NXdlPack          = Problem::NXdlPack;
    static constexpr index_t KXdlPack          = Problem::KXdlPack;
    
    static constexpr index_t MPackIterPerWarp = MIterPerWarp / MXdlPack;
    static constexpr index_t NPackIterPerWarp = NIterPerWarp / NXdlPack;
    static constexpr index_t KPackIterPerWarp = KIterPerWarp / KXdlPack;

    CK_TILE_HOST_DEVICE static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > 0;
    }

    CK_TILE_HOST_DEVICE static constexpr TailNumber GetBlockLoopTailNum(index_t /* num_loop */)
    {
        return TailNumber::Full;
    }

    template <bool HasHotLoop, typename Callable>
    CK_TILE_HOST_DEVICE static auto TailHandler(Callable&& f, bool /* has_hot_loop */, TailNumber /* tail_num */)
    {
        return f(bool_constant<HasHotLoop>{}, constant<TailNumber::Full>{});
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return PipelinePolicy::GetSmemSize();
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeA()
    {
        return APackedSize;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeB()
    {
        return BPackedSize;
    }

    static constexpr bool Preshuffle = false;

    template <typename... Args>
    CK_TILE_DEVICE auto operator()(Args&&... args) const
    {
        auto c_warp_tensors = Run_(std::forward<Args>(args)...);

        // Block GEMM Acc register tile
        using CWarpDstr = typename WG::CWarpDstr;
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};
        auto c_block_tile                   = BlockFlatmm{}.MakeCBlockTile();
        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                c_block_tile.set_y_sliced_thread_data(
                    merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                    c_warp_tensors(mIter)(nIter).get_thread_buffer());
            });
        });
        return c_block_tile;
    }

    template <typename ADramBlockWindowTmp,
              typename BFlatBlockWindowTmp,
              typename ScaleADramBlockWindowTmp,
              typename ScaleBDramBlockWindowTmp>
    CK_TILE_DEVICE auto Run_(const ADramBlockWindowTmp& a_copy_dram_window_tmp,
                             const BFlatBlockWindowTmp& b_flat_dram_block_window_tmp,
                             const ScaleADramBlockWindowTmp& scale_a_window,
                             const ScaleBDramBlockWindowTmp& scale_b_window,
                             index_t num_loop,
                             void* __restrict__ p_smem_ping,
                             void* __restrict__ p_smem_pong) const
    {
        using CWarpTensor = typename WG::CWarpTensor;

        // A DRAM Window
        auto a_dram_window =
            make_tile_window(PipelinePolicy::MakeMX_AAsyncLoadDramDescriptor(
                                 a_copy_dram_window_tmp.at(number<0>{}).get_bottom_tensor_view()),
                             a_copy_dram_window_tmp.at(number<0>{}).get_window_lengths(),
                             a_copy_dram_window_tmp.at(number<0>{}).get_window_origin(),
                             PipelinePolicy::MakeMX_ADramTileDistribution());

        // B DRAM Window
        auto b_dram_window =
            make_tile_window(PipelinePolicy::MakeMX_BAsyncLoadDramDescriptor(
                                 b_flat_dram_block_window_tmp.at(number<0>{}).get_bottom_tensor_view()),
                             b_flat_dram_block_window_tmp.at(number<0>{}).get_window_lengths(),
                             b_flat_dram_block_window_tmp.at(number<0>{}).get_window_origin(),
                             PipelinePolicy::MakeMX_BDramTileDistribution());

        // Scale A DRAM Window
        // With 1D K-only packing: window size is [MWarp * WG::kM, kKPerBlock / 32 / KXdlPack]
        constexpr index_t ScaleBlockSize = 32;
        auto scale_a_dram_window = make_tile_window(
            scale_a_window.get_bottom_tensor_view(),
            make_tuple(number<MWarp * WG::kM>{}, number<kKPerBlock / ScaleBlockSize / KXdlPack>{}),
            scale_a_window.get_window_origin(),
            PipelinePolicy::MakeMX_ScaleA_FlatDramTileDistribution());
        const auto scale_a_dram_step_m = amd_wave_read_first_lane(
            scale_a_dram_window.get_load_offset(tuple<number<MWarp * WG::kM>, number<0>>{}));
        const auto scale_a_dram_step_k = amd_wave_read_first_lane(
            scale_a_dram_window.get_load_offset(tuple<number<0>, number<kKPerBlock / ScaleBlockSize / KXdlPack>>{}));

        // Scale B DRAM Window
        // With 1D K-only packing and [K/32/4, N] layout: window size is [kKPerBlock / 32 / KXdlPack, NWarp * WG::kN]
        auto scale_b_dram_window = make_tile_window(
            scale_b_window.get_bottom_tensor_view(),
            make_tuple(number<kKPerBlock / ScaleBlockSize / KXdlPack>{}, number<NWarp * WG::kN>{}),
            scale_b_window.get_window_origin(),
            PipelinePolicy::MakeMX_ScaleB_DramTileDistribution());
        const auto scale_b_dram_step_k = amd_wave_read_first_lane(
            scale_b_dram_window.get_load_offset(tuple<number<kKPerBlock / ScaleBlockSize / KXdlPack>, number<0>>{}));
        const auto scale_b_dram_step_n = amd_wave_read_first_lane(
            scale_b_dram_window.get_load_offset(tuple<number<0>, number<NWarp * WG::kN>>{}));

        // LDS Views
        ADataType* p_a_lds_ping = static_cast<ADataType*>(p_smem_ping);
        ADataType* p_a_lds_pong = static_cast<ADataType*>(p_smem_pong);
        
        constexpr index_t a_lds_bytes = PipelinePolicy::GetSmemSizeA();
        BDataType* p_b_lds_ping = reinterpret_cast<BDataType*>(reinterpret_cast<char*>(p_smem_ping) + a_lds_bytes);
        BDataType* p_b_lds_pong = reinterpret_cast<BDataType*>(reinterpret_cast<char*>(p_smem_pong) + a_lds_bytes);

        constexpr auto a_lds_block_desc = PipelinePolicy::MakeMX_ALdsBlockDescriptor();
        constexpr auto b_lds_block_desc = PipelinePolicy::MakeMX_BLdsBlockDescriptor();

        auto a_lds_block_ping = make_tensor_view<address_space_enum::lds>(p_a_lds_ping, a_lds_block_desc);
        auto a_lds_block_pong = make_tensor_view<address_space_enum::lds>(p_a_lds_pong, a_lds_block_desc);
        auto b_lds_block_ping = make_tensor_view<address_space_enum::lds>(p_b_lds_ping, b_lds_block_desc);
        auto b_lds_block_pong = make_tensor_view<address_space_enum::lds>(p_b_lds_pong, b_lds_block_desc);

        // Store Windows (for Async Copy)
        auto a_store_lds_window_ping = make_tile_window(a_lds_block_ping, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {0, 0});
        auto a_store_lds_window_pong = make_tile_window(a_lds_block_pong, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {0, 0});
        auto b_store_lds_window_ping = make_tile_window(b_lds_block_ping, make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}), {0, 0});
        auto b_store_lds_window_pong = make_tile_window(b_lds_block_pong, make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}), {0, 0});

        // Load Windows (for Warp Load)
        auto a_warp_window_ping = make_tile_window(a_lds_block_ping, make_tuple(number<MWarp * WG::kM>{}, number<WG::kK>{}), {0, 0}, PipelinePolicy::MakeMX_ALDS_TileDistribution());
        auto a_warp_window_pong = make_tile_window(a_lds_block_pong, make_tuple(number<MWarp * WG::kM>{}, number<WG::kK>{}), {0, 0}, PipelinePolicy::MakeMX_ALDS_TileDistribution());
        auto b_warp_window_ping = make_tile_window(b_lds_block_ping, make_tuple(number<NWarp * WG::kN>{}, number<WG::kK>{}), {0, 0}, PipelinePolicy::MakeMX_BLDS_TileDistribution());
        auto b_warp_window_pong = make_tile_window(b_lds_block_pong, make_tuple(number<NWarp * WG::kN>{}, number<WG::kK>{}), {0, 0}, PipelinePolicy::MakeMX_BLDS_TileDistribution());

        // Register Tiles
        statically_indexed_array<statically_indexed_array<CWarpTensor, NIterPerWarp>, MIterPerWarp> c_warp_tensors;
        
        // Initialize C
        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                clear_tile(c_warp_tensors(mIter)(nIter));
            });
        });

        // Scale Tiles
        // With 1D K-only packing: one scale tile per M/N iter, indexed by K packed iter
        // K dimension: each K iter processes WG::kK elements, each int32 has KXdlPack scales covering KXdlPack*32 elements
        // So each KIterPerWarp needs KIterPerWarp/(KXdlPack) packed scale elements
        constexpr index_t ScaleKPackedPerIter = (KIterPerWarp * WG::kK) / (32 * KXdlPack);
        using ScaleATileType = statically_indexed_array<statically_indexed_array<decltype(load_tile_with_offset(scale_a_dram_window, tuple<number<0>, number<0>>{})), ScaleKPackedPerIter>, MIterPerWarp>;
        using ScaleBTileType = statically_indexed_array<statically_indexed_array<decltype(load_tile_with_offset(scale_b_dram_window, tuple<number<0>, number<0>>{})), ScaleKPackedPerIter>, NIterPerWarp>;

        ScaleATileType scale_a_tile_ping, scale_a_tile_pong;
        ScaleBTileType scale_b_tile_ping, scale_b_tile_pong;

        auto async_load_tile_ = [](auto lds, auto dram) {
            async_load_tile(lds, dram, number<-1>{}, true_type{}, true_type{});
        };

        auto load_scales_ = [&](auto& scale_a, auto& scale_b) {
            // Load scales for each M/N iteration
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                static_for<0, ScaleKPackedPerIter, 1>{}([&](auto kPacked) {
                    scale_a(mIter)(kPacked) = load_tile_with_offset(
                        scale_a_dram_window, mIter * scale_a_dram_step_m + kPacked * scale_a_dram_step_k);
                });
            });
            static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                static_for<0, ScaleKPackedPerIter, 1>{}([&](auto kPacked) {
                    // Scale B is [K/32/4, N], so K is first dimension
                    scale_b(nIter)(kPacked) = load_tile_with_offset(
                        scale_b_dram_window, kPacked * scale_b_dram_step_k + nIter * scale_b_dram_step_n);
                });
            });
            move_tile_window(scale_a_dram_window, {0, kKPerBlock / ScaleBlockSize / KXdlPack});
            move_tile_window(scale_b_dram_window, {kKPerBlock / ScaleBlockSize / KXdlPack, 0});
        };

        // Helper for Main Loop
        auto warp_gemm_loop = [&](auto& a_warp_window, auto& b_warp_window, auto& scale_a, auto& scale_b) {
            // Define register tiles types for double buffering
            using AValType = decltype(load_tile_with_offset(a_warp_window, tuple<number<0>, number<0>>{}));
            using BValType = decltype(load_tile_with_offset(b_warp_window, tuple<number<0>, number<0>>{}));
            
            statically_indexed_array<statically_indexed_array<AValType, MIterPerWarp>, 2> a_vals;
            statically_indexed_array<statically_indexed_array<BValType, NIterPerWarp>, 2> b_vals;

            auto load_k = [&]<typename K, typename Buf>(const K&, const Buf& buf_idx) {
                 static_for<0, MIterPerWarp, 1>{}([&](auto m_iter) {
                     a_vals(buf_idx)(m_iter) = load_tile_with_offset(
                        a_warp_window,
                        tuple<number<m_iter * MWarp * WG::kM>, number<K{} * WG::kK>>{});
                 });
                 static_for<0, NIterPerWarp, 1>{}([&](auto n_iter) {
                     b_vals(buf_idx)(n_iter) = load_tile_with_offset(
                        b_warp_window,
                        tuple<number<n_iter * NWarp * WG::kN>, number<K{} * WG::kK>>{});
                 });
            };

            // Prologue: Load K=0
            load_k(number<0>{}, number<0>{});

            static_for<0, KIterPerWarp, 1>{}([&](auto k_iter) {
                constexpr auto cur_buf = k_iter % 2;
                constexpr auto nxt_buf = (k_iter + 1) % 2;

                // Prefetch K+1
                if constexpr(k_iter < KIterPerWarp - 1) {
                    load_k(number<k_iter + 1>{}, number<nxt_buf>{});
                }

                // Map k_iter to packed scale index
                // Each k_iter processes WG::kK elements
                // Each packed int32 contains KXdlPack scales, each covering 32 elements
                // So we need k_iter * WG::kK / (32 * KXdlPack) to get the packed index
                // and k_iter * WG::kK / 32 % KXdlPack to get which scale within the pack
                constexpr index_t kScalePacked = (k_iter * WG::kK) / (32 * KXdlPack);
                constexpr index_t kScaleInPack = ((k_iter * WG::kK) / 32) % KXdlPack;

                static_for<0, MIterPerWarp, 1>{}([&](auto m_iter) {
                    // OpSel selects which of the KXdlPack packed e8m0 values to use
                    constexpr auto OpSelA = kScaleInPack;

                    static_for<0, NIterPerWarp, 1>{}([&](auto n_iter) {
                        // OpSel selects which of the KXdlPack packed e8m0 values to use
                        constexpr auto OpSelB = kScaleInPack;

                        WG{}.template operator()<OpSelA, OpSelB>(
                            c_warp_tensors(m_iter)(n_iter),
                            bit_cast<typename WG::AWarpTensor>(a_vals(number<cur_buf>{})(m_iter)),
                            bit_cast<typename WG::BWarpTensor>(b_vals(number<cur_buf>{})(n_iter)),
                            scale_a(m_iter)(number<kScalePacked>{}).get_thread_buffer()[0],
                            scale_b(n_iter)(number<kScalePacked>{}).get_thread_buffer()[0]);
                    });
                });
            });
        };

        // Prologue: Load first block
        async_load_tile_(a_store_lds_window_ping, a_dram_window);
        async_load_tile_(b_store_lds_window_ping, b_dram_window);
        
        // Load Scales (Ping - Iter 0)
        load_scales_(scale_a_tile_ping, scale_b_tile_ping);

        // Load Scales (Pong - Iter 1)
        if (num_loop > 1) {
            load_scales_(scale_a_tile_pong, scale_b_tile_pong);
        }

        // Move DRAM windows
        move_tile_window(a_dram_window, {0, kKPerBlock});
        move_tile_window(b_dram_window, {0, kKPerBlock});
        // Scale windows already moved in load_scales_

        // Main Loop
        index_t i = 0;
        do {
            // Wait for LDS load
            s_waitcnt<0>(); 
            block_sync_lds();

            // Trigger next load (Ping-Pong)
            if (i < num_loop - 1) {
                if (i % 2 == 0) {
                    async_load_tile_(a_store_lds_window_pong, a_dram_window);
                    async_load_tile_(b_store_lds_window_pong, b_dram_window);
                } else {
                    async_load_tile_(a_store_lds_window_ping, a_dram_window);
                    async_load_tile_(b_store_lds_window_ping, b_dram_window);
                }
                move_tile_window(a_dram_window, {0, kKPerBlock});
                move_tile_window(b_dram_window, {0, kKPerBlock});
            }

            // Compute
            if (i % 2 == 0) {
                warp_gemm_loop(a_warp_window_ping, b_warp_window_ping, scale_a_tile_ping, scale_b_tile_ping);
                // Load next scales (Ping - Iter i+2)
                if (i + 2 < num_loop) {
                    load_scales_(scale_a_tile_ping, scale_b_tile_ping);
                }
            } else {
                warp_gemm_loop(a_warp_window_pong, b_warp_window_pong, scale_a_tile_pong, scale_b_tile_pong);
                // Load next scales (Pong - Iter i+2)
                if (i + 2 < num_loop) {
                    load_scales_(scale_a_tile_pong, scale_b_tile_pong);
                }
            }
            
            i++;
        } while (i < num_loop);

        return c_warp_tensors;
    }
};

} // namespace ck_tile
