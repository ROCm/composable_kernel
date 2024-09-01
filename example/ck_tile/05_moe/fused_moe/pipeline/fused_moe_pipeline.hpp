// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async_default_policy.hpp"
#include "ck_tile/ops/fmha/block/block_dropout.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

// a variation of qr/ks/vs, where we use async copy to load k (potentially v in the future)
template <typename Problem_, typename Policy_ = BlockFmhaPipelineQRKSVSAsyncDefaultPolicy>
struct FusedMoePipeline
{
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    using ADataType     = remove_cvref_t<typename Problem::ADataType>;
    using GDataType     = remove_cvref_t<typename Problem::GDataType>;
    using UDataType     = remove_cvref_t<typename Problem::UDataType>;
    using DDataType     = remove_cvref_t<typename Problem::DDataType>;
    using ODataType     = remove_cvref_t<typename Problem::ODataType>;
    using AccDataType   = remove_cvref_t<typename Problem::AccDataType>;
    using ScaleDataType = remove_cvref_t<typename Problem::ScaleDataType>;

    using FusedMoeTileShape = remove_cvref_t<typename Problem::FusedMoeTileShape>;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kBlockM_0      = FusedMoeTileShape::kBlockM_0;
    static constexpr index_t kBlockN_0      = FusedMoeTileShape::kBlockN_0;
    static constexpr index_t kBlockK_0      = FusedMoeTileShape::kBlockK_0;
    static constexpr index_t kWarpM_0       = FusedMoeTileShape::kWarpM_0;
    static constexpr index_t kWarpN_0       = FusedMoeTileShape::kWarpN_0;
    static constexpr index_t kWarpK_0       = FusedMoeTileShape::kWarpK_0;
    static constexpr index_t kBlockWarpsM_0 = FusedMoeTileShape::kBlockWarpsM_0;
    static constexpr index_t kBlockWarpsN_0 = FusedMoeTileShape::kBlockWarpsN_0;
    static constexpr index_t kBlockWarpsK_0 = FusedMoeTileShape::kBlockWarpsK_0;
    static constexpr index_t kSubBlockM_0   = FusedMoeTileShape::kSubBlockM_0;
    static constexpr index_t kSubBlockN_0   = FusedMoeTileShape::kSubBlockN_0;
    static constexpr index_t kSubBlockK_0   = FusedMoeTileShape::kSubBlockK_0;
    static constexpr index_t kWarpRepeatM_0 = FusedMoeTileShape::kWarpRepeatM_0;
    static constexpr index_t kWarpRepeatN_0 = FusedMoeTileShape::kWarpRepeatN_0;
    static constexpr index_t kWarpRepeatK_0 = FusedMoeTileShape::kWarpRepeatK_0;

    static constexpr index_t kBlockM_1      = FusedMoeTileShape::kBlockM_1;
    static constexpr index_t kBlockN_1      = FusedMoeTileShape::kBlockN_1;
    static constexpr index_t kBlockK_1      = FusedMoeTileShape::kBlockK_1;
    static constexpr index_t kWarpM_1       = FusedMoeTileShape::kWarpM_1;
    static constexpr index_t kWarpN_1       = FusedMoeTileShape::kWarpN_1;
    static constexpr index_t kWarpK_1       = FusedMoeTileShape::kWarpK_1;
    static constexpr index_t kBlockWarpsM_1 = FusedMoeTileShape::kBlockWarpsM_1;
    static constexpr index_t kBlockWarpsN_1 = FusedMoeTileShape::kBlockWarpsN_1;
    static constexpr index_t kBlockWarpsK_1 = FusedMoeTileShape::kBlockWarpsK_1;
    static constexpr index_t kSubBlockM_1   = FusedMoeTileShape::kSubBlockM_1;
    static constexpr index_t kSubBlockN_1   = FusedMoeTileShape::kSubBlockN_1;
    static constexpr index_t kSubBlockK_1   = FusedMoeTileShape::kSubBlockK_1;
    static constexpr index_t kWarpRepeatM_1 = FusedMoeTileShape::kWarpRepeatM_1;
    static constexpr index_t kWarpRepeatN_1 = FusedMoeTileShape::kWarpRepeatN_1;
    static constexpr index_t kWarpRepeatK_1 = FusedMoeTileShape::kWarpRepeatK_1;

    using MBlockType                    = decltype(GetMatrixCoreSwizzledBlockTIle_0<Problem>());
    static constexpr index_t kBlockNr_0 = MBlockType {}
    ::at(number<0>{});
    static constexpr index_t kBlockKr_0 = MBlockType {}
    ::at(number<1>{});
    static constexpr index_t kBlockWaveFlatten = MBlockType {}
    ::at(number<2>{});

    static constexpr index_t kBlockPerCu = []() {
        if constexpr(Problem::kBlockPerCu != -1)
            return Problem::kBlockPerCu;
        else
        {
            // minimize occupancy
            return 2;
        }
    }();

    static constexpr const char* name = "qr_async";

    using DropoutType = std::conditional_t<kHasDropout, BlockDropout, NullBlockDropout>;

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    // this is the thread-offset along row/col
    CK_TILE_HOST_DEVICE static auto GetAIndex()
    {
        constexpr auto a_dist = Policy::template MakeGlobalTileDistribution_A<Problem>();
        const auto a_coord    = a_dist.calculate_index();
        return a_coord;
    }

    // this is the thread-offset along row/col
    CK_TILE_HOST_DEVICE static auto GetOIndex()
    {
        constexpr auto o_dist = Policy::template MakeOGlobalTileDistribution<Problem>();
        const auto o_coord    = o_dist.calculate_index();
        return o_coord;
    }

    template <typename AGlobalTensorView,
              typename GGlobalTileWindow,
              typename UGlobalTileWindow,
              typename DGlobalTileWindow,
              typename OGlobalTensorView>
    CK_TILE_DEVICE auto operator()(const AGlobalTensorView& a_gtile_window_tmp,
                                   const GGlobalTileWindow& g_gtile_window_tmp,
                                   const UGlobalTileWindow& u_gtile_window_tmp,
                                   const DGlobalTileWindow& d_gtile_window_tmp,
                                   OGlobalTensorView& o_gtile_window_tmp,
                                   //  const void  *  sorted_weight_ptr,
                                   ScaleDataType scale,
                                   CK_TILE_LDS_ADDR void* smem_0,
                                   CK_TILE_LDS_ADDR void* smem_1,
                                   index_t dim_size,
                                   index_t hidden_size)
    {
        constexpr auto gemm_0 = Policy::template GetGemm0<Problem>();
        constexpr auto gemm_1 = Policy::template GetGemm1<Problem>();

        auto a_gtile_window =
            make_tile_window(a_gtile_window_tmp.get_bottom_tensor_view(),
                             a_gtile_window_tmp.get_window_lengths(),
                             a_gtile_window_tmp.get_window_origin(),
                             Policy::template MakeGlobalTileDistribution_A<Problem>());

        auto g_gtile_window =
            make_tile_window(g_gtile_window_tmp.get_bottom_tensor_view(),
                             g_gtile_window_tmp.get_window_lengths(),
                             g_gtile_window_tmp.get_window_origin(),
                             Policy::template MakeGlobalTileDistribution_G<Problem>());

        auto u_gtile_window =
            make_tile_window(u_gtile_window_tmp.get_bottom_tensor_view(),
                             u_gtile_window_tmp.get_window_lengths(),
                             u_gtile_window_tmp.get_window_origin(),
                             Policy::template MakeGlobalTileDistribution_U<Problem>());

        auto d_gtile_window =
            make_tile_window(d_gtile_window_tmp.get_bottom_tensor_view(),
                             d_gtile_window_tmp.get_window_lengths(),
                             d_gtile_window_tmp.get_window_origin(),
                             Policy::template MakeGlobalTileDistribution_D<Problem>());

        auto o_gtile_window =
            make_tile_window(o_gtile_window_tmp.get_bottom_tensor_view(),
                             o_gtile_window_tmp.get_window_lengths(),
                             o_gtile_window_tmp.get_window_origin(),
                             Policy::template MakeOGlobalTileDistribution<Problem>());
        using g_thread_type = decltype(load_tile(g_gtile_window));
        using u_thread_type = decltype(load_tile(u_gtile_window));
        using d_thread_type = decltype(load_tile(d_gtile_window));

        const index_t loops_0 = (dim_size + kBlockK_0 - 1) / kBlockK_0;
        const index_t loops_1 = (dim_size + kBlockN_1 - 1) / kBlockN_1;

        // auto a_smem_ptr = reinterpret_cast<ADataType*>(smem_ptr) + a_smem_offset;

        // issues_warps_lanes
        auto a_sst_0 =
            make_tile_window(make_tensor_view<address_space_enum::lds>(
                                 smem_0, Policy::template MakeLdsStoreDesc_A<Problem>()),
                             Policy::template MakeLdsStoreDesc_A<Problem>().get_lengths(),
                             {0, 0, 0});

        // issues_warps_lanes
        auto a_sst_1 =
            make_tile_window(make_tensor_view<address_space_enum::lds>(
                                 smem_1, Policy::template MakeLdsStoreDesc_A<Problem>()),
                             Policy::template MakeLdsStoreDesc_A<Problem>().get_lengths(),
                             {0, 0, 0});

        // m*k
        auto a_sld_0 = make_tile_window(make_tensor_view<address_space_enum::lds>(
                                            smem_0, Policy::template MakeLdsLoadDesc_A<Problem>()),
                                        Policy::template MakeLdsLoadDesc_A<Problem>().get_lengths(),
                                        {0, 0});

        // m*k
        auto a_sld_1 = make_tile_window(make_tensor_view<address_space_enum::lds>(
                                            smem_1, Policy::template MakeLdsLoadDesc_A<Problem>()),
                                        Policy::template MakeLdsLoadDesc_A<Problem>().get_lengths(),
                                        {0, 0});

        g_thread_type g_tile[2];
        using WarpGemm0  = Policy::GetWarpGemm0<Problem>();
        using WarpGemm1  = Policy::GetWarpGemm1<Problem>();
        auto warp_gemm_0 = WarpGemm0{};
        auto warp_gemm_1 = WarpGemm1{};

        // TODO: N fist, M next
        const index_t i_mwarp_0 = get_warp_id() / kBlockWarpsN_0;

        // create and pre-cache a warp-window
        auto make_a_warp_windows = [&](auto a_sld_) {
            // construct A-warp-window
            auto warp_window = make_tile_window(
                a_sld_.get_bottom_tensor_view(),
                make_tuple(number<WarpGemm0::kM>{}, number<WarpGemm0::kK>{}),
                a_sld_.get_window_origin() + multi_index<2>{i_mwarp_0 * WarpGemm0::kM, 0},
                make_static_tile_distribution(typename WarpGemm0::AWarpDstrEncoding{}));
            statically_indexed_array<
                statically_indexed_array<decltype(warp_window), kWarpRepeatK_0>,
                kWarpRepeatM_0>
                ws;
            // pre-cache the warp windows
            static_for<0, kWarpRepeatM_0, 1>{}([&](auto i_m_iter) {
                static_for<0, kWarpRepeatK_0, 1>{}([&](auto i_k_iter) {
                    ws(i_m_iter)(i_k_iter) = warp_window;
                    move_tile_window(ws(i_m_iter)(i_k_iter),
                                     {i_m_iter * NPerBlockPerIter, i_k_iter * KPerBlockPerIter});
                });
            });
            return ws;
        };

        auto a_warp_windows_0 = make_a_warp_windows(a_sld_0);
        auto a_warp_windows_1 = make_a_warp_windows(a_sld_1);

        constexpr auto true_v  = bool_constant<true>{};
        constexpr auto false_v = bool_constant<false>{};
        auto do_load_a0        = [&](auto& a_store_, auto move_) {
            async_load_tile(a_store_, a_gtile_window);
            if constexpr(move_)
                move_tile_window(a_gtile_window, {number<0>{}, number<kBlockK_0>{}});
        };

        auto do_load_b0 = [&](auto& g_tile_, auto& u_tile_, auto move_) {
            g_tile_ = load_tile(g_gtile_window);
            u_tile_ = load_tile(u_gtile_window);
            if constexpr(move_)
            {
                move_tile_window(g_gtile_window, {number<0>{}, number<kBlockKr_0>{}, number<0>{}});
                move_tile_window(u_gtile_window, {number<0>{}, number<kBlockKr_0>{}, number<0>{}});
            }
        };

        auto do_load_b1 = [&](auto& d_tile_, auto move_) {
            d_tile_ = load_tile(d_gtile_window);
            if constexpr(move_)
            {
                move_tile_window(d_gtile_window, {number<0>{}, number<kBlockKr_0>{}, number<0>{}});
            }
        };

        // using AWarpTensor = typename decltype(warp_gemm_0)::AWarpTensor{};
        // using CWarpTensor =

        auto acc_g = MakeCBlockTile_Gemm0<Problem>();
        auto acc_u = MakeCBlockTile_Gemm0<Problem>();

        // async_load_tile(a_sst_0, a_gtile_window); move_tile_window(a_gtile_window, {number<0>{},
        // number<kBlockK_0>{}}); g_tile[0] = load_tile(g_gtile_window);
        // move_tile_window(g_gtile_window, {number<0>{}, number<kBlockK_0>{}}); u_tile[0] =
        // load_tile(u_gtile_window); move_tile_window(u_gtile_window, {number<0>{},
        // number<kBlockK_0>{}}); async_load_tile(a_sst_1, a_gtile_window);
        // move_tile_window(a_gtile_window, {number<0>{}, number<kBlockK_0>{}}); g_tile[1] =
        // load_tile(g_gtile_window); move_tile_window(g_gtile_window, {number<0>{},
        // number<kBlockK_0>{}}); u_tile[1] = load_tile(u_gtile_window);
        // move_tile_window(u_gtile_window, {number<0>{}, number<kBlockK_0>{}});

        auto do_gemm_0 =
            [&](auto& acc_g_, auto& acc_u_, auto& a_windows_, auto& g_tile_, auto& u_tile_) {
                // as_br (asmem, breg)
                static_for<0, kWarpRepeatK_0, 1>{}([&](auto i_k) {
                    static_for<0, kWarpRepeatM_0, 1>{}([&](auto i_m) {
                        const auto w_a = load_tile(a_windows_(i_m)(i_k));

                        static_for<0, kWarpRepeatN_0, 1>{}([&](auto i_n) {
                            constexpr auto beg_acc =
                                sequence<i_m * kSubBlockM_0, i_n * kSubBlockN_0>{};
                            constexpr auto end_acc =
                                sequence<(i_m + 1) * kSubBlockM_0, (i_n + 1) * kSubBlockN_0>{};

                            // 3d indexing for permuted g/u/d
                            constexpr auto beg_b =
                                sequence<i_m * kBlockWarpsM_0, i_n * kSubBlockN_0, 0>{};
                            constexpr auto end_b =
                                sequence<(i_m + 1) * kBlockWarpsM_0, (i_n + 1) * kSubBlockN_0, 0>{};

                            auto w_acc_g = get_slice_tile(acc_g_, beg_acc, end_acc);
                            auto w_acc_u = get_slice_tile(acc_u_, beg_acc, end_acc);
                            auto w_g     = get_slice_tile(g_tile_, beg_b, end_b);
                            auto w_u     = get_slice_tile(u_tile_, beg_b, end_b);

                            warp_gemm_0(w_acc_g, w_a, w_g);
                            warp_gemm_0(w_acc_u, w_a, w_u);

                            set_slice_tile(acc_g_, w_acc_g, beg_acc, end_acc);
                            set_slice_tile(acc_u_, w_acc_u, beg_acc, end_acc);
                        });
                    });
                });
            };

        auto do_gemm_1 = [&](auto& acc_d_, auto& a_tile_, auto& d_tile_) {
            // ar_br (areg, breg)
            static_for<0, kWarpRepeatK_1, 1>{}([&](auto i_k) {
                static_for<0, kWarpRepeatM_1, 1>{}([&](auto i_m) {
                    constexpr auto beg_a = sequence<i_m * kSubBlockM_1, i_k * kSubBlockK_1>{};
                    constexpr auto end_a =
                        sequence<(i_m + 1) * kSubBlockM_1, (i_k + 1) * kSubBlockK_1>{};
                    const auto w_a = get_slice_tile(a_tile_, beg_a, end_a);

                    static_for<0, kWarpRepeatN_1, 1>{}([&](auto i_n) {
                        constexpr auto beg_acc = sequence<i_m * kSubBlockM_0, i_n * kSubBlockN_0>{};
                        constexpr auto end_acc =
                            sequence<(i_m + 1) * kSubBlockM_0, (i_n + 1) * kSubBlockN_0>{};

                        // 3d indexing for permuted g/u/d
                        constexpr auto beg_b =
                            sequence<i_m * kBlockWarpsM_0, i_n * kSubBlockN_0, 0>{};
                        constexpr auto end_b =
                            sequence<(i_m + 1) * kBlockWarpsM_0, (i_n + 1) * kSubBlockN_0, 0>{};

                        auto w_acc_d = get_slice_tile(acc_d_, beg_acc, end_acc);
                        auto w_d     = get_slice_tile(d_tile_, beg_b, end_b);

                        warp_gemm_1(w_acc_d, w_a, w_d);

                        set_slice_tile(acc_d_, w_acc_d, beg_acc, end_acc);
                    });
                });
            });
        };

        // start of pipeline
        do_load_a0(a_sst_0, true_v);
        do_load_b0(g_tile[0], u_tile[0], true_v);
        do_load_a0(a_sst_1, true_v);
        do_load_b0(g_tile[1], u_tile[1], true_v);

        clear_tile(acc_g);
        clear_tile(acc_u);

        index_t i_0 = 0;
        while(i_0 < (loops_0 - 2))
        {
            // first buffer
            do_gemm_0(acc_g, acc_u, a_warp_windows_0, g_tile[0], u_tile[0]);
            do_load_a0(a_sst_0, true_v);
            do_load_b0(g_tile[0], u_tile[0], true_v);
            i_0++;

            // second buffer
            do_gemm_0(acc_g, acc_u, a_warp_windows_1, g_tile[1], u_tile[1]);
            do_load_a0(a_sst_1, true_v);
            do_load_b0(g_tile[1], u_tile[1], true_v);
            i_0++;
        }

        // first buffer
        do_gemm_0(acc_g, acc_u, a_warp_windows_0, g_tile[0], u_tile[0]);

        // prefetch
        d_thread_type d_tile[2];
        do_load_b1(d_tile[0], true_v);
        do_load_b1(d_tile[1], true_v);

        // second buffer
        do_gemm_0(acc_g, acc_u, a_warp_windows_1, g_tile[1], u_tile[1]);

        // redice acc_g/u
        constexpr auto acc_spans_0 = decltype(acc_g)::get_distributed_spans();
        sweep_tile_span(acc_spans_0[number<0>{}], [&](auto idx0) {
            sweep_tile_span(acc_spans_0[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                element_wise::Silu{}(acc_g(i_j_idx), acc_g(i_j_idx));
                acc_g(i_j_idx) *= acc_u(i_j_idx);
            });
        });

        const auto y = [&]() {
            if constexpr(std::is_same_v<YDataType, fp16_t>)
                return impl::cast_tile_pk_fp16_fp32<YDataType>(acc_g);
            else
                return cast_tile<YDataType>(acc_g);
        }();

        auto acc_d = MakeCBlockTile_Gemm1<Problem>();
        clear_tile(acc_d);

        // TODO: reshuffle? 32x32x8 mfma can avlid LDS reshuffle
        index_t i_1 == 0;
        while(i_1 < (loops_1 - 2))
        {
            // first buffer
            do_gemm_1(acc_d, y, d_tile[0]);
            do_load_b1(d_tile[0], true_v);
            i_1++;

            // second buffer
            do_gemm_1(acc_d, y, d_tile[1]);
            do_load_b1(d_tile[1], true_v);
            i_1++;
        }

        // first buffer
        do_gemm_0(a_warp_windows_0, g_tile[0], g_tile[1]);
        i_0++;

        // second buffer
        do_gemm_0(a_warp_windows_1, g_tile[1], g_tile[1]);
        i_0++;
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename RandValDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename QElementFunction,
              typename KElementFunction,
              typename VElementFunction,
              typename BiasElementFunction,
              typename LSEElementFunction,
              typename SAccElementFunction,
              typename PComputeElementFunction,
              typename OAccElementFunction,
              typename PositionEncoding>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp, // M0*K0 tile
               const QElementFunction& q_element_func,
               const KDramBlockWindowTmp& k_dram_block_window_tmp, // N0*K0 tile
               const KElementFunction& /*k_element_func*/,
               const VDramBlockWindowTmp& v_dram_block_window_tmp, // N1*K1 tile
               const VElementFunction& v_element_func,
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
               const BiasElementFunction& bias_element_func,
               RandValDramBlockWindowTmp& randval_dram_block_window_tmp,
               LSEDramBlockWindowTmp& lse_dram_window_tmp, // M0*1 tile
               const LSEElementFunction& lse_element_func,
               const SAccElementFunction& s_acc_element_func,
               const PComputeElementFunction& p_compute_element_func,
               const OAccElementFunction& o_acc_element_func,
               FmhaMask mask,
               PositionEncoding position_encoding,
               float scale_s,
               void* smem_ptr,
               DropoutType& dropout) const
    {
        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kK0 == KDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kN1 == VDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kK1 == VDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kM0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        constexpr auto LdsSeq = Policy::template GetLdsBufferSequence<Problem>();

        // K tile in LDS
        auto k_lds_ptr   = reinterpret_cast<KDataType*>(smem_ptr);
        auto k_lds_store = generate_tuple(
            [&](auto i_buf) {
                return make_tile_window(
                    make_tensor_view<address_space_enum::lds>(
                        k_lds_ptr, Policy::template MakeKLdsStoreBlockDescriptor<Problem>(i_buf)),
                    Policy::template MakeKLdsStoreBlockDescriptor<Problem>(i_buf).get_lengths(),
                    {0, 0, 0});
            },
            number<Policy::NumPrefetchK>{});

#if K_LDS_LOAD_USE_OFFSET_TRANSFORM
        auto k_lds_load = generate_tuple(
            [&](auto i_buf) {
                return make_tile_window(
                    make_tensor_view<address_space_enum::lds>(
                        k_lds_ptr, Policy::template MakeKLdsLoadBlockDescriptor<Problem>(i_buf)),
                    Policy::template MakeKLdsLoadBlockDescriptor<Problem>(i_buf).get_lengths(),
                    {0, 0});
            },
            number<Policy::NumPrefetchK>{});
#else
        auto k_lds_Load_view = make_tensor_view<address_space_enum::lds>(
            k_lds_ptr, Policy::template MakeKLdsLoadBlockDescriptor<Problem>());

        auto k_lds_load =
            make_tile_window(k_lds_Load_view,
                             Policy::template MakeKLdsLoadBlockDescriptor<Problem>().get_lengths(),
                             {0, 0});
#endif

        // V tile in LDS
        auto v_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<VDataType*>(smem_ptr),
            Policy::template MakeVLdsBlockDescriptor<Problem>());
        auto v_lds_window = make_tile_window(
            v_lds, Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetKVBlockGemm<Problem>();

        auto q_dram_window = make_tile_window(
            q_dram_block_window_tmp.get_bottom_tensor_view(),
            q_dram_block_window_tmp.get_window_lengths(),
            q_dram_block_window_tmp.get_window_origin(),
            Policy::template MakeQDramTileDistribution<Problem, decltype(gemm_0)>());
        q_dram_window.init_raw();

        // TODO: we use async Copy for K, which is inline asm
        // a side effect is we have to use inline asm for q as well
        auto q = decltype(load_tile(q_dram_window)){}; // reg = copy(some_tensor_vew)
        set_tile(q, number<0>{});                      // use per-dword clear to avoid scratch
        load_tile_raw(q, q_dram_window);
        __builtin_amdgcn_sched_barrier(0);

        using SaccBlockTileType = decltype(gemm_0.MakeCBlockTile());
        auto s_acc              = SaccBlockTileType{};

        // reduction function for softmax
        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        // infer Sacc, S, P, M, L, Oacc type
        using SBlockTileType = decltype(cast_tile<SMPLComputeDataType>(s_acc));

        using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
            SBlockTileType{}, sequence<1>{}, f_max, SMPLComputeDataType{0}));

        using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());

        // init Oacc, M, L
        auto o_acc = OaccBlockTileType{};
        auto m     = MLBlockTileType{};
        auto l     = MLBlockTileType{};

        clear_tile(o_acc);
        set_tile(m, -numeric<SMPLComputeDataType>::infinity());
        clear_tile(l);

        __builtin_amdgcn_sched_barrier(0);
        const auto q_origin = q_dram_window.get_window_origin();
        const auto [seqlen_k_start, seqlen_k_end] =
            mask.GetTileRangeAlongX(q_origin.at(number<0>{}), number<kM0>{}, number<kN0>{});

        const auto num_total_loop = integer_divide_ceil(seqlen_k_end - seqlen_k_start, kN0);

        // check early exit
        if constexpr(FmhaMask::IsMasking || kPadSeqLenK)
        {
            if(num_total_loop <= 0)
            {
                if constexpr(kStoreLSE)
                {
                    auto lse =
                        make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

                    set_tile(lse, -numeric<SMPLComputeDataType>::infinity());

                    store_tile(lse_dram_window_tmp, tile_elementwise_in(lse_element_func, lse));
                }
                buffer_load_fence_raw(0); // rocm-6.1, if whole tile is masked out, need to fence(0)
                                          // otherwise will have compute error(maybe compiler bug?)

                // Note: here occ are all cleard, return it
                return o_acc;
            }
            __builtin_amdgcn_sched_barrier(0); // make sure sched_barrier(0) for this check
        }

        auto k_dram_block_window =
            make_tile_window(k_dram_block_window_tmp.get_bottom_tensor_view(),
                             k_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_k_start, 0});

        auto k_dram_window = make_tile_window(
            k_dram_block_window.get_bottom_tensor_view(),
            k_dram_block_window.get_window_lengths(),
            k_dram_block_window.get_window_origin(),
            Policy::template MakeKDramTileDistribution<Problem>()); // K DRAM tile window for
                                                                    // load
        k_dram_window.init_raw();
        constexpr auto k_oob_ck = bool_constant<true>{};
        constexpr auto k_pre_np = [&]() {
            if constexpr(kPadSeqLenK &&
                         (BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                          (BiasEnum != BlockAttentionBiasEnum::NO_BIAS && kHasDropout)))
                return bool_constant<true>{};
            else
                return bool_constant<false>{};
        }();

        const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();
        auto bias_dram_window  = make_tile_window(
            bias_dram_block_window_tmp.get_bottom_tensor_view(),
            bias_dram_block_window_tmp.get_window_lengths(),
            {bias_origin.at(number<0>{}), seqlen_k_start}, // M/N
            Policy::template MakeBiasDramTileDistribution<Problem, decltype(gemm_0)>());

        auto randval_dram_window = dropout.template MakeRandvalDramWindow<decltype(gemm_0)>(
            randval_dram_block_window_tmp, seqlen_k_start);

        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp.get_bottom_tensor_view(),
                             v_dram_block_window_tmp.get_window_lengths(),
                             {0, seqlen_k_start}, // TODO: hdim split?
                             Policy::template MakeVDramTileDistribution<Problem>());

        // prefetch K tile
        async_load_tile_raw(k_lds_store(LdsSeq.at(number<0>{})), k_dram_window, k_oob_ck, k_pre_np);
        move_tile_window(k_dram_window, {0, kK0});
        __builtin_amdgcn_sched_barrier(0);

        buffer_load_fence_raw(k_dram_window.get_num_access(), q.get_thread_buffer());
        (void)q_element_func; // ??? rocm-6.x if use q element func will have scratch on hdim=64/32
        // auto q_tile = q;      // tile_elementwise_in(q_element_func, q);

        index_t i_total_loops      = 0;
        constexpr index_t k0_loops = kK0BlockLength / kK0;
        constexpr index_t k1_loops = kN0 / kK1;

        static_assert(1 <= k0_loops);
        static_assert(1 <= k1_loops);
        // main loop
        do
        {
            // STAGE 1, QK gemm
            clear_tile(s_acc); // initialize C
            if constexpr(k0_loops > 1)
            {
                static_for<0, k0_loops - 1, 1>{}([&](auto i_k0) {
                    async_load_tile_raw(k_lds_store(number<LdsSeq.at(number<i_k0 + 1>{})>{}),
                                        k_dram_window,
                                        k_oob_ck,
                                        k_pre_np);
                    if constexpr(i_k0 < k0_loops - 1)
                        move_tile_window(k_dram_window, {0, kK0});

                    async_load_fence_raw(k_dram_window.get_num_access());
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
                    gemm_0(s_acc,
                           get_slice_tile(
                               q, sequence<0, i_k0 * kK0>{}, sequence<kM0, (i_k0 + 1) * kK0>{}),
#if K_LDS_LOAD_USE_OFFSET_TRANSFORM
                           k_lds_load[number<LdsSeq.at(number<i_k0>{})>{}]);

#else
                           get_slice_tile(k_lds_load,
                                          sequence<(LdsSeq.at(number<i_k0>{})) * kN0, 0>{},
                                          sequence<(LdsSeq.at(number<i_k0>{}) + 1) * kN0, kK0>{}));
#endif
                });
            }

            // TODO: this to fix a bug when loop smaller than 2,
            // the following fence/barrier will be scheduled inside 1st loop
            if constexpr(k0_loops <= 2)
                __builtin_amdgcn_sched_barrier(0);

            async_load_fence_raw();
            __builtin_amdgcn_s_barrier();

            const auto bias_tile = load_tile(bias_dram_window); // load bias tile
            auto v_buf           = load_tile(v_dram_window, bool_constant<false>{});
            __builtin_amdgcn_sched_barrier(0);
            { // tail
                gemm_0(s_acc,
                       get_slice_tile(
                           q, sequence<0, (k0_loops - 1) * kK0>{}, sequence<kM0, k0_loops * kK0>{}),
#if K_LDS_LOAD_USE_OFFSET_TRANSFORM
                       k_lds_load[number<LdsSeq.at(number<k0_loops - 1>{})>{}]);

#else
                       get_slice_tile(
                           k_lds_load,
                           sequence<(LdsSeq.at(number<k0_loops - 1>{})) * kN0, 0>{},
                           sequence<(LdsSeq.at(number<k0_loops - 1>{}) + 1) * kN0, kK0>{}));
#endif
            }
            __builtin_amdgcn_sched_barrier(1);

            // STAGE 2, scale_s, add bias, mask, softmax
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                s_acc = tile_elementwise_in(s_acc_element_func, s_acc);
                tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, s_acc);
                tile_elementwise_inout(
                    [&](auto& x, const auto& y) {
#if !CK_TILE_FMHA_FWD_FAST_EXP2
                        x += type_convert<SaccDataType>(bias_element_func(y));
#else
                        x += log2e_v<SaccDataType> *
                             type_convert<SaccDataType>(bias_element_func(y));
#endif
                    },
                    s_acc,
                    bias_tile);
            }
            else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
            {
                const auto k_origin    = k_dram_block_window.get_window_origin();
                constexpr auto s_spans = decltype(s_acc)::get_distributed_spans();
                s_acc                  = tile_elementwise_in(s_acc_element_func, s_acc);
                sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(s_spans[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            s_acc.get_tile_distribution(), make_tuple(idx0, idx1));

                        const auto row = q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                        const auto col = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);

                        s_acc(i_j_idx) *= scale_s;
                        position_encoding.update(s_acc(i_j_idx), row, col);
                    });
                });
            }
            else
            {
                s_acc = tile_elementwise_in(s_acc_element_func, s_acc);
#if !CK_TILE_FMHA_FWD_FAST_EXP2
                tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, s_acc);
#endif
            }
            move_tile_window(bias_dram_window, {0, kN0});
            if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
            {
                const auto k_origin      = k_dram_block_window.get_window_origin();
                bool need_perpixel_check = mask.IsEdgeTile(q_origin.at(number<0>{}),
                                                           k_origin.at(number<0>{}),
                                                           number<kM0>{},
                                                           number<kN0>{});

                if(need_perpixel_check)
                {
                    set_tile_if(
                        s_acc, -numeric<SMPLComputeDataType>::infinity(), [&](auto tile_idx) {
                            const auto row = q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                            const auto col = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                            return mask.IsOutOfBound(row, col);
                        });
                }
            }

            const auto s = cast_tile<SMPLComputeDataType>(s_acc); // S{j}
            auto m_local = block_tile_reduce<SMPLComputeDataType>(
                s,
                sequence<1>{},
                f_max,
                -numeric<SMPLComputeDataType>::infinity()); // m_local = rowmax(S{j})
            block_tile_reduce_sync(m_local, f_max, bool_constant<false>{});

            const auto m_old = m; // m{j-1}
            tile_elementwise_inout(
                [](auto& e0, auto e1, auto e2) { e0 = max(e1, e2); }, m, m_old, m_local); // m{j}

            auto p_compute = make_static_distributed_tensor<SMPLComputeDataType>(
                s.get_tile_distribution()); // Pcompute{j}

            __builtin_amdgcn_sched_barrier(0x7F);
            // store & prefetch next v, after the max reduction
            if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                auto v_shuffle_tmp = make_static_distributed_tensor<VDataType>(
                    Policy::template MakeShuffledVRegBlockDescriptor<Problem>());
                shuffle_tile(v_shuffle_tmp, v_buf);

                auto v_lds_window_tmp =
                    get_slice_tile(v_lds_window,
                                   sequence<(LdsSeq.at(number<k0_loops>{})) * kN1, 0>{},
                                   sequence<(LdsSeq.at(number<k0_loops>{}) + 1) * kN1, kK1>{});

                store_tile(
                    v_lds_window_tmp,
                    tile_elementwise_in(v_element_func, v_shuffle_tmp)); // store the prefetch
            }
            else
            {
                auto v_lds_window_tmp =
                    get_slice_tile(v_lds_window,
                                   sequence<(LdsSeq.at(number<k0_loops>{})) * kN1, 0>{},
                                   sequence<(LdsSeq.at(number<k0_loops>{}) + 1) * kN1, kK1>{});
                store_tile(v_lds_window_tmp,
                           tile_elementwise_in(v_element_func, v_buf)); // store the prefetch
            }

            if constexpr(k1_loops > 1)
            {
                move_tile_window(
                    v_dram_window,
                    {0, kK1}); // will have scratch if move this right after load_tile(v_dram)...
                v_buf = load_tile(v_dram_window, bool_constant<false>{}); // load next v_buf
            }
            __builtin_amdgcn_sched_barrier(0);

            static const auto get_validated_m = [](SMPLComputeDataType raw_m) {
                /// NOTICE: bias might be materialized mask including -inf values, need
                /// consideration. alibi does not have this problem
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             FmhaMask::IsMasking)
                {
                    return raw_m == -numeric<SMPLComputeDataType>::infinity()
                               ? type_convert<SMPLComputeDataType>(0.f)
                               : raw_m;
                }
                else
                {
                    return raw_m;
                }
            };

            constexpr auto p_spans = decltype(p_compute)::get_distributed_spans();
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                auto row_max = scale_s * get_validated_m(m[i_idx]);
#endif
                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                 BiasEnum == BlockAttentionBiasEnum::ALIBI)
                    {
                        p_compute(i_j_idx) = exp2(s[i_j_idx] - get_validated_m(m[i_idx]));
                    }
                    else
                    {
                        p_compute(i_j_idx) = exp2(scale_s * s[i_j_idx] - row_max);
                    }
#else
                    p_compute(i_j_idx)     = exp(s[i_j_idx] - get_validated_m(m[i_idx]));
#endif
                });
            });

            auto rowsum_p = block_tile_reduce<SMPLComputeDataType>(
                p_compute, sequence<1>{}, f_sum, SMPLComputeDataType{0}); // rowsum(Pcompute{j})

            block_tile_reduce_sync(rowsum_p, f_sum, bool_constant<false>{});
            // l{j}, Oacc{j}
            constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
            sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                const auto tmp = [&]() {
                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                 BiasEnum == BlockAttentionBiasEnum::ALIBI)
                    {
                        return exp2(m_old[i_idx] - get_validated_m(m[i_idx]));
                    }
                    else
                    {
                        auto row_max = scale_s * get_validated_m(m[i_idx]);
                        return exp2(scale_s * m_old[i_idx] - row_max);
                    }
                }();
#else
                const auto tmp = exp(m_old[i_idx] - get_validated_m(m[i_idx]));
#endif
                l(i_idx) = tmp * l[i_idx] + rowsum_p[i_idx];
                sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    // FIXME: this use different equation from FA v2 paper,
                    // but produce correc result.
                    // Is the equation wrong?
                    o_acc(i_j_idx) *= tmp;
                });
            });

            if constexpr(kHasDropout)
            {
                auto randval_ptr =
                    reinterpret_cast<char*>(smem_ptr) + Policy::template GetSmemSizeKV<Problem>();
                dropout.template Run<decltype(gemm_0), SMPLComputeDataType, RandValOutputDataType>(
                    randval_ptr,
                    seqlen_k_start + i_total_loops * kN0,
                    p_compute,
                    randval_dram_window);
            }

            const auto p = [&]() {
                if constexpr(std::is_same_v<PDataType, fp16_t>)
                    return impl::cast_tile_pk_fp16_fp32<PDataType>(
                        tile_elementwise_in(p_compute_element_func, p_compute));
                else
                    return cast_tile<PDataType>(
                        tile_elementwise_in(p_compute_element_func, p_compute));
            }();

            // STAGE 3, KV gemm
            if constexpr(k1_loops > 1)
            {
                static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
                    if constexpr(i_k1 != 0 && i_k1 < k1_loops - 1)
                    {
                        v_buf = load_tile(v_dram_window, bool_constant<false>{}); // load next v_buf
                    }
                    block_sync_lds();
                    gemm_1(o_acc,
                           get_slice_tile(
                               p, sequence<0, i_k1 * kK1>{}, sequence<kM0, (i_k1 + 1) * kK1>{}),
                           get_slice_tile(
                               v_lds_window,
                               sequence<(LdsSeq.at(number<k0_loops + i_k1>{})) * kN1, 0>{},
                               sequence<(LdsSeq.at(number<k0_loops + i_k1>{}) + 1) * kN1, kK1>{}));

                    if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
                    {
                        auto v_shuffle_tmp = make_static_distributed_tensor<VDataType>(
                            Policy::template MakeShuffledVRegBlockDescriptor<Problem>());
                        shuffle_tile(v_shuffle_tmp, v_buf);
                        auto v_lds_window_tmp = get_slice_tile(
                            v_lds_window,
                            sequence<(LdsSeq.at(number<k0_loops + i_k1 + 1>{})) * kN1, 0>{},
                            sequence<(LdsSeq.at(number<k0_loops + i_k1 + 1>{}) + 1) * kN1, kK1>{});
                        store_tile(v_lds_window_tmp,
                                   tile_elementwise_in(v_element_func,
                                                       v_shuffle_tmp)); // store the prefetch
                    }
                    else
                    {
                        auto v_lds_window_tmp = get_slice_tile(
                            v_lds_window,
                            sequence<(LdsSeq.at(number<k0_loops + i_k1 + 1>{})) * kN1, 0>{},
                            sequence<(LdsSeq.at(number<k0_loops + i_k1 + 1>{}) + 1) * kN1, kK1>{});
                        store_tile(v_lds_window_tmp,
                                   tile_elementwise_in(v_element_func, v_buf)); // store next v_buf
                    }
                    if constexpr(i_k1 < k1_loops - 1)
                        move_tile_window(v_dram_window, {0, kK1});
                });
            }
            i_total_loops++;
            if(i_total_loops < num_total_loop)
            {
                // move K tile windows
                move_tile_window(k_dram_block_window, {kN0, 0});
                k_dram_window.set_window_origin(k_dram_block_window.get_window_origin());

                if constexpr(k1_loops >= 2 &&
                             LdsSeq.at(number<0>{}) == LdsSeq.at(number<k0_loops + k1_loops - 2>{}))
                    __builtin_amdgcn_s_barrier();
                async_load_tile_raw(
                    k_lds_store(LdsSeq.at(number<0>{})), k_dram_window, k_oob_ck, k_pre_np);
                move_tile_window(k_dram_window, {0, kK0});
            }
            // tail
            {
                block_sync_lds();
                gemm_1(
                    o_acc,
                    get_slice_tile(p, sequence<0, (k1_loops - 1) * kK1>{}, sequence<kM0, kN0>{}),
                    get_slice_tile(
                        v_lds_window,
                        sequence<(LdsSeq.at(number<k0_loops + k1_loops - 1>{})) * kN1, 0>{},
                        sequence<(LdsSeq.at(number<k0_loops + k1_loops - 1>{}) + 1) * kN1, kK1>{}));
            }
        } while(i_total_loops < num_total_loop);

        // store lse
        if constexpr(kStoreLSE)
        {
            auto lse = make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

            constexpr auto lse_spans = decltype(lse)::get_distributed_spans();
            sweep_tile_span(lse_spans[number<0>{}], [&, m_ = m, l_ = l](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             BiasEnum == BlockAttentionBiasEnum::ALIBI)
                {
                    lse(i_idx) = m_[i_idx] * R_LOG2E + log(l_[i_idx]);
                }
                else
                {
                    lse(i_idx) = m_[i_idx] * scale_s * R_LOG2E + log(l_[i_idx]);
                }
#else
                lse(i_idx) = m_[i_idx] + log(l_[i_idx]);
#endif
            });

            store_tile(lse_dram_window_tmp, tile_elementwise_in(lse_element_func, lse));
        }

        // finally, O
        constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();

        sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            const auto tmp       = [&]() {
                if constexpr(FmhaMask::IsMasking)
                {
                    return l[i_idx] == 0.f ? 0.f : 1 / l[i_idx];
                }
                else
                    return 1 / l[i_idx];
            }();
            sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                o_acc(i_j_idx) *= tmp;
            });
        });

        o_acc = tile_elementwise_in(o_acc_element_func, o_acc);

        return o_acc;
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename RandValDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename PositionEncoding>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,       // M0*K0 tile
               const KDramBlockWindowTmp& k_dram_block_window_tmp,       // N0*K0 tile
               const VDramBlockWindowTmp& v_dram_block_window_tmp,       // N1*K1 tile
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
               RandValDramBlockWindowTmp& randval_dram_block_window_tmp, // M0*N0 tile
               LSEDramBlockWindowTmp& lse_dram_block_window_tmp,         // M0*1 tile
               FmhaMask mask,
               PositionEncoding position_encoding,
               float scale_s,
               void* smem_ptr,
               DropoutType& dropout) const
    {
        return operator()(q_dram_block_window_tmp,
                          identity{},
                          k_dram_block_window_tmp,
                          identity{},
                          v_dram_block_window_tmp,
                          identity{},
                          bias_dram_block_window_tmp,
                          identity{},
                          randval_dram_block_window_tmp,
                          lse_dram_block_window_tmp,
                          identity{},
                          identity{},
                          identity{},
                          identity{},
                          mask,
                          position_encoding,
                          scale_s,
                          smem_ptr,
                          dropout);
    }
};

} // namespace ck_tile
