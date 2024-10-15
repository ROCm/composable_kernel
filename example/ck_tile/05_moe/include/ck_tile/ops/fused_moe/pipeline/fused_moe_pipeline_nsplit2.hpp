// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "fused_moe_pipeline_nsplit2_policy.hpp"
#include "fused_moe_pipeline_problem.hpp"
#include "fused_moe_tile_shape.hpp"
#include "fused_moe_traits.hpp"
#include "fused_moe_weight_permute_enum.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"


namespace ck_tile {

/*
This pipeline split the gemm-n of B matrix for less register pressure
(assume B matrix is much larger than A)
*/
template <typename Problem_, typename Policy_ = ck_tile::FusedMoePipelineNSplit2Policy>
struct FusedMoePipelineNSplit2
{
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    using ADataType     = remove_cvref_t<typename Problem::ADataType>;
    using GDataType     = remove_cvref_t<typename Problem::GDataType>;
    using UDataType     = remove_cvref_t<typename Problem::UDataType>;
    using DDataType     = remove_cvref_t<typename Problem::DDataType>;
    using ODataType     = remove_cvref_t<typename Problem::ODataType>;
    using AccDataType   = remove_cvref_t<typename Problem::AccDataType>;
    using YDataType   = remove_cvref_t<typename Problem::AccDataType>;
    using ScaleDataType = remove_cvref_t<typename Problem::ScaleDataType>;

    using FusedMoeTileShape = remove_cvref_t<typename Problem::FusedMoeTileShape>;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kBlockNSub_0   = FusedMoeTileShape::kBlockNSub_0;
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

    static_assert(kBlockN_0 == 2 * kBlockNSub_0); // this pipeline only support split2
    static_assert(kWarpRepeatN_0 % 2 == 0);

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

    using MBlockType_0 = decltype(Policy::template GetMatrixCoreSwizzledBlockTIle_0<Problem>());
    static constexpr index_t kBlockNr_0 = MBlockType_0::at(number<0>{});
    static constexpr index_t kBlockKr_0 = MBlockType_0::at(number<1>{});
    static constexpr index_t kBlockWaveFlatten = MBlockType_0::at(number<2>{});
    static_assert(kBlockNr_0 % 2 == 0);
    static constexpr index_t kBlockSubNr_0 = kBlockNr_0 / 2;

    using MBlockType_1 = decltype(Policy::template GetMatrixCoreSwizzledBlockTIle_1<Problem>());
    static constexpr index_t kBlockNr_1 = MBlockType_1::at(number<0>{});
    static constexpr index_t kBlockKr_1 = MBlockType_1::at(number<1>{});
    static constexpr index_t kBlockSubKr_1 = kBlockKr_1 / 2;
    static_assert(kBlockSubNr_0 == kBlockSubKr_1);

    static constexpr index_t kAlignmentA = Policy::template GetAlignment_A<Problem>();
    static constexpr index_t kAlignmentG = Policy::template GetAlignment_G<Problem>();
    static constexpr index_t kAlignmentU = Policy::template GetAlignment_U<Problem>();
    static constexpr index_t kAlignmentD = Policy::template GetAlignment_D<Problem>();
    static constexpr index_t kAlignmentO = Policy::template GetAlignment_O<Problem>();

    static constexpr index_t kBlockPerCu = []() {
        if constexpr(Problem::kBlockPerCu != -1)
            return Problem::kBlockPerCu;
        else
        {
            // minimize occupancy
            return 2;
        }
    }();

    static constexpr const char* name = "fused_moe_ns2";

   // using DropoutType = std::conditional_t<kHasDropout, BlockDropout, NullBlockDropout>;

    // TODO: there are multiple buffers
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeSingleBuffer()
    {
        return Policy::template GetSmemSizeSingleBuffer();
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
                                   //ScaleDataType scale,
                                   CK_TILE_LDS_ADDR void* smem_0,
                                   CK_TILE_LDS_ADDR void* smem_1,
                                   index_t dim_size,
                                   index_t /*hidden_size*/)
    {
        constexpr auto I0 = number<0>{};
        constexpr auto I1 = number<1>{};

        auto a_win = make_tile_window(a_gtile_window_tmp.get_bottom_tensor_view(),
                                      a_gtile_window_tmp.get_window_lengths(),
                                      a_gtile_window_tmp.get_window_origin(),
                                      Policy::template MakeGlobalTileDistribution_A<Problem>());

        auto g_win = generate_tuple(
            [&](auto i) {
                return make_tile_window(g_gtile_window_tmp.get_bottom_tensor_view(),
                                        make_tuple(number<kBlockSubNr_0>{},
                                                   number<kBlockKr_0>{},
                                                   number<kBlockWaveFlatten>{}),
                                        {number<kBlockSubNr_0 * i>{}, I0, I0},
                                        Policy::template MakeGlobalTileDistribution_G<Problem>());
            },
            number<2>{});

        auto u_win = generate_tuple(
            [&](auto i) {
                return make_tile_window(u_gtile_window_tmp.get_bottom_tensor_view(),
                                        make_tuple(number<kBlockSubNr_0>{},
                                                   number<kBlockKr_0>{},
                                                   number<kBlockWaveFlatten>{}),
                                        {number<kBlockSubNr_0 * i>{}, I0, I0},
                                        Policy::template MakeGlobalTileDistribution_U<Problem>());
            },
            number<2>{});

        auto d_win = generate_tuple(
            [&](auto i) {
                return make_tile_window(d_gtile_window_tmp.get_bottom_tensor_view(),
                                        make_tuple(number<kBlockNr_1>{},
                                                   number<kBlockSubKr_1>{},
                                                   number<kBlockWaveFlatten>{}),
                                        {I0, number<kBlockSubKr_1 * i>{}, I0},
                                        Policy::template MakeGlobalTileDistribution_U<Problem>());
            },
            number<2>{});

        make_tile_window(d_gtile_window_tmp.get_bottom_tensor_view(),
                         d_gtile_window_tmp.get_window_lengths(),
                         d_gtile_window_tmp.get_window_origin(),
                         Policy::template MakeGlobalTileDistribution_D<Problem>());

        auto o_win = make_tile_window(o_gtile_window_tmp.get_bottom_tensor_view(),
                                      o_gtile_window_tmp.get_window_lengths(),
                                      o_gtile_window_tmp.get_window_origin(),
                                      Policy::template MakeOGlobalTileDistribution<Problem>());

        using g_thread_type = decltype(load_tile(g_win[I0]));
        using u_thread_type = decltype(load_tile(u_win[I0]));
        using d_thread_type = decltype(load_tile(d_win[I0]));

        const index_t loops_0 = (dim_size + kBlockK_0 - 1) / kBlockK_0;
        const index_t loops_1 = (dim_size + kBlockN_1 - 1) / kBlockN_1;

        // issues_warps_lanes
        auto a_st0 = make_tile_window(make_tensor_view<address_space_enum::lds>(
                                          smem_0, Policy::template MakeLdsStoreDesc_A<Problem>()),
                                      Policy::template MakeLdsStoreDesc_A<Problem>().get_lengths(),
                                      {0, 0, 0});

        // issues_warps_lanes
        auto a_st1 = make_tile_window(make_tensor_view<address_space_enum::lds>(
                                          smem_1, Policy::template MakeLdsStoreDesc_A<Problem>()),
                                      Policy::template MakeLdsStoreDesc_A<Problem>().get_lengths(),
                                      {0, 0, 0});

        // m*k
        auto a_ld0 = make_tile_window(make_tensor_view<address_space_enum::lds>(
                                          smem_0, Policy::template MakeLdsLoadDesc_A<Problem>()),
                                      Policy::template MakeLdsLoadDesc_A<Problem>().get_lengths(),
                                      {0, 0});

        // m*k
        auto a_ld1 = make_tile_window(make_tensor_view<address_space_enum::lds>(
                                          smem_1, Policy::template MakeLdsLoadDesc_A<Problem>()),
                                      Policy::template MakeLdsLoadDesc_A<Problem>().get_lengths(),
                                      {0, 0});

        statically_indexed_array<g_thread_type, 2> g_tls;
        statically_indexed_array<u_thread_type, 2> u_tls;
        using WarpGemm0  = remove_cvref_t<decltype(Policy::template GetWarpGemm0<Problem>())>;
        using WarpGemm1  = remove_cvref_t<decltype(Policy::template GetWarpGemm1<Problem>())>;
        auto warp_gemm_0 = WarpGemm0{};
        auto warp_gemm_1 = WarpGemm1{};

        // TODO: N fist, M next

        // create and pre-cache a_reg warp-window
        auto make_a_warp_windows = [&](auto a_sld_) {
            const index_t i_mwarp_0 = get_warp_id() / kBlockWarpsN_0;
            // construct A-warp-window
            auto warp_window = make_tile_window(
                a_sld_.get_bottom_tensor_view(),
                make_tuple(number<WarpGemm0::kM>{}, number<WarpGemm0::kK>{}),
                a_sld_.get_window_origin() + multi_index<2>{i_mwarp_0 * WarpGemm0::kM, 0},
                make_static_tile_distribution(typename WarpGemm0::AWarpDstrEncoding{}));
            return warp_window;
        };

        auto a_warp_windows_0 = make_a_warp_windows(a_ld0);
        auto a_warp_windows_1 = make_a_warp_windows(a_ld1);

        auto load_a = [&](auto& a_store_) {
            async_load_tile(a_store_, a_win);
            move_tile_window(a_win, {number<0>{}, number<kBlockK_0>{}});
        };

        auto load_n = [&](auto& g_tile_, auto& u_tile_, auto& g_window_, auto& u_window_) {
            g_tile_ = load_tile(g_window_);
            u_tile_ = load_tile(u_window_);
            move_tile_window(g_window_, {number<0>{}, number<kBlockKr_0>{}, number<0>{}});
            move_tile_window(u_window_, {number<0>{}, number<kBlockKr_0>{}, number<0>{}});
        };

        auto load_d = [&](auto& d_tile_) {
            d_tile_ = load_tile(d_win);
            move_tile_window(d_win, {number<0>{}, number<kBlockKr_0>{}, number<0>{}});
        };

        auto acc_g = generate_tuple([&](auto) {Policy::template MakeCBlockTile_Gemm0<Problem>(); }, number<2>{});
        auto acc_u = generate_tuple([&](auto) {Policy::template MakeCBlockTile_Gemm0<Problem>(); }, number<2>{});

        // Note this function only do gemm of single Nsplit
        // clang-format off
        auto gemm_0 = [&](auto& acc_g_, auto& acc_u_, auto& a_, auto& g_, auto& u_) {
            static_for<0, kWarpRepeatK_0, 1>{}([&](auto i_k) {
                static_for<0, kWarpRepeatM_0, 1>{}([&](auto i_m) {
                    constexpr auto beg_a = sequence<i_m * kSubBlockM_0, i_k * kSubBlockK_0 >{};
                    constexpr auto end_a = sequence<(i_m+1) * kSubBlockM_0, (i_k+1) * kSubBlockK_0 >{};
                    auto w_a = get_slice_tile(a_, beg_a, end_a);

                    static_for<0, kWarpRepeatN_0 / 2, 1>{}([&](auto i_n) {
                        constexpr auto beg_acc = sequence<i_m * kSubBlockM_0, i_n * kSubBlockN_0>{};
                        constexpr auto end_acc = sequence<(i_m + 1) * kSubBlockM_0, (i_n + 1) * kSubBlockN_0>{};
                        constexpr auto beg_b = sequence<i_n * kSubBlockN_0, i_k * kSubBlockK_0, 0>{};
                        constexpr auto end_b = sequence<(i_n + 1) * kSubBlockN_0, (i_k + 1) * kSubBlockK_0, 0>{};

                        auto w_acc_g = get_slice_tile(acc_g_, beg_acc, end_acc);
                        auto w_acc_u = get_slice_tile(acc_u_, beg_acc, end_acc);
                        auto w_g     = get_slice_tile(g_, beg_b, end_b);
                        auto w_u     = get_slice_tile(u_, beg_b, end_b);

                        warp_gemm_0(w_acc_g, w_a, w_g);
                        warp_gemm_0(w_acc_u, w_a, w_u);

                        set_slice_tile(acc_g_, w_acc_g, beg_acc, end_acc);
                        set_slice_tile(acc_u_, w_acc_u, beg_acc, end_acc);
                    });
                });
            });
        };
        // clang-format on

        // clang-format off
        auto gemm_1 = [&](auto& acc_d_, auto& y_, auto& d_) {
            static_for<0, kWarpRepeatK_1, 1>{}([&](auto i_k) {
                static_for<0, kWarpRepeatM_1, 1>{}([&](auto i_m) {
                    constexpr auto beg_a = sequence<i_m * kSubBlockM_1, i_k * kSubBlockK_1>{};
                    constexpr auto end_a = sequence<(i_m + 1) * kSubBlockM_1, (i_k + 1) * kSubBlockK_1>{};
                    const auto w_y = get_slice_tile(y_, beg_a, end_a);

                    static_for<0, kWarpRepeatN_1, 1>{}([&](auto i_n) {
                        constexpr auto beg_acc = sequence<i_m * kSubBlockM_1, i_n * kSubBlockN_1>{};
                        constexpr auto end_acc = sequence<(i_m + 1) * kSubBlockM_1, (i_n + 1) * kSubBlockN_1>{};

                        constexpr auto beg_d = sequence<i_n * kSubBlockN_1, i_k * kSubBlockK_1, 0>{};
                        constexpr auto end_d = sequence<(i_n + 1) * kSubBlockN_1, (i_k + 1) * kSubBlockK_1, 0>{};

                        auto w_acc_d = get_slice_tile(acc_d_, beg_acc, end_acc);
                        auto w_d     = get_slice_tile(d_, beg_d, end_d);

                        warp_gemm_1(w_acc_d, w_y, w_d);

                        set_slice_tile(acc_d_, w_acc_d, beg_acc, end_acc);
                    });
                });
            });
        };
        // clang-format on

        constexpr auto issues_a = number<a_win.get_num_of_access()>{};
        constexpr auto issues_g = number<g_win[I0].get_num_of_access()>{};
        constexpr auto issues_u = number<u_win[I0].get_num_of_access()>{};
        constexpr auto issues_b = issues_g + issues_u;
        constexpr auto issues_d = number<d_win[I0].get_num_of_access()>{};
        constexpr auto issues_o = number<o_win.get_num_of_access()>{};

        // start of pipeline
        // clang-format off
        load_a(a_st0);
        load_n(g_tls[I0], u_tls[I0], g_win[I0], u_win[I0]);
        load_n(g_tls[I1], u_tls[I1], g_win[I1], u_win[I1]);
        load_a(a_st1);

        clear_tile(acc_g[I0]); clear_tile(acc_g[I1]); clear_tile(acc_u[I0]); clear_tile(acc_u[I1]);

        auto a_reg = decltype(load_tile(a_warp_windows_0)){};
        index_t i_0 = 0;
        while(i_0 < (loops_0 - 2))
        {
            // first buffer
            buffer_load_fence(issues_b + issues_b + issues_a);
            wave_barrier();   a_reg = load_tile(a_warp_windows_0);

            buffer_load_fence(issues_b + issues_a);
            gemm_0(acc_g[I0], acc_u[I0], a_reg, g_tls[I0], u_tls[I0]);
            load_n(g_tls[I0], u_tls[I0], g_win[I0], u_win[I0]);

            buffer_load_fence(issues_b + issues_a);
            gemm_0(acc_g[I1], acc_u[I1], a_reg, g_tls[I1], u_tls[I1]);
            load_n(g_tls[I1], u_tls[I1], g_win[I1], u_win[I1]);
            load_a(a_st0);
            i_0++;

            // second buffer
            buffer_load_fence(issues_b + issues_b + issues_a);
            wave_barrier();   a_reg = load_tile(a_warp_windows_1);

            buffer_load_fence(issues_b + issues_a);
            gemm_0(acc_g[I0], acc_u[I0], a_reg, g_tls[I0], u_tls[I0]);
            load_n(g_tls[I0], u_tls[I0], g_win[I0], u_win[I0]);

            buffer_load_fence(issues_b + issues_a);
            gemm_0(acc_g[I1], acc_u[I1], a_reg, g_tls[I1], u_tls[I1]);
            load_n(g_tls[I1], u_tls[I1], g_win[I1], u_win[I1]);
            load_a(a_st1);
            i_0++;
        }

        // first buffer
        buffer_load_fence(issues_b + issues_b + issues_a);
        wave_barrier();   a_reg = load_tile(a_warp_windows_0);
        gemm_0(acc_g[I0], acc_u[I0], a_reg, g_tls[I0], u_tls[I0]);
        load_n(g_tls[I0], u_tls[I0], g_win[I0], u_win[I0]);

        buffer_load_fence(issues_b + issues_a);
        gemm_0(acc_g[I1], acc_u[I1], a_reg, g_tls[I1], u_tls[I1]);
        load_n(g_tls[I1], u_tls[I1], g_win[I1], u_win[I1]);

        // second buffer
        buffer_load_fence(issues_b + issues_b);
        wave_barrier();   a_reg = load_tile(a_warp_windows_1);

        buffer_load_fence(issues_b);
        gemm_0(acc_g[I0], acc_u[I0], a_reg, g_tls[I0], u_tls[I0]);

        // prefetch
        statically_indexed_array<d_thread_type, 2> d_tls;
        load_d(d_tls[0]);   load_d(d_tls[1]);
        buffer_load_fence(issues_d + issues_d);
        gemm_0(acc_g[I1], acc_u[I1], a_reg, g_tls[I1], u_tls[I1]);

        // redice acc_g/u
        constexpr auto acc_spans_0 = decltype(acc_g)::get_distributed_spans();
        sweep_tile_span(acc_spans_0[number<0>{}], [&](auto idx0) {
            sweep_tile_span(acc_spans_0[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                ck::tensor_operation::element_wise::Silu{}(acc_g[I0](i_j_idx), acc_g[I0](i_j_idx));
                ck::tensor_operation::element_wise::Silu{}(acc_g[I1](i_j_idx), acc_g[I1](i_j_idx));
                acc_g[I0](i_j_idx) *= acc_u[I0](i_j_idx);
                acc_g[I1](i_j_idx) *= acc_u[I1](i_j_idx);
            });
        });

        const auto y_reg = generate_tuple([&](auto i) {
            if constexpr(std::is_same_v<YDataType, fp16_t>) return impl::cast_tile_pk_fp16_fp32<YDataType>(acc_g[i]);
            else return cast_tile<YDataType>(acc_g[i]); }, number<2>{});

        auto acc_d = Policy::template MakeCBlockTile_Gemm1<Problem>();
        // TODO: reshuffle? 32x32x8 mfma can avlid LDS reshuffle

        // Second gemm
        clear_tile(acc_d);
        // first buffer
        buffer_load_fence(issues_d);
        gemm_1(acc_d, y_reg[I0], d_tls[I0]);    load_d(d_tls[I0]);

        // second buffer
        buffer_load_fence(issues_d);
        gemm_1(acc_d, y_reg[I1], d_tls[I1]);    load_d(d_tls[I1]);
        update_tile(o_win, acc_d);

        index_t i_1 = 0;
        while(i_1 < (loops_1 - 2))
        {
            clear_tile(acc_d);
            // first buffer
            buffer_load_fence(issues_d + issues_o);
            gemm_1(acc_d, y_reg[I0], d_tls[I0]);    load_d(d_tls[I0]);

            // second buffer
            buffer_load_fence(issues_d + issues_o);
            gemm_1(acc_d, y_reg[I1], d_tls[I1]);    load_d(d_tls[I1]);
            update_tile(o_win, acc_d);
            i_1++;
        }

        clear_tile(acc_d);
        // first buffer
        buffer_load_fence(issues_d + issues_o);
        gemm_1(acc_d, y_reg[I0], d_tls[I0]);

        // second buffer
        buffer_load_fence(issues_o);
        gemm_1(acc_d, y_reg[I1], d_tls[I1]);
        update_tile(o_win, acc_d);
        // clang-format on
    }
};

} // namespace ck_tile
