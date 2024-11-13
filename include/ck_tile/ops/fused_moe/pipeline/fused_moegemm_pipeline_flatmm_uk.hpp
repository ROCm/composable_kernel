// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/fused_moe/pipeline/fused_moegemm_pipeline_flatmm_policy.hpp"

namespace ck_tile {

/*
This pipeline deal with a gemm(actually 2 gemm) with one very small(token), one very big(weight)
we need to design the pipeline such that all waves along gemm-N dim (gemm-m only 1 wave)

    <----- gemm-N ------>
    +----+----+----+----+
    | w0 | w1 | w2 | w3 | gemm-m
    +----+----+----+----+
*/
template <typename Problem_, typename Policy_ = FusedMoeGemmPipelineFlatmmPolicy>
struct FusedMoeGemmPipeline_FlatmmUk
{
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    using BlockShape = typename Problem::BlockShape; // this is FusedMoeGemmShape

    using ADataType            = typename Problem::ADataType;
    using GDataType            = typename Problem::GDataType;
    using DDataType            = typename Problem::DDataType;
    using AccDataType          = typename Problem::AccDataType;
    using ODataType            = typename Problem::ODataType;
    using AScaleDataType       = typename Problem::AScaleDataType;
    using GScaleDataType       = typename Problem::GScaleDataType;
    using DScaleDataType       = typename Problem::DScaleDataType;
    using YSmoothScaleDataType = typename Problem::YSmoothScaleDataType;
    using TopkWeightDataType   = typename Problem::TopkWeightDataType;
    using IndexDataType        = typename Problem::IndexDataType;
    using YDataType            = typename Problem::YDataType;

    using Traits = typename Problem::Traits;

    static constexpr bool IsGateOnly          = Traits::IsGateOnly;
    static constexpr bool UseSmoothQuant      = Traits::UseSmoothQuant;
    static constexpr bool PadHiddenSize       = Traits::PadHiddenSize;
    static constexpr bool PadIntermediateSize = Traits::PadIntermediateSize;

    static constexpr index_t kAlignmentA = Policy::template GetAlignment_A<Problem>();
    static constexpr index_t kAlignmentG = Policy::template GetAlignment_G<Problem>();
    static constexpr index_t kAlignmentD = Policy::template GetAlignment_D<Problem>();
    static constexpr index_t kAlignmentO = Policy::template GetAlignment_O<Problem>();

    static constexpr index_t SLD_A = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::SLD_A);
    static constexpr index_t GLD_A = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::GLD_A);
    static constexpr index_t GLD_B = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::GLD_B);
    static constexpr index_t GST_O = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::GST_O);

    static constexpr index_t kBlockPerCu = []() {
        if constexpr(Problem::kBlockPerCu != -1)
            return Problem::kBlockPerCu;
        else
        {
            // minimize occupancy
            return 2;
        }
    }();

    static constexpr const char* name = "flatmm_uk";

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        constexpr index_t smem_0 = Policy::template GetUK_1<Problem>().GetSmemSize();
        constexpr index_t smem_1 = Policy::template GetUK_1<Problem>().GetSmemSize();
        constexpr index_t smem_bridge =
            BlockShape::Block_M0 * BlockShape::Block_N0 * sizeof(YDataType);
        return max(smem_0, max(smem_1, smem_bridge));
    }

    // this is the thread-offset along row/col
    CK_TILE_HOST_DEVICE static auto GetACoord()
    {
        constexpr auto a_dist = Policy::template MakeGlobalTileDistribution_A<Problem>();
        const auto a_coord    = a_dist.calculate_index();
        return a_coord;
    }

    // this is the thread-offset along row/col
    CK_TILE_HOST_DEVICE static auto GetOCoord()
    {
        constexpr auto o_dist = Policy::template MakeOGlobalTileDistribution<Problem>();
        const auto o_coord    = o_dist.calculate_index();
        return o_coord;
    }

    CK_TILE_DEVICE constexpr auto GetNumRowCoords_A()
    {
        constexpr index_t KLans   = BlockShape::Block_K0 / kAlignmentA;
        constexpr index_t MLans   = BlockShape::BlockSize / KLans;
        constexpr index_t MRepeat = BlockShape::Block_M0 / MLans;

        return MRepeat;
    }

    // TODO: properlly support scatter/gather
    CK_TILE_DEVICE auto GetRowCoords_A(index_t base_offset)
    {
        constexpr index_t KLans   = BlockShape::Block_K0 / kAlignmentA;
        constexpr index_t MLans   = BlockShape::BlockSize / KLans;
        constexpr index_t MRepeat = BlockShape::Block_M0 / MLans;

        auto base_coord = threadIdx.x / KLans + base_offset;

        array<index_t, MRepeat> coords;
        static_for<0, MRepeat, 1>{}([&](auto i) { coords.at(i) = base_coord + i * MLans; });

        return coords;
    }

    template <typename ROW_COORDS>
    CK_TILE_DEVICE auto GetRowID_A(const ROW_COORDS coords,
                                   const IndexDataType* sorted_token_ids_ptr)
    {
        constexpr index_t n_size = coords.size();

        array<index_t, n_size> row_ids;
        static_for<0, n_size, 1>{}([&](auto i) {
            row_ids.at(i) = sorted_token_ids_ptr[coords[i]]; // base_coord + i * MLans;
        });

        return row_ids;
    }

    // TODO: properlly support scatter/gather
    CK_TILE_DEVICE auto GetRowCoords_O(index_t base_offset)
    {
        constexpr index_t WarpGemmLane_M   = 16; // TODO: use 16x16
        constexpr index_t WarpGemmRepeat_M = BlockShape::Block_M0 / WarpGemmLane_M;

        auto base_coord = threadIdx.x % WarpGemmLane_M + base_offset;

        array<index_t, WarpGemmRepeat_M> coords;
        static_for<0, WarpGemmRepeat_M, 1>{}(
            [&](auto i) { coords.at(i) = base_coord + i * WarpGemmLane_M; });

        return coords;
    }

    template <typename ROW_COORDS>
    CK_TILE_DEVICE auto GetWeightScale(const ROW_COORDS coords,
                                       const TopkWeightDataType* sorted_weight_ptr)
    {
        constexpr index_t n_size = coords.size();

        array<TopkWeightDataType, n_size> w;
        static_for<0, n_size, 1>{}([&](auto i) {
            w.at(i) = sorted_weight_ptr[coords[i]]; // base_coord + i * MLans;
        });

        return w;
    }

    CK_TILE_DEVICE auto GetRowCoords_O()
    {
        constexpr index_t NLans   = BlockShape::Block_N1 / kAlignmentA;
        constexpr index_t MLans   = BlockShape::BlockSize / NLans;
        constexpr index_t MRepeat = BlockShape::Block_M1 / MLans;

        auto base_coord = threadIdx.x / NLans;

        array<index_t, MRepeat> coords;
        static_for<0, MRepeat, 1>{}([&](auto i) { coords.at(i) = base_coord + i * MLans; });

        return coords;
    }
    /*
        struct FusedMoeGemmKargs
        {
            const void* a_ptr;              // [m, k], input token
            const void* a_scale_ptr;        // [m, 1], token scale
            const void* g_ptr;              // [e, n, k]/[e, 2*n, k], pre-shuffle([e, nr, kr, w])
            const void* d_ptr;              // [e, n, k], pre-shuffle([e, nr, kr, w])
            const void* g_scale_ptr;        // [e, 1, n], gate(up) scale
            const void* d_scale_ptr;        // [e, 1, k], down scale
            const void* y_smooth_scale_ptr; // [e, 1, n], smooth-quant-scale for 2nd gemm input
            void* o_ptr;                    // [m, k], output token

            const void* sorted_token_ids_ptr;
            const void* sorted_weight_ptr;
            const void* sorted_expert_ids_ptr;
            const void* num_sorted_tiles_ptr;

            index_t hidden_size;       // k
            index_t intermediate_size; // n (TP slice this)
            index_t num_tokens;        // input number of tokens for current iteration
            index_t num_experts;       // number of groups
            index_t topk;              // need this?

            index_t stride_token; // for input/output, stride for each row, should >= hidden_size
        };
    */
    template <typename Karg>
    CK_TILE_DEVICE auto operator()(const Karg& kargs,
                                   CK_TILE_LDS_ADDR void* smem,
                                   index_t sorted_tile_id,
                                   index_t intermediate_tile_id)
    {
        constexpr index_t hidden_radio_0            = IsGateOnly ? 1 : 2;
        ck_tile::index_t shared_intermediate_size_0 = kargs.intermediate_size;
        // w1 (Down, N size)
        ck_tile::index_t shared_intermediate_size_1 = kargs.intermediate_size / hidden_radio_0;

        index_t nr_0 = shared_intermediate_size_0 / BlockShape::Warp_N0; // divide N in W
        index_t kr_0 = kargs.hidden_size / BlockShape::Warp_K0;          // divide K in W
        index_t nr_1 = kargs.hidden_size / BlockShape::Warp_N1;
        index_t kr_1 = shared_intermediate_size_1 / BlockShape::Warp_K1;

        const IndexDataType expert_id = __builtin_amdgcn_readfirstlane(
            reinterpret_cast<const IndexDataType*>(kargs.sorted_expert_ids_ptr)[sorted_tile_id]);
        index_t expert_stride_0 = shared_intermediate_size_0 * kargs.hidden_size;
        index_t expert_stride_1 = shared_intermediate_size_1 * kargs.hidden_size;

        // nr*kr*w
        index_t interm_idx_nr = __builtin_amdgcn_readfirstlane(
            intermediate_tile_id *
            BlockShape::Block_Nr0); // intermediate_tile_id * Block_N / (N in W)

        // printf("bid:%d,%d, sorted_tile_id:%d(, intermediate_tile_id:%d, expert_id:%d,
        // interm_idx_nr:%d\n", static_cast<int>(blockIdx.x),
        //     static_cast<int>(blockIdx.y), sorted_tile_id, intermediate_tile_id, expert_id,
        //     interm_idx_nr);

        auto row_coords_a = GetRowCoords_A(sorted_tile_id * BlockShape::Block_M0);
        auto row_ids_a    = GetRowID_A(
            row_coords_a, reinterpret_cast<const IndexDataType*>(kargs.sorted_token_ids_ptr));
        auto a_coords = generate_tuple(
            [&](auto i) {
                return row_ids_a[i] * kargs.stride_token +
                       threadIdx.x % (BlockShape::Block_K0 / kAlignmentA) * kAlignmentA;
            },
            number<row_ids_a.size()>{});
        auto a_res =
            make_wave_buffer_resource(reinterpret_cast<const ADataType*>(kargs.a_ptr),
                                      kargs.num_tokens * kargs.stride_token * sizeof(ADataType));

        auto g_win = [&]() {
            const GDataType* g_ptr = reinterpret_cast<const GDataType*>(kargs.g_ptr) +
                                     static_cast<long_index_t>(expert_id) * expert_stride_0 +
                                     interm_idx_nr * kr_0 * BlockShape::Block_W0;
            auto g_view_ = make_naive_tensor_view<address_space_enum::global>(
                g_ptr,
                make_tuple(nr_0, kr_0, number<BlockShape::Block_W0>{}),
                make_tuple(kr_0 * BlockShape::Block_W0, number<BlockShape::Block_W0>{}, 1),
                number<kAlignmentG>{},
                number<1>{});

            // number<BlockShape::Block_Nr0>{}.fff();
            // number<kAlignmentG>{}.zzz();
            auto g_window_ = make_tile_window_linear_raw(
                g_view_,
                make_tuple(number<BlockShape::Block_Nr0>{},
                           number<BlockShape::Block_Kr0>{},
                           number<BlockShape::Block_W0>{}),
                {0, 0, 0},
                Policy::template MakeGlobalTileDistribution_G<Problem>(),
                sequence<0, 1, 1>{});
            return g_window_;
        }();
        // number<decltype(g_win)::NumAccess_NonLinear>{}.rrr2();
        auto g_res    = g_win.get_bottom_tensor_view().get_buffer_view().cached_buf_res_;
        auto g_coords = generate_tuple([&](auto i) { return g_win.cached_coords_[i].get_offset(); },
                                       number<decltype(g_win)::NumAccess_NonLinear>{});

        const auto d_win = [&]() {
            const DDataType* d_ptr = reinterpret_cast<const DDataType*>(kargs.d_ptr) +
                                     static_cast<long_index_t>(expert_id) * expert_stride_1 +
                                     interm_idx_nr * BlockShape::Block_W1;
            // note interm_idx_nr is along the gemm-k dim of 2nd gemm

            const auto d_view_ = make_naive_tensor_view<address_space_enum::global>(
                d_ptr,
                make_tuple(nr_1, kr_1, BlockShape::Block_W1),
                make_tuple(kr_1 * BlockShape::Block_W1, BlockShape::Block_W1, 1),
                number<kAlignmentD>{},
                number<1>{});

            const auto d_window_ = make_tile_window_linear_raw(
                d_view_,
                make_tuple(number<BlockShape::Block_Nr1>{},
                           number<BlockShape::Block_Kr1>{},
                           number<BlockShape::Block_W1>{}),
                {0, 0, 0},
                Policy::template MakeGlobalTileDistribution_D<Problem>(),
                sequence<0, 1, 1>{});
            return d_window_;
        }();
        auto d_res = d_win.get_bottom_tensor_view().get_buffer_view().cached_buf_res_;
#if 0
        auto d_coords     = generate_tuple([&](auto i) { 
            return d_win.cached_coords_[i].get_offset(); },
                                       number<decltype(d_win)::NumAccess_NonLinear>{});
#else
        // TODO: load D order is N0.K0...127, N64.K0...127, N0.K128...255, N64.K128...255
        //      block-k=512, block-n=128
        //                                    |<----- W_   ----->|
        //       Nr(2)*Nw(4)* Kr *Kr0(4)*Kr1(4) * [Kl(4)*Nl(16)*Kv(8)]->one issue
        //          y   p          y     y         p     p       y
        //          1              2     0(imm)
        auto d_coords = [&]() {
            constexpr index_t Nr_          = 2;
            constexpr index_t Nw_          = 4;
            constexpr index_t Kr0_         = 4;
            constexpr index_t Kr1_         = 4;
            constexpr index_t Kl_          = 4;
            constexpr index_t Nl_          = 16;
            constexpr index_t Kv_          = 8;
            constexpr index_t W_           = Kl_ * Nl_ * Kv_;
            constexpr index_t num_offsets_ = Nr_ * Kr0_;
            index_t base_os_ = (threadIdx.x % 64) * Kv_ + (threadIdx.x / 64) * Kr0_ * Kr1_ * W_;
            return generate_tuple(
                [&](auto i) {
                    constexpr auto i_nr_  = number<i % Nr_>{};
                    constexpr auto i_kr0_ = number<i / Nr_>{};

                    return i_nr_ * shared_intermediate_size_1 * Nw_ * Nl_ + i_kr0_ * Kr1_ * W_ +
                           base_os_;
                },
                number<num_offsets_>{});
        }();
#endif
        auto o_coords = generate_tuple(
            [&](auto i) {
                return row_ids_a[i] * kargs.stride_token +
                       threadIdx.x % (BlockShape::Block_N1 / kAlignmentO) * kAlignmentO;
            },
            number<row_ids_a.size()>{});

        auto o_flags =
            generate_tuple([&](auto i) { return cmp_lt_to_exec(row_ids_a[i], kargs.num_tokens); },
                           number<row_ids_a.size()>{});

        auto bridge_sst_win = [&]() {
            constexpr auto desc_ = Policy::template MakeBridgeLdsStoreForUKDesc<Problem>();
            constexpr auto dist_ = Policy::template GetUK_0<Problem>().MakeCBlockDist();
            return make_tile_window_linear(
                make_tensor_view<address_space_enum::lds>(
                    reinterpret_cast<YDataType*>(smem),
                    desc_),
                desc_.get_lengths(),
                {0, 0},
                dist_);
        }();
        auto o_res =
            make_wave_buffer_resource(reinterpret_cast<const ODataType*>(kargs.o_ptr),
                                      kargs.num_tokens * kargs.stride_token * sizeof(ODataType));

        auto row_coords_o = GetRowCoords_O(sorted_tile_id * BlockShape::Block_M0);
        auto w_scale      = GetWeightScale(
            row_coords_o, reinterpret_cast<const TopkWeightDataType*>(kargs.sorted_weight_ptr));
#if 0
        printf("bid:%d,%d, tid:%d, sorted_tile_id:%d(, intermediate_tile_id:%d, e:%d, "
               "interm_idx_nr:%d, coords:a:%d,%d,%d, row_ids_a:%d,%d,%d, (%d)g_coords:%d.%d.%d, "
               "o_coords:%d,%d,%d,%d,%d,%d,%d,%d(%d,%d,%d,%d,%d,%d,%d,%d)\n",
               static_cast<int>(blockIdx.x),
               static_cast<int>(blockIdx.y),
               static_cast<int>(threadIdx.x),
               sorted_tile_id,
               intermediate_tile_id,
               expert_id,
               interm_idx_nr,
               row_coords_a[0],
               row_coords_a[1],
               row_coords_a[7],
               row_ids_a[0],
               row_ids_a[1],
               row_ids_a[7],
               kr_0 * BlockShape::Block_W0,
               g_coords[number<0>{}],
               g_coords[number<1>{}],
               g_coords[number<7>{}],
               o_coords[number<0>{}],
               o_coords[number<1>{}],
               o_coords[number<2>{}],
               o_coords[number<3>{}],
               o_coords[number<4>{}],
               o_coords[number<5>{}],
               o_coords[number<6>{}],
               o_coords[number<7>{}],
               // (row_ids_a[0] >= kargs.num_tokens ? 1 : 0),
               // (row_ids_a[1] >= kargs.num_tokens ? 1 : 0),
               // (row_ids_a[2] >= kargs.num_tokens ? 1 : 0),
               // (row_ids_a[3] >= kargs.num_tokens ? 1 : 0),
               // (row_ids_a[4] >= kargs.num_tokens ? 1 : 0),
               // (row_ids_a[5] >= kargs.num_tokens ? 1 : 0),
               // (row_ids_a[6] >= kargs.num_tokens ? 1 : 0),
               // (row_ids_a[7] >= kargs.num_tokens ? 1 : 0)

               (row_ids_a[0] < kargs.num_tokens && static_cast<index_t>(o_coords[number<0>{}]) >=
                                                       (kargs.num_tokens * kargs.stride_token)
                    ? 7777
                    : 0),
               (row_ids_a[1] < kargs.num_tokens && static_cast<index_t>(o_coords[number<1>{}]) >=
                                                       (kargs.num_tokens * kargs.stride_token)
                    ? 7777
                    : 0),
               (row_ids_a[2] < kargs.num_tokens && static_cast<index_t>(o_coords[number<2>{}]) >=
                                                       (kargs.num_tokens * kargs.stride_token)
                    ? 7777
                    : 0),
               (row_ids_a[3] < kargs.num_tokens && static_cast<index_t>(o_coords[number<3>{}]) >=
                                                       (kargs.num_tokens * kargs.stride_token)
                    ? 7777
                    : 0),
               (row_ids_a[4] < kargs.num_tokens && static_cast<index_t>(o_coords[number<4>{}]) >=
                                                       (kargs.num_tokens * kargs.stride_token)
                    ? 7777
                    : 0),
               (row_ids_a[5] < kargs.num_tokens && static_cast<index_t>(o_coords[number<5>{}]) >=
                                                       (kargs.num_tokens * kargs.stride_token)
                    ? 7777
                    : 0),
               (row_ids_a[6] < kargs.num_tokens && static_cast<index_t>(o_coords[number<6>{}]) >=
                                                       (kargs.num_tokens * kargs.stride_token)
                    ? 7777
                    : 0),
               (row_ids_a[7] < kargs.num_tokens && static_cast<index_t>(o_coords[number<7>{}]) >=
                                                       (kargs.num_tokens * kargs.stride_token)
                    ? 7777
                    : 0)

        );
#endif
        auto uk_0  = Policy::template GetUK_0<Problem>();
        auto acc_0 = uk_0(a_res,
                          a_coords,
                          g_res,
                          g_coords,
                          smem,
                          kargs.hidden_size,
                          BlockShape::Block_K0, // tile offset for B matrix each unroll
                          BlockShape::Block_Kr0 *
                              BlockShape::Block_W0); // tile offset for B matrix each unroll

        // return ;
        //sweep_tile(acc_0,
        //           [&](auto idx) { typename Problem::GateActivation{}(acc_0(idx), acc_0[idx]); });
        sweep_tile(acc_0,
                   [&](auto idx0, auto idx1) {
                        fp32x2_t v_ {acc_0(idx0), acc_0(idx1)};
                        typename Problem::GateActivation{}(v_, v_);
                        acc_0(idx0) = v_.x;
                        acc_0(idx1) = v_.y;
                    },
                    sequence<1, 2>{});

#if 0
        printf("bid:%d,%d, tid:%d, sorted_tile_id:%d(, intermediate_tile_id:%d, e:%d, "
               "interm_idx_nr:%d, coords:a:%d,%d,%d, row_ids_a:%d,%d,%d, (%d)g_coords:%d.%d.%d, bridge_sst_win:%d"
               "acc:%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n",
               static_cast<int>(blockIdx.x),
               static_cast<int>(blockIdx.y),
               static_cast<int>(threadIdx.x),
               sorted_tile_id,
               intermediate_tile_id,
               expert_id,
               interm_idx_nr,
               row_coords_a[0],
               row_coords_a[1],
               row_coords_a[7],
               row_ids_a[0],
               row_ids_a[1],
               row_ids_a[7],
               kr_0 * BlockShape::Block_W0,
               g_coords[number<0>{}],
               g_coords[number<1>{}],
               g_coords[number<7>{}],
               bridge_sst_win.cached_coords_[number<0>{}].get_offset(),
                acc_0.get_thread_buffer()[number<0>{}],
                acc_0.get_thread_buffer()[number<1>{}],
                acc_0.get_thread_buffer()[number<2>{}],
                acc_0.get_thread_buffer()[number<3>{}],
                acc_0.get_thread_buffer()[number<4>{}],
                acc_0.get_thread_buffer()[number<5>{}],
                acc_0.get_thread_buffer()[number<6>{}],
                acc_0.get_thread_buffer()[number<7>{}],
                acc_0.get_thread_buffer()[number<8 + 0>{}],
                acc_0.get_thread_buffer()[number<8 + 1>{}],
                acc_0.get_thread_buffer()[number<8 + 2>{}],
                acc_0.get_thread_buffer()[number<8 + 3>{}],
                acc_0.get_thread_buffer()[number<8 + 4>{}],
                acc_0.get_thread_buffer()[number<8 + 5>{}],
                acc_0.get_thread_buffer()[number<8 + 6>{}],
                acc_0.get_thread_buffer()[number<8 + 7>{}]);
#endif

        auto y_pre = cast_tile<YDataType>(acc_0);
        store_tile(bridge_sst_win, y_pre);
        block_sync_lds();

        auto uk_1 = Policy::template GetUK_1<Problem>();
        uk_1(d_res,
             d_coords,
             o_res,
             o_coords,
             o_flags,
             smem,
             kargs.hidden_size, // total n number
             w_scale,
             BlockShape::Block_Nr1 * kr_1 * BlockShape::Block_W1, // along N
             BlockShape::Block_N1);                               // along N
    }
};

} // namespace ck_tile
