// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_fmha_lse.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_pingpong.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_kvshared_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v2.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v2_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_problem.hpp"
#include "ck_tile/ops/gemm/warp/warp_wmma_gemm.hpp"

namespace ck_tile {

// qr_tdm_pingpong's TDM staging for K and V that are one tensor (the caller passes the same
// K and V views): each K/V tile is staged once, by one TDM transfer into one of two LDS
// buffers, and serves both gemms. Trip i reads buffer i & 1 while the TDM fills the other
// with tile i + 1.
//
// Each warp (one or two) takes two 16-key blocks of every tile (gemm_0's: warp w blocks w and
// w + kNWarp) and runs an online softmax of its own over them. gemm_0 (S = Q K^T, the warps
// on the keys) and gemm_1 (O_w = P_w V_w, one per warp over its keys) both run the
// transposed-C WMMA, which computes S^T = K Q^T and O_w^T = V_w^T P_w^T: a lane of S holds
// one query and 8 keys of each block, which is the layout gemm_1 takes P in, so P goes from
// the exponent straight into gemm_1, with no S round trip through LDS and each score
// exponentiated once; V^T is read transposed from LDS, block by block. Two warps meet only at
// the K/V buffer barrier, and at the end, where their (m, l, O) are merged through LDS. The V
// view is not read.
template <typename Problem_, typename Policy_ = BlockFmhaPipelineQRKSVSTdmKVSharedPolicy>
struct BlockFmhaPipelineQRKSVSTdmKVShared : BlockFmhaPipelineQRKSVSTdmPingPong<Problem_, Policy_>
{
    using PingPong = BlockFmhaPipelineQRKSVSTdmPingPong<Problem_, Policy_>;

    using typename PingPong::BlockFmhaShape;
    using typename PingPong::FmhaMask;
    using typename PingPong::KDataType;
    using typename PingPong::LSEDataType;
    using typename PingPong::OaccDataType;
    using typename PingPong::PDataType;
    using typename PingPong::Policy;
    using typename PingPong::Problem;
    using typename PingPong::QDataType;
    using typename PingPong::SaccDataType;
    using typename PingPong::SMPLComputeDataType;
    using typename PingPong::VDataType;

    using PingPong::BiasEnum;
    using PingPong::I0;
    using PingPong::I1;
    using PingPong::kHasLogitsSoftCap;
    using PingPong::kHasSink;
    using PingPong::kHasUnevenSplits;
    using PingPong::kK0;
    using PingPong::kM0;
    using PingPong::kN0;
    using PingPong::kN1;
    using PingPong::kNWarp;
    using PingPong::kPadSeqLenK;
    using PingPong::kStoreLSE;
    using PingPong::kSubQKHeaddim;
    using PingPong::ReduceRowSync;

    static constexpr const char* name = "qr_tdm_kvshared";

    static_assert(kN1 == kSubQKHeaddim && std::is_same_v<KDataType, VDataType> &&
                      std::is_same_v<QDataType, KDataType> && std::is_same_v<PDataType, KDataType>,
                  "qr_tdm_kvshared: Q, K, V and P of one head dim and data type");
    static_assert(std::is_same_v<KDataType, half_t> || std::is_same_v<KDataType, bf16_t>,
                  "qr_tdm_kvshared: fp16 or bf16");
    static_assert(kM0 == 16 && (kNWarp == 1 || kNWarp == 2) && kN0 == kNWarp * 32 && kK0 == 32 &&
                      BlockFmhaShape::Gemm0BlockWarps::at(I0) == 1 &&
                      BlockFmhaShape::NumWarps == kNWarp && kSubQKHeaddim % 32 == 0,
                  "qr_tdm_kvshared: 16 queries, one or two warps of 32 keys each");
    static_assert(!kHasLogitsSoftCap, "qr_tdm_kvshared: no logits soft-cap");
    static_assert(PingPong::kIsSoftmax, "qr_tdm_kvshared: softmax only (no GDPA mode)");
    static_assert(std::is_same_v<SMPLComputeDataType, float> &&
                  std::is_same_v<SaccDataType, float> && std::is_same_v<OaccDataType, float>);

    static constexpr index_t kKeysPerWarp = kN0 / kNWarp;      // 32, one gemm_1 K step
    static constexpr index_t kKeyBlocks   = kKeysPerWarp / 16; // a warp's 16-key blocks
    static constexpr index_t kOBlocks     = kN1 / 16;          // gemm_1 N steps (16 d each)
    static constexpr index_t kOHalf       = kOBlocks / kNWarp; // O blocks a warp stores
    static constexpr index_t k0_loops     = kSubQKHeaddim / kK0;

    // The gfx1250 16x16x32 WMMA with the transposed C, on both gemms (named, not dispatched:
    // the pipeline is gfx1250's, and the host pass must see the same layouts). With two
    // 8-element accesses per operand (EDouble) a lane's K slots are 0..7 and 16..23 (8..15
    // and 24..31 in the upper lanes), the LDS reads of qr_tdm's other gemms.
    static constexpr bool kIsFp16 = std::is_same_v<KDataType, half_t>;
    using WarpGemm =
        std::conditional_t<kIsFp16,
                           WarpGemmWmma_f32_16x16x32_f16_f16<true, WGAttrNumAccessEnum::Double>,
                           WarpGemmWmma_f32_16x16x32_bf16_bf16<true, WGAttrNumAccessEnum::Double>>;
    static_assert(WarpGemm::kM == 16 && WarpGemm::kN == 16 && WarpGemm::kK == kKeysPerWarp);

    // gemm_0, S = Q K^T, kK0 of the head dim at a time, the warps on the keys: a lane of S
    // holds one query and, per 16-key block, 8 keys (WMMA's C is S^T = K Q^T)
    using QKBlockGemm = BlockGemmARegBRegCRegV2<
        BlockGemmProblem<QDataType,
                         KDataType,
                         SaccDataType,
                         Problem::kBlockSize,
                         TileGemmShape<sequence<kM0, kN0, kK0>,
                                       typename BlockFmhaShape::Gemm0BlockWarps,
                                       sequence<16, 16, 32>>>,
        BlockGemmARegBRegCRegV2CustomPolicy<QDataType,
                                            KDataType,
                                            SaccDataType,
                                            typename BlockFmhaShape::Gemm0BlockWarps,
                                            WarpGemm,
                                            GemmLoopOrder::MNK>>;

    // gemm_0 over each warp's first 16-key block only (the trip whose other block lies past
    // seqlen_k): N = 16 keys per warp, which are S's first column slice
    using QKHalfBlockGemm = BlockGemmARegBRegCRegV2<
        BlockGemmProblem<QDataType,
                         KDataType,
                         SaccDataType,
                         Problem::kBlockSize,
                         TileGemmShape<sequence<kM0, kNWarp * 16, kK0>,
                                       typename BlockFmhaShape::Gemm0BlockWarps,
                                       sequence<16, 16, 32>>>,
        BlockGemmARegBRegCRegV2CustomPolicy<QDataType,
                                            KDataType,
                                            SaccDataType,
                                            typename BlockFmhaShape::Gemm0BlockWarps,
                                            WarpGemm,
                                            GemmLoopOrder::MNK>>;
    static_assert(kKeyBlocks == 2 &&
                      std::is_same_v<decltype(QKHalfBlockGemm::MakeABlockDistributionEncode()),
                                     decltype(QKBlockGemm::MakeABlockDistributionEncode())>,
                  "the half gemm takes Q as the whole one does");

    // gemm_1 of one warp over its 32 keys, O_w = P_w V_w. Its P_w operand puts a lane's keys
    // where S has them (K slots 16 b + 0..15: the warp's block b), so P_w is S's registers.
    using PVWarpGemm  = WarpGemm;
    using PWarpTensor = typename PVWarpGemm::AWarpTensor;
    using OWarpTensor = typename PVWarpGemm::CWarpTensor;

    // V^T, gemm_1's B operand (16 d by 32 keys), one 16-key block at a time, read transposed
    // from the row-major K/V tile
    using VWarpTensor = typename PVWarpGemm::BWarpTensor;
    using VBlockTile =
        decltype(get_slice_tile(VWarpTensor{}, sequence<0, 0>{}, sequence<16, 16>{}));
    using VTransposedLoadEncoding = typename InputTileDistributionTraits<
        typename VBlockTile::StaticTileDistribution::DstrEncode,
        VDataType>::TransposedDstrEncode;

    // A (query, column) tile in the transposed-C WMMA's C layout: query = lane % 16; column =
    // warp, 16-column block, lane / 16, then 8 columns in registers.
    template <index_t kCols>
    CK_TILE_HOST_DEVICE static constexpr auto MakeWarpLocalDistribution()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<kM0, 1>, sequence<kNWarp, kCols / (kNWarp * 16), 2, 8>>,
                tuple<sequence<2>, sequence<2, 1>>,
                tuple<sequence<0>, sequence<2, 0>>,
                sequence<1, 2, 2>,
                sequence<1, 1, 3>>{});
    }

    // O as the merge leaves it: warp w holds O blocks [w, w + 1) * kOHalf, a lane 8
    // consecutive d of one query (the epilogue's LDS shuffle then stores whole rows)
    CK_TILE_HOST_DEVICE static constexpr auto MakeODistribution()
    {
        return MakeWarpLocalDistribution<kN1>();
    }

    // The LSE of the 16 queries: query = lane % 16, the same in both warps and lane halves
    CK_TILE_HOST_DEVICE static constexpr auto MakeLseDistribution()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<kNWarp, 2>,
                                       tuple<sequence<kM0, 1>>,
                                       tuple<sequence<0>, sequence<0, 1>>,
                                       tuple<sequence<0>, sequence<1, 0>>,
                                       sequence<1>,
                                       sequence<1>>{});
    }

    // A warp's per-query value (m or l): query = lane % 16, the lane halves alike
    CK_TILE_HOST_DEVICE static constexpr auto MakeWarpRowDistribution()
    {
        return make_static_tile_distribution(tile_distribution_encoding<sequence<2>,
                                                                        tuple<sequence<kM0, 1>>,
                                                                        tuple<sequence<0, 1>>,
                                                                        tuple<sequence<0, 0>>,
                                                                        sequence<1>,
                                                                        sequence<1>>{});
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename LSEaccDramBlockWindowTmp,
              typename PositionEncoding>
    CK_TILE_HOST_DEVICE auto operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,
                                        const KDramBlockWindowTmp& k_dram_block_window_tmp,
                                        const VDramBlockWindowTmp& /* v: the K view */,
                                        const BiasDramBlockWindowTmp& bias_dram_block_window_tmp,
                                        LSEaccDramBlockWindowTmp& lse_acc_dram_window_tmp,
                                        FmhaMask mask,
                                        PositionEncoding position_encoding,
                                        float scale_s,
                                        void* smem_arena,
                                        float sink_v,
                                        bool lse_log2 = false) const
    {
        using Layout = typename Policy::template LdsArenaLayout<Problem>;

        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>>,
            "wrong!");
        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kSubQKHeaddim == QDramBlockWindowTmp{}.get_window_lengths()[I1] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kM0 == BiasDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kN0 == BiasDramBlockWindowTmp{}.get_window_lengths()[I1],
                      "wrong!");

        constexpr auto gemm_0   = QKBlockGemm{};
        using SaccBlockTileType = decltype(gemm_0.MakeCBlockTile());
        static_assert(SaccBlockTileType::get_thread_buffer_size() ==
                          PWarpTensor::get_thread_buffer_size(),
                      "qr_tdm_kvshared: a lane of S holds what it gives gemm_1 of P");

        auto o_acc = make_static_distributed_tensor<OaccDataType>(MakeODistribution());

        // With bias or ALiBi the scores come scaled (ApplyScaleAndBias); otherwise scale_s
        // goes into the exponent.
        constexpr bool kScaledScores = BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                       BiasEnum == BlockAttentionBiasEnum::ALIBI;
        const float exp_scale = kScaledScores ? 1.f : scale_s;

        // A finite sink seeds m (log2 domain) and l, as qr_tdm_pingpong's InitRowMaxSum
        const bool has_sink_logit = __builtin_isinf_sign(sink_v) >= 0;
#if CK_TILE_FMHA_FWD_FAST_EXP2
        const float sink_m = sink_v * (kScaledScores ? scale_s : 1.f) * log2e_v<float>;
#else
        const float sink_m = sink_v;
#endif

        const auto q_origin = q_dram_block_window_tmp.get_window_origin();

        const auto tile_range_result      = PingPong::GetKTileRange(mask, q_origin);
        const auto sink_seq_end           = tile_range_result.get(number<0>{});
        const auto logical_seqlen_k_start = tile_range_result.get(number<1>{});
        const auto logical_seqlen_k_end   = tile_range_result.get(number<2>{});

        const auto num_sink_loop = integer_divide_ceil(sink_seq_end, kN0);

        if constexpr(FmhaMask::IsMasking || kPadSeqLenK || kHasUnevenSplits)
        {
            const index_t logical_num_total_loop =
                integer_divide_ceil(logical_seqlen_k_end - logical_seqlen_k_start, kN0) +
                num_sink_loop;
            if(logical_num_total_loop <= 0)
            {
                if constexpr(kStoreLSE)
                {
                    // the sink alone, or -inf
                    auto lse = make_static_distributed_tensor<LSEDataType>(MakeLseDistribution());
                    set_tile(lse,
                             fmha_lse_from_natural(has_sink_logit ? sink_v * scale_s
                                                                  : -numeric<float>::infinity(),
                                                   lse_log2));
                    store_tile(lse_acc_dram_window_tmp, lse);
                }
                clear_tile(o_acc);
                return o_acc;
            }
        }

        TDMConfig tdm_config_q = PingPong::template MakeTdmConfig<
            typename Policy::template LdsPaddingConfigQ<Problem>>();
        TDMConfig tdm_config_kv = PingPong::template MakeTdmConfig<
            typename Policy::template LdsPaddingConfigKV<Problem>>();

        // Q tile in LDS, then in registers for the whole loop
        auto q_dram_window = make_tile_window(
            q_dram_block_window_tmp, Policy::template MakeQDramTileDistribution<Problem>());
        auto q_lds_view = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<QDataType*>(static_cast<char*>(smem_arena) + Layout::kQOffset),
            Policy::template MakeQLdsBlockDescriptor<Problem>());
        auto q_lds_store_window = make_tile_window(
            q_lds_view, Policy::template MakeQLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});
        auto q_lds_read_window = make_tile_window(
            q_lds_view,
            make_tuple(number<kM0>{}, number<kK0>{}),
            {0, 0},
            make_static_tile_distribution(QKBlockGemm::MakeABlockDistributionEncode()));

        load_tile_tdm(tdm_config_q, q_lds_store_window, q_dram_window);

        // For sink: kv_load_start is 0 when there are sink tokens, otherwise seqlen_k_start.
        const auto kv_load_start =
            (sink_seq_end == 0 && logical_seqlen_k_start > 0) ? logical_seqlen_k_start : 0;
        const index_t physical_seqlen_k_start = logical_seqlen_k_start;
        const index_t physical_seqlen_k_end   = logical_seqlen_k_end;

        const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();
        auto bias_dram_window =
            make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                             bias_dram_block_window_tmp.get_window_lengths(),
                             {bias_origin.at(number<0>{}), kv_load_start},
                             SaccBlockTileType{}.get_tile_distribution());

        // K/V tiles, whole: tile t goes to buffer t & 1
        static_assert(Policy::kNumKVBuffers == 2);
        auto kv_dram_window =
            make_tile_window(k_dram_block_window_tmp,
                             {kv_load_start, 0},
                             Policy::template MakeKDramTileDistribution<Problem, true>());

        const auto kv_smem_buf = [&](index_t i_buf) {
            return reinterpret_cast<KDataType*>(static_cast<char*>(smem_arena) +
                                                i_buf * Layout::kKVStride);
        };

        auto kv_lds_view = make_tensor_view<address_space_enum::lds>(
            kv_smem_buf(0), Policy::template MakeKVLdsBlockDescriptor<Problem>());
        auto kv_lds_write_window =
            make_tile_window(kv_lds_view,
                             Policy::template MakeKVLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0});

        // gemm_0's K reader: the warps' keys, kK0 of the head dim at a time
        auto k_lds_read_window = make_tile_window(
            kv_lds_view,
            make_tuple(number<kN0>{}, number<kK0>{}),
            {0, 0},
            make_static_tile_distribution(QKBlockGemm::MakeBBlockDistributionEncode()));

        // the half gemm's: the warps' first 16-key blocks
        auto k_half_lds_read_window = make_tile_window(
            kv_lds_view,
            make_tuple(number<kNWarp * 16>{}, number<kK0>{}),
            {0, 0},
            make_static_tile_distribution(QKHalfBlockGemm::MakeBBlockDistributionEncode()));

        // gemm_1's V^T readers, one per 16-key block of this warp's (gemm_0 interleaves the
        // warps' blocks), 16 d at a time
        const index_t i_warp    = amd_wave_read_first_lane(get_warp_id());
        auto v_lds_read_windows = generate_tuple(
            [&](auto i_b) {
                return make_tile_window(kv_lds_view,
                                        make_tuple(number<16>{}, number<16>{}),
                                        {(i_b * kNWarp + i_warp) * 16, 0},
                                        make_static_tile_distribution(VTransposedLoadEncoding{}));
            },
            number<kKeyBlocks>{});

        const index_t num_total_loop =
            integer_divide_ceil(physical_seqlen_k_end - physical_seqlen_k_start, kN0) +
            num_sink_loop;

        // Tile t: the window moves from tile t - 1 (the sink->normal jump at the boundary
        // between the sink tiles and the others). Tiles past the last are not loaded, so the
        // merge can use buffer num_total_loop & 1.
        const auto load_next_kv_tile = [&](index_t t) {
            if constexpr(kHasSink)
            {
                if(t == num_sink_loop)
                    move_tile_window(kv_dram_window, {physical_seqlen_k_start - sink_seq_end, 0});
            }
            move_tile_window(kv_dram_window, {kN0, 0});
            kv_lds_write_window.set_bottom_tensor_view_data_ptr(kv_smem_buf(t & 1));
            load_tile_tdm(tdm_config_kv, kv_lds_write_window, kv_dram_window);
        };

        // tile 0, issued while Q is still in flight
        load_tile_tdm(tdm_config_kv, kv_lds_write_window, kv_dram_window);

        // retire Q only; tile 0, issued after it, stays in flight
        s_wait_tensorcnt_barrier<1>();
        // Q for the whole loop, in gemm_0's kK0 slices
        using QSliceTile = decltype(load_tile(q_lds_read_window));
        statically_indexed_array<QSliceTile, k0_loops> q_slices;
        static_for<0, k0_loops, 1>{}([&](auto i_k0) {
            q_slices(i_k0) = load_tile(q_lds_read_window);
            move_tile_window(q_lds_read_window, {0, kK0});
        });

        // This warp's m (the lazily rebased row max), partial l and O. The lane halves share
        // a query; the sink is counted once, by warp 0's lower half.
        auto m_w = make_static_distributed_tensor<SMPLComputeDataType>(MakeWarpRowDistribution());
        auto l_w = make_static_distributed_tensor<SMPLComputeDataType>(MakeWarpRowDistribution());
        const bool counts_sink = i_warp == 0 && get_lane_id() < 16;
        set_tile(m_w, i_warp == 0 && has_sink_logit ? sink_m : -numeric<float>::infinity());
        set_tile(l_w, counts_sink && has_sink_logit ? 1.f : 0.f);
        statically_indexed_array<OWarpTensor, kOBlocks> o_w;
        static_for<0, kOBlocks, 1>{}([&](auto i_d) { clear_tile(o_w(i_d)); });

        auto s_acc = gemm_0.MakeCBlockTile();
        statically_indexed_array<VWarpTensor, kOBlocks> v_t;

        // STAGE 1, S = Q K^T over this warp's keys, and V^T for STAGE 4 (read under the
        // softmax). kHalf: the warp's first key block only; the other's S is -inf and its
        // V^T 0 (P 0 times whatever the TDM wrote past seqlen_k).
        const auto load_s_and_v = [&](KDataType* kv_read_ptr, auto half) {
            constexpr bool kHalf = decltype(half)::value;
            if constexpr(kHalf)
            {
                auto s_half = QKHalfBlockGemm{}.MakeCBlockTile();
                clear_tile(s_half);
                k_half_lds_read_window.set_bottom_tensor_view_data_ptr(kv_read_ptr);
                static_for<0, k0_loops, 1>{}([&](auto i_k0) {
                    QKHalfBlockGemm{}(s_half, q_slices[i_k0], load_tile(k_half_lds_read_window));
                    move_tile_window(k_half_lds_read_window, {0, kK0});
                });
                move_tile_window(k_half_lds_read_window, {0, -kK0 * k0_loops});
                set_tile(s_acc, -numeric<SaccDataType>::infinity());
                set_slice_tile(s_acc, s_half, sequence<0, 0>{}, sequence<kM0, kNWarp * 16>{});
            }
            else
            {
                clear_tile(s_acc);
                k_lds_read_window.set_bottom_tensor_view_data_ptr(kv_read_ptr);
                static_for<0, k0_loops, 1>{}([&](auto i_k0) {
                    gemm_0(s_acc, q_slices[i_k0], load_tile(k_lds_read_window));
                    move_tile_window(k_lds_read_window, {0, kK0});
                });
                move_tile_window(k_lds_read_window, {0, -kK0 * k0_loops});
            }

            constexpr index_t kVBlocks = kHalf ? 1 : kKeyBlocks;
            static_for<0, kOBlocks, 1>{}([&](auto i_d) {
                if constexpr(kHalf)
                    clear_tile(v_t(i_d));
            });
            static_for<0, kVBlocks, 1>{}([&](auto i_b) {
                auto& window = v_lds_read_windows(i_b);
                window.set_bottom_tensor_view_data_ptr(kv_read_ptr);
                static_for<0, kOBlocks, 1>{}([&](auto i_d) {
                    set_slice_tile(v_t(i_d),
                                   load_tile_transpose(window),
                                   sequence<0, i_b * 16>{},
                                   sequence<16, (i_b + 1) * 16>{});
                    move_tile_window(window, {0, 16});
                });
                move_tile_window(window, {0, -kN1});
            });
        };

        index_t i_total_loops = 0;
        do
        {
            KDataType* const kv_read_ptr = kv_smem_buf(i_total_loops & 1);

            // One barrier for two hazards: tile i has landed, and every read of the buffer
            // tile i + 1 goes to (the last trip's gemms, or on the first trip Q's) is done.
            s_wait_tensorcnt_barrier<0, 0>();

            // the next tile, if any
            if(i_total_loops + 1 < num_total_loop)
                load_next_kv_tile(i_total_loops + 1);

            const index_t kv_tile_start =
                (num_sink_loop > i_total_loops)
                    ? kv_load_start + i_total_loops * kN0
                    : physical_seqlen_k_start + (i_total_loops - num_sink_loop) * kN0;

            // A warp whose keys all lie past seqlen_k (on the last tile; its first block is
            // its lowest) skips the trip: its m, l and O stand.
            if(i_total_loops >= num_sink_loop &&
               kv_tile_start + i_warp * 16 >= physical_seqlen_k_end)
                continue;

            // The warp's second key block lies past seqlen_k (on the last tile): gemm_0 and
            // V^T over its first block only
            if(i_total_loops >= num_sink_loop &&
               kv_tile_start + (kNWarp + i_warp) * 16 >= physical_seqlen_k_end)
                load_s_and_v(kv_read_ptr, bool_constant<true>{});
            else
                load_s_and_v(kv_read_ptr, bool_constant<false>{});

            // STAGE 2: scale_s, add bias, mask
            PingPong::ApplyScaleAndBias(
                s_acc, bias_dram_window, q_origin, kv_tile_start, scale_s, position_encoding);
            PingPong::MaskScores(s_acc,
                                 mask,
                                 q_origin,
                                 kv_tile_start,
                                 i_total_loops == num_total_loop - 1,
                                 physical_seqlen_k_end);
            if constexpr(kHasSink)
            {
                if(i_total_loops == num_sink_loop - 1)
                    move_tile_window(bias_dram_window, {0, physical_seqlen_k_start - sink_seq_end});
            }
            move_tile_window(bias_dram_window, {0, kN0});

            // STAGE 3, this warp's online softmax, with qr_tdm_pingpong's lazy rebasing: m_w
            // moves, and l_w and O_w are rescaled, only when a row max outgrows it by more
            // than kRebaseThreshold (P then stays below 2^kRebaseThreshold)
            const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
            auto m_local     = block_tile_reduce<SMPLComputeDataType>(
                s_acc, sequence<1>{}, f_max, -numeric<SMPLComputeDataType>::infinity());
            ReduceRowSync(m_local, f_max); // the lane halves; each warp keeps its own

            constexpr float kRebaseThreshold = 8.f;
            float& m_row                     = m_w.get_thread_buffer()(I0);
            const float m_new                = m_local.get_thread_buffer()[I0];
            if(exp_scale * (m_new - m_row) > kRebaseThreshold)
            {
                const float tmp = exp2(exp_scale * (m_row - m_new));
                m_row           = m_new;
                tile_elementwise_inout([&](auto& x) { x *= tmp; }, l_w);
                static_for<0, kOBlocks, 1>{}([&](auto i_d) {
                    tile_elementwise_inout([&](auto& x) { x *= tmp; }, o_w(i_d));
                });
            }
            const float row_max = m_row < -numeric<float>::max() ? 0.f : exp_scale * m_row;

            // P = exp2(scale (S - m)), in S's registers, which gemm_1 takes as P_w
            PWarpTensor p_w;
            float l_new = 0.f;
            static_for<0, PWarpTensor::get_thread_buffer_size(), 1>{}([&](auto i) {
                const float e = exp2(exp_scale * s_acc.get_thread_buffer()[i] - row_max);
                l_new += e;
                p_w.get_thread_buffer()(i) = type_convert<PDataType>(e);
            });
            l_w.get_thread_buffer()(I0) += l_new;

            // STAGE 4, O_w += P_w V_w
            static_for<0, kOBlocks, 1>{}([&](auto i_d) { PVWarpGemm{}(o_w(i_d), p_w, v_t(i_d)); });
        } while(++i_total_loops < num_total_loop);

        // the query's l over both lane halves, and the O blocks this warp stores
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };
        ReduceRowSync(l_w, f_sum);

        float m_q = m_w.get_thread_buffer()[I0], l_sum = l_w.get_thread_buffer()[I0];
        constexpr index_t kBlockFloats = OWarpTensor::get_thread_buffer_size();
        if constexpr(kNWarp == 1)
        {
            // O = O_w / l; a fully masked row, l == 0, gives 0
            const float inv_l = l_sum == 0.f ? 0.f : 1.f / l_sum;
            static_for<0, kOBlocks, 1>{}([&](auto i_d) {
                static_for<0, kBlockFloats, 1>{}([&](auto i) {
                    o_acc.get_thread_buffer()(number<i_d * kBlockFloats + i>{}) =
                        o_w[i_d].get_thread_buffer()[i] * inv_l;
                });
            });
        }
        else
        {
            // Merge the warps: each hands the other its m, l and the O blocks the other
            // stores, through the K/V buffer no tile went to (read last before the last trip's
            // barrier). Per query, m = max m_w, l = sum exp(m_w - m) l_w, O = sum exp(m_w - m)
            // O_w / l.
            static_assert(kNWarp == 2);
            constexpr index_t kXRowFloats  = kOHalf * 16 + 4; // 16 B apart: no bank conflicts
            constexpr index_t kXWarpFloats = kM0 * (2 + kXRowFloats);
            static_assert(kNWarp * kXWarpFloats * sizeof(float) <= Layout::kKVStride);
            float* const x_base = reinterpret_cast<float*>(
                static_cast<char*>(smem_arena) + (num_total_loop & 1) * Layout::kKVStride);
            const auto x_rows = [&](index_t i_w, index_t i_col) {
                return make_naive_tensor_view<address_space_enum::lds>(x_base + i_w * kXWarpFloats +
                                                                           i_col * kM0,
                                                                       make_tuple(number<kM0>{}),
                                                                       make_tuple(number<1>{}),
                                                                       number<1>{},
                                                                       number<1>{});
            };
            const auto x_o = [&](index_t i_w) {
                return make_naive_tensor_view<address_space_enum::lds>(
                    x_base + i_w * kXWarpFloats + 2 * kM0,
                    make_tuple(number<kM0>{}, number<kOHalf * 16>{}),
                    make_tuple(number<kXRowFloats>{}, number<1>{}),
                    number<8>{},
                    number<1>{});
            };
            const auto row_window = [&](index_t i_w, index_t i_col) {
                return make_tile_window(
                    x_rows(i_w, i_col), make_tuple(number<kM0>{}), {0}, MakeWarpRowDistribution());
            };
            const auto o_window = [&](index_t i_w, index_t i_d) {
                return make_tile_window(
                    x_o(i_w),
                    make_tuple(number<kM0>{}, number<16>{}),
                    {0, i_d * 16},
                    make_static_tile_distribution(typename PVWarpGemm::CWarpDstrEncoding{}));
            };

            const index_t i_other = kNWarp - 1 - i_warp;
            {
                auto m_window = row_window(i_warp, 0);
                auto l_window = row_window(i_warp, 1);
                store_tile(m_window, m_w);
                store_tile(l_window, l_w);
                // the blocks the other warp stores
                static_for<0, kOHalf, 1>{}([&](auto i_d) {
                    auto window = o_window(i_warp, i_d);
                    store_tile(window, i_warp == 0 ? o_w[number<kOHalf + i_d>{}] : o_w[i_d]);
                });
            }
            block_sync_lds();

            const float m_o = load_tile(row_window(i_other, 0)).get_thread_buffer()[I0];
            const float l_o = load_tile(row_window(i_other, 1)).get_thread_buffer()[I0];
            statically_indexed_array<OWarpTensor, kOHalf> o_other;
            static_for<0, kOHalf, 1>{}(
                [&](auto i_d) { o_other(i_d) = load_tile(o_window(i_other, i_d)); });

            const float m_w_row = m_q;
            m_q                 = max(m_w_row, m_o);
            const auto rescale  = [&](float m_i) {
                return m_i < -numeric<float>::max() ? 0.f : exp2(exp_scale * (m_i - m_q));
            };
            const float f_w = rescale(m_w_row);
            const float f_o = rescale(m_o);
            l_sum           = f_w * l_sum + f_o * l_o;

            // O blocks [w, w + 1) * kOHalf: (f_w O_w + f_o O_other) / l; a fully masked
            // row, l == 0, gives 0
            const float inv_l = l_sum == 0.f ? 0.f : 1.f / l_sum;
            static_for<0, kOHalf, 1>{}([&](auto i_d) {
                const OWarpTensor& own = i_warp == 0 ? o_w[i_d] : o_w[number<kOHalf + i_d>{}];
                static_for<0, kBlockFloats, 1>{}([&](auto i) {
                    o_acc.get_thread_buffer()(number<i_d * kBlockFloats + i>{}) =
                        (f_w * own.get_thread_buffer()[i] +
                         f_o * o_other[i_d].get_thread_buffer()[i]) *
                        inv_l;
                });
            });
        }

        if constexpr(kStoreLSE)
        {
            // as StoreLse; a fully masked row stores -inf, from log(0)
            const float m_lse = m_q < -numeric<float>::max() ? 0.f : m_q;
            auto lse          = make_static_distributed_tensor<LSEDataType>(MakeLseDistribution());
            set_tile(lse, fmha_lse(kScaledScores ? m_lse : m_lse * scale_s, l_sum, lse_log2));
            store_tile(lse_acc_dram_window_tmp, lse);
        }
        return o_acc;
    }

    // The kernel's other LDS argument order (see FmhaFwdKernel).
    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename LSEaccDramBlockWindowTmp,
              typename PositionEncoding>
    CK_TILE_HOST_DEVICE auto operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,
                                        const KDramBlockWindowTmp& k_dram_block_window_tmp,
                                        const VDramBlockWindowTmp& v_dram_block_window_tmp,
                                        const BiasDramBlockWindowTmp& bias_dram_block_window_tmp,
                                        LSEaccDramBlockWindowTmp& lse_acc_dram_window_tmp,
                                        FmhaMask mask,
                                        PositionEncoding position_encoding,
                                        float scale_s,
                                        float sink_v,
                                        void* smem_arena,
                                        bool lse_log2 = false) const
    {
        return (*this)(q_dram_block_window_tmp,
                       k_dram_block_window_tmp,
                       v_dram_block_window_tmp,
                       bias_dram_block_window_tmp,
                       lse_acc_dram_window_tmp,
                       mask,
                       position_encoding,
                       scale_s,
                       smem_arena,
                       sink_v,
                       lse_log2);
    }
};

} // namespace ck_tile
