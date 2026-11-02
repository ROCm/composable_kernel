// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_pingpong.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_kvshared_policy.hpp"

namespace ck_tile {

// qr_tdm_pingpong's TDM staging for K and V that are one tensor (the caller passes the same
// K and V views): each K/V tile is staged once, by one TDM transfer into one of two LDS
// buffers, and serves both gemms. Trip i reads buffer i & 1 while the TDM fills the other
// with tile i + 1.
//
// Each warp (one or two) takes 32 keys of every tile and runs an online softmax of its own
// over them: S^T = K Q^T, then O^T += V^T P^T with V^T read transposed. The WMMA C layout
// of S^T (a lane holds one query and 16 keys) is the B operand layout of the PV WMMA, so P
// goes from the exponent straight into it, with no S round trip through LDS and each score
// exponentiated once. Two warps meet only at the K/V buffer barrier, and at the end, where
// their (m, l, O) are merged through LDS. The V view is not read.
template <typename Problem_, typename Policy_ = BlockFmhaPipelineQRKSVSTdmKVSharedPolicy>
struct BlockFmhaPipelineQRKSVSTdmKVShared : BlockFmhaPipelineQRKSVSTdmPingPong<Problem_, Policy_>
{
    using PingPong = BlockFmhaPipelineQRKSVSTdmPingPong<Problem_, Policy_>;

    using typename PingPong::BlockFmhaShape;
    using typename PingPong::FmhaMask;
    using typename PingPong::KDataType;
    using typename PingPong::LSEDataType;
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
    using PingPong::kM0;
    using PingPong::kN0;
    using PingPong::kN1;
    using PingPong::kNWarp;
    using PingPong::kPadSeqLenK;
    using PingPong::kStoreLSE;
    using PingPong::kSubQKHeaddim;

    static constexpr const char* name = "qr_tdm_kvshared";

    static_assert(kN1 == kSubQKHeaddim && std::is_same_v<KDataType, VDataType> &&
                      std::is_same_v<QDataType, KDataType>,
                  "qr_tdm_kvshared: Q, K and V of one head dim and data type");
    static_assert(std::is_same_v<KDataType, half_t> || std::is_same_v<KDataType, bf16_t>,
                  "qr_tdm_kvshared: fp16 or bf16");
    static_assert(kM0 == 16 && (kNWarp == 1 || kNWarp == 2) && kN0 == kNWarp * 32 &&
                      BlockFmhaShape::Gemm0BlockWarps::at(I0) == 1 &&
                      BlockFmhaShape::NumWarps == kNWarp && kSubQKHeaddim % 32 == 0,
                  "qr_tdm_kvshared: 16 queries, one or two warps of 32 keys each");
    static_assert(!kHasLogitsSoftCap, "qr_tdm_kvshared: no logits soft-cap");
    static_assert(std::is_same_v<SMPLComputeDataType, float> &&
                  std::is_same_v<SaccDataType, float>);

    // Lane vectors of the 16x16x32 WMMA: 16 inputs (two 8-element halves), 8 accumulators
    static constexpr bool kIsFp16 = std::is_same_v<KDataType, half_t>;
    using RawType                 = std::conditional_t<kIsFp16, llvm_fp16_t, __bf16>;
    using RawX8                   = std::conditional_t<kIsFp16, llvm_fp16x8_t, llvm_bf16x8_t>;
    using RawX16                  = std::conditional_t<kIsFp16, llvm_fp16x16_t, llvm_bf16x16_t>;
    using F32x8                   = float __attribute__((ext_vector_type(8)));
    using LdsChar                 = __attribute__((address_space(3))) char;

    static constexpr index_t kQKSteps  = kSubQKHeaddim / 32;  // QK WMMAs per 16 keys
    static constexpr index_t kOBlocks  = kSubQKHeaddim / 16;  // PV WMMAs (O^T row blocks)
    static constexpr index_t kKeyBlock = kN0 / (kNWarp * 16); // 16-key blocks per warp
    static constexpr index_t kOHalf    = kOBlocks / kNWarp;   // O^T blocks a warp stores

    // A (query, column) tile laid out as a WMMA C tile of the transposed product: query =
    // lane % 16; column = warp, 16-column block, lane / 16, then 8 columns in registers.
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

    // S^T of a trip as a (query, key) tile, the WMMA C layout of S^T = K Q^T: a warp's keys
    CK_TILE_HOST_DEVICE static constexpr auto MakeSDistribution()
    {
        return MakeWarpLocalDistribution<kN0>();
    }

    // O as the merge leaves it, in O^T's C layout: warp w holds O^T blocks [w, w + 1) *
    // kOHalf, a lane 8 consecutive d of one query (the epilogue's LDS shuffle then stores
    // whole rows)
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

    // The LDS arena in address space 3. As in amd_transpose_load_to_vgpr, a C-style cast
    // changes the address space.
    CK_TILE_DEVICE static LdsChar* AsLds(void* p)
    {
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#endif
        return (LdsChar*)(p);
#ifdef __clang__
#pragma clang diagnostic pop
#endif
    }

    // 8 elements at p, 8 at p + 32 B: a WMMA operand row, K 0..7 and 16..23 (lanes 0..15)
    // or 8..15 and 24..31 (lanes 16..31)
    CK_TILE_DEVICE static RawX16 LoadOperand(const LdsChar* p)
    {
        const RawX8 lo = *reinterpret_cast<const __attribute__((address_space(3))) RawX8*>(p);
        const RawX8 hi = *reinterpret_cast<const __attribute__((address_space(3))) RawX8*>(p + 32);
        return __builtin_shufflevector(
            lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    }

    // The same operand from the columns of row-major rows: each 8-lane group reads the 8
    // rows its lanes address, lane j getting their element j; the halves are 16 rows apart.
    // Without second_half the second half is zero.
    CK_TILE_DEVICE static RawX16
    LoadOperandTransposed(const LdsChar* p, index_t row_bytes, bool second_half)
    {
#if defined(__gfx125__)
        using Ptr     = __attribute__((address_space(3))) RawX8*;
        const auto lo = [&](const LdsChar* q) {
            if constexpr(kIsFp16)
                return __builtin_amdgcn_ds_load_tr16_b128_v8f16(
                    reinterpret_cast<Ptr>(const_cast<LdsChar*>(q)));
            else
                return __builtin_amdgcn_ds_load_tr16_b128_v8bf16(
                    reinterpret_cast<Ptr>(const_cast<LdsChar*>(q)));
        };
        const RawX8 a = lo(p);
        RawX8 b{};
        if(second_half)
            b = lo(p + 16 * row_bytes);
        return __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
#else
        ignore = p;
        ignore = row_bytes;
        ignore = second_half;
        return RawX16{};
#endif
    }

    CK_TILE_DEVICE static F32x8 Wmma(const RawX16& a, const RawX16& b, const F32x8& c)
    {
#if defined(__gfx125__)
        if constexpr(kIsFp16)
            return __builtin_amdgcn_wmma_f32_16x16x32_f16(false, a, false, b, 0, c, false, false);
        else
            return __builtin_amdgcn_wmma_f32_16x16x32_bf16(false, a, false, b, 0, c, false, false);
#else
        ignore = a;
        ignore = b;
        return c;
#endif
    }

    // x of lane l ^ 16 (the other half of the lanes holding the same query)
    CK_TILE_DEVICE static float OtherHalf(float x)
    {
        return bit_cast<float>(__builtin_amdgcn_permlanex16(
            bit_cast<int32_t>(x), bit_cast<int32_t>(x), 0x76543210, 0xfedcba98, false, false));
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
                                        float sink_v) const
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

        auto o_acc = make_static_distributed_tensor<SMPLComputeDataType>(MakeODistribution());

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
                    set_tile(lse, has_sink_logit ? sink_v * scale_s : -numeric<float>::infinity());
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

        // Q tile in LDS
        auto q_dram_window = make_tile_window(
            q_dram_block_window_tmp, Policy::template MakeQDramTileDistribution<Problem>());
        auto q_lds_view = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<QDataType*>(static_cast<char*>(smem_arena) + Layout::kQOffset),
            Policy::template MakeQLdsBlockDescriptor<Problem>());
        auto q_lds_store_window = make_tile_window(
            q_lds_view, Policy::template MakeQLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

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
                             MakeSDistribution());

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

        constexpr index_t kKVRowBytes =
            Policy::template MakeKVLdsBlockDescriptor<Problem>().calculate_offset(
                make_tuple(number<1>{}, number<0>{})) *
            sizeof(KDataType);
        constexpr index_t kQRowBytes =
            Policy::template MakeQLdsBlockDescriptor<Problem>().calculate_offset(
                make_tuple(number<1>{}, number<0>{})) *
            sizeof(QDataType);

        LdsChar* const arena = AsLds(smem_arena);
        const index_t lane   = __lane_id();
        const index_t i_warp = amd_wave_read_first_lane(get_warp_id());
        const index_t i_q    = lane % 16; // the lane's query
        const index_t i_half = lane / 16;

        // retire Q only; tile 0, issued after it, stays in flight
        s_wait_tensorcnt_barrier<1>();

        // Q^T, the QK B operand, for the whole loop
        RawX16 q_operand[kQKSteps];
        static_for<0, kQKSteps, 1>{}([&](auto i_k) {
            q_operand[i_k] = LoadOperand(arena + Layout::kQOffset + i_q * kQRowBytes +
                                         (i_k * 32 + i_half * 8) * sizeof(QDataType));
        });

        // This warp's m (the lazily rebased row max), partial l and O^T. Lanes l and l ^ 16
        // share a query; the sink is counted once, by warp 0's lanes 0..15.
        float m_w = -numeric<float>::infinity();
        float l_w = 0.f;
        if(i_warp == 0 && has_sink_logit)
        {
            m_w = sink_m;
            if(i_half == 0)
                l_w = 1.f;
        }
        F32x8 o_t[kOBlocks];
        static_for<0, kOBlocks, 1>{}([&](auto i_d) { o_t[i_d] = F32x8{0}; });

        auto s_acc = make_static_distributed_tensor<SaccDataType>(MakeSDistribution());

        index_t i_total_loops = 0;
        do
        {
            const LdsChar* const kv = arena + (i_total_loops & 1) * Layout::kKVStride;

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

            // A warp whose keys all lie past seqlen_k (on the last tile) skips the trip: its
            // m, l and O^T stand.
            if(i_total_loops >= num_sink_loop &&
               kv_tile_start + i_warp * kKeyBlock * 16 >= physical_seqlen_k_end)
                continue;

            // One warp with at most 16 keys left (on the last tile) computes S^T for its
            // first 16-key block only and reads half of V^T: the other block's P is 0. (With
            // two warps, whose second one skips such a trip, this costs the first more in
            // registers than it saves.)
            const bool all_blocks = kNWarp > 1 || i_total_loops < num_sink_loop ||
                                    kv_tile_start + kKeyBlock * 16 / 2 < physical_seqlen_k_end;

            // STAGE 1, S^T = K Q^T over this warp's keys, K read row-wise
            const LdsChar* const k_rows =
                kv + (i_warp * kKeyBlock * 16 + i_q) * kKVRowBytes + i_half * 8 * sizeof(KDataType);
            RawX16 k_operand[kKeyBlock][kQKSteps];
            static_for<0, kKeyBlock, 1>{}([&](auto i_kb) {
                if(i_kb == 0 || all_blocks)
                {
                    static_for<0, kQKSteps, 1>{}([&](auto i_k) {
                        k_operand[i_kb][i_k] = LoadOperand(k_rows + i_kb * 16 * kKVRowBytes +
                                                           i_k * 32 * sizeof(KDataType));
                    });
                }
            });
            static_for<0, kKeyBlock, 1>{}([&](auto i_kb) {
                F32x8 s_t{-numeric<float>::infinity()};
                if(i_kb == 0 || all_blocks)
                {
                    s_t = F32x8{0};
                    static_for<0, kQKSteps, 1>{}(
                        [&](auto i_k) { s_t = Wmma(k_operand[i_kb][i_k], q_operand[i_k], s_t); });
                }
                static_for<0, 8, 1>{}([&](auto i) {
                    s_acc.get_thread_buffer()[i_kb.value * 8 + i.value] = s_t[i.value];
                });
            });
            // V^T for STAGE 4, read under the softmax
            const LdsChar* const v_cols =
                kv + (i_warp * kKeyBlock * 16 + i_half * 8 + lane % 8) * kKVRowBytes +
                (lane / 8) % 2 * 8 * sizeof(KDataType);
            RawX16 v_operand[kOBlocks];
            static_for<0, kOBlocks, 1>{}([&](auto i_d) {
                v_operand[i_d] = LoadOperandTransposed(
                    v_cols + i_d * 16 * sizeof(KDataType), kKVRowBytes, all_blocks);
            });

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

            // STAGE 3, this warp's online softmax, with qr_tdm_pingpong's lazy rebasing
            const auto& s = s_acc.get_thread_buffer();
            float m_local = s[number<0>{}];
            static_for<1, kKeyBlock * 8, 1>{}([&](auto i) { m_local = max(m_local, s[i]); });
            m_local = max(m_local, OtherHalf(m_local));

            constexpr float kRebaseThreshold = 8.f;
            const bool need_rebase           = exp_scale * (m_local - m_w) > kRebaseThreshold;
            if(__builtin_amdgcn_ballot_w32(need_rebase))
            {
                const float tmp = need_rebase ? exp2(exp_scale * (m_w - m_local)) : 1.f;
                if(need_rebase)
                    m_w = m_local;
                l_w *= tmp;
                static_for<0, kOBlocks, 1>{}([&](auto i_d) { o_t[i_d] *= tmp; });
            }
            const float row_max = m_w < -numeric<float>::max() ? 0.f : exp_scale * m_w;

            RawX16 p;
            static_for<0, kKeyBlock * 8, 1>{}([&](auto i) {
                const float e = exp2(exp_scale * s[i] - row_max);
                l_w += e;
                p[i.value] = static_cast<RawType>(e);
            });

            // STAGE 4, O^T += V^T P^T
            static_for<0, kOBlocks, 1>{}(
                [&](auto i_d) { o_t[i_d] = Wmma(v_operand[i_d], p, o_t[i_d]); });
        } while(++i_total_loops < num_total_loop);

        float m_q = m_w, l_sum = 0.f, l_q = 0.f;
        if constexpr(kNWarp == 1)
        {
            l_sum             = l_w + OtherHalf(l_w);
            const float inv_l = l_sum == 0.f ? 0.f : 1.f / l_sum;
            static_for<0, kOBlocks, 1>{}([&](auto i_d) {
                static_for<0, 8, 1>{}([&](auto i) {
                    o_acc.get_thread_buffer()[i_d.value * 8 + i.value] = o_t[i_d][i.value] * inv_l;
                });
            });
        }
        else
        {
            // Merge the warps: each hands the other its m, l and the O^T blocks the other
            // stores, through the K/V buffer no tile went to (read last before the last trip's
            // barrier). Per query, m = max m_w, l = sum exp(m_w - m) l_w, O = sum exp(m_w - m)
            // O_w / l.
            constexpr index_t kXRowFloats  = kOHalf * 16 + 4; // 16 B apart: no bank conflicts
            constexpr index_t kXWarpFloats = kM0 * (2 + kXRowFloats);
            static_assert(kNWarp * kXWarpFloats * sizeof(float) <= Layout::kKVStride);
            const auto x_lds = [&](index_t i_w) {
                return reinterpret_cast<float*>(static_cast<char*>(smem_arena) +
                                                (num_total_loop & 1) * Layout::kKVStride) +
                       i_w * kXWarpFloats;
            };
            const index_t i_other = kNWarp - 1 - i_warp;
            l_q                   = l_w + OtherHalf(l_w);
            {
                float* const x = x_lds(i_warp);
                if(i_half == 0)
                {
                    x[i_q * 2 + 0] = m_w;
                    x[i_q * 2 + 1] = l_q;
                }
                float* const row = x + kM0 * 2 + i_q * kXRowFloats + i_half * 8;
                static_for<0, kOHalf, 1>{}([&](auto i_d) {
                    const F32x8 o = i_warp == 0 ? o_t[kOHalf + i_d] : o_t[i_d];
                    static_for<0, 8, 1>{}([&](auto i) { row[i_d * 16 + i.value] = o[i.value]; });
                });
            }
            block_sync_lds();

            const float* const x = x_lds(i_other);
            const float m_o      = x[i_q * 2 + 0];
            m_q                  = max(m_w, m_o);
            const auto rescale   = [&](float m_i) {
                return m_i < -numeric<float>::max() ? 0.f : exp2(exp_scale * (m_i - m_q));
            };
            const float f_w = rescale(m_w);
            const float f_o = rescale(m_o);
            l_sum           = f_w * l_q + f_o * x[i_q * 2 + 1];
            // a fully masked row, l == 0, gives 0
            const float inv_l     = l_sum == 0.f ? 0.f : 1.f / l_sum;
            const float* const xr = x + kM0 * 2 + i_q * kXRowFloats + i_half * 8;
            static_for<0, kOHalf, 1>{}([&](auto i_d) {
                const F32x8 o = i_warp == 0 ? o_t[i_d] : o_t[kOHalf + i_d];
                static_for<0, 8, 1>{}([&](auto i) {
                    o_acc.get_thread_buffer()[i_d.value * 8 + i.value] =
                        (f_w * o[i.value] + f_o * xr[i_d * 16 + i.value]) * inv_l;
                });
            });
        }
        if constexpr(kStoreLSE)
        {
            // natural log, as StoreLse; a fully masked row stores -inf, from log(0)
            constexpr float kLn2 = 1.f / log2e_v<float>;
            const float m_lse    = m_q < -numeric<float>::max() ? 0.f : m_q;
            auto lse = make_static_distributed_tensor<LSEDataType>(MakeLseDistribution());
            lse.get_thread_buffer()[0] =
                m_lse * (kScaledScores ? kLn2 : scale_s * kLn2) + log(l_sum);
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
                                        void* smem_arena) const
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
                       sink_v);
    }
};

} // namespace ck_tile
