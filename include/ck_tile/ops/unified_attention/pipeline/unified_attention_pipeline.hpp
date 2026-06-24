// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/unified_attention/pipeline/unified_attention_pipeline_default_policy.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline.hpp"
#include "ck_tile/ops/unified_attention/pipeline/unified_attention_core_loop_scheduler.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"
// UA_DECODE_INTERLEAVE (decode_pipeline_research_plan.md §13 — HK 4-wave borrow)
//   0 (default): serial-decode loop fully drains vmcnt<0> each iter, then issues
//     the next KV prefetch -> only ~1 KV tile is ever in flight (the MLP cap E0
//     measured on the bandwidth-bound decode loop).
//   1: issue the next tile's K+V async prefetch BEFORE the consume-wait and relax
//     the wait to a partial vmcnt that keeps that prefetch streaming under the
//     QK/PV compute -> ~2 KV tiles in flight, no extra LDS (still 2 ring buffers).
//   Decode-only (NumWarpGroups==1); the FA4 prefill ping-pong path is untouched.
#ifndef UA_DECODE_INTERLEAVE
#define UA_DECODE_INTERLEAVE 0
#endif

#define ENABLE_ASM_MARKER 1
#if ENABLE_ASM_MARKER
#define ASM_MARKER(marker)               \
    __builtin_amdgcn_sched_barrier(0);   \
    asm volatile("; [POYENC] " #marker); \
    __builtin_amdgcn_sched_barrier(0);
#else
#define ASM_MARKER(marker)
#endif

// Two pipeline regimes share this operator() (see this folder's README.md):
//  - FA4 (NumWarpGroups == 2, prefill): matrix‖softmax overlap. Both warp groups
//    run the deferred-PV sequence (alu1(k-1) -> PV(k-1) -> QK(k) -> alu0(k) ->
//    D_upd) split into MATRIX (PV+QK) and SOFTMAX (alu1/alu0/D_upd) phases, primed
//    one phase apart so each SIMD hides its matrix work under its partner's VALU.
//    K/V are prefetched a tile ahead into a shared double buffer. Gated by kFA4.
//  - Serial decode (NumWarpGroups == 1): single-warp-group deferred-PV pipeline
//    with a 2-buffer async KV prefetch.

namespace ck_tile {

// kPageSize_ > 0 pins page_size to a compile-time constant (host dispatcher routes
// to a matching instance; 0 = runtime fallback), strength-reducing every / * % and
// enabling the real `KY0_step_N <= kPageSize` Tier gate vs the conservative <=16.
template <typename Problem_,
          typename Policy_                  = UnifiedAttentionPipelineDefaultPolicy,
          ck_tile::index_t kPageSize_       = 0,
          bool kIsPaged_                    = true>
struct UnifiedAttentionPipeline
{
    using Problem             = ck_tile::remove_cvref_t<Problem_>;
    using Policy              = ck_tile::remove_cvref_t<Policy_>;

    // Compile-time page size (0 = runtime). See class-level comment above.
    static constexpr ck_tile::index_t kPageSize       = kPageSize_;
    static constexpr bool             kHasCePageSize = (kPageSize_ > 0);
    // Paged KV (block_tables indirection) vs contiguous/THD KV. When false the
    // logical token index IS the physical row (kernel folds the per-sequence base
    // into the K/V pointer), so all paging math is compiled out to a linear offset.
    static constexpr bool             kIsPaged       = kIsPaged_;
    using QDataType           = ck_tile::remove_cvref_t<typename Problem::QDataType>;
    using KDataType           = ck_tile::remove_cvref_t<typename Problem::KDataType>;
    using VDataType           = ck_tile::remove_cvref_t<typename Problem::VDataType>;
    using SaccDataType        = ck_tile::remove_cvref_t<typename Problem::SaccDataType>;
    using SMPLComputeDataType = ck_tile::remove_cvref_t<typename Problem::SMPLComputeDataType>;
    using PDataType           = ck_tile::remove_cvref_t<typename Problem::PDataType>;
    using OaccDataType        = ck_tile::remove_cvref_t<typename Problem::OaccDataType>;
    using ODataType           = ck_tile::remove_cvref_t<typename Problem::ODataType>;
    using FmhaMask            = ck_tile::remove_cvref_t<typename Problem::FmhaMask>;

    static_assert(std::is_same_v<SaccDataType, SMPLComputeDataType>,
                  "we will the same dist tensor 'sp_compute' for both gemm0 & softmax");

    using UnifiedAttentionShape = ck_tile::remove_cvref_t<typename Problem::UnifiedAttentionShape>;

    static constexpr ck_tile::index_t kBlockSize = Problem::kBlockSize;

    static constexpr ck_tile::index_t kBlockM = UnifiedAttentionShape::kBlockM;
    static constexpr ck_tile::index_t kBlockQ = UnifiedAttentionShape::kBlockQ;

    static constexpr ck_tile::index_t kWarpGemmM =
        UnifiedAttentionShape::Gemm0WarpTile::at(ck_tile::number<0>{});

    static constexpr ck_tile::index_t kPageBlockSize = UnifiedAttentionShape::kPageBlockSize;
    static constexpr ck_tile::index_t kHeadDim       = UnifiedAttentionShape::kHeadDim;
    static constexpr ck_tile::index_t kHeadDimPadded = UnifiedAttentionShape::kHeadDimPadded;

    static_assert(kHeadDimPadded <= 256, "hdim bigger than 256 is not suitable for this pipeline!");

    // static constexpr bool kPadSeqLenQ  = Problem::kPadSeqLenQ;
    static constexpr bool kPadHeadDimQ = Problem::kPadHeadDim;
    static constexpr bool kPadHeadDimV = Problem::kPadHeadDim;
    // static constexpr bool kStoreLSE    = Problem::kStoreLSE;

    // DRAM-view vector length (also the buffer_load vector length).
    static constexpr ck_tile::index_t kAlignmentQ =
        kPadHeadDimQ ? 1 : Policy::template GetAlignmentQ<Problem>();
    // Must match the K/V load distribution's KVector (the per-warp-group load count
    // GetK/VLoadNumWarps) to keep buffer_load width in sync with the async copies.
    static constexpr ck_tile::index_t kAlignmentK =
        kPadHeadDimQ ? 1
                     : Policy::template GetAlignmentK<Problem,
                                                      Policy::template GetKLoadNumWarps<Problem>()>();
    static constexpr ck_tile::index_t kAlignmentV =
        kPadHeadDimV ? 1
                     : Policy::template GetAlignmentV<Problem,
                                                      Policy::template GetVLoadNumWarps<Problem>()>();

    static constexpr ck_tile::index_t kAlignmentO =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentO<Problem>();

    static constexpr ck_tile::index_t kBlockPerCu = []() {
        if constexpr(Problem::kBlockPerCu != -1)
            return Problem::kBlockPerCu;
        else
        {
            return 2;
        }
    }();

    // Tier-2 LDS-resident page-table cache: 4096 entries × 4 B = 16 KiB. Covers
    // sk ≤ kPageTableLdsEntries * page_size tokens (asserted in the kernel).
    static constexpr ck_tile::index_t kPageTableLdsEntries = 4096;

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetPageTableLdsBytes()
    {
        // Allocate only for instances whose Tier-0 gate fires; mirror operator()'s
        // gate exactly (see lock-step note inside).
        using KDist = decltype(Policy::template MakeKDramTileDistribution<Problem>());
        using VDist = decltype(Policy::template MakeVDramTileDistribution<Problem>());
        constexpr ck_tile::index_t KNRepeat =
            KDist::DstrEncode::hs_lengthss_[ck_tile::number<0>{}][ck_tile::number<0>{}];
        constexpr ck_tile::index_t VNRepeat =
            VDist::DstrEncode::hs_lengthss_[ck_tile::number<0>{}][ck_tile::number<0>{}];
        constexpr ck_tile::index_t KY0_step_N =
            KDist::DstrEncode::hs_lengthss_[ck_tile::number<0>{}][ck_tile::number<1>{}] *
            KDist::DstrEncode::hs_lengthss_[ck_tile::number<0>{}][ck_tile::number<2>{}];
        constexpr ck_tile::index_t VY0_step_N =
            VDist::DstrEncode::hs_lengthss_[ck_tile::number<0>{}][ck_tile::number<1>{}] *
            VDist::DstrEncode::hs_lengthss_[ck_tile::number<0>{}][ck_tile::number<2>{}];
        constexpr ck_tile::index_t kPageSizeCap =
            kHasCePageSize ? kPageSize : ck_tile::index_t{16};
        // Lock-step with kScalarPromote*/kFallbackLds* in operator(): divergence
        // reserves no LDS for an offset the runtime path uses, corrupting K/V.
        constexpr bool kScalarPromoteK =
            (KNRepeat >= 2) && (KY0_step_N <= kPageSizeCap);
        constexpr bool kFallbackLdsK =
            Policy::kKFallbackLds && kIsPaged && kHasCePageSize && !kScalarPromoteK;
        constexpr bool kScalarPromoteV =
            (VNRepeat >= 2) && (VY0_step_N <= kPageSizeCap);
        constexpr bool kFallbackLdsV =
            Policy::kKFallbackLds && kIsPaged && kHasCePageSize && !kScalarPromoteV;
        constexpr bool kHasTier0K = kScalarPromoteK || kFallbackLdsK;
        constexpr bool kHasTier0V = kScalarPromoteV || kFallbackLdsV;
        if constexpr (kHasTier0K || kHasTier0V)
            return kPageTableLdsEntries * sizeof(ck_tile::index_t);
        else
            return 0;
    }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        // Two layouts share the smem base: (1) o_lds overlapped with s_lds, and
        // (2) the K/V double buffer + p_lds. The s_lds term is the tightest bound
        // for FP8. The Tier-2 page-table cache (if any) is appended at the end.
        return ck_tile::max(ck_tile::max(kBlockM * kHeadDimPadded * sizeof(PDataType),
                                         kBlockM * kPageBlockSize * sizeof(SaccDataType)),
                            Policy::template GetSmemSize<Problem>() +
                                kBlockM * kPageBlockSize * sizeof(PDataType)) +
               GetPageTableLdsBytes();
    }

    template <ck_tile::index_t MPerBlock, ck_tile::index_t NPerBlock>
    CK_TILE_DEVICE static constexpr auto MakeSimpleLdsDesc()
    {
        using namespace ck_tile;
        constexpr auto lds_block_desc =
            make_naive_tensor_descriptor(make_tuple(number<MPerBlock>{}, number<NPerBlock>{}),
                                         make_tuple(number<NPerBlock>{}, number<1>{}),
                                         number<1>{},
                                         number<1>{});

        return lds_block_desc;
    }

    template <ck_tile::index_t MPerBlock>
    CK_TILE_DEVICE static constexpr auto MakeSimpleLdsDesc1D()
    {
        using namespace ck_tile;
        constexpr auto lds_block_desc = make_naive_tensor_descriptor(
            make_tuple(number<MPerBlock>{}), make_tuple(number<1>{}), number<1>{}, number<1>{});

        return lds_block_desc;
    }

    template <typename DataType, typename Descriptor>
    CK_TILE_DEVICE static constexpr auto make_lds_tile_window(void* base, const Descriptor& desc)
    {
        using namespace ck_tile;

        auto tensor_view =
            make_tensor_view<address_space_enum::lds>(reinterpret_cast<DataType*>(base), desc);
        return make_tile_window(tensor_view, desc.get_lengths(), {0, 0});
    }

    // vmcnt=0~63, lgkmcnt=0~15, expcnt=0~7
    template <uint16_t Vmcnt, uint8_t Lgkmcnt, uint8_t Expcnt = 7>
    CK_TILE_DEVICE static constexpr void s_waitcnt()
    {
        // vmcnt use bits {[15:14],[3:0]}
        // expcnt use bits [6:4]
        // lgkmcnt use bits [11:8]
        __builtin_amdgcn_s_waitcnt((((0b110000 & Vmcnt) << (14 - 4)) | (0b1111 & Vmcnt)) |
                                   ((0b111 & Expcnt) << 4) | ((0b1111 & Lgkmcnt) << 8));
    }

    template <uint16_t Vmcnt>
    CK_TILE_DEVICE static constexpr void s_waitcnt_vmcnt()
    {
        s_waitcnt<Vmcnt, 15>();
    }

    template <uint8_t Lgkmcnt>
    CK_TILE_DEVICE static constexpr void s_waitcnt_lgkmcnt()
    {
        s_waitcnt<63, Lgkmcnt>();
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename QElementFunction,
              typename KElementFunction,
              typename VElementFunction,
              typename SAccElementFunction,
              typename PComputeElementFunction,
              typename OAccElementFunction>
    CK_TILE_DEVICE auto operator()(
        const QDramBlockWindowTmp& q_dram_block_window_tmp, // kBlockM * kHeadDimPadded tile
        const QElementFunction& q_element_func,
        const KDramBlockWindowTmp& k_dram_block_window_tmp, // kPageBlockSize * kHeadDimPadded tile
        [[maybe_unused]] const KElementFunction& k_element_func,
        const VDramBlockWindowTmp& v_dram_block_window_tmp, // kHeadDimPadded * kPageBlockSize tile
        [[maybe_unused]] const VElementFunction& v_element_func,
        const index_t num_blocks,
        const index_t num_blocks_start,
        const void* block_tables_ptr,
        index_t block_table_offset,
        // Runtime page size. Ignored (asserted to match) when kPageSize_ > 0;
        // the body always reads the local `page_size` resolved below.
        const index_t page_size_runtime,
        [[maybe_unused]] const SAccElementFunction& s_acc_element_func,
        const PComputeElementFunction& p_compute_element_func,
        const OAccElementFunction& o_acc_element_func,
        FmhaMask mask,
        float scale_s,
        void* smem_ptr,
        long_index_t k_row_stride         = 0,
        long_index_t v_row_stride         = 0,
        // Runtime kBlockQ = kBlockM / num_queries_per_kv; 0 = fall back to the
        // compile-time kBlockQ from UnifiedAttentionShape.
        const index_t num_queries_per_kv = 0,
        // Set when the K/V cache byte size can exceed INT32_MAX: routes async
        // loads through the 64-bit-base `global_load_lds` path (lower throughput
        // but correct). False uses the shared-SRD `buffer_load_dword_lds` path.
        const bool cache_ptr_int32_overflow_possible = false,
        // Per-tensor FP8 V descale, applied to o_acc once after the 1/l norm
        // (1.0f for non-FP8). Deferring it outside the loop is exact; split-KV
        // partials carry it through the combine unchanged.
        const float v_descale = 1.0f) const
    {
        using namespace ck_tile;
        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

        // Resolve `page_size` once: the constexpr literal when kPageSize_ > 0
        // (strength-reducing every / * % below), else the runtime value.
        if constexpr (kHasCePageSize) { assert(page_size_runtime == kPageSize); }
        const index_t page_size = kHasCePageSize ? kPageSize : page_size_runtime;

        static_assert(
            kBlockM == QDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                kPageBlockSize == KDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                kHeadDimPadded == KDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                kPageBlockSize == VDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                kHeadDimPadded == VDramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
            "wrong!");

        static_assert(sizeof(SaccDataType) * kPageBlockSize * kBlockM <= GetSmemSize());
        auto s_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<SaccDataType*>(static_cast<char*>(smem_ptr)),
            MakeSimpleLdsDesc<kBlockM, kPageBlockSize>());
        [[maybe_unused]] auto s_lds_window = make_tile_window(
            s_lds, make_tuple(number<kBlockM>{}, number<kPageBlockSize>{}), {0, 0});

        auto p_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<PDataType*>(static_cast<char*>(smem_ptr) +
                                         Policy::template GetSmemSize<Problem>()),
            MakeSimpleLdsDesc<kBlockM, kPageBlockSize>());
        [[maybe_unused]] auto p_lds_window = make_tile_window(
            p_lds, make_tuple(number<kBlockM>{}, number<kPageBlockSize>{}), {0, 0});

        auto o_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<PDataType*>(static_cast<char*>(smem_ptr)),
            MakeSimpleLdsDesc<kBlockM, kHeadDimPadded>());
        [[maybe_unused]] auto o_lds_window = make_tile_window(
            o_lds, make_tuple(number<kBlockM>{}, number<kHeadDimPadded>{}), {0, 0});

        auto m_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<SMPLComputeDataType*>(static_cast<char*>(smem_ptr) +
                                                   Policy::template GetSmemSize<Problem>()),
            MakeSimpleLdsDesc1D<kBlockM>());
        [[maybe_unused]] auto m_lds_window =
            make_tile_window(m_lds, make_tuple(number<kBlockM>{}), {0});

        const index_t warp_group_id = get_warp_id() / 4;

        // K and V are loaded cooperatively across all NumWarps waves.
        constexpr index_t VLoadNumWarps = Policy::template GetVLoadNumWarps<Problem>();
        constexpr index_t KLoadNumWarps = Policy::template GetKLoadNumWarps<Problem>();

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetPVBlockGemm<Problem>();

        // FP8 P-tile re-layout windows (LDS roundtrip). FP8 forces the PV gemm into
        // WGAttrNumAccess::Single, so the QK-C and PV-A distributions diverge: store
        // P through p_lds under QK-C, block-sync, reload under PV-A. (BF16/FP16
        // layouts agree.) Windows hoisted so make_tile_window runs once.
        [[maybe_unused]] auto p_lds_store_window_qkc = make_tile_window(
            p_lds_window,
            decltype(gemm_0.MakeCBlockTile())::get_tile_distribution());
        [[maybe_unused]] auto p_lds_load_window_pva = make_tile_window(
            p_lds_window,
            Policy::template MakePRegTileDistribution<Problem>());

        auto q_dram_window = make_tile_window_linear(
            q_dram_block_window_tmp, Policy::template MakeQRegTileDistribution<Problem>());

        // softmax reductions
        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        // KV LDS double buffer: 2 K buffers in slots [0,2), 2 V buffers in [2,4)
        // (the V store descriptor's K-buffer base == kRingStages).
        constexpr index_t kRingStages = 2;

        constexpr index_t KStoreWarpShift = 0;
        auto k_lds_window_store = generate_tuple(
            [&](auto i_buf) {
                return make_lds_tile_window<KDataType>(
                    smem_ptr,
                    Policy::template MakeKLdsStoreBlockDescriptor<Problem,
                                                                  KLoadNumWarps,
                                                                  KStoreWarpShift>(i_buf));
            },
            number<kRingStages>{});

        auto v_lds_window_store = generate_tuple(
            [&](auto i_buf) {
                return make_lds_tile_window<KDataType>(
                    smem_ptr,
                    Policy::template MakeVLdsStoreBlockDescriptor<Problem,
                                                                  VLoadNumWarps,
                                                                  kRingStages>(i_buf));
            },
            number<kRingStages>{});

        statically_indexed_array<
            decltype(make_tile_window(
                make_lds_tile_window<KDataType>(
                    nullptr,
                    Policy::template MakeKLdsLoadBlockDescriptor<Problem, KLoadNumWarps>()),
                Policy::template MakeKRegTileDistribution<Problem>())),
            kRingStages>
            k_lds_window_load;

        statically_indexed_array<
            decltype(make_tile_window(
                make_lds_tile_window<VDataType>(
                    nullptr,
                    Policy::template MakeVLdsLoadBlockDescriptor<Problem, VLoadNumWarps>()),
                Policy::template MakeVRegTileDistribution<Problem>())),
            kRingStages>
            v_lds_window_load;

        decltype(make_static_distributed_tensor<QDataType>(
            Policy::template MakeQRegTileDistribution<Problem>())) q_tile;

        // k_tile / v_tile are separate (not a union) so the K ds_read can overlap
        // the PV MFMA on the LSU; a union serialized them. Occupancy is LDS-bound.
        struct kv_tile_type
        {
            CK_TILE_DEVICE kv_tile_type() {}

            decltype(load_tile(k_lds_window_load(number<0>{}))) k_tile;

            decltype(load_tile_transpose(v_lds_window_load(number<0>{}))) v_tile;
        } kv_tile;

        union sp_compute_type
        {
            CK_TILE_DEVICE sp_compute_type() {}

            decltype(gemm_0.MakeCBlockTile()) sp_compute;
            decltype(make_static_distributed_tensor<PDataType>(
                Policy::template MakePRegTileDistribution<Problem>())) p;
        };
        // kv128: collapse the deferred-PV score/P double buffer to one slot so the
        // fp32 tile fits the 256-VGPR ceiling (WAR hazard resolved by a pure-VGPR
        // PV->QK serialization). Smaller tiles keep the double buffer; the accessor
        // ignores the slot index.
        static constexpr bool kUseSingleSp = (kPageBlockSize >= 128);
        struct sp_holder_t
        {
            sp_compute_type s_;
            CK_TILE_DEVICE constexpr sp_compute_type& operator()(index_t) { return s_; }
        };
        std::conditional_t<kUseSingleSp,
                           sp_holder_t,
                           statically_indexed_array<sp_compute_type, 2>>
            sp;

        decltype(gemm_1.MakeCBlockTile()) o_acc;
        constexpr index_t fmha_alu_D_reg_cnt = 6;
        static_assert(fmha_alu_D_reg_cnt <= o_acc.thread_buf_.size());

        decltype(block_tile_reduce<SMPLComputeDataType>(
            sp(number<0>{}).sp_compute, sequence<1>{}, f_max, SMPLComputeDataType{0})) m;
        decltype(m) l;
        // Conditional-rescale state (FA4 prefill): committed max the o_acc/l
        // accumulators normalise against, its prior value, plus a wave-uniform
        // "this tile advanced m_commit" flag.
        decltype(m) m_commit;
        decltype(m) m_commit_old;
        bool need_rescale = true;

        // K buffers occupy LDS slots [0, kRingStages); V buffers follow at
        // [kRingStages, 2*kRingStages), matching the V store descriptor base.
        static_for<0, kRingStages, 1>{}([&](auto idx) {
            k_lds_window_load(idx) = make_tile_window(
                make_lds_tile_window<KDataType>(
                    static_cast<char*>(smem_ptr) + (idx)*Policy::template GetSmemSizeKV<Problem>(),
                    Policy::template MakeKLdsLoadBlockDescriptor<Problem, KLoadNumWarps>()),
                Policy::template MakeKRegTileDistribution<Problem>());
        });

        static_for<0, kRingStages, 1>{}([&](auto idx) {
            v_lds_window_load(idx) =
                make_tile_window(make_lds_tile_window<VDataType>(
                                     static_cast<char*>(smem_ptr) +
                                         (idx + kRingStages) *
                                             Policy::template GetSmemSizeKV<Problem>(),
                                     Policy::template MakeVLdsLoadBlockDescriptor<Problem,
                                                                                 VLoadNumWarps>()),
                                 Policy::template MakeVRegTileDistribution<Problem>());
        });

        {
            auto origin_q      = load_tile(q_dram_window);
            auto transformed_q = tile_elementwise_in(q_element_func, origin_q);

            q_tile = transformed_q;
        }

        clear_tile(o_acc);
        set_tile(m, bit_cast<float>(0xff7fffff)); // a bit larger than -infinity
        clear_tile(l);
        // -inf-ish init: the first tile always commits, with m_commit_old == -inf
        // giving o_acc_scale == exp2(-inf) == 0, a no-op on the cleared o_acc/l.
        set_tile(m_commit, bit_cast<float>(0xff7fffff));
        set_tile(m_commit_old, bit_cast<float>(0xff7fffff));

        const auto q_origin = q_dram_window.get_window_origin();

        const auto num_total_loop = num_blocks;
        index_t k_block_idx       = 0;
        index_t v_block_idx       = 0;

        // check early exit if no work to do
        if constexpr(FmhaMask::IsMasking)
        {
            if(num_total_loop - num_blocks_start <= 0)
            {
                // o_acc already cleared. lse = -inf so the split-KV combine
                // weighs this empty partial as zero (exp(-inf) == 0).
                auto lse_early =
                    make_static_distributed_tensor<SMPLComputeDataType>(m.get_tile_distribution());
                set_tile(lse_early, -ck_tile::numeric<SMPLComputeDataType>::infinity());
                return ck_tile::make_tuple(o_acc, lse_early);
            }
        }

        index_t i_total_loops = num_blocks_start;
        const ck_tile::index_t* block_tables_ptr_ =
            reinterpret_cast<const ck_tile::index_t*>(block_tables_ptr);
        assert(k_block_idx == v_block_idx);
        // Split-KV start offset in *tokens*, added to logical_token so the page
        // lookup hits the right page. block_table_offset is NOT shifted (it indexes
        // page_size-sized pages, num_blocks_start counts kPageBlockSize tiles).
        const index_t split_token_offset = num_blocks_start * kPageBlockSize;

        // Unified page-offset formula, per (thread, Y0-iter) pair:
        //     logical_token = tile_idx * kPageBlockSize + thread_N_pos + i*Y0_step_N
        //     phys_page      = block_tables[block_table_offset + logical_token/page_size]
        //     page_offsets[i] = (phys_page*page_size + logical_token%page_size) * row_stride
        // Indirection lives entirely in page_offsets (refreshed per tile, no per-iter
        // SRD rebase). Requires Y0_step_N to divide page_size so a wave-wide load
        // never straddles a page. page_offsets are int32 (long path for overflow).
        const auto k_dist = Policy::template MakeKDramTileDistribution<Problem, KLoadNumWarps>();
        const auto v_dist = Policy::template MakeVDramTileDistribution<Problem, VLoadNumWarps>();
        using KDstrType   = decltype(k_dist);
        using VDstrType   = decltype(v_dist);
        // Issue (Y0) dim is H0[0] with stride H0[1]*H0[2]; KNRepeat is its extent.
        constexpr index_t KNRepeat =
            KDstrType::DstrEncode::hs_lengthss_[number<0>{}][number<0>{}];
        constexpr index_t VNRepeat =
            VDstrType::DstrEncode::hs_lengthss_[number<0>{}][number<0>{}];
        constexpr index_t KY0_step_N =
            KDstrType::DstrEncode::hs_lengthss_[number<0>{}][number<1>{}] *
            KDstrType::DstrEncode::hs_lengthss_[number<0>{}][number<2>{}];
        constexpr index_t VY0_step_N =
            VDstrType::DstrEncode::hs_lengthss_[number<0>{}][number<1>{}] *
            VDstrType::DstrEncode::hs_lengthss_[number<0>{}][number<2>{}];

        // K/V are loaded cooperatively by all warps, so warp_id % NumWarps is the
        // identity here; the modulo keeps the partition index in range.
        const auto k_part = ck_tile::array<index_t, 2>{get_warp_id() % KLoadNumWarps, get_lane_id()};
        const auto v_part = ck_tile::array<index_t, 2>{get_warp_id() % VLoadNumWarps, get_lane_id()};
        const auto k_thread_coord    = k_dist.calculate_index(k_part);
        const auto v_thread_coord    = v_dist.calculate_index(v_part);
        const index_t k_thread_n_pos = k_thread_coord[number<0>{}];
        const index_t v_thread_n_pos = v_thread_coord[number<0>{}];

        // Page offsets widened to long_index_t so the `_long` load path can
        // address pools exceeding INT32_MAX bytes; the non-long path narrows back
        // to int32 (safe when cache_ptr_int32_overflow_possible == false).
        statically_indexed_array<long_index_t, KNRepeat> k_page_offsets;
        statically_indexed_array<long_index_t, VNRepeat> v_page_offsets;

        // Tier 0 — scalar-promote the block_tables[] page-index lookup. When the
        // per-lane n-span in one issue (KY0_step_N tokens) fits in one page,
        // phys_page is warp-uniform, so readfirstlane lets LLVM emit one per-warp
        // s_load_dword instead of 64 per-lane global loads. Gate is `<= kPageSize`
        // when constexpr, else a conservative `<= 16` hedge.
        constexpr index_t kKPageSizeCap = kHasCePageSize ? kPageSize : index_t{16};
        constexpr index_t kVPageSizeCap = kHasCePageSize ? kPageSize : index_t{16};
        constexpr bool kScalarPromoteKPageIdx =
            (KNRepeat >= 2) && (KY0_step_N <= kKPageSizeCap);
        constexpr bool kScalarPromoteVPageIdx =
            (VNRepeat >= 2) && (VY0_step_N <= kVPageSizeCap);

        // Multi-page K/V (Y0_step_N > page, e.g. ps16/ps32) routes its per-lane
        // fallback through the LDS page-table cache rather than per-lane global reads.
        constexpr bool kFallbackUsesLdsK = Policy::kKFallbackLds && kIsPaged &&
            kHasCePageSize && !kScalarPromoteKPageIdx;
        constexpr bool kFallbackUsesLdsV = Policy::kKFallbackLds && kIsPaged &&
            kHasCePageSize && !kScalarPromoteVPageIdx;


        // Tier 2 — LDS-resident page-table cache. One cooperative bulk load at entry
        // stages this CTA's block_tables slice into LDS, turning each later refresh
        // into a ds_read_b32 broadcast instead of an s_load_dword + scoreboard wait.
        constexpr bool kKNeedsPageTableLds = kScalarPromoteKPageIdx || kFallbackUsesLdsK;
        constexpr bool kVNeedsPageTableLds = kScalarPromoteVPageIdx || kFallbackUsesLdsV;
        constexpr bool kUsePageTableLds =
            kIsPaged && (kKNeedsPageTableLds || kVNeedsPageTableLds);
        constexpr index_t kPageTableLdsOffset =
            GetSmemSize() - GetPageTableLdsBytes();
        auto block_tables_lds = reinterpret_cast<int32_t*>(
            static_cast<char*>(smem_ptr) + kPageTableLdsOffset);

        // Per-split window: each CTA only references pages in [split_start_page,
        // split_end_page), so bulk-load just that slice and shift the refresh lookup
        // by split_start_page, keeping the LDS index in [0, split_window_pages).
        const index_t split_start_page = static_cast<index_t>(
            (static_cast<long_index_t>(num_blocks_start) * kPageBlockSize) / page_size);
        const index_t split_end_page = static_cast<index_t>(
            (static_cast<long_index_t>(num_total_loop) * kPageBlockSize + page_size - 1) /
            page_size);
        const index_t split_window_pages = split_end_page - split_start_page;
        // Sliding LDS window: `lds_window_base` is the absolute page index of LDS
        // entry 0, so a refresh for absolute page p reads block_tables_lds[p -
        // lds_window_base]. If the whole window fits, slide_page_table() never
        // fires; otherwise it slides forward as the monotonic tile loop consumes it.
        index_t lds_window_base = split_start_page;
        if constexpr (kUsePageTableLds)
        {
            const index_t init_pages =
                split_window_pages < kPageTableLdsEntries ? split_window_pages
                                                          : static_cast<index_t>(kPageTableLdsEntries);
            const index_t tid = get_thread_local_1d_id();
            for (index_t i = tid; i < init_pages; i += Problem::kBlockSize)
            {
                block_tables_lds[i] = block_tables_ptr_[block_table_offset + split_start_page + i];
            }
            // Cross-lane LDS: a refresh may read another lane's write. The s_barrier
            // orders cross-wave, but single-warp CTAs (TinyDecode) elide it along
            // with its implicit lgkmcnt(0) drain, so add an explicit drain (no-op on
            // multi-warp tiers, load-bearing on single-warp).
            s_waitcnt_lgkmcnt<0>();
            __builtin_amdgcn_s_barrier();
        }

        // Within-tile phys_page dedup: with a compile-time page_size that divides
        // (or is divided by) the tile, the issue->page map is compile-time, so
        // phys_page is resolved once per distinct page and reused.
        constexpr bool kDedupPages =
            kHasCePageSize &&
            (kPageBlockSize % kPageSize == 0 || kPageSize % kPageBlockSize == 0);

        // Single-page SRD-rebase: when the whole tile sits in one page, the per-issue
        // offset splits into a wave-uniform U(tile) folded into the SRD base once per
        // tile, plus a loop-invariant per-lane L. Drops per-lane 64-bit addr math
        // from the tile loop. Gated to single-page geometry AND (scalar-promote or
        // the trivial NRepeat==1 single-issue tile).
        constexpr bool kSinglePageGeom =
            kHasCePageSize && (kPageSize % kPageBlockSize == 0);
        constexpr bool kRebaseKSrd =
            kSinglePageGeom &&
            (kScalarPromoteKPageIdx || (KNRepeat == 1 && KY0_step_N <= kPageSize));
        constexpr bool kRebaseVSrd =
            kSinglePageGeom &&
            (kScalarPromoteVPageIdx || (VNRepeat == 1 && VY0_step_N <= kPageSize));


        // Wave-uniform per-tile base offsets (in elements) folded into the SRD
        // base at window construction; written by refresh_*_offsets.
        long_index_t k_srd_base_offset = 0;
        long_index_t v_srd_base_offset = 0;

        // Cross-stagger phys_page carry: K runs a tile ahead of V sharing the same
        // page map, so V reuses the phys_page K broadcast. Keyed by tile parity:
        // refresh_k writes ring[tile&1], refresh_v reads it (rewritten only at N+2,
        // after V consumed N). Valid only when both take the single-page rebase path.
        constexpr bool kCarryKVPhys = kRebaseKSrd && kRebaseVSrd;
        int32_t kv_phys_ring0 = 0;
        int32_t kv_phys_ring1 = 0;

        auto refresh_k_offsets = [&](index_t k_tile_idx, auto is_init) {
            (void)is_init;
            if constexpr(!kIsPaged)
            {
                // Contiguous (THD) K: no page table; logical token is the physical
                // row, so the offset collapses to logical_token * row_stride.
                if constexpr(decltype(is_init)::value)
                {
                    static_for<0, KNRepeat, 1>{}([&](auto i) {
                        const index_t logical_token =
                            split_token_offset + k_tile_idx * kPageBlockSize + k_thread_n_pos +
                            static_cast<index_t>(i.value) * KY0_step_N;
                        k_page_offsets(i) =
                            static_cast<long_index_t>(logical_token) * k_row_stride;
                    });
                }
                else
                {
                    // Steady state: k_tile_idx advances by +1/tile, so every repeat's
                    // byte offset moves by the same loop-invariant kPageBlockSize*
                    // row_stride. Incrementing avoids the per-tile 64-bit address math.
                    const long_index_t k_tile_stride =
                        static_cast<long_index_t>(kPageBlockSize) *
                        static_cast<long_index_t>(k_row_stride);
                    static_for<0, KNRepeat, 1>{}(
                        [&](auto i) { k_page_offsets(i) += k_tile_stride; });
                }
            }
            else if constexpr(kRebaseKSrd)
            {
                // Wave-uniform element offset of the tile's first token, folded into
                // the SRD base. readfirstlane forces uniformity for the SGPR operand.
                const index_t tile_base_token =
                    split_token_offset + k_tile_idx * kPageBlockSize;
                const int32_t base_page =
                    __builtin_amdgcn_readfirstlane(tile_base_token / kPageSize);
                const int32_t phys_page = __builtin_amdgcn_readfirstlane(
                    block_tables_lds[base_page - lds_window_base]);
                // Publish for the staggered V refresh (kCarryKVPhys); the parity
                // branch is wave-uniform so the ring stays in SGPRs.
                if constexpr(kCarryKVPhys)
                {
                    if(k_tile_idx & 1)
                        kv_phys_ring1 = phys_page;
                    else
                        kv_phys_ring0 = phys_page;
                }
                k_srd_base_offset =
                    (static_cast<long_index_t>(phys_page) * kPageSize +
                     (tile_base_token - static_cast<long_index_t>(base_page) * kPageSize)) *
                    k_row_stride;
                // Loop-invariant per-lane within-tile offset (hoisted by the compiler).
                static_for<0, KNRepeat, 1>{}([&](auto i) {
                    constexpr index_t ii = i.value;
                    k_page_offsets(i) =
                        (static_cast<long_index_t>(ii) * KY0_step_N + k_thread_n_pos) *
                        k_row_stride;
                });
            }
            else if constexpr(kScalarPromoteKPageIdx && kDedupPages)
            {
                // Multi-page tile (single-page took the rebase branch): one uniform
                // readfirstlane for the first page; per-issue page is base_page + a
                // compile-time relative offset, so no further readfirstlane.
                const index_t tile_base_token =
                    split_token_offset + k_tile_idx * kPageBlockSize;
                const int32_t base_page =
                    __builtin_amdgcn_readfirstlane(tile_base_token / kPageSize);
                int32_t phys_page = block_tables_lds[base_page - lds_window_base];
                static_for<0, KNRepeat, 1>{}([&](auto i) {
                    constexpr index_t ii = i.value;
                    constexpr index_t grp = (ii * KY0_step_N) / kPageSize;
                    // Re-read phys_page only when this issue crosses a page boundary
                    // (compile-time); otherwise reuse the VGPR.
                    if constexpr(ii > 0)
                    {
                        constexpr index_t grp_prev = ((ii - 1) * KY0_step_N) / kPageSize;
                        if constexpr(grp != grp_prev)
                            phys_page =
                                block_tables_lds[base_page + grp - lds_window_base];
                    }
                    const index_t logical_token =
                        tile_base_token + ii * KY0_step_N + k_thread_n_pos;
                    const index_t within_page =
                        logical_token - (base_page + grp) * kPageSize;
                    k_page_offsets(i) =
                        (static_cast<long_index_t>(phys_page) * kPageSize + within_page) *
                        k_row_stride;
                });
            }
            else if constexpr(kScalarPromoteKPageIdx)
            {
                static_for<0, KNRepeat, 1>{}([&](auto i) {
                    // Compute the uniform per-`i` base in scalar; force the
                    // resulting page-table index into an SGPR. Tier 2 reads
                    // the phys_page from the LDS cache populated above (one
                    // ds_read_b32 broadcast per warp); Tier 0 falls back to
                    // the s_load_dword path when the cache is absent.
                    const index_t i_base_token = split_token_offset +
                                                 k_tile_idx * kPageBlockSize +
                                                 static_cast<index_t>(i.value) * KY0_step_N;
                    const int32_t i_base_page  = __builtin_amdgcn_readfirstlane(
                        i_base_token / page_size);
                    const int32_t phys_page    = block_tables_lds[i_base_page - lds_window_base];
                    const index_t logical_token = i_base_token + k_thread_n_pos;
                    const index_t within_page   = logical_token - i_base_page * page_size;
                    k_page_offsets(i) =
                        (static_cast<long_index_t>(phys_page) * page_size + within_page) *
                        k_row_stride;
                });
            }
            else
            {
                static_for<0, KNRepeat, 1>{}([&](auto i) {
                    // Per-lane path; phys_page read from the LDS cache under
                    // kFallbackUsesLdsK, else from global.
                    const index_t logical_token = split_token_offset +
                                                  k_tile_idx * kPageBlockSize + k_thread_n_pos +
                                                  static_cast<index_t>(i.value) * KY0_step_N;
                    const index_t logical_page  = logical_token / page_size;
                    const index_t within_page   = logical_token - logical_page * page_size;
                    const index_t phys_page =
                        kFallbackUsesLdsK
                            ? block_tables_lds[logical_page - lds_window_base]
                            : block_tables_ptr_[block_table_offset + logical_page];
                    k_page_offsets(i) =
                        (static_cast<long_index_t>(phys_page) * page_size + within_page) *
                        k_row_stride;
                });
            }
        };
        auto refresh_v_offsets = [&](index_t v_tile_idx, auto is_init) {
            (void)is_init;
            if constexpr(!kIsPaged)
            {
                // Contiguous (THD) V — see refresh_k_offsets.
                if constexpr(decltype(is_init)::value)
                {
                    static_for<0, VNRepeat, 1>{}([&](auto i) {
                        const index_t logical_token =
                            split_token_offset + v_tile_idx * kPageBlockSize + v_thread_n_pos +
                            static_cast<index_t>(i.value) * VY0_step_N;
                        v_page_offsets(i) =
                            static_cast<long_index_t>(logical_token) * v_row_stride;
                    });
                }
                else
                {
                    const long_index_t v_tile_stride =
                        static_cast<long_index_t>(kPageBlockSize) *
                        static_cast<long_index_t>(v_row_stride);
                    static_for<0, VNRepeat, 1>{}(
                        [&](auto i) { v_page_offsets(i) += v_tile_stride; });
                }
            }
            else if constexpr(kRebaseVSrd)
            {
                // Single-page SRD rebase — see refresh_k_offsets.
                const index_t tile_base_token =
                    split_token_offset + v_tile_idx * kPageBlockSize;
                const int32_t base_page =
                    __builtin_amdgcn_readfirstlane(tile_base_token / kPageSize);
                // Reuse the phys_page K broadcast for this tile (kCarryKVPhys) to
                // elide V's own read; else read it directly.
                const int32_t phys_page = [&]() -> int32_t {
                    if constexpr(kCarryKVPhys)
                        return (v_tile_idx & 1) ? kv_phys_ring1 : kv_phys_ring0;
                    else
                        return __builtin_amdgcn_readfirstlane(
                            block_tables_lds[base_page - lds_window_base]);
                }();
                v_srd_base_offset =
                    (static_cast<long_index_t>(phys_page) * kPageSize +
                     (tile_base_token - static_cast<long_index_t>(base_page) * kPageSize)) *
                    v_row_stride;
                static_for<0, VNRepeat, 1>{}([&](auto i) {
                    constexpr index_t ii = i.value;
                    v_page_offsets(i) =
                        (static_cast<long_index_t>(ii) * VY0_step_N + v_thread_n_pos) *
                        v_row_stride;
                });
            }
            else if constexpr(kScalarPromoteVPageIdx && kDedupPages)
            {
                // Multi-page only (single-page took the rebase branch above);
                // see refresh_k_offsets for the dedup rationale.
                const index_t tile_base_token =
                    split_token_offset + v_tile_idx * kPageBlockSize;
                const int32_t base_page =
                    __builtin_amdgcn_readfirstlane(tile_base_token / kPageSize);
                int32_t phys_page = block_tables_lds[base_page - lds_window_base];
                static_for<0, VNRepeat, 1>{}([&](auto i) {
                    constexpr index_t ii = i.value;
                    constexpr index_t grp = (ii * VY0_step_N) / kPageSize;
                    if constexpr(ii > 0)
                    {
                        constexpr index_t grp_prev = ((ii - 1) * VY0_step_N) / kPageSize;
                        if constexpr(grp != grp_prev)
                            phys_page =
                                block_tables_lds[base_page + grp - lds_window_base];
                    }
                    const index_t logical_token =
                        tile_base_token + ii * VY0_step_N + v_thread_n_pos;
                    const index_t within_page =
                        logical_token - (base_page + grp) * kPageSize;
                    v_page_offsets(i) =
                        (static_cast<long_index_t>(phys_page) * kPageSize + within_page) *
                        v_row_stride;
                });
            }
            else if constexpr(kScalarPromoteVPageIdx)
            {
                static_for<0, VNRepeat, 1>{}([&](auto i) {
                    const index_t i_base_token = split_token_offset +
                                                 v_tile_idx * kPageBlockSize +
                                                 static_cast<index_t>(i.value) * VY0_step_N;
                    const int32_t i_base_page  = __builtin_amdgcn_readfirstlane(
                        i_base_token / page_size);
                    // Window-relative index; see K-path comment for rationale.
                    const int32_t phys_page    = block_tables_lds[i_base_page - lds_window_base];
                    const index_t logical_token = i_base_token + v_thread_n_pos;
                    const index_t within_page   = logical_token - i_base_page * page_size;
                    v_page_offsets(i) =
                        (static_cast<long_index_t>(phys_page) * page_size + within_page) *
                        v_row_stride;
                });
            }
            else
            {
                static_for<0, VNRepeat, 1>{}([&](auto i) {
                    const index_t logical_token = split_token_offset +
                                                  v_tile_idx * kPageBlockSize + v_thread_n_pos +
                                                  static_cast<index_t>(i.value) * VY0_step_N;
                    const index_t logical_page  = logical_token / page_size;
                    const index_t within_page   = logical_token - logical_page * page_size;
                    const index_t phys_page =
                        kFallbackUsesLdsV
                            ? block_tables_lds[logical_page - lds_window_base]
                            : block_tables_ptr_[block_table_offset + logical_page];
                    v_page_offsets(i) =
                        (static_cast<long_index_t>(phys_page) * page_size + within_page) *
                        v_row_stride;
                });
            }
        };

        refresh_k_offsets(k_block_idx, std::true_type{});
        refresh_v_offsets(v_block_idx, std::true_type{});

        auto k_view = k_dram_block_window_tmp.get_bottom_tensor_view();
        auto v_view = v_dram_block_window_tmp.get_bottom_tensor_view();

        // Single-page SRD rebase: fold the wave-uniform per-tile page offset into the
        // SRD base (rebased per tile in K/V_mem_load), leaving only the loop-invariant
        // within-tile offset in the scatter array. Stash the pool base so the rebase
        // recomputes base = pool_base + U(tile); buffer_size_ stays at full extent.
        [[maybe_unused]] auto* const k_pool_base = k_view.get_buffer_view().p_data_;
        [[maybe_unused]] auto* const v_pool_base = v_view.get_buffer_view().p_data_;
        if constexpr(kRebaseKSrd)
            k_view.get_buffer_view().p_data_ = k_pool_base + k_srd_base_offset;
        if constexpr(kRebaseVSrd)
            v_view.get_buffer_view().p_data_ = v_pool_base + v_srd_base_offset;

        auto k_dram_window =
            make_tile_scatter_gather(k_view,
                                     k_dram_block_window_tmp.get_window_lengths(),
                                     {0, 0},
                                     k_dist,
                                     k_page_offsets);
        k_dram_window.init_raw();

        // In the single-page rebase regime V's per-lane scatter array is bit-identical
        // to K's (the per-tile divergence is the wave-uniform SRD base, already folded
        // in), so feed the same array to both windows to coalesce page_idx_ storage.
        constexpr bool kShareKVScatter =
            kRebaseKSrd && kRebaseVSrd && (KNRepeat == VNRepeat);
        auto v_dram_window = make_tile_scatter_gather(
            v_view,
            v_dram_block_window_tmp.get_window_lengths(),
            {0, 0},
            v_dist,
            [&]() -> const auto& {
                if constexpr(kShareKVScatter)
                    return k_page_offsets;
                else
                    return v_page_offsets;
            }());
        v_dram_window.init_raw();

        // prefetch K tile
        constexpr index_t k0_loops = 1;
        constexpr index_t k1_loops = 1;
        static_assert(1 == k0_loops);
        static_assert(1 == k1_loops);
        // static_assert(kPageBlockSize == kHeadDimPadded);

        constexpr index_t NumWarpGroups = Problem::kBlockSize / Policy::NumThreadPerWarpGroup;
        static_assert(NumWarpGroups == 1 || NumWarpGroups == 2);

        // Conditional (skipped) online-softmax rescale: FA4 prefill (2 WG, VALU-
        // bound) only; decode (1 WG, memory-bound) keeps the always-rescale path.
        constexpr bool kCondRescale = (NumWarpGroups == 2);

        // FA4 overlap forbids any block-wide s_barrier inside one group's softmax
        // phase (the matrix group never reaches it -> deadlock). The only risk is the
        // FP8 P-tile QK-C->PV-A relayout in fmha_alu1: 32x32 tiles use a barrier-free
        // path (K=16 within-wave permute, K=64 cvt-only); the block-wide LDS roundtrip
        // is only on the m16 tier, which is always 1-WG (excluded by the guard).
        using Gemm1WarpTileFA4 = typename UnifiedAttentionShape::Gemm1WarpTile;
        constexpr bool kFP8RelayoutWithinWave =
            (Gemm1WarpTileFA4::at(number<0>{}) == 32) &&
            (Gemm1WarpTileFA4::at(number<1>{}) == 32) &&
            (Gemm1WarpTileFA4::at(number<2>{}) == 16 ||
             Gemm1WarpTileFA4::at(number<2>{}) == 64);
        // FA4 is the only 2-WG prefill pipeline; the static_assert below pins that
        // every 2-WG instance is FA4-eligible (fail loudly otherwise).
        constexpr bool kFA4 = (NumWarpGroups == 2) &&
                              (!std::is_same_v<PDataType, fp8_t> || kFP8RelayoutWithinWave);
        static_assert(NumWarpGroups == 1 || kFA4,
                      "2-warp-group UA instances must be FA4-capable (32x32x16 FP8 P "
                      "relayout); the legacy ping-pong baseline was removed.");

        constexpr int K_mem_su_ld_insts = k_dram_window.get_num_of_access();
        constexpr int V_mem_su_ld_insts = v_dram_window.get_num_of_access();

        // Two async load paths, dispatched on the wave-uniform overflow flag:
        //   false: buffer_load_dword_lds (4 GB-capped SRD, int32 voffsets).
        //   true:  global_load_lds, per-lane 64-bit bases (lifts 4 GB, slower).

        // The post-load refresh prepares page_offsets for the NEXT tile. On the last
        // per-split tile no load follows; an unconditional refresh would read one past
        // the last valid logical_page (OOB block_tables fault). Gating on the per-split
        // iter count leaves the table stale on the final iter (harmless).
        const index_t num_iters_per_split = num_total_loop - num_blocks_start;

        // Slide the LDS page-table window forward. MUST be called by ALL waves at
        // a CTA convergence barrier, before the iteration's prefetch/refresh
        // consumes the window. The predicate is wave-uniform (tile bookkeeping +
        // compile-time geometry) so the two internal s_barriers stay matched
        // across warp groups. When the whole per-split window already fits in LDS
        // it is a never-taken branch (bit-identical to the single-shot load).
        auto slide_page_table = [&]() {
            if constexpr(kUsePageTableLds)
            {
                if(split_window_pages <= kPageTableLdsEntries)
                    return;
                const index_t lo_tile = k_block_idx < v_block_idx ? k_block_idx : v_block_idx;
                // +2: the next-tile prefetch (refresh) issued after this barrier
                // advances the bookkeeping index by one before reading the window.
                const index_t hi_tile =
                    (k_block_idx > v_block_idx ? k_block_idx : v_block_idx) + 2;
                const index_t need_lo = static_cast<index_t>(
                    (split_token_offset + static_cast<long_index_t>(lo_tile) * kPageBlockSize) /
                    page_size);
                const index_t need_hi = static_cast<index_t>(
                    (split_token_offset + static_cast<long_index_t>(hi_tile) * kPageBlockSize +
                     (page_size - 1)) /
                    page_size);
                if(need_lo < lds_window_base ||
                   need_hi >= lds_window_base + kPageTableLdsEntries)
                {
                    __builtin_amdgcn_s_barrier(); // all waves done with the old window
                    const index_t avail = split_end_page - need_lo;
                    const index_t cnt   = avail < kPageTableLdsEntries
                                              ? avail
                                              : static_cast<index_t>(kPageTableLdsEntries);
                    const index_t tid = get_thread_local_1d_id();
                    for(index_t i = tid; i < cnt; i += Problem::kBlockSize)
                        block_tables_lds[i] = block_tables_ptr_[block_table_offset + need_lo + i];
                    lds_window_base = need_lo;
                    s_waitcnt_lgkmcnt<0>();
                    __builtin_amdgcn_s_barrier(); // publish the new window
                }
            }
        };

        auto K_mem_load = [&](auto k_lds_write_idx) {
            // Cooperative K async load: all NumWarps waves load their shard.
            if(cache_ptr_int32_overflow_possible)
                async_load_tile_raw_long(k_lds_window_store(k_lds_write_idx), k_dram_window);
            else
                async_load_tile_raw(k_lds_window_store(k_lds_write_idx), k_dram_window);
            k_block_idx++;
            if(k_block_idx < num_iters_per_split)
            {
                refresh_k_offsets(k_block_idx, std::false_type{});
                if constexpr(kRebaseKSrd)
                    // Per-tile SRD rebase: only the wave-uniform base moves.
                    k_dram_window.rebase_buffer_base(k_pool_base + k_srd_base_offset);
                else
                    k_dram_window.update_page_idx(k_page_offsets);
            }
        };

        auto V_mem_load = [&](auto v_lds_write_idx) {
            if(cache_ptr_int32_overflow_possible)
                async_load_tile_raw_long(v_lds_window_store(v_lds_write_idx), v_dram_window);
            else
                async_load_tile_raw(v_lds_window_store(v_lds_write_idx), v_dram_window);
            v_block_idx++;
            if(v_block_idx < num_iters_per_split)
            {
                refresh_v_offsets(v_block_idx, std::false_type{});
                if constexpr(kRebaseVSrd)
                    v_dram_window.rebase_buffer_base(v_pool_base + v_srd_base_offset);
                else
                    v_dram_window.update_page_idx(v_page_offsets);
            }
        };

        auto K_lds_load = [&](auto k_lds_read_idx) {
            kv_tile.k_tile = load_tile(k_lds_window_load(k_lds_read_idx));
        };

        auto V_lds_load = [&](auto v_lds_read_idx) {
            kv_tile.v_tile = load_tile_transpose(v_lds_window_load(v_lds_read_idx));
        };

        decltype(m) m_old;
        SMPLComputeDataType o_acc_scale; // rescale o_acc in fmha_alu1() & fmha_alu_D_upd()
        // sp_delta follows sp: single slot for the kv128 tile, double otherwise.
        struct sp_delta_holder_t
        {
            decltype(sp(number<0>{}).sp_compute) d_;
            CK_TILE_DEVICE constexpr decltype(d_)& operator()(index_t) { return d_; }
        };
        std::conditional_t<kUseSingleSp,
                           sp_delta_holder_t,
                           statically_indexed_array<decltype(sp(number<0>{}).sp_compute), 2>>
            sp_delta;

        auto fmha_alu0 = [&](auto sp_reg_idx) {
            m_old = m; // m{j-1}
            static_assert(m.thread_buf_.size() == 1,
                          "assuming that each thread holds 1 rowmax value");
            auto m_latest = block_tile_reduce<SMPLComputeDataType>(
                sp(sp_reg_idx).sp_compute, sequence<1>{}, f_max, m.thread_buf_[0]);
#if defined(__gfx950__)
            if constexpr(kWarpGemmM == 32)
            {
                int32x2_t swapped_regs =
                    __builtin_amdgcn_permlane32_swap(bit_cast<int32_t>(m_latest.thread_buf_[0]),
                                                     bit_cast<int32_t>(m_latest.thread_buf_[0]),
                                                     false,
                                                     false);
                m_latest.thread_buf_[0] = f_max(bit_cast<SMPLComputeDataType>(swapped_regs.x),
                                                bit_cast<SMPLComputeDataType>(swapped_regs.y));
            }
            else
            {
                block_tile_reduce_sync(m_latest, f_max, bool_constant<false>{});
            }
#else
            block_tile_reduce_sync(m_latest, f_max, bool_constant<false>{});
#endif
            m = m_latest;
            if constexpr(kCondRescale)
            {
                // Wave-uniformly decide whether the running max pulled more than τ=8
                // ahead of the committed max; ballot the per-lane predicate so the
                // downstream rescale guard is a scalar s_cbranch.
                const bool nr_local =
                    (scale_s * (m.thread_buf_[0] - m_commit.thread_buf_[0])) > 8.0f;
                need_rescale                = (__builtin_amdgcn_ballot_w64(nr_local) != 0ull);
                m_commit_old.thread_buf_[0] = m_commit.thread_buf_[0];
                if(need_rescale)
                {
                    m_commit.thread_buf_[0] = m.thread_buf_[0];
                }
            }
            // Score-shift base: committed max (conditional path), else running max.
            auto& m_shift = kCondRescale ? m_commit : m;

            constexpr auto p_spans =
                std::decay_t<decltype(sp(sp_reg_idx).sp_compute)>::get_distributed_spans();
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx        = make_tuple(idx0, idx1);
                    sp_delta(sp_reg_idx)(i_j_idx) = detail::fma_impl_vsv(
                        sp(sp_reg_idx).sp_compute(i_j_idx), scale_s, -scale_s * m_shift(i_j_idx));
                });
            });
        };

        auto fmha_alu1 = [&](auto sp_reg_idx) {
            constexpr auto p_spans =
                std::decay_t<decltype(sp(sp_reg_idx).sp_compute)>::get_distributed_spans();
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    sp(sp_reg_idx).sp_compute(i_j_idx) =
                        ck_tile::exp2(sp_delta(sp_reg_idx)(i_j_idx));
                });
            });

            auto rowsum_p = block_tile_reduce<SMPLComputeDataType>(
                sp(sp_reg_idx).sp_compute,
                sequence<1>{},
                f_sum,
                SMPLComputeDataType{0}); // rowsum(Pcompute{j})
            static_assert(rowsum_p.thread_buf_.size() == 1,
                          "assuming that each thread holds 1 rowsum value");
#if defined(__gfx950__)
            if constexpr(kWarpGemmM == 32)
            {
                int32x2_t swapped_regs =
                    __builtin_amdgcn_permlane32_swap(bit_cast<int32_t>(rowsum_p.thread_buf_[0]),
                                                     bit_cast<int32_t>(rowsum_p.thread_buf_[0]),
                                                     false,
                                                     false);
                rowsum_p.thread_buf_[0] = f_sum(bit_cast<SMPLComputeDataType>(swapped_regs.x),
                                                bit_cast<SMPLComputeDataType>(swapped_regs.y));
            }
            else
            {
                block_tile_reduce_sync(rowsum_p, f_sum, bool_constant<false>{});
            }
#else
            block_tile_reduce_sync(rowsum_p, f_sum, bool_constant<false>{});
#endif

            // l{j}: the final add is written in inline asm (add_impl_vv) to anchor
            // these instructions here, since 'l' is first consumed much later.
            constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
            sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                // Denominator rescale: committed-max delta (==1 no-op on non-committing
                // tiles) for the conditional path; running-max delta otherwise.
                const auto tmp =
                    kCondRescale
                        ? ck_tile::exp2(scale_s * (m_commit_old[i_idx] - m_commit[i_idx]))
                        : ck_tile::exp2(scale_s * (m_old[i_idx] - m[i_idx]));
                l(i_idx) = detail::add_impl_vv(tmp * l[i_idx], rowsum_p[i_idx]);
            });

            // update partial o_acc [0, fmha_alu_D_reg_cnt). Skip unless this tile
            // committed (o_acc_scale/need_rescale deferred from fmha_alu_D_upd).
            if(!kCondRescale || need_rescale)
                static_for<0, fmha_alu_D_reg_cnt, 1>{}([&](auto idx) {
                    o_acc.thread_buf_[idx] =
                        detail::mul_impl_vv(o_acc.thread_buf_[idx], o_acc_scale);
                });

            // Conversions written in inline asm to anchor them here ('p' is consumed
            // much later, so the compiler would otherwise sink them).
            static_assert(sp(sp_reg_idx).p.thread_buf_.size() % 2 == 0);
            if constexpr(std::is_same_v<PDataType, fp8_t>)
            {
                // FP8 P packing for the PV gemm. The fp32->fp8 cvt CHAINS two
                // cvt_pk_fp8_f32 per 4 lanes (the second uses the first's result as
                // `old`) to match cast_tile_pk_fp8_fp32 byte-for-byte.
                static_assert(sp(sp_reg_idx).p.thread_buf_.size() % 4 == 0,
                              "fp8 P conversion expects packs of 4 fp32 lanes per "
                              "thread; widen the warp gemm M distribution if this "
                              "trips.");

                // FP8 QK-C -> PV-A re-layout (the union aliases the two diverging
                // FP8 layouts). Two strategies:
                //   (A) paired-lane (lane^32) in-register swap — one permlane32_swap
                //       (gfx950) / ds_bpermute (gfx942) per K-iter, no LDS/barrier.
                //       Covers the 32x32 tiles (K=16 and K=64 share the swap pattern).
                //   (B) LDS roundtrip for any other shape (16x16x32 m16); ~2-3x slower.
                using PVWarpTile = typename UnifiedAttentionShape::Gemm1WarpTile;
                if constexpr(PVWarpTile::at(number<0>{}) == 32 &&
                             PVWarpTile::at(number<1>{}) == 32 &&
                             (PVWarpTile::at(number<2>{}) == 16 ||
                              PVWarpTile::at(number<2>{}) == 64))
                {
                    // ---- (A) Fused cvt + paired-lane swap (32x32x16 / 32x32x64).
                    // Per 8-fp8 chunk: cvt 8 fp32 -> 2 packed uint32 (lo/hi), swap
                    // the "bad" pack to lane^32, write back 8 fp8 bytes.
                    static_assert(sp(sp_reg_idx).p.thread_buf_.size() % 8 == 0,
                                  "FP8 32x32 (K=16/K=64) cross-lane permute expects "
                                  "PV per-thread buffer in chunks of 8 fp8.");

                    // gfx950 does the l^32 swap in one VALU op; the lane-id math
                    // below is only for the ds_bpermute fallback (gfx942).
#if !defined(__gfx950__)
                    const int lane_id     = ck_tile::get_lane_id();
                    const int paired_addr = (lane_id ^ 32) << 2; // bytes
                    const bool is_sub_0   = (lane_id & 32) == 0;
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
                    int dummy_old;
                    static_for<0, sp(sp_reg_idx).p.thread_buf_.size(), 8>{}([&](auto k_base) {
                        auto& p  = sp(sp_reg_idx).p;
                        auto& sc = sp(sp_reg_idx).sp_compute;

                        const float a = p_compute_element_func(sc.thread_buf_[k_base + 0]);
                        const float b = p_compute_element_func(sc.thread_buf_[k_base + 1]);
                        const float c = p_compute_element_func(sc.thread_buf_[k_base + 2]);
                        const float d = p_compute_element_func(sc.thread_buf_[k_base + 3]);
                        const float e = p_compute_element_func(sc.thread_buf_[k_base + 4]);
                        const float f = p_compute_element_func(sc.thread_buf_[k_base + 5]);
                        const float g = p_compute_element_func(sc.thread_buf_[k_base + 6]);
                        const float h = p_compute_element_func(sc.thread_buf_[k_base + 7]);

                        const uint32_t lo_tmp =
                            __builtin_amdgcn_cvt_pk_fp8_f32(a, b, dummy_old, /*hi=*/false);
                        const uint32_t lo_pack =
                            __builtin_amdgcn_cvt_pk_fp8_f32(c, d, lo_tmp, /*hi=*/true);
                        const uint32_t hi_tmp =
                            __builtin_amdgcn_cvt_pk_fp8_f32(e, f, dummy_old, /*hi=*/false);
                        const uint32_t hi_pack =
                            __builtin_amdgcn_cvt_pk_fp8_f32(g, h, hi_tmp, /*hi=*/true);

#if defined(__gfx950__)
                        // permlane32_swap exchanges operand0's high half with
                        // operand1's low half, so (lo_pack, hi_pack) returns
                        // {out_lo, out_hi} directly for every lane.
                        const auto swapped =
                            __builtin_amdgcn_permlane32_swap(lo_pack, hi_pack,
                                                             /*fi=*/false,
                                                             /*bound_ctrl=*/false);
                        const uint32_t out_lo = swapped[0];
                        const uint32_t out_hi = swapped[1];
#else
                        // Issue ds_bpermute as early as possible so its LDS-DMA
                        // latency overlaps with the byte writes below (and with
                        // the next K-chunk's cvts after this iter unrolls).
                        const uint32_t own_bad = is_sub_0 ? hi_pack : lo_pack;
                        const uint32_t recv    = __builtin_amdgcn_ds_bpermute(
                            paired_addr, static_cast<int>(own_bad));

                        const uint32_t out_lo = is_sub_0 ? lo_pack : recv;
                        const uint32_t out_hi = is_sub_0 ? recv    : hi_pack;
#endif

                        p.thread_buf_[k_base + 0] =
                            bit_cast<fp8_t>(static_cast<fp8_raw_t>((out_lo >>  0) & 0xFFu));
                        p.thread_buf_[k_base + 1] =
                            bit_cast<fp8_t>(static_cast<fp8_raw_t>((out_lo >>  8) & 0xFFu));
                        p.thread_buf_[k_base + 2] =
                            bit_cast<fp8_t>(static_cast<fp8_raw_t>((out_lo >> 16) & 0xFFu));
                        p.thread_buf_[k_base + 3] =
                            bit_cast<fp8_t>(static_cast<fp8_raw_t>((out_lo >> 24) & 0xFFu));
                        p.thread_buf_[k_base + 4] =
                            bit_cast<fp8_t>(static_cast<fp8_raw_t>((out_hi >>  0) & 0xFFu));
                        p.thread_buf_[k_base + 5] =
                            bit_cast<fp8_t>(static_cast<fp8_raw_t>((out_hi >>  8) & 0xFFu));
                        p.thread_buf_[k_base + 6] =
                            bit_cast<fp8_t>(static_cast<fp8_raw_t>((out_hi >> 16) & 0xFFu));
                        p.thread_buf_[k_base + 7] =
                            bit_cast<fp8_t>(static_cast<fp8_raw_t>((out_hi >> 24) & 0xFFu));
                    });
#pragma clang diagnostic pop
                }
                else
                {
                    // ---- (B) LDS roundtrip: cvt into sp.p, store under QK-C, barrier,
                    // load back under PV-A, copy into sp.p for the gemm_1 call.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
                    int dummy_old;
                    static_for<0, sp(sp_reg_idx).p.thread_buf_.size(), 4>{}([&](auto idx) {
                        const float a = p_compute_element_func(sp(sp_reg_idx).sp_compute.thread_buf_[idx + 0]);
                        const float b = p_compute_element_func(sp(sp_reg_idx).sp_compute.thread_buf_[idx + 1]);
                        const float c = p_compute_element_func(sp(sp_reg_idx).sp_compute.thread_buf_[idx + 2]);
                        const float d = p_compute_element_func(sp(sp_reg_idx).sp_compute.thread_buf_[idx + 3]);

                        const uint32_t lo =
                            __builtin_amdgcn_cvt_pk_fp8_f32(a, b, dummy_old, /*hi=*/false);
                        const uint32_t packed =
                            __builtin_amdgcn_cvt_pk_fp8_f32(c, d, lo, /*hi=*/true);
                        sp(sp_reg_idx).p.thread_buf_[idx + 0] =
                            bit_cast<fp8_t>(static_cast<fp8_raw_t>((packed >>  0) & 0xFFu));
                        sp(sp_reg_idx).p.thread_buf_[idx + 1] =
                            bit_cast<fp8_t>(static_cast<fp8_raw_t>((packed >>  8) & 0xFFu));
                        sp(sp_reg_idx).p.thread_buf_[idx + 2] =
                            bit_cast<fp8_t>(static_cast<fp8_raw_t>((packed >> 16) & 0xFFu));
                        sp(sp_reg_idx).p.thread_buf_[idx + 3] =
                            bit_cast<fp8_t>(static_cast<fp8_raw_t>((packed >> 24) & 0xFFu));
                    });
#pragma clang diagnostic pop

                    auto p_qkc = make_static_distributed_tensor<PDataType>(
                        sp(sp_reg_idx).sp_compute.get_tile_distribution());
                    static_assert(
                        decltype(p_qkc.thread_buf_)::size() ==
                            decltype(sp(sp_reg_idx).p.thread_buf_)::size(),
                        "QK-C and PV-A per-thread fp8 buffers must match in size for "
                        "the LDS roundtrip aliasing (holds via the union).");
                    static_for<0, decltype(p_qkc.thread_buf_)::size(), 1>{}(
                        [&](auto i) {
                            p_qkc.thread_buf_[i] = sp(sp_reg_idx).p.thread_buf_[i];
                        });

                    __builtin_amdgcn_s_barrier();
                    store_tile(p_lds_store_window_qkc, p_qkc);
                    __builtin_amdgcn_s_barrier();
                    auto p_pva = load_tile(p_lds_load_window_pva);
                    static_for<0, decltype(p_pva.thread_buf_)::size(), 1>{}(
                        [&](auto i) {
                            sp(sp_reg_idx).p.thread_buf_[i] = p_pva.thread_buf_[i];
                        });
                }
            }
            else
            {
                static_for<0, sp(sp_reg_idx).p.thread_buf_.size(), 2>{}([&](auto idx) {
                    float x = p_compute_element_func(sp(sp_reg_idx).sp_compute.thread_buf_[idx]);
                    float y = p_compute_element_func(sp(sp_reg_idx).sp_compute.thread_buf_[idx + 1]);
                    if constexpr(std::is_same_v<PDataType, fp16_t>)
                    {
                        auto casted                           = detail::cvt_pk_fp16_f32(x, y);
                        sp(sp_reg_idx).p.thread_buf_[idx]     = casted.x;
                        sp(sp_reg_idx).p.thread_buf_[idx + 1] = casted.y;
                    }
                    else
                    {
                        auto casted                           = cvt_pk_bf16_f32(x, y);
                        sp(sp_reg_idx).p.thread_buf_[idx]     = casted.x;
                        sp(sp_reg_idx).p.thread_buf_[idx + 1] = casted.y;
                    }
                });
            }

            // Keep fmha_alu1() at the end of the phase: the surrounding inline asm
            // can interfere with sched_group_barrier() ordering.
        };

        auto gemm = [&](auto sp_reg_idx, auto gemm_idx) {
            if constexpr(gemm_idx == 0)
            {
                clear_tile(sp(sp_reg_idx).sp_compute); // initialize C
                gemm_0(sp(sp_reg_idx).sp_compute,
                       get_slice_tile(q_tile,
                                      sequence<0, (k0_loops - 1) * kHeadDimPadded>{},
                                      sequence<kBlockM, k0_loops * kHeadDimPadded>{}),
                       get_slice_tile(kv_tile.k_tile,
                                      sequence<0, (k0_loops - 1) * kHeadDimPadded>{},
                                      sequence<kPageBlockSize, k0_loops * kHeadDimPadded>{}));
            }
            else
            {
                gemm_1(o_acc,
                       get_slice_tile(sp(sp_reg_idx).p,
                                      sequence<0, (k1_loops - 1) * kPageBlockSize>{},
                                      sequence<kBlockM, k1_loops * kPageBlockSize>{}),
                       get_slice_tile(kv_tile.v_tile,
                                      sequence<0, (k1_loops - 1) * kPageBlockSize>{},
                                      sequence<kHeadDimPadded, k1_loops * kPageBlockSize>{}));
            }
        };

        auto fmha_alu_D_upd = [&] {
            // exp2(0) == 1 on non-committing tiles, so the guarded multiplies are
            // skipped (scalar s_cbranch on the wave-uniform need_rescale).
            o_acc_scale =
                kCondRescale
                    ? ck_tile::exp2(scale_s *
                                    (m_commit_old.thread_buf_[0] - m_commit.thread_buf_[0]))
                    : ck_tile::exp2(scale_s * (m_old.thread_buf_[0] - m.thread_buf_[0]));

            static_assert((o_acc.thread_buf_.size() - fmha_alu_D_reg_cnt) % 2 == 0);

            if(!kCondRescale || need_rescale)
            {
                fp32x2_t pk_o_acc_scale;
                pk_o_acc_scale.x = o_acc_scale;
                pk_o_acc_scale.y = o_acc_scale;

                // Packed v_pk_mul_f32; this call should sit at the end of a phase.
                static_for<fmha_alu_D_reg_cnt, o_acc.thread_buf_.size(), 2>{}([&](auto idx) {
                    fp32x2_t input;
                    input.x = o_acc.thread_buf_[idx];
                    input.y = o_acc.thread_buf_[idx + 1];

                    auto output = detail::pk_mul_f32(input, pk_o_acc_scale);

                    o_acc.thread_buf_[idx]     = output.x;
                    o_acc.thread_buf_[idx + 1] = output.y;
                });
            }
        };

        // Resolve kBlockQ: runtime kBlockM / num_queries_per_kv, or the static
        // kBlockQ when the caller passes 0. Stored once, reused per K-tile mask.
        const index_t kBlockQ_dyn =
            (num_queries_per_kv > 0) ? (kBlockM / num_queries_per_kv) : kBlockQ;

        auto fmha_mask = [&](auto sp_reg_idx) {
            if constexpr(FmhaMask::IsMasking)
            {
                bool need_perpixel_check = mask.IsEdgeTile(q_origin.at(number<0>{}),
                                                           i_total_loops * kPageBlockSize,
                                                           kBlockQ_dyn,
                                                           static_cast<index_t>(kPageBlockSize));
                if(need_perpixel_check)
                {
                    set_tile_if(sp(sp_reg_idx).sp_compute,
                                -numeric<SMPLComputeDataType>::infinity(),
                                [&](auto tile_idx) {
                                    const auto row =
                                        q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                                    const auto col =
                                        i_total_loops * kPageBlockSize + tile_idx.at(number<1>{});
                                    return mask.IsOutOfBound(row, col);
                                });
                }
            }
        };

        // FA4 mask variant: the matrix‖softmax split decouples the masked KV tile
        // from the loop counter i_total_loops, so the absolute KV-tile index is
        // passed explicitly to keep the masked column window aligned with the tile
        // held in sp(sp_reg_idx).
        [[maybe_unused]] auto fmha_mask_at = [&](auto sp_reg_idx, index_t kv_tile_idx) {
            if constexpr(FmhaMask::IsMasking)
            {
                const index_t col_base   = kv_tile_idx * kPageBlockSize;
                bool need_perpixel_check = mask.IsEdgeTile(q_origin.at(number<0>{}),
                                                           col_base,
                                                           kBlockQ_dyn,
                                                           static_cast<index_t>(kPageBlockSize));
                if(need_perpixel_check)
                {
                    set_tile_if(sp(sp_reg_idx).sp_compute,
                                -numeric<SMPLComputeDataType>::infinity(),
                                [&](auto tile_idx) {
                                    const auto row =
                                        q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                                    const auto col = col_base + tile_idx.at(number<1>{});
                                    return mask.IsOutOfBound(row, col);
                                });
                }
            }
        };

        // FA4 matrix‖softmax overlap core loop. Both warp groups run the deferred-PV
        // pipeline cut into two barrier-delimited phases per slot pi:
        //   MATRIX(pi)  : PV(pi) = o_acc += P(pi) @ V(k-1); QK(1-pi) = Q @ K(k)
        //   SOFTMAX(pi) : mask, alu0, D_upd, alu1 on slot 1-pi -> P(1-pi)
        // o_acc rescale lives in SOFTMAX so MATRIX stays pure-matrix. The groups are
        // primed one phase apart (cl_p==0 MATRIX-first, cl_p==1 SOFTMAX-first) so each
        // block barrier hides one group's matrix work under the other's VALU. K/V are
        // prefetched a tile ahead at the 8-warp converged barrier.
        //
        // fa4_sm_tile: absolute KV-tile index of the NEXT tile to mask/softmax,
        // advanced once per softmax. Starts at num_blocks_start (split-KV offset).
        [[maybe_unused]] index_t fa4_sm_tile = num_blocks_start;

        // fp8 prefetches in the matrix phase; bf16/fp16 in the softmax phase
        // (residency still enforced by the next MATRIX's vmcnt drain + barrier).
        constexpr bool kPrefetchInSoftmax = !std::is_same_v<KDataType, fp8_t>;

        [[maybe_unused]] auto core_loop_fa4 = [&](auto cl_p) {
            auto gemm0 = number<0>{};
            auto gemm1 = number<1>{};

            // MATRIX phase: deferred PV(k-1) then QK(k), pure matrix pipe. Both LDS
            // reads live here: V up front (overlaps the lgkmcnt drain), K between the
            // PV and QK MFMAs so its read overlaps the PV MFMA.
            auto fa4_matrix = [&](auto pi) {
                auto pv_sp = pi;               // PV source: P(pi) from prev SOFTMAX
                auto qk_sp = number<1>{} - pi; // QK target slot
                auto k_rd  = number<1>{} - pi;

                // V buf pi was published before a prior barrier, so residency is
                // guaranteed; issued first so its LDS latency overlaps the drain.
                V_lds_load(pi);
                s_waitcnt_lgkmcnt<0>();
                gemm(pv_sp, gemm1);     // o_acc += P(pi) @ V(k-1)
                // K read into its own registers so this ds_read runs on the LSU
                // during the PV MFMA; sched_barriers pin it here.
                __builtin_amdgcn_sched_barrier(0);
                K_lds_load(k_rd);
                __builtin_amdgcn_sched_barrier(0);
                s_waitcnt_lgkmcnt<0>();
                gemm(qk_sp, gemm0); // sp(1-pi).sp_compute = Q @ K(k)
            };

            // SOFTMAX phase on the just-QK'd slot (1-pi): mask, alu0, D_upd, alu1 ->
            // P(1-pi) for the next MATRIX phase's PV.
            auto fa4_softmax = [&](auto pi) {
                auto sm_sp = number<1>{} - pi;
                fmha_mask_at(sm_sp, fa4_sm_tile++);
                fmha_alu0(sm_sp);
                fmha_alu_D_upd();
                fmha_alu1(sm_sp);
            };

            // One KV tile == one MATRIX + one SOFTMAX phase, separated by two block
            // barriers. The tile-k+1 prefetch is issued right after the first barrier
            // (all 8 warps converged) so the cooperative async load covers the full
            // tile. Prefetch targets the buffer the next iteration reads (K→buf[pi],
            // V→buf[1-pi]); the opposite buffers are read this iteration, no alias.
            auto iteration = [&](auto pi) {
                bool result = true;
                auto K_pf = pi;               // next-tile K buffer
                auto V_pf = number<1>{} - pi; // next-tile V buffer

                auto prefetch = [&] {
                    if(i_total_loops + 1 < num_total_loop)
                        K_mem_load(K_pf);
                    V_mem_load(V_pf);
                };

                auto barrier = [] {
                    __builtin_amdgcn_sched_barrier(0); // pin: nothing crosses the
                    __builtin_amdgcn_s_barrier();      // block barrier (keeps the
                    __builtin_amdgcn_sched_barrier(0); // cooperative load converged)
                };

                if constexpr(cl_p == 0)
                {
                    // WG0 : MATRIX(pi) then SOFTMAX(pi). WG0 is the V loader; each
                    // slot drains its V loads (vmcnt<0>) before the barrier so the
                    // barrier publishes its cooperative writes to the next readers.
                    ASM_MARKER("fa4 MATRIX Wave0-3");
                    s_waitcnt_vmcnt<0>(); // V for THIS matrix has arrived -> publish
                    barrier();
                    slide_page_table();
                    if constexpr(!kPrefetchInSoftmax)
                        prefetch();
                    fa4_matrix(pi);

                    // slot B SOFTMAX(pi): no VMEM drain (slot A already published V;
                    // the only outstanding load is the next-tile prefetch no reader
                    // touches yet). The barrier still publishes this phase's LDS.
                    ASM_MARKER("fa4 SOFTMAX Wave0-3");
                    barrier();
                    if constexpr(kPrefetchInSoftmax)
                        prefetch();
                    fa4_softmax(pi);

                    if(num_total_loop <= ++i_total_loops)
                        result = false;
                }
                else
                {
                    // WG1 : SOFTMAX then MATRIX(pi). WG1 is the K loader, primed one
                    // phase ahead of WG0.
                    ASM_MARKER("fa4 SOFTMAX Wave4-7");
                    s_waitcnt_vmcnt<0>(); // K for WG0's matrix has arrived -> publish
                    barrier();
                    slide_page_table();
                    if constexpr(kPrefetchInSoftmax)
                        prefetch();
                    fa4_softmax(number<1>{} - pi);

                    // slot B MATRIX(pi): prefetch from here (not the SOFTMAX slot) so
                    // the load-issue overhead overlaps WG0's softmax. The load lands
                    // a phase later than WG0's, shrinking the hide window.
                    ASM_MARKER("fa4 MATRIX Wave4-7");
                    barrier();
                    if constexpr(!kPrefetchInSoftmax)
                        prefetch();
                    fa4_matrix(pi);

                    if(num_total_loop <= ++i_total_loops)
                        result = false;
                }
                return result;
            };
            return iteration(number<0>{}) && iteration(number<1>{});
        };

        // FA4 deferred-PV epilogue: the final SOFTMAX produced P for a tile whose PV
        // is not yet folded into o_acc. Run that last PV (alu1 already ran).
        [[maybe_unused]] auto fa4_post_process = [&](auto last_pv_sp, auto last_v_buf) {
            s_waitcnt_vmcnt<0>();
            __builtin_amdgcn_s_barrier();
            V_lds_load(last_v_buf);
            s_waitcnt_lgkmcnt<0>();
            gemm(last_pv_sp, /*gemm_idx=*/number<1>{});
        };

        auto fmha_post_process = [&](auto d) {
            auto ps_pi        = number<1>{} - d;
            auto V_lds_rd_idx = ps_pi;

            // Wait for the last V async load before reading LDS. The final iteration
            // never prefetches K, so only V loads are outstanding; drain to vmcnt<0>
            // (a count-specific wait races when V and K issue the same #insts).
            s_waitcnt_vmcnt<0>();
            __builtin_amdgcn_s_barrier();

            V_lds_load(V_lds_rd_idx);
            fmha_alu1(ps_pi);

            s_waitcnt_lgkmcnt<0>();

            auto xdl_SP_p23_reg_idx = ps_pi;
            gemm(xdl_SP_p23_reg_idx, /*gemm_idx=*/number<1>{});
        };

        // pre-stage
        {
            ASM_MARKER("before pre-stage");
            // (1) load K0 to LDS & VGPR
            K_mem_load(number<0>{}); // mem_K0

            s_waitcnt_vmcnt<0>();
            __builtin_amdgcn_s_barrier();

            K_lds_load(number<0>{}); // lds_K0

            s_waitcnt_lgkmcnt<0>();
            __builtin_amdgcn_s_barrier();

            // (2) prefetch K1 and V0 to LDS in parallel with GEMM0
            if(1 < num_total_loop)
            {
                K_mem_load(number<1>{}); // mem_K1
            }
            V_mem_load(number<0>{}); // mem_V0

            // (3) mfma (Q*K0) + softmax
            gemm(number<0>{}, /*gemm_idx=*/number<0>{});

            // FA4 leaves sp(0) at raw QK(0); each warp group does tile-0's softmax
            // itself (softmax-first folds it into iter 0, matrix-first primes it
            // below). The K2 prefetch is skipped: the FA4 loop prefetches exactly
            // one tile ahead, so iter 0 must issue K2 (else it clobbers K2 with K3).
            if constexpr(kFA4)
            {
                ++i_total_loops;
                if(num_total_loop <= i_total_loops)
                {
                    goto label_main_loops_exit;
                }
                ASM_MARKER("end pre-stage (FA4)");
            }
            else
            {
                fmha_mask(number<0>{});
                fmha_alu0(number<0>{});
                fmha_alu_D_upd();

                ++i_total_loops;
                if(num_total_loop <= i_total_loops)
                {
                    goto label_main_loops_exit;
                }

                // K2 prefetch into buf0 (the freed K0 slot).
                if(2 < num_total_loop)
                {
                    K_mem_load(number<0>{}); // mem_K2

                    s_waitcnt_vmcnt<K_mem_su_ld_insts + V_mem_su_ld_insts>();
                    __builtin_amdgcn_s_barrier();
                }

                ASM_MARKER("end pre-stage");
            }
        }

        if(1 < num_total_loop)
        {
            if constexpr(NumWarpGroups == 1)
            {
                // --- Single warp group: serial pipeline with async prefetch ---
                // After pre-stage: sp(0) holds QK(0) (alu0/D_upd done, alu1 pending),
                // V0 loading to buf 0, K1 in buf 1, K2 loading to buf 0.

                // Step 1: consume V0, K1 -> produce PV(0), QK(1)
#if UA_DECODE_INTERLEAVE
                // Issue the V1 prefetch BEFORE the consume-wait, then drain only
                // down to the two newest tiles (K2 from pre-stage + V1) so both
                // stream HBM->LDS under this step's compute. V0/K1 (older) are
                // guaranteed complete by the partial threshold.
                slide_page_table();
                V_mem_load(number<1>{}); // prefetch V1 -> buf 1
                s_waitcnt_vmcnt<K_mem_su_ld_insts + V_mem_su_ld_insts>();
                __builtin_amdgcn_s_barrier();
#else
                s_waitcnt_vmcnt<0>();
                __builtin_amdgcn_s_barrier();

                slide_page_table(); // keep the page-table window covering the next prefetch
                V_mem_load(number<1>{}); // prefetch V1 -> buf 1 (overlaps with compute)
#endif

                V_lds_load(number<0>{}); // V0 from LDS -> kv_tile.v_tile
                s_waitcnt_lgkmcnt<0>();
                fmha_alu1(number<0>{}); // finalize sp(0) -> P(0)
                gemm(number<0>{}, /*gemm_idx=*/number<1>{}); // PV: P(0)*V0

                K_lds_load(number<1>{}); // K1 from LDS -> kv_tile.k_tile
                s_waitcnt_lgkmcnt<0>();

                gemm(number<1>{}, /*gemm_idx=*/number<0>{}); // QK: Q*K1 -> sp(1)
                fmha_mask(number<1>{});
                fmha_alu0(number<1>{});
                fmha_alu_D_upd();
                i_total_loops++;

                while(i_total_loops < num_total_loop)
                {
                    // Even step: V from buf 1, K from buf 0, QK -> sp(0)
#if UA_DECODE_INTERLEAVE
                    // Issue next K(buf1)/V(buf0) BEFORE the wait so they overlap
                    // this step's compute; drain only to the just-issued tile.
                    // The consumed tiles (V buf1, K buf0) are older -> complete.
                    slide_page_table();
                    if(i_total_loops + 1 < num_total_loop)
                    {
                        K_mem_load(number<1>{}); // next K -> K buf 1
                        V_mem_load(number<0>{}); // next V -> V buf 0
                        s_waitcnt_vmcnt<K_mem_su_ld_insts + V_mem_su_ld_insts>();
                    }
                    else
                    {
                        // Terminal step: no real next tile, just drain.
                        s_waitcnt_vmcnt<0>();
                    }
                    __builtin_amdgcn_s_barrier();
#else
                    s_waitcnt_vmcnt<0>();
                    __builtin_amdgcn_s_barrier();

                    slide_page_table();
                    // Prefetch next K/V (separate LDS regions; overlaps compute below)
                    if(i_total_loops + 1 < num_total_loop)
                        K_mem_load(number<1>{}); // next K -> K buf 1
                    V_mem_load(number<0>{}); // next V -> V buf 0
#endif

                    V_lds_load(number<1>{}); // V from V buf 1 -> kv_tile.v_tile
                    s_waitcnt_lgkmcnt<0>();
                    fmha_alu1(number<1>{}); // finalize sp(1) -> P(1)
                    gemm(number<1>{}, /*gemm_idx=*/number<1>{}); // PV: P(1)*V

                    K_lds_load(number<0>{}); // K from K buf 0 -> kv_tile.k_tile
                    s_waitcnt_lgkmcnt<0>();

                    gemm(number<0>{}, /*gemm_idx=*/number<0>{}); // QK -> sp(0)
                    fmha_mask(number<0>{});
                    fmha_alu0(number<0>{});
                    fmha_alu_D_upd();
                    i_total_loops++;

                    if(i_total_loops >= num_total_loop)
                        break;

                    // Odd step: V from buf 0, K from buf 1, QK -> sp(1)
#if UA_DECODE_INTERLEAVE
                    slide_page_table();
                    if(i_total_loops + 1 < num_total_loop)
                    {
                        K_mem_load(number<0>{}); // next K -> K buf 0
                        V_mem_load(number<1>{}); // next V -> V buf 1
                        s_waitcnt_vmcnt<K_mem_su_ld_insts + V_mem_su_ld_insts>();
                    }
                    else
                    {
                        s_waitcnt_vmcnt<0>();
                    }
                    __builtin_amdgcn_s_barrier();
#else
                    s_waitcnt_vmcnt<0>();
                    __builtin_amdgcn_s_barrier();

                    slide_page_table();
                    // Prefetch next iteration's K/V
                    if(i_total_loops + 1 < num_total_loop)
                        K_mem_load(number<0>{}); // next K -> K buf 0
                    V_mem_load(number<1>{}); // next V -> V buf 1
#endif

                    V_lds_load(number<0>{}); // V from V buf 0 -> kv_tile.v_tile
                    s_waitcnt_lgkmcnt<0>();
                    fmha_alu1(number<0>{}); // finalize sp(0) -> P(0)
                    gemm(number<0>{}, /*gemm_idx=*/number<1>{}); // PV: P(0)*V

                    K_lds_load(number<1>{}); // K from K buf 1 -> kv_tile.k_tile
                    s_waitcnt_lgkmcnt<0>();

                    gemm(number<1>{}, /*gemm_idx=*/number<0>{}); // QK -> sp(1)
                    fmha_mask(number<1>{});
                    fmha_alu0(number<1>{});
                    fmha_alu_D_upd();
                    i_total_loops++;
                }
            }
            else
            {
                // --- Two warp groups: FA4 matrix‖softmax overlap ---
                // sp(0) holds raw QK(0). WG0 (matrix-first) primes tile-0's
                // softmax once then runs MATRIX-then-SOFTMAX; WG1 (softmax-first)
                // folds tile-0's softmax into its first iteration and runs
                // SOFTMAX-then-MATRIX, so the groups sit one phase apart.
                if(warp_group_id == 0)
                {
                    __builtin_amdgcn_s_setprio(0);
                    fmha_mask_at(number<0>{}, fa4_sm_tile++); // tile num_blocks_start
                    fmha_alu0(number<0>{});
                    fmha_alu_D_upd();
                    fmha_alu1(number<0>{}); // sp(0).p = P(0)
                    // Prime v_tile for MATRIX(0); WG0 loaded V buf 0 in the
                    // pre-stage so its own vmcnt<0> proves residency.
                    s_waitcnt_vmcnt<0>();
                    V_lds_load(number<0>{});
                    while(core_loop_fa4(number<0>{}))
                        ;
                }
                if(warp_group_id != 0)
                {
                    __builtin_amdgcn_s_setprio(0);
                    while(core_loop_fa4(number<1>{}))
                        ;
                }
            }
        }
    label_main_loops_exit:
        // Finalize whichever SP slot was left "alu0-done, alu1-pending" at loop end.
        // The slot depends on the parity of iterations performed (num_total_loop -
        // num_blocks_start), not num_total_loop (they differ under split-KV).
        const index_t num_iters = num_total_loop - num_blocks_start;
        // FA4 drain (NumWarpGroups==2) vs baseline post-process (serial decode).
        if constexpr(kFA4)
        {
            // Deferred-PV drain. WG0 already softmaxed the pending slot inside the
            // loop, so only its PV remains; WG1 (and the degenerate num_iters==1
            // case) still owes the softmax first. Both issue exactly one s_barrier
            // (inside fa4_post_process; the softmax tail is barrier-free) so the two
            // warp groups stay in lockstep.
            auto fa4_epi = [&](auto slot) {
                // WG1 always owes the final tile's softmax. WG0 normally softmaxed it
                // in-loop, except num_iters==1 (pre-stage jumped straight here). The
                // predicate is num_iters==1, NOT num_total_loop==1: a trailing 1-tile
                // split has num_total_loop>1 while num_iters==1.
                if(warp_group_id != 0 || num_iters == 1)
                {
                    fmha_mask_at(slot, fa4_sm_tile++); // last tile (num_total_loop-1)
                    fmha_alu0(slot);
                    fmha_alu_D_upd();
                    fmha_alu1(slot);
                }
                fa4_post_process(slot, slot);
            };
            if(num_iters % 2)
                fa4_epi(number<0>{});
            if(!(num_iters % 2))
                fa4_epi(number<1>{});
        }
        else
        {
            if(num_iters % 2)
            {
                fmha_post_process(number<1>{});
            }
            if(!(num_iters % 2))
            {
                fmha_post_process(number<0>{});
            }
        }

        // finally, O — normalize by l
        constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();

        sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            // Fuse the V FP8 descale into the per-row norm (v_descale == 1.0f for
            // non-FP8). Masked rows with no valid keys keep zeros (l == 0 guard).
            const auto tmp       = [&]() {
                if constexpr(FmhaMask::IsMasking)
                {
                    return l[i_idx] == 0.f ? 0.f : v_descale / l[i_idx];
                }
                else
                    return v_descale / l[i_idx];
            }();
            sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                o_acc(i_j_idx) *= tmp;
            });
        });

        o_acc = tile_elementwise_in(o_acc_element_func, o_acc);

        // Log-sum-exp side-output (natural-log domain) for the split-KV combine
        // (ignored by non-split callers). `m` is the unscaled rowmax and scale_s has
        // a baked-in log2(e), so LSE = log(sum exp(scale*s_k)) = scale*m + log(l)
        // with scale = scale_s/log2(e). The combine re-weights via exp(lse - lse_max).
        const auto scale_natlog =
            scale_s / static_cast<SMPLComputeDataType>(C_LOG2E);
        auto lse = make_static_distributed_tensor<SMPLComputeDataType>(m.get_tile_distribution());
        // o_acc/l are carried in the m_commit frame (FA4) or m (decode); LSE uses the
        // same frame l was summed in.
        sweep_tile_span(o_spans[number<0>{}],
                        [&, m_ = (kCondRescale ? m_commit : m), l_ = l](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            if constexpr(FmhaMask::IsMasking)
            {
                lse(i_idx) =
                    (l_[i_idx] == 0.f)
                        ? -ck_tile::numeric<SMPLComputeDataType>::infinity()
                        : scale_natlog * m_[i_idx] + ck_tile::log(l_[i_idx]);
            }
            else
            {
                lse(i_idx) = scale_natlog * m_[i_idx] + ck_tile::log(l_[i_idx]);
            }
        });

        return ck_tile::make_tuple(o_acc, lse);
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(
        const QDramBlockWindowTmp& q_dram_block_window_tmp, // kBlockM * kHeadDimPadded tile
        const KDramBlockWindowTmp& k_dram_block_window_tmp, // kPageBlockSize * kHeadDimPadded tile
        const VDramBlockWindowTmp& v_dram_block_window_tmp, // kHeadDimPadded * kPageBlockSize tile
        const index_t num_blocks,
        const index_t num_blocks_start,
        const void* block_tables_ptr,
        index_t block_table_offset,
        const index_t page_size, // PageSize in tokens (cache rows per page)
        FmhaMask mask,
        float scale_s,
        void* smem_ptr,
        long_index_t k_row_stride        = 0,
        long_index_t v_row_stride        = 0,
        // Forwards to the full-args operator() so callers can plumb in a
        // runtime kBlockQ. See the documentation on that overload.
        const index_t num_queries_per_kv = 0,
        // See the doc on the full-args operator().
        const bool cache_ptr_int32_overflow_possible = false,
        // See the doc on the full-args operator(). Defaults to 1.0f so
        // non-FP8 callers see no behavior change.
        const float v_descale = 1.0f) const
    {
        using namespace ck_tile;

        return operator()(q_dram_block_window_tmp,
                          identity{},
                          k_dram_block_window_tmp,
                          identity{},
                          v_dram_block_window_tmp,
                          identity{},
                          num_blocks,
                          num_blocks_start,
                          block_tables_ptr,
                          block_table_offset,
                          page_size,
                          identity{},
                          identity{},
                          identity{},
                          mask,
                          scale_s,
                          smem_ptr,
                          k_row_stride,
                          v_row_stride,
                          num_queries_per_kv,
                          cache_ptr_int32_overflow_possible,
                          v_descale);
    }
};

} // namespace ck_tile
