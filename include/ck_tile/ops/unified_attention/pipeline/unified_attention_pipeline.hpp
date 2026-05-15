// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/unified_attention/pipeline/unified_attention_pipeline_default_policy.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"
#define ENABLE_ASM_MARKER 1
#if ENABLE_ASM_MARKER
#define ASM_MARKER(marker)               \
    __builtin_amdgcn_sched_barrier(0);   \
    asm volatile("; [POYENC] " #marker); \
    __builtin_amdgcn_sched_barrier(0);
#else
#define ASM_MARKER(marker)
#endif

#define ADD_SBARRIER_FOR_PHASE0 1
#if !defined(CK_TILE_DISABLE_PACKED_FP32)
#define CK_TILE_DISABLE_PACKED_FP32 0
#endif

#define WARP_ID 0
#define LANE_ID 0

#define ENABLE_DEBUG_STMTS 1
#if ENABLE_DEBUG_STMTS
#define DEBUG_STMTS \
    if(get_block_1d_id() == 0 && get_warp_id() == WARP_ID && get_lane_id() == LANE_ID)
#else
#define DEBUG_STMTS if constexpr(false)
#endif

namespace ck_tile {

template <typename Problem_, typename Policy_ = UnifiedAttentionPipelineDefaultPolicy>
struct UnifiedAttentionPipeline
{
    using Problem             = ck_tile::remove_cvref_t<Problem_>;
    using Policy              = ck_tile::remove_cvref_t<Policy_>;
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

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
    static constexpr ck_tile::index_t kAlignmentQ =
        kPadHeadDimQ ? 1 : Policy::template GetAlignmentQ<Problem>();
    static constexpr ck_tile::index_t kAlignmentK =
        kPadHeadDimQ ? 1 : Policy::template GetAlignmentK<Problem>();
    static constexpr ck_tile::index_t kAlignmentV =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentV<Problem>();

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

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        // create another LDS buffer for p
        return ck_tile::max(kBlockM * kHeadDimPadded * sizeof(PDataType),
                            Policy::template GetSmemSize<Problem>() +
                                kBlockM * kPageBlockSize * sizeof(PDataType));
    }

    // for debug only
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

    // for debug only
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
        const index_t page_size, // PageSize in tokens (cache rows per page)
        [[maybe_unused]] const SAccElementFunction& s_acc_element_func,
        const PComputeElementFunction& p_compute_element_func,
        const OAccElementFunction& o_acc_element_func,
        FmhaMask mask,
        float scale_s,
        void* smem_ptr,
        long_index_t k_row_stride         = 0,
        long_index_t v_row_stride         = 0,
        // Runtime kBlockQ = kBlockM / num_queries_per_kv. Default of 0 means
        // "fall back to the compile-time `kBlockQ` from `UnifiedAttentionShape`"
        // so existing callers don't have to change. The kernel template passes
        // the runtime value (from kargs) to remove the static dependency.
        const index_t num_queries_per_kv = 0,
        // Caller-supplied flag: set to true when the K/V cache total byte
        // size can exceed INT32_MAX. Routes K/V async loads through the
        // 64-bit-base `global_load_lds` path (correct but lower throughput).
        // False uses the original shared-SRD `buffer_load_dword_lds` path.
        const bool cache_ptr_int32_overflow_possible = false) const
    {
        using namespace ck_tile;
        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

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

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetPVBlockGemm<Problem>();

        auto q_dram_window = make_tile_window_linear(
            q_dram_block_window_tmp, Policy::template MakeQRegTileDistribution<Problem>());

        // auto q_dram_window = q_dram_block_window_tmp;
        // reduction function for softmax
        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        auto k_lds_window_store = generate_tuple(
            [&](auto i_buf) {
                return make_lds_tile_window<KDataType>(
                    smem_ptr, Policy::template MakeKLdsStoreBlockDescriptor<Problem>(i_buf));
            },
            number<2>{});

        auto v_lds_window_store = generate_tuple(
            [&](auto i_buf) {
                return make_lds_tile_window<KDataType>(
                    smem_ptr, Policy::template MakeVLdsStoreBlockDescriptor<Problem>(i_buf));
            },
            number<2>{});

        statically_indexed_array<decltype(make_tile_window(
                                     make_lds_tile_window<KDataType>(
                                         nullptr,
                                         Policy::template MakeKLdsLoadBlockDescriptor<Problem>()),
                                     Policy::template MakeKRegTileDistribution<Problem>())),
                                 2>
            k_lds_window_load;

        statically_indexed_array<decltype(make_tile_window(
                                     make_lds_tile_window<VDataType>(
                                         nullptr,
                                         Policy::template MakeVLdsLoadBlockDescriptor<Problem>()),
                                     Policy::template MakeVRegTileDistribution<Problem>())),
                                 2>
            v_lds_window_load;

        decltype(make_static_distributed_tensor<QDataType>(
            Policy::template MakeQRegTileDistribution<Problem>())) q_tile;

        union kv_tile_type
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
        statically_indexed_array<sp_compute_type, 2> sp;

        decltype(gemm_1.MakeCBlockTile()) o_acc;
        constexpr index_t fmha_alu_D_reg_cnt = 6; // threshold to decide how many fmha_alu_D_upd()
                                                  // instructions should we move to fmha_alu1()
        static_assert(fmha_alu_D_reg_cnt <= o_acc.thread_buf_.size());

        decltype(block_tile_reduce<SMPLComputeDataType>(
            sp(number<0>{}).sp_compute, sequence<1>{}, f_max, SMPLComputeDataType{0})) m;
        decltype(m) l;

        // initialize k_lds_window and v_lds_window
        static_for<0, 2, 1>{}([&](auto idx) {
            k_lds_window_load(idx) = make_tile_window(
                make_lds_tile_window<KDataType>(
                    static_cast<char*>(smem_ptr) + (idx)*Policy::template GetSmemSizeKV<Problem>(),
                    Policy::template MakeKLdsLoadBlockDescriptor<Problem>()),
                Policy::template MakeKRegTileDistribution<Problem>());
        });

        static_for<0, 2, 1>{}([&](auto idx) {
            v_lds_window_load(idx) =
                make_tile_window(make_lds_tile_window<VDataType>(
                                     static_cast<char*>(smem_ptr) +
                                         (idx + 2) * Policy::template GetSmemSizeKV<Problem>(),
                                     Policy::template MakeVLdsLoadBlockDescriptor<Problem>()),
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

        const auto q_origin = q_dram_window.get_window_origin();

        const auto num_total_loop = num_blocks;
        index_t k_block_idx       = 0;
        index_t v_block_idx       = 0;

        // check early exit if no work to do
        if constexpr(FmhaMask::IsMasking)
        {
            if(num_total_loop - num_blocks_start <= 0)
            {
                // Note: o_acc is already cleared above. q loaded but no fence
                // (ignored). lse must be -infinity so the split-KV combine
                // weighs this empty partial as zero (exp(-inf) == 0); for
                // single-split callers the value is harmless (ignored).
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
        // Split-KV start offset in *tokens* (not in tiles or pages). We add
        // this to logical_token below so the page-table lookup uses the right
        // page; we do NOT shift block_table_offset because num_blocks_start is
        // counted in kPageBlockSize-sized tiles, while block_tables is indexed
        // in page_size-sized pages — the two differ whenever kPageBlockSize !=
        // page_size and shifting tiles-as-pages reads the wrong entries.
        const index_t split_token_offset = num_blocks_start * kPageBlockSize;

        // Pass-2: unified page-offset formula. The kPageBlockSize <= page_size
        // constraint is gone. For every (thread, Y0-iter) pair we compute:
        //
        //     logical_token = tile_idx * kPageBlockSize
        //                   + thread_N_pos                 // lane/warp partition
        //                   + i * Y0_step_N                // per-Y0-iter advance
        //     logical_page  = logical_token / page_size    // index into block_tables
        //     within_page   = logical_token % page_size    // row inside the page
        //     phys_page     = block_tables[block_table_offset + logical_page]
        //     page_offsets[i] = (phys_page * page_size + within_page) * row_stride
        //
        // The page indirection moves entirely into page_offsets, so the per-iter
        // SRD rebase (set_bottom_tensor_view_data_ptr + init_raw) is dropped —
        // we just call update_page_idx() to refresh offsets between tiles. This
        // works for any (kPageBlockSize, page_size) pair where Y0_step_N (= the
        // inner N stride from the dist encoding, N1 * N2) divides page_size, so
        // a single wave-wide load instruction never straddles a page boundary.
        // If page_size < Y0_step_N, per-lane VGPR SRDs would be required and we
        // don't currently support that.
        //
        // TODO(overflow): page_offsets are index_t (int32). For caches whose
        // num_blocks * page_size * row_stride exceeds INT32_MAX, the offsets
        // wrap and reads return wrong data. The previous pass had a one-shot
        // base-pointer shift heuristic for this case (`use_ptr_rebase`); it has
        // been removed here because it does not interact well with the unified
        // formula when block_tables are non-monotonic (a far-away page produces
        // a large negative relative offset that the HW OOB check clamps to 0).
        // A robust fix would either plumb long_index_t through the gather load
        // path or compute a per-batch min-page shift in a pre-pass.
        const auto k_dist = Policy::template MakeKDramTileDistribution<Problem>();
        const auto v_dist = Policy::template MakeVDramTileDistribution<Problem>();
        using KDstrType   = decltype(k_dist);
        using VDstrType   = decltype(v_dist);
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

        const auto k_thread_coord    = k_dist.calculate_index();
        const auto v_thread_coord    = v_dist.calculate_index();
        const index_t k_thread_n_pos = k_thread_coord[number<0>{}];
        const index_t v_thread_n_pos = v_thread_coord[number<0>{}];

        // Page offsets are widened to long_index_t so the `_long` load path
        // (global_load_lds, per-lane 64-bit base) can address pools whose
        // `num_blocks * page_size * row_stride * sizeof(T)` exceeds INT32_MAX.
        // Small-domain values (logical_token, logical_page, within_page,
        // phys_page) stay int32 — they're bounded by the per-CTA sequence
        // and never overflow. The original `async_load_tile_raw` path
        // implicitly narrows this back to int32 when it forwards the value
        // through `async_get_vectorized_elements_raw` — that's intentional,
        // and safe whenever `cache_ptr_int32_overflow_possible == false`.
        statically_indexed_array<long_index_t, KNRepeat> k_page_offsets;
        statically_indexed_array<long_index_t, VNRepeat> v_page_offsets;

        auto refresh_k_offsets = [&](index_t k_tile_idx) {
            static_for<0, KNRepeat, 1>{}([&](auto i) {
                const index_t logical_token = split_token_offset +
                                              k_tile_idx * kPageBlockSize + k_thread_n_pos +
                                              static_cast<index_t>(i.value) * KY0_step_N;
                const index_t logical_page  = logical_token / page_size;
                const index_t within_page   = logical_token - logical_page * page_size;
                const index_t phys_page =
                    block_tables_ptr_[block_table_offset + logical_page];
                k_page_offsets(i) =
                    (static_cast<long_index_t>(phys_page) * page_size + within_page) *
                    k_row_stride;
            });
        };
        auto refresh_v_offsets = [&](index_t v_tile_idx) {
            static_for<0, VNRepeat, 1>{}([&](auto i) {
                const index_t logical_token = split_token_offset +
                                              v_tile_idx * kPageBlockSize + v_thread_n_pos +
                                              static_cast<index_t>(i.value) * VY0_step_N;
                const index_t logical_page  = logical_token / page_size;
                const index_t within_page   = logical_token - logical_page * page_size;
                const index_t phys_page =
                    block_tables_ptr_[block_table_offset + logical_page];
                v_page_offsets(i) =
                    (static_cast<long_index_t>(phys_page) * page_size + within_page) *
                    v_row_stride;
            });
        };

        refresh_k_offsets(k_block_idx);
        refresh_v_offsets(v_block_idx);

        auto k_view = k_dram_block_window_tmp.get_bottom_tensor_view();
        auto v_view = v_dram_block_window_tmp.get_bottom_tensor_view();

        auto k_dram_window =
            make_tile_scatter_gather(k_view,
                                     k_dram_block_window_tmp.get_window_lengths(),
                                     {0, 0},
                                     k_dist,
                                     k_page_offsets);
        k_dram_window.init_raw();

        auto v_dram_window =
            make_tile_scatter_gather(v_view,
                                     v_dram_block_window_tmp.get_window_lengths(),
                                     {0, 0},
                                     v_dist,
                                     v_page_offsets);
        v_dram_window.init_raw();

        // prefetch K tile
        constexpr index_t k0_loops = 1;
        constexpr index_t k1_loops = 1;
        static_assert(1 == k0_loops);
        static_assert(1 == k1_loops);
        // static_assert(kPageBlockSize == kHeadDimPadded);

        constexpr index_t NumWarpGroups = Problem::kBlockSize / Policy::NumThreadPerWarpGroup;
        static_assert(NumWarpGroups == 1 || NumWarpGroups == 2);

        [[maybe_unused]] auto print_dist_tensor = [&](const auto& dist_tensor, const char* name) {
            printf("[POYENC] %s (size=%d): %5.2f",
                   name,
                   decltype(dist_tensor.thread_buf_)::size(),
                   ck_tile::type_convert<float>(dist_tensor.thread_buf_[0]));
            static_for<1, decltype(dist_tensor.thread_buf_)::size(), 1>{}([&](auto i) {
                printf(", %5.2f", ck_tile::type_convert<float>(dist_tensor.thread_buf_[i]));
            });
            printf("\n");
        };

        [[maybe_unused]] auto print_lds = [&](auto lds_tile_window, const char* name) {
            const auto num_rows = lds_tile_window.get_window_lengths().at(number<0>{});
            const auto num_cols = lds_tile_window.get_window_lengths().at(number<1>{});

            auto desc = lds_tile_window.get_bottom_tensor_view().desc_;
            auto data = lds_tile_window.get_bottom_tensor_view().buf_.p_data_;

            if constexpr(true || num_rows < num_cols)
            {
                for(int row = 0; row < num_rows; ++row)
                {
                    int offset = desc.calculate_offset(make_tuple(row, 0));
                    printf("[DEVICE] %s[%3d] = %5.2f",
                           name,
                           row,
                           ck_tile::type_convert<float>(data[offset]));
                    for(int col = 1; col < num_cols; ++col)
                    {
                        printf(", ");
                        offset = desc.calculate_offset(make_tuple(row, col));
                        printf("%5.2f", ck_tile::type_convert<float>(data[offset]));
                    }
                    printf("\n");
                }
            }
            else
            {
                for(int col = 0; col < num_cols; ++col)
                {
                    int offset = desc.calculate_offset(make_tuple(0, col));
                    printf("[DEVICE] %s[%3d] = %5.2f",
                           name,
                           col,
                           ck_tile::type_convert<float>(data[offset]));
                    for(int row = 1; row < num_rows; ++row)
                    {
                        printf(", ");
                        offset = desc.calculate_offset(make_tuple(row, col));
                        printf("%5.2f", ck_tile::type_convert<float>(data[offset]));
                    }
                    printf("\n");
                }
            }
        };

        [[maybe_unused]] auto print_lds_1d = [&](auto lds_tile_window, const char* name) {
            const auto num_elems = lds_tile_window.get_window_lengths().at(number<0>{});

            auto desc = lds_tile_window.get_bottom_tensor_view().desc_;
            auto data = lds_tile_window.get_bottom_tensor_view().buf_.p_data_;

            int offset = desc.calculate_offset(make_tuple(0));
            printf("[DEVICE] %s = %5.2f", name, ck_tile::type_convert<float>(data[offset]));
            for(int e = 1; e < num_elems; ++e)
            {
                printf(", ");
                offset = desc.calculate_offset(make_tuple(e));
                printf("%5.2f", ck_tile::type_convert<float>(data[offset]));
            }
            printf("\n");
        };

        // K_mem_su_ld_insts = 1 for 32 x 128
        // V_mem_su_ld_insts = 1 for 128 x 32
        constexpr int K_mem_su_ld_insts = k_dram_window.get_num_of_access();
        constexpr int V_mem_su_ld_insts = v_dram_window.get_num_of_access();

        // Page block index tracking
        // const index_t kv_page_size_in_blocks =
        //     PageSize / kPageBlockSize;
        // index_t kv_block_idx = 0;
        // only for block 0 and thread
        if(blockIdx.x == 0 && threadIdx.x == 0) {}

        // Pass-2: page indirection lives in page_offsets, not in the SRD. We
        // refresh the per-iter offsets table and push it to the window via
        // update_page_idx(); the SRD itself stays put (no init_raw per iter).
        //
        // Two load paths, dispatched on the runtime overflow flag:
        //   - false: `async_load_tile_raw` → `buffer_load_dword_lds` with a
        //     wave-uniform 4 GB-capped SRD. Faster, but per-lane voffsets
        //     are int32 so the path is only correct while
        //     `num_blocks * page_size * row_stride * sizeof(T) ≤ INT32_MAX`.
        //   - true: `async_load_tile_raw_long` → `global_load_lds_dwordx*`
        //     with per-lane 64-bit base pointers, lifting the 4 GB limit
        //     at the cost of lower throughput.
        // The branch is on a wave-uniform value, so no execution divergence.
        //
        // For diagnostic purposes: the wave's N-position span within a
        // single buffer_load instruction is `(LaneGroups-1)*NumWarps + 1`.
        // When that's > the minimum page_size (≈16) the K-tile distribution
        // touches multiple pages per issue, so the small-cache buffer_load
        // path still works (per-lane voffsets fit while the cache ≤ 4 GB)
        // but the per-issue SRD-rebase optimization (not implemented today)
        // wouldn't be applicable — only `global_load_lds` works once the
        // cache exceeds 4 GB.
        constexpr index_t KWaveSpanInN =
            (KDstrType::DstrEncode::hs_lengthss_[number<0>{}][number<1>{}] - 1) *
                KDstrType::DstrEncode::hs_lengthss_[number<0>{}][number<2>{}] +
            1;
        (void)KWaveSpanInN; // currently informational only

        auto K_mem_load = [&](auto k_lds_write_idx) {
            if(cache_ptr_int32_overflow_possible)
                async_load_tile_raw_long(k_lds_window_store(k_lds_write_idx), k_dram_window);
            else
                async_load_tile_raw(k_lds_window_store(k_lds_write_idx), k_dram_window);
            k_block_idx++;
            refresh_k_offsets(k_block_idx);
            k_dram_window.update_page_idx(k_page_offsets);
        };

        auto V_mem_load = [&](auto v_lds_write_idx) {
            if(cache_ptr_int32_overflow_possible)
                async_load_tile_raw_long(v_lds_window_store(v_lds_write_idx), v_dram_window);
            else
                async_load_tile_raw(v_lds_window_store(v_lds_write_idx), v_dram_window);
            v_block_idx++;
            refresh_v_offsets(v_block_idx);
            v_dram_window.update_page_idx(v_page_offsets);
        };

        auto K_lds_load = [&](auto k_lds_read_idx) {
            kv_tile.k_tile = load_tile(k_lds_window_load(k_lds_read_idx));
        };

        auto V_lds_load = [&](auto v_lds_read_idx) {
            kv_tile.v_tile = load_tile_transpose(v_lds_window_load(v_lds_read_idx));
        };

        decltype(m) m_old;
        SMPLComputeDataType o_acc_scale; // rescale o_acc in fmha_alu1() & fmha_alu_D_upd()
        /// TODO: remove the sp_delta and use sp_compute directly
        statically_indexed_array<decltype(sp(number<0>{}).sp_compute), 2> sp_delta;

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

            constexpr auto p_spans =
                std::decay_t<decltype(sp(sp_reg_idx).sp_compute)>::get_distributed_spans();
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx        = make_tuple(idx0, idx1);
                    sp_delta(sp_reg_idx)(i_j_idx) = detail::fma_impl_vsv(
                        sp(sp_reg_idx).sp_compute(i_j_idx), scale_s, -scale_s * m(i_j_idx));
                });
            });
            /// TODO: move some fmha_alu1() code here if necessary
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

            // l{j}
            /// Note: The compiler keeps moving the following instructions elsewhere because 'l'
            /// is first consumed later. To anchor them here, we rewrite the final addition in
            /// inline assembly to create a dependency, forcing the dependent instructions to
            /// be emitted at this point.
            constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
            sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                const auto tmp       = ck_tile::exp2(scale_s * (m_old[i_idx] - m[i_idx]));

                l(i_idx) = detail::add_impl_vv(tmp * l[i_idx], rowsum_p[i_idx]);
            });

            // update partial o_acc [0, fmha_alu_D_reg_cnt)
            static_for<0, fmha_alu_D_reg_cnt, 1>{}([&](auto idx) {
                o_acc.thread_buf_[idx] = detail::mul_impl_vv(o_acc.thread_buf_[idx], o_acc_scale);
            });

            /// Note: The compiler keeps sinking the conversion instructions because the
            /// result 'p' is only consumed later. To anchor them here, we rewrite
            /// the cast_tile() call as inline assembly, forcing the conversions to be
            /// emitted at this point.
            static_assert(sp(sp_reg_idx).p.thread_buf_.size() % 2 == 0);
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

            /// Note: Place fmha_alu1() at the end of the phase. The surrounding inline assembly
            /// can interfere with the behavior of sched_group_barrier(), so ending the phase here
            /// avoids unintended reordering.
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

        auto cl_calc = [&](auto sp_reg_idx, auto gemm_idx) {
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
                fmha_alu0(number<1>{} - sp_reg_idx);
            }
        };

        auto fmha_alu_D_upd = [&] {
            o_acc_scale = ck_tile::exp2(scale_s * (m_old.thread_buf_[0] - m.thread_buf_[0]));

            fp32x2_t pk_o_acc_scale;
            pk_o_acc_scale.x = o_acc_scale;
            pk_o_acc_scale.y = o_acc_scale;

            static_assert((o_acc.thread_buf_.size() - fmha_alu_D_reg_cnt) % 2 == 0);
#if CK_TILE_DISABLE_PACKED_FP32
            static_assert(fmha_alu_D_reg_cnt + 2 <= o_acc.thread_buf_.size());
            static_for<fmha_alu_D_reg_cnt, fmha_alu_D_reg_cnt + 2, 1>{}(
                [&](auto idx) { o_acc.thread_buf_[idx] *= o_acc_scale; });
#endif

            constexpr auto issued_D_reg_cnt =
#if CK_TILE_DISABLE_PACKED_FP32
                fmha_alu_D_reg_cnt + 2
#else
                fmha_alu_D_reg_cnt
#endif
                ;
            /// NOTICE: Use inline asm v_pk_mul_f32 to reduce latency. The fmha_alu_D_upd() call
            /// should be placed at the end of a phase.
            // update partial o_acc after [issued_D_reg_cnt]
            static_for<issued_D_reg_cnt, o_acc.thread_buf_.size(), 2>{}([&](auto idx) {
                fp32x2_t input;
                input.x = o_acc.thread_buf_[idx];
                input.y = o_acc.thread_buf_[idx + 1];

                auto output = detail::pk_mul_f32(input, pk_o_acc_scale);

                o_acc.thread_buf_[idx]     = output.x;
                o_acc.thread_buf_[idx + 1] = output.y;
            });
        };

        // Resolve kBlockQ at runtime when the caller plumbs in
        // num_queries_per_kv (=> kBlockQ = kBlockM / num_qpkv). Fall back to
        // the static `kBlockQ` from `UnifiedAttentionShape` when the caller
        // passes 0 (back-compat). Stored once, reused per K-tile mask check.
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

        auto cl_load = [&](auto load_type, auto mem_wr_idx, auto lds_rd_idx) {
            if constexpr(load_type == 0)
            {
                V_mem_load(mem_wr_idx);
                K_lds_load(lds_rd_idx);
            }
            else
            {
                K_mem_load(mem_wr_idx);
                V_lds_load(lds_rd_idx);
            }
        };

        auto core_loop = [&](auto cl_p) {
            auto gemm0 = number<0>{};
            auto gemm1 = number<1>{};

            auto memV = number<0>{};
            auto memK = number<1>{};

            using Scheduler = CoreLoopScheduler<Problem, FmhaMask::IsMasking>;

            auto iteration = [&](auto pi) {
                auto xdl_SP_p01_reg_idx = number<1>{} - pi;
                auto xdl_SP_p23_reg_idx = pi;

                auto K_w0_lds_wr_idx = number<1>{} - pi;
                auto V_w0_lds_wr_idx = pi;
                auto K_w0_lds_rd_idx = pi;
                auto V_w0_lds_rd_idx = pi;

                auto K_w4_lds_wr_idx = number<1>{} - pi;
                auto V_w4_lds_wr_idx = number<1>{} - pi;
                auto K_w4_lds_rd_idx = number<1>{} - pi;
                auto V_w4_lds_rd_idx = pi;

                bool result = true;

                if constexpr(cl_p == 0)
                {
#if ADD_SBARRIER_FOR_PHASE0
                    __builtin_amdgcn_sched_barrier(0);
                    __builtin_amdgcn_s_barrier();
#endif
                    __builtin_amdgcn_sched_barrier(0);
                    // phase0
                    if constexpr(pi == 0)
                    {
                        ASM_MARKER("phase0 Wave0-3 (pi=0)");
                    }
                    else
                    {
                        ASM_MARKER("phase0 Wave0-3 (pi=1)");
                    }
                    s_waitcnt_lgkmcnt<0>();
                    __builtin_amdgcn_sched_barrier(0);
                    cl_calc(xdl_SP_p01_reg_idx, gemm0);
                    fmha_alu1(xdl_SP_p23_reg_idx);

                    Scheduler::schedule(cl_p, number<0>{});
                    __builtin_amdgcn_sched_barrier(0);
                    // phase1
                    ASM_MARKER("phase1 Wave0-3");
                    s_waitcnt_vmcnt<K_mem_su_ld_insts + V_mem_su_ld_insts>();
                    __builtin_amdgcn_sched_barrier(0);
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
                    cl_load(memK, K_w0_lds_wr_idx, V_w0_lds_rd_idx);
                    Scheduler::schedule(cl_p, number<1>{});
                    fmha_mask(xdl_SP_p01_reg_idx);

                    __builtin_amdgcn_sched_barrier(0);
                    // phase2
                    ASM_MARKER("phase2 Wave0-3");
                    s_waitcnt_lgkmcnt<0>();
                    __builtin_amdgcn_sched_barrier(0);
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
                    asm volatile("s_nop 0");
                    __builtin_amdgcn_sched_barrier(0);
                    cl_calc(xdl_SP_p23_reg_idx, gemm1);

                    Scheduler::schedule(cl_p, number<2>{});
                    __builtin_amdgcn_sched_barrier(0);
                    fmha_alu_D_upd();

                    __builtin_amdgcn_sched_barrier(0);
                    // phase3
                    ASM_MARKER("phase3 Wave0-3");
                    s_waitcnt_vmcnt<K_mem_su_ld_insts + V_mem_su_ld_insts>();
                    __builtin_amdgcn_sched_barrier(0);
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
                    cl_load(memV, V_w0_lds_wr_idx, K_w0_lds_rd_idx);

                    Scheduler::schedule(cl_p, number<3>{});
                    if(num_total_loop <= ++i_total_loops)
                    {
                        result = false;
                    }
                }
                else
                {
#if ADD_SBARRIER_FOR_PHASE0
                    __builtin_amdgcn_sched_barrier(0);
                    __builtin_amdgcn_s_barrier();
#endif
                    __builtin_amdgcn_sched_barrier(0);
                    // phase0
                    if constexpr(pi == 0)
                    {
                        ASM_MARKER("phase0 Wave4-7 (pi=0)");
                    }
                    else
                    {
                        ASM_MARKER("phase0 Wave4-7 (pi=1)");
                    }
                    cl_load(memV, V_w4_lds_wr_idx, K_w4_lds_rd_idx);

                    Scheduler::schedule(cl_p, number<0>{});
                    __builtin_amdgcn_sched_barrier(0);
                    // phase1
                    ASM_MARKER("phase1 Wave4-7");
                    s_waitcnt<K_mem_su_ld_insts + V_mem_su_ld_insts, 0>();
                    __builtin_amdgcn_sched_barrier(0);
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
                    asm volatile("s_nop 1");
                    __builtin_amdgcn_sched_barrier(0);
                    cl_calc(xdl_SP_p01_reg_idx, gemm0);
                    fmha_alu1(xdl_SP_p23_reg_idx);

                    Scheduler::schedule(cl_p, number<1>{});
                    __builtin_amdgcn_sched_barrier(0);
                    // phase2
                    ASM_MARKER("phase2 Wave4-7");
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
                    cl_load(memK, K_w4_lds_wr_idx, V_w4_lds_rd_idx);
                    Scheduler::schedule(cl_p, number<2>{});
                    fmha_mask(xdl_SP_p01_reg_idx);

                    if(num_total_loop <= ++i_total_loops)
                    {
                        result = false;
                    }

                    __builtin_amdgcn_sched_barrier(0);
                    // phase3
                    ASM_MARKER("phase3 Wave4-7");
                    s_waitcnt<K_mem_su_ld_insts + V_mem_su_ld_insts, 0>();
                    __builtin_amdgcn_sched_barrier(0);
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
                    asm volatile("s_nop 1");
                    __builtin_amdgcn_sched_barrier(0);
                    cl_calc(xdl_SP_p23_reg_idx, gemm1);

                    Scheduler::schedule(cl_p, number<3>{});
                    __builtin_amdgcn_sched_barrier(0);
                    fmha_alu_D_upd();
                }
                return result;
            };
            return iteration(number<0>{}) && iteration(number<1>{});
        };

        auto fmha_post_process = [&](auto d) {
            auto ps_pi        = number<1>{} - d;
            auto V_lds_rd_idx = ps_pi;

            // Wait for the last V tile's async load to complete before reading from LDS.
            // The main loop's final iteration never prefetches K (i_total_loops+1 ==
            // num_total_loop), so only V loads are outstanding here.  The original
            // s_waitcnt_vmcnt<K_mem_su_ld_insts> was a no-op when V_su_ld_insts ==
            // K_su_ld_insts (e.g. both 2 for kPageBlockSize=32), causing a race where
            // V_lds_load read stale LDS before the async V load finished.
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

            fmha_mask(number<0>{});
            /// TODO: find better way to map fmha_alu(0,96) call
            fmha_alu0(number<0>{});
            fmha_alu_D_upd();

            ++i_total_loops;
            if(num_total_loop <= i_total_loops)
            {
                goto label_main_loops_exit;
            }

            if(2 < num_total_loop)
            {
                K_mem_load(number<0>{}); // mem_K2

                s_waitcnt_vmcnt<K_mem_su_ld_insts + V_mem_su_ld_insts>();
                __builtin_amdgcn_s_barrier();
            }

            ASM_MARKER("end pre-stage");
        }

        if(1 < num_total_loop)
        {
            if constexpr(NumWarpGroups == 1)
            {
                // --- Single warp group: serial pipeline with async prefetch ---
                // After pre-stage:
                //   sp(0) has QK for block 0 (alu0 + alu_D_upd done, alu1 NOT done)
                //   V0 loading to LDS (V buf 0)
                //   K1 in LDS (K buf 1) if num_total_loop >= 2
                //   K2 loading to LDS (K buf 0) if num_total_loop >= 3

                // Step 1: consume V0, K1 -> produce PV(0), QK(1)
                s_waitcnt_vmcnt<0>();
                __builtin_amdgcn_s_barrier();

                V_mem_load(number<1>{}); // prefetch V1 -> buf 1 (overlaps with compute)

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
                    // kv_tile is a union: must finish PV GEMM (v_tile) before K load
                    s_waitcnt_vmcnt<0>();
                    __builtin_amdgcn_s_barrier();

                    // Prefetch next iteration's K/V (overlaps with all compute below)
                    // K/V use separate LDS regions so no conflict with current reads
                    if(i_total_loops + 1 < num_total_loop)
                        K_mem_load(number<1>{}); // next K -> K buf 1
                    V_mem_load(number<0>{}); // next V -> V buf 0

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
                    s_waitcnt_vmcnt<0>();
                    __builtin_amdgcn_s_barrier();

                    // Prefetch next iteration's K/V
                    if(i_total_loops + 1 < num_total_loop)
                        K_mem_load(number<0>{}); // next K -> K buf 0
                    V_mem_load(number<1>{}); // next V -> V buf 1

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
                // --- Two warp groups: interleaved pipeline ---
                if(warp_group_id == 0)
                {
                    V_mem_load(number<1>{}); // V1
                    K_lds_load(number<1>{}); // K1

                    __builtin_amdgcn_s_setprio(0);
                    __builtin_amdgcn_s_barrier();
                    while(core_loop(number<0>{}))
                        ;
                }
                if(warp_group_id != 0)
                {
                    __builtin_amdgcn_s_setprio(1);
                    __builtin_amdgcn_s_barrier();
                    while(core_loop(number<1>{}))
                        ;
                }
            }
        }
    label_main_loops_exit:
        // The post-process call finalizes whichever SP register was left in
        // an "alu0-done, alu1-pending" state at the end of the main loop.
        // Which one that is depends on the parity of the *number of
        // iterations performed* (= num_total_loop - num_blocks_start), not
        // num_total_loop itself. For the non-split path num_blocks_start
        // is always 0 so the two parities coincide; the split-KV path with
        // num_blocks_start > 0 needs the corrected expression below.
        const index_t num_iters = num_total_loop - num_blocks_start;
        if(num_iters % 2)
        {
            fmha_post_process(number<1>{});
        }
        if(!(num_iters % 2))
        {
            fmha_post_process(number<0>{});
        }

        // finally, O — normalize by l
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

        // Build the log-sum-exp side-output (natural-log domain) for the
        // split-KV combine kernel. For non-split callers this is ignored.
        //
        // Note `m` here is the *unscaled* rowmax of the raw qk dot products
        // (the pipeline computes `m = block_tile_reduce(sp_compute, max)`
        // before applying `scale_s`). Likewise `l = sum exp2(scale_s*(s-m))`
        // is the natural-domain softmax denominator (since `scale_s` carries
        // a baked-in log2(e), `exp2(scale_s*x) == exp(scale*x)`). Combined,
        //   LSE = log(sum exp(scale * s_k))
        //       = scale * m + log(l)
        //       = scale_s/log2(e) * m + log(l).
        // The combine kernel re-weights partials with exp(lse - lse_max).
        const auto scale_natlog =
            scale_s / static_cast<SMPLComputeDataType>(C_LOG2E);
        auto lse = make_static_distributed_tensor<SMPLComputeDataType>(m.get_tile_distribution());
        sweep_tile_span(o_spans[number<0>{}], [&, m_ = m, l_ = l](auto idx0) {
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
        const bool cache_ptr_int32_overflow_possible = false) const
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
                          cache_ptr_int32_overflow_possible);
    }
};

} // namespace ck_tile
