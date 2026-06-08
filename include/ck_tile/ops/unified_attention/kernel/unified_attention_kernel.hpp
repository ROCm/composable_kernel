// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
#include "ck_tile/ops/unified_attention/block/block_masking.hpp"
#include "ck_tile/core/numeric/math.hpp"

#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace ck_tile {

template <typename UnifiedAttentionPipeline_, typename EpiloguePipeline_>
struct UnifiedAttentionKernel
{
    using UnifiedAttentionPipeline = ck_tile::remove_cvref_t<UnifiedAttentionPipeline_>;
    using EpiloguePipeline         = ck_tile::remove_cvref_t<EpiloguePipeline_>;
    static constexpr ck_tile::index_t kBlockSize  = UnifiedAttentionPipeline::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = UnifiedAttentionPipeline::kBlockPerCu;
    static_assert(kBlockPerCu > 0);

    using QDataType    = ck_tile::remove_cvref_t<typename UnifiedAttentionPipeline::QDataType>;
    using KDataType    = ck_tile::remove_cvref_t<typename UnifiedAttentionPipeline::KDataType>;
    using VDataType    = ck_tile::remove_cvref_t<typename UnifiedAttentionPipeline::VDataType>;
    using ODataType    = ck_tile::remove_cvref_t<typename UnifiedAttentionPipeline::ODataType>;
    using SaccDataType = ck_tile::remove_cvref_t<typename UnifiedAttentionPipeline::SaccDataType>;
    using FmhaMask     = ck_tile::remove_cvref_t<typename UnifiedAttentionPipeline::FmhaMask>;
    static constexpr bool kHasMask = FmhaMask::IsMasking;

    static constexpr bool kPadSeqLenK  = UnifiedAttentionPipeline::kPadSeqLenK;
    static constexpr bool kPadSeqLenQ  = UnifiedAttentionPipeline::kPadSeqLenQ;
    static constexpr bool kPadHeadDimQ = UnifiedAttentionPipeline::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV = UnifiedAttentionPipeline::kPadHeadDimV;

    static constexpr index_t kHeadDim       = UnifiedAttentionPipeline::kHeadDim;
    static constexpr index_t kHeadDimPadded = UnifiedAttentionPipeline::kHeadDimPadded;

    // kBlockQ = kBlockM // num_queries_per_kv
    // kBlockQ is the block size for q seqlen
    /// static constexpr index_t kBlockQ = UnifiedAttentionPipeline::kBlockQ;
    static constexpr index_t kBlockM = UnifiedAttentionPipeline::kBlockM;
    static constexpr index_t kBlockQ = UnifiedAttentionPipeline::kBlockQ;
    // BLOCK size for K seqlen
    static constexpr index_t kPageBlockSize = UnifiedAttentionPipeline::kPageBlockSize;

    // kargs use aggregate initializer, so no constructor will provided
    // use inheritance to minimize karg size
    // user need to use MakeKargs() function to create kargs.
    // The attention is default causal
    struct UnifiedAttentionCommonKargs
    {
        const void* q_ptr;
        const void* k_ptr; // [num_blks, page_size, num_kv_heads, head_size]
        const void* v_ptr; // [num_blks, page_size, num_kv_heads, head_size]
        void* o_ptr;

        ck_tile::index_t num_blks;
        ck_tile::index_t num_head_q;
        // for MQA/GQA, nhead could be different. This parameter is nhead_q / nhead_k
        // if this param is larger than 1, indicate MQA/GQA case
        const ck_tile::index_t num_queries_per_kv;
        // scales
        //
        // `scale_s` is the softmax scale (1/sqrt(d) by convention) AFTER:
        //   1. multiplication by log2(e) so the pipeline's exp2 reproduces
        //      the natural-exponent softmax (preserved from the pre-FP8
        //      design — see MakeKargs below);
        //   2. fusion of the FP8 per-tensor Q/K descales (q_descale,
        //      k_descale) for FP8 problems. This matches Triton's
        //      unified_attention reference, which computes
        //      `qk_scale = sm_scale * q_scale * k_scale` and bakes the
        //      log2(e) factor into the exp2 inside the kernel. For
        //      non-FP8 dtypes the host passes q_descale = k_descale = 1.0
        //      so the value reduces to `sm_scale * log2(e)`.
        //
        // `v_descale` is the FP8 per-tensor V descale, deferred to the
        // post-loop `o_acc *= v_descale / l` step inside the pipeline
        // (mathematically exact — V is a linear factor on the
        // unnormalised attention output). For non-FP8 dtypes the host
        // passes 1.0f so this is a free no-op.
        //
        // The legacy `scale` / `scale_k` / `scale_v` / `scale_out` fields
        // are kept in the kargs struct for ABI continuity with downstream
        // code that constructed kargs directly; they are not read by the
        // pipeline any more.
        float scale_s;
        float scale;
        float scale_k;
        float scale_v;
        float scale_out;
        float v_descale;

        ck_tile::index_t page_size;

        ck_tile::index_t total_num_q_blocks;
        ck_tile::index_t query_stride_0;
        ck_tile::index_t query_stride_1;
        ck_tile::index_t stride_k_cache_0;
        ck_tile::index_t stride_k_cache_1;
        ck_tile::index_t stride_k_cache_2;
        ck_tile::index_t stride_k_cache_3;
        ck_tile::index_t stride_v_cache_0;
        ck_tile::index_t stride_v_cache_1;
        ck_tile::index_t stride_v_cache_2;
        ck_tile::index_t stride_v_cache_3;
        ck_tile::index_t output_stride_0;
        ck_tile::index_t output_stride_1;

        // Sliding-window attention parameters (FA-style left/right window).
        // Defaults match the non-SWA identity: `(-1, -1, false)` reproduces
        // the previous hard-coded `(-1, 0, false)` causal mask once the
        // kernel consumes them (Phase 2 / 3). Until then these are pure
        // payload — the operator() still passes literal `(-1, 0, false)` to
        // make_generic_attention_mask_from_lr_window.
        ck_tile::index_t window_size_left  = -1;
        ck_tile::index_t window_size_right = -1;
        bool             is_top_left       = false;
    };

    struct UnifiedAttentionVarlenKargs : UnifiedAttentionCommonKargs
    {
        const int32_t* block_tables_ptr;
        ck_tile::index_t block_table_stride;
        const int32_t* seq_lens_ptr;        // seq len in each batch
        const int32_t* query_start_len_ptr; // [num_seqs+1]

        ck_tile::index_t num_seqs; // number of batches for q

        // KV-segment parallelism (split-KV within unified attention).
        // Each CTA derives its own `i_split` from `blockIdx.z` — the host
        // launches a single 3D grid with z = num_splits and the kernel
        // dispatches all splits in parallel.
        ck_tile::index_t num_splits = 1;
        void* lse_acc_ptr = nullptr;     // [nhead, num_splits, total_q] float
        void* o_acc_ptr = nullptr;       // [nhead, num_splits, total_q, hdim_v] float
        ck_tile::index_t split_stride_lse_acc = 0;
        ck_tile::index_t split_stride_o_acc = 0;
        ck_tile::index_t nhead_stride_lse_acc = 0;
        ck_tile::index_t nhead_stride_o_acc = 0;

        // Runtime selector for the K/V async-load path in the pipeline. See
        // `unified_attention_args::cache_ptr_int32_overflow_possible`.
        bool cache_ptr_int32_overflow_possible = false;

        // Contiguous (THD) KV only: cu_seqlens of the KV cache, [num_seqs+1].
        // kv_start_len_ptr[seq] is this sequence's first KV token, folded into
        // the K/V base pointer. Unused (may be null) when the kernel is paged.
        const int32_t* kv_start_len_ptr = nullptr;
    };

    using Kargs = UnifiedAttentionVarlenKargs;

    CK_TILE_HOST static constexpr Kargs MakeKargs(const void* q_ptr,
                                                  const void* k_ptr,
                                                  const void* v_ptr,
                                                  void* o_ptr,
                                                  ck_tile::index_t num_blks,
                                                  ck_tile::index_t num_head_q,
                                                  const ck_tile::index_t num_queries_per_kv,
                                                  float scale_s,
                                                  float scale,
                                                  float scale_k,
                                                  float scale_v,
                                                  float scale_out,
                                                  float q_descale,
                                                  float k_descale,
                                                  float v_descale,
                                                  ck_tile::index_t page_size,
                                                  ck_tile::index_t total_num_q_blocks,
                                                  ck_tile::index_t query_stride_0,
                                                  ck_tile::index_t query_stride_1,
                                                  ck_tile::index_t stride_k_cache_0,
                                                  ck_tile::index_t stride_k_cache_1,
                                                  ck_tile::index_t stride_k_cache_2,
                                                  ck_tile::index_t stride_k_cache_3,
                                                  ck_tile::index_t stride_v_cache_0,
                                                  ck_tile::index_t stride_v_cache_1,
                                                  ck_tile::index_t stride_v_cache_2,
                                                  ck_tile::index_t stride_v_cache_3,
                                                  ck_tile::index_t output_stride_0,
                                                  ck_tile::index_t output_stride_1,
                                                  const int32_t* block_tables_ptr,
                                                  ck_tile::index_t block_table_stride,
                                                  const int32_t* seq_lens_ptr,
                                                  const int32_t* query_start_len_ptr,
                                                  ck_tile::index_t num_seqs,
                                                  ck_tile::index_t num_splits = 1,
                                                  void* lse_acc_ptr           = nullptr,
                                                  void* o_acc_ptr             = nullptr,
                                                  ck_tile::index_t split_stride_lse_acc   = 0,
                                                  ck_tile::index_t split_stride_o_acc     = 0,
                                                  ck_tile::index_t nhead_stride_lse_acc   = 0,
                                                  ck_tile::index_t nhead_stride_o_acc     = 0,
                                                  bool cache_ptr_int32_overflow_possible  = false,
                                                  ck_tile::index_t window_size_left       = -1,
                                                  ck_tile::index_t window_size_right      = -1,
                                                  bool is_top_left                        = false,
                                                  const int32_t* kv_start_len_ptr         = nullptr)
    {
        // Fuse the Q/K FP8 descales into `scale_s` so the softmax sees a
        // single combined scalar — matches the Triton FP8 reference
        // (qk_scale = sm_scale * q_scale * k_scale) and avoids extra
        // arithmetic per element inside the kernel. The log2(e) factor
        // is included here so the device-side exp2 produces the
        // natural-exponent softmax. For non-FP8 dtypes the host passes
        // q_descale = k_descale = 1.0, which reduces this to the original
        // `scale_s * log2(e)`.
        const float scale_s_fused =
            static_cast<float>(scale_s * q_descale * k_descale * ck_tile::log2e_v<>);
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     o_ptr,
                     num_blks,
                     num_head_q,
                     num_queries_per_kv,
                     scale_s_fused,
                     scale,
                     scale_k,
                     scale_v,
                     scale_out,
                     v_descale,
                     page_size,
                     total_num_q_blocks,
                     query_stride_0,
                     query_stride_1,
                     stride_k_cache_0,
                     stride_k_cache_1,
                     stride_k_cache_2,
                     stride_k_cache_3,
                     stride_v_cache_0,
                     stride_v_cache_1,
                     stride_v_cache_2,
                     stride_v_cache_3,
                     output_stride_0,
                     output_stride_1,
                     window_size_left,
                     window_size_right,
                     is_top_left},
                    block_tables_ptr,
                    block_table_stride,
                    seq_lens_ptr,
                    query_start_len_ptr,
                    num_seqs,
                    num_splits,
                    lse_acc_ptr,
                    o_acc_ptr,
                    split_stride_lse_acc,
                    split_stride_o_acc,
                    nhead_stride_lse_acc,
                    nhead_stride_o_acc,
                    cache_ptr_int32_overflow_possible,
                    kv_start_len_ptr};

        return kargs;
    }

    CK_TILE_HOST static constexpr auto GridSize2D(ck_tile::index_t num_kv_heads,
                                                  ck_tile::index_t total_num_q_blocks,
                                                  ck_tile::index_t num_splits = 1)
    {
        // z-dim carries the split index; num_splits == 1 is the existing
        // (non-split) launch with dim3(N, 1, 1).
        return dim3(num_kv_heads * total_num_q_blocks, 1, num_splits);
    }

    // Binary search to find the sequence index for a given target index
    CK_TILE_DEVICE static constexpr ck_tile::index_t
    find_seq_idx(const int32_t* query_start_len_ptr,
                 ck_tile::index_t target_idx,
                 ck_tile::index_t num_seqs,
                 ck_tile::index_t block_q,
                 bool use_q_block_mode)
    {
        ck_tile::index_t left  = 0;
        ck_tile::index_t right = num_seqs;

        while(left < right)
        {
            ck_tile::index_t mid     = (left + right) / 2;
            ck_tile::index_t val     = amd_wave_read_first_lane(query_start_len_ptr[mid]);
            ck_tile::index_t mid_val = use_q_block_mode ? (val / block_q + mid) : val;

            if(mid_val <= target_idx)
            {
                left = mid + 1;
            }
            else
            {
                right = mid;
            }
        }

        return left - 1;
    }

    CK_TILE_DEVICE static constexpr auto GetTileIndex(const ck_tile::index_t pid,
                                                      const Kargs& kargs)
    {
        using namespace ck_tile;

        ck_tile::index_t num_head_kv = kargs.num_head_q / kargs.num_queries_per_kv;

        return ck_tile::make_tuple(pid % num_head_kv, pid / num_head_kv);
    }
    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return ck_tile::max(UnifiedAttentionPipeline::GetSmemSize(),
                            EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_HOST static constexpr auto GridSizeDecode(ck_tile::index_t num_kv_heads,
                                                      ck_tile::index_t num_seqs,
                                                      ck_tile::index_t num_splits = 1)
    {
        return dim3(num_kv_heads, num_seqs, num_splits);
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        using namespace ck_tile;

        const index_t num_queries_per_kv = kargs.num_queries_per_kv;

        // kBlockQ derived at runtime from num_queries_per_kv. The static
        // `kBlockQ` from the pipeline trait is anchored at kBlockM (i.e. it
        // describes num_qpkv == 1) so the same compiled binary serves every
        // num_qpkv that divides kBlockM evenly -- e.g. the d=128 variants
        // can run both MHA and GQA-N at runtime with no recompile. The host
        // side (select_config) is responsible for enforcing kBlockM %
        // num_queries_per_kv == 0.
        const index_t kBlockQ_dyn = kBlockM / num_queries_per_kv;

        // Split-KV: each CTA handles one (kv_head, q_block, split) tuple. The
        // split index lives in z — when num_splits == 1 (the only z value)
        // this is just `0` and costs nothing.
        const index_t i_split = blockIdx.z;

        index_t kv_head_idx;
        index_t seq_idx;
        index_t q_block_local_idx;
        index_t cur_batch_in_all_start_index;
        index_t cur_batch_query_len;

        if(gridDim.y > 1)
        {
            // Decode grid: dim3(num_kv_heads, num_seqs, num_splits)
            // Direct mapping, no binary search, no padding CTAs.
            kv_head_idx              = blockIdx.x;
            seq_idx                  = blockIdx.y;
            q_block_local_idx        = 0;
            cur_batch_in_all_start_index = kargs.query_start_len_ptr[seq_idx];
            const index_t stop       = kargs.query_start_len_ptr[seq_idx + 1];
            cur_batch_query_len      = amd_wave_read_first_lane(stop - cur_batch_in_all_start_index);
        }
        else
        {
            // Standard 1D grid (x-folded) with binary search; z-dim carries
            // the split index just like in the decode branch.
            ck_tile::index_t pid = blockIdx.x;

            const auto [kv_head_idx_, q_block_global_idx] = GetTileIndex(pid, kargs);
            kv_head_idx = kv_head_idx_;

            if(q_block_global_idx >= kargs.total_num_q_blocks)
            {
                return;
            }

            seq_idx = find_seq_idx(kargs.query_start_len_ptr,
                                   q_block_global_idx,
                                   kargs.num_seqs,
                                   kBlockQ_dyn,
                                   true);

            const index_t q_block_start_idx =
                kargs.query_start_len_ptr[seq_idx] / kBlockQ_dyn + seq_idx;

            q_block_local_idx =
                amd_wave_read_first_lane(q_block_global_idx - q_block_start_idx);

            cur_batch_in_all_start_index = kargs.query_start_len_ptr[seq_idx];
            const index_t cur_batch_in_all_stop_index = kargs.query_start_len_ptr[seq_idx + 1];

            cur_batch_query_len =
                amd_wave_read_first_lane(cur_batch_in_all_stop_index - cur_batch_in_all_start_index);

            if(q_block_local_idx * kBlockQ_dyn >= cur_batch_query_len)
            {
                return;
            }
        }

        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        const index_t query_pos = amd_wave_read_first_lane(q_block_local_idx * kBlockQ_dyn);
        const index_t seq_len   = kargs.seq_lens_ptr[seq_idx];

        const index_t context_len = amd_wave_read_first_lane(seq_len - cur_batch_query_len);

        // Upper bound on the last KV column this Q-tile can attend to. For
        // a causal mask the *last row* in the tile attends up to col=row+1
        // (exclusive), giving `last_row + 1 = ctx + (qbidx+1)*kBlockQ_dyn`.
        // SWA (`IsLocal=true`) extends this by up to `window_size_right`
        // tokens past the diagonal, so we add `max(window_size_right, 0)`
        // to the causal value. The subsequent `min(seq_len)` then clips
        // SWA windows that would otherwise overshoot the end of the
        // sequence. Step D (mask.GetTileRangeAlongX) further clips the
        // KV range from both sides to the actual SWA window — this bound
        // only has to be a safe *upper* envelope across all rows in the
        // Q-tile, not the exact per-row range.
        //
        // GQA pack with kBlockM % num_queries_per_kv != 0 (e.g. d=128,
        // qpkv=6 -> kBlockQ_dyn=21, 21*6=126 < 128): the kBlockM-row tile
        // spills 2 rows into the *next* query position (offset
        // (kBlockM-1)/qpkv = 21, one past the last owned query 20). Those
        // spill rows are co-owned by block N+1, which writes them with the
        // correct (longer) causal KV range. If block N bounds its KV range
        // by the last *owned* query (kBlockQ_dyn-1) it computes those spill
        // rows one key short, so the overlapping store races block N+1 and
        // yields a nondeterministic ~1-row error. Bounding by the tile's
        // actual last row instead makes block N's spill-row result identical
        // to block N+1's -> the duplicate store is idempotent. For ratios
        // that divide kBlockM this reduces to (kBlockQ_dyn-1), a no-op.
        [[maybe_unused]] const index_t last_tile_row_q_off =
            (kBlockM - 1) / num_queries_per_kv;
        [[maybe_unused]] const index_t swa_right_extra =
            (FmhaMask::IsLocal && kargs.window_size_right > 0) ? kargs.window_size_right : 0;
        index_t _max_seq_prefix_len;
        if constexpr(FmhaMask::IsMasking)
        {
            _max_seq_prefix_len = amd_wave_read_first_lane(
                (context_len + q_block_local_idx * kBlockQ_dyn + last_tile_row_q_off + 1 +
                 swa_right_extra));

            if(seq_len < _max_seq_prefix_len)
            {
                _max_seq_prefix_len = seq_len;
            }
        }
        else
        {
            // Non-causal (mask_type=0): attention is bidirectional, so every
            // query tile attends to the *entire* KV sequence. The causal
            // prefix horizon above would clip the KV loop to each tile's
            // diagonal and silently compute causal results, so for the
            // unmasked kernel the envelope is the full sequence length.
            _max_seq_prefix_len = seq_len;
        }

        const auto max_seq_prefix_len = _max_seq_prefix_len;
        index_t total_num_kv_blocks =
            amd_wave_read_first_lane((max_seq_prefix_len + kPageBlockSize - 1) / kPageBlockSize);

        // KV-segment parallelism: split KV range across workgroups.
        // `i_split` came from blockIdx.z above; with num_splits == 1 it's 0
        // and these min/max bounds reduce to [0, total_num_kv_blocks).
        index_t num_blocks_start = 0;
        index_t num_blocks = total_num_kv_blocks;
        if(kargs.num_splits > 1)
        {
            // The split PARTITION (blocks_per_split + start) must be identical
            // across every query tile of this sequence. With a GQA pack where
            // kBlockM % num_queries_per_kv != 0 (e.g. d=128, qpkv=6) the tile's
            // last 1-2 MFMA rows spill into the *next* query tile's first token,
            // so that token is co-owned: query tile N (spill row) and tile N+1
            // (real row) both store the same (token, split) workspace slot.
            //
            // Deriving blocks_per_split from the *per-tile causal* horizon
            // (total_num_kv_blocks, which grows with q_block_local_idx) makes
            // split s cover a different KV-block range in tile N vs tile N+1.
            // The two co-owned stores then hold partials computed over different
            // ranges and race non-deterministically -> a ~1-row error on the
            // tile-boundary token (observed only under split-KV + causal +
            // non-dividing GQA ratio; MHA and ratios that divide kBlockM are
            // immune because no token is shared across tiles).
            //
            // Fix: partition over the causal-INDEPENDENT full sequence block
            // count so split s maps to the same blocks in every tile, then clamp
            // only the END by the per-tile causal horizon. The extra blocks an
            // earlier tile would skip are fully masked per-pixel for the shared
            // token, so both co-owned stores compute the identical partial and
            // the duplicate store is idempotent again. For num_splits == 1 this
            // path is not taken, so the non-split behaviour is unchanged.
            const index_t full_num_kv_blocks =
                amd_wave_read_first_lane((seq_len + kPageBlockSize - 1) / kPageBlockSize);
            const index_t blocks_per_split = ck_tile::max(
                index_t(1), (full_num_kv_blocks + kargs.num_splits - 1) / kargs.num_splits);
            num_blocks_start = ck_tile::min(blocks_per_split * i_split, total_num_kv_blocks);
            num_blocks       = ck_tile::min(blocks_per_split * (i_split + 1), total_num_kv_blocks);
            if(num_blocks_start >= num_blocks)
            {
                return; // this split has no work
            }
        }
        long_index_t kv_head_offset    = static_cast<long_index_t>(kv_head_idx) * kargs.stride_k_cache_2;

        // Q/K/V DRAM and DRAM window.
        // Use long_index_t for the per-CTA base offsets into Q and O: for
        // large total_q (e.g. big-batch prefill) cur_batch_in_all_start_index *
        // stride exceeds 2^31 and an int32 (index_t) offset wraps, sending the
        // store to a bogus address (observed as a "write to read-only page"
        // fault). The cache_ptr_int32_overflow_possible flag only widens the
        // gathered K/V cache addressing; Q and O are separate base pointers and
        // must be widened independently. These offsets are computed once per
        // CTA, so the cost is negligible.
        long_index_t q_ptr_offset_0 = static_cast<long_index_t>(cur_batch_in_all_start_index) *
                                      kargs.query_stride_0; // move the pointer to the batch start
        long_index_t q_ptr_offset_1 =
            static_cast<long_index_t>(kv_head_idx) * num_queries_per_kv *
            kargs.query_stride_1; // move the pointer to the correct head group start
        long_index_t q_ptr_offset = q_ptr_offset_0 + q_ptr_offset_1;

        long_index_t o_ptr_offset_0 = static_cast<long_index_t>(cur_batch_in_all_start_index) *
                                      kargs.output_stride_0; // move the pointer to the batch start
        long_index_t o_ptr_offset_1 =
            static_cast<long_index_t>(kv_head_idx) * num_queries_per_kv *
            kargs.output_stride_1; // move the pointer to the correct head group start
        long_index_t o_ptr_offset  = o_ptr_offset_0 + o_ptr_offset_1;
        index_t block_table_offset = seq_idx * kargs.block_table_stride;

        const QDataType* q_ptr = reinterpret_cast<const QDataType*>(kargs.q_ptr) + q_ptr_offset;
        const KDataType* k_ptr = reinterpret_cast<const KDataType*>(kargs.k_ptr) + kv_head_offset;
        const VDataType* v_ptr = reinterpret_cast<const VDataType*>(kargs.v_ptr) + kv_head_offset;
        ODataType* o_ptr       = reinterpret_cast<ODataType*>(kargs.o_ptr) + o_ptr_offset;

        // Contiguous (THD) KV: fold this sequence's KV start token into the K/V
        // base pointer (the paged path instead resolves it per tile through
        // block_tables). After this the pipeline addresses KV tokens linearly
        // from the sequence base — see UnifiedAttentionPipeline::kIsPaged.
        if constexpr(!UnifiedAttentionPipeline::kIsPaged)
        {
            const long_index_t kv_start_token =
                static_cast<long_index_t>(kargs.kv_start_len_ptr[seq_idx]);
            k_ptr += kv_start_token * static_cast<long_index_t>(kargs.stride_k_cache_1);
            v_ptr += kv_start_token * static_cast<long_index_t>(kargs.stride_v_cache_1);
        }
        // Row count bounding the K/V buffer view. Paged: the whole physical
        // cache (num_blks * page_size). Contiguous: just this sequence's KV
        // length, so the hardware buffer bound masks the over-read past
        // seq_len that the last tile would otherwise issue (no page boundary
        // confines it in the contiguous layout).
        const long_index_t kv_cache_rows =
            UnifiedAttentionPipeline::kIsPaged
                ? static_cast<long_index_t>(kargs.num_blks) * kargs.page_size
                : static_cast<long_index_t>(seq_len);

        index_t query_len_padded = amd_wave_read_first_lane(
            integer_divide_ceil(cur_batch_query_len, kBlockQ_dyn) * kBlockQ_dyn);
        // const bool is_query_len_padded = (cur_batch_query_len % kBlockQ_dyn == 0);

        // Q/K/V DRAM and DRAM window
        const auto q_dram = [&]() {
            const auto q_dram_base = make_naive_tensor_view<address_space_enum::global>(
                q_ptr,
                make_tuple(cur_batch_query_len, num_queries_per_kv, kHeadDim),
                make_tuple(kargs.query_stride_0, kargs.query_stride_1, 1),
                number<UnifiedAttentionPipeline::kAlignmentQ>{},
                number<1>{});

            const auto q_dram_pad =
                pad_tensor_view( // aling seqlen with kBlockQ and head dim with kHeadDimPadded
                    q_dram_base,
                    // block sizes (kBlockQ is runtime here; pad_tensor_view
                    // accepts a mixed compile-time / runtime tuple)
                    make_tuple(kBlockQ_dyn, 1, kHeadDimPadded),
                    sequence<true, false, kPadHeadDimQ>{}); // pads to (seq_len_padded, num_head_q,
                                                            // kHeadDimPadded)

            const auto q_dram_merged = transform_tensor_view(
                q_dram_pad,
                make_tuple(make_merge_transform(make_tuple(query_len_padded, num_queries_per_kv)),
                           make_pass_through_transform(kHeadDimPadded)),
                make_tuple(sequence<0, 1>{}, sequence<2>{}),
                make_tuple(sequence<0>{},
                           sequence<1>{})); // flattens the first two dims, head idx is the fastest
                                            // changing dim in the merged dim

            return q_dram_merged;
        }();
        // static_assert(q_dram.desc_[number<0>{}] == 0,
        // "q_dram.get_bottom_tensor_view()[number<0>{}] == 0");

        // Q has the shape (k_head, seq_len, num_queries_per_kv, head_dim)
        // stride for dim 0 (num_queries_per_kv * head_dim, head_dim, 1)
        auto q_dram_window =
            make_tile_window(q_dram,
                             make_tuple(number<kBlockM>{}, number<kHeadDimPadded>{}),
                             {query_pos * num_queries_per_kv, 0});

        const auto k_dram = [&]() {
            // Use long_index_t for size/strides to prevent int32 overflow
            // when row * stride exceeds 2^31 (happens at ~66K blocks for d64/GQA-8).
            const auto k_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                k_ptr,
                make_tuple(kv_cache_rows,
                           static_cast<long_index_t>(kHeadDim)),
                make_tuple(static_cast<long_index_t>(kargs.stride_k_cache_1),
                           static_cast<long_index_t>(kargs.stride_k_cache_3)),
                number<UnifiedAttentionPipeline::kAlignmentK>{},
                number<1>{});

            const auto k_dram_pad =
                pad_tensor_view(k_dram_naive,
                                make_tuple(kPageBlockSize, kHeadDimPadded),
                                sequence<false, kPadHeadDimQ>{});

            return k_dram_pad;
        }();

        auto k_dram_window = make_tile_window(
            k_dram, make_tuple(number<kPageBlockSize>{}, number<kHeadDimPadded>{}), {0, 0});

        const auto v_dram = [&]() {
            const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                v_ptr,
                make_tuple(kv_cache_rows,
                           static_cast<long_index_t>(kHeadDim)),
                make_tuple(static_cast<long_index_t>(kargs.stride_v_cache_1),
                           static_cast<long_index_t>(kargs.stride_v_cache_3)),
                number<UnifiedAttentionPipeline::kAlignmentV>{},
                number<1>{});

            const auto v_dram_pad = pad_tensor_view(v_dram_naive,
                                                    make_tuple(kPageBlockSize, kHeadDimPadded),
                                                    sequence<false, kPadHeadDimQ>{});

            return v_dram_pad;
        }();

        auto v_dram_window = make_tile_window(
            v_dram, make_tuple(number<kPageBlockSize>{}, number<kHeadDimPadded>{}), {0, 0});

        FmhaMask mask = [&]() {
            if constexpr(kHasMask)
                // Window args default to (-1, -1, false) on the host side, which
                // make_generic_attention_mask_from_lr_window collapses to the
                // previous hard-coded bottom-right causal layout (the `< 0`
                // branches inside the helper). For SWA (Phase 3 IsLocal=true
                // instances) the real (left, right, is_top_left) flow through
                // unchanged and the mask honours both window bounds.
                return ck_tile::make_generic_attention_mask_from_lr_window<FmhaMask>(
                    kargs.window_size_left,
                    kargs.window_size_right,
                    cur_batch_query_len, // y_total
                    seq_len,             // x_total
                    num_queries_per_kv,  // the same sequence index is repeated num_queries_per_kv
                                         // times along x dim of the tile
                    kargs.is_top_left);
            else
                return FmhaMask{cur_batch_query_len, seq_len};
        }();

        // Step D: Sliding-Window-Attention tile-range clip.
        // The per-pixel mask check inside the pipeline already returns the
        // correct (zeroed) score for tokens outside the SWA window, so
        // skipping this block is correctness-preserving. The point of the
        // clip is to skip entire KV tiles that fall completely outside the
        // window — for long-context decode that's the difference between
        // O(seq_k / kPageBlockSize) and O(window / kPageBlockSize)
        // iterations. The intersection with the current split's
        // [num_blocks_start, num_blocks) is taken so split-KV stays correct.
        // Step D: Sliding-Window-Attention KV-tile clip.
        //
        // This is REQUIRED for correctness, not just an optimisation. The
        // online-softmax pipeline interleaves `m` / `l` updates with prefetch
        // and warp-group barriers; an all-(-inf) tile (one wholly outside the
        // SWA window) feeds NaN/garbage into the `m` accumulator at the
        // barrier boundary, corrupting subsequent tiles. Skipping these
        // tiles entirely keeps every iterated tile either fully-inside the
        // window or a true edge tile that the per-pixel mask can clean up.
        //
        // The intersection with the current split's [num_blocks_start,
        // num_blocks) is taken so split-KV stays correct.
        if constexpr(FmhaMask::IsMasking && FmhaMask::IsLocal)
        {
            const auto sw_range = mask.GetTileRangeAlongX(
                query_pos * num_queries_per_kv,
                kBlockQ_dyn,
                static_cast<index_t>(kPageBlockSize));
            const index_t sw_x_start = sw_range.at(number<0>{});
            const index_t sw_x_end   = sw_range.at(number<1>{});

            // GetTileRangeAlongX returns token offsets already aligned to
            // kPageBlockSize; the divide here is exact.
            const index_t sw_block_start = sw_x_start / kPageBlockSize;
            const index_t sw_block_end =
                (sw_x_end + kPageBlockSize - 1) / kPageBlockSize;

            num_blocks_start = ck_tile::max(num_blocks_start, sw_block_start);
            num_blocks       = ck_tile::min(num_blocks, sw_block_end);

            if(num_blocks_start >= num_blocks)
                return; // this Q-tile has no KV inside the SWA window
        }

        // Pass-2: the pipeline now uses a unified per-(thread, Y0-iter) page
        // offset formula and accepts page_size in tokens directly. The earlier
        // `kPageBlockSize <= page_size` constraint (which required at least one
        // kernel tile to fit in a cache page) is gone — tiles may span multiple
        // pages as long as the inner-N step (Y0_step_N from the K/V tile dist)
        // divides page_size cleanly.
        //
        // Pipeline returns make_tuple(o_acc, lse) where o_acc is the normalized
        // attention output (post divide-by-l) and lse is the per-row log-sum-exp
        // in natural-log domain. For num_splits == 1 we ignore lse and forward
        // o_acc through the user's epilogue (bf16/fp16 cast + store to o_ptr).
        // For num_splits > 1 we instead write o_acc and lse to FP32 workspaces
        // — a separate combine kernel will merge across splits.

        auto pipeline_result = UnifiedAttentionPipeline{}(q_dram_window,
                                                          k_dram_window,
                                                          v_dram_window,
                                                          num_blocks,
                                                          num_blocks_start,
                                                          kargs.block_tables_ptr,
                                                          block_table_offset,
                                                          kargs.page_size,
                                                          mask,
                                                          kargs.scale_s,
                                                          smem_ptr,
                                                          static_cast<long_index_t>(kargs.stride_k_cache_1),
                                                          static_cast<long_index_t>(kargs.stride_v_cache_1),
                                                          num_queries_per_kv,
                                                          kargs.cache_ptr_int32_overflow_possible,
                                                          kargs.v_descale);
        auto& o_acc_tile = pipeline_result[number<0>{}];
        auto& lse_tile   = pipeline_result[number<1>{}];

        if(kargs.num_splits > 1)
        {
            // ----- Split-KV write path -----
            // Workspaces (FP32) are assumed in layout:
            //   o_acc_ptr   : [num_q_heads, num_splits, total_q, hdim_v]
            //   lse_acc_ptr : [num_q_heads, num_splits, total_q]
            // The host passes nhead/split strides; the q_token axis is contiguous
            // (= hdim_v for o_acc, = 1 for lse_acc) so we hardcode that here.

            const index_t head_q_base = kv_head_idx * num_queries_per_kv;

            float* o_acc_base = reinterpret_cast<float*>(kargs.o_acc_ptr) +
                                static_cast<long_index_t>(head_q_base) * kargs.nhead_stride_o_acc +
                                static_cast<long_index_t>(i_split) * kargs.split_stride_o_acc +
                                static_cast<long_index_t>(cur_batch_in_all_start_index) * kHeadDim;

            auto o_acc_dram = [&]() {
                const auto o_acc_base_view = make_naive_tensor_view<address_space_enum::global>(
                    o_acc_base,
                    make_tuple(cur_batch_query_len, num_queries_per_kv, kHeadDim),
                    make_tuple(static_cast<long_index_t>(kHeadDim),
                               static_cast<long_index_t>(kargs.nhead_stride_o_acc),
                               static_cast<long_index_t>(1)),
                    number<1>{},
                    number<1>{});

                const auto o_acc_pad = pad_tensor_view(
                    o_acc_base_view,
                    make_tuple(kBlockQ_dyn, 1, kHeadDimPadded),
                    sequence<true, false, kPadHeadDimQ>{});

                return transform_tensor_view(
                    o_acc_pad,
                    make_tuple(make_merge_transform(make_tuple(query_len_padded, num_queries_per_kv)),
                               make_pass_through_transform(kHeadDimPadded)),
                    make_tuple(sequence<0, 1>{}, sequence<2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }();

            auto o_acc_window =
                make_tile_window(o_acc_dram,
                                 make_tuple(number<kBlockM>{}, number<kHeadDimPadded>{}),
                                 {query_pos * num_queries_per_kv, 0});

            // FP32-out epilogue: cast_tile<float>(o_acc) is a no-op, but the
            // pad-aware store path (UseRawStore=true) is the same machinery the
            // user's epilogue uses, so storage semantics are unchanged.
            using SplitOEpilogue =
                Default2DEpilogue<Default2DEpilogueProblem<float, float, true, true, true>>;
            SplitOEpilogue{}(o_acc_window, o_acc_tile, nullptr);

            // ----- LSE write -----
            float* lse_acc_base =
                reinterpret_cast<float*>(kargs.lse_acc_ptr) +
                static_cast<long_index_t>(head_q_base) * kargs.nhead_stride_lse_acc +
                static_cast<long_index_t>(i_split) * kargs.split_stride_lse_acc +
                static_cast<long_index_t>(cur_batch_in_all_start_index);

            auto lse_acc_dram = [&]() {
                const auto lse_acc_base_view = make_naive_tensor_view<address_space_enum::global>(
                    lse_acc_base,
                    make_tuple(cur_batch_query_len, num_queries_per_kv),
                    make_tuple(static_cast<long_index_t>(1),
                               static_cast<long_index_t>(kargs.nhead_stride_lse_acc)),
                    number<1>{},
                    number<1>{});

                const auto lse_acc_pad = pad_tensor_view(
                    lse_acc_base_view, make_tuple(kBlockQ_dyn, 1), sequence<true, false>{});

                return transform_tensor_view(
                    lse_acc_pad,
                    make_tuple(make_merge_transform(make_tuple(query_len_padded, num_queries_per_kv))),
                    make_tuple(sequence<0, 1>{}),
                    make_tuple(sequence<0>{}));
            }();

            auto lse_acc_window =
                make_tile_window(lse_acc_dram, make_tuple(number<kBlockM>{}), {query_pos * num_queries_per_kv});

            store_tile(lse_acc_window, lse_tile);
        }
        else
        {
            // ----- Non-split (current) path -----
            auto o_dram = [&]() {
                const auto o_dram_base = make_naive_tensor_view<address_space_enum::global>(
                    o_ptr,
                    make_tuple(cur_batch_query_len, num_queries_per_kv, kHeadDim),
                    make_tuple(kargs.output_stride_0, kargs.output_stride_1, 1),
                    number<UnifiedAttentionPipeline::kAlignmentO>{},
                    number<1>{});

                const auto o_dram_pad =
                    pad_tensor_view(o_dram_base,
                                    make_tuple(kBlockQ_dyn, 1, kHeadDimPadded),
                                    sequence<true, false, kPadHeadDimQ>{});

                return transform_tensor_view(
                    o_dram_pad,
                    make_tuple(make_merge_transform(make_tuple(query_len_padded, num_queries_per_kv)),
                               make_pass_through_transform(kHeadDimPadded)),
                    make_tuple(sequence<0, 1>{}, sequence<2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }();

            auto o_dram_window =
                make_tile_window(o_dram,
                                 make_tuple(number<kBlockM>{}, number<kHeadDimPadded>{}),
                                 {query_pos * num_queries_per_kv, 0});

            EpiloguePipeline{}(o_dram_window, o_acc_tile, nullptr);
        }
    }
};
} // namespace ck_tile
