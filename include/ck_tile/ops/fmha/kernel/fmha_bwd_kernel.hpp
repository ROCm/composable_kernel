// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_selector.hpp"

#include <algorithm>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <memory>

// S[seqlen_q, seqlen_k] = Q[seqlen_q, hdim_q] @ K[seqlen_k, hdim_q]
// S'[seqlen_q, seqlen_k] = S[seqlen_q, seqlen_k] * Scale[1]
// S''[seqlen_q, seqlen_k] = S'[seqlen_q, seqlen_k] + Bias[seqlen_q, seqlen_k]
// P[seqlen_q, seqlen_k] = Softmax(S''[seqlen_q, seqlen_k])
// dV[seqlen_k, hdim_v] = P^T[seqlen_k, seqlen_q] @ dO^T[hdim_v, seqlen_q]
// dP[seqlen_q, seqlen_k] = dO[seqlen_q, hdim_v] @ V[seqlen_k, hdim_v]
// D[seqlen_q] = rowsum(dO[seqlen_q, hdim_v] * O[seqlen_q, hdim_v])
// dS''[seqlen_q, seqlen_k] = P[seqlen_q, seqlen_k] * (dP[seqlen_q, seqlen_k] - D[seqlen_q])
// dBias[seqlen_q, seqlen_k] = dS'[seqlen_q, seqlen_k] = dS''[seqlen_q, seqlen_k]
// dK[seqlen_k, hdim_q] = dS'^T[seqlen_k, seqlen_q] @ Q^T[hdim_q, seqlen_q] * Scale[1]
// dQ[seqlen_q, hdim_q] = dS'[seqlen_q, seqlen_k] @ K^T[hdim_q, seqlen_k] * Scale[1]

namespace ck_tile {

// Per-CU state for group-mode deterministic persistent scheduling.
// alignas(16): enables aligned 128-bit loads; sizeof == 32 (6×4 + 8 pad).
struct alignas(16) FmhaBwdGroupPersistentCuState
{
    index_t w_lo;       // global position of this CU's first K-chunk (= pb + head*hw + c*sq)
    index_t w_hi;       // global position of next CU's first K-chunk (exclusive upper bound)
    index_t ibatch;     // first batch this CU touches (batch_size = no work sentinel)
    index_t isplit;     // isplit for the first (batch, head) this CU touches
    index_t head_start; // head index for the first batch this CU touches
    index_t c_start;    // chunk index for the first (batch, head) this CU touches
    // 8 bytes implicit padding
};

// Per-batch precomputed values used in the group-mode persistent dispatch loop.
// Avoids per-iteration reads from seqstart_q/k_ptr and nsplits_ptr.
// alignas(16): sizeof == 16 (3×4 + 4 pad), fits in a single 128-bit load.
struct alignas(16) FmhaBwdBatchState
{
    index_t sq;      // seqlen_q for this batch (seqstart_q[b+1] - seqstart_q[b])
    index_t nc;      // number of K-chunks: ceil(seqlen_k / kN0)
    index_t nsplits; // dq_acc split count for this batch
    // 4 bytes implicit padding
};

template <typename AccDataType, bool kIsGroupMode, bool kIsDeterministic>
struct FmhaBwdWorkspaceManager
{
    // CPU workspace (prepared by host, read-only for kernels):

    // index_t nsplits[batch or 1]
    //   - per-batch nsplits array (batch element in deterministic group mode)

    // [OPTIONAL, only for deterministic group mode]
    // long_index_t dq_acc_offsets[batch]
    //   - per-batch offset array

    // [OPTIONAL, only for deterministic group mode persistent]
    // FmhaBwdGroupPersistentCuState cu_state[num_cus]
    //   — per-CU packed dispatch state (ibatch, isplit, head_start, c_start, w_lo)
    // FmhaBwdBatchState batch_state[batch]
    //   — per-batch precomputed sq / nc / nsplits

    // GPU WORKSPACE BELOW (read & written by kernels):

    // [OPTIONAL, only for !kUseQrQtrDorPipeline]
    // AccDataType dq_acc[total_elements]
    //   - dq_acc compact buffer (zeroed if necessary)
    //   - total_elements = sum_i(nhead * nsplits_i * seqq_i) * hdim_q
    //   - Layout within each batch: [nhead, nsplits_i, seqq_i, hdim_q]
    //   - note: use physical (including padding) length for seqq_i for group mode

    static constexpr size_t ALIGNMENT = 16;

    template <bool kUseQrQtrDorPipeline>
    CK_TILE_HOST static size_t GetDqAccSplitsSize(const int batch)
    {
        if constexpr(kUseQrQtrDorPipeline)
            return 0;
        const auto dqAccSplitsElems =
            (kIsGroupMode && kIsDeterministic) ? static_cast<size_t>(batch) : 1;
        return integer_least_multiple(sizeof(index_t) * dqAccSplitsElems, ALIGNMENT);
    }
    CK_TILE_HOST static size_t GetDqAccOffsetsSize(const int batch)
    {
        // batch + 1: extra sentinel slot at [batch] holds total dq_acc element
        // count for DqAccPrezeroKernel.
        const auto dqAccOffsetsElems =
            (kIsGroupMode && kIsDeterministic) ? static_cast<size_t>(batch + 1) : 0;
        return integer_least_multiple(sizeof(long_index_t) * dqAccOffsetsElems, ALIGNMENT);
    }
    // cu_state[num_cus]: per-CU persistent state packed into one array (group det only).
    CK_TILE_HOST static size_t GetCuStateSize(const int num_cus)
    {
        if constexpr(kIsGroupMode && kIsDeterministic)
            return integer_least_multiple(sizeof(FmhaBwdGroupPersistentCuState) * num_cus,
                                          ALIGNMENT);
        return 0;
    }
    // batch_state[batch]: per-batch sq/nc/nsplits for group det dispatch loop.
    CK_TILE_HOST static size_t GetBatchStateSize(const int batch)
    {
        if constexpr(kIsGroupMode && kIsDeterministic)
            return integer_least_multiple(sizeof(FmhaBwdBatchState) * batch, ALIGNMENT);
        return 0;
    }

    template <bool kUseQrQtrDorPipeline>
    CK_TILE_HOST static size_t GetWorkspaceHostSize(const int batch)
    {
        if constexpr(kUseQrQtrDorPipeline)
            return 0;
        const size_t raw = GetBatchStateOffset(batch) + GetBatchStateSize(batch);
        // Pad to 4K so dq_acc buffer always starts on a page-aligned boundary.
        return integer_least_multiple(raw, static_cast<size_t>(4096));
    }

    CK_TILE_HOST static size_t GetDqAccSplitsOffset(const int) { return 0; }
    template <bool kUseQrQtrDorPipeline>
    CK_TILE_HOST static size_t GetDqAccOffsetsOffset(const int batch)
    {
        return GetDqAccSplitsSize<kUseQrQtrDorPipeline>(batch);
    }
    CK_TILE_HOST static size_t GetCuStateOffset(const int batch)
    {
        return GetDqAccSplitsSize<false>(batch) + GetDqAccOffsetsSize(batch);
    }
    CK_TILE_HOST static size_t GetBatchStateOffset(const int batch)
    {
        return GetCuStateOffset(batch) + GetCuStateSize(get_num_cus());
    }
    template <bool kUseQrQtrDorPipeline>
    CK_TILE_HOST static size_t GetDqAccDataOffset(const int batch)
    {
        return GetWorkspaceHostSize<kUseQrQtrDorPipeline>(batch);
    }

    template <typename T>
    CK_TILE_HOST static T* workspace_ptr(void* base, size_t offset)
    {
        return reinterpret_cast<T*>(static_cast<char*>(base) + offset);
    }

    // Fill CPU prepared workspace and return size of non CPU prepared workspace size
    template <bool kUseQrQtrDorPipeline, index_t kN0, index_t kM0>
    CK_TILE_HOST static size_t
    PrepareWorkspaceHost(void* cpu_ws,
                         index_t batch_size,
                         index_t hdim_q,
                         index_t nhead_q,
                         index_t seqlen_q           = 0, // only for batch mode
                         index_t seqlen_k           = 0, // only for deterministic batch mode
                         const index_t* seqstart_qs = nullptr,
                         const index_t* seqstart_ks = nullptr)
    {
        if constexpr(kUseQrQtrDorPipeline)
        {
            // QrQtrDor writes dq directly; no workspace is allocated so cpu_ws is nullptr.
            throw std::logic_error(
                "PrepareWorkspaceHost: QrQtrDor pipeline does not use workspace");
        }
        // alignas(16) FmhaBwdGroupPersistentCuState writes use x86 SIMD; fault on misalign.
        if(reinterpret_cast<uintptr_t>(cpu_ws) % 16 != 0)
            throw std::runtime_error("PrepareWorkspaceHost: cpu_ws must be 16-byte aligned");
        const auto nsplits = reinterpret_cast<index_t*>(cpu_ws);
        const auto offsets =
            workspace_ptr<long_index_t>(cpu_ws, GetDqAccSplitsSize<false>(batch_size));
        if constexpr(kIsGroupMode)
            if(!seqstart_qs || !seqstart_ks)
                throw std::runtime_error("seqstart_qs and seqstart_ks are required for group mode");

        if constexpr(!kIsDeterministic)
        {
            nsplits[0] = 1;
            if constexpr(!kIsGroupMode)
                return sizeof(AccDataType) * static_cast<long_index_t>(batch_size) * nhead_q *
                       seqlen_q * hdim_q;
            else
                return sizeof(AccDataType) * static_cast<long_index_t>(nhead_q) *
                       seqstart_qs[batch_size] * hdim_q;
        }
        else if constexpr(kIsGroupMode)
        { // deterministic group mode (persistent)
            // Step 1: compute prefix_batch and target_w using per-batch seqlens.
            // prefix_batch[b] = sum_{i<b}(nhead * nc[i] * sq_work[i]); drives CU partition.
            const index_t num_cus = get_num_cus();
            auto prefix_batch     = std::make_unique<index_t[]>(batch_size + 1);
            auto* cu_states_out =
                workspace_ptr<FmhaBwdGroupPersistentCuState>(cpu_ws, GetCuStateOffset(batch_size));
            auto* batch_states =
                workspace_ptr<FmhaBwdBatchState>(cpu_ws, GetBatchStateOffset(batch_size));

            // Build CU states in logical-CU order; copied to cu_states_out
            // with an XCD-contiguous remap at the end.
            auto cu_states = std::make_unique<FmhaBwdGroupPersistentCuState[]>(num_cus);

            // sq_work: sq aligned to kM0 for work-distribution purposes.
            // If sq==0, use kM0 so CUs are still dispatched and write dK/dV=0.
            const auto sq_work = [](index_t sq) -> index_t {
                return sq == 0 ? kM0 : integer_least_multiple(sq, kM0);
            };

            // No K work anywhere (all seqlen_k==0): no dQ accumulation, so 0
            // dq_acc bytes and no CU partition. Mark cu_states inactive
            // (ibatch sentinel) so GPU early-returns; batch_states / nsplits /
            // offsets are never read past the sentinel, so any consistent
            // zero-ish fill is fine. dK/dV have zero K rows in this case, so
            // there is nothing to write into them.
            if(seqstart_ks[batch_size] == 0)
            {
                std::fill_n(batch_states, batch_size, FmhaBwdBatchState{0, 0, 1});
                std::fill_n(nsplits, batch_size, index_t{1});
                // batch+1 entries: per-batch starts + sentinel total at [batch]
                std::fill_n(offsets, batch_size + 1, long_index_t{0});
                std::fill_n(cu_states_out,
                            num_cus,
                            FmhaBwdGroupPersistentCuState{0, 0, batch_size, 0, 0, 0});
                return 0;
            }

            prefix_batch[0] = 0;
            for(index_t b = 0; b < batch_size; ++b)
            {
                const index_t sq    = seqstart_qs[b + 1] - seqstart_qs[b];
                const index_t nc    = integer_divide_ceil(seqstart_ks[b + 1] - seqstart_ks[b], kN0);
                prefix_batch[b + 1] = prefix_batch[b] + nhead_q * nc * sq_work(sq);
            }
            const index_t target_w = integer_divide_ceil(prefix_batch[batch_size], num_cus);

            // Step 2: fill batch_states.sq/.nc; nsplits is bumped in step 3 to max(isplit+1).
            for(index_t b = 0; b < batch_size; ++b)
            {
                const index_t sq   = seqstart_qs[b + 1] - seqstart_qs[b];
                const index_t sq_w = sq_work(sq);
                const index_t nc   = integer_divide_ceil(seqstart_ks[b + 1] - seqstart_ks[b], kN0);
                batch_states[b].sq = sq_w; // GPU uses sq_w for w_chunk tracking
                batch_states[b].nc = nc;
                batch_states[b].nsplits = 1; // floor; bumped in step 3 by max(isplit + 1)
            }

            // Step 3: fill cu_states via two-pointer scan.
            // w_lo = global K-chunk start (pb + head_start*hw + c_start*sq); GPU compares
            // w_chunk < w_hi for boundaries. w_hi is set in a post-pass to cu_states[c+1].w_lo.
            index_t cu_lo = 0;
            for(index_t b = 0; b < batch_size; ++b)
            {
                const index_t sq   = seqstart_qs[b + 1] - seqstart_qs[b];
                const index_t sq_w = sq_work(sq);
                const index_t nc   = integer_divide_ceil(seqstart_ks[b + 1] - seqstart_ks[b], kN0);
                const index_t hw   = nc * sq_w; // use sq_w so sq=0 batches get work
                const index_t pb   = prefix_batch[b];
                const index_t cu_hi =
                    min(num_cus, integer_divide_ceil(prefix_batch[b + 1], target_w));
                for(index_t c = cu_lo; c < cu_hi; ++c)
                {
                    const index_t w_lo  = c * target_w;
                    cu_states[c].ibatch = b;
                    if(hw > 0)
                    {
                        const index_t head_start =
                            max(static_cast<index_t>((w_lo - pb) / hw), index_t(0));
                        const index_t w_head   = pb + head_start * hw;
                        const index_t wc_start = max(w_lo - w_head, index_t(0));
                        const index_t c_start =
                            wc_start > 0 ? integer_divide_ceil(wc_start, sq_w) : 0;
                        // denom = max(sq_w, target_w) keeps isplit in [0, nc-1] (the upper bound
                        // assumed by GetWorkspaceDeviceSizeUpperBound). Clamp absorbs empty CUs
                        // whose rounded-up wc_start lands past the last K-row; they don't write
                        // dq_acc on GPU so the slot value is harmless.
                        const index_t denom = max(sq_w, target_w);
                        const index_t raw_isp =
                            wc_start > 0 ? integer_divide_ceil(wc_start, denom) : 0;
                        cu_states[c].isplit     = min(raw_isp, max(nc - 1, index_t(0)));
                        cu_states[c].head_start = head_start;
                        cu_states[c].c_start    = c_start;
                        cu_states[c].w_lo       = pb + head_start * hw + c_start * sq_w;

                        // Only count CUs that do real K-row work (c_start < nc) so that
                        // nsplits matches the set of slots actually written by atomic_add.
                        // CUs with c_start >= nc start past the head's K-rows (advance to
                        // next head); their isplit would otherwise pad nsplits with a slot
                        // that nobody writes — reduction would read garbage from it.
                        if(c_start < nc)
                            batch_states[b].nsplits =
                                max(batch_states[b].nsplits, cu_states[c].isplit + 1);
                    }
                    else
                    {
                        cu_states[c].isplit     = 0;
                        cu_states[c].head_start = 0;
                        cu_states[c].c_start    = 0;
                        cu_states[c].w_lo       = pb; // hw==0: degenerate, w_lo=batch start
                    }
                }
                cu_lo = cu_hi;
            }
            // Inactive CUs: use total_w as w_lo sentinel so the post-pass sets
            // the last active CU's w_hi = total_w correctly.
            const index_t total_w = prefix_batch[batch_size];
            for(index_t c = cu_lo; c < num_cus; ++c)
            {
                cu_states[c].w_lo       = total_w;
                cu_states[c].w_hi       = total_w;
                cu_states[c].ibatch     = batch_size; // sentinel → early return on GPU
                cu_states[c].isplit     = 0;
                cu_states[c].head_start = 0;
                cu_states[c].c_start    = 0;
            }
            // Post-pass: set w_hi[c] = w_lo[c+1] (global start of next CU's first K-chunk).
            for(index_t c = 0; c < num_cus - 1; ++c)
                cu_states[c].w_hi = cu_states[c + 1].w_lo;
            cu_states[num_cus - 1].w_hi = total_w;

            // XCD-contiguous remap so each XCD's round-robin blockIdx.x values
            // map to a contiguous range of logical CUs. Mirrors
            // GemmSpatiallyLocalTilePartitioner::RemapXCD; tall_xcds handles the
            // non-divisible case.
            constexpr index_t NUM_XCDS = 8;
            const index_t ids_per_xcd  = (num_cus + NUM_XCDS - 1) / NUM_XCDS;
            const index_t tall_xcds    = (num_cus % NUM_XCDS == 0) ? NUM_XCDS : num_cus % NUM_XCDS;
            for(index_t b = 0; b < num_cus; ++b)
            {
                const index_t xcd      = b % NUM_XCDS;
                const index_t local_id = b / NUM_XCDS;
                const index_t logical  = (xcd < tall_xcds)
                                             ? xcd * ids_per_xcd + local_id
                                             : tall_xcds * ids_per_xcd +
                                                  (xcd - tall_xcds) * (ids_per_xcd - 1) + local_id;
                cu_states_out[b]       = cu_states[logical];
            }

            for(index_t b = 0; b < batch_size; ++b)
                nsplits[b] = batch_states[b].nsplits;

            // Step 4: compute per-batch dq_acc offsets (compact layout, depends on nsplits)
            offsets[0] = 0;
            index_t i  = 0;
            for(; i < batch_size - 1; ++i)
            {
                offsets[i + 1] = offsets[i] + static_cast<long_index_t>(nhead_q) * nsplits[i] *
                                                  (seqstart_qs[i + 1] - seqstart_qs[i]) * hdim_q;
            }
            const long_index_t dq_acc_elems =
                offsets[i] + static_cast<long_index_t>(nhead_q) * nsplits[i] *
                                 (seqstart_qs[i + 1] - seqstart_qs[i]) * hdim_q;
            // Sentinel slot consumed by DqAccPrezeroKernel.
            offsets[batch_size] = dq_acc_elems;
            return sizeof(AccDataType) * dq_acc_elems;
        }
        else // deterministic batch mode (kUsePersistent)
        {
            const index_t dqdqkdv_workers = get_num_cus();
            const index_t jobs_per_head   = integer_divide_ceil(seqlen_k, kN0);
            const index_t total_jobs      = batch_size * nhead_q * jobs_per_head;
            const index_t jobs_per_worker = integer_divide_ceil(total_jobs, dqdqkdv_workers);
            if(jobs_per_head % jobs_per_worker == 0)
                nsplits[0] = jobs_per_head / jobs_per_worker;
            else if(jobs_per_worker % jobs_per_head == 0)
                nsplits[0] = 1;
            else
                nsplits[0] = 1 + integer_divide_ceil(jobs_per_head - 1, jobs_per_worker);
            return sizeof(AccDataType) * static_cast<long_index_t>(batch_size) * nhead_q *
                   nsplits[0] * seqlen_q * hdim_q;
        }
    }

    template <bool kUseQrQtrDorPipeline, bool kHasMask>
    CK_TILE_HOST static constexpr bool NeedsZeroDqAcc()
    {
        constexpr bool kUsePersistent = !kUseQrQtrDorPipeline && kIsDeterministic;
        // Group + persistent + deterministic: dq_acc is zeroed by a separate
        // DqAccPrezeroKernel (see kNeedsKernelPrezeroDqAcc) launched before this
        // kernel, avoiding the launcher memset over the workspace upper bound
        // (~20x larger than the actual region for large seqlen_k).
        if constexpr(kUsePersistent && kIsGroupMode)
            return false;
        // Persistent (batch and group): uses atomic_add → buffer must start at zero
        //   so that accumulated dq values are correct.
        // Non-deterministic: uses atomic_add → buffer must start at zero.
        if constexpr(kUsePersistent || !kIsDeterministic)
            return true;
        // Non-persistent deterministic: uses set, but causal mask may skip some tiles
        // leaving dq_acc slots unwritten — zero them out first.
        return kHasMask;
    }

    // Upper bound on PrepareWorkspaceHost's size, computable without seqstart so
    // the device workspace can be allocated before any D2H.
    //
    // total_seqlen_q_padded: total q tokens incl. per-batch padding.
    //   Batch: max_batch * seqlen_q. Group: seqstart_q[batch].
    // max_seqlen_k: deterministic-only; pass per-batch padded max if the caller
    //   does internal k padding, otherwise the logical max is fine.
    template <bool kUseQrQtrDorPipeline, index_t kN0>
    CK_TILE_HOST static size_t GetWorkspaceDeviceSizeUpperBound(index_t max_batch,
                                                                index_t hdim_q,
                                                                index_t nhead_q,
                                                                index_t total_seqlen_q_padded,
                                                                index_t max_seqlen_k)
    {
        if constexpr(kUseQrQtrDorPipeline)
            return 0;

        index_t nsplits_factor = 1;
        if constexpr(kIsDeterministic)
        {
            if constexpr(kIsGroupMode)
            {
                nsplits_factor = integer_divide_ceil(max_seqlen_k, kN0);
            }
            else // persistent
            {
                const index_t dqdqkdv_workers = get_num_cus();
                const index_t jobs_per_head   = integer_divide_ceil(max_seqlen_k, kN0);
                const index_t total_jobs      = max_batch * nhead_q * jobs_per_head;
                const index_t jobs_per_worker = integer_divide_ceil(total_jobs, dqdqkdv_workers);
                if(jobs_per_head % jobs_per_worker == 0)
                    nsplits_factor = jobs_per_head / jobs_per_worker;
                else if(jobs_per_worker % jobs_per_head == 0)
                    nsplits_factor = 1;
                else
                    nsplits_factor = 1 + integer_divide_ceil(jobs_per_head - 1, jobs_per_worker);
            }
        }

        return sizeof(AccDataType) * static_cast<long_index_t>(nhead_q) * nsplits_factor *
               total_seqlen_q_padded * hdim_q;
    }
};

template <typename FmhaPipeline_,
          typename KGradEpiloguePipeline_,
          typename VGradEpiloguePipeline_,
          typename QGradEpiloguePipeline_ = void>
struct FmhaBwdDQDKDVKernel
{
    using FmhaPipeline                            = ck_tile::remove_cvref_t<FmhaPipeline_>;
    using KGradEpiloguePipeline                   = ck_tile::remove_cvref_t<KGradEpiloguePipeline_>;
    using VGradEpiloguePipeline                   = ck_tile::remove_cvref_t<VGradEpiloguePipeline_>;
    using QGradEpiloguePipeline                   = ck_tile::remove_cvref_t<QGradEpiloguePipeline_>;
    static constexpr ck_tile::index_t kBlockSize  = FmhaPipeline::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = FmhaPipeline::kBlockPerCu;
    static constexpr bool kUseQrQtrDorPipeline =
        ck_tile::fmha_bwd_qr_qtr_dor_pipeline<FmhaPipeline>::value;
    static_assert(!kUseQrQtrDorPipeline || !std::is_same_v<QGradEpiloguePipeline_, void>,
                  "QrQtrDorPipeline needs QGradEpiloguePipeline");

    using QDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::QDataType>;
    using KDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::KDataType>;
    using VDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::VDataType>;
    using BiasDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::BiasDataType>;
    using GemmDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::GemmDataType>;
    using LSEDataType  = ck_tile::remove_cvref_t<typename FmhaPipeline::LSEDataType>;
    using AccDataType  = ck_tile::remove_cvref_t<typename FmhaPipeline::AccDataType>;
    using DDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::DDataType>;
    using RandValOutputDataType =
        ck_tile::remove_cvref_t<typename FmhaPipeline::RandValOutputDataType>;
    using OGradDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::OGradDataType>;
    using QGradDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::QGradDataType>;
    using KGradDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::KGradDataType>;
    using VGradDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::VGradDataType>;
    using BiasGradDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::BiasGradDataType>;

    static constexpr bool kIsGroupMode    = FmhaPipeline::kIsGroupMode;
    static constexpr index_t kPadHeadDimQ = FmhaPipeline::kPadHeadDimQ;
    static constexpr index_t kPadHeadDimV = FmhaPipeline::kPadHeadDimV;
    static constexpr auto BiasEnum        = FmhaPipeline::BiasEnum;
    static constexpr bool kHasBiasGrad    = FmhaPipeline::kHasBiasGrad;
    using FmhaMask                    = ck_tile::remove_cvref_t<typename FmhaPipeline::FmhaMask>;
    using FmhaDropout                 = ck_tile::remove_cvref_t<typename FmhaPipeline::FmhaDropout>;
    static constexpr bool kHasMask    = FmhaMask::IsMasking;
    static constexpr bool kHasDropout = FmhaDropout::IsDropout;
    static constexpr bool kIsStoreRandval  = FmhaDropout::IsStoreRandval;
    static constexpr bool kIsDeterministic = FmhaPipeline::kIsDeterministic;
    static constexpr bool kUseTrLoad       = FmhaPipeline::kUseTrLoad;
    static constexpr index_t kMaxSeqLenQ   = FmhaPipeline::BlockFmhaShape::kMaxSeqLenQ;
    static_assert(kUseQrQtrDorPipeline == (kMaxSeqLenQ != 0));
#if defined(__gfx950__)
    static constexpr bool kIsAvailable = true;
#else
    static constexpr bool kIsAvailable = !kUseTrLoad;
#endif
    static constexpr bool kUsePersistent = kIsDeterministic && !kUseQrQtrDorPipeline;
    using WorkspaceManager = FmhaBwdWorkspaceManager<AccDataType, kIsGroupMode, kIsDeterministic>;

    // clang-format off
    template <typename T> struct t2s;
    template <> struct t2s<float> { static constexpr const char * name = "fp32"; };
    template <> struct t2s<ck_tile::fp16_t> { static constexpr const char * name = "fp16"; };
    template <> struct t2s<ck_tile::bf16_t> { static constexpr const char * name = "bf16"; };
    // clang-format on

    CK_TILE_HOST static std::string GetName()
    {
        // sync with generate.py
        // clang-format off
        using bfs  = typename FmhaPipeline::BlockFmhaShape;
        using gbr0 = typename bfs::Gemm0BlockWarps;
        using gbr1 = typename bfs::Gemm1BlockWarps;
        using gbr4 = typename bfs::Gemm4BlockWarps;
        using gwt0 = typename bfs::Gemm0WarpTile;
        using gwt1 = typename bfs::Gemm1WarpTile;
        #define _SS_  std::string
        #define _TS_  std::to_string
        auto pn = [&] () {
            std::string n;
            if (kPadHeadDimQ) n += "d" + _TS_(kPadHeadDimQ);
            if (kPadHeadDimV) n += "dv"+ _TS_(kPadHeadDimV);
            return n.empty() ? n : std::string("p") + n; }();
        return
            _SS_("fmha_bwd_d") + _TS_(bfs::kQKHeaddim) + "_" + _SS_(t2s<QDataType>::name) +
            "_" + (kIsGroupMode ? "group" : "batch") + "_" +
            "b" + _TS_(bfs::kM0) + "x" + _TS_(bfs::kN0) + "x" + _TS_(bfs::kK0) + "x" + _TS_(bfs::kK1) + "x" + _TS_(bfs::kK2) + "x" + _TS_(bfs::kK3) + "x" +
                    _TS_(bfs::kK4) + "x" + _TS_(bfs::kQKHeaddim) + "x" + _TS_(bfs::kVHeaddim) + "_" +
            "r" + _TS_(gbr0::at(ck_tile::number<0>{})) + "x" + _TS_(gbr0::at(ck_tile::number<1>{})) + "x" + _TS_(gbr0::at(ck_tile::number<2>{})) + "_" +
            "r" + _TS_(gbr1::at(ck_tile::number<0>{})) + "x" + _TS_(gbr1::at(ck_tile::number<1>{})) + "x" + _TS_(gbr1::at(ck_tile::number<2>{})) + "_" +
            "r" + _TS_(gbr4::at(ck_tile::number<0>{})) + "x" + _TS_(gbr4::at(ck_tile::number<1>{})) + "x" + _TS_(gbr4::at(ck_tile::number<2>{})) + "_" +
            "w" + _TS_(gwt0::at(ck_tile::number<0>{})) + "x" + _TS_(gwt0::at(ck_tile::number<1>{})) + "x" + _TS_(gwt0::at(ck_tile::number<2>{})) + "_" +
            "w" + _TS_(gwt1::at(ck_tile::number<0>{})) + "x" + _TS_(gwt1::at(ck_tile::number<1>{})) + "x" + _TS_(gwt1::at(ck_tile::number<2>{})) + "_" +
            ("o" + _TS_(kBlockPerCu)) + "_" +
            ("maxq" + _TS_(kMaxSeqLenQ)) +
            (pn.empty() ? "_npad" : "_" + pn) +
            (BiasEnum == BlockAttentionBiasEnum::NO_BIAS ? _SS_("_nbias") : (_SS_("_") + BlockAttentionBiasEnumToStr<BiasEnum>::name)) +
            (kHasBiasGrad ? "_dbias" : "_ndbias") + (kHasMask ? "_" + _SS_(FmhaMask::name) : "_nmask") + (kHasDropout ? gwt0::at(ck_tile::number<0>{}) == 16? "_dropout_wg16":"_dropout_wg32" : "_ndropout" ) +
            (kIsStoreRandval ? "_storerandval" : "" ) + (kIsDeterministic ? "_deterministic" : "_ndeterministic" ) + (kUseTrLoad ? "_trload" : "_ntrload");
        #undef _SS_
        #undef _TS_
        // clang-format on
    }
    template <typename... Args>
    CK_TILE_HOST static constexpr auto GetWorkspaceHostSize(Args&&... args)
    {
        return WorkspaceManager::template GetWorkspaceHostSize<kUseQrQtrDorPipeline>(
            std::forward<Args>(args)...);
    }
    template <typename... Args>
    CK_TILE_HOST static constexpr auto PrepareWorkspaceHost(Args&&... args)
    {
        return WorkspaceManager::template PrepareWorkspaceHost<kUseQrQtrDorPipeline,
                                                               FmhaPipeline::BlockFmhaShape::kN0,
                                                               FmhaPipeline::BlockFmhaShape::kM0>(
            std::forward<Args>(args)...);
    }
    template <typename... Args>
    CK_TILE_HOST static size_t GetWorkspaceDeviceSizeUpperBound(Args&&... args)
    {
        return WorkspaceManager::template GetWorkspaceDeviceSizeUpperBound<
            kUseQrQtrDorPipeline,
            FmhaPipeline::BlockFmhaShape::kN0>(std::forward<Args>(args)...);
    }
    CK_TILE_HOST static constexpr bool NeedsZeroDqAcc()
    {
        return WorkspaceManager::template NeedsZeroDqAcc<kUseQrQtrDorPipeline, kHasMask>();
    }
    // Group + persistent + deterministic is the only path where NeedsZeroDqAcc()
    // is false yet dq_acc still has unowned slots (per-head varying active isplit
    // sets) that must be zeroed beforehand.
    static constexpr bool kNeedsKernelPrezeroDqAcc =
        kIsGroupMode && kIsDeterministic && !kUseQrQtrDorPipeline;

    // Flat-zeroes the active dq_acc region before the main kernel.
    struct DqAccPrezeroKernel
    {
        static constexpr index_t kBlockSize  = 256;
        static constexpr index_t kBlockPerCu = 1;
        struct Kargs
        {
            void* dq_acc_ptr;
            const long_index_t* total_elem_ptr;
        };
        CK_TILE_HOST static dim3 GridSize() { return dim3(get_num_cus()); }
        CK_TILE_HOST static dim3 BlockSize() { return dim3(kBlockSize); }
        CK_TILE_DEVICE void operator()(Kargs kargs) const
        {
            // Total elements are float32 dq_acc counts; uint4 packs 4 floats.
            const long_index_t total = *kargs.total_elem_ptr;
            const long_index_t n4    = total / 4;
            uint4* p                 = reinterpret_cast<uint4*>(kargs.dq_acc_ptr);
            // per_block aligned to kBlockSize: keeps every non-tail iteration
            // full-warp and consecutive writes within one block share HBM rows.
            const long_index_t n_tiles = ck_tile::integer_divide_ceil(n4, kBlockSize);
            const long_index_t per_block =
                ck_tile::integer_divide_ceil(n_tiles, gridDim.x) * kBlockSize;
            const long_index_t start = blockIdx.x * per_block;
            const long_index_t end   = ck_tile::min(start + per_block, n4);
            for(long_index_t off = start + threadIdx.x; off < end; off += kBlockSize)
                p[off] = uint4{0u, 0u, 0u, 0u};
            // Tail: at most 3 floats if total isn't a multiple of 4.
            const long_index_t tail = total % 4;
            if(blockIdx.x == 0 && threadIdx.x < tail)
                reinterpret_cast<float*>(kargs.dq_acc_ptr)[n4 * 4 + threadIdx.x] = 0.0f;
        }
    };

    CK_TILE_HOST static typename DqAccPrezeroKernel::Kargs
    MakeDqAccPrezeroKargs(void* workspace_ptr, int batch)
    {
        auto* ws = static_cast<char*>(workspace_ptr);
        // Sentinel slot address: end of the batch+1 extended dq_acc_offsets array.
        const size_t total_elem_off =
            WorkspaceManager::template GetDqAccOffsetsOffset<kUseQrQtrDorPipeline>(batch) +
            batch * sizeof(long_index_t);
        const size_t dq_acc_off =
            WorkspaceManager::template GetDqAccDataOffset<kUseQrQtrDorPipeline>(batch);
        return {ws + dq_acc_off, reinterpret_cast<const long_index_t*>(ws + total_elem_off)};
    }

    template <ck_tile::index_t I> // to avoid duplicated base class prblem, introduce an template
                                  // arg
    struct FmhaBwdEmptyKargs
    {
    };

    // kargs use aggregate initializer, so no constructor will provided
    // use inheritance to minimize karg size
    // user need to use MakeKargs() function to create kargs.
    struct FmhaBwdCommonKargs
    {
        const void* q_ptr;
        const void* k_ptr;
        const void* v_ptr;
        const void* lse_ptr;
        const void* do_ptr;
        const void* d_ptr;
        void* dq_acc_ptr; // can be dq_ptr for qrqtrdor pipeline
        void* dk_ptr;
        void* dv_ptr;

        ck_tile::index_t seqlen_q;
        ck_tile::index_t seqlen_k;
        ck_tile::index_t hdim_q;
        ck_tile::index_t hdim_v;

        // for MQA/GQA, nhead could be different. This parameter is nhead_q / nhead_k
        // if this param is larger than 1, indicate MQA/GQA case
        ck_tile::index_t nhead_q;
        ck_tile::index_t nhead_ratio_qk;
        float raw_scale;
        float scale;

        ck_tile::index_t stride_q;
        ck_tile::index_t stride_k;
        ck_tile::index_t stride_v;
        ck_tile::index_t stride_do;
        ck_tile::index_t stride_dk;
        ck_tile::index_t stride_dv;

        ck_tile::index_t nhead_stride_q;
        ck_tile::index_t nhead_stride_k;
        ck_tile::index_t nhead_stride_v;
        ck_tile::index_t nhead_stride_do;
        ck_tile::index_t nhead_stride_lsed;
        ck_tile::index_t nhead_stride_dk;
        ck_tile::index_t nhead_stride_dv;
    };

    // strides for the QrQtrDor pipeline which writes dq directly (no split accumulator)
    struct FmhaBwdQrQtrDorKargs
    {
        ck_tile::index_t stride_dq;
        ck_tile::index_t nhead_stride_dq;
        std::conditional_t<kIsGroupMode, FmhaBwdEmptyKargs<0>, ck_tile::index_t> batch_stride_dq;
    };

    struct FmhaBwdCommonBiasKargs
    {
        const void* bias_ptr               = nullptr;
        ck_tile::index_t stride_bias       = 0;
        ck_tile::index_t nhead_stride_bias = 0;
    };

    struct FmhaBwdBatchModeBiasKargs : FmhaBwdCommonBiasKargs
    {
        ck_tile::index_t batch_stride_bias = 0;
    };

    struct FmhaBwdAlibiKargs
    {
        // alibi is batch*nhead*1, no matter in batch/group mode, they are the same
        const void* alibi_slope_ptr;
        ck_tile::index_t alibi_slope_stride; // stride in batch, or 0 for all batch share same slope
    };

    struct FmhaBwdCommonBiasGradKargs
    {
        void* dbias_ptr                     = nullptr;
        ck_tile::index_t stride_dbias       = 0;
        ck_tile::index_t nhead_stride_dbias = 0;
    };

    struct FmhaBwdBatchModeBiasGradKargs : FmhaBwdCommonBiasGradKargs
    {
        ck_tile::index_t batch_stride_dbias = 0;
    };

    struct FmhaBwdMaskKargs
    {
        ck_tile::index_t window_size_left, window_size_right;
        ck_tile::GenericAttentionMaskEnum mask_type;
    };

    struct FmhaBwdDropoutSeedOffset
    {
        template <typename T>
        union ValueOrPointer
        {
            T val;
            const T* ptr;
        };

        ValueOrPointer<uint64_t> drop_seed;
        ValueOrPointer<uint64_t> drop_offset;
        bool is_drop_seed_offset_from_host;
    };

    struct FmhaBwdCommonDropoutKargs : FmhaBwdDropoutSeedOffset
    {
        void init_dropout(float p_drop, uint64_t seed, uint64_t offset, float raw_scale)
        {
            float p_undrop = 1.0 - p_drop;
            p_undrop_in_uint8_t =
                uint8_t(std::floor(p_undrop * std::numeric_limits<uint8_t>::max()));
            rp_undrop       = 1.0 / p_undrop;
            scale_rp_undrop = rp_undrop * raw_scale;

            this->drop_seed.val                 = seed;
            this->drop_offset.val               = offset;
            this->is_drop_seed_offset_from_host = true;
        }

        void init_dropout(float p_drop,
                          const uint64_t* seed_ptr,
                          const uint64_t* offset_ptr,
                          float raw_scale)
        {
            float p_undrop = 1.0 - p_drop;
            p_undrop_in_uint8_t =
                uint8_t(std::floor(p_undrop * std::numeric_limits<uint8_t>::max()));
            rp_undrop       = 1.0 / p_undrop;
            scale_rp_undrop = rp_undrop * raw_scale;

            this->drop_seed.ptr                 = seed_ptr;
            this->drop_offset.ptr               = offset_ptr;
            this->is_drop_seed_offset_from_host = false;
        }

        float rp_undrop             = 1;
        float scale_rp_undrop       = 1;
        uint8_t p_undrop_in_uint8_t = std::numeric_limits<uint8_t>::max();
        void* rand_val_ptr          = nullptr;

        ck_tile::index_t stride_randval       = 0;
        ck_tile::index_t nhead_stride_randval = 0;
    };

    struct FmhaBwdBatchModeDropoutKargs : FmhaBwdCommonDropoutKargs
    {
        ck_tile::index_t batch_stride_randval = 0;
    };

    struct FmhaBwdDeterministicKargs
    {
        ck_tile::index_t batch;              // used for persistent kernel implementation
        const ck_tile::index_t* nsplits_ptr; // per-batch nsplits (group) or single scalar (batch)
        // group mode persistent scheduling tables (read from CPU workspace by GPU):
        const FmhaBwdGroupPersistentCuState* cu_state_ptr; // per-CU packed state, size [num_cus]
        const FmhaBwdBatchState* batch_state_ptr;          // per-batch sq/nc/nsplits, size [batch]
    };

    struct FmhaBwdBatchModeKargs
        : FmhaBwdCommonKargs,
          std::conditional_t<BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS,
                             FmhaBwdBatchModeBiasKargs,
                             std::conditional_t<BiasEnum == BlockAttentionBiasEnum::ALIBI,
                                                FmhaBwdAlibiKargs,
                                                FmhaBwdEmptyKargs<1>>>,
          std::conditional_t<kHasBiasGrad, FmhaBwdBatchModeBiasGradKargs, FmhaBwdEmptyKargs<2>>,
          std::conditional_t<kHasMask, FmhaBwdMaskKargs, FmhaBwdEmptyKargs<3>>,
          std::conditional_t<kHasDropout, FmhaBwdBatchModeDropoutKargs, FmhaBwdEmptyKargs<4>>,
          std::conditional_t<kIsDeterministic, FmhaBwdDeterministicKargs, FmhaBwdEmptyKargs<5>>,
          std::conditional_t<kUseQrQtrDorPipeline, FmhaBwdQrQtrDorKargs, FmhaBwdEmptyKargs<6>>
    {
        ck_tile::index_t batch_stride_q;
        ck_tile::index_t batch_stride_k;
        ck_tile::index_t batch_stride_v;
        ck_tile::index_t batch_stride_do;
        ck_tile::index_t batch_stride_lsed;
        ck_tile::index_t batch_stride_dk;
        ck_tile::index_t batch_stride_dv;
    };

    struct FmhaBwdGroupModeKargs
        : FmhaBwdCommonKargs,
          std::conditional_t<BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS,
                             FmhaBwdCommonBiasKargs,
                             std::conditional_t<BiasEnum == BlockAttentionBiasEnum::ALIBI,
                                                FmhaBwdAlibiKargs,
                                                FmhaBwdEmptyKargs<0>>>,
          std::conditional_t<kHasBiasGrad, FmhaBwdCommonBiasGradKargs, FmhaBwdEmptyKargs<1>>,
          std::conditional_t<kHasMask, FmhaBwdMaskKargs, FmhaBwdEmptyKargs<2>>,
          std::conditional_t<kHasDropout, FmhaBwdCommonDropoutKargs, FmhaBwdEmptyKargs<3>>,
          std::conditional_t<kIsDeterministic, FmhaBwdDeterministicKargs, FmhaBwdEmptyKargs<4>>,
          std::conditional_t<kUseQrQtrDorPipeline, FmhaBwdQrQtrDorKargs, FmhaBwdEmptyKargs<5>>
    {
        const int32_t* seqstart_q_ptr;
        const int32_t* seqstart_k_ptr;
        const int32_t* seqlen_q_ptr;    // per-batch actual length [batch]
        const int32_t* seqlen_k_ptr;    // per-batch actual length [batch]
        const int32_t* cu_seqlen_q_ptr; // cumulative seqlen [batch+1], optional
        const int32_t* cu_seqlen_k_ptr; // cumulative seqlen [batch+1], optional
        // per-batch element offset into dq_acc buffer (compact layout); used when deterministic
        const ck_tile::long_index_t* dq_acc_batch_offset_ptr;
    };

    using Kargs = std::conditional_t<kIsGroupMode, FmhaBwdGroupModeKargs, FmhaBwdBatchModeKargs>;

    // std::variant<> can't take in a list initializer, overload for backward compatibility
    template <typename... Ts>
    CK_TILE_HOST static constexpr Kargs
    MakeKargs(Ts... args, const std::tuple<uint64_t, uint64_t>& drop_seed_offset)
    {
        return MakeKargsImpl(
            args..., std::make_pair(std::get<0>(drop_seed_offset), std::get<1>(drop_seed_offset)));
    }

    // std::variant<> can't take in a list initializer, overload for backward compatibility
    template <typename... Ts>
    CK_TILE_HOST static constexpr Kargs
    MakeKargs(Ts... args, const std::tuple<const void*, const void*>& drop_seed_offset)
    {
        return MakeKargsImpl(
            args..., std::make_pair(std::get<0>(drop_seed_offset), std::get<1>(drop_seed_offset)));
    }

    template <bool Cond = !kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargsImpl(const void* q_ptr,
                  const void* k_ptr,
                  const void* v_ptr,
                  const void* bias_ptr,
                  const void* lse_ptr,
                  const void* do_ptr,
                  const void* d_ptr,
                  void* rand_val_ptr,
                  void* dq_ptr, // only used with qrqtrdor pipeline
                  void* dk_ptr,
                  void* dv_ptr,
                  void* dbias_ptr,
                  void* workspace_ptr,
                  ck_tile::index_t seqlen_q,
                  ck_tile::index_t seqlen_k,
                  ck_tile::index_t batch,
                  ck_tile::index_t hdim_q,
                  ck_tile::index_t hdim_v,
                  ck_tile::index_t nhead_q,
                  ck_tile::index_t nhead_ratio_qk,
                  float scale,
                  ck_tile::index_t stride_q,
                  ck_tile::index_t stride_k,
                  ck_tile::index_t stride_v,
                  ck_tile::index_t stride_bias,
                  ck_tile::index_t stride_randval,
                  ck_tile::index_t stride_do,
                  ck_tile::index_t stride_dq, // only used for QrQtrDor pipeline
                  ck_tile::index_t stride_dk,
                  ck_tile::index_t stride_dv,
                  ck_tile::index_t stride_dbias,
                  ck_tile::index_t nhead_stride_q,
                  ck_tile::index_t nhead_stride_k,
                  ck_tile::index_t nhead_stride_v,
                  ck_tile::index_t nhead_stride_bias,
                  ck_tile::index_t nhead_stride_randval,
                  ck_tile::index_t nhead_stride_do,
                  ck_tile::index_t nhead_stride_lsed,
                  ck_tile::index_t nhead_stride_dq, // only used for QrQtrDor pipeline
                  ck_tile::index_t nhead_stride_dk,
                  ck_tile::index_t nhead_stride_dv,
                  ck_tile::index_t nhead_stride_dbias,
                  ck_tile::index_t batch_stride_q,
                  ck_tile::index_t batch_stride_k,
                  ck_tile::index_t batch_stride_v,
                  ck_tile::index_t batch_stride_bias,
                  ck_tile::index_t batch_stride_randval,
                  ck_tile::index_t batch_stride_do,
                  ck_tile::index_t batch_stride_lsed,
                  ck_tile::index_t batch_stride_dq, // only used for QrQtrDor pipeline
                  ck_tile::index_t batch_stride_dk,
                  ck_tile::index_t batch_stride_dv,
                  ck_tile::index_t batch_stride_dbias,
                  ck_tile::index_t window_size_left,
                  ck_tile::index_t window_size_right,
                  ck_tile::index_t mask_type,
                  float p_drop,
                  std::variant<std::pair<uint64_t, uint64_t>, std::pair<const void*, const void*>>
                      drop_seed_offset)
    {
        uint8_t* ws = reinterpret_cast<uint8_t*>(workspace_ptr);
        Kargs kargs{
            {q_ptr,
             k_ptr,
             v_ptr,
             lse_ptr,
             do_ptr,
             d_ptr,
             [&]() {
                 if constexpr(kUseQrQtrDorPipeline)
                     return dq_ptr;
                 else
                     return ws +
                            WorkspaceManager::template GetDqAccDataOffset<kUseQrQtrDorPipeline>(
                                batch);
             }(),
             dk_ptr,
             dv_ptr,
             seqlen_q,
             seqlen_k,
             hdim_q,
             hdim_v,
             nhead_q,
             nhead_ratio_qk,
             scale,
             static_cast<float>(scale * ck_tile::log2e_v<>),
             stride_q,
             stride_k,
             stride_v,
             stride_do,
             stride_dk,
             stride_dv,
             nhead_stride_q,
             nhead_stride_k,
             nhead_stride_v,
             nhead_stride_do,
             nhead_stride_lsed,
             nhead_stride_dk,
             nhead_stride_dv}, // args for common karg
            {},                // placeholder for bias
            {},                // placeholder for dbias
            {},                // placeholder for mask
            {},                // placeholder for dropout
            {},                // placeholder for deterministic
            {},                // placeholder for QrQtrDor
            batch_stride_q,
            batch_stride_k,
            batch_stride_v,
            batch_stride_do,
            batch_stride_lsed,
            batch_stride_dk,
            batch_stride_dv};

        if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
        {
            kargs.bias_ptr          = bias_ptr;
            kargs.stride_bias       = stride_bias;
            kargs.nhead_stride_bias = nhead_stride_bias;
            kargs.batch_stride_bias = batch_stride_bias;
        }
        else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
        {
            kargs.alibi_slope_ptr    = bias_ptr;
            kargs.alibi_slope_stride = stride_bias;
        }

        if constexpr(kHasBiasGrad)
        {
            kargs.dbias_ptr          = dbias_ptr;
            kargs.stride_dbias       = stride_dbias;
            kargs.nhead_stride_dbias = nhead_stride_dbias;
            kargs.batch_stride_dbias = batch_stride_dbias;
        }

        if constexpr(kHasMask)
        {
            kargs.window_size_left  = window_size_left;
            kargs.window_size_right = window_size_right;
            kargs.mask_type         = static_cast<ck_tile::GenericAttentionMaskEnum>(mask_type);
        }

        if constexpr(kHasDropout)
        {
            if(drop_seed_offset.index() == 0) // seed & offset come from host
            {
                const auto& [seed, offset] = std::get<0>(drop_seed_offset);
                kargs.init_dropout(p_drop, seed, offset, scale);
            }
            else // seed & offset come from device
            {
                const auto& [seed_ptr, offset_ptr] = std::get<1>(drop_seed_offset);
                kargs.init_dropout(p_drop,
                                   reinterpret_cast<const uint64_t*>(seed_ptr),
                                   reinterpret_cast<const uint64_t*>(offset_ptr),
                                   scale);
            }

            if constexpr(kIsStoreRandval)
            {
                kargs.rand_val_ptr         = rand_val_ptr;
                kargs.stride_randval       = stride_randval;
                kargs.nhead_stride_randval = nhead_stride_randval;
                kargs.batch_stride_randval = batch_stride_randval;
            }
        }

        if constexpr(kUseQrQtrDorPipeline)
        {
            kargs.stride_dq       = stride_dq;
            kargs.nhead_stride_dq = nhead_stride_dq;
            kargs.batch_stride_dq = batch_stride_dq;
        }

        if constexpr(kUsePersistent)
        {
            kargs.batch       = batch;
            kargs.nsplits_ptr = reinterpret_cast<const ck_tile::index_t*>(
                reinterpret_cast<const uint8_t*>(workspace_ptr) +
                WorkspaceManager::GetDqAccSplitsOffset(batch));
        }

        return kargs;
    }

    template <bool Cond = kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargsImpl(const void* q_ptr,
                  const void* k_ptr,
                  const void* v_ptr,
                  const void* bias_ptr,
                  const void* lse_ptr,
                  const void* do_ptr,
                  const void* d_ptr,
                  void* rand_val_ptr,
                  void* dq_ptr,
                  void* dk_ptr,
                  void* dv_ptr,
                  void* dbias_ptr,
                  void* workspace_ptr,
                  const void* seqstart_q_ptr,
                  const void* seqstart_k_ptr,
                  const void* seqlen_q_ptr,
                  const void* seqlen_k_ptr,
                  const void* cu_seqlen_q_ptr,
                  const void* cu_seqlen_k_ptr,
                  ck_tile::index_t batch,
                  ck_tile::index_t hdim_q,
                  ck_tile::index_t hdim_v,
                  ck_tile::index_t nhead_q,
                  ck_tile::index_t nhead_ratio_qk,
                  float scale,
                  ck_tile::index_t stride_q,
                  ck_tile::index_t stride_k,
                  ck_tile::index_t stride_v,
                  ck_tile::index_t stride_bias,
                  ck_tile::index_t stride_randval,
                  ck_tile::index_t stride_do,
                  ck_tile::index_t stride_dq, // only used for QrQtrDor pipeline
                  ck_tile::index_t stride_dk,
                  ck_tile::index_t stride_dv,
                  ck_tile::index_t stride_dbias,
                  ck_tile::index_t nhead_stride_q,
                  ck_tile::index_t nhead_stride_k,
                  ck_tile::index_t nhead_stride_v,
                  ck_tile::index_t nhead_stride_bias,
                  ck_tile::index_t nhead_stride_randval,
                  ck_tile::index_t nhead_stride_do,
                  ck_tile::index_t nhead_stride_lsed,
                  ck_tile::index_t nhead_stride_dq, // only used for QrQtrDor pipeline
                  ck_tile::index_t nhead_stride_dk,
                  ck_tile::index_t nhead_stride_dv,
                  ck_tile::index_t nhead_stride_dbias,
                  ck_tile::index_t window_size_left,
                  ck_tile::index_t window_size_right,
                  ck_tile::index_t mask_type,
                  float p_drop,
                  std::variant<std::pair<uint64_t, uint64_t>, std::pair<const void*, const void*>>
                      drop_seed_offset)
    {
        const auto ws = reinterpret_cast<uint8_t*>(workspace_ptr);
        Kargs kargs{
            {q_ptr,
             k_ptr,
             v_ptr,
             lse_ptr,
             do_ptr,
             d_ptr,
             [&]() {
                 if constexpr(kUseQrQtrDorPipeline)
                     return dq_ptr;
                 else
                     return ws +
                            WorkspaceManager::template GetDqAccDataOffset<kUseQrQtrDorPipeline>(
                                batch);
             }(),
             dk_ptr,
             dv_ptr,
             -1, // seqlen will be updated by another pointer
             -1, //
             hdim_q,
             hdim_v,
             nhead_q,
             nhead_ratio_qk,
             scale,
             static_cast<float>(scale * ck_tile::log2e_v<>),
             stride_q,
             stride_k,
             stride_v,
             stride_do,
             stride_dk,
             stride_dv,
             nhead_stride_q,
             nhead_stride_k,
             nhead_stride_v,
             nhead_stride_do,
             nhead_stride_lsed,
             nhead_stride_dk,
             nhead_stride_dv}, // args for common karg
            {},                // placeholder for bias
            {},                // placeholder for dbias
            {},                // placeholder for mask
            {},                // placeholder for dropout
            {},                // placeholder for deterministic
            {},                // placeholder for QrQtrDor
            reinterpret_cast<const int32_t*>(seqstart_q_ptr),
            reinterpret_cast<const int32_t*>(seqstart_k_ptr),
            reinterpret_cast<const int32_t*>(seqlen_q_ptr),
            reinterpret_cast<const int32_t*>(seqlen_k_ptr),
            reinterpret_cast<const int32_t*>(cu_seqlen_q_ptr),
            reinterpret_cast<const int32_t*>(cu_seqlen_k_ptr),
            nullptr, // dq_acc_batch_offset_ptr (set below for non-QrQtrDor deterministic)
        };

        if constexpr(!kUseQrQtrDorPipeline)
            kargs.dq_acc_batch_offset_ptr = reinterpret_cast<const long_index_t*>(
                ws + WorkspaceManager::template GetDqAccOffsetsOffset<kUseQrQtrDorPipeline>(batch));

        if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
        {
            kargs.bias_ptr          = bias_ptr;
            kargs.stride_bias       = stride_bias;
            kargs.nhead_stride_bias = nhead_stride_bias;
        }
        else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
        {
            kargs.alibi_slope_ptr    = bias_ptr;
            kargs.alibi_slope_stride = stride_bias;
        }
        if constexpr(kHasBiasGrad)
        {
            kargs.dbias_ptr          = dbias_ptr;
            kargs.stride_dbias       = stride_dbias;
            kargs.nhead_stride_dbias = nhead_stride_dbias;
        }
        if constexpr(kHasMask)
        {
            kargs.window_size_left  = window_size_left;
            kargs.window_size_right = window_size_right;
            kargs.mask_type         = static_cast<ck_tile::GenericAttentionMaskEnum>(mask_type);
        }
        if constexpr(kHasDropout)
        {
            if(drop_seed_offset.index() == 0) // seed & offset come from host
            {
                const auto& [seed, offset] = std::get<0>(drop_seed_offset);
                kargs.init_dropout(p_drop, seed, offset, scale);
            }
            else // seed & offset come from device
            {
                const auto& [seed_ptr, offset_ptr] = std::get<1>(drop_seed_offset);
                kargs.init_dropout(p_drop,
                                   reinterpret_cast<const uint64_t*>(seed_ptr),
                                   reinterpret_cast<const uint64_t*>(offset_ptr),
                                   scale);
            }

            if constexpr(kIsStoreRandval)
            {
                kargs.rand_val_ptr         = rand_val_ptr;
                kargs.stride_randval       = stride_randval;
                kargs.nhead_stride_randval = nhead_stride_randval;
            }
        }
        if constexpr(kUseQrQtrDorPipeline)
        {
            kargs.stride_dq       = stride_dq;
            kargs.nhead_stride_dq = nhead_stride_dq;
        }

        if constexpr(kUsePersistent)
        {
            kargs.batch       = batch;
            kargs.nsplits_ptr = reinterpret_cast<const ck_tile::index_t*>(
                ws + WorkspaceManager::GetDqAccSplitsOffset(batch));
            if constexpr(kIsGroupMode)
            {
                kargs.cu_state_ptr = reinterpret_cast<const FmhaBwdGroupPersistentCuState*>(
                    ws + WorkspaceManager::GetCuStateOffset(batch));
                kargs.batch_state_ptr = reinterpret_cast<const FmhaBwdBatchState*>(
                    ws + WorkspaceManager::GetBatchStateOffset(batch));
            }
        }

        return kargs;
    }

    CK_TILE_HOST static constexpr auto
    GridSize(ck_tile::index_t batch_size_, ck_tile::index_t nhead_, ck_tile::index_t seqlen_k_)
    {
        const index_t jobs_per_head =
            kUseQrQtrDorPipeline ? 1 : integer_divide_ceil(seqlen_k_, FmhaPipeline::kN0);
        if constexpr(kUsePersistent)
            return dim3(get_num_cus(), 1, 1);
        else
            return dim3(jobs_per_head, nhead_, batch_size_);
    }

    CK_TILE_HOST static dim3 BlockSize()
    {
        if(is_wave32())
        {
            return dim3(kBlockSize / 2);
        }
        else
        {
            return dim3(kBlockSize);
        }
    }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return ck_tile::max(FmhaPipeline::GetSmemSize(),
                            KGradEpiloguePipeline::GetSmemSize(),
                            VGradEpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        if constexpr(kIsAvailable)
        {
            if constexpr(!kUsePersistent)
            {
                if constexpr(kUseQrQtrDorPipeline || kIsGroupMode)
                {
                    run_(std::move(kargs), blockIdx, blockIdx.x, 0);
                }
                else
                {
                    static_assert(!kIsDeterministic,
                                  "Deterministic Batch Mode should use persistent kernel");
                    run_(std::move(kargs), blockIdx, blockIdx.x, 1);
                }
            }
            else
            {
                static_assert(!kUseQrQtrDorPipeline,
                              "Persistent kernel is not compatible with QR/QTR/DOR pipeline");

                // 0,1,2,3,4,5 ==> 0,5,1,4,2,3 for load balance in triangular mask case
                constexpr auto tile_n_interleave = [](index_t x, index_t n) {
                    if constexpr(kHasMask == false)
                        return x;
                    else
                        return x % 2 == 0 ? (x / 2) : (n - 1 - x / 2);
                };
                if constexpr(!kIsGroupMode)
                {
                    // Batch mode persistent: uniform seqlen_k across all batches
                    const index_t worker_id  = blockIdx.x;
                    const index_t worker_num = gridDim.x;

                    const index_t jobs_per_head =
                        integer_divide_ceil(kargs.seqlen_k, FmhaPipeline::kN0);
                    const index_t total_heads     = kargs.batch * kargs.nhead_q;
                    const index_t total_jobs      = jobs_per_head * total_heads;
                    const index_t jobs_per_worker = integer_divide_ceil(total_jobs, worker_num);

                    const index_t begin_job_id = worker_id * jobs_per_worker;
                    if(begin_job_id >= total_jobs)
                        return; // worker_id exceeds total jobs, exit early
                    const index_t end_job_id = min((worker_id + 1) * jobs_per_worker, total_jobs);

                    const auto n_splits = kargs.nsplits_ptr[0];
                    index_t job_id      = begin_job_id;
                    index_t i_split = integer_divide_ceil(job_id % jobs_per_head, jobs_per_worker);
                    do
                    { // loop over jobs assigned to this worker
                        const index_t i_head_flatten = job_id / jobs_per_head;
                        const index_t i_tile_n_      = job_id % jobs_per_head;
                        const index_t i_tile_n       = tile_n_interleave(i_tile_n_, jobs_per_head);
                        const index_t i_batch        = i_head_flatten / kargs.nhead_q;
                        const index_t i_nhead        = i_head_flatten % kargs.nhead_q;

                        if(i_tile_n_ == 0) // reset dq_acc writing idx when starting a new head
                            i_split = 0;
                        run_(kargs, dim3(i_tile_n, i_nhead, i_batch), i_split, n_splits);
                    } while(++job_id < end_job_id);
                }
                else
                {
                    // Group mode persistent: variable seqlen per batch, dispatch via gist algo.
                    // cu_state_ptr is XCD-remapped on host, so blockIdx.x indexes directly.
                    const FmhaBwdGroupPersistentCuState* cs = kargs.cu_state_ptr + blockIdx.x;
                    const index_t w_hi                      = amd_wave_read_first_lane(cs->w_hi);
                    index_t ibatch                          = amd_wave_read_first_lane(cs->ibatch);
                    index_t isplit                          = amd_wave_read_first_lane(cs->isplit);
                    index_t head_start = amd_wave_read_first_lane(cs->head_start);
                    index_t c_start_0  = amd_wave_read_first_lane(cs->c_start);
                    index_t w_chunk    = amd_wave_read_first_lane(cs->w_lo);
                    if(ibatch >= kargs.batch)
                        return; // this CU has no work (sentinel: ibatch == batch_size)

                    // Loop exits when w_chunk reaches w_hi. Inner check guards the rare case
                    // where head_start hits nhead_q and ibatch is bumped past batch before exit.
                    do
                    {
                        if(ibatch >= kargs.batch)
                            return;
                        // sq/nc/nsplits are read inline (not pre-hoisted) to shorten their live
                        // range across the inlined run_() body, reducing SGPR pressure.
                        const FmhaBwdBatchState* bs = kargs.batch_state_ptr + ibatch;

                        while(head_start < kargs.nhead_q)
                        {
                            // dq_acc was flat-zeroed by DqAccPrezeroKernel before launch.
                            while(c_start_0 < amd_wave_read_first_lane(bs->nc) && w_chunk < w_hi)
                            {
                                run_(kargs,
                                     dim3(tile_n_interleave(c_start_0,
                                                            amd_wave_read_first_lane(bs->nc)),
                                          head_start,
                                          ibatch),
                                     isplit,
                                     amd_wave_read_first_lane(bs->nsplits));
                                w_chunk += amd_wave_read_first_lane(bs->sq);
                                ++c_start_0;
                            }
                            if(w_chunk >= w_hi)
                                return;
                            // w_chunk is now at the start of the next head
                            c_start_0 = 0;
                            isplit    = 0;
                            ++head_start;
                        }
                        head_start = 0;
                        ++ibatch;
                    } while(w_chunk < w_hi);
                }
            }
        }
    }

    CK_TILE_DEVICE void
    run_(Kargs kargs, const dim3& tile_index, const index_t i_split, const index_t n_splits) const
    {
        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        // divide problem
        const index_t i_tile_n = tile_index.x;
        const index_t i_nhead  = tile_index.y;
        const index_t i_batch  = tile_index.z;

        const index_t i_n0 = amd_wave_read_first_lane(i_tile_n * FmhaPipeline::kN0);

        long_index_t batch_offset_q       = 0;
        long_index_t batch_offset_k       = 0;
        long_index_t batch_offset_v       = 0;
        long_index_t batch_offset_bias    = 0;
        long_index_t batch_offset_randval = 0;
        long_index_t batch_offset_do      = 0;
        long_index_t batch_offset_lsed    = 0;
        long_index_t batch_offset_dq_acc  = 0;
        long_index_t batch_offset_dk      = 0;
        long_index_t batch_offset_dv      = 0;
        long_index_t batch_offset_dbias   = 0;
        // dq_acc per-nhead stride uses padded seqlen_q in group mode; equals kargs.seqlen_q
        // in batch mode. See FmhaBwdWorkspaceManager doc.
        index_t physical_seqlen_q = kargs.seqlen_q;

        if constexpr(kIsGroupMode)
        {
            // get starting offset for each batch
            const long_index_t query_start = kargs.seqstart_q_ptr[i_batch];
            const long_index_t key_start   = kargs.seqstart_k_ptr[i_batch];

            physical_seqlen_q =
                static_cast<index_t>(kargs.seqstart_q_ptr[i_batch + 1] - query_start);

            batch_offset_q    = query_start * kargs.stride_q;
            batch_offset_k    = key_start * kargs.stride_k;
            batch_offset_v    = key_start * kargs.stride_v;
            batch_offset_do   = query_start * kargs.stride_do;
            batch_offset_lsed = query_start;
            // All !kUseQrQtrDorPipeline paths use per-batch compact dq_acc layout
            // QrQtrDor: direct write to dq_ptr (flat layout with per-nhead strides)
            if constexpr(kUseQrQtrDorPipeline)
                batch_offset_dq_acc = query_start * kargs.stride_dq;
            else if constexpr(!kIsDeterministic)
                batch_offset_dq_acc = query_start * kargs.hdim_q * kargs.nhead_q;
            else
                batch_offset_dq_acc = kargs.dq_acc_batch_offset_ptr[i_batch];
            batch_offset_dk = key_start * kargs.stride_dk;
            batch_offset_dv = key_start * kargs.stride_dv;
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                batch_offset_bias = query_start * kargs.stride_bias;
            }
            if constexpr(kHasBiasGrad)
            {
                batch_offset_dbias = query_start * kargs.stride_dbias;
            }
            else
            {
                batch_offset_dbias = key_start;
            }
            if constexpr(kIsStoreRandval)
            {
                batch_offset_randval = query_start * kargs.stride_randval;
            }

            // Priority: cu_seqlen_q_ptr > seqlen_q_ptr > physical_seqlen_q
            if(kargs.cu_seqlen_q_ptr != nullptr)
            {
                kargs.seqlen_q =
                    kargs.cu_seqlen_q_ptr[i_batch + 1] - kargs.cu_seqlen_q_ptr[i_batch];
            }
            else
            {
                kargs.seqlen_q =
                    kargs.seqlen_q_ptr ? kargs.seqlen_q_ptr[i_batch] : physical_seqlen_q;
            }

            // Priority: cu_seqlen_k_ptr > seqlen_k_ptr > seqstart_k
            if(kargs.cu_seqlen_k_ptr != nullptr)
            {
                kargs.seqlen_k =
                    kargs.cu_seqlen_k_ptr[i_batch + 1] - kargs.cu_seqlen_k_ptr[i_batch];
            }
            else if(kargs.seqlen_k_ptr != nullptr)
            {
                kargs.seqlen_k = kargs.seqlen_k_ptr[i_batch];
            }
            else
            {
                const auto adjusted_seqstart_k_ptr = kargs.seqstart_k_ptr + i_batch;
                kargs.seqlen_k = adjusted_seqstart_k_ptr[1] - adjusted_seqstart_k_ptr[0];
            }

            // skip if logical lengths are zero
            if(kargs.seqlen_q == 0 && kargs.seqlen_k == 0)
            {
                return;
            }

            // # of required blocks is different in each groups, terminate unnecessary blocks
            // earlier
            if constexpr(!kUseQrQtrDorPipeline)
                if(kargs.seqlen_k <= i_n0)
                    return;
        }
        else
        {
            batch_offset_q    = static_cast<long_index_t>(i_batch) * kargs.batch_stride_q;
            batch_offset_k    = static_cast<long_index_t>(i_batch) * kargs.batch_stride_k;
            batch_offset_v    = static_cast<long_index_t>(i_batch) * kargs.batch_stride_v;
            batch_offset_do   = static_cast<long_index_t>(i_batch) * kargs.batch_stride_do;
            batch_offset_lsed = static_cast<long_index_t>(i_batch) * kargs.batch_stride_lsed;

            if constexpr(kUseQrQtrDorPipeline)
                batch_offset_dq_acc = static_cast<long_index_t>(i_batch) * kargs.batch_stride_dq;
            else if constexpr(!kIsDeterministic)
                batch_offset_dq_acc = static_cast<long_index_t>(i_batch) * kargs.nhead_q *
                                      kargs.seqlen_q * kargs.hdim_q;
            else
                batch_offset_dq_acc = static_cast<long_index_t>(i_batch) * kargs.nhead_q *
                                      n_splits * kargs.seqlen_q * kargs.hdim_q;
            batch_offset_dk = static_cast<long_index_t>(i_batch) * kargs.batch_stride_dk;
            batch_offset_dv = static_cast<long_index_t>(i_batch) * kargs.batch_stride_dv;
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                batch_offset_bias = static_cast<long_index_t>(i_batch) * kargs.batch_stride_bias;
            }
            if constexpr(kHasBiasGrad)
            {
                batch_offset_dbias = static_cast<long_index_t>(i_batch) * kargs.batch_stride_dbias;
            }
            if constexpr(kIsStoreRandval)
            {
                batch_offset_randval =
                    static_cast<long_index_t>(i_batch) * kargs.batch_stride_randval;
            }
        }

        // for simplicity, batch stride we just modify the pointer
        const QDataType* q_ptr = reinterpret_cast<const QDataType*>(kargs.q_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_q +
                                 batch_offset_q;
        const KDataType* k_ptr =
            reinterpret_cast<const KDataType*>(kargs.k_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_k +
            batch_offset_k;
        const VDataType* v_ptr =
            reinterpret_cast<const VDataType*>(kargs.v_ptr) +
            static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_v +
            batch_offset_v;
        const LSEDataType* lse_ptr = reinterpret_cast<const LSEDataType*>(kargs.lse_ptr) +
                                     static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_lsed +
                                     batch_offset_lsed;
        const DDataType* d_ptr = reinterpret_cast<const DDataType*>(kargs.d_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_lsed +
                                 batch_offset_lsed;
        const OGradDataType* do_ptr = reinterpret_cast<const OGradDataType*>(kargs.do_ptr) +
                                      static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_do +
                                      batch_offset_do;
        auto dk_ptr = reinterpret_cast<KGradDataType*>(kargs.dk_ptr) +
                      static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_dk + batch_offset_dk;
        auto dv_ptr = reinterpret_cast<VGradDataType*>(kargs.dv_ptr) +
                      static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_dv + batch_offset_dv;

        // Q/K/V/LSE/D/dO/dQ/dK/dV DRAM and DRAM window
        const auto q_dram_naive = make_naive_tensor_view<address_space_enum::global>(
            q_ptr,
            make_tuple(kargs.seqlen_q, kargs.hdim_q),
            make_tuple(kargs.stride_q, 1),
            number<FmhaPipeline::kAlignmentQ>{},
            number<1>{});
        const auto q_dram = pad_tensor_view(
            q_dram_naive,
            make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kQKHeaddim>{}),
            sequence<false, (kPadHeadDimQ > 0)>{});

        const auto k_dram_naive = make_naive_tensor_view<address_space_enum::global>(
            k_ptr,
            make_tuple(kargs.seqlen_k, kargs.hdim_q),
            make_tuple(kargs.stride_k, 1),
            number<FmhaPipeline::kAlignmentK>{},
            number<1>{});
        const auto k_dram = pad_tensor_view(
            k_dram_naive,
            make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kQKHeaddim>{}),
            sequence<false, (kPadHeadDimQ > 0)>{});

        const auto v_dram = [&]() {
            const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                v_ptr,
                make_tuple(kargs.seqlen_k, kargs.hdim_v),
                make_tuple(kargs.stride_v, 1),
                number<FmhaPipeline::kAlignmentV>{},
                number<1>{});
            return pad_tensor_view(
                v_dram_naive,
                make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kVHeaddim>{}),
                sequence<false, (kPadHeadDimV > 0)>{});
        }();

        // lse and d should be fine to read unpaded data as they are not on the reduction dimension
        const auto lse_dram = make_naive_tensor_view_packed<address_space_enum::global>(
            lse_ptr, make_tuple(kargs.seqlen_q), number<FmhaPipeline::kM0>{});

        const auto d_dram = make_naive_tensor_view_packed<address_space_enum::global>(
            d_ptr, make_tuple(kargs.seqlen_q), number<FmhaPipeline::kM0>{});

        const auto do_dram_naive = make_naive_tensor_view<address_space_enum::global>(
            do_ptr,
            make_tuple(kargs.seqlen_q, kargs.hdim_v),
            make_tuple(kargs.stride_do, 1),
            number<FmhaPipeline::kAlignmentOGrad>{},
            number<1>{});
        const auto do_dram = pad_tensor_view(
            do_dram_naive,
            make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kVHeaddim>{}),
            sequence<false, (kPadHeadDimV > 0)>{});

        auto q_dram_window = make_tile_window(
            q_dram,
            make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kQKHeaddim>{}),
            {0, 0});

        auto k_dram_window = make_tile_window(
            k_dram,
            make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kQKHeaddim>{}),
            {i_n0, 0});

        auto v_dram_window = make_tile_window(
            v_dram,
            make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kVHeaddim>{}),
            {i_n0, 0});

        auto do_dram_window = make_tile_window(
            do_dram,
            make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kVHeaddim>{}),
            {0, 0});

        auto dq_dram_window = [&, i_nhead_ = i_nhead]() {
            constexpr bool kUseKSplit = !kUseQrQtrDorPipeline && kIsDeterministic;
            using DType = std::conditional_t<kUseQrQtrDorPipeline, QGradDataType, AccDataType>;

            auto dq_acc_ptr = reinterpret_cast<DType*>(kargs.dq_acc_ptr) + [&]() {
                if constexpr(kUseQrQtrDorPipeline)
                {
                    return batch_offset_dq_acc +
                           static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_dq;
                }
                else if constexpr(!kIsDeterministic)
                {
                    return batch_offset_dq_acc +
                           static_cast<long_index_t>(i_nhead_) * physical_seqlen_q * kargs.hdim_q;
                }
                else
                {
                    const long_index_t split_stride =
                        static_cast<long_index_t>(physical_seqlen_q) * kargs.hdim_q;
                    const auto nsplits = [&]() {
                        if constexpr(!kIsGroupMode)
                            return n_splits; // batch persistent: passed from nsplits_ptr[0]
                        else if constexpr(kUsePersistent)
                            return n_splits; // group persistent: passed from nsplits_ptr[ibatch]
                        else
                            return integer_divide_ceil(kargs.seqlen_k,
                                                       FmhaPipeline::kN0); // group non-persistent
                    }();
                    return batch_offset_dq_acc + (i_nhead_ * nsplits + i_split) * split_stride;
                }
            }();

            // kUseKSplit && !kUsePersistent is true only for QrQtrDor+deterministic,
            // which writes dq directly (not through dq_acc splits) — use 'set'.
            // All other deterministic paths are persistent and use 'atomic_add':
            //   a single CU may process multiple chunks of the same (batch, head, isplit)
            //   sequentially, so contributions must accumulate rather than overwrite.
            // Non-deterministic paths also use 'atomic_add' (kUseKSplit=false).
            constexpr auto DstInMemOp = conditional_expr<(kUseKSplit && !kUsePersistent)>(
                memory_operation_enum::set, memory_operation_enum::atomic_add);
            const index_t stride_dq_acc = [&]() {
                if constexpr(kUseQrQtrDorPipeline)
                    return kargs.stride_dq;
                else
                    return kargs.hdim_q;
            }();
            const auto dq_acc_dram_naive =
                make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    dq_acc_ptr,
                    make_tuple(kargs.seqlen_q, kargs.hdim_q),
                    make_tuple(stride_dq_acc, 1),
                    number<FmhaPipeline::kAlignmentQGrad>{},
                    number<1>{});
            const auto dq_acc_dram = pad_tensor_view(
                dq_acc_dram_naive,
                make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kQKHeaddim>{}),
                sequence<false, (kPadHeadDimQ > 0)>{});
            return make_tile_window(
                dq_acc_dram,
                make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kQKHeaddim>{}),
                {0, 0});
        }();

        auto lse_dram_window =
            make_tile_window(lse_dram, make_tuple(number<FmhaPipeline::kM0>{}), {0});

        auto d_dram_window = make_tile_window(d_dram, make_tuple(number<FmhaPipeline::kM0>{}), {0});

        /// FIXME: Before C++20, capturing structured binding variables are not supported. Remove
        /// following copy capture of the 'i_nhead' if in C++20
        constexpr auto bias_dram_window_lengths =
            make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN0>{});
        const auto bias_dram_window = [&, i_nhead_ = i_nhead]() {
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                const BiasDataType* bias_ptr =
                    reinterpret_cast<const BiasDataType*>(kargs.bias_ptr) +
                    static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_bias +
                    batch_offset_bias;

                const auto bias_dram = [&]() {
                    const auto bias_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                        bias_ptr,
                        make_tuple(kargs.seqlen_q, kargs.seqlen_k),
                        make_tuple(kargs.stride_bias, 1),
                        number<FmhaPipeline::kAlignmentBias>{},
                        number<1>{});

                    return pad_tensor_view(
                        bias_dram_naive, bias_dram_window_lengths, sequence<false, true>{});
                }();

                return make_tile_window(bias_dram, bias_dram_window_lengths, {0, i_n0});
            }
            else
            {
                return make_null_tile_window(bias_dram_window_lengths);
            }
        }();

        auto dbias_dram_window = [&, i_nhead_ = i_nhead]() {
            if constexpr(kHasBiasGrad)
            {
                BiasGradDataType* dbias_ptr =
                    reinterpret_cast<BiasGradDataType*>(kargs.dbias_ptr) +
                    static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_dbias +
                    batch_offset_dbias;

                auto dbias_dram = [&]() {
                    const auto dbias_dram_naive =
                        make_naive_tensor_view<address_space_enum::global>(
                            dbias_ptr,
                            make_tuple(kargs.seqlen_q, kargs.seqlen_k),
                            make_tuple(kargs.stride_dbias, 1),
                            number<FmhaPipeline::kAlignmentBias>{},
                            number<1>{});

                    return pad_tensor_view(
                        dbias_dram_naive, bias_dram_window_lengths, sequence<false, true>{});
                }();

                return make_tile_window(dbias_dram, bias_dram_window_lengths, {0, i_n0});
            }
            else
            {
                return make_null_tile_window(bias_dram_window_lengths);
            }
        }();

        // WA i_batch capture structure binding before c++20
        auto position_encoding = [&, i_batch_ = i_batch, i_nhead_ = i_nhead]() {
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
            {
                // data loading, shared by entire wg
                // TODO: how to use s_read?
                AccDataType slope = *(reinterpret_cast<const AccDataType*>(kargs.alibi_slope_ptr) +
                                      i_batch_ * kargs.alibi_slope_stride + i_nhead_);
                slope *= ck_tile::log2e_v<>;
                if constexpr(kHasMask)
                {
                    return make_alibi_from_lr_mask<AccDataType, false>(slope,
                                                                       kargs.window_size_left,
                                                                       kargs.window_size_right,
                                                                       kargs.seqlen_q,
                                                                       kargs.seqlen_k,
                                                                       kargs.mask_type);
                }
                else
                {
                    return Alibi<AccDataType, false>{
                        slope, kargs.seqlen_q, kargs.seqlen_k, AlibiMode::FROM_BOTTOM_RIGHT};
                }
            }
            else
            {
                return EmptyPositionEncoding<AccDataType>{};
            }
        }();

        // dropout
        float rp_undrop       = 1;
        float scale_rp_undrop = 1;
        if constexpr(kHasDropout)
        {
            rp_undrop       = kargs.rp_undrop;
            scale_rp_undrop = kargs.scale_rp_undrop;
        }
        auto dropout = [&, i_nhead_ = i_nhead, i_batch_ = i_batch]() {
            if constexpr(kHasDropout)
            {
                return FmhaDropout{i_batch_,
                                   i_nhead_,
                                   kargs.nhead_q,
                                   kargs.is_drop_seed_offset_from_host ? kargs.drop_seed.val
                                                                       : *kargs.drop_seed.ptr,
                                   kargs.is_drop_seed_offset_from_host ? kargs.drop_offset.val
                                                                       : *kargs.drop_offset.ptr,
                                   kargs.rp_undrop,
                                   kargs.p_undrop_in_uint8_t};
            }
            else
            {
                return FmhaDropout{};
            };
        }();

        auto randval_dram_window = [&, i_nhead_ = i_nhead]() {
            constexpr auto randval_dram_window_lengths =
                make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN0>{});
            if constexpr(kIsStoreRandval)
            {
                RandValOutputDataType* rand_val_ptr =
                    reinterpret_cast<RandValOutputDataType*>(kargs.rand_val_ptr) +
                    static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_randval +
                    batch_offset_randval;

                const auto randval_dram = [&]() {
                    const auto randval_dram_naive =
                        make_naive_tensor_view<address_space_enum::global>(
                            rand_val_ptr,
                            make_tuple(kargs.seqlen_q, kargs.seqlen_k),
                            make_tuple(kargs.stride_randval, 1),
                            number<1>{},
                            number<1>{});

                    return pad_tensor_view(
                        randval_dram_naive, randval_dram_window_lengths, sequence<false, true>{});
                }();

                return make_tile_window(randval_dram, randval_dram_window_lengths, {0, i_n0});
            }
            else
            {
                return make_null_tile_window(randval_dram_window_lengths);
            }
        }();

        FmhaMask mask = [&]() {
            if constexpr(kHasMask)
                return ck_tile::make_generic_attention_mask_from_lr_window<FmhaMask>(
                    kargs.window_size_left,
                    kargs.window_size_right,
                    kargs.seqlen_q,
                    kargs.seqlen_k,
                    kargs.mask_type == GenericAttentionMaskEnum::MASK_FROM_TOP_LEFT);
            else
                return FmhaMask{kargs.seqlen_q, kargs.seqlen_k};
        }();

        auto dk_dram = [&]() {
            const auto dk_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                dk_ptr,
                make_tuple(kargs.seqlen_k, kargs.hdim_q),
                make_tuple(kargs.stride_dk, 1),
                number<FmhaPipeline::kAlignmentKGrad>{},
                number<1>{});

            return pad_tensor_view(
                dk_dram_naive,
                make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kQKHeaddim>{}),
                sequence<false, (kPadHeadDimQ > 0)>{});
        }();

        auto dv_dram = [&]() {
            const auto dv_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                dv_ptr,
                make_tuple(kargs.seqlen_k, kargs.hdim_v),
                make_tuple(kargs.stride_dv, 1),
                number<FmhaPipeline::kAlignmentVGrad>{},
                number<1>{});

            return pad_tensor_view(
                dv_dram_naive,
                make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kVHeaddim>{}),
                sequence<false, (kPadHeadDimV > 0)>{});
        }();

        auto dk_dram_window = make_tile_window(
            dk_dram,
            make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kQKHeaddim>{}),
            {i_n0, 0});

        auto dv_dram_window = make_tile_window(
            dv_dram,
            make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kVHeaddim>{}),
            {i_n0, 0});
        if constexpr(!kUseQrQtrDorPipeline)
        {
            auto [dk_acc_tile, dv_acc_tile] = FmhaPipeline{}(smem_ptr,
                                                             q_dram_window,
                                                             k_dram_window,
                                                             v_dram_window,
                                                             bias_dram_window,
                                                             randval_dram_window,
                                                             do_dram_window,
                                                             lse_dram_window,
                                                             d_dram_window,
                                                             dq_dram_window,
                                                             dbias_dram_window,
                                                             mask,
                                                             position_encoding,
                                                             kargs.raw_scale,
                                                             kargs.scale,
                                                             rp_undrop,
                                                             scale_rp_undrop,
                                                             dropout);

#if defined(__gfx11__) || defined(__gfx12__)
            // Workaround for a compiler bug (SWDEV-559729): v_wmma instructions can be incorrectly
            // placed in divergent branches used to store padded tensors (when some lanes are
            // inactive due to padding). Inline asm with dummy dependencies on VGPRs of the tensors
            // prevents the compiler doing this.
            if constexpr(kPadHeadDimQ > 0)
            {
                impl::insert_dummy_dep(dk_acc_tile.get_thread_buffer());
            }
            if constexpr(kPadHeadDimV > 0)
            {
                impl::insert_dummy_dep(dv_acc_tile.get_thread_buffer());
            }
#endif

            KGradEpiloguePipeline{}(dk_dram_window, dk_acc_tile, nullptr);
            VGradEpiloguePipeline{}(dv_dram_window, dv_acc_tile, nullptr);
        }
        else
        {
            FmhaPipeline{}(smem_ptr,
                           q_dram_window,
                           k_dram_window,
                           v_dram_window,
                           bias_dram_window,
                           randval_dram_window,
                           do_dram_window,
                           lse_dram_window,
                           d_dram_window,
                           dq_dram_window,
                           dk_dram_window,
                           dv_dram_window,
                           dbias_dram_window,
                           QGradEpiloguePipeline{},
                           KGradEpiloguePipeline{},
                           VGradEpiloguePipeline{},
                           mask,
                           position_encoding,
                           kargs.raw_scale,
                           kargs.scale,
                           rp_undrop,
                           scale_rp_undrop,
                           dropout);
        }
    }
};

template <typename FmhaBwdOGradDotO_>
struct FmhaBwdOGradDotOKernel
{
    using FmhaBwdOGradDotO                        = ck_tile::remove_cvref_t<FmhaBwdOGradDotO_>;
    static constexpr ck_tile::index_t kBlockSize  = FmhaBwdOGradDotO::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = FmhaBwdOGradDotO::kBlockPerCu;
    static constexpr ck_tile::index_t kM0         = kBlockSize;
    static constexpr ck_tile::index_t kVHeaddim   = FmhaBwdOGradDotO::kVHeaddim;

    using DDataType     = ck_tile::remove_cvref_t<typename FmhaBwdOGradDotO::DDataType>;
    using ODataType     = ck_tile::remove_cvref_t<typename FmhaBwdOGradDotO::ODataType>;
    using OGradDataType = ck_tile::remove_cvref_t<typename FmhaBwdOGradDotO::OGradDataType>;
    using LSEDataType   = ck_tile::remove_cvref_t<typename FmhaBwdOGradDotO::LSEDataType>;

    static constexpr bool kIsGroupMode = FmhaBwdOGradDotO::kIsGroupMode;
    static constexpr bool kPadSeqLenQ  = FmhaBwdOGradDotO::kPadSeqLenQ;
    static constexpr bool kPadHeadDimV = FmhaBwdOGradDotO::kPadHeadDimV;

    // clang-format off
    template <typename T> struct t2s;
    template <> struct t2s<float> { static constexpr const char * name = "fp32"; };
    template <> struct t2s<ck_tile::fp16_t> { static constexpr const char * name = "fp16"; };
    template <> struct t2s<ck_tile::bf16_t> { static constexpr const char * name = "bf16"; };
    // clang-format on

    CK_TILE_HOST static std::string GetName()
    {
        // sync with generate.py
        // clang-format off
        
        #define _SS_  std::string
        #define _TS_  std::to_string
        auto pn = [&] () {
            std::string n;
            if (kPadSeqLenQ) n += "s";
            if (kPadHeadDimV) n += "dv";
            return n.empty() ? n : std::string("p") + n; }();
        return
            _SS_("fmha_bwd_dot_do_o_d") + _TS_(kVHeaddim) + "_" + _SS_(t2s<ODataType>::name) +
            "_b" + _TS_(kM0) + "_" + (kIsGroupMode ? "group" : "batch") + "_" +
            ("o" + _TS_(kBlockPerCu)) + (pn.empty() ? "_npad" : "_" + pn);
        #undef _SS_
        #undef _TS_
        // clang-format on
    }

    // kargs use aggregate initializer, so no constructor will provided
    // use inheritance to minimize karg size
    // user need to use MakeKargs() function to create kargs.
    struct FmhaBwdOGradDotOCommonKargs
    {
        const void* o_ptr;
        const void* do_ptr;
        void* d_ptr;
        const void* lse_ptr; // log-sum-exp from forward pass, shape [batch, nhead, seqlen_q]
        const LSEDataType* sink_ptr; // sink scores, shape [batch, nhead]; nullptr disables sink
        LSEDataType* d_sink_ptr; // sink gradient output, shape [nhead]; nullptr disables sink grad

        float p_undrop;

        ck_tile::index_t seqlen_q;
        ck_tile::index_t hdim_v;
        ck_tile::index_t nhead; // used to index sink_ptr / d_sink_ptr

        ck_tile::index_t stride_do;
        ck_tile::index_t stride_o;

        ck_tile::index_t nhead_stride_do;
        ck_tile::index_t nhead_stride_o;
        // LSE and D always share the same layout; this stride covers both.
        ck_tile::index_t nhead_stride_lsed;
    };

    struct FmhaBwdOGradDotOBatchModeKargs : FmhaBwdOGradDotOCommonKargs
    {
        ck_tile::index_t batch_stride_do;
        ck_tile::index_t batch_stride_o;
        // LSE and D always share the same layout; this stride covers both.
        ck_tile::index_t batch_stride_lsed;
    };

    struct FmhaBwdOGradDotOGroupModeKargs : FmhaBwdOGradDotOCommonKargs
    {
        const int32_t* seqstart_q_ptr;
        const int32_t* seqlen_q_ptr;    // per-batch actual length [batch]
        const int32_t* cu_seqlen_q_ptr; // cumulative seqlen [batch+1], optional
    };

    using Kargs = std::
        conditional_t<kIsGroupMode, FmhaBwdOGradDotOGroupModeKargs, FmhaBwdOGradDotOBatchModeKargs>;

    template <bool Cond = !kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* o_ptr,
              const void* do_ptr,
              void* d_ptr,
              const void* lse_ptr,
              const void* sink_ptr,
              void* d_sink_ptr,
              float p_undrop,
              ck_tile::index_t seqlen_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t nhead,
              ck_tile::index_t stride_do,
              ck_tile::index_t stride_o,
              ck_tile::index_t nhead_stride_do,
              ck_tile::index_t nhead_stride_o,
              ck_tile::index_t nhead_stride_lsed,
              ck_tile::index_t batch_stride_do,
              ck_tile::index_t batch_stride_o,
              ck_tile::index_t batch_stride_lsed)
    {
        Kargs kargs{{o_ptr,
                     do_ptr,
                     d_ptr,
                     lse_ptr,
                     reinterpret_cast<const LSEDataType*>(sink_ptr),
                     reinterpret_cast<LSEDataType*>(d_sink_ptr),
                     p_undrop,
                     seqlen_q,
                     hdim_v,
                     nhead,
                     stride_do,
                     stride_o,
                     nhead_stride_do,
                     nhead_stride_o,
                     nhead_stride_lsed},
                    batch_stride_do,
                    batch_stride_o,
                    batch_stride_lsed};

        return kargs;
    }

    template <bool Cond = kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* o_ptr,
              const void* do_ptr,
              void* d_ptr,
              const void* lse_ptr,
              const void* sink_ptr,
              void* d_sink_ptr,
              float p_undrop,
              const void* seqstart_q_ptr,
              const void* seqlen_q_ptr,
              const void* cu_seqlen_q_ptr,
              ck_tile::index_t hdim_v,
              ck_tile::index_t nhead,
              ck_tile::index_t stride_do,
              ck_tile::index_t stride_o,
              ck_tile::index_t nhead_stride_do,
              ck_tile::index_t nhead_stride_o,
              ck_tile::index_t nhead_stride_lsed)
    {
        Kargs kargs{{o_ptr,
                     do_ptr,
                     d_ptr,
                     lse_ptr,
                     reinterpret_cast<const LSEDataType*>(sink_ptr),
                     reinterpret_cast<LSEDataType*>(d_sink_ptr),
                     p_undrop,
                     -1, // seqlen will be updated by another pointer
                     hdim_v,
                     nhead,
                     stride_do,
                     stride_o,
                     nhead_stride_do,
                     nhead_stride_o,
                     nhead_stride_lsed},
                    reinterpret_cast<const int32_t*>(seqstart_q_ptr),
                    reinterpret_cast<const int32_t*>(seqlen_q_ptr),
                    reinterpret_cast<const int32_t*>(cu_seqlen_q_ptr)};

        return kargs;
    }

    CK_TILE_HOST static constexpr auto
    GridSize(ck_tile::index_t batch_size_, ck_tile::index_t nhead_, ck_tile::index_t seqlen_q_)
    {
        return dim3(ck_tile::integer_divide_ceil(seqlen_q_, kM0), nhead_, batch_size_);
    }

    CK_TILE_DEVICE static constexpr auto GetTileIndex()
    {
        const index_t i_block = blockIdx.x;
        const index_t i_nhead = blockIdx.y;
        const index_t i_batch = blockIdx.z;

        return ck_tile::make_tuple(i_block, i_nhead, i_batch);
    }

    CK_TILE_HOST static dim3 BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize() { return 0; }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        // divide problem
        const auto [i_tile_m, i_nhead, i_batch] = GetTileIndex();

        const index_t i_m0 = amd_wave_read_first_lane(i_tile_m * kM0);

        long_index_t batch_offset_o    = 0;
        long_index_t batch_offset_do   = 0;
        long_index_t batch_offset_lsed = 0;

        if constexpr(kIsGroupMode)
        {
            // get starting offset for each batch
            const long_index_t query_start = kargs.seqstart_q_ptr[i_batch];

            batch_offset_o    = query_start * kargs.stride_o;
            batch_offset_do   = query_start * kargs.stride_do;
            batch_offset_lsed = query_start;

            // Priority: cu_seqlen_q_ptr > seqlen_q_ptr > physical_seqlen_q
            if(kargs.cu_seqlen_q_ptr != nullptr)
            {
                kargs.seqlen_q =
                    kargs.cu_seqlen_q_ptr[i_batch + 1] - kargs.cu_seqlen_q_ptr[i_batch];
            }
            else
            {
                // get real # queries & # keys under group mode
                const auto adjusted_seqstart_q_ptr = kargs.seqstart_q_ptr + i_batch;
                const ck_tile::index_t physical_seqlen_q =
                    adjusted_seqstart_q_ptr[1] - adjusted_seqstart_q_ptr[0];
                kargs.seqlen_q = kargs.seqlen_q_ptr
                                     ? static_cast<ck_tile::index_t>(kargs.seqlen_q_ptr[i_batch])
                                     : physical_seqlen_q;
            }

            // # of required blocks is different in each groups, terminate unnecessary blocks
            // earlier
            if(kargs.seqlen_q <= i_m0)
            {
                return;
            }
        }
        else
        {
            batch_offset_o    = static_cast<long_index_t>(i_batch) * kargs.batch_stride_o;
            batch_offset_do   = static_cast<long_index_t>(i_batch) * kargs.batch_stride_do;
            batch_offset_lsed = static_cast<long_index_t>(i_batch) * kargs.batch_stride_lsed;
        }

        // Read per-head sink score and convert to log2 domain so the pipeline can use exp2.
        // Pre-multiply by log2e so that exp2(sink_value - log2e*lse) == exp(raw_sink - lse).
        // -inf is left unchanged (log2e * -inf == -inf) to keep P_sink -> 0 when sink is disabled.
        const LSEDataType sink_value =
            kargs.sink_ptr != nullptr
                ? log2e_v<LSEDataType> *
                      kargs.sink_ptr[static_cast<long_index_t>(i_batch) * kargs.nhead + i_nhead]
                : -numeric<LSEDataType>::infinity();

        // for simplicity, batch stride we just modify the pointer
        const ODataType* o_ptr = reinterpret_cast<const ODataType*>(kargs.o_ptr) +
                                 static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_o +
                                 batch_offset_o;
        const OGradDataType* do_ptr = reinterpret_cast<const OGradDataType*>(kargs.do_ptr) +
                                      static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_do +
                                      batch_offset_do;
        const LSEDataType* lse_ptr = reinterpret_cast<const LSEDataType*>(kargs.lse_ptr) +
                                     static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_lsed +
                                     batch_offset_lsed;

        DDataType* d_ptr = reinterpret_cast<DDataType*>(kargs.d_ptr) +
                           static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_lsed +
                           batch_offset_lsed;

        // O/dO/D DRAM and DRAM window
        const auto o_dram = [&]() {
            auto o_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                o_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_v),
                make_tuple(kargs.stride_o, 1),
                number<FmhaBwdOGradDotO::kAlignmentO>{},
                number<1>{});
            return pad_tensor_view(o_dram_naive,
                                   make_tuple(number<kM0>{}, number<kVHeaddim>{}),
                                   sequence<kPadSeqLenQ, kPadHeadDimV>{});
        }();
        const auto do_dram = [&]() {
            auto do_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                do_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_v),
                make_tuple(kargs.stride_do, 1),
                number<FmhaBwdOGradDotO::kAlignmentOGrad>{},
                number<1>{});
            return pad_tensor_view(do_dram_naive,
                                   make_tuple(number<kM0>{}, number<kVHeaddim>{}),
                                   sequence<kPadSeqLenQ, kPadHeadDimV>{});
        }();
        auto d_dram = [&]() {
            const auto d_dram_naive = make_naive_tensor_view_packed<address_space_enum::global>(
                d_ptr, make_tuple(kargs.seqlen_q), number<1>{});
            return pad_tensor_view(
                d_dram_naive, make_tuple(number<kM0>{}), sequence<kPadSeqLenQ>{});
        }();

        auto o_dram_window =
            make_tile_window(o_dram, make_tuple(number<kM0>{}, number<kVHeaddim>{}), {i_m0, 0});
        auto do_dram_window =
            make_tile_window(do_dram, make_tuple(number<kM0>{}, number<kVHeaddim>{}), {i_m0, 0});
        auto d_dram_window = make_tile_window(d_dram, make_tuple(number<kM0>{}), {i_m0});

        // nullptr when sink grad is disabled; the pipeline checks this to skip the sink path
        LSEDataType* atomic_sink_grad_ptr =
            kargs.d_sink_ptr == nullptr ? nullptr : kargs.d_sink_ptr + i_nhead;

        // lse_ptr is always valid (also needed by the main bwd kernel).
        // The actual load happens inside the pipeline only when atomic_sink_grad_ptr != nullptr.
        auto lse_dram = [&]() {
            const auto lse_dram_naive = make_naive_tensor_view_packed<address_space_enum::global>(
                lse_ptr, make_tuple(kargs.seqlen_q), number<1>{});
            return pad_tensor_view(
                lse_dram_naive, make_tuple(number<kM0>{}), sequence<kPadSeqLenQ>{});
        }();
        auto lse_dram_window = make_tile_window(lse_dram, make_tuple(number<kM0>{}), {i_m0});

        FmhaBwdOGradDotO{}(o_dram_window,
                           do_dram_window,
                           lse_dram_window,
                           d_dram_window,
                           sink_value,
                           kargs.p_undrop,
                           atomic_sink_grad_ptr);
    }
};

template <typename FmhaBwdConvertQGrad_>
struct FmhaBwdConvertQGradKernel
{
    using FmhaBwdConvertQGrad                     = ck_tile::remove_cvref_t<FmhaBwdConvertQGrad_>;
    static constexpr ck_tile::index_t kBlockSize  = FmhaBwdConvertQGrad::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = FmhaBwdConvertQGrad::kBlockPerCu;
    static constexpr ck_tile::index_t kM0         = FmhaBwdConvertQGrad::kM0;
    static constexpr ck_tile::index_t kQKHeaddim  = FmhaBwdConvertQGrad::kQKHeaddim;

    using AccDataType   = ck_tile::remove_cvref_t<typename FmhaBwdConvertQGrad::AccDataType>;
    using QGradDataType = ck_tile::remove_cvref_t<typename FmhaBwdConvertQGrad::QGradDataType>;

    static constexpr bool kIsGroupMode     = FmhaBwdConvertQGrad::kIsGroupMode;
    static constexpr bool kPadSeqLenQ      = FmhaBwdConvertQGrad::kPadSeqLenQ;
    static constexpr bool kPadHeadDimQ     = FmhaBwdConvertQGrad::kPadHeadDimQ;
    static constexpr bool kIsDeterministic = FmhaBwdConvertQGrad::kIsDeterministic;
    static constexpr bool kUsePersistent   = kIsDeterministic;
    using WorkspaceManager = FmhaBwdWorkspaceManager<AccDataType, kIsGroupMode, kIsDeterministic>;

    // clang-format off
    template <typename T> struct t2s;
    template <> struct t2s<float> { static constexpr const char * name = "fp32"; };
    template <> struct t2s<ck_tile::fp16_t> { static constexpr const char * name = "fp16"; };
    template <> struct t2s<ck_tile::bf16_t> { static constexpr const char * name = "bf16"; };
    // clang-format on

    CK_TILE_HOST static std::string GetName()
    {
        // sync with generate.py
        // clang-format off
        
        #define _SS_  std::string
        #define _TS_  std::to_string
        auto pn = [&] () {
            std::string n;
            if (kPadSeqLenQ) n += "s";
            if (kPadHeadDimQ) n += "d";
            return n.empty() ? n : std::string("p") + n; }();
        return
            _SS_("fmha_bwd_convert_dq_d") + _TS_(kQKHeaddim) + "_"
            + _SS_(t2s<QGradDataType>::name) + "_"
            + "b" + _TS_(kM0) + "_"
            + (kIsGroupMode ? "group" : "batch") + "_"
            + ("o" + _TS_(kBlockPerCu)) + (pn.empty() ? "_npad" : "_" + pn)
            + (kIsDeterministic ? "_deterministic" : "_ndeterministic") ;
        #undef _SS_
        #undef _TS_
        // clang-format on
    }

    // to avoid duplicated base class prblem, introduce an template arg
    template <ck_tile::index_t I>
    struct FmhaBwdConvertQGradEmptyKargs
    {
    };

    // kargs use aggregate initializer, so no constructor will provided
    // use inheritance to minimize karg size
    // user need to use MakeKargs() function to create kargs.
    struct FmhaBwdConvertQGradCommonKargs
    {
        const void* dq_acc_ptr;
        void* dq_ptr;

        ck_tile::index_t nhead_q;
        ck_tile::index_t seqlen_q;
        ck_tile::index_t seqlen_k;
        ck_tile::index_t hdim_q;

        ck_tile::index_t stride_dq;
        ck_tile::index_t nhead_stride_dq;
    };

    struct FmhaBwdConvertQGradDeterministicKargs
    {
        const index_t* nsplits_ptr;
    };

    struct FmhaBwdConvertQGradBatchModeKargs
        : FmhaBwdConvertQGradCommonKargs,
          std::conditional_t<kIsDeterministic,
                             FmhaBwdConvertQGradDeterministicKargs,
                             FmhaBwdConvertQGradEmptyKargs<0>>
    {
        index_t batch_stride_dq;
    };

    struct FmhaBwdConvertQGradGroupModeKargs
        : FmhaBwdConvertQGradCommonKargs,
          std::conditional_t<kIsDeterministic,
                             FmhaBwdConvertQGradDeterministicKargs,
                             FmhaBwdConvertQGradEmptyKargs<0>>
    {
        const int32_t* seqstart_q_ptr;
        const int32_t* seqstart_k_ptr;
        const int32_t* seqlen_q_ptr;    // per-batch actual length [batch]
        const int32_t* seqlen_k_ptr;    // per-batch actual length [batch]
        const int32_t* cu_seqlen_q_ptr; // cumulative seqlen [batch+1], optional
        const int32_t* cu_seqlen_k_ptr; // cumulative seqlen [batch+1], optional
        // per-batch element offset into compact dq_acc buffer
        const long_index_t* dq_acc_batch_offset_ptr;
    };

    using Kargs = std::conditional_t<kIsGroupMode,
                                     FmhaBwdConvertQGradGroupModeKargs,
                                     FmhaBwdConvertQGradBatchModeKargs>;

    template <bool Cond = !kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* workspace,
              void* dq_ptr,
              ck_tile::index_t batch_size,
              ck_tile::index_t nhead_q,
              ck_tile::index_t seqlen_q,
              ck_tile::index_t seqlen_k,
              ck_tile::index_t hdim_q,
              ck_tile::index_t stride_dq,
              ck_tile::index_t nhead_stride_dq,
              ck_tile::index_t batch_stride_dq)
    {
        const uint8_t* ws = reinterpret_cast<const uint8_t*>(workspace);
        Kargs kargs{
            {ws + WorkspaceManager::template GetDqAccDataOffset<false>(batch_size),
             dq_ptr,
             nhead_q,
             seqlen_q,
             seqlen_k,
             hdim_q,
             stride_dq,
             nhead_stride_dq},
            {},
            batch_stride_dq,
        };
        if constexpr(kIsDeterministic)
        {
            kargs.nsplits_ptr = reinterpret_cast<const index_t*>(
                ws + WorkspaceManager::GetDqAccSplitsOffset(batch_size));
        }

        return kargs;
    }

    template <bool Cond = kIsGroupMode>
    CK_TILE_HOST static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(const void* workspace,
              void* dq_ptr,
              ck_tile::index_t batch_size,
              ck_tile::index_t nhead_q,
              const void* seqstart_q_ptr,
              const void* seqstart_k_ptr,
              const void* seqlen_q_ptr,
              const void* seqlen_k_ptr,
              const void* cu_seqlen_q_ptr,
              const void* cu_seqlen_k_ptr,
              ck_tile::index_t hdim_q,
              ck_tile::index_t stride_dq,
              ck_tile::index_t nhead_stride_dq)
    {
        const uint8_t* ws = reinterpret_cast<const uint8_t*>(workspace);
        Kargs kargs{{ws + WorkspaceManager::template GetDqAccDataOffset<false>(batch_size),
                     dq_ptr,
                     nhead_q,
                     -1, // seqlen will be updated by another pointer
                     -1, //
                     hdim_q,
                     stride_dq,
                     nhead_stride_dq},
                    {},
                    reinterpret_cast<const int32_t*>(seqstart_q_ptr),
                    reinterpret_cast<const int32_t*>(seqstart_k_ptr),
                    reinterpret_cast<const int32_t*>(seqlen_q_ptr),
                    reinterpret_cast<const int32_t*>(seqlen_k_ptr),
                    reinterpret_cast<const int32_t*>(cu_seqlen_q_ptr),
                    reinterpret_cast<const int32_t*>(cu_seqlen_k_ptr),
                    reinterpret_cast<const long_index_t*>(
                        ws + WorkspaceManager::template GetDqAccOffsetsOffset<false>(batch_size))};

        if constexpr(kIsDeterministic)
        {
            kargs.nsplits_ptr = reinterpret_cast<const index_t*>(
                ws + WorkspaceManager::GetDqAccSplitsOffset(batch_size));
        }

        return kargs;
    }

    CK_TILE_HOST static constexpr auto
    GridSize(ck_tile::index_t batch_size_, ck_tile::index_t nhead_, ck_tile::index_t seqlen_q_)
    {
        return dim3(ck_tile::integer_divide_ceil(seqlen_q_, kM0), nhead_, batch_size_);
    }

    CK_TILE_DEVICE static constexpr auto GetTileIndex()
    {
        const index_t i_block = blockIdx.x;
        const index_t i_nhead = blockIdx.y;
        const index_t i_batch = blockIdx.z;

        return ck_tile::make_tuple(i_block, i_nhead, i_batch);
    }

    CK_TILE_HOST static dim3 BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize() { return 0; }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        // divide problem
        const auto [i_tile_m, i_nhead, i_batch] = GetTileIndex();

        const index_t i_m0 = amd_wave_read_first_lane(i_tile_m * kM0);

        long_index_t batch_offset_dq     = 0;
        long_index_t batch_offset_dq_acc = 0;
        index_t physical_seqlen_q        = 0;
        if constexpr(kIsGroupMode)
        {
            const long_index_t query_start = kargs.seqstart_q_ptr[i_batch];
            physical_seqlen_q              = kargs.seqstart_q_ptr[i_batch + 1] - query_start;
            // get starting offset for each batch
            batch_offset_dq = query_start * kargs.stride_dq;
            if constexpr(!kIsDeterministic)
                batch_offset_dq_acc = query_start * kargs.hdim_q * kargs.nhead_q;
            else
                batch_offset_dq_acc = kargs.dq_acc_batch_offset_ptr[i_batch];

            if(kargs.cu_seqlen_q_ptr != nullptr)
                kargs.seqlen_q =
                    kargs.cu_seqlen_q_ptr[i_batch + 1] - kargs.cu_seqlen_q_ptr[i_batch];
            else if(kargs.seqlen_q_ptr != nullptr)
                // get real # queries & # keys under group mode
                kargs.seqlen_q = static_cast<ck_tile::index_t>(kargs.seqlen_q_ptr[i_batch]);
            else
                kargs.seqlen_q = physical_seqlen_q;

            if constexpr(kIsDeterministic)
            {
                const auto adjusted_seqstart_k_ptr = kargs.seqstart_k_ptr + i_batch;
                const ck_tile::index_t physical_seqlen_k =
                    adjusted_seqstart_k_ptr[1] - adjusted_seqstart_k_ptr[0];

                // Priority: cu_seqlen_k_ptr > seqlen_k_ptr > physical_seqlen_k
                if(kargs.cu_seqlen_k_ptr != nullptr)
                {
                    kargs.seqlen_k =
                        kargs.cu_seqlen_k_ptr[i_batch + 1] - kargs.cu_seqlen_k_ptr[i_batch];
                }
                else
                {
                    kargs.seqlen_k =
                        kargs.seqlen_k_ptr
                            ? static_cast<ck_tile::index_t>(kargs.seqlen_k_ptr[i_batch])
                            : physical_seqlen_k;
                }
            }
            // # of required blocks is different in each groups, terminate unnecessary blocks
            // earlier
            if(kargs.seqlen_q <= i_m0)
            {
                return;
            }
        }
        else
        {
            batch_offset_dq   = static_cast<long_index_t>(i_batch) * kargs.batch_stride_dq;
            physical_seqlen_q = kargs.seqlen_q;
            // batch mode: nsplits was pre-computed by PrepareWorkspaceHost and stored in workspace
            index_t nsplits = 1;
            if constexpr(kIsDeterministic)
                nsplits = kargs.nsplits_ptr[0];
            const long_index_t nhead_stride_dq_acc =
                static_cast<long_index_t>(nsplits) * kargs.seqlen_q * kargs.hdim_q;
            batch_offset_dq_acc =
                static_cast<long_index_t>(i_batch) * kargs.nhead_q * nhead_stride_dq_acc;
        }

        // for simplicity, batch stride we just modify the pointer
        QGradDataType* dq_ptr = reinterpret_cast<QGradDataType*>(kargs.dq_ptr) +
                                static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_dq +
                                batch_offset_dq;

        // dQAcc/dQ DRAM and DRAM window
        // compact layout: stride_dq_acc=hdim_q, split_stride=physical_seqlen_q*hdim_q,
        //                 nhead_stride=nsplits*physical_seqlen_q*hdim_q
        const long_index_t split_stride_dq_acc =
            static_cast<long_index_t>(physical_seqlen_q) * kargs.hdim_q;
        const index_t nsplits = [&, i_batch_ = i_batch]() {
            if constexpr(!kIsDeterministic)
                return 1;
            else if constexpr(!kIsGroupMode)
                return kargs.nsplits_ptr[0];
            else // deterministic group mode
                return kargs.nsplits_ptr[i_batch_];
        }();
        const long_index_t nhead_stride_dq_acc = split_stride_dq_acc * nsplits;

        const auto dq_acc_dram = [&, i_nhead_ = i_nhead]() {
            if constexpr(kIsDeterministic)
            {
                const AccDataType* dq_acc_ptr =
                    reinterpret_cast<const AccDataType*>(kargs.dq_acc_ptr) +
                    static_cast<long_index_t>(i_nhead_) * nhead_stride_dq_acc + batch_offset_dq_acc;

                auto dq_acc_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    dq_acc_ptr,
                    make_tuple(nsplits, kargs.seqlen_q, kargs.hdim_q),
                    make_tuple(split_stride_dq_acc, kargs.hdim_q, 1),
                    number<FmhaBwdConvertQGrad::kAlignmentQGradAcc>{},
                    number<1>{});
                return pad_tensor_view(dq_acc_dram_naive,
                                       make_tuple(number<1>{}, number<kM0>{}, number<kQKHeaddim>{}),
                                       sequence<false, kPadSeqLenQ, kPadHeadDimQ>{});
            }
            else
            {
                const AccDataType* dq_acc_ptr =
                    reinterpret_cast<const AccDataType*>(kargs.dq_acc_ptr) +
                    static_cast<long_index_t>(i_nhead_) * nhead_stride_dq_acc + batch_offset_dq_acc;

                auto dq_acc_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                    dq_acc_ptr,
                    make_tuple(kargs.seqlen_q, kargs.hdim_q),
                    make_tuple(kargs.hdim_q, 1),
                    number<FmhaBwdConvertQGrad::kAlignmentQGradAcc>{},
                    number<1>{});
                return pad_tensor_view(dq_acc_dram_naive,
                                       make_tuple(number<kM0>{}, number<kQKHeaddim>{}),
                                       sequence<kPadSeqLenQ, kPadHeadDimQ>{});
            }
        }();

        auto dq_dram = [&]() {
            auto dq_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                dq_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_q),
                make_tuple(kargs.stride_dq, 1),
                number<FmhaBwdConvertQGrad::kAlignmentQGrad>{},
                number<1>{});
            return pad_tensor_view(dq_dram_naive,
                                   make_tuple(number<kM0>{}, number<kQKHeaddim>{}),
                                   sequence<kPadSeqLenQ, kPadHeadDimQ>{});
        }();

        auto dq_acc_dram_window = [&]() {
            if constexpr(kIsDeterministic)
            {
                return make_tile_window(
                    dq_acc_dram,
                    make_tuple(number<1>{}, number<kM0>{}, number<kQKHeaddim>{}),
                    {0, i_m0, 0});
            }
            else
            {
                return make_tile_window(
                    dq_acc_dram, make_tuple(number<kM0>{}, number<kQKHeaddim>{}), {i_m0, 0});
            }
        }();

        auto dq_dram_window =
            make_tile_window(dq_dram, make_tuple(number<kM0>{}, number<kQKHeaddim>{}), {i_m0, 0});

        if constexpr(kIsDeterministic)
        {
            FmhaBwdConvertQGrad{}(dq_acc_dram_window, dq_dram_window, nsplits);
        }
        else
        {
            FmhaBwdConvertQGrad{}(dq_acc_dram_window, dq_dram_window);
        }
    }
};

} // namespace ck_tile
