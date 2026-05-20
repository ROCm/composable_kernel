// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/reduce.hpp"

namespace ck_tile {

template <typename Problem_>
struct SpargeBlockMapPipeline
{
    using Problem        = remove_cvref_t<Problem_>;
    using QDataType      = remove_cvref_t<typename Problem::QDataType>;
    using KDataType      = remove_cvref_t<typename Problem::KDataType>;
    using BlockFmhaShape = remove_cvref_t<typename Problem::BlockFmhaShape>;

    static constexpr index_t kBlockSize = Problem::kBlockSize;
    static constexpr index_t kM0        = BlockFmhaShape::kM0;
    static constexpr index_t kN0        = BlockFmhaShape::kN0;
    static constexpr index_t D          = BlockFmhaShape::kQKHeaddim;
    static constexpr index_t NumWarps   = BlockFmhaShape::NumWarps;
    static constexpr index_t WarpSize   = get_warp_size();

    static constexpr index_t KPerThread       = 16 / sizeof(QDataType);
    static constexpr index_t KThreads         = D / KPerThread;
    static constexpr index_t SeqThreadPerWarp = WarpSize / KThreads;
    static constexpr index_t MPerThread       = kM0 / (SeqThreadPerWarp * NumWarps);
    static constexpr index_t NPerThread       = kN0 / (SeqThreadPerWarp * NumWarps);

    static constexpr index_t kBlockPerCu = 1;
    static constexpr index_t kMaxKBlocks = 1024;

    // LDS layout (non-overlapping, all used simultaneously in K-block loop):
    //   [0 .. kReduceBytes)                       cross-warp reduction scratch slab 0
    //   [kReduceBytes .. 2*kReduceBytes)          cross-warp reduction scratch slab 1
    //                                             (ping-pong for K-loop double buffer)
    //   [kScoreOffset ..)                         scores[N_k]
    //   [kBmapOffset  ..)                         block_map[N_k]
    //   [kSmallOffset ..)                         softmax/selection argmax scratch (2*NumWarps
    //   floats)
    // Column-stride pad: k_idx*(KPerThread+1) instead of k_idx*KPerThread to break
    // the 4-way intra-warp bank conflict. Per-warp slab size: KThreads * (KPerThread + 1) floats.
    static constexpr index_t kColPaddedStride  = KPerThread + 1;
    static constexpr index_t kPerWarpFloats    = KThreads * kColPaddedStride;
    static constexpr index_t kReduceBytes      = NumWarps * kPerWarpFloats * sizeof(float);
    static constexpr index_t kReduceTotalBytes = 2 * kReduceBytes; // 2 slabs (K-loop ping-pong)
    static constexpr index_t kScoreOffset      = kReduceTotalBytes;
    static constexpr index_t kBmapOffset       = kScoreOffset + kMaxKBlocks * sizeof(float);
    static constexpr index_t kSmallOffset      = kBmapOffset + kMaxKBlocks * sizeof(uint8_t);

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return kSmallOffset + 2 * NumWarps * sizeof(float);
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeQBlockDistribution()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<MPerThread, NumWarps, SeqThreadPerWarp>,
                                             sequence<KThreads, KPerThread>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeKBlockDistribution()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<NPerThread, NumWarps, SeqThreadPerWarp>,
                                             sequence<KThreads, KPerThread>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    // Extract tile data into a local float array via static_for (compile-time indices).
    template <index_t BufSize, typename Tile>
    CK_TILE_DEVICE static void tile_to_float(const Tile& tile, float (&out)[BufSize])
    {
        static_assert(Tile::get_thread_buffer_size() == BufSize);
        const auto& buf = tile.get_thread_buffer();
        static_for<0, BufSize, 1>{}([&](auto i) { out[i.value] = type_convert<float>(buf[i]); });
    }

    // Column-wise (dim=0) sum: accumulate SeqPerThread rows into KPerThread partial sums,
    // then xor-shuffle across m_idx within warp.
    template <index_t SeqPerThread>
    CK_TILE_DEVICE static void column_reduce_thread_and_warp(const float* __restrict__ data,
                                                             float (&col_acc)[KPerThread])
    {
        for(index_t k = 0; k < KPerThread; ++k)
            col_acc[k] = 0.f;

        for(index_t m = 0; m < SeqPerThread; ++m)
            for(index_t k = 0; k < KPerThread; ++k)
                col_acc[k] += data[m * KPerThread + k];

        for(index_t stride = KThreads; stride < WarpSize; stride *= 2)
            for(index_t k = 0; k < KPerThread; ++k)
                col_acc[k] += warp_shuffle(col_acc[k], __lane_id() ^ stride);
    }

    // Cross-warp LDS reduction for column sums.
    // Templated TrailingSync flag: when false, the trailing __syncthreads() is dropped —
    // only safe when the next access targets a *different* slab and the intervening work
    // does not read smem_reduce. Used at the slab_b call in the K-loop, where the next
    // iter's first cross-warp reduce writes to slab_a and is preceded by its own leading sync.
    template <bool TrailingSync = true>
    CK_TILE_DEVICE static void column_reduce_cross_warp(float (&col_acc)[KPerThread],
                                                        float* __restrict__ smem_reduce)
    {
        const index_t tid     = static_cast<index_t>(threadIdx.x);
        const index_t warp_id = tid / WarpSize;
        const index_t lane_id = tid % WarpSize;
        const index_t k_idx   = lane_id % KThreads;
        const index_t m_idx   = lane_id / KThreads;

        // Column-stride pad: stride k_idx by (KPerThread+1)=9 instead of 8, changing
        // per-lane bank from (k_idx*8+k)%32 to (k_idx*9+k)%32. For k=0, lanes
        // (k_idx={0,4,8,12}) hit banks {0,4,8,12} instead of all 0.
        if(m_idx == 0)
            for(index_t k = 0; k < KPerThread; ++k)
                smem_reduce[warp_id * kPerWarpFloats + k_idx * kColPaddedStride + k] = col_acc[k];
        __syncthreads();

        for(index_t k = 0; k < KPerThread; ++k)
            col_acc[k] = 0.f;
        for(index_t w = 0; w < NumWarps; ++w)
            for(index_t k = 0; k < KPerThread; ++k)
                col_acc[k] += smem_reduce[w * kPerWarpFloats + k_idx * kColPaddedStride + k];
        if constexpr(TrailingSync)
            __syncthreads();
    }

    // Compute ||v||^2 per row: sum along KPerThread then xor-shuffle across k_idx.
    template <index_t SeqPerThread>
    CK_TILE_DEVICE static void row_reduce_sq_norm(const float* __restrict__ data,
                                                  float (&row_norms)[SeqPerThread],
                                                  index_t actual_seq)
    {
        const index_t tid     = static_cast<index_t>(threadIdx.x);
        const index_t warp_id = tid / WarpSize;
        const index_t m_idx   = (tid % WarpSize) / KThreads;

        for(index_t m = 0; m < SeqPerThread; ++m)
        {
            float sq = 0.f;
            for(index_t k = 0; k < KPerThread; ++k)
            {
                float v = data[m * KPerThread + k];
                sq += v * v;
            }
            for(index_t stride = 1; stride < KThreads; stride *= 2)
                sq += warp_shuffle(sq, __lane_id() ^ stride);

            index_t gsq  = m * (SeqThreadPerWarp * NumWarps) + warp_id * SeqThreadPerWarp + m_idx;
            row_norms[m] = (gsq < actual_seq) ? sq : 0.f;
        }
    }

    // Column reduce of normalised rows: sum_hat[d] = sum_i data[i,d] / ||data[i,:]||.
    template <index_t SeqPerThread>
    CK_TILE_DEVICE static void column_reduce_normalised(const float* __restrict__ data,
                                                        const float* __restrict__ row_norms,
                                                        float (&col_acc)[KPerThread],
                                                        index_t actual_seq)
    {
        const index_t tid     = static_cast<index_t>(threadIdx.x);
        const index_t warp_id = tid / WarpSize;
        const index_t m_idx   = (tid % WarpSize) / KThreads;

        for(index_t k = 0; k < KPerThread; ++k)
            col_acc[k] = 0.f;

        for(index_t m = 0; m < SeqPerThread; ++m)
        {
            // Round 12: hardware fast rsqrt (v_rsq_f32, ~1 ULP) replaces sw sqrt+rcp.
            float inv_norm = (row_norms[m] > 0.f) ? rsqrtf(row_norms[m]) : 0.f;
            index_t gsq    = m * (SeqThreadPerWarp * NumWarps) + warp_id * SeqThreadPerWarp + m_idx;
            if(gsq < actual_seq)
                for(index_t k = 0; k < KPerThread; ++k)
                    col_acc[k] += data[m * KPerThread + k] * inv_norm;
        }

        for(index_t stride = KThreads; stride < WarpSize; stride *= 2)
            for(index_t k = 0; k < KPerThread; ++k)
                col_acc[k] += warp_shuffle(col_acc[k], __lane_id() ^ stride);
    }

    // Scalar reduce across k_idx lanes (within warp).
    CK_TILE_DEVICE static float reduce_across_k(float v)
    {
        for(index_t stride = 1; stride < KThreads; stride *= 2)
            v += warp_shuffle(v, __lane_id() ^ stride);
        return v;
    }

    // Full-block scalar reduce (warp xor + cross-warp LDS).
    CK_TILE_DEVICE static float block_reduce_sum(float v, float* smem_small)
    {
        const index_t tid     = static_cast<index_t>(threadIdx.x);
        const index_t warp_id = tid / WarpSize;
        const index_t lane_id = tid % WarpSize;

        for(index_t stride = 1; stride < WarpSize; stride *= 2)
            v += warp_shuffle(v, __lane_id() ^ stride);
        if(lane_id == 0)
            smem_small[warp_id] = v;
        __syncthreads();
        if(tid == 0)
        {
            float s = 0.f;
            for(index_t w = 0; w < NumWarps; ++w)
                s += smem_small[w];
            smem_small[0] = s;
        }
        __syncthreads();
        return smem_small[0];
    }

    CK_TILE_DEVICE static float block_reduce_max(float v, float* smem_small)
    {
        const index_t tid     = static_cast<index_t>(threadIdx.x);
        const index_t warp_id = tid / WarpSize;
        const index_t lane_id = tid % WarpSize;

        for(index_t stride = 1; stride < WarpSize; stride *= 2)
            v = max(v, warp_shuffle(v, __lane_id() ^ stride));
        if(lane_id == 0)
            smem_small[warp_id] = v;
        __syncthreads();
        if(tid == 0)
        {
            float s = smem_small[0];
            for(index_t w = 1; w < NumWarps; ++w)
                s = max(s, smem_small[w]);
            smem_small[0] = s;
        }
        __syncthreads();
        return smem_small[0];
    }

    // ======================================================================
    template <typename QWindowType, typename KWindowType>
    CK_TILE_DEVICE void operator()(const QWindowType& q_window_in,
                                   const KWindowType& /*k_window_in*/,
                                   index_t seqlen_q,
                                   index_t /*seqlen_k*/,
                                   index_t qb,
                                   index_t N_k,
                                   index_t /*nhead_ratio_qk*/,
                                   float simthreshd1,
                                   float cdfthreshd,
                                   float topk,
                                   float scale,
                                   uint8_t* block_map_ptr,
                                   int32_t* lut_ptr,
                                   int32_t* valid_block_num_ptr,
                                   const KDataType* __restrict__ pooled_k_ws_ptr,
                                   const uint8_t* __restrict__ sim_k_ws_ptr,
                                   void* smem_ptr,
                                   index_t mask_type,
                                   bool attention_sink) const
    {
        const index_t tid = static_cast<index_t>(threadIdx.x);

        // mask_enum::mask_top_left == 1 (01_fmha/mask.hpp:16). Multiplicative
        // form handles BLKQ=64,BLKK=128 (kM0<kN0) and the kM0>=kN0 case.
        const bool is_causal_tl = (mask_type == 1);

        // K-loop no longer reduces; only Q-stats uses smem_float0.
        // smem_float1 slab is allocated for layout compat but unused.
        auto* smem_float0 = reinterpret_cast<float*>(smem_ptr);
        auto* smem_scores =
            reinterpret_cast<float*>(reinterpret_cast<char*>(smem_ptr) + kScoreOffset);
        auto* smem_bmap =
            reinterpret_cast<uint8_t*>(reinterpret_cast<char*>(smem_ptr) + kBmapOffset);
        auto* smem_small =
            reinterpret_cast<float*>(reinterpret_cast<char*>(smem_ptr) + kSmallOffset);

        const index_t bs_q   = min(static_cast<index_t>(kM0), seqlen_q - qb * kM0);
        const float inv_bs_q = (bs_q > 0) ? (1.0f / static_cast<float>(bs_q)) : 0.f;

        // ==================================================================
        // Q Block Statistics
        // ==================================================================
        auto q_tile = load_tile(q_window_in);

        float q_data[MPerThread * KPerThread];
        tile_to_float<MPerThread * KPerThread>(q_tile, q_data);

        // 1a. L2 norm per token
        float psq[MPerThread];
        row_reduce_sq_norm<MPerThread>(q_data, psq, bs_q);

        // 1b. Column sum -> mean
        // Drop trailing sync: next reduce reuses same slab (smem_float0) with its own
        // leading __syncthreads() before reading. pooled_q_mean is register-only between reduces.
        float pooled_q_mean[KPerThread];
        column_reduce_thread_and_warp<MPerThread>(q_data, pooled_q_mean);
        column_reduce_cross_warp<false>(pooled_q_mean, smem_float0);
        for(index_t k = 0; k < KPerThread; ++k)
            pooled_q_mean[k] *= inv_bs_q;

        // 1c. Normalised sum_hat
        // Drop trailing sync: next cross-warp reduce in K-loop iter 0 writes
        // slab_a=smem_float0 (kb=0 even); its leading __syncthreads() covers the WAR.
        // sum_hat is register-only here.
        float sum_hat[KPerThread];
        column_reduce_normalised<MPerThread>(q_data, psq, sum_hat, bs_q);
        column_reduce_cross_warp<false>(sum_hat, smem_float0);

        // 1d. sim_q = ||sum_hat||^2 / bs_q^2
        float sh_sq = 0.f;
        for(index_t k = 0; k < KPerThread; ++k)
            sh_sq += sum_hat[k] * sum_hat[k];
        sh_sq               = reduce_across_k(sh_sq);
        const float denom_q = static_cast<float>(bs_q) * static_cast<float>(bs_q);
        const bool sim_q    = (denom_q > 0.f) && ((sh_sq / denom_q) > simthreshd1);

        // Not similar → force all K blocks ON, early exit
        if(!sim_q)
        {
            // R32 Item 2: only fill causal-valid prefix when active.
            const index_t causal_kb_end =
                is_causal_tl ? min(N_k, integer_divide_ceil((qb + 1) * kM0, kN0)) : N_k;

            for(index_t i = tid; i < N_k; i += kBlockSize)
                block_map_ptr[i] = (i < causal_kb_end) ? 1 : 0;

            // R32 Item 3: sink force. Under top-left causal, kb=0 always
            // causal-valid for qb>=0 -> no-op; meaningful for mask=no + sink=1.
            if(attention_sink && tid == 0)
                block_map_ptr[0] = 1;
            __syncthreads(); // sink visible to LUT-build below

            if(lut_ptr != nullptr && tid == 0)
            {
                int32_t valid = 0, prev = 0;
                for(index_t kb = 0; kb < causal_kb_end; ++kb)
                {
                    lut_ptr[valid] = static_cast<int32_t>(kb) - prev;
                    prev           = static_cast<int32_t>(kb);
                    ++valid;
                }
                for(index_t i = valid; i < N_k; ++i)
                    lut_ptr[i] = 0;
                *valid_block_num_ptr = valid;
            }
            return;
        }

        // ==================================================================
        // K Block Loop
        // ==================================================================
        for(index_t i = tid; i < N_k; i += kBlockSize)
            smem_bmap[i] = 0;
        __syncthreads();

        // K-stats precomputed by SpargeKStatsKernel. Each thread loads its own
        // KPerThread-slice of pooled_k_mean from DRAM workspace; sim_k is a single byte.
        // No K-tile load, no cross-warp reduce in the K-loop.
        const index_t lane_id_kb = tid % WarpSize;
        const index_t k_idx_kb   = lane_id_kb % KThreads;

        for(index_t kb = 0; kb < N_k; ++kb)
        {
            // R32 Item 2: top-left causal at block grain.
            // (qb,kb) past-diagonal iff kb*kN0 >= (qb+1)*kM0.
            const bool causal_killed = is_causal_tl && (kb * kN0 >= (qb + 1) * kM0);

            const KDataType* p_kb = pooled_k_ws_ptr + kb * D + k_idx_kb * KPerThread;
            float pooled_k_mean[KPerThread];
            for(index_t k = 0; k < KPerThread; ++k)
                pooled_k_mean[k] = type_convert<float>(p_kb[k]);

            float dot = 0.f;
            for(index_t k = 0; k < KPerThread; ++k)
                dot += pooled_q_mean[k] * pooled_k_mean[k];
            dot = reduce_across_k(dot);

            const bool sim_k = (sim_k_ws_ptr[kb] != 0);

            if(tid == 0)
            {
                // INVARIANT (mirrors SpargeAttn ref utils.py:175-180):
                //   ~sim_k blocks are forced ON in the bitmap (final_map[~sim_k]=1)
                //   AND have score = -inf so the selection step (topk / cdf) does NOT
                //   pick them again (would double-count toward topk budget).
                // R32: causal_killed gates the force-on so past-diagonal blocks are
                // NOT forced ON; bmap stays 0, scores -inf so selection excludes them.
                if(causal_killed)
                    smem_scores[kb] = -numeric<float>::infinity(); // bmap stays 0
                else if(!sim_k)
                {
                    smem_bmap[kb]   = 1;
                    smem_scores[kb] = -numeric<float>::infinity();
                }
                else
                    smem_scores[kb] = dot * scale;
            }
        }
        __syncthreads(); // guard selection's reads of smem_bmap / smem_scores

        // ==================================================================
        // Softmax + Selection
        // ==================================================================

        // max
        float lmax = -numeric<float>::infinity();
        for(index_t i = tid; i < N_k; i += kBlockSize)
            lmax = max(lmax, smem_scores[i]);
        const float max_score = block_reduce_max(lmax, smem_small);

        // exp + sum
        float lsum = 0.f;
        for(index_t i = tid; i < N_k; i += kBlockSize)
        {
            float e        = (smem_scores[i] > -numeric<float>::infinity())
                                 ? __builtin_expf(smem_scores[i] - max_score)
                                 : 0.f;
            smem_scores[i] = e;
            lsum += e;
        }
        const float sum_exp = block_reduce_sum(lsum, smem_small);

        // Round 13i: argmax is invariant under positive scaling (inv_sum > 0). When
        // topk > 0 we never read normalised values for cdfthreshd, so skip the
        // normalise pass entirely (saves N_k LDS writes + 1 __syncthreads). The
        // cdfthreshd path (topk <= 0) still requires normalised scores so the
        // accumulator `cumulative_prob` matches probabilities.
        const bool topk_active = (topk > 0.f);
        const float inv_sum    = (!topk_active && sum_exp > 0.f) ? (1.0f / sum_exp) : 0.f;
        if(!topk_active)
        {
            for(index_t i = tid; i < N_k; i += kBlockSize)
                smem_scores[i] *= inv_sum;
            __syncthreads();
        }

        // Selection: iterative argmax
        index_t num_to_select =
            topk_active
                ? max(static_cast<index_t>(1), static_cast<index_t>(topk * static_cast<float>(N_k)))
                : N_k;

        float cumulative_prob = 0.f;
        for(index_t round = 0; round < num_to_select; ++round)
        {
            // thread-local argmax
            float best_val   = -1.f;
            index_t best_idx = 0;
            for(index_t i = tid; i < N_k; i += kBlockSize)
            {
                if(smem_scores[i] > best_val || (smem_scores[i] == best_val && i < best_idx))
                {
                    best_val = smem_scores[i];
                    best_idx = i;
                }
            }

            // warp argmax
            for(index_t stride = 1; stride < WarpSize; stride *= 2)
            {
                float rv   = warp_shuffle(best_val, __lane_id() ^ stride);
                index_t ri = warp_shuffle(best_idx, __lane_id() ^ stride);
                if(rv > best_val || (rv == best_val && ri < best_idx))
                {
                    best_val = rv;
                    best_idx = ri;
                }
            }

            // cross-warp argmax via LDS
            const index_t lane_id = tid % WarpSize;
            const index_t warp_id = tid / WarpSize;
            if(lane_id == 0)
            {
                smem_small[warp_id]            = best_val;
                smem_small[NumWarps + warp_id] = bit_cast<float>(static_cast<int32_t>(best_idx));
            }
            __syncthreads();

            // Round 13g: collapse 2 syncs/round into 1. tid==0 computes the global
            // winner AND writes the sentinel (smem_bmap=1, smem_scores=-1) in the same
            // critical section, gated by bv>0. All threads then read smem_small[0] for
            // the early break / cumulative_prob accumulation. Saves 1 __syncthreads per
            // round (~32 syncs @ N_k=64 topk=0.5).
            if(tid == 0)
            {
                float bv   = smem_small[0];
                index_t bi = bit_cast<int32_t>(smem_small[NumWarps]);
                for(index_t w = 1; w < NumWarps; ++w)
                {
                    float wv   = smem_small[w];
                    index_t wi = bit_cast<int32_t>(smem_small[NumWarps + w]);
                    if(wv > bv || (wv == bv && wi < bi))
                    {
                        bv = wv;
                        bi = wi;
                    }
                }
                // Write sentinel into bmap/scores in the same critical section.
                // Guarded by bv > 0 so we never poison a valid score with -1.
                if(bv > 0.f)
                {
                    smem_bmap[bi]   = 1;
                    smem_scores[bi] = -1.f;
                }
                smem_small[0] = bv;
            }
            __syncthreads();

            float g_val = smem_small[0];

            if(g_val <= 0.f)
                break;

            if(topk > 0.f)
            {
                if(round + 1 >= num_to_select)
                    break;
            }
            else
            {
                cumulative_prob += g_val;
                if(cumulative_prob >= cdfthreshd)
                    break;
            }
        }

        // ==================================================================
        // Write outputs to global memory
        // ==================================================================
        // R32 Item 3: force smem_bmap[0]=1 BEFORE LUT collation reads it.
        // Reuses existing LUT-build loop (R31 §4: don't manually insert into
        // delta stream). Causal post-multiply unnecessary: D.2 sets killed
        // scores to -inf; selection gate L490 `bv > 0` excludes them, so
        // smem_bmap[bi]=1 never fires for killed blocks.
        if(attention_sink && tid == 0)
            smem_bmap[0] = 1;
        __syncthreads();

        for(index_t i = tid; i < N_k; i += kBlockSize)
            block_map_ptr[i] = smem_bmap[i];

        if(lut_ptr != nullptr && tid == 0)
        {
            int32_t valid = 0, prev = 0;
            for(index_t kb = 0; kb < N_k; ++kb)
            {
                if(smem_bmap[kb] != 0)
                {
                    lut_ptr[valid] = static_cast<int32_t>(kb) - prev;
                    prev           = static_cast<int32_t>(kb);
                    ++valid;
                }
            }
            for(index_t i = valid; i < N_k; ++i)
                lut_ptr[i] = 0;
            *valid_block_num_ptr = valid;
        }
    }
};

} // namespace ck_tile
