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

    // LDS layout (non-overlapping, all used simultaneously in Phase 2):
    //   [0 .. kReduceBytes)         cross-warp reduction scratch
    //   [kScoreOffset ..)           scores[N_k]
    //   [kBmapOffset  ..)           block_map[N_k]
    //   [kSmallOffset ..)           Phase 3 argmax scratch (2*NumWarps floats)
    static constexpr index_t kReduceBytes = NumWarps * D * sizeof(float);
    static constexpr index_t kScoreOffset = kReduceBytes;
    static constexpr index_t kBmapOffset  = kScoreOffset + kMaxKBlocks * sizeof(float);
    static constexpr index_t kSmallOffset = kBmapOffset + kMaxKBlocks * sizeof(uint8_t);

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
    CK_TILE_DEVICE static void column_reduce_cross_warp(float (&col_acc)[KPerThread],
                                                        float* __restrict__ smem_reduce)
    {
        const index_t tid     = static_cast<index_t>(threadIdx.x);
        const index_t warp_id = tid / WarpSize;
        const index_t lane_id = tid % WarpSize;
        const index_t k_idx   = lane_id % KThreads;
        const index_t m_idx   = lane_id / KThreads;

        if(m_idx == 0)
            for(index_t k = 0; k < KPerThread; ++k)
                smem_reduce[warp_id * D + k_idx * KPerThread + k] = col_acc[k];
        __syncthreads();

        for(index_t k = 0; k < KPerThread; ++k)
            col_acc[k] = 0.f;
        for(index_t w = 0; w < NumWarps; ++w)
            for(index_t k = 0; k < KPerThread; ++k)
                col_acc[k] += smem_reduce[w * D + k_idx * KPerThread + k];
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
            float inv_norm = (row_norms[m] > 0.f) ? (1.0f / __builtin_sqrtf(row_norms[m])) : 0.f;
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
                                   const KWindowType& k_window_in,
                                   index_t seqlen_q,
                                   index_t seqlen_k,
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
                                   void* smem_ptr) const
    {
        const index_t tid = static_cast<index_t>(threadIdx.x);

        auto* smem_float = reinterpret_cast<float*>(smem_ptr);
        auto* smem_scores =
            reinterpret_cast<float*>(reinterpret_cast<char*>(smem_ptr) + kScoreOffset);
        auto* smem_bmap =
            reinterpret_cast<uint8_t*>(reinterpret_cast<char*>(smem_ptr) + kBmapOffset);
        auto* smem_small =
            reinterpret_cast<float*>(reinterpret_cast<char*>(smem_ptr) + kSmallOffset);

        const index_t bs_q   = min(static_cast<index_t>(kM0), seqlen_q - qb * kM0);
        const float inv_bs_q = (bs_q > 0) ? (1.0f / static_cast<float>(bs_q)) : 0.f;

        // ==================================================================
        // Phase 1: Q Block Statistics
        // ==================================================================
        auto q_tile = load_tile(q_window_in);

        float q_data[MPerThread * KPerThread];
        tile_to_float<MPerThread * KPerThread>(q_tile, q_data);

        // 1a. L2 norm per token
        float psq[MPerThread];
        row_reduce_sq_norm<MPerThread>(q_data, psq, bs_q);

        // 1b. Column sum -> mean
        float pooled_q_mean[KPerThread];
        column_reduce_thread_and_warp<MPerThread>(q_data, pooled_q_mean);
        column_reduce_cross_warp(pooled_q_mean, smem_float);
        for(index_t k = 0; k < KPerThread; ++k)
            pooled_q_mean[k] *= inv_bs_q;

        // 1c. Normalised sum_hat
        float sum_hat[KPerThread];
        column_reduce_normalised<MPerThread>(q_data, psq, sum_hat, bs_q);
        column_reduce_cross_warp(sum_hat, smem_float);

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
            for(index_t i = tid; i < N_k; i += kBlockSize)
                block_map_ptr[i] = 1;

            if(lut_ptr != nullptr && tid == 0)
            {
                int32_t valid = 0, prev = 0;
                for(index_t kb = 0; kb < N_k; ++kb)
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
        // Phase 2: K Block Loop
        // ==================================================================
        for(index_t i = tid; i < N_k; i += kBlockSize)
            smem_bmap[i] = 0;
        __syncthreads();

        auto k_window = k_window_in;

        for(index_t kb = 0; kb < N_k; ++kb)
        {
            const index_t bs_k   = min(static_cast<index_t>(kN0), seqlen_k - kb * kN0);
            const float inv_bs_k = (bs_k > 0) ? (1.0f / static_cast<float>(bs_k)) : 0.f;

            auto k_tile = load_tile(k_window);

            float k_data[NPerThread * KPerThread];
            tile_to_float<NPerThread * KPerThread>(k_tile, k_data);

            // K mean
            float pooled_k_mean[KPerThread];
            column_reduce_thread_and_warp<NPerThread>(k_data, pooled_k_mean);
            column_reduce_cross_warp(pooled_k_mean, smem_float);
            for(index_t k = 0; k < KPerThread; ++k)
                pooled_k_mean[k] *= inv_bs_k;

            // dot(pooled_q_mean, pooled_k_mean)
            float dot = 0.f;
            for(index_t k = 0; k < KPerThread; ++k)
                dot += pooled_q_mean[k] * pooled_k_mean[k];
            dot = reduce_across_k(dot);

            // K L2 norms + normalised sum_hat
            float k_psq[NPerThread];
            row_reduce_sq_norm<NPerThread>(k_data, k_psq, bs_k);

            float k_sum_hat[KPerThread];
            column_reduce_normalised<NPerThread>(k_data, k_psq, k_sum_hat, bs_k);
            column_reduce_cross_warp(k_sum_hat, smem_float);

            // sim_k
            float ksh_sq = 0.f;
            for(index_t k = 0; k < KPerThread; ++k)
                ksh_sq += k_sum_hat[k] * k_sum_hat[k];
            ksh_sq              = reduce_across_k(ksh_sq);
            const float denom_k = static_cast<float>(bs_k) * static_cast<float>(bs_k);
            const bool sim_k    = (denom_k > 0.f) && ((ksh_sq / denom_k) > simthreshd1);

            if(tid == 0)
            {
                if(!sim_k)
                {
                    smem_bmap[kb]   = 1;
                    smem_scores[kb] = -numeric<float>::infinity();
                }
                else
                {
                    smem_scores[kb] = dot * scale;
                }
            }
            __syncthreads();

            move_tile_window(k_window, {kN0, 0});
        }

        // ==================================================================
        // Phase 3: Softmax + Selection
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

        // normalise
        const float inv_sum = (sum_exp > 0.f) ? (1.0f / sum_exp) : 0.f;
        for(index_t i = tid; i < N_k; i += kBlockSize)
            smem_scores[i] *= inv_sum;
        __syncthreads();

        // Selection: iterative argmax
        index_t num_to_select =
            (topk > 0.f)
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
                smem_small[0] = bv;
                smem_small[1] = bit_cast<float>(static_cast<int32_t>(bi));
            }
            __syncthreads();

            float g_val   = smem_small[0];
            index_t g_idx = bit_cast<int32_t>(smem_small[1]);

            if(g_val <= 0.f)
                break;

            if(tid == 0)
            {
                smem_bmap[g_idx]   = 1;
                smem_scores[g_idx] = -1.f;
            }
            __syncthreads();

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
