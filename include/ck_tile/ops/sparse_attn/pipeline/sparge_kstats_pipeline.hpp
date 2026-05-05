// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/sparse_attn/pipeline/sparge_blockmap_pipeline.hpp"

namespace ck_tile {

// Kernel A of the K-stat precompute split: one work-group per (b, hk, kb)
// computes pooled_k_mean and sim_k for that K-block once. Kernel B then reads
// from the workspace instead of recomputing per Q-block.
template <typename Problem_>
struct SpargeKStatsPipeline
{
    using Problem   = remove_cvref_t<Problem_>;
    using Base      = SpargeBlockMapPipeline<Problem>;
    using QDataType = typename Base::QDataType;
    using KDataType = typename Base::KDataType;

    static constexpr index_t kBlockSize = Base::kBlockSize;
    static constexpr index_t kM0        = Base::kM0;
    static constexpr index_t kN0        = Base::kN0;
    static constexpr index_t D          = Base::D;
    static constexpr index_t NumWarps   = Base::NumWarps;
    static constexpr index_t WarpSize   = Base::WarpSize;

    static constexpr index_t KPerThread       = Base::KPerThread;
    static constexpr index_t KThreads         = Base::KThreads;
    static constexpr index_t SeqThreadPerWarp = Base::SeqThreadPerWarp;
    static constexpr index_t NPerThread       = Base::NPerThread;

    static constexpr index_t kBlockPerCu = 1;

    static constexpr index_t kColPaddedStride = Base::kColPaddedStride;
    static constexpr index_t kPerWarpFloats   = Base::kPerWarpFloats;
    static constexpr index_t kReduceBytes     = NumWarps * kPerWarpFloats * sizeof(float);

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return kReduceBytes; }

    CK_TILE_HOST_DEVICE static constexpr auto MakeKBlockDistribution()
    {
        return Base::MakeKBlockDistribution();
    }

    // operator(): one work-group, one K-block. Writes D fp32 + 1 uint8 to workspace.
    template <typename KWindowType>
    CK_TILE_DEVICE void operator()(const KWindowType& k_window,
                                   index_t seqlen_k,
                                   index_t kb,
                                   float simthreshd1,
                                   float* __restrict__ pooled_k_out, // D floats
                                   uint8_t* __restrict__ sim_k_out,  // 1 byte
                                   void* smem_ptr) const
    {
        const index_t tid = static_cast<index_t>(threadIdx.x);
        auto* smem_reduce = reinterpret_cast<float*>(smem_ptr);

        const index_t bs_k   = min(static_cast<index_t>(kN0), seqlen_k - kb * kN0);
        const float inv_bs_k = (bs_k > 0) ? (1.0f / static_cast<float>(bs_k)) : 0.f;

        auto k_tile = load_tile(k_window);

        float k_data[NPerThread * KPerThread];
        Base::template tile_to_float<NPerThread * KPerThread>(k_tile, k_data);

        const index_t warp_id = tid / WarpSize;
        const index_t lane_id = tid % WarpSize;
        const index_t k_idx   = lane_id % KThreads;
        const index_t m_idx   = lane_id / KThreads;

        // pooled_k_mean: column sum then cross-warp reduce.
        // R21A: drop trailing sync (next cross_warp_reduce has its own leading sync).
        float pooled_k_mean[KPerThread];
        Base::template column_reduce_thread_and_warp<NPerThread>(k_data, pooled_k_mean);
        Base::template column_reduce_cross_warp<false>(pooled_k_mean, smem_reduce);
        for(index_t k = 0; k < KPerThread; ++k)
            pooled_k_mean[k] *= inv_bs_k;

        // R21A: write pooled_k_mean to global early so its register liveness ends here,
        // freeing VGPR before k_sum_hat becomes live.
        if(warp_id == 0 && m_idx == 0)
        {
            for(index_t k = 0; k < KPerThread; ++k)
                pooled_k_out[k_idx * KPerThread + k] = pooled_k_mean[k];
        }

        // K row L2 norms + normalised column sum (k_sum_hat)
        float k_psq[NPerThread];
        Base::template row_reduce_sq_norm<NPerThread>(k_data, k_psq, bs_k);

        float k_sum_hat[KPerThread];
        Base::template column_reduce_normalised<NPerThread>(k_data, k_psq, k_sum_hat, bs_k);
        // R21A: drop trailing sync (no further smem read; only intra-warp shuffle + global write).
        Base::template column_reduce_cross_warp<false>(k_sum_hat, smem_reduce);

        // sim_k = (||k_sum_hat||^2 / bs_k^2) > simthreshd1
        float ksh_sq = 0.f;
        for(index_t k = 0; k < KPerThread; ++k)
            ksh_sq += k_sum_hat[k] * k_sum_hat[k];
        ksh_sq              = Base::reduce_across_k(ksh_sq);
        const float denom_k = static_cast<float>(bs_k) * static_cast<float>(bs_k);
        const bool sim_k    = (denom_k > 0.f) && ((ksh_sq / denom_k) > simthreshd1);

        if(tid == 0)
            *sim_k_out = sim_k ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
    }
};

} // namespace ck_tile
