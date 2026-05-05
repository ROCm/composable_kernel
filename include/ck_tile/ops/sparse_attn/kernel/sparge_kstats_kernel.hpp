// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"
#include <type_traits>

namespace ck_tile {

// Kernel A wrapper: grid (N_k, nhead_k, batch). Each work-group precomputes
// K-block stats (pooled_k_mean[D], sim_k) for one (b, hk, kb) into a workspace
// that Kernel B (block_map) reads instead of recomputing per Q-block.
template <typename Pipeline_>
struct SpargeKStatsKernel
{
    using Pipeline = remove_cvref_t<Pipeline_>;

    static constexpr index_t kBlockSize  = Pipeline::kBlockSize;
    static constexpr index_t kBlockPerCu = Pipeline::kBlockPerCu;

    using QDataType = typename Pipeline::QDataType;
    using KDataType = typename Pipeline::KDataType;

    static constexpr index_t kN0 = Pipeline::kN0;
    static constexpr index_t D   = Pipeline::D;

    static constexpr index_t kAlignment = 16 / sizeof(KDataType);

    struct Kargs
    {
        const void* k_ptr;

        index_t seqlen_k;
        index_t hdim_q;
        index_t nhead_k;

        index_t stride_k;
        index_t nhead_stride_k;
        index_t batch_stride_k;

        float simthreshd1;

        void* pooled_k_ptr; // [batch, nhead_k, N_k, D] fp32
        void* sim_k_ptr;    // [batch, nhead_k, N_k] uint8

        index_t N_k;

        // R21A Phase 4 + R21B fix: optional per-head simthreshd1.
        // Buffer is sized [nhead_q] floats to match SpargeAttn upstream contract
        // (utils.py:324, Headnum=q.size(1)). Kernel only indexes the first
        // nhead_k entries via [hk]. nullptr => use scalar `simthreshd1`.
        const float* simthreshd1_per_head;
    };

    CK_TILE_HOST static constexpr auto MakeKargs(const void* k_ptr,
                                                 index_t seqlen_k,
                                                 index_t hdim_q,
                                                 index_t nhead_k,
                                                 index_t stride_k,
                                                 index_t nhead_stride_k,
                                                 index_t batch_stride_k,
                                                 float simthreshd1,
                                                 void* pooled_k_ptr,
                                                 void* sim_k_ptr,
                                                 const float* simthreshd1_per_head = nullptr)
    {
        const index_t N_k = integer_divide_ceil(seqlen_k, kN0);
        return Kargs{k_ptr,
                     seqlen_k,
                     hdim_q,
                     nhead_k,
                     stride_k,
                     nhead_stride_k,
                     batch_stride_k,
                     simthreshd1,
                     pooled_k_ptr,
                     sim_k_ptr,
                     N_k,
                     simthreshd1_per_head};
    }

    CK_TILE_HOST static constexpr auto GridSize(index_t batch, index_t nhead_k, index_t seqlen_k)
    {
        const index_t N_k = integer_divide_ceil(seqlen_k, kN0);
        return dim3(N_k, nhead_k, batch);
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(kBlockSize); }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const index_t kb = static_cast<index_t>(blockIdx.x);
        const index_t hk = static_cast<index_t>(blockIdx.y);
        const index_t b  = static_cast<index_t>(blockIdx.z);

        const auto* k_base = reinterpret_cast<const KDataType*>(kargs.k_ptr) +
                             b * kargs.batch_stride_k + hk * kargs.nhead_stride_k +
                             kb * kN0 * kargs.stride_k;

        const auto k_dram_naive = make_naive_tensor_view<address_space_enum::global>(
            k_base,
            make_tuple(kargs.seqlen_k - kb * kN0, D),
            make_tuple(kargs.stride_k, 1),
            number<kAlignment>{},
            number<1>{});
        const auto k_dram = pad_tensor_view(
            k_dram_naive, make_tuple(number<kN0>{}, number<D>{}), sequence<true, false>{});

        auto k_window = make_tile_window(k_dram,
                                         make_tuple(number<kN0>{}, number<D>{}),
                                         {0, 0},
                                         Pipeline::MakeKBlockDistribution());

        const index_t N_k         = kargs.N_k;
        const index_t khead_off   = (b * kargs.nhead_k + hk) * N_k;
        auto* pooled_k_out = reinterpret_cast<float*>(kargs.pooled_k_ptr) + (khead_off + kb) * D;
        auto* sim_k_out    = reinterpret_cast<uint8_t*>(kargs.sim_k_ptr) + (khead_off + kb);

        __shared__ char smem[Pipeline::GetSmemSize()];

        // R21A Phase 4: per-head simthreshd1 if provided, else scalar broadcast.
        const float simthreshd1_eff = (kargs.simthreshd1_per_head != nullptr)
                                          ? kargs.simthreshd1_per_head[hk]
                                          : kargs.simthreshd1;

        Pipeline{}(k_window,
                   kargs.seqlen_k,
                   kb,
                   simthreshd1_eff,
                   pooled_k_out,
                   sim_k_out,
                   static_cast<void*>(smem));
    }
};

} // namespace ck_tile
