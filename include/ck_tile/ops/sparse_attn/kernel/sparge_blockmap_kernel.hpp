// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"
#include <type_traits>

namespace ck_tile {

template <typename Pipeline_>
struct SpargeBlockMapKernel
{
    using Pipeline = remove_cvref_t<Pipeline_>;

    static constexpr index_t kBlockSize  = Pipeline::kBlockSize;
    static constexpr index_t kBlockPerCu = Pipeline::kBlockPerCu;

    using QDataType = typename Pipeline::QDataType;
    using KDataType = typename Pipeline::KDataType;

    static constexpr index_t kM0 = Pipeline::kM0;
    static constexpr index_t kN0 = Pipeline::kN0;
    static constexpr index_t D   = Pipeline::D;

    static constexpr index_t kAlignment = 16 / sizeof(QDataType);

    struct Kargs
    {
        const void* q_ptr;
        const void* k_ptr;

        index_t seqlen_q;
        index_t seqlen_k;
        index_t hdim_q;

        index_t nhead_q;
        index_t nhead_ratio_qk;

        index_t stride_q;
        index_t stride_k;
        index_t nhead_stride_q;
        index_t nhead_stride_k;
        index_t batch_stride_q;
        index_t batch_stride_k;

        float simthreshd1;
        float cdfthreshd;
        float topk;
        float scale;

        void* block_map_ptr;
        void* lut_ptr;
        void* valid_block_num_ptr;

        index_t N_k;
    };

    CK_TILE_HOST static constexpr auto MakeKargs(const void* q_ptr,
                                                 const void* k_ptr,
                                                 index_t seqlen_q,
                                                 index_t seqlen_k,
                                                 index_t hdim_q,
                                                 index_t nhead_q,
                                                 index_t nhead_ratio_qk,
                                                 index_t stride_q,
                                                 index_t stride_k,
                                                 index_t nhead_stride_q,
                                                 index_t nhead_stride_k,
                                                 index_t batch_stride_q,
                                                 index_t batch_stride_k,
                                                 float simthreshd1,
                                                 float cdfthreshd,
                                                 float topk,
                                                 float scale,
                                                 void* block_map_ptr,
                                                 void* lut_ptr,
                                                 void* valid_block_num_ptr)
    {
        const index_t N_k = integer_divide_ceil(seqlen_k, kN0);
        return Kargs{q_ptr,
                     k_ptr,
                     seqlen_q,
                     seqlen_k,
                     hdim_q,
                     nhead_q,
                     nhead_ratio_qk,
                     stride_q,
                     stride_k,
                     nhead_stride_q,
                     nhead_stride_k,
                     batch_stride_q,
                     batch_stride_k,
                     simthreshd1,
                     cdfthreshd,
                     topk,
                     scale,
                     block_map_ptr,
                     lut_ptr,
                     valid_block_num_ptr,
                     N_k};
    }

    CK_TILE_HOST static constexpr auto GridSize(index_t batch, index_t nhead_q, index_t seqlen_q)
    {
        const index_t Q_blk = integer_divide_ceil(seqlen_q, kM0);
        return dim3(Q_blk, nhead_q, batch);
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(kBlockSize); }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const index_t qb = static_cast<index_t>(blockIdx.x);
        const index_t hq = static_cast<index_t>(blockIdx.y);
        const index_t b  = static_cast<index_t>(blockIdx.z);

        const index_t hk = hq / kargs.nhead_ratio_qk;

        // Q pointer for this (batch, head, q_block)
        const auto* q_base = reinterpret_cast<const QDataType*>(kargs.q_ptr) +
                             b * kargs.batch_stride_q + hq * kargs.nhead_stride_q +
                             qb * kM0 * kargs.stride_q;

        // K pointer for this (batch, head_k)
        const auto* k_base = reinterpret_cast<const KDataType*>(kargs.k_ptr) +
                             b * kargs.batch_stride_k + hk * kargs.nhead_stride_k;

        // Q DRAM view with OOB padding
        const auto q_dram_naive = make_naive_tensor_view<address_space_enum::global>(
            q_base,
            make_tuple(kargs.seqlen_q - qb * kM0, D),
            make_tuple(kargs.stride_q, 1),
            number<kAlignment>{},
            number<1>{});
        const auto q_dram = pad_tensor_view(
            q_dram_naive, make_tuple(number<kM0>{}, number<D>{}), sequence<true, false>{});

        auto q_window = make_tile_window(q_dram,
                                         make_tuple(number<kM0>{}, number<D>{}),
                                         {0, 0},
                                         Pipeline::MakeQBlockDistribution());

        // K DRAM view with OOB padding
        const auto k_dram_naive =
            make_naive_tensor_view<address_space_enum::global>(k_base,
                                                               make_tuple(kargs.seqlen_k, D),
                                                               make_tuple(kargs.stride_k, 1),
                                                               number<kAlignment>{},
                                                               number<1>{});
        const auto k_dram = pad_tensor_view(
            k_dram_naive, make_tuple(number<kN0>{}, number<D>{}), sequence<true, false>{});

        auto k_window = make_tile_window(k_dram,
                                         make_tuple(number<kN0>{}, number<D>{}),
                                         {0, 0},
                                         Pipeline::MakeKBlockDistribution());

        // Output pointers for this (batch, head, q_block)
        const index_t N_k = kargs.N_k;
        const index_t bmap_offset =
            (b * kargs.nhead_q + hq) * integer_divide_ceil(kargs.seqlen_q, kM0) * N_k + qb * N_k;
        auto* bmap_ptr = reinterpret_cast<uint8_t*>(kargs.block_map_ptr) + bmap_offset;

        int32_t* lut_out   = nullptr;
        int32_t* valid_out = nullptr;
        if(kargs.lut_ptr != nullptr)
        {
            lut_out = reinterpret_cast<int32_t*>(kargs.lut_ptr) + bmap_offset;
            const index_t valid_offset =
                (b * kargs.nhead_q + hq) * integer_divide_ceil(kargs.seqlen_q, kM0) + qb;
            valid_out = reinterpret_cast<int32_t*>(kargs.valid_block_num_ptr) + valid_offset;
        }

        // Shared memory
        __shared__ char smem[Pipeline::GetSmemSize()];

        Pipeline{}(q_window,
                   k_window,
                   kargs.seqlen_q,
                   kargs.seqlen_k,
                   qb,
                   N_k,
                   kargs.nhead_ratio_qk,
                   kargs.simthreshd1,
                   kargs.cdfthreshd,
                   kargs.topk,
                   kargs.scale,
                   bmap_ptr,
                   lut_out,
                   valid_out,
                   static_cast<void*>(smem));
    }
};

} // namespace ck_tile
