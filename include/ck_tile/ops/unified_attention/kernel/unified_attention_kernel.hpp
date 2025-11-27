// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
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

    // TODO add yjese
    static constexpr index_t HEAD_SIZE        = UnifiedAttentionPipeline::HEAD_SIZE;
    static constexpr index_t HEAD_SIZE_PADDED = UnifiedAttentionPipeline::HEAD_SIZE_PADDED;

    // BLOCK_Q = BLOCK_M // num_queries_per_kv
    // BLOCK_Q is the block size for q seqlen
    /// static constexpr index_t BLOCK_Q = UnifiedAttentionPipeline::BLOCK_Q;
    static constexpr index_t BLOCK_M = UnifiedAttentionPipeline::BLOCK_M;
    static constexpr index_t BLOCK_Q = UnifiedAttentionPipeline::BLOCK_Q;
    // BLOCK size for K seqlen
    static constexpr index_t BLOCK_SIZE = UnifiedAttentionPipeline::BLOCK_SIZE;

    // kargs use aggregate initializer, so no constructor will provided
    // use inheritance to minimize karg size
    // user need to use MakeKargs() function to create kargs.
    // The attention is default causal
    struct UnifiedAttentionCommonKargs
    {
        const void* q_ptr;
        const void* k_ptr; // [num_blks, blk_size, num_kv_heads, head_size]
        const void* v_ptr; // [num_blks, blk_size, num_kv_heads, head_size]
        void* o_ptr;

        ck_tile::index_t num_blks;
        ck_tile::index_t num_head_q;
        // for MQA/GQA, nhead could be different. This parameter is nhead_q / nhead_k
        // if this param is larger than 1, indicate MQA/GQA case
        const ck_tile::index_t num_queries_per_kv;
        // scales
        float scale_s;
        float scale;
        float scale_k;
        float scale_v;
        float scale_out;

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
    };

    struct UnifiedAttentionVarlenKargs : UnifiedAttentionCommonKargs
    {
        const int32_t* block_tables_ptr;
        ck_tile::index_t block_table_stride;
        const int32_t* seq_lens_ptr;        // seq len in each batch
        const int32_t* query_start_len_ptr; // [num_seqs+1]

        ck_tile::index_t num_seqs; // number of batches for q
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
                                                  ck_tile::index_t num_seqs)
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     o_ptr,
                     num_blks,
                     num_head_q,
                     num_queries_per_kv,
                     static_cast<float>(scale_s * ck_tile::log2e_v<>),
                     scale,
                     scale_k,
                     scale_v,
                     scale_out,
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
                     output_stride_1},
                    block_tables_ptr,
                    block_table_stride,
                    seq_lens_ptr,
                    query_start_len_ptr,
                    num_seqs};

        return kargs;
    }

    CK_TILE_HOST static constexpr auto GridSize2D(ck_tile::index_t num_kv_heads,
                                                  ck_tile::index_t total_num_q_blocks)
    {
        return dim3(num_kv_heads * total_num_q_blocks);
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

    CK_TILE_DEVICE static constexpr auto RemapTileIndices(const ck_tile::index_t pid,
                                                          const Kargs& kargs)
    {
        using namespace ck_tile;

        constexpr index_t NUM_XCDS = 8;
        const index_t GRID_MN      = kargs.total_num_q_blocks * (kargs.num_head_q / kargs.num_queries_per_kv);

        // Number of pids per XCD in the new arrangement
        const index_t pids_per_xcd = (GRID_MN + NUM_XCDS - 1) / NUM_XCDS;

        // When GRID_MN cannot divide NUM_XCDS, some xcds will have
        // pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
        // We calculate the number of xcds that have pids_per_xcd pids as tall_xcds
        index_t tall_xcds = GRID_MN % NUM_XCDS;
        tall_xcds         = tall_xcds == 0 ? NUM_XCDS : tall_xcds;

        // Compute current XCD and local pid within the XCD
        const index_t xcd       = pid % NUM_XCDS;
        const index_t local_pid = pid / NUM_XCDS;

        // Calculate new pid based on the new grouping
        index_t remapped_pid = 0; // Initialize to avoid constexpr error
        if(xcd < tall_xcds)
        {
            remapped_pid = xcd * pids_per_xcd + local_pid;
        }
        else
        {
            remapped_pid =
                tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid;
        }

        return remapped_pid;
    }

    CK_TILE_DEVICE static constexpr auto GetTileIndex(const ck_tile::index_t pid,
                                                      const Kargs& kargs)
    {
        using namespace ck_tile;

        ck_tile::index_t total_num_q_blocks = kargs.total_num_q_blocks;
        // const index_t num_tile_n1 = ck_tile::integer_divide_ceil(kargs.hdim_v,
        // UnifiedAttentionPipeline::kN1);

        return ck_tile::make_tuple(pid / total_num_q_blocks, pid % total_num_q_blocks);
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return ck_tile::max(UnifiedAttentionPipeline::GetSmemSize(),
                            EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        using namespace ck_tile;

        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        ck_tile::index_t pid = blockIdx.x;

        const index_t num_queries_per_kv = kargs.num_queries_per_kv;

        assert(BLOCK_M / num_queries_per_kv == BLOCK_Q);

        // const index_t BLOCK_Q = BLOCK_M / num_queries_per_kv;
        // for simplicity, batch stride we just modify the pointer
        // const index_t num_head_q = kargs.num_head_q;

        // const index_t num_head_k = num_head_q / num_queries_per_kv;
        pid = RemapTileIndices(pid, kargs);

        // divide problem
        const auto [kv_head_idx, q_block_global_idx] = GetTileIndex(pid, kargs);

        // grid size is (num_kv_heads, total_num_q_blocks)
        // total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
        // q.shape[0] is total number of query tokens across all batches
        // one q_block spans BLOCK_Q = BLOCK_M // num_queries_per_kv number of query token groups.
        // One query token group shares one kv token

        const index_t seq_idx = find_seq_idx(kargs.query_start_len_ptr,
                                             q_block_global_idx,
                                             kargs.num_seqs,
                                             BLOCK_Q,
                                             true); // which batch

        const index_t q_block_start_idx = kargs.query_start_len_ptr[seq_idx] / BLOCK_Q + seq_idx;

        const index_t q_block_local_idx = amd_wave_read_first_lane(q_block_global_idx - q_block_start_idx);

        const index_t cur_batch_in_all_start_index = kargs.query_start_len_ptr[seq_idx];
        const index_t cur_batch_in_all_stop_index = kargs.query_start_len_ptr[seq_idx + 1];

        const index_t cur_batch_query_len =
             amd_wave_read_first_lane(cur_batch_in_all_stop_index - cur_batch_in_all_start_index);

        // TODO check if we get the block size info from pipeline
        if(q_block_local_idx * BLOCK_Q >= cur_batch_query_len)
        {
            return;
        }

        const index_t query_pos = amd_wave_read_first_lane(q_block_local_idx * BLOCK_Q);
        const index_t seq_len   = kargs.seq_lens_ptr[seq_idx];

        const index_t context_len =  amd_wave_read_first_lane(seq_len - cur_batch_query_len);

        index_t _max_seq_prefix_len =
             amd_wave_read_first_lane((context_len + q_block_local_idx * BLOCK_Q + (BLOCK_M - 1)
             + 1));

        if(seq_len < _max_seq_prefix_len)
        {
            _max_seq_prefix_len = seq_len;
        }

        const auto max_seq_prefix_len = _max_seq_prefix_len;
        const index_t num_blocks      =  amd_wave_read_first_lane((max_seq_prefix_len + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // TODO sliding window
        const index_t num_blocks_start = 0;
        index_t kv_head_offset         = kv_head_idx * kargs.stride_k_cache_2;

        // Q/K/V DRAM and DRAM window
        index_t q_ptr_offset_0 = cur_batch_in_all_start_index *
                                 kargs.query_stride_0; // move the pointer to the batch start
        index_t q_ptr_offset_1 =
            kv_head_idx * num_queries_per_kv *
            kargs.query_stride_1; // move the pointer to the correct head group start
        index_t q_ptr_offset = q_ptr_offset_0 + q_ptr_offset_1;

        index_t o_ptr_offset_0 = cur_batch_in_all_start_index *
                                 kargs.output_stride_0; // move the pointer to the batch start
        index_t o_ptr_offset_1 =
            kv_head_idx * num_queries_per_kv *
            kargs.output_stride_1; // move the pointer to the correct head group start
        index_t o_ptr_offset       = o_ptr_offset_0 + o_ptr_offset_1;
        index_t block_table_offset = seq_idx * kargs.block_table_stride;

        const QDataType* q_ptr = reinterpret_cast<const QDataType*>(kargs.q_ptr) + q_ptr_offset;
        const KDataType* k_ptr = reinterpret_cast<const KDataType*>(kargs.k_ptr) + kv_head_offset;
        const VDataType* v_ptr = reinterpret_cast<const VDataType*>(kargs.v_ptr) + kv_head_offset;
        ODataType* o_ptr       = reinterpret_cast<ODataType*>(kargs.o_ptr) + o_ptr_offset;

        index_t query_len_padded =  amd_wave_read_first_lane(integer_divide_ceil(cur_batch_query_len, BLOCK_Q) * BLOCK_Q);
        // const bool is_query_len_padded = (cur_batch_query_len % BLOCK_Q == 0);

        // Q/K/V DRAM and DRAM window
        const auto q_dram = [&]() {
            const auto q_dram_base = make_naive_tensor_view<address_space_enum::global>(
                q_ptr,
                make_tuple(cur_batch_query_len, num_queries_per_kv, HEAD_SIZE),
                make_tuple(kargs.query_stride_0, kargs.query_stride_1, 1),
                number<UnifiedAttentionPipeline::kAlignmentQ>{},
                number<1>{});

            const auto q_dram_pad =
                pad_tensor_view( // aling seqlen with BLOCK_Q and head dim with HEAD_SIZE_PADDED
                    q_dram_base,
                    // block sizes
                    make_tuple(number<BLOCK_Q>{}, 1, HEAD_SIZE_PADDED),
                    sequence<true, false, kPadHeadDimQ>{}); // pads to (seq_len_padded, num_head_q,
                                                            // HEAD_SIZE_PADDED)

            const auto q_dram_merged = transform_tensor_view(
                q_dram_pad,
                make_tuple(make_merge_transform(make_tuple(query_len_padded, num_queries_per_kv)),
                           make_pass_through_transform(HEAD_SIZE_PADDED)),
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
                             make_tuple(number<BLOCK_M>{}, number<HEAD_SIZE_PADDED>{}),
                             {query_pos * num_queries_per_kv, 0});

        const auto k_dram = [&]() {
            // HEAD dim is skipped as defined in the ptrs
            const auto k_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                k_ptr,
                make_tuple(kargs.num_blks * BLOCK_SIZE, HEAD_SIZE),
                make_tuple(kargs.stride_k_cache_1, kargs.stride_k_cache_3),
                number<UnifiedAttentionPipeline::kAlignmentK>{},
                number<1>{});

            const auto k_dram_pad = pad_tensor_view(k_dram_naive,
                                                    // TODO can the BLOCK_SIZE_RAW needs padding?
                                                    make_tuple(BLOCK_SIZE, HEAD_SIZE_PADDED),
                                                    sequence<false, kPadHeadDimQ>{});

            return k_dram_pad;
        }();

        auto k_dram_window = make_tile_window(
            k_dram, make_tuple(number<BLOCK_SIZE>{}, number<HEAD_SIZE_PADDED>{}), {0, 0});

        const auto v_dram = [&]() {
            const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                v_ptr,
                make_tuple(kargs.num_blks * BLOCK_SIZE, HEAD_SIZE),
                make_tuple(kargs.stride_v_cache_1, kargs.stride_v_cache_3),
                number<UnifiedAttentionPipeline::kAlignmentV>{},
                number<1>{});

            const auto v_dram_pad = pad_tensor_view(v_dram_naive,
                                                    make_tuple(BLOCK_SIZE, HEAD_SIZE_PADDED),
                                                    sequence<false, kPadHeadDimQ>{});

            return v_dram_pad;
        }();

        auto v_dram_window = make_tile_window(
            v_dram, make_tuple(number<BLOCK_SIZE>{}, number<HEAD_SIZE_PADDED>{}), {0, 0});

        FmhaMask mask = [&]() {
            if constexpr(kHasMask)
                return ck_tile::make_generic_attention_mask_from_lr_window<FmhaMask>(
                    -1, 
                    0,
                    cur_batch_query_len, // y_total
                    seq_len,           // x_total
                    num_queries_per_kv, // the same sequence index is repeated num_queries_per_kv
                                       // times along x dim of the tile
                    false
                );
            else
                return FmhaMask{cur_batch_query_len, seq_len};
        }();

        auto o_acc_tile = [&]() {
            return UnifiedAttentionPipeline{}(q_dram_window,
                                              k_dram_window,
                                              v_dram_window,
                                              num_blocks,
                                              num_blocks_start,
                                              kargs.block_tables_ptr,
                                              block_table_offset,
                                              mask,
                                              kargs.scale_s,
                                              smem_ptr);
        }();

        // O DRAM and O DRAM window
        auto o_dram = [&]() {
            const auto o_dram_base = make_naive_tensor_view<address_space_enum::global>(
                o_ptr,
                make_tuple(cur_batch_query_len, num_queries_per_kv, HEAD_SIZE),
                make_tuple(kargs.output_stride_0, kargs.output_stride_1, 1),
                number<UnifiedAttentionPipeline::kAlignmentO>{},
                number<1>{});

            const auto o_dram_pad =
                pad_tensor_view( // aling cu_seqlen with BLOCK_Q and head dim with HEAD_SIZE_PADDED
                    o_dram_base,
                    // block sizes
                    make_tuple(BLOCK_Q, 1, HEAD_SIZE_PADDED),
                    sequence<true, false, kPadHeadDimQ>{}); // pads to (seq_len_padded, num_head_q,
                                                            // HEAD_SIZE_PADDED)

            const auto o_dram_merged = transform_tensor_view(
                o_dram_pad,
                make_tuple(make_merge_transform(make_tuple(query_len_padded, num_queries_per_kv)),
                           make_pass_through_transform(HEAD_SIZE_PADDED)),
                make_tuple(sequence<0, 1>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return o_dram_merged;
        }();

        auto o_dram_window =
            make_tile_window(o_dram,
                             make_tuple(number<BLOCK_M>{}, number<HEAD_SIZE_PADDED>{}),
                             {query_pos * num_queries_per_kv, 0});

        EpiloguePipeline{}(o_dram_window, o_acc_tile, nullptr);
    }
};
} // namespace ck_tile
