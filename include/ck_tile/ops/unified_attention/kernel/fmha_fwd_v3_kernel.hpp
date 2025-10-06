// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/fmha/block/block_masking.hpp"

#include <type_traits>
#include <utility>

namespace ck_tile {

template <typename FmhaPipeline_, typename EpiloguePipeline_>
struct FmhaFwdV3Kernel
{
    using FmhaPipeline                            = ck_tile::remove_cvref_t<FmhaPipeline_>;
    using EpiloguePipeline                        = ck_tile::remove_cvref_t<EpiloguePipeline_>;
    static constexpr ck_tile::index_t kBlockSize  = FmhaPipeline::kBlockSize;
    static constexpr ck_tile::index_t kBlockPerCu = FmhaPipeline::kBlockPerCu;
    static_assert(kBlockPerCu > 0);

    using QDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::QDataType>;
    using KDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::KDataType>;
    using VDataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::VDataType>;
    using ODataType    = ck_tile::remove_cvref_t<typename FmhaPipeline::ODataType>;
    using SaccDataType = ck_tile::remove_cvref_t<typename FmhaPipeline::SaccDataType>;

    static constexpr bool kIsGroupMode = FmhaPipeline::kIsGroupMode;
    static constexpr bool kPadSeqLenQ  = FmhaPipeline::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK  = FmhaPipeline::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ = FmhaPipeline::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV = FmhaPipeline::kPadHeadDimV;

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

        ck_tile::index_t hdim_q;
        ck_tile::index_t hdim_v;

        ck_tile::index_t num_head_q;
        // for MQA/GQA, nhead could be different. This parameter is nhead_q / nhead_k
        // if this param is larger than 1, indicate MQA/GQA case
        ck_tile::index_t num_queries_per_kv;
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
        ck_tile::index_t HEAD_SIZE_PADDED;
    };


    struct UnifiedAttentionVarlenKargs
    {
        const int32_t* block_tables_ptr;
        const int32_t* seq_lens_ptr; // seq len in each batch
        const int32_t* query_start_len_ptr; // [num_seqs+1]

        ck_tile::index_t num_seqs; // number of batches for q
        ck_tile::index_t BLOCK_SIZE; // Block size for kv cache. to 2's exponent????
        ck_tile::index_t BLOCK_Q; // Block size for kv cache. to 2's exponent????
    };

    struct Kargs {
        UnifiedAttentionCommonKargs unifiedAttentionCommonKargs;
        UnifiedAttentionVarlenKargs unifiedAttentionVarlenKargs;
    };

    // using Kargs = FmhaFwdGroupModeKargs;

    CK_TILE_HOST static constexpr Kargs MakeKargs(
              const void* q_ptr,
              const void* k_ptr,
              const void* v_ptr,
              void* o_ptr,
              ck_tile::index_t hdim_q,
              ck_tile::index_t hdim_v,
              ck_tile::index_t num_head_q,
              ck_tile::index_t num_queries_per_kv,
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
                const int32_t* seq_lens_ptr,
                const int32_t* query_start_len_ptr,
                ck_tile::index_t num_seqs,
                ck_tile::index_t BLOCK_SIZE,
                ck_tile::index_t BLOCK_Q
        )
    {
        Kargs kargs{{q_ptr,
                     k_ptr,
                     v_ptr,
                     o_ptr,
                     hdim_q,
                     hdim_v,
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
                    {
                        block_tables_ptr,
                        seq_lens_ptr,
                        query_start_len_ptr,
                        num_seqs,
                        BLOCK_SIZE,
                        BLOCK_Q,
                    }};

        return kargs;
    }

    CK_TILE_HOST static constexpr auto GridSize2D(ck_tile::index_t num_kv_heads,
                                                ck_tile::index_t total_num_q_blocks)
    {
        return dim3(num_kv_heads * total_num_q_blocks, 0, 0);
    }

    // CK_TILE_HOST static constexpr auto GridSize3D(ck_tile::index_t num_kv_heads,
    //                                             ck_tile::index_t total_num_q_blocks)
    // {
    //     // TODO: fix 3D grid
    //     return dim2(num_kv_heads, total_num_q_blocks);
    // }

    // Binary search to find the sequence index for a given target index
    CK_TILE_DEVICE static constexpr ck_tile::index_t
    find_seq_idx(const int32_t* query_start_len_ptr,
                 ck_tile::index_t target_idx,
                 ck_tile::index_t num_seqs,
                 ck_tile::index_t BLOCK_Q,
                 bool use_q_block_mode)
    {
        ck_tile::index_t left = 0;
        ck_tile::index_t right = num_seqs;
        
        while (left < right)
        {
            ck_tile::index_t mid = (left + right) / 2;
            ck_tile::index_t val = query_start_len_ptr[mid];
            ck_tile::index_t mid_val = use_q_block_mode ? (val / BLOCK_Q + mid) : val;
            
            if (mid_val <= target_idx)
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
    
    CK_TILE_DEVICE static constexpr auto
    RemapTileIndices(const ck_tile::index_t pid, const Kargs& kargs)
    {
        using namespace ck_tile;
        
        constexpr index_t NUM_XCDS = 8;
        const index_t GRID_MN = kargs.unifiedAttentionCommonKargs.total_num_q_blocks * 
                            (kargs.unifiedAttentionCommonKargs.num_head_q);
        
        // Number of pids per XCD in the new arrangement
        const index_t pids_per_xcd = (GRID_MN + NUM_XCDS - 1) / NUM_XCDS;
        
        // When GRID_MN cannot divide NUM_XCDS, some xcds will have
        // pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
        // We calculate the number of xcds that have pids_per_xcd pids as tall_xcds
        index_t tall_xcds = GRID_MN % NUM_XCDS;
        tall_xcds = tall_xcds == 0 ? NUM_XCDS : tall_xcds;
        
        // Compute current XCD and local pid within the XCD
        const index_t xcd = pid % NUM_XCDS;
        const index_t local_pid = pid / NUM_XCDS;
        
        // Calculate new pid based on the new grouping
        index_t remapped_pid = 0; // Initialize to avoid constexpr error
        if(xcd < tall_xcds)
        {
            remapped_pid = xcd * pids_per_xcd + local_pid;
        }
        else
        {
            remapped_pid = tall_xcds * pids_per_xcd + 
                        (xcd - tall_xcds) * (pids_per_xcd - 1) + 
                        local_pid;
        }
        
        return remapped_pid;
    }

    CK_TILE_DEVICE static constexpr auto GetTileIndex(const ck_tile::index_t pid, const Kargs& kargs)
    {
        using namespace ck_tile;

        ck_tile::index_t total_num_q_blocks = kargs.unifiedAttentionCommonKargs.total_num_q_blocks;
        // const index_t num_tile_n1 = ck_tile::integer_divide_ceil(kargs.hdim_v,
        // FmhaPipeline::kN1);

        const index_t i_tile_m = pid % total_num_q_blocks;  // Query block index
        const index_t i_tile_n = pid / total_num_q_blocks;  // Head index

        return ck_tile::make_tuple(i_tile_m, i_tile_n);
    }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return ck_tile::max(FmhaPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }



    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        using namespace ck_tile;

        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        ck_tile::index_t pid = blockIdx.x;

        pid = RemapTileIndices(pid, kargs);

        // divide problem
        const auto [kv_head_idx, q_block_global_idx] = GetTileIndex(pid, kargs);

        const index_t seq_idx = find_seq_idx(
            kargs.unifiedAttentionVarlenKargs.query_start_len_ptr, q_block_global_idx, kargs.unifiedAttentionVarlenKargs.num_seqs, kargs.unifiedAttentionCommonKargs.BLOCK_Q, true
        ); // which batch

        const index_t q_block_start_idx = amd_wave_read_first_lane(kargs.unifiedAttentionVarlenKargs.query_start_len_ptr[seq_idx]);

        const index_t q_block_local_idx = amd_wave_read_first_lane(q_block_global_idx - q_block_start_idx);

        const index_t cur_batch_in_all_start_index = amd_wave_read_first_lane(kargs.unifiedAttentionVarlenKargs.query_start_len_ptr[seq_idx]);
        const index_t cur_batch_in_all_stop_index = amd_wave_read_first_lane(kargs.unifiedAttentionVarlenKargs.query_start_len_ptr[seq_idx + 1]);

        const index_t cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index;

        // TODO check if we get the block size info from pipeline
        if (q_block_local_idx * kargs.unifiedAttentionVarlenKargs.BLOCK_Q >= cur_batch_query_len) {
            return;
        }

        const index_t query_pos = q_block_local_idx * kargs.unifiedAttentionVarlenKargs.BLOCK_Q;


        // for simplicity, batch stride we just modify the pointer
        const QDataType* q_ptr = reinterpret_cast<const QDataType*>(kargs.unifiedAttentionCommonKargs.q_ptr) +
                                 static_cast<long_index_t>(kv_head_idx) * kargs.unifiedAttentionCommonKargs.num_queries_per_kv * kargs.unifiedAttentionCommonKargs.query_stride_1 +
                                 static_cast<long_index_t>(cur_batch_in_all_start_index) * kargs.unifiedAttentionCommonKargs.query_stride_0;
        // const KDataType* k_ptr =
        //     reinterpret_cast<const KDataType*>(kargs.k_ptr) +
        //     static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_k +
        //     batch_offset_k;
        // const VDataType* v_ptr =
        //     reinterpret_cast<const VDataType*>(kargs.v_ptr) +
        //     static_cast<long_index_t>(i_nhead / kargs.nhead_ratio_qk) * kargs.nhead_stride_v +
        //     batch_offset_v;
        ODataType* o_ptr = reinterpret_cast<ODataType*>(kargs.o_ptr) +
                           static_cast<long_index_t>(i_nhead) * kargs.nhead_stride_o +
                           batch_offset_o;

        // Q/K/V DRAM and DRAM window
        const auto q_dram = [&]() {
            const auto q_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                q_ptr,
                make_tuple(kargs.seqlen_q, kargs.unifiedAttentionVarlenKargs.),
                make_tuple(kargs.stride_q, 1),
                number<FmhaPipeline::kAlignmentQ>{},
                number<1>{});

            return pad_tensor_view(
                q_dram_naive,
                make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kSubQKHeaddim>{}),
                sequence<kPadSeqLenQ, kPadHeadDimQ>{});
        }();
        const auto k_dram = [&]() {
            const auto k_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                k_ptr,
                make_tuple(kargs.seqlen_k, kargs.hdim_q),
                make_tuple(kargs.stride_k, 1),
                number<FmhaPipeline::kAlignmentK>{},
                number<1>{});

            return pad_tensor_view(
                k_dram_naive,
                make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kK0>{}),
                sequence<kPadSeqLenK, kPadHeadDimQ>{});
        }();
        const auto v_dram = [&]() {
            const auto v_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                v_ptr,
                make_tuple(kargs.seqlen_k, kargs.hdim_v),
                make_tuple(kargs.stride_v, 1),
                number<FmhaPipeline::kAlignmentV>{},
                number<1>{});

            return pad_tensor_view(
                v_dram_naive,
                make_tuple(number<FmhaPipeline::kK1>{}, number<FmhaPipeline::kN1>{}),
                sequence<kPadSeqLenK, kPadHeadDimV>{});
        }();

        auto q_dram_window = make_tile_window(
            q_dram,
            make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kSubQKHeaddim>{}),
            {i_m0, 0});

        auto k_dram_window = make_tile_window(
            k_dram, make_tuple(number<FmhaPipeline::kN0>{}, number<FmhaPipeline::kK0>{}), {0, 0});

        auto v_dram_window =
            make_tile_window(v_dram,
                             make_tuple(number<FmhaPipeline::kK1>{}, number<FmhaPipeline::kN1>{}),
                             {0, i_n1});

        // lse
        auto lse_dram_window = [&, i_nhead_ = i_nhead]() {
            constexpr auto lse_dram_window_lengths = make_tuple(number<FmhaPipeline::kM0>{});
            if constexpr(kStoreLSE)
            {
                LSEDataType* lse_ptr =
                    reinterpret_cast<LSEDataType*>(kargs.lse_ptr) +
                    static_cast<long_index_t>(i_nhead_) * kargs.nhead_stride_lse + batch_offset_lse;

                const auto lse_dram = [&]() {
                    const auto lse_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                        lse_ptr,
                        make_tuple(kargs.seqlen_q),
                        make_tuple(1),
                        number<1>{},
                        number<1>{});

                    return pad_tensor_view(
                        lse_dram_naive, lse_dram_window_lengths, sequence<kPadSeqLenQ>{});
                }();

                return make_tile_window(lse_dram, lse_dram_window_lengths, {i_m0});
            }
            else
            {
                return make_null_tile_window(lse_dram_window_lengths);
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

        auto o_acc_tile = [&]() {
            return FmhaPipeline{}(q_dram_window,
                                  k_dram_window,
                                  v_dram_window,
                                  lse_dram_window,
                                  mask,
                                  kargs.scale_s,
                                  smem_ptr);
        }();

        // O DRAM and O DRAM window
        auto o_dram = [&]() {
            const auto o_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                o_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_v),
                make_tuple(kargs.stride_o, 1),
                number<FmhaPipeline::kAlignmentO>{},
                number<1>{});

            return pad_tensor_view(
                o_dram_naive,
                make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN1>{}),
                sequence<kPadSeqLenQ, kPadHeadDimV>{});
        }();

        auto o_dram_window =
            make_tile_window(o_dram,
                             make_tuple(number<FmhaPipeline::kM0>{}, number<FmhaPipeline::kN1>{}),
                             {i_m0, i_n1});

        EpiloguePipeline{}(o_dram_window, o_acc_tile, nullptr);
    }
};
} // namespace ck_tile
