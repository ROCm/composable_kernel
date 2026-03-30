// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "unified_attention.hpp"
#include "unified_attention_impl.hpp"
#include "mask.hpp"

namespace ck_tile {

std::ostream& operator<<(std::ostream& stream,
                         const unified_attention_args::data_type_enum& data_type)
{
    switch(data_type)
    {
    case unified_attention_args::data_type_enum::fp16: return stream << "fp16";
    case unified_attention_args::data_type_enum::bf16: return stream << "bf16";
    default: return stream << "unknown";
    }
}

// Helper macro to reduce dispatch boilerplate.
// Dispatches based on DataType, IsMasking, HeadSize, BlockM, NumQPerKV.
#define DISPATCH_UNIFIED_ATTENTION(DType, IsMask, HSize, BM, NQPKV) \
    { \
        using kernel_traits = unified_attention_kernel_traits<DType, IsMask, HSize, BM, NQPKV>; \
        return unified_attention_kernel_dispatch<kernel_traits>(args, config); \
    }

// Dispatch macros for three tile tiers (default block_size).
#define DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM(DType, IsMask, HSize, BM, NQPKV) \
    { \
        using kernel_traits = unified_attention_decode_kernel_traits<DType, IsMask, HSize, BM, NQPKV>; \
        return unified_attention_kernel_dispatch<kernel_traits>(args, config); \
    }

#define DISPATCH_UNIFIED_ATTENTION_DECODE_SMALL(DType, IsMask, HSize, BM, NQPKV) \
    { \
        using kernel_traits = unified_attention_decode_small_kernel_traits<DType, IsMask, HSize, BM, NQPKV>; \
        return unified_attention_kernel_dispatch_decode<kernel_traits>(args, config); \
    }

#define DISPATCH_UNIFIED_ATTENTION_DECODE_TINY(DType, IsMask, HSize, BM, NQPKV) \
    { \
        using kernel_traits = unified_attention_decode_tiny_kernel_traits<DType, IsMask, HSize, BM, NQPKV>; \
        return unified_attention_kernel_dispatch_decode<kernel_traits>(args, config); \
    }

// block_size=32 dispatch macros (6th template arg = 32).
#define DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM_BS32(DType, IsMask, HSize, BM, NQPKV) \
    { \
        using kernel_traits = unified_attention_decode_kernel_traits<DType, IsMask, HSize, BM, NQPKV, 32>; \
        return unified_attention_kernel_dispatch<kernel_traits>(args, config); \
    }

#define DISPATCH_UNIFIED_ATTENTION_DECODE_SMALL_BS32(DType, IsMask, HSize, BM, NQPKV) \
    { \
        using kernel_traits = unified_attention_decode_small_kernel_traits<DType, IsMask, HSize, BM, NQPKV, 32>; \
        return unified_attention_kernel_dispatch_decode<kernel_traits>(args, config); \
    }

#define DISPATCH_UNIFIED_ATTENTION_DECODE_TINY_BS32(DType, IsMask, HSize, BM, NQPKV) \
    { \
        using kernel_traits = unified_attention_decode_tiny_kernel_traits<DType, IsMask, HSize, BM, NQPKV, 32>; \
        return unified_attention_kernel_dispatch_decode<kernel_traits>(args, config); \
    }

enum class tile_tier { large, medium, small, tiny };

static tile_tier select_tile_tier(const unified_attention_args& args)
{
    const index_t avg_q = args.num_seqs > 0 ? args.num_tokens / args.num_seqs : args.num_tokens;
    const index_t kBlockQ_tiny = 16 / args.num_queries_per_kv;  // kBlockQ for 1-warp 16x16 kernel

    if(avg_q <= kBlockQ_tiny)
        return tile_tier::tiny;    // pure decode: 1 warp, 16x16 MFMA, kBlockM=16

    const index_t kBlockQ_small = 64 / args.num_queries_per_kv;  // kBlockQ for 2-warp kernel
    if(avg_q <= kBlockQ_small)
        return tile_tier::small;   // short decode: 2 warps, kBlockM=64

    // 4-warp serial pipeline outperforms 8-warp interleaved on all prefill shapes
    // (verified by exhaustive sweep over 363 shapes from production trace).
    return tile_tier::medium;      // all prefill: 4 warps, kBlockM=128
}

std::pair<bool, float> unified_attention(const unified_attention_args& args,
                                         const stream_config& config)
{
    const bool is_mask = (args.mask_type != static_cast<int>(mask_enum::no_mask));
    const auto tier = select_tile_tier(args);

    // d128, MHA (num_queries_per_kv == 1)
    if(args.hdim == 128 && args.num_queries_per_kv == 1)
    {
        if(args.data_type == unified_attention_args::data_type_enum::fp16)
        {
            if(!is_mask) DISPATCH_UNIFIED_ATTENTION(unified_attention_args::data_type_enum::fp16, false, 128, 256, 1)
            else         DISPATCH_UNIFIED_ATTENTION(unified_attention_args::data_type_enum::fp16, true,  128, 256, 1)
        }
        else if(args.data_type == unified_attention_args::data_type_enum::bf16)
        {
            if(!is_mask) DISPATCH_UNIFIED_ATTENTION(unified_attention_args::data_type_enum::bf16, false, 128, 256, 1)
            else         DISPATCH_UNIFIED_ATTENTION(unified_attention_args::data_type_enum::bf16, true,  128, 256, 1)
        }
    }

    // d64, GQA-8 (num_queries_per_kv == 8)
    if(args.hdim == 64 && args.num_queries_per_kv == 8)
    {
        const bool use_bs32 = (args.page_blk_size < 64);

        if(tier == tile_tier::tiny)
        {
            if(args.data_type == unified_attention_args::data_type_enum::fp16)
            {
                if(use_bs32) {
                    if(!is_mask) DISPATCH_UNIFIED_ATTENTION_DECODE_TINY_BS32(unified_attention_args::data_type_enum::fp16, false, 64, 16, 8)
                    else         DISPATCH_UNIFIED_ATTENTION_DECODE_TINY_BS32(unified_attention_args::data_type_enum::fp16, true,  64, 16, 8)
                } else {
                    if(!is_mask) DISPATCH_UNIFIED_ATTENTION_DECODE_TINY(unified_attention_args::data_type_enum::fp16, false, 64, 16, 8)
                    else         DISPATCH_UNIFIED_ATTENTION_DECODE_TINY(unified_attention_args::data_type_enum::fp16, true,  64, 16, 8)
                }
            }
            else if(args.data_type == unified_attention_args::data_type_enum::bf16)
            {
                if(use_bs32) {
                    if(!is_mask) DISPATCH_UNIFIED_ATTENTION_DECODE_TINY_BS32(unified_attention_args::data_type_enum::bf16, false, 64, 16, 8)
                    else         DISPATCH_UNIFIED_ATTENTION_DECODE_TINY_BS32(unified_attention_args::data_type_enum::bf16, true,  64, 16, 8)
                } else {
                    if(!is_mask) DISPATCH_UNIFIED_ATTENTION_DECODE_TINY(unified_attention_args::data_type_enum::bf16, false, 64, 16, 8)
                    else         DISPATCH_UNIFIED_ATTENTION_DECODE_TINY(unified_attention_args::data_type_enum::bf16, true,  64, 16, 8)
                }
            }
        }
        else if(tier == tile_tier::small)
        {
            if(args.data_type == unified_attention_args::data_type_enum::fp16)
            {
                if(use_bs32) {
                    if(!is_mask) DISPATCH_UNIFIED_ATTENTION_DECODE_SMALL_BS32(unified_attention_args::data_type_enum::fp16, false, 64, 64, 8)
                    else         DISPATCH_UNIFIED_ATTENTION_DECODE_SMALL_BS32(unified_attention_args::data_type_enum::fp16, true,  64, 64, 8)
                } else {
                    if(!is_mask) DISPATCH_UNIFIED_ATTENTION_DECODE_SMALL(unified_attention_args::data_type_enum::fp16, false, 64, 64, 8)
                    else         DISPATCH_UNIFIED_ATTENTION_DECODE_SMALL(unified_attention_args::data_type_enum::fp16, true,  64, 64, 8)
                }
            }
            else if(args.data_type == unified_attention_args::data_type_enum::bf16)
            {
                if(use_bs32) {
                    if(!is_mask) DISPATCH_UNIFIED_ATTENTION_DECODE_SMALL_BS32(unified_attention_args::data_type_enum::bf16, false, 64, 64, 8)
                    else         DISPATCH_UNIFIED_ATTENTION_DECODE_SMALL_BS32(unified_attention_args::data_type_enum::bf16, true,  64, 64, 8)
                } else {
                    if(!is_mask) DISPATCH_UNIFIED_ATTENTION_DECODE_SMALL(unified_attention_args::data_type_enum::bf16, false, 64, 64, 8)
                    else         DISPATCH_UNIFIED_ATTENTION_DECODE_SMALL(unified_attention_args::data_type_enum::bf16, true,  64, 64, 8)
                }
            }
        }
        else if(tier == tile_tier::medium)
        {
            if(args.data_type == unified_attention_args::data_type_enum::fp16)
            {
                if(use_bs32) {
                    if(!is_mask) DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM_BS32(unified_attention_args::data_type_enum::fp16, false, 64, 128, 8)
                    else         DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM_BS32(unified_attention_args::data_type_enum::fp16, true,  64, 128, 8)
                } else {
                    if(!is_mask) DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM(unified_attention_args::data_type_enum::fp16, false, 64, 128, 8)
                    else         DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM(unified_attention_args::data_type_enum::fp16, true,  64, 128, 8)
                }
            }
            else if(args.data_type == unified_attention_args::data_type_enum::bf16)
            {
                if(use_bs32) {
                    if(!is_mask) DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM_BS32(unified_attention_args::data_type_enum::bf16, false, 64, 128, 8)
                    else         DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM_BS32(unified_attention_args::data_type_enum::bf16, true,  64, 128, 8)
                } else {
                    if(!is_mask) DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM(unified_attention_args::data_type_enum::bf16, false, 64, 128, 8)
                    else         DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM(unified_attention_args::data_type_enum::bf16, true,  64, 128, 8)
                }
            }
        }
        else
        {
            // Large prefill: 8 warps, kBlockM=256 (kBlockQ=32)
            // No bs32 variant -- NumIssues < 1 for 8-warp tier with block_size=32.
            if(args.data_type == unified_attention_args::data_type_enum::fp16)
            {
                if(!is_mask) DISPATCH_UNIFIED_ATTENTION(unified_attention_args::data_type_enum::fp16, false, 64, 256, 8)
                else         DISPATCH_UNIFIED_ATTENTION(unified_attention_args::data_type_enum::fp16, true,  64, 256, 8)
            }
            else if(args.data_type == unified_attention_args::data_type_enum::bf16)
            {
                if(!is_mask) DISPATCH_UNIFIED_ATTENTION(unified_attention_args::data_type_enum::bf16, false, 64, 256, 8)
                else         DISPATCH_UNIFIED_ATTENTION(unified_attention_args::data_type_enum::bf16, true,  64, 256, 8)
            }
        }
    }

    std::cerr << "unified_attention: no matching kernel instance for hdim=" << args.hdim
              << " num_queries_per_kv=" << args.num_queries_per_kv
              << " data_type=" << args.data_type << " mask_type=" << args.mask_type << std::endl;
    return std::make_pair(false, -1.f);
}

#undef DISPATCH_UNIFIED_ATTENTION_DECODE_TINY_BS32
#undef DISPATCH_UNIFIED_ATTENTION_DECODE_SMALL_BS32
#undef DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM_BS32
#undef DISPATCH_UNIFIED_ATTENTION_DECODE_TINY
#undef DISPATCH_UNIFIED_ATTENTION_DECODE_SMALL
#undef DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM
#undef DISPATCH_UNIFIED_ATTENTION

} // namespace ck_tile
