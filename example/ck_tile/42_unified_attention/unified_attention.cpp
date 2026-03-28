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

// Dispatch macros for three tile tiers.
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

enum class tile_tier { large, medium, small };

static tile_tier select_tile_tier(const unified_attention_args& args)
{
    const index_t avg_q = args.num_seqs > 0 ? args.num_tokens / args.num_seqs : args.num_tokens;
    const index_t kBlockQ_small = 64 / args.num_queries_per_kv;  // kBlockQ for 2-warp kernel

    if(avg_q <= kBlockQ_small)
        return tile_tier::small;   // pure decode: 2 warps, kBlockM=64

    const index_t kBlockQ_medium = 128 / args.num_queries_per_kv; // kBlockQ for 4-warp kernel
    if(avg_q <= kBlockQ_medium * 2)
        return tile_tier::medium;  // many short seqs: 4 warps, kBlockM=128

    return tile_tier::large;       // long prefill: 8 warps, kBlockM=256
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
        if(tier == tile_tier::small)
        {
            // Small decode: 2 warps, kBlockM=64 (kBlockQ=8)
            if(args.data_type == unified_attention_args::data_type_enum::fp16)
            {
                if(!is_mask) DISPATCH_UNIFIED_ATTENTION_DECODE_SMALL(unified_attention_args::data_type_enum::fp16, false, 64, 64, 8)
                else         DISPATCH_UNIFIED_ATTENTION_DECODE_SMALL(unified_attention_args::data_type_enum::fp16, true,  64, 64, 8)
            }
            else if(args.data_type == unified_attention_args::data_type_enum::bf16)
            {
                if(!is_mask) DISPATCH_UNIFIED_ATTENTION_DECODE_SMALL(unified_attention_args::data_type_enum::bf16, false, 64, 64, 8)
                else         DISPATCH_UNIFIED_ATTENTION_DECODE_SMALL(unified_attention_args::data_type_enum::bf16, true,  64, 64, 8)
            }
        }
        else if(tier == tile_tier::medium)
        {
            // Medium: 4 warps, kBlockM=128 (kBlockQ=16)
            if(args.data_type == unified_attention_args::data_type_enum::fp16)
            {
                if(!is_mask) DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM(unified_attention_args::data_type_enum::fp16, false, 64, 128, 8)
                else         DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM(unified_attention_args::data_type_enum::fp16, true,  64, 128, 8)
            }
            else if(args.data_type == unified_attention_args::data_type_enum::bf16)
            {
                if(!is_mask) DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM(unified_attention_args::data_type_enum::bf16, false, 64, 128, 8)
                else         DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM(unified_attention_args::data_type_enum::bf16, true,  64, 128, 8)
            }
        }
        else
        {
            // Large prefill: 8 warps, kBlockM=256 (kBlockQ=32)
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

#undef DISPATCH_UNIFIED_ATTENTION_DECODE_SMALL
#undef DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM
#undef DISPATCH_UNIFIED_ATTENTION

} // namespace ck_tile
