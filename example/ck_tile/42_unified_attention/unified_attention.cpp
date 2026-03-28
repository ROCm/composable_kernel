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

// Helper macro for decode-tuned dispatch (4 warps, kBlockM=128).
#define DISPATCH_UNIFIED_ATTENTION_DECODE(DType, IsMask, HSize, BM, NQPKV) \
    { \
        using kernel_traits = unified_attention_decode_kernel_traits<DType, IsMask, HSize, BM, NQPKV>; \
        return unified_attention_kernel_dispatch<kernel_traits>(args, config); \
    }

static bool is_decode_shape(const unified_attention_args& args)
{
    const index_t kBlockQ_prefill = 256 / args.num_queries_per_kv;
    return args.num_tokens <= args.num_seqs * kBlockQ_prefill;
}

std::pair<bool, float> unified_attention(const unified_attention_args& args,
                                         const stream_config& config)
{
    const bool is_mask = (args.mask_type != static_cast<int>(mask_enum::no_mask));
    const bool use_decode = is_decode_shape(args);

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
        if(use_decode)
        {
            // Decode-tuned: 4 warps, kBlockM=128 (kBlockQ=16)
            if(args.data_type == unified_attention_args::data_type_enum::fp16)
            {
                if(!is_mask) DISPATCH_UNIFIED_ATTENTION_DECODE(unified_attention_args::data_type_enum::fp16, false, 64, 128, 8)
                else         DISPATCH_UNIFIED_ATTENTION_DECODE(unified_attention_args::data_type_enum::fp16, true,  64, 128, 8)
            }
            else if(args.data_type == unified_attention_args::data_type_enum::bf16)
            {
                if(!is_mask) DISPATCH_UNIFIED_ATTENTION_DECODE(unified_attention_args::data_type_enum::bf16, false, 64, 128, 8)
                else         DISPATCH_UNIFIED_ATTENTION_DECODE(unified_attention_args::data_type_enum::bf16, true,  64, 128, 8)
            }
        }
        else
        {
            // Prefill: 8 warps, kBlockM=256 (kBlockQ=32)
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

#undef DISPATCH_UNIFIED_ATTENTION_DECODE

#undef DISPATCH_UNIFIED_ATTENTION

} // namespace ck_tile
