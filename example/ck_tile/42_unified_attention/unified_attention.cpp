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

std::pair<bool, float> unified_attention(const unified_attention_args& args,
                                         const stream_config& config)
{
    const bool is_mask = (args.mask_type != static_cast<int>(mask_enum::no_mask));

    // Route based on (data_type, mask, hdim, num_queries_per_kv).
    // Only d128 MHA (8 warps, kBlockM=256) instances available.
    // Decode-tuned instances require pipeline changes (NumWarpGroups must == 2,
    // which means exactly 8 warps; fewer warps are not supported).
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

    std::cerr << "unified_attention: no matching kernel instance for hdim=" << args.hdim
              << " num_queries_per_kv=" << args.num_queries_per_kv
              << " data_type=" << args.data_type << " mask_type=" << args.mask_type << std::endl;
    return std::make_pair(false, -1.f);
}

#undef DISPATCH_UNIFIED_ATTENTION

} // namespace ck_tile
