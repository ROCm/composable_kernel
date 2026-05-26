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

// SWA-aware variant: requires explicit BlockSize (since IsLocal is the 7th template arg).
// HeadSize<=64 -> BlockSize=64; HeadSize=128 -> BlockSize=32. Caller must supply.
#define DISPATCH_UNIFIED_ATTENTION_LOCAL(DType, HSize, BM, NQPKV, BSize) \
    { \
        using kernel_traits = unified_attention_kernel_traits<DType, /*IsMasking=*/true, HSize, BM, NQPKV, BSize, /*IsLocal=*/true>; \
        return unified_attention_kernel_dispatch<kernel_traits>(args, config); \
    }

// SWA-aware decode dispatchers for bs32. These mirror the non-local *_BS32 macros
// but flip IsLocal=true on the 7th template arg, so the kernel uses the
// sliding-window mask AND the per-Q-tile KV-block iteration clipping.
#define DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM_BS32_LOCAL(DType, HSize, BM, NQPKV) \
    { \
        using kernel_traits = unified_attention_decode_kernel_traits<DType, /*IsMasking=*/true, HSize, BM, NQPKV, 32, /*IsLocal=*/true>; \
        return unified_attention_kernel_dispatch<kernel_traits>(args, config); \
    }

// Tiny decode with page_blk_size=32: 1 warp, 16x16 MFMA, kBlockM=16, kBlockQ=2.
#define DISPATCH_UNIFIED_ATTENTION_DECODE_TINY_BS32(DType, IsMask, HSize, BM, NQPKV) \
    { \
        using kernel_traits = unified_attention_decode_tiny_kernel_traits<DType, IsMask, HSize, BM, NQPKV, 32>; \
        return unified_attention_kernel_dispatch_decode<kernel_traits>(args, config); \
    }

#define DISPATCH_UNIFIED_ATTENTION_DECODE_TINY_BS32_LOCAL(DType, HSize, BM, NQPKV) \
    { \
        using kernel_traits = unified_attention_decode_tiny_kernel_traits<DType, /*IsMasking=*/true, HSize, BM, NQPKV, 32, /*IsLocal=*/true>; \
        return unified_attention_kernel_dispatch_decode<kernel_traits>(args, config); \
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

enum class tile_tier { large, medium, small, tiny };

static tile_tier select_tile_tier(const unified_attention_args& args)
{
    const index_t avg_q = args.num_seqs > 0 ? args.num_tokens / args.num_seqs : args.num_tokens;
    const index_t kBlockQ_tiny = 16 / args.num_queries_per_kv;
    const index_t kBlockQ_small = 64 / args.num_queries_per_kv;
    [[maybe_unused]] const index_t kBlockQ_medium = 128 / args.num_queries_per_kv;

    // Decode tiers use a 2D grid (num_kv_heads, num_seqs) that assumes each
    // seq has at most kBlockQ tokens. For mixed batches where some seqs have
    // many more tokens, we must use the medium tier (1D grid with Q tile iteration).
    const index_t max_q = args.max_seqlen_q > 0 ? args.max_seqlen_q : avg_q;

    if(avg_q <= kBlockQ_tiny && max_q <= kBlockQ_tiny)
        return tile_tier::tiny;

    if(avg_q <= kBlockQ_small && max_q <= kBlockQ_small)
        return tile_tier::small;

    return tile_tier::medium;
}

std::pair<bool, float> unified_attention(const unified_attention_args& args,
                                         const stream_config& config)
{
    const bool is_mask = (args.mask_type != static_cast<int>(mask_enum::no_mask));
    // Real SWA = "at least one non-trivial window edge". Plain causal lives at
    // (left=-1, right=0); without this guard it would hit the IsLocal=true path
    // and fail for shape tiers where we have not (yet) instantiated local kernels.
    //   left  >= 0   : finite look-back  (e.g. causal SWA, dense SWA, diagonal-only)
    //   right >  0   : finite look-ahead (bidirectional SWA, anti-causal SWA)
    // Note "right >= 0" would mis-classify plain causal (right=0) as SWA.
    const bool is_local =
        is_mask && (args.window_size_left >= 0 || args.window_size_right > 0);
    auto tier = select_tile_tier(args);
    // SWA-instance availability matrix (IsLocal=true). Anything not listed here
    // returns {false, 0} so the caller (e.g. _try_ck_unified_attention) falls
    // back to a backend that handles it (Triton). Falling through to the
    // IsLocal=false path would silently ignore window_size_left and produce
    // wrong outputs, so we reject explicitly.
    //
    //   shape          | page_blk_size | tier we route to            | instance
    //   ---------------+---------------+-----------------------------+---------------------
    //   d128 MHA       |   >= 32       | tile_tier::large            | d128_*_mask_local
    //   d64  GQA-8     |   >= 64       | tile_tier::large            | d64_*_mask_gqa8_local
    //   d64  GQA-8     |   == 32, tiny | tile_tier::tiny             | d64_*_mask_gqa8_bs32_decode_t_local
    //   d64  GQA-8     |   == 32, med  | tile_tier::medium           | d64_*_mask_gqa8_bs32_decode_local
    //   (small+bs32 SWA has no instance yet -> Triton fallback; GPT-OSS shows zero
    //    such shapes in practice. Bumping it to medium is plausible but wastes a
    //    full kBlockM=128 tile when only kBlockM=64 is needed -- revisit if the
    //    workload changes.)
    if(is_local)
    {
        const bool d128_mha = (args.hdim == 128 && args.num_queries_per_kv == 1);
        const bool d64_gqa8 = (args.hdim == 64  && args.num_queries_per_kv == 8);
        if(d128_mha)
        {
            if(args.page_blk_size < 32) return {false, 0.f};
            tier = tile_tier::large;
        }
        else if(d64_gqa8)
        {
            if(args.page_blk_size >= 64)
            {
                tier = tile_tier::large;
            }
            else if(args.page_blk_size >= 32)
            {
                if(tier != tile_tier::tiny && tier != tile_tier::medium)
                    return {false, 0.f};
                // keep selected tier as-is
            }
            else
            {
                return {false, 0.f};
            }
        }
        else
        {
            return {false, 0.f};
        }
    }

    // d128, MHA (num_queries_per_kv == 1)
    if(args.hdim == 128 && args.num_queries_per_kv == 1)
    {
        if(args.data_type == unified_attention_args::data_type_enum::fp16)
        {
            if(!is_mask)       DISPATCH_UNIFIED_ATTENTION(unified_attention_args::data_type_enum::fp16, false, 128, 256, 1)
            else if(is_local)  DISPATCH_UNIFIED_ATTENTION_LOCAL(unified_attention_args::data_type_enum::fp16,         128, 256, 1, 32)
            else               DISPATCH_UNIFIED_ATTENTION(unified_attention_args::data_type_enum::fp16, true,  128, 256, 1)
        }
        else if(args.data_type == unified_attention_args::data_type_enum::bf16)
        {
            if(!is_mask)       DISPATCH_UNIFIED_ATTENTION(unified_attention_args::data_type_enum::bf16, false, 128, 256, 1)
            else if(is_local)  DISPATCH_UNIFIED_ATTENTION_LOCAL(unified_attention_args::data_type_enum::bf16,         128, 256, 1, 32)
            else               DISPATCH_UNIFIED_ATTENTION(unified_attention_args::data_type_enum::bf16, true,  128, 256, 1)
        }
    }

    // d64, GQA-8 (num_queries_per_kv == 8)
    if(args.hdim == 64 && args.num_queries_per_kv == 8)
    {
        const bool use_bs32 = (args.page_blk_size < 64);

        if(tier == tile_tier::tiny)
        {
            if(use_bs32) {
                // Tiny+bs32: 1 warp, 16x16 MFMA, kBlockM=16, kBlockQ=2 (matches Triton decode).
                // Decode grid requires max_seqlen_q <= kBlockQ (see select_tile_tier).
                if(args.data_type == unified_attention_args::data_type_enum::fp16)
                {
                    if(!is_mask)      DISPATCH_UNIFIED_ATTENTION_DECODE_TINY_BS32(unified_attention_args::data_type_enum::fp16, false, 64, 16, 8)
                    else if(is_local) DISPATCH_UNIFIED_ATTENTION_DECODE_TINY_BS32_LOCAL(unified_attention_args::data_type_enum::fp16,         64, 16, 8)
                    else              DISPATCH_UNIFIED_ATTENTION_DECODE_TINY_BS32(unified_attention_args::data_type_enum::fp16, true,  64, 16, 8)
                }
                else if(args.data_type == unified_attention_args::data_type_enum::bf16)
                {
                    if(!is_mask)      DISPATCH_UNIFIED_ATTENTION_DECODE_TINY_BS32(unified_attention_args::data_type_enum::bf16, false, 64, 16, 8)
                    else if(is_local) DISPATCH_UNIFIED_ATTENTION_DECODE_TINY_BS32_LOCAL(unified_attention_args::data_type_enum::bf16,         64, 16, 8)
                    else              DISPATCH_UNIFIED_ATTENTION_DECODE_TINY_BS32(unified_attention_args::data_type_enum::bf16, true,  64, 16, 8)
                }
            } else {
                // bs64 tiny: 1 warp, 16x16 MFMA, kBlockM=16, kBlockQ=2.
                if(args.data_type == unified_attention_args::data_type_enum::fp16)
                {
                    if(!is_mask) DISPATCH_UNIFIED_ATTENTION_DECODE_TINY(unified_attention_args::data_type_enum::fp16, false, 64, 16, 8)
                    else         DISPATCH_UNIFIED_ATTENTION_DECODE_TINY(unified_attention_args::data_type_enum::fp16, true,  64, 16, 8)
                }
                else if(args.data_type == unified_attention_args::data_type_enum::bf16)
                {
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
                    if(!is_mask)      DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM_BS32(unified_attention_args::data_type_enum::fp16, false, 64, 128, 8)
                    else if(is_local) DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM_BS32_LOCAL(unified_attention_args::data_type_enum::fp16,         64, 128, 8)
                    else              DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM_BS32(unified_attention_args::data_type_enum::fp16, true,  64, 128, 8)
                } else {
                    if(!is_mask) DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM(unified_attention_args::data_type_enum::fp16, false, 64, 128, 8)
                    else         DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM(unified_attention_args::data_type_enum::fp16, true,  64, 128, 8)
                }
            }
            else if(args.data_type == unified_attention_args::data_type_enum::bf16)
            {
                if(use_bs32) {
                    if(!is_mask)      DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM_BS32(unified_attention_args::data_type_enum::bf16, false, 64, 128, 8)
                    else if(is_local) DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM_BS32_LOCAL(unified_attention_args::data_type_enum::bf16,         64, 128, 8)
                    else              DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM_BS32(unified_attention_args::data_type_enum::bf16, true,  64, 128, 8)
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
                if(!is_mask)      DISPATCH_UNIFIED_ATTENTION(unified_attention_args::data_type_enum::fp16, false, 64, 256, 8)
                else if(is_local) DISPATCH_UNIFIED_ATTENTION_LOCAL(unified_attention_args::data_type_enum::fp16,         64, 256, 8, 64)
                else              DISPATCH_UNIFIED_ATTENTION(unified_attention_args::data_type_enum::fp16, true,  64, 256, 8)
            }
            else if(args.data_type == unified_attention_args::data_type_enum::bf16)
            {
                if(!is_mask)      DISPATCH_UNIFIED_ATTENTION(unified_attention_args::data_type_enum::bf16, false, 64, 256, 8)
                else if(is_local) DISPATCH_UNIFIED_ATTENTION_LOCAL(unified_attention_args::data_type_enum::bf16,         64, 256, 8, 64)
                else              DISPATCH_UNIFIED_ATTENTION(unified_attention_args::data_type_enum::bf16, true,  64, 256, 8)
            }
        }
    }

    std::cerr << "unified_attention: no matching kernel instance for hdim=" << args.hdim
              << " num_queries_per_kv=" << args.num_queries_per_kv
              << " data_type=" << args.data_type << " mask_type=" << args.mask_type << std::endl;
    return std::make_pair(false, -1.f);
}

#undef DISPATCH_UNIFIED_ATTENTION_DECODE_TINY_BS32_LOCAL
#undef DISPATCH_UNIFIED_ATTENTION_DECODE_TINY_BS32
#undef DISPATCH_UNIFIED_ATTENTION_DECODE_BS32_NARROW_LOCAL
#undef DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM_BS32_LOCAL
#undef DISPATCH_UNIFIED_ATTENTION_DECODE_BS32_NARROW
#undef DISPATCH_UNIFIED_ATTENTION_DECODE_SMALL_BS32
#undef DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM_BS32
#undef DISPATCH_UNIFIED_ATTENTION_DECODE_TINY
#undef DISPATCH_UNIFIED_ATTENTION_DECODE_SMALL
#undef DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM
#undef DISPATCH_UNIFIED_ATTENTION_LOCAL
#undef DISPATCH_UNIFIED_ATTENTION

} // namespace ck_tile
