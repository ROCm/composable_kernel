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

// =============================================================================
// Config selection
//
// The job is split in two halves so each is small enough to read in one sitting:
//
//   1. KernelVariant + select_config(args)
//        - KernelVariant is a flat enum of every compiled kernel instance the
//          module knows about. Each entry fixes the static knobs (kBlockM,
//          warp count, MFMA shape, pipeline policy).
//        - select_config() is the ONLY place where shape-based runtime
//          decisions live. It reads (hdim, num_queries_per_kv, avg_q,
//          max_seqlen_q) and emits a KernelConfig.
//
//   2. dispatch_<variant>() helpers + the final switch
//        - Each KernelVariant has a tiny helper that fans out over the
//          (dtype, mask) cross-product and calls into the existing
//          DISPATCH_UNIFIED_ATTENTION_* macros. The macros and the
//          per-variant traits classes are unchanged from before; only the
//          selection logic moved.
//
// page_size is intentionally NOT part of this enum. The multi-page-tile
// fix in the pipeline made the compile-time tile-N (kBlockN) independent
// of the runtime page_blk_size, so every variant is correct for any page
// size. Selection is driven purely by Q-tile shape.
// =============================================================================

enum class KernelVariant {
    // d=128 MHA (num_queries_per_kv = 1)
    prefill_d128_mha,            // kBlockM=256, 8 warps, 32x32 mfma
    decode_d128_mha_m128,        // kBlockM=128, 4 warps, 32x32 mfma  (kBlockQ=128)

    // d=64 GQA-8 (num_queries_per_kv = 8)
    prefill_d64_gqa8,            // kBlockM=256, 8 warps, 32x32 mfma
    decode_d64_gqa8_m128,        // kBlockM=128, 4 warps, 32x32 mfma
    decode_d64_gqa8_m64,         // kBlockM=64,  2 warps, 32x32 mfma
    decode_d64_gqa8_m16,         // kBlockM=16,  1 warp,  16x16 mfma
};

struct KernelConfig {
    KernelVariant variant;
    bool          unsupported = false;
};

namespace {

// Internal tier classification — used only by select_config. The tier name is
// just shorthand for a kBlockM choice; with num_queries_per_kv=8 the tiers
// correspond to kBlockQ thresholds {2, 8, 16}.
enum class tile_tier { medium, small, tiny };

tile_tier select_tile_tier(const unified_attention_args& args)
{
    const index_t avg_q = args.num_seqs > 0 ? args.num_tokens / args.num_seqs
                                            : args.num_tokens;
    const index_t kBlockQ_tiny  = 16 / args.num_queries_per_kv;
    const index_t kBlockQ_small = 64 / args.num_queries_per_kv;

    // Decode tiers use a 2D grid (num_kv_heads, num_seqs) that assumes each
    // seq has at most kBlockQ tokens. For mixed batches where some seqs have
    // many more tokens, fall back to the medium tier (1D grid with Q iteration).
    const index_t max_q = args.max_seqlen_q > 0 ? args.max_seqlen_q : avg_q;

    if (avg_q <= kBlockQ_tiny  && max_q <= kBlockQ_tiny)  return tile_tier::tiny;
    if (avg_q <= kBlockQ_small && max_q <= kBlockQ_small) return tile_tier::small;
    return tile_tier::medium;
}

} // anonymous namespace

KernelConfig select_config(const unified_attention_args& args)
{
    KernelConfig cfg;

    // d=128 MHA — two variants today:
    //   * decode_d128_mha_m128 : kBlockM=128, 4 warps. Used when avg/max Q
    //     both fit in 128. Cuts Q-tile waste roughly 2x vs prefill for
    //     short-Q workloads.
    //   * prefill_d128_mha     : kBlockM=256, 8 warps. Everything else.
    if (args.hdim == 128 && args.num_queries_per_kv == 1)
    {
        const index_t avg_q = args.num_seqs > 0 ? args.num_tokens / args.num_seqs
                                                : args.num_tokens;
        const index_t max_q = args.max_seqlen_q > 0 ? args.max_seqlen_q : avg_q;

        if (avg_q <= 128 && max_q <= 128)
            cfg.variant = KernelVariant::decode_d128_mha_m128;
        else
            cfg.variant = KernelVariant::prefill_d128_mha;
        return cfg;
    }

    // d=64 GQA-8 — pure tile-tier ladder. page_size has no influence here.
    if (args.hdim == 64 && args.num_queries_per_kv == 8)
    {
        switch (select_tile_tier(args))
        {
        case tile_tier::tiny:   cfg.variant = KernelVariant::decode_d64_gqa8_m16;  break;
        case tile_tier::small:  cfg.variant = KernelVariant::decode_d64_gqa8_m64;  break;
        case tile_tier::medium: cfg.variant = KernelVariant::decode_d64_gqa8_m128; break;
        }
        return cfg;
    }

    cfg.unsupported = true;
    return cfg;
}

// -----------------------------------------------------------------------------
// Dispatch macros and per-variant dispatch helpers.
//
// Each DISPATCH_* macro instantiates one (traits, dtype, mask, ...) combo and
// returns. The per-variant helpers below pick the right macro family and fan
// out over (dtype, mask). They look repetitive on purpose: a follow-up commit
// will collapse the 4 traits classes into one templated `kernel_traits<V>`,
// at which point these helpers become one-liners.
// -----------------------------------------------------------------------------

// Helper macro: dispatches based on DataType, IsMasking, HeadSize, BlockM, NumQPerKV.
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

namespace {

using DType = unified_attention_args::data_type_enum;

std::pair<bool, float> dispatch_prefill_d128_mha(
    const unified_attention_args& args, const stream_config& config)
{
    const bool is_mask = (args.mask_type != static_cast<int>(mask_enum::no_mask));
    if (args.data_type == DType::fp16) {
        if (!is_mask) DISPATCH_UNIFIED_ATTENTION(DType::fp16, false, 128, 256, 1)
        else          DISPATCH_UNIFIED_ATTENTION(DType::fp16, true,  128, 256, 1)
    } else if (args.data_type == DType::bf16) {
        if (!is_mask) DISPATCH_UNIFIED_ATTENTION(DType::bf16, false, 128, 256, 1)
        else          DISPATCH_UNIFIED_ATTENTION(DType::bf16, true,  128, 256, 1)
    }
    return {false, -1.f};
}

std::pair<bool, float> dispatch_decode_d128_mha_m128(
    const unified_attention_args& args, const stream_config& config)
{
    const bool is_mask = (args.mask_type != static_cast<int>(mask_enum::no_mask));
    if (args.data_type == DType::fp16) {
        if (!is_mask) DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM(DType::fp16, false, 128, 128, 1)
        else          DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM(DType::fp16, true,  128, 128, 1)
    } else if (args.data_type == DType::bf16) {
        if (!is_mask) DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM(DType::bf16, false, 128, 128, 1)
        else          DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM(DType::bf16, true,  128, 128, 1)
    }
    return {false, -1.f};
}

std::pair<bool, float> dispatch_prefill_d64_gqa8(
    const unified_attention_args& args, const stream_config& config)
{
    const bool is_mask = (args.mask_type != static_cast<int>(mask_enum::no_mask));
    if (args.data_type == DType::fp16) {
        if (!is_mask) DISPATCH_UNIFIED_ATTENTION(DType::fp16, false, 64, 256, 8)
        else          DISPATCH_UNIFIED_ATTENTION(DType::fp16, true,  64, 256, 8)
    } else if (args.data_type == DType::bf16) {
        if (!is_mask) DISPATCH_UNIFIED_ATTENTION(DType::bf16, false, 64, 256, 8)
        else          DISPATCH_UNIFIED_ATTENTION(DType::bf16, true,  64, 256, 8)
    }
    return {false, -1.f};
}

std::pair<bool, float> dispatch_decode_d64_gqa8_m128(
    const unified_attention_args& args, const stream_config& config)
{
    const bool is_mask = (args.mask_type != static_cast<int>(mask_enum::no_mask));
    if (args.data_type == DType::fp16) {
        if (!is_mask) DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM(DType::fp16, false, 64, 128, 8)
        else          DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM(DType::fp16, true,  64, 128, 8)
    } else if (args.data_type == DType::bf16) {
        if (!is_mask) DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM(DType::bf16, false, 64, 128, 8)
        else          DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM(DType::bf16, true,  64, 128, 8)
    }
    return {false, -1.f};
}

std::pair<bool, float> dispatch_decode_d64_gqa8_m64(
    const unified_attention_args& args, const stream_config& config)
{
    const bool is_mask = (args.mask_type != static_cast<int>(mask_enum::no_mask));
    if (args.data_type == DType::fp16) {
        if (!is_mask) DISPATCH_UNIFIED_ATTENTION_DECODE_SMALL(DType::fp16, false, 64, 64, 8)
        else          DISPATCH_UNIFIED_ATTENTION_DECODE_SMALL(DType::fp16, true,  64, 64, 8)
    } else if (args.data_type == DType::bf16) {
        if (!is_mask) DISPATCH_UNIFIED_ATTENTION_DECODE_SMALL(DType::bf16, false, 64, 64, 8)
        else          DISPATCH_UNIFIED_ATTENTION_DECODE_SMALL(DType::bf16, true,  64, 64, 8)
    }
    return {false, -1.f};
}

std::pair<bool, float> dispatch_decode_d64_gqa8_m16(
    const unified_attention_args& args, const stream_config& config)
{
    const bool is_mask = (args.mask_type != static_cast<int>(mask_enum::no_mask));
    if (args.data_type == DType::fp16) {
        if (!is_mask) DISPATCH_UNIFIED_ATTENTION_DECODE_TINY(DType::fp16, false, 64, 16, 8)
        else          DISPATCH_UNIFIED_ATTENTION_DECODE_TINY(DType::fp16, true,  64, 16, 8)
    } else if (args.data_type == DType::bf16) {
        if (!is_mask) DISPATCH_UNIFIED_ATTENTION_DECODE_TINY(DType::bf16, false, 64, 16, 8)
        else          DISPATCH_UNIFIED_ATTENTION_DECODE_TINY(DType::bf16, true,  64, 16, 8)
    }
    return {false, -1.f};
}

} // anonymous namespace

std::pair<bool, float> unified_attention(const unified_attention_args& args,
                                         const stream_config& config)
{
    const auto cfg = select_config(args);

    if (cfg.unsupported)
    {
        std::cerr << "unified_attention: no matching kernel instance for hdim=" << args.hdim
                  << " num_queries_per_kv=" << args.num_queries_per_kv
                  << " data_type=" << args.data_type
                  << " mask_type=" << args.mask_type << std::endl;
        return std::make_pair(false, -1.f);
    }

    switch (cfg.variant)
    {
    case KernelVariant::prefill_d128_mha:     return dispatch_prefill_d128_mha(args, config);
    case KernelVariant::decode_d128_mha_m128: return dispatch_decode_d128_mha_m128(args, config);
    case KernelVariant::prefill_d64_gqa8:     return dispatch_prefill_d64_gqa8(args, config);
    case KernelVariant::decode_d64_gqa8_m128: return dispatch_decode_d64_gqa8_m128(args, config);
    case KernelVariant::decode_d64_gqa8_m64:  return dispatch_decode_d64_gqa8_m64(args, config);
    case KernelVariant::decode_d64_gqa8_m16:  return dispatch_decode_d64_gqa8_m16(args, config);
    }
    return std::make_pair(false, -1.f);
}

#undef DISPATCH_UNIFIED_ATTENTION_DECODE_TINY
#undef DISPATCH_UNIFIED_ATTENTION_DECODE_SMALL
#undef DISPATCH_UNIFIED_ATTENTION_DECODE_MEDIUM
#undef DISPATCH_UNIFIED_ATTENTION

} // namespace ck_tile
