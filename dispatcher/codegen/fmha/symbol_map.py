#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import hashlib

# Architecture tag → C++ arch trait type.
# Source: CK's include/ck_tile/core/arch/amd_gpu_traits.hpp
#   gfx9* → gfx9_t, gfx11* → gfx11_t, gfx12* → gfx12_t.
ARCH_TAG_MAP = {
    "gfx90a": "ck_tile::gfx9_t",
    "gfx942": "ck_tile::gfx9_t",
    "gfx950": "ck_tile::gfx9_t",
    "gfx1100": "ck_tile::gfx11_t",
    "gfx1201": "ck_tile::gfx12_t",
}

# Architecture → preprocessor guard for conditional compilation.
# Source: HIP compiler predefined macros (__gfx90a__, __gfx942__, etc.).
ARCH_PREPROC_MAP = {
    "gfx90a": "defined(__gfx90a__)",
    "gfx942": "defined(__gfx942__)",
    "gfx950": "defined(__gfx950__)",
    "gfx1100": "defined(__gfx1100__)",
    "gfx1201": "defined(__gfx1201__)",
}

# Forward dtype → C++ type config struct.
# Source: example/ck_tile/01_fmha/fmha_fwd.hpp FmhaFwdTypeConfig<> specializations
#   and codegen/cpp_symbol_map.py FWD_DTYPE_MAP.
FWD_DTYPE_MAP = {
    "fp32": "FmhaFwdFp32",
    "fp16": "FmhaFwdFp16",
    "bf16": "FmhaFwdBf16",
    "fp8": "FmhaFwdFp8",
    "bf8": "FmhaFwdBf8",
    "fp8fp16": "FmhaFwdFp8Fp16",
    "fp8bf16": "FmhaFwdFp8Bf16",
    "fp8fp32": "FmhaFwdFp8Fp32",
}

# Backward dtype → C++ type config struct.
# Source: example/ck_tile/01_fmha/fmha_bwd.hpp FmhaBwdTypeConfig<> specializations.
# BWD currently only supports fp16/bf16/fp32.
BWD_DTYPE_MAP = {
    "fp32": "FmhaBwdFp32",
    "fp16": "FmhaBwdFp16",
    "bf16": "FmhaBwdBf16",
}

# Kernel family → C++ enum.
# Source: include/ck_tile/dispatcher/fmha_types.hpp FmhaKernelFamily enum.
KERNEL_FAMILY_TO_ENUM = {
    "fwd": "FmhaKernelFamily::Fwd",
    "fwd_pagedkv": "FmhaKernelFamily::FwdPagedKv",
    "fwd_splitkv": "FmhaKernelFamily::FwdSplitKv",
    "fwd_splitkv_combine": "FmhaKernelFamily::FwdSplitKvCombine",
    "fwd_appendkv": "FmhaKernelFamily::FwdAppendKv",
    "batch_prefill": "FmhaKernelFamily::BatchPrefill",
    "bwd_dot_do_o": "FmhaKernelFamily::BwdDotDoO",
    "bwd_dq_dk_dv": "FmhaKernelFamily::BwdDqDkDv",
    "bwd_convert_dq": "FmhaKernelFamily::BwdConvertDq",
}

# API family → C++ enum.
# Source: include/ck_tile/dispatcher/fmha_types.hpp FmhaApiFamily enum.
API_FAMILY_TO_ENUM = {
    "fwd": "FmhaApiFamily::Fwd",
    "fwd_pagedkv": "FmhaApiFamily::FwdPagedKv",
    "fwd_splitkv": "FmhaApiFamily::FwdSplitKv",
    "fwd_appendkv": "FmhaApiFamily::FwdAppendKv",
    "batch_prefill": "FmhaApiFamily::BatchPrefill",
    "bwd": "FmhaApiFamily::Bwd",
}

# Mask type → canonical form and C++ types.
# Source: include/ck_tile/ops/fmha/block/block_attention_mask.hpp
#   SimplifiedGenericAttentionMask<is_causal> and GenericAttentionMask<has_mask, has_local>.
MASK_CANONICAL = {
    "no": "no",
    "no_mask": "no",
    "causal": "top_left",
    "top_left": "top_left",
    "t": "top_left",
    "bottom_right": "bottom_right",
    "b": "bottom_right",
    "generic": "generic",
    "window_generic": "generic",
    "g": "generic",
}

MASK_TO_CPP = {
    "no": "ck_tile::SimplifiedGenericAttentionMask<false>",
    "top_left": "ck_tile::SimplifiedGenericAttentionMask<true>",
    "bottom_right": "ck_tile::SimplifiedGenericAttentionMask<true>",
    "generic": "ck_tile::GenericAttentionMask<true, true>",
}

MASK_TO_CPP_GENERIC = {
    "no": "FmhaMasks::NoMask",
    "top_left": "FmhaMasks::CausalMask",
    "bottom_right": "FmhaMasks::CausalMask",
    "generic": "FmhaMasks::GenericMask",
}

MASK_TO_INT = {
    "no": 0,
    "top_left": 1,
    "bottom_right": 2,
    "generic": 3,
}

# Bias type → canonical form and C++ enum.
# Source: include/ck_tile/ops/fmha/block/block_attention_bias_enum.hpp.
BIAS_CANONICAL = {
    "no": "no",
    "no_bias": "no",
    "bias": "bias",
    "elementwise": "bias",
    "elementwise_bias": "bias",
    "alibi": "alibi",
}

BIAS_TO_CPP = {
    "no": "ck_tile::BlockAttentionBiasEnum::NO_BIAS",
    "bias": "ck_tile::BlockAttentionBiasEnum::ELEMENTWISE_BIAS",
    "alibi": "ck_tile::BlockAttentionBiasEnum::ALIBI",
}

BIAS_TO_INT = {
    "no": 0,
    "bias": 1,
    "alibi": 2,
}

# Quantization scale type → canonical form and C++ enum.
# Source: include/ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp.
QSCALE_CANONICAL = {
    "no": "no",
    "no_scale": "no",
    "pertensor": "pertensor",
    "blockscale": "blockscale",
    "kv_blockscale": "kv_blockscale",
}

QSCALE_TO_CPP = {
    "no": "ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE",
    "pertensor": "ck_tile::BlockAttentionQuantScaleEnum::PERTENSOR",
    "blockscale": "ck_tile::BlockAttentionQuantScaleEnum::BLOCKSCALE",
    "kv_blockscale": "ck_tile::BlockAttentionQuantScaleEnum::KV_BLOCKSCALE",
}

QSCALE_TO_INT = {
    "no": 0,
    "pertensor": 1,
    "blockscale": 2,
    "kv_blockscale": 3,
}

# Rotary embedding type → canonical form and C++ enum.
# Source: include/ck_tile/ops/fmha/block/rotary_embedding_enum.hpp.
ROPE_CANONICAL = {
    "none": "none",
    "no": "none",
    "inter": "inter",
    "interleaved": "inter",
    "half": "half",
    "half_rotated": "half",
}

ROPE_TO_CPP = {
    "none": "ck_tile::RotaryEmbeddingEnum::NONE",
    "inter": "ck_tile::RotaryEmbeddingEnum::INTERLEAVED",
    "half": "ck_tile::RotaryEmbeddingEnum::HALF_ROTATED",
}

ROPE_TO_INT = {
    "none": 0,
    "inter": 1,
    "half": 2,
}

# V layout → C++ bool (true = row-major, false = column-major).
# Source: TileFmhaShape<..., IsVLayoutRowMajor> template parameter.
LAYOUT_TO_BOOL = {
    "r": "true",
    "row": "true",
    "row_major": "true",
    "c": "false",
    "col": "false",
    "col_major": "false",
}

# KV cache memory layout → canonical form and C++ enum.
# Source: include/ck_tile/ops/fmha/block/block_attention_kv_cache.hpp.
KV_MEMORY_LAYOUT_CANONICAL = {
    "vectorized": "vectorized",
    "linear": "linear",
}

KV_MEMORY_LAYOUT_TO_CPP = {
    "vectorized": "ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT",
    "linear": "ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::LINEAR_LAYOUT",
}

KV_MEMORY_LAYOUT_TO_INT = {
    "vectorized": 0,
    "linear": 1,
}

# KV lookup table type → canonical form and C++ enum.
# Source: include/ck_tile/ops/fmha/block/block_attention_kv_cache.hpp.
KV_LOOKUP_CANONICAL = {
    "sglang": "sglang",
    "vllm": "vllm",
}

KV_LOOKUP_TO_CPP = {
    "sglang": "ck_tile::BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D",
    "vllm": "ck_tile::BlockAttentionKVCacheLookupTableEnum::VLLM_BLOCK_TABLE_2D",
}

KV_LOOKUP_TO_INT = {
    "vllm": 0,
    "sglang": 1,
}

# Pipeline tag → C++ pipeline class.
# Source: include/ck_tile/ops/fmha/pipeline/ — one header per pipeline variant.
PIPELINE_TO_CPP = {
    "qr": "ck_tile::BlockFmhaPipelineQRKSVS",
    "qr_async": "ck_tile::BlockFmhaPipelineQRKSVSAsync",
    "qs": "ck_tile::BlockFmhaPipelineQSKSVS",
    "qr_async_trload": "ck_tile::BlockFmhaPipelineQRKSVSAsyncTrload",
    "v3": "ck_tile::BlockFmhaFwdV3Pipeline",
    "qr_async_trload_v3": "ck_tile::BlockFmhaFwdV3Pipeline",
    "qr_pagedkv": "ck_tile::BlockFmhaFwdPagedKVPipelineQRKSVS",
    "qr_nwarp_sshuffle": "ck_tile::BlockFmhaFwdSplitKVPipelineNWarpSShuffleQRKSVS",
    "appendkv": "ck_tile::BlockFmhaFwdAppendKVPipeline",
    "batch_prefill_async": "ck_tile::BlockFmhaBatchPrefillPipelineQRKSVSAsync",
}

# Pipeline tag → C++ pipeline enum value.
# Source: include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_enum.hpp.
PIPELINE_ENUM_TO_CPP = {
    "qr": "ck_tile::BlockFmhaPipelineEnum::QRKSVS",
    "qr_async": "ck_tile::BlockFmhaPipelineEnum::QRKSVS_ASYNC",
    "qs": "ck_tile::BlockFmhaPipelineEnum::QSKSVS",
    "qr_async_trload": "ck_tile::BlockFmhaPipelineEnum::QRKSVS_ASYNC_TRLOAD",
    "v3": "ck_tile::BlockFmhaPipelineEnum::QRKSVS_ASYNC_TRLOAD_V3",
    "qr_async_trload_v3": "ck_tile::BlockFmhaPipelineEnum::QRKSVS_ASYNC_TRLOAD_V3",
    "qr_pagedkv": "ck_tile::BlockFmhaPipelineEnum::QRKSVS",
    "qr_nwarp_sshuffle": "ck_tile::BlockFmhaPipelineEnum::QRKSVS",
    "batch_prefill_async": "ck_tile::BlockFmhaPipelineEnum::QRKSVS_ASYNC",
}

BOOL_MAP = {
    True: "true",
    False: "false",
    "t": "true",
    "f": "false",
}


def canonical_mask(value: str) -> str:
    return MASK_CANONICAL.get(value, value)


def canonical_bias(value: str) -> str:
    return BIAS_CANONICAL.get(value, value)


def canonical_qscale(value: str) -> str:
    return QSCALE_CANONICAL.get(value, value)


def canonical_rope(value: str) -> str:
    return ROPE_CANONICAL.get(value, value)


def canonical_kv_memory_layout(value: str) -> str:
    return KV_MEMORY_LAYOUT_CANONICAL.get(value, value)


def canonical_kv_lookup(value: str) -> str:
    return KV_LOOKUP_CANONICAL.get(value, value)


def sanitize_token(value) -> str:
    return str(value).replace("::", "_").replace("/", "_").replace(" ", "_")


def kernel_name_from_config(config: dict) -> str:
    sig = config["signature"]
    alg = config["algorithm"]

    family = sanitize_token(sig["family"])
    dtype = sanitize_token(sig["data_type"])
    mode = sanitize_token(sig["mode"])
    vlayout = sanitize_token(sig["vlayout"])
    mask = sanitize_token(canonical_mask(sig["mask"]))
    bias = sanitize_token(canonical_bias(sig["bias"]))
    qscale = sanitize_token(canonical_qscale(sig["qscale"]))
    rope = sanitize_token(canonical_rope(sig["rope"]))
    kv_memory = sanitize_token(canonical_kv_memory_layout(sig["kv_memory_layout"]))
    kv_lookup = sanitize_token(canonical_kv_lookup(sig["kv_lookup_table"]))
    pipeline = sanitize_token(alg["pipeline"])

    canonical_blob = json.dumps(
        {
            "family": family,
            "dtype": dtype,
            "mode": mode,
            "vlayout": vlayout,
            "mask": mask,
            "bias": bias,
            "qscale": qscale,
            "rope": rope,
            "kv_memory": kv_memory,
            "kv_lookup": kv_lookup,
            "sig": sig,
            "alg": alg,
        },
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha1(canonical_blob).hexdigest()[:12]

    return (
        f"fmha_{family}_{dtype}_{mode}_h{sig['hdim_q']}x{sig['hdim_v']}"
        f"_{pipeline}_{digest}"
    )
