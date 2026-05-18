#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Unified FMHA code generator for the dispatcher.

This generator intentionally sits between the hand-maintained FMHA example codegen
and the dispatcher's runtime-registry model:

- it consumes explicit kernel configurations or profile-filtered config lists
- it emits one header per FMHA kernel specialization
- it emits dispatcher wrapper headers that create FmhaKernelInstance objects
- it emits one .cpp translation unit per generated kernel header
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, Union

# Ensure parent (codegen/) is on path for codegen_common and sibling modules
_CODEGEN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODEGEN_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from codegen_common import parallel_generate  # noqa: E402
from validation import load_arch_specs, profile_allows, validate_config  # noqa: E402
from symbol_map import (  # noqa: E402
    ARCH_PREPROC_MAP,
    ARCH_TAG_MAP,
    BIAS_TO_CPP,
    BIAS_TO_INT,
    BOOL_MAP,
    BWD_DTYPE_MAP,
    FWD_DTYPE_MAP,
    KERNEL_FAMILY_TO_ENUM,
    KV_LOOKUP_TO_INT,
    KV_LOOKUP_TO_CPP,
    KV_MEMORY_LAYOUT_TO_CPP,
    KV_MEMORY_LAYOUT_TO_INT,
    LAYOUT_TO_BOOL,
    MASK_TO_CPP,
    MASK_TO_CPP_GENERIC,
    MASK_TO_INT,
    PIPELINE_ENUM_TO_CPP,
    QSCALE_TO_CPP,
    QSCALE_TO_INT,
    ROPE_TO_CPP,
    ROPE_TO_INT,
    canonical_bias,
    canonical_kv_lookup,
    canonical_kv_memory_layout,
    canonical_mask,
    canonical_qscale,
    canonical_rope,
    kernel_name_from_config,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _bool_cpp(value) -> str:
    return BOOL_MAP[bool(value)]


def _mask_cpp(value: str) -> str:
    return MASK_TO_CPP[canonical_mask(value)]


def _bias_cpp(value: str) -> str:
    return BIAS_TO_CPP[canonical_bias(value)]


def _qscale_cpp(value: str) -> str:
    return QSCALE_TO_CPP[canonical_qscale(value)]


def _rope_cpp(value: str) -> str:
    return ROPE_TO_CPP[canonical_rope(value)]


def _kv_memory_cpp(value: str) -> str:
    return KV_MEMORY_LAYOUT_TO_CPP[canonical_kv_memory_layout(value)]


def _kv_lookup_cpp(value: str) -> str:
    return KV_LOOKUP_TO_CPP[canonical_kv_lookup(value)]


def _bwd_block_tile(tile: list, sig: dict) -> str:
    """Format the bwd block tile sequence.

    Source: fmha_bwd.hpp FmhaBwdDQDKDVTileSize — 9 elements:
    (bm0, bn0, bk0, bn1, bk1, bk0max, tile6, tile7, tile8).
    If tile has only 6 elements (forward-style), maps to BWD format using the
    forward-to-backward heuristic from fmha_bwd.py.
    """
    if len(tile) >= 9:
        return ", ".join(str(t) for t in tile[:9])
    return (
        f"{tile[0]}, {tile[1]}, {tile[2]}, {tile[3]}, {tile[4]}, "
        f"{tile[3]}, {tile[5]}, {sig['hdim_q']}, {sig['hdim_v']}"
    )


def _canonicalize_config(raw_config: dict, target_arch: str, arch_specs: dict) -> dict:
    defaults = arch_specs["defaults"]

    if "signature" not in raw_config or "algorithm" not in raw_config:
        raise ValueError(
            "FMHA config-json must contain 'signature' and 'algorithm' objects"
        )

    sig = dict(raw_config["signature"])
    alg = dict(raw_config["algorithm"])

    sig.setdefault("family", "fwd")
    sig.setdefault("data_type", "fp16")
    sig.setdefault("mode", "batch")
    sig.setdefault("vlayout", "r")
    sig.setdefault("hdim_q", 128)
    sig.setdefault("hdim_v", sig["hdim_q"])
    sig.setdefault("mask", "no")
    sig.setdefault("bias", "no")
    sig.setdefault("lse", False)
    sig.setdefault("dropout", False)
    sig.setdefault("qscale", "no")
    sig.setdefault("rope", "none")
    sig.setdefault("logits", False)
    sig.setdefault("paged_kv", False)
    sig.setdefault("fp8_static_quant", False)
    sig.setdefault("skip_min_seqlen_q", False)
    sig.setdefault("sink", False)
    sig.setdefault("dbias", False)
    sig.setdefault("store_randval", False)
    sig.setdefault("deterministic", False)
    sig.setdefault("kv_memory_layout", "vectorized")
    sig.setdefault("kv_lookup_table", "sglang")
    sig.setdefault("page_size", 1)

    sig["mask"] = canonical_mask(sig["mask"])
    sig["bias"] = canonical_bias(sig["bias"])
    sig["qscale"] = canonical_qscale(sig["qscale"])
    sig["rope"] = canonical_rope(sig["rope"])
    sig["kv_memory_layout"] = canonical_kv_memory_layout(sig["kv_memory_layout"])
    sig["kv_lookup_table"] = canonical_kv_lookup(sig["kv_lookup_table"])

    alg.setdefault("pipeline", "qr")
    alg.setdefault("tile", list(defaults["tile"]))
    alg.setdefault("wave", list(defaults["wave"]))
    alg.setdefault("warp", list(defaults["warp"]))
    alg.setdefault("padding", list(defaults["padding"]))
    alg.setdefault("use_trload", False)
    alg.setdefault("hdim_q_alignment", sig["hdim_q"])
    alg.setdefault("hdim_v_alignment", sig["hdim_v"])
    alg.setdefault("block_per_cu", defaults["block_per_cu"])
    alg.setdefault("num_wave_groups", defaults["num_wave_groups"])
    alg.setdefault("max_splits_log2", 0)
    alg.setdefault("max_seq_len_q", 0)
    alg.setdefault("selection_rank", defaults["selection_rank"])
    alg.setdefault("constraint_tag", "")

    return {
        "arch": raw_config.get("arch", target_arch),
        "signature": sig,
        "algorithm": alg,
        "profile": raw_config.get("profile"),
        "receipt": raw_config.get("receipt"),
    }


def _fwd_kernel_body(name: str, config: dict) -> str:
    sig = config["signature"]
    alg = config["algorithm"]
    arch_tag = ARCH_TAG_MAP[config["arch"]]
    dtype_cpp = FWD_DTYPE_MAP[sig["data_type"]]
    mode_cpp = "true" if sig["mode"] == "group" else "false"
    vlayout_cpp = LAYOUT_TO_BOOL[sig["vlayout"]]
    tile = alg["tile"]
    wave = alg["wave"]
    warp = alg["warp"]
    pad = alg["padding"]
    use_trload = _bool_cpp(alg["use_trload"])
    pipeline_name = alg["pipeline"]
    pipeline_cpp = {
        "qr": "ck_tile::BlockFmhaPipelineQRKSVS",
        "qr_async": "ck_tile::BlockFmhaPipelineQRKSVSAsync",
        "qs": "ck_tile::BlockFmhaPipelineQSKSVS",
        "qr_async_trload": "ck_tile::BlockFmhaPipelineQRKSVSAsyncTrload",
        "qr_async_trload_v3": "ck_tile::BlockFmhaFwdV3Pipeline",
        "v3": "ck_tile::BlockFmhaFwdV3Pipeline",
    }[pipeline_name]

    ns = f"ns_{name}"
    arch_check = ARCH_PREPROC_MAP.get(config["arch"], "1")
    return f"""// SPDX-License-Identifier: MIT
// Auto-generated FMHA forward kernel specialization
#pragma once

#include "ck_tile/ops/fmha/block/variants.hpp"
#include "example/ck_tile/01_fmha/fmha_fwd.hpp"

#if !defined(__HIP_DEVICE_COMPILE__) || ({arch_check})

namespace {ns} {{

using fmha_dtype = {dtype_cpp};
using fmha_block_tile = ck_tile::sequence<{tile[0]}, {tile[1]}, {tile[2]}, {tile[3]}, {tile[4]}, {tile[5]}>;

using fmha_shape = ck_tile::TileFmhaShape<fmha_block_tile,
                                          ck_tile::sequence<{wave[0]}, {wave[1]}, {wave[2]}>,
                                          ck_tile::sequence<{warp[0]}, {warp[1]}, {warp[2]}>,
                                          ck_tile::sequence<{wave[3]}, {wave[4]}, {wave[5]}>,
                                          ck_tile::sequence<{warp[3]}, {warp[4]}, {warp[5]}>,
                                          {vlayout_cpp}>;

using fmha_traits = ck_tile::TileFmhaTraits<{_bool_cpp(pad[0])},
                                            {_bool_cpp(pad[1])},
                                            {_bool_cpp(pad[2])},
                                            {_bool_cpp(pad[3])},
                                            {_bool_cpp(sig["logits"])},
                                            {_bias_cpp(sig["bias"])},
                                            false,
                                            {_bool_cpp(sig["lse"])},
                                            {_bool_cpp(sig["dropout"])},
                                            {_qscale_cpp(sig["qscale"])},
                                            {alg["block_per_cu"]},
                                            {_bool_cpp(sig["skip_min_seqlen_q"])},
                                            {_bool_cpp(sig["sink"])}>;

using fmha_variant = ck_tile::ComposedAttention<{_bool_cpp(sig["logits"])} * ck_tile::LOGITS_SOFT_CAP,
                                                CK_TILE_FMHA_FWD_FAST_EXP2>;
using fmha_mask = {MASK_TO_CPP_GENERIC.get(canonical_mask(sig["mask"]), _mask_cpp(sig["mask"])) if pipeline_name in ("v3", "qr_async_trload_v3") else _mask_cpp(sig["mask"])};

using fmha_pipeline_problem = ck_tile::BlockFmhaPipelineProblem<
    typename FmhaFwdTypeConfig<fmha_dtype>::QDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::KDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::VDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::SaccDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::SMPLComputeDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::BiasDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::RandValOutputDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::LSEDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::PDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::OaccDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::ODataType,
    fmha_shape,
    {mode_cpp},
    fmha_variant,
    fmha_mask,
    {use_trload},
    fmha_traits>;

using fmha_pipeline = {pipeline_cpp}<fmha_pipeline_problem>;
using fmha_epilogue = ck_tile::Default2DEpilogue<
    ck_tile::Default2DEpilogueProblem<typename FmhaFwdTypeConfig<fmha_dtype>::OaccDataType,
                                      typename FmhaFwdTypeConfig<fmha_dtype>::ODataType,
                                      {_bool_cpp(pad[0])},
                                      {_bool_cpp(pad[3])}>>;
using fmha_kernel = {"ck_tile::FmhaFwdV3Kernel" if pipeline_name in ("v3", "qr_async_trload_v3") else "ck_tile::FmhaFwdKernel"}<fmha_pipeline, fmha_epilogue>;

using trait = fmha_fwd_traits_<{sig["hdim_q"]},
                               {dtype_cpp},
                               {mode_cpp},
                               {tile[0]}, {tile[1]}, {tile[2]}, {tile[3]}, {tile[4]}, {tile[5]},
                               {vlayout_cpp},
                               {PIPELINE_ENUM_TO_CPP[pipeline_name]},
                               {_bool_cpp(sig["logits"])},
                               fmha_mask,
                               {_bias_cpp(sig["bias"])},
                               {_bool_cpp(sig["lse"])},
                               {_bool_cpp(sig["dropout"])},
                               {_qscale_cpp(sig["qscale"])},
                               {_bool_cpp(pad[0])},
                               {_bool_cpp(pad[1])},
                               {_bool_cpp(pad[2])},
                               {_bool_cpp(pad[3])},
                               {use_trload},
                               {_bool_cpp(sig["skip_min_seqlen_q"])},
                               {_bool_cpp(sig["sink"])}>;
}} // namespace {ns}

template <>
inline float fmha_fwd_<{ns}::trait, {arch_tag}>(const ck_tile::stream_config& s, fmha_fwd_args a)
{{
    using k_ = {ns}::fmha_kernel;
    auto [kargs, grids] = {"fmha_fwd_v3_create_kargs_and_grids" if pipeline_name in ("v3", "qr_async_trload_v3") else "fmha_fwd_create_kargs_and_grids"}<k_>(a);
    const dim3 blocks = k_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
    return ck_tile::launch_kernel(
        s, ck_tile::make_kernel<kBlockPerCu, {arch_tag}>(k_{{}}, grids, blocks, 0, kargs));
}}

namespace {ns} {{
inline float run(const ck_tile::stream_config& s, fmha_fwd_args a)
{{
    return fmha_fwd_<trait, {arch_tag}>(s, a);
}}

inline void launch(const ck_tile::stream_config& s, fmha_fwd_args a)
{{
    auto sc = s;
    sc.time_kernel_ = false;
    (void)fmha_fwd_<trait, {arch_tag}>(sc, a);
}}

}} // namespace {ns}

#endif // arch guard
"""


def _pagedkv_kernel_body(name: str, config: dict) -> str:
    sig = config["signature"]
    alg = config["algorithm"]
    arch_tag = ARCH_TAG_MAP[config["arch"]]
    arch_check = ARCH_PREPROC_MAP.get(config["arch"], "1")
    dtype_cpp = FWD_DTYPE_MAP[sig["data_type"]]
    mode_cpp = "true" if sig["mode"] == "group" else "false"
    vlayout_cpp = LAYOUT_TO_BOOL[sig["vlayout"]]
    tile = alg["tile"]
    wave = alg["wave"]
    warp = alg["warp"]
    pad = alg["padding"]
    ns = f"ns_{name}"
    return f"""// SPDX-License-Identifier: MIT
#pragma once

#include "example/ck_tile/01_fmha/fmha_fwd.hpp"

#if !defined(__HIP_DEVICE_COMPILE__) || ({arch_check})

namespace {ns} {{

using fmha_dtype = {dtype_cpp};
using fmha_block_tile = ck_tile::sequence<{tile[0]}, {tile[1]}, {tile[2]}, {tile[3]}, {tile[4]}, {tile[5]}>;
using fmha_shape = ck_tile::TileFmhaShape<fmha_block_tile,
                                          ck_tile::sequence<{wave[0]}, {wave[1]}, {wave[2]}>,
                                          ck_tile::sequence<{warp[0]}, {warp[1]}, {warp[2]}>,
                                          ck_tile::sequence<{wave[3]}, {wave[4]}, {wave[5]}>,
                                          ck_tile::sequence<{warp[3]}, {warp[4]}, {warp[5]}>,
                                          {vlayout_cpp}>;

using fmha_trait = ck_tile::TileFmhaFwdPagedKVTraits<{_bool_cpp(pad[0])},
                                                     {_bool_cpp(pad[1])},
                                                     {_bool_cpp(pad[2])},
                                                     {_bool_cpp(pad[3])},
                                                     {_bool_cpp(sig["logits"])},
                                                     {_bias_cpp(sig["bias"])},
                                                     false,
                                                     {_bool_cpp(sig["lse"])},
                                                     {_bool_cpp(sig["paged_kv"])},
                                                     {_bool_cpp(sig["fp8_static_quant"])},
                                                     {alg["block_per_cu"]},
                                                     {_bool_cpp(sig["skip_min_seqlen_q"])},
                                                     {_bool_cpp(sig["sink"])}>;

using fmha_variant = ck_tile::ComposedAttention<{_bool_cpp(sig["logits"])} * ck_tile::LOGITS_SOFT_CAP,
                                                CK_TILE_FMHA_FWD_FAST_EXP2>;
using fmha_mask = {_mask_cpp(sig["mask"])};

using fmha_pipeline_problem = ck_tile::BlockFmhaFwdPagedKVPipelineProblem<
    typename FmhaFwdTypeConfig<fmha_dtype>::QDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::KDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::VDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::SaccDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::SMPLComputeDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::BiasDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::LSEDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::PDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::OaccDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::ODataType,
    fmha_shape,
    {mode_cpp},
    fmha_variant,
    fmha_mask,
    fmha_trait>;

using fmha_pipeline = ck_tile::BlockFmhaFwdPagedKVPipelineQRKSVS<fmha_pipeline_problem>;
using fmha_epilogue = ck_tile::Default2DEpilogue<
    ck_tile::Default2DEpilogueProblem<typename FmhaFwdTypeConfig<fmha_dtype>::OaccDataType,
                                      typename FmhaFwdTypeConfig<fmha_dtype>::ODataType,
                                      {_bool_cpp(pad[0])},
                                      {_bool_cpp(pad[3])}>>;
using fmha_kernel = ck_tile::FmhaFwdPagedKVKernel<fmha_pipeline, fmha_epilogue>;

using trait = fmha_fwd_pagedkv_traits_<{sig["hdim_q"]},
                                       {dtype_cpp},
                                       {mode_cpp},
                                       {tile[0]}, {tile[1]}, {tile[2]}, {tile[3]}, {tile[4]}, {tile[5]},
                                       {vlayout_cpp},
                                       {PIPELINE_ENUM_TO_CPP["qr_pagedkv"]},
                                       {_bool_cpp(sig["logits"])},
                                       fmha_mask,
                                       {_bias_cpp(sig["bias"])},
                                       {_bool_cpp(sig["lse"])},
                                       {_bool_cpp(sig["paged_kv"])},
                                       {_bool_cpp(sig["fp8_static_quant"])},
                                       {_bool_cpp(pad[0])},
                                       {_bool_cpp(pad[1])},
                                       {_bool_cpp(pad[2])},
                                       {_bool_cpp(pad[3])},
                                       {_bool_cpp(sig["skip_min_seqlen_q"])},
                                       {_bool_cpp(sig["sink"])}>;
}} // namespace {ns}

template <>
inline float fmha_fwd_pagedkv_<{ns}::trait, {arch_tag}>(const ck_tile::stream_config& s,
                                                        fmha_fwd_pagedkv_args a)
{{
    using k_ = {ns}::fmha_kernel;
    auto [kargs, grids] = fmha_fwd_pagedkv_create_kargs_and_grids<k_>(a);
    const dim3 blocks = k_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
    return ck_tile::launch_kernel(
        s, ck_tile::make_kernel<kBlockPerCu, {arch_tag}>(k_{{}}, grids, blocks, 0, kargs));
}}

namespace {ns} {{
inline float run(const ck_tile::stream_config& s, fmha_fwd_pagedkv_args a)
{{
    return fmha_fwd_pagedkv_<trait, {arch_tag}>(s, a);
}}

inline void launch(const ck_tile::stream_config& s, fmha_fwd_pagedkv_args a)
{{
    auto sc = s;
    sc.time_kernel_ = false;
    (void)fmha_fwd_pagedkv_<trait, {arch_tag}>(sc, a);
}}

}} // namespace {ns}

#endif // arch guard
"""


def _splitkv_kernel_body(name: str, config: dict) -> str:
    sig = config["signature"]
    alg = config["algorithm"]
    arch_tag = ARCH_TAG_MAP[config["arch"]]
    arch_check = ARCH_PREPROC_MAP.get(config["arch"], "1")
    dtype_cpp = FWD_DTYPE_MAP[sig["data_type"]]
    mode_cpp = "true" if sig["mode"] == "group" else "false"
    vlayout_cpp = LAYOUT_TO_BOOL[sig["vlayout"]]
    tile = alg["tile"]
    wave = alg["wave"]
    warp = alg["warp"]
    pad = alg["padding"]
    pipeline_cpp = {
        "qr": "ck_tile::BlockFmhaFwdSplitKVPipelineQRKSVS",
        "qr_nwarp_sshuffle": "ck_tile::BlockFmhaFwdSplitKVPipelineNWarpSShuffleQRKSVS",
    }[alg["pipeline"]]
    ns = f"ns_{name}"
    return f"""// SPDX-License-Identifier: MIT
#pragma once

#include "example/ck_tile/01_fmha/fmha_fwd.hpp"

#if !defined(__HIP_DEVICE_COMPILE__) || ({arch_check})

namespace {ns} {{

using fmha_dtype = {dtype_cpp};
using fmha_variant = ck_tile::ComposedAttention<{_bool_cpp(sig["logits"])} * ck_tile::LOGITS_SOFT_CAP,
                                                CK_TILE_FMHA_FWD_FAST_EXP2>;
using fmha_mask = {_mask_cpp(sig["mask"])};
using fmha_block_tile = ck_tile::sequence<{tile[0]}, {tile[1]}, {tile[2]}, {tile[3]}, {tile[4]}, {tile[5]}>;
using fmha_shape = ck_tile::TileFmhaShape<fmha_block_tile,
                                          ck_tile::sequence<{wave[0]}, {wave[1]}, {wave[2]}>,
                                          ck_tile::sequence<{warp[0]}, {warp[1]}, {warp[2]}>,
                                          ck_tile::sequence<{wave[3]}, {wave[4]}, {wave[5]}>,
                                          ck_tile::sequence<{warp[3]}, {warp[4]}, {warp[5]}>,
                                          {vlayout_cpp}>;
using fmha_trait = ck_tile::TileFmhaFwdSplitKVTraits<{_bool_cpp(pad[0])},
                                                     {_bool_cpp(pad[1])},
                                                     {_bool_cpp(pad[2])},
                                                     {_bool_cpp(pad[3])},
                                                     {_bool_cpp(sig["logits"])},
                                                     {_bias_cpp(sig["bias"])},
                                                     false,
                                                     {_bool_cpp(sig["lse"])},
                                                     {_bool_cpp(sig["fp8_static_quant"])},
                                                     {_bool_cpp(sig["paged_kv"])},
                                                     true,
                                                     false,
                                                     {alg["block_per_cu"]},
                                                     {_bool_cpp(sig["sink"])}>;
using fmha_pipeline_problem = ck_tile::BlockFmhaFwdSplitKVPipelineProblem<
    typename FmhaFwdTypeConfig<fmha_dtype>::QDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::KDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::VDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::SaccDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::SMPLComputeDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::BiasDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::LSEDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::PDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::OaccDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::OaccDataType,
    fmha_shape,
    {mode_cpp},
    fmha_variant,
    fmha_mask,
    fmha_trait>;
using fmha_pipeline = {pipeline_cpp}<fmha_pipeline_problem>;
using fmha_epilogue = ck_tile::Default2DEpilogue<
    ck_tile::Default2DEpilogueProblem<typename FmhaFwdTypeConfig<fmha_dtype>::OaccDataType,
                                      typename FmhaFwdTypeConfig<fmha_dtype>::OaccDataType,
                                      false,
                                      false>>;
using fmha_kernel = ck_tile::FmhaFwdSplitKVKernel<fmha_pipeline, fmha_epilogue>;

using trait = fmha_fwd_splitkv_traits_<{sig["hdim_q"]},
                                       {dtype_cpp},
                                       {mode_cpp},
                                       {tile[0]}, {tile[1]}, {tile[2]}, {tile[3]}, {tile[4]}, {tile[5]},
                                       {vlayout_cpp},
                                       {PIPELINE_ENUM_TO_CPP[alg["pipeline"]]},
                                       {_bool_cpp(sig["logits"])},
                                       fmha_mask,
                                       {_bias_cpp(sig["bias"])},
                                       {_bool_cpp(sig["lse"])},
                                       {_bool_cpp(sig["fp8_static_quant"])},
                                       {_bool_cpp(sig["paged_kv"])},
                                       {_bool_cpp(sig["sink"])},
                                       {_bool_cpp(pad[0])},
                                       {_bool_cpp(pad[1])},
                                       {_bool_cpp(pad[2])},
                                       {_bool_cpp(pad[3])}>;
}} // namespace {ns}

template <>
inline void fmha_fwd_splitkv_oneshot_<{ns}::trait, {arch_tag}>(const ck_tile::stream_config& s,
                                                               fmha_fwd_splitkv_args a)
{{
    using k_ = {ns}::fmha_kernel;
    auto [kargs, grids] = fmha_fwd_splitkv_create_kargs_and_grids<k_>(a);
    const dim3 blocks = k_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
    ck_tile::make_kernel<kBlockPerCu, {arch_tag}>(k_{{}}, grids, blocks, 0, kargs)(
        ck_tile::stream_config{{s.stream_id_}});
}}

namespace {ns} {{
inline void launch(const ck_tile::stream_config& s, fmha_fwd_splitkv_args a)
{{
    fmha_fwd_splitkv_oneshot_<trait, {arch_tag}>(s, a);
}}

}} // namespace {ns}

#endif // arch guard
"""


def _splitkv_combine_kernel_body(name: str, config: dict) -> str:
    sig = config["signature"]
    alg = config["algorithm"]
    arch_tag = ARCH_TAG_MAP[config["arch"]]
    arch_check = ARCH_PREPROC_MAP.get(config["arch"], "1")
    dtype_cpp = FWD_DTYPE_MAP[sig["data_type"]]
    mode_cpp = "true" if sig["mode"] == "group" else "false"
    tile = alg["tile"]
    pad = alg["padding"]
    ns = f"ns_{name}"
    return f"""// SPDX-License-Identifier: MIT
#pragma once

#include "example/ck_tile/01_fmha/fmha_fwd.hpp"

#if !defined(__HIP_DEVICE_COMPILE__) || ({arch_check})

using fmha_dtype = {dtype_cpp};
namespace {{
template <ck_tile::index_t kLogMaxSplits>
struct {ns}_instance {{
using fmha_trait = ck_tile::TileFmhaFwdSplitKVCombineTraits<{_bool_cpp(pad[0])},
                                                            {_bool_cpp(pad[3])},
                                                            {_bool_cpp(sig["lse"])},
                                                            {_bool_cpp(sig["fp8_static_quant"])},
                                                            kLogMaxSplits,
                                                            {alg["block_per_cu"]}>;

using fmha_pipeline_problem = ck_tile::BlockFmhaSplitKVCombinePipelineProblem<
    typename FmhaFwdTypeConfig<fmha_dtype>::LSEDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::OaccDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::ODataType,
    {sig["hdim_v"]},
    {mode_cpp},
    {tile[3]},
    fmha_trait>;

using fmha_pipeline = ck_tile::BlockFmhaFwdSplitKVCombinePipeline<fmha_pipeline_problem>;
using fmha_epilogue = ck_tile::Default2DEpilogue<
    ck_tile::Default2DEpilogueProblem<typename FmhaFwdTypeConfig<fmha_dtype>::OaccDataType,
                                      typename FmhaFwdTypeConfig<fmha_dtype>::ODataType,
                                      false,
                                      false>>;
using fmha_kernel = ck_tile::FmhaFwdSplitKVCombineKernel<fmha_pipeline, fmha_epilogue>;

static void run(const ck_tile::stream_config& s, fmha_fwd_splitkv_args a)
{{
    using k_ = fmha_kernel;
    auto [kargs, grids] = fmha_fwd_splitkv_combine_create_kargs_and_grids<k_>(a);
    const dim3 blocks = k_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
    ck_tile::make_kernel<kBlockPerCu, {arch_tag}>(k_{{}}, grids, blocks, 0, kargs)(
        ck_tile::stream_config{{s.stream_id_}});
}}
}}; // struct {ns}_instance
}} // anonymous namespace

namespace {ns} {{
using trait = fmha_fwd_splitkv_combine_traits_<{sig["hdim_v"]},
                                               {dtype_cpp},
                                               {mode_cpp},
                                               {tile[3]},
                                               {_bool_cpp(sig["lse"])},
                                               {_bool_cpp(sig["fp8_static_quant"])},
                                               {_bool_cpp(pad[0])},
                                               {_bool_cpp(pad[3])}>;
}} // namespace {ns}

template <>
inline void fmha_fwd_splitkv_combine_oneshot_<{ns}::trait, {arch_tag}>(
    const ck_tile::stream_config& s, fmha_fwd_splitkv_args a)
{{
    if (a.num_splits <= 8) {{
        {ns}_instance<3>::run(s, a);
    }} else if (a.num_splits <= 16) {{
        {ns}_instance<4>::run(s, a);
    }} else if (a.num_splits <= 32) {{
        {ns}_instance<5>::run(s, a);
    }} else if (a.num_splits <= 64) {{
        {ns}_instance<6>::run(s, a);
    }} else if (a.num_splits <= 128) {{
        {ns}_instance<7>::run(s, a);
    }}
}}

namespace {ns} {{
inline void launch(const ck_tile::stream_config& s, fmha_fwd_splitkv_args a)
{{
    fmha_fwd_splitkv_combine_oneshot_<trait, {arch_tag}>(s, a);
}}

}} // namespace {ns}

#endif // arch guard
"""


def _appendkv_kernel_body(name: str, config: dict) -> str:
    sig = config["signature"]
    alg = config["algorithm"]
    arch_tag = ARCH_TAG_MAP[config["arch"]]
    arch_check = ARCH_PREPROC_MAP.get(config["arch"], "1")
    dtype_cpp = FWD_DTYPE_MAP[sig["data_type"]]
    vlayout_cpp = LAYOUT_TO_BOOL[sig["vlayout"]]
    tile = alg["tile"]
    pad = alg["padding"]
    ns = f"ns_{name}"
    return f"""// SPDX-License-Identifier: MIT
#pragma once

#include "example/ck_tile/01_fmha/fmha_fwd.hpp"

#if !defined(__HIP_DEVICE_COMPILE__) || ({arch_check})

namespace {ns} {{

using fmha_dtype = {dtype_cpp};
using fmha_trait = ck_tile::TileFmhaFwdAppendKVTraits<{_bool_cpp(pad[0])},
                                                      {_bool_cpp(pad[1])},
                                                      {_bool_cpp(pad[2])},
                                                      {_bool_cpp(pad[3])},
                                                      {alg["block_per_cu"]}>;
using fmha_pipeline_problem = ck_tile::BlockFmhaFwdAppendKVPipelineProblem<
    typename FmhaFwdTypeConfig<fmha_dtype>::QDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::KDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::VDataType,
    {tile[0]},
    {tile[1]},
    {tile[2]},
    {tile[3]},
    {vlayout_cpp},
    {_rope_cpp(sig["rope"])},
    {_bool_cpp(sig["paged_kv"])},
    fmha_trait>;
using fmha_pipeline = ck_tile::BlockFmhaFwdAppendKVPipeline<fmha_pipeline_problem>;
using fmha_kernel = ck_tile::FmhaFwdAppendKVKernel<fmha_pipeline>;

using trait = fmha_fwd_appendkv_traits_<{sig["hdim_q"]},
                                        {dtype_cpp},
                                        {tile[0]},
                                        {tile[1]},
                                        {tile[2]},
                                        {tile[3]},
                                        {vlayout_cpp},
                                        {_bool_cpp(pad[0])},
                                        {_bool_cpp(pad[1])},
                                        {_bool_cpp(pad[2])},
                                        {_bool_cpp(pad[3])},
                                        {_rope_cpp(sig["rope"])},
                                        {_bool_cpp(sig["paged_kv"])}>;
}} // namespace {ns}

template <>
inline float fmha_fwd_appendkv_<{ns}::trait, {arch_tag}>(const ck_tile::stream_config& s,
                                                         fmha_fwd_appendkv_args a)
{{
    using k_ = {ns}::fmha_kernel;
    auto [kargs, grids] = fmha_fwd_appendkv_create_kargs_and_grids<k_>(a);
    const dim3 blocks = k_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
    return ck_tile::launch_kernel(
        s, ck_tile::make_kernel<kBlockPerCu, {arch_tag}>(k_{{}}, grids, blocks, 0, kargs));
}}

namespace {ns} {{
inline float run(const ck_tile::stream_config& s, fmha_fwd_appendkv_args a)
{{
    return fmha_fwd_appendkv_<trait, {arch_tag}>(s, a);
}}

inline void launch(const ck_tile::stream_config& s, fmha_fwd_appendkv_args a)
{{
    auto sc = s;
    sc.time_kernel_ = false;
    (void)fmha_fwd_appendkv_<trait, {arch_tag}>(sc, a);
}}

}} // namespace {ns}

#endif // arch guard
"""


def _batch_prefill_kernel_body(name: str, config: dict) -> str:
    sig = config["signature"]
    alg = config["algorithm"]
    arch_check = ARCH_PREPROC_MAP.get(config["arch"], "1")
    dtype_cpp = FWD_DTYPE_MAP[sig["data_type"]]
    mode_cpp = "true" if sig["mode"] == "group" else "false"
    vlayout_cpp = LAYOUT_TO_BOOL[sig["vlayout"]]
    tile = alg["tile"]
    wave = alg["wave"]
    warp = alg["warp"]
    pad = alg["padding"]
    ns = f"ns_{name}"
    return f"""// SPDX-License-Identifier: MIT
#pragma once

#include "example/ck_tile/01_fmha/fmha_fwd.hpp"

#if !defined(__HIP_DEVICE_COMPILE__) || ({arch_check})

namespace {ns} {{

using fmha_dtype = {dtype_cpp};
using fmha_block_tile = ck_tile::sequence<{tile[0]}, {tile[1]}, {tile[2]}, {tile[3]}, {tile[4]}, {tile[5]}>;
using fmha_shape = ck_tile::TileFmhaShape<fmha_block_tile,
                                          ck_tile::sequence<{wave[0]}, {wave[1]}, {wave[2]}>,
                                          ck_tile::sequence<{warp[0]}, {warp[1]}, {warp[2]}>,
                                          ck_tile::sequence<{wave[3]}, {wave[4]}, {wave[5]}>,
                                          ck_tile::sequence<{warp[3]}, {warp[4]}, {warp[5]}>,
                                          {vlayout_cpp}>;
using fmha_trait = ck_tile::TileFmhaBatchPrefillTraits<{_bool_cpp(pad[0])},
                                                       {_bool_cpp(pad[1])},
                                                       {_bool_cpp(pad[2])},
                                                       {_bool_cpp(pad[3])},
                                                       {_bool_cpp(sig["logits"])},
                                                       {_bias_cpp(sig["bias"])},
                                                       false,
                                                       {_bool_cpp(sig["lse"])},
                                                       {_bool_cpp(sig["dropout"])},
                                                       {_qscale_cpp(sig["qscale"])},
                                                       {alg["block_per_cu"]},
                                                       false,
                                                       {sig["page_size"]},
                                                       {_kv_memory_cpp(sig["kv_memory_layout"])},
                                                       {_kv_lookup_cpp(sig["kv_lookup_table"])}>;
using fmha_variant = ck_tile::ComposedAttention<{_bool_cpp(sig["logits"])} * ck_tile::LOGITS_SOFT_CAP,
                                                CK_TILE_FMHA_FWD_FAST_EXP2>;
using fmha_mask = {_mask_cpp(sig["mask"])};
using fmha_pipeline_problem = ck_tile::BlockFmhaBatchPrefillPipelineProblem<
    typename FmhaFwdTypeConfig<fmha_dtype>::QDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::KDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::VDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::SaccDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::SMPLComputeDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::BiasDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::RandValOutputDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::LSEDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::PDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::OaccDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::ODataType,
    fmha_shape,
    {mode_cpp},
    fmha_variant,
    fmha_mask,
    false,
    {sig["page_size"]},
    fmha_trait>;
using fmha_pipeline = ck_tile::BlockFmhaBatchPrefillPipelineQRKSVSAsync<fmha_pipeline_problem>;
using fmha_epilogue = ck_tile::Default2DEpilogue<
    ck_tile::Default2DEpilogueProblem<typename FmhaFwdTypeConfig<fmha_dtype>::OaccDataType,
                                      typename FmhaFwdTypeConfig<fmha_dtype>::ODataType,
                                      {_bool_cpp(pad[0])},
                                      {_bool_cpp(pad[3])}>>;
using fmha_kernel = ck_tile::FmhaBatchPrefillWithPagedKVCacheKernel<fmha_pipeline, fmha_epilogue>;

using trait = fmha_fwd_batch_prefill_traits_<{sig["hdim_q"]},
                                             {dtype_cpp},
                                             {mode_cpp},
                                             {tile[0]}, {tile[1]}, {tile[2]}, {tile[3]}, {tile[4]}, {tile[5]},
                                             {vlayout_cpp},
                                             {PIPELINE_ENUM_TO_CPP["batch_prefill_async"]},
                                             {_bool_cpp(sig["logits"])},
                                             fmha_mask,
                                             {_bias_cpp(sig["bias"])},
                                             {_bool_cpp(sig["lse"])},
                                             {_bool_cpp(sig["dropout"])},
                                             {_qscale_cpp(sig["qscale"])},
                                             {_bool_cpp(pad[0])},
                                             {_bool_cpp(pad[1])},
                                             {_bool_cpp(pad[2])},
                                             {_bool_cpp(pad[3])},
                                             false,
                                             false,
                                             {sig["page_size"]},
                                             {_kv_memory_cpp(sig["kv_memory_layout"])},
                                             {_kv_lookup_cpp(sig["kv_lookup_table"])}>;
}} // namespace {ns}

template <>
inline float fmha_batch_prefill_<{ns}::trait>(const ck_tile::stream_config& s, fmha_batch_prefill_args a)
{{
    using k_ = {ns}::fmha_kernel;
    auto [kargs, grids] = fmha_batch_prefill_create_kargs_and_grids<k_>(a);
    const dim3 blocks = k_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
    return ck_tile::launch_kernel(s, ck_tile::make_kernel<kBlockPerCu>(k_{{}}, grids, blocks, 0, kargs));
}}

namespace {ns} {{
inline float run(const ck_tile::stream_config& s, fmha_batch_prefill_args a)
{{
    return fmha_batch_prefill_<trait>(s, a);
}}

inline void launch(const ck_tile::stream_config& s, fmha_batch_prefill_args a)
{{
    auto sc = s;
    sc.time_kernel_ = false;
    (void)fmha_batch_prefill_<trait>(sc, a);
}}

}} // namespace {ns}

#endif // arch guard
"""


def _bwd_dot_do_o_kernel_body(name: str, config: dict) -> str:
    sig = config["signature"]
    alg = config["algorithm"]
    arch_tag = ARCH_TAG_MAP[config["arch"]]
    arch_check = ARCH_PREPROC_MAP.get(config["arch"], "1")
    dtype_cpp = BWD_DTYPE_MAP[sig["data_type"]]
    mode_cpp = "true" if sig["mode"] == "group" else "false"
    tile = alg["tile"]
    pad = alg["padding"]
    ns = f"ns_{name}"
    return f"""// SPDX-License-Identifier: MIT
#pragma once

#include "example/ck_tile/01_fmha/fmha_bwd.hpp"

#if !defined(__HIP_DEVICE_COMPILE__) || ({arch_check})

namespace {ns} {{

using fmha_dtype = {dtype_cpp};
using fmha_trait = ck_tile::TileFmhaBwdOGradDotOTraits<{_bool_cpp(pad[0])},
                                                       {_bool_cpp(pad[3])},
                                                       {alg["block_per_cu"]}>;
using fmha_pipeline_problem = ck_tile::BlockFmhaBwdOGradDotOPipelineProblem<
    typename FmhaBwdTypeConfig<fmha_dtype>::ODataType,
    typename FmhaBwdTypeConfig<fmha_dtype>::OGradDataType,
    typename FmhaBwdTypeConfig<fmha_dtype>::DDataType,
    {tile[0]},
    {sig["hdim_v"]},
    {mode_cpp},
    fmha_trait>;
using fmha_pipeline = typename ck_tile::BlockFmhaBwdOGradDotO<fmha_pipeline_problem>;
using fmha_kernel = ck_tile::FmhaBwdOGradDotOKernel<fmha_pipeline>;

using trait = fmha_bwd_dot_do_o_traits_<{sig["hdim_v"]},
                                        {dtype_cpp},
                                        {mode_cpp},
                                        {_bool_cpp(pad[0])},
                                        {_bool_cpp(pad[3])}>;
}} // namespace {ns}

template <>
inline void fmha_bwd_dot_do_o_oneshot_<{ns}::trait, {arch_tag}>(const ck_tile::stream_config& s,
                                                                fmha_bwd_args a)
{{
    using k_ = {ns}::fmha_kernel;
    auto [kargs, grids] = fmha_bwd_dot_do_o_create_kargs_and_grids<k_>(a);
    const dim3 blocks = k_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
    ck_tile::make_kernel<kBlockPerCu, {arch_tag}>(k_{{}}, grids, blocks, 0, kargs)(
        ck_tile::stream_config{{s.stream_id_}});
}}

namespace {ns} {{
inline void launch(const ck_tile::stream_config& s, fmha_bwd_args a)
{{
    fmha_bwd_dot_do_o_oneshot_<trait, {arch_tag}>(s, a);
}}

}} // namespace {ns}

#endif // arch guard
"""


def _bwd_dq_dk_dv_kernel_body(name: str, config: dict) -> str:
    sig = config["signature"]
    alg = config["algorithm"]
    arch_tag = ARCH_TAG_MAP[config["arch"]]
    arch_check = ARCH_PREPROC_MAP.get(config["arch"], "1")
    dtype_cpp = BWD_DTYPE_MAP[sig["data_type"]]
    mode_cpp = "true" if sig["mode"] == "group" else "false"
    tile = alg["tile"]
    wave = alg["wave"]
    warp = alg["warp"]
    pad = alg["padding"]
    ns = f"ns_{name}"
    # BlockDropoutBwd<kHasDropout, kIsWG32, kIsStoreRandval>
    # wg16 variants use kIsWG32=false; wg32 variants use kIsWG32=true
    dropout_variant = sig.get("dropout_variant", "")
    is_wg32 = "wg32" in dropout_variant if dropout_variant else True
    is_store = "storerandval" in dropout_variant if dropout_variant else False
    has_dropout = bool(sig["dropout"])
    dropout_cpp = (
        f"ck_tile::BlockDropoutBwd<{_bool_cpp(has_dropout)}, "
        f"{_bool_cpp(is_wg32 if has_dropout else True)}, "
        f"{_bool_cpp(is_store)}>"
    )
    return f"""// SPDX-License-Identifier: MIT
#pragma once

#include "example/ck_tile/01_fmha/fmha_bwd.hpp"

#if !defined(__HIP_DEVICE_COMPILE__) || ({arch_check})

namespace {ns} {{

using fmha_dtype = {dtype_cpp};
using fmha_block_tile = ck_tile::sequence<{_bwd_block_tile(tile, sig)}>;
using fmha_block_warps0 = ck_tile::sequence<{wave[0]}, {wave[1]}, {wave[2]}>;
using fmha_block_warps1 = ck_tile::sequence<{wave[3]}, {wave[4]}, {wave[5]}>;
using fmha_block_warps2 = ck_tile::sequence<{wave[6]}, {wave[7]}, {wave[8]}>;
using fmha_warp_tile0 = ck_tile::sequence<{warp[0]}, {warp[1]}, {warp[2]}>;
using fmha_warp_tile1 = ck_tile::sequence<{warp[3]}, {warp[4]}, {warp[5]}>;
using fmha_warp_tile2 = ck_tile::sequence<{warp[0]}, {warp[1]}, ck_tile::min({warp[2]}, {tile[6] if len(tile) >= 7 else warp[2]})>;
using fmha_shape = ck_tile::TileFmhaBwdShape<fmha_block_tile,
                                             fmha_block_warps0,
                                             fmha_warp_tile0,
                                             fmha_block_warps1,
                                             fmha_warp_tile1,
                                             fmha_block_warps0,
                                             fmha_warp_tile0,
                                             fmha_block_warps1,
                                             fmha_warp_tile1,
                                             fmha_block_warps2,
                                             fmha_warp_tile2,
                                             {alg["max_seq_len_q"]}>;
using fmha_trait = ck_tile::TileFmhaBwdTraits<{int(pad[2])},
                                              {int(pad[3])},
                                              {_bias_cpp(sig["bias"])},
                                              {_bool_cpp(sig["dbias"])},
                                              {alg["block_per_cu"]}>;
using fmha_mask = {_mask_cpp(sig["mask"])};
using fmha_dropout = {dropout_cpp};
using fmha_problem = ck_tile::BlockFmhaBwdPipelineProblem<
    typename FmhaBwdTypeConfig<fmha_dtype>::QDataType,
    typename FmhaBwdTypeConfig<fmha_dtype>::KDataType,
    typename FmhaBwdTypeConfig<fmha_dtype>::VDataType,
    typename FmhaBwdTypeConfig<fmha_dtype>::GemmDataType,
    typename FmhaBwdTypeConfig<fmha_dtype>::LSEDataType,
    typename FmhaBwdTypeConfig<fmha_dtype>::AccDataType,
    typename FmhaBwdTypeConfig<fmha_dtype>::DDataType,
    typename FmhaBwdTypeConfig<fmha_dtype>::BiasDataType,
    typename FmhaBwdTypeConfig<fmha_dtype>::RandValOutputDataType,
    typename FmhaBwdTypeConfig<fmha_dtype>::ODataType,
    typename FmhaBwdTypeConfig<fmha_dtype>::OGradDataType,
    typename FmhaBwdTypeConfig<fmha_dtype>::QGradDataType,
    typename FmhaBwdTypeConfig<fmha_dtype>::KGradDataType,
    typename FmhaBwdTypeConfig<fmha_dtype>::VGradDataType,
    typename FmhaBwdTypeConfig<fmha_dtype>::BiasGradDataType,
    fmha_shape,
    {mode_cpp},
    {_bool_cpp(sig["deterministic"])},
    fmha_mask,
    fmha_dropout,
    {_bool_cpp(alg["use_trload"])},
    fmha_trait>;
using fmha_pipeline = ck_tile::BlockFmhaBwdDQDKDVPipeline<fmha_problem>;
using dk_epi = ck_tile::Default2DEpilogue<
    ck_tile::Default2DEpilogueProblem<typename FmhaBwdTypeConfig<fmha_dtype>::AccDataType,
                                      typename FmhaBwdTypeConfig<fmha_dtype>::KGradDataType,
                                      false,
                                      ({int(pad[2])} > 0)>>;
using dv_epi = ck_tile::Default2DEpilogue<
    ck_tile::Default2DEpilogueProblem<typename FmhaBwdTypeConfig<fmha_dtype>::AccDataType,
                                      typename FmhaBwdTypeConfig<fmha_dtype>::VGradDataType,
                                      false,
                                      ({int(pad[3])} > 0)>>;
using dq_epi = ck_tile::Default2DEpilogue<
    ck_tile::Default2DEpilogueProblem<typename FmhaBwdTypeConfig<fmha_dtype>::AccDataType,
                                      typename FmhaBwdTypeConfig<fmha_dtype>::QGradDataType,
                                      false,
                                      ({int(pad[2])} > 0)>>;
using fmha_kernel = ck_tile::FmhaBwdDQDKDVKernel<fmha_pipeline, dk_epi, dv_epi, dq_epi>;

using trait = fmha_bwd_dq_dk_dv_traits_<{sig["hdim_q"]},
                                        {dtype_cpp},
                                        {mode_cpp},
                                        fmha_mask,
                                        fmha_dropout,
                                        {_bias_cpp(sig["bias"])},
                                        {_bool_cpp(sig["dbias"])},
                                        {int(pad[2])},
                                        {int(pad[3])},
                                        {_bool_cpp(sig["deterministic"])},
                                        {_bool_cpp(alg["use_trload"])},
                                        {alg["max_seq_len_q"]},
                                        {tile[1]}>;
}} // namespace {ns}

template <>
inline void fmha_bwd_dq_dk_dv_oneshot_<{ns}::trait, {arch_tag}>(
    const ck_tile::stream_config& s, fmha_bwd_args a)
{{
    using k_ = {ns}::fmha_kernel;
    auto [kargs, grids] = fmha_bwd_dq_dk_dv_create_kargs_and_grids<k_>(a);
    const dim3 blocks = k_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
    ck_tile::make_kernel<kBlockPerCu, {arch_tag}>(k_{{}}, grids, blocks, 0, kargs)(
        ck_tile::stream_config{{s.stream_id_}});
}}

namespace {ns} {{
inline void launch(const ck_tile::stream_config& s, fmha_bwd_args a)
{{
    fmha_bwd_dq_dk_dv_oneshot_<trait, {arch_tag}>(s, a);
}}

}} // namespace {ns}

#endif // arch guard
"""


def _bwd_convert_dq_kernel_body(name: str, config: dict) -> str:
    sig = config["signature"]
    alg = config["algorithm"]
    arch_tag = ARCH_TAG_MAP[config["arch"]]
    arch_check = ARCH_PREPROC_MAP.get(config["arch"], "1")
    dtype_cpp = BWD_DTYPE_MAP[sig["data_type"]]
    mode_cpp = "true" if sig["mode"] == "group" else "false"
    tile = alg["tile"]
    pad = alg["padding"]
    ns = f"ns_{name}"
    return f"""// SPDX-License-Identifier: MIT
#pragma once

#include "example/ck_tile/01_fmha/fmha_bwd.hpp"

#if !defined(__HIP_DEVICE_COMPILE__) || ({arch_check})

namespace {ns} {{

using fmha_dtype = {dtype_cpp};
using fmha_trait = ck_tile::TileFmhaBwdConvertQGradTraits<{_bool_cpp(pad[0])},
                                                          {_bool_cpp(pad[2])},
                                                          {alg["block_per_cu"]}>;
using fmha_problem = ck_tile::BlockFmhaBwdConvertQGradPipelineProblem<
    typename FmhaBwdTypeConfig<fmha_dtype>::AccDataType,
    typename FmhaBwdTypeConfig<fmha_dtype>::QGradDataType,
    256,
    {tile[0]},
    {tile[1]},
    {sig["hdim_q"]},
    {mode_cpp},
    {_bool_cpp(sig["deterministic"])},
    fmha_trait>;
using fmha_pipeline = typename ck_tile::BlockFmhaBwdConvertQGrad<fmha_problem>;
using fmha_kernel = ck_tile::FmhaBwdConvertQGradKernel<fmha_pipeline>;

using trait = fmha_bwd_convert_dq_traits_<{sig["hdim_q"]},
                                          {dtype_cpp},
                                          {mode_cpp},
                                          {_bool_cpp(pad[0])},
                                          {_bool_cpp(pad[2])},
                                          {_bool_cpp(sig["deterministic"])},
                                          {tile[1]}>;
}} // namespace {ns}

template <>
inline void fmha_bwd_convert_dq_oneshot_<{ns}::trait, {arch_tag}>(
    const ck_tile::stream_config& s, fmha_bwd_args a)
{{
    using k_ = {ns}::fmha_kernel;
    auto [kargs, grids] = fmha_bwd_convert_dq_create_kargs_and_grids<k_>(a);
    const dim3 blocks = k_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
    ck_tile::make_kernel<kBlockPerCu, {arch_tag}>(k_{{}}, grids, blocks, 0, kargs)(
        ck_tile::stream_config{{s.stream_id_}});
}}

namespace {ns} {{
inline void launch(const ck_tile::stream_config& s, fmha_bwd_args a)
{{
    fmha_bwd_convert_dq_oneshot_<trait, {arch_tag}>(s, a);
}}

}} // namespace {ns}

#endif // arch guard
"""


def render_kernel_header(name: str, config: dict) -> str:
    family = config["signature"]["family"]
    if family == "fwd":
        return _fwd_kernel_body(name, config)
    if family == "fwd_pagedkv":
        return _pagedkv_kernel_body(name, config)
    if family == "fwd_splitkv":
        return _splitkv_kernel_body(name, config)
    if family == "fwd_splitkv_combine":
        return _splitkv_combine_kernel_body(name, config)
    if family == "fwd_appendkv":
        return _appendkv_kernel_body(name, config)
    if family == "batch_prefill":
        return _batch_prefill_kernel_body(name, config)
    if family == "bwd_dot_do_o":
        return _bwd_dot_do_o_kernel_body(name, config)
    if family == "bwd_dq_dk_dv":
        return _bwd_dq_dk_dv_kernel_body(name, config)
    if family == "bwd_convert_dq":
        return _bwd_convert_dq_kernel_body(name, config)
    raise KeyError(f"Unsupported FMHA family: {family}")


def render_wrapper_header(
    name: str, config: dict, kernel_path: Path, output_dir: Path
) -> str:
    sig = config["signature"]
    alg = config["algorithm"]
    family = sig["family"]
    rel_path = kernel_path.relative_to(output_dir)
    ns = f"ns_{name}"

    if family in {"fwd", "fwd_pagedkv", "fwd_appendkv", "batch_prefill"}:
        backend_factory = "make_timed_fmha_kernel"
    else:
        backend_factory = "make_oneshot_fmha_kernel"

    args_type_map = {
        "fwd": "fmha_fwd_args",
        "fwd_pagedkv": "fmha_fwd_pagedkv_args",
        "fwd_splitkv": "fmha_fwd_splitkv_args",
        "fwd_splitkv_combine": "fmha_fwd_splitkv_args",
        "fwd_appendkv": "fmha_fwd_appendkv_args",
        "batch_prefill": "fmha_batch_prefill_args",
        "bwd_dot_do_o": "fmha_bwd_args",
        "bwd_dq_dk_dv": "fmha_bwd_args",
        "bwd_convert_dq": "fmha_bwd_args",
    }

    run_symbol = "run" if backend_factory == "make_timed_fmha_kernel" else "launch"

    tile = alg["tile"]
    wave = alg["wave"]
    warp = alg["warp"]
    pad = alg["padding"]

    return f"""// SPDX-License-Identifier: MIT
#pragma once

// Kernel header first: includes example fmha_fwd.hpp or fmha_bwd.hpp
// which defines all necessary types (enums, args, traits).
#include "{rel_path}"
// Signal to fmha_types.hpp which types are already defined.
#define CK_TILE_FMHA_{"BWD" if family.startswith("bwd") else "FWD"}_TYPES_FROM_EXAMPLE 1
#include "ck_tile/dispatcher/fmha_dispatcher.hpp"
#include "ck_tile/dispatcher/backends/generated_fmha_backend.hpp"

namespace ck_tile {{
namespace dispatcher {{
namespace generated {{

inline FmhaKernelInstancePtr make_{name}(const std::string& gfx_arch = "{config["arch"]}")
{{
    FmhaKernelKey key;
    key.signature.family = {KERNEL_FAMILY_TO_ENUM[family]};
    key.signature.data_type = "{sig["data_type"]}";
    key.signature.is_group_mode = {str(sig["mode"] == "group").lower()};
    key.signature.is_v_rowmajor = {str(sig["vlayout"] == "r").lower()};
    key.signature.has_logits_soft_cap = {str(sig["logits"]).lower()};
    key.signature.mask_type = {MASK_TO_INT[sig["mask"]]};
    key.signature.bias_type = {BIAS_TO_INT[sig["bias"]]};
    key.signature.has_lse = {str(sig["lse"]).lower()};
    key.signature.has_dropout = {str(sig["dropout"]).lower()};
    key.signature.qscale_type = {QSCALE_TO_INT[sig["qscale"]]};
    key.signature.rope_type = {ROPE_TO_INT[sig["rope"]]};
    key.signature.use_paged_kv = {str(sig["paged_kv"]).lower()};
    key.signature.do_fp8_static_quant = {str(sig["fp8_static_quant"]).lower()};
    key.signature.skip_min_seqlen_q = {str(sig["skip_min_seqlen_q"]).lower()};
    key.signature.has_sink = {str(sig["sink"]).lower()};
    key.signature.has_dbias = {str(sig["dbias"]).lower()};
    key.signature.is_store_randval = {str(sig["store_randval"]).lower()};
    key.signature.is_deterministic = {str(sig["deterministic"]).lower()};
    key.signature.kv_memory_layout = {KV_MEMORY_LAYOUT_TO_INT[sig["kv_memory_layout"]]};
    key.signature.kv_lookup_table = {KV_LOOKUP_TO_INT[sig["kv_lookup_table"]]};
    key.signature.page_size = {sig["page_size"]};
    key.signature.hdim_q = {sig["hdim_q"]};
    key.signature.hdim_v = {sig["hdim_v"]};

    key.algorithm.tile_shape = {{{tile[0]}, {tile[1]}, {tile[2]}, {tile[3]}, {tile[4]}, {tile[5]}}};
    key.algorithm.wave_shape = {{{wave[0]}, {wave[1]}, {wave[2]}, {wave[3]}, {wave[4]}, {wave[5]}, {wave[6]}, {wave[7]}, {wave[8]}}};
    key.algorithm.warp_tile_shape = {{{warp[0]}, {warp[1]}, {warp[2]}, {warp[3]}, {warp[4]}, {warp[5]}, {warp[6]}, {warp[7]}, {warp[8]}}};
    key.algorithm.pipeline = "{alg["pipeline"]}";
    key.algorithm.pad_s = {str(pad[0]).lower()};
    key.algorithm.pad_sk = {str(pad[1]).lower()};
    key.algorithm.pad_d = {str(pad[2]).lower()};
    key.algorithm.pad_dv = {str(pad[3]).lower()};
    key.algorithm.use_trload = {str(alg["use_trload"]).lower()};
    key.algorithm.block_per_cu = {alg["block_per_cu"]};
    key.algorithm.num_wave_groups = {alg["num_wave_groups"]};
    key.algorithm.max_splits_log2 = {alg["max_splits_log2"]};
    key.algorithm.max_seq_len_q = {alg["max_seq_len_q"]};
    key.algorithm.hdim_q_alignment = {alg["hdim_q_alignment"]};
    key.algorithm.hdim_v_alignment = {alg["hdim_v_alignment"]};
    key.algorithm.selection_rank = {alg["selection_rank"]};
    key.algorithm.constraint_tag = "{alg["constraint_tag"]}";
    key.gfx_arch = gfx_arch;

    return backends::{backend_factory}<{args_type_map[family]}>(key, "{name}", {ns}::{run_symbol});
}}

}} // namespace generated
}} // namespace dispatcher
}} // namespace ck_tile
"""


def generate_cpp_compilation_unit(name: str) -> str:
    return f"""// SPDX-License-Identifier: MIT
// Auto-generated compilation unit for {name}

#include "{name}.hpp"

namespace ck_tile {{ namespace generated {{
volatile bool _{name}_loaded = true;
}} }}
"""


class _GenItem:
    def __init__(self, output_dir: Path, config: dict):
        self.output_dir = output_dir
        self.config = config
        self.name = kernel_name_from_config(config)

    def __str__(self) -> str:
        return self.name


def _generate_one(item: _GenItem):
    name = item.name
    output_dir = item.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    wrapper_dir = output_dir / "dispatcher_wrappers"
    wrapper_dir.mkdir(parents=True, exist_ok=True)

    kernel_path = output_dir / f"{name}.hpp"
    kernel_path.write_text(render_kernel_header(name, item.config))

    wrapper_path = wrapper_dir / f"dispatcher_wrapper_{name}.hpp"
    wrapper_path.write_text(
        render_wrapper_header(name, item.config, kernel_path, output_dir)
    )

    cpp_path = output_dir / f"{name}.cpp"
    cpp_path.write_text(generate_cpp_compilation_unit(name))

    return str(kernel_path), str(wrapper_path), str(cpp_path)


def _iter_configs(config_blob: Union[dict, list]) -> Iterable[dict]:
    if isinstance(config_blob, list):
        return config_blob
    if "configs" in config_blob:
        return config_blob["configs"]
    return [config_blob]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified FMHA dispatcher code generator"
    )
    parser.add_argument(
        "--output", "--output-dir", dest="output_dir", type=Path, required=True
    )
    parser.add_argument(
        "--gpu-target", "--arch", dest="gpu_target", type=str, default="gfx942"
    )
    parser.add_argument("--config-json", type=str, required=True)
    parser.add_argument("--profile", type=str)
    parser.add_argument("--receipt", type=str)
    parser.add_argument("--no-parallel", action="store_true")
    args = parser.parse_args()

    arch_specs = load_arch_specs()
    raw = json.loads(args.config_json)
    configs = []
    failures = []

    for entry in _iter_configs(raw):
        cfg = _canonicalize_config(entry, args.gpu_target, arch_specs)
        profile_name = cfg.get("profile") or args.profile
        receipt_name = cfg.get("receipt") or args.receipt

        validation = validate_config(cfg, arch_specs)
        if not validation.valid:
            failures.append((cfg, validation.errors))
            continue

        if not profile_allows(cfg, profile=profile_name, receipt=receipt_name):
            failures.append(
                (
                    cfg,
                    [
                        f"profile filter rejected config ({profile_name or receipt_name})"
                    ],
                )
            )
            continue

        configs.append(cfg)

    if failures:
        for cfg, errors in failures:
            log.error(
                "Rejected FMHA config %s",
                cfg.get("signature", {}).get("family", "unknown"),
            )
            for error in errors:
                log.error("  %s", error)
        if not configs:
            return 1

    items = [_GenItem(args.output_dir, config) for config in configs]
    parallel_generate(
        _generate_one, items, parallel=not args.no_parallel and len(items) > 1
    )

    log.info("Generated %d FMHA kernel specialization(s)", len(items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
