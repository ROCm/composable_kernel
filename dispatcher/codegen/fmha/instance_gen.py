#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
FMHA instance generation — generates tile configs and expands kernel instances.

Three layers:
  1. Tile generation — enumerate valid (bm0, bn0, bk0, warp) combinations
  2. Feature enumeration — enumerate valid (mask, bias, lse, dropout, padding) combinations
  3. Instance expansion — cross-product tiles × features × modes → kernel configs

All hardware facts and constraints come from specs.py.
All symbol mappings come from symbol_map.py.

Usage:
    python -m fmha.instance_gen configs/receipt0_fwd.json --arch gfx950
    python -m fmha.instance_gen configs/fwd_ci.json --arch gfx950 --list
"""

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_DISPATCHER_ROOT = _THIS_DIR.parents[1]
sys.path.insert(0, str(_DISPATCHER_ROOT / "python"))
sys.path.insert(0, str(_THIS_DIR))

from validation import (  # noqa: E402
    ARCH_DTYPES,
    BIASES,
    BOOLS,
    BWD_CONVERT_DQ_HDIMS,
    BWD_CONVERT_DQ_TILE_GROUPS,
    BWD_DOT_DO_O_HDIMS,
    BWD_DQ_DK_DV_EXTRA_TILES,
    BWD_DQ_DK_DV_TILES_FP16,
    BWD_DQ_WAVE_WARP,
    BWD_DROPOUTS,
    BWD_EXTRA_PAD_COMBOS,
    BWD_PAD_COMBOS,
    BWD_SMALL_DROPOUTS,
    DT_FP16_BF16,
    DT_FP32,
    DT_FP8,
    DT_FP8FP32,
    ELEMENT_SIZES,
    K0_MAX_SUBMAX_MAP,
    LDS_LIMITS,
    MASKS,
    SPLITKV_COMBINE_HDIMS_FP16,
    SPLITKV_COMBINE_HDIMS_FP8,
    SUPPORTED_HDIMS,
    VALID_BK0,
    VALID_BM0,
    VALID_BN0,
    WARP_CLASSES,
    check_gfx9_tile_constraints,
    check_gfx950_tile_constraints,
    check_group_mode_padding,
    check_logits_bias,
    check_qr_mfma_insts,
    receipt_filter,
    tile_passes_all_constraints,
)
from fmha_utils import FmhaKernelConfig  # noqa: E402  (from dispatcher/python/)


# =============================================================================
# Tile configuration dataclass
# =============================================================================


@dataclass(frozen=True)
class FmhaTileConfig:
    """Complete FMHA tile configuration with all derived parameters.

    Field naming follows CK's TileFmhaShape template parameters:
    - bm0/bn0/bk0: block tile for Gemm0 (Q*K^T), from sequence<M,N,K,N1,K1,K0Max>
    - bn1/bk1: block tile for Gemm1 (P*V)
    - bk0max: kSubQKHeaddim from tile_fmha_shape.hpp
    - rm0: wave repeat in M direction = bm0/wm0
    - wm0/wn0/wk0: MFMA/WMMA warp tile from warp_gemm_dispatcher.hpp
    """

    bm0: int
    bn0: int
    bk0: int
    bn1: int  # = hdim_v
    bk1: int  # = 32 typically
    bk0max: int  # = K0_MAX_SUBMAX_MAP[hdim_q]
    rm0: int  # wave repeat = bm0/wm0
    wm0: int
    wn0: int
    wk0: int
    wm1: int
    wn1: int
    wk1: int
    rn0: int = 1
    rk0: int = 1
    rm1: int = 1
    rn1: int = 1
    rk1: int = 1

    @property
    def tile_6(self) -> Tuple[int, int, int, int, int, int]:
        return (self.bm0, self.bn0, self.bk0, self.bn1, self.bk1, self.bk0max)


# =============================================================================
# BK1 derivation
# =============================================================================


def derive_bk1(bm0: int, bn0: int, bk0: int, hdim_q: int, hdim_v: int) -> int:
    """Derive bk1 from tile config for fp16/bf16/fp32.

    Source: fmha_fwd.py FmhaFwdTileSize definitions — bk1 (element 4) is
    always 32 except for three specific configs where it's 16.
    These special cases come from the CK example's hand-tuned tile tables.
    """
    if (bm0, bn0, bk0, hdim_q) in (
        (128, 64, 32, 128),
        (32, 128, 32, 128),
        (32, 128, 16, 48),
    ):
        return 16
    return 32


def derive_bk1_fp8(bm0: int, bn0: int, bk0: int, hdim_q: int, hdim_v: int) -> int:
    """Derive bk1 for fp8 dtypes.

    Source: fmha_fwd.py FP8 tile definitions — bk1 always equals bk0.
    """
    return bk0


# =============================================================================
# Tile generation
# =============================================================================


def generate_fwd_tiles(
    arch: str,
    dtype: str,
    hdim_q: int,
    hdim_v: int,
    pipeline: str = "qr_async",
    apply_constraints: bool = True,
) -> List[FmhaTileConfig]:
    """Generate fwd tile configurations.

    apply_constraints=True (default): filter through tile_passes_all_constraints
        — used by rules-mode benchmarking and codegen.
    apply_constraints=False: only basic sanity (warp alignment, bk0<=hdim_q)
        — used by exhaustive-mode benchmarking to find tiles the C++ compiler
        accepts that our rules might reject.
    """
    warp_classes = WARP_CLASSES.get(dtype, [(32, 32, 16)])
    bk0max = K0_MAX_SUBMAX_MAP.get(hdim_q, hdim_q)
    is_fp8 = "fp8" in dtype or dtype in ("bf8", "mxfp8", "mxfp4")

    tiles: List[FmhaTileConfig] = []
    for bm0 in VALID_BM0:
        for bn0 in VALID_BN0:
            for bk0 in VALID_BK0:
                if bk0 > hdim_q:
                    continue
                for wm0, wn0, wk0 in warp_classes:
                    if bm0 % wm0 != 0 or bn0 % wn0 != 0 or bk0 % wk0 != 0:
                        continue
                    if apply_constraints and not tile_passes_all_constraints(
                        arch,
                        dtype,
                        hdim_q,
                        hdim_v,
                        pipeline,
                        bm0,
                        bn0,
                        bk0,
                        wm0,
                        wn0,
                        wk0,
                    ):
                        continue

                    rm0 = bm0 // wm0
                    bk1 = (
                        derive_bk1_fp8(bm0, bn0, bk0, hdim_q, hdim_v)
                        if is_fp8
                        else derive_bk1(bm0, bn0, bk0, hdim_q, hdim_v)
                    )

                    tiles.append(
                        FmhaTileConfig(
                            bm0=bm0,
                            bn0=bn0,
                            bk0=bk0,
                            bn1=hdim_v,
                            bk1=bk1,
                            bk0max=bk0max,
                            rm0=rm0,
                            rm1=rm0,
                            wm0=wm0,
                            wn0=wn0,
                            wk0=wk0,
                            wm1=wm0,
                            wn1=wn0,
                            wk1=wk0,
                        )
                    )

    return tiles


def generate_splitkv_tiles(
    arch: str,
    dtype: str,
    hdim_q: int,
    hdim_v: int,
    apply_constraints: bool = True,
) -> List[FmhaTileConfig]:
    """Generate splitkv tiles.

    Uses fixed warp class per dtype: (16,16,16) for fp16/bf16/fp32,
    (32,32,32) for fp8. These match the warp tiles used in the CK example's
    splitkv tile definitions (fmha_fwd.py KernelComponentFactory*.get_splitkv_tiles()).
    LDS limit: 64 KiB (non-async pipeline, arch.hpp get_smem_capacity for non-gfx950).

    apply_constraints=False skips LDS check (for exhaustive mode).
    """
    bk0max = K0_MAX_SUBMAX_MAP.get(hdim_q, hdim_q)
    is_fp8 = "fp8" in dtype or dtype == "bf8"
    wm0, wn0, wk0 = (32, 32, 32) if is_fp8 else (16, 16, 16)

    tiles: List[FmhaTileConfig] = []
    for bm0 in VALID_BM0:
        for bn0 in VALID_BN0:
            for bk0 in VALID_BK0:
                if bk0 > hdim_q:
                    continue
                if bm0 % wm0 != 0 or bk0 % wk0 != 0 or bn0 % wn0 != 0:
                    continue
                if apply_constraints:
                    elem_size = ELEMENT_SIZES.get(dtype, 2)
                    lds_limit = LDS_LIMITS.get("qr", 65536)
                    if (bm0 * bk0 + bn0 * bk0) * elem_size > lds_limit:
                        continue

                rm0 = bm0 // wm0
                bk1 = bk0 if is_fp8 else 32

                tiles.append(
                    FmhaTileConfig(
                        bm0=bm0,
                        bn0=bn0,
                        bk0=bk0,
                        bn1=hdim_v,
                        bk1=bk1,
                        bk0max=bk0max,
                        rm0=rm0,
                        rm1=rm0,
                        wm0=wm0,
                        wn0=wn0,
                        wk0=wk0,
                        wm1=wm0,
                        wn1=wn0,
                        wk1=wk0,
                    )
                )

    return tiles


def generate_pagedkv_tiles(
    arch: str,
    dtype: str,
    hdim_q: int,
    hdim_v: int,
    apply_constraints: bool = True,
) -> List[FmhaTileConfig]:
    """PagedKV uses same tile rules as splitkv."""
    return generate_splitkv_tiles(arch, dtype, hdim_q, hdim_v, apply_constraints)


def generate_bwd_tiles(
    arch: str,
    dtype: str,
    hdim_q: int,
    hdim_v: int,
    apply_constraints: bool = True,
) -> List[FmhaTileConfig]:
    """Generate BWD tile configurations.

    apply_constraints=False skips LDS check (for exhaustive mode).
    """
    warp_classes = WARP_CLASSES.get(dtype, [(32, 32, 16)])
    bk0max = K0_MAX_SUBMAX_MAP.get(hdim_q, hdim_q)
    is_fp8 = "fp8" in dtype or dtype in ("bf8", "mxfp8", "mxfp4")

    tiles: List[FmhaTileConfig] = []
    for bm0 in VALID_BM0:
        for bn0 in VALID_BN0:
            for bk0 in VALID_BK0:
                if bk0 > hdim_q:
                    continue

                for wm0, wn0, wk0 in warp_classes:
                    if bm0 % wm0 != 0 or bk0 % wk0 != 0 or bn0 % wn0 != 0:
                        continue
                    if apply_constraints:
                        elem_size = ELEMENT_SIZES.get(dtype, 2)
                        lds_limit = LDS_LIMITS.get("qs", 65536)
                        if (bm0 * bk0 + bn0 * bk0) * elem_size > lds_limit:
                            continue

                    rm0 = bm0 // wm0
                    bk1 = bk0 if is_fp8 else 32

                    tiles.append(
                        FmhaTileConfig(
                            bm0=bm0,
                            bn0=bn0,
                            bk0=bk0,
                            bn1=hdim_v,
                            bk1=bk1,
                            bk0max=bk0max,
                            rm0=rm0,
                            rm1=rm0,
                            wm0=wm0,
                            wn0=wn0,
                            wk0=wk0,
                            wm1=wm0,
                            wn1=wn0,
                            wk1=wk0,
                        )
                    )

    return tiles


def validate_tile(
    tile: "FmhaTileConfig",
    arch: str,
    dtype: str,
    hdim_q: int,
    hdim_v: int,
    pipeline: str = "qr_async",
) -> bool:
    """Validate a single tile configuration against all constraints."""
    return tile_passes_all_constraints(
        arch,
        dtype,
        hdim_q,
        hdim_v,
        pipeline,
        tile.bm0,
        tile.bn0,
        tile.bk0,
        tile.wm0,
        tile.wn0,
        tile.wk0,
    )


# =============================================================================
# Pipeline spec dataclasses
# =============================================================================


@dataclass(frozen=True)
class PipelineSpec:
    """One FWD pipeline variant with its feature flags and padding."""

    tag: str
    mask: str
    bias: str
    lse: str
    dropout: str
    logits: str
    skip: str
    sink: str
    qscale: str = "no"
    spad: str = "f"
    skpad: str = "f"
    dpad: str = "f"
    dvpad: str = "f"


@dataclass(frozen=True)
class SplitKVPipelineSpec:
    """Split-KV main kernel pipeline variant."""

    tag: str
    mask: str
    bias: str
    logits: str
    sink: str
    pagedkv: str = "f"
    squant: str = "f"
    spad: str = "f"
    skpad: str = "f"
    dpad: str = "f"
    dvpad: str = "f"
    lse: str = "t"


@dataclass(frozen=True)
class SplitKVCombineSpec:
    """Split-KV combine kernel pipeline variant."""

    spad: str
    dvpad: str
    lse: str
    squant: str = "f"


@dataclass(frozen=True)
class AppendKVPipelineSpec:
    """Append-KV pipeline variant."""

    rope: str = "none"
    pagedkv: str = "f"
    spad: str = "t"
    skpad: str = "t"
    dpad: str = "t"
    dvpad: str = "t"


@dataclass(frozen=True)
class BatchPrefillPipelineSpec:
    """Batch prefill pipeline variant."""

    mask: str
    bias: str
    logits: str
    sink: str
    lse: str = "f"
    dropout: str = "f"
    skip: str = "f"
    qscale: str = "no"
    page_size: int = 0
    kv_memory_layout: str = "vectorized"
    kv_lookup_table: str = "sglang"
    spad: str = "t"
    skpad: str = "t"
    dpad: str = "t"
    dvpad: str = "t"


@dataclass(frozen=True)
class BwdPipelineSpec:
    """BWD pipeline variant."""

    family: str
    mask: str = "no"
    bias: str = "no"
    dbias: str = "f"
    dropout: str = "f"
    deterministic: str = "f"
    spad: str = "t"
    skpad: str = "t"
    dpad: str = "t"
    dvpad: str = "t"


# =============================================================================
# Feature-product generators
# =============================================================================


def _fwd_specs_fp16bf16(
    hdim: int,
    hdim_v: int,
    receipt: int,
) -> List[PipelineSpec]:
    """Pipeline specs for fp16/bf16 on gfx9/gfx950.

    Source: fmha_fwd.py KernelComponentFactoryGfx9.get_pipelines() —
    hdim=256 always uses 'qr' (non-async, since bk0 can equal 256).
    Non-256 hdims use 'qr_async' for non-bias configs (async DMA),
    'qr' for bias configs (bias requires Q in LDS).
    Receipt=1 (ck_extended) adds extra 'qr' variants for non-bias.
    """
    specs: List[PipelineSpec] = []

    for logits, mask, bias, lse, dropout, skip, sink in itertools.product(
        BOOLS,
        MASKS,
        BIASES,
        BOOLS,
        BOOLS,
        BOOLS,
        BOOLS,
    ):
        if hdim == 256 and hdim_v == 256:
            specs.append(
                PipelineSpec(
                    "qr",
                    mask,
                    bias,
                    lse,
                    dropout,
                    logits,
                    skip,
                    sink,
                    spad="f",
                    skpad="f",
                    dpad="f",
                    dvpad="f",
                )
            )
            specs.append(
                PipelineSpec(
                    "qr",
                    mask,
                    bias,
                    lse,
                    dropout,
                    logits,
                    skip,
                    sink,
                    spad="t",
                    skpad="t",
                    dpad="f",
                    dvpad="f",
                )
            )
            specs.append(
                PipelineSpec(
                    "qr",
                    mask,
                    bias,
                    lse,
                    dropout,
                    logits,
                    skip,
                    sink,
                    spad="t",
                    skpad="t",
                    dpad="t",
                    dvpad="t",
                )
            )
        else:
            if bias == "bias":
                specs.append(
                    PipelineSpec(
                        "qr",
                        mask,
                        bias,
                        lse,
                        dropout,
                        logits,
                        skip,
                        sink,
                        spad="f",
                        skpad="f",
                        dpad="f",
                        dvpad="f",
                    )
                )
                specs.append(
                    PipelineSpec(
                        "qr",
                        mask,
                        bias,
                        lse,
                        dropout,
                        logits,
                        skip,
                        sink,
                        spad="t",
                        skpad="t",
                        dpad="t",
                        dvpad="t",
                    )
                )
            else:
                specs.append(
                    PipelineSpec(
                        "qr_async",
                        mask,
                        bias,
                        lse,
                        dropout,
                        logits,
                        skip,
                        sink,
                        spad="t",
                        skpad="f",
                        dpad="t",
                        dvpad="t",
                    )
                )
                specs.append(
                    PipelineSpec(
                        "qr_async",
                        mask,
                        bias,
                        lse,
                        dropout,
                        logits,
                        skip,
                        sink,
                        spad="t",
                        skpad="t",
                        dpad="t",
                        dvpad="t",
                    )
                )
            if receipt == 1 and bias != "bias":
                specs.append(
                    PipelineSpec(
                        "qr",
                        mask,
                        bias,
                        lse,
                        dropout,
                        logits,
                        skip,
                        sink,
                        spad="t",
                        skpad="t",
                        dpad="t",
                        dvpad="t",
                    )
                )

    return specs


def _fwd_specs_gfx950_extra(hdim: int, hdim_v: int) -> List[PipelineSpec]:
    """Additional trload/v3 pipelines for gfx950 fp16/bf16.

    Source: fmha_fwd.py CompatibilityRuleFactoryGfx950 —
    qr_async_trload only supports hdims (64,64) and (128,128),
    requires no logits/bias/dropout/skip.
    qr_async_trload_v3 only supports (128,128), no/causal mask only.
    """
    specs: List[PipelineSpec] = []

    for logits, mask, bias, lse, dropout, skip, sink in itertools.product(
        BOOLS,
        MASKS,
        BIASES,
        BOOLS,
        BOOLS,
        BOOLS,
        BOOLS,
    ):
        if (
            (hdim, hdim_v) in [(64, 64), (128, 128)]
            and logits == "f"
            and bias == "no"
            and dropout == "f"
            and skip == "f"
        ):
            specs.append(
                PipelineSpec(
                    "qr_async_trload",
                    mask,
                    bias,
                    lse,
                    dropout,
                    logits,
                    skip,
                    sink,
                    spad="f",
                    skpad="f",
                    dpad="f",
                    dvpad="f",
                )
            )
            specs.append(
                PipelineSpec(
                    "qr_async_trload",
                    mask,
                    bias,
                    lse,
                    dropout,
                    logits,
                    skip,
                    sink,
                    spad="f",
                    skpad="f",
                    dpad="t",
                    dvpad="t",
                )
            )

    if (hdim, hdim_v) == (128, 128):
        for logits, mask in itertools.product(BOOLS, ["no", "causal"]):
            specs.append(
                PipelineSpec(
                    "qr_async_trload_v3",
                    mask,
                    "no",
                    "f",
                    "f",
                    logits,
                    "f",
                    "f",
                    spad="t",
                    skpad="t",
                    dpad="f",
                    dvpad="f",
                )
            )

    return specs


def _fwd_specs_fp8(hdim: int, hdim_v: int) -> List[PipelineSpec]:
    """Pipeline specs for fp8bf16/fp8fp32.

    Source: fmha_fwd.py KernelComponentFactoryGfx9._DT_FP8 pipelines —
    hdim=64 uses 'qr' (non-async), others use 'qr_async'.
    FP8 supports pertensor and blockscale quantization (qscale).
    No lse, dropout, skip, or bias for fp8.
    """
    specs: List[PipelineSpec] = []

    for logits, qscale, mask, bias, sink in itertools.product(
        BOOLS,
        ["no", "pertensor", "blockscale"],
        MASKS,
        ["no"],
        BOOLS,
    ):
        tag = "qr" if hdim == 64 else "qr_async"
        specs.append(
            PipelineSpec(
                tag,
                mask,
                bias,
                "f",
                "f",
                logits,
                "f",
                sink,
                qscale=qscale,
                spad="t",
                skpad="f",
                dpad="t",
                dvpad="t",
            )
        )
        specs.append(
            PipelineSpec(
                tag,
                mask,
                bias,
                "f",
                "f",
                logits,
                "f",
                sink,
                qscale=qscale,
                spad="t",
                skpad="t",
                dpad="t",
                dvpad="t",
            )
        )

    return specs


def _fwd_specs_fp32(hdim: int, hdim_v: int) -> List[PipelineSpec]:
    """Pipeline specs for fp32.

    Source: fmha_fwd.py KernelComponentFactoryGfx9._DT_FP32 —
    always uses 'qr' pipeline (no async for fp32).
    Full feature set (mask, bias, lse, dropout, logits, etc.).
    """
    specs: List[PipelineSpec] = []

    for logits, mask, bias, lse, dropout, skip, sink in itertools.product(
        BOOLS,
        MASKS,
        BIASES,
        BOOLS,
        BOOLS,
        BOOLS,
        BOOLS,
    ):
        specs.append(
            PipelineSpec(
                "qr",
                mask,
                bias,
                lse,
                dropout,
                logits,
                skip,
                sink,
                spad="f",
                skpad="f",
                dpad="f",
                dvpad="f",
            )
        )
        specs.append(
            PipelineSpec(
                "qr",
                mask,
                bias,
                lse,
                dropout,
                logits,
                skip,
                sink,
                spad="f",
                skpad="t",
                dpad="f",
                dvpad="f",
            )
        )
        specs.append(
            PipelineSpec(
                "qr",
                mask,
                bias,
                lse,
                dropout,
                logits,
                skip,
                sink,
                spad="t",
                skpad="t",
                dpad="t",
                dvpad="t",
            )
        )

    return specs


def get_pipelines_for_config(
    arch: str,
    dtype: str,
    hdim: int,
    hdim_v: int,
    receipt: int = 0,
) -> List[PipelineSpec]:
    """Get all valid pipeline specs for a given (arch, dtype, hdim, hdim_v, receipt)."""
    if dtype in DT_FP32:
        specs = _fwd_specs_fp32(hdim, hdim_v)
    elif dtype in DT_FP16_BF16:
        specs = _fwd_specs_fp16bf16(hdim, hdim_v, receipt)
        if arch == "gfx950":
            specs.extend(_fwd_specs_gfx950_extra(hdim, hdim_v))
    elif dtype in DT_FP8 or dtype in DT_FP8FP32:
        specs = _fwd_specs_fp8(hdim, hdim_v)
    else:
        return []

    return [
        s
        for s in specs
        if check_logits_bias(s.logits, s.bias) and receipt_filter(receipt, dtype, s)
    ]


# --- SplitKV ---


def get_splitkv_pipelines(
    dtype: str, hdim: int, receipt: int = 0
) -> List[SplitKVPipelineSpec]:
    """Split-KV main kernel pipelines."""
    specs: List[SplitKVPipelineSpec] = []
    SPLITKV_MASKS = ["no", "causal"]

    if dtype in DT_FP16_BF16:
        for logits, mask, bias, pagedkv, sink in itertools.product(
            BOOLS, SPLITKV_MASKS, BIASES, BOOLS, BOOLS
        ):
            if not check_logits_bias(logits, bias):
                continue
            specs.append(
                SplitKVPipelineSpec(
                    "qr",
                    mask,
                    bias,
                    logits,
                    sink,
                    pagedkv,
                    spad="f",
                    skpad="t",
                    dpad="f",
                    dvpad="f",
                )
            )
            specs.append(
                SplitKVPipelineSpec(
                    "qr",
                    mask,
                    bias,
                    logits,
                    sink,
                    pagedkv,
                    spad="t",
                    skpad="f",
                    dpad="f",
                    dvpad="f",
                )
            )
            specs.append(
                SplitKVPipelineSpec(
                    "qr",
                    mask,
                    bias,
                    logits,
                    sink,
                    pagedkv,
                    spad="t",
                    skpad="t",
                    dpad="f",
                    dvpad="f",
                )
            )
            specs.append(
                SplitKVPipelineSpec(
                    "qr",
                    mask,
                    bias,
                    logits,
                    sink,
                    pagedkv,
                    spad="t",
                    skpad="t",
                    dpad="t",
                    dvpad="t",
                )
            )
    elif dtype in ("fp8", "bf8"):
        for logits, mask, bias in itertools.product(BOOLS, SPLITKV_MASKS, BIASES):
            if not check_logits_bias(logits, bias):
                continue
            specs.append(
                SplitKVPipelineSpec(
                    "qr",
                    mask,
                    bias,
                    logits,
                    "f",
                    "f",
                    squant="t",
                    spad="f",
                    skpad="f",
                    dpad="f",
                    dvpad="f",
                )
            )
            specs.append(
                SplitKVPipelineSpec(
                    "qr",
                    mask,
                    bias,
                    logits,
                    "f",
                    "f",
                    squant="t",
                    spad="t",
                    skpad="t",
                    dpad="f",
                    dvpad="f",
                )
            )

    if receipt != 0:
        specs = [s for s in specs if _splitkv_receipt_filter(receipt, dtype, s)]
    return specs


def _splitkv_receipt_filter(
    receipt: int, dtype: str, spec: SplitKVPipelineSpec
) -> bool:
    if receipt == 2:
        return (
            dtype in ("fp16", "bf16")
            and spec.bias in ("no", "alibi")
            and spec.squant == "f"
            and spec.sink == "f"
        )
    if receipt == 4:
        return (
            dtype in ("fp16", "bf16")
            and spec.bias in ("no", "bias")
            and spec.squant == "f"
            and spec.sink == "f"
        )
    if receipt == 200:
        return dtype in ("fp16", "bf16") and spec.squant == "f"
    if receipt == 600:
        return dtype in ("fp16", "bf16") and spec.squant == "f"
    if receipt in (800, 801):
        return dtype == "fp32"
    return True


def get_splitkv_combine_pipelines(
    dtype: str, receipt: int = 0
) -> List[SplitKVCombineSpec]:
    """Split-KV combine kernel pipelines."""
    specs: List[SplitKVCombineSpec] = []
    squant = "t" if dtype in ("fp8", "bf8") else "f"

    if dtype in DT_FP16_BF16:
        for spad, dvpad, lse in itertools.product(BOOLS, BOOLS, BOOLS):
            specs.append(SplitKVCombineSpec(spad, dvpad, lse, squant))
    elif dtype in ("fp8", "bf8"):
        for spad, dvpad in itertools.product(BOOLS, BOOLS):
            specs.append(SplitKVCombineSpec(spad, dvpad, "f", squant))
    return specs


# --- PagedKV ---


def get_pagedkv_pipelines(
    dtype: str, hdim: int, receipt: int = 0
) -> List[PipelineSpec]:
    """PagedKV prefill pipelines."""
    specs: List[PipelineSpec] = []

    if dtype in DT_FP16_BF16:
        for logits, mask, bias, sink in itertools.product(BOOLS, MASKS, BIASES, BOOLS):
            if not check_logits_bias(logits, bias):
                continue
            specs.append(
                PipelineSpec(
                    "qr_pagedkv",
                    mask,
                    bias,
                    "f",
                    "f",
                    logits,
                    "f",
                    sink,
                    spad="t",
                    skpad="f",
                    dpad="f",
                    dvpad="f",
                )
            )
            specs.append(
                PipelineSpec(
                    "qr_pagedkv",
                    mask,
                    bias,
                    "f",
                    "f",
                    logits,
                    "f",
                    sink,
                    spad="t",
                    skpad="t",
                    dpad="f",
                    dvpad="f",
                )
            )
    elif dtype in ("fp8", "bf8"):
        for logits, mask, bias in itertools.product(BOOLS, MASKS, BIASES):
            if not check_logits_bias(logits, bias):
                continue
            specs.append(
                PipelineSpec(
                    "qr_pagedkv",
                    mask,
                    bias,
                    "f",
                    "f",
                    logits,
                    "f",
                    "f",
                    spad="f",
                    skpad="f",
                    dpad="f",
                    dvpad="f",
                )
            )
            specs.append(
                PipelineSpec(
                    "qr_pagedkv",
                    mask,
                    bias,
                    "f",
                    "f",
                    logits,
                    "f",
                    "f",
                    spad="t",
                    skpad="t",
                    dpad="f",
                    dvpad="f",
                )
            )

    if receipt != 0:
        specs = [s for s in specs if receipt_filter(receipt, dtype, s)]
    return specs


# --- AppendKV ---


def get_appendkv_pipelines(
    dtype: str, hdim: int, receipt: int = 0
) -> List[AppendKVPipelineSpec]:
    """Append-KV pipelines."""
    specs: List[AppendKVPipelineSpec] = []

    if dtype in DT_FP16_BF16:
        for pagedkv in ["t", "f"]:
            specs.append(
                AppendKVPipelineSpec(
                    rope="none",
                    pagedkv=pagedkv,
                    spad="f",
                    skpad="t",
                    dpad="f",
                    dvpad="f",
                )
            )
            specs.append(
                AppendKVPipelineSpec(
                    rope="none",
                    pagedkv=pagedkv,
                    spad="t",
                    skpad="t",
                    dpad="t",
                    dvpad="t",
                )
            )
            specs.append(
                AppendKVPipelineSpec(
                    rope="interleaved",
                    pagedkv=pagedkv,
                    spad="f",
                    skpad="t",
                    dpad="t",
                    dvpad="f",
                )
            )
            specs.append(
                AppendKVPipelineSpec(
                    rope="interleaved",
                    pagedkv=pagedkv,
                    spad="t",
                    skpad="t",
                    dpad="t",
                    dvpad="t",
                )
            )
            specs.append(
                AppendKVPipelineSpec(
                    rope="half_rotated",
                    pagedkv=pagedkv,
                    spad="f",
                    skpad="t",
                    dpad="t",
                    dvpad="f",
                )
            )
            specs.append(
                AppendKVPipelineSpec(
                    rope="half_rotated",
                    pagedkv=pagedkv,
                    spad="t",
                    skpad="t",
                    dpad="t",
                    dvpad="t",
                )
            )
    elif dtype in ("fp8", "bf8"):
        specs.append(
            AppendKVPipelineSpec(
                rope="none", pagedkv="f", spad="t", skpad="t", dpad="t", dvpad="t"
            )
        )
    return specs


# --- Batch Prefill ---


def get_batch_prefill_pipelines(
    dtype: str, hdim: int, receipt: int = 0
) -> List[BatchPrefillPipelineSpec]:
    """Batch prefill pipelines."""
    specs: List[BatchPrefillPipelineSpec] = []
    PREFILL_MASKS = ["no", "causal"]

    if dtype in DT_FP16_BF16:
        for logits, mask, bias, lse, dropout, kvl, kvt in itertools.product(
            BOOLS,
            PREFILL_MASKS,
            BIASES,
            BOOLS,
            BOOLS,
            ["vectorized", "linear"],
            ["vllm", "sglang"],
        ):
            if not check_logits_bias(logits, bias):
                continue
            specs.append(
                BatchPrefillPipelineSpec(
                    mask,
                    bias,
                    logits,
                    "f",
                    lse,
                    dropout,
                    "f",
                    page_size=0,
                    kv_memory_layout=kvl,
                    kv_lookup_table=kvt,
                )
            )
    elif dtype == "fp8bf16":
        for logits, qscale, mask, bias, kvl, kvt in itertools.product(
            BOOLS,
            ["pertensor", "kv_blockscale"],
            MASKS,
            ["no"],
            ["vectorized", "linear"],
            ["vllm", "sglang"],
        ):
            if not check_logits_bias(logits, bias):
                continue
            specs.append(
                BatchPrefillPipelineSpec(
                    mask,
                    bias,
                    logits,
                    "f",
                    "f",
                    "f",
                    "f",
                    qscale=qscale,
                    page_size=0,
                    kv_memory_layout=kvl,
                    kv_lookup_table=kvt,
                )
            )
    return specs


# --- BWD ---


def get_bwd_dq_dk_dv_pipelines(dtype: str, receipt: int = 0) -> List[BwdPipelineSpec]:
    """BWD dq_dk_dv feature product."""
    if dtype not in DT_FP16_BF16:
        return []
    specs: List[BwdPipelineSpec] = []
    for mask, bias, dbias, dropout, deterministic in itertools.product(
        MASKS,
        BIASES,
        BOOLS,
        BWD_DROPOUTS,
        BOOLS,
    ):
        if bias != "bias" and dbias == "t":
            continue
        for dpad, dvpad in BWD_PAD_COMBOS:
            specs.append(
                BwdPipelineSpec(
                    "bwd_dq_dk_dv",
                    mask,
                    bias,
                    dbias,
                    dropout,
                    deterministic,
                    dpad=dpad,
                    dvpad=dvpad,
                )
            )
    return specs


def get_bwd_dq_dk_dv_extra_pipelines(
    dtype: str, is_small: bool = False, receipt: int = 0
) -> List[BwdPipelineSpec]:
    """BWD dq_dk_dv extra tile pipelines (reduced feature set)."""
    if dtype not in DT_FP16_BF16:
        return []
    specs: List[BwdPipelineSpec] = []
    dropouts = BWD_SMALL_DROPOUTS if is_small else BWD_DROPOUTS
    for mask, bias, dbias, dropout, deterministic in itertools.product(
        MASKS,
        BIASES,
        BOOLS,
        dropouts,
        BOOLS,
    ):
        if bias != "bias" and dbias == "t":
            continue
        for dpad, dvpad in BWD_EXTRA_PAD_COMBOS:
            specs.append(
                BwdPipelineSpec(
                    "bwd_dq_dk_dv",
                    mask,
                    bias,
                    dbias,
                    dropout,
                    deterministic,
                    dpad=dpad,
                    dvpad=dvpad,
                )
            )
    return specs


def get_bwd_dot_do_o_pipelines(dtype: str) -> List[BwdPipelineSpec]:
    """BWD dot_do_o: spad x dvpad variants."""
    if dtype not in DT_FP16_BF16:
        return []
    return [
        BwdPipelineSpec("bwd_dot_do_o", spad=s, dvpad=d)
        for s, d in itertools.product(BOOLS, BOOLS)
    ]


def get_bwd_convert_dq_pipelines(dtype: str, hdim: int = 0) -> List[BwdPipelineSpec]:
    """BWD convert_dq: spad x deterministic x dpad."""
    if dtype not in DT_FP16_BF16:
        return []
    dpads = ["f", "t", "8"] if hdim == 128 else BOOLS
    return [
        BwdPipelineSpec("bwd_convert_dq", spad=s, deterministic=d, dpad=dp)
        for s, d, dp in itertools.product(BOOLS, BOOLS, dpads)
    ]


# =============================================================================
# Tile compatibility (used by expand to double-check)
# =============================================================================


def tile_compatible(
    arch: str,
    dtype: str,
    hdim: int,
    hdim_v: int,
    pipeline_tag: str,
    tile: Tuple[int, ...],
) -> bool:
    """Check if a tile tuple passes arch-specific constraints (subset of tile_passes_all_constraints)."""

    bm0, bn0, bk0 = tile[0], tile[1], tile[2]

    if not check_gfx9_tile_constraints(
        dtype, hdim, hdim_v, pipeline_tag, bm0, bn0, bk0
    ):
        return False
    if arch == "gfx950":
        if not check_gfx950_tile_constraints(hdim, hdim_v, pipeline_tag, bm0, bn0):
            return False
    # Use default warp for mfma check
    wn0, wk0 = 32, 16
    warp_classes = WARP_CLASSES.get(dtype, [(32, 32, 16)])
    if warp_classes:
        _, wn0, wk0 = warp_classes[0]
    if not check_qr_mfma_insts(arch, hdim, pipeline_tag, bn0, bk0, wn0, wk0):
        return False
    return True


# =============================================================================
# BWD wave/warp lookup
# =============================================================================


def bwd_dq_wave_warp(tile, hq, trload=False):
    """Look up BWD wave/warp config for a tile."""
    trl = "t" if trload else "f"
    key = (tile[0], tile[1], tile[2], trl)
    entry = BWD_DQ_WAVE_WARP.get(key)
    if entry is None:
        for k, v in BWD_DQ_WAVE_WARP.items():
            if k[:3] == (tile[0], tile[1], tile[2]):
                entry = v
                break
    if entry is None:
        bn0 = tile[1]
        wn = min(4, max(1, bn0 // 32))
        return {
            "wave_m0": 1,
            "wave_n0": wn,
            "wave_k0": 1,
            "wave_m1": 4,
            "wave_n1": 1,
            "wave_k1": 1,
            "wave_m2": 1,
            "wave_n2": wn,
            "wave_k2": 1,
            "warp_m0": 16,
            "warp_n0": 16,
            "warp_k0": 32,
            "warp_m1": 16,
            "warp_n1": 16,
            "warp_k1": 16,
            "warp_m2": 16,
            "warp_n2": 16,
            "warp_k2": 16,
        }
    w = entry["wave"]
    wk1 = entry["warp_k1"]
    return {
        "wave_m0": w[0],
        "wave_n0": w[1],
        "wave_k0": w[2],
        "wave_m1": w[3],
        "wave_n1": w[4],
        "wave_k1": w[5],
        "wave_m2": w[6],
        "wave_n2": w[7],
        "wave_k2": w[8],
        "warp_m0": 16,
        "warp_n0": 16,
        "warp_k0": 32,
        "warp_m1": 16,
        "warp_n1": 16,
        "warp_k1": wk1,
        "warp_m2": 16,
        "warp_n2": 16,
        "warp_k2": 16,
    }


# =============================================================================
# Instance expansion
# =============================================================================

VARIANT_TO_FAMILY = {
    "fwd": "fwd",
    "bwd": "bwd_dq_dk_dv",
    "splitkv": "fwd_splitkv",
    "appendkv": "fwd_appendkv",
    "pagedkv": "fwd_pagedkv",
    "batch_prefill": "batch_prefill",
}

MODES = ["batch", "group"]

_MASK_MAP = {"no": "no", "causal": "top_left", "generic": "generic"}
_BIAS_MAP = {"no": "no", "bias": "bias", "alibi": "alibi"}


def _pad_val(s: str) -> int:
    if s == "f":
        return 0
    if s == "t":
        return 1
    return int(s)


def expand_sweep(
    config_path: Optional[str],
    arch: str,
    receipt: int = 0,
    mode: str = "rules",
    restrict_hdims: Optional[List[Tuple[int, int]]] = None,
    default_variant: str = "fwd",
) -> List[FmhaKernelConfig]:
    """Expand sweep into full kernel instance list.

    Args:
        config_path: Path to JSON sweep config, or None for defaults
            (only valid with mode="exhaustive").
        arch: Target GPU arch ("gfx950" etc.).
        receipt: Receipt level (0 = full, higher = filtered).
        mode: "rules" applies tile_passes_all_constraints + receipt-driven
            pipeline×feature coupling. "exhaustive" skips constraints and uses
            a raw cartesian feature product (variant must be "fwd").
        restrict_hdims: If set, only generate configs for these (hq, hv) pairs.
        default_variant: Variant to use when config_path is None.
    """
    if config_path is None:
        if mode != "exhaustive":
            raise ValueError("config_path is required for mode='rules'")
        config = {"variant": default_variant, "trait_config": {}}
    else:
        with open(config_path) as f:
            config = json.load(f)

    variant = config.get("variant", default_variant)

    # Build allow-list filters from JSON trait_config
    trait_cfg = config.get("trait_config", {})

    def _allow(key: str) -> Optional[Set[str]]:
        entry = trait_cfg.get(key)
        if entry is None:
            return None
        return set(entry.get("values", []))

    allowed_dtypes = _allow("data_type")
    allowed_pipes = _allow("pipeline")
    allowed_masks = _allow("mask")
    allowed_biases = _allow("bias")
    allowed_modes = _allow("mode")
    allowed_lse = _allow("lse")
    allowed_dropout = _allow("dropout")
    allowed_logits = _allow("logits")
    allowed_sink = _allow("sink")
    allowed_paged_kv = _allow("paged_kv")

    # block_per_cu: int or list of ints to sweep
    bpc_entry = trait_cfg.get("block_per_cu", {})
    block_per_cu_values = bpc_entry.get("values", [-1])
    if isinstance(block_per_cu_values, int):
        block_per_cu_values = [block_per_cu_values]

    # Intersect with arch support
    arch_dtypes = set(ARCH_DTYPES.get(arch, ARCH_DTYPES.get("gfx950", [])))
    dtypes = (
        sorted(allowed_dtypes & arch_dtypes) if allowed_dtypes else sorted(arch_dtypes)
    )

    configs: List[FmhaKernelConfig] = []

    if mode == "exhaustive":
        if variant == "fwd":
            configs = _expand_fwd_exhaustive(
                arch,
                dtypes,
                allowed_pipes,
                allowed_masks,
                allowed_biases,
                allowed_modes,
                allowed_lse,
                allowed_dropout,
                allowed_logits,
                allowed_sink,
                block_per_cu_values,
                restrict_hdims,
            )
        elif variant == "splitkv":
            configs = _expand_splitkv_exhaustive(
                arch,
                dtypes,
                allowed_masks,
                allowed_biases,
                allowed_modes,
                allowed_logits,
                allowed_sink,
                allowed_paged_kv,
                restrict_hdims,
            )
        elif variant == "pagedkv":
            configs = _expand_pagedkv_exhaustive(
                arch,
                dtypes,
                allowed_masks,
                allowed_biases,
                allowed_modes,
                restrict_hdims,
            )
        elif variant == "bwd":
            configs = _expand_bwd_exhaustive(
                arch,
                dtypes,
                allowed_masks,
                allowed_biases,
                allowed_modes,
                restrict_hdims,
            )
        elif variant in ("appendkv", "batch_prefill"):
            # These have fixed tiles (no tile sweep), so exhaustive = rules mode
            if variant == "appendkv":
                configs = _expand_appendkv(
                    arch, dtypes, 0, restrict_hdims=restrict_hdims
                )
            else:
                configs = _expand_batch_prefill(
                    arch,
                    dtypes,
                    0,
                    allowed_masks,
                    allowed_biases,
                    restrict_hdims=restrict_hdims,
                )
        else:
            raise ValueError(f"Exhaustive mode not supported for variant {variant!r}")
    elif variant == "fwd":
        configs = _expand_fwd(
            arch,
            dtypes,
            receipt,
            allowed_pipes,
            allowed_masks,
            allowed_biases,
            allowed_modes,
            allowed_lse,
            allowed_dropout,
            allowed_logits,
            allowed_sink,
            block_per_cu_values,
            restrict_hdims=restrict_hdims,
        )
    elif variant == "splitkv":
        configs = _expand_splitkv(
            arch,
            dtypes,
            receipt,
            allowed_masks,
            allowed_biases,
            allowed_modes,
            allowed_logits,
            allowed_sink,
            allowed_paged_kv,
            restrict_hdims=restrict_hdims,
        )
    elif variant == "pagedkv":
        configs = _expand_pagedkv(
            arch,
            dtypes,
            receipt,
            allowed_masks,
            allowed_biases,
            allowed_modes,
            restrict_hdims=restrict_hdims,
        )
    elif variant == "appendkv":
        configs = _expand_appendkv(arch, dtypes, receipt, restrict_hdims=restrict_hdims)
    elif variant == "batch_prefill":
        configs = _expand_batch_prefill(
            arch,
            dtypes,
            receipt,
            allowed_masks,
            allowed_biases,
            restrict_hdims=restrict_hdims,
        )
    elif variant == "bwd":
        configs = _expand_bwd(
            arch,
            dtypes,
            receipt,
            allowed_masks,
            allowed_biases,
            allowed_modes,
            restrict_hdims=restrict_hdims,
        )

    # Dedup
    seen: set = set()
    unique: List[FmhaKernelConfig] = []
    for c in configs:
        if c.name not in seen:
            seen.add(c.name)
            unique.append(c)
    return unique


def _build_fwd_kernel_config(
    *,
    arch: str,
    dtype: str,
    mode: str,
    hq: int,
    hv: int,
    pipeline: str,
    tc: FmhaTileConfig,
    pad_s: int = 0,
    pad_sk: int = 0,
    pad_d: int = 0,
    pad_dv: int = 0,
    mask: str = "no",
    bias: str = "no",
    lse: bool = False,
    dropout: bool = False,
    logits: bool = False,
    sink: bool = False,
    skip_min_seqlen_q: bool = False,
    qscale: str = "no",
    block_per_cu: int = -1,
) -> FmhaKernelConfig:
    """Single source of truth for fwd FmhaKernelConfig kwargs derived from a tile."""
    return FmhaKernelConfig(
        family="fwd",
        data_type=dtype,
        mode=mode,
        hdim_q=hq,
        hdim_v=hv,
        pipeline=pipeline,
        tile_m0=tc.bm0,
        tile_n0=tc.bn0,
        tile_k0=tc.bk0,
        tile_n1=tc.bn1,
        tile_k1=tc.bk1,
        tile_k0max=tc.bk0max,
        wave_m0=tc.rm0,
        wave_n0=1,
        wave_k0=1,
        wave_m1=tc.rm0,
        wave_n1=1,
        wave_k1=1,
        warp_m0=tc.wm0,
        warp_n0=tc.wn0,
        warp_k0=tc.wk0,
        warp_m1=tc.wm1,
        warp_n1=tc.wn1,
        warp_k1=tc.wk1,
        pad_s=pad_s,
        pad_sk=pad_sk,
        pad_d=pad_d,
        pad_dv=pad_dv,
        mask=mask,
        bias=bias,
        lse=lse,
        dropout=dropout,
        logits=logits,
        sink=sink,
        skip_min_seqlen_q=skip_min_seqlen_q,
        qscale=qscale,
        block_per_cu=block_per_cu,
        gfx_arch=arch,
    )


def _expand_fwd(
    arch,
    dtypes,
    receipt,
    allowed_pipes,
    allowed_masks,
    allowed_biases,
    allowed_modes,
    allowed_lse,
    allowed_dropout,
    allowed_logits,
    allowed_sink,
    block_per_cu_values=None,
    restrict_hdims=None,
):
    if block_per_cu_values is None:
        block_per_cu_values = [-1]
    configs = []
    for dtype in dtypes:
        hdims = SUPPORTED_HDIMS.get(dtype, [])
        if restrict_hdims is not None:
            hdims = [hv for hv in hdims if hv in restrict_hdims]
        for hq, hv in hdims:
            pipeline_specs = get_pipelines_for_config(arch, dtype, hq, hv, receipt)
            _tile_cache: Dict[str, List[FmhaTileConfig]] = {}
            for mode in MODES:
                if allowed_modes is not None and mode not in allowed_modes:
                    continue
                for spec in pipeline_specs:
                    if not check_group_mode_padding(mode, spec.spad, spec.skpad):
                        continue
                    if allowed_pipes is not None and spec.tag not in allowed_pipes:
                        continue
                    mm = _MASK_MAP.get(spec.mask, spec.mask)
                    mb = _BIAS_MAP.get(spec.bias, spec.bias)
                    lv = spec.lse == "t"
                    dv = spec.dropout == "t"
                    lgv = spec.logits == "t"
                    sv = spec.sink == "t"
                    skv = spec.skip == "t"
                    if allowed_masks is not None and mm not in allowed_masks:
                        continue
                    if allowed_biases is not None and mb not in allowed_biases:
                        continue
                    if allowed_lse is not None and lv not in allowed_lse:
                        continue
                    if allowed_dropout is not None and dv not in allowed_dropout:
                        continue
                    if allowed_logits is not None and lgv not in allowed_logits:
                        continue
                    if allowed_sink is not None and sv not in allowed_sink:
                        continue
                    if spec.tag not in _tile_cache:
                        _tile_cache[spec.tag] = generate_fwd_tiles(
                            arch, dtype, hq, hv, spec.tag
                        )
                    for tc in _tile_cache[spec.tag]:
                        t6 = (tc.bm0, tc.bn0, tc.bk0, tc.bn1, tc.bk1, tc.bk0max)
                        if not tile_compatible(arch, dtype, hq, hv, spec.tag, t6):
                            continue
                        for bpc in block_per_cu_values:
                            configs.append(
                                _build_fwd_kernel_config(
                                    arch=arch,
                                    dtype=dtype,
                                    mode=mode,
                                    hq=hq,
                                    hv=hv,
                                    pipeline=spec.tag,
                                    tc=tc,
                                    pad_s=_pad_val(spec.spad),
                                    pad_sk=_pad_val(spec.skpad),
                                    pad_d=_pad_val(spec.dpad),
                                    pad_dv=_pad_val(spec.dvpad),
                                    mask=mm,
                                    bias=mb,
                                    lse=lv,
                                    dropout=dv,
                                    logits=lgv,
                                    sink=sv,
                                    skip_min_seqlen_q=skv,
                                    qscale=spec.qscale,
                                    block_per_cu=bpc,
                                )
                            )
    return configs


def _expand_fwd_exhaustive(
    arch,
    dtypes,
    allowed_pipes,
    allowed_masks,
    allowed_biases,
    allowed_modes,
    allowed_lse,
    allowed_dropout,
    allowed_logits,
    allowed_sink,
    block_per_cu_values,
    restrict_hdims,
):
    """Exhaustive fwd expansion: ALL tiles (no constraint filter) × full feature cross-product.

    Differs from _expand_fwd in two ways:
      1. Tiles come from generate_fwd_tiles(..., apply_constraints=False)
      2. Features are a raw cartesian product (no pipeline-receipt coupling)

    Used by --tiles=exhaustive in the benchmark to discover compilable tiles
    that the rules engine rejects.
    """
    pipelines = (
        sorted(allowed_pipes)
        if allowed_pipes
        else ["qr", "qr_async", "qr_async_trload", "qr_async_trload_v3"]
    )
    modes = sorted(allowed_modes) if allowed_modes else MODES
    masks = (
        sorted(allowed_masks) if allowed_masks else ["no", "top_left", "bottom_right"]
    )
    biases = sorted(allowed_biases) if allowed_biases else ["no", "bias", "alibi"]
    lse_vals = sorted(allowed_lse) if allowed_lse else [False, True]
    dropout_vals = sorted(allowed_dropout) if allowed_dropout else [False, True]
    logits_vals = sorted(allowed_logits) if allowed_logits else [False, True]
    sink_vals = sorted(allowed_sink) if allowed_sink else [False]
    bpc_vals = block_per_cu_values if block_per_cu_values else [-1, 1, 2]

    configs: List[FmhaKernelConfig] = []
    for dtype in dtypes:
        hdims = SUPPORTED_HDIMS.get(dtype, [])
        if restrict_hdims is not None:
            hdims = [hv for hv in hdims if hv in restrict_hdims]
        for hq, hv in hdims:
            for pipeline in pipelines:
                tiles = generate_fwd_tiles(
                    arch, dtype, hq, hv, pipeline, apply_constraints=False
                )
                for tc in tiles:
                    for mode, mask, bias, lv, dv, lgv, sv, bpc in itertools.product(
                        modes,
                        masks,
                        biases,
                        lse_vals,
                        dropout_vals,
                        logits_vals,
                        sink_vals,
                        bpc_vals,
                    ):
                        configs.append(
                            _build_fwd_kernel_config(
                                arch=arch,
                                dtype=dtype,
                                mode=mode,
                                hq=hq,
                                hv=hv,
                                pipeline=pipeline,
                                tc=tc,
                                mask=mask,
                                bias=bias,
                                lse=lv,
                                dropout=dv,
                                logits=lgv,
                                sink=sv,
                                block_per_cu=bpc,
                            )
                        )
    return configs


def _expand_splitkv_exhaustive(
    arch,
    dtypes,
    allowed_masks,
    allowed_biases,
    allowed_modes,
    allowed_logits,
    allowed_sink,
    allowed_paged_kv,
    restrict_hdims,
):
    """Exhaustive splitkv: ALL tiles (no LDS filter) × full feature product."""
    modes = sorted(allowed_modes) if allowed_modes else MODES
    masks = (
        sorted(allowed_masks) if allowed_masks else ["no", "top_left", "bottom_right"]
    )
    biases = sorted(allowed_biases) if allowed_biases else ["no", "bias", "alibi"]
    logits_vals = sorted(allowed_logits) if allowed_logits else [False, True]
    sink_vals = sorted(allowed_sink) if allowed_sink else [False]

    configs: List[FmhaKernelConfig] = []
    for dtype in dtypes:
        hdims = SUPPORTED_HDIMS.get(dtype, [])
        if restrict_hdims is not None:
            hdims = [hv for hv in hdims if hv in restrict_hdims]
        for hq, hv in hdims:
            tiles = generate_splitkv_tiles(arch, dtype, hq, hv, apply_constraints=False)
            for tc in tiles:
                for mode, mask, bias, lgv, sv in itertools.product(
                    modes,
                    masks,
                    biases,
                    logits_vals,
                    sink_vals,
                ):
                    configs.append(
                        FmhaKernelConfig(
                            family="fwd_splitkv",
                            data_type=dtype,
                            mode=mode,
                            hdim_q=hq,
                            hdim_v=hv,
                            pipeline="qr",
                            tile_m0=tc.bm0,
                            tile_n0=tc.bn0,
                            tile_k0=tc.bk0,
                            tile_n1=tc.bn1,
                            tile_k1=tc.bk1,
                            tile_k0max=tc.bk0max,
                            wave_m0=tc.rm0,
                            wave_n0=1,
                            wave_k0=1,
                            wave_m1=tc.rm0,
                            wave_n1=1,
                            wave_k1=1,
                            warp_m0=tc.wm0,
                            warp_n0=tc.wn0,
                            warp_k0=tc.wk0,
                            warp_m1=tc.wm1,
                            warp_n1=tc.wn1,
                            warp_k1=tc.wk1,
                            mask=mask,
                            bias=bias,
                            lse=True,
                            logits=lgv,
                            sink=sv,
                            gfx_arch=arch,
                        )
                    )
    return configs


def _expand_pagedkv_exhaustive(
    arch,
    dtypes,
    allowed_masks,
    allowed_biases,
    allowed_modes,
    restrict_hdims,
):
    """Exhaustive pagedkv: ALL tiles (no LDS filter) × full feature product."""
    modes = sorted(allowed_modes) if allowed_modes else MODES
    masks = (
        sorted(allowed_masks) if allowed_masks else ["no", "top_left", "bottom_right"]
    )
    biases = sorted(allowed_biases) if allowed_biases else ["no", "bias", "alibi"]

    configs: List[FmhaKernelConfig] = []
    for dtype in dtypes:
        hdims = SUPPORTED_HDIMS.get(dtype, [])
        if restrict_hdims is not None:
            hdims = [hv for hv in hdims if hv in restrict_hdims]
        for hq, hv in hdims:
            tiles = generate_pagedkv_tiles(arch, dtype, hq, hv, apply_constraints=False)
            for tc in tiles:
                for mode, mask, bias in itertools.product(modes, masks, biases):
                    configs.append(
                        FmhaKernelConfig(
                            family="fwd_pagedkv",
                            data_type=dtype,
                            mode=mode,
                            hdim_q=hq,
                            hdim_v=hv,
                            pipeline="qr_pagedkv",
                            tile_m0=tc.bm0,
                            tile_n0=tc.bn0,
                            tile_k0=tc.bk0,
                            tile_n1=tc.bn1,
                            tile_k1=tc.bk1,
                            tile_k0max=tc.bk0max,
                            wave_m0=tc.rm0,
                            wave_n0=1,
                            wave_k0=1,
                            wave_m1=tc.rm0,
                            wave_n1=1,
                            wave_k1=1,
                            warp_m0=tc.wm0,
                            warp_n0=tc.wn0,
                            warp_k0=tc.wk0,
                            warp_m1=tc.wm1,
                            warp_n1=tc.wn1,
                            warp_k1=tc.wk1,
                            mask=mask,
                            bias=bias,
                            paged_kv=True,
                            gfx_arch=arch,
                        )
                    )
    return configs


def _expand_bwd_exhaustive(
    arch,
    dtypes,
    allowed_masks,
    allowed_biases,
    allowed_modes,
    restrict_hdims,
):
    """Exhaustive bwd: ALL tiles (no LDS filter) × full feature product.

    Note: BWD uses spec-defined fixed tiles for dq_dk_dv, but we can still
    exhaust the dot_do_o and convert_dq with unfiltered tile generation.
    For dq_dk_dv we use generate_bwd_tiles(apply_constraints=False) since
    CK's bwd tile tables are hand-curated and the exhaustive sweep should
    explore beyond them.
    """
    modes = sorted(allowed_modes) if allowed_modes else MODES
    masks = (
        sorted(allowed_masks) if allowed_masks else ["no", "top_left", "bottom_right"]
    )
    biases = sorted(allowed_biases) if allowed_biases else ["no", "bias", "alibi"]
    deterministic_vals = [False, True]
    dropout_vals = ["no", "p", "rp"]

    configs: List[FmhaKernelConfig] = []
    for dtype in dtypes:
        if dtype not in ("fp16", "bf16"):
            continue

        # dot_do_o — fixed tile, just sweep features
        dot_specs = get_bwd_dot_do_o_pipelines(dtype)
        for hd in BWD_DOT_DO_O_HDIMS:
            if restrict_hdims is not None and (hd, hd) not in restrict_hdims:
                continue
            for mode in modes:
                for spec in dot_specs:
                    configs.append(
                        FmhaKernelConfig(
                            family="bwd_dot_do_o",
                            data_type=dtype,
                            mode=mode,
                            hdim_q=hd,
                            hdim_v=hd,
                            pipeline="qr",
                            tile_m0=64,
                            pad_s=_pad_val(spec.spad),
                            pad_dv=_pad_val(spec.dvpad),
                            gfx_arch=arch,
                        )
                    )

        # dq_dk_dv — exhaustive tiles
        hdims = SUPPORTED_HDIMS.get(dtype, [])
        if restrict_hdims is not None:
            hdims = [hv for hv in hdims if hv in restrict_hdims]
        for hq, hv in hdims:
            tiles = generate_bwd_tiles(arch, dtype, hq, hv, apply_constraints=False)
            for tc in tiles:
                for mode, mask, bias, dropout, det in itertools.product(
                    modes,
                    masks,
                    biases,
                    dropout_vals,
                    deterministic_vals,
                ):
                    ww = bwd_dq_wave_warp((tc.bm0, tc.bn0, tc.bk0), hq)
                    configs.append(
                        FmhaKernelConfig(
                            family="bwd_dq_dk_dv",
                            data_type=dtype,
                            mode=mode,
                            hdim_q=hq,
                            hdim_v=hv,
                            pipeline="qr",
                            tile_m0=tc.bm0,
                            tile_n0=tc.bn0,
                            tile_k0=tc.bk0,
                            tile_n1=tc.bn1,
                            tile_k1=tc.bk1,
                            tile_k0max=tc.bk0max,
                            mask=mask,
                            bias=bias,
                            dropout=(dropout != "no"),
                            dropout_variant=dropout,
                            deterministic=det,
                            gfx_arch=arch,
                            **ww,
                        )
                    )

        # convert_dq — no tile sweep (fixed tile), just feature sweep
        for hd in BWD_CONVERT_DQ_HDIMS:
            if restrict_hdims is not None and (hd, hd) not in restrict_hdims:
                continue
            for mode, det in itertools.product(modes, deterministic_vals):
                configs.append(
                    FmhaKernelConfig(
                        family="bwd_convert_dq",
                        data_type=dtype,
                        mode=mode,
                        hdim_q=hd,
                        hdim_v=hd,
                        pipeline="qr",
                        tile_m0=64,
                        deterministic=det,
                        gfx_arch=arch,
                    )
                )
    return configs


def _expand_splitkv(
    arch,
    dtypes,
    receipt,
    allowed_masks,
    allowed_biases,
    allowed_modes,
    allowed_logits=None,
    allowed_sink=None,
    allowed_paged_kv=None,
    restrict_hdims=None,
):
    configs = []
    for dtype in dtypes:
        hdims = SUPPORTED_HDIMS.get(dtype, [])
        if restrict_hdims is not None:
            hdims = [hv for hv in hdims if hv in restrict_hdims]
        for hq, hv in hdims:
            tiles = generate_splitkv_tiles(arch, dtype, hq, hv)
            sk_specs = get_splitkv_pipelines(dtype, hq, receipt)
            for tc in tiles:
                for mode in MODES:
                    if allowed_modes is not None and mode not in allowed_modes:
                        continue
                    for spec in sk_specs:
                        if mode == "group" and not (
                            spec.spad == "t" and spec.skpad == "t"
                        ):
                            continue
                        mm = _MASK_MAP.get(spec.mask, spec.mask)
                        mb = _BIAS_MAP.get(spec.bias, spec.bias)
                        if allowed_masks is not None and mm not in allowed_masks:
                            continue
                        if allowed_biases is not None and mb not in allowed_biases:
                            continue
                        lgv = spec.logits == "t"
                        sv = spec.sink == "t"
                        pkv = spec.pagedkv == "t"
                        if allowed_logits is not None and lgv not in allowed_logits:
                            continue
                        if allowed_sink is not None and sv not in allowed_sink:
                            continue
                        if allowed_paged_kv is not None and pkv not in allowed_paged_kv:
                            continue
                        configs.append(
                            FmhaKernelConfig(
                                family="fwd_splitkv",
                                data_type=dtype,
                                mode=mode,
                                hdim_q=hq,
                                hdim_v=hv,
                                pipeline=spec.tag,
                                tile_m0=tc.bm0,
                                tile_n0=tc.bn0,
                                tile_k0=tc.bk0,
                                tile_n1=tc.bn1,
                                tile_k1=tc.bk1,
                                tile_k0max=tc.bk0max,
                                wave_m0=tc.rm0,
                                wave_n0=1,
                                wave_k0=1,
                                wave_m1=tc.rm0,
                                wave_n1=1,
                                wave_k1=1,
                                warp_m0=tc.wm0,
                                warp_n0=tc.wn0,
                                warp_k0=tc.wk0,
                                warp_m1=tc.wm1,
                                warp_n1=tc.wn1,
                                warp_k1=tc.wk1,
                                pad_s=_pad_val(spec.spad),
                                pad_sk=_pad_val(spec.skpad),
                                pad_d=_pad_val(spec.dpad),
                                pad_dv=_pad_val(spec.dvpad),
                                mask=mm,
                                bias=mb,
                                lse=True,
                                logits=lgv,
                                sink=sv,
                                paged_kv=pkv,
                                gfx_arch=arch,
                            )
                        )
    # Combine kernels
    for dtype in dtypes:
        comb_specs = get_splitkv_combine_pipelines(dtype, receipt)
        if not comb_specs:
            continue
        hdims = (
            SPLITKV_COMBINE_HDIMS_FP16
            if dtype in ("fp16", "bf16")
            else SPLITKV_COMBINE_HDIMS_FP8
            if dtype in ("fp8", "bf8")
            else []
        )
        for hv in hdims:
            for mode in MODES:
                if allowed_modes is not None and mode not in allowed_modes:
                    continue
                for spec in comb_specs:
                    if mode == "group" and spec.spad != "t":
                        continue
                    configs.append(
                        FmhaKernelConfig(
                            family="fwd_splitkv_combine",
                            data_type=dtype,
                            mode=mode,
                            hdim_q=hv,
                            hdim_v=hv,
                            pipeline="splitkv_combine",
                            tile_m0=32,
                            tile_n0=hv,
                            tile_k0=32,
                            tile_n1=32,
                            pad_s=_pad_val(spec.spad),
                            pad_dv=_pad_val(spec.dvpad),
                            lse=(spec.lse == "t"),
                            gfx_arch=arch,
                        )
                    )
    return configs


def _expand_pagedkv(
    arch,
    dtypes,
    receipt,
    allowed_masks,
    allowed_biases,
    allowed_modes,
    restrict_hdims=None,
):
    configs = []
    for dtype in dtypes:
        hdims = SUPPORTED_HDIMS.get(dtype, [])
        if restrict_hdims is not None:
            hdims = [hv for hv in hdims if hv in restrict_hdims]
        for hq, hv in hdims:
            tiles = generate_pagedkv_tiles(arch, dtype, hq, hv)
            pk_specs = get_pagedkv_pipelines(dtype, hq, receipt)
            for tc in tiles:
                for mode in MODES:
                    if allowed_modes is not None and mode not in allowed_modes:
                        continue
                    for spec in pk_specs:
                        if mode == "group" and not (
                            spec.spad == "t" and spec.skpad == "t"
                        ):
                            continue
                        mm = _MASK_MAP.get(spec.mask, spec.mask)
                        mb = _BIAS_MAP.get(spec.bias, spec.bias)
                        if allowed_masks is not None and mm not in allowed_masks:
                            continue
                        if allowed_biases is not None and mb not in allowed_biases:
                            continue
                        configs.append(
                            FmhaKernelConfig(
                                family="fwd_pagedkv",
                                data_type=dtype,
                                mode=mode,
                                hdim_q=hq,
                                hdim_v=hv,
                                pipeline=spec.tag,
                                tile_m0=tc.bm0,
                                tile_n0=tc.bn0,
                                tile_k0=tc.bk0,
                                tile_n1=tc.bn1,
                                tile_k1=tc.bk1,
                                tile_k0max=tc.bk0max,
                                wave_m0=tc.rm0,
                                wave_n0=1,
                                wave_k0=1,
                                wave_m1=tc.rm0,
                                wave_n1=1,
                                wave_k1=1,
                                warp_m0=tc.wm0,
                                warp_n0=tc.wn0,
                                warp_k0=tc.wk0,
                                warp_m1=tc.wm1,
                                warp_n1=tc.wn1,
                                warp_k1=tc.wk1,
                                pad_s=_pad_val(spec.spad),
                                pad_sk=_pad_val(spec.skpad),
                                pad_d=_pad_val(spec.dpad),
                                pad_dv=_pad_val(spec.dvpad),
                                mask=mm,
                                bias=mb,
                                logits=(spec.logits == "t"),
                                skip_min_seqlen_q=(spec.skip == "t"),
                                sink=(spec.sink == "t"),
                                paged_kv=True,
                                gfx_arch=arch,
                            )
                        )
    return configs


def _expand_appendkv(arch, dtypes, receipt, restrict_hdims=None):
    configs = []
    for dtype in dtypes:
        ak_specs = get_appendkv_pipelines(dtype, 0, receipt)
        hdims = SUPPORTED_HDIMS.get(dtype, [])
        if restrict_hdims is not None:
            hdims = [hv for hv in hdims if hv in restrict_hdims]
        for hq, hv in hdims:
            for spec in ak_specs:
                configs.append(
                    FmhaKernelConfig(
                        family="fwd_appendkv",
                        data_type=dtype,
                        mode="batch",
                        hdim_q=hq,
                        hdim_v=hv,
                        pipeline="appendkv",
                        tile_m0=64,
                        tile_n0=64,
                        tile_k0=hq,
                        tile_n1=hv,
                        pad_s=_pad_val(spec.spad),
                        pad_sk=_pad_val(spec.skpad),
                        pad_d=_pad_val(spec.dpad),
                        pad_dv=_pad_val(spec.dvpad),
                        rope={
                            "none": "none",
                            "interleaved": "interleaved",
                            "half_rotated": "half_rotated",
                        }.get(spec.rope, spec.rope),
                        paged_kv=(spec.pagedkv == "t"),
                        gfx_arch=arch,
                    )
                )
    return configs


def _expand_batch_prefill(
    arch, dtypes, receipt, allowed_masks, allowed_biases, restrict_hdims=None
):
    configs = []
    page_sizes = [1, 16, 1024]

    def _bp_bk1(bm0, bn0, bk0, hq):
        if bm0 == 64 and bn0 == 128 and bk0 == 64 and hq == 128:
            return 64
        return 32

    for dtype in dtypes:
        hdims = SUPPORTED_HDIMS.get(dtype, [])
        if restrict_hdims is not None:
            hdims = [hv for hv in hdims if hv in restrict_hdims]
        for hq, hv in hdims:
            tiles = generate_splitkv_tiles(arch, dtype, hq, hv)
            bp_specs = get_batch_prefill_pipelines(dtype, hq, receipt)
            for tc in tiles:
                bk1 = _bp_bk1(tc.bm0, tc.bn0, tc.bk0, hq)
                for spec in bp_specs:
                    mm = _MASK_MAP.get(spec.mask, spec.mask)
                    mb = _BIAS_MAP.get(spec.bias, spec.bias)
                    if allowed_masks is not None and mm not in allowed_masks:
                        continue
                    if allowed_biases is not None and mb not in allowed_biases:
                        continue
                    for ps in page_sizes:
                        if ps == 1 and spec.kv_memory_layout != "linear":
                            continue
                        if spec.qscale == "kv_blockscale" and ps < tc.bn0:
                            continue
                        configs.append(
                            FmhaKernelConfig(
                                family="batch_prefill",
                                data_type=dtype,
                                mode="group",
                                hdim_q=hq,
                                hdim_v=hv,
                                pipeline="qr_async",
                                tile_m0=tc.bm0,
                                tile_n0=tc.bn0,
                                tile_k0=tc.bk0,
                                tile_n1=tc.bn1,
                                tile_k1=bk1,
                                tile_k0max=tc.bk0max,
                                wave_m0=tc.rm0,
                                wave_n0=1,
                                wave_k0=1,
                                wave_m1=tc.rm0,
                                wave_n1=1,
                                wave_k1=1,
                                warp_m0=tc.wm0,
                                warp_n0=tc.wn0,
                                warp_k0=tc.wk0,
                                warp_m1=tc.wm1,
                                warp_n1=tc.wn1,
                                warp_k1=tc.wk1,
                                pad_s=1,
                                pad_sk=1,
                                pad_d=1,
                                pad_dv=1,
                                mask=mm,
                                bias=mb,
                                lse=(spec.lse == "t"),
                                dropout=(spec.dropout == "t"),
                                logits=(spec.logits == "t"),
                                paged_kv=True,
                                page_size=ps,
                                kv_memory_layout=spec.kv_memory_layout,
                                kv_lookup_table=spec.kv_lookup_table,
                                qscale=spec.qscale,
                                gfx_arch=arch,
                            )
                        )
    return configs


def _expand_bwd(
    arch,
    dtypes,
    receipt,
    allowed_masks,
    allowed_biases,
    allowed_modes,
    restrict_hdims=None,
):
    configs = []
    for dtype in dtypes:
        if dtype not in ("fp16", "bf16"):
            continue

        # dot_do_o
        dot_specs = get_bwd_dot_do_o_pipelines(dtype)
        for hd in BWD_DOT_DO_O_HDIMS:
            for mode in MODES:
                if allowed_modes is not None and mode not in allowed_modes:
                    continue
                for spec in dot_specs:
                    if mode == "group" and spec.spad != "t":
                        continue
                    configs.append(
                        FmhaKernelConfig(
                            family="bwd_dot_do_o",
                            data_type=dtype,
                            mode=mode,
                            hdim_q=hd,
                            hdim_v=hd,
                            pipeline="qr",
                            tile_m0=64,
                            pad_s=_pad_val(spec.spad),
                            pad_dv=_pad_val(spec.dvpad),
                            gfx_arch=arch,
                        )
                    )

        # dq_dk_dv: main tiles
        dq_specs = get_bwd_dq_dk_dv_pipelines(dtype, receipt)
        for (hq, hv), tile in sorted(BWD_DQ_DK_DV_TILES_FP16.items()):
            for mode in MODES:
                if allowed_modes is not None and mode not in allowed_modes:
                    continue
                for spec in dq_specs:
                    mm = _MASK_MAP.get(spec.mask, spec.mask)
                    mb = _BIAS_MAP.get(spec.bias, spec.bias)
                    if allowed_masks is not None and mm not in allowed_masks:
                        continue
                    if allowed_biases is not None and mb not in allowed_biases:
                        continue
                    ww = bwd_dq_wave_warp(tile, hq)
                    configs.append(
                        FmhaKernelConfig(
                            family="bwd_dq_dk_dv",
                            data_type=dtype,
                            mode=mode,
                            hdim_q=hq,
                            hdim_v=hv,
                            pipeline="qr",
                            tile_m0=tile[0],
                            tile_n0=tile[1],
                            tile_k0=tile[2],
                            tile_n1=tile[3] if len(tile) > 3 else hv,
                            tile_k1=tile[4] if len(tile) > 4 else tile[2],
                            tile_k0max=tile[5] if len(tile) > 5 else hq,
                            tile_bwd6=tile[6] if len(tile) > 6 else 0,
                            tile_bwd7=tile[7] if len(tile) > 7 else 0,
                            tile_bwd8=tile[8] if len(tile) > 8 else 0,
                            pad_s=_pad_val(spec.spad),
                            pad_sk=_pad_val(spec.skpad),
                            pad_d=_pad_val(spec.dpad),
                            pad_dv=_pad_val(spec.dvpad),
                            mask=mm,
                            bias=mb,
                            dbias=(spec.dbias == "t"),
                            dropout=(spec.dropout != "no"),
                            dropout_variant=spec.dropout,
                            deterministic=(spec.deterministic == "t"),
                            gfx_arch=arch,
                            **ww,
                        )
                    )

        # dq_dk_dv: extra tiles
        for (hq, hv), extra_entries in BWD_DQ_DK_DV_EXTRA_TILES.items():
            for tile, tag, is_batch_only in extra_entries:
                dq_extra_specs = get_bwd_dq_dk_dv_extra_pipelines(
                    dtype, is_small=is_batch_only, receipt=receipt
                )
                for mode in ["batch"] if is_batch_only else MODES:
                    if allowed_modes is not None and mode not in allowed_modes:
                        continue
                    for spec in dq_extra_specs:
                        mm = _MASK_MAP.get(spec.mask, spec.mask)
                        mb = _BIAS_MAP.get(spec.bias, spec.bias)
                        if allowed_masks is not None and mm not in allowed_masks:
                            continue
                        if allowed_biases is not None and mb not in allowed_biases:
                            continue
                        ww = bwd_dq_wave_warp(tile, hq, trload=(tag == "trload"))
                        configs.append(
                            FmhaKernelConfig(
                                family="bwd_dq_dk_dv",
                                data_type=dtype,
                                mode=mode,
                                hdim_q=hq,
                                hdim_v=hv,
                                pipeline="qr",
                                tile_m0=tile[0],
                                tile_n0=tile[1],
                                tile_k0=tile[2],
                                tile_n1=tile[3] if len(tile) > 3 else hv,
                                tile_k1=tile[4] if len(tile) > 4 else tile[2],
                                tile_k0max=tile[5] if len(tile) > 5 else hq,
                                tile_bwd6=tile[6] if len(tile) > 6 else 0,
                                tile_bwd7=tile[7] if len(tile) > 7 else 0,
                                tile_bwd8=tile[8] if len(tile) > 8 else 0,
                                tile_tag=tag,
                                use_trload=(tag == "trload"),
                                pad_s=_pad_val(spec.spad),
                                pad_sk=_pad_val(spec.skpad),
                                pad_d=_pad_val(spec.dpad),
                                pad_dv=_pad_val(spec.dvpad),
                                mask=mm,
                                bias=mb,
                                dbias=(spec.dbias == "t"),
                                dropout=(spec.dropout != "no"),
                                dropout_variant=spec.dropout,
                                deterministic=(spec.deterministic == "t"),
                                gfx_arch=arch,
                                **ww,
                            )
                        )

        # convert_dq
        for hd in BWD_CONVERT_DQ_HDIMS:
            cvt_specs = get_bwd_convert_dq_pipelines(dtype, hd)
            n_tile_groups = BWD_CONVERT_DQ_TILE_GROUPS.get(hd, 1)
            for mode in MODES:
                if allowed_modes is not None and mode not in allowed_modes:
                    continue
                for spec in cvt_specs:
                    if mode == "group" and spec.spad != "t":
                        continue
                    for tile_grp in range(n_tile_groups):
                        configs.append(
                            FmhaKernelConfig(
                                family="bwd_convert_dq",
                                data_type=dtype,
                                mode=mode,
                                hdim_q=hd,
                                hdim_v=hd,
                                pipeline="qr",
                                tile_m0=64,
                                tile_tag=f"g{tile_grp}" if tile_grp > 0 else "",
                                pad_s=_pad_val(spec.spad),
                                pad_d=_pad_val(spec.dpad),
                                deterministic=(spec.deterministic == "t"),
                                gfx_arch=arch,
                            )
                        )
    return configs


# =============================================================================
# Filter utility
# =============================================================================


def apply_filter(
    configs: List[FmhaKernelConfig], expr: str = "", filter_file: str = ""
) -> List[FmhaKernelConfig]:
    """Apply user-defined filters to a config list."""
    result = configs

    if filter_file:
        import importlib.util

        spec = importlib.util.spec_from_file_location("user_filter", filter_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, "filter_config")
        result = [c for c in result if fn(c)]

    if expr:
        result = [c for c in result if eval(expr, {"c": c})]  # noqa: S307

    return result


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="FMHA instance enumeration")
    parser.add_argument("config", help="Sweep config JSON")
    parser.add_argument("--arch", default="gfx950")
    parser.add_argument("--receipt", type=int, default=0)
    parser.add_argument(
        "--filter",
        dest="filter_expr",
        default="",
        help='Python expression per config, e.g. "c.hdim_q == 128"',
    )
    parser.add_argument(
        "--filter-file",
        default="",
        help="Path to .py file with filter_config(c) -> bool",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--count-only", action="store_true")
    args = parser.parse_args()

    configs = expand_sweep(args.config, args.arch, args.receipt)
    before = len(configs)
    configs = apply_filter(configs, args.filter_expr, args.filter_file)
    filtered = before - len(configs)

    print(
        f"Expanded {args.config} -> {before} configs"
        f"{f' (filtered {filtered}, kept {len(configs)})' if filtered else ''}"
    )

    if args.count_only:
        return

    if args.list:
        for i, c in enumerate(configs):
            print(f"  [{i}] {c.name}")


if __name__ == "__main__":
    main()
