#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Build example kernels - generates and compiles kernels for a single example.

Detects if example is GEMM or Conv based on macro presence, extracts all
configuration parameters, and generates appropriate kernels.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple


def find_hipcc() -> str:
    for path in [os.environ.get("HIPCC"), "/opt/rocm/bin/hipcc", shutil.which("hipcc")]:
        if path and os.path.isfile(path):
            return path
    return "hipcc"


def find_ar() -> str:
    for path in [
        "/opt/rocm/llvm/bin/llvm-ar",
        shutil.which("llvm-ar"),
        shutil.which("ar"),
    ]:
        if path and os.path.isfile(path):
            return path
    return "ar"


def extract_balanced_parens(text: str, start_pos: int) -> str:
    """Extract content between balanced parentheses."""
    if start_pos >= len(text) or text[start_pos] != "(":
        return ""
    depth = 0
    for i, c in enumerate(text[start_pos:], start_pos):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return text[start_pos + 1 : i]
    return ""


def parse_conv_declarations(content: str) -> List[Dict]:
    """Parse DECL_GROUPED_CONV_KERNEL_SET declarations with all parameters."""
    kernels = []

    for match in re.finditer(r"DECL_GROUPED_CONV_KERNEL_SET\s*\(", content):
        body = extract_balanced_parens(content, match.end() - 1)
        if not body:
            continue

        # Parse each .add() call
        for add_match in re.finditer(r"\.add\s*\(", body):
            add_body = extract_balanced_parens(body, add_match.end() - 1)

            kernel = {}

            # ConvSig parameters - handle both single dtype and multi-dtype
            # Multi-dtype: .dtype("fp16", "fp16", "fp16", "fp32") or .dtype("fp16", "bf16", "fp16")
            if m := re.search(
                r'\.dtype\s*\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"(?:\s*,\s*"([^"]+)")?\s*\)',
                add_body,
            ):
                kernel["dtype_in"] = m.group(1)
                kernel["dtype_wei"] = m.group(2)
                kernel["dtype_out"] = m.group(3)
                kernel["dtype_acc"] = m.group(4) if m.group(4) else "fp32"
                kernel["dtype"] = m.group(1)  # Default for codegen
            # Single dtype: .dtype("fp16")
            elif m := re.search(r'\.dtype\s*\(\s*"([^"]+)"\s*\)', add_body):
                kernel["dtype"] = m.group(1)
                kernel["dtype_in"] = m.group(1)
                kernel["dtype_wei"] = m.group(1)
                kernel["dtype_out"] = m.group(1)
                kernel["dtype_acc"] = "fp32"
            if m := re.search(r'\.layout\s*\(\s*"([^"]+)"', add_body):
                kernel["layout"] = m.group(1)
            if m := re.search(r'\.conv_type\s*\(\s*"([^"]+)"', add_body):
                kernel["conv_type"] = m.group(1)
            if m := re.search(r"\.dims\s*\(\s*(\d+)\s*\)", add_body):
                kernel["ndim"] = int(m.group(1))

            # ConvAlgo parameters - tile(G, M, N) where G=batch, M=output, N=reduction
            if m := re.search(
                r"\.tile\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", add_body
            ):
                kernel["tile_g"] = int(m.group(1))  # batch tile (usually 1)
                kernel["tile_m"] = int(m.group(2))  # output channel tile
                kernel["tile_n"] = int(m.group(3))  # input channel tile (reduction)

            # wave(M_Warp, N_Warp, K_Warp) - warp distribution
            if m := re.search(
                r"\.wave\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", add_body
            ):
                kernel["warp_m"] = int(m.group(1))
                kernel["warp_n"] = int(m.group(2))
                kernel["warp_k"] = int(m.group(3))

            # warp(M_Warp_Tile, N_Warp_Tile, K_Warp_Tile) - warp tile sizes
            if m := re.search(
                r"\.warp\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", add_body
            ):
                kernel["warp_tile_m"] = int(m.group(1))
                kernel["warp_tile_n"] = int(m.group(2))
                kernel["warp_tile_k"] = int(m.group(3))

            # vector_sizes(A, B, C)
            if m := re.search(
                r"\.vector_sizes\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", add_body
            ):
                kernel["vector_a"] = int(m.group(1))
                kernel["vector_b"] = int(m.group(2))
                kernel["vector_c"] = int(m.group(3))

            # Single-value parameters
            if m := re.search(r'\.pipeline\s*\(\s*"([^"]+)"', add_body):
                kernel["pipeline"] = m.group(1)
            if m := re.search(r'\.scheduler\s*\(\s*"([^"]+)"', add_body):
                kernel["scheduler"] = m.group(1)
            if m := re.search(r'\.epilogue\s*\(\s*"([^"]+)"', add_body):
                kernel["epilogue"] = m.group(1)
            if m := re.search(r"\.block_per_cu\s*\(\s*(\d+)\s*\)", add_body):
                kernel["block_per_cu"] = int(m.group(1))
            if m := re.search(r"\.num_wave_groups\s*\(\s*(\d+)\s*\)", add_body):
                kernel["num_wave_groups"] = int(m.group(1))
            if m := re.search(r"\.num_groups_to_merge\s*\(\s*(\d+)\s*\)", add_body):
                kernel["num_groups_to_merge"] = int(m.group(1))
            if m := re.search(
                r"\.double_smem_buffer\s*\(\s*(true|false)\s*\)", add_body, re.I
            ):
                kernel["double_smem_buffer"] = m.group(1).lower() == "true"

            # Architecture
            if m := re.search(r'"(gfx\d+)"', add_body):
                kernel["arch"] = m.group(1)

            if kernel.get("dtype"):
                # Auto-fill missing parameters with defaults (autocorrect)
                kernel = auto_fill_conv_defaults(kernel)
                kernels.append(kernel)

    return kernels


def parse_fmha_declarations(content: str) -> List[Dict]:
    """Parse DECL_FMHA_KERNEL_SET declarations into config-json-ready dicts."""
    kernels = []

    def parse_bool(value: str) -> bool:
        return value.strip().lower() == "true"

    def parse_int_list(match_text: str) -> List[int]:
        return [int(v.strip()) for v in match_text.split(",") if v.strip()]

    for match in re.finditer(r"DECL_FMHA_KERNEL_SET\s*\(", content):
        body = extract_balanced_parens(content, match.end() - 1)
        if not body:
            continue

        for add_match in re.finditer(r"\.add\s*\(", body):
            add_body = extract_balanced_parens(body, add_match.end() - 1)
            if not add_body:
                continue

            sig = {
                "family": "fwd",
                "data_type": "fp16",
                "mode": "batch",
                "vlayout": "r",
                "hdim_q": 128,
                "hdim_v": 128,
                "mask": "no",
                "bias": "no",
                "lse": False,
                "dropout": False,
                "qscale": "no",
                "rope": "none",
                "logits": False,
                "paged_kv": False,
                "fp8_static_quant": False,
                "skip_min_seqlen_q": False,
                "sink": False,
                "dbias": False,
                "store_randval": False,
                "deterministic": False,
                "kv_memory_layout": "vectorized",
                "kv_lookup_table": "sglang",
                "page_size": 1,
            }
            profile = None
            receipt = None
            alg = {
                "pipeline": "qr",
                "tile": [128, 64, 32, 128, 32, 128],
                "wave": [2, 2, 1, 2, 2, 1, 1, 1, 1],
                "warp": [32, 32, 16, 32, 32, 16, 16, 16, 16],
                "padding": [True, True, True, True],
                "use_trload": False,
                "hdim_q_alignment": 128,
                "hdim_v_alignment": 128,
                "block_per_cu": 1,
                "num_wave_groups": 1,
                "max_splits_log2": 0,
                "max_seq_len_q": 0,
                "selection_rank": 0,
                "constraint_tag": "",
            }

            if m := re.search(r'\.family\s*\(\s*"([^"]+)"\s*\)', add_body):
                sig["family"] = m.group(1)
            if m := re.search(r'\.dtype\s*\(\s*"([^"]+)"\s*\)', add_body):
                sig["data_type"] = m.group(1)
            if m := re.search(r'\.mode\s*\(\s*"([^"]+)"\s*\)', add_body):
                sig["mode"] = m.group(1)
            if m := re.search(r'\.vlayout\s*\(\s*"([^"]+)"\s*\)', add_body):
                sig["vlayout"] = m.group(1)
            if m := re.search(r"\.hdim\s*\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?\)", add_body):
                sig["hdim_q"] = int(m.group(1))
                sig["hdim_v"] = int(m.group(2)) if m.group(2) else int(m.group(1))
            if m := re.search(r'\.mask\s*\(\s*"([^"]+)"\s*\)', add_body):
                sig["mask"] = m.group(1)
            if m := re.search(r'\.bias\s*\(\s*"([^"]+)"\s*\)', add_body):
                sig["bias"] = m.group(1)
            if m := re.search(r"\.lse\s*\(\s*(true|false)\s*\)", add_body, re.I):
                sig["lse"] = parse_bool(m.group(1))
            if m := re.search(r"\.dropout\s*\(\s*(true|false)\s*\)", add_body, re.I):
                sig["dropout"] = parse_bool(m.group(1))
            if m := re.search(r'\.qscale\s*\(\s*"([^"]+)"\s*\)', add_body):
                sig["qscale"] = m.group(1)
            if m := re.search(r'\.rope\s*\(\s*"([^"]+)"\s*\)', add_body):
                sig["rope"] = m.group(1)
            if m := re.search(r"\.logits\s*\(\s*(true|false)\s*\)", add_body, re.I):
                sig["logits"] = parse_bool(m.group(1))
            if m := re.search(r"\.paged_kv\s*\(\s*(true|false)\s*\)", add_body, re.I):
                sig["paged_kv"] = parse_bool(m.group(1))
            if m := re.search(
                r"\.fp8_static_quant\s*\(\s*(true|false)\s*\)", add_body, re.I
            ):
                sig["fp8_static_quant"] = parse_bool(m.group(1))
            if m := re.search(r"\.skip\s*\(\s*(true|false)\s*\)", add_body, re.I):
                sig["skip_min_seqlen_q"] = parse_bool(m.group(1))
            if m := re.search(r"\.sink\s*\(\s*(true|false)\s*\)", add_body, re.I):
                sig["sink"] = parse_bool(m.group(1))
            if m := re.search(r"\.dbias\s*\(\s*(true|false)\s*\)", add_body, re.I):
                sig["dbias"] = parse_bool(m.group(1))
            if m := re.search(
                r"\.store_randval\s*\(\s*(true|false)\s*\)", add_body, re.I
            ):
                sig["store_randval"] = parse_bool(m.group(1))
            if m := re.search(
                r"\.deterministic\s*\(\s*(true|false)\s*\)", add_body, re.I
            ):
                sig["deterministic"] = parse_bool(m.group(1))
            if m := re.search(
                r'\.kv_cache\s*\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*(?:,\s*(\d+)\s*)?\)',
                add_body,
            ):
                sig["kv_memory_layout"] = m.group(1)
                sig["kv_lookup_table"] = m.group(2)
                sig["page_size"] = int(m.group(3)) if m.group(3) else 1
            if m := re.search(r'\.profile\s*\(\s*"([^"]+)"\s*\)', add_body):
                profile = m.group(1)
            if m := re.search(r"\.receipt\s*\(\s*(\d+)\s*\)", add_body):
                receipt = int(m.group(1))

            # Tile: bulk .tile(m0,n0,k0,n1,k1,k0max) or named .tile_m0(v)...
            if m := re.search(
                r"\.tile\s*\(\s*([0-9,\s]+)\)",
                add_body,
            ):
                values = parse_int_list(m.group(1))
                if len(values) == 6:
                    alg["tile"] = values
            for field_idx, field_name in enumerate(
                ["tile_m0", "tile_n0", "tile_k0", "tile_n1", "tile_k1", "tile_k0max"]
            ):
                if m := re.search(rf"\.{field_name}\s*\(\s*(\d+)\s*\)", add_body):
                    alg["tile"][field_idx] = int(m.group(1))

            # Wave: bulk .wave(m0,n0,k0,...) or named .wave_m0(v)...
            if m := re.search(r"\.wave\s*\(\s*([0-9,\s]+)\)", add_body):
                values = parse_int_list(m.group(1))
                if len(values) == 3:
                    values += [2, 2, 1, 1, 1, 1]
                elif len(values) == 6:
                    values += [1, 1, 1]
                if len(values) == 9:
                    alg["wave"] = values
            for field_idx, field_name in enumerate(
                [
                    "wave_m0",
                    "wave_n0",
                    "wave_k0",
                    "wave_m1",
                    "wave_n1",
                    "wave_k1",
                    "wave_m2",
                    "wave_n2",
                    "wave_k2",
                ]
            ):
                if m := re.search(rf"\.{field_name}\s*\(\s*(\d+)\s*\)", add_body):
                    alg["wave"][field_idx] = int(m.group(1))

            # Warp: bulk .warp(m0,n0,k0,...) or named .warp_m0(v)...
            if m := re.search(r"\.warp\s*\(\s*([0-9,\s]+)\)", add_body):
                values = parse_int_list(m.group(1))
                if len(values) == 3:
                    values += [32, 32, 16, 16, 16, 16]
                elif len(values) == 6:
                    values += [16, 16, 16]
                if len(values) == 9:
                    alg["warp"] = values
            for field_idx, field_name in enumerate(
                [
                    "warp_m0",
                    "warp_n0",
                    "warp_k0",
                    "warp_m1",
                    "warp_n1",
                    "warp_k1",
                    "warp_m2",
                    "warp_n2",
                    "warp_k2",
                ]
            ):
                if m := re.search(rf"\.{field_name}\s*\(\s*(\d+)\s*\)", add_body):
                    alg["warp"][field_idx] = int(m.group(1))
            if m := re.search(r'\.pipeline\s*\(\s*"([^"]+)"\s*\)', add_body):
                alg["pipeline"] = m.group(1)
            if m := re.search(
                r"\.padding\s*\(\s*(true|false)\s*,\s*(true|false)\s*,\s*(true|false)\s*,\s*(true|false)\s*\)",
                add_body,
                re.I,
            ):
                alg["padding"] = [parse_bool(m.group(i)) for i in range(1, 5)]
            if m := re.search(r"\.trload\s*\(\s*(true|false)\s*\)", add_body, re.I):
                alg["use_trload"] = parse_bool(m.group(1))
            if m := re.search(r"\.alignments\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", add_body):
                alg["hdim_q_alignment"] = int(m.group(1))
                alg["hdim_v_alignment"] = int(m.group(2))
            if m := re.search(r"\.block_per_cu\s*\(\s*(\d+)\s*\)", add_body):
                alg["block_per_cu"] = int(m.group(1))
            if m := re.search(r"\.num_wave_groups\s*\(\s*(\d+)\s*\)", add_body):
                alg["num_wave_groups"] = int(m.group(1))
            if m := re.search(r"\.max_splits_log2\s*\(\s*(\d+)\s*\)", add_body):
                alg["max_splits_log2"] = int(m.group(1))
            if m := re.search(r"\.max_seq_len_q\s*\(\s*(\d+)\s*\)", add_body):
                alg["max_seq_len_q"] = int(m.group(1))
            if m := re.search(r"\.selection_rank\s*\(\s*(\d+)\s*\)", add_body):
                alg["selection_rank"] = int(m.group(1))
            if m := re.search(r'\.constraint\s*\(\s*"([^"]+)"\s*\)', add_body):
                alg["constraint_tag"] = m.group(1)

            arch = "gfx942"
            if m := re.search(r'"(gfx\d+)"', add_body):
                arch = m.group(1)

            entry = {"arch": arch, "signature": sig, "algorithm": alg}
            if profile is not None:
                entry["profile"] = profile
            if receipt is not None:
                entry["receipt"] = receipt
            kernels.append(entry)

    return kernels


def auto_fill_conv_defaults(kernel: Dict) -> Dict:
    """Auto-fill missing conv parameters with sensible defaults (autofill + autocorrect).

    This implements:
    1. AUTOFILL: Missing parameters are filled with valid defaults (ConvConfigComputeV3)
    2. AUTOCORRECT: Invalid values are corrected to valid ones
    """
    # Default tile configuration matching ConvConfigComputeV3
    defaults = {
        "tile_g": 1,
        "tile_m": 16,
        "tile_n": 64,
        "warp_m": 1,
        "warp_n": 4,
        "warp_k": 1,
        "warp_tile_m": 16,
        "warp_tile_n": 16,
        "warp_tile_k": 32,
        "pipeline": "compv3",
        "scheduler": "intrawave",
        "epilogue": "cshuffle",
        "vector_a": 4,
        "vector_b": 8,
        "vector_c": 8,
        "block_per_cu": 1,
        "num_wave_groups": 1,
        "num_groups_to_merge": 1,
        "ndim": 2,
        "layout": "nhwgc",
        "conv_type": "forward",
        "arch": "gfx942",
    }

    # AUTOFILL: Fill missing parameters with defaults
    autofilled = []
    for key, value in defaults.items():
        if key not in kernel or kernel[key] is None or kernel[key] == -1:
            kernel[key] = value
            autofilled.append(f"{key}={value}")

    if autofilled:
        print(f"    [AUTOFILL] {', '.join(autofilled)}")

    # AUTOCORRECT: Fix invalid wave configurations for gfx942
    valid_wave_configs = [(1, 4, 1), (2, 2, 1), (4, 1, 1)]
    current_wave = (
        kernel.get("warp_m", 1),
        kernel.get("warp_n", 4),
        kernel.get("warp_k", 1),
    )

    if current_wave not in valid_wave_configs:
        old = current_wave
        kernel["warp_m"] = 1
        kernel["warp_n"] = 4
        kernel["warp_k"] = 1
        print(f"    [AUTOCORRECT] wave{old} -> wave(1,4,1) (invalid for gfx942)")

    # AUTOCORRECT: Fix invalid pipeline for backward ops
    conv_type = kernel.get("conv_type", "forward")
    pipeline = kernel.get("pipeline", "compv3")

    if conv_type in ["bwd_data", "bwd_weight"] and pipeline in ["compv4", "compv5"]:
        old_pipeline = pipeline
        kernel["pipeline"] = "compv3"
        print(
            f"    [AUTOCORRECT] pipeline {old_pipeline} -> compv3 (invalid for {conv_type})"
        )

    return kernel


def expand_conv_wildcards(kernel: Dict, arch: str = "gfx942") -> List[Dict]:
    """Expand wildcard parameters to multiple valid configurations.

    When users specify wildcards (-1 or *), this expands them to all
    valid configurations for the target architecture.
    """
    expanded = []

    # Valid wave configurations for gfx942
    valid_wave_configs = [(1, 4, 1), (2, 2, 1), (4, 1, 1)]

    # Valid warp tile configurations for gfx942 fp16
    valid_warp_configs = [(16, 16, 32), (32, 32, 16)]

    # Check if expansion is needed
    needs_wave = kernel.get("warp_m") is None or kernel.get("warp_m") == -1
    needs_warp = kernel.get("warp_tile_m") is None or kernel.get("warp_tile_m") == -1

    if not needs_wave and not needs_warp:
        return [kernel]

    # Expand wave configurations
    wave_configs = (
        valid_wave_configs
        if needs_wave
        else [
            (kernel.get("warp_m", 2), kernel.get("warp_n", 2), kernel.get("warp_k", 1))
        ]
    )

    # Expand warp tile configurations
    warp_configs = (
        valid_warp_configs
        if needs_warp
        else [
            (
                kernel.get("warp_tile_m", 32),
                kernel.get("warp_tile_n", 32),
                kernel.get("warp_tile_k", 16),
            )
        ]
    )

    for wm, wn, wk in wave_configs:
        for wtm, wtn, wtk in warp_configs:
            new_kernel = kernel.copy()
            new_kernel["warp_m"] = wm
            new_kernel["warp_n"] = wn
            new_kernel["warp_k"] = wk
            new_kernel["warp_tile_m"] = wtm
            new_kernel["warp_tile_n"] = wtn
            new_kernel["warp_tile_k"] = wtk
            expanded.append(new_kernel)

    return expanded


def parse_int_or_wildcard(val: str) -> int:
    """Parse integer or return -1 for wildcards.

    Supported wildcard formats:
    - ANY_INT: Macro defined as -1
    - -1: Direct numeric wildcard
    - "*": String wildcard (also maps to -1 for integer params)
    """
    val = val.strip()
    if val == "ANY_INT" or val == "-1" or val == "*":
        return -1
    return int(val)


def parse_gemm_declarations(content: str) -> List[Dict]:
    """Parse DECL_KERNEL_SET declarations for GEMM.

    Supports wildcards:
    - ANY_INT for numeric params (wave, warp) -> expands to all valid combos
    - "*" for string params (pipeline, scheduler) -> expands to valid options

    Each kernel is tagged with its kernel_set name for separate registration.
    """
    kernels = []

    for match in re.finditer(r"DECL_KERNEL_SET\s*\(\s*(\w+)\s*,", content):
        kernel_set_name = match.group(1)
        body = extract_balanced_parens(
            content, match.start() + content[match.start() :].find("(")
        )
        if not body:
            continue

        for add_match in re.finditer(r"\.add\s*\(", body):
            add_body = extract_balanced_parens(body, add_match.end() - 1)

            kernel = {}

            # Signature parameters
            if m := re.search(r'\.dtype\s*\(\s*"([^"]+)"', add_body):
                kernel["dtype"] = m.group(1)
            if m := re.search(r'\.layout\s*\(\s*"([^"]+)"', add_body):
                kernel["layout"] = m.group(1)
            if m := re.search(r'\.elementwise\s*\(\s*"([^"]+)"\s*,\s*(\d+)', add_body):
                kernel["elementwise_op"] = m.group(1)
                kernel["num_d_tensors"] = int(m.group(2))

            # Algorithm parameters - support ANY_INT wildcard
            if m := re.search(
                r"\.tile\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", add_body
            ):
                kernel["tile_m"] = int(m.group(1))
                kernel["tile_n"] = int(m.group(2))
                kernel["tile_k"] = int(m.group(3))

            # Wave: support ANY_INT, -1, and "*" as wildcards
            if m := re.search(
                r"\.wave\s*\(\s*([\w*-]+)\s*,\s*([\w*-]+)\s*,\s*([\w*-]+)\s*\)",
                add_body,
            ):
                kernel["warp_m"] = parse_int_or_wildcard(m.group(1))
                kernel["warp_n"] = parse_int_or_wildcard(m.group(2))
                kernel["warp_k"] = parse_int_or_wildcard(m.group(3))

            # Warp: support ANY_INT, -1, and "*" as wildcards
            if m := re.search(
                r"\.warp\s*\(\s*([\w*-]+)\s*,\s*([\w*-]+)\s*,\s*([\w*-]+)\s*\)",
                add_body,
            ):
                kernel["warp_tile_m"] = parse_int_or_wildcard(m.group(1))
                kernel["warp_tile_n"] = parse_int_or_wildcard(m.group(2))
                kernel["warp_tile_k"] = parse_int_or_wildcard(m.group(3))

            # Pipeline/Scheduler: support "*" wildcard
            if m := re.search(r'\.pipeline\s*\(\s*"([^"]+)"', add_body):
                kernel["pipeline"] = m.group(1)
            if m := re.search(r'\.scheduler\s*\(\s*"([^"]+)"', add_body):
                kernel["scheduler"] = m.group(1)
            if m := re.search(r'\.epilogue\s*\(\s*"([^"]+)"', add_body):
                kernel["epilogue"] = m.group(1)
            if m := re.search(
                r"\.pad\s*\(\s*(true|false)\s*,\s*(true|false)\s*,\s*(true|false)",
                add_body,
                re.I,
            ):
                kernel["pad_m"] = m.group(1).lower() == "true"
                kernel["pad_n"] = m.group(2).lower() == "true"
                kernel["pad_k"] = m.group(3).lower() == "true"

            # Shorthand format: .add("dtype", "layout", M, N, K)
            if not kernel.get("dtype"):
                if m := re.match(
                    r'\s*"(\w+)"\s*,\s*"(\w+)"\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)',
                    add_body,
                ):
                    kernel["dtype"] = m.group(1)
                    kernel["layout"] = m.group(2)
                    kernel["tile_m"] = int(m.group(3))
                    kernel["tile_n"] = int(m.group(4))
                    kernel["tile_k"] = int(m.group(5))

            if kernel.get("dtype"):
                kernel["kernel_set"] = kernel_set_name
                kernels.append(kernel)

    # Expand wildcards to multiple kernels
    expanded = []
    for kernel in kernels:
        expanded.extend(expand_gemm_wildcards(kernel))

    # Apply autocorrect to each expanded kernel
    return [auto_fill_gemm_defaults(k) for k in expanded]


def expand_gemm_wildcards(kernel: Dict, arch: str = "gfx942") -> List[Dict]:
    """Expand wildcard parameters to multiple valid configurations.

    When users specify ANY_INT (-1) or "*", this expands them to all
    valid configurations for the target architecture.

    Note: Block size constraint filters invalid combos:
    - (tile_m/warp_tile_m) * (tile_n/warp_tile_n) * 64 <= 1024
    - For 128x128 tile: only (32,32,k) works (16 warps * 64 = 1024)
    - For 64x64 tile: both (16,16,k) and (32,32,k) work
    """
    # Valid wave configurations for gfx942
    valid_wave_configs = [(1, 4, 1), (2, 2, 1), (4, 1, 1)]

    # Valid warp tile configurations for gfx942 fp16
    valid_warp_configs = [(16, 16, 32), (32, 32, 16)]

    # Valid pipelines and schedulers
    valid_pipelines = ["compv3"]  # compv4 requires special handling
    valid_schedulers = ["intrawave"]

    # Check what needs expansion
    needs_wave = kernel.get("warp_m") == -1
    needs_warp = kernel.get("warp_tile_m") == -1
    needs_pipeline = kernel.get("pipeline") == "*"
    needs_scheduler = kernel.get("scheduler") == "*"

    if not any([needs_wave, needs_warp, needs_pipeline, needs_scheduler]):
        return [kernel]

    # Determine configs to iterate
    wave_configs = (
        valid_wave_configs
        if needs_wave
        else [
            (kernel.get("warp_m", 2), kernel.get("warp_n", 2), kernel.get("warp_k", 1))
        ]
    )
    warp_configs = (
        valid_warp_configs
        if needs_warp
        else [
            (
                kernel.get("warp_tile_m", 32),
                kernel.get("warp_tile_n", 32),
                kernel.get("warp_tile_k", 16),
            )
        ]
    )
    pipelines = (
        valid_pipelines if needs_pipeline else [kernel.get("pipeline", "compv3")]
    )
    schedulers = (
        valid_schedulers if needs_scheduler else [kernel.get("scheduler", "intrawave")]
    )

    expanded = []
    for wm, wn, wk in wave_configs:
        for wtm, wtn, wtk in warp_configs:
            # Check block size constraint: (tile_m/warp_tile_m) * (tile_n/warp_tile_n) * 64 <= 1024
            tile_m = kernel.get("tile_m", 128)
            tile_n = kernel.get("tile_n", 128)
            num_warps = (tile_m // wtm) * (tile_n // wtn)
            if num_warps * 64 > 1024:
                continue  # Skip invalid config

            for pipe in pipelines:
                for sched in schedulers:
                    new_kernel = kernel.copy()
                    new_kernel["warp_m"] = wm
                    new_kernel["warp_n"] = wn
                    new_kernel["warp_k"] = wk
                    new_kernel["warp_tile_m"] = wtm
                    new_kernel["warp_tile_n"] = wtn
                    new_kernel["warp_tile_k"] = wtk
                    new_kernel["pipeline"] = pipe
                    new_kernel["scheduler"] = sched
                    expanded.append(new_kernel)

    if expanded:
        print(f"    [WILDCARD] Expanded 1 declaration -> {len(expanded)} kernel(s)")

    return expanded if expanded else [kernel]


def auto_fill_gemm_defaults(kernel: Dict) -> Dict:
    """Auto-fill missing GEMM parameters with sensible defaults (autofill + autocorrect).

    This implements:
    1. AUTOFILL: Missing parameters are filled with valid defaults
    2. AUTOCORRECT: Invalid values are corrected to valid ones (e.g., wave(1,1,1) -> wave(2,2,1))
    """
    defaults = {
        "tile_m": 128,
        "tile_n": 128,
        "tile_k": 64,
        "warp_m": 2,
        "warp_n": 2,
        "warp_k": 1,
        "warp_tile_m": 32,
        "warp_tile_n": 32,
        "warp_tile_k": 16,
        "pipeline": "compv3",
        "scheduler": "intrawave",
        "epilogue": "cshuffle",
        "pad_m": False,
        "pad_n": False,
        "pad_k": False,
        "layout": "rcr",
    }

    # AUTOFILL: Fill missing parameters with defaults
    autofilled = []
    for key, value in defaults.items():
        if key not in kernel or kernel[key] is None or kernel[key] == -1:
            kernel[key] = value
            autofilled.append(f"{key}={value}")

    if autofilled:
        print(f"    [AUTOFILL] {', '.join(autofilled)}")

    # AUTOCORRECT: Fix invalid wave configurations for gfx942
    # Valid wave configs: (1,4,1), (2,2,1), (4,1,1)
    valid_wave_configs = [(1, 4, 1), (2, 2, 1), (4, 1, 1)]
    current_wave = (
        kernel.get("warp_m", 2),
        kernel.get("warp_n", 2),
        kernel.get("warp_k", 1),
    )

    if current_wave not in valid_wave_configs:
        # Correct to (2,2,1) which is a balanced default
        old = current_wave
        kernel["warp_m"] = 2
        kernel["warp_n"] = 2
        kernel["warp_k"] = 1
        print(f"    [AUTOCORRECT] wave{old} -> wave(2,2,1) (invalid for gfx942)")

    # AUTOCORRECT: Fix invalid pipeline/scheduler combinations
    invalid_combos = [
        ("compv3", "interwave"),
        ("compv4", "interwave"),
    ]
    current_combo = (
        kernel.get("pipeline", "compv3"),
        kernel.get("scheduler", "intrawave"),
    )
    if current_combo in invalid_combos:
        old = current_combo
        kernel["scheduler"] = "intrawave"
        print(
            f"    [AUTOCORRECT] {old[0]}/{old[1]} -> {old[0]}/intrawave (invalid combo)"
        )

    # AUTOCORRECT: Fix warp tile to avoid exceeding max block size (1024 threads)
    # Block size = (tile_m / warp_tile_m) * (tile_n / warp_tile_n) * 64
    tile_m = kernel.get("tile_m", 128)
    tile_n = kernel.get("tile_n", 128)
    warp_tile_m = kernel.get("warp_tile_m", 32)
    warp_tile_n = kernel.get("warp_tile_n", 32)

    num_warps = (tile_m // warp_tile_m) * (tile_n // warp_tile_n)
    block_size = num_warps * 64  # 64 threads per warp

    if block_size > 1024:
        # Find valid warp tile that fits
        old_warp = (warp_tile_m, warp_tile_n, kernel.get("warp_tile_k", 16))

        # For large tiles, use larger warp tiles
        if tile_m >= 256:
            kernel["warp_tile_m"] = 64
        if tile_n >= 256:
            kernel["warp_tile_n"] = 64

        # Recalculate
        num_warps = (tile_m // kernel["warp_tile_m"]) * (
            tile_n // kernel["warp_tile_n"]
        )
        block_size = num_warps * 64

        if block_size <= 1024:
            new_warp = (
                kernel["warp_tile_m"],
                kernel["warp_tile_n"],
                kernel["warp_tile_k"],
            )
            print(
                f"    [AUTOCORRECT] warp{old_warp} -> warp{new_warp} (block_size={block_size})"
            )
        else:
            # Still too large, try even larger warp tiles
            kernel["warp_tile_m"] = tile_m // 4
            kernel["warp_tile_n"] = tile_n // 4
            new_warp = (
                kernel["warp_tile_m"],
                kernel["warp_tile_n"],
                kernel["warp_tile_k"],
            )
            print(
                f"    [AUTOCORRECT] warp{old_warp} -> warp{new_warp} (block_size adjusted)"
            )

    return kernel


def strip_cpp_strings_and_comments(content: str) -> str:
    """Strip C++ string literals and comments that could cause false positives.

    Only strips:
    - Comments (// and /* */) - always stripped
    - Raw string literals (R"...") - always stripped (can contain anything)
    - Regular strings ONLY if they contain problematic patterns like DECL_KERNEL_SET

    Preserves normal string literals like "fp16", "rcr" which are needed for parsing.
    """
    result = []
    i = 0
    n = len(content)

    # Patterns that indicate a string is problematic and should be stripped
    problematic_patterns = [
        "DECL_KERNEL_SET",
        "DECL_GROUPED_CONV_KERNEL_SET",
        "DECL_FMHA_KERNEL_SET",
        ".add(",
    ]

    while i < n:
        # Check for raw string literal: R"delimiter(...)delimiter"
        # Always strip these as they can contain arbitrary content
        if i < n - 1 and content[i] == "R" and content[i + 1] == '"':
            # Find the delimiter (between R" and ()
            j = i + 2
            delimiter_start = j
            while j < n and content[j] != "(":
                j += 1
            delimiter = content[delimiter_start:j]
            # Find the closing )delimiter"
            end_marker = ")" + delimiter + '"'
            end_pos = content.find(end_marker, j + 1)
            if end_pos != -1:
                # Replace with spaces to preserve line numbers
                span = content[i : end_pos + len(end_marker)]
                result.append("".join("\n" if c == "\n" else " " for c in span))
                i = end_pos + len(end_marker)
                continue

        # Check for regular string literal - only strip if it contains problematic patterns
        if content[i] == '"':
            j = i + 1
            while j < n:
                if content[j] == "\\" and j + 1 < n:
                    j += 2  # Skip escaped character
                elif content[j] == '"':
                    j += 1
                    break
                else:
                    j += 1
            string_content = content[i:j]

            # Only strip if this string contains problematic patterns
            should_strip = any(pat in string_content for pat in problematic_patterns)
            if should_strip:
                result.append(" " * len(string_content))
            else:
                result.append(string_content)
            i = j
            continue

        # Check for single-line comment - always strip
        if i < n - 1 and content[i : i + 2] == "//":
            j = i
            while j < n and content[j] != "\n":
                j += 1
            result.append(" " * (j - i))
            i = j
            continue

        # Check for multi-line comment - always strip
        if i < n - 1 and content[i : i + 2] == "/*":
            end_pos = content.find("*/", i + 2)
            if end_pos != -1:
                span = content[i : end_pos + 2]
                # Preserve newlines in multi-line comments
                result.append("".join("\n" if c == "\n" else " " for c in span))
                i = end_pos + 2
                continue

        result.append(content[i])
        i += 1

    return "".join(result)


def detect_and_parse(source_path: Path) -> Tuple[str, List[Dict]]:
    """Detect example type and parse kernel declarations.

    Properly strips string literals and comments before parsing to avoid
    picking up declarations inside strings or commented-out code.
    """
    content = source_path.read_text()
    content = strip_cpp_strings_and_comments(content)

    if "DECL_FMHA_KERNEL_SET" in content:
        return "fmha", parse_fmha_declarations(content)
    elif "DECL_GROUPED_CONV_KERNEL_SET" in content:
        return "conv", parse_conv_declarations(content)
    elif "DECL_KERNEL_SET" in content:
        return "gemm", parse_gemm_declarations(content)
    return "unknown", []


def generate_gemm_registration(
    kernel_headers: List[Path], example_name: str, kernels: List[Dict] = None
) -> str:
    """Generate GEMM kernel registration code for the dispatcher registry.

    Uses GeneratedKernelInstance<SelectedKernel> to wrap the generated kernels
    and provide the KernelInstance interface for the Dispatcher.

    If kernels list is provided with kernel_set info, generates separate
    registration functions per kernel set.
    """
    if not kernel_headers:
        return "    // No kernels to register"

    # Build mapping from kernel config pattern to kernel set
    kernel_to_set = {}
    kernel_sets = set()
    if kernels:
        for k in kernels:
            tile_m = k.get("tile_m", 128)
            tile_n = k.get("tile_n", 128)
            tile_k = k.get("tile_k", 64)
            warp_m = k.get("warp_m", 2)
            warp_n = k.get("warp_n", 2)
            warp_k = k.get("warp_k", 1)
            warp_tile_m = k.get("warp_tile_m", 32)
            warp_tile_n = k.get("warp_tile_n", 32)
            warp_tile_k = k.get("warp_tile_k", 16)

            # Pattern that appears in kernel filename
            key_pattern = f"{tile_m}x{tile_n}x{tile_k}_{warp_m}x{warp_n}x{warp_k}_{warp_tile_m}x{warp_tile_n}x{warp_tile_k}"
            kernel_set = k.get("kernel_set", "default")
            kernel_to_set[key_pattern] = kernel_set
            kernel_sets.add(kernel_set)

    def generate_registration_block(h: Path) -> str:
        """Generate registration code for a single kernel."""
        kernel_name = h.stem
        ns = f"ns_{kernel_name}"

        # Parse pipeline, scheduler, and layout from kernel name
        # Format: gemm_fp16_rcr_compv3_cshuffle_intrawave_...
        parts = kernel_name.split("_")
        pipeline = "CompV3"
        scheduler = "Intrawave"
        epilogue = "CShuffle"
        datatype = "FP16"
        layout_a = "RowMajor"
        layout_b = "ColMajor"
        layout_c = "RowMajor"

        # Parse datatype (e.g., fp16, bf16, fp32)
        dtype_map = {
            "fp16": "FP16",
            "bf16": "BF16",
            "fp32": "FP32",
            "fp64": "FP64",
            "int8": "INT8",
        }

        # Parse layout from 3-char codes (e.g., rcr, rrr, rrc, ccc)
        # r = RowMajor, c = ColMajor
        layout_map = {"r": "RowMajor", "c": "ColMajor"}

        # Find pipeline, epilogue, scheduler in the name parts
        pipeline_map = {
            "mem": "Mem",
            "compv1": "CompV1",
            "compv2": "CompV2",
            "compv3": "CompV3",
            "compv4": "CompV4",
            "compv5": "CompV5",
            "preshufflev1": "PreShuffleV1",
            "preshufflev2": "PreShuffleV2",
        }
        scheduler_map = {
            "intrawave": "Intrawave",
            "interwave": "Interwave",
            "auto": "Auto",
        }
        epilogue_map = {"default": "Default", "cshuffle": "CShuffle", "none": "None"}

        for part in parts:
            if part in pipeline_map:
                pipeline = pipeline_map[part]
            if part in scheduler_map:
                scheduler = scheduler_map[part]
            if part in epilogue_map:
                epilogue = epilogue_map[part]
            if part in dtype_map:
                datatype = dtype_map[part]
            # Parse 3-char layout codes (e.g., rcr, rrr)
            if len(part) == 3 and all(c in "rc" for c in part):
                layout_a = layout_map[part[0]]
                layout_b = layout_map[part[1]]
                layout_c = layout_map[part[2]]

        block = []
        block.append(f"        // Register kernel: {kernel_name}")
        block.append("        {")
        block.append(f"            using SelectedKernel = {ns}::SelectedKernel;")
        block.append("            ck_tile::dispatcher::KernelKey key;")
        block.append(
            f"            key.signature.dtype_a = ck_tile::dispatcher::DataType::{datatype};"
        )
        block.append(
            f"            key.signature.dtype_b = ck_tile::dispatcher::DataType::{datatype};"
        )
        block.append(
            f"            key.signature.dtype_c = ck_tile::dispatcher::DataType::{datatype};"
        )
        block.append(
            "            key.signature.dtype_acc = ck_tile::dispatcher::DataType::FP32;"
        )
        block.append(
            f"            key.signature.layout_a = ck_tile::dispatcher::LayoutTag::{layout_a};"
        )
        block.append(
            f"            key.signature.layout_b = ck_tile::dispatcher::LayoutTag::{layout_b};"
        )
        block.append(
            f"            key.signature.layout_c = ck_tile::dispatcher::LayoutTag::{layout_c};"
        )
        block.append("            key.algorithm.tile_shape.m = SelectedKernel::TileM;")
        block.append("            key.algorithm.tile_shape.n = SelectedKernel::TileN;")
        block.append("            key.algorithm.tile_shape.k = SelectedKernel::TileK;")
        block.append(
            "            key.algorithm.wave_shape.m = SelectedKernel::WarpPerBlock_M;"
        )
        block.append(
            "            key.algorithm.wave_shape.n = SelectedKernel::WarpPerBlock_N;"
        )
        block.append(
            "            key.algorithm.wave_shape.k = SelectedKernel::WarpPerBlock_K;"
        )
        block.append(
            "            key.algorithm.warp_tile_shape.m = SelectedKernel::WarpTileM;"
        )
        block.append(
            "            key.algorithm.warp_tile_shape.n = SelectedKernel::WarpTileN;"
        )
        block.append(
            "            key.algorithm.warp_tile_shape.k = SelectedKernel::WarpTileK;"
        )
        block.append(
            "            key.algorithm.block_size = SelectedKernel::BlockSize;"
        )
        block.append(
            f"            key.algorithm.pipeline = ck_tile::dispatcher::Pipeline::{pipeline};"
        )
        block.append(
            f"            key.algorithm.scheduler = ck_tile::dispatcher::Scheduler::{scheduler};"
        )
        block.append(
            f"            key.algorithm.epilogue = ck_tile::dispatcher::Epilogue::{epilogue};"
        )
        block.append("            key.gfx_arch = arch;")
        block.append(
            f'            auto instance = std::make_shared<ck_tile::dispatcher::backends::GeneratedKernelInstance<SelectedKernel>>(key, "{kernel_name}");'
        )
        block.append("            registry.register_kernel(instance);")
        block.append("        }")
        return "\n".join(block)

    def find_kernel_set(header: Path) -> str:
        """Find which kernel set a header belongs to."""
        name = header.stem
        for pattern, kset in kernel_to_set.items():
            if pattern in name:
                return kset
        return "default"

    # Group kernels by set
    kernels_by_set = {}
    for h in kernel_headers:
        kset = find_kernel_set(h)
        if kset not in kernels_by_set:
            kernels_by_set[kset] = []
        kernels_by_set[kset].append(h)

    # If only one set or no set info, use simple registration
    if len(kernels_by_set) <= 1:
        lines = ["    (void)arch;", ""]
        for h in kernel_headers:
            lines.append(generate_registration_block(h))
        return "\n".join(lines)

    # Multiple sets - generate registration for all, plus store per-set info
    lines = ["    // Register ALL kernels from all sets", "    (void)arch;", ""]
    for h in kernel_headers:
        lines.append(generate_registration_block(h))

    # Store per-set mapping for separate function generation
    global _kernels_by_set_cache
    _kernels_by_set_cache = (kernels_by_set, generate_registration_block)

    return "\n".join(lines)


# Global cache for per-set kernel info
_kernels_by_set_cache = None


def generate_per_set_functions(source_stem: str) -> str:
    """Generate separate registration functions for each kernel set.

    Generates:
    1. Per-set functions: register_<set_name>(registry, arch)
    2. String-based dispatcher: register_kernel_set("set_name", registry, arch)
    3. get_kernel_set_names() to list available sets
    """
    global _kernels_by_set_cache
    if not _kernels_by_set_cache:
        return ""

    kernels_by_set, gen_block = _kernels_by_set_cache
    _kernels_by_set_cache = None  # Clear cache

    lines = []
    set_names = []

    # Generate per-set functions
    for set_name, headers in kernels_by_set.items():
        safe_name = set_name.replace("-", "_")
        set_names.append((set_name, safe_name))
        lines.append(
            f"inline void register_{safe_name}(ck_tile::dispatcher::Registry& registry, const std::string& arch) {{"
        )
        lines.append("    (void)arch;")
        for h in headers:
            lines.append(gen_block(h))
        lines.append("}")
        lines.append("")

    # Generate string-based dispatcher (only if multiple sets)
    if len(set_names) > 0:
        lines.append("// Dynamic registration by kernel set name")
        lines.append(
            "inline bool register_kernel_set(const std::string& set_name, ck_tile::dispatcher::Registry& registry, const std::string& arch) {"
        )
        for set_name, safe_name in set_names:
            lines.append(
                f'    if (set_name == "{set_name}") {{ register_{safe_name}(registry, arch); return true; }}'
            )
        lines.append("    return false; // Unknown set name")
        lines.append("}")
        lines.append("")

        # Generate helper to list available set names
        lines.append("// Get list of available kernel set names")
        lines.append("inline std::vector<std::string> get_kernel_set_names() {")
        names_str = ", ".join(f'"{name}"' for name, _ in set_names)
        lines.append(f"    return {{{names_str}}};")
        lines.append("}")
        lines.append("")

    return "\n".join(lines)


def generate_conv_registration(
    kernel_headers: List[Path], example_name: str, kernels: List[Dict]
) -> str:
    """Generate Conv kernel registration code for the dispatcher registry.

    Creates real GroupedConvKernelInstance entries backed by the generated
    launcher's launch() method via the conv backend RunFn factories.
    """
    if not kernel_headers:
        return "    // No kernels to register"

    lines = []

    for i, h in enumerate(kernel_headers):
        kname = h.stem
        ns = f"ns_{kname}"
        launcher = f"{ns}::{kname}_Launcher"

        # Determine direction and ndim from the kernel header name
        if "_fwd_" in kname:
            direction = "Forward"
            run_fn_factory = "make_conv_fwd_run_fn"
        elif "_bwd_data_" in kname or "_bwdd_" in kname:
            direction = "BackwardData"
            run_fn_factory = "make_conv_bwd_data_run_fn"
        elif "_bwd_weight_" in kname or "_bwdw_" in kname:
            direction = "BackwardWeight"
            run_fn_factory = "make_conv_bwd_weight_run_fn"
        else:
            direction = "Forward"
            run_fn_factory = "make_conv_fwd_run_fn"

        ndim = 3 if "_3d_" in kname else 2

        # Parse dtype from name (e.g. grouped_conv_fwd_fp16_...)
        dtype = "fp16"
        for dt in ["fp16", "bf16", "fp32"]:
            if f"_{dt}_" in kname:
                dtype = dt
                break

        # Parse tile, wave, warp from name.
        # Format: ..._TILExTILExTILE_WAVExWAVExWAVE_WARPxWARPxWARP_...
        import re as _re

        tile_m, tile_n, tile_k = 1, 128, 128
        wave_m, wave_n, wave_k = 2, 2, 1
        warp_m, warp_n, warp_k = 32, 32, 16

        triplets = _re.findall(r"_(\d+)x(\d+)x(\d+)", kname)
        if len(triplets) >= 1:
            tile_m, tile_n, tile_k = (
                int(triplets[0][0]),
                int(triplets[0][1]),
                int(triplets[0][2]),
            )
        if len(triplets) >= 2:
            wave_m, wave_n, wave_k = (
                int(triplets[1][0]),
                int(triplets[1][1]),
                int(triplets[1][2]),
            )
        if len(triplets) >= 3:
            warp_m, warp_n, warp_k = (
                int(triplets[2][0]),
                int(triplets[2][1]),
                int(triplets[2][2]),
            )

        pipeline = "compv4" if "compv4" in kname else "compv3"
        scheduler = "interwave" if "interwave" in kname else "intrawave"
        epilogue = "cshuffle" if "cshuffle" in kname else "default"

        # ConvConfigBase defaults
        vec_a, vec_b, vec_c = 4, 8, 8
        block_per_cu = 1
        num_wave_groups = 1
        num_groups_to_merge = 1

        lines.append(f"    // Kernel {i + 1}: {kname}")
        lines.append("    {")
        lines.append(f"        ck_tile::dispatcher::GroupedConvKernelKey key_{i};")
        lines.append(f'        key_{i}.dtype_in     = "{dtype}";')
        lines.append(f'        key_{i}.dtype_wei    = "{dtype}";')
        lines.append(f'        key_{i}.dtype_out    = "{dtype}";')
        lines.append(f'        key_{i}.layout       = "nhwgc";')
        lines.append(f"        key_{i}.ndim_spatial = {ndim};")
        lines.append(
            f"        key_{i}.op           = ck_tile::dispatcher::GroupedConvOp::{direction};"
        )
        lines.append(f"        key_{i}.tile_m       = {tile_m};")
        lines.append(f"        key_{i}.tile_n       = {tile_n};")
        lines.append(f"        key_{i}.tile_k       = {tile_k};")
        lines.append(f"        key_{i}.wave_m       = {wave_m};")
        lines.append(f"        key_{i}.wave_n       = {wave_n};")
        lines.append(f"        key_{i}.wave_k       = {wave_k};")
        lines.append(f"        key_{i}.warp_m       = {warp_m};")
        lines.append(f"        key_{i}.warp_n       = {warp_n};")
        lines.append(f"        key_{i}.warp_k       = {warp_k};")
        lines.append(f'        key_{i}.pipeline     = "{pipeline}";')
        lines.append(f'        key_{i}.scheduler    = "{scheduler}";')
        lines.append(f'        key_{i}.epilogue     = "{epilogue}";')
        lines.append(f"        key_{i}.vector_size_a      = {vec_a};")
        lines.append(f"        key_{i}.vector_size_b      = {vec_b};")
        lines.append(f"        key_{i}.vector_size_c      = {vec_c};")
        lines.append(f"        key_{i}.block_per_cu       = {block_per_cu};")
        lines.append(f"        key_{i}.num_wave_groups    = {num_wave_groups};")
        lines.append(f"        key_{i}.num_groups_to_merge = {num_groups_to_merge};")
        lines.append(f"        key_{i}.arch         = arch;")
        lines.append(
            f"        auto run_fn_{i} = ck_tile::dispatcher::backends::{run_fn_factory}<{launcher}, {ndim}>();"
        )
        lines.append(
            f'        auto inst_{i} = std::make_shared<ck_tile::dispatcher::GroupedConvKernelInstance>(key_{i}, "{kname}", std::move(run_fn_{i}));'
        )
        lines.append(f"        registry.register_kernel(key_{i}, inst_{i});")
        lines.append("    }")

    return "\n".join(lines)


def generate_fmha_registration(wrapper_headers: List[Path], source_stem: str) -> str:
    """Generate FMHA registration code using dispatcher wrapper factories."""
    if not wrapper_headers:
        return "    // No FMHA kernels to register"

    lines = ["    (void)arch;", ""]
    for header in sorted(wrapper_headers):
        stem = header.stem.replace("dispatcher_wrapper_", "")
        lines.append(f"    // Register FMHA kernel: {stem}")
        lines.append(
            f"    registry.register_kernel(ck_tile::dispatcher::generated::make_{stem}(arch));"
        )
    return "\n".join(lines)


def _build_conv_codegen_cmd(
    idx: int, k: Dict, codegen_dir: Path, output_dir: Path
) -> Tuple[int, List[str], str]:
    """Build the command for a single conv kernel codegen invocation."""
    variant_map = {
        "forward": "forward",
        "bwd_data": "bwd_data",
        "backward_data": "bwd_data",
        "bwd_weight": "bwd_weight",
        "backward_weight": "bwd_weight",
    }
    variant = variant_map.get(k.get("conv_type", "forward"), "forward")

    cmd = [
        sys.executable,
        str(codegen_dir / "unified_grouped_conv_codegen.py"),
        "--datatype",
        k.get("dtype", "fp16"),
        "--variant",
        variant,
        "--ndim",
        str(k.get("ndim", 2)),
        "--output",
        str(output_dir),
    ]

    if k.get("tile_m"):
        cmd.extend(["--tile-m", str(k["tile_m"])])
    if k.get("tile_n"):
        cmd.extend(["--tile-n", str(k["tile_n"])])
    if k.get("warp_m"):
        cmd.extend(["--warp-m", str(k["warp_m"])])
    if k.get("warp_n"):
        cmd.extend(["--warp-n", str(k["warp_n"])])
    if k.get("warp_k"):
        cmd.extend(["--warp-k", str(k["warp_k"])])
    if k.get("warp_tile_m"):
        cmd.extend(["--warp-tile-m", str(k["warp_tile_m"])])
    if k.get("warp_tile_n"):
        cmd.extend(["--warp-tile-n", str(k["warp_tile_n"])])
    if k.get("warp_tile_k"):
        cmd.extend(["--warp-tile-k", str(k["warp_tile_k"])])
    if k.get("pipeline"):
        cmd.extend(["--pipeline", k["pipeline"]])
    if k.get("scheduler"):
        cmd.extend(["--scheduler", k["scheduler"]])
    if k.get("epilogue"):
        cmd.extend(["--epilogue", k["epilogue"]])
    if k.get("vector_a"):
        cmd.extend(["--vector-a", str(k["vector_a"])])
    if k.get("vector_b"):
        cmd.extend(["--vector-b", str(k["vector_b"])])
    if k.get("vector_c"):
        cmd.extend(["--vector-c", str(k["vector_c"])])
    if k.get("block_per_cu"):
        cmd.extend(["--block-per-cu", str(k["block_per_cu"])])
    if k.get("num_wave_groups"):
        cmd.extend(["--num-wave-groups", str(k["num_wave_groups"])])
    if k.get("num_groups_to_merge"):
        cmd.extend(["--num-groups-to-merge", str(k["num_groups_to_merge"])])
    if k.get("double_smem_buffer") is not None:
        cmd.extend(["--double-smem-buffer", str(k["double_smem_buffer"]).lower()])
    if k.get("tile_k"):
        cmd.extend(["--tile-k", str(k["tile_k"])])

    return (idx, cmd, str(codegen_dir))


def _run_conv_codegen(args: Tuple) -> Tuple[int, bool, str]:
    """Run unified_grouped_conv_codegen.py for a single kernel config (picklable for ProcessPoolExecutor)."""
    idx, cmd, cwd = args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        return (idx, False, result.stderr[:300])
    return (idx, True, "")


def _build_fmha_codegen_cmd(
    idx: int, k: Dict, codegen_dir: Path, output_dir: Path, gpu_target: str
) -> Tuple[int, List[str], str]:
    payload = {
        "arch": k.get("arch", gpu_target),
        "signature": k["signature"],
        "algorithm": k["algorithm"],
    }
    if k.get("profile") is not None:
        payload["profile"] = k["profile"]
    if k.get("receipt") is not None:
        payload["receipt"] = k["receipt"]

    config_json = json.dumps(payload)
    cmd = [
        sys.executable,
        str(codegen_dir / "fmha" / "codegen.py"),
        "--output-dir",
        str(output_dir),
        "--gpu-target",
        gpu_target,
        "--config-json",
        config_json,
    ]
    return (idx, cmd, str(codegen_dir))


def _run_fmha_codegen(args: Tuple) -> Tuple[int, bool, str]:
    idx, cmd, cwd = args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        return (idx, False, result.stderr[:400] or result.stdout[:400])
    return (idx, True, "")


def generate_fmha_kernels(
    kernels: List[Dict], output_dir: Path, codegen_dir: Path, gpu_target: str
) -> bool:
    """Generate FMHA kernels for all declarations using unified FMHA codegen."""
    if not kernels:
        return False

    # FMHA generator revisions can change emitted names or wrapper content.
    # Clear previously generated FMHA files for this example directory so we
    # only compile the current declaration set.
    for pattern in ("fmha_*.hpp", "fmha_*.cpp", "fmha_*.o"):
        for path in output_dir.glob(pattern):
            path.unlink(missing_ok=True)
    wrapper_dir = output_dir / "dispatcher_wrappers"
    if wrapper_dir.exists():
        for path in wrapper_dir.glob("dispatcher_wrapper_fmha_*.hpp"):
            path.unlink(missing_ok=True)

    unique_kernels = []
    seen = set()
    for k in kernels:
        key = json.dumps(k, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        unique_kernels.append(k)

    work_items = [
        _build_fmha_codegen_cmd(idx, k, codegen_dir, output_dir, gpu_target)
        for idx, k in enumerate(unique_kernels)
    ]

    success_count = 0
    max_workers = min(len(work_items), os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_fmha_codegen, w): w[0] for w in work_items}
        for future in as_completed(futures):
            idx, ok, err = future.result()
            if ok:
                success_count += 1
            else:
                print(f"  FMHA codegen error for kernel {idx + 1}: {err}")

    return success_count > 0


def generate_conv_kernels(
    kernels: List[Dict], output_dir: Path, codegen_dir: Path
) -> bool:
    """Generate Conv kernels for ALL declarations using unified codegen.

    Launches all codegen subprocesses in parallel via ProcessPoolExecutor
    for significantly faster generation when multiple conv kernels are declared.
    """
    if not kernels:
        return False

    work_items = [
        _build_conv_codegen_cmd(idx, k, codegen_dir, output_dir)
        for idx, k in enumerate(kernels)
    ]

    success_count = 0
    max_workers = min(len(work_items), os.cpu_count() or 4)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_conv_codegen, w): w[0] for w in work_items}
        for future in as_completed(futures):
            idx, ok, err = future.result()
            if ok:
                success_count += 1
            else:
                print(f"  Codegen error for kernel {idx + 1}: {err}")

    return success_count > 0


def _run_gemm_codegen(args: Tuple) -> Tuple[int, bool, str]:
    """Run unified_gemm_codegen.py for a single kernel config (picklable for ProcessPoolExecutor)."""
    idx, cmd, cwd = args
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        return (idx, False, result.stderr[:300])
    return (idx, True, "")


def generate_gemm_kernels(
    kernels: List[Dict], output_dir: Path, codegen_dir: Path
) -> bool:
    """Generate GEMM kernels for ALL declarations using unified codegen.

    Launches all codegen subprocesses in parallel via ProcessPoolExecutor
    for significantly faster generation when multiple kernels are declared.
    """
    import json

    if not kernels:
        return False

    # Build all commands upfront
    work_items = []
    for idx, k in enumerate(kernels):
        variant = "multi_d" if k.get("elementwise_op") else "standard"

        tile_config = {
            "tile_m": [k.get("tile_m", 128)],
            "tile_n": [k.get("tile_n", 128)],
            "tile_k": [k.get("tile_k", 32)],
            "warp_m": [k.get("warp_m", 2)],
            "warp_n": [k.get("warp_n", 2)],
            "warp_k": [k.get("warp_k", 1)],
            "warp_tile_m": [k.get("warp_tile_m", 32)],
            "warp_tile_n": [k.get("warp_tile_n", 32)],
            "warp_tile_k": [k.get("warp_tile_k", 16)],
        }

        trait_config = {
            "pipeline": [k.get("pipeline", "compv3")],
            "epilogue": [k.get("epilogue", "cshuffle")],
            "scheduler": [k.get("scheduler", "intrawave")],
            "pad_m": [k.get("pad_m", False)],
            "pad_n": [k.get("pad_n", False)],
            "pad_k": [k.get("pad_k", False)],
            "persistent": [False],
        }

        config_json = json.dumps(
            {"tile_config": tile_config, "trait_config": trait_config}
        )

        cmd = [
            sys.executable,
            str(codegen_dir / "unified_gemm_codegen.py"),
            "--datatype",
            k.get("dtype", "fp16"),
            "--layout",
            k.get("layout", "rcr"),
            "--variants",
            variant,
            "--output",
            str(output_dir),
            "--tile-config-json",
            config_json,
        ]

        work_items.append((idx, cmd, str(codegen_dir)))

    # Run all codegen subprocesses in parallel
    success_count = 0
    max_workers = min(len(work_items), os.cpu_count() or 4)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_gemm_codegen, w): w[0] for w in work_items}
        for future in as_completed(futures):
            idx, ok, err = future.result()
            if ok:
                success_count += 1
            else:
                print(f"  Codegen error for kernel {idx + 1}: {err}")

    return success_count > 0


def compile_kernel(args: Tuple) -> Tuple[str, bool, str]:
    """Compile a single kernel to object file."""
    kernel_hpp, output_dir, include_dirs, hipcc, gpu_target, idx, total = args
    kernel_name = kernel_hpp.stem

    wrapper_cpp = output_dir / f"{kernel_name}.cpp"
    wrapper_cpp.write_text(
        f'#include "{kernel_hpp.name}"\nnamespace {{ volatile bool _k{idx} = true; }}\n'
    )

    obj_file = output_dir / f"{kernel_name}.o"

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
    from fmha_utils import fmha_compile_flags  # noqa: E402

    cmd = fmha_compile_flags(gpu_target, hipcc, family="bwd")

    for inc_dir in include_dirs:
        cmd.extend(["-I", str(inc_dir)])
    cmd.extend(["-I", str(kernel_hpp.parent)])
    cmd.extend(["-o", str(obj_file), str(wrapper_cpp)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return (kernel_name, False, result.stderr[:500])
    return (kernel_name, True, str(obj_file))


def main():
    parser = argparse.ArgumentParser(description="Build example kernels")
    parser.add_argument("source", type=Path, help="C++ source file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-dirs", type=str, required=True)
    parser.add_argument("--gpu-target", type=str, default="gfx942")
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument(
        "--target-name", type=str, help="CMake target name (for library naming)"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    codegen_dir = script_dir.parent / "codegen"
    source_stem = args.source.stem  # e.g., "01_basic_gemm"
    target_name = args.target_name or source_stem  # e.g., "gemm_01_basic" from CMake

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Detect and parse
    example_type, kernels = detect_and_parse(args.source)

    if example_type == "conv":
        k = kernels[0] if kernels else {}
        variant = k.get("conv_type", "forward")
        print(
            f"[{target_name}] Conv {k.get('dtype', 'fp16')} {variant} {k.get('ndim', 2)}D ({len(kernels)} declarations)"
        )
    elif example_type == "fmha":
        k = kernels[0] if kernels else {}
        sig = k.get("signature", {})
        print(
            f"[{target_name}] FMHA {sig.get('family', 'fwd')} {sig.get('data_type', 'fp16')} "
            f"{sig.get('mode', 'batch')} hq={sig.get('hdim_q', 128)} hv={sig.get('hdim_v', 128)} "
            f"({len(kernels)} declarations)"
        )
    elif example_type == "gemm":
        k = kernels[0] if kernels else {}
        print(
            f"[{target_name}] GEMM {k.get('dtype', 'fp16')} {k.get('layout', 'rcr')} ({len(kernels)} declarations)"
        )
    else:
        print(f"[{target_name}] No kernel declarations - creating empty library")
        lib_path = args.output_dir / f"lib{target_name}_kernels.a"
        subprocess.run([find_ar(), "rcs", str(lib_path)], check=True)
        header = args.output_dir / f"{source_stem}_kernels.hpp"
        header.write_text(f"// No kernels for {target_name}\n#pragma once\n")
        return 0

    # Generate kernels
    print(f"[{target_name}] Generating kernels...")
    if example_type == "conv":
        success = generate_conv_kernels(kernels, args.output_dir, codegen_dir)
    elif example_type == "fmha":
        success = generate_fmha_kernels(
            kernels, args.output_dir, codegen_dir, args.gpu_target
        )
    else:
        success = generate_gemm_kernels(kernels, args.output_dir, codegen_dir)

    if not success:
        print(f"[{target_name}] Kernel generation failed!")
        return 1

    # Find generated headers
    if example_type == "gemm":
        kernel_headers = list(args.output_dir.glob("gemm_*.hpp"))
        wrapper_headers = list(
            (args.output_dir / "dispatcher_wrappers").glob(
                "dispatcher_wrapper_gemm_*.hpp"
            )
        )
    elif example_type == "fmha":
        kernel_headers = [
            h
            for h in args.output_dir.glob("fmha_*.hpp")
            if not h.name.startswith("dispatcher_wrapper_")
        ]
        wrapper_headers = list(
            (args.output_dir / "dispatcher_wrappers").glob(
                "dispatcher_wrapper_fmha_*.hpp"
            )
        )
    else:
        prefix_map = {
            "forward": "grouped_conv_fwd",
            "bwd_data": "grouped_conv_bwd_data",
            "bwd_weight": "grouped_conv_bwd_weight",
        }
        # Collect headers from ALL variants present in declarations
        variants_used = set(k.get("conv_type", "forward") for k in kernels)
        kernel_headers = []
        for variant in variants_used:
            prefix = prefix_map.get(variant, "grouped_conv_fwd")
            kernel_headers.extend(args.output_dir.glob(f"{prefix}_*.hpp"))

    if not kernel_headers:
        print(f"[{target_name}] No kernel headers generated!")
        return 1

    print(f"[{target_name}] Compiling {len(kernel_headers)} kernels...")

    include_dirs = [Path(p.strip()) for p in args.include_dirs.split(",")]
    hipcc = find_hipcc()

    work = [
        (
            h,
            args.output_dir,
            include_dirs,
            hipcc,
            args.gpu_target,
            i + 1,
            len(kernel_headers),
        )
        for i, h in enumerate(kernel_headers)
    ]

    obj_files = []
    failed = []

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(compile_kernel, w): w[0].name for w in work}
        for future in as_completed(futures):
            name, ok, result = future.result()
            if ok:
                obj_files.append(result)
            else:
                failed.append((name, result))
                print(f"[{target_name}] FAILED: {name}")

    if failed:
        print(f"[{target_name}] {len(failed)} kernels failed")
        for name, err in failed[:3]:
            print(f"  {name}: {err[:200]}")
        return 1

    # Create static library (use target_name for CMake compatibility)
    lib_path = args.output_dir / f"lib{target_name}_kernels.a"
    subprocess.run([find_ar(), "rcs", str(lib_path)] + obj_files, check=True)

    # Generate registration header (use source_stem for header name to match CMake's EXAMPLE_STEM)
    header_path = args.output_dir / f"{source_stem}_kernels.hpp"

    # Build includes
    includes = "\n".join(f'#include "{h.name}"' for h in kernel_headers)

    # Build kernel registration entries
    # Function name uses source_stem (e.g., register_01_basic_gemm_kernels)
    func_name = f"register_{source_stem}_kernels"

    # Generate registration code based on example type
    if example_type == "gemm":
        register_body = generate_gemm_registration(kernel_headers, target_name, kernels)
    else:
        register_body = generate_conv_registration(kernel_headers, target_name, kernels)

    # Generate appropriate header based on example type
    if example_type == "conv" and kernel_headers:
        launcher_aliases = []

        # Helper to find kernel by dtype and type
        def find_kernel_by_dtype_type(headers, dtype, conv_type_marker):
            """Find kernel matching dtype and conv type, prioritize fp16."""
            matching = [h for h in headers if conv_type_marker in h.stem]
            # Prefer fp16 over bf16 for default launchers
            fp16_kernels = [h for h in matching if f"_{dtype}_" in h.stem]
            return (
                fp16_kernels[0] if fp16_kernels else (matching[0] if matching else None)
            )

        # Check what conv types are in the declarations
        has_fwd = any("forward" in k.get("conv_type", "forward") for k in kernels)
        has_bwd_data = any("bwd_data" in k.get("conv_type", "") for k in kernels)
        has_bwd_weight = any("bwd_weight" in k.get("conv_type", "") for k in kernels)

        # Export dtype-specific launcher aliases for each available dtype
        for dtype in ["fp16", "bf16", "fp32"]:
            dtype_fwd_kernels = [
                h
                for h in kernel_headers
                if "_fwd_" in h.stem and f"_{dtype}_" in h.stem
            ]
            if dtype_fwd_kernels:
                k = dtype_fwd_kernels[0]
                ns = f"ns_{k.stem}"
                dtype_upper = dtype.upper()
                launcher_aliases.append(
                    f"using {dtype_upper}FwdKernelLauncher = {ns}::{k.stem}_Launcher;"
                )

        # Export generic launcher aliases (prioritize fp16)
        if has_fwd:
            fwd_kernel = find_kernel_by_dtype_type(kernel_headers, "fp16", "_fwd_")
            if fwd_kernel:
                fwd_ns = f"ns_{fwd_kernel.stem}"
                launcher_aliases.append(
                    f"using FwdKernelLauncher = {fwd_ns}::{fwd_kernel.stem}_Launcher;"
                )
                launcher_aliases.append(
                    f"using FirstKernelLauncher = {fwd_ns}::{fwd_kernel.stem}_Launcher;"
                )

        if has_bwd_data:
            bwd_data_kernel = find_kernel_by_dtype_type(
                kernel_headers, "fp16", "_bwd_data_"
            )
            if not bwd_data_kernel:
                bwd_data_kernel = find_kernel_by_dtype_type(
                    kernel_headers, "fp16", "_bwdd_"
                )
            if bwd_data_kernel:
                bwd_data_ns = f"ns_{bwd_data_kernel.stem}"
                launcher_aliases.append(
                    f"using BwdDataKernelLauncher = {bwd_data_ns}::{bwd_data_kernel.stem}_Launcher;"
                )
                if not has_fwd:
                    launcher_aliases.append(
                        f"using FirstKernelLauncher = {bwd_data_ns}::{bwd_data_kernel.stem}_Launcher;"
                    )

        if has_bwd_weight:
            bwd_weight_kernel = find_kernel_by_dtype_type(
                kernel_headers, "fp16", "_bwd_weight_"
            )
            if not bwd_weight_kernel:
                bwd_weight_kernel = find_kernel_by_dtype_type(
                    kernel_headers, "fp16", "_bwdw_"
                )
            if bwd_weight_kernel:
                bwd_weight_ns = f"ns_{bwd_weight_kernel.stem}"
                launcher_aliases.append(
                    f"using BwdWeightKernelLauncher = {bwd_weight_ns}::{bwd_weight_kernel.stem}_Launcher;"
                )
                if not has_fwd and not has_bwd_data:
                    launcher_aliases.append(
                        f"using FirstKernelLauncher = {bwd_weight_ns}::{bwd_weight_kernel.stem}_Launcher;"
                    )

        launcher_section = "\n".join(launcher_aliases)

        header_content = f"""// Auto-generated for {target_name}
#pragma once

{includes}

#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/kernel_instance.hpp"
#include "ck_tile/dispatcher/kernel_key.hpp"
#include "ck_tile/dispatcher/grouped_conv_registry.hpp"
#include "ck_tile/dispatcher/backends/generated_conv_backend.hpp"

namespace generated {{

// Kernel launchers for direct use
{launcher_section}

// Registration function (takes GroupedConvRegistry for conv kernels)
inline void {func_name}(ck_tile::dispatcher::GroupedConvRegistry& registry, const std::string& arch) {{
{register_body}
}}

}} // namespace generated

// Generic registration - avoids hardcoding the example name in user code
// Safe for single-example executables (typical use case)
#ifndef REGISTER_GENERATED_KERNELS
#define REGISTER_GENERATED_KERNELS(registry, arch) ::generated::{func_name}(registry, arch)
#endif
"""
    elif example_type == "fmha":
        wrapper_includes = "\n".join(
            f'#include "dispatcher_wrappers/{h.name}"' for h in sorted(wrapper_headers)
        )
        register_body = generate_fmha_registration(wrapper_headers, source_stem)
        header_content = f"""// Auto-generated for {target_name}
#pragma once

{wrapper_includes}

#include "ck_tile/dispatcher/fmha_registry.hpp"
#include "ck_tile/dispatcher/fmha_dispatcher.hpp"

namespace generated {{

inline void {func_name}(ck_tile::dispatcher::FmhaRegistry& registry, const std::string& arch) {{
{register_body}
}}

}} // namespace generated

#ifndef REGISTER_GENERATED_KERNELS
#define REGISTER_GENERATED_KERNELS(registry, arch) ::generated::{func_name}(registry, arch)
#endif
"""
    else:
        # GEMM: Generate per-set functions if multiple kernel sets declared
        per_set_funcs = generate_per_set_functions(source_stem)

        header_content = f"""// Auto-generated for {target_name}
#pragma once

{includes}

#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/kernel_instance.hpp"
#include "ck_tile/dispatcher/kernel_key.hpp"
#include "ck_tile/dispatcher/backends/generated_kernel_backend.hpp"

namespace generated {{

// Register ALL kernels from all declared sets
inline void {func_name}(ck_tile::dispatcher::Registry& registry, const std::string& arch) {{
{register_body}
}}

{per_set_funcs}
}} // namespace generated

// Generic registration - avoids hardcoding the example name in user code
// Safe for single-example executables (typical use case)
#ifndef REGISTER_GENERATED_KERNELS
#define REGISTER_GENERATED_KERNELS(registry, arch) ::generated::{func_name}(registry, arch)
#endif

// Register a specific kernel set by name (for multi-registry patterns)
// Usage: REGISTER_KERNEL_SET("compute_bound_set", registry, arch)
#ifndef REGISTER_KERNEL_SET
#define REGISTER_KERNEL_SET(set_name, registry, arch) ::generated::register_kernel_set(set_name, registry, arch)
#endif
"""
    header_path.write_text(header_content)

    print(f"[{target_name}] OK {len(obj_files)} kernels compiled")
    return 0


if __name__ == "__main__":
    sys.exit(main())
