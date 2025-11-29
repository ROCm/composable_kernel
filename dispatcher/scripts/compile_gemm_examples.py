#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Cross-platform build script for declarative kernel workflow.

Uses existing ctypes_utils.py for path management and codegen.

Usage:
    python3 compile_gemm_examples.py <source_file.cpp> [output_name]

Example:
    python3 compile_gemm_examples.py examples/cpp/01_basic_gemm.cpp my_app
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
import shutil

# Add dispatcher/python to path to reuse existing utilities
SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "python"))

# Import existing utilities (after sys.path modification)
from ctypes_utils import (  # noqa: E402
    get_dispatcher_root,
    get_ck_root,
    get_build_dir,
    get_generated_kernels_dir,
    CodegenRunner,
)


# =============================================================================
# Terminal Colors (cross-platform)
# =============================================================================


class Colors:
    if sys.platform != "win32" and sys.stdout.isatty():
        GREEN = "\033[0;32m"
        YELLOW = "\033[1;33m"
        RED = "\033[0;31m"
        NC = "\033[0m"
    else:
        GREEN = YELLOW = RED = NC = ""


def print_phase(msg: str):
    print(f"{Colors.YELLOW}{msg}{Colors.NC}")


def print_success(msg: str):
    print(f"{Colors.GREEN}{msg}{Colors.NC}")


def print_error(msg: str):
    print(f"{Colors.RED}{msg}{Colors.NC}", file=sys.stderr)


# =============================================================================
# Compiler Detection
# =============================================================================


def find_hipcc() -> str:
    """Find hipcc compiler."""
    candidates = [
        os.environ.get("HIPCC"),
        "/opt/rocm/bin/hipcc",
        "/opt/rocm/hip/bin/hipcc",
        shutil.which("hipcc"),
    ]

    for path in candidates:
        if path and os.path.isfile(path):
            return path

    raise RuntimeError(
        "hipcc not found. Please install ROCm or set HIPCC environment variable."
    )


# =============================================================================
# Declaration Extraction
# =============================================================================


def extract_conv_kernel_declarations(source_file: Path) -> list:
    """Extract CONVOLUTION kernel declarations from C++ source file.

    Supports DECL_CONV_KERNEL_SET macro with Signature/Algorithm/Arch pattern.
    """
    content = source_file.read_text()
    declarations = []
    seen = set()

    # Pattern: DECL_CONV_KERNEL_SET(name, .add(...).add(...))
    set_pattern = r"DECL_CONV_KERNEL_SET\s*\(\s*(\w+)\s*,([^;]+)\)"

    for match in re.finditer(set_pattern, content, re.DOTALL):
        set_name = match.group(1)
        set_body = match.group(2)

        # Pattern 1: Simple add("dtype", "layout", "conv_type", tile_k, tile_c)
        simple_add = (
            r'\.add\s*\(\s*"(\w+)"\s*,\s*"(\w+)"\s*,\s*"(\w+)"\s*,\s*(\d+)\s*,\s*(\d+)'
        )
        for add_match in re.finditer(simple_add, set_body):
            dtype = add_match.group(1)
            layout = add_match.group(2)
            conv_type = add_match.group(3)
            tile_k = int(add_match.group(4))
            tile_c = int(add_match.group(5))

            name = f"{set_name}:{dtype}_{layout}_{conv_type}_{tile_k}x{tile_c}"
            if name not in seen:
                seen.add(name)
                declarations.append(
                    {
                        "type": "conv",
                        "dtype": dtype,
                        "layout": layout,
                        "conv_type": conv_type,
                        "num_dims": 2,  # Default
                        "groups": 1,
                        "tile_n": 1,
                        "tile_k": tile_k,
                        "tile_c": tile_c,
                        "wave_m": -1,
                        "wave_n": -1,
                        "wave_k": 1,
                        "warp_m": -1,
                        "warp_n": -1,
                        "warp_k": 16,
                        "pipeline": "compv4",
                        "scheduler": "intrawave",
                        "name": name,
                        "set": set_name,
                        "arch": "gfx942",
                    }
                )

        # Pattern 2: Full specification with ConvSig() and ConvAlgo()
        # .add(ConvSig()...., ConvAlgo()...., "arch")
        full_add_pattern = (
            r'\.add\s*\(\s*(ConvSig\(\)[^,]+),\s*(ConvAlgo\(\)[^,]+),\s*"(\w+)"\s*\)'
        )

        for add_match in re.finditer(full_add_pattern, set_body, re.DOTALL):
            sig_str = add_match.group(1)
            algo_str = add_match.group(2)
            arch = add_match.group(3)

            # Parse signature
            dtype = "fp16"
            dtype_match = re.search(r'\.dtype\s*\(\s*"(\w+)"', sig_str)
            if dtype_match:
                dtype = dtype_match.group(1)

            layout = "nhwc"
            layout_match = re.search(r'\.layout\s*\(\s*"(\w+)"', sig_str)
            if layout_match:
                layout = layout_match.group(1)

            conv_type = "forward"
            conv_type_match = re.search(r'\.conv_type\s*\(\s*"(\w+)"', sig_str)
            if conv_type_match:
                conv_type = conv_type_match.group(1)

            num_dims = 2
            dims_match = re.search(r"\.dims\s*\(\s*(\d+)", sig_str)
            if dims_match:
                num_dims = int(dims_match.group(1))

            groups = 1
            groups_match = re.search(r"\.groups\s*\(\s*(\d+)", sig_str)
            if groups_match:
                groups = int(groups_match.group(1))

            # Parse algorithm
            tile_n, tile_k, tile_c = 1, 128, 128
            tile_match = re.search(
                r"\.tile\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", algo_str
            )
            if tile_match:
                tile_n = int(tile_match.group(1))
                tile_k = int(tile_match.group(2))
                tile_c = int(tile_match.group(3))

            wave_m, wave_n, wave_k = -1, -1, 1
            wave_match = re.search(
                r"\.wave\s*\(\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*(\d+))?", algo_str
            )
            if wave_match:
                wave_m = int(wave_match.group(1))
                wave_n = int(wave_match.group(2))
                wave_k = int(wave_match.group(3) or 1)

            warp_m, warp_n, warp_k = -1, -1, 16
            warp_match = re.search(
                r"\.warp\s*\(\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*(\d+))?", algo_str
            )
            if warp_match:
                warp_m = int(warp_match.group(1))
                warp_n = int(warp_match.group(2))
                warp_k = int(warp_match.group(3) or 16)

            pipeline = "compv4"
            pipeline_match = re.search(r'\.pipeline\s*\(\s*"(\w+)"', algo_str)
            if pipeline_match:
                pipeline = pipeline_match.group(1)

            scheduler = "intrawave"
            scheduler_match = re.search(r'\.scheduler\s*\(\s*"(\w+)"', algo_str)
            if scheduler_match:
                scheduler = scheduler_match.group(1)

            name = f"{set_name}:{dtype}_{layout}_{conv_type}_{tile_k}x{tile_c}"
            if name not in seen:
                seen.add(name)
                declarations.append(
                    {
                        "type": "conv",
                        "dtype": dtype,
                        "layout": layout,
                        "conv_type": conv_type,
                        "num_dims": num_dims,
                        "groups": groups,
                        "tile_n": tile_n,
                        "tile_k": tile_k,
                        "tile_c": tile_c,
                        "wave_m": wave_m,
                        "wave_n": wave_n,
                        "wave_k": wave_k,
                        "warp_m": warp_m,
                        "warp_n": warp_n,
                        "warp_k": warp_k,
                        "pipeline": pipeline,
                        "scheduler": scheduler,
                        "name": name,
                        "set": set_name,
                        "arch": arch,
                    }
                )

    return declarations


def expand_conv_declaration_with_arch_filter(decl: dict, arch: str = "gfx942") -> list:
    """Expand a convolution declaration to all valid combinations.

    Like GEMM, convolution supports wildcard expansion for:
    - wave/warp: If -1, generates all valid combinations
    - pipeline/scheduler: If "*", generates all valid trait combinations
    """
    # Import arch filter
    codegen_dir = get_dispatcher_root() / "codegen"
    sys.path.insert(0, str(codegen_dir))

    try:
        from arch_specs_generated import (
            WARP_SUPPORTED_COMBINATIONS,
            WARP_TILE_SUPPORTED_COMBINATIONS,
            TRAIT_UNSUPPORTED_COMBINATIONS,
        )
    except ImportError:
        # Fallback
        WARP_SUPPORTED_COMBINATIONS = {
            "gfx942": [[1, 4, 1], [2, 2, 1], [4, 1, 1]],
        }
        WARP_TILE_SUPPORTED_COMBINATIONS = {
            "gfx942": {"fp16_fp16_fp16": [[16, 16, 16], [32, 32, 16]]},
        }
        TRAIT_UNSUPPORTED_COMBINATIONS = set()

    d = decl.copy()
    tile_k = d.get("tile_k", 128)
    tile_c = d.get("tile_c", 128)
    dtype = d.get("dtype", "fp16")

    # Check what needs expansion
    needs_wave_expansion = d.get("wave_m", -1) < 0 or d.get("wave_n", -1) < 0
    needs_warp_expansion = d.get("warp_m", -1) < 0 or d.get("warp_n", -1) < 0
    needs_pipeline_expansion = d.get("pipeline", "compv4") == "*"
    needs_scheduler_expansion = d.get("scheduler", "intrawave") == "*"

    if (
        not needs_wave_expansion
        and not needs_warp_expansion
        and not needs_pipeline_expansion
        and not needs_scheduler_expansion
    ):
        return [d]

    # Build valid combinations
    if needs_wave_expansion or needs_warp_expansion:
        wave_configs = WARP_SUPPORTED_COMBINATIONS.get(arch, [[2, 2, 1]])
        dtype_key = f"{dtype}_{dtype}_{dtype}"
        warp_tile_configs = WARP_TILE_SUPPORTED_COMBINATIONS.get(arch, {}).get(
            dtype_key, [[32, 32, 16], [16, 16, 16]]
        )
    else:
        wave_configs = [[d.get("wave_m", 2), d.get("wave_n", 2), d.get("wave_k", 1)]]
        warp_tile_configs = [
            [d.get("warp_m", 32), d.get("warp_n", 32), d.get("warp_k", 16)]
        ]

    # Pipeline/scheduler combinations
    ALL_PIPELINES = ["compv3", "compv4"]
    ALL_SCHEDULERS = ["intrawave", "interwave"]

    pipelines = (
        ALL_PIPELINES if needs_pipeline_expansion else [d.get("pipeline", "compv4")]
    )
    schedulers = (
        ALL_SCHEDULERS
        if needs_scheduler_expansion
        else [d.get("scheduler", "intrawave")]
    )

    expanded = []

    for wm, wn, wk in wave_configs:
        for wtm, wtn, wtk in warp_tile_configs:
            # Check divisibility for conv (M=output spatial, N=K channels, K=C channels)
            # Simplified check for now
            if tile_k % (wn * wtn) != 0:
                continue
            if tile_c % (wk * wtk) != 0:
                continue

            for pipeline in pipelines:
                for scheduler in schedulers:
                    # Check trait combination
                    if (
                        pipeline,
                        "cshuffle",
                        scheduler,
                    ) in TRAIT_UNSUPPORTED_COMBINATIONS:
                        continue

                    expanded_d = d.copy()
                    expanded_d["wave_m"] = wm
                    expanded_d["wave_n"] = wn
                    expanded_d["wave_k"] = wk
                    expanded_d["warp_m"] = wtm
                    expanded_d["warp_n"] = wtn
                    expanded_d["warp_k"] = wtk
                    expanded_d["pipeline"] = pipeline
                    expanded_d["scheduler"] = scheduler

                    expanded_d["name"] = (
                        f"conv_{d['conv_type']}_{dtype}_{d['num_dims']}d_{pipeline}_"
                        f"{scheduler}_{tile_k}x{tile_c}_{wm}x{wn}x{wk}"
                    )
                    expanded.append(expanded_d)

    if not expanded:
        # Fallback to defaults
        d["wave_m"] = 2
        d["wave_n"] = 2
        d["wave_k"] = 1
        d["warp_m"] = 32
        d["warp_n"] = 32
        d["warp_k"] = 16
        d["pipeline"] = "compv4"
        d["scheduler"] = "intrawave"
        return [d]

    return expanded


def generate_conv_kernels(declarations: list, gpu_target: str = "gfx942") -> int:
    """Generate convolution kernels using unified_conv_codegen."""
    kernel_dir = get_generated_kernels_dir()
    kernel_dir.mkdir(parents=True, exist_ok=True)

    # Import conv codegen
    codegen_dir = get_dispatcher_root() / "codegen"
    sys.path.insert(0, str(codegen_dir))

    try:
        from unified_conv_codegen import (
            UnifiedConvCodegen,
            ConvKernelConfig,
            ConvVariant,
        )
    except ImportError as e:
        print_error(f"  Failed to import conv codegen: {e}")
        return 0

    codegen = UnifiedConvCodegen(kernel_dir)
    total_generated = 0

    for decl in declarations:
        dtype = decl.get("dtype", "fp16")
        conv_type = decl.get("conv_type", "forward")
        num_dims = decl.get("num_dims", 2)

        # Map to ConvVariant
        variant = ConvVariant.FORWARD
        if conv_type == "bwd_data":
            variant = ConvVariant.BWD_DATA
        elif conv_type == "bwd_weight":
            variant = ConvVariant.BWD_WEIGHT

        # Create ConvKernelConfig
        config = ConvKernelConfig(
            variant=variant,
            pipeline=decl.get("pipeline", "compv4"),
            scheduler=decl.get("scheduler", "intrawave"),
            tile_m=decl.get("tile_k", 128),  # K is M in conv GEMM view
            tile_n=decl.get("tile_c", 128),  # C is N in conv GEMM view
            tile_k=64,
            wave_m=decl.get("wave_m", 2),
            wave_n=decl.get("wave_n", 2),
            warp_m=decl.get("warp_m", 32),
            warp_n=decl.get("warp_n", 32),
            warp_k=decl.get("warp_k", 16),
            ndim=num_dims,
        )

        try:
            filepath = codegen.generate_kernel(config, dtype)
            total_generated += 1
            print(f"    Generated: {filepath.name}")
        except Exception as e:
            print_error(f"    Failed to generate {decl['name']}: {e}")

    return total_generated


# Original GEMM extraction continues here
def extract_kernel_declarations(source_file: Path) -> list:
    """Extract GEMM kernel declarations from C++ source file."""
    content = source_file.read_text()
    declarations = []
    seen = set()

    # -------------------------------------------------------------------------
    # Pattern 1: Legacy DECLARE_GEMM_KERNEL(dtype, layout, tile_m, tile_n, tile_k)
    # -------------------------------------------------------------------------
    legacy_pattern = r"DECLARE_(?:GEMM_)?KERNEL\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)"
    for match in re.findall(legacy_pattern, content):
        dtype, layout, tm, tn, tk = match
        name = f"{dtype}_{layout}_{tm}x{tn}x{tk}"
        if name not in seen:
            seen.add(name)
            declarations.append(
                {
                    "dtype_a": dtype,
                    "dtype_b": dtype,
                    "dtype_c": dtype,
                    "layout": layout,
                    "tile_m": int(tm),
                    "tile_n": int(tn),
                    "tile_k": int(tk),
                    "wave_m": -1,
                    "wave_n": -1,
                    "wave_k": 1,
                    "warp_m": -1,
                    "warp_n": -1,
                    "warp_k": 16,
                    "pipeline": "compv4",
                    "scheduler": "intrawave",
                    "epilogue": "cshuffle",
                    "name": name,
                    "wildcard": False,
                }
            )

    # -------------------------------------------------------------------------
    # Pattern 2: Fluent API: DECL_KERNEL(Signature()..., Algorithm()..., arch)
    # -------------------------------------------------------------------------
    # Match DECL_KERNEL( ... );  blocks
    fluent_pattern = r'DECL_KERNEL\s*\(\s*(Signature\(\)[^,]+),\s*(Algorithm\(\)[^,]+)(?:,\s*"([^"]+)")?\s*\)'

    for match in re.finditer(fluent_pattern, content, re.DOTALL):
        sig_str = match.group(1)
        algo_str = match.group(2)
        arch = match.group(3) or "gfx942"

        # Parse Signature
        sig = {"dtype_a": "fp16", "dtype_b": "fp16", "dtype_c": "fp16", "layout": "rcr"}

        # .dtype("fp16", "fp16", "fp16", "fp32") or .dtype("fp16")
        dtype_match = re.search(
            r'\.dtype\("([^"]+)"(?:,\s*"([^"]+)")?(?:,\s*"([^"]+)")?', sig_str
        )
        if dtype_match:
            sig["dtype_a"] = dtype_match.group(1)
            sig["dtype_b"] = dtype_match.group(2) or dtype_match.group(1)
            sig["dtype_c"] = dtype_match.group(3) or dtype_match.group(1)

        # .layout("rcr") or .layout("row", "col", "row")
        layout_match = re.search(
            r'\.layout\("([^"]+)"(?:,\s*"([^"]+)")?(?:,\s*"([^"]+)")?', sig_str
        )
        if layout_match:
            if layout_match.group(2):  # Three-arg form
                la = layout_match.group(1)
                lb = layout_match.group(2)
                lc = layout_match.group(3) or "row"
                sig["layout"] = (
                    ("r" if la == "row" else "c")
                    + ("r" if lb == "row" else "c")
                    + ("r" if lc == "row" else "c")
                )
            else:  # Single arg "rcr"
                sig["layout"] = layout_match.group(1)

        # Parse Algorithm
        algo = {}

        # .tile(128, 128, 32)
        tile_match = re.search(r"\.tile\((\d+),\s*(\d+),\s*(\d+)\)", algo_str)
        if tile_match:
            algo["tile_m"] = int(tile_match.group(1))
            algo["tile_n"] = int(tile_match.group(2))
            algo["tile_k"] = int(tile_match.group(3))

        # .wave(2, 2, 1)
        wave_match = re.search(r"\.wave\((\d+),\s*(\d+)(?:,\s*(\d+))?\)", algo_str)
        if wave_match:
            algo["wave_m"] = int(wave_match.group(1))
            algo["wave_n"] = int(wave_match.group(2))
            algo["wave_k"] = int(wave_match.group(3) or 1)

        # .warp(32, 32, 16)
        warp_match = re.search(r"\.warp\((\d+),\s*(\d+)(?:,\s*(\d+))?\)", algo_str)
        if warp_match:
            algo["warp_m"] = int(warp_match.group(1))
            algo["warp_n"] = int(warp_match.group(2))
            algo["warp_k"] = int(warp_match.group(3) or 16)

        # .pipeline("compv4"), .scheduler("intrawave"), .epilogue("cshuffle")
        for field in ["pipeline", "scheduler", "epilogue"]:
            fmatch = re.search(rf'\.{field}\("([^"]+)"\)', algo_str)
            if fmatch:
                algo[field] = fmatch.group(1)

        # Build declaration
        tm = algo.get("tile_m", 128)
        tn = algo.get("tile_n", 128)
        tk = algo.get("tile_k", 32)

        name = f"{sig['dtype_a']}_{sig['layout']}_{tm}x{tn}x{tk}"

        if name not in seen:
            seen.add(name)
            declarations.append(
                {
                    "dtype_a": sig["dtype_a"],
                    "dtype_b": sig["dtype_b"],
                    "dtype_c": sig["dtype_c"],
                    "layout": sig["layout"],
                    "tile_m": tm,
                    "tile_n": tn,
                    "tile_k": tk,
                    "wave_m": algo.get("wave_m", -1),
                    "wave_n": algo.get("wave_n", -1),
                    "wave_k": algo.get("wave_k", 1),
                    "warp_m": algo.get("warp_m", -1),
                    "warp_n": algo.get("warp_n", -1),
                    "warp_k": algo.get("warp_k", 16),
                    "pipeline": algo.get("pipeline", "compv4"),
                    "scheduler": algo.get("scheduler", "intrawave"),
                    "epilogue": algo.get("epilogue", "cshuffle"),
                    "arch": arch,
                    "name": name,
                    "wildcard": False,
                }
            )

    # -------------------------------------------------------------------------
    # Pattern 3: DECL_KERNEL_ALL(dtype, layout) - wildcard
    # -------------------------------------------------------------------------
    all_pattern = r"DECL_KERNEL(?:S)?_ALL\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)"
    for match in re.findall(all_pattern, content):
        dtype, layout = match
        name = f"wildcard_{dtype}_{layout}"
        if name not in seen:
            seen.add(name)
            declarations.append(
                {
                    "dtype_a": dtype,
                    "dtype_b": dtype,
                    "dtype_c": dtype,
                    "layout": layout,
                    "tile_m": -1,
                    "tile_n": -1,
                    "tile_k": -1,
                    "wave_m": -1,
                    "wave_n": -1,
                    "wave_k": 1,
                    "warp_m": -1,
                    "warp_n": -1,
                    "warp_k": 16,
                    "pipeline": "compv4",
                    "scheduler": "intrawave",
                    "epilogue": "cshuffle",
                    "name": name,
                    "wildcard": True,
                }
            )

    # -------------------------------------------------------------------------
    # Pattern 4: DECL_KERNEL_SIMPLE(dtype, layout, tm, tn, tk)
    # -------------------------------------------------------------------------
    simple_pattern = r"DECL_KERNEL_SIMPLE\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)"
    for match in re.findall(simple_pattern, content):
        dtype, layout, tm, tn, tk = match
        name = f"{dtype}_{layout}_{tm}x{tn}x{tk}"
        if name not in seen:
            seen.add(name)
            declarations.append(
                {
                    "dtype_a": dtype,
                    "dtype_b": dtype,
                    "dtype_c": dtype,
                    "layout": layout,
                    "tile_m": int(tm),
                    "tile_n": int(tn),
                    "tile_k": int(tk),
                    "wave_m": -1,
                    "wave_n": -1,
                    "wave_k": 1,
                    "warp_m": -1,
                    "warp_n": -1,
                    "warp_k": 16,
                    "pipeline": "compv4",
                    "scheduler": "intrawave",
                    "epilogue": "cshuffle",
                    "name": name,
                    "wildcard": False,
                    "set": None,
                }
            )

    # -------------------------------------------------------------------------
    # Pattern 5: DECL_KERNEL_SET(name, .add(...).add(...))
    # Named kernel sets for multiple registries
    # -------------------------------------------------------------------------
    set_pattern = r"DECL_KERNEL_SET\s*\(\s*(\w+)\s*,([^;]+)\)"
    for match in re.finditer(set_pattern, content, re.DOTALL):
        set_name = match.group(1)
        set_body = match.group(2)

        # Parse .add("dtype", "layout", tm, tn, tk) calls
        add_simple = r'\.add\s*\(\s*"(\w+)"\s*,\s*"(\w+)"\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)'
        for add_match in re.findall(add_simple, set_body):
            dtype, layout, tm, tn, tk = add_match
            name = f"{set_name}:{dtype}_{layout}_{tm}x{tn}x{tk}"
            if name not in seen:
                seen.add(name)
                declarations.append(
                    {
                        "dtype_a": dtype,
                        "dtype_b": dtype,
                        "dtype_c": dtype,
                        "layout": layout,
                        "tile_m": int(tm),
                        "tile_n": int(tn),
                        "tile_k": int(tk),
                        "wave_m": -1,
                        "wave_n": -1,
                        "wave_k": 1,
                        "warp_m": -1,
                        "warp_n": -1,
                        "warp_k": 16,
                        "pipeline": "compv4",
                        "scheduler": "intrawave",
                        "epilogue": "cshuffle",
                        "name": name,
                        "wildcard": False,
                        "set": set_name,
                    }
                )

        # Parse .add(Signature()..., Algorithm()...) fluent calls
        add_fluent = r"\.add\s*\(\s*Signature\(\)([^,]*),\s*Algorithm\(\)([^)]*\))\s*\)"
        for add_match in re.finditer(add_fluent, set_body, re.DOTALL):
            sig_str = add_match.group(1)
            algo_str = add_match.group(2)

            # Parse dtype and layout from Signature
            dtype = "fp16"
            layout = "rcr"
            dtype_m = re.search(r'\.dtype\("([^"]+)"', sig_str)
            if dtype_m:
                dtype = dtype_m.group(1)
            layout_m = re.search(r'\.layout\("([^"]+)"', sig_str)
            if layout_m:
                layout = layout_m.group(1)

            # Parse tile from Algorithm
            tm, tn, tk = 128, 128, 32
            tile_m = re.search(r"\.tile\((\d+),\s*(\d+),\s*(\d+)\)", algo_str)
            if tile_m:
                tm, tn, tk = (
                    int(tile_m.group(1)),
                    int(tile_m.group(2)),
                    int(tile_m.group(3)),
                )

            # Parse wave/warp (optional)
            wave_m, wave_n, wave_k = -1, -1, 1
            wave_match = re.search(r"\.wave\((\d+),\s*(\d+)(?:,\s*(\d+))?\)", algo_str)
            if wave_match:
                wave_m, wave_n = int(wave_match.group(1)), int(wave_match.group(2))
                wave_k = int(wave_match.group(3) or 1)

            warp_m, warp_n, warp_k = -1, -1, 16
            warp_match = re.search(r"\.warp\((\d+),\s*(\d+)(?:,\s*(\d+))?\)", algo_str)
            if warp_match:
                warp_m, warp_n = int(warp_match.group(1)), int(warp_match.group(2))
                warp_k = int(warp_match.group(3) or 16)

            name = f"{set_name}:{dtype}_{layout}_{tm}x{tn}x{tk}"
            if name not in seen:
                seen.add(name)
                declarations.append(
                    {
                        "dtype_a": dtype,
                        "dtype_b": dtype,
                        "dtype_c": dtype,
                        "layout": layout,
                        "tile_m": tm,
                        "tile_n": tn,
                        "tile_k": tk,
                        "wave_m": wave_m,
                        "wave_n": wave_n,
                        "wave_k": wave_k,
                        "warp_m": warp_m,
                        "warp_n": warp_n,
                        "warp_k": warp_k,
                        "pipeline": "compv4",
                        "scheduler": "intrawave",
                        "epilogue": "cshuffle",
                        "name": name,
                        "wildcard": False,
                        "set": set_name,
                    }
                )

    return declarations


def expand_declaration_with_arch_filter(decl: dict, arch: str = "gfx942") -> list:
    """Expand a declaration to all valid combinations using arch filter.

    Expands wildcards for:
    - wave/warp: If -1, generates all valid wave/warp_tile combinations
    - pipeline/scheduler/epilogue: If "*", generates all valid trait combinations

    Uses the arch_filter module for architecture-specific validation.
    """
    # Import arch filter
    codegen_dir = get_dispatcher_root() / "codegen"
    sys.path.insert(0, str(codegen_dir))

    try:
        from arch_specs_generated import (
            WARP_SUPPORTED_COMBINATIONS,
            WARP_TILE_SUPPORTED_COMBINATIONS,
            TRAIT_UNSUPPORTED_COMBINATIONS,
        )
    except ImportError:
        # Fallback to hardcoded valid combinations
        WARP_SUPPORTED_COMBINATIONS = {
            "gfx90a": [[1, 4, 1], [2, 2, 1], [4, 1, 1]],
            "gfx942": [[1, 4, 1], [2, 2, 1], [4, 1, 1]],
            "gfx950": [[1, 4, 1], [2, 2, 1], [4, 1, 1]],
        }
        WARP_TILE_SUPPORTED_COMBINATIONS = {
            "gfx942": {"fp16_fp16_fp16": [[16, 16, 16], [32, 32, 16]]},
        }
        TRAIT_UNSUPPORTED_COMBINATIONS = {
            ("compv3", "cshuffle", "interwave"),
            ("compv3", "default", "interwave"),
            ("compv4", "cshuffle", "interwave"),
            ("compv4", "default", "interwave"),
        }

    d = decl.copy()
    tm = d.get("tile_m", 128)
    tn = d.get("tile_n", 128)
    tk = d.get("tile_k", 32)
    dtype = d.get("dtype_a", "fp16")

    # Check what needs expansion
    needs_wave_expansion = d.get("wave_m", -1) < 0 or d.get("wave_n", -1) < 0
    needs_warp_expansion = d.get("warp_m", -1) < 0 or d.get("warp_n", -1) < 0
    needs_pipeline_expansion = d.get("pipeline", "compv4") == "*"
    needs_scheduler_expansion = d.get("scheduler", "intrawave") == "*"
    needs_epilogue_expansion = d.get("epilogue", "cshuffle") == "*"
    needs_pad_m_expansion = d.get("pad_m", 1) == -1
    needs_pad_n_expansion = d.get("pad_n", 1) == -1
    needs_pad_k_expansion = d.get("pad_k", 1) == -1
    needs_trait_expansion = (
        needs_pipeline_expansion
        or needs_scheduler_expansion
        or needs_epilogue_expansion
    )
    needs_pad_expansion = (
        needs_pad_m_expansion or needs_pad_n_expansion or needs_pad_k_expansion
    )

    if (
        not needs_wave_expansion
        and not needs_warp_expansion
        and not needs_trait_expansion
        and not needs_pad_expansion
    ):
        # Already fully specified
        return [d]

    # === Build valid combinations ===

    # Wave/warp combinations
    if needs_wave_expansion or needs_warp_expansion:
        wave_configs = WARP_SUPPORTED_COMBINATIONS.get(arch, [[2, 2, 1]])
        dtype_key = f"{dtype}_{dtype}_{dtype}"
        warp_tile_configs = WARP_TILE_SUPPORTED_COMBINATIONS.get(arch, {}).get(
            dtype_key, [[32, 32, 16], [16, 16, 16]]
        )
    else:
        wave_configs = [[d.get("wave_m", 2), d.get("wave_n", 2), d.get("wave_k", 1)]]
        warp_tile_configs = [
            [d.get("warp_m", 32), d.get("warp_n", 32), d.get("warp_k", 16)]
        ]

    # Pipeline/scheduler/epilogue combinations
    # Valid options per category
    ALL_PIPELINES = ["compv3", "compv4"]  # Most common; add more if needed
    ALL_SCHEDULERS = ["intrawave", "interwave"]
    ALL_EPILOGUES = ["cshuffle", "default"]
    ALL_PAD_OPTIONS = [False, True]  # 0 and 1

    pipelines = (
        ALL_PIPELINES if needs_pipeline_expansion else [d.get("pipeline", "compv4")]
    )
    schedulers = (
        ALL_SCHEDULERS
        if needs_scheduler_expansion
        else [d.get("scheduler", "intrawave")]
    )
    epilogues = (
        ALL_EPILOGUES if needs_epilogue_expansion else [d.get("epilogue", "cshuffle")]
    )
    pad_m_opts = ALL_PAD_OPTIONS if needs_pad_m_expansion else [bool(d.get("pad_m", 1))]
    pad_n_opts = ALL_PAD_OPTIONS if needs_pad_n_expansion else [bool(d.get("pad_n", 1))]
    pad_k_opts = ALL_PAD_OPTIONS if needs_pad_k_expansion else [bool(d.get("pad_k", 1))]

    expanded = []

    # Generate all valid combinations
    for wm, wn, wk in wave_configs:
        for wtm, wtn, wtk in warp_tile_configs:
            # Check divisibility constraints
            if tm % (wm * wtm) != 0:
                continue
            if tn % (wn * wtn) != 0:
                continue
            if tk % (wk * wtk) != 0:
                continue

            for pipeline in pipelines:
                for scheduler in schedulers:
                    for epilogue in epilogues:
                        # Check trait combination is valid
                        if (
                            pipeline,
                            epilogue,
                            scheduler,
                        ) in TRAIT_UNSUPPORTED_COMBINATIONS:
                            continue

                        for pad_m in pad_m_opts:
                            for pad_n in pad_n_opts:
                                for pad_k in pad_k_opts:
                                    # Create expanded declaration
                                    expanded_d = d.copy()
                                    expanded_d["wave_m"] = wm
                                    expanded_d["wave_n"] = wn
                                    expanded_d["wave_k"] = wk
                                    expanded_d["warp_m"] = wtm
                                    expanded_d["warp_n"] = wtn
                                    expanded_d["warp_k"] = wtk
                                    expanded_d["pipeline"] = pipeline
                                    expanded_d["scheduler"] = scheduler
                                    expanded_d["epilogue"] = epilogue
                                    expanded_d["pad_m"] = int(pad_m)
                                    expanded_d["pad_n"] = int(pad_n)
                                    expanded_d["pad_k"] = int(pad_k)

                                    pad_str = f"{'T' if pad_m else 'F'}{'T' if pad_n else 'F'}{'T' if pad_k else 'F'}"
                                    expanded_d["name"] = (
                                        f"{dtype}_{d.get('layout', 'rcr')}_{pipeline}_{scheduler}_"
                                        f"pad{pad_str}_{tm}x{tn}x{tk}_{wm}x{wn}x{wk}"
                                    )
                                    expanded_d["wildcard"] = False
                                    expanded.append(expanded_d)

    if not expanded:
        # No valid combinations found, return single default
        print(f"  Warning: No valid combinations for {tm}x{tn}x{tk} on {arch}")
        d["wave_m"] = 2
        d["wave_n"] = 2
        d["wave_k"] = 1
        d["warp_m"] = 32
        d["warp_n"] = 32
        d["warp_k"] = 16
        d["pipeline"] = "compv4"
        d["scheduler"] = "intrawave"
        d["epilogue"] = "cshuffle"
        return [d]

    return expanded


def auto_fill_declaration(decl: dict) -> dict:
    """Auto-fill with single default (for backward compat)."""
    expanded = expand_declaration_with_arch_filter(decl, decl.get("arch", "gfx942"))
    return expanded[0] if expanded else decl


# =============================================================================
# Build Functions
# =============================================================================


def generate_kernels(declarations: list, gpu_target: str = "gfx942") -> int:
    """Generate kernels using CodegenRunner from ctypes_utils."""
    kernel_dir = get_generated_kernels_dir()
    kernel_dir.mkdir(parents=True, exist_ok=True)

    # Group by dtype+layout for efficient generation
    groups = {}
    for decl in declarations:
        dtype = decl.get("dtype_a", decl.get("dtype", "fp16"))
        layout = decl.get("layout", "rcr")
        key = (dtype, layout)
        if key not in groups:
            groups[key] = []
        groups[key].append(auto_fill_declaration(decl))

    total_generated = 0

    for (dtype, layout), decls in groups.items():
        print(f"  Generating {dtype} {layout} kernels...")

        # Check for wildcards - if any decl is wildcard, generate all
        has_wildcard = any(d.get("wildcard", False) for d in decls)

        # Use CodegenRunner from ctypes_utils
        runner = CodegenRunner(
            datatype=dtype,
            layout=layout,
            gpu_target=gpu_target,
        )

        result = runner.generate("standard")

        if result.success:
            total_generated += result.kernel_count
            if has_wildcard:
                print(f"    [wildcard] Generated all {result.kernel_count} variants")
        else:
            print_error(f"    Failed: {result.stderr[:200]}")

    return total_generated


def find_kernel_header(decl: dict) -> Path:
    """Find a matching kernel header file for a declaration."""
    kernel_dir = get_generated_kernels_dir()

    dtype = decl.get("dtype_a", decl.get("dtype", "fp16"))
    layout = decl.get("layout", "rcr")
    tile_m = decl.get("tile_m", -1)
    tile_n = decl.get("tile_n", -1)
    tile_k = decl.get("tile_k", -1)

    def is_standard_kernel(path: Path) -> bool:
        """Check if this is a standard GEMM kernel (not preshuffle/multid/etc)"""
        name = path.name
        excludes = ["preshuffle", "multid", "Gelu", "Relu", "multi_d"]
        return not any(ex in name for ex in excludes)

    # Try exact tile match first (standard kernels only)
    if tile_m > 0 and tile_n > 0 and tile_k > 0:
        pattern = f"gemm_{dtype}_{layout}*_{tile_m}x{tile_n}x{tile_k}_*.hpp"
        matches = [p for p in kernel_dir.glob(pattern) if is_standard_kernel(p)]
        if matches:
            return matches[0]

    # Fall back to any matching dtype/layout (standard kernels)
    pattern = f"gemm_{dtype}_{layout}*.hpp"
    matches = [p for p in kernel_dir.glob(pattern) if is_standard_kernel(p)]
    if matches:
        # Prefer 128x128x32 tiles
        for m in matches:
            if "128x128x32" in m.name:
                return m
        return matches[0]

    # Fall back to any standard kernel
    matches = [p for p in kernel_dir.glob("gemm_*.hpp") if is_standard_kernel(p)]
    return matches[0] if matches else None


def find_conv_kernel_header(decl: dict) -> Path:
    """Find a matching convolution kernel header file."""
    kernel_dir = get_generated_kernels_dir()

    dtype = decl.get("dtype", "fp16")
    conv_type = decl.get("conv_type", "forward")
    num_dims = decl.get("num_dims", 2)
    tile_k = decl.get("tile_k", -1)
    tile_c = decl.get("tile_c", -1)

    # Map conv_type to filename prefix
    type_prefix = "fwd" if conv_type == "forward" else conv_type.replace("bwd_", "")

    # Try exact match first
    if tile_k > 0 and tile_c > 0:
        pattern = f"conv_{type_prefix}_{dtype}_{num_dims}d_*_{tile_k}x{tile_c}*.hpp"
        matches = list(kernel_dir.glob(pattern))
        if matches:
            return matches[0]

    # Fall back to any matching dtype and conv_type
    pattern = f"conv_{type_prefix}_{dtype}_{num_dims}d_*.hpp"
    matches = list(kernel_dir.glob(pattern))
    if matches:
        return matches[0]

    # Fall back to any conv kernel
    pattern = f"conv_{type_prefix}_*.hpp"
    matches = list(kernel_dir.glob(pattern))
    if matches:
        return matches[0]

    # Fall back to any conv kernel at all
    matches = list(kernel_dir.glob("conv_*.hpp"))
    return matches[0] if matches else None


def build_dispatcher_library(hipcc: str) -> bool:
    """Build the dispatcher library if needed."""
    build_dir = get_build_dir()
    lib_path = build_dir / "libck_tile_dispatcher.a"

    if lib_path.exists():
        return True

    print("  Building dispatcher library...")
    build_dir.mkdir(parents=True, exist_ok=True)

    dispatcher_dir = get_dispatcher_root()

    # Run cmake
    cmake_cmd = ["cmake", str(dispatcher_dir), f"-DCMAKE_CXX_COMPILER={hipcc}"]
    result = subprocess.run(
        cmake_cmd, cwd=str(build_dir), capture_output=True, text=True
    )
    if result.returncode != 0:
        print_error(f"CMake failed: {result.stderr}")
        return False

    # Run make
    make_cmd = ["make", "ck_tile_dispatcher", f"-j{os.cpu_count() or 4}"]
    result = subprocess.run(
        make_cmd, cwd=str(build_dir), capture_output=True, text=True
    )
    if result.returncode != 0:
        print_error(f"Make failed: {result.stderr}")
        return False

    return True


def compile_application(
    source_file: Path,
    output_bin: Path,
    kernel_header: Path,
    hipcc: str,
    gpu_target: str = "gfx942",
) -> bool:
    """Compile the application with hipcc."""
    ck_root = get_ck_root()
    dispatcher_dir = get_dispatcher_root()
    build_dir = get_build_dir()
    kernel_dir = get_generated_kernels_dir()

    includes = [
        f"-I{ck_root / 'include'}",
        f"-I{dispatcher_dir / 'include'}",
        f"-I{kernel_dir}",
    ]

    cmd = [
        hipcc,
        "-std=c++17",
        "-O3",
        f"--offload-arch={gpu_target}",
        *includes,
        "-include",
        str(kernel_header),
        f"-L{build_dir}",
        "-lck_tile_dispatcher",
        "-o",
        str(output_bin),
        str(source_file),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Filter out nodiscard warnings
    if result.stderr:
        lines = result.stderr.split("\n")
        errors = [line for line in lines if "error:" in line.lower()]
        if errors:
            for err_line in errors[:5]:
                print_error(f"  {err_line}")

    return result.returncode == 0


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Build CK Tile application with declarative kernels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python3 compile_gemm_examples.py examples/cpp/01_basic_gemm_declarative.cpp my_app
    
In your C++ code, declare kernels like:
    DECLARE_GEMM_KERNEL(fp16, rcr, 128, 128, 32);
    DECLARE_GEMM_KERNEL(bf16, rcr, 256, 256, 64);
""",
    )
    parser.add_argument("source", help="Source file (.cpp)")
    parser.add_argument(
        "output", nargs="?", help="Output name (default: source basename)"
    )
    parser.add_argument(
        "--gpu-target", default="gfx942", help="GPU target architecture"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Resolve paths using utilities from ctypes_utils
    dispatcher_dir = get_dispatcher_root()
    build_dir = get_build_dir()

    source_file = Path(args.source)
    if not source_file.is_absolute():
        # Try relative to dispatcher dir first, then CWD
        candidates = [
            dispatcher_dir / args.source,
            dispatcher_dir / "examples" / args.source,  # examples/gemm/cpp/...
            Path.cwd() / args.source,
        ]
        for candidate in candidates:
            if candidate.exists():
                source_file = candidate
                break

    if not source_file.exists():
        print_error(f"Source file not found: {source_file}")
        return 1

    output_name = args.output or source_file.stem
    output_bin = build_dir / output_name

    # Ensure build directory exists
    build_dir.mkdir(parents=True, exist_ok=True)

    print_success("=== CK Tile Declarative Kernel Build ===")
    print()

    # Phase 1: Extract declarations (both GEMM and Conv)
    print_phase("Phase 1: Scanning for kernel declarations...")

    gemm_declarations = extract_kernel_declarations(source_file)
    conv_declarations = extract_conv_kernel_declarations(source_file)

    if not gemm_declarations and not conv_declarations:
        print_error("  No kernel declarations found!")
        print("  Add DECL_KERNEL_SET for GEMM or DECL_CONV_KERNEL_SET for Conv")
        return 1

    # Handle GEMM declarations
    if gemm_declarations:
        print(f"\n  GEMM: Found {len(gemm_declarations)} declaration(s)")

        # Group by kernel set
        sets = {}
        for decl in gemm_declarations:
            set_name = decl.get("set") or "(global)"
            if set_name not in sets:
                sets[set_name] = []
            sets[set_name].append(decl)

        for set_name, set_decls in sets.items():
            print(f"    [{set_name}] ({len(set_decls)} kernels):")
            for decl in set_decls[:5]:
                needs_expansion = (
                    decl.get("wave_m", -1) < 0 or decl.get("warp_m", -1) < 0
                )
                suffix = " [expands]" if needs_expansion else ""
                display_name = (
                    decl["name"].split(":")[-1] if ":" in decl["name"] else decl["name"]
                )
                print(f"      - {display_name}{suffix}")
            if len(set_decls) > 5:
                print(f"      ... and {len(set_decls) - 5} more")

        # Expand GEMM declarations
        expanded_gemm = []
        for decl in gemm_declarations:
            arch = decl.get("arch", args.gpu_target)
            expanded = expand_declaration_with_arch_filter(decl, arch)
            expanded_gemm.extend(expanded)

        if len(expanded_gemm) > len(gemm_declarations):
            print(f"\n    Expanded to {len(expanded_gemm)} GEMM configurations")

        gemm_declarations = expanded_gemm

    # Handle Conv declarations
    if conv_declarations:
        print(f"\n  CONV: Found {len(conv_declarations)} declaration(s)")

        # Group by kernel set
        sets = {}
        for decl in conv_declarations:
            set_name = decl.get("set") or "(global)"
            if set_name not in sets:
                sets[set_name] = []
            sets[set_name].append(decl)

        for set_name, set_decls in sets.items():
            print(f"    [{set_name}] ({len(set_decls)} kernels):")
            for decl in set_decls[:5]:
                needs_expansion = (
                    decl.get("wave_m", -1) < 0 or decl.get("warp_m", -1) < 0
                )
                suffix = " [expands]" if needs_expansion else ""
                display_name = (
                    decl["name"].split(":")[-1] if ":" in decl["name"] else decl["name"]
                )
                print(f"      - {display_name}{suffix}")
            if len(set_decls) > 5:
                print(f"      ... and {len(set_decls) - 5} more")

        # Expand Conv declarations
        expanded_conv = []
        for decl in conv_declarations:
            arch = decl.get("arch", args.gpu_target)
            expanded = expand_conv_declaration_with_arch_filter(decl, arch)
            expanded_conv.extend(expanded)

        if len(expanded_conv) > len(conv_declarations):
            print(f"\n    Expanded to {len(expanded_conv)} CONV configurations")

        conv_declarations = expanded_conv

    print()

    # Phase 2: Generate kernels
    print_phase("Phase 2: Generating kernels...")

    total_generated = 0

    # Generate GEMM kernels
    if gemm_declarations:
        print("  GEMM kernels:")
        num_gemm = generate_kernels(gemm_declarations, args.gpu_target)
        total_generated += num_gemm
        print(f"    Generated: {num_gemm}")

    # Generate Conv kernels
    if conv_declarations:
        print("  CONV kernels:")
        num_conv = generate_conv_kernels(conv_declarations, args.gpu_target)
        total_generated += num_conv
        print(f"    Generated: {num_conv}")

    print(f"  Total kernel files: {total_generated}")
    print()

    # Phase 3: Find kernel header
    print_phase("Phase 3: Selecting kernel for compilation...")

    kernel_headers = []

    # Find GEMM kernel header
    if gemm_declarations:
        first_gemm = gemm_declarations[0]
        gemm_header = find_kernel_header(first_gemm)
        if gemm_header:
            kernel_headers.append(gemm_header)
            print(f"  GEMM: {gemm_header.name}")

    # Find Conv kernel header
    if conv_declarations:
        first_conv = conv_declarations[0]
        conv_header = find_conv_kernel_header(first_conv)
        if conv_header:
            kernel_headers.append(conv_header)
            print(f"  CONV: {conv_header.name}")

    if not kernel_headers:
        print_error("  No kernel headers found!")
        return 1

    # Use first available header (can be extended to use multiple)
    kernel_header = kernel_headers[0]
    print()

    # Phase 4: Build dispatcher library
    print_phase("Phase 4: Building dispatcher library...")
    hipcc = find_hipcc()

    if not build_dispatcher_library(hipcc):
        print_error("  Failed to build dispatcher library!")
        return 1
    print("  Done")
    print()

    # Phase 5: Compile application
    print_phase("Phase 5: Compiling application...")

    if not compile_application(
        source_file, output_bin, kernel_header, hipcc, args.gpu_target
    ):
        print_error("  Compilation failed!")
        return 1

    print(f"  Output: {output_bin}")
    print()

    # Done
    print_success("=== Build Complete ===")
    print()
    print("Run with:")
    print(f"  {output_bin}")
    print()
    print("List declared kernels:")
    print(f"  {output_bin} --list-kernels")

    return 0


if __name__ == "__main__":
    sys.exit(main())
