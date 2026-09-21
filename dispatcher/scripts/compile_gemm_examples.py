#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

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
    """Extract GROUPED CONVOLUTION kernel declarations from C++ source file.

    Supports DECL_GROUPED_CONV_KERNEL_SET macro with ConvSig/ConvAlgo pattern.
    Extracts all parameters: dtype, layout, conv_type, dims, tile, wave, warp, pipeline, scheduler.
    """
    content = source_file.read_text()
    declarations = []
    seen = set()

    # Pattern: DECL_GROUPED_CONV_KERNEL_SET(name, .add(...).add(...))
    set_pattern = r"DECL_GROUPED_CONV_KERNEL_SET\s*\(\s*(\w+)\s*,([^;]+)\)"

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
                        "num_dims": 2,
                        "groups": 1,
                        "tile_n": 1,
                        "tile_k": tile_k,
                        "tile_c": tile_c,
                        "wave_m": -1,  # Wildcard - will expand
                        "wave_n": -1,
                        "wave_k": 1,
                        "warp_m": -1,
                        "warp_n": -1,
                        "warp_k": 16,
                        "pipeline": "compv3",
                        "scheduler": "intrawave",
                        "epilogue": "cshuffle",
                        "name": name,
                        "set": set_name,
                        "arch": "gfx942",
                    }
                )

        # Pattern 2: Full specification with ConvSig() and ConvAlgo()
        # Match .add( ConvSig()..., ConvAlgo()..., "arch" )
        # Use robust parsing that handles multi-line and comments

        # Find all .add( blocks containing ConvSig
        add_blocks = re.findall(
            r"\.add\s*\(\s*ConvSig\(\)([\s\S]*?)(?=\.add\s*\(|$)", set_body
        )

        for add_block in add_blocks:
            # Find ConvAlgo and arch in this block
            algo_match = re.search(r'ConvAlgo\(\)([\s\S]*?),\s*"(\w+)"\s*\)', add_block)
            if not algo_match:
                continue

            sig_str = add_block[: add_block.find("ConvAlgo()")]
            algo_str = algo_match.group(1)
            arch = algo_match.group(2)

            # Parse ConvSig
            dtype = "fp16"
            dtype_match = re.search(r'\.dtype\s*\(\s*"([^"]+)"', sig_str)
            if dtype_match:
                dtype = dtype_match.group(1)

            layout = "nhwgc"
            layout_match = re.search(r'\.layout\s*\(\s*"([^"]+)"', sig_str)
            if layout_match:
                layout = layout_match.group(1)

            conv_type = "forward"
            conv_type_match = re.search(r'\.conv_type\s*\(\s*"([^"]+)"', sig_str)
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

            # Parse ConvAlgo
            tile_n, tile_k, tile_c = 1, 128, 128
            tile_match = re.search(
                r"\.tile\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", algo_str
            )
            if tile_match:
                tile_n = int(tile_match.group(1))
                tile_k = int(tile_match.group(2))
                tile_c = int(tile_match.group(3))

            wave_m, wave_n, wave_k = 2, 2, 1
            wave_match = re.search(
                r"\.wave\s*\(\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*(\d+))?\s*\)", algo_str
            )
            if wave_match:
                wave_m = int(wave_match.group(1))
                wave_n = int(wave_match.group(2))
                wave_k = int(wave_match.group(3) or 1)

            warp_m, warp_n, warp_k = 32, 32, 16
            warp_match = re.search(
                r"\.warp\s*\(\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*(\d+))?\s*\)", algo_str
            )
            if warp_match:
                warp_m = int(warp_match.group(1))
                warp_n = int(warp_match.group(2))
                warp_k = int(warp_match.group(3) or 16)

            pipeline = "compv3"
            pipeline_match = re.search(r'\.pipeline\s*\(\s*"([^"]+)"', algo_str)
            if pipeline_match:
                pipeline = pipeline_match.group(1)

            scheduler = "intrawave"
            scheduler_match = re.search(r'\.scheduler\s*\(\s*"([^"]+)"', algo_str)
            if scheduler_match:
                scheduler = scheduler_match.group(1)

            epilogue = "cshuffle"
            epilogue_match = re.search(r'\.epilogue\s*\(\s*"([^"]+)"', algo_str)
            if epilogue_match:
                epilogue = epilogue_match.group(1)

            # Build unique name with full config
            name = f"{set_name}:{dtype}_{conv_type}_{num_dims}d_{pipeline}_{scheduler}_{tile_k}x{tile_c}_{wave_m}x{wave_n}x{wave_k}"
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
                        "epilogue": epilogue,
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
    """Generate grouped convolution kernels using unified_grouped_conv_codegen."""
    kernel_dir = get_generated_kernels_dir()
    kernel_dir.mkdir(parents=True, exist_ok=True)

    codegen_dir = get_dispatcher_root() / "codegen"
    sys.path.insert(0, str(codegen_dir))

    try:
        from unified_grouped_conv_codegen import (
            UnifiedGroupedConvCodegen as UnifiedConvCodegen,
            GroupedConvKernelConfig as ConvKernelConfig,
            GroupedConvVariant as ConvVariant,
            TileConfig,
            GroupedConvTraitConfig as TraitConfig,
        )
    except ImportError as e:
        print_error(f"  Failed to import grouped conv codegen: {e}")
        return 0

    codegen = UnifiedConvCodegen(kernel_dir)
    total_generated = 0

    # Group by dtype and variant for efficient generation
    groups = {}
    for decl in declarations:
        dtype = decl.get("dtype", "fp16")
        conv_type = decl.get("conv_type", "forward")
        num_dims = decl.get("num_dims", 2)
        key = (dtype, conv_type, num_dims)
        if key not in groups:
            groups[key] = []
        groups[key].append(decl)

    for (dtype, conv_type, num_dims), decls in groups.items():
        print(f"    Generating {dtype} {conv_type} {num_dims}D kernels...")

        # Map to ConvVariant
        variant = ConvVariant.FORWARD
        if conv_type == "bwd_data":
            variant = ConvVariant.BACKWARD_DATA
        elif conv_type == "bwd_weight":
            variant = ConvVariant.BACKWARD_WEIGHT

        for decl in decls:
            pipeline = decl.get("pipeline", "compv3")
            scheduler = decl.get("scheduler", "intrawave")
            epilogue = decl.get("epilogue", "cshuffle")

            tile_k = decl.get("tile_k", 128)
            tile_c = decl.get("tile_c", 128)
            wave_m = decl.get("wave_m", 2)
            wave_n = decl.get("wave_n", 2)
            warp_m = decl.get("warp_m", 32)
            warp_n = decl.get("warp_n", 32)
            warp_k = decl.get("warp_k", 16)

            # Adjust tile_k for compv4
            adj_tile_k = 64 * 2 if pipeline == "compv4" else 64

            # Create TileConfig
            tile_config = TileConfig(
                tile_m=tile_k,  # K is M in conv GEMM view
                tile_n=tile_c,  # C is N in conv GEMM view
                tile_k=adj_tile_k,
                warp_m=wave_m,
                warp_n=wave_n,
                warp_k=1,
                warp_tile_m=warp_m,
                warp_tile_n=warp_n,
                warp_tile_k=warp_k,
            )

            # Create TraitConfig
            trait_config = TraitConfig(
                pipeline=pipeline,
                scheduler=scheduler,
                epilogue=epilogue,
                double_smem_buffer=(pipeline == "compv4"),
                pad_m=True,
                pad_n=True,
                pad_k=True,
            )

            # Create ConvKernelConfig
            config = ConvKernelConfig(
                tile=tile_config,
                trait=trait_config,
                variant=variant,
                ndim_spatial=num_dims,
                arch=gpu_target,
            )

            try:
                filepath = codegen.generate_kernel(config, dtype)
                total_generated += 1
                print(f"      Generated: {filepath.name}")
            except Exception as e:
                print_error(f"      Failed to generate {decl['name']}: {e}")

    return total_generated


# Original GEMM extraction continues here
def extract_kernel_declarations(source_file: Path) -> list:
    """Extract GEMM kernel declarations from C++ source file."""
    content = source_file.read_text()
    declarations = []
    seen = set()

    # -------------------------------------------------------------------------
    # Pattern 1: Simple DECL_KERNEL_SIMPLE(dtype, layout, tile_m, tile_n, tile_k)
    # -------------------------------------------------------------------------
    legacy_pattern = r"DECL_KERNEL_SIMPLE\s*\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)"
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
    # Match only DECL_KERNEL_SET at start of line (not in comments)
    # -------------------------------------------------------------------------
    set_pattern = r"^DECL_KERNEL_SET\s*\(\s*(\w+)\s*,([\s\S]*?)\)\s*;"
    for match in re.finditer(set_pattern, content, re.MULTILINE):
        set_name = match.group(1)
        set_body = match.group(2)

        # Parse .add("dtype", "layout", tm, tn, tk) calls - simple form
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

        # Parse .add(Signature()..., Algorithm()..., "arch") fluent calls
        # Robust approach: find each .add( block and parse methods individually
        # This handles any method order and optional methods

        # Split set_body into .add() blocks
        add_blocks = []
        add_starts = [m.start() for m in re.finditer(r"\.add\s*\(", set_body)]

        for i, start in enumerate(add_starts):
            # Find the matching closing paren by counting parens
            depth = 0
            end = start
            in_string = False
            escape_next = False

            for j, ch in enumerate(set_body[start:], start):
                if escape_next:
                    escape_next = False
                    continue
                if ch == "\\":
                    escape_next = True
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        end = j + 1
                        break

            if end > start:
                add_blocks.append(set_body[start:end])

        for add_block in add_blocks:
            # Skip if doesn't have both Signature() and Algorithm()
            if "Signature()" not in add_block or "Algorithm()" not in add_block:
                continue

            # Split on Algorithm() to separate Signature and Algorithm parts
            algo_idx = add_block.find("Algorithm()")
            if algo_idx == -1:
                continue

            sig_str = add_block[:algo_idx]
            algo_str = add_block[algo_idx:]  # Include Algorithm() and everything after

            # Parse dtype from Signature - handles .dtype("fp16", "fp16", "fp16", "fp32")
            dtype = "fp16"
            dtype_m = re.search(r'\.dtype\s*\(\s*"([^"]+)"', sig_str)
            if dtype_m:
                dtype = dtype_m.group(1)

            # Parse layout from Signature - handles .layout("row", "col", "row")
            layout = "rcr"
            layout_m = re.search(
                r'\.layout\s*\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"', sig_str
            )
            if layout_m:
                la, lb, lc = layout_m.group(1), layout_m.group(2), layout_m.group(3)
                layout = (
                    ("r" if la == "row" else "c")
                    + ("r" if lb == "row" else "c")
                    + ("r" if lc == "row" else "c")
                )
            else:
                # Single arg form: .layout("rcr")
                layout_m = re.search(r'\.layout\s*\(\s*"([^"]+)"', sig_str)
                if layout_m:
                    layout = layout_m.group(1)

            # Parse tile from Algorithm
            tm, tn, tk = 128, 128, 32
            tile_m = re.search(
                r"\.tile\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", algo_str
            )
            if tile_m:
                tm, tn, tk = (
                    int(tile_m.group(1)),
                    int(tile_m.group(2)),
                    int(tile_m.group(3)),
                )

            # Parse wave
            wave_m, wave_n, wave_k = 2, 2, 1
            wave_match = re.search(
                r"\.wave\s*\(\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*(\d+))?\s*\)", algo_str
            )
            if wave_match:
                wave_m, wave_n = int(wave_match.group(1)), int(wave_match.group(2))
                wave_k = int(wave_match.group(3) or 1)

            # Parse warp
            warp_m, warp_n, warp_k = 32, 32, 16
            warp_match = re.search(
                r"\.warp\s*\(\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*(\d+))?\s*\)", algo_str
            )
            if warp_match:
                warp_m, warp_n = int(warp_match.group(1)), int(warp_match.group(2))
                warp_k = int(warp_match.group(3) or 16)

            # Parse pipeline - NEW: extract from declaration
            pipeline = "compv4"
            pipeline_m = re.search(r'\.pipeline\s*\(\s*"([^"]+)"', algo_str)
            if pipeline_m:
                pipeline = pipeline_m.group(1)

            # Parse scheduler - NEW: extract from declaration
            scheduler = "intrawave"
            scheduler_m = re.search(r'\.scheduler\s*\(\s*"([^"]+)"', algo_str)
            if scheduler_m:
                scheduler = scheduler_m.group(1)

            # Parse epilogue - NEW: extract from declaration
            epilogue = "cshuffle"
            epilogue_m = re.search(r'\.epilogue\s*\(\s*"([^"]+)"', algo_str)
            if epilogue_m:
                epilogue = epilogue_m.group(1)

            # Parse padding - NEW: extract from declaration
            pad_m, pad_n, pad_k = False, False, False
            pad_match = re.search(
                r"\.pad\s*\(\s*(true|false)\s*,\s*(true|false)\s*,\s*(true|false)\s*\)",
                algo_str,
                re.IGNORECASE,
            )
            if pad_match:
                pad_m = pad_match.group(1).lower() == "true"
                pad_n = pad_match.group(2).lower() == "true"
                pad_k = pad_match.group(3).lower() == "true"

            # Parse elementwise from Signature - for Multi-D kernels
            elementwise_op = "PassThrough"
            num_d_tensors = 0
            elem_match = re.search(
                r'\.elementwise\s*\(\s*"([^"]+)"\s*,\s*(\d+)\s*\)',
                sig_str,
            )
            if elem_match:
                elementwise_op = elem_match.group(1)
                num_d_tensors = int(elem_match.group(2))

            name = f"{set_name}:{dtype}_{layout}_{pipeline}_{scheduler}_{tm}x{tn}x{tk}_{wave_m}x{wave_n}x{wave_k}"
            if elementwise_op != "PassThrough":
                name += f"_{elementwise_op}_d{num_d_tensors}"
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
                        "pipeline": pipeline,
                        "scheduler": scheduler,
                        "epilogue": epilogue,
                        "pad_m": pad_m,
                        "pad_n": pad_n,
                        "pad_k": pad_k,
                        "elementwise_op": elementwise_op,
                        "num_d_tensors": num_d_tensors,
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

    # Wave configurations
    if needs_wave_expansion:
        wave_configs = WARP_SUPPORTED_COMBINATIONS.get(arch, [[2, 2, 1]])
    else:
        wave_configs = [[d.get("wave_m", 2), d.get("wave_n", 2), d.get("wave_k", 1)]]

    # Warp tile configurations
    if needs_warp_expansion:
        arch_warp_tiles = WARP_TILE_SUPPORTED_COMBINATIONS.get(arch, {})

        # Try to find warp tile configs for this dtype
        # Keys are like: fp16_fp16_fp32, int8_int8_int32, etc.
        warp_tile_configs = None
        dtype_key_variants = [
            f"{dtype}_{dtype}_{dtype}",  # e.g., fp32_fp32_fp32
            f"{dtype}_{dtype}_fp32",  # e.g., fp16_fp16_fp32
            f"{dtype}_{dtype}_int32",  # e.g., int8_int8_int32
        ]
        for dtype_key in dtype_key_variants:
            warp_tile_configs = arch_warp_tiles.get(dtype_key, None)
            if warp_tile_configs is not None:
                break

        # If dtype is not supported on this arch, return empty list
        if warp_tile_configs is None:
            return []
    else:
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


def get_arch_filter_data():
    """Load arch filter data from arch_specs_generated if available."""
    codegen_dir = get_dispatcher_root() / "codegen"
    sys.path.insert(0, str(codegen_dir))

    try:
        from arch_specs_generated import (
            TRAIT_UNSUPPORTED_COMBINATIONS,
            WARP_SUPPORTED_COMBINATIONS,
            WARP_TILE_SUPPORTED_COMBINATIONS,
            get_supported_archs,
        )

        return {
            "trait_unsupported": TRAIT_UNSUPPORTED_COMBINATIONS,
            "warp_combos": WARP_SUPPORTED_COMBINATIONS,
            "warp_tile_combos": WARP_TILE_SUPPORTED_COMBINATIONS,
            "supported_archs": get_supported_archs(),
        }
    except ImportError:
        # Fallback defaults
        return {
            "trait_unsupported": {
                ("compv3", "cshuffle", "interwave"),
                ("compv3", "default", "interwave"),
                ("compv4", "cshuffle", "interwave"),
                ("compv4", "default", "interwave"),
            },
            "warp_combos": {
                "gfx942": [[1, 4, 1], [2, 2, 1], [4, 1, 1]],
            },
            "warp_tile_combos": {
                "gfx942": {"fp16_fp16_fp16": [[16, 16, 16], [32, 32, 16]]},
            },
            "supported_archs": ["gfx90a", "gfx942", "gfx950"],
        }


def is_wildcard_declaration(decl: dict) -> bool:
    """Check if declaration has wildcards that need expansion."""
    # Wave/warp wildcards
    if decl.get("wave_m", 2) < 0 or decl.get("wave_n", 2) < 0:
        return True
    if decl.get("warp_m", 32) < 0 or decl.get("warp_n", 32) < 0:
        return True
    # Pipeline/scheduler wildcards
    if decl.get("pipeline", "compv4") == "*":
        return True
    if decl.get("scheduler", "intrawave") == "*":
        return True
    if decl.get("epilogue", "cshuffle") == "*":
        return True
    return False


def validate_kernel_config(decl: dict, arch: str = "gfx942") -> tuple:
    """Validate a kernel configuration against known supported combinations.

    Uses arch_specs_generated for architecture-specific validation.

    For wildcard declarations (-1 values or "*" strings), validation is skipped
    because the expansion phase will generate only valid combinations.

    Returns: (is_valid, error_message)
    """
    # Skip validation for wildcards - expansion will filter invalid combos
    if is_wildcard_declaration(decl):
        return (True, None)

    arch_data = get_arch_filter_data()

    pipeline = decl.get("pipeline", "compv4")
    epilogue = decl.get("epilogue", "cshuffle")
    scheduler = decl.get("scheduler", "intrawave")
    dtype = decl.get("dtype_a", "fp16")

    wave_m = decl.get("wave_m", 2)
    wave_n = decl.get("wave_n", 2)
    wave_k = decl.get("wave_k", 1)

    warp_m = decl.get("warp_m", 32)
    warp_n = decl.get("warp_n", 32)
    warp_k = decl.get("warp_k", 16)

    errors = []

    # Check trait combination (pipeline, epilogue, scheduler)
    combo = (pipeline, epilogue, scheduler)
    if combo in arch_data["trait_unsupported"]:
        errors.append(
            f"Unsupported trait combination: pipeline={pipeline}, epilogue={epilogue}, scheduler={scheduler}\n"
            f"    Valid schedulers for {pipeline}+{epilogue}: intrawave"
        )

    # Check wave configuration for this arch
    warp_combos = arch_data["warp_combos"].get(arch, [[2, 2, 1]])
    wave_cfg = [wave_m, wave_n, wave_k]
    if wave_cfg not in warp_combos:
        valid_str = ", ".join(f"[{c[0]},{c[1]},{c[2]}]" for c in warp_combos)
        errors.append(
            f"Unsupported wave configuration [{wave_m},{wave_n},{wave_k}] for {arch}\n"
            f"    Valid wave configs: {valid_str}"
        )

    # Check warp tile configuration for this arch and dtype
    dtype_key = f"{dtype}_{dtype}_{dtype}"
    warp_tile_combos = (
        arch_data["warp_tile_combos"]
        .get(arch, {})
        .get(dtype_key, [[32, 32, 16], [16, 16, 16]])
    )
    warp_cfg = [warp_m, warp_n, warp_k]
    if warp_cfg not in warp_tile_combos:
        valid_str = ", ".join(f"[{c[0]},{c[1]},{c[2]}]" for c in warp_tile_combos[:5])
        errors.append(
            f"Unsupported warp tile [{warp_m},{warp_n},{warp_k}] for {arch}/{dtype}\n"
            f"    Valid warp tiles: {valid_str}"
        )

    # Check arch is supported
    if arch not in arch_data["supported_archs"]:
        errors.append(
            f"Unsupported architecture: {arch}\n"
            f"    Supported: {', '.join(arch_data['supported_archs'])}"
        )

    if errors:
        return (False, "\n".join(errors))

    return (True, None)


def build_exact_kernel_filename(decl: dict) -> str:
    """Build the exact kernel filename from a fully-specified declaration.

    Standard format:
    gemm_{dtype}_{layout}_{pipeline}_{epilogue}_{scheduler}_{pad_m}_{pad_n}_{pad_k}_{preshuffle}_{tile}_{wave}_{warp}.hpp

    Multi-D format:
    gemm_{dtype}_{layout}_{pipeline}_{epilogue}_{scheduler}_{pad_m}_{pad_n}_{pad_k}_{preshuffle}_{tile}_{wave}_{warp}_multid_{op}_d{num}.hpp
    """
    dtype = decl.get("dtype_a", decl.get("dtype", "fp16"))
    layout = decl.get("layout", "rcr")
    pipeline = decl.get("pipeline", "compv4")
    epilogue = decl.get("epilogue", "cshuffle")
    scheduler = decl.get("scheduler", "intrawave")

    pad_m = "True" if decl.get("pad_m", False) else "False"
    pad_n = "True" if decl.get("pad_n", False) else "False"
    pad_k = "True" if decl.get("pad_k", False) else "False"
    preshuffle = "True" if decl.get("preshuffle", False) else "False"

    tile_m = decl.get("tile_m", 128)
    tile_n = decl.get("tile_n", 128)
    tile_k = decl.get("tile_k", 32)

    wave_m = decl.get("wave_m", 2)
    wave_n = decl.get("wave_n", 2)
    wave_k = decl.get("wave_k", 1)

    warp_m = decl.get("warp_m", 32)
    warp_n = decl.get("warp_n", 32)
    warp_k = decl.get("warp_k", 16)

    tile_str = f"{tile_m}x{tile_n}x{tile_k}"
    wave_str = f"{wave_m}x{wave_n}x{wave_k}"
    warp_str = f"{warp_m}x{warp_n}x{warp_k}"

    base = f"gemm_{dtype}_{layout}_{pipeline}_{epilogue}_{scheduler}_{pad_m}_{pad_n}_{pad_k}_{preshuffle}_{tile_str}_{wave_str}_{warp_str}"

    # Handle Multi-D kernels
    elementwise_op = decl.get("elementwise_op", "PassThrough")
    num_d_tensors = decl.get("num_d_tensors", 0)
    if elementwise_op != "PassThrough" and num_d_tensors > 0:
        base += f"_multid_{elementwise_op}_d{num_d_tensors}"

    return f"{base}.hpp"


def generate_specific_kernel(decl: dict, gpu_target: str = "gfx942") -> bool:
    """Generate a specific kernel based on declaration."""
    dtype = decl.get("dtype_a", decl.get("dtype", "fp16"))
    layout = decl.get("layout", "rcr")

    print(f"    Generating kernel for {dtype}/{layout}...")

    # Use CodegenRunner to generate
    runner = CodegenRunner(
        datatype=dtype,
        layout=layout,
        gpu_target=gpu_target,
    )

    result = runner.generate("standard")
    return result.success


def find_kernel_header(decl: dict, gpu_target: str = "gfx942") -> Path:
    """Find a matching kernel header file for a declaration.

    Tries multiple matching strategies:
    1. Exact filename match
    2. Match with key parameters (dtype, layout, pipeline, scheduler, tile)
    3. Match with just dtype, layout, and tile (more flexible)
    4. Any kernel with matching dtype and layout

    If no kernel exists, attempts to generate it.
    Returns None only if all strategies fail.
    """
    kernel_dir = get_generated_kernels_dir()

    dtype = decl.get("dtype_a", decl.get("dtype", "fp16"))
    layout = decl.get("layout", "rcr")
    pipeline = decl.get("pipeline", "compv4")
    scheduler = decl.get("scheduler", "intrawave")
    tile_m = decl.get("tile_m", 128)
    tile_n = decl.get("tile_n", 128)
    tile_k = decl.get("tile_k", 32)
    wave_m = decl.get("wave_m", 2)
    wave_n = decl.get("wave_n", 2)
    wave_k = decl.get("wave_k", 1)

    tile_str = f"{tile_m}x{tile_n}x{tile_k}"
    wave_str = f"{wave_m}x{wave_n}x{wave_k}"

    # Build exact filename
    exact_filename = build_exact_kernel_filename(decl)
    exact_path = kernel_dir / exact_filename

    # Strategy 1: Exact filename match
    if exact_path.exists():
        print(f"    Found exact kernel: {exact_filename}")
        return exact_path

    # Strategy 2: Match with key parameters
    pattern = (
        f"gemm_{dtype}_{layout}_{pipeline}_*_{scheduler}_*_{tile_str}_{wave_str}_*.hpp"
    )
    matches = list(kernel_dir.glob(pattern))
    if matches:
        print(f"    Found matching kernel: {matches[0].name}")
        return matches[0]

    # Strategy 3: Match with just dtype, layout, tile (ignore wave/warp)
    pattern = f"gemm_{dtype}_{layout}_{pipeline}_*_{scheduler}_*_{tile_str}_*.hpp"
    matches = list(kernel_dir.glob(pattern))
    if matches:
        print(f"    Found kernel with matching tile: {matches[0].name}")
        return matches[0]

    # Strategy 4: Match with just dtype, layout (most flexible, for wildcards)
    # Prefer kernels with intrawave scheduler (known to work)
    pattern = f"gemm_{dtype}_{layout}_*_intrawave_*_{tile_str}_*.hpp"
    matches = list(kernel_dir.glob(pattern))
    if matches:
        print(f"    Found kernel with intrawave: {matches[0].name}")
        return matches[0]

    # Strategy 5: Any kernel with matching dtype and layout
    pattern = f"gemm_{dtype}_{layout}_*_{tile_str}_*.hpp"
    matches = list(kernel_dir.glob(pattern))
    if matches:
        print(f"    Found kernel with matching dtype/layout/tile: {matches[0].name}")
        return matches[0]

    # Strategy 6: Try to generate the kernel
    print("    No matching kernel found, attempting to generate...")
    if generate_specific_kernel(decl, gpu_target):
        # Check strategies again after generation
        for pattern in [
            f"gemm_{dtype}_{layout}_{pipeline}_*_{scheduler}_*_{tile_str}_*.hpp",
            f"gemm_{dtype}_{layout}_*_intrawave_*_{tile_str}_*.hpp",
            f"gemm_{dtype}_{layout}_*_{tile_str}_*.hpp",
        ]:
            matches = list(kernel_dir.glob(pattern))
            if matches:
                print(f"    Generated: {matches[0].name}")
                return matches[0]

    # All strategies failed - return None (caller will try next expanded decl)
    return None


def is_conv_wildcard_declaration(decl: dict) -> bool:
    """Check if conv declaration has wildcards that need expansion."""
    if decl.get("wave_m", 2) < 0 or decl.get("wave_n", 2) < 0:
        return True
    if decl.get("warp_m", 32) < 0 or decl.get("warp_n", 32) < 0:
        return True
    if decl.get("pipeline", "compv3") == "*":
        return True
    if decl.get("scheduler", "intrawave") == "*":
        return True
    return False


def validate_conv_kernel_config(decl: dict, arch: str = "gfx942") -> tuple:
    """Validate a conv kernel configuration against arch filter.

    For wildcard declarations, validation is skipped (expansion handles it).

    Returns: (is_valid, error_message)
    """
    # Skip validation for wildcards
    if is_conv_wildcard_declaration(decl):
        return (True, None)

    arch_data = get_arch_filter_data()

    pipeline = decl.get("pipeline", "compv3")
    epilogue = decl.get("epilogue", "cshuffle")
    scheduler = decl.get("scheduler", "intrawave")
    dtype = decl.get("dtype", "fp16")

    wave_m = decl.get("wave_m", 2)
    wave_n = decl.get("wave_n", 2)
    wave_k = decl.get("wave_k", 1)

    warp_m = decl.get("warp_m", 32)
    warp_n = decl.get("warp_n", 32)
    warp_k = decl.get("warp_k", 16)

    errors = []

    # Check trait combination
    combo = (pipeline, epilogue, scheduler)
    if combo in arch_data["trait_unsupported"]:
        errors.append(
            f"Unsupported trait combination: pipeline={pipeline}, epilogue={epilogue}, scheduler={scheduler}\n"
            f"    Valid schedulers for {pipeline}+{epilogue}: intrawave"
        )

    # Check wave configuration
    warp_combos = arch_data["warp_combos"].get(arch, [[2, 2, 1]])
    wave_cfg = [wave_m, wave_n, wave_k]
    if wave_cfg not in warp_combos:
        valid_str = ", ".join(f"[{c[0]},{c[1]},{c[2]}]" for c in warp_combos)
        errors.append(
            f"Unsupported wave configuration [{wave_m},{wave_n},{wave_k}] for {arch}\n"
            f"    Valid wave configs: {valid_str}"
        )

    # Check warp tile configuration
    dtype_key = f"{dtype}_{dtype}_{dtype}"
    warp_tile_combos = (
        arch_data["warp_tile_combos"]
        .get(arch, {})
        .get(dtype_key, [[32, 32, 16], [16, 16, 16]])
    )
    warp_cfg = [warp_m, warp_n, warp_k]
    if warp_cfg not in warp_tile_combos:
        valid_str = ", ".join(f"[{c[0]},{c[1]},{c[2]}]" for c in warp_tile_combos[:5])
        errors.append(
            f"Unsupported warp tile [{warp_m},{warp_n},{warp_k}] for {arch}/{dtype}\n"
            f"    Valid warp tiles: {valid_str}"
        )

    # Check arch is supported
    if arch not in arch_data["supported_archs"]:
        errors.append(
            f"Unsupported architecture: {arch}\n"
            f"    Supported: {', '.join(arch_data['supported_archs'])}"
        )

    if errors:
        return (False, "\n".join(errors))

    return (True, None)


def build_exact_conv_kernel_filename(decl: dict) -> str:
    """Build the exact conv kernel filename from a fully-specified declaration.

    Conv filename format:
    conv_{type}_{dtype}_{ndim}d_{pipeline}_{epilogue}_{scheduler}_{tile}_{wave}.hpp

    Example:
    conv_fwd_fp16_2d_compv3_cshuffle_intrawave_128x128x32_2x2x1.hpp
    """
    dtype = decl.get("dtype", "fp16")
    conv_type = decl.get("conv_type", "forward")
    num_dims = decl.get("num_dims", 2)
    pipeline = decl.get("pipeline", "compv3")
    epilogue = decl.get("epilogue", "cshuffle")
    scheduler = decl.get("scheduler", "intrawave")

    # Map conv_type to filename prefix
    if conv_type == "forward":
        type_prefix = "fwd"
    elif conv_type == "bwd_data":
        type_prefix = "bwd_data"
    elif conv_type == "bwd_weight":
        type_prefix = "bwd_weight"
    else:
        type_prefix = conv_type

    tile_k = decl.get("tile_k", 128)
    tile_c = decl.get("tile_c", 128)

    wave_m = decl.get("wave_m", 2)
    wave_n = decl.get("wave_n", 2)
    wave_k = decl.get("wave_k", 1)

    tile_str = f"{tile_k}x{tile_c}x32"  # Conv uses tile_k x tile_c x 32 format
    wave_str = f"{wave_m}x{wave_n}x{wave_k}"

    return f"conv_{type_prefix}_{dtype}_{num_dims}d_{pipeline}_{epilogue}_{scheduler}_{tile_str}_{wave_str}.hpp"


def generate_specific_conv_kernel(decl: dict, gpu_target: str = "gfx942") -> bool:
    """Generate a specific conv kernel based on declaration."""
    dtype = decl.get("dtype", "fp16")
    conv_type = decl.get("conv_type", "forward")
    num_dims = decl.get("num_dims", 2)

    print(f"    Generating conv kernel for {dtype}/{conv_type}/{num_dims}d...")

    # Map to variant name
    if conv_type == "forward":
        variant = "forward"
    elif conv_type == "bwd_data":
        variant = "bwd_data"
    elif conv_type == "bwd_weight":
        variant = "bwd_weight"
    else:
        variant = "forward"

    # Use unified_grouped_conv_codegen
    codegen_dir = get_dispatcher_root() / "codegen"
    codegen_script = codegen_dir / "unified_grouped_conv_codegen.py"
    output_dir = get_generated_kernels_dir()

    cmd = [
        "python3",
        str(codegen_script),
        "--datatype",
        dtype,
        "--variant",
        variant,
        "--ndim",
        str(num_dims),
        "--arch",
        gpu_target,
        "--output",
        str(output_dir),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def find_conv_kernel_header(decl: dict, gpu_target: str = "gfx942") -> Path:
    """Find the EXACT matching conv kernel header file for a declaration.

    If the kernel doesn't exist, attempts to generate it.
    Returns None only if generation also fails.
    """
    kernel_dir = get_generated_kernels_dir()

    # Build exact filename
    exact_filename = build_exact_conv_kernel_filename(decl)
    exact_path = kernel_dir / exact_filename

    # Check if exact kernel exists
    if exact_path.exists():
        print(f"    Found exact conv kernel: {exact_filename}")
        return exact_path

    # Try to find with glob (in case of minor variations)
    dtype = decl.get("dtype", "fp16")
    conv_type = decl.get("conv_type", "forward")
    num_dims = decl.get("num_dims", 2)
    pipeline = decl.get("pipeline", "compv3")
    scheduler = decl.get("scheduler", "intrawave")
    tile_k = decl.get("tile_k", 128)
    tile_c = decl.get("tile_c", 128)
    wave_m = decl.get("wave_m", 2)
    wave_n = decl.get("wave_n", 2)
    wave_k = decl.get("wave_k", 1)

    # Map conv_type to prefix
    if conv_type == "forward":
        type_prefix = "fwd"
    elif conv_type == "bwd_data":
        type_prefix = "bwd_data"
    elif conv_type == "bwd_weight":
        type_prefix = "bwd_weight"
    else:
        type_prefix = conv_type

    tile_str = f"{tile_k}x{tile_c}"
    wave_str = f"{wave_m}x{wave_n}x{wave_k}"

    # Search pattern with key parameters
    pattern = f"conv_{type_prefix}_{dtype}_{num_dims}d_{pipeline}_*_{scheduler}_*{tile_str}*_{wave_str}.hpp"
    matches = list(kernel_dir.glob(pattern))

    if matches:
        print(f"    Found matching conv kernel: {matches[0].name}")
        return matches[0]

    # Kernel doesn't exist - try to generate it
    print(f"    Conv kernel not found: {exact_filename}")
    print("    Attempting to generate...")

    if generate_specific_conv_kernel(decl, gpu_target):
        # Check again after generation
        matches = list(kernel_dir.glob(pattern))
        if matches:
            print(f"    Generated: {matches[0].name}")
            return matches[0]

        # Check for exact match
        if exact_path.exists():
            print(f"    Generated: {exact_filename}")
            return exact_path

    # Still not found - print helpful error
    print_error(
        "    ERROR: Could not find or generate conv kernel matching declaration:"
    )
    print_error(f"      dtype={dtype}, conv_type={conv_type}, num_dims={num_dims}")
    print_error(f"      pipeline={pipeline}, scheduler={scheduler}")
    print_error(f"      tile={tile_k}x{tile_c}, wave={wave_str}")
    print_error(f"    Expected: {exact_filename}")
    print_error(f"    Available conv kernels in {kernel_dir}:")

    available = list(kernel_dir.glob(f"conv_{type_prefix}_{dtype}_{num_dims}d_*.hpp"))[
        :5
    ]
    for k in available:
        print_error(f"      - {k.name}")
    if len(list(kernel_dir.glob(f"conv_{type_prefix}_{dtype}_{num_dims}d_*.hpp"))) > 5:
        print_error("      ... and more")

    return None


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
    DECL_KERNEL_SET(my_kernels,
        .add(Signature().dtype("fp16").layout("rcr"),
             Algorithm().tile(128, 128, 32).wave(2, 2, 1).warp(32, 32, 16)
                        .pipeline("compv4").scheduler("intrawave"))
    );
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
        print(
            "  Add DECL_KERNEL_SET for GEMM or DECL_GROUPED_CONV_KERNEL_SET for Grouped Conv"
        )
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

        # Validate declarations against arch filter
        print(f"\n    Validating against {args.gpu_target} arch filter...")
        wildcard_count = 0
        invalid_count = 0
        auto_corrections = []

        for decl in gemm_declarations:
            arch = decl.get("arch", args.gpu_target)
            decl_name = (
                decl["name"].split(":")[-1] if ":" in decl["name"] else decl["name"]
            )

            # Check for wildcards
            if is_wildcard_declaration(decl):
                wildcard_count += 1
                continue  # Wildcards validated during expansion

            is_valid, error_msg = validate_kernel_config(decl, arch)
            if not is_valid:
                print(f"\n    WARNING Invalid configuration: {decl_name}")

                # Parse the error and show specific auto-corrections
                corrections = []
                original_values = {}

                if "wave configuration" in error_msg.lower():
                    original_values["wave"] = (
                        f"[{decl.get('wave_m', 2)}, {decl.get('wave_n', 2)}, {decl.get('wave_k', 1)}]"
                    )
                    decl["wave_m"] = -1
                    decl["wave_n"] = -1
                    corrections.append(
                        f"wave: {original_values['wave']} -> [wildcard expansion]"
                    )

                if "warp tile" in error_msg.lower():
                    original_values["warp"] = (
                        f"[{decl.get('warp_m', 32)}, {decl.get('warp_n', 32)}, {decl.get('warp_k', 16)}]"
                    )
                    decl["warp_m"] = -1
                    decl["warp_n"] = -1
                    corrections.append(
                        f"warp_tile: {original_values['warp']} -> [wildcard expansion]"
                    )

                if "trait combination" in error_msg.lower():
                    original_values["pipeline"] = decl.get("pipeline", "compv4")
                    original_values["scheduler"] = decl.get("scheduler", "intrawave")
                    decl["pipeline"] = "*"
                    decl["scheduler"] = "*"
                    corrections.append(
                        f"pipeline: {original_values['pipeline']} -> [wildcard expansion]"
                    )
                    corrections.append(
                        f"scheduler: {original_values['scheduler']} -> [wildcard expansion]"
                    )

                # Print the auto-corrections
                print("      AUTO-CORRECTION:")
                for corr in corrections:
                    print(f"        - {corr}")
                auto_corrections.append((decl_name, corrections))

                invalid_count += 1
                wildcard_count += 1

        if invalid_count > 0:
            print(
                f"\n    WARNING {invalid_count} invalid config(s) auto-corrected via wildcard expansion"
            )

        if wildcard_count > 0:
            print(
                f"    OK {len(gemm_declarations) - wildcard_count} explicit + {wildcard_count} wildcard (will expand)"
            )
        else:
            print(f"    OK All {len(gemm_declarations)} configurations valid")

        # Expand GEMM declarations (for wildcards)
        print("\n    Expanding wildcards to valid configurations...")
        expanded_gemm = []
        for decl in gemm_declarations:
            arch = decl.get("arch", args.gpu_target)
            decl_name = (
                decl["name"].split(":")[-1] if ":" in decl["name"] else decl["name"]
            )

            expanded = expand_declaration_with_arch_filter(decl, arch)
            expanded_gemm.extend(expanded)

            # Show what the wildcard expanded to
            if len(expanded) > 1:
                print(
                    f"      {decl_name}: expanded to {len(expanded)} valid configurations"
                )
                # Show first few expanded configs
                for exp in expanded[:3]:
                    wave_str = f"[{exp['wave_m']}, {exp['wave_n']}, {exp['wave_k']}]"
                    warp_str = f"[{exp['warp_m']}, {exp['warp_n']}, {exp['warp_k']}]"
                    print(
                        f"        -> wave={wave_str}, warp={warp_str}, pipeline={exp['pipeline']}, scheduler={exp['scheduler']}"
                    )
                if len(expanded) > 3:
                    print(f"        ... and {len(expanded) - 3} more")
            elif len(expanded) == 1 and is_wildcard_declaration(decl):
                exp = expanded[0]
                wave_str = f"[{exp['wave_m']}, {exp['wave_n']}, {exp['wave_k']}]"
                warp_str = f"[{exp['warp_m']}, {exp['warp_n']}, {exp['warp_k']}]"
                print(f"      {decl_name}: -> wave={wave_str}, warp={warp_str}")

        if len(expanded_gemm) > len(gemm_declarations):
            print(
                f"\n    Total: {len(gemm_declarations)} declarations -> {len(expanded_gemm)} configurations"
            )

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
                needs_expansion = is_conv_wildcard_declaration(decl)
                suffix = " [expands]" if needs_expansion else ""
                display_name = (
                    decl["name"].split(":")[-1] if ":" in decl["name"] else decl["name"]
                )
                print(f"      - {display_name}{suffix}")
            if len(set_decls) > 5:
                print(f"      ... and {len(set_decls) - 5} more")

        # Validate Conv declarations against arch filter
        print(f"\n    Validating against {args.gpu_target} arch filter...")
        wildcard_count = 0
        invalid_count = 0
        auto_corrections = []

        for decl in conv_declarations:
            arch = decl.get("arch", args.gpu_target)
            decl_name = (
                decl["name"].split(":")[-1] if ":" in decl["name"] else decl["name"]
            )

            # Check for wildcards
            if is_conv_wildcard_declaration(decl):
                wildcard_count += 1
                continue  # Wildcards validated during expansion

            is_valid, error_msg = validate_conv_kernel_config(decl, arch)
            if not is_valid:
                print(f"\n    WARNING Invalid conv configuration: {decl_name}")

                # Parse the error and show specific auto-corrections
                corrections = []
                original_values = {}

                if "wave configuration" in error_msg.lower():
                    original_values["wave"] = (
                        f"[{decl.get('wave_m', 2)}, {decl.get('wave_n', 2)}, {decl.get('wave_k', 1)}]"
                    )
                    decl["wave_m"] = -1
                    decl["wave_n"] = -1
                    corrections.append(
                        f"wave: {original_values['wave']} -> [wildcard expansion]"
                    )

                if "warp tile" in error_msg.lower():
                    original_values["warp"] = (
                        f"[{decl.get('warp_m', 32)}, {decl.get('warp_n', 32)}, {decl.get('warp_k', 16)}]"
                    )
                    decl["warp_m"] = -1
                    decl["warp_n"] = -1
                    corrections.append(
                        f"warp_tile: {original_values['warp']} -> [wildcard expansion]"
                    )

                if "trait combination" in error_msg.lower():
                    original_values["pipeline"] = decl.get("pipeline", "compv3")
                    original_values["scheduler"] = decl.get("scheduler", "intrawave")
                    decl["pipeline"] = "*"
                    decl["scheduler"] = "*"
                    corrections.append(
                        f"pipeline: {original_values['pipeline']} -> [wildcard expansion]"
                    )
                    corrections.append(
                        f"scheduler: {original_values['scheduler']} -> [wildcard expansion]"
                    )

                # Print the auto-corrections
                print("      AUTO-CORRECTION:")
                for corr in corrections:
                    print(f"        - {corr}")
                auto_corrections.append((decl_name, corrections))

                invalid_count += 1
                wildcard_count += 1

        if invalid_count > 0:
            print(
                f"\n    WARNING {invalid_count} invalid config(s) auto-corrected via wildcard expansion"
            )

        if wildcard_count > 0:
            print(
                f"    OK {len(conv_declarations) - wildcard_count} explicit + {wildcard_count} wildcard (will expand)"
            )
        else:
            print(f"    OK All {len(conv_declarations)} configurations valid")

        # Expand Conv declarations (for wildcards)
        print("\n    Expanding wildcards to valid configurations...")
        expanded_conv = []
        for decl in conv_declarations:
            arch = decl.get("arch", args.gpu_target)
            decl_name = (
                decl["name"].split(":")[-1] if ":" in decl["name"] else decl["name"]
            )

            expanded = expand_conv_declaration_with_arch_filter(decl, arch)
            expanded_conv.extend(expanded)

            # Show what the wildcard expanded to
            if len(expanded) > 1:
                print(
                    f"      {decl_name}: expanded to {len(expanded)} valid configurations"
                )
                for exp in expanded[:3]:
                    wave_str = f"[{exp['wave_m']}, {exp['wave_n']}, {exp['wave_k']}]"
                    warp_str = f"[{exp['warp_m']}, {exp['warp_n']}, {exp['warp_k']}]"
                    print(
                        f"        -> wave={wave_str}, warp={warp_str}, pipeline={exp['pipeline']}, scheduler={exp['scheduler']}"
                    )
                if len(expanded) > 3:
                    print(f"        ... and {len(expanded) - 3} more")
            elif len(expanded) == 1 and is_conv_wildcard_declaration(decl):
                exp = expanded[0]
                wave_str = f"[{exp['wave_m']}, {exp['wave_n']}, {exp['wave_k']}]"
                warp_str = f"[{exp['warp_m']}, {exp['warp_n']}, {exp['warp_k']}]"
                print(f"      {decl_name}: -> wave={wave_str}, warp={warp_str}")

        if len(expanded_conv) > len(conv_declarations):
            print(
                f"\n    Total: {len(conv_declarations)} declarations -> {len(expanded_conv)} configurations"
            )

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

    # Find GEMM kernel header (try each expanded declaration until one matches)
    if gemm_declarations:
        gemm_header = None
        for decl in gemm_declarations:
            header = find_kernel_header(decl, args.gpu_target)
            if header:
                gemm_header = header
                break

        if gemm_header:
            kernel_headers.append(gemm_header)
            print(f"  GEMM: {gemm_header.name}")
        else:
            print_error("  GEMM: No kernel found matching any declaration!")
            print_error(
                "  The kernels declared in DECL_KERNEL_SET must exist or be generatable."
            )
            return 1

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
