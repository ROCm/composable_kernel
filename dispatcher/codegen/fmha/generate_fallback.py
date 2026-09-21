#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Generate FMHA fallback kernel + dispatch header for the Python ctypes library.

Mirrors generate_conv_dispatch_header.py: generates a single FMHA forward
kernel and creates a dispatch header that can be force-included into
fmha_ctypes_lib.cpp.

Usage:
    python3 generate_fmha_fallback.py --output-dir /path/to/output --gpu-target gfx950
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


# Default kernel config for fallback — a single fwd fp16 kernel with
# known-good tile (128x128x32, qr_async) for basic smoke-test capability.
# Source: tile dims from fmha_fwd.py FmhaFwdTileSize for hdim=128 fp16.
DEFAULT_CONFIG = {
    "arch": "gfx950",
    "signature": {
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
    },
    "algorithm": {
        "pipeline": "qr_async",
        "tile": [128, 128, 32, 128, 32, 128],
        "wave": [4, 1, 1, 4, 1, 1, 1, 1, 1],
        "warp": [32, 32, 16, 32, 32, 16, 16, 16, 16],
        "padding": [True, True, True, True],
        "block_per_cu": 1,
        "num_wave_groups": 1,
        "max_splits_log2": 0,
        "max_seq_len_q": 0,
    },
}


def generate_dispatch_header(output_dir: Path, wrapper_dir: Path) -> Path:
    """Generate fmha_python_dispatch.hpp from the wrapper headers."""
    wrappers = sorted(wrapper_dir.glob("dispatcher_wrapper_fmha_*.hpp"))
    if not wrappers:
        raise RuntimeError(f"No FMHA dispatcher wrappers found in {wrapper_dir}")

    kernel_names = []
    make_calls = []
    for w in wrappers:
        stem = w.stem.replace("dispatcher_wrapper_", "")
        kernel_names.append(stem)
        make_calls.append(
            f"    registry.register_kernel("
            f"ck_tile::dispatcher::generated::make_{stem}(arch));"
        )

    lines = [
        "// Auto-generated FMHA dispatch header for Python ctypes library",
        "#pragma once",
        "",
    ]
    for w in wrappers:
        lines.append(f'#include "dispatcher_wrappers/{w.name}"')

    lines += [
        "",
        '#include "ck_tile/dispatcher/fmha_registry.hpp"',
        '#include "ck_tile/dispatcher/fmha_dispatcher.hpp"',
        "",
        "namespace generated {",
        "",
        "inline void register_fmha_python_kernels("
        "ck_tile::dispatcher::FmhaRegistry& registry, const std::string& arch) {",
        "    (void)arch;",
    ]
    lines += make_calls
    lines += [
        "}",
        "",
        "} // namespace generated",
        "",
        "#ifndef REGISTER_GENERATED_KERNELS",
        "#define REGISTER_GENERATED_KERNELS(registry, arch) "
        "::generated::register_fmha_python_kernels(registry, arch)",
        "#endif",
        "",
        "// Stable C ABI for dlopen/dlsym-based kernel registration.",
        '// Plugins call dlsym(handle, "ck_fmha_register_kernels") to get this.',
        'extern "C" __attribute__((visibility("default")))',
        "int ck_fmha_register_kernels(",
        "    ck_tile::dispatcher::FmhaRegistry& registry, const char* arch) {",
        "    ::generated::register_fmha_python_kernels(registry, arch ? std::string(arch) : std::string());",
        f"    return {len(kernel_names)};",
        "}",
        "",
        "// Kernel inventory for Python introspection",
        f"static const int FMHA_KERNEL_COUNT = {len(kernel_names)};",
        "static const char* FMHA_KERNEL_NAMES[] = {"
        + ", ".join(f'"{n}"' for n in kernel_names)
        + "};",
        "",
    ]

    header_path = output_dir / "fmha_python_dispatch.hpp"
    header_path.write_text("\n".join(lines) + "\n")
    return header_path


def compile_kernels(output_dir: Path, gpu_target: str, include_dirs: str) -> Path:
    """Compile kernel .cpp files into a static library."""
    import shutil

    hipcc = shutil.which("hipcc") or "/opt/rocm/bin/hipcc"
    kernel_cpps = sorted(output_dir.glob("fmha_*.cpp"))
    if not kernel_cpps:
        raise RuntimeError("No kernel .cpp files to compile")

    import re

    # Use the shared compile flags from fmha_utils
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    from fmha_utils import fmha_compile_flags  # noqa: E402

    base_flags = fmha_compile_flags(gpu_target, hipcc, family="bwd")

    inc_flags = []
    for d in re.split(r"[;:]", include_dirs):
        d = d.strip()
        if d:
            inc_flags.extend(["-I", d])

    objs = []
    for cpp in kernel_cpps:
        obj = cpp.with_suffix(".o")
        cmd = base_flags + inc_flags + [str(cpp), "-o", str(obj)]
        print(f"  Compiling {cpp.name}...")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"    FAILED: {r.stderr}", file=sys.stderr)
            raise RuntimeError(f"Failed to compile {cpp.name}")
        objs.append(str(obj))

    lib_path = output_dir / "libfmha_python_fallback.a"
    ar_cmd = ["ar", "rcs", str(lib_path)] + objs
    subprocess.check_call(ar_cmd)
    print(f"  Static lib: {lib_path}")
    return lib_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate FMHA fallback kernel for Python library"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu-target", default="gfx950")
    parser.add_argument(
        "--config-json",
        default=None,
        help="Override default kernel config (JSON string)",
    )
    parser.add_argument(
        "--compile", action="store_true", help="Also compile the kernel .cpp into a .a"
    )
    parser.add_argument(
        "--include-dirs",
        default="",
        help="Semicolon-separated include directories for compilation",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    codegen_dir = Path(__file__).parent
    codegen_script = codegen_dir / "codegen.py"

    # Accept either a single config dict or a list of configs
    if args.config_json:
        parsed = json.loads(args.config_json)
        if isinstance(parsed, list):
            # Multi-config: pass list directly to unified_fmha_codegen
            codegen_input = parsed
        else:
            # Single config: merge with defaults
            config = dict(DEFAULT_CONFIG)
            config["arch"] = args.gpu_target
            config["signature"] = dict(DEFAULT_CONFIG["signature"])
            config["algorithm"] = dict(DEFAULT_CONFIG["algorithm"])
            config.update(parsed)
            codegen_input = config
    else:
        config = dict(DEFAULT_CONFIG)
        config["arch"] = args.gpu_target
        config["signature"] = dict(DEFAULT_CONFIG["signature"])
        config["algorithm"] = dict(DEFAULT_CONFIG["algorithm"])
        codegen_input = config

    print(f"Generating FMHA fallback kernel for {args.gpu_target}...")
    print(f"  Output: {output_dir}")

    cmd = [
        sys.executable,
        str(codegen_script),
        "--output-dir",
        str(output_dir),
        "--gpu-target",
        args.gpu_target,
        "--config-json",
        json.dumps(codegen_input),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(codegen_dir))
    if result.returncode != 0:
        print(f"  Codegen FAILED:\n{result.stderr}", file=sys.stderr)
        return 1

    wrapper_dir = output_dir / "dispatcher_wrappers"
    if not wrapper_dir.exists():
        print("  ERROR: No dispatcher_wrappers dir created", file=sys.stderr)
        return 1

    header_path = generate_dispatch_header(output_dir, wrapper_dir)
    print(f"  Dispatch header: {header_path}")

    kernel_cpps = list(output_dir.glob("fmha_*.cpp"))
    print(f"  Kernel TUs: {len(kernel_cpps)}")

    if args.compile and kernel_cpps:
        compile_kernels(output_dir, args.gpu_target, args.include_dirs)

    return 0


if __name__ == "__main__":
    sys.exit(main())
