#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Stream-K deep-core registry test (requires a GPU + hipcc).

Guards the deep-core path that lets Stream-K ride the registry like regular GEMM:
codegen -> generated SK wrapper -> Registry -> Dispatcher::run() (workspace alloc
+ strategy-aware reset) -> generated_tile_backend_streamk -> verify vs reference.

Each reduction strategy (atomic/linear/tree) is a *distinct compiled kernel*
(SkReductionStrategy is a compile-time constexpr), so we generate all three from a
single tile config and build the 04 registry driver once per strategy, force-
including that strategy's header. For each we assert:
  * the encode_identifier() suffix matches the strategy (..._streamk[_linear|_tree])
  * the Dispatcher selects that kernel by Problem::reduction_strategy
  * the result verifies against the reference GEMM

The test SKIPs (exit 77) when no GPU or no hipcc is available, so it is safe in
CPU-only CI; it only runs the heavy build+launch where a GPU is present.

Usage:
    python3 test_streamk_registry.py
    python3 test_streamk_registry.py --arch gfx942 --m 3840 --n 4096 --k 2048
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DISPATCHER_DIR = Path(__file__).resolve().parent.parent
CK_DIR = DISPATCHER_DIR.parent
CODEGEN = DISPATCHER_DIR / "codegen" / "unified_gemm_codegen.py"
DRIVER = DISPATCHER_DIR / "examples" / "gemm" / "cpp" / "04_streamk_registry_driver.cpp"
REGISTRY_SRC = DISPATCHER_DIR / "src" / "registry.cpp"
DISPATCHER_SRC = DISPATCHER_DIR / "src" / "dispatcher.cpp"

SKIP = 77  # ctest SKIP_RETURN_CODE

# One tile config, all three reduction strategies.
TILE = "128x128x64_2x2x1_32x32x16"
TILE_CONFIG_JSON = json.dumps(
    {
        "tile_config": {
            "tile_m": [128], "tile_n": [128], "tile_k": [64],
            "warp_m": [2], "warp_n": [2], "warp_k": [1],
            "warp_tile_m": [32], "warp_tile_n": [32], "warp_tile_k": [16],
            "block_size": [256],
        },
        "trait_config": {
            "pipeline": ["compv3"], "epilogue": ["cshuffle"], "scheduler": ["intrawave"],
            "pad_m": [False], "pad_n": [False], "pad_k": [False], "persistent": [False],
        },
        "streamk_config": {"reduction_strategy": ["atomic", "linear", "tree"]},
    }
)

# strategy -> (header variant suffix, expected encode_identifier suffix)
STRATEGIES = {
    "atomic": ("streamk", "_streamk"),
    "linear": ("streamk_linear", "_streamk_linear"),
    "tree": ("streamk_tree", "_streamk_tree"),
}

# Datatypes the Stream-K dispatcher codegen supports end-to-end. fp8/bf8 inputs
# accumulate in fp32 and write an fp16 C tensor (get_output_dtype), matching
# Tile Engine; the registry identifier keys on the input dtype (dtype_a), so the
# expected encode_identifier prefix is "{dtype}_{layout}" for each.
DATATYPES = ["fp16", "bf16", "fp8", "bf8"]

# Layouts Tile Engine builds Stream-K for (all keep C row-major, which the atomic
# C-reset relies on). Full coverage = DATATYPES x LAYOUTS x STRATEGIES.
LAYOUTS = ["rcr", "rrr", "ccr", "crr"]


def detect_arch(fallback=None):
    # Resolve the gfx target without shelling out to rocminfo. Preference order:
    # the arch the build already configured with (passed via --arch from
    # CMakeLists.txt) is handled by the caller; here we fall back to the standard
    # ROCm environment variables and then the amdgpu-arch / offload-arch LLVM
    # tools, which query the driver directly and ship with the ROCm/LLVM toolchain.
    for env in ("PYTORCH_ROCM_ARCH", "HCC_AMDGPU_TARGET", "AMDGPU_TARGETS", "GPU_TARGETS"):
        val = os.environ.get(env)
        if val:
            return re.split(r"[;,]", val)[0].strip()
    for tool in ("amdgpu-arch", "offload-arch"):
        exe = shutil.which(tool)
        if exe:
            try:
                out = run([exe], timeout=30).stdout
                m = re.search(r"\bgfx[0-9a-f]+\b", out)
                if m:
                    return m.group(0)
            except Exception:
                pass
    return fallback


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=None)
    ap.add_argument("--m", type=int, default=3840)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument(
        "--datatypes", default=",".join(DATATYPES),
        help="Comma-separated datatypes to test (default: all TE-equivalent).",
    )
    ap.add_argument(
        "--layouts", default=",".join(LAYOUTS),
        help="Comma-separated layouts to test (default: all TE-equivalent).",
    )
    args = ap.parse_args()
    datatypes = [d.strip() for d in args.datatypes.split(",") if d.strip()]
    layouts = [l.strip() for l in args.layouts.split(",") if l.strip()]

    hipcc = shutil.which("hipcc")
    if not hipcc:
        print("SKIP: hipcc not found")
        return SKIP

    arch = args.arch or detect_arch()
    if not arch:
        print("SKIP: no GPU / could not detect gfx arch")
        return SKIP
    print(f"Stream-K registry test on {arch} @ {args.m}x{args.n}x{args.k}")

    inc = ["-I", str(CK_DIR / "include"), "-I", str(DISPATCHER_DIR / "include")]

    with tempfile.TemporaryDirectory(prefix="sk_reg_test_") as td:
        # Build the dtype-independent core objects once (no force-include).
        reg_o, disp_o = Path(td) / "registry.o", Path(td) / "dispatcher.o"
        for src, obj in ((REGISTRY_SRC, reg_o), (DISPATCHER_SRC, disp_o)):
            c = run(
                [hipcc, "-std=c++17", f"--offload-arch={arch}", "-O3", *inc,
                 "-c", str(src), "-o", str(obj)],
                timeout=900,
            )
            if c.returncode != 0:
                print(f"FAIL: compiling {src.name}\n" + c.stderr[-2000:])
                return 1

        failures = []
        for dtype in datatypes:
            for layout in layouts:
                failures += run_for_combo(
                    dtype, layout, td, arch, args, hipcc, inc, reg_o, disp_o
                )

        if failures:
            print("\nSTREAM-K REGISTRY TEST FAILED:")
            for f in failures:
                print(" - " + f)
            return 1

    print(
        "All Stream-K combos registered, dispatched, and verified "
        f"(datatypes: {', '.join(datatypes)} | layouts: {', '.join(layouts)})."
    )
    return 0


def run_for_combo(dtype, layout, td, arch, args, hipcc, inc, reg_o, disp_o):
    """Generate + build + run all reduction strategies for one (dtype, layout).

    Returns a list of failure strings (empty on success)."""
    failures = []
    # Verify each built kernel against the CLI shape AND a small-M/N, large-K
    # shape. The latter maximizes the Stream-K split factor, which is exactly
    # where the split-K-aware verification tolerance matters: a plain single-pass
    # tolerance spuriously FAILs correct atomic results on this shape. The driver
    # binary is shape-independent, so this only adds runs, not rebuilds.
    shapes = [(args.m, args.n, args.k), (128, 128, 16384)]
    gen = Path(td) / f"gen_{dtype}_{layout}"

    # 1) generate all three strategy headers from one tile config
    g = run(
        [
            sys.executable, str(CODEGEN),
            "--datatype", dtype, "--layout", layout,
            "--gpu-target", arch, "--variants", "stream_k",
            "--tile-config-json", TILE_CONFIG_JSON,
            "--output-dir", str(gen),
        ],
        timeout=600,
    )
    if g.returncode != 0:
        return [f"{dtype}/{layout}: codegen failed\n" + g.stderr[-2000:]]

    for strat, (variant, want_suffix) in STRATEGIES.items():
        tag = f"{dtype}/{layout}/{strat}"
        header = gen / (
            f"gemm_{dtype}_{layout}_compv3_cshuffle_intrawave_"
            f"False_False_False_False_{TILE}_{variant}.hpp"
        )
        if not header.exists():
            failures.append(f"{tag}: generated header missing ({header.name})")
            continue

        stem = f"{dtype}_{layout}_{variant}"
        drv_o, exe = Path(td) / f"d_{stem}.o", Path(td) / f"skreg_{stem}"
        c = run(
            [hipcc, "-std=c++17", f"--offload-arch={arch}", "-O3",
             "-DCK_TILE_SINGLE_KERNEL_INCLUDE", f'-DGFX_ARCH="{arch}"',
             *inc, "-I", str(gen), "-include", str(header),
             "-c", str(DRIVER), "-o", str(drv_o)],
            timeout=900,
        )
        if c.returncode != 0:
            failures.append(f"{tag}: driver compile failed\n{c.stderr[-1500:]}")
            continue
        l = run(
            [hipcc, f"--offload-arch={arch}", str(drv_o), str(disp_o),
             str(reg_o), "-o", str(exe)],
            timeout=300,
        )
        if l.returncode != 0:
            failures.append(f"{tag}: link failed\n{l.stderr[-1500:]}")
            continue

        for (sm, sn, sk) in shapes:
            r = run(
                [str(exe), "--m", str(sm), "--n", str(sn),
                 "--k", str(sk), "--strategy", strat, "--validate", "1"],
                timeout=300,
            )
            out = r.stdout
            ok_verify = "Verification: PASS" in out
            # Guard the identifier parse: a crashed/silent driver prints no
            # "identifier=" token, so split(...)[1] would raise IndexError and
            # abort the run instead of recording a clean failure.
            ok_suffix = False
            if f"identifier={dtype}_{layout}" in out and "identifier=" in out:
                token = out.split("identifier=", 1)[1].split()[0]
                ok_suffix = want_suffix in token
            if r.returncode != 0 or not ok_verify or not ok_suffix:
                failures.append(
                    f"{tag} @ {sm}x{sn}x{sk}: rc={r.returncode} verify={ok_verify} "
                    f"suffix_ok={ok_suffix}\n{out[-800:]}{r.stderr[-400:]}"
                )
            else:
                tflops = next(
                    (ln for ln in out.splitlines() if "TFlops" in ln), ""
                ).strip()
                print(
                    f"  PASS {dtype:5s} {layout:4s} {strat:6s} {sm}x{sn}x{sk} "
                    f"-> {want_suffix}  | {tflops}"
                )

    return failures


if __name__ == "__main__":
    sys.exit(main())
