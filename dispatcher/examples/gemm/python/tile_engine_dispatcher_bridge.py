#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Example 12: Tile Engine -> Dispatcher bridge (gallery)

Unlike examples 01-11 (which drive the Dispatcher's native ctypes Registry),
this example exercises the *Tile Engine -> Dispatcher bridge* in
``dispatcher/python/gemm_utils.py``. The bridge is the path the Tile Engine
itself uses: one common ``GemmKernelConfig`` feeds codegen, force-include
compile, and a flat extern "C" ABI, and ``GpuGemmRunner`` runs the resulting
.so against a NumPy reference.

It is a small gallery of three demos that together cover the surface the bridge
gained over its original fp16/rcr-only slice:

  matrix  every (dtype, layout) pair the universal GEMM supports -- fp16 and
          bf16 across the four row/col A/B combinations (row-major C only).
  shapes  why padding matters: a padded kernel accepts an awkward, non-tile-
          aligned problem (M, N not divisible by the tile) while the equivalent
          no-pad kernel rejects it -- the same selection rule the Tile Engine
          sees when it sweeps pad on/off.
  sweep   the "search space" idea: one fixed signature, several *algorithms*
          (tile / wave / pipeline), built and ranked by measured TFLOPS -- a
          miniature of what the Tile Engine driver does at scale.

(Kernel *variants* such as Stream-K and grouped GEMM ride the same bridge but
live on separate branches; this example stays within the regular GEMM stack.)

Usage:
    python3 tile_engine_dispatcher_bridge.py                 # runs all three demos
    python3 tile_engine_dispatcher_bridge.py --demo matrix
    python3 tile_engine_dispatcher_bridge.py --demo shapes
    python3 tile_engine_dispatcher_bridge.py --demo sweep
    python3 tile_engine_dispatcher_bridge.py --size 1024 --rtol 2e-2 --arch gfx950
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))
import numpy as np  # noqa: E402

from gemm_utils import (  # noqa: E402
    GemmKernelConfig,
    GemmProblem,
    GpuGemmRunner,
    setup_multiple_gemm_dispatchers,
)
from ctypes_utils import detect_gpu_arch  # noqa: E402

# A single algorithm known to compile and run on gfx942. Only the Signature
# (dtype + layout) varies in the matrix demo; the Algorithm is held fixed so the
# demo isolates the bridge's dtype/layout generality.
_ALGO = dict(
    tile_m=64, tile_n=64, tile_k=64,
    wave_m=4, wave_n=1, wave_k=1,
    warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
    pipeline="compv3", scheduler="intrawave", epilogue="cshuffle",
    pad_m=False, pad_n=False, pad_k=False,
)

# (dtype, layout) pairs. Column-major C (e.g. rcc) is rejected at build by the
# universal GEMM, so every case keeps row-major C -- which leaves exactly four
# A/B combinations (rcr/rrr/ccr/crr). Both dtypes cover all four.
_CASES = [
    ("fp16", "rcr"), ("fp16", "rrr"), ("fp16", "ccr"), ("fp16", "crr"),
    ("bf16", "rcr"), ("bf16", "rrr"), ("bf16", "ccr"), ("bf16", "crr"),
]

_LAYOUT_WORD = {"r": "row", "c": "col"}


def _emulate(x: np.ndarray, dtype: str) -> np.ndarray:
    """Round fp32 inputs to the kernel's storage dtype so the CPU reference
    matches what the GPU actually multiplies."""
    if dtype == "bf16":
        u32 = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
        rounded = (u32 + ((u32 >> 16) & 1) + np.uint32(0x7FFF)) >> 16
        return (rounded.astype(np.uint32) << 16).view(np.float32)
    return x.astype(np.float16).astype(np.float32)


def _max_rel(out: np.ndarray, ref: np.ndarray) -> float:
    # Global relative error (normalize by the largest reference magnitude):
    # per-element ratios explode on the near-zero entries that K-length
    # accumulation of zero-mean data produces, so they are not meaningful.
    denom = float(np.max(np.abs(ref))) + 1e-12
    return float(np.max(np.abs(out - ref))) / denom


def _config(dtype: str, layout: str, arch: str, **algo) -> GemmKernelConfig:
    la, lb, lc = layout
    return GemmKernelConfig(
        dtype_a=dtype, dtype_b=dtype, dtype_c=dtype,
        layout_a=_LAYOUT_WORD[la], layout_b=_LAYOUT_WORD[lb], layout_c=_LAYOUT_WORD[lc],
        gfx_arch=arch, **(algo or _ALGO),
    )


def _reference(A, B, dtype):
    # Emulate both input quantization (A,B stored as dtype) and the output store
    # (GPU writes C back as dtype_c), so round on both ends before comparing.
    return _emulate(_emulate(A, dtype) @ _emulate(B, dtype), dtype)


# ---------------------------------------------------------------------------
# Demo 1: dtype x layout matrix
# ---------------------------------------------------------------------------
def demo_matrix(size, rtol, arch):
    print(f"\n[matrix] dtype x layout, M=N=K={size}, rtol={rtol:g}")
    problem = GemmProblem(M=size, N=size, K=size)
    configs = [_config(dt, lay, arch) for dt, lay in _CASES]
    so_paths = setup_multiple_gemm_dispatchers(configs, verbose=False)

    rng = np.random.default_rng(42)
    A = (rng.standard_normal((problem.M, problem.K)) * 0.1).astype(np.float32)
    B = (rng.standard_normal((problem.K, problem.N)) * 0.1).astype(np.float32)

    n_pass = 0
    for (dtype, layout), so in zip(_CASES, so_paths):
        tag = f"{dtype}/{layout}"
        if so is None:
            print(f"  {tag:10s} BUILD FAILED")
            continue
        result = GpuGemmRunner(lib_path=so).run(A, B, problem)
        if not result.success:
            print(f"  {tag:10s} RUN FAILED (status {result.status})")
            continue
        mr = _max_rel(result.output, _reference(A, B, dtype))
        ok = mr <= rtol
        n_pass += ok
        print(f"  {tag:10s} tflops={result.tflops:7.1f}  max_rel={mr:.2e}  "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"  -> {n_pass}/{len(_CASES)} passed")
    return n_pass, len(_CASES)


# ---------------------------------------------------------------------------
# Demo 2: padding vs an awkward (non-tile-aligned) shape
# ---------------------------------------------------------------------------
def demo_shapes(rtol, arch):
    print("\n[shapes] padding lets a kernel accept a non-tile-aligned problem")
    # 128-tile kernels; awkward M, N do not divide 128 (K stays divisible by 8
    # for the fp16 vectorized reduction load).
    algo_pad = dict(
        tile_m=128, tile_n=128, tile_k=32, wave_m=2, wave_n=2, wave_k=1,
        warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
        pipeline="compv4", scheduler="intrawave", epilogue="cshuffle",
        pad_m=True, pad_n=True, pad_k=True,
    )
    algo_nopad = dict(algo_pad, pad_m=False, pad_n=False, pad_k=False)

    cfg_pad = _config("fp16", "rcr", arch, **algo_pad)
    cfg_nopad = _config("fp16", "rcr", arch, **algo_nopad)
    so_pad, so_nopad = setup_multiple_gemm_dispatchers([cfg_pad, cfg_nopad], verbose=False)

    M, N, K = 257, 129, 512  # awkward: 257, 129 not divisible by 128
    problem = GemmProblem(M=M, N=N, K=K)
    rng = np.random.default_rng(7)
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
    B = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
    ref = _reference(A, B, "fp16")

    print(f"  awkward problem M={M} N={N} K={K} (M,N not divisible by tile 128)")
    n_pass = 0
    expectations = [("padded", so_pad, True), ("no-pad", so_nopad, False)]
    for label, so, should_pass in expectations:
        if so is None:
            print(f"  {label:8s} BUILD FAILED")
            continue
        result = GpuGemmRunner(lib_path=so).run(A, B, problem)
        if result.success:
            mr = _max_rel(result.output, ref)
            accepted = mr <= rtol
            outcome = f"ACCEPTED  tflops={result.tflops:7.1f}  max_rel={mr:.2e}"
        else:
            accepted = False
            # status -2 == select_kernel found no kernel whose tiling fits.
            outcome = f"REJECTED  (status {result.status})"
        # "Correct" = the no-pad kernel rejects and the padded one accepts.
        correct = accepted == should_pass
        n_pass += correct
        print(f"  {label:8s} {outcome:42s} {'as expected' if correct else 'UNEXPECTED'}")
    print(f"  -> {n_pass}/2 behaved as expected (padded accepts, no-pad rejects)")
    return n_pass, 2


# ---------------------------------------------------------------------------
# Demo 3: algorithm sweep over one fixed signature
# ---------------------------------------------------------------------------
def demo_sweep(size, rtol, arch):
    print(f"\n[sweep] fixed fp16/rcr signature, several algorithms, M=N=K={size}")
    # A handful of distinct algorithms (the Tile Engine sweeps thousands of
    # these). Each is a different tile / wave / pipeline point in the search
    # space; padding is on so any size is accepted.
    algos = [
        dict(tile_m=128, tile_n=128, tile_k=32, wave_m=2, wave_n=2, wave_k=1,
             warp_tile_m=32, warp_tile_n=32, warp_tile_k=16, pipeline="compv4"),
        dict(tile_m=256, tile_n=128, tile_k=32, wave_m=2, wave_n=2, wave_k=1,
             warp_tile_m=32, warp_tile_n=32, warp_tile_k=16, pipeline="compv4"),
        dict(tile_m=128, tile_n=128, tile_k=64, wave_m=2, wave_n=2, wave_k=1,
             warp_tile_m=32, warp_tile_n=32, warp_tile_k=16, pipeline="compv3"),
        dict(tile_m=64, tile_n=64, tile_k=64, wave_m=4, wave_n=1, wave_k=1,
             warp_tile_m=16, warp_tile_n=16, warp_tile_k=16, pipeline="compv3"),
    ]
    common = dict(scheduler="intrawave", epilogue="cshuffle",
                  pad_m=True, pad_n=True, pad_k=True)
    configs = [_config("fp16", "rcr", arch, **dict(a, **common)) for a in algos]
    so_paths = setup_multiple_gemm_dispatchers(configs, verbose=False)

    problem = GemmProblem(M=size, N=size, K=size)
    rng = np.random.default_rng(123)
    A = (rng.standard_normal((size, size)) * 0.1).astype(np.float32)
    B = (rng.standard_normal((size, size)) * 0.1).astype(np.float32)
    ref = _reference(A, B, "fp16")

    rows = []
    for cfg, so in zip(configs, so_paths):
        label = f"{cfg.tile_m}x{cfg.tile_n}x{cfg.tile_k}/{cfg.pipeline}"
        if so is None:
            rows.append((label, None, None))
            continue
        result = GpuGemmRunner(lib_path=so).run(A, B, problem)
        if not result.success:
            rows.append((label, None, None))
            continue
        rows.append((label, result.tflops, _max_rel(result.output, ref)))

    ranked = sorted((r for r in rows if r[1] is not None),
                    key=lambda r: r[1], reverse=True)
    print(f"  {'rank':>4} {'algorithm':<24} {'tflops':>9} {'max_rel':>10}")
    for i, (label, tflops, mr) in enumerate(ranked, 1):
        print(f"  {i:>4} {label:<24} {tflops:>9.1f} {mr:>10.2e}")
    for label, tflops, _ in rows:
        if tflops is None:
            print(f"  {'-':>4} {label:<24} {'BUILD/RUN FAILED':>20}")
    if ranked:
        print(f"  -> fastest: {ranked[0][0]} at {ranked[0][1]:.1f} TFLOPS")
    n_ok = sum(1 for _, _, mr in ranked if mr <= rtol)
    return n_ok, len(configs)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tile Engine -> Dispatcher bridge example (gallery)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--demo", choices=["matrix", "shapes", "sweep", "all"],
                        default="all", help="which demo to run (default: all)")
    parser.add_argument("--size", type=int, default=512, help="M=N=K (default 512)")
    parser.add_argument("--rtol", type=float, default=2e-2,
                        help="relative tolerance (default 2e-2)")
    parser.add_argument("--arch", default=detect_gpu_arch(),
                        help="GPU target arch (default: auto-detected via rocminfo)")
    args = parser.parse_args()

    demos = ["matrix", "shapes", "sweep"] if args.demo == "all" else [args.demo]
    total_pass = 0
    total = 0
    for d in demos:
        if d == "matrix":
            p, t = demo_matrix(args.size, args.rtol, args.arch)
        elif d == "shapes":
            p, t = demo_shapes(args.rtol, args.arch)
        else:
            p, t = demo_sweep(args.size, args.rtol, args.arch)
        total_pass += p
        total += t

    print(f"\n{total_pass}/{total} checks passed across {len(demos)} demo(s)")
    return 0 if total_pass == total else 1


if __name__ == "__main__":
    sys.exit(main())