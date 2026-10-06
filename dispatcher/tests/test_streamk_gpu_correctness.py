#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
GPU correctness tests for the Stream-K GEMM dispatcher — all three reduction
strategies.

The shared sweep (``test_gemm_search_space.py --variant stream_k``) only ever
builds the ``atomic`` strategy: ``default_ci_config.json`` carries no
``streamk_config`` block, so ``expand_sweep`` falls back to ``["atomic"]``.
``linear`` and ``tree`` — which reduce through a device workspace instead of
global atomics — have never been verified through the ctypes bridge. This test
closes that gap.

Requires a gfx942 GPU (MI300X) and hipcc in PATH. Skips cleanly (exit 77) when
no GPU is visible or the detected arch is not gfx942, so it is safe to invoke
unconditionally from a CI lane.

Shapes:
  S1  M=4096 N=4096 K=2048  — all three strategies. Output tiles vastly exceed
      the occupancy limit, so the partitioner leaves few (if any) Stream-K
      tiles and ``num_wgs_per_tile`` stays <= 2 under atomic.
  S2  M=128  N=128  K=8192   — linear/tree ONLY. This is the Stream-K-heavy
      regime: one output tile split across many workgroups.
      ``estimate_num_wgs_per_tile`` is gated on the Atomic strategy
      (streamk_gemm_tile_partitioner_impl.hpp:279) and returns 1 otherwise, so
      linear/tree keep a tight tolerance here where atomic would need ~5e-1
      rtol for bf16 — a vacuous assertion. The atomic side of this regime is
      already covered in C++ by test_streamk_registry.py.

Tolerance is CK's own split-K-aware bound, ported from
``dispatcher/examples/gemm/cpp/streamk_driver_common.hpp`` +
``include/ck_tile/host/check_err.hpp`` — asserted element-wise, not as a
scalar max-relative-error.

Run:
  python3 test_streamk_gpu_correctness.py
  python3 test_streamk_gpu_correctness.py -v                    # verbose hipcc
  python3 test_streamk_gpu_correctness.py --gfx gfx942
  python3 test_streamk_gpu_correctness.py --dtypes fp16,bf16    # trim matrix
"""

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from ctypes_utils import detect_gpu_arch
from gemm_utils import (
    GemmKernelConfig,
    GemmProblem,
    GpuGemmRunner,
    numpy_dtype_for,
    output_dtype_for,
    setup_multiple_gemm_dispatchers,
)

log = logging.getLogger("streamk_gpu")

PASS, FAIL = "PASS", "FAIL"
SKIP_EXIT = 77

# gfx90a stays out: it has no fp8/bf8 MFMA, so half the matrix below is
# unrunnable there. gfx950 used to be excluded for a different reason -- the
# gemm_utils host codec was FNUZ-only while the arch encodes fp8 as OCP -- which
# had nothing to do with Stream-K. That codec now follows the arch (see
# dispatcher_common.fp8_uses_ocp), so gfx950 is back in.
SUPPORTED_ARCHS = ("gfx942", "gfx950")

STRATEGIES = ("atomic", "linear", "tree")
ALL_DTYPES = ("fp16", "bf16", "fp8", "bf8")

# pad_m/n/k are all False, so every dimension must be a multiple of the tile.
TILE_M = TILE_N = TILE_K = 64

S1 = (4096, 4096, 2048)  # dp-dominant: all strategies
S2 = (128, 128, 8192)    # sk-dominant: linear/tree only

# Upper bound on the workgroups reducing into a single output tile. Not knowable
# from Python (it needs hipOccupancyMaxActiveBlocksPerMultiprocessor), which is
# exactly why the shapes above pin it to a provable bound.
S1_ATOMIC_WGS_PER_TILE = 2
NON_ATOMIC_WGS_PER_TILE = 1


# ---------------------------------------------------------------------------
# Tolerance: a Python port of CK's split-K-aware bound
# ---------------------------------------------------------------------------

# Explicit-mantissa bit counts, matching ck_tile::numeric<T>::mant():
#   fp32 numeric.hpp:92 | fp16 half.hpp:233 | bf16 bfloat16.hpp:453
#   fp8/e4m3 float8.hpp:220 | bf8/e5m2 float8.hpp:238
_MANT_BITS = {"fp32": 23, "fp16": 10, "bf16": 7, "fp8": 3, "bf8": 2}


def _relative_threshold(compute: str, out: str, acc: str, n_accum: int) -> float:
    """Port of ck_tile::get_relative_threshold<Compute, Out, Acc>(n_accum)."""
    compute_error = 2.0 ** -_MANT_BITS[compute] * 0.5
    output_error = 2.0 ** -_MANT_BITS[out] * 0.5
    midway_error = max(compute_error, output_error)
    acc_error = 2.0 ** -_MANT_BITS[acc] * 0.5 * n_accum
    return max(acc_error, midway_error)


def _absolute_threshold(
    compute: str, out: str, acc: str, maxv: float, n_accum: int
) -> float:
    """Port of ck_tile::get_absolute_threshold<Compute, Out, Acc>(maxv, n_accum)."""
    if maxv == 0.0:
        return 0.0
    expo = math.floor(math.log2(abs(maxv)))
    compute_error = 2.0 ** (expo - _MANT_BITS[compute]) * 0.5
    # Full ULP, not half: the output error also absorbs the hardware-vs-software
    # conversion difference. This asymmetry is deliberate in check_err.hpp.
    output_error = 2.0 ** (expo - _MANT_BITS[out]) * 1.0
    midway_error = max(compute_error, output_error)
    acc_error = 2.0 ** (expo - _MANT_BITS[acc]) * 0.5 * n_accum
    return max(acc_error, midway_error)


def _streamk_tolerance(dtype: str, out_dtype: str, K: int, num_wgs_per_tile: int,
                       maxv: float):
    """Port of streamk_tolerance() from streamk_driver_common.hpp:85-101.

    Stream-K accumulates a tile's K-range across ``num_wgs_per_tile`` partials,
    so the bound is the looser of two regimes: the per-split MAC error (over
    ``K / num_wgs`` accumulations at accumulator precision) and the cross-split
    reduction error (over ``num_wgs`` accumulations at OUTPUT precision, which is
    what atomics and the workspace both use).
    """
    k_per_split = -(-K // num_wgs_per_tile)  # integer ceil
    rtol_base = _relative_threshold(dtype, out_dtype, "fp32", k_per_split)
    atol_base = _absolute_threshold(
        dtype, out_dtype, "fp32", maxv / num_wgs_per_tile, k_per_split
    )
    rtol_split_k = _relative_threshold(out_dtype, out_dtype, out_dtype,
                                       num_wgs_per_tile)
    atol_split_k = _absolute_threshold(out_dtype, out_dtype, out_dtype, maxv,
                                       num_wgs_per_tile)
    return max(rtol_base, rtol_split_k), max(atol_base, atol_split_k)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quantize(x: np.ndarray, dtype: str) -> np.ndarray:
    """Round through ``dtype`` and back to fp32.

    The kernel consumes rounded operands, so the reference must too — otherwise
    the comparison measures the quantization step, not the kernel.
    """
    return x.astype(numpy_dtype_for(dtype)).astype(np.float32)


def _max_rel(got: np.ndarray, ref: np.ndarray) -> float:
    """Global relative error, for logging only. The assertion is element-wise."""
    denom = float(np.max(np.abs(ref)))
    if denom == 0.0:
        return float(np.max(np.abs(got)))
    return float(np.max(np.abs(got.astype(np.float32) - ref))) / denom


def _detect_arch():
    try:
        arch = detect_gpu_arch(fallback="")
    except Exception:
        return None
    return arch or None


def _make_config(dtype: str, strategy: str, gfx_arch: str) -> GemmKernelConfig:
    """One Stream-K kernel config, spelled out rather than defaulted.

    Mirrors what the CI sweep builds green today for variant=stream_k on gfx942.
    ``epilogue`` must stay "cshuffle": the codegen emits nothing for any other
    value, which surfaces later as a confusing "no .hpp generated" build error.
    """
    return GemmKernelConfig(
        dtype_a=dtype,
        dtype_b=dtype,
        dtype_c=output_dtype_for(dtype),
        dtype_acc="fp32",
        layout_a="row",
        layout_b="col",
        layout_c="row",
        tile_m=TILE_M,
        tile_n=TILE_N,
        tile_k=TILE_K,
        wave_m=2,
        wave_n=2,
        wave_k=1,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=16,
        pipeline="compv3",
        scheduler="intrawave",
        epilogue="cshuffle",
        pad_m=False,
        pad_n=False,
        pad_k=False,
        persistent=False,
        variant="stream_k",
        reduction_strategy=strategy,
        gfx_arch=gfx_arch,
    )


class _Shape:
    """One shape: fp32 operands plus a per-dtype quantized reference.

    A and B are generated once and shared by every dtype/strategy; the fp32
    matmul is memoized per dtype so the 4096-cubed reference is computed four
    times, not twelve.
    """

    def __init__(self, label: str, M: int, N: int, K: int, seed: int):
        self.label = label
        self.M, self.N, self.K = M, N, K
        rng = np.random.default_rng(seed)
        self.A = (rng.standard_normal((M, K)) * 0.1).astype(np.float32)
        self.B = (rng.standard_normal((K, N)) * 0.1).astype(np.float32)
        self.problem = GemmProblem(M=M, N=N, K=K)
        self._refs = {}

    def reference(self, dtype: str) -> np.ndarray:
        ref = self._refs.get(dtype)
        if ref is None:
            acc = _quantize(self.A, dtype) @ _quantize(self.B, dtype)
            ref = _quantize(acc, output_dtype_for(dtype))
            self._refs[dtype] = ref
        return ref


def _verify(label, so, shape, dtype, strategy):
    """Build a runner for ``so``, launch once, and check the output."""
    runner = GpuGemmRunner(lib_path=so)
    result = runner.run(shape.A, shape.B, shape.problem)
    if not result.success:
        return FAIL, f"{label}: launch failed (status={result.status})"

    got = np.asarray(result.output, dtype=np.float32)
    if not np.all(np.isfinite(got)):
        return FAIL, f"{label}: output contains NaN/Inf"
    # An all-zero C is the signature of a Stream-K kernel that never landed its
    # partials — the most likely failure mode for an unexercised strategy.
    if not np.any(got):
        return FAIL, f"{label}: output is all zeros"

    ref = shape.reference(dtype)
    maxv = float(np.max(np.abs(ref)))
    num_wgs = (
        S1_ATOMIC_WGS_PER_TILE if strategy == "atomic" else NON_ATOMIC_WGS_PER_TILE
    )
    rtol, atol = _streamk_tolerance(
        dtype, output_dtype_for(dtype), shape.K, num_wgs, maxv
    )

    bad = np.abs(got - ref) > atol + rtol * np.abs(ref)
    rel = _max_rel(got, ref)
    if bad.any():
        n_bad = int(bad.sum())
        return FAIL, (
            f"{label}: {n_bad}/{got.size} elements outside "
            f"rtol={rtol:.3e} atol={atol:.3e} (max_rel={rel:.3e})"
        )
    return PASS, (
        f"{label}: OK  max_rel={rel:.3e}  rtol={rtol:.3e} atol={atol:.3e}  "
        f"{result.time_ms:.3f} ms"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_matrix(dtypes):
    """(label, dtype, strategy, shape_key) for every case we intend to run.

    S2 is deliberately linear/tree-only — see the module docstring.
    """
    cases = []
    for dtype in dtypes:
        for strategy in STRATEGIES:
            cases.append((f"S1/{dtype}/{strategy}", dtype, strategy, "S1"))
    for dtype in dtypes:
        for strategy in STRATEGIES:
            if strategy == "atomic":
                continue
            cases.append((f"S2/{dtype}/{strategy}", dtype, strategy, "S2"))
    return cases


def main():
    parser = argparse.ArgumentParser(
        description="Stream-K GPU correctness tests (atomic / linear / tree)"
    )
    # No hardcoded default: it must stay possible to tell "user asked for gfx942"
    # from "we are on an unrelated box", so the skip below can fire.
    parser.add_argument("--gfx", default=None,
                        help="GPU arch override (default: auto-detect; gfx942 only)")
    parser.add_argument("--dtypes", default=",".join(ALL_DTYPES),
                        help=f"comma-separated subset of {','.join(ALL_DTYPES)}")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    gfx = args.gfx or _detect_arch()
    if not gfx:
        print("SKIP: no supported GPU detected (rocminfo); Stream-K GPU tests skipped")
        return SKIP_EXIT
    if gfx not in SUPPORTED_ARCHS:
        print(f"SKIP: Stream-K GPU tests are {'/'.join(SUPPORTED_ARCHS)}-only; "
              f"detected {gfx}")
        return SKIP_EXIT

    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    unknown = [d for d in dtypes if d not in ALL_DTYPES]
    if unknown:
        print(f"ERROR: unknown dtype(s) {unknown}; choose from {list(ALL_DTYPES)}")
        return 2

    # bf16/fp8/bf8 host codecs come from ml_dtypes. fp16 needs nothing, so drop
    # the others rather than skipping the whole suite.
    try:
        import ml_dtypes  # noqa: F401
    except ImportError:
        dropped = [d for d in dtypes if d != "fp16"]
        if dropped:
            log.warning("ml_dtypes not installed; dropping %s", ",".join(dropped))
        dtypes = [d for d in dtypes if d == "fp16"]
    if not dtypes:
        print("SKIP: no runnable dtypes (install ml_dtypes for bf16/fp8/bf8)")
        return SKIP_EXIT

    # The default 50 warmup / 100 repeat iterations are pure waste for a
    # correctness run; each launch here re-zeros and recomputes the full C.
    os.environ.setdefault("CK_TILE_BENCH_WARMUP", "1")
    os.environ.setdefault("CK_TILE_BENCH_REPEAT", "2")

    cases = _build_matrix(dtypes)
    # One kernel per (dtype, strategy); S1 and S2 share it since M/N/K are
    # runtime arguments, not codegen constants.
    kernel_keys = sorted({(d, s) for _, d, s, _ in cases})
    configs = [_make_config(d, s, gfx) for d, s in kernel_keys]

    log.info("Building %d Stream-K kernels for %d cases ...",
             len(configs), len(cases))
    sos = setup_multiple_gemm_dispatchers(configs, verbose=args.verbose)
    by_key = dict(zip(kernel_keys, sos))

    shapes = {
        "S1": _Shape("S1", *S1, seed=42),
        "S2": _Shape("S2", *S2, seed=43),
    }

    results = []
    for label, dtype, strategy, shape_key in cases:
        log.info("--- Running %s ---", label)
        so = by_key.get((dtype, strategy))
        if so is None:
            results.append((label, FAIL, f"{label}: codegen/compile failed"))
            log.info("[%s] %s", FAIL, results[-1][2])
            continue
        try:
            status, detail = _verify(
                label, so, shapes[shape_key], dtype, strategy
            )
        except Exception as exc:
            status, detail = FAIL, f"{label}: exception: {exc}"
        results.append((label, status, detail))
        log.info("[%s] %s", status, detail)

    print("\n=== Summary ===")
    passed = sum(1 for _, s, _ in results if s == PASS)
    for name, status, detail in results:
        print(f"  [{status:4s}] {detail}")
    print(f"\n{passed}/{len(results)} passed")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
