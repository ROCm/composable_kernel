#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Dispatcher GEMM search space runner (all bridged GEMM variants).

Enumerates the GemmKernelConfig search space by calling expand_sweep with the
variant's tile_engine default_ci_config.json (multi_d and multi_abd have their
own, carrying the fused-epilogue block) across all (dtype, layout) strata,
samples a budget-limited subset with a daily rotating seed, compiles each config
via the bridge, runs on GPU, and reports correctness + TFLOPS.

One runner covers every variant expand_sweep understands -- standard, grouped,
multi_d, multi_abd and stream_k -- because they share the same enumerate/sample/
build/report machinery and differ only in which Gpu*Runner to drive and how the
numpy reference is computed. Select with --variant.

This is the dispatcher equivalent of tile_engine's per-op benchmark scripts
driven with TILE_ENGINE_SAMPLING_TIER=daily, with one deliberate difference:
those scripts never verify numerics and always exit 0. This one validates every
kernel against a numpy reference and exits 1 on any mismatch or build failure.

Exit codes: 0 = all pass, 1 = failure, 77 = skipped (no GPU) per the ctest
SKIP_RETURN_CODE convention.

Usage:
    python3 test_gemm_search_space.py --arch gfx942 --budget 500
    python3 test_gemm_search_space.py --variant grouped --groups 4
    python3 test_gemm_search_space.py --variant multi_abd   # fp16/rcrr only
    python3 test_gemm_search_space.py --variant multi_d     # sweeps MultiDAdd/Multiply
    python3 test_gemm_search_space.py --variant multi_d --elementwise-op PassThrough
    python3 test_gemm_search_space.py --budget 0   # full space, no cap
    python3 test_gemm_search_space.py --budget 500 --seed 42 --json results.json
"""

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import replace
from datetime import date
from pathlib import Path

_DISPATCHER = Path(__file__).parent.parent
sys.path.insert(0, str(_DISPATCHER / "python"))

import numpy as np

from ctypes_utils import detect_gpu_arch
from gemm_utils import (
    GemmProblem,
    GpuGemmRunner,
    GpuGroupedGemmRunner,
    GpuMultiABDRunner,
    GpuMultiDGemmRunner,
    GroupedGemmProblem,
    MultiDGemmProblem,
    _cde_reference,
    expand_sweep,
    numpy_dtype_for,
    setup_multiple_gemm_dispatchers,
)

_TE_GEMM = _DISPATCHER.parent / "tile_engine/ops/gemm"
_CI_CONFIG = _TE_GEMM / "configs/default_ci_config.json"

# multi_d and multi_abd have their own CI configs carrying the
# ``multi_d_config`` / ``multi_abd_config`` blocks that describe the fused
# epilogue (which element-wise op, how many D tensors). The generic config has
# neither, so pointing every variant at it left expand_sweep with nothing to
# enumerate and the fusion untested. gemm_multi_d_full_benchmark.py already
# reads these same files.
_VARIANT_CI_CONFIG = {
    "multi_d": _TE_GEMM / "gemm_multi_d/configs/default_ci_config.json",
    "multi_abd": _TE_GEMM / "gemm_multi_abd/configs/default_ci_config.json",
}


def _ci_config_for(variant: str) -> Path:
    return _VARIANT_CI_CONFIG.get(variant, _CI_CONFIG)
_DTYPES = ["fp16", "bf16", "fp8", "bf8"]
_LAYOUTS = ["rcr", "rrr", "crr", "ccr"]
_RTOL = 2e-2
_SKIP = 77

# The arches whose kernels this sweep can actually build and run, matching
# SUPPORTED_ARCHS in the aquant/abquant GPU tests. The no-GPU gate below is not
# sufficient on its own: a box with an *unsupported* GPU (gfx90a, say) passes it,
# then fails every sampled config at the hipcc step and reports FAIL. That is the
# same false signal the exit-77 convention exists to remove, pointing the other
# way -- so gate on the resolved arch, forced or detected, exactly as bquant does.
_SUPPORTED_ARCHS = ("gfx942", "gfx950")

# Per-variant defaults for --dtypes / --layouts. multi_abd and multi_d are
# fp16-only end-to-end (codegen, ctypes lib and runner all assume fp16) and take
# the 4-char (A,B,C/E,D) layout code rather than the 3-char one. Only the A and
# B chars vary -- the TE epilogue writes C/E and reads D row-major.
#
# Widening multi_d from one layout to four costs no extra runtime: _sample()
# allocates floor(budget / n_strata) configs per stratum, so a fixed --budget is
# redistributed rather than multiplied. multi_abd stays single-layout until its
# bridge is verified the same way multi_d now is.
_MULTI_LAYOUTS = ["rcrr", "rrrr", "ccrr", "crrr"]
_VARIANT_DEFAULTS = {
    "standard": (_DTYPES, _LAYOUTS),
    "grouped": (_DTYPES, _LAYOUTS),
    "stream_k": (_DTYPES, _LAYOUTS),
    "multi_d": (["fp16"], _MULTI_LAYOUTS),
    "multi_abd": (["fp16"], ["rcrr"]),
}
_VARIANTS = tuple(_VARIANT_DEFAULTS)


def _daily_seed() -> int:
    return int(hashlib.md5(date.today().isoformat().encode()).hexdigest(), 16) % (2**31)


def _sample(strata: dict, budget: int, seed: int) -> list:
    """Stratified budget-limited sample: equal share per non-empty stratum.

    Each stratum gets floor(budget/n), with ``budget % n`` strata taking one
    extra; a stratum smaller than its share releases the difference to a final
    top-up draw, so the full budget is used whenever the space allows.

    The strata are enumerated in sorted order and then *shuffled* before the
    remainder is handed out, rather than being visited in sorted order. Sorted
    order alone made the remainder deterministic in a way the seed could not
    move: when ``budget < n`` every stratum's share is 0, the whole budget is
    remainder, and it all lands on the lexicographically-first strata. At the
    real 16 strata (bf16|bf8|fp16|fp8 x ccr|crr|rcr|rrr), --budget 4 therefore
    picked bf16/{ccr,crr,rcr,rrr} for every seed, so the ctest registration was
    permanently bf16-only and the daily seed bought nothing. Seeding the shuffle
    from the same rng keeps the result reproducible from the seed alone.
    """
    nonempty = {k: v for k, v in strata.items() if v}
    if not nonempty or budget <= 0:
        return []
    rng = random.Random(seed)
    per, remainder = divmod(budget, len(nonempty))
    strata_order = sorted(nonempty.items())
    rng.shuffle(strata_order)
    selected, leftover = [], []
    for i, (_, configs) in enumerate(strata_order):
        # Clamp to the stratum size: budget < n makes `per` 0, and a stratum
        # smaller than its share would otherwise ask random.sample for more
        # items than exist.
        alloc = min(per + (1 if i < remainder else 0), len(configs))
        # Sample indices, not objects: it makes "which ones were not picked"
        # exact without relying on object identity or equality.
        picked = set(rng.sample(range(len(configs)), alloc)) if alloc > 0 else set()
        selected.extend(configs[j] for j in picked)
        leftover.extend(configs[j] for j in range(len(configs)) if j not in picked)
    if len(selected) < budget and leftover:
        selected.extend(rng.sample(leftover, min(budget - len(selected), len(leftover))))
    rng.shuffle(selected)
    return selected[:budget]


def _max_rel(out: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(out - ref))) / (float(np.max(np.abs(ref))) + 1e-12)


def _quantize(x: np.ndarray, dtype: str) -> np.ndarray:
    """Round-trip fp32 through the kernel's host operand dtype.

    The grouped/multi_d runners cast operands with ``numpy_dtype_for`` (fp16 or
    an ml_dtypes bf16/fp8-FNUZ/bf8-FNUZ type), so the reference must see the same
    quantized values the kernel does.
    """
    return x.astype(numpy_dtype_for(dtype)).astype(np.float32)


def _layout_code(cfg, n: int) -> str:
    """``rcr``/``rcrr`` style code from a config's row/col layout words."""
    words = [cfg.layout_a, cfg.layout_b, cfg.layout_c, getattr(cfg, "layout_d", "row")]
    return "".join(w[0] for w in words[:n])


class _Operands:
    """Shared host operands for one sweep, generated once and reused per kernel."""

    def __init__(self, size: int, seed: int, groups: int, num_d: int):
        rng = np.random.default_rng(seed)
        self.size = size
        self.seed = seed
        self.A = (rng.standard_normal((size, size)) * 0.1).astype(np.float32)
        self.B = (rng.standard_normal((size, size)) * 0.1).astype(np.float32)
        self.problem = GemmProblem(M=size, N=size, K=size)

        # Grouped: deliberately NON-uniform M so a kernel that mishandles the
        # per-group offsets fails instead of silently passing. All Ms stay
        # multiples of 64 (the CI config's only tile_m) because pad_m is false.
        step = max(64, size // 4)
        ms = [max(64, size - i * step) for i in range(groups)]
        self.groups = [(m, size, size) for m in ms]
        self.grouped_problem = GroupedGemmProblem(groups=self.groups)
        self.A_list = [self.A[:m] for m in ms]
        self.B_list = [self.B for _ in ms]

        # Multi-D: D tensors are MxN, stored fp16 by the runner.
        self.Ds = [
            (rng.standard_normal((size, size)) * 0.1).astype(np.float16)
            for _ in range(max(num_d, 1))
        ]


def _make_runner(variant: str, cfg, so: Path):
    if variant == "grouped":
        return GpuGroupedGemmRunner(
            lib_path=so, dtype=cfg.dtype_a, layout=_layout_code(cfg, 3)
        )
    if variant == "multi_d":
        return GpuMultiDGemmRunner(lib_path=so)
    if variant == "multi_abd":
        return GpuMultiABDRunner(
            lib_path=so,
            layout4=_layout_code(cfg, 4),
            a_elementwise_op=cfg.a_elementwise_op,
            b_elementwise_op=cfg.b_elementwise_op,
            cde_elementwise_op=cfg.cde_elementwise_op,
        )
    # standard and stream_k share the single-problem C ABI.
    return GpuGemmRunner(lib_path=so)


def _invoke(variant: str, runner, ops: _Operands):
    """One timed launch. Returns an object with .success/.time_ms."""
    if variant == "grouped":
        return runner.run(ops.A_list, ops.B_list, ops.grouped_problem)
    if variant == "multi_d":
        nd = runner.num_d_tensors
        return runner.run(
            ops.A, ops.B, ops.Ds[:nd],
            MultiDGemmProblem(M=ops.size, N=ops.size, K=ops.size, num_d=nd),
        )
    if variant == "multi_abd":
        # Verification is a separate final call (see _verify): keeping it off the
        # timed path avoids recomputing a full fp32 reference on every repeat.
        return runner.run(ops.problem, seed=ops.seed, verify=False)
    return runner.run(ops.A, ops.B, ops.problem)


def _verify(variant: str, cfg, runner, result, ops: _Operands) -> float:
    """max_rel of the GPU result against a numpy reference."""
    if variant == "multi_abd":
        # The runner generates its own operands, so it also owns the reference
        # (mirroring ck_tile::reference_gemm_multiple_abd). One extra launch.
        verified = runner.run(ops.problem, seed=ops.seed, verify=True, verify_tol=_RTOL)
        if verified.max_rel is None:
            raise RuntimeError(f"multi_abd verification did not run (status={verified.status})")
        return verified.max_rel

    # Only GemmResult carries max_rel; the grouped/multi_d result types do not.
    if getattr(result, "max_rel", None) is not None:
        return result.max_rel

    dtype = cfg.dtype_a
    if variant == "grouped":
        worst = 0.0
        Bq = _quantize(ops.B, dtype)
        for out, A_g in zip(result.outputs, ops.A_list):
            ref = _quantize(A_g, dtype) @ Bq
            worst = max(worst, _max_rel(np.asarray(out).astype(np.float32), ref))
        return worst

    if variant == "multi_d":
        # multi_d's fused epilogue op lives in `elementwise_op`; the similarly
        # named `cde_elementwise_op` is the multi_abd field and stays at its
        # PassThrough default here, so reading it would reference the wrong op.
        # Read it per-config rather than from --elementwise-op: the sweep now
        # enumerates MultiDAdd and MultiDMultiply together, so a single pinned
        # op would mis-reference half the configs.
        #
        # nd comes from the kernel, not the config, and the D pool is sized to
        # the largest num_d in the sample -- num_d is itself swept over {1, 2}.
        nd = runner.num_d_tensors
        acc = _quantize(ops.A, dtype) @ _quantize(ops.B, dtype)
        ref = _cde_reference(cfg.elementwise_op, acc, ops.Ds[:nd])
        return _max_rel(result.output.astype(np.float32), ref)

    # standard / stream_k
    #
    # Quantize operands to the *input* dtype and the result to the *output*
    # dtype -- they differ (expand_sweep maps fp8/bf8 inputs to an fp16 C), and
    # conflating them is what a naive "round everything to `dtype`" reference
    # gets wrong: for fp8/bf8 it would round the operands to fp16, leaving the
    # reference strictly more precise than the kernel and failing every fp8/bf8
    # config by construction.
    acc = _quantize(ops.A, dtype) @ _quantize(ops.B, dtype)
    ref = _quantize(acc, cfg.dtype_c)
    return _max_rel(result.output, ref)


def run(args) -> int:
    # fallback="" so a failed detection is falsy and the skip below can fire.
    # detect_gpu_arch defaults to fallback="gfx942" (ctypes_utils.py), which
    # never returns empty -- with the default, a CPU-only runner walked straight
    # into hipcc instead of reporting skipped, which is the exact false signal
    # the exit-77 convention exists to remove. Matches the aquant/abquant/bquant
    # GPU tests, which all spell this out the same way.
    arch = args.arch or detect_gpu_arch(fallback="") or None
    if not arch:
        print("SKIP: no GPU detected and --arch not given")
        return _SKIP

    if arch not in _SUPPORTED_ARCHS:
        print(f"SKIP: search-space sweep is {'/'.join(_SUPPORTED_ARCHS)}-only; "
              f"got {arch}")
        return _SKIP

    variant = args.variant
    def_dtypes, def_layouts = _VARIANT_DEFAULTS[variant]
    dtypes = [d.strip() for d in args.dtypes.split(",")] if args.dtypes else def_dtypes
    layouts = [l.strip() for l in args.layouts.split(",")] if args.layouts else def_layouts
    budget = args.budget
    seed = args.seed if args.seed is not None else _daily_seed()
    size = args.size

    print(f"variant={variant}  arch={arch}  dtypes={dtypes}  layouts={layouts}")
    print(f"budget={budget if budget > 0 else 'unlimited'}  seed={seed}  size={size}")

    # --- Enumerate: same JSON config as tile_engine, across all strata ---
    print("\nEnumerating search space...")
    t0 = time.time()
    strata = {}
    ci_config = _ci_config_for(variant)
    for dtype in dtypes:
        for layout in layouts:
            configs = expand_sweep(str(ci_config), arch=arch, dtype=dtype,
                                   layout=layout, variant=variant)
            if variant == "multi_d" and args.elementwise_op is not None:
                # Only when explicitly asked for. The config dataclass derives
                # .name (and the codegen's elementwise_ops list) from this field,
                # so replacing it is enough to select a different kernel -- which
                # makes `--elementwise-op PassThrough` a usable A/B experiment.
                # It is not a usable default: PassThrough takes Ds as an unnamed
                # parameter pack device-side and _cde_reference returns acc
                # host-side, so both sides compute A@B and a kernel that never
                # loads D still passes.
                configs = [replace(c, elementwise_op=args.elementwise_op)
                           for c in configs]
            strata[f"{dtype}/{layout}"] = configs
    total = sum(len(v) for v in strata.values())
    print(f"  {total} valid configs across {len(strata)} strata ({time.time()-t0:.1f}s)")
    for key, cfgs in sorted(strata.items()):
        print(f"    {key}: {len(cfgs)}")

    # --- Sample ---
    if budget > 0 and budget < total:
        configs = _sample(strata, budget=budget, seed=seed)
        print(f"\nSampled {len(configs)}/{total} configs (budget={budget})")
    else:
        configs = [c for v in strata.values() for c in v]
        print(f"\nRunning all {len(configs)} configs (no cap)")

    if not configs:
        print("ERROR: no configs to run")
        return 1

    # --- Build ---
    print(f"\nBuilding {len(configs)} .so files (parallel JIT)...")
    t0 = time.time()
    so_paths = setup_multiple_gemm_dispatchers(configs, verbose=False)
    build_ok = sum(1 for p in so_paths if p is not None)
    print(f"  Built {build_ok}/{len(configs)} in {time.time()-t0:.1f}s "
          f"({len(configs)-build_ok} failed)")

    # --- Run ---
    print(f"\nRunning {build_ok} kernels on GPU (M=N=K={size}, "
          f"warmup={args.warmup}, repeat={args.repeat})...")
    # num_d is per-kernel, but every config in a sweep shares the same operand
    # shapes, so generate the largest D set once and slice per kernel.
    max_num_d = max((getattr(c, "num_d_tensors", 0) for c in configs), default=0)
    ops = _Operands(size=size, seed=seed, groups=args.groups, num_d=max_num_d)
    flops = ops.grouped_problem.flops if variant == "grouped" else ops.problem.flops

    results = []
    n_pass = n_fail = n_build_fail = 0

    for cfg, so in zip(configs, so_paths):
        if so is None:
            n_build_fail += 1
            results.append({"name": cfg.name, "status": "build_fail"})
            continue
        try:
            runner = _make_runner(variant, cfg, so)
            for _ in range(args.warmup):
                _invoke(variant, runner, ops)
            times = []
            result = None
            for _ in range(max(1, args.repeat)):
                result = _invoke(variant, runner, ops)
                if result.success:
                    times.append(result.time_ms)
        except Exception as exc:
            n_fail += 1
            results.append({"name": cfg.name, "status": "run_error", "error": str(exc)})
            continue
        if result is None or not result.success:
            n_fail += 1
            results.append({"name": cfg.name, "status": "run_fail",
                            "error": f"status={getattr(result, 'status', 'unknown')}"})
            continue
        # Use avg TFLOPS from timed repeat runs; fall back to result.tflops (single run).
        if times:
            avg_ms = sum(times) / len(times)
            tflops = (flops / (avg_ms * 1e-3)) / 1e12
        else:
            tflops = result.tflops
        try:
            mr = _verify(variant, cfg, runner, result, ops)
        except Exception as exc:
            n_fail += 1
            results.append({"name": cfg.name, "status": "verify_error", "error": str(exc)})
            continue
        ok = mr <= _RTOL
        n_pass += ok
        n_fail += not ok
        results.append({"name": cfg.name, "status": "pass" if ok else "fail",
                        "tflops": round(tflops, 3), "max_rel": round(mr, 6)})

    # --- Report ---
    print(f"\n{'='*60}")
    print(f"Results: {n_pass} pass / {n_fail} fail / {n_build_fail} build-fail "
          f"/ {len(configs)} total")

    if args.json:
        out = {"variant": variant, "arch": arch, "size": size, "seed": seed,
               "budget": budget, "total_configs": total, "n_pass": n_pass,
               "n_fail": n_fail, "n_build_fail": n_build_fail, "kernels": results}
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"Results written to {args.json}")

    if n_fail > 0 or n_build_fail > 0:
        print("\nFailed kernels:")
        for r in results:
            if r["status"] != "pass":
                print(f"  {r['name']}: {r['status']} "
                      f"{r.get('error', r.get('max_rel', ''))}")

    # A build failure is a failure. Excluding it would mean a total codegen or
    # compile breakage -- every kernel failing to build, nothing verified --
    # still reports a green lane.
    return 0 if n_fail == 0 and n_build_fail == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(
        description="Dispatcher GEMM search space runner (all bridged variants)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--variant", default="standard", choices=_VARIANTS,
                   help="GEMM variant to sweep (default: standard)")
    p.add_argument("--arch", default=None,
                   help="GPU arch (default: auto-detect via rocminfo)")
    p.add_argument("--dtypes", default=None,
                   help="Comma-separated dtypes (default: per-variant; "
                        f"{','.join(_DTYPES)} for standard/grouped/stream_k, "
                        "fp16 for multi_d/multi_abd)")
    p.add_argument("--layouts", default=None,
                   help="Comma-separated layouts (default: per-variant; "
                        f"{','.join(_LAYOUTS)} for standard/grouped/stream_k, "
                        "rcrr for multi_d/multi_abd)")
    p.add_argument("--groups", type=int, default=4,
                   help="Sub-problem count for --variant grouped (default: 4)")
    p.add_argument("--elementwise-op", default=None,
                   choices=("PassThrough", "MultiDAdd", "MultiDMultiply"),
                   help="Pin the epilogue op for --variant multi_d, overriding "
                        "the sweep (default: sweep MultiDAdd/MultiDMultiply from "
                        "the config). PassThrough discards the D tensors on both "
                        "the device and reference sides, so it verifies plain "
                        "A@B only. Ignored by every other variant.")
    p.add_argument("--budget", type=int, default=500,
                   help="Max configs to run; 0 = no cap (default: 500)")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed; default = daily rotating seed")
    p.add_argument("--warmup", type=int, default=5,
                   help="Warmup iterations per kernel (default: 5)")
    p.add_argument("--repeat", type=int, default=5,
                   help="Timed iterations per kernel (default: 5)")
    p.add_argument("--size", type=int, default=1024,
                   help="M=N=K problem size (default: 1024)")
    p.add_argument("--json", default=None,
                   help="Write results JSON to this path")
    return run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
