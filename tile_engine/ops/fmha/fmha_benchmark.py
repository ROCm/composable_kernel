#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
FMHA tile engine benchmark runner.

Uses the dispatcher's setup_multiple_fmha_dispatchers() for pipelined JIT
compilation, then runs GPU benchmarks and reports results.

Usage:
    python fmha_benchmark.py configs/fwd.json
    python fmha_benchmark.py configs/fwd.json --workers 256 --build-dir /tmp/fmha_build
    python fmha_benchmark.py configs/fwd.json --problems "2,8,1024,128" --verify
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

_DISPATCHER_ROOT = Path(__file__).resolve().parents[3] / "dispatcher"
sys.path.insert(0, str(_DISPATCHER_ROOT / "python"))
sys.path.insert(0, str(_DISPATCHER_ROOT / "codegen"))

from fmha_utils import (  # noqa: E402
    FmhaProblem,
    FmhaRunner,
    cpu_attention_fwd,
    detect_gpu_arch,
    setup_multiple_fmha_dispatchers,
)

from fmha.instance_gen import expand_sweep, apply_filter  # noqa: E402

# Reusable multi-GPU job dispatcher (op-agnostic)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from parallel_runner import run_parallel_on_gpus  # noqa: E402


def _compute_result(
    config,
    prob,
    time_ms,
    output,
    ref,
    is_causal,
    ns,
    api_family,
    dtype_tol,
    gpu_id=None,
):
    """Compute tflops, max_err, status and build result dict + display line.

    Returns (result_dict, display_line) or None if time_ms is None/0.
    """
    tflops = prob.num_ops / (time_ms * 1e-3) / 1e12 if time_ms > 0 else 0
    if is_causal and time_ms > 0:
        sq, sk = prob.seqlen_q, prob.seqlen_k
        causal_ratio = (min(sq, sk) + 1) / (2.0 * sk)
        tflops = prob.num_ops * causal_ratio / (time_ms * 1e-3) / 1e12

    max_err = 0.0
    status = "OK"
    if ref is not None and output is not None:
        max_err = float(np.abs(output.astype(np.float32) - ref).max())
        atol, rtol = dtype_tol
        tol = atol + rtol * np.abs(ref).max()
        status = "PASS" if max_err < tol else "FAIL"

    splits_tag = f"  [ns={ns}]" if api_family == "splitkv" else ""
    display_name = f"{config.name}{splits_tag}"
    gpu_tag = f"  [GPU{gpu_id}]" if gpu_id is not None else ""
    display_line = (
        f"  {display_name:<105} {time_ms:>10.3f}"
        f" {tflops:>10.2f} {max_err:>10.2e} {status:>6}{gpu_tag}"
    )

    result_dict = {
        "kernel": config.name,
        "dtype": config.data_type,
        "hdim_q": config.hdim_q,
        "hdim_v": config.hdim_v,
        "pipeline": config.pipeline,
        "mode": config.mode,
        "mask": config.mask,
        "bias": config.bias,
        "tile_m0": config.tile_m0,
        "tile_n0": config.tile_n0,
        "tile_k0": config.tile_k0,
        "tile_n1": config.tile_n1,
        "tile_k1": config.tile_k1,
        "tile_k0max": config.tile_k0max,
        "warp_m0": config.warp_m0,
        "warp_n0": config.warp_n0,
        "warp_k0": config.warp_k0,
        "block_per_cu": config.block_per_cu,
        "num_splits": ns if api_family == "splitkv" else None,
        "problem": {
            "batch": prob.batch,
            "nhead_q": prob.nhead_q,
            "nhead_k": prob.nhead_k,
            "seqlen_q": prob.seqlen_q,
            "seqlen_k": prob.seqlen_k,
            "hdim_q": prob.hdim_q,
            "hdim_v": prob.hdim_v,
        },
        "latency_ms": time_ms,
        "tflops": tflops,
        "max_err": max_err,
        "status": status,
    }
    return result_dict, display_line


def _run_kernel_isolated(
    lib_path, arch, prob, run_kwargs, data_dir, gpu_id=0, timeout=120
):
    """Run a single kernel in a subprocess. Returns (time_ms, output_path) or (None, error_msg).

    Survives GPU faults — if the subprocess crashes, returns an error instead of killing main.
    """
    import json as _json
    import subprocess as sp

    # Write a small runner script that the subprocess will execute.
    # Use json.dumps for string values to safely escape quotes/backslashes in paths.
    _lib = _json.dumps(str(lib_path))
    _arch = _json.dumps(str(arch))
    _pydir = _json.dumps(str(_DISPATCHER_ROOT / "python"))
    _ddir = _json.dumps(str(data_dir))
    script = f'''
import sys, os, numpy as np
os.environ["HIP_VISIBLE_DEVICES"] = "{gpu_id}"
sys.path.insert(0, {_pydir})
from fmha_utils import FmhaRunner, FmhaProblem

runner = FmhaRunner.from_library({_lib}, {_arch})
_d = {_ddir}
Q = np.load(os.path.join(_d, "Q.npy"))
K = np.load(os.path.join(_d, "K.npy"))
V = np.load(os.path.join(_d, "V.npy"))
prob = FmhaProblem(batch={prob.batch}, nhead_q={prob.nhead_q}, nhead_k={prob.nhead_k},
                   seqlen_q={prob.seqlen_q}, seqlen_k={prob.seqlen_k},
                   hdim_q={prob.hdim_q}, hdim_v={prob.hdim_v})
result = runner.run(Q, K, V, prob, **{run_kwargs!r})
if result.success:
    np.save(os.path.join(_d, "O.npy"), result.output)
    print(f"TIME={{result.time_ms}}")
else:
    print("FAIL")
runner.cleanup()
'''
    script_path = os.path.join(data_dir, "run_kernel.py")
    with open(script_path, "w") as f:
        f.write(script)

    try:
        r = sp.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "HIP_VISIBLE_DEVICES": str(gpu_id)},
        )
        if r.returncode != 0:
            err = r.stderr[-200:] if r.stderr else f"exit code {r.returncode}"
            return None, None, f"CRASH: {err.strip()}"
        # Parse time from stdout
        for line in r.stdout.strip().split("\n"):
            if line.startswith("TIME="):
                time_ms = float(line[5:])
                out_path = os.path.join(data_dir, "O.npy")
                output = np.load(out_path) if os.path.exists(out_path) else None
                return time_ms, output, None
        return None, None, "No TIME output"
    except sp.TimeoutExpired:
        return None, None, "TIMEOUT"
    except Exception as e:
        return None, None, str(e)


def parse_problems(spec: str) -> List[FmhaProblem]:
    """Parse problem specs: 'batch,nhead,seqlen,hdim;...'"""
    problems = []
    for part in spec.split(";"):
        vals = [int(x) for x in part.split(",")]
        if len(vals) == 4:
            b, h, s, d = vals
            problems.append(
                FmhaProblem(
                    batch=b,
                    nhead_q=h,
                    nhead_k=h,
                    seqlen_q=s,
                    seqlen_k=s,
                    hdim_q=d,
                    hdim_v=d,
                )
            )
        elif len(vals) == 6:
            b, hq, hk, sq, sk, d = vals
            problems.append(
                FmhaProblem(
                    batch=b,
                    nhead_q=hq,
                    nhead_k=hk,
                    seqlen_q=sq,
                    seqlen_k=sk,
                    hdim_q=d,
                    hdim_v=d,
                )
            )
    return problems


def main():
    parser = argparse.ArgumentParser(description="FMHA Tile Engine Benchmark")
    parser.add_argument(
        "configs", nargs="*", help="Sweep config JSON(s) (optional for exhaustive)"
    )
    parser.add_argument("--arch", default=detect_gpu_arch())
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count() or 8, help="Parallel JIT workers"
    )
    parser.add_argument(
        "--problems",
        default="2,8,1024,128",
        help="Problem sizes: batch,nhead,seqlen,hdim",
    )

    parser.add_argument(
        "--no-verify", action="store_true", help="Skip CPU reference verification"
    )
    parser.add_argument(
        "--best", action="store_true", help="Show best kernel per problem"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="CSV output path (default: <build-dir>/results.csv). Use --no-csv to disable.",
    )
    parser.add_argument("--no-csv", action="store_true", help="Disable CSV output")
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Path to detailed log file (compilation status, failures, timings)",
    )
    parser.add_argument(
        "--build-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "build"),
        help="JIT build output directory",
    )
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--filter",
        dest="filter_expr",
        default="",
        help='Python expr per config, e.g. "c.hdim_q == 128"',
    )
    parser.add_argument(
        "--filter-file", default="", help="Path to .py with filter_config(c) -> bool"
    )
    parser.add_argument(
        "--tiles",
        choices=["rules", "exhaustive"],
        default="rules",
        help="Tile enumeration mode: 'rules' (default) uses constraint-based generation; "
        "'exhaustive' brute-forces ALL compilable tiles (like the oracle)",
    )
    parser.add_argument(
        "--num-splits",
        default="1,2,4,8",
        help="Comma-separated num_splits values to sweep for splitkv (default: 1,2,4,8)",
    )
    parser.add_argument(
        "--isolate",
        action="store_true",
        help="Run each kernel in a subprocess to survive GPU faults (slower but fault-tolerant)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs to use for parallel benchmarking (e.g. '0,1,2,3'). "
        "Implies --isolate. Each GPU runs one kernel at a time.",
    )
    args = parser.parse_args()

    # --gpus implies --isolate
    if args.gpus:
        args.isolate = True
    gpu_ids = [int(x) for x in args.gpus.split(",")] if args.gpus else [0]

    problems = parse_problems(args.problems)
    num_splits_list = [int(x) for x in args.num_splits.split(",")]
    build_dir = Path(args.build_dir).resolve()

    if args.clean and build_dir.exists():
        print(f"  Cleaning {build_dir} ...")
        shutil.rmtree(build_dir)

    build_dir.mkdir(parents=True, exist_ok=True)

    # Phase 0: Expand configs
    all_configs = []
    restrict_hdims = sorted({(p.hdim_q, p.hdim_v) for p in problems})
    if args.tiles == "exhaustive":
        # Exhaustive mode: all tiles (no constraint filter) × full feature cross-product.
        # JSON config is optional — if provided, its trait_config scopes the sweep.
        cfg_path = args.configs[0] if args.configs else None
        all_configs = expand_sweep(
            cfg_path,
            args.arch,
            0,
            mode="exhaustive",
            restrict_hdims=restrict_hdims,
        )
        print(
            f"  Exhaustive: {len(all_configs)} total combos (all tiles × all features)"
        )
    else:
        if not args.configs:
            parser.error(
                "Config JSON(s) required for rules mode. Use --tiles exhaustive to run without."
            )
        for cfg_path in args.configs:
            configs = expand_sweep(
                cfg_path,
                args.arch,
                0,
                mode="rules",
                restrict_hdims=restrict_hdims,
            )
            all_configs.extend(configs)
            print(f"  {cfg_path}: {len(configs)} kernel configs")

    if args.filter_expr or args.filter_file:
        before = len(all_configs)
        all_configs = apply_filter(all_configs, args.filter_expr, args.filter_file)
        print(f"  Filter: {before} -> {len(all_configs)} configs")

    # Remove standalone combine configs -- they are auto-paired during JIT
    all_configs = [c for c in all_configs if c.family != "fwd_splitkv_combine"]

    print(f"\n{'=' * 70}")
    print("FMHA Tile Engine Benchmark")
    print(f"{'=' * 70}")
    print(f"  Arch:     {args.arch}")
    print(f"  Kernels:  {len(all_configs)}")
    print(f"  Problems: {len(problems)}")
    print(f"  Workers:  {args.workers}")
    print(f"  Build:    {build_dir}")

    # Phase 1: Pipelined JIT via the dispatcher
    print(
        f"\n--- Phase 1: JIT compile ({len(all_configs)} kernels,"
        f" {args.workers} workers) ---"
    )
    jit_t0 = time.perf_counter()

    def _progress(stage, done, total):
        elapsed = time.perf_counter() - jit_t0
        pct = done * 100 // total
        print(
            f"\r  [{stage}] {done}/{total} ({pct}%) - {elapsed:.0f}s",
            end="",
            flush=True,
        )
        if done == total:
            print()

    setups = setup_multiple_fmha_dispatchers(
        all_configs,
        output_dir=build_dir,
        verbose=True,
        max_workers=args.workers,
        progress_callback=_progress,
    )

    jit_time = time.perf_counter() - jit_t0
    built = sum(1 for s in setups if s.success)
    failed = len(all_configs) - built
    print(f"\n  Built {built}/{len(all_configs)} in {jit_time:.0f}s ({failed} failed)")

    # Load runners for successfully compiled kernels
    for setup in setups:
        if setup.success and setup.library_path and setup.runner is None:
            try:
                setup.runner = FmhaRunner.from_library(setup.library_path, args.arch)
            except Exception as e:
                print(f"  Warning: Failed to load runner: {e}")
                setup.success = False

    if args.compile_only:
        print(f"\n{'=' * 70}")
        print(f"  Compile-only mode. {built}/{len(all_configs)} kernels compiled.")
        if failed > 0:
            print("\n  Failed kernels:")
            for cfg, s in zip(all_configs, setups):
                if not s.success:
                    err = (s.error or "unknown")[:80]
                    print(f"    {cfg.name}: {err}")
        if args.tiles == "exhaustive":
            # Oracle-style analysis: find tiles missed by rules vs compilable
            from fmha.instance_gen import validate_tile, FmhaTileConfig  # noqa: E402

            missed = []
            for cfg, s in zip(all_configs, setups):
                if s.success:
                    tile = FmhaTileConfig(
                        bm0=cfg.tile_m0,
                        bn0=cfg.tile_n0,
                        bk0=cfg.tile_k0,
                        bn1=cfg.tile_n1,
                        bk1=cfg.tile_k1,
                        bk0max=cfg.tile_k0max,
                        rm0=cfg.wave_m0,
                        rn0=1,
                        rk0=1,
                        rm1=cfg.wave_m1,
                        rn1=1,
                        rk1=1,
                        wm0=cfg.warp_m0,
                        wn0=cfg.warp_n0,
                        wk0=cfg.warp_k0,
                        wm1=cfg.warp_m1,
                        wn1=cfg.warp_n1,
                        wk1=cfg.warp_k1,
                    )
                    if not validate_tile(
                        tile,
                        args.arch,
                        cfg.data_type,
                        cfg.hdim_q,
                        cfg.hdim_v,
                        cfg.pipeline,
                    ):
                        missed.append(cfg)
            if missed:
                print(
                    f"\n  MISSED by rules ({len(missed)} tiles compile but rules reject):"
                )
                seen = set()
                for cfg in missed:
                    key = (cfg.tile_m0, cfg.tile_n0, cfg.tile_k0)
                    if key not in seen:
                        seen.add(key)
                        print(
                            f"    ({cfg.tile_m0:>3}, {cfg.tile_n0:>3}, {cfg.tile_k0:>3})"
                        )
            else:
                print(
                    "\n  Rules are COMPLETE — all compilable tiles are generated by rules."
                )
        print(f"{'=' * 70}")
        return

    # Phase 2: Benchmark
    print(f"\n--- Phase 2: Benchmark ({built} kernels x {len(problems)} problems) ---")

    dtype_map = {
        "fp16": np.float16,
        "bf16": np.float32,
        "fp32": np.float32,
        "fp8": np.float16,
        "fp8bf16": np.float16,
        "fp8fp32": np.float16,
        "bf8": np.float16,
        "mxfp8": np.float16,
        "mxfp4": np.float16,
    }
    # Tolerance per dtype: (atol, rtol)
    _DTYPE_TOL = {
        "fp16": (1e-3, 1e-3),
        "bf16": (1e-2, 1e-2),
        "fp32": (1e-5, 1e-5),
        "fp8": (16.0, 0.0),
        "fp8bf16": (16.0, 0.0),
        "fp8fp32": (16.0, 0.0),
        "bf8": (16.0, 0.0),
        "mxfp8": (16.0, 0.0),
        "mxfp4": (32.0, 0.0),
    }
    np.random.seed(42)
    all_results = []
    bench_t0 = time.perf_counter()

    for prob_idx, prob in enumerate(problems):
        first_dtype = all_configs[0].data_type if all_configs else "fp16"
        first_mask = all_configs[0].mask if all_configs else "no"
        np_dtype = dtype_map.get(first_dtype, np.float16)
        dtype_tol = _DTYPE_TOL.get(first_dtype, (1e-2, 1e-2))
        # Use uniform [0, 1] like CK example (default 'uf' mode) -- produces
        # peaked softmax distributions that actually test kernel correctness.
        # randn*0.1 makes softmax nearly uniform for large hdim, hiding bugs.
        Q = np.random.uniform(0, 1, prob.q_shape()).astype(np_dtype)
        K = np.random.uniform(0, 1, prob.k_shape()).astype(np_dtype)
        V = np.random.uniform(0, 1, prob.v_shape()).astype(np_dtype)

        _MASK_INT = {"no": 0, "top_left": 1, "bottom_right": 2, "generic": 3}
        first_mask_int = _MASK_INT.get(first_mask, 0)

        ref = None
        if not args.no_verify:
            # For bf16: truncate inputs to bf16 precision before computing reference,
            # so reference sees the SAME data the kernel sees (after bf16 encoding).
            if first_dtype == "bf16":
                from fmha_utils import _float32_to_bf16, _bf16_to_float32

                Q_ref = _bf16_to_float32(_float32_to_bf16(Q.astype(np.float32)))
                K_ref = _bf16_to_float32(_float32_to_bf16(K.astype(np.float32)))
                V_ref = _bf16_to_float32(_float32_to_bf16(V.astype(np.float32)))
            else:
                Q_ref = Q.astype(np.float32)
                K_ref = K.astype(np.float32)
                V_ref = V.astype(np.float32)
            ref = cpu_attention_fwd(
                Q_ref,
                K_ref,
                V_ref,
                prob.scale,
                mask_type=first_mask_int,
            )

        h_str = (
            f"H={prob.nhead_q}"
            if prob.nhead_q == prob.nhead_k
            else f"Hq={prob.nhead_q} Hk={prob.nhead_k}"
        )
        s_str = (
            f"S={prob.seqlen_q}"
            if prob.seqlen_q == prob.seqlen_k
            else f"Sq={prob.seqlen_q} Sk={prob.seqlen_k}"
        )
        prob_str = f"B={prob.batch} {h_str} {s_str} D={prob.hdim_q}"
        print(f"\n  Problem [{prob_idx}]: {prob_str}")
        print(
            f"  {'Kernel':<105} {'Time(ms)':>10} {'TFLOPS':>10}"
            f" {'MaxErr':>10} {'Status':>6}"
        )
        print(f"  {'-' * 145}")

        _BIAS_INT = {"no": 0, "bias": 1, "alibi": 2}

        # Build list of (config, setup, run_kwargs, ns) jobs for benchmarking
        bench_jobs = []
        for config, setup in zip(all_configs, setups):
            if not setup.success:
                continue
            if not args.isolate and setup.runner is None:
                continue
            if config.hdim_q != prob.hdim_q or config.hdim_v != prob.hdim_v:
                continue

            mask_int = _MASK_INT.get(config.mask, 0)
            is_causal = config.mask in ("top_left", "bottom_right")
            is_group = config.mode == "group"

            _FAMILY_TO_API = {
                "fwd_splitkv": "splitkv",
                "fwd_pagedkv": "pagedkv",
                "fwd_appendkv": "appendkv",
            }
            api_family = _FAMILY_TO_API.get(config.family, config.family)
            splits_to_try = num_splits_list if api_family == "splitkv" else [0]

            for ns in splits_to_try:
                run_kwargs = dict(
                    mask_type=mask_int,
                    bias_type=_BIAS_INT.get(config.bias, 0),
                    has_lse=int(config.lse),
                    has_dropout=int(config.dropout),
                    has_logits=int(config.logits),
                    has_sink=int(config.sink),
                    data_type=config.data_type,
                    is_group_mode=int(is_group),
                    is_v_rowmajor=int(config.vlayout == "r"),
                    api_family=api_family,
                    window_left=-1,
                    window_right=0 if is_causal else -1,
                )
                if api_family == "splitkv":
                    run_kwargs["num_splits"] = ns
                bench_jobs.append(
                    (config, setup, run_kwargs, ns, api_family, is_causal)
                )

        if args.isolate and len(gpu_ids) > 1:
            # ---- Multi-GPU parallel isolated execution ----
            import tempfile

            # Save input data once, shared by all subprocesses
            shared_data_dir = tempfile.mkdtemp(prefix="fmha_shared_")
            np.save(os.path.join(shared_data_dir, "Q.npy"), Q)
            np.save(os.path.join(shared_data_dir, "K.npy"), K)
            np.save(os.path.join(shared_data_dir, "V.npy"), V)

            def _run_one(job, gpu_id):
                config, setup, run_kwargs, ns, api_family, is_causal = job
                # Per-job output dir (unique per subprocess)
                job_dir = tempfile.mkdtemp(prefix=f"fmha_gpu{gpu_id}_")
                # Symlink shared inputs instead of copying
                for fname in ("Q.npy", "K.npy", "V.npy"):
                    os.symlink(
                        os.path.join(shared_data_dir, fname),
                        os.path.join(job_dir, fname),
                    )
                time_ms, output, err = _run_kernel_isolated(
                    setup.library_path, args.arch, prob, run_kwargs, job_dir, gpu_id
                )
                shutil.rmtree(job_dir, ignore_errors=True)
                return (config, time_ms, output, err, ns, api_family, is_causal, gpu_id)

            print(f"  Running {len(bench_jobs)} kernels across {len(gpu_ids)} GPUs ...")
            for _, result in run_parallel_on_gpus(bench_jobs, gpu_ids, _run_one):
                config, time_ms, output, err, ns, api_family, is_causal, gpu_id = result
                if err:
                    splits_tag = f"  [ns={ns}]" if api_family == "splitkv" else ""
                    print(
                        f"  {config.name}{splits_tag:<105} {'---':>10} {'---':>10} {'---':>10} GPU{gpu_id} {err[:15]}"
                    )
                    continue

                r, line = _compute_result(
                    config,
                    prob,
                    time_ms,
                    output,
                    ref,
                    is_causal,
                    ns,
                    api_family,
                    dtype_tol,
                    gpu_id,
                )
                print(line)
                all_results.append(r)

            shutil.rmtree(shared_data_dir, ignore_errors=True)

        else:
            # ---- Sequential execution (in-process or single-GPU isolated) ----
            for config, setup, run_kwargs, ns, api_family, is_causal in bench_jobs:
                time_ms = None
                output = None
                if args.isolate:
                    import tempfile

                    data_dir = tempfile.mkdtemp(prefix="fmha_run_")
                    np.save(os.path.join(data_dir, "Q.npy"), Q)
                    np.save(os.path.join(data_dir, "K.npy"), K)
                    np.save(os.path.join(data_dir, "V.npy"), V)
                    time_ms, output, err = _run_kernel_isolated(
                        setup.library_path,
                        args.arch,
                        prob,
                        run_kwargs,
                        data_dir,
                        gpu_ids[0],
                    )
                    shutil.rmtree(data_dir, ignore_errors=True)
                    if err:
                        print(
                            f"  {config.name:<105} {'---':>10} {'---':>10} {'---':>10} {err[:20]:>6}"
                        )
                        continue
                else:
                    result = setup.runner.run(Q, K, V, prob, **run_kwargs)
                    if not result.success:
                        continue
                    time_ms = result.time_ms
                    output = result.output

                r, line = _compute_result(
                    config,
                    prob,
                    time_ms,
                    output,
                    ref,
                    is_causal,
                    ns,
                    api_family,
                    dtype_tol,
                )
                print(line)
                all_results.append(r)

    bench_time = time.perf_counter() - bench_t0

    # Cleanup
    for setup in setups:
        if setup.success and setup.runner:
            try:
                setup.runner.cleanup()
            except Exception:
                pass

    # Report
    print(f"\n{'=' * 70}")
    print(f"  JIT:       {jit_time:.0f}s ({built} kernels)")
    print(f"  Benchmark: {bench_time:.1f}s")
    print(f"  Results:   {len(all_results)} measurements")

    if all_results:
        from collections import defaultdict

        by_problem = defaultdict(list)
        for r in all_results:
            key = json.dumps(r["problem"], sort_keys=True)
            by_problem[key].append(r)

        print("\n  Best kernel per problem:")
        for key, results in by_problem.items():
            best = max(results, key=lambda x: x["tflops"])
            prob = json.loads(key)
            ns_tag = f"  [ns={best['num_splits']}]" if best.get("num_splits") else ""
            h_str = (
                f"H={prob['nhead_q']}"
                if prob["nhead_q"] == prob["nhead_k"]
                else f"Hq={prob['nhead_q']} Hk={prob['nhead_k']}"
            )
            s_str = (
                f"S={prob['seqlen_q']}"
                if prob["seqlen_q"] == prob["seqlen_k"]
                else f"Sq={prob['seqlen_q']} Sk={prob['seqlen_k']}"
            )
            print(
                f"    B={prob['batch']} {h_str}"
                f" {s_str} D={prob['hdim_q']}"
                f" -> {best['kernel']}{ns_tag}"
                f" ({best['tflops']:.2f} TFLOPS, {best['latency_ms']:.3f} ms)"
            )

    # CSV output: default to <build-dir>/results.csv; merge with existing file
    # keeping the faster result (higher tflops) for duplicate kernel+problem keys.
    _CSV_FIELDS = [
        "kernel",
        "dtype",
        "pipeline",
        "mode",
        "mask",
        "bias",
        "hdim_q",
        "hdim_v",
        "tile_m0",
        "tile_n0",
        "tile_k0",
        "tile_n1",
        "tile_k1",
        "tile_k0max",
        "warp_m0",
        "warp_n0",
        "warp_k0",
        "block_per_cu",
        "num_splits",
        "batch",
        "nhead_q",
        "nhead_k",
        "seqlen_q",
        "seqlen_k",
        "latency_ms",
        "tflops",
        "max_err",
        "status",
    ]
    csv_path = args.csv if args.csv else str(build_dir / "results.csv")
    if not args.no_csv and all_results:
        # Build map of new results keyed by (kernel, problem-tuple)
        def _csv_key(row):
            p = row["problem"] if "problem" in row else row
            return (
                row["kernel"],
                row.get("num_splits", 0),
                p.get("batch"),
                p.get("nhead_q"),
                p.get("nhead_k"),
                p.get("seqlen_q"),
                p.get("seqlen_k"),
                p.get("hdim_q"),
                p.get("hdim_v"),
            )

        # Load existing CSV if present
        existing = {}
        if os.path.isfile(csv_path):
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert numeric fields back from strings
                    for k in row:
                        if k in ("latency_ms", "tflops", "max_err"):
                            try:
                                row[k] = float(row[k])
                            except (ValueError, TypeError):
                                pass
                        elif k in (
                            "hdim_q",
                            "hdim_v",
                            "tile_m0",
                            "tile_n0",
                            "tile_k0",
                            "tile_n1",
                            "tile_k1",
                            "tile_k0max",
                            "warp_m0",
                            "warp_n0",
                            "warp_k0",
                            "block_per_cu",
                            "num_splits",
                            "batch",
                            "nhead_q",
                            "nhead_k",
                            "seqlen_q",
                            "seqlen_k",
                        ):
                            try:
                                row[k] = int(row[k])
                            except (ValueError, TypeError):
                                pass
                    key = _csv_key(row)
                    existing[key] = row

        # Merge new results — keep whichever is faster
        for r in all_results:
            row = {**r, **r["problem"]}
            del row["problem"]
            key = _csv_key(r)
            prev = existing.get(key)
            if prev is None or float(row.get("tflops", 0)) > float(
                prev.get("tflops", 0)
            ):
                existing[key] = row

        # Write merged + sorted CSV
        merged = sorted(
            existing.values(), key=lambda x: float(x.get("tflops", 0)), reverse=True
        )
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in merged:
                writer.writerow(row)
        print(f"\n  CSV: {csv_path} ({len(merged)} rows, sorted by tflops)")

    if args.json:
        report = {
            "metadata": {
                "arch": args.arch,
                "jit_time_s": jit_time,
                "bench_time_s": bench_time,
                "num_kernels": len(all_configs),
                "num_built": built,
                "num_problems": len(problems),
            },
            "results": all_results,
        }
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  JSON: {args.json}")

    if args.log:
        from datetime import datetime

        with open(args.log, "w") as lf:
            lf.write(f"FMHA Benchmark Log - {datetime.now().isoformat()}\n")
            lf.write(f"{'=' * 80}\n\n")
            lf.write(f"Command: {' '.join(sys.argv)}\n")
            lf.write(f"Arch: {args.arch}\n")
            lf.write(f"Tiles mode: {args.tiles}\n")
            lf.write(f"Workers: {args.workers}\n")
            lf.write(f"Build dir: {build_dir}\n")
            lf.write(f"Total configs: {len(all_configs)}\n")
            lf.write(f"Built: {built}\n")
            lf.write(f"Failed: {failed}\n")
            lf.write(f"JIT time: {jit_time:.1f}s\n")
            lf.write(f"Bench time: {bench_time:.1f}s\n")
            lf.write(f"Problems: {[str(p) for p in problems]}\n\n")

            # All configs attempted
            lf.write(f"{'=' * 80}\n")
            lf.write(f"ALL CONFIGS ({len(all_configs)})\n")
            lf.write(f"{'=' * 80}\n\n")
            for i, (cfg, setup) in enumerate(zip(all_configs, setups)):
                status = "OK" if setup.success else "FAILED"
                lf.write(f"[{i:4d}] {status:6s} {cfg.name}\n")
                lf.write(
                    f"         tile=({cfg.tile_m0},{cfg.tile_n0},{cfg.tile_k0},{cfg.tile_n1},{cfg.tile_k1},{cfg.tile_k0max})"
                    f"  warp=({cfg.warp_m0},{cfg.warp_n0},{cfg.warp_k0})"
                    f"  bpc={cfg.block_per_cu}\n"
                )
                if not setup.success and setup.error:
                    lf.write(f"         error: {setup.error}\n")
                lf.write("\n")

            # Failed configs summary
            lf.write(f"\n{'=' * 80}\n")
            lf.write(f"FAILED CONFIGS ({failed})\n")
            lf.write(f"{'=' * 80}\n\n")
            for cfg, setup in zip(all_configs, setups):
                if not setup.success:
                    lf.write(f"  {cfg.name}\n")
                    if setup.error:
                        lf.write(f"    {setup.error}\n")

            # Benchmark results
            if all_results:
                lf.write(f"\n{'=' * 80}\n")
                lf.write(f"BENCHMARK RESULTS ({len(all_results)} measurements)\n")
                lf.write(f"{'=' * 80}\n\n")
                sorted_results = sorted(all_results, key=lambda x: -x["tflops"])
                for r in sorted_results:
                    p = r["problem"]
                    lf.write(
                        f"  {r['tflops']:8.2f} TFLOPS  {r['latency_ms']:8.3f} ms"
                        f"  B={p['batch']} H={p['nhead_q']} S={p['seqlen_q']} D={p['hdim_q']}"
                        f"  {r['kernel']}\n"
                    )

        print(f"  Log: {args.log}")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
