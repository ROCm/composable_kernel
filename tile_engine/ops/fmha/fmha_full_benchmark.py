#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Full FMHA benchmark sweep.

JIT-compiles FMHA kernels, then for EACH test shape finds all matching
kernels and benchmarks them, streaming results incrementally to CSV/JSON.

Results are printed live per-shape with the best kernel highlighted.
TFLOPS and latency come directly from CK's HIP event timing.

Usage:
    # Full sweep
    python fmha_full_benchmark.py --workers 256

    # Quick end-to-end test
    python fmha_full_benchmark.py --category smoke --variant fwd --max-kernels 10 --workers 4

    # Filter to h128 fp16
    python fmha_full_benchmark.py --filter "c.hdim_q == 128 and c.data_type == 'fp16'"
"""

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_DISPATCHER_ROOT = _THIS_DIR.parents[2] / "dispatcher"
sys.path.insert(0, str(_DISPATCHER_ROOT / "python"))
sys.path.insert(0, str(_DISPATCHER_ROOT / "codegen"))
sys.path.insert(0, str(_THIS_DIR))

from fmha_utils import (  # noqa: E402
    detect_gpu_arch,
    setup_multiple_fmha_dispatchers,
)
from fmha.instance_gen import expand_sweep, apply_filter  # noqa: E402

YAML_PATH = _THIS_DIR / "ck_fmha_testing_matrix.yaml"

VARIANT_CONFIGS = {
    "fwd": "configs/receipt0_fwd.json",
    "splitkv": "configs/splitkv.json",
    "pagedkv": "configs/pagedkv.json",
    "appendkv": "configs/appendkv.json",
    "batch_prefill": "configs/batch_prefill.json",
    "bwd": "configs/bwd.json",
}

# Variant -> YAML section mapping. KV-cache variants use forward_tests shapes.
VARIANT_YAML_SECTIONS = {
    "fwd": ["forward_tests"],
    "splitkv": ["forward_tests"],
    "pagedkv": ["forward_tests"],
    "appendkv": ["forward_tests"],
    "batch_prefill": ["forward_tests"],
    "bwd": ["backward_tests"],
}

DTYPE_CK = {"fp16": "fp16", "bf16": "bf16", "fp8bf16": "fp8bf16", "fp8fp32": "fp8fp32"}
DTYPE_NP = {
    "fp16": np.float16,
    "bf16": np.float16,
    "fp32": np.float32,
    "fp8bf16": np.float16,
    "fp8fp32": np.float16,
}
ELEM_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4, "fp8bf16": 1, "fp8fp32": 1}

MASK_INT = {"no": 0, "top_left": 1, "generic": 3}
BIAS_INT = {"no": 0, "bias": 1, "alibi": 2}
KV_LAYOUT_INT = {"vectorized": 0, "linear": 1}
KV_LOOKUP_INT = {"vllm": 0, "sglang": 1}


@dataclass
class TestShape:
    name: str
    category: str
    variant: str
    batch: int
    seqlen_q: int
    seqlen_k: int
    nhead_q: int
    nhead_k: int
    hdim_q: int
    hdim_v: int
    dtype: str
    mask: str = "no_mask"
    bias: str = "none"
    dropout: float = 0.0
    lse: bool = False


def parse_yaml(
    yaml_path: Path, category: str = "smoke", sections: Optional[List[str]] = None
) -> List[TestShape]:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    shapes = []
    cats = ["smoke"]
    if category in ("full", "nightly"):
        cats.append("full")
    if category == "nightly":
        cats.append("nightly")

    section_variant_map = [("forward_tests", "fwd"), ("backward_tests", "bwd")]
    if sections:
        section_variant_map = [(s, v) for s, v in section_variant_map if s in sections]

    for section, variant in section_variant_map:
        if section not in data:
            continue
        for cat in cats:
            for test in data[section].get(cat, []):
                for combo in itertools.product(
                    test.get("batch", [1]),
                    test.get("seqlen_q", [1024]),
                    test.get("seqlen_k", [1024]),
                    test.get("nhead_q", [16]),
                    test.get("nhead_k", [16]),
                    test.get("hdim_q", [128]),
                    test.get("hdim_v", [128]),
                    test.get("dtype", ["fp16"]),
                    test.get("mask", ["no_mask"]),
                    test.get("bias", ["none"]),
                    test.get("dropout", [0.0]),
                    test.get("lse", [False]),
                ):
                    b, sq, sk, hq, hk, dq, dv, dt, m, bi, dr, ls = combo
                    shapes.append(
                        TestShape(
                            test["name"],
                            cat,
                            variant,
                            b,
                            sq,
                            sk,
                            hq,
                            hk,
                            dq,
                            dv,
                            dt,
                            mask=m,
                            bias=bi,
                            dropout=dr,
                            lse=ls,
                        )
                    )
    return shapes


def bandwidth_gb_s(shape: TestShape, latency_ms: float) -> float:
    if latency_ms <= 0:
        return 0.0
    eb = ELEM_BYTES.get(shape.dtype, 2)
    total = (
        shape.batch
        * (
            shape.nhead_q * shape.seqlen_q * shape.hdim_q
            + shape.nhead_k * shape.seqlen_k * shape.hdim_q
            + shape.nhead_k * shape.seqlen_k * shape.hdim_v
            + shape.nhead_q * shape.seqlen_q * shape.hdim_v
        )
        * eb
    )
    return total / (latency_ms * 1e6)


FAMILY_TO_API = {
    "fwd": "fwd",
    "fwd_splitkv": "splitkv",
    "fwd_splitkv_combine": "splitkv",
    "fwd_pagedkv": "pagedkv",
    "fwd_appendkv": "appendkv",
    "batch_prefill": "batch_prefill",
    "bwd_dot_do_o": "bwd",
    "bwd_dq_dk_dv": "bwd",
    "bwd_convert_dq": "bwd",
}


def _config_to_serializable(config, so_path: str) -> dict:
    """Convert FmhaKernelConfig + so_path to a picklable dict for subprocess."""
    return {
        "so_path": so_path,
        "api_family": FAMILY_TO_API.get(config.family, "fwd"),
        "data_type": config.data_type,
        "kernel": config.name,
        "family": config.family,
        "mode": config.mode,
        "pipeline": config.pipeline,
        "tile_m0": config.tile_m0,
        "tile_n0": config.tile_n0,
        "tile_k0": config.tile_k0,
        "tile_n1": config.tile_n1,
        "tile_k1": config.tile_k1,
        "tile_k0max": config.tile_k0max,
        "pad_s": config.pad_s,
        "pad_sk": config.pad_sk,
        "pad_d": config.pad_d,
        "pad_dv": config.pad_dv,
        "mask": config.mask,
        "bias": config.bias,
        "lse": config.lse,
        "dropout": config.dropout,
        "logits": config.logits,
        "sink": config.sink,
        "skip": config.skip_min_seqlen_q,
        "qscale": config.qscale,
        "paged_kv": config.paged_kv,
        "rope": config.rope,
        "deterministic": config.deterministic,
        "dbias": config.dbias,
        "mask_int": MASK_INT.get(config.mask, 0),
        "bias_int": BIAS_INT.get(config.bias, 0),
        "has_lse": int(config.lse),
        "has_dropout": int(config.dropout not in (False, 0, "no", "False")),
        "has_logits": int(config.logits),
        "has_sink": int(config.sink),
        "has_skip": int(config.skip_min_seqlen_q),
        "has_dbias": int(getattr(config, "dbias", False)),
        "is_store_randval": int(getattr(config, "store_randval", False)),
        "page_size": getattr(config, "page_size", 16),
        "kv_layout": KV_LAYOUT_INT.get(
            getattr(config, "kv_memory_layout", "vectorized"), 0
        ),
        "kv_lookup": KV_LOOKUP_INT.get(getattr(config, "kv_lookup_table", "sglang"), 1),
    }


def _shape_to_dict(shape: TestShape) -> dict:
    return {
        "name": shape.name,
        "category": shape.category,
        "variant": shape.variant,
        "batch": shape.batch,
        "seqlen_q": shape.seqlen_q,
        "seqlen_k": shape.seqlen_k,
        "nhead_q": shape.nhead_q,
        "nhead_k": shape.nhead_k,
        "hdim_q": shape.hdim_q,
        "hdim_v": shape.hdim_v,
        "dtype": shape.dtype,
        "mask": shape.mask,
        "bias": shape.bias,
        "dropout": shape.dropout,
        "lse": shape.lse,
    }


def main():
    p = argparse.ArgumentParser(description="Full FMHA Benchmark Sweep")
    p.add_argument("--arch", default=detect_gpu_arch())
    p.add_argument("--category", default="smoke", choices=["smoke", "full", "nightly"])
    p.add_argument("--variant", default="all")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--build-dir", default="/tmp/fmha_full_bench")
    p.add_argument("--filter", dest="filter_expr", default="")
    p.add_argument("--filter-file", default="")
    p.add_argument("--csv", default="fmha_sweep_results.csv")
    p.add_argument("--json", default="fmha_sweep_results.json")
    p.add_argument("--compile-only", action="store_true")
    p.add_argument("--max-kernels", type=int, default=0)
    p.add_argument(
        "--shape-timeout",
        type=int,
        default=600,
        help="Per-shape timeout in seconds (0=none)",
    )
    args = p.parse_args()

    build_dir = Path(args.build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    variants = list(VARIANT_CONFIGS.keys()) if args.variant == "all" else [args.variant]

    # ---- Phase 1: Parse shapes ----
    print(f"\n{'=' * 80}")
    print("Phase 1: Parse test shapes")
    print(f"{'=' * 80}")

    all_shapes: List[TestShape] = []
    for variant in variants:
        sections = VARIANT_YAML_SECTIONS.get(variant, ["forward_tests"])
        vshapes = parse_yaml(YAML_PATH, args.category, sections=sections)
        for s in vshapes:
            s.variant = variant
        all_shapes.extend(vshapes)

    print(f"  Category: {args.category}")
    print(f"  Variants: {variants}")
    print(f"  Total shapes: {len(all_shapes)}")

    # ---- Phase 2: Compile ----
    print(f"\n{'=' * 80}")
    print("Phase 2: Compile kernels")
    print(f"{'=' * 80}")

    # kernel_index: (hdim_q, hdim_v, dtype, variant) -> list of (so_path, cfg_dict)
    kernel_index: Dict[tuple, List[tuple]] = {}

    from concurrent.futures import ProcessPoolExecutor as _PPE

    _compile_pool = _PPE(max_workers=args.workers)
    BATCH_SIZE = 200

    for variant in variants:
        cfg_path = str(_THIS_DIR / VARIANT_CONFIGS[variant])
        if not Path(cfg_path).exists():
            continue
        configs = expand_sweep(cfg_path, args.arch)
        if args.filter_expr or args.filter_file:
            configs = apply_filter(configs, args.filter_expr, args.filter_file)
        if args.max_kernels > 0:
            configs = configs[: args.max_kernels]
        if not configs:
            continue

        n_batches = (len(configs) + BATCH_SIZE - 1) // BATCH_SIZE
        print(
            f"\n  {variant}: {len(configs)} configs, {args.workers} workers, {n_batches} batches..."
        )
        t0 = time.perf_counter()
        setups = []
        total_ok = 0
        for bi in range(n_batches):
            batch_cfgs = configs[bi * BATCH_SIZE : (bi + 1) * BATCH_SIZE]
            batch_setups = setup_multiple_fmha_dispatchers(
                batch_cfgs,
                output_dir=build_dir,
                max_workers=args.workers,
                executor=_compile_pool,
            )
            batch_ok = sum(1 for s in batch_setups if s.success)
            batch_n = len(batch_cfgs)
            total_ok += batch_ok
            setups.extend(zip(batch_cfgs, batch_setups))
            del batch_setups, batch_cfgs
            print(
                f"    Batch {bi + 1}/{n_batches}: {batch_ok}/{batch_n} "
                f"(total {total_ok}, {time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
        ok = total_ok
        print(f"    Built {ok}/{len(configs)} in {time.perf_counter() - t0:.0f}s")

        for config, setup in setups:
            if not setup.success:
                continue
            so_path = getattr(setup, "library_path", "") or ""
            if not so_path:
                candidate = build_dir / f"libdispatcher_fmha_{config.name}.so"
                if candidate.exists():
                    so_path = str(candidate)
            if not so_path:
                continue
            cfg_dict = _config_to_serializable(config, so_path)
            key = (config.hdim_q, config.hdim_v, config.data_type, variant, config.mode)
            kernel_index.setdefault(key, []).append((so_path, cfg_dict))

    _compile_pool.shutdown(wait=True)
    del _compile_pool

    total_built = sum(len(v) for v in kernel_index.values())
    print(f"\n  Total compiled: {total_built}")
    print(f"  Unique (hdim,dtype,variant) groups: {len(kernel_index)}")

    if args.compile_only:
        print(f"\n  Compile-only. {total_built} kernels ready.")
        return

    # ---- Phase 3: Benchmark (serial, one subprocess per kernel) ----
    print(f"\n{'=' * 80}")
    print("Phase 3: Benchmark (one subprocess per kernel, serial GPU)")
    print(f"{'=' * 80}")

    csv_path = Path(args.csv) if os.path.isabs(args.csv) else _THIS_DIR / args.csv
    csv_fields = [
        "problem_name",
        "batch",
        "seqlen_q",
        "seqlen_k",
        "nhead_q",
        "nhead_k",
        "hdim_q",
        "hdim_v",
        "dtype",
        "kernel",
        "family",
        "mode",
        "pipeline",
        "tile_m0",
        "tile_n0",
        "tile_k0",
        "tile_n1",
        "tile_k1",
        "tile_k0max",
        "pad_s",
        "pad_sk",
        "pad_d",
        "pad_dv",
        "mask",
        "bias",
        "lse",
        "dropout",
        "logits",
        "sink",
        "skip",
        "qscale",
        "paged_kv",
        "rope",
        "deterministic",
        "dbias",
        "latency_ms",
        "tflops",
        "bandwidth_gb_s",
    ]

    # Resume: load already-completed measurements
    completed: set = set()
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                completed.add(
                    (
                        row.get("kernel", ""),
                        row.get("problem_name", ""),
                        str(row.get("batch", "")),
                        str(row.get("seqlen_q", "")),
                        row.get("dtype", ""),
                    )
                )
        csv_file = open(csv_path, "a", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        print(f"  Resuming: {len(completed)} measurements already in CSV")
    else:
        csv_file = open(csv_path, "w", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()

    # Pre-filter: match shapes to kernels by (hdim, dtype, variant, mode).
    # YAML shapes are batch-mode only. Group-mode kernels need seqstart arrays
    # which batch shapes don't provide -- they all GPU fault.
    runnable = []
    for shape in all_shapes:
        ck_dtype = DTYPE_CK.get(shape.dtype, shape.dtype)
        key = (shape.hdim_q, shape.hdim_v, ck_dtype, shape.variant, "batch")
        entries = kernel_index.get(key, [])
        if entries:
            runnable.append((shape, entries))

    # Flatten to work items, skip already-completed
    def _resume_key(cfg, shape):
        return (
            cfg.get("kernel", ""),
            shape.name,
            str(shape.batch),
            str(shape.seqlen_q),
            shape.dtype,
        )

    work_items = []
    skipped = 0
    for shape, kernel_entries in runnable:
        for so_path, cfg in kernel_entries:
            if _resume_key(cfg, shape) in completed:
                skipped += 1
            else:
                work_items.append((shape, so_path, cfg))

    total_work = len(work_items) + skipped
    total_measurements = len(completed)
    total_gpu_faults = 0
    bench_t0 = time.perf_counter()
    BENCH_BATCH = 50

    worker_path = _THIS_DIR / "run_one_kernel.py"
    worker_env = os.environ.copy()
    worker_env["FMHA_PYPATH_1"] = str(_DISPATCHER_ROOT / "python")
    worker_env["FMHA_PYPATH_2"] = str(_DISPATCHER_ROOT / "codegen")

    CFG_KEYS = [
        "kernel",
        "family",
        "mode",
        "pipeline",
        "tile_m0",
        "tile_n0",
        "tile_k0",
        "tile_n1",
        "tile_k1",
        "tile_k0max",
        "pad_s",
        "pad_sk",
        "pad_d",
        "pad_dv",
        "mask",
        "bias",
        "lse",
        "dropout",
        "logits",
        "sink",
        "skip",
        "qscale",
        "paged_kv",
        "rope",
        "deterministic",
        "dbias",
    ]

    print(f"  Runnable shapes: {len(runnable)}")
    print(f"  Total kernel x shape pairs: {total_work}")
    print(f"  Already completed: {skipped}")
    print(f"  Pending: {len(work_items)}")
    print(f"  Batch size: {BENCH_BATCH} (retry individually on fault)")
    print()

    def _run_subprocess(payload_bytes, timeout=10):
        proc = subprocess.Popen(
            [sys.executable, str(worker_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=worker_env,
        )
        timed_out = False
        stdout_bytes = b""
        try:
            stdout_bytes, _ = proc.communicate(input=payload_bytes, timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            timed_out = True
        finally:
            pid = proc.pid
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            for pipe in [proc.stdin, proc.stdout, proc.stderr]:
                if pipe and not pipe.closed:
                    pipe.close()
            gpucore = _THIS_DIR / f"gpucore.{pid}"
            if gpucore.exists():
                gpucore.unlink(missing_ok=True)
        rc = -1 if timed_out else proc.returncode
        return stdout_bytes, rc

    def _record_result(r, shape, cfg, shape_dict):
        nonlocal total_measurements
        lat_ms, tflops = r["ms"], r["tflops"]
        bw = bandwidth_gb_s(shape, lat_ms)
        row = {
            "problem_name": shape.name,
            "batch": shape.batch,
            "seqlen_q": shape.seqlen_q,
            "seqlen_k": shape.seqlen_k,
            "nhead_q": shape.nhead_q,
            "nhead_k": shape.nhead_k,
            "hdim_q": shape.hdim_q,
            "hdim_v": shape.hdim_v,
            "dtype": shape.dtype,
        }
        for k in CFG_KEYS:
            row[k] = cfg.get(k, "")
        row["latency_ms"] = round(lat_ms, 4)
        row["tflops"] = round(tflops, 2)
        row["bandwidth_gb_s"] = round(bw, 2)
        writer.writerow(row)
        csv_file.flush()
        total_measurements += 1
        return tflops, lat_ms

    # Process in batches
    n_batches = (len(work_items) + BENCH_BATCH - 1) // BENCH_BATCH
    processed = 0
    for bi in range(n_batches):
        batch = work_items[bi * BENCH_BATCH : (bi + 1) * BENCH_BATCH]

        items = []
        for shape, so_path, cfg in batch:
            cfg["so_path"] = so_path
            items.append(
                {"so_path": so_path, "shape": _shape_to_dict(shape), "cfg": cfg}
            )

        batch_timeout = len(batch) * 2 + 5
        payload = json.dumps({"items": items}).encode()
        stdout_bytes, rc = _run_subprocess(payload, timeout=batch_timeout)

        if rc == 0:
            batch_ok = 0
            for line in stdout_bytes.decode().strip().split("\n"):
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                idx = r.get("idx", -1)
                if not r.get("ok") or idx < 0 or idx >= len(batch):
                    continue
                shape, so_path, cfg = batch[idx]
                _record_result(r, shape, cfg, items[idx]["shape"])
                batch_ok += 1
            processed += len(batch)
            print(
                f"  [batch {bi + 1}/{n_batches}] {batch_ok}/{len(batch)} ok  "
                f"({processed}/{len(work_items)} done, {total_measurements} total)",
                flush=True,
            )
        else:
            # Collect partial results flushed before the fault
            partial_done = set()
            for line in stdout_bytes.decode().strip().split("\n"):
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                idx = r.get("idx", -1)
                if r.get("ok") and 0 <= idx < len(batch):
                    shape, so_path, cfg = batch[idx]
                    _record_result(r, shape, cfg, items[idx]["shape"])
                    partial_done.add(idx)

            # Retry the rest one by one
            retry = [(i, b) for i, b in enumerate(batch) if i not in partial_done]
            print(
                f"  [batch {bi + 1}/{n_batches}] FAULT after {len(partial_done)}/{len(batch)} ok, "
                f"retrying {len(retry)} individually...",
                flush=True,
            )
            for idx, (shape, so_path, cfg) in retry:
                cfg["so_path"] = so_path
                p = json.dumps(
                    {"so_path": so_path, "shape": items[idx]["shape"], "cfg": cfg}
                ).encode()
                out, single_rc = _run_subprocess(p, timeout=10)
                if single_rc != 0:
                    total_gpu_faults += 1
                    continue
                try:
                    r = json.loads(out.decode().strip().split("\n")[0])
                except (json.JSONDecodeError, ValueError):
                    continue
                if r.get("ok"):
                    tflops, lat_ms = _record_result(r, shape, cfg, items[idx]["shape"])
                    print(
                        f"    {tflops:>7.1f} TFLOPS  {lat_ms:.4f}ms  "
                        f"{cfg.get('kernel', '?')[:45]} | {shape.name}",
                        flush=True,
                    )
            processed += len(batch)
            print(f"    ({processed}/{len(work_items)} done)", flush=True)

    csv_file.close()
    bench_time = time.perf_counter() - bench_t0

    # ---- Phase 4: Summary ----
    print(f"\n{'=' * 80}")
    print("Results")
    print(f"{'=' * 80}")
    print(f"  Total work items: {total_work}")
    print(f"  Skipped (resumed): {skipped}")
    print(f"  Measurements: {total_measurements}")
    print(f"  GPU faults: {total_gpu_faults}")
    print(f"  Benchmark time: {bench_time:.1f}s")
    print(f"  CSV: {csv_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
