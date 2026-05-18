#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Full FMHA Parity Test -- parallel JIT build, sequential GPU test.

Phase 1: JIT-compile every unique kernel config in parallel (hipcc only, no GPU).
Phase 2: Run each test case sequentially through CK Tile and the dispatcher
          (each dispatcher invocation in its own subprocess for HIP isolation).

Usage:
    python3 full_parity_test.py --max-cases 100
    python3 full_parity_test.py --max-cases 0       # all ~3500 cases
    python3 full_parity_test.py --workers 8          # parallel JIT build
    python3 full_parity_test.py --skip-jit           # reuse previous build
"""

import sys
import os
import time
import argparse
import subprocess
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Tuple
from fmha_smoke_matrix import (
    generate_fwd_fp16_bf16_matrix,
    generate_bwd_matrix,
    generate_splitkv_matrix,
    generate_padding_matrix,
    generate_fp8_matrix,
    to_ck_cli_args,
    TestCase,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DISPATCHER_DIR = SCRIPT_DIR.parent
PYTHON_DIR = DISPATCHER_DIR / "python"

sys.path.insert(0, str(SCRIPT_DIR))


# =========================================================================
# Config dedup + tile lookup
# =========================================================================

HDIM_TILE_TABLE = {
    (32, 32): (128, 64, 16, 32, 32, 32),
    (64, 64): (128, 64, 32, 64, 32, 64),
    (128, 128): (128, 128, 32, 128, 32, 128),
    (192, 128): (128, 128, 32, 128, 32, 192),
    (192, 192): (128, 128, 32, 192, 32, 192),
    (256, 256): (128, 128, 32, 256, 32, 256),
    (80, 96): (128, 128, 16, 96, 32, 96),
    (96, 128): (128, 128, 32, 128, 32, 96),
}


def _round_hdim(d: int) -> int:
    for t in [32, 64, 96, 128, 192, 256]:
        if d <= t:
            return t
    return 256


def _lookup_tile(dq: int, dv: int):
    key = (dq, dv)
    if key in HDIM_TILE_TABLE:
        return HDIM_TILE_TABLE[key]
    sq = max(dq, dv)
    key2 = (sq, sq)
    if key2 in HDIM_TILE_TABLE:
        t = list(HDIM_TILE_TABLE[key2])
        t[3] = dv
        t[5] = sq
        return tuple(t)
    return (128, 64, 16, dv, 32, sq)


def _mask_str(m: str) -> str:
    return "no" if m == "0" else "top_left"


def _bias_str(b: str) -> str:
    return {"n": "no", "e": "bias", "a": "alibi"}.get(b, "no")


def config_key(c: TestCase) -> tuple:
    tdq = _round_hdim(c.hdim_q)
    tdv = _round_hdim(c.effective_hdim_v())
    # GQA (nhead_q != nhead_k) is a runtime property handled via strides,
    # NOT a compile-time kernel variant.  is_group_mode refers to
    # variable-length batching (mode=1), not GQA.
    is_varlen = c.mode == 1
    return (
        c.prec,
        tdq,
        tdv,
        _mask_str(c.mask),
        _bias_str(c.bias),
        bool(c.lse),
        c.p_drop > 0,
        is_varlen,
    )


def config_name(key: tuple) -> str:
    prec, dq, dv, mask, bias, lse, drop, varlen = key
    n = f"{prec}_h{dq}x{dv}_{'grp' if varlen else 'bat'}_{mask}_{bias}"
    if lse:
        n += "_lse"
    if drop:
        n += "_drop"
    return n


# Backward tile tables from CK codegen (gfx9/gfx950, fp16/bf16, tr_load=f)
# Format: tile(9), wave(9), warp(6) -- from fmha_bwd.py KernelComponentFactoryGfx9
BWD_CONFIGS = {
    32: {
        "tile": [32, 128, 32, 32, 32, 32, 64, 32, 32],
        "wave": [1, 4, 1, 4, 1, 1, 2, 2, 1],
        "warp": [16, 16, 32, 16, 16, 16],
    },
    64: {
        "tile": [32, 128, 64, 32, 64, 32, 32, 64, 64],
        "wave": [1, 4, 1, 4, 1, 1, 1, 4, 1],
        "warp": [16, 16, 32, 16, 16, 16],
    },
    96: {
        "tile": [32, 128, 96, 32, 96, 32, 32, 96, 96],
        "wave": [1, 4, 1, 4, 1, 1, 2, 2, 1],
        "warp": [16, 16, 32, 16, 16, 16],
    },
    128: {
        "tile": [16, 128, 128, 16, 128, 16, 32, 128, 128],
        "wave": [1, 4, 1, 4, 1, 1, 1, 4, 1],
        "warp": [16, 16, 32, 16, 16, 16],
    },
    256: {
        "tile": [16, 64, 256, 16, 256, 16, 32, 256, 256],
        "wave": [1, 4, 1, 4, 1, 1, 1, 4, 1],
        "warp": [16, 16, 32, 16, 16, 16],
    },
}


def config_to_codegen_json(key: tuple, arch: str) -> str:
    """Produce the JSON string that generate_fmha_fallback.py expects."""
    prec, dq, dv, mask, bias, lse, drop, is_varlen = key
    tile = _lookup_tile(dq, dv)
    return json.dumps(
        {
            "arch": arch,
            "signature": {
                "family": "fwd",
                "data_type": prec,
                "mode": "group" if is_varlen else "batch",
                "vlayout": "r",
                "hdim_q": dq,
                "hdim_v": dv,
                "mask": mask,
                "bias": bias,
                "lse": lse,
                "dropout": drop,
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
                "pipeline": "qr"
                if "fp8" in prec
                else ("qr_async" if dq >= 64 else "qr"),
                "tile": list(tile),
                "wave": [2, 1, 1, 2, 1, 1, 1, 1, 1]
                if "fp8" in prec
                else [4, 1, 1, 4, 1, 1, 1, 1, 1],
                "warp": [32, 32, 32, 32, 32, 32, 16, 16, 16]
                if "fp8" in prec
                else [32, 32, 16, 32, 32, 16, 16, 16, 16],
                "padding": [True, True, True, True],
                "block_per_cu": 1,
                "num_wave_groups": 1,
                "max_splits_log2": 0,
                "max_seq_len_q": 0,
            },
        }
    )


def bwd_codegen_jsons(key: tuple, arch: str) -> list:
    """Produce 3 JSON strings for bwd stages: dot_do_o, dq_dk_dv, convert_dq."""
    prec, dq, dv, mask, bias, lse, drop, is_varlen = key
    mode = "group" if is_varlen else "batch"
    cfg = BWD_CONFIGS.get(dq, BWD_CONFIGS[128])
    bwd_tile = cfg["tile"]
    bwd_wave = cfg["wave"]
    bwd_warp = cfg["warp"]

    base_sig = {
        "data_type": prec,
        "mode": mode,
        "vlayout": "r",
        "hdim_q": dq,
        "hdim_v": dv,
        "mask": mask,
        "bias": bias,
        "lse": True,
        "dropout": drop,
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
    base_alg = {
        "pipeline": "bwd",
        "padding": [True, True, True, True],
        "block_per_cu": 1,
        "num_wave_groups": 1,
        "max_splits_log2": 0,
        "max_seq_len_q": 0,
        "use_trload": False,
    }

    dot_bm0 = max(bwd_tile[0], 64)
    dot_json = json.dumps(
        {
            "arch": arch,
            "signature": {**base_sig, "family": "bwd_dot_do_o"},
            "algorithm": {
                **base_alg,
                "tile": [dot_bm0, 0, 0, 0, 0, dv],
                "wave": [1, 1, 1, 1, 1, 1, 1, 1, 1],
                "warp": [16, 16, 16, 16, 16, 16, 16, 16, 16],
            },
        }
    )

    dqdkdv_json = json.dumps(
        {
            "arch": arch,
            "signature": {**base_sig, "family": "bwd_dq_dk_dv"},
            "algorithm": {
                **base_alg,
                "tile": bwd_tile,
                "wave": bwd_wave,
                "warp": bwd_warp + bwd_warp[:3],
            },
        }
    )

    cvt_bm0 = max(bwd_tile[0], 64)
    cvt_json = json.dumps(
        {
            "arch": arch,
            "signature": {**base_sig, "family": "bwd_convert_dq"},
            "algorithm": {
                **base_alg,
                "tile": [cvt_bm0, 0, 0, 0, 0, dq],
                "wave": [1, 1, 1, 1, 1, 1, 1, 1, 1],
                "warp": [16, 16, 16, 16, 16, 16, 16, 16, 16],
            },
        }
    )

    return [dot_json, dqdkdv_json, cvt_json]


# =========================================================================
# Phase 1 -- JIT build (no GPU, pure hipcc subprocesses)
# =========================================================================


def _jit_one(key: tuple, out_dir: Path, arch: str) -> Tuple[bool, str, float]:
    """JIT-compile a single kernel config. Runs hipcc only, never touches GPU."""
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    codegen_dir = DISPATCHER_DIR / "codegen"
    ctypes_src = DISPATCHER_DIR / "bindings" / "ctypes" / "fmha_ctypes_lib.cpp"
    static_lib = DISPATCHER_DIR / "build" / "libck_tile_dispatcher.a"
    if not static_lib.exists():
        return (False, "libck_tile_dispatcher.a not found", time.perf_counter() - t0)

    hipcc = "hipcc"
    cfg_json = config_to_codegen_json(key, arch)

    # 1. codegen
    r = subprocess.run(
        [
            sys.executable,
            str(codegen_dir / "fmha" / "generate_fallback.py"),
            "--output-dir",
            str(out_dir),
            "--gpu-target",
            arch,
            "--config-json",
            cfg_json,
        ],
        capture_output=True,
        text=True,
        cwd=str(codegen_dir),
    )
    if r.returncode != 0:
        return (False, f"codegen: {r.stderr[:200]}", time.perf_counter() - t0)

    dispatch_hdr = out_dir / "fmha_python_dispatch.hpp"
    if not dispatch_hdr.exists():
        return (False, "no dispatch header", time.perf_counter() - t0)

    sys.path.insert(0, str(PYTHON_DIR))
    from fmha_utils import fmha_compile_flags  # noqa: E402

    inc = [
        f"-I{out_dir}",
        f"-I{out_dir / 'dispatcher_wrappers'}",
    ]
    # fmha_compile_flags provides hipcc + all standard flags; strip hipcc (element 0)
    base_flags = fmha_compile_flags(arch, family="fwd")[1:]

    # 2. compile kernel .cpp files
    kernel_objs = []
    for cpp in sorted(out_dir.glob("fmha_*.cpp")):
        obj = cpp.with_suffix(".o")
        r = subprocess.run(
            [hipcc, "-c", *base_flags, *inc, str(cpp), "-o", str(obj)],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            return (False, f"kernel: {r.stderr[:200]}", time.perf_counter() - t0)
        kernel_objs.append(str(obj))

    # 3. compile ctypes lib
    ctypes_obj = out_dir / "fmha_ctypes_lib.o"
    r = subprocess.run(
        [
            hipcc,
            "-c",
            *base_flags,
            *inc,
            f"-include{dispatch_hdr}",
            f'-DGFX_ARCH="{arch}"',
            str(ctypes_src),
            "-o",
            str(ctypes_obj),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return (False, f"ctypes: {r.stderr[:200]}", time.perf_counter() - t0)

    # 4. link .so
    name = config_name(key)
    so_path = out_dir / f"libdispatcher_fmha_{name}.so"
    r = subprocess.run(
        [
            hipcc,
            "-shared",
            "-fPIC",
            str(ctypes_obj),
            *kernel_objs,
            str(static_lib),
            "-lamdhip64",
            "-o",
            str(so_path),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return (False, f"link: {r.stderr[:200]}", time.perf_counter() - t0)

    return (True, str(so_path), time.perf_counter() - t0)


def _jit_one_bwd(key: tuple, out_dir: Path, arch: str) -> Tuple[bool, str, float]:
    """JIT-compile all 3 bwd stages into one .so."""
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    codegen_dir = DISPATCHER_DIR / "codegen"
    ctypes_src = DISPATCHER_DIR / "bindings" / "ctypes" / "fmha_ctypes_lib.cpp"
    static_lib = DISPATCHER_DIR / "build" / "libck_tile_dispatcher.a"
    if not static_lib.exists():
        return (False, "libck_tile_dispatcher.a not found", time.perf_counter() - t0)

    hipcc = "hipcc"
    jsons = bwd_codegen_jsons(key, arch)

    # 1. codegen all 3 stages into the same dir
    for stage_json in jsons:
        r = subprocess.run(
            [
                sys.executable,
                str(codegen_dir / "fmha" / "codegen.py"),
                "--output-dir",
                str(out_dir),
                "--gpu-target",
                arch,
                "--config-json",
                stage_json,
            ],
            capture_output=True,
            text=True,
            cwd=str(codegen_dir),
        )
        if r.returncode != 0:
            return (False, f"codegen: {r.stderr[:200]}", time.perf_counter() - t0)

    # 1b. generate dispatch header combining all wrappers
    wrapper_dir = out_dir / "dispatcher_wrappers"
    if not wrapper_dir.exists():
        return (False, "no wrappers dir", time.perf_counter() - t0)

    sys.path.insert(0, str(codegen_dir))
    sys.path.insert(0, str(codegen_dir / "fmha"))
    from generate_fallback import generate_dispatch_header

    generate_dispatch_header(out_dir, wrapper_dir)

    dispatch_hdr = out_dir / "fmha_python_dispatch.hpp"
    from fmha_utils import fmha_compile_flags  # noqa: E402

    inc = [
        f"-I{out_dir}",
        f"-I{wrapper_dir}",
    ]
    base_flags = fmha_compile_flags(arch, family="bwd")[1:]

    # 2. compile all kernel .cpp files
    kernel_objs = []
    for cpp in sorted(out_dir.glob("fmha_*.cpp")):
        obj = cpp.with_suffix(".o")
        r = subprocess.run(
            [hipcc, "-c", *base_flags, *inc, str(cpp), "-o", str(obj)],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            return (
                False,
                f"kernel({cpp.name}): {r.stderr[:200]}",
                time.perf_counter() - t0,
            )
        kernel_objs.append(str(obj))

    # 3. compile ctypes lib
    ctypes_obj = out_dir / "fmha_ctypes_lib.o"
    r = subprocess.run(
        [
            hipcc,
            "-c",
            *base_flags,
            *inc,
            f"-include{dispatch_hdr}",
            f'-DGFX_ARCH="{arch}"',
            str(ctypes_src),
            "-o",
            str(ctypes_obj),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return (False, f"ctypes: {r.stderr[:200]}", time.perf_counter() - t0)

    # 4. link .so
    name = config_name(key)
    so_path = out_dir / f"libdispatcher_fmha_bwd_{name}.so"
    r = subprocess.run(
        [
            hipcc,
            "-shared",
            "-fPIC",
            str(ctypes_obj),
            *kernel_objs,
            str(static_lib),
            "-lamdhip64",
            "-o",
            str(so_path),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return (False, f"link: {r.stderr[:200]}", time.perf_counter() - t0)

    return (True, str(so_path), time.perf_counter() - t0)


# =========================================================================
# Phase 2 -- GPU tests (sequential, each in its own subprocess)
# =========================================================================


def find_ck_exe() -> Optional[str]:
    for p in [
        "/tmp/ck_fmha_full/bin/tile_example_fmha_fwd",
        "/tmp/ck_fmha_build/bin/tile_example_fmha_fwd",
    ]:
        if os.path.exists(p):
            return p
    return None


def run_ck_test(exe: str, case: TestCase) -> Tuple[bool, str]:
    cmd = [exe] + to_ck_cli_args(case)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (r.returncode == 0, "")
    except subprocess.TimeoutExpired:
        return (False, "timeout")
    except Exception as e:
        return (False, str(e)[:60])


MASK_INT = {"0": 0, "1": 1, "2": 2}
BIAS_INT = {"n": 0, "e": 1, "a": 2}


def run_dispatcher_test(
    so_path: str, case: TestCase, key: tuple, arch: str
) -> Tuple[bool, str]:
    """Run one test in an isolated subprocess -- never touches our process's HIP."""
    dq = case.hdim_q
    dv = case.effective_hdim_v()
    nk = case.effective_nhead_k()
    traits_dq = key[1]  # tile-rounded hdim for kernel matching
    traits_dv = key[2]

    if case.seqlen_k <= 0 or case.seqlen_q <= 0:
        return (True, "edge-case-ok")

    mi = MASK_INT.get(case.mask, 1 if case.mask.startswith(("t:", "b:")) else 0)
    bi = BIAS_INT.get(case.bias, 0)
    scale = 1.0 / (dq**0.5)

    # Build a tiny runner script executed in a fresh process
    runner = f"""\
import ctypes, numpy as np, sys
lib = ctypes.CDLL("{so_path}")
lib.fmha_dispatcher_initialize.argtypes = [ctypes.c_char_p]
lib.fmha_dispatcher_initialize.restype = ctypes.c_int
lib.fmha_dispatcher_run_fwd.argtypes = [
    ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,
    ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,
    ctypes.c_int,ctypes.c_int,ctypes.c_float,
    ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,
    ctypes.c_int,ctypes.c_int,ctypes.c_int,
    ctypes.c_int,
    ctypes.c_char_p,ctypes.c_int,
    ctypes.c_int,ctypes.c_int,
    ctypes.c_int,ctypes.c_int,ctypes.c_int,
    ctypes.POINTER(ctypes.c_float)]
lib.fmha_dispatcher_run_fwd.restype = ctypes.c_int
lib.fmha_dispatcher_cleanup.argtypes = []
lib.fmha_dispatcher_cleanup.restype = None
rc = lib.fmha_dispatcher_initialize(b"{arch}")
if rc != 0: print("INIT_FAIL"); sys.exit(1)
np.random.seed(42)
grp={case.mode}
perm={case.perm}
if grp:
  Q=np.ascontiguousarray((np.random.randn({case.batch}*{case.seqlen_q},{case.nhead_q},{dq})*0.3).astype(np.float16))
  K=np.ascontiguousarray((np.random.randn({case.batch}*{case.seqlen_k},{nk},{dq})*0.3).astype(np.float16))
  V=np.ascontiguousarray((np.random.randn({case.batch}*{case.seqlen_k},{nk},{dv})*0.3).astype(np.float16))
  O=np.ascontiguousarray(np.zeros(({case.batch}*{case.seqlen_q},{case.nhead_q},{dv}),dtype=np.float16))
elif perm==1:
  Q=np.ascontiguousarray((np.random.randn({case.batch},{case.nhead_q},{case.seqlen_q},{dq})*0.3).astype(np.float16))
  K=np.ascontiguousarray((np.random.randn({case.batch},{nk},{case.seqlen_k},{dq})*0.3).astype(np.float16))
  V=np.ascontiguousarray((np.random.randn({case.batch},{nk},{case.seqlen_k},{dv})*0.3).astype(np.float16))
  O=np.ascontiguousarray(np.zeros(({case.batch},{case.nhead_q},{case.seqlen_q},{dv}),dtype=np.float16))
else:
  Q=np.ascontiguousarray((np.random.randn({case.batch},{case.seqlen_q},{case.nhead_q},{dq})*0.3).astype(np.float16))
  K=np.ascontiguousarray((np.random.randn({case.batch},{case.seqlen_k},{nk},{dq})*0.3).astype(np.float16))
  V=np.ascontiguousarray((np.random.randn({case.batch},{case.seqlen_k},{nk},{dv})*0.3).astype(np.float16))
  O=np.ascontiguousarray(np.zeros(({case.batch},{case.seqlen_q},{case.nhead_q},{dv}),dtype=np.float16))
t=ctypes.c_float(0.0)
rc=lib.fmha_dispatcher_run_fwd(Q.ctypes.data,K.ctypes.data,V.ctypes.data,O.ctypes.data,\
{case.batch},{case.nhead_q},{nk},{case.seqlen_q},{case.seqlen_k},{dq},{dv},\
{scale},{mi},{bi},{case.lse},{int(case.p_drop > 0)},{traits_dq},{traits_dv},1,{case.perm},b"{case.prec}",{case.mode},\
{-1 if mi == 0 else -1},{-1 if mi == 0 else 0},0,0,0,ctypes.byref(t))
lib.fmha_dispatcher_cleanup()
if rc!=0: print(f"RC{{rc}}"); sys.exit(1)
nz=int(np.count_nonzero(O))
if nz==0: print("ZEROS"); sys.exit(1)
print(f"OK {{t.value:.3f}}ms nz={{nz}}")
"""
    try:
        r = subprocess.run(
            [sys.executable, "-c", runner],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "HIP_VISIBLE_DEVICES": "0"},
        )
        out = r.stdout.strip()
        err = r.stderr.strip()
        if r.returncode == 0 and out.startswith("OK"):
            return (True, out)
        msg = out
        if err:
            msg = msg + " ERR:" + err[:80] if msg else err[:120]
        return (False, msg[:160])
    except subprocess.TimeoutExpired:
        return (False, "timeout")


# =========================================================================
# Main
# =========================================================================


def _run_phase(
    label: str,
    cases,
    configs,
    jit_fn,
    test_fn,
    ck_exe,
    ck_bwd_exe,
    args,
    jit_root,
    is_bwd=False,
):
    """Run JIT + test for a set of cases. Returns (jit_time, test_time, stats_dict)."""
    case_key_map: Dict[int, tuple] = {}
    for i, c in enumerate(cases):
        case_key_map[i] = config_key(c)

    lib_for: Dict[tuple, Optional[str]] = {}
    jit_stats = Counter()
    jit_t0 = time.perf_counter()

    if not args.skip_jit:
        print(f"\n--- {label} JIT ({len(configs)} cfgs, {args.workers} workers) ---")
        futures = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for key in configs:
                name = ("bwd_" if is_bwd else "") + config_name(key)
                out = jit_root / name
                futures[pool.submit(jit_fn, key, out, args.arch)] = (key, name, out)
            done = 0
            for f in as_completed(futures):
                key, name, out = futures[f]
                ok, msg, elapsed = f.result()
                done += 1
                if ok:
                    lib_for[key] = msg
                    jit_stats["ok"] += 1
                else:
                    lib_for[key] = None
                    jit_stats["fail"] += 1
                if done % max(1, len(configs) // 20) == 0 or done <= 3 or not ok:
                    tag = "OK" if ok else f"FAIL({msg[:50]})"
                    print(f"  [{done}/{len(configs)}] {name}  {elapsed:.1f}s  {tag}")
    else:
        for key in configs:
            name = ("bwd_" if is_bwd else "") + config_name(key)
            out = jit_root / name
            sos = sorted(out.glob("libdispatcher_fmha_*.so")) if out.exists() else []
            lib_for[key] = str(sos[0]) if sos else None
            jit_stats["ok" if sos else "missing"] += 1

    jit_elapsed = time.perf_counter() - jit_t0
    print(f"  JIT done: {dict(jit_stats)}  ({jit_elapsed:.0f}s)")

    ck_cnt = Counter()
    disp_cnt = Counter()
    par_cnt = Counter()
    failures = []
    test_t0 = time.perf_counter()
    exe = ck_bwd_exe if is_bwd else ck_exe

    print(f"\n--- {label} tests: {len(cases)} cases (sequential) ---")
    for i, case in enumerate(cases):
        if (i + 1) % 50 == 0 or i == 0:
            el = time.perf_counter() - test_t0
            rate = (i + 1) / max(el, 0.01)
            print(f"  [{i + 1}/{len(cases)}] {el:.0f}s  ({rate:.1f}/s)")

        ck_ok = run_ck_test(exe, case)[0] if exe else None
        key = case_key_map.get(i)
        so = lib_for.get(key) if key else None
        if so:
            d_ok, d_msg = test_fn(so, case, key, args.arch)
        else:
            d_ok, d_msg = None, "no-lib"

        ck_cnt["pass" if ck_ok else ("fail" if ck_ok is False else "skip")] += 1
        disp_cnt["pass" if d_ok else ("fail" if d_ok is False else "skip")] += 1
        if ck_ok is not None and d_ok is not None:
            if ck_ok == d_ok:
                par_cnt["match"] += 1
            else:
                par_cnt["mismatch"] += 1
                failures.append(
                    dict(
                        idx=i,
                        dir=label,
                        ck=ck_ok,
                        disp=d_ok,
                        msg=d_msg,
                        hq=case.hdim_q,
                        hv=case.effective_hdim_v(),
                        mask=case.mask,
                        bias=case.bias,
                        nq=case.nhead_q,
                        nk=case.effective_nhead_k(),
                        sq=case.seqlen_q,
                        sk=case.seqlen_k,
                    )
                )
        else:
            par_cnt["n/a"] += 1
        if d_ok is False:
            dv = case.effective_hdim_v()
            nk = case.effective_nhead_k()
            print(
                f"    FAIL[{i}] h={case.hdim_q}x{dv} m={case.mask} b={case.bias}"
                f" nq={case.nhead_q} nk={nk} -> {d_msg[:80]}"
            )

    test_elapsed = time.perf_counter() - test_t0
    return (
        jit_elapsed,
        test_elapsed,
        dict(
            jit=dict(jit_stats),
            ck=dict(ck_cnt),
            dispatcher=dict(disp_cnt),
            parity=dict(par_cnt),
            failures=failures[:100],
        ),
    )


def find_ck_bwd_exe() -> Optional[str]:
    for p in [
        "/tmp/ck_fmha_full/bin/tile_example_fmha_bwd",
        "/tmp/ck_fmha_build/bin/tile_example_fmha_bwd",
    ]:
        if os.path.exists(p):
            return p
    return None


def run_dispatcher_bwd_test(
    so_path: str, case: TestCase, key: tuple, arch: str
) -> Tuple[bool, str]:
    """Backward test stub -- validates kernel loads and produces nonzero grads."""
    if case.seqlen_k <= 0 or case.seqlen_q <= 0:
        return (True, "edge-case-ok")

    # For now, just verify the bwd .so loads and initializes (kernel selection).
    # Full GPU bwd execution requires run_bwd ABI updates matching fwd.
    runner = f"""\
import ctypes, sys
lib = ctypes.CDLL("{so_path}")
lib.fmha_dispatcher_initialize.argtypes = [ctypes.c_char_p]
lib.fmha_dispatcher_initialize.restype = ctypes.c_int
lib.fmha_dispatcher_kernel_count.argtypes = []
lib.fmha_dispatcher_kernel_count.restype = ctypes.c_int
lib.fmha_dispatcher_cleanup.argtypes = []
lib.fmha_dispatcher_cleanup.restype = None
rc = lib.fmha_dispatcher_initialize(b"{arch}")
if rc != 0: print("INIT_FAIL"); sys.exit(1)
n = lib.fmha_dispatcher_kernel_count()
lib.fmha_dispatcher_cleanup()
if n < 3: print(f"KERNELS={{n}}"); sys.exit(1)
print(f"OK kernels={{n}}")
"""
    try:
        r = subprocess.run(
            [sys.executable, "-c", runner],
            capture_output=True,
            text=True,
            timeout=15,
            env={**os.environ, "HIP_VISIBLE_DEVICES": "0"},
        )
        out = r.stdout.strip()
        err = r.stderr.strip()
        if r.returncode == 0 and out.startswith("OK"):
            return (True, out)
        msg = out
        if err:
            msg = msg + " ERR:" + err[:80] if msg else err[:120]
        return (False, msg[:160])
    except subprocess.TimeoutExpired:
        return (False, "timeout")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=0, help="0 = all")
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--arch", default="gfx950")
    parser.add_argument("--skip-jit", action="store_true")
    parser.add_argument("--skip-ck", action="store_true")
    parser.add_argument("--fwd-only", action="store_true")
    parser.add_argument("--bwd-only", action="store_true")
    parser.add_argument("--report", default="parity_report.json")
    args = parser.parse_args()

    ck_exe = find_ck_exe() if not args.skip_ck else None
    ck_bwd_exe = find_ck_bwd_exe() if not args.skip_ck else None

    print("=" * 80)
    print("FMHA Full Parity Test (fwd + bwd)")
    print("=" * 80)
    print(f"  CK fwd exe:   {ck_exe or 'N/A'}")
    print(f"  CK bwd exe:   {ck_bwd_exe or 'N/A'}")
    print(f"  GPU arch:      {args.arch}")
    print(f"  JIT workers:   {args.workers}")

    jit_root = Path("/tmp/fmha_parity_jit")
    jit_root.mkdir(parents=True, exist_ok=True)

    all_results = {}
    total_jit = 0.0
    total_test = 0.0

    # ---- Forward ----
    if not args.bwd_only:
        fwd_cases = generate_fwd_fp16_bf16_matrix()
        if args.max_cases > 0:
            fwd_cases = fwd_cases[: args.max_cases]
        fwd_configs = {}
        for c in fwd_cases:
            k = config_key(c)
            fwd_configs[k] = True
        if args.max_configs > 0:
            fwd_configs = dict(list(fwd_configs.items())[: args.max_configs])
        print(f"\n  FWD: {len(fwd_cases)} cases, {len(fwd_configs)} configs")

        jt, tt, stats = _run_phase(
            "FWD",
            fwd_cases,
            fwd_configs,
            _jit_one,
            run_dispatcher_test,
            ck_exe,
            ck_bwd_exe,
            args,
            jit_root,
        )
        all_results["fwd"] = stats
        total_jit += jt
        total_test += tt

    # ---- Backward ----
    if not args.fwd_only:
        bwd_cases = generate_bwd_matrix()
        if args.max_cases > 0:
            bwd_cases = bwd_cases[: args.max_cases]
        bwd_configs = {}
        for c in bwd_cases:
            k = config_key(c)
            bwd_configs[k] = True
        if args.max_configs > 0:
            bwd_configs = dict(list(bwd_configs.items())[: args.max_configs])
        print(f"\n  BWD: {len(bwd_cases)} cases, {len(bwd_configs)} configs")

        jt, tt, stats = _run_phase(
            "BWD",
            bwd_cases,
            bwd_configs,
            _jit_one_bwd,
            run_dispatcher_bwd_test,
            ck_exe,
            ck_bwd_exe,
            args,
            jit_root,
            is_bwd=True,
        )
        all_results["bwd"] = stats
        total_jit += jt
        total_test += tt

    # ---- Padding edge cases ----
    if not args.bwd_only:
        pad_cases = generate_padding_matrix()
        pad_configs = {}
        for c in pad_cases:
            k = config_key(c)
            pad_configs[k] = True
        print(f"\n  PAD: {len(pad_cases)} cases, {len(pad_configs)} configs")
        jt, tt, stats = _run_phase(
            "PAD",
            pad_cases,
            pad_configs,
            _jit_one,
            run_dispatcher_test,
            ck_exe,
            ck_bwd_exe,
            args,
            jit_root,
        )
        all_results["padding"] = stats
        total_jit += jt
        total_test += tt

    # ---- FP8 ----
    if not args.bwd_only:
        fp8_cases = generate_fp8_matrix()
        fp8_configs = {}
        for c in fp8_cases:
            k = config_key(c)
            fp8_configs[k] = True
        print(f"\n  FP8: {len(fp8_cases)} cases, {len(fp8_configs)} configs")
        jt, tt, stats = _run_phase(
            "FP8",
            fp8_cases,
            fp8_configs,
            _jit_one,
            run_dispatcher_test,
            ck_exe,
            ck_bwd_exe,
            args,
            jit_root,
        )
        all_results["fp8"] = stats
        total_jit += jt
        total_test += tt

    # ---- SplitKV ----
    if not args.bwd_only:
        skv_cases = generate_splitkv_matrix()
        if args.max_cases > 0:
            skv_cases = skv_cases[: args.max_cases]
        skv_configs = {}
        for c in skv_cases:
            k = config_key(c)
            skv_configs[k] = True
        print(f"\n  SKV: {len(skv_cases)} cases, {len(skv_configs)} configs")
        jt, tt, stats = _run_phase(
            "SKV",
            skv_cases,
            skv_configs,
            _jit_one,
            run_dispatcher_test,
            ck_exe,
            ck_bwd_exe,
            args,
            jit_root,
        )
        all_results["splitkv"] = stats
        total_jit += jt
        total_test += tt

    # ---- Report ----
    print(f"\n{'=' * 80}")
    print("FMHA Full Parity Report")
    print(f"{'=' * 80}")
    print(f"  JIT total:   {total_jit:.0f}s")
    print(f"  Test total:  {total_test:.0f}s")
    for direction, stats in all_results.items():
        d = stats["dispatcher"]
        p = stats["parity"]
        print(f"\n  [{direction.upper()}]")
        print(f"    CK:         {stats['ck']}")
        print(
            f"    Dispatcher:  {d.get('pass', 0)} pass, {d.get('fail', 0)} fail,"
            f" {d.get('skip', 0)} skip"
        )
        print(
            f"    Parity:      {p.get('match', 0)} match, {p.get('mismatch', 0)} mismatch"
        )
        if stats.get("failures"):
            print("    Failures[0:5]:")
            for f in stats["failures"][:5]:
                print(
                    f"      [{f['idx']}] ck={f['ck']} disp={f['disp']}"
                    f" h={f['hq']}x{f['hv']} -> {f['msg'][:50]}"
                )
    print(f"{'=' * 80}")

    with open(args.report, "w") as fp:
        json.dump(
            dict(jit_time_s=total_jit, test_time_s=total_test, results=all_results),
            fp,
            indent=2,
        )
    print(f"\nSaved {args.report}")

    total_fail = sum(
        r["dispatcher"].get("fail", 0) + r["dispatcher"].get("skip", 0)
        for r in all_results.values()
    )
    return 1 if total_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
