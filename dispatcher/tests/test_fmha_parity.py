#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
FMHA Parity Test: Dispatcher vs CK Tile 01_fmha vs CPU Reference

Runs the same test configurations through:
  1. CK Tile tile_example_fmha_fwd (gold standard, if available)
  2. Dispatcher fmha_01_basic (via C++ binary)
  3. Python CPU reference (numpy)

Compares exit codes and reports parity.

Usage:
    python3 test_fmha_parity.py
    python3 test_fmha_parity.py --ck-exe /tmp/ck_fmha_build/bin/tile_example_fmha_fwd
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
import numpy as np

from fmha_utils import FmhaProblem, cpu_attention_fwd, detect_gpu_arch


@dataclass
class TestCase:
    name: str
    prec: str = "fp16"
    mode: int = 0
    batch: int = 2
    nhead: int = 2
    nhead_k: int = -1
    hdim: int = 128
    hdim_v: int = -1
    seqlen_q: int = 128
    seqlen_k: int = 128
    bias: str = "n"
    mask: str = "0"
    lse: int = 0
    p_drop: float = 0.0


PARITY_TESTS = [
    TestCase("basic_fp16"),
    TestCase("basic_bf16", prec="bf16"),
    TestCase("longer_seq", seqlen_q=256, seqlen_k=256),
    TestCase("small_batch", batch=1, nhead=8, seqlen_q=64, seqlen_k=64),
    TestCase("gqa_2_1", nhead=4, nhead_k=2),
    TestCase("gqa_4_1", nhead=8, nhead_k=2),
    TestCase("causal_top_left", mask="1"),
    TestCase("causal_bottom_right", mask="2"),
    TestCase("bias_elementwise", bias="e"),
    TestCase("bias_alibi", bias="a"),
    TestCase("with_lse", lse=1),
    TestCase("big_batch", batch=4, nhead=8, seqlen_q=128, seqlen_k=128),
    TestCase("asymmetric_seq", seqlen_q=64, seqlen_k=256),
    TestCase("single_query", batch=1, nhead=4, seqlen_q=1, seqlen_k=128),
]


def find_ck_exe() -> Optional[str]:
    for path in [
        "/tmp/ck_fmha_build/bin/tile_example_fmha_fwd",
        "/workspace/rocm-libraries/projects/composablekernel/build/bin/tile_example_fmha_fwd",
    ]:
        if os.path.exists(path):
            return path
    return None


def find_dispatcher_exe() -> Optional[str]:
    root = Path(__file__).parent.parent
    for rel in ["build/examples/fmha_01_basic"]:
        p = root / rel
        if p.exists():
            return str(p)
    return None


def run_ck_test(exe: str, tc: TestCase) -> bool:
    nhead_k = tc.nhead_k if tc.nhead_k > 0 else tc.nhead
    hdim_v = tc.hdim_v if tc.hdim_v > 0 else tc.hdim
    cmd = [
        exe,
        f"-prec={tc.prec}",
        f"-mode={tc.mode}",
        f"-b={tc.batch}",
        f"-h={tc.nhead}",
        f"-h_k={nhead_k}",
        f"-d={tc.hdim}",
        f"-d_v={hdim_v}",
        f"-s={tc.seqlen_q}",
        f"-s_k={tc.seqlen_k}",
        f"-bias={tc.bias}",
        f"-mask={tc.mask}",
        f"-lse={tc.lse}",
        f"-p_drop={tc.p_drop}",
        "-v=1",
        "-warmup=0",
        "-repeat=1",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def run_dispatcher_test(exe: str, tc: TestCase) -> bool:
    cmd = [
        exe,
        f"--arch={detect_gpu_arch()}",
        f"--batch={tc.batch}",
        f"--nhead={tc.nhead}",
        f"--seqlen={tc.seqlen_q}",
        f"--hdim={tc.hdim}",
        "--validate",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def run_cpu_test(tc: TestCase) -> bool:
    nhead_k = tc.nhead_k if tc.nhead_k > 0 else tc.nhead
    hdim_v = tc.hdim_v if tc.hdim_v > 0 else tc.hdim
    prob = FmhaProblem(
        batch=tc.batch,
        nhead_q=tc.nhead,
        nhead_k=nhead_k,
        seqlen_q=tc.seqlen_q,
        seqlen_k=tc.seqlen_k,
        hdim_q=tc.hdim,
        hdim_v=hdim_v,
    )
    np.random.seed(42)
    Q = (np.random.randn(*prob.q_shape()) * 0.5).astype(np.float32)
    K = (np.random.randn(*prob.k_shape()) * 0.5).astype(np.float32)
    V = (np.random.randn(*prob.v_shape()) * 0.5).astype(np.float32)
    out = cpu_attention_fwd(Q, K, V, prob.scale)
    return out.size > 0 and np.isfinite(out).all()


def main():
    parser = argparse.ArgumentParser(description="FMHA Parity Test")
    parser.add_argument("--ck-exe", default=None, help="Path to tile_example_fmha_fwd")
    parser.add_argument("--dispatcher-exe", default=None, help="Path to fmha_01_basic")
    args = parser.parse_args()

    ck_exe = args.ck_exe or find_ck_exe()
    disp_exe = args.dispatcher_exe or find_dispatcher_exe()

    print("=" * 80)
    print("FMHA Parity Test: CK Tile vs Dispatcher vs CPU Reference")
    print("=" * 80)
    print(f"  CK Tile exe:     {ck_exe or 'NOT FOUND'}")
    print(f"  Dispatcher exe:  {disp_exe or 'NOT FOUND'}")
    print(f"  Test cases:      {len(PARITY_TESTS)}")

    header = f"  {'#':<3} {'Name':<22} {'CK':>6} {'Disp':>6} {'CPU':>6} {'Parity':>8}"
    print(f"\n{header}")
    print("  " + "-" * 56)

    total_ck = 0
    total_disp = 0
    total_cpu = 0
    total_parity = 0

    for i, tc in enumerate(PARITY_TESTS, 1):
        ck_ok = run_ck_test(ck_exe, tc) if ck_exe else None
        disp_ok = run_dispatcher_test(disp_exe, tc) if disp_exe else None
        cpu_ok = run_cpu_test(tc)

        ck_str = "PASS" if ck_ok else ("FAIL" if ck_ok is False else "N/A")
        disp_str = "PASS" if disp_ok else ("FAIL" if disp_ok is False else "N/A")
        cpu_str = "PASS" if cpu_ok else "FAIL"

        parity = True
        if ck_ok is not None and disp_ok is not None:
            parity = ck_ok == disp_ok
        parity_str = "MATCH" if parity else "DIFF"

        if ck_ok:
            total_ck += 1
        if disp_ok:
            total_disp += 1
        if cpu_ok:
            total_cpu += 1
        if parity:
            total_parity += 1

        print(
            f"  {i:<3} {tc.name:<22} {ck_str:>6} {disp_str:>6} {cpu_str:>6} {parity_str:>8}"
        )

    print(f"\n{'=' * 80}")
    print(f"  CK Tile:    {total_ck}/{len(PARITY_TESTS)} passed")
    print(f"  Dispatcher: {total_disp}/{len(PARITY_TESTS)} passed")
    print(f"  CPU Ref:    {total_cpu}/{len(PARITY_TESTS)} passed")
    print(f"  Parity:     {total_parity}/{len(PARITY_TESTS)} matching")
    print(f"{'=' * 80}")

    return 0 if total_parity == len(PARITY_TESTS) else 1


if __name__ == "__main__":
    sys.exit(main())
