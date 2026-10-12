#!/usr/bin/env python3
"""Regression test for the gfx1250 Tdm forward pipelines (hdim 128).

hstu_attention_{no,with}_softmax_fwd_tdm_pipeline.hpp keep two Lds copies of K and V and
alternate between them from one kv-tile to the next, so mistakes in that bookkeeping only
show up for particular kv-tile counts (one, odd, even, none at all in an empty split-kv
range). The cases below vary those counts on all four Tdm kernel variants (MTile 64/128,
plain/split-kv), with and without softmax:

    test_tdm_pipeline.py [--exe PATH] [--no-kernel-check] [--list] [-k SUBSTRING]

Pass criterion. The gfx1250 baseline does not pass the example's own validation everywhere:
with softmax up to ~4% of the outputs miss its tolerance, by at most 2e-3. So a case passes
if its max error stays within MAX_ERR. The share of wrong values is no use as a criterion:
a pipeline reading V from the wrong copy got only 3.6% wrong on a split-kv case, about what
the baseline gets. Measured with deliberately broken pipelines (K or V read from the wrong
copy, the loop not stopping after an odd number of kv-tiles, a single Lds copy allocated):
every one of them failed here with a max error of at least 2e-2.

Kernel check. With rocprofv3 on PATH every case runs under a kernel trace, which must show
only Tdm kernels; the variants seen are tallied and every one of them must be hit at least
once, so a dispatch change that silently stops using Tdm fails the test instead of passing it.
"""

import argparse
import csv
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

BUILD = "build"
EXE = f"{BUILD}/bin/tile_example_hstu_attention_fwd"

# Between the baseline's worst max error (2e-3) and the smallest one of a broken pipeline
# (2e-2), see the docstring.
MAX_ERR = 5e-3

# kN0 of the Tdm tiles is 32 (MTile 64) and 64 (MTile 128); these lengths give 1, 2, 3, ...
# kv-tiles, both odd and even, on either tile.
SEQLENS_M64 = "1,31,32,33,64,65,96,127,128,129,192,300"
SEQLENS_M128 = "1,63,64,65,128,129,191,192,255,256,320,383,384,511,512,640"

CASES = [
    # name, extra args; every case also gets the common args in run_case()
    ("m64_nomask", dict(jagged=1, b=12, seqlens=SEQLENS_M64, causal=0)),
    ("m64_causal", dict(jagged=1, b=12, seqlens=SEQLENS_M64, causal=1)),
    ("m64_causal_local_context_target",
     dict(jagged=1, b=12, seqlens=SEQLENS_M64, causal=1, local_len=5, context_len=8,
          minfull_len=7, targets=8)),
    ("m128_nomask", dict(jagged=1, b=16, nhead=8, seqlens=SEQLENS_M128, causal=0)),
    ("m128_causal", dict(jagged=1, b=16, nhead=8, seqlens=SEQLENS_M128, causal=1)),
    ("m128_causal_local_context_target",
     dict(jagged=1, b=16, nhead=8, seqlens=SEQLENS_M128, causal=1, local_len=5, context_len=8,
          minfull_len=7, targets=8)),
    # a single kv-tile per workgroup, and empty kv ranges once split-kv kicks in
    ("single_tile_batched", dict(jagged=0, b=8, seqlens=20, causal=1)),
    ("batched_causal", dict(jagged=0, b=4, seqlens=1000, causal=1)),
    # cross attention, seqlen_kv >> seqlen_q, goes to MTile 128 and split-kv
    ("cross_attention", dict(jagged=1, b=2, seqlens="128,100", seqlens_kv="1000,1100",
                             causal=0)),
    ("group", dict(jagged=1, g=2, b=8, seqlens="100,200,300,400,500,600,700,800",
                   g_max_seqlens="400,800", g_local_lens="0,0", g_context_lens="0,0",
                   g_minfull_lens="0,0", g_attn_scales="0,0", causal=1)),
]


def kernel_variants(trace_dir):
    """Return the set of HSTU fwd kernel variants found in a rocprofv3 kernel trace."""
    found = set()
    for path in glob.glob(os.path.join(trace_dir, "**", "*kernel_trace.csv"), recursive=True):
        with open(path) as f:
            for row in csv.DictReader(f):
                name = row["Kernel_Name"]
                if "HstuAttentionFwd" not in name or "Combine" in name:
                    continue
                pipeline = "Tdm" if "PipelineQRKSVSTdm" in name else "NOT-Tdm"
                softmax = "softmax" if "WithSoftmaxFwdPipeline" in name else "silu"
                split = "splitkv" if "SplitKVKernel" in name else "plain"
                m = re.search(r"TileSettingClass<ck_tile::sequence<(\d+),", name)
                mtile = f"M{m.group(1)}" if m else "M?"
                found.add((pipeline, softmax, split, mtile))
    return found


def run_case(exe, kernel_check, name, prec, softmax, training, disable_splitkv, case):
    args = dict(nhead=4, hdim_qk=128, hdim_v=128, causal=1, local_len=0, context_len=0,
                minfull_len=0, targets=0)
    args.update(case)
    cmd = [exe, "-v=1", f"-prec={prec}", f"-softmax={softmax}", f"-training={training}"]
    cmd += [f"-{k}={v}" for k, v in args.items()]
    env = dict(os.environ, HSTU_DISABLE_SPLITKV="1" if disable_splitkv else "0")

    trace_dir = tempfile.mkdtemp(prefix="tdm_kt_") if kernel_check else None
    full = (["rocprofv3", "--kernel-trace", "--output-format", "csv", "-d", trace_dir,
             "-o", "kt", "--"] if kernel_check else []) + cmd
    proc = subprocess.run(full, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True)
    out = proc.stdout

    variants = set()
    if kernel_check:
        variants = kernel_variants(trace_dir)
        shutil.rmtree(trace_dir, ignore_errors=True)

    # On a validation failure the example prints one such line per checked tensor (output,
    # and lse when training) and returns non-zero; keep the worst of them.
    reports = re.findall(r"max err: (\S+), number of errors: (\d+), (\S+)% wrong values", out)
    max_err = max((float(r[0]) for r in reports), default=0.0)
    n_err = max((int(r[1]) for r in reports), default=0)
    wrong = max((float(r[2]) for r in reports), default=0.0)

    problems = []
    if proc.returncode != 0 and not reports:
        problems.append(f"exit code {proc.returncode} without a validation report")
    if max_err > MAX_ERR:
        problems.append(f"max error above {MAX_ERR}")
    if kernel_check:
        if not variants:
            problems.append("no HSTU fwd kernel in the trace")
        if any(v[0] != "Tdm" for v in variants):
            problems.append("a non-Tdm kernel ran")

    label = (f"{name} prec={prec} softmax={softmax} training={training} "
             f"splitkv={'off' if disable_splitkv else 'auto'}")
    kinds = " ".join(sorted(f"{v[2]}/{v[3]}" for v in variants))
    status = "FAIL" if problems else "ok  "
    print(f"{status} {label:<75} max_err={max_err:.3g} errors={n_err} ({wrong:.3g}%) "
          f"[{kinds}] {'; '.join(problems)}", flush=True)
    if problems:
        print("     + " + " ".join(cmd), flush=True)
    return not problems, variants


def main():
    parser = argparse.ArgumentParser(description="Regression test for the Tdm fwd pipelines.")
    parser.add_argument("--exe", default=EXE, help=f"example binary (default: {EXE})")
    parser.add_argument("--no-kernel-check", action="store_true",
                        help="do not run under rocprofv3 to check which kernels ran")
    parser.add_argument("--list", action="store_true", help="list the case names and exit")
    parser.add_argument("-k", default="", help="only run cases whose name contains this")
    args = parser.parse_args()

    if args.list:
        print("\n".join(name for name, _ in CASES))
        return 0

    kernel_check = not args.no_kernel_check
    if kernel_check and shutil.which("rocprofv3") is None:
        print("rocprofv3 not found, skipping the kernel check (or pass --no-kernel-check)")
        kernel_check = False

    failed, seen = 0, set()
    for name, case in CASES:
        if args.k not in name:
            continue
        for prec in ("bf16", "fp16"):
            for softmax, training in ((0, 0), (1, 0), (1, 1)):
                for disable_splitkv in (False, True):
                    ok, variants = run_case(args.exe, kernel_check, name, prec, softmax,
                                            training, disable_splitkv, case)
                    failed += not ok
                    seen |= variants

    if kernel_check and not args.k:
        required = {("Tdm", s, k, m) for s in ("silu", "softmax") for k in ("plain", "splitkv")
                    for m in ("M64", "M128")}
        for v in sorted(required - seen):
            print(f"FAIL kernel variant never exercised: {'/'.join(v)}")
            failed += 1

    print(f"{'FAILED' if failed else 'PASSED'}: {failed} failure(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
