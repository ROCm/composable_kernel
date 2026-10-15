#!/usr/bin/env python3
"""Regression test for the gfx1250 Tdm forward pipelines.

hstu_attention_{no,with}_softmax_fwd_tdm_pipeline.hpp keep two Lds copies of K and V and
alternate between them from one kv-tile to the next, so mistakes in that bookkeeping only
show up for particular kv-tile counts (one, odd, even, none at all in an empty split-kv
range). The cases below vary those counts on all Tdm kernel variants (MTile 64/128,
plain/split-kv), with and without softmax, for every head-dim bucket that uses Tdm:

    test_tdm_pipeline.py [--exe PATH] [--hdim 64,96,128,256] [--max-err X|inf] [-k REGEX]
                         [--timeout S] [--no-kernel-check] [--list]

-k selects runs by a regular expression searched in the full label
("hdim=QK/V case prec=.. softmax=.. training=.. splitkv=.."); the variant check then only
asks for the softmax modes selected (pass --no-kernel-check if a selection cannot reach
every variant). --max-err inf only reports.

Buckets. A run with head dims (qk, v) goes to the smallest MaxK bucket of 64/96/128/256
that holds both. Each bucket runs every case at qk == v == MaxK, plus a few padded or
asymmetric pairs (PAIRS) on a subset of the cases.

Pass criterion. The gfx1250 baseline does not pass the example's own validation everywhere:
with softmax a few percent of the outputs miss its tolerance. So a run passes if its max
error stays within MAX_ERR[bucket]. The share of wrong values is no use as a criterion: a
pipeline reading V from the wrong copy got only 3.6% wrong on a split-kv case, about what
the baseline gets. The max error the example reports is taken over the out-of-tolerance
values only and is 0 when a run passes its tolerance; 0 does not mean bit-exact.

MAX_ERR[bucket] is chosen inside [2 x worst baseline, min mutant / 2], where the worst
baseline is the largest max error of this test on the Tdm build and on a build with Tdm
disabled (all buckets on trload), and min mutant is the smallest, over deliberately broken
pipelines, of their largest max error. The runs use the example's default scale (alpha 0,
i.e. 1/sqrt(hdim)); a large alpha makes softmax one-hot, so only the diagonal kv-tile
would be checked.

    bucket  worst baseline  min mutant  MAX_ERR   runtime
    64      2.93e-3         0.328       1e-2      3 min (222 runs, gfx1250)
    96      2.93e-3         0.189       7e-3      4 min (222 runs, gfx1250)
    128     1.95e-3         0.152       5e-3      4 min (190 runs, gfx1250)
    256     1.46e-3         0.129       5e-3      5.5 min (222 runs, gfx1250)

The runtime is that of the whole bucket with the kernel check. The slowest single run took
about 5 s (bucket 256; 6.8 s while the machine was busy compiling); --timeout defaults to 30 s.

Mutants (applied out of tree to the Tdm pipelines and rebuilt for the jagged bf16 lse=false
instances only, so they show on the bf16 training=0 runs of the m64_*, m128_* and
cross_attention cases; the group cases have their own instances): K or V read from the
wrong copy, a single Lds copy, one kv-tile too many (split-kv runs only), the K or V Tdm
pad disabled, and the K or V head-dim view shortened by 8 (padded pairs only). Largest max
error per mutant:

    bucket  K, V wrong copy; one copy  K, V pad disabled  one kv-tile more  K, V view - 8
    64      inf, 2.0e36, 1.98          2.6e35, inf        0.328             0.425, 1
    96      420, 4.4e36, 1.98          5.5e35, inf        0.264             0.189, 1
    128     1.59, 3.5e36, 2            1.7e35, inf        0.227             0.152, 0.996
    256     249, 6.3e36, 2             4.1e35, inf        0.21              0.129, 1

Each one fails on all of those runs, except one kv-tile more (split-kv runs only) and the
shortened views (only the pairs with a padded qk resp. v).

Kernel check. With rocprofv3 every run is traced. Only Tdm kernels may run, and each one
must have its bucket's tile (EXPECTED_TDM_TILES), MaxK == bucket, kN0 == kN0Sub == kK1 and
the head-dim pad flags the dispatcher derives from (qk, v). Every Tdm variant of a selected
bucket (silu/softmax x plain/split-kv x tile) must be hit at least once.
"""

import argparse
import csv
import glob
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

BUILD = "build"
EXE = f"{BUILD}/bin/tile_example_hstu_attention_fwd"

# Tdm tile per bucket and MTile slot, as (M, N0, N0Sub, N1, K1, MaxK). Mirrors the Tdm rows of
# hstu_attention_fwd_setting_gfx125.hpp; change both together.
EXPECTED_TDM_TILES = {
    64: {64: (64, 32, 32, 64, 32, 64), 128: (128, 64, 64, 64, 64, 64)},
    96: {64: (64, 32, 32, 96, 32, 96), 128: (128, 64, 64, 96, 64, 96)},
    128: {64: (64, 32, 32, 128, 32, 128), 128: (128, 64, 64, 128, 64, 128)},
    256: {64: (64, 32, 32, 256, 32, 256), 128: (64, 32, 32, 256, 32, 256)},
}

# See the docstring for how these are chosen.
MAX_ERR = {64: 1e-2, 96: 7e-3, 128: 5e-3, 256: 5e-3}

# Padded or asymmetric (hdim_qk, hdim_v) pairs per bucket, all multiples of 8. The pairs with
# qk != v are where a V-side mistake shows: the example fills Q, K and V from one seed, so K
# and V are equal when qk == v.
PAIRS = {
    64: [(64, 48), (48, 64), (32, 32)],
    96: [(96, 80), (72, 96), (80, 72)],
    128: [(128, 112), (112, 128)],
    256: [(256, 192), (160, 256), (200, 192)],
}

DEFAULT_HDIMS = [64, 96, 128, 256]

BUCKETS = (64, 96, 128, 256)

# The Tdm tiles have kN0 32 (MTile 64) or 64 (MTile 128) except where a bucket says otherwise;
# these lengths give 1, 2, 3, ... kv-tiles, both odd and even, for kN0 32 and 64 (checked in
# check_seqlens()).
SEQLENS_M64 = "1,31,32,33,64,65,96,127,128,129,192,300"
SEQLENS_M128 = "1,63,64,65,128,129,191,192,255,256,320,383,384,511,512,640"

GROUP = dict(jagged=1, g=2, b=8, seqlens="100,200,300,400,500,600,700,800",
             g_max_seqlens="400,800", g_local_lens="0,0", g_context_lens="0,0",
             g_minfull_lens="0,0", g_attn_scales="0,0")

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
    ("batched_nocausal", dict(jagged=0, b=4, seqlens=1000, causal=0)),
    # cross attention, seqlen_kv >> seqlen_q, goes to MTile 128 and split-kv
    ("cross_attention", dict(jagged=1, b=2, seqlens="128,100", seqlens_kv="1000,1100",
                             causal=0)),
    ("group", dict(GROUP, causal=1)),
    ("group_nocausal", dict(GROUP, causal=0)),
]

# The cases the padded pairs run on: jagged, batched, group and the split-kv MTile-128 path.
PAIR_CASES = ("m64_causal", "batched_causal", "group", "cross_attention")


def bucket_of(qk, v):
    return next(b for b in BUCKETS if qk <= b and v <= b)


def check_seqlens(hdims):
    """Each SEQLENS list must give 1 kv-tile, an odd count > 1 and an even count of every kN0."""
    for b in hdims:
        for mtile, tile in EXPECTED_TDM_TILES[b].items():
            seqlens = SEQLENS_M64 if mtile == 64 else SEQLENS_M128
            counts = {math.ceil(int(s) / tile[1]) for s in seqlens.split(",")}
            if not (1 in counts and any(c > 1 and c % 2 for c in counts)
                    and any(c % 2 == 0 for c in counts)):
                raise SystemExit(f"SEQLENS for MTile {mtile} do not cover kN0 {tile[1]} "
                                 f"of bucket {b}: kv-tile counts {sorted(counts)}")


def kernel_variants(trace_dir):
    """Return the HSTU fwd kernels in a rocprofv3 kernel trace as
    (pipeline, softmax, split, tile, pad_qk, pad_v) tuples."""
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
                m = re.search(r"TileSettingClass<ck_tile::sequence<([\d, ]+)>", name)
                tile = tuple(int(x) for x in m.group(1).split(",")) if m else None
                m = re.search(r"HstuAttentionFwdTraits<(\w+), (\w+), (\w+), (\w+), -?\d+>", name)
                pads = (m.group(3) == "true", m.group(4) == "true") if m else (None, None)
                found.add((pipeline, softmax, split, tile) + pads)
    return found


def kernel_problems(variants, qk, v):
    b = bucket_of(qk, v)
    problems = set()
    if not variants:
        problems.add("no HSTU fwd kernel in the trace")
    for pipeline, _, _, tile, pad_qk, pad_v in variants:
        if pipeline != "Tdm":
            problems.add("a non-Tdm kernel ran")
        elif tile not in EXPECTED_TDM_TILES[b].values():
            problems.add(f"tile {tile} is not a bucket-{b} Tdm tile")
        elif tile[5] != b or not tile[1] == tile[2] == tile[4]:
            problems.add(f"tile {tile}: MaxK != {b} or kN0, kN0Sub, kK1 differ")
        elif (pad_qk, pad_v) != (qk % tile[5] != 0, v % tile[3] != 0):
            problems.add(f"head-dim pad flags {(pad_qk, pad_v)} do not match ({qk}, {v})")
    return sorted(problems)


def run_case(args, kernel_check, max_err, qk, v, name, prec, softmax, training,
             disable_splitkv, case):
    cargs = dict(nhead=4, hdim_qk=qk, hdim_v=v, causal=1, local_len=0, context_len=0,
                 minfull_len=0, targets=0)
    cargs.update(case)
    cmd = [args.exe, "-v=1", f"-prec={prec}", f"-softmax={softmax}", f"-training={training}"]
    cmd += [f"-{k}={x}" for k, x in cargs.items()]
    env = dict(os.environ, HSTU_DISABLE_SPLITKV="1" if disable_splitkv else "0")

    trace_dir = tempfile.mkdtemp(prefix="tdm_kt_") if kernel_check else None
    full = (["rocprofv3", "--kernel-trace", "--output-format", "csv", "-d", trace_dir,
             "-o", "kt", "--"] if kernel_check else []) + cmd
    problems = []
    try:
        proc = subprocess.run(full, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, timeout=args.timeout)
        out, returncode = proc.stdout, proc.returncode
    except subprocess.TimeoutExpired:
        out, returncode = "", None
        problems.append(f"timeout after {args.timeout} s")

    variants = set()
    if kernel_check:
        variants = kernel_variants(trace_dir)
        shutil.rmtree(trace_dir, ignore_errors=True)

    # On a validation failure the example prints one such line per checked tensor (output,
    # and lse when training) and returns non-zero; keep the worst of them.
    reports = re.findall(r"max err: (\S+), number of errors: (\d+), (\S+)% wrong values", out)
    err = max((float(r[0]) for r in reports), default=0.0)
    n_err = max((int(r[1]) for r in reports), default=0)
    wrong = max((float(r[2]) for r in reports), default=0.0)

    if returncode not in (0, None) and not reports:
        problems.append(f"exit code {returncode} without a validation report")
    if err > max_err:
        problems.append(f"max error above {max_err}")
    if kernel_check and returncode is not None:
        problems += kernel_problems(variants, qk, v)

    label = (f"hdim={qk}/{v} {name} prec={prec} softmax={softmax} training={training} "
             f"splitkv={'off' if disable_splitkv else 'auto'}")
    kinds = " ".join(sorted(f"{x[2]}/M{x[3][0]}" for x in variants if x[3]))
    status = "FAIL" if problems else "ok  "
    print(f"{status} {label:<92} max_err={err:.3g} errors={n_err} ({wrong:.3g}%) "
          f"[{kinds}] {'; '.join(problems)}", flush=True)
    if problems:
        print("     + " + " ".join(cmd), flush=True)
    return not problems, variants


def runs(hdims):
    """Yield (qk, v, name, prec, softmax, training, disable_splitkv, case) for every run."""
    for b in hdims:
        for qk, v in [(b, b)] + PAIRS.get(b, []):
            for name, case in CASES:
                if (qk, v) != (b, b) and name not in PAIR_CASES:
                    continue
                for prec in ("bf16", "fp16"):
                    for softmax, training in ((0, 0), (1, 0), (1, 1)):
                        if (qk, v) != (b, b) and training:
                            continue
                        # non-cross MTile-128 runs never split, so the off run would repeat
                        for disable_splitkv in ((False,) if name.startswith("m128_")
                                                else (False, True)):
                            yield qk, v, name, prec, softmax, training, disable_splitkv, case


def main():
    parser = argparse.ArgumentParser(description="Regression test for the Tdm fwd pipelines.")
    parser.add_argument("--exe", default=EXE, help=f"example binary (default: {EXE})")
    parser.add_argument("--hdim", default=",".join(map(str, DEFAULT_HDIMS)),
                        help="comma-separated buckets to run (default: %(default)s)")
    parser.add_argument("--max-err", default=None,
                        help="override MAX_ERR for every bucket; 'inf' only reports")
    parser.add_argument("--timeout", type=float, default=30, help="seconds per run")
    parser.add_argument("--no-kernel-check", action="store_true",
                        help="do not run under rocprofv3 to check which kernels ran")
    parser.add_argument("--list", action="store_true", help="list the run labels and exit")
    parser.add_argument("-k", default="", help="only run labels matching this regex")
    args = parser.parse_args()

    hdims = [int(h) for h in args.hdim.split(",")]
    for b in hdims:
        if b not in EXPECTED_TDM_TILES:
            raise SystemExit(f"bucket {b} has no Tdm tiles in this test")
    check_seqlens(hdims)
    pattern = re.compile(args.k)

    def selected(r):
        qk, v, name, prec, softmax, training, disable_splitkv, _ = r
        label = (f"hdim={qk}/{v} {name} prec={prec} softmax={softmax} training={training} "
                 f"splitkv={'off' if disable_splitkv else 'auto'}")
        return label if pattern.search(label) else None

    todo = [(selected(r), r) for r in runs(hdims) if selected(r)]
    if args.list:
        print("\n".join(label for label, _ in todo))
        return 0

    kernel_check = not args.no_kernel_check
    if kernel_check and shutil.which("rocprofv3") is None:
        print("rocprofv3 not found; install it or pass --no-kernel-check")
        return 2

    failed, seen = 0, {}
    for _, (qk, v, name, prec, softmax, training, disable_splitkv, case) in todo:
        b = bucket_of(qk, v)
        max_err = float(args.max_err) if args.max_err is not None else MAX_ERR[b]
        ok, variants = run_case(args, kernel_check, max_err, qk, v, name, prec, softmax,
                                training, disable_splitkv, case)
        failed += not ok
        seen.setdefault(b, set()).update(variants)

    if kernel_check:
        modes = {"softmax" if r[4] else "silu" for _, r in todo}
        for b in sorted({bucket_of(r[0], r[1]) for _, r in todo}):
            required = {("Tdm", s, k, t) for s in modes for k in ("plain", "splitkv")
                        for t in set(EXPECTED_TDM_TILES[b].values())}
            for x in sorted(required - {x[:4] for x in seen.get(b, set())}):
                print(f"FAIL kernel variant never exercised: hdim {b} {x[1]}/{x[2]} tile {x[3]}")
                failed += 1

    print(f"{'FAILED' if failed else 'PASSED'}: {failed} failure(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
