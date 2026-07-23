#!/usr/bin/env python3
"""Benchmark HSTU attention forward for batched (non-jagged) causal cases.

Python port of bench_batched_causal.sh, keeping the same functionality and
argument list:

    bench_batched_causal.py [use_softmax]

The single optional positional argument defaults to 0, matching the original
shell script. When it is 1 the base binary is invoked with an extra
``-softmax=1`` flag.
"""

import argparse
import subprocess
import sys

BUILD = "build"
BIN = f"{BUILD}/bin/tile_example_hstu_attention"


def run(exe, dtype, num_batch, num_head, hdim, seqlen, causal):
    """Build and execute a single benchmark case, echoing it (like set -x)."""
    cmd = (
        exe
        + ["-v=0", f"-prec={dtype}", f"-b={num_batch}", "-jagged=0",
           f"-nhead={num_head}", f"-hdim_qk={hdim}", f"-hdim_v={hdim}",
           f"-seqlens={seqlen}", f"-causal={causal}", "-local_len=0",
           "-context_len=0", "-minfull_len=0", "-targets=0", "-perf=1"]
    )

    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark HSTU attention forward batched causal cases.")
    parser.add_argument("use_softmax", nargs="?", type=int, default=0,
                        help="use softmax variant (default: 0)")
    args = parser.parse_args()

    if args.use_softmax == 1:
        exe = [BIN, "-softmax=1"]
    else:
        exe = [BIN]

    dtype = "bf16"
    hdim = 128
    num_batch = 32
    num_head = 4

    rc = 0
    for seqlen in (1024, 2048, 4096, 8192, 16384, 32768):
        # no causal
        ret = run(exe, dtype, num_batch, num_head, hdim, seqlen, 0)
        if ret != 0:
            rc = ret
        # has causal
        ret = run(exe, dtype, num_batch, num_head, hdim, seqlen, 1)
        if ret != 0:
            rc = ret

    return rc


if __name__ == "__main__":
    sys.exit(main())
