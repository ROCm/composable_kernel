#!/usr/bin/env python3
"""Benchmark HSTU attention forward across dtypes and sequence lengths.

Python port of benchmark_hstu_attention.sh, keeping the same functionality and
argument list:

    benchmark_hstu_attention.py [use_softmax]

The single optional positional argument defaults to 0, matching the original
shell script. When it is 1 the base binary is invoked with an extra
``-softmax=1`` flag.
"""

import argparse
import subprocess
import sys

BUILD = "build"
BIN = f"{BUILD}/bin/tile_example_hstu_attention"


def run(exe, dtype, seqlen, jagged):
    """Build and execute a single benchmark case, echoing it (like set -x)."""
    cmd = (
        exe
        + ["-v=0", f"-prec={dtype}", "-b=512", f"-jagged={jagged}",
           "-nhead=2", "-hdim_qk=128", "-hdim_v=128", f"-seqlens={seqlen}",
           "-causal=1", "-local_len=5", "-context_len=8", "-minfull_len=7",
           "-targets=8", "-perf=1"]
    )

    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark HSTU attention forward across dtypes/seqlens.")
    parser.add_argument("use_softmax", nargs="?", type=int, default=0,
                        help="use softmax variant (default: 0)")
    args = parser.parse_args()

    if args.use_softmax == 1:
        exe = [BIN, "-softmax=1"]
    else:
        exe = [BIN]

    rc = 0
    for dtype in ("fp16", "bf16"):
        for seqlen in (512, 1024, 3072):
            # jagged is true
            ret = run(exe, dtype, seqlen, 1)
            if ret != 0:
                rc = ret
            # jagged is false
            ret = run(exe, dtype, seqlen, 0)
            if ret != 0:
                rc = ret

    return rc


if __name__ == "__main__":
    sys.exit(main())
