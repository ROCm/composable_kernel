#!/usr/bin/env python3
"""Benchmark HSTU attention forward, jagged causal, mattn256 full256, hdim64.

Python port of bench_jagged_causal_mattn256_full256_hdim64.sh, keeping the same
functionality and argument list:

    bench_jagged_causal_mattn256_full256_hdim64.py [use_softmax]

The optional positional argument use_softmax defaults to 0, matching the
original shell script. The environment variable TEST_HSTU_FWD_TRAINING
controls the -training flag value (default 0).
"""

import argparse
import os
import subprocess
import sys

BUILD = "build"


def run(exe_prefix, dtype, b, seqlens, targets):
    """Build and execute a single benchmark case, echoing the command."""
    cmd = (
        exe_prefix
        + ["-v=0", f"-prec={dtype}", f"-b={b}", "-jagged=1", "-nhead=4",
           "-hdim_qk=64", "-hdim_v=64", f"-seqlens={seqlens}", "-causal=1",
           "-local_len=256", "-context_len=0", "-minfull_len=256",
           f"-targets={targets}", "-max_target=20", "-perf=1", "-alpha=2.0"]
    )
    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    print("")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark HSTU attention forward (jagged causal mattn256 full256 hdim64).")
    parser.add_argument("use_softmax", nargs="?", type=int, default=0,
                        help="use softmax (default: 0)")
    args = parser.parse_args()

    training = os.environ.get("TEST_HSTU_FWD_TRAINING", "0")

    binary = f"{BUILD}/bin/tile_example_hstu_attention_fwd"
    if args.use_softmax == 1:
        exe_prefix = [binary, "-softmax=1", f"-training={training}"]
    else:
        exe_prefix = [binary, f"-training={training}"]

    dtype = "bf16"

    target8 = "10,10,14,17,16,12,14,9"
    target16 = "13,17,16,13,7,14,3,18,15,15,1,9,18,18,7,10"
    target32 = ("13,17,16,13,7,14,3,18,15,15,1,9,18,18,7,10,"
                "11,0,4,8,2,10,20,14,11,7,4,6,9,7,14,17")

    seqlens = ["1004", "2028", "3052", "4076", "8172", "16364"]

    rc = 0
    for b, targets in ((8, target8), (16, target16), (32, target32)):
        for seq in seqlens:
            ret = run(exe_prefix, dtype, b, seq, targets)
            if ret != 0:
                rc = ret

    return rc


if __name__ == "__main__":
    sys.exit(main())
