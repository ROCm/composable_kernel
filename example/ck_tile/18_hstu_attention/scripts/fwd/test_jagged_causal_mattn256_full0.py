#!/usr/bin/env python3
"""Run the HSTU attention forward jagged-causal (mattn256/full0) tests.

Python port of test_jagged_causal_mattn256_full0.sh, keeping the same
functionality and argument list:

    test_jagged_causal_mattn256_full0.py [norm_dist]

The optional positional argument is the norm distance and defaults to 0,
matching the original shell script. This script does not support a softmax
option.
"""

import argparse
import os
import subprocess
import sys

BUILD = "build"
BIN = f"{BUILD}/bin/tile_example_hstu_attention_fwd"

LOCAL_LEN = 256
MINFULL_LEN = 0

SEQLENS = [1004, 2028, 3052, 4076, 8172, 16364]

TARGET8 = "10,10,14,17,16,12,14,9"
TARGET16 = "13,17,16,13,7,14,3,18,15,15,1,9,18,18,7,10"
TARGET32 = "13,17,16,13,7,14,3,18,15,15,1,9,18,18,7,10,11,0,4,8,2,10,20,14,11,7,4,6,9,7,14,17"


def run(exe, dtype, b, seqlen, targets, ndist):
    """Build and execute a single test case, echoing the command (like set -x)."""
    cmd = exe + [
        "-v=1",
        f"-prec={dtype}",
        f"-b={b}",
        "-jagged=1",
        "-nhead=4",
        "-hdim_qk=128",
        "-hdim_v=128",
        f"-seqlens={seqlen}",
        "-causal=1",
        f"-local_len={LOCAL_LEN}",
        "-context_len=0",
        f"-minfull_len={MINFULL_LEN}",
        f"-targets={targets}",
        "-max_target=20",
        "-alpha=2.0",
        f"-norm_dist={ndist}",
    ]
    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    print("")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run HSTU attention forward jagged-causal mattn256/full0 tests.")
    parser.add_argument("norm_dist", nargs="?", type=int, default=0,
                        help="norm dist (default: 0)")
    args = parser.parse_args()

    ndist = args.norm_dist

    training = os.environ.get("TEST_HSTU_FWD_TRAINING", "0")

    exe = [BIN, f"-training={training}"]

    dtype = "bf16"

    groups = [(8, TARGET8), (16, TARGET16), (32, TARGET32)]

    rc = 0
    for b, targets in groups:
        for seqlen in SEQLENS:
            ret = run(exe, dtype, b, seqlen, targets, ndist)
            if ret != 0:
                rc = ret

    return rc


if __name__ == "__main__":
    sys.exit(main())
