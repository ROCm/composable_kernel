#!/usr/bin/env python3
"""Run the HSTU attention backward jagged-causal (mattn0/full0) dropout tests.

Python port of test_jagged_causal_mattn0_full0_with_dropout_bwd.sh, keeping the
same functionality and argument list:

    test_jagged_causal_mattn0_full0_with_dropout_bwd.py [use_softmax]

The single optional positional argument selects whether softmax is enabled and
defaults to 0, matching the original shell script. Dropout (-p_drop=0.2) is
always enabled.
"""

import argparse
import subprocess
import sys

BUILD = "build"
EXE_BASE = f"{BUILD}/bin/tile_example_hstu_attention_bwd"

DTYPE = "bf16"
NDIST = 0
LOCAL_LEN = 0
MINFULL_LEN = 0

TARGET8 = "10,10,14,17,16,12,14,9"
TARGET16 = "13,17,16,13,7,14,3,18,15,15,1,9,18,18,7,10"
TARGET32 = "13,17,16,13,7,14,3,18,15,15,1,9,18,18,7,10,11,0,4,8,2,10,20,14,11,7,4,6,9,7,14,17"

SEQLENS = [1004, 2028, 3052, 4076, 8172, 16364]


def run(exe_prefix, b, seqlen, targets):
    """Build and execute a single test case, echoing the command (like set -x)."""
    cmd = (
        list(exe_prefix)
        + ["-v=1", f"-prec={DTYPE}", f"-b={b}", "-jagged=1", "-nhead=4",
           "-hdim_qk=128", "-hdim_v=128", f"-seqlens={seqlen}", "-causal=1",
           f"-local_len={LOCAL_LEN}", "-context_len=0",
           f"-minfull_len={MINFULL_LEN}", f"-targets={targets}",
           "-max_target=20", "-alpha=2.0", f"-norm_dist={NDIST}"]
    )

    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    print("")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run HSTU attention backward jagged-causal "
                    "(mattn0/full0) dropout tests.")
    parser.add_argument("use_softmax", nargs="?", type=int, default=0,
                        help="enable softmax (default: 0)")
    args = parser.parse_args()

    if args.use_softmax == 1:
        exe_prefix = [EXE_BASE, "-softmax=1", "-p_drop=0.2"]
    else:
        exe_prefix = [EXE_BASE, "-p_drop=0.2"]

    cases = [
        (8, TARGET8),
        (16, TARGET16),
        (32, TARGET32),
    ]

    rc = 0
    for b, targets in cases:
        for seqlen in SEQLENS:
            ret = run(exe_prefix, b, seqlen, targets)
            if ret != 0:
                rc = ret

    return rc


if __name__ == "__main__":
    sys.exit(main())
