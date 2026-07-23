#!/usr/bin/env python3
"""Run the HSTU attention forward hdim96/hdim64 correctness tests.

Merges the former test_hstu_attention_hdim96_hdim64.py and
test_hstu_softmax_attention_hdim96_hdim64.py into a single script:

    test_hstu_attention_hdim96_hdim64.py [use_softmax]

This script can be used for verifying the use of WarpGemm 32x32x16 which is
used by hdim64 + softmax.

The optional positional argument defaults to 0. When use_softmax is 1, the
executable is invoked with -softmax=1. The attention scale, norm dist
and dtype are hardcoded to match the original shell scripts. The training flag
is read from the TEST_HSTU_FWD_TRAINING environment variable (default 0). The
body sweeps hdim over 96 then 64, running the 14 test cases for each.
"""

import argparse
import os
import subprocess
import sys

BUILD = "build"
EXE = f"{BUILD}/bin/tile_example_hstu_attention_bwd"


def run(exe_flags, dtype, attn_scale, ndist, hdim, **kwargs):
    """Build and execute a single test case, echoing the command (like set -x)."""
    cmd = (
        [EXE]
        + exe_flags
        + ["-v=1", f"-prec={dtype}", "-b=10"]
        + [f"-jagged={kwargs['jagged']}"]
        + ["-nhead=4", f"-hdim_qk={hdim}", f"-hdim_v={hdim}"]
        + [f"-seqlens={kwargs['seqlens']}"]
        + [f"-causal={kwargs['causal']}"]
        + [f"-local_len={kwargs['local_len']}"]
        + [f"-context_len={kwargs['context_len']}"]
        + [f"-minfull_len={kwargs['minfull_len']}"]
        + [f"-targets={kwargs['targets']}"]
        + [f"-attn_scale={attn_scale}", f"-norm_dist={ndist}"]
    )

    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run HSTU attention forward hdim96/hdim64 correctness "
                    "tests.")
    parser.add_argument("use_softmax", nargs="?", type=int, default=0,
                        help="use softmax (default: 0)")
    args = parser.parse_args()

    # EXE-level flags applied right after the binary path (see original .sh).
    if args.use_softmax == 1:
        exe_flags = ["-softmax=1"]
    else:
        exe_flags = ["-softmax=0"]

    attn_scale = "1.0"
    ndist = "1"
    dtype = "fp16"

    seqlens_jagged = "300,300,290,280,310"

    # Each entry mirrors one invocation in the shell script.
    cases = [
        # no masking batched
        dict(jagged=0, seqlens=256, causal=0, local_len=0, context_len=0,
             minfull_len=0, targets=0),
        # no masking jagged
        dict(jagged=1, seqlens=seqlens_jagged, causal=0, local_len=0,
             context_len=0, minfull_len=0, targets=0),
        # batched causal
        dict(jagged=0, seqlens=256, causal=1, local_len=0, context_len=0,
             minfull_len=0, targets=0),
        # jagged causal
        dict(jagged=1, seqlens=seqlens_jagged, causal=1, local_len=0,
             context_len=0, minfull_len=0, targets=0),
        # batched causal+local
        dict(jagged=0, seqlens=256, causal=1, local_len=5, context_len=0,
             minfull_len=0, targets=0),
        # jagged causal+local
        dict(jagged=1, seqlens=seqlens_jagged, causal=1, local_len=5,
             context_len=0, minfull_len=0, targets=0),
        # batched causal+local+context
        dict(jagged=0, seqlens=256, causal=1, local_len=5, context_len=8,
             minfull_len=7, targets=0),
        # jagged causal+local+context
        dict(jagged=1, seqlens=seqlens_jagged, causal=1, local_len=5,
             context_len=8, minfull_len=7, targets=0),
        # batched causal+local+context+target
        dict(jagged=0, seqlens=256, causal=1, local_len=5, context_len=8,
             minfull_len=7, targets=8),
        # jagged causal+local+context+target
        dict(jagged=1, seqlens=seqlens_jagged, causal=1, local_len=5,
             context_len=8, minfull_len=7, targets=8),
        # jagged no-causal+local+context+target
        dict(jagged=1, seqlens=seqlens_jagged, causal=0, local_len=5,
             context_len=8, minfull_len=7, targets=8),
        # jagged causal+local+target (minfull_len > max_uih_len)
        dict(jagged=1, seqlens=seqlens_jagged, causal=1, local_len=5,
             context_len=0, minfull_len=290, targets=8),
        # jagged causal+local+context+target (minfull_len > max_uih_len)
        dict(jagged=1, seqlens=seqlens_jagged, causal=1, local_len=5,
             context_len=8, minfull_len=290, targets=8),
        # jagged no-causal+local+context+target (minfull_len > max_uih_len)
        dict(jagged=1, seqlens=seqlens_jagged, causal=0, local_len=5,
             context_len=3, minfull_len=290, targets=8),
    ]

    rc = 0
    for hdim in (96, 64):
        for case in cases:
            ret = run(exe_flags, dtype, attn_scale, ndist, hdim, **case)
            if ret != 0:
                rc = ret

    return rc


if __name__ == "__main__":
    sys.exit(main())
