#!/usr/bin/env python3
"""Run the HSTU attention forward correctness tests.

Merges the former test_hstu_attention.py and test_hstu_softmax_attention.py
into a single script:

    test_hstu_attention.py [use_softmax] [attn_scale] [norm_dist]

All arguments are optional positional arguments and default to 0. When
use_softmax is 1, the executable is invoked with -softmax=1.

The training mode is read from the TEST_HSTU_FWD_TRAINING environment variable
(default 0), matching the original shell scripts.
"""

import argparse
import os
import subprocess
import sys

BUILD = "build"
EXE = f"{BUILD}/bin/tile_example_hstu_attention_fwd"


def run(exe_prefix, dtype, attn_scale, ndist, **kwargs):
    """Build and execute a single test case, echoing the command (like set -x)."""
    # Argument layout mirrors the original shell script, one option per case.
    cmd = (
        list(exe_prefix)
        + ["-v=1", f"-prec={dtype}", "-b=10"]
        + [f"-jagged={kwargs['jagged']}"]
        + ["-nhead=4", "-hdim_qk=128", "-hdim_v=128"]
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
        description="Run HSTU attention forward correctness tests.")
    parser.add_argument("use_softmax", nargs="?", type=int, default=0,
                        help="use softmax (default: 0)")
    parser.add_argument("attn_scale", nargs="?", default=0,
                        help="attention scale (default: 0)")
    parser.add_argument("norm_dist", nargs="?", default=0,
                        help="norm dist (default: 0)")
    args = parser.parse_args()

    attn_scale = args.attn_scale
    ndist = args.norm_dist

    training = os.environ.get("TEST_HSTU_FWD_TRAINING", "0")

    if args.use_softmax == 1:
        exe_prefix = [EXE, "-p_drop=0.2",  "-softmax=1", f"-training={training}"]
    else:
        exe_prefix = [EXE, "-p_drop=0.2", f"-training=0"]

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
    for dtype in ("fp16", "bf16"):
        for case in cases:
            ret = run(exe_prefix, dtype, attn_scale, ndist, **case)
            if ret != 0:
                rc = ret

    return rc


if __name__ == "__main__":
    sys.exit(main())
