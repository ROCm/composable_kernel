#!/usr/bin/env python3
"""Run the HSTU attention forward hdim96/hdim64 correctness tests.

Python port of test_hstu_attention_hdim96_hdim64.sh, keeping the same
functionality. This script can be used for verifying the use of WarpGemm
32x32x16 which is used by hdim64 + softmax.

It takes no command-line arguments; the attention scale, norm dist and dtype
are hardcoded to match the original shell script. The body sweeps hdim over
96 then 64, running the 14 test cases for each.
"""

import argparse
import subprocess
import sys

BUILD = "build"
EXE = f"{BUILD}/bin/tile_example_hstu_attention_fwd"

# EXE-level flags applied right after the binary path (see original .sh).
EXE_FLAGS = ["-softmax=0"]


def run(dtype, attn_scale, ndist, hdim, **kwargs):
    """Build and execute a single test case, echoing the command (like set -x)."""
    cmd = (
        [EXE]
        + EXE_FLAGS
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
    parser.parse_args()

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
            ret = run(dtype, attn_scale, ndist, hdim, **case)
            if ret != 0:
                rc = ret

    return rc


if __name__ == "__main__":
    sys.exit(main())
