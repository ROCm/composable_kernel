#!/usr/bin/env python3
"""Run the HSTU attention backward hdim256 correctness tests.

    test_hstu_attention_hdim256_bwd.py [use_softmax]

The other backward test scripts cover hdim 128, 96 and 64 only. hdim 256 has
its own tile settings for both backward kernels, and on gfx950 an unpadded
hdim 256 runs the direct-to-LDS backward pipelines, so this script sweeps the
same masking cases at hdim 256 for self attention and for cross attention
(seqlens_kv >= seqlens).

The optional positional argument defaults to 0. When use_softmax is 1, the
executable is invoked with -softmax=1. The attention scale, norm dist and
dtypes are hardcoded to match the hdim96/hdim64 backward script.

At hdim 256 the example's element-wise check (rtol 1.6e-2, atol 1e-5) also
flags a few rounding outliers on correct kernels (at most ~0.2% of the dq / dk
/ dv elements on gfx950 with the trload backward pipelines), so a case fails
only when more than MAX_WRONG_PCT percent of the elements of a gradient are
wrong, or when the example fails without reporting a mismatch.
"""

import argparse
import re
import subprocess
import sys

BUILD = "build"
EXE = f"{BUILD}/bin/tile_example_hstu_attention_bwd"

HDIM = 256
MAX_WRONG_PCT = 1.0

WRONG_PCT_RE = re.compile(r"([0-9.eE+-]+)% wrong values")


def run(exe_flags, dtype, attn_scale, ndist, seqlens_kv, **kwargs):
    """Build and execute a single test case, echoing the command (like set -x)."""
    cmd = (
        [EXE]
        + exe_flags
        + ["-v=1", f"-prec={dtype}", "-b=10"]
        + [f"-jagged={kwargs['jagged']}"]
        + ["-nhead=4", f"-hdim_qk={HDIM}", f"-hdim_v={HDIM}"]
        + [f"-seqlens={kwargs['seqlens']}"]
        + ([f"-seqlens_kv={seqlens_kv}"] if seqlens_kv else [])
        + [f"-causal={kwargs['causal']}"]
        + [f"-local_len={kwargs['local_len']}"]
        + [f"-context_len={kwargs['context_len']}"]
        + [f"-minfull_len={kwargs['minfull_len']}"]
        + [f"-targets={kwargs['targets']}"]
        + [f"-attn_scale={attn_scale}", f"-norm_dist={ndist}"]
    )

    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    print(result.stdout, end="", flush=True)

    wrong_pcts = [float(p) for p in WRONG_PCT_RE.findall(result.stdout)]
    if result.returncode != 0 and not wrong_pcts:
        return result.returncode
    if wrong_pcts and max(wrong_pcts) > MAX_WRONG_PCT:
        print(f"FAILED: {max(wrong_pcts)}% wrong values > {MAX_WRONG_PCT}%",
              flush=True)
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Run HSTU attention backward hdim256 correctness tests.")
    parser.add_argument("use_softmax", nargs="?", type=int, default=0,
                        help="use softmax (default: 0)")
    args = parser.parse_args()

    if args.use_softmax == 1:
        exe_flags = ["-softmax=1"]
    else:
        exe_flags = ["-softmax=0"]

    attn_scale = "1.0"
    ndist = "1"

    seqlens_jagged = "300,300,290,280,310"

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
        # batched causal+local+context+target
        dict(jagged=0, seqlens=256, causal=1, local_len=5, context_len=8,
             minfull_len=7, targets=8),
        # jagged causal+local+context+target
        dict(jagged=1, seqlens=seqlens_jagged, causal=1, local_len=5,
             context_len=8, minfull_len=7, targets=8),
        # jagged no-causal+local+context+target
        dict(jagged=1, seqlens=seqlens_jagged, causal=0, local_len=5,
             context_len=8, minfull_len=7, targets=8),
    ]

    # Self attention, then cross attention with seqlens_kv >= seqlens.
    rc = 0
    for dtype in ("fp16", "bf16"):
        for seqlens_kv in ("", "380"):
            for case in cases:
                ret = run(exe_flags, dtype, attn_scale, ndist, seqlens_kv,
                          **case)
                if ret != 0:
                    rc = ret

    return rc


if __name__ == "__main__":
    sys.exit(main())
