#!/usr/bin/env python3
"""Run the HSTU cross-attention backward correctness tests.

Python port of test_hstu_cross_attention_bwd.sh, keeping the same functionality
and argument list:

    test_hstu_cross_attention_bwd.py [use_softmax]

The single optional positional argument defaults to 0, matching the original
shell script. When it equals 1, the base executable is invoked with an extra
-softmax=1 flag.
"""

import argparse
import subprocess
import sys

BUILD = "build"
EXE = f"{BUILD}/bin/tile_example_hstu_attention_bwd"


def run(exe, dtype, **kwargs):
    """Build and execute a single test case, echoing the command (like set -x)."""
    # exe is a list so the optional -softmax=1 prefix is preserved verbatim.
    # Argument layout mirrors the original shell script, one option per case.
    cmd = (
        list(exe)
        + ["-v=1", f"-prec={dtype}", "-b=10"]
        + [f"-jagged={kwargs['jagged']}"]
        + ["-nhead=4", "-hdim_qk=128", "-hdim_v=128"]
        + [f"-seqlens={kwargs['seqlens']}"]
        + ["-seqlens_kv=380"]
        + [f"-causal={kwargs['causal']}"]
        + [f"-local_len={kwargs['local_len']}"]
        + [f"-context_len={kwargs['context_len']}"]
        + [f"-minfull_len={kwargs['minfull_len']}"]
        + [f"-targets={kwargs['targets']}"]
        + ["-attn_scale=0", "-norm_dist=0"]
    )

    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    return result.returncode


def run_special(exe):
    """Run the single special case that verifies masking when seqlen_kv > seqlen_q."""
    cmd = list(exe) + [
        "-v=1", "-prec=bf16", "-b=3", "-jagged=1", "-nhead=1",
        "-hdim_qk=128", "-hdim_v=128", "-seqlens=52,55,58",
        "-seqlens_kv=70,76,80", "-causal=1", "-local_len=0", "-context_len=0",
        "-minfull_len=0", "-targets=4,5,6", "-attn_scale=0", "-norm_dist=0",
    ]
    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run HSTU cross-attention backward correctness tests.")
    parser.add_argument("use_softmax", nargs="?", default=0, type=int,
                        help="use softmax (default: 0)")
    args = parser.parse_args()

    if args.use_softmax == 1:
        exe = [EXE, "-softmax=1"]
    else:
        exe = [EXE]

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
            ret = run(exe, dtype, **case)
            if ret != 0:
                rc = ret

    # This case verifies masking when seqlen_kv > seqlen_q by comparing the
    # saved mask tensor with the output of test_pytorch_hstu_mask_v2.py.
    ret = run_special(exe)
    if ret != 0:
        rc = ret

    return rc


if __name__ == "__main__":
    sys.exit(main())
