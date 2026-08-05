#!/usr/bin/env python3
"""Run the HSTU cross-attention forward correctness tests.

Python port of test_hstu_cross_attention.sh, keeping the same functionality and
argument list:

    test_hstu_cross_attention.py [USE_SOFTMAX]

USE_SOFTMAX is an optional positional argument and defaults to 0, matching the
original shell script. When USE_SOFTMAX==1, -softmax=1 is added to the EXE
prefix. The Training value is read from the TEST_HSTU_FWD_TRAINING environment
variable (default 0).
"""

import argparse
import os
import subprocess
import sys

BUILD = "build"
BINARY = f"{BUILD}/bin/tile_example_hstu_attention_fwd"


def run(exe, dtype, **kwargs):
    """Build and execute a single test case, echoing the command (like set -x)."""
    cmd = (
        exe
        + ["-v=1", f"-prec={dtype}", "-b=10"]
        + [f"-jagged={kwargs['jagged']}"]
        + ["-nhead=4", "-hdim_qk=128", "-hdim_v=128"]
        + [f"-seqlens={kwargs['seqlens']}"]
        + [f"-seqlens_kv={kwargs['seqlens_kv']}"]
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


def main():
    parser = argparse.ArgumentParser(
        description="Run HSTU cross-attention forward correctness tests.")
    parser.add_argument("use_softmax", nargs="?", type=int, default=0,
                        help="use softmax (default: 0)")
    args = parser.parse_args()

    training = os.environ.get("TEST_HSTU_FWD_TRAINING", "0")

    if args.use_softmax == 1:
        exe = [BINARY, "-softmax=1", f"-training={training}"]
    else:
        exe = [BINARY, f"-training={training}"]

    seqlens_jagged = "300,300,290,280,310"

    cases = [
        # no masking batched
        dict(jagged=0, seqlens=256, seqlens_kv=300, causal=0, local_len=0,
             context_len=0, minfull_len=0, targets=0),
        # no masking jagged
        dict(jagged=1, seqlens=seqlens_jagged, seqlens_kv=380, causal=0,
             local_len=0, context_len=0, minfull_len=0, targets=0),
        # batched causal
        dict(jagged=0, seqlens=256, seqlens_kv=300, causal=1, local_len=0,
             context_len=0, minfull_len=0, targets=0),
        # jagged causal
        dict(jagged=1, seqlens=seqlens_jagged, seqlens_kv=380, causal=1,
             local_len=0, context_len=0, minfull_len=0, targets=0),
        # batched causal+local
        dict(jagged=0, seqlens=256, seqlens_kv=300, causal=1, local_len=5,
             context_len=0, minfull_len=0, targets=0),
        # jagged causal+local
        dict(jagged=1, seqlens=seqlens_jagged, seqlens_kv=380, causal=1,
             local_len=5, context_len=0, minfull_len=0, targets=0),
        # batched causal+local+context
        dict(jagged=0, seqlens=256, seqlens_kv=300, causal=1, local_len=5,
             context_len=8, minfull_len=7, targets=0),
        # jagged causal+local+context
        dict(jagged=1, seqlens=seqlens_jagged, seqlens_kv=380, causal=1,
             local_len=5, context_len=8, minfull_len=7, targets=0),
        # batched causal+local+context+target
        dict(jagged=0, seqlens=256, seqlens_kv=300, causal=1, local_len=5,
             context_len=8, minfull_len=7, targets=8),
        # jagged causal+local+context+target
        dict(jagged=1, seqlens=seqlens_jagged, seqlens_kv=380, causal=1,
             local_len=5, context_len=8, minfull_len=7, targets=8),
        # jagged no-causal+local+context+target
        dict(jagged=1, seqlens=seqlens_jagged, seqlens_kv=380, causal=0,
             local_len=5, context_len=8, minfull_len=7, targets=8),
        # jagged causal+local+target (minfull_len > max_uih_len)
        dict(jagged=1, seqlens=seqlens_jagged, seqlens_kv=380, causal=1,
             local_len=5, context_len=0, minfull_len=290, targets=8),
        # jagged causal+local+context+target (minfull_len > max_uih_len)
        dict(jagged=1, seqlens=seqlens_jagged, seqlens_kv=380, causal=1,
             local_len=5, context_len=8, minfull_len=290, targets=8),
        # jagged no-causal+local+context+target (minfull_len > max_uih_len)
        dict(jagged=1, seqlens=seqlens_jagged, seqlens_kv=380, causal=0,
             local_len=5, context_len=3, minfull_len=290, targets=8),
    ]

    rc = 0
    for dtype in ("fp16", "bf16"):
        for case in cases:
            ret = run(exe, dtype, **case)
            if ret != 0:
                rc = ret

    # This case is used to verify the masking when seqlen_kv > seqlen_q by
    # comparing the saved mask tensor with test_pytorch_hstu_mask_v2.py.
    special = (
        exe
        + ["-v=1", "-prec=bf16", "-b=3", "-jagged=1", "-nhead=1",
           "-hdim_qk=128", "-hdim_v=128", "-seqlens=52,55,58",
           "-seqlens_kv=70,76,80", "-causal=1", "-local_len=0",
           "-context_len=0", "-minfull_len=0", "-targets=4,5,6",
           "-attn_scale=0", "-norm_dist=0", "-save_mask=1"]
    )
    print("+ " + " ".join(special), flush=True)
    ret = subprocess.run(special).returncode
    if ret != 0:
        rc = ret

    return rc


if __name__ == "__main__":
    sys.exit(main())
