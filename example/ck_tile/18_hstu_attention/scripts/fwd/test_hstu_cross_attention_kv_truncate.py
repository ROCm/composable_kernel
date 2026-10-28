#!/usr/bin/env python3
"""Run the HSTU cross-attention forward KV-truncation tests.

    test_hstu_cross_attention_kv_truncate.py

Each case runs the jagged non-causal SiLU cross-attention forward with
-kv_truncate_len, which keeps the first (or with -kv_keep_last=1, the last)
kv_truncate_len keys of every key/value sequence. The CPU validation (-v=1)
runs the reference without truncation on just the kept keys.

The kv sequence lengths cover sequences shorter than, equal to and longer than
kv_truncate_len, including ones at least twice as long so the first and last
windows do not overlap, and an empty sequence. The last case truncates a long
sequence to a few keys, which changes the split-kv choice of the dispatch.
"""

import subprocess
import sys

BUILD = "build"
EXE = f"{BUILD}/bin/tile_example_hstu_attention_fwd"

# (seqlens_kv, kv_truncate_len)
KV_CASES = [
    ("50,64,100,128,300,0,257,512", 64),
    ("50,64,100,128,300,0,257,512", 128),
    ("1000,1500,2048,700", 300),
    ("100,200,300", 512),
    ("4096,4000,3900,4096", 32),
]


def run(dtype, keep_last, seqlens_kv, kv_truncate_len):
    """Build and execute a single test case, echoing the command (like set -x)."""
    num_batch = len(seqlens_kv.split(","))
    cmd = [
        EXE,
        "-v=1",
        f"-prec={dtype}",
        f"-b={num_batch}",
        "-jagged=1",
        "-nhead=4",
        "-hdim_qk=128",
        "-hdim_v=128",
        "-seqlens=128",
        f"-seqlens_kv={seqlens_kv}",
        "-targets=8,16,32,64,100,7,1,128",
        "-causal=0",
        "-local_len=0",
        "-context_len=0",
        "-minfull_len=0",
        "-softmax=0",
        "-attn_scale=0",
        "-norm_dist=0",
        f"-kv_truncate_len={kv_truncate_len}",
        f"-kv_keep_last={keep_last}",
    ]

    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    return result.returncode


def main():
    rc = 0
    for dtype in ("fp16", "bf16"):
        for keep_last in (0, 1):
            for seqlens_kv, kv_truncate_len in KV_CASES:
                ret = run(dtype, keep_last, seqlens_kv, kv_truncate_len)
                if ret != 0:
                    rc = ret

    return rc


if __name__ == "__main__":
    sys.exit(main())
