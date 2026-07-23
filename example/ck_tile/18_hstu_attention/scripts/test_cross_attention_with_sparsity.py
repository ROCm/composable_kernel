#!/usr/bin/env python3
"""Run the HSTU cross-attention forward sparsity tests.

Python port of test_cross_attention_with_sparsity.sh, keeping the same
functionality and argument list:

    test_cross_attention_with_sparsity.py [USE_SOFTMAX]

USE_SOFTMAX is an optional positional argument and defaults to 0, matching the
original shell script. When USE_SOFTMAX==1, -softmax=1 is added to the EXE
prefix (which already contains -v=1). The Training value is read from the
TEST_HSTU_FWD_TRAINING environment variable (default 0).

The test cases are generated using sparsity = 0.95. For each batch size the
five max_target configurations use a batch-specific targets list, and each
configuration is run over the shared list of kv sequence lengths.
"""

import argparse
import os
import subprocess
import sys

BUILD = "build"
BINARY = f"{BUILD}/bin/tile_example_hstu_attention"

# Shared kv sequence lengths, run for every (batch, targets) configuration.
KV_SEQLENS = [1022, 2044, 4088, 6132, 8176, 16352]

# Batch size -> list of five max_target targets lists (order: 32, 128, 160,
# 256, 300), taken verbatim from the shell script.
BATCH_TGTS = {
    4: [
        "13,20,3,15",
        "87,4,117,115",
        "98,11,127,41",
        "230,64,105,232",
        "49,127,151,227",
    ],
    8: [
        "12,16,24,30,28,6,6,5",
        "22,11,86,32,14,4,14,116",
        "144,140,65,145,16,146,155,79",
        "37,21,8,157,215,70,99,184",
        "103,262,156,253,161,119,16,201",
    ],
    16: [
        "23,14,17,17,14,27,29,7,27,1,3,27,27,8,7,1",
        "103,86,121,21,35,17,2,27,93,31,11,108,5,86,21,51",
        "157,18,118,152,158,67,116,20,160,32,98,14,31,104,17,77",
        "239,156,221,117,90,193,151,218,173,42,123,6,54,210,114,190",
        "107,34,279,276,189,97,34,192,242,260,34,132,277,193,18,8",
    ],
    32: [
        "7,19,3,11,15,26,26,21,10,4,16,6,14,17,26,19,11,31,16,12,6,1,28,11,18,13,11,2,26,13,31,6",
        "27,77,78,74,44,52,79,5,61,108,72,50,85,19,10,108,103,79,69,37,81,51,70,113,39,33,123,91,33,109,70,40",
        "57,135,96,34,104,112,52,156,67,13,82,20,127,37,30,93,48,133,2,23,44,141,106,16,138,62,138,34,139,41,52,120",
        "63,211,77,104,202,134,227,156,125,72,29,173,239,197,210,240,147,82,101,209,56,187,181,172,195,165,231,46,178,201,125,78",
        "213,290,136,222,173,57,175,244,100,6,152,254,132,118,200,219,63,110,37,197,61,130,271,214,228,4,131,120,151,95,45,248",
    ],
}

BATCHES = [4, 8, 16, 32]


def run(exe, batch, kv, tgts):
    """Build and execute a single test case, echoing the command (like set -x)."""
    cmd = (
        exe
        + ["-prec=bf16", f"-b={batch}", "-jagged=1", "-nhead=4",
           "-hdim_qk=128", "-hdim_v=128", "-seqlens=128", f"-seqlens_kv={kv}",
           "-causal=0", "-local_len=0", "-context_len=0", "-minfull_len=0",
           f"-targets={tgts}", "-attn_scale=0", "-norm_dist=0"]
    )

    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run HSTU cross-attention forward sparsity tests.")
    parser.add_argument("use_softmax", nargs="?", type=int, default=0,
                        help="use softmax (default: 0)")
    args = parser.parse_args()

    training = os.environ.get("TEST_HSTU_FWD_TRAINING", "0")

    if args.use_softmax == 1:
        exe = [BINARY, "-v=1", "-softmax=1", f"-training={training}"]
    else:
        exe = [BINARY, "-v=1", f"-training={training}"]

    rc = 0
    for batch in BATCHES:
        for tgts in BATCH_TGTS[batch]:
            for kv in KV_SEQLENS:
                ret = run(exe, batch, kv, tgts)
                if ret != 0:
                    rc = ret

    return rc


if __name__ == "__main__":
    sys.exit(main())
