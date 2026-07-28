#!/usr/bin/env python3
"""Benchmark HSTU attention forward for jagged causal cases.

Python port of bench_jagged_causal.sh, keeping the same functionality and
argument list:

    bench_jagged_causal.py [use_softmax]

The single optional positional argument defaults to 0, matching the original
shell script. When it is 1 the base binary is invoked with an extra
``-softmax=1`` flag.
"""

import argparse
import subprocess
import sys

BUILD = "build"
BIN = f"{BUILD}/bin/tile_example_hstu_attention_fwd"

# Per-batch sequence-length lists, stored comma-joined (the shell script builds
# these from space-separated lists via its add_comma helper).
SL1024 = "889,602,63,923,219,400,572,297,896,115,792,313,134,605,424,582,376,975,67,50,41,582,306,580,803,680,44,117,141,688,579,958"
SL2048 = "34,822,1581,415,1458,408,1897,968,176,640,1148,623,521,1734,135,874,662,1132,1907,283,679,818,1679,1723,1601,655,1774,1810,317,507,1347,1127"
SL4096 = "1497,2516,3179,2891,190,3572,640,3025,464,1824,712,1519,2727,2621,1135,704,1752,1665,384,1796,2567,2329,1926,2911,3787,2185,17,898,2186,3725,719,1515"
SL8192 = "4571,3202,270,1540,8169,3365,6055,7181,2942,4213,2717,3593,7748,4646,5502,4489,6525,2481,7397,2983,5667,1003,7926,3659,6129,6647,3758,6244,4175,2327,849,5261"
SL16384 = "6956,7177,338,13755,10382,13392,10150,15592,15929,5256,6825,3804,5197,13415,14099,12418,13772,13659,5998,3715,9862,9183,11826,12964,6041,6712,12846,475,4672,7690,12280,10175"


def run(exe, dtype, num_batch, num_head, hdim, seqlens, causal, target):
    """Build and execute a single benchmark case, echoing it (like set -x)."""
    cmd = (
        exe
        + ["-v=0", f"-prec={dtype}", f"-b={num_batch}", "-jagged=1",
           f"-nhead={num_head}", f"-hdim_qk={hdim}", f"-hdim_v={hdim}",
           f"-seqlens={seqlens}", f"-causal={causal}", "-local_len=0",
           "-context_len=0", "-minfull_len=0", f"-targets={target}", "-perf=1"]
    )

    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark HSTU attention forward jagged causal cases.")
    parser.add_argument("use_softmax", nargs="?", type=int, default=0,
                        help="use softmax variant (default: 0)")
    args = parser.parse_args()

    if args.use_softmax == 1:
        exe = [BIN, "-softmax=1"]
    else:
        exe = [BIN]

    dtype = "bf16"
    hdim = 128
    num_batch = 32
    num_head = 4
    target = 20

    rc = 0
    for seqlens in (SL1024, SL2048, SL4096, SL8192, SL16384):
        ret = run(exe, dtype, num_batch, num_head, hdim, seqlens, 0, target)
        if ret != 0:
            rc = ret
        print("")
        ret = run(exe, dtype, num_batch, num_head, hdim, seqlens, 1, target)
        if ret != 0:
            rc = ret
        print("")

    return rc


if __name__ == "__main__":
    sys.exit(main())
