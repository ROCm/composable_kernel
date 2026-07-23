#!/usr/bin/env python3
"""Run the grouped HSTU attention forward correctness tests.

Python port of test_group_hstu_attention.sh, keeping the same functionality and
argument list:

    test_group_hstu_attention.py [norm_dist ...]

The norm distance is derived from the positional arguments exactly as in the
original shell script (see main() for details).
"""

import argparse
import subprocess
import sys

BUILD = "build"
EXE = f"{BUILD}/bin/tile_example_hstu_attention"


def run(exe_prefix, dtype, ndist, **kwargs):
    """Build and execute a single test case, echoing the command (like set -x)."""
    # Argument layout mirrors the original shell script, one option per case.
    cmd = (
        list(exe_prefix)
        + ["-v=1", f"-prec={dtype}", "-b=18", "-g=3", "-nhead=4",
           "-hdim_qk=128", "-hdim_v=128"]
        + [f"-seqlens={kwargs['seqlens']}"]
        + [f"-causal={kwargs['causal']}"]
        + [f"-targets={kwargs['targets']}"]
        + [f"-norm_dist={ndist}"]
        + [f"-g_max_seqlens={kwargs['g_max_seqlens']}"]
        + [f"-g_local_lens={kwargs['g_local_lens']}"]
        + [f"-g_context_lens={kwargs['g_context_lens']}"]
        + [f"-g_minfull_lens={kwargs['g_minfull_lens']}"]
        + [f"-g_attn_scales={kwargs['g_attn_scales']}"]
    )

    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    return result.returncode


def run_special(exe_prefix, ndist, **kwargs):
    """Build and execute one of the special bf16 cases (echoing like set -x)."""
    cmd = (
        list(exe_prefix)
        + ["-v=1", "-prec=bf16", f"-b={kwargs['b']}", f"-g={kwargs['g']}",
           f"-nhead={kwargs['nhead']}", f"-hdim_qk={kwargs['hdim_qk']}",
           f"-hdim_v={kwargs['hdim_v']}", "-causal=1"]
        + [f"-seqlens={kwargs['seqlens']}"]
        + [f"-targets={kwargs['targets']}"]
        + [f"-norm_dist={ndist}", "-alpha=0.25"]
        + [f"-g_max_seqlens={kwargs['g_max_seqlens']}"]
        + [f"-g_local_lens={kwargs['g_local_lens']}"]
        + [f"-g_context_lens={kwargs['g_context_lens']}"]
        + [f"-g_minfull_lens={kwargs['g_minfull_lens']}"]
        + [f"-g_attn_scales={kwargs['g_attn_scales']}"]
    )

    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    return result.returncode


# EXE-level extra flags placed immediately after the binary path (before -v=1).
EXE_PREFIX = [EXE]

SEQLENS = "300,300,290,280,310,308,312"

# Each entry mirrors one grouped invocation in the shell script.
CASES = [
    # no masking
    dict(seqlens=SEQLENS, causal=0, targets=0, g_max_seqlens="310,312,312",
         g_local_lens="0,3,0", g_context_lens="0,0,0", g_minfull_lens="0,0,0",
         g_attn_scales="0,0.1,0"),
    # causal
    dict(seqlens=SEQLENS, causal=1, targets=0, g_max_seqlens="310,312,312",
         g_local_lens="0,3,0", g_context_lens="0,0,0", g_minfull_lens="0,0,0",
         g_attn_scales="0,0.1,0"),
    # causal+local
    dict(seqlens=SEQLENS, causal=1, targets=0, g_max_seqlens="310,312,312",
         g_local_lens="5,5,5", g_context_lens="0,0,0", g_minfull_lens="0,0,0",
         g_attn_scales="0,0.1,0"),
    # causal+local+context
    dict(seqlens=SEQLENS, causal=1, targets=0, g_max_seqlens="310,312,312",
         g_local_lens="5,5,5", g_context_lens="8,8,8", g_minfull_lens="7,7,7",
         g_attn_scales="0,0.1,0"),
    # causal+local+context+target
    dict(seqlens=SEQLENS, causal=1, targets=8, g_max_seqlens="310,312,312",
         g_local_lens="5,5,5", g_context_lens="8,8,8", g_minfull_lens="7,7,7",
         g_attn_scales="0,0.1,0"),
    # no-causal+local+context+target
    dict(seqlens=SEQLENS, causal=0, targets=8, g_max_seqlens="310,312,312",
         g_local_lens="5,5,5", g_context_lens="8,8,8", g_minfull_lens="7,7,7",
         g_attn_scales="0,0.1,0"),
    # causal+local+target (minfull_len > max_uih_len)
    dict(seqlens=SEQLENS, causal=1, targets=8, g_max_seqlens="310,312,312",
         g_local_lens="5,5,5", g_context_lens="8,8,8",
         g_minfull_lens="290,290,290", g_attn_scales="0,0.1,0"),
    # causal+local+context+target (minfull_len > max_uih_len)
    dict(seqlens=SEQLENS, causal=1, targets=8, g_max_seqlens="310,312,312",
         g_local_lens="5,5,5", g_context_lens="8,8,8",
         g_minfull_lens="290,290,290", g_attn_scales="0,0.1,0"),
    # no-causal+local+context+target (minfull_len > max_uih_len)
    dict(seqlens=SEQLENS, causal=0, targets=8, g_max_seqlens="310,312,312",
         g_local_lens="5,5,5", g_context_lens="3,3,3",
         g_minfull_lens="290,290,290", g_attn_scales="0,0.1,0"),
]

SPECIAL_CASES = [
    dict(b=32, g=4, nhead=4, hdim_qk=16, hdim_v=64,
         seqlens=("159,176,195,224,237,188,176,167,153,187,181,162,211,236,"
                  "177,180,251,183,175,176,172,163,242,176,202,255,200,217,"
                  "201,252,162,188"),
         targets=("401,72,259,50,104,475,147,205,192,331,231,199,273,344,434,"
                  "356,369,238,362,467,140,96,49,113,115,38,96,66,225,343,293,"
                  "220"),
         g_max_seqlens="768,768,768,768", g_local_lens="25,27,17,32",
         g_context_lens="0,0,0,0", g_minfull_lens="49,3,33,26",
         g_attn_scales="0.0013,0.0013,0.0013,0.0013"),
    dict(b=16, g=2, nhead=109, hdim_qk=16, hdim_v=16,
         seqlens="89,84,80,60,69,78,67,61,65,98,94,85,88,60,89,84",
         targets="20,4,7,5,3,16,7,5,15,11,6,16,14,11,15,11",
         g_max_seqlens="120,120", g_local_lens="13,2", g_context_lens="0,0",
         g_minfull_lens="14,9", g_attn_scales="0.0083,0.0083"),
    dict(b=8, g=2, nhead=4, hdim_qk=16, hdim_v=16,
         seqlens="81,77,91,72,95,87,73,88",
         targets="5,11,4,15,1,18,4,8",
         g_max_seqlens="120,120", g_local_lens="13,2", g_context_lens="0,0",
         g_minfull_lens="14,9", g_attn_scales="0.0083,0.0083"),
]


def main():
    parser = argparse.ArgumentParser(
        description="Run grouped HSTU attention forward correctness tests.")
    parser.add_argument("args", nargs="*",
                        help="optional positional args (norm_dist ...)")
    parsed = parser.parse_args()

    # Mirror the original bash: `ndist=0; if [ $# -ge 2 ]; then ndist=$1; fi`.
    # ndist becomes the first positional arg ONLY when two or more args are
    # supplied; otherwise it stays 0.
    ndist = 0
    if len(parsed.args) >= 2:
        ndist = parsed.args[0]

    rc = 0
    for dtype in ("fp16", "bf16"):
        for case in CASES:
            ret = run(EXE_PREFIX, dtype, ndist, **case)
            if ret != 0:
                rc = ret

    for case in SPECIAL_CASES:
        ret = run_special(EXE_PREFIX, ndist, **case)
        if ret != 0:
            rc = ret

    return rc


if __name__ == "__main__":
    sys.exit(main())
