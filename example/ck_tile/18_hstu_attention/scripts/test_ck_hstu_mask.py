#!/usr/bin/env python3
"""Run the HSTU attention forward mask-saving tests.

Python port of test_ck_hstu_mask.sh, keeping the same functionality:

    test_ck_hstu_mask.py

There are no arguments. Two invocations save the mask tensor to
ck_hstu_mask.dat, and after each invocation the file is renamed to
ck_hstu_mask_<N>.dat (N=0 then 1), matching the original shell script.
"""

import argparse
import os
import shutil
import subprocess
import sys

BUILD = "build"
EXE = f"{BUILD}/bin/tile_example_hstu_attention"

MASK_FILE = "ck_hstu_mask.dat"


def run(cmd):
    """Execute a single test case, echoing the command (like set -x)."""
    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    return result.returncode


def move_mask(index):
    """Rename the produced mask file to ck_hstu_mask_<index>.dat if present."""
    if os.path.exists(MASK_FILE):
        shutil.move(MASK_FILE, f"ck_hstu_mask_{index}.dat")


def main():
    argparse.ArgumentParser(
        description="Run HSTU attention forward mask-saving tests.").parse_args()

    cases = [
        # minfull_len=0
        [EXE, "-v=1", "-prec=fp16", "-b=3", "-jagged=1", "-nhead=1",
         "-hdim_qk=128", "-hdim_v=128", "-seqlens=49,52,55", "-causal=1",
         "-local_len=4", "-context_len=3", "-minfull_len=0", "-targets=4,5,6",
         "-save_mask=1"],
        # minfull_len=6
        [EXE, "-v=1", "-prec=fp16", "-b=3", "-jagged=1", "-nhead=1",
         "-hdim_qk=128", "-hdim_v=128", "-seqlens=49,52,55", "-causal=1",
         "-local_len=4", "-context_len=3", "-minfull_len=6", "-targets=4,5,6",
         "-save_mask=1"],
    ]

    rc = 0
    for index, cmd in enumerate(cases):
        ret = run(cmd)
        if ret != 0:
            rc = ret
        move_mask(index)

    return rc


if __name__ == "__main__":
    sys.exit(main())
