#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared Tile-Engine perf-flag construction for the block-scale quant bridges.

Single source of truth for the ``-mllvm`` flag set the bridge ctypes ``.so`` is
compiled with, so the flags cannot drift across the per-op utils files.  Before
this module, ``_te_perf_flags`` / ``_coerce_flag_ok`` were copy-pasted verbatim
into ``gemm_{aquant,rowcolquant,tensor_quant}_utils.py`` (5-flag set), and
``gemm_abquant_utils.py`` carried its own near-identical copy with two extra
flags -- a live example of the duplication drifting.

The base set is the 5 authoritative develop Tile-Engine flags
(``composablekernel/CMakeLists.txt`` L521/L528/L535/L546/L547).  abquant's gfx950
EightWaves fast path needs two additional flags; they are passed via ``extra=``
so the single base definition stays canonical.
"""

import os
import subprocess

# The authoritative develop Tile-Engine -mllvm perf flag set (5 flags).
_TE_BASE_FLAGS = [
    "-fno-offload-uniform-block",
    "-mllvm", "--lsr-drop-solution=1",
    "-mllvm", "-enable-post-misched=0",
    "-mllvm", "-amdgpu-early-inline-all=true",
    "-mllvm", "-amdgpu-function-calls=false",
]


def coerce_flag_supported(hipcc):
    """True iff the local clang accepts ``-mllvm -amdgpu-coerce-illegal-types=1``.

    ROCm 7.2 clang>=22 removed it and aborts the compile, so gate on it.  The
    kernels are bit-accurate without it (it only tightens register allocation on
    older toolchains).
    """
    try:
        r = subprocess.run(
            [hipcc, "-x", "hip", "-c", "-mllvm",
             "-amdgpu-coerce-illegal-types=1", "-", "-o", "/dev/null"],
            input="int main(){return 0;}", text=True,
            capture_output=True, timeout=60)
        return r.returncode == 0
    except Exception:
        return False


def te_perf_flags(hipcc, extra=None):
    """The Tile-Engine ``-mllvm`` perf flags for a bridge ctypes ``.so`` compile.

    Without these, ``hipcc -O3`` register allocation on the block-scale hot loops
    spills to scratch and collapses occupancy, so the bridge kernel runs slower
    than the byte-identical Old-TE kernel.  Kept in lockstep with the develop TE
    build for fair parity.  Disabled entirely when ``CK_BRIDGE_NO_TE_FLAGS=1``.

    ``extra`` appends op-specific ``-mllvm`` flags (e.g. abquant EightWaves:
    ``-enable-noalias-to-md-conversion``, ``-greedy-reverse-local-assignment``)
    before the toolchain-gated coerce flag, preserving the original per-op order.
    """
    if os.environ.get("CK_BRIDGE_NO_TE_FLAGS") == "1":
        return []
    flags = list(_TE_BASE_FLAGS)
    if extra:
        flags += list(extra)
    if coerce_flag_supported(hipcc):
        flags += ["-mllvm", "-amdgpu-coerce-illegal-types=1"]
    return flags
