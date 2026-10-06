#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Codegen tests for the per-architecture FMHA-backward pipeline selection.

Drives the *real* `generate.py` -- the same entry point CMake invokes at
configure time -- and inspects the `BlockFmhaBwdPipelineProblem` template
arguments it emits.

Decode pipeline selection is carried per instance by the codegen rather than by
a global macro, so that gfx950's decode tiles do not land in a pipeline built on
TDM, an instruction only gfx12 has. These tests hold that contract:

1. Every gfx950 decode instance opts *out* of the TDM decode pipeline.
2. Every gfx1250 decode instance opts *in*.
3. The two trailing flags are emitted in the order the problem template
   declares them.

Point 3 matters because both flags are `bool` with a default: swapping them
compiles cleanly and silently selects the wrong pipeline, which is exactly the
failure these tests exist to catch.

Pure Python -- no GPU and no HIP toolchain required, only the codegen step.
"""

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


# --------------------------------------------------------------------------- #
# Locate the fmha example directory (holds generate.py + codegen/) and the
# header that declares the problem template.
# --------------------------------------------------------------------------- #

_HERE = os.path.dirname(os.path.abspath(__file__))
_CK_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
_FMHA_EX = os.path.join(_CK_ROOT, "example", "ck_tile", "01_fmha")
_GENERATE_PY = os.path.join(_FMHA_EX, "generate.py")
_PROBLEM_HPP = os.path.join(
    _CK_ROOT,
    "include",
    "ck_tile",
    "ops",
    "fmha",
    "pipeline",
    "block_fmha_bwd_pipeline_problem.hpp",
)

# A decode instance is one built from a tile with a non-zero max_seq_q, which
# the generated file name records as `_maxq<N>_`.
_DECODE_RE = re.compile(r"_maxq(\d+)_")

# `... , <Traits>, <kUseTdmKRKTR>, <kUseTdmDecode>>;` closes the problem alias.
_PROBLEM_TAIL_RE = re.compile(
    r"BlockFmhaBwdPipelineProblem<.*?"
    r"fmha_bwd_trait_\d+,\s*(true|false),\s*(true|false)>\s*;",
    re.DOTALL,
)


def _run_generate(target, output_dir, optdim="64,128"):
    """Invoke the real generate.py for one architecture."""
    cmd = [
        sys.executable,
        _GENERATE_PY,
        "--targets",
        target,
        "--api",
        "bwd",
        "--receipt",
        "3",
        "--optdim",
        optdim,
        "--output_dir",
        str(output_dir),
    ]
    return subprocess.run(cmd, cwd=_FMHA_EX, capture_output=True, text=True)


def _decode_instances(output_dir):
    """Map {generated .cpp name -> (kUseTdmKRKTR, kUseTdmDecode)} for decode tiles."""
    found = {}
    for root, _dirs, files in os.walk(output_dir):
        for name in files:
            if not name.endswith(".cpp"):
                continue
            m = _DECODE_RE.search(name)
            if not m or int(m.group(1)) == 0:
                continue  # maxq0 is a non-decode tile
            text = Path(os.path.join(root, name)).read_text()
            tail = _PROBLEM_TAIL_RE.search(text)
            if tail:
                found[name] = (tail.group(1) == "true", tail.group(2) == "true")
    return found


class TestFmhaBwdDecodeDispatchCodegen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(_GENERATE_PY):
            raise unittest.SkipTest(f"generate.py not found at {_GENERATE_PY}")

    def _generate(self, target):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        res = _run_generate(target, tmp.name)
        self.assertEqual(
            res.returncode,
            0,
            msg=(
                f"generate.py failed for {target}.\n"
                f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
            ),
        )
        instances = _decode_instances(tmp.name)
        self.assertTrue(
            instances,
            msg=(
                f"no decode instances generated for {target}; the tile table or "
                "the generated file naming changed and this test no longer "
                "inspects what it thinks it does"
            ),
        )
        return instances

    def test_gfx950_decode_stays_off_the_tdm_pipeline(self):
        """gfx950 has no TDM; its decode tiles must not opt into that pipeline."""
        offenders = {
            name: flags
            for name, flags in self._generate("gfx950").items()
            if flags[1]  # kUseTdmDecode
        }
        self.assertEqual(
            offenders,
            {},
            msg=(
                "gfx950 decode instances opted into the TDM decode pipeline. "
                "That pipeline issues tensor loads gfx950 does not have.\n"
                f"Offending instances: {sorted(offenders)}"
            ),
        )

    def test_gfx1250_decode_takes_the_tdm_pipeline(self):
        """gfx1250 decode tiles are the ones the TDM pipeline was written for."""
        instances = self._generate("gfx1250")
        missing = {name: flags for name, flags in instances.items() if not flags[1]}
        self.assertEqual(
            missing,
            {},
            msg=(
                "gfx1250 decode instances did not opt into the TDM decode "
                "pipeline, so they fall back to the plain one, which has no "
                f"gfx12 form.\nInstances: {sorted(missing)}"
            ),
        )

    def test_trailing_flag_order_matches_the_problem_template(self):
        """
        Both flags are defaulted bools, so emitting them in the wrong order
        compiles and silently picks the wrong pipeline. Pin the order against
        the template declaration rather than against a literal.
        """
        self.assertTrue(
            os.path.isfile(_PROBLEM_HPP),
            msg=f"problem header not found at {_PROBLEM_HPP}",
        )
        decl = Path(_PROBLEM_HPP).read_text()
        order = re.findall(r"bool\s+(kUseTdm\w*)_\s*=\s*false", decl)
        self.assertEqual(
            order,
            ["kUseTdmKRKTR", "kUseTdmDecode"],
            msg=(
                "the trailing template parameters of BlockFmhaBwdPipelineProblem "
                f"changed; codegen emits them positionally. Found: {order}"
            ),
        )

        # A gfx1250 decode instance exercises both flags with different values,
        # so a swap would show up here.
        instances = self._generate("gfx1250")
        for name, (krktr, decode) in instances.items():
            self.assertFalse(
                krktr,
                msg=(
                    f"{name} is a decode instance and must not also select the "
                    "non-decode TdmKRKTR pipeline; the two flags look swapped"
                ),
            )
            self.assertTrue(decode, msg=f"{name} lost its TDM decode flag")

    def test_gfx1250_decode_dispatch_boundaries(self):
        sys.path.insert(0, _FMHA_EX)
        self.addCleanup(sys.path.remove, _FMHA_EX)
        from codegen.ops.fmha_bwd import FmhaBwdApiTrait, KernelComponentFactoryGfx125

        tiles = [
            tile
            for tile in KernelComponentFactoryGfx125.get_dq_dk_dv_tiles("fp16", "t")
            if tile.tdm_decode
        ]
        self.assertTrue(tiles)
        for tile in tiles:
            for mode in ("batch", "group"):
                trait = FmhaBwdApiTrait(
                    KernelComponentFactoryGfx125.arch, 0, tile.F_bhdq, "fp16", mode,
                    tile, "no", "no", "false", "dropout", "true", 0, 0,
                    "false", "simplified", "t",
                )
                prefix = "max_" if mode == "group" else ""
                expected = (
                    f" && (t.{prefix}seqlen_q <= 32)"
                    " && (t.batch * t.nhead_q >= 768)"
                )
                self.assertEqual(trait.max_seq_q_cond, expected)
                expression = trait.max_seq_q_cond.removeprefix(" && ").replace("&&", "and")
                for query_length, grid, matches in (
                    (31, 768, True), (32, 768, True), (33, 768, False),
                    (32, 767, False), (32, 769, True),
                ):
                    traits = SimpleNamespace(
                        seqlen_q=query_length if mode == "batch" else query_length * grid,
                        max_seqlen_q=query_length, batch=grid, nhead_q=1,
                    )
                    with self.subTest(head_dim=tile.F_bhdq, mode=mode,
                                      query_length=query_length, grid=grid):
                        self.assertEqual(eval(expression, {"__builtins__": {}}, {"t": traits}),
                                         matches)


if __name__ == "__main__":
    unittest.main()
