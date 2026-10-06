#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Integration test for CK-tile FMHA-forward custom tune-config codegen.

Drives the *real* `generate.py` entry point (the same one CMake invokes
at configure time) with `CK_TILE_FMHA_FWD_CUSTOM_TUNE_CONFIG_FILE`
pointing at a JSON payload that declares TWO tiles identical in every
field except `F_occupancy` (-1 and 2). This exercises both features
introduced together:

1. `CK_TILE_FMHA_FWD_CUSTOM_TUNE_CONFIG_FILE`
   The JSON must override the default per-arch `KernelComponentFactoryGfx*`
   so our custom tile geometry reaches codegen intact.

2. Per-occupancy kernel instantiation (`kOccupancy_` template parameter)
   The two tiles must produce two *distinct* generated .cpp filenames
   (one with `_o2` suffix, one without) — the file-name-level evidence
   that `kOccupancy_` participates in codegen symbol uniqueness, which
   is what prevents the ODR / link collision documented in
   aiter/csrc/cpp_itfs/mha/tools/README.md.

The test is pure Python (no GPU, no HIP toolchain required); it only
runs the codegen step.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


# --------------------------------------------------------------------------- #
# Locate the fmha example directory (holds generate.py + codegen/).
# --------------------------------------------------------------------------- #

_HERE = os.path.dirname(os.path.abspath(__file__))
_FMHA_EX = os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "example", "ck_tile", "01_fmha")
)
_GENERATE_PY = os.path.join(_FMHA_EX, "generate.py")


# --------------------------------------------------------------------------- #
# Test fixture: a JSON tune-config with two tiles differing only in
# F_occupancy, using a tile geometry that already exists in the default
# `KernelComponentFactoryGfx9.get_hdim_tile_size_dict` for bf16 @
# (128, 128). This ensures both compatibility rules and pipeline
# enumeration accept it without any rule relaxation.
# --------------------------------------------------------------------------- #

_TILE_HKEY_STR = "128,128"
_TILE_STEM_PREFIX = "b128x128x32x128x32x128"
_INTEGRATION_TARGET = "gfx942"


def _minimal_tile(occupancy: int) -> dict:
    return {
        "F_bm0": 128,
        "F_bn0": 128,
        "F_bk0": 32,
        "F_bn1": 128,
        "F_bk1": 32,
        "F_bk0max": 128,
        "F_rm0": 4,
        "F_rn0": 1,
        "F_rk0": 1,
        "F_rm1": 4,
        "F_rn1": 1,
        "F_rk1": 1,
        "F_wm0": 32,
        "F_wn0": 32,
        "F_wk0": 32,
        "F_wm1": 32,
        "F_wn1": 32,
        "F_wk1": 32,
        "F_occupancy": occupancy,
    }


def _build_config(target: str = _INTEGRATION_TARGET) -> dict:
    """Two tiles that differ only in `F_occupancy` (-1 and 2)."""
    return {
        "schema_version": 1,
        "target": target,
        "dtypes": ["bf16"],
        "tiles": {
            "bf16": {
                _TILE_HKEY_STR: [_minimal_tile(-1), _minimal_tile(2)],
            },
        },
        "relax_rules": {"disable_check_hdim_tile": False},
    }


class _JsonConfigOnDisk:
    """RAII helper: dump `cfg` to a temp .json file and remove it on exit."""

    def __init__(self, cfg: dict):
        self._cfg = cfg
        self.path = ""

    def __enter__(self) -> str:
        fp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(self._cfg, fp)
        fp.flush()
        fp.close()
        self.path = fp.name
        return self.path

    def __exit__(self, exc_type, exc, tb):
        try:
            os.unlink(self.path)
        except OSError:
            pass


def _run_generate(env_json_path, extra_args, targets=_INTEGRATION_TARGET):
    """Invoke `generate.py` with the custom config env var set."""
    env = dict(os.environ)
    env["CK_TILE_FMHA_FWD_CUSTOM_TUNE_CONFIG_FILE"] = env_json_path
    cmd = [
        sys.executable,
        _GENERATE_PY,
        "--targets",
        targets,
        "--api",
        "fwd",
        "--optdim",
        "32,64,80,128,256",
    ] + list(extra_args)
    return subprocess.run(
        cmd,
        cwd=_FMHA_EX,
        env=env,
        capture_output=True,
        text=True,
    )


# --------------------------------------------------------------------------- #
# The one and only test: real generate.py, real JSON, real .cpp output.
# --------------------------------------------------------------------------- #


class TestGenerateBlobsWithCustomConfig(unittest.TestCase):
    """
    Verifies that setting `CK_TILE_FMHA_FWD_CUSTOM_TUNE_CONFIG_FILE`
    causes `generate.py` to emit TWO distinct .cpp blobs for the same
    tile geometry — one with `_o2` suffix (pinned occupancy) and one
    without (default occupancy). This is the file-level contract that
    protects both features from regressing.
    """

    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(_GENERATE_PY):
            raise unittest.SkipTest(f"generate.py not found at {_GENERATE_PY}")

    def test_list_blobs_contains_two_occupancy_variants(self):
        cfg = _build_config()
        with _JsonConfigOnDisk(cfg) as cfg_path, tempfile.TemporaryDirectory() as tmp:
            list_txt = os.path.join(tmp, "fwd_blob_list.txt")
            res = _run_generate(cfg_path, ["--list_blobs", str(list_txt)])
            self.assertEqual(
                res.returncode,
                0,
                msg=(
                    "generate.py --list_blobs failed under custom tune "
                    f"config.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
                ),
            )
            self.assertTrue(
                os.path.isfile(list_txt),
                msg="--list_blobs did not produce the expected file",
            )
            blobs = [
                ln.strip()
                for ln in Path(list_txt).read_text().splitlines()
                if ln.strip()
            ]

            # Extract the tile-stem portion of each blob path.
            stems = []
            for b in blobs:
                base = os.path.basename(b)
                idx = base.find(_TILE_STEM_PREFIX)
                if idx >= 0:
                    stems.append(base[idx:])

            has_pinned = any("_o2" in s for s in stems)
            has_default = any(_TILE_STEM_PREFIX in s and "_o" not in s for s in stems)
            self.assertTrue(
                has_pinned,
                msg=(
                    "no blob with `_o2` suffix — pinned-occupancy tile "
                    f"missing from codegen output.\nStems: {stems}"
                ),
            )
            self.assertTrue(
                has_default,
                msg=(
                    "no blob without `_o` suffix — default-occupancy "
                    f"tile missing from codegen output.\nStems: {stems}"
                ),
            )

    def test_write_blobs_emits_distinct_occupancy_files(self):
        cfg = _build_config()
        with _JsonConfigOnDisk(cfg) as cfg_path, tempfile.TemporaryDirectory() as tmp:
            res = _run_generate(cfg_path, ["--output_dir", str(tmp)])
            self.assertEqual(
                res.returncode,
                0,
                msg=(
                    "generate.py --output_dir failed under custom tune "
                    f"config.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
                ),
            )

            # Find every generated .cpp for our tile geometry.
            cpp_paths = []
            for root, _dirs, files in os.walk(tmp):
                for f in files:
                    if f.endswith(".cpp") and _TILE_STEM_PREFIX in f:
                        cpp_paths.append(os.path.join(root, f))

            self.assertGreaterEqual(
                len(cpp_paths),
                2,
                msg=(
                    "expected at least 2 generated .cpp for our tile; "
                    f"found {len(cpp_paths)}: {cpp_paths}"
                ),
            )

            saw_default = False
            saw_pinned = False
            for p in cpp_paths:
                base = os.path.basename(p)
                if "_o2" in base:
                    saw_pinned = True
                elif "_o" not in base:
                    saw_default = True

            self.assertTrue(
                saw_default and saw_pinned,
                msg=(
                    "codegen did not emit BOTH a default-occupancy blob "
                    "and a pinned-occupancy (_o2) blob for the same tile "
                    f"geometry.\nBlobs seen: {cpp_paths}"
                ),
            )


if __name__ == "__main__":
    unittest.main()
