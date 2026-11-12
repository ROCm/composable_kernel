#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Codegen tests for the float->bf16 rounding mode of generated FMHA sources.

The FMHA CMake targets select round-to-nearest-even through
CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT (5 for RDNA forward, 3 for backward). Builds
that run generate.py with their own build system do not see those flags, and
without them ck_tile falls back to truncation. The generated sources therefore
set the same default themselves, unless the build already defines it.

Pure Python; the preprocessor checks additionally need ROCm clang.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_CK_ROOT = Path(__file__).resolve().parents[3]
_FMHA_EX = _CK_ROOT / "example" / "ck_tile" / "01_fmha"
_GENERATE_PY = _FMHA_EX / "generate.py"

_ROCM_ROOT = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
_CLANG = next(
    (
        str(p)
        for p in (
            _ROCM_ROOT / "llvm" / "bin" / "clang",
            _ROCM_ROOT / "lib" / "llvm" / "bin" / "clang",
        )
        if p.is_file()
    ),
    shutil.which("clang"),
)

_MACRO = "CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT"
_STANDARD_ASM = "3"
_STANDARD_CNAN = "5"


def _generate(api, output_dir):
    cmd = [
        sys.executable,
        str(_GENERATE_PY),
        "--targets",
        "gfx9,gfx115",
        "--api",
        api,
        "--receipt",
        "4",
        "--optdim",
        "128",
        "--output_dir",
        str(output_dir),
    ]
    return subprocess.run(cmd, cwd=_FMHA_EX, capture_output=True, text=True)


def _prologue(text):
    """Everything before the first #include, i.e. what decides the macro."""
    return text[: text.index("#include")]


class TestFmhaBf16RoundingCodegen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.sources = {}
        for api in ("fwd", "fwd_splitkv", "fwd_appendkv", "bwd"):
            out = Path(cls._tmp.name) / api
            out.mkdir()
            res = _generate(api, out)
            if res.returncode != 0:
                raise RuntimeError(f"generate.py --api {api} failed:\n{res.stderr}")
            # Kernel instances only; the *_api.cpp dispatchers are host code.
            cls.sources[api] = {
                p.name: p.read_text()
                for p in out.glob("*.cpp")
                if not p.name.endswith("_api.cpp")
            }

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_every_instance_sets_the_default_before_any_include(self):
        for api, files in self.sources.items():
            self.assertTrue(files, msg=f"no sources generated for {api}")
            for name, text in files.items():
                with self.subTest(api=api, file=name):
                    self.assertIn(f"#define {_MACRO} ", _prologue(text))

    def _effective_mode(self, prologue, arch, extra=()):
        with tempfile.NamedTemporaryFile("w", suffix=".hip") as src:
            src.write(prologue)
            src.flush()
            res = subprocess.run(
                [_CLANG, "-x", "hip", "--cuda-device-only", f"--offload-arch={arch}"]
                + ["-nogpulib", "-nogpuinc", "-E", "-dM", *extra, src.name],
                capture_output=True,
                text=True,
            )
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        for line in res.stdout.splitlines():
            parts = line.split()
            if parts[:2] == ["#define", _MACRO]:
                return parts[2]
        return None

    def _one(self, api, needle):
        name = next(n for n in sorted(self.sources[api]) if needle in n)
        return _prologue(self.sources[api][name])

    @unittest.skipIf(_CLANG is None, "clang is required")
    def test_effective_mode_per_arch(self):
        fwd_rdna = self._one("fwd", "_gfx115.cpp")
        fwd_gfx9 = self._one("fwd", "_gfx9.cpp")
        bwd = self._one("bwd", "_bf16_")
        # RDNA forward and all backward kernels round to nearest even.
        self.assertEqual(self._effective_mode(fwd_rdna, "gfx1151"), _STANDARD_CNAN)
        self.assertEqual(self._effective_mode(fwd_rdna, "gfx1201"), _STANDARD_CNAN)
        self.assertEqual(self._effective_mode(bwd, "gfx1151"), _STANDARD_ASM)
        self.assertEqual(self._effective_mode(bwd, "gfx942"), _STANDARD_ASM)
        # gfx9 forward keeps the ck_tile default, as in the CMake build.
        self.assertIsNone(self._effective_mode(fwd_gfx9, "gfx942"))

    @unittest.skipIf(_CLANG is None, "clang is required")
    def test_build_definition_takes_precedence(self):
        override = [f"-D{_MACRO}=2"]
        for api in ("fwd", "bwd"):
            with self.subTest(api=api):
                self.assertEqual(
                    self._effective_mode(self._one(api, "_bf16_"), "gfx1151", override),
                    "2",
                )


if __name__ == "__main__":
    unittest.main()
