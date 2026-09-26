#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Pin that the committed arch_specs_generated.py / arch_specs_generated.hpp are
exactly what generate_arch_specs.py emits from arch_specs.json.

A hand edit to either generated file is silently dropped the next time anyone
regenerates, so every rule must live in arch_specs.json or in the generator.
Only the "Generated at:" timestamp line is allowed to differ.

Run: python3 -m pytest tests/test_arch_specs_regeneration.py -v
"""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
CODEGEN_DIR = DISPATCHER_DIR / "codegen"
COMMITTED_PY = CODEGEN_DIR / "arch_specs_generated.py"
COMMITTED_HPP = (
    DISPATCHER_DIR / "include" / "ck_tile" / "dispatcher" / "arch_specs_generated.hpp"
)


def _strip_timestamp(text: str) -> list:
    return [line for line in text.splitlines() if "Generated at:" not in line]


class TestArchSpecsRegeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="arch_specs_regen_")
        out = Path(cls._tmp.name)
        subprocess.run(
            [
                sys.executable,
                str(CODEGEN_DIR / "generate_arch_specs.py"),
                "--json",
                str(CODEGEN_DIR / "arch_specs.json"),
                "--output-dir",
                str(out),
                "--cpp-output-dir",
                str(out),
            ],
            check=True,
            capture_output=True,
        )
        cls.regen_py = out / "arch_specs_generated.py"
        cls.regen_hpp = out / "arch_specs_generated.hpp"

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _assert_same(self, regenerated: Path, committed: Path):
        got = _strip_timestamp(regenerated.read_text())
        want = _strip_timestamp(committed.read_text())
        self.assertEqual(
            got,
            want,
            f"{committed.name} does not match what generate_arch_specs.py emits "
            f"from arch_specs.json. Edit arch_specs.json or the generator and "
            f"regenerate instead of editing the generated file.",
        )

    def test_python_module_matches_committed(self):
        self._assert_same(self.regen_py, COMMITTED_PY)

    def test_cpp_header_matches_committed(self):
        if not (shutil.which("clang-format-18") or shutil.which("clang-format")):
            self.skipTest("clang-format not available; header is left unformatted")
        self._assert_same(self.regen_hpp, COMMITTED_HPP)


if __name__ == "__main__":
    unittest.main()
