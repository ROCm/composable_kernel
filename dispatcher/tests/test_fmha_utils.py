#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-only checks for FMHA compile flags and both JIT build paths."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import fmha_utils  # noqa: E402


FRAMEWORK_GATE = "-DUSE_NEW_UNIFIED_FRAMEWORK=0"
ARCH_CASES = (("gfx1250", 1), ("gfx1250:xnack-", 1), ("gfx942", 0), ("gfx950", 0))


class TestFmhaCompileFlags(unittest.TestCase):
    def test_framework_gate_in_forward_and_backward_base_flags(self):
        for arch, expected in ARCH_CASES:
            for family in ("fwd", "bwd"):
                with self.subTest(arch=arch, family=family):
                    flags = fmha_utils.fmha_compile_flags(arch, "hipcc", family)
                    self.assertEqual(flags.count(FRAMEWORK_GATE), expected)
                    self.assertIn(f"--offload-arch={arch}", flags)
                    self.assertEqual(
                        "-DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3" in flags,
                        family == "bwd",
                    )


class TestFmhaJitCompileCommands(unittest.TestCase):
    """Capture compiler invocations without requiring hipcc or a GPU."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        ctypes_src = self.root / "bindings" / "ctypes" / "fmha_ctypes_lib.cpp"
        ctypes_src.parent.mkdir(parents=True)
        ctypes_src.touch()
        self.compile_commands = []
        for name, value in (
            ("get_dispatcher_root", self.root),
            ("_find_hipcc", "hipcc"),
            ("_find_static_lib", self.root / "libck_tile_dispatcher.a"),
        ):
            patcher = patch.object(fmha_utils, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

        for name, side_effect in (("run", self._run), ("call", self._call)):
            patcher = patch.object(fmha_utils.subprocess, name, side_effect=side_effect)
            patcher.start()
            self.addCleanup(patcher.stop)
        loader = patch.object(fmha_utils.FmhaRunner, "from_library", return_value=object())
        loader.start()
        self.addCleanup(loader.stop)

    def _call(self, cmd, **kwargs):
        if "--config-json" in cmd:
            out = Path(cmd[cmd.index("--output-dir") + 1])
            (out / "fmha_python_dispatch.hpp").touch()
            (out / "fmha_kernel.cpp").touch()
        else:
            if "-c" in cmd:
                self.compile_commands.append(cmd)
            Path(cmd[cmd.index("-o") + 1]).touch()
        return 0

    def _run(self, cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, self._call(cmd, **kwargs), "", "")

    def _assert_kernel_and_ctypes_gate(self, arch, expected):
        self.assertEqual(len(self.compile_commands), 2)
        sources = []
        for cmd in self.compile_commands:
            self.assertEqual(cmd.count(FRAMEWORK_GATE), expected, cmd)
            self.assertIn(f"--offload-arch={arch}", cmd)
            sources.extend(Path(arg).name for arg in cmd if arg.endswith(".cpp"))
        self.assertCountEqual(sources, ["fmha_kernel.cpp", "fmha_ctypes_lib.cpp"])

    def test_single_kernel_jit_gates_both_translation_units(self):
        for arch, expected in ARCH_CASES:
            with self.subTest(arch=arch):
                self.compile_commands.clear()
                config = fmha_utils.FmhaKernelConfig(gfx_arch=arch)
                result = fmha_utils.setup_fmha_dispatcher(config, self.root / arch)
                self.assertTrue(result.success, result.error)
                self._assert_kernel_and_ctypes_gate(arch, expected)

    def test_multi_kernel_jit_gates_both_translation_units(self):
        executor = Mock()
        executor.map.side_effect = lambda worker, jobs: map(worker, jobs)
        for arch, expected in ARCH_CASES:
            with self.subTest(arch=arch):
                self.compile_commands.clear()
                config = fmha_utils.FmhaKernelConfig(gfx_arch=arch)
                results = fmha_utils.setup_multiple_fmha_dispatchers(
                    [config], self.root / arch, executor=executor
                )
                self.assertEqual(len(results), 1)
                self.assertTrue(results[0].success, results[0].error)
                self._assert_kernel_and_ctypes_gate(arch, expected)


if __name__ == "__main__":
    unittest.main()
