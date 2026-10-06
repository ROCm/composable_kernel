# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""MX bridge CMake target and compiler-definition tests; no GPU required."""

import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

_DISP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_DISP / "python"))

from dispatcher_common import arch_feature_defines, unified_framework_flags  # noqa: E402


class TestMxGemmCMake(unittest.TestCase):
    @unittest.skipUnless(
        shutil.which("cmake") and shutil.which("c++"),
        "requires CMake and a host C++ compiler",
    )
    def test_cmake_mx_target_and_effective_features(self):
        # Configure the real bindings CMake with an imported HIP target; no GPU is needed.
        for variable in ("GPU_TARGETS", "CK_TILE_GEMM_GPU_TARGET"):
            for arch in ("gfx950:sramecc+:xnack-", "gfx1250:xnack-", "gfx942"):
                for has_header in (False, True):
                    with (
                        self.subTest(variable=variable, arch=arch, header=has_header),
                        tempfile.TemporaryDirectory() as tmp,
                    ):
                        tmp = Path(tmp)
                        build = tmp / "build"
                        if has_header:
                            headers = build / "generated_kernels"
                            headers.mkdir(parents=True)
                            (headers / "mx_gemm_test.hpp").touch()
                        (tmp / "CMakeLists.txt").write_text(
                            "cmake_minimum_required(VERSION 3.16)\n"
                            "project(mx_cmake_probe LANGUAGES CXX)\n"
                            "add_library(hip::device INTERFACE IMPORTED)\n"
                            # A conflicting inherited definition must not leak into this target.
                            "add_compile_definitions(CK_USE_FNUZ_FP8 CK_TILE_USE_WMMA=0)\n"
                            f'set({variable} "{arch}")\n'
                            f'add_subdirectory("{_DISP / "bindings/ctypes"}" ctypes)\n'
                        )
                        result = subprocess.run(
                            [
                                "cmake",
                                "-S",
                                str(tmp),
                                "-B",
                                str(build),
                                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                            ],
                            capture_output=True,
                            text=True,
                            timeout=60,
                        )
                        self.assertEqual(
                            result.returncode, 0, result.stdout + result.stderr
                        )
                        commands = json.loads(
                            (build / "compile_commands.json").read_text()
                        )
                        matches = [
                            c
                            for c in commands
                            if c["file"].endswith("/mx_gemm_ctypes_lib.cpp")
                        ]
                        base_arch = arch.split(":", 1)[0]
                        if base_arch == "gfx942":
                            self.assertFalse(matches)
                            continue
                        self.assertEqual(len(matches), 1)
                        command = shlex.split(matches[0]["command"])
                        self.assertIn(f'-DGFX_ARCH="{base_arch}"', command)
                        self.assertIn("-std=gnu++17", command)
                        self.assertEqual(
                            "-DCK_TILE_SINGLE_KERNEL_INCLUDE" in command, has_header
                        )
                        flags = [
                            arg for arg in command[1:] if arg.startswith(("-D", "-U"))
                        ]
                        preprocessed = subprocess.run(
                            [command[0], *flags, "-dM", "-E", "-x", "c++", "-"],
                            input="",
                            capture_output=True,
                            text=True,
                            check=True,
                        )
                        macros = {
                            line.split()[1]: " ".join(line.split()[2:])
                            for line in preprocessed.stdout.splitlines()
                        }
                        self.assertNotIn("CK_USE_FNUZ_FP8", macros)
                        for flag in arch_feature_defines(
                            base_arch
                        ) + unified_framework_flags(base_arch):
                            key, _, value = flag[2:].partition("=")
                            self.assertEqual(macros.get(key), value or "1", key)


if __name__ == "__main__":
    unittest.main()
