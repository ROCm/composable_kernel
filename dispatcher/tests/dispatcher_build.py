# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Build the dispatcher static library that registry-routed GPU tests link against.

Shared by the pytest ``dispatcher_static_lib`` fixture (conftest.py) and by the
script / unittest entry points ctest runs directly (run_unittest_77.py and the
``main()`` of the script-style GPU tests). Without it those entry points only
check whether ``libck_tile_dispatcher.a`` already exists, so a clean ctest run
skips them forever instead of building the prerequisite and testing.
"""

import shutil
import subprocess
from pathlib import Path

_ARCHIVE = None


def ensure_dispatcher_static_lib() -> Path:
    """Configure (if needed) and build ``ck_tile_dispatcher``; return the archive.

    Raises RuntimeError when the build fails or does not produce the archive, so
    callers report a failure rather than a skip. The result is cached per process.
    """
    global _ARCHIVE
    if _ARCHIVE is not None:
        return _ARCHIVE

    root = Path(__file__).resolve().parents[1]
    build = root / "build"
    archive = build / "libck_tile_dispatcher.a"
    hipcc = shutil.which("hipcc") or "/opt/rocm/bin/hipcc"
    commands = []
    if not (build / "CMakeCache.txt").exists():
        commands.append([
            "cmake", "-S", str(root), "-B", str(build),
            f"-DCMAKE_CXX_COMPILER={hipcc}", "-DCMAKE_BUILD_TYPE=Release",
        ])
    # An existing archive may belong to an earlier checkout. Let CMake decide
    # whether its dependency graph is current, even when the file exists.
    commands.append([
        "cmake", "--build", str(build), "--target", "ck_tile_dispatcher", "-j4",
    ])
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(
                f"{' '.join(command)} failed:\n{result.stdout}{result.stderr}"
            )
    if not archive.exists():
        raise RuntimeError(f"Missing dispatcher dependency: {archive}")
    _ARCHIVE = archive
    return archive
