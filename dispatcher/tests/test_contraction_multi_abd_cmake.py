# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Configure the real operator without ROCm and inspect its generated targets."""

import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys

import pytest

CK_ROOT = Path(__file__).resolve().parents[2]
OP_DIR = CK_ROOT / "tile_engine/ops/gemm/contraction_multi_abd"
pytestmark = pytest.mark.skipif(shutil.which("cmake") is None, reason="requires CMake")


@pytest.mark.parametrize("targets,families", [
    ("gfx942;gfx1250", {"gfx9": ["gfx942"], "gfx1250": ["gfx1250"]}),
    ("gfx1250;gfx950;gfx942", {"gfx9": ["gfx950", "gfx942"], "gfx1250": ["gfx1250"]}),
    ("gfx942;gfx950", {"gfx9": ["gfx942", "gfx950"]}),
    ("gfx1250", {"gfx1250": ["gfx1250"]}),
    ("gfx942:xnack-", {"gfx9": ["gfx942:xnack-"]}),
    ("gfx950:xnack-", {"gfx9": ["gfx950:xnack-"]}),
    ("gfx942:sramecc+:xnack-;gfx950:xnack-;gfx1250",
     {"gfx9": ["gfx942:sramecc+:xnack-", "gfx950:xnack-"], "gfx1250": ["gfx1250"]}),
    ("gfx1250;gfx942:xnack-;gfx942:xnack+",
     {"gfx9": ["gfx942:xnack-", "gfx942:xnack+"], "gfx1250": ["gfx1250"]}),
])
def test_family_codegen_and_compiler_targets(tmp_path, targets, families):
    source = tmp_path / "source"
    build = tmp_path / "build"
    source.mkdir()
    # Model the global hip::device dependency to catch accidental inheritance
    # of the project-wide target list; no actual compiler invocation is needed.
    (source / "CMakeLists.txt").write_text(f'''cmake_minimum_required(VERSION 3.21)
project(contraction_family_test LANGUAGES CXX)
set(Python3_EXECUTABLE "{sys.executable}")
set(SUPPORTED_GPU_TARGETS "{targets}")
set(CONTRACTION_MULTI_ABD_MAX_INSTANCES 1 CACHE STRING "")
set(CONTRACTION_MULTI_ABD_DATATYPE "fp16;bf16" CACHE STRING "")
add_library(hip::device INTERFACE IMPORTED)
add_library(hip::host INTERFACE IMPORTED)
set_property(TARGET hip::device PROPERTY INTERFACE_COMPILE_OPTIONS "--offload-arch=WRONG")
link_libraries(hip::device)
add_subdirectory("{OP_DIR}" op EXCLUDE_FROM_ALL)
get_property(targets DIRECTORY "{OP_DIR}" PROPERTY BUILDSYSTEM_TARGETS)
foreach(t IN LISTS targets)
    get_target_property(kind ${{t}} TYPE)
    if(kind STREQUAL "EXECUTABLE")
        file(GENERATE OUTPUT "${{CMAKE_BINARY_DIR}}/${{t}}.txt" CONTENT
"$<TARGET_PROPERTY:${{t}},HIP_ARCHITECTURES>\n$<TARGET_PROPERTY:${{t}},LINK_OPTIONS>\n$<TARGET_PROPERTY:${{t}},LINK_LIBRARIES>\n")
    endif()
endforeach()
''')
    env = dict(os.environ)
    env.pop("CONTRACTION_MULTI_ABD_CONFIG_FILE", None)
    result = subprocess.run([
        "cmake", "-S", str(source), "-B", str(build),
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    ], capture_output=True, text=True, env=env, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    commands = json.loads((build / "compile_commands.json").read_text())
    assert len(commands) == 2 * len(families)
    for family, archs in families.items():
        assert f"codegen arch: {archs[0].split(':')[0]}\n" in result.stdout
        family_dir = build / "op" / family if len(families) > 1 else build / "op"
        for dtype in ("fp16", "bf16"):
            headers = list((family_dir / dtype / "rcr").glob("*.hpp"))
            assert len(headers) == 1
            header = headers[0]
            tile = "16x16x32" if family == "gfx1250" else "16x16x16"
            assert tile in header.stem
            prefix = family + "_" if len(families) > 1 else ""
            name = "benchmark_contraction_multi_abd_" + prefix + header.stem
            props = (build / (name + ".txt")).read_text().splitlines()
            assert props[0].split(";") == archs
            assert "hip::device" not in props[2]
            assert "hip::host" in props[2]
            command = next(x["command"] for x in commands if str(header) in x["command"])
            assert "WRONG" not in command
            # Exact tokens catch both dropped feature suffixes and accidental
            # inheritance of another family's target flags.
            expected = [f"--offload-arch={arch}" for arch in archs]
            compile_flags = [arg for arg in shlex.split(command)
                             if arg.startswith("--offload-arch=")]
            link_flags = [arg for arg in props[1].split(";")
                          if arg.startswith("--offload-arch=")]
            assert compile_flags == expected
            assert link_flags == expected
    # EXCLUDE_FROM_ALL must leave the default build empty and successful.
    result = subprocess.run(["cmake", "--build", str(build)], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("targets,dtype,layout,error", [
    ("gfx942;gfx1250", "fp8", "rcr", "not supported on gfx1250"),
    ("gfx1250", "bf8", "rcr", "not supported on gfx1250"),
    ("gfx90a", "fp16", "rrr", "Only 'rcr'"),
    ("gfx942", "fp16", "crr", "Only 'rcr'"),
    ("gfx950", "fp16", "ccr", "Only 'rcr'"),
    ("gfx1250", "fp16", "rrr", "Only 'rcr'"),
])
def test_unsupported_surface_fails_at_configure(tmp_path, targets, dtype, layout, error):
    (tmp_path / "CMakeLists.txt").write_text(f'''cmake_minimum_required(VERSION 3.21)
project(contraction_gate_test LANGUAGES CXX)
set(Python3_EXECUTABLE "{sys.executable}")
set(SUPPORTED_GPU_TARGETS "{targets}")
set(CONTRACTION_MULTI_ABD_DATATYPE "{dtype}" CACHE STRING "")
set(CONTRACTION_MULTI_ABD_LAYOUT "{layout}" CACHE STRING "")
set(CONTRACTION_MULTI_ABD_MAX_INSTANCES 1 CACHE STRING "")
add_library(hip::host INTERFACE IMPORTED)
add_subdirectory("{OP_DIR}" op EXCLUDE_FROM_ALL)
''')
    env = dict(os.environ)
    env.pop("CONTRACTION_MULTI_ABD_CONFIG_FILE", None)
    result = subprocess.run(["cmake", "-S", str(tmp_path), "-B", str(tmp_path / "build")],
                            capture_output=True, text=True, env=env, timeout=60)
    assert result.returncode != 0
    assert error in " ".join((result.stdout + result.stderr).split())
