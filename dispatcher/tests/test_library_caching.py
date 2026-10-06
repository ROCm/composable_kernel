#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Integration tests for bootstrapping and reusing a requested GEMM library."""

import sys
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codegen"))
import ctypes_utils as cu


@pytest.fixture(scope="module")
def cache_environment(dispatcher_static_lib, tmp_path_factory):
    # Do not delete or depend on libraries in a developer's normal build tree.
    build = tmp_path_factory.mktemp("gemm_cache")
    (build / "libck_tile_dispatcher.a").symlink_to(dispatcher_static_lib)
    (build / "generated_kernels").mkdir()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cu, "get_build_dir", lambda: build)
        yield build


@pytest.fixture
def config(gpu_arch, cache_environment):
    from arch_specs_generated import get_warp_tile_combos

    tiles = [tuple(t) for t in get_warp_tile_combos(gpu_arch.split(":", 1)[0], "fp16_fp16_fp32")]
    assert tiles, f"No fp16 warp tile for {gpu_arch}"
    wm, wn, wk = (32, 32, 16) if (32, 32, 16) in tiles else tiles[0]
    return cu.KernelConfig(
        tile_m=128, tile_n=128, tile_k=64,
        warp_m=wm, warp_n=wn, warp_k=wk,
        pipeline="compv4", gfx_arch=gpu_arch,
    )


def setup(config, **kwargs):
    result = cu.setup_gemm_dispatcher(config, **kwargs)
    assert result.success, result.error
    assert result.lib is not None
    assert result.lib.get_kernel_name() == result.kernel_header.stem
    return result


class TestLibraryCaching:
    def test_01_unique_library_naming(self, config):
        result = setup(config)
        name = result.lib.path.name
        for part in ("fp16", "rcr", "128x128x64", "2x2x1", config.gfx_arch,
                     f"{config.warp_m}x{config.warp_n}x{config.warp_k}",
                     "compv4", "cshuffle", "intrawave"):
            assert part in name

    def test_02_library_build_and_cache(self, config):
        first = setup(config)
        before = first.lib.path.stat().st_mtime_ns
        second = setup(config)
        assert second.lib.path == first.lib.path
        assert second.lib.path.stat().st_mtime_ns == before

    def test_03_different_configs_different_libraries(self, config):
        first = setup(config)
        second = setup(replace(config, tile_k=32))
        assert first.lib.path != second.lib.path
        assert first.lib.get_kernel_name() != second.lib.get_kernel_name()

    def test_04_cache_message_verification(self, config, capsys):
        setup(config)
        capsys.readouterr()
        setup(config, verbose=True)
        assert "Using cached library" in capsys.readouterr().out

    def test_arch_and_padding_distinguish_libraries(self):
        config = cu.KernelConfig(gfx_arch="gfx950")
        names = {
            cu._gemm_library_name(config),
            cu._gemm_library_name(replace(config, gfx_arch="gfx1250")),
            cu._gemm_library_name(replace(config, pad_k=False)),
            cu._gemm_library_name(replace(config, warp_m=16)),
        }
        assert len(names) == 4

    def test_failed_rebuild_does_not_use_default_library(self, config, monkeypatch):
        config = replace(config, tile_n=64)
        monkeypatch.setattr(cu.CodegenRunner, "_rebuild_library_for_config", lambda *a: None)
        monkeypatch.setattr(
            cu.DispatcherLib, "auto",
            lambda: pytest.fail("A failed build must not fall back to a different kernel"),
        )
        result = cu.setup_gemm_dispatcher(config, auto_rebuild=True)
        assert not result.success
        assert result.lib is None
        assert "Failed to build" in result.error


def test_cmake_fallback_build_and_run(gpu_arch, tmp_path):
    """The standalone CMake target must build and compute on the running GPU."""
    from gemm_utils import GemmProblem, GpuGemmRunner

    root = Path(__file__).resolve().parents[1]
    build = tmp_path / "cmake"
    commands = [
        ["cmake", "-S", str(root), "-B", str(build),
         f"-DCMAKE_CXX_COMPILER={shutil.which('hipcc') or '/opt/rocm/bin/hipcc'}",
         "-DCMAKE_BUILD_TYPE=Release", f"-DGPU_TARGETS={gpu_arch}",
         "-DBUILD_DISPATCHER_EXAMPLES=ON"],
        ["cmake", "--build", str(build), "--target", "dispatcher_gemm_lib", "-j2"],
    ]
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, result.stdout + result.stderr

    runner = GpuGemmRunner(build / "examples" / "libdispatcher_gemm_lib.so")
    rng = np.random.default_rng(11042)
    for M, N, K in ((128, 128, 128), (256, 128, 256)):
        A = rng.uniform(-1, 1, (M, K)).astype(np.float16).astype(np.float32)
        B = rng.uniform(-1, 1, (K, N)).astype(np.float16).astype(np.float32)
        result = runner.run(A, B, GemmProblem(M=M, N=N, K=K))
        assert result.status == 0
        reference = A @ B
        error = np.max(np.abs(result.output.astype(np.float32) - reference)) / np.max(np.abs(reference))
        assert np.isfinite(error) and error < 2e-3, error


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
