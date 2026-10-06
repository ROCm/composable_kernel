# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU checks for cache identity and current-tree integration dependencies."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import ctypes_utils as cu


@pytest.mark.parametrize("cached", [False, True])
def test_no_rebuild_uses_only_config_specific_cache(monkeypatch, tmp_path, cached):
    config = cu.KernelConfig(gfx_arch="gfx1250")
    name = "same_kernel_name_on_both_targets"
    generated = Mock()
    generated.generate_from_config.return_value = SimpleNamespace(
        success=True, instance_names=[name], stderr=""
    )
    monkeypatch.setattr(cu, "CodegenRunner", Mock(return_value=generated))
    monkeypatch.setattr(cu, "validate_kernel_config", lambda c: SimpleNamespace(is_valid=True))
    monkeypatch.setattr(cu, "get_build_dir", lambda: tmp_path)
    monkeypatch.setattr(cu, "get_generated_kernels_dir", lambda: tmp_path / "headers")
    legacy = Mock()
    legacy.get_kernel_name.return_value = name
    auto = Mock(return_value=legacy)
    monkeypatch.setattr(cu.DispatcherLib, "auto", auto)
    candidate = Mock()
    candidate.initialize.return_value = True
    candidate.get_kernel_name.return_value = name
    load = Mock(return_value=candidate)
    monkeypatch.setattr(cu.DispatcherLib, "load", load)
    monkeypatch.setattr(cu, "Registry", Mock())
    monkeypatch.setattr(cu, "Dispatcher", Mock())
    path = tmp_path / "examples" / cu._gemm_library_name(config)
    if cached:
        path.parent.mkdir()
        path.touch()

    result = cu.setup_gemm_dispatcher(config, auto_rebuild=False)

    auto.assert_not_called()
    generated._rebuild_library_for_config.assert_not_called()
    if cached:
        assert result.success
        assert result.lib is candidate
        load.assert_called_once_with(path)
    else:
        assert not result.success
        assert "enable auto_rebuild" in result.error
        load.assert_not_called()


@pytest.mark.parametrize("configured", [False, True])
def test_existing_static_archive_still_runs_build(monkeypatch, tmp_path, configured):
    # conftest's fixture and the script-style GPU tests share this helper.
    spec = importlib.util.spec_from_file_location("dependency_build", ROOT / "tests/dispatcher_build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "__file__", str(tmp_path / "tests/dispatcher_build.py"))
    build = tmp_path / "build"
    build.mkdir()
    archive = build / "libck_tile_dispatcher.a"
    archive.touch()
    if configured:
        (build / "CMakeCache.txt").touch()
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.ensure_dispatcher_static_lib() == archive
    assert commands[-1] == [
        "cmake", "--build", str(build), "--target", "ck_tile_dispatcher", "-j4"
    ]
    assert len(commands) == (1 if configured else 2)
    if not configured:
        assert commands[0][:2] == ["cmake", "-S"]
