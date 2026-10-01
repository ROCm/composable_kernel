# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host regressions for kernel targets and generated-header selection.

- The ABQuant EightWaves pipeline only exists on gfx950. On any other target
  the kernel compiles to an empty body, so the config must be rejected.
- A serialized quant bridge config records its target, so the codegen CLI can
  not generate it for a different arch.
- The multi-config GEMM codegen worker returns the header it generated, not
  another header in the shared output directory that differs only in padding.
"""

import dataclasses
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
CODEGEN = ROOT / "codegen"
sys.path[:0] = [str(ROOT / "python"), str(CODEGEN)]

import ctypes_utils as cu  # noqa: E402
import gemm_abquant_utils as ab  # noqa: E402
from codegen_common import validate_abquant_eight_waves_target  # noqa: E402

OPS = ("aquant", "bquant", "abquant", "rowcolquant")
BRIDGES = {op: importlib.import_module(f"gemm_{op}_utils") for op in OPS}
ABQUANT_CODEGEN = importlib.import_module("unified_gemm_abquant_codegen")


def _cli(op, *args):
    return subprocess.run(
        [sys.executable, str(CODEGEN / f"unified_gemm_{op}_codegen.py"), *args],
        capture_output=True, text=True, timeout=120,
    )


def _eight_waves_config():
    config = ab.default_fp8_config(bquant_group_n=128, gfx_arch="gfx950")
    assert config.pipeline == "eightwaves" and config.eight_waves
    return config


# -----------------------------------------------------------------------------
# ABQuant EightWaves is gfx950-only
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ("gfx942", "gfx1250", "gfx1250:xnack-"))
@pytest.mark.parametrize("pipeline, eight_waves", [
    ("eightwaves", True), ("eightwaves", False), ("compv3", True)])
def test_eight_waves_helper_rejects_non_gfx950(arch, pipeline, eight_waves):
    with pytest.raises(ValueError, match="EightWaves pipeline requires gfx950"):
        validate_abquant_eight_waves_target(pipeline, eight_waves, arch)


@pytest.mark.parametrize("arch", ("gfx950", "gfx950:xnack-", "", None))
def test_eight_waves_helper_accepts_gfx950_and_unresolved(arch):
    validate_abquant_eight_waves_target("eightwaves", True, arch)


@pytest.mark.parametrize("arch", ("gfx942", "gfx950", "gfx1250"))
def test_eight_waves_helper_leaves_compv3_alone(arch):
    validate_abquant_eight_waves_target("compv3", False, arch)


@pytest.mark.parametrize("arch", ("gfx942", "gfx1250", "gfx1250:xnack-"))
def test_bridge_rejects_eight_waves_off_gfx950(arch):
    config = _eight_waves_config()
    with pytest.raises(ValueError, match="EightWaves pipeline requires gfx950"):
        dataclasses.replace(config, gfx_arch=arch)
    # Retargeting after construction is caught before codegen.
    config.gfx_arch = arch
    with pytest.raises(ValueError, match="EightWaves pipeline requires gfx950"):
        config.to_codegen_config()


def test_bridge_keeps_eight_warp_compv3_on_gfx1250():
    # Same 4x2x1 warp layout as EightWaves, but on the CompV3 pipeline.
    config = dataclasses.replace(_eight_waves_config(), pipeline="compv3",
                                 eight_waves=False, gfx_arch="gfx1250")
    assert config.warp_m * config.warp_n * config.warp_k == 8
    assert config.to_codegen_config()["gfx_arch"] == "gfx1250"


@pytest.mark.parametrize("arch", ("gfx942", "gfx1250"))
def test_codegen_rejects_eight_waves_off_gfx950(arch):
    data = dict(_eight_waves_config().to_codegen_config(), gfx_arch=arch)
    with pytest.raises(ValueError, match="EightWaves pipeline requires gfx950"):
        ABQUANT_CODEGEN._validate_target_config(data, arch)
    result = _cli("abquant", "--list-names", "--gfx-arch", arch, "--config-json", json.dumps(data))
    assert result.returncode == 1, result.stdout + result.stderr
    assert not result.stdout.strip()


def test_codegen_accepts_eight_waves_on_gfx950():
    data = _eight_waves_config().to_codegen_config()
    ABQUANT_CODEGEN._validate_target_config(data, "gfx950")


# -----------------------------------------------------------------------------
# Serialized bridge configs keep their target
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("arch", ("gfx942", "gfx950", "gfx1250"))
def test_codegen_config_records_target(op, arch):
    config = BRIDGES[op].default_fp8_config(gfx_arch=arch)
    assert config.to_codegen_config()["gfx_arch"] == arch


@pytest.mark.parametrize("op", OPS)
def test_cli_rejects_retargeted_bridge_config(op, tmp_path):
    data = json.dumps(BRIDGES[op].default_fp8_config(gfx_arch="gfx950").to_codegen_config())
    out = tmp_path / "gfx1250"
    result = _cli(op, "--gfx-arch", "gfx1250", "--output-dir", str(out), "--config-json", data)
    assert result.returncode == 1, result.stdout + result.stderr
    assert not list(out.glob("*.hpp"))

    out = tmp_path / "gfx950"
    result = _cli(op, "--gfx-arch", "gfx950", "--output-dir", str(out), "--config-json", data)
    assert result.returncode == 0, result.stdout + result.stderr
    assert list(out.glob("*.hpp"))


@pytest.mark.parametrize("op", OPS)
def test_cli_uses_recorded_target_without_flag(op, tmp_path):
    # The bridges run codegen without --gfx-arch, so the recorded target must
    # be the one validated instead of the CLI default.
    config = BRIDGES[op].default_fp8_config(gfx_arch="gfx1250")
    result = _cli(op, "--output-dir", str(tmp_path), "--config-json",
                  json.dumps(config.to_codegen_config()))
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / f"{config.name}.hpp").exists()


# -----------------------------------------------------------------------------
# Multi-config worker picks its own header
# -----------------------------------------------------------------------------


def _worker_args(out_dir, pad_m, codegen_script=CODEGEN / "unified_gemm_codegen.py"):
    config = cu.KernelConfig(
        gfx_arch="gfx1250", dtype_a="fp16", dtype_b="fp16", dtype_c="fp16",
        tile_m=128, tile_n=128, tile_k=64, wave_m=2, wave_n=2, wave_k=1,
        warp_m=16, warp_n=16, warp_k=32, pipeline="compv3", epilogue="cshuffle",
        scheduler="intrawave", pad_m=pad_m,
    )
    assert cu.validate_kernel_config(config).is_valid
    tile_config = {
        "tile_config": {
            "tile_m": [config.tile_m], "tile_n": [config.tile_n], "tile_k": [config.tile_k],
            "warp_m": [config.wave_m], "warp_n": [config.wave_n], "warp_k": [config.wave_k],
            "warp_tile_m": [config.warp_m], "warp_tile_n": [config.warp_n],
            "warp_tile_k": [config.warp_k],
        },
        "trait_config": {
            "pipeline": [config.pipeline], "epilogue": [config.epilogue],
            "scheduler": [config.scheduler], "pad_m": [config.pad_m],
            "pad_n": [config.pad_n], "pad_k": [config.pad_k], "persistent": [False],
        },
    }
    pattern = (
        f"gemm_{config.dtype_a}_{config.layout}_{config.pipeline}_{config.epilogue}"
        f"_{config.scheduler}_*_{config.tile_str}_{config.wave_m}x{config.wave_n}x{config.wave_k}"
        f"_{config.warp_m}x{config.warp_n}x{config.warp_k}.hpp"
    )
    return {
        "python": sys.executable, "codegen_script": str(codegen_script),
        "output_dir": str(out_dir), "dtype": config.dtype_a, "layout": config.layout,
        "gpu_target": config.gfx_arch, "tile_config_json": tile_config,
        "hpp_glob_pattern": pattern,
    }


@pytest.mark.parametrize("order", [(False, True), (True, False)])
def test_worker_returns_its_own_header_in_shared_dir(order, tmp_path):
    selected = []
    for pad_m in order:
        ok, path, error = cu._generate_single_kernel_subprocess(_worker_args(tmp_path, pad_m))
        assert ok, error
        header = Path(path)
        assert header.parent == tmp_path and header.exists()
        assert cu._parse_gemm_header_metadata(header)["pad_m"] is pad_m
        selected.append(header.name)
    assert selected[0] != selected[1]
    assert sorted(p.name for p in tmp_path.glob("*.hpp")) == sorted(selected)
    assert not [p for p in tmp_path.iterdir() if p.is_dir()], "private codegen dir left behind"


def test_worker_rejects_ambiguous_codegen_output(tmp_path):
    fake = tmp_path / "fake_codegen.py"
    fake.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "out = Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
        "for pad in ('True', 'False'):\n"
        "    (out / f'gemm_fp16_rcr_compv3_cshuffle_intrawave_{pad}_True_True_False"
        "_128x128x64_2x2x1_16x16x32.hpp').write_text('')\n"
    )
    out = tmp_path / "out"
    out.mkdir()
    ok, path, error = cu._generate_single_kernel_subprocess(
        _worker_args(out, True, codegen_script=fake))
    assert not ok and path is None
    assert "Expected one .hpp" in error
    assert not list(out.iterdir())
