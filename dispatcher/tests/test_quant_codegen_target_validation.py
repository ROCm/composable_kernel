# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The block-scale quant codegen CLIs apply the same target rules as the bridges.

The CLIs are a second way to produce kernels, next to the Python bridges, so an
unsupported arch, an i4 variant on gfx1250 or a non-WMMA warp tile on gfx1250
must fail there too instead of writing headers that compile into wrong results.
"""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

DISPATCHER = Path(__file__).resolve().parents[1]
CODEGEN = DISPATCHER / "codegen"
sys.path.insert(0, str(DISPATCHER / "python"))
sys.path.insert(0, str(CODEGEN))

OPS = ("aquant", "bquant", "abquant", "rowcolquant")
MODULES = {op: importlib.import_module(f"unified_gemm_{op}_codegen") for op in OPS}


def _cli(op, *args):
    return subprocess.run(
        [sys.executable, str(CODEGEN / f"unified_gemm_{op}_codegen.py"), *args],
        capture_output=True, text=True, timeout=120,
    )


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("arch", ("gfx942", "gfx950", "gfx1250"))
def test_default_config_is_valid_for_its_target(op, arch):
    module = MODULES[op]
    config = module._default_config(arch)
    module._validate_target_config(config, arch)
    specs = module._build_specs(config)
    assert specs, f"{op}: default config for {arch} produced no kernels"
    if arch == "gfx1250":
        assert all("i4" not in s.variant_key for s in specs), (
            f"{op}: gfx1250 default config still emits an i4 variant"
        )
        assert all(t["warp_tile_k"] in (64, 128) for t in config["tile_configs"])


@pytest.mark.parametrize("op", OPS)
def test_cli_rejects_unsupported_target(op):
    result = _cli(op, "--list-names", "--gfx-arch", "gfx1201")
    assert result.returncode == 1, result.stdout + result.stderr
    assert not result.stdout.strip(), "no kernel names may be listed for a rejected target"


@pytest.mark.parametrize("op", OPS)
def test_cli_rejects_non_wmma_warp_tile_on_gfx1250(op):
    config = MODULES[op]._default_config("gfx942")
    assert {t["warp_tile_k"] for t in config["tile_configs"]} == {32}
    result = _cli(op, "--list-names", "--gfx-arch", "gfx1250", "--config-json", json.dumps(config))
    assert result.returncode == 1, result.stdout + result.stderr


@pytest.mark.parametrize("op", OPS)
def test_cli_rejects_recorded_arch_mismatch(op):
    config = dict(MODULES[op]._default_config("gfx1250"), gfx_arch="gfx942")
    result = _cli(op, "--list-names", "--gfx-arch", "gfx1250", "--config-json", json.dumps(config))
    assert result.returncode == 1, result.stdout + result.stderr


def test_aquant_cli_rejects_i4_on_gfx1250():
    config = MODULES["aquant"]._default_config("gfx950")
    assert any("i4" in s.variant_key for s in MODULES["aquant"]._build_specs(config))
    result = _cli("aquant", "--list-names", "--gfx-arch", "gfx1250",
                  "--config-json", json.dumps(dict(config, **{
                      k: v for k, v in MODULES["aquant"]._default_config("gfx1250").items()
                      if k != "variant_keys"})))
    assert result.returncode == 1, result.stdout + result.stderr


def test_bquant_cli_rejects_mx_off_gfx950():
    module = MODULES["bquant"]
    mx = sorted(module.MX_VARIANTS)[0]
    config = dict(module._default_config("gfx942"), variant_keys=[mx])
    result = _cli("bquant", "--list-names", "--gfx-arch", "gfx942", "--config-json", json.dumps(config))
    assert result.returncode == 1, result.stdout + result.stderr


@pytest.mark.parametrize("op", OPS)
def test_cli_default_gfx950_listing_is_unchanged(op):
    module = MODULES[op]
    result = _cli(op, "--list-names")
    assert result.returncode == 0, result.stderr
    expected = [s.name for s in module._build_specs(module._default_config("gfx950"))]
    assert result.stdout.split() == expected
