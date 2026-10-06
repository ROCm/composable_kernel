# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The gfx1250 tile_engine block-scale configs must stay buildable on gfx1250.

default_config_gfx1250.json is not wired into a CMake target, so nothing else
compiles it. These checks keep it on the WMMA fragments the bridges accept and
in step with the schema of default_config.json.
"""

import itertools
import json
import sys
from pathlib import Path

import pytest

DISPATCHER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DISPATCHER / "codegen"))

from codegen_common import validate_gfx1250_quant_warp_tile  # noqa: E402

CONFIG_ROOT = DISPATCHER.parent / "tile_engine" / "ops" / "gemm" / "block_scale_gemm"
OPS = ("aquant", "bquant", "abquant", "rowcolquant", "tensor_quant")


def _load(op, name):
    path = CONFIG_ROOT / f"gemm_{op}" / "configs" / name
    if not path.exists():
        pytest.skip(f"{path} not present in this checkout")
    with open(path) as f:
        return json.load(f)


def _values(entry):
    if "values" in entry:
        return list(entry["values"])
    return list(range(entry["min"], entry["max"] + 1, entry["step"]))


@pytest.mark.parametrize("op", OPS)
def test_schema_matches_default_config(op):
    base = _load(op, "default_config.json")
    gfx1250 = _load(op, "default_config_gfx1250.json")
    assert sorted(gfx1250) == sorted(base), f"{op}: top-level keys differ"
    for section in ("tile_config", "trait_config"):
        assert sorted(gfx1250[section]) == sorted(base[section]), (
            f"{op}: {section} keys differ from default_config.json"
        )
    for key, entry in gfx1250["trait_config"].items():
        extra = set(_values(entry)) - set(_values(base["trait_config"][key]))
        assert not extra, f"{op}: trait {key} adds values {extra} not in default_config.json"
    for key in ("group_size_k", "k_block_per_cu"):
        if key in base:
            assert gfx1250[key] == base[key], f"{op}: {key} differs from default_config.json"


@pytest.mark.parametrize("op", OPS)
def test_warp_tiles_are_gfx1250_wmma_fragments(op):
    tile = _load(op, "default_config_gfx1250.json")["tile_config"]
    for m, n, k in itertools.product(
        _values(tile["warp_tile_m"]), _values(tile["warp_tile_n"]), _values(tile["warp_tile_k"])
    ):
        # Raises for any fragment the bridge would reject on gfx1250.
        validate_gfx1250_quant_warp_tile(
            m, n, k, "gfx1250", bridge=op, logical_k32=(op == "tensor_quant")
        )


@pytest.mark.parametrize("op", OPS)
def test_every_warp_layout_has_a_divisible_block_tile(op):
    config = _load(op, "default_config_gfx1250.json")
    tile = config["tile_config"]
    tiles = {dim: _values(tile[f"tile_{dim}"]) for dim in "mnk"}
    for dim in "mn":
        for warps, warp_tile in itertools.product(
            _values(tile[f"warp_{dim}"]), _values(tile[f"warp_tile_{dim}"])
        ):
            step = warps * warp_tile
            assert any(t % step == 0 for t in tiles[dim]), (
                f"{op}: no tile_{dim} in {tiles[dim]} is a multiple of "
                f"warp_{dim}*warp_tile_{dim} = {warps}*{warp_tile}"
            )
    group_k = config.get("group_size_k")
    for warp_tile_k in _values(tile["warp_tile_k"]):
        fits = [t for t in tiles["k"] if t % warp_tile_k == 0
                and (group_k is None or t % group_k == 0 or group_k % t == 0)]
        assert fits, (
            f"{op}: no tile_k in {tiles['k']} fits warp_tile_k={warp_tile_k}"
            + (f" and group_size_k={group_k}" if group_k else "")
        )
