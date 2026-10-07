#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
UnifiedGemmCodegen must resolve the arch-validation dtype triple exactly: a
mapped --datatype (fp32) is validated as itself, and an unmapped one (pk_fp4)
raises naming the value instead of being silently validated as fp16.
--show-arch-info prints the warp tiles listed under that triple, and the
gfx1250 fp32 gates drop configs inside _get_configs_for_variant.

Run: python3 -m pytest -q tests/test_codegen_dtype_resolution.py
"""

import sys
from pathlib import Path

import pytest

DISPATCHER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))
sys.path.insert(0, str(DISPATCHER_DIR / "python"))

import unified_gemm_codegen as ugc  # noqa: E402
from arch_filter import ELEMENT_SIZE_MAP, WARP_TILE_SUPPORTED_COMBINATIONS  # noqa: E402
from codegen_common import CommonTypeMappings, TileConfig  # noqa: E402
from ctypes_utils import KernelConfig, validate_kernel_config  # noqa: E402


class _RecordingFilter:
    """Stands in for ArchFilter and records the dtypes it is asked about."""

    def __init__(self):
        self.calls = []

    def is_kernel_valid(self, **kwargs):
        self.calls.append(kwargs)
        return True


def _codegen(tmp_path, datatype):
    return ugc.UnifiedGemmCodegen(
        output_dir=tmp_path, datatype=datatype, layout="rcr", gpu_target="gfx942"
    )


def test_unmapped_datatype_raises_naming_value(tmp_path):
    with pytest.raises(ValueError, match="'pk_fp4'"):
        _codegen(tmp_path, "pk_fp4")


def test_show_arch_info_unmapped_datatype_raises():
    with pytest.raises(ValueError, match="'pk_fp4'"):
        ugc._show_arch_info("gfx942", "pk_fp4")


@pytest.mark.skipif(not ugc.HAS_ARCH_FILTER, reason="arch_filter not importable")
@pytest.mark.parametrize(
    "gpu_target,datatype,key",
    [
        ("gfx1250:xnack-", "fp32", "fp32_fp32_fp32"),
        ("gfx942", "fp16", "fp16_fp16_fp32"),
    ],
)
def test_show_arch_info_prints_the_listed_warp_tiles(capsys, gpu_target, datatype, key):
    # Keyed by the accumulator triple and looked up with the suffix stripped.
    arch = gpu_target.split(":")[0]
    ugc._show_arch_info(gpu_target, datatype)
    out = capsys.readouterr().out
    assert f"Architecture Info for {arch} ===" in out
    block = out.split(f"Warp tile configurations for {key} ")[1].split("\n\n")[0]
    printed = [line.strip() for line in block.splitlines()[1:]]
    expected = [str(t) for t in WARP_TILE_SUPPORTED_COMBINATIONS[arch][key]]
    assert printed and printed == expected
    assert arch != "gfx1250" or printed == ["[16, 16, 4]"]


@pytest.mark.skipif(not ugc.HAS_ARCH_FILTER, reason="arch_filter not importable")
@pytest.mark.parametrize("layout", ["rcr", "rrr"])
def test_gfx1250_fp32_configs_apply_both_gates(tmp_path, layout):
    # The codegen drops the 256x256 / 4-wave tile (accumulator spill) for every
    # pipeline and comp_tdm_v2 outside rcr (TDM fp32 layout rule).
    codegen = ugc.UnifiedGemmCodegen(
        output_dir=tmp_path, datatype="fp32", layout=layout, gpu_target="gfx1250"
    )
    codegen.arch_filter = _RecordingFilter()
    tiles = [TileConfig(t, t, 32, 2, 2, 1, 16, 16, 4) for t in (256, 128)]
    traits = [
        ugc.TraitConfig(pipe, epi, "intrawave", False, False, False)
        for pipe, epi in (("compv3", "cshuffle"), ("comp_tdm_v2", "tdm"))
    ]
    codegen._get_tile_configs = lambda: tiles
    codegen._get_trait_configs = lambda: traits

    configs = codegen._get_configs_for_variant(ugc.GemmVariant.STANDARD)

    got = {(c.tile.tile_m, c.trait.pipeline) for c in configs}
    tdm = {(128, "comp_tdm_v2")} if layout == "rcr" else set()
    assert got == {(128, "compv3")} | tdm


@pytest.mark.skipif(not ugc.HAS_ARCH_FILTER, reason="arch_filter not importable")
def test_fp32_validated_as_fp32(tmp_path):
    codegen = _codegen(tmp_path, "fp32")
    codegen.arch_filter = _RecordingFilter()
    tile = TileConfig(128, 128, 32, 2, 2, 1, 32, 32, 8)

    assert codegen._is_tile_arch_valid(tile)

    (call,) = codegen.arch_filter.calls
    triple = (call["datatype_a"], call["datatype_b"], call["datatype_c"])
    assert triple == ("fp32", "fp32", "fp32")
    assert ELEMENT_SIZE_MAP.get(call["datatype_a"], 2) == 4
    assert ELEMENT_SIZE_MAP.get(call["datatype_b"], 2) == 4


@pytest.mark.parametrize("dtype", CommonTypeMappings.ARCH_VALIDATION_DTYPES)
def test_every_mapped_dtype_resolves_to_itself(dtype):
    a, b, acc = CommonTypeMappings.get_arch_dtype_triple(dtype)
    assert (a, b) == (dtype, dtype)
    assert acc == CommonTypeMappings.get_acc_dtype(dtype)


@pytest.mark.skipif(not ugc.HAS_ARCH_FILTER, reason="arch_filter not importable")
@pytest.mark.parametrize(
    "gpu_target,listed",
    [("gfx942", True), ("gfx1250", True), ("gfx1250:xnack-", True), ("gfx1201", False)],
)
def test_fp32_needs_listed_warp_tiles(tmp_path, gpu_target, listed):
    # An arch whose table has no fp32 warp-tile entry must reject fp32 tiles
    # rather than let the arch filter pass every warp tile unchecked.
    codegen = ugc.UnifiedGemmCodegen(
        output_dir=tmp_path, datatype="fp32", layout="rcr", gpu_target=gpu_target
    )
    codegen.arch_filter = _RecordingFilter()
    tile = TileConfig(64, 64, 32, 2, 2, 1, 16, 16, 4)

    assert codegen._is_tile_arch_valid(tile, variant=ugc.GemmVariant.STANDARD) is listed
    assert len(codegen.arch_filter.calls) == int(listed)


def _fp32_config(arch, warp):
    return KernelConfig(
        dtype_a="fp32",
        dtype_b="fp32",
        dtype_c="fp32",
        tile_m=64,
        tile_n=64,
        tile_k=32,
        warp_m=warp[0],
        warp_n=warp[1],
        warp_k=warp[2],
        pipeline="compv3",
        gfx_arch=arch,
    )


@pytest.mark.parametrize(
    "arch,warp,valid",
    [
        ("gfx1250", (16, 16, 4), True),
        ("gfx1250", (32, 32, 16), False),
        ("gfx950", (32, 32, 8), True),
        ("gfx950", (32, 32, 16), False),
    ],
)
def test_ctypes_fp32_warp_tile_uses_arch_table(arch, warp, valid):
    # The warp tile is checked against the arch's fp32 list; the old
    # [[32,32,16],[16,16,16]] default no longer admits fp16 shapes for fp32.
    result = validate_kernel_config(_fp32_config(arch, warp))
    warp_errors = [e for e in result.errors if "warp tile" in e.lower()]
    assert (not warp_errors) is valid, result.errors


def test_ctypes_unlisted_dtype_key_is_an_error():
    # gfx1201 has no fp32 entry: report the missing key, suggest no warp tile.
    result = validate_kernel_config(_fp32_config("gfx1201", (16, 16, 16)))
    assert any("No warp tiles listed for fp32_fp32_fp32" in e for e in result.errors)
    assert "warp_m" not in result.suggested_fixes
