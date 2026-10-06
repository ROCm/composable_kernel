#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared, parametrized CPU unit tests for the block-scale quant GEMM bridges.

Collapses the three near-identical templates that every quant bridge
(aquant / abquant / bquant / rowcolquant / tensor_quant) used to copy-paste:

  * config-name **prefix** + **tiles-in-name** contract,
  * byte-exact codegen<->utils **kernel-name contract**, and
  * codegen-JSON **projection roundtrip**.

The per-op specifics (which ``make_*_kernel_name`` builder, which projected keys,
the exact prefix format) live in ``_quant_bridge_descriptors.py``; here we just
iterate them.  Every assertion is byte-exact per op and preserves the meaning of
the copies it replaces -- the op-specific ``warp_tile_k`` arch tables and all
GPU-confirmed regression tests stay in their per-op files, untouched.

No GPU / hipcc required.
"""

import pytest

from _quant_bridge_descriptors import all_descriptors

_DESCRIPTORS = all_descriptors()
_IDS = [d.op for d in _DESCRIPTORS]


@pytest.mark.parametrize("desc", _DESCRIPTORS, ids=_IDS)
def test_config_name_prefix(desc):
    """Each config's .name starts with the op/variant prefix (+ any layout token)."""
    for variant_label, ctor in desc.prefix_cases:
        cfg = ctor()
        if desc.prefix_check_variant:
            expected = f"gemm_{desc.op}_{variant_label}_"
            assert cfg.name.startswith(expected), cfg.name
        else:
            # abquant / bquant: prefix carries the config's own variant_key.
            assert cfg.name.startswith(f"gemm_{desc.op}_{cfg.variant_key}"), cfg.name
        for token in desc.name_must_contain:
            assert token in cfg.name, cfg.name


@pytest.mark.parametrize("desc", _DESCRIPTORS, ids=_IDS)
def test_config_name_encodes_tiles(desc):
    """The block tile and warp tile shapes must appear in the kernel .name."""
    cfg = desc.tiles_ctor()
    assert f"{cfg.tile_m}x{cfg.tile_n}x{cfg.tile_k}" in cfg.name
    assert f"{cfg.warp_tile_m}x{cfg.warp_tile_n}x{cfg.warp_tile_k}" in cfg.name


@pytest.mark.parametrize("desc", _DESCRIPTORS, ids=_IDS)
def test_name_contract_byte_exact(desc):
    """utils .name must be byte-exact with the op's codegen name builder."""
    for ctor in desc.contract_ctors:
        cfg = ctor()
        assert cfg.name == desc.name_contract(cfg), cfg.name


@pytest.mark.parametrize("desc", _DESCRIPTORS, ids=_IDS)
def test_codegen_projection_roundtrip(desc):
    """to_codegen_config() must project the op's variant/layout/group keys."""
    cfg = desc.projection_ctor()
    d = cfg.to_codegen_config()
    desc.projection_checks(cfg, d)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
