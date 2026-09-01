#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Per-op descriptor table for the block-scale quant GEMM bridge CPU unit tests.

This module is imported (never collected -- its name has no ``test_`` prefix) by
``test_quant_bridge_shared.py`` to drive the three near-identical CPU templates
that every quant bridge used to copy-paste per op:

  * the config-name **prefix** + **tiles-in-name** contract,
  * the byte-exact codegen<->utils **kernel-name contract**, and
  * the codegen-JSON **projection roundtrip**.

Each :class:`BridgeDescriptor` carries op-specific closures so the shared tests
stay byte-exact per op (the name-contract closure calls the correct
``make_*_kernel_name`` builder with the exact kwargs that op registers; the
projection closure asserts the exact keys that op projects).  The op-specific
``warp_tile_k`` arch tables stay in the per-op files, since their expected values
and ctor sets diverge and encode GPU-confirmed defects.

Import side effects: this appends ``python/`` and ``codegen/`` to ``sys.path`` so
the utils / codegen modules resolve, exactly as each per-op test file did.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Tuple

_DISP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DISP / "python"))
sys.path.insert(0, str(_DISP / "codegen"))


@dataclass
class BridgeDescriptor:
    """One block-scale quant bridge op, for the shared CPU contract tests."""

    op: str
    # (variant-label, config-ctor) pairs whose config .name must start with
    # ``gemm_<op>_<variant-label>_``.  When ``prefix_check_variant`` is False the
    # variant label is not asserted in the prefix (e.g. abquant/bquant, whose
    # prefix is checked as ``gemm_<op>_<cfg.variant_key>``).
    prefix_cases: List[Tuple[str, Callable]]
    prefix_check_variant: bool
    # A representative ctor used for the tiles-in-name check.
    tiles_ctor: Callable
    # Every ctor whose byte-exact name contract must hold.
    contract_ctors: List[Callable]
    # cfg -> expected-name; calls the op's make_*_kernel_name with exact kwargs.
    name_contract: Callable
    # (cfg, projected_dict) -> None; asserts the exact projected keys for this op.
    projection_checks: Callable
    # Representative ctor whose to_codegen_config() is projected.
    projection_ctor: Callable
    # Optional extra layout token every name must contain (e.g. "rcr").
    name_must_contain: Tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# aquant
# ---------------------------------------------------------------------------


def _aquant_descriptor() -> BridgeDescriptor:
    from gemm_aquant_utils import (
        default_fp8_config,
        default_bf8_config,
        default_fp8i4_config,
        default_bf8i4_config,
        default_fp8_preshufflequant_config,
        default_bf8i4_preshufflequant_config,
    )
    from codegen_common import make_gemm_aquant_kernel_name

    decode = [
        ("fp8", default_fp8_config),
        ("bf8", default_bf8_config),
        ("fp8i4", default_fp8i4_config),
        ("bf8i4", default_bf8i4_config),
    ]

    def name_contract(cfg):
        return make_gemm_aquant_kernel_name(
            variant_key=cfg.variant_key,
            layout=cfg.layout,
            pipeline=cfg.pipeline_key,
            epilogue="cshuffle",
            scheduler=cfg.scheduler,
            tile_m=cfg.tile_m, tile_n=cfg.tile_n, tile_k=cfg.tile_k,
            warp_m=cfg.warp_m, warp_n=cfg.warp_n, warp_k=cfg.warp_k,
            warp_tile_m=cfg.warp_tile_m, warp_tile_n=cfg.warp_tile_n,
            warp_tile_k=cfg.warp_tile_k,
            quant_group_m=cfg.quant_group_m,
            quant_group_n=cfg.quant_group_n,
            quant_group_k=cfg.quant_group_k,
            preshuffle_aquant=cfg.preshuffle_aquant,
        )

    def projection(cfg, d):
        assert d["variant_keys"] == [cfg.variant_key]
        assert d["layouts"] == [cfg.layout]
        assert d["quant_groups"][0]["quant_group_k"] == cfg.quant_group_k
        assert d["preshuffle_aquant"] == cfg.preshuffle_aquant

    return BridgeDescriptor(
        op="aquant",
        prefix_cases=decode,
        prefix_check_variant=True,
        tiles_ctor=default_fp8_config,
        contract_ctors=[c for _, c in decode]
        + [default_fp8_preshufflequant_config, default_bf8i4_preshufflequant_config],
        name_contract=name_contract,
        projection_checks=projection,
        projection_ctor=default_fp8_config,
    )


# ---------------------------------------------------------------------------
# abquant
# ---------------------------------------------------------------------------


def _abquant_descriptor() -> BridgeDescriptor:
    from gemm_abquant_utils import (
        default_fp8_config,
        default_bf8_config,
        default_fp4_config,
        default_fp8_preshufflequant_config,
        default_fp8_preshuffleb_config,
        default_bf8_preshuffleb_config,
        default_fp4_preshuffleb_config,
        default_fp8_preshuffleb_preshufflequant_config,
    )
    from codegen_common import make_gemm_abquant_kernel_name

    all_ctors = [
        default_fp8_config,
        default_bf8_config,
        default_fp4_config,
        default_fp8_preshufflequant_config,
        default_fp8_preshuffleb_config,
        default_bf8_preshuffleb_config,
        default_fp4_preshuffleb_config,
        default_fp8_preshuffleb_preshufflequant_config,
    ]

    def name_contract(cfg):
        return make_gemm_abquant_kernel_name(
            variant_key=cfg.variant_key,
            layout=cfg.layout,
            pipeline=cfg.pipeline,
            epilogue=cfg.epilogue,
            scheduler=cfg.scheduler,
            tile_m=cfg.tile_m, tile_n=cfg.tile_n, tile_k=cfg.tile_k,
            warp_m=cfg.warp_m, warp_n=cfg.warp_n, warp_k=cfg.warp_k,
            warp_tile_m=cfg.warp_tile_m, warp_tile_n=cfg.warp_tile_n,
            warp_tile_k=cfg.warp_tile_k,
            aquant_group_k=cfg.aquant_group_k,
            bquant_group_n=cfg.bquant_group_n,
            bquant_group_k=cfg.bquant_group_k,
            preshuffle_b=cfg.preshuffle_b,
            preshuffle_bquant=cfg.preshuffle_bquant,
            eight_waves=cfg.eight_waves,
        )

    def projection(cfg, d):
        assert d["variant_keys"] == [cfg.variant_key]
        assert d["layouts"] == [cfg.layout]
        assert d["aquant_group_k"] == cfg.aquant_group_k
        assert d["bquant_groups"][0]["bquant_group_n"] == cfg.bquant_group_n
        assert d["preshuffle_b"] == cfg.preshuffle_b

    # abquant's prefix check uses cfg.variant_key (not a fixed label) and asserts
    # "rcr" in the name; model that with variant labels drawn from the ctors.
    prefix_cases = [(None, c) for c in all_ctors]

    return BridgeDescriptor(
        op="abquant",
        prefix_cases=prefix_cases,
        prefix_check_variant=False,
        tiles_ctor=default_fp8_config,
        contract_ctors=all_ctors,
        name_contract=name_contract,
        projection_checks=projection,
        projection_ctor=default_fp8_config,
        name_must_contain=("rcr",),
    )


# ---------------------------------------------------------------------------
# bquant
# ---------------------------------------------------------------------------


def _bquant_descriptor() -> BridgeDescriptor:
    from gemm_bquant_utils import (
        NAME_PREFIX,
        default_fp8_config,
        default_bf8_config,
        default_fp8i4_config,
        default_bf8i4_config,
        default_fp8_preshuffleb_config,
        default_fp8_preshufflequant_config,
        default_fp8_preshuffleb_bquant_config,
        default_mx_bf16bf16_config,
        default_mx_bf16bf8_config,
        default_mx_bf16fp4_config,
    )
    from codegen_common import make_bquant_kernel_name

    base = [default_fp8_config, default_bf8_config,
            default_fp8i4_config, default_bf8i4_config]
    mx = [default_mx_bf16bf16_config, default_mx_bf16bf8_config,
          default_mx_bf16fp4_config]
    all_ctors = base + [
        default_fp8_preshuffleb_config,
        default_fp8_preshufflequant_config,
        default_fp8_preshuffleb_bquant_config,
    ] + mx

    def name_contract(cfg):
        return make_bquant_kernel_name(
            variant_key=cfg.variant_key,
            layout=cfg.layout,
            pipeline=cfg.pipeline,
            epilogue=cfg.epilogue,
            scheduler=cfg.scheduler,
            tile_m=cfg.tile_m, tile_n=cfg.tile_n, tile_k=cfg.tile_k,
            warp_m=cfg.warp_m, warp_n=cfg.warp_n, warp_k=cfg.warp_k,
            warp_tile_m=cfg.warp_tile_m, warp_tile_n=cfg.warp_tile_n,
            warp_tile_k=cfg.warp_tile_k,
            quant_group_m=cfg.quant_group_m,
            quant_group_n=cfg.quant_group_n,
            quant_group_k=cfg.quant_group_k,
            preshuffle_b=cfg.preshuffle_b,
            preshuffle_bquant=cfg.preshuffle_bquant,
            name_prefix=NAME_PREFIX,
        )

    def projection(cfg, d):
        assert d["variant_keys"] == [cfg.variant_key]
        assert d["layouts"] == [cfg.layout]
        assert d["quant_groups"][0]["quant_group_k"] == cfg.quant_group_k
        assert d["preshuffle_b"] == cfg.preshuffle_b

    prefix_cases = [(None, c) for c in all_ctors]

    return BridgeDescriptor(
        op="bquant",
        prefix_cases=prefix_cases,
        prefix_check_variant=False,
        tiles_ctor=default_fp8_config,
        contract_ctors=all_ctors,
        name_contract=name_contract,
        projection_checks=projection,
        projection_ctor=default_fp8_config,
    )


# ---------------------------------------------------------------------------
# rowcolquant
# ---------------------------------------------------------------------------


def _rowcolquant_descriptor() -> BridgeDescriptor:
    from gemm_rowcolquant_utils import (
        default_fp8_config,
        default_bf8_config,
    )
    from codegen_common import make_gemm_rowcolquant_kernel_name

    cases = [("fp8", default_fp8_config), ("bf8", default_bf8_config)]

    def name_contract(cfg):
        return make_gemm_rowcolquant_kernel_name(
            variant_key=cfg.variant_key,
            layout=cfg.layout,
            pipeline=cfg.pipeline,
            epilogue=cfg.epilogue,
            scheduler=cfg.scheduler,
            tile_m=cfg.tile_m, tile_n=cfg.tile_n, tile_k=cfg.tile_k,
            warp_m=cfg.warp_m, warp_n=cfg.warp_n, warp_k=cfg.warp_k,
            warp_tile_m=cfg.warp_tile_m, warp_tile_n=cfg.warp_tile_n,
            warp_tile_k=cfg.warp_tile_k,
        )

    def projection(cfg, d):
        assert d["variant_keys"] == [cfg.variant_key]
        assert d["layouts"] == [cfg.layout]
        assert d["tile_configs"][0]["tile_m"] == cfg.tile_m
        assert d["tile_configs"][0]["warp_tile_k"] == cfg.warp_tile_k

    return BridgeDescriptor(
        op="rowcolquant",
        prefix_cases=[(v + "_rcr", c) for v, c in cases],
        prefix_check_variant=True,
        tiles_ctor=default_fp8_config,
        contract_ctors=[c for _, c in cases],
        name_contract=name_contract,
        projection_checks=projection,
        projection_ctor=default_fp8_config,
    )


# ---------------------------------------------------------------------------
# tensor_quant
# ---------------------------------------------------------------------------


def _tensor_quant_descriptor() -> BridgeDescriptor:
    from gemm_tensor_quant_utils import (
        default_fp8_config,
        default_bf8_config,
    )
    from unified_gemm_tensor_quant_codegen import make_tensor_quant_kernel_name

    cases = [("fp8", default_fp8_config), ("bf8", default_bf8_config)]

    def name_contract(cfg):
        return make_tensor_quant_kernel_name(
            variant_key=cfg.variant_key,
            layout=cfg.layout,
            pipeline=cfg.pipeline,
            epilogue=cfg.epilogue,
            scheduler=cfg.scheduler,
            tile_m=cfg.tile_m, tile_n=cfg.tile_n, tile_k=cfg.tile_k,
            warp_m=cfg.warp_m, warp_n=cfg.warp_n, warp_k=cfg.warp_k,
            warp_tile_m=cfg.warp_tile_m, warp_tile_n=cfg.warp_tile_n,
            warp_tile_k=cfg.warp_tile_k,
        )

    def projection(cfg, d):
        assert d["variant_keys"] == [cfg.variant_key]
        assert d["layouts"] == [cfg.layout]
        assert d["pipeline"] == cfg.pipeline
        assert d["scheduler"] == cfg.scheduler
        tc = d["tile_configs"][0]
        assert tc["tile_m"] == cfg.tile_m
        assert tc["warp_tile_k"] == cfg.warp_tile_k

    return BridgeDescriptor(
        op="tensor_quant",
        prefix_cases=[(v + "_rcr", c) for v, c in cases],
        prefix_check_variant=True,
        tiles_ctor=default_fp8_config,
        contract_ctors=[c for _, c in cases],
        name_contract=name_contract,
        projection_checks=projection,
        projection_ctor=default_fp8_config,
    )


def all_descriptors() -> List[BridgeDescriptor]:
    return [
        _aquant_descriptor(),
        _abquant_descriptor(),
        _bquant_descriptor(),
        _rowcolquant_descriptor(),
        _tensor_quant_descriptor(),
    ]
