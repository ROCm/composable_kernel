#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Shared codegen infrastructure for the unified_*_codegen.py kernel-header generators.

Started as the common core of unified_gemm_codegen.py plus the arch-aware expansion
helpers from unified_grouped_conv_codegen.py. It now also carries the block-scale
quant layer -- kernel-name and epilogue-selection rules, the emitters for the C++
blocks every quant header shares, the spec-sweep plumbing, and the run_codegen_cli
driver -- for the plain and grouped {a,b,ab,tensor,rowcol}quant generators.

Two invariants live here and must not be re-derived anywhere else:

* the KERNEL_NAME format, which is a byte-exact contract between codegen and the
  Python runtime that dlopen()s the resulting .so, and
* fp8_warp_tile_k_for_arch(), the arch -> WarpTileK rule, where a wrong value
  compiles cleanly and then silently produces all-zero output on gfx942.
"""

import argparse
import itertools
import json
import logging
import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

log = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

ANY_INT = -1


# ============================================================================
# Tile and Trait Configuration (shared between GEMM and Conv)
# ============================================================================


@dataclass
class TileConfig:
    """Tile configuration parameters shared by GEMM and grouped conv."""

    tile_m: int
    tile_n: int
    tile_k: int
    warp_m: int
    warp_n: int
    warp_k: int
    warp_tile_m: int
    warp_tile_n: int
    warp_tile_k: int

    def is_valid(self) -> bool:
        if self.tile_m <= 0 or self.tile_n <= 0 or self.tile_k <= 0:
            return False
        return (
            self.tile_m % (self.warp_m * self.warp_tile_m) == 0
            and self.tile_n % (self.warp_n * self.warp_tile_n) == 0
            and self.tile_k % (self.warp_k * self.warp_tile_k) == 0
        )


@dataclass
class TraitConfigBase:
    """
    Base kernel trait configuration shared by GEMM and grouped conv.

    GEMM extends this with ``persistent``; grouped conv extends with
    ``double_smem_buffer`` and ``num_groups_to_merge``.
    """

    pipeline: str  # mem, compv3, compv4, compv5, ...
    epilogue: str  # cshuffle, default
    scheduler: str  # intrawave, interwave
    pad_m: bool
    pad_n: bool
    pad_k: bool

    # Unsupported (pipeline, epilogue, scheduler) combinations.
    # Only 'mem' and 'basic_v1' pipelines support interwave; all compute
    # pipelines (compv3/v4/v5/v6/async) only support intrawave.
    _UNSUPPORTED: ClassVar[FrozenSet] = frozenset(
        {
            ("compv3", "cshuffle", "interwave"),
            ("compv3", "default", "interwave"),
            ("compv4", "cshuffle", "interwave"),
            ("compv4", "default", "interwave"),
            ("compv5", "cshuffle", "interwave"),
            ("compv5", "default", "interwave"),
            ("compv6", "cshuffle", "interwave"),
            ("compv6", "default", "interwave"),
            ("comp_async", "cshuffle", "interwave"),
            ("comp_async", "default", "interwave"),
            ("basic_async_v1", "cshuffle", "interwave"),
            ("basic_async_v1", "default", "interwave"),
        }
    )

    def is_valid(self) -> bool:
        return (self.pipeline, self.epilogue, self.scheduler) not in self._UNSUPPORTED


# ============================================================================
# Type Mappings (centralized for both GEMM and conv codegen)
# ============================================================================


class CommonTypeMappings:
    """Centralized type mappings shared by GEMM and grouped conv codegen."""

    DTYPE_TO_CK = {
        "fp16": "fp16_t",
        "bf16": "bf16_t",
        "fp32": "float",
        "fp8": "fp8_t",
        "bf8": "bf8_t",
        "int8": "int8_t",
        "int32": "int32_t",
    }

    DTYPE_TO_CK_QUALIFIED = {
        "fp16": "ck_tile::fp16_t",
        "bf16": "ck_tile::bf16_t",
        "fp32": "float",
        "fp8": "ck_tile::fp8_t",
        "bf8": "ck_tile::bf8_t",
        "int8": "int8_t",
        "int32": "int32_t",
    }

    DTYPE_TO_DISPATCHER = {
        "fp16": "DataType::FP16",
        "bf16": "DataType::BF16",
        "fp32": "DataType::FP32",
        "fp8": "DataType::FP8",
        "bf8": "DataType::BF8",
        "int8": "DataType::INT8",
        "int32": "DataType::INT32",
    }

    # GEMM-specific layout mappings ("r"/"c" for row/column major).
    # Convolution layouts (NHWGC, GKYXC, etc.) are handled by
    # unified_grouped_conv_codegen.py via GroupedConvLayout / GroupedConvTypeMappings.
    GEMM_LAYOUT_TO_CK = {
        "r": "tensor_layout::gemm::RowMajor",
        "c": "tensor_layout::gemm::ColumnMajor",
    }
    LAYOUT_TO_CK = GEMM_LAYOUT_TO_CK  # backward compat alias

    GEMM_LAYOUT_TO_DISPATCHER = {
        "r": "LayoutTag::RowMajor",
        "c": "LayoutTag::ColMajor",
    }
    LAYOUT_TO_DISPATCHER = GEMM_LAYOUT_TO_DISPATCHER  # backward compat alias

    # GEMM-only pipeline mappings (used by unified_gemm_codegen.py).
    # Convolution pipelines are in GroupedConvTypeMappings
    # (unified_grouped_conv_codegen.py). CK Tile conv supports:
    # BASIC_V1, Mem, CompV3, CompV4, CompV5, CompV6, ASYNC_V1, ASYNC_V4.
    # The dispatcher currently generates: mem, compv3, compv4.
    # preshufflev2 is GEMM-only (weight pre-shuffle for GEMM, not conv).
    PIPELINE_TO_CK = {
        "mem": "GemmPipelineAgBgCrMem",
        "compv3": "GemmPipelineAgBgCrCompV3",
        "compv4": "GemmPipelineAgBgCrCompV4",
        "compv5": "GemmPipelineAgBgCrCompV5",
        "preshufflev2": "WeightPreshufflePipelineAGmemBGmemCRegV2",
    }

    PIPELINE_TO_BASE = {
        "mem": "BaseGemmPipelineAgBgCrMem",
        "compv3": "BaseGemmPipelineAgBgCrCompV3",
        "compv4": "BaseGemmPipelineAgBgCrCompV4",
        "compv5": "BaseGemmPipelineAgBgCrCompV5",
        "preshufflev2": "BaseWeightPreshufflePipelineAGmemBGmemCRegV2",
    }

    PIPELINE_TO_DISPATCHER = {
        "mem": "Pipeline::Mem",
        "compv3": "Pipeline::CompV3",
        "compv4": "Pipeline::CompV4",
        "compv5": "Pipeline::CompV5",
        "preshufflev2": "Pipeline::PreShuffleV2",
    }

    SCHEDULER_TO_CK = {
        "intrawave": "GemmPipelineScheduler::Intrawave",
        "interwave": "GemmPipelineScheduler::Interwave",
        "default": "GemmPipelineScheduler::Default",
    }

    SCHEDULER_TO_DISPATCHER = {
        "intrawave": "Scheduler::Intrawave",
        "interwave": "Scheduler::Interwave",
        "default": "Scheduler::Auto",
    }

    EPILOGUE_TO_DISPATCHER = {
        "cshuffle": "Epilogue::CShuffle",
        "default": "Epilogue::Default",
    }

    @staticmethod
    def get_output_dtype(dtype: str) -> str:
        """Get output (C) datatype for an A/B element dtype.

        Low-precision float inputs accumulate into and store as fp16
        (fp8/bf8 -> fp16); int8 stores its int32 accumulator (int8 -> int32).
        Everything else stores in its own dtype.
        """
        if dtype in ("fp8", "bf8"):
            return "fp16"
        if dtype == "int8":
            return "int32"
        return dtype

    @staticmethod
    def get_acc_dtype(dtype: str) -> str:
        """Get accumulator datatype for an A/B element dtype.

        Integer GEMM accumulates in int32; every float dtype accumulates in
        fp32.
        """
        return "int32" if dtype == "int8" else "fp32"


# ============================================================================
# Code Generation Helpers
# ============================================================================


def generate_cpp_compilation_unit(kernel_name: str) -> str:
    """Generate a .cpp compilation unit that includes a kernel header.

    This is the standard pattern: one .cpp per kernel that just includes
    the generated .hpp header, causing template instantiation.
    """
    return (
        f"// Auto-generated compilation unit for {kernel_name}\n"
        f'#include "{kernel_name}.hpp"\n'
    )


def parallel_generate(
    generate_fn: Callable[[T], R],
    items: Sequence[T],
    parallel: bool = True,
    max_workers: Optional[int] = None,
) -> List[R]:
    """Run ``generate_fn`` over ``items``, optionally in parallel.

    Logs per-item progress (best-of-conv pattern).
    Returns a flat list of results in completion order.

    ``max_workers`` caps the thread pool size; ``None`` keeps the previous
    behavior of letting ThreadPoolExecutor pick its own default.
    """
    results: List[R] = []
    if not items:
        return results

    if parallel and len(items) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(generate_fn, item): item for item in items}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                log.info("Generated: %s", futures[future])
    else:
        for item in items:
            result = generate_fn(item)
            results.append(result)
            log.info("Generated: %s", item)

    return results


# ============================================================================
# Arch-Aware Expansion Helpers (adopted from conv kernel_decl.hpp)
# ============================================================================

# These load from arch_specs_generated when available, falling back to
# hardcoded defaults that match the most common arch (gfx942).

_arch_data_cache: Optional[Dict] = None


def _get_arch_data() -> Dict:
    """Load arch filter data, with caching."""
    global _arch_data_cache
    if _arch_data_cache is not None:
        return _arch_data_cache

    try:
        from arch_specs_generated import (
            WARP_SUPPORTED_COMBINATIONS,
            WARP_TILE_SUPPORTED_COMBINATIONS,
            TRAIT_UNSUPPORTED_COMBINATIONS,
            get_supported_archs,
        )

        _arch_data_cache = {
            "warp_combos": WARP_SUPPORTED_COMBINATIONS,
            "warp_tile_combos": WARP_TILE_SUPPORTED_COMBINATIONS,
            "trait_unsupported": TRAIT_UNSUPPORTED_COMBINATIONS,
            "supported_archs": get_supported_archs(),
        }
    except ImportError:
        _arch_data_cache = {
            "warp_combos": {
                "gfx942": [[1, 4, 1], [2, 2, 1], [4, 1, 1]],
                "gfx90a": [[1, 4, 1], [2, 2, 1], [4, 1, 1]],
            },
            "warp_tile_combos": {
                "gfx942": {"fp16_fp16_fp32": [[16, 16, 16], [32, 32, 16]]},
                "gfx90a": {"fp16_fp16_fp32": [[16, 16, 16], [32, 32, 16]]},
            },
            "trait_unsupported": {
                ("compv3", "cshuffle", "interwave"),
                ("compv4", "cshuffle", "interwave"),
            },
            "supported_archs": ["gfx90a", "gfx942", "gfx950"],
        }
    return _arch_data_cache


def valid_wave_configs(arch: str) -> List[List[int]]:
    """Return valid [wave_m, wave_n, wave_k] combos for *arch*."""
    data = _get_arch_data()
    return data["warp_combos"].get(arch, [[2, 2, 1]])


def valid_warp_configs(arch: str, dtype: str) -> List[List[int]]:
    """Return valid [warp_tile_m, warp_tile_n, warp_tile_k] combos for *arch*/*dtype*.

    The dtype key is constructed as ``{dtype}_{dtype}_{acc}`` where acc is
    fp32 for float types and int32 for int8.
    """
    data = _get_arch_data()
    acc = "int32" if dtype == "int8" else "fp32"
    dtype_key = f"{dtype}_{dtype}_{acc}"
    arch_tiles = data["warp_tile_combos"].get(arch, {})
    return arch_tiles.get(dtype_key, [[32, 32, 16]])


def valid_trait_configs() -> List[Tuple[str, str]]:
    """Return valid (pipeline, scheduler) pairs.

    Compute pipelines only support intrawave; mem supports both.
    """
    return [
        ("compv3", "intrawave"),
        ("compv4", "intrawave"),
        ("compv5", "intrawave"),
        ("mem", "intrawave"),
        ("mem", "interwave"),
    ]


def needs_wave_expansion(config: dict) -> bool:
    """True if wave_m or wave_n is a wildcard (ANY_INT = -1)."""
    return config.get("wave_m", 2) == ANY_INT or config.get("wave_n", 2) == ANY_INT


def needs_warp_expansion(config: dict) -> bool:
    """True if warp_m or warp_n is a wildcard (ANY_INT = -1)."""
    return config.get("warp_m", 32) == ANY_INT or config.get("warp_n", 32) == ANY_INT


def needs_pipeline_expansion(config: dict) -> bool:
    """True if pipeline is a wildcard (\"*\")."""
    return config.get("pipeline", "compv4") == "*"


# ============================================================================
# Block-scale quant type mappings
# ============================================================================
#
# The quant kernel headers are emitted with CK_TILE_SINGLE_KERNEL_INCLUDE and
# carry no `using namespace ck_tile`, so every type they name must be FULLY
# QUALIFIED. That makes these maps deliberately distinct from
# CommonTypeMappings.GEMM_LAYOUT_TO_CK / .SCHEDULER_TO_CK, whose values are
# unqualified for the in-namespace GEMM codegen. Do not "unify" the two: they
# emit different C++ and only one of them compiles in a quant header.
#
# QUANT_SCHEDULER_TO_CK also omits CommonTypeMappings' "default" key -- the
# quant sweeps only ever produce "intrawave" / "interwave", and an unmapped key
# should raise KeyError at codegen time rather than emit a silently wrong
# scheduler.

QUANT_LAYOUT_TO_CK = {
    "r": "ck_tile::tensor_layout::gemm::RowMajor",
    "c": "ck_tile::tensor_layout::gemm::ColumnMajor",
}

QUANT_SCHEDULER_TO_CK = {
    "intrawave": "ck_tile::GemmPipelineScheduler::Intrawave",
    "interwave": "ck_tile::GemmPipelineScheduler::Interwave",
}


# ============================================================================
# Shared quant kernel-name / epilogue construction
# ============================================================================
#
# Every block-scale quant family builds its KERNEL_NAME from the same skeleton
# and selects its epilogue from the same rule; only the prefix, the quant-group
# segments, the trailing flags, and one per-family boolean differ. The two
# helpers below hold that shared shape. The per-family public functions that
# follow are thin wrappers over them, kept as separate names because each is
# imported by name from both a codegen module and a runtime utils module -- the
# codegen<->runtime kernel-name contract is byte-exact, and collapsing the names
# has bitten this module before (see the shadowing NOTE further down).


def quant_effective_epilogue(
    tile_n: int,
    warp_n: int,
    warp_tile_n: int,
    quant_group_n: int,
    tiled_mma_permute_n: bool = False,
) -> str:
    """Return the epilogue tag ("permute_n" / "cshuffle") the codegen will emit.

    Mirrors the epilogue selection in run_gemm_quant_example.inc:208-252:
      TiledPermuteN = (BQuantGroupSize::kN > 1) ? false : GemmConfig::TiledMMAPermuteN
      GemmEpilogue  = TiledPermuteN ? PermuteNEpilogue : CShuffleEpilogue
    and the PreshuffleB configs' override TiledMMAPermuteN = (N_Repeat % 2 == 0),
    where N_Repeat = TileN / (WarpN * WarpTileN).

    ``tiled_mma_permute_n`` is the GemmConfig-level flag, which is a property of
    the config struct rather than of tile geometry -- GemmConfigBase sets it
    false and only the PreshuffleB configs override it (gemm_utils.hpp:214-215).
    Each family passes its own value; see the wrappers below.
    """
    n_repeat = tile_n // (warp_n * warp_tile_n)
    use_permute_n = tiled_mma_permute_n and (n_repeat % 2 == 0) and (quant_group_n == 1)
    return "permute_n" if use_permute_n else "cshuffle"


def make_quant_kernel_name(
    *,
    prefix: str,
    variant_key: str,
    layout: str,
    pipeline: str,
    epilogue: str,
    scheduler: str,
    tile_m: int, tile_n: int, tile_k: int,
    warp_m: int, warp_n: int, warp_k: int,
    warp_tile_m: int, warp_tile_n: int, warp_tile_k: int,
    group_segments: Sequence[str] = (),
    flags: Sequence[Tuple[bool, str]] = (),
) -> str:
    """Join the canonical quant KERNEL_NAME segments.

    ``epilogue`` must already be the EFFECTIVE epilogue (what the codegen emits),
    not the user-requested one -- resolve it via quant_effective_epilogue at the
    call site. ``group_segments`` are pre-formatted quant-group strings such as
    ``"qg1x1x128"``; ``flags`` are ``(condition, segment)`` pairs appended in
    order when the condition holds.
    """
    parts = [
        prefix,
        variant_key,
        layout,
        pipeline,
        epilogue,
        scheduler,
        f"{tile_m}x{tile_n}x{tile_k}",
        f"{warp_m}x{warp_n}x{warp_k}",
        f"{warp_tile_m}x{warp_tile_n}x{warp_tile_k}",
        *group_segments,
    ]
    parts.extend(segment for cond, segment in flags if cond)
    return "_".join(parts)


# ============================================================================
# BQuant kernel name construction
# ============================================================================


def bquant_effective_epilogue(
    tile_n: int,
    warp_n: int,
    warp_tile_n: int,
    quant_group_n: int,
    preshuffle_b: bool = False,
) -> str:
    """Return the epilogue tag that the codegen will actually emit for the given tile params.

    Mirrors the TiledMMAPermuteN / TiledPermuteN epilogue selection in
    run_gemm_quant_example.inc:208-252:
      TiledMMAPermuteN = PreshuffleB && (N_Repeat % 2 == 0)   (GemmConfig)
      TiledPermuteN    = (BQuantGroupSize::kN > 1) ? false : TiledMMAPermuteN
      GemmEpilogue     = TiledPermuteN ? PermuteNEpilogue : CShuffleEpilogue
    where N_Repeat = TileN / (WarpN * WarpTileN).

    CRITICAL: TiledMMAPermuteN is false in GemmConfigBase and is ONLY overridden to
    (N_Repeat % 2 == 0) by the PreshuffleB configs (gemm_utils.hpp:214-215). Every
    non-PreshuffleB config -- including MX (microscale, PreshuffleB=false) -- inherits
    false and therefore uses CShuffleEpilogue. Omitting the ``preshuffle_b`` gate made
    the bridge emit a PermuteNEpilogue for even-N_Repeat MX kernels (e.g. mx_bf16bf8
    128-tile, N_Repeat=2), a different + ~16-17% slower kernel than Old-TE's CShuffle.

    Returns "permute_n" when PermuteNEpilogue is selected, "cshuffle" otherwise.
    """
    return quant_effective_epilogue(
        tile_n, warp_n, warp_tile_n, quant_group_n,
        tiled_mma_permute_n=preshuffle_b,
    )


def make_bquant_kernel_name(
    variant_key: str,
    layout: str,
    pipeline: str,
    epilogue: str,  # ignored — actual epilogue is computed from tile params via bquant_effective_epilogue
    scheduler: str,
    tile_m: int, tile_n: int, tile_k: int,
    warp_m: int, warp_n: int, warp_k: int,
    warp_tile_m: int, warp_tile_n: int, warp_tile_k: int,
    quant_group_m: int,
    quant_group_n: int,
    quant_group_k: int,
    preshuffle_b: bool = False,
    preshuffle_bquant: bool = False,
    name_prefix: str = "grouped_gemm_bquant",
) -> str:
    """Return the canonical BQuant kernel name used as KERNEL_NAME in generated headers.

    Both BQuantKernelConfig (utils) and BQuantKernelSpec (codegen) delegate to this
    function so the two sides are guaranteed to stay byte-exact.

    The epilogue segment in the name reflects the epilogue the codegen actually emits
    (computed via bquant_effective_epilogue from tile params) rather than the
    user-specified epilogue string, so the name always matches the compiled kernel.
    The ``epilogue`` parameter is accepted for call-site compatibility but not used.

    ``name_prefix`` selects the operator family. It defaults to
    ``"grouped_gemm_bquant"`` for backward compatibility with the quant-grouped
    (single-problem) BQuant bridge already in tree; the plain non-grouped
    ``gemm_bquant`` bridge under 38_block_scale_gemm passes ``"gemm_bquant"``.
    """
    return make_quant_kernel_name(
        prefix=name_prefix,
        variant_key=variant_key,
        layout=layout,
        pipeline=pipeline,
        epilogue=bquant_effective_epilogue(
            tile_n, warp_n, warp_tile_n, quant_group_n, preshuffle_b
        ),
        scheduler=scheduler,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        warp_m=warp_m, warp_n=warp_n, warp_k=warp_k,
        warp_tile_m=warp_tile_m, warp_tile_n=warp_tile_n, warp_tile_k=warp_tile_k,
        group_segments=(f"qg{quant_group_m}x{quant_group_n}x{quant_group_k}",),
        flags=(
            (preshuffle_b, "preshuffleb"),
            (preshuffle_bquant, "preshufflebq"),
        ),
    )


# ============================================================================
# Non-grouped gemm_rowcolquant kernel name construction
# ============================================================================
# NOTE: distinct from the GROUPED make_rowcolquant_kernel_name further down,
# which takes dtype + pad_m/pad_n/pad_k/persistent and emits a
# "grouped_gemm_rowcolquant_..." name. This one emits "gemm_rowcolquant_...".
# Both families independently landed a builder named
# make_rowcolquant_kernel_name; because Python keeps only the last def, holding
# both under that one name silently rebinds one family's callers to the other's
# signature (the same shadowing bug previously caught on make_aquant_kernel_name
# / make_abquant_kernel_name). Renamed with the gemm_ prefix convention already
# used by make_gemm_aquant_kernel_name / make_gemm_abquant_kernel_name.


def make_gemm_rowcolquant_kernel_name(
    variant_key: str,
    layout: str,
    pipeline: str,
    epilogue: str,
    scheduler: str,
    tile_m: int, tile_n: int, tile_k: int,
    warp_m: int, warp_n: int, warp_k: int,
    warp_tile_m: int, warp_tile_n: int, warp_tile_k: int,
) -> str:
    """Return the canonical RowColQuant kernel name used as KERNEL_NAME in headers.

    Both RowColQuantKernelConfig (utils) and RowColQuantKernelSpec (codegen)
    delegate to this function so the two sides are guaranteed to stay byte-exact.

    RowColQuant has no quant-group segment (scales are global per-row / per-col
    vectors), and GemmConfigRowColQuant fixes TiledMMAPermuteN=false so the
    epilogue is always CShuffle.  The ``epilogue`` argument is therefore accepted
    for call-site symmetry with the BQuant helper but always emitted verbatim.
    """
    return make_quant_kernel_name(
        prefix="gemm_rowcolquant",
        variant_key=variant_key,
        layout=layout,
        pipeline=pipeline,
        epilogue=epilogue,
        scheduler=scheduler,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        warp_m=warp_m, warp_n=warp_n, warp_k=warp_k,
        warp_tile_m=warp_tile_m, warp_tile_n=warp_tile_n, warp_tile_k=warp_tile_k,
    )


# ============================================================================
# Arch-derived warp tile K
# ============================================================================


def fp8_warp_tile_k_for_arch(gfx_arch: str, *, preshuffle_quant: bool = False) -> int:
    """Arch-derived WarpTileK for an 8-bit float operand with M_Warp_Tile=16.

    Mirrors ``ck_tile::get_k_warp_tile<fp8_t/bf8_t, M_Warp_Tile=16, IsFlatMM>()``
    (include/ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp)::

        gfx950                        -> 128  (both plain and preshufflequant)
        gfx942/other, plain           ->  32
        gfx942/other, preshufflequant ->  64

    This rule must exist exactly once. Using 128 on gfx942 compiles cleanly and
    then produces **all-zeros output** -- there is no valid 16x16x128 fp8/bf8
    warp-gemm on gfx942 -- so a second, drifting copy is a silent-wrong-answer
    bug rather than a build failure.

    ``preshuffle_quant`` applies to AQuant's preshufflequant configs; every
    other quant operator passes the default.
    """
    if "gfx950" in gfx_arch:
        return 128
    return 64 if preshuffle_quant else 32


# ============================================================================
# Quant spec-sweep helpers
# ============================================================================
#
# The five ``_build_specs()`` functions share their whole middle: the same
# ``itertools.product`` over (variant x layout x tile x optional group), the
# same unknown-variant / unsupported-pipeline warn-and-skip, the same
# ``TileConfig(**tile_dict)`` construction and ``is_valid()`` guard.
#
# NOT extracted, deliberately: the ``config.get()`` pull-outs and the
# ``specs.append(<Op>KernelSpec(...))`` call. Those are where the operators
# genuinely differ -- different dataclasses, different field sets, different
# defaults for the same key (``pad_k`` is True everywhere but ABQuant, and
# AQuant's default scheduler is computed from ``preshuffle_aquant``). Routing
# them through one helper would mean a kwargs blob that no longer type-checks
# against the target dataclass. Left in the per-op scripts.


def tile_config_from_dict(tile_dict: Mapping[str, int]) -> TileConfig:
    """Build a :class:`TileConfig` from a sweep-config tile dict.

    All nine keys are required; a missing one is a malformed config and should
    raise rather than silently default.
    """
    return TileConfig(
        tile_m=tile_dict["tile_m"],
        tile_n=tile_dict["tile_n"],
        tile_k=tile_dict["tile_k"],
        warp_m=tile_dict["warp_m"],
        warp_n=tile_dict["warp_n"],
        warp_k=tile_dict["warp_k"],
        warp_tile_m=tile_dict["warp_tile_m"],
        warp_tile_n=tile_dict["warp_tile_n"],
        warp_tile_k=tile_dict["warp_tile_k"],
    )


def rcr_only_layout_guard(layout: str) -> Optional[str]:
    """Layout guard for operators that only support ``rcr``.

    Downstream codegen indexes ``layout[0/1/2]`` and looks each char up in the
    op's LAYOUT_TO_CK map, so anything outside the supported scope would raise
    IndexError/KeyError. Skip cleanly with a warning instead.
    """
    if layout != "rcr":
        return f"Unsupported layout {layout} (only rcr) -- skipping"
    return None


def iter_quant_axes(
    config: dict,
    *,
    variants: Mapping[str, Any],
    logger: logging.Logger,
    pipeline: Optional[str] = None,
    pipeline_map: Optional[Mapping[str, Any]] = None,
    extra_axis: Optional[Tuple[str, List[dict]]] = None,
    layout_guard: Optional[Callable[[str], Optional[str]]] = None,
) -> Iterator[Tuple[str, str, TileConfig, dict]]:
    """Yield ``(variant_key, layout, tile, extra)`` for every valid sweep point.

    Guards run in the order variant -> pipeline -> layout -> tile validity, and
    each one warns (or, for tile validity, debug-logs) and skips exactly as the
    hand-rolled loops did. ``logger`` is the *caller's* logger so the module
    name in the log record still identifies the operator.

    ``pipeline_map`` is optional -- AQuant has no pipeline axis. ``extra_axis``
    is ``(config_key, default)`` for the fourth product axis (``quant_groups``
    for AQuant/BQuant, ``bquant_groups`` for ABQuant); operators without one
    receive ``{}`` as ``extra``.
    """
    extra_key, extra_default = extra_axis if extra_axis else (None, None)
    extra_values: List[dict] = (
        config.get(extra_key, extra_default) if extra_key else [{}]
    )

    for variant_key, layout, tile_dict, extra in itertools.product(
        config.get("variant_keys", ["fp8"]),
        config.get("layouts", ["rcr"]),
        config.get("tile_configs", []),
        extra_values,
    ):
        if variant_key not in variants:
            logger.warning("Unknown variant_key %s -- skipping", variant_key)
            continue
        if pipeline_map is not None and pipeline not in pipeline_map:
            logger.warning("Unsupported pipeline %s -- skipping", pipeline)
            continue
        if layout_guard is not None:
            reason = layout_guard(layout)
            if reason is not None:
                logger.warning("%s", reason)
                continue

        tile = tile_config_from_dict(tile_dict)
        if not tile.is_valid():
            logger.debug("Invalid tile config %s -- skipping", tile)
            continue

        yield variant_key, layout, tile, extra


def quant_decode_default_config(*, warp_tile_k: int, **overrides) -> dict:
    """The GemmConfigQuantDecode sweep shared by tensor_quant/rowcolquant/bquant.

    fp8+bf8, rcr only, compv3 + cshuffle + intrawave, tile 16x64x256 with warp
    1x4x1 and warp_tile 16x16x``warp_tile_k``. ``overrides`` is merged last, so
    an operator can add its own keys (BQuant's ``quant_groups`` /
    ``preshuffle_*``) or restate an existing one.

    AQuant and ABQuant deliberately do not use this: AQuant has no pipeline or
    epilogue axis and sweeps four layouts, and ABQuant is a 128x128x128 prefill
    tile with ``pad_k=False``. Expressing either as overrides of this base would
    mean *removing* keys, which is less readable than their own literal.
    """
    config = {
        "variant_keys": ["fp8", "bf8"],
        "layouts": ["rcr"],
        "pipeline": "compv3",
        "epilogue": "cshuffle",
        "scheduler": "intrawave",
        "tile_configs": [
            # GemmConfigQuantDecode<fp8_t>: M=16, N=64, K=256/sizeof(8bit)=256
            {"tile_m": 16, "tile_n": 64, "tile_k": 256,
             "warp_m": 1, "warp_n": 4, "warp_k": 1,
             "warp_tile_m": 16, "warp_tile_n": 16,
             "warp_tile_k": warp_tile_k},
        ],
        "pad_m": False,
        "pad_n": False,
        "pad_k": True,
        "block_size": 256,
        "k_block_per_cu": 1,
        "double_smem_buffer": False,
    }
    config.update(overrides)
    return config


# ============================================================================
# AQuant kernel name construction
# ============================================================================


def aquant_effective_epilogue(
    tile_n: int,
    warp_n: int,
    warp_tile_n: int,
    quant_group_n: int,
) -> str:
    """Return the epilogue tag the codegen will emit for AQuant kernels.

    Mirrors the same TiledMMAPermuteN / use_permute_n_epilogue logic as BQuant
    (the PermuteN condition is driven by B-side tile geometry, regardless of
    which side is quantised).  Returns "permute_n" when PermuteNEpilogue is
    selected, "cshuffle" otherwise.

    Note the unconditional ``tiled_mma_permute_n=True``: unlike the BQuant helper
    this family has no preshuffle gate, so parity alone decides. Preserved
    as-is -- see TestQuantEffectiveEpilogue, which pins the asymmetry.
    """
    return quant_effective_epilogue(
        tile_n, warp_n, warp_tile_n, quant_group_n, tiled_mma_permute_n=True,
    )


def make_aquant_kernel_name(
    variant_key: str,
    layout: str,
    pipeline: str,
    epilogue: str,  # ignored — actual epilogue is computed from tile params
    scheduler: str,
    tile_m: int, tile_n: int, tile_k: int,
    warp_m: int, warp_n: int, warp_k: int,
    warp_tile_m: int, warp_tile_n: int, warp_tile_k: int,
    quant_group_m: int,
    quant_group_n: int,
    quant_group_k: int,
    preshuffle_aq: bool = False,
) -> str:
    """Return the canonical AQuant kernel name used as KERNEL_NAME in generated headers.

    Both AQuantKernelConfig (utils) and AQuantKernelSpec (codegen) delegate to
    this function so the two sides are guaranteed to stay byte-exact.
    """
    return make_quant_kernel_name(
        prefix="grouped_gemm_aquant",
        variant_key=variant_key,
        layout=layout,
        pipeline=pipeline,
        epilogue=aquant_effective_epilogue(tile_n, warp_n, warp_tile_n, quant_group_n),
        scheduler=scheduler,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        warp_m=warp_m, warp_n=warp_n, warp_k=warp_k,
        warp_tile_m=warp_tile_m, warp_tile_n=warp_tile_n, warp_tile_k=warp_tile_k,
        group_segments=(f"aqg{quant_group_m}x{quant_group_n}x{quant_group_k}",),
        flags=((preshuffle_aq, "preshuffleaq"),),
    )


# ============================================================================
# ABQuant kernel name construction
# ============================================================================


def abquant_effective_epilogue(
    tile_n: int,
    warp_n: int,
    warp_tile_n: int,
    bquant_group_n: int,
    pipeline: str = "compv3",
) -> str:
    """Return the epilogue tag the codegen will emit for ABQuant kernels.

    The PermuteN selection is governed by the B-side quant group N, matching
    the same logic used by BQuant / AQuant. EightWaves and PreshuffleB use
    TransposeC=true (transposed accumulator layout) which is incompatible with
    PermuteNEpilogue — both must always use CShuffleEpilogue (TiledMMAPermuteN=false
    in the C++ test fixtures for both GemmConfigEightWaves and GemmConfigPreshuffleB_ABQuant_Prefill).
    """
    return quant_effective_epilogue(
        tile_n, warp_n, warp_tile_n, bquant_group_n,
        tiled_mma_permute_n=pipeline not in ("eightwaves", "preshuffleb"),
    )


def make_abquant_kernel_name(
    variant_key: str,
    layout: str,
    pipeline: str,
    epilogue: str,  # ignored — actual epilogue is computed from tile params
    scheduler: str,
    tile_m: int, tile_n: int, tile_k: int,
    warp_m: int, warp_n: int, warp_k: int,
    warp_tile_m: int, warp_tile_n: int, warp_tile_k: int,
    aquant_group_m: int,
    aquant_group_n: int,
    aquant_group_k: int,
    bquant_group_m: int,
    bquant_group_n: int,
    bquant_group_k: int,
    preshuffle_b: bool = False,
    preshuffle_aq: bool = False,
    preshuffle_bq: bool = False,
    transpose_c: bool = False,
) -> str:
    """Return the canonical ABQuant kernel name used as KERNEL_NAME in generated headers.

    Both ABQuantKernelConfig (utils) and ABQuantKernelSpec (codegen) delegate
    to this function so the two sides are guaranteed to stay byte-exact.
    """
    return make_quant_kernel_name(
        prefix="grouped_gemm_abquant",
        variant_key=variant_key,
        layout=layout,
        pipeline=pipeline,
        epilogue=abquant_effective_epilogue(
            tile_n, warp_n, warp_tile_n, bquant_group_n, pipeline
        ),
        scheduler=scheduler,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        warp_m=warp_m, warp_n=warp_n, warp_k=warp_k,
        warp_tile_m=warp_tile_m, warp_tile_n=warp_tile_n, warp_tile_k=warp_tile_k,
        group_segments=(
            f"aqg{aquant_group_m}x{aquant_group_n}x{aquant_group_k}",
            f"bqg{bquant_group_m}x{bquant_group_n}x{bquant_group_k}",
        ),
        flags=(
            (preshuffle_b, "preshuffleb"),
            (preshuffle_aq, "preshuffleaq"),
            (preshuffle_bq, "preshufflebq"),
            (transpose_c, "transposec"),
        ),
    )


# =============================================================================
# RowColQuant / TensorQuant shared definitions
# =============================================================================
#
# These two operators are structurally identical -- they differ only in the
# ck_tile::QuantType enum the codegen emits -- so their naming, tile defaults and
# trait defaults live here rather than being duplicated per operator.
#
# Both the codegen (unified_grouped_gemm_{rowcolquant,tensorquant}_codegen.py) and
# the runtime wrappers (python/grouped_gemm_{rowcolquant,tensorquant}_utils.py)
# import from this module. That direction matters: previously the utils layer
# imported the codegen module to reach the name builder, which inverted the
# intended layering.

# The only pipeline/epilogue combination these kernels support. The generated
# header hardwires CompV3 + CShuffle, so any other value would produce a kernel
# whose *name* disagrees with the code it contains. _build_specs validates against
# these maps and skips unsupported combinations rather than mislabelling a kernel.
ROWCOL_TENSOR_QUANT_PIPELINE_MAP = {
    "compv3": "ck_tile::GemmPipelineAgBgCrCompV3",
}

ROWCOL_TENSOR_QUANT_BASE_PIPELINE_MAP = {
    "compv3": "ck_tile::BaseGemmPipelineAgBgCrCompV3",
}

ROWCOL_TENSOR_QUANT_EPILOGUE_MAP = {
    "cshuffle": "ck_tile::CShuffleEpilogue",
}

# The dispatcher's ctypes bridge validates stride_A == K, stride_B == K and
# stride_C == N, which is the packed "rcr" (row-major A, column-major B,
# row-major C) layout. Any other layout string would flip BLayout in the
# generated header while the bridge kept rejecting it at runtime.
ROWCOL_TENSOR_QUANT_SUPPORTED_LAYOUTS = ("rcr",)

# Default tile shape. Single source of truth for the codegen sweep default and the
# runtime default_{fp8,bf8}_config() helpers; if these drift apart the two halves
# generate different kernel names and the .so lookup fails at load time.
# Mirrors tile_engine/ops/gemm/grouped_gemm_quant/grouped_gemm_{rowcolquant,
# tensorquant}/configs/default_ci_config.json.
ROWCOL_TENSOR_QUANT_DEFAULT_TILE = {
    "tile_m": 128, "tile_n": 128, "tile_k": 64,
    "warp_m": 2, "warp_n": 2, "warp_k": 1,
    "warp_tile_m": 32, "warp_tile_n": 32, "warp_tile_k": 16,
}

# Default traits, shared for the same reason as the tile above. pad_m is enabled
# because these kernels are used with M values that are not tile-aligned.
ROWCOL_TENSOR_QUANT_DEFAULT_TRAITS = {
    "pipeline": "compv3",
    "epilogue": "cshuffle",
    "scheduler": "intrawave",
    "pad_m": True,
    "pad_n": False,
    "pad_k": True,
    "persistent": False,
    "block_size": 256,
    "k_block_per_cu": 1,
}


def _make_rowcol_tensor_quant_kernel_name(
    op: str,
    dtype: str,
    layout: str,
    pipeline: str,
    epilogue: str,
    scheduler: str,
    pad_m: bool,
    pad_n: bool,
    pad_k: bool,
    persistent: bool,
    tile_m: int, tile_n: int, tile_k: int,
    warp_m: int, warp_n: int, warp_k: int,
    warp_tile_m: int, warp_tile_n: int, warp_tile_k: int,
) -> str:
    """Shared implementation behind make_{rowcolquant,tensorquant}_kernel_name."""
    tile_str = (
        f"{tile_m}x{tile_n}x{tile_k}_"
        f"{warp_m}x{warp_n}x{warp_k}_"
        f"{warp_tile_m}x{warp_tile_n}x{warp_tile_k}"
    )
    return (
        f"grouped_gemm_{op}_{dtype}_{layout}_{pipeline}_{epilogue}_{scheduler}_"
        f"{str(pad_m).capitalize()}_{str(pad_n).capitalize()}_{str(pad_k).capitalize()}_"
        f"{str(persistent).capitalize()}_{tile_str}"
    )


def make_rowcolquant_kernel_name(
    dtype: str,
    layout: str,
    pipeline: str,
    epilogue: str,
    scheduler: str,
    pad_m: bool,
    pad_n: bool,
    pad_k: bool,
    persistent: bool,
    tile_m: int, tile_n: int, tile_k: int,
    warp_m: int, warp_n: int, warp_k: int,
    warp_tile_m: int, warp_tile_n: int, warp_tile_k: int,
) -> str:
    """Return the canonical RowColQuant kernel name used as KERNEL_NAME.

    Both RowColQuantKernelConfig (utils) and RowColQuantKernelSpec (codegen)
    delegate here so the Python side and the compiled .so are byte-exact.
    Matches the naming produced by
    tile_engine/.../grouped_gemm_rowcolquant_instance_builder.py.
    """
    return _make_rowcol_tensor_quant_kernel_name(
        "rowcolquant", dtype, layout, pipeline, epilogue, scheduler,
        pad_m, pad_n, pad_k, persistent,
        tile_m, tile_n, tile_k,
        warp_m, warp_n, warp_k,
        warp_tile_m, warp_tile_n, warp_tile_k,
    )


def make_tensorquant_kernel_name(
    dtype: str,
    layout: str,
    pipeline: str,
    epilogue: str,
    scheduler: str,
    pad_m: bool,
    pad_n: bool,
    pad_k: bool,
    persistent: bool,
    tile_m: int, tile_n: int, tile_k: int,
    warp_m: int, warp_n: int, warp_k: int,
    warp_tile_m: int, warp_tile_n: int, warp_tile_k: int,
) -> str:
    """Return the canonical TensorQuant kernel name used as KERNEL_NAME.

    See make_rowcolquant_kernel_name; the two differ only in the operator segment.
    """
    return _make_rowcol_tensor_quant_kernel_name(
        "tensorquant", dtype, layout, pipeline, epilogue, scheduler,
        pad_m, pad_n, pad_k, persistent,
        tile_m, tile_n, tile_k,
        warp_m, warp_n, warp_k,
        warp_tile_m, warp_tile_n, warp_tile_k,
    )



# Non-grouped gemm_aquant kernel name construction
# ============================================================================
# NOTE: distinct from the grouped_gemm_aquant helpers above. Both families used
# to share the name make_aquant_kernel_name / aquant_effective_epilogue in this
# module; because Python keeps only the last def, the grouped consumers silently
# picked up the non-grouped versions (different signature -> TypeError). These
# are renamed with a gemm_ prefix so each family binds its own helper.


def gemm_aquant_effective_epilogue(
    tile_n: int,
    warp_n: int,
    warp_tile_n: int,
    quant_group_n: int,
    requested_epilogue: str = "cshuffle",
) -> str:
    """Return the epilogue tag the aquant codegen will actually emit.

    Two orthogonal decisions:
      1. PermuteN vs non-PermuteN (TiledMMAPermuteN in run_gemm_quant_example.inc).
         AQuant decode/preshufflequant configs never enable TiledMMAPermuteN
         (PreshuffleB is rejected for AQuant), so PermuteN is never selected.
      2. CShuffle vs Default epilogue -- the sweep's ``epilogue`` trait, which
         Old-TE's gemm_instance_builder honors via populate_{cshuffle,default}_gemm_aquant.
    Since PermuteN is never used, the effective tag is just the requested trait
    ("cshuffle" or "default"), which keeps the bridge kernel name byte-exact with
    the matched Old-TE stem (..._mem_default_... vs ..._mem_cshuffle_...).
    """
    _ = (tile_n, warp_n, warp_tile_n, quant_group_n)
    # AQuant never enables TiledMMAPermuteN; the epilogue is whatever was requested.
    return requested_epilogue


def make_gemm_aquant_kernel_name(
    variant_key: str,
    layout: str,
    pipeline: str,
    epilogue: str,  # "cshuffle" or "default" -- sweep epilogue trait, mirrors Old-TE
    scheduler: str,
    tile_m: int, tile_n: int, tile_k: int,
    warp_m: int, warp_n: int, warp_k: int,
    warp_tile_m: int, warp_tile_n: int, warp_tile_k: int,
    quant_group_m: int,
    quant_group_n: int,
    quant_group_k: int,
    preshuffle_aquant: bool = False,
) -> str:
    """Return the canonical AQuant kernel name used as KERNEL_NAME in generated headers.

    Both AQuantKernelConfig (utils) and AQuantKernelSpec (codegen) delegate to this
    function so the two sides stay byte-exact.

    Naming convention:
        gemm_aquant_{variant}_{layout}_{pipeline}_{epilogue}_{scheduler}_
        {TileM}x{TileN}x{TileK}_{WarpM}x{WarpN}x{WarpK}_{WtM}x{WtN}x{WtK}_
        qg{gM}x{gN}x{gK}[_preshufflequant]

    The ``epilogue`` slot in the name is the *effective* epilogue -- see
    gemm_aquant_effective_epilogue (AQuant never uses PermuteN, so it is exactly
    the requested "cshuffle"/"default" trait).
    """
    return make_quant_kernel_name(
        prefix="gemm_aquant",
        variant_key=variant_key,
        layout=layout,
        pipeline=pipeline,
        epilogue=gemm_aquant_effective_epilogue(
            tile_n, warp_n, warp_tile_n, quant_group_n, requested_epilogue=epilogue
        ),
        scheduler=scheduler,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        warp_m=warp_m, warp_n=warp_n, warp_k=warp_k,
        warp_tile_m=warp_tile_m, warp_tile_n=warp_tile_n, warp_tile_k=warp_tile_k,
        group_segments=(f"qg{quant_group_m}x{quant_group_n}x{quant_group_k}",),
        flags=((preshuffle_aquant, "preshufflequant"),),
    )


# ============================================================================


# Non-grouped gemm_abquant kernel name construction
# ============================================================================
# NOTE: distinct from the grouped_gemm_abquant helper above (which takes
# aquant_group_m/n/k + bquant_group_m/n/k + transpose_c). Renamed with a gemm_
# prefix so the grouped and non-grouped families no longer collide on the shared
# name make_abquant_kernel_name in this module.


def make_gemm_abquant_kernel_name(
    variant_key: str,
    layout: str,
    pipeline: str,
    epilogue: str,  # ignored — actual epilogue is computed from tile params via bquant_effective_epilogue
    scheduler: str,
    tile_m: int, tile_n: int, tile_k: int,
    warp_m: int, warp_n: int, warp_k: int,
    warp_tile_m: int, warp_tile_n: int, warp_tile_k: int,
    aquant_group_k: int,
    bquant_group_n: int,
    bquant_group_k: int,
    preshuffle_b: bool = False,
    preshuffle_bquant: bool = False,
    eight_waves: bool = False,
) -> str:
    """Return the canonical ABQuant kernel name used as KERNEL_NAME in generated headers.

    ABQuant kernels quantize BOTH A and B with independent group shapes:
      AQuantGroupSize is always 1x1x{aquant_group_k} (row/K quant on A)
      BQuantGroupSize is 1x{bquant_group_n}x{bquant_group_k}

    Both unified_gemm_abquant_codegen.py (BQuantKernelSpec analogue) and
    gemm_abquant_utils.py's ABQuantKernelConfig delegate here so the two sides
    stay byte-exact.

    The epilogue segment reflects the epilogue actually emitted. For ABQuant the
    PermuteN epilogue is disabled whenever BQuantGroupSize::kN > 1 (mirrors
    ``TiledPermuteN`` in run_gemm_quant_example.inc), so we pass bquant_group_n
    into bquant_effective_epilogue via its quant_group_n slot.
    """
    # The PermuteN epilogue is selected iff GemmConfig::TiledMMAPermuteN is true
    # (run_gemm_quant_example.inc: TiledPermuteN = kN>1 ? false : TiledMMAPermuteN).
    # TiledMMAPermuteN is a per-config-struct property, NOT pure tile geometry:
    # ONLY the preshuffleB configs (GemmConfigPreshuffleB_*_Prefill) override it to
    # (N_Repeat % 2 == 0). The compv3 (GemmConfigABQuantPrefill /
    # GemmConfigPreshuffleBQuantPrefill) and eight_waves (GemmConfig*EightWaves)
    # configs inherit TiledMMAPermuteN=false from GemmConfigBase -> always CShuffle.
    #
    # KNOWN DEFECT, preserved here deliberately: the call below does not forward
    # ``preshuffle_b``, so bquant_effective_epilogue sees its default False and
    # both arms of this branch yield "cshuffle" -- permute_n is unreachable for
    # this family. Pinned by TestQuantKernelNames.
    # test_gemm_abquant_never_emits_permute_n; fixing it changes emitted kernel
    # names and needs its own commit, not this refactor.
    if preshuffle_b and not eight_waves:
        effective_epilogue = bquant_effective_epilogue(
            tile_n, warp_n, warp_tile_n, bquant_group_n
        )
    else:
        effective_epilogue = "cshuffle"
    return make_quant_kernel_name(
        prefix="gemm_abquant",
        variant_key=variant_key,
        layout=layout,
        pipeline=pipeline,
        epilogue=effective_epilogue,
        scheduler=scheduler,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        warp_m=warp_m, warp_n=warp_n, warp_k=warp_k,
        warp_tile_m=warp_tile_m, warp_tile_n=warp_tile_n, warp_tile_k=warp_tile_k,
        group_segments=(
            f"aqg1x1x{aquant_group_k}",
            f"bqg1x{bquant_group_n}x{bquant_group_k}",
        ),
        flags=(
            (preshuffle_b, "preshuffleb"),
            (preshuffle_bquant, "preshufflebq"),
            (eight_waves, "eightwaves"),
        ),
    )


# TensorQuant kernel name construction
# ============================================================================
# Moved here from unified_gemm_tensor_quant_codegen.py, which was the last quant
# family still defining its name builder locally -- and the only one whose utils
# module imported the builder from the codegen module rather than from here. The
# codegen module re-exports both names, so existing importers are unaffected.


def tensor_quant_effective_epilogue(tile_n: int, warp_n: int, warp_tile_n: int) -> str:
    """Return the epilogue tag the codegen will emit for TensorQuant kernels.

    Mirrors run_gemm_quant_example.inc's TensorQuant path:
        TiledPermuteN = GemmConfig::TiledMMAPermuteN   (BQuantGroupSize::kN==1 always here)
    TensorQuant uses GemmConfigQuantDecode, which inherits TiledMMAPermuteN=false
    from GemmConfigBase, so this always returns "cshuffle" for the supported set.
    """
    # quant_group_n is fixed at 1 for TensorQuant (a single scalar scale), and
    # tiled_mma_permute_n=False short-circuits the parity check regardless.
    return quant_effective_epilogue(
        tile_n, warp_n, warp_tile_n, quant_group_n=1, tiled_mma_permute_n=False,
    )


def make_tensor_quant_kernel_name(
    variant_key: str,
    layout: str,
    pipeline: str,
    epilogue: str,  # ignored -- actual epilogue computed via tensor_quant_effective_epilogue
    scheduler: str,
    tile_m: int, tile_n: int, tile_k: int,
    warp_m: int, warp_n: int, warp_k: int,
    warp_tile_m: int, warp_tile_n: int, warp_tile_k: int,
) -> str:
    """Return the canonical TensorQuant kernel name used as KERNEL_NAME.

    The epilogue segment reflects the epilogue the codegen actually emits
    (computed from tile params via tensor_quant_effective_epilogue) so the name
    always matches the compiled kernel. TensorQuant has no quant-group segment:
    the scale is a single scalar for the whole tensor.
    """
    return make_quant_kernel_name(
        prefix="gemm_tensor_quant",
        variant_key=variant_key,
        layout=layout,
        pipeline=pipeline,
        epilogue=tensor_quant_effective_epilogue(tile_n, warp_n, warp_tile_n),
        scheduler=scheduler,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        warp_m=warp_m, warp_n=warp_n, warp_k=warp_k,
        warp_tile_m=warp_tile_m, warp_tile_n=warp_tile_n, warp_tile_k=warp_tile_k,
    )


# ============================================================================
# Shared header emit + CLI driver for the block-scale quant codegen scripts
# ============================================================================
#
# The five per-op codegen scripts (unified_gemm_{tensor_quant,rowcolquant,aquant,
# abquant,bquant}_codegen.py) each emitted an identical .hpp preamble and an
# identical CK_TILE_SINGLE_KERNEL_INCLUDE footer, and each carried a near-verbatim
# copy of the generate_kernels / _generate_one / main() CLI driver. Those blocks
# now live here so a fix happens once; the op-specific header body (pipeline /
# epilogue / QuantType) and the per-op config sweep stay in the per-op scripts.


def emit_generated_header_preamble(title: str, module_name: str, extra: str = "") -> str:
    """Emit the shared auto-generated-header prologue: license, DO-NOT-EDIT line,
    ``#pragma once`` and the four ck_tile includes every quant kernel header needs.

    ``title`` is the human op label (e.g. ``"Gemm TensorQuant"``); ``module_name``
    is the generator script that owns the file. ``extra`` is inserted between the
    includes and the opening ``namespace`` (bquant uses it for its arch guard); it
    should already carry its own surrounding newlines, and defaults to a single
    blank line to match the no-extra layout.
    """
    tail = extra if extra else "\n"
    return f"""\
// SPDX-License-Identifier: MIT
// Auto-generated {title} kernel header.
// DO NOT EDIT -- regenerate via {module_name}
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm_quant.hpp"
#include "ck_tile/ops/epilogue.hpp"
{tail}"""


def emit_single_kernel_include_footer(
    *,
    ns: str,
    struct: str,
    ck_a: str,
    ck_b: str,
    ck_c: str,
    ck_q: str,
    ck_acc: str,
    extra_lines: str = "",
) -> str:
    """Emit the shared ``#ifdef CK_TILE_SINGLE_KERNEL_INCLUDE`` footer.

    Every op re-exports SelectedKernel, KERNEL_NAME and the A/B/C/Q/Acc type
    aliases into the global namespace for the force-include (single-kernel) build.
    Op-specific trailing exports (QuantGroupSize, layouts, GroupSizeK, ...) are
    passed via ``extra_lines`` (each line terminated as the caller wants).
    """
    body = f"""\
#ifdef CK_TILE_SINGLE_KERNEL_INCLUDE
using SelectedKernel = {ns}::{struct};
constexpr const char* KERNEL_NAME = {ns}::KERNEL_NAME;
using ADataType   = {ck_a};
using BDataType   = {ck_b};
using CDataType   = {ck_c};
using QDataType   = {ck_q};
using AccDataType = {ck_acc};
"""
    if extra_lines:
        body += extra_lines if extra_lines.endswith("\n") else extra_lines + "\n"
    body += "#endif // CK_TILE_SINGLE_KERNEL_INCLUDE\n"
    return body


# ============================================================================
# Shared C++ fragments emitted into every quant kernel header
# ============================================================================
#
# The per-op generators build their .hpp body as one big f-string. Large runs of
# that body are byte-identical between operators -- the epilogue block alone was
# repeated twelve times. The emitters below own those runs so a fix lands once.
#
# Each returns a fragment WITHOUT a trailing newline, indented for substitution
# on a line of its own (``{block}``), and with literal braces already resolved --
# do not write ``{{`` here, these produce final text rather than another template.
#
# NOT extracted, deliberately: the per-op tile *flags* block (kPad*, Preshuffle*,
# TransposeC, TiledMMAPermuteN, ...). It looks shared but is not -- the operators
# disagree on which constants exist, on their values, and even on the `=` column
# (``kPadM            =`` in rowcolquant/abquant vs ``kPadM           =`` in
# aquant/bquant). Forcing it through one emitter would take ~10 parameters and
# enshrine that whitespace drift. Left in the per-op scripts.


_QUANT_EPILOGUE_TAIL = {
    "cshuffle": "",
    "permute_n": ",\n                    false,\n                    1",
}


def emit_quant_epilogue_block(kind: str, ns: str) -> str:
    """Emit the ``using GemmEpilogue = ...`` block for a quant kernel body.

    ``kind`` is the *effective* epilogue tag -- ``"cshuffle"`` or ``"permute_n"``,
    i.e. what quant_effective_epilogue returned, not what the user requested. The
    two forms differ only in the class name and in PermuteN's two extra trailing
    template arguments (``false, 1``).
    """
    try:
        tail = _QUANT_EPILOGUE_TAIL[kind]
    except KeyError:
        raise ValueError(
            f"unknown epilogue kind {kind!r}; expected 'cshuffle' or 'permute_n'"
        ) from None
    cls = "CShuffle" if kind == "cshuffle" else "PermuteN"
    return f"""\
            using GemmEpilogue = ck_tile::{cls}Epilogue<
                ck_tile::{cls}EpilogueProblem<
                    typename PipelineProblem::AComputeDataType,
                    typename PipelineProblem::BComputeDataType,
                    ck_tile::tuple<>,
                    AccDataType,
                    CDataType,
                    ck_tile::tuple<>,
                    {ns}::CLayout,
                    ck_tile::element_wise::PassThrough,
                    TilePartitioner::MPerBlock,
                    TilePartitioner::NPerBlock,
                    WarpM, WarpN,
                    WarpTileM, WarpTileN, WarpTileK,
                    TransposeC{tail}>>;"""


def emit_quant_tile_dims(tile: Any, *, block_size: int, k_block_per_cu: int) -> str:
    """Emit the ``TileM``..``kBlockPerCu`` constant block (identical in all ops)."""
    return f"""\
    static constexpr ck_tile::index_t TileM      = {tile.tile_m};
    static constexpr ck_tile::index_t TileN      = {tile.tile_n};
    static constexpr ck_tile::index_t TileK      = {tile.tile_k};
    static constexpr ck_tile::index_t WarpM      = {tile.warp_m};
    static constexpr ck_tile::index_t WarpN      = {tile.warp_n};
    static constexpr ck_tile::index_t WarpK      = {tile.warp_k};
    static constexpr ck_tile::index_t WarpTileM  = {tile.warp_tile_m};
    static constexpr ck_tile::index_t WarpTileN  = {tile.warp_tile_n};
    static constexpr ck_tile::index_t WarpTileK  = {tile.warp_tile_k};
    static constexpr ck_tile::index_t BlockSize  = {block_size};
    static constexpr int               kBlockPerCu = {k_block_per_cu};"""


def emit_quant_tile_shape(partitioner: str = "ck_tile::GemmTile1DPartitioner<TileShape>") -> str:
    """Emit the ``TileShape`` / ``TilePartitioner`` aliases.

    ``TileShape`` is identical in every operator. ``partitioner`` is the only
    divergence: ABQuant uses ``GemmSpatiallyLocalTilePartitioner<TileShape,8,4>``
    for prefill-tile L2 locality; everyone else uses the 1D partitioner.
    """
    return f"""\
    using TileShape = ck_tile::TileGemmShape<
        ck_tile::sequence<TileM, TileN, TileK>,
        ck_tile::sequence<WarpM, WarpN, WarpK>,
        ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>>;

    using TilePartitioner = {partitioner};"""


def emit_quant_gemm_traits(quant_type: str, ns: str) -> str:
    """Emit ``using GemmTraits = ck_tile::TileGemmQuantTraits<...>``.

    ``quant_type`` is the bare ``ck_tile::QuantType`` enumerator name (e.g.
    ``"BQuantGrouped"``) -- the only token that differs between operators. The
    block names the kPad*/Preshuffle*/TransposeC/DoubleSmemBuffer constants the
    per-op flags block must therefore still define.
    """
    return f"""\
    using GemmTraits = ck_tile::TileGemmQuantTraits<
        kPadM, kPadN, kPadK,
        APreshuffleQuant, BPreshuffleQuant, PreshuffleB,
        {ns}::ALayout, {ns}::BLayout, {ns}::CLayout,
        ck_tile::QuantType::{quant_type},
        {ns}::AQLayout, {ns}::BQLayout,
        TransposeC, DoubleSmemBuffer>;"""


_QUANT_LAUNCH_PREAMBLE_DEFAULT = (
    "        // hot-loop / tail dispatch -- mirrors run_gemm_quant_example.inc\n"
)


def emit_quant_launch_prologue(*, splitk_k: str, preamble: str = "") -> str:
    """Emit ``launch()`` down to the opening of the ``Run`` lambda.

    ``splitk_k`` is the third argument to ``get_splitk_batch_k_read`` and is a
    REAL per-op divergence, not incidental: ``WarpTileK`` for aquant / abquant /
    rowcolquant, ``TileK`` for bquant and the grouped ops, ``K1`` for tensor_quant.
    Getting it wrong changes split-K addressing, so it is required, not defaulted.

    ``preamble`` replaces the default one-line comment and covers everything
    between the opening brace and ``const ck_tile::index_t K_split`` -- extra
    commentary, or tensor_quant's ``constexpr ck_tile::index_t K1`` declaration.
    It must end with a newline.
    """
    head = preamble or _QUANT_LAUNCH_PREAMBLE_DEFAULT
    return f"""\
    static float launch(const ck_tile::QuantGemmHostArgs& args,
                        const ck_tile::stream_config& s)
    {{
{head}        const ck_tile::index_t K_split =
            (args.k_batch == 1)
                ? ck_tile::integer_least_multiple(args.K, TileK)
                : ck_tile::get_splitk_batch_k_read(args.K, args.k_batch, {splitk_k});

        const ck_tile::index_t num_loop  = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop          = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        const auto Run = [&](auto has_hot_loop_, auto tail_number_) {{"""


QUANT_LAUNCH_CALL_PLAIN = """\
            return ck_tile::launch_kernel(
                s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));"""


def emit_quant_kernel_attr_launch(eight_waves_expr: str) -> str:
    """Emit the ``kernel_attr<...>`` launch used by abquant and tensor_quant.

    This overload is not interchangeable with QUANT_LAUNCH_CALL_PLAIN: it selects
    a different ``kentry`` specialization and so a different register allocation.
    Ops that mirror Old-TE's launch must keep using it.
    """
    return f"""\
            using k_attr_t = ck_tile::kernel_attr<{eight_waves_expr}>;
            return ck_tile::launch_kernel(
                s,
                ck_tile::make_kernel<kBlockPerCu, k_attr_t>(
                    Kernel{{}}, grids, blocks, 0, kargs));"""


def emit_quant_launch_tail(
    *, quant_type: str, launch_call: str = QUANT_LAUNCH_CALL_PLAIN, extra: str = ""
) -> str:
    """Emit from ``using Kernel = ...`` to the close of ``launch()``.

    ``quant_type`` is the bare ``ck_tile::QuantType`` enumerator. ``launch_call``
    is QUANT_LAUNCH_CALL_PLAIN or the result of emit_quant_kernel_attr_launch.
    ``extra`` is inserted after the grid/block setup and before the launch call
    (tensor_quant uses it for its ``eight_waves`` ``#ifdef``); it must end with a
    newline. The caller still emits the struct's own closing ``};``.
    """
    return f"""\
            using Kernel = ck_tile::QuantGemmKernel<
                TilePartitioner, GemmPipeline, GemmEpilogue,
                ck_tile::QuantType::{quant_type}>;

            auto kargs = Kernel::MakeKernelArgs(args);
            if(!Kernel::IsSupportedArgument(kargs))
                return -1.0f;

            const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
            const dim3 blocks = Kernel::BlockSize();
{extra}{launch_call}
        }};

        return BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
    }}"""


def generate_kernels_generic(
    *,
    op_label: str,
    generator: Any,
    specs: Sequence[Any],
    output_dir: Path,
    parallel: bool = True,
) -> List[Path]:
    """Write one ``<spec.name>.hpp`` per spec via ``generator.generate(spec)``.

    Shared body of every per-op ``generate_kernels`` / ``_generate_one``: make the
    output dir, fan out over specs (threaded when parallel and >1 spec), log
    per-file progress, and swallow+log per-spec failures exactly as before.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if not specs:
        log.warning(
            "No kernel specs produced from config -- check variant_keys and tile_configs"
        )
        return []

    log.info("Generating %d %s kernel headers into %s", len(specs), op_label, output_dir)
    generated: List[Path] = []

    def _generate_one(spec: Any) -> Path:
        header = generator.generate(spec)
        out_path = output_dir / f"{spec.name}.hpp"
        out_path.write_text(header)
        log.info("  wrote %s", out_path.name)
        return out_path

    if parallel and len(specs) > 1:
        with concurrent.futures.ThreadPoolExecutor() as ex:
            futures = {ex.submit(_generate_one, s): s for s in specs}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    generated.append(fut.result())
                except Exception as e:  # noqa: BLE001
                    log.error("Failed generating %s: %s", futures[fut].name, e)
    else:
        for spec in specs:
            try:
                generated.append(_generate_one(spec))
            except Exception as e:  # noqa: BLE001
                log.error("Failed generating %s: %s", spec.name, e)

    log.info("Generated %d / %d headers", len(generated), len(specs))
    return generated


def run_codegen_cli(
    *,
    description: str,
    op_label: str,
    make_generator: Callable[[], Any],
    build_specs: Callable[[dict], Sequence[Any]],
    default_config: Callable[..., dict],
    arch_aware: bool = False,
    default_gfx_arch: str = "gfx950",
) -> int:
    """Shared argparse + config-load + list/generate driver for the quant codegen CLIs.

    ``arch_aware`` adds ``--gfx-arch`` and mirrors the existing per-op behavior
    exactly: generation always uses ``default_config()`` (no arch arg), while
    ``--list-names`` uses ``default_config(gfx_arch)``.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output-dir", type=Path,
        help="Directory to write generated .hpp files (required unless --list-names)")
    parser.add_argument(
        "--config", type=Path,
        help="JSON config file (defaults to built-in sweep)")
    parser.add_argument(
        "--config-json", type=str,
        help="Inline JSON config string")
    parser.add_argument(
        "--no-parallel", action="store_true",
        help="Disable parallel generation")
    parser.add_argument(
        "--list-names", action="store_true",
        help="Print kernel names that would be generated and exit")
    if arch_aware:
        parser.add_argument(
            "--gfx-arch", type=str, default=default_gfx_arch,
            help="Target GPU arch for the built-in default config's arch-derived "
                 "WarpTileK. Ignored when --config/--config-json is given.")
    args = parser.parse_args()

    cfg: Optional[dict] = None
    if args.config_json:
        try:
            cfg = json.loads(args.config_json)
        except json.JSONDecodeError as e:
            log.error("Invalid --config-json: %s", e)
            return 1
    elif args.config:
        with open(args.config) as f:
            cfg = json.load(f)

    if args.list_names:
        list_cfg = cfg or (default_config(args.gfx_arch) if arch_aware else default_config())
        for s in build_specs(list_cfg):
            print(s.name)
        return 0

    if args.output_dir is None:
        parser.error("--output-dir is required unless --list-names is given")

    specs = build_specs(cfg or default_config())
    paths = generate_kernels_generic(
        op_label=op_label,
        generator=make_generator(),
        specs=specs,
        output_dir=args.output_dir,
        parallel=not args.no_parallel,
    )
    return 0 if paths else 1
