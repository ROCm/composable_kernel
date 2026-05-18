#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Shared codegen infrastructure for GEMM, grouped convolution, and FMHA code generators.

Extracted from unified_gemm_codegen.py + arch-aware expansion helpers from conv.
Both unified_gemm_codegen.py and unified_grouped_conv_codegen.py import from here
to eliminate duplication.
"""

import logging
import concurrent.futures
from dataclasses import dataclass
from typing import (
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    List,
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
    }

    DTYPE_TO_CK_QUALIFIED = {
        "fp16": "ck_tile::fp16_t",
        "bf16": "ck_tile::bf16_t",
        "fp32": "float",
        "fp8": "ck_tile::fp8_t",
        "bf8": "ck_tile::bf8_t",
        "int8": "int8_t",
    }

    DTYPE_TO_DISPATCHER = {
        "fp16": "DataType::FP16",
        "bf16": "DataType::BF16",
        "fp32": "DataType::FP32",
        "fp8": "DataType::FP8",
        "bf8": "DataType::BF8",
        "int8": "DataType::INT8",
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
        """Get output datatype (fp8/bf8 -> fp16)."""
        return "fp16" if dtype in ("fp8", "bf8") else dtype


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
) -> List[R]:
    """Run ``generate_fn`` over ``items``, optionally in parallel.

    Logs per-item progress (best-of-conv pattern).
    Returns a flat list of results in completion order.
    """
    results: List[R] = []
    if not items:
        return results

    if parallel and len(items) > 1:
        with concurrent.futures.ThreadPoolExecutor() as executor:
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
