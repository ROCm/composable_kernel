#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Batched Contraction Multiple ABD Code Generator

Generates one .hpp per kernel config for the dispatcher's ctypes path.
Each header defines a SelectedKernel struct with a static launch() method
taking BatchedContractionMultiABDHostArgs — compiled per-kernel via force-include:

    hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE \\
          batched_contraction_multi_abd_ctypes_lib.cpp

Naming convention (byte-exact with ContractionMultiABDKernelConfig.name
in contraction_multi_abd_utils.py):

    contraction_multi_abd_{dtype}_{layout}_{pipeline}_{epilogue}_{scheduler}_
    {pad_m}_{pad_n}_{pad_k}_{persistent}_
    {TileM}x{TileN}x{TileK}_{WarpM}x{WarpN}x{WarpK}_{WtM}x{WtN}x{WtK}_
    na{NumA}_nb{NumB}_nd{NumD}_g{NumDimG}_m{NumDimM}_n{NumDimN}_k{NumDimK}

Reference:
    include/ck_tile/ops/batched_contraction/kernel/batched_contraction_multi_abd_kernel.hpp
    example/ck_tile/53_contraction_multi_abd/
"""

import argparse
import itertools
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow running this script from anywhere by ensuring codegen dir is on the path
_CODEGEN_DIR = Path(__file__).parent
if str(_CODEGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_CODEGEN_DIR))

from codegen_common import TileConfig, parallel_generate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Dtype mappings
# =============================================================================

_DTYPE_TO_CK: Dict[str, str] = {
    "fp16": "ck_tile::half_t",
    "bf16": "ck_tile::bf16_t",
    "fp32": "float",
    "fp8":  "ck_tile::fp8_t",
    "bf8":  "ck_tile::bf8_t",
}

_LAYOUT_TO_CK: Dict[str, str] = {
    "r": "ck_tile::tensor_layout::gemm::RowMajor",
    "c": "ck_tile::tensor_layout::gemm::ColumnMajor",
}

_PIPELINE_TO_CK: Dict[str, str] = {
    "mem":    "ck_tile::GemmPipelineAgBgCrMem",
    "compv3": "ck_tile::GemmPipelineAgBgCrCompV3",
    "compv4": "ck_tile::GemmPipelineAgBgCrCompV4",
}

_SCHEDULER_TO_CK: Dict[str, str] = {
    "intrawave": "ck_tile::GemmPipelineScheduler::Intrawave",
    "interwave": "ck_tile::GemmPipelineScheduler::Interwave",
    "default":   "ck_tile::GemmPipelineScheduler::Default",
}

# Unsupported (pipeline, scheduler) combos — compute pipelines only support intrawave
_UNSUPPORTED_PIPELINE_SCHEDULER = frozenset({
    ("compv3", "interwave"),
    ("compv4", "interwave"),
    ("comp_async", "interwave"),
})

# The only epilogues this generator emits. Anything else must be rejected up front:
# the emitter's if/else would otherwise silently fall through to default2d and
# produce a kernel that does not match the name it is stored under.
SUPPORTED_EPILOGUES = ("cshuffle", "default2d")


def validate_contraction_multi_abd_params(
    *,
    epilogue: str,
    persistent: bool,
    num_a_tensor: int,
    num_b_tensor: int,
) -> None:
    """Reject parameter combinations this operator cannot honour.

    Shared by the codegen spec and the dispatcher-side config (as
    make_contraction_multi_abd_kernel_name is) so the two cannot drift: a
    combination rejected when generating a header must also be rejected when
    a caller builds a config for it, and with the same message.

    Raises ValueError on an unsupported combination.
    """
    if epilogue not in SUPPORTED_EPILOGUES:
        raise ValueError(
            f"Unsupported epilogue: {epilogue!r}. "
            f"Supported values are {list(SUPPORTED_EPILOGUES)}."
        )
    if persistent:
        # batched_contraction_multi_abd_kernel.hpp exposes only GridSize() --
        # there is no MaxOccupancyGridSize(), so a persistent variant cannot be
        # emitted. Fail here rather than generating a header whose name says
        # persistent=True while the kernel is not persistent.
        raise ValueError(
            "persistent=True is not supported by batched_contraction_multi_abd: "
            "the kernel has no persistent (occupancy-limited grid) variant. "
            "Set persistent to false in the config."
        )
    # BatchedContractionMultiABDKernel::launch loops over the (A, B) tensor pairs
    # but *stores* the epilogue result rather than accumulating it, so only the
    # last pair survives -- any count above 1 silently returns a wrong answer.
    # Refuse the combination until the kernel itself accumulates; a hard error
    # beats plausible garbage.
    if num_a_tensor != 1 or num_b_tensor != 1:
        raise ValueError(
            f"num_a_tensor={num_a_tensor}, num_b_tensor={num_b_tensor}: "
            "batched_contraction_multi_abd currently supports only a single A and "
            "a single B tensor. The kernel's (A, B) loop overwrites instead of "
            "accumulating, so larger counts would produce silently incorrect "
            "results. Multiple D tensors (num_d_tensor > 1) are supported."
        )


# =============================================================================
# Kernel name construction (byte-exact with Python utils side)
# =============================================================================


def make_contraction_multi_abd_kernel_name(
    *,
    dtype: str,
    layout: str,
    pipeline: str,
    epilogue: str,
    scheduler: str,
    pad_m: bool,
    pad_n: bool,
    pad_k: bool,
    persistent: bool,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    warp_m: int,
    warp_n: int,
    warp_k: int,
    warp_tile_m: int,
    warp_tile_n: int,
    warp_tile_k: int,
    num_a_tensor: int,
    num_b_tensor: int,
    num_d_tensor: int,
    num_dim_g: int,
    num_dim_m: int,
    num_dim_n: int,
    num_dim_k: int,
    a_elementwise: str = "PassThrough",
    b_elementwise: str = "PassThrough",
    cde_elementwise: str = "MultiDAdd",
) -> str:
    """
    Construct the canonical kernel name.

    This function is the single source of truth for the kernel name string.
    It is imported by contraction_multi_abd_utils.py so both sides stay byte-exact.

    Elementwise operation traits are included so that two configs that differ
    only in elementwise op produce distinct names and distinct .so files.
    """
    pad_m_str = "True" if pad_m else "False"
    pad_n_str = "True" if pad_n else "False"
    pad_k_str = "True" if pad_k else "False"
    pers_str  = "True" if persistent else "False"

    tile_str = (
        f"{tile_m}x{tile_n}x{tile_k}_"
        f"{warp_m}x{warp_n}x{warp_k}_"
        f"{warp_tile_m}x{warp_tile_n}x{warp_tile_k}"
    )

    return (
        f"contraction_multi_abd_{dtype}_{layout}"
        f"_{pipeline}_{epilogue}_{scheduler}"
        f"_{pad_m_str}_{pad_n_str}_{pad_k_str}_{pers_str}"
        f"_{tile_str}"
        f"_na{num_a_tensor}_nb{num_b_tensor}_nd{num_d_tensor}"
        f"_g{num_dim_g}_m{num_dim_m}_n{num_dim_n}_k{num_dim_k}"
        f"_ew{a_elementwise}_{b_elementwise}_{cde_elementwise}"
    )


# =============================================================================
# Kernel spec dataclass
# =============================================================================


@dataclass
class ContractionMultiABDKernelSpec:
    dtype: str
    layout: str           # 3-char: e.g. "rcr"
    pipeline: str
    epilogue: str         # "cshuffle" or "default2d"
    scheduler: str

    tile_m: int
    tile_n: int
    tile_k: int
    warp_m: int
    warp_n: int
    warp_k: int
    warp_tile_m: int
    warp_tile_n: int
    warp_tile_k: int

    pad_m: bool = False
    pad_n: bool = False
    pad_k: bool = False
    persistent: bool = False

    num_a_tensor: int = 1
    num_b_tensor: int = 1
    num_d_tensor: int = 1
    num_dim_g: int = 1
    num_dim_m: int = 2
    num_dim_n: int = 2
    num_dim_k: int = 1

    a_elementwise: str = "PassThrough"
    b_elementwise: str = "PassThrough"
    cde_elementwise: str = "MultiDAdd"

    def __post_init__(self):
        validate_contraction_multi_abd_params(
            epilogue=self.epilogue,
            persistent=self.persistent,
            num_a_tensor=self.num_a_tensor,
            num_b_tensor=self.num_b_tensor,
        )

    @property
    def name(self) -> str:
        return make_contraction_multi_abd_kernel_name(
            dtype=self.dtype,
            layout=self.layout,
            pipeline=self.pipeline,
            epilogue=self.epilogue,
            scheduler=self.scheduler,
            pad_m=self.pad_m,
            pad_n=self.pad_n,
            pad_k=self.pad_k,
            persistent=self.persistent,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            warp_m=self.warp_m,
            warp_n=self.warp_n,
            warp_k=self.warp_k,
            warp_tile_m=self.warp_tile_m,
            warp_tile_n=self.warp_tile_n,
            warp_tile_k=self.warp_tile_k,
            num_a_tensor=self.num_a_tensor,
            num_b_tensor=self.num_b_tensor,
            num_d_tensor=self.num_d_tensor,
            num_dim_g=self.num_dim_g,
            num_dim_m=self.num_dim_m,
            num_dim_n=self.num_dim_n,
            num_dim_k=self.num_dim_k,
            a_elementwise=self.a_elementwise,
            b_elementwise=self.b_elementwise,
            cde_elementwise=self.cde_elementwise,
        )


# =============================================================================
# Header generator
# =============================================================================


class ContractionMultiABDHeaderGenerator:
    """Generates a self-contained .hpp file for one ContractionMultiABDKernelSpec."""

    def generate(self, spec: ContractionMultiABDKernelSpec) -> str:
        dtype_ck = _DTYPE_TO_CK.get(spec.dtype)
        if dtype_ck is None:
            raise ValueError(f"Unsupported dtype: {spec.dtype!r}")

        if len(spec.layout) != 3:
            raise ValueError(f"Layout must be 3 chars (e.g. 'rcr'), got {spec.layout!r}")
        a_layout_ck = _LAYOUT_TO_CK[spec.layout[0]]
        b_layout_ck = _LAYOUT_TO_CK[spec.layout[1]]
        e_layout_ck = _LAYOUT_TO_CK[spec.layout[2]]

        pipeline_ck  = _PIPELINE_TO_CK.get(spec.pipeline)
        scheduler_ck = _SCHEDULER_TO_CK.get(spec.scheduler)
        if pipeline_ck is None:
            raise ValueError(f"Unsupported pipeline: {spec.pipeline!r}")
        if scheduler_ck is None:
            raise ValueError(f"Unsupported scheduler: {spec.scheduler!r}")

        if (spec.pipeline, spec.scheduler) in _UNSUPPORTED_PIPELINE_SCHEDULER:
            raise ValueError(
                f"Unsupported (pipeline, scheduler) combo: ({spec.pipeline}, {spec.scheduler})"
            )

        kernel_name = spec.name

        # Build A/B/D type alias lists
        as_types  = ", ".join(f"A{i}DataType" for i in range(spec.num_a_tensor))
        bs_types  = ", ".join(f"B{i}DataType" for i in range(spec.num_b_tensor))
        ds_types  = ", ".join(f"D{i}DataType" for i in range(spec.num_d_tensor))

        as_dtype_defs  = "\n".join(
            f"using A{i}DataType = {dtype_ck};" for i in range(spec.num_a_tensor)
        )
        bs_dtype_defs  = "\n".join(
            f"using B{i}DataType = {dtype_ck};" for i in range(spec.num_b_tensor)
        )
        ds_dtype_defs  = "\n".join(
            f"using D{i}DataType = {dtype_ck};" for i in range(spec.num_d_tensor)
        )
        as_layout_defs = "\n".join(
            f"using A{i}Layout = {a_layout_ck};" for i in range(spec.num_a_tensor)
        )
        bs_layout_defs = "\n".join(
            f"using B{i}Layout = {b_layout_ck};" for i in range(spec.num_b_tensor)
        )
        ds_layout_defs = "\n".join(
            f"using D{i}Layout = {e_layout_ck};" for i in range(spec.num_d_tensor)
        )
        as_layout_list = ", ".join(f"A{i}Layout" for i in range(spec.num_a_tensor))
        bs_layout_list = ", ".join(f"B{i}Layout" for i in range(spec.num_b_tensor))
        ds_layout_list = ", ".join(f"D{i}Layout" for i in range(spec.num_d_tensor))

        pad_m_str = "true"  if spec.pad_m      else "false"
        pad_n_str = "true"  if spec.pad_n      else "false"
        pad_k_str = "true"  if spec.pad_k      else "false"
        dbl_smem  = "true"  if spec.pipeline == "compv4" else "false"

        # Epilogue block. Unknown values are rejected by the spec's __post_init__,
        # so the else branch below is reached only for "default2d".
        if spec.epilogue == "cshuffle":
            epilogue_block = f"""\
    using CDEElementWise = ck_tile::element_wise::{spec.cde_elementwise};

    using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<
        AsDataType,
        BsDataType,
        DsDataType,
        AccDataType,
        EDataType,
        DsLayout,
        ELayout,
        CDEElementWise,
        TileM,
        TileN,
        WarpPerBlock_M,
        WarpPerBlock_N,
        WarpTileM,
        WarpTileN,
        WarpTileK,
        TransposeC>;

    using GemmEpilogue = ck_tile::CShuffleEpilogue<EpilogueProblem>;"""
        else:  # default2d
            epilogue_block = f"""\
    using CDEElementWise = ck_tile::element_wise::{spec.cde_elementwise};

    using EpilogueProblem = ck_tile::DefaultGemm2DEpilogueProblem<
        AsDataType,
        BsDataType,
        DsDataType,
        AccDataType,
        EDataType,
        DsLayout,
        ELayout,
        CDEElementWise,
        TileM,
        TileN,
        kPadM,
        kPadN,
        WarpTileM,
        WarpTileN,
        WarpTileK,
        TransposeC>;

    using GemmEpilogue = ck_tile::DefaultGemm2DEpilogue<EpilogueProblem>;"""

        code = f"""\
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// AUTO-GENERATED by unified_contraction_multi_abd_codegen.py — do not edit manually.
// Force-include this file with -DCK_TILE_SINGLE_KERNEL_INCLUDE to expose
// SelectedKernel and KERNEL_NAME at global scope for batched_contraction_multi_abd_ctypes_lib.cpp.

#pragma once

#include <array>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/batched_contraction.hpp"
#include "ck_tile/ops/batched_contraction_multi_abd.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"

namespace ns_{kernel_name} {{

// ---------------------------------------------------------------------------
// Data type aliases
// ---------------------------------------------------------------------------
{as_dtype_defs}
{bs_dtype_defs}
{ds_dtype_defs}

using EDataType   = {dtype_ck};
using AccDataType = float;

using AsDataType = ck_tile::tuple<{as_types}>;
using BsDataType = ck_tile::tuple<{bs_types}>;
using DsDataType = ck_tile::tuple<{ds_types}>;

// ---------------------------------------------------------------------------
// Tensor count constants
// ---------------------------------------------------------------------------
static constexpr ck_tile::index_t NumATensor = {spec.num_a_tensor};
static constexpr ck_tile::index_t NumBTensor = {spec.num_b_tensor};
static constexpr ck_tile::index_t NumDTensor = {spec.num_d_tensor};
static constexpr ck_tile::index_t NumDimG    = {spec.num_dim_g};
static constexpr ck_tile::index_t NumDimM    = {spec.num_dim_m};
static constexpr ck_tile::index_t NumDimN    = {spec.num_dim_n};
static constexpr ck_tile::index_t NumDimK    = {spec.num_dim_k};

// ---------------------------------------------------------------------------
// Layout aliases -- per-tensor, then tuple for multi-ABD pipeline
// ---------------------------------------------------------------------------
using ALayout  = {a_layout_ck};
using BLayout  = {b_layout_ck};
using ELayout  = {e_layout_ck};
{as_layout_defs}
{bs_layout_defs}
{ds_layout_defs}
using AsLayout = ck_tile::tuple<{as_layout_list}>;
using BsLayout = ck_tile::tuple<{bs_layout_list}>;
using DsLayout = ck_tile::tuple<{ds_layout_list}>;

// ---------------------------------------------------------------------------
// Kernel name (byte-exact with ContractionMultiABDKernelConfig.name)
// ---------------------------------------------------------------------------
constexpr const char* KERNEL_NAME = "{kernel_name}";

// ---------------------------------------------------------------------------
// SelectedKernel — the single kernel baked into this .so
// ---------------------------------------------------------------------------
struct SelectedKernel
{{
    static constexpr ck_tile::index_t BlockSize      = 256;
    static constexpr ck_tile::index_t TileM          = {spec.tile_m};
    static constexpr ck_tile::index_t TileN          = {spec.tile_n};
    static constexpr ck_tile::index_t TileK          = {spec.tile_k};
    static constexpr ck_tile::index_t WarpPerBlock_M = {spec.warp_m};
    static constexpr ck_tile::index_t WarpPerBlock_N = {spec.warp_n};
    static constexpr ck_tile::index_t WarpPerBlock_K = {spec.warp_k};
    static constexpr ck_tile::index_t WarpTileM      = {spec.warp_tile_m};
    static constexpr ck_tile::index_t WarpTileN      = {spec.warp_tile_n};
    static constexpr ck_tile::index_t WarpTileK      = {spec.warp_tile_k};

    static constexpr bool kPadM            = {pad_m_str};
    static constexpr bool kPadN            = {pad_n_str};
    static constexpr bool kPadK            = {pad_k_str};
    static constexpr bool TransposeC       = false;
    static constexpr bool DoubleSmemBuffer = {dbl_smem};

    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<TileM, TileN, TileK>,
        ck_tile::sequence<WarpPerBlock_M, WarpPerBlock_N, WarpPerBlock_K>,
        ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>>;

    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape, 8, 4>;

    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<
        kPadM, kPadN, kPadK,
        DoubleSmemBuffer,
        AsLayout, BsLayout, ELayout,
        TransposeC>;

    using Problem = ck_tile::BatchedContractionMultiABDProblem<
        AsDataType, BsDataType, DsDataType, EDataType,
        NumDimG, NumDimM, NumDimN, NumDimK>;

    using AElementWise = ck_tile::element_wise::{spec.a_elementwise};
    using BElementWise = ck_tile::element_wise::{spec.b_elementwise};

    static constexpr auto scheduler = {scheduler_ck};

    using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<
        AsDataType, BsDataType, AccDataType,
        GemmShape, GemmUniversalTraits,
        scheduler, AElementWise, BElementWise>;

    using GemmPipeline = {pipeline_ck}<UniversalGemmProblem>;

{epilogue_block}

    using Kernel = ck_tile::BatchedContractionMultiABDKernel<
        Problem, TilePartitioner, GemmPipeline, GemmEpilogue>;

    static float launch(
        const ck_tile::BatchedContractionMultiABDHostArgs<
            NumDimG, NumDimM, NumDimN, NumDimK,
            NumATensor, NumBTensor, NumDTensor>& args,
        const ck_tile::stream_config& stream)
    {{
        // Delegate to the wrapper's own launch(): it iterates the (A, B) tensor
        // pairs and dispatches the single-A/B inner kernel for each. The wrapper
        // itself is not a device functor, so it must not be handed to make_kernel.
        return Kernel::launch(args, stream);
    }}
}};

}} // namespace ns_{kernel_name}

// ---------------------------------------------------------------------------
// Re-export to global scope when force-included by the ctypes lib
// ---------------------------------------------------------------------------
#ifdef CK_TILE_SINGLE_KERNEL_INCLUDE

using SelectedKernel = ns_{kernel_name}::SelectedKernel;
constexpr const char* KERNEL_NAME = ns_{kernel_name}::KERNEL_NAME;

// Tensor type/layout aliases (used by ctypes lib for validation and sizing)
using AsDataType = ns_{kernel_name}::AsDataType;
using BsDataType = ns_{kernel_name}::BsDataType;
using DsDataType = ns_{kernel_name}::DsDataType;
using EDataType  = ns_{kernel_name}::EDataType;

using ALayout  = ns_{kernel_name}::ALayout;
using BLayout  = ns_{kernel_name}::BLayout;
using ELayout  = ns_{kernel_name}::ELayout;
using DsLayout = ns_{kernel_name}::DsLayout;

// Tensor count constants
static constexpr ck_tile::index_t NumATensors = ns_{kernel_name}::NumATensor;
static constexpr ck_tile::index_t NumBTensors = ns_{kernel_name}::NumBTensor;
static constexpr ck_tile::index_t NumDTensors = ns_{kernel_name}::NumDTensor;

// Dimension count constants
static constexpr ck_tile::index_t NumDimsG = ns_{kernel_name}::NumDimG;
static constexpr ck_tile::index_t NumDimsM = ns_{kernel_name}::NumDimM;
static constexpr ck_tile::index_t NumDimsN = ns_{kernel_name}::NumDimN;
static constexpr ck_tile::index_t NumDimsK = ns_{kernel_name}::NumDimK;

#endif // CK_TILE_SINGLE_KERNEL_INCLUDE
"""
        return code


# =============================================================================
# Spec enumeration
# =============================================================================


def _expand_nested_config(config: dict) -> dict:
    """
    Convert the JSON file format (tile_config / trait_config nested keys) into
    the flat format that build_specs() reads (pipelines, tile_configs, etc.).

    The JSON files produced by the tile engine ship ranges like:
        "tile_config": {"tile_m": {"min": 64, "max": 256, "step": 64}, "warp_m": {"values": [4,2,1]}}
        "trait_config": {"pipeline": {"values": ["compv3"]}, "scheduler": {"values": ["intrawave"]}}

    build_specs() expects flat keys: dtypes, layouts, pipelines, tile_configs, pad_options, etc.
    If neither nested format is present the dict is returned as-is (already flat).
    """
    if "tile_config" not in config and "trait_config" not in config:
        return config  # already flat (e.g., from to_codegen_config() or inline overrides)

    def _expand_range(spec: dict) -> List[int]:
        if "values" in spec:
            return list(spec["values"])
        mn, mx, st = spec.get("min", 1), spec.get("max", 1), spec.get("step", 1)
        return list(range(mn, mx + 1, st))

    flat = dict(config)  # shallow copy; keeps flat keys from CMake merge (dtypes, layouts, ...)

    tc = config.get("tile_config", {})
    if tc:
        tile_dim_keys = ["tile_m", "tile_n", "tile_k",
                         "warp_m", "warp_n", "warp_k",
                         "warp_tile_m", "warp_tile_n", "warp_tile_k"]
        tile_dim_lists = {k: _expand_range(tc[k]) for k in tile_dim_keys if k in tc}
        tile_cfgs = [
            dict(zip(tile_dim_lists.keys(), combo))
            for combo in itertools.product(*tile_dim_lists.values())
        ]
        flat["tile_configs"] = tile_cfgs

    tr = config.get("trait_config", {})
    if tr:
        # Every trait key must be listed here. An unrecognised key used to be
        # dropped in silence, so a config that said {"dtype": {"values":
        # ["bf16"]}} -- the obvious spelling, given its "pipeline"/"scheduler"
        # siblings -- generated a full sweep of fp16 kernels and reported
        # success. Name the offender instead.
        known_traits = {"dtype", "layout", "pipeline", "scheduler", "epilogue",
                        "pad_m", "pad_n", "pad_k", "persistent"}
        unknown = sorted(set(tr) - known_traits)
        if unknown:
            raise ValueError(
                f"Unknown trait_config key(s): {unknown}. "
                f"Supported keys are {sorted(known_traits)}."
            )

        if "dtype"     in tr: flat["dtypes"]     = list(tr["dtype"]["values"])
        if "layout"    in tr: flat["layouts"]    = list(tr["layout"]["values"])
        if "pipeline"  in tr: flat["pipelines"]  = list(tr["pipeline"]["values"])
        if "scheduler" in tr: flat["schedulers"] = list(tr["scheduler"]["values"])
        if "epilogue"  in tr: flat["epilogues"]  = list(tr["epilogue"]["values"])
        # pad and persistent options — zip into pad_options list of dicts.
        # persistent belongs here because it is part of the kernel name and must
        # reach the spec; leaving it out silently pinned every kernel to False
        # regardless of what the config declared.
        pad_m_vals       = tr.get("pad_m",       {}).get("values", [False])
        pad_n_vals       = tr.get("pad_n",       {}).get("values", [False])
        pad_k_vals       = tr.get("pad_k",       {}).get("values", [False])
        persistent_vals  = tr.get("persistent",  {}).get("values", [False])
        flat["pad_options"] = [
            {"pad_m": pm, "pad_n": pn, "pad_k": pk, "persistent": pers}
            for pm, pn, pk, pers in itertools.product(
                pad_m_vals, pad_n_vals, pad_k_vals, persistent_vals)
        ]

    flat.pop("tile_config", None)
    flat.pop("trait_config", None)
    return flat


def build_specs(config: dict) -> List[ContractionMultiABDKernelSpec]:
    """Enumerate all specs from a config dict."""
    config = _expand_nested_config(config)

    # dtypes/layouts accept anything _DTYPE_TO_CK and the layout decoder know,
    # but not every value reaches a compiling kernel today. Measured on gfx942
    # at 256x256x64 / compv3 / cshuffle: fp16, bf16, fp8 and bf8 build; fp32
    # does not. Of the layouts only rcr builds -- rrr/ccr/crr trip the
    # row-major-B static_assert in gemm_pipeline_ag_bg_cr_comp_v3.hpp, and that
    # still fires with pad_k on, at 128x128x32, and on the mem pipeline, so it
    # is not a tile-shape accident. These are left as values you may pass
    # rather than hard errors because the constraint lives in the pipeline and
    # may lift there; the support matrices record what is actually usable.
    dtypes     = config.get("dtypes",     ["fp16"])
    layouts    = config.get("layouts",    ["rcr"])
    pipelines  = config.get("pipelines",  ["compv3"])
    epilogues  = config.get("epilogues",  ["cshuffle"])
    schedulers = config.get("schedulers", ["intrawave"])

    pad_options = config.get("pad_options", [{"pad_m": False, "pad_n": False, "pad_k": False}])

    tile_cfgs  = config.get("tile_configs", [
        {"tile_m": 256, "tile_n": 256, "tile_k": 64,
         "warp_m": 2,   "warp_n": 2,   "warp_k": 1,
         "warp_tile_m": 32, "warp_tile_n": 32, "warp_tile_k": 16},
    ])

    num_a_tensors = config.get("num_a_tensors", [1])
    num_b_tensors = config.get("num_b_tensors", [1])
    num_d_tensors = config.get("num_d_tensors", [1])
    dim_combos    = config.get("dim_combos", [
        {"num_dim_g": 1, "num_dim_m": 2, "num_dim_n": 2, "num_dim_k": 1}
    ])

    a_elementwise  = config.get("a_elementwise",  "PassThrough")
    b_elementwise  = config.get("b_elementwise",  "PassThrough")
    cde_elementwise = config.get("cde_elementwise", "MultiDAdd")

    specs = []
    for (dtype, layout, pipeline, epilogue, scheduler,
         pad_opt, tile_cfg,
         na, nb, nd, dim_combo) in itertools.product(
            dtypes, layouts, pipelines, epilogues, schedulers,
            pad_options, tile_cfgs,
            num_a_tensors, num_b_tensors, num_d_tensors, dim_combos):

        if (pipeline, scheduler) in _UNSUPPORTED_PIPELINE_SCHEDULER:
            continue

        tc = TileConfig(
            tile_m=tile_cfg["tile_m"], tile_n=tile_cfg["tile_n"], tile_k=tile_cfg["tile_k"],
            warp_m=tile_cfg["warp_m"], warp_n=tile_cfg["warp_n"], warp_k=tile_cfg["warp_k"],
            warp_tile_m=tile_cfg["warp_tile_m"],
            warp_tile_n=tile_cfg["warp_tile_n"],
            warp_tile_k=tile_cfg["warp_tile_k"],
        )
        if not tc.is_valid():
            continue

        specs.append(ContractionMultiABDKernelSpec(
            dtype=dtype,
            layout=layout,
            pipeline=pipeline,
            epilogue=epilogue,
            scheduler=scheduler,
            tile_m=tc.tile_m, tile_n=tc.tile_n, tile_k=tc.tile_k,
            warp_m=tc.warp_m, warp_n=tc.warp_n, warp_k=tc.warp_k,
            warp_tile_m=tc.warp_tile_m,
            warp_tile_n=tc.warp_tile_n,
            warp_tile_k=tc.warp_tile_k,
            pad_m=pad_opt.get("pad_m", False),
            pad_n=pad_opt.get("pad_n", False),
            pad_k=pad_opt.get("pad_k", False),
            persistent=pad_opt.get("persistent", False),
            num_a_tensor=na,
            num_b_tensor=nb,
            num_d_tensor=nd,
            num_dim_g=dim_combo["num_dim_g"],
            num_dim_m=dim_combo["num_dim_m"],
            num_dim_n=dim_combo["num_dim_n"],
            num_dim_k=dim_combo["num_dim_k"],
            a_elementwise=a_elementwise,
            b_elementwise=b_elementwise,
            cde_elementwise=cde_elementwise,
        ))

    return specs


# =============================================================================
# Generation entry points
# =============================================================================


def generate_one(spec: ContractionMultiABDKernelSpec, output_dir: Path) -> Optional[Path]:
    gen = ContractionMultiABDHeaderGenerator()
    code = gen.generate(spec)
    out_path = output_dir / f"{spec.name}.hpp"
    out_path.write_text(code)
    log.debug("Generated %s", out_path)
    return out_path


def build_capped_specs(config: dict) -> List[ContractionMultiABDKernelSpec]:
    """Enumerate specs and apply the ``max_instances`` cap, if one is configured.

    Both the generator and ``--list-name`` go through here so the two always agree
    on the kernel set. Listing the uncapped names while generating a capped subset
    makes CMake expect headers that were never written.
    """
    specs = build_specs(config)

    # Honor max_instances cap if set (from CONTRACTION_MULTI_ABD_MAX_INSTANCES CMake var).
    max_instances = config.get("max_instances", None)
    if max_instances:
        try:
            cap = int(max_instances)
        except (ValueError, TypeError):
            log.warning("Ignoring non-integer max_instances=%r", max_instances)
            return specs
        if cap > 0 and len(specs) > cap:
            log.info("Capping kernel instances from %d to %d (max_instances=%d)",
                     len(specs), cap, cap)
            specs = specs[:cap]
    return specs


def generate_kernels(output_dir: Path, config: dict, *, max_workers: int = 8) -> List[Path]:
    """Generate all kernel headers in parallel, return list of paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = build_capped_specs(config)

    log.info("Generating %d kernel headers in %s", len(specs), output_dir)

    def _gen(spec):
        return generate_one(spec, output_dir)

    return parallel_generate(_gen, specs, max_workers=max_workers)


# =============================================================================
# CLI
# =============================================================================


_DEFAULT_CONFIG: dict = {
    "dtypes":     ["fp16"],
    "layouts":    ["rcr"],
    "pipelines":  ["compv3"],
    "epilogues":  ["cshuffle"],
    "schedulers": ["intrawave"],
    "pad_options": [{"pad_m": False, "pad_n": False, "pad_k": False}],
    "tile_configs": [
        {"tile_m": 256, "tile_n": 256, "tile_k": 64,
         "warp_m": 2,   "warp_n": 2,   "warp_k": 1,
         "warp_tile_m": 32, "warp_tile_n": 32, "warp_tile_k": 16},
    ],
    "num_a_tensors": [1],
    "num_b_tensors": [1],
    "num_d_tensors": [1],
    "dim_combos": [
        {"num_dim_g": 1, "num_dim_m": 2, "num_dim_n": 2, "num_dim_k": 1}
    ],
    "a_elementwise":   "PassThrough",
    "b_elementwise":   "PassThrough",
    "cde_elementwise": "MultiDAdd",
}


def _parse_args():
    p = argparse.ArgumentParser(
        description="Generate batched_contraction_multi_abd kernel headers."
    )
    p.add_argument("--output-dir", required=True, help="Directory to write .hpp files")
    p.add_argument("--config",     default=None,  help="JSON config file (optional)")
    p.add_argument("--list-name",  action="store_true",
                   help="Print the kernel name for the default config (for --list-name use)")
    p.add_argument("--max-workers", type=int, default=8, help="Codegen parallelism")
    return p.parse_args()


def main():
    args = _parse_args()

    config = dict(_DEFAULT_CONFIG)
    if args.config:
        import json
        with open(args.config) as f:
            config.update(json.load(f))

    if args.list_name:
        # Must use the same capped set the generator writes, otherwise CMake is
        # told to expect headers that generate_kernels() never produces.
        specs = build_capped_specs(config)
        for s in specs:
            print(s.name)
        return

    paths = generate_kernels(Path(args.output_dir), config, max_workers=args.max_workers)
    log.info("Wrote %d headers", len(paths))


if __name__ == "__main__":
    main()
