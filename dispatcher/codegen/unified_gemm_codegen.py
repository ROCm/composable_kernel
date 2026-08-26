#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Unified GEMM Code Generator - Single Source of Truth

This is THE unified code generator for all GEMM kernel variants:
- Standard GEMM (C = A x B)
- Preshuffle GEMM (optimized weight access)
- Multi-D GEMM (element-wise fusion)

Generates both CK Tile kernels AND dispatcher wrappers in one pass.
Replaces all tile_engine GEMM codegen.
"""

import json
import argparse
import itertools
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import concurrent.futures

from codegen_common import (
    TileConfig,
    TraitConfigBase,
    CommonTypeMappings as TypeMappings,
)

# Import architecture filter for GPU-specific validation
try:
    from arch_filter import ArchFilter, KernelConfig as ArchKernelConfig, OperatorType

    HAS_ARCH_FILTER = True
except ImportError:
    HAS_ARCH_FILTER = False
    ArchFilter = None
    ArchKernelConfig = None
    OperatorType = None


# =============================================================================
# Preshuffle Validation (copied from tile_engine/ops/commons/gemm_validation_utils.py)
# =============================================================================

ELEMENT_SIZE_MAP = {
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
    "fp64": 8,
    "fp8": 1,
    "bf8": 1,
    "int8": 1,
}


def _validate_preshuffle_vector_load(
    warp_tile_m: int,
    warp_tile_k: int,
    datatype: str,
    m_iter_per_warp: float,
    wave_size: int = 64,
    vector_load_size: int = 16,
) -> bool:
    """
    Validate vector load alignment for preshuffle pipeline.

    Checks: (warp_tile_m * warp_tile_k * elem_size * m_iter_per_warp / wave_size) % vector_load_size == 0
    """
    elem_size = ELEMENT_SIZE_MAP.get(datatype, 2)
    access_size = (warp_tile_m * warp_tile_k * elem_size * m_iter_per_warp) / wave_size
    return access_size % vector_load_size == 0


def _validate_preshuffle_m0_m1_m2(
    tile_m: int,
    tile_k: int,
    warp_m: int,
    warp_n: int,
    warp_k: int,
    datatype: str,
    vector_load_size: int = 16,
    warp_size: int = 64,
) -> bool:
    """
    Validate M0, M1, M2 configuration for preshuffle matrix A row-major layout.
    Ensures proper memory access pattern alignment.
    """
    try:
        elem_size = ELEMENT_SIZE_MAP.get(datatype, 2)
        MPerBlock = tile_m

        # Calculate K1
        K1 = vector_load_size / elem_size
        if K1 != int(K1):
            return False
        K1 = int(K1)

        # Calculate K0
        if tile_k % K1 != 0:
            return False
        K0 = tile_k // K1

        # Calculate M2
        if warp_size % K0 != 0:
            return False
        M2 = warp_size // K0

        # Calculate number of warps
        NumWarps = warp_m * warp_n * warp_k
        M0 = NumWarps

        # Calculate M1
        if (M2 * M0) == 0:
            return False
        if MPerBlock % (M2 * M0) != 0:
            return False
        M1 = MPerBlock // (M2 * M0)

        # Validate: M0 * M1 * M2 == MPerBlock
        return (M0 * M1 * M2) == MPerBlock

    except (ZeroDivisionError, ValueError):
        return False


def is_preshuffle_config_valid(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    warp_m: int,
    warp_n: int,
    warp_k: int,
    warp_tile_m: int,
    warp_tile_n: int,
    warp_tile_k: int,
    datatype: str,
) -> bool:
    """
    Comprehensive preshuffle configuration validation.
    Copied from tile_engine/ops/commons/gemm_validation_utils.py
    """
    # Basic divisibility checks
    if tile_m % (warp_m * warp_tile_m) != 0:
        return False
    if tile_n % (warp_n * warp_tile_n) != 0:
        return False
    if tile_k % (warp_k * warp_tile_k) != 0:
        return False

    # Calculate m_iter_per_warp
    m_iter_per_warp = tile_m / (warp_m * warp_tile_m)

    # Validate vector load alignment
    if not _validate_preshuffle_vector_load(
        warp_tile_m,
        warp_tile_k,
        datatype,
        m_iter_per_warp,
        wave_size=64,
        vector_load_size=16,
    ):
        return False

    # Validate M0/M1/M2 configuration
    if not _validate_preshuffle_m0_m1_m2(
        tile_m,
        tile_k,
        warp_m,
        warp_n,
        warp_k,
        datatype,
        vector_load_size=16,
        warp_size=64,
    ):
        return False

    return True


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

log = logging.getLogger(__name__)


def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


# ============================================================================
# Configuration and Data Structures
# ============================================================================


class GemmVariant(Enum):
    """GEMM kernel variants"""

    STANDARD = "standard"
    PRESHUFFLE = "preshuffle"
    MULTI_D = "multi_d"
    BATCHED = "batched"
    MULTI_ABD = "multi_abd"
    GROUPED = "grouped"
    # Stream-K. COVERAGE LIMITATION: the dispatcher does NOT yet emit the full
    # Old-TE Stream-K tile surface. The kernels generated here are driven by the
    # tile list passed to this codegen, which is narrower than tile_engine's:
    # measured per layout, e.g. fp16/bf16 rcr TE=180 vs DISP=73 tiles (124 TE-only,
    # 17 DISP-only); ccr TE=144 vs DISP=73; fp8/bf8 closer (rcr TE=296 vs DISP=232)
    # but still short. TE-vs-DISP numeric+perf parity is therefore validated
    # per matched tile config, NOT over the whole TE tile space -- "functional
    # equivalence" should be read with that scope. Closing the gap means feeding
    # the missing TE tiles into the tile list (the codegen handles them); the
    # divergent DISP-only tiles are configs TE does not enumerate at all.
    # NOTE: this limitation is inherent only to driving the codegen standalone.
    # When the bridge is implemented on top of this codegen, the tile list is
    # supplied by Tile-Engine directly, so the emitted Stream-K surface matches
    # the full Old-TE tile space by construction and the gap closes.
    STREAM_K = "stream_k"


# TileConfig imported from codegen_common


@dataclass
class TraitConfig(TraitConfigBase):
    """GEMM-specific trait configuration extending TraitConfigBase with persistent mode."""

    persistent: bool = False


@dataclass
class KernelConfig:
    """Complete kernel configuration"""

    tile: TileConfig
    trait: TraitConfig
    variant: GemmVariant = GemmVariant.STANDARD

    # Variant-specific
    preshuffle: bool = False
    permute_n: bool = False  # B-preshuffle permutation variant (preshuffle only)
    elementwise_op: str = "PassThrough"
    num_d_tensors: int = 0
    d_layout: str = "r"  # Layout for D tensors (r=row, c=col) - same for all D tensors
    # Stream-K reduction strategy: "atomic" (partials atomic-add into C),
    # "linear", or "tree" (partials accumulate through a device workspace).
    reduction_strategy: str = "atomic"

    # Multi-ABD variant: arrays of A/B tensors and per-group element-wise ops.
    # These are behavior-affecting and MUST participate in key_name() so distinct
    # tensor counts / elementwise ops never alias to the same kernel.
    num_a_tensors: int = 1
    num_b_tensors: int = 1
    a_elementwise_op: str = "PassThrough"
    b_elementwise_op: str = "PassThrough"
    cde_elementwise_op: str = "PassThrough"

    # Fixed parameters
    block_size: int = 256
    k_block_per_cu: int = 1
    num_wave_groups: int = 1

    def name(self, datatype: str, layout: str) -> str:
        """C++ alias for template instance"""
        return f"ck_tile_gemm_{self.key_name(datatype, layout)}"

    def key_name(self, datatype: str, layout: str) -> str:
        """
        Unique identifier for this kernel configuration.

        All parameters that affect kernel behavior MUST be included to ensure
        unique names for unique configurations:
        - Data type and layout (signature)
        - Tile, warp, warp_tile dimensions (algorithm)
        - Pipeline, epilogue, scheduler (traits)
        - Padding flags (affects divisibility requirements)
        - Persistent mode
        - Preshuffle variant
        - Multi-D: elementwise op, num D tensors, D layout
        - Occupancy: wave groups, k_block_per_cu (if non-default)
        """
        parts = []
        # Signature
        parts.append(f"dt_{datatype}")
        parts.append(f"ly_{layout}")

        # Tile configuration
        parts.append(f"tile_{self.tile.tile_m}x{self.tile.tile_n}x{self.tile.tile_k}")
        parts.append(f"warp_{self.tile.warp_m}x{self.tile.warp_n}x{self.tile.warp_k}")
        parts.append(
            f"wtile_{self.tile.warp_tile_m}x{self.tile.warp_tile_n}x{self.tile.warp_tile_k}"
        )

        # Traits
        parts.append(f"pipe_{self.trait.pipeline}")
        parts.append(f"epi_{self.trait.epilogue}")
        parts.append(f"sched_{self.trait.scheduler}")

        # Padding flags (only if not all True - the common case)
        if not (self.trait.pad_m and self.trait.pad_n and self.trait.pad_k):
            parts.append(
                f"pad{int(self.trait.pad_m)}{int(self.trait.pad_n)}{int(self.trait.pad_k)}"
            )

        # Persistent mode
        if self.trait.persistent:
            parts.append("persist")

        # Preshuffle variant
        if self.preshuffle:
            parts.append("preshuffle")
            if self.permute_n:
                parts.append("permuteN")

        # Batched variant
        if self.variant == GemmVariant.BATCHED:
            parts.append("batched")

        # Multi-D variant: include elementwise op, num tensors, and D layout
        if self.variant == GemmVariant.MULTI_D:
            parts.append(f"ew_{self.elementwise_op}")
            parts.append(f"nd{self.num_d_tensors}")
            parts.append(f"dly_{self.d_layout}")

        # Multi-ABD variant: include per-group tensor counts, all three
        # element-wise ops, and the D layout. Every one of these changes the
        # generated kernel's types, so they must be in the unique key.
        if self.variant == GemmVariant.MULTI_ABD:
            parts.append(f"na{self.num_a_tensors}")
            parts.append(f"nb{self.num_b_tensors}")
            parts.append(f"nd{self.num_d_tensors}")
            parts.append(f"aew_{self.a_elementwise_op}")
            parts.append(f"bew_{self.b_elementwise_op}")
            parts.append(f"cdew_{self.cde_elementwise_op}")
            parts.append(f"dly_{self.d_layout}")
        # Stream-K variant: reduction strategy distinguishes otherwise-identical
        # kernels (each strategy is a separate compiled binary).
        if self.variant == GemmVariant.STREAM_K:
            parts.append(f"redux_{self.reduction_strategy}")

        # Occupancy parameters (only if non-default)
        if self.num_wave_groups != 1:
            parts.append(f"wg{self.num_wave_groups}")
        if self.k_block_per_cu != 1:
            parts.append(f"kbpc{self.k_block_per_cu}")

        return "_".join(parts)

    def dict_items(self):
        """Iterator over (field, value) pairs"""
        return asdict(self).items()


# ============================================================================
# Type Mappings
# ============================================================================


# TypeMappings imported from codegen_common as CommonTypeMappings -> TypeMappings alias


# ============================================================================
# Kernel Name Generator
# ============================================================================


class KernelNaming:
    """Unified kernel naming"""

    @staticmethod
    def generate(config: KernelConfig, datatype: str, layout: str) -> str:
        """Generate kernel name following tile_engine convention"""
        t = config.tile
        tr = config.trait

        # For multi-d / multi-abd, use 4-char layout (abcd), otherwise 3-char (abc).
        # For multi-abd the 4th char is the D layout (A,B,E,D convention); the
        # incoming ``layout`` is the 3-char A,B,E slice.
        if config.variant in (GemmVariant.MULTI_D, GemmVariant.MULTI_ABD):
            full_layout = layout + config.d_layout  # e.g., "rcr" + "r" = "rcrr"
        else:
            full_layout = layout

        name = (
            f"gemm_{datatype}_{full_layout}_{tr.pipeline}_{tr.epilogue}_{tr.scheduler}"
        )
        name += f"_{str(tr.pad_m).capitalize()}_{str(tr.pad_n).capitalize()}"
        name += f"_{str(tr.pad_k).capitalize()}_{str(tr.persistent).capitalize()}"
        name += f"_{t.tile_m}x{t.tile_n}x{t.tile_k}"
        name += f"_{t.warp_m}x{t.warp_n}x{t.warp_k}"
        name += f"_{t.warp_tile_m}x{t.warp_tile_n}x{t.warp_tile_k}"

        # Add variant suffix
        if config.variant == GemmVariant.PRESHUFFLE:
            name += "_preshuffle"
            if config.permute_n:
                name += "_permuteN"
        elif config.variant == GemmVariant.MULTI_D:
            name += f"_multid_{config.elementwise_op}_d{config.num_d_tensors}"
        elif config.variant == GemmVariant.BATCHED:
            name += "_batched"
        elif config.variant == GemmVariant.MULTI_ABD:
            # Encode every behavior-affecting multi-abd parameter so distinct
            # tensor counts / elementwise ops get distinct kernel names.
            name += (
                f"_multiabd_a{config.num_a_tensors}_b{config.num_b_tensors}"
                f"_d{config.num_d_tensors}"
                f"_{config.a_elementwise_op}_{config.b_elementwise_op}"
                f"_{config.cde_elementwise_op}"
            )
        elif config.variant == GemmVariant.GROUPED:
            name += "_grouped"
        elif config.variant == GemmVariant.STREAM_K:
            name += "_streamk"
            # Atomic keeps the bare "_streamk" suffix for name parity with the
            # original single-strategy bridge; linear/tree are disambiguated.
            if config.reduction_strategy != "atomic":
                name += f"_{config.reduction_strategy}"

        return name


# ============================================================================
# CK Tile Kernel Generator
# ============================================================================


class CKTileKernelGenerator:
    """Generates CK Tile kernel instance code"""

    def __init__(self, datatype: str, layout: str):
        self.datatype = datatype
        self.layout = layout
        self.tm = TypeMappings()

    def generate(self, config: KernelConfig) -> str:
        """Generate complete CK Tile kernel"""
        kernel_name = KernelNaming.generate(config, self.datatype, self.layout)

        return f"""{self._header(kernel_name, config)}
{self._types(config, kernel_name)}
{self._selected_kernel_struct(config, kernel_name)}
"""

    def _header(self, kernel_name: str, config: KernelConfig) -> str:
        """Generate header includes"""
        includes = """// SPDX-License-Identifier: MIT
// Auto-generated CK Tile GEMM kernel
#pragma once

#include <cstdint>
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"

"""

        if config.variant == GemmVariant.MULTI_D:
            includes += """
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_multi_d_kernel.hpp"
"""

        if config.variant == GemmVariant.MULTI_ABD:
            includes += """
#include <array>
#include <tuple>
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_multi_abd_kernel.hpp"
"""

        if config.variant == GemmVariant.GROUPED:
            includes += """
#include <vector>
#include <hip/hip_runtime.h>
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp"
"""

        if config.preshuffle:
            includes += """
#include "ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_v2.hpp"
#include "ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_base_policy.hpp"
"""

        if config.variant == GemmVariant.BATCHED:
            includes += """
#include "ck_tile/ops/gemm/kernel/batched_gemm_kernel.hpp"
"""

        if config.variant == GemmVariant.STREAM_K:
            includes += """
#include <functional>
#include <hip/hip_runtime.h>
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/ops/gemm/kernel/streamk_gemm/streamk_gemm_kernel.hpp"
#include "ck_tile/ops/gemm/kernel/streamk_gemm/streamk_gemm_tile_partitioner.hpp"
"""

        return includes

    def _types(self, config: KernelConfig, kernel_name: str) -> str:
        """Generate type definitions - just the namespace import, types are in kernel namespace"""
        # Note: Data types and layouts are now defined inside each kernel's unique namespace
        # to avoid type alias redefinition conflicts when mixing layouts (e.g., RCR + RRR)
        types = """
// Use ck_tile namespace for generated code
using namespace ck_tile;
"""
        return types

    def _kernel_local_types(self, config: KernelConfig) -> str:
        """Generate data type and layout definitions inside kernel namespace"""
        output_dtype = self.tm.get_output_dtype(self.datatype)
        acc_dtype = self.tm.get_acc_dtype(self.datatype)

        return f"""
    // Data types (inside namespace to avoid conflicts across layouts)
    using ADataType = {self.tm.DTYPE_TO_CK[self.datatype]};
    using BDataType = {self.tm.DTYPE_TO_CK[self.datatype]};
    using AccDataType = {self.tm.DTYPE_TO_CK[acc_dtype]};
    using CDataType = {self.tm.DTYPE_TO_CK[output_dtype]};

    // Layouts (inside namespace to avoid conflicts when mixing layouts)
    using ALayout = {self.tm.LAYOUT_TO_CK[self.layout[0]]};
    using BLayout = {self.tm.LAYOUT_TO_CK[self.layout[1]]};
    using CLayout = {self.tm.LAYOUT_TO_CK[self.layout[2]]};
"""

    def _multi_d_types(self, config: KernelConfig) -> str:
        """Generate multi-d type definitions (inside namespace to avoid conflicts)"""
        if config.variant != GemmVariant.MULTI_D:
            return ""

        d_types = ", ".join(["CDataType"] * config.num_d_tensors)
        d_layout_ck = self.tm.LAYOUT_TO_CK[config.d_layout]
        d_layouts = ", ".join([d_layout_ck] * config.num_d_tensors)

        return f"""
// Multi-D types (defined in namespace to avoid conflicts)
using DsDataType = tuple<{d_types}>;
using DLayout = {d_layout_ck};  // D tensor layout (can differ from C)
using DsLayout = tuple<{d_layouts}>;
using ElementWiseFn = element_wise::{config.elementwise_op};
static constexpr index_t NumDTensor = {config.num_d_tensors};
using GemmMultiDArgs = GemmMultiDHostArgs<NumDTensor>;
"""

    def _multi_d_single_include(self, config: KernelConfig) -> str:
        """Multi-D symbol exports + KEY macros for the force-included header.

        The multi_d ctypes lib (multi_d_gemm_ctypes_lib.cpp) force-includes ONE
        generated header and calls SelectedKernel::launch(GemmMultiDArgs, ...)
        directly, so it needs the Multi-D types (NumDTensor / DsDataType /
        DsLayout / DLayout / ElementWiseFn / GemmMultiDArgs) and the num-D /
        elementwise-op signature fields at global scope. These live in the
        kernel's private namespace by default; re-export them here (guarded by
        CK_TILE_SINGLE_KERNEL_INCLUDE) so single-include builds can reach them.
        Standard/preshuffle configs emit nothing extra.
        """
        if config.variant != GemmVariant.MULTI_D:
            return ""
        ns_name = "ns_" + KernelNaming.generate(
            config, self.datatype, self.layout
        ).replace("-", "_")
        return f"""// Multi-D symbol exports for the single-include ctypes lib.
// NB: ALayout/BLayout/CLayout are already exported by the enclosing
// CK_TILE_SINGLE_KERNEL_INCLUDE block; re-exporting them here would be a C++
// redefinition (breaks multi_d_gemm_ctypes_lib.cpp). Only the Multi-D-specific
// types/macros are added below.
using DsDataType     = {ns_name}::DsDataType;
using DsLayout       = {ns_name}::DsLayout;
using DLayout        = {ns_name}::DLayout;
using ElementWiseFn  = {ns_name}::ElementWiseFn;
static constexpr ck_tile::index_t NumDTensor = {ns_name}::NumDTensor;
using GemmMultiDArgs = {ns_name}::GemmMultiDArgs;
// Multi-D signature descriptors (consumed by the ctypes lib / KernelKey).
#define GEMM_KEY_MULTI_D 1
#define GEMM_KEY_NUM_D_TENSORS {config.num_d_tensors}
#define GEMM_KEY_ELEMENTWISE_OP "{config.elementwise_op}"
#define GEMM_KEY_D_LAYOUT "{config.d_layout}"
"""

    def _multi_abd_types(self, config: KernelConfig) -> str:
        """Generate multi-ABD type definitions (inside namespace to avoid conflicts).

        Multi-ABD uses tuples of A, B and D tensors (each group homogeneous in
        dtype/layout here, matching the Tile Engine op) plus three separate
        element-wise functions (A, B, CDE). The layouts mirror the TE builder's
        4-char ``rcrr`` convention: A=layout[0], B=layout[1], E(=C)=layout[2],
        D=config.d_layout.
        """
        if config.variant != GemmVariant.MULTI_ABD:
            return ""

        a_layout_ck = self.tm.LAYOUT_TO_CK[self.layout[0]]
        b_layout_ck = self.tm.LAYOUT_TO_CK[self.layout[1]]
        e_layout_ck = self.tm.LAYOUT_TO_CK[self.layout[2]]
        d_layout_ck = self.tm.LAYOUT_TO_CK[config.d_layout]

        a_types = ", ".join(["ADataType"] * config.num_a_tensors)
        b_types = ", ".join(["BDataType"] * config.num_b_tensors)
        d_types = ", ".join(["CDataType"] * config.num_d_tensors)
        a_layouts = ", ".join([a_layout_ck] * config.num_a_tensors)
        b_layouts = ", ".join([b_layout_ck] * config.num_b_tensors)
        d_layouts = ", ".join([d_layout_ck] * config.num_d_tensors)

        return f"""
// Multi-ABD types (defined in namespace to avoid conflicts).
// EDataType is the output type (aliased to CDataType by the standard path).
using EDataType = CDataType;
using AsDataType = tuple<{a_types}>;
using BsDataType = tuple<{b_types}>;
using DsDataType = tuple<{d_types}>;
using AsLayout = tuple<{a_layouts}>;
using BsLayout = tuple<{b_layouts}>;
using ELayout = {e_layout_ck};
using DsLayout = tuple<{d_layouts}>;
static constexpr index_t NumATensors = {config.num_a_tensors};
static constexpr index_t NumBTensors = {config.num_b_tensors};
static constexpr index_t NumDTensors = {config.num_d_tensors};
using AElementWiseFn = element_wise::{config.a_elementwise_op};
using BElementWiseFn = element_wise::{config.b_elementwise_op};
using CDEElementWiseFn = element_wise::{config.cde_elementwise_op};
using GemmMultiABDArgs =
    GemmMultiABDHostArgs<NumATensors, NumBTensors, NumDTensors>;
"""

    def _multi_abd_global_exports(self, config: KernelConfig, ns_name: str) -> str:
        """Re-export the multi-ABD tensor counts to the global namespace.

        The gemm_multi_abd ctypes lib is registry-bypass: it force-includes this
        header and constructs GemmMultiABDHostArgs<NumATensors, NumBTensors,
        NumDTensors> at global scope. Those counts live inside the kernel's
        namespace, so under CK_TILE_SINGLE_KERNEL_INCLUDE we surface them (and a
        few convenience macros) globally. Empty for non-multi-abd variants.
        """
        if config.variant != GemmVariant.MULTI_ABD:
            return ""
        return f"""
// Multi-ABD tensor counts (re-exported for the registry-bypass ctypes lib).
constexpr index_t NumATensors = {ns_name}::NumATensors;
constexpr index_t NumBTensors = {ns_name}::NumBTensors;
constexpr index_t NumDTensors = {ns_name}::NumDTensors;
#define GEMM_MULTI_ABD_NUM_A {config.num_a_tensors}
#define GEMM_MULTI_ABD_NUM_B {config.num_b_tensors}
#define GEMM_MULTI_ABD_NUM_D {config.num_d_tensors}
"""

    def _selected_kernel_struct(self, config: KernelConfig, kernel_name: str) -> str:
        """Generate SelectedKernel struct with unique name in unique namespace"""
        t = config.tile
        tr = config.trait
        output_dtype = self.tm.get_output_dtype(self.datatype)
        acc_dtype = self.tm.get_acc_dtype(self.datatype)

        # Generate unique struct name and namespace from kernel name
        struct_name = f"Kernel_{kernel_name}"
        # Create valid C++ namespace name (replace invalid chars)
        ns_name = "ns_" + kernel_name.replace("-", "_")

        multi_d_types = self._multi_d_types(config) + self._multi_abd_types(config)

        # Old-TE (gemm_instance_builder.py) wires UsePersistentKernel from the swept
        # 'persistent' flag only for universal/preshuffle/grouped/mx/batched — NOT multi_d,
        # whose kernel path leaves it at the template default (false). Honoring it for multi_d
        # spuriously flips AccumVGPR 224->384 and spills to scratch (~32% slower on small
        # register-bound shapes), so force it off here to stay byte-identical to Old-TE.
        use_persistent_kernel = tr.persistent and config.variant != GemmVariant.MULTI_D

        return f"""
namespace {ns_name} {{
constexpr const char* KERNEL_NAME = "{kernel_name}";

// Data types (inside namespace to avoid conflicts across different kernels)
using ADataType = {self.tm.DTYPE_TO_CK[self.datatype]};
using BDataType = {self.tm.DTYPE_TO_CK[self.datatype]};
using AccDataType = {self.tm.DTYPE_TO_CK[acc_dtype]};
using CDataType = {self.tm.DTYPE_TO_CK[output_dtype]};

// Layouts (inside namespace to avoid conflicts when mixing layouts like RCR + RRR)
using ALayout = {self.tm.LAYOUT_TO_CK[self.layout[0]]};
using BLayout = {self.tm.LAYOUT_TO_CK[self.layout[1]]};
using CLayout = {self.tm.LAYOUT_TO_CK[self.layout[2]]};
{multi_d_types}
struct {struct_name} {{
    // Data types (required by backend as member types)
    using ADataType = {ns_name}::ADataType;
    using BDataType = {ns_name}::BDataType;
    using CDataType = {ns_name}::CDataType;
    using AccDataType = {ns_name}::AccDataType;
    
    // Configuration
    static constexpr index_t BlockSize = {config.block_size};
    static constexpr index_t TileM = {t.tile_m};
    static constexpr index_t TileN = {t.tile_n};
    static constexpr index_t TileK = {t.tile_k};
    static constexpr index_t WarpPerBlock_M = {t.warp_m};
    static constexpr index_t WarpPerBlock_N = {t.warp_n};
    static constexpr index_t WarpPerBlock_K = {t.warp_k};
    static constexpr index_t WarpTileM = {t.warp_tile_m};
    static constexpr index_t WarpTileN = {t.warp_tile_n};
    static constexpr index_t WarpTileK = {t.warp_tile_k};
    
    // Traits
    static constexpr bool kPadM = {str(tr.pad_m).lower()};
    static constexpr bool kPadN = {str(tr.pad_n).lower()};
    static constexpr bool kPadK = {str(tr.pad_k).lower()};
    static constexpr bool TransposeC = false;
    static constexpr bool UsePersistentKernel = {str(use_persistent_kernel).lower()};
    static constexpr bool DoubleSmemBuffer = {str(tr.pipeline == "compv4" or tr.pipeline == "preshufflev2").lower()};
    static constexpr bool UseStructuredSparsity = false;
    static constexpr bool Preshuffle = {str(config.preshuffle).lower()};
    // PermuteN selects the B-preshuffle permutation used by the host-side
    // shuffle_b_permuteN (true) vs shuffle_b (false). The preshuffle ctypes lib
    // reads this value to apply the same permutation the device kernel expects.
    static constexpr bool PermuteN = {str(config.permute_n).lower()};
    static constexpr index_t NumWaveGroups = {config.num_wave_groups};
    
    {self._tile_types(config, ns_name)}
    {self._launch_function(config)}
}};

// Alias for tile_engine style compatibility (when used with -include)
using SelectedKernel = {struct_name};
using SelectedKernelLauncher = {struct_name};
}} // namespace {ns_name}

// Export to global namespace ONLY for single-kernel includes
// Define CK_TILE_SINGLE_KERNEL_INCLUDE before including this header to enable these aliases
#ifdef CK_TILE_SINGLE_KERNEL_INCLUDE
using {struct_name} = {ns_name}::{struct_name};
using SelectedKernel = {ns_name}::{struct_name};
constexpr const char* KERNEL_NAME = {ns_name}::KERNEL_NAME;
using ADataType = {self.tm.DTYPE_TO_CK_QUALIFIED[self.datatype]};
using BDataType = {self.tm.DTYPE_TO_CK_QUALIFIED[self.datatype]};
using CDataType = {self.tm.DTYPE_TO_CK_QUALIFIED[output_dtype]};
using AccDataType = {self.tm.DTYPE_TO_CK_QUALIFIED[acc_dtype]};
// Layout aliases in the global namespace so single-kernel consumers (the
// ctypes lib's fallback key derivation and the preshuffle B-shuffle) can query
// this kernel's operand layouts by type.
{self._multi_abd_global_exports(config, ns_name)}
using ALayout = {ns_name}::ALayout;
using BLayout = {ns_name}::BLayout;
using CLayout = {ns_name}::CLayout;

// KernelKey field descriptors for the force-included kernel.
// The ctypes library builds the registry KernelKey from these so the
// registered entry reflects this kernel's real traits (not a hard-coded
// fp16/rcr default). Enum-valued fields are emitted as the exact strings
// consumed by string_to_dtype/layout/pipeline/scheduler/epilogue in
// kernel_key.hpp; shape/flag fields are emitted as numeric/0-1 literals.
#define GEMM_KEY_DTYPE_A "{self.datatype}"
#define GEMM_KEY_DTYPE_B "{self.datatype}"
#define GEMM_KEY_DTYPE_C "{output_dtype}"
#define GEMM_KEY_DTYPE_ACC "{acc_dtype}"
#define GEMM_KEY_LAYOUT_A "{self.layout[0]}"
#define GEMM_KEY_LAYOUT_B "{self.layout[1]}"
#define GEMM_KEY_LAYOUT_C "{self.layout[2]}"
#define GEMM_KEY_PIPELINE "{tr.pipeline}"
#define GEMM_KEY_SCHEDULER "{tr.scheduler}"
#define GEMM_KEY_EPILOGUE "{tr.epilogue}"
#define GEMM_KEY_TILE_M {t.tile_m}
#define GEMM_KEY_TILE_N {t.tile_n}
#define GEMM_KEY_TILE_K {t.tile_k}
#define GEMM_KEY_WAVE_M {t.warp_m}
#define GEMM_KEY_WAVE_N {t.warp_n}
#define GEMM_KEY_WAVE_K {t.warp_k}
#define GEMM_KEY_WARP_TILE_M {t.warp_tile_m}
#define GEMM_KEY_WARP_TILE_N {t.warp_tile_n}
#define GEMM_KEY_WARP_TILE_K {t.warp_tile_k}
#define GEMM_KEY_BLOCK_SIZE {config.block_size}
#define GEMM_KEY_NUM_WAVE_GROUPS {config.num_wave_groups}
#define GEMM_KEY_PAD_M {int(tr.pad_m)}
#define GEMM_KEY_PAD_N {int(tr.pad_n)}
#define GEMM_KEY_PAD_K {int(tr.pad_k)}
#define GEMM_KEY_PERSISTENT {int(tr.persistent)}
#define GEMM_KEY_DOUBLE_BUFFER {int(tr.pipeline == "compv4" or tr.pipeline == "preshufflev2")}
#define GEMM_KEY_PRESHUFFLE {int(config.preshuffle)}
#define GEMM_KEY_TRANSPOSE_C 0
#define GEMM_KEY_GROUPED 0
#define GEMM_KEY_BATCHED {int(config.variant == GemmVariant.BATCHED)}
#define GEMM_KEY_SPLIT_K 1
{self._multi_d_single_include(config)}#endif // CK_TILE_SINGLE_KERNEL_INCLUDE
"""

    def _tile_types(self, config: KernelConfig, ns_name: str) -> str:
        """Generate tile type definitions - uses namespace-qualified types"""
        return (
            f"""// Tile shape
    using TileShape = TileGemmShape<
        sequence<TileM, TileN, TileK>,
        sequence<WarpPerBlock_M, WarpPerBlock_N, WarpPerBlock_K>,
        sequence<WarpTileM, WarpTileN, WarpTileK>,
        false, false>;
    
    using TilePartitioner = GemmSpatiallyLocalTilePartitioner<TileShape, 8, 4>;
    using Traits = TileGemmTraits<kPadM, kPadN, kPadK, {ns_name}::ALayout, {ns_name}::BLayout, {ns_name}::CLayout, NumWaveGroups>;
    using GemmPipelineProblem = GemmPipelineProblem<ADataType, BDataType, AccDataType, TileShape, Traits>;
    using BaseGemmPipeline = """
            + self.tm.PIPELINE_TO_BASE[config.trait.pipeline]
            + """<GemmPipelineProblem>;"""
        )

    def _launch_function(self, config: KernelConfig) -> str:
        """Generate launch function"""
        if config.variant == GemmVariant.MULTI_ABD:
            return self._launch_function_multi_abd(config)
        if config.variant == GemmVariant.MULTI_D:
            return self._launch_function_multi_d(config)
        if config.variant == GemmVariant.BATCHED:
            return self._launch_function_batched(config)
        if config.variant == GemmVariant.GROUPED:
            return self._launch_function_grouped(config)
        if config.variant == GemmVariant.STREAM_K:
            return self._launch_function_streamk(config)
        if config.preshuffle:
            return self._launch_function_preshuffle(config)
        return self._launch_function_standard(config)

    def _launch_function_standard(self, config: KernelConfig) -> str:
        """Generate launch function for standard GEMM"""
        return f"""
    static float launch(const GemmHostArgs& args, const stream_config& stream) {{
        const index_t k_grain = args.k_batch * TileK;
        const index_t K_split = (args.K + k_grain - 1) / k_grain * TileK;
        const index_t num_loop = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);
        
        float ave_time{{0}};
        
        constexpr auto scheduler = {self.tm.SCHEDULER_TO_CK[config.trait.scheduler]};
        
        using UniversalGemmProblem = UniversalGemmPipelineProblem<
            ADataType, BDataType, AccDataType, TileShape,
            TileGemmUniversalTraits<kPadM, kPadN, kPadK, DoubleSmemBuffer,
                                            ALayout, BLayout, CLayout, TransposeC,
                                            UseStructuredSparsity, UsePersistentKernel,
                                            NumWaveGroups, Preshuffle>,
            scheduler>;
        
        using GemmPipeline = {self.tm.PIPELINE_TO_CK[config.trait.pipeline]}<UniversalGemmProblem>;
        {self._epilogue_code(config)}
        
        using GemmKernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
        
        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {{
            auto kargs = GemmKernel::MakeKernelArgs(args);
            
            if (!GemmKernel::IsSupportedArgument(kargs)) {{
                throw std::runtime_error("Arguments not supported!");
            }}
            
            const dim3 grids = {"GemmKernel::MaxOccupancyGridSize(stream)" if config.trait.persistent else "GemmKernel::GridSize(args.M, args.N, args.k_batch)"};
            const dim3 blocks = GemmKernel::BlockSize();
            
            constexpr int kBlockPerCu = {config.k_block_per_cu};
            ave_time = launch_kernel(stream,
                make_kernel<kBlockPerCu>(GemmKernel{{}}, grids, blocks, 0, kargs));
            
            return ave_time;
        }};

        BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
        return ave_time;
    }}"""

    def _launch_function_batched(self, config: KernelConfig) -> str:
        """Generate launch function for batched GEMM.

        Mirrors the Tile Engine batched_gemm instance
        (gemm_instance_builder.py, kernel_name_prefix == "batched_gemm"):
          * takes ``ck_tile::BatchedGemmHostArgs`` (adds batch_count + per-batch
            strides on top of the single-problem args);
          * uses the batched-specific ``TileGemmUniversalTraits`` (no persistent/
            preshuffle/wave-group template tail);
          * builds ``ck_tile::BatchedGemmKernel`` and sizes the grid with the
            4-arg ``GridSize(M, N, k_batch, batch_count)``.
        Unlike the standard path there is no hot-loop TailHandler branch -- the
        batched kernel launches directly, matching Old-TE.
        """
        return f"""
    static float launch(const ck_tile::BatchedGemmHostArgs& args, const stream_config& stream) {{
        float ave_time{{0}};

        constexpr auto scheduler = {self.tm.SCHEDULER_TO_CK[config.trait.scheduler]};

        using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<
            kPadM, kPadN, kPadK, DoubleSmemBuffer,
            ALayout, BLayout, CLayout, TransposeC>;

        using UniversalGemmProblem = UniversalGemmPipelineProblem<
            ADataType, BDataType, AccDataType, TileShape,
            GemmUniversalTraits,
            scheduler>;

        using GemmPipeline = {self.tm.PIPELINE_TO_CK[config.trait.pipeline]}<UniversalGemmProblem>;
        {self._epilogue_code(config)}

        using GemmKernel = ck_tile::BatchedGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;

        auto kargs = GemmKernel::MakeKernelArgs(args);

        if (!GemmKernel::IsSupportedArgument(kargs)) {{
            throw std::runtime_error("Arguments not supported for batched gemm kernel!");
        }}

        const dim3 grids = GemmKernel::GridSize(args.M, args.N, args.k_batch, args.batch_count);
        const dim3 blocks = GemmKernel::BlockSize();

        constexpr int kBlockPerCu = {config.k_block_per_cu};
        if(args.k_batch > 1) {{
            // Split-K path: the universal GEMM epilogue stores C with
            // memory_operation_enum::atomic_add when k_batch > 1, so C must be
            // zero at the start of EVERY kernel invocation. The benchmarking
            // timing loop re-runs the kernel cold_niters + nrepeat times, so a
            // single pre-launch memset is not enough -- C accumulates across
            // iterations. Pass a hipMemsetAsync callable as the first callable
            // to launch_kernel (the documented ck_tile pattern, see
            // kernel_launch.hpp) so C is re-zeroed before each timed launch.
            // Use hipMemsetAsync on the launch stream (sc.stream_id_): the
            // synchronous hipMemset ignores stream_config and would serialise
            // onto the default stream, skewing timing/correctness whenever a
            // non-default stream is used. Only the batch slices actually
            // written (M*N per batch) need clearing; use the batch stride to
            // cover padded layouts.
            const std::size_t c_bytes =
                static_cast<std::size_t>(args.batch_stride_E) *
                static_cast<std::size_t>(args.batch_count) * sizeof(CDataType);
            ave_time = launch_kernel(stream,
                [=](const stream_config& sc) {{
                    // Check the return: a failed reset would let the split-K
                    // atomic-add epilogue accumulate into a non-zero C and
                    // silently produce wrong results (mirrors the stream-K path).
                    if(hipMemsetAsync(args.e_ptr, 0, c_bytes, sc.stream_id_) != hipSuccess) {{
                        throw std::runtime_error(
                            "batched split-K: hipMemsetAsync failed to reset C between iterations");
                    }}
                }},
                make_kernel<kBlockPerCu>(GemmKernel{{}}, grids, blocks, 0, kargs));
        }} else {{
            // k_batch == 1 (the Old-TE parity default): byte-identical to the
            // Tile Engine batched_gemm launch -- single kernel callable, no
            // memset (the epilogue uses memory_operation_enum::set).
            ave_time = launch_kernel(stream,
                make_kernel<kBlockPerCu>(GemmKernel{{}}, grids, blocks, 0, kargs));
        }}

        return ave_time;
    }}"""

    def _launch_function_grouped(self, config: KernelConfig) -> str:
        """Generate launch function for grouped GEMM.

        Follows the dispatcher's workspace idiom (see grouped_conv stream-K launch in
        unified_grouped_conv_codegen.py): signature is (args, stream); the device
        workspace is allocated internally via DeviceMem rather than passed in. The
        grouped kernel's per-group arg vector is built with MakeKargs, copied to the
        workspace, and the device pointer + group count are passed to the kernel.
        """
        persistent = config.trait.persistent
        grid_expr = (
            "GemmKernel::MaxOccupancyGridSize(stream)"
            if persistent
            else "dim3(kargs.empty() ? 0 : kargs.back().block_end, 1, 1)"
        )
        return f"""
    static float launch(const std::vector<ck_tile::GroupedGemmHostArgs<>>& gemm_descs,
                        const stream_config& stream) {{
        if(gemm_descs.empty()) return 0.0f;

        float ave_time{{0}};

        constexpr auto scheduler = {self.tm.SCHEDULER_TO_CK[config.trait.scheduler]};

        using UniversalGemmProblem = UniversalGemmPipelineProblem<
            ADataType, BDataType, AccDataType, TileShape,
            TileGemmUniversalTraits<kPadM, kPadN, kPadK, DoubleSmemBuffer,
                                    ALayout, BLayout, CLayout, TransposeC,
                                    UseStructuredSparsity, UsePersistentKernel,
                                    NumWaveGroups, Preshuffle>,
            scheduler>;

        using GemmPipeline = {self.tm.PIPELINE_TO_CK[config.trait.pipeline]}<UniversalGemmProblem>;
        {self._epilogue_code(config)}

        using GemmKernel = ck_tile::GroupedGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;

        auto kargs = GemmKernel::MakeKargs(gemm_descs);
        if(!GemmKernel::IsSupportedArgument(kargs)) {{
            throw std::runtime_error("Arguments not supported for grouped gemm kernel");
        }}

        // Workspace allocated internally (dispatcher idiom, mirrors grouped_conv stream-K).
        const std::size_t ws_size = kargs.size() * sizeof(ck_tile::GemmTransKernelArg<>);
        ck_tile::DeviceMem workspace_dev(ws_size);
        HIP_CHECK_ERROR(hipMemcpyWithStream(workspace_dev.GetDeviceBuffer(),
                                            kargs.data(),
                                            ws_size,
                                            hipMemcpyHostToDevice,
                                            stream.stream_id_));

        const dim3 grids  = {grid_expr};
        const dim3 blocks = GemmKernel::BlockSize();

        constexpr int kBlockPerCu = {config.k_block_per_cu};
        ave_time = launch_kernel(stream,
            make_kernel<kBlockPerCu>(GemmKernel{{}}, grids, blocks, 0,
                cast_pointer_to_constant_address_space(workspace_dev.GetDeviceBuffer()),
                kargs.size()));

        return ave_time;
    }}"""

    def _launch_function_preshuffle(self, config: KernelConfig) -> str:
        """Generate launch function for preshuffle GEMM (weight preshuffle variant)

        Preshuffle uses WeightPreshufflePipelineAGmemBGmemCRegV2 which has a different
        API than standard pipelines. It's designed for weight-preshuffled GEMM operations.
        """
        return f"""
    static float launch(const GemmHostArgs& args, const stream_config& stream) {{
        const index_t k_grain = args.k_batch * TileK;
        const index_t K_split = (args.K + k_grain - 1) / k_grain * TileK;
        const index_t num_loop = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);
        
        float ave_time{{0}};
        
        constexpr auto scheduler = GemmPipelineScheduler::Default;  // Preshuffle uses Default scheduler
        
        // Preshuffle uses TileFlatmmShape instead of TileGemmShape for the problem
        using UniversalGemmProblem = UniversalGemmPipelineProblem<
            ADataType, BDataType, AccDataType, TileShape,
            TileGemmUniversalTraits<kPadM, kPadN, kPadK, DoubleSmemBuffer,
                                            ALayout, BLayout, CLayout, TransposeC,
                                            UseStructuredSparsity, UsePersistentKernel,
                                            NumWaveGroups, Preshuffle>,
            scheduler>;
        
        using GemmPipeline = WeightPreshufflePipelineAGmemBGmemCRegV2<UniversalGemmProblem>;
        {self._epilogue_code(config)}
        
        using GemmKernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
        
        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {{
            auto kargs = GemmKernel::MakeKernelArgs(args);
            
            if (!GemmKernel::IsSupportedArgument(kargs)) {{
                throw std::runtime_error("Arguments not supported for preshuffle kernel!");
            }}
            
            const dim3 grids = {"GemmKernel::MaxOccupancyGridSize(stream)" if config.trait.persistent else "GemmKernel::GridSize(args.M, args.N, args.k_batch)"};
            const dim3 blocks = GemmKernel::BlockSize();
            
            constexpr int kBlockPerCu = {config.k_block_per_cu};
            ave_time = launch_kernel(stream,
                make_kernel<kBlockPerCu>(GemmKernel{{}}, grids, blocks, 0, kargs));
            
            return ave_time;
        }};

        BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
        return ave_time;
    }}"""

    def _launch_function_multi_d(self, config: KernelConfig) -> str:
        """Generate launch function for Multi-D GEMM"""
        return f"""
    // Multi-D launch function - takes GemmMultiDHostArgs with D tensor pointers
    static float launch(const GemmMultiDArgs& args, const stream_config& stream) {{
        const index_t k_grain = args.k_batch * TileK;
        const index_t K_split = (args.K + k_grain - 1) / k_grain * TileK;
        const index_t num_loop = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);
        
        float ave_time{{0}};
        
        constexpr auto scheduler = {self.tm.SCHEDULER_TO_CK[config.trait.scheduler]};
        
        using UniversalGemmProblem = UniversalGemmPipelineProblem<
            ADataType, BDataType, AccDataType, TileShape,
            TileGemmUniversalTraits<kPadM, kPadN, kPadK, DoubleSmemBuffer,
                                            ALayout, BLayout, CLayout, TransposeC,
                                            UseStructuredSparsity, UsePersistentKernel,
                                            NumWaveGroups, Preshuffle>,
            scheduler>;
        
        using GemmPipeline = {self.tm.PIPELINE_TO_CK[config.trait.pipeline]}<UniversalGemmProblem>;
        {self._epilogue_code(config)}
        
        // Use GemmKernelMultiD for Multi-D variant
        using GemmKernel = ck_tile::GemmKernelMultiD<TilePartitioner, GemmPipeline, GemmEpilogue>;
        
        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {{
            auto kargs = GemmKernel::MakeKernelArgs(args);
            
            if (!GemmKernel::IsSupportedArgument(kargs)) {{
                throw std::runtime_error("Arguments not supported! Multi-D currently doesn't support k_batch > 1");
            }}
            
            const dim3 grids = GemmKernel::GridSize(args.M, args.N, args.k_batch);
            const dim3 blocks = GemmKernel::BlockSize();
            
            constexpr int kBlockPerCu = {config.k_block_per_cu};
            ave_time = launch_kernel(stream,
                make_kernel<kBlockPerCu>(GemmKernel{{}}, grids, blocks, 0, kargs));
            
            return ave_time;
        }};

        BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
        return ave_time;
    }}
    
    // Overload for standard GemmHostArgs (converts to Multi-D args with empty D tensors)
    static float launch(const GemmHostArgs& args, const stream_config& stream) {{
        std::array<const void*, NumDTensor> empty_ds{{}};
        std::array<index_t, NumDTensor> empty_strides{{}};
        for (index_t i = 0; i < NumDTensor; ++i) {{
            empty_ds[i] = nullptr;
            empty_strides[i] = 0;
        }}
        GemmMultiDArgs multi_d_args{{
            args.a_ptr,
            args.b_ptr,
            empty_ds,
            args.e_ptr,
            args.k_batch,
            args.M,
            args.N,
            args.K,
            args.stride_A,
            args.stride_B,
            empty_strides,
            args.stride_C
        }};
        return launch(multi_d_args, stream);
    }}"""

    def _launch_function_multi_abd(self, config: KernelConfig) -> str:
        """Generate launch function for Multi-ABD GEMM.

        Mirrors the Tile Engine gemm_multi_abd instance builder's ``launch``:
        tuple A/B/E layouts in the traits, tuple A/B dtypes plus the A and B
        element-wise functions in the pipeline problem, and GemmKernelMultiABD
        as the kernel. Multi-ABD supports only k_batch = 1, so it launches the
        kernel directly (no hot-loop tail handler, matching Old-TE).
        """
        return f"""
    // Multi-ABD launch function - takes GemmMultiABDHostArgs with tuple A/B/D.
    static float launch(const GemmMultiABDArgs& args, const stream_config& stream) {{
        float ave_time{{0}};

        constexpr auto scheduler = {self.tm.SCHEDULER_TO_CK[config.trait.scheduler]};

        // Traits use tuple layouts for multi-abd (AsLayout/BsLayout/ELayout).
        using Traits = TileGemmUniversalTraits<kPadM, kPadN, kPadK, DoubleSmemBuffer,
                                               AsLayout, BsLayout, ELayout, TransposeC>;

        using UniversalGemmProblem = UniversalGemmPipelineProblem<
            AsDataType, BsDataType, AccDataType, TileShape, Traits, scheduler,
            AElementWiseFn, BElementWiseFn>;

        using GemmPipeline = {self.tm.PIPELINE_TO_CK[config.trait.pipeline]}<UniversalGemmProblem>;
        {self._epilogue_code(config)}

        using GemmKernel = ck_tile::GemmKernelMultiABD<TilePartitioner, GemmPipeline, GemmEpilogue>;

        auto kargs = GemmKernel::MakeKernelArgs(args);

        if (!GemmKernel::IsSupportedArgument(kargs)) {{
            throw std::runtime_error("Arguments not supported! Multi-ABD only supports k_batch = 1");
        }}

        const dim3 grids = GemmKernel::GridSize(args.M, args.N, args.k_batch);
        const dim3 blocks = GemmKernel::BlockSize();

        constexpr int kBlockPerCu = {config.k_block_per_cu};
        ave_time = launch_kernel(stream,
            make_kernel<kBlockPerCu>(GemmKernel{{}}, grids, blocks, 0, kargs));

        return ave_time;
    }}"""

    def _launch_function_streamk(self, config: KernelConfig) -> str:
        """Generate launch function for Stream-K GEMM (the dispatcher way).

        Stream-K is a single GEMM that splits the K dimension across CUs and
        reduces partial results through a device workspace. Unlike Tile Engine
        (which takes an external workspace pointer), the dispatcher allocates the
        workspace INTERNALLY via DeviceMem inside launch(args, stream).

        The reduction strategy is taken from the config (atomic/linear/tree).
        Atomic: partial tiles atomic-add into C, so C is zeroed before every
        kernel invocation. Linear/Tree: partials accumulate through the device
        workspace, which is zeroed instead. Both are handled by the preprocess
        callback passed to launch_kernel_time_mask.
        """
        reduction_ck = {
            "atomic": "Atomic",
            "linear": "Linear",
            "tree": "Tree",
        }[config.reduction_strategy]
        # The Atomic strategy zeroes C with a row-major hipMemset2D (pitch =
        # stride_E rows of N elems). A column-major C would be zeroed incorrectly
        # and atomic accumulation would then corrupt results, so fail loudly at
        # compile time rather than silently. Linear/Tree zero the workspace, not C,
        # so they carry no such requirement.
        c_rowmajor_assert = (
            """
    static_assert(
        std::is_same_v<ck_tile::remove_cvref_t<CLayout>,
                       ck_tile::tensor_layout::gemm::RowMajor>,
        "Stream-K Atomic reduction requires a row-major C: the hipMemset2D C-reset "
        "assumes row-major layout and would zero a column-major C incorrectly.");
"""
            if config.reduction_strategy == "atomic"
            else ""
        )
        return f"""{c_rowmajor_assert}
    // ---- Stream-K kernel type, hoisted to struct scope so the workspace API
    // ---- (GetWorkSpaceSize + external-workspace launch) can reuse the same type. ----
    static constexpr auto SkScheduler = {self.tm.SCHEDULER_TO_CK[config.trait.scheduler]};
    static constexpr auto SkReductionStrategy = ck_tile::StreamKReductionStrategy::{reduction_ck};
    static constexpr int  SkBlockPerCu = {config.k_block_per_cu};

    using SkGemmUniversalTraits = TileGemmUniversalTraits<kPadM, kPadN, kPadK, DoubleSmemBuffer,
                                        ALayout, BLayout, CLayout, TransposeC,
                                        UseStructuredSparsity, UsePersistentKernel,
                                        NumWaveGroups, Preshuffle>;
    using SkUniversalGemmProblem = UniversalGemmPipelineProblem<
        ADataType, BDataType, AccDataType, TileShape, SkGemmUniversalTraits, SkScheduler>;
    using SkGemmPipeline = {self.tm.PIPELINE_TO_CK[config.trait.pipeline]}<SkUniversalGemmProblem>;
    {self._epilogue_code(config)}
    using SkStreamKTilePartitioner =
        ck_tile::StreamKTilePartitioner<TileShape, SkReductionStrategy, UsePersistentKernel>;
    using StreamKGemmKernel =
        ck_tile::StreamKKernel<SkStreamKTilePartitioner, SkGemmPipeline, GemmEpilogue>;

    // Device workspace (bytes) this kernel needs for `args`. 0 for Atomic;
    // >0 for Linear/Tree. The Dispatcher uses this to size the buffer it owns.
    static std::size_t GetWorkSpaceSize(const ck_tile::StreamKHostArgs& args) {{
        auto kargs = StreamKGemmKernel::MakeKernelArgs(args);
        return StreamKGemmKernel::GetWorkSpaceSize(kargs);
    }}

    // Whether the kernel can actually partition this problem (enough tiles across
    // CUs). Lets the dispatcher's supports() reject too-small problems and fall
    // back to a non-Stream-K kernel instead of throwing at launch.
    static bool IsSupported(const ck_tile::StreamKHostArgs& args) {{
        return StreamKGemmKernel::IsSupportedArgument(StreamKGemmKernel::MakeKernelArgs(args));
    }}

    // Internal-workspace launch: allocates a fresh DeviceMem on every call.
    // Kept unchanged for the bridge ctypes lib and the standalone 03 driver.
    static float launch(const ck_tile::StreamKHostArgs& args, const stream_config& stream) {{
        auto kargs = StreamKGemmKernel::MakeKernelArgs(args);
        const auto ws_size = StreamKGemmKernel::GetWorkSpaceSize(kargs);
        ck_tile::DeviceMem workspace_dev(ws_size);
        workspace_dev.SetZero();
        StreamKGemmKernel::SetWorkSpacePointer(kargs, workspace_dev.GetDeviceBuffer());

        if (!StreamKGemmKernel::IsSupportedArgument(kargs)) {{
            throw std::runtime_error("Arguments not supported for stream-k kernel!");
        }}

        const dim3 grids = StreamKGemmKernel::GridSize(kargs.tile_partitioner);
        const dim3 blocks = StreamKGemmKernel::BlockSize();

        // Atomic reduction accumulates into C, so reset buffers before each run.
        auto reset_data_buffers = [&]() {{
            if constexpr (SkReductionStrategy == ck_tile::StreamKReductionStrategy::Atomic) {{
                // Stride-aware: CLayout is row-major with stride_E elems/row, so a
                // padded C is zeroed correctly (not just the contiguous M*N case).
                if(hipMemset2DAsync(args.e_ptr,
                    args.stride_E * sizeof(CDataType),
                    0,
                    args.N * sizeof(CDataType),
                    args.M,
                    stream.stream_id_) != hipSuccess) {{
                    throw std::runtime_error(
                        "stream-k: hipMemset2DAsync failed to reset C between iterations");
                }}
            }} else {{
                workspace_dev.SetZero();
            }}
        }};
        std::function<void()> preprocess = reset_data_buffers;

        float ave_time = launch_kernel_time_mask(stream, preprocess,
            make_kernel<SkBlockPerCu>(StreamKGemmKernel{{}}, grids, blocks, 0, kargs));
        return ave_time;
    }}

    // External-workspace launch (PR-D): the Dispatcher owns and reuses the
    // reduction buffer and passes it in. `workspace` may be null for Atomic
    // (size 0). The per-iteration reset stays here because it needs CDataType
    // and the reduction strategy, which the dtype-erased Dispatcher lacks.
    static float launch(const ck_tile::StreamKHostArgs& args, const stream_config& stream,
                        void* workspace) {{
        auto kargs = StreamKGemmKernel::MakeKernelArgs(args);
        const auto ws_size = StreamKGemmKernel::GetWorkSpaceSize(kargs);
        if (workspace != nullptr) {{
            StreamKGemmKernel::SetWorkSpacePointer(kargs, workspace);
        }}

        if (!StreamKGemmKernel::IsSupportedArgument(kargs)) {{
            throw std::runtime_error("Arguments not supported for stream-k kernel!");
        }}

        const dim3 grids = StreamKGemmKernel::GridSize(kargs.tile_partitioner);
        const dim3 blocks = StreamKGemmKernel::BlockSize();

        auto reset_data_buffers = [&]() {{
            if constexpr (SkReductionStrategy == ck_tile::StreamKReductionStrategy::Atomic) {{
                // Stride-aware: CLayout is row-major with stride_E elems/row, so a
                // padded C is zeroed correctly (not just the contiguous M*N case).
                if(hipMemset2DAsync(args.e_ptr,
                    args.stride_E * sizeof(CDataType),
                    0,
                    args.N * sizeof(CDataType),
                    args.M,
                    stream.stream_id_) != hipSuccess) {{
                    throw std::runtime_error(
                        "stream-k: hipMemset2DAsync failed to reset C between iterations");
                }}
            }} else {{
                if(hipMemsetAsync(workspace, 0, ws_size, stream.stream_id_) != hipSuccess) {{
                    throw std::runtime_error(
                        "stream-k: hipMemsetAsync failed to reset reduction workspace");
                }}
            }}
        }};
        std::function<void()> preprocess = reset_data_buffers;

        float ave_time = launch_kernel_time_mask(stream, preprocess,
            make_kernel<SkBlockPerCu>(StreamKGemmKernel{{}}, grids, blocks, 0, kargs));
        return ave_time;
    }}"""

    def _epilogue_code(self, config: KernelConfig) -> str:
        """Generate epilogue code"""
        if config.variant == GemmVariant.BATCHED:
            # Respect the requested epilogue: the batched sweep space includes
            # BOTH "cshuffle" and "default" (see the shipped default_config.json
            # trait_config.epilogue). Emitting CShuffle unconditionally would
            # silently build a cshuffle kernel under a "_epi_default" name and
            # skip the cshuffle-store correctness gate (_cshuffle_repeat_ok) for
            # configs that requested "default" -- a name/impl mismatch that also
            # breaks the config->codegen->runtime byte-parity invariant.
            if config.trait.epilogue == "cshuffle":
                return """
        using EpilogueProblem = CShuffleEpilogueProblem<
            ADataType, BDataType, tuple<>, AccDataType, CDataType,
            tuple<>, CLayout, element_wise::PassThrough,
            TilePartitioner::MPerBlock, TilePartitioner::NPerBlock,
            WarpPerBlock_M, WarpPerBlock_N, WarpTileM, WarpTileN, WarpTileK,
            UniversalGemmProblem::TransposeC>;
        using GemmEpilogue = CShuffleEpilogue<EpilogueProblem>;"""
            return """
        using EpilogueProblem = DefaultGemm2DEpilogueProblem<
            ADataType, BDataType, tuple<>, AccDataType, CDataType,
            tuple<>, CLayout, element_wise::PassThrough,
            TilePartitioner::MPerBlock, TilePartitioner::NPerBlock,
            kPadM, kPadN, WarpTileM, WarpTileN, WarpTileK, TransposeC>;
        using GemmEpilogue = DefaultGemm2DEpilogue<EpilogueProblem>;"""
        if config.variant == GemmVariant.MULTI_ABD:
            # Multi-ABD epilogue: tuple A/B/D dtypes and D layouts, EDataType as
            # output, and the CDE element-wise function. Matches the TE builder's
            # CShuffleEpilogueProblem for gemm_multi_abd.
            return """
        using EpilogueProblem = CShuffleEpilogueProblem<
            AsDataType, BsDataType, DsDataType, AccDataType, EDataType,
            DsLayout, ELayout, CDEElementWiseFn,
            TilePartitioner::MPerBlock, TilePartitioner::NPerBlock,
            WarpPerBlock_M, WarpPerBlock_N, WarpTileM, WarpTileN, WarpTileK,
            TransposeC>;
        using GemmEpilogue = CShuffleEpilogue<EpilogueProblem>;"""
        if config.variant == GemmVariant.MULTI_D:
            return """
        using EpilogueProblem = CShuffleEpilogueProblem<
            ADataType, BDataType, DsDataType, AccDataType, CDataType,
            DsLayout, CLayout, ElementWiseFn,
            TilePartitioner::MPerBlock, TilePartitioner::NPerBlock,
            WarpPerBlock_M, WarpPerBlock_N, WarpTileM, WarpTileN, WarpTileK,
            TransposeC, NumWaveGroups, false, 1, 1, DoubleSmemBuffer>;
        using GemmEpilogue = CShuffleEpilogue<EpilogueProblem>;"""
        elif config.trait.epilogue == "cshuffle":
            return """
        using EpilogueProblem = CShuffleEpilogueProblem<
            ADataType, BDataType, tuple<>, AccDataType, CDataType,
            tuple<>, CLayout, element_wise::PassThrough,
            TilePartitioner::MPerBlock, TilePartitioner::NPerBlock,
            WarpPerBlock_M, WarpPerBlock_N, WarpTileM, WarpTileN, WarpTileK,
            TransposeC, NumWaveGroups, false, 1, 1, DoubleSmemBuffer>;
        using GemmEpilogue = CShuffleEpilogue<EpilogueProblem>;"""
        else:
            return """
        using EpilogueProblem = DefaultGemm2DEpilogueProblem<
            ADataType, BDataType, tuple<>, AccDataType, CDataType,
            tuple<>, CLayout, element_wise::PassThrough,
            TilePartitioner::MPerBlock, TilePartitioner::NPerBlock,
            kPadM, kPadN, WarpTileM, WarpTileN, WarpTileK, TransposeC>;
        using GemmEpilogue = DefaultGemm2DEpilogue<EpilogueProblem>;"""


# ============================================================================
# Dispatcher Wrapper Generator
# ============================================================================


class DispatcherWrapperGenerator:
    """Generates dispatcher wrapper code"""

    def __init__(self, datatype: str, layout: str):
        self.datatype = datatype
        self.layout = layout
        self.tm = TypeMappings()

    def generate(
        self, config: KernelConfig, kernel_path: Path, output_dir: Path
    ) -> str:
        """Generate dispatcher wrapper"""
        kernel_name = KernelNaming.generate(config, self.datatype, self.layout)
        output_dtype = self.tm.get_output_dtype(self.datatype)
        acc_dtype = self.tm.get_acc_dtype(self.datatype)
        rel_path = kernel_path.relative_to(output_dir)

        # Stream-K kernels need the Stream-K backend (StreamKHostArgs launch) and
        # the SK key fields, so the registry can tell atomic/linear/tree apart and
        # the right launch path compiles. All other variants use the regular backend.
        is_streamk = config.variant == GemmVariant.STREAM_K
        backend_inc = (
            "generated_tile_backend_streamk.hpp"
            if is_streamk
            else "generated_kernel_backend.hpp"
        )

        sk_fields = ""
        if is_streamk:
            rs = {"atomic": "Atomic", "linear": "Linear", "tree": "Tree"}[
                config.reduction_strategy
            ]
            ws = str(config.reduction_strategy != "atomic").lower()
            sk_fields = f"""
    key.algorithm.pad_m = {str(config.trait.pad_m).lower()};
    key.algorithm.pad_n = {str(config.trait.pad_n).lower()};
    key.algorithm.pad_k = {str(config.trait.pad_k).lower()};
    key.algorithm.streamk = true;
    key.algorithm.reduction_strategy = ::ck_tile::dispatcher::ReductionStrategy::{rs};
    key.algorithm.workspace = {ws};"""

        if is_streamk:
            ret_stmt = (
                "return backends::create_generated_streamk_kernel<KernelStruct, "
                "KernelStruct::ADataType, KernelStruct::BDataType, "
                "KernelStruct::CDataType, KernelStruct::AccDataType>"
                f'(key, "{kernel_name}");'
            )
        else:
            ret_stmt = (
                "return std::make_shared<backends::GeneratedKernelInstance<KernelStruct>>"
                f'(key, "{kernel_name}");'
            )

        return f"""// SPDX-License-Identifier: MIT
// Auto-generated dispatcher wrapper
#pragma once

#include "ck_tile/dispatcher.hpp"
#include "ck_tile/dispatcher/backends/{backend_inc}"
#include "{rel_path}"

namespace ck_tile {{
namespace dispatcher {{
namespace generated {{

using ::ck_tile::dispatcher::KernelInstancePtr;
using ::ck_tile::dispatcher::KernelKey;
using ::ck_tile::dispatcher::DataType;
using ::ck_tile::dispatcher::LayoutTag;
using ::ck_tile::dispatcher::Pipeline;
using ::ck_tile::dispatcher::Scheduler;
using ::ck_tile::dispatcher::Epilogue;
using Priority = ::ck_tile::dispatcher::Registry::Priority;
namespace backends = ::ck_tile::dispatcher::backends;

inline KernelInstancePtr make_{kernel_name}(const std::string& gfx_arch = "gfx942") {{
    // Use the unique kernel struct name
    using KernelStruct = Kernel_{kernel_name};
    
    KernelKey key;
    
    // Signature
    key.signature.dtype_a = {self.tm.DTYPE_TO_DISPATCHER[self.datatype]};
    key.signature.dtype_b = {self.tm.DTYPE_TO_DISPATCHER[self.datatype]};
    key.signature.dtype_c = {self.tm.DTYPE_TO_DISPATCHER[output_dtype]};
    key.signature.dtype_acc = {self.tm.DTYPE_TO_DISPATCHER[acc_dtype]};
    key.signature.layout_a = {self.tm.LAYOUT_TO_DISPATCHER[self.layout[0]]};
    key.signature.layout_b = {self.tm.LAYOUT_TO_DISPATCHER[self.layout[1]]};
    key.signature.layout_c = {self.tm.LAYOUT_TO_DISPATCHER[self.layout[2]]};
    key.signature.transpose_a = false;
    key.signature.transpose_b = false;
    key.signature.grouped = false;
    key.signature.split_k = 1;
    key.signature.elementwise_op = "{config.elementwise_op}";
    key.signature.num_d_tensors = {config.num_d_tensors};
    key.signature.structured_sparsity = false;
    
    // Algorithm
    key.algorithm.tile_shape = {{{config.tile.tile_m}, {config.tile.tile_n}, {config.tile.tile_k}}};
    key.algorithm.wave_shape = {{{config.tile.warp_m}, {config.tile.warp_n}, {config.tile.warp_k}}};
    key.algorithm.warp_tile_shape = {{{config.tile.warp_tile_m}, {config.tile.warp_tile_n}, {config.tile.warp_tile_k}}};
    key.algorithm.pipeline = {self.tm.PIPELINE_TO_DISPATCHER[config.trait.pipeline]};
    key.algorithm.scheduler = {self.tm.SCHEDULER_TO_DISPATCHER[config.trait.scheduler]};
    key.algorithm.epilogue = {self.tm.EPILOGUE_TO_DISPATCHER[config.trait.epilogue]};
    key.algorithm.block_size = {config.block_size};
    key.algorithm.double_buffer = {str(config.trait.pipeline in ("compv4", "preshufflev2")).lower()};
    key.algorithm.persistent = {str(config.trait.persistent).lower()};
    key.algorithm.preshuffle = {str(config.preshuffle).lower()};
    key.algorithm.transpose_c = false;
    key.algorithm.num_wave_groups = {config.num_wave_groups};{sk_fields}

    key.gfx_arch = gfx_arch;

    {ret_stmt}
}}

}}}}}}
"""


# ============================================================================
# Main Unified Generator
# ============================================================================


class UnifiedGemmCodegen:
    """Unified GEMM code generator - single entry point"""

    def __init__(
        self,
        output_dir: Path,
        datatype: str,
        layout: str,
        gpu_target: str = "gfx942",
        config_file: Optional[Path] = None,
        variants: List[GemmVariant] = None,
        use_preselected: Optional[str] = None,
        enable_arch_filter: bool = True,
        kernel_set_name: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.datatype = datatype
        # Support 3-char (rcr) or 4-char (rcrr) layout codes
        # 4th char specifies D tensor layout for multi-d
        self.layout = layout[:3]  # A, B, C layouts
        self.d_layout = (
            layout[3] if len(layout) >= 4 else layout[2]
        )  # D layout (default = C layout)
        self.gpu_target = gpu_target
        self.variants = variants or [GemmVariant.STANDARD]
        self.use_preselected = use_preselected
        self.kernel_set_name = kernel_set_name

        # Create directories - optionally with kernel set subdirectory
        if kernel_set_name:
            self.kernel_dir = self.output_dir / kernel_set_name
        else:
            self.kernel_dir = self.output_dir
        self.kernel_dir.mkdir(parents=True, exist_ok=True)
        self.wrapper_dir = self.kernel_dir / "dispatcher_wrappers"
        self.wrapper_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self.config = self._load_config(config_file)

        # Initialize architecture filter for GPU-specific validation
        self.arch_filter = None
        if enable_arch_filter and HAS_ARCH_FILTER:
            try:
                self.arch_filter = ArchFilter(gpu_target, strict_mode=False)
                log.info(f"Architecture filter enabled for {gpu_target}")
            except ValueError as e:
                log.warning(f"Could not create arch filter: {e}")

        # Initialize generators (use self.layout which is the 3-char A,B,C layout)
        self.ck_gen = CKTileKernelGenerator(datatype, self.layout)
        self.disp_gen = DispatcherWrapperGenerator(datatype, self.layout)

    def _load_config(self, config_file: Optional[Path]) -> Dict:
        """Load or create default configuration"""
        if config_file and config_file.exists():
            with open(config_file) as f:
                return json.load(f)

        # Match tile_engine default configs for GEMM/Preshuffle/Multi-D
        # See: tile_engine/ops/gemm/configs/default_config.json
        #      tile_engine/ops/gemm_preshuffle/configs/default_config.json
        #      tile_engine/ops/gemm_multi_d/configs/default_config.json
        return {
            "tile_config": {
                # tile_m/n/k: 64-256 step 64 = [64, 128, 192, 256]
                "tile_m": [64, 128, 192, 256],
                "tile_n": [64, 128, 192, 256],
                "tile_k": [64, 128, 192, 256],
                # warp configs matching tile_engine
                "warp_m": [1, 2, 4],
                "warp_n": [1, 2, 4],
                "warp_k": [1],
                # warp_tile configs matching tile_engine
                "warp_tile_m": [4, 16, 32],
                "warp_tile_n": [16, 32, 64],
                "warp_tile_k": [8, 16, 32, 64, 128],
            },
            "trait_config": {
                "pipeline": ["compv3", "compv4", "mem"],
                "epilogue": ["cshuffle", "default"],
                "scheduler": ["intrawave", "interwave"],
                "pad_m": [False],
                "pad_n": [False],
                "pad_k": [False],
                "persistent": [False, True],
            },
            "multi_d_config": {
                # Note: Only MultiDAdd and MultiDMultiply are compatible with multi-D GEMM.
                # Relu/Gelu are unary ops with signature (y, x), not multi-D signature (e, c, ds...)
                "elementwise_ops": ["MultiDAdd", "MultiDMultiply"],
                "num_d_tensors": [1, 2],
            },
            "multi_abd_config": {
                # Match the Tile Engine gemm_multi_abd instance builder defaults:
                # 2 A tensors, 2 B tensors, 2 D tensors, all PassThrough ops.
                "num_a_tensors": 2,
                "num_b_tensors": 2,
                "num_d_tensors": 2,
                "a_elementwise_op": "PassThrough",
                "b_elementwise_op": "PassThrough",
                "cde_elementwise_op": "PassThrough",
            },
            "streamk_config": {
                # Each reduction strategy compiles to a separate kernel binary.
                "reduction_strategy": ["atomic", "linear", "tree"],
            },
        }

    def generate_all(self, parallel: bool = True) -> Dict:
        """Generate all kernels.

        When parallel=True, all configs across all variants are collected first,
        then generated concurrently in a single thread pool for maximum throughput.
        """
        log.info("Generating GEMM kernels:")
        log.info(f"  Datatype: {self.datatype}")
        log.info(f"  Layout: {self.layout}")
        log.info(f"  Variants: {[v.value for v in self.variants]}")
        if self.use_preselected:
            log.info(f"  Using preselected set: {self.use_preselected}")

        results = {"kernels": [], "wrappers": [], "failed": []}

        # Collect ALL configs across all variants/preselected sets upfront
        all_configs = []
        if self.use_preselected:
            all_configs = self._get_preselected_configs()
            log.info(f"  Total configurations: {len(all_configs)}")
        else:
            for variant in self.variants:
                configs = self._get_configs_for_variant(variant)
                log.info(f"  {variant.value}: {len(configs)} configurations")
                all_configs.extend(configs)
            log.info(f"  Total across all variants: {len(all_configs)}")

        # Generate all configs in a single parallel pass
        if parallel and all_configs:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self._generate_one, cfg) for cfg in all_configs
                ]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        k, w = future.result()
                        results["kernels"].append(k)
                        results["wrappers"].append(w)
                    except Exception as e:
                        results["failed"].append(str(e))
                        log.error(f"Failed: {e}")
        else:
            for cfg in all_configs:
                try:
                    k, w = self._generate_one(cfg)
                    results["kernels"].append(k)
                    results["wrappers"].append(w)
                except Exception as e:
                    results["failed"].append(str(e))
                    log.error(f"Failed: {e}")

        if results["wrappers"]:
            self._generate_registration_header(results["wrappers"])

        return results

    def _get_preselected_configs(self) -> List[KernelConfig]:
        """Get preselected kernel configurations"""
        try:
            from preselected_kernels import get_preselected_set

            return get_preselected_set(self.use_preselected)
        except ImportError:
            log.warning(
                "preselected_kernels module not found, falling back to config-based generation"
            )
            return []
        except ValueError as e:
            log.error(f"Invalid preselected set: {e}")
            return []

    @staticmethod
    def _cshuffle_repeat_ok(tile: TileConfig) -> bool:
        """CShuffle-store correctness gate (narrowed; see issue #9684).

        The CShuffle epilogue mis-stores the accumulator only for ONE specific
        combination: an ODD per-wave repeat (>1) -- tile / (warp * warp_tile) --
        paired with a 32-wide warp tile in that dimension. GPU-verified on
        gfx942: the tile_m=192 / warp_m=2 / warp_tile_m=32 configs
        (MRepeat = 192/(2*32) = 3) return garbage (built-in verify=fail,
        max_rel ~1.3), while EVERY other non-power-of-two repeat is numerically
        correct -- including MRepeat=3 with warp_tile_m=16 (192/(4*16)) and even
        non-pow2 repeats such as 6 and 12. Confirmed by building all 116
        non-pow2-repeat cshuffle/192 configs and running the op's CPU validation:
        exactly the 26 matching the signature above fail; the other 90 pass. The
        earlier "per-wave repeat must be a power of two" rule was therefore too
        broad and needlessly dropped those 90 valid configs.

        This is CShuffle-specific: the "default" (DefaultGemm2DEpilogue) path
        stores directly (not through the LDS repack) and is numerically correct
        for non-pow2 repeats. Only call this for kernels whose resolved epilogue
        is "cshuffle".
        """

        def _dim_bad(repeat: int, warp_tile: int) -> bool:
            return repeat > 1 and repeat % 2 == 1 and warp_tile == 32

        m_repeat = tile.tile_m // (tile.warp_m * tile.warp_tile_m)
        n_repeat = tile.tile_n // (tile.warp_n * tile.warp_tile_n)
        return not (
            _dim_bad(m_repeat, tile.warp_tile_m)
            or _dim_bad(n_repeat, tile.warp_tile_n)
        )

    def _get_configs_for_variant(self, variant: GemmVariant) -> List[KernelConfig]:
        """Get all configurations for a variant

        Args:
            variant: GEMM variant (STANDARD, PRESHUFFLE, MULTI_D, GROUPED)

        Returns:
            List of valid kernel configurations for the variant
        """
        configs = []

        # Get base configs
        tile_configs = self._get_tile_configs()
        trait_configs = self._get_trait_configs()

        for tile, trait in itertools.product(tile_configs, trait_configs):
            # Perform variant-specific architecture validation against the
            # trait's ACTUAL pipeline/scheduler (not a hard-coded compv4).
            if self.arch_filter and HAS_ARCH_FILTER:
                if not self._is_tile_arch_valid(
                    tile,
                    variant,
                    pipeline=trait.pipeline,
                    scheduler=trait.scheduler,
                ):
                    continue

            if variant in (GemmVariant.STANDARD, GemmVariant.BATCHED):
                # CShuffle-store correctness gate: skip non-pow2 repeat tiles
                # only for the cshuffle epilogue (see _cshuffle_repeat_ok). The
                # "default" epilogue is correct with non-pow2 repeats, so it is
                # NOT gated here. Batched GEMM shares the standard tile/trait
                # sweep -- it only differs in the launch/kernel type.
                if trait.epilogue == "cshuffle" and not self._cshuffle_repeat_ok(tile):
                    continue
                configs.append(KernelConfig(tile=tile, trait=trait, variant=variant))

            elif variant == GemmVariant.STREAM_K:
                # Stream-K reuses the standard trait space but requires the cshuffle
                # epilogue (the only epilogue the stream-K kernel supports). Each
                # reduction strategy (atomic/linear/tree) is a distinct compiled
                # kernel, so we expand one config per requested strategy.
                if trait.epilogue == "cshuffle":
                    streamk_cfg = self.config.get("streamk_config", {})
                    strategies = streamk_cfg.get("reduction_strategy", ["atomic"])
                    for reduction_strategy in strategies:
                        configs.append(
                            KernelConfig(
                                tile=tile,
                                trait=trait,
                                variant=variant,
                                reduction_strategy=reduction_strategy,
                            )
                        )

            elif variant == GemmVariant.PRESHUFFLE:
                # Preshuffle uses a fixed pipeline (preshufflev2) and scheduler
                # (default); the epilogue is swept ([default, cshuffle] in the TE
                # gemm_preshuffle default_config). permute_n selects the B-shuffle
                # permutation and is a global config knob (matches Old-TE).
                # NOTE: for the bridge this value arrives already pinned to False
                # via gemm_utils.py::BRIDGE_PERMUTE_N (its to_codegen_json forces
                # it), even though the TE default_config.json / default_ci_config.json
                # ship permute_n=true -- that TE default is a host-marker for a
                # distinct permuteN pipeline the bridge does not emit, so it does
                # not map to a separate bridged device kernel. The get(...) default
                # here is only the fallback for a raw config with no key.
                permute_n = bool(self.config.get("permute_n", False))
                preshuffle_trait = TraitConfig(
                    pipeline="preshufflev2",
                    epilogue=trait.epilogue,
                    scheduler="default",
                    pad_m=trait.pad_m,
                    pad_n=trait.pad_n,
                    pad_k=trait.pad_k,
                    persistent=trait.persistent,
                )
                # Emit one preshuffle config per (tile, epilogue, persistent),
                # de-duplicating over the swept pipeline/scheduler so a full sweep
                # does not create N identical preshuffle kernels. When the caller
                # already pins the pipeline to preshufflev2 (the bridge
                # single-config path), accept it directly; otherwise collapse the
                # sweep onto its first pipeline (compv3) + scheduler (intrawave).
                is_pinned = (
                    trait.pipeline == "preshufflev2" and trait.scheduler == "default"
                )
                is_sweep_anchor = (
                    trait.pipeline == "compv3" and trait.scheduler == "intrawave"
                )
                if not (is_pinned or is_sweep_anchor):
                    continue
                # The CShuffle-store pow2 repeat gate applies only to the cshuffle
                # epilogue (the default epilogue stores directly and is correct).
                if trait.epilogue == "cshuffle" and not self._cshuffle_repeat_ok(tile):
                    continue
                configs.append(
                    KernelConfig(
                        tile=tile,
                        trait=preshuffle_trait,
                        variant=variant,
                        preshuffle=True,
                        permute_n=permute_n,
                    )
                )

            elif variant == GemmVariant.MULTI_D:
                # CShuffle-store correctness gate: applies only when the
                # (swept) epilogue is cshuffle; the default epilogue is exempt.
                if trait.epilogue == "cshuffle" and not self._cshuffle_repeat_ok(tile):
                    continue
                multi_d = self.config.get("multi_d_config", {})
                for ew_op, num_d in itertools.product(
                    multi_d.get("elementwise_ops", ["MultiDAdd"]),
                    multi_d.get("num_d_tensors", [1]),
                ):
                    configs.append(
                        KernelConfig(
                            tile=tile,
                            trait=trait,
                            variant=variant,
                            elementwise_op=ew_op,
                            num_d_tensors=num_d,
                            d_layout=self.d_layout,  # Use extracted D layout
                        )
                    )

            elif variant == GemmVariant.MULTI_ABD:
                # Multi-ABD always uses the CShuffle epilogue (it fuses the D
                # tensors + CDE elementwise through LDS), so the CShuffle-store
                # power-of-two repeat correctness gate always applies.
                if not self._cshuffle_repeat_ok(tile):
                    continue
                multi_abd = self.config.get("multi_abd_config", {})
                num_a = multi_abd.get("num_a_tensors", 2)
                num_b = multi_abd.get("num_b_tensors", 2)
                num_d = multi_abd.get("num_d_tensors", 2)
                a_ew = multi_abd.get("a_elementwise_op", "PassThrough")
                b_ew = multi_abd.get("b_elementwise_op", "PassThrough")
                cde_ew = multi_abd.get("cde_elementwise_op", "PassThrough")
                configs.append(
                    KernelConfig(
                        tile=tile,
                        trait=trait,
                        variant=variant,
                        num_a_tensors=num_a,
                        num_b_tensors=num_b,
                        num_d_tensors=num_d,
                        a_elementwise_op=a_ew,
                        b_elementwise_op=b_ew,
                        cde_elementwise_op=cde_ew,
                        d_layout=self.d_layout,
                    )
                )

            elif variant == GemmVariant.GROUPED:
                # Grouped GEMM uses the same tile/trait configs as STANDARD —
                # the only difference is the kernel type (GroupedGemmKernel vs GemmKernel)
                configs.append(KernelConfig(tile=tile, trait=trait, variant=variant))

        return configs

    def _get_tile_configs(self) -> List[TileConfig]:
        """Get valid tile configurations, filtered by architecture constraints"""
        tc = self.config["tile_config"]
        configs = []
        rejected_count = 0

        for params in itertools.product(
            tc["tile_m"],
            tc["tile_n"],
            tc["tile_k"],
            tc["warp_m"],
            tc["warp_n"],
            tc["warp_k"],
            tc["warp_tile_m"],
            tc["warp_tile_n"],
            tc["warp_tile_k"],
        ):
            tile = TileConfig(*params)

            # Basic validation
            if not tile.is_valid():
                rejected_count += 1
                continue

            # NOTE: the CShuffle-store pow2 MRepeat/NRepeat correctness gate is
            # NOT applied here. It is epilogue-specific (only the CShuffle
            # epilogue mis-stores non-pow2 repeats; the "default" epilogue is
            # correct), so it is applied per (tile, trait) in
            # _get_configs_for_variant once the epilogue is known. See
            # _cshuffle_repeat_ok.

            # Architecture-specific validation. This is a pre-filter run before
            # tiles are paired with traits, so keep a tile if it is legal under
            # ANY configured pipeline/scheduler; the precise per-trait check
            # happens later in _get_configs_for_variant. Filtering here with a
            # single hard-coded pipeline (compv4) wrongly dropped tiles that are
            # legal under mem/compv3.
            if self.arch_filter and HAS_ARCH_FILTER:
                trait_cfg = self.config.get("trait_config", {})
                pipelines = trait_cfg.get("pipeline") or ["compv4"]
                schedulers = trait_cfg.get("scheduler") or ["intrawave"]
                if not any(
                    self._is_tile_arch_valid(tile, pipeline=pl, scheduler=sc)
                    for pl in pipelines
                    for sc in schedulers
                ):
                    rejected_count += 1
                    continue

            configs.append(tile)

        if rejected_count > 0:
            log.debug(f"Rejected {rejected_count} tile configs for {self.gpu_target}")

        return configs

    def _is_tile_arch_valid(
        self,
        tile: TileConfig,
        variant: GemmVariant = None,
        pipeline: str = None,
        scheduler: str = None,
    ) -> bool:
        """Check if tile configuration is valid for target architecture

        Args:
            tile: Tile configuration to validate
            variant: GEMM variant (affects operator-specific constraints)
            pipeline: Trait pipeline to validate against. Pass the config's
                actual pipeline -- omitting it falls back to ``compv4``, whose
                MFMA constraints are stricter than ``mem``/``compv3`` and would
                wrongly reject tiles that are legal under those pipelines.
            scheduler: Trait scheduler to validate against (defaults to
                ``intrawave`` for the same reason).
        """
        if not self.arch_filter or not HAS_ARCH_FILTER:
            return True

        # Determine data types based on self.datatype
        # Note: dtype_c is the ACCUMULATOR type, not output type (which may be fp16)
        # WMMA instructions on gfx942 always use fp32 accumulator for fp16 inputs
        dtype_map = {
            "fp16": ("fp16", "fp16", "fp32"),  # A=fp16, B=fp16, Acc=fp32
            "bf16": ("bf16", "bf16", "fp32"),  # A=bf16, B=bf16, Acc=fp32
            "fp8": ("fp8", "fp8", "fp32"),  # A=fp8, B=fp8, Acc=fp32
            "bf8": ("bf8", "bf8", "fp32"),  # A=bf8, B=bf8, Acc=fp32
            "int8": ("int8", "int8", "int32"),  # A=int8, B=int8, Acc=int32
        }
        dtype_a, dtype_b, dtype_c = dtype_map.get(
            self.datatype, ("fp16", "fp16", "fp32")
        )

        # Map GEMM variant to operator type for validation
        operator = None
        if pipeline is None:
            pipeline = "compv4"  # Default (representative compute pipeline)
        if scheduler is None:
            scheduler = "intrawave"  # Default

        if OperatorType is not None and variant is not None:
            variant_to_operator = {
                GemmVariant.STANDARD: OperatorType.GEMM,
                GemmVariant.PRESHUFFLE: OperatorType.GEMM_PRESHUFFLE,
                GemmVariant.MULTI_D: OperatorType.GEMM_MULTI_D,
                # Multi-ABD shares Multi-D's XDL/tile constraints (same
                # UniversalGemm pipeline + CShuffle epilogue), so validate it
                # against the GEMM_MULTI_D operator rules.
                GemmVariant.MULTI_ABD: OperatorType.GEMM_MULTI_D,
                GemmVariant.GROUPED: OperatorType.GEMM_GROUPED,
                GemmVariant.STREAM_K: OperatorType.GEMM_STREAMK,
            }
            operator = variant_to_operator.get(variant, OperatorType.GEMM)

            # Preshuffle requires specific pipeline and scheduler
            if variant == GemmVariant.PRESHUFFLE:
                pipeline = "preshufflev2"
                scheduler = "default"

        # Use preshuffle-specific validation (comprehensive CK-specific checks)
        if variant == GemmVariant.PRESHUFFLE:
            if not is_preshuffle_config_valid(
                tile_m=tile.tile_m,
                tile_n=tile.tile_n,
                tile_k=tile.tile_k,
                warp_m=tile.warp_m,
                warp_n=tile.warp_n,
                warp_k=tile.warp_k,
                warp_tile_m=tile.warp_tile_m,
                warp_tile_n=tile.warp_tile_n,
                warp_tile_k=tile.warp_tile_k,
                datatype=self.datatype,
            ):
                return False

        return self.arch_filter.is_kernel_valid(
            datatype_a=dtype_a,
            datatype_b=dtype_b,
            datatype_c=dtype_c,
            tile_m=tile.tile_m,
            tile_n=tile.tile_n,
            tile_k=tile.tile_k,
            warp_m=tile.warp_m,
            warp_n=tile.warp_n,
            warp_k=tile.warp_k,
            warp_tile_m=tile.warp_tile_m,
            warp_tile_n=tile.warp_tile_n,
            warp_tile_k=tile.warp_tile_k,
            pipeline=pipeline,
            scheduler=scheduler,
            layout=self.layout,
            operator=operator,
        )

    def _get_trait_configs(self) -> List[TraitConfig]:
        """Get valid trait configurations, filtered by architecture constraints"""
        tc = self.config["trait_config"]
        configs = []
        rejected_count = 0

        for params in itertools.product(
            tc["pipeline"],
            tc["epilogue"],
            tc["scheduler"],
            tc["pad_m"],
            tc["pad_n"],
            tc["pad_k"],
            tc["persistent"],
        ):
            trait = TraitConfig(*params)

            # Basic trait validation (unsupported combinations)
            if not trait.is_valid():
                rejected_count += 1
                continue

            configs.append(trait)

        if rejected_count > 0:
            log.debug(f"Rejected {rejected_count} trait configs")

        return configs

    def _generate_one(self, config: KernelConfig) -> Tuple[str, str]:
        """Generate one kernel and wrapper"""
        kernel_name = KernelNaming.generate(config, self.datatype, self.layout)

        # Generate CK Tile kernel
        kernel_code = self.ck_gen.generate(config)
        kernel_path = self.kernel_dir / f"{kernel_name}.hpp"
        kernel_path.write_text(kernel_code)

        # Generate dispatcher wrapper
        wrapper_code = self.disp_gen.generate(config, kernel_path, self.kernel_dir)
        wrapper_path = self.wrapper_dir / f"dispatcher_wrapper_{kernel_name}.hpp"
        wrapper_path.write_text(wrapper_code)

        # Generate .cpp compilation unit for per-kernel parallel builds
        cpp_path = self.kernel_dir / f"{kernel_name}.cpp"
        cpp_code = f'''// SPDX-License-Identifier: MIT
// Auto-generated compilation unit for: {kernel_name}
// Enables per-kernel parallel compilation with make -j

#include "{kernel_name}.hpp"

namespace ck_tile {{ namespace generated {{
    volatile bool _{kernel_name.replace("-", "_")}_loaded = true;
}} }}
'''
        cpp_path.write_text(cpp_code)

        return str(kernel_path), str(wrapper_path)

    def _generate_registration_header(self, wrapper_paths: List[str]):
        """Generate master registration header"""
        kernel_names = [
            Path(w).stem.replace("dispatcher_wrapper_", "") for w in wrapper_paths
        ]

        includes = "\n".join(
            [f'#include "dispatcher_wrapper_{n}.hpp"' for n in kernel_names]
        )
        registrations = "\n        ".join(
            [
                f"registry.register_kernel(generated::make_{n}(gfx_arch), priority);"
                for n in kernel_names
            ]
        )

        content = f"""// SPDX-License-Identifier: MIT
// Auto-generated master registration
#pragma once

#include "ck_tile/dispatcher.hpp"
{includes}

namespace ck_tile {{
namespace dispatcher {{

using ::ck_tile::dispatcher::Registry;
using Priority = ::ck_tile::dispatcher::Registry::Priority;

inline void register_all_tile_gemm_kernels(
    const std::string& gfx_arch = "gfx942",
    Priority priority = Priority::Normal)
{{
    auto& registry = Registry::instance();
    {registrations}
}}

inline std::size_t get_tile_gemm_kernel_count() {{ return {len(kernel_names)}; }}

}}}}
"""

        reg_path = self.wrapper_dir / "register_all_kernels.hpp"
        reg_path.write_text(content)
        logging.info(f"Generated registration header: {reg_path}")


# ============================================================================
# CLI
# ============================================================================


def _show_arch_info(gpu_target: str, datatype: str):
    """Display supported configurations for a GPU architecture"""
    if not HAS_ARCH_FILTER:
        print("Architecture filter module not available")
        return

    try:
        from arch_filter import (
            get_supported_archs,
            WARP_SUPPORTED_COMBINATIONS,
            WARP_TILE_SUPPORTED_COMBINATIONS,
            LDS_CAPACITY_LIMITS,
            TRAIT_UNSUPPORTED_COMBINATIONS,
        )

        print(f"\n=== Architecture Info for {gpu_target} ===\n")

        # Supported architectures
        print(f"Supported GPUs: {get_supported_archs()}")

        # Warp configurations
        warp_cfgs = WARP_SUPPORTED_COMBINATIONS.get(gpu_target, [])
        print("\nWarp configurations [warp_m, warp_n, warp_k]:")
        for cfg in warp_cfgs:
            print(f"  {cfg}")

        # Warp tile configurations for data type
        dtype_map = {
            "fp16": "fp16_fp16_fp16",
            "bf16": "bf16_bf16_bf16",
            "fp8": "fp8_fp8_fp16",
            "bf8": "bf8_bf8_fp16",
            "int8": "int8_int8_int32",
        }
        dtype_key = dtype_map.get(datatype, "fp16_fp16_fp16")

        gpu_combos = WARP_TILE_SUPPORTED_COMBINATIONS.get(gpu_target, {})
        warp_tiles = gpu_combos.get(dtype_key, [])
        print(
            f"\nWarp tile configurations for {dtype_key} [warp_tile_m, warp_tile_n, warp_tile_k]:"
        )
        for cfg in warp_tiles:
            print(f"  {cfg}")

        # All supported data types
        print(f"\nAll supported data types on {gpu_target}:")
        for dtype in gpu_combos.keys():
            print(f"  {dtype}")

        # LDS limits
        print("\nLDS capacity limits:")
        for pipeline, limit in LDS_CAPACITY_LIMITS.items():
            print(f"  {pipeline}: {limit // 1024}KB")

        # Unsupported trait combinations
        print("\nUnsupported trait combinations (pipeline, epilogue, scheduler):")
        for combo in TRAIT_UNSUPPORTED_COMBINATIONS:
            print(f"  {combo}")

        print()

    except Exception as e:
        print(f"Error showing arch info: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified GEMM Code Generator - Single Source of Truth"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory"
    )
    parser.add_argument(
        "--datatype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32", "fp8", "bf8", "int8", "pk_fp4"],
        help="Data type (fp16, bf16, fp32, fp8, bf8, int8, pk_fp4)",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="rcr",
        help="Layout (e.g., rcr for A=row, B=col, C=row; or rcrr for multi-d with D=row)",
    )
    parser.add_argument(
        "--gpu-target",
        type=str,
        default="gfx942",
        help="Target GPU (gfx90a, gfx942, gfx950, gfx1201)",
    )
    parser.add_argument("--config", type=Path, help="Configuration JSON file")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[
            "standard",
            "preshuffle",
            "multi_d",
            "stream_k",
            "batched",
            "multi_abd",
            "grouped",
        ],
        default=["standard"],
        help="Variants to generate",
    )
    parser.add_argument(
        "--preselected",
        type=str,
        help="Use preselected kernel set (e.g., fp16_rcr_essential)",
    )
    parser.add_argument(
        "--no-parallel", action="store_true", help="Disable parallel generation"
    )
    parser.add_argument(
        "--register", action="store_true", help="Generate dispatcher registration code"
    )
    parser.add_argument(
        "--no-arch-filter",
        action="store_true",
        help="Disable architecture-specific kernel filtering",
    )
    parser.add_argument(
        "--show-arch-info",
        action="store_true",
        help="Show supported configurations for target GPU and exit",
    )
    parser.add_argument(
        "--kernel-set",
        type=str,
        help="Kernel set name (creates subdirectory for organization)",
    )
    parser.add_argument(
        "--tile-config-json",
        type=str,
        help="JSON string specifying exact tile configuration (for minimal builds)",
    )

    args = parser.parse_args()

    # Handle inline tile config JSON for minimal/single-kernel builds
    if args.tile_config_json:
        try:
            cfg = json.loads(args.tile_config_json)

            # Build proper config structure
            full_config = {}

            # Extract tile config
            tile_keys = [
                "tile_m",
                "tile_n",
                "tile_k",
                "warp_m",
                "warp_n",
                "warp_k",
                "warp_tile_m",
                "warp_tile_n",
                "warp_tile_k",
                "block_size",
            ]
            tile_config = {k: cfg[k] for k in tile_keys if k in cfg}
            if tile_config:
                full_config["tile_config"] = tile_config

            # Extract trait config
            trait_keys = ["pipeline", "epilogue", "scheduler"]
            trait_config = {k: cfg[k] for k in trait_keys if k in cfg}
            # Add default pad/persistent values
            trait_config.setdefault("pad_m", [False])
            trait_config.setdefault("pad_n", [False])
            trait_config.setdefault("pad_k", [False])
            trait_config.setdefault("persistent", [False])
            if trait_config:
                full_config["trait_config"] = trait_config

            # Extract multi_d config (for multi_d variant)
            if "elementwise_ops" in cfg or "num_d_tensors" in cfg:
                multi_d_config = {}
                if "elementwise_ops" in cfg:
                    multi_d_config["elementwise_ops"] = cfg["elementwise_ops"]
                if "num_d_tensors" in cfg:
                    multi_d_config["num_d_tensors"] = cfg["num_d_tensors"]
                full_config["multi_d_config"] = multi_d_config

            # Use already structured config if provided
            if "tile_config" in cfg:
                full_config = cfg

            # Write to temp file and use as config
            import tempfile
            import os as _os

            _tmp_config = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            )
            try:
                json.dump(full_config, _tmp_config)
                _tmp_config.close()
                args.config = Path(_tmp_config.name)
            except Exception:
                _tmp_config.close()
                _os.unlink(_tmp_config.name)
                raise
        except json.JSONDecodeError as e:
            logging.error(f"Invalid tile-config-json: {e}")
            return 1
        except KeyError as e:
            logging.error(f"Missing required key in tile-config-json: {e}")
            return 1

    # Show architecture info if requested
    if args.show_arch_info:
        _show_arch_info(args.gpu_target, args.datatype)
        return 0

    variants = [GemmVariant(v) for v in args.variants] if not args.preselected else None

    codegen = UnifiedGemmCodegen(
        output_dir=args.output_dir,
        datatype=args.datatype,
        layout=args.layout,
        gpu_target=args.gpu_target,
        config_file=args.config,
        variants=variants,
        use_preselected=args.preselected,
        enable_arch_filter=not args.no_arch_filter,
        kernel_set_name=args.kernel_set,
    )

    results = codegen.generate_all(parallel=not args.no_parallel)

    logging.info("\nGeneration complete.")
    logging.info(f"  Kernels: {len(results['kernels'])}")
    logging.info(f"  Wrappers: {len(results['wrappers'])}")
    logging.info(f"  Failed: {len(results['failed'])}")

    if results["failed"]:
        logging.error(f"\nFailed kernels: {len(results['failed'])}")
        for err in results["failed"][:5]:
            logging.error(f"  {err}")

    # Generate dispatcher registration if requested
    if args.register:
        logging.info("\nGenerating dispatcher registration code...")
        try:
            from generate_dispatcher_registration import (
                scan_generated_headers,
                generate_registration_header,
                generate_registration_cpp,
            )

            kernels = scan_generated_headers(args.output_dir)
            reg_dir = args.output_dir / "registration"
            reg_dir.mkdir(exist_ok=True)

            generate_registration_header(
                kernels, reg_dir / "dispatcher_registration.hpp"
            )
            generate_registration_cpp(kernels, reg_dir / "dispatcher_registration.cpp")

            logging.info(f"Generated registration code for {len(kernels)} kernels")
        except Exception as e:
            logging.error(f"Failed to generate registration code: {e}")
            return 1

    # Clean up temp config file if we created one
    if args.tile_config_json and args.config and args.config.exists():
        try:
            import os as _os

            _os.unlink(args.config)
        except OSError:
            pass

    return 0 if not results["failed"] else 1


if __name__ == "__main__":
    exit(main())
