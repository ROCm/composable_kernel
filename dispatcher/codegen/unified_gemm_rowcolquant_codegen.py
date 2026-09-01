#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Gemm RowColQuant Code Generator

Generates one .hpp per kernel config for the dispatcher's ctypes path.
Each header defines a SelectedKernel struct with a static launch() method
taking QuantGemmHostArgs -- compiled per-kernel via force-include:

    hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE gemm_rowcolquant_ctypes_lib.cpp

RowColQuant = per-row scale of A ([M, 1], broadcast over N) + per-column scale of
B ([1, N], broadcast over M).  Both scale tensors are AccDataType (float).  Unlike
BQuantGrouped there is no quant-group size: the scales are global row/col vectors,
so the pipeline is the regular GemmPipelineAgBgCrCompV3 fed a
GemmRowColTensorQuantPipelineProblem (see run_gemm_quant_example.inc).

Scope: fp8 and bf8 dtype variants, rcr layout (RowMajor A / ColMajor B /
RowMajor C), compv3 pipeline, intrawave scheduler -- exactly the set that
gemm_quant_rowcol.cpp registers in Old-TE.

Naming convention (byte-exact with RowColQuantKernelConfig.name in
gemm_rowcolquant_utils.py):
    gemm_rowcolquant_{dtype}_{layout}_{pipeline}_{epilogue}_{scheduler}_
    {TileM}x{TileN}x{TileK}_{WarpM}x{WarpN}x{WarpK}_{WtM}x{WtN}x{WtK}

Reference:
    example/ck_tile/38_block_scale_gemm/gemm_quant_rowcol.cpp
    example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc
    example/ck_tile/38_block_scale_gemm/gemm_utils.hpp  (GemmConfigRowColQuant)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List

from codegen_common import (
    QUANT_LAYOUT_TO_CK,
    QUANT_SCHEDULER_TO_CK,
    TileConfig,
    emit_generated_header_preamble,
    emit_quant_epilogue_block,
    emit_quant_gemm_traits,
    emit_quant_launch_prologue,
    emit_quant_launch_tail,
    emit_quant_tile_dims,
    emit_quant_tile_shape,
    emit_single_kernel_include_footer,
    fp8_warp_tile_k_for_arch,
    iter_quant_axes,
    make_gemm_rowcolquant_kernel_name,
    quant_decode_default_config,
    rcr_only_layout_guard,
    run_codegen_cli,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Dtype variant definitions
# Each entry maps to GemmQuantTypeConfig<A, B, C, Acc> in gemm_quant_rowcol.cpp.
# RowColQuant scales (AQ row scale, BQ col scale) are always AccDataType=float.
# =============================================================================

ROWCOLQUANT_VARIANTS: Dict[str, Dict[str, str]] = {
    "fp8": {
        "ck_a": "ck_tile::fp8_t",
        "ck_b": "ck_tile::fp8_t",
        "ck_c": "ck_tile::half_t",
        "ck_acc": "float",
    },
    "bf8": {
        "ck_a": "ck_tile::bf8_t",
        "ck_b": "ck_tile::bf8_t",
        "ck_c": "ck_tile::half_t",
        "ck_acc": "float",
    },
}

# Layout strings supported: only rcr (RowMajor A, ColMajor B, RowMajor C).
# gemm_quant_rowcol.cpp only registers the a_layout=="R" && b_layout=="C" path,
# and gemm_calc_quant static_asserts CLayout == RowMajor.
ROWCOLQUANT_LAYOUT_TO_CK = QUANT_LAYOUT_TO_CK

# For RowColQuant the pipeline is the plain compute pipeline (not a quant pipeline);
# the quantization is folded in through GemmRowColTensorQuantPipelineProblem.
ROWCOLQUANT_PIPELINE_MAP = {
    "compv3": "ck_tile::GemmPipelineAgBgCrCompV3",
}

# Base pipeline selection: RowColQuant is not PreshuffleB / AQuant / ABQuant /
# eight_waves / IS_FP8BLOCKSCALE, so gemm_calc_quant falls into the final else
# branch -> BaseWeightPreshufflePipelineAGmemBGmemCRegV2.
ROWCOLQUANT_BASE_PIPELINE_MAP = {
    "compv3": "ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2",
}

ROWCOLQUANT_SCHEDULER_TO_CK = QUANT_SCHEDULER_TO_CK


# =============================================================================
# Configuration dataclasses
# =============================================================================


# Was a verbatim redeclaration of codegen_common.TileConfig, fields and
# is_valid() alike. Aliased rather than renamed so call sites read unchanged.
RowColQuantTileConfig = TileConfig


@dataclass
class RowColQuantKernelSpec:
    """Complete specification for one RowColQuant kernel."""

    variant_key: str          # "fp8" or "bf8"
    layout: str               # "rcr"
    pipeline: str             # "compv3"
    epilogue: str             # "cshuffle"
    scheduler: str            # "intrawave"
    tile: RowColQuantTileConfig
    pad_m: bool = False
    pad_n: bool = False
    pad_k: bool = False
    block_size: int = 256
    k_block_per_cu: int = 1
    double_smem_buffer: bool = False

    @property
    def name(self) -> str:
        t = self.tile
        return make_gemm_rowcolquant_kernel_name(
            variant_key=self.variant_key,
            layout=self.layout,
            pipeline=self.pipeline,
            epilogue=self.epilogue,
            scheduler=self.scheduler,
            tile_m=t.tile_m, tile_n=t.tile_n, tile_k=t.tile_k,
            warp_m=t.warp_m, warp_n=t.warp_n, warp_k=t.warp_k,
            warp_tile_m=t.warp_tile_m, warp_tile_n=t.warp_tile_n, warp_tile_k=t.warp_tile_k,
        )


# =============================================================================
# Header generator
# =============================================================================


class RowColQuantKernelHeaderGenerator:
    """Generates a .hpp kernel specialization header for one RowColQuantKernelSpec."""

    def generate(self, spec: RowColQuantKernelSpec) -> str:
        variant = ROWCOLQUANT_VARIANTS[spec.variant_key]
        t = spec.tile
        ns = "ns_" + spec.name
        struct = "Kernel_" + spec.name

        ck_a = variant["ck_a"]
        ck_b = variant["ck_b"]
        ck_c = variant["ck_c"]
        ck_acc = variant["ck_acc"]

        layout_a_ck = ROWCOLQUANT_LAYOUT_TO_CK[spec.layout[0]]
        layout_b_ck = ROWCOLQUANT_LAYOUT_TO_CK[spec.layout[1]]
        layout_c_ck = ROWCOLQUANT_LAYOUT_TO_CK[spec.layout[2]]
        # RowColQuant: AQ is the row scale (RowMajor [M,1]), BQ is the col scale
        # (ColMajor [1,N]) -- matches Row{}, Col{} passed for aq/bq in
        # run_gemm_example_with_layouts for the R/C case.
        layout_aq_ck = ROWCOLQUANT_LAYOUT_TO_CK["r"]
        layout_bq_ck = ROWCOLQUANT_LAYOUT_TO_CK["c"]

        pipeline_ck = ROWCOLQUANT_PIPELINE_MAP[spec.pipeline]
        base_pipeline_ck = ROWCOLQUANT_BASE_PIPELINE_MAP[spec.pipeline]
        scheduler_ck = ROWCOLQUANT_SCHEDULER_TO_CK[spec.scheduler]

        pad_m = str(spec.pad_m).lower()
        pad_n = str(spec.pad_n).lower()
        pad_k = str(spec.pad_k).lower()
        double_smem_buffer = str(spec.double_smem_buffer).lower()

        # GemmConfigRowColQuant derives from GemmConfigBase which has
        # TiledMMAPermuteN=false, so RowColQuant always uses the CShuffle epilogue.
        epilogue_block = emit_quant_epilogue_block("cshuffle", ns)
        tile_dims = emit_quant_tile_dims(
            t, block_size=spec.block_size, k_block_per_cu=spec.k_block_per_cu
        )
        tile_shape = emit_quant_tile_shape()
        gemm_traits = emit_quant_gemm_traits("RowColQuant", ns)
        launch_prologue = emit_quant_launch_prologue(
            splitk_k="WarpTileK",
            preamble=(
                "        // hot-loop / tail dispatch -- mirrors run_gemm_quant_example.inc.\n"
                "        // RowColQuant always uses k_batch==1 semantics for K_split (no split-K\n"
                "        // path is registered in Old-TE for rowcol).\n"
            ),
        )
        launch_tail = emit_quant_launch_tail(quant_type="RowColQuant")

        return emit_generated_header_preamble(
            "Gemm RowColQuant", "unified_gemm_rowcolquant_codegen.py"
        ) + f"""\
namespace {ns} {{

constexpr const char* KERNEL_NAME = "{spec.name}";

using ADataType   = {ck_a};
using BDataType   = {ck_b};
using CDataType   = {ck_c};
// RowColQuant scales (row scale of A, col scale of B) are the accumulator type.
using QDataType   = {ck_acc};
using AccDataType = {ck_acc};

using ALayout  = {layout_a_ck};
using BLayout  = {layout_b_ck};
using CLayout  = {layout_c_ck};
using AQLayout = {layout_aq_ck};
using BQLayout = {layout_bq_ck};

struct {struct} {{
    using ADataType   = {ns}::ADataType;
    using BDataType   = {ns}::BDataType;
    using CDataType   = {ns}::CDataType;
    using QDataType   = {ns}::QDataType;
    using AccDataType = {ns}::AccDataType;

{tile_dims}

    static constexpr bool kPadM            = {pad_m};
    static constexpr bool kPadN            = {pad_n};
    static constexpr bool kPadK            = {pad_k};
    static constexpr bool APreshuffleQuant = false;
    static constexpr bool BPreshuffleQuant = false;
    static constexpr bool PreshuffleB      = false;
    static constexpr bool TransposeC       = false;
    static constexpr bool DoubleSmemBuffer = {double_smem_buffer};

{tile_shape}

{gemm_traits}

    using GemmPipelineProblemBase = ck_tile::GemmPipelineProblemBase<
        ADataType, BDataType, AccDataType, TileShape, GemmTraits>;

    using BaseGemmPipeline = {base_pipeline_ck}<GemmPipelineProblemBase>;

{launch_prologue}
            // NOTE: the 3rd template arg is the pipeline's *compute/C* type which
            // for RowColQuant is the accumulator (float), NOT the final CDataType.
            // run_gemm_quant_example.inc passes AccDataType here; the narrowing to
            // the real CDataType (half) happens in the CShuffle epilogue below.
            using PipelineProblem = ck_tile::GemmRowColTensorQuantPipelineProblem<
                ADataType,
                BDataType,
                AccDataType,
                AccDataType,
                TileShape,
                GemmTraits,
                TransposeC,
                ADataType,        // ComputeDataType
                {scheduler_ck},
                has_hot_loop_.value,
                tail_number_.value>;

            using GemmPipeline = {pipeline_ck}<PipelineProblem>;

{epilogue_block}

{launch_tail}
}};

using SelectedKernel = {struct};

}} // namespace {ns}

""" + emit_single_kernel_include_footer(
            ns=ns,
            struct=struct,
            ck_a=ck_a,
            ck_b=ck_b,
            ck_c=ck_c,
            ck_q=ck_acc,
            ck_acc=ck_acc,
        )


# =============================================================================
# Config sweep
# =============================================================================


def _default_config() -> dict:
    """Default sweep config matching GemmConfigRowColQuant tile defaults.

    GemmConfigRowColQuant<fp8_t> is the shared decode tile (16x64x256).
    WarpTileK is arch-derived (128 on gfx950, 32 on gfx942; 128 silently
    outputs all-zeros on gfx942). The Python driver
    (gemm_rowcolquant_utils.default_*_config -> _warp_tile_k_for) sets this
    per-arch; this standalone fallback uses the gfx950 value.

    pad_k=False overrides the shared decode default (True): RowColQuant runs
    the unpadded-K pipeline for Old-TE perf parity, unlike tensor_quant and
    bquant which keep the padded default.
    """
    return quant_decode_default_config(
        warp_tile_k=fp8_warp_tile_k_for_arch("gfx950"),
        pad_k=False,
    )


def _build_specs(config: dict) -> List[RowColQuantKernelSpec]:
    specs = []
    pipeline  = config.get("pipeline", "compv3")
    epilogue  = config.get("epilogue", "cshuffle")
    scheduler = config.get("scheduler", "intrawave")
    pad_m     = config.get("pad_m", False)
    pad_n     = config.get("pad_n", False)
    pad_k     = config.get("pad_k", False)
    block_size         = config.get("block_size", 256)
    k_block_per_cu     = config.get("k_block_per_cu", 1)
    double_smem_buffer = config.get("double_smem_buffer", False)

    for variant_key, layout, tile, _ in iter_quant_axes(
        config,
        variants=ROWCOLQUANT_VARIANTS,
        logger=log,
        pipeline=pipeline,
        pipeline_map=ROWCOLQUANT_PIPELINE_MAP,
        layout_guard=rcr_only_layout_guard,
    ):
        specs.append(RowColQuantKernelSpec(
            variant_key=variant_key,
            layout=layout,
            pipeline=pipeline,
            epilogue=epilogue,
            scheduler=scheduler,
            tile=tile,
            pad_m=pad_m,
            pad_n=pad_n,
            pad_k=pad_k,
            block_size=block_size,
            k_block_per_cu=k_block_per_cu,
            double_smem_buffer=double_smem_buffer,
        ))

    return specs

# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    return run_codegen_cli(
        description="Gemm RowColQuant kernel header generator",
        op_label="RowColQuant",
        make_generator=RowColQuantKernelHeaderGenerator,
        build_specs=_build_specs,
        default_config=_default_config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
