#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Gemm TensorQuant (single per-tensor scale) Code Generator

Generates one .hpp per kernel config for the dispatcher's ctypes path.
Each header defines a SelectedKernel struct with a static launch() method
taking QuantGemmHostArgs -- compiled per-kernel via force-include:

    hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE gemm_tensor_quant_ctypes_lib.cpp

Scope (behavioral parity with Old-TE gemm_quant_tensor.cpp):
    QuantType::TensorQuant -- ONE scalar scale for A and ONE scalar scale for B.
    dtypes:   fp8, bf8   (A==B; QDataType=float; CDataType=half; Acc=float)
    layout:   rcr only   (RowMajor A, ColumnMajor B, RowMajor C)
    pipeline: compv3      (GemmPipelineAgBgCrCompV3 -- the "regular" pipeline;
                           TensorQuant reuses the non-quant compute pipeline and
                           applies the scalar scales in the epilogue)
    scheduler: intrawave

TensorQuant vs BQuantGrouped differences (mirrors run_gemm_quant_example.inc):
    - PipelineProblem   = GemmRowColTensorQuantPipelineProblem (NOT GemmBQuantPipelineProblem)
    - GemmPipeline      = GemmPipelineAgBgCrCompV3            (NOT BQuant pipeline)
    - Base pipeline     = BaseWeightPreshufflePipelineAGmemBGmemCRegV2 (else-branch, PreshuffleB=false)
    - QuantGroupSize    = QuantGroupShape<1,1,1> (placeholder; tensor path ignores it)
    - Both aq_ptr AND bq_ptr are single scalar floats (read as *aq_ptr / *bq_ptr in kernel)
    - Epilogue is invoked with (aq_scale, bq_scale) extra args
    - TiledPermuteN = GemmConfig::TiledMMAPermuteN (false for GemmConfigQuantDecode -> cshuffle)

Naming convention (byte-exact with TensorQuantKernelConfig.name in gemm_tensor_quant_utils.py):
    gemm_tensor_quant_{dtype}_{layout}_{pipeline}_{epilogue}_{scheduler}_
    {TileM}x{TileN}x{TileK}_{WarpM}x{WarpN}x{WarpK}_{WtM}x{WtN}x{WtK}

Reference:
    example/ck_tile/38_block_scale_gemm/gemm_quant_tensor.cpp
    example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc
    example/ck_tile/38_block_scale_gemm/gemm_utils.hpp  (GemmConfigQuantDecode)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List

from codegen_common import (
    QUANT_LAYOUT_TO_CK,
    QUANT_SCHEDULER_TO_CK,
    TileConfig,
    # Re-exported for gemm_tensor_quant_utils.py / tests, which import it from
    # this module; not called here.
    emit_generated_header_preamble,
    emit_quant_epilogue_block,
    emit_quant_gemm_traits,
    emit_quant_kernel_attr_launch,
    emit_quant_launch_prologue,
    emit_quant_launch_tail,
    emit_quant_tile_dims,
    emit_quant_tile_shape,
    emit_single_kernel_include_footer,
    fp8_warp_tile_k_for_arch,
    iter_quant_axes,
    make_tensor_quant_kernel_name,
    quant_decode_default_config,
    rcr_only_layout_guard,
    run_codegen_cli,
    tensor_quant_effective_epilogue,  # noqa: F401
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Dtype variant definitions
# Each entry: A==B element dtype; C=half; Q=float (scalar scale); Acc=float.
# Matches gemm_quant_tensor.cpp: GemmQuantTypeConfig<fp8_t, fp8_t, half_t, float>
#                                GemmQuantTypeConfig<bf8_t, bf8_t, half_t, float>
# =============================================================================

TENSOR_QUANT_VARIANTS: Dict[str, Dict[str, str]] = {
    "fp8": {
        "dtype_a": "fp8",
        "dtype_b": "fp8",
        "dtype_c": "half",
        "dtype_q": "float",
        "ck_a": "ck_tile::fp8_t",
        "ck_b": "ck_tile::fp8_t",
        "ck_c": "ck_tile::half_t",
        "ck_q": "float",
        "ck_acc": "float",
    },
    "bf8": {
        "dtype_a": "bf8",
        "dtype_b": "bf8",
        "dtype_c": "half",
        "dtype_q": "float",
        "ck_a": "ck_tile::bf8_t",
        "ck_b": "ck_tile::bf8_t",
        "ck_c": "ck_tile::half_t",
        "ck_q": "float",
        "ck_acc": "float",
    },
}

# Layout strings supported: only rcr (RowMajor A, ColumnMajor B, RowMajor C).
# run_gemm_quant_example.inc only dispatches a_layout=="R" && b_layout=="C" for
# these fp8/bf8 types, and CLayout is static_asserted to RowMajor.
TENSOR_QUANT_LAYOUT_TO_CK = QUANT_LAYOUT_TO_CK

# TensorQuant uses the regular (non-quant) compute pipeline; only compv3 is
# emitted by GemmConfigQuantDecode (Scheduler=Intrawave).
TENSOR_QUANT_PIPELINE_MAP = {
    "compv3": "ck_tile::GemmPipelineAgBgCrCompV3",
}

# For TensorQuant (PreshuffleB=false, not AQuant/ABQuant, IS_FP8BLOCKSCALE=false)
# run_gemm_quant_example.inc's base_gemm_pipeline lambda falls through to the
# final else branch -> BaseWeightPreshufflePipelineAGmemBGmemCRegV2.
TENSOR_QUANT_BASE_PIPELINE_MAP = {
    "compv3": "ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2",
}

TENSOR_QUANT_SCHEDULER_TO_CK = QUANT_SCHEDULER_TO_CK


# =============================================================================
# Kernel name construction (shared by codegen + utils so they stay byte-exact)
# =============================================================================
#
# Both builders now live in codegen_common alongside the other quant families;
# they are imported at the top of this module and re-exported here so that
# gemm_tensor_quant_utils.py and tests/test_tensor_quant_bridge.py, which import
# them from this module, keep working unchanged.


# =============================================================================
# Configuration dataclasses
# =============================================================================


# Was a verbatim redeclaration of codegen_common.TileConfig, fields and
# is_valid() alike. Aliased rather than renamed so call sites read unchanged.
TensorQuantTileConfig = TileConfig


@dataclass
class TensorQuantKernelSpec:
    """Complete specification for one TensorQuant kernel."""

    variant_key: str          # "fp8" or "bf8"
    layout: str               # "rcr"
    pipeline: str             # "compv3"
    epilogue: str             # "cshuffle"
    scheduler: str            # "intrawave"
    tile: TensorQuantTileConfig
    double_smem_buffer: bool = False
    pad_m: bool = False
    pad_n: bool = False
    pad_k: bool = True
    block_size: int = 256
    k_block_per_cu: int = 1

    @property
    def name(self) -> str:
        t = self.tile
        return make_tensor_quant_kernel_name(
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


class TensorQuantKernelHeaderGenerator:
    """Generates a .hpp kernel specialization header for one TensorQuantKernelSpec."""

    def generate(self, spec: TensorQuantKernelSpec) -> str:
        variant = TENSOR_QUANT_VARIANTS[spec.variant_key]
        t = spec.tile
        ns = "ns_" + spec.name
        struct = "Kernel_" + spec.name

        ck_a = variant["ck_a"]
        ck_b = variant["ck_b"]
        ck_c = variant["ck_c"]
        ck_q = variant["ck_q"]
        ck_acc = variant["ck_acc"]

        layout_a_ck = TENSOR_QUANT_LAYOUT_TO_CK[spec.layout[0]]
        layout_b_ck = TENSOR_QUANT_LAYOUT_TO_CK[spec.layout[1]]
        layout_c_ck = TENSOR_QUANT_LAYOUT_TO_CK[spec.layout[2]]
        # AQ/BQ layouts are placeholders for TensorQuant (single scalar scale);
        # mirror the example which passes Col{} for both AQ and BQ in the rcr path.
        layout_aq_ck = TENSOR_QUANT_LAYOUT_TO_CK["c"]
        layout_bq_ck = TENSOR_QUANT_LAYOUT_TO_CK["c"]

        pipeline_ck = TENSOR_QUANT_PIPELINE_MAP[spec.pipeline]
        base_pipeline_ck = TENSOR_QUANT_BASE_PIPELINE_MAP[spec.pipeline]
        scheduler_ck = TENSOR_QUANT_SCHEDULER_TO_CK[spec.scheduler]

        pad_m = str(spec.pad_m).lower()
        pad_n = str(spec.pad_n).lower()
        pad_k = str(spec.pad_k).lower()
        double_smem_buffer = str(spec.double_smem_buffer).lower()

        # TensorQuant (GemmConfigQuantDecode) always uses the CShuffle epilogue
        # (TiledMMAPermuteN=false). The CShuffleEpilogue is invoked with the two
        # scalar scales (aq_scale, bq_scale) inside the kernel for TensorQuant.
        epilogue_block = emit_quant_epilogue_block("cshuffle", ns)

        tile_dims = emit_quant_tile_dims(
            t, block_size=spec.block_size, k_block_per_cu=spec.k_block_per_cu
        )
        tile_shape = emit_quant_tile_shape()
        gemm_traits = emit_quant_gemm_traits("TensorQuant", ns)
        launch_prologue = emit_quant_launch_prologue(
            splitk_k="K1",
            preamble=(
                "        // hot-loop / tail dispatch -- mirrors run_gemm_quant_example.inc.\n"
                "        // K1 = WarpTileK; K_split uses K_Tile for k_batch==1.\n"
                "        constexpr ck_tile::index_t K1 = WarpTileK;\n"
            ),
        )
        launch_tail = emit_quant_launch_tail(
            quant_type="TensorQuant",
            launch_call=emit_quant_kernel_attr_launch("eight_waves"),
            extra="""
            // Launch through the SAME kernel_attr<...> / kentry overload Old-TE
            // uses (run_gemm_quant_example.inc), NOT the plain make_kernel path.
            // Old-TE computes:
            //   eight_waves = IS_FP8BLOCKSCALE && (M_Warp*N_Warp*K_Warp == 8) &&
            //                 K_Warp_Tile == 128;   // under CK_GFX950_SUPPORT
            // For TensorQuant IS_FP8BLOCKSCALE is false, so eight_waves is always
            // false here -- but we mirror the full expression so the emitted
            // kentry<Attr, MinBlockPerCu, ...> specialization is byte-for-byte the
            // same instantiation Old-TE compiles (this is what makes the resulting
            // kernel identical: VGPR 132 to match Old-TE, vs 136 for the plain
            // make_kernel<kBlockPerCu> / kentry<MinBlockPerCu, ...> overload).
            constexpr bool eight_waves =
#ifdef CK_GFX950_SUPPORT
                false /* IS_FP8BLOCKSCALE=false for TensorQuant */ &&
                (WarpM * WarpN * WarpK == 8) && (WarpTileK == 128);
#else
                false;
#endif
""",
        )

        return emit_generated_header_preamble(
            "Gemm TensorQuant", "unified_gemm_tensor_quant_codegen.py"
        ) + f"""\
namespace {ns} {{

constexpr const char* KERNEL_NAME = "{spec.name}";

using ADataType   = {ck_a};
using BDataType   = {ck_b};
using CDataType   = {ck_c};
using QDataType   = {ck_q};
using AccDataType = {ck_acc};

using ALayout  = {layout_a_ck};
using BLayout  = {layout_b_ck};
using CLayout  = {layout_c_ck};
using AQLayout = {layout_aq_ck};
using BQLayout = {layout_bq_ck};

// Placeholder QuantGroupSize -- TensorQuant applies one scalar scale for A and
// one scalar scale for B, so the group size is unused (matches the example's
// QuantGroupShape<1,1,1> place holder).
using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 1>>;

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

    // ComputeDataType for the base pipeline problem: the example uses
    // AComputeDataType=void for the non-fp8-blockscale TensorQuant path.
    using GemmPipelineProblemBase = ck_tile::GemmPipelineProblemBase<
        ADataType, BDataType, AccDataType, TileShape, GemmTraits>;

    using BaseGemmPipeline = {base_pipeline_ck}<GemmPipelineProblemBase>;

{launch_prologue}
            // TensorQuant reuses the regular GEMM compute pipeline via
            // GemmRowColTensorQuantPipelineProblem; the scalar scales are applied
            // in the epilogue (see gemm_quant_kernel.hpp TensorQuant branch).
            // Mirrors run_gemm_quant_example.inc: the compute pipeline's C slot is
            // AccDataType (float), NOT the final CDataType (half). The half
            // down-conversion happens in the CShuffle epilogue. Passing CDataType
            // here selects a non-existent fp8x(16x16x128)->half warp-gemm.
            using PipelineProblem = ck_tile::GemmRowColTensorQuantPipelineProblem<
                ADataType,
                BDataType,
                AccDataType,      // CDataType slot = AccDataType (float)
                AccDataType,
                TileShape,
                GemmTraits,
                TransposeC,
                void,             // AComputeDataType (void for non-fp8-blockscale TensorQuant; mirrors run_gemm_quant_example.inc)
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
            ck_q=ck_q,
            ck_acc=ck_acc,
            extra_lines=f"using QuantGroupSize = {ns}::QuantGroupSize;",
        )


# =============================================================================
# Config sweep
# =============================================================================


_DEFAULT_GFX_ARCH = "gfx950"


def _default_config(gfx_arch: str = _DEFAULT_GFX_ARCH) -> dict:
    """Default sweep config matching GemmConfigQuantDecode tile defaults.

    GemmConfigQuantDecode<fp8_t/bf8_t>: M=16, N=64, K=256/sizeof(8bit)=256,
    warp 1x4x1, warp_tile 16x16x K_warp. WarpTileK is arch-derived
    (get_k_warp_tile<fp8_t/bf8_t, M_Warp_Tile=16>() = 128 on gfx950, 32 on gfx942).
    """
    return quant_decode_default_config(
        warp_tile_k=fp8_warp_tile_k_for_arch(gfx_arch),
    )


def _build_specs(config: dict) -> List[TensorQuantKernelSpec]:
    specs = []
    pipeline  = config.get("pipeline", "compv3")
    epilogue  = config.get("epilogue", "cshuffle")
    # TensorQuant codegen only ever emits the CShuffle epilogue (see
    # tensor_quant_effective_epilogue), and the kernel name ignores this field.
    # Coerce any other request so the config can't imply a kernel we don't build.
    if epilogue != "cshuffle":
        log.warning(
            "TensorQuant codegen only emits the CShuffle epilogue; "
            "overriding requested epilogue %r with 'cshuffle'.", epilogue,
        )
        epilogue = "cshuffle"
    scheduler = config.get("scheduler", "intrawave")
    pad_m     = config.get("pad_m", False)
    pad_n     = config.get("pad_n", False)
    pad_k     = config.get("pad_k", True)
    block_size         = config.get("block_size", 256)
    k_block_per_cu     = config.get("k_block_per_cu", 1)
    double_smem_buffer = config.get("double_smem_buffer", False)

    for variant_key, layout, tile, _ in iter_quant_axes(
        config,
        variants=TENSOR_QUANT_VARIANTS,
        logger=log,
        pipeline=pipeline,
        pipeline_map=TENSOR_QUANT_PIPELINE_MAP,
        layout_guard=rcr_only_layout_guard,
    ):
        specs.append(TensorQuantKernelSpec(
            variant_key=variant_key,
            layout=layout,
            pipeline=pipeline,
            epilogue=epilogue,
            scheduler=scheduler,
            tile=tile,
            double_smem_buffer=double_smem_buffer,
            pad_m=pad_m,
            pad_n=pad_n,
            pad_k=pad_k,
            block_size=block_size,
            k_block_per_cu=k_block_per_cu,
        ))

    return specs

# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    return run_codegen_cli(
        description="Gemm TensorQuant kernel header generator",
        op_label="TensorQuant",
        make_generator=TensorQuantKernelHeaderGenerator,
        build_specs=_build_specs,
        default_config=_default_config,
        arch_aware=True,
        default_gfx_arch=_DEFAULT_GFX_ARCH,
    )


if __name__ == "__main__":
    raise SystemExit(main())
