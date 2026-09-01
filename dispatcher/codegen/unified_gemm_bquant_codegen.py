#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Gemm BQuant Code Generator (non-grouped, block-scale GEMM)

Bridges the plain BQuant (B-only quantized) block-scale GEMM operator from
example/ck_tile/38_block_scale_gemm to the dispatcher's ctypes path. This is the
NON-grouped gemm_bquant: a single GEMM problem with grouped-quant weight scales,
distinct from the multi-problem grouped_gemm_bquant under 17_grouped_gemm.

Generates one .hpp per kernel config for the dispatcher's ctypes path.
Each header defines a SelectedKernel struct with a static launch() method
taking QuantGemmHostArgs -- compiled per-kernel via force-include:

    hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE gemm_bquant_ctypes_lib.cpp

Force-include defines (from generated kernel header):
    SelectedKernel, KERNEL_NAME
    ADataType, BDataType, CDataType, QDataType, AccDataType, QuantGroupSize

Scope (100% parity with the Old-TE plain bquant examples):
    fp8, bf8, fp8i4, bf8i4                        (decode tile, compv3)
    mx_bf16bf16, mx_bf16bf8, mx_bf16fp4           (MX e8m0 block scale, microscale)
  x {non-preshuffle, preshuffleb, preshufflebq, preshuffleb+preshufflebq}
    where each variant supports the preshuffle phases Old-TE ships for it.

Naming convention (byte-exact with BQuantKernelConfig.name in gemm_bquant_utils.py):
    gemm_bquant_{dtype_a}_{layout}_{pipeline}_{epilogue}_{scheduler}_
    {TileM}x{TileN}x{TileK}_{WarpM}x{WarpN}x{WarpK}_{WtM}x{WtN}x{WtK}_
    qg{gM}x{gN}x{gK}[_preshuffleb][_preshufflebq]

Reference:
    example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_{fp8,bf8,fp8i4,bf8i4}.cpp
    example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_mx_bf16{bf16,bf8,fp4}.cpp
    example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_preshuffle*_*.cpp
    example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc
    example/ck_tile/38_block_scale_gemm/gemm_utils.hpp  (GemmConfigQuantDecode etc.)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List

from codegen_common import (
    QUANT_LAYOUT_TO_CK,
    QUANT_SCHEDULER_TO_CK,
    TileConfig,
    bquant_effective_epilogue,
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
    make_bquant_kernel_name,
    quant_decode_default_config,
    run_codegen_cli,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Operator family prefix -- the only structural difference vs the (mis-named)
# grouped_gemm_bquant bridge, which references these exact same examples.
NAME_PREFIX = "gemm_bquant"


# =============================================================================
# Dtype variant definitions
# Each entry: (ADataType, BDataType, CDataType, QDataType, AccDataType)
# Matches example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_*.cpp
# =============================================================================

BQUANT_VARIANTS: Dict[str, Dict[str, str]] = {
    # gemm_bquant_quantgrouped_fp8.cpp:
    #   GemmQuantTypeConfig<fp8_t, fp8_t, half_t, float>
    "fp8": {
        "ck_a": "ck_tile::fp8_t",
        "ck_b": "ck_tile::fp8_t",
        "ck_c": "ck_tile::half_t",
        "ck_q": "float",
        "ck_acc": "float",
    },
    # gemm_bquant_quantgrouped_bf8.cpp:
    #   GemmQuantTypeConfig<bf8_t, bf8_t, half_t, float>
    "bf8": {
        "ck_a": "ck_tile::bf8_t",
        "ck_b": "ck_tile::bf8_t",
        "ck_c": "ck_tile::half_t",
        "ck_q": "float",
        "ck_acc": "float",
    },
    # gemm_bquant_quantgrouped_fp8i4.cpp:
    #   GemmQuantTypeConfig<fp8_t, pk_int4_t, half_t, fp8_t>
    "fp8i4": {
        "ck_a": "ck_tile::fp8_t",
        "ck_b": "ck_tile::pk_int4_t",
        "ck_c": "ck_tile::half_t",
        "ck_q": "ck_tile::fp8_t",
        "ck_acc": "float",
    },
    # gemm_bquant_quantgrouped_bf8i4.cpp:
    #   GemmQuantTypeConfig<bf8_t, pk_int4_t, half_t, bf8_t>
    "bf8i4": {
        "ck_a": "ck_tile::bf8_t",
        "ck_b": "ck_tile::pk_int4_t",
        "ck_c": "ck_tile::half_t",
        "ck_q": "ck_tile::bf8_t",
        "ck_acc": "float",
    },
    # MX microscale variants -- Q-type is e8m0 (block scale), pipeline = microscale.
    # gemm_bquant_quantgrouped_mx_bf16bf16.cpp:
    #   GemmQuantTypeConfig<bf16_t, bf16_t, bf16_t, e8m0_t>
    "mx_bf16bf16": {
        "ck_a": "ck_tile::bf16_t",
        "ck_b": "ck_tile::bf16_t",
        "ck_c": "ck_tile::bf16_t",
        "ck_q": "ck_tile::e8m0_t",
        "ck_acc": "float",
    },
    # gemm_bquant_quantgrouped_mx_bf16bf8.cpp:
    #   GemmQuantTypeConfig<bf16_t, bf8_t, bf16_t, e8m0_t>
    "mx_bf16bf8": {
        "ck_a": "ck_tile::bf16_t",
        "ck_b": "ck_tile::bf8_t",
        "ck_c": "ck_tile::bf16_t",
        "ck_q": "ck_tile::e8m0_t",
        "ck_acc": "float",
    },
    # gemm_bquant_quantgrouped_mx_bf16fp4.cpp:
    #   GemmQuantTypeConfig<bf16_t, pk_fp4_t, bf16_t, e8m0_t>
    "mx_bf16fp4": {
        "ck_a": "ck_tile::bf16_t",
        "ck_b": "ck_tile::pk_fp4_t",
        "ck_c": "ck_tile::bf16_t",
        "ck_q": "ck_tile::e8m0_t",
        "ck_acc": "float",
    },
}

# Variants whose QDataType is e8m0 (MX block scale) -- these require gfx950 and
# use the microscale pipeline. Non-MX variants run on gfx942 + gfx950.
MX_VARIANTS = {"mx_bf16bf16", "mx_bf16bf8", "mx_bf16fp4"}

# Layout strings supported: only rcr (RowMajor A, ColMajor B, RowMajor C) --
# the standard GEMM layout the Old-TE bquant examples exercise (a_layout=R,
# b_layout=C in run_gemm_example_prec_type).
BQUANT_LAYOUT_TO_CK = QUANT_LAYOUT_TO_CK

# Pipeline map for BQuant kernels.
#   "compv3"       -> BQuantGemmPipelineAgBgCrCompV3    (non-preshuffle + preshufflebq)
#   "preshuffleb"  -> WPQuantBPipelineAgBgCrV2          (PreshuffleB=true variants)
#   "microscale"   -> MicroscaleGemmPipelineAgBgCrCompV3 (MX e8m0 scale variants)
BQUANT_PIPELINE_MAP = {
    "compv3":      "ck_tile::BQuantGemmPipelineAgBgCrCompV3",
    "preshuffleb": "ck_tile::WPQuantBPipelineAgBgCrV2",
    "microscale":  "ck_tile::MicroscaleGemmPipelineAgBgCrCompV3",
}

BQUANT_BASE_PIPELINE_MAP = {
    "compv3":      "ck_tile::BaseGemmPipelineAgBgCrCompV3",
    "preshuffleb": "ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2",
    # MX BQuant (QDataType=e8m0, PreshuffleB=false) falls into the else branch in
    # run_gemm_quant_example.inc -- same base as preshuffleb.
    "microscale":  "ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2",
}

BQUANT_SCHEDULER_TO_CK = QUANT_SCHEDULER_TO_CK


# =============================================================================
# Configuration dataclasses
# =============================================================================


# Was a verbatim redeclaration of codegen_common.TileConfig, fields and
# is_valid() alike. Aliased rather than renamed so call sites read unchanged.
BQuantTileConfig = TileConfig


@dataclass
class BQuantKernelSpec:
    """Complete specification for one non-grouped gemm_bquant kernel."""

    variant_key: str          # "fp8", "bf8", "fp8i4", "bf8i4", "mx_bf16*"
    layout: str               # "rcr"
    pipeline: str             # "compv3" | "preshuffleb" | "microscale"
    epilogue: str             # "cshuffle" (effective epilogue computed from tile)
    scheduler: str            # "intrawave"
    tile: BQuantTileConfig
    quant_group_m: int = 1
    quant_group_n: int = 1
    quant_group_k: int = 128
    preshuffle_b: bool = False
    preshuffle_bquant: bool = False
    double_smem_buffer: bool = False
    pad_m: bool = False
    pad_n: bool = False
    pad_k: bool = True
    block_size: int = 256
    k_block_per_cu: int = 1

    @property
    def name(self) -> str:
        t = self.tile
        return make_bquant_kernel_name(
            variant_key=self.variant_key,
            layout=self.layout,
            pipeline=self.pipeline,
            epilogue=self.epilogue,
            scheduler=self.scheduler,
            tile_m=t.tile_m, tile_n=t.tile_n, tile_k=t.tile_k,
            warp_m=t.warp_m, warp_n=t.warp_n, warp_k=t.warp_k,
            warp_tile_m=t.warp_tile_m, warp_tile_n=t.warp_tile_n, warp_tile_k=t.warp_tile_k,
            quant_group_m=self.quant_group_m,
            quant_group_n=self.quant_group_n,
            quant_group_k=self.quant_group_k,
            preshuffle_b=self.preshuffle_b,
            preshuffle_bquant=self.preshuffle_bquant,
            name_prefix=NAME_PREFIX,
        )


# =============================================================================
# Header generator
# =============================================================================


class BQuantKernelHeaderGenerator:
    """Generates a .hpp kernel specialization header for one BQuantKernelSpec."""

    def generate(self, spec: BQuantKernelSpec) -> str:
        variant = BQUANT_VARIANTS[spec.variant_key]
        t = spec.tile
        ns = "ns_" + spec.name
        struct = "Kernel_" + spec.name

        ck_a = variant["ck_a"]
        ck_b = variant["ck_b"]
        ck_c = variant["ck_c"]
        ck_q = variant["ck_q"]
        ck_acc = variant["ck_acc"]

        layout_a_ck = BQUANT_LAYOUT_TO_CK[spec.layout[0]]
        layout_b_ck = BQUANT_LAYOUT_TO_CK[spec.layout[1]]
        layout_c_ck = BQUANT_LAYOUT_TO_CK[spec.layout[2]]
        # BQ layout must match Old-TE's run_gemm_example_prec_type for the rcr
        # (a=R, b=C) path, which passes bq_layout = Col:
        #   run_gemm_example_with_layouts(Row, Col, Col, Col, Row)
        #                                 (A,  AQ,  B,   BQ,  C)
        # ColMajor is also REQUIRED by the WPQuantB (preshuffleb) pipeline, which
        # static_asserts "Bq must be col major". The BQ scale tensor is stored
        # [ceil(K/gK), ceil(N/gN)] with a column-major leading dim.
        layout_bq_ck = BQUANT_LAYOUT_TO_CK["c"]
        # AQ layout placeholder (unused for BQuant-only, same as A layout)
        layout_aq_ck = layout_a_ck

        pipeline_ck = BQUANT_PIPELINE_MAP[spec.pipeline]
        base_pipeline_ck = BQUANT_BASE_PIPELINE_MAP[spec.pipeline]
        scheduler_ck = BQUANT_SCHEDULER_TO_CK[spec.scheduler]

        # MX variants gate on gfx950 (native MX support). We emit a #error when
        # the kernel is compiled without CK_GFX950_SUPPORT so an accidental
        # gfx942 build fails loudly rather than silently miscompiling.
        arch_guard = ""
        if spec.variant_key in MX_VARIANTS:
            arch_guard = (
                "\n#ifndef CK_GFX950_SUPPORT\n"
                f'#error "{spec.name} is an MX (e8m0 block scale) kernel and requires '
                'gfx950 (CK_GFX950_SUPPORT). Do not build it for other archs."\n'
                "#endif\n"
            )

        pad_m = str(spec.pad_m).lower()
        pad_n = str(spec.pad_n).lower()
        pad_k = str(spec.pad_k).lower()
        preshuffle_b = str(spec.preshuffle_b).lower()
        preshuffle_bquant = str(spec.preshuffle_bquant).lower()
        double_smem_buffer = str(spec.double_smem_buffer).lower()

        # GemmConfig::TiledMMAPermuteN drives whether the B weight matrix is
        # pre-shuffled via shuffle_b_permuteN (permute_n) or plain shuffle_b, and
        # whether the BQ scale tensor is bq_permuteN'd. Only the PreshuffleB configs
        # override the GemmConfigBase default (false) to (N_Repeat % 2 == 0); every
        # other config inherits false. Mirrors GemmConfigPreshuffleB_BQuant_Prefill
        # (gemm_utils.hpp:214-215) and run_gemm_quant_example.inc:773,799-800 (which
        # select the permuteN path when TiledMMAPermuteN && BQuantGroupSize::kN == 1).
        n_repeat = (
            t.tile_n // (t.warp_n * t.warp_tile_n) if (t.warp_n * t.warp_tile_n) else 0
        )
        tiled_mma_permute_n = spec.preshuffle_b and (n_repeat % 2 == 0)
        tiled_mma_permute_n_str = str(tiled_mma_permute_n).lower()

        # BCastPolicy: Old-TE (run_gemm_quant_example.inc:117-120) selects
        #   b_cast_policy = (ADataType == BDataType) ? BeforeLDSWrite : AfterLDSRead
        # The GemmBQuantPipelineProblem BCastPolicy_ template arg defaults to
        # AfterLDSRead, so kernels where A and B share a dtype (fp8/fp8, bf8/bf8,
        # mx bf16/bf16) MUST override to BeforeLDSWrite -- otherwise the bridge
        # compiles a different, slower pipeline than Old-TE (mx_bf16bf16 was ~43%
        # off on gfx950). fp8i4/bf8i4/mx_bf16bf8/mx_bf16fp4 (A != B) keep AfterLDSRead.
        b_cast_before_lds = ck_a == ck_b
        b_cast_policy_ck = (
            "ck_tile::CastPolicy::BeforeLDSWrite"
            if b_cast_before_lds
            else "ck_tile::CastPolicy::AfterLDSRead"
        )

        # Determine which epilogue the kernel will use, mirroring run_gemm_quant_example.inc.
        # Delegates to bquant_effective_epilogue (same logic used by make_bquant_kernel_name)
        # so the generated C++ and the kernel name always agree.
        epilogue_kind = bquant_effective_epilogue(
            t.tile_n, t.warp_n, t.warp_tile_n, spec.quant_group_n, spec.preshuffle_b
        )

        epilogue_block = emit_quant_epilogue_block(epilogue_kind, ns)

        tile_dims = emit_quant_tile_dims(
            t, block_size=spec.block_size, k_block_per_cu=spec.k_block_per_cu
        )
        tile_shape = emit_quant_tile_shape()
        gemm_traits = emit_quant_gemm_traits("BQuantGrouped", ns)
        launch_prologue = emit_quant_launch_prologue(splitk_k="TileK")
        launch_tail = emit_quant_launch_tail(quant_type="BQuantGrouped")

        return emit_generated_header_preamble(
            "non-grouped gemm_bquant (block-scale) GEMM",
            "unified_gemm_bquant_codegen.py",
            extra=(arch_guard + "\n") if arch_guard else "",
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

// Single QuantGroupSize alias -- same type used for both AQ and BQ slots in the
// pipeline template; AQ is disabled via aq_ptr=nullptr at runtime for BQuant-only.
using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<
    {spec.quant_group_m}, {spec.quant_group_n}, {spec.quant_group_k}>>;

struct {struct} {{
    using ADataType   = {ns}::ADataType;
    using BDataType   = {ns}::BDataType;
    using CDataType   = {ns}::CDataType;
    using QDataType   = {ns}::QDataType;
    using AccDataType = {ns}::AccDataType;

{tile_dims}
    static constexpr ck_tile::index_t GroupSizeK = {spec.quant_group_k};

    static constexpr bool kPadM           = {pad_m};
    static constexpr bool kPadN           = {pad_n};
    static constexpr bool kPadK           = {pad_k};
    static constexpr bool APreshuffleQuant = false;
    static constexpr bool BPreshuffleQuant = {preshuffle_bquant};
    static constexpr bool PreshuffleB     = {preshuffle_b};
    static constexpr bool TransposeC      = false;
    static constexpr bool DoubleSmemBuffer = {double_smem_buffer};
    // TiledMMAPermuteN: selects shuffle_b_permuteN + bq_permuteN vs plain shuffle_b
    // for the B weight matrix / BQ scale tensor (see gemm_bquant_ctypes_lib.cpp).
    // Mirrors GemmConfigPreshuffleB_BQuant_Prefill (gemm_utils.hpp:214-215).
    static constexpr bool TiledMMAPermuteN = {tiled_mma_permute_n_str};

{tile_shape}

    // Config exposing the member names ck_tile::shuffle_b / shuffle_b_permuteN /
    // bq_permuteN expect (N_Tile, N_Warp, N_Warp_Tile, K_Warp_Tile). Used by the
    // ctypes lib to pre-shuffle the B weight matrix and BQ scales for PreshuffleB
    // kernels, matching Old-TE's host-side shuffle in
    // run_gemm_quant_example.inc:770-789 and the bq_permuteN at :799-815.
    struct BShuffleConfig {{
        static constexpr ck_tile::index_t N_Tile      = TileN;
        static constexpr ck_tile::index_t N_Warp      = WarpN;
        static constexpr ck_tile::index_t N_Warp_Tile = WarpTileN;
        static constexpr ck_tile::index_t K_Warp_Tile = WarpTileK;
    }};

{gemm_traits}

    using GemmPipelineProblemBase = ck_tile::GemmPipelineProblemBase<
        ADataType, BDataType, AccDataType, TileShape, GemmTraits>;

    using BaseGemmPipeline = {base_pipeline_ck}<GemmPipelineProblemBase>;

{launch_prologue}
            using PipelineProblem = ck_tile::GemmBQuantPipelineProblem<
                ADataType,
                BDataType,
                QDataType,
                AccDataType,
                TileShape,
                GemmTraits,
                QuantGroupSize,
                ADataType,        // ComputeDataType
                {scheduler_ck},
                has_hot_loop_.value,
                tail_number_.value,
                {b_cast_policy_ck}>;  // BCastPolicy -- Old-TE: A==B ? BeforeLDSWrite : AfterLDSRead

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
            extra_lines=(
                f"using QuantGroupSize = {ns}::QuantGroupSize;\n"
                f"constexpr ck_tile::index_t GroupSizeK = {ns}::{struct}::GroupSizeK;"
            ),
        )


# =============================================================================
# Config sweep
# =============================================================================


def _default_config() -> dict:
    """Default sweep config matching GemmConfigQuantDecode tile defaults.

    NOTE: this built-in header-enumeration sweep is gfx950-only, hence the
    literal arch below. Arch-correct warp_tile_k (gfx942 fp8/bf8 -> 32) is
    produced by the bridge via gemm_bquant_utils._warp_tile_k_for(); a gfx942
    sweep must pass a config with warp_tile_k=32 (128 silently outputs
    all-zeros on gfx942).
    """
    return quant_decode_default_config(
        warp_tile_k=fp8_warp_tile_k_for_arch("gfx950"),
        quant_groups=[
            {"quant_group_m": 1, "quant_group_n": 1, "quant_group_k": 128},
        ],
        preshuffle_b=False,
        preshuffle_bquant=False,
    )


def _build_specs(config: dict) -> List[BQuantKernelSpec]:
    specs = []
    pipeline  = config.get("pipeline", "compv3")
    epilogue  = config.get("epilogue", "cshuffle")
    scheduler = config.get("scheduler", "intrawave")
    pad_m     = config.get("pad_m", False)
    pad_n     = config.get("pad_n", False)
    pad_k     = config.get("pad_k", True)
    block_size         = config.get("block_size", 256)
    k_block_per_cu     = config.get("k_block_per_cu", 1)
    double_smem_buffer = config.get("double_smem_buffer", False)
    preshuffle_b       = config.get("preshuffle_b", False)
    preshuffle_bquant  = config.get("preshuffle_bquant", False)

    for variant_key, layout, tile, qg in iter_quant_axes(
        config,
        variants=BQUANT_VARIANTS,
        logger=log,
        pipeline=pipeline,
        pipeline_map=BQUANT_PIPELINE_MAP,
        extra_axis=("quant_groups",
                    [{"quant_group_m": 1, "quant_group_n": 1, "quant_group_k": 128}]),
    ):
        specs.append(BQuantKernelSpec(
            variant_key=variant_key,
            layout=layout,
            pipeline=pipeline,
            epilogue=epilogue,
            scheduler=scheduler,
            tile=tile,
            quant_group_m=qg.get("quant_group_m", 1),
            quant_group_n=qg.get("quant_group_n", 1),
            quant_group_k=qg.get("quant_group_k", 128),
            preshuffle_b=preshuffle_b,
            preshuffle_bquant=preshuffle_bquant,
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
        description="non-grouped gemm_bquant (block-scale) GEMM kernel header generator",
        op_label="BQuant",
        make_generator=BQuantKernelHeaderGenerator,
        build_specs=_build_specs,
        default_config=_default_config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
