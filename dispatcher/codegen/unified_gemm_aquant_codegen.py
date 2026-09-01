#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
AQuant (A-only quantized) GEMM Code Generator

Generates one .hpp per kernel config for the dispatcher's ctypes path.
Each header defines a SelectedKernel struct with a static launch() method
taking QuantGemmHostArgs -- compiled per-kernel via force-include:

    hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE gemm_aquant_ctypes_lib.cpp

Scope (matches Old-TE gemm_aquant_quantgrouped*.cpp):
  dtypes : fp8, bf8, fp8i4 (A=pk_int4), bf8i4 (A=pk_int4)
  layouts: rcr, rrr, crr, ccr  (non-preshufflequant)
           rcr, rrr, crr       (preshufflequant -- ccr rejected by Old-TE)
  pipeline: compv3  ->  AQuantGemmPipelineAgBgCrMem       (non-preshufflequant)
                        AQuantGemmPipelineAgBgCrCompV3    (preshufflequant)
  host args = ck_tile::QuantGemmHostArgs (aq_ptr set, bq_ptr = nullptr)

Naming convention (byte-exact with AQuantKernelConfig.name in gemm_aquant_utils.py):
    gemm_aquant_{variant}_{layout}_{pipeline}_{epilogue}_{scheduler}_
    {TileM}x{TileN}x{TileK}_{WarpM}x{WarpN}x{WarpK}_{WtM}x{WtN}x{WtK}_
    qg{gM}x{gN}x{gK}[_preshufflequant]

Reference:
    example/ck_tile/38_block_scale_gemm/gemm_aquant_quantgrouped.cpp
    example/ck_tile/38_block_scale_gemm/gemm_aquant_quantgrouped_preshufflequant.cpp
    example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc
    example/ck_tile/38_block_scale_gemm/gemm_utils.hpp
        (GemmConfigQuantDecodeInterwave, GemmConfigPreshuffleQuantDecode)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

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
    make_gemm_aquant_kernel_name,
    run_codegen_cli,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Dtype variant definitions
# Each entry maps a variant key to the (A, B, C, Q) CK types.
# For AQuant the *A* matrix is the quantized operand:
#   fp8/bf8  : A and B are the same 8-bit float, Q (A-scale) is float
#   fp8i4    : A = pk_int4 (quantized weight), B = fp8, Q (A-scale) = fp8
#   bf8i4    : A = pk_int4 (quantized weight), B = bf8, Q (A-scale) = bf8
# Matches GemmQuantTypeConfig<A, B, C, Q> in the Old-TE aquant .cpp files.
# =============================================================================

AQUANT_VARIANTS: Dict[str, Dict[str, str]] = {
    "fp8": {
        "ck_a": "ck_tile::fp8_t",
        "ck_b": "ck_tile::fp8_t",
        "ck_c": "ck_tile::half_t",
        "ck_q": "float",
        "ck_acc": "float",
    },
    "bf8": {
        "ck_a": "ck_tile::bf8_t",
        "ck_b": "ck_tile::bf8_t",
        "ck_c": "ck_tile::half_t",
        "ck_q": "float",
        "ck_acc": "float",
    },
    "fp8i4": {
        "ck_a": "ck_tile::pk_int4_t",
        "ck_b": "ck_tile::fp8_t",
        "ck_c": "ck_tile::half_t",
        "ck_q": "ck_tile::fp8_t",
        "ck_acc": "float",
    },
    "bf8i4": {
        "ck_a": "ck_tile::pk_int4_t",
        "ck_b": "ck_tile::bf8_t",
        "ck_c": "ck_tile::half_t",
        "ck_q": "ck_tile::bf8_t",
        "ck_acc": "float",
    },
}

# Layout characters -> CK layout type.
AQUANT_LAYOUT_TO_CK = QUANT_LAYOUT_TO_CK

# The 3-char layout tag encodes (A, B, C).  C is always RowMajor for quant kernels
# (static_assert in gemm_calc_quant).  The AQ (A-scale) tensor layout is ALWAYS
# RowMajor: Old-TE's aquant instance builder hardcodes
#   `using AQLayout = ck_tile::tensor_layout::gemm::RowMajor;`
# (tile_engine gemm_instance_builder.py populate .. AQLayout) and passes it explicitly
# to TileGemmQuantTraits for EVERY layout (rcr/rrr/crr/ccr). It does NOT let AQLayout
# default to ALayout. Emitting ColumnMajor for ccr (A=C B=C) produced a DIFFERENT
# compiled kernel type than Old-TE -- the strict objdump same-kernel gate flags it as
# KERNEL_MISMATCH (bridge AQ=ColumnMajor 'SG_' vs Old-TE AQ=RowMajor 'SH_' in the
# mangled TileGemmQuantTraits args), the ccr mem AQ-layout bug. Mirror Old-TE: AQ is
# RowMajor for all layouts.
AQUANT_AQ_LAYOUT = {
    "rcr": "r",
    "rrr": "r",
    "crr": "r",
    "ccr": "r",
}

# Pipeline map for AQuant kernels.
#   non-preshufflequant -> AQuantGemmPipelineAgBgCrMem   (base: BaseGemmPipelineAgBgCrMem)
#   preshufflequant     -> AQuantGemmPipelineAgBgCrCompV3(base: BaseGemmPipelineAgBgCrCompV3)
AQUANT_PIPELINE_MAP = {
    "mem": "ck_tile::AQuantGemmPipelineAgBgCrMem",
    "compv3": "ck_tile::AQuantGemmPipelineAgBgCrCompV3",
}

AQUANT_BASE_PIPELINE_MAP = {
    "mem": "ck_tile::BaseGemmPipelineAgBgCrMem",
    "compv3": "ck_tile::BaseGemmPipelineAgBgCrCompV3",
}

AQUANT_SCHEDULER_TO_CK = QUANT_SCHEDULER_TO_CK


# =============================================================================
# Configuration dataclasses
# =============================================================================


# Was a verbatim redeclaration of codegen_common.TileConfig, fields and
# is_valid() alike. Aliased rather than renamed so call sites read unchanged.
AQuantTileConfig = TileConfig


@dataclass
class AQuantKernelSpec:
    """Complete specification for one AQuant kernel."""

    variant_key: str          # "fp8", "bf8", "fp8i4", "bf8i4"
    layout: str               # "rcr", "rrr", "crr", "ccr"
    scheduler: str            # "interwave" (decode) or "intrawave" (preshufflequant)
    tile: AQuantTileConfig
    quant_group_m: int = 1
    quant_group_n: int = 1
    quant_group_k: int = 128
    preshuffle_aquant: bool = False
    # Pipeline selection DECOUPLED from preshuffle_aquant. None -> derive from
    # preshuffle_aquant (back-compat). Set "compv3" with preshuffle_aquant=False to emit
    # AQuantGemmPipelineAgBgCrCompV3 with the Traits APreshuffleQuant=false branch (the
    # compv3-without-preshuffle family Old-TE builds but the old coupling could not).
    pipeline: Optional[str] = None
    double_smem_buffer: bool = False
    pad_m: bool = False
    pad_n: bool = False
    pad_k: bool = False
    block_size: int = 256
    k_block_per_cu: int = 1
    # Epilogue variant the sweep asked for. Old-TE's aquant instance builder emits
    # DefaultGemm2DEpilogue for "default" and CShuffleEpilogue for "cshuffle"
    # (populate_default_gemm_aquant / populate_cshuffle_gemm_aquant). Both are valid
    # for QuantGemmKernel<..., AQuantGrouped>; the bridge must match whichever the
    # matched Old-TE stem uses or the CShuffle LDS-staging path spills (+scratch) and
    # runs ~2x slower on large tiles (the mem_default regression).
    epilogue: str = "cshuffle"

    @property
    def pipeline_key(self) -> str:
        """Pipeline map key, DECOUPLED from preshuffle.

        Explicit ``pipeline`` wins; otherwise derive from preshuffle_aquant
        (preshufflequant -> compv3, decode -> mem). Enables compv3 + APreshuffleQuant=false.
        """
        if self.pipeline is not None:
            return self.pipeline
        return "compv3" if self.preshuffle_aquant else "mem"

    @property
    def name(self) -> str:
        t = self.tile
        return make_gemm_aquant_kernel_name(
            variant_key=self.variant_key,
            layout=self.layout,
            pipeline=self.pipeline_key,
            epilogue=self.epilogue,
            scheduler=self.scheduler,
            tile_m=t.tile_m, tile_n=t.tile_n, tile_k=t.tile_k,
            warp_m=t.warp_m, warp_n=t.warp_n, warp_k=t.warp_k,
            warp_tile_m=t.warp_tile_m, warp_tile_n=t.warp_tile_n, warp_tile_k=t.warp_tile_k,
            quant_group_m=self.quant_group_m,
            quant_group_n=self.quant_group_n,
            quant_group_k=self.quant_group_k,
            preshuffle_aquant=self.preshuffle_aquant,
        )


# =============================================================================
# Header generator
# =============================================================================


class AQuantKernelHeaderGenerator:
    """Generates a .hpp kernel specialization header for one AQuantKernelSpec."""

    def generate(self, spec: AQuantKernelSpec) -> str:
        variant = AQUANT_VARIANTS[spec.variant_key]
        t = spec.tile
        ns = "ns_" + spec.name
        struct = "Kernel_" + spec.name

        ck_a = variant["ck_a"]
        ck_b = variant["ck_b"]
        ck_c = variant["ck_c"]
        ck_q = variant["ck_q"]
        ck_acc = variant["ck_acc"]

        layout_a_ck = AQUANT_LAYOUT_TO_CK[spec.layout[0]]
        layout_b_ck = AQUANT_LAYOUT_TO_CK[spec.layout[1]]
        layout_c_ck = AQUANT_LAYOUT_TO_CK[spec.layout[2]]
        layout_aq_ck = AQUANT_LAYOUT_TO_CK[AQUANT_AQ_LAYOUT[spec.layout]]
        # BQ layout is unused for AQuant-only (bq_ptr=nullptr), but it is still a
        # template parameter of TileGemmQuantTraits and so is part of the compiled
        # kernel's type. Old-TE's gemm_aquant instance passes ONLY AQLayout to
        # TileGemmQuantTraits and lets BQLayout take its template default,
        # `BQLayout_ = BLayout_` (tile_gemm_quant_traits.hpp:44). So Old-TE's BQLayout
        # is exactly BLayout: ColumnMajor for rcr/ccr (B col-major) but RowMajor for
        # rrr/crr (B row-major). Hardcoding ColumnMajor here matched Old-TE only for the
        # col-major-B layouts and emitted a DIFFERENT kernel type for rrr/crr, which the
        # objdump same-kernel gate flags as KERNEL_MISMATCH (the rrr/crr compv3 residual).
        # Mirror Old-TE: default BQLayout to BLayout.
        layout_bq_ck = layout_b_ck

        pipeline_key = spec.pipeline_key
        pipeline_ck = AQUANT_PIPELINE_MAP[pipeline_key]
        base_pipeline_ck = AQUANT_BASE_PIPELINE_MAP[pipeline_key]
        scheduler_ck = AQUANT_SCHEDULER_TO_CK[spec.scheduler]

        pad_m = str(spec.pad_m).lower()
        pad_n = str(spec.pad_n).lower()
        pad_k = str(spec.pad_k).lower()
        preshuffle_aquant = str(spec.preshuffle_aquant).lower()
        double_smem_buffer = str(spec.double_smem_buffer).lower()

        # AQuant configs never enable TiledMMAPermuteN (see gemm_aquant_effective_epilogue),
        # so PermuteN is never used. The remaining choice is CShuffle vs DefaultGemm2D,
        # driven by spec.epilogue: Old-TE emits DefaultGemm2DEpilogue for "default" and
        # CShuffleEpilogue for "cshuffle" (populate_{default,cshuffle}_gemm_aquant). Using
        # CShuffle for a "default" stem adds LDS staging + scratch spill (~2x slower).
        if spec.epilogue == "default":
            # Mirror populate_default_gemm_aquant: DefaultGemm2DEpilogueProblem takes
            # kPadM/kPadN (not MWave/NWave) and no CShuffle LDS staging.
            epilogue_block = f"""\
            using GemmEpilogue = ck_tile::DefaultGemm2DEpilogue<
                ck_tile::DefaultGemm2DEpilogueProblem<
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
                    kPadM, kPadN,
                    WarpTileM, WarpTileN, WarpTileK,
                    TransposeC>>;"""
        else:
            epilogue_block = emit_quant_epilogue_block("cshuffle", ns)
        tile_dims = emit_quant_tile_dims(
            t, block_size=spec.block_size, k_block_per_cu=spec.k_block_per_cu
        )
        # AQuant uses the spatially-local partitioner (prefill L2 locality), same as ABQuant.
        tile_shape = emit_quant_tile_shape(
            "ck_tile::GemmSpatiallyLocalTilePartitioner<TileShape, 8, 4>"
        )
        gemm_traits = emit_quant_gemm_traits("AQuantGrouped", ns)
        launch_prologue = emit_quant_launch_prologue(splitk_k="WarpTileK")
        launch_tail = emit_quant_launch_tail(quant_type="AQuantGrouped")

        return emit_generated_header_preamble(
            "AQuant (A-only quantized) GEMM", "unified_gemm_aquant_codegen.py"
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

// QuantGroupShape<sequence<gM, gN, gK>> -- same type used for the AQ slot in the
// pipeline template; BQ is disabled via bq_ptr=nullptr at runtime for AQuant-only.
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
    static constexpr bool APreshuffleQuant = {preshuffle_aquant};
    static constexpr bool BPreshuffleQuant = false;
    static constexpr bool PreshuffleB     = false;
    static constexpr bool TransposeC      = false;
    static constexpr bool DoubleSmemBuffer = {double_smem_buffer};

{tile_shape}

{gemm_traits}

    using GemmPipelineProblemBase = ck_tile::GemmPipelineProblemBase<
        ADataType, BDataType, AccDataType, TileShape, GemmTraits>;

    using BaseGemmPipeline = {base_pipeline_ck}<GemmPipelineProblemBase>;

{launch_prologue}
            // GemmAQuantPipelineProblem<A, AQ, B, C(=Acc), Shape, Traits,
            //   AQuantGroupSize, TransposeC, ComputeDataType, Scheduler, hot, tail>
            using PipelineProblem = ck_tile::GemmAQuantPipelineProblem<
                ADataType,
                QDataType,
                BDataType,
                AccDataType,
                TileShape,
                GemmTraits,
                QuantGroupSize,
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
            ck_q=ck_q,
            ck_acc=ck_acc,
            extra_lines=(
                f"using QuantGroupSize = {ns}::QuantGroupSize;\n"
                f"using ALayout = {ns}::ALayout;\n"
                f"using BLayout = {ns}::BLayout;\n"
                f"using CLayout = {ns}::CLayout;\n"
                "// AQ scale-tensor layout: always RowMajor (matches Old-TE, which hardcodes\n"
                "// AQLayout=RowMajor for every layout). The ctypes lib derives stride_AQ from\n"
                "// this compile-time type (RowMajor -> QK_A).\n"
                f"using AQLayout = {ns}::AQLayout;\n"
                f"constexpr ck_tile::index_t GroupSizeK = {ns}::{struct}::GroupSizeK;"
            ),
        )


# =============================================================================
# Config sweep
# =============================================================================


def _default_config(gfx_arch: str = "gfx950") -> dict:
    """Default sweep config matching GemmConfigQuantDecodeInterwave tile defaults.

    Non-preshufflequant decode kernels for every dtype x layout Old-TE supports.
    WarpTileK is arch-derived (get_k_warp_tile<fp8/bf8_t, 16>() = 128 on gfx950,
    32 on gfx942 for the decode path).
    """
    return {
        "variant_keys": ["fp8", "bf8", "fp8i4", "bf8i4"],
        "layouts": ["rcr", "rrr", "crr", "ccr"],
        "epilogues": ["cshuffle", "default"],
        # Old-TE never emits an interwave mem/decode kernel: gemm_validation_utils
        # AQUANT_TRAIT_UNSUPPORTED_COMBINATIONS marks ("mem","default","interwave")
        # and ("mem","cshuffle","interwave") as unsupported for every layout, so the
        # aquant mem pipeline is always Intrawave. Emitting Interwave here produces a
        # kernel that (a) matches no Old-TE stem and (b) takes the Interwave LDS path
        # that spills ~600B scratch and runs ~20-26% slower on the ccr access pattern
        # (the ccr mem regression). Mirror Old-TE: mem/decode = intrawave.
        "scheduler": "intrawave",
        "tile_configs": [
            # GemmConfigQuantDecodeInterwave: M=16, N=64, K=256/sizeof(PrecType)=256
            {"tile_m": 16, "tile_n": 64, "tile_k": 256,
             "warp_m": 1, "warp_n": 4, "warp_k": 1,
             "warp_tile_m": 16, "warp_tile_n": 16,
             "warp_tile_k": fp8_warp_tile_k_for_arch(gfx_arch, preshuffle_quant=False)},
        ],
        "quant_groups": [
            {"quant_group_m": 1, "quant_group_n": 1, "quant_group_k": 128},
        ],
        "pad_m": False,
        "pad_n": False,
        "pad_k": False,
        "block_size": 256,
        "k_block_per_cu": 1,
        "double_smem_buffer": False,
        "preshuffle_aquant": False,
    }


def _build_specs(config: dict) -> List[AQuantKernelSpec]:
    specs = []
    preshuffle_aquant = config.get("preshuffle_aquant", False)
    # Both aquant pipelines use Intrawave: preshufflequant uses Intrawave
    # (GemmConfigBase default) and the decode/mem path is Intrawave too -- Old-TE
    # marks every mem+interwave and compv3+interwave trait combination unsupported
    # (gemm_validation_utils AQUANT_TRAIT_UNSUPPORTED_COMBINATIONS), so there is no
    # interwave aquant kernel on the Old-TE side to pair against. An explicit
    # "scheduler" in the config still wins (the harness passes the Old-TE stem's
    # scheduler through verbatim); this only fixes the unattended default.
    default_scheduler = "intrawave"
    scheduler = config.get("scheduler", default_scheduler)
    # Pipeline selection, DECOUPLED from preshuffle_aquant. If absent, leave None so the
    # spec derives it from preshuffle_aquant (back-compat). An explicit "compv3" with
    # preshuffle_aquant=False selects the compv3-without-preshuffle family.
    pipeline = config.get("pipeline", None)
    pad_m     = config.get("pad_m", False)
    pad_n     = config.get("pad_n", False)
    pad_k     = config.get("pad_k", False)
    block_size         = config.get("block_size", 256)
    k_block_per_cu     = config.get("k_block_per_cu", 1)
    double_smem_buffer = config.get("double_smem_buffer", False)
    # Epilogue trait cross-product. Old-TE's default_config sweeps
    # ["cshuffle", "default"]; the two produce distinct kernels (CShuffleEpilogue
    # vs DefaultGemm2DEpilogue) so the bridge must generate both to match every
    # Old-TE stem. Back-compat default is cshuffle-only when unset.
    epilogues = config.get("epilogues")
    if epilogues is None:
        epilogues = [config.get("epilogue", "cshuffle")]

    def _layout_guard(layout: str) -> Optional[str]:
        if layout not in AQUANT_AQ_LAYOUT:
            return f"Unsupported layout {layout} -- skipping"
        # Old-TE rejects the ccr layout for the preshufflequant path.
        if preshuffle_aquant and layout == "ccr":
            return "ccr layout is unsupported for preshufflequant -- skipping"
        return None

    # AQuant has no pipeline axis, so no pipeline_map is passed.
    for variant_key, layout, tile, qg in iter_quant_axes(
        config,
        variants=AQUANT_VARIANTS,
        logger=log,
        extra_axis=("quant_groups",
                    [{"quant_group_m": 1, "quant_group_n": 1, "quant_group_k": 128}]),
        layout_guard=_layout_guard,
    ):
        for epilogue in epilogues:
            if epilogue not in ("cshuffle", "default"):
                log.warning("Unknown epilogue %s -- skipping", epilogue)
                continue
            specs.append(AQuantKernelSpec(
                variant_key=variant_key,
                layout=layout,
                scheduler=scheduler,
                tile=tile,
                quant_group_m=qg.get("quant_group_m", 1),
                quant_group_n=qg.get("quant_group_n", 1),
                quant_group_k=qg.get("quant_group_k", 128),
                preshuffle_aquant=preshuffle_aquant,
                pipeline=pipeline,
                double_smem_buffer=double_smem_buffer,
                pad_m=pad_m,
                pad_n=pad_n,
                pad_k=pad_k,
                block_size=block_size,
                k_block_per_cu=k_block_per_cu,
                epilogue=epilogue,
            ))

    return specs

# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    return run_codegen_cli(
        description="AQuant (A-only quantized) GEMM kernel header generator",
        op_label="AQuant",
        make_generator=AQuantKernelHeaderGenerator,
        build_specs=_build_specs,
        default_config=_default_config,
        arch_aware=True,
        default_gfx_arch="gfx950",
    )


if __name__ == "__main__":
    raise SystemExit(main())
