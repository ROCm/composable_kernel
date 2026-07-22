#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
GroupedGemm BQuant Code Generator

Generates one .hpp per kernel config for the dispatcher's ctypes path.
Each header defines a SelectedKernel struct with a static launch() method
taking QuantGemmHostArgs — compiled per-kernel via force-include:

    hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE grouped_gemm_bquant_ctypes_lib.cpp

Initial scope: fp8 and bf8 dtype variants, non-preshuffle, compv3 pipeline,
rcr layout, configurable QuantGroupShape.

Naming convention (byte-exact with BQuantKernelConfig.name in grouped_gemm_bquant_utils.py):
    grouped_gemm_bquant_{dtype_a}_{layout}_{pipeline}_{epilogue}_{scheduler}_
    {TileM}x{TileN}x{TileK}_{WarpM}x{WarpN}x{WarpK}_{WtM}x{WtN}x{WtK}_
    qg{gM}x{gN}x{gK}[_preshuffleb][_preshufflebq]

Reference:
    example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_fp8.cpp
    example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc
    example/ck_tile/38_block_scale_gemm/gemm_utils.hpp  (GemmConfigQuantDecode)
"""

import argparse
import itertools
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from codegen_common import make_bquant_kernel_name, bquant_effective_epilogue

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Dtype variant definitions
# Each entry: (dtype_key, ADataType, BDataType, CDataType, QDataType)
# Matches example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_*.cpp
# =============================================================================

BQUANT_VARIANTS: Dict[str, Dict[str, str]] = {
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
    "fp8i4": {
        "dtype_a": "fp8",
        "dtype_b": "pk_int4",
        "dtype_c": "half",
        "dtype_q": "fp8",
        "ck_a": "ck_tile::fp8_t",
        "ck_b": "ck_tile::pk_int4_t",
        "ck_c": "ck_tile::half_t",
        "ck_q": "ck_tile::fp8_t",
        "ck_acc": "float",
    },
    "bf8i4": {
        "dtype_a": "bf8",
        "dtype_b": "pk_int4",
        "dtype_c": "half",
        "dtype_q": "bf8",
        "ck_a": "ck_tile::bf8_t",
        "ck_b": "ck_tile::pk_int4_t",
        "ck_c": "ck_tile::half_t",
        "ck_q": "ck_tile::bf8_t",
        "ck_acc": "float",
    },
    # MX microscale variants — Q-type is e8m0 (block scale), pipeline = microscale
    "mx_bf16bf16": {
        "dtype_a": "bf16",
        "dtype_b": "bf16",
        "dtype_c": "bf16",
        "dtype_q": "e8m0",
        "ck_a": "ck_tile::bf16_t",
        "ck_b": "ck_tile::bf16_t",
        "ck_c": "ck_tile::bf16_t",
        "ck_q": "ck_tile::e8m0_t",
        "ck_acc": "float",
    },
    "mx_bf16bf8": {
        "dtype_a": "bf16",
        "dtype_b": "bf8",
        "dtype_c": "bf16",
        "dtype_q": "e8m0",
        "ck_a": "ck_tile::bf16_t",
        "ck_b": "ck_tile::bf8_t",
        "ck_c": "ck_tile::bf16_t",
        "ck_q": "ck_tile::e8m0_t",
        "ck_acc": "float",
    },
    "mx_bf16fp4": {
        "dtype_a": "bf16",
        "dtype_b": "pk_fp4",
        "dtype_c": "bf16",
        "dtype_q": "e8m0",
        "ck_a": "ck_tile::bf16_t",
        "ck_b": "ck_tile::pk_fp4_t",
        "ck_c": "ck_tile::bf16_t",
        "ck_q": "ck_tile::e8m0_t",
        "ck_acc": "float",
    },
}

# Layout strings supported: only rcr for initial implementation
# (RowMajor A, ColMajor B, RowMajor C) — standard GEMM layout for quant kernels
BQUANT_LAYOUT_TO_CK = {
    "r": "ck_tile::tensor_layout::gemm::RowMajor",
    "c": "ck_tile::tensor_layout::gemm::ColumnMajor",
}

# Pipeline map for BQuant kernels.
# "preshuffleb"  -> WPQuantBPipelineAgBgCrV2        (preshuffle_b=true variants)
# "microscale"   -> MicroscaleGemmPipelineAgBgCrCompV3 (MX e8m0 scale variants)
BQUANT_PIPELINE_MAP = {
    "compv3":      "ck_tile::BQuantGemmPipelineAgBgCrCompV3",
    "preshuffleb": "ck_tile::WPQuantBPipelineAgBgCrV2",
    "microscale":  "ck_tile::MicroscaleGemmPipelineAgBgCrCompV3",
}

BQUANT_BASE_PIPELINE_MAP = {
    "compv3":      "ck_tile::BaseGemmPipelineAgBgCrCompV3",
    "preshuffleb": "ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2",
    # MX BQuant (QDataType=e8m0, PreshuffleB=false) falls into the else branch in
    # run_gemm_quant_example.inc — same base as preshuffleb.
    "microscale":  "ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2",
}

BQUANT_SCHEDULER_TO_CK = {
    "intrawave": "ck_tile::GemmPipelineScheduler::Intrawave",
    "interwave": "ck_tile::GemmPipelineScheduler::Interwave",
}


# =============================================================================
# Configuration dataclasses
# =============================================================================


@dataclass
class BQuantTileConfig:
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
class BQuantKernelSpec:
    """Complete specification for one BQuant kernel."""

    variant_key: str          # "fp8" or "bf8"
    layout: str               # "rcr"
    pipeline: str             # "compv3"
    epilogue: str             # "cshuffle"
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
        # BQ is always RowMajor: scales are stored [ceil(K/gK), ceil(N/gN)]
        layout_bq_ck = BQUANT_LAYOUT_TO_CK["r"]
        # AQ layout placeholder (unused for BQuant-only, same as A layout)
        layout_aq_ck = layout_a_ck

        pipeline_ck = BQUANT_PIPELINE_MAP[spec.pipeline]
        base_pipeline_ck = BQUANT_BASE_PIPELINE_MAP[spec.pipeline]
        scheduler_ck = BQUANT_SCHEDULER_TO_CK[spec.scheduler]

        pad_m = str(spec.pad_m).lower()
        pad_n = str(spec.pad_n).lower()
        pad_k = str(spec.pad_k).lower()
        preshuffle_b = str(spec.preshuffle_b).lower()
        preshuffle_bquant = str(spec.preshuffle_bquant).lower()
        double_smem_buffer = str(spec.double_smem_buffer).lower()

        # Determine which epilogue the kernel will use, mirroring run_gemm_quant_example.inc.
        # Delegates to bquant_effective_epilogue (same logic used by make_bquant_kernel_name)
        # so the generated C++ and the kernel name always agree.
        use_permute_n_epilogue = (
            bquant_effective_epilogue(t.tile_n, t.warp_n, t.warp_tile_n, spec.quant_group_n)
            == "permute_n"
        )

        # Build the epilogue block outside the f-string to keep it readable.
        # PermuteNEpilogueProblem takes two extra trailing args (false, 1) vs CShuffleEpilogueProblem.
        if use_permute_n_epilogue:
            epilogue_block = f"""\
            using GemmEpilogue = ck_tile::PermuteNEpilogue<
                ck_tile::PermuteNEpilogueProblem<
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
                    TransposeC,
                    false,
                    1>>;"""
        else:
            epilogue_block = f"""\
            using GemmEpilogue = ck_tile::CShuffleEpilogue<
                ck_tile::CShuffleEpilogueProblem<
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
                    TransposeC>>;"""

        return f"""\
// SPDX-License-Identifier: MIT
// Auto-generated BQuantGrouped GEMM kernel header.
// DO NOT EDIT — regenerate via unified_grouped_gemm_bquant_codegen.py
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm_quant.hpp"
#include "ck_tile/ops/epilogue.hpp"

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

// Single QuantGroupSize alias — same type used for both AQ and BQ slots in the
// pipeline template; AQ is disabled via aq_ptr=nullptr at runtime for BQuant-only.
using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<
    {spec.quant_group_m}, {spec.quant_group_n}, {spec.quant_group_k}>>;

struct {struct} {{
    using ADataType   = {ns}::ADataType;
    using BDataType   = {ns}::BDataType;
    using CDataType   = {ns}::CDataType;
    using QDataType   = {ns}::QDataType;
    using AccDataType = {ns}::AccDataType;

    static constexpr ck_tile::index_t TileM      = {t.tile_m};
    static constexpr ck_tile::index_t TileN      = {t.tile_n};
    static constexpr ck_tile::index_t TileK      = {t.tile_k};
    static constexpr ck_tile::index_t WarpM      = {t.warp_m};
    static constexpr ck_tile::index_t WarpN      = {t.warp_n};
    static constexpr ck_tile::index_t WarpK      = {t.warp_k};
    static constexpr ck_tile::index_t WarpTileM  = {t.warp_tile_m};
    static constexpr ck_tile::index_t WarpTileN  = {t.warp_tile_n};
    static constexpr ck_tile::index_t WarpTileK  = {t.warp_tile_k};
    static constexpr ck_tile::index_t BlockSize  = {spec.block_size};
    static constexpr int               kBlockPerCu = {spec.k_block_per_cu};
    static constexpr ck_tile::index_t GroupSizeK = {spec.quant_group_k};

    static constexpr bool kPadM           = {pad_m};
    static constexpr bool kPadN           = {pad_n};
    static constexpr bool kPadK           = {pad_k};
    static constexpr bool APreshuffleQuant = false;
    static constexpr bool BPreshuffleQuant = {preshuffle_bquant};
    static constexpr bool PreshuffleB     = {preshuffle_b};
    static constexpr bool TransposeC      = false;
    static constexpr bool DoubleSmemBuffer = {double_smem_buffer};

    using TileShape = ck_tile::TileGemmShape<
        ck_tile::sequence<TileM, TileN, TileK>,
        ck_tile::sequence<WarpM, WarpN, WarpK>,
        ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>>;

    using TilePartitioner = ck_tile::GemmTile1DPartitioner<TileShape>;

    using GemmTraits = ck_tile::TileGemmQuantTraits<
        kPadM, kPadN, kPadK,
        APreshuffleQuant, BPreshuffleQuant, PreshuffleB,
        {ns}::ALayout, {ns}::BLayout, {ns}::CLayout,
        ck_tile::QuantType::BQuantGrouped,
        {ns}::AQLayout, {ns}::BQLayout,
        TransposeC, DoubleSmemBuffer>;

    using GemmPipelineProblemBase = ck_tile::GemmPipelineProblemBase<
        ADataType, BDataType, AccDataType, TileShape, GemmTraits>;

    using BaseGemmPipeline = {base_pipeline_ck}<GemmPipelineProblemBase>;

    static float launch(const ck_tile::QuantGemmHostArgs& args,
                        const ck_tile::stream_config& s)
    {{
        // hot-loop / tail dispatch — mirrors run_gemm_quant_example.inc
        const ck_tile::index_t K_split =
            (args.k_batch == 1)
                ? ck_tile::integer_least_multiple(args.K, TileK)
                : ck_tile::get_splitk_batch_k_read(args.K, args.k_batch, TileK);

        const ck_tile::index_t num_loop  = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop          = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        const auto Run = [&](auto has_hot_loop_, auto tail_number_) {{
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
                tail_number_.value>;

            using GemmPipeline = {pipeline_ck}<PipelineProblem>;

{epilogue_block}

            using Kernel = ck_tile::QuantGemmKernel<
                TilePartitioner, GemmPipeline, GemmEpilogue,
                ck_tile::QuantType::BQuantGrouped>;

            auto kargs = Kernel::MakeKernelArgs(args);
            if(!Kernel::IsSupportedArgument(kargs))
                return -1.0f;

            const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
            const dim3 blocks = Kernel::BlockSize();
            return ck_tile::launch_kernel(
                s, ck_tile::make_kernel<kBlockPerCu>(Kernel{{}}, grids, blocks, 0, kargs));
        }};

        return BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
    }}
}};

using SelectedKernel = {struct};

}} // namespace {ns}

#ifdef CK_TILE_SINGLE_KERNEL_INCLUDE
using SelectedKernel = {ns}::{struct};
constexpr const char* KERNEL_NAME = {ns}::KERNEL_NAME;
using ADataType   = {ck_a};
using BDataType   = {ck_b};
using CDataType   = {ck_c};
using QDataType   = {ck_q};
using AccDataType = {ck_acc};
using QuantGroupSize = {ns}::QuantGroupSize;
constexpr ck_tile::index_t GroupSizeK = {ns}::{struct}::GroupSizeK;
#endif // CK_TILE_SINGLE_KERNEL_INCLUDE
"""


# =============================================================================
# Config sweep
# =============================================================================


def _default_config() -> dict:
    """Default sweep config matching GemmConfigQuantDecode tile defaults."""
    return {
        "variant_keys": ["fp8", "bf8"],
        "layouts": ["rcr"],
        "pipeline": "compv3",
        "epilogue": "cshuffle",
        "scheduler": "intrawave",
        "tile_configs": [
            # GemmConfigQuantDecode<fp8_t>: M=16, N=64, K=256/sizeof(fp8_t)=256
            # WarpTileK=128: get_k_warp_tile<fp8_t, M_Warp_Tile=16>() on gfx950 = 128
            {"tile_m": 16, "tile_n": 64, "tile_k": 256,
             "warp_m": 1, "warp_n": 4, "warp_k": 1,
             "warp_tile_m": 16, "warp_tile_n": 16, "warp_tile_k": 128},
        ],
        "quant_groups": [
            {"quant_group_m": 1, "quant_group_n": 1, "quant_group_k": 128},
        ],
        "pad_m": False,
        "pad_n": False,
        "pad_k": True,
        "block_size": 256,
        "k_block_per_cu": 1,
        "double_smem_buffer": False,
        "preshuffle_b": False,
        "preshuffle_bquant": False,
    }


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

    for variant_key, layout, tile_dict, qg in itertools.product(
        config.get("variant_keys", ["fp8"]),
        config.get("layouts", ["rcr"]),
        config.get("tile_configs", []),
        config.get("quant_groups", [{"quant_group_m": 1, "quant_group_n": 1, "quant_group_k": 128}]),
    ):
        if variant_key not in BQUANT_VARIANTS:
            log.warning("Unknown variant_key %s — skipping", variant_key)
            continue
        if pipeline not in BQUANT_PIPELINE_MAP:
            log.warning("Unsupported pipeline %s — skipping", pipeline)
            continue

        tile = BQuantTileConfig(
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
        if not tile.is_valid():
            log.debug("Invalid tile config %s — skipping", tile)
            continue

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
# Generation entry point
# =============================================================================


def generate_kernels(
    output_dir: Path,
    config: Optional[dict] = None,
    parallel: bool = True,
) -> List[Path]:
    """Generate all BQuant kernel headers into output_dir.

    Returns list of generated .hpp paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or _default_config()
    specs = _build_specs(cfg)

    if not specs:
        log.warning("No kernel specs produced from config — check variant_keys and tile_configs")
        return []

    log.info("Generating %d BQuant kernel headers into %s", len(specs), output_dir)

    gen = BQuantKernelHeaderGenerator()
    generated: List[Path] = []

    def _generate_one(spec: BQuantKernelSpec) -> Path:
        header = gen.generate(spec)
        out_path = output_dir / f"{spec.name}.hpp"
        out_path.write_text(header)
        log.info("  wrote %s", out_path.name)
        return out_path

    if parallel and len(specs) > 1:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as ex:
            futures = {ex.submit(_generate_one, s): s for s in specs}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    generated.append(fut.result())
                except Exception as e:
                    log.error("Failed generating %s: %s", futures[fut].name, e)
    else:
        for spec in specs:
            try:
                generated.append(_generate_one(spec))
            except Exception as e:
                log.error("Failed generating %s: %s", spec.name, e)

    log.info("Generated %d / %d headers", len(generated), len(specs))
    return generated


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="BQuantGrouped GEMM kernel header generator"
    )
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write generated .hpp files")
    parser.add_argument("--config", type=Path,
                        help="JSON config file (defaults to built-in sweep)")
    parser.add_argument("--config-json", type=str,
                        help="Inline JSON config string")
    parser.add_argument("--no-parallel", action="store_true",
                        help="Disable parallel generation")
    parser.add_argument("--list-names", action="store_true",
                        help="Print kernel names that would be generated and exit")
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
        specs = _build_specs(cfg or _default_config())
        for s in specs:
            print(s.name)
        return 0

    paths = generate_kernels(
        output_dir=args.output_dir,
        config=cfg,
        parallel=not args.no_parallel,
    )
    return 0 if paths else 1


if __name__ == "__main__":
    raise SystemExit(main())
