#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
GroupedGemm ABQuant Code Generator

Generates one .hpp per kernel config for the dispatcher's ctypes path.
Each header defines a SelectedKernel struct with a static launch() method
taking QuantGemmHostArgs — compiled per-kernel via force-include:

    hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE grouped_gemm_abquant_ctypes_lib.cpp

ABQuant: A-side and B-side quantization simultaneously (QuantType::ABQuantGrouped).
Covers fp8, bf8 dtype variants with separate AQuantGroupSize and BQuantGroupSize.

Constraint: AQuantGroupSize::kK == BQuantGroupSize::kK (enforced at codegen time).

BQ layout: ColumnMajor [ceil(K/bK), ceil(N/bN)] — the kernel asserts ColumnMajor for BQ.

Pipeline selection (mirrors run_gemm_quant_example.inc):
  compv3:     ABQuantGemmPipelineAgBgCrCompV3 (GemmConfigABQuantPrefill, non-gfx950)
  eightwaves: ABQuantGemmPipelineAgBgCrEightWaves (GemmConfigEightWaves, gfx950)
  preshuffleb: WPABQuantBPipelineAgBgCrV2 (GemmConfigPreshuffleB_ABQuant_Prefill)

Naming convention (byte-exact with ABQuantKernelConfig.name in grouped_gemm_abquant_utils.py):
    grouped_gemm_abquant_{variant_key}_{layout}_{pipeline}_{epilogue}_{scheduler}_
    {TileM}x{TileN}x{TileK}_{WarpM}x{WarpN}x{WarpK}_{WtM}x{WtN}x{WtK}_
    aqg{aM}x{aN}x{aK}_bqg{bM}x{bN}x{bK}
    [_preshuffleb][_preshuffleaq][_preshufflebq][_transposec]

Reference:
    example/ck_tile/38_block_scale_gemm/gemm_abquant_quantgrouped_fp8.cpp
    example/ck_tile/38_block_scale_gemm/gemm_abquant_quantgrouped.h
    example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc
    example/ck_tile/38_block_scale_gemm/gemm_utils.hpp
"""

import argparse
import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from codegen_common import make_abquant_kernel_name, abquant_effective_epilogue

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Dtype variant definitions
# Both AQ and BQ share the same QDataType for ABQuant.
# =============================================================================

ABQUANT_VARIANTS: Dict[str, Dict[str, str]] = {
    "fp8": {
        "dtype_a": "fp8",
        "dtype_b": "fp8",
        "dtype_c": "half",
        "dtype_q": "float",     # both AQDataType and BQDataType
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

ABQUANT_LAYOUT_TO_CK = {
    "r": "ck_tile::tensor_layout::gemm::RowMajor",
    "c": "ck_tile::tensor_layout::gemm::ColumnMajor",
}

# Pipeline map for ABQuant kernels.
# compv3     → ABQuantGemmPipelineAgBgCrCompV3    (non-gfx950, GemmConfigABQuantPrefill)
# eightwaves → ABQuantGemmPipelineAgBgCrEightWaves (gfx950, GemmConfigEightWaves)
# preshuffleb→ WPABQuantBPipelineAgBgCrV2          (GemmConfigPreshuffleB_ABQuant_Prefill)
ABQUANT_PIPELINE_MAP = {
    "compv3":      "ck_tile::ABQuantGemmPipelineAgBgCrCompV3",
    "eightwaves":  "ck_tile::ABQuantGemmPipelineAgBgCrEightWaves",
    "preshuffleb": "ck_tile::WPABQuantBPipelineAgBgCrV2",
}

ABQUANT_BASE_PIPELINE_MAP = {
    "compv3":      "ck_tile::BaseGemmPipelineAgBgCrCompV3",
    "eightwaves":  "ck_tile::BaseGemmPipelineAgBgCrCompV3",
    "preshuffleb": "ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2",
}

ABQUANT_SCHEDULER_TO_CK = {
    "intrawave": "ck_tile::GemmPipelineScheduler::Intrawave",
    "interwave": "ck_tile::GemmPipelineScheduler::Interwave",
}


# =============================================================================
# Configuration dataclasses
# =============================================================================


@dataclass
class ABQuantTileConfig:
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
class ABQuantKernelSpec:
    """Complete specification for one ABQuant kernel."""

    variant_key: str          # "fp8", "bf8"
    layout: str               # "rcr"
    pipeline: str             # "compv3", "eightwaves", "preshuffleb"
    epilogue: str             # "cshuffle" (effective may be permute_n)
    scheduler: str            # "intrawave"
    tile: ABQuantTileConfig

    # Separate group sizes for A-side and B-side quantization
    aquant_group_m: int = 1
    aquant_group_n: int = 1
    aquant_group_k: int = 128

    bquant_group_m: int = 1
    bquant_group_n: int = 1
    bquant_group_k: int = 128

    preshuffle_b: bool  = False   # PreshuffleB (weight preshuffle)
    preshuffle_aq: bool = False   # APreshuffleQuant
    preshuffle_bq: bool = False   # BPreshuffleQuant
    transpose_c: bool   = False   # TransposeC (true for eightwaves/gfx950)
    double_smem_buffer: bool = False
    pad_m: bool = False
    pad_n: bool = False
    pad_k: bool = False           # ABQuant prefill tiles have kPadK=false
    block_size: int = 256
    k_block_per_cu: int = 1

    def __post_init__(self):
        if self.aquant_group_k != self.bquant_group_k:
            raise ValueError(
                f"ABQuant requires AQuantGroupSize::kK == BQuantGroupSize::kK, "
                f"got {self.aquant_group_k} != {self.bquant_group_k}"
            )

    @property
    def name(self) -> str:
        t = self.tile
        return make_abquant_kernel_name(
            variant_key=self.variant_key,
            layout=self.layout,
            pipeline=self.pipeline,
            epilogue=self.epilogue,
            scheduler=self.scheduler,
            tile_m=t.tile_m, tile_n=t.tile_n, tile_k=t.tile_k,
            warp_m=t.warp_m, warp_n=t.warp_n, warp_k=t.warp_k,
            warp_tile_m=t.warp_tile_m, warp_tile_n=t.warp_tile_n, warp_tile_k=t.warp_tile_k,
            aquant_group_m=self.aquant_group_m,
            aquant_group_n=self.aquant_group_n,
            aquant_group_k=self.aquant_group_k,
            bquant_group_m=self.bquant_group_m,
            bquant_group_n=self.bquant_group_n,
            bquant_group_k=self.bquant_group_k,
            preshuffle_b=self.preshuffle_b,
            preshuffle_aq=self.preshuffle_aq,
            preshuffle_bq=self.preshuffle_bq,
            transpose_c=self.transpose_c,
        )


# =============================================================================
# Header generator
# =============================================================================


class ABQuantKernelHeaderGenerator:
    """Generates a .hpp kernel specialization header for one ABQuantKernelSpec."""

    def generate(self, spec: ABQuantKernelSpec) -> str:
        variant = ABQUANT_VARIANTS[spec.variant_key]
        t = spec.tile
        ns = "ns_" + spec.name
        struct = "Kernel_" + spec.name

        ck_a   = variant["ck_a"]
        ck_b   = variant["ck_b"]
        ck_c   = variant["ck_c"]
        ck_q   = variant["ck_q"]   # same type for both AQ and BQ
        ck_acc = variant["ck_acc"]

        layout_a_ck = ABQUANT_LAYOUT_TO_CK[spec.layout[0]]
        layout_b_ck = ABQUANT_LAYOUT_TO_CK[spec.layout[1]]
        layout_c_ck = ABQUANT_LAYOUT_TO_CK[spec.layout[2]]
        # AQ is RowMajor: [ceil(M/gM), ceil(K/gK)]
        layout_aq_ck = ABQUANT_LAYOUT_TO_CK["r"]
        # BQ is ColumnMajor: [ceil(K/gK), ceil(N/gN)] — kernel asserts against RowMajor BQ.
        layout_bq_ck = ABQUANT_LAYOUT_TO_CK["c"]

        pipeline_ck      = ABQUANT_PIPELINE_MAP[spec.pipeline]
        base_pipeline_ck = ABQUANT_BASE_PIPELINE_MAP[spec.pipeline]
        scheduler_ck     = ABQUANT_SCHEDULER_TO_CK[spec.scheduler]

        pad_m = str(spec.pad_m).lower()
        pad_n = str(spec.pad_n).lower()
        pad_k = str(spec.pad_k).lower()
        preshuffle_b  = str(spec.preshuffle_b).lower()
        preshuffle_aq = str(spec.preshuffle_aq).lower()
        preshuffle_bq = str(spec.preshuffle_bq).lower()
        transpose_c       = str(spec.transpose_c).lower()
        double_smem_buffer = str(spec.double_smem_buffer).lower()
        is_eight_waves    = str(spec.pipeline == "eightwaves").lower()

        # Epilogue selection — B-side tile geometry governs PermuteN (except EightWaves).
        use_permute_n_epilogue = (
            abquant_effective_epilogue(
                t.tile_n, t.warp_n, t.warp_tile_n, spec.bquant_group_n, spec.pipeline
            ) == "permute_n"
        )

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
// Auto-generated ABQuantGrouped GEMM kernel header.
// DO NOT EDIT — regenerate via unified_grouped_gemm_abquant_codegen.py
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
using QDataType   = {ck_q};   // shared by AQDataType and BQDataType
using AccDataType = {ck_acc};

using ALayout  = {layout_a_ck};
using BLayout  = {layout_b_ck};
using CLayout  = {layout_c_ck};
using AQLayout = {layout_aq_ck};  // RowMajor: [ceil(M/aM), ceil(K/aK)]
using BQLayout = {layout_bq_ck};  // ColumnMajor: [ceil(K/bK), ceil(N/bN)]

// Separate group sizes — ABQuant requires AQuantGroupSize::kK == BQuantGroupSize::kK
using AQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<
    {spec.aquant_group_m}, {spec.aquant_group_n}, {spec.aquant_group_k}>>;
using BQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<
    {spec.bquant_group_m}, {spec.bquant_group_n}, {spec.bquant_group_k}>>;

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

    static constexpr bool kPadM           = {pad_m};
    static constexpr bool kPadN           = {pad_n};
    static constexpr bool kPadK           = {pad_k};
    static constexpr bool APreshuffleQuant = {preshuffle_aq};
    static constexpr bool BPreshuffleQuant = {preshuffle_bq};
    static constexpr bool PreshuffleB     = {preshuffle_b};
    static constexpr bool TransposeC      = {transpose_c};
    static constexpr bool DoubleSmemBuffer = {double_smem_buffer};
    static constexpr bool IsEightWaves    = {is_eight_waves};

    using TileShape = ck_tile::TileGemmShape<
        ck_tile::sequence<TileM, TileN, TileK>,
        ck_tile::sequence<WarpM, WarpN, WarpK>,
        ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>>;

    using TilePartitioner = ck_tile::GemmTile1DPartitioner<TileShape>;

    using GemmTraits = ck_tile::TileGemmQuantTraits<
        kPadM, kPadN, kPadK,
        APreshuffleQuant, BPreshuffleQuant, PreshuffleB,
        {ns}::ALayout, {ns}::BLayout, {ns}::CLayout,
        ck_tile::QuantType::ABQuantGrouped,
        {ns}::AQLayout, {ns}::BQLayout,
        TransposeC, DoubleSmemBuffer>;

    using GemmPipelineProblemBase = ck_tile::GemmPipelineProblemBase<
        ADataType, BDataType, AccDataType, TileShape, GemmTraits>;

    using BaseGemmPipeline = {base_pipeline_ck}<GemmPipelineProblemBase>;

    static float launch(const ck_tile::QuantGemmHostArgs& args,
                        const ck_tile::stream_config& s)
    {{
        const ck_tile::index_t K_split =
            (args.k_batch == 1)
                ? ck_tile::integer_least_multiple(args.K, TileK)
                : ck_tile::get_splitk_batch_k_read(args.K, args.k_batch, TileK);

        const ck_tile::index_t num_loop  = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop          = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        const auto Run = [&](auto has_hot_loop_, auto tail_number_) {{
            using PipelineProblem = ck_tile::GemmABQuantPipelineProblem<
                ADataType,
                QDataType,     // AQDataType (A-side scale, same type as BQ)
                BDataType,
                QDataType,     // BQDataType (B-side scale)
                AccDataType,   // 5th arg is AccDataType, matching run_gemm_quant_example.inc
                TileShape,
                GemmTraits,
                AQuantGroupSize,
                BQuantGroupSize,
                TransposeC,
                ADataType,     // AComputeDataType
                {scheduler_ck},
                has_hot_loop_.value,
                tail_number_.value>;

            using GemmPipeline = {pipeline_ck}<PipelineProblem>;

{epilogue_block}

            using Kernel = ck_tile::QuantGemmKernel<
                TilePartitioner, GemmPipeline, GemmEpilogue,
                ck_tile::QuantType::ABQuantGrouped>;

            auto kargs = Kernel::MakeKernelArgs(args);
            if(!Kernel::IsSupportedArgument(kargs))
                return -1.0f;

            const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
            const dim3 blocks = Kernel::BlockSize();
            // EightWaves pipeline requires no-packed-fp32-ops to co-execute matrix ops.
            using KAttr = ck_tile::kernel_attr<IsEightWaves>;
            return ck_tile::launch_kernel(
                s, ck_tile::make_kernel<kBlockPerCu, KAttr>(Kernel{{}}, grids, blocks, 0, kargs));
        }};

        return BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
    }}
}};

using SelectedKernel = {struct};

}} // namespace {ns}

#ifdef CK_TILE_SINGLE_KERNEL_INCLUDE
using SelectedKernel = {ns}::{struct};
constexpr const char* KERNEL_NAME = {ns}::KERNEL_NAME;
using ADataType      = {ck_a};
using BDataType      = {ck_b};
using CDataType      = {ck_c};
using QDataType      = {ck_q};
using AccDataType    = {ck_acc};
using AQuantGroupSize = {ns}::AQuantGroupSize;
using BQuantGroupSize = {ns}::BQuantGroupSize;
#endif // CK_TILE_SINGLE_KERNEL_INCLUDE
"""


# =============================================================================
# Config sweep
# =============================================================================


def _default_config() -> dict:
    """Default sweep config: fp8 ABQuant, compv3 pipeline, 128x128x128 tile (non-gfx950)."""
    return {
        "variant_keys": ["fp8", "bf8"],
        "layouts": ["rcr"],
        "pipeline": "compv3",
        "epilogue": "cshuffle",
        "scheduler": "intrawave",
        "tile_configs": [
            # GemmConfigABQuantPrefill<fp8_t>: M=128, N=128, K=128, Warp 1x4x1
            {"tile_m": 128, "tile_n": 128, "tile_k": 128,
             "warp_m": 1, "warp_n": 4, "warp_k": 1,
             "warp_tile_m": 16, "warp_tile_n": 16, "warp_tile_k": 16},
        ],
        "aquant_groups": [
            {"aquant_group_m": 1, "aquant_group_n": 1, "aquant_group_k": 128},
        ],
        "bquant_groups": [
            {"bquant_group_m": 1, "bquant_group_n": 1, "bquant_group_k": 128},
        ],
        "preshuffle_b": False,
        "preshuffle_aq": False,
        "preshuffle_bq": False,
        "transpose_c": False,
        "double_smem_buffer": False,
        "pad_m": False,
        "pad_n": False,
        "pad_k": False,
        "block_size": 256,
        "k_block_per_cu": 1,
    }


def _build_specs(config: dict) -> List[ABQuantKernelSpec]:
    specs = []
    pipeline   = config.get("pipeline", "compv3")
    epilogue   = config.get("epilogue", "cshuffle")
    scheduler  = config.get("scheduler", "intrawave")
    pad_m      = config.get("pad_m", False)
    pad_n      = config.get("pad_n", False)
    pad_k      = config.get("pad_k", False)
    block_size         = config.get("block_size", 256)
    k_block_per_cu     = config.get("k_block_per_cu", 1)
    double_smem_buffer = config.get("double_smem_buffer", False)
    preshuffle_b  = config.get("preshuffle_b", False)
    preshuffle_aq = config.get("preshuffle_aq", False)
    preshuffle_bq = config.get("preshuffle_bq", False)
    transpose_c   = config.get("transpose_c", False)

    default_aquant = [{"aquant_group_m": 1, "aquant_group_n": 1, "aquant_group_k": 128}]
    default_bquant = [{"bquant_group_m": 1, "bquant_group_n": 1, "bquant_group_k": 128}]

    for variant_key, layout, tile_dict, aqg, bqg in itertools.product(
        config.get("variant_keys", ["fp8"]),
        config.get("layouts", ["rcr"]),
        config.get("tile_configs", []),
        config.get("aquant_groups", default_aquant),
        config.get("bquant_groups", default_bquant),
    ):
        if variant_key not in ABQUANT_VARIANTS:
            log.warning("Unknown variant_key %s — skipping", variant_key)
            continue
        if pipeline not in ABQUANT_PIPELINE_MAP:
            log.warning("Unsupported pipeline %s — skipping", pipeline)
            continue

        aqk = aqg.get("aquant_group_k", 128)
        bqk = bqg.get("bquant_group_k", 128)
        if aqk != bqk:
            log.warning(
                "ABQuant kK mismatch: aquant_group_k=%d != bquant_group_k=%d — skipping",
                aqk, bqk,
            )
            continue

        tile = ABQuantTileConfig(
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

        try:
            specs.append(ABQuantKernelSpec(
                variant_key=variant_key,
                layout=layout,
                pipeline=pipeline,
                epilogue=epilogue,
                scheduler=scheduler,
                tile=tile,
                aquant_group_m=aqg.get("aquant_group_m", 1),
                aquant_group_n=aqg.get("aquant_group_n", 1),
                aquant_group_k=aqk,
                bquant_group_m=bqg.get("bquant_group_m", 1),
                bquant_group_n=bqg.get("bquant_group_n", 1),
                bquant_group_k=bqk,
                preshuffle_b=preshuffle_b,
                preshuffle_aq=preshuffle_aq,
                preshuffle_bq=preshuffle_bq,
                transpose_c=transpose_c,
                double_smem_buffer=double_smem_buffer,
                pad_m=pad_m,
                pad_n=pad_n,
                pad_k=pad_k,
                block_size=block_size,
                k_block_per_cu=k_block_per_cu,
            ))
        except ValueError as e:
            log.warning("Skipping invalid spec: %s", e)

    return specs

def generate_kernels(
    output_dir: Path,
    config: Optional[dict] = None,
    parallel: bool = True,
) -> List[Path]:
    """Generate all ABQuant kernel headers into output_dir. Returns list of generated .hpp paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or _default_config()
    specs = _build_specs(cfg)

    if not specs:
        log.warning("No kernel specs produced — check variant_keys, tile_configs, and kK equality")
        return []

    log.info("Generating %d ABQuant kernel headers into %s", len(specs), output_dir)

    gen = ABQuantKernelHeaderGenerator()
    generated: List[Path] = []

    def _generate_one(spec: ABQuantKernelSpec) -> Path:
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
        description="ABQuantGrouped GEMM kernel header generator"
    )
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write generated .hpp files")
    parser.add_argument("--config", type=Path,
                        help="JSON config file")
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
