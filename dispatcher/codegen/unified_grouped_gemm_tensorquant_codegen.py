#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
GroupedGemm TensorQuant Code Generator

Generates one .hpp per kernel config for the dispatcher's ctypes path.
Each header defines a SelectedKernel struct with a static launch() method
taking vector<QuantGroupedGemmHostArgs> — compiled per-kernel via force-include:

    hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE grouped_gemm_tensorquant_ctypes_lib.cpp

TensorQuant: A and B each have a single scalar scale (one scale per tensor).
AQDataType=BQDataType=float; kernel uses QuantType::TensorQuant.

Naming convention (byte-exact with TensorQuantKernelConfig.name in grouped_gemm_tensorquant_utils.py):
    grouped_gemm_tensorquant_{dtype}_{layout}_{pipeline}_{epilogue}_{scheduler}_
    {padm}_{padn}_{padk}_{persistent}_{TileM}x{TileN}x{TileK}_{WarpM}x{WarpN}x{WarpK}_{WtM}x{WtN}x{WtK}

Reference:
    tile_engine/ops/gemm/grouped_gemm_quant/grouped_gemm_tensorquant/grouped_gemm_tensorquant_instance_builder.py
"""

import argparse
import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from codegen_common import (
    ROWCOL_TENSOR_QUANT_BASE_PIPELINE_MAP,
    ROWCOL_TENSOR_QUANT_DEFAULT_TILE,
    ROWCOL_TENSOR_QUANT_DEFAULT_TRAITS,
    ROWCOL_TENSOR_QUANT_EPILOGUE_MAP,
    ROWCOL_TENSOR_QUANT_PIPELINE_MAP,
    ROWCOL_TENSOR_QUANT_SUPPORTED_LAYOUTS,
    make_tensorquant_kernel_name,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Dtype variant definitions
# =============================================================================

TENSORQUANT_VARIANTS: Dict[str, Dict[str, str]] = {
    "fp8": {
        "ck_a":   "ck_tile::fp8_t",
        "ck_b":   "ck_tile::fp8_t",
        "ck_c":   "ck_tile::half_t",
        "ck_aq":  "float",
        "ck_bq":  "float",
        "ck_acc": "float",
    },
    "bf8": {
        "ck_a":   "ck_tile::bf8_t",
        "ck_b":   "ck_tile::bf8_t",
        "ck_c":   "ck_tile::half_t",
        "ck_aq":  "float",
        "ck_bq":  "float",
        "ck_acc": "float",
    },
}

TENSORQUANT_LAYOUT_TO_CK = {
    "r": "ck_tile::tensor_layout::gemm::RowMajor",
    "c": "ck_tile::tensor_layout::gemm::ColumnMajor",
}

TENSORQUANT_SCHEDULER_TO_CK = {
    "intrawave": "ck_tile::GemmPipelineScheduler::Intrawave",
    "interwave": "ck_tile::GemmPipelineScheduler::Interwave",
    "default":   "ck_tile::GemmPipelineScheduler::Default",
}

# TensorQuant currently supports only the CompV3 pipeline with a CShuffle epilogue.
# These maps make the config keys load-bearing: the emitted C++ is interpolated from
# them, and _build_specs rejects any key that is absent. Without that, a config
# naming an unsupported pipeline would produce a header *named* for it while
# containing a CompV3 kernel -- silently mislabelled for any name-keyed autotuner.
TENSORQUANT_PIPELINE_MAP      = dict(ROWCOL_TENSOR_QUANT_PIPELINE_MAP)
TENSORQUANT_BASE_PIPELINE_MAP = dict(ROWCOL_TENSOR_QUANT_BASE_PIPELINE_MAP)
TENSORQUANT_EPILOGUE_MAP      = dict(ROWCOL_TENSOR_QUANT_EPILOGUE_MAP)
TENSORQUANT_SUPPORTED_LAYOUTS = ROWCOL_TENSOR_QUANT_SUPPORTED_LAYOUTS


# =============================================================================
# Kernel name helper (byte-exact with instance builder naming)
# =============================================================================
#
# make_tensorquant_kernel_name lives in codegen_common alongside the aquant/bquant/
# abquant builders and is re-exported here so existing
# `from unified_grouped_gemm_tensorquant_codegen import make_tensorquant_kernel_name`
# imports keep working.
__all__ = ["make_tensorquant_kernel_name", "generate_kernels", "main"]


# =============================================================================
# Configuration dataclasses
# =============================================================================


@dataclass
class TensorQuantTileConfig:
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
class TensorQuantKernelSpec:
    """Complete specification for one TensorQuant kernel."""

    dtype: str          # "fp8" or "bf8"
    layout: str         # "rcr"
    pipeline: str       # "compv3"
    epilogue: str       # "cshuffle"
    scheduler: str      # "intrawave"
    pad_m: bool
    pad_n: bool
    pad_k: bool
    persistent: bool
    tile: TensorQuantTileConfig
    block_size: int = 256
    k_block_per_cu: int = 1

    @property
    def name(self) -> str:
        t = self.tile
        return make_tensorquant_kernel_name(
            dtype=self.dtype,
            layout=self.layout,
            pipeline=self.pipeline,
            epilogue=self.epilogue,
            scheduler=self.scheduler,
            pad_m=self.pad_m,
            pad_n=self.pad_n,
            pad_k=self.pad_k,
            persistent=self.persistent,
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
        variant = TENSORQUANT_VARIANTS[spec.dtype]
        t = spec.tile
        ns = "ns_" + spec.name
        struct = "Kernel_" + spec.name

        ck_a   = variant["ck_a"]
        ck_b   = variant["ck_b"]
        ck_c   = variant["ck_c"]
        ck_aq  = variant["ck_aq"]
        ck_bq  = variant["ck_bq"]
        ck_acc = variant["ck_acc"]

        layout_a_ck  = TENSORQUANT_LAYOUT_TO_CK[spec.layout[0]]
        layout_b_ck  = TENSORQUANT_LAYOUT_TO_CK[spec.layout[1]]
        layout_c_ck  = TENSORQUANT_LAYOUT_TO_CK[spec.layout[2]]
        # AQ layout is RowMajor, BQ layout is ColumnMajor (follows B convention; nominal for single-scalar quant)
        layout_aq_ck = TENSORQUANT_LAYOUT_TO_CK["r"]
        layout_bq_ck = TENSORQUANT_LAYOUT_TO_CK["c"]

        scheduler_ck = TENSORQUANT_SCHEDULER_TO_CK[spec.scheduler]

        # Interpolated rather than hardwired, so the kernel name and the emitted code
        # cannot disagree. _build_specs guarantees these keys exist.
        pipeline_ck      = TENSORQUANT_PIPELINE_MAP[spec.pipeline]
        base_pipeline_ck = TENSORQUANT_BASE_PIPELINE_MAP[spec.pipeline]
        epilogue_ck      = TENSORQUANT_EPILOGUE_MAP[spec.epilogue]

        pad_m      = str(spec.pad_m).lower()
        pad_n      = str(spec.pad_n).lower()
        pad_k      = str(spec.pad_k).lower()
        persistent = str(spec.persistent).lower()

        grid_size_expr = (
            "Kernel::MaxOccupancyGridSize(stream)"
            if spec.persistent
            else "Kernel::GridSize(gemm_descs)"
        )

        return f"""\
// SPDX-License-Identifier: MIT
// Auto-generated TensorQuant Grouped GEMM kernel header.
// DO NOT EDIT — regenerate via unified_grouped_gemm_tensorquant_codegen.py
#pragma once

#include <cstdint>
#include <vector>
#include <hip/hip_runtime.h>
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm_quant.hpp"
#include "ck_tile/ops/gemm_quant/kernel/grouped_gemm_quant_kernel.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"

namespace {ns} {{

constexpr const char* KERNEL_NAME = "{spec.name}";

using ADataType   = {ck_a};
using BDataType   = {ck_b};
using CDataType   = {ck_c};
using AQDataType  = {ck_aq};
using BQDataType  = {ck_bq};
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
    using AQDataType  = {ns}::AQDataType;
    using BQDataType  = {ns}::BQDataType;
    using AccDataType = {ns}::AccDataType;

    static constexpr ck_tile::index_t TileM          = {t.tile_m};
    static constexpr ck_tile::index_t TileN          = {t.tile_n};
    static constexpr ck_tile::index_t TileK          = {t.tile_k};
    static constexpr ck_tile::index_t WarpPerBlock_M = {t.warp_m};
    static constexpr ck_tile::index_t WarpPerBlock_N = {t.warp_n};
    static constexpr ck_tile::index_t WarpPerBlock_K = {t.warp_k};
    static constexpr ck_tile::index_t WarpTileM      = {t.warp_tile_m};
    static constexpr ck_tile::index_t WarpTileN      = {t.warp_tile_n};
    static constexpr ck_tile::index_t WarpTileK      = {t.warp_tile_k};
    // Informational only: the launch below uses Kernel::BlockSize(), which the
    // pipeline derives from the warp counts. Changing the `block_size` config key
    // changes this constant but not the launch geometry.
    static constexpr ck_tile::index_t BlockSize       = {spec.block_size};
    static constexpr int               kBlockPerCu    = {spec.k_block_per_cu};

    static constexpr bool kPadM               = {pad_m};
    static constexpr bool kPadN               = {pad_n};
    static constexpr bool kPadK               = {pad_k};
    static constexpr bool TransposeC          = false;
    static constexpr bool DoubleSmemBuffer    = false;
    static constexpr bool APreshuffleQuant    = false;
    static constexpr bool BPreshuffleQuant    = false;
    static constexpr bool PreshuffleB         = false;
    static constexpr bool UsePersistentKernel = {persistent};

    // TileGemmShape's trailing template parameters are PermuteA_ / PermuteB_. This
    // bridge does not use preshuffled operands, so both are false.
    static constexpr bool PermuteA = false;
    static constexpr bool PermuteB = false;

    using TileShape = ck_tile::TileGemmShape<
        ck_tile::sequence<TileM, TileN, TileK>,
        ck_tile::sequence<WarpPerBlock_M, WarpPerBlock_N, WarpPerBlock_K>,
        ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>,
        PermuteA, PermuteB>;

    // GemmSpatiallyLocalTilePartitioner groups workgroups to improve cache reuse; the
    // two integers are GroupNum (number of big groups) and M01 (groups in the M dim
    // within a spatially local WGP). The values below are the gfx94x-tuned defaults
    // used by the tile_engine instance builder and the 17_grouped_gemm examples, where
    // they appear as TileParitionerGroupNum / TileParitionerM01.
    static constexpr ck_tile::index_t TilePartitionerGroupNum = 8;
    static constexpr ck_tile::index_t TilePartitionerM01      = 4;

    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<
        TileShape, TilePartitionerGroupNum, TilePartitionerM01>;

    using GemmQuantTraits = ck_tile::TileGemmQuantTraits<
        kPadM, kPadN, kPadK,
        APreshuffleQuant,
        BPreshuffleQuant,
        PreshuffleB,
        {ns}::ALayout, {ns}::BLayout, {ns}::CLayout,
        ck_tile::QuantType::TensorQuant,
        {ns}::AQLayout, {ns}::BQLayout,
        TransposeC,
        DoubleSmemBuffer,
        UsePersistentKernel>;

    using Traits = ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, {ns}::ALayout, {ns}::BLayout, {ns}::CLayout>;
    using GemmPipelineProblem = ck_tile::GemmPipelineProblem<
        ADataType, BDataType, AccDataType, TileShape, Traits>;
    using BaseGemmPipeline = {base_pipeline_ck}<GemmPipelineProblem>;

    // preprocess runs once before every kernel invocation, including each iteration of
    // the timing loop (see ck_tile::launch_kernel_time_mask). Callers must use it to
    // re-zero C whenever k_batch > 1: split-K selects the atomic_add epilogue, so a
    // C that is zeroed only once ends up holding the sum over cold_niters + nrepeat
    // launches. The overload below supplies a no-op and is safe only for k_batch == 1,
    // where the epilogue is `set` and repeated launches are idempotent.
    template <typename PreprocessFunc>
    static float launch(const std::vector<ck_tile::QuantGroupedGemmHostArgs>& gemm_descs,
                        const ck_tile::stream_config& stream,
                        void* kargs_ptr,
                        PreprocessFunc preprocess)
    {{
        constexpr auto scheduler = {scheduler_ck};

        const ck_tile::index_t k_grain = gemm_descs[0].k_batch * TileShape::kK;
        const ck_tile::index_t K_split = (gemm_descs[0].K + k_grain - 1) / k_grain * TileShape::kK;
        const ck_tile::index_t num_loop = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        float ave_time{{0}};

        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {{
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v  = tail_number_.value;

            using QuantGemmProblem = ck_tile::GemmRowColTensorQuantPipelineProblem<
                ADataType, BDataType, AccDataType, AccDataType,
                TileShape, GemmQuantTraits,
                TransposeC, BDataType,
                scheduler,
                has_hot_loop_v,
                tail_number_v>;

            using GemmPipeline = {pipeline_ck}<QuantGemmProblem>;

            using GemmEpilogue = {epilogue_ck}<
                ck_tile::CShuffleEpilogueProblem<
                    ADataType, BDataType, ck_tile::tuple<>,
                    AccDataType, CDataType, ck_tile::tuple<>,
                    {ns}::CLayout, ck_tile::element_wise::PassThrough,
                    TilePartitioner::MPerBlock,
                    TilePartitioner::NPerBlock,
                    WarpPerBlock_M, WarpPerBlock_N,
                    WarpTileM, WarpTileN, WarpTileK,
                    QuantGemmProblem::TransposeC>>;

            using Kernel = ck_tile::QuantGroupedGemmKernel<
                TilePartitioner, GemmPipeline, GemmEpilogue,
                GemmQuantTraits::kQuantType>;

            auto kargs = Kernel::MakeKargs(gemm_descs);
            if(!Kernel::IsSupportedArgument(kargs)) {{
                return -1.0f;
            }}

            const dim3 grids  = {grid_size_expr};
            const dim3 blocks = Kernel::BlockSize();

            HIP_CHECK_ERROR(hipMemcpyWithStream(kargs_ptr,
                                                kargs.data(),
                                                kargs.size() * sizeof(ck_tile::QuantGemmTransKernelArg),
                                                hipMemcpyHostToDevice,
                                                stream.stream_id_));

            constexpr int kBlockPerCu_ = kBlockPerCu;
            return ave_time = ck_tile::launch_kernel_time_mask(
                stream,
                preprocess,
                ck_tile::make_kernel<kBlockPerCu_>(
                    Kernel{{}},
                    grids,
                    blocks,
                    0,
                    ck_tile::cast_pointer_to_constant_address_space(kargs_ptr),
                    gemm_descs.size()));
        }};

        return ave_time = BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
    }}

    // Convenience overload for k_batch == 1. See the note on preprocess above.
    static float launch(const std::vector<ck_tile::QuantGroupedGemmHostArgs>& gemm_descs,
                        const ck_tile::stream_config& stream,
                        void* kargs_ptr)
    {{
        return launch(gemm_descs, stream, kargs_ptr, []() {{}});
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
using AQDataType  = {ck_aq};
using BQDataType  = {ck_bq};
using AccDataType = {ck_acc};
#endif // CK_TILE_SINGLE_KERNEL_INCLUDE
"""


# =============================================================================
# Config sweep
# =============================================================================


def _default_config() -> dict:
    # Traits and tile come from codegen_common so this default and the runtime
    # default_{fp8,bf8}_config() in grouped_gemm_tensorquant_utils.py cannot drift.
    return {
        "dtypes": ["fp8", "bf8"],
        "layouts": list(TENSORQUANT_SUPPORTED_LAYOUTS),
        **ROWCOL_TENSOR_QUANT_DEFAULT_TRAITS,
        "tile_configs": [dict(ROWCOL_TENSOR_QUANT_DEFAULT_TILE)],
    }


def _build_specs(config: dict) -> List[TensorQuantKernelSpec]:
    specs = []
    defaults   = ROWCOL_TENSOR_QUANT_DEFAULT_TRAITS
    pipeline   = config.get("pipeline", defaults["pipeline"])
    epilogue   = config.get("epilogue", defaults["epilogue"])
    scheduler  = config.get("scheduler", defaults["scheduler"])
    pad_m      = config.get("pad_m", defaults["pad_m"])
    pad_n      = config.get("pad_n", defaults["pad_n"])
    pad_k      = config.get("pad_k", defaults["pad_k"])
    persistent = config.get("persistent", defaults["persistent"])
    block_size     = config.get("block_size", defaults["block_size"])
    k_block_per_cu = config.get("k_block_per_cu", defaults["k_block_per_cu"])

    # Reject unsupported pipeline/epilogue/scheduler up front rather than emitting a
    # header whose name advertises something the generated code does not implement.
    if pipeline not in TENSORQUANT_PIPELINE_MAP:
        log.warning(
            "Unsupported pipeline '%s' (supported: %s) — no kernels generated",
            pipeline, ", ".join(sorted(TENSORQUANT_PIPELINE_MAP)),
        )
        return []
    if epilogue not in TENSORQUANT_EPILOGUE_MAP:
        log.warning(
            "Unsupported epilogue '%s' (supported: %s) — no kernels generated",
            epilogue, ", ".join(sorted(TENSORQUANT_EPILOGUE_MAP)),
        )
        return []
    if scheduler not in TENSORQUANT_SCHEDULER_TO_CK:
        log.warning(
            "Unsupported scheduler '%s' (supported: %s) — no kernels generated",
            scheduler, ", ".join(sorted(TENSORQUANT_SCHEDULER_TO_CK)),
        )
        return []

    for dtype, layout, tile_dict in itertools.product(
        config.get("dtypes", ["fp8"]),
        config.get("layouts", list(TENSORQUANT_SUPPORTED_LAYOUTS)),
        config.get("tile_configs", []),
    ):
        if dtype not in TENSORQUANT_VARIANTS:
            log.warning("Unknown dtype %s — skipping", dtype)
            continue

        # A non-rcr layout would flip BLayout in the generated header while the ctypes
        # bridge kept requiring stride_A == K, stride_B == K, stride_C == N, so the
        # kernel would build but every call would be rejected at runtime.
        if layout not in TENSORQUANT_SUPPORTED_LAYOUTS:
            log.warning(
                "Unsupported layout '%s' (supported: %s) — skipping",
                layout, ", ".join(TENSORQUANT_SUPPORTED_LAYOUTS),
            )
            continue

        tile = TensorQuantTileConfig(
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

        specs.append(TensorQuantKernelSpec(
            dtype=dtype,
            layout=layout,
            pipeline=pipeline,
            epilogue=epilogue,
            scheduler=scheduler,
            pad_m=pad_m,
            pad_n=pad_n,
            pad_k=pad_k,
            persistent=persistent,
            tile=tile,
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
    """Generate all TensorQuant kernel headers into output_dir. Returns list of generated .hpp paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or _default_config()
    specs = _build_specs(cfg)

    if not specs:
        log.warning("No kernel specs produced from config — check dtypes and tile_configs")
        return []

    log.info("Generating %d TensorQuant kernel headers into %s", len(specs), output_dir)

    gen = TensorQuantKernelHeaderGenerator()
    generated: List[Path] = []

    def _generate_one(spec: TensorQuantKernelSpec) -> Path:
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
        description="TensorQuant Grouped GEMM kernel header generator"
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
