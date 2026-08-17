#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Batched-contraction code generator (TileEngine -> Dispatcher bridge).

Generates one .hpp per kernel config for the dispatcher's ctypes path. Each header
defines a SelectedKernel struct with a static launch() taking
``ck_tile::BatchedContractionHostArgs<NUM_D_TENSORS>`` -- compiled per-kernel via
force-include:

    hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE batched_contraction_ctypes_lib.cpp

The generated instance mirrors the Old-TE
``tile_engine/ops/gemm/batched_contraction/batched_contraction_instance_builder.py``
type assembly exactly (BatchedContractionProblem / BatchedContractionKernel,
UniversalGemmPipeline, CShuffle/Default epilogue), so bridge and Old-TE build the
same kernel.

Naming (byte-exact with BatchedContractionKernelConfig.name in
batched_contraction_utils.py, both delegate to make_batched_contraction_kernel_name):

    batched_contraction_{dtype}_{layout}_{pipeline}_{epilogue}_{scheduler}_
    {PadM}_{PadN}_{PadK}_{Persistent}_
    {TileM}x{TileN}x{TileK}_{WarpM}x{WarpN}x{WarpK}_{WtM}x{WtN}x{WtK}_
    g{G}m{M}n{N}k{K}[_d{numD}][_{elementwise}]
"""

import argparse
import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Type / map tables (mirror gemm_instance_builder.py for batched_contraction)
# =============================================================================

DTYPE_TO_CK = {
    "fp16": "ck_tile::half_t",
    "bf16": "ck_tile::bf16_t",
    "fp32": "float",
}

LAYOUT_TO_CK = {
    "r": "ck_tile::tensor_layout::gemm::RowMajor",
    "c": "ck_tile::tensor_layout::gemm::ColumnMajor",
}

PIPELINE_IMPL_MAP = {
    "mem": "ck_tile::GemmPipelineAgBgCrMem",
    "compv3": "ck_tile::GemmPipelineAgBgCrCompV3",
    "compv4": "ck_tile::GemmPipelineAgBgCrCompV4",
}
BASE_PIPELINE_MAP = {
    "mem": "ck_tile::BaseGemmPipelineAgBgCrMem",
    "compv3": "ck_tile::BaseGemmPipelineAgBgCrCompV3",
    "compv4": "ck_tile::BaseGemmPipelineAgBgCrCompV4",
}
SCHEDULER_TO_CK = {
    "intrawave": "ck_tile::GemmPipelineScheduler::Intrawave",
    "interwave": "ck_tile::GemmPipelineScheduler::Interwave",
    "default": "ck_tile::GemmPipelineScheduler::Default",
}
# Old-TE argparse normalizes Add/Mul -> MultiDAdd/MultiDMultiply.
ELEMENTWISE_TO_CK = {
    "PassThrough": "PassThrough",
    "MultiDAdd": "MultiDAdd",
    "MultiDMultiply": "MultiDMultiply",
}

# Epilogue variants the generator can emit: "cshuffle" -> CShuffleEpilogue,
# "default" -> DefaultGemm2DEpilogue. Any other value is a config error (e.g. a
# JSON typo) and must be rejected rather than silently emitting the Default path.
VALID_EPILOGUES = ("cshuffle", "default")

DOUBLE_SMEM_PIPELINES = {"compv4", "preshufflev2", "comp_async"}


# =============================================================================
# Shared kernel-name construction (single source of truth)
# =============================================================================


def _cap(b: bool) -> str:
    return "True" if b else "False"


def make_batched_contraction_kernel_name(
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
    num_dim_g: int,
    num_dim_m: int,
    num_dim_n: int,
    num_dim_k: int,
    num_d_tensors: int = 0,
    elementwise: str = "PassThrough",
    k_block_per_cu: int = 1,
) -> str:
    """Canonical kernel name. BatchedContractionKernelConfig.name and the codegen
    both call this so the compiled .so and the Python config always agree."""
    parts = [
        "batched_contraction",
        dtype,
        layout,
        pipeline,
        epilogue,
        scheduler,
        _cap(pad_m),
        _cap(pad_n),
        _cap(pad_k),
        _cap(persistent),
        f"{tile_m}x{tile_n}x{tile_k}",
        f"{warp_m}x{warp_n}x{warp_k}",
        f"{warp_tile_m}x{warp_tile_n}x{warp_tile_k}",
        f"g{num_dim_g}m{num_dim_m}n{num_dim_n}k{num_dim_k}",
    ]
    if num_d_tensors > 0:
        parts.append(f"d{num_d_tensors}")
    if elementwise != "PassThrough":
        parts.append(elementwise)
    # k_block_per_cu changes the launched kernel; encode it so distinct values do
    # not collide on one name. Omit for the default (1) to keep names stable.
    if k_block_per_cu != 1:
        parts.append(f"kbpc{k_block_per_cu}")
    return "_".join(parts)


# =============================================================================
# Config dataclasses
# =============================================================================


@dataclass
class BCTileConfig:
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
        if min(self.tile_m, self.tile_n, self.tile_k) <= 0:
            return False
        return (
            self.tile_m % (self.warp_m * self.warp_tile_m) == 0
            and self.tile_n % (self.warp_n * self.warp_tile_n) == 0
            and self.tile_k % (self.warp_k * self.warp_tile_k) == 0
        )


@dataclass
class BCKernelSpec:
    dtype: str
    layout: str  # 3-char abe
    pipeline: str
    epilogue: str
    scheduler: str
    tile: BCTileConfig
    pad_m: bool = False
    pad_n: bool = False
    pad_k: bool = False
    persistent: bool = False
    num_dim_g: int = 1
    num_dim_m: int = 1
    num_dim_n: int = 1
    num_dim_k: int = 1
    num_d_tensors: int = 0
    elementwise: str = "PassThrough"
    block_size: int = 256
    k_block_per_cu: int = 1

    @property
    def name(self) -> str:
        t = self.tile
        return make_batched_contraction_kernel_name(
            dtype=self.dtype, layout=self.layout, pipeline=self.pipeline,
            epilogue=self.epilogue, scheduler=self.scheduler,
            pad_m=self.pad_m, pad_n=self.pad_n, pad_k=self.pad_k, persistent=self.persistent,
            tile_m=t.tile_m, tile_n=t.tile_n, tile_k=t.tile_k,
            warp_m=t.warp_m, warp_n=t.warp_n, warp_k=t.warp_k,
            warp_tile_m=t.warp_tile_m, warp_tile_n=t.warp_tile_n, warp_tile_k=t.warp_tile_k,
            num_dim_g=self.num_dim_g, num_dim_m=self.num_dim_m,
            num_dim_n=self.num_dim_n, num_dim_k=self.num_dim_k,
            num_d_tensors=self.num_d_tensors, elementwise=self.elementwise,
            k_block_per_cu=self.k_block_per_cu,
        )


# =============================================================================
# Header generator
# =============================================================================


class BCHeaderGenerator:
    def generate(self, spec: BCKernelSpec) -> str:
        if spec.dtype not in DTYPE_TO_CK:
            raise ValueError(f"unsupported dtype {spec.dtype}")
        if len(spec.layout) != 3 or any(c not in LAYOUT_TO_CK for c in spec.layout):
            raise ValueError(f"layout must be 3 chars of r/c, got {spec.layout}")
        if spec.pipeline not in PIPELINE_IMPL_MAP:
            raise ValueError(f"unsupported pipeline {spec.pipeline}")
        if spec.epilogue not in VALID_EPILOGUES:
            raise ValueError(
                f"unsupported epilogue {spec.epilogue!r}; expected one of {VALID_EPILOGUES}"
            )
        if spec.elementwise not in ELEMENTWISE_TO_CK:
            raise ValueError(
                f"unsupported elementwise {spec.elementwise!r}; "
                f"expected one of {tuple(ELEMENTWISE_TO_CK)}"
            )

        t = spec.tile
        ns = "ns_" + spec.name
        struct = "Kernel_" + spec.name

        ck_ab = DTYPE_TO_CK[spec.dtype]
        # e-type: fp8/bf8 -> fp16 in Old-TE, but bridge only supports fp16/bf16/fp32
        ck_e = DTYPE_TO_CK[spec.dtype]
        layout_a = LAYOUT_TO_CK[spec.layout[0]]
        layout_b = LAYOUT_TO_CK[spec.layout[1]]
        layout_e = LAYOUT_TO_CK[spec.layout[2]]

        pipeline_impl = PIPELINE_IMPL_MAP[spec.pipeline]
        base_pipeline = BASE_PIPELINE_MAP[spec.pipeline]
        scheduler_ck = SCHEDULER_TO_CK[spec.scheduler]
        elementwise_ck = ELEMENTWISE_TO_CK[spec.elementwise]
        double_smem = "true" if spec.pipeline in DOUBLE_SMEM_PIPELINES else "false"

        if spec.num_d_tensors == 0:
            ds_dtype = "ck_tile::tuple<>"
            ds_layout = "ck_tile::tuple<>"
        else:
            ds_dtype = "ck_tile::tuple<" + ", ".join(["DBaseDataType"] * spec.num_d_tensors) + ">"
            ds_layout = "ck_tile::tuple<" + ", ".join(["ELayout"] * spec.num_d_tensors) + ">"

        if spec.epilogue == "cshuffle":
            epilogue_block = f"""\
        using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<
            ADataType, BDataType, DsDataType, AccDataType, EDataType,
            DsLayout, ELayout, CDEElementWise,
            TileM, TileN, WarpPerBlock_M, WarpPerBlock_N,
            WarpTileM, WarpTileN, WarpTileK, TransposeC>;
        using GemmEpilogue = ck_tile::CShuffleEpilogue<EpilogueProblem>;"""
        else:
            epilogue_block = f"""\
        using EpilogueProblem = ck_tile::DefaultGemm2DEpilogueProblem<
            ADataType, BDataType, DsDataType, AccDataType, EDataType,
            DsLayout, ELayout, CDEElementWise,
            TileM, TileN, kPadM, kPadN,
            WarpTileM, WarpTileN, WarpTileK, TransposeC>;
        using GemmEpilogue = ck_tile::DefaultGemm2DEpilogue<EpilogueProblem>;"""

        return f"""\
// SPDX-License-Identifier: MIT
// Auto-generated batched-contraction kernel header.
// DO NOT EDIT -- regenerate via unified_batched_contraction_codegen.py
#pragma once

#include <array>
#include <vector>
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/batched_contraction.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"

namespace {ns} {{

constexpr const char* KERNEL_NAME = "{spec.name}";

using ADataType     = {ck_ab};
using BDataType     = {ck_ab};
using AccDataType   = float;
using EDataType     = {ck_e};
using DBaseDataType = {ck_ab};
using DsDataType    = {ds_dtype};

using ALayout  = {layout_a};
using BLayout  = {layout_b};
using ELayout  = {layout_e};
using DsLayout = {ds_layout};

using CDEElementWise = ck_tile::element_wise::{elementwise_ck};

static constexpr ck_tile::index_t NUM_D_TENSORS = {spec.num_d_tensors};
static constexpr ck_tile::index_t NUM_DIM_G     = {spec.num_dim_g};
static constexpr ck_tile::index_t NUM_DIM_M     = {spec.num_dim_m};
static constexpr ck_tile::index_t NUM_DIM_N     = {spec.num_dim_n};
static constexpr ck_tile::index_t NUM_DIM_K     = {spec.num_dim_k};

struct {struct} {{
    using ADataType   = {ns}::ADataType;
    using BDataType   = {ns}::BDataType;
    using AccDataType = {ns}::AccDataType;
    using EDataType   = {ns}::EDataType;

    static constexpr ck_tile::index_t TileM = {t.tile_m};
    static constexpr ck_tile::index_t TileN = {t.tile_n};
    static constexpr ck_tile::index_t TileK = {t.tile_k};
    static constexpr ck_tile::index_t WarpPerBlock_M = {t.warp_m};
    static constexpr ck_tile::index_t WarpPerBlock_N = {t.warp_n};
    static constexpr ck_tile::index_t WarpPerBlock_K = {t.warp_k};
    static constexpr ck_tile::index_t WarpTileM = {t.warp_tile_m};
    static constexpr ck_tile::index_t WarpTileN = {t.warp_tile_n};
    static constexpr ck_tile::index_t WarpTileK = {t.warp_tile_k};
    static constexpr ck_tile::index_t BlockSize = {spec.block_size};

    static constexpr bool kPadM = {str(spec.pad_m).lower()};
    static constexpr bool kPadN = {str(spec.pad_n).lower()};
    static constexpr bool kPadK = {str(spec.pad_k).lower()};
    static constexpr bool TransposeC = false;
    static constexpr bool DoubleSmemBuffer = {double_smem};

    using TileShape = ck_tile::TileGemmShape<
        ck_tile::sequence<TileM, TileN, TileK>,
        ck_tile::sequence<WarpPerBlock_M, WarpPerBlock_N, WarpPerBlock_K>,
        ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>>;

    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<TileShape, 8, 4>;

    using Traits = ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, ELayout>;

    using GemmPipelineProblem = ck_tile::GemmPipelineProblem<
        ADataType, BDataType, AccDataType, TileShape, Traits>;

    using BaseGemmPipeline = {base_pipeline}<GemmPipelineProblem>;

    static float launch(const ck_tile::BatchedContractionHostArgs<NUM_D_TENSORS>& args,
                        const ck_tile::stream_config& stream)
    {{
        constexpr auto scheduler = {scheduler_ck};

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<
            ADataType, BDataType, AccDataType, TileShape,
            ck_tile::TileGemmUniversalTraits<kPadM, kPadN, kPadK, DoubleSmemBuffer,
                                             ALayout, BLayout, ELayout, TransposeC>,
            scheduler>;

        using GemmPipeline = {pipeline_impl}<UniversalGemmProblem>;

{epilogue_block}

        using ContractionProblem = ck_tile::BatchedContractionProblem<
            ADataType, BDataType, DsDataType, EDataType,
            NUM_DIM_G, NUM_DIM_M, NUM_DIM_N, NUM_DIM_K, NUM_D_TENSORS>;

        using ContractionKernel = ck_tile::BatchedContractionKernel<
            ContractionProblem, TilePartitioner, GemmPipeline, GemmEpilogue>;

        auto kargs = ContractionKernel::MakeKernelArgs(args);
        if(!ContractionKernel::IsSupportedArguments(kargs))
            return -1.0f;

        const dim3 grids  = ContractionKernel::GridSize(kargs);
        const dim3 blocks = ContractionKernel::GetBlockSize();

        constexpr int kBlockPerCu = {spec.k_block_per_cu};
        return ck_tile::launch_kernel(
            stream,
            ck_tile::make_kernel<kBlockPerCu>(ContractionKernel{{}}, grids, blocks, 0, kargs));
    }}
}};

using SelectedKernel = {struct};

}} // namespace {ns}

#ifdef CK_TILE_SINGLE_KERNEL_INCLUDE
using SelectedKernel = {ns}::{struct};
constexpr const char* KERNEL_NAME = {ns}::KERNEL_NAME;
using ADataType     = {ns}::ADataType;
using BDataType     = {ns}::BDataType;
using EDataType     = {ns}::EDataType;
using AccDataType   = {ns}::AccDataType;
using DBaseDataType = {ns}::DBaseDataType;
static constexpr int CONTRACTION_KEY_NUM_D_TENSORS = {spec.num_d_tensors};
static constexpr int CONTRACTION_KEY_NUM_DIM_G     = {spec.num_dim_g};
static constexpr int CONTRACTION_KEY_NUM_DIM_M     = {spec.num_dim_m};
static constexpr int CONTRACTION_KEY_NUM_DIM_N     = {spec.num_dim_n};
static constexpr int CONTRACTION_KEY_NUM_DIM_K     = {spec.num_dim_k};
#endif // CK_TILE_SINGLE_KERNEL_INCLUDE
"""


# =============================================================================
# Config -> specs
# =============================================================================


def _spec_from_config(cfg: dict) -> BCKernelSpec:
    tc = cfg["tile_config"] if "tile_config" in cfg else cfg
    tile = BCTileConfig(
        tile_m=tc["tile_m"], tile_n=tc["tile_n"], tile_k=tc["tile_k"],
        warp_m=tc["warp_m"], warp_n=tc["warp_n"], warp_k=tc["warp_k"],
        warp_tile_m=tc["warp_tile_m"], warp_tile_n=tc["warp_tile_n"], warp_tile_k=tc["warp_tile_k"],
    )
    return BCKernelSpec(
        dtype=cfg.get("datatype", "fp16"),
        layout=cfg.get("layout", "rcr"),
        pipeline=cfg.get("pipeline", "compv3"),
        epilogue=cfg.get("epilogue", "cshuffle"),
        scheduler=cfg.get("scheduler", "intrawave"),
        tile=tile,
        pad_m=cfg.get("pad_m", False),
        pad_n=cfg.get("pad_n", False),
        pad_k=cfg.get("pad_k", False),
        persistent=cfg.get("persistent", False),
        num_dim_g=cfg.get("num_dim_g", 1),
        num_dim_m=cfg.get("num_dim_m", 1),
        num_dim_n=cfg.get("num_dim_n", 1),
        num_dim_k=cfg.get("num_dim_k", 1),
        num_d_tensors=cfg.get("num_d_tensors", 0),
        elementwise=cfg.get("elementwise", "PassThrough"),
        block_size=cfg.get("block_size", 256),
        k_block_per_cu=cfg.get("k_block_per_cu", 1),
    )


def generate_kernel(output_dir: Path, cfg: dict) -> Optional[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = _spec_from_config(cfg)
    if not spec.tile.is_valid():
        log.error("invalid tile config for %s", spec.name)
        return None
    header = BCHeaderGenerator().generate(spec)
    out = output_dir / f"{spec.name}.hpp"
    out.write_text(header)
    log.info("wrote %s", out.name)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Batched-contraction kernel header generator")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--config-json", type=str)
    ap.add_argument("--list-name", action="store_true")
    args = ap.parse_args()

    if args.config_json:
        cfg = json.loads(args.config_json)
    elif args.config:
        cfg = json.loads(Path(args.config).read_text())
    else:
        cfg = {}

    if args.list_name:
        print(_spec_from_config(cfg).name)
        return 0

    return 0 if generate_kernel(args.output_dir, cfg) else 1


if __name__ == "__main__":
    raise SystemExit(main())
