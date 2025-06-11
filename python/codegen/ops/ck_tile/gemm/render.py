# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

from .instance import GEMM
from dataclasses import asdict
from string import Template

instance_template = Template(r"""
namespace $instance_name {
    // block tile
    constexpr int32_t TileM = $tile_m;
    constexpr int32_t TileN = $tile_n;
    constexpr int32_t TileK = $tile_k;
    // warps per block
    constexpr int32_t WarpM = $warp_m;
    constexpr int32_t WarpN = $warp_n;
    constexpr int32_t WarpK = $warp_k;
    // xdl tile
    constexpr int32_t WarpTileM = $warp_tile_m;
    constexpr int32_t WarpTileN = $warp_tile_n;
    constexpr int32_t WarpTileK = $warp_tile_k;

    constexpr bool kPadM = $m_is_padded;
    constexpr bool kPadN = $n_is_padded;
    constexpr bool kPadK = $k_is_padded;

    using ALayout = $layout_a;
    using BLayout = $layout_b;
    using CLayout = $layout_c;

    using ADataType = $datatype_a;
    using BDataType = $datatype_b;
    using CDataType = $datatype_c;
    using AccDataType = F32;

    constexpr bool permuteA = false;
    constexpr bool permuteB = false;
    constexpr bool DoubleSmemBuffer = $has_double_smem_buffer;
    constexpr bool TransposeC = false;

    constexpr int kBlockPerCu                          = 1;
    constexpr ck_tile::index_t TilePartitionerGroupNum = 8;
    constexpr ck_tile::index_t TilePartitionerM01      = 4;

    using GemmShape =
        ck_tile::TileGemmShape<ck_tile::sequence<TileM, TileN, TileK>,
                               ck_tile::sequence<WarpM, WarpN, WarpK>,
                               ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>,
                               permuteA,
                               permuteB>;

    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                   TilePartitionerGroupNum,
                                                   TilePartitionerM01>;

    using Traits =
        ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout>;

    using GemmUniversalTraits =
        ck_tile::TileGemmUniversalTraits<kPadM, kPadN, kPadK, 
                                         DoubleSmemBuffer,
                                         ALayout, BLayout, CLayout, 
                                         TransposeC>;

    using GemmPipelineProblem =
        ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, GemmShape, Traits>;

    $rendered_scheduler

    template<bool has_hot_loop_v, ck_tile::TailNumber tail_number_v>
    using UniversalGemmProblem =
        ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                BDataType,
                                                AccDataType,
                                                GemmShape,
                                                GemmUniversalTraits,
                                                scheduler,
                                                has_hot_loop_v,
                                                tail_number_v>;

    $rendered_pipeline

    $rendered_epilogue

    template<bool has_hot_loop_v, ck_tile::TailNumber tail_number_v>
    using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline<has_hot_loop_v, tail_number_v>, GemmEpilogue>;
} // namespace $instance_name
""")


def render(instance: GEMM):
    def render_epilogue(epilogue_type):
        if epilogue_type == "Default":
            return r"""
    using EpilogueProblem = ck_tile::DefaultGemm2DEpilogueProblem<ADataType,
                                                                    BDataType,
                                                                    AccDataType,
                                                                    CDataType,
                                                                    CLayout,
                                                                    kPadM,
                                                                    kPadN,
                                                                    WarpTileM,
                                                                    WarpTileN,
                                                                    WarpTileK,
                                                                    TransposeC>;
    using GemmEpilogue = ck_tile::DefaultGemm2DEpilogue<EpilogueProblem>;
    """
        elif epilogue_type == "CShuffle":
            return r"""
    constexpr auto kMemoryOperation = ck_tile::memory_operation_enum::set;
    using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<ADataType,
                                                                BDataType,
                                                                AccDataType,
                                                                CDataType,
                                                                CLayout,
                                                                GemmPipelineProblem::kBlockSize,
                                                                TileM,
                                                                TileN,
                                                                WarpM,
                                                                WarpN,
                                                                WarpTileM,
                                                                WarpTileN,
                                                                WarpTileK,
                                                                TransposeC,
                                                                kMemoryOperation>;

    using GemmEpilogue = ck_tile::CShuffleEpilogue<EpilogueProblem>;
    """
        else:
            raise AssertionError("Epilogue must be set")

    def render_pipeline(pipeline_type):
        return rf"""
    using BaseGemmPipeline = ck_tile::BaseGemmPipelineAgBgCr{pipeline_type}<GemmPipelineProblem>;

    template<bool has_hot_loop_v, ck_tile::TailNumber tail_number_v>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCr{pipeline_type}<UniversalGemmProblem<has_hot_loop_v, tail_number_v>>;
    """

    def render_scheduler(scheduler_type):
        return rf"""
    constexpr auto scheduler = ck_tile::GemmPipelineScheduler::{scheduler_type};
    """

    rendered_definition = instance_template.substitute(
        instance_name=instance.name(),
        **asdict(instance),
        rendered_scheduler=render_scheduler(instance.scheduler),
        rendered_pipeline=render_pipeline(instance.pipeline),
        rendered_epilogue=render_epilogue(instance.epilogue),
        has_double_smem_buffer=("true" if instance.pipeline == "CompV4" else "false"),
    )
    return rendered_definition
