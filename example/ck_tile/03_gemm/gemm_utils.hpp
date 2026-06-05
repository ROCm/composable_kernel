// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/utility/json_dump.hpp"

#include <string>
#include <variant>

struct GemmConfigBase
{
    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool PermuteA = false;
    static constexpr bool PermuteB = false;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;

    static constexpr int kBlockPerCu                         = 1;
    static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    static constexpr ck_tile::index_t TileParitionerM01      = 4;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool Preshuffle                = false;
    static constexpr bool TiledMMAPermuteN          = false;

    static constexpr ck_tile::index_t kClusterSizeM       = 1;
    static constexpr ck_tile::index_t kClusterSizeN       = 1;
    static constexpr ck_tile::index_t BlockedXDLN_PerWarp = 1;
    static constexpr ck_tile::DataCachePrefetchKind DataCachePrefetchA =
        ck_tile::DataCachePrefetchKind::None;
    static constexpr ck_tile::DataCachePrefetchKind DataCachePrefetchB =
        ck_tile::DataCachePrefetchKind::None;
};

template <typename PrecType>
struct GemmConfigMemoryInterwave : public GemmConfigBase
{
    // Memory friendly for Interwave scheduler
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 32;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = sizeof(PrecType) == 2 ? 8 : 16;

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::MEMORY;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Interwave;
};

template <typename PrecType>
struct GemmConfigMemoryIntrawave : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 32;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = sizeof(PrecType) == 2 ? 8 : 16;

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::MEMORY;
};

template <typename PrecType>
struct GemmConfigComputeV3 : public GemmConfigBase
{
    // Compute V3 only support Intrawave scheduler
    static constexpr ck_tile::index_t M_Tile = 16;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 256 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;
};

template <typename PrecType>
struct GemmConfigComputeV3_1 : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;
};

template <typename PrecType>
struct GemmConfigComputeV3_2 : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;

    static constexpr int kBlockPerCu = 2;
};

template <typename PrecType>
struct GemmConfigComputeV3_WMMA : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 64 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;

    static constexpr int kBlockPerCu = 2;
};

template <typename PrecType>
struct GemmConfigComputeV3_WMMA_ClusterLaunch : public GemmConfigComputeV3_WMMA<PrecType>
{
    static constexpr ck_tile::index_t kClusterSizeM = 2;
    static constexpr ck_tile::index_t kClusterSizeN = 2;
};

template <typename PrecType>
struct GemmConfigComputeV4 : public GemmConfigBase
{
    // Compute V4 only support Intrawave scheduler
    // Using the ping pong reader in the lds level
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 64 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();

    static constexpr bool DoubleSmemBuffer          = true;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V4;
};

template <typename PrecType>
struct GemmConfigComputeV4_1 : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();

    static constexpr bool DoubleSmemBuffer          = true;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V4;
};

template <typename PrecType>
struct GemmConfigComputeV5 : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 64 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 2;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V5;
    static constexpr ck_tile::index_t NumWaveGroups = 2;
};

template <typename PrecType>
struct GemmConfigComputeV6 : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 32;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool DoubleSmemBuffer          = false;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V6;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
};

template <typename PrecType>
struct GemmConfigComputeAsync : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 64;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 256;

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 128;

    static constexpr bool DoubleSmemBuffer          = true;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_ASYNC;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool UseStructuredSparsity     = false;
};

template <typename PrecType>
struct GemmConfigPreshuffleDecode : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 16;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 256 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile, true>();

    static constexpr int kBlockPerCu                = 1;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Default;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::PRESHUFFLE_V2;
    static constexpr bool Preshuffle                = true;
    static constexpr bool DoubleSmemBuffer          = true;
    static constexpr int N_Repeat                   = N_Tile / N_Warp_Tile / N_Warp;
    static constexpr bool TiledMMAPermuteN          = N_Repeat % 2 == 0;
};

template <typename PrecType>
struct GemmConfigPreshufflePrefill : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile, true>();

    static constexpr int kBlockPerCu                = 2;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Default;
    static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::PRESHUFFLE_V2;
    static constexpr bool Preshuffle                = true;
    static constexpr bool DoubleSmemBuffer          = true;
    static constexpr int N_Repeat                   = N_Tile / N_Warp_Tile / N_Warp;
    static constexpr bool TiledMMAPermuteN          = N_Repeat % 2 == 0;

    static constexpr bool Async = false;
};

template <typename PrecType>
struct GemmConfigPreshufflePrefillAsync : public GemmConfigPreshufflePrefill<PrecType>
{
    static constexpr ck_tile::index_t N_Tile = 256;

    // N_Repeat is even in this config
    static constexpr bool TiledMMAPermuteN = true;

    static constexpr bool Async = true;
};

template <typename PrecType>
struct GemmConfigPreshufflePrefill_Wmma : public GemmConfigPreshufflePrefill<PrecType>
{
    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile, true>();
};

template <typename PrecType>
struct GemmConfigMixedPrec_Wmma : public GemmConfigComputeV3_WMMA<PrecType>
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 128;
};

template <typename ADataType, typename BDataType = ADataType, typename CDataType = ADataType>
struct GemmTypeConfig;

template <>
struct GemmTypeConfig<ck_tile::tf32_t, ck_tile::tf32_t, float>
{
    using ADataType   = ck_tile::tf32_t;
    using BDataType   = ck_tile::tf32_t;
    using AccDataType = float;
    using CDataType   = float;
};

template <>
struct GemmTypeConfig<ck_tile::half_t>
{
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::half_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
    // ToDo: Add more bias config to support different categories of GEMM.
};

template <>
struct GemmTypeConfig<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t>
{
    using ADataType   = ck_tile::bf16_t;
    using BDataType   = ck_tile::bf16_t;
    using AccDataType = float;
    using CDataType   = ck_tile::bf16_t;
};

template <>
struct GemmTypeConfig<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t>
{
    using ADataType   = ck_tile::fp8_t;
    using BDataType   = ck_tile::fp8_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
};

template <>
struct GemmTypeConfig<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t>
{
    using ADataType   = ck_tile::bf8_t;
    using BDataType   = ck_tile::bf8_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
};

template <>
struct GemmTypeConfig<ck_tile::fp8_t, ck_tile::pk_int4_t, ck_tile::half_t>
{
    using ADataType   = ck_tile::fp8_t;
    using BDataType   = ck_tile::pk_int4_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
};

template <>
struct GemmTypeConfig<ck_tile::bf8_t, ck_tile::pk_int4_t, ck_tile::half_t>
{
    using ADataType   = ck_tile::bf8_t;
    using BDataType   = ck_tile::pk_int4_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
};

template <>
struct GemmTypeConfig<ck_tile::half_t, ck_tile::pk_int4_t, ck_tile::half_t>
{
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::pk_int4_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
};

template <>
struct GemmTypeConfig<ck_tile::int8_t, ck_tile::int8_t, int32_t>
{
    using ADataType   = ck_tile::int8_t;
    using BDataType   = ck_tile::int8_t;
    using AccDataType = int32_t;
    using CDataType   = int32_t;
};

template <>
struct GemmTypeConfig<ck_tile::pk_fp4_t, ck_tile::pk_fp4_t, ck_tile::half_t>
{
    using ADataType   = ck_tile::pk_fp4_t;
    using BDataType   = ck_tile::pk_fp4_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
};

template <>
struct GemmTypeConfig<ck_tile::fp8_t, ck_tile::pk_fp4_t, ck_tile::half_t>
{
    using ADataType   = ck_tile::fp8_t;
    using BDataType   = ck_tile::pk_fp4_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
};

template <>
struct GemmTypeConfig<ck_tile::bf8_t, ck_tile::pk_fp4_t, ck_tile::half_t>
{
    using ADataType   = ck_tile::bf8_t;
    using BDataType   = ck_tile::pk_fp4_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
};

template <ck_tile::GemmPipeline PipelineId>
struct PipelineTypeTraits;

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::MEMORY>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrMem<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::BASIC_V1>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAGmemBGmemCRegV1<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::BASIC_V2>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_V3>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_V4>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV4<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_V5>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV5<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_V6>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV6<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_ASYNC>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompAsync<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::PRESHUFFLE_V2>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::WeightPreshufflePipelineAGmemBGmemCRegV2<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_ASYNC_V2>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompAsyncV2<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_TDM_V1>
{
    template <typename PipelineProblem>
    using GemmPipeline =
        ck_tile::GemmPipelineAgBgCrCompTDMV1<PipelineProblem,
                                             ck_tile::GemmPipelineAgBgCrCompTDMDefaultPolicy<
                                                 false,
                                                 PipelineProblem::Traits::DataCachePrefetchA,
                                                 PipelineProblem::Traits::DataCachePrefetchB>>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_TDM_V2>
{
    template <typename PipelineProblem>
    using GemmPipeline =
        ck_tile::GemmPipelineAgBgCrCompTDMV2<PipelineProblem,
                                             ck_tile::GemmPipelineAgBgCrCompTDMDefaultPolicy<
                                                 true,
                                                 PipelineProblem::Traits::DataCachePrefetchA,
                                                 PipelineProblem::Traits::DataCachePrefetchB>>;
};

template <>
struct PipelineTypeTraits<ck_tile::GemmPipeline::PRESHUFFLE_TDM>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::WeightPreshufflePipelineAGmemBGmemCRegTDM<
        PipelineProblem,
        ck_tile::UniversalWeightPreshufflePipelineAgBgCrTDMPolicy<
            PipelineProblem::Traits::DataCachePrefetchA,
            PipelineProblem::Traits::DataCachePrefetchB>>;
};

template <ck_tile::GemmPipeline PipelineId, typename Problem>
struct EpilogueTypeTraits
{
    using Epilogue = ck_tile::CShuffleEpilogue<Problem>;
};

template <typename Problem>
struct EpilogueTypeTraits<ck_tile::GemmPipeline::COMPUTE_TDM_V1, Problem>
{
    using Epilogue = ck_tile::TdmEpilogue<Problem>;
};

template <typename Problem>
struct EpilogueTypeTraits<ck_tile::GemmPipeline::COMPUTE_TDM_V2, Problem>
{
    using Epilogue = ck_tile::TdmEpilogue<Problem>;
};

inline auto create_args()
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "3840", "m dimension")
        .insert("n", "4096", "n dimension")
        .insert("k", "2048", "k dimension")
        .insert("a_layout", "R", "A tensor data layout - Row by default")
        .insert("b_layout", "C", "B tensor data layout - Column by default")
        .insert("c_layout", "R", "C tensor data layout - Row by default")
        .insert("stride_a", "0", "Tensor A stride")
        .insert("stride_b", "0", "Tensor B stride")
        .insert("stride_c", "0", "Tensor C stride")
        .insert("v", "2", "0. No validation, 1. Validation on CPU, 2. Validation on GPU")
        .insert("prec", "fp16", "data type. fp16/bf16/fp8/bf8/pk_int4_t/tf32 (tf32 only on gfx950)")
        .insert("warmup", "50", "number of iterations before benchmark the kernel")
        .insert("repeat", "100", "number of iterations to benchmark the kernel")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer")
        .insert("split_k", "1", "splitK value")
        .insert("init", "0", "0:random, 1:linear, 2:constant(1)")
        .insert("persistent", "0", "0:non-persistent, 1:persistent")
        .insert("json", "0", "0: No Json, 1: Dump Results in Json format")
        .insert("jsonfile", "gemm.json", "json file name to dump results")
        .insert("flush_cache", "true", "flush cache before running the kernel, defaults to true")
        .insert("rotating_count", "1000", "rotating count, defaults to 1000")
        .insert("test_async", "0", "0: normal gemm, 1: test async input scheduler");
    return arg_parser;
}

// host API
template <typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          bool Persistent = false,
          typename CDEElementWise>
float gemm(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s);
