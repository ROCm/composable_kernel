// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

#define CK_TILE_PIPELINE_COMPUTE_V3 1
#define CK_TILE_PIPELINE_MEMORY 2
#define CK_TILE_PIPELINE_COMPUTE_V4 3

#ifndef CK_TILE_PIPELINE_DEFAULT
#define CK_TILE_PIPELINE_DEFAULT CK_TILE_PIPELINE_COMPUTE_V3
#endif

using A0DataType = ck_tile::half_t;
using A1DataType = ck_tile::half_t;

using B0DataType = ck_tile::half_t;
using B1DataType = ck_tile::half_t;

using D0DataType = ck_tile::half_t;
using D1DataType = ck_tile::half_t;

using EDataType = ck_tile::half_t;

using AsDataType = ck_tile::tuple<A0DataType, A1DataType>;
using BsDataType = ck_tile::tuple<B0DataType, B1DataType>;
using DsDataType = ck_tile::tuple<D0DataType, D1DataType>;

using AccDataType = float;

struct GemmConfigMemory
{
    // Memory friendly for Interwave scheduler
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 32;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 8;

    static constexpr bool DoubleSmemBuffer     = false;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_MEMORY;
    static constexpr auto Scheduler            = ck_tile::GemmPipelineScheduler::Interwave;
};

struct GemmConfigV3
{
    // Compute friendly for Intrawave scheduler
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool DoubleSmemBuffer     = false;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V3;
    static constexpr auto Scheduler            = ck_tile::GemmPipelineScheduler::Intrawave;
};

struct GemmConfigV4
{
    // Compute friendly for Intrawave scheduler
    // Using the ping pong reader in the lds level
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 32;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool DoubleSmemBuffer     = true;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V4;
    static constexpr auto Scheduler            = ck_tile::GemmPipelineScheduler::Intrawave;
};

struct GemmConfigV3_Wmma
{
    // Compute friendly for Intrawave scheduler
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 64;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool DoubleSmemBuffer     = false;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V3;
    static constexpr auto Scheduler            = ck_tile::GemmPipelineScheduler::Intrawave;
};

template <ck_tile::index_t PipelineId>
struct PipelineTypeTraits;

template <>
struct PipelineTypeTraits<CK_TILE_PIPELINE_MEMORY>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrMem<PipelineProblem>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrMem<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<CK_TILE_PIPELINE_COMPUTE_V3>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<PipelineProblem>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<PipelineProblem>;
};

template <>
struct PipelineTypeTraits<CK_TILE_PIPELINE_COMPUTE_V4>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV4<PipelineProblem>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV4<PipelineProblem>;
};

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "3840", "m dimension")
        .insert("n", "4096", "n dimension")
        .insert("k", "4096", "k dimension")
        .insert("as_layout", "R", "As tensor data layout - Row by default")
        .insert("bs_layout", "C", "Bs tensor data layout - Col by default")
        .insert("ds_layout", "R", "Ds tensor data layout - Row by default")
        .insert("e_layout", "R", "E tensor data layout - Row by default")
        .insert("stride_as", "0", "Tensor A stride")
        .insert("stride_bs", "0", "Tensor B stride")
        .insert("stride_ds", "0", "Tensor Ds stride")
        .insert("stride_e", "0", "Tensor E stride")
        .insert("v", "1", "0. No validation, 1. Validation on GPU")
        .insert("warmup", "50", "number of iterations before benchmark the kernel")
        .insert("repeat", "100", "number of iterations to benchmark the kernel")
        .insert("kbatch", "1", "kbatch for SplitK");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}
using gemm_multi_abd_kargs =
    ck_tile::GemmMultiABDHostArgs<AsDataType::size(), BsDataType::size(), DsDataType::size()>;

template <typename GemmConfig,
          typename AsDataType,
          typename BsDataType,
          typename DsDataType,
          typename AccDataType,
          typename CDataType,
          typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename CLayout,
          typename AElementWise,
          typename BElementWise,
          typename CDEElementWise>
float gemm_multi_abd(const gemm_multi_abd_kargs& kargs, const ck_tile::stream_config& s);
