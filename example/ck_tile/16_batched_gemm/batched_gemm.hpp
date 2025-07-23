// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm/kernel/batched_gemm_kernel.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

#define CK_TILE_PIPELINE_COMPUTE_V3 1
#define CK_TILE_PIPELINE_MEMORY 2
#define CK_TILE_PIPELINE_COMPUTE_V4 3

#ifndef CK_TILE_PIPELINE_DEFAULT
#define CK_TILE_PIPELINE_DEFAULT CK_TILE_PIPELINE_COMPUTE_V3
#endif

#if(CK_TILE_PIPELINE_DEFAULT == CK_TILE_PIPELINE_MEMORY)
#define GEMM_PIPELINE ck_tile::GemmPipelineAgBgCrMem
#define UNIVERSAL_GEMM_PIPELINE ck_tile::BaseGemmPipelineAgBgCrMem
#define GEMM_PIPELINE_SCHEDULER ck_tile::GemmPipelineScheduler::Interwave
#elif(CK_TILE_PIPELINE_DEFAULT == CK_TILE_PIPELINE_COMPUTE_V3)
#define GEMM_PIPELINE ck_tile::GemmPipelineAgBgCrCompV3
#define UNIVERSAL_GEMM_PIPELINE ck_tile::BaseGemmPipelineAgBgCrCompV3
#define GEMM_PIPELINE_SCHEDULER ck_tile::GemmPipelineScheduler::Intrawave
#elif(CK_TILE_PIPELINE_DEFAULT == CK_TILE_PIPELINE_COMPUTE_V4)
#define GEMM_PIPELINE ck_tile::GemmPipelineAgBgCrCompV4
#define UNIVERSAL_GEMM_PIPELINE ck_tile::BaseGemmPipelineAgBgCrCompV4
#define GEMM_PIPELINE_SCHEDULER ck_tile::GemmPipelineScheduler::Intrawave
#else
#error "unsupported CK_TILE_PIPELINE_DEFAULT value"
#endif

template <typename DataType>
struct BatchedGemmTypeConfig;

template <>
struct BatchedGemmTypeConfig<ck_tile::half_t>
{
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::half_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
};

using Types = BatchedGemmTypeConfig<ck_tile::half_t>;

// Specific type aliases for easy access
using ADataType   = Types::ADataType;
using BDataType   = Types::BDataType;
using AccDataType = Types::AccDataType;
using CDataType   = Types::CDataType;

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "512", "m dimension")
        .insert("n", "1024", "n dimension")
        .insert("k", "2048", "k dimension")
        .insert("stride_a", "0", "Tensor A stride")
        .insert("stride_b", "0", "Tensor B stride")
        .insert("stride_c", "0", "Tensor C stride")
        .insert("a_layout", "R", "A tensor data layout - Row by default")
        .insert("b_layout", "C", "B tensor data layout - Row by default")
        .insert("c_layout", "R", "C tensor data layout - Row by default")
        .insert("batch_stride_a", "1048576", "Batch A stride")
        .insert("batch_stride_b", "2097152", "Batch B stride")
        .insert("batch_stride_c", "524288", "Batch C stride")
        .insert("batch_count", "8", "Batch count")
        .insert("v", "2", "0. No validation, 1. Validation on CPU, 2. Validation on GPU")
        .insert("prec", "fp16", "data type. fp16/bf16/fp8/bf8")
        .insert("warmup", "50", "number of iterations before benchmark the kernel")
        .insert("repeat", "100", "number of iterations to benchmark the kernel")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer")
        .insert("split_k", "1", "splitK value");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

// host API
float batched_gemm(const ck_tile::BatchedGemmHostArgs& args, const ck_tile::stream_config& s);
