// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"

// Pipeline definitions (from batched_gemm.hpp)
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

// ✅ FIXED: Corrected template declaration
template <typename DataType>
struct BatchedContractionTypeConfig;

template <>
struct BatchedContractionTypeConfig<float>
{
    using ADataType   = float;
    using BDataType   = float;
    using AccDataType = float;
    using EDataType   = float;
};

template <>
struct BatchedContractionTypeConfig<ck_tile::half_t>
{
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::half_t;
    using AccDataType = float;
    using EDataType   = ck_tile::half_t;
};

template <>
struct BatchedContractionTypeConfig<ck_tile::bf16_t>
{
    using ADataType   = ck_tile::bf16_t;
    using BDataType   = ck_tile::bf16_t;
    using AccDataType = float;
    using EDataType   = ck_tile::bf16_t;
};

// Data type traits for printing
template <typename T>
struct DataTypeTraits;

template <>
struct DataTypeTraits<float>
{
    static constexpr const char* name = "fp32";
};

template <>
struct DataTypeTraits<ck_tile::half_t>
{
    static constexpr const char* name = "fp16";
};

template <>
struct DataTypeTraits<ck_tile::bf16_t>
{
    static constexpr const char* name = "bf16";
};

// Layout helper (for future use)
template <typename Layout>
constexpr bool is_row_major()
{
    return std::is_same_v<Layout, ck_tile::tensor_layout::gemm::RowMajor>;
}

using ContractionTypes = BatchedContractionTypeConfig<float>;

using ADataType   = ContractionTypes::ADataType;
using BDataType   = ContractionTypes::BDataType;
using AccDataType = ContractionTypes::AccDataType;
using EDataType   = ContractionTypes::EDataType;

// Argument parser function (copied from batched_gemm pattern)
auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "512", "m dimension")
        .insert("n", "1024", "n dimension")
        .insert("k", "2048", "k dimension")
        .insert("stride_a", "0", "Tensor A stride")
        .insert("stride_b", "0", "Tensor B stride")
        .insert("stride_e", "0", "Tensor E stride")
        .insert("a_layout", "R", "A tensor data layout - Row by default")
        .insert("b_layout", "C", "B tensor data layout - Col by default")
        .insert("e_layout", "R", "E tensor data layout - Row by default")
        .insert("batch_stride_a", "1048576", "Batch A stride")
        .insert("batch_stride_b", "2097152", "Batch B stride")
        .insert("batch_stride_e", "524288", "Batch E stride")
        .insert("batch_count", "8", "Batch count")
        .insert("v", "1", "0. No validation, 1. Validation on CPU")
        .insert("prec", "fp32", "data type. fp32/fp16/bf16")
        .insert("warmup", "5", "number of iterations before benchmark the kernel")
        .insert("repeat", "10", "number of iterations to benchmark the kernel")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer")
        .insert("split_k", "1", "splitK value")
        .insert("log", "1", "log level for debugging");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}
