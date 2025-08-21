// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstring>
#include <string>
#include <tuple>

#include "ck_tile/host.hpp"
#include "gemm_multi_d_dispatcher.hpp"
#include "gemm_multi_d_common.hpp"

template <typename T>
struct DataTypeTraits;

template <>
struct DataTypeTraits<float>
{
    static constexpr const char* name = "fp32";
};

template <>
struct DataTypeTraits<double>
{
    static constexpr const char* name = "fp64";
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

template <>
struct DataTypeTraits<ck_tile::fp8_t>
{
    static constexpr const char* name = "fp8";
};

template <>
struct DataTypeTraits<ck_tile::bf8_t>
{
    static constexpr const char* name = "bf8";
};

template <>
struct DataTypeTraits<ck_tile::int8_t>
{
    static constexpr const char* name = "int8";
};

template <>
struct DataTypeTraits<ck_tile::int32_t>
{
    static constexpr const char* name = "int32";
};

template <>
struct DataTypeTraits<ck_tile::pk_int4_t>
{
    static constexpr const char* name = "pk_int4_t";
};

template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

inline auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "3840", "The value for m dimension. Default is 3840.")
        .insert("n", "4096", "The value for n dimension. Default is 4096.")
        .insert("k", "2048", "The value for k dimension. Default is 2048.")
        .insert("stride_a", "0", "The stride value for tensor A. Default is 0.")
        .insert("stride_b", "0", "The stride value for tensor B. Default is 0.")
        .insert("stride_ds", "0", "The stride value for tensor Ds  Default is 0.")
        .insert("stride_e", "0", "The stride value for tensor E  Default is 0.")
        .insert("split_k", "1", "The split value for k dimension. Default is 1.")
        .insert("verify",
                "1",
                "The type of validation. Set to 0 for no validation, 1 for validation on CPU, or 2 "
                "for validation on GPU. Default is 1, validation on CPU, as validation on GPU is "
                "not supported.")
        .insert("log",
                "false",
                "Wether output kernel instance information or not. Possible values are true or "
                "false. Default is false")
        .insert("warmup",
                "50",
                "The number of iterations before benchmarking the kernel. Default is 50.")
        .insert("repeat",
                "100",
                "The number of iterations for benchmarking the kernel. Default is 100.")
        .insert("timer",
                "true",
                "Indicates whether the timer is a GPU timer. Possible values are true or false. "
                "Default is true.")
        .insert("init",
                "0",
                "The method of tensor initialization. Set to 0 for random, to 1 for linear, or 2 "
                "for constant(1). Default is 0, random.")
        .insert("flush_cache",
                "false",
                "To flush cache, possible values are true or false. "
                "Default is false.")
        .insert("rotating_count", "5", "number of iterations to rotate the cache. default is 5.")
        .insert("metric",
                "0",
                "Metric with which to measure kernel performance. Set to 0 for latency, 1 for "
                "tflops, or 2 for bandwidth. Default is 0, latency.")
        .insert("csv_filename",
                "gemm_multi_d_kernel",
                "The filename of benchmark result. Default is set to gemm_multi_d_kernel.")
        .insert(
            "pipeline",
            "compv3",
            "The type of pipeline. Possible values are compv3, compv4 or mem. Default is compv3.")
        .insert("scheduler",
                "intrawave",
                "The type of pipeline. Possible values are compv3, compv4 or mem. Default is "
                "compv3.")
        .insert(
            "epilogue",
            "cshuffle",
            "The type of epilogue. Possible values are cshuffle or default. Default is cshuffle.")
        .insert("pad_m",
                "false",
                "Whether pad or not in m direction. Possible values are true or false. Default is "
                "false.")
        .insert("pad_n",
                "false",
                "Whether pad or not in n direction. Possible values are true or false. Default is "
                "false.")
        .insert("pad_k",
                "false",
                "Whether pad or not in k direction. Possible values are true or false. Default is "
                "false.");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

auto get_kernel_func_by_trait(const ck_tile::ArgParser& arg_parser)
{
    KernelTraits trait;
    trait.pipeline  = arg_parser.get_str("pipeline");
    trait.scheduler = arg_parser.get_str("scheduler");
    trait.epilogue  = arg_parser.get_str("epilogue");
    trait.pad_m     = arg_parser.get_bool("pad_m");
    trait.pad_n     = arg_parser.get_bool("pad_n");
    trait.pad_k     = arg_parser.get_bool("pad_k");

    return GemmMultiDDispatcher::dispatch(trait);
}
