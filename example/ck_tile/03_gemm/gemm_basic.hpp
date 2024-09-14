
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/host.hpp"
#include <string>

template <typename DataType>
struct GemmBasicTypeConfig;

template <>
struct GemmBasicTypeConfig<ck_tile::half_t>
{
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::half_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t; // type convert
    // ToDo: Add more bias config to support different categories of GEMM.
};

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

using Types = GemmBasicTypeConfig<ck_tile::half_t>;

// Specific type aliases for easy access
using ADataType   = Types::ADataType;
using BDataType   = Types::BDataType;
using AccDataType = Types::AccDataType;
using CDataType   = Types::CDataType;

struct gemm_basic_args
{
    const void* p_a;
    const void* p_b;
    void* p_c;
    float epsilon;
    ck_tile::index_t kbatch;
    ck_tile::index_t M;
    ck_tile::index_t N;
    ck_tile::index_t K;
    ck_tile::index_t stride_A;
    ck_tile::index_t stride_B;
    ck_tile::index_t stride_C;
};

// host API
float gemm_calc(gemm_basic_args args, const ck_tile::stream_config& s);

void run_naive_gemm(ck_tile::HostTensor<ADataType>& a_host,
                    ck_tile::HostTensor<BDataType>& b_host,
                    ck_tile::HostTensor<CDataType>& c_host,
                    ck_tile::index_t M,
                    ck_tile::index_t N,
                    ck_tile::index_t K,
                    ck_tile::index_t stride_a,
                    ck_tile::index_t stride_b,
                    ck_tile::index_t stride_c);
