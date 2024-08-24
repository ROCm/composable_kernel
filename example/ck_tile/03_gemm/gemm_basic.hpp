
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.


#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include <string>

template <typename DataType>
struct GemmBasicTypeConfig;

template <>
struct GemmBasicTypeConfig<ck_tile::half_t> {
    using XDataType         = ck_tile::half_t;
    using YDataType         = ck_tile::half_t;
    using AccDataType       = float;
    using ODataType         = ck_tile::half_t; //type convert
    // ToDo: Add more bias config to support different categories of GEMM.
};

template<ck_tile::MatrixALayout A, ck_tile::MatrixBLayout B, 
         ck_tile::MatrixCLayout C>
struct LayoutConfig {
    static constexpr ck_tile::MatrixALayout LayoutA = A;
    static constexpr ck_tile::MatrixBLayout LayoutB = B;
    static constexpr ck_tile::MatrixCLayout LayoutC = C;
};

template<typename T>
struct DataTypeTraits;

template<>
struct DataTypeTraits<float> {
    static constexpr const char* name = "float";
};

template<>
struct DataTypeTraits<double> {
    static constexpr const char* name = "double";
};

template<>
struct DataTypeTraits<ck_tile::half_t> {
    static constexpr const char* name = "fp16";
};

using Types = GemmBasicTypeConfig<ck_tile::half_t>;

// Specific type aliases for easy access
using XDataType = Types::XDataType;
using YDataType = Types::YDataType;
using AccDataType = Types::AccDataType;
using ODataType = Types::ODataType;

struct gemm_basic_args {
    const void* p_x;
    const void* p_y;
    void* p_z;
    float epsilon;
    ck_tile::index_t batch_size;
    ck_tile::index_t M;
    ck_tile::index_t N;
    ck_tile::index_t K;
    ck_tile::index_t stride_A;
    ck_tile::index_t stride_B;
    ck_tile::index_t stride_C;
    static constexpr ck_tile::index_t kBlockPerCu = 1;
};

// host API
float gemm_calc(gemm_basic_args args, const ck_tile::stream_config& s);
