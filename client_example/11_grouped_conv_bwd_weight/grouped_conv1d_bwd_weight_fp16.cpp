// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

using InDataType  = ck::half_t;
using WeiDataType = ck::half_t;
using OutDataType = ck::half_t;

using InLayout  = ck::tensor_layout::convolution::GNWC;
using WeiLayout = ck::tensor_layout::convolution::GKXC;
using OutLayout = ck::tensor_layout::convolution::GNWK;

static constexpr ck::index_t NumDimSpatial = 1;
static constexpr ck::index_t G             = 32;
static constexpr ck::index_t N             = 256;
static constexpr ck::index_t K             = 192;
static constexpr ck::index_t C             = 192;
static constexpr ck::index_t X             = 3;
static constexpr ck::index_t Wi            = 28;
static constexpr ck::index_t Wo            = 28;
static constexpr std::array<ck::index_t, NDimSpatial + 3> input_strides{
    G * N * Wi * C, N* Wi* C, Wi* C, C, 1};
static constexpr std::array<ck::index_t, NDimSpatial + 3> output_strides{
    G * N * Wo * K, N* Wo* K, Wo* K, K, 1};

int main()
{
    return run_grouped_conv_bwd_weight<NumDimSpatial,
                                       InDataType,
                                       WeiDataType,
                                       OutDataType,
                                       InLayout,
                                       WeiLayout,
                                       OutLayout>(
               G, N, K, C, {Wi}, {X}, {Wo}, input_strides, output_strides, {} {1}, {1}, {1}, {1})
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}
