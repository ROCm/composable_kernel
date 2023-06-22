// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

using InDataType  = float;
using WeiDataType = float;
using OutDataType = float;

using InLayout  = ck::tensor_layout::convolution::GNDHWC;
using WeiLayout = ck::tensor_layout::convolution::GKZYXC;
using OutLayout = ck::tensor_layout::convolution::GNDHWK;

static constexpr ck::index_t NumDimSpatial = 3;
static constexpr ck::index_t G             = 8;
static constexpr ck::index_t N             = 64;
static constexpr ck::index_t K             = 128;
static constexpr ck::index_t C             = 128;
static constexpr ck::index_t Z             = 3;
static constexpr ck::index_t Y             = 3;
static constexpr ck::index_t X             = 3;
static constexpr ck::index_t Di            = 28;
static constexpr ck::index_t Hi            = 28;
static constexpr ck::index_t Wi            = 3;
static constexpr ck::index_t Do            = 28;
static constexpr ck::index_t Ho            = 28;
static constexpr ck::index_t Wo            = 3;
static constexpr std::array<ck::index_t, NDimSpatial + 3> input_strides{
    G * N * Di * Hi * Wi * C, N* Di* Hi* Wi* C, Di* Hi* Wi* C, Hi* Wi* C, Wi* C, C, 1};
static constexpr std::array<ck::index_t, NDimSpatial + 3> output_strides{
    G * N * Do * Ho * Wo * K, N* Do* Ho* Wo* K, Do* Ho* Wo* K, Ho* Wo* K, Wo* K, K, 1};

int main()
{
    return run_grouped_conv_bwd_weight<NumDimSpatial,
                                       InDataType,
                                       WeiDataType,
                                       OutDataType,
                                       InLayout,
                                       WeiLayout,
                                       OutLayout>(G,
                                                  N,
                                                  K,
                                                  C,
                                                  {Di, Hi, Wi},
                                                  {Z, Y, X},
                                                  {Do, Ho, Wo},
                                                  input_strides,
                                                  output_strides,
                                                  {1, 1, 1},
                                                  {1, 1, 1},
                                                  {1, 1, 1},
                                                  {1, 1, 1})
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}
