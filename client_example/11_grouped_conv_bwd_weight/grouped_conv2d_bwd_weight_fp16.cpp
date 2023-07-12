// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

using InDataType  = ck::half_t;
using WeiDataType = ck::half_t;
using OutDataType = ck::half_t;

using InLayout  = ck::tensor_layout::convolution::GNHWC;
using WeiLayout = ck::tensor_layout::convolution::GKYXC;
using OutLayout = ck::tensor_layout::convolution::GNHWK;

static constexpr ck::index_t NumDimSpatial = 2;
static constexpr ck::index_t G             = 32;
static constexpr ck::index_t N             = 256;
static constexpr ck::index_t K             = 192;
static constexpr ck::index_t C             = 192;
static constexpr ck::index_t Y             = 3;
static constexpr ck::index_t X             = 3;
static constexpr ck::index_t Hi            = 28;
static constexpr ck::index_t Wi            = 28;
static constexpr ck::index_t Ho            = 28;
static constexpr ck::index_t Wo            = 28;
static constexpr std::array<ck::index_t, NumDimSpatial> input_spatial_lengths{Hi, Wi};
static constexpr std::array<ck::index_t, NumDimSpatial> filter_spatial_lengths{Y, X};
static constexpr std::array<ck::index_t, NumDimSpatial> output_spatial_lengths{Ho, Wo};
static constexpr std::array<ck::index_t, NumDimSpatial + 3> input_strides{
    N * Hi * Wi * C, Hi* Wi* C, Wi* C, C, 1};
static constexpr std::array<ck::index_t, NumDimSpatial + 3> output_strides{
    N * Ho * Wo * K, Ho* Wo* K, Wo* K, K, 1};
static constexpr std::array<ck::index_t, NumDimSpatial> conv_filter_strides{1, 1};
static constexpr std::array<ck::index_t, NumDimSpatial> conv_filter_dilations{1, 1};
static constexpr std::array<ck::index_t, NumDimSpatial> input_left_pads{1, 1};
static constexpr std::array<ck::index_t, NumDimSpatial> input_right_pads{1, 1};

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
                                                  input_spatial_lengths,
                                                  filter_spatial_lengths,
                                                  output_spatial_lengths,
                                                  input_strides,
                                                  output_strides,
                                                  conv_filter_strides,
                                                  conv_filter_dilations,
                                                  input_left_pads,
                                                  input_right_pads)
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}
