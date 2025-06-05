// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

using InDataType  = ck::half_t;
using WeiDataType = ck::half_t;
using OutDataType = ck::half_t;

using InLayout  = ck::tensor_layout::convolution::NHWGC;
using WeiLayout = ck::tensor_layout::convolution::GKYXC;
using OutLayout = ck::tensor_layout::convolution::NHWGK;

static constexpr ck::index_t NumDimSpatial = 2;
static constexpr ck::index_t G             = 1;
static constexpr ck::index_t N             = 128;
static constexpr ck::index_t K             = 384;
static constexpr ck::index_t C             = 128;
static constexpr ck::index_t Y             = 1;
static constexpr ck::index_t X             = 1;
static constexpr ck::index_t Hi            = 24;
static constexpr ck::index_t Wi            = 48;
static constexpr ck::index_t Ho            = 24;
static constexpr ck::index_t Wo            = 48;
static constexpr std::array<ck::index_t, NumDimSpatial + 3> input_lengths{N, Hi, Wi, G, C};
static constexpr std::array<ck::index_t, NumDimSpatial + 3> filter_lengths{G, K, C, Y, X};
static constexpr std::array<ck::index_t, NumDimSpatial + 3> output_lengths{N, Ho, Wo, G, K};
static constexpr std::array<ck::index_t, NumDimSpatial + 3> input_strides{
    128, 147456, 1, 6144, 128};
static constexpr std::array<ck::index_t, NumDimSpatial + 3> weights_strides{
    49152, 128, 1, 128, 128};
static constexpr std::array<ck::index_t, NumDimSpatial + 3> output_strides{
    384, 442368, 1, 18432, 384};
static constexpr std::array<ck::index_t, NumDimSpatial> conv_filter_strides{1, 1};
static constexpr std::array<ck::index_t, NumDimSpatial> conv_filter_dilations{1, 1};
static constexpr std::array<ck::index_t, NumDimSpatial> input_left_pads{0, 0};
static constexpr std::array<ck::index_t, NumDimSpatial> input_right_pads{0, 0};

int main()
{
    return run_grouped_conv_bwd_weight<NumDimSpatial,
                                       InDataType,
                                       WeiDataType,
                                       OutDataType,
                                       InLayout,
                                       WeiLayout,
                                       OutLayout>(input_lengths,
                                                  input_strides,
                                                  filter_lengths,
                                                  weights_strides,
                                                  output_lengths,
                                                  output_strides,
                                                  conv_filter_strides,
                                                  conv_filter_dilations,
                                                  input_left_pads,
                                                  input_right_pads)
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}
