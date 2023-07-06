// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

using InDataType  = ck::half_t;
using WeiDataType = ck::half_t;
using OutDataType = ck::half_t;

using InLayout  = ck::tensor_layout::convolution::NDHWGC;
using WeiLayout = ck::tensor_layout::convolution::GKZYXC;
using OutLayout = ck::tensor_layout::convolution::NDHWGK;

static constexpr ck::index_t NumDimSpatial = 3;
static constexpr ck::index_t G             = 1;
static constexpr ck::index_t N             = 64;
static constexpr ck::index_t K             = 128;
static constexpr ck::index_t C             = 64;
static constexpr ck::index_t Z             = 3;
static constexpr ck::index_t Y             = 3;
static constexpr ck::index_t X             = 3;
static constexpr ck::index_t Di            = 28;
static constexpr ck::index_t Hi            = 28;
static constexpr ck::index_t Wi            = 3;
static constexpr ck::index_t Do            = 28;
static constexpr ck::index_t Ho            = 28;
static constexpr ck::index_t Wo            = 3;

int main()
{
    return run_grouped_conv_fwd<NumDimSpatial,
                                InDataType,
                                WeiDataType,
                                OutDataType,
                                InLayout,
                                WeiLayout,
                                OutLayout>(
               {N, Di, Hi, Wi, G, C}, {G, K, Z, Y, X, C}, {N, Do, Ho, Wo, G, K})
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}
