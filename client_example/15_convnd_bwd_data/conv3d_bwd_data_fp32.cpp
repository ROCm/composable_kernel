// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

using InDataType  = float;
using WeiDataType = float;
using OutDataType = float;

using InLayout  = ck::tensor_layout::convolution::NDHWC;
using WeiLayout = ck::tensor_layout::convolution::KZYXC;
using OutLayout = ck::tensor_layout::convolution::NDHWK;

static constexpr ck::index_t NumDimSpatial = 3;
static constexpr ck::index_t N             = 64;
static constexpr ck::index_t K             = 128;
static constexpr ck::index_t C             = 64;
static constexpr ck::index_t Z             = 3;
static constexpr ck::index_t Y             = 3;
static constexpr ck::index_t X             = 3;
static constexpr ck::index_t Di            = 28;
static constexpr ck::index_t Hi            = 28;
static constexpr ck::index_t Wi            = 28;
static constexpr ck::index_t Do            = 28;
static constexpr ck::index_t Ho            = 28;
static constexpr ck::index_t Wo            = 28;

int main()
{
    return run_conv_bwd_data<NumDimSpatial,
                             InDataType,
                             WeiDataType,
                             OutDataType,
                             InLayout,
                             WeiLayout,
                             OutLayout>(N, K, C, {Di, Hi, Wi}, {Z, Y, X}, {Do, Ho, Wo})
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}
