// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include <vector>

namespace ck {
namespace test {

template <ck::index_t NDimSpatial>
struct ConvParams
{
    ck::index_t N, K, C, G;
    std::vector<ck::index_t> input_spatial;
    std::vector<ck::index_t> filter_spatial;
    std::vector<ck::index_t> output_spatial;
    std::vector<ck::index_t> strides;
    std::vector<ck::index_t> dilations;
    std::vector<ck::index_t> pads;
};

// Common test shapes for all convolution tests (fwd, bwd_data, bwd_weight)
namespace conv_test_shapes {

// 2D Conv, FP16, Small
inline ConvParams<2> get_2d_small()
{
    ConvParams<2> params;
    params.N              = 2;
    params.K              = 8;
    params.C              = 8;
    params.G              = 1;
    params.input_spatial  = {7, 7};
    params.filter_spatial = {3, 3};
    params.output_spatial = {5, 5};
    params.strides        = {1, 1};
    params.dilations      = {1, 1};
    params.pads           = {0, 0};
    return params;
}

// 2D Conv, FP32, Medium
inline ConvParams<2> get_2d_medium()
{
    ConvParams<2> params;
    params.N              = 4;
    params.K              = 16;
    params.C              = 16;
    params.G              = 1;
    params.input_spatial  = {14, 14};
    params.filter_spatial = {3, 3};
    params.output_spatial = {12, 12};
    params.strides        = {1, 1};
    params.dilations      = {1, 1};
    params.pads           = {0, 0};
    return params;
}

// 1D Conv, FP16
inline ConvParams<1> get_1d()
{
    ConvParams<1> params;
    params.N              = 2;
    params.K              = 8;
    params.C              = 8;
    params.G              = 1;
    params.input_spatial  = {16};
    params.filter_spatial = {3};
    params.output_spatial = {14};
    params.strides        = {1};
    params.dilations      = {1};
    params.pads           = {0};
    return params;
}

// 3D Conv, FP16, Small
inline ConvParams<3> get_3d_small()
{
    ConvParams<3> params;
    params.N              = 1;
    params.K              = 8;
    params.C              = 8;
    params.G              = 1;
    params.input_spatial  = {5, 5, 5};
    params.filter_spatial = {3, 3, 3};
    params.output_spatial = {3, 3, 3};
    params.strides        = {1, 1, 1};
    params.dilations      = {1, 1, 1};
    params.pads           = {0, 0, 0};
    return params;
}

// 2D Conv with stride
inline ConvParams<2> get_2d_stride2()
{
    ConvParams<2> params;
    params.N              = 2;
    params.K              = 8;
    params.C              = 8;
    params.G              = 1;
    params.input_spatial  = {8, 8};
    params.filter_spatial = {3, 3};
    params.output_spatial = {3, 3};
    params.strides        = {2, 2};
    params.dilations      = {1, 1};
    params.pads           = {0, 0};
    return params;
}

// 2D Grouped Conv, FP16, G=2
inline ConvParams<2> get_2d_grouped_g2()
{
    ConvParams<2> params;
    params.N              = 2;
    params.K              = 8;  // 8 total output channels
    params.C              = 16; // 16 total input channels (8 per group with G=2)
    params.G              = 2;
    params.input_spatial  = {7, 7};
    params.filter_spatial = {3, 3};
    params.output_spatial = {5, 5};
    params.strides        = {1, 1};
    params.dilations      = {1, 1};
    params.pads           = {0, 0};
    return params;
}

// 2D Grouped Conv, FP32, G=4
inline ConvParams<2> get_2d_grouped_g4()
{
    ConvParams<2> params;
    params.N              = 1;
    params.K              = 16; // 16 total output channels
    params.C              = 16; // 16 total input channels (4 per group with G=4)
    params.G              = 4;
    params.input_spatial  = {8, 8};
    params.filter_spatial = {3, 3};
    params.output_spatial = {6, 6};
    params.strides        = {1, 1};
    params.dilations      = {1, 1};
    params.pads           = {0, 0};
    return params;
}

} // namespace conv_test_shapes
} // namespace test
} // namespace ck
