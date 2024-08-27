// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"
#include "ck/ck.hpp"

using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using F32  = float;
using I32  = int32_t;
using ck::index_t;
using NDHWC = ck::tensor_layout::convolution::NDHWC;

struct PoolingParam
{
    PoolingParam(const std::vector<index_t>& length,
                 const std::vector<index_t>& window_spatial_lengths,
                 const std::vector<index_t>& window_strides,
                 const std::vector<index_t>& window_dilations,
                 const std::vector<index_t>& input_left_pads,
                 const std::vector<index_t>& input_right_pads)
        : length_(length),
          window_spatial_lengths_(window_spatial_lengths),
          window_strides_(window_strides),
          window_dilations_(window_dilations),
          input_left_pads_(input_left_pads),
          input_right_pads_(input_right_pads)
    {
    }
    std::vector<index_t> length_;
    std::vector<index_t> window_spatial_lengths_;
    std::vector<index_t> window_strides_;
    std::vector<index_t> window_dilations_;
    std::vector<index_t> input_left_pads_;
    std::vector<index_t> input_right_pads_;
};
