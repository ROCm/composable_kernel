// Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/conv_signature_concepts.hpp"

namespace ck_tile::builder::test {

struct FilterExtent
{
    ck::index_t width  = 1;
    ck::index_t height = 1;
    ck::index_t depth  = 1;
};

struct TensorExtent
{
    ck::index_t batch_size      = 1;  // N
    ck::index_t groups          = 1;  // G
    ck::index_t input_channels  = 1;  // C
    ck::index_t output_channels = 1;  // K
    FilterExtent image          = {}; // W, H, D
    FilterExtent filter         = {}; // X, Y, Z
};

template <auto SIGNATURE>
    requires ValidConvSignature<SIGNATURE>
struct ConvArgs
{
    constexpr static auto SPATIAL_DIM = SIGNATURE.spatial_dim;

    TensorExtent lengths;
    // TODO(Robin): Tensor strides
    // TODO(Robin): D tensor strides

    // TODO(Robin): Defaults??
    FilterExtent filter_strides;
    FilterExtent filter_dilation;
    FilterExtent input_left_pad;
    FilterExtent input_right_pad;
};

} // namespace ck_tile::builder::test
