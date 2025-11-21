// Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck/library/utility/convolution_parameter.hpp"

namespace ck_tile::builder::test {

struct FilterExtent
{
    ck::index_t width  = 1;
    ck::index_t height = 1;
    ck::index_t depth  = 1;

    template <int SPATIAL_DIM>
    std::vector<ck::index_t> to_vector() const
    {
        if constexpr(SPATIAL_DIM == 1)
        {
            return {std::initializer_list<ck::index_t>{this->width}};
        }
        else if constexpr(SPATIAL_DIM == 2)
        {
            return {{this->height, this->width}};
        }
        else if constexpr(SPATIAL_DIM == 3)
        {
            return {{this->depth, this->height, this->width}};
        }
    }
};

template <int SPATIAL_DIM>
std::array<ck::index_t, SPATIAL_DIM + 3> to_ck_lengths(const std::array<ck::index_t, 3>& gnc,
                                                       const FilterExtent& whd)
{
    std::array<ck::index_t, SPATIAL_DIM + 3> result = {0};
    result[0]                                       = gnc[0];
    result[1]                                       = gnc[1];
    result[2]                                       = gnc[2];

    if constexpr(SPATIAL_DIM == 1)
    {
        result[3] = whd.width;
    }
    else if constexpr(SPATIAL_DIM == 2)
    {
        result[3] = whd.height;
        result[4] = whd.width;
    }
    else if constexpr(SPATIAL_DIM == 3)
    {
        result[3] = whd.depth;
        result[4] = whd.height;
        result[5] = whd.width;
    }

    return result;
}

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

    ck::utils::conv::ConvParam to_conv_param() const
    {
        return ck::utils::conv::ConvParam(SPATIAL_DIM,
                                          this->lengths.groups,
                                          this->lengths.batch_size,
                                          this->lengths.output_channels,
                                          this->lengths.input_channels,
                                          this->lengths.filter.to_vector<SPATIAL_DIM>(),
                                          this->lengths.image.to_vector<SPATIAL_DIM>(),
                                          this->filter_strides.to_vector<SPATIAL_DIM>(),
                                          this->filter_dilation.to_vector<SPATIAL_DIM>(),
                                          this->input_left_pad.to_vector<SPATIAL_DIM>(),
                                          this->input_right_pad.to_vector<SPATIAL_DIM>(), );
    }
};

} // namespace ck_tile::builder::test
