// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef CONV_COMMON_HPP
#define CONV_COMMON_HPP

#include "ck/ck.hpp"

namespace ck {
namespace ref {

// Device-compatible dimension structure for GPU reference kernels
// Replaces passing 24 individual parameters
struct ConvDims
{
    index_t N, K, C;
    index_t Di, Hi, Wi;
    index_t Z, Y, X;
    index_t Do, Ho, Wo;
    index_t stride_z, stride_y, stride_x;
    index_t dilation_z, dilation_y, dilation_x;
    index_t pad_z, pad_y, pad_x;
};

} // namespace ref

// Helper function to extract dimensions from ConvParam for GPU kernels
// Defined in ck::utils::conv namespace for convenience
namespace utils {
namespace conv {

inline ck::ref::ConvDims
extract_conv_dims(const ConvParam& conv_param, ck::index_t NDimSpatial, bool apply_group = true)
{
    ck::ref::ConvDims dims;
    dims.N = conv_param.N_;
    dims.K = conv_param.K_;
    dims.C = apply_group ? (conv_param.C_ * conv_param.G_) : conv_param.C_;

    dims.Di = (NDimSpatial >= 3) ? conv_param.input_spatial_lengths_[0] : 1;
    dims.Hi = (NDimSpatial >= 2) ? conv_param.input_spatial_lengths_[NDimSpatial >= 3 ? 1 : 0] : 1;
    dims.Wi = conv_param.input_spatial_lengths_[NDimSpatial - 1];

    dims.Z = (NDimSpatial >= 3) ? conv_param.filter_spatial_lengths_[0] : 1;
    dims.Y = (NDimSpatial >= 2) ? conv_param.filter_spatial_lengths_[NDimSpatial >= 3 ? 1 : 0] : 1;
    dims.X = conv_param.filter_spatial_lengths_[NDimSpatial - 1];

    dims.Do = (NDimSpatial >= 3) ? conv_param.output_spatial_lengths_[0] : 1;
    dims.Ho = (NDimSpatial >= 2) ? conv_param.output_spatial_lengths_[NDimSpatial >= 3 ? 1 : 0] : 1;
    dims.Wo = conv_param.output_spatial_lengths_[NDimSpatial - 1];

    dims.stride_z = (NDimSpatial >= 3) ? conv_param.conv_filter_strides_[0] : 1;
    dims.stride_y =
        (NDimSpatial >= 2) ? conv_param.conv_filter_strides_[NDimSpatial >= 3 ? 1 : 0] : 1;
    dims.stride_x = conv_param.conv_filter_strides_[NDimSpatial - 1];

    dims.dilation_z = (NDimSpatial >= 3) ? conv_param.conv_filter_dilations_[0] : 1;
    dims.dilation_y =
        (NDimSpatial >= 2) ? conv_param.conv_filter_dilations_[NDimSpatial >= 3 ? 1 : 0] : 1;
    dims.dilation_x = conv_param.conv_filter_dilations_[NDimSpatial - 1];

    dims.pad_z = (NDimSpatial >= 3) ? conv_param.input_left_pads_[0] : 0;
    dims.pad_y = (NDimSpatial >= 2) ? conv_param.input_left_pads_[NDimSpatial >= 3 ? 1 : 0] : 0;
    dims.pad_x = conv_param.input_left_pads_[NDimSpatial - 1];

    return dims;
}

} // namespace conv
} // namespace utils
} // namespace ck

#endif
