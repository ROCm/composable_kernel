// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef CONV_COMMON_HPP
#define CONV_COMMON_HPP

#include "ck/ck.hpp"
#include "ck/library/utility/convolution_parameter.hpp"

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

// Layout transformation kernels for testing
namespace ref {
namespace layout_transform {

// Input transformation: GNCDHW <-> NDHWGC (supports grouped convolutions)
template <typename DataType>
__global__ void transform_input_GNCDHW_to_NDHWGC(const DataType* __restrict__ src,
                                                 DataType* __restrict__ dst,
                                                 ck::index_t G,
                                                 ck::index_t N,
                                                 ck::index_t C,
                                                 ck::index_t D,
                                                 ck::index_t H,
                                                 ck::index_t W)
{
    ck::index_t total = G * N * C * D * H * W;
    ck::index_t idx   = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < total)
    {
        // Calculate indices in GNCDHW layout (source)
        ck::index_t w = idx % W;
        ck::index_t h = (idx / W) % H;
        ck::index_t d = (idx / (W * H)) % D;
        ck::index_t c = (idx / (W * H * D)) % C;
        ck::index_t n = (idx / (W * H * D * C)) % N;
        ck::index_t g = idx / (W * H * D * C * N);

        // Calculate linear index in NDHWGC layout (destination)
        // NDHWGC: n*(D*H*W*G*C) + d*(H*W*G*C) + h*(W*G*C) + w*(G*C) + g*C + c
        ck::index_t dst_idx = (((((n * D + d) * H + h) * W + w) * G + g) * C + c);

        dst[dst_idx] = src[idx];
    }
}

template <typename DataType>
__global__ void transform_input_NDHWGC_to_GNCDHW(const DataType* __restrict__ src,
                                                 DataType* __restrict__ dst,
                                                 ck::index_t G,
                                                 ck::index_t N,
                                                 ck::index_t C,
                                                 ck::index_t D,
                                                 ck::index_t H,
                                                 ck::index_t W)
{
    ck::index_t total = G * N * C * D * H * W;
    ck::index_t idx   = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < total)
    {
        // Calculate indices in NDHWGC layout (source)
        ck::index_t c = idx % C;
        ck::index_t g = (idx / C) % G;
        ck::index_t w = (idx / (C * G)) % W;
        ck::index_t h = (idx / (C * G * W)) % H;
        ck::index_t d = (idx / (C * G * W * H)) % D;
        ck::index_t n = idx / (C * G * W * H * D);

        // Calculate linear index in GNCDHW layout (destination)
        // GNCDHW: g*(N*C*D*H*W) + n*(C*D*H*W) + c*(D*H*W) + d*(H*W) + h*W + w
        ck::index_t dst_idx = (((((g * N + n) * C + c) * D + d) * H + h) * W + w);

        dst[dst_idx] = src[idx];
    }
}

// Weight transformation: GKCZYX <-> KZYXGC (supports grouped convolutions)
template <typename DataType>
__global__ void transform_weight_GKCZYX_to_KZYXGC(const DataType* __restrict__ src,
                                                  DataType* __restrict__ dst,
                                                  ck::index_t G,
                                                  ck::index_t K,
                                                  ck::index_t C,
                                                  ck::index_t Z,
                                                  ck::index_t Y,
                                                  ck::index_t X)
{
    ck::index_t total = G * K * C * Z * Y * X;
    ck::index_t idx   = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < total)
    {
        // Calculate indices in GKCZYX layout (source)
        ck::index_t x = idx % X;
        ck::index_t y = (idx / X) % Y;
        ck::index_t z = (idx / (X * Y)) % Z;
        ck::index_t c = (idx / (X * Y * Z)) % C;
        ck::index_t k = (idx / (X * Y * Z * C)) % K;
        ck::index_t g = idx / (X * Y * Z * C * K);

        // Calculate linear index in KZYXGC layout (destination)
        // KZYXGC: k*(Z*Y*X*G*C) + z*(Y*X*G*C) + y*(X*G*C) + x*(G*C) + g*C + c
        ck::index_t dst_idx = (((((k * Z + z) * Y + y) * X + x) * G + g) * C + c);

        dst[dst_idx] = src[idx];
    }
}

template <typename DataType>
__global__ void transform_weight_KZYXGC_to_GKCZYX(const DataType* __restrict__ src,
                                                  DataType* __restrict__ dst,
                                                  ck::index_t G,
                                                  ck::index_t K,
                                                  ck::index_t C,
                                                  ck::index_t Z,
                                                  ck::index_t Y,
                                                  ck::index_t X)
{
    ck::index_t total = G * K * C * Z * Y * X;
    ck::index_t idx   = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < total)
    {
        // Calculate indices in KZYXGC layout (source)
        ck::index_t c = idx % C;
        ck::index_t g = (idx / C) % G;
        ck::index_t x = (idx / (C * G)) % X;
        ck::index_t y = (idx / (C * G * X)) % Y;
        ck::index_t z = (idx / (C * G * X * Y)) % Z;
        ck::index_t k = idx / (C * G * X * Y * Z);

        // Calculate linear index in GKCZYX layout (destination)
        // GKCZYX: g*(K*C*Z*Y*X) + k*(C*Z*Y*X) + c*(Z*Y*X) + z*(Y*X) + y*X + x
        ck::index_t dst_idx = (((((g * K + k) * C + c) * Z + z) * Y + y) * X + x);

        dst[dst_idx] = src[idx];
    }
}

// Output transformation: GNKDHW <-> NDHWGK (supports grouped convolutions)
template <typename DataType>
__global__ void transform_output_GNKDHW_to_NDHWGK(const DataType* __restrict__ src,
                                                  DataType* __restrict__ dst,
                                                  ck::index_t G,
                                                  ck::index_t N,
                                                  ck::index_t K,
                                                  ck::index_t D,
                                                  ck::index_t H,
                                                  ck::index_t W)
{
    ck::index_t total = G * N * K * D * H * W;
    ck::index_t idx   = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < total)
    {
        // Calculate indices in GNKDHW layout (source)
        ck::index_t w = idx % W;
        ck::index_t h = (idx / W) % H;
        ck::index_t d = (idx / (W * H)) % D;
        ck::index_t k = (idx / (W * H * D)) % K;
        ck::index_t n = (idx / (W * H * D * K)) % N;
        ck::index_t g = idx / (W * H * D * K * N);

        // Calculate linear index in NDHWGK layout (destination)
        // NDHWGK: n*(D*H*W*G*K) + d*(H*W*G*K) + h*(W*G*K) + w*(G*K) + g*K + k
        ck::index_t dst_idx = (((((n * D + d) * H + h) * W + w) * G + g) * K + k);

        dst[dst_idx] = src[idx];
    }
}

template <typename DataType>
__global__ void transform_output_NDHWGK_to_GNKDHW(const DataType* __restrict__ src,
                                                  DataType* __restrict__ dst,
                                                  ck::index_t G,
                                                  ck::index_t N,
                                                  ck::index_t K,
                                                  ck::index_t D,
                                                  ck::index_t H,
                                                  ck::index_t W)
{
    ck::index_t total = G * N * K * D * H * W;
    ck::index_t idx   = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < total)
    {
        // Calculate indices in NDHWGK layout (source)
        ck::index_t k = idx % K;
        ck::index_t g = (idx / K) % G;
        ck::index_t w = (idx / (K * G)) % W;
        ck::index_t h = (idx / (K * G * W)) % H;
        ck::index_t d = (idx / (K * G * W * H)) % D;
        ck::index_t n = idx / (K * G * W * H * D);

        // Calculate linear index in GNKDHW layout (destination)
        // GNKDHW: g*(N*K*D*H*W) + n*(K*D*H*W) + k*(D*H*W) + d*(H*W) + h*W + w
        ck::index_t dst_idx = (((((g * N + n) * K + k) * D + d) * H + h) * W + w);

        dst[dst_idx] = src[idx];
    }
}

} // namespace layout_transform
} // namespace ref
} // namespace ck

#endif
