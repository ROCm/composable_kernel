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
    index_t N, K, C, G; // Added G for grouped convolutions
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

// Generic transpose kernel using permutation array
// Permutation format: dst_dim[i] comes from src_dim[perm[i]]
// Example: GNCDHW -> NDHWGC uses perm = [1, 3, 4, 5, 0, 2]
//   dst[0]=src[1]=N, dst[1]=src[3]=D, dst[2]=src[4]=H,
//   dst[3]=src[5]=W, dst[4]=src[0]=G, dst[5]=src[2]=C
template <typename DataType>
__global__ void generic_transpose(const DataType* __restrict__ src,
                                  DataType* __restrict__ dst,
                                  const ck::index_t* dims, // Source dimension lengths [num_dims]
                                  const int* perm,         // Permutation [num_dims]
                                  int num_dims,
                                  ck::index_t total_elements)
{
    constexpr int MAX_DIMS = 8; // Support up to 8 dimensions

    ck::index_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= total_elements)
        return;

    // Decode source linear index to multi-dimensional indices
    ck::index_t src_indices[MAX_DIMS];
    ck::index_t tmp = idx;
    for(int i = num_dims - 1; i >= 0; --i)
    {
        src_indices[i] = tmp % dims[i];
        tmp /= dims[i];
    }

    // Apply permutation: dst_indices[i] = src_indices[perm[i]]
    // Also compute destination dimensions: dst_dims[i] = src_dims[perm[i]]
    ck::index_t dst_indices[MAX_DIMS];
    ck::index_t dst_dims[MAX_DIMS];
    for(int i = 0; i < num_dims; ++i)
    {
        dst_indices[i] = src_indices[perm[i]];
        dst_dims[i]    = dims[perm[i]];
    }

    // Encode destination multi-dimensional indices to linear index
    ck::index_t dst_idx = 0;
    ck::index_t stride  = 1;
    for(int i = num_dims - 1; i >= 0; --i)
    {
        dst_idx += dst_indices[i] * stride;
        stride *= dst_dims[i];
    }

    dst[dst_idx] = src[idx];
}

// Helper wrapper to launch generic transpose with device memory for arrays
template <typename DataType>
void launch_generic_transpose(const DataType* src,
                              DataType* dst,
                              const std::vector<ck::index_t>& dims,
                              const std::vector<int>& perm,
                              hipStream_t stream = nullptr)
{
    // Calculate total elements
    ck::index_t total = 1;
    for(auto d : dims)
        total *= d;

    // Allocate device memory for dims and perm arrays
    ck::index_t* d_dims;
    int* d_perm;
    (void)hipMalloc(&d_dims, dims.size() * sizeof(ck::index_t));
    (void)hipMalloc(&d_perm, perm.size() * sizeof(int));

    (void)hipMemcpy(d_dims, dims.data(), dims.size() * sizeof(ck::index_t), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_perm, perm.data(), perm.size() * sizeof(int), hipMemcpyHostToDevice);

    // Launch kernel
    constexpr int block_size = 256;
    int grid_size            = (total + block_size - 1) / block_size;

    hipLaunchKernelGGL(generic_transpose<DataType>,
                       dim3(grid_size),
                       dim3(block_size),
                       0,
                       stream,
                       src,
                       dst,
                       d_dims,
                       d_perm,
                       static_cast<int>(dims.size()),
                       total);

    // Free device memory
    (void)hipFree(d_dims);
    (void)hipFree(d_perm);
}

} // namespace layout_transform
} // namespace ref
} // namespace ck

#endif
