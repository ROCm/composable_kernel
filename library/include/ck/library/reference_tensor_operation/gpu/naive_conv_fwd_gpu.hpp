// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/type_convert.hpp"
#include "ck/library/reference_tensor_operation/gpu/conv_common.hpp"
#include "ck/library/reference_tensor_operation/gpu/layout_utils.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

namespace ck {
namespace ref {

/*
 * \brief naive implementation of 3D convolution. Layout is (NDHWC, KZYXC, NDHWK).
 *
 * \param N number of batches
 * \param K number of filters
 * \param C number of channels of weight
 * \param (Di, Hi, Wi) depth, height and width dimension of data
 * \param (Z, Y, X) depth, height and width dimensions of weights
 * \param (Do, Ho, Wo) depth, height and width dimension of output
 * \param (stride_z, stride_y, stride_x) strides
 * \param (dilation_z, dilation_y, dilation_x) dilations
 * \param (pad_z, pad_y, pad_x) pads
 */
template <typename TIn,
          typename TWei,
          typename TOut,
          typename TAcc,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation>
__global__ void naive_conv_fwd_ndhwc_kzyxc_ndhwk(const TIn* __restrict__ p_in,
                                                 const TWei* __restrict__ p_wei,
                                                 TOut* __restrict__ p_out,
                                                 const ConvDims dims)
{
    const index_t tid                = blockIdx.x * blockDim.x + threadIdx.x;
    const index_t num_threads        = blockDim.x * gridDim.x;
    const long_index_t output_length = dims.N * dims.Do * dims.Ho * dims.Wo * dims.K;

    const index_t out_strides[] = {
        dims.Do * dims.Ho * dims.Wo * dims.K, dims.Ho * dims.Wo * dims.K, dims.Wo * dims.K, dims.K};
    const index_t in_strides[] = {
        dims.Di * dims.Hi * dims.Wi * dims.C, dims.Hi * dims.Wi * dims.C, dims.Wi * dims.C, dims.C};
    const index_t wei_strides[] = {
        dims.Z * dims.Y * dims.X * dims.C, dims.Y * dims.X * dims.C, dims.X * dims.C, dims.C};

    constexpr auto in_op  = InElementwiseOperation{};
    constexpr auto wei_op = WeiElementwiseOperation{};
    constexpr auto out_op = OutElementwiseOperation{};

    TIn in_val   = TIn{0};
    TWei wei_val = TWei{0};
    TOut out_val = TOut{0};

    for(long_index_t ii = tid; ii < output_length; ii += num_threads)
    {
        const index_t n  = ii / out_strides[0];
        index_t k_flat   = ii - n * out_strides[0];
        const index_t dO = k_flat / out_strides[1];
        k_flat -= dO * out_strides[1];
        const index_t ho = k_flat / out_strides[2];
        k_flat -= ho * out_strides[2];
        const index_t wo = k_flat / out_strides[3];
        k_flat -= wo * out_strides[3];

        // Always accumulate in float (FP8/BF8 don't support arithmetic)
        float acc_float = 0.0f;

        const TIn* in_n = p_in + static_cast<long_index_t>(n) * in_strides[0];

        // For grouped convolutions: decompose flat output channel into group and per-group channel
        const index_t K_per_group = dims.K / dims.G;
        const index_t C_per_group = dims.C / dims.G;
        const index_t group       = k_flat / K_per_group;
        const index_t k           = k_flat % K_per_group;
        const index_t c_start     = group * C_per_group;
        const index_t c_end       = c_start + C_per_group;

        // Weight layout is KZYXGC: k*(Z*Y*X*G*C) + z*(Y*X*G*C) + y*(X*G*C) + x*(G*C) + g*C + c
        // Stride for k is Z*Y*X*G*C_per_group = Z*Y*X*C_total
        const TWei* wei_k = p_wei + static_cast<long_index_t>(k) * wei_strides[0];

        for(index_t z = 0; z < dims.Z; ++z)
        {
            index_t di          = dims.stride_z * dO - dims.pad_z + dims.dilation_z * z;
            const TIn* in_n_di  = in_n + di * in_strides[1];
            const TWei* wei_k_z = wei_k + z * wei_strides[1];

            for(index_t y = 0; y < dims.Y; ++y)
            {
                index_t hi            = dims.stride_y * ho - dims.pad_y + dims.dilation_y * y;
                const TIn* in_n_di_hi = in_n_di + hi * in_strides[2];
                const TWei* wei_k_z_y = wei_k_z + y * wei_strides[2];

                for(index_t x = 0; x < dims.X; ++x)
                {
                    index_t wi = dims.stride_x * wo - dims.pad_x + dims.dilation_x * x;
                    const TIn* in_n_di_hi_wi = in_n_di_hi + wi * in_strides[3];
                    const TWei* wei_k_z_y_x  = wei_k_z_y + x * wei_strides[3];

                    if(di >= 0 && di < dims.Di && hi >= 0 && hi < dims.Hi && wi >= 0 &&
                       wi < dims.Wi)
                    {
                        // Only iterate over channels in this group
                        for(index_t c = c_start; c < c_end; ++c)
                        {
                            // Load values from memory
                            TIn in_loaded = in_n_di_hi_wi[c];
                            // Weight layout is KZYXGC: need to offset by group*C_per_group +
                            // local_c
                            index_t c_local = c - c_start;
                            TWei wei_loaded = wei_k_z_y_x[group * C_per_group + c_local];

                            // Apply element-wise operations
                            in_op(in_val, in_loaded);
                            wei_op(wei_val, wei_loaded);

                            // Always convert to float for multiplication (FP8/BF8 don't support
                            // direct arithmetic)
                            float in_f  = type_convert<float>(in_val);
                            float wei_f = type_convert<float>(wei_val);

                            // Accumulate in float
                            acc_float += in_f * wei_f;
                        }
                    }
                }
            }
        }

        // Convert float accumulator to TAcc, then to output type
        TAcc acc    = type_convert<TAcc>(acc_float);
        TOut result = type_convert<TOut>(acc);

        // Apply output element-wise operation (if any)
        out_op(out_val, result);

        // Write transformed result
        p_out[ii] = out_val;
    }
}
/*
 * \brief Layout-aware wrapper for naive forward convolution
 *
 * Automatically handles transformations between user-specified layouts and the
 * naive kernel's internal layout (NDHWGC, KZYXGC, NDHWGK).
 *
 * Template parameters specify the desired layouts using types from
 * ck::tensor_layout::convolution namespace.
 *
 * Example usage:
 *   conv_fwd_with_layouts<GNCDHW, GKCZYX, GNKDHW>(...);
 */
template <typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename TIn,
          typename TWei,
          typename TOut,
          typename TAcc,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation>
void conv_fwd_with_layouts(const TIn* p_in,
                           const TWei* p_wei,
                           TOut* p_out,
                           const ConvDims dims,
                           index_t NDimSpatial,
                           hipStream_t stream = nullptr)
{
    static_assert(
        std::is_base_of<tensor_layout::convolution::BaseConvolutionLayout, InLayout>::value,
        "InLayout must derive from BaseConvolutionLayout");
    static_assert(
        std::is_base_of<tensor_layout::convolution::BaseConvolutionLayout, WeiLayout>::value,
        "WeiLayout must derive from BaseConvolutionLayout");
    static_assert(
        std::is_base_of<tensor_layout::convolution::BaseConvolutionLayout, OutLayout>::value,
        "OutLayout must derive from BaseConvolutionLayout");

    using namespace layout_utils;

    // Step 1: Determine dimension orderings and permutations
    auto user_input_order  = InputLayoutTrait<InLayout>::dim_order();
    auto user_weight_order = WeightLayoutTrait<WeiLayout>::dim_order();
    auto user_output_order = OutputLayoutTrait<OutLayout>::dim_order();

    std::vector<int> naive_input_order, naive_weight_order, naive_output_order;
    if(NDimSpatial == 3)
    {
        naive_input_order  = get_naive_input_order_3d();
        naive_weight_order = get_naive_weight_order_3d();
        naive_output_order = get_naive_output_order_3d();
    }
    else if(NDimSpatial == 2)
    {
        naive_input_order  = get_naive_input_order_2d();
        naive_weight_order = get_naive_weight_order_2d();
        naive_output_order = get_naive_output_order_2d();
    }
    else // 1D
    {
        naive_input_order  = get_naive_input_order_1d();
        naive_weight_order = get_naive_weight_order_1d();
        naive_output_order = get_naive_output_order_1d();
    }

    // Compute permutations
    auto input_to_naive  = compute_permutation(user_input_order, naive_input_order);
    auto weight_to_naive = compute_permutation(user_weight_order, naive_weight_order);
    auto naive_to_output = compute_permutation(naive_output_order, user_output_order);

    // Step 2: Check if transformations are needed
    bool needs_input_xform  = (user_input_order != naive_input_order);
    bool needs_weight_xform = (user_weight_order != naive_weight_order);
    bool needs_output_xform = (naive_output_order != user_output_order);

    // Step 3: Allocate temporary buffers if needed
    TIn* p_in_naive   = nullptr;
    TWei* p_wei_naive = nullptr;
    TOut* p_out_naive = nullptr;

    if(needs_input_xform)
    {
        size_t in_size = dims.N * dims.Di * dims.Hi * dims.Wi * dims.C * sizeof(TIn);
        HIP_CHECK_ERROR(hipMalloc(&p_in_naive, in_size));
    }
    else
    {
        p_in_naive = const_cast<TIn*>(p_in);
    }

    if(needs_weight_xform)
    {
        size_t wei_size = dims.K * dims.Z * dims.Y * dims.X * dims.C * sizeof(TWei);
        HIP_CHECK_ERROR(hipMalloc(&p_wei_naive, wei_size));
    }
    else
    {
        p_wei_naive = const_cast<TWei*>(p_wei);
    }

    if(needs_output_xform)
    {
        size_t out_size = dims.N * dims.Do * dims.Ho * dims.Wo * dims.K * sizeof(TOut);
        HIP_CHECK_ERROR(hipMalloc(&p_out_naive, out_size));
    }
    else
    {
        p_out_naive = p_out;
    }

    // Step 4: Transform inputs to naive format
    if(needs_input_xform)
    {
        std::vector<index_t> input_dims = build_input_dims(dims, NDimSpatial);
        layout_transform::launch_generic_transpose<TIn>(
            p_in, p_in_naive, input_dims, input_to_naive, stream);
    }

    if(needs_weight_xform)
    {
        std::vector<index_t> weight_dims = build_weight_dims(dims, NDimSpatial);
        layout_transform::launch_generic_transpose<TWei>(
            p_wei, p_wei_naive, weight_dims, weight_to_naive, stream);
    }

    // Step 5: Call naive kernel
    constexpr int block_size   = 256;
    long_index_t output_length = dims.N * dims.Do * dims.Ho * dims.Wo * dims.K;
    int grid_size              = (output_length + block_size - 1) / block_size;

    hipLaunchKernelGGL((naive_conv_fwd_ndhwc_kzyxc_ndhwk<TIn,
                                                         TWei,
                                                         TOut,
                                                         TAcc,
                                                         InElementwiseOperation,
                                                         WeiElementwiseOperation,
                                                         OutElementwiseOperation>),
                       dim3(grid_size),
                       dim3(block_size),
                       0,
                       stream,
                       p_in_naive,
                       p_wei_naive,
                       p_out_naive,
                       dims);

    // Step 6: Transform output back to user layout
    if(needs_output_xform)
    {
        std::vector<index_t> output_dims = build_naive_output_dims(dims, NDimSpatial);
        layout_transform::launch_generic_transpose<TOut>(
            p_out_naive, p_out, output_dims, naive_to_output, stream);
    }

    // Step 7: Free temporary buffers
    if(needs_input_xform)
    {
        HIP_CHECK_ERROR(hipFree(p_in_naive));
    }
    if(needs_weight_xform)
    {
        HIP_CHECK_ERROR(hipFree(p_wei_naive));
    }
    if(needs_output_xform)
    {
        HIP_CHECK_ERROR(hipFree(p_out_naive));
    }
}

} // namespace ref
} // namespace ck
