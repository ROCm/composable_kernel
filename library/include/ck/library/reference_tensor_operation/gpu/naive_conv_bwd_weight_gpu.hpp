// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/type_convert.hpp"
#include "ck/library/reference_tensor_operation/gpu/conv_common.hpp"

namespace ck {
namespace ref {

/*
 * \brief naive implementation of 3D convolution backward weight.
 *        Layout is (NDHWC, KZYXC, NDHWK).
 *        Computes gradient with respect to weights.
 *
 * \param N number of batches
 * \param K number of filters (output channels)
 * \param C number of input channels
 * \param (Di, Hi, Wi) depth, height and width dimension of input
 * \param (Z, Y, X) depth, height and width dimensions of filter
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
__global__ void naive_conv_bwd_weight_ndhwc_kzyxc_ndhwk(const TIn* __restrict__ p_in,
                                                        TWei* __restrict__ p_wei_grad,
                                                        const TOut* __restrict__ p_out_grad,
                                                        const ConvDims dims)
{
    const index_t tid                = blockIdx.x * blockDim.x + threadIdx.x;
    const index_t num_threads        = blockDim.x * gridDim.x;
    const long_index_t weight_length = dims.K * dims.Z * dims.Y * dims.X * dims.C;

    const index_t in_strides[] = {
        dims.Di * dims.Hi * dims.Wi * dims.C, dims.Hi * dims.Wi * dims.C, dims.Wi * dims.C, dims.C};
    const index_t out_strides[] = {
        dims.Do * dims.Ho * dims.Wo * dims.K, dims.Ho * dims.Wo * dims.K, dims.Wo * dims.K, dims.K};
    const index_t wei_strides[] = {
        dims.Z * dims.Y * dims.X * dims.C, dims.Y * dims.X * dims.C, dims.X * dims.C, dims.C};

    constexpr auto in_op  = InElementwiseOperation{};
    constexpr auto wei_op = WeiElementwiseOperation{};
    constexpr auto out_op = OutElementwiseOperation{};

    TIn in_val   = TIn{0};
    TWei wei_val = TWei{0};
    TOut out_val = TOut{0};

    for(long_index_t ii = tid; ii < weight_length; ii += num_threads)
    {
        // Decode linear index to (k, z, y, x, c)
        const index_t k = ii / wei_strides[0];
        index_t tmp     = ii - k * wei_strides[0];
        const index_t z = tmp / wei_strides[1];
        tmp -= z * wei_strides[1];
        const index_t y = tmp / wei_strides[2];
        tmp -= y * wei_strides[2];
        const index_t x = tmp / wei_strides[3];
        tmp -= x * wei_strides[3];
        const index_t c = tmp;

        // Always accumulate in float
        float acc_float = 0.0f;

        // Loop over batch
        for(index_t n = 0; n < dims.N; ++n)
        {
            const TIn* in_n   = p_in + static_cast<long_index_t>(n) * in_strides[0];
            const TOut* out_n = p_out_grad + static_cast<long_index_t>(n) * out_strides[0];

            // Loop over output spatial dimensions
            for(index_t d_o = 0; d_o < dims.Do; ++d_o)
            {
                // Calculate input position from output position
                index_t di = d_o * dims.stride_z - dims.pad_z + z * dims.dilation_z;
                if(di < 0 || di >= dims.Di)
                    continue;

                const TIn* in_n_di   = in_n + di * in_strides[1];
                const TOut* out_n_do = out_n + d_o * out_strides[1];

                for(index_t ho = 0; ho < dims.Ho; ++ho)
                {
                    index_t hi = ho * dims.stride_y - dims.pad_y + y * dims.dilation_y;
                    if(hi < 0 || hi >= dims.Hi)
                        continue;

                    const TIn* in_n_di_hi   = in_n_di + hi * in_strides[2];
                    const TOut* out_n_do_ho = out_n_do + ho * out_strides[2];

                    for(index_t wo = 0; wo < dims.Wo; ++wo)
                    {
                        index_t wi = wo * dims.stride_x - dims.pad_x + x * dims.dilation_x;
                        if(wi < 0 || wi >= dims.Wi)
                            continue;

                        // Load values from memory (like forward does)
                        const TIn* in_ptr   = in_n_di_hi + wi * in_strides[3];
                        const TOut* out_ptr = out_n_do_ho + wo * out_strides[3];

                        TIn in_loaded   = in_ptr[c];
                        TOut out_loaded = out_ptr[k];

                        // Apply element-wise operations
                        in_op(in_val, in_loaded);
                        out_op(out_val, out_loaded);

                        // Convert to float for multiplication
                        float in_f  = type_convert<float>(in_val);
                        float out_f = type_convert<float>(out_val);

                        acc_float += out_f * in_f;
                    }
                }
            }
        }

        // Convert float accumulator to TAcc, then to weight type
        TAcc acc    = type_convert<TAcc>(acc_float);
        TWei result = type_convert<TWei>(acc);

        // Apply weight element-wise operation (if any)
        wei_op(wei_val, result);

        // Write transformed result
        p_wei_grad[ii] = wei_val;
    }
}
} // namespace ref
} // namespace ck
