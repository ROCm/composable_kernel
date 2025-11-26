// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/type_convert.hpp"
#include "ck/library/reference_tensor_operation/gpu/conv_common.hpp"

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
        index_t k        = ii - n * out_strides[0];
        const index_t dO = k / out_strides[1];
        k -= dO * out_strides[1];
        const index_t ho = k / out_strides[2];
        k -= ho * out_strides[2];
        const index_t wo = k / out_strides[3];
        k -= wo * out_strides[3];

        // Always accumulate in float (FP8/BF8 don't support arithmetic)
        float acc_float = 0.0f;

        const TIn* in_n   = p_in + static_cast<long_index_t>(n) * in_strides[0];
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
                        for(index_t c = 0; c < dims.C; ++c)
                        {
                            // Load values from memory
                            TIn in_loaded   = in_n_di_hi_wi[c];
                            TWei wei_loaded = wei_k_z_y_x[c];

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
} // namespace ref
} // namespace ck
