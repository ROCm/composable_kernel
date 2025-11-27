// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/type_convert.hpp"
#include "ck/library/reference_tensor_operation/gpu/conv_common.hpp"

namespace ck {
namespace ref {

/*
 * \brief naive implementation of 3D convolution backward data.
 *        Layout is (NDHWC, KZYXC, NDHWK).
 *        Computes gradient with respect to input.
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
__global__ void naive_conv_bwd_data_ndhwc_kzyxc_ndhwk(TIn* __restrict__ p_in_grad,
                                                      const TWei* __restrict__ p_wei,
                                                      const TOut* __restrict__ p_out_grad,
                                                      const ConvDims dims)
{
    const index_t tid               = blockIdx.x * blockDim.x + threadIdx.x;
    const index_t num_threads       = blockDim.x * gridDim.x;
    const long_index_t input_length = dims.N * dims.Di * dims.Hi * dims.Wi * dims.C;

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

    for(long_index_t ii = tid; ii < input_length; ii += num_threads)
    {
        // Decode linear index to (n, di, hi, wi, c)
        const index_t n  = ii / in_strides[0];
        index_t tmp      = ii - n * in_strides[0];
        const index_t di = tmp / in_strides[1];
        tmp -= di * in_strides[1];
        const index_t hi = tmp / in_strides[2];
        tmp -= hi * in_strides[2];
        const index_t wi = tmp / in_strides[3];
        tmp -= wi * in_strides[3];
        const index_t c = tmp;

        // Always accumulate in float
        float acc_float = 0.0f;

        const TOut* out_n = p_out_grad + static_cast<long_index_t>(n) * out_strides[0];

        // Loop over output channels
        for(index_t k = 0; k < dims.K; ++k)
        {
            const TWei* wei_k = p_wei + static_cast<long_index_t>(k) * wei_strides[0];

            // Loop over filter dimensions
            for(index_t z = 0; z < dims.Z; ++z)
            {
                // Calculate output position from input position (inverse of forward)
                index_t d_tmp = di + dims.pad_z - z * dims.dilation_z;
                if(d_tmp % dims.stride_z != 0)
                    continue;
                index_t d_o = d_tmp / dims.stride_z;
                if(d_o < 0 || d_o >= dims.Do)
                    continue;

                const TOut* out_n_do = out_n + d_o * out_strides[1];
                const TWei* wei_k_z  = wei_k + z * wei_strides[1];

                for(index_t y = 0; y < dims.Y; ++y)
                {
                    index_t h_tmp = hi + dims.pad_y - y * dims.dilation_y;
                    if(h_tmp % dims.stride_y != 0)
                        continue;
                    index_t ho = h_tmp / dims.stride_y;
                    if(ho < 0 || ho >= dims.Ho)
                        continue;

                    const TOut* out_n_do_ho = out_n_do + ho * out_strides[2];
                    const TWei* wei_k_z_y   = wei_k_z + y * wei_strides[2];

                    for(index_t x = 0; x < dims.X; ++x)
                    {
                        index_t w_tmp = wi + dims.pad_x - x * dims.dilation_x;
                        if(w_tmp % dims.stride_x != 0)
                            continue;
                        index_t wo = w_tmp / dims.stride_x;
                        if(wo < 0 || wo >= dims.Wo)
                            continue;

                        const TOut* out_n_do_ho_wo = out_n_do_ho + wo * out_strides[3];
                        const TWei* wei_k_z_y_x    = wei_k_z_y + x * wei_strides[3];

                        // Load values from memory
                        TOut out_loaded = out_n_do_ho_wo[k];
                        TWei wei_loaded = wei_k_z_y_x[c];

                        // Apply element-wise operations (like forward does)
                        out_op(out_val, out_loaded);
                        wei_op(wei_val, wei_loaded);

                        // Convert to float for multiplication
                        float out_f = type_convert<float>(out_val);
                        float wei_f = type_convert<float>(wei_val);

                        acc_float += out_f * wei_f;
                    }
                }
            }
        }

        // Convert float accumulator to TAcc, then to input type
        TAcc acc   = type_convert<TAcc>(acc_float);
        TIn result = type_convert<TIn>(acc);

        // Apply input element-wise operation (if any)
        in_op(in_val, result);

        // Write transformed result
        p_in_grad[ii] = in_val;
    }
}
} // namespace ref
} // namespace ck
