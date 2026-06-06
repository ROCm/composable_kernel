// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/type_convert.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/reference_tensor_operation/gpu/naive_conv_utils.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include <array>

namespace ck {
namespace ref {

// Optimized backward weight convolution kernel working with packed (contiguous) tensors with
// multi-ABD support
// Assumes row-major packing: input[G][N][C][spatial], output_grad[G][N][K][spatial],
// weight_grad[G][K][C][filter]
// Computes gradient with respect to weights
template <index_t NDimSpatial,
          index_t NumAExtra, // Number of extra A (input) tensors
          index_t NumBExtra, // Number of extra B (output gradient) tensors
          index_t NumD,      // Number of D tensors
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename DDataType, // D tensor data type
          typename InElementOp,
          typename WeiElementOp,
          typename OutElementOp>
__global__ void
naive_conv_bwd_weight_packed_multi_abd(const InDataType* const* __restrict__ p_ins,
                                       WeiDataType* __restrict__ p_wei_grad,
                                       const OutDataType* const* __restrict__ p_out_grads,
                                       const DDataType* const* __restrict__ p_ds,
                                       const long_index_t* const* __restrict__ p_d_strides,
                                       long_index_t G,
                                       long_index_t N,
                                       long_index_t K,
                                       long_index_t C,
                                       long_index_t Di,
                                       long_index_t Hi,
                                       long_index_t Wi,
                                       long_index_t Z,
                                       long_index_t Y,
                                       long_index_t X,
                                       long_index_t Do,
                                       long_index_t Ho,
                                       long_index_t Wo,
                                       long_index_t stride_z,
                                       long_index_t stride_y,
                                       long_index_t stride_x,
                                       long_index_t dilation_z,
                                       long_index_t dilation_y,
                                       long_index_t dilation_x,
                                       long_index_t pad_z,
                                       long_index_t pad_y,
                                       long_index_t pad_x,
                                       long_index_t in_sg,
                                       long_index_t in_sn,
                                       long_index_t in_sc,
                                       long_index_t in_sd,
                                       long_index_t in_sh,
                                       long_index_t in_sw,
                                       long_index_t out_sg,
                                       long_index_t out_sn,
                                       long_index_t out_sk,
                                       long_index_t out_sd,
                                       long_index_t out_sh,
                                       long_index_t out_sw,
                                       long_index_t wei_sg,
                                       long_index_t wei_sk,
                                       long_index_t wei_sc,
                                       long_index_t wei_sz,
                                       long_index_t wei_sy,
                                       long_index_t wei_sx,
                                       InElementOp in_op,
                                       WeiElementOp wei_op,
                                       OutElementOp out_op)
{
    const long_index_t tid         = blockIdx.x * blockDim.x + threadIdx.x;
    const long_index_t num_threads = blockDim.x * gridDim.x;

    InDataType in_val   = InDataType{0};
    WeiDataType wei_val = WeiDataType{0};
    OutDataType out_val = OutDataType{0};

    if constexpr(NDimSpatial == 1)
    {
        const long_index_t num_wei      = G * K * C * X;
        const long_index_t in_stride_g  = in_sg;
        const long_index_t in_stride_n  = in_sn;
        const long_index_t in_stride_c  = in_sc;
        const long_index_t out_stride_g = out_sg;
        const long_index_t out_stride_n = out_sn;
        const long_index_t out_stride_k = out_sk;

        for(long_index_t idx = tid; idx < num_wei; idx += num_threads)
        {
            long_index_t remaining = idx;
            const long_index_t x   = remaining % X;
            remaining /= X;
            const long_index_t c = remaining % C;
            remaining /= C;
            const long_index_t k = remaining % K;
            const long_index_t g = remaining / K;

            float acc = 0.0f;
            // Base pointers for current group
            const InDataType* input_g        = p_ins[0] + g * in_stride_g;
            const OutDataType* output_grad_g = p_out_grads[0] + g * out_stride_g;

            // Loop over batch and output positions
            for(long_index_t n = 0; n < N; ++n)
            {
                // Pointers at current batch and input channel
                const InDataType* input_at_n_c = input_g + n * in_stride_n + c * in_stride_c;
                const OutDataType* output_grad_at_n_k =
                    output_grad_g + n * out_stride_n + k * out_stride_k;

                for(long_index_t wo = 0; wo < Wo; ++wo)
                {
                    long_index_t wi = wo * stride_x + x * dilation_x - pad_x;
                    if(wi >= 0 && wi < Wi)
                    {
                        // Handle input element-wise operation with extra A tensors
                        detail::apply_multi_tensor_elementwise_op<NumAExtra>(
                            in_val,
                            in_op,
                            input_at_n_c,
                            p_ins + 1,
                            g * in_stride_g + n * in_stride_n + c * in_stride_c,
                            wi * in_sw);

                        // Handle output gradient element-wise operation with extra B tensors
                        detail::apply_multi_tensor_elementwise_op<NumBExtra>(
                            out_val,
                            out_op,
                            output_grad_at_n_k,
                            p_out_grads + 1,
                            g * out_stride_g + n * out_stride_n + k * out_stride_k,
                            wo * out_sw);

                        acc += type_convert<float>(out_val) * type_convert<float>(in_val);
                    }
                }
            }

            detail::apply_d_tensor_elementwise_op<NumD>(
                wei_val, wei_op, acc, p_ds, p_d_strides, g, k, c, x);

            p_wei_grad[g * wei_sg + k * wei_sk + c * wei_sc + x * wei_sx] = wei_val;
        }
    }
    else if constexpr(NDimSpatial == 2)
    {
        const long_index_t num_wei      = G * K * C * Y * X;
        const long_index_t in_stride_g  = in_sg;
        const long_index_t in_stride_n  = in_sn;
        const long_index_t in_stride_c  = in_sc;
        const long_index_t in_stride_h  = in_sh;
        const long_index_t out_stride_g = out_sg;
        const long_index_t out_stride_n = out_sn;
        const long_index_t out_stride_k = out_sk;
        const long_index_t out_stride_h = out_sh;

        for(long_index_t idx = tid; idx < num_wei; idx += num_threads)
        {
            long_index_t remaining = idx;
            const long_index_t x   = remaining % X;
            remaining /= X;
            const long_index_t y = remaining % Y;
            remaining /= Y;
            const long_index_t c = remaining % C;
            remaining /= C;
            const long_index_t k = remaining % K;
            const long_index_t g = remaining / K;

            float acc = 0.0f;
            // Base pointers for current group
            const InDataType* input_g        = p_ins[0] + g * in_stride_g;
            const OutDataType* output_grad_g = p_out_grads[0] + g * out_stride_g;

            // Loop over batch and output positions
            for(index_t n = 0; n < N; ++n)
            {
                // Pointers at current batch and input channel
                const InDataType* input_at_n_c = input_g + n * in_stride_n + c * in_stride_c;
                const OutDataType* output_grad_at_n_k =
                    output_grad_g + n * out_stride_n + k * out_stride_k;

                for(index_t ho = 0; ho < Ho; ++ho)
                {
                    long_index_t hi = ho * stride_y + y * dilation_y - pad_y;
                    if(hi >= 0 && hi < Hi)
                    {
                        // Pointers at current spatial height
                        const InDataType* input_at_h = input_at_n_c + hi * in_stride_h;
                        const OutDataType* output_grad_at_h =
                            output_grad_at_n_k + ho * out_stride_h;

                        for(index_t wo = 0; wo < Wo; ++wo)
                        {
                            long_index_t wi = wo * stride_x + x * dilation_x - pad_x;
                            if(wi >= 0 && wi < Wi)
                            {
                                // Handle input element-wise operation with extra A tensors
                                detail::apply_multi_tensor_elementwise_op<NumAExtra>(
                                    in_val,
                                    in_op,
                                    input_at_h,
                                    p_ins + 1,
                                    g * in_stride_g + n * in_stride_n + c * in_stride_c +
                                        hi * in_stride_h,
                                    wi * in_sw);

                                // Handle output gradient element-wise operation with extra B
                                // tensors
                                detail::apply_multi_tensor_elementwise_op<NumBExtra>(
                                    out_val,
                                    out_op,
                                    output_grad_at_h,
                                    p_out_grads + 1,
                                    g * out_stride_g + n * out_stride_n + k * out_stride_k +
                                        ho * out_stride_h,
                                    wo * out_sw);

                                acc += type_convert<float>(out_val) * type_convert<float>(in_val);
                            }
                        }
                    }
                }
            }

            detail::apply_d_tensor_elementwise_op<NumD>(wei_val,
                                                        wei_op,
                                                        acc,
                                                        p_ds,
                                                        p_d_strides,
                                                        g,
                                                        k,
                                                        c,
                                                        y * p_d_strides[0][3] +
                                                            x * p_d_strides[0][4]);

            p_wei_grad[g * wei_sg + k * wei_sk + c * wei_sc + y * wei_sy + x * wei_sx] = wei_val;
        }
    }
    else if constexpr(NDimSpatial == 3)
    {
        const long_index_t num_wei      = G * K * C * Z * Y * X;
        const long_index_t in_stride_g  = in_sg;
        const long_index_t in_stride_n  = in_sn;
        const long_index_t in_stride_c  = in_sc;
        const long_index_t in_stride_d  = in_sd;
        const long_index_t in_stride_h  = in_sh;
        const long_index_t out_stride_g = out_sg;
        const long_index_t out_stride_n = out_sn;
        const long_index_t out_stride_k = out_sk;
        const long_index_t out_stride_d = out_sd;
        const long_index_t out_stride_h = out_sh;

        for(long_index_t idx = tid; idx < num_wei; idx += num_threads)
        {
            long_index_t remaining = idx;
            const long_index_t x   = remaining % X;
            remaining /= X;
            const long_index_t y = remaining % Y;
            remaining /= Y;
            const long_index_t z = remaining % Z;
            remaining /= Z;
            const long_index_t c = remaining % C;
            remaining /= C;
            const long_index_t k = remaining % K;
            const long_index_t g = remaining / K;

            float acc = 0.0f;
            // Base pointers for current group
            const InDataType* input_g        = p_ins[0] + g * in_stride_g;
            const OutDataType* output_grad_g = p_out_grads[0] + g * out_stride_g;

            // Loop over batch and output positions
            for(index_t n = 0; n < N; ++n)
            {
                // Pointers at current batch and input channel
                const InDataType* input_at_n_c = input_g + n * in_stride_n + c * in_stride_c;
                const OutDataType* output_grad_at_n_k =
                    output_grad_g + n * out_stride_n + k * out_stride_k;

                for(index_t do_idx = 0; do_idx < Do; ++do_idx)
                {
                    long_index_t di = do_idx * stride_z + z * dilation_z - pad_z;
                    if(di >= 0 && di < Di)
                    {
                        // Pointers at current spatial depth
                        const InDataType* input_at_d = input_at_n_c + di * in_stride_d;
                        const OutDataType* output_grad_at_d =
                            output_grad_at_n_k + do_idx * out_stride_d;

                        for(index_t ho = 0; ho < Ho; ++ho)
                        {
                            long_index_t hi = ho * stride_y + y * dilation_y - pad_y;
                            if(hi >= 0 && hi < Hi)
                            {
                                // Pointers at current spatial depth and height
                                const InDataType* input_at_d_h = input_at_d + hi * in_stride_h;
                                const OutDataType* output_grad_at_d_h =
                                    output_grad_at_d + ho * out_stride_h;

                                for(index_t wo = 0; wo < Wo; ++wo)
                                {
                                    long_index_t wi = wo * stride_x + x * dilation_x - pad_x;
                                    if(wi >= 0 && wi < Wi)
                                    {
                                        // Handle input element-wise operation with extra A tensors
                                        detail::apply_multi_tensor_elementwise_op<NumAExtra>(
                                            in_val,
                                            in_op,
                                            input_at_d_h,
                                            p_ins + 1,
                                            g * in_stride_g + n * in_stride_n + c * in_stride_c +
                                                di * in_stride_d + hi * in_stride_h,
                                            wi * in_sw);

                                        // Handle output gradient element-wise operation with extra
                                        // B tensors
                                        detail::apply_multi_tensor_elementwise_op<NumBExtra>(
                                            out_val,
                                            out_op,
                                            output_grad_at_d_h,
                                            p_out_grads + 1,
                                            g * out_stride_g + n * out_stride_n + k * out_stride_k +
                                                do_idx * out_stride_d + ho * out_stride_h,
                                            wo * out_sw);

                                        acc += type_convert<float>(out_val) *
                                               type_convert<float>(in_val);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            detail::apply_d_tensor_elementwise_op<NumD>(
                wei_val,
                wei_op,
                acc,
                p_ds,
                p_d_strides,
                g,
                k,
                c,
                z * p_d_strides[0][3] + y * p_d_strides[0][4] + x * p_d_strides[0][5]);

            p_wei_grad[g * wei_sg + k * wei_sk + c * wei_sc + z * wei_sz + y * wei_sy +
                       x * wei_sx] = wei_val;
        }
    }
}

// GPU reference backward weight convolution with multi-ABD support - takes ConvParam directly
template <ck::index_t NumAElementwise = 0,
          ck::index_t NumBElementwise = 0,
          ck::index_t NumDElementwise = 0,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename TIn,
          typename TWei,
          typename TOut,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          typename TD = TWei> // D tensor type, defaults to TWei for backward compatibility
void naive_conv_bwd_weight_multi_abd(
    const std::array<const TIn*, NumAElementwise + 1>& p_ins,
    TWei* p_wei_grad,
    const std::array<const TOut*, NumBElementwise + 1>& p_outs,
    const std::array<const TD*, NumDElementwise>& p_ds,
    const ck::utils::conv::ConvParam& conv_param,
    [[maybe_unused]] const std::array<std::vector<long_index_t>, NumDElementwise>& d_lengths,
    const std::array<std::vector<long_index_t>, NumDElementwise>& d_strides,
    InElementwiseOperation in_element_op   = InElementwiseOperation{},
    WeiElementwiseOperation wei_element_op = WeiElementwiseOperation{},
    OutElementwiseOperation out_element_op = OutElementwiseOperation{},
    hipStream_t stream                     = nullptr)
{
    const auto ndim = conv_param.num_dim_spatial_;

    const long_index_t G = conv_param.G_;
    const long_index_t N = conv_param.N_;
    const long_index_t C = conv_param.C_;
    const long_index_t K = conv_param.K_;

    std::vector<long_index_t> in_lengths  = {G, N, C};
    std::vector<long_index_t> wei_lengths = {G, K, C};
    std::vector<long_index_t> out_lengths = {G, N, K};

    for(index_t i = 0; i < ndim; ++i)
    {
        in_lengths.push_back(static_cast<long_index_t>(conv_param.input_spatial_lengths_[i]));
        wei_lengths.push_back(static_cast<long_index_t>(conv_param.filter_spatial_lengths_[i]));
        out_lengths.push_back(static_cast<long_index_t>(conv_param.output_spatial_lengths_[i]));
    }

    // Calculate total elements for grid size
    long_index_t wei_total = 1;
    for(auto l : wei_lengths)
        wei_total *= l;

    // Compute strides from layout
    std::vector<long_index_t> in_strides = compute_conv_tensor_strides<InLayout>(in_lengths, ndim);
    std::vector<long_index_t> wei_strides =
        compute_conv_tensor_strides<WeiLayout>(wei_lengths, ndim);
    std::vector<long_index_t> out_strides =
        compute_conv_tensor_strides<OutLayout>(out_lengths, ndim);

    // Prepare D tensor stride arrays on device
    std::vector<SimpleDeviceMem> d_stride_bufs;
    std::array<long_index_t*, NumDElementwise> p_d_strides_dev = {};

    if constexpr(NumDElementwise > 0)
    {
        d_stride_bufs.reserve(NumDElementwise);

        for(index_t i = 0; i < NumDElementwise; ++i)
        {
            d_stride_bufs.emplace_back(d_strides[i].size() * sizeof(long_index_t));
            p_d_strides_dev[i] = static_cast<long_index_t*>(d_stride_bufs[i].GetDeviceBuffer());

            HIP_CHECK_ERROR(hipMemcpy(p_d_strides_dev[i],
                                      d_strides[i].data(),
                                      d_strides[i].size() * sizeof(long_index_t),
                                      hipMemcpyHostToDevice));
        }
    }

    // Create device arrays of pointers
    SimpleDeviceMem ins_ptrs_buf((NumAElementwise + 1) * sizeof(TIn*));
    SimpleDeviceMem out_grads_ptrs_buf((NumBElementwise + 1) * sizeof(TOut*));
    SimpleDeviceMem ds_ptrs_buf(NumDElementwise * sizeof(TD*));
    SimpleDeviceMem d_strides_ptrs_buf(NumDElementwise * sizeof(index_t*));

    TIn** d_ins_ptrs        = static_cast<TIn**>(ins_ptrs_buf.GetDeviceBuffer());
    TOut** d_out_grads_ptrs = static_cast<TOut**>(out_grads_ptrs_buf.GetDeviceBuffer());
    TD** d_ds_ptrs          = static_cast<TD**>(ds_ptrs_buf.GetDeviceBuffer());
    long_index_t** d_d_strides_ptrs =
        static_cast<long_index_t**>(d_strides_ptrs_buf.GetDeviceBuffer());

    HIP_CHECK_ERROR(hipMemcpy(
        d_ins_ptrs, p_ins.data(), (NumAElementwise + 1) * sizeof(TIn*), hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(d_out_grads_ptrs,
                              p_outs.data(),
                              (NumBElementwise + 1) * sizeof(TOut*),
                              hipMemcpyHostToDevice));

    if constexpr(NumDElementwise > 0)
    {
        std::array<const TD*, NumDElementwise> p_ds_dev;
        for(index_t i = 0; i < NumDElementwise; ++i)
        {
            p_ds_dev[i] = p_ds[i];
        }

        HIP_CHECK_ERROR(hipMemcpy(
            d_ds_ptrs, p_ds_dev.data(), NumDElementwise * sizeof(TD*), hipMemcpyHostToDevice));
        HIP_CHECK_ERROR(hipMemcpy(d_d_strides_ptrs,
                                  p_d_strides_dev.data(),
                                  NumDElementwise * sizeof(index_t*),
                                  hipMemcpyHostToDevice));
    }

    // Build conv parameter vectors for kernel invocation
    std::vector<long_index_t> conv_strides(ndim);
    std::vector<long_index_t> conv_dilations(ndim);
    std::vector<long_index_t> input_pads(ndim);
    for(index_t i = 0; i < ndim; ++i)
    {
        conv_strides[i]   = static_cast<long_index_t>(conv_param.conv_filter_strides_[i]);
        conv_dilations[i] = static_cast<long_index_t>(conv_param.conv_filter_dilations_[i]);
        input_pads[i]     = static_cast<long_index_t>(conv_param.input_left_pads_[i]);
    }

    // Run backward weight convolution kernel directly on original tensors using layout strides
    constexpr int block_size              = 256;
    const long_index_t wei_grid_unclamped = (wei_total + block_size - 1) / block_size;
    // gridDim.x * blockDim.x must not overflow uint32_t; the kernel uses a grid-stride loop.
    constexpr long_index_t max_grid =
        static_cast<long_index_t>(std::numeric_limits<uint32_t>::max()) / block_size;
    const int wei_grid = static_cast<int>(std::min(wei_grid_unclamped, max_grid));

    if(ndim == 1)
    {
        // in_strides:  [sg, sn, sc, sw]  (indices 0..3)
        // out_strides: [sg, sn, sk, sw]  (indices 0..3)
        // wei_strides: [sg, sk, sc, sx]  (indices 0..3)
        naive_conv_bwd_weight_packed_multi_abd<1,
                                               NumAElementwise,
                                               NumBElementwise,
                                               NumDElementwise,
                                               TIn,
                                               TWei,
                                               TOut,
                                               TD,
                                               InElementwiseOperation,
                                               WeiElementwiseOperation,
                                               OutElementwiseOperation>
            <<<wei_grid, block_size, 0, stream>>>(d_ins_ptrs,
                                                  p_wei_grad,
                                                  d_out_grads_ptrs,
                                                  d_ds_ptrs,
                                                  d_d_strides_ptrs,
                                                  G,
                                                  N,
                                                  K,
                                                  C,
                                                  1,
                                                  1,
                                                  in_lengths[3],
                                                  1,
                                                  1,
                                                  wei_lengths[3],
                                                  1,
                                                  1,
                                                  out_lengths[3],
                                                  1,
                                                  1,
                                                  conv_strides[0],
                                                  1,
                                                  1,
                                                  conv_dilations[0],
                                                  0,
                                                  0,
                                                  input_pads[0],
                                                  in_strides[0],
                                                  in_strides[1],
                                                  in_strides[2],
                                                  0,
                                                  0,
                                                  in_strides[3],
                                                  out_strides[0],
                                                  out_strides[1],
                                                  out_strides[2],
                                                  0,
                                                  0,
                                                  out_strides[3],
                                                  wei_strides[0],
                                                  wei_strides[1],
                                                  wei_strides[2],
                                                  0,
                                                  0,
                                                  wei_strides[3],
                                                  in_element_op,
                                                  wei_element_op,
                                                  out_element_op);
    }
    else if(ndim == 2)
    {
        // in_strides:  [sg, sn, sc, sh, sw]  (indices 0..4)
        // out_strides: [sg, sn, sk, sh, sw]  (indices 0..4)
        // wei_strides: [sg, sk, sc, sy, sx]  (indices 0..4)
        naive_conv_bwd_weight_packed_multi_abd<2,
                                               NumAElementwise,
                                               NumBElementwise,
                                               NumDElementwise,
                                               TIn,
                                               TWei,
                                               TOut,
                                               TD,
                                               InElementwiseOperation,
                                               WeiElementwiseOperation,
                                               OutElementwiseOperation>
            <<<wei_grid, block_size, 0, stream>>>(d_ins_ptrs,
                                                  p_wei_grad,
                                                  d_out_grads_ptrs,
                                                  d_ds_ptrs,
                                                  d_d_strides_ptrs,
                                                  G,
                                                  N,
                                                  K,
                                                  C,
                                                  1,
                                                  in_lengths[3],
                                                  in_lengths[4],
                                                  1,
                                                  wei_lengths[3],
                                                  wei_lengths[4],
                                                  1,
                                                  out_lengths[3],
                                                  out_lengths[4],
                                                  1,
                                                  conv_strides[0],
                                                  conv_strides[1],
                                                  1,
                                                  conv_dilations[0],
                                                  conv_dilations[1],
                                                  0,
                                                  input_pads[0],
                                                  input_pads[1],
                                                  in_strides[0],
                                                  in_strides[1],
                                                  in_strides[2],
                                                  0,
                                                  in_strides[3],
                                                  in_strides[4],
                                                  out_strides[0],
                                                  out_strides[1],
                                                  out_strides[2],
                                                  0,
                                                  out_strides[3],
                                                  out_strides[4],
                                                  wei_strides[0],
                                                  wei_strides[1],
                                                  wei_strides[2],
                                                  0,
                                                  wei_strides[3],
                                                  wei_strides[4],
                                                  in_element_op,
                                                  wei_element_op,
                                                  out_element_op);
    }
    else // 3D
    {
        // in_strides:  [sg, sn, sc, sd, sh, sw]  (indices 0..5)
        // out_strides: [sg, sn, sk, sd, sh, sw]  (indices 0..5)
        // wei_strides: [sg, sk, sc, sz, sy, sx]  (indices 0..5)
        naive_conv_bwd_weight_packed_multi_abd<3,
                                               NumAElementwise,
                                               NumBElementwise,
                                               NumDElementwise,
                                               TIn,
                                               TWei,
                                               TOut,
                                               TD,
                                               InElementwiseOperation,
                                               WeiElementwiseOperation,
                                               OutElementwiseOperation>
            <<<wei_grid, block_size, 0, stream>>>(d_ins_ptrs,
                                                  p_wei_grad,
                                                  d_out_grads_ptrs,
                                                  d_ds_ptrs,
                                                  d_d_strides_ptrs,
                                                  G,
                                                  N,
                                                  K,
                                                  C,
                                                  in_lengths[3],
                                                  in_lengths[4],
                                                  in_lengths[5],
                                                  wei_lengths[3],
                                                  wei_lengths[4],
                                                  wei_lengths[5],
                                                  out_lengths[3],
                                                  out_lengths[4],
                                                  out_lengths[5],
                                                  conv_strides[0],
                                                  conv_strides[1],
                                                  conv_strides[2],
                                                  conv_dilations[0],
                                                  conv_dilations[1],
                                                  conv_dilations[2],
                                                  input_pads[0],
                                                  input_pads[1],
                                                  input_pads[2],
                                                  in_strides[0],
                                                  in_strides[1],
                                                  in_strides[2],
                                                  in_strides[3],
                                                  in_strides[4],
                                                  in_strides[5],
                                                  out_strides[0],
                                                  out_strides[1],
                                                  out_strides[2],
                                                  out_strides[3],
                                                  out_strides[4],
                                                  out_strides[5],
                                                  wei_strides[0],
                                                  wei_strides[1],
                                                  wei_strides[2],
                                                  wei_strides[3],
                                                  wei_strides[4],
                                                  wei_strides[5],
                                                  in_element_op,
                                                  wei_element_op,
                                                  out_element_op);
    }

    HIP_CHECK_ERROR(hipGetLastError());

    // Memory automatically freed by SimpleDeviceMem destructors
}

// Original naive_conv_bwd_weight - now a zero-overhead wrapper
template <typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename TIn,
          typename TWei,
          typename TOut,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation>
inline void
naive_conv_bwd_weight(const TIn* p_in,
                      TWei* p_wei_grad,
                      const TOut* p_out,
                      const ck::utils::conv::ConvParam& conv_param,
                      InElementwiseOperation in_element_op   = InElementwiseOperation{},
                      WeiElementwiseOperation wei_element_op = WeiElementwiseOperation{},
                      OutElementwiseOperation out_element_op = OutElementwiseOperation{},
                      hipStream_t stream                     = nullptr)
{
    std::array<const TIn*, 1> p_ins                    = {p_in};
    std::array<const TOut*, 1> p_outs                  = {p_out};
    std::array<const TWei*, 0> p_ds                    = {};
    std::array<std::vector<long_index_t>, 0> d_lengths = {};
    std::array<std::vector<long_index_t>, 0> d_strides = {};

    naive_conv_bwd_weight_multi_abd<0, 0, 0, InLayout, WeiLayout, OutLayout>(p_ins,
                                                                             p_wei_grad,
                                                                             p_outs,
                                                                             p_ds,
                                                                             conv_param,
                                                                             d_lengths,
                                                                             d_strides,
                                                                             in_element_op,
                                                                             wei_element_op,
                                                                             out_element_op,
                                                                             stream);
}

} // namespace ref
} // namespace ck
