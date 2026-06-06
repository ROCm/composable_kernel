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

template <index_t NDimSpatial,
          index_t NumAExtra,
          index_t NumBExtra,
          index_t NumD,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename DDataType,
          typename InElementOp,
          typename WeiElementOp,
          typename OutElementOp>
__global__ void
naive_conv_bwd_data_packed_multi_abd(InDataType* __restrict__ p_in,
                                     const WeiDataType* const* __restrict__ p_weis,
                                     const OutDataType* const* __restrict__ p_outs,
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
                                     long_index_t wei_sg,
                                     long_index_t wei_sk,
                                     long_index_t wei_sc,
                                     long_index_t wei_sz,
                                     long_index_t wei_sy,
                                     long_index_t wei_sx,
                                     long_index_t out_sg,
                                     long_index_t out_sn,
                                     long_index_t out_sk,
                                     long_index_t out_sd,
                                     long_index_t out_sh,
                                     long_index_t out_sw,
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
        const long_index_t num_in = G * N * C * Wi;

        for(long_index_t idx = tid; idx < num_in; idx += num_threads)
        {
            long_index_t remaining = idx;
            const long_index_t wi  = remaining % Wi;
            remaining /= Wi;
            const long_index_t c = remaining % C;
            remaining /= C;
            const long_index_t n = remaining % N;
            const long_index_t g = remaining / N;

            float acc                          = 0.0f;
            const OutDataType* output_grad_g_n = p_outs[0] + g * out_sg + n * out_sn;
            const WeiDataType* weight_g        = p_weis[0] + g * wei_sg;

            for(index_t x = 0; x < X; ++x)
            {
                long_index_t w_tmp = wi + pad_x - x * dilation_x;
                if(w_tmp % stride_x == 0)
                {
                    long_index_t wo = w_tmp / stride_x;
                    if(wo >= 0 && wo < Wo)
                    {
                        const OutDataType* output_grad_g_n_k = output_grad_g_n;
                        const WeiDataType* weight_g_k_c      = weight_g + c * wei_sc;

                        for(index_t k = 0; k < K; ++k)
                        {
                            detail::apply_multi_tensor_elementwise_op<NumAExtra>(
                                out_val,
                                out_op,
                                output_grad_g_n_k,
                                p_outs + 1,
                                g * out_sg + n * out_sn,
                                k * out_sk + wo * out_sw);

                            detail::apply_multi_tensor_elementwise_op<NumBExtra>(
                                wei_val,
                                wei_op,
                                weight_g_k_c,
                                p_weis + 1,
                                g * wei_sg + c * wei_sc,
                                k * wei_sk + x * wei_sx);

                            acc += type_convert<float>(out_val) * type_convert<float>(wei_val);
                        }
                    }
                }
            }

            detail::apply_d_tensor_elementwise_op<NumD>(
                in_val, in_op, acc, p_ds, p_d_strides, g, n, c, wi);

            p_in[g * in_sg + n * in_sn + c * in_sc + wi * in_sw] = in_val;
        }
    }
    else if constexpr(NDimSpatial == 2)
    {
        const long_index_t num_in = G * N * C * Hi * Wi;

        for(long_index_t idx = tid; idx < num_in; idx += num_threads)
        {
            long_index_t remaining = idx;
            const long_index_t wi  = remaining % Wi;
            remaining /= Wi;
            const long_index_t hi = remaining % Hi;
            remaining /= Hi;
            const long_index_t c = remaining % C;
            remaining /= C;
            const long_index_t n = remaining % N;
            const long_index_t g = remaining / N;

            float acc                          = 0.0f;
            const OutDataType* output_grad_g_n = p_outs[0] + g * out_sg + n * out_sn;
            const WeiDataType* weight_g        = p_weis[0] + g * wei_sg;

            for(index_t y = 0; y < Y; ++y)
            {
                long_index_t h_tmp = hi + pad_y - y * dilation_y;
                if(h_tmp % stride_y == 0)
                {
                    long_index_t ho = h_tmp / stride_y;
                    if(ho >= 0 && ho < Ho)
                    {
                        const OutDataType* output_grad_at_h = output_grad_g_n + ho * out_sh;
                        const WeiDataType* weight_at_c_y    = weight_g + c * wei_sc + y * wei_sy;

                        for(index_t x = 0; x < X; ++x)
                        {
                            long_index_t w_tmp = wi + pad_x - x * dilation_x;
                            if(w_tmp % stride_x == 0)
                            {
                                long_index_t wo = w_tmp / stride_x;
                                if(wo >= 0 && wo < Wo)
                                {
                                    for(index_t k = 0; k < K; ++k)
                                    {
                                        detail::apply_multi_tensor_elementwise_op<NumAExtra>(
                                            out_val,
                                            out_op,
                                            output_grad_at_h,
                                            p_outs + 1,
                                            g * out_sg + n * out_sn + ho * out_sh,
                                            k * out_sk + wo * out_sw);

                                        detail::apply_multi_tensor_elementwise_op<NumBExtra>(
                                            wei_val,
                                            wei_op,
                                            weight_at_c_y,
                                            p_weis + 1,
                                            g * wei_sg + c * wei_sc + y * wei_sy,
                                            k * wei_sk + x * wei_sx);

                                        acc += type_convert<float>(out_val) *
                                               type_convert<float>(wei_val);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            detail::apply_d_tensor_elementwise_op<NumD>(in_val,
                                                        in_op,
                                                        acc,
                                                        p_ds,
                                                        p_d_strides,
                                                        g,
                                                        n,
                                                        c,
                                                        hi * p_d_strides[0][3] +
                                                            wi * p_d_strides[0][4]);

            p_in[g * in_sg + n * in_sn + c * in_sc + hi * in_sh + wi * in_sw] = in_val;
        }
    }
    else if constexpr(NDimSpatial == 3)
    {
        const long_index_t num_in = G * N * C * Di * Hi * Wi;

        for(long_index_t idx = tid; idx < num_in; idx += num_threads)
        {
            long_index_t remaining = idx;
            const long_index_t wi  = remaining % Wi;
            remaining /= Wi;
            const long_index_t hi = remaining % Hi;
            remaining /= Hi;
            const long_index_t di = remaining % Di;
            remaining /= Di;
            const long_index_t c = remaining % C;
            remaining /= C;
            const long_index_t n = remaining % N;
            const long_index_t g = remaining / N;

            float acc                          = 0.0f;
            const OutDataType* output_grad_g_n = p_outs[0] + g * out_sg + n * out_sn;
            const WeiDataType* weight_g        = p_weis[0] + g * wei_sg;

            for(index_t z = 0; z < Z; ++z)
            {
                long_index_t d_tmp = di + pad_z - z * dilation_z;
                if(d_tmp % stride_z == 0)
                {
                    long_index_t do_idx = d_tmp / stride_z;
                    if(do_idx >= 0 && do_idx < Do)
                    {
                        const OutDataType* output_grad_at_d = output_grad_g_n + do_idx * out_sd;
                        const WeiDataType* weight_at_c_z    = weight_g + c * wei_sc + z * wei_sz;

                        for(index_t y = 0; y < Y; ++y)
                        {
                            long_index_t h_tmp = hi + pad_y - y * dilation_y;
                            if(h_tmp % stride_y == 0)
                            {
                                long_index_t ho = h_tmp / stride_y;
                                if(ho >= 0 && ho < Ho)
                                {
                                    const OutDataType* output_grad_at_d_h =
                                        output_grad_at_d + ho * out_sh;
                                    const WeiDataType* weight_at_c_z_y = weight_at_c_z + y * wei_sy;

                                    for(index_t x = 0; x < X; ++x)
                                    {
                                        long_index_t w_tmp = wi + pad_x - x * dilation_x;
                                        if(w_tmp % stride_x == 0)
                                        {
                                            long_index_t wo = w_tmp / stride_x;
                                            if(wo >= 0 && wo < Wo)
                                            {
                                                for(index_t k = 0; k < K; ++k)
                                                {
                                                    detail::apply_multi_tensor_elementwise_op<
                                                        NumAExtra>(out_val,
                                                                   out_op,
                                                                   output_grad_at_d_h,
                                                                   p_outs + 1,
                                                                   g * out_sg + n * out_sn +
                                                                       do_idx * out_sd +
                                                                       ho * out_sh,
                                                                   k * out_sk + wo * out_sw);

                                                    detail::apply_multi_tensor_elementwise_op<
                                                        NumBExtra>(wei_val,
                                                                   wei_op,
                                                                   weight_at_c_z_y,
                                                                   p_weis + 1,
                                                                   g * wei_sg + c * wei_sc +
                                                                       z * wei_sz + y * wei_sy,
                                                                   k * wei_sk + x * wei_sx);

                                                    acc += type_convert<float>(out_val) *
                                                           type_convert<float>(wei_val);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            detail::apply_d_tensor_elementwise_op<NumD>(
                in_val,
                in_op,
                acc,
                p_ds,
                p_d_strides,
                g,
                n,
                c,
                di * p_d_strides[0][3] + hi * p_d_strides[0][4] + wi * p_d_strides[0][5]);

            p_in[g * in_sg + n * in_sn + c * in_sc + di * in_sd + hi * in_sh + wi * in_sw] = in_val;
        }
    }
}

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
          typename TD = TIn>
void naive_conv_bwd_data_multi_abd(
    TIn* p_in,
    const std::array<const TWei*, NumBElementwise + 1>& p_weis,
    const std::array<const TOut*, NumAElementwise + 1>& p_outs,
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

    long_index_t in_total = 1;
    for(auto l : in_lengths)
        in_total *= l;

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

    // Create device pointer arrays (use original pointers directly, no packing)
    SimpleDeviceMem weis_ptrs_buf((NumBElementwise + 1) * sizeof(TWei*));
    SimpleDeviceMem outs_ptrs_buf((NumAElementwise + 1) * sizeof(TOut*));
    SimpleDeviceMem ds_ptrs_buf(NumDElementwise * sizeof(TD*));
    SimpleDeviceMem d_strides_ptrs_buf(NumDElementwise * sizeof(long_index_t*));

    TWei** d_weis_ptrs = static_cast<TWei**>(weis_ptrs_buf.GetDeviceBuffer());
    TOut** d_outs_ptrs = static_cast<TOut**>(outs_ptrs_buf.GetDeviceBuffer());
    TD** d_ds_ptrs     = static_cast<TD**>(ds_ptrs_buf.GetDeviceBuffer());
    long_index_t** d_d_strides_ptrs =
        static_cast<long_index_t**>(d_strides_ptrs_buf.GetDeviceBuffer());

    HIP_CHECK_ERROR(hipMemcpy(
        d_weis_ptrs, p_weis.data(), (NumBElementwise + 1) * sizeof(TWei*), hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(
        d_outs_ptrs, p_outs.data(), (NumAElementwise + 1) * sizeof(TOut*), hipMemcpyHostToDevice));

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
                                  NumDElementwise * sizeof(long_index_t*),
                                  hipMemcpyHostToDevice));
    }

    std::vector<long_index_t> conv_strides(ndim);
    std::vector<long_index_t> conv_dilations(ndim);
    std::vector<long_index_t> input_pads(ndim);
    for(index_t i = 0; i < ndim; ++i)
    {
        conv_strides[i]   = static_cast<long_index_t>(conv_param.conv_filter_strides_[i]);
        conv_dilations[i] = static_cast<long_index_t>(conv_param.conv_filter_dilations_[i]);
        input_pads[i]     = static_cast<long_index_t>(conv_param.input_left_pads_[i]);
    }

    // Extract strides indexed as [G,N,C,spatial...] and [G,K,C,spatial...] / [G,N,K,spatial...]
    // in_strides:  [0]=sg [1]=sn [2]=sc [3]=sd [4]=sh [5]=sw
    // wei_strides: [0]=sg [1]=sk [2]=sc [3]=sz [4]=sy [5]=sx
    // out_strides: [0]=sg [1]=sn [2]=sk [3]=sd [4]=sh [5]=sw
    const long_index_t in_sg = in_strides[0];
    const long_index_t in_sn = in_strides[1];
    const long_index_t in_sc = in_strides[2];
    const long_index_t in_sd = (ndim >= 3) ? in_strides[3] : 0;
    const long_index_t in_sh = (ndim >= 2) ? in_strides[ndim == 3 ? 4 : 3] : 0;
    const long_index_t in_sw = in_strides[ndim == 3 ? 5 : (ndim == 2 ? 4 : 3)];

    const long_index_t wei_sg = wei_strides[0];
    const long_index_t wei_sk = wei_strides[1];
    const long_index_t wei_sc = wei_strides[2];
    const long_index_t wei_sz = (ndim >= 3) ? wei_strides[3] : 0;
    const long_index_t wei_sy = (ndim >= 2) ? wei_strides[ndim == 3 ? 4 : 3] : 0;
    const long_index_t wei_sx = wei_strides[ndim == 3 ? 5 : (ndim == 2 ? 4 : 3)];

    const long_index_t out_sg = out_strides[0];
    const long_index_t out_sn = out_strides[1];
    const long_index_t out_sk = out_strides[2];
    const long_index_t out_sd = (ndim >= 3) ? out_strides[3] : 0;
    const long_index_t out_sh = (ndim >= 2) ? out_strides[ndim == 3 ? 4 : 3] : 0;
    const long_index_t out_sw = out_strides[ndim == 3 ? 5 : (ndim == 2 ? 4 : 3)];

    constexpr int block_size             = 256;
    const long_index_t in_grid_unclamped = (in_total + block_size - 1) / block_size;
    // gridDim.x * blockDim.x must not overflow uint32_t; kernel uses a grid-stride loop.
    constexpr long_index_t max_grid =
        static_cast<long_index_t>(std::numeric_limits<uint32_t>::max()) / block_size;
    const int in_grid = static_cast<int>(std::min(in_grid_unclamped, max_grid));

    if(ndim == 1)
    {
        naive_conv_bwd_data_packed_multi_abd<1,
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
            <<<in_grid, block_size, 0, stream>>>(p_in,
                                                 d_weis_ptrs,
                                                 d_outs_ptrs,
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
                                                 in_sg,
                                                 in_sn,
                                                 in_sc,
                                                 in_sd,
                                                 in_sh,
                                                 in_sw,
                                                 wei_sg,
                                                 wei_sk,
                                                 wei_sc,
                                                 wei_sz,
                                                 wei_sy,
                                                 wei_sx,
                                                 out_sg,
                                                 out_sn,
                                                 out_sk,
                                                 out_sd,
                                                 out_sh,
                                                 out_sw,
                                                 in_element_op,
                                                 wei_element_op,
                                                 out_element_op);
    }
    else if(ndim == 2)
    {
        naive_conv_bwd_data_packed_multi_abd<2,
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
            <<<in_grid, block_size, 0, stream>>>(p_in,
                                                 d_weis_ptrs,
                                                 d_outs_ptrs,
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
                                                 in_sg,
                                                 in_sn,
                                                 in_sc,
                                                 in_sd,
                                                 in_sh,
                                                 in_sw,
                                                 wei_sg,
                                                 wei_sk,
                                                 wei_sc,
                                                 wei_sz,
                                                 wei_sy,
                                                 wei_sx,
                                                 out_sg,
                                                 out_sn,
                                                 out_sk,
                                                 out_sd,
                                                 out_sh,
                                                 out_sw,
                                                 in_element_op,
                                                 wei_element_op,
                                                 out_element_op);
    }
    else // 3D
    {
        naive_conv_bwd_data_packed_multi_abd<3,
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
            <<<in_grid, block_size, 0, stream>>>(p_in,
                                                 d_weis_ptrs,
                                                 d_outs_ptrs,
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
                                                 in_sg,
                                                 in_sn,
                                                 in_sc,
                                                 in_sd,
                                                 in_sh,
                                                 in_sw,
                                                 wei_sg,
                                                 wei_sk,
                                                 wei_sc,
                                                 wei_sz,
                                                 wei_sy,
                                                 wei_sx,
                                                 out_sg,
                                                 out_sn,
                                                 out_sk,
                                                 out_sd,
                                                 out_sh,
                                                 out_sw,
                                                 in_element_op,
                                                 wei_element_op,
                                                 out_element_op);
    }

    HIP_CHECK_ERROR(hipGetLastError());
}

template <typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename TIn,
          typename TWei,
          typename TOut,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation>
inline void naive_conv_bwd_data(TIn* p_in,
                                const TWei* p_wei,
                                const TOut* p_out,
                                const ck::utils::conv::ConvParam& conv_param,
                                InElementwiseOperation in_element_op   = InElementwiseOperation{},
                                WeiElementwiseOperation wei_element_op = WeiElementwiseOperation{},
                                OutElementwiseOperation out_element_op = OutElementwiseOperation{},
                                hipStream_t stream                     = nullptr)
{
    std::array<const TWei*, 1> p_weis                  = {p_wei};
    std::array<const TOut*, 1> p_outs                  = {p_out};
    std::array<const TIn*, 0> p_ds                     = {};
    std::array<std::vector<long_index_t>, 0> d_lengths = {};
    std::array<std::vector<long_index_t>, 0> d_strides = {};

    naive_conv_bwd_data_multi_abd<0, 0, 0, InLayout, WeiLayout, OutLayout>(p_in,
                                                                           p_weis,
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
