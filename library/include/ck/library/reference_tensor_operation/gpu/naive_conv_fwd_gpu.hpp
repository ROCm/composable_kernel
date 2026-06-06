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
__global__ void naive_conv_fwd_packed_multi_abd(const InDataType* const* __restrict__ p_ins,
                                                const WeiDataType* const* __restrict__ p_weis,
                                                const DDataType* const* __restrict__ p_ds,
                                                const long_index_t* const* __restrict__ p_d_strides,
                                                OutDataType* __restrict__ p_out,
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
        const long_index_t num_out = G * N * K * Wo;

        for(long_index_t idx = tid; idx < num_out; idx += num_threads)
        {
            long_index_t remaining = idx;
            const long_index_t wo  = remaining % Wo;
            remaining /= Wo;
            const long_index_t k = remaining % K;
            remaining /= K;
            const long_index_t n = remaining % N;
            const long_index_t g = remaining / N;

            float acc                     = 0.0f;
            const InDataType* input_g_n   = p_ins[0] + g * in_sg + n * in_sn;
            const WeiDataType* weight_g_k = p_weis[0] + g * wei_sg + k * wei_sk;

            for(index_t c = 0; c < C; ++c)
            {
                const InDataType* input_at_c   = input_g_n + c * in_sc;
                const WeiDataType* weight_at_c = weight_g_k + c * wei_sc;

                for(index_t x = 0; x < X; ++x)
                {
                    long_index_t wi = wo * stride_x + x * dilation_x - pad_x;
                    if(wi >= 0 && wi < Wi)
                    {
                        detail::apply_multi_tensor_elementwise_op<NumAExtra>(in_val,
                                                                             in_op,
                                                                             input_at_c,
                                                                             p_ins + 1,
                                                                             g * in_sg + n * in_sn +
                                                                                 c * in_sc,
                                                                             wi * in_sw);

                        detail::apply_multi_tensor_elementwise_op<NumBExtra>(
                            wei_val,
                            wei_op,
                            weight_at_c,
                            p_weis + 1,
                            g * wei_sg + k * wei_sk + c * wei_sc,
                            x * wei_sx);

                        acc += type_convert<float>(in_val) * type_convert<float>(wei_val);
                    }
                }
            }

            detail::apply_d_tensor_elementwise_op<NumD>(
                out_val, out_op, acc, p_ds, p_d_strides, g, n, k, wo);

            p_out[g * out_sg + n * out_sn + k * out_sk + wo * out_sw] = out_val;
        }
    }
    else if constexpr(NDimSpatial == 2)
    {
        const long_index_t num_out = G * N * K * Ho * Wo;

        for(long_index_t idx = tid; idx < num_out; idx += num_threads)
        {
            long_index_t remaining = idx;
            const long_index_t wo  = remaining % Wo;
            remaining /= Wo;
            const long_index_t ho = remaining % Ho;
            remaining /= Ho;
            const long_index_t k = remaining % K;
            remaining /= K;
            const long_index_t n = remaining % N;
            const long_index_t g = remaining / N;

            float acc                     = 0.0f;
            const InDataType* input_g_n   = p_ins[0] + g * in_sg + n * in_sn;
            const WeiDataType* weight_g_k = p_weis[0] + g * wei_sg + k * wei_sk;

            for(index_t c = 0; c < C; ++c)
            {
                const InDataType* input_at_c   = input_g_n + c * in_sc;
                const WeiDataType* weight_at_c = weight_g_k + c * wei_sc;

                for(index_t y = 0; y < Y; ++y)
                {
                    long_index_t hi = ho * stride_y + y * dilation_y - pad_y;
                    if(hi >= 0 && hi < Hi)
                    {
                        const InDataType* input_at_h   = input_at_c + hi * in_sh;
                        const WeiDataType* weight_at_y = weight_at_c + y * wei_sy;

                        for(index_t x = 0; x < X; ++x)
                        {
                            long_index_t wi = wo * stride_x + x * dilation_x - pad_x;
                            if(wi >= 0 && wi < Wi)
                            {
                                detail::apply_multi_tensor_elementwise_op<NumAExtra>(
                                    in_val,
                                    in_op,
                                    input_at_h,
                                    p_ins + 1,
                                    g * in_sg + n * in_sn + c * in_sc + hi * in_sh,
                                    wi * in_sw);

                                detail::apply_multi_tensor_elementwise_op<NumBExtra>(
                                    wei_val,
                                    wei_op,
                                    weight_at_y,
                                    p_weis + 1,
                                    g * wei_sg + k * wei_sk + c * wei_sc + y * wei_sy,
                                    x * wei_sx);

                                acc += type_convert<float>(in_val) * type_convert<float>(wei_val);
                            }
                        }
                    }
                }
            }

            detail::apply_d_tensor_elementwise_op<NumD>(out_val,
                                                        out_op,
                                                        acc,
                                                        p_ds,
                                                        p_d_strides,
                                                        g,
                                                        n,
                                                        k,
                                                        ho * p_d_strides[0][3] +
                                                            wo * p_d_strides[0][4]);

            p_out[g * out_sg + n * out_sn + k * out_sk + ho * out_sh + wo * out_sw] = out_val;
        }
    }
    else if constexpr(NDimSpatial == 3)
    {
        const long_index_t num_out = G * N * K * Do * Ho * Wo;

        for(long_index_t idx = tid; idx < num_out; idx += num_threads)
        {
            long_index_t remaining = idx;
            const long_index_t wo  = remaining % Wo;
            remaining /= Wo;
            const long_index_t ho = remaining % Ho;
            remaining /= Ho;
            const long_index_t do_idx = remaining % Do;
            remaining /= Do;
            const long_index_t k = remaining % K;
            remaining /= K;
            const long_index_t n = remaining % N;
            const long_index_t g = remaining / N;

            float acc                     = 0.0f;
            const InDataType* input_g_n   = p_ins[0] + g * in_sg + n * in_sn;
            const WeiDataType* weight_g_k = p_weis[0] + g * wei_sg + k * wei_sk;

            for(index_t c = 0; c < C; ++c)
            {
                const InDataType* input_at_c   = input_g_n + c * in_sc;
                const WeiDataType* weight_at_c = weight_g_k + c * wei_sc;

                for(index_t z = 0; z < Z; ++z)
                {
                    long_index_t di = do_idx * stride_z + z * dilation_z - pad_z;
                    if(di >= 0 && di < Di)
                    {
                        const InDataType* input_at_d   = input_at_c + di * in_sd;
                        const WeiDataType* weight_at_z = weight_at_c + z * wei_sz;

                        for(index_t y = 0; y < Y; ++y)
                        {
                            long_index_t hi = ho * stride_y + y * dilation_y - pad_y;
                            if(hi >= 0 && hi < Hi)
                            {
                                const InDataType* input_at_d_h   = input_at_d + hi * in_sh;
                                const WeiDataType* weight_at_z_y = weight_at_z + y * wei_sy;

                                for(index_t x = 0; x < X; ++x)
                                {
                                    long_index_t wi = wo * stride_x + x * dilation_x - pad_x;
                                    if(wi >= 0 && wi < Wi)
                                    {
                                        detail::apply_multi_tensor_elementwise_op<NumAExtra>(
                                            in_val,
                                            in_op,
                                            input_at_d_h,
                                            p_ins + 1,
                                            g * in_sg + n * in_sn + c * in_sc + di * in_sd +
                                                hi * in_sh,
                                            wi * in_sw);

                                        detail::apply_multi_tensor_elementwise_op<NumBExtra>(
                                            wei_val,
                                            wei_op,
                                            weight_at_z_y,
                                            p_weis + 1,
                                            g * wei_sg + k * wei_sk + c * wei_sc + z * wei_sz +
                                                y * wei_sy,
                                            x * wei_sx);

                                        acc += type_convert<float>(in_val) *
                                               type_convert<float>(wei_val);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            detail::apply_d_tensor_elementwise_op<NumD>(
                out_val,
                out_op,
                acc,
                p_ds,
                p_d_strides,
                g,
                n,
                k,
                do_idx * p_d_strides[0][3] + ho * p_d_strides[0][4] + wo * p_d_strides[0][5]);

            p_out[g * out_sg + n * out_sn + k * out_sk + do_idx * out_sd + ho * out_sh +
                  wo * out_sw] = out_val;
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
          typename TD = TOut>
void naive_conv_fwd_multi_abd(
    const std::array<const TIn*, NumAElementwise + 1>& p_ins,
    const std::array<const TWei*, NumBElementwise + 1>& p_weis,
    const std::array<const TD*, NumDElementwise>& p_ds,
    TOut* p_out,
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

    long_index_t out_total = 1;
    for(auto l : out_lengths)
        out_total *= l;

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
    SimpleDeviceMem ins_ptrs_buf((NumAElementwise + 1) * sizeof(TIn*));
    SimpleDeviceMem weis_ptrs_buf((NumBElementwise + 1) * sizeof(TWei*));
    SimpleDeviceMem ds_ptrs_buf(NumDElementwise * sizeof(TD*));
    SimpleDeviceMem d_strides_ptrs_buf(NumDElementwise * sizeof(long_index_t*));

    TIn** d_ins_ptrs   = static_cast<TIn**>(ins_ptrs_buf.GetDeviceBuffer());
    TWei** d_weis_ptrs = static_cast<TWei**>(weis_ptrs_buf.GetDeviceBuffer());
    TD** d_ds_ptrs     = static_cast<TD**>(ds_ptrs_buf.GetDeviceBuffer());
    long_index_t** d_d_strides_ptrs =
        static_cast<long_index_t**>(d_strides_ptrs_buf.GetDeviceBuffer());

    HIP_CHECK_ERROR(hipMemcpy(
        d_ins_ptrs, p_ins.data(), (NumAElementwise + 1) * sizeof(TIn*), hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(
        d_weis_ptrs, p_weis.data(), (NumBElementwise + 1) * sizeof(TWei*), hipMemcpyHostToDevice));

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

    constexpr int block_size              = 256;
    const long_index_t out_grid_unclamped = (out_total + block_size - 1) / block_size;
    // gridDim.x * blockDim.x must not overflow uint32_t; the kernel uses a grid-stride loop.
    constexpr long_index_t max_grid =
        static_cast<long_index_t>(std::numeric_limits<uint32_t>::max()) / block_size;
    const int out_grid = static_cast<int>(std::min(out_grid_unclamped, max_grid));

    if(ndim == 1)
    {
        naive_conv_fwd_packed_multi_abd<1,
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
            <<<out_grid, block_size, 0, stream>>>(d_ins_ptrs,
                                                  d_weis_ptrs,
                                                  d_ds_ptrs,
                                                  d_d_strides_ptrs,
                                                  p_out,
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
        naive_conv_fwd_packed_multi_abd<2,
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
            <<<out_grid, block_size, 0, stream>>>(d_ins_ptrs,
                                                  d_weis_ptrs,
                                                  d_ds_ptrs,
                                                  d_d_strides_ptrs,
                                                  p_out,
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
        naive_conv_fwd_packed_multi_abd<3,
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
            <<<out_grid, block_size, 0, stream>>>(d_ins_ptrs,
                                                  d_weis_ptrs,
                                                  d_ds_ptrs,
                                                  d_d_strides_ptrs,
                                                  p_out,
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
inline void naive_conv_fwd(const TIn* p_in,
                           const TWei* p_wei,
                           TOut* p_out,
                           const ck::utils::conv::ConvParam& conv_param,
                           InElementwiseOperation in_element_op   = InElementwiseOperation{},
                           WeiElementwiseOperation wei_element_op = WeiElementwiseOperation{},
                           OutElementwiseOperation out_element_op = OutElementwiseOperation{},
                           hipStream_t stream                     = nullptr)
{
    std::array<const TIn*, 1> p_ins                    = {p_in};
    std::array<const TWei*, 1> p_weis                  = {p_wei};
    std::array<const TOut*, 0> p_ds                    = {};
    std::array<std::vector<long_index_t>, 0> d_lengths = {};
    std::array<std::vector<long_index_t>, 0> d_strides = {};

    naive_conv_fwd_multi_abd<0, 0, 0, InLayout, WeiLayout, OutLayout>(p_ins,
                                                                      p_weis,
                                                                      p_ds,
                                                                      p_out,
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
