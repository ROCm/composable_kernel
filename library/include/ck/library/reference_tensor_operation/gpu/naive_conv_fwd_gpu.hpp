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

// Optimized convolution kernel working with packed (contiguous) tensors with multi-ABD support
// Assumes row-major packing: input[G][N][C][spatial], weight[G][K][C][filter],
// output[G][N][K][spatial]
template <index_t NDimSpatial,
          index_t NumAExtra, // Number of extra A (input) tensors
          index_t NumBExtra, // Number of extra B (weight) tensors
          index_t NumD,      // Number of D tensors
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename DDataType, // D tensor data type
          typename InElementOp,
          typename WeiElementOp,
          typename OutElementOp>
__global__ void naive_conv_fwd_packed_multi_abd(
    const InDataType* const* __restrict__ p_ins,    // Array of input pointers (1 + NumAExtra)
    const WeiDataType* const* __restrict__ p_weis,  // Array of weight pointers (1 + NumBExtra)
    const DDataType* const* __restrict__ p_ds,      // Array of D tensor pointers
    const index_t* const* __restrict__ p_d_strides, // Array of D tensor stride arrays
    OutDataType* __restrict__ p_out,
    index_t G,
    index_t N,
    index_t K,
    index_t C,
    index_t Di,
    index_t Hi,
    index_t Wi,
    index_t Z,
    index_t Y,
    index_t X,
    index_t Do,
    index_t Ho,
    index_t Wo,
    index_t stride_z,
    index_t stride_y,
    index_t stride_x,
    index_t dilation_z,
    index_t dilation_y,
    index_t dilation_x,
    index_t pad_z,
    index_t pad_y,
    index_t pad_x,
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
        const long_index_t num_out      = G * N * K * Wo;
        const long_index_t in_stride_g  = N * C * Wi;
        const long_index_t in_stride_n  = C * Wi;
        const long_index_t in_stride_c  = Wi;
        const long_index_t wei_stride_g = K * C * X;
        const long_index_t wei_stride_k = C * X;
        const long_index_t wei_stride_c = X;
        const long_index_t out_stride_g = N * K * Wo;
        const long_index_t out_stride_n = K * Wo;
        const long_index_t out_stride_k = Wo;

        for(long_index_t idx = tid; idx < num_out; idx += num_threads)
        {
            index_t remaining = idx;
            const index_t wo  = remaining % Wo;
            remaining /= Wo;
            const index_t k = remaining % K;
            remaining /= K;
            const index_t n = remaining % N;
            const index_t g = remaining / N;

            float acc = 0.0f;
            // Base pointers for current group, batch, and output channel
            const InDataType* input_g_n   = p_ins[0] + g * in_stride_g + n * in_stride_n;
            const WeiDataType* weight_g_k = p_weis[0] + g * wei_stride_g + k * wei_stride_k;

            for(index_t c = 0; c < C; ++c)
            {
                // Pointers at current input channel
                const InDataType* input_at_c   = input_g_n + c * in_stride_c;
                const WeiDataType* weight_at_c = weight_g_k + c * wei_stride_c;

                for(index_t x = 0; x < X; ++x)
                {
                    long_index_t wi = wo * stride_x + x * dilation_x - pad_x;
                    if(wi >= 0 && wi < Wi)
                    {
                        // Handle input element-wise operation with extra A tensors
                        detail::apply_multi_tensor_elementwise_op<NumAExtra>(
                            in_val,
                            in_op,
                            input_at_c,
                            p_ins + 1,
                            g * in_stride_g + n * in_stride_n + c * in_stride_c,
                            wi);

                        // Handle weight element-wise operation with extra B tensors
                        detail::apply_multi_tensor_elementwise_op<NumBExtra>(
                            wei_val,
                            wei_op,
                            weight_at_c,
                            p_weis + 1,
                            g * wei_stride_g + k * wei_stride_k + c * wei_stride_c,
                            x);

                        acc += type_convert<float>(in_val) * type_convert<float>(wei_val);
                    }
                }
            }

            detail::apply_d_tensor_elementwise_op<NumD>(
                out_val, out_op, acc, p_ds, p_d_strides, g, n, k, wo);

            p_out[g * out_stride_g + n * out_stride_n + k * out_stride_k + wo] = out_val;
        }
    }
    else if constexpr(NDimSpatial == 2)
    {
        const long_index_t num_out      = G * N * K * Ho * Wo;
        const long_index_t in_stride_g  = N * C * Hi * Wi;
        const long_index_t in_stride_n  = C * Hi * Wi;
        const long_index_t in_stride_c  = Hi * Wi;
        const long_index_t in_stride_h  = Wi;
        const long_index_t wei_stride_g = K * C * Y * X;
        const long_index_t wei_stride_k = C * Y * X;
        const long_index_t wei_stride_c = Y * X;
        const long_index_t wei_stride_y = X;
        const long_index_t out_stride_g = N * K * Ho * Wo;
        const long_index_t out_stride_n = K * Ho * Wo;
        const long_index_t out_stride_k = Ho * Wo;
        const long_index_t out_stride_h = Wo;

        for(long_index_t idx = tid; idx < num_out; idx += num_threads)
        {
            index_t remaining = idx;
            const index_t wo  = remaining % Wo;
            remaining /= Wo;
            const index_t ho = remaining % Ho;
            remaining /= Ho;
            const index_t k = remaining % K;
            remaining /= K;
            const index_t n = remaining % N;
            const index_t g = remaining / N;

            float acc = 0.0f;
            // Base pointers for current group, batch, and output channel
            const InDataType* input_g_n   = p_ins[0] + g * in_stride_g + n * in_stride_n;
            const WeiDataType* weight_g_k = p_weis[0] + g * wei_stride_g + k * wei_stride_k;

            for(index_t c = 0; c < C; ++c)
            {
                // Pointers at current input channel
                const InDataType* input_at_c   = input_g_n + c * in_stride_c;
                const WeiDataType* weight_at_c = weight_g_k + c * wei_stride_c;

                for(index_t y = 0; y < Y; ++y)
                {
                    long_index_t hi = ho * stride_y + y * dilation_y - pad_y;
                    if(hi >= 0 && hi < Hi)
                    {
                        // Pointers at current spatial height and filter Y position
                        const InDataType* input_at_h   = input_at_c + hi * in_stride_h;
                        const WeiDataType* weight_at_y = weight_at_c + y * wei_stride_y;

                        for(index_t x = 0; x < X; ++x)
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
                                    wi);

                                // Handle weight element-wise operation with extra B tensors
                                detail::apply_multi_tensor_elementwise_op<NumBExtra>(
                                    wei_val,
                                    wei_op,
                                    weight_at_y,
                                    p_weis + 1,
                                    g * wei_stride_g + k * wei_stride_k + c * wei_stride_c +
                                        y * wei_stride_y,
                                    x);

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

            p_out[g * out_stride_g + n * out_stride_n + k * out_stride_k + ho * out_stride_h + wo] =
                out_val;
        }
    }
    else if constexpr(NDimSpatial == 3)
    {
        const long_index_t num_out      = G * N * K * Do * Ho * Wo;
        const long_index_t in_stride_g  = N * C * Di * Hi * Wi;
        const long_index_t in_stride_n  = C * Di * Hi * Wi;
        const long_index_t in_stride_c  = Di * Hi * Wi;
        const long_index_t in_stride_d  = Hi * Wi;
        const long_index_t in_stride_h  = Wi;
        const long_index_t wei_stride_g = K * C * Z * Y * X;
        const long_index_t wei_stride_k = C * Z * Y * X;
        const long_index_t wei_stride_c = Z * Y * X;
        const long_index_t wei_stride_z = Y * X;
        const long_index_t wei_stride_y = X;
        const long_index_t out_stride_g = N * K * Do * Ho * Wo;
        const long_index_t out_stride_n = K * Do * Ho * Wo;
        const long_index_t out_stride_k = Do * Ho * Wo;
        const long_index_t out_stride_d = Ho * Wo;
        const long_index_t out_stride_h = Wo;

        for(long_index_t idx = tid; idx < num_out; idx += num_threads)
        {
            index_t remaining = idx;
            const index_t wo  = remaining % Wo;
            remaining /= Wo;
            const index_t ho = remaining % Ho;
            remaining /= Ho;
            const index_t do_idx = remaining % Do;
            remaining /= Do;
            const index_t k = remaining % K;
            remaining /= K;
            const index_t n = remaining % N;
            const index_t g = remaining / N;

            float acc = 0.0f;
            // Base pointers for current group, batch, and output channel
            const InDataType* input_g_n   = p_ins[0] + g * in_stride_g + n * in_stride_n;
            const WeiDataType* weight_g_k = p_weis[0] + g * wei_stride_g + k * wei_stride_k;

            for(index_t c = 0; c < C; ++c)
            {
                // Pointers at current input channel
                const InDataType* input_at_c   = input_g_n + c * in_stride_c;
                const WeiDataType* weight_at_c = weight_g_k + c * wei_stride_c;

                for(index_t z = 0; z < Z; ++z)
                {
                    long_index_t di = do_idx * stride_z + z * dilation_z - pad_z;
                    if(di >= 0 && di < Di)
                    {
                        // Pointers at current spatial depth
                        const InDataType* input_at_d   = input_at_c + di * in_stride_d;
                        const WeiDataType* weight_at_z = weight_at_c + z * wei_stride_z;

                        for(index_t y = 0; y < Y; ++y)
                        {
                            long_index_t hi = ho * stride_y + y * dilation_y - pad_y;
                            if(hi >= 0 && hi < Hi)
                            {
                                // Pointers at current spatial depth and height
                                const InDataType* input_at_d_h   = input_at_d + hi * in_stride_h;
                                const WeiDataType* weight_at_z_y = weight_at_z + y * wei_stride_y;

                                for(index_t x = 0; x < X; ++x)
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
                                            wi);

                                        // Handle weight element-wise operation with extra B tensors
                                        detail::apply_multi_tensor_elementwise_op<NumBExtra>(
                                            wei_val,
                                            wei_op,
                                            weight_at_z_y,
                                            p_weis + 1,
                                            g * wei_stride_g + k * wei_stride_k + c * wei_stride_c +
                                                z * wei_stride_z + y * wei_stride_y,
                                            x);

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

            p_out[g * out_stride_g + n * out_stride_n + k * out_stride_k + do_idx * out_stride_d +
                  ho * out_stride_h + wo] = out_val;
        }
    }
}

// GPU reference convolution with multi-ABD support - takes ConvParam directly
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
          typename TD = TOut> // D tensor type, defaults to TOut for backward compatibility
void naive_conv_fwd_multi_abd(
    const std::array<const TIn*, NumAElementwise + 1>& p_ins,
    const std::array<const TWei*, NumBElementwise + 1>& p_weis,
    const std::array<const TD*, NumDElementwise>& p_ds,
    TOut* p_out,
    const ck::utils::conv::ConvParam& conv_param,
    [[maybe_unused]] const std::array<std::vector<index_t>, NumDElementwise>& d_lengths,
    const std::array<std::vector<index_t>, NumDElementwise>& d_strides,
    InElementwiseOperation in_element_op   = InElementwiseOperation{},
    WeiElementwiseOperation wei_element_op = WeiElementwiseOperation{},
    OutElementwiseOperation out_element_op = OutElementwiseOperation{},
    hipStream_t stream                     = nullptr)
{
    const auto ndim = conv_param.num_dim_spatial_;

    const index_t G = conv_param.G_;
    const index_t N = conv_param.N_;
    const index_t C = conv_param.C_;
    const index_t K = conv_param.K_;

    std::vector<index_t> in_lengths  = {G, N, C};
    std::vector<index_t> wei_lengths = {G, K, C};
    std::vector<index_t> out_lengths = {G, N, K};

    for(index_t i = 0; i < ndim; ++i)
    {
        in_lengths.push_back(static_cast<index_t>(conv_param.input_spatial_lengths_[i]));
        wei_lengths.push_back(static_cast<index_t>(conv_param.filter_spatial_lengths_[i]));
        out_lengths.push_back(static_cast<index_t>(conv_param.output_spatial_lengths_[i]));
    }

    // Calculate total elements for buffer allocation
    long_index_t in_total = 1, wei_total = 1, out_total = 1;
    for(auto l : in_lengths)
        in_total *= l;
    for(auto l : wei_lengths)
        wei_total *= l;
    for(auto l : out_lengths)
        out_total *= l;

    // Allocate packed buffers for all A and B tensors
    // Use separate allocations to avoid copy assignment issues with RAII wrapper
    std::vector<SimpleDeviceMem> in_packed_bufs;
    in_packed_bufs.reserve(NumAElementwise + 1);
    for(index_t i = 0; i <= NumAElementwise; ++i)
    {
        in_packed_bufs.emplace_back(in_total * sizeof(TIn));
    }

    std::vector<SimpleDeviceMem> wei_packed_bufs;
    wei_packed_bufs.reserve(NumBElementwise + 1);
    for(index_t i = 0; i <= NumBElementwise; ++i)
    {
        wei_packed_bufs.emplace_back(wei_total * sizeof(TWei));
    }

    SimpleDeviceMem out_packed_buf(out_total * sizeof(TOut));

    // Get packed buffer pointers
    std::array<TIn*, NumAElementwise + 1> p_ins_packed;
    for(index_t i = 0; i <= NumAElementwise; ++i)
    {
        p_ins_packed[i] = static_cast<TIn*>(in_packed_bufs[i].GetDeviceBuffer());
    }

    std::array<TWei*, NumBElementwise + 1> p_weis_packed;
    for(index_t i = 0; i <= NumBElementwise; ++i)
    {
        p_weis_packed[i] = static_cast<TWei*>(wei_packed_bufs[i].GetDeviceBuffer());
    }

    TOut* p_out_packed = static_cast<TOut*>(out_packed_buf.GetDeviceBuffer());

    // Compute strides and allocate device arrays for pack/unpack
    std::vector<index_t> in_strides  = compute_conv_tensor_strides<InLayout>(in_lengths, ndim);
    std::vector<index_t> wei_strides = compute_conv_tensor_strides<WeiLayout>(wei_lengths, ndim);
    std::vector<index_t> out_strides = compute_conv_tensor_strides<OutLayout>(out_lengths, ndim);

    const size_t dim_count = in_lengths.size();
    SimpleDeviceMem in_lengths_buf(dim_count * sizeof(index_t));
    SimpleDeviceMem in_strides_buf(dim_count * sizeof(index_t));
    SimpleDeviceMem wei_lengths_buf(dim_count * sizeof(index_t));
    SimpleDeviceMem wei_strides_buf(dim_count * sizeof(index_t));
    SimpleDeviceMem out_lengths_buf(dim_count * sizeof(index_t));
    SimpleDeviceMem out_strides_buf(dim_count * sizeof(index_t));

    index_t* d_in_lengths  = static_cast<index_t*>(in_lengths_buf.GetDeviceBuffer());
    index_t* d_in_strides  = static_cast<index_t*>(in_strides_buf.GetDeviceBuffer());
    index_t* d_wei_lengths = static_cast<index_t*>(wei_lengths_buf.GetDeviceBuffer());
    index_t* d_wei_strides = static_cast<index_t*>(wei_strides_buf.GetDeviceBuffer());
    index_t* d_out_lengths = static_cast<index_t*>(out_lengths_buf.GetDeviceBuffer());
    index_t* d_out_strides = static_cast<index_t*>(out_strides_buf.GetDeviceBuffer());

    HIP_CHECK_ERROR(hipMemcpy(
        d_in_lengths, in_lengths.data(), dim_count * sizeof(index_t), hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(
        d_in_strides, in_strides.data(), dim_count * sizeof(index_t), hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(
        d_wei_lengths, wei_lengths.data(), dim_count * sizeof(index_t), hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(
        d_wei_strides, wei_strides.data(), dim_count * sizeof(index_t), hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(
        d_out_lengths, out_lengths.data(), dim_count * sizeof(index_t), hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(
        d_out_strides, out_strides.data(), dim_count * sizeof(index_t), hipMemcpyHostToDevice));

    // Pack input and weight tensors to contiguous layout
    constexpr int block_size = 256;

    // Pack all A tensors
    for(index_t i = 0; i <= NumAElementwise; ++i)
    {
        strided_copy_kernel<TIn, false>
            <<<(in_total + block_size - 1) / block_size, block_size, 0, stream>>>(
                p_ins[i], p_ins_packed[i], d_in_lengths, d_in_strides, dim_count, in_total);
    }

    // Pack all B tensors
    for(index_t i = 0; i <= NumBElementwise; ++i)
    {
        strided_copy_kernel<TWei, false>
            <<<(wei_total + block_size - 1) / block_size, block_size, 0, stream>>>(
                p_weis[i], p_weis_packed[i], d_wei_lengths, d_wei_strides, dim_count, wei_total);
    }

    // Prepare D tensor stride arrays on device
    // NOTE: D tensors are NOT packed - they are used directly with their original strides
    // to support broadcasting (e.g., BiasGK layout with zero strides)
    std::vector<SimpleDeviceMem> d_stride_bufs;
    std::array<index_t*, NumDElementwise> p_d_strides_dev = {};

    if constexpr(NumDElementwise > 0)
    {
        d_stride_bufs.reserve(NumDElementwise);

        for(index_t i = 0; i < NumDElementwise; ++i)
        {
            // Allocate and copy strides to device
            d_stride_bufs.emplace_back(d_strides[i].size() * sizeof(index_t));
            p_d_strides_dev[i] = static_cast<index_t*>(d_stride_bufs[i].GetDeviceBuffer());

            HIP_CHECK_ERROR(hipMemcpy(p_d_strides_dev[i],
                                      d_strides[i].data(),
                                      d_strides[i].size() * sizeof(index_t),
                                      hipMemcpyHostToDevice));
        }
    }

    // Create device arrays of pointers
    SimpleDeviceMem ins_ptrs_buf((NumAElementwise + 1) * sizeof(TIn*));
    SimpleDeviceMem weis_ptrs_buf((NumBElementwise + 1) * sizeof(TWei*));
    SimpleDeviceMem ds_ptrs_buf(NumDElementwise * sizeof(TD*));
    SimpleDeviceMem d_strides_ptrs_buf(NumDElementwise * sizeof(index_t*));

    TIn** d_ins_ptrs           = static_cast<TIn**>(ins_ptrs_buf.GetDeviceBuffer());
    TWei** d_weis_ptrs         = static_cast<TWei**>(weis_ptrs_buf.GetDeviceBuffer());
    TD** d_ds_ptrs             = static_cast<TD**>(ds_ptrs_buf.GetDeviceBuffer());
    index_t** d_d_strides_ptrs = static_cast<index_t**>(d_strides_ptrs_buf.GetDeviceBuffer());

    HIP_CHECK_ERROR(hipMemcpy(d_ins_ptrs,
                              p_ins_packed.data(),
                              (NumAElementwise + 1) * sizeof(TIn*),
                              hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(d_weis_ptrs,
                              p_weis_packed.data(),
                              (NumBElementwise + 1) * sizeof(TWei*),
                              hipMemcpyHostToDevice));

    if constexpr(NumDElementwise > 0)
    {
        // D tensors use original pointers (not packed) to support broadcasting
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
    std::vector<index_t> conv_strides(ndim);
    std::vector<index_t> conv_dilations(ndim);
    std::vector<index_t> input_pads(ndim);
    for(index_t i = 0; i < ndim; ++i)
    {
        conv_strides[i]   = static_cast<index_t>(conv_param.conv_filter_strides_[i]);
        conv_dilations[i] = static_cast<index_t>(conv_param.conv_filter_dilations_[i]);
        input_pads[i]     = static_cast<index_t>(conv_param.input_left_pads_[i]);
    }

    // Run convolution kernel on packed data
    const int out_grid = (out_total + block_size - 1) / block_size;

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
                                                  p_out_packed,
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
                                                  p_out_packed,
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
                                                  p_out_packed,
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
                                                  in_element_op,
                                                  wei_element_op,
                                                  out_element_op);
    }

    // Unpack
    strided_copy_kernel<TOut, true><<<out_grid, block_size, 0, stream>>>(
        p_out_packed, p_out, d_out_lengths, d_out_strides, dim_count, out_total);

    HIP_CHECK_ERROR(hipGetLastError());

    // Memory automatically freed by SimpleDeviceMem destructors
}

// Original naive_conv_fwd - now a zero-overhead wrapper
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
    std::array<const TIn*, 1> p_ins               = {p_in};
    std::array<const TWei*, 1> p_weis             = {p_wei};
    std::array<const TOut*, 0> p_ds               = {};
    std::array<std::vector<index_t>, 0> d_lengths = {};
    std::array<std::vector<index_t>, 0> d_strides = {};

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
