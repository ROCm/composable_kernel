// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/type_convert.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/reference_tensor_operation/gpu/conv_common.hpp"
#include "ck/library/reference_tensor_operation/gpu/pack_unpack_kernels.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {
namespace ref {

// Optimized convolution kernel working with packed (contiguous) tensors
// Assumes row-major packing: input[G][N][C][spatial], weight[G][K][C][filter],
// output[G][N][K][spatial]
template <index_t NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InElementOp,
          typename WeiElementOp,
          typename OutElementOp>
__global__ void naive_conv_fwd_packed(const InDataType* __restrict__ p_in,
                                      const WeiDataType* __restrict__ p_wei,
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
                                      index_t pad_x)
{
    const long_index_t tid         = blockIdx.x * blockDim.x + threadIdx.x;
    const long_index_t num_threads = blockDim.x * gridDim.x;

    constexpr auto in_op  = InElementOp{};
    constexpr auto wei_op = WeiElementOp{};
    constexpr auto out_op = OutElementOp{};

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

            float acc                 = 0.0f;
            const InDataType* in_g    = p_in + g * in_stride_g + n * in_stride_n;
            const WeiDataType* wei_gk = p_wei + g * wei_stride_g + k * wei_stride_k;

            for(index_t c = 0; c < C; ++c)
            {
                const InDataType* in_gc    = in_g + c * in_stride_c;
                const WeiDataType* wei_gkc = wei_gk + c * wei_stride_c;

                for(index_t x = 0; x < X; ++x)
                {
                    long_index_t wi = wo * stride_x + x * dilation_x - pad_x;
                    if(wi >= 0 && wi < Wi)
                    {
                        in_op(in_val, in_gc[wi]);
                        wei_op(wei_val, wei_gkc[x]);
                        acc += type_convert<float>(in_val) * type_convert<float>(wei_val);
                    }
                }
            }

            OutDataType result = type_convert<OutDataType>(acc);
            out_op(out_val, result);
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

            float acc                 = 0.0f;
            const InDataType* in_gn   = p_in + g * in_stride_g + n * in_stride_n;
            const WeiDataType* wei_gk = p_wei + g * wei_stride_g + k * wei_stride_k;

            for(index_t c = 0; c < C; ++c)
            {
                const InDataType* in_gnc   = in_gn + c * in_stride_c;
                const WeiDataType* wei_gkc = wei_gk + c * wei_stride_c;

                for(index_t y = 0; y < Y; ++y)
                {
                    long_index_t hi = ho * stride_y + y * dilation_y - pad_y;
                    if(hi >= 0 && hi < Hi)
                    {
                        const InDataType* in_gnch   = in_gnc + hi * in_stride_h;
                        const WeiDataType* wei_gkcy = wei_gkc + y * wei_stride_y;

                        for(index_t x = 0; x < X; ++x)
                        {
                            long_index_t wi = wo * stride_x + x * dilation_x - pad_x;
                            if(wi >= 0 && wi < Wi)
                            {
                                in_op(in_val, in_gnch[wi]);
                                wei_op(wei_val, wei_gkcy[x]);
                                acc += type_convert<float>(in_val) * type_convert<float>(wei_val);
                            }
                        }
                    }
                }
            }

            OutDataType result = type_convert<OutDataType>(acc);
            out_op(out_val, result);
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

            float acc                 = 0.0f;
            const InDataType* in_gn   = p_in + g * in_stride_g + n * in_stride_n;
            const WeiDataType* wei_gk = p_wei + g * wei_stride_g + k * wei_stride_k;

            for(index_t c = 0; c < C; ++c)
            {
                const InDataType* in_gnc   = in_gn + c * in_stride_c;
                const WeiDataType* wei_gkc = wei_gk + c * wei_stride_c;

                for(index_t z = 0; z < Z; ++z)
                {
                    long_index_t di = do_idx * stride_z + z * dilation_z - pad_z;
                    if(di >= 0 && di < Di)
                    {
                        const InDataType* in_gncd   = in_gnc + di * in_stride_d;
                        const WeiDataType* wei_gkcz = wei_gkc + z * wei_stride_z;

                        for(index_t y = 0; y < Y; ++y)
                        {
                            long_index_t hi = ho * stride_y + y * dilation_y - pad_y;
                            if(hi >= 0 && hi < Hi)
                            {
                                const InDataType* in_gncdh   = in_gncd + hi * in_stride_h;
                                const WeiDataType* wei_gkczy = wei_gkcz + y * wei_stride_y;

                                for(index_t x = 0; x < X; ++x)
                                {
                                    long_index_t wi = wo * stride_x + x * dilation_x - pad_x;
                                    if(wi >= 0 && wi < Wi)
                                    {
                                        in_op(in_val, in_gncdh[wi]);
                                        wei_op(wei_val, wei_gkczy[x]);
                                        acc += type_convert<float>(in_val) *
                                               type_convert<float>(wei_val);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            OutDataType result = type_convert<OutDataType>(acc);
            out_op(out_val, result);
            p_out[g * out_stride_g + n * out_stride_n + k * out_stride_k + do_idx * out_stride_d +
                  ho * out_stride_h + wo] = out_val;
        }
    }
}

// GPU reference convolution - takes lengths/strides directly
// Used by both standalone tests and profiler
template <typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename TIn,
          typename TWei,
          typename TOut,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation>
void naive_conv_fwd(const TIn* p_in,
                    const TWei* p_wei,
                    TOut* p_out,
                    const std::vector<index_t>& in_lengths,
                    const std::vector<index_t>& in_strides,
                    const std::vector<index_t>& wei_lengths,
                    const std::vector<index_t>& wei_strides,
                    const std::vector<index_t>& out_lengths,
                    const std::vector<index_t>& out_strides,
                    const std::vector<index_t>& conv_strides,
                    const std::vector<index_t>& conv_dilations,
                    const std::vector<index_t>& input_pads,
                    hipStream_t stream = nullptr)
{
    // Just call the optimized pack/conv/unpack from the same file
    constexpr int block_size = 256;

    // Calculate total elements
    long_index_t in_total = 1, wei_total = 1, out_total = 1;
    for(auto l : in_lengths)
        in_total *= l;
    for(auto l : wei_lengths)
        wei_total *= l;
    for(auto l : out_lengths)
        out_total *= l;

    // Determine NDimSpatial from length array size
    index_t NDimSpatial = in_lengths.size() - 3;

    // Extract dimensions
    index_t G = in_lengths[0];
    index_t N = in_lengths[1];
    index_t C = wei_lengths[2]; // Total C
    index_t K = wei_lengths[1]; // Total K per group

    // Allocate packed buffers
    TIn* p_in_packed;
    TWei* p_wei_packed;
    TOut* p_out_packed;
    HIP_CHECK_ERROR(hipMalloc(&p_in_packed, in_total * sizeof(TIn)));
    HIP_CHECK_ERROR(hipMalloc(&p_wei_packed, wei_total * sizeof(TWei)));
    HIP_CHECK_ERROR(hipMalloc(&p_out_packed, out_total * sizeof(TOut)));

    // Allocate device arrays
    index_t *d_in_lengths, *d_in_strides, *d_wei_lengths, *d_wei_strides, *d_out_lengths,
        *d_out_strides;
    const size_t dim_count = in_lengths.size();
    HIP_CHECK_ERROR(hipMalloc(&d_in_lengths, dim_count * sizeof(index_t)));
    HIP_CHECK_ERROR(hipMalloc(&d_in_strides, dim_count * sizeof(index_t)));
    HIP_CHECK_ERROR(hipMalloc(&d_wei_lengths, dim_count * sizeof(index_t)));
    HIP_CHECK_ERROR(hipMalloc(&d_wei_strides, dim_count * sizeof(index_t)));
    HIP_CHECK_ERROR(hipMalloc(&d_out_lengths, dim_count * sizeof(index_t)));
    HIP_CHECK_ERROR(hipMalloc(&d_out_strides, dim_count * sizeof(index_t)));

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

    // Pack
    int in_grid = (in_total + block_size - 1) / block_size;
    pack_strided_tensor<<<in_grid, block_size, 0, stream>>>(
        p_in, p_in_packed, d_in_lengths, d_in_strides, dim_count, in_total);

    int wei_grid = (wei_total + block_size - 1) / block_size;
    pack_strided_tensor<<<wei_grid, block_size, 0, stream>>>(
        p_wei, p_wei_packed, d_wei_lengths, d_wei_strides, dim_count, wei_total);

    // Run convolution
    int out_grid = (out_total + block_size - 1) / block_size;

    if(NDimSpatial == 1)
    {
        naive_conv_fwd_packed<1,
                              TIn,
                              TWei,
                              TOut,
                              InElementwiseOperation,
                              WeiElementwiseOperation,
                              OutElementwiseOperation>
            <<<out_grid, block_size, 0, stream>>>(p_in_packed,
                                                  p_wei_packed,
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
                                                  input_pads[0]);
    }
    else if(NDimSpatial == 2)
    {
        naive_conv_fwd_packed<2,
                              TIn,
                              TWei,
                              TOut,
                              InElementwiseOperation,
                              WeiElementwiseOperation,
                              OutElementwiseOperation>
            <<<out_grid, block_size, 0, stream>>>(p_in_packed,
                                                  p_wei_packed,
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
                                                  input_pads[1]);
    }
    else // 3D
    {
        naive_conv_fwd_packed<3,
                              TIn,
                              TWei,
                              TOut,
                              InElementwiseOperation,
                              WeiElementwiseOperation,
                              OutElementwiseOperation>
            <<<out_grid, block_size, 0, stream>>>(p_in_packed,
                                                  p_wei_packed,
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
                                                  input_pads[2]);
    }

    // Unpack
    unpack_to_strided_tensor<<<out_grid, block_size, 0, stream>>>(
        p_out_packed, p_out, d_out_lengths, d_out_strides, dim_count, out_total);

    HIP_CHECK_ERROR(hipGetLastError());

    // Free buffers
    HIP_CHECK_ERROR(hipFree(p_in_packed));
    HIP_CHECK_ERROR(hipFree(p_wei_packed));
    HIP_CHECK_ERROR(hipFree(p_out_packed));
    HIP_CHECK_ERROR(hipFree(d_in_lengths));
    HIP_CHECK_ERROR(hipFree(d_in_strides));
    HIP_CHECK_ERROR(hipFree(d_wei_lengths));
    HIP_CHECK_ERROR(hipFree(d_wei_strides));
    HIP_CHECK_ERROR(hipFree(d_out_lengths));
    HIP_CHECK_ERROR(hipFree(d_out_strides));
}

} // namespace ref
} // namespace ck
