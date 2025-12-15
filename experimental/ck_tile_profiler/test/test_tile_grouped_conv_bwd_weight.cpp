// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <typeinfo>
#include <sstream>
#include <gtest/gtest.h>

#include "ck_tile/host.hpp"
#include "ck_tile/host/convolution_parameter.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "ck_tile/ops/grouped_convolution/utils/grouped_convolution_utils.hpp"
#include "ck_tile/ref/naive_grouped_conv_bwd_weight_gpu.hpp"

#include "ck_tile_profiler/gemm_configs.hpp"
#include "ck_tile_profiler/tile_grouped_conv_bwd_weight_invoker.hpp"

namespace ck_tile {
namespace profiler {

template <typename InDataType, typename WeiDataType, typename AccDataType, typename OutDataType>
auto calculate_rtol_atol(const ck_tile::index_t GemmK,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(InDataType) < sizeof(WeiDataType), InDataType, WeiDataType>;
    // Calculate thresholds
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, OutDataType, AccDataType>(
        ck_tile::integer_divide_ceil(GemmK, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, OutDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(GemmK, kbatch));
    // Calculate error due to split_k accumulation
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<OutDataType, OutDataType, OutDataType>(kbatch);
    const auto atol_split_k =
        ck_tile::get_absolute_threshold<OutDataType, OutDataType, OutDataType>(
            max_accumulated_value, kbatch);
    // Use higher threshold
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

template <ck_tile::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          index_t PipelineVersion,
          index_t VectorSize,
          typename ComputeTypeA = InDataType,
          typename ComputeTypeB = ComputeTypeA>
bool profile_grouped_conv_bwd_weight_impl(const ck_tile::conv::ConvParam& conv_param,
                                          const int split_k)
{
    using AccDataType  = float;
    using InElementOp  = ck_tile::element_wise::PassThrough;
    using WeiElementOp = ck_tile::element_wise::PassThrough;
    using OutElementOp = ck_tile::element_wise::PassThrough;

    const auto in_g_n_c_wis_desc =
        ck_tile::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);
    const auto wei_g_k_c_xs_desc =
        ck_tile::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);
    const auto out_g_n_k_wos_desc =
        ck_tile::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    ck_tile::HostTensor<InDataType> input(in_g_n_c_wis_desc);
    ck_tile::HostTensor<WeiDataType> weight(wei_g_k_c_xs_desc);
    ck_tile::HostTensor<OutDataType> output(out_g_n_k_wos_desc);

    ck_tile::FillMonotonicSeq<WeiDataType>{}(output);
    ck_tile::FillMonotonicSeq<InDataType>{}(input);

    ck_tile::DeviceMem input_dev_buf(input.get_element_space_size_in_bytes());
    ck_tile::DeviceMem weight_dev_buf(weight.get_element_space_size_in_bytes());
    ck_tile::DeviceMem output_dev_buf(output.get_element_space_size_in_bytes());

    weight_dev_buf.SetZero();
    input_dev_buf.ToDevice(input.data());
    output_dev_buf.ToDevice(output.data());

    // GPU reference verification
    ck_tile::DeviceMem weight_ref_dev_buf(weight.get_element_space_size_in_bytes());
    weight_ref_dev_buf.SetZero();

    // Launch GPU reference kernel
    ck_tile::naive_grouped_conv_bwd_weight<NDimSpatial, InDataType, WeiDataType, OutDataType>(
        reinterpret_cast<const InDataType*>(input_dev_buf.GetDeviceBuffer()),
        reinterpret_cast<WeiDataType*>(weight_ref_dev_buf.GetDeviceBuffer()),
        reinterpret_cast<const OutDataType*>(output_dev_buf.GetDeviceBuffer()),
        conv_param.G_,
        conv_param.N_,
        conv_param.K_,
        conv_param.C_,
        conv_param.input_spatial_lengths_,
        conv_param.filter_spatial_lengths_,
        conv_param.output_spatial_lengths_,
        conv_param.conv_filter_strides_,
        conv_param.conv_filter_dilations_,
        conv_param.input_left_pads_);

    const auto op =
        ck_tile::ops::GroupedConvolutionBackwardWeightInvoker<NDimSpatial,
                                                              InLayout,
                                                              WeiLayout,
                                                              OutLayout,
                                                              InDataType,
                                                              WeiDataType,
                                                              OutDataType,
                                                              InElementOp,
                                                              WeiElementOp,
                                                              OutElementOp,
                                                              ConvolutionSpecialization::Default,
                                                              1,
                                                              64,
                                                              16,
                                                              64,
                                                              4,
                                                              1,
                                                              1,
                                                              16,
                                                              16,
                                                              32,
                                                              16,
                                                              2,
                                                              2,
                                                              false,
                                                              CK_TILE_PIPELINE_COMPUTE_V3>{};

    ck_tile::GroupedConvBwdWeightHostArgs args(conv_param,
                                               input_dev_buf.GetDeviceBuffer(),
                                               weight_dev_buf.GetDeviceBuffer(),
                                               {},
                                               output_dev_buf.GetDeviceBuffer(),
                                               split_k);

    // Split-K autodeduction is not supported.
    if(op.IsSupportedArgument(args))
    {

        constexpr int n_warmup = 0;
        constexpr int n_repeat = 1;

        op.Run(args, false, n_warmup, n_repeat);
        weight_dev_buf.FromDevice(weight.data());

        // Copy GPU reference result to host for comparison
        ck_tile::HostTensor<WeiDataType> weight_gpu_ref(wei_g_k_c_xs_desc);
        weight_ref_dev_buf.FromDevice(weight_gpu_ref.data());

        ck_tile::index_t GemmK = conv_param.N_;
        for(ck_tile::index_t i = 0; i < NDimSpatial; ++i)
        {
            GemmK *= conv_param.output_spatial_lengths_[i];
        }
        const float max_accumulated_value =
            *std::max_element(weight_gpu_ref.mData.begin(), weight_gpu_ref.mData.end());
        const auto rtol_atol =
            calculate_rtol_atol<InDataType, WeiDataType, AccDataType, OutDataType>(
                GemmK, split_k, max_accumulated_value);
        return ck_tile::check_err(weight,
                                  weight_gpu_ref,
                                  "Error: Incorrect results!",
                                  rtol_atol.at(ck_tile::number<0>{}),
                                  rtol_atol.at(ck_tile::number<1>{}));
    }
    else
    {
        return true;
    }
}

TEST(TileGroupedConv, BackwardWeight)
{
    std::vector<ck_tile::conv::ConvParam> conv_params;
    conv_params.push_back({2, 2, 32, 128, 256, {1, 1}, {7, 7}, {2, 2}, {1, 1}, {0, 0}, {0, 0}});
    conv_params.push_back({2, 2, 32, 128, 256, {3, 3}, {14, 14}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    conv_params.push_back({2, 2, 32, 128, 256, {1, 1}, {3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});
    conv_params.push_back({2, 1, 1, 1, 32, {3, 3}, {32, 32}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    conv_params.push_back({2, 1, 1, 64, 3, {3, 3}, {32, 32}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    conv_params.push_back({2, 1, 1, 1, 1, {3, 3}, {32, 32}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    conv_params.push_back({2, 96, 1, 1, 1, {3, 3}, {120, 160}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});

    for(auto conv_param : conv_params)
    {
        for(int split_k = 1; split_k <= 4; split_k++)
        {
            bool passed = profile_grouped_conv_bwd_weight_impl<
                2,                                          /*NDimSpatial*/
                ck_tile::tensor_layout::convolution::NHWGC, /*InLayout*/
                ck_tile::tensor_layout::convolution::GKYXC, /*WeiLayout*/
                ck_tile::tensor_layout::convolution::NHWGK, /*OutLayout*/
                ck_tile::bfloat16_t,                        /*InDataType*/
                ck_tile::bfloat16_t,                        /*WeiDataType*/
                ck_tile::bfloat16_t,                        /*OutDataType*/
                CK_TILE_PIPELINE_COMPUTE_V3,
                2,
                ck_tile::bfloat16_t, /*ComputeTypeA*/
                ck_tile::bfloat16_t /*ComputeTypeB*/>(conv_param /*params*/, split_k);

            passed &= profile_grouped_conv_bwd_weight_impl<
                2,                                          /*NDimSpatial*/
                ck_tile::tensor_layout::convolution::NHWGC, /*InLayout*/
                ck_tile::tensor_layout::convolution::GKYXC, /*WeiLayout*/
                ck_tile::tensor_layout::convolution::NHWGK, /*OutLayout*/
                ck_tile::bfloat16_t,                        /*InDataType*/
                ck_tile::bfloat16_t,                        /*WeiDataType*/
                ck_tile::bfloat16_t,                        /*OutDataType*/
                CK_TILE_PIPELINE_COMPUTE_V3,
                1,
                ck_tile::bfloat16_t, /*ComputeTypeA*/
                ck_tile::bfloat16_t /*ComputeTypeB*/>(conv_param /*params*/, split_k);

            passed &= profile_grouped_conv_bwd_weight_impl<
                2,                                          /*NDimSpatial*/
                ck_tile::tensor_layout::convolution::NHWGC, /*InLayout*/
                ck_tile::tensor_layout::convolution::GKYXC, /*WeiLayout*/
                ck_tile::tensor_layout::convolution::NHWGK, /*OutLayout*/
                ck_tile::bfloat16_t,                        /*InDataType*/
                ck_tile::bfloat16_t,                        /*WeiDataType*/
                ck_tile::bfloat16_t,                        /*OutDataType*/
                CK_TILE_PIPELINE_MEMORY,
                2,
                ck_tile::bfloat16_t, /*ComputeTypeA*/
                ck_tile::bfloat16_t /*ComputeTypeB*/>(conv_param /*params*/, split_k);

            EXPECT_TRUE(passed);
        }
    }
}

} // namespace profiler
} // namespace ck_tile
