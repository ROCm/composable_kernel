// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <typeinfo>
#include <sstream>

#include "ck_tile/host.hpp"
#include "ck_tile/host/convolution_parameter.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "ck_tile/ops/grouped_convolution/utils/grouped_convolution_utils.hpp"
#include "ck_tile/library/tensor_operation_instance/gpu/tile_grouped_conv_bwd_data_factory.hpp"
#include "ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_data_kernel.hpp"
#include "ck_tile/host/reference/reference_grouped_conv_bwd_data.hpp"
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
          typename ComputeTypeA = InDataType,
          typename ComputeTypeB = ComputeTypeA>
bool profile_grouped_conv_bwd_data_impl(int do_verification,
                                          int init_method,
                                          bool /*do_log*/,
                                          bool time_kernel,
                                          const ck_tile::conv::ConvParam& conv_param,
                                          const std::string& split_k,
                                          ck_tile::index_t instance_index = -1)
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

    std::cout << "input: " << input.mDesc << std::endl;
    std::cout << "weight: " << weight.mDesc << std::endl;
    std::cout << "output: " << output.mDesc << std::endl;

    switch(init_method)
    {
    case 0:
        ck_tile::FillUniformDistribution<WeiDataType>{-1.f, 1.f}(weight);
        ck_tile::FillUniformDistribution<OutDataType>{-1.f, 1.f}(output);
        break;
    case 1:
        ck_tile::FillMonotonicSeq<WeiDataType>{}(weight);
        ck_tile::FillMonotonicSeq<OutDataType>{}(output);
        break;
    case 2:
        ck_tile::FillUniformDistribution<WeiDataType>{1.f, 1.f}(weight);
        ck_tile::FillUniformDistribution<OutDataType>{1.f, 1.f}(output);
        break;
    default:
        input.SetZero();
        output.SetZero();
    }

    using DeviceOp = ops::GroupedConvolutionBackwardDataBaseInvoker<
                                                        NDimSpatial,
                                                        InLayout,
                                                        WeiLayout,
                                                        OutLayout,
                                                        InDataType,
                                                        WeiDataType,
                                                        OutDataType,
                                                        InElementOp,
                                                        WeiElementOp,
                                                        OutElementOp,
                                                        ComputeTypeA,
                                                        ComputeTypeB>;

    // get device op instances
    const auto ops = ck_tile::ops::DeviceOperationInstanceFactory<DeviceOp>::GetInstances();

    std::cout << "found " << ops.size() << " instances" << std::endl;

    std::string best_op_name;
    float best_avg_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;
    std::string best_split_k("1");

    std::vector<ck_tile::index_t> split_k_list = {1, 2, 4, 8, 16, 32, 64, 128};
    if(split_k != "all")
    {
        try
        {
            ck_tile::index_t split_k_value = std::stoi(split_k);
            split_k_list              = {split_k_value};
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            exit(EXIT_FAILURE);
        }
    }

    // First, calculate the reference result if verification is needed.
    ck_tile::HostTensor<InDataType> input_host_ref(in_g_n_c_wis_desc);
    input_host_ref.SetZero();
    if (do_verification)
    {
        ck_tile::reference_grouped_conv_bwd_data<NDimSpatial, InDataType, WeiDataType, OutDataType>(
                            input_host_ref,
                            weight,
                            output,
                            conv_param.conv_filter_strides_,
                            conv_param.conv_filter_dilations_,
                            conv_param.input_left_pads_,
                            conv_param.input_right_pads_);
    }


    index_t num_kernel = 0;
    bool all_pass = true;
    for(auto& op : ops)
    {
        for(std::size_t split_k_id = 0; split_k_id < split_k_list.size(); split_k_id++)
        {
            auto split_k_value     = split_k_list[split_k_id];
            auto split_k_param_str = std::to_string(split_k_value);

            ck_tile::DeviceMem input_dev_buf(input.get_element_space_size_in_bytes());
            ck_tile::DeviceMem weight_dev_buf(weight.get_element_space_size_in_bytes());
            ck_tile::DeviceMem output_dev_buf(output.get_element_space_size_in_bytes());

            input_dev_buf.SetZero();
            weight_dev_buf.ToDevice(weight.data());
            output_dev_buf.ToDevice(output.data());

            ck_tile::GroupedConvBwdDataHostArgs args(conv_param,
                                               input_dev_buf.GetDeviceBuffer(),
                                               weight_dev_buf.GetDeviceBuffer(),
                                               {},
                                               output_dev_buf.GetDeviceBuffer(),
                                               split_k_value);

            // Split-K autodeduction is not supported.                                   
            if(op->IsSupportedArgument(args) && split_k_value >= 1)
            {
                num_kernel++;
                if((instance_index != -1) && (instance_index + 1 != num_kernel))
                {
                    // skip test if instance_index is specified
                    continue;
                }

                std::string op_name = op->GetName(args);
                std::cout << op_name << ", SplitK " << split_k_param_str << " is profiled..." << std::endl;

                // Run verification first. If it doesn't pass, no need to do performance measurement.
                bool pass = false; 
                if(do_verification)
                {
                    constexpr int n_warmup = 0;
                    constexpr int n_repeat = 1;

                    op->Run(args, false, n_warmup, n_repeat);
                    input_dev_buf.FromDevice(input.data());

                    const ck_tile::index_t GemmK = weight.get_element_size() / (conv_param.G_ * conv_param.K_);
                    const float max_accumulated_value =
                        *std::max_element(input_host_ref.mData.begin(), input_host_ref.mData.end());
                    const auto rtol_atol =
                        calculate_rtol_atol<InDataType, WeiDataType, AccDataType, OutDataType>(
                            GemmK, split_k_value, max_accumulated_value);

                    pass = ck_tile::check_err(input,
                                            input_host_ref,
                                            "Error: Incorrect results!",
                                            rtol_atol.at(ck_tile::number<0>{}),
                                            rtol_atol.at(ck_tile::number<1>{}));

                    std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                            << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                            << std::endl;
                    std::cout << "The CPU verification result is:" << (pass ? "correct" : "fail") << std::endl;

                    all_pass &= pass;
                }

                bool is_valid = do_verification ? pass : true;

                if (is_valid)
                {
                    constexpr int n_warmup = 5;
                    constexpr int n_repeat = 50;
                    float avg_time = op->Run(args, time_kernel, n_warmup, n_repeat);

                    std::size_t flop      = conv_param.GetFlops();
                    std::size_t num_btype = conv_param.GetByte<InDataType, WeiDataType, OutDataType>();

                    float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
                    float gb_per_sec = num_btype / 1.E6 / avg_time;

                    std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << tflops
                            << " TFlops, " << gb_per_sec << " GB/s, " << op_name << ", SplitK "
                            << split_k_param_str << std::endl;
                    
                    if(tflops > best_tflops)
                    {
                        best_op_name    = op_name;
                        best_tflops     = tflops;
                        best_avg_time   = avg_time;
                        best_gb_per_sec = gb_per_sec;
                        best_split_k    = split_k_param_str;
                    }
                }
            }
            else 
            {
                std::cout << op->GetName(args) << ", SplitK " << split_k_param_str
                          << " does not support this problem." << std::endl;
            }
        }
    }

    std::stringstream ss;
    ss << "\n********************************" 
       << "\nCK Tile best configuration parameters:" 
       << "\n********************************" 
       << "\nname: " << best_op_name
       << "\navg_time: " << best_avg_time << "\ntflops: " << best_tflops
       << "\nGB/s: " << best_gb_per_sec 
       << "\nSplitK: " << best_split_k
       << std::endl;

    std::cout << ss.str();

    const char* log_file = std::getenv("CK_TILE_PROFILER_LOG_FILE");
    if(log_file != nullptr)
    {
        std::ofstream out(log_file, std::ios::app);
        if(out.is_open())
        {
            std::stringstream out_ss;
            out_ss << "CK Tile best configuration:" << std::endl
                   << "name: " << best_op_name << std::endl
                   << "avg_time: " << best_avg_time << std::endl
                   << "SplitK: " << best_split_k << std::endl
                   << "all_pass " << (all_pass ? "true" : "false") << std::endl;
            out << out_ss.str();
            out.close();
        }
    }

    if(instance_index != -1)
    {
        std::cout << "grouped_conv_bwd_data_instance (" << instance_index << "/" << num_kernel
                  << "): Passed" << std::endl;
    }
    return all_pass;
}

} // namespace profiler
} // namespace ck_tile
