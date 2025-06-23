// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <typeinfo>
#include <sstream>

#include "ck/ck.hpp"
#include "ck/utility/env.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/split_k_arg.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_weight.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_weight.hpp"

CK_DECLARE_ENV_VAR_STR(CK_PROFILER_DISABLED_OPS)
CK_DECLARE_ENV_VAR_STR(CK_PROFILER_OUTPUT_FILE)

namespace ck {
namespace profiler {

struct PerfResults
{
    void update_best_occupancy_split_k(const std::string& op_name, float avg_time, ck::index_t split_k_arg, ck::index_t oversubscription)
    {
        if(avg_time < best_occupancy_split_k_avg_time_)
        {
            best_occupancy_split_k_op_name_    = op_name;
            best_occupancy_split_k_avg_time_   = avg_time;
            best_occupancy_split_k_value_      = split_k_arg;
            best_occupancy_split_k_oversubscription_ = oversubscription;
        }

        ranking_.emplace_back(op_name, split_k_arg, oversubscription, avg_time);
        std::sort(ranking_.begin(), ranking_.end(),
                  [](const auto& a, const auto& b) { return std::get<3>(a) < std::get<3>(b); });
    };

    void update_fixed_split_k(const std::string& op_name, float avg_time, ck::index_t split_k_arg)
    {
        if (avg_time < fixed_split_k_avg_time_)
        {
            fixed_split_k_op_name_    = op_name;
            fixed_split_k_avg_time_   = avg_time;
            fixed_split_k_value_      = split_k_arg;
        }

        ranking_.emplace_back(op_name, split_k_arg, -1, avg_time);
        std::sort(ranking_.begin(), ranking_.end(),
                  [](const auto& a, const auto& b) { return std::get<3>(a) < std::get<3>(b); });
    };

    static std::string split_k_str(const ck::tensor_operation::device::ParamsSplitK split_k_params, ck::index_t split_k_arg)
    {
        return split_k_params.split_k_mode_ == ck::tensor_operation::device::SplitKMode::BestOccupancyWithOversubscription
            ? std::to_string(split_k_arg) + " (best occupancy, oversubscription = " + std::to_string(split_k_params.oversubscription_) + ")"
            : std::to_string(split_k_params.split_k_value_);
    };
 
    std::string print_fixed_split_k() const
    {
        ck::index_t rank, total_num;
        std::tie(rank, total_num) = get_ranking(fixed_split_k_op_name_, fixed_split_k_value_);
        std::stringstream ss;
        ss << "\nFIXED SPLIT-K RESULTS"
           << "\n========================";
        ss << "\nname: " << fixed_split_k_op_name_ 
            << "\navg_time: " << fixed_split_k_avg_time_
            << "\nGEMM-K: " << k_dim_size_
            << "\nSplitK " << fixed_split_k_value_
            << "\nRanking: " << rank << " / " << total_num;
        return ss.str();
    }

    std::string print_best_occupancy_split_k() const
    {
        ck::index_t rank, total_num;
        std::tie(rank, total_num) = get_ranking(best_occupancy_split_k_op_name_, best_occupancy_split_k_value_, best_occupancy_split_k_oversubscription_);
        std::stringstream ss;
        ss << "\nBEST OCCUPANCY SPLIT-K RESULTS"
           << "\n========================";
        ss << "\nname: " << best_occupancy_split_k_op_name_ 
            << "\navg_time: " << best_occupancy_split_k_avg_time_
            << "\nGEMM-K: " << k_dim_size_
            << "\nOversubscription: " << best_occupancy_split_k_oversubscription_
            << "\nSplitK " << best_occupancy_split_k_value_
            << "\nRanking: " << rank << " / " << total_num;
        return ss.str();
    }

    std::tuple<size_t, size_t> get_ranking(const std::string& op_name, ck::index_t split_k, ck::index_t oversubscription=-1) const
    {
        auto it = std::find_if(ranking_.begin(), ranking_.end(),
                               [&](const auto& entry) {
                                   return std::get<0>(entry) == op_name && std::get<1>(entry) == split_k && 
                                          (oversubscription < 0 || std::get<2>(entry) == oversubscription);
                               });
        if(it != ranking_.end())
        {
            const auto ranking = std::distance(ranking_.begin(), it) + 1;
            return std::make_tuple(ranking, ranking_.size());
        }
        return std::make_tuple(ranking_.size()+1, ranking_.size());
    };

    void set_k_dim_size(ck::index_t k_dim_size)
    {
        if (k_dim_size_ > 0 && k_dim_size != k_dim_size_)
        {
            std::cerr << "Error: k_dim_size cannot be set multiple times. Old value " << k_dim_size_ << ". New value " << k_dim_size << std::endl;
            exit(EXIT_FAILURE);
        }
        k_dim_size_ = k_dim_size;
    }

    // Fixed split-K results
    std::string fixed_split_k_op_name_{""};
    float fixed_split_k_avg_time_{std::numeric_limits<float>::max()};
    ck::index_t fixed_split_k_value_{0};

    // Best occupancy split-K results
    std::string best_occupancy_split_k_op_name_{""};
    float best_occupancy_split_k_avg_time_{std::numeric_limits<float>::max()};
    ck::index_t best_occupancy_split_k_value_{0};
    ck::index_t best_occupancy_split_k_oversubscription_{0};

    // K-dim size
    ck::index_t k_dim_size_ = -1;

    std::vector<std::tuple<std::string, ck::index_t, ck::index_t, float>> ranking_;
};

void write_perf_results_to_file(const PerfResults& perf_results_global, 
                                const std::vector<PerfResults>& perf_results_list)
{
    const auto& results_file = ck::EnvGetString(CK_ENV(CK_PROFILER_OUTPUT_FILE));

    const std::string separator(";");
    const auto& write_to_file = [&](const PerfResults res, std::ofstream& file, bool only_one_op = false) {
        const auto gemm_k_size = res.k_dim_size_ > 0 ? std::to_string(res.k_dim_size_) : "N/A";
        ck::index_t rank_fixed_split_k, rank_best_occupancy_split_k, total_num;
        std::tie(rank_fixed_split_k, total_num) = res.get_ranking(res.fixed_split_k_op_name_, res.fixed_split_k_value_);
        std::tie(rank_best_occupancy_split_k, std::ignore) = res.get_ranking(res.best_occupancy_split_k_op_name_, res.best_occupancy_split_k_value_, res.best_occupancy_split_k_oversubscription_);

        file << res.fixed_split_k_op_name_ << separator
             << res.fixed_split_k_avg_time_ << separator
             << res.fixed_split_k_value_ << separator
             << rank_fixed_split_k << separator;
        if (!only_one_op) 
        {
            file << res.best_occupancy_split_k_op_name_ << separator;
        }
        file << res.best_occupancy_split_k_avg_time_ << separator
             << res.best_occupancy_split_k_value_ << separator
             << res.best_occupancy_split_k_oversubscription_ << separator
             << rank_best_occupancy_split_k << separator
             << gemm_k_size << separator
             << total_num;
    };

    if(!results_file.empty())
    {
        std::ofstream file(results_file, std::ios::out | std::ios::app);
        if(file.is_open())
        {
            // First the global results
            write_to_file(perf_results_global, file);
            file << separator; 

            // Then the local results - one set for each op
            const auto size = perf_results_list.size();
            for (size_t i = 0; i < size; ++i)
            {
                write_to_file(perf_results_list[i], file, true);
                if (i < size - 2) file << separator; 
            }
            file << std::endl;
            file.close();
        }
        else
        {
            std::cerr << "Failed to open results file: " << results_file << std::endl;
        }
    }
}

std::vector<std::string> get_disabled_ops()
{
    const auto& disabled_ops = ck::EnvGetString(CK_ENV(CK_PROFILER_DISABLED_OPS));
    std::vector<std::string> result;  
    std::stringstream ss(disabled_ops);  
    std::string item;  
  
    while (std::getline(ss, item, ';')) {  
        result.push_back(item);  
    }  

    std::cout << "Disabled " << result.size() << " ops: " << std::endl;
    for (const auto& op : result) {
        std::cout << "\t" << op << std::endl;
    }

    return result;  
}

bool is_operator_disabled(const std::string& op_name, const std::string& disabled_op)
{
    // Extract the base operator name (everything before the first "<")
    size_t template_pos = op_name.find('<');
    std::string base_op_name;
    
    if (template_pos != std::string::npos)
    {
        // If template parameters exist, extract only the base name
        base_op_name = op_name.substr(0, template_pos);
    }
    else
    {
        // No template parameters, use the whole name
        base_op_name = op_name;
    }
    
    return base_op_name == disabled_op;
}

template <ck::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename ComputeTypeA = InDataType,
          typename ComputeTypeB = ComputeTypeA>
bool profile_grouped_conv_bwd_weight_impl(int do_verification,
                                          int init_method,
                                          bool do_log,
                                          bool time_kernel,
                                          const ck::utils::conv::ConvParam& conv_param,
                                          const std::string& split_k)
{
    using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
    using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
    using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

    const auto in_element_op  = InElementOp{};
    const auto wei_element_op = WeiElementOp{};
    const auto out_element_op = OutElementOp{};

    const auto in_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);

    const auto wei_g_k_c_xs_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);

    const auto out_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    Tensor<InDataType> input(in_g_n_c_wis_desc);
    Tensor<WeiDataType> weight_host_result(wei_g_k_c_xs_desc);
    Tensor<WeiDataType> weight_device_result(wei_g_k_c_xs_desc);
    Tensor<OutDataType> output(out_g_n_k_wos_desc);

    std::cout << "input: " << input.mDesc << std::endl;
    std::cout << "weight: " << weight_host_result.mDesc << std::endl;
    std::cout << "output: " << output.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        input.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        output.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});
        break;
    default:
        input.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
        output.GenerateTensorValue(GeneratorTensor_3<OutDataType>{-0.5, 0.5});
    }

    DeviceMem in_device_buf(sizeof(InDataType) * input.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) *
                             weight_device_result.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutDataType) * output.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(input.mData.data());
    out_device_buf.ToDevice(output.mData.data());

    float max_accumulated_value = 0;
    if(do_verification)
    {
        std::cout << "Running reference implementation for verification..." << std::endl;
        auto ref_conv     = ck::tensor_operation::host::ReferenceConvBwdWeight<NDimSpatial,
                                                                           InDataType,
                                                                           WeiDataType,
                                                                           OutDataType,
                                                                           InElementOp,
                                                                           WeiElementOp,
                                                                           OutElementOp>{};
        auto ref_invoker  = ref_conv.MakeInvoker();
        auto ref_argument = ref_conv.MakeArgument(input,
                                                  weight_host_result,
                                                  output,
                                                  conv_param.conv_filter_strides_,
                                                  conv_param.conv_filter_dilations_,
                                                  conv_param.input_left_pads_,
                                                  conv_param.input_right_pads_,
                                                  in_element_op,
                                                  wei_element_op,
                                                  out_element_op,
                                                  {},
                                                  {},
                                                  {});

        ref_invoker.Run(ref_argument);
        max_accumulated_value =
            *std::max_element(weight_host_result.mData.begin(), weight_host_result.mData.end());
    }

    using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvBwdWeight<NDimSpatial,
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
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    // profile device Conv instances
    bool all_pass = true;

    std::array<ck::index_t, NDimSpatial + 3> input_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> filter_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> output_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> input_strides{};
    std::array<ck::index_t, NDimSpatial + 3> weights_strides{};
    std::array<ck::index_t, NDimSpatial + 3> output_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input_left_pads{};
    std::array<ck::index_t, NDimSpatial> input_right_pads{};

    auto range_copy = [](const auto& from, auto to) { std::copy(begin(from), end(from), to); };

    range_copy(in_g_n_c_wis_desc.GetLengths(), begin(input_lengths));
    range_copy(in_g_n_c_wis_desc.GetStrides(), begin(input_strides));
    range_copy(wei_g_k_c_xs_desc.GetLengths(), begin(filter_lengths));
    range_copy(wei_g_k_c_xs_desc.GetStrides(), begin(weights_strides));
    range_copy(out_g_n_k_wos_desc.GetLengths(), begin(output_lengths));
    range_copy(out_g_n_k_wos_desc.GetStrides(), begin(output_strides));
    range_copy(conv_param.conv_filter_strides_, begin(conv_filter_strides));
    range_copy(conv_param.conv_filter_dilations_, begin(conv_filter_dilations));
    range_copy(conv_param.input_left_pads_, begin(input_left_pads));
    range_copy(conv_param.input_right_pads_, begin(input_right_pads));

    std::vector<ck::index_t> fixed_split_k_list = {1, 2, 4, 8, 16, 32, 64, 128};
    std::vector<ck::index_t> subs_factor_list = {0, 1, 2, 4, 8, 16, 32, 64};
    bool profile_all = true;
    if(split_k != "all")
    {
        const auto split_k_val = std::stoi(split_k);
        fixed_split_k_list = {split_k_val};
        subs_factor_list = {};
        profile_all = false;
    }

    std::vector<ck::tensor_operation::device::ParamsSplitK> split_k_list;
    for (size_t i=0; i < fixed_split_k_list.size(); ++i)
    {
        ck::tensor_operation::device::ParamsSplitK params_split_k_fixed;
        params_split_k_fixed.split_k_value_ = fixed_split_k_list[i];
        split_k_list.push_back(params_split_k_fixed);

        if (i < subs_factor_list.size())
        {
            ck::tensor_operation::device::ParamsSplitK params_split_k_best_occupancy;
            params_split_k_best_occupancy.split_k_mode_ = ck::tensor_operation::device::SplitKMode::BestOccupancyWithOversubscription;
            params_split_k_best_occupancy.oversubscription_ = subs_factor_list[i];
            split_k_list.push_back(params_split_k_best_occupancy);
        }
    }

    PerfResults perf_results_global;
    std::vector<PerfResults> perf_results_list;
    const auto& disabled_ops = get_disabled_ops();

    for(auto& op_ptr : op_ptrs)
    {

        std::string op_name = op_ptr->GetTypeString();

        // Skip disabled ops
        if(std::any_of(disabled_ops.begin(), disabled_ops.end(), [&op_name](const std::string& disabled_op) {
            return is_operator_disabled(op_name, disabled_op);
        }))
        {
            std::cout << "Skipping disabled op: " << op_name << std::endl;
            continue;
        }

        PerfResults perf_results_local;
        bool supports_split_k_optimization = false;
        bool is_supported = false;

        for(std::size_t split_k_id = 0; split_k_id < split_k_list.size(); split_k_id++)
        {
            auto argument_ptr = op_ptr->MakeArgumentPointer(
                static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
                static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                input_lengths,
                input_strides,
                filter_lengths,
                weights_strides,
                output_lengths,
                output_strides,
                conv_filter_strides,
                conv_filter_dilations,
                input_left_pads,
                input_right_pads,
                in_element_op,
                wei_element_op,
                out_element_op,
                split_k_list[split_k_id]);

            auto split_k_arg_value = split_k_list[split_k_id].split_k_value_;
            auto* split_k_arg = dynamic_cast<ck::tensor_operation::device::ArgumentSplitK*>(argument_ptr.get());
            if (split_k_arg)
            {
                split_k_arg_value = split_k_arg->k_batch();
                const auto k_dim_size = split_k_arg->k_dim_size();
                if (k_dim_size > 0)
                {
                    perf_results_local.set_k_dim_size(k_dim_size);
                    perf_results_global.set_k_dim_size(k_dim_size);
                }
                supports_split_k_optimization = true;
            }

            // Skip the best occupancy values if the op does not support split-k optimization
            if (split_k_list[split_k_id].split_k_mode_ == 
                ck::tensor_operation::device::SplitKMode::BestOccupancyWithOversubscription && !supports_split_k_optimization)
            {
                continue;
            }

            const std::size_t workspace_sz = op_ptr->GetWorkSpaceSize(argument_ptr.get());
            DeviceMem workspace_dev(workspace_sz);
            op_ptr->SetWorkSpacePointer(argument_ptr.get(), workspace_dev.GetDeviceBuffer());

            if(op_ptr->IsSupportedArgument(argument_ptr.get()))
            {
                is_supported = true;

                auto invoker_ptr = op_ptr->MakeInvokerPointer();

                float avg_time =
                    invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

                std::size_t flop      = conv_param.GetFlops();
                std::size_t num_btype = conv_param.GetByte<InDataType, WeiDataType, OutDataType>();

                float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
                float gb_per_sec = num_btype / 1.E6 / avg_time;

                std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << tflops
                          << " TFlops, " << gb_per_sec << " GB/s, " << op_name << ", SplitK "
                          << PerfResults::split_k_str(split_k_list[split_k_id], split_k_arg_value) << std::endl;
                
                if (split_k_list[split_k_id].split_k_mode_ ==
                    ck::tensor_operation::device::SplitKMode::BestOccupancyWithOversubscription)
                {
                    const auto oversubscription = split_k_list[split_k_id].oversubscription_;
                    
                    perf_results_global.update_best_occupancy_split_k(
                            op_name,
                            avg_time,                                                         
                            split_k_arg_value,
                            oversubscription);

                    perf_results_local.update_best_occupancy_split_k(
                            op_name,
                            avg_time,                                                       
                            split_k_arg_value,
                            oversubscription);   
                }
                else 
                {
                    perf_results_global.update_fixed_split_k(op_name,
                                                            avg_time,                                                                
                                                            split_k_arg_value);

                    perf_results_local.update_fixed_split_k(op_name,
                                                                avg_time,                                                               
                                                                split_k_arg_value);
                }
                

                if(do_verification)
                {
                    wei_device_buf.FromDevice(weight_device_result.mData.data());

                    using ComputeType =
                        std::conditional_t<sizeof(ComputeTypeA) < sizeof(ComputeTypeB),
                                           ComputeTypeA,
                                           ComputeTypeB>;
                    using AccDataType =
                        std::conditional_t<std::is_same_v<ComputeType, int8_t>, int32_t, float>;
                    const index_t num_accums         = output.GetElementSize() / conv_param.K_;
                    const index_t num_accums_split_k = split_k_arg_value;
                    // Calculate thresholds
                    auto rtol =
                        ck::utils::get_relative_threshold<ComputeType, WeiDataType, AccDataType>(
                            num_accums / num_accums_split_k);
                    auto atol =
                        ck::utils::get_absolute_threshold<ComputeType, WeiDataType, AccDataType>(
                            max_accumulated_value / num_accums_split_k,
                            num_accums / num_accums_split_k);
                    // Calculate error due to split_k accumulation
                    auto rtol_split_k =
                        ck::utils::get_relative_threshold<WeiDataType, WeiDataType, WeiDataType>(
                            num_accums_split_k);
                    auto atol_split_k =
                        ck::utils::get_absolute_threshold<WeiDataType, WeiDataType, WeiDataType>(
                            max_accumulated_value, num_accums_split_k);
                    // Use higher threshold
                    rtol = std::max(rtol, rtol_split_k);
                    atol = std::max(atol, atol_split_k);
                    // Use default atol for splitK == 1
                    bool pass = ck::utils::check_err(weight_device_result,
                                                     weight_host_result,
                                                     "Error: Incorrect results!",
                                                     rtol,
                                                     atol);
                    std::cout << "Relative error threshold: " << rtol
                              << " Absolute error threshold: " << atol << std::endl;

                    if(!pass)
                    {
                        std::cout << "Fail info: " << op_ptr->GetTypeString() << std::endl;
                    }

                    all_pass &= pass;

                    if(do_log)
                    {
                        LogRangeAsType<float>(std::cout << "output : ", output.mData, ",")
                            << std::endl;
                        LogRangeAsType<float>(
                            std::cout << "weight (device): ", weight_device_result.mData, ",")
                            << std::endl;
                        LogRangeAsType<float>(
                            std::cout << "weight (host): ", weight_host_result.mData, ",")
                            << std::endl;
                        LogRangeAsType<float>(std::cout << "input: ", input.mData, ",")
                            << std::endl;
                    }
                }
            }
            else
            {
                std::cout << op_ptr->GetTypeString() << " does not support this problem"
                          << std::endl;
            }
        }

        if (supports_split_k_optimization && is_supported)
        {
            perf_results_list.push_back(perf_results_local);
        }
    }

    if (perf_results_list.size() > 0)
    {
        std::cerr << perf_results_global.print_fixed_split_k() << std::endl;

        if (profile_all)
        {
            std::cerr << perf_results_global.print_best_occupancy_split_k() << std::endl;
            write_perf_results_to_file(perf_results_global, perf_results_list);
        }
    }
    else 
    {
        std::cerr << "No supported/enabled ops found for this problem." << std::endl;
    }
    
    return all_pass;
}

} // namespace profiler
} // namespace ck
