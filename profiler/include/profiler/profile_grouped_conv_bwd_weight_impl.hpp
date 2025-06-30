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

using SplitKStrategy = ck::tensor_operation::device::SplitKStrategy;
using ParamsSplitK = ck::tensor_operation::device::ParamsSplitK;

struct BestPerformance
{
    std::string op_name_{""};
    float avg_time_{std::numeric_limits<float>::max()};
    float tflops_{std::numeric_limits<float>::min()};
    ck::index_t split_k_value_{0};
};

struct PerfResults
{
    // Best performance for each split-K strategy
    std::map<SplitKStrategy, BestPerformance> best_performance_{};

    // GEMM problem parameters
    ck::index_t m_dim_size_{-1};
    ck::index_t n_dim_size_{-1};
    ck::index_t k_dim_size_{-1};
    float arithmetic_intensity_{0.0f};
    std::string data_type_{""};

    std::vector<std::tuple<std::string, ck::index_t, SplitKStrategy, float>> ranking_;

    void update_best_perf(const std::string& op_name, float avg_time, float tflops, ck::index_t split_k_arg, SplitKStrategy strategy)
    {
        const auto& current_best_perf = best_performance_[strategy];
        if(tflops > current_best_perf.tflops_)
        {
            best_performance_[strategy] = {op_name, avg_time, tflops, split_k_arg};
        }

        ranking_.emplace_back(op_name, split_k_arg, strategy, tflops);
        std::sort(ranking_.begin(), ranking_.end(),
                  [](const auto& a, const auto& b) { return std::get<3>(a) > std::get<3>(b); });
    };

    static std::string split_k_str(const ParamsSplitK split_k_params, ck::index_t split_k_arg)
    {
        switch (split_k_params.strategy_)
        {
        case SplitKStrategy::FixedSplitK:
            return std::to_string(split_k_params.fixed_value_);
            break;
        case SplitKStrategy::BestOccupancy:
            return std::to_string(split_k_arg) + " (best occupancy)";
            break;
        case SplitKStrategy::Optimized:
            return std::to_string(split_k_arg) + " (optimized)";
            break;
        default:
            break;
        }
    };
 
    std::string print_best_performance() const
    {
        const auto& to_string = [](const SplitKStrategy strategy) {
            switch (strategy)
            {
            case SplitKStrategy::FixedSplitK:
                return "Fixed Split-K";
            case SplitKStrategy::BestOccupancy:
                return "Best Occupancy";
            case SplitKStrategy::Optimized:
                return "Optimized";
            default:
                return "Unknown Strategy";
            }
        }; 

        std::stringstream ss;
        ss << "\nProblem Parameters"
           << "\n========================";
        ss << "\nm_dim_size: " << m_dim_size_
           << "\nn_dim_size: " << n_dim_size_
           << "\nk_dim_size: " << k_dim_size_
           << "\narithmetic_intensity: " << arithmetic_intensity_
           << "\ndata_type: " << data_type_;
        for (const auto& strategy : {SplitKStrategy::FixedSplitK, SplitKStrategy::BestOccupancy, SplitKStrategy::Optimized})
        {
            const auto& best_perf = best_performance_.find(strategy);
            if (best_perf != best_performance_.end())
            {
                ck::index_t rank, total_num;
                std::tie(rank, total_num) = get_ranking(best_perf->second.op_name_, best_perf->second.split_k_value_, strategy);
                
                ss << "\n\nBEST PERFORMANCE RESULTS (" << to_string(strategy) << ")"
                   << "\n========================";
                ss << "\nname: " << best_perf->second.op_name_ 
                    << "\navg_time: " << best_perf->second.avg_time_
                    << "\ntflops: " << best_perf->second.tflops_
                    << "\nSplitK: " << best_perf->second.split_k_value_
                    << "\nRanking: " << rank << " / " << total_num;
            }
        }

        return ss.str();
    }

    std::tuple<size_t, size_t> get_ranking(const std::string& op_name, ck::index_t split_k, SplitKStrategy strategy) const
    {
        auto it = std::find_if(ranking_.begin(), ranking_.end(),
                               [&](const auto& entry) {
                                   return std::get<0>(entry) == op_name && std::get<1>(entry) == split_k && 
                                          (std::get<2>(entry) == strategy);
                               });
        if(it != ranking_.end())
        {
            const auto ranking = std::distance(ranking_.begin(), it) + 1;
            return std::make_tuple(ranking, ranking_.size());
        }
        return std::make_tuple(ranking_.size()+1, ranking_.size());
    };

    void set_common_params(ck::index_t m_dim_size, ck::index_t n_dim_size, ck::index_t k_dim_size, float arithmetic_intensity, const std::string& data_type)
    {
        if (data_type_.empty())
        {
            data_type_ = data_type;
        }
        else if (data_type_ != data_type)
        {
            std::cerr << "Error: data_type cannot be set multiple times. Old value " << data_type_ << ". New value " << data_type << std::endl;
            exit(EXIT_FAILURE);
        }

        if (m_dim_size <= 0 || n_dim_size <= 0 || k_dim_size <= 0)
        {
            std::cerr << "Error: m_dim_size, n_dim_size, and k_dim_size must be positive integers." << std::endl;
            exit(EXIT_FAILURE);
        }

        if (m_dim_size_ > 0 && m_dim_size != m_dim_size_)
        {
            std::cerr << "Error: m_dim_size cannot be set multiple times. Old value " << m_dim_size_ << ". New value " << m_dim_size << std::endl;
            exit(EXIT_FAILURE);
        }
        m_dim_size_ = m_dim_size;

        if (n_dim_size_ > 0 && n_dim_size != n_dim_size_)
        {
            std::cerr << "Error: n_dim_size cannot be set multiple times. Old value " << n_dim_size_ << ". New value " << n_dim_size << std::endl;
            exit(EXIT_FAILURE);
        }
        n_dim_size_ = n_dim_size;

        if (k_dim_size_ > 0 && k_dim_size != k_dim_size_)
        {
            std::cerr << "Error: k_dim_size cannot be set multiple times. Old value " << k_dim_size_ << ". New value " << k_dim_size << std::endl;
            exit(EXIT_FAILURE);
        }
        k_dim_size_ = k_dim_size;

        const float eps = std::numeric_limits<float>::epsilon();
        if (arithmetic_intensity_ > 0.0f && std::abs(arithmetic_intensity - arithmetic_intensity_) > eps)
        {
            std::cerr << "Error: arithmetic_intensity cannot be set multiple times. Old value " << arithmetic_intensity_ << ". New value " << arithmetic_intensity << std::endl;
            exit(EXIT_FAILURE);
        }
        arithmetic_intensity_ = arithmetic_intensity;

        if (!data_type_.empty() && data_type != data_type_)
        {
            std::cerr << "Error: data_type cannot be set multiple times. Old value " << data_type_ << ". New value " << data_type << std::endl;
            exit(EXIT_FAILURE);
        }
        data_type_ = data_type;
    }
};

void write_perf_results_to_file(const PerfResults& perf_results_global, 
                                const std::vector<PerfResults>& perf_results_list)
{
    const auto& results_file = ck::EnvGetString(CK_ENV(CK_PROFILER_OUTPUT_FILE));

    if (results_file.empty())
    {
        return;
    }

    const std::string separator(";");

    const auto& to_string = [](SplitKStrategy strategy) {
        switch (strategy)
        {
        case SplitKStrategy::FixedSplitK:
            return "SplitKStrategy::FixedSplitK";
        case SplitKStrategy::BestOccupancy:
            return "SplitKStrategy::BestOccupancy";
        case SplitKStrategy::Optimized:
            return "SplitKStrategy::Optimized";
        default:
            return "Unknown Strategy";
        }
    };

    const auto& write_to_file = [&](const PerfResults res, std::ofstream& file, bool only_one_op = false) {

        ck::index_t total_num = -1;
        bool write_op_name = true;
        for (const auto& strategy : {SplitKStrategy::FixedSplitK, SplitKStrategy::BestOccupancy, SplitKStrategy::Optimized})
        {
            const auto& best_perf = res.best_performance_.find(strategy);
            if (best_perf != res.best_performance_.end())
            {
                BestPerformance perf;
                std::tie(std::ignore, perf) = *best_perf;
                ck::index_t rank;
                std::tie(rank, total_num) = res.get_ranking(perf.op_name_, perf.split_k_value_, strategy);
                if (write_op_name)
                {
                    file << perf.op_name_ << separator;
                    if (only_one_op)
                    {
                        // If only one op is written, we do not need to write the op name again
                        write_op_name = false;
                    }
                }
                file << perf.avg_time_ << separator
                     << perf.tflops_ << separator
                     << perf.split_k_value_ << separator
                     << rank << separator
                     << to_string(strategy) << separator;
            }
        }
        file << total_num;
    };

    if(!results_file.empty())
    {
        std::ofstream file(results_file, std::ios::out | std::ios::app);
        if(file.is_open())
        {
            // Write the common props, GEMM shapes and the arithmetic intensity
            file << perf_results_global.m_dim_size_ << separator
                 << perf_results_global.n_dim_size_ << separator
                 << perf_results_global.k_dim_size_ << separator
                 << perf_results_global.arithmetic_intensity_ << separator
                 << perf_results_global.data_type_ << separator;

            // First the global results
            write_to_file(perf_results_global, file);
            file << separator; 

            // Then the local results - one set for each op
            const auto size = perf_results_list.size();
            for (size_t i = 0; i < size; ++i)
            {
                write_to_file(perf_results_list[i], file, true);
                if (i < size - 1) file << separator; 
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
    // using SplitKStrategy = ck::tensor_operation::device::SplitKStrategy;
    // using ParamsSplitK = ck::tensor_operation::device::ParamsSplitK;

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
    std::vector<SplitKStrategy> best_occupancy_list = {SplitKStrategy::BestOccupancy /*, SplitKStrategy::Optimized*/};
    bool profile_all = true;
    if(split_k != "all")
    {
        const auto split_k_val = std::stoi(split_k);
        fixed_split_k_list = {split_k_val};
        best_occupancy_list = {};
        profile_all = false;
    }

    std::vector<ParamsSplitK> split_k_list;
    for (size_t i=0; i < best_occupancy_list.size(); ++i)
    {
        ParamsSplitK params_split_k_best_occupancy;
        params_split_k_best_occupancy.strategy_ = best_occupancy_list[i];
        split_k_list.push_back(params_split_k_best_occupancy);
    }

    for (size_t i=0; i < fixed_split_k_list.size(); ++i)
    {
        ParamsSplitK params_split_k_fixed;
        params_split_k_fixed.fixed_value_ = fixed_split_k_list[i];
        split_k_list.push_back(params_split_k_fixed);
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

            auto split_k_arg_value = split_k_list[split_k_id].fixed_value_;
            auto* split_k_arg = dynamic_cast<ck::tensor_operation::device::ArgumentSplitK*>(argument_ptr.get());
            if (split_k_arg)
            {
                split_k_arg_value = split_k_arg->k_batch();
                const auto k_dim_size = split_k_arg->k_dim_size();
                const auto m_dim_size = split_k_arg->m_dim_size();
                const auto n_dim_size = split_k_arg->n_dim_size();
                const auto arithmetic_intensity = split_k_arg->arithmetic_intensity();
                const auto& data_type = split_k_arg->data_type();
                if (k_dim_size > 0)
                {
                    perf_results_local.set_common_params(m_dim_size, n_dim_size, k_dim_size, arithmetic_intensity, data_type);
                    perf_results_global.set_common_params(m_dim_size, n_dim_size, k_dim_size, arithmetic_intensity, data_type);
                }
                supports_split_k_optimization = true;
            }

            // Skip the best occupancy values if the op does not support split-k optimization
            if (split_k_list[split_k_id].strategy_ != SplitKStrategy::FixedSplitK && !supports_split_k_optimization)
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

                constexpr int n_warm_up = 25;
                constexpr int n_repeat = 100;
                StreamConfig config{nullptr, time_kernel};
                config.cold_niters_ = n_warm_up;
                config.nrepeat_ = n_repeat;

                float avg_time = invoker_ptr->Run(argument_ptr.get(), config);

                std::size_t flop      = conv_param.GetFlops();
                std::size_t num_btype = conv_param.GetByte<InDataType, WeiDataType, OutDataType>();

                float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
                float gb_per_sec = num_btype / 1.E6 / avg_time;

                std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << tflops
                          << " TFlops, " << gb_per_sec << " GB/s, " << op_name << ", SplitK "
                          << PerfResults::split_k_str(split_k_list[split_k_id], split_k_arg_value) << std::endl;
                
                perf_results_local.update_best_perf(op_name,
                                                    avg_time,
                                                    tflops,
                                                    split_k_arg_value,
                                                    split_k_list[split_k_id].strategy_);
                perf_results_global.update_best_perf(op_name,
                                                     avg_time,
                                                     tflops,
                                                     split_k_arg_value,
                                                     split_k_list[split_k_id].strategy_);

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
        std::cerr << perf_results_global.print_best_performance() << std::endl;

        if (profile_all)
        {
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
