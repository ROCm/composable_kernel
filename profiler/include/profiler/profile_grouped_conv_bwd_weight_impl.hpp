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
#include "ck/tensor_operation/gpu/device/device_conv_fwd.hpp"
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
    void update_best_op(std::string& op_name, float avg_time, float tflops, float gb_per_sec, ck::index_t split_k, ck::index_t split_k_arg)
    {
        if(tflops > best_tflops_)
        {
            best_op_name_    = op_name;
            best_avg_time_   = avg_time;
            best_tflops_     = tflops;
            best_gb_per_sec_ = gb_per_sec;
            best_split_k_    = split_k;
            best_split_k_arg_ = split_k_arg;
        }
        const auto split_k_value = split_k > 0 ? split_k : split_k_arg;
        ranking_.emplace_back(op_name, split_k_value, tflops);
        std::sort(ranking_.begin(), ranking_.end(),
                  [](const auto& a, const auto& b) { return std::get<2>(a) > std::get<2>(b); });
    };

    void update_opt_split_k(std::string& op_name, float avg_time, float tflops, float gb_per_sec, ck::index_t split_k_arg)
    {
        if(tflops > opt_split_k_tflops_)
        {
            opt_split_k_best_op_name_    = op_name;
            opt_split_k_avg_time_        = avg_time;
            opt_split_k_tflops_          = tflops;
            opt_split_k_gb_per_sec_      = gb_per_sec;
            opt_split_k_best_arg_        = split_k_arg;
        }
    };

    std::tuple<size_t, size_t> get_ranking(const std::string& op_name, ck::index_t split_k) const
    {
        auto it = std::find_if(ranking_.begin(), ranking_.end(),
                               [&](const auto& entry) {
                                   return std::get<0>(entry) == op_name && std::get<1>(entry) == split_k;
                               });
        if(it != ranking_.end())
        {
            const auto ranking = std::distance(ranking_.begin(), it) + 1;
            return std::make_tuple(ranking, ranking_.size());
        }
        return std::make_tuple(ranking_.size()+1, ranking_.size());
    };

    static std::string split_k_str(ck::index_t split_k_value, ck::index_t split_k_arg_value)
    {
        return split_k_value > 0 ? std::to_string(split_k_value) : std::to_string(split_k_arg_value) + " (optimized)";
    };

    std::string print_best_op() const
    {
        std::stringstream ss;
        ss << "\nname: " << best_op_name_ << "\navg_time: " << best_avg_time_
            << "\ntflops: " << best_tflops_ << "\nGB/s: " << best_gb_per_sec_ << ", SplitK "
            << split_k_str(best_split_k_, best_split_k_arg_);
        return ss.str();
    }

    std::string print_best_split_k() const
    {
        std::stringstream ss;
        ss << "\nname: " << opt_split_k_best_op_name_ << "\navg_time: " << opt_split_k_avg_time_
            << "\ntflops: " << opt_split_k_tflops_
            << "\nGB/s: " << opt_split_k_gb_per_sec_
            << ", SplitK " << split_k_str(-1, opt_split_k_best_arg_);
        return ss.str();
    }

    std::string best_op_name_;
    float best_avg_time_      = 0;
    float best_tflops_        = 0;
    float best_gb_per_sec_    = 0;
    ck::index_t best_split_k_ = 1;
    ck::index_t best_split_k_arg_ = 1;
    std::string opt_split_k_best_op_name_;
    float opt_split_k_avg_time_      = 0;
    float opt_split_k_tflops_        = 0;
    float opt_split_k_gb_per_sec_    = 0;
    ck::index_t opt_split_k_best_arg_ = 1;
    std::vector<std::tuple<std::string, ck::index_t, float>> ranking_;
};

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

void write_perf_results_to_file(const PerfResults& perf_results_global, 
                                const std::vector<PerfResults>& perf_results_list)
{
    const auto& results_file = ck::EnvGetString(CK_ENV(CK_PROFILER_OUTPUT_FILE));
    if(!results_file.empty())
    {
        std::ofstream file(results_file, std::ios::out | std::ios::app);
        if(file.is_open())
        {
            file << "Best configuration parameters:"
                 << perf_results_global.print_best_op() << std::endl;

            if (!perf_results_list.empty())
            {
                file << "Optimized split-K results:"
                     << perf_results_global.print_best_split_k() << std::endl;
            }

            for (const auto& res : perf_results_list)
            {
                file << res.print_best_op() << std::endl;
            }
            file.close();
        }
        else
        {
            std::cerr << "Failed to open results file: " << results_file << std::endl;
        }
    }
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

    std::vector<ck::index_t> split_k_list = {/*Split-k parameter autodeduction*/-1, 1, 2, 4, 8, 16, 32, 64, 128};
    bool profile_all = true;
    if(split_k != "all")
    {
        const auto split_k_val = std::stoi(split_k);
        split_k_list = {split_k_val};
        profile_all = false;
    }

    std::vector<PerfResults> perf_results_list;
    PerfResults perf_results_global;

    const auto& disabled_ops = get_disabled_ops();

    for(auto& op_ptr : op_ptrs)
    {
        std::string op_name = op_ptr->GetTypeString();

        // Skip disabled ops
        if(std::any_of(disabled_ops.begin(), disabled_ops.end(), [&op_name](const std::string& disabled_op) {
            return op_name.find(disabled_op) == 0;
        }))
        {
            std::cout << "Skipping disabled op: " << op_name << std::endl;
            continue;
        }

        PerfResults perf_results_local;
        bool supports_split_k_optimization = false;

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

            auto split_k_arg_value = split_k_list[split_k_id];
            auto* split_k_arg = dynamic_cast<ck::tensor_operation::device::ArgumentSplitK*>(argument_ptr.get());
            if (split_k_arg)
            {
                split_k_arg_value = split_k_arg->k_batch();
                supports_split_k_optimization = true;
            }

            // Skip the -1 value if the op does not support split-k optimization
            if (split_k_list[split_k_id] == -1 && !supports_split_k_optimization)
            {
                continue;
            }

            const std::size_t workspace_sz = op_ptr->GetWorkSpaceSize(argument_ptr.get());
            DeviceMem workspace_dev(workspace_sz);
            op_ptr->SetWorkSpacePointer(argument_ptr.get(), workspace_dev.GetDeviceBuffer());

            if(op_ptr->IsSupportedArgument(argument_ptr.get()))
            {
                // using atomic add, so need to reset input
                wei_device_buf.SetZero();

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

                perf_results_global.update_best_op(op_name,
                                                    avg_time,
                                                    tflops,
                                                    gb_per_sec,
                                                    split_k_list[split_k_id],
                                                    split_k_arg_value);
                
                if (supports_split_k_optimization)
                {
                    perf_results_local.update_best_op(op_name,
                                                        avg_time,
                                                        tflops,
                                                        gb_per_sec,
                                                        split_k_list[split_k_id],
                                                        split_k_arg_value);

                    if ( split_k_list[split_k_id] == -1)
                    {
                        perf_results_global.update_opt_split_k(op_name,
                                                            avg_time,
                                                            tflops,
                                                            gb_per_sec,
                                                            split_k_arg_value);

                        perf_results_local.update_opt_split_k(op_name,
                                                                avg_time,
                                                                tflops,
                                                                gb_per_sec,
                                                                split_k_arg_value);
                    }         
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
                    const index_t num_accums_split_k = split_k_list[split_k_id];
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
                    rtol      = std::max(rtol, rtol_split_k);
                    atol      = std::max(atol, atol_split_k);
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

        if (supports_split_k_optimization)
        {
            perf_results_list.push_back(perf_results_local);
        }
    }

    std::cout << "Best configuration parameters:"
              << perf_results_global.print_best_op() << std::endl;

    if (profile_all && perf_results_list.size() > 0)
    {
        std::cout << "Optimized split-K results:"
                  << perf_results_global.print_best_split_k() << std::endl;
        const auto& local_per_result = std::find_if(perf_results_list.begin(), perf_results_list.end(),
            [&](const PerfResults& res) { return res.opt_split_k_best_op_name_ == perf_results_global.opt_split_k_best_op_name_; });
        std::cout << "Global ranking: "
                  << std::get<0>(perf_results_global.get_ranking(perf_results_global.opt_split_k_best_op_name_, perf_results_global.opt_split_k_best_arg_))
                  << " / " << std::get<1>(perf_results_global.get_ranking(perf_results_global.opt_split_k_best_op_name_, perf_results_global.opt_split_k_best_arg_))
                  << std::endl;
        std::cout << "Local ranking: "
                  << std::get<0>(local_per_result->get_ranking(perf_results_global.opt_split_k_best_op_name_, perf_results_global.opt_split_k_best_arg_))
                  << " / " << std::get<1>(local_per_result->get_ranking(perf_results_global.opt_split_k_best_op_name_, perf_results_global.opt_split_k_best_arg_))
                  << std::endl;
    }

    return all_pass;
}

} // namespace profiler
} // namespace ck
