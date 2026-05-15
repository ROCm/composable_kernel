// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <typeinfo>

#include "ck/ck.hpp"
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
#include "ck/library/reference_tensor_operation/gpu/naive_conv_bwd_weight_gpu.hpp"
#include "ck/library/utility/gpu_verification.hpp"

namespace ck {
namespace profiler {

namespace bwd_weight {
template <ck::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InElementOp,
          typename WeiElementOp,
          typename OutElementOp,
          typename ComputeTypeA,
          typename ComputeTypeB>
void print_instances()
{
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

    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    for(const auto& op_ptr : op_ptrs)
    {
#ifdef CK_EXPERIMENTAL_BUILDER
        const auto& instance_str = op_ptr->GetInstanceString();
        if(!instance_str.empty())
        {
            std::cout << instance_str << std::endl;
        }
        else
        {
            std::cout << op_ptr->GetTypeString() << std::endl;
        }
#else
        std::cout << op_ptr->GetTypeString() << std::endl;
#endif
    }
}
} // namespace bwd_weight

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
                                          const std::string& split_k,
                                          index_t instance_index = -1,
                                          bool list_instances    = false)
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

    std::cout << "input: " << in_g_n_c_wis_desc << std::endl;
    std::cout << "weight: " << wei_g_k_c_xs_desc << std::endl;
    std::cout << "output: " << out_g_n_k_wos_desc << std::endl;

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

    // Create host tensors
    Tensor<InDataType> input(in_g_n_c_wis_desc);
    Tensor<WeiDataType> weight_host_result(wei_g_k_c_xs_desc);
    Tensor<WeiDataType> weight_device_result(wei_g_k_c_xs_desc);
    Tensor<OutDataType> output(out_g_n_k_wos_desc);

    // Get element space sizes for allocation
    const auto input_element_space_size  = in_g_n_c_wis_desc.GetElementSpaceSize();
    const auto weight_element_space_size = wei_g_k_c_xs_desc.GetElementSpaceSize();
    const auto output_element_space_size = out_g_n_k_wos_desc.GetElementSpaceSize();

    // Allocate GPU buffers
    DeviceMem in_device_buf(sizeof(InDataType) * input_element_space_size);
    DeviceMem wei_device_buf(sizeof(WeiDataType) * weight_element_space_size);
    DeviceMem out_device_buf(sizeof(OutDataType) * output_element_space_size);

    // Don't create reference if we're only listing instances
    if(list_instances)
        do_verification = 0;

    // Initialize tensors based on do_verification:
    // - do_verification=2: GPU-side initialization
    // - do_verification=0,1: CPU-side initialization
    if(do_verification == 2)
    {
        // GPU-side initialization for GPU verification workflow
        switch(init_method)
        {
        case 0:
            // Zero initialization
            in_device_buf.SetZero();
            out_device_buf.SetZero();
            break;
        case 1:
            // Discrete integer values in range [-5, 5]
            in_device_buf.FillUniformRandInteger<InDataType>(-5, 5);
            out_device_buf.FillUniformRandInteger<OutDataType>(-5, 5);
            break;
        default:
            // Continuous float values
            in_device_buf.FillUniformRandFp<InDataType>(0.0f, 1.0f);
            out_device_buf.FillUniformRandFp<OutDataType>(-0.5f, 0.5f);
        }
    }
    else
    {
        // CPU-side initialization for do_verification=0,1
        switch(init_method)
        {
        case 0: break; // Tensors are already zero-initialized by default
        case 1:
            input.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
            output.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});
            break;
        default:
            input.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
            output.GenerateTensorValue(GeneratorTensor_3<OutDataType>{-0.5, 0.5});
        }

        // Copy initialized host data to device
        in_device_buf.ToDevice(input.mData.data());
        out_device_buf.ToDevice(output.mData.data());
    }

    // Allocate GPU reference buffer (used only if do_verification == 2)
    DeviceMem gpu_ref_wei_buf(
        do_verification == 2 ? sizeof(WeiDataType) * weight_host_result.mDesc.GetElementSpaceSize()
                             : 0);

    float max_accumulated_value = 0;
    if(do_verification)
    {
        if(do_verification == 1)
        {
            // CPU reference
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
        else if(do_verification == 2)
        {
            // Use GPU reference with GPU verification
            std::cout << "Using GPU reference with GPU verification" << std::endl;

            // Call GPU reference with ConvParam directly
            ck::ref::naive_conv_bwd_weight<InLayout,
                                           WeiLayout,
                                           OutLayout,
                                           InDataType,
                                           WeiDataType,
                                           OutDataType,
                                           InElementOp,
                                           WeiElementOp,
                                           OutElementOp>(
                static_cast<const InDataType*>(in_device_buf.GetDeviceBuffer()),
                static_cast<WeiDataType*>(gpu_ref_wei_buf.GetDeviceBuffer()),
                static_cast<const OutDataType*>(out_device_buf.GetDeviceBuffer()),
                conv_param,
                in_element_op,
                wei_element_op,
                out_element_op);
        }
    }

    std::string best_op_name;
    float best_avg_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;
    std::string best_split_k("1");
    index_t best_instance_index = 0;
    index_t valid_instances     = 0;

    // profile device Conv instances
    bool all_pass           = true;
    bool dummy_run_executed = false;

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

    std::vector<ck::index_t> split_k_list = {/*auto deduce value*/ -1, 1, 2, 4, 8, 16, 32, 64, 128};

    if(split_k != "all")
    {
        try
        {
            ck::index_t split_k_value = std::stoi(split_k);
            split_k_list              = {split_k_value};
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            exit(EXIT_FAILURE);
        }
    }

    if(list_instances)
    {
        std::cout << "\nValid instances for this problem:" << std::endl;
    }

    index_t num_kernel = 0;
    for(size_t i = 0; i < op_ptrs.size(); i++)
    {
        if((instance_index != -1) && (instance_index != static_cast<int>(i)))
        {
            // skip test if instance_index is specified
            continue;
        }
        auto& op_ptr = op_ptrs[i];
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

            auto split_k_value     = split_k_list[split_k_id];
            auto split_k_param_str = std::to_string(split_k_value);

            // If split_k was determined by the device implementation, get the resulting value.
            if(split_k_value < 0)
            {
                auto* split_k_arg =
                    dynamic_cast<ck::tensor_operation::device::ArgumentSplitK*>(argument_ptr.get());
                if(split_k_arg)
                {
                    split_k_value     = split_k_arg->k_batch_;
                    split_k_param_str = std::to_string(split_k_value) + " (best occupancy)";
                }
                else
                {
                    // We may have an implementation whose argument is not derived from
                    // ArgumentSplitK, which means we can not determine the splitK value. Warn.
                    printf("Warning: Unable to determine split_k value for this instance!\n");
                }
            }

            // Not all device implementation actually do anything with the passed split_k value but
            // it needs to be positive to determine error tolerances.
            if(split_k_value < 0)
            {
                split_k_value = 1;
            }

            const std::size_t workspace_sz = op_ptr->GetWorkSpaceSize(argument_ptr.get());
            DeviceMem workspace_dev(workspace_sz);
            op_ptr->SetWorkSpacePointer(argument_ptr.get(), workspace_dev.GetDeviceBuffer());

            if(op_ptr->IsSupportedArgument(argument_ptr.get()))
            {
                num_kernel++;

                // List instances mode - just print and continue
                if(list_instances)
                {
                    std::cout << "[" << (num_kernel - 1) << "] " << op_ptr->GetTypeString()
                              << " (SplitK=" << split_k_param_str << ")" << std::endl;
                    continue;
                }

                // Skip if a specific instance was requested and this isn't it
                const bool running_specific_instance = (instance_index != -1);
                const bool current_is_target         = (num_kernel - 1 == instance_index);
                if(running_specific_instance && !current_is_target)
                {
                    continue;
                }
                valid_instances++;

                std::string op_name = op_ptr->GetTypeString();

                auto invoker_ptr = op_ptr->MakeInvokerPointer();

                if(time_kernel && !dummy_run_executed)
                {
                    // Run first instance as dummy to get proper time from the first instance
                    invoker_ptr->Run(argument_ptr.get(),
                                     StreamConfig{nullptr,
                                                  time_kernel,
                                                  0 /*log_level*/,
                                                  5 /*cold_iters*/,
                                                  50 /*nrepeat_*/,
                                                  time_kernel /*flush_cache*/});
                    dummy_run_executed = true;
                }
                float avg_time = invoker_ptr->Run(argument_ptr.get(),
                                                  StreamConfig{nullptr,
                                                               time_kernel,
                                                               0 /*log_level*/,
                                                               5 /*cold_iters*/,
                                                               50 /*nrepeat_*/,
                                                               time_kernel /*flush_cache*/});

                std::size_t flop      = conv_param.GetFlops();
                std::size_t num_btype = conv_param.GetByte<InDataType, WeiDataType, OutDataType>();

                float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
                float gb_per_sec = num_btype / 1.E6 / avg_time;

                std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << tflops
                          << " TFlops, " << gb_per_sec << " GB/s, " << op_name << ", SplitK "
                          << split_k_param_str << std::endl;

                if(tflops > best_tflops)
                {
                    best_op_name        = op_name;
                    best_tflops         = tflops;
                    best_avg_time       = avg_time;
                    best_gb_per_sec     = gb_per_sec;
                    best_split_k        = split_k_param_str;
                    best_instance_index = num_kernel - 1;
                }

                // Synchronize before verification to ensure kernel has completed
                if(do_verification > 0 && !time_kernel)
                {
                    hip_check_error(hipStreamSynchronize(nullptr));
                }

                if(do_verification == 2)
                {
                    // GPU verification path
                    using ComputeType =
                        std::conditional_t<sizeof(ComputeTypeA) < sizeof(ComputeTypeB),
                                           ComputeTypeA,
                                           ComputeTypeB>;
                    using AccDataType =
                        std::conditional_t<std::is_same_v<ComputeType, int8_t>, int32_t, float>;

                    const index_t num_accums =
                        output.GetElementSize() / (conv_param.K_ * conv_param.G_);
                    const index_t num_accums_split_k = split_k_value;
                    // Get maximum accumulated value from reference
                    const std::size_t tensor_size =
                        weight_device_result.mDesc.GetElementSpaceSize();
                    max_accumulated_value =
                        gpu_reduce_max<WeiDataType>(gpu_ref_wei_buf.GetDeviceBuffer(), tensor_size);
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

                    // Perform GPU verification
                    auto gpu_result =
                        ck::profiler::gpu_verify<WeiDataType>(wei_device_buf.GetDeviceBuffer(),
                                                              gpu_ref_wei_buf.GetDeviceBuffer(),
                                                              rtol,
                                                              atol,
                                                              tensor_size);

                    if(!gpu_result)
                    {
                        // GPU verification failed - print detailed error summary
                        gpu_result.print_error_summary();
                        all_pass = false;

                        std::cout << "Fail info: splitK: " << split_k_value << " "
                                  << op_ptr->GetTypeString() << std::endl;

                        if(do_log)
                        {
                            // Copy buffers to host for logging
                            wei_device_buf.FromDevice(weight_device_result.mData.data());
                            gpu_ref_wei_buf.FromDevice(weight_host_result.mData.data());

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
                else if(do_verification == 1)
                {
                    // CPU verification path (original behavior)
                    wei_device_buf.FromDevice(weight_device_result.mData.data());

                    using ComputeType =
                        std::conditional_t<sizeof(ComputeTypeA) < sizeof(ComputeTypeB),
                                           ComputeTypeA,
                                           ComputeTypeB>;
                    using AccDataType =
                        std::conditional_t<std::is_same_v<ComputeType, int8_t>, int32_t, float>;
                    const index_t num_accums =
                        output.GetElementSize() / (conv_param.K_ * conv_param.G_);
                    const index_t num_accums_split_k = split_k_value;
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

                    if(!pass)
                    {
                        std::cout << "Relative error threshold: " << rtol
                                  << " Absolute error threshold: " << atol << std::endl;
                        std::cout << "Fail info: splitK: " << split_k_value << " "
                                  << op_ptr->GetTypeString() << std::endl;
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
            else if(list_instances || instance_index == -1)
            {
                std::cout << op_ptr->GetTypeString() << " does not support this problem"
                          << std::endl;
            }
        }
    }

    if(list_instances)
    {
        std::cout << "\nTotal: " << num_kernel << " valid instances" << std::endl;
        return true;
    }

    printf("\033[36mvalids: %ld\033[0m\n", static_cast<long>(valid_instances));

    if(instance_index != -1 && valid_instances == 0)
    {
        std::cerr << "Error: instance_index " << instance_index
                  << " exceeds the number of valid instances (" << num_kernel << ")" << std::endl;
        return false;
    }

    std::cout << "Best configuration parameters:" << "\nname: " << best_op_name << " (instance "
              << best_instance_index << ")" << "\navg_time: " << best_avg_time
              << "\ntflops: " << best_tflops << "\nGB/s: " << best_gb_per_sec << ", SplitK "
              << best_split_k << std::endl;

    return all_pass;
}

} // namespace profiler
} // namespace ck
