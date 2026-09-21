// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <iostream>
#include <typeinfo>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_clamp.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_dynamic_op.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_convinvscale.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"
#include "ck/library/reference_tensor_operation/gpu/naive_conv_fwd_gpu.hpp"
#include "ck/library/utility/gpu_verification.hpp"

namespace ck {
namespace profiler {

namespace fwd {
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
    using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<NDimSpatial,
                                                                                   InLayout,
                                                                                   WeiLayout,
                                                                                   ck::Tuple<>,
                                                                                   OutLayout,
                                                                                   InDataType,
                                                                                   WeiDataType,
                                                                                   ck::Tuple<>,
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
} // namespace fwd

template <ck::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename AComputeType = InDataType,
          typename BComputeType = AComputeType,
          typename IndexType    = ck::index_t,
          typename OutElementOp = ck::tensor_operation::element_wise::PassThrough>
bool profile_grouped_conv_fwd_impl(int do_verification,
                                   int init_method,
                                   bool do_log,
                                   bool time_kernel,
                                   const ck::utils::conv::ConvParam& conv_param,
                                   const OutElementOp out_element_op = OutElementOp{},
                                   index_t instance_index            = -1,
                                   bool list_instances               = false)
{
    using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
    using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;

    const auto in_element_op  = InElementOp{};
    const auto wei_element_op = WeiElementOp{};

    const auto in_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);

    const auto wei_g_k_c_xs_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);

    const auto out_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    std::array<IndexType, NDimSpatial + 3> a_g_n_c_wis_lengths{};
    std::array<IndexType, NDimSpatial + 3> a_g_n_c_wis_strides{};
    std::array<IndexType, NDimSpatial + 3> b_g_k_c_xs_lengths{};
    std::array<IndexType, NDimSpatial + 3> b_g_k_c_xs_strides{};
    std::array<IndexType, NDimSpatial + 3> e_g_n_k_wos_lengths{};
    std::array<IndexType, NDimSpatial + 3> e_g_n_k_wos_strides{};
    std::array<IndexType, NDimSpatial> conv_filter_strides{};
    std::array<IndexType, NDimSpatial> conv_filter_dilations{};
    std::array<IndexType, NDimSpatial> input_left_pads{};
    std::array<IndexType, NDimSpatial> input_right_pads{};

    auto copy = [](const auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

    copy(in_g_n_c_wis_desc.GetLengths(), a_g_n_c_wis_lengths);
    copy(in_g_n_c_wis_desc.GetStrides(), a_g_n_c_wis_strides);
    copy(wei_g_k_c_xs_desc.GetLengths(), b_g_k_c_xs_lengths);
    copy(wei_g_k_c_xs_desc.GetStrides(), b_g_k_c_xs_strides);
    copy(out_g_n_k_wos_desc.GetLengths(), e_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_desc.GetStrides(), e_g_n_k_wos_strides);
    copy(conv_param.conv_filter_strides_, conv_filter_strides);
    copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_param.input_left_pads_, input_left_pads);
    copy(conv_param.input_right_pads_, input_right_pads);

    std::cout << "input: " << in_g_n_c_wis_desc << std::endl;
    std::cout << "weight: " << wei_g_k_c_xs_desc << std::endl;
    std::cout << "output: " << out_g_n_k_wos_desc << std::endl;

    using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<NDimSpatial,
                                                                                   InLayout,
                                                                                   WeiLayout,
                                                                                   ck::Tuple<>,
                                                                                   OutLayout,
                                                                                   InDataType,
                                                                                   WeiDataType,
                                                                                   ck::Tuple<>,
                                                                                   OutDataType,
                                                                                   InElementOp,
                                                                                   WeiElementOp,
                                                                                   OutElementOp,
                                                                                   AComputeType,
                                                                                   BComputeType>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "ckProfiler found " << op_ptrs.size() << " instances" << std::endl;

    // Create host tensors
    Tensor<InDataType> input({1});
    Tensor<WeiDataType> weight({1});
    Tensor<OutDataType> host_output({1});
    Tensor<OutDataType> device_output({1});
    if(init_method != 0 || do_verification != 0)
    {
        input         = Tensor<InDataType>(in_g_n_c_wis_desc);
        weight        = Tensor<WeiDataType>(wei_g_k_c_xs_desc);
        host_output   = Tensor<OutDataType>(out_g_n_k_wos_desc);
        device_output = Tensor<OutDataType>(out_g_n_k_wos_desc);
    }

    // Get element space sizes for allocation
    const auto input_size  = in_g_n_c_wis_desc.GetElementSpaceSize();
    const auto weight_size = wei_g_k_c_xs_desc.GetElementSpaceSize();
    const auto output_size = out_g_n_k_wos_desc.GetElementSpaceSize();

    // Allocate GPU memory
    DeviceMem in_device_buf(sizeof(InDataType) * input_size);
    DeviceMem wei_device_buf(sizeof(WeiDataType) * weight_size);
    DeviceMem out_device_buf(sizeof(OutDataType) * output_size);

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
            wei_device_buf.SetZero();
            break;
        case 1:
            // Discrete integer generation: {-5, -4, -3, ..., 3, 4}
            in_device_buf.FillUniformRandInteger<InDataType>(-5, 5);
            wei_device_buf.FillUniformRandInteger<WeiDataType>(-5, 5);
            break;
        default:
            // Continuous float generation
            in_device_buf.FillUniformRandFp<InDataType>(0.0f, 1.0f);
            wei_device_buf.FillUniformRandFp<WeiDataType>(-0.5f, 0.5f);
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
            weight.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
            break;
        default:
            input.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
            weight.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
        }

        if(init_method != 0)
        {
            // Copy initialized host data to device
            in_device_buf.ToDevice(input.mData.data());
            wei_device_buf.ToDevice(weight.mData.data());
        }
    }

    // Allocate GPU reference buffer (used only if do_verification == 2)
    DeviceMem gpu_ref_out_buf(
        do_verification == 2 ? sizeof(OutDataType) * device_output.mDesc.GetElementSpaceSize() : 0);

    // run reference op
    if(do_verification == 2)
    {
        // Use GPU reference with GPU verification
        std::cout << "Using GPU reference with GPU verification" << std::endl;

        // Call GPU reference with ConvParam directly
        ref::naive_conv_fwd<InLayout,
                            WeiLayout,
                            OutLayout,
                            InDataType,
                            WeiDataType,
                            OutDataType,
                            InElementOp,
                            WeiElementOp,
                            OutElementOp>(
            reinterpret_cast<const InDataType*>(in_device_buf.GetDeviceBuffer()),
            reinterpret_cast<const WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
            reinterpret_cast<OutDataType*>(gpu_ref_out_buf.GetDeviceBuffer()),
            conv_param,
            in_element_op,
            wei_element_op,
            out_element_op);
    }
    else if(do_verification == 1)
    {
        // Use CPU reference for verification (default)
        auto ref_conv = ck::tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                     InDataType,
                                                                     WeiDataType,
                                                                     OutDataType,
                                                                     InElementOp,
                                                                     WeiElementOp,
                                                                     OutElementOp>{};

        auto ref_invoker  = ref_conv.MakeInvoker();
        auto ref_argument = ref_conv.MakeArgument(input,
                                                  weight,
                                                  host_output,
                                                  conv_param.conv_filter_strides_,
                                                  conv_param.conv_filter_dilations_,
                                                  conv_param.input_left_pads_,
                                                  conv_param.input_right_pads_,
                                                  in_element_op,
                                                  wei_element_op,
                                                  out_element_op);

        // init host output to zero
        host_output.SetZero();

        ref_invoker.Run(ref_argument);
    }

    std::string best_op_name;
    float best_avg_time         = 0;
    float best_tflops           = 0;
    float best_gb_per_sec       = 0;
    index_t num_kernel          = 0;
    index_t valid_instances     = 0;
    index_t best_instance_index = 0;

    // profile device op instances
    bool pass               = true;
    bool dummy_run_executed = false;

    auto run_impl = [&](auto& op_ptr, auto& argument_ptr) {
        // workspace_sz will be equal to 0 for other layout than NGCHW
        const std::size_t workspace_sz = op_ptr->GetWorkSpaceSize(argument_ptr.get());
        DeviceMem workspace_dev(0);
        if(workspace_sz)
        {
            workspace_dev.Realloc(workspace_sz);
            op_ptr->SetWorkSpacePointer(argument_ptr.get(), workspace_dev.GetDeviceBuffer());
        }

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            num_kernel++;

            // List instances mode - just print and continue
            if(list_instances)
            {
                std::cout << "[" << (num_kernel - 1) << "] " << op_ptr->GetTypeString()
                          << std::endl;
                return;
            }

            // Skip if a specific instance was requested and this isn't it
            const bool running_specific_instance = (instance_index != -1);
            const bool current_is_target         = (num_kernel - 1 == instance_index);
            if(running_specific_instance && !current_is_target)
            {
                return;
            }

            std::string op_name = op_ptr->GetTypeString();
            valid_instances++;

            auto invoker_ptr = op_ptr->MakeInvokerPointer();

            // Run first instance twice to get proper time
            if(time_kernel && !dummy_run_executed)
            {
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

            float tflops = static_cast<float>(flop) / 1.E9 / avg_time;

            float gb_per_sec = num_btype / 1.E6 / avg_time;

            std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << tflops << " TFlops, "
                      << gb_per_sec << " GB/s, " << op_name << std::endl;

            if(tflops > best_tflops)
            {
                best_op_name        = op_name;
                best_tflops         = tflops;
                best_avg_time       = avg_time;
                best_gb_per_sec     = gb_per_sec;
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
                // Calculate number of accumulations (C * filter spatial dimensions)
                std::size_t filter_spatial_size = 1;
                for(auto len : conv_param.filter_spatial_lengths_)
                {
                    filter_spatial_size *= len;
                }
                const int num_accums = static_cast<int>(conv_param.C_ * filter_spatial_size);

                // Perform GPU verification (max value computed internally on GPU)
                const std::size_t tensor_size = device_output.mDesc.GetElementSpaceSize();
                auto gpu_result = ck::profiler::gpu_verify<OutDataType, AComputeType, OutDataType>(
                    out_device_buf.GetDeviceBuffer(),
                    gpu_ref_out_buf.GetDeviceBuffer(),
                    num_accums,
                    tensor_size);

                if(!gpu_result)
                {
                    // GPU verification failed - print detailed error summary
                    gpu_result.print_error_summary();
                    pass = false;

                    if(do_log)
                    {
                        // Copy buffers to host for logging
                        out_device_buf.FromDevice(device_output.mData.data());
                        gpu_ref_out_buf.FromDevice(host_output.mData.data());

                        LogRangeAsType<float>(std::cout << "input : ", input.mData, ",")
                            << std::endl;
                        LogRangeAsType<float>(std::cout << "weight: ", weight.mData, ",")
                            << std::endl;
                        LogRangeAsType<float>(
                            std::cout << "host_output  : ", host_output.mData, ",")
                            << std::endl;
                        LogRangeAsType<float>(
                            std::cout << "device_output: ", device_output.mData, ",")
                            << std::endl;
                    }
                }
            }
            else if(do_verification == 1)
            {
                // CPU verification path (original behavior)
                out_device_buf.FromDevice(device_output.mData.data());

                pass = pass & ck::utils::check_err(device_output, host_output);

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "input : ", input.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "weight: ", weight.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "host_output  : ", host_output.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "device_output: ", device_output.mData, ",")
                        << std::endl;
                }
            }
        }
        else if(list_instances || instance_index == -1)
        {
            std::cout << op_ptr->GetTypeString() << " does not support this problem" << std::endl;
        }
    };

    if(list_instances)
    {
        std::cout << "\nValid instances for this problem:" << std::endl;
    }

    // Run first instance twice to get proper time
    {
        auto argument_ptr = op_ptrs[0]->MakeArgumentPointer(in_device_buf.GetDeviceBuffer(),
                                                            wei_device_buf.GetDeviceBuffer(),
                                                            {},
                                                            out_device_buf.GetDeviceBuffer(),
                                                            a_g_n_c_wis_lengths,
                                                            a_g_n_c_wis_strides,
                                                            b_g_k_c_xs_lengths,
                                                            b_g_k_c_xs_strides,
                                                            {},
                                                            {},
                                                            e_g_n_k_wos_lengths,
                                                            e_g_n_k_wos_strides,
                                                            conv_filter_strides,
                                                            conv_filter_dilations,
                                                            input_left_pads,
                                                            input_right_pads,
                                                            in_element_op,
                                                            wei_element_op,
                                                            out_element_op);

        run_impl(op_ptrs[0], argument_ptr);
    }
    for(size_t i = 0; i < op_ptrs.size(); i++)
    {
        if((instance_index != -1) && (instance_index != static_cast<int>(i)))
        {
            // skip test if instance_index is specified
            continue;
        }
        auto& op_ptr      = op_ptrs[i];
        auto argument_ptr = op_ptr->MakeArgumentPointer(in_device_buf.GetDeviceBuffer(),
                                                        wei_device_buf.GetDeviceBuffer(),
                                                        {},
                                                        out_device_buf.GetDeviceBuffer(),
                                                        a_g_n_c_wis_lengths,
                                                        a_g_n_c_wis_strides,
                                                        b_g_k_c_xs_lengths,
                                                        b_g_k_c_xs_strides,
                                                        {},
                                                        {},
                                                        e_g_n_k_wos_lengths,
                                                        e_g_n_k_wos_strides,
                                                        conv_filter_strides,
                                                        conv_filter_dilations,
                                                        input_left_pads,
                                                        input_right_pads,
                                                        in_element_op,
                                                        wei_element_op,
                                                        out_element_op);

        run_impl(op_ptr, argument_ptr);
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
              << "\ntflops: " << best_tflops << "\nGB/s: " << best_gb_per_sec << std::endl;
    if(instance_index != -1)
    {
        std::cout << "grouped_conv_fwd_instance (" << instance_index << "/" << num_kernel
                  << "): Passed" << std::endl;
    }
    return pass;
}

} // namespace profiler
} // namespace ck
