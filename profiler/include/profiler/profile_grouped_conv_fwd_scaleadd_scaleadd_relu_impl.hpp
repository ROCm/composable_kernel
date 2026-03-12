// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_scaleadd_scaleadd_relu.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/host_tensor.hpp"

namespace ck {
namespace profiler {

template <typename DataType>
inline constexpr double get_rtol_scaleadd()
{
    if constexpr(std::is_same_v<DataType, float>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, double>)
    {
        return 1e-6;
    }
    else if constexpr(std::is_same_v<DataType, ck::half_t>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, ck::bhalf_t>)
    {
        return 5e-2;
    }
    else if constexpr(std::is_same_v<DataType, int32_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, int8_t>)
    {
        return 1e-1;
    }
    else
    {
        return 1e-3;
    }
}

template <typename DataType>
inline constexpr double get_atol_scaleadd()
{
    if constexpr(std::is_same_v<DataType, float>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, double>)
    {
        return 1e-6;
    }
    else if constexpr(std::is_same_v<DataType, ck::half_t>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, ck::bhalf_t>)
    {
        return 5e-2;
    }
    else if constexpr(std::is_same_v<DataType, int32_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, int8_t>)
    {
        return 1e-1;
    }
    else
    {
        return 1e-3;
    }
}

template <ck::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename OutElementOp,
          typename AComputeType = InDataType,
          typename BComputeType = AComputeType>
bool profile_grouped_conv_fwd_scaleadd_scaleadd_relu_impl(
    int do_verification,
    int init_method,
    bool do_log,
    bool time_kernel,
    const ck::utils::conv::ConvParam& conv_param)
{
    auto pass = true;

    using CShuffleDataType = float;

    using BiasDataType = std::conditional_t<std::is_same_v<InDataType, int8_t>, float, InDataType>;

    using PassThrough  = ck::tensor_operation::element_wise::PassThrough;
    using InElementOp  = PassThrough;
    using WeiElementOp = PassThrough;

    const auto in_element_op  = InElementOp{};
    const auto wei_element_op = WeiElementOp{};

    const auto out_element_op = OutElementOp{1.0f, 2.0f};

    const auto in_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);

    const auto wei_g_k_c_xs_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);

    const auto out_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    const index_t G = conv_param.G_;
    const index_t K = conv_param.K_;

    auto bias1_ndhwgk_desc = out_g_n_k_wos_desc;
    auto bias2_g_k_desc    = HostTensorDescriptor({G, K});

    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_strides{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_strides{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial + 3> bias1_ndhwgk_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> bias1_ndhwgk_strides{};
    std::array<ck::index_t, NDimSpatial + 3> bias2_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> bias2_g_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input_left_pads{};
    std::array<ck::index_t, NDimSpatial> input_right_pads{};

    auto copy = [](const auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

    copy(in_g_n_c_wis_desc.GetLengths(), a_g_n_c_wis_lengths);
    copy(in_g_n_c_wis_desc.GetStrides(), a_g_n_c_wis_strides);
    copy(wei_g_k_c_xs_desc.GetLengths(), b_g_k_c_xs_lengths);
    copy(wei_g_k_c_xs_desc.GetStrides(), b_g_k_c_xs_strides);
    copy(out_g_n_k_wos_desc.GetLengths(), e_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_desc.GetStrides(), e_g_n_k_wos_strides);
    copy(out_g_n_k_wos_desc.GetLengths(), bias1_ndhwgk_lengths);
    copy(out_g_n_k_wos_desc.GetStrides(), bias1_ndhwgk_strides);
    copy(out_g_n_k_wos_desc.GetLengths(), bias2_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_desc.GetStrides(), bias2_g_n_k_wos_strides);
    copy(conv_param.conv_filter_strides_, conv_filter_strides);
    copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_param.input_left_pads_, input_left_pads);
    copy(conv_param.input_right_pads_, input_right_pads);

    constexpr ck::index_t spatial_offset = 3;
    bias2_g_n_k_wos_strides[1]           = 0;
    for(int i = 0; i < NDimSpatial; i++)
    {
        bias2_g_n_k_wos_strides[i + spatial_offset] = 0;
    }

    Tensor<InDataType> input(in_g_n_c_wis_desc);
    Tensor<WeiDataType> weight(wei_g_k_c_xs_desc);
    Tensor<CShuffleDataType> c(out_g_n_k_wos_desc);
    Tensor<OutDataType> host_output(out_g_n_k_wos_desc);
    Tensor<OutDataType> device_output(out_g_n_k_wos_desc);
    Tensor<BiasDataType> bias1(bias1_ndhwgk_desc);
    Tensor<BiasDataType> bias2(bias2_g_k_desc);

    std::cout << "input: " << input.mDesc << std::endl;
    std::cout << "weight: " << weight.mDesc << std::endl;
    std::cout << "output: " << host_output.mDesc << std::endl;
    std::cout << "bias1 (NDHWGK): " << bias1.mDesc << std::endl;
    std::cout << "bias2 (G_K): " << bias2.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        input.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        weight.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-1, 1});
        bias1.GenerateTensorValue(GeneratorTensor_2<BiasDataType>{-1, 1});
        bias2.GenerateTensorValue(GeneratorTensor_2<BiasDataType>{-1, 1});
        break;
    default:
        input.GenerateTensorValue(GeneratorTensor_3<InDataType>{-5.0, 5.0});
        weight.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-1.0, 1.0});
        bias1.GenerateTensorValue(GeneratorTensor_3<BiasDataType>{-0.5, 0.5});
        bias2.GenerateTensorValue(GeneratorTensor_3<BiasDataType>{-0.5, 0.5});
    }

    DeviceMem in_device_buf(sizeof(InDataType) * input.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * weight.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutDataType) * device_output.mDesc.GetElementSpaceSize());
    DeviceMem bias1_device_buf(sizeof(BiasDataType) * bias1.mDesc.GetElementSpaceSize());
    DeviceMem bias2_device_buf(sizeof(BiasDataType) * bias2.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(input.mData.data());
    wei_device_buf.ToDevice(weight.mData.data());
    bias1_device_buf.ToDevice(bias1.mData.data());
    bias2_device_buf.ToDevice(bias2.mData.data());

    // run reference op
    if(do_verification)
    {
        std::cout << "\nVerifying algorithm against reference convolution..." << std::endl;
        std::cout << "\tUsing (rel_tol,abs_tol) = (" << std::setprecision(7)
                  << get_rtol_scaleadd<OutDataType>() << ", " << get_atol_scaleadd<OutDataType>()
                  << ")" << std::endl;

        auto ref_conv = ck::tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                     InDataType,
                                                                     WeiDataType,
                                                                     CShuffleDataType,
                                                                     InElementOp,
                                                                     WeiElementOp,
                                                                     PassThrough>{};

        auto ref_invoker  = ref_conv.MakeInvoker();
        auto ref_argument = ref_conv.MakeArgument(input,
                                                  weight,
                                                  c,
                                                  conv_param.conv_filter_strides_,
                                                  conv_param.conv_filter_dilations_,
                                                  conv_param.input_left_pads_,
                                                  conv_param.input_right_pads_,
                                                  in_element_op,
                                                  wei_element_op,
                                                  PassThrough{});

        c.SetZero();
        ref_invoker.Run(ref_argument);

        host_output.ForEach([&](auto&, auto idx) {
            const auto g_idx = idx[0];
            const auto k_idx = idx[2];

            const auto conv_shuffle = ck::type_convert<CShuffleDataType>(c(idx));

            if constexpr(std::is_same_v<OutDataType, int8_t>)
            {
                const auto conv_val = ck::type_convert<OutDataType>(conv_shuffle);

                const auto bias1_val = bias1(idx);
                const auto bias2_val = bias2(g_idx, k_idx);

                OutDataType out_val{};
                out_element_op(out_val, conv_val, bias1_val, bias2_val);

                host_output(idx) = ck::type_convert<OutDataType>(out_val);
            }
            else
            {
                const auto conv_val = conv_shuffle;

                const auto bias1_val = ck::type_convert<CShuffleDataType>(bias1(idx));
                const auto bias2_val = ck::type_convert<CShuffleDataType>(bias2(g_idx, k_idx));

                CShuffleDataType out_val{};
                out_element_op(out_val, conv_val, bias1_val, bias2_val);

                host_output(idx) = ck::type_convert<OutDataType>(out_val);
            }
        });
    }

    std::string best_op_name;
    float best_avg_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    auto run_impl = [&](auto& op_ptr, auto& argument_ptr) {
        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            out_device_buf.SetZero();

            std::string op_name = op_ptr->GetTypeString();

            auto invoker_ptr = op_ptr->MakeInvokerPointer();

            float avg_time =
                invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

            std::size_t flop      = conv_param.GetFlops();
            std::size_t num_btype = conv_param.GetByte<InDataType, WeiDataType, OutDataType>();

            float tflops = static_cast<float>(flop) / 1.E9 / avg_time;

            float gb_per_sec = num_btype / 1.E6 / avg_time;

            std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << tflops << " TFlops, "
                      << gb_per_sec << " GB/s, " << op_name << std::endl;

            if(tflops > best_tflops)
            {
                best_op_name    = op_name;
                best_tflops     = tflops;
                best_avg_time   = avg_time;
                best_gb_per_sec = gb_per_sec;
            }

            if(do_verification)
            {
                out_device_buf.FromDevice(device_output.mData.data());

                pass = pass & ck::utils::check_err(device_output,
                                                   host_output,
                                                   "Error: Device and Host results do not match!",
                                                   get_rtol_scaleadd<OutDataType>(),
                                                   get_atol_scaleadd<OutDataType>());

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "input : ", input.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "weight: ", weight.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "bias1: ", bias1.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "bias2: ", bias2.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "host_output  : ", host_output.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "device_output: ", device_output.mData, ",")
                        << std::endl;
                }
            }
        }
        else
        {
            std::cout << op_ptr->GetTypeString() << " does not support this problem" << std::endl;
        }
    };

    using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<
        NDimSpatial,
        InLayout,
        WeiLayout,
        ck::Tuple<OutLayout, ck::tensor_layout::convolution::G_K>,
        OutLayout,
        InDataType,
        WeiDataType,
        ck::Tuple<BiasDataType, BiasDataType>,
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

    for(auto& op_ptr : op_ptrs)
    {
        auto argument_ptr = op_ptr->MakeArgumentPointer(
            in_device_buf.GetDeviceBuffer(),
            wei_device_buf.GetDeviceBuffer(),
            {bias1_device_buf.GetDeviceBuffer(), bias2_device_buf.GetDeviceBuffer()},
            out_device_buf.GetDeviceBuffer(),
            a_g_n_c_wis_lengths,
            a_g_n_c_wis_strides,
            b_g_k_c_xs_lengths,
            b_g_k_c_xs_strides,
            {bias1_ndhwgk_lengths, bias2_g_n_k_wos_lengths},
            {bias1_ndhwgk_strides, bias2_g_n_k_wos_strides},
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

    std::cout << "Best configuration parameters:" << "\nname: " << best_op_name
              << "\navg_time: " << best_avg_time << "\ntflops: " << best_tflops
              << "\nGB/s: " << best_gb_per_sec << std::endl;
    return pass;
}

} // namespace profiler
} // namespace ck
