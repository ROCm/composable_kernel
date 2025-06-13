// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.


template <ck::index_t NDimSpatial>
using HostConvBwdInstance = ck::tensor_operation::host::ReferenceConvBwdData<NDimSpatial,
                                                                         InUserDataType,
                                                                         WeiUserDataType,
                                                                         CShuffleDataType,
                                                                         InElementOp,
                                                                         WeiElementOp,
                                                                         PassThrough>;

template <ck::index_t NDimSpatial>
bool run_grouped_conv_bwd(const ExecutionConfig& config,
                          const ck::utils::conv::ConvParam& conv_param)
{
    static_assert(1 <= NDimSpatial && NDimSpatial <= 3, "Unsupported NDimSpatial");

    const auto in_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<
            InputLayout<NDimSpatial>>(conv_param);

    const auto wei_g_k_c_xs_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<
            WeightLayout<NDimSpatial>>(conv_param);

    const auto out_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<
            OutputLayout<NDimSpatial>>(conv_param);

    //const auto in_g_n_c_wis_desc  = make_input_descriptor(conv_param);
    //const auto wei_g_k_c_xs_desc  = make_weight_descriptor(conv_param);
    //const auto out_g_n_k_wos_desc = make_output_descriptor(conv_param);

    Tensor<InUserDataType> out(out_g_n_k_wos_desc);
    Tensor<WeiUserDataType> wei(wei_g_k_c_xs_desc);
    Tensor<OutUserDataType> in_host(in_g_n_c_wis_desc);
    Tensor<OutUserDataType> in_device(in_g_n_c_wis_desc);

    std::cout << "in: " << in_device.mDesc << std::endl;
    std::cout << "wei: " << wei.mDesc << std::endl;
    std::cout << "out: " << out.mDesc << std::endl;

    switch(config.init_method)
    {
    case 0: break;
    case 1:
        out.GenerateTensorValue(GeneratorTensor_2<InUserDataType>{-5, 5});
        wei.GenerateTensorValue(GeneratorTensor_2<WeiUserDataType>{-5, 5});
        break;
    #if 0
    case 2:
        out.GenerateTensorValue(GeneratorTensor_2<InUserDataType>{1, 2});
        wei.GenerateTensorValue(GeneratorTensor_2<WeiUserDataType>{4, 10});
        break;
    #endif
    default:
        out.GenerateTensorValue(GeneratorTensor_3<InUserDataType>{0.0, 1.0});
        wei.GenerateTensorValue(GeneratorTensor_3<WeiUserDataType>{-0.5, 0.5});
    }

    DeviceMem out_device_buf(sizeof(InKernelDataType) * out.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiUserDataType) * wei.mDesc.GetElementSpaceSize());
    DeviceMem in_device_buf(sizeof(OutKernelDataType) * in_device.mDesc.GetElementSpaceSize());

    out_device_buf.ToDevice(out.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());

    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_strides{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_strides{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input_left_pads{};
    std::array<ck::index_t, NDimSpatial> input_right_pads{};

    auto copy = [](auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

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

    
    if(config.do_verification)
    {
        auto ref_conv = ck::tensor_operation::host::ReferenceConvBwdData<NDimSpatial,
                                                                         InUserDataType,
                                                                         WeiUserDataType,
                                                                         OutUserDataType,
                                                                         PassThrough,
                                                                         WeiElementOp,
                                                                         OutElementOp>();

        auto ref_invoker  = ref_conv.MakeInvoker();
        auto ref_argument = ref_conv.MakeArgument(in_host,
                                                  wei,
                                                  out,
                                                  conv_param.conv_filter_strides_,
                                                  conv_param.conv_filter_dilations_,
                                                  conv_param.input_left_pads_,
                                                  conv_param.input_right_pads_,
                                                  InElementOp{},
                                                  WeiElementOp{},
                                                  OutElementOp{});

        ref_invoker.Run(ref_argument);
    }
#if defined(ENABLE_CONV_FACTORY)
    float best_tflops = 0;
    float best_gb_per_sec = 0;
    float best_avg_time = 0;
    std::string best_kernel = "";

    ck::index_t instance_idx = 0;
    bool found_kernel= false;
    ck::static_for<0, std::tuple_size_v<DeviceConvBwdFactory>, 1>{}([&](auto i) -> void {
        // do Conv
        const auto device_conv_bwd_instance = std::get<i>(DeviceConvBwdFactory{});
        using DeviceConvBwdInstance = ck::remove_cvref_t<decltype(device_conv_bwd_instance)>;
        auto conv     = DeviceConvBwdInstance{};
        auto invoker  = conv.MakeInvoker();
        auto argument = conv.MakeArgument(in_device_buf.GetDeviceBuffer(),
                                          wei_device_buf.GetDeviceBuffer(),
                                          std::array<const void*, 0>{},
                                          out_device_buf.GetDeviceBuffer(),
                                          a_g_n_c_wis_lengths,
                                          a_g_n_c_wis_strides,
                                          b_g_k_c_xs_lengths,
                                          b_g_k_c_xs_strides,
                                          std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                          std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                          e_g_n_k_wos_lengths,
                                          e_g_n_k_wos_strides,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          InElementOp{},
                                          WeiElementOp{},
                                          OutElementOp{});

        if(conv.IsSupportedArgument(argument))
        {
            found_kernel = true;
            float avg_time = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});
            if(config.time_kernel)
            {
                std::size_t flop = conv_param.GetFlops();
                std::size_t num_btype =
                    conv_param.GetByte<InUserDataType, WeiUserDataType, OutUserDataType>();

                float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
                float gb_per_sec = num_btype / 1.E6 / avg_time;
                std::cout << "Perf: " << avg_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                          << " GB/s, " << conv.GetTypeString() << std::endl;

                if(tflops > best_tflops)
                {
                    best_tflops     = tflops;
                    best_gb_per_sec = gb_per_sec;
                    best_avg_time   = avg_time;
                    best_kernel     = conv.GetTypeString();
                    instance_idx    = i;
                }
            }
            if(config.do_verification)
            {
                in_device_buf.FromDevice(in_device.mData.data());
                ck::utils::check_err(
                    in_device.mData, in_host.mData, "Error: incorrect results!", 1e-3f, 1e-3f);
            }
        }
    });
    if (found_kernel == false)
    {
        std::cerr << "wrong! device_conv with the specified compilation parameters does "
                "not support this Conv problem"
            << std::endl;
        return true;
    }
    if(config.time_kernel)
    {
        std::cout << "Best Perf: " << best_avg_time << " ms, " << best_tflops << " TFlops, " << best_gb_per_sec
                  << " GB/s" << std::endl
                  << "Instance: " << instance_idx << " DeviceOp: " << best_kernel << std::endl;
    }

#else
    // do Conv
    auto conv     = DeviceConvBwdInstance<NDimSpatial>{};
    auto invoker  = conv.MakeInvoker();
    auto argument = conv.MakeArgument(in_device_buf.GetDeviceBuffer(),
                                      wei_device_buf.GetDeviceBuffer(),
                                      std::array<const void*, 0>{},
                                      out_device_buf.GetDeviceBuffer(),
                                      a_g_n_c_wis_lengths,
                                      a_g_n_c_wis_strides,
                                      b_g_k_c_xs_lengths,
                                      b_g_k_c_xs_strides,
                                      std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                      std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                      e_g_n_k_wos_lengths,
                                      e_g_n_k_wos_strides,
                                      conv_filter_strides,
                                      conv_filter_dilations,
                                      input_left_pads,
                                      input_right_pads,
                                      InElementOp{},
                                      WeiElementOp{},
                                      OutElementOp{});

    if(!conv.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_conv with the specified compilation parameters does "
            "not support this Conv problem");
    }

    float avg_time = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});

    std::size_t flop      = conv_param.GetFlops();
    std::size_t num_btype = conv_param.GetByte<InUserDataType, WeiUserDataType, OutUserDataType>();

    float tflops     = static_cast<float>(flop) / 1.E9 / avg_time;
    float gb_per_sec = num_btype / 1.E6 / avg_time;
    std::cout << "Perf: " << avg_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << conv.GetTypeString() << std::endl;
    
    if(config.do_verification)
    {
       in_device_buf.FromDevice(in_device.mData.data());
#ifdef BUILD_INT4_EXAMPLE
        const Tensor<OutUserDataType> out_device_converted(out_device);

        return ck::utils::check_err(
            out_device_converted.mData, out_host.mData, "Error: incorrect results!", 1e-5f, 1e-4f);
#else
        return ck::utils::check_err(
            in_device.mData, in_host.mData, "Error: incorrect results!", 1e-3f, 1e-3f);
#endif
    }
#endif

    return true;
}

bool run_grouped_conv_bwd_example(int argc, char* argv[])
{
    ExecutionConfig config;
    ck::utils::conv::ConvParam conv_param = DefaultConvParam;

    if(!parse_cmd_args(argc, argv, config, conv_param))
    {
        return false;
    }

    switch(conv_param.num_dim_spatial_)
    {
    //case 1: return run_grouped_conv_bwd<1>(config, conv_param);
    case 2: return run_grouped_conv_bwd<2>(config, conv_param);
    //case 3: return run_grouped_conv_bwd<3>(config, conv_param);
    }

    return false;
}
