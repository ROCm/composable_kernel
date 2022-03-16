#pragma once

#include "host_interface.hpp"


struct DeviceMem
{
    float* ptr_mem=nullptr;
    int size;
    DeviceMem(int _size): size(_size){}
    float* GetDeviceBuffer()
    {
        return ptr_mem;
    }
};

namespace ck {
namespace app {

void profile_conv_fwd_impl(int do_verification,
                           int init_method,
                           bool do_log,
                           int nrepeat,
                           ck::index_t N,
                           ck::index_t K,
                           ck::index_t C,
                           std::vector<ck::index_t> input_spatial_lengths,
                           std::vector<ck::index_t> filter_spatial_lengths,
                           std::vector<ck::index_t> output_spatial_lengths,
                           std::vector<ck::index_t> conv_filter_strides,
                           std::vector<ck::index_t> conv_filter_dilations,
                           std::vector<ck::index_t> input_left_pads,
                           std::vector<ck::index_t> input_right_pads)
{
    const ck::index_t Y = filter_spatial_lengths[0];
    const ck::index_t X = filter_spatial_lengths[1];

    const ck::index_t Hi = input_spatial_lengths[0];
    const ck::index_t Wi = input_spatial_lengths[1];

    const ck::index_t Ho = output_spatial_lengths[0];
    const ck::index_t Wo = output_spatial_lengths[1];

    const auto in_sz = 1000;
    const auto wei_sz = 1000;
    const auto out_sz = 1000;

    using WeiDataType = float;
    using InDataType = float;
    using OutDataType = float;

    DeviceMem in_device_buf(sizeof(InDataType) * in_sz);
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei_sz);
    DeviceMem out_device_buf(sizeof(OutDataType) * out_sz);
    // data is already on device!


    // add device Conv instances
    std::vector<DeviceConvFwdPtr_t> conv_ptrs;

    add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f32_instances_t(conv_ptrs);
    if(conv_ptrs.empty())
    {
        throw std::runtime_error("wrong! no device Conv instance found");
    }

    std::string best_conv_name;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    // profile device Conv instances
    for(auto& conv_ptr : conv_ptrs)
    {
        auto argument_ptr = conv_ptr.MakeArgumentPointer(
            static_cast<void*>(in_device_buf.GetDeviceBuffer()),
            static_cast<void*>(wei_device_buf.GetDeviceBuffer()),
            static_cast<void*>(out_device_buf.GetDeviceBuffer()),
            N,
            K,
            C,
            input_spatial_lengths,
            filter_spatial_lengths,
            output_spatial_lengths,
            conv_filter_strides,
            conv_filter_dilations,
            input_left_pads,
            input_right_pads);

        auto invoker_ptr = conv_ptr.MakeInvokerPointer();

        if(conv_ptr.IsSupportedArgument(argument_ptr.get()))
        {
            std::string conv_name = conv_ptr.GetTypeString();

            float ave_time = invoker_ptr->Run(argument_ptr.get(), nrepeat);

            std::size_t flop = std::size_t(2) * N * K * Ho * Wo * C * Y * X;

            std::size_t num_btype = sizeof(InDataType) * (N * C * Hi * Wi) +
                                    sizeof(WeiDataType) * (K * C * Y * X) +
                                    sizeof(OutDataType) * (N * K * Ho * Wo);

            float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

            float gb_per_sec = num_btype / 1.E6 / ave_time;

            std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                      << " GB/s, " << conv_name << std::endl;

            if(tflops > best_tflops)
            {
                best_conv_name  = conv_name;
                best_tflops     = tflops;
                best_ave_time   = ave_time;
                best_gb_per_sec = gb_per_sec;
            }
        }
    }

    std::cout << "Best Perf: " << best_ave_time << " ms, " << best_tflops << " TFlops, "
              << best_gb_per_sec << " GB/s, " << best_conv_name << std::endl;
}

} // namespace profiler
} // namespace ck
