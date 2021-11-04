#pragma once
#include "device_conv_xdl_instance.hpp"

namespace ck {
namespace profiler {
namespace device_conv_xdl_instance {

} // namespace device_conv_xdl_instance

template <typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout>
void profile_conv(int do_verification,
                  int init_method,
                  bool do_log,
                  int nrepeat,
                  int N,
                  int K,
                  int C,
                  int Y,
                  int X,
                  int Hi,
                  int Wi,
                  int conv_stride_h,
                  int conv_stride_w,
                  int conv_dilation_h,
                  int conv_dilation_w,
                  int in_left_pad_h,
                  int in_left_pad_w,
                  int in_right_pad_h,
                  int in_right_pad_w)
{
    const int YEff = (Y - 1) * conv_dilation_h + 1;
    const int XEff = (X - 1) * conv_dilation_w + 1;

    const int Ho = (Hi + in_left_pad_h + in_right_pad_h - YEff) / conv_stride_h + 1;
    const int Wo = (Wi + in_left_pad_w + in_right_pad_w - XEff) / conv_stride_w + 1;

    auto f_host_tensor_descriptor =
        [](std::size_t n, std::size_t c, std::size_t h, std::size_t w, auto layout) {
            if constexpr(is_same<decltype(layout), tensor_layout::convolution::NCHW>::value)
            {
                return HostTensorDescriptor(std::vector<std::size_t>({n, c, h, w}),
                                            std::vector<std::size_t>({c * h * w, h * w, w, 1}));
            }
            else if constexpr(is_same<decltype(layout), tensor_layout::convolution::NHWC>::value)
            {
                return HostTensorDescriptor(std::vector<std::size_t>({n, c, h, w}),
                                            std::vector<std::size_t>({h * w * c, w * c, c, 1}));
            }
        };

    Tensor<InDataType> in_n_c_hi_wi(f_host_tensor_descriptor(N, C, Hi, Wi, InLayout{});
    Tensor<WeiDataType> wei_k_c_y_x({
        f_host_tensor_descriptor(K, C, Y, X, WeiLayout{});
    Tensor<OutDataType> out_n_k_ho_wo_host_result(f_host_tensor_descriptor(N, K, H, W, OutLayout{});
    Tensor<OutDataType> out_n_k_ho_wo_device_result(f_host_tensor_descriptor(N, K, H, W, OutLayout{});

    std::cout << "in_n_c_hi_wi: " << in_n_c_hi_wi.mDesc << std::endl;
    std::cout << "wei_k_c_y_x: " << wei_k_c_y_x.mDesc << std::endl;
    std::cout << "out_n_k_ho_wo: " << out_n_k_ho_wo_host_result.mDesc << std::endl;

    switch(init_method)
    {
        case 0: break;
        case 1:
            in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_2{-5, 5});
            wei_k_c_y_x.GenerateTensorValue(GeneratorTensor_2{-5, 5});
            break;
        default:
            in_c_hi_wi.GenerateTensorValue(GeneratorTensor_3<float>{0.0, 1.0});
            wei_k_c_y_x.GenerateTensorValue(GeneratorTensor_3<float>{-0.5, 0.5});
    }

    if(do_verification)
    {
            host_conv_nchw_kcyx_nkhw(in_n_c_hi_wi, wei_k_c_y_x, out_n_k_ho_wo_host_result);
    }

    DeviceMem in_device_buf(sizeof(InDataType) * in_n_c_hi_wi.mDesc.GetElementSpace());
    DeviceMem wei_nevice_buf(sizeof(WeiDataType) * wei_k_c_y_x.mDesc.GetElementSpace());
    DeviceMem out_device_buf(sizeof(OutDataType) *
                             out_n_k_ho_wo_device_result.mDesc.GetElementSpace());

    in_device_buf.ToDevice(in_n_c_hi_wi.mData.data());
    wei_device_buf.ToDevice(wei_k_c_y_x.mData.data());
    out_device_buf.ToDevice(out_n_k_ho_wo.mData.data());

    // add device Conv instances
    std::vector<DeviceConvFwdPtr> conv_ptrs;

    device_conv_xdl_instance::add_device_conv_xdl_instance<InDataType,
                                                           WeiDataType,
                                                           OutDataType,
                                                           InLayout,
                                                           WeiLayout,
                                                           OutLayout>(conv_ptrs);

    if(conv_ptrs.size() <= 0)
    {
            throw std::runtime_error("wrong! no device GEMM instance found");
    }

    // profile device Conv instances
    for(auto& conv_ptr : conv_ptrs)
    {
            auto argument_ptr = conv_ptr->MakeArgumentPointer(
                static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
                static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                N,
                K,
                C,
                Y,
                X,
                Hi,
                Wi,
                conv_stride_h,
                conv_stride_w,
                conv_dilation_h,
                conv_dilation_w,
                in_left_pad_h,
                in_left_pad_w,
                in_right_pad_h,
                in_right_pad_w);

            auto invoker_ptr = conv_ptr->MakeInvokerPointer();

            if(conv_ptr->IsSupportedArgument(argument_ptr.get()))
            {
                float ave_time = invoker_ptr->Run(argument_ptr.get(), nrepeat);

                std::size_t flop = std::size_t(2) * N * K * Ho * Wo * C * Y * X;

                std::size_t num_btype = sizeof(InDataType) * (N * C * Hi * Wi) +
                                        sizeof(WeiDataType) * (K * C * Y * X) +
                                        sizeof(OutDataType) * (N * K * Ho * Wo);

                float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

                float gb_per_sec = num_btype / 1.E6 / ave_time;

                std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                          << " GB/s" << std::endl;
            }
            else
            {
                std::cout << "this device conv instance does not support this conv problem"
                          << std::endl;
            }

            if(do_verification)
            {
                out_device_buf.FromDevice(out_n_k_ho_wo_device_result.mData.data());

                check_error(out_n_k_ho_wo_host_result, out_n_k_ho_wo_device_result);

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "in : ", in_n_c_hi_wi.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "wei: ", wei_k_c_y_x.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(
                        std::cout << "out_host  : ", out_n_k_ho_wo_host_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(
                        std::cout << "out_device: ", out_n_k_ho_wo_device_result.mData, ",")
                        << std::endl;
                }
            }
    }
}

} // namespace profiler
} // namespace profiler
