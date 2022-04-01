#pragma once
#include "config.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "tensor_layout.hpp"
#include "device_tensor.hpp"
#include "device_conv_bwd_data.hpp"
#include "element_wise_operation.hpp"
#include "reference_conv_bwd_data.hpp"

using F16  = ck::half_t;
using F32  = float;
using BF16 = ck::bhalf_t;
using INT8 = int8_t;
namespace ck {
namespace tensor_operation {
namespace device {
namespace device_conv2d_bwd_data_instance {

using DeviceConvBwdDataNoOpPtr =
    DeviceConvBwdDataPtr<ck::tensor_operation::element_wise::PassThrough,
                         ck::tensor_operation::element_wise::PassThrough,
                         ck::tensor_operation::element_wise::PassThrough>;
void add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_f32_instances(
    std::vector<DeviceConvBwdDataNoOpPtr>&);
void add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_f16_instances(
    std::vector<DeviceConvBwdDataNoOpPtr>&);
void add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_bf16_instances(
    std::vector<DeviceConvBwdDataNoOpPtr>&);
void add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_int8_instances(
    std::vector<DeviceConvBwdDataNoOpPtr>&);
} // namespace device_conv2d_bwd_data_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

namespace ck {
namespace profiler {

template <int NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename AccDataType,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout>
void profile_conv_bwd_data_impl(int do_verification,
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

    auto f_host_tensor_descriptor =
        [](std::size_t N_, std::size_t C_, std::size_t H, std::size_t W, auto layout) {
            if constexpr(is_same<decltype(layout), ck::tensor_layout::convolution::NCHW>::value ||
                         is_same<decltype(layout), ck::tensor_layout::convolution::KCYX>::value ||
                         is_same<decltype(layout), ck::tensor_layout::convolution::NKHW>::value)
            {
                return HostTensorDescriptor(std::vector<std::size_t>({N_, C_, H, W}),
                                            std::vector<std::size_t>({C_ * H * W, H * W, W, 1}));
            }
            else if constexpr(is_same<decltype(layout), tensor_layout::convolution::NHWC>::value ||
                              is_same<decltype(layout), tensor_layout::convolution::KYXC>::value ||
                              is_same<decltype(layout), tensor_layout::convolution::NHWK>::value)
            {
                return HostTensorDescriptor(std::vector<std::size_t>({N_, C_, H, W}),
                                            std::vector<std::size_t>({C_ * H * W, 1, W * C_, C_}));
            }
        };

    Tensor<InDataType> in_n_c_hi_wi_host_result(f_host_tensor_descriptor(N, C, Hi, Wi, InLayout{}));
    Tensor<InDataType> in_n_c_hi_wi_device_result(
        f_host_tensor_descriptor(N, C, Hi, Wi, InLayout{}));
    Tensor<WeiDataType> wei_k_c_y_x(f_host_tensor_descriptor(K, C, Y, X, WeiLayout{}));
    Tensor<OutDataType> out_n_k_ho_wo(f_host_tensor_descriptor(N, K, Ho, Wo, OutLayout{}));

    std::cout << "in_n_c_hi_wi: " << in_n_c_hi_wi_host_result.mDesc << std::endl;
    std::cout << "wei_k_c_y_x: " << wei_k_c_y_x.mDesc << std::endl;
    std::cout << "out_n_k_ho_wo: " << out_n_k_ho_wo.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        out_n_k_ho_wo.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        wei_k_c_y_x.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        break;
    default:
        out_n_k_ho_wo.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
        wei_k_c_y_x.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
    }

    using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
    using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
    using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

    const auto in_element_op  = InElementOp{};
    const auto wei_element_op = WeiElementOp{};
    const auto out_element_op = OutElementOp{};

    if(do_verification)
    {
        using ReferenceConvBwdDataInstance =
            ck::tensor_operation::host::ReferenceConvBwdData<InDataType,
                                                             WeiDataType,
                                                             OutDataType,
                                                             AccDataType,
                                                             InElementOp,
                                                             WeiElementOp,
                                                             OutElementOp>;

        auto ref_conv     = ReferenceConvBwdDataInstance{};
        auto ref_invoker  = ref_conv.MakeInvoker();
        auto ref_argument = ref_conv.MakeArgument(in_n_c_hi_wi_host_result,
                                                  wei_k_c_y_x,
                                                  out_n_k_ho_wo,
                                                  conv_filter_strides,
                                                  conv_filter_dilations,
                                                  input_left_pads,
                                                  input_right_pads,
                                                  in_element_op,
                                                  wei_element_op,
                                                  out_element_op);

        ref_invoker.Run(ref_argument);
    }

    DeviceMem in_device_buf(sizeof(InDataType) *
                            in_n_c_hi_wi_device_result.mDesc.GetElementSpace());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei_k_c_y_x.mDesc.GetElementSpace());
    DeviceMem out_device_buf(sizeof(OutDataType) * out_n_k_ho_wo.mDesc.GetElementSpace());

    out_device_buf.ToDevice(out_n_k_ho_wo.mData.data());
    wei_device_buf.ToDevice(wei_k_c_y_x.mData.data());

    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    using DeviceConvBwdDataNoOpPtr =
        ck::tensor_operation::device::DeviceConvBwdDataPtr<PassThrough, PassThrough, PassThrough>;

    // add device Conv instances
    std::vector<DeviceConvBwdDataNoOpPtr> conv_ptrs;
    if constexpr(ck::is_same_v<ck::remove_cv_t<InDataType>, float> &&
                 ck::is_same_v<ck::remove_cv_t<WeiDataType>, float> &&
                 ck::is_same_v<ck::remove_cv_t<OutDataType>, float>)
    {
        ck::tensor_operation::device::device_conv2d_bwd_data_instance::
            add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_f32_instances(conv_ptrs);
    }
    else if constexpr(ck::is_same_v<ck::remove_cv_t<InDataType>, ck::half_t> &&
                      ck::is_same_v<ck::remove_cv_t<WeiDataType>, ck::half_t> &&
                      ck::is_same_v<ck::remove_cv_t<OutDataType>, ck::half_t>)
    {
        ck::tensor_operation::device::device_conv2d_bwd_data_instance::
            add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_f16_instances(conv_ptrs);
    }
    else if constexpr(ck::is_same_v<ck::remove_cv_t<InDataType>, ck::bhalf_t> &&
                      ck::is_same_v<ck::remove_cv_t<WeiDataType>, ck::bhalf_t> &&
                      ck::is_same_v<ck::remove_cv_t<OutDataType>, ck::bhalf_t>)
    {
        ck::tensor_operation::device::device_conv2d_bwd_data_instance::
            add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_bf16_instances(conv_ptrs);
    }
    else if constexpr(ck::is_same_v<ck::remove_cv_t<InDataType>, int8_t> &&
                      ck::is_same_v<ck::remove_cv_t<WeiDataType>, int8_t> &&
                      ck::is_same_v<ck::remove_cv_t<OutDataType>, int8_t>)
    {
        ck::tensor_operation::device::device_conv2d_bwd_data_instance::
            add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_int8_instances(conv_ptrs);
    }

    if(conv_ptrs.size() <= 0)
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
        auto argument_ptr = conv_ptr->MakeArgumentPointer(
            static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
            static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
            static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
            N,
            K,
            C,
            input_spatial_lengths,
            filter_spatial_lengths,
            output_spatial_lengths,
            conv_filter_strides,
            conv_filter_dilations,
            input_left_pads,
            input_right_pads,
            in_element_op,
            wei_element_op,
            out_element_op);

        auto invoker_ptr = conv_ptr->MakeInvokerPointer();

        if(conv_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            std::string conv_name = conv_ptr->GetTypeString();

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

            if(do_verification)
            {
                in_device_buf.FromDevice(in_n_c_hi_wi_device_result.mData.data());

                check_error(in_n_c_hi_wi_host_result, in_n_c_hi_wi_device_result);

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "in : ", out_n_k_ho_wo.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "wei: ", wei_k_c_y_x.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(
                        std::cout << "out_host  : ", in_n_c_hi_wi_host_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(
                        std::cout << "out_device: ", in_n_c_hi_wi_device_result.mData, ",")
                        << std::endl;
                }
            }
        }
    }

    std::cout << "Best Perf: " << best_ave_time << " ms, " << best_tflops << " TFlops, "
              << best_gb_per_sec << " GB/s, " << best_conv_name << std::endl;
}

} // namespace profiler
} // namespace ck
