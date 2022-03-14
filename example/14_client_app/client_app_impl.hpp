#pragma once
#include "config.hpp"
#include "device.hpp"
#include "tensor_layout.hpp"
#include "device_conv_fwd.hpp"
#include "element_wise_operation.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_conv2d_fwd_instance {

using DeviceConvFwdNoOpPtr = DeviceConvFwdPtr<ck::tensor_operation::element_wise::PassThrough,
                                              ck::tensor_operation::element_wise::PassThrough,
                                              ck::tensor_operation::element_wise::PassThrough>;

void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f32_instances(std::vector<DeviceConvFwdNoOpPtr>&);

void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f16_instances(std::vector<DeviceConvFwdNoOpPtr>&);

void add_device_conv2d_fwd_xdl_c_shuffle_nhwc_kyxc_nhwk_f16_instances(
    std::vector<DeviceConvFwdNoOpPtr>&);

void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_bf16_instances(std::vector<DeviceConvFwdNoOpPtr>&);

void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_int8_instances(std::vector<DeviceConvFwdNoOpPtr>&);
} // namespace device_conv2d_fwd_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

namespace ck {
namespace app {

template <int NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout>
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
    using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
    using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
    using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

    const auto in_element_op  = InElementOp{};
    const auto wei_element_op = WeiElementOp{};
    const auto out_element_op = OutElementOp{};

    const auto in_sz = 1000;
    const auto wei_sz = 1000;
    const auto out_sz = 1000;

    DeviceMem in_device_buf(sizeof(InDataType) * in_sz);
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei_sz);
    DeviceMem out_device_buf(sizeof(OutDataType) * out_sz);
    // data is already on device!
    // in_device_buf.ToDevice(in_n_c_hi_wi.mData.data());
    // wei_device_buf.ToDevice(wei_k_c_y_x.mData.data());

    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    using DeviceConvFwdNoOpPtr =
        ck::tensor_operation::device::DeviceConvFwdPtr<PassThrough, PassThrough, PassThrough>;

    // add device Conv instances
    std::vector<DeviceConvFwdNoOpPtr> conv_ptrs;

        ck::tensor_operation::device::device_conv2d_fwd_instance::
            add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f32_instances(conv_ptrs);
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
        }
    }

    std::cout << "Best Perf: " << best_ave_time << " ms, " << best_tflops << " TFlops, "
              << best_gb_per_sec << " GB/s, " << best_conv_name << std::endl;
}

} // namespace profiler
} // namespace ck
