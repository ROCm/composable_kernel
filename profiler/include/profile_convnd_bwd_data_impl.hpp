#pragma once
#include "config.hpp"
#include "device.hpp"
#include "conv_utils.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "tensor_layout.hpp"
#include "device_tensor.hpp"
#include "device_conv_bwd_data.hpp"
#include "element_wise_operation.hpp"
#include "reference_conv_bwd_data.hpp"

using F16  = ck::half_t;
using F32  = float;
using BF16 = ushort;
using INT8 = int8_t;
namespace ck {
namespace tensor_operation {
namespace device {
namespace device_conv2d_bwd_data_instance {

using DeviceConvBwdDataNoOpPtr =
    DeviceConvBwdDataPtr<ck::tensor_operation::element_wise::PassThrough,
                         ck::tensor_operation::element_wise::PassThrough,
                         ck::tensor_operation::element_wise::PassThrough>;
void add_device_conv1d_bwd_data_xdl_nwc_kxc_nwk_f32_instances(
    std::vector<DeviceConvBwdDataNoOpPtr>&);
void add_device_conv1d_bwd_data_xdl_nwc_kxc_nwk_f16_instances(
    std::vector<DeviceConvBwdDataNoOpPtr>&);
void add_device_conv1d_bwd_data_xdl_nwc_kxc_nwk_bf16_instances(
    std::vector<DeviceConvBwdDataNoOpPtr>&);
void add_device_conv1d_bwd_data_xdl_nwc_kxc_nwk_int8_instances(
    std::vector<DeviceConvBwdDataNoOpPtr>&);

void add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_f32_instances(
    std::vector<DeviceConvBwdDataNoOpPtr>&);
void add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_f16_instances(
    std::vector<DeviceConvBwdDataNoOpPtr>&);
void add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_bf16_instances(
    std::vector<DeviceConvBwdDataNoOpPtr>&);
void add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_int8_instances(
    std::vector<DeviceConvBwdDataNoOpPtr>&);

void add_device_conv3d_bwd_data_xdl_ndhwc_kzyxc_ndhwk_f32_instances(
    std::vector<DeviceConvBwdDataNoOpPtr>&);
void add_device_conv3d_bwd_data_xdl_ndhwc_kzyxc_ndhwk_f16_instances(
    std::vector<DeviceConvBwdDataNoOpPtr>&);
void add_device_conv3d_bwd_data_xdl_ndhwc_kzyxc_ndhwk_bf16_instances(
    std::vector<DeviceConvBwdDataNoOpPtr>&);
void add_device_conv3d_bwd_data_xdl_ndhwc_kzyxc_ndhwk_int8_instances(
    std::vector<DeviceConvBwdDataNoOpPtr>&);
} // namespace device_conv2d_bwd_data_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

namespace ck {
namespace profiler {
using DeviceConvBwdDataNoOpPtr =
    ck::tensor_operation::device::device_conv2d_bwd_data_instance::DeviceConvBwdDataNoOpPtr;

HostTensorDescriptor GetInputHostTensorDescriptor(const std::vector<std::size_t>& dims,
                                                  int num_dim_spatial = 2)
{
    namespace tl = ck::tensor_layout::convolution;

    switch(num_dim_spatial)
    {
    case 3: {
        return ck::conv_util::GetHostTensorDescriptor(dims, tl::NDHWC{});
    }
    case 2: {
        return ck::conv_util::GetHostTensorDescriptor(dims, tl::NHWC{});
    }
    case 1: {
        return ck::conv_util::GetHostTensorDescriptor(dims, tl::NWC{});
    }
    default: {
        throw std::runtime_error("Unsupported number of spatial dimensions provided!");
    }
    }
}
HostTensorDescriptor GetFiltersHostTensorDescriptor(const std::vector<std::size_t>& dims,
                                                    int num_dim_spatial = 2)
{
    namespace tl = ck::tensor_layout::convolution;

    switch(num_dim_spatial)
    {
    case 3: {
        return ck::conv_util::GetHostTensorDescriptor(dims, tl::KZYXC{});
    }
    case 2: {
        return ck::conv_util::GetHostTensorDescriptor(dims, tl::KYXC{});
    }
    case 1: {
        return ck::conv_util::GetHostTensorDescriptor(dims, tl::KXC{});
    }
    default: {
        throw std::runtime_error("Unsupported number of spatial dimensions provided!");
    }
    }
}

HostTensorDescriptor GetOutputHostTensorDescriptor(const std::vector<std::size_t>& dims,
                                                   int num_dim_spatial = 2)
{
    namespace tl = ck::tensor_layout::convolution;

    switch(num_dim_spatial)
    {
    case 3: {
        return ck::conv_util::GetHostTensorDescriptor(dims, tl::NDHWK{});
    }
    case 2: {
        return ck::conv_util::GetHostTensorDescriptor(dims, tl::NHWK{});
    }
    case 1: {
        return ck::conv_util::GetHostTensorDescriptor(dims, tl::NWK{});
    }

    default: {
        throw std::runtime_error("Unsupported number of spatial dimensions provided!");
    }
    }
}
template <typename InDataType, typename WeiDataType, typename OutDataType>
void GetDeviceConvBwdDataOpPtr(
    InDataType, WeiDataType, OutDataType, std::vector<DeviceConvBwdDataNoOpPtr>&, int)
{
    std::cout << "can not find device conv bwd data" << std::endl;
    exit(1);
}
template <>
void GetDeviceConvBwdDataOpPtr(
    F32, F32, F32, std::vector<DeviceConvBwdDataNoOpPtr>& conv_ptrs, int num_dim_spatial)
{
    switch(num_dim_spatial)
    {
    case 1:
        ck::tensor_operation::device::device_conv2d_bwd_data_instance::
            add_device_conv1d_bwd_data_xdl_nwc_kxc_nwk_f32_instances(conv_ptrs);
        break;
    case 2:
        ck::tensor_operation::device::device_conv2d_bwd_data_instance::
            add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_f32_instances(conv_ptrs);
        break;
    case 3:
        ck::tensor_operation::device::device_conv2d_bwd_data_instance::
            add_device_conv3d_bwd_data_xdl_ndhwc_kzyxc_ndhwk_f32_instances(conv_ptrs);
        break;
    default: break;
    }
}
template <>
void GetDeviceConvBwdDataOpPtr(
    F16, F16, F16, std::vector<DeviceConvBwdDataNoOpPtr>& conv_ptrs, int num_dim_spatial)
{
    switch(num_dim_spatial)
    {
    case 1:
        ck::tensor_operation::device::device_conv2d_bwd_data_instance::
            add_device_conv1d_bwd_data_xdl_nwc_kxc_nwk_f16_instances(conv_ptrs);
        break;
    case 2:
        ck::tensor_operation::device::device_conv2d_bwd_data_instance::
            add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_f16_instances(conv_ptrs);
        break;
    case 3:
        ck::tensor_operation::device::device_conv2d_bwd_data_instance::
            add_device_conv3d_bwd_data_xdl_ndhwc_kzyxc_ndhwk_f16_instances(conv_ptrs);
        break;
    default: break;
    }
}
template <>
void GetDeviceConvBwdDataOpPtr(
    BF16, BF16, BF16, std::vector<DeviceConvBwdDataNoOpPtr>& conv_ptrs, int num_dim_spatial)
{
    switch(num_dim_spatial)
    {
    case 1:
        ck::tensor_operation::device::device_conv2d_bwd_data_instance::
            add_device_conv1d_bwd_data_xdl_nwc_kxc_nwk_bf16_instances(conv_ptrs);
        break;
    case 2:
        ck::tensor_operation::device::device_conv2d_bwd_data_instance::
            add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_bf16_instances(conv_ptrs);
        break;
    case 3:
        ck::tensor_operation::device::device_conv2d_bwd_data_instance::
            add_device_conv3d_bwd_data_xdl_ndhwc_kzyxc_ndhwk_bf16_instances(conv_ptrs);
        break;
    default: break;
    }
}
template <>
void GetDeviceConvBwdDataOpPtr(
    INT8, INT8, INT8, std::vector<DeviceConvBwdDataNoOpPtr>& conv_ptrs, int num_dim_spatial)
{
    switch(num_dim_spatial)
    {
    case 1:
        ck::tensor_operation::device::device_conv2d_bwd_data_instance::
            add_device_conv1d_bwd_data_xdl_nwc_kxc_nwk_int8_instances(conv_ptrs);
        break;
    case 2:
        ck::tensor_operation::device::device_conv2d_bwd_data_instance::
            add_device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk_int8_instances(conv_ptrs);
        break;
    case 3:
        ck::tensor_operation::device::device_conv2d_bwd_data_instance::
            add_device_conv3d_bwd_data_xdl_ndhwc_kzyxc_ndhwk_int8_instances(conv_ptrs);
        break;
    default: break;
    }
}

template <int NDimSpatial,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename AccDataType,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout>
void profile_convnd_bwd_data_impl(int do_verification,
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
    using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
    using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
    using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

    const auto in_element_op  = InElementOp{};
    const auto wei_element_op = WeiElementOp{};
    const auto out_element_op = OutElementOp{};

    std::vector<std::size_t> input_dims{static_cast<std::size_t>(N), static_cast<std::size_t>(C)};
    input_dims.insert(
        std::end(input_dims), std::begin(input_spatial_lengths), std::end(input_spatial_lengths));

    std::vector<std::size_t> filter_dims{static_cast<std::size_t>(K), static_cast<std::size_t>(C)};
    filter_dims.insert(std::end(filter_dims),
                       std::begin(filter_spatial_lengths),
                       std::end(filter_spatial_lengths));

    std::vector<std::size_t> output_dims{static_cast<std::size_t>(N), static_cast<std::size_t>(K)};
    output_dims.insert(std::end(output_dims),
                       std::begin(output_spatial_lengths),
                       std::end(output_spatial_lengths));

    Tensor<InDataType> in_n_c_hi_wi_host_result(
        GetInputHostTensorDescriptor(input_dims, NDimSpatial));
    Tensor<InDataType> in_n_c_hi_wi_device_result(
        GetInputHostTensorDescriptor(input_dims, NDimSpatial));
    Tensor<WeiDataType> wei_k_c_y_x(GetFiltersHostTensorDescriptor(filter_dims, NDimSpatial));
    Tensor<OutDataType> out_n_k_ho_wo(GetOutputHostTensorDescriptor(output_dims, NDimSpatial));

    std::cout << "in_n_c_hi_wi: " << in_n_c_hi_wi_host_result.mDesc << std::endl;
    std::cout << "wei_k_c_y_x: " << wei_k_c_y_x.mDesc << std::endl;
    std::cout << "out_n_k_ho_wo: " << out_n_k_ho_wo.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        out_n_k_ho_wo.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});
        wei_k_c_y_x.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        break;
    default:
        out_n_k_ho_wo.GenerateTensorValue(GeneratorTensor_1<OutDataType>{1});
        wei_k_c_y_x.GenerateTensorValue(GeneratorTensor_1<WeiDataType>{1});
    }

    DeviceMem in_device_buf(sizeof(InDataType) *
                            in_n_c_hi_wi_device_result.mDesc.GetElementSpace());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei_k_c_y_x.mDesc.GetElementSpace());
    DeviceMem out_device_buf(sizeof(OutDataType) * out_n_k_ho_wo.mDesc.GetElementSpace());

    out_device_buf.ToDevice(out_n_k_ho_wo.mData.data());
    wei_device_buf.ToDevice(wei_k_c_y_x.mData.data());

    // reset input to zero
    in_n_c_hi_wi_device_result.GenerateTensorValue(GeneratorTensor_1<InDataType>{0});
    in_device_buf.ToDevice(in_n_c_hi_wi_device_result.mData.data());

    if(do_verification)
    {
        auto RunReference = [&](auto& ref_conv) {
            auto ref_invoker = ref_conv.MakeInvoker();

            auto ref_argument = ref_conv.MakeArgument(in_n_c_hi_wi_host_result,
                                                      wei_k_c_y_x,
                                                      out_n_k_ho_wo,
                                                      conv_filter_strides,
                                                      conv_filter_dilations,
                                                      input_left_pads,
                                                      input_right_pads,
                                                      InElementOp{},
                                                      WeiElementOp{},
                                                      OutElementOp{});
            ref_invoker.Run(ref_argument);
        };
        switch(NDimSpatial)
        {
        case 3: {
            auto ref_conv = ck::tensor_operation::host::ReferenceConvBwdData<InDataType,
                                                                             WeiDataType,
                                                                             OutDataType,
                                                                             AccDataType,
                                                                             InElementOp,
                                                                             WeiElementOp,
                                                                             OutElementOp,
                                                                             3>();
            RunReference(ref_conv);
            break;
        }
        case 2: {
            auto ref_conv = ck::tensor_operation::host::ReferenceConvBwdData<InDataType,
                                                                             WeiDataType,
                                                                             OutDataType,
                                                                             AccDataType,
                                                                             InElementOp,
                                                                             WeiElementOp,
                                                                             OutElementOp,
                                                                             2>();
            RunReference(ref_conv);
            break;
        }
        case 1: {
            auto ref_conv = ck::tensor_operation::host::ReferenceConvBwdData<InDataType,
                                                                             WeiDataType,
                                                                             OutDataType,
                                                                             AccDataType,
                                                                             InElementOp,
                                                                             WeiElementOp,
                                                                             OutElementOp,
                                                                             1>();
            RunReference(ref_conv);
            break;
        }
        default: {
            throw std::runtime_error("Unsupported number of spatial dimensions provided!");
        }
        }
    }

    // add device Conv instances
    std::vector<DeviceConvBwdDataNoOpPtr> conv_ptrs;
    GetDeviceConvBwdDataOpPtr(InDataType{}, WeiDataType{}, OutDataType{}, conv_ptrs, NDimSpatial);

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

            std::size_t flop =
                ck::conv_util::GetFlops(N, C, K, filter_spatial_lengths, output_spatial_lengths);
            std::size_t num_btype = ck::conv_util::GetBtype<InDataType, WeiDataType, OutDataType>(
                N, C, K, input_spatial_lengths, filter_spatial_lengths, output_spatial_lengths);

            float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
            float gb_per_sec = num_btype / 1.E6 / ave_time;

            std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                      << " GB/s" << std::endl;

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
