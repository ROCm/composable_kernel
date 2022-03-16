#include "config.hpp"
#include "device.hpp"
#include "conv_utils.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "host_conv.hpp"
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

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

using DeviceConvBwdDataNoOpPtr =
    ck::tensor_operation::device::device_conv2d_bwd_data_instance::DeviceConvBwdDataNoOpPtr;
using DeviceConvBwdDataBasePtr =
    ck::tensor_operation::device::DeviceConvBwdDataPtr<InElementOp, WeiElementOp, OutElementOp>;

template <typename T>
static bool check_out(const Tensor<T>& ref, const Tensor<T>& result)
{
    float max_diff = 1e-6;

    for(int i = 0; i < ref.mData.size(); ++i)
    {
        float diff = std::abs(double(ref.mData[i]) - double(result.mData[i]));
        if(max_diff < diff)
        {
            return false;
        }
    }

    return true;
}

void PrintUseMsg()
{
    std::cout << "arg1: verification (0=no, 1=yes)\n"
              << "arg2: initialization (0=no init, 1=random value, 2= init to 1 )\n"
              << "arg3: run kernel # of times (>1)\n"
              << "arg4: N spatial dimensions (default 2)\n"
              << "Following arguments (depending on number of spatial dims):\n"
              << " N, K, C, \n"
              << " <filter spatial dimensions>, (ie Y, X for 2D)\n"
              << " <input image spatial dimensions>, (ie Hi, Wi for 2D)\n"
              << " <strides>, (ie Sy, Sx for 2D)\n"
              << " <dilations>, (ie Dy, Dx for 2D)\n"
              << " <left padding>, (ie LeftPy, LeftPx for 2D)\n"
              << " <right padding>, (ie RightPy, RightPx for 2D)\n"
              << std::endl;
}

ck::conv_util::ConvParams ParseConvParams(int num_dim_spatial, char* argv[])
{
    // (N, K, C) + num_dim_spatial * 6 (filter, input, strides, dilations, pad left, pad right)
    ck::conv_util::ConvParams params;
    int arg_idx = 6;

    params.num_dim_spatial = num_dim_spatial;
    params.N               = std::stoi(argv[arg_idx++]);
    params.K               = std::stoi(argv[arg_idx++]);
    params.C               = std::stoi(argv[arg_idx++]);

    params.filter_spatial_lengths.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.filter_spatial_lengths[i] = std::stoi(argv[arg_idx++]);
    }
    params.input_spatial_lengths.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.input_spatial_lengths[i] = std::stoi(argv[arg_idx++]);
    }
    params.conv_filter_strides.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.conv_filter_strides[i] = std::stoi(argv[arg_idx++]);
    }
    params.conv_filter_dilations.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.conv_filter_dilations[i] = std::stoi(argv[arg_idx++]);
    }
    params.input_left_pads.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.input_left_pads[i] = std::stoi(argv[arg_idx++]);
    }
    params.input_right_pads.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.input_right_pads[i] = std::stoi(argv[arg_idx++]);
    }

    return params;
}

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

void GetDeviceConvBwdDataOpPtr(F32,
                               std::vector<DeviceConvBwdDataNoOpPtr>& conv_ptrs,
                               int num_dim_spatial)
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
void GetDeviceConvBwdDataOpPtr(F16,
                               std::vector<DeviceConvBwdDataNoOpPtr>& conv_ptrs,
                               int num_dim_spatial)
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
void GetDeviceConvBwdDataOpPtr(BF16,
                               std::vector<DeviceConvBwdDataNoOpPtr>& conv_ptrs,
                               int num_dim_spatial)
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
void GetDeviceConvBwdDataOpPtr(INT8,
                               std::vector<DeviceConvBwdDataNoOpPtr>& conv_ptrs,
                               int num_dim_spatial)
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
int main(int argc, char* argv[])
{
    int data_type        = 1;
    bool do_verification = 0;
    int init_method      = 0;
    int nrepeat          = 5;
    int num_dim_spatial  = 2;

    ck::conv_util::ConvParams params;

    if(argc == 5)
    {
        data_type       = std::stoi(argv[1]);
        do_verification = std::stoi(argv[2]);
        init_method     = std::stoi(argv[3]);
        nrepeat         = std::stoi(argv[4]);
    }
    else
    {
        data_type       = std::stoi(argv[1]);
        do_verification = std::stoi(argv[2]);
        init_method     = std::stoi(argv[3]);
        nrepeat         = std::stoi(argv[4]);
        num_dim_spatial = std::stoi(argv[5]);
        // check args number
        int conv_args     = 3 + num_dim_spatial * 6;
        int cmdline_nargs = conv_args + 6;
        if(cmdline_nargs != argc)
        {
            PrintUseMsg();
            exit(1);
        }

        params = ParseConvParams(num_dim_spatial, argv);
    }

    auto Run = [&](auto input_type, auto wei_type, auto out_type) {
        using InDataType  = decltype(input_type);
        using WeiDataType = decltype(wei_type);
        using OutDataType = decltype(out_type);

        std::vector<std::size_t> input_dims{static_cast<std::size_t>(params.N),
                                            static_cast<std::size_t>(params.C)};
        input_dims.insert(std::end(input_dims),
                          std::begin(params.input_spatial_lengths),
                          std::end(params.input_spatial_lengths));

        std::vector<std::size_t> filter_dims{static_cast<std::size_t>(params.K),
                                             static_cast<std::size_t>(params.C)};
        filter_dims.insert(std::end(filter_dims),
                           std::begin(params.filter_spatial_lengths),
                           std::end(params.filter_spatial_lengths));

        const std::vector<ck::index_t>& output_spatial_lengths = params.GetOutputSpatialLengths();
        std::vector<std::size_t> output_dims{static_cast<std::size_t>(params.N),
                                             static_cast<std::size_t>(params.K)};
        output_dims.insert(std::end(output_dims),
                           std::begin(output_spatial_lengths),
                           std::end(output_spatial_lengths));

        Tensor<InDataType> in_n_c_hi_wi_host_result(
            GetInputHostTensorDescriptor(input_dims, num_dim_spatial));
        Tensor<InDataType> in_n_c_hi_wi_device_result(
            GetInputHostTensorDescriptor(input_dims, num_dim_spatial));
        Tensor<WeiDataType> wei_k_c_y_x(
            GetFiltersHostTensorDescriptor(filter_dims, num_dim_spatial));
        Tensor<OutDataType> out_n_k_ho_wo(
            GetOutputHostTensorDescriptor(output_dims, num_dim_spatial));

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

        // get host result
        {
            auto RunReference = [&](auto& ref_conv) {
                auto ref_invoker = ref_conv.MakeInvoker();

                auto ref_argument = ref_conv.MakeArgument(in_n_c_hi_wi_host_result,
                                                          wei_k_c_y_x,
                                                          out_n_k_ho_wo,
                                                          params.conv_filter_strides,
                                                          params.conv_filter_dilations,
                                                          params.input_left_pads,
                                                          params.input_right_pads,
                                                          InElementOp{},
                                                          WeiElementOp{},
                                                          OutElementOp{});
                ref_invoker.Run(ref_argument);
            };
            switch(num_dim_spatial)
            {
            case 3: {
                auto ref_conv = ck::tensor_operation::host::ReferenceConvBwdData<InDataType,
                                                                                 WeiDataType,
                                                                                 OutDataType,
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
        GetDeviceConvBwdDataOpPtr(InDataType{}, conv_ptrs, num_dim_spatial);

        if(conv_ptrs.size() <= 0)
        {
            throw std::runtime_error("wrong! no device Conv instance found");
        }

        // profile device Conv instances
        bool success = true;
        for(auto& conv_ptr : conv_ptrs)
        {
            auto argument_ptr = conv_ptr->MakeArgumentPointer(
                static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
                static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                params.N,
                params.K,
                params.C,
                params.input_spatial_lengths,
                params.filter_spatial_lengths,
                output_spatial_lengths,
                params.conv_filter_strides,
                params.conv_filter_dilations,
                params.input_left_pads,
                params.input_right_pads,
                InElementOp{},
                WeiElementOp{},
                OutElementOp{});

            if(conv_ptr->IsSupportedArgument(argument_ptr.get()))
            {
                auto invoker_ptr = conv_ptr->MakeInvokerPointer();
                invoker_ptr->Run(argument_ptr.get(), 1);

                in_device_buf.FromDevice(in_n_c_hi_wi_device_result.mData.data());

                if(!check_out(in_n_c_hi_wi_host_result, in_n_c_hi_wi_device_result))
                {
                    std::cout << "Fail Info: " << conv_ptr->GetTypeString() << std::endl;
                    success = false;
                }
                else
                {
                    std::cout << "Pass Info: " << conv_ptr->GetTypeString() << std::endl;
                }
            }
            else
            {
                std::cout << "Not support Info: " << conv_ptr->GetTypeString() << std::endl;
            }
        }

        if(success)
        {
            std::cout << "test conv2d bwd : Pass" << std::endl;
        }
        else
        {
            std::cout << "test conv2d bwd: Fail " << std::endl;
        }
    };

    if(data_type == 0)
    {
        Run(F32(), F32(), F32());
    }
    else if(data_type == 1)
    {
        Run(F16(), F16(), F16());
    }
    else if(data_type == 2)
    {
        Run(BF16(), BF16(), BF16());
    }
    else if(data_type == 3)
    {
        Run(INT8(), INT8(), INT8());
    }
    else
    {
        return 1;
    }

    return 0;
}
