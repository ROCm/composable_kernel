#include "config.hpp"
#include "device.hpp"
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

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

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

int main(int argc, char* argv[])
{
    int data_type   = 0;
    int init_method = 0;

    // Conv shape
    ck::index_t N               = 128;
    ck::index_t K               = 256;
    ck::index_t C               = 192;
    ck::index_t Y               = 3;
    ck::index_t X               = 3;
    ck::index_t Hi              = 71;
    ck::index_t Wi              = 71;
    ck::index_t conv_stride_h   = 2;
    ck::index_t conv_stride_w   = 2;
    ck::index_t conv_dilation_h = 1;
    ck::index_t conv_dilation_w = 1;
    ck::index_t in_left_pad_h   = 1;
    ck::index_t in_left_pad_w   = 1;
    ck::index_t in_right_pad_h  = 1;
    ck::index_t in_right_pad_w  = 1;

    if(argc == 1)
    {
        data_type   = 1;
        init_method = 1;
    }
    else if(argc == 3)
    {
        data_type   = std::stoi(argv[1]);
        init_method = std::stoi(argv[2]);
    }
    else if(argc == 18)
    {
        data_type   = std::stoi(argv[1]);
        init_method = std::stoi(argv[2]);

        N               = std::stoi(argv[3]);
        K               = std::stoi(argv[4]);
        C               = std::stoi(argv[5]);
        Y               = std::stoi(argv[6]);
        X               = std::stoi(argv[7]);
        Hi              = std::stoi(argv[8]);
        Wi              = std::stoi(argv[9]);
        conv_stride_h   = std::stoi(argv[10]);
        conv_stride_w   = std::stoi(argv[11]);
        conv_dilation_h = std::stoi(argv[12]);
        conv_dilation_w = std::stoi(argv[13]);
        in_left_pad_h   = std::stoi(argv[14]);
        in_left_pad_w   = std::stoi(argv[15]);
        in_right_pad_h  = std::stoi(argv[16]);
        in_right_pad_w  = std::stoi(argv[17]);
    }
    else
    {
        printf("arg1: data type (0=fp32, 1=fp16, 2= bfp16, 3= int8_t )\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3 to 17: N, K, C, Y, X, Hi, Wi, Sy, Sx, Dy, Dx, LeftPy, LeftPx, RightPy, "
               "RightPx\n");
        exit(1);
    }

    auto Run = [&](auto input_type, auto wei_type, auto out_type, auto acc_type) {
        using InDataType  = decltype(input_type);
        using WeiDataType = decltype(wei_type);
        using OutDataType = decltype(out_type);
        using AccDataType = decltype(acc_type);

        using ReferenceConvBwdInstance =
            ck::tensor_operation::host::ReferenceConvBwdData<InDataType,
                                                             WeiDataType,
                                                             OutDataType,
                                                             AccDataType,
                                                             InElementOp,
                                                             WeiElementOp,
                                                             OutElementOp>;

        const ck::index_t YEff = (Y - 1) * conv_dilation_h + 1;
        const ck::index_t XEff = (X - 1) * conv_dilation_w + 1;

        const ck::index_t Ho = (Hi + in_left_pad_h + in_right_pad_h - YEff) / conv_stride_h + 1;
        const ck::index_t Wo = (Wi + in_left_pad_w + in_right_pad_w - XEff) / conv_stride_w + 1;

        const std::vector<ck::index_t> input_spatial_lengths{{Hi, Wi}};
        const std::vector<ck::index_t> filter_spatial_lengths{{Y, X}};
        const std::vector<ck::index_t> output_spatial_lengths{{Ho, Wo}};
        const std::vector<ck::index_t> conv_filter_strides{{conv_stride_h, conv_stride_w}};
        const std::vector<ck::index_t> conv_filter_dilations{{conv_dilation_h, conv_dilation_w}};
        const std::vector<ck::index_t> input_left_pads{{in_left_pad_h, in_left_pad_w}};
        const std::vector<ck::index_t> input_right_pads{{in_right_pad_h, in_right_pad_w}};

        auto f_host_tensor_descriptor =
            [](std::size_t N_, std::size_t C_, std::size_t H, std::size_t W) {
                return HostTensorDescriptor(std::vector<std::size_t>({N_, C_, H, W}),
                                            std::vector<std::size_t>({C_ * H * W, 1, W * C_, C_}));
            };

        Tensor<OutDataType> out_n_k_ho_wo(f_host_tensor_descriptor(N, K, Ho, Wo));
        Tensor<WeiDataType> wei_k_c_y_x(f_host_tensor_descriptor(K, C, Y, X));
        Tensor<InDataType> in_n_c_hi_wi_host_result(f_host_tensor_descriptor(N, C, Hi, Wi));
        Tensor<InDataType> in_n_c_hi_wi_device_result(f_host_tensor_descriptor(N, C, Hi, Wi));

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

        // get host result
        {
            auto ref_conv    = ReferenceConvBwdInstance{};
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
        }

        using PassThrough              = ck::tensor_operation::element_wise::PassThrough;
        using DeviceConvBwdDataNoOpPtr = ck::tensor_operation::device::
            DeviceConvBwdDataPtr<PassThrough, PassThrough, PassThrough>;

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

        // profile device Conv instances
        bool success = true;
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
            return 0;
        }
        else
        {
            std::cout << "test conv2d bwd: Fail " << std::endl;
            return -1;
        }
    };

    if(data_type == 0)
    {
        return Run(F32(), F32(), F32(), F32());
    }
    else if(data_type == 1)
    {
        return Run(F16(), F16(), F16(), F32());
    }
    else if(data_type == 2)
    {
        return Run(BF16(), BF16(), BF16(), F32());
    }
    else if(data_type == 3)
    {
        return Run(INT8(), INT8(), INT8(), int());
    }
    else
    {
        return 1;
    }
}
