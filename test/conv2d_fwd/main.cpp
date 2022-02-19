#include "config.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "host_conv.hpp"
#include "tensor_layout.hpp"
#include "device_tensor.hpp"
#include "device_conv_fwd.hpp"
#include "element_wise_operation.hpp"
#include "reference_conv_fwd.hpp"

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

    if(argc == 3)
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
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: run kernel # of times (>1)\n");
        printf("arg4 to 18: N, K, C, Y, X, Hi, Wi, Sy, Sx, Dy, Dx, LeftPy, LeftPx, RightPy, "
               "RightPx\n");
        exit(1);
    }

    auto Run = [&](auto input_type, auto wei_type, auto out_type) {
        using InDataType  = decltype(input_type);
        using WeiDataType = decltype(wei_type);
        using OutDataType = decltype(out_type);

        using ReferenceConvFwdInstance = ck::tensor_operation::host::ReferenceConvFwd<InDataType,
                                                                                      WeiDataType,
                                                                                      OutDataType,
                                                                                      InElementOp,
                                                                                      WeiElementOp,
                                                                                      OutElementOp>;

        const ck::index_t YEff = (Y - 1) * conv_dilation_h + 1;
        const ck::index_t XEff = (X - 1) * conv_dilation_w + 1;

        const ck::index_t Ho = (Hi + in_left_pad_h + in_right_pad_h - YEff) / conv_stride_h + 1;
        const ck::index_t Wo = (Wi + in_left_pad_w + in_right_pad_w - XEff) / conv_stride_w + 1;

        const std::vector<ck::index_t> input_spatial_lengths{Hi, Wi};
        const std::vector<ck::index_t> filter_spatial_lengths{Y, X};
        const std::vector<ck::index_t> output_spatial_lengths{Ho, Wo};
        const std::vector<ck::index_t> conv_filter_strides{conv_stride_h, conv_stride_w};
        const std::vector<ck::index_t> conv_filter_dilations{conv_dilation_h, conv_dilation_w};
        const std::vector<ck::index_t> input_left_pads{in_left_pad_h, in_left_pad_w};
        const std::vector<ck::index_t> input_right_pads{in_right_pad_h, in_right_pad_w};

        auto f_host_tensor_descriptor =
            [](std::size_t N_, std::size_t C_, std::size_t H, std::size_t W) {
                return HostTensorDescriptor(std::vector<std::size_t>({N_, C_, H, W}),
                                            std::vector<std::size_t>({C_ * H * W, 1, W * C_, C_}));
            };

        Tensor<InDataType> in_n_c_hi_wi(f_host_tensor_descriptor(N, C, Hi, Wi));
        Tensor<WeiDataType> wei_k_c_y_x(f_host_tensor_descriptor(K, C, Y, X));
        Tensor<OutDataType> out_n_k_ho_wo_host_result(f_host_tensor_descriptor(N, K, Ho, Wo));
        Tensor<OutDataType> out_n_k_ho_wo_device_result(f_host_tensor_descriptor(N, K, Ho, Wo));

        std::cout << "in_n_c_hi_wi: " << in_n_c_hi_wi.mDesc << std::endl;
        std::cout << "wei_k_c_y_x: " << wei_k_c_y_x.mDesc << std::endl;
        std::cout << "out_n_k_ho_wo: " << out_n_k_ho_wo_host_result.mDesc << std::endl;

        switch(init_method)
        {
        case 0: break;
        case 1:
            in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
            wei_k_c_y_x.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
            break;
        default:
            in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_3<InDataType>{0, 1});
            wei_k_c_y_x.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-1, 1});
        }

        DeviceMem in_device_buf(sizeof(InDataType) * in_n_c_hi_wi.mDesc.GetElementSpace());
        DeviceMem wei_device_buf(sizeof(WeiDataType) * wei_k_c_y_x.mDesc.GetElementSpace());
        DeviceMem out_device_buf(sizeof(OutDataType) *
                                 out_n_k_ho_wo_device_result.mDesc.GetElementSpace());

        in_device_buf.ToDevice(in_n_c_hi_wi.mData.data());
        wei_device_buf.ToDevice(wei_k_c_y_x.mData.data());

        using PassThrough = ck::tensor_operation::element_wise::PassThrough;

        using DeviceConvFwdNoOpPtr =
            ck::tensor_operation::device::DeviceConvFwdPtr<PassThrough, PassThrough, PassThrough>;

        // add device Conv instances
        std::vector<DeviceConvFwdNoOpPtr> conv_ptrs;

        if constexpr(ck::is_same_v<ck::remove_cv_t<InDataType>, float> &&
                     ck::is_same_v<ck::remove_cv_t<WeiDataType>, float> &&
                     ck::is_same_v<ck::remove_cv_t<OutDataType>, float>)
        {
            ck::tensor_operation::device::device_conv2d_fwd_instance::
                add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f32_instances(conv_ptrs);
        }
        else if constexpr(ck::is_same_v<ck::remove_cv_t<InDataType>, ck::half_t> &&
                          ck::is_same_v<ck::remove_cv_t<WeiDataType>, ck::half_t> &&
                          ck::is_same_v<ck::remove_cv_t<OutDataType>, ck::half_t>)
        {
            ck::tensor_operation::device::device_conv2d_fwd_instance::
                add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f16_instances(conv_ptrs);

            ck::tensor_operation::device::device_conv2d_fwd_instance::
                add_device_conv2d_fwd_xdl_c_shuffle_nhwc_kyxc_nhwk_f16_instances(conv_ptrs);
        }
        else if constexpr(ck::is_same_v<ck::remove_cv_t<InDataType>, ushort> &&
                          ck::is_same_v<ck::remove_cv_t<WeiDataType>, ushort> &&
                          ck::is_same_v<ck::remove_cv_t<OutDataType>, ushort>)
        {
            ck::tensor_operation::device::device_conv2d_fwd_instance::
                add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_bf16_instances(conv_ptrs);
        }
        else if constexpr(ck::is_same_v<ck::remove_cv_t<InDataType>, int8_t> &&
                          ck::is_same_v<ck::remove_cv_t<WeiDataType>, int8_t> &&
                          ck::is_same_v<ck::remove_cv_t<OutDataType>, int8_t>)
        {
            ck::tensor_operation::device::device_conv2d_fwd_instance::
                add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_int8_instances(conv_ptrs);
        }

        if(conv_ptrs.size() <= 0)
        {
            throw std::runtime_error("wrong! no device Conv instance found");
        }

        auto ref_conv    = ReferenceConvFwdInstance{};
        auto ref_invoker = ref_conv.MakeInvoker();

        auto ref_argument = ref_conv.MakeArgument(in_n_c_hi_wi,
                                                  wei_k_c_y_x,
                                                  out_n_k_ho_wo_host_result,
                                                  conv_filter_strides,
                                                  conv_filter_dilations,
                                                  input_left_pads,
                                                  input_right_pads,
                                                  InElementOp{},
                                                  WeiElementOp{},
                                                  OutElementOp{});

        ref_invoker.Run(ref_argument);

        // profile device Conv instances
        bool success = false;
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
                PassThrough{},
                PassThrough{},
                PassThrough{});

            auto invoker_ptr = conv_ptr->MakeInvokerPointer();

            if(conv_ptr->IsSupportedArgument(argument_ptr.get()))
            {
                invoker_ptr->Run(argument_ptr.get(), 0);

                out_device_buf.FromDevice(out_n_k_ho_wo_device_result.mData.data());
                if(!check_out(out_n_k_ho_wo_host_result, out_n_k_ho_wo_device_result))
                {
                    success = false;
                    break;
                }
                success = true;
            }
        }

        if(success)
        {
            std::cout << "test conv2d fwd : Pass" << std::endl;
        }
        else
        {
            std::cout << "test conv2d fwd: Fail " << std::endl;
        }
    };

    if(data_type == 0)
    {
        Run(float(), float(), float());
    }
    else if(data_type == 1)
    {
        Run(ck::half_t(), ck::half_t(), ck::half_t());
    }
    else if(data_type == 2)
    {
        Run(ushort(), ushort(), ushort());
    }
    else if(data_type == 3)
    {
        Run(int8_t(), int8_t(), int8_t());
    }
    else
    {
        return 1;
    }

    return 0;
}
