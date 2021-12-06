#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>
#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "device_tensor.hpp"
#include "tensor_layout.hpp"
#include "device_conv2d_fwd_xdl_bias_activation_add_nhwc_kyxc_nhwk.hpp"
#include "element_wise_operation.hpp"

using InDataType  = ck::half_t;
using WeiDataType = ck::half_t;
using OutDataType = ck::half_t;
using AccDataType = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InLayout  = ck::tensor_layout::convolution::NHWC;
using WeiLayout = ck::tensor_layout::convolution::KYXC;
using OutLayout = ck::tensor_layout::convolution::NHWK;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::AddReluAdd;

// clang-format off
using DeviceConvFwdInstance = ck::tensor_operation::device::
    DeviceConv2dFwdXdl_Bias_Activation_Add_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K
//      |    InData|     WeiData|     OutData|     AccData|          In|         Wei|           Out| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer|  BBlockTransfer|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| CThreadTransfer| CThreadTransfer| ABlockLds| BBlockLds|
//      |      Type|        Type|        Type|        Type| Elementwise| Elementwise|   Elementwise|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|     ThreadSlice|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|     ThreadSlice|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| SrcDstVectorDim|       DstScalar| AddExtraM| AddExtraN|
//      |          |            |            |            |   Operation|   Operation|     Operation|      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_N_K1| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1| Lengths_K0_N_K1| Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|                |       PerVector|          |          |
//      |          |            |            |            |            |            |              |      |      |      |      |   |     |     |     |     |                |                |               |               |               |               |               |                |                |               |               |              |               |               |                |                |          |          |
        <InDataType, WeiDataType, OutDataType, AccDataType, InElementOp, WeiElementOp, OutElementOp,   256,   128,   256,     4,  8,   32,   32,    2,    4,      S<1, 2, 8>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      S<1, 4, 8>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,               7,               1,      true,      true>;
// clang-format on

template <typename TIn,
          typename TWei,
          typename TOut,
          typename InElementOp,
          typename WeiElementOp,
          typename OutElementOp>
void host_reference_calculation(const Tensor<TIn>& in_n_c_hi_wi,
                                const Tensor<TWei>& wei_k_c_y_x,
                                Tensor<TOut>& out_n_k_ho_wo,
                                const Tensor<TOut>& bias_k,
                                const Tensor<TOut>& resi_n_k_ho_wo,
                                const std::vector<ck::index_t>& conv_strides,
                                const std::vector<ck::index_t>& conv_dilations,
                                const std::vector<ck::index_t>& in_left_pads,
                                const std::vector<ck::index_t>& /* in_right_pads */,
                                const InElementOp& in_element_op,
                                const WeiElementOp& wei_element_op,
                                const OutElementOp& out_element_op)
{
    auto f_nchw = [&](auto n, auto k, auto ho, auto wo) {
        double v = 0;
        for(int c = 0; c < wei_k_c_y_x.mDesc.GetLengths()[1]; ++c)
        {
            for(int y = 0; y < wei_k_c_y_x.mDesc.GetLengths()[2]; ++y)
            {
                int hi = ho * conv_strides[0] + y * conv_dilations[0] - in_left_pads[0];
                for(int x = 0; x < wei_k_c_y_x.mDesc.GetLengths()[3]; ++x)
                {
                    int wi = wo * conv_strides[1] + x * conv_dilations[1] - in_left_pads[1];
                    if(hi >= 0 && hi < in_n_c_hi_wi.mDesc.GetLengths()[2] && wi >= 0 &&
                       wi < in_n_c_hi_wi.mDesc.GetLengths()[3])
                    {
                        v += in_element_op(static_cast<const double>(in_n_c_hi_wi(n, c, hi, wi))) *
                             wei_element_op(static_cast<const double>(wei_k_c_y_x(k, c, y, x)));
                    }
                }
            }
        }

        out_n_k_ho_wo(n, k, ho, wo) = out_element_op(v, bias_k(k), resi_n_k_ho_wo(n, k, ho, wo));
    };

    make_ParallelTensorFunctor(f_nchw,
                               out_n_k_ho_wo.mDesc.GetLengths()[0],
                               out_n_k_ho_wo.mDesc.GetLengths()[1],
                               out_n_k_ho_wo.mDesc.GetLengths()[2],
                               out_n_k_ho_wo.mDesc.GetLengths()[3])(
        std::thread::hardware_concurrency());
}

int main(int argc, char* argv[])
{
    bool do_verification = 0;
    int init_method      = 0;
    int nrepeat          = 5;

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

    if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        nrepeat         = std::stoi(argv[3]);
    }
    else if(argc == 19)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        nrepeat         = std::stoi(argv[3]);

        N               = std::stoi(argv[4]);
        K               = std::stoi(argv[5]);
        C               = std::stoi(argv[6]);
        Y               = std::stoi(argv[7]);
        X               = std::stoi(argv[8]);
        Hi              = std::stoi(argv[9]);
        Wi              = std::stoi(argv[10]);
        conv_stride_h   = std::stoi(argv[11]);
        conv_stride_w   = std::stoi(argv[12]);
        conv_dilation_h = std::stoi(argv[13]);
        conv_dilation_w = std::stoi(argv[14]);
        in_left_pad_h   = std::stoi(argv[15]);
        in_left_pad_w   = std::stoi(argv[16]);
        in_right_pad_h  = std::stoi(argv[17]);
        in_right_pad_w  = std::stoi(argv[18]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: run kernel # of times (>1)\n");
        printf("arg4 to 18: N, K, C, Y, X, Hi, Wi, Sy, Sx, Dy, Dx, LeftPy, LeftPx, RightPy, "
               "RightPx\n");
        exit(0);
    }

    const ck::index_t YEff = (Y - 1) * conv_dilation_h + 1;
    const ck::index_t XEff = (X - 1) * conv_dilation_w + 1;

    const ck::index_t Ho = (Hi + in_left_pad_h + in_right_pad_h - YEff) / conv_stride_h + 1;
    const ck::index_t Wo = (Wi + in_left_pad_w + in_right_pad_w - XEff) / conv_stride_w + 1;

    const std::vector<ck::index_t> conv_filter_strides{{conv_stride_h, conv_stride_w}};
    const std::vector<ck::index_t> conv_filter_dilations{{conv_dilation_h, conv_dilation_w}};
    const std::vector<ck::index_t> input_left_pads{{in_left_pad_h, in_left_pad_w}};
    const std::vector<ck::index_t> input_right_pads{{in_right_pad_h, in_right_pad_w}};

    // tensor layout
    auto f_host_tensor_descriptor = [](std::size_t N_,
                                       std::size_t C_,
                                       std::size_t H,
                                       std::size_t W,
                                       auto layout) {
        if constexpr(ck::is_same<decltype(layout), ck::tensor_layout::convolution::NCHW>::value ||
                     ck::is_same<decltype(layout), ck::tensor_layout::convolution::KCYX>::value ||
                     ck::is_same<decltype(layout), ck::tensor_layout::convolution::NKHW>::value)
        {
            return HostTensorDescriptor(std::vector<std::size_t>({N_, C_, H, W}),
                                        std::vector<std::size_t>({C_ * H * W, H * W, W, 1}));
        }
        else if constexpr(ck::is_same<decltype(layout),
                                      ck::tensor_layout::convolution::NHWC>::value ||
                          ck::is_same<decltype(layout),
                                      ck::tensor_layout::convolution::KYXC>::value ||
                          ck::is_same<decltype(layout),
                                      ck::tensor_layout::convolution::NHWK>::value)
        {
            return HostTensorDescriptor(std::vector<std::size_t>({N_, C_, H, W}),
                                        std::vector<std::size_t>({C_ * H * W, 1, W * C_, C_}));
        }
    };

    Tensor<InDataType> in_n_c_hi_wi(f_host_tensor_descriptor(N, C, Hi, Wi, InLayout{}));
    Tensor<WeiDataType> wei_k_c_y_x(f_host_tensor_descriptor(K, C, Y, X, WeiLayout{}));
    Tensor<OutDataType> out_n_k_ho_wo_host_result(
        f_host_tensor_descriptor(N, K, Ho, Wo, OutLayout{}));
    Tensor<OutDataType> out_n_k_ho_wo_device_result(
        f_host_tensor_descriptor(N, K, Ho, Wo, OutLayout{}));

    // bias: assume contiguous 1d vector
    Tensor<OutDataType> bias_k(
        HostTensorDescriptor(std::vector<std::size_t>({static_cast<std::size_t>(K)})));

    // residual: assume same layout as output tensor
    Tensor<OutDataType> resi_n_k_ho_wo(f_host_tensor_descriptor(N, K, Ho, Wo, OutLayout{}));

    std::cout << "in_n_c_hi_wi: " << in_n_c_hi_wi.mDesc << std::endl;
    std::cout << "wei_k_c_y_x: " << wei_k_c_y_x.mDesc << std::endl;
    std::cout << "out_n_k_ho_wo: " << out_n_k_ho_wo_host_result.mDesc << std::endl;
    std::cout << "bias_k: " << bias_k.mDesc << std::endl;
    std::cout << "resi_n_k_ho_wo: " << resi_n_k_ho_wo.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        wei_k_c_y_x.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        bias_k.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});
        resi_n_k_ho_wo.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});
        break;
    default:
        in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
        wei_k_c_y_x.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
        bias_k.GenerateTensorValue(GeneratorTensor_3<OutDataType>{0.0, 1.0});
        resi_n_k_ho_wo.GenerateTensorValue(GeneratorTensor_3<OutDataType>{0.0, 1.0});
    }

    DeviceMem in_device_buf(sizeof(InDataType) * in_n_c_hi_wi.mDesc.GetElementSpace());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei_k_c_y_x.mDesc.GetElementSpace());
    DeviceMem out_device_buf(sizeof(OutDataType) *
                             out_n_k_ho_wo_device_result.mDesc.GetElementSpace());
    DeviceMem bias_device_buf(sizeof(OutDataType) * bias_k.mDesc.GetElementSpace());
    DeviceMem resi_device_buf(sizeof(OutDataType) * resi_n_k_ho_wo.mDesc.GetElementSpace());

    in_device_buf.ToDevice(in_n_c_hi_wi.mData.data());
    wei_device_buf.ToDevice(wei_k_c_y_x.mData.data());
    bias_device_buf.ToDevice(bias_k.mData.data());
    resi_device_buf.ToDevice(resi_n_k_ho_wo.mData.data());

    auto conv    = DeviceConvFwdInstance{};
    auto invoker = conv.MakeInvoker();
    auto argument =
        conv.MakeArgument(static_cast<const InDataType*>(in_device_buf.GetDeviceBuffer()),
                          static_cast<const WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                          static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                          static_cast<const OutDataType*>(bias_device_buf.GetDeviceBuffer()),
                          static_cast<const OutDataType*>(resi_device_buf.GetDeviceBuffer()),
                          N,
                          K,
                          C,
                          std::vector<ck::index_t>{{Hi, Wi}},
                          std::vector<ck::index_t>{{Y, X}},
                          std::vector<ck::index_t>{{Ho, Wo}},
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
            "wrong! device operator with the specified compilation parameters does "
            "not support this problem");
    }

    float ave_time = invoker.Run(argument, nrepeat);

    std::size_t flop = std::size_t(2) * N * K * Ho * Wo * C * Y * X;

    std::size_t num_btype = sizeof(InDataType) * (N * C * Hi * Wi) +
                            sizeof(WeiDataType) * (K * C * Y * X) +
                            sizeof(OutDataType) * (N * K * Ho * Wo) + sizeof(OutDataType) * (K) +
                            sizeof(OutDataType) * (N * K * Ho * Wo);

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    if(do_verification)
    {
        host_reference_calculation(in_n_c_hi_wi,
                                   wei_k_c_y_x,
                                   out_n_k_ho_wo_host_result,
                                   bias_k,
                                   resi_n_k_ho_wo,
                                   conv_filter_strides,
                                   conv_filter_dilations,
                                   input_left_pads,
                                   input_right_pads,
                                   InElementOp{},
                                   WeiElementOp{},
                                   OutElementOp{});

        out_device_buf.FromDevice(out_n_k_ho_wo_device_result.mData.data());

        check_error(out_n_k_ho_wo_host_result, out_n_k_ho_wo_device_result);
    }
}
