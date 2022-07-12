#include <sstream>
#include "ck/ck.hpp"
#include "ck/device_utility/kernel_launch.hpp"
#include "ck/library/host_tensor/device_memory.hpp"
#include "ck/library/host_tensor/host_tensor.hpp"
#include "ck/library/host_tensor/host_tensor_generator.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/cpu/device/device_convnd_fwd_bias_activation_add_avx2_nhwc_kyxc_nhwk.hpp"
#include "ck/tensor_operation/cpu/element/element_wise_operation_cpu.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd_bias_activation_add.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd_bias_activation.hpp"
#include "ck/utility/dynamic_buffer_cpu.hpp"
#include "ck/utility/envvar.hpp"
#include "ck/device_utility/xdnn_desc.hpp"
#include <omp.h>

#define AVX2_DATA_ALIGNMENT 32

#define TEST_FUSION_BIAS_RELU_ADD 0
#define TEST_FUSION_BIAS_RELU 1
#define TEST_FUSION_BIAS 2
#define TEST_FUSION TEST_FUSION_BIAS

#define TEST_LAYOUT_NHWC_KYXC_NHWK 0
#define TEST_LAYOUT_NHWC_KYXCK8_NHWK 1
#define TEST_LAYOUT_NHWC_YXCK_NHWK 2
#define TEST_LAYOUT TEST_LAYOUT_NHWC_KYXCK8_NHWK

using F32 = float;
using F16 = ck::half_t;

namespace ck {
namespace tensor_operation {
namespace cpu {
namespace device {
namespace device_conv2d_fwd_bias_activation_add_avx2_instance {

using PassThrough = ck::tensor_operation::cpu::element_wise::PassThrough;
using AddReluAdd  = ck::tensor_operation::cpu::element_wise::AddReluAdd;
using AddRelu     = ck::tensor_operation::cpu::element_wise::AddRelu;
using Add         = ck::tensor_operation::cpu::element_wise::Add;

// ------------------ nhwc-kyxc-nhwk
void add_device_conv2d_fwd_bias_relu_add_avx2_nhwc_kyxc_nhwk(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PassThrough, PassThrough, AddReluAdd>>&
        instances);

void add_device_conv2d_fwd_bias_relu_add_avx2_nhwc_kyxc_nhwk_local_c(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PassThrough, PassThrough, AddReluAdd>>&
        instances);

void add_device_conv2d_fwd_bias_relu_add_avx2_nhwc_kyxc_nhwk_mt(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PassThrough, PassThrough, AddReluAdd>>&
        instances);

void add_device_conv2d_fwd_bias_relu_avx2_nhwc_kyxc_nhwk(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PassThrough, PassThrough, AddRelu>>& instances);

void add_device_conv2d_fwd_bias_relu_avx2_nhwc_kyxc_nhwk_local_c(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PassThrough, PassThrough, AddRelu>>& instances);

void add_device_conv2d_fwd_bias_relu_avx2_nhwc_kyxc_nhwk_mt(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PassThrough, PassThrough, AddRelu>>& instances);

void add_device_conv2d_fwd_bias_avx2_nhwc_kyxc_nhwk(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PassThrough, PassThrough, Add>>& instances);

void add_device_conv2d_fwd_bias_avx2_nhwc_kyxc_nhwk_local_c(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PassThrough, PassThrough, Add>>& instances);

void add_device_conv2d_fwd_bias_avx2_nhwc_kyxc_nhwk_mt(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PassThrough, PassThrough, Add>>& instances);

// ------------------ nhwc-kcyxk8-nhwk
void add_device_conv2d_fwd_bias_relu_add_avx2_nhwc_kyxck8_nhwk(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PassThrough, PassThrough, AddReluAdd>>&
        instances);

void add_device_conv2d_fwd_bias_relu_add_avx2_nhwc_kyxck8_nhwk_local_c(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PassThrough, PassThrough, AddReluAdd>>&
        instances);

void add_device_conv2d_fwd_bias_relu_add_avx2_nhwc_kyxck8_nhwk_mt(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PassThrough, PassThrough, AddReluAdd>>&
        instances);

void add_device_conv2d_fwd_bias_relu_avx2_nhwc_kyxck8_nhwk(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PassThrough, PassThrough, AddRelu>>& instances);

void add_device_conv2d_fwd_bias_relu_avx2_nhwc_kyxck8_nhwk_local_c(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PassThrough, PassThrough, AddRelu>>& instances);

void add_device_conv2d_fwd_bias_relu_avx2_nhwc_kyxck8_nhwk_mt(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PassThrough, PassThrough, AddRelu>>& instances);

void add_device_conv2d_fwd_bias_avx2_nhwc_kyxck8_nhwk(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PassThrough, PassThrough, Add>>& instances);

void add_device_conv2d_fwd_bias_avx2_nhwc_kyxck8_nhwk_local_c(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PassThrough, PassThrough, Add>>& instances);

void add_device_conv2d_fwd_bias_avx2_nhwc_kyxck8_nhwk_mt(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PassThrough, PassThrough, Add>>& instances);

// ------------------ nhwc-yxck-nhwk
void add_device_conv2d_fwd_bias_relu_add_avx2_nhwc_yxck_nhwk(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PassThrough, PassThrough, AddReluAdd>>&
        instances);

void add_device_conv2d_fwd_bias_relu_add_avx2_nhwc_yxck_nhwk_local_c(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PassThrough, PassThrough, AddReluAdd>>&
        instances);

void add_device_conv2d_fwd_bias_relu_add_avx2_nhwc_yxck_nhwk_mt(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PassThrough, PassThrough, AddReluAdd>>&
        instances);

void add_device_conv2d_fwd_bias_relu_avx2_nhwc_yxck_nhwk(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PassThrough, PassThrough, AddRelu>>& instances);

void add_device_conv2d_fwd_bias_relu_avx2_nhwc_yxck_nhwk_local_c(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PassThrough, PassThrough, AddRelu>>& instances);

void add_device_conv2d_fwd_bias_relu_avx2_nhwc_yxck_nhwk_mt(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PassThrough, PassThrough, AddRelu>>& instances);

void add_device_conv2d_fwd_bias_avx2_nhwc_yxck_nhwk(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PassThrough, PassThrough, Add>>& instances);

void add_device_conv2d_fwd_bias_avx2_nhwc_yxck_nhwk_local_c(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PassThrough, PassThrough, Add>>& instances);

void add_device_conv2d_fwd_bias_avx2_nhwc_yxck_nhwk_mt(
    std::vector<DeviceConvFwdBiasActivationAddPtr<PassThrough, PassThrough, Add>>& instances);

} // namespace device_conv2d_fwd_bias_activation_add_avx2_instance
} // namespace device
} // namespace cpu
} // namespace tensor_operation
} // namespace ck

using InElementOp  = ck::tensor_operation::cpu::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::cpu::element_wise::PassThrough;
#if TEST_FUSION == TEST_FUSION_BIAS_RELU_ADD
using OutElementOp = ck::tensor_operation::cpu::element_wise::AddReluAdd;
#elif TEST_FUSION == TEST_FUSION_BIAS_RELU
using OutElementOp = ck::tensor_operation::cpu::element_wise::AddRelu;
#elif TEST_FUSION == TEST_FUSION_BIAS
using OutElementOp = ck::tensor_operation::cpu::element_wise::Add;
#endif

template <typename T>
static bool
check_out(const T* ref, const T* result, std::size_t len, double nrms, int per_pixel_check = 0)
{
    std::size_t error_count = 0;
    float max_diff          = 1e-5;

    double square_difference = .0;
    double mag1              = .0;
    double mag2              = .0;

    for(std::size_t i = 0; i < len; ++i)
    {
        double ri = (double)ref[i];
        double pi = (double)result[i];
        double d  = ri - pi;

        if(per_pixel_check)
        {
            if(max_diff < std::abs(d))
            {
                error_count++;
                printf(
                    "idx:%3d, ref:%f, res:%f (diff:%f)\n", i, double(ref[i]), double(result[i]), d);
            }
        }

        square_difference += d * d;
        if(std::abs(mag1) < std::abs(ri))
            mag1 = ri;
        if(std::abs(mag2) < std::abs(pi))
            mag2 = pi;
    }

    double mag = std::max({std::fabs(mag1), std::fabs(mag2), std::numeric_limits<double>::min()});
    double computed_nrms = std::sqrt(square_difference) / (std::sqrt(len) * mag);

    if(computed_nrms >= nrms)
        printf("nrms:%lf, mag1:%lf, mag2:%lf, expected_nrms is %1f\n",
               computed_nrms,
               mag1,
               mag2,
               nrms);

    return computed_nrms < nrms && error_count == 0;
}

float calculate_gflops() {}

template <typename T>
void transpose_kyxc_2_kyxc8k(Tensor<T>& dst,
                             const Tensor<T>& src,
                             ck::index_t K,
                             ck::index_t Y,
                             ck::index_t X,
                             ck::index_t C)
{
    ck::index_t batch = K / 8;
    ck::index_t row   = 8;
    ck::index_t col   = C * Y * X;
    for(auto i_b = 0; i_b < batch; i_b++)
    {
        for(auto i_r = 0; i_r < row; i_r++)
        {
            for(auto i_c = 0; i_c < col; i_c++)
            {
                ck::index_t src_idx = i_b * row * col + i_r * col + i_c;
                ck::index_t dst_idx = i_b * col * row + i_c * row + i_r;
                dst.mData[dst_idx]  = src.mData[src_idx];
            }
        }
    }
}

template <typename T>
void transpose_kyxc_2_yxck(Tensor<T>& dst,
                           const Tensor<T>& src,
                           ck::index_t K,
                           ck::index_t Y,
                           ck::index_t X,
                           ck::index_t C)
{
    ck::index_t batch = 1;
    ck::index_t row   = K;
    ck::index_t col   = C * Y * X;
    for(auto i_b = 0; i_b < batch; i_b++)
    {
        for(auto i_r = 0; i_r < row; i_r++)
        {
            for(auto i_c = 0; i_c < col; i_c++)
            {
                ck::index_t src_idx = i_b * row * col + i_r * col + i_c;
                ck::index_t dst_idx = i_b * col * row + i_c * row + i_r;
                dst.mData[dst_idx]  = src.mData[src_idx];
            }
        }
    }
}

int main(int argc, char* argv[])
{
    int data_type   = 0;
    int init_method = 0;

    // Conv shape
    ck::index_t N               = 2;
    ck::index_t K               = 256;
    ck::index_t C               = 192;
    ck::index_t Y               = 3;
    ck::index_t X               = 3;
    ck::index_t Hi              = 71;
    ck::index_t Wi              = 71;
    ck::index_t conv_stride_h   = 1;
    ck::index_t conv_stride_w   = 1;
    ck::index_t conv_dilation_h = 1;
    ck::index_t conv_dilation_w = 1;
    ck::index_t in_left_pad_h   = 1;
    ck::index_t in_left_pad_w   = 1;
    ck::index_t in_right_pad_h  = 1;
    ck::index_t in_right_pad_w  = 1;

    if(ck::getenv_int("CK_USE_XDNN_DESC", 0) == 1)
    {
        assert(argc == 4 || argc == 5);
        data_type   = std::stoi(argv[1]);
        init_method = std::stoi(argv[2]);

        ck::desc_t xdnn_desc;

        if(str2desc(&xdnn_desc, argv[argc - 1]) == XDNN_OK)
        {
            N               = xdnn_desc.mb;
            K               = xdnn_desc.oc;
            C               = xdnn_desc.ic;
            Y               = xdnn_desc.kh;
            X               = xdnn_desc.kw;
            Hi              = xdnn_desc.ih;
            Wi              = xdnn_desc.iw;
            conv_stride_h   = xdnn_desc.sh;
            conv_stride_w   = xdnn_desc.sw;
            conv_dilation_h = xdnn_desc.dh;
            conv_dilation_w = xdnn_desc.dw;
            in_left_pad_h   = xdnn_desc.ph;
            in_left_pad_w   = xdnn_desc.pw;
            in_right_pad_h  = xdnn_desc.ph;
            in_right_pad_w  = xdnn_desc.pw;
        }
        else
        {
            printf("fail to parse xdnn arg:%s\n", argv[argc - 1]);
            exit(1);
        }
        if(argc == 5)
            N = std::stoi(argv[3]);
    }
    else
    {
        if(argc == 1)
        {
            data_type   = 0;
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
            printf("arg1: data type (0=fp32, 1=fp16)\n");
            printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
            printf("arg3 to 17: N, K, C, Y, X, Hi, Wi, Sy, Sx, Dy, Dx, LeftPy, LeftPx, RightPy, "
                   "RightPx\n");
            exit(1);
        }
    }

    auto Run = [&](auto input_type, auto wei_type, auto out_type) {
        using InDataType  = decltype(input_type);
        using WeiDataType = decltype(wei_type);
        using OutDataType = decltype(out_type);

#if TEST_FUSION == TEST_FUSION_BIAS_RELU_ADD
        using ReferenceConvFwdInstance =
            ck::tensor_operation::host::ReferenceConvFwd_Bias_Activation_Add<InDataType,
                                                                             WeiDataType,
                                                                             OutDataType,
                                                                             InElementOp,
                                                                             WeiElementOp,
                                                                             OutElementOp>;
#else
        using ReferenceConvFwdInstance =
            ck::tensor_operation::host::ReferenceConvFwd_Bias_Activation<InDataType,
                                                                         WeiDataType,
                                                                         OutDataType,
                                                                         InElementOp,
                                                                         WeiElementOp,
                                                                         OutElementOp>;
#endif

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

        auto f_host_tensor_descriptor = [](std::size_t N_,
                                           std::size_t C_,
                                           std::size_t H_,
                                           std::size_t W_) {
            return HostTensorDescriptor(std::vector<std::size_t>({N_, C_, H_, W_}),
                                        std::vector<std::size_t>({C_ * H_ * W_, 1, W_ * C_, C_}));
        };

        Tensor<InDataType> in_n_c_hi_wi(f_host_tensor_descriptor(N, C, Hi, Wi));
        Tensor<WeiDataType> wei_k_c_y_x(f_host_tensor_descriptor(K, C, Y, X));
#if TEST_LAYOUT == TEST_LAYOUT_NHWC_KYXCK8_NHWK
        Tensor<WeiDataType> wei_k_c_y_x_k8(
            f_host_tensor_descriptor(K, C, Y, X)); // TODO: This is only to hold data
#endif
#if TEST_LAYOUT == TEST_LAYOUT_NHWC_YXCK_NHWK
        Tensor<WeiDataType> wei_y_x_c_k(
            f_host_tensor_descriptor(K, C, Y, X)); // TODO: This is only to hold data
#endif
        Tensor<OutDataType> out_n_k_ho_wo_host_result(f_host_tensor_descriptor(N, K, Ho, Wo));

        // bias: assume contiguous 1d vector
        Tensor<OutDataType> bias(
            HostTensorDescriptor(std::vector<std::size_t>({static_cast<std::size_t>(K)})));

        // residual: assume same layout as output tensor
        Tensor<OutDataType> residual(f_host_tensor_descriptor(N, K, Ho, Wo));

        std::cout << "in (N, C, Hi, Wi): " << in_n_c_hi_wi.mDesc << std::endl;
        std::cout << "wei(K, C,  Y,  X): " << wei_k_c_y_x.mDesc << std::endl;
        std::cout << "out(N, K, Ho, Wo): " << out_n_k_ho_wo_host_result.mDesc << std::endl;
        std::cout << "bias: " << bias.mDesc << std::endl;
        std::cout << "residual: " << residual.mDesc << std::endl;
        std::cout << "LPad(H, W):" << in_left_pad_h << "," << in_left_pad_w
                  << ", RPad(H, W):" << in_right_pad_h << "," << in_right_pad_w
                  << ", Stride(H, W):" << conv_stride_h << ", " << conv_stride_w
                  << ", Dilation(H, W):" << conv_dilation_h << ", " << conv_dilation_w
                  << ", Threads:" << omp_get_max_threads() << std::endl;

        fflush(stdout);

        int per_pixel_check = 0;
        switch(init_method)
        {
        case 0:
            in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_1<InDataType>{});
            wei_k_c_y_x.GenerateTensorValue(GeneratorTensor_1<WeiDataType>{});
            bias.GenerateTensorValue(GeneratorTensor_1<WeiDataType>{});
            residual.GenerateTensorValue(GeneratorTensor_1<WeiDataType>{});
            per_pixel_check = 1;
            break;
        case 1:

            in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
            // in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_1<InDataType>{});
            wei_k_c_y_x.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
            // wei_k_c_y_x.GenerateTensorValue(GeneratorTensor_1<WeiDataType>{});
            bias.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
            residual.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
            per_pixel_check = 1;
            break;

        case 2:
            in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
            wei_k_c_y_x.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
            bias.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
            residual.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
            break;
        default:
            in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_3<InDataType>{0, 1});
            wei_k_c_y_x.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-1, 1});
            bias.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
            residual.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
        }

        DeviceAlignedMemCPU in_device_buf(sizeof(InDataType) * in_n_c_hi_wi.mDesc.GetElementSpace(),
                                          AVX2_DATA_ALIGNMENT);
        DeviceAlignedMemCPU wei_device_buf(
            sizeof(WeiDataType) * wei_k_c_y_x.mDesc.GetElementSpace(), AVX2_DATA_ALIGNMENT);
        DeviceAlignedMemCPU out_device_buf(sizeof(OutDataType) *
                                               out_n_k_ho_wo_host_result.mDesc.GetElementSpace(),
                                           AVX2_DATA_ALIGNMENT);

        DeviceAlignedMemCPU bias_device_buf(sizeof(OutDataType) * bias.mDesc.GetElementSpace(),
                                            AVX2_DATA_ALIGNMENT);
        DeviceAlignedMemCPU resi_device_buf(sizeof(OutDataType) * residual.mDesc.GetElementSpace(),
                                            AVX2_DATA_ALIGNMENT);

        in_device_buf.ToDevice(in_n_c_hi_wi.mData.data());
#if TEST_LAYOUT == TEST_LAYOUT_NHWC_KYXC_NHWK
        wei_device_buf.ToDevice(wei_k_c_y_x.mData.data());
#endif
#if TEST_LAYOUT == TEST_LAYOUT_NHWC_KYXCK8_NHWK
        transpose_kyxc_2_kyxc8k(wei_k_c_y_x_k8, wei_k_c_y_x, K, Y, X, C);
        wei_device_buf.ToDevice(wei_k_c_y_x_k8.mData.data());
#endif
#if TEST_LAYOUT == TEST_LAYOUT_NHWC_YXCK_NHWK
        transpose_kyxc_2_yxck(wei_y_x_c_k, wei_k_c_y_x, K, Y, X, C);
        wei_device_buf.ToDevice(wei_y_x_c_k.mData.data());
#endif
        bias_device_buf.ToDevice(bias.mData.data());
        resi_device_buf.ToDevice(residual.mData.data());

        // get host result
        int cpu_validation = ck::getenv_int("CK_CPU_VALIDATION", 1);
        if(cpu_validation)
        {
            auto ref_conv    = ReferenceConvFwdInstance{};
            auto ref_invoker = ref_conv.MakeInvoker();

            auto ref_argument = ref_conv.MakeArgument(in_n_c_hi_wi,
                                                      wei_k_c_y_x,
                                                      out_n_k_ho_wo_host_result,
                                                      bias,
#if TEST_FUSION == TEST_FUSION_BIAS_RELU_ADD
                                                      residual,
#endif
                                                      conv_filter_strides,
                                                      conv_filter_dilations,
                                                      input_left_pads,
                                                      input_right_pads,
                                                      InElementOp{},
                                                      WeiElementOp{},
                                                      OutElementOp{});
            ref_invoker.Run(ref_argument);
        }

        using PassThrough = ck::tensor_operation::cpu::element_wise::PassThrough;
        using AddReluAdd  = ck::tensor_operation::cpu::element_wise::AddReluAdd;
        using AddRelu     = ck::tensor_operation::cpu::element_wise::AddRelu;
        using Add         = ck::tensor_operation::cpu::element_wise::Add;

#if TEST_FUSION == TEST_FUSION_BIAS_RELU_ADD
        using DeviceConvFwdNoOpPtr = ck::tensor_operation::cpu::device::
            DeviceConvFwdBiasActivationAddPtr<PassThrough, PassThrough, AddReluAdd>;
#elif TEST_FUSION == TEST_FUSION_BIAS_RELU
        using DeviceConvFwdNoOpPtr = ck::tensor_operation::cpu::device::
            DeviceConvFwdBiasActivationAddPtr<PassThrough, PassThrough, AddRelu>;
#elif TEST_FUSION == TEST_FUSION_BIAS
        using DeviceConvFwdNoOpPtr = ck::tensor_operation::cpu::device::
            DeviceConvFwdBiasActivationAddPtr<PassThrough, PassThrough, Add>;
#endif

        // add device Conv instances
        std::vector<DeviceConvFwdNoOpPtr> conv_ptrs;

        if constexpr(ck::is_same_v<ck::remove_cv_t<InDataType>, float> &&
                     ck::is_same_v<ck::remove_cv_t<WeiDataType>, float> &&
                     ck::is_same_v<ck::remove_cv_t<OutDataType>, float>)
        {
#if TEST_FUSION == TEST_FUSION_BIAS_RELU_ADD
#if TEST_LAYOUT == TEST_LAYOUT_NHWC_KYXC_NHWK
            if(omp_get_max_threads() > 1)
            {
                ck::tensor_operation::cpu::device::
                    device_conv2d_fwd_bias_activation_add_avx2_instance::
                        add_device_conv2d_fwd_bias_relu_add_avx2_nhwc_kyxc_nhwk_mt(conv_ptrs);
                ck::tensor_operation::cpu::device::
                    device_conv2d_fwd_bias_activation_add_avx2_instance::
                        add_device_conv2d_fwd_bias_relu_add_avx2_nhwc_kyxc_nhwk(conv_ptrs);
            }
            else
            {
                if(K % 8 == 0)
                    ck::tensor_operation::cpu::device::
                        device_conv2d_fwd_bias_activation_add_avx2_instance::
                            add_device_conv2d_fwd_bias_relu_add_avx2_nhwc_kyxc_nhwk(conv_ptrs);
                else
                    ck::tensor_operation::cpu::device::
                        device_conv2d_fwd_bias_activation_add_avx2_instance::
                            add_device_conv2d_fwd_bias_relu_add_avx2_nhwc_kyxc_nhwk_local_c(
                                conv_ptrs);
            }
#endif
#if TEST_LAYOUT == TEST_LAYOUT_NHWC_KYXCK8_NHWK
            if(omp_get_max_threads() > 1)
            {
                ck::tensor_operation::cpu::device::
                    device_conv2d_fwd_bias_activation_add_avx2_instance::
                        add_device_conv2d_fwd_bias_relu_add_avx2_nhwc_kyxck8_nhwk_mt(conv_ptrs);
                ck::tensor_operation::cpu::device::
                    device_conv2d_fwd_bias_activation_add_avx2_instance::
                        add_device_conv2d_fwd_bias_relu_add_avx2_nhwc_kyxck8_nhwk(conv_ptrs);
            }
            else
            {
                if(K % 8 == 0)
                    ck::tensor_operation::cpu::device::
                        device_conv2d_fwd_bias_activation_add_avx2_instance::
                            add_device_conv2d_fwd_bias_relu_add_avx2_nhwc_kyxck8_nhwk(conv_ptrs);
                else
                    ck::tensor_operation::cpu::device::
                        device_conv2d_fwd_bias_activation_add_avx2_instance::
                            add_device_conv2d_fwd_bias_relu_add_avx2_nhwc_kyxck8_nhwk_local_c(
                                conv_ptrs);
            }
#endif
#if TEST_LAYOUT == TEST_LAYOUT_NHWC_YXCK_NHWK
            if(omp_get_max_threads() > 1)
            {
                ck::tensor_operation::cpu::device::
                    device_conv2d_fwd_bias_activation_add_avx2_instance::
                        add_device_conv2d_fwd_bias_relu_add_avx2_nhwc_yxck_nhwk_mt(conv_ptrs);
                ck::tensor_operation::cpu::device::
                    device_conv2d_fwd_bias_activation_add_avx2_instance::
                        add_device_conv2d_fwd_bias_relu_add_avx2_nhwc_yxck_nhwk(conv_ptrs);
            }
            else
            {
                if(K % 8 == 0)
                    ck::tensor_operation::cpu::device::
                        device_conv2d_fwd_bias_activation_add_avx2_instance::
                            add_device_conv2d_fwd_bias_relu_add_avx2_nhwc_yxck_nhwk(conv_ptrs);
                else
                    ck::tensor_operation::cpu::device::
                        device_conv2d_fwd_bias_activation_add_avx2_instance::
                            add_device_conv2d_fwd_bias_relu_add_avx2_nhwc_yxck_nhwk_local_c(
                                conv_ptrs);
            }
#endif
#elif TEST_FUSION == TEST_FUSION_BIAS_RELU
#if TEST_LAYOUT == TEST_LAYOUT_NHWC_KYXC_NHWK
            if(omp_get_max_threads() > 1)
            {
                ck::tensor_operation::cpu::device::
                    device_conv2d_fwd_bias_activation_add_avx2_instance::
                        add_device_conv2d_fwd_bias_relu_avx2_nhwc_kyxc_nhwk_mt(conv_ptrs);
                ck::tensor_operation::cpu::device::
                    device_conv2d_fwd_bias_activation_add_avx2_instance::
                        add_device_conv2d_fwd_bias_relu_avx2_nhwc_kyxc_nhwk(conv_ptrs);
            }
            else
            {
                if(K % 8 == 0)
                    ck::tensor_operation::cpu::device::
                        device_conv2d_fwd_bias_activation_add_avx2_instance::
                            add_device_conv2d_fwd_bias_relu_avx2_nhwc_kyxc_nhwk(conv_ptrs);
                else
                    ck::tensor_operation::cpu::device::
                        device_conv2d_fwd_bias_activation_add_avx2_instance::
                            add_device_conv2d_fwd_bias_relu_avx2_nhwc_kyxc_nhwk_local_c(conv_ptrs);
            }
#endif
#if TEST_LAYOUT == TEST_LAYOUT_NHWC_KYXCK8_NHWK
            if(omp_get_max_threads() > 1)
            {
                ck::tensor_operation::cpu::device::
                    device_conv2d_fwd_bias_activation_add_avx2_instance::
                        add_device_conv2d_fwd_bias_relu_avx2_nhwc_kyxck8_nhwk_mt(conv_ptrs);
                ck::tensor_operation::cpu::device::
                    device_conv2d_fwd_bias_activation_add_avx2_instance::
                        add_device_conv2d_fwd_bias_relu_avx2_nhwc_kyxck8_nhwk(conv_ptrs);
            }
            else
            {
                if(K % 8 == 0)
                    ck::tensor_operation::cpu::device::
                        device_conv2d_fwd_bias_activation_add_avx2_instance::
                            add_device_conv2d_fwd_bias_relu_avx2_nhwc_kyxck8_nhwk(conv_ptrs);
                else
                    ck::tensor_operation::cpu::device::
                        device_conv2d_fwd_bias_activation_add_avx2_instance::
                            add_device_conv2d_fwd_bias_relu_avx2_nhwc_kyxck8_nhwk_local_c(
                                conv_ptrs);
            }
#endif
#if TEST_LAYOUT == TEST_LAYOUT_NHWC_YXCK_NHWK
            if(omp_get_max_threads() > 1)
            {
                ck::tensor_operation::cpu::device::
                    device_conv2d_fwd_bias_activation_add_avx2_instance::
                        add_device_conv2d_fwd_bias_relu_avx2_nhwc_yxck_nhwk_mt(conv_ptrs);
                ck::tensor_operation::cpu::device::
                    device_conv2d_fwd_bias_activation_add_avx2_instance::
                        add_device_conv2d_fwd_bias_relu_avx2_nhwc_yxck_nhwk(conv_ptrs);
            }
            else
            {
                if(K % 8 == 0)
                    ck::tensor_operation::cpu::device::
                        device_conv2d_fwd_bias_activation_add_avx2_instance::
                            add_device_conv2d_fwd_bias_relu_avx2_nhwc_yxck_nhwk(conv_ptrs);
                else
                    ck::tensor_operation::cpu::device::
                        device_conv2d_fwd_bias_activation_add_avx2_instance::
                            add_device_conv2d_fwd_bias_relu_avx2_nhwc_yxck_nhwk_local_c(conv_ptrs);
            }
#endif
#elif TEST_FUSION == TEST_FUSION_BIAS
#if TEST_LAYOUT == TEST_LAYOUT_NHWC_KYXC_NHWK
            if(omp_get_max_threads() > 1)
            {
                ck::tensor_operation::cpu::device::
                    device_conv2d_fwd_bias_activation_add_avx2_instance::
                        add_device_conv2d_fwd_bias_avx2_nhwc_kyxc_nhwk_mt(conv_ptrs);
                ck::tensor_operation::cpu::device::
                    device_conv2d_fwd_bias_activation_add_avx2_instance::
                        add_device_conv2d_fwd_bias_avx2_nhwc_kyxc_nhwk(conv_ptrs);
            }
            else
            {
                if(K % 8 == 0)
                    ck::tensor_operation::cpu::device::
                        device_conv2d_fwd_bias_activation_add_avx2_instance::
                            add_device_conv2d_fwd_bias_avx2_nhwc_kyxc_nhwk(conv_ptrs);
                else
                    ck::tensor_operation::cpu::device::
                        device_conv2d_fwd_bias_activation_add_avx2_instance::
                            add_device_conv2d_fwd_bias_avx2_nhwc_kyxc_nhwk_local_c(conv_ptrs);
            }
#endif
#if TEST_LAYOUT == TEST_LAYOUT_NHWC_KYXCK8_NHWK
            if(omp_get_max_threads() > 1)
            {
                ck::tensor_operation::cpu::device::
                    device_conv2d_fwd_bias_activation_add_avx2_instance::
                        add_device_conv2d_fwd_bias_avx2_nhwc_kyxck8_nhwk_mt(conv_ptrs);
                ck::tensor_operation::cpu::device::
                    device_conv2d_fwd_bias_activation_add_avx2_instance::
                        add_device_conv2d_fwd_bias_avx2_nhwc_kyxck8_nhwk(conv_ptrs);
            }
            else
            {
                if(K % 8 == 0)
                    ck::tensor_operation::cpu::device::
                        device_conv2d_fwd_bias_activation_add_avx2_instance::
                            add_device_conv2d_fwd_bias_avx2_nhwc_kyxck8_nhwk(conv_ptrs);
                else
                    ck::tensor_operation::cpu::device::
                        device_conv2d_fwd_bias_activation_add_avx2_instance::
                            add_device_conv2d_fwd_bias_avx2_nhwc_kyxck8_nhwk_local_c(conv_ptrs);
            }
#endif
#if TEST_LAYOUT == TEST_LAYOUT_NHWC_YXCK_NHWK
            if(omp_get_max_threads() > 1)
            {
                ck::tensor_operation::cpu::device::
                    device_conv2d_fwd_bias_activation_add_avx2_instance::
                        add_device_conv2d_fwd_bias_avx2_nhwc_yxck_nhwk_mt(conv_ptrs);
                ck::tensor_operation::cpu::device::
                    device_conv2d_fwd_bias_activation_add_avx2_instance::
                        add_device_conv2d_fwd_bias_avx2_nhwc_yxck_nhwk(conv_ptrs);
            }
            else
            {
                if(K % 8 == 0)
                    ck::tensor_operation::cpu::device::
                        device_conv2d_fwd_bias_activation_add_avx2_instance::
                            add_device_conv2d_fwd_bias_avx2_nhwc_yxck_nhwk(conv_ptrs);
                else
                    ck::tensor_operation::cpu::device::
                        device_conv2d_fwd_bias_activation_add_avx2_instance::
                            add_device_conv2d_fwd_bias_avx2_nhwc_yxck_nhwk_local_c(conv_ptrs);
            }
#endif
#endif
        }

        if(conv_ptrs.size() <= 0)
        {
            throw std::runtime_error("wrong! no device Conv instance found");
        }

        // profile device Conv instances
        bool success                    = true;
        double fastest_kernel_time      = std::numeric_limits<double>::max();
        std::string fastest_kernel_name = "";
        double fastest_kernel_gflops    = 0;
        int loop                        = ck::getenv_int("CK_LOOP", 10);
        for(auto& conv_ptr : conv_ptrs)
        {
            auto argument_ptr = conv_ptr->MakeArgumentPointer(
                static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
                static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                static_cast<const OutDataType*>(bias_device_buf.GetDeviceBuffer()),
                static_cast<const OutDataType*>(resi_device_buf.GetDeviceBuffer()),
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
                double time      = invoker_ptr->Run(argument_ptr.get(), StreamConfig{}, loop);

                double total_flop = static_cast<double>(2) * N * C * Ho * Wo * K * Y * X;

                double gflops = (total_flop * 1e-6) / time;

                if(cpu_validation &&
                   !check_out(out_n_k_ho_wo_host_result.mData.data(),
                              reinterpret_cast<OutDataType*>(out_device_buf.mpDeviceBuf),
                              out_n_k_ho_wo_host_result.mData.size(),
                              1e-6,
                              per_pixel_check))
                {
                    std::cout << "Fail Info: " << conv_ptr->GetTypeString() << std::endl;
                    success = false;
                }
                else
                {
                    std::cout << (cpu_validation ? "Pass" : "Ignore")
                              << " Info: " << conv_ptr->GetTypeString() << ", Time:" << time
                              << "ms, Gflops:" << gflops << std::endl;

                    if(time < fastest_kernel_time)
                    {
                        fastest_kernel_time   = time;
                        fastest_kernel_name   = conv_ptr->GetTypeString();
                        fastest_kernel_gflops = gflops;
                    }
                }
            }
            else
            {
                std::cout << "Not support Info: " << conv_ptr->GetTypeString() << std::endl;
            }
        }

        if(fastest_kernel_time != std::numeric_limits<double>::max())
        {
            double total_flop = static_cast<double>(2) * N * C * Ho * Wo * K * Y * X;
            std::cout << "fastest: " << fastest_kernel_name;
            std::cout << ", total_gflop:" << ((double)total_flop / 1.0e9)
                      << ", time(ms):" << fastest_kernel_time
                      << ", Gflops:" << fastest_kernel_gflops << std::endl;
        }
        return 0;
        // if(success)
        // {
        //     std::cout << "test conv2d fwd cpu : Pass" << std::endl;
        //     return 0;
        // }
        // else
        // {
        //     std::cout << "test conv2d fwd cpu: Fail " << std::endl;
        //     return -1;
        // }
    };

    if(data_type == 0)
    {
        return Run(F32(), F32(), F32());
    }
    else
    {
        return 1;
    }
}
