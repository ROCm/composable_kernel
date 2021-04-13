#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "driver_dynamic_convolution_forward_implicit_gemm_v4r4_nhwc_kyxc_nhwk.hpp"

template <class TInWei,
          ck::index_t InWeiVectorSize,
          class TAcc,
          class TOut,
          class InDesc,
          class WeiDesc,
          class OutDesc,
          class ConvStrides,
          class ConvDilations,
          class InLeftPads,
          class InRightPads>
void device_dynamic_convolution_forward_implicit_gemm_v4r4_nhwc_kyxc_nhwk(
    InDesc,
    const Tensor<TInWei>& in_n_c_hi_wi,
    WeiDesc,
    const Tensor<TInWei>& wei_k_c_y_x,
    OutDesc,
    Tensor<TOut>& out_n_k_ho_wo,
    ConvStrides,
    ConvDilations,
    InLeftPads,
    InRightPads,
    ck::index_t nrepeat)
{
    using namespace ck;

    std::cout << "device_dynamic_convolution_forward_implicit_gemm_v4r4_nhwc_kyxc_nhwk"
              << std::endl;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto N = OutDesc::GetLengths()[I0];
    constexpr auto K = OutDesc::GetLengths()[I1];
    constexpr auto C = WeiDesc::GetLengths()[I1];

    constexpr auto Hi = InDesc::GetLengths()[I2];
    constexpr auto Wi = InDesc::GetLengths()[I3];

    constexpr auto Ho = OutDesc::GetLengths()[I2];
    constexpr auto Wo = OutDesc::GetLengths()[I3];

    constexpr auto Y = WeiDesc::GetLengths()[I2];
    constexpr auto X = WeiDesc::GetLengths()[I3];

    constexpr auto C0 = C / Number<InWeiVectorSize>{};
    constexpr auto C1 = Number<InWeiVectorSize>{};

#if 1
    // run-time variables
    constexpr auto in_n_hi_wi_c0_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_multi_index(N, Hi, Wi, C0));
    constexpr auto wei_k_y_x_c0_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_multi_index(K, Y, X, C0));
    constexpr auto out_n_ho_wo_k_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_multi_index(N, Ho, Wo, K));

    const auto conv_strides   = to_multi_index(ConvStrides{});
    const auto conv_dilations = to_multi_index(ConvDilations{});
    const auto in_left_pads   = to_multi_index(InLeftPads{});
    const auto in_right_pads  = to_multi_index(InRightPads{});
#else
    // compile-time variables
    constexpr auto in_n_hi_wi_c0_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(N, Hi, Wi, C0));
    constexpr auto wei_k_y_x_c0_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(K, Y, X, C0));
    constexpr auto out_n_ho_wo_k_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(N, Ho, Wo, K));

    const auto conv_strides   = sequence_to_tuple_of_number(ConvStrides{});
    const auto conv_dilations = sequence_to_tuple_of_number(ConvDilations{});
    const auto in_left_pads   = sequence_to_tuple_of_number(InLeftPads{});
    const auto in_right_pads  = sequence_to_tuple_of_number(InRightPads{});
#endif

    Tensor<TInWei> in_n_hi_wi_c(
        make_HostTensorDescriptor(make_native_tensor_descriptor_packed(Sequence<N, Hi, Wi, C>{})));
    Tensor<TInWei> wei_k_y_x_c(
        make_HostTensorDescriptor(make_native_tensor_descriptor_packed(Sequence<K, Y, X, C>{})));
    Tensor<TOut> out_n_ho_wo_k(
        make_HostTensorDescriptor(make_native_tensor_descriptor_packed(Sequence<N, Ho, Wo, K>{})));

    auto f_nchw2nhwc = [&](auto n, auto hi, auto wi, auto c) {
        in_n_hi_wi_c(n, hi, wi, c) = in_n_c_hi_wi(n, c, hi, wi);
    };

    auto f_kcyx2kyxc = [&](auto k, auto y, auto x, auto c) {
        wei_k_y_x_c(k, y, x, c) = wei_k_c_y_x(k, c, y, x);
    };

    auto f_nkhw2nhwk = [&](auto n, auto ho, auto wo, auto k) {
        out_n_ho_wo_k(n, ho, wo, k) = out_n_k_ho_wo(n, k, ho, wo);
    };

    make_ParallelTensorFunctor(f_nchw2nhwc, N, Hi, Wi, C)();
    make_ParallelTensorFunctor(f_kcyx2kyxc, K, Y, X, C)();
    make_ParallelTensorFunctor(f_nkhw2nhwk, N, Ho, Wo, K)();

    DeviceMem in_n_hi_wi_c_device_buf(sizeof(TInWei) * in_n_hi_wi_c.mDesc.GetElementSpace());
    DeviceMem wei_k_y_x_c_device_buf(sizeof(TInWei) * wei_k_y_x_c.mDesc.GetElementSpace());
    DeviceMem out_n_ho_wo_k_device_buf(sizeof(TOut) * out_n_ho_wo_k.mDesc.GetElementSpace());

    in_n_hi_wi_c_device_buf.ToDevice(in_n_hi_wi_c.mData.data());
    wei_k_y_x_c_device_buf.ToDevice(wei_k_y_x_c.mData.data());
    out_n_ho_wo_k_device_buf.ToDevice(out_n_ho_wo_k.mData.data());

#if 0
    // cdata = 16, BlockSize = 64, 16x64x4
    constexpr index_t BlockSize = 64;

    constexpr index_t GemmMPerBlock = 16;
    constexpr index_t GemmNPerBlock = 64;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerThread = 2;
    constexpr index_t GemmNPerThread = 2;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 8;

    constexpr index_t ThreadGemmDataPerReadM = 2;
    constexpr index_t ThreadGemmDataPerReadN = 2;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<1, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<4, 16>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<4, 1>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<1, 64>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK = 4;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmM1 = 2;
#elif 0
    // cdata = 32, BlockSize = 64, 16x128x4
    constexpr index_t BlockSize = 64;

    constexpr index_t GemmMPerBlock = 16;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerThread = 2;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 8;

    constexpr index_t ThreadGemmDataPerReadM = 2;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<1, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<4, 16>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<4, 2>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<1, 64>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK = 4;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmM1 = 2;
#elif 0
    // cdata = 64, BlockSize = 64, 16x256x2
    constexpr index_t BlockSize = 64;

    constexpr index_t GemmMPerBlock = 16;
    constexpr index_t GemmNPerBlock = 256;
    constexpr index_t GemmKPerBlock = 2;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 1;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 16;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<1, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<2, 16>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<2, 4>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<1, 64>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK = 2;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmM1 = 4;
#elif 0
    // cdata = 64, BlockSize = 64, 16x256x4
    constexpr index_t BlockSize = 64;

    constexpr index_t GemmMPerBlock = 16;
    constexpr index_t GemmNPerBlock = 256;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 1;
    constexpr index_t GemmNLevel1Cluster = 16;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<1, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<4, 16>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<4, 4>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<1, 64>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK = 4;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmM1 = 4;
#elif 0
    // cdata = 64, BlockSize = 128, 32x256x4
    constexpr index_t BlockSize = 128;

    constexpr index_t GemmMPerBlock = 32;
    constexpr index_t GemmNPerBlock = 256;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 16;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<1, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<4, 32>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<4, 2>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<1, 128>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK = 4;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmM1 = 4;
#elif 0
    // cdata = 64, BlockSize = 128, 32x256x8
    constexpr index_t BlockSize = 128;

    constexpr index_t GemmMPerBlock = 32;
    constexpr index_t GemmNPerBlock = 256;
    constexpr index_t GemmKPerBlock = 8;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 16;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<2, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<4, 32>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 2;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<8, 2>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<1, 128>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK = 8;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmM1 = 4;
#elif 1
    // cdata = 64, BlockSize = 256, 128x128x8
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 8;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 8;
    constexpr index_t GemmNLevel1Cluster = 8;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<4, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<2, 128>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 4;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<4, 1>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<2, 128>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK = 4;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmM1 = 4;
#elif 1
    // cdata = 64, BlockSize = 256, 128x128x16
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 16;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 8;
    constexpr index_t GemmNLevel1Cluster = 8;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<4, 2>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<4, 64>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 4;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 2;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<8, 1>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<2, 128>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK = 8;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmM1 = 4;
#endif

    constexpr auto conv_driver =
#if 1
        DriverDynamicConvolutionForwardImplicitGemm_v4r4_nhwc_kyxc_nhwk_pad
#elif 0
        DriverDynamicConvolutionForwardImplicitGemm_v4r4_nhwc_kyxc_nhwk_no_pad
#elif 1
        DriverDynamicConvolutionForwardImplicitGemm_v4r4_nhwc_kyxc_nhwk_1x1
#endif
        <BlockSize,
         typename vector_type<TInWei, InWeiVectorSize>::type,
         TAcc,
         TOut,
         GemmMPerBlock,
         GemmNPerBlock,
         GemmKPerBlock,
         GemmMPerThread,
         GemmNPerThread,
         GemmKPerThread,
         GemmMLevel0Cluster,
         GemmNLevel0Cluster,
         GemmMLevel1Cluster,
         GemmNLevel1Cluster,
         GemmABlockTransferThreadSliceLengths_GemmK_GemmM,
         GemmABlockTransferThreadClusterLengths_GemmK_GemmM,
         GemmABlockTransferSrcScalarPerVector_GemmK,
         GemmABlockTransferDstScalarPerVector_GemmM,
         GemmBBlockTransferThreadSliceLengths_GemmK_GemmN,
         GemmBBlockTransferThreadClusterLengths_GemmK_GemmN,
         GemmBBlockTransferSrcScalarPerVector_GemmK,
         GemmBBlockTransferDstScalarPerVector_GemmN,
         GemmCThreadTransferDstScalarPerVector_GemmM1>{};

    conv_driver.Run(wei_k_y_x_c0_desc,
                    in_n_hi_wi_c0_desc,
                    out_n_ho_wo_k_desc,
                    conv_strides,
                    conv_dilations,
                    in_left_pads,
                    in_right_pads,
                    static_cast<typename vector_type<TInWei, InWeiVectorSize>::type*>(
                        wei_k_y_x_c_device_buf.GetDeviceBuffer()),
                    static_cast<typename vector_type<TInWei, InWeiVectorSize>::type*>(
                        in_n_hi_wi_c_device_buf.GetDeviceBuffer()),
                    static_cast<TOut*>(out_n_ho_wo_k_device_buf.GetDeviceBuffer()));

    out_n_ho_wo_k_device_buf.FromDevice(out_n_ho_wo_k.mData.data());

    auto f_nhwk2nkhw = [&](auto n, auto k, auto ho, auto wo) {
        out_n_k_ho_wo(n, k, ho, wo) = out_n_ho_wo_k(n, ho, wo, k);
    };

    make_ParallelTensorFunctor(f_nhwk2nkhw, N, K, Ho, Wo)();
}
