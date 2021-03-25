#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "gridwise_operation_wrapper.hpp"
#include "gridwise_convolution_forward_implicit_gemm_v4r4_nhwc_kyxc_nhwk.hpp"

template <class T,
          class InDesc,
          class WeiDesc,
          class OutDesc,
          class ConvStrides,
          class ConvDilations,
          class InLeftPads,
          class InRightPads>
void device_convolution_forward_implicit_gemm_v4r4_nhwc_kyxc_nhwk(InDesc,
                                                                  const Tensor<T>& in_nchw,
                                                                  WeiDesc,
                                                                  const Tensor<T>& wei_kcyx,
                                                                  OutDesc,
                                                                  Tensor<T>& out_nkhw,
                                                                  ConvStrides,
                                                                  ConvDilations,
                                                                  InLeftPads,
                                                                  InRightPads,
                                                                  ck::index_t nrepeat)
{
    std::cout << "device_convolution_forward_implicit_gemm_v4r4_nhwc_kyxc_nhwk" << std::endl;

    using namespace ck;

    using TDevice = typename conditional<is_same<half_float::half, T>::value, half_t, T>::type;

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

    // compile-time variables
    constexpr auto in_n_hi_wi_c_desc =
        make_native_tensor_descriptor_packed(Sequence<N, Hi, Wi, C>{});
    constexpr auto wei_k_y_x_c_desc = make_native_tensor_descriptor_packed(Sequence<K, Y, X, C>{});
    constexpr auto out_n_ho_wo_k_desc =
        make_native_tensor_descriptor_packed(Sequence<N, Ho, Wo, K>{});

    Tensor<float> in_nhwc(
        make_HostTensorDescriptor(make_native_tensor_descriptor_packed(Sequence<N, Hi, Wi, C>{})));
    Tensor<float> wei_kyxc(
        make_HostTensorDescriptor(make_native_tensor_descriptor_packed(Sequence<K, Y, X, C>{})));
    Tensor<float> out_nhwk(
        make_HostTensorDescriptor(make_native_tensor_descriptor_packed(Sequence<N, Ho, Wo, K>{})));

    auto f_nchw2nhwc = [&](auto n, auto hi, auto wi, auto c) {
        in_nhwc(n, hi, wi, c) = in_nchw(n, c, hi, wi);
    };

    auto f_kcyx2kyxc = [&](auto k, auto y, auto x, auto c) {
        wei_kyxc(k, y, x, c) = wei_kcyx(k, c, y, x);
    };

    auto f_nkhw2nhwk = [&](auto n, auto ho, auto wo, auto k) {
        out_nhwk(n, ho, wo, k) = out_nkhw(n, k, ho, wo);
    };

    make_ParallelTensorFunctor(f_nchw2nhwc, N, Hi, Wi, C)(std::thread::hardware_concurrency());
    make_ParallelTensorFunctor(f_kcyx2kyxc, K, Y, X, C)(std::thread::hardware_concurrency());
    make_ParallelTensorFunctor(f_nkhw2nhwk, N, Ho, Wo, K)(std::thread::hardware_concurrency());

    std::size_t data_sz = sizeof(T);

    DeviceMem in_nhwc_device_buf(data_sz * in_nhwc.mDesc.GetElementSpace());
    DeviceMem wei_kyxc_device_buf(data_sz * wei_kyxc.mDesc.GetElementSpace());
    DeviceMem out_nhwk_device_buf(data_sz * out_nhwk.mDesc.GetElementSpace());

    in_nhwc_device_buf.ToDevice(in_nhwc.mData.data());
    wei_kyxc_device_buf.ToDevice(wei_kyxc.mData.data());
    out_nhwk_device_buf.ToDevice(out_nhwk.mData.data());

#if 1
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

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<1, 1>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<4, 16>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 1;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<4, 1>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<1, 64>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmK  = 4;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 1;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmM1 = 2;
#endif

    constexpr index_t GemmM = K;
    constexpr index_t GemmN = N * Ho * Wo;

    constexpr index_t GridSize = math::integer_divide_ceil(GemmM, GemmMPerBlock) *
                                 math::integer_divide_ceil(GemmN, GemmNPerBlock);

    printf("%s: BlockSize %u, GridSize %u \n", __func__, BlockSize, GridSize);

    using gridwise_conv = GridwiseConvolutionForwardImplicitGemm_v4r4_nhwc_kyxc_nhwk<
        GridSize,
        BlockSize,
        TDevice,
        TDevice,
        decltype(in_n_hi_wi_c_desc),
        decltype(wei_k_y_x_c_desc),
        decltype(out_n_ho_wo_k_desc),
        ConvStrides,
        ConvDilations,
        InLeftPads,
        InRightPads,
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
        ThreadGemmDataPerReadM,
        ThreadGemmDataPerReadN,
        GemmABlockCopyThreadSliceLengths_GemmK_GemmM,
        GemmABlockCopyThreadClusterLengths_GemmK_GemmM,
        GemmABlockCopySrcDataPerRead_GemmK,
        GemmABlockCopyDstDataPerWrite_GemmM,
        GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
        GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
        GemmBBlockCopySrcDataPerRead_GemmK,
        GemmBBlockCopyDstDataPerWrite_GemmN,
        GemmCThreadCopyDstDataPerWrite_GemmM1>;

    for(index_t i = 0; i < 5; ++i)
    {
        std::cout << "Start running " << nrepeat << " times..." << std::endl;

        KernelTimer timer;
        timer.Start();

        for(index_t j = 0; j < nrepeat; ++j)
        {
            launch_kernel(run_gridwise_operation<gridwise_conv,
                                                 const TDevice* const __restrict__,
                                                 const TDevice* const __restrict__,
                                                 TDevice* const __restrict__>,
                          dim3(GridSize),
                          dim3(BlockSize),
                          0,
                          0,
                          static_cast<TDevice*>(in_nhwc_device_buf.GetDeviceBuffer()),
                          static_cast<TDevice*>(wei_kyxc_device_buf.GetDeviceBuffer()),
                          static_cast<TDevice*>(out_nhwk_device_buf.GetDeviceBuffer()));
        }

        timer.End();

        float ave_time = timer.GetElapsedTime() / nrepeat;

        float perf = (float)calculate_convolution_flops(InDesc{}, WeiDesc{}, OutDesc{}) /
                     (std::size_t(1000) * 1000 * 1000) / ave_time;

        std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s" << std::endl;
    }

    out_nhwk_device_buf.FromDevice(out_nhwk.mData.data());

    auto f_nhwk2nkhw = [&](auto n, auto k, auto ho, auto wo) {
        out_nkhw(n, k, ho, wo) = out_nhwk(n, ho, wo, k);
    };

    make_ParallelTensorFunctor(f_nhwk2nkhw, N, K, Ho, Wo)(std::thread::hardware_concurrency());
}
