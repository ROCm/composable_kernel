#pragma once
#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "gridwise_operation_wrapper.hpp"
#include "gridwise_convolution_backward_data_implicit_gemm_v5r1_nhwc_kyxc_nhwk.hpp"

namespace launcher {

using namespace ck;

template <typename T,
          typename InDesc,
          typename WeiDesc,
          typename OutDesc,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
void device_convolution_backward_data_implicit_gemm_v5r1_nhwc_kyxc_nhwk(InDesc in_nchw_desc,
                                                                        Tensor<T>& in_nchw,
                                                                        WeiDesc wei_kcyx_desc,
                                                                        const Tensor<T>& wei_kcyx,
                                                                        OutDesc out_nkhw_desc,
                                                                        const Tensor<T>& out_nkhw,
                                                                        ConvStrides,
                                                                        ConvDilations,
                                                                        InLeftPads,
                                                                        InRightPads,
                                                                        std::size_t nrepeat)
{
    constexpr index_t N = out_nkhw_desc.GetLengths()[0];
    constexpr index_t K = out_nkhw_desc.GetLengths()[1];
    constexpr index_t C = wei_kcyx_desc.GetLengths()[1];

    constexpr index_t Hi = in_nchw_desc.GetLengths()[2];
    constexpr index_t Wi = in_nchw_desc.GetLengths()[3];

    constexpr index_t Ho = out_nkhw_desc.GetLengths()[2];
    constexpr index_t Wo = out_nkhw_desc.GetLengths()[3];

    constexpr index_t Y = wei_kcyx_desc.GetLengths()[2];
    constexpr index_t X = wei_kcyx_desc.GetLengths()[3];

    constexpr index_t ConvStrideH = ConvStrides{}[0];
    constexpr index_t ConvStrideW = ConvStrides{}[1];

    constexpr index_t ConvDilationH = ConvDilations{}[0];
    constexpr index_t ConvDilationW = ConvDilations{}[1];

    constexpr auto in_nhwc_desc  = make_native_tensor_descriptor_packed(Sequence<N, Hi, Wi, C>{});
    constexpr auto wei_kyxc_desc = make_native_tensor_descriptor_packed(Sequence<K, Y, X, C>{});
    constexpr auto out_nhwk_desc = make_native_tensor_descriptor_packed(Sequence<N, Ho, Wo, K>{});

    Tensor<float> in_nhwc(make_HostTensorDescriptor(in_nhwc_desc));
    Tensor<float> wei_kyxc(make_HostTensorDescriptor(wei_kyxc_desc));
    Tensor<float> out_nhwk(make_HostTensorDescriptor(out_nhwk_desc));

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

#if 0
    // cdata = 64, BlockSize = 256, 128x128x8
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock              = 128;
    constexpr index_t GemmNPerBlock              = 128;
    constexpr index_t GemmKPerBlock              = 8;
    constexpr index_t GemmMPerThread             = 4;
    constexpr index_t GemmNPerThread             = 4;
    constexpr index_t GemmKPerThread             = 1;
    constexpr index_t GemmMLevel0Cluster         = 4;
    constexpr index_t GemmNLevel0Cluster         = 4;
    constexpr index_t GemmMLevel1Cluster         = 4;
    constexpr index_t GemmNLevel1Cluster         = 4;
    constexpr index_t GemmThreadGemmDataPerReadM = 4;
    constexpr index_t GemmThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK0_GemmK1_GemmK2_GemmM   = Sequence<1, 1, 1, 4>;
    using GemmABlockCopyThreadClusterLengths_GemmK0_GemmK1_GemmK2_GemmM = Sequence<1, 1, 8, 32>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmM  = 4;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 4;

    using GemmBBlockCopyThreadSliceLengths_GemmK0_GemmK1_GemmK2_GemmN   = Sequence<1, 1, 4, 1>;
    using GemmBBlockCopyThreadClusterLengths_GemmK0_GemmK1_GemmK2_GemmN = Sequence<1, 1, 2, 128>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmK2 = 4;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 1;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 1;
#elif 1
    // cdata = 64, BlockSize = 256, 128x128x16
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock              = 128;
    constexpr index_t GemmNPerBlock              = 128;
    constexpr index_t GemmKPerBlock              = 16;
    constexpr index_t GemmMPerThread             = 4;
    constexpr index_t GemmNPerThread             = 4;
    constexpr index_t GemmKPerThread             = 1;
    constexpr index_t GemmMLevel0Cluster         = 4;
    constexpr index_t GemmNLevel0Cluster         = 4;
    constexpr index_t GemmMLevel1Cluster         = 4;
    constexpr index_t GemmNLevel1Cluster         = 4;
    constexpr index_t GemmThreadGemmDataPerReadM = 4;
    constexpr index_t GemmThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK0_GemmK1_GemmK2_GemmM   = Sequence<1, 1, 2, 4>;
    using GemmABlockCopyThreadClusterLengths_GemmK0_GemmK1_GemmK2_GemmM = Sequence<1, 1, 8, 32>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmM  = 4;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 4;

    using GemmBBlockCopyThreadSliceLengths_GemmK0_GemmK1_GemmK2_GemmN   = Sequence<1, 1, 8, 1>;
    using GemmBBlockCopyThreadClusterLengths_GemmK0_GemmK1_GemmK2_GemmN = Sequence<1, 1, 2, 128>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmK2 = 4;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 1;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 1;
#endif

    constexpr index_t GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
    constexpr index_t GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

    constexpr index_t YTilda = ConvStrideH / GcdStrideDilationH;
    constexpr index_t XTilda = ConvStrideW / GcdStrideDilationW;

    constexpr index_t YDot = math::integer_divide_ceil(Y, YTilda);
    constexpr index_t XDot = math::integer_divide_ceil(X, XTilda);

    constexpr index_t HTilda = Ho + math::integer_divide_ceil(ConvDilationH * (Y - 1), ConvStrideH);
    constexpr index_t WTilda = Wo + math::integer_divide_ceil(ConvDilationW * (X - 1), ConvStrideW);

    constexpr index_t HTildaLeft = math::integer_divide_floor(
        math::max(0, InLeftPads{}[0] - ConvDilationH * (YTilda - 1)), ConvStrides{}[0]);
    constexpr index_t WTildaLeft = math::integer_divide_floor(
        math::max(0, InLeftPads{}[1] - ConvDilationW * (XTilda - 1)), ConvStrides{}[1]);

    constexpr index_t HTildaRight = math::min(
        HTilda, math::integer_divide_ceil(InLeftPads{}[0] + Hi - 1, ConvStrides{}[0]) + 1);
    constexpr index_t WTildaRight = math::min(
        WTilda, math::integer_divide_ceil(InLeftPads{}[1] + Wi - 1, ConvStrides{}[1]) + 1);

    constexpr index_t HTildaSlice = HTildaRight - HTildaLeft;
    constexpr index_t WTildaSlice = WTildaRight - WTildaLeft;

    constexpr index_t GemmM = C;
    constexpr index_t GemmN = N * HTildaSlice * WTildaSlice;

    constexpr index_t GridSize = math::integer_divide_ceil(GemmM, GemmMPerBlock) *
                                 math::integer_divide_ceil(GemmN, GemmNPerBlock);

    printf("%s: BlockSize %u, GridSize %u \n", __func__, BlockSize, GridSize);

    for(index_t i = 0; i < 5; ++i)
    {
        std::cout << "Start running " << nrepeat << " times..." << std::endl;

        KernelTimer timer;
        timer.Start();

        for(index_t i = 0; i < nrepeat; ++i)
        {
            using GridwiseConvBwdData =
                GridwiseConvolutionBackwardDataImplicitGemm_v5r1_nhwc_kyxc_nhwk<
                    GridSize,
                    BlockSize,
                    T,
                    T,
                    decltype(in_nhwc_desc),
                    decltype(wei_kyxc_desc),
                    decltype(out_nhwk_desc),
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
                    GemmThreadGemmDataPerReadM,
                    GemmThreadGemmDataPerReadN,
                    GemmABlockCopyThreadSliceLengths_GemmK0_GemmK1_GemmK2_GemmM,
                    GemmABlockCopyThreadClusterLengths_GemmK0_GemmK1_GemmK2_GemmM,
                    GemmABlockCopySrcDataPerRead_GemmM,
                    GemmABlockCopyDstDataPerWrite_GemmM,
                    GemmBBlockCopyThreadSliceLengths_GemmK0_GemmK1_GemmK2_GemmN,
                    GemmBBlockCopyThreadClusterLengths_GemmK0_GemmK1_GemmK2_GemmN,
                    GemmBBlockCopySrcDataPerRead_GemmK2,
                    GemmBBlockCopyDstDataPerWrite_GemmN,
                    GemmCThreadCopyDstDataPerWrite_GemmN1>;

            static_for<0, GridwiseConvBwdData::GetNumberOfGemm(), 1>{}([&](auto gemm_id) {
                constexpr auto gemm_sizes        = GridwiseConvBwdData::GetGemmSize(gemm_id);
                constexpr index_t gemm_k2        = gemm_sizes[Number<4>{}];
                constexpr bool is_gemm_not_empty = gemm_k2 > 0;

                // only compile and run if GEMM is no empty
                static_if<is_gemm_not_empty>{}([&](auto fwd) {
                    launch_kernel(run_gridwise_operation<GridwiseConvBwdData,
                                                         T* const __restrict__,
                                                         const T* const __restrict__,
                                                         const T* const __restrict__,
                                                         decltype(gemm_id)>,
                                  dim3(GridSize),
                                  dim3(BlockSize),
                                  0,
                                  0,
                                  static_cast<T*>(in_nhwc_device_buf.GetDeviceBuffer()),
                                  static_cast<T*>(wei_kyxc_device_buf.GetDeviceBuffer()),
                                  static_cast<T*>(out_nhwk_device_buf.GetDeviceBuffer()),
                                  fwd(gemm_id));
                });
            });
        }

        timer.End();

        float ave_time = timer.GetElapsedTime() / nrepeat;

        float perf = (float)calculate_convolution_flops(InDesc{}, WeiDesc{}, OutDesc{}) /
                     (std::size_t(1000) * 1000 * 1000) / ave_time;

        std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s" << std::endl;
    }

    in_nhwc_device_buf.FromDevice(in_nhwc.mData.data());

    auto f_nhwc2nchw = [&](auto n, auto c, auto hi, auto wi) {
        in_nchw(n, c, hi, wi) = in_nhwc(n, hi, wi, c);
    };

    make_ParallelTensorFunctor(f_nhwc2nchw, N, C, Hi, Wi)(std::thread::hardware_concurrency());
}

} // namespace launcher
