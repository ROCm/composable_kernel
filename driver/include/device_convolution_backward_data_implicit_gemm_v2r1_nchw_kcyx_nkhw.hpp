#pragma once
#include <unistd.h>
#include "device.hpp"
#include "tensor.hpp"
#include "gridwise_operation_wrapper.hpp"
#include "gridwise_convolution_backward_data_implicit_gemm_v2r1_nchw_kcyx_nkhw.hpp"

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
void device_convolution_backward_data_implicit_gemm_v2r1_nchw_kcyx_nkhw(InDesc in_nchw_desc,
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
    using namespace ck;

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

    std::size_t data_sz = sizeof(T);
    DeviceMem in_nchw_device_buf(data_sz * in_nchw.mDesc.GetElementSpace());
    DeviceMem wei_kcyx_device_buf(data_sz * wei_kcyx.mDesc.GetElementSpace());
    DeviceMem out_nkhw_device_buf(data_sz * out_nkhw.mDesc.GetElementSpace());

    in_nchw_device_buf.ToDevice(in_nchw.mData.data());
    wei_kcyx_device_buf.ToDevice(wei_kcyx.mData.data());
    out_nkhw_device_buf.ToDevice(out_nkhw.mData.data());

#if 1
    // BlockSize = 256, each thread hold 64 data
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock              = 128;
    constexpr index_t GemmNPerBlock              = 128;
    constexpr index_t GemmKPerBlock              = 8;
    constexpr index_t GemmMPerThreadSubC         = 4;
    constexpr index_t GemmNPerThreadSubC         = 4;
    constexpr index_t GemmMLevel0Cluster         = 4;
    constexpr index_t GemmNLevel0Cluster         = 4;
    constexpr index_t GemmMLevel1Cluster         = 4;
    constexpr index_t GemmNLevel1Cluster         = 4;
    constexpr index_t GemmKPerThreadLoop         = 1;
    constexpr index_t GemmThreadGemmDataPerReadM = 4;
    constexpr index_t GemmThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<4, 1>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<2, 128>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmM  = 1;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<4, 1>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<2, 128>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 1;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 1;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 1;
#elif 1
    // BlockSize = 256, each thread hold 64 data
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock              = 128;
    constexpr index_t GemmNPerBlock              = 128;
    constexpr index_t GemmKPerBlock              = 8;
    constexpr index_t GemmMPerThreadSubC         = 4;
    constexpr index_t GemmNPerThreadSubC         = 4;
    constexpr index_t GemmMLevel0Cluster         = 4;
    constexpr index_t GemmNLevel0Cluster         = 4;
    constexpr index_t GemmMLevel1Cluster         = 4;
    constexpr index_t GemmNLevel1Cluster         = 4;
    constexpr index_t GemmKPerThreadLoop         = 1;
    constexpr index_t GemmThreadGemmDataPerReadM = 4;
    constexpr index_t GemmThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<1, 4>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<8, 32>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmM  = 1;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<1, 4>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<8, 32>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 1;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 1;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 1;
#elif 1
    // BlockSize = 256, each thread hold 64 data
    // for 1x1 weight, 8x8 input
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock              = 128;
    constexpr index_t GemmNPerBlock              = 128;
    constexpr index_t GemmKPerBlock              = 8;
    constexpr index_t GemmMPerThreadSubC         = 4;
    constexpr index_t GemmNPerThreadSubC         = 4;
    constexpr index_t GemmMLevel0Cluster         = 4;
    constexpr index_t GemmNLevel0Cluster         = 4;
    constexpr index_t GemmMLevel1Cluster         = 4;
    constexpr index_t GemmNLevel1Cluster         = 4;
    constexpr index_t GemmKPerThreadLoop         = 1;
    constexpr index_t GemmThreadGemmDataPerReadM = 4;
    constexpr index_t GemmThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<1, 4>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<8, 32>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmM  = 4;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 4;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<1, 4>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<8, 32>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 4;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 4;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 4;
#endif

    constexpr index_t gcd_stride_dilation_h = math::gcd(ConvStrideH, ConvDilationH);
    constexpr index_t gcd_stride_dilation_w = math::gcd(ConvStrideW, ConvDilationW);

    constexpr index_t Ytilda = ConvStrideH / gcd_stride_dilation_h;
    constexpr index_t Xtilda = ConvStrideW / gcd_stride_dilation_w;

    constexpr index_t Ydot = math::integer_divide_ceil(Y, Ytilda);
    constexpr index_t Xdot = math::integer_divide_ceil(X, Xtilda);

    constexpr index_t Htilda = Ho + math::integer_divide_ceil(ConvDilationH * (Y - 1), ConvStrideH);
    constexpr index_t Wtilda = Wo + math::integer_divide_ceil(ConvDilationW * (X - 1), ConvStrideW);

    constexpr index_t HtildaLeft = math::integer_divide_floor(
        math::max(0, InLeftPads{}[0] - ConvDilationH * (Ytilda - 1)), ConvStrides{}[0]);
    constexpr index_t WtildaLeft = math::integer_divide_floor(
        math::max(0, InLeftPads{}[1] - ConvDilationW * (Xtilda - 1)), ConvStrides{}[1]);

    constexpr index_t HtildaRight = math::min(
        Htilda, math::integer_divide_ceil(InLeftPads{}[0] + Hi - 1, ConvStrides{}[0]) + 1);
    constexpr index_t WtildaRight = math::min(
        Wtilda, math::integer_divide_ceil(InLeftPads{}[1] + Wi - 1, ConvStrides{}[1]) + 1);

    constexpr index_t HtildaTrim = HtildaRight - HtildaLeft;
    constexpr index_t WtildaTrim = WtildaRight - WtildaLeft;

    constexpr index_t GemmM = C * Ytilda * Xtilda;
    constexpr index_t GemmN = N * HtildaTrim * WtildaTrim;

    constexpr index_t GridSize = math::integer_divide_ceil(GemmM, GemmMPerBlock) *
                                 math::integer_divide_ceil(GemmN, GemmNPerBlock);

    printf("%s: BlockSize %u, GridSize %u \n", __func__, BlockSize, GridSize);

    constexpr auto gridwise_conv = GridwiseConvolutionBackwardDataImplicitGemm_v2r1_nchw_kcyx_nkhw<
        GridSize,
        BlockSize,
        T,
        T,
        decltype(in_nchw_desc),
        decltype(wei_kcyx_desc),
        decltype(out_nkhw_desc),
        ConvStrides,
        ConvDilations,
        InLeftPads,
        InRightPads,
        GemmMPerBlock,
        GemmNPerBlock,
        GemmKPerBlock,
        GemmMPerThreadSubC,
        GemmNPerThreadSubC,
        GemmMLevel0Cluster,
        GemmNLevel0Cluster,
        GemmMLevel1Cluster,
        GemmNLevel1Cluster,
        GemmKPerThreadLoop,
        GemmThreadGemmDataPerReadM,
        GemmThreadGemmDataPerReadN,
        GemmABlockCopyThreadSliceLengths_GemmK_GemmM,
        GemmABlockCopyThreadClusterLengths_GemmK_GemmM,
        GemmABlockCopySrcDataPerRead_GemmM,
        GemmABlockCopyDstDataPerWrite_GemmM,
        GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
        GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
        GemmBBlockCopySrcDataPerRead_GemmN,
        GemmBBlockCopyDstDataPerWrite_GemmN,
        GemmCThreadCopyDstDataPerWrite_GemmN1>{};

    for(index_t i = 0; i < nrepeat; ++i)
    {
        float time = launch_and_time_kernel(run_gridwise_operation<decltype(gridwise_conv),
                                                                   T* const __restrict__,
                                                                   const T* const __restrict__,
                                                                   const T* const __restrict__>,
                                            dim3(GridSize),
                                            dim3(BlockSize),
                                            0,
                                            0,
                                            gridwise_conv,
                                            static_cast<T*>(in_nchw_device_buf.GetDeviceBuffer()),
                                            static_cast<T*>(wei_kcyx_device_buf.GetDeviceBuffer()),
                                            static_cast<T*>(out_nkhw_device_buf.GetDeviceBuffer()));

        printf("Elapsed time : %f ms, %f TFlop/s\n",
               time,
               (float)calculate_convolution_flops(InDesc{}, WeiDesc{}, OutDesc{}) /
                   (std::size_t(1000) * 1000 * 1000) / time);
        usleep(std::min(time * 1000, float(10000)));
    }

    in_nchw_device_buf.FromDevice(in_nchw.mData.data());
}

} // namespace launcher
