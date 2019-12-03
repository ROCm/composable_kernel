#pragma once
#include <unistd.h>
#include "device.hpp"
#include "tensor.hpp"
#include "gridwise_operation_wrapper.hpp"
#include "gridwise_convolution_backward_data_implicit_gemm_v2r1_nchw_kcyx_nkhw.hpp"

template <typename T,
          typename InDesc,
          typename WeiDesc,
          typename OutDesc,
          typename ConvStrides,
          typename ConvDilations,
          typename LeftPads,
          typename RightPads>
void device_convolution_backward_data_implicit_gemm_v2r1_nchw_kcyx_nkhw(InDesc in_nchw_desc,
                                                                        Tensor<T>& in_nchw,
                                                                        WeiDesc wei_kcyx_desc,
                                                                        const Tensor<T>& wei_kcyx,
                                                                        OutDesc out_nkhw_desc,
                                                                        const Tensor<T>& out_nkhw,
                                                                        ConvStrides,
                                                                        ConvDilations,
                                                                        LeftPads,
                                                                        RightPads,
                                                                        std::size_t nrepeat)
{
    using namespace ck;

    constexpr index_t N  = out_nkhw_desc.GetLengths()[0];
    constexpr index_t K  = out_nkhw_desc.GetLengths()[1];
    constexpr index_t Ho = out_nkhw_desc.GetLengths()[2];
    constexpr index_t Wo = out_nkhw_desc.GetLengths()[3];

    constexpr index_t C = wei_kcyx_desc.GetLengths()[1];
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

    using GemmABlockCopySubLengths     = Sequence<4, 1>;   // Gemm-K, Gemm-M
    using GemmABlockCopyClusterLengths = Sequence<2, 128>; // Gemm-K, Gemm-M

    constexpr index_t GemmABlockCopyDataPerAccess = 1; // Gemm-M

    using GemmBBlockCopySubLengths     = Sequence<4, 1>;   // Gemm-K, Gemm-N
    using GemmBBlockCopyClusterLengths = Sequence<2, 128>; // Gemm-K, Gemm-N

    constexpr index_t GemmBBlockCopyDataPerAccess = 1; // Gemm-N

    constexpr index_t GemmCThreadCopyDataPerAccess = 1; // Gemm-N
#elif 0
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

    using GemmABlockCopySubLengths     = Sequence<1, 4>;  // Gemm-K, Gemm-M
    using GemmABlockCopyClusterLengths = Sequence<8, 32>; // Gemm-K, Gemm-M

    constexpr index_t GemmABlockCopyDataPerAccess = 4; // Gemm-M

    using GemmBBlockCopySubLengths     = Sequence<4, 1>;   // Gemm-K, Gemm-N
    using GemmBBlockCopyClusterLengths = Sequence<2, 128>; // Gemm-K, Gemm-N

    constexpr index_t GemmBBlockCopyDataPerAccess = 1; // Gemm-N

    constexpr index_t GemmCThreadCopyDataPerAccess = 1; // Gemm-N
#endif

    // TODO: this algo support any stride and dilation. But for now, let's fix them to be 1 for
    // simplicity
    constexpr index_t hcf_stride_dilation_h = math::hcf(ConvStrideH, ConvDilationH);
    constexpr index_t hcf_stride_dilation_w = math::hcf(ConvStrideW, ConvDilationW);

    constexpr index_t Ytilda = ConvStrideH / hcf_stride_dilation_h; // may be wrong
    constexpr index_t Xtilda = ConvStrideW / hcf_stride_dilation_w; // may be wrong

    constexpr index_t Ydot = math::integer_divide_ceil(Y, Ytilda);
    constexpr index_t Xdot = math::integer_divide_ceil(X, Xtilda);

    constexpr index_t right_pad_ho = (ConvDilationH / hcf_stride_dilation_h) * (Y - Ytilda);
    constexpr index_t right_pad_wo = (ConvDilationW / hcf_stride_dilation_w) * (X - Xtilda);

    constexpr index_t Htilda = Ho + right_pad_ho;
    constexpr index_t Wtilda = Wo + right_pad_wo;

    constexpr index_t GemmK = K * Ydot * Xdot;
    constexpr index_t GemmM = C * Ytilda * Xtilda;
    constexpr index_t GemmN = N * Htilda * Wtilda;

    constexpr index_t GridSize = ((GemmM + GemmMPerBlock - 1) / GemmMPerBlock) *
                                 ((GemmN + GemmNPerBlock - 1) / GemmNPerBlock);

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
        LeftPads,
        RightPads,
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
        GemmABlockCopySubLengths,
        GemmABlockCopyClusterLengths,
        GemmABlockCopyDataPerAccess,
        GemmBBlockCopySubLengths,
        GemmBBlockCopyClusterLengths,
        GemmBBlockCopyDataPerAccess,
        GemmCThreadCopyDataPerAccess>{};

    for(index_t i = 0; i < nrepeat; ++i)
    {
        float time = launch_kernel(run_gridwise_operation<decltype(gridwise_conv),
                                                          T* const __restrict__,
                                                          const T* const __restrict__,
                                                          const T* const __restrict__>,
                                   dim3(GridSize),
                                   dim3(BlockSize),
                                   0,
                                   gridwise_conv,
                                   const_cast<T* const __restrict__>(
                                       static_cast<T*>(in_nchw_device_buf.GetDeviceBuffer())),
                                   const_cast<const T* const __restrict__>(
                                       static_cast<T*>(wei_kcyx_device_buf.GetDeviceBuffer())),
                                   const_cast<const T* const __restrict__>(
                                       static_cast<T*>(out_nkhw_device_buf.GetDeviceBuffer())));

        printf("Elapsed time : %f ms, %f TFlop/s\n",
               time,
               (float)calculate_convolution_flops(InDesc{}, WeiDesc{}, OutDesc{}) /
                   (std::size_t(1000) * 1000 * 1000) / time);
        usleep(std::min(time * 1000, float(10000)));
    }

    in_nchw_device_buf.FromDevice(in_nchw.mData.data());
}
