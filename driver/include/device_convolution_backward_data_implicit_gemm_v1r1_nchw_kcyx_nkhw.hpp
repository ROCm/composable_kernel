#pragma once
#include <unistd.h>
#include "device.hpp"
#include "tensor.hpp"
#include "gridwise_operation_wrapper.hpp"
#include "gridwise_convolution_backward_data_implicit_gemm_v1r1_nchw_kcyx_nkhw.hpp"

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
void device_convolution_backward_data_implicit_gemm_v1r1_nchw_kcyx_nkhw(InDesc in_nchw_desc,
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

    constexpr index_t N  = out_nkhw_desc.GetLengths()[0];
    constexpr index_t K  = out_nkhw_desc.GetLengths()[1];
    constexpr index_t Ho = out_nkhw_desc.GetLengths()[2];
    constexpr index_t Wo = out_nkhw_desc.GetLengths()[3];

    constexpr index_t C = wei_kcyx_desc.GetLengths()[1];
    constexpr index_t Y = wei_kcyx_desc.GetLengths()[2];
    constexpr index_t X = wei_kcyx_desc.GetLengths()[3];

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

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<1, 4>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<8, 32>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmM  = 4;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 4;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<4, 1>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<2, 128>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 1;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 1;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 1;
#endif

    constexpr index_t GemmM = C * Y * X;
    constexpr index_t GemmN = N * Ho * Wo;

    constexpr index_t GridSize = math::integer_divide_ceil(GemmM, GemmMPerBlock) *
                                 math::integer_divide_ceil(GemmN, GemmNPerBlock);

    printf("%s: BlockSize %u, GridSize %u \n", __func__, BlockSize, GridSize);

    constexpr auto gridwise_conv = GridwiseConvolutionBackwardDataImplicitGemm_v1r1_nchw_kcyx_nkhw<
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
