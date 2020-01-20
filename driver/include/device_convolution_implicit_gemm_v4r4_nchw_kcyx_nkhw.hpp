#pragma once
#include <unistd.h>
#include "device.hpp"
#include "tensor.hpp"
#include "gridwise_convolution_kernel_wrapper.hpp"
#include "gridwise_convolution_implicit_gemm_v4r4_nchw_kcyx_nkhw.hpp"

template <class T,
          class InDesc,
          class WeiDesc,
          class OutDesc,
          class ConvStrides,
          class ConvDilations,
          class InLeftPads,
          class InRightPads>
void device_convolution_implicit_gemm_v4r4_nchw_kcyx_nkhw(InDesc,
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
    using namespace ck;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_nchw_desc =
        make_native_tensor_descriptor(InDesc::GetLengths(), InDesc::GetStrides());
    constexpr auto wei_kcyx_desc =
        make_native_tensor_descriptor(WeiDesc::GetLengths(), WeiDesc::GetStrides());
    constexpr auto out_nkhw_desc =
        make_native_tensor_descriptor(OutDesc::GetLengths(), OutDesc::GetStrides());

    constexpr index_t N  = out_nkhw_desc.GetLength(I0);
    constexpr index_t K  = out_nkhw_desc.GetLength(I1);
    constexpr index_t Ho = out_nkhw_desc.GetLength(I2);
    constexpr index_t Wo = out_nkhw_desc.GetLength(I3);

    std::size_t data_sz = sizeof(T);
    DeviceMem in_nchw_device_buf(data_sz * in_nchw.mDesc.GetElementSpace());
    DeviceMem wei_kcyx_device_buf(data_sz * wei_kcyx.mDesc.GetElementSpace());
    DeviceMem out_nkhw_device_buf(data_sz * out_nkhw.mDesc.GetElementSpace());

    in_nchw_device_buf.ToDevice(in_nchw.mData.data());
    wei_kcyx_device_buf.ToDevice(wei_kcyx.mData.data());
    out_nkhw_device_buf.ToDevice(out_nkhw.mData.data());

#if 0
    // BlockSize = 256, GemmKPerBlock = 8
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 8;

    constexpr index_t GemmMPerThreadSubC     = 4;
    constexpr index_t GemmNPerThreadSubC     = 4;
    constexpr index_t GemmMLevel0Cluster     = 4;
    constexpr index_t GemmNLevel0Cluster     = 4;
    constexpr index_t GemmMLevel1Cluster     = 4;
    constexpr index_t GemmNLevel1Cluster     = 4;
    constexpr index_t GemmKPerThreadLoop     = 1;
    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<4, 1>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<2, 128>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 4;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<4, 1>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<2, 128>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 1;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 1;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 1;
#elif 1
    // BlockSize = 256, GemmKPerBlock = 16
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 16;

    constexpr index_t GemmMPerThreadSubC     = 4;
    constexpr index_t GemmNPerThreadSubC     = 4;
    constexpr index_t GemmMLevel0Cluster     = 4;
    constexpr index_t GemmNLevel0Cluster     = 4;
    constexpr index_t GemmMLevel1Cluster     = 4;
    constexpr index_t GemmNLevel1Cluster     = 4;
    constexpr index_t GemmKPerThreadLoop     = 1;
    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<4, 2>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<4, 64>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 4;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<4, 2>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<4, 64>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 1;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 1;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 1;
#elif 0
    // BlockSize = 256, GemmKPerBlock = 8
    // 1x1 filter, 8x8 image
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 8;

    constexpr index_t GemmMPerThreadSubC     = 4;
    constexpr index_t GemmNPerThreadSubC     = 4;
    constexpr index_t GemmMLevel0Cluster     = 4;
    constexpr index_t GemmNLevel0Cluster     = 4;
    constexpr index_t GemmMLevel1Cluster     = 4;
    constexpr index_t GemmNLevel1Cluster     = 4;
    constexpr index_t GemmKPerThreadLoop     = 1;
    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<4, 1>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<2, 128>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 4;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<1, 4>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<8, 32>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 4;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 4;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 4;
#elif 1
    // BlockSize = 256, GemmKPerBlock = 16
    // 1x1 filter, 8x8 image
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 16;

    constexpr index_t GemmMPerThreadSubC     = 4;
    constexpr index_t GemmNPerThreadSubC     = 4;
    constexpr index_t GemmMLevel0Cluster     = 4;
    constexpr index_t GemmNLevel0Cluster     = 4;
    constexpr index_t GemmMLevel1Cluster     = 4;
    constexpr index_t GemmNLevel1Cluster     = 4;
    constexpr index_t GemmKPerThreadLoop     = 1;
    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<4, 2>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<4, 64>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 4;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<2, 4>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<8, 32>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 4;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 4;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 4;
#elif 1
    // 1x1 filter, 14x14 image
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 8;

    constexpr index_t GemmMPerThreadSubC     = 4;
    constexpr index_t GemmNPerThreadSubC     = 4;
    constexpr index_t GemmMLevel0Cluster     = 4;
    constexpr index_t GemmNLevel0Cluster     = 4;
    constexpr index_t GemmMLevel1Cluster     = 4;
    constexpr index_t GemmNLevel1Cluster     = 4;
    constexpr index_t GemmKPerThreadLoop     = 1;
    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<4, 1>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<2, 128>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 4;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<2, 2>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<4, 64>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 2;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 2;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 2;
#endif

    constexpr index_t GemmM = K;
    constexpr index_t GemmN = N * Ho * Wo;

    constexpr index_t GridSize = math::integer_divide_ceil(GemmM, GemmMPerBlock) *
                                 math::integer_divide_ceil(GemmN, GemmNPerBlock);

    printf("%s: BlockSize %u, GridSize %u \n", __func__, BlockSize, GridSize);

    constexpr auto gridwise_conv = GridwiseConvolutionImplicitGemm_v4r4_nchw_kcyx_nkhw<
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
        ThreadGemmDataPerReadM,
        ThreadGemmDataPerReadN,
        GemmABlockCopyThreadSliceLengths_GemmK_GemmM,
        GemmABlockCopyThreadClusterLengths_GemmK_GemmM,
        GemmABlockCopySrcDataPerRead_GemmK,
        GemmABlockCopyDstDataPerWrite_GemmM,
        GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
        GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
        GemmBBlockCopySrcDataPerRead_GemmN,
        GemmBBlockCopyDstDataPerWrite_GemmN,
        GemmCThreadCopyDstDataPerWrite_GemmN1>{};

    for(index_t i = 0; i < nrepeat; ++i)
    {
        float time =
            launch_and_time_kernel(run_gridwise_convolution_kernel<decltype(gridwise_conv), T>,
                                   dim3(GridSize),
                                   dim3(BlockSize),
                                   0,
                                   0,
                                   static_cast<T*>(in_nchw_device_buf.GetDeviceBuffer()),
                                   static_cast<T*>(wei_kcyx_device_buf.GetDeviceBuffer()),
                                   static_cast<T*>(out_nkhw_device_buf.GetDeviceBuffer()));

        printf("Elapsed time : %f ms, %f TFlop/s\n",
               time,
               (float)calculate_convolution_flops(InDesc{}, WeiDesc{}, OutDesc{}) /
                   (std::size_t(1000) * 1000 * 1000) / time);
        usleep(std::min(time * 1000, float(10000)));
    }

    out_nkhw_device_buf.FromDevice(out_nkhw.mData.data());
}
