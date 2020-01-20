#pragma once
#include <unistd.h>
#include "device.hpp"
#include "tensor.hpp"
#include "gridwise_convolution_kernel_wrapper.hpp"
#include "gridwise_convolution_implicit_gemm_v4r2_nchw_kcyx_nkhw_lds_double_buffer.hpp"

using namespace ck;

template <class T,
          class InDesc,
          class WeiDesc,
          class OutDesc,
          class ConvStrides,
          class ConvDilations>
void device_convolution_implicit_gemm_v4r2_nchw_kcyx_nkhw(InDesc,
                                                          const Tensor<T>& in_nchw,
                                                          WeiDesc,
                                                          const Tensor<T>& wei_kcyx,
                                                          OutDesc,
                                                          Tensor<T>& out_nkhw,
                                                          ConvStrides,
                                                          ConvDilations,
                                                          index_t nrepeat)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_nchw_desc  = InDesc{};
    constexpr auto wei_kcyx_desc = WeiDesc{};
    constexpr auto out_nkhw_desc = OutDesc{};

    constexpr index_t Hi = in_nchw_desc.GetLength(I2);
    constexpr index_t Wi = in_nchw_desc.GetLength(I3);

    constexpr index_t N  = out_nkhw_desc.GetLength(I0);
    constexpr index_t Ho = out_nkhw_desc.GetLength(I2);
    constexpr index_t Wo = out_nkhw_desc.GetLength(I3);

    constexpr index_t K = wei_kcyx_desc.GetLength(I0);
    constexpr index_t C = wei_kcyx_desc.GetLength(I1);
    constexpr index_t Y = wei_kcyx_desc.GetLength(I2);
    constexpr index_t X = wei_kcyx_desc.GetLength(I3);

    std::size_t data_sz = sizeof(T);
    DeviceMem in_nchw_device_buf(data_sz * in_nchw.mDesc.GetElementSpace());
    DeviceMem wei_kcyx_device_buf(data_sz * wei_kcyx.mDesc.GetElementSpace());
    DeviceMem out_nkhw_device_buf(data_sz * out_nkhw.mDesc.GetElementSpace());

    in_nchw_device_buf.ToDevice(in_nchw.mData.data());
    wei_kcyx_device_buf.ToDevice(wei_kcyx.mData.data());
    out_nkhw_device_buf.ToDevice(out_nkhw.mData.data());

#if 0
    // 1x1 filter, 8x8 image
    constexpr index_t N0  = 1;
    constexpr index_t Ho0 = 2;
    constexpr index_t Wo0 = 1;

    constexpr index_t N2  = 4;
    constexpr index_t Ho2 = 1;
    constexpr index_t Wo2 = 1;

    constexpr index_t BlockSize = 256;

    constexpr index_t BPerBlock = 16;
    constexpr index_t KPerBlock = 128;
    constexpr index_t EPerBlock = 8;

    constexpr index_t GemmMPerThreadSubC = 4;
    constexpr index_t GemmNPerThreadSubC = 4;
    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 4;
    constexpr index_t GemmKPerThreadLoop = 1;
    constexpr index_t GemmDataPerReadA   = 4;
    constexpr index_t GemmDataPerReadB   = 4;

    using InBlockCopySubLengths_E_N0_Ho0_Wo0_B_N2_Ho2_Wo2     = Sequence<1, 1, 1, 1, 1, 4, 1, 1>;
    using InBlockCopyClusterLengths_E_N0_Ho0_Wo0_B_N2_Ho2_Wo2 = Sequence<8, 1, 2, 1, 16, 1, 1, 1>;
    using InBlockCopyThreadClusterArrangeOrder =
        Sequence<0, 1, 5, 2, 6, 3, 4, 7>; // [E, N0, N2, Ho0, Ho2, Wo0, B, Wo2]
    using InBlockCopySrcAccessOrder =
        Sequence<0, 1, 5, 2, 6, 3, 4, 7>; // [E, N0, N2, Ho0, Ho2, Wo0, B, Wo2]
    using InBlockCopyDstAccessOrder =
        Sequence<0, 1, 2, 3, 4, 5, 6, 7>; // [E, N0, Ho0, Wo0, B, N2, Ho2, Wo2]

    constexpr index_t InBlockCopyDataPerAccess_W2 = 1;

    using WeiBlockCopySubLengths_E_K            = Sequence<4, 1>;
    using WeiBlockCopyClusterLengths_E_K        = Sequence<2, 128>;
    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 4;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 1;
#elif 1
    // 1x1 filter, 8x8 image
    constexpr index_t N0  = 1;
    constexpr index_t Ho0 = 2;
    constexpr index_t Wo0 = 1;

    constexpr index_t N2  = 2;
    constexpr index_t Ho2 = 2;
    constexpr index_t Wo2 = 1;

    constexpr index_t BlockSize = 256;

    constexpr index_t BPerBlock = 16;
    constexpr index_t KPerBlock = 128;
    constexpr index_t EPerBlock = 8;

    constexpr index_t GemmMPerThreadSubC = 4;
    constexpr index_t GemmNPerThreadSubC = 4;
    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 4;
    constexpr index_t GemmKPerThreadLoop = 1;
    constexpr index_t GemmDataPerReadA   = 4;
    constexpr index_t GemmDataPerReadB   = 4;

    using InBlockCopySubLengths_E_N0_Ho0_Wo0_B_N2_Ho2_Wo2     = Sequence<1, 1, 2, 1, 1, 2, 1, 1>;
    using InBlockCopyClusterLengths_E_N0_Ho0_Wo0_B_N2_Ho2_Wo2 = Sequence<8, 1, 1, 1, 16, 1, 2, 1>;
    using InBlockCopyThreadClusterArrangeOrder =
        Sequence<0, 1, 5, 2, 6, 3, 4, 7>; // [E, N0, N2, Ho0, Ho2, Wo0, B, Wo2]
    using InBlockCopySrcAccessOrder =
        Sequence<0, 1, 5, 2, 6, 3, 4, 7>; // [E, N0, N2, Ho0, Ho2, Wo0, B, Wo2]
    using InBlockCopyDstAccessOrder =
        Sequence<0, 1, 2, 3, 4, 5, 6, 7>; // [E, N0, Ho0, Wo0, B, N2, Ho2, Wo2]

    constexpr index_t InBlockCopyDataPerAccess_W2 = 1;

    using WeiBlockCopySubLengths_E_K            = Sequence<4, 1>;
    using WeiBlockCopyClusterLengths_E_K        = Sequence<2, 128>;
    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 4;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 1;
#endif

    constexpr index_t N1  = N / (N0 * N2);
    constexpr index_t Ho1 = Ho / (Ho0 * Ho2);
    constexpr index_t Wo1 = Wo / (Wo0 * Wo2);

    constexpr index_t B = N1 * Ho1 * Wo1;

    constexpr index_t GridSize =
        ((B + BPerBlock - 1) / BPerBlock) * ((K + KPerBlock - 1) / KPerBlock);

    printf("%s: BlockSize %u, GridSize %u \n", __func__, BlockSize, GridSize);

    for(index_t i = 0; i < nrepeat; ++i)
    {
        constexpr auto gridwise_conv =
            GridwiseConvolutionImplicitGemm_v4r2_nchw_kcyx_nkhw_lds_double_buffer<
                GridSize,
                BlockSize,
                T,
                decltype(in_nchw_desc),
                decltype(wei_kcyx_desc),
                decltype(out_nkhw_desc),
                ConvStrides,
                ConvDilations,
                N1,
                N2,
                Ho1,
                Ho2,
                Wo1,
                Wo2,
                BPerBlock,
                KPerBlock,
                EPerBlock,
                GemmMPerThreadSubC,
                GemmNPerThreadSubC,
                GemmMLevel0Cluster,
                GemmNLevel0Cluster,
                GemmMLevel1Cluster,
                GemmNLevel1Cluster,
                GemmKPerThreadLoop,
                GemmDataPerReadA,
                GemmDataPerReadB,
                InBlockCopySubLengths_E_N0_Ho0_Wo0_B_N2_Ho2_Wo2,
                InBlockCopyClusterLengths_E_N0_Ho0_Wo0_B_N2_Ho2_Wo2,
                InBlockCopyThreadClusterArrangeOrder,
                InBlockCopySrcAccessOrder,
                InBlockCopyDstAccessOrder,
                InBlockCopyDataPerAccess_W2,
                WeiBlockCopySubLengths_E_K,
                WeiBlockCopyClusterLengths_E_K,
                WeiBlockCopyThreadClusterArrangeOrder,
                WeiBlockCopySrcAccessOrder,
                WeiBlockCopyDstAccessOrder,
                WeiBlockCopySrcDataPerRead_E,
                WeiBlockCopyDstDataPerWrite_K>{};

        float time =
            launch_and_time_kernel(run_gridwise_convolution_kernel<decltype(gridwise_conv), T>,
                                   dim3(GridSize),
                                   dim3(BlockSize),
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
