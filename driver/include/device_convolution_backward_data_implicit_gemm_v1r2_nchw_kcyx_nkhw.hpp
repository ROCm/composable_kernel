#pragma once
#include <unistd.h>
#include "device.hpp"
#include "tensor.hpp"
#include "gridwise_operation_wrapper.hpp"
#include "gridwise_convolution_backward_data_implicit_gemm_v1r2_nchw_kcyx_nkhw_lds_double_buffer.hpp"

template <typename T,
          typename InDesc,
          typename WeiDesc,
          typename OutDesc,
          typename ConvStrides,
          typename ConvDilations,
          typename LeftPads,
          typename RightPads>
void device_convolution_backward_data_implicit_gemm_v1r2_nchw_kcyx_nkhw(InDesc in_nchw_desc,
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

    constexpr index_t BPerBlock = 32;
    constexpr index_t EPerBlock = 32;
    constexpr index_t KPerBlock = 8;

    constexpr index_t GemmMPerThreadSubC = 4;
    constexpr index_t GemmNPerThreadSubC = 4;
    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 4;
    constexpr index_t GemmKPerThreadLoop = 1;
    constexpr index_t GemmDataPerReadA   = 4;
    constexpr index_t GemmDataPerReadB   = 4;

    using OutBlockCopySubLengths_K_B_N0     = Sequence<1, 1, 4>;
    using OutBlockCopyClusterLengths_K_B_N0 = Sequence<8, 32, 1>;

    constexpr index_t OutBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t OutBlockCopyDstDataPerWrite_N0 = 4;

    using WeiBlockCopySubLengths_K_E_C0     = Sequence<1, 4, 1>;
    using WeiBlockCopyClusterLengths_K_E_C0 = Sequence<8, 8, 4>;

    constexpr index_t WeiBlockCopySrcDataPerRead_E   = 4;
    constexpr index_t WeiBlockCopyDstDataPerWrite_C0 = 1;

    constexpr index_t InThreadCopyDstDataPerWrite_B = 1;
#endif

    constexpr index_t C0 = GemmMPerThreadSubC;
    constexpr index_t N0 = GemmNPerThreadSubC;

    constexpr index_t C1 = C / C0;
    constexpr index_t N1 = N / N0;

    constexpr index_t E = C1 * Y * X;
    constexpr index_t B = (N1 * Ho * Wo);

    constexpr index_t GridSize =
        ((E + EPerBlock - 1) / EPerBlock) * ((B + BPerBlock - 1) / BPerBlock);

    printf("%s: BlockSize %u, GridSize %u \n", __func__, BlockSize, GridSize);

    constexpr auto gridwise_conv =
        GridwiseConvolutionBackwardDataImplicitGemm_v1r2_nchw_kcyx_nkhw_lds_double_buffer<
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
            EPerBlock,
            BPerBlock,
            KPerBlock,
            GemmMPerThreadSubC,
            GemmNPerThreadSubC,
            GemmMLevel0Cluster,
            GemmNLevel0Cluster,
            GemmMLevel1Cluster,
            GemmNLevel1Cluster,
            GemmKPerThreadLoop,
            GemmDataPerReadA,
            GemmDataPerReadB,
            OutBlockCopySubLengths_K_B_N0,
            OutBlockCopyClusterLengths_K_B_N0,
            OutBlockCopySrcDataPerRead_B,
            OutBlockCopyDstDataPerWrite_N0,
            WeiBlockCopySubLengths_K_E_C0,
            WeiBlockCopyClusterLengths_K_E_C0,
            WeiBlockCopySrcDataPerRead_E,
            WeiBlockCopyDstDataPerWrite_C0,
            InThreadCopyDstDataPerWrite_B>{};

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
