#pragma once
#include <unistd.h>
#include "device.hpp"
#include "tensor.hpp"
#include "gridwise_convolution_kernel_wrapper.hpp"
#include "gridwise_convolution_implicit_gemm_v3_nchw_cyxk_nkhw.hpp"
#include "gridwise_convolution_implicit_gemm_v3_nchw_cyxk_nkhw_lds_double_buffer.hpp"

template <class T, class InDesc, class WeiDesc, class OutDesc>
void device_convolution_implicit_gemm_v3_nchw_cyxk_nkhw(InDesc,
                                                        const Tensor<T>& in_nchw,
                                                        WeiDesc,
                                                        const Tensor<T>& wei_kcyx,
                                                        OutDesc,
                                                        Tensor<T>& out_nkhw,
                                                        index_t nrepeat)
{
    using namespace ck;

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

    // reorder weight
    auto wei_cyxk_desc = make_ConstantTensorDescriptor_packed(Sequence<C, Y, X, K>{});
    ostream_ConstantTensorDescriptor(wei_cyxk_desc, std::cout << "wei_cyxk_desc: ");

    Tensor<T> wei_cyxk(make_TensorDescriptor(wei_cyxk_desc));

    auto f_reorder_kcyx2cyxk = [&](auto k, auto c, auto y, auto x) {
        wei_cyxk(c, y, x, k) = wei_kcyx(k, c, y, x);
    };

    make_ParallelTensorFunctor(f_reorder_kcyx2cyxk, K, C, Y, X)(
        std::thread::hardware_concurrency());

    std::size_t data_sz = sizeof(T);
    DeviceMem in_nchw_device_buf(data_sz * in_nchw.mDesc.GetElementSpace());
    DeviceMem wei_cyxk_device_buf(data_sz * wei_cyxk.mDesc.GetElementSpace());
    DeviceMem out_nkhw_device_buf(data_sz * out_nkhw.mDesc.GetElementSpace());

    in_nchw_device_buf.ToDevice(in_nchw.mData.data());
    wei_cyxk_device_buf.ToDevice(wei_cyxk.mData.data());
    out_nkhw_device_buf.ToDevice(out_nkhw.mData.data());

    constexpr index_t N1 = 2;
    constexpr index_t N2 = 4;

    constexpr index_t B = (N * Ho * Wo) / (N1 * N2);

#if 1
    constexpr index_t BlockSize = 256;

    constexpr index_t BPerBlock = 16;
    constexpr index_t KPerBlock = 128;
    constexpr index_t CPerBlock = 8;

    constexpr index_t GemmMPerThreadSubC = 4;
    constexpr index_t GemmNPerThreadSubC = 4;
    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 4;
    constexpr index_t GemmKPerThreadLoop = 1;
    constexpr index_t GemmDataPerReadA   = 4;
    constexpr index_t GemmDataPerReadB   = 4;

    using InBlockCopySubLengths_C_N1_B_N2     = Sequence<1, 1, 1, 4>;
    using InBlockCopyClusterLengths_C_N1_B_N2 = Sequence<8, 2, 16, 1>;

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = 4;

    using WeiBlockCopySubLengths_C_K     = Sequence<1, 4>;
    using WeiBlockCopyClusterLengths_C_K = Sequence<8, 32>;

    constexpr index_t WeiBlockCopyDataPerAccess_K = 4;
#endif

    constexpr index_t GridSize =
        ((B + BPerBlock - 1) / BPerBlock) * ((K + KPerBlock - 1) / KPerBlock);

    printf("%s: BlockSize %u, GridSize %u \n", __func__, BlockSize, GridSize);

    for(index_t i = 0; i < nrepeat; ++i)
    {
        constexpr auto gridwise_conv =
#if 0
            GridwiseConvolutionImplicitGemm_v3_nchw_cyxk_nkhw
#else
            GridwiseConvolutionImplicitGemm_v3_nchw_cyxk_nkhw_lds_double_buffer
#endif
            <GridSize,
             BlockSize,
             T,
             decltype(in_nchw_desc),
             decltype(wei_cyxk_desc),
             decltype(out_nkhw_desc),
             BPerBlock,
             KPerBlock,
             CPerBlock,
             N1,
             N2,
             GemmMPerThreadSubC,
             GemmNPerThreadSubC,
             GemmMLevel0Cluster,
             GemmNLevel0Cluster,
             GemmMLevel1Cluster,
             GemmNLevel1Cluster,
             GemmKPerThreadLoop,
             GemmDataPerReadA,
             GemmDataPerReadB,
             InBlockCopySubLengths_C_N1_B_N2,
             InBlockCopyClusterLengths_C_N1_B_N2,
             InBlockCopySrcDataPerRead_B,
             InBlockCopyDstDataPerWrite_N2,
             WeiBlockCopySubLengths_C_K,
             WeiBlockCopyClusterLengths_C_K,
             WeiBlockCopyDataPerAccess_K>{};

        float time =
            launch_and_time_kernel(run_gridwise_convolution_kernel<decltype(gridwise_conv), T>,
                                   dim3(GridSize),
                                   dim3(BlockSize),
                                   0,
                                   static_cast<T*>(in_nchw_device_buf.GetDeviceBuffer()),
                                   static_cast<T*>(wei_cyxk_device_buf.GetDeviceBuffer()),
                                   static_cast<T*>(out_nkhw_device_buf.GetDeviceBuffer()));

        printf("Elapsed time : %f ms, %f TFlop/s\n",
               time,
               (float)calculate_convolution_flops(InDesc{}, WeiDesc{}, OutDesc{}) /
                   (std::size_t(1000) * 1000 * 1000) / time);
        usleep(std::min(time * 1000, float(10000)));
    }

    out_nkhw_device_buf.FromDevice(out_nkhw.mData.data());
}
