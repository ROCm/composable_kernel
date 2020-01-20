#pragma once
#include <unistd.h>
#include "device.hpp"
#include "tensor.hpp"
#include "gridwise_convolution_kernel_wrapper.hpp"
#include "gridwise_convolution_implicit_gemm_v2_chwn_cyxk_khwn.hpp"
#include "gridwise_convolution_implicit_gemm_v2_chwn_cyxk_khwn_lds_double_buffer.hpp"

using namespace ck;

template <class T, class InDesc, class WeiDesc, class OutDesc>
void device_convolution_implicit_gemm_v2_chwn_cyxk_khwn(InDesc,
                                                        const Tensor<T>& in_nchw,
                                                        WeiDesc,
                                                        const Tensor<T>& wei_kcyx,
                                                        OutDesc,
                                                        Tensor<T>& out_nkhw,
                                                        index_t nrepeat)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_nchw_desc  = InDesc{};
    constexpr auto wei_kcyx_desc = WeiDesc{};
    constexpr auto out_nkhw_desc = OutDesc{};

    constexpr index_t N  = in_nchw_desc.GetLength(I0);
    constexpr index_t Hi = in_nchw_desc.GetLength(I2);
    constexpr index_t Wi = in_nchw_desc.GetLength(I3);

    constexpr index_t Ho = out_nkhw_desc.GetLength(I2);
    constexpr index_t Wo = out_nkhw_desc.GetLength(I3);

    constexpr index_t K = wei_kcyx_desc.GetLength(I0);
    constexpr index_t C = wei_kcyx_desc.GetLength(I1);
    constexpr index_t Y = wei_kcyx_desc.GetLength(I2);
    constexpr index_t X = wei_kcyx_desc.GetLength(I3);

    constexpr index_t BGhostRead = (Y - 1) * Wi + (X - 1);

    // convert in_nchw to in_cnhw
    auto in_chwn_desc = make_ConstantTensorDescriptor(Sequence<C, Hi, Wi, N>{});
    ostream_ConstantTensorDescriptor(in_chwn_desc, std::cout << "in_chwn_desc: ");

    Tensor<T> in_chwn(make_TensorDescriptor(in_chwn_desc));

    make_ParallelTensorFunctor(
        [&](auto n, auto c, auto hi, auto wi) { in_chwn(c, hi, wi, n) = in_nchw(n, c, hi, wi); },
        N,
        C,
        Hi,
        Wi)(std::thread::hardware_concurrency());

    // convert wei_kcyx to wei_cyxk
    auto wei_cyxk_desc = make_ConstantTensorDescriptor(Sequence<C, Y, X, K>{});
    ostream_ConstantTensorDescriptor(wei_cyxk_desc, std::cout << "wei_cyxk_desc: ");

    Tensor<T> wei_cyxk(make_TensorDescriptor(wei_cyxk_desc));

    make_ParallelTensorFunctor(
        [&](auto k, auto c, auto y, auto x) { wei_cyxk(c, y, x, k) = wei_kcyx(k, c, y, x); },
        K,
        C,
        Y,
        X)(std::thread::hardware_concurrency());

    // conver out_nkhw to out_knhw
    auto out_khwn_desc = make_ConstantTensorDescriptor(Sequence<K, Ho, Wo, N>{});
    ostream_ConstantTensorDescriptor(out_khwn_desc, std::cout << "out_khwn_desc: ");

    Tensor<T> out_khwn(make_TensorDescriptor(out_khwn_desc));

#if 0
    // 3x3, 34x34
    // need to use register double buffer for GEMM
    constexpr index_t BPerBlock = 128;
    constexpr index_t KPerBlock = 64;
    constexpr index_t CPerBlock = 4;

    constexpr index_t BPerThread = 8;
    constexpr index_t KPerThread = 8;

    constexpr index_t GemmMPerThreadSubC = 4;
    constexpr index_t GemmNPerThreadSubC = 4;
    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 8;
    constexpr index_t GemmKPerThreadLoop = 1;

    constexpr index_t InBlockCopyThreadPerDim0 = 4;
    constexpr index_t InBlockCopyThreadPerDim1 = 16;

    constexpr index_t WeiBlockCopyThreadPerDim0 = 4;
    constexpr index_t WeiBlockCopyThreadPerDim1 = 16;

    constexpr index_t InBlockCopyDataPerRead  = 4;
    constexpr index_t WeiBlockCopyDataPerRead = 4;
    constexpr index_t OutThreadCopyDataPerWrite = 4;

    constexpr index_t BlockSize = 128;
#elif 0
    // 1x1, 28x28, 64 threads
    constexpr index_t BPerBlock = 64;
    constexpr index_t KPerBlock = 64;
    constexpr index_t CPerBlock = 8;

    constexpr index_t BPerThread = 8;
    constexpr index_t KPerThread = 8;

    constexpr index_t GemmMPerThreadSubC = 4;
    constexpr index_t GemmNPerThreadSubC = 4;
    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 4;
    constexpr index_t GemmKPerThreadLoop = 1;

    constexpr index_t GemmThreadPerColumnPerCluster = 8;
    constexpr index_t GemmThreadPerRowPerCluster    = 8;

    constexpr index_t InBlockCopyThreadPerDim0 = 4;
    constexpr index_t InBlockCopyThreadPerDim1 = 16;

    constexpr index_t WeiBlockCopyThreadPerDim0 = 4;
    constexpr index_t WeiBlockCopyThreadPerDim1 = 16;

    constexpr index_t InBlockCopyDataPerRead  = 4;
    constexpr index_t WeiBlockCopyDataPerRead = 4;

    constexpr index_t BlockSize = 64;
#elif 0
    // 1x1, 28x28, 128 threads, no lds-double-buffer
    // 1x1, 28x28, 128 threads, with lds-double-buffer, max_register = 128
    constexpr index_t BPerBlock = 64;
    constexpr index_t KPerBlock = 128;
    constexpr index_t CPerBlock = 8;

    constexpr index_t BPerThread = 8;
    constexpr index_t KPerThread = 8;

    constexpr index_t GemmMPerThreadSubC = 4;
    constexpr index_t GemmNPerThreadSubC = 4;
    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 4;
    constexpr index_t GemmKPerThreadLoop = 1;

    constexpr index_t GemmThreadPerColumnPerCluster = 8;
    constexpr index_t GemmThreadPerRowPerCluster    = 8;

    constexpr index_t InBlockCopyThreadPerDim0 = 4;
    constexpr index_t InBlockCopyThreadPerDim1 = 16;

    constexpr index_t WeiBlockCopyThreadPerDim0 = 4;
    constexpr index_t WeiBlockCopyThreadPerDim1 = 16;

    constexpr index_t InBlockCopyDataPerRead  = 4;
    constexpr index_t WeiBlockCopyDataPerRead = 4;

    constexpr index_t BlockSize = 128;
#elif 0
    // 1x1, 28x28, 256 thread
    constexpr index_t BPerBlock = 128;
    constexpr index_t KPerBlock = 128;
    constexpr index_t CPerBlock = 8;

    constexpr index_t BPerThread = 8;
    constexpr index_t KPerThread = 8;

    constexpr index_t GemmMPerThreadSubC = 4;
    constexpr index_t GemmNPerThreadSubC = 4;
    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 4;
    constexpr index_t GemmKPerThreadLoop = 1;

    constexpr index_t GemmThreadPerColumnPerCluster = 8;
    constexpr index_t GemmThreadPerRowPerCluster    = 8;

    constexpr index_t InBlockCopyThreadPerDim0 = 4;
    constexpr index_t InBlockCopyThreadPerDim1 = 16;

    constexpr index_t WeiBlockCopyThreadPerDim0 = 4;
    constexpr index_t WeiBlockCopyThreadPerDim1 = 16;

    constexpr index_t InBlockCopyDataPerRead  = 4;
    constexpr index_t WeiBlockCopyDataPerRead = 4;

    constexpr index_t BlockSize = 256;
#elif 0
    // 1x1, 14x14, Pascal, enable lds_double_buffer, disable register double buffer
    constexpr index_t BPerBlock = 64;
    constexpr index_t KPerBlock = 128;
    constexpr index_t CPerBlock = 8;

    constexpr index_t BPerThread = 8;
    constexpr index_t KPerThread = 8;

    constexpr index_t GemmMPerThreadSubC = 4;
    constexpr index_t GemmNPerThreadSubC = 4;
    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 4;
    constexpr index_t GemmKPerThreadLoop = 1;
    constexpr index_t GemmDataPerReadA   = 4;
    constexpr index_t GemmDataPerReadB   = 4;

    constexpr index_t InBlockCopyThreadPerDim0 = 4;
    constexpr index_t InBlockCopyThreadPerDim1 = 16;

    constexpr index_t WeiBlockCopyThreadPerDim0 = 4;
    constexpr index_t WeiBlockCopyThreadPerDim1 = 16;

    constexpr index_t InBlockCopyDataPerRead    = 4;
    constexpr index_t WeiBlockCopyDataPerRead   = 4;
    constexpr index_t OutThreadCopyDataPerWrite = 4;

    constexpr index_t BlockSize = 128;
#elif 1
    // 1x1, 14x14, Vega 20, enable lds_double_buffer, disable register_double_buffer
    constexpr index_t BPerBlock = 128;
    constexpr index_t KPerBlock = 128;
    constexpr index_t CPerBlock = 8;

    constexpr index_t BPerThread = 8;
    constexpr index_t KPerThread = 8;

    constexpr index_t GemmMPerThreadSubC = 4;
    constexpr index_t GemmNPerThreadSubC = 4;
    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 4;
    constexpr index_t GemmKPerThreadLoop = 1;
    constexpr index_t GemmDataPerReadA   = 4;
    constexpr index_t GemmDataPerReadB   = 4;

    constexpr index_t InBlockCopyThreadPerDim0 = 4;
    constexpr index_t InBlockCopyThreadPerDim1 = 16;

    constexpr index_t WeiBlockCopyThreadPerDim0 = 4;
    constexpr index_t WeiBlockCopyThreadPerDim1 = 16;

    constexpr index_t InBlockCopyDataPerRead    = 4;
    constexpr index_t WeiBlockCopyDataPerRead   = 4;
    constexpr index_t OutThreadCopyDataPerWrite = 4;

    constexpr index_t BlockSize = 256;
#endif

    constexpr index_t GridSize =
        ((N * Hi * Wi + BPerBlock - 1) / BPerBlock) * ((K + KPerBlock - 1) / KPerBlock);

    printf("%s: BlockSize %u, GridSize %u \n", __func__, BlockSize, GridSize);

    // mem
    std::size_t data_sz = sizeof(T);
    DeviceMem in_chwn_device_buf(data_sz * (in_chwn.mDesc.GetElementSpace() + BGhostRead +
                                            BPerBlock)); // reserve extra space for BGhostRead
    DeviceMem wei_cyxk_device_buf(data_sz * wei_cyxk.mDesc.GetElementSpace());
    DeviceMem out_khwn_device_buf(data_sz * out_khwn.mDesc.GetElementSpace());

    in_chwn_device_buf.ToDevice(in_chwn.mData.data());
    wei_cyxk_device_buf.ToDevice(wei_cyxk.mData.data());
    out_khwn_device_buf.ToDevice(out_khwn.mData.data());

    for(index_t i = 0; i < nrepeat; ++i)
    {
        constexpr auto gridwise_conv =
#if 0
            GridwiseConvolutionImplicitGemm_v2_chwn_cyxk_khwn
#else
            GridwiseConvolutionImplicitGemm_v2_chwn_cyxk_khwn_lds_double_buffer
#endif
            <GridSize,
             BlockSize,
             T,
             decltype(in_chwn_desc),
             decltype(wei_cyxk_desc),
             decltype(out_khwn_desc),
             BPerBlock,
             KPerBlock,
             CPerBlock,
             BPerThread,
             KPerThread,
             GemmMPerThreadSubC,
             GemmNPerThreadSubC,
             GemmMLevel0Cluster,
             GemmNLevel0Cluster,
             GemmMLevel1Cluster,
             GemmNLevel1Cluster,
             GemmKPerThreadLoop,
             GemmDataPerReadA,
             GemmDataPerReadB,
             InBlockCopyThreadPerDim0,
             InBlockCopyThreadPerDim1,
             WeiBlockCopyThreadPerDim0,
             WeiBlockCopyThreadPerDim1,
             InBlockCopyDataPerRead,
             WeiBlockCopyDataPerRead,
             OutThreadCopyDataPerWrite>{};

        float time =
            launch_and_time_kernel(run_gridwise_convolution_kernel<decltype(gridwise_conv), T>,
                                   dim3(GridSize),
                                   dim3(BlockSize),
                                   0,
                                   static_cast<T*>(in_chwn_device_buf.GetDeviceBuffer()),
                                   static_cast<T*>(wei_cyxk_device_buf.GetDeviceBuffer()),
                                   static_cast<T*>(out_khwn_device_buf.GetDeviceBuffer()));

        printf("Elapsed time : %f ms, %f TFlop/s\n",
               time,
               (float)calculate_convolution_flops(InDesc{}, WeiDesc{}, OutDesc{}) /
                   (std::size_t(1000) * 1000 * 1000) / time);
        usleep(std::min(time * 1000, float(10000)));
    }

    out_khwn_device_buf.FromDevice(out_khwn.mData.data());

    // convert out_khwn to out_nkhw
    make_ParallelTensorFunctor(
        [&](auto n, auto k, auto ho, auto wo) { out_nkhw(n, k, ho, wo) = out_khwn(k, ho, wo, n); },
        N,
        K,
        Ho,
        Wo)(std::thread::hardware_concurrency());
}
