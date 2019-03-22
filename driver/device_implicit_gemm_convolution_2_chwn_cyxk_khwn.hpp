#pragma once
#include <unistd.h>
#include "device.hpp"
#include "gridwise_implicit_gemm_convolution_2_chwn_cyxk_khwn.hip.hpp"
#include "gridwise_implicit_gemm_convolution_2_chwn_cyxk_khwn_lds_double_buffer.hip.hpp"

template <class T, class InDesc, class WeiDesc, class OutDesc>
void device_implicit_gemm_convolution_2_chwn_cyxk_khwn(InDesc,
                                                       const Tensor<T>& in_nchw,
                                                       WeiDesc,
                                                       const Tensor<T>& wei_kcyx,
                                                       OutDesc,
                                                       Tensor<T>& out_nkhw,
                                                       unsigned nrepeat)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_nchw_desc  = InDesc{};
    constexpr auto wei_kcyx_desc = WeiDesc{};
    constexpr auto out_nkhw_desc = OutDesc{};

    constexpr unsigned N  = in_nchw_desc.GetLength(I0);
    constexpr unsigned Hi = in_nchw_desc.GetLength(I2);
    constexpr unsigned Wi = in_nchw_desc.GetLength(I3);

    constexpr unsigned Ho = out_nkhw_desc.GetLength(I2);
    constexpr unsigned Wo = out_nkhw_desc.GetLength(I3);

    constexpr unsigned K = wei_kcyx_desc.GetLength(I0);
    constexpr unsigned C = wei_kcyx_desc.GetLength(I1);
    constexpr unsigned Y = wei_kcyx_desc.GetLength(I2);
    constexpr unsigned X = wei_kcyx_desc.GetLength(I3);

    constexpr unsigned BGhostRead = (Y - 1) * Wi + (X - 1);

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
    constexpr unsigned BPerBlock = 128;
    constexpr unsigned KPerBlock = 64;
    constexpr unsigned CPerBlock = 4;

    constexpr unsigned BPerThread = 8;
    constexpr unsigned KPerThread = 8;

    constexpr unsigned GemmMPerThreadSubC = 4;
    constexpr unsigned GemmNPerThreadSubC = 4;
    constexpr unsigned GemmMLevel0Cluster = 4;
    constexpr unsigned GemmNLevel0Cluster = 2;
    constexpr unsigned GemmMLevel1Cluster = 2;
    constexpr unsigned GemmNLevel1Cluster = 8;
    constexpr unsigned GemmKPerThreadLoop = 1;

    constexpr unsigned GemmThreadPerColumnPerCluster = 8;
    constexpr unsigned GemmThreadPerRowPerCluster    = 8;

    constexpr unsigned InBlockCopyThreadPerDim0 = 4;
    constexpr unsigned InBlockCopyThreadPerDim1 = 16;

    constexpr unsigned WeiBlockCopyThreadPerDim0 = 4;
    constexpr unsigned WeiBlockCopyThreadPerDim1 = 16;

    constexpr unsigned InBlockCopyDataPerRead  = 4;
    constexpr unsigned WeiBlockCopyDataPerRead = 4;

    constexpr unsigned BlockSize = 128;
#elif 0
    // 1x1, 28x28, 64 threads
    constexpr unsigned BPerBlock = 64;
    constexpr unsigned KPerBlock = 64;
    constexpr unsigned CPerBlock = 8;

    constexpr unsigned BPerThread = 8;
    constexpr unsigned KPerThread = 8;

    constexpr unsigned GemmMPerThreadSubC = 4;
    constexpr unsigned GemmNPerThreadSubC = 4;
    constexpr unsigned GemmMLevel0Cluster = 4;
    constexpr unsigned GemmNLevel0Cluster = 2;
    constexpr unsigned GemmMLevel1Cluster = 2;
    constexpr unsigned GemmNLevel1Cluster = 4;
    constexpr unsigned GemmKPerThreadLoop = 1;

    constexpr unsigned GemmThreadPerColumnPerCluster = 8;
    constexpr unsigned GemmThreadPerRowPerCluster    = 8;

    constexpr unsigned InBlockCopyThreadPerDim0 = 4;
    constexpr unsigned InBlockCopyThreadPerDim1 = 16;

    constexpr unsigned WeiBlockCopyThreadPerDim0 = 4;
    constexpr unsigned WeiBlockCopyThreadPerDim1 = 16;

    constexpr unsigned InBlockCopyDataPerRead  = 4;
    constexpr unsigned WeiBlockCopyDataPerRead = 4;

    constexpr unsigned BlockSize = 64;
#elif 1
    // 1x1, 28x28, 128 threads, no lds-double-buffer
    // 1x1, 28x28, 128 threads, with lds-double-buffer, max_register = 128
    constexpr unsigned BPerBlock = 64;
    constexpr unsigned KPerBlock = 128;
    constexpr unsigned CPerBlock = 8;

    constexpr unsigned BPerThread = 8;
    constexpr unsigned KPerThread = 8;

    constexpr unsigned GemmMPerThreadSubC = 4;
    constexpr unsigned GemmNPerThreadSubC = 4;
    constexpr unsigned GemmMLevel0Cluster = 4;
    constexpr unsigned GemmNLevel0Cluster = 2;
    constexpr unsigned GemmMLevel1Cluster = 4;
    constexpr unsigned GemmNLevel1Cluster = 4;
    constexpr unsigned GemmKPerThreadLoop = 1;

    constexpr unsigned GemmThreadPerColumnPerCluster = 8;
    constexpr unsigned GemmThreadPerRowPerCluster    = 8;

    constexpr unsigned InBlockCopyThreadPerDim0 = 4;
    constexpr unsigned InBlockCopyThreadPerDim1 = 16;

    constexpr unsigned WeiBlockCopyThreadPerDim0 = 4;
    constexpr unsigned WeiBlockCopyThreadPerDim1 = 16;

    constexpr unsigned InBlockCopyDataPerRead  = 4;
    constexpr unsigned WeiBlockCopyDataPerRead = 4;

    constexpr unsigned BlockSize = 128;
#elif 1
    // 1x1, 28x28, 256 thread
    constexpr unsigned BPerBlock = 128;
    constexpr unsigned KPerBlock = 128;
    constexpr unsigned CPerBlock = 8;

    constexpr unsigned BPerThread = 8;
    constexpr unsigned KPerThread = 8;

    constexpr unsigned GemmMPerThreadSubC = 4;
    constexpr unsigned GemmNPerThreadSubC = 4;
    constexpr unsigned GemmMLevel0Cluster = 4;
    constexpr unsigned GemmNLevel0Cluster = 4;
    constexpr unsigned GemmMLevel1Cluster = 4;
    constexpr unsigned GemmNLevel1Cluster = 4;
    constexpr unsigned GemmKPerThreadLoop = 1;

    constexpr unsigned GemmThreadPerColumnPerCluster = 8;
    constexpr unsigned GemmThreadPerRowPerCluster    = 8;

    constexpr unsigned InBlockCopyThreadPerDim0 = 4;
    constexpr unsigned InBlockCopyThreadPerDim1 = 16;

    constexpr unsigned WeiBlockCopyThreadPerDim0 = 4;
    constexpr unsigned WeiBlockCopyThreadPerDim1 = 16;

    constexpr unsigned InBlockCopyDataPerRead  = 4;
    constexpr unsigned WeiBlockCopyDataPerRead = 4;

    constexpr unsigned BlockSize = 256;
#endif

    constexpr unsigned GridSize =
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

    for(unsigned i = 0; i < nrepeat; ++i)
    {
        float time = launch_kernel(
#if 0
            gridwise_implicit_gemm_convolution_2_chwn_cyxk_khwn
#else
            gridwise_implicit_gemm_convolution_2_chwn_cyxk_khwn_lds_double_buffer
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
             GemmThreadPerColumnPerCluster,
             GemmThreadPerRowPerCluster,
             GemmMPerThreadSubC,
             GemmNPerThreadSubC,
             GemmMLevel0Cluster,
             GemmNLevel0Cluster,
             GemmMLevel1Cluster,
             GemmNLevel1Cluster,
             GemmKPerThreadLoop,
             InBlockCopyThreadPerDim0,
             InBlockCopyThreadPerDim1,
             WeiBlockCopyThreadPerDim0,
             WeiBlockCopyThreadPerDim1,
             InBlockCopyDataPerRead,
             WeiBlockCopyDataPerRead>,
            dim3(GridSize),
            dim3(BlockSize),
            static_cast<T*>(in_chwn_device_buf.GetDeviceBuffer()),
            static_cast<T*>(wei_cyxk_device_buf.GetDeviceBuffer()),
            static_cast<T*>(out_khwn_device_buf.GetDeviceBuffer()));

        printf("Elapsed time : %f ms\n", time);
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
