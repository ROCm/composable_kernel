#pragma once
#include <unistd.h>
#include "device.hpp"
#include "gridwise_implicit_gemm_convolution_2_cnhw_csrk_knhw.cuh"
#include "gridwise_implicit_gemm_convolution_2_cnhw_csrk_knhw_lds_double_buffer.cuh"

template <class T, class InDesc, class WeiDesc, class OutDesc>
void device_implicit_gemm_convolution_2_cnhw_csrk_knhw(InDesc,
                                                       const Tensor<T>& in_nchw,
                                                       WeiDesc,
                                                       const Tensor<T>& wei_kcsr,
                                                       OutDesc,
                                                       Tensor<T>& out_nkhw,
                                                       unsigned nrepeat)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_nchw_desc  = InDesc{};
    constexpr auto wei_kcsr_desc = WeiDesc{};
    constexpr auto out_nkhw_desc = OutDesc{};

    constexpr unsigned N  = in_nchw_desc.GetLength(I0);
    constexpr unsigned Hi = in_nchw_desc.GetLength(I2);
    constexpr unsigned Wi = in_nchw_desc.GetLength(I3);

    constexpr unsigned Ho = out_nkhw_desc.GetLength(I2);
    constexpr unsigned Wo = out_nkhw_desc.GetLength(I3);

    constexpr unsigned K = wei_kcsr_desc.GetLength(I0);
    constexpr unsigned C = wei_kcsr_desc.GetLength(I1);
    constexpr unsigned S = wei_kcsr_desc.GetLength(I2);
    constexpr unsigned R = wei_kcsr_desc.GetLength(I3);

    constexpr unsigned BGhostRead = (S - 1) * Wi + (R - 1);

    // convert in_nchw to in_cnhw
    auto in_cnhw_desc = make_ConstantTensorDescriptor(Sequence<C, N, Hi, Wi>{});
    ostream_ConstantTensorDescriptor(in_cnhw_desc, std::cout << "in_cnhw_desc: ");

    Tensor<T> in_cnhw(make_TensorDescriptor(in_cnhw_desc));

    auto f_reorder_nchw2cnhw = [&](auto n, auto c, auto hi, auto wi) {
        in_cnhw(c, n, hi, wi) = in_nchw(n, c, hi, wi);
    };

    make_ParallelTensorFunctor(f_reorder_nchw2cnhw, N, C, Hi, Wi)(
        std::thread::hardware_concurrency());

    // convert wei_kcsr to wei_csrk
    auto wei_csrk_desc = make_ConstantTensorDescriptor(Sequence<C, S, R, K>{});
    ostream_ConstantTensorDescriptor(wei_csrk_desc, std::cout << "wei_csrk_desc: ");

    Tensor<T> wei_csrk(make_TensorDescriptor(wei_csrk_desc));

    auto f_reorder_kcsr2csrk = [&](auto k, auto c, auto s, auto r) {
        wei_csrk(c, s, r, k) = wei_kcsr(k, c, s, r);
    };

    make_ParallelTensorFunctor(f_reorder_kcsr2csrk, K, C, S, R)(
        std::thread::hardware_concurrency());

    // conver out_nkhw to out_knhw
    auto out_knhw_desc = make_ConstantTensorDescriptor(Sequence<K, N, Ho, Wo>{});
    ostream_ConstantTensorDescriptor(out_knhw_desc, std::cout << "out_knhw_desc: ");

    Tensor<T> out_knhw(make_TensorDescriptor(out_knhw_desc));

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
    DeviceMem in_cnhw_device_buf(data_sz * (in_cnhw.mDesc.GetElementSpace() + BGhostRead +
                                            BPerBlock)); // reserve extra space for BGhostRead
    DeviceMem wei_csrk_device_buf(data_sz * wei_csrk.mDesc.GetElementSpace());
    DeviceMem out_knhw_device_buf(data_sz * out_knhw.mDesc.GetElementSpace());

    in_cnhw_device_buf.ToDevice(in_cnhw.mData.data());
    wei_csrk_device_buf.ToDevice(wei_csrk.mData.data());
    out_knhw_device_buf.ToDevice(out_knhw.mData.data());

    for(unsigned i = 0; i < nrepeat; ++i)
    {
        float time = launch_kernel(
#if 0
            gridwise_implicit_gemm_convolution_2_cnhw_csrk_knhw
#else
            gridwise_implicit_gemm_convolution_2_cnhw_csrk_knhw_lds_double_buffer
#endif
            <GridSize,
             BlockSize,
             T,
             decltype(in_cnhw_desc),
             decltype(wei_csrk_desc),
             decltype(out_knhw_desc),
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
            static_cast<T*>(in_cnhw_device_buf.GetDeviceBuffer()),
            static_cast<T*>(wei_csrk_device_buf.GetDeviceBuffer()),
            static_cast<T*>(out_knhw_device_buf.GetDeviceBuffer()));

        printf("Elapsed time : %f ms\n", time);
        usleep(std::min(time * 1000, float(10000)));
    }

    out_knhw_device_buf.FromDevice(out_knhw.mData.data());

    // convert out_knhw to out_nkhw
    auto f_reorder_knhw2nkhw = [&](auto n, auto k, auto ho, auto wo) {
        out_nkhw(n, k, ho, wo) = out_knhw(k, n, ho, wo);
    };

    make_ParallelTensorFunctor(f_reorder_knhw2nkhw, N, K, Ho, Wo)(
        std::thread::hardware_concurrency());
}
