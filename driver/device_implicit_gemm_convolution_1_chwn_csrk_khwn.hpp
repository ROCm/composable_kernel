#pragma once
#include <unistd.h>
#include "device.hpp"
#include "gridwise_implicit_gemm_convolution_1_chwn_csrk_khwn.hip.hpp"

template <class T, class InDesc, class WeiDesc, class OutDesc>
void device_implicit_gemm_convolution_1_chwn_csrk_khwn(InDesc,
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

    constexpr unsigned Hi = in_nchw_desc.GetLength(I2);
    constexpr unsigned Wi = in_nchw_desc.GetLength(I3);

    constexpr unsigned N  = out_nkhw_desc.GetLength(I0);
    constexpr unsigned Ho = out_nkhw_desc.GetLength(I2);
    constexpr unsigned Wo = out_nkhw_desc.GetLength(I3);

    constexpr unsigned K = wei_kcsr_desc.GetLength(I0);
    constexpr unsigned C = wei_kcsr_desc.GetLength(I1);
    constexpr unsigned Y = wei_kcsr_desc.GetLength(I2);
    constexpr unsigned X = wei_kcsr_desc.GetLength(I3);

    // reorder weight
    auto wei_csrk_desc = make_ConstantTensorDescriptor(Sequence<C, Y, X, K>{});
    ostream_ConstantTensorDescriptor(wei_csrk_desc, std::cout << "wei_csrk_desc: ");

    Tensor<T> wei_csrk(make_TensorDescriptor(wei_csrk_desc));

    auto f_reorder_kcsr2csrk = [&](auto k, auto c, auto y, auto x) {
        wei_csrk(c, y, x, k) = wei_kcsr(k, c, y, x);
    };

    make_ParallelTensorFunctor(f_reorder_kcsr2csrk, K, C, Y, X)(
        std::thread::hardware_concurrency());

    // reorder input
    auto in_chwn_desc = make_ConstantTensorDescriptor(Sequence<C, Hi, Wi, N>{});
    ostream_ConstantTensorDescriptor(in_chwn_desc, std::cout << "in_chwn_desc: ");

    Tensor<T> in_chwn(make_TensorDescriptor(in_chwn_desc));

    auto f_reorder_nchw2chwn = [&](auto n, auto c, auto hi, auto wi) {
        in_chwn(c, hi, wi, n) = in_nchw(n, c, hi, wi);
    };

    make_ParallelTensorFunctor(f_reorder_nchw2chwn, N, C, Hi, Wi)(
        std::thread::hardware_concurrency());

    // output
    auto out_khwn_desc = make_ConstantTensorDescriptor(Sequence<K, Ho, Wo, N>{});
    ostream_ConstantTensorDescriptor(out_khwn_desc, std::cout << "out_khwn_desc: ");

    Tensor<T> out_khwn(make_TensorDescriptor(out_khwn_desc));

    std::size_t data_sz = sizeof(T);
    DeviceMem in_chwn_device_buf(data_sz * in_chwn.mDesc.GetElementSpace());
    DeviceMem wei_csrk_device_buf(data_sz * wei_csrk.mDesc.GetElementSpace());
    DeviceMem out_khwn_device_buf(data_sz * out_khwn.mDesc.GetElementSpace());

    in_chwn_device_buf.ToDevice(in_chwn.mData.data());
    wei_csrk_device_buf.ToDevice(wei_csrk.mData.data());
    out_khwn_device_buf.ToDevice(out_khwn.mData.data());

#if 1
    // for 3x3, 34x34
    constexpr unsigned NPerBlock  = 16;
    constexpr unsigned KPerBlock  = 64;
    constexpr unsigned CPerBlock  = 4;
    constexpr unsigned HoPerBlock = 2;
    constexpr unsigned WoPerBlock = 4;

    constexpr unsigned NPerThread  = 8;
    constexpr unsigned KPerThread  = 8;
    constexpr unsigned HoPerThread = 1;
    constexpr unsigned WoPerThread = 1;

    constexpr unsigned InBlockCopy_ThreadPerDimC = 4;
    constexpr unsigned InBlockCopy_ThreadPerDimH = 4;
    constexpr unsigned InBlockCopy_ThreadPerDimW = 2;
    constexpr unsigned InBlockCopy_ThreadPerDimN = 4;
    constexpr unsigned InBlockCopyDataPerRead    = 4;

    constexpr unsigned WeiBlockCopyDataPerRead = 4;

    constexpr unsigned GemmMPerThreadSubC = 4;
    constexpr unsigned GemmNPerThreadSubC = 4;
    constexpr unsigned GemmMLevel0Cluster = 4;
    constexpr unsigned GemmNLevel0Cluster = 2;
    constexpr unsigned GemmMLevel1Cluster = 2;
    constexpr unsigned GemmNLevel1Cluster = 4;
    constexpr unsigned GemmKPerThreadLoop = 1;

    constexpr unsigned OutThreadCopyDataPerWrite = 2;

    constexpr unsigned BlockSize = 128;
#elif 0
    // for 5x5, 36x36
    constexpr unsigned NPerBlock  = 16;
    constexpr unsigned KPerBlock  = 64;
    constexpr unsigned CPerBlock  = 2;
    constexpr unsigned HoPerBlock = 2;
    constexpr unsigned WoPerBlock = 4;

    constexpr unsigned NPerThread  = 8;
    constexpr unsigned KPerThread  = 8;
    constexpr unsigned HoPerThread = 1;
    constexpr unsigned WoPerThread = 1;

    constexpr unsigned WeiBlockCopyThreadPerDim0 = 4;
    constexpr unsigned WeiBlockCopyThreadPerDim1 = 32;

    constexpr unsigned InBlockCopy_ThreadPerDimC = 2;
    constexpr unsigned InBlockCopy_ThreadPerDimH = 2;
    constexpr unsigned InBlockCopy_ThreadPerDimW = 4;
    constexpr unsigned InBlockCopy_ThreadPerDimN = 4;
    constexpr unsigned InBlockCopyDataPerRead    = 4;

    constexpr unsigned WeiBlockCopyDataPerRead = 2;

    constexpr unsigned GemmMPerThreadSubC = 4;
    constexpr unsigned GemmNPerThreadSubC = 4;
    constexpr unsigned GemmMLevel0Cluster = 4;
    constexpr unsigned GemmNLevel0Cluster = 2;
    constexpr unsigned GemmMLevel1Cluster = 2;
    constexpr unsigned GemmNLevel1Cluster = 4;
    constexpr unsigned GemmKPerThreadLoop = 1;

    constexpr unsigned OutThreadCopyDataPerWrite = 2;

    constexpr unsigned BlockSize = 128;
#elif 0
    // 3x3 58x58, NKC = 64, 64, 256
    constexpr unsigned NPerBlock  = 16;
    constexpr unsigned KPerBlock  = 64;
    constexpr unsigned CPerBlock  = 4;
    constexpr unsigned HoPerBlock = 2;
    constexpr unsigned WoPerBlock = 4;

    constexpr unsigned NPerThread  = 4;
    constexpr unsigned KPerThread  = 16;
    constexpr unsigned CPerThread  = 1;
    constexpr unsigned HoPerThread = 1;
    constexpr unsigned WoPerThread = 1;

    constexpr unsigned WeiBlockCopyThreadPerDim0 = 4;
    constexpr unsigned WeiBlockCopyThreadPerDim1 = 32;

    constexpr unsigned InBlockCopyDataPerRead  = 2; // not used, yet
    constexpr unsigned WeiBlockCopyDataPerRead = 4;

    constexpr unsigned BlockSize = 128;
#elif 0
    // 3x3 58x58, NKC = 16,256,128
    constexpr unsigned NPerBlock  = 8;
    constexpr unsigned KPerBlock  = 64;
    constexpr unsigned CPerBlock  = 2;
    constexpr unsigned HoPerBlock = 4;
    constexpr unsigned WoPerBlock = 4;

    constexpr unsigned NPerThread  = 4;
    constexpr unsigned KPerThread  = 16;
    constexpr unsigned CPerThread  = 1;
    constexpr unsigned HoPerThread = 1;
    constexpr unsigned WoPerThread = 1;

    constexpr unsigned BlockSize = 128;
#elif 0
    // for 7x7, 38x38
    constexpr unsigned NPerBlock  = 8;
    constexpr unsigned KPerBlock  = 64;
    constexpr unsigned CPerBlock  = 1;
    constexpr unsigned HoPerBlock = 4;
    constexpr unsigned WoPerBlock = 4;

    constexpr unsigned NPerThread  = 4;
    constexpr unsigned KPerThread  = 16;
    constexpr unsigned CPerThread  = 1;
    constexpr unsigned HoPerThread = 1;
    constexpr unsigned WoPerThread = 1;

    constexpr unsigned WeiBlockCopyThreadPerDim0 = 4;
    constexpr unsigned WeiBlockCopyThreadPerDim1 = 32;

    constexpr unsigned InBlockCopyDataPerRead  = 4; // not used, yet
    constexpr unsigned WeiBlockCopyDataPerRead = 4;

    constexpr unsigned BlockSize = 128;
#elif 0
    // for 3x3, 56x56
    constexpr unsigned NPerBlock  = 32;
    constexpr unsigned KPerBlock  = 64;
    constexpr unsigned CPerBlock  = 4;
    constexpr unsigned HoPerBlock = 2;
    constexpr unsigned WoPerBlock = 2;

    constexpr unsigned NPerThread  = 4;
    constexpr unsigned KPerThread  = 16;
    constexpr unsigned CPerThread  = 1;
    constexpr unsigned HoPerThread = 1;
    constexpr unsigned WoPerThread = 1;

    constexpr unsigned BlockSize = 128;
#elif 1
    // for 1x1, 28x28
    constexpr unsigned NPerBlock  = 16;
    constexpr unsigned KPerBlock  = 128;
    constexpr unsigned CPerBlock  = 8;
    constexpr unsigned HoPerBlock = 2;
    constexpr unsigned WoPerBlock = 2;

    constexpr unsigned NPerThread  = 4;
    constexpr unsigned KPerThread  = 16;
    constexpr unsigned CPerThread  = 1;
    constexpr unsigned HoPerThread = 1;
    constexpr unsigned WoPerThread = 1;

    constexpr unsigned InBlockCopy_ThreadPerDimC = 8;
    constexpr unsigned InBlockCopy_ThreadPerDimH = 2;
    constexpr unsigned InBlockCopy_ThreadPerDimW = 2;
    constexpr unsigned InBlockCopy_ThreadPerDimN = 4;
    constexpr unsigned InBlockCopyDataPerRead    = 4;

    constexpr unsigned WeiBlockCopyDataPerRead = 4;

    constexpr unsigned GemmMPerThreadSubC = 4;
    constexpr unsigned GemmNPerThreadSubC = 4;
    constexpr unsigned GemmMLevel0Cluster = 4;
    constexpr unsigned GemmNLevel0Cluster = 2;
    constexpr unsigned GemmMLevel1Cluster = 2;
    constexpr unsigned GemmNLevel1Cluster = 4;
    constexpr unsigned GemmKPerThreadLoop = 1;

    constexpr unsigned OutThreadCopyDataPerWrite = 2;

    constexpr unsigned BlockSize = 128;
#endif

    constexpr unsigned GridSize =
        ((N + NPerBlock - 1) / NPerBlock) * ((K + KPerBlock - 1) / KPerBlock) *
        ((Ho + HoPerBlock - 1) / HoPerBlock) * ((Wo + WoPerBlock - 1) / WoPerBlock);

    printf("%s: BlockSize %u, GridSize %u \n", __func__, BlockSize, GridSize);

    for(unsigned i = 0; i < nrepeat; ++i)
    {
        float time = launch_kernel(
            gridwise_implicit_gemm_convolution_1_chwn_csrk_khwn<GridSize,
                                                                BlockSize,
                                                                T,
                                                                decltype(in_chwn_desc),
                                                                decltype(wei_csrk_desc),
                                                                decltype(out_khwn_desc),
                                                                NPerBlock,
                                                                KPerBlock,
                                                                CPerBlock,
                                                                HoPerBlock,
                                                                WoPerBlock,
                                                                NPerThread,
                                                                KPerThread,
                                                                HoPerThread,
                                                                WoPerThread,
                                                                Sequence<InBlockCopy_ThreadPerDimC,
                                                                         InBlockCopy_ThreadPerDimH,
                                                                         InBlockCopy_ThreadPerDimW,
                                                                         InBlockCopy_ThreadPerDimN>,
                                                                InBlockCopyDataPerRead,
                                                                WeiBlockCopyDataPerRead,
                                                                GemmMPerThreadSubC,
                                                                GemmNPerThreadSubC,
                                                                GemmMLevel0Cluster,
                                                                GemmNLevel0Cluster,
                                                                GemmMLevel1Cluster,
                                                                GemmNLevel1Cluster,
                                                                GemmKPerThreadLoop,
                                                                OutThreadCopyDataPerWrite>,
            dim3(GridSize),
            dim3(BlockSize),
            static_cast<T*>(in_chwn_device_buf.GetDeviceBuffer()),
            static_cast<T*>(wei_csrk_device_buf.GetDeviceBuffer()),
            static_cast<T*>(out_khwn_device_buf.GetDeviceBuffer()));

        printf("Elapsed time : %f ms\n", time);
        usleep(std::min(time * 1000, float(10000)));
    }

    out_khwn_device_buf.FromDevice(out_khwn.mData.data());

    // reorder output
    auto f_reorder_khwn2nkhw = [&](auto k, auto ho, auto wo, auto n) {
        out_nkhw(n, k, ho, wo) = out_khwn(k, ho, wo, n);
    };

    make_ParallelTensorFunctor(f_reorder_khwn2nkhw, K, Ho, Wo, N)(
        std::thread::hardware_concurrency());
}
