#pragma once
#include <unistd.h>
#include "device.hpp"
#include "gridwise_convolution_wrapper.hip.hpp"
#include "gridwise_convolution_implicit_gemm_v1_chwn_cyxk_khwn.hip.hpp"
#include "gridwise_convolution_implicit_gemm_v1_chwn_cyxk_khwn_lds_double_buffer.hip.hpp"
#include "gridwise_convolution_implicit_gemm_v1r2_chwn_cyxk_khwn.hip.hpp"

template <class T, class InDesc, class WeiDesc, class OutDesc>
void device_implicit_gemm_convolution_1_chwn_cyxk_khwn(InDesc,
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
    auto wei_cyxk_desc = make_ConstantTensorDescriptor(Sequence<C, Y, X, K>{});
    ostream_ConstantTensorDescriptor(wei_cyxk_desc, std::cout << "wei_cyxk_desc: ");

    Tensor<T> wei_cyxk(make_TensorDescriptor(wei_cyxk_desc));

    auto f_reorder_kcyx2cyxk = [&](auto k, auto c, auto y, auto x) {
        wei_cyxk(c, y, x, k) = wei_kcyx(k, c, y, x);
    };

    make_ParallelTensorFunctor(f_reorder_kcyx2cyxk, K, C, Y, X)(
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
    DeviceMem wei_cyxk_device_buf(data_sz * wei_cyxk.mDesc.GetElementSpace());
    DeviceMem out_khwn_device_buf(data_sz * out_khwn.mDesc.GetElementSpace());

    in_chwn_device_buf.ToDevice(in_chwn.mData.data());
    wei_cyxk_device_buf.ToDevice(wei_cyxk.mData.data());
    out_khwn_device_buf.ToDevice(out_khwn.mData.data());

#if 0
    // for 3x3, 34x34
    constexpr index_t NPerBlock  = 16;
    constexpr index_t KPerBlock  = 64;
    constexpr index_t CPerBlock  = 4;
    constexpr index_t HoPerBlock = 2;
    constexpr index_t WoPerBlock = 4;

    constexpr index_t NPerThread  = 4;
    constexpr index_t KPerThread  = 8;
    constexpr index_t HoPerThread = 1;
    constexpr index_t WoPerThread = 2;

    constexpr index_t InBlockCopy_ThreadPerDimC = 4;
    constexpr index_t InBlockCopy_ThreadPerDimH = 4;
    constexpr index_t InBlockCopy_ThreadPerDimW = 2;
    constexpr index_t InBlockCopy_ThreadPerDimN = 4;
    constexpr index_t InBlockCopyDataPerRead    = 4;

    constexpr index_t WeiBlockCopyDataPerRead = 4;

    constexpr index_t GemmMPerThreadSubC = 4;
    constexpr index_t GemmNPerThreadSubC = 4;
    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 4;
    constexpr index_t GemmKPerThreadLoop = 1;
    constexpr index_t GemmDataPerReadA   = 4;
    constexpr index_t GemmDataPerReadB   = 4;

    constexpr index_t OutThreadCopyDataPerWrite = 2;

    constexpr index_t BlockSize = 128;
#elif 0
    // for 5x5, 36x36
    constexpr index_t NPerBlock  = 16;
    constexpr index_t KPerBlock  = 64;
    constexpr index_t CPerBlock  = 2;
    constexpr index_t HoPerBlock = 2;
    constexpr index_t WoPerBlock = 4;

    constexpr index_t NPerThread  = 8;
    constexpr index_t KPerThread  = 8;
    constexpr index_t HoPerThread = 1;
    constexpr index_t WoPerThread = 1;

    constexpr index_t WeiBlockCopyThreadPerDim0 = 4;
    constexpr index_t WeiBlockCopyThreadPerDim1 = 32;

    constexpr index_t InBlockCopy_ThreadPerDimC = 2;
    constexpr index_t InBlockCopy_ThreadPerDimH = 2;
    constexpr index_t InBlockCopy_ThreadPerDimW = 4;
    constexpr index_t InBlockCopy_ThreadPerDimN = 4;
    constexpr index_t InBlockCopyDataPerRead    = 4;

    constexpr index_t WeiBlockCopyDataPerRead = 2;

    constexpr index_t GemmMPerThreadSubC = 4;
    constexpr index_t GemmNPerThreadSubC = 4;
    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 4;
    constexpr index_t GemmKPerThreadLoop = 1;

    constexpr index_t OutThreadCopyDataPerWrite = 2;

    constexpr index_t BlockSize = 128;
#elif 0
    // 3x3 58x58
    constexpr index_t NPerBlock  = 16;
    constexpr index_t KPerBlock  = 64;
    constexpr index_t CPerBlock  = 4;
    constexpr index_t HoPerBlock = 2;
    constexpr index_t WoPerBlock = 4;

    constexpr index_t NPerThread  = 4;
    constexpr index_t KPerThread  = 16;
    constexpr index_t CPerThread  = 1;
    constexpr index_t HoPerThread = 1;
    constexpr index_t WoPerThread = 1;

    constexpr index_t WeiBlockCopyThreadPerDim0 = 4;
    constexpr index_t WeiBlockCopyThreadPerDim1 = 32;

    constexpr index_t InBlockCopyDataPerRead  = 2; // not used, yet
    constexpr index_t WeiBlockCopyDataPerRead = 4;

    constexpr index_t BlockSize = 128;
#elif 0
    // for 7x7, 38x38
    constexpr index_t NPerBlock  = 16;
    constexpr index_t KPerBlock  = 128;
    constexpr index_t CPerBlock  = 8;
    constexpr index_t HoPerBlock = 2;
    constexpr index_t WoPerBlock = 2;

    constexpr index_t NPerThread  = 4;
    constexpr index_t KPerThread  = 8;
    constexpr index_t HoPerThread = 1;
    constexpr index_t WoPerThread = 2;

    constexpr index_t GemmMPerThreadSubC = 4;
    constexpr index_t GemmNPerThreadSubC = 4;
    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 2;
    constexpr index_t GemmKPerThreadLoop = 1;

    constexpr index_t InBlockCopy_ThreadPerDimC = 2;
    constexpr index_t InBlockCopy_ThreadPerDimH = 4;
    constexpr index_t InBlockCopy_ThreadPerDimW = 4;
    constexpr index_t InBlockCopy_ThreadPerDimN = 4;
    constexpr index_t InBlockCopyDataPerRead    = 4;

    constexpr index_t WeiBlockCopyDataPerRead   = 4;
    constexpr index_t OutThreadCopyDataPerWrite = 4;

    constexpr index_t BlockSize = 128;
#elif 0
    // for 3x3, 56x56, v1, Pacal
    constexpr index_t NPerBlock  = 32;
    constexpr index_t KPerBlock  = 64;
    constexpr index_t CPerBlock  = 4;
    constexpr index_t HoPerBlock = 2;
    constexpr index_t WoPerBlock = 2;

    constexpr index_t NPerThread  = 4;
    constexpr index_t KPerThread  = 8;
    constexpr index_t HoPerThread = 1;
    constexpr index_t WoPerThread = 2;

    constexpr index_t InBlockCopy_ThreadPerDimC = 1;
    constexpr index_t InBlockCopy_ThreadPerDimH = 4;
    constexpr index_t InBlockCopy_ThreadPerDimW = 4;
    constexpr index_t InBlockCopy_ThreadPerDimN = 8;
    constexpr index_t InBlockCopyDataPerRead    = 4;

    constexpr index_t WeiBlockCopyDataPerRead = 4;

    constexpr index_t GemmMPerThreadSubC = 4;
    constexpr index_t GemmNPerThreadSubC = 4;
    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 4;
    constexpr index_t GemmKPerThreadLoop = 1;

    constexpr index_t OutThreadCopyDataPerWrite = 2;

    constexpr index_t BlockSize = 128;
#elif 0
    // for 3x3, 56x56, v1r2, Pascal
    // for 3x3, 34x34, v1r2, Pascal
    constexpr index_t NPerBlock  = 16;
    constexpr index_t KPerBlock  = 128;
    constexpr index_t CPerBlock  = 8;
    constexpr index_t HoPerBlock = 2;
    constexpr index_t WoPerBlock = 2;

    constexpr index_t NPerThread  = 4;
    constexpr index_t KPerThread  = 8;
    constexpr index_t HoPerThread = 1;
    constexpr index_t WoPerThread = 2;

    constexpr index_t GemmMPerThreadSubC = 4;
    constexpr index_t GemmNPerThreadSubC = 4;
    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 2;
    constexpr index_t GemmKPerThreadLoop = 1;
    constexpr index_t GemmDataPerReadA   = 1;
    constexpr index_t GemmDataPerReadB   = 1;

    constexpr index_t InBlockCopy_ThreadPerDimC = 2;
    constexpr index_t InBlockCopy_ThreadPerDimH = 4;
    constexpr index_t InBlockCopy_ThreadPerDimW = 4;
    constexpr index_t InBlockCopy_ThreadPerDimN = 4;
    constexpr index_t InBlockCopyDataPerRead    = 4;

    constexpr index_t WeiBlockCopyDataPerRead   = 4;
    constexpr index_t OutThreadCopyDataPerWrite = 4;

    constexpr index_t BlockSize = 128;
#elif 1
    // for 3x3, 28x28, v1, Pacal
    constexpr index_t NPerBlock  = 32;
    constexpr index_t KPerBlock  = 64;
    constexpr index_t CPerBlock  = 4;
    constexpr index_t HoPerBlock = 2;
    constexpr index_t WoPerBlock = 2;

    constexpr index_t NPerThread  = 4;
    constexpr index_t KPerThread  = 8;
    constexpr index_t HoPerThread = 1;
    constexpr index_t WoPerThread = 2;

    constexpr index_t InBlockCopy_ThreadPerDimC = 1;
    constexpr index_t InBlockCopy_ThreadPerDimH = 4;
    constexpr index_t InBlockCopy_ThreadPerDimW = 4;
    constexpr index_t InBlockCopy_ThreadPerDimN = 8;
    constexpr index_t InBlockCopyDataPerRead    = 4;

    constexpr index_t WeiBlockCopyDataPerRead = 4;

    constexpr index_t GemmMPerThreadSubC = 4;
    constexpr index_t GemmNPerThreadSubC = 4;
    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 4;
    constexpr index_t GemmKPerThreadLoop = 1;
    constexpr index_t GemmDataPerReadA   = 4;
    constexpr index_t GemmDataPerReadB   = 4;

    constexpr index_t OutThreadCopyDataPerWrite = 2;

    constexpr index_t BlockSize = 128;
#elif 0
    // for 1x1, 28x28
    constexpr index_t NPerBlock  = 16;
    constexpr index_t KPerBlock  = 128;
    constexpr index_t CPerBlock  = 8;
    constexpr index_t HoPerBlock = 2;
    constexpr index_t WoPerBlock = 2;

    constexpr index_t NPerThread  = 4;
    constexpr index_t KPerThread  = 16;
    constexpr index_t CPerThread  = 1;
    constexpr index_t HoPerThread = 1;
    constexpr index_t WoPerThread = 1;

    constexpr index_t InBlockCopy_ThreadPerDimC = 8;
    constexpr index_t InBlockCopy_ThreadPerDimH = 2;
    constexpr index_t InBlockCopy_ThreadPerDimW = 2;
    constexpr index_t InBlockCopy_ThreadPerDimN = 4;
    constexpr index_t InBlockCopyDataPerRead    = 4;

    constexpr index_t WeiBlockCopyDataPerRead = 4;

    constexpr index_t GemmMPerThreadSubC = 4;
    constexpr index_t GemmNPerThreadSubC = 4;
    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 4;
    constexpr index_t GemmKPerThreadLoop = 1;

    constexpr index_t OutThreadCopyDataPerWrite = 2;

    constexpr index_t BlockSize = 128;
#elif 1
    // for 1x1, 14x14, Pascal
    constexpr index_t NPerBlock  = 16;
    constexpr index_t KPerBlock  = 128;
    constexpr index_t CPerBlock  = 8;
    constexpr index_t HoPerBlock = 2;
    constexpr index_t WoPerBlock = 2;

    constexpr index_t NPerThread  = 8;
    constexpr index_t KPerThread  = 8;
    constexpr index_t HoPerThread = 1;
    constexpr index_t WoPerThread = 1;

    constexpr index_t GemmMPerThreadSubC = 4;
    constexpr index_t GemmNPerThreadSubC = 4;
    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 2;
    constexpr index_t GemmKPerThreadLoop = 1;

    constexpr index_t InBlockCopy_ThreadPerDimC = 8;
    constexpr index_t InBlockCopy_ThreadPerDimH = 2;
    constexpr index_t InBlockCopy_ThreadPerDimW = 2;
    constexpr index_t InBlockCopy_ThreadPerDimN = 4;
    constexpr index_t InBlockCopyDataPerRead    = 4;

    constexpr index_t WeiBlockCopyDataPerRead   = 4;
    constexpr index_t OutThreadCopyDataPerWrite = 2;

    constexpr index_t BlockSize = 128;
#endif

    constexpr index_t GridSize =
        ((N + NPerBlock - 1) / NPerBlock) * ((K + KPerBlock - 1) / KPerBlock) *
        ((Ho + HoPerBlock - 1) / HoPerBlock) * ((Wo + WoPerBlock - 1) / WoPerBlock);

    printf("%s: BlockSize %u, GridSize %u \n", __func__, BlockSize, GridSize);

    for(index_t i = 0; i < nrepeat; ++i)
    {
        constexpr auto gridwise_conv =
#if 1
            GridwiseConvolutionImplicitGemm_v1_chwn_cyxk_khwn
#elif 1
            GridwiseConvolutionImplicitGemm_v1r2_chwn_cyxk_khwn
#elif 0
            GridwiseConvolutionImplicitGemm_v1_chwn_cyxk_khwn_lds_double_buffer
#endif
            <GridSize,
             BlockSize,
             T,
             decltype(in_chwn_desc),
             decltype(wei_cyxk_desc),
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
             GemmMPerThreadSubC,
             GemmNPerThreadSubC,
             GemmMLevel0Cluster,
             GemmNLevel0Cluster,
             GemmMLevel1Cluster,
             GemmNLevel1Cluster,
             GemmKPerThreadLoop,
             GemmDataPerReadA,
             GemmDataPerReadB,
             Sequence<InBlockCopy_ThreadPerDimC,
                      InBlockCopy_ThreadPerDimH,
                      InBlockCopy_ThreadPerDimW,
                      InBlockCopy_ThreadPerDimN>,
             InBlockCopyDataPerRead,
             WeiBlockCopyDataPerRead,
             OutThreadCopyDataPerWrite>{};

        float time = launch_kernel(run_gridwise_convolution<decltype(gridwise_conv), T>,
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

    // reorder output
    auto f_reorder_khwn2nkhw = [&](auto k, auto ho, auto wo, auto n) {
        out_nkhw(n, k, ho, wo) = out_khwn(k, ho, wo, n);
    };

    make_ParallelTensorFunctor(f_reorder_khwn2nkhw, K, Ho, Wo, N)(
        std::thread::hardware_concurrency());
}
