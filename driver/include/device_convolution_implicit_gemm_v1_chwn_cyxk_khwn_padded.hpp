#pragma once
#include <unistd.h>
#include "device.hpp"
#include "tensor.hpp"
#include "gridwise_convolution_implicit_gemm_v1r3_chwn_cyxk_khwn_padded.hpp"

using namespace ck;

template <typename T, class InDesc, class WeiDesc, class OutDesc, class LeftPads, class RightPads>
void device_convolution_implicit_gemm_v1_chwn_cyxk_khwn_padded(InDesc,
                                                               const Tensor<T>& in_nchw,
                                                               WeiDesc,
                                                               const Tensor<T>& wei_kcyx,
                                                               OutDesc,
                                                               Tensor<T>& out_nkhw,
                                                               LeftPads,
                                                               RightPads,
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
    auto wei_cyxk_desc = make_ConstantTensorDescriptor_packed(Sequence<C, Y, X, K>{});
    ostream_ConstantTensorDescriptor(wei_cyxk_desc, std::cout << "wei_cyxk_desc: ");

    Tensor<T> wei_cyxk(make_TensorDescriptor(wei_cyxk_desc));

    auto f_reorder_kcyx2cyxk = [&](auto k, auto c, auto y, auto x) {
        wei_cyxk(c, y, x, k) = wei_kcyx(k, c, y, x);
    };

    make_ParallelTensorFunctor(f_reorder_kcyx2cyxk, K, C, Y, X)(
        std::thread::hardware_concurrency());

    // reorder input
    auto in_chwn_desc = make_ConstantTensorDescriptor_packed(Sequence<C, Hi, Wi, N>{});
    ostream_ConstantTensorDescriptor(in_chwn_desc, std::cout << "in_chwn_desc: ");

    Tensor<T> in_chwn(make_TensorDescriptor(in_chwn_desc));

    auto f_reorder_nchw2chwn = [&](auto n, auto c, auto hi, auto wi) {
        in_chwn(c, hi, wi, n) = in_nchw(n, c, hi, wi);
    };

    make_ParallelTensorFunctor(f_reorder_nchw2chwn, N, C, Hi, Wi)(
        std::thread::hardware_concurrency());

    // output
    auto out_khwn_desc = make_ConstantTensorDescriptor_packed(Sequence<K, Ho, Wo, N>{});
    ostream_ConstantTensorDescriptor(out_khwn_desc, std::cout << "out_khwn_desc: ");

    Tensor<T> out_khwn(make_TensorDescriptor(out_khwn_desc));

    std::size_t data_sz = sizeof(T);
    DeviceMem in_chwn_device_buf(data_sz * in_chwn.mDesc.GetElementSpace());
    DeviceMem wei_cyxk_device_buf(data_sz * wei_cyxk.mDesc.GetElementSpace());
    DeviceMem out_khwn_device_buf(data_sz * out_khwn.mDesc.GetElementSpace());

    in_chwn_device_buf.ToDevice(in_chwn.mData.data());
    wei_cyxk_device_buf.ToDevice(wei_cyxk.mData.data());
    out_khwn_device_buf.ToDevice(out_khwn.mData.data());

#if 1
    // v1r3, 3x3, 32x32, 1x1 pad
    constexpr index_t BlockSize = 256;

    constexpr index_t NPerBlock  = 32;
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
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 2;
    constexpr index_t GemmKPerThreadLoop = 1;
    constexpr index_t GemmDataPerReadA   = 4;
    constexpr index_t GemmDataPerReadB   = 4;

    using InBlockCopySubLengths_CHWN             = Sequence<1, 1, 1, 4>;
    using InBlockCopyClusterLengths_CHWN         = Sequence<8, 2, 2, 8>;
    constexpr index_t InBlockCopyDataPerAccess_N = 4;

    using WeiBlockCopySubLengths_CK               = Sequence<1, 4>;
    using WeiBlockCopyClusterLengths_CK           = Sequence<8, 32>;
    constexpr index_t WeiBlockCopyDataPerAccess_K = 4;

    constexpr index_t OutThreadCopyDataPerAccess_N = 4;
#endif

#if 1 // debug
    constexpr index_t GridSize =
        (N / NPerBlock) * (K / KPerBlock) * (Ho / HoPerBlock) * (Wo / WoPerBlock);
#else
    constexpr index_t GridSize = 1;
#endif

    printf("%s: BlockSize %u, GridSize %u \n", __func__, BlockSize, GridSize);

    constexpr auto gridwise_conv =
        GridwiseConvolutionImplicitGemm_v1r3_chwn_cyxk_khwn_padded<GridSize,
                                                                   BlockSize,
                                                                   T,
                                                                   decltype(in_chwn_desc),
                                                                   decltype(wei_cyxk_desc),
                                                                   decltype(out_khwn_desc),
                                                                   LeftPads,
                                                                   RightPads,
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
                                                                   InBlockCopySubLengths_CHWN,
                                                                   InBlockCopyClusterLengths_CHWN,
                                                                   InBlockCopyDataPerAccess_N,
                                                                   WeiBlockCopySubLengths_CK,
                                                                   WeiBlockCopyClusterLengths_CK,
                                                                   WeiBlockCopyDataPerAccess_K,
                                                                   OutThreadCopyDataPerAccess_N>{};

    for(index_t i = 0; i < nrepeat; ++i)
    {
        float time = launch_kernel(run_gridwise_convolution_kernel<decltype(gridwise_conv), T>,
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
