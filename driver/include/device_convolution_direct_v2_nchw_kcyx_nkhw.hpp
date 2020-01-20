#pragma once
#include <unistd.h>
#include "device.hpp"
#include "tensor.hpp"
#include "gridwise_convolution_kernel_wrapper.hpp"
#include "gridwise_convolution_direct_v2_nchw_kcyx_nkhw.hpp"

using namespace ck;

template <class T, class InDesc, class WeiDesc, class OutDesc>
void device_convolution_direct_v2_nchw_kcyx_nkhw(InDesc,
                                                 const Tensor<T>& in,
                                                 WeiDesc,
                                                 const Tensor<T>& wei,
                                                 OutDesc,
                                                 Tensor<T>& out,
                                                 index_t nrepeat)
{
    std::size_t data_sz = sizeof(T);
    DeviceMem in_device_buf(data_sz * in.mDesc.GetElementSpace());
    DeviceMem wei_device_buf(data_sz * wei.mDesc.GetElementSpace());
    DeviceMem out_device_buf(data_sz * out.mDesc.GetElementSpace());

    int num_thread = std::thread::hardware_concurrency();

    in_device_buf.ToDevice(in.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());
    out_device_buf.ToDevice(out.mData.data());

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_desc  = InDesc{};
    constexpr auto wei_desc = WeiDesc{};
    constexpr auto out_desc = OutDesc{};

#if 1
    // 3x3, 34x34, 128 thread
    constexpr index_t NPerBlock  = 2;
    constexpr index_t KPerBlock  = 32;
    constexpr index_t CPerBlock  = 4;
    constexpr index_t HoPerBlock = 2;
    constexpr index_t WoPerBlock = 32;

    constexpr index_t NPerThread  = 2;
    constexpr index_t KPerThread  = 4;
    constexpr index_t CPerThread  = 2;
    constexpr index_t HoPerThread = 2;
    constexpr index_t WoPerThread = 2;

    constexpr index_t InBlockCopyDataPerRead  = 1;
    constexpr index_t WeiBlockCopyDataPerRead = 1;

    constexpr index_t BlockSize = 128;
#endif

    constexpr index_t GridSize =
        (out_desc.GetLength(I0) / NPerBlock) * (out_desc.GetLength(I1) / KPerBlock) *
        (out_desc.GetLength(I2) / HoPerBlock) * (out_desc.GetLength(I3) / WoPerBlock);

    printf("%s: BlockSize %u, GridSize %u \n", __func__, BlockSize, GridSize);

    for(index_t i = 0; i < nrepeat; ++i)
    {
        using gridwise_conv = GridwiseConvolutionDirect_v2_nchw_kcyx_nkhw<GridSize,
                                                                          BlockSize,
                                                                          T,
                                                                          InDesc,
                                                                          WeiDesc,
                                                                          OutDesc,
                                                                          NPerBlock,
                                                                          KPerBlock,
                                                                          CPerBlock,
                                                                          HoPerBlock,
                                                                          WoPerBlock,
                                                                          NPerThread,
                                                                          KPerThread,
                                                                          CPerThread,
                                                                          HoPerThread,
                                                                          WoPerThread,
                                                                          InBlockCopyDataPerRead,
                                                                          WeiBlockCopyDataPerRead>;
        float time = launch_and_time_kernel(run_gridwise_convolution_kernel<gridwise_conv, T>,
                                            dim3(GridSize),
                                            dim3(BlockSize),
                                            0,
                                            static_cast<T*>(in_device_buf.GetDeviceBuffer()),
                                            static_cast<T*>(wei_device_buf.GetDeviceBuffer()),
                                            static_cast<T*>(out_device_buf.GetDeviceBuffer()));

        printf("Elapsed time : %f ms\n", time);
        usleep(std::min(time * 1000, float(10000)));
    }

    out_device_buf.FromDevice(out.mData.data());
}
