#pragma once
#include <unistd.h>
#include "device.hpp"
#include "gridwise_implicit_gemm_convolution_1_nchw_kcsr_nkhw.hip.hpp"

template <class T, class InDesc, class WeiDesc, class OutDesc>
void device_implicit_gemm_convolution_1_nchw_kcsr_nkhw(InDesc,
                                                       const Tensor<T>& in,
                                                       WeiDesc,
                                                       const Tensor<T>& wei,
                                                       OutDesc,
                                                       Tensor<T>& out,
                                                       unsigned nrepeat)
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
    // 3x3, 34x34
    constexpr unsigned NPerBlock  = 1;
    constexpr unsigned KPerBlock  = 64;
    constexpr unsigned CPerBlock  = 2;
    constexpr unsigned HoPerBlock = 4;
    constexpr unsigned WoPerBlock = 32;

    constexpr unsigned KPerThread  = 16;
    constexpr unsigned CPerThread  = 1;
    constexpr unsigned HoPerThread = 2;
    constexpr unsigned WoPerThread = 2;

    constexpr unsigned BlockSize = 128;
#endif

    constexpr unsigned GridSize =
        (out_desc.GetLength(I0) / NPerBlock) * (out_desc.GetLength(I1) / KPerBlock) *
        (out_desc.GetLength(I2) / HoPerBlock) * (out_desc.GetLength(I3) / WoPerBlock);

    printf("%s: BlockSize %u, GridSize %u \n", __func__, BlockSize, GridSize);

    for(unsigned i = 0; i < nrepeat; ++i)
    {
        float time = launch_kernel(gridwise_implicit_gemm_convolution_1_nchw_kcsr_nkhw<GridSize,
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
                                                                                       KPerThread,
                                                                                       CPerThread,
                                                                                       HoPerThread,
                                                                                       WoPerThread>,
                                   dim3(GridSize),
                                   dim3(BlockSize),
                                   static_cast<T*>(in_device_buf.GetDeviceBuffer()),
                                   static_cast<T*>(wei_device_buf.GetDeviceBuffer()),
                                   static_cast<T*>(out_device_buf.GetDeviceBuffer()));

        printf("Elapsed time : %f ms\n", time);
        usleep(std::min(time * 1000, float(10000)));
    }

    out_device_buf.FromDevice(out.mData.data());
}
