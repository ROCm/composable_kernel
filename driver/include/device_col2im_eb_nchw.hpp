#pragma once
#include <unistd.h>
#include "device.hpp"
#include "tensor.hpp"
#include "gridwise_operation_wrapper.hpp"
#include "gridwise_col2im_eb_nchw.hpp"

template <typename T,
          typename ColDesc,
          typename ImgDesc,
          typename FilterSizes,
          typename OutputSizes,
          typename ConvStrides,
          typename ConvDilations,
          typename LeftPads,
          typename RightPads>
void device_col2im_eb_nchw(ColDesc,
                           const Tensor<T>& col_eb,
                           ImgDesc,
                           Tensor<T>& img_nchw,
                           FilterSizes,
                           OutputSizes,
                           ConvStrides,
                           ConvDilations,
                           LeftPads,
                           RightPads,
                           std::size_t nrepeat)
{
    using namespace ck;

    constexpr auto col_eb_desc   = ColDesc{};
    constexpr auto img_nchw_desc = ImgDesc{};

    constexpr index_t N  = img_nchw_desc.GetLengths()[0];
    constexpr index_t C  = img_nchw_desc.GetLengths()[1];
    constexpr index_t Hi = img_nchw_desc.GetLengths()[2];
    constexpr index_t Wi = img_nchw_desc.GetLengths()[3];

    constexpr index_t E = col_eb_desc.GetLengths()[0];
    constexpr index_t B = col_eb_desc.GetLengths()[1];

    std::size_t data_sz = sizeof(T);
    DeviceMem col_eb_device_buf(data_sz * col_eb.mDesc.GetElementSpace());
    DeviceMem img_nchw_device_buf(data_sz * img_nchw.mDesc.GetElementSpace());

    col_eb_device_buf.ToDevice(col_eb.mData.data());
    img_nchw_device_buf.ToDevice(img_nchw.mData.data());

#if 1
    constexpr index_t BlockSize = 256;

    constexpr index_t EPerBlock = 128;
    constexpr index_t BPerBlock = 128;

    using BlockCopySubLengths_E_B            = Sequence<8, 8>;
    using BlockCopyClusterLengths_E_B        = Sequence<16, 16>;
    using BlockCopyThreadClusterArrangeOrder = Sequence<0, 1>; // [E, B]
    using BlockCopySrcAccessOrder            = Sequence<0, 1>; // [E, B]
    using BlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, B]

    constexpr index_t BlockCopyDataPerAccess_B = 1;
#endif

    constexpr index_t GridSize =
        ((E + EPerBlock - 1) / EPerBlock) * ((B + BPerBlock - 1) / BPerBlock);

    printf("%s: BlockSize %u, GridSize %u \n", __func__, BlockSize, GridSize);

    constexpr auto gridwise_col2im = GridwiseCol2Im_eb_nchw<GridSize,
                                                            BlockSize,
                                                            T,
                                                            ColDesc,
                                                            ImgDesc,
                                                            FilterSizes,
                                                            OutputSizes,
                                                            ConvStrides,
                                                            ConvDilations,
                                                            LeftPads,
                                                            RightPads,
                                                            EPerBlock,
                                                            BPerBlock,
                                                            BlockCopySubLengths_E_B,
                                                            BlockCopyClusterLengths_E_B,
                                                            BlockCopyThreadClusterArrangeOrder,
                                                            BlockCopySrcAccessOrder,
                                                            BlockCopyDstAccessOrder,
                                                            BlockCopyDataPerAccess_B>{};

    for(index_t i = 0; i < nrepeat; ++i)
    {
        float time = launch_kernel(run_gridwise_operation<decltype(gridwise_col2im),
                                                          const T* const __restrict__,
                                                          T* const __restrict__>,
                                   dim3(GridSize),
                                   dim3(BlockSize),
                                   0,
                                   gridwise_col2im,
                                   const_cast<const T* const __restrict__>(
                                       static_cast<T*>(col_eb_device_buf.GetDeviceBuffer())),
                                   const_cast<T* const __restrict__>(
                                       static_cast<T*>(img_nchw_device_buf.GetDeviceBuffer())));

        printf("Elapsed time : %f ms\n", time);
        usleep(std::min(time * 1000, float(10000)));
    }

    img_nchw_device_buf.FromDevice(img_nchw.mData.data());
}
