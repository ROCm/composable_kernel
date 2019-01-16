#pragma once
#include "gridwise_implicit_gemm_convolution_nchw_kcsr.cuh"

template <class T, class InDesc, class WeiDesc, class OutDesc>
void device_implicit_gemm_convolution_nchw_kcsr(
    InDesc, const Tensor<T>& in, WeiDesc, const Tensor<T>& wei, OutDesc, Tensor<T>& out)
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

#if 0
    constexpr unsigned NPerBlock  = 1;
    constexpr unsigned KPerBlock  = 1;
    constexpr unsigned CPerBlock  = 1;
    constexpr unsigned HoPerBlock = 2;
    constexpr unsigned WoPerBlock = 32;

    constexpr unsigned KPerThread  = 1;
    constexpr unsigned CPerThread  = 1;
    constexpr unsigned HoPerThread = 2;
    constexpr unsigned WoPerThread = 2;

    constexpr unsigned BlockSize = 16;
#elif 1
    constexpr unsigned NPerBlock  = 2;
    constexpr unsigned KPerBlock  = 32;
    constexpr unsigned CPerBlock  = 4;
    constexpr unsigned HoPerBlock = 2;
    constexpr unsigned WoPerBlock = 32;

    constexpr unsigned KPerThread  = 4;
    constexpr unsigned CPerThread  = 2;
    constexpr unsigned HoPerThread = 2;
    constexpr unsigned WoPerThread = 2;

    constexpr unsigned BlockSize = 128;
#elif 0
    constexpr unsigned NPerBlock  = 2;
    constexpr unsigned KPerBlock  = 64;
    constexpr unsigned CPerBlock  = 4;
    constexpr unsigned HoPerBlock = 2;
    constexpr unsigned WoPerBlock = 32;

    constexpr unsigned KPerThread  = 4;
    constexpr unsigned CPerThread  = 2;
    constexpr unsigned HoPerThread = 2;
    constexpr unsigned WoPerThread = 2;

    constexpr unsigned BlockSize = 256;
#endif

    constexpr unsigned GridSize =
        (out_desc.GetLength(I0) / NPerBlock) * (out_desc.GetLength(I1) / KPerBlock) *
        (out_desc.GetLength(I2) / HoPerBlock) * (out_desc.GetLength(I3) / WoPerBlock);

    dim3 block_dim(BlockSize);
    dim3 grid_dim(GridSize);

    printf("%s: BlockSize %u, GridSize %u \n", __func__, BlockSize, GridSize);

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventRecord(start, 0);

    gridwise_implicit_gemm_convolution_nchw_kcsr<GridSize,
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
                                                 WoPerThread>
        <<<grid_dim, block_dim>>>(InDesc{},
                                  static_cast<T*>(in_device_buf.GetDeviceBuffer()),
                                  WeiDesc{},
                                  static_cast<T*>(wei_device_buf.GetDeviceBuffer()),
                                  OutDesc{},
                                  static_cast<T*>(out_device_buf.GetDeviceBuffer()));

    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed time : %f ms\n", elapsedTime);

    checkCudaErrors(cudaGetLastError());
    out_device_buf.FromDevice(out.mData.data());
}
