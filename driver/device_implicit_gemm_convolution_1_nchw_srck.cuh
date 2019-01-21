#pragma once
#include "gridwise_implicit_gemm_convolution_1_nchw_srck.cuh"

template <class T, class InDesc, class WeiDesc, class OutDesc>
void device_implicit_gemm_convolution_1_nchw_srck(InDesc,
                                                  const Tensor<T>& in_nchw,
                                                  WeiDesc,
                                                  const Tensor<T>& wei_kcsr,
                                                  OutDesc,
                                                  Tensor<T>& out_nkhw)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_nchw_desc  = InDesc{};
    constexpr auto wei_kcsr_desc = WeiDesc{};
    constexpr auto out_nkhw_desc = OutDesc{};

    constexpr unsigned N  = out_nkhw_desc.GetLength(I0);
    constexpr unsigned Ho = out_nkhw_desc.GetLength(I2);
    constexpr unsigned Wo = out_nkhw_desc.GetLength(I3);

    constexpr unsigned K = wei_kcsr_desc.GetLength(I0);
    constexpr unsigned C = wei_kcsr_desc.GetLength(I1);
    constexpr unsigned S = wei_kcsr_desc.GetLength(I2);
    constexpr unsigned R = wei_kcsr_desc.GetLength(I3);

    auto wei_srck_desc = make_ConstantTensorDescriptor(Sequence<S, R, C, K>{});
    ostream_ConstantTensorDescriptor(wei_srck_desc, std::cout << "wei_srck_desc: ");

    Tensor<T> wei_srck(make_TensorDescriptor(wei_srck_desc));

    auto f_reorder_kcsr2srck = [&](auto k, auto c, auto s, auto r) {
        wei_srck(s, r, c, k) = wei_kcsr(k, c, s, r);
    };

    make_ParallelTensorFunctor(f_reorder_kcsr2srck, K, C, S, R)(
        std::thread::hardware_concurrency());

    std::size_t data_sz = sizeof(T);
    DeviceMem in_nchw_device_buf(data_sz * in_nchw.mDesc.GetElementSpace());
    DeviceMem wei_srck_device_buf(data_sz * wei_srck.mDesc.GetElementSpace());
    DeviceMem out_nkhw_device_buf(data_sz * out_nkhw.mDesc.GetElementSpace());

    int num_thread = std::thread::hardware_concurrency();

    in_nchw_device_buf.ToDevice(in_nchw.mData.data());
    wei_srck_device_buf.ToDevice(wei_srck.mData.data());
    out_nkhw_device_buf.ToDevice(out_nkhw.mData.data());

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
#elif 0
    constexpr unsigned NPerBlock  = 1;
    constexpr unsigned KPerBlock  = 64;
    constexpr unsigned CPerBlock  = 2;
    constexpr unsigned HoPerBlock = 4;
    constexpr unsigned WoPerBlock = 32;

    constexpr unsigned KPerThread  = 8;
    constexpr unsigned CPerThread  = 1;
    constexpr unsigned HoPerThread = 2;
    constexpr unsigned WoPerThread = 4;

    constexpr unsigned BlockSize = 128;
#elif 1
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
#elif 0
    constexpr unsigned NPerBlock  = 2;
    constexpr unsigned KPerBlock  = 64;
    constexpr unsigned CPerBlock  = 2;
    constexpr unsigned HoPerBlock = 2;
    constexpr unsigned WoPerBlock = 32;

    constexpr unsigned KPerThread  = 16;
    constexpr unsigned CPerThread  = 1;
    constexpr unsigned HoPerThread = 2;
    constexpr unsigned WoPerThread = 1;

    constexpr unsigned BlockSize = 128;
#endif

    constexpr unsigned GridSize =
        ((N + NPerBlock - 1) / NPerBlock) * ((K + KPerBlock - 1) / KPerBlock) *
        ((Ho + HoPerBlock - 1) / HoPerBlock) * ((Wo + WoPerBlock - 1) / WoPerBlock);

    dim3 block_dim(BlockSize);
    dim3 grid_dim(GridSize);

    printf("%s: BlockSize %u, GridSize %u \n", __func__, BlockSize, GridSize);

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventRecord(start, 0);

    gridwise_implicit_gemm_convolution_1_nchw_srck<GridSize,
                                                   BlockSize,
                                                   T,
                                                   decltype(in_nchw_desc),
                                                   decltype(wei_srck_desc),
                                                   decltype(out_nkhw_desc),
                                                   NPerBlock,
                                                   KPerBlock,
                                                   CPerBlock,
                                                   HoPerBlock,
                                                   WoPerBlock,
                                                   KPerThread,
                                                   CPerThread,
                                                   HoPerThread,
                                                   WoPerThread>
        <<<grid_dim, block_dim>>>(in_nchw_desc,
                                  static_cast<T*>(in_nchw_device_buf.GetDeviceBuffer()),
                                  wei_srck_desc,
                                  static_cast<T*>(wei_srck_device_buf.GetDeviceBuffer()),
                                  out_nkhw_desc,
                                  static_cast<T*>(out_nkhw_device_buf.GetDeviceBuffer()));

    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed time : %f ms\n", elapsedTime);

    checkCudaErrors(cudaGetLastError());
    out_nkhw_device_buf.FromDevice(out_nkhw.mData.data());
}
