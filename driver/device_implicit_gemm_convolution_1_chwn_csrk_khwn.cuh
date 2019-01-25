#pragma once
#include "gridwise_implicit_gemm_convolution_1_chwn_csrk_khwn.cuh"
#include <unistd.h>

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
    constexpr unsigned S = wei_kcsr_desc.GetLength(I2);
    constexpr unsigned R = wei_kcsr_desc.GetLength(I3);

    // reorder weight
    auto wei_csrk_desc = make_ConstantTensorDescriptor(Sequence<C, S, R, K>{});
    ostream_ConstantTensorDescriptor(wei_csrk_desc, std::cout << "wei_csrk_desc: ");

    Tensor<T> wei_csrk(make_TensorDescriptor(wei_csrk_desc));

    auto f_reorder_kcsr2csrk = [&](auto k, auto c, auto s, auto r) {
        wei_csrk(c, s, r, k) = wei_kcsr(k, c, s, r);
    };

    make_ParallelTensorFunctor(f_reorder_kcsr2csrk, K, C, S, R)(
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

#if 0
    constexpr unsigned NPerBlock  = 1;
    constexpr unsigned KPerBlock  = 1;
    constexpr unsigned CPerBlock  = 1;
    constexpr unsigned HoPerBlock = 2;
    constexpr unsigned WoPerBlock = 4;

    constexpr unsigned NPerThread  = 1;
    constexpr unsigned KPerThread  = 1;
    constexpr unsigned CPerThread  = 1;
    constexpr unsigned HoPerThread = 1;
    constexpr unsigned WoPerThread = 1;

    constexpr unsigned BlockSize = 8;
#elif 1
    // for 3x3, 34x34 | 3x3 58x58
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

    constexpr unsigned BlockSize = 128;
#elif 0
    // for 5x5, 36x36
    constexpr unsigned NPerBlock  = 16;
    constexpr unsigned KPerBlock  = 64;
    constexpr unsigned CPerBlock  = 2;
    constexpr unsigned HoPerBlock = 2;
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
#endif

    constexpr unsigned GridSize =
        ((N + NPerBlock - 1) / NPerBlock) * ((K + KPerBlock - 1) / KPerBlock) *
        ((Ho + HoPerBlock - 1) / HoPerBlock) * ((Wo + WoPerBlock - 1) / WoPerBlock);

    dim3 block_dim(BlockSize);
    dim3 grid_dim(GridSize);

    printf("%s: BlockSize %u, GridSize %u \n", __func__, BlockSize, GridSize);

    for(unsigned i = 0; i < nrepeat; ++i)
    {
        cudaEvent_t start, stop;
        float elapsedTime;

        cudaEventCreate(&start);
        cudaEventRecord(start, 0);

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
                                                            CPerThread,
                                                            HoPerThread,
                                                            WoPerThread>
            <<<grid_dim, block_dim>>>(in_chwn_desc,
                                      static_cast<T*>(in_chwn_device_buf.GetDeviceBuffer()),
                                      wei_csrk_desc,
                                      static_cast<T*>(wei_csrk_device_buf.GetDeviceBuffer()),
                                      out_khwn_desc,
                                      static_cast<T*>(out_khwn_device_buf.GetDeviceBuffer()));

        cudaEventCreate(&stop);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsedTime, start, stop);
        printf("Elapsed time : %f ms\n", elapsedTime);

        usleep(10000);
    }

    checkCudaErrors(cudaGetLastError());
    out_khwn_device_buf.FromDevice(out_khwn.mData.data());

    // reorder output
    auto f_reorder_khwn2nkhw = [&](auto k, auto ho, auto wo, auto n) {
        out_nkhw(n, k, ho, wo) = out_khwn(k, ho, wo, n);
    };

    make_ParallelTensorFunctor(f_reorder_khwn2nkhw, K, Ho, Wo, N)(
        std::thread::hardware_concurrency());
}
