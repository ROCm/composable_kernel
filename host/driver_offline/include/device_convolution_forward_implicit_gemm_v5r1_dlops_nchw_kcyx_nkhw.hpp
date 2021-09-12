#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "driver_convolution_forward_implicit_gemm_v5r1_dlops_nchw_kcyx_nkhw_outpad.hpp"

template <typename TInWei,
          ck::index_t InWeiVectorSize,
          typename TAcc,
          typename TOut,
          typename InLengths,
          typename WeiLengths,
          typename OutLengths,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
void device_convolution_forward_implicit_gemm_v5r1_dlops_nchw_kcyx_nkhw(
    const InLengths& in_n_c_hi_wi_lengths,
    const WeiLengths& wei_k_c_y_x_lengths,
    const OutLengths& out_n_k_ho_wo_lengths,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads,
    const Tensor<TInWei>& in_n_c_hi_wi,
    const Tensor<TInWei>& wei_k_c_y_x,
    Tensor<TOut>& out_n_k_ho_wo,
    ck::index_t nrepeat)
{
    using namespace ck;

    std::cout << __func__ << std::endl;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    const auto N = out_n_k_ho_wo_lengths[I0];
    const auto K = out_n_k_ho_wo_lengths[I1];
    const auto C = wei_k_c_y_x_lengths[I1];

    const auto Hi = in_n_c_hi_wi_lengths[I2];
    const auto Wi = in_n_c_hi_wi_lengths[I3];

    const auto Ho = out_n_k_ho_wo_lengths[I2];
    const auto Wo = out_n_k_ho_wo_lengths[I3];

    const auto Y = wei_k_c_y_x_lengths[I2];
    const auto X = wei_k_c_y_x_lengths[I3];

#if 0
    const auto C0 = C / Number<InWeiVectorSize>{};
    const auto C1 = Number<InWeiVectorSize>{};

    const auto K0 = K / Number<InWeiVectorSize>{};
    const auto K1 = Number<InWeiVectorSize>{};
#else
    const auto C0 = 1;
    const auto C1 = C;

    const auto K0 = 1;
    const auto K1 = K;
#endif

    Tensor<TInWei> in_n_c0_hi_wi_c1(
        HostTensorDescriptor(std::initializer_list<index_t>{N, C0, Hi, Wi, C1}));
    Tensor<TInWei> wei_k_c0_y_x_c1(
        HostTensorDescriptor(std::initializer_list<index_t>{K, C0, Y, X, C1}));
    Tensor<TOut> out_n_k0_ho_wo_k1(
        HostTensorDescriptor(std::initializer_list<index_t>{N, K0, Ho, Wo, K1}));

    auto f_nchw2nc0hwc1 = [&](auto n, auto hi, auto wi, auto c) {
        in_n_c0_hi_wi_c1(n, c / C1, hi, wi, c % C1) = in_n_c_hi_wi(n, c, hi, wi);
    };

    auto f_kcyx2kc0yxc1 = [&](auto k, auto y, auto x, auto c) {
        wei_k_c0_y_x_c1(k, c / C1, y, x, c % C1) = wei_k_c_y_x(k, c, y, x);
    };

    make_ParallelTensorFunctor(f_nchw2nc0hwc1, N, Hi, Wi, C)();
    make_ParallelTensorFunctor(f_kcyx2kc0yxc1, K, Y, X, C)();

    DeviceMem in_n_c0_hi_wi_c1_device_buf(sizeof(TInWei) *
                                          in_n_c0_hi_wi_c1.mDesc.GetElementSpace());
    DeviceMem wei_k_c0_y_x_c1_device_buf(sizeof(TInWei) * wei_k_c0_y_x_c1.mDesc.GetElementSpace());
    DeviceMem out_n_k0_ho_wo_k1_device_buf(sizeof(TOut) *
                                           out_n_k0_ho_wo_k1.mDesc.GetElementSpace());

    in_n_c0_hi_wi_c1_device_buf.ToDevice(in_n_c0_hi_wi_c1.mData.data());
    wei_k_c0_y_x_c1_device_buf.ToDevice(wei_k_c0_y_x_c1.mData.data());

    const auto in_n_c0_hi_wi_c1_desc =
        make_naive_tensor_descriptor_packed(make_tuple(N, C0, Hi, Wi, C1));
    const auto wei_k_c0_y_x_c1_desc =
        make_naive_tensor_descriptor_packed(make_tuple(K, C0, Y, X, C1));
    const auto out_n_k0_ho_wo_k1_desc =
        make_naive_tensor_descriptor_packed(make_tuple(N, K0, Ho, Wo, K1));

#if 1
    // cdata = 64, BlockSize = 64, 16x8x32x4
    constexpr index_t BlockSize = 64;

    constexpr index_t KPerBlock  = 16;
    constexpr index_t HoPerBlock = 8;
    constexpr index_t WoPerBlock = 32;

    constexpr index_t E1        = 4 * 9;
    constexpr index_t E2        = 4;
    constexpr index_t EPerBlock = 4;

    constexpr index_t KPerThread  = KPerBlock;
    constexpr index_t HoPerThread = 2;
    constexpr index_t WoPerThread = 2;
    constexpr index_t EPerThread  = 1;

    using ABlockTransferThreadSliceLengths_E0_E1_K_E2   = Sequence<1, 9, 1, E2>;
    using ABlockTransferThreadClusterLengths_E0_E1_K_E2 = Sequence<1, 4, 16, 1>;

    constexpr index_t ABlockTransferSrcScalarPerVector_E2 = E2;
    constexpr index_t ABlockTransferDstScalarPerVector_E2 = E2;

    constexpr index_t BThreadTransferSrcScalarPerVector_E2 = E2;

    constexpr index_t CThreadTransferDstScalarPerVector_K = 8;
#endif

    constexpr auto conv_driver =
        DriverDynamicConvolutionForwardImplicitGemmDlops_v5r1_nchw_kcyx_nkhw_outpad<
            BlockSize,
            TInWei,
            TAcc,
            TOut,
            E1,
            E2,
            KPerBlock,
            HoPerBlock,
            WoPerBlock,
            EPerBlock,
            KPerThread,
            HoPerThread,
            WoPerThread,
            EPerThread,
            ABlockTransferThreadSliceLengths_E0_E1_K_E2,
            ABlockTransferThreadClusterLengths_E0_E1_K_E2,
            ABlockTransferSrcScalarPerVector_E2,
            ABlockTransferDstScalarPerVector_E2,
            BThreadTransferSrcScalarPerVector_E2,
            CThreadTransferDstScalarPerVector_K>{};

    const auto ave_time =
        conv_driver.Run(wei_k_c0_y_x_c1_desc,
                        in_n_c0_hi_wi_c1_desc,
                        out_n_k0_ho_wo_k1_desc,
                        conv_strides,
                        conv_dilations,
                        in_left_pads,
                        in_right_pads,
                        static_cast<TInWei*>(wei_k_c0_y_x_c1_device_buf.GetDeviceBuffer()),
                        static_cast<TInWei*>(in_n_c0_hi_wi_c1_device_buf.GetDeviceBuffer()),
                        static_cast<TOut*>(out_n_k0_ho_wo_k1_device_buf.GetDeviceBuffer()),
                        nrepeat);

    out_n_k0_ho_wo_k1_device_buf.FromDevice(out_n_k0_ho_wo_k1.mData.data());

    auto f_nk0hwk1_to_nkhw = [&](auto n, auto k, auto ho, auto wo) {
        out_n_k_ho_wo(n, k, ho, wo) = out_n_k0_ho_wo_k1(n, k / K1, ho, wo, k % K1);
    };

    make_ParallelTensorFunctor(f_nk0hwk1_to_nkhw, N, K, Ho, Wo)();

    {
        float perf = static_cast<float>(std::size_t(2) * N * K * Ho * Wo * C * Y * X) /
                     (std::size_t(1000) * 1000 * 1000) / ave_time;

        std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s" << std::endl;
    }
}
