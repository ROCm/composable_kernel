#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "driver_convolution_maxpool_forward_implicit_gemm_v5r1_dlops_nc0hwc1_kc0yxc1_nk0hwk1.hpp"

template <typename TInWei,
          typename TAcc,
          typename TOut,
          ck::index_t activ_type,
          typename InLengths,
          typename WeiLengths,
          typename MaxLengths,
          typename OutLengths,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
void device_convolution_maxpool_forward_implicit_gemm_v5r1_dlops_nc0hwc1_kc0yxc1_nk0hwk1(
    const InLengths& in_n_c0_hi_wi_c1_lengths,
    const WeiLengths& wei_k_c0_y_x_c1_lengths,
    const MaxLengths& max_n_k0_hx_wx_k1_lengths,
    const OutLengths& out_n_k0_ho_wo_k1_lengths,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads,
    const Tensor<TInWei>& in_n_c0_hi_wi_c1,
    const Tensor<TInWei>& wei_k_c0_y_x_c1,
    const Tensor<TOut>& bias_k0_k1,
    Tensor<TOut>& out_n_k0_ho_wo_k1,
    Tensor<TOut>& max_n_k0_hx_wx_k1,
    ck::index_t nrepeat)
{
    using namespace ck;

    std::cout << __func__ << std::endl;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};

    const auto N  = out_n_k0_ho_wo_k1_lengths[I0];
    const auto K0 = out_n_k0_ho_wo_k1_lengths[I1];
    const auto Ho = out_n_k0_ho_wo_k1_lengths[I2];
    const auto Wo = out_n_k0_ho_wo_k1_lengths[I3];
    const auto K1 = out_n_k0_ho_wo_k1_lengths[I4];

    const auto C0 = in_n_c0_hi_wi_c1_lengths[I1];
    const auto Hi = in_n_c0_hi_wi_c1_lengths[I2];
    const auto Wi = in_n_c0_hi_wi_c1_lengths[I3];
    const auto C1 = in_n_c0_hi_wi_c1_lengths[I4];

    const auto K = wei_k_c0_y_x_c1_lengths[I0];
    const auto Y = wei_k_c0_y_x_c1_lengths[I2];
    const auto X = wei_k_c0_y_x_c1_lengths[I3];

    const auto Hx = max_n_k0_hx_wx_k1_lengths[I2];
    const auto Wx = max_n_k0_hx_wx_k1_lengths[I3];

    DeviceMem in_n_c0_hi_wi_c1_device_buf(sizeof(TInWei) *
                                          in_n_c0_hi_wi_c1.mDesc.GetElementSpace());
    DeviceMem wei_k_c0_y_x_c1_device_buf(sizeof(TInWei) * wei_k_c0_y_x_c1.mDesc.GetElementSpace());
    DeviceMem bias_k0_k1_device_buf(sizeof(TOut) * bias_k0_k1.mDesc.GetElementSpace());
    DeviceMem out_n_k0_ho_wo_k1_device_buf(sizeof(TOut) *
                                           out_n_k0_ho_wo_k1.mDesc.GetElementSpace());
    DeviceMem max_n_k0_hx_wx_k1_device_buf(sizeof(TOut) *
                                           max_n_k0_hx_wx_k1.mDesc.GetElementSpace());

    in_n_c0_hi_wi_c1_device_buf.ToDevice(in_n_c0_hi_wi_c1.mData.data());
    wei_k_c0_y_x_c1_device_buf.ToDevice(wei_k_c0_y_x_c1.mData.data());
    bias_k0_k1_device_buf.ToDevice(bias_k0_k1.mData.data());
    max_n_k0_hx_wx_k1_device_buf.ToDevice(max_n_k0_hx_wx_k1.mData.data());

    constexpr index_t InWeiVectorSize = 8;

    if(C1 % InWeiVectorSize != 0)
    {
        throw std::runtime_error("wrong! C1 cannot be divided by InWeiVectorSize");
    }

#if 0
    constexpr index_t BlockSize = 256;

    constexpr index_t KPerBlock  = 32;
    constexpr index_t HoPerBlock = 8;
    constexpr index_t WoPerBlock = 64;

    constexpr index_t E1        = C0 * 9;
    constexpr index_t E2        = 1;
    constexpr index_t E1PerBlock = C0;

    constexpr index_t KPerThread  = 16;
    constexpr index_t HoPerThread = 2;
    constexpr index_t WoPerThread = 2;
    constexpr index_t EPerThread  = 1;

    using ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2   = Sequence<1, 9, 1, E2>;
    using ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2 = Sequence<1, E1PerBlock, KPerBlock, 1>;

    constexpr index_t ABlockTransferSrcScalarPerVector_E2 = E2;
    constexpr index_t ABlockTransferDstScalarPerVector_E2 = E2;

    constexpr index_t BThreadTransferSrcScalarPerVector_E2 = E2;

    constexpr index_t CThreadTransferDstScalarPerVector_K = K1;
#elif 1
    constexpr auto BlockSize = 64;

    constexpr auto KPerBlock  = K;
    constexpr auto HoPerBlock = 8;
    constexpr auto WoPerBlock = 32;

    constexpr auto E1         = C0 * 9;
    constexpr auto E2         = 1;
    constexpr auto K2         = 2;
    constexpr auto E1PerBlock = C0;

    constexpr auto KPerThread  = K;
    constexpr auto HoPerThread = 2;
    constexpr auto WoPerThread = 2;
    constexpr auto EPerThread  = 1;

    using ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2 = Sequence<1, 9, 1, 1, E2>;
    using ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2 =
        Sequence<1, E1PerBlock, 1, KPerBlock, 1>;

    constexpr auto ABlockTransferSrcScalarPerVector_E2 = E2;
    constexpr auto ABlockTransferDstScalarPerVector_E2 = E2;

    constexpr auto BThreadTransferSrcScalarPerVector_E2 = E2;

    constexpr auto CThreadTransferDstScalarPerVector_K = K1;
#endif

    const auto in_n_c0_hi_wi_c1_desc =
        make_naive_tensor_descriptor_packed(make_tuple(N, C0, Hi, Wi, C1));
    const auto wei_k_c0_y_x_c1_desc =
        make_naive_tensor_descriptor_packed(make_tuple(K, C0, Y, X, C1));
    const auto max_n_k0_hx_wx_k1_desc =
        make_naive_tensor_descriptor_packed(make_tuple(N, K0, Hx, Wx, K1));
    const auto out_n_k0_ho_wo_k1_desc =
        make_naive_tensor_descriptor_packed(make_tuple(N, K0, Ho, Wo, K1));

    static_assert(in_n_c0_hi_wi_c1_desc.IsKnownAtCompileTime(), "");
    static_assert(wei_k_c0_y_x_c1_desc.IsKnownAtCompileTime(), "");
    static_assert(max_n_k0_hx_wx_k1_desc.IsKnownAtCompileTime(), "");
    static_assert(out_n_k0_ho_wo_k1_desc.IsKnownAtCompileTime(), "");

    constexpr auto conv_driver =
        DriverDynamicConvolutionForwardImplicitGemmDlops_v5r1_nc0hwc1_kc0yxc1_nk0hwk1_maxpool<
            BlockSize,
            typename vector_type<TInWei, InWeiVectorSize>::type,
            TAcc,
            TOut,
            E1,
            E2,
            K2,
            KPerBlock,
            HoPerBlock,
            WoPerBlock,
            E1PerBlock,
            KPerThread,
            HoPerThread,
            WoPerThread,
            EPerThread,
            ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2,
            ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2,
            ABlockTransferSrcScalarPerVector_E2,
            ABlockTransferDstScalarPerVector_E2,
            BThreadTransferSrcScalarPerVector_E2,
            CThreadTransferDstScalarPerVector_K,
            activ_type>{};

    for(int i = 0; i < 5; i++)
    {

        const auto ave_time =
            conv_driver.Run(wei_k_c0_y_x_c1_desc,
                            in_n_c0_hi_wi_c1_desc,
                            out_n_k0_ho_wo_k1_desc,
                            max_n_k0_hx_wx_k1_desc,
                            conv_strides,
                            conv_dilations,
                            in_left_pads,
                            in_right_pads,
                            static_cast<typename vector_type<TInWei, InWeiVectorSize>::type*>(
                                wei_k_c0_y_x_c1_device_buf.GetDeviceBuffer()),
                            static_cast<typename vector_type<TInWei, InWeiVectorSize>::type*>(
                                in_n_c0_hi_wi_c1_device_buf.GetDeviceBuffer()),
                            static_cast<TOut*>(bias_k0_k1_device_buf.GetDeviceBuffer()),
                            static_cast<TOut*>(out_n_k0_ho_wo_k1_device_buf.GetDeviceBuffer()),
                            static_cast<TOut*>(max_n_k0_hx_wx_k1_device_buf.GetDeviceBuffer()),
                            nrepeat);

        {
            float perf = static_cast<float>(std::size_t(2) * N * K * Ho * Wo * C0 * C1 * Y * X) /
                         (std::size_t(1000) * 1000 * 1000) / ave_time;

            std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s"
                      << std::endl;
        }
    }

    out_n_k0_ho_wo_k1_device_buf.FromDevice(out_n_k0_ho_wo_k1.mData.data());
    max_n_k0_hx_wx_k1_device_buf.FromDevice(max_n_k0_hx_wx_k1.mData.data());
}
