#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "driver_convolution_forward_implicit_gemm_v5r1_dlops_nhwc_kyxc_nhwk.hpp"

template <typename TInWei,
          typename TAcc,
          typename TOut,
          typename InLengths,
          typename WeiLengths,
          typename OutLengths,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
void device_convolution_forward_implicit_gemm_v5r1_dlops_nhwc_kyxc_nhwk(
    const InLengths& in_n_hi_wi_c_lengths,
    const WeiLengths& wei_k_y_x_c_lengths,
    const OutLengths& out_n_ho_wo_k_lengths,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads,
    const Tensor<TInWei>& in_n_hi_wi_c,
    const Tensor<TInWei>& wei_k_y_x_c,
    Tensor<TOut>& out_n_ho_wo_k,
    ck::index_t nrepeat)
{
    using namespace ck;

    std::cout << __func__ << std::endl;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    const auto Hi = in_n_hi_wi_c_lengths[I1];
    const auto Wi = in_n_hi_wi_c_lengths[I2];

    const auto N  = out_n_ho_wo_k_lengths[I0];
    const auto Ho = out_n_ho_wo_k_lengths[I1];
    const auto Wo = out_n_ho_wo_k_lengths[I2];
    const auto K  = out_n_ho_wo_k_lengths[I3];

    const auto Y = wei_k_y_x_c_lengths[I1];
    const auto X = wei_k_y_x_c_lengths[I2];
    const auto C = wei_k_y_x_c_lengths[I3];

    DeviceMem in_n_hi_wi_c_device_buf(sizeof(TInWei) * in_n_hi_wi_c.mDesc.GetElementSpace());
    DeviceMem wei_k_y_x_c_device_buf(sizeof(TInWei) * wei_k_y_x_c.mDesc.GetElementSpace());
    DeviceMem out_n_ho_wo_k_device_buf(sizeof(TOut) * out_n_ho_wo_k.mDesc.GetElementSpace());

    in_n_hi_wi_c_device_buf.ToDevice(in_n_hi_wi_c.mData.data());
    wei_k_y_x_c_device_buf.ToDevice(wei_k_y_x_c.mData.data());

    const auto in_n_hi_wi_c_desc  = make_naive_tensor_descriptor_packed(make_tuple(N, Hi, Wi, C));
    const auto wei_k_y_x_c_desc   = make_naive_tensor_descriptor_packed(make_tuple(K, Y, X, C));
    const auto out_n_ho_wo_k_desc = make_naive_tensor_descriptor_packed(make_tuple(N, Ho, Wo, K));

#if 0
    // cdata = 64, BlockSize = 64, 16x8x32x4
    constexpr index_t BlockSize = 64;

    constexpr index_t KPerBlock  = 16;
    constexpr index_t HoPerBlock = 16;
    constexpr index_t WoPerBlock = 16;

    constexpr index_t E1        = 2;
    constexpr index_t E2        = 8;
    constexpr index_t EPerBlock = 2;

    constexpr index_t KPerThread  = KPerBlock;
    constexpr index_t HoPerThread = 2;
    constexpr index_t WoPerThread = 2;
    constexpr index_t EPerThread  = 1;

    using ABlockTransferThreadSliceLengths_E0_E1_K_E2   = Sequence<1, 1, 1, 8>;
    using ABlockTransferThreadClusterLengths_E0_E1_K_E2 = Sequence<1, EPerBlock, 16, 1>;

    constexpr index_t ABlockTransferSrcScalarPerVector_E2 = E2;
    constexpr index_t ABlockTransferDstScalarPerVector_E2 = E2;

    constexpr index_t BThreadTransferSrcScalarPerVector_E2 = E2;

    constexpr index_t CThreadTransferDstScalarPerVector_K = 8;

#else
    // cdata = 64, BlockSize = 64, 16x8x32x4
    constexpr index_t BlockSize = 64;

    constexpr index_t KPerBlock  = 16;
    constexpr index_t HoPerBlock = 8;
    constexpr index_t WoPerBlock = 32;

    constexpr index_t E1        = 2 * 9;
    constexpr index_t E2        = 8;
    constexpr index_t EPerBlock = 2;

    constexpr index_t KPerThread  = KPerBlock;
    constexpr index_t HoPerThread = 2;
    constexpr index_t WoPerThread = 2;
    constexpr index_t EPerThread  = 1;

    using ABlockTransferThreadSliceLengths_E0_E1_K_E2   = Sequence<1, 9, 1, E2>;
    using ABlockTransferThreadClusterLengths_E0_E1_K_E2 = Sequence<1, 2, 16, 1>;

    constexpr index_t ABlockTransferSrcScalarPerVector_E2 = E2;
    constexpr index_t ABlockTransferDstScalarPerVector_E2 = E2;

    constexpr index_t BThreadTransferSrcScalarPerVector_E2 = E2;

    constexpr index_t CThreadTransferDstScalarPerVector_K = 8;
#endif

    constexpr auto conv_driver =
        DriverDynamicConvolutionForwardImplicitGemmDlops_v5r1_nhwc_kyxc_nhwk_outpad<
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
        conv_driver.Run(wei_k_y_x_c_desc,
                        in_n_hi_wi_c_desc,
                        out_n_ho_wo_k_desc,
                        conv_strides,
                        conv_dilations,
                        in_left_pads,
                        in_right_pads,
                        static_cast<TInWei*>(wei_k_y_x_c_device_buf.GetDeviceBuffer()),
                        static_cast<TInWei*>(in_n_hi_wi_c_device_buf.GetDeviceBuffer()),
                        static_cast<TOut*>(out_n_ho_wo_k_device_buf.GetDeviceBuffer()),
                        nrepeat);

    {
        float perf = static_cast<float>(std::size_t(2) * N * K * Ho * Wo * C * Y * X) /
                     (std::size_t(1000) * 1000 * 1000) / ave_time;

        std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s" << std::endl;
    }

    out_n_ho_wo_k_device_buf.FromDevice(out_n_ho_wo_k.mData.data());
}
