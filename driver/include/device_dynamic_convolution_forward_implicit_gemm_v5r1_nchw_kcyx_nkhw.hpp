#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "driver_dynamic_convolution_forward_implicit_gemm_v5r1_nchw_kcyx_nkhw.hpp"
#include "driver_dynamic_convolution_forward_implicit_gemm_v5r1_nchw_kcyx_nkhw_outpad.hpp"

template <class TInWei,
          ck::index_t InWeiVectorSize,
          class TAcc,
          class TOut,
          class InDesc,
          class WeiDesc,
          class OutDesc,
          class ConvStrides,
          class ConvDilations,
          class InLeftPads,
          class InRightPads>
void device_dynamic_convolution_forward_implicit_gemm_v5r1_nchw_kcyx_nkhw(
    InDesc,
    const Tensor<TInWei>& in_n_c_hi_wi,
    WeiDesc,
    const Tensor<TInWei>& wei_k_c_y_x,
    OutDesc,
    Tensor<TOut>& out_n_k_ho_wo,
    ConvStrides,
    ConvDilations,
    InLeftPads,
    InRightPads,
    ck::index_t nrepeat)
{
    using namespace ck;

    std::cout << "device_dynamic_convolution_forward_implicit_gemm_v5r1_nchw_kcyx_nkhw"
              << std::endl;

    DeviceMem in_n_c_hi_wi_device_buf(sizeof(TInWei) * in_n_c_hi_wi.mDesc.GetElementSpace());
    DeviceMem wei_k_c_y_x_device_buf(sizeof(TInWei) * wei_k_c_y_x.mDesc.GetElementSpace());
    DeviceMem out_n_k_ho_wo_device_buf(sizeof(TOut) * out_n_k_ho_wo.mDesc.GetElementSpace());

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto N = OutDesc::GetLengths()[I0];
    constexpr auto K = OutDesc::GetLengths()[I1];
    constexpr auto C = WeiDesc::GetLengths()[I1];

    constexpr auto Hi = InDesc::GetLengths()[I2];
    constexpr auto Wi = InDesc::GetLengths()[I3];

    constexpr auto Ho = OutDesc::GetLengths()[I2];
    constexpr auto Wo = OutDesc::GetLengths()[I3];

    constexpr auto Y = WeiDesc::GetLengths()[I2];
    constexpr auto X = WeiDesc::GetLengths()[I3];

    constexpr auto C0 = C / Number<InWeiVectorSize>{};
    constexpr auto C1 = Number<InWeiVectorSize>{};

    constexpr auto K0 = K / Number<InWeiVectorSize>{};
    constexpr auto K1 = Number<InWeiVectorSize>{};

#if 0
    // run-time variables
    const auto in_n_c_hi_wi_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(to_multi_index(InDesc::GetLengths()));
    const auto wei_k_c_y_x_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(to_multi_index(WeiDesc::GetLengths()));
    const auto out_n_k_ho_wo_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(to_multi_index(OutDesc::GetLengths()));

    const auto conv_strides   = to_multi_index(ConvStrides{});
    const auto conv_dilations = to_multi_index(ConvDilations{});
    const auto in_left_pads   = to_multi_index(InLeftPads{});
    const auto in_right_pads  = to_multi_index(InRightPads{});
#else
    // compile-time variables
    const auto in_n_c0_hi_wi_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(N, C0, Hi, Wi));
    const auto wei_k_c0_y_x_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(K, C0, Y, X));
    const auto out_n_k0_ho_wo_k1_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(N, K0, Ho, Wo, K1));

    const auto conv_strides     = sequence_to_tuple_of_number(ConvStrides{});
    const auto conv_dilations   = sequence_to_tuple_of_number(ConvDilations{});
    const auto in_left_pads     = sequence_to_tuple_of_number(InLeftPads{});
    const auto in_right_pads    = sequence_to_tuple_of_number(InRightPads{});
#endif

    Tensor<TInWei> in_n_c0_hi_wi_c1(make_HostTensorDescriptor(
        make_native_tensor_descriptor_packed(Sequence<N, C0, Hi, Wi, C1>{})));
    Tensor<TInWei> wei_k_c0_y_x_c1(make_HostTensorDescriptor(
        make_native_tensor_descriptor_packed(Sequence<K, C0, Y, X, C1>{})));
    Tensor<TOut> out_n_k0_ho_wo_k1(make_HostTensorDescriptor(
        make_native_tensor_descriptor_packed(Sequence<N, K0, Ho, Wo, K1>{})));

    auto f_nchw2nc0hwc1 = [&](auto n, auto hi, auto wi, auto c) {
        in_n_c0_hi_wi_c1(n, c / InWeiVectorSize, hi, wi, c % InWeiVectorSize) =
            in_n_c_hi_wi(n, c, hi, wi);
    };

    auto f_kcyx2kc0yxc1 = [&](auto k, auto y, auto x, auto c) {
        wei_k_c0_y_x_c1(k, c / InWeiVectorSize, y, x, c % InWeiVectorSize) =
            wei_k_c_y_x(k, c, y, x);
    };

    make_ParallelTensorFunctor(f_nchw2nc0hwc1, N, Hi, Wi, C)();
    make_ParallelTensorFunctor(f_kcyx2kc0yxc1, K, Y, X, C)();

    in_n_c_hi_wi_device_buf.ToDevice(in_n_c0_hi_wi_c1.mData.data());
    wei_k_c_y_x_device_buf.ToDevice(wei_k_c0_y_x_c1.mData.data());

#if 1
    // cdata = 64, BlockSize = 64, 16x8x32x4
    constexpr index_t BlockSize = 64;

    constexpr index_t KPerBlock  = K;
    constexpr index_t HoPerBlock = 8;
    constexpr index_t WoPerBlock = 32;
    constexpr index_t EPerBlock  = C0;

    constexpr index_t KPerThread  = KPerBlock;
    constexpr index_t HoPerThread = 2;
    constexpr index_t WoPerThread = 2;
    constexpr index_t EPerThread  = EPerBlock;

    using ABlockTransferThreadSliceLengths_E_K   = Sequence<3, 1>;
    using ABlockTransferThreadClusterLengths_E_K = Sequence<3 * EPerBlock, KPerBlock>;

    constexpr index_t ABlockTransferSrcScalarPerVector_E = 1;
    constexpr index_t ABlockTransferDstScalarPerVector_K = 1;

    constexpr index_t BThreadTransferSrcScalarPerVector_W = 1;

    constexpr index_t CThreadTransferDstScalarPerVector_W = K1;

    static_assert(KPerThread % CThreadTransferDstScalarPerVector_W == 0, "");
#else
    constexpr index_t BlockSize = 64;

    constexpr index_t KPerBlock  = 16;
    constexpr index_t HoPerBlock = 8;
    constexpr index_t WoPerBlock = 32;
    constexpr index_t EPerBlock  = 1;

    constexpr index_t KPerThread  = 16;
    constexpr index_t HoPerThread = 2;
    constexpr index_t WoPerThread = 2;
    constexpr index_t EPerThread  = EPerBlock;

    using ABlockTransferThreadSliceLengths_E_K   = Sequence<9, 1>;
    using ABlockTransferThreadClusterLengths_E_K = Sequence<EPerBlock, 16>;

    constexpr index_t ABlockTransferSrcScalarPerVector_E = 1;
    constexpr index_t ABlockTransferDstScalarPerVector_K = 1;

    constexpr index_t BThreadTransferSrcScalarPerVector_W = 1;

    constexpr index_t CThreadTransferDstScalarPerVector_W = K1;

    static_assert(KPerThread % CThreadTransferDstScalarPerVector_W == 0, "");
#endif

    constexpr auto conv_driver =
#if 0
        DriverDynamicConvolutionForwardImplicitGemm_v5r1_nchw_kcyx_nkhw_pad<
#else
        DriverDynamicConvolutionForwardImplicitGemm_v5r1_nchw_kcyx_nkhw_outpad<
#endif
                       BlockSize,
                   typename vector_type<TInWei, InWeiVectorSize>::type, TAcc, TOut, KPerBlock,
                   HoPerBlock, WoPerBlock, EPerBlock, KPerThread, HoPerThread, WoPerThread,
                   EPerThread, ABlockTransferThreadSliceLengths_E_K,
                   ABlockTransferThreadClusterLengths_E_K, ABlockTransferSrcScalarPerVector_E,
                   ABlockTransferDstScalarPerVector_K, BThreadTransferSrcScalarPerVector_W,
                   CThreadTransferDstScalarPerVector_W > {};

    conv_driver.Run(wei_k_c0_y_x_desc,
                    in_n_c0_hi_wi_desc,
                    out_n_k0_ho_wo_k1_desc,
                    conv_strides,
                    conv_dilations,
                    in_left_pads,
                    in_right_pads,
                    static_cast<typename vector_type<TInWei, InWeiVectorSize>::type*>(
                        wei_k_c_y_x_device_buf.GetDeviceBuffer()),
                    static_cast<typename vector_type<TInWei, InWeiVectorSize>::type*>(
                        in_n_c_hi_wi_device_buf.GetDeviceBuffer()),
                    static_cast<TOut*>(out_n_k_ho_wo_device_buf.GetDeviceBuffer()));

    out_n_k_ho_wo_device_buf.FromDevice(out_n_k0_ho_wo_k1.mData.data());

    auto f_nk0hwk1_to_nkhw = [&](auto n, auto k, auto ho, auto wo) {
        out_n_k_ho_wo(n, k, ho, wo) =
            out_n_k0_ho_wo_k1(n, k / InWeiVectorSize, ho, wo, k % InWeiVectorSize);
    };

    make_ParallelTensorFunctor(f_nk0hwk1_to_nkhw, N, K, Ho, Wo)();
}
