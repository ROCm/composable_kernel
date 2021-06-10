#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "transform_forward_convolution_into_gemm_v4r4_nhwc_kyxc_nhwk.hpp"
#include "driver_dynamic_gemm_v1r2.hpp"

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
void device_dynamic_convolution_forward_implicit_gemm_v4r4_nhwc_kyxc_nhwk(
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
    constexpr auto I4 = Number<4>{};
    constexpr auto I5 = Number<5>{};
    constexpr auto I6 = Number<6>{};
    constexpr auto I7 = Number<7>{};
    constexpr auto I8 = Number<8>{};

    DeviceMem in_n_hi_wi_c_device_buf(sizeof(TInWei) * in_n_hi_wi_c.mDesc.GetElementSpace());
    DeviceMem wei_k_y_x_c_device_buf(sizeof(TInWei) * wei_k_y_x_c.mDesc.GetElementSpace());
    DeviceMem out_n_ho_wo_k_device_buf(sizeof(TOut) * out_n_ho_wo_k.mDesc.GetElementSpace());

    in_n_hi_wi_c_device_buf.ToDevice(in_n_hi_wi_c.mData.data());
    wei_k_y_x_c_device_buf.ToDevice(wei_k_y_x_c.mData.data());
    out_n_ho_wo_k_device_buf.ToDevice(out_n_ho_wo_k.mData.data());

    const auto in_n_hi_wi_c_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(in_n_hi_wi_c_lengths);
    const auto wei_k_y_x_c_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(wei_k_y_x_c_lengths);
    const auto out_n_ho_wo_k_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(out_n_ho_wo_k_lengths);

#if 0
    // cdata = 16, BlockSize = 64, 16x64x4
    constexpr index_t BlockSize = 64;

    constexpr index_t GemmMPerBlockM1 = 16;
    constexpr index_t GemmNPerBlockN1 = 64;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmM1PerThreadM111 = 2;
    constexpr index_t GemmN1PerThreadN111 = 2;
    constexpr index_t GemmKPerThread  = 1;

    constexpr index_t GemmM11N11ThreadClusterM1101 = 2;
    constexpr index_t GemmM11N11ThreadClusterN1101 = 2;
    constexpr index_t GemmM11N11ThreadClusterM1100 = 2;
    constexpr index_t GemmM11N11ThreadClusterN1100 = 8;

    using GemmABlockTransferThreadSliceLengths_K_M0_M1   = Sequence<1, 1, 1>;
    using GemmABlockTransferThreadClusterLengths_K_M0_M1 = Sequence<4, 1, 16>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_K = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_M1 = 1;

    using GemmBBlockTransferThreadSliceLengths_K_N0_N1   = Sequence<4, 1, 1>;
    using GemmBBlockTransferThreadClusterLengths_K_N0_N1 = Sequence<1, 1, 64>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_K = 4;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_N1 = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_M11 = 2;
#elif 0
    // cdata = 32, BlockSize = 64, 16x128x4
    constexpr index_t BlockSize = 64;

    constexpr index_t GemmMPerBlockM1 = 16;
    constexpr index_t GemmNPerBlockN1 = 128;
    constexpr index_t GemmKPerBlock   = 4;

    constexpr index_t GemmM1PerThreadM111 = 2;
    constexpr index_t GemmN1PerThreadN111 = 4;
    constexpr index_t GemmKPerThread      = 1;

    constexpr index_t GemmM11N11ThreadClusterM1101 = 2;
    constexpr index_t GemmM11N11ThreadClusterN1101 = 2;
    constexpr index_t GemmM11N11ThreadClusterM1100 = 2;
    constexpr index_t GemmM11N11ThreadClusterN1100 = 8;

    using GemmABlockTransferThreadSliceLengths_K_M0_M1   = Sequence<1, 1, 1>;
    using GemmABlockTransferThreadClusterLengths_K_M0_M1 = Sequence<4, 1, 16>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_K  = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_M1 = 1;

    using GemmBBlockTransferThreadSliceLengths_K_N0_N1   = Sequence<4, 1, 2>;
    using GemmBBlockTransferThreadClusterLengths_K_N0_N1 = Sequence<1, 1, 64>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_K  = 4;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_N1 = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_M11 = 2;
#elif 0
    // cdata = 64, BlockSize = 64, 16x256x2
    constexpr index_t BlockSize = 64;

    constexpr index_t GemmMPerBlockM1 = 16;
    constexpr index_t GemmNPerBlockN1 = 256;
    constexpr index_t GemmKPerBlock   = 2;

    constexpr index_t GemmM1PerThreadM111 = 4;
    constexpr index_t GemmN1PerThreadN111 = 4;
    constexpr index_t GemmKPerThread      = 1;

    constexpr index_t GemmM11N11ThreadClusterM1101 = 1;
    constexpr index_t GemmM11N11ThreadClusterN1101 = 2;
    constexpr index_t GemmM11N11ThreadClusterM1100 = 2;
    constexpr index_t GemmM11N11ThreadClusterN1100 = 16;

    using GemmABlockTransferThreadSliceLengths_K_M0_M1   = Sequence<1, 1, 1>;
    using GemmABlockTransferThreadClusterLengths_K_M0_M1 = Sequence<2, 1, 16>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_K  = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_M1 = 1;

    using GemmBBlockTransferThreadSliceLengths_K_N0_N1   = Sequence<2, 1, 4>;
    using GemmBBlockTransferThreadClusterLengths_K_N0_N1 = Sequence<1, 1, 64>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_K  = 2;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_N1 = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_M11 = 4;
#elif 0
    // cdata = 64, BlockSize = 64, 16x256x4
    constexpr index_t BlockSize = 64;

    constexpr index_t GemmMPerBlockM1 = 16;
    constexpr index_t GemmNPerBlockN1 = 256;
    constexpr index_t GemmKPerBlock   = 4;

    constexpr index_t GemmM1PerThreadM111 = 4;
    constexpr index_t GemmN1PerThreadN111 = 4;
    constexpr index_t GemmKPerThread      = 1;

    constexpr index_t GemmM11N11ThreadClusterM1101 = 2;
    constexpr index_t GemmM11N11ThreadClusterN1101 = 2;
    constexpr index_t GemmM11N11ThreadClusterM1100 = 1;
    constexpr index_t GemmM11N11ThreadClusterN1100 = 16;

    using GemmABlockTransferThreadSliceLengths_K_M0_M1   = Sequence<1, 1, 1>;
    using GemmABlockTransferThreadClusterLengths_K_M0_M1 = Sequence<4, 1, 16>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_K  = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_M1 = 1;

    using GemmBBlockTransferThreadSliceLengths_K_N0_N1   = Sequence<4, 1, 4>;
    using GemmBBlockTransferThreadClusterLengths_K_N0_N1 = Sequence<1, 1, 64>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_K  = 4;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_N1 = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_M11 = 4;
#elif 0
    // cdata = 64, BlockSize = 128, 32x256x4
    constexpr index_t BlockSize = 128;

    constexpr index_t GemmMPerBlockM1 = 32;
    constexpr index_t GemmNPerBlockN1 = 256;
    constexpr index_t GemmKPerBlock   = 4;

    constexpr index_t GemmM1PerThreadM111 = 4;
    constexpr index_t GemmN1PerThreadN111 = 4;
    constexpr index_t GemmKPerThread      = 1;

    constexpr index_t GemmM11N11ThreadClusterM1101 = 2;
    constexpr index_t GemmM11N11ThreadClusterN1101 = 2;
    constexpr index_t GemmM11N11ThreadClusterM1100 = 2;
    constexpr index_t GemmM11N11ThreadClusterN1100 = 16;

    using GemmABlockTransferThreadSliceLengths_K_M0_M1   = Sequence<1, 1, 1>;
    using GemmABlockTransferThreadClusterLengths_K_M0_M1 = Sequence<4, 1, 32>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_K  = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_M1 = 1;

    using GemmBBlockTransferThreadSliceLengths_K_N0_N1   = Sequence<4, 1, 2>;
    using GemmBBlockTransferThreadClusterLengths_K_N0_N1 = Sequence<1, 1, 128>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_K  = 4;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_N1 = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_M11 = 4;
#elif 0
    // cdata = 64, BlockSize = 128, 32x256x8
    constexpr index_t BlockSize = 128;

    constexpr index_t GemmMPerBlockM1 = 32;
    constexpr index_t GemmNPerBlockN1 = 256;
    constexpr index_t GemmKPerBlock   = 8;

    constexpr index_t GemmM1PerThreadM111 = 4;
    constexpr index_t GemmN1PerThreadN111 = 4;
    constexpr index_t GemmKPerThread      = 1;

    constexpr index_t GemmM11N11ThreadClusterM1101 = 2;
    constexpr index_t GemmM11N11ThreadClusterN1101 = 2;
    constexpr index_t GemmM11N11ThreadClusterM1100 = 2;
    constexpr index_t GemmM11N11ThreadClusterN1100 = 16;

    using GemmABlockTransferThreadSliceLengths_K_M0_M1   = Sequence<2, 1, 1>;
    using GemmABlockTransferThreadClusterLengths_K_M0_M1 = Sequence<4, 1, 32>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_K  = 2;
    constexpr index_t GemmABlockTransferDstScalarPerVector_M1 = 1;

    using GemmBBlockTransferThreadSliceLengths_K_N0_N1   = Sequence<8, 1, 2>;
    using GemmBBlockTransferThreadClusterLengths_K_N0_N1 = Sequence<1, 1, 128>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_K  = 8;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_N1 = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_M11 = 4;
#elif 1
    // cdata = 64, BlockSize = 256, 128x128x8
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlockM1 = 128;
    constexpr index_t GemmNPerBlockN1 = 128;
    constexpr index_t GemmKPerBlock   = 8;

    constexpr index_t GemmM1PerThreadM111 = 4;
    constexpr index_t GemmN1PerThreadN111 = 4;
    constexpr index_t GemmKPerThread      = 1;

    constexpr index_t GemmM11N11ThreadClusterM1100 = 8;
    constexpr index_t GemmM11N11ThreadClusterN1100 = 8;
    constexpr index_t GemmM11N11ThreadClusterM1101 = 2;
    constexpr index_t GemmM11N11ThreadClusterN1101 = 2;

    using GemmABlockTransferThreadSliceLengths_K_M0_M1   = Sequence<4, 1, 1>;
    using GemmABlockTransferThreadClusterLengths_K_M0_M1 = Sequence<2, 1, 128>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_K  = 4;
    constexpr index_t GemmABlockTransferDstScalarPerVector_M1 = 1;

    using GemmBBlockTransferThreadSliceLengths_K_N0_N1   = Sequence<4, 1, 1>;
    using GemmBBlockTransferThreadClusterLengths_K_N0_N1 = Sequence<2, 1, 128>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_K  = 4;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_N1 = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_M11 = 4;
#elif 1
    // cdata = 64, BlockSize = 256, 128x128x16
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlockM1 = 128;
    constexpr index_t GemmNPerBlockN1 = 128;
    constexpr index_t GemmKPerBlock   = 16;

    constexpr index_t GemmM1PerThreadM111 = 4;
    constexpr index_t GemmN1PerThreadN111 = 4;
    constexpr index_t GemmKPerThread      = 1;

    constexpr index_t GemmM11N11ThreadClusterM1100 = 8;
    constexpr index_t GemmM11N11ThreadClusterN1100 = 8;
    constexpr index_t GemmM11N11ThreadClusterM1101 = 2;
    constexpr index_t GemmM11N11ThreadClusterN1101 = 2;

    using GemmABlockTransferThreadSliceLengths_K_M0_M1   = Sequence<4, 1, 2>;
    using GemmABlockTransferThreadClusterLengths_K_M0_M1 = Sequence<4, 1, 64>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_K  = 4;
    constexpr index_t GemmABlockTransferDstScalarPerVector_M1 = 2;

    using GemmBBlockTransferThreadSliceLengths_K_N0_N1   = Sequence<8, 1, 1>;
    using GemmBBlockTransferThreadClusterLengths_K_N0_N1 = Sequence<2, 1, 128>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_K  = 8;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_N1 = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_M11 = 4;
#endif

#if 1
    const auto descs =
        transform_forward_convolution_into_gemm_v4r4_nhwc_kyxc_nhwk_pad(wei_k_y_x_c_desc,
                                                                        in_n_hi_wi_c_desc,
                                                                        out_n_ho_wo_k_desc,
                                                                        conv_strides,
                                                                        conv_dilations,
                                                                        in_left_pads,
                                                                        in_right_pads);

#if 0
    // HACK: hacks that control index calculation when iterating over A, B, C matrix
    constexpr auto wei_gemmk_gemmm0_gemmn1_grid_iterator_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{}),
                   make_tuple(Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{}));

    constexpr auto in_gemmk_gemmn0_gemmn1_grid_iterator_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0>{}),
                   make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0>{}));

    constexpr auto out_gemmm0_gemmm10_gemmm11_gemmn0_gemmn10_gemmn11_grid_iterator_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{}),
                   make_tuple(Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{}));

    constexpr auto wei_gemmk_gemmm0_gemmm1_grid_move_slice_window_iterator_hacks =
        Sequence<0, 0, 0, 0, 0>{};

    constexpr auto in_gemmk_gemmn0_gemmn1_grid_move_slice_window_iterator_hacks =
        Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0>{};
#else
    // HACK: hacks that control index calculation when iterating over A, B, C matrix
    constexpr auto wei_gemmk_gemmm0_gemmn1_grid_iterator_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{}),
                   make_tuple(Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{}));

    constexpr auto in_gemmk_gemmn0_gemmn1_grid_iterator_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}),
                   make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}));

    constexpr auto out_gemmm0_gemmm10_gemmm11_gemmn0_gemmn10_gemmn11_grid_iterator_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{}),
                   make_tuple(Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{}));

    constexpr auto wei_gemmk_gemmm0_gemmm1_grid_move_slice_window_iterator_hacks =
        Sequence<0, 0, 0, 0, 0>{};

    constexpr auto in_gemmk_gemmn0_gemmn1_grid_move_slice_window_iterator_hacks =
        Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{};
#endif

#else
    const auto descs =
        transform_forward_convolution_into_gemm_v4r4_nhwc_kyxc_nhwk_1x1(wei_k_y_x_c_desc,
                                                                        in_n_hi_wi_c_desc,
                                                                        out_n_ho_wo_k_desc,
                                                                        conv_strides,
                                                                        conv_dilations,
                                                                        in_left_pads,
                                                                        in_right_pads);

    // HACK: hacks that control index calculation when iterating over A, B, C matrix
    constexpr auto wei_gemmk_gemmm0_gemmn1_grid_iterator_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{}),
                   make_tuple(Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{}));

    constexpr auto in_gemmk_gemmn0_gemmn1_grid_iterator_hacks = make_tuple(
        make_tuple(Sequence<0, 0, 0, 0, 0>{}, Sequence<0, 0, 0, 0, 0>{}, Sequence<0, 0, 0, 0, 0>{}),
        make_tuple(
            Sequence<0, 0, 0, 0, 0>{}, Sequence<0, 0, 0, 0, 0>{}, Sequence<0, 0, 0, 0, 0>{}));

    constexpr auto out_gemmm0_gemmm1_gemmn0_gemmn1_global_iterator_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{}),
                   make_tuple(Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{}));

    constexpr auto wei_gemmk_gemmm_global_move_slice_window_iterator_hacks =
        Sequence<0, 0, 0, 0, 0>{};

    constexpr auto in_gemmk_gemmn_global_move_slice_window_iterator_hacks =
        Sequence<0, 0, 0, 0, 0>{};
#endif

    const auto wei_gemmk_gemmm_grid_desc = descs[I0];
    const auto in_gemmk_gemmn_grid_desc  = descs[I1];
    const auto out_gemmm_gemmn_grid_desc = descs[I2];

    for(index_t i = 0; i < 5; ++i)
    {
        float ave_time = driver_dynamic_gemm_v1r2<
            BlockSize,
            TInWei,
            TAcc,
            TOut,
            InMemoryDataOperation::Set,
            decltype(wei_gemmk_gemmm_grid_desc),
            decltype(in_gemmk_gemmn_grid_desc),
            decltype(out_gemmm_gemmn_grid_desc),
            GemmMPerBlockM1,
            GemmNPerBlockN1,
            GemmKPerBlock,
            GemmM1PerThreadM111,
            GemmN1PerThreadN111,
            GemmKPerThread,
            GemmM11N11ThreadClusterM1100,
            GemmM11N11ThreadClusterN1100,
            GemmM11N11ThreadClusterM1101,
            GemmM11N11ThreadClusterN1101,
            GemmABlockTransferThreadSliceLengths_K_M0_M1,
            GemmABlockTransferThreadClusterLengths_K_M0_M1,
            Sequence<1, 2, 0>, // ABlockTransferThreadClusterArrangeOrder
            Sequence<1, 2, 0>, // ABlockTransferSrcAccessOrder
            0,                 // ABlockTransferSrcVectorDim
            GemmABlockTransferSrcScalarPerVector_K,
            GemmABlockTransferDstScalarPerVector_M1,
            false, // don't move back src coordinate after threadwise copy
            GemmBBlockTransferThreadSliceLengths_K_N0_N1,
            GemmBBlockTransferThreadClusterLengths_K_N0_N1,
            Sequence<1, 2, 0>, // BBlockTransferThreadClusterArrangeOrder
            Sequence<1, 2, 0>, // BBlockTransferSrcAccessOrder
            0,                 // BBlockTransferSrcVectorDim
            GemmBBlockTransferSrcScalarPerVector_K,
            GemmBBlockTransferDstScalarPerVector_N1,
            false,                      // don't move back src coordinate after threadwise copy
            Sequence<3, 4, 5, 0, 1, 2>, // CThreadTransferSrcDstAccessOrder
            2,                          // CThreadTransferSrcDstVectorDim
            GemmCThreadTransferDstScalarPerVector_M11,
            decltype(wei_gemmk_gemmm0_gemmn1_grid_iterator_hacks),
            decltype(in_gemmk_gemmn0_gemmn1_grid_iterator_hacks),
            decltype(out_gemmm0_gemmm10_gemmm11_gemmn0_gemmn10_gemmn11_grid_iterator_hacks),
            decltype(wei_gemmk_gemmm0_gemmm1_grid_move_slice_window_iterator_hacks),
            decltype(in_gemmk_gemmn0_gemmn1_grid_move_slice_window_iterator_hacks)>(
            static_cast<TInWei*>(wei_k_y_x_c_device_buf.GetDeviceBuffer()),
            static_cast<TInWei*>(in_n_hi_wi_c_device_buf.GetDeviceBuffer()),
            static_cast<TOut*>(out_n_ho_wo_k_device_buf.GetDeviceBuffer()),
            wei_gemmk_gemmm_grid_desc,
            in_gemmk_gemmn_grid_desc,
            out_gemmm_gemmn_grid_desc,
            wei_gemmk_gemmm0_gemmn1_grid_iterator_hacks,
            in_gemmk_gemmn0_gemmn1_grid_iterator_hacks,
            out_gemmm0_gemmm10_gemmm11_gemmn0_gemmn10_gemmn11_grid_iterator_hacks,
            wei_gemmk_gemmm0_gemmm1_grid_move_slice_window_iterator_hacks,
            in_gemmk_gemmn0_gemmn1_grid_move_slice_window_iterator_hacks,
            nrepeat);

        {
            const auto N = out_n_ho_wo_k_lengths[I0];
            const auto K = out_n_ho_wo_k_lengths[I3];
            const auto C = wei_k_y_x_c_lengths[I3];

            const auto Hi = in_n_hi_wi_c_lengths[I1];
            const auto Wi = in_n_hi_wi_c_lengths[I2];

            const auto Ho = out_n_ho_wo_k_lengths[I1];
            const auto Wo = out_n_ho_wo_k_lengths[I2];

            const auto Y = wei_k_y_x_c_lengths[I1];
            const auto X = wei_k_y_x_c_lengths[I2];

            float perf = (float)(std::size_t(2) * N * K * Ho * Wo * C * Y * X) /
                         (std::size_t(1000) * 1000 * 1000) / ave_time;

            std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s"
                      << std::endl;
        }
    }

    // copy result back to host
    out_n_ho_wo_k_device_buf.FromDevice(out_n_ho_wo_k.mData.data());
}
