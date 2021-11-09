#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "transform_forward_convolution3d_into_gemm_v4r4r4_ndhwc_kzyxc_ndhwk.hpp"
#include "driver_gemm_xdlops_v2r3.hpp"
#include "driver_batched_gemm_xdlops_v2r3.hpp"

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
void device_convolution3d_forward_implicit_gemm_v4r4r4_xdlops_ndhwc_kzyxc_ndhwk(
    const InLengths& in_n_di_hi_wi_c_lengths,
    const WeiLengths& wei_k_z_y_x_c_lengths,
    const OutLengths& out_n_do_ho_wo_k_lengths,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads,
    const Tensor<TInWei>& in_n_di_hi_wi_c,
    const Tensor<TInWei>& wei_k_z_y_x_c,
    Tensor<TOut>& out_n_do_ho_wo_k,
    ck::index_t nrepeat)
{
    using namespace ck;

    std::cout << __func__ << std::endl;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};

    DeviceMem in_n_di_hi_wi_c_device_buf(sizeof(TInWei) * in_n_di_hi_wi_c.mDesc.GetElementSpace());
    DeviceMem wei_k_z_y_x_c_device_buf(sizeof(TInWei) * wei_k_z_y_x_c.mDesc.GetElementSpace());
    DeviceMem out_n_do_ho_wo_k_device_buf(sizeof(TOut) * out_n_do_ho_wo_k.mDesc.GetElementSpace());

    in_n_di_hi_wi_c_device_buf.ToDevice(in_n_di_hi_wi_c.mData.data());
    wei_k_z_y_x_c_device_buf.ToDevice(wei_k_z_y_x_c.mData.data());
    out_n_do_ho_wo_k_device_buf.ToDevice(out_n_do_ho_wo_k.mData.data());

    const auto in_n_di_hi_wi_c_desc  = make_naive_tensor_descriptor_packed<true>(in_n_di_hi_wi_c_lengths);
    const auto wei_k_z_y_x_c_desc   = make_naive_tensor_descriptor_packed<true>(wei_k_z_y_x_c_lengths);
    const auto out_n_do_ho_wo_k_desc = make_naive_tensor_descriptor_packed<true>(out_n_do_ho_wo_k_lengths);
    printf("%s: %d %ld %ld\n", __FILE__, __LINE__, in_n_di_hi_wi_c_desc.GetElementSpaceSize(), in_n_di_hi_wi_c_desc.GetElementSize());

#if 0
    // [M, N, K0, K1] = [256, 128, 4, 4], C = 128, for fp32
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 256;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerXDL = 32;
    constexpr index_t GemmNPerXDL = 32;
    constexpr index_t GemmK1       = 4;

    constexpr index_t MRepeat = 4;
    constexpr index_t NRepeat = 2;

    using GemmABlockTransferThreadSliceLengths_GemmK0_GemmM_GemmK1   = Sequence<1, 4, 4>;
    using GemmABlockTransferThreadClusterLengths_GemmK0_GemmM_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK1 = 4;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmK1 = 4;

    using GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1   = Sequence<1, 2, 4>;
    using GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK1 = 4;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmK1 = 4;

    constexpr index_t GemmCThreadTransferDstScalarPerVector = 1;
#elif 0
    // [M, N, K0, K1] = [128, 128, 4, 4], C = 128, for fp32
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerXDL = 32;
    constexpr index_t GemmNPerXDL = 32;
    constexpr index_t GemmK1      = 4;

    constexpr index_t MRepeat = 2;
    constexpr index_t NRepeat = 2;

    using GemmABlockTransferThreadSliceLengths_GemmK0_GemmM_GemmK1   = Sequence<1, 2, 4>;
    using GemmABlockTransferThreadClusterLengths_GemmK0_GemmM_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK1 = 4;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmK1 = 4;

    using GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1   = Sequence<1, 2, 4>;
    using GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK1 = 4;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmK1 = 4;

    constexpr index_t GemmCThreadTransferDstScalarPerVector = 1;
#elif 0
    // [M, N, K0, K1] = [256, 256, 4, 8], C = 256, for fp16
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 256;
    constexpr index_t GemmNPerBlock = 256;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerXDL = 32;
    constexpr index_t GemmNPerXDL = 32;
    constexpr index_t GemmK1      = 8;

    constexpr index_t MRepeat = 4;
    constexpr index_t NRepeat = 4;

    using GemmABlockTransferThreadSliceLengths_GemmK0_GemmM_GemmK1   = Sequence<1, 4, 8>;
    using GemmABlockTransferThreadClusterLengths_GemmK0_GemmM_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK1 = 8;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmK1 = 8;

    using GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1   = Sequence<1, 4, 8>;
    using GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK1 = 8;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmK1 = 8;

    constexpr index_t GemmCThreadTransferDstScalarPerVector = 1;
#elif 0
    // [M, N, K0, K1] = [256, 128, 4, 8], C = 128, for fp16
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 256;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerXDL = 32;
    constexpr index_t GemmNPerXDL = 32;
    constexpr index_t GemmK1      = 8;

    constexpr index_t MRepeat = 4;
    constexpr index_t NRepeat = 2;

    using GemmABlockTransferThreadSliceLengths_GemmK0_GemmM_GemmK1   = Sequence<1, 4, 8>;
    using GemmABlockTransferThreadClusterLengths_GemmK0_GemmM_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK1 = 8;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmK1 = 8;

    using GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1   = Sequence<1, 2, 8>;
    using GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK1 = 8;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmK1 = 8;

    constexpr index_t GemmCThreadTransferDstScalarPerVector = 1;
#elif 0
    // [M, N, K0, K1] = [128, 256, 4, 8], C = 128, for fp16
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 256;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerXDL = 32;
    constexpr index_t GemmNPerXDL = 32;
    constexpr index_t GemmK1      = 8;

    constexpr index_t MRepeat = 2;
    constexpr index_t NRepeat = 4;

    using GemmABlockTransferThreadSliceLengths_GemmK0_GemmM_GemmK1   = Sequence<1, 2, 8>;
    using GemmABlockTransferThreadClusterLengths_GemmK0_GemmM_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK1 = 8;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmK1 = 8;

    using GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1   = Sequence<1, 4, 8>;
    using GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK1 = 8;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmK1 = 8;

    constexpr index_t GemmCThreadTransferDstScalarPerVector = 1;
#elif 0
    // [M, N, K0, K1] = [128, 128, 4, 8], C = 64, for fp16
    // constexpr index_t BlockSize = 256;
    constexpr index_t BlockSize = 64;

    constexpr index_t GemmMPerBlock = 16;
    constexpr index_t GemmNPerBlock = 16;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerXDL = 16;
    constexpr index_t GemmNPerXDL = 16;
    constexpr index_t GemmK1      = 8;

    constexpr index_t MRepeat = 1;
    constexpr index_t NRepeat = 1;

    using GemmABlockTransferThreadSliceLengths_GemmK0_GemmM_GemmK1   = Sequence<1, 1, 8>;
    using GemmABlockTransferThreadClusterLengths_GemmK0_GemmM_GemmK1 = Sequence<4, 16, 1>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK1 = 8;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmK1 = 8;

    using GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1   = Sequence<1, 1, 8>;
    using GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1 = Sequence<4, 16, 1>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK1 = 8;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmK1 = 8;

    constexpr index_t GemmCThreadTransferDstScalarPerVector = 1;
#elif 1
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerXDL = 16;
    constexpr index_t GemmNPerXDL = 16;
    constexpr index_t GemmK1      = 8;

    constexpr index_t MRepeat = 4;
    constexpr index_t NRepeat = 4;

    using GemmABlockTransferThreadSliceLengths_GemmK0_GemmM_GemmK1   = Sequence<1, 2, 8>;
    using GemmABlockTransferThreadClusterLengths_GemmK0_GemmM_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK1 = 8;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmK1 = 8;

    using GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1   = Sequence<1, 2, 8>;
    using GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK1 = 8;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmK1 = 8;

    constexpr index_t GemmCThreadTransferDstScalarPerVector = 1;
#elif 0
    // [M, N, K0, K1] = [128, 64, 4, 8], C = 64, for fp16
    constexpr index_t BlockSize = 128;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 64;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerXDL = 32;
    constexpr index_t GemmNPerXDL = 32;
    constexpr index_t GemmK1      = 8;

    constexpr index_t MRepeat = 2;
    constexpr index_t NRepeat = 2;

    using GemmABlockTransferThreadSliceLengths_GemmK0_GemmM_GemmK1   = Sequence<1, 4, 8>;
    using GemmABlockTransferThreadClusterLengths_GemmK0_GemmM_GemmK1 = Sequence<4, 32, 1>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK1 = 8;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmK1 = 8;

    using GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1   = Sequence<1, 2, 8>;
    using GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1 = Sequence<4, 32, 1>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK1 = 8;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmK1 = 8;

    constexpr index_t GemmCThreadTransferDstScalarPerVector = 1;
#elif 1
    // [M, N, K0, K1] = [128, 64, 4, 8], C = 32, for fp16
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 64;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerXDL = 32;
    constexpr index_t GemmNPerXDL = 32;
    constexpr index_t GemmK1      = 8;

    constexpr index_t MRepeat = 2;
    constexpr index_t NRepeat = 1;

    using GemmABlockTransferThreadSliceLengths_GemmK0_GemmM_GemmK1   = Sequence<1, 2, 8>;
    using GemmABlockTransferThreadClusterLengths_GemmK0_GemmM_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK1 = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmK1 = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1   = Sequence<1, 1, 8>;
    using GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1 = Sequence<4, 64, 1>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmK1 = 1;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmK1 = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector = 1;
#endif

#define _jfy_ver_ 0
#if _jfy_ver_ == 1
    const auto descs =
        transform_forward_convolution3d_into_gemm_v4r4r4_nhwc_kyxc_nhwk_pad(in_n_di_hi_wi_c_desc,
                                                                            wei_k_z_y_x_c_desc,
                                                                            out_n_do_ho_wo_k_desc,
                                                                            conv_strides,
                                                                            conv_dilations,
                                                                            in_left_pads,
                                                                            in_right_pads,
                                                                            Number<GemmK1>{});

    const auto in_gemmk0_gemmm_gemmk1_grid_desc  = descs[I0];
    const auto wei_gemmk0_gemmn_gemmk1_grid_desc = descs[I1];
    const auto out_gemmm_gemmn_grid_desc         = descs[I2];

    printf("%s: %d %ld %ld\n", __FILE__, __LINE__, in_gemmk0_gemmm_gemmk1_grid_desc.GetElementSpaceSize(), in_gemmk0_gemmm_gemmk1_grid_desc.GetElementSize());

    // HACK: hacks that control index calculation when iterating over A, B, C matrix
    constexpr auto in_gemmk0_gemmm_gemmk1_grid_step_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 0+: GemmK0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 1+: GemmM
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}),  // 2+: GemmK1
                   make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 0-: GemmK0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 1-: GemmM
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{})); // 2-: GemmK1

    constexpr auto wei_gemmk0_gemmn_gemmk1_grid_step_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},   // 0+: GemmK0
                              Sequence<0, 0, 0, 0, 0>{},   // 1+: GemmN
                              Sequence<0, 0, 0, 0, 0>{}),  // 2+: GemmK1
                   make_tuple(Sequence<0, 0, 0, 0, 0>{},   // 0-: GemmK0
                              Sequence<0, 0, 0, 0, 0>{},   // 1-: GemmN
                              Sequence<0, 0, 0, 0, 0>{})); // 2-: GemmK1

    constexpr auto out_m0_n0_m1_n1_m2_m3_m4_n2_grid_step_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 0+: M0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 1+: N0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 2+: M1
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 3+: N1
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 4+: M2
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 5+: M3
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 6+: M4
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}),  // 7+: N2
                   make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 0-: M0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 1-: N0
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 2-: M1
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 3-: N1
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 4-: M2
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 5-: M3
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},   // 6-: M4
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{})); // 7-: N2

    constexpr auto in_gemmk0_gemmm_gemmk1_grid_move_slice_window_step_hacks =
        Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{};

    constexpr auto wei_gemmk0_gemmn_gemmk1_grid_move_slice_window_step_hacks =
        Sequence<0, 0, 0, 0, 0>{};

    for(index_t i = 0; i < 5; ++i)
    {
        float ave_time = driver_gemm_xdlops_v2r3<
            BlockSize,
            TInWei,
            TAcc,
            TOut,
            InMemoryDataOperationEnum_t::Set,
            decltype(in_gemmk0_gemmm_gemmk1_grid_desc),
            decltype(wei_gemmk0_gemmn_gemmk1_grid_desc),
            decltype(out_gemmm_gemmn_grid_desc),
            GemmMPerBlock,
            GemmNPerBlock,
            GemmKPerBlock,
            GemmMPerXDL,
            GemmNPerXDL,
            GemmK1,
            MRepeat,
            NRepeat,
            GemmABlockTransferThreadSliceLengths_GemmK0_GemmM_GemmK1,
            GemmABlockTransferThreadClusterLengths_GemmK0_GemmM_GemmK1,
            Sequence<1, 0, 2>,
            Sequence<1, 0, 2>,
            2,
            GemmABlockTransferSrcScalarPerVector_GemmK1,
            GemmABlockTransferDstScalarPerVector_GemmK1,
            false, // don't move back src coordinate after threadwise copy
            GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1,
            GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1,
            Sequence<1, 0, 2>,
            Sequence<1, 0, 2>,
            2,
            GemmBBlockTransferSrcScalarPerVector_GemmK1,
            GemmBBlockTransferDstScalarPerVector_GemmK1,
            false, // don't move back src coordinate after threadwise copy
            Sequence<2, 3, 0, 1, 7, 5, 4, 6>,
            7,
            GemmCThreadTransferDstScalarPerVector,
            decltype(in_gemmk0_gemmm_gemmk1_grid_step_hacks),
            decltype(wei_gemmk0_gemmn_gemmk1_grid_step_hacks),
            decltype(out_m0_n0_m1_n1_m2_m3_m4_n2_grid_step_hacks),
            decltype(in_gemmk0_gemmm_gemmk1_grid_move_slice_window_step_hacks),
            decltype(wei_gemmk0_gemmn_gemmk1_grid_move_slice_window_step_hacks),
            false, // CAccessOrderMRepeatNRepeat
            true,  // ABlockLdsExtraM
            true   // BBlockLdsExtraN
            >(static_cast<TInWei*>(in_n_di_hi_wi_c_device_buf.GetDeviceBuffer()),
              static_cast<TInWei*>(wei_k_z_y_x_c_device_buf.GetDeviceBuffer()),
              static_cast<TOut*>(out_n_do_ho_wo_k_device_buf.GetDeviceBuffer()),
              in_gemmk0_gemmm_gemmk1_grid_desc,
              wei_gemmk0_gemmn_gemmk1_grid_desc,
              out_gemmm_gemmn_grid_desc,
              debug::debug_driver_gemm_xdlops_v2r3::M01,
              debug::debug_driver_gemm_xdlops_v2r3::N01,
              in_gemmk0_gemmm_gemmk1_grid_step_hacks,
              wei_gemmk0_gemmn_gemmk1_grid_step_hacks,
              out_m0_n0_m1_n1_m2_m3_m4_n2_grid_step_hacks,
              in_gemmk0_gemmm_gemmk1_grid_move_slice_window_step_hacks,
              wei_gemmk0_gemmn_gemmk1_grid_move_slice_window_step_hacks,
              nrepeat);

        {
            const auto N = out_n_do_ho_wo_k_lengths[I0];
            const auto K = out_n_do_ho_wo_k_lengths[I4];
            const auto C = wei_k_z_y_x_c_lengths[I4];

            const auto Do = out_n_do_ho_wo_k_lengths[I1];
            const auto Ho = out_n_do_ho_wo_k_lengths[I2];
            const auto Wo = out_n_do_ho_wo_k_lengths[I3];

            const auto Z = wei_k_z_y_x_c_lengths[I1];
            const auto Y = wei_k_z_y_x_c_lengths[I2];
            const auto X = wei_k_z_y_x_c_lengths[I3];

            float perf = static_cast<float>((std::size_t(2) * N * K * Do * Ho * Wo * C * Z * Y * X)) /
                         (std::size_t(1000) * 1000 * 1000) / ave_time;

            std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s"
                      << std::endl;
        }
    }
#else
    const auto descs =
        transform_forward_convolution3d_into_gemm_v4r4r4_nhwc_kyxc_nhwk_pad_splitN(in_n_di_hi_wi_c_desc,
                                                                                   wei_k_z_y_x_c_desc,
                                                                                   out_n_do_ho_wo_k_desc,
                                                                                   conv_strides,
                                                                                   conv_dilations,
                                                                                   in_left_pads,
                                                                                   in_right_pads,
                                                                                   Number<GemmK1>{});

    const auto in_s_gemmk0_gemmm_gemmk1_grid_desc  = descs[I0];
    const auto wei_gemmk0_gemmn_gemmk1_grid_desc = descs[I1];
    const auto out_s_gemmm_gemmn_grid_desc = descs[I2];

    printf("%s: %d %ld %ld\n", __FILE__, __LINE__, in_s_gemmk0_gemmm_gemmk1_grid_desc.GetElementSpaceSize(), in_s_gemmk0_gemmm_gemmk1_grid_desc.GetElementSize());

    // HACK: hacks that control index calculation when iterating over A, B, C matrix
    constexpr auto in_gemmk0_gemmm_gemmk1_grid_step_hacks =
        make_tuple(make_tuple(uniform_sequence_gen<24, 0>::type{}, 
                              uniform_sequence_gen<24, 0>::type{}, 
                              uniform_sequence_gen<24, 0>::type{}, 
                              uniform_sequence_gen<24, 0>::type{}, 
                              uniform_sequence_gen<24, 0>::type{}, 
                              uniform_sequence_gen<24, 0>::type{}, 
                              uniform_sequence_gen<24, 0>::type{}, 
                              uniform_sequence_gen<24, 0>::type{}),
                   make_tuple(uniform_sequence_gen<24, 0>::type{}, 
                              uniform_sequence_gen<24, 0>::type{}, 
                              uniform_sequence_gen<24, 0>::type{}, 
                              uniform_sequence_gen<24, 0>::type{}, 
                              uniform_sequence_gen<24, 0>::type{}, 
                              uniform_sequence_gen<24, 0>::type{}, 
                              uniform_sequence_gen<24, 0>::type{}, 
                              uniform_sequence_gen<24, 0>::type{}));

    constexpr auto wei_gemmk0_gemmn_gemmk1_grid_step_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},   // 0+: GemmK0
                              Sequence<0, 0, 0, 0, 0>{},   // 1+: GemmN
                              Sequence<0, 0, 0, 0, 0>{}),  // 2+: GemmK1
                   make_tuple(Sequence<0, 0, 0, 0, 0>{},   // 0-: GemmK0
                              Sequence<0, 0, 0, 0, 0>{},   // 1-: GemmN
                              Sequence<0, 0, 0, 0, 0>{})); // 2-: GemmK1

    constexpr auto out_m0_n0_m1_n1_m2_m3_m4_n2_grid_step_hacks =
        make_tuple(make_tuple(uniform_sequence_gen<31, 0>::type{}, 
                              uniform_sequence_gen<31, 0>::type{}, 
                              uniform_sequence_gen<31, 0>::type{}, 
                              uniform_sequence_gen<31, 0>::type{}, 
                              uniform_sequence_gen<31, 0>::type{}, 
                              uniform_sequence_gen<31, 0>::type{}, 
                              uniform_sequence_gen<31, 0>::type{}, 
                              uniform_sequence_gen<31, 0>::type{}),
                   make_tuple(uniform_sequence_gen<31, 0>::type{}, 
                              uniform_sequence_gen<31, 0>::type{}, 
                              uniform_sequence_gen<31, 0>::type{}, 
                              uniform_sequence_gen<31, 0>::type{}, 
                              uniform_sequence_gen<31, 0>::type{}, 
                              uniform_sequence_gen<31, 0>::type{}, 
                              uniform_sequence_gen<31, 0>::type{}, 
                              uniform_sequence_gen<31, 0>::type{}));

    constexpr auto in_gemmk0_gemmm_gemmk1_grid_move_slice_window_step_hacks =
        Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{};

    constexpr auto wei_gemmk0_gemmn_gemmk1_grid_move_slice_window_step_hacks =
        Sequence<0, 0, 0, 0, 0>{};


    const auto Di = in_n_di_hi_wi_c_desc.GetLength(I1);
    const auto Hi = in_n_di_hi_wi_c_desc.GetLength(I2);
    const auto Wi = in_n_di_hi_wi_c_desc.GetLength(I3);
    const auto C = in_n_di_hi_wi_c_desc.GetLength(I4);
    const auto Do = out_n_do_ho_wo_k_lengths[I1];
    const auto Ho = out_n_do_ho_wo_k_lengths[I2];
    const auto Wo = out_n_do_ho_wo_k_lengths[I3];
    const auto K = out_n_do_ho_wo_k_lengths[I4];

    const auto N = out_n_do_ho_wo_k_lengths[I0];
    const auto S = in_s_gemmk0_gemmm_gemmk1_grid_desc.GetLength(I0);
    const index_t a_batch_stride = N / S * Di * Hi * Wi * C;
    const index_t c_batch_stride = N / S * Do * Ho * Wo * K;
    printf("a_batch_stride = %d, c_batch_stride = %d\n", a_batch_stride, c_batch_stride);
    printf("a_offset(0,0,0,0) = %d, a_offset(1,0,0,0) = %d\n", 
           in_s_gemmk0_gemmm_gemmk1_grid_desc.CalculateOffset(make_multi_index(0, 0, 0, 0)), 
           in_s_gemmk0_gemmm_gemmk1_grid_desc.CalculateOffset(make_multi_index(1, 0, 0, 0)));
    printf("c_offset(0,0,0) = %d, c_offset(1,0,0) = %d\n", 
           out_s_gemmm_gemmn_grid_desc.CalculateOffset(make_multi_index(0, 0, 0)), 
           out_s_gemmm_gemmn_grid_desc.CalculateOffset(make_multi_index(1, 0, 0)));
    fflush(stdout);

    for(index_t i = 0; i < 1; ++i)
    {
        float ave_time = driver_batched_gemm_xdlops_v2r3<
            BlockSize,
            TInWei,
            TAcc,
            TOut,
            InMemoryDataOperationEnum_t::Set,
            decltype(in_s_gemmk0_gemmm_gemmk1_grid_desc),
            decltype(wei_gemmk0_gemmn_gemmk1_grid_desc),
            decltype(out_s_gemmm_gemmn_grid_desc),
            GemmMPerBlock,
            GemmNPerBlock,
            GemmKPerBlock,
            GemmMPerXDL,
            GemmNPerXDL,
            GemmK1,
            MRepeat,
            NRepeat,
            GemmABlockTransferThreadSliceLengths_GemmK0_GemmM_GemmK1,
            GemmABlockTransferThreadClusterLengths_GemmK0_GemmM_GemmK1,
            Sequence<1, 0, 2>,
            Sequence<1, 0, 2>,
            2,
            GemmABlockTransferSrcScalarPerVector_GemmK1,
            GemmABlockTransferDstScalarPerVector_GemmK1,
            false, // don't move back src coordinate after threadwise copy
            GemmBBlockTransferThreadSliceLengths_GemmK0_GemmN_GemmK1,
            GemmBBlockTransferThreadClusterLengths_GemmK0_GemmN_GemmK1,
            Sequence<1, 0, 2>,
            Sequence<1, 0, 2>,
            2,
            GemmBBlockTransferSrcScalarPerVector_GemmK1,
            GemmBBlockTransferDstScalarPerVector_GemmK1,
            false, // don't move back src coordinate after threadwise copy
            Sequence<2, 3, 0, 1, 7, 5, 4, 6>,
            7,
            GemmCThreadTransferDstScalarPerVector,
            decltype(in_gemmk0_gemmm_gemmk1_grid_step_hacks),
            decltype(wei_gemmk0_gemmn_gemmk1_grid_step_hacks),
            decltype(out_m0_n0_m1_n1_m2_m3_m4_n2_grid_step_hacks),
            decltype(in_gemmk0_gemmm_gemmk1_grid_move_slice_window_step_hacks),
            decltype(wei_gemmk0_gemmn_gemmk1_grid_move_slice_window_step_hacks),
            false, // CAccessOrderMRepeatNRepeat
            true,  // ABlockLdsExtraM
            true   // BBlockLdsExtraN
            >(static_cast<TInWei*>(in_n_di_hi_wi_c_device_buf.GetDeviceBuffer()),
              static_cast<TInWei*>(wei_k_z_y_x_c_device_buf.GetDeviceBuffer()),
              static_cast<TOut*>(out_n_do_ho_wo_k_device_buf.GetDeviceBuffer()),
              in_s_gemmk0_gemmm_gemmk1_grid_desc,
              wei_gemmk0_gemmn_gemmk1_grid_desc,
              out_s_gemmm_gemmn_grid_desc,
              debug::debug_driver_gemm_xdlops_v2r3::M01,
              debug::debug_driver_gemm_xdlops_v2r3::N01,
              in_gemmk0_gemmm_gemmk1_grid_step_hacks,
              wei_gemmk0_gemmn_gemmk1_grid_step_hacks,
              out_m0_n0_m1_n1_m2_m3_m4_n2_grid_step_hacks,
              in_gemmk0_gemmm_gemmk1_grid_move_slice_window_step_hacks,
              wei_gemmk0_gemmn_gemmk1_grid_move_slice_window_step_hacks,
              nrepeat);

        {
            const auto Z = wei_k_z_y_x_c_lengths[I1];
            const auto Y = wei_k_z_y_x_c_lengths[I2];
            const auto X = wei_k_z_y_x_c_lengths[I3];

            float perf = static_cast<float>((std::size_t(2) * N * K * Do * Ho * Wo * C * Z * Y * X)) /
                         (std::size_t(1000) * 1000 * 1000) / ave_time;

            std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s"
                      << std::endl;
        }
    }
#endif

    // copy result back to host
    out_n_do_ho_wo_k_device_buf.FromDevice(out_n_do_ho_wo_k.mData.data());
}

