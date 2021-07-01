#ifndef CK_DRIVER_DYNAMIC_CONVOLUTION_FORWARD_IMPLICIT_GEMM_V4R4_XDLOPS_NCHW_KCYX_NKHW_HPP
#define CK_DRIVER_DYNAMIC_CONVOLUTION_FORWARD_IMPLICIT_GEMM_V4R4_XDLOPS_NCHW_KCYX_NKHW_HPP

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "driver_dynamic_gemm_xdlops_v1.hpp"
#include "driver_dynamic_gemm_xdlops_v2.hpp"

namespace ck {

// GemmM = K
// GemmN = N * Ho * Wo
// GemmK = C * Y * X
template <typename FloatAB,
          index_t GemmMPerBlock,
          index_t GemmNPerBlock,
          index_t GemmMPerWave,
          index_t GemmNPerWave,
          index_t GemmKPack,
          typename... Wei,
          typename... In,
          typename... Out,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
__host__ __device__ constexpr auto
transform_forward_convolution_into_gemm_v4r4_xdlops_nchw_kcyx_nkhw_pad(
    const DynamicTensorDescriptor<Wei...>& wei_k_c_y_x_global_desc,
    const DynamicTensorDescriptor<In...>& in_n_c_hi_wi_global_desc,
    const DynamicTensorDescriptor<Out...>& out_n_k_ho_wo_global_desc,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    const auto N = in_n_c_hi_wi_global_desc.GetLength(I0);
    const auto C = in_n_c_hi_wi_global_desc.GetLength(I1);
    const auto K = out_n_k_ho_wo_global_desc.GetLength(I1);

    const auto Hi = in_n_c_hi_wi_global_desc.GetLength(I2);
    const auto Wi = in_n_c_hi_wi_global_desc.GetLength(I3);

    const auto Ho = out_n_k_ho_wo_global_desc.GetLength(I2);
    const auto Wo = out_n_k_ho_wo_global_desc.GetLength(I3);

    const auto Y = wei_k_c_y_x_global_desc.GetLength(I2);
    const auto X = wei_k_c_y_x_global_desc.GetLength(I3);

    const auto ConvStrideH = conv_strides[I0];
    const auto ConvStrideW = conv_strides[I1];

    const auto ConvDilationH = conv_dilations[I0];
    const auto ConvDilationW = conv_dilations[I1];

    const auto InLeftPadH = in_left_pads[I0];
    const auto InLeftPadW = in_left_pads[I1];

    const auto InRightPadH = in_right_pads[I0];
    const auto InRightPadW = in_right_pads[I1];

    const auto GemmM  = K;
    const auto GemmN  = N * Ho * Wo;
    const auto GemmK  = C * Y * X;
    const auto GemmK0 = GemmK / GemmKPack;

    // weight tensor
    const auto wei_gemmk_gemmm_global_desc = transform_dynamic_tensor_descriptor(
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(K, C * Y * X)),
        make_tuple(make_pass_through_transform(K), make_pass_through_transform(C * Y * X)),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<1>{}, Sequence<0>{}));

    const auto wei_gemmk0_gemmm_gemmk1_global_desc = transform_dynamic_tensor_descriptor(
        wei_gemmk_gemmm_global_desc,
        make_tuple(make_unmerge_transform(make_tuple(GemmK0, GemmKPack)),
                   make_pass_through_transform(GemmM)),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

    // input tensor
    const auto in_n_c_hip_wip_global_desc = transform_dynamic_tensor_descriptor(
        in_n_c_hi_wi_global_desc,
        make_tuple(make_pass_through_transform(N),
                   make_pass_through_transform(C),
                   make_pad_transform(Hi, InLeftPadH, InRightPadH),
                   make_pad_transform(Wi, InLeftPadW, InRightPadW)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

    const auto in_n_c_y_ho_x_wo_global_desc = transform_dynamic_tensor_descriptor(
        in_n_c_hip_wip_global_desc,
        make_tuple(make_pass_through_transform(N),
                   make_pass_through_transform(C),
                   make_embed_transform(make_tuple(Y, Ho), make_tuple(ConvDilationH, ConvStrideH)),
                   make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW))),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

    const auto in_gemmk_gemmn_global_desc =
        transform_dynamic_tensor_descriptor(in_n_c_y_ho_x_wo_global_desc,
                                            make_tuple(make_merge_transform(make_tuple(C, Y, X)),
                                                       make_merge_transform(make_tuple(N, Ho, Wo))),
                                            make_tuple(Sequence<1, 2, 4>{}, Sequence<0, 3, 5>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

    const auto in_gemmk0_gemmn_gemmk1_global_desc = transform_dynamic_tensor_descriptor(
        in_gemmk_gemmn_global_desc,
        make_tuple(make_unmerge_transform(make_tuple(GemmK0, GemmKPack)),
                   make_pass_through_transform(GemmN)),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

    // output tensor
    const auto out_gemmm_gemmn_global_desc = transform_dynamic_tensor_descriptor(
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(N, K, Ho * Wo)),
        make_tuple(make_pass_through_transform(K), make_merge_transform(make_tuple(N, Ho * Wo))),
        make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}));

    assert(GemmM == out_gemmm_gemmn_global_desc.GetLength(I0));
    assert(GemmN == out_gemmm_gemmn_global_desc.GetLength(I1));
    assert(GemmK0 == in_gemmk0_gemmn_gemmk1_global_desc.GetLength(I0));
    assert(GemmK0 == wei_gemmk0_gemmm_gemmk1_global_desc.GetLength(I0));

    assert(GemmM % GemmMPerBlock == 0 && GemmN % GemmNPerBlock == 0 && GemmK0 % GemmKPerBlock == 0);

    constexpr auto xdlops_gemm = XdlopsGemm<FloatAB, GemmMPerWave, GemmNPerWave, GemmKPack>{};

    constexpr auto CLayout = xdlops_gemm.GetCLayout();

    constexpr index_t M0 = CLayout.M1();
    constexpr index_t M1 = CLayout.N1();
    constexpr index_t M2 = CLayout.M0();

    const auto out_m0_m1_m2_n_global_desc = transform_dynamic_tensor_descriptor(
        out_gemmm_gemmn_global_desc,
        make_tuple(make_unmerge_transform(make_tuple(GemmM / (M1 * M2), M1, M2)),
                   make_pass_through_transform(GemmN)),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}));

    // out_gemm_block_cluster_desc
    const auto out_gemm_block_cluster_desc = make_cluster_descriptor_v2(
        make_tuple(GemmM / Number<GemmMPerBlock>{}, GemmN / Number<GemmNPerBlock>{}));

    // hack to control index calculation when iterating over wei_gemmk0_gemmm_gemmk1_global tensor
    constexpr auto wei_gemmk0_gemmm_gemmk1_global_iterator_hacks = make_tuple(
        make_tuple(Sequence<0, 0, 0, 0, 0>{}, Sequence<0, 0, 0, 0, 0>{}, Sequence<0, 0, 0, 0, 0>{}),
        make_tuple(
            Sequence<0, 0, 0, 0, 0>{}, Sequence<0, 0, 0, 0, 0>{}, Sequence<0, 0, 0, 0, 0>{}));

    constexpr auto wei_gemmk0_gemmm_gemmk1_global_move_slice_window_iterator_hacks =
        Sequence<0, 0, 0, 0, 0>{};

    // hack to control index calculation when iterating over in_gemmk0_gemmn_gemmk1_global tensor
    constexpr auto in_gemmk0_gemmn_gemmk1_global_iterator_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0>{}),
                   make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0>{}));

    constexpr auto in_gemmk0_gemmn_gemmk1_global_move_slice_window_iterator_hacks =
        Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0>{};

    // hack to control index calculation when iterating over out_gemmm0_gemmm1_gemmn0_gemmn1_global
    // tensor hack for NKHW format
    constexpr auto out_m0_m1_m2_n_global_iterator_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 1, 0, 0>{}),
                   make_tuple(Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 2, 0, 0>{}));

    return make_tuple(wei_gemmk0_gemmm_gemmk1_global_desc,
                      in_gemmk0_gemmn_gemmk1_global_desc,
                      out_m0_m1_m2_n_global_desc,
                      out_gemm_block_cluster_desc,
                      wei_gemmk0_gemmm_gemmk1_global_iterator_hacks,
                      in_gemmk0_gemmn_gemmk1_global_iterator_hacks,
                      out_m0_m1_m2_n_global_iterator_hacks,
                      wei_gemmk0_gemmm_gemmk1_global_move_slice_window_iterator_hacks,
                      in_gemmk0_gemmn_gemmk1_global_move_slice_window_iterator_hacks);
}

// GemmM = K
// GemmN = N * Ho * Wo
// GemmK = C * Y * X
template <typename FloatAB,
          index_t GemmMPerBlock,
          index_t GemmNPerBlock,
          index_t GemmMPerWave,
          index_t GemmNPerWave,
          index_t GemmKPack,
          typename... Wei,
          typename... In,
          typename... Out,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
__host__ __device__ constexpr auto
transform_forward_convolution_into_gemm_v4r4_xdlops_nchw_kcyx_nkhw_1x1(
    const DynamicTensorDescriptor<Wei...>& wei_k_c_y_x_global_desc,
    const DynamicTensorDescriptor<In...>& in_n_c_hi_wi_global_desc,
    const DynamicTensorDescriptor<Out...>& out_n_k_ho_wo_global_desc,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    const auto N = in_n_c_hi_wi_global_desc.GetLength(I0);
    const auto C = in_n_c_hi_wi_global_desc.GetLength(I1);
    const auto K = out_n_k_ho_wo_global_desc.GetLength(I1);

    const auto Hi = in_n_c_hi_wi_global_desc.GetLength(I2);
    const auto Wi = in_n_c_hi_wi_global_desc.GetLength(I3);

    const auto Ho = out_n_k_ho_wo_global_desc.GetLength(I2);
    const auto Wo = out_n_k_ho_wo_global_desc.GetLength(I3);

    const auto Y = wei_k_c_y_x_global_desc.GetLength(I2);
    const auto X = wei_k_c_y_x_global_desc.GetLength(I3);

    const auto ConvStrideH = conv_strides[I0];
    const auto ConvStrideW = conv_strides[I1];

    const auto ConvDilationH = conv_dilations[I0];
    const auto ConvDilationW = conv_dilations[I1];

    const auto InLeftPadH = in_left_pads[I0];
    const auto InLeftPadW = in_left_pads[I1];

    const auto InRightPadH = in_right_pads[I0];
    const auto InRightPadW = in_right_pads[I1];

    const auto GemmM  = K;
    const auto GemmN  = N * Ho * Wo;
    const auto GemmK  = C * Y * X;
    const auto GemmK0 = GemmK / GemmKPack;

    assert(Y == 1 && X == 1 && ConvStrideH == 1 && ConvStrideW == 1 && ConvDilationH == 1 &&
           ConvDilationW == 1 && InLeftPadH == 0 && InLeftPadW == 0 && InRightPadH == 0 &&
           InRightPadW == 0);

    // weight tensor
    const auto wei_gemmk_gemmm_global_desc = transform_dynamic_tensor_descriptor(
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(K, C)),
        make_tuple(make_pass_through_transform(K), make_pass_through_transform(C)),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<1>{}, Sequence<0>{}));

    const auto wei_gemmk0_gemmm_gemmk1_global_desc = transform_dynamic_tensor_descriptor(
        wei_gemmk_gemmm_global_desc,
        make_tuple(make_unmerge_transform(make_tuple(GemmK0, GemmKPack)),
                   make_pass_through_transform(GemmM)),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

    // input tensor
    const auto in_gemmk_gemmn_global_desc = transform_dynamic_tensor_descriptor(
        in_n_c_hi_wi_global_desc,
        make_tuple(make_pass_through_transform(C), make_merge_transform(make_tuple(N, Ho, Wo))),
        make_tuple(Sequence<1>{}, Sequence<0, 2, 3>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}));

    const auto in_gemmk0_gemmn_gemmk1_global_desc = transform_dynamic_tensor_descriptor(
        in_gemmk_gemmn_global_desc,
        make_tuple(make_unmerge_transform(make_tuple(GemmK0, GemmKPack)),
                   make_pass_through_transform(GemmN)),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

    // output tensor
    const auto out_gemmm_gemmn_global_desc = transform_dynamic_tensor_descriptor(
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(N, K, Ho * Wo)),
        make_tuple(make_pass_through_transform(K), make_merge_transform(make_tuple(N, Ho * Wo))),
        make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}));

    assert(GemmM == out_gemmm_gemmn_global_desc.GetLength(I0));
    assert(GemmN == out_gemmm_gemmn_global_desc.GetLength(I1));
    assert(GemmK0 == in_gemmk0_gemmn_gemmk1_global_desc.GetLength(I0));
    assert(GemmK0 == wei_gemmk0_gemmm_gemmk1_global_desc.GetLength(I0));

    assert(GemmM % GemmMPerBlock == 0 && GemmN % GemmNPerBlock == 0 && GemmK0 % GemmKPerBlock == 0);

    constexpr auto xdlops_gemm = XdlopsGemm<FloatAB, GemmMPerWave, GemmNPerWave, GemmKPack>{};

    constexpr auto CLayout = xdlops_gemm.GetCLayout();

    constexpr index_t M0 = CLayout.M1();
    constexpr index_t M1 = CLayout.N1();
    constexpr index_t M2 = CLayout.M0();

    const auto out_m0_m1_m2_n_global_desc = transform_dynamic_tensor_descriptor(
        out_gemmm_gemmn_global_desc,
        make_tuple(make_unmerge_transform(make_tuple(GemmM / (M1 * M2), M1, M2)),
                   make_pass_through_transform(GemmN)),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}));

    // out_gemm_block_cluster_desc
    const auto out_gemm_block_cluster_desc = make_cluster_descriptor_v2(
        make_tuple(GemmM / Number<GemmMPerBlock>{}, GemmN / Number<GemmNPerBlock>{}));

    // hack to control index calculation when iterating over wei_gemmk0_gemmm_gemmk1_global tensor
    constexpr auto wei_gemmk0_gemmm_gemmk1_global_iterator_hacks = make_tuple(
        make_tuple(Sequence<0, 0, 0, 0, 0>{}, Sequence<0, 0, 0, 0, 0>{}, Sequence<0, 0, 0, 0, 0>{}),
        make_tuple(
            Sequence<0, 0, 0, 0, 0>{}, Sequence<0, 0, 0, 0, 0>{}, Sequence<0, 0, 0, 0, 0>{}));

    constexpr auto wei_gemmk0_gemmm_gemmk1_global_move_slice_window_iterator_hacks =
        Sequence<0, 0, 0, 0, 0>{};

    // hack to control index calculation when iterating over in_gemmk0_gemmn_gemmk1_global tensor
    constexpr auto in_gemmk0_gemmn_gemmk1_global_iterator_hacks = make_tuple(
        make_tuple(Sequence<0, 0, 0, 0, 0>{}, Sequence<0, 0, 1, 0, 0>{}, Sequence<0, 0, 0, 0, 0>{}),
        make_tuple(
            Sequence<0, 0, 0, 0, 0>{}, Sequence<0, 0, 2, 0, 0>{}, Sequence<0, 0, 0, 0, 0>{}));

    constexpr auto in_gemmk0_gemmn_gemmk1_global_move_slice_window_iterator_hacks =
        Sequence<0, 1, 2, 0, 0>{};

    // hack to control index calculation when iterating over out_gemmm0_gemmm1_gemmn0_gemmn1_global
    // tensor hack for NKHW format
    constexpr auto out_m0_m1_m2_n_global_iterator_hacks =
        make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 1, 0, 0>{}),
                   make_tuple(Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 0, 0, 0>{},
                              Sequence<0, 0, 2, 0, 0>{}));

    return make_tuple(wei_gemmk0_gemmm_gemmk1_global_desc,
                      in_gemmk0_gemmn_gemmk1_global_desc,
                      out_m0_m1_m2_n_global_desc,
                      out_gemm_block_cluster_desc,
                      wei_gemmk0_gemmm_gemmk1_global_iterator_hacks,
                      in_gemmk0_gemmn_gemmk1_global_iterator_hacks,
                      out_m0_m1_m2_n_global_iterator_hacks,
                      wei_gemmk0_gemmm_gemmk1_global_move_slice_window_iterator_hacks,
                      in_gemmk0_gemmn_gemmk1_global_move_slice_window_iterator_hacks);
}

} // namespace ck
#endif
