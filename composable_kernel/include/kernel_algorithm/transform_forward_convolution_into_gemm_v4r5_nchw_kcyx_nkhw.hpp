#ifndef CK_TRANSFORM_FORWARD_CONVOLUTION_INTO_CONTRACTION_V4R5_NCHW_KCYX_NKHW_HPP
#define CK_TRANSFORM_FORWARD_CONVOLUTION_INTO_CONTRACTION_V4R5_NCHW_KCYX_NKHW_HPP

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"

namespace ck {

// GemmM = K
// GemmN = N * Ho * Wo
// GemmK = C * Y * X
template <index_t N0_,
          typename... Wei,
          typename... In,
          typename... Out,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
__host__ __device__ constexpr auto
transform_forward_convolution_into_contraction_v4r5_nchw_kcyx_nkhw_pad(
    const DynamicTensorDescriptor<Wei...>& wei_k_c_y_x_grid_desc,
    const DynamicTensorDescriptor<In...>& in_n_c_hi_wi_grid_desc,
    const DynamicTensorDescriptor<Out...>& out_n_k_ho_wo_grid_desc,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    const auto N = in_n_c_hi_wi_grid_desc.GetLength(I0);
    const auto C = in_n_c_hi_wi_grid_desc.GetLength(I1);
    const auto K = out_n_k_ho_wo_grid_desc.GetLength(I1);

    const auto Hi = in_n_c_hi_wi_grid_desc.GetLength(I2);
    const auto Wi = in_n_c_hi_wi_grid_desc.GetLength(I3);

    const auto Ho = out_n_k_ho_wo_grid_desc.GetLength(I2);
    const auto Wo = out_n_k_ho_wo_grid_desc.GetLength(I3);

    const auto Y = wei_k_c_y_x_grid_desc.GetLength(I2);
    const auto X = wei_k_c_y_x_grid_desc.GetLength(I3);

    const auto ConvStrideH = conv_strides[I0];
    const auto ConvStrideW = conv_strides[I1];

    const auto ConvDilationH = conv_dilations[I0];
    const auto ConvDilationW = conv_dilations[I1];

    const auto InLeftPadH = in_left_pads[I0];
    const auto InLeftPadW = in_left_pads[I1];

    const auto InRightPadH = in_right_pads[I0];
    const auto InRightPadW = in_right_pads[I1];

    // weight tensor
    const auto wei_gk_gm0_gm1_grid_desc = transform_dynamic_tensor_descriptor(
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(K, C * Y * X)),
        make_tuple(make_unmerge_transform(make_tuple(I1, K)),
                   make_pass_through_transform(C * Y * X)),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<1, 2>{}, Sequence<0>{}));

    // input tensor
    const auto in_n_c_hip_wip_grid_desc = transform_dynamic_tensor_descriptor(
        in_n_c_hi_wi_grid_desc,
        make_tuple(make_pass_through_transform(N),
                   make_pass_through_transform(C),
                   make_pad_transform(Hi, InLeftPadH, InRightPadH),
                   make_pad_transform(Wi, InLeftPadW, InRightPadW)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

    constexpr auto N0 = Number<N0_>{};
    const auto N1     = N / N0;

    const auto in_n0_n1_c_y_ho_x_wo_grid_desc = transform_dynamic_tensor_descriptor(
        in_n_c_hip_wip_grid_desc,
        make_tuple(make_unmerge_transform(make_tuple(N0, N1)),
                   make_pass_through_transform(C),
                   make_embed_transform(make_tuple(Y, Ho), make_tuple(ConvDilationH, ConvStrideH)),
                   make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW))),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3, 4>{}, Sequence<5, 6>{}));

    const auto in_gk_gn0_gn1_grid_desc = transform_dynamic_tensor_descriptor(
        in_n0_n1_c_y_ho_x_wo_grid_desc,
        make_tuple(make_merge_transform(make_tuple(C, Y, X)),
                   make_pass_through_transform(N0),
                   make_merge_transform(make_tuple(N1, Ho, Wo))),
        make_tuple(Sequence<2, 3, 5>{}, Sequence<0>{}, Sequence<1, 4, 6>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

    // output tensor
    const auto out_n_k_howo_grid_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(make_tuple(N, K, Ho * Wo));

    const auto out_n0_n1_1_k_howo_grid_desc = transform_dynamic_tensor_descriptor(
        out_n_k_howo_grid_desc,
        make_tuple(make_unmerge_transform(make_tuple(Number<N0>{}, N1)),
                   make_unmerge_transform(make_tuple(I1, K)),
                   make_pass_through_transform(Ho * Wo)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
        make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}, Sequence<4>{}));

    const auto out_gm0_gm1_gn0_gn1_grid_desc = transform_dynamic_tensor_descriptor(
        out_n0_n1_1_k_howo_grid_desc,
        make_tuple(make_pass_through_transform(I1),
                   make_pass_through_transform(K),
                   make_pass_through_transform(Number<N0>{}),
                   make_merge_transform_v2_magic_division(make_tuple(N1, Ho * Wo))),
        make_tuple(Sequence<2>{}, Sequence<3>{}, Sequence<0>{}, Sequence<1, 4>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

    return make_tuple(
        wei_gk_gm0_gm1_grid_desc, in_gk_gn0_gn1_grid_desc, out_gm0_gm1_gn0_gn1_grid_desc);
}

} // namespace ck
#endif
