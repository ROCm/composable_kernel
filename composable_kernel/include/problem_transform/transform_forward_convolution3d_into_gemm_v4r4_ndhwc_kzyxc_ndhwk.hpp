#ifndef CK_TRANSFORM_FORWARD_CONVOLUTION3D_INTO_GEMM_v4r4_NDHWC_KZYXC_NDHWK_HPP
#define CK_TRANSFORM_FORWARD_CONVOLUTION3D_INTO_GEMM_v4r4_NDHWC_KZYXC_NDHWK_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"

namespace ck {

// GemmM = K
// GemmN = N * Ho * Wo
// GemmK = C * Y * X
template <typename... Wei,
          typename... In,
          typename... Out,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
__host__ __device__ constexpr auto
transform_forward_convolution3d_into_gemm_v4r4_nhwc_kyxc_nhwk_pad(
    const TensorDescriptor<Wei...>& wei_k_z_y_x_c_grid_desc,
    const TensorDescriptor<In...>& in_n_di_hi_wi_c_grid_desc,
    const TensorDescriptor<Out...>& out_n_do_ho_wo_k_grid_desc,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};

    const auto N = in_n_di_hi_wi_c_grid_desc.GetLength(I0);
    const auto C = in_n_di_hi_wi_c_grid_desc.GetLength(I4);
    const auto K = out_n_do_ho_wo_k_grid_desc.GetLength(I4);

    const auto Di = in_n_di_hi_wi_c_grid_desc.GetLength(I1);
    const auto Hi = in_n_di_hi_wi_c_grid_desc.GetLength(I2);
    const auto Wi = in_n_di_hi_wi_c_grid_desc.GetLength(I3);

    const auto Do = out_n_do_ho_wo_k_grid_desc.GetLength(I1);
    const auto Ho = out_n_do_ho_wo_k_grid_desc.GetLength(I2);
    const auto Wo = out_n_do_ho_wo_k_grid_desc.GetLength(I3);

    const auto Z = wei_k_z_y_x_c_grid_desc.GetLength(I1);
    const auto Y = wei_k_z_y_x_c_grid_desc.GetLength(I2);
    const auto X = wei_k_z_y_x_c_grid_desc.GetLength(I3);

    const auto ConvStrideD = conv_strides[I0];
    const auto ConvStrideH = conv_strides[I1];
    const auto ConvStrideW = conv_strides[I2];

    const auto ConvDilationD = conv_dilations[I0];
    const auto ConvDilationH = conv_dilations[I1];
    const auto ConvDilationW = conv_dilations[I2];

    const auto InLeftPadD = in_left_pads[I0];
    const auto InLeftPadH = in_left_pads[I1];
    const auto InLeftPadW = in_left_pads[I2];

    const auto InRightPadD = in_right_pads[I0];
    const auto InRightPadH = in_right_pads[I1];
    const auto InRightPadW = in_right_pads[I2];

    // weight tensor
    const auto wei_gemmk_gemmm_grid_desc = transform_tensor_descriptor(
        make_naive_tensor_descriptor_packed(make_tuple(K, Z * Y * X * C)),
        make_tuple(make_pass_through_transform(K), make_pass_through_transform(Z * Y * X * C)),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<1>{}, Sequence<0>{}));

    // input tensor
    const auto n_dip_hip_wip_c_grid_desc = transform_tensor_descriptor(
        in_n_di_hi_wi_c_grid_desc,
        make_tuple(make_pass_through_transform(N),
                   make_pad_transform(Di, InLeftPadD, InRightPadD),
                   make_pad_transform(Hi, InLeftPadH, InRightPadH),
                   make_pad_transform(Wi, InLeftPadW, InRightPadW),
                   make_pass_through_transform(C)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

    const auto in_n_z_do_y_ho_x_wo_c_grid_desc = transform_tensor_descriptor(
        n_dip_hip_wip_c_grid_desc,
        make_tuple(make_pass_through_transform(N),
                   make_embed_transform(make_tuple(Z, Do), make_tuple(ConvDilationD, ConvStrideD)),
                   make_embed_transform(make_tuple(Y, Ho), make_tuple(ConvDilationH, ConvStrideH)),
                   make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW)),
                   make_pass_through_transform(C)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>),
        make_tuple(
            Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5, 6>, Sequence<7>{}));

    const auto in_gemmk_gemmn_grid_desc =
        transform_tensor_descriptor(in_n_z_do_y_ho_x_wo_c_grid_desc,
                                    make_tuple(make_merge_transform(make_tuple(Z, Y, X, C)),
                                               make_merge_transform(make_tuple(N, Do, Ho, Wo))),
                                    make_tuple(Sequence<1, 3, 5, 7>{}, Sequence<0, 2, 4, 6>{}),
                                    make_tuple(Sequence<0>{}, Sequence<1>{}));

    // output tensor
    const auto out_gemmm_gemmn_grid_desc = transform_tensor_descriptor(
        make_naive_tensor_descriptor_packed(make_tuple(N * Do * Ho * Wo, K)),
        make_tuple(make_pass_through_transform(N * Do * Ho * Wo), make_pass_through_transform(K)),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<1>{}, Sequence<0>{}));

    return make_tuple(
        wei_gemmk_gemmm_grid_desc, in_gemmk_gemmn_grid_desc, out_gemmm_gemmn_grid_desc);
}

template <typename... Wei,
          typename... In,
          typename... Out,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
__host__ __device__ constexpr auto
transform_forward_convolution3d_into_gemm_v4r4_nhwc_kyxc_nhwk_1x1(
    const TensorDescriptor<Wei...>& wei_k_z_y_x_c_grid_desc,
    const TensorDescriptor<In...>& in_n_di_hi_wi_c_grid_desc,
    const TensorDescriptor<Out...>& out_n_do_ho_wo_k_grid_desc,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};

    const auto N = in_n_di_hi_wi_c_grid_desc.GetLength(I0);
    const auto C = in_n_di_hi_wi_c_grid_desc.GetLength(I4);
    const auto K = out_n_do_ho_wo_k_grid_desc.GetLength(I4);

    const auto Do = out_n_do_ho_wo_k_grid_desc.GetLength(I1);
    const auto Ho = out_n_do_ho_wo_k_grid_desc.GetLength(I2);
    const auto Wo = out_n_do_ho_wo_k_grid_desc.GetLength(I3);

    const auto Z = wei_k_z_y_x_c_grid_desc.GetLength(I1);
    const auto Y = wei_k_z_y_x_c_grid_desc.GetLength(I2);
    const auto X = wei_k_z_y_x_c_grid_desc.GetLength(I3);

    const auto ConvStrideD = conv_strides[I0];
    const auto ConvStrideH = conv_strides[I1];
    const auto ConvStrideW = conv_strides[I2];

    const auto ConvDilationD = conv_dilations[I0];
    const auto ConvDilationH = conv_dilations[I1];
    const auto ConvDilationW = conv_dilations[I2];

    const auto InLeftPadD = in_left_pads[I0];
    const auto InLeftPadH = in_left_pads[I1];
    const auto InLeftPadW = in_left_pads[I2];

    const auto InRightPadD = in_right_pads[I0];
    const auto InRightPadH = in_right_pads[I1];
    const auto InRightPadW = in_right_pads[I2];

    assert(Z == 1 && Y == 1 && X == 1 && ConvStrideD == 1 && ConvStrideH == 1 && ConvStrideW == 1 &&
           ConvDilationD == 1 && ConvDilationH == 1 && ConvDilationW == 1 && InLeftPadD == 0 &&
           InLeftPadH == 0 && InLeftPadW == 0 && InRightPadD == 0 && InRightPadH == 0 &&
           InRightPadW == 0);

    // weight tensor
    const auto wei_gemmk_gemmm_grid_desc = transform_tensor_descriptor(
        make_naive_tensor_descriptor_packed(make_tuple(K, C)),
        make_tuple(make_pass_through_transform(K), make_pass_through_transform(C)),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<1>{}, Sequence<0>{}));

    // input tensor
    const auto in_gemmk_gemmn_grid_desc = transform_tensor_descriptor(
        make_naive_tensor_descriptor_packed(make_tuple(N * Do * Ho * Wo, C)),
        make_tuple(make_pass_through_transform(N * Do * Ho * Wo), make_pass_through_transform(C)),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<1>{}, Sequence<0>{}));

    // output tensor
    const auto out_gemmm_gemmn_grid_desc = transform_tensor_descriptor(
        make_naive_tensor_descriptor_packed(make_tuple(N * Do * Ho * Wo, K)),
        make_tuple(make_pass_through_transform(N * Do * Ho * Wo), make_pass_through_transform(K)),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<1>{}, Sequence<0>{}));

    return make_tuple(
        wei_gemmk_gemmm_grid_desc, in_gemmk_gemmn_grid_desc, out_gemmm_gemmn_grid_desc);
}

} // namespace ck
#endif
