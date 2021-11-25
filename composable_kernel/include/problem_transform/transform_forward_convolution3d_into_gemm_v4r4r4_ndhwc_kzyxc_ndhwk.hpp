#ifndef CK_TRANSFORM_FORWARD_CONVOLUTION3D_INTO_GEMM_V4R4R4_NHWC_KYXC_NHWK_HPP
#define CK_TRANSFORM_FORWARD_CONVOLUTION3D_INTO_GEMM_V4R4R4_NHWC_KYXC_NHWK_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"

namespace ck {

// A: in
// B: wei
// C: out
// GemmM = N * Do * Ho * Wo
// GemmN = K
// GemmK = Z * Y * X * C
template <typename... In,
          typename... Wei,
          typename... Out,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads,
          index_t GemmK1Value>
__host__ __device__ constexpr auto
transform_forward_convolution3d_into_gemm_v4r4r4_nhwc_kyxc_nhwk_pad(
    const TensorDescriptor<In...>& in_n_di_hi_wi_c_grid_desc,
    const TensorDescriptor<Wei...>& wei_k_z_y_x_c_grid_desc,
    const TensorDescriptor<Out...>& out_n_do_ho_wo_k_grid_desc,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads,
    Number<GemmK1Value>)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};

    constexpr auto GemmK1 = Number<GemmK1Value>{};

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

    const auto GemmM  = N * Do * Ho * Wo;
    const auto GemmN  = K;
    const auto GemmK  = Z * Y * X * C;
    const auto GemmK0 = GemmK / GemmK1;

    // A: input tensor
    const auto in_n_dip_hip_wip_c_grid_desc = transform_tensor_descriptor(
        in_n_di_hi_wi_c_grid_desc,
        make_tuple(make_pass_through_transform(N),
                   make_pad_transform(Di, InLeftPadD, InRightPadD),
                   make_pad_transform(Hi, InLeftPadH, InRightPadH),
                   make_pad_transform(Wi, InLeftPadW, InRightPadW),
                   make_pass_through_transform(C)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

    const auto in_n_z_do_y_ho_x_wo_c_grid_desc = transform_tensor_descriptor(
        in_n_dip_hip_wip_c_grid_desc,
        make_tuple(make_pass_through_transform(N),
                   make_embed_transform(make_tuple(Z, Do), make_tuple(ConvDilationD, ConvStrideD)),
                   make_embed_transform(make_tuple(Y, Ho), make_tuple(ConvDilationH, ConvStrideH)),
                   make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW)),
                   make_pass_through_transform(C)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
        make_tuple(
            Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5, 6>{}, Sequence<7>{}));

    const auto in_gemmk_gemmm_grid_desc =
        transform_tensor_descriptor(in_n_z_do_y_ho_x_wo_c_grid_desc,
                                    make_tuple(make_merge_transform(make_tuple(Z, Y, X, C)),
                                               make_merge_transform(make_tuple(N, Do, Ho, Wo))),
                                    make_tuple(Sequence<1, 3, 5, 7>{}, Sequence<0, 2, 4, 6>{}),
                                    make_tuple(Sequence<0>{}, Sequence<1>{}));

    const auto in_gemmk0_gemmm_gemmk1_grid_desc =
        transform_tensor_descriptor(in_gemmk_gemmm_grid_desc,
                                    make_tuple(make_unmerge_transform(make_tuple(GemmK0, GemmK1)),
                                               make_pass_through_transform(GemmM)),
                                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                                    make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

    // B: weight tensor
    const auto wei_gemmk_gemmn_grid_desc = transform_tensor_descriptor(
        make_naive_tensor_descriptor_packed(make_tuple(K, Z * Y * X * C)),
        make_tuple(make_pass_through_transform(K), make_pass_through_transform(Z * Y * X * C)),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<1>{}, Sequence<0>{}));

    const auto wei_gemmk0_gemmn_gemmk1_grid_desc =
        transform_tensor_descriptor(wei_gemmk_gemmn_grid_desc,
                                    make_tuple(make_unmerge_transform(make_tuple(GemmK0, GemmK1)),
                                               make_pass_through_transform(GemmN)),
                                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                                    make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

    // C: output tensor
    const auto out_gemmm_gemmn_grid_desc = transform_tensor_descriptor(
        make_naive_tensor_descriptor_packed(make_tuple(N * Do * Ho * Wo, K)),
        make_tuple(make_pass_through_transform(N * Do * Ho * Wo), make_pass_through_transform(K)),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}));

    return make_tuple(in_gemmk0_gemmm_gemmk1_grid_desc,
                      wei_gemmk0_gemmn_gemmk1_grid_desc,
                      out_gemmm_gemmn_grid_desc);
    // n0_gemmk0_gemmn_gemmk1, gemmk0_gemmn_gemmk1, n0_gemmm_gemmn
}

template <typename... In,
          typename... Wei,
          typename... Out,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads,
          index_t GemmK1Value>
__host__ __device__ constexpr auto
transform_forward_convolution3d_into_gemm_v4r4r4_nhwc_kyxc_nhwk_pad_splitN(
    const TensorDescriptor<In...>& in_n_di_hi_wi_c_grid_desc,
    const TensorDescriptor<Wei...>& wei_k_z_y_x_c_grid_desc,
    const TensorDescriptor<Out...>& out_n_do_ho_wo_k_grid_desc,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads,
    Number<GemmK1Value>)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};

    constexpr auto GemmK1 = Number<GemmK1Value>{};

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

    // N1 should satisfy that
    //   1) N1 = 2^i;
    //   2) N1 * (Do * Ho * Wo * K) < (2^31 - 1)
    //   3) N1 * (Di * Hi * Wi * C) < (2^31 - 1)
    //
    // Do NOT confuse (N0, N1) in this function with (N0, N1) in gridewise GEMM.
    auto N1 = N + 1;

    {
        const auto stride =
            std::max(long_index_t(Do) * Ho * Wo * K, long_index_t(Di) * Hi * Wi * C);
        const index_t max_stride = 2147483647;

        for(int n0 = 1; n0 <= N; ++n0)
        {
            int n1 = N / n0;
            if(n0 * n1 == N && long_index_t(n1) * long_index_t(stride) < max_stride)
            {
                N1 = n1;
                break;
            }
        }

        const auto N0 = N / N1;
        if(N0 * N1 != N)
        {
            throw std::runtime_error(__func__ +
                                     std::string(": failed to umerge N into (N0, N1).\n"));
        }
    }

    // jfy_debug
    // N1 = N / 2;
    // jfy_debug_end

    const auto N0 = N / N1;
    const auto GemmM  = N1 * Do * Ho * Wo;
    const auto GemmN  = K;
    const auto GemmK  = Z * Y * X * C;
    const auto GemmK0 = GemmK / GemmK1;

    // A: input tensor
    const auto in_n_dip_hip_wip_c_grid_desc = transform_tensor_descriptor(
        in_n_di_hi_wi_c_grid_desc,
        make_tuple(make_pass_through_transform(N),
                   make_pad_transform(Di, InLeftPadD, InRightPadD),
                   make_pad_transform(Hi, InLeftPadH, InRightPadH),
                   make_pad_transform(Wi, InLeftPadW, InRightPadW),
                   make_pass_through_transform(C)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

    const auto in_n0_n1_z_do_y_ho_x_wo_c_grid_desc = transform_tensor_descriptor(
        in_n_dip_hip_wip_c_grid_desc,
        make_tuple(make_unmerge_transform(make_tuple(N0, N1)),
                   make_embed_transform(make_tuple(Z, Do), make_tuple(ConvDilationD, ConvStrideD)),
                   make_embed_transform(make_tuple(Y, Ho), make_tuple(ConvDilationH, ConvStrideH)),
                   make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW)),
                   make_pass_through_transform(C)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
        make_tuple(
            Sequence<0, 1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}, Sequence<6, 7>{}, Sequence<8>{}));

    const auto in_n0_gemmk_gemmm_grid_desc = transform_tensor_descriptor(
        in_n0_n1_z_do_y_ho_x_wo_c_grid_desc,
        make_tuple(make_pass_through_transform(N0),
                   make_merge_transform(make_tuple(Z, Y, X, C)),
                   make_merge_transform(make_tuple(N1, Do, Ho, Wo))),
        make_tuple(Sequence<0>{}, Sequence<2, 4, 6, 8>{}, Sequence<1, 3, 5, 7>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

    const auto in_n0_gemmk0_gemmm_gemmk1_grid_desc =
        transform_tensor_descriptor(in_n0_gemmk_gemmm_grid_desc,
                                    make_tuple(make_pass_through_transform(N0),
                                               make_unmerge_transform(make_tuple(GemmK0, GemmK1)),
                                               make_pass_through_transform(GemmM)),
                                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                                    make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2>{}));

    // B: weight tensor
    const auto wei_gemmk_gemmn_grid_desc = transform_tensor_descriptor(
        make_naive_tensor_descriptor_packed(make_tuple(K, Z * Y * X * C)),
        make_tuple(make_pass_through_transform(K), make_pass_through_transform(Z * Y * X * C)),
        make_tuple(Sequence<0>{}, Sequence<1>{}),
        make_tuple(Sequence<1>{}, Sequence<0>{}));

    const auto wei_gemmk0_gemmn_gemmk1_grid_desc =
        transform_tensor_descriptor(wei_gemmk_gemmn_grid_desc,
                                    make_tuple(make_unmerge_transform(make_tuple(GemmK0, GemmK1)),
                                               make_pass_through_transform(GemmN)),
                                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                                    make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

    // C: output tensor
    const auto out_n0_gemmm_gemmn_grid_desc = transform_tensor_descriptor(
        make_naive_tensor_descriptor_packed<true>(make_tuple(N0, N1 * Do * Ho * Wo, K)),
        make_tuple(make_pass_through_transform(N0),
                   make_pass_through_transform(N1 * Do * Ho * Wo),
                   make_pass_through_transform(K)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

    return make_tuple(in_n0_gemmk0_gemmm_gemmk1_grid_desc,
                      wei_gemmk0_gemmn_gemmk1_grid_desc,
                      out_n0_gemmm_gemmn_grid_desc);
}

} // namespace ck
#endif
