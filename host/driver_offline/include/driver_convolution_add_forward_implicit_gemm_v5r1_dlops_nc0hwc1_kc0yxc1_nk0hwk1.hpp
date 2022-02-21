#ifndef DRIVER_CONVOLUTION_ADD_FORWARD_IMPLICIT_GEMM_V5R1_DLOPS_NC0HWc1_KC0YXC1_NK0HWK1_HPP
#define DRIVER_CONVOLUTION_ADD_FORWARD_IMPLICIT_GEMM_V5R1_DLOPS_NC0HWc1_KC0YXC1_NK0HWK1_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm_dlops_v3.hpp"

template <ck::index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          ck::index_t E1_,
          ck::index_t E2_,
          ck::index_t K2_,
          ck::index_t KPerBlock,
          ck::index_t HoPerBlock,
          ck::index_t WoPerBlock,
          ck::index_t E1PerBlock,
          ck::index_t KPerThread,
          ck::index_t HoPerThread,
          ck::index_t WoPerThread,
          ck::index_t EPerThread,
          typename ABlockTransferThreadSliceLengths_E0_E1_K0_K1_E2,
          typename ABlockTransferThreadClusterLengths_E0_E1_K0_K1_E2,
          ck::index_t ABlockTransferSrcScalarPerVector_E2,
          ck::index_t ABlockTransferDstScalarPerVector_E2,
          ck::index_t BThreadTransferSrcScalarPerVector_E2,
          ck::index_t CThreadTransferDstScalarPerVector_K,
          ck::ActivTypeEnum_t activ_type>
struct DriverDynamicConvolutionForwardImplicitGemmDlops_v5r1_nc0hwc1_kc0yxc1_nk0hwk1_add
{
    template <typename... Wei,
              typename... In,
              typename... Add,
              typename... Out,
              typename ConvStrides,
              typename ConvDilations,
              typename InLeftPads,
              typename InRightPads>
    __host__ float Run(const ck::TensorDescriptor<Wei...>& wei_k_c0_y_x_c1_global_desc,
                       const ck::TensorDescriptor<In...>& in_n_c0_hi_wi_c1_global_desc,
                       const ck::TensorDescriptor<Out...>& out_n_k0_ho_wo_k1_global_desc,
                       const ck::TensorDescriptor<Add...>& add_n_k0_hox2_wox2_k1_global_desc,
                       const ConvStrides& conv_strides,
                       const ConvDilations& conv_dilations,
                       const InLeftPads& in_left_pads,
                       const InRightPads& in_right_pads,
                       const FloatAB* __restrict__ p_a_grid,
                       const FloatAB* __restrict__ p_b_grid,
                       const FloatC* __restrict__ p_bias_grid,
                       FloatC* __restrict__ p_d_grid,
                       const int nrepeat) const
    {
        using namespace ck;

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};

        const auto N  = in_n_c0_hi_wi_c1_global_desc.GetLength(I0);
        const auto C0 = in_n_c0_hi_wi_c1_global_desc.GetLength(I1);
        const auto Hi = in_n_c0_hi_wi_c1_global_desc.GetLength(I2);
        const auto Wi = in_n_c0_hi_wi_c1_global_desc.GetLength(I3);
        // const auto C1 = in_n_c0_hi_wi_c1_global_desc.GetLength(I4);

        const auto K0 = out_n_k0_ho_wo_k1_global_desc.GetLength(I1);
        const auto Ho = out_n_k0_ho_wo_k1_global_desc.GetLength(I2);
        const auto Wo = out_n_k0_ho_wo_k1_global_desc.GetLength(I3);
        const auto K1 = out_n_k0_ho_wo_k1_global_desc.GetLength(I4);

        const auto Hox2 = add_n_k0_hox2_wox2_k1_global_desc.GetLength(I2);
        const auto Wox2 = add_n_k0_hox2_wox2_k1_global_desc.GetLength(I3);

        const auto K = wei_k_c0_y_x_c1_global_desc.GetLength(I0);
        const auto Y = wei_k_c0_y_x_c1_global_desc.GetLength(I2);
        const auto X = wei_k_c0_y_x_c1_global_desc.GetLength(I3);

        const auto ConvStrideH = conv_strides[I0];
        const auto ConvStrideW = conv_strides[I1];

        const auto ConvDilationH = conv_dilations[I0];
        const auto ConvDilationW = conv_dilations[I1];

#if CK_EXPERIMENTAL_STATIC_TENSOR_DESCRIPTOR
        const auto Hop = Number<(Ho + HoPerBlock - 1) / HoPerBlock * HoPerBlock>{};
        const auto Wop = Number<(Wo + WoPerBlock - 1) / WoPerBlock * WoPerBlock>{};

        const auto OutRightPadH = Hop - Ho;
        const auto OutRightPadW = Wop - Wo;

        const auto OutRightPadHx = Number<OutRightPadH * 2>{};
        const auto OutRightPadWx = Number<OutRightPadW * 2>{};
#else
        const auto Hop = (Ho + HoPerBlock - 1) / HoPerBlock * HoPerBlock;
        const auto Wop = (Wo + WoPerBlock - 1) / WoPerBlock * WoPerBlock;

        const auto OutRightPadH = Hop - Ho;
        const auto OutRightPadW = Wop - Wo;

        const auto OutRightPadHx = OutRightPadH * 2;
        const auto OutRightPadWx = OutRightPadW * 2;
#endif

        const auto InLeftPadH = in_left_pads[I0];
        const auto InLeftPadW = in_left_pads[I1];

        const auto InRightPadH = in_right_pads[I0] + OutRightPadH * ConvStrideH;
        const auto InRightPadW = in_right_pads[I1] + OutRightPadW * ConvStrideW;

        const auto E = C0 * Y * X;

        constexpr auto E1 = Number<E1_>{};
        constexpr auto E2 = Number<E2_>{};
        constexpr auto K2 = Number<K2_>{};

        const auto E0 = E / E1;

        // weight tensor
        const auto a_e_k_e2_grid_desc = transform_tensor_descriptor(
            make_naive_tensor_descriptor_packed(make_tuple(K, C0 * Y * X, E2)),
            make_tuple(make_pass_through_transform(K),
                       make_pass_through_transform(C0 * Y * X),
                       make_pass_through_transform(E2)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<1>{}, Sequence<0>{}, Sequence<2>{}));

        const auto a_e0_e1_k_e2_grid_desc =
            transform_tensor_descriptor(a_e_k_e2_grid_desc,
                                        make_tuple(make_unmerge_transform(make_tuple(E0, E1)),
                                                   make_pass_through_transform(K),
                                                   make_pass_through_transform(E2)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                                        make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}));

        // input tensor
        const auto in_n_c0_hip_wip_e2_global_desc = transform_tensor_descriptor(
            make_naive_tensor_descriptor_packed(make_tuple(N, C0, Hi, Wi, E2)),
            make_tuple(make_pass_through_transform(N),
                       make_pass_through_transform(C0),
                       make_pad_transform(Hi, InLeftPadH, InRightPadH),
                       make_pad_transform(Wi, InLeftPadW, InRightPadW),
                       make_pass_through_transform(E2)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

        const auto in_n_c0_y_ho_x_wo_e2_global_desc = transform_tensor_descriptor(
            in_n_c0_hip_wip_e2_global_desc,
            make_tuple(
                make_pass_through_transform(N),
                make_pass_through_transform(C0),
                make_embed_transform(make_tuple(Y, Hop), make_tuple(ConvDilationH, ConvStrideH)),
                make_embed_transform(make_tuple(X, Wop), make_tuple(ConvDilationW, ConvStrideW)),
                make_pass_through_transform(E2)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(
                Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}, Sequence<6>{}));

        const auto in_e_n_ho_wo_e2_grid_desc = transform_tensor_descriptor(
            in_n_c0_y_ho_x_wo_e2_global_desc,
            make_tuple(make_merge_transform(make_tuple(C0, Y, X)),
                       make_pass_through_transform(N),
                       make_pass_through_transform(Hop),
                       make_pass_through_transform(Wop),
                       make_pass_through_transform(E2)),
            make_tuple(
                Sequence<1, 2, 4>{}, Sequence<0>{}, Sequence<3>{}, Sequence<5>{}, Sequence<6>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

        const auto b_e0_e1_n_ho_wo_e2_grid_desc = transform_tensor_descriptor(
            in_e_n_ho_wo_e2_grid_desc,
            make_tuple(make_unmerge_transform(make_tuple(E0, E1)),
                       make_pass_through_transform(N),
                       make_pass_through_transform(Hop),
                       make_pass_through_transform(Wop),
                       make_pass_through_transform(E2)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(
                Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}, Sequence<5>{}));

        // output tensor
        const auto c_k_n_hop_wop_grid_desc = transform_tensor_descriptor(
            make_naive_tensor_descriptor_packed(make_tuple(N, K0, Ho, Wo, K1)),
            make_tuple(make_merge_transform(make_tuple(K0, K1)),
                       make_pass_through_transform(N),
                       make_pad_transform(Ho, I0, OutRightPadH),
                       make_pad_transform(Wo, I0, OutRightPadW)),
            make_tuple(Sequence<1, 4>{}, Sequence<0>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        // add tensor
        const auto d_k_n_hopx2_wopx2_grid_desc = transform_tensor_descriptor(
            make_naive_tensor_descriptor_packed(make_tuple(N, K0, Hox2, Wox2, K1)),
            make_tuple(make_merge_transform(make_tuple(K0, K1)),
                       make_pass_through_transform(N),
                       make_pad_transform(Hox2, I0, OutRightPadHx),
                       make_pad_transform(Wox2, I0, OutRightPadWx)),
            make_tuple(Sequence<1, 4>{}, Sequence<0>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        std::cerr << "Hop = " << Hop << " Wop = " << Wop << std::endl;

        if(!((K % KPerBlock) == 0 && (Hop % HoPerBlock) == 0 && (Wop % WoPerBlock) == 0 &&
             (E1 % E1PerBlock) == 0))
        {
            throw std::runtime_error("wrong! GEMM size no divisible");
        }

        // clang-format off

        // hack to control index calculation when iterating over a_e0_e1_k_e2_global tensor
        constexpr auto a_e0_e1_k_e2_global_step_hacks =
            make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}),
                       make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}));

        constexpr auto a_e0_e1_k_e2_global_move_slice_window_step_hack =
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{};

        // hack to control index calculation when iterating over b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_global tensor
        constexpr auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_global_step_hacks = 
            make_tuple(
                make_tuple(
                    Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}, 
                    Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}, 
                    Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}, 
                    Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}, 
                    Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}, 
                    Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}, 
                    Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}, 
                    Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}, 
                    Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}, 
                    Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}), 
                make_tuple(
                    Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}, 
                    Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}, 
                    Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}, 
                    Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}, 
                    Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}, 
                    Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}, 
                    Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}, 
                    Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}, 
                    Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}, 
                    Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{})
            ); 

        constexpr auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_global_move_slice_window_step_hack =
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{};

        // hack to control index calculation when iterating over c_k0_k1_n_h0_h1_h2_w0_w1_w2_global tensor
        constexpr auto c_k0_k1_n_h0_h1_h2_w0_w1_w2_global_tensor_step_hacks =
            make_tuple(make_tuple(Sequence<0, 1, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 1, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{}),
                       make_tuple(Sequence<0, 2, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 2, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{}));

        constexpr auto d_k0_k1_n_h0_h1_h2x2_w0_w1_w2x2_global_tensor_step_hacks =
            make_tuple(make_tuple(Sequence<0, 1, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 1, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{}),
                       make_tuple(Sequence<0, 2, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 2, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0>{}));

        // clang-format on

        // GEMM
        using GridwiseGemm = GridwiseGemmDlops_km_kn_mn_v3<
            BlockSize,
            FloatAB,
            FloatAcc,
            FloatC,
            InMemoryDataOperationEnum_t::Set,
            decltype(a_e0_e1_k_e2_grid_desc),
            decltype(b_e0_e1_n_ho_wo_e2_grid_desc),
            decltype(c_k_n_hop_wop_grid_desc),
            decltype(d_k_n_hopx2_wopx2_grid_desc),
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
            Sequence<2, 3, 0, 1, 4>,
            Sequence<0, 1, 2, 3, 4>,
            4,
            ABlockTransferSrcScalarPerVector_E2,
            ABlockTransferDstScalarPerVector_E2,
            false, // don't move back src coordinate after threadwise copy
            Sequence<0, 1, 2, 3, 4, 5, 6, 7, 8, 9>, // E0, E1, N, H0, H1, H2, W0, W1, W2, E2
            9,
            BThreadTransferSrcScalarPerVector_E2,
            false, // don't move back src coordinate after threadwise copy, which will be fused with
                   // MoveSrcSliceWindow() to save addr computation
            Sequence<0, 1, 2, 3, 4, 5, 6, 7, 8>, // K0, K1, N, H0, H1, I2, H2, W0, W1, I2, W2
            1,
            CThreadTransferDstScalarPerVector_K,
            decltype(a_e0_e1_k_e2_global_step_hacks),
            decltype(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_global_step_hacks),
            decltype(c_k0_k1_n_h0_h1_h2_w0_w1_w2_global_tensor_step_hacks),
            decltype(d_k0_k1_n_h0_h1_h2x2_w0_w1_w2x2_global_tensor_step_hacks),
            decltype(a_e0_e1_k_e2_global_move_slice_window_step_hack),
            decltype(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_global_move_slice_window_step_hack)>;

        const auto a_e0_e1_k0_k1_e2_grid_desc =
            GridwiseGemm::MakeAE0E1K0K1E2GridDescriptor(a_e0_e1_k_e2_grid_desc);
        const auto b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc =
            GridwiseGemm::MakeBE0E1NH0H1H2W0W1W2E2GridDescriptor(b_e0_e1_n_ho_wo_e2_grid_desc);
        const auto c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc =
            GridwiseGemm::MakeCK0K1NH0H1H2W0W1W2GridDescriptor(c_k_n_hop_wop_grid_desc);
        const auto d_k0_k1_n_h0_h1_h2x2_w0_w1_w2x2_grid_desc =
            GridwiseGemm::MakeDK0K1NH0H1HxW0W1WxGridDescriptorResizeAdd(
                d_k_n_hopx2_wopx2_grid_desc);

        using AGridDesc_E0_E1_K0_K1_E2 = decltype(a_e0_e1_k0_k1_e2_grid_desc);
        using BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2 =
            decltype(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc);
        using CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2 = decltype(c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc);
        using DGridDesc_K0_K1_N_H0_H1_H2x2_W0_W1_W2x2 =
            decltype(d_k0_k1_n_h0_h1_h2x2_w0_w1_w2x2_grid_desc);

        const auto grid_size = (K / KPerBlock) * (Hop / HoPerBlock) * (Wop / WoPerBlock) * N;

        const bool has_main_e0_block_loop = E0 > 1;

        std::cerr << "has_main_e0_block_loop = " << has_main_e0_block_loop << std::endl;

        const auto cblockid_to_k_n_h_w_block_cluster_adaptor =
            GridwiseGemm::MakeCBlockIdToKNHoWoBlockClusterAdaptor(c_k_n_hop_wop_grid_desc);

        using CBlockIdToBlockClusterAdaptor_K_N_H_W =
            decltype(cblockid_to_k_n_h_w_block_cluster_adaptor);

        float ave_time = 0;

#if CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VALUE

        if(has_main_e0_block_loop)
        {
            const auto kernel = kernel_gemm_dlops_v3_resize_add<
                GridwiseGemm,
                FloatAB,
                FloatC,
                remove_reference_t<AGridDesc_E0_E1_K0_K1_E2>,
                remove_reference_t<BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2>,
                remove_reference_t<CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2>,
                remove_reference_t<DGridDesc_K0_K1_N_H0_H1_H2x2_W0_W1_W2x2>,
                remove_reference_t<CBlockIdToBlockClusterAdaptor_K_N_H_W>,
                true,
                activ_type>;

            ave_time = launch_and_time_kernel(kernel,
                                              nrepeat,
                                              dim3(grid_size),
                                              dim3(BlockSize),
                                              0,
                                              p_a_grid,
                                              p_b_grid,
                                              p_bias_grid,
                                              p_d_grid,
                                              a_e0_e1_k0_k1_e2_grid_desc,
                                              b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                                              c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc,
                                              d_k0_k1_n_h0_h1_h2x2_w0_w1_w2x2_grid_desc,
                                              cblockid_to_k_n_h_w_block_cluster_adaptor);
        }
        else
        {
            const auto kernel = kernel_gemm_dlops_v3_resize_add<
                GridwiseGemm,
                FloatAB,
                FloatC,
                remove_reference_t<AGridDesc_E0_E1_K0_K1_E2>,
                remove_reference_t<BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2>,
                remove_reference_t<CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2>,
                remove_reference_t<DGridDesc_K0_K1_N_H0_H1_H2x2_W0_W1_W2x2>,
                remove_reference_t<CBlockIdToBlockClusterAdaptor_K_N_H_W>,
                false,
                activ_type>;

            ave_time = launch_and_time_kernel(kernel,
                                              nrepeat,
                                              dim3(grid_size),
                                              dim3(BlockSize),
                                              0,
                                              p_a_grid,
                                              p_b_grid,
                                              p_bias_grid,
                                              p_d_grid,
                                              a_e0_e1_k0_k1_e2_grid_desc,
                                              b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc,
                                              c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc,
                                              d_k0_k1_n_h0_h1_h2x2_w0_w1_w2x2_grid_desc,
                                              cblockid_to_k_n_h_w_block_cluster_adaptor);
        }

#elif CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VOID_POINTER
        DeviceMem a_e0_e1_k0_k1_e2_grid_desc_dev_buf(sizeof(AGridDesc_E0_E1_K0_K1_E2));
        DeviceMem b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc_dev_buf(
            sizeof(BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2));
        DeviceMem c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc_dev_buf(
            sizeof(CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2));
        DeviceMem d_k0_k1_n_h0_h1_h2x2_w0_w1_w2x2_grid_desc_dev_buf(
            sizeof(DGridDesc_K0_K1_N_H0_H1_H2x2_W0_W1_W2x2));
        DeviceMem cblockid_to_k_n_h_w_block_cluster_adaptor_dev_buf(
            sizeof(CBlockIdToBlockClusterAdaptor_K_N_H_W));

        a_e0_e1_k0_k1_e2_grid_desc_dev_buf.ToDevice(&a_e0_e1_k0_k1_e2_grid_desc);
        b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc_dev_buf.ToDevice(
            &b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc);
        c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc_dev_buf.ToDevice(
            &c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc);
        d_k0_k1_n_h0_h1_h2x2_w0_w1_w2x2_grid_desc_dev_buf.ToDevice(
            &d_k0_k1_n_h0_h1_h2x2_w0_w1_w2x2_grid_desc);
        cblockid_to_k_n_h_w_block_cluster_adaptor_dev_buf.ToDevice(
            &cblockid_to_k_n_h_w_block_cluster_adaptor);

        if(has_main_e0_block_loop)
        {

            const auto kernel = kernel_gemm_dlops_v3_resize_add<
                GridwiseGemm,
                FloatAB,
                FloatC,
                remove_reference_t<AGridDesc_E0_E1_K0_K1_E2>,
                remove_reference_t<BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2>,
                remove_reference_t<CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2>,
                remove_reference_t<DGridDesc_K0_K1_N_H0_H1_H2x2_W0_W1_W2x2>,
                remove_reference_t<CBlockIdToBlockClusterAdaptor_K_N_H_W>,
                true,
                activ_type>;

            ave_time = launch_and_time_kernel(
                kernel,
                nrepeat,
                dim3(grid_size),
                dim3(BlockSize),
                0,
                p_a_grid,
                p_b_grid,
                p_bias_grid,
                p_d_grid,
                cast_pointer_to_constant_address_space(
                    a_e0_e1_k0_k1_e2_grid_desc_dev_buf.GetDeviceBuffer()),
                cast_pointer_to_constant_address_space(
                    b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc_dev_buf.GetDeviceBuffer()),
                cast_pointer_to_constant_address_space(
                    c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc_dev_buf.GetDeviceBuffer()),
                cast_pointer_to_constant_address_space(
                    d_k0_k1_n_h0_h1_h2x2_w0_w1_w2x2_grid_desc_dev_buf.GetDeviceBuffer()),
                cast_pointer_to_constant_address_space(
                    cblockid_to_k_n_h_w_block_cluster_adaptor_dev_buf.GetDeviceBuffer()));
        }
        else
        {
            const auto kernel = kernel_gemm_dlops_v3_resize_add<
                GridwiseGemm,
                FloatAB,
                FloatC,
                remove_reference_t<AGridDesc_E0_E1_K0_K1_E2>,
                remove_reference_t<BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2>,
                remove_reference_t<CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2>,
                remove_reference_t<DGridDesc_K0_K1_N_H0_H1_H2x2_W0_W1_W2x2>,
                remove_reference_t<CBlockIdToBlockClusterAdaptor_K_N_H_W>,
                false,
                activ_type>;

            ave_time = launch_and_time_kernel(
                kernel,
                nrepeat,
                dim3(grid_size),
                dim3(BlockSize),
                0,
                p_a_grid,
                p_b_grid,
                p_bias_grid,
                p_d_grid,
                cast_pointer_to_constant_address_space(
                    a_e0_e1_k0_k1_e2_grid_desc_dev_buf.GetDeviceBuffer()),
                cast_pointer_to_constant_address_space(
                    b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc_dev_buf.GetDeviceBuffer()),
                cast_pointer_to_constant_address_space(
                    c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc_dev_buf.GetDeviceBuffer()),
                cast_pointer_to_constant_address_space(
                    d_k0_k1_n_h0_h1_h2x2_w0_w1_w2x2_grid_desc_dev_buf.GetDeviceBuffer()),
                cast_pointer_to_constant_address_space(
                    cblockid_to_k_n_h_w_block_cluster_adaptor_dev_buf.GetDeviceBuffer()));
        }
#elif CK_EXPERIMENTAL_STATIC_TENSOR_DESCRIPTOR
        {
            static_assert(a_e0_e1_k_e2_grid_desc.IsKnownAtCompileTime(), "");
            static_assert(b_e0_e1_n_h0_h1_h2_w0_w1_w2_e2_grid_desc.IsKnownAtCompileTime(), "");
            static_assert(d_k0_k1_n_h0_h1_h2x2_w0_w1_w2x2_grid_desc.IsKnownAtCompileTime(), "");
            static_assert(c_k0_k1_n_h0_h1_h2_w0_w1_w2_grid_desc.IsKnownAtCompileTime(), "");
            static_assert(cblockid_to_k_n_h_w_block_cluster_adaptor.IsKnownAtCompileTime(), "");

            const auto kernel = kernel_gemm_dlops_v3_resize_add<
                GridwiseGemm,
                FloatAB,
                FloatC,
                remove_reference_t<AGridDesc_E0_E1_K0_K1_E2>,
                remove_reference_t<BGridDesc_E0_E1_N_H0_H1_H2_W0_W1_W2_E2>,
                remove_reference_t<CGridDesc_K0_K1_N_H0_H1_H2_W0_W1_W2>,
                remove_reference_t<DGridDesc_K0_K1_N_H0_H1_H2x2_W0_W1_W2x2>,
                remove_reference_t<CBlockIdToBlockClusterAdaptor_K_N_H_W>,
                has_main_e0_block_loop,
                activ_type>;

            ave_time = launch_and_time_kernel(kernel,
                                              nrepeat,
                                              dim3(grid_size),
                                              dim3(BlockSize),
                                              0,
                                              p_a_grid,
                                              p_b_grid,
                                              p_bias_grid,
                                              p_d_grid);
        }
#endif
        return ave_time;
    }
};
#endif
