#ifndef CK_GRIDWISE_CONVOLUTION_BACKWARD_DATA_IMPLICIT_GEMM_V2R1_NCHW_KCYX_NKHW_HPP
#define CK_GRIDWISE_CONVOLUTION_BACKWARD_DATA_IMPLICIT_GEMM_V2R1_NCHW_KCYX_NKHW_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm.hpp"

namespace ck {

// GemmK = K * Ydot * Xdot;
// GemmM = C * Ytilda * Xtilda;
// GemmN = N * Htilda * Wtilda;
template <index_t GridSize,
          index_t BlockSize,
          typename Float,
          typename AccFloat,
          typename InGlobalDesc,
          typename WeiGlobalDesc,
          typename OutGlobalDesc,
          typename ConvStrides,
          typename ConvDilations,
          typename LeftPads,
          typename RightPads,
          index_t GemmMPerBlock,
          index_t GemmNPerBlock,
          index_t GemmKPerBlock,
          index_t GemmMPerThreadSubC,
          index_t GemmNPerThreadSubC,
          index_t GemmMLevel0Cluster,
          index_t GemmNLevel0Cluster,
          index_t GemmMLevel1Cluster,
          index_t GemmNLevel1Cluster,
          index_t GemmKPerThreadLoop,
          index_t GemmThreadGemmDataPerReadM,
          index_t GemmThreadGemmDataPerReadN,
          typename GemmABlockCopySubLengths,     // Gemm-K, Gemm-M
          typename GemmABlockCopyClusterLengths, // Gemm-K, Gemm-M
          index_t GemmABlockCopyDataPerAccess,   // Gemm-M
          typename GemmBBlockCopySubLengths,     // Gemm-K, Gemm-N
          typename GemmBBlockCopyClusterLengths, // Gemm-K, Gemm-N
          index_t GemmBBlockCopyDataPerAccess,   // Gemm-N
          index_t GemmCThreadCopyDataPerAccess   // Gemm-N
          >
struct GridwiseConvolutionBackwardDataImplicitGemm_v2r1_nchw_kcyx_nkhw
{
    __device__ void Run(Float* __restrict__ p_in_global,
                        const Float* __restrict__ p_wei_global,
                        const Float* __restrict__ p_out_global) const
    {
        constexpr auto in_n_c_hi_wi_global_desc  = InGlobalDesc{};
        constexpr auto wei_k_c_y_x_global_desc   = WeiGlobalDesc{};
        constexpr auto out_n_k_ho_wo_global_desc = OutGlobalDesc{};

        constexpr index_t N  = in_n_c_hi_wi_global_desc.GetLengths()[0];
        constexpr index_t C  = in_n_c_hi_wi_global_desc.GetLengths()[1];
        constexpr index_t Hi = in_n_c_hi_wi_global_desc.GetLengths()[2];
        constexpr index_t Wi = in_n_c_hi_wi_global_desc.GetLengths()[3];

        constexpr index_t K  = out_n_k_ho_wo_global_desc.GetLengths()[1];
        constexpr index_t Ho = out_n_k_ho_wo_global_desc.GetLengths()[2];
        constexpr index_t Wo = out_n_k_ho_wo_global_desc.GetLengths()[3];

        constexpr index_t Y = wei_k_c_y_x_global_desc.GetLengths()[2];
        constexpr index_t X = wei_k_c_y_x_global_desc.GetLengths()[3];

        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];

        // sanity-check for vectorized memory load
        static_assert((Wo == 1 || (ConvStrideW == 1 || GemmCThreadCopyDataPerAccess == 1)) &&
                          (X == 1 || ConvDilationW % GemmCThreadCopyDataPerAccess == 0),
                      "wrong! aligment requirement for vectorized global load of input tensor will "
                      "be violated");

        constexpr index_t hcf_stride_dilation_h = math::hcf(ConvStrideH, ConvDilationH);
        constexpr index_t hcf_stride_dilation_w = math::hcf(ConvStrideW, ConvDilationW);

        constexpr index_t Ytilda = ConvStrideH / hcf_stride_dilation_h;
        constexpr index_t Xtilda = ConvStrideW / hcf_stride_dilation_w;

        constexpr index_t Ydot = math::integer_divide_ceil(Y, Ytilda);
        constexpr index_t Xdot = math::integer_divide_ceil(X, Xtilda);

        constexpr index_t right_pad_ho = (ConvDilationH / hcf_stride_dilation_h) * (Y - Ytilda);
        constexpr index_t right_pad_wo = (ConvDilationW / hcf_stride_dilation_w) * (X - Xtilda);

        constexpr index_t Htilda = Ho + right_pad_ho;
        constexpr index_t Wtilda = Wo + right_pad_wo;

        // weight tensor
        constexpr auto wei_k_c_yp_xp_global_desc = transform_tensor_descriptor(
            wei_k_c_y_x_global_desc,
            make_tuple(PassThrough<K>{},
                       PassThrough<C>{},
                       Pad<Sequence<Y, X>,
                           Sequence<0, 0>,
                           Sequence<Ydot * Ytilda - Y, Xdot * Xtilda - X>,
                           true>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}));

        constexpr auto wei_k_c_ydot_ytilda_xdot_xtilda_global_desc = transform_tensor_descriptor(
            wei_k_c_yp_xp_global_desc,
            make_tuple(PassThrough<K>{},
                       PassThrough<C>{},
                       Embed<Sequence<Ydot, Ytilda>,
                             Sequence<ConvStrideH / hcf_stride_dilation_h, 1, 0>>{},
                       Embed<Sequence<Xdot, Xtilda>,
                             Sequence<ConvStrideW / hcf_stride_dilation_w, 1, 0>>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

        constexpr auto wei_gemmk_gemmm_global_desc = transform_tensor_descriptor(
            wei_k_c_ydot_ytilda_xdot_xtilda_global_desc,
            make_tuple(Merge<Sequence<K, Ydot, Xdot>>{}, Merge<Sequence<C, Ytilda, Xtilda>>{}),
            make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        // output tensor
        constexpr auto out_n_k_hop_wop_global_desc =
            transform_tensor_descriptor(out_n_k_ho_wo_global_desc,
                                        make_tuple(PassThrough<N>{},
                                                   PassThrough<K>{},
                                                   Pad<Sequence<Ho, Wo>,
                                                       Sequence<0, 0>,
                                                       Sequence<right_pad_ho, right_pad_wo>,
                                                       true>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}));

        constexpr auto out_n_k_ydot_htilda_xdot_wtilda_global_desc = transform_tensor_descriptor(
            out_n_k_hop_wop_global_desc,
            make_tuple(PassThrough<N>{},
                       PassThrough<K>{},
                       Embed<Sequence<Ydot, Htilda>,
                             Sequence<-ConvDilationH / hcf_stride_dilation_h, 1, 0>>{},
                       Embed<Sequence<Xdot, Wtilda>,
                             Sequence<-ConvDilationW / hcf_stride_dilation_w, 1, 0>>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

        constexpr auto out_gemmk_gemmn_global_desc = transform_tensor_descriptor(
            out_n_k_ydot_htilda_xdot_wtilda_global_desc,
            make_tuple(Merge<Sequence<K, Ydot, Xdot>>{}, Merge<Sequence<N, Htilda, Wtilda>>{}),
            make_tuple(Sequence<1, 2, 4>{}, Sequence<0, 3, 5>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        // input tensor
        constexpr auto in_n_c_hip_wip_global_desc = transform_tensor_descriptor(
            in_n_c_hi_wi_global_desc,
            make_tuple(PassThrough<N>{},
                       PassThrough<C>{},
                       Pad<Sequence<Hi, Wi>, LeftPads, RightPads, true>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}));

        constexpr auto in_n_c_ytilda_htilda_xtilda_wtilda_global_desc = transform_tensor_descriptor(
            in_n_c_hip_wip_global_desc,
            make_tuple(PassThrough<N>{},
                       PassThrough<C>{},
                       Embed<Sequence<Ytilda, Htilda>, Sequence<ConvDilationH, ConvStrideH, 0>>{},
                       Embed<Sequence<Xtilda, Wtilda>, Sequence<ConvDilationW, ConvStrideW, 0>>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

        constexpr auto in_gemmm_gemmn_global_desc = transform_tensor_descriptor(
            in_n_c_ytilda_htilda_xtilda_wtilda_global_desc,
            make_tuple(Merge<Sequence<C, Ytilda, Xtilda>>{}, Merge<Sequence<N, Htilda, Wtilda>>{}),
            make_tuple(Sequence<1, 2, 4>{}, Sequence<0, 3, 5>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        // GEMM
        constexpr auto gridwise_gemm =
            GridwiseGemmTransposedANormalBNormalC_v1r1<GridSize,
                                                       BlockSize,
                                                       Float,
                                                       AccFloat,
                                                       decltype(wei_gemmk_gemmm_global_desc),
                                                       decltype(out_gemmk_gemmn_global_desc),
                                                       decltype(in_gemmm_gemmn_global_desc),
                                                       InMemoryDataOperation::none,
                                                       GemmMPerBlock,
                                                       GemmNPerBlock,
                                                       GemmKPerBlock,
                                                       GemmMPerThreadSubC,
                                                       GemmNPerThreadSubC,
                                                       GemmMLevel0Cluster,
                                                       GemmNLevel0Cluster,
                                                       GemmMLevel1Cluster,
                                                       GemmNLevel1Cluster,
                                                       GemmKPerThreadLoop,
                                                       GemmThreadGemmDataPerReadM,
                                                       GemmThreadGemmDataPerReadN,
                                                       GemmABlockCopySubLengths,
                                                       GemmABlockCopyClusterLengths,
                                                       GemmABlockCopyDataPerAccess,
                                                       GemmBBlockCopySubLengths,
                                                       GemmBBlockCopyClusterLengths,
                                                       GemmBBlockCopyDataPerAccess,
                                                       GemmCThreadCopyDataPerAccess>{};

        gridwise_gemm.Run(p_wei_global, p_out_global, p_in_global);
    }
};

} // namespace ck
#endif
