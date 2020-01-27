#ifndef CK_GRIDWISE_CONVOLUTION_BACKWARD_DATA_IMPLICIT_GEMM_V2R1_NCHW_KCYX_NKHW_HPP
#define CK_GRIDWISE_CONVOLUTION_BACKWARD_DATA_IMPLICIT_GEMM_V2R1_NCHW_KCYX_NKHW_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm.hpp"

namespace ck {

// GemmM = C * Ytilda * Xtilda;
// GemmN = N * HtildaNonZero * WtildaNonZero;
// GemmK = K * Ydot * Xdot;
template <index_t GridSize,
          index_t BlockSize,
          typename Float,
          typename AccFloat,
          typename InGlobalDesc,
          typename WeiGlobalDesc,
          typename OutGlobalDesc,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads,
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
          typename GemmABlockCopyThreadSliceLengths_GemmK_GemmM,
          typename GemmABlockCopyThreadClusterLengths_GemmK_GemmM,
          index_t GemmABlockCopySrcDataPerRead_GemmM,
          index_t GemmABlockCopyDstDataPerWrite_GemmM,
          typename GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
          typename GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
          index_t GemmBBlockCopySrcDataPerRead_GemmN,
          index_t GemmBBlockCopyDstDataPerWrite_GemmN,
          index_t GemmCThreadCopyDstDataPerWrite_GemmN1>
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

#if 0 // debug
        // sanity-check for vectorized memory load
        // TODO: this logic may not be correct for bwd-data
        static_assert(
            (Wo == 1 || (ConvStrideW == 1 || GemmCThreadCopyDstDataPerWrite_GemmN1 == 1)) &&
                (X == 1 || ConvDilationW % GemmCThreadCopyDstDataPerWrite_GemmN1 == 0),
            "wrong! aligment requirement for vectorized global load of input tensor will "
            "be violated");
#endif

        constexpr index_t gcd_stride_dilation_h = math::gcd(ConvStrideH, ConvDilationH);
        constexpr index_t gcd_stride_dilation_w = math::gcd(ConvStrideW, ConvDilationW);

        constexpr index_t Ytilda = ConvStrideH / gcd_stride_dilation_h;
        constexpr index_t Xtilda = ConvStrideW / gcd_stride_dilation_w;

        constexpr index_t Ydot = math::integer_divide_ceil(Y, Ytilda);
        constexpr index_t Xdot = math::integer_divide_ceil(X, Xtilda);

        constexpr index_t Htilda =
            Ho + math::integer_divide_ceil(ConvDilationH * (Y - 1), ConvStrideH);
        constexpr index_t Wtilda =
            Wo + math::integer_divide_ceil(ConvDilationW * (X - 1), ConvStrideW);

        constexpr index_t HtildaLeft = math::integer_divide_floor(
            math::max(0, InLeftPads{}[0] - ConvDilationH * (Ytilda - 1)), ConvStrides{}[0]);
        constexpr index_t WtildaLeft = math::integer_divide_floor(
            math::max(0, InLeftPads{}[1] - ConvDilationW * (Xtilda - 1)), ConvStrides{}[1]);

        constexpr index_t HtildaRight = math::min(
            Htilda, math::integer_divide_ceil(InLeftPads{}[0] + Hi - 1, ConvStrides{}[0]) + 1);
        constexpr index_t WtildaRight = math::min(
            Wtilda, math::integer_divide_ceil(InLeftPads{}[1] + Wi - 1, ConvStrides{}[1]) + 1);

        constexpr index_t HtildaTrim = HtildaRight - HtildaLeft;
        constexpr index_t WtildaTrim = WtildaRight - WtildaLeft;

        // weight tensor
        constexpr auto wei_k_c_ydot_ytilda_xdot_xtilda_global_desc = transform_tensor_descriptor(
            wei_k_c_y_x_global_desc,
            make_tuple(PassThrough<K>{},
                       PassThrough<C>{},
                       Embed<Y,
                             Sequence<Ydot, Ytilda>,
                             Sequence<ConvStrideH / gcd_stride_dilation_h, 1, 0>>{},
                       Embed<X,
                             Sequence<Xdot, Xtilda>,
                             Sequence<ConvStrideW / gcd_stride_dilation_w, 1, 0>>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

        constexpr auto wei_gemmk_gemmm_global_desc = transform_tensor_descriptor(
            wei_k_c_ydot_ytilda_xdot_xtilda_global_desc,
            make_tuple(Merge<Sequence<K, Ydot, Xdot>>{}, Merge<Sequence<C, Ytilda, Xtilda>>{}),
            make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        // output tensor
        constexpr auto out_n_k_ydot_htilda_xdot_wtilda_global_desc = transform_tensor_descriptor(
            out_n_k_ho_wo_global_desc,
            make_tuple(PassThrough<N>{},
                       PassThrough<K>{},
                       Embed<Ho,
                             Sequence<Ydot, Htilda>,
                             Sequence<-ConvDilationH / gcd_stride_dilation_h, 1, 0>>{},
                       Embed<Wo,
                             Sequence<Xdot, Wtilda>,
                             Sequence<-ConvDilationW / gcd_stride_dilation_w, 1, 0>>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

        constexpr auto out_n_k_ydot_htildatrim_xdot_wtildatrim_global_desc =
            transform_tensor_descriptor(
                out_n_k_ydot_htilda_xdot_wtilda_global_desc,
                make_tuple(PassThrough<N>{},
                           PassThrough<K>{},
                           PassThrough<Ytilda>{},
                           PassThrough<Xtilda>{},
                           Slice<Sequence<Htilda, Wtilda>,
                                 Sequence<HtildaLeft, WtildaLeft>,
                                 Sequence<HtildaRight, WtildaRight>>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<4>{}, Sequence<3, 5>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<4>{}, Sequence<3, 5>{}));

        constexpr auto out_gemmk_gemmn_global_desc =
            transform_tensor_descriptor(out_n_k_ydot_htildatrim_xdot_wtildatrim_global_desc,
                                        make_tuple(Merge<Sequence<K, Ydot, Xdot>>{},
                                                   Merge<Sequence<N, HtildaTrim, WtildaTrim>>{}),
                                        make_tuple(Sequence<1, 2, 4>{}, Sequence<0, 3, 5>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

#if 1 // debug
        constexpr bool in_skip_all_out_of_bound_check = false;
#else
        constexpr bool in_skip_all_out_of_bound_check = true;
#endif

        // input tensor
        constexpr auto in_n_c_hip_wip_global_desc = transform_tensor_descriptor(
            in_n_c_hi_wi_global_desc,
            make_tuple(
                PassThrough<N>{},
                PassThrough<C>{},
                Pad<Sequence<Hi, Wi>, InLeftPads, InRightPads, in_skip_all_out_of_bound_check>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}));

        constexpr index_t Hip = in_n_c_hip_wip_global_desc.GetLengths()[2];
        constexpr index_t Wip = in_n_c_hip_wip_global_desc.GetLengths()[3];

        constexpr auto in_n_c_ytilda_htilda_xtilda_wtilda_global_desc = transform_tensor_descriptor(
            in_n_c_hip_wip_global_desc,
            make_tuple(PassThrough<N>{},
                       PassThrough<C>{},
                       Embed<Hip,
                             Sequence<Ytilda, Htilda>,
                             Sequence<ConvDilationH, ConvStrideH, 0>,
                             in_skip_all_out_of_bound_check>{},
                       Embed<Wip,
                             Sequence<Xtilda, Wtilda>,
                             Sequence<ConvDilationW, ConvStrideW, 0>,
                             in_skip_all_out_of_bound_check>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

        constexpr auto in_n_c_ytilda_htildatrim_xtilda_wtildatrim_global_desc =
            transform_tensor_descriptor(
                in_n_c_ytilda_htilda_xtilda_wtilda_global_desc,
                make_tuple(PassThrough<N>{},
                           PassThrough<C>{},
                           PassThrough<Ytilda>{},
                           PassThrough<Xtilda>{},
                           Slice<Sequence<Htilda, Wtilda>,
                                 Sequence<HtildaLeft, WtildaLeft>,
                                 Sequence<HtildaRight, WtildaRight>>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<4>{}, Sequence<3, 5>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<4>{}, Sequence<3, 5>{}));

        constexpr auto in_gemmm_gemmn_global_desc =
            transform_tensor_descriptor(in_n_c_ytilda_htildatrim_xtilda_wtildatrim_global_desc,
                                        make_tuple(Merge<Sequence<C, Ytilda, Xtilda>>{},
                                                   Merge<Sequence<N, HtildaTrim, WtildaTrim>>{}),
                                        make_tuple(Sequence<1, 2, 4>{}, Sequence<0, 3, 5>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

        // GEMM
        constexpr auto gridwise_gemm =
            GridwiseGemmTransposedANormalBNormalC_v1<GridSize,
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
                                                     GemmABlockCopyThreadSliceLengths_GemmK_GemmM,
                                                     GemmABlockCopyThreadClusterLengths_GemmK_GemmM,
                                                     Sequence<0, 1>,
                                                     Sequence<0, 1>,
                                                     1,
                                                     GemmABlockCopySrcDataPerRead_GemmM,
                                                     GemmABlockCopyDstDataPerWrite_GemmM,
                                                     GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
                                                     GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
                                                     Sequence<0, 1>,
                                                     Sequence<0, 1>,
                                                     1,
                                                     GemmBBlockCopySrcDataPerRead_GemmN,
                                                     GemmBBlockCopyDstDataPerWrite_GemmN,
                                                     Sequence<0, 1, 2, 3>,
                                                     3,
                                                     GemmCThreadCopyDstDataPerWrite_GemmN1>{};

        gridwise_gemm.Run(p_wei_global, p_out_global, p_in_global);
    }
};

} // namespace ck
#endif
