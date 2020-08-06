#ifndef CK_GRIDWISE_CONVOLUTION_BACKWARD_DATA_IMPLICIT_GEMM_V5R1_NHWC_KYXC_NHWK_HPP
#define CK_GRIDWISE_CONVOLUTION_BACKWARD_DATA_IMPLICIT_GEMM_V5R1_NHWC_KYXC_NHWK_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm.hpp"

namespace ck {

// Number of GEMMs = YTilda * XTilda
// GemmM  = C
// GemmN  = N * HTildaSlice * WTildaSlice
// GemmK0 = YDotSlice
// GemmK1 = XDotSlice
// GemmK2 = K
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
          index_t GemmMPerThread,
          index_t GemmNPerThread,
          index_t GemmKPerThread,
          index_t GemmMLevel0Cluster,
          index_t GemmNLevel0Cluster,
          index_t GemmMLevel1Cluster,
          index_t GemmNLevel1Cluster,
          index_t ThreadGemmDataPerRead_GemmM,
          index_t ThreadGemmDataPerRead_GemmN,
          typename GemmABlockCopyThreadSliceLengths_GemmK0_GemmK1_GemmK2_GemmM,
          typename GemmABlockCopyThreadClusterLengths_GemmK0_GemmK1_GemmK2_GemmM,
          index_t GemmABlockCopySrcDataPerRead_GemmM,
          index_t GemmABlockCopyDstDataPerWrite_GemmM,
          typename GemmBBlockCopyThreadSliceLengths_GemmK0_GemmK1_GemmK2_GemmN,
          typename GemmBBlockCopyThreadClusterLengths_GemmK0_GemmK1_GemmK2_GemmN,
          index_t GemmBBlockCopySrcDataPerRead_GemmK2,
          index_t GemmBBlockCopyDstDataPerWrite_GemmN,
          index_t GemmCThreadCopyDstDataPerWrite_GemmN1>
struct GridwiseConvolutionBackwardDataImplicitGemm_v5r1_nhwc_kyxc_nhwk
{
    __host__ __device__ static constexpr index_t GetNumberOfGemm()
    {
        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];

        constexpr index_t GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
        constexpr index_t GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

        constexpr index_t YTilda = ConvStrideH / GcdStrideDilationH;
        constexpr index_t XTilda = ConvStrideW / GcdStrideDilationW;

        return YTilda * XTilda;
    }

    __host__ __device__ static constexpr auto GetGemmSizeImpl(index_t iYTilda, index_t iXTilda)
    {
        constexpr index_t N  = InGlobalDesc::GetLengths()[0];
        constexpr index_t Hi = InGlobalDesc::GetLengths()[1];
        constexpr index_t Wi = InGlobalDesc::GetLengths()[2];
        constexpr index_t C  = InGlobalDesc::GetLengths()[3];

        constexpr index_t Ho = OutGlobalDesc::GetLengths()[1];
        constexpr index_t Wo = OutGlobalDesc::GetLengths()[2];
        constexpr index_t K  = OutGlobalDesc::GetLengths()[3];

        constexpr index_t Y = WeiGlobalDesc::GetLengths()[1];
        constexpr index_t X = WeiGlobalDesc::GetLengths()[2];

        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];

        constexpr index_t GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
        constexpr index_t GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

        constexpr index_t YTilda = ConvStrideH / GcdStrideDilationH;
        constexpr index_t XTilda = ConvStrideW / GcdStrideDilationW;

        constexpr index_t YDot = math::integer_divide_ceil(Y, YTilda);
        constexpr index_t XDot = math::integer_divide_ceil(X, XTilda);

        constexpr index_t HTilda =
            Ho + math::integer_divide_ceil(ConvDilationH * (Y - 1), ConvStrideH);
        constexpr index_t WTilda =
            Wo + math::integer_divide_ceil(ConvDilationW * (X - 1), ConvStrideW);

        // only work on HTilda and WTilda that contribute to non-padding area of input tensor
        constexpr index_t iHTildaLeft = math::integer_divide_floor(
            math::max(0, InLeftPads{}[0] - ConvDilationH * (YTilda - 1)), ConvStrides{}[0]);
        constexpr index_t iWTildaLeft = math::integer_divide_floor(
            math::max(0, InLeftPads{}[1] - ConvDilationW * (XTilda - 1)), ConvStrides{}[1]);

        constexpr index_t iHTildaRight = math::min(
            HTilda, math::integer_divide_ceil(InLeftPads{}[0] + Hi - 1, ConvStrides{}[0]) + 1);
        constexpr index_t iWTildaRight = math::min(
            WTilda, math::integer_divide_ceil(InLeftPads{}[1] + Wi - 1, ConvStrides{}[1]) + 1);

        constexpr index_t HTildaSlice = iHTildaRight - iHTildaLeft;
        constexpr index_t WTildaSlice = iWTildaRight - iWTildaLeft;

        // GemmM and GemmN
        constexpr index_t GemmM = C;
        constexpr index_t GemmN = N * HTildaSlice * WTildaSlice;

        // GemmK is different for each GEMM
        index_t YDotSlice = (iYTilda + 1) * YDot <= Y ? YDot : Y % YDot;
        index_t XDotSlice = (iXTilda + 1) * XDot <= X ? XDot : X % XDot;

        index_t GemmK0 = YDotSlice;
        index_t GemmK1 = XDotSlice;
        index_t GemmK2 = K;

        return Array<index_t, 5>{GemmM, GemmN, GemmK0, GemmK1, GemmK2};
    }

    __host__ __device__ static constexpr auto GetGemmSize(index_t gemm_id)
    {
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationW = ConvDilations{}[1];

        constexpr index_t GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

        constexpr index_t XTilda = ConvStrideW / GcdStrideDilationW;

        index_t iYTilda = gemm_id / XTilda;
        index_t iXTilda = gemm_id % XTilda;

        return GetGemmSizeImpl(iYTilda, iXTilda);
    }

    template <index_t iYTilda, index_t iXTilda>
    __device__ static void RunImpl(Float* __restrict__ p_in_global,
                                   const Float* __restrict__ p_wei_global,
                                   const Float* __restrict__ p_out_global)
    {
        constexpr auto in_n_hi_wi_c_global_desc  = InGlobalDesc{};
        constexpr auto wei_k_y_x_c_global_desc   = WeiGlobalDesc{};
        constexpr auto out_n_ho_wo_k_global_desc = OutGlobalDesc{};

        constexpr index_t N  = in_n_hi_wi_c_global_desc.GetLengths()[0];
        constexpr index_t Hi = in_n_hi_wi_c_global_desc.GetLengths()[1];
        constexpr index_t Wi = in_n_hi_wi_c_global_desc.GetLengths()[2];
        constexpr index_t C  = in_n_hi_wi_c_global_desc.GetLengths()[3];

        constexpr index_t Ho = out_n_ho_wo_k_global_desc.GetLengths()[1];
        constexpr index_t Wo = out_n_ho_wo_k_global_desc.GetLengths()[2];
        constexpr index_t K  = out_n_ho_wo_k_global_desc.GetLengths()[3];

        constexpr index_t Y = wei_k_y_x_c_global_desc.GetLengths()[1];
        constexpr index_t X = wei_k_y_x_c_global_desc.GetLengths()[2];

        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];

        constexpr index_t GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
        constexpr index_t GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

        constexpr index_t YTilda = ConvStrideH / GcdStrideDilationH;
        constexpr index_t XTilda = ConvStrideW / GcdStrideDilationW;

        constexpr index_t YDot = math::integer_divide_ceil(Y, YTilda);
        constexpr index_t XDot = math::integer_divide_ceil(X, XTilda);

        constexpr index_t YDotSlice = (iYTilda + 1) * YDot <= Y ? YDot : Y % YDot;
        constexpr index_t XDotSlice = (iXTilda + 1) * XDot <= X ? XDot : X % XDot;

        constexpr index_t HTilda =
            Ho + math::integer_divide_ceil(ConvDilationH * (Y - 1), ConvStrideH);
        constexpr index_t WTilda =
            Wo + math::integer_divide_ceil(ConvDilationW * (X - 1), ConvStrideW);

        // only work on HTilda and WTilda that contribute to non-padding area of input tensor
        constexpr index_t iHTildaLeft = math::integer_divide_floor(
            math::max(0, InLeftPads{}[0] - ConvDilationH * (YTilda - 1)), ConvStrides{}[0]);
        constexpr index_t iWTildaLeft = math::integer_divide_floor(
            math::max(0, InLeftPads{}[1] - ConvDilationW * (XTilda - 1)), ConvStrides{}[1]);

        constexpr index_t iHTildaRight = math::min(
            HTilda, math::integer_divide_ceil(InLeftPads{}[0] + Hi - 1, ConvStrides{}[0]) + 1);
        constexpr index_t iWTildaRight = math::min(
            WTilda, math::integer_divide_ceil(InLeftPads{}[1] + Wi - 1, ConvStrides{}[1]) + 1);

        constexpr index_t HTildaSlice = iHTildaRight - iHTildaLeft;
        constexpr index_t WTildaSlice = iWTildaRight - iWTildaLeft;

        // A matrix: weight
        // weight out-of-bound check can be skipped
        constexpr bool wei_skip_out_of_bound_check = true;

        constexpr auto wei_k_ydot_ytilda_xdot_xtilda_c_global_desc = transform_tensor_descriptor(
            wei_k_y_x_c_global_desc,
            make_tuple(PassThrough<K>{},
                       Embed<Y,
                             Sequence<YDot, YTilda>,
                             Sequence<ConvStrideH / GcdStrideDilationH, 1, 0>,
                             wei_skip_out_of_bound_check>{},
                       Embed<X,
                             Sequence<XDot, XTilda>,
                             Sequence<ConvStrideW / GcdStrideDilationW, 1, 0>,
                             wei_skip_out_of_bound_check>{},
                       PassThrough<C>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

        constexpr auto wei_k_ydotslice_xdotslice_c_global_desc = transform_tensor_descriptor(
            wei_k_ydot_ytilda_xdot_xtilda_c_global_desc,
            make_tuple(
                PassThrough<K>{},
                Slice<Sequence<YDot, XDot>, Sequence<0, 0>, Sequence<YDotSlice, XDotSlice>>{},
                Freeze<Sequence<YTilda, XTilda>, Sequence<iYTilda, iXTilda>>{},
                PassThrough<C>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2, 4>{}, Sequence<5>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<>{}, Sequence<3>{}));

        constexpr auto wei_gemmk0_gemmk1_gemmk2_gemmm_global_desc =
            reorder_tensor_descriptor_given_lower2upper(wei_k_ydotslice_xdotslice_c_global_desc,
                                                        Sequence<2, 0, 1, 3>{});

// B matrix: output tensor
// TODO sometimes output tensor out-of-bound check can be skipped, find out all such
// situations
#if !CK_EXPERIMENTAL_IMPLICIT_GEMM_BACKWARD_DATA_V4R1_OUTPUT_SKIP_OUT_OF_BOUND_CHECK
        constexpr bool out_skip_out_of_bound_check = false;
#else
        constexpr bool out_skip_out_of_bound_check = true;
#endif

        constexpr auto out_n_ydot_htilda_xdot_wtilda_k_global_desc = transform_tensor_descriptor(
            out_n_ho_wo_k_global_desc,
            make_tuple(PassThrough<N>{},
                       Embed<Ho,
                             Sequence<YDot, HTilda>,
                             Sequence<-ConvDilationH / GcdStrideDilationH, 1, 0>,
                             out_skip_out_of_bound_check>{},
                       Embed<Wo,
                             Sequence<XDot, WTilda>,
                             Sequence<-ConvDilationW / GcdStrideDilationW, 1, 0>,
                             out_skip_out_of_bound_check>{},
                       PassThrough<K>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

        constexpr auto out_n_ydotslice_htildaslice_xdotslice_wtildaslice_k_global_desc =
            transform_tensor_descriptor(
                out_n_ydot_htilda_xdot_wtilda_k_global_desc,
                make_tuple(
                    PassThrough<N>{},
                    Slice<Sequence<YDot, XDot>, Sequence<0, 0>, Sequence<YDotSlice, XDotSlice>>{},
                    Slice<Sequence<HTilda, WTilda>,
                          Sequence<iHTildaLeft, iWTildaLeft>,
                          Sequence<iHTildaRight, iWTildaRight>>{},
                    PassThrough<K>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2, 4>{}, Sequence<5>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2, 4>{}, Sequence<5>{}));

        constexpr auto out_gemmk0_gemmk1_gemmk2_gemmn_global_desc = transform_tensor_descriptor(
            out_n_ydotslice_htildaslice_xdotslice_wtildaslice_k_global_desc,
            make_tuple(PassThrough<YDotSlice>{},
                       PassThrough<XDotSlice>{},
                       PassThrough<K>{},
                       Merge<Sequence<N, HTildaSlice, WTildaSlice>>{}),
            make_tuple(Sequence<1>{}, Sequence<3>{}, Sequence<5>{}, Sequence<0, 2, 4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

// C matrix: input tensor
// TODO sometimes input out-of-bound check can be skipped, find out all such situations
#if !CK_EXPERIMENTAL_IMPLICIT_GEMM_BACKWARD_DATA_V4R1_INPUT_SKIP_OUT_OF_BOUND_CHECK
        constexpr bool in_skip_out_of_bound_check = false;
#else
        constexpr bool in_skip_out_of_bound_check  = true;
#endif

        constexpr auto in_n_hip_wip_c_global_desc = transform_tensor_descriptor(
            in_n_hi_wi_c_global_desc,
            make_tuple(PassThrough<N>{},
                       Pad<Sequence<Hi, Wi>, InLeftPads, InRightPads, in_skip_out_of_bound_check>{},
                       PassThrough<C>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

        constexpr index_t Hip = in_n_hip_wip_c_global_desc.GetLengths()[1];
        constexpr index_t Wip = in_n_hip_wip_c_global_desc.GetLengths()[2];

        constexpr auto in_n_ytilda_htilda_xtilda_wtilda_c_global_desc = transform_tensor_descriptor(
            in_n_hip_wip_c_global_desc,
            make_tuple(PassThrough<N>{},
                       Embed<Hip,
                             Sequence<YTilda, HTilda>,
                             Sequence<ConvDilationH, ConvStrideH, 0>,
                             in_skip_out_of_bound_check>{},
                       Embed<Wip,
                             Sequence<XTilda, WTilda>,
                             Sequence<ConvDilationW, ConvStrideW, 0>,
                             in_skip_out_of_bound_check>{},
                       PassThrough<C>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

        constexpr auto in_n_htildaslice_wtildaslice_c_global_desc = transform_tensor_descriptor(
            in_n_ytilda_htilda_xtilda_wtilda_c_global_desc,
            make_tuple(PassThrough<N>{},
                       Freeze<Sequence<YTilda, XTilda>, Sequence<iYTilda, iXTilda>>{},
                       Slice<Sequence<HTilda, WTilda>,
                             Sequence<iHTildaLeft, iWTildaLeft>,
                             Sequence<iHTildaRight, iWTildaRight>>{},
                       PassThrough<C>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2, 4>{}, Sequence<5>{}),
            make_tuple(Sequence<0>{}, Sequence<>{}, Sequence<1, 2>{}, Sequence<3>{}));

        constexpr auto in_gemmm_gemmn_global_desc = transform_tensor_descriptor(
            in_n_htildaslice_wtildaslice_c_global_desc,
            make_tuple(PassThrough<C>{}, Merge<Sequence<N, HTildaSlice, WTildaSlice>>{}),
            make_tuple(Sequence<3>{}, Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        // call GEMM
        constexpr auto gridwise_gemm = GridwiseGemmTransposedANormalBNormalC_v2<
            GridSize,
            BlockSize,
            Float,
            AccFloat,
            decltype(wei_gemmk0_gemmk1_gemmk2_gemmm_global_desc),
            decltype(out_gemmk0_gemmk1_gemmk2_gemmn_global_desc),
            decltype(in_gemmm_gemmn_global_desc),
            InMemoryDataOperation::Set,
            GemmMPerBlock,
            GemmNPerBlock,
            GemmKPerBlock,
            GemmMPerThread,
            GemmNPerThread,
            GemmKPerThread,
            GemmMLevel0Cluster,
            GemmNLevel0Cluster,
            GemmMLevel1Cluster,
            GemmNLevel1Cluster,
            ThreadGemmDataPerRead_GemmM,
            ThreadGemmDataPerRead_GemmN,
            GemmABlockCopyThreadSliceLengths_GemmK0_GemmK1_GemmK2_GemmM,
            GemmABlockCopyThreadClusterLengths_GemmK0_GemmK1_GemmK2_GemmM,
            Sequence<0, 1, 2, 3>,
            Sequence<0, 1, 2, 3>,
            3,
            GemmABlockCopySrcDataPerRead_GemmM,
            GemmABlockCopyDstDataPerWrite_GemmM,
            GemmBBlockCopyThreadSliceLengths_GemmK0_GemmK1_GemmK2_GemmN,
            GemmBBlockCopyThreadClusterLengths_GemmK0_GemmK1_GemmK2_GemmN,
            Sequence<0, 1, 3, 2>,
            Sequence<0, 1, 3, 2>,
            2,
            GemmBBlockCopySrcDataPerRead_GemmK2,
            GemmBBlockCopyDstDataPerWrite_GemmN,
            Sequence<2, 3, 0, 1>,
            3,
            GemmCThreadCopyDstDataPerWrite_GemmN1>{};

        gridwise_gemm.Run(p_wei_global, p_out_global, p_in_global);
    }

    template <index_t GemmId>
    __device__ static void Run(Float* __restrict__ p_in_global,
                               const Float* __restrict__ p_wei_global,
                               const Float* __restrict__ p_out_global,
                               Number<GemmId>)
    {
        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];

        constexpr index_t GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
        constexpr index_t GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

        constexpr index_t YTilda = ConvStrideH / GcdStrideDilationH;
        constexpr index_t XTilda = ConvStrideW / GcdStrideDilationW;

        constexpr index_t iYTilda = GemmId / XTilda;
        constexpr index_t iXTilda = GemmId % XTilda;

        static_assert(iYTilda < YTilda && iXTilda < XTilda, "wrong! iYtilda, iXtilda");

        RunImpl<iYTilda, iXTilda>(p_in_global, p_wei_global, p_out_global);
    }
};

} // namespace ck
#endif
