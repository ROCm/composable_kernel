#ifndef CK_GRIDWISE_CONVOLUTION_FORWARD_IMPLICIT_GEMM_V4R4_NHWC_KYXC_NHWK_HPP
#define CK_GRIDWISE_CONVOLUTION_FORWARD_IMPLICIT_GEMM_V4R4_NHWC_KYXC_NHWK_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm.hpp"

namespace ck {

// GemmM = K
// GemmN = N * Ho * Wo
// GemmK = C * Y * X
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
          typename GemmABlockCopyThreadSliceLengths_GemmK_GemmM,
          typename GemmABlockCopyThreadClusterLengths_GemmK_GemmM,
          index_t GemmABlockCopySrcDataPerRead_GemmK,
          index_t GemmABlockCopyDstDataPerWrite_GemmM,
          typename GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
          typename GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
          index_t GemmBBlockCopySrcDataPerRead_GemmK,
          index_t GemmBBlockCopyDstDataPerWrite_GemmN,
          index_t GemmCThreadCopyDstDataPerWrite_GemmM1>
struct GridwiseConvolutionForwardImplicitGemm_v4r4_nhwc_kyxc_nhwk
{
    __device__ void Run(const Float* const __restrict__ p_in_global,
                        const Float* const __restrict__ p_wei_global,
                        Float* const __restrict__ p_out_global) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr auto in_n_hi_wi_c_global_desc  = InGlobalDesc{};
        constexpr auto wei_k_y_x_c_global_desc   = WeiGlobalDesc{};
        constexpr auto out_n_ho_wo_k_global_desc = OutGlobalDesc{};

        constexpr index_t N  = in_n_hi_wi_c_global_desc.GetLengths()[I0];
        constexpr index_t Hi = in_n_hi_wi_c_global_desc.GetLengths()[I1];
        constexpr index_t Wi = in_n_hi_wi_c_global_desc.GetLengths()[I2];
        constexpr index_t C  = in_n_hi_wi_c_global_desc.GetLengths()[I3];

        constexpr index_t K  = out_n_ho_wo_k_global_desc.GetLengths()[I3];
        constexpr index_t Ho = out_n_ho_wo_k_global_desc.GetLengths()[I1];
        constexpr index_t Wo = out_n_ho_wo_k_global_desc.GetLengths()[I2];

        constexpr index_t Y = wei_k_y_x_c_global_desc.GetLengths()[I1];
        constexpr index_t X = wei_k_y_x_c_global_desc.GetLengths()[I2];

        constexpr index_t ConvStrideH = ConvStrides{}[I0];
        constexpr index_t ConvStrideW = ConvStrides{}[I1];

        constexpr index_t ConvDilationH = ConvDilations{}[I0];
        constexpr index_t ConvDilationW = ConvDilations{}[I1];

        // weight tensor
        constexpr auto wei_gemmk_gemmm_global_desc = reorder_tensor_descriptor_given_upper2lower(
            unfold_tensor_descriptor(wei_k_y_x_c_global_desc, I1, I3), Sequence<1, 0>{});

        // input tensor
        constexpr auto in_n_hip_wip_c_global_desc =
            transform_tensor_descriptor(in_n_hi_wi_c_global_desc,
                                        make_tuple(PassThrough<N>{},
                                                   Pad<Sequence<Hi, Wi>, InLeftPads, InRightPads>{},
                                                   PassThrough<C>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

        constexpr index_t Hip = in_n_hip_wip_c_global_desc.GetLengths()[I1];
        constexpr index_t Wip = in_n_hip_wip_c_global_desc.GetLengths()[I2];

        constexpr auto in_n_y_ho_x_wo_c_global_desc = transform_tensor_descriptor(
            in_n_hip_wip_c_global_desc,
            make_tuple(PassThrough<N>{},
                       Embed<Hip, Sequence<Y, Ho>, Sequence<ConvDilationH, ConvStrideH, 0>>{},
                       Embed<Wip, Sequence<X, Wo>, Sequence<ConvDilationW, ConvStrideW, 0>>{},
                       PassThrough<C>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

        constexpr auto in_gemmk_gemmn_global_desc = transform_tensor_descriptor(
            in_n_y_ho_x_wo_c_global_desc,
            make_tuple(Merge<Sequence<Y, X, C>>{}, Merge<Sequence<N, Ho, Wo>>{}),
            make_tuple(Sequence<1, 3, 5>{}, Sequence<0, 2, 4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        // output tensor
        constexpr auto out_gemmm_gemmn_global_desc = transform_tensor_descriptor(
            unfold_tensor_descriptor(out_n_ho_wo_k_global_desc, I0, I2),
            make_tuple(PassThrough<K>{}, Merge<Sequence<N * Ho * Wo>>{}),
            make_tuple(Sequence<1>{}, Sequence<0>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        // GEMM
        constexpr auto gridwise_gemm =
            GridwiseGemmTransposedANormalBNormalC_v1<GridSize,
                                                     BlockSize,
                                                     Float,
                                                     AccFloat,
                                                     decltype(wei_gemmk_gemmm_global_desc),
                                                     decltype(in_gemmk_gemmn_global_desc),
                                                     decltype(out_gemmm_gemmn_global_desc),
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
                                                     GemmABlockCopyThreadSliceLengths_GemmK_GemmM,
                                                     GemmABlockCopyThreadClusterLengths_GemmK_GemmM,
                                                     Sequence<1, 0>,
                                                     Sequence<1, 0>,
                                                     0,
                                                     GemmABlockCopySrcDataPerRead_GemmK,
                                                     GemmABlockCopyDstDataPerWrite_GemmM,
                                                     GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
                                                     GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
                                                     Sequence<1, 0>,
                                                     Sequence<1, 0>,
                                                     0,
                                                     GemmBBlockCopySrcDataPerRead_GemmK,
                                                     GemmBBlockCopyDstDataPerWrite_GemmN,
                                                     Sequence<2, 3, 0, 1>,
                                                     1,
                                                     GemmCThreadCopyDstDataPerWrite_GemmM1>{};

        gridwise_gemm.Run(p_wei_global, p_in_global, p_out_global);
    }
};

} // namespace ck
#endif
