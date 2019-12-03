#ifndef CK_GRIDWISE_CONVOLUTION_BACKWARD_DATA_IMPLICIT_GEMM_V1R1_NCHW_KCYX_NKHW_HPP
#define CK_GRIDWISE_CONVOLUTION_BACKWARD_DATA_IMPLICIT_GEMM_V1R1_NCHW_KCYX_NKHW_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm.hpp"

namespace ck {

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
          index_t EPerBlock,
          index_t BPerBlock,
          index_t KPerBlock,
          index_t GemmMPerThreadSubC,
          index_t GemmNPerThreadSubC,
          index_t GemmMLevel0Cluster,
          index_t GemmNLevel0Cluster,
          index_t GemmMLevel1Cluster,
          index_t GemmNLevel1Cluster,
          index_t GemmKPerThreadLoop,
          index_t GemmThreadGemmDataPerReadM,
          index_t GemmThreadGemmDataPerReadN,
          typename WeiBlockCopySubLengths_K_E,
          typename WeiBlockCopyClusterLengths_K_E,
          index_t WeiBlockCopyDataPerAccess_E,
          typename OutBlockCopySubLengths_K_B,
          typename OutBlockCopyClusterLengths_K_B,
          index_t OutBlockCopyDataPerAccess_B,
          index_t InThreadCopyDataPerAccess_B>
struct GridwiseConvolutionBackwardDataImplicitGemm_v1r1_nchw_kcyx_nkhw
{
    __device__ void Run(Float* __restrict__ p_in_global,
                        const Float* __restrict__ p_wei_global,
                        const Float* __restrict__ p_out_global) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr auto True = integral_constant<bool, true>{};

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

        constexpr index_t E = C * Y * X;
        constexpr index_t B = N * Ho * Wo;

        // sanity-check for vectorized memory load
        static_assert((Wo == 1 || (ConvStrideW == 1 || InThreadCopyDataPerAccess_B == 1)) &&
                          (X == 1 || ConvDilationW % InThreadCopyDataPerAccess_B == 0),
                      "wrong! aligment requirement for vectorized global load of input tensor will "
                      "be violated");

        // output tensor
        constexpr auto out_n_k_howo_global_desc =
            unfold_tensor_descriptor(out_n_k_ho_wo_global_desc, I2, I3);

        constexpr auto out_k_b_global_desc =
            transform_tensor_descriptor(out_n_k_howo_global_desc,
                                        make_tuple(PassThrough<K>{}, Merge<Sequence<N, Ho * Wo>>{}),
                                        make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

        // weight tensor
        constexpr auto wei_k_e_global_desc =
            unfold_tensor_descriptor(wei_k_c_y_x_global_desc, I1, I3);

        // input tensor
        constexpr auto in_n_c_hip_wip_global_desc = transform_tensor_descriptor(
            in_n_c_hi_wi_global_desc,
            make_tuple(
                PassThrough<N>{}, PassThrough<C>{}, Pad<Sequence<Hi, Wi>, LeftPads, RightPads>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}));

        constexpr auto in_n_c_y_ho_x_wo_global_desc = transform_tensor_descriptor(
            in_n_c_hip_wip_global_desc,
            make_tuple(PassThrough<N>{},
                       PassThrough<C>{},
                       Embed<Sequence<Y, Ho>, Sequence<ConvDilationH, ConvStrideH, 0>>{},
                       Embed<Sequence<X, Wo>, Sequence<ConvDilationW, ConvStrideW, 0>>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

        constexpr auto in_e_b_global_desc = transform_tensor_descriptor(
            in_n_c_y_ho_x_wo_global_desc,
            make_tuple(Merge<Sequence<C, Y, X>>{}, Merge<Sequence<N, Ho, Wo>>{}),
            make_tuple(Sequence<1, 2, 4>{}, Sequence<0, 3, 5>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        // GEMM: atomic add
        constexpr auto gridwise_gemm =
            GridwiseGemmTransposedANormalBNormalC_v1r1<GridSize,
                                                       BlockSize,
                                                       Float,
                                                       AccFloat,
                                                       decltype(wei_k_e_global_desc),
                                                       decltype(out_k_b_global_desc),
                                                       decltype(in_e_b_global_desc),
                                                       InMemoryDataOperation::atomic_add,
                                                       EPerBlock,
                                                       BPerBlock,
                                                       KPerBlock,
                                                       GemmMPerThreadSubC,
                                                       GemmNPerThreadSubC,
                                                       GemmMLevel0Cluster,
                                                       GemmNLevel0Cluster,
                                                       GemmMLevel1Cluster,
                                                       GemmNLevel1Cluster,
                                                       GemmKPerThreadLoop,
                                                       GemmThreadGemmDataPerReadM,
                                                       GemmThreadGemmDataPerReadN,
                                                       WeiBlockCopySubLengths_K_E,
                                                       WeiBlockCopyClusterLengths_K_E,
                                                       WeiBlockCopyDataPerAccess_E,
                                                       OutBlockCopySubLengths_K_B,
                                                       OutBlockCopyClusterLengths_K_B,
                                                       OutBlockCopyDataPerAccess_B,
                                                       InThreadCopyDataPerAccess_B>{};

        gridwise_gemm.Run(p_wei_global, p_out_global, p_in_global);
    }
};

} // namespace ck
#endif
