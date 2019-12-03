#ifndef CK_GRIDWISE_COL2IM_EB_NCHW_HPP
#define CK_GRIDWISE_COL2IM_EB_NCHW_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "blockwise_generic_tensor_slice_copy.hpp"

namespace ck {

// B = merge(N, Ho, Wo)
template <index_t GridSize,
          index_t BlockSize,
          typename Float,
          typename ColGlobalDesc,
          typename ImgGlobalDesc,
          typename FilterSizes,
          typename OutputSizes,
          typename ConvStrides,
          typename ConvDilations,
          typename LeftPads,
          typename RightPads,
          index_t EPerBlock,
          index_t BPerBlock,
          typename BlockCopySubLengths_E_B,
          typename BlockCopyClusterLengths_E_B,
          typename BlockCopyThreadClusterArrangeOrder,
          typename BlockCopySrcAccessOrder,
          typename BlockCopyDstAccessOrder,
          index_t BlockCopyDataPerAccess_B>
struct GridwiseCol2Im_eb_nchw
{
    __device__ void Run(const Float* const __restrict__ p_col_global,
                        Float* const __restrict__ p_img_global) const
    {
        constexpr auto col_e_b_global_desc       = ColGlobalDesc{};
        constexpr auto img_n_c_hi_wi_global_desc = ImgGlobalDesc{};

        constexpr index_t N  = img_n_c_hi_wi_global_desc.GetLengths()[0];
        constexpr index_t C  = img_n_c_hi_wi_global_desc.GetLengths()[1];
        constexpr index_t Hi = img_n_c_hi_wi_global_desc.GetLengths()[2];
        constexpr index_t Wi = img_n_c_hi_wi_global_desc.GetLengths()[3];

        constexpr index_t Ho = OutputSizes{}[0];
        constexpr index_t Wo = OutputSizes{}[1];

        constexpr index_t Y = FilterSizes{}[0];
        constexpr index_t X = FilterSizes{}[1];

        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];

        constexpr index_t E = C * Y * X;
        constexpr index_t B = N * Ho * Wo;

        // sanity-check for vectorized memory load
        static_assert((Wo == 1 || (ConvStrideW == 1 || BlockCopyDataPerAccess_B == 1)) &&
                          (X == 1 || ConvDilationW % BlockCopyDataPerAccess_B == 0),
                      "wrong! aligment requirement for vectorized global load of input tensor will "
                      "be violated");

        // divide block work by [E, B]
        static_assert(E % EPerBlock == 0 && B % BPerBlock == 0,
                      "wrong! cannot divide work evenly among block");

        constexpr index_t EBlockWork = E / EPerBlock;
        constexpr index_t BBlockWork = B / BPerBlock;

        constexpr auto block_work_desc =
            make_cluster_descriptor(Sequence<EBlockWork, BBlockWork>{});

        const auto block_work_id = block_work_desc.CalculateClusterIndex(get_block_1d_id());

        const index_t e_block_data_on_global = block_work_id[0] * EPerBlock;
        const index_t b_block_data_on_global = block_work_id[1] * BPerBlock;

        // construct img_eb_global_desc
        constexpr auto img_n_c_hip_wip_global_desc = transform_tensor_descriptor(
            img_n_c_hi_wi_global_desc,
            make_tuple(
                PassThrough<N>{}, PassThrough<C>{}, Pad<Sequence<Hi, Wi>, LeftPads, RightPads>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}));

        constexpr auto img_n_c_y_ho_x_wo_global_desc = transform_tensor_descriptor(
            img_n_c_hip_wip_global_desc,
            make_tuple(PassThrough<N>{},
                       PassThrough<C>{},
                       Embed<Sequence<Y, Ho>, Sequence<ConvDilationH, ConvStrideH, 0>>{},
                       Embed<Sequence<X, Wo>, Sequence<ConvDilationW, ConvStrideW, 0>>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

        constexpr auto img_e_b_global_desc = transform_tensor_descriptor(
            img_n_c_y_ho_x_wo_global_desc,
            make_tuple(Merge<Sequence<C, Y, X>>{}, Merge<Sequence<N, Ho, Wo>>{}),
            make_tuple(Sequence<1, 2, 4>{}, Sequence<0, 3, 5>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        // blockwise atomic accumulation
        auto blockwise_copy = BlockwiseGenericTensorSliceCopy_v4<BlockSize,
                                                                 decltype(col_e_b_global_desc),
                                                                 decltype(img_e_b_global_desc),
                                                                 Sequence<EPerBlock, BPerBlock>,
                                                                 BlockCopySubLengths_E_B,
                                                                 BlockCopyClusterLengths_E_B,
                                                                 BlockCopyThreadClusterArrangeOrder,
                                                                 BlockCopySrcAccessOrder,
                                                                 BlockCopyDstAccessOrder,
                                                                 1,
                                                                 1,
                                                                 BlockCopyDataPerAccess_B,
                                                                 BlockCopyDataPerAccess_B,
                                                                 AddressSpace::vgpr,
                                                                 AddressSpace::vgpr,
                                                                 AddressSpace::global,
                                                                 InMemoryDataOperation::atomic_add>(
            {e_block_data_on_global, b_block_data_on_global},
            {e_block_data_on_global, b_block_data_on_global});

        // blockwise copy
        blockwise_copy.Run(p_col_global, p_img_global);
    }
};

} // namespace ck
#endif
