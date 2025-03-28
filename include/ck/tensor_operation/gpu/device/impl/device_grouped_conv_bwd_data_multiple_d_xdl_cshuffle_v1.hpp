// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/library/utility/numeric.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_data_multiple_d.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_data_specialization.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_bwd_data_to_gemm_v1.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_ngchw_to_nhwgc.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_elementwise_2d.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/io.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

namespace {

/*
 * \brief Wrapper function of GridwiseGemm::Run to realize BatchedGEMM.
 *
 * \tparam ComputePtrOffsetOfBatch Class that computes the base pointer offsets of A, B, C matrix
 * given the batch. For example, ComputePtrOffsetOfStridedBatch() computes the offsets of evenly
 * strided batched, but we can easily extend to other layouts. The returned offset can be either \p
 * index_t or \p long_index_t. If it returns \p long_index_t, we are not subject to the 2GB
 * limitations.
 *
 * \tparam Block2ETileMap Block2ETileMap::CalculateBottomIndex() takes in id of a workgroup and
 * returns the 2D index of the tile that it computes. \see
 * GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3::Run().
 *
 * \note Using \p ComputePtrOffsetOfBatch gives us the flexibility that 2 workgroups can compute 2
 * tiles from different matrices. Keep in mind that these 2 matrices can share the same grid
 * descriptor (like in BatchedGEMM), or use their own grid descriptors (in GroupedGemm). \link
 * device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk.hpp kernel_gemm_xdlops_v2r3_for_conv3d \endlink for \link
 * DeviceConv3d \endlink uses the same concept, but currently does NOT encapsulate the computing of
 * pointer offset into \p ComputePtrOffsetOfStridedBatch.
 *
 * \note \p Block2ETileMap allows customized mapping between a workgroup and the C-tile it computes.
 * Together with \p ComputePtrOffsetOfBatch, we can reuse GridwiseGemm (and GridwiseGemm fusion ) to
 * realize BatchedGemm and GroupedGemm (and the corresponding GEMM fusion).
 *
 */
template <typename GridwiseGemm,
          typename ABDataType,
          typename DsPointer,
          typename EDataType,
          typename AElementwiseOp,
          typename BElementwiseOp,
          typename CDEElementwiseOp,
          typename AGridDesc_AK0_M_AK1,
          typename BGridDesc_BK0_N_BK1,
          typename DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
          typename Block2ETileMap,
          typename ComputePtrOffsetOfBatch,
          typename ComputePtrOffsetOfN,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_grouped_conv_bwd_data_multiple_d_xdl_cshuffle(
            const ABDataType* __restrict__ p_a_grid,
            const ABDataType* __restrict__ p_b_grid,
            DsPointer p_ds_grid,
            EDataType* __restrict__ p_e_grid,
            const AElementwiseOp a_element_op,
            const BElementwiseOp b_element_op,
            const CDEElementwiseOp cde_element_op,
            const AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1,
            const BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1,
            const DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                ds_grid_desc_mblock_mperblock_nblock_nperblock,
            const EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
                e_grid_desc_mblock_mperblock_nblock_nperblock_,
            const Block2ETileMap block_2_ctile_map,
            const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch,
            const ComputePtrOffsetOfN compute_ptr_offset_of_n)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx9__))
    // offset base pointer for each work-group
    const index_t n_idx = __builtin_amdgcn_readfirstlane(blockIdx.z);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);

    const long_index_t a_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx));
    const long_index_t b_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx));
    const long_index_t e_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetEPtrOffset(g_idx));

    const auto ds_batch_offset = compute_ptr_offset_of_batch.GetDsPtrOffset(g_idx);

    const long_index_t a_n_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_n.GetAPtrOffset(n_idx));
    const long_index_t e_n_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_n.GetEPtrOffset(n_idx));

    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    DsPointer p_ds_grid_grp;

    static constexpr index_t NumDTensor =
        DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock::Size();

    static_for<0, NumDTensor, 1>{}(
        [&](auto i) { p_ds_grid_grp(i) = p_ds_grid[i] + ds_batch_offset[i]; });

    GridwiseGemm::template Run<HasMainKBlockLoop>(p_a_grid + a_batch_offset + a_n_offset,
                                                  p_b_grid + b_batch_offset,
                                                  p_ds_grid_grp,
                                                  p_e_grid + e_batch_offset + e_n_offset,
                                                  p_shared,
                                                  a_element_op,
                                                  b_element_op,
                                                  cde_element_op,
                                                  a_grid_desc_ak0_m_ak1,
                                                  b_grid_desc_bk0_n_bk1,
                                                  ds_grid_desc_mblock_mperblock_nblock_nperblock,
                                                  e_grid_desc_mblock_mperblock_nblock_nperblock_,
                                                  block_2_ctile_map);
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_ds_grid;
    ignore = p_e_grid;
    ignore = a_grid_desc_ak0_m_ak1;
    ignore = b_grid_desc_bk0_n_bk1;
    ignore = ds_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = e_grid_desc_mblock_mperblock_nblock_nperblock_;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = cde_element_op;
    ignore = compute_ptr_offset_of_batch;
    ignore = compute_ptr_offset_of_n;
    ignore = block_2_ctile_map;
#endif
}

} // namespace

// Conv backward data multiple D:
//   input : output image A: [G, N, K, Ho, Wo]
//   input : weight B: [G, K, C, Y, X],
//   input : D0, D1, ... : [G, N, K, Ho, Wo]
//   output : input image E: [G, N, C, Hi, Wi]
//   C = a_op(A) * b_op(B)
//   E = cde_op(C, D0, D1, ...)
template <index_t NDimSpatial,
          typename ALayout,   // output image
          typename BLayout,   // weight
          typename DsLayout,  // bias
          typename ELayout,   // input image
          typename ADataType, // output image
          typename BDataType, // weight
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,       // bias
          typename EDataType,        // input image
          typename AElementwiseOp,   // output image
          typename BElementwiseOp,   // weight
          typename CDEElementwiseOp, // C, bias, and input image
          ConvolutionBackwardDataSpecialization ConvBackwardDataSpecialization,
          bool DoPadGemmM,
          bool DoPadGemmN,
          index_t NumGemmKPrefetchStage,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          index_t BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEBlockTransferScalarPerVector_NPerBlock,
          LoopScheduler LoopSched                        = make_default_loop_scheduler(),
          typename AComputeType                          = ADataType,
          typename BComputeType                          = AComputeType,
          index_t MaxTransposeTransferInScalarPerVector  = 1,
          index_t MaxTransposeTransferOutScalarPerVector = 1>
struct DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1
    : public DeviceGroupedConvBwdDataMultipleD<NDimSpatial,
                                               ALayout,    // output image
                                               BLayout,    // weight
                                               DsLayout,   // bias
                                               ELayout,    // input image
                                               ADataType,  // output image
                                               BDataType,  // weight
                                               DsDataType, // bias
                                               EDataType,  // input image
                                               AElementwiseOp,
                                               BElementwiseOp,
                                               CDEElementwiseOp,
                                               AComputeType,
                                               BComputeType>
{
    // TODO: Extend support for more spatial dimensions.
    static_assert(NDimSpatial == 2 || NDimSpatial == 3,
                  "wrong! only implemented for 2D and 3D now");

    using DeviceOp = DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1;

    static constexpr index_t NumDTensor = DsDataType::Size();

    // TODO: Add support for different A and B data types.
    using ABDataType = ADataType;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    using ALayoutAfterTranspose =
        std::conditional_t<is_NGCHW_GKYXC_NGKHW<ELayout, BLayout, ALayout>(),
                           tensor_layout::convolution::NHWGK,
                           std::conditional_t<is_NGCDHW_GKZYXC_NGKDHW<ELayout, BLayout, ALayout>(),
                                              tensor_layout::convolution::NDHWGK,
                                              ALayout>>;
    using ELayoutAfterTranspose =
        std::conditional_t<is_NGCHW_GKYXC_NGKHW<ELayout, BLayout, ALayout>(),
                           tensor_layout::convolution::NHWGC,
                           std::conditional_t<is_NGCDHW_GKZYXC_NGKDHW<ELayout, BLayout, ALayout>(),
                                              tensor_layout::convolution::NDHWGC,
                                              ELayout>>;

    using ConvToGemmBwdDataTransform = TransformConvBwdDataToGemm_v1<NDimSpatial,
                                                                     ConvBackwardDataSpecialization,
                                                                     AK1,
                                                                     BK1,
                                                                     MPerBlock,
                                                                     NPerBlock,
                                                                     KPerBlock,
                                                                     DoPadGemmM,
                                                                     DoPadGemmN,
                                                                     ALayoutAfterTranspose,
                                                                     BLayout,
                                                                     ELayoutAfterTranspose,
                                                                     true, /*SplitConvN*/
                                                                     ABDataType,
                                                                     EDataType>;

    static auto
    GetDummyABDsEGridDescriptor(const ConvToGemmBwdDataTransform& conv_to_gemm_transform)
    {
        const auto a_grid_desc_ak0_m_ak1 = conv_to_gemm_transform.MakeADescriptor_AK0_M_AK1();

        const auto b_grid_desc_bk0_n_bk1 = conv_to_gemm_transform.MakeBDescriptor_BK0_N_BK1();

        const auto ds_grid_desc_m_n = generate_tuple(
            [&](auto i) {
                using DLayout   = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
                using ConvToGemmBwdDataTransformD =
                    TransformConvBwdDataToGemm_v1<NDimSpatial,
                                                  ConvBackwardDataSpecialization,
                                                  AK1,
                                                  BK1,
                                                  MPerBlock,
                                                  NPerBlock,
                                                  KPerBlock,
                                                  DoPadGemmM,
                                                  DoPadGemmN,
                                                  ALayoutAfterTranspose,
                                                  BLayout,
                                                  DLayout,
                                                  true, /*SplitConvN*/
                                                  ABDataType,
                                                  DDataType>;
                return ConvToGemmBwdDataTransformD{}.MakeCDescriptor_M_N();
            },
            Number<NumDTensor>{});

        const auto e_grid_desc_m_n = conv_to_gemm_transform.MakeCDescriptor_M_N();

        return make_tuple(
            a_grid_desc_ak0_m_ak1, b_grid_desc_bk0_n_bk1, ds_grid_desc_m_n, e_grid_desc_m_n);
    }

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemmMultipleD_xdl_cshuffle<
        ABDataType,
        ABDataType,
        AComputeType,
        AccDataType,
        CShuffleDataType,
        DsDataType,
        EDataType,
        AElementwiseOp,
        BElementwiseOp,
        CDEElementwiseOp,
        InMemoryDataOperationEnum::Set,
        NumGemmKPrefetchStage,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        MPerXDL,
        NPerXDL,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        false,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEBlockTransferScalarPerVector_NPerBlock,
        LoopSched,
        PipelineVersion::v1,
        BComputeType>;

    template <typename Desc_K0_M_K1>
    static auto transform_k0_m_k1_to_m_k(const Desc_K0_M_K1& desc_k0_m_k1)
    {
        const auto grid_desc_m_k = transform_tensor_descriptor(
            desc_k0_m_k1,
            make_tuple(make_pass_through_transform(desc_k0_m_k1.GetLength(I1)),
                       make_merge_transform(
                           make_tuple(desc_k0_m_k1.GetLength(I0), desc_k0_m_k1.GetLength(I2)))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return grid_desc_m_k;
    }

    // desc
    constexpr static ConvToGemmBwdDataTransform dummy_conv_to_gemm_transform;
    using ABDsEGridDesc = decltype(GetDummyABDsEGridDescriptor(dummy_conv_to_gemm_transform));

    using AGridDesc_AK0_M_AK1 = remove_cvref_t<tuple_element_t<0, ABDsEGridDesc>>;
    using BGridDesc_BK0_N_BK1 = remove_cvref_t<tuple_element_t<1, ABDsEGridDesc>>;
    using DsGridDesc_M_N      = remove_cvref_t<tuple_element_t<2, ABDsEGridDesc>>;
    using EGridDesc_M_N       = remove_cvref_t<tuple_element_t<3, ABDsEGridDesc>>;

    using AGridDesc_M_K = decltype(transform_k0_m_k1_to_m_k(AGridDesc_AK0_M_AK1{}));
    using BGridDesc_N_K = decltype(transform_k0_m_k1_to_m_k(BGridDesc_BK0_N_BK1{}));

    using DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock =
        decltype(GridwiseGemm::MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            DsGridDesc_M_N{}));
    using EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock =
        decltype(GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            EGridDesc_M_N{}));

    // block-to-e-tile map
    using Block2ETileMap =
        remove_cvref_t<decltype(GridwiseGemm::MakeDefaultBlock2ETileMap(EGridDesc_M_N{}))>;
    using Block2TileMapElementwise = BlockToCTileMap_M00_N0_M01Adapt<NPerBlock, MPerBlock>;

    static constexpr index_t ClusterLengthMPerBlock =
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::At(1);
    static constexpr index_t ClusterLengthNPerBlock =
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::At(3);

    static constexpr auto conv_ngchw_to_nhwgc_transformer =
        TransformConvNGCHWToNHWGC<ELayout,
                                  BLayout,
                                  ALayout,
                                  NDimSpatial,
                                  NPerBlock / ClusterLengthNPerBlock,
                                  MPerBlock / ClusterLengthMPerBlock>{};

    static constexpr index_t TransposeTransferInScalarPerVectorAligned =
        std::min(MPerBlock / ClusterLengthMPerBlock, MaxTransposeTransferInScalarPerVector);
    static constexpr index_t TransposeTransferOutScalarPerVectorAligned =
        std::min(MPerBlock / ClusterLengthMPerBlock, MaxTransposeTransferOutScalarPerVector);

    using NGCHWTransposeDescType =
        remove_cvref_t<decltype(conv_ngchw_to_nhwgc_transformer
                                    .template MakeNGCHWTransposeDesc<NDimSpatial>({}, {}))>;
    using NHWGCTransposeDescType =
        remove_cvref_t<decltype(conv_ngchw_to_nhwgc_transformer
                                    .template MakeNHWGCTransposeDesc<NDimSpatial>({}, {}))>;

    static constexpr index_t ElementwiseBlocksize = ClusterLengthMPerBlock * ClusterLengthNPerBlock;

    using GridwiseElementwiseInputTranspose =
        GridwiseElementwise<Tuple<NGCHWTransposeDescType>,
                            Tuple<NHWGCTransposeDescType>,
                            Tuple<const ADataType*>,
                            Tuple<ADataType*>,
                            Block2TileMapElementwise,
                            element_wise::PassThrough,
                            ElementwiseBlocksize,
                            NPerBlock,
                            MPerBlock,
                            NPerBlock / ClusterLengthNPerBlock,
                            MPerBlock / ClusterLengthMPerBlock,
                            Sequence<1, 0>,
                            Sequence<TransposeTransferInScalarPerVectorAligned>,
                            Sequence<CDEBlockTransferScalarPerVector_NPerBlock>,
                            I1,
                            I0>;

    using GridwiseElementwiseOutputTranspose =
        GridwiseElementwise<Tuple<NHWGCTransposeDescType>,
                            Tuple<NGCHWTransposeDescType>,
                            Tuple<const EDataType*>,
                            Tuple<EDataType*>,
                            Block2TileMapElementwise,
                            element_wise::PassThrough,
                            ElementwiseBlocksize,
                            NPerBlock,
                            MPerBlock,
                            NPerBlock / ClusterLengthNPerBlock,
                            MPerBlock / ClusterLengthMPerBlock,
                            Sequence<1, 0>,
                            Sequence<CDEBlockTransferScalarPerVector_NPerBlock>,
                            Sequence<TransposeTransferOutScalarPerVectorAligned>,
                            I0,
                            I1>;
    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const void* p_a,                                 // output image
                 const void* p_b,                                 // weight
                 const std::array<const void*, NumDTensor>& p_ds, // bias
                 void* p_e,                                       // input image
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_c_wis_lengths,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_c_wis_lengths,
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<index_t, NDimSpatial>& input_left_pads,
                 const std::array<index_t, NDimSpatial>& input_right_pads,
                 const AElementwiseOp& a_element_op,
                 const BElementwiseOp& b_element_op,
                 const CDEElementwiseOp& cde_element_op)
            : p_a_grid_{static_cast<const ADataType*>(p_a)},
              p_b_grid_{static_cast<const BDataType*>(p_b)},
              p_ds_grid_{},
              p_e_grid_{static_cast<EDataType*>(p_e)},
              num_group_{a_g_n_k_wos_lengths[0]},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op},
              a_g_n_k_wos_lengths_{a_g_n_k_wos_lengths},
              b_g_k_c_xs_lengths_{b_g_k_c_xs_lengths},
              e_g_n_c_wis_lengths_{e_g_n_c_wis_lengths},
              conv_filter_strides_{conv_filter_strides},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads}
        {
            std::array<index_t, NDimSpatial + 3> a_g_n_k_wos_strides_transposed =
                conv_ngchw_to_nhwgc_transformer.TransposeInOutStrides(a_g_n_k_wos_lengths,
                                                                      a_g_n_k_wos_strides);
            std::array<index_t, NDimSpatial + 3> e_g_n_c_wis_strides_transposed =
                conv_ngchw_to_nhwgc_transformer.TransposeInOutStrides(e_g_n_c_wis_lengths,
                                                                      e_g_n_c_wis_strides);

            // populate Ds pointer
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                p_ds_grid_(i) = static_cast<const DDataType*>(p_ds[i]);
            });

            static_for<0, NumDTensor, 1>{}([&](auto i) {
                compute_ptr_offset_of_batch_.BatchStrideDs_(i) = ds_g_n_c_wis_strides[i][0];
            });

            static constexpr auto NonSpatialDimsNum = Number<3>{};

            static constexpr auto DIdx = Number<NonSpatialDimsNum>{};
            static constexpr auto HIdx =
                NDimSpatial == 2 ? Number<NonSpatialDimsNum>{} : Number<NonSpatialDimsNum + 1>{};
            static constexpr auto WIdx = NDimSpatial == 2 ? Number<NonSpatialDimsNum + 1>{}
                                                          : Number<NonSpatialDimsNum + 2>{};

            static constexpr auto ZIdx = Number<NonSpatialDimsNum>{};
            static constexpr auto YIdx =
                NDimSpatial == 2 ? Number<NonSpatialDimsNum>{} : Number<NonSpatialDimsNum + 1>{};
            static constexpr auto XIdx = NDimSpatial == 2 ? Number<NonSpatialDimsNum + 1>{}
                                                          : Number<NonSpatialDimsNum + 2>{};

            // problem definition
            const index_t Z = b_g_k_c_xs_lengths[ZIdx];
            const index_t Y = b_g_k_c_xs_lengths[YIdx];
            const index_t X = b_g_k_c_xs_lengths[XIdx];

            const index_t ConvStrideD = conv_filter_strides[DIdx - NonSpatialDimsNum];
            const index_t ConvStrideH = conv_filter_strides[HIdx - NonSpatialDimsNum];
            const index_t ConvStrideW = conv_filter_strides[WIdx - NonSpatialDimsNum];

            const index_t ConvDilationD = conv_filter_dilations[DIdx - NonSpatialDimsNum];
            const index_t ConvDilationH = conv_filter_dilations[HIdx - NonSpatialDimsNum];
            const index_t ConvDilationW = conv_filter_dilations[WIdx - NonSpatialDimsNum];

            const auto GcdStrideDilationD = math::gcd(ConvStrideD, ConvDilationD);
            const auto GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
            const auto GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

            const auto ZTilde = NDimSpatial == 3 ? ConvStrideD / GcdStrideDilationD : 1;
            const auto YTilde = ConvStrideH / GcdStrideDilationH;
            const auto XTilde = ConvStrideW / GcdStrideDilationW;

            for(index_t i_ztilde = 0; i_ztilde < ZTilde; ++i_ztilde)
            {

                for(index_t i_ytilde = 0; i_ytilde < YTilde; ++i_ytilde)
                {
                    for(index_t i_xtilde = 0; i_xtilde < XTilde; ++i_xtilde)
                    {
                        // check slice is valid
                        const auto ZDotSlice =
                            NDimSpatial == 3 ? math::integer_divide_ceil(Z - i_ztilde, ZTilde) : 1;
                        const auto YDotSlice = math::integer_divide_ceil(Y - i_ytilde, YTilde);
                        const auto XDotSlice = math::integer_divide_ceil(X - i_xtilde, XTilde);

                        if(YDotSlice * XDotSlice * ZDotSlice <= 0)
                        {
                            continue;
                        }

                        std::array<index_t, NDimSpatial> tildes;
                        if constexpr(NDimSpatial == 2)
                        {
                            tildes = {i_ytilde, i_xtilde};
                        }
                        else if constexpr(NDimSpatial == 3)
                        {
                            tildes = {i_ztilde, i_ytilde, i_xtilde};
                        }
                        else
                        {
                            throw std::runtime_error("wrong! only implemented for 2D and 3D now");
                        }

                        ConvToGemmBwdDataTransform conv_to_gemm_transform_{
                            a_g_n_k_wos_lengths,
                            a_g_n_k_wos_strides_transposed,
                            b_g_k_c_xs_lengths,
                            b_g_k_c_xs_strides,
                            e_g_n_c_wis_lengths,
                            e_g_n_c_wis_strides_transposed,
                            conv_filter_strides,
                            conv_filter_dilations,
                            input_left_pads,
                            input_right_pads,
                            tildes};

                        conv_N_per_block_ = conv_to_gemm_transform_.N_;

                        const auto a_grid_desc_ak0_m_ak1 =
                            conv_to_gemm_transform_.MakeADescriptor_AK0_M_AK1();

                        const auto b_grid_desc_bk0_n_bk1 =
                            conv_to_gemm_transform_.MakeBDescriptor_BK0_N_BK1();

                        DsGridDesc_M_N ds_grid_desc_m_n;

                        // populate Ds desc
                        static_for<0, NumDTensor, 1>{}([&](auto i) {
                            using DLayout   = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;
                            using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
                            using ConvToGemmBwdDataTransformD =
                                TransformConvBwdDataToGemm_v1<NDimSpatial,
                                                              ConvBackwardDataSpecialization,
                                                              AK1,
                                                              BK1,
                                                              MPerBlock,
                                                              NPerBlock,
                                                              KPerBlock,
                                                              DoPadGemmM,
                                                              DoPadGemmN,
                                                              ALayoutAfterTranspose,
                                                              BLayout,
                                                              DLayout,
                                                              true, /*SplitConvN*/
                                                              ABDataType,
                                                              DDataType>;
                            ConvToGemmBwdDataTransformD conv_to_gemm_transform_d{
                                a_g_n_k_wos_lengths,
                                a_g_n_k_wos_strides_transposed,
                                b_g_k_c_xs_lengths,
                                b_g_k_c_xs_strides,
                                ds_g_n_c_wis_lengths[i],
                                ds_g_n_c_wis_strides[i],
                                conv_filter_strides,
                                conv_filter_dilations,
                                input_left_pads,
                                input_right_pads,
                                tildes};

                            ds_grid_desc_m_n(i) = conv_to_gemm_transform_d.MakeCDescriptor_M_N();
                        });

                        const auto e_grid_desc_m_n = conv_to_gemm_transform_.MakeCDescriptor_M_N();

                        // desc for problem definition
                        const auto a_grid_desc_m_k =
                            transform_k0_m_k1_to_m_k(a_grid_desc_ak0_m_ak1);
                        const auto b_grid_desc_n_k =
                            transform_k0_m_k1_to_m_k(b_grid_desc_bk0_n_bk1);

                        a_grid_desc_m_k_container_.push_back(a_grid_desc_m_k);
                        b_grid_desc_n_k_container_.push_back(b_grid_desc_n_k);
                        ds_grid_desc_m_n_container_.push_back(ds_grid_desc_m_n);
                        e_grid_desc_m_n_container_.push_back(e_grid_desc_m_n);

                        // desc for blockwise copy
                        a_grid_desc_ak0_m_ak1_container_.push_back(a_grid_desc_ak0_m_ak1);
                        b_grid_desc_bk0_n_bk1_container_.push_back(b_grid_desc_bk0_n_bk1);

                        // block-to-e-tile-map
                        auto block_2_etile_map =
                            GridwiseGemm::MakeDefaultBlock2ETileMap(e_grid_desc_m_n);

                        block_2_etile_map_container_.push_back(block_2_etile_map);

                        if(GridwiseGemm::CheckValidity(a_grid_desc_m_k,
                                                       b_grid_desc_n_k,
                                                       ds_grid_desc_m_n,
                                                       e_grid_desc_m_n,
                                                       block_2_etile_map))
                        {
                            ds_grid_desc_mblock_mperblock_nblock_nperblock_container_.push_back(
                                GridwiseGemm::
                                    MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                                        ds_grid_desc_m_n));

                            e_grid_desc_mblock_mperblock_nblock_nperblock_container_.push_back(
                                GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                                    e_grid_desc_m_n));
                        }
                    }
                }
            }
            // A/B/Ds/E Batch Stride
            compute_ptr_offset_of_batch_.BatchStrideA_ = a_g_n_k_wos_strides_transposed[0];
            compute_ptr_offset_of_batch_.BatchStrideB_ = b_g_k_c_xs_strides[0];
            compute_ptr_offset_of_batch_.BatchStrideE_ = e_g_n_c_wis_strides_transposed[0];

            compute_ptr_offset_of_n_.BatchStrideA_ =
                a_g_n_k_wos_strides_transposed[1] * conv_N_per_block_;
            compute_ptr_offset_of_n_.BatchStrideE_ =
                e_g_n_c_wis_strides_transposed[1] * conv_N_per_block_;

            num_workgroups_per_Conv_N_ = a_g_n_k_wos_lengths_[I1] / conv_N_per_block_;

            if constexpr(is_NGCHW_GKYXC_NGKHW<ELayout, BLayout, ALayout>() ||
                         is_NGCDHW_GKZYXC_NGKDHW<ELayout, BLayout, ALayout>())
            {
                // Use not modified base strides
                a_in_transpose_desc_ =
                    conv_ngchw_to_nhwgc_transformer.template MakeNGCHWTransposeDesc<NDimSpatial>(
                        a_g_n_k_wos_lengths, a_g_n_k_wos_strides, num_workgroups_per_Conv_N_);
                a_out_transpose_desc_ =
                    conv_ngchw_to_nhwgc_transformer.template MakeNHWGCTransposeDesc<NDimSpatial>(
                        a_g_n_k_wos_lengths, a_g_n_k_wos_strides, num_workgroups_per_Conv_N_);

                e_in_transpose_desc_ =
                    conv_ngchw_to_nhwgc_transformer.template MakeNHWGCTransposeDesc<NDimSpatial>(
                        e_g_n_c_wis_lengths, e_g_n_c_wis_strides, num_workgroups_per_Conv_N_);
                e_out_transpose_desc_ =
                    conv_ngchw_to_nhwgc_transformer.template MakeNGCHWTransposeDesc<NDimSpatial>(
                        e_g_n_c_wis_lengths, e_g_n_c_wis_strides, num_workgroups_per_Conv_N_);

                elementwise_block_2_ctile_map_transpose_a_ = Block2TileMapElementwise{
                    a_in_transpose_desc_.GetLength(I0), a_in_transpose_desc_.GetLength(I1)};
                elementwise_block_2_ctile_map_transpose_e_ = Block2TileMapElementwise{
                    e_in_transpose_desc_.GetLength(I0), e_in_transpose_desc_.GetLength(I1)};

                compute_ptr_offset_of_workspace_n_.BatchStrideA_ =
                    a_g_n_k_wos_strides[1] * conv_N_per_block_;
                compute_ptr_offset_of_workspace_n_.BatchStrideE_ =
                    e_g_n_c_wis_strides[1] * conv_N_per_block_;
            }
        }

        std::size_t GetWorkspaceATensorSizeBytes() const
        {
            const long_index_t a_acum = ck::accumulate_n<long_index_t>(
                a_g_n_k_wos_lengths_.begin(), NDimSpatial + I3, 1, std::multiplies<>());
            return sizeof(ADataType) * a_acum;
        }

        std::size_t GetWorkspaceETensorSizeBytes() const
        {
            const long_index_t e_accum = ck::accumulate_n<long_index_t>(
                e_g_n_c_wis_lengths_.begin(), NDimSpatial + I3, 1, std::multiplies<>());
            return sizeof(EDataType) * e_accum;
        }

        std::size_t GetWorkspaceSizeBytes() const
        {
            // Transpose require workspace for A and B
            if constexpr(is_NGCHW_GKYXC_NGKHW<ELayout, BLayout, ALayout>() ||
                         is_NGCDHW_GKZYXC_NGKDHW<ELayout, BLayout, ALayout>())
            {
                return GetWorkspaceATensorSizeBytes() + GetWorkspaceETensorSizeBytes();
            }
            else
            {
                return 0;
            }
        }

        void Print() const
        {
            for(std::size_t i = 0; i < a_grid_desc_ak0_m_ak1_container_.size(); i++)
            {
                std::cout << "a_grid_desc_ak0_m_ak1_container_"
                          << a_grid_desc_ak0_m_ak1_container_[i] << std::endl;

                std::cout << "b_grid_desc_bk0_n_bk1_container_"
                          << b_grid_desc_bk0_n_bk1_container_[i] << std::endl;

                static_for<0, NumDTensor, 1>{}([&](auto j) {
                    std::cout << "ds_grid_desc_mblock_mperblock_nblock_nperblock_container_"
                              << ds_grid_desc_mblock_mperblock_nblock_nperblock_container_[i][j]
                              << std::endl;
                });

                std::cout << "e_grid_desc_mblock_mperblock_nblock_nperblock_container_"
                          << e_grid_desc_mblock_mperblock_nblock_nperblock_container_[i]
                          << std::endl;
            }
        }

        // pointers
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        typename GridwiseGemm::DsGridPointer p_ds_grid_;
        EDataType* p_e_grid_;

        // tensor descriptor for problem definition
        index_t num_group_;
        index_t conv_N_per_block_;
        std::vector<AGridDesc_M_K> a_grid_desc_m_k_container_;
        std::vector<BGridDesc_N_K> b_grid_desc_n_k_container_;
        std::vector<DsGridDesc_M_N> ds_grid_desc_m_n_container_;
        std::vector<EGridDesc_M_N> e_grid_desc_m_n_container_;

        // tensor descriptor for block-wise copy
        std::vector<AGridDesc_AK0_M_AK1> a_grid_desc_ak0_m_ak1_container_;
        std::vector<BGridDesc_BK0_N_BK1> b_grid_desc_bk0_n_bk1_container_;
        std::vector<DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>
            ds_grid_desc_mblock_mperblock_nblock_nperblock_container_;
        std::vector<EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>
            e_grid_desc_mblock_mperblock_nblock_nperblock_container_;

        // block-to-e-tile map
        std::vector<Block2ETileMap> block_2_etile_map_container_;
        Block2TileMapElementwise elementwise_block_2_ctile_map_transpose_a_,
            elementwise_block_2_ctile_map_transpose_e_;

        NGCHWTransposeDescType a_in_transpose_desc_, e_out_transpose_desc_;
        NHWGCTransposeDescType a_out_transpose_desc_, e_in_transpose_desc_;

        // for computing batch offset
        ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor> compute_ptr_offset_of_batch_;
        ComputePtrOffsetOfStridedBatch<I1, I1, I0> compute_ptr_offset_of_n_;
        ComputePtrOffsetOfStridedBatch<I1, I1, I0> compute_ptr_offset_of_workspace_n_;

        // element-wise op
        AElementwiseOp a_element_op_;
        BElementwiseOp b_element_op_;
        CDEElementwiseOp cde_element_op_;

        std::array<index_t, NDimSpatial + 3> a_g_n_k_wos_lengths_;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_lengths_;
        std::array<index_t, NDimSpatial + 3> e_g_n_c_wis_lengths_;
        std::array<index_t, NDimSpatial> conv_filter_strides_;
        std::array<index_t, NDimSpatial> input_left_pads_;
        std::array<index_t, NDimSpatial> input_right_pads_;

        index_t num_workgroups_per_Conv_N_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float RunGemm(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            float ave_time = 0;

            const index_t gdy = arg.num_group_;
            const index_t gdz = arg.num_workgroups_per_Conv_N_;

            const ADataType* p_a_grid = arg.p_a_grid_;
            EDataType* p_e_grid       = arg.p_e_grid_;

            if constexpr(is_NGCHW_GKYXC_NGKHW<ELayout, BLayout, ALayout>() ||
                         is_NGCDHW_GKZYXC_NGKDHW<ELayout, BLayout, ALayout>())
            {
                p_a_grid = type_convert<const ADataType*>(arg.p_workspace_);
                p_e_grid = type_convert<EDataType*>(arg.p_workspace_) +
                           arg.GetWorkspaceATensorSizeBytes() / sizeof(EDataType);
            }

            for(std::size_t i = 0; i < arg.a_grid_desc_ak0_m_ak1_container_.size(); i++)
            {
                if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_m_k_container_[i],
                                                arg.b_grid_desc_n_k_container_[i],
                                                arg.ds_grid_desc_m_n_container_[i],
                                                arg.e_grid_desc_m_n_container_[i],
                                                arg.block_2_etile_map_container_[i]))
                {
                    throw std::runtime_error("wrong! device_op has invalid setting");
                }

                const index_t gdx = arg.block_2_etile_map_container_[i].CalculateGridSize(
                    arg.e_grid_desc_m_n_container_[i]);

                const auto GemmK = arg.a_grid_desc_m_k_container_[i].GetLength(I1);

                auto launch_kernel = [&](auto has_main_k_block_loop) {
                    constexpr bool has_main_loop = has_main_k_block_loop.value;

                    const auto kernel = kernel_grouped_conv_bwd_data_multiple_d_xdl_cshuffle<
                        GridwiseGemm,
                        ADataType, // TODO: distiguish A/B datatype
                        typename GridwiseGemm::DsGridPointer,
                        EDataType,
                        AElementwiseOp,
                        BElementwiseOp,
                        CDEElementwiseOp,
                        DeviceOp::AGridDesc_AK0_M_AK1,
                        DeviceOp::BGridDesc_BK0_N_BK1,
                        DeviceOp::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                        DeviceOp::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                        Block2ETileMap,
                        ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                        ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                        has_main_loop>;

                    return launch_and_time_kernel(
                        stream_config,
                        kernel,
                        dim3(gdx, gdy, gdz),
                        dim3(BlockSize),
                        0,
                        p_a_grid,
                        arg.p_b_grid_,
                        arg.p_ds_grid_,
                        p_e_grid,
                        arg.a_element_op_,
                        arg.b_element_op_,
                        arg.cde_element_op_,
                        arg.a_grid_desc_ak0_m_ak1_container_[i],
                        arg.b_grid_desc_bk0_n_bk1_container_[i],
                        arg.ds_grid_desc_mblock_mperblock_nblock_nperblock_container_[i],
                        arg.e_grid_desc_mblock_mperblock_nblock_nperblock_container_[i],
                        arg.block_2_etile_map_container_[i],
                        arg.compute_ptr_offset_of_batch_,
                        arg.compute_ptr_offset_of_n_);
                };

                if(GridwiseGemm::CalculateHasMainKBlockLoop(GemmK))
                {
                    ave_time += launch_kernel(integral_constant<bool, true>{});
                }
                else
                {
                    ave_time += launch_kernel(integral_constant<bool, false>{});
                }
            }

            return ave_time;
        }

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            float ave_time = 0;

            if(stream_config.log_level_ > 0)
            {
                arg.Print();
            }
            // Transpose from NGKHW to NHWGK
            if constexpr(is_NGCHW_GKYXC_NGKHW<ELayout, BLayout, ALayout>() ||
                         is_NGCDHW_GKZYXC_NGKDHW<ELayout, BLayout, ALayout>())
            {
                EDataType* p_e_in_grid = type_convert<EDataType*>(arg.p_workspace_) +
                                         arg.GetWorkspaceATensorSizeBytes() / sizeof(EDataType);

                const auto clear_workspace = [&]() {
                    hip_check_error(hipMemsetAsync(p_e_in_grid,
                                                   0,
                                                   arg.GetWorkspaceETensorSizeBytes(),
                                                   stream_config.stream_id_));
                };

                const index_t grid_size =
                    arg.elementwise_block_2_ctile_map_transpose_a_.CalculateGridSize(
                        arg.a_in_transpose_desc_) *
                    arg.num_workgroups_per_Conv_N_;

                ADataType* p_a_out_grid = type_convert<ADataType*>(arg.p_workspace_);

                auto kernel_transpose =
                    kernel_batched_elementwise<GridwiseElementwiseInputTranspose,
                                               ck::Tuple<NGCHWTransposeDescType>,
                                               ck::Tuple<NHWGCTransposeDescType>,
                                               ck::Tuple<const ADataType*>,
                                               ck::Tuple<ADataType*>,
                                               Block2TileMapElementwise,
                                               element_wise::PassThrough,
                                               I1,
                                               I1>;

                ave_time += launch_and_time_kernel_with_preprocess(
                    stream_config,
                    clear_workspace,
                    kernel_transpose,
                    dim3(grid_size),
                    dim3(ElementwiseBlocksize),
                    0,
                    make_tuple(arg.a_in_transpose_desc_),
                    make_tuple(arg.a_out_transpose_desc_),
                    make_tuple(arg.p_a_grid_),
                    make_tuple(p_a_out_grid),
                    arg.elementwise_block_2_ctile_map_transpose_a_,
                    element_wise::PassThrough{},
                    arg.num_workgroups_per_Conv_N_,
                    std::array<index_t, I1>{
                        static_cast<index_t>(arg.compute_ptr_offset_of_workspace_n_.BatchStrideA_)},
                    std::array<index_t, I1>{
                        static_cast<index_t>(arg.compute_ptr_offset_of_n_.BatchStrideA_)});
            }
            ave_time += RunGemm(arg, stream_config);
            // Transpose from NHWGC to NGCHW
            if constexpr(is_NGCHW_GKYXC_NGKHW<ELayout, BLayout, ALayout>() ||
                         is_NGCDHW_GKZYXC_NGKDHW<ELayout, BLayout, ALayout>())
            {
                const index_t grid_size =
                    arg.elementwise_block_2_ctile_map_transpose_e_.CalculateGridSize(
                        arg.e_in_transpose_desc_) *
                    arg.num_workgroups_per_Conv_N_;

                const EDataType* p_e_in_grid =
                    type_convert<EDataType*>(arg.p_workspace_) +
                    arg.GetWorkspaceATensorSizeBytes() / sizeof(EDataType);

                EDataType* p_e_out_grid = arg.p_e_grid_;

                auto kernel_transpose =
                    kernel_batched_elementwise<GridwiseElementwiseOutputTranspose,
                                               ck::Tuple<NHWGCTransposeDescType>,
                                               ck::Tuple<NGCHWTransposeDescType>,
                                               ck::Tuple<const EDataType*>,
                                               ck::Tuple<EDataType*>,
                                               Block2TileMapElementwise,
                                               element_wise::PassThrough,
                                               I1,
                                               I1>;

                ave_time += launch_and_time_kernel(
                    stream_config,
                    kernel_transpose,
                    dim3(grid_size),
                    dim3(ElementwiseBlocksize),
                    0,
                    make_tuple(arg.e_in_transpose_desc_),
                    make_tuple(arg.e_out_transpose_desc_),
                    make_tuple(p_e_in_grid),
                    make_tuple(p_e_out_grid),
                    arg.elementwise_block_2_ctile_map_transpose_e_,
                    element_wise::PassThrough{},
                    arg.num_workgroups_per_Conv_N_,
                    std::array<index_t, I1>{
                        static_cast<index_t>(arg.compute_ptr_offset_of_n_.BatchStrideE_)},
                    std::array<index_t, I1>{static_cast<index_t>(
                        arg.compute_ptr_offset_of_workspace_n_.BatchStrideE_)});
            }

            return ave_time;
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(!ck::is_xdl_supported())
        {
            return false;
        }

        const index_t ConvG = arg.b_g_k_c_xs_lengths_[0];
        const index_t ConvK = arg.b_g_k_c_xs_lengths_[1];
        const index_t ConvC = arg.b_g_k_c_xs_lengths_[2];

        // Specifialization
        if constexpr(ConvBackwardDataSpecialization ==
                     ConvolutionBackwardDataSpecialization::Filter1x1Stride1Pad0)
        {
            // check if it's 1x1, stride=1 pad = 0 conv
            for(int i = 0; i < NDimSpatial; i++)
            {
                if(!(arg.b_g_k_c_xs_lengths_[3 + i] == 1 && arg.conv_filter_strides_[i] == 1 &&
                     arg.input_left_pads_[i] == 0 && arg.input_right_pads_[i] == 0))
                {
                    return false;
                }
            }
        }

        // vector load for A matrix from global memory to LDS
        if constexpr(is_same_v<ALayout, tensor_layout::convolution::GNHWK> ||
                     is_same_v<ALayout, tensor_layout::convolution::GNDHWK> ||
                     is_same_v<ALayout, tensor_layout::convolution::NHWGK> ||
                     is_same_v<ALayout, tensor_layout::convolution::NDHWGK> ||
                     is_same_v<ALayout, tensor_layout::convolution::NGKHW> ||
                     is_same_v<ALayout, tensor_layout::convolution::NGKDHW>)
        {
            if(!(ABlockTransferSrcVectorDim == 2 && ConvK % ABlockTransferSrcScalarPerVector == 0))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        // vector load for B matrix from global memory to LDS
        if constexpr(is_same_v<BLayout, tensor_layout::convolution::GKYXC> ||
                     is_same_v<BLayout, tensor_layout::convolution::GKZYXC>)
        {
            if(!(BBlockTransferSrcVectorDim == 1 && ConvC % BBlockTransferSrcScalarPerVector == 0))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        // vector store for Ds
        bool ds_valid = true;

        static_for<0, NumDTensor, 1>{}([&](auto i) {
            using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

            if constexpr(is_same_v<DLayout, tensor_layout::convolution::GNHWC> ||
                         is_same_v<DLayout, tensor_layout::convolution::GNDHWC> ||
                         is_same_v<DLayout, tensor_layout::convolution::NHWGC> ||
                         is_same_v<DLayout, tensor_layout::convolution::NDHWGC> ||
                         is_same_v<DLayout, tensor_layout::convolution::G_NHW_C> ||
                         is_same_v<DLayout, tensor_layout::convolution::GC> ||
                         is_same_v<DLayout, tensor_layout::convolution::G_C>)
            {
                // vector load D matrix from global memory
                if(!(ConvC % CDEBlockTransferScalarPerVector_NPerBlock == 0))
                {
                    ds_valid = false;
                }
            }
            else
            {
                ds_valid = false;
            }
        });

        if(!ds_valid)
        {
            return false;
        }

        // vector store for E
        if constexpr(is_same_v<ELayout, tensor_layout::convolution::GNHWC> ||
                     is_same_v<ELayout, tensor_layout::convolution::GNDHWC> ||
                     is_same_v<ELayout, tensor_layout::convolution::NHWGC> ||
                     is_same_v<ELayout, tensor_layout::convolution::NDHWGC> ||
                     is_same_v<ELayout, tensor_layout::convolution::NGCHW> ||
                     is_same_v<ELayout, tensor_layout::convolution::NGCDHW>)
        {
            // vector store C matrix into global memory
            if(!(ConvC % CDEBlockTransferScalarPerVector_NPerBlock == 0))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        // Gridwise GEMM size
        for(std::size_t i = 0; i < arg.a_grid_desc_ak0_m_ak1_container_.size(); i++)
        {
            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_m_k_container_[i],
                                            arg.b_grid_desc_n_k_container_[i],
                                            arg.ds_grid_desc_m_n_container_[i],
                                            arg.e_grid_desc_m_n_container_[i],
                                            arg.block_2_etile_map_container_[i]))
            {
                return false;
            }
        }

        if constexpr(is_NGCHW_GKYXC_NGKHW<ELayout, BLayout, ALayout>() ||
                     is_NGCDHW_GKZYXC_NGKDHW<ELayout, BLayout, ALayout>())
        {
            if((ConvG * ConvC) % CDEBlockTransferScalarPerVector_NPerBlock != 0)
            {
                return false;
            }

            if((ConvG * ConvK) % CDEBlockTransferScalarPerVector_NPerBlock != 0)
            {
                return false;
            }

            const index_t a_spatial_acum = ck::accumulate_n<index_t>(
                arg.a_g_n_k_wos_lengths_.begin() + I3, NDimSpatial, 1, std::multiplies<>());
            const index_t e_spatial_acum = ck::accumulate_n<index_t>(
                arg.e_g_n_c_wis_lengths_.begin() + I3, NDimSpatial, 1, std::multiplies<>());

            if(a_spatial_acum % TransposeTransferInScalarPerVectorAligned != 0)
            {
                return false;
            }

            if(e_spatial_acum % TransposeTransferOutScalarPerVectorAligned != 0)
            {
                return false;
            }

            if(!arg.p_workspace_)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout
                        << "Warning: Workspace for "
                           "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1::Argument is not "
                           "allocated, use SetWorkSpacePointer."
                        << std::endl;
                }
                return false;
            }
        }

        return true;
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto
    MakeArgument(const void* p_a,                                                 // output image
                 const void* p_b,                                                 // weight
                 const std::array<const void*, NumDTensor>& p_ds,                 // bias
                 void* p_e,                                                       // input image
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output image
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides, // output image
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,  // weight
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,  // weight
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_c_wis_lengths, // bias
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_c_wis_strides,                                        // bias
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_c_wis_lengths, // input image
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_c_wis_strides, // input image
                 const std::array<index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<index_t, NDimSpatial>& input_left_pads,
                 const std::array<index_t, NDimSpatial>& input_right_pads,
                 const AElementwiseOp& a_element_op,
                 const BElementwiseOp& b_element_op,
                 const CDEElementwiseOp& cde_element_op)
    {
        return Argument{p_a,
                        p_b,
                        p_ds,
                        p_e,
                        a_g_n_k_wos_lengths,
                        a_g_n_k_wos_strides,
                        b_g_k_c_xs_lengths,
                        b_g_k_c_xs_strides,
                        ds_g_n_c_wis_lengths,
                        ds_g_n_c_wis_strides,
                        e_g_n_c_wis_lengths,
                        e_g_n_c_wis_strides,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        a_element_op,
                        b_element_op,
                        cde_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const void* p_a,                                                 // output image
        const void* p_b,                                                 // weight
        const std::array<const void*, NumDTensor>& p_ds,                 // bias
        void* p_e,                                                       // input image
        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output image
        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides, // output image
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,  // weight
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,  // weight
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
            ds_g_n_c_wis_lengths, // bias
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
            ds_g_n_c_wis_strides,                                        // bias
        const std::array<index_t, NDimSpatial + 3>& e_g_n_c_wis_lengths, // input image
        const std::array<index_t, NDimSpatial + 3>& e_g_n_c_wis_strides, // input image
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& input_left_pads,
        const std::array<index_t, NDimSpatial>& input_right_pads,
        const AElementwiseOp& a_element_op,
        const BElementwiseOp& b_element_op,
        const CDEElementwiseOp& cde_element_op) override
    {
        return std::make_unique<Argument>(p_a,
                                          p_b,
                                          p_ds,
                                          p_e,
                                          a_g_n_k_wos_lengths,
                                          a_g_n_k_wos_strides,
                                          b_g_k_c_xs_lengths,
                                          b_g_k_c_xs_strides,
                                          ds_g_n_c_wis_lengths,
                                          ds_g_n_c_wis_strides,
                                          e_g_n_c_wis_lengths,
                                          e_g_n_c_wis_strides,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          a_element_op,
                                          b_element_op,
                                          cde_element_op);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1 << ", "
            << getConvBackwardDataSpecializationString(ConvBackwardDataSpecialization) << ", "
            << MPerXDL << ", "
            << NPerXDL << ", "
            << MXdlPerWave << ", "
            << NXdlPerWave << ", "
            << ABlockTransferSrcScalarPerVector << ", "
            << BBlockTransferSrcScalarPerVector << ", "
            << CShuffleMXdlPerWavePerShuffle << ", "
            << CShuffleNXdlPerWavePerShuffle;

            if constexpr(is_NGCHW_GKYXC_NGKHW<ELayout, BLayout, ALayout>() ||
                        is_NGCDHW_GKZYXC_NGKDHW<ELayout, BLayout, ALayout>()) {
                    str << ", TransposeTransferInScalarPerVectorAligned: "
                    << TransposeTransferInScalarPerVectorAligned <<", "
                    << "TransposeTransferOutScalarPerVectorAligned: " << TransposeTransferOutScalarPerVectorAligned;
                }
    
                
            str << ">";

        return str.str();
    }

    size_t GetWorkSpaceSize(const BaseArgument* p_arg) const override
    {
        auto arg = dynamic_cast<const Argument*>(p_arg);
        if(arg)
        {
            return arg->GetWorkspaceSizeBytes();
        }
        else
            throw std::runtime_error(
                "The argument pointer is not an object of "
                "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1::Argument structure!");
    }

    void SetWorkSpacePointer(BaseArgument* p_arg,
                             void* p_workspace,
                             const StreamConfig& = StreamConfig{}) const override
    {
        auto p_arg_ = dynamic_cast<Argument*>(p_arg);
        if(p_arg_)
        {
            p_arg_->p_workspace_ = p_workspace;
        }
        else
            throw std::runtime_error(
                "The argument pointer is not an object of "
                "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1::Argument structure!");
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
