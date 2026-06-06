// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>

#include <sstream>

#include "ck/library/utility/numeric.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/utility/env.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_data_multiple_d.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_data_specialization.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_bwd_data_to_gemm_v1.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_ngchw_to_nhwgc.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_conv_v3.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_elementwise_2d.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/flush_cache.hpp"
#include "ck/host_utility/io.hpp"

#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/device/tensor_size_check.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

namespace {

template <typename GridwiseGemm,
          typename AGridDesc_AK0_M_AK1,
          typename BGridDesc_BK0_N_BK1,
          typename EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t MaxGroupedGemmGroupsNum,
          typename GemmArgs,
          typename ComputePtrOffsetOfBatch,
          typename ComputePtrOffsetOfN,
          InMemoryDataOperationEnum EGlobalMemoryDataOperation,
          bool HasMainKBlockLoop,
          bool NoMainKBlockLoop,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
    kernel_grouped_conv_bwd_data_xdl_cshuffle_v3(
        typename GridwiseGemm::Argument karg,
        const std::array<GemmArgs, MaxGroupedGemmGroupsNum> gemm_kernel_args,
        const index_t gemms_count,
        const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch,
        const index_t num_k_per_block)
{
#if defined(__gfx9__)
    // offset base pointer for each work-group
    const index_t block_args_id = __builtin_amdgcn_readfirstlane(blockIdx.x);
    const index_t g_idx         = __builtin_amdgcn_readfirstlane(blockIdx.y);
    const index_t k_idx         = __builtin_amdgcn_readfirstlane(blockIdx.z * num_k_per_block);

    const long_index_t a_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx));
    const long_index_t b_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx));
    const long_index_t e_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetEPtrOffset(g_idx));

    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte(get_device_arch())];

    index_t left     = 0;
    index_t right    = gemms_count;
    index_t group_id = index_t((left + right) / 2);
    while((!(block_args_id >= gemm_kernel_args[group_id].BlockStart_ &&
             block_args_id < gemm_kernel_args[group_id].BlockEnd_)) &&
          left <= right)
    {
        if(block_args_id < gemm_kernel_args[group_id].BlockStart_)
        {
            right = group_id;
        }
        else
        {
            left = group_id;
        }
        group_id = index_t((left + right) / 2);
    }

    if constexpr(GridwiseGemm::DirectLoadEnabled)
    {
#if defined(__gfx950__)
        const auto a_grid_desc_ak0_m_ak1_transformed =
            GridwiseGemm::template TransformGrid<AGridDesc_AK0_M_AK1,
                                                 GridwiseGemm::AK0Number,
                                                 GridwiseGemm::AK1Number>(
                gemm_kernel_args[group_id].a_grid_desc_ak0_m_ak1_);
        if(gemm_kernel_args[group_id].HasMainKBlockLoop_)
        {
            GridwiseGemm::template Run<decltype(a_grid_desc_ak0_m_ak1_transformed),
                                       BGridDesc_BK0_N_BK1,
                                       EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                                       true,
                                       EGlobalMemoryDataOperation,
                                       TailNum>(
                karg.p_a_grid + a_batch_offset,
                karg.p_b_grid + b_batch_offset,
                karg.p_c_grid + e_batch_offset,
                p_shared,
                karg,
                a_grid_desc_ak0_m_ak1_transformed,
                gemm_kernel_args[group_id].b_grid_desc_bk0_n_bk1_,
                gemm_kernel_args[group_id].e_grid_desc_mblock_mperblock_nblock_nperblock_,
                k_idx,
                gridDim.z,
                blockIdx.x - gemm_kernel_args[group_id].BlockStart_);
        }
        else
        {
            GridwiseGemm::template Run<decltype(a_grid_desc_ak0_m_ak1_transformed),
                                       BGridDesc_BK0_N_BK1,
                                       EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                                       false,
                                       EGlobalMemoryDataOperation,
                                       TailNum>(
                karg.p_a_grid + a_batch_offset,
                karg.p_b_grid + b_batch_offset,
                karg.p_c_grid + e_batch_offset,
                p_shared,
                karg,
                a_grid_desc_ak0_m_ak1_transformed,
                gemm_kernel_args[group_id].b_grid_desc_bk0_n_bk1_,
                gemm_kernel_args[group_id].e_grid_desc_mblock_mperblock_nblock_nperblock_,
                k_idx,
                gridDim.z,
                blockIdx.x - gemm_kernel_args[group_id].BlockStart_);
        }
#endif
    }
    else
    {
        if(gemm_kernel_args[group_id].HasMainKBlockLoop_)
        {
            GridwiseGemm::template Run<AGridDesc_AK0_M_AK1,
                                       BGridDesc_BK0_N_BK1,
                                       EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                                       true,
                                       EGlobalMemoryDataOperation,
                                       TailNum>(
                karg.p_a_grid + a_batch_offset,
                karg.p_b_grid + b_batch_offset,
                karg.p_c_grid + e_batch_offset,
                p_shared,
                karg,
                gemm_kernel_args[group_id].a_grid_desc_ak0_m_ak1_,
                gemm_kernel_args[group_id].b_grid_desc_bk0_n_bk1_,
                gemm_kernel_args[group_id].e_grid_desc_mblock_mperblock_nblock_nperblock_,
                k_idx,
                gridDim.z,
                blockIdx.x - gemm_kernel_args[group_id].BlockStart_);
        }
        else
        {
            GridwiseGemm::template Run<AGridDesc_AK0_M_AK1,
                                       BGridDesc_BK0_N_BK1,
                                       EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                                       false,
                                       EGlobalMemoryDataOperation,
                                       TailNum>(
                karg.p_a_grid + a_batch_offset,
                karg.p_b_grid + b_batch_offset,
                karg.p_c_grid + e_batch_offset,
                p_shared,
                karg,
                gemm_kernel_args[group_id].a_grid_desc_ak0_m_ak1_,
                gemm_kernel_args[group_id].b_grid_desc_bk0_n_bk1_,
                gemm_kernel_args[group_id].e_grid_desc_mblock_mperblock_nblock_nperblock_,
                k_idx,
                gridDim.z,
                blockIdx.x - gemm_kernel_args[group_id].BlockStart_);
        }
    }
#else
    ignore = karg;
    ignore = gemm_kernel_args;
    ignore = gemms_count;
    ignore = compute_ptr_offset_of_batch;
    ignore = num_k_per_block;

#endif // End of if (!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx11__) || defined(__gfx12__))
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
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerXdl,
          index_t NPerXdl,
          index_t MRepeat,
          index_t NRepeat,
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
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename AComputeType                       = ADataType,
          typename BComputeType                       = AComputeType,
          bool DirectLoad                             = false,
          bool LargeTensors                           = false>
struct DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffleV3
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

    static_assert(std::is_same_v<ALayout, tensor_layout::convolution::NHWGK> ||
                      std::is_same_v<ALayout, tensor_layout::convolution::NDHWGK>,
                  "A not NGHWC");
    static_assert(std::is_same_v<BLayout, tensor_layout::convolution::GKYXC> ||
                      std::is_same_v<BLayout, tensor_layout::convolution::GKZYXC>,
                  "B not GKYXC");
    static_assert(std::is_same_v<ELayout, tensor_layout::convolution::NHWGC> ||
                      std::is_same_v<ELayout, tensor_layout::convolution::NDHWGC>,
                  "C not NGHWK");

    // MaxGroupedGemmGroupsNum  is used to specify number of gemm args in compile time. With this
    // implementation we can avoid copy data to workspace before kernel launch since number of
    // groups is runtime parameter. If number of groups is larger than MaxGroupedGemmGroupsNum  then
    // we run this kernel in the loop.
    static constexpr index_t MaxGroupedGemmGroupsNum =
        ConvBackwardDataSpecialization ==
                ConvolutionBackwardDataSpecialization::Filter1x1Stride1Pad0
            ? 1
            : 32;

    using IndexType = std::conditional_t<LargeTensors, ck::long_index_t, ck::index_t>;

    using DeviceOp = DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffleV3;

    // Wave32 support: compute effective MXdlPerWave for wave64 and wave32 modes.
    // The bwd_data template uses MRepeat/NRepeat as MXdlPerWave/NXdlPerWave and
    // MPerXdl/NPerXdl (lowercase 'dl') instead of MPerXDL/NPerXDL.
    template <bool IsWave64,
              index_t MPerXdlAligned = MPerXdl,
              index_t NPerXdlAligned = NPerXdl,
              index_t NRepeatAligned = NRepeat>
    static constexpr auto GetMXdlPerWave()
    {
        return GetXdlPerWave2<BlockSize,
                              NPerBlock,
                              MPerBlock,
                              NPerXdlAligned,
                              MPerXdlAligned,
                              NRepeatAligned,
                              IsWave64>();
    }

    static constexpr bool Wave32Force16MNPerXdl = sizeof(ADataType) == 2 && sizeof(BDataType) == 2;
    static constexpr index_t Wave32MaxMNPerXdl =
        Wave32Force16MNPerXdl ? 16 : math::max(MPerXdl, NPerXdl);

    static constexpr auto MXdlPerWave64 = GetMXdlPerWave<true>();
    static constexpr auto MXdlPerWave32 = GetMXdlPerWave<false,
                                                         Wave32MaxMNPerXdl,
                                                         Wave32MaxMNPerXdl,
                                                         NRepeat*(NPerXdl / Wave32MaxMNPerXdl)>();

    static constexpr index_t NumDTensor = DsDataType::Size();
    static constexpr auto I0            = Number<0>{};
    static constexpr auto I1            = Number<1>{};
    static constexpr auto I2            = Number<2>{};
    static constexpr auto I3            = Number<3>{};
    static_assert(NumDTensor == 0, "Not supported");
    // static_assert(DirectLoad, "Not supported");

    static constexpr GemmSpecialization GemmSpec = GemmSpecialization::MNKPadding;
    static constexpr bool IsSplitKSupported      = false;

    // TODO: Add support for different A and B data types.
    using ABDataType = ADataType;

    using ConvToGemmBwdDataTransform = TransformConvBwdDataToGemm_v1<NDimSpatial,
                                                                     ConvBackwardDataSpecialization,
                                                                     AK1,
                                                                     BK1,
                                                                     MPerBlock,
                                                                     NPerBlock,
                                                                     KPerBlock,
                                                                     DoPadGemmM,
                                                                     DoPadGemmN,
                                                                     ALayout,
                                                                     BLayout,
                                                                     ELayout,
                                                                     false, /*SplitConvN*/
                                                                     ABDataType,
                                                                     EDataType,
                                                                     1,
                                                                     IndexType>;

    // Dummy function just used to create an alias to Grid Descriptors
    static auto
    GetDummyABDsEGridDescriptor(const ConvToGemmBwdDataTransform& conv_to_gemm_transform)
    {
        const auto a_grid_desc_ak0_m_ak1 = conv_to_gemm_transform.MakeADescriptor_AK0_M_AK1();

        const auto b_grid_desc_bk0_n_bk1 = conv_to_gemm_transform.MakeBDescriptor_BK0_N_BK1();

        const auto e_grid_desc_mblock_mperblock_nblock_nperblock =
            GridwiseGemm::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                conv_to_gemm_transform.MakeCDescriptor_M_N(), IndexType{1}, IndexType{1});

        return make_tuple(a_grid_desc_ak0_m_ak1,
                          b_grid_desc_bk0_n_bk1,
                          e_grid_desc_mblock_mperblock_nblock_nperblock);
    }

    static constexpr index_t ABlockTransferSrcScalarPerVectorAligned =
        ABlockTransferSrcScalarPerVector * sizeof(ADataType) == 8
            ? 4 / sizeof(ADataType)
            : ABlockTransferSrcScalarPerVector;
    static constexpr index_t BBlockTransferSrcScalarPerVectorAligned =
        BBlockTransferSrcScalarPerVector * sizeof(BDataType) == 8
            ? 4 / sizeof(BDataType)
            : BBlockTransferSrcScalarPerVector;

    static constexpr bool ALdsScalarLoadToVgpr = false;
    static constexpr bool BLdsScalarLoadToVgpr = true;

    // Parameterized GridwiseGemm template to support both wave64 (MPerXdl/NPerXdl) and
    // wave32 (Wave32MaxMNPerXdl/Wave32MaxMNPerXdl) XDL instruction sizes.
    template <index_t MRepeat_, index_t MPerXdl_, index_t NPerXdl_>
    using GridwiseGemmBase = GridwiseGemm_xdl_cshuffle_conv_v3<
        tensor_layout::gemm::RowMajor,
        tensor_layout::gemm::RowMajor,
        tensor_layout::gemm::RowMajor,
        ADataType,
        BDataType,
        AccDataType,
        EDataType,
        EDataType,
        AElementwiseOp,
        BElementwiseOp,
        CDEElementwiseOp,
        GemmSpec,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        MPerXdl_,
        NPerXdl_,
        MRepeat_,
        NRepeat*(NPerXdl / NPerXdl_),
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        DirectLoad ? ABlockTransferSrcScalarPerVectorAligned : ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        false,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        DirectLoad ? BBlockTransferSrcScalarPerVectorAligned : BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false,
        BBlockLdsExtraN,
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle*(NPerXdl / NPerXdl_),
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CShuffleBlockTransferScalarPerVector,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        AComputeType,
        BComputeType,
        DirectLoad,
        DirectLoad && ALdsScalarLoadToVgpr,
        DirectLoad && BLdsScalarLoadToVgpr,
        LargeTensors>;

    using GridwiseGemm64 = GridwiseGemmBase<math::max(MXdlPerWave64, 1), MPerXdl, NPerXdl>;
    using GridwiseGemm32 =
        GridwiseGemmBase<math::max(MXdlPerWave32, 1), Wave32MaxMNPerXdl, Wave32MaxMNPerXdl>;
    // Default GridwiseGemm alias for use in non-wave-size-dependent code paths
    using GridwiseGemm = GridwiseGemm64;

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

    // Note: the dummy function is used just to create the alias
    constexpr static ConvToGemmBwdDataTransform dummy_conv_to_gemm_transform;
    using ABDsEGridDesc = decltype(GetDummyABDsEGridDescriptor(dummy_conv_to_gemm_transform));

    using AGridDesc_AK0_M_AK1                  = remove_cvref_t<tuple_element_t<0, ABDsEGridDesc>>;
    using BGridDesc_BK0_N_BK1                  = remove_cvref_t<tuple_element_t<1, ABDsEGridDesc>>;
    using EGridDesc_MPerBlock_NBlock_NPerBlock = remove_cvref_t<tuple_element_t<2, ABDsEGridDesc>>;

    using AGridDesc_M_K = decltype(transform_k0_m_k1_to_m_k(AGridDesc_AK0_M_AK1{}));
    using BGridDesc_N_K = decltype(transform_k0_m_k1_to_m_k(BGridDesc_BK0_N_BK1{}));

    struct GemmArgs
    {
        GemmArgs() = default;
        GemmArgs(AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1,
                 BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1,
                 EGridDesc_MPerBlock_NBlock_NPerBlock e_grid_desc_mblock_mperblock_nblock_nperblock,
                 index_t BlockStart,
                 index_t BlockEnd,
                 bool HasMainKBlockLoop)
            : a_grid_desc_ak0_m_ak1_(a_grid_desc_ak0_m_ak1),
              b_grid_desc_bk0_n_bk1_(b_grid_desc_bk0_n_bk1),
              e_grid_desc_mblock_mperblock_nblock_nperblock_(
                  e_grid_desc_mblock_mperblock_nblock_nperblock),
              BlockStart_(BlockStart),
              BlockEnd_(BlockEnd),
              HasMainKBlockLoop_(HasMainKBlockLoop)

        {
        }
        // tensor descriptors for block/thread-wise copy
        AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1_;
        BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1_;
        EGridDesc_MPerBlock_NBlock_NPerBlock e_grid_desc_mblock_mperblock_nblock_nperblock_;
        index_t BlockStart_, BlockEnd_;
        bool HasMainKBlockLoop_;
    };
    // block-to-e-tile map for elementwise kernels
    using Block2TileMapInOutElementwise =
        BlockToCTileMap_M00_N0_M01Adapt<NPerBlock, MPerBlock, void, IndexType>;
    using Block2TileMapWeiElementwise =
        BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock, void, IndexType>;
    static constexpr index_t ClusterLengthMPerBlock =
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::At(1);
    static constexpr index_t ClusterLengthNPerBlock =
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::At(3);

    static constexpr index_t ElementwiseBlocksize = ClusterLengthMPerBlock * ClusterLengthNPerBlock;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const void* p_a,                            // output image
                 const void* p_b,                            // weight
                 const std::array<const void*, NumDTensor>&, // bias
                 void* p_e,                                  // input image
                 const std::array<IndexType, NDimSpatial + 3>& a_g_n_k_wos_lengths,
                 const std::array<IndexType, NDimSpatial + 3>& a_g_n_k_wos_strides,
                 const std::array<IndexType, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                 const std::array<IndexType, NDimSpatial + 3>& b_g_k_c_xs_strides,
                 const std::array<std::array<IndexType, NDimSpatial + 3>, NumDTensor>&,
                 const std::array<std::array<IndexType, NDimSpatial + 3>, NumDTensor>&,
                 const std::array<IndexType, NDimSpatial + 3>& e_g_n_c_wis_lengths,
                 const std::array<IndexType, NDimSpatial + 3>& e_g_n_c_wis_strides,
                 const std::array<IndexType, NDimSpatial>& conv_filter_strides,
                 const std::array<IndexType, NDimSpatial>& conv_filter_dilations,
                 const std::array<IndexType, NDimSpatial>& input_left_pads,
                 const std::array<IndexType, NDimSpatial>& input_right_pads,
                 const AElementwiseOp& a_element_op,
                 const BElementwiseOp& b_element_op,
                 const CDEElementwiseOp& cde_element_op,
                 ck::index_t split_k     = 1,
                 bool stride_overflow_in = false)
            : p_a_grid_{static_cast<const ADataType*>(p_a)},
              p_b_grid_{static_cast<const BDataType*>(p_b)},
              p_e_grid_{static_cast<EDataType*>(p_e)},
              num_group_{static_cast<index_t>(a_g_n_k_wos_lengths[0])},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op},
              a_g_n_k_wos_lengths_{a_g_n_k_wos_lengths},
              a_g_n_k_wos_strides_{a_g_n_k_wos_strides},
              b_g_k_c_xs_lengths_{b_g_k_c_xs_lengths},
              b_g_k_c_xs_strides_{b_g_k_c_xs_strides},
              e_g_n_c_wis_lengths_{e_g_n_c_wis_lengths},
              e_g_n_c_wis_strides_{e_g_n_c_wis_strides},
              conv_filter_strides_{conv_filter_strides},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads},
              k_batch_{split_k}
        {
            stride_overflow             = stride_overflow_in;
            bool image_covered_dilation = true;
            bool image_covered_strides  = true;
            for(index_t d = 0; d < NDimSpatial; d++)
            {
                // If dilation and stride is not equal we will have some empty places
                image_covered_dilation &=
                    conv_filter_dilations[d] == 1 || conv_filter_strides[d] == 1;
                // If stride is larger than windows size then we will have some empty places
                image_covered_strides &= conv_filter_strides[d] <= b_g_k_c_xs_lengths[d + I3];
            }
            bool if_d_is_output_mem = false;
            bwd_needs_zero_out = k_batch_ > 1 || !image_covered_dilation || !image_covered_strides;

            // Temporary workaround untill prove/fix above conditions.
            bwd_needs_zero_out = !if_d_is_output_mem;
            e_space_size_bytes =
                ck::accumulate_n<long_index_t>(
                    e_g_n_c_wis_lengths_.begin(), NDimSpatial + I3, 1, std::multiplies<>()) *
                sizeof(EDataType);

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
            const auto Z = b_g_k_c_xs_lengths[ZIdx];
            const auto Y = b_g_k_c_xs_lengths[YIdx];
            const auto X = b_g_k_c_xs_lengths[XIdx];

            const auto ConvStrideD = conv_filter_strides[DIdx - NonSpatialDimsNum];
            const auto ConvStrideH = conv_filter_strides[HIdx - NonSpatialDimsNum];
            const auto ConvStrideW = conv_filter_strides[WIdx - NonSpatialDimsNum];

            const auto ConvDilationD = conv_filter_dilations[DIdx - NonSpatialDimsNum];
            const auto ConvDilationH = conv_filter_dilations[HIdx - NonSpatialDimsNum];
            const auto ConvDilationW = conv_filter_dilations[WIdx - NonSpatialDimsNum];

            const auto GcdStrideDilationD = math::gcd(ConvStrideD, ConvDilationD);
            const auto GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
            const auto GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

            const auto ZTilde = NDimSpatial == 3 ? ConvStrideD / GcdStrideDilationD : IndexType{1};
            const auto YTilde = ConvStrideH / GcdStrideDilationH;
            const auto XTilde = ConvStrideW / GcdStrideDilationW;

            index_t grid_size = 0;
            // Allocate place for sets of gemms
            gemm_kernel_args_.resize(
                math::integer_divide_ceil(ZTilde * YTilde * XTilde, MaxGroupedGemmGroupsNum));

            for(IndexType i_ztilde = 0; i_ztilde < ZTilde; ++i_ztilde)
            {
                for(IndexType i_ytilde = 0; i_ytilde < YTilde; ++i_ytilde)
                {
                    for(IndexType i_xtilde = 0; i_xtilde < XTilde; ++i_xtilde)
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

                        std::array<IndexType, NDimSpatial> tildes;
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

                        ConvToGemmBwdDataTransform conv_to_gemm_transform_{a_g_n_k_wos_lengths,
                                                                           a_g_n_k_wos_strides,
                                                                           b_g_k_c_xs_lengths,
                                                                           b_g_k_c_xs_strides,
                                                                           e_g_n_c_wis_lengths,
                                                                           e_g_n_c_wis_strides,
                                                                           conv_filter_strides,
                                                                           conv_filter_dilations,
                                                                           input_left_pads,
                                                                           input_right_pads,
                                                                           tildes,
                                                                           k_batch_};

                        conv_N_per_block_ = conv_to_gemm_transform_.N_;

                        const auto a_grid_desc_ak0_m_ak1 = [&]() {
                            return conv_to_gemm_transform_.MakeADescriptor_AK0_M_AK1();
                        }();

                        const auto b_grid_desc_bk0_n_bk1 = [&]() {
                            return conv_to_gemm_transform_.MakeBDescriptor_BK0_N_BK1();
                        }();

                        // desc for problem definition
                        const auto a_grid_desc_m_k =
                            transform_k0_m_k1_to_m_k(a_grid_desc_ak0_m_ak1);
                        const auto b_grid_desc_n_k =
                            transform_k0_m_k1_to_m_k(b_grid_desc_bk0_n_bk1);

                        const IndexType GemmM = a_grid_desc_m_k.GetLength(I0);
                        const IndexType GemmN = b_grid_desc_n_k.GetLength(I0);

                        const auto e_grid_desc_mblock_mperblock_nblock_nperblock =
                            GridwiseGemm::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                                conv_to_gemm_transform_.MakeCDescriptor_M_N(),
                                GridwiseGemm::CalculateMBlock(GemmM),
                                GridwiseGemm::CalculateNBlock(GemmN));

                        a_grid_desc_m_k_container_.push_back(a_grid_desc_m_k);
                        b_grid_desc_n_k_container_.push_back(b_grid_desc_n_k);
                        e_grid_desc_mblock_mperblock_nblock_nperblock_container_.push_back(
                            e_grid_desc_mblock_mperblock_nblock_nperblock);

                        const index_t grid_size_grp =
                            std::get<0>(GridwiseGemm::CalculateGridSize(GemmM, GemmN, 1, 1));
                        const index_t BlockStart = grid_size;
                        const index_t BlockEnd   = grid_size + grid_size_grp;

                        grid_size += grid_size_grp;

                        // const index_t GemmM = a_grid_desc_m_k.GetLength(I0);
                        // const index_t GemmN = b_grid_desc_n_k.GetLength(I0);
                        const index_t GemmK = a_grid_desc_m_k.GetLength(I1);

                        // onst auto MBlock = GridwiseGemmCTranspose::CalculateMBlock(GemmM);
                        // onst auto NBlock = GridwiseGemmCTranspose::CalculateNBlock(GemmN);

                        index_t k_grain = split_k * KPerBlock;
                        index_t K_split = (GemmK + k_grain - 1) / k_grain * KPerBlock;

                        const bool HasMainKBlockLoop =
                            GridwiseGemm::CalculateHasMainKBlockLoop(K_split);

                        gemm_kernel_args_[gemms_count_ / MaxGroupedGemmGroupsNum]
                                         [gemms_count_ % MaxGroupedGemmGroupsNum] =
                                             GemmArgs{a_grid_desc_ak0_m_ak1,
                                                      b_grid_desc_bk0_n_bk1,
                                                      e_grid_desc_mblock_mperblock_nblock_nperblock,
                                                      BlockStart,
                                                      BlockEnd,
                                                      HasMainKBlockLoop};
                        gemms_count_++;
                        if(gemms_count_ % MaxGroupedGemmGroupsNum == 0)
                        {
                            gemms_grid_size_.push_back(grid_size);
                            grid_size = 0;
                        }
                    }
                }
            }
            gemm_kernel_args_.resize(
                math::integer_divide_ceil(gemms_count_, MaxGroupedGemmGroupsNum));
            gemms_grid_size_.push_back(grid_size);

            // A/B/Ds/E Batch Stride
            compute_ptr_offset_of_batch_.BatchStrideA_ = a_g_n_k_wos_strides[0];
            compute_ptr_offset_of_batch_.BatchStrideB_ = b_g_k_c_xs_strides[0];
            compute_ptr_offset_of_batch_.BatchStrideE_ = e_g_n_c_wis_strides[0];

            num_workgroups_per_Conv_N_ =
                static_cast<index_t>(a_g_n_k_wos_lengths_[I1]) / conv_N_per_block_;
        }

        std::size_t GetWorkspaceSizeBytes() const { return 0; }

        void Print() const
        {
            for(std::size_t i = 0; i < a_grid_desc_m_k_container_.size(); i++)
            {
                std::cout << "a_grid_desc_m_ak_container_" << a_grid_desc_m_k_container_[i]
                          << std::endl;

                std::cout << "b_grid_desc_n_bk_container_" << b_grid_desc_n_k_container_[i]
                          << std::endl;

                std::cout << "e_grid_desc_mblock_mperblock_nblock_nperblock_container_"
                          << e_grid_desc_mblock_mperblock_nblock_nperblock_container_[i]
                          << std::endl;
            }
        }

        // pointers
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        EDataType* p_e_grid_;

        // tensor descriptor for problem definition
        index_t num_group_;
        index_t conv_N_per_block_;
        std::vector<AGridDesc_M_K> a_grid_desc_m_k_container_;
        std::vector<BGridDesc_N_K> b_grid_desc_n_k_container_;
        std::vector<EGridDesc_MPerBlock_NBlock_NPerBlock>
            e_grid_desc_mblock_mperblock_nblock_nperblock_container_;

        // tensor descriptor for block-wise copy
        std::vector<AGridDesc_AK0_M_AK1> a_grid_desc_ak0_m_ak1_container_;
        std::vector<BGridDesc_BK0_N_BK1> b_grid_desc_bk0_n_bk1_container_;

        // for computing batch offset
        ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor> compute_ptr_offset_of_batch_;

        // element-wise op
        AElementwiseOp a_element_op_;
        BElementwiseOp b_element_op_;
        CDEElementwiseOp cde_element_op_;

        std::array<IndexType, NDimSpatial + 3> a_g_n_k_wos_lengths_;
        std::array<IndexType, NDimSpatial + 3> a_g_n_k_wos_strides_;
        std::array<IndexType, NDimSpatial + 3> b_g_k_c_xs_lengths_;
        std::array<IndexType, NDimSpatial + 3> b_g_k_c_xs_strides_;
        std::array<IndexType, NDimSpatial + 3> e_g_n_c_wis_lengths_;
        std::array<IndexType, NDimSpatial + 3> e_g_n_c_wis_strides_;
        std::array<IndexType, NDimSpatial> conv_filter_strides_;
        std::array<IndexType, NDimSpatial> input_left_pads_;
        std::array<IndexType, NDimSpatial> input_right_pads_;

        const index_t k_batch_;
        index_t num_workgroups_per_Conv_N_;
        std::vector<index_t> gemms_grid_size_;
        index_t gemms_count_ = 0;
        std::vector<std::array<GemmArgs, MaxGroupedGemmGroupsNum>> gemm_kernel_args_;

        bool bwd_needs_zero_out;
        long_index_t e_space_size_bytes;
        bool stride_overflow;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        template <typename GridwiseGemm_, InMemoryDataOperationEnum ElementOp>
        float RunMultiDGemm(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            float ave_time = 0;

            const index_t gdy = arg.num_group_;
            const index_t gdz = arg.k_batch_;

            const ADataType* p_a_grid = arg.p_a_grid_;
            const BDataType* p_b_grid = arg.p_b_grid_;
            EDataType* p_e_grid       = arg.p_e_grid_;

            for(std::size_t gemm_set_id = 0; gemm_set_id < arg.gemm_kernel_args_.size();
                gemm_set_id++)
            {
                const index_t GemmM = arg.a_grid_desc_m_k_container_[gemm_set_id].GetLength(I0);
                const index_t GemmN = arg.b_grid_desc_n_k_container_[gemm_set_id].GetLength(I0);
                const index_t GemmK = arg.a_grid_desc_m_k_container_[gemm_set_id].GetLength(I1);
                typename GridwiseGemm_::Argument gemm_arg{
                    p_a_grid, p_b_grid, p_e_grid, GemmM, GemmN, GemmK, I0, I0, I0, arg.k_batch_};

                const index_t gdx = arg.gemms_grid_size_[gemm_set_id];

                const index_t gemms_count_for_set =
                    gemm_set_id == arg.gemm_kernel_args_.size() - 1
                        ? arg.gemms_count_ - MaxGroupedGemmGroupsNum * gemm_set_id
                        : MaxGroupedGemmGroupsNum;

                const std::array<GemmArgs, MaxGroupedGemmGroupsNum>& gemm_kernel_args =
                    arg.gemm_kernel_args_[gemm_set_id];

                const auto clear_workspace = [&]() {
                    if(arg.bwd_needs_zero_out && gemm_set_id == 0)
                    {
                        hip_check_error(hipMemsetAsync(
                            p_e_grid, 0, arg.e_space_size_bytes, stream_config.stream_id_));
                    }
                };

                bool has_loop_in_all_gemm = true;
                bool no_loop_in_all_gemm  = true;
                for(auto i = 0; i < gemms_count_for_set; i++)
                {
                    has_loop_in_all_gemm &= gemm_kernel_args[i].HasMainKBlockLoop_;
                    no_loop_in_all_gemm &= !gemm_kernel_args[i].HasMainKBlockLoop_;
                }

                auto launch_kernel = [&](auto has_main_k_block_loop_, auto no_main_k_block_loop) {
                    constexpr bool has_main_loop = has_main_k_block_loop_.value;
                    constexpr bool no_main_loop  = no_main_k_block_loop.value;
                    const auto kernel            = kernel_grouped_conv_bwd_data_xdl_cshuffle_v3<
                                   GridwiseGemm_,
                                   DeviceOp::AGridDesc_AK0_M_AK1,
                                   DeviceOp::BGridDesc_BK0_N_BK1,
                                   DeviceOp::EGridDesc_MPerBlock_NBlock_NPerBlock,
                                   MaxGroupedGemmGroupsNum,
                                   GemmArgs,
                                   ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                   ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                                   ElementOp,
                                   has_main_loop,
                                   no_main_loop>;

                    return launch_and_time_kernel_with_preprocess(stream_config,
                                                                  clear_workspace,
                                                                  kernel,
                                                                  dim3(gdx, gdy, gdz),
                                                                  dim3(BlockSize),
                                                                  0,
                                                                  gemm_arg,
                                                                  gemm_kernel_args,
                                                                  gemms_count_for_set,
                                                                  arg.compute_ptr_offset_of_batch_,
                                                                  1);
                };
                if(has_loop_in_all_gemm)
                {
                    ave_time += launch_kernel(integral_constant<bool, true>{},
                                              integral_constant<bool, false>{});
                }
                else if(no_loop_in_all_gemm)
                {
                    ave_time += launch_kernel(integral_constant<bool, false>{},
                                              integral_constant<bool, true>{});
                }
                else
                {
                    ave_time += launch_kernel(integral_constant<bool, false>{},
                                              integral_constant<bool, false>{});
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

            if(get_warp_size() == 64)
            {
                if constexpr(MXdlPerWave64 > 0)
                {
                    if(arg.k_batch_ > 1)
                    {
                        if constexpr(IsSplitKSupported)
                        {
                            ave_time +=
                                RunMultiDGemm<GridwiseGemm64, InMemoryDataOperationEnum::AtomicAdd>(
                                    arg, stream_config);
                        }
                    }
                    else
                    {
                        ave_time += RunMultiDGemm<GridwiseGemm64, InMemoryDataOperationEnum::Set>(
                            arg, stream_config);
                    }
                }
            }
            else
            {
                if constexpr(MXdlPerWave32 > 0)
                {
                    if(arg.k_batch_ > 1)
                    {
                        if constexpr(IsSplitKSupported)
                        {
                            ave_time +=
                                RunMultiDGemm<GridwiseGemm32, InMemoryDataOperationEnum::AtomicAdd>(
                                    arg, stream_config);
                        }
                    }
                    else
                    {
                        ave_time += RunMultiDGemm<GridwiseGemm32, InMemoryDataOperationEnum::Set>(
                            arg, stream_config);
                    }
                }
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
        if constexpr(!LargeTensors)
        {
            if(arg.stride_overflow)
            {
                return false;
            }
        }

        if(get_warp_size() != 64)
        {
            // TODO: Enable for warp size 32
            return false;
        }
        // Reject if the current warp size has no valid XDL configuration
        // Warp size 32 is temporary not supported but leave it for the future
        if(get_warp_size() == 64)
        {
            if constexpr(MXdlPerWave64 == 0)
            {
                return false;
            }
        }
        else
        {
            if constexpr(MXdlPerWave32 == 0)
            {
                return false;
            }
        }

        // check device
        if constexpr(DirectLoad)
        {
            if(get_device_name() != "gfx950")
            {
                return false;
            }
        }

        if constexpr(!IsSplitKSupported)
        {
            if(arg.k_batch_ > 1)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "SplitK tests are not supported!" << " In " << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }

                return false;
            }
        }

        if(ck::is_gfx11_supported() && arg.k_batch_ > 1)
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "SplitK tests are not supported!" << " In " << __FILE__ << ":"
                          << __LINE__ << ", in function: " << __func__ << std::endl;
            }

            return false;
        }

        const index_t ConvK = static_cast<index_t>(arg.b_g_k_c_xs_lengths_[1]);
        const index_t ConvC = static_cast<index_t>(arg.b_g_k_c_xs_lengths_[2]);

        // Specialization
        if constexpr(ConvBackwardDataSpecialization ==
                     ConvolutionBackwardDataSpecialization::Filter1x1Stride1Pad0)
        {
            // check if it's 1x1, stride=1 pad = 0 conv
            for(int i = 0; i < NDimSpatial; i++)
            {
                if(!(arg.b_g_k_c_xs_lengths_[3 + i] == 1 && arg.conv_filter_strides_[i] == 1 &&
                     arg.input_left_pads_[i] == 0 && arg.input_right_pads_[i] == 0))
                {
                    if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                    {
                        std::cout << "ConvBwdDataSpecialization is unsupported!" << " In "
                                  << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                                  << std::endl;
                    }

                    return false;
                }
            }
        }

        // vector load for A matrix from global memory to LDS
        if constexpr(is_same_v<ALayout, tensor_layout::convolution::NHWGK> ||
                     is_same_v<ALayout, tensor_layout::convolution::NDHWGK>)
        {
            if(!(ABlockTransferSrcVectorDim == 2 && ConvK % ABlockTransferSrcScalarPerVector == 0))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "VectorDim is wrong!" << " In " << __FILE__ << ":" << __LINE__
                              << ", in function: " << __func__ << std::endl;
                }

                return false;
            }
        }

        else
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Unsupported A Layout!" << " In " << __FILE__ << ":" << __LINE__
                          << ", in function: " << __func__ << std::endl;
            }

            return false;
        }

        // vector load for B matrix from global memory to LDS
        if constexpr(is_same_v<BLayout, tensor_layout::convolution::GKYXC> ||
                     is_same_v<BLayout, tensor_layout::convolution::GKZYXC>)
        {

            if(!(BBlockTransferSrcVectorDim == 1 && ConvC % BBlockTransferSrcScalarPerVector == 0))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "VectorDim is wrong!" << " In " << __FILE__ << ":" << __LINE__
                              << ", in function: " << __func__ << std::endl;
                }

                return false;
            }
        }
        else
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Unsupported B Layout!" << " In " << __FILE__ << ":" << __LINE__
                          << ", in function: " << __func__ << std::endl;
            }

            return false;
        }

        // vector store for E
        if constexpr(is_same_v<ELayout, tensor_layout::convolution::NHWGC> ||
                     is_same_v<ELayout, tensor_layout::convolution::NDHWGC>)
        {
            // vector store C matrix into global memory
            if(!(ConvC % CShuffleBlockTransferScalarPerVector == 0))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {

                    std::cout << "VectorDim is wrong!" << " In " << __FILE__ << ":" << __LINE__
                              << ", in function: " << __func__ << std::endl;
                }

                return false;
            }
        }
        else
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Unsupported E Layout!" << " In " << __FILE__ << ":" << __LINE__
                          << ", in function: " << __func__ << std::endl;
            }

            return false;
        }

        // Check gridwise gemm validity
        for(std::size_t i = 0; i < arg.a_grid_desc_ak0_m_ak1_container_.size(); i++)
        {
            const index_t GemmM = arg.a_grid_desc_ak0_m_ak1_container_[i].GetLength(I1);
            const index_t GemmN = arg.b_grid_desc_bk0_n_bk1_container_[i].GetLength(I1);
            const index_t GemmK = arg.a_grid_desc_ak0_m_ak1_container_[i].GetLength(I0) *
                                  arg.a_grid_desc_ak0_m_ak1_container_[i].GetLength(I2);
            // Create gemm arguments with dummy values to check for validity
            typename GridwiseGemm::Argument gemm_arg{nullptr, // p_as_grid
                                                     nullptr, // p_bs_grid
                                                     nullptr, // p_e_grid
                                                     GemmM,   // M
                                                     GemmN,   // N
                                                     GemmK,   // K
                                                     I0,      // StrideAs
                                                     I0,      // StrideBs
                                                     I0,      // StrideE
                                                     arg.k_batch_};

            const auto num_k_loop = gemm_arg.AK0 / (KPerBlock / AK1);
            if constexpr(BlkGemmPipelineVer != BlockGemmPipelineVersion::v1)
            {
                if(num_k_loop <= GridwiseGemm::BlockwiseGemmPipe::PrefetchStages)
                {
                    return false;
                }
            }
        }

        if constexpr(!LargeTensors)
        {
            constexpr long_index_t TwoGB = (long_index_t{1} << 31);
            for(std::size_t i = 0; i < arg.a_grid_desc_ak0_m_ak1_container_.size(); i++)
            {
                if(!(arg.a_grid_desc_ak0_m_ak1_container_[i].GetElementSpaceSize() *
                             sizeof(ADataType) <=
                         TwoGB &&
                     arg.e_grid_desc_mblock_mperblock_nblock_nperblock_container_[i]
                                 .GetElementSpaceSize() *
                             sizeof(EDataType) <=
                         TwoGB))
                {
                    if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                    {
                        std::cout << "[NHWGC Layout] One of the tensor is exceeding 2GB "
                                     "memory size!"
                                  << " In " << __FILE__ << ":" << __LINE__
                                  << ", in function: " << __func__ << std::endl;
                    }
                    return false;
                }
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
                 const CDEElementwiseOp& cde_element_op,
                 const ck::index_t split_k = 1)
    {
        if constexpr(!LargeTensors)
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
                            cde_element_op,
                            split_k};
        }
        else
        {
            std::array<long_index_t, NDimSpatial + 3> a_g_n_k_wos_lengths_i64;
            std::array<long_index_t, NDimSpatial + 3> a_g_n_k_wos_strides_i64;
            std::array<long_index_t, NDimSpatial + 3> b_g_k_c_xs_lengths_i64;
            std::array<long_index_t, NDimSpatial + 3> b_g_k_c_xs_strides_i64;
            std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor>
                ds_g_n_c_wis_lengths_i64;
            std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor>
                ds_g_n_c_wis_strides_i64;
            std::array<long_index_t, NDimSpatial + 3> e_g_n_c_wis_lengths_i64;
            std::array<long_index_t, NDimSpatial + 3> e_g_n_c_wis_strides_i64;
            std::array<long_index_t, NDimSpatial> conv_filter_strides_i64;
            std::array<long_index_t, NDimSpatial> conv_filter_dilations_i64;
            std::array<long_index_t, NDimSpatial> input_left_pads_i64;
            std::array<long_index_t, NDimSpatial> input_right_pads_i64;

            array_convert(a_g_n_k_wos_lengths_i64, a_g_n_k_wos_lengths);
            array_convert(a_g_n_k_wos_strides_i64, a_g_n_k_wos_strides);
            array_convert(b_g_k_c_xs_lengths_i64, b_g_k_c_xs_lengths);
            array_convert(b_g_k_c_xs_strides_i64, b_g_k_c_xs_strides);
            for(index_t d = 0; d < NumDTensor; d++)
            {
                array_convert(ds_g_n_c_wis_lengths_i64[d], ds_g_n_c_wis_lengths[d]);
                array_convert(ds_g_n_c_wis_strides_i64[d], ds_g_n_c_wis_strides[d]);
            }
            array_convert(e_g_n_c_wis_lengths_i64, e_g_n_c_wis_lengths);
            array_convert(e_g_n_c_wis_strides_i64, e_g_n_c_wis_strides);
            array_convert(conv_filter_strides_i64, conv_filter_strides);
            array_convert(conv_filter_dilations_i64, conv_filter_dilations);
            array_convert(input_left_pads_i64, input_left_pads);
            array_convert(input_right_pads_i64, input_right_pads);

            return Argument{p_a,
                            p_b,
                            p_ds,
                            p_e,
                            a_g_n_k_wos_lengths_i64,
                            a_g_n_k_wos_strides_i64,
                            b_g_k_c_xs_lengths_i64,
                            b_g_k_c_xs_strides_i64,
                            ds_g_n_c_wis_lengths_i64,
                            ds_g_n_c_wis_strides_i64,
                            e_g_n_c_wis_lengths_i64,
                            e_g_n_c_wis_strides_i64,
                            conv_filter_strides_i64,
                            conv_filter_dilations_i64,
                            input_left_pads_i64,
                            input_right_pads_i64,
                            a_element_op,
                            b_element_op,
                            cde_element_op,
                            split_k};
        }
    }

    static auto MakeArgument(
        const void* p_a,                                                      // output image
        const void* p_b,                                                      // weight
        const std::array<const void*, NumDTensor>& p_ds,                      // bias
        void* p_e,                                                            // input image
        const std::array<long_index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output image
        const std::array<long_index_t, NDimSpatial + 3>& a_g_n_k_wos_strides, // output image
        const std::array<long_index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,  // weight
        const std::array<long_index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,  // weight
        const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor>&
            ds_g_n_c_wis_lengths, // bias
        const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor>&
            ds_g_n_c_wis_strides,                                             // bias
        const std::array<long_index_t, NDimSpatial + 3>& e_g_n_c_wis_lengths, // input image
        const std::array<long_index_t, NDimSpatial + 3>& e_g_n_c_wis_strides, // input image
        const std::array<long_index_t, NDimSpatial>& conv_filter_strides,
        const std::array<long_index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<long_index_t, NDimSpatial>& input_left_pads,
        const std::array<long_index_t, NDimSpatial>& input_right_pads,
        const AElementwiseOp& a_element_op,
        const BElementwiseOp& b_element_op,
        const CDEElementwiseOp& cde_element_op,
        const ck::index_t split_k = 1)
    {
        if constexpr(LargeTensors)
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
                            cde_element_op,
                            split_k};
        }
        else
        {
            bool ds_ovf = false;
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
                ds_ovf |= tensor_exceeds_2gb<DDataType>(ds_g_n_c_wis_lengths[i]);
            });
            const bool stride_ovf = tensor_exceeds_2gb<ADataType>(a_g_n_k_wos_lengths) ||
                                    tensor_exceeds_2gb<BDataType>(b_g_k_c_xs_lengths) ||
                                    tensor_exceeds_2gb<EDataType>(e_g_n_c_wis_lengths) || ds_ovf;

            std::array<index_t, NDimSpatial + 3> a_g_n_k_wos_lengths_i32;
            std::array<index_t, NDimSpatial + 3> a_g_n_k_wos_strides_i32;
            std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_lengths_i32;
            std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_strides_i32;
            std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_c_wis_lengths_i32;
            std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_c_wis_strides_i32;
            std::array<index_t, NDimSpatial + 3> e_g_n_c_wis_lengths_i32;
            std::array<index_t, NDimSpatial + 3> e_g_n_c_wis_strides_i32;
            std::array<index_t, NDimSpatial> conv_filter_strides_i32;
            std::array<index_t, NDimSpatial> conv_filter_dilations_i32;
            std::array<index_t, NDimSpatial> input_left_pads_i32;
            std::array<index_t, NDimSpatial> input_right_pads_i32;

            array_convert(a_g_n_k_wos_lengths_i32, a_g_n_k_wos_lengths);
            array_convert(a_g_n_k_wos_strides_i32, a_g_n_k_wos_strides);
            array_convert(b_g_k_c_xs_lengths_i32, b_g_k_c_xs_lengths);
            array_convert(b_g_k_c_xs_strides_i32, b_g_k_c_xs_strides);
            for(index_t d = 0; d < NumDTensor; d++)
            {
                array_convert(ds_g_n_c_wis_lengths_i32[d], ds_g_n_c_wis_lengths[d]);
                array_convert(ds_g_n_c_wis_strides_i32[d], ds_g_n_c_wis_strides[d]);
            }
            array_convert(e_g_n_c_wis_lengths_i32, e_g_n_c_wis_lengths);
            array_convert(e_g_n_c_wis_strides_i32, e_g_n_c_wis_strides);
            array_convert(conv_filter_strides_i32, conv_filter_strides);
            array_convert(conv_filter_dilations_i32, conv_filter_dilations);
            array_convert(input_left_pads_i32, input_left_pads);
            array_convert(input_right_pads_i32, input_right_pads);

            return Argument{p_a,
                            p_b,
                            p_ds,
                            p_e,
                            a_g_n_k_wos_lengths_i32,
                            a_g_n_k_wos_strides_i32,
                            b_g_k_c_xs_lengths_i32,
                            b_g_k_c_xs_strides_i32,
                            ds_g_n_c_wis_lengths_i32,
                            ds_g_n_c_wis_strides_i32,
                            e_g_n_c_wis_lengths_i32,
                            e_g_n_c_wis_strides_i32,
                            conv_filter_strides_i32,
                            conv_filter_dilations_i32,
                            input_left_pads_i32,
                            input_right_pads_i32,
                            a_element_op,
                            b_element_op,
                            cde_element_op,
                            split_k,
                            stride_ovf};
        }
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
        const CDEElementwiseOp& cde_element_op,
        const ck::index_t split_k = 1) override
    {
        if constexpr(!LargeTensors)
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
                                              cde_element_op,
                                              split_k);
        }
        else
        {
            std::array<long_index_t, NDimSpatial + 3> a_g_n_k_wos_lengths_i64;
            std::array<long_index_t, NDimSpatial + 3> a_g_n_k_wos_strides_i64;
            std::array<long_index_t, NDimSpatial + 3> b_g_k_c_xs_lengths_i64;
            std::array<long_index_t, NDimSpatial + 3> b_g_k_c_xs_strides_i64;
            std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor>
                ds_g_n_c_wis_lengths_i64;
            std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor>
                ds_g_n_c_wis_strides_i64;
            std::array<long_index_t, NDimSpatial + 3> e_g_n_c_wis_lengths_i64;
            std::array<long_index_t, NDimSpatial + 3> e_g_n_c_wis_strides_i64;
            std::array<long_index_t, NDimSpatial> conv_filter_strides_i64;
            std::array<long_index_t, NDimSpatial> conv_filter_dilations_i64;
            std::array<long_index_t, NDimSpatial> input_left_pads_i64;
            std::array<long_index_t, NDimSpatial> input_right_pads_i64;

            array_convert(a_g_n_k_wos_lengths_i64, a_g_n_k_wos_lengths);
            array_convert(a_g_n_k_wos_strides_i64, a_g_n_k_wos_strides);
            array_convert(b_g_k_c_xs_lengths_i64, b_g_k_c_xs_lengths);
            array_convert(b_g_k_c_xs_strides_i64, b_g_k_c_xs_strides);
            for(index_t d = 0; d < NumDTensor; d++)
            {
                array_convert(ds_g_n_c_wis_lengths_i64[d], ds_g_n_c_wis_lengths[d]);
                array_convert(ds_g_n_c_wis_strides_i64[d], ds_g_n_c_wis_strides[d]);
            }
            array_convert(e_g_n_c_wis_lengths_i64, e_g_n_c_wis_lengths);
            array_convert(e_g_n_c_wis_strides_i64, e_g_n_c_wis_strides);
            array_convert(conv_filter_strides_i64, conv_filter_strides);
            array_convert(conv_filter_dilations_i64, conv_filter_dilations);
            array_convert(input_left_pads_i64, input_left_pads);
            array_convert(input_right_pads_i64, input_right_pads);

            return std::make_unique<Argument>(p_a,
                                              p_b,
                                              p_ds,
                                              p_e,
                                              a_g_n_k_wos_lengths_i64,
                                              a_g_n_k_wos_strides_i64,
                                              b_g_k_c_xs_lengths_i64,
                                              b_g_k_c_xs_strides_i64,
                                              ds_g_n_c_wis_lengths_i64,
                                              ds_g_n_c_wis_strides_i64,
                                              e_g_n_c_wis_lengths_i64,
                                              e_g_n_c_wis_strides_i64,
                                              conv_filter_strides_i64,
                                              conv_filter_dilations_i64,
                                              input_left_pads_i64,
                                              input_right_pads_i64,
                                              a_element_op,
                                              b_element_op,
                                              cde_element_op,
                                              split_k);
        }
    }

    std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const void* p_a,                                                      // output image
        const void* p_b,                                                      // weight
        const std::array<const void*, NumDTensor>& p_ds,                      // bias
        void* p_e,                                                            // input image
        const std::array<long_index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output image
        const std::array<long_index_t, NDimSpatial + 3>& a_g_n_k_wos_strides, // output image
        const std::array<long_index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,  // weight
        const std::array<long_index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,  // weight
        const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor>&
            ds_g_n_c_wis_lengths, // bias
        const std::array<std::array<long_index_t, NDimSpatial + 3>, NumDTensor>&
            ds_g_n_c_wis_strides,                                             // bias
        const std::array<long_index_t, NDimSpatial + 3>& e_g_n_c_wis_lengths, // input image
        const std::array<long_index_t, NDimSpatial + 3>& e_g_n_c_wis_strides, // input image
        const std::array<long_index_t, NDimSpatial>& conv_filter_strides,
        const std::array<long_index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<long_index_t, NDimSpatial>& input_left_pads,
        const std::array<long_index_t, NDimSpatial>& input_right_pads,
        const AElementwiseOp& a_element_op,
        const BElementwiseOp& b_element_op,
        const CDEElementwiseOp& cde_element_op,
        const ck::index_t split_k = 1) override
    {
        if constexpr(LargeTensors)
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
                                              cde_element_op,
                                              split_k);
        }
        else
        {
            bool ds_ovf = false;
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
                ds_ovf |= tensor_exceeds_2gb<DDataType>(ds_g_n_c_wis_lengths[i]);
            });
            const bool stride_ovf = tensor_exceeds_2gb<ADataType>(a_g_n_k_wos_lengths) ||
                                    tensor_exceeds_2gb<BDataType>(b_g_k_c_xs_lengths) ||
                                    tensor_exceeds_2gb<EDataType>(e_g_n_c_wis_lengths) || ds_ovf;

            std::array<index_t, NDimSpatial + 3> a_g_n_k_wos_lengths_i32;
            std::array<index_t, NDimSpatial + 3> a_g_n_k_wos_strides_i32;
            std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_lengths_i32;
            std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_strides_i32;
            std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_c_wis_lengths_i32;
            std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_c_wis_strides_i32;
            std::array<index_t, NDimSpatial + 3> e_g_n_c_wis_lengths_i32;
            std::array<index_t, NDimSpatial + 3> e_g_n_c_wis_strides_i32;
            std::array<index_t, NDimSpatial> conv_filter_strides_i32;
            std::array<index_t, NDimSpatial> conv_filter_dilations_i32;
            std::array<index_t, NDimSpatial> input_left_pads_i32;
            std::array<index_t, NDimSpatial> input_right_pads_i32;

            array_convert(a_g_n_k_wos_lengths_i32, a_g_n_k_wos_lengths);
            array_convert(a_g_n_k_wos_strides_i32, a_g_n_k_wos_strides);
            array_convert(b_g_k_c_xs_lengths_i32, b_g_k_c_xs_lengths);
            array_convert(b_g_k_c_xs_strides_i32, b_g_k_c_xs_strides);
            for(index_t d = 0; d < NumDTensor; d++)
            {
                array_convert(ds_g_n_c_wis_lengths_i32[d], ds_g_n_c_wis_lengths[d]);
                array_convert(ds_g_n_c_wis_strides_i32[d], ds_g_n_c_wis_strides[d]);
            }
            array_convert(e_g_n_c_wis_lengths_i32, e_g_n_c_wis_lengths);
            array_convert(e_g_n_c_wis_strides_i32, e_g_n_c_wis_strides);
            array_convert(conv_filter_strides_i32, conv_filter_strides);
            array_convert(conv_filter_dilations_i32, conv_filter_dilations);
            array_convert(input_left_pads_i32, input_left_pads);
            array_convert(input_right_pads_i32, input_right_pads);

            return std::make_unique<Argument>(p_a,
                                              p_b,
                                              p_ds,
                                              p_e,
                                              a_g_n_k_wos_lengths_i32,
                                              a_g_n_k_wos_strides_i32,
                                              b_g_k_c_xs_lengths_i32,
                                              b_g_k_c_xs_strides_i32,
                                              ds_g_n_c_wis_lengths_i32,
                                              ds_g_n_c_wis_strides_i32,
                                              e_g_n_c_wis_lengths_i32,
                                              e_g_n_c_wis_strides_i32,
                                              conv_filter_strides_i32,
                                              conv_filter_dilations_i32,
                                              input_left_pads_i32,
                                              input_right_pads_i32,
                                              a_element_op,
                                              b_element_op,
                                              cde_element_op,
                                              split_k,
                                              stride_ovf);
        }
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffleV3"
            << (DirectLoad ? "_DirectLoad" : "")
            << (LargeTensors ? "_Large_Tensor" : "")
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1 << ", "
            << getConvBackwardDataSpecializationString(ConvBackwardDataSpecialization) << ", "
            << MPerXdl << ", "
            << NPerXdl << ", "
            << MRepeat << ", "
            << NRepeat << ", "
            << ABlockTransferSrcScalarPerVector << ", "
            << BBlockTransferSrcScalarPerVector << ", "
            << CShuffleMRepeatPerShuffle << ", "
            << CShuffleNRepeatPerShuffle;
                
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
                "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffleV3::Argument structure!");
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
                "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffleV3::Argument structure!");
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
