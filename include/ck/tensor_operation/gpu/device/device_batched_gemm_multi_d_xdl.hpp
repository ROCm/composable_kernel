// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_multi_d.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/device_utility/device_prop.hpp"
#include "ck/device_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

/*
 * \brief Wrapper function of GridwiseGemm::Run to realize BatchedGEMM.
 *
 * \tparam ComputePtrOffsetOfBatch Class that computes the base pointer offsets of A, B, C matrix
 * given the batch. For example, ComputePtrOffsetOfStridedBatch() computes the offsets of evenly
 * strided batched, but we can easily extend to other layouts. The returned offset can be either \p
 * index_t or \p long_index_t. If it returns \p long_index_t, we are not subject to the 2GB
 * limitations.
 *
 * \tparam Block2CTileMap Block2CTileMap::CalculateBottomIndex() takes in id of a workgroup and
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
 * \note \p Block2CTileMap allows customized mapping between a workgroup and the C-tile it computes.
 * Together with \p ComputePtrOffsetOfBatch, we can reuse GridwiseGemm (and GridwiseGemm fusion ) to
 * realize BatchedGemm and GroupedGemm (and the corresponding GEMM fusion).
 *
 */
template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatDsPointer,
          typename FloatC,
          typename AGridDesc_AK0_M_AK1,
          typename BGridDesc_BK0_N_BK1,
          typename DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename ComputePtrOffsetOfBatch,
          typename Block2CTileMap,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_batched_gemm_xdl(const FloatAB* __restrict__ p_a_grid,
                                const FloatAB* __restrict__ p_b_grid,
                                FloatDsPointer p_ds_grid,
                                FloatC* __restrict__ p_e_grid,
                                const index_t batch_count,
                                const AGridDesc_AK0_M_AK1 a_grid_desc_k0_m_k1,
                                const BGridDesc_BK0_N_BK1 b_grid_desc_k0_n_k1,
                                const DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                                    ds_grid_desc_mblock_mperblock_nblock_nperblock,
                                const EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
                                    e_grid_desc_mblock_mperblock_nblock_nperblock_,
                                const AElementwiseOperation a_element_op,
                                const BElementwiseOperation b_element_op,
                                const CDEElementwiseOperation cde_element_op,
                                const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch,
                                const Block2CTileMap block_2_ctile_map)
{

#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__))
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    const long_index_t a_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx)));
    const long_index_t b_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx)));
    const long_index_t e_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetEPtrOffset(g_idx)));

    const auto ds_batch_offset = compute_ptr_offset_of_batch.GetDsPtrOffset(g_idx);

    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    FloatDsPointer p_ds_grid_grp;

    static constexpr index_t NumDTensor =
        DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock::Size();

    static_for<0, NumDTensor, 1>{}(
        [&](auto i) { p_ds_grid_grp(i) = p_ds_grid[i] + ds_batch_offset[i]; });

    GridwiseGemm::template Run<HasMainKBlockLoop>(p_a_grid + a_batch_offset,
                                                  p_b_grid + b_batch_offset,
                                                  p_ds_grid_grp,
                                                  p_e_grid + e_batch_offset,
                                                  p_shared,
                                                  a_element_op,
                                                  b_element_op,
                                                  cde_element_op,
                                                  a_grid_desc_k0_m_k1,
                                                  b_grid_desc_k0_n_k1,
                                                  ds_grid_desc_mblock_mperblock_nblock_nperblock,
                                                  e_grid_desc_mblock_mperblock_nblock_nperblock_,
                                                  block_2_ctile_map);
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_ds_grid;
    ignore = p_e_grid;
    ignore = batch_count;
    ignore = a_grid_desc_k0_m_k1;
    ignore = b_grid_desc_k0_n_k1;
    ignore = ds_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = e_grid_desc_mblock_mperblock_nblock_nperblock_;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = cde_element_op;
    ignore = compute_ptr_offset_of_batch;
    ignore = block_2_ctile_map;
#endif
}

template <typename ALayout,
          typename BLayout,
          typename DELayout,
          typename ADataType,
          typename BDataType,
          typename GemmAccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          GemmSpecialization GemmSpec,
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
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEBlockTransferScalarPerVector_NPerBlock,
          LoopScheduler LoopSched = make_default_loop_scheduler()>
struct DeviceBatchedGemmMultiDXdl : public DeviceBatchedGemmMultiD<ALayout,
                                                                   BLayout,
                                                                   DELayout,
                                                                   ADataType,
                                                                   BDataType,
                                                                   DsDataType,
                                                                   EDataType,
                                                                   AElementwiseOperation,
                                                                   BElementwiseOperation,
                                                                   CDEElementwiseOperation>
{
    using DeviceOp = DeviceBatchedGemmMultiDXdl;

    static constexpr index_t NumDTensor = DsDataType::Size();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    static auto MakeAGridDescriptor_AK0_M_AK1(index_t MRaw, index_t KRaw, index_t StrideA)
    {
        const auto a_grid_desc_mraw_kraw = [&]() {
            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, KRaw),
                                                    make_tuple(StrideA, I1));
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, KRaw),
                                                    make_tuple(I1, StrideA));
            }
        }();

        const auto M = math::integer_divide_ceil(MRaw, MPerBlock) * MPerBlock;
        const auto K = math::integer_divide_ceil(KRaw, KPerBlock) * KPerBlock;

        const auto MPad = M - MRaw;
        const auto KPad = K - KRaw;

        if constexpr(GemmSpec == GemmSpecialization::MKPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad both M and K
            assert(K % AK1 == 0);

            const auto AK0 = K / AK1;

            const auto a_grid_desc_m_k =
                transform_tensor_descriptor(a_grid_desc_mraw_kraw,
                                            make_tuple(make_right_pad_transform(MRaw, MPad),
                                                       make_right_pad_transform(KRaw, KPad)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto a_grid_desc_ak0_m_ak1 =
                transform_tensor_descriptor(a_grid_desc_m_k,
                                            make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                                                       make_pass_through_transform(M)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::MPadding ||
                          GemmSpec == GemmSpecialization::MNPadding)
        {
            // pad M, but not K
            assert(KRaw % AK1 == 0);

            const auto AK0 = KRaw / AK1;

            const auto a_grid_desc_ak0_m_ak1 =
                transform_tensor_descriptor(a_grid_desc_mraw_kraw,
                                            make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                                                       make_right_pad_transform(MRaw, MPad)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::KPadding ||
                          GemmSpec == GemmSpecialization::NKPadding)
        {
            // pad K, but not M
            assert(K % AK1 == 0);

            const auto AK0 = K / AK1;

            const auto a_grid_desc_m_k = transform_tensor_descriptor(
                a_grid_desc_mraw_kraw,
                make_tuple(make_pass_through_transform(MRaw), make_right_pad_transform(KRaw, KPad)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto a_grid_desc_ak0_m_ak1 =
                transform_tensor_descriptor(a_grid_desc_m_k,
                                            make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                                                       make_pass_through_transform(MRaw)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
        else
        {
            // not pad M or K
            assert(KRaw % AK1 == 0);

            const auto AK0 = KRaw / AK1;

            const auto a_grid_desc_ak0_m_ak1 =
                transform_tensor_descriptor(a_grid_desc_mraw_kraw,
                                            make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                                                       make_pass_through_transform(MRaw)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
    }

    static auto MakeBGridDescriptor_BK0_N_BK1(index_t KRaw, index_t NRaw, index_t StrideB)
    {
        const auto b_grid_desc_nraw_kraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(NRaw, KRaw),
                                                    make_tuple(I1, StrideB));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(NRaw, KRaw),
                                                    make_tuple(StrideB, I1));
            }
        }();

        const auto N = math::integer_divide_ceil(NRaw, NPerBlock) * NPerBlock;
        const auto K = math::integer_divide_ceil(KRaw, KPerBlock) * KPerBlock;

        const auto NPad = N - NRaw;
        const auto KPad = K - KRaw;

        if constexpr(GemmSpec == GemmSpecialization::NKPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad both N and K
            assert(K % BK1 == 0);

            const auto BK0 = K / BK1;

            const auto b_grid_desc_n_k =
                transform_tensor_descriptor(b_grid_desc_nraw_kraw,
                                            make_tuple(make_right_pad_transform(NRaw, NPad),
                                                       make_right_pad_transform(KRaw, KPad)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto b_grid_desc_bk0_n_bk1 =
                transform_tensor_descriptor(b_grid_desc_n_k,
                                            make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                       make_pass_through_transform(N)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::NPadding ||
                          GemmSpec == GemmSpecialization::MNPadding)
        {
            // pad N, but not K
            assert(KRaw % BK1 == 0);

            const auto BK0 = KRaw / BK1;

            const auto b_grid_desc_bk0_n_bk1 =
                transform_tensor_descriptor(b_grid_desc_nraw_kraw,
                                            make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                       make_right_pad_transform(NRaw, NPad)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::KPadding ||
                          GemmSpec == GemmSpecialization::MKPadding)
        {
            // pad K, but not N
            assert(K % BK1 == 0);

            const auto BK0 = K / BK1;

            const auto b_grid_desc_n_k = transform_tensor_descriptor(
                b_grid_desc_nraw_kraw,
                make_tuple(make_pass_through_transform(NRaw), make_right_pad_transform(KRaw, KPad)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto b_grid_desc_bk0_n_bk1 =
                transform_tensor_descriptor(b_grid_desc_n_k,
                                            make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                       make_pass_through_transform(NRaw)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
        else
        {
            // not pad N or K
            assert(KRaw % BK1 == 0);

            const auto BK0 = KRaw / BK1;

            const auto b_grid_desc_bk0_n_bk1 =
                transform_tensor_descriptor(b_grid_desc_nraw_kraw,
                                            make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                       make_pass_through_transform(NRaw)),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
    }

    static auto MakeEGridDescriptor_M_N(index_t MRaw, index_t NRaw, index_t StrideE)
    {
        const auto c_grid_desc_mraw_nraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, DELayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(StrideE, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, DELayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(I1, StrideE));
            }
        }();

        const auto M = math::integer_divide_ceil(MRaw, MPerBlock) * MPerBlock;
        const auto N = math::integer_divide_ceil(NRaw, NPerBlock) * NPerBlock;

        const auto MPad = M - MRaw;
        const auto NPad = N - NRaw;

        if constexpr(GemmSpec == GemmSpecialization::MNPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad M and N
            return transform_tensor_descriptor(c_grid_desc_mraw_nraw,
                                               make_tuple(make_right_pad_transform(MRaw, MPad),
                                                          make_right_pad_transform(NRaw, NPad)),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else if constexpr(GemmSpec == GemmSpecialization::MPadding ||
                          GemmSpec == GemmSpecialization::MKPadding)
        {
            // pad M, but not N
            return transform_tensor_descriptor(
                c_grid_desc_mraw_nraw,
                make_tuple(make_right_pad_transform(MRaw, MPad), make_pass_through_transform(NRaw)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else if constexpr(GemmSpec == GemmSpecialization::NPadding ||
                          GemmSpec == GemmSpecialization::NKPadding)
        {
            // pad N, but not M
            return transform_tensor_descriptor(
                c_grid_desc_mraw_nraw,
                make_tuple(make_pass_through_transform(MRaw), make_right_pad_transform(NRaw, NPad)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else
        {
            // not pad M or N
            return c_grid_desc_mraw_nraw;
        }
    }

    using AGridDesc_AK0_M_AK1 = decltype(MakeAGridDescriptor_AK0_M_AK1(1, 1, 1));
    using BGridDesc_BK0_N_BK1 = decltype(MakeBGridDescriptor_BK0_N_BK1(1, 1, 1));
    using EGridDesc_M_N       = decltype(MakeEGridDescriptor_M_N(1, 1, 1));

    struct ComputePtrOffsetOfStridedBatch
    {
        ComputePtrOffsetOfStridedBatch(index_t BatchStrideA,
                                       index_t BatchStrideB,
                                       std::array<ck::index_t, NumDTensor> BatchStrideDs,
                                       index_t BatchStrideE)
            : BatchStrideA_(BatchStrideA),
              BatchStrideB_(BatchStrideB),
              BatchStrideDs_(BatchStrideDs),
              BatchStrideE_(BatchStrideE)
        {
        }

        __host__ __device__ constexpr long_index_t GetAPtrOffset(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideA_);
        }

        __host__ __device__ constexpr long_index_t GetBPtrOffset(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideB_);
        }

        __host__ __device__ constexpr auto GetDsPtrOffset(index_t g_idx) const
        {
            std::array<long_index_t, NumDTensor> ds_offset;
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                ds_offset[i] = g_idx * static_cast<long_index_t>(BatchStrideDs_[i]);
            });
            return ds_offset;
        }

        __host__ __device__ constexpr long_index_t GetEPtrOffset(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideE_);
        }

        private:
        index_t BatchStrideA_;
        index_t BatchStrideB_;
        std::array<ck::index_t, NumDTensor> BatchStrideDs_;
        index_t BatchStrideE_;
    };

    using GridwiseGemm = GridwiseGemmMultipleD_k0mk1_k0nk1_mn_xdl_cshuffle<
        ADataType, // TODO: distinguish A/B datatype
        GemmAccDataType,
        CShuffleDataType,
        DsDataType,
        EDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
        InMemoryDataOperationEnum::Set,
        AGridDesc_AK0_M_AK1,
        BGridDesc_BK0_N_BK1,
        EGridDesc_M_N,
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
        LoopSched>;

    using CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock = decltype(
        GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(EGridDesc_M_N{}));
    using Block2CTileMap = typename GridwiseGemm::DefaultBlock2ETileMap;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const void* p_a_grid,
                 const void* p_b_grid,
                 std::array<const void*, NumDTensor> p_ds_grid,
                 void* p_e_grid,
                 index_t M,
                 index_t N,
                 index_t K,
                 index_t StrideA,
                 index_t StrideB,
                 std::array<ck::index_t, NumDTensor> StrideDs,
                 index_t StrideE,
                 index_t BatchStrideA,
                 index_t BatchStrideB,
                 std::array<ck::index_t, NumDTensor> BatchStrideDs,
                 index_t BatchStrideE,
                 index_t Batch,
                 index_t M01,
                 index_t N01,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation cde_element_op)
            : p_a_grid_{static_cast<const ADataType*>(p_a_grid)},
              p_b_grid_{static_cast<const BDataType*>(p_b_grid)},
              p_ds_grid_{}, // FIXME
              p_e_grid_{static_cast<EDataType*>(p_e_grid)},
              Batch_(Batch),
              a_grid_desc_ak0_m_ak1_{
                  DeviceBatchedGemmMultiDXdl::MakeAGridDescriptor_AK0_M_AK1(M, K, StrideA)},
              b_grid_desc_bk0_n_bk1_{
                  DeviceBatchedGemmMultiDXdl::MakeBGridDescriptor_BK0_N_BK1(K, N, StrideB)},
              ds_grid_desc_mblock_mperblock_nblock_nperblock_{},
              e_grid_desc_m_n_{DeviceBatchedGemmMultiDXdl::MakeEGridDescriptor_M_N(M, N, StrideE)},
              e_grid_desc_mblock_mperblock_nblock_nperblock_{},
              compute_ptr_offset_of_batch_{BatchStrideA, BatchStrideB, BatchStrideDs, BatchStrideE},
              block_2_ctile_map_{GridwiseGemm::MakeDefaultBlock2ETileMap(e_grid_desc_m_n_)},
              M01_{M01},
              N01_{N01},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op}
        {
            if(GridwiseGemm::CheckValidity(a_grid_desc_ak0_m_ak1_,
                                           b_grid_desc_bk0_n_bk1_,
                                           e_grid_desc_m_n_,
                                           block_2_ctile_map_))
            {
                e_grid_desc_mblock_mperblock_nblock_nperblock_ =
                    GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        e_grid_desc_m_n_);

                static_for<0, NumDTensor, 1>{}([&](auto i) {
                    using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                    p_ds_grid_(i) = static_cast<const DDataType*>(p_ds_grid[i]);

                    const auto d_grid_desc_m_n =
                        DeviceOp::MakeEGridDescriptor_M_N(M, N, StrideDs[i]);

                    ds_grid_desc_mblock_mperblock_nblock_nperblock_(i) =
                        GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                            d_grid_desc_m_n);
                });
            }
        }

        //  private:
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        typename GridwiseGemm::DsGridPointer p_ds_grid_;
        EDataType* p_e_grid_;
        index_t Batch_;

        AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1_;
        BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1_;
        StaticallyIndexedArray<
            typename GridwiseGemm::EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
            NumDTensor>
            ds_grid_desc_mblock_mperblock_nblock_nperblock_; // FIXME: Ds desc may be of different
                                                             // type from E
        EGridDesc_M_N e_grid_desc_m_n_;
        typename GridwiseGemm::EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            e_grid_desc_mblock_mperblock_nblock_nperblock_;

        ComputePtrOffsetOfStridedBatch compute_ptr_offset_of_batch_;
        Block2CTileMap block_2_ctile_map_;
        index_t M01_;
        index_t N01_;
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceBatchedGemmMultiDXdl::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            {
                std::cout << "arg.a_grid_desc_ak0_m_ak1_{"
                          << arg.a_grid_desc_ak0_m_ak1_.GetLength(I0) << ", "
                          << arg.a_grid_desc_ak0_m_ak1_.GetLength(I1) << ", "
                          << arg.a_grid_desc_ak0_m_ak1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.b_grid_desc_bk0_n_bk1_{"
                          << arg.b_grid_desc_bk0_n_bk1_.GetLength(I0) << ", "
                          << arg.b_grid_desc_bk0_n_bk1_.GetLength(I1) << ", "
                          << arg.b_grid_desc_bk0_n_bk1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.e_grid_desc_m_n_{" << arg.e_grid_desc_m_n_.GetLength(I0) << ", "
                          << arg.e_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;
            }

            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_ak0_m_ak1_,
                                            arg.b_grid_desc_bk0_n_bk1_,
                                            arg.e_grid_desc_m_n_,
                                            arg.block_2_ctile_map_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseBatchedGemm_km_kn_m0m1n0n1_xdlops_v2r3 has invalid setting");
            }

            const index_t grid_size =
                arg.block_2_ctile_map_.CalculateGridSize(arg.e_grid_desc_m_n_) * arg.Batch_;

            const auto K =
                arg.a_grid_desc_ak0_m_ak1_.GetLength(I0) * arg.a_grid_desc_ak0_m_ak1_.GetLength(I2);

            auto launch_kernel = [&](auto has_main_k_block_loop) {
                constexpr bool has_main_loop = has_main_k_block_loop.value;

                const auto kernel = kernel_batched_gemm_xdl<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    typename GridwiseGemm::DsGridPointer,
                    EDataType,
                    DeviceOp::AGridDesc_AK0_M_AK1,
                    DeviceOp::BGridDesc_BK0_N_BK1,
                    ck::StaticallyIndexedArray<
                        typename GridwiseGemm::EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                        NumDTensor>,
                    typename GridwiseGemm::EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CDEElementwiseOperation,
                    ComputePtrOffsetOfStridedBatch,
                    remove_reference_t<Block2CTileMap>,
                    has_main_loop>;

                return launch_and_time_kernel(stream_config,
                                              kernel,
                                              dim3(grid_size),
                                              dim3(BlockSize),
                                              0,
                                              arg.p_a_grid_,
                                              arg.p_b_grid_,
                                              arg.p_ds_grid_,
                                              arg.p_e_grid_,
                                              arg.Batch_,
                                              arg.a_grid_desc_ak0_m_ak1_,
                                              arg.b_grid_desc_bk0_n_bk1_,
                                              arg.ds_grid_desc_mblock_mperblock_nblock_nperblock_,
                                              arg.e_grid_desc_mblock_mperblock_nblock_nperblock_,
                                              arg.a_element_op_,
                                              arg.b_element_op_,
                                              arg.cde_element_op_,
                                              arg.compute_ptr_offset_of_batch_,
                                              arg.block_2_ctile_map_);
            };

            float ave_time = 0;

            if(GridwiseGemm::CalculateHasMainKBlockLoop(K))
            {
                ave_time = launch_kernel(integral_constant<bool, true>{});
            }
            else
            {
                ave_time = launch_kernel(integral_constant<bool, false>{});
            }

            return ave_time;
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        return GridwiseGemm::CheckValidity(arg.a_grid_desc_ak0_m_ak1_,
                                           arg.b_grid_desc_bk0_n_bk1_,
                                           arg.e_grid_desc_m_n_,
                                           arg.block_2_ctile_map_);
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const void* p_a,
                             const void* p_b,
                             std::array<const void*, NumDTensor> p_ds,
                             void* p_c,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t StrideA,
                             index_t StrideB,
                             std::array<index_t, NumDTensor> StrideDs,
                             index_t StrideE,
                             index_t BatchStrideA,
                             index_t BatchStrideB,
                             std::array<ck::index_t, NumDTensor> BatchStrideDs,
                             index_t BatchStrideE,
                             index_t Batch,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CDEElementwiseOperation cde_element_op)
    {
        return Argument{p_a,
                        p_b,
                        p_ds,
                        p_c,
                        M,
                        N,
                        K,
                        StrideA,
                        StrideB,
                        StrideDs,
                        StrideE,
                        BatchStrideA,
                        BatchStrideB,
                        BatchStrideDs,
                        BatchStrideE,
                        Batch,
                        1,
                        1,
                        a_element_op,
                        b_element_op,
                        cde_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        std::array<const void*, NumDTensor> p_ds,
                        void* p_c,
                        index_t M,
                        index_t N,
                        index_t K,
                        index_t StrideA,
                        index_t StrideB,
                        std::array<ck::index_t, NumDTensor> StrideDs,
                        index_t StrideE,
                        index_t BatchStrideA,
                        index_t BatchStrideB,
                        std::array<ck::index_t, NumDTensor> BatchStrideDs,
                        index_t BatchStrideE,
                        index_t Batch,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op) override
    {
        return std::make_unique<Argument>(p_a,
                                          p_b,
                                          p_ds,
                                          p_c,
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideDs,
                                          StrideE,
                                          BatchStrideA,
                                          BatchStrideB,
                                          BatchStrideDs,
                                          BatchStrideE,
                                          Batch,
                                          1,
                                          1,
                                          a_element_op,
                                          b_element_op,
                                          cde_element_op);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceBatchedGemmMultiDXdl"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock
            << AK1 << ", "
            << BK1 << ", "
            << getGemmSpecializationString(GemmSpec)
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
