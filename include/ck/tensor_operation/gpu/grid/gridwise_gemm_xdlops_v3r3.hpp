// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v6r3.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_common.hpp"

namespace ck {

template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename AGridDesc_K0_M_K1,
          typename BGridDesc_K0_N_K1,
          typename CGridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl,
          typename C0GridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl,
          typename C1GridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename Block2CTileMap,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(GridwiseGemm::MaxBlockSize, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_gemm_xdlops_v3r3(
        const FloatAB* __restrict__ p_a_grid,
        const FloatAB* __restrict__ p_b_grid,
        FloatC* __restrict__ p_c_grid,
        const FloatC* __restrict__ p_c0_grid,
        const FloatC* __restrict__ p_c1_grid,
        const AGridDesc_K0_M_K1 a_grid_desc_k0_m_k1,
        const BGridDesc_K0_N_K1 b_grid_desc_k0_n_k1,
        const CGridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl
            c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
        const C0GridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl
            c0_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
        const C1GridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl
            c1_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
        const AElementwiseOperation a_element_op,
        const BElementwiseOperation b_element_op,
        const CElementwiseOperation c_element_op,
        const Block2CTileMap block_2_ctile_map)
{
#if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx94__) || defined(__gfx11__) || \
    defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<>())
    {
        __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte(get_device_arch())];

        GridwiseGemm::template Run<HasMainKBlockLoop>(
            p_a_grid,
            p_b_grid,
            p_c_grid,
            p_c0_grid,
            p_c1_grid,
            p_shared,
            a_grid_desc_k0_m_k1,
            b_grid_desc_k0_n_k1,
            c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
            c0_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
            c1_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
            a_element_op,
            b_element_op,
            c_element_op,
            block_2_ctile_map);
    }
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_c_grid;
    ignore = p_c0_grid;
    ignore = p_c1_grid;
    ignore = a_grid_desc_k0_m_k1;
    ignore = b_grid_desc_k0_n_k1;
    ignore = c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl;
    ignore = c0_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl;
    ignore = c1_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = c_element_op;
    ignore = block_2_ctile_map;
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
}

template <
    index_t BlockSize,
    typename FloatAB,
    typename FloatAcc,
    typename FloatC,
    InMemoryDataOperationEnum CGlobalMemoryDataOperation,
    typename AGridDesc_K0_M_K1,
    typename BGridDesc_K0_N_K1,
    typename CGridDesc_M_N,
    typename C0GridDesc_M_N,
    typename C1GridDesc_M_N,
    typename AElementwiseOperation,
    typename BElementwiseOperation,
    typename CElementwiseOperation,
    index_t MPerBlock,
    index_t NPerBlock,
    index_t K0PerBlock,
    index_t MPerXdl,
    index_t NPerXdl,
    index_t K1Value,
    index_t MXdlPerWave,
    index_t NXdlPerWave,
    typename ABlockTransferThreadClusterLengths_K0_M_K1,
    typename ABlockTransferThreadClusterArrangeOrder,
    typename ABlockTransferSrcAccessOrder,
    index_t ABlockTransferSrcVectorDim,
    index_t ABlockTransferSrcScalarPerVector,
    index_t ABlockTransferDstScalarPerVector_K1,
    bool AThreadTransferSrcResetCoordinateAfterRun,
    bool ABlockLdsExtraM,
    typename BBlockTransferThreadClusterLengths_K0_N_K1,
    typename BBlockTransferThreadClusterArrangeOrder,
    typename BBlockTransferSrcAccessOrder,
    index_t BBlockTransferSrcVectorDim,
    index_t BBlockTransferSrcScalarPerVector,
    index_t BBlockTransferDstScalarPerVector_K1,
    bool BThreadTransferSrcResetCoordinateAfterRun,
    bool BBlockLdsExtraN,
    index_t CShuffleMXdlPerWavePerShuffle,
    index_t CShuffleNXdlPerWavePerShuffle,
    typename CBlockTransferClusterLengths_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl,
    index_t CBlockTransferScalarPerVector_NWaveNPerXdl,
    index_t NumGemmKPrefetchStage = 1,
    PipelineVersion PipelineVer   = PipelineVersion::v1>
struct GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v3r3
    : public GridwiseGemm_xdl_cshuffle_base<
          tensor_layout::gemm::RowMajor,
          tensor_layout::gemm::ColumnMajor,
          tensor_layout::gemm::RowMajor,
          FloatAB,
          FloatAB,
          FloatAcc,
          FloatC,
          Tuple<>,
          FloatC,
          AElementwiseOperation,
          BElementwiseOperation,
          BlockSize,
          MPerBlock,
          NPerBlock,
          K0PerBlock * K1Value,
          K1Value,
          K1Value,
          MPerXdl,
          NPerXdl,
          MXdlPerWave,
          NXdlPerWave,
          ABlockTransferThreadClusterLengths_K0_M_K1,
          ABlockTransferThreadClusterArrangeOrder,
          ABlockTransferSrcAccessOrder,
          ABlockTransferSrcVectorDim,
          ABlockTransferSrcScalarPerVector,
          ABlockTransferDstScalarPerVector_K1,
          AThreadTransferSrcResetCoordinateAfterRun,
          ABlockLdsExtraM,
          BBlockTransferThreadClusterLengths_K0_N_K1,
          BBlockTransferThreadClusterArrangeOrder,
          BBlockTransferSrcAccessOrder,
          BBlockTransferSrcVectorDim,
          BBlockTransferSrcScalarPerVector,
          BBlockTransferDstScalarPerVector_K1,
          BThreadTransferSrcResetCoordinateAfterRun,
          BBlockLdsExtraN,
          CShuffleMXdlPerWavePerShuffle,
          CShuffleNXdlPerWavePerShuffle,
          CBlockTransferClusterLengths_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl,
          Sequence<CBlockTransferScalarPerVector_NWaveNPerXdl>,
          FloatAB,
          FloatAB,
          true> // ForceNaiveLayout
{
    using Base = GridwiseGemm_xdl_cshuffle_base<
        tensor_layout::gemm::RowMajor,
        tensor_layout::gemm::ColumnMajor,
        tensor_layout::gemm::RowMajor,
        FloatAB,
        FloatAB,
        FloatAcc,
        FloatC,
        Tuple<>,
        FloatC,
        AElementwiseOperation,
        BElementwiseOperation,
        BlockSize,
        MPerBlock,
        NPerBlock,
        K0PerBlock * K1Value,
        K1Value,
        K1Value,
        MPerXdl,
        NPerXdl,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CBlockTransferClusterLengths_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl,
        Sequence<CBlockTransferScalarPerVector_NWaveNPerXdl>,
        FloatAB,
        FloatAB,
        true>; // ForceNaiveLayout

    using Base::I0;
    using Base::I1;
    using Base::I2;
    using Base::I3;
    using ThisThreadBlock = typename Base::ThisThreadBlock;
    using Base::GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1;
    using Base::GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1;

    // K1 should be Number<...>
    static constexpr auto AK0 = Base::AK0Number;
    static constexpr auto BK0 = Base::BK0Number;
    static constexpr auto AK1 = Base::AK1Number;
    static constexpr auto BK1 = Base::BK1Number;
    // K1 should be Number<...>
    static constexpr auto K1 = Base::AK1Number;

    using GridwiseGemmPipe = remove_cvref_t<
        decltype(GridwiseGemmPipeline_Selector<PipelineVer, NumGemmKPrefetchStage>())>;

    template <
        InMemoryDataOperationEnum CGlobalMemoryDataOperation_ = InMemoryDataOperationEnum::Set>
    __device__ static bool constexpr IsValidCompilationParameter()
    {
        constexpr bool valid = tensor_operation::device::IsValidGemmCompilationParameter<
            BlockSize,
            MPerBlock,
            NPerBlock,
            MPerXdl,
            NPerXdl,
            MXdlPerWave,
            NXdlPerWave,
            FloatC,
            CGlobalMemoryDataOperation_>();
        if constexpr(!valid)
        {
            return false;
        }

        if constexpr(K1Value % MfmaSelector<FloatAB, MPerXdl, NPerXdl, FloatAB, true>::selected_mfma
                                   .k_per_blk !=
                     0)
        {
            return false;
        }
        return true;
    }

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    template <typename Block2CTileMap>
    __host__ static bool CheckValidity(const AGridDesc_K0_M_K1& a_grid_desc_k0_m_k1,
                                       const BGridDesc_K0_N_K1& b_grid_desc_k0_n_k1,
                                       const CGridDesc_M_N& c_grid_desc_m_n,
                                       const Block2CTileMap& block_2_ctile_map)
    {
        static_assert(is_known_at_compile_time<remove_cv_t<decltype(K1)>>::value,
                      "wrong! K1 need to be known at compile-time");

        static_assert((MPerBlock % (MPerXdl * MXdlPerWave) == 0) &&
                          (NPerBlock % (NXdlPerWave * NPerXdl)) == 0,
                      "Invalid tuning param!");

        if(!is_xdl_wmma_k_supported<FloatAB, K0PerBlock * K1Value, K1Value>())
        {
            return false;
        }

        const auto M  = a_grid_desc_k0_m_k1.GetLength(I1);
        const auto N  = b_grid_desc_k0_n_k1.GetLength(I1);
        const auto K0 = a_grid_desc_k0_m_k1.GetLength(I0);

        if(!(M == c_grid_desc_m_n.GetLength(I0) && N == c_grid_desc_m_n.GetLength(I1) &&
             K0 == b_grid_desc_k0_n_k1.GetLength(I0) && K1 == a_grid_desc_k0_m_k1.GetLength(I2) &&
             K1 == b_grid_desc_k0_n_k1.GetLength(I2)))
            return false;

        if(!(M % MPerBlock == 0 && N % NPerBlock == 0 && K0 % K0PerBlock == 0))
            return false;

        // check gridwise gemm pipeline
        const auto num_k_loop = K0 / K0PerBlock;

        if(!GridwiseGemmPipe::IsSupported(num_k_loop))
        {
            return false;
        }

        if(!block_2_ctile_map.CheckValidity(c_grid_desc_m_n))
        {
            return false;
        }

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        return true;
    }

    __host__ __device__ static constexpr bool CalculateHasMainKBlockLoop(index_t K)
    {
        const index_t num_loop = K / (K0PerBlock * K1);

        return GridwiseGemmPipe::CalculateHasMainLoop(num_loop);
    }

    template <typename CGridDesc_M_N_>
    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl(
        const CGridDesc_M_N_& c_grid_desc_m_n)
    {
        const auto M = c_grid_desc_m_n.GetLength(I0);
        const auto N = c_grid_desc_m_n.GetLength(I1);

        const auto MBlock = M / MPerBlock;
        const auto NBlock = N / NPerBlock;

        constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t NWave =
            NXdlPerWave * NPerXdl == 0 ? 1 : NPerBlock / (NXdlPerWave * NPerXdl);

        const auto c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl =
            transform_tensor_descriptor(
                c_grid_desc_m_n,
                make_tuple(make_unmerge_transform(make_tuple(
                               MBlock, Number<MXdlPerWave>{}, Number<MWave * MPerXdl>{})),
                           make_unmerge_transform(make_tuple(
                               NBlock, Number<NXdlPerWave>{}, Number<NWave * NPerXdl>{}))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 2>{}, Sequence<3, 4, 5>{}));

        return c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl;
    }

    // return block_id to C matrix tile idx (m0, n0) mapping
    __host__ __device__ static constexpr auto MakeDefaultBlock2CTileMap(
        const CGridDesc_M_N& c_grid_desc_m_n, index_t /* M01 */, index_t /* N01 */)
    {
        return BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock, CGridDesc_M_N>(
            c_grid_desc_m_n);
    }
    using CGridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl =
        remove_cvref_t<
            decltype(MakeCGridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl(
                CGridDesc_M_N{}))>;

    using C0GridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl =
        remove_cvref_t<
            decltype(MakeCGridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl(
                C0GridDesc_M_N{}))>;

    using C1GridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl =
        remove_cvref_t<
            decltype(MakeCGridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl(
                C1GridDesc_M_N{}))>;

    using DefaultBlock2CTileMap =
        remove_cvref_t<decltype(MakeDefaultBlock2CTileMap(CGridDesc_M_N{}, 1, 1))>;

    template <bool HasMainKBlockLoop, typename Block2CTileMap>
    __device__ static void
    Run(const FloatAB* __restrict__ p_a_grid,
        const FloatAB* __restrict__ p_b_grid,
        FloatC* __restrict__ p_c_grid,
        const FloatC* __restrict__ p_c0_grid,
        const FloatC* __restrict__ p_c1_grid,
        void* __restrict__ p_shared,
        const AGridDesc_K0_M_K1& a_grid_desc_k0_m_k1,
        const BGridDesc_K0_N_K1& b_grid_desc_k0_n_k1,
        const CGridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl&
            c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
        const C0GridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl&
            c0_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
        const C1GridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl&
            c1_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
        const AElementwiseOperation& a_element_op,
        const BElementwiseOperation& b_element_op,
        const CElementwiseOperation& c_element_op,
        const Block2CTileMap& block_2_ctile_map)
    {
        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc_k0_m_k1.GetElementSpaceSize());
        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_grid_desc_k0_n_k1.GetElementSpaceSize());
        auto c_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_grid,
            c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl
                .GetElementSpaceSize());
        auto c0_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c0_grid,
            c0_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl
                .GetElementSpaceSize());
        auto c1_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c1_grid,
            c1_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl
                .GetElementSpaceSize());

        const auto K0 = a_grid_desc_k0_m_k1.GetLength(I0);

        // divide block work by [M, N]
        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        if(!block_2_ctile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(
                   c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl
                       .GetLength(I0),
                   c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl
                       .GetLength(I3))))
        {
            return;
        }

        // HACK: this force m/n_block_data_idx_on_grid into SGPR
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * MPerBlock);

        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * NPerBlock);

        // lds max alignment
        constexpr auto max_lds_align = K1;

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_k0_m_k1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1(get_device_arch());

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_k0_n_k1 =
            GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1(get_device_arch());

        // A matrix blockwise copy
        auto a_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                AElementwiseOperation,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<K0PerBlock, MPerBlock, K1>,
                                                ABlockTransferThreadClusterLengths_K0_M_K1,
                                                ABlockTransferThreadClusterArrangeOrder,
                                                FloatAB,
                                                FloatAB,
                                                decltype(a_grid_desc_k0_m_k1),
                                                decltype(a_block_desc_k0_m_k1),
                                                ABlockTransferSrcAccessOrder,
                                                Sequence<1, 0, 2>,
                                                ABlockTransferSrcVectorDim,
                                                2,
                                                ABlockTransferSrcScalarPerVector,
                                                ABlockTransferDstScalarPerVector_K1,
                                                1,
                                                1,
                                                AThreadTransferSrcResetCoordinateAfterRun,
                                                true>(
                a_grid_desc_k0_m_k1,
                make_multi_index(0, m_block_data_idx_on_grid, 0),
                a_element_op,
                a_block_desc_k0_m_k1,
                make_multi_index(0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});

        // B matrix blockwise copy
        auto b_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                BElementwiseOperation,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<K0PerBlock, NPerBlock, K1>,
                                                BBlockTransferThreadClusterLengths_K0_N_K1,
                                                BBlockTransferThreadClusterArrangeOrder,
                                                FloatAB,
                                                FloatAB,
                                                decltype(b_grid_desc_k0_n_k1),
                                                decltype(b_block_desc_k0_n_k1),
                                                BBlockTransferSrcAccessOrder,
                                                Sequence<1, 0, 2>,
                                                BBlockTransferSrcVectorDim,
                                                2,
                                                BBlockTransferSrcScalarPerVector,
                                                BBlockTransferDstScalarPerVector_K1,
                                                1,
                                                1,
                                                BThreadTransferSrcResetCoordinateAfterRun,
                                                true>(
                b_grid_desc_k0_n_k1,
                make_multi_index(0, n_block_data_idx_on_grid, 0),
                b_element_op,
                b_block_desc_k0_n_k1,
                make_multi_index(0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});

        // GEMM definition
        //   c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[K0PerBlock, MPerBlock] is in LDS
        //     b_mtx[K0PerBlock, NPerBlock] is in LDS
        //     c_mtx[MPerBlock, NPerBlock] is distributed among threads, and saved in
        //       register
        // sanity check

        auto blockwise_gemm =
            BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1<BlockSize,
                                                                FloatAB,
                                                                FloatAB,
                                                                FloatAcc,
                                                                decltype(a_block_desc_k0_m_k1),
                                                                decltype(b_block_desc_k0_n_k1),
                                                                MPerXdl,
                                                                NPerXdl,
                                                                MXdlPerWave,
                                                                NXdlPerWave,
                                                                K1>{};

        auto c_thread_buf = blockwise_gemm.GetCThreadBuffer();

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size_aligned =
            math::integer_least_multiple(a_block_desc_k0_m_k1.GetElementSpaceSize(), max_lds_align);

        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<FloatAB*>(p_shared), a_block_desc_k0_m_k1.GetElementSpaceSize());

        auto b_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<FloatAB*>(p_shared) + a_block_space_size_aligned,
            b_block_desc_k0_n_k1.GetElementSpaceSize());

        constexpr auto a_block_slice_copy_step = make_multi_index(K0PerBlock, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(K0PerBlock, 0, 0);

        // gridwise GEMM pipeline
        const index_t K0BlockMainLoop = __builtin_amdgcn_readfirstlane(K0 / K0PerBlock);

        GridwiseGemmPipe::template Run<HasMainKBlockLoop>(a_grid_desc_k0_m_k1,
                                                          a_block_desc_k0_m_k1,
                                                          a_blockwise_copy,
                                                          a_grid_buf,
                                                          a_block_buf,
                                                          a_block_slice_copy_step,
                                                          b_grid_desc_k0_n_k1,
                                                          b_block_desc_k0_n_k1,
                                                          b_blockwise_copy,
                                                          b_grid_buf,
                                                          b_block_buf,
                                                          b_block_slice_copy_step,
                                                          blockwise_gemm,
                                                          c_thread_buf,
                                                          K0BlockMainLoop);

        // shuffle C and write out
        {
            static_assert(MXdlPerWave % CShuffleMXdlPerWavePerShuffle == 0 &&
                              NXdlPerWave % CShuffleNXdlPerWavePerShuffle == 0,
                          "wrong!");

            constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);
            constexpr index_t NWave = NPerBlock / (NXdlPerWave * NPerXdl);

            constexpr auto c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
                Base::template GetCThreadDescriptor<false, decltype(blockwise_gemm)>();

            constexpr auto
                c_block_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl = Base::
                    GetCBlockDescriptor_MBlock_NXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl(
                        get_device_arch());

            auto c_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                static_cast<FloatC*>(p_shared),
                c_block_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl
                    .GetElementSpaceSize());

            constexpr auto c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2 = Base::template GetCBlockThreadDescriptor<
                false,
                decltype(blockwise_gemm),
                decltype(c_block_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl)>();

            // VGPR to LDS
            auto c_thread_copy_vgpr_to_lds = Base::template GetCThreadCopyVgprToLds<false>(
                blockwise_gemm,
                c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                ck::tensor_operation::element_wise::PassThrough{});

            auto c_block_copy_lds_to_global = ThreadGroupTensorSliceTransfer_v6r3<
                ThisThreadBlock,            // ThreadGroup
                CElementwiseOperation,      // ElementwiseOperation,
                CGlobalMemoryDataOperation, // DstInMemOp,
                Sequence<1,
                         CShuffleMXdlPerWavePerShuffle,
                         MWave * MPerXdl,
                         1,
                         CShuffleNXdlPerWavePerShuffle,
                         NWave * NPerXdl>, // BlockSliceLengths,
                CBlockTransferClusterLengths_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl,
                Sequence<0, 1, 2, 3, 4, 5>, // typename ThreadClusterArrangeOrder,
                FloatC,                     // typename Src0Data,
                FloatC,                     // typename Src1Data,
                FloatC,                     // typename Src2Data,
                FloatC,                     // typename DstData,
                decltype(c_block_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl),
                decltype(c0_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl),
                decltype(c1_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl),
                decltype(c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl),
                Sequence<0, 1, 2, 3, 4, 5>,                 // typename DimAccessOrder,
                5,                                          // index_t VectorDim,
                CBlockTransferScalarPerVector_NWaveNPerXdl, // index_t ScalarPerVector,
                true,  // bool ThreadTransferSrc0ResetCoordinateAfterRun,
                false, // bool ThreadTransferSrc1ResetCoordinateAfterRun,
                false, // bool ThreadTransferSrc2ResetCoordinateAfterRun,
                false> // bool ThreadTransferDstResetCoordinateAfterRun>
                {c_block_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
                 make_multi_index(0, 0, 0, 0, 0, 0),
                 c0_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
                 make_multi_index(block_work_idx[I0], 0, 0, block_work_idx[I1], 0, 0),
                 c1_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
                 make_multi_index(block_work_idx[I0], 0, 0, block_work_idx[I1], 0, 0),
                 c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
                 make_multi_index(block_work_idx[I0], 0, 0, block_work_idx[I1], 0, 0),
                 c_element_op};

            constexpr auto mxdlperwave_forward_step =
                make_multi_index(0, CShuffleMXdlPerWavePerShuffle, 0, 0, 0, 0);
            constexpr auto nxdlperwave_forward_step =
                make_multi_index(0, 0, 0, 0, CShuffleNXdlPerWavePerShuffle, 0);
            constexpr auto nxdlperwave_backward_step =
                make_multi_index(0, 0, 0, 0, -CShuffleNXdlPerWavePerShuffle, 0);

            static_for<0, MXdlPerWave, CShuffleMXdlPerWavePerShuffle>{}([&](auto mxdlperwave_iter) {
                constexpr auto mxdlperwave = mxdlperwave_iter;

                static_for<0,
                           NXdlPerWave,
                           CShuffleNXdlPerWavePerShuffle>{}([&](auto nxdlperwave_iter) {
                    constexpr bool nxdlperwave_forward_sweep =
                        (mxdlperwave % (2 * CShuffleMXdlPerWavePerShuffle) == 0);

                    constexpr index_t nxdlperwave_value =
                        nxdlperwave_forward_sweep
                            ? nxdlperwave_iter
                            : (NXdlPerWave - nxdlperwave_iter - CShuffleNXdlPerWavePerShuffle);

                    constexpr auto nxdlperwave = Number<nxdlperwave_value>{};

                    // make sure it's safe to do ds_write
                    block_sync_lds();

                    // VGPR to LDS
                    c_thread_copy_vgpr_to_lds.Run(
                        c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                        make_tuple(mxdlperwave, nxdlperwave, I0, I0, I0, I0, I0, I0),
                        c_thread_buf,
                        c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                        c_block_buf);

                    // make sure it's safe to do ds_read
                    block_sync_lds();

                    // LDS to global
                    c_block_copy_lds_to_global.Run(
                        c_block_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
                        c_block_buf,
                        c0_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
                        c0_grid_buf,
                        c1_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
                        c1_grid_buf,
                        c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
                        c_grid_buf);

                    // move on nxdlperwave dimension
                    if constexpr(nxdlperwave_forward_sweep &&
                                 (nxdlperwave < NXdlPerWave - CShuffleNXdlPerWavePerShuffle))
                    {
                        c_block_copy_lds_to_global.MoveSrc1SliceWindow(
                            c0_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
                            nxdlperwave_forward_step);

                        c_block_copy_lds_to_global.MoveSrc2SliceWindow(
                            c1_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
                            nxdlperwave_forward_step);

                        c_block_copy_lds_to_global.MoveDstSliceWindow(
                            c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
                            nxdlperwave_forward_step);
                    }
                    else if constexpr((!nxdlperwave_forward_sweep) && (nxdlperwave > 0))
                    {
                        c_block_copy_lds_to_global.MoveSrc1SliceWindow(
                            c0_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
                            nxdlperwave_backward_step);

                        c_block_copy_lds_to_global.MoveSrc2SliceWindow(
                            c1_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
                            nxdlperwave_backward_step);

                        c_block_copy_lds_to_global.MoveDstSliceWindow(
                            c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
                            nxdlperwave_backward_step);
                    }
                });

                // move on mxdlperwave dimension
                if constexpr(mxdlperwave < MXdlPerWave - CShuffleMXdlPerWavePerShuffle)
                {
                    c_block_copy_lds_to_global.MoveSrc1SliceWindow(
                        c0_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
                        mxdlperwave_forward_step);

                    c_block_copy_lds_to_global.MoveSrc2SliceWindow(
                        c1_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
                        mxdlperwave_forward_step);

                    c_block_copy_lds_to_global.MoveDstSliceWindow(
                        c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
                        mxdlperwave_forward_step);
                }
            });
        }
    }
};

} // namespace ck
