// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_space_filling_curve.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_common.hpp"

namespace ck {

template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename AGridDesc_AK0_M_AK1,
          typename BGridDesc_BK0_N_BK1,
          typename CGridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename Block2CTileMap,
          bool HasMainK0BlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(GridwiseGemm::MaxBlockSize, CK_MIN_BLOCK_PER_CU)
#endif
    kernel_gemm_xdlops_v3r1(
        const FloatAB* __restrict__ p_a_grid,
        const FloatAB* __restrict__ p_b_grid,
        FloatC* __restrict__ p_c_grid,
        const AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1,
        const BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1,
        const CGridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl
            c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
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

        GridwiseGemm::template Run<HasMainK0BlockLoop>(
            p_a_grid,
            p_b_grid,
            p_c_grid,
            p_shared,
            a_grid_desc_ak0_m_ak1,
            b_grid_desc_bk0_n_bk1,
            c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
            a_element_op,
            b_element_op,
            c_element_op,
            block_2_ctile_map);
    }
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_c_grid;
    ignore = a_grid_desc_ak0_m_ak1;
    ignore = b_grid_desc_bk0_n_bk1;
    ignore = c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl;
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
    typename FloatCShuffle,
    typename FloatC,
    InMemoryDataOperationEnum CGlobalMemoryDataOperation,
    typename AGridDesc_AK0_M_AK1,
    typename BGridDesc_BK0_N_BK1,
    typename CGridDesc_M_N,
    typename AElementwiseOperation,
    typename BElementwiseOperation,
    typename CElementwiseOperation,
    index_t MPerBlock,
    index_t NPerBlock,
    index_t KPerBlock,
    index_t AK1Value,
    index_t BK1Value,
    index_t MPerXdl,
    index_t NPerXdl,
    index_t MXdlPerWave,
    index_t NXdlPerWave,
    typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
    typename ABlockTransferThreadClusterArrangeOrder,
    typename ABlockTransferSrcAccessOrder,
    index_t ABlockTransferSrcVectorDim,
    index_t ABlockTransferSrcScalarPerVector,
    index_t ABlockTransferDstScalarPerVector_K1,
    bool AThreadTransferSrcResetCoordinateAfterRun,
    bool ABlockLdsExtraM,
    typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
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
struct GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v3r1
    : public GridwiseGemm_xdl_cshuffle_base<
          tensor_layout::gemm::RowMajor,
          tensor_layout::gemm::ColumnMajor,
          tensor_layout::gemm::RowMajor,
          FloatAB,
          FloatAB,
          FloatAcc,
          FloatCShuffle,
          Tuple<>,
          FloatC,
          AElementwiseOperation,
          BElementwiseOperation,
          BlockSize,
          MPerBlock,
          NPerBlock,
          KPerBlock,
          AK1Value,
          BK1Value,
          MPerXdl,
          NPerXdl,
          MXdlPerWave,
          NXdlPerWave,
          ABlockTransferThreadClusterLengths_AK0_M_AK1,
          ABlockTransferThreadClusterArrangeOrder,
          ABlockTransferSrcAccessOrder,
          ABlockTransferSrcVectorDim,
          ABlockTransferSrcScalarPerVector,
          ABlockTransferDstScalarPerVector_K1,
          AThreadTransferSrcResetCoordinateAfterRun,
          ABlockLdsExtraM,
          BBlockTransferThreadClusterLengths_BK0_N_BK1,
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
        FloatCShuffle,
        Tuple<>,
        FloatC,
        AElementwiseOperation,
        BElementwiseOperation,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1Value,
        BK1Value,
        MPerXdl,
        NPerXdl,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
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
        true>;

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

    using GridwiseGemmPipe = remove_cvref_t<
        decltype(GridwiseGemmPipeline_Selector<PipelineVer, NumGemmKPrefetchStage>())>;

    template <
        InMemoryDataOperationEnum CGlobalMemoryDataOperation_ = InMemoryDataOperationEnum::Set>
    __device__ static bool constexpr IsValidCompilationParameter()
    {
        return ck::tensor_operation::device::IsValidGemmCompilationParameter<
            BlockSize,
            MPerBlock,
            NPerBlock,
            MPerXdl,
            NPerXdl,
            MXdlPerWave,
            NXdlPerWave,
            FloatC,
            CGlobalMemoryDataOperation>();
    }

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    template <typename Block2CTileMap>
    __host__ __device__ static constexpr bool
    CheckValidity(const AGridDesc_AK0_M_AK1& a_grid_desc_ak0_m_ak1,
                  const BGridDesc_BK0_N_BK1& b_grid_desc_bk0_n_bk1,
                  const CGridDesc_M_N& c_grid_desc_m_n,
                  const Block2CTileMap& block_2_ctile_map)
    {
        // static_assert(is_known_at_compile_time<remove_cv_t<decltype(AK1)>>::value &&
        //               is_known_at_compile_time<remove_cv_t<decltype(BK1)>>::value,
        //               "wrong! K1 need to be known at compile-time");

        static_assert((MPerBlock % (MPerXdl * MXdlPerWave) == 0) &&
                          (NPerBlock % (NXdlPerWave * NPerXdl)) == 0,
                      "Invalid tuning param!");

        const auto M = a_grid_desc_ak0_m_ak1.GetLength(I1);
        const auto N = b_grid_desc_bk0_n_bk1.GetLength(I1);
        const auto K = a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2);

        if(!(M == c_grid_desc_m_n.GetLength(I0) && N == c_grid_desc_m_n.GetLength(I1)))
            return false;

        if(!(M % MPerBlock == 0 && N % NPerBlock == 0 && K % KPerBlock == 0))
            return false;

        // check gridwise gemm pipeline
        const auto num_k_loop = K / KPerBlock;

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
        const index_t num_loop = K / KPerBlock;

        return GridwiseGemmPipe::CalculateHasMainLoop(num_loop);
    }

    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl(
        const CGridDesc_M_N& c_grid_desc_m_n)
    {
        const auto M = c_grid_desc_m_n.GetLength(I0);
        const auto N = c_grid_desc_m_n.GetLength(I1);

        const auto MBlock = M / MPerBlock;
        const auto NBlock = N / NPerBlock;

        constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t NWave = NPerBlock / (NXdlPerWave * NPerXdl);

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

    using DefaultBlock2CTileMap =
        remove_cvref_t<decltype(MakeDefaultBlock2CTileMap(CGridDesc_M_N{}, 1, 1))>;

    template <bool HasMainK0BlockLoop, typename Block2CTileMap>
    __device__ static void
    Run(const FloatAB* __restrict__ p_a_grid,
        const FloatAB* __restrict__ p_b_grid,
        FloatC* __restrict__ p_c_grid,
        void* __restrict__ p_shared,
        const AGridDesc_AK0_M_AK1& a_grid_desc_ak0_m_ak1,
        const BGridDesc_BK0_N_BK1& b_grid_desc_bk0_n_bk1,
        const CGridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl&
            c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
        const AElementwiseOperation& a_element_op,
        const BElementwiseOperation& b_element_op,
        const CElementwiseOperation& c_element_op,
        const Block2CTileMap& block_2_ctile_map)
    {
        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc_ak0_m_ak1.GetElementSpaceSize());
        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_grid_desc_bk0_n_bk1.GetElementSpaceSize());
        auto c_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_grid,
            c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl
                .GetElementSpaceSize());
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
        constexpr auto max_lds_align = math::lcm(AK1, BK1);

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_ak0_m_ak1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1(get_device_arch());

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_bk0_n_bk1 =
            GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1(get_device_arch());

        // A matrix blockwise copy
        auto a_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                AElementwiseOperation,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<AK0, MPerBlock, AK1>,
                                                ABlockTransferThreadClusterLengths_AK0_M_AK1,
                                                ABlockTransferThreadClusterArrangeOrder,
                                                FloatAB,
                                                FloatAB,
                                                decltype(a_grid_desc_ak0_m_ak1),
                                                decltype(a_block_desc_ak0_m_ak1),
                                                ABlockTransferSrcAccessOrder,
                                                Sequence<1, 0, 2>,
                                                ABlockTransferSrcVectorDim,
                                                2,
                                                ABlockTransferSrcScalarPerVector,
                                                ABlockTransferDstScalarPerVector_K1,
                                                1,
                                                1,
                                                AThreadTransferSrcResetCoordinateAfterRun,
                                                true,
                                                NumGemmKPrefetchStage>(
                a_grid_desc_ak0_m_ak1,
                make_multi_index(0, m_block_data_idx_on_grid, 0),
                a_element_op,
                a_block_desc_ak0_m_ak1,
                make_multi_index(0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});

        // B matrix blockwise copy
        auto b_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                BElementwiseOperation,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<BK0, NPerBlock, BK1>,
                                                BBlockTransferThreadClusterLengths_BK0_N_BK1,
                                                BBlockTransferThreadClusterArrangeOrder,
                                                FloatAB,
                                                FloatAB,
                                                decltype(b_grid_desc_bk0_n_bk1),
                                                decltype(b_block_desc_bk0_n_bk1),
                                                BBlockTransferSrcAccessOrder,
                                                Sequence<1, 0, 2>,
                                                BBlockTransferSrcVectorDim,
                                                2,
                                                BBlockTransferSrcScalarPerVector,
                                                BBlockTransferDstScalarPerVector_K1,
                                                1,
                                                1,
                                                BThreadTransferSrcResetCoordinateAfterRun,
                                                true,
                                                NumGemmKPrefetchStage>(
                b_grid_desc_bk0_n_bk1,
                make_multi_index(0, n_block_data_idx_on_grid, 0),
                b_element_op,
                b_block_desc_bk0_n_bk1,
                make_multi_index(0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});

        // GEMM definition
        //   c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[K0PerBlock, MPerBlock] is in LDS
        //     b_mtx[K0PerBlock, NPerBlock] is in LDS
        //     c_mtx[MPerBlock, NPerBlock] is distributed among threads, and saved in
        //       register
        // sanity check
        constexpr auto lcm_AK1_BK1 = math::lcm(AK1, BK1);
        constexpr bool is_single_rate_mfma =
            (((is_same<FloatAB, half_t>::value || is_same<FloatAB, bhalf_t>::value) &&
              lcm_AK1_BK1 <= 4) ||
             (is_same<FloatAB, int8_t>::value && lcm_AK1_BK1 <= 8) ||
             ((is_same<FloatAB, f8_t>::value || is_same<FloatAB, bf8_t>::value) &&
              lcm_AK1_BK1 < 32))
                ? true
                : false;
        constexpr auto is_scale_mfma = false;
        constexpr index_t k_pack     = math::max(
            lcm_AK1_BK1,
            MfmaSelector<FloatAB, MPerXdl, NPerXdl, FloatAB, is_single_rate_mfma, is_scale_mfma>::
                selected_mfma.k_per_blk);

        auto blockwise_gemm =
            BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1<BlockSize,
                                                                FloatAB,
                                                                FloatAB,
                                                                FloatAcc,
                                                                decltype(a_block_desc_ak0_m_ak1),
                                                                decltype(b_block_desc_bk0_n_bk1),
                                                                MPerXdl,
                                                                NPerXdl,
                                                                MXdlPerWave,
                                                                NXdlPerWave,
                                                                k_pack>{};

        auto c_thread_buf = blockwise_gemm.GetCThreadBuffer();

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);

        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<FloatAB*>(p_shared), a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        auto b_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<FloatAB*>(p_shared) + a_block_space_size_aligned,
            b_block_desc_bk0_n_bk1.GetElementSpaceSize());

        constexpr auto a_block_slice_copy_step = make_multi_index(KPerBlock / AK1, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(KPerBlock / BK1, 0, 0);

        // gridwise GEMM pipeline
        const index_t num_k_block_main_loop = __builtin_amdgcn_readfirstlane(
            (a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2)) /
            KPerBlock);

        GridwiseGemmPipe::template Run<HasMainK0BlockLoop>(a_grid_desc_ak0_m_ak1,
                                                           a_block_desc_ak0_m_ak1,
                                                           a_blockwise_copy,
                                                           a_grid_buf,
                                                           a_block_buf,
                                                           a_block_slice_copy_step,
                                                           b_grid_desc_bk0_n_bk1,
                                                           b_block_desc_bk0_n_bk1,
                                                           b_blockwise_copy,
                                                           b_grid_buf,
                                                           b_block_buf,
                                                           b_block_slice_copy_step,
                                                           blockwise_gemm,
                                                           c_thread_buf,
                                                           num_k_block_main_loop);

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

            auto c_shuffle_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
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

            // LDS to global
            auto c_block_copy_lds_to_global = ThreadGroupTensorSliceTransfer_v6r1<
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
                FloatCShuffle,              // typename SrcData,
                FloatC,                     // typename DstData,
                decltype(c_block_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl),
                decltype(c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl),
                Sequence<0, 1, 2, 3, 4, 5>,                 // typename DimAccessOrder,
                5,                                          // index_t VectorDim,
                CBlockTransferScalarPerVector_NWaveNPerXdl, // index_t ScalarPerVector,
                true,  // bool ThreadTransferSrcResetCoordinateAfterRun,
                false> // bool ThreadTransferDstResetCoordinateAfterRun>
                {c_block_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
                 make_multi_index(0, 0, 0, 0, 0, 0),
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
                        c_shuffle_block_buf);

                    // make sure it's safe to do ds_read
                    block_sync_lds();

                    // LDS to global
                    c_block_copy_lds_to_global.Run(
                        c_block_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
                        c_shuffle_block_buf,
                        c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
                        c_grid_buf);

                    // move on nxdlperwave dimension
                    if constexpr(nxdlperwave_forward_sweep &&
                                 (nxdlperwave < NXdlPerWave - CShuffleNXdlPerWavePerShuffle))
                    {
                        c_block_copy_lds_to_global.MoveDstSliceWindow(
                            c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
                            nxdlperwave_forward_step);
                    }
                    else if constexpr((!nxdlperwave_forward_sweep) && (nxdlperwave > 0))
                    {
                        c_block_copy_lds_to_global.MoveDstSliceWindow(
                            c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
                            nxdlperwave_backward_step);
                    }
                });

                // move on mxdlperwave dimension
                if constexpr(mxdlperwave < MXdlPerWave - CShuffleMXdlPerWavePerShuffle)
                {
                    c_block_copy_lds_to_global.MoveDstSliceWindow(
                        c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl,
                        mxdlperwave_forward_step);
                }
            });
        }
    }
};

} // namespace ck
