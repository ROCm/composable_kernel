// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v6r1.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename AGridDesc_AK0_M_AK1,
          typename BGridDesc_BK0_N_BK1,
          typename CGridDescriptor_M0_N0_M1_N1_M2_N2_N3_N4,
          typename Block2CTileMap,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_gemm_xdl_direct_c_write_out(
            const FloatAB* __restrict__ p_a_grid,
            const FloatAB* __restrict__ p_b_grid,
            FloatC* __restrict__ p_c_grid,
            const AElementwiseOperation a_element_op,
            const BElementwiseOperation b_element_op,
            const CElementwiseOperation c_element_op,
            const AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1,
            const BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1,
            const CGridDescriptor_M0_N0_M1_N1_M2_N2_N3_N4 c_grid_desc_M0_N0_M1_N1_M2_N2_N3_N4,
            const Block2CTileMap block_2_ctile_map)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__) || \
    defined(__gfx940__))
    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    GridwiseGemm::template Run<HasMainKBlockLoop>(p_a_grid,
                                                  p_b_grid,
                                                  p_c_grid,
                                                  p_shared,
                                                  a_element_op,
                                                  b_element_op,
                                                  c_element_op,
                                                  a_grid_desc_ak0_m_ak1,
                                                  b_grid_desc_bk0_n_bk1,
                                                  c_grid_desc_M0_N0_M1_N1_M2_N2_N3_N4,
                                                  block_2_ctile_map);
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_c_grid;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = c_element_op;
    ignore = a_grid_desc_ak0_m_ak1;
    ignore = b_grid_desc_bk0_n_bk1;
    ignore = c_grid_desc_M0_N0_M1_N1_M2_N2_N3_N4;
    ignore = block_2_ctile_map;
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
}

template <typename FloatAB,
          typename FloatGemmAcc,
          typename FloatC,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          typename AGridDesc_AK0_M_AK1,
          typename BGridDesc_BK0_N_BK1,
          typename CGridDesc_M_N,
          index_t NumGemmKPrefetchStage,
          index_t BlockSize,
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
          index_t ABlockTransferDstScalarPerVector_AK1,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          index_t BBlockLdsExtraN,
          typename CThreadTransferDstAccessOrder,
          index_t CThreadTransferDstVectorDim,
          index_t CThreadTransferDstScalarPerVector,
          LoopScheduler LoopSched,
          PipelineVersion PipelineVer = PipelineVersion::v1>
struct GridwiseGemm_k0mk1_k0nk1_mn_xdl_v1
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};

    // K1 should be Number<...>
    static constexpr auto AK0 = Number<KPerBlock / AK1Value>{};
    static constexpr auto BK0 = Number<KPerBlock / BK1Value>{};
    static constexpr auto AK1 = Number<AK1Value>{};
    static constexpr auto BK1 = Number<BK1Value>{};

    using ThisThreadBlock  = ThisThreadBlock<BlockSize>;
    using GridwiseGemmPipe = remove_cvref_t<decltype(
        GridwiseGemmPipeline_Selector<PipelineVer, NumGemmKPrefetchStage, LoopSched>())>;

    template <typename ABlockDesc_AK0_M_AK1>
    __host__ __device__ static constexpr auto
    MakeGemmAMmaTileDescriptor_M0_M1_M2_K(const ABlockDesc_AK0_M_AK1&)
    {
        constexpr index_t MWaves = MPerBlock / (MXdlPerWave * MPerXdl);

        return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<MXdlPerWave, MWaves, MPerXdl>(
            ABlockDesc_AK0_M_AK1{});
    }

    template <typename BBlockDesc_BK0_N_BK1>
    __host__ __device__ static constexpr auto
    MakeGemmBMmaTileDescriptor_N0_N1_N2_K(const BBlockDesc_BK0_N_BK1&)
    {
        constexpr index_t NWaves = NPerBlock / (NXdlPerWave * NPerXdl);

        return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<NXdlPerWave, NWaves, NPerXdl>(
            BBlockDesc_BK0_N_BK1{});
    }

    __host__ __device__ static constexpr auto GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1()
    {
        // A matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(AK0, Number<MPerBlock>{}, AK1),
            make_tuple(Number<MPerBlock + ABlockLdsExtraM>{} * AK1, AK1, I1));
    }

    __host__ __device__ static constexpr auto GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1()
    {
        // B matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(BK0, Number<NPerBlock>{}, BK1),
            make_tuple(Number<NPerBlock + BBlockLdsExtraN>{} * BK1, BK1, I1));
    }

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_desc_ak0_m_ak1 = GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();
        constexpr auto b_block_desc_bk0_n_bk1 = GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        // lds max alignment
        constexpr auto max_lds_align = math::lcm(AK1, BK1);

        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_space_size_aligned = math::integer_least_multiple(
            b_block_desc_bk0_n_bk1.GetElementSpaceSize(), max_lds_align);

        return (a_block_space_size_aligned + b_block_space_size_aligned) * sizeof(FloatAB);
    }

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    template <typename Block2CTileMap>
    __host__ __device__ static constexpr bool
    CheckValidity(const AGridDesc_AK0_M_AK1& a_grid_desc_ak0_m_ak1,
                  const BGridDesc_BK0_N_BK1& b_grid_desc_bk0_n_bk1,
                  const CGridDesc_M_N& c_grid_desc_m_n,
                  const Block2CTileMap& block_2_ctile_map)
    {
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
    MakeCGridDescriptor_M0_N0_M1_N1_M2_N2_N3_N4(const CGridDesc_M_N& c_grid_desc_m_n)
    {
        using ABlockDesc_AK0_M_AK1 =
            remove_cvref_t<decltype(GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1())>;
        using BBlockDesc_AK0_N_AK1 =
            remove_cvref_t<decltype(GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1())>;

        using GemmAMmaTileDesc =
            remove_cvref_t<decltype(MakeGemmAMmaTileDescriptor_M0_M1_M2_K(ABlockDesc_AK0_M_AK1{}))>;
        using GemmBMmaTileDesc =
            remove_cvref_t<decltype(MakeGemmBMmaTileDescriptor_N0_N1_N2_K(BBlockDesc_AK0_N_AK1{}))>;

        constexpr index_t KPack = math::max(
            math::lcm(AK1, BK1), MfmaSelector<FloatAB, MPerXdl, NPerXdl>::selected_mfma.k_per_blk);

        using BlockwiseGemm = BlockwiseGemmXdlops_v2<BlockSize,
                                                     FloatAB,
                                                     FloatGemmAcc,
                                                     ABlockDesc_AK0_M_AK1,
                                                     BBlockDesc_AK0_N_AK1,
                                                     GemmAMmaTileDesc,
                                                     GemmBMmaTileDesc,
                                                     MPerBlock,
                                                     NPerBlock,
                                                     KPerBlock,
                                                     MPerXdl,
                                                     NPerXdl,
                                                     MXdlPerWave,
                                                     NXdlPerWave,
                                                     KPack,
                                                     true>; // TransposeC
                                                            // A MMaTileKStride
                                                            // B MMaTileKStride

        return BlockwiseGemm::MakeCGridDescriptor_M0_N0_M1_N1_M2_N2_N3_N4(c_grid_desc_m_n);
    }

    // return block_id to C matrix tile idx (m0, n0) mapping
    __host__ __device__ static constexpr auto
    MakeDefaultBlock2CTileMap(const CGridDesc_M_N& c_grid_desc_m_n)
    {
        return BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock, CGridDesc_M_N>(
            c_grid_desc_m_n);
    }

    using CGridDescriptor_M0_N0_M1_N1_M2_N2_N3_N4 =
        remove_cvref_t<decltype(MakeCGridDescriptor_M0_N0_M1_N1_M2_N2_N3_N4(CGridDesc_M_N{}))>;

    using DefaultBlock2CTileMap =
        remove_cvref_t<decltype(MakeDefaultBlock2CTileMap(CGridDesc_M_N{}))>;

    template <bool HasMainKBlockLoop, typename Block2CTileMap>
    __device__ static void
    Run(const FloatAB* __restrict__ p_a_grid,
        const FloatAB* __restrict__ p_b_grid,
        FloatC* __restrict__ p_c_grid,
        void* __restrict__ p_shared,
        const AElementwiseOperation& a_element_op,
        const BElementwiseOperation& b_element_op,
        const CElementwiseOperation& c_element_op,
        const AGridDesc_AK0_M_AK1& a_grid_desc_ak0_m_ak1,
        const BGridDesc_BK0_N_BK1& b_grid_desc_bk0_n_bk1,
        const CGridDescriptor_M0_N0_M1_N1_M2_N2_N3_N4& c_grid_desc_m0_n0_m1_n1_m2_n2_n3_n4,
        const Block2CTileMap& block_2_ctile_map)
    {
        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc_ak0_m_ak1.GetElementSpaceSize());
        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_grid_desc_bk0_n_bk1.GetElementSpaceSize());
        auto c_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_grid, c_grid_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetElementSpaceSize());

        // divide block work by [M, N]
        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        if(!block_2_ctile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(a_grid_desc_ak0_m_ak1.GetLength(I1) / MPerBlock,
                          b_grid_desc_bk0_n_bk1.GetLength(I1) / NPerBlock)))
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
        constexpr auto a_block_desc_ak0_m_ak1 = GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_bk0_n_bk1 = GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

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
                                                ABlockTransferDstScalarPerVector_AK1,
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
                                                BBlockTransferDstScalarPerVector_BK1,
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
        constexpr index_t KPack = math::max(
            math::lcm(AK1, BK1), MfmaSelector<FloatAB, MPerXdl, NPerXdl>::selected_mfma.k_per_blk);

        auto blockwise_gemm = BlockwiseGemmXdlops_v2<
            BlockSize,
            FloatAB,
            FloatGemmAcc,
            decltype(a_block_desc_ak0_m_ak1),
            decltype(b_block_desc_bk0_n_bk1),
            decltype(MakeGemmAMmaTileDescriptor_M0_M1_M2_K(a_block_desc_ak0_m_ak1)),
            decltype(MakeGemmBMmaTileDescriptor_N0_N1_N2_K(b_block_desc_bk0_n_bk1)),
            MPerBlock,
            NPerBlock,
            KPerBlock,
            MPerXdl,
            NPerXdl,
            MXdlPerWave,
            NXdlPerWave,
            KPack,
            true>{}; // TransposeC
        // A MMaTileKStride
        // B MMaTileKStride

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
        static_assert(std::is_default_constructible_v<GridwiseGemmPipe>);
        const auto gridwise_gemm_pipeline = GridwiseGemmPipe{};

        const index_t num_k_block_main_loop = __builtin_amdgcn_readfirstlane(
            (a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2)) /
            KPerBlock);

        gridwise_gemm_pipeline.template Run<HasMainKBlockLoop>(a_grid_desc_ak0_m_ak1,
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

        // output: register to global memory
        {
            // M0 - MRepeat / MXdlPerWave
            // N0 - NRepeat / NXdlPerWave
            // M1 - MWaves
            // N1 - NWaves
            // M2 - mfma_instr.num_threads_per_blk
            // N2 - mfma_instr.num_groups_per_blk
            // N3 - mfma_instr.num_input_blks
            // N4 - mfma_instr.group_size
            // {M0, N0, 1, 1, 1, 4, 1, 4}

            // c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4 is only used to get lengths
            constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4 =
                blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();

            constexpr auto M0 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I0);
            constexpr auto N0 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I1);
            constexpr auto M1 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I2);
            constexpr auto N1 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I3);
            constexpr auto M2 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I4);
            constexpr auto N2 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I5);
            constexpr auto N3 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I6);
            constexpr auto N4 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I7);

            constexpr auto CScalarPerVector       = CThreadTransferDstScalarPerVector; // N3
            constexpr index_t MFMAItemsPerThread  = 16; // TODO: this is fixed for MFMA 32x32x8
            constexpr index_t CMThreadVectors     = MFMAItemsPerThread / CScalarPerVector; // M3
            constexpr auto CNVectors              = N2 * N3 * N4 / CScalarPerVector;       // N2
            constexpr index_t CMThreadVectorGroup = get_warp_size() / CNVectors;           // M2

            constexpr auto c_thread_desc_m0_n0_m1_n1_m2_m3_n2_n3 =
                make_naive_tensor_descriptor_packed(make_tuple(
                    M0, N0, I1, I1, I1, Number<CMThreadVectors>{}, I1, Number<CScalarPerVector>{}));

            const auto M0_grid = c_grid_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I0);
            const auto N0_grid = c_grid_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I1);
            const auto M1_grid = c_grid_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I2);
            const auto N1_grid = c_grid_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I3);
            const auto M2_grid = c_grid_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I4);
            const auto N2_grid = c_grid_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I5);
            const auto N3_grid = c_grid_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I6);
            const auto N4_grid = c_grid_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I7);

            // if (blockIdx.x == 0 && ThisThreadBlock::GetThreadId() == 0)
            // {
            //     printf("grid: [M0: %d, N0: %d, M1: %d, N1: %d, M2: %d, N2: %d, N3: %d, N4:
            //     %d]\n",
            //             M0_grid,
            //             N0_grid,
            //             M1_grid,
            //             N1_grid,
            //             M2_grid.value,
            //             N2_grid.value,
            //             N3_grid.value,
            //             N4_grid.value);
            // }

            const auto c_grid_desc_m0_n0_m1_n1_m2_n234_tmp = transform_tensor_descriptor(
                c_grid_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                make_tuple(
                    make_pass_through_transform(M0_grid),
                    make_pass_through_transform(N0_grid),
                    make_pass_through_transform(M1_grid),
                    make_pass_through_transform(N1_grid),
                    make_pass_through_transform(M2_grid),
                    make_merge_transform(make_tuple(
                        N3_grid, N2_grid, N4_grid)) // num_groups_per_blk * group_size * num blks
                    ),
                make_tuple(Sequence<0>{},
                           Sequence<1>{},
                           Sequence<2>{},
                           Sequence<3>{},
                           Sequence<4>{},
                           Sequence<5, 6, 7>{}),
                make_tuple(Sequence<0>{},
                           Sequence<1>{},
                           Sequence<2>{},
                           Sequence<3>{},
                           Sequence<4>{},
                           Sequence<5>{}));

            // if (blockIdx.x == 0 && ThisThreadBlock::GetThreadId() == 0)
            // {
            //     printf("grid tmp: [M0: %d, N0: %d, M1: %d, N1: %d, M2: %d, N234: %d]\n",
            //             c_grid_desc_m0_n0_m1_n1_m2_n234_tmp.GetLength(I0),
            //             c_grid_desc_m0_n0_m1_n1_m2_n234_tmp.GetLength(I1),
            //             c_grid_desc_m0_n0_m1_n1_m2_n234_tmp.GetLength(I2),
            //             c_grid_desc_m0_n0_m1_n1_m2_n234_tmp.GetLength(I3),
            //             c_grid_desc_m0_n0_m1_n1_m2_n234_tmp.GetLength(I4).value,
            //             c_grid_desc_m0_n0_m1_n1_m2_n234_tmp.GetLength(I5).value);
            // }

            const auto c_grid_desc_m0_n0_m1_n1_m2_m3_n2_n3_new = transform_tensor_descriptor(
                c_grid_desc_m0_n0_m1_n1_m2_n234_tmp,
                make_tuple(make_pass_through_transform(M0_grid), // M0 - MRepeat / MXdlPerWave
                           make_pass_through_transform(N0_grid), // N0 - NRepeat / NXdlPerWave
                           make_pass_through_transform(M1_grid), // M1 - MWaves
                           make_pass_through_transform(N1_grid), // N1 - NWaves
                           make_unmerge_transform(make_tuple(Number<CMThreadVectorGroup>{},
                                                             CMThreadVectors)), // M2 -> (M2, M3)
                           make_unmerge_transform(make_tuple(Number<CNVectors>{},
                                                             Number<CScalarPerVector>{})) // N2, N3
                           ),
                make_tuple(Sequence<0>{},
                           Sequence<1>{},
                           Sequence<2>{},
                           Sequence<3>{},
                           Sequence<4>{},
                           Sequence<5>{}),
                make_tuple(Sequence<0>{},
                           Sequence<1>{},
                           Sequence<2>{},
                           Sequence<3>{},
                           Sequence<4, 5>{},
                           Sequence<6, 7>{}));

            // if (blockIdx.x == 0 && ThisThreadBlock::GetThreadId() == 0)
            // {
            //     printf("grid_new: [M0: %d, N0: %d, M1: %d, N1: %d, M2: %d, N2: %d, N3: %d]\n",
            //             c_grid_desc_m0_n0_m1_n1_m2_m3_n2_n3_new.GetLength(I0),
            //             c_grid_desc_m0_n0_m1_n1_m2_m3_n2_n3_new.GetLength(I1),
            //             c_grid_desc_m0_n0_m1_n1_m2_m3_n2_n3_new.GetLength(I2),
            //             c_grid_desc_m0_n0_m1_n1_m2_m3_n2_n3_new.GetLength(I3),
            //             c_grid_desc_m0_n0_m1_n1_m2_m3_n2_n3_new.GetLength(I4).value,
            //             c_grid_desc_m0_n0_m1_n1_m2_m3_n2_n3_new.GetLength(I5).value,
            //             c_grid_desc_m0_n0_m1_n1_m2_m3_n2_n3_new.GetLength(I6).value);
            // }

            const auto wave_idx = blockwise_gemm.GetWaveIdx();

            const auto lane_id_to_m2_n2_adaptor = make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(
                    make_tuple(Number<CMThreadVectorGroup>{}, Number<CNVectors>{}))),
                make_tuple(Sequence<0, 1>{}),
                make_tuple(Sequence<0>{}));

            const auto lane_data_idx_on_block =
                lane_id_to_m2_n2_adaptor.CalculateBottomIndex(make_multi_index(wave_idx[I2]));

            // if (blockIdx.x == 0 && (ThisThreadBlock::GetThreadId() == 0 ||
            //                         ThisThreadBlock::GetThreadId() == 16 ||
            //                         ThisThreadBlock::GetThreadId() == 75 ||
            //                         ThisThreadBlock::GetThreadId() == 234 ))
            // {
            //     printf("[tid on blck %d] M1: %d, N1: %d, M2: %d, N2: %d\n",
            //             ThisThreadBlock::GetThreadId(),
            //             lane_data_idx_on_block[I0],
            //             lane_data_idx_on_block[I1],
            //             lane_data_idx_on_block[I2],
            //             lane_data_idx_on_block[I3]);
            // }

            const auto m_thread_data_on_grid_to_m0_m1_m2_adaptor = make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(M0, M1, M2))),
                make_tuple(Sequence<0, 1, 2>{}),
                make_tuple(Sequence<0>{}));

            const auto m_thread_data_on_grid_idx =
                m_thread_data_on_grid_to_m0_m1_m2_adaptor.CalculateBottomIndex(
                    make_multi_index(m_block_data_idx_on_grid));

            const auto n_thread_data_on_grid_to_n0_n1_n2_n3_n4_adaptor =
                make_single_stage_tensor_adaptor(
                    make_tuple(make_merge_transform(make_tuple(N0, N1, N2, N3, N4))),
                    make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                    make_tuple(Sequence<0>{}));

            const auto n_thread_data_on_grid_idx =
                n_thread_data_on_grid_to_n0_n1_n2_n3_n4_adaptor.CalculateBottomIndex(
                    make_multi_index(n_block_data_idx_on_grid));

            auto c_thread_copy = ThreadwiseTensorSliceTransfer_v1r3<
                FloatGemmAcc,
                FloatC,
                decltype(c_thread_desc_m0_n0_m1_n1_m2_m3_n2_n3),
                decltype(c_grid_desc_m0_n0_m1_n1_m2_m3_n2_n3_new),
                CElementwiseOperation,
                Sequence<M0, N0, I1, I1, I1, CMThreadVectors, I1, CScalarPerVector>, // SliceLengths
                CThreadTransferDstAccessOrder,
                CThreadTransferDstVectorDim,
                CScalarPerVector,
                CGlobalMemoryDataOperation,
                1,
                true>{c_grid_desc_m0_n0_m1_n1_m2_m3_n2_n3_new,
                      make_multi_index(m_thread_data_on_grid_idx[I0],
                                       n_thread_data_on_grid_idx[I0],
                                       wave_idx[I0],
                                       wave_idx[I1],
                                       lane_data_idx_on_block[I0],
                                       I0,
                                       lane_data_idx_on_block[I1],
                                       I0),
                      c_element_op};

            // if (blockIdx.x == 0 || blockIdx.x == 5)
            // {                                                       //  M1, N1, M2, M3, N2, N3
            //     if (ThisThreadBlock::GetThreadId() == 0  ||
            //         ThisThreadBlock::GetThreadId() == 3  ||         // [ 0,  0,
            //         ThisThreadBlock::GetThreadId() == 16 ||         // [ 0,  0,
            //         ThisThreadBlock::GetThreadId() == 33 ||         // [ 0,  0,
            //         ThisThreadBlock::GetThreadId() == 64 ||         // [ 0,  1,
            //         ThisThreadBlock::GetThreadId() == 96 ||         // [ 0,  1,
            //         ThisThreadBlock::GetThreadId() == 130 ||        // [ 1,  0,
            //         ThisThreadBlock::GetThreadId() == 224           // [ 1,  1,
            //        )
            //     {
            //         printf("[B:%d, T:%d] -> dst_slice_origin_idx: [%d, %d, %d, %d, %d, %d, %d,
            //         %d]\n",
            //                 get_block_1d_id(),
            //                 ThisThreadBlock::GetThreadId(),
            //                 m_thread_data_on_grid_idx[I0],
            //                 n_thread_data_on_grid_idx[I0],
            //                 wave_idx[I0],
            //                 wave_idx[I1],
            //                 lane_data_idx_on_block[I0],
            //                 I0.value,
            //                 lane_data_idx_on_block[I1],
            //                 I0.value);
            //     }
            // }

            c_thread_copy.Run(c_thread_desc_m0_n0_m1_n1_m2_m3_n2_n3,
                              make_tuple(I0, I0, I0, I0, I0, I0, I0, I0),
                              c_thread_buf,
                              c_grid_desc_m0_n0_m1_n1_m2_m3_n2_n3_new,
                              c_grid_buf);
        }
    }
};

} // namespace ck
