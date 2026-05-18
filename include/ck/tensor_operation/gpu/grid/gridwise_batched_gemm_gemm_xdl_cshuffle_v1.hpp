// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_common.hpp"

namespace ck {

template <typename FloatAB,
          typename FloatGemmAcc,
          typename FloatCShuffle,
          typename FloatC,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename AccElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CElementwiseOperation,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          typename AGridDesc_AK0_M_AK1,
          typename BGridDesc_BK0_N_BK1,
          typename B1GridDesc_BK0_N_BK1,
          typename CGridDesc_M_N,
          index_t NumGemmKPrefetchStage,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t Gemm1NPerBlock,
          index_t Gemm1KPerBlock,
          index_t AK1Value,
          index_t BK1Value,
          index_t B1K1Value,
          index_t MPerXdl,
          index_t NPerXdl,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          index_t Gemm1NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          bool AThreadTransferSrcResetCoordinateAfterRun, // ignored
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BThreadTransferSrcResetCoordinateAfterRun, // ignored
          index_t BBlockLdsExtraN,
          typename B1BlockTransferThreadClusterLengths_BK0_N_BK1,
          typename B1BlockTransferThreadClusterArrangeOrder,
          typename B1BlockTransferSrcAccessOrder,
          index_t B1BlockTransferSrcVectorDim,
          index_t B1BlockTransferSrcScalarPerVector,
          index_t B1BlockTransferDstScalarPerVector_BK1,
          bool B1ThreadTransferSrcResetCoordinateAfterRun,
          index_t B1BlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          LoopScheduler LoopSched,
          PipelineVersion PipelineVer = PipelineVersion::v1>
struct GridwiseBatchedGemmGemm_Xdl_CShuffle
    : public GridwiseGemm_xdl_cshuffle_base<
          tensor_layout::gemm::RowMajor,
          tensor_layout::gemm::ColumnMajor,
          tensor_layout::gemm::RowMajor,
          FloatAB,
          FloatAB,
          FloatGemmAcc,
          FloatCShuffle,
          Tuple<>,
          FloatC,
          AElementwiseOperation,
          BElementwiseOperation,
          BlockSize,
          MPerBlock,
          Gemm1NPerBlock,
          Gemm1KPerBlock,
          AK1Value,
          B1K1Value,
          MPerXdl,
          NPerXdl,
          MXdlPerWave,
          Gemm1NXdlPerWave,
          ABlockTransferThreadClusterLengths_AK0_M_AK1,
          ABlockTransferThreadClusterArrangeOrder,
          ABlockTransferSrcAccessOrder,
          ABlockTransferSrcVectorDim,
          ABlockTransferSrcScalarPerVector,
          ABlockTransferDstScalarPerVector_AK1,
          AThreadTransferSrcResetCoordinateAfterRun,
          ABlockLdsExtraM,
          B1BlockTransferThreadClusterLengths_BK0_N_BK1,
          B1BlockTransferThreadClusterArrangeOrder,
          B1BlockTransferSrcAccessOrder,
          B1BlockTransferSrcVectorDim,
          B1BlockTransferSrcScalarPerVector,
          B1BlockTransferDstScalarPerVector_BK1,
          B1ThreadTransferSrcResetCoordinateAfterRun,
          B1BlockLdsExtraN,
          CShuffleMXdlPerWavePerShuffle,
          CShuffleNXdlPerWavePerShuffle,
          CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          Sequence<CShuffleBlockTransferScalarPerVector_NPerBlock>,
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
        FloatGemmAcc,
        FloatCShuffle,
        Tuple<>,
        FloatC,
        AElementwiseOperation,
        BElementwiseOperation,
        BlockSize,
        MPerBlock,
        Gemm1NPerBlock,
        Gemm1KPerBlock,
        AK1Value,
        B1K1Value,
        MPerXdl,
        NPerXdl,
        MXdlPerWave,
        Gemm1NXdlPerWave,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsExtraM,
        B1BlockTransferThreadClusterLengths_BK0_N_BK1,
        B1BlockTransferThreadClusterArrangeOrder,
        B1BlockTransferSrcAccessOrder,
        B1BlockTransferSrcVectorDim,
        B1BlockTransferSrcScalarPerVector,
        B1BlockTransferDstScalarPerVector_BK1,
        B1ThreadTransferSrcResetCoordinateAfterRun,
        B1BlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        Sequence<CShuffleBlockTransferScalarPerVector_NPerBlock>,
        FloatAB,
        FloatAB,
        true>; // ForceNaiveLayout

    using Base0 = GridwiseGemm_xdl_cshuffle_base<
        tensor_layout::gemm::RowMajor,
        tensor_layout::gemm::ColumnMajor,
        tensor_layout::gemm::RowMajor,
        FloatAB,
        FloatAB,
        FloatGemmAcc,
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
        ABlockTransferDstScalarPerVector_AK1,
        AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        BThreadTransferSrcResetCoordinateAfterRun, // ignored
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        Sequence<CShuffleBlockTransferScalarPerVector_NPerBlock>,
        FloatAB,
        FloatAB,
        true>; // ForceNaiveLayout

    using Base::I0;
    using Base::I1;
    using Base::I2;
    using Base::I3;
    using Base::I4;
    using Base::I5;
    using Base::I6;
    using Base::I7;

    using ThisThreadBlock = typename Base::ThisThreadBlock;

    static_assert(LoopSched == LoopScheduler::Default,
                  "Non-default loop scheduler is currently not supported");

    // K1 should be Number<...>
    // Gemm0
    static constexpr auto AK0 = Base0::AK0Number;
    static constexpr auto BK0 = Base0::BK0Number;
    static constexpr auto AK1 = Base0::AK1Number;
    static constexpr auto BK1 = Base0::BK1Number;
    // Gemm1
    static constexpr auto B1K0 = Base::BK0Number;
    static constexpr auto B1K1 = Base::BK1Number;

    using GridwiseGemmPipe = remove_cvref_t<
        decltype(GridwiseGemmPipeline_Selector<PipelineVer, NumGemmKPrefetchStage>())>;

    template <typename ABlockDesc_AK0_M_AK1>
    __host__ __device__ static constexpr auto
    MakeGemm0AMmaTileDescriptor_M0_M1_M2_K(const ABlockDesc_AK0_M_AK1&)
    {
        constexpr index_t MWaves = MPerBlock / (MXdlPerWave * MPerXdl);

        return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<MXdlPerWave, MWaves, MPerXdl>(
            ABlockDesc_AK0_M_AK1{});
    }

    template <typename BBlockDesc_BK0_N_BK1>
    __host__ __device__ static constexpr auto
    MakeGemm0BMmaTileDescriptor_N0_N1_N2_K(const BBlockDesc_BK0_N_BK1&)
    {
        constexpr index_t NWaves = NPerBlock / (NXdlPerWave * NPerXdl);

        return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<NXdlPerWave, NWaves, NPerXdl>(
            BBlockDesc_BK0_N_BK1{});
    }

    template <typename ABlockDesc_AK0_M_AK1>
    __host__ __device__ static constexpr auto
    MakeGemm1AMmaTileDescriptor_M0_M1_M2_K(const ABlockDesc_AK0_M_AK1&)
    {
        return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<MXdlPerWave, 1, 1>(ABlockDesc_AK0_M_AK1{});
    }

    template <typename BBlockDesc_BK0_N_BK1>
    __device__ static constexpr auto
    MakeGemm1BMmaTileDescriptor_N0_N1_N2_K(const BBlockDesc_BK0_N_BK1&)
    {
        constexpr index_t Gemm1NWaves = Gemm1NPerBlock / (Gemm1NXdlPerWave * NPerXdl);
        constexpr auto mfma_info      = MfmaSelector<FloatAB, MPerXdl, NPerXdl>::selected_mfma;
        if constexpr(mfma_info.k_per_blk > mfma_info.group_size && mfma_info.num_input_blks > 1)
        {
            constexpr auto KGroup      = mfma_info.k_per_blk / mfma_info.group_size;
            constexpr auto K0PerXdlops = mfma_info.num_input_blks;
            constexpr auto KPerXdlops  = mfma_info.k_per_blk * mfma_info.num_input_blks;
            static_assert(mfma_info.group_size % B1K1 == 0);
            static_assert(Gemm1KPerBlock % KPerXdlops == 0);
            static_assert(B1K0 == BBlockDesc_BK0_N_BK1{}.GetLength(Number<0>{}));
            static_assert(B1K1 == BBlockDesc_BK0_N_BK1{}.GetLength(Number<2>{}));

            constexpr auto K0 = Gemm1KPerBlock / KPerXdlops;
            constexpr auto K1 = KGroup;
            constexpr auto K2 = K0PerXdlops;
            constexpr auto K3 = KPerXdlops / K1 / K2 / B1K1;
            constexpr auto N  = BBlockDesc_BK0_N_BK1{}.GetLength(Number<1>{});

            constexpr auto b1_blockdesc_k0_k1_k2_k3_n_k4 = transform_tensor_descriptor(
                BBlockDesc_BK0_N_BK1{},
                make_tuple(make_unmerge_transform(
                               make_tuple(Number<K0>{}, Number<K1>{}, Number<K2>{}, Number<K3>{})),
                           make_pass_through_transform(N),
                           make_pass_through_transform(B1K1)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0, 1, 2, 3>{}, Sequence<4>{}, Sequence<5>{}));

            constexpr auto b1_permute_blockdesc_k0_n_k1 = transform_tensor_descriptor(
                b1_blockdesc_k0_k1_k2_k3_n_k4,
                make_tuple(make_merge_transform_v3_division_mod(
                               make_tuple(Number<K0>{}, Number<K2>{}, Number<K1>{}, Number<K3>{})),
                           make_pass_through_transform(N),
                           make_pass_through_transform(B1K1)),
                make_tuple(Sequence<0, 2, 1, 3>{}, Sequence<4>{}, Sequence<5>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<Gemm1NXdlPerWave, Gemm1NWaves, NPerXdl>(
                b1_permute_blockdesc_k0_n_k1);
        }
        else
        {
            return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<Gemm1NXdlPerWave, Gemm1NWaves, NPerXdl>(
                BBlockDesc_BK0_N_BK1{});
        }
    }

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        const index_t gemm0_bytes_end = (SharedMemTrait::a_block_space_size_aligned +
                                         SharedMemTrait::b_block_space_size_aligned) *
                                        sizeof(FloatAB);
        const index_t gemm1_bytes_end =
            (SharedMemTrait::b1_block_space_offset + SharedMemTrait::b1_block_space_size_aligned) *
            sizeof(FloatAB);
        const index_t c_block_bytes_end =
            SharedMemTrait::c_block_space_size * sizeof(FloatCShuffle);

        return math::max(gemm0_bytes_end, gemm1_bytes_end, c_block_bytes_end);
    }

    template <
        InMemoryDataOperationEnum CGlobalMemoryDataOperation_ = InMemoryDataOperationEnum::Set>
    __device__ static bool constexpr IsValidCompilationParameter()
    {
        constexpr bool valid = ck::tensor_operation::device::IsValidGemmCompilationParameter<
            BlockSize,
            MPerBlock,
            NPerBlock,
            MPerXdl,
            NPerXdl,
            MXdlPerWave,
            NXdlPerWave,
            FloatC,
            CGlobalMemoryDataOperation>();
        if constexpr(!valid)
        {
            return false;
        }

        return true;
    }

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    template <typename Block2CTileMap>
    __host__ __device__ static constexpr bool
    CheckValidity(const AGridDesc_AK0_M_AK1& a_grid_desc_ak0_m_ak1,
                  const BGridDesc_BK0_N_BK1& b_grid_desc_bk0_n_bk1,
                  const B1GridDesc_BK0_N_BK1& b1_grid_desc_bk0_n_bk1,
                  const CGridDesc_M_N& c_grid_desc_m_n,
                  const Block2CTileMap& block_2_ctile_map)
    {
        static_assert((MPerBlock % (MPerXdl * MXdlPerWave) == 0) &&
                          (NPerBlock % (NXdlPerWave * NPerXdl)) == 0,
                      "Invalid tuning param!");

        const auto M = a_grid_desc_ak0_m_ak1.GetLength(I1);
        const auto N = b_grid_desc_bk0_n_bk1.GetLength(I1);
        const auto K = a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2);
        const auto Gemm1N = b1_grid_desc_bk0_n_bk1.GetLength(I1);

        if(!(M == c_grid_desc_m_n.GetLength(I0) && Gemm1N == c_grid_desc_m_n.GetLength(I1)))
        {
            return false;
        }

        if(!(M % MPerBlock == 0 && N % NPerBlock == 0 && K % KPerBlock == 0 &&
             Gemm1N % Gemm1NPerBlock == 0))
        {
            return false;
        }

        // check gemm0 gridwise gemm pipeline
        const auto num_gemm0_k_loop = K / KPerBlock;
        if(!GridwiseGemmPipe::IsSupported(num_gemm0_k_loop))
        {
            return false;
        }

        // check gemm1 gridwise gemm pipeline
        if(!(NPerBlock % Gemm1KPerBlock == 0))
        {
            return false;
        }

        const auto num_gemm1_k_inner_loop = NPerBlock / Gemm1KPerBlock;
        if(!GridwiseGemmPipe::IsSupported(num_gemm1_k_inner_loop))
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
    MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(const CGridDesc_M_N& c_grid_desc_m_n)
    {
        const auto M = c_grid_desc_m_n.GetLength(I0);
        const auto N = c_grid_desc_m_n.GetLength(I1);

        const auto MBlock = M / MPerBlock;
        const auto NBlock = N / Gemm1NPerBlock;

        const auto c_grid_desc_mblock_mperblock_nblock_nperblock = transform_tensor_descriptor(
            c_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{})),
                       make_unmerge_transform(make_tuple(NBlock, Number<Gemm1NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

        return c_grid_desc_mblock_mperblock_nblock_nperblock;
    }

    // return block_id to C matrix tile idx (m0, n0) mapping
    __host__ __device__ static constexpr auto
    MakeDefaultBlock2CTileMap(const CGridDesc_M_N& c_grid_desc_m_n)
    {
        return BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, Gemm1NPerBlock, CGridDesc_M_N>(
            c_grid_desc_m_n);
    }

    using CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock =
        remove_cvref_t<decltype(MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            CGridDesc_M_N{}))>;

    using DefaultBlock2CTileMap =
        remove_cvref_t<decltype(MakeDefaultBlock2CTileMap(CGridDesc_M_N{}))>;

    struct SharedMemTrait
    {
        // LDS allocation for A and B: be careful of alignment
        static constexpr auto a_block_desc_ak0_m_ak1 =
            Base0::GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1(get_device_arch());
        static constexpr auto b_block_desc_bk0_n_bk1 =
            Base0::GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1(get_device_arch());
        static constexpr auto b1_block_desc_bk0_n_bk1 =
            Base::GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1(get_device_arch());

        static constexpr auto max_lds_align = math::lcm(math::lcm(AK1, BK1), B1K1);

        static constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);
        static constexpr auto b_block_space_size_aligned = math::integer_least_multiple(
            b_block_desc_bk0_n_bk1.GetElementSpaceSize(), max_lds_align);
        static constexpr auto b1_block_space_size_aligned = math::integer_least_multiple(
            b1_block_desc_bk0_n_bk1.GetElementSpaceSize(), max_lds_align);

        static constexpr auto a_block_space_offset  = 0;
        static constexpr auto b_block_space_offset  = a_block_space_size_aligned.value;
        static constexpr auto b1_block_space_offset = 0;

        // LDS allocation for C shuffle in LDS
        static constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            Base::GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(get_device_arch());
        static constexpr auto c_block_space_size =
            c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize();
    };

    template <bool HasMainKBlockLoop, typename Block2CTileMap>
    __device__ static void Run(const FloatAB* __restrict__ p_a_grid,
                               const FloatAB* __restrict__ p_b_grid,
                               const FloatAB* __restrict__ p_b1_grid,
                               FloatC* __restrict__ p_c_grid,
                               void* __restrict__ p_shared,
                               const AElementwiseOperation& a_element_op,
                               const BElementwiseOperation& b_element_op,
                               const AccElementwiseOperation& acc_element_op,
                               const B1ElementwiseOperation& b1_element_op,
                               const CElementwiseOperation& c_element_op,
                               const AGridDesc_AK0_M_AK1& a_grid_desc_ak0_m_ak1,
                               const BGridDesc_BK0_N_BK1& b_grid_desc_bk0_n_bk1,
                               const B1GridDesc_BK0_N_BK1& b1_grid_desc_bk0_n_bk1,
                               const CGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock&
                                   c_grid_desc_mblock_mperblock_nblock_nperblock,
                               const Block2CTileMap& block_2_ctile_map)
    {
        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc_ak0_m_ak1.GetElementSpaceSize());
        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_grid_desc_bk0_n_bk1.GetElementSpaceSize());
        const auto b1_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b1_grid, b1_grid_desc_bk0_n_bk1.GetElementSpaceSize());

        // divide block work by [M, N]
        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        if(!block_2_ctile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I0),
                          c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I2))))
        {
            return;
        }

        // HACK: this force m/n_block_data_idx_on_grid into SGPR
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * MPerBlock);

        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * Gemm1NPerBlock);

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_ak0_m_ak1 =
            Base0::GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1(get_device_arch());

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_bk0_n_bk1 =
            Base0::GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1(get_device_arch());

        //
        // set up Gemm0
        //

        // A matrix blockwise copy
        auto a_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                AElementwiseOperation,
                                                tensor_operation::element_wise::PassThrough,
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
                                                true, // SrcResetCoord
                                                true, // DstResetCoord
                                                NumGemmKPrefetchStage>(
                a_grid_desc_ak0_m_ak1,
                make_multi_index(0, m_block_data_idx_on_grid, 0),
                a_element_op,
                a_block_desc_ak0_m_ak1,
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});

        // B matrix blockwise copy
        auto b_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                BElementwiseOperation,
                                                tensor_operation::element_wise::PassThrough,
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
                                                true, // SrcResetCoord
                                                true, // DstResetCoord
                                                NumGemmKPrefetchStage>(
                b_grid_desc_bk0_n_bk1,
                make_multi_index(0, 0, 0), // will loop over GemmN dimension
                b_element_op,
                b_block_desc_bk0_n_bk1,
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});

        // Fused Gemm+Gemm pipeline
        // for n in N0:
        //   for k in K0:
        //     acc[m][n] += A[m][k] * B0[k][n]
        //   acc1[m][o] += acc[m][n] * B1[n][o]

        // sanity check
        constexpr auto lcm_AK1_BK1 = math::lcm(AK1, BK1);
        constexpr bool is_single_rate_mfma =
            (((is_same<FloatAB, half_t>::value || is_same<FloatAB, bhalf_t>::value) &&
              lcm_AK1_BK1 <= 4) ||
             (is_same<FloatAB, int8_t>::value && lcm_AK1_BK1 <= 8) ||
             ((is_same<FloatAB, f8_t>::value || is_same<FloatAB, bf8_t>::value) &&
#if defined(__gfx125__)
              lcm_AK1_BK1 < 128))
#else
              lcm_AK1_BK1 < 32))
#endif
                ? true
                : false;
        constexpr auto is_scale_mfma = false;
        constexpr index_t KPack      = math::max(
            lcm_AK1_BK1,
            MfmaSelector<FloatAB, MPerXdl, NPerXdl, FloatAB, is_single_rate_mfma, is_scale_mfma>::
                selected_mfma.k_per_blk);

        auto blockwise_gemm = BlockwiseGemmXdlops_v2<
            BlockSize,
            FloatAB,
            FloatGemmAcc,
            decltype(a_block_desc_ak0_m_ak1),
            decltype(b_block_desc_bk0_n_bk1),
            decltype(MakeGemm0AMmaTileDescriptor_M0_M1_M2_K(a_block_desc_ak0_m_ak1)),
            decltype(MakeGemm0BMmaTileDescriptor_N0_N1_N2_K(b_block_desc_bk0_n_bk1)),
            MPerBlock,
            NPerBlock,
            KPerBlock,
            MPerXdl,
            NPerXdl,
            MXdlPerWave,
            NXdlPerWave,
            KPack,
            true>{}; // TransposeC

        auto acc_thread_buf = blockwise_gemm.GetCThreadBuffer();

        // LDS allocation for A and B: be careful of alignment
        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<FloatAB*>(p_shared) + SharedMemTrait::a_block_space_offset,
            a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        auto b_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<FloatAB*>(p_shared) + SharedMemTrait::b_block_space_offset,
            b_block_desc_bk0_n_bk1.GetElementSpaceSize());

        constexpr auto a_block_slice_copy_step = make_multi_index(KPerBlock / AK1, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(KPerBlock / BK1, 0, 0);
        const auto a_block_reset_copy_step =
            make_multi_index(-a_grid_desc_ak0_m_ak1.GetLength(I0), 0, 0);
        const auto b_block_reset_copy_step =
            make_multi_index(-b_grid_desc_bk0_n_bk1.GetLength(I0), NPerBlock, 0);

        // gridwise GEMM pipeline
        // Only supports LoopScheduler::Default
        const auto gridwise_gemm_pipeline = GridwiseGemmPipeline_Selector<PipelineVer,
                                                                          NumGemmKPrefetchStage,
                                                                          LoopScheduler::Default>();

        const index_t num_k_block_main_loop = __builtin_amdgcn_readfirstlane(
            (a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2)) /
            KPerBlock);

        //
        // set up Gemm1
        //

        // Acc matrix threadwise copy: AccVGPR to VGPR and downcast to XDL input data type
        constexpr auto acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4 =
            blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();

        constexpr auto m0 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I0);
        constexpr auto n0 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I1);
        constexpr auto m1 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I2);
        constexpr auto n1 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I3);
        constexpr auto m2 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I4);
        constexpr auto n2 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I5);
        constexpr auto n3 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I6);
        constexpr auto n4 = acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLength(I7);

        constexpr auto b1_block_slice_copy_step = make_multi_index(Gemm1KPerBlock / B1K1, 0, 0);

        // acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4 to acc_thread_desc_k0_m_k1
        // n0_n1_n2_n3 -> k0
        // m0_m1_m2 -> m
        // n4 -> k1
        // NOTE: had to use merge_v3 or will spit out compilation errors
        constexpr auto acc_thread_desc_k0_m_k1 = transform_tensor_descriptor(
            acc_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4,
            make_tuple(make_merge_transform_v3_division_mod(make_tuple(n0, n1, n2, n3)),
                       make_merge_transform_v3_division_mod(make_tuple(m0, m1, m2)),
                       make_pass_through_transform(n4)),
            make_tuple(Sequence<1, 3, 5, 6>{}, Sequence<0, 2, 4>{}, Sequence<7>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        // A1 matrix in AccVGPR
        // N2 num_groups_per_blk, N3 num_input_blks, N4 group_size
        constexpr auto AccN3 =
            blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLength(I6);

        constexpr auto A1ThreadSlice_K0_M_K1 =
            make_tuple(Number<Gemm1KPerBlock / n4 / AccN3>{}, Number<m0 * m1 * m2>{}, Number<n4>{});

        constexpr auto A1ThreadSliceK0 = A1ThreadSlice_K0_M_K1[I0];
        constexpr auto A1ThreadSliceM  = A1ThreadSlice_K0_M_K1[I1];
        constexpr auto A1ThreadSliceK1 = A1ThreadSlice_K0_M_K1[I2];

#if defined(__gfx11__)
        constexpr auto a1_thread_desc_k0_m_k1 = make_naive_tensor_descriptor_packed(
            make_tuple(A1ThreadSliceK0, A1ThreadSliceM, Number<A1ThreadSliceK1 * 2>{}));
        auto a1_blockwise_copy = ThreadwiseTensorSliceTransfer_StaticToStatic_InterRow<
            FloatGemmAcc,
            FloatAB,
            decltype(acc_thread_desc_k0_m_k1),
            decltype(a1_thread_desc_k0_m_k1),
            decltype(acc_element_op),
            Sequence<A1ThreadSliceK0, A1ThreadSliceM, A1ThreadSliceK1>,
            Sequence<1, 0, 2>,
            2,
            n4,
            0x76543210,
            0xfedcba98,
            true>{make_tuple(0, 0, 0)};
#else
        constexpr auto a1_thread_desc_k0_m_k1 = make_naive_tensor_descriptor(
            A1ThreadSlice_K0_M_K1,
            make_tuple(A1ThreadSliceM * A1ThreadSliceK1, A1ThreadSliceK1, I1));
        // A1 matrix blockwise copy
        auto a1_blockwise_copy = ThreadwiseTensorSliceTransfer_StaticToStatic<
            FloatGemmAcc,
            FloatAB,
            decltype(acc_thread_desc_k0_m_k1),
            decltype(a1_thread_desc_k0_m_k1),
            decltype(acc_element_op),
            Sequence<A1ThreadSliceK0, A1ThreadSliceM, A1ThreadSliceK1>,
            Sequence<1, 0, 2>,
            2,
            n4>{acc_element_op};
#endif
        // B1 matrix in LDS memory, dst of blockwise copy
        constexpr auto b1_block_desc_bk0_n_bk1 =
            Base::GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1(get_device_arch());

        // B1 matrix blockwise copy
        auto b1_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                BElementwiseOperation,
                                                tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<B1K0, Gemm1NPerBlock, B1K1>,
                                                B1BlockTransferThreadClusterLengths_BK0_N_BK1,
                                                B1BlockTransferThreadClusterArrangeOrder,
                                                FloatAB,
                                                FloatAB,
                                                decltype(b1_grid_desc_bk0_n_bk1),
                                                decltype(b1_block_desc_bk0_n_bk1),
                                                B1BlockTransferSrcAccessOrder,
                                                Sequence<1, 0, 2>,
                                                B1BlockTransferSrcVectorDim,
                                                2,
                                                B1BlockTransferSrcScalarPerVector,
                                                B1BlockTransferDstScalarPerVector_BK1,
                                                1,
                                                1,
                                                B1ThreadTransferSrcResetCoordinateAfterRun,
                                                true, // DstResetCoord
                                                NumGemmKPrefetchStage>(
                b1_grid_desc_bk0_n_bk1,
                make_multi_index(0, n_block_data_idx_on_grid, 0),
                b1_element_op,
                b1_block_desc_bk0_n_bk1,
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});

        auto a1_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatAB>(
            a1_thread_desc_k0_m_k1.GetElementSpaceSize());

        // reuse LDS space for gemm0's b_block_buf
        auto b1_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<FloatAB*>(p_shared) + SharedMemTrait::b1_block_space_offset,
            b1_block_desc_bk0_n_bk1.GetElementSpaceSize());

        // selected_mfma.group_size or B1K1 <= Gemm1KPack <= selected_mfma.group_size
        // selected_mfma.k_per_blk <= Gemm1KPack
        //
        // Following similar rationale behind Gemm0KPack, let Gemm1KPack be the lowest common
        // multiples of A1K1 (predetermined by selected_mfma.group_size) and B1K1. But in this case
        // Gemm1KPack can't be higher than A1K1 itself because A1 matrix is distributed in VGPRs
        // with 'group_size' amount of contiguous elements. Having Gemm1KPack greater than A1K1 will
        // cause mismatch in summation index for example c[0:7] = a1[[0:3, 8:11]] * b1[0:7].
        // therefore we may just as well assign Gemm1KPack = group_size
#if defined(__gfx11__)
        constexpr index_t Gemm1KPack =
            MfmaSelector<FloatAB, MPerXdl, NPerXdl>::selected_mfma.group_size * 2;
        static_assert(
            XdlopsGemm<FloatAB, MPerXdl, NPerXdl, Gemm1KPack, FloatAB, false>{}.K0PerXdlops == 1);
#else
        constexpr auto mfma_info     = MfmaSelector<FloatAB, MPerXdl, NPerXdl>::selected_mfma;
        constexpr index_t Gemm1KPack = math::max(mfma_info.k_per_blk, mfma_info.group_size);
#endif
        auto gemm1_blockwise_gemm = BlockwiseGemmXdlops_v2<
            BlockSize,
            FloatAB,
            FloatGemmAcc,
            decltype(a1_thread_desc_k0_m_k1),
            decltype(b1_block_desc_bk0_n_bk1),
            decltype(MakeGemm1AMmaTileDescriptor_M0_M1_M2_K(a1_thread_desc_k0_m_k1)),
            decltype(MakeGemm1BMmaTileDescriptor_N0_N1_N2_K(b1_block_desc_bk0_n_bk1)),
            MPerBlock,
            Gemm1NPerBlock,
            Gemm1KPerBlock,
            MPerXdl,
            NPerXdl,
            MXdlPerWave,
            Gemm1NXdlPerWave,
            Gemm1KPack,
            false,      // TransposeC
            Gemm1KPack, // AMmaKStride
            Gemm1KPack *
                XdlopsGemm<FloatAB, MPerXdl, NPerXdl, Gemm1KPack, FloatAB, false>{}.K0PerXdlops>{
            // BMmaKStride
            make_tuple(0, 0, 0, 0)}; // A_origin

        auto c_thread_buf = gemm1_blockwise_gemm.GetCThreadBuffer();

        const index_t num_gemm1_k_block_outer_loop =
            b_grid_desc_bk0_n_bk1.GetLength(I1) / NPerBlock;
        constexpr index_t num_gemm1_k_block_inner_loop = NPerBlock / Gemm1KPerBlock;

        // Initialize C
        c_thread_buf.Clear();

        // gemm1 K loop
        index_t gemm1_k_block_outer_index = 0;
        do
        {
            // gemm0
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
                                                                   acc_thread_buf,
                                                                   num_k_block_main_loop);
            // gemm1
            {
                // TODO: explore using dynamic buffer for a1 thread buffer
                // For a1_blockwise_copy, the goal is to satisfy pipeline requirements RunRead(),
                // RunWrite(), and MoveSliceWindow(). But it is impossible to implement given that
                // the A1 source buffer is static buffer holding the output of first GEMM and
                // requires constexpr offset by design. Therefore, we pass tensor coordinate offset
                // explicitly in Run() below.

                // preload data into LDS
                b1_blockwise_copy.RunRead(b1_grid_desc_bk0_n_bk1, b1_grid_buf);

                b1_blockwise_copy.MoveSrcSliceWindow(b1_grid_desc_bk0_n_bk1,
                                                     b1_block_slice_copy_step);

                block_sync_lds(); // wait for gemm0 LDS read

                b1_blockwise_copy.RunWrite(b1_block_desc_bk0_n_bk1, b1_block_buf);

                // main body
                if constexpr(num_gemm1_k_block_inner_loop > 1)
                {
                    static_for<0, num_gemm1_k_block_inner_loop - 1, 1>{}([&](auto i) {
                        a1_blockwise_copy.Run(acc_thread_desc_k0_m_k1,
                                              make_tuple(Number<i * A1ThreadSliceK0>{}, I0, I0),
                                              acc_thread_buf,
                                              a1_thread_desc_k0_m_k1,
                                              make_tuple(I0, I0, I0),
                                              a1_thread_buf);

                        b1_blockwise_copy.RunRead(b1_grid_desc_bk0_n_bk1, b1_grid_buf);

                        block_sync_lds();

                        gemm1_blockwise_gemm.Run(a1_thread_buf, b1_block_buf, c_thread_buf);

                        block_sync_lds();

                        b1_blockwise_copy.MoveSrcSliceWindow(b1_grid_desc_bk0_n_bk1,
                                                             b1_block_slice_copy_step);

                        b1_blockwise_copy.RunWrite(b1_block_desc_bk0_n_bk1, b1_block_buf);
                    });
                }
                // tail
                {
                    a1_blockwise_copy.Run(
                        acc_thread_desc_k0_m_k1,
                        make_tuple(
                            Number<(num_gemm1_k_block_inner_loop - 1) * A1ThreadSliceK0>{}, I0, I0),
                        acc_thread_buf,
                        a1_thread_desc_k0_m_k1,
                        make_tuple(I0, I0, I0),
                        a1_thread_buf);

                    block_sync_lds();

                    gemm1_blockwise_gemm.Run(a1_thread_buf, b1_block_buf, c_thread_buf);
                }
            } // end gemm1

            a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc_ak0_m_ak1,
                                                a_block_reset_copy_step); // rewind K
            b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc_bk0_n_bk1,
                                                b_block_reset_copy_step); // rewind K and step N

            block_sync_lds(); // wait for gemm1 LDS read
        } while(++gemm1_k_block_outer_index < num_gemm1_k_block_outer_loop); // end j loop

        // shuffle C and write out
        Base::template RunEpilogue<CGlobalMemoryDataOperation, false, false>(
            gemm1_blockwise_gemm,
            c_grid_desc_mblock_mperblock_nblock_nperblock,
            c_thread_buf,
            block_work_idx[I0],
            block_work_idx[I1],
            p_shared,
            p_c_grid,
            c_element_op);
    }
};

} // namespace ck
