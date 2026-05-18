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
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_common.hpp"

namespace ck {

// GEMM:
//   input : A[M, K]
//   input : B[N, K]
//   input : D0[M, N], D1[M, N], ...
//   output : E[M, N]
//   C = a_op(A) * b_op(B)
//   E = cde_op(C, D0, D1, ...)
// Assume:
//   D0, D1, ... and E have the same layout
template <typename ADataType, // FIXME: don't assume A/B have same datatype
          typename BDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename EDataType,
          typename ComputeType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
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
          typename ABlockTransferThreadClusterLengths_KBatch_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_KBatch_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          index_t BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEShuffleBlockTransferScalarPerVector_NPerBlock,
          LoopScheduler LoopSched,
          PipelineVersion PipelineVer = PipelineVersion::v1>
struct GridwiseGemmMultipleD_xdl_splitk_cshuffle
    : public GridwiseGemm_xdl_cshuffle_base<
          tensor_layout::gemm::RowMajor,
          tensor_layout::gemm::ColumnMajor,
          tensor_layout::gemm::RowMajor,
          ADataType,
          BDataType,
          AccDataType,
          CShuffleDataType,
          DsDataType,
          EDataType,
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
          BThreadTransferSrcResetCoordinateAfterRun,
          BBlockLdsExtraN,
          CShuffleMXdlPerWavePerShuffle,
          CShuffleNXdlPerWavePerShuffle,
          CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          Sequence<CDEShuffleBlockTransferScalarPerVector_NPerBlock>,
          AComputeDataType_,
          BComputeDataType_,
          true> // ForceNaiveLayout
{
    using Base = GridwiseGemm_xdl_cshuffle_base<
        tensor_layout::gemm::RowMajor,
        tensor_layout::gemm::ColumnMajor,
        tensor_layout::gemm::RowMajor,
        ADataType,
        BDataType,
        AccDataType,
        CShuffleDataType,
        DsDataType,
        EDataType,
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
        BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        Sequence<CDEShuffleBlockTransferScalarPerVector_NPerBlock>,
        AComputeDataType_,
        BComputeDataType_,
        true>; // ForceNaiveLayout

    using Base::GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1;
    using Base::GetABlockDescriptor_AKB_AK0PerBlock_MPerBlock_AK1;
    using Base::GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1;
    using Base::GetBBlockDescriptor_BKB_BK0PerBlock_NPerBlock_BK1;
    using Base::I0;
    using Base::I1;
    using Base::I2;
    using ThisThreadBlock               = typename Base::ThisThreadBlock;
    static constexpr index_t NumDTensor = DsDataType::Size();

    using GemmSpecialization = ck::tensor_operation::device::GemmSpecialization;

    // K1 should be Number<...>
    static constexpr auto AK1         = Base::AK1Number;
    static constexpr auto BK1         = Base::BK1Number;
    static constexpr auto AK0PerBlock = Base::AK0Number;
    static constexpr auto BK0PerBlock = Base::BK0Number;

    using GridwiseGemmPipe = remove_cvref_t<
        decltype(GridwiseGemmPipeline_Selector<PipelineVer, NumGemmKPrefetchStage, LoopSched>())>;

    // ck::Tuple<const D0DataType*, const D1DataType*, ...>
    static constexpr auto MakeDsGridPointer()
    {
        return generate_tuple(
            [&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                return static_cast<const DDataType*>(nullptr);
            },
            Number<NumDTensor>{});
    }

    __host__ __device__ static auto CalculateMPadded(index_t M)
    {
        return math::integer_least_multiple(M, MPerBlock);
    }

    __host__ __device__ static auto CalculateNPadded(index_t N)
    {
        return math::integer_least_multiple(N, NPerBlock);
    }

    __host__ __device__ static auto CalculateKPadded(index_t K, index_t K_Batch)
    {
        return math::integer_least_multiple(K, KPerBlock * K_Batch);
    }

    template <typename ALayout, GemmSpecialization GemmSpec>
    __host__ __device__ static auto
    MakeAGridDescriptor_KBatch_AK0_M_AK1(index_t M, index_t K, index_t StrideA, index_t KBatch)
    {
        const auto a_grid_desc_m_k = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(StrideA, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ALayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(I1, StrideA));
            }
        }();

        const auto MPad = CalculateMPadded(M);
        const auto KPad = CalculateKPadded(K, KBatch);

        const auto a_grid_desc_m_kpad = transform_tensor_descriptor(
            a_grid_desc_m_k,
            make_tuple(make_pass_through_transform(M), make_right_pad_transform(K, KPad - K)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        const auto AK0 = KPad / (KBatch * AK1);

        if constexpr(GemmSpec == tensor_operation::device::GemmSpecialization::MPadding ||
                     GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding ||
                     GemmSpec == tensor_operation::device::GemmSpecialization::MKPadding ||
                     GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding)
        {
            // const auto PadM = (MPerBlock - M % MPerBlock) % MPerBlock;
            return transform_tensor_descriptor(
                a_grid_desc_m_kpad,
                make_tuple(make_unmerge_transform(make_tuple(KBatch, AK0, AK1)),
                           make_right_pad_transform(M, MPad - M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                a_grid_desc_m_kpad,
                make_tuple(make_unmerge_transform(make_tuple(KBatch, AK0, AK1)),
                           make_pass_through_transform(M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));
        }
    }

    template <typename BLayout, GemmSpecialization GemmSpec>
    __host__ __device__ static auto
    MakeBGridDescriptor_KBatch_BK0_N_BK1(index_t K, index_t N, index_t StrideB, index_t KBatch)
    {
        const auto b_grid_desc_k_n = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(K, N), make_tuple(StrideB, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(K, N), make_tuple(I1, StrideB));
            }
        }();

        const auto NPad = CalculateNPadded(N);
        const auto KPad = CalculateKPadded(K, KBatch);

        const auto b_grid_desc_kpad_n = transform_tensor_descriptor(
            b_grid_desc_k_n,
            make_tuple(make_right_pad_transform(K, KPad - K), make_pass_through_transform(N)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        const auto BK0 = KPad / (KBatch * BK1);

        if constexpr(GemmSpec == tensor_operation::device::GemmSpecialization::NPadding ||
                     GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding ||
                     GemmSpec == tensor_operation::device::GemmSpecialization::NKPadding ||
                     GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding)
        {
            // const auto PadN = (NPerBlock - N % NPerBlock) % NPerBlock;
            return transform_tensor_descriptor(
                b_grid_desc_kpad_n,
                make_tuple(make_unmerge_transform(make_tuple(KBatch, BK0, BK1)),
                           make_right_pad_transform(N, NPad - N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                b_grid_desc_kpad_n,
                make_tuple(make_unmerge_transform(make_tuple(KBatch, BK0, BK1)),
                           make_pass_through_transform(N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));
        }
    }

    // E desc for destination in blockwise copy
    template <typename EGridDesc_M_N>
    __host__ __device__ static constexpr auto
    MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(const EGridDesc_M_N& e_grid_desc_m_n)
    {
        const auto M = e_grid_desc_m_n.GetLength(I0);
        const auto N = e_grid_desc_m_n.GetLength(I1);

        const auto MBlock = M / MPerBlock;
        const auto NBlock = N / NPerBlock;

        const auto e_grid_desc_mblock_mperblock_nblock_nperblock = transform_tensor_descriptor(
            e_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{})),
                       make_unmerge_transform(make_tuple(NBlock, Number<NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

        return e_grid_desc_mblock_mperblock_nblock_nperblock;
    }

    // Ds desc for source in blockwise copy
    template <typename DsGridDesc_M_N>
    __host__ __device__ static constexpr auto
    MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(const DsGridDesc_M_N& ds_grid_desc_m_n)
    {
        return generate_tuple(
            [&](auto i) {
                return MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(ds_grid_desc_m_n[i]);
            },
            Number<NumDTensor>{});
    }

    // return block_id to E matrix tile idx (m0, n0) mapping
    template <typename EGridDesc_M_N>
    __host__ __device__ static constexpr auto
    MakeDefaultBlock2ETileMap(const EGridDesc_M_N& e_grid_desc_m_n)
    {
        return BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock, EGridDesc_M_N>(
            e_grid_desc_m_n);
    }

    IS_VALID_COMPILATION_PARAMETER_IMPL(EDataType)

    template <typename ALayout,
              typename BLayout,
              typename DsLayout,
              typename ELayout,
              GemmSpecialization GemmSpec>
    __host__ __device__ static constexpr bool
    CheckValidity(const index_t M,
                  const index_t N,
                  const index_t K,
                  const index_t StrideA,
                  const index_t StrideB,
                  const std::array<index_t, NumDTensor> StrideDs,
                  const index_t StrideE,
                  const index_t KBatch)
    {
        const auto a_grid_desc_kbatch_ak0_m_ak1 =
            MakeAGridDescriptor_KBatch_AK0_M_AK1<ALayout, GemmSpec>(M, K, StrideA, KBatch);
        const auto b_grid_desc_kbatch_bk0_n_bk1 =
            MakeBGridDescriptor_KBatch_BK0_N_BK1<BLayout, GemmSpec>(K, N, StrideB, KBatch);

        ignore = StrideDs;

        const auto e_grid_desc_m_n = MakeEGridDescriptor_M_N<ELayout, GemmSpec>(M, N, StrideE);

#if 0
        // check tile size
        if(!(M % MPerBlock == 0 && N % NPerBlock == 0 && K % KPerBlock == 0))
        {
            return false;
        }
#endif

        // check gridwise gemm pipeline
        const auto num_k_loop = K / KPerBlock;

        if(!GridwiseGemmPipe::IsSupported(num_k_loop))
        {
            return false;
        }

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        // check tensor size: cannot be larger than 2GB each
        constexpr long_index_t TwoGB = (long_index_t{1} << 31);

        if(!(a_grid_desc_kbatch_ak0_m_ak1.GetElementSpaceSize() * sizeof(ADataType) <= TwoGB &&
             b_grid_desc_kbatch_bk0_n_bk1.GetElementSpaceSize() * sizeof(BDataType) <= TwoGB &&
             e_grid_desc_m_n.GetElementSpaceSize() * sizeof(EDataType) <= TwoGB))
        {
            return false;
        }

        return true;
    }

    __host__ __device__ static constexpr bool CalculateHasMainKBlockLoop(index_t K)
    {
        const index_t num_loop = K / KPerBlock;

        return GridwiseGemmPipe::CalculateHasMainLoop(num_loop);
    }

    using DsGridPointer = decltype(MakeDsGridPointer());

    template <typename ELayout, GemmSpecialization GemmSpec>
    __host__ __device__ static auto
    MakeEGridDescriptor_M_N(index_t MRaw, index_t NRaw, index_t StrideE)
    {
        constexpr auto matrix_padder =
            ck::tensor_operation::device::MatrixPadder<GemmSpec, index_t, index_t, index_t>{
                MPerBlock, NPerBlock, KPerBlock};
        const auto e_grid_desc_mraw_nraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ELayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(StrideE, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ELayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(I1, StrideE));
            }
        }();

        return matrix_padder.PadCDescriptor_M_N(e_grid_desc_mraw_nraw);
    }

    template <typename DsLayout, GemmSpecialization GemmSpec>
    __host__ __device__ static auto
    MakeDsGridDescriptor_M_N(const std::array<index_t, NumDTensor>& MRaws,
                             const std::array<index_t, NumDTensor>& NRaws,
                             const std::array<index_t, NumDTensor>& DsStride)
    {
        return generate_tuple(
            [&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

                return MakeEGridDescriptor_M_N<DLayout, GemmSpec>(MRaws[i], NRaws[i], DsStride[i]);
            },
            Number<NumDTensor>{});
    }

    __device__ __host__ static constexpr auto GetMPerBlock() { return MPerBlock; }

    template <bool HasMainKBlockLoop,
              InMemoryDataOperationEnum EGlobalMemoryDataOperation,
              index_t NumDTensor_,
              typename DsDataType_,
              typename AGridDesc_KBatch_AK0_M_AK1,
              typename BGridDesc_KBatch_BK0_N_BK1,
              typename DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
              typename EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
              typename CDEElementwiseOperation_,
              typename Block2ETileMap>
    __device__ static void Run(const ADataType* __restrict__ p_a_grid,
                               const BDataType* __restrict__ p_b_grid,
                               DsGridPointer p_ds_grid,
                               EDataType* __restrict__ p_e_grid,
                               void* __restrict__ p_shared,
                               uint32_t* barrier_count_finished,
                               const index_t KBatch,
                               const AElementwiseOperation& a_element_op,
                               const BElementwiseOperation& b_element_op,
                               const CDEElementwiseOperation_& cde_element_op,
                               const AGridDesc_KBatch_AK0_M_AK1& a_grid_desc_kbatch_ak0_m_ak1,
                               const BGridDesc_KBatch_BK0_N_BK1& b_grid_desc_kbatch_bk0_n_bk1,
                               const DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock&
                                   ds_grid_desc_mblock_mperblock_nblock_nperblock,
                               const EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock&
                                   e_grid_desc_mblock_mperblock_nblock_nperblock,
                               const Block2ETileMap& block_2_etile_map)
    {
        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc_kbatch_ak0_m_ak1.GetElementSpaceSize());

        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_grid_desc_kbatch_bk0_n_bk1.GetElementSpaceSize());

        const auto ds_grid_buf = generate_tuple(
            [&](auto i) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_ds_grid[i],
                    ds_grid_desc_mblock_mperblock_nblock_nperblock[i].GetElementSpaceSize());
            },
            Number<NumDTensor_>{});

        auto e_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_e_grid, e_grid_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

        // divide block work by [M, N]
        const auto block_work_idx =
            block_2_etile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        // HACK: this force m/n_block_data_idx_on_grid into SGPR
        const index_t kbatch_id = __builtin_amdgcn_readfirstlane(block_work_idx[I0]);

        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * MPerBlock);

        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I2] * NPerBlock);

        // lds max alignment
        constexpr auto max_lds_align = math::lcm(AK1, BK1);

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_ak0_m_ak1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1(get_device_arch());

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_bk0_n_bk1 =
            GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1(get_device_arch());

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_kbatch_ak0_m_ak1 =
            GetABlockDescriptor_AKB_AK0PerBlock_MPerBlock_AK1(a_block_desc_ak0_m_ak1);

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_kbatch_bk0_n_bk1 =
            GetBBlockDescriptor_BKB_BK0PerBlock_NPerBlock_BK1(b_block_desc_bk0_n_bk1);

        // A matrix blockwise copy
        auto a_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                AElementwiseOperation,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<1, AK0PerBlock, MPerBlock, AK1>,
                                                ABlockTransferThreadClusterLengths_KBatch_AK0_M_AK1,
                                                ABlockTransferThreadClusterArrangeOrder,
                                                ADataType,
                                                ComputeType,
                                                decltype(a_grid_desc_kbatch_ak0_m_ak1),
                                                decltype(a_block_desc_kbatch_ak0_m_ak1),
                                                ABlockTransferSrcAccessOrder,
                                                Sequence<2, 0, 1, 3>,
                                                ABlockTransferSrcVectorDim,
                                                3,
                                                ABlockTransferSrcScalarPerVector,
                                                ABlockTransferDstScalarPerVector_AK1,
                                                1,
                                                1,
                                                AThreadTransferSrcResetCoordinateAfterRun,
                                                true,
                                                NumGemmKPrefetchStage>(
                a_grid_desc_kbatch_ak0_m_ak1,
                make_multi_index(kbatch_id, 0, m_block_data_idx_on_grid, 0),
                a_element_op,
                a_block_desc_kbatch_ak0_m_ak1,
                make_multi_index(0, 0, 0, 0),
                ck::tensor_operation::element_wise::PassThrough{});

        // B matrix blockwise copy
        auto b_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                BElementwiseOperation,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<1, BK0PerBlock, NPerBlock, BK1>,
                                                BBlockTransferThreadClusterLengths_KBatch_BK0_N_BK1,
                                                BBlockTransferThreadClusterArrangeOrder,
                                                BDataType,
                                                ComputeType,
                                                decltype(b_grid_desc_kbatch_bk0_n_bk1),
                                                decltype(b_block_desc_kbatch_bk0_n_bk1),
                                                BBlockTransferSrcAccessOrder,
                                                Sequence<2, 0, 1, 3>,
                                                BBlockTransferSrcVectorDim,
                                                3,
                                                BBlockTransferSrcScalarPerVector,
                                                BBlockTransferDstScalarPerVector_BK1,
                                                1,
                                                1,
                                                BThreadTransferSrcResetCoordinateAfterRun,
                                                true,
                                                NumGemmKPrefetchStage>(
                b_grid_desc_kbatch_bk0_n_bk1,
                make_multi_index(kbatch_id, 0, n_block_data_idx_on_grid, 0),
                b_element_op,
                b_block_desc_kbatch_bk0_n_bk1,
                make_multi_index(0, 0, 0, 0),
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
            (((is_same<ComputeType, half_t>::value || is_same<ComputeType, bhalf_t>::value) &&
              lcm_AK1_BK1 <= 4) ||
             (is_same<ComputeType, int8_t>::value && lcm_AK1_BK1 <= 8) ||
             ((is_same<ComputeType, f8_t>::value || is_same<ComputeType, bf8_t>::value) &&
#if defined(__gfx125__)
              lcm_AK1_BK1 < 128))
#else
              lcm_AK1_BK1 < 32))
#endif
                ? true
                : false;
        constexpr auto is_scale_mfma = false;
        constexpr index_t KPack      = math::max(lcm_AK1_BK1,
                                            MfmaSelector<ComputeType,
                                                              MPerXdl,
                                                              NPerXdl,
                                                              ComputeType,
                                                              is_single_rate_mfma,
                                                              is_scale_mfma>::selected_mfma.k_per_blk);

        auto blockwise_gemm = BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_Selector<
            BlockSize,
            ComputeType,
            AccDataType,
            decltype(a_block_desc_ak0_m_ak1),
            decltype(b_block_desc_bk0_n_bk1),
            MPerXdl,
            NPerXdl,
            MXdlPerWave,
            NXdlPerWave,
            KPack,
            LoopSched>();

#if 1
        if(block_work_idx[I0] == 0)
        {
            const index_t nThreadSize = CDEShuffleBlockTransferScalarPerVector_NPerBlock;
            const index_t numNThreads = NPerBlock / nThreadSize;
            const index_t numMThreads = BlockSize / numNThreads;
            const index_t mThreadSize = MPerBlock / numMThreads;

            const index_t m_tid = get_thread_local_1d_id() / numNThreads;
            const index_t n_tid = get_thread_local_1d_id() % numNThreads;

            auto c_thread_desc_mblock_mperblock_nblock_nperblock =
                make_naive_tensor_descriptor_packed(
                    make_tuple(I1, Number<mThreadSize>{}, I1, Number<nThreadSize>{}));

            StaticBuffer<AddressSpaceEnum::Vgpr,
                         EDataType,
                         c_thread_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize(),
                         true>
                e_thread_zero_buf;

            auto c_thread_copy = ThreadwiseTensorSliceTransfer_v1r3<
                EDataType,
                EDataType,
                decltype(c_thread_desc_mblock_mperblock_nblock_nperblock),
                decltype(e_grid_desc_mblock_mperblock_nblock_nperblock),
                ck::tensor_operation::element_wise::PassThrough,
                Sequence<1, mThreadSize, 1, nThreadSize>,
                Sequence<0, 1, 2, 3>,
                3,
                CDEShuffleBlockTransferScalarPerVector_NPerBlock,
                InMemoryDataOperationEnum::Set,
                1,
                true>{e_grid_desc_mblock_mperblock_nblock_nperblock,
                      make_multi_index(block_work_idx[I1],
                                       m_tid * mThreadSize,
                                       block_work_idx[I2],
                                       n_tid * nThreadSize),
                      ck::tensor_operation::element_wise::PassThrough {}};

            c_thread_copy.Run(c_thread_desc_mblock_mperblock_nblock_nperblock,
                              make_tuple(I0, I0, I0, I0),
                              e_thread_zero_buf,
                              e_grid_desc_mblock_mperblock_nblock_nperblock,
                              e_grid_buf);

            __syncthreads();

            if(threadIdx.x == 0)
            {
                atomicAdd(barrier_count_finished, 1);
            }
        }
#endif

        auto c_thread_buf = blockwise_gemm.GetCThreadBuffer();

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);

        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<ComputeType*>(p_shared), a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        auto b_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<ComputeType*>(p_shared) + a_block_space_size_aligned,
            b_block_desc_bk0_n_bk1.GetElementSpaceSize());

        constexpr auto a_block_slice_copy_step = make_multi_index(0, KPerBlock / AK1, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(0, KPerBlock / BK1, 0, 0);

        // gridwise GEMM pipeline
        const auto gridwise_gemm_pipeline =
            GridwiseGemmPipeline_Selector<PipelineVer, NumGemmKPrefetchStage, LoopSched>();

        const index_t num_k_block_main_loop =
            __builtin_amdgcn_readfirstlane((a_grid_desc_kbatch_ak0_m_ak1.GetLength(I1) *
                                            a_grid_desc_kbatch_ak0_m_ak1.GetLength(I3)) /
                                           KPerBlock);

        gridwise_gemm_pipeline.template Run<HasMainKBlockLoop>(a_grid_desc_kbatch_ak0_m_ak1,
                                                               a_block_desc_kbatch_ak0_m_ak1,
                                                               a_blockwise_copy,
                                                               a_grid_buf,
                                                               a_block_buf,
                                                               a_block_slice_copy_step,
                                                               b_grid_desc_kbatch_bk0_n_bk1,
                                                               b_block_desc_kbatch_bk0_n_bk1,
                                                               b_blockwise_copy,
                                                               b_grid_buf,
                                                               b_block_buf,
                                                               b_block_slice_copy_step,
                                                               blockwise_gemm,
                                                               c_thread_buf,
                                                               num_k_block_main_loop);
        if(threadIdx.x == 0)
        {
            while(__atomic_load_n(barrier_count_finished, __ATOMIC_RELAXED) == 0) {}
        }

        __syncthreads();

        // shuffle C and write out
        Base::template RunMultiDEpilogue<EGlobalMemoryDataOperation, false, false, true>(
            blockwise_gemm,
            ds_grid_desc_mblock_mperblock_nblock_nperblock,
            e_grid_desc_mblock_mperblock_nblock_nperblock,
            c_thread_buf,
            block_work_idx[I1],
            block_work_idx[I2],
            p_shared,
            p_ds_grid,
            p_e_grid,
            cde_element_op);
        if(threadIdx.x == 0)
        {
            index_t k_id_finished_t = atomicAdd(barrier_count_finished, 1);

            if(k_id_finished_t == KBatch)
            {
                *barrier_count_finished = 0;
            }
        }
    }

    template <bool HasMainKBlockLoop,
              InMemoryDataOperationEnum EGlobalMemoryDataOperation,
              GemmSpecialization GemmSpec,
              typename ALayout,
              typename BLayout,
              typename DsLayout,
              typename ELayout,
              typename Block2ETileMap>
    __device__ static void Run(const void* __restrict__ p_a_grid_,
                               const void* __restrict__ p_b_grid_,
                               DsGridPointer p_ds_grid,
                               void* __restrict__ p_e_grid_,
                               void* __restrict__ p_shared,
                               uint32_t* barrier_count_finished,
                               const AElementwiseOperation& a_element_op,
                               const BElementwiseOperation& b_element_op,
                               const CDEElementwiseOperation& cde_element_op,
                               const index_t M,
                               const index_t N,
                               const index_t K,
                               const index_t StrideA,
                               const index_t StrideB,
                               const std::array<index_t, NumDTensor> StrideDs,
                               const index_t StrideE,
                               const index_t KBatch,
                               const Block2ETileMap& block_2_etile_map)
    {
        const auto p_a_grid = reinterpret_cast<const ADataType*>(p_a_grid_);
        const auto p_b_grid = reinterpret_cast<const BDataType*>(p_b_grid_);
        const auto p_e_grid = reinterpret_cast<EDataType*>(p_e_grid_);

        using DsGridDesc_M_N =
            remove_cvref_t<decltype(MakeDsGridDescriptor_M_N<DsLayout, GemmSpec>({}, {}, {}))>;

        DsGridDesc_M_N ds_grid_desc_m_n;

        static_for<0, NumDTensor, 1>{}([&](auto j) {
            using DLayout = remove_cvref_t<tuple_element_t<j.value, DsLayout>>;

            ds_grid_desc_m_n(j) = MakeEGridDescriptor_M_N<DLayout, GemmSpec>(M, N, StrideDs[j]);
        });

        const auto e_grid_desc_m_n = MakeEGridDescriptor_M_N<ELayout, GemmSpec>(M, N, StrideE);

        // tensor descriptors for block/thread-wise copy
        const auto a_grid_desc_kbatch_ak0_m_ak1 =
            MakeAGridDescriptor_KBatch_AK0_M_AK1<ALayout, GemmSpec>(M, K, StrideA, KBatch);

        const auto b_grid_desc_kbatch_bk0_n_bk1 =
            MakeBGridDescriptor_KBatch_BK0_N_BK1<BLayout, GemmSpec>(K, N, StrideB, KBatch);

        using DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock =
            remove_cvref_t<decltype(MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                DsGridDesc_M_N{}))>;

        DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock ds_grid_desc_mblock_mperblock_nblock_nperblock;

        static_for<0, NumDTensor, 1>{}([&](auto j) {
            ds_grid_desc_mblock_mperblock_nblock_nperblock(j) =
                MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(ds_grid_desc_m_n[j]);
        });

        const auto e_grid_desc_mblock_mperblock_nblock_nperblock =
            MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(e_grid_desc_m_n);

        const auto block_work_idx =
            block_2_etile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        const index_t kbatch_id = __builtin_amdgcn_readfirstlane(block_work_idx[I0]);

        if(kbatch_id == KBatch - 1)
        {
            Run<HasMainKBlockLoop, EGlobalMemoryDataOperation, NumDTensor, DsDataType>(
                p_a_grid,
                p_b_grid,
                p_ds_grid,
                p_e_grid,
                p_shared,
                barrier_count_finished,
                KBatch,
                a_element_op,
                b_element_op,
                cde_element_op,
                a_grid_desc_kbatch_ak0_m_ak1,
                b_grid_desc_kbatch_bk0_n_bk1,
                ds_grid_desc_mblock_mperblock_nblock_nperblock,
                e_grid_desc_mblock_mperblock_nblock_nperblock,
                block_2_etile_map);
        }
        else
        {
            Run<HasMainKBlockLoop, EGlobalMemoryDataOperation, 0, Tuple<>>(
                p_a_grid,
                p_b_grid,
                p_ds_grid,
                p_e_grid,
                p_shared,
                barrier_count_finished,
                KBatch,
                a_element_op,
                b_element_op,
                ck::tensor_operation::element_wise::PassThrough{},
                a_grid_desc_kbatch_ak0_m_ak1,
                b_grid_desc_kbatch_bk0_n_bk1,
                ds_grid_desc_mblock_mperblock_nblock_nperblock,
                e_grid_desc_mblock_mperblock_nblock_nperblock,
                block_2_etile_map);
        }
    }
};

} // namespace ck
