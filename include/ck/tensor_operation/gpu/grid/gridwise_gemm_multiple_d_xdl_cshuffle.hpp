// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/env.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
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
template <typename ADataType,
          typename BDataType,
          typename AComputeDataType_,
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename EDataType,
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
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEShuffleBlockTransferScalarPerVector_NPerBlock,
          LoopScheduler LoopSched,
          PipelineVersion PipelineVer      = PipelineVersion::v1,
          typename BComputeDataType_       = AComputeDataType_,
          bool DoElementwiseBeforeCShuffle = false>
struct GridwiseGemmMultipleD_xdl_cshuffle
    : public GridwiseGemm_xdl_cshuffle_base<
          tensor_layout::gemm::RowMajor,
          tensor_layout::gemm::ColumnMajor,
          tensor_layout::gemm::RowMajor,
          AComputeDataType_,
          BComputeDataType_,
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
          true> // ForceNaiveLdsLayout
{
    using Base = GridwiseGemm_xdl_cshuffle_base<
        tensor_layout::gemm::RowMajor,
        tensor_layout::gemm::ColumnMajor,
        tensor_layout::gemm::RowMajor,
        AComputeDataType_,
        BComputeDataType_,
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
        true>; // ForceNaiveLdsLayout

    using Base::GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1;
    using Base::GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1;
    using Base::I0;
    using Base::I1;
    using Base::I2;
    using ThisThreadBlock = typename Base::ThisThreadBlock;

    static constexpr index_t NumDTensor = DsDataType::Size();
    static_assert(!DoElementwiseBeforeCShuffle || NumDTensor == 0);

    using GemmSpecialization = ck::tensor_operation::device::GemmSpecialization;

    // K1 should be Number<...>
    static constexpr auto AK1         = Base::AK1Number;
    static constexpr auto BK1         = Base::BK1Number;
    static constexpr auto AK0PerBlock = Base::AK0Number;
    static constexpr auto BK0PerBlock = Base::BK0Number;

    using GridwiseGemmPipe = remove_cvref_t<
        decltype(GridwiseGemmPipeline_Selector<PipelineVer, NumGemmKPrefetchStage, LoopSched>())>;

#if CK_GFX90A_DENORM_WORKAROUND
    using AComputeDataType =
        conditional_t<is_same_v<AComputeDataType_, ck::half_t>, ck::bhalf_t, AComputeDataType_>;
    using BComputeDataType =
        conditional_t<is_same_v<BComputeDataType_, ck::half_t>, ck::bhalf_t, BComputeDataType_>;
#else
    using AComputeDataType =
        conditional_t<is_same_v<AComputeDataType_, ck::tf32_t>, float, AComputeDataType_>;
    using BComputeDataType =
        conditional_t<is_same_v<BComputeDataType_, ck::tf32_t>, float, BComputeDataType_>;
#endif

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

    // A desc for source in blockwise copy
    template <typename AGridDesc_M_K>
    __host__ __device__ static constexpr auto
    MakeDefaultAGridDescriptor_AK0_M_AK1(const AGridDesc_M_K& a_grid_desc_m_k)
    {
        const auto M = a_grid_desc_m_k.GetLength(I0);
        const auto K = a_grid_desc_m_k.GetLength(I1);

        const auto AK0 = K / AK1;

        return transform_tensor_descriptor(a_grid_desc_m_k,
                                           make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                                                      make_pass_through_transform(M)),
                                           make_tuple(Sequence<1>{}, Sequence<0>{}),
                                           make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    // B desc for source in blockwise copy
    template <typename BGridDesc_N_K>
    __host__ __device__ static constexpr auto
    MakeDefaultBGridDescriptor_BK0_N_BK1(const BGridDesc_N_K& b_grid_desc_n_k)
    {
        const auto N = b_grid_desc_n_k.GetLength(I0);
        const auto K = b_grid_desc_n_k.GetLength(I1);

        const auto BK0 = K / BK1;

        return transform_tensor_descriptor(b_grid_desc_n_k,
                                           make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                                                      make_pass_through_transform(N)),
                                           make_tuple(Sequence<1>{}, Sequence<0>{}),
                                           make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
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

    template <typename ALayout, typename BLayout, typename ELayout>
    __host__ __device__ static bool
    CheckTensorTransfersValidity(index_t MRaw, index_t NRaw, index_t KRaw)
    {
        // Check if the vector dim is K1 or M|N
        const auto A_vector_dim_size = ABlockTransferSrcVectorDim == 2 ? KRaw : MRaw;
        const auto B_vector_dim_size = BBlockTransferSrcVectorDim == 2 ? KRaw : NRaw;
        const auto E_vector_dim_size = NRaw;

        // check vector load for A tensor
        if constexpr(is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
        {
            if(!(A_vector_dim_size == KRaw &&
                 A_vector_dim_size % ABlockTransferSrcScalarPerVector == 0))
                return false;
        }
        else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
        {
            if(!(A_vector_dim_size == MRaw &&
                 A_vector_dim_size % ABlockTransferSrcScalarPerVector == 0))
                return false;
        }
        else
        {
            return false;
        }

        if constexpr(is_same_v<tensor_layout::gemm::RowMajor, BLayout>)
        {
            if(!(B_vector_dim_size == NRaw &&
                 B_vector_dim_size % BBlockTransferSrcScalarPerVector == 0))
                return false;
        }
        else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, BLayout>)
        {
            if(!(B_vector_dim_size == KRaw &&
                 B_vector_dim_size % BBlockTransferSrcScalarPerVector == 0))
                return false;
        }
        else
        {
            return false;
        }

        if constexpr(is_same_v<tensor_layout::gemm::RowMajor, ELayout>)
        {
            if(!(E_vector_dim_size == NRaw &&
                 E_vector_dim_size % CDEShuffleBlockTransferScalarPerVector_NPerBlock == 0))
                return false;
        }
        else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, ELayout>)
        {
            if(!(E_vector_dim_size == NRaw &&
                 CDEShuffleBlockTransferScalarPerVector_NPerBlock == 1))
                return false;
        }
        else
        {
            return false;
        }

        return true;
    }

    template <bool IsGfx11>
    static constexpr index_t GetEstimateVgprCount()
    {
        constexpr index_t MWave    = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t NWave    = NPerBlock / (NXdlPerWave * NPerXdl);
        constexpr index_t WaveSize = BlockSize / (MWave * NWave);

        // VGPR used in LDS loading and WMMA
        constexpr index_t BaseInputVgprCount =
            MPerBlock * KPerBlock / MWave / WaveSize * sizeof(ADataType) / sizeof(uint32_t) +
            NPerBlock * KPerBlock / NWave / WaveSize * sizeof(BDataType) / sizeof(uint32_t);
        // WMMA input is duplicated in GFX11
        constexpr index_t InputVgprCount = IsGfx11 ? BaseInputVgprCount * 2 : BaseInputVgprCount;
        // VGPR used in Accumulator
        constexpr index_t AccVgprCount =
            MPerBlock * NPerBlock / BlockSize * sizeof(AccDataType) / sizeof(uint32_t);

        if constexpr(PipelineVer == PipelineVersion::v1)
        {
            return InputVgprCount + AccVgprCount + BaseInputVgprCount * (NumGemmKPrefetchStage - 1);
        }
        else if constexpr(PipelineVer == PipelineVersion::v2)
        {
            return InputVgprCount + AccVgprCount + BaseInputVgprCount;
        }
        else if constexpr(PipelineVer == PipelineVersion::weight_only)
        {
            return InputVgprCount + AccVgprCount;
        }
        else if constexpr(PipelineVer == PipelineVersion::v4)
        {
            return InputVgprCount * 2 + AccVgprCount;
        }
        else
        {
            // invalid pipeline version
            static_assert(0);
        }
    }

    __host__ static index_t GetSharedMemoryNumberOfByteOnHost()
    {
#if !defined(__HIPCC_RTC__) || !defined(CK_CODE_GEN_RTC)
        if(is_gfx125_supported())
        {
            return Base::GetSharedMemoryNumberOfByte(gfx125_t{});
        }
        else if(ck::get_device_name() == "gfx950")
        {
            return Base::GetSharedMemoryNumberOfByte(gfx950_t{});
        }
        else
#endif
        {
            return Base::GetSharedMemoryNumberOfByte(gfx_invalid_t{});
        }
    }

    template <
        InMemoryDataOperationEnum CGlobalMemoryDataOperation_ = InMemoryDataOperationEnum::Set>
    __device__ static bool constexpr IsValidCompilationParameter()
    {
#if defined(__gfx11__) || defined(__gfx120__)
        if constexpr(is_same_v<AComputeDataType_, float>)
        {

            return false;
        }
        if constexpr(KPerBlock < 16)
        {
            return false;
        }
#endif

#if defined(__gfx125__)
        if constexpr(sizeof(AComputeDataType) == 1)
        {
            if constexpr(KPerBlock % 64)
            {
                return false;
            }
        }
        else if constexpr(sizeof(AComputeDataType) == 2)
        {
            if constexpr(KPerBlock % 32)
            {
                return false;
            }
        }
#endif

        if constexpr(Base::GetSharedMemoryNumberOfByte(get_device_arch()) >
                     get_lds_size(get_device_arch()))
        {
            return false;
        }

        constexpr bool IsGfx11            = is_same_v<decltype(get_device_arch()), gfx11_t>;
        constexpr auto EstimateVgprCount  = GetEstimateVgprCount<IsGfx11>();
        constexpr auto AvailableVgprCount = get_max_vgpr_count(get_device_arch());
        if constexpr(EstimateVgprCount > (AvailableVgprCount + AvailableVgprCount / 4))
        {
            return false;
        }

        return ck::tensor_operation::device::IsValidGemmCompilationParameter<
            BlockSize,
            MPerBlock,
            NPerBlock,
            MPerXdl,
            NPerXdl,
            MXdlPerWave,
            NXdlPerWave,
            EDataType,
            CGlobalMemoryDataOperation_>();
    }

    template <typename AGridDesc_M_K,
              typename BGridDesc_N_K,
              typename DsGridDesc_M_N,
              typename EGridDesc_M_N,
              typename Block2ETileMap>
    __host__ __device__ static constexpr bool CheckValidity(const AGridDesc_M_K& a_grid_desc_m_k,
                                                            const BGridDesc_N_K& b_grid_desc_n_k,
                                                            const DsGridDesc_M_N& ds_grid_desc_m_n,
                                                            const EGridDesc_M_N& e_grid_desc_m_n,
                                                            [[maybe_unused]] const Block2ETileMap&,
                                                            index_t k_batch = 1)
    {
        static_assert((MPerBlock % (MPerXdl * MXdlPerWave) == 0) &&
                          (NPerBlock % (NXdlPerWave * NPerXdl)) == 0,
                      "Invalid tuning param!");

        static_assert(KPerBlock % AK1Value == 0 && KPerBlock % BK1Value == 0,
                      "KPerBlock must be divisible by AK1Value and BK1Value!");

#ifndef __HIPCC_RTC__
        if constexpr(KPerBlock < 16)
        {
            if(ck::is_gfx12_supported() || ck::is_gfx11_supported())
            {
                return false;
            }
        }
#endif

        const auto M  = a_grid_desc_m_k.GetLength(I0);
        const auto N  = b_grid_desc_n_k.GetLength(I0);
        const auto AK = a_grid_desc_m_k.GetLength(I1);
        const auto BK = b_grid_desc_n_k.GetLength(I1);

        // check consistency of desc
        if(!(M == e_grid_desc_m_n.GetLength(I0) && N == e_grid_desc_m_n.GetLength(I1) && AK == BK))
        {
            return false;
        }
        bool valid = true;

        static_for<0, NumDTensor, 1>{}([&](auto i) {
            valid = valid && (M == ds_grid_desc_m_n[i].GetLength(I0) &&
                              N == ds_grid_desc_m_n[i].GetLength(I1));
        });

        if(!valid)
        {
            return false;
        }

        // check tile size
        if(!(M % MPerBlock == 0 && N % NPerBlock == 0 && AK % KPerBlock == 0))
        {
            return false;
        }

        // check gridwise gemm pipeline
        const auto num_k_loop = AK / (KPerBlock * k_batch);
        if(!GridwiseGemmPipe::IsSupported(num_k_loop))
        {
            return false;
        }

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        // check tensor size: cannot be larger than 2GB each
        constexpr long_index_t TwoGB = (long_index_t{1} << 31);

        if(!(a_grid_desc_m_k.GetElementSpaceSize() * sizeof(ADataType) <= TwoGB &&
             b_grid_desc_n_k.GetElementSpaceSize() * sizeof(BDataType) <= TwoGB &&
             e_grid_desc_m_n.GetElementSpaceSize() * sizeof(EDataType) <= TwoGB))
        {
            return false;
        }

#if !defined(__HIPCC_RTC__) || !defined(CK_CODE_GEN_RTC)
        if(!is_xdl_wmma_k_supported<AComputeDataType, KPerBlock>())
        {
            return false;
        }

        if(GetSharedMemoryNumberOfByteOnHost() > get_lds_size())
        {
            return false;
        }
        const auto availableVgprCount = []() {
            if(ck::is_gfx125_supported())
            {
                return get_max_vgpr_count(gfx125_t{});
            }
            else if(ck::is_gfx120_supported())
            {
                return get_max_vgpr_count(gfx120_t{});
            }
            else if(ck::is_gfx11_supported())
            {
                return get_max_vgpr_count(gfx11_t{});
            }
            else
            {
                return get_max_vgpr_count(gfx9_t{});
            }
        }();
        const auto estimateVgprCount =
            ck::is_gfx11_supported() ? GetEstimateVgprCount<true>() : GetEstimateVgprCount<false>();
        if(estimateVgprCount > (availableVgprCount + availableVgprCount / 4))
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "Estimated VGPR count (" << estimateVgprCount
                          << ") exceeds available VGPR count (" << availableVgprCount << ")! "
                          << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                          << std::endl;
            }
            return false;
        }
#endif

        return true;
    }

    __host__ __device__ static constexpr bool CalculateHasMainKBlockLoop(index_t K,
                                                                         index_t k_batch = 1)
    {
        const index_t num_loop = K / (KPerBlock * k_batch);

        return GridwiseGemmPipe::CalculateHasMainLoop(num_loop);
    }

    using DsGridPointer = decltype(MakeDsGridPointer());

    template <typename ALayout, GemmSpecialization GemmSpec>
    __host__ __device__ static auto
    MakeAGridDescriptor_M_K(index_t MRaw, index_t KRaw, index_t StrideA)
    {
        constexpr auto matrix_padder =
            ck::tensor_operation::device::MatrixPadder<GemmSpec, index_t, index_t, index_t>{
                MPerBlock, NPerBlock, KPerBlock};

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

        return matrix_padder.PadADescriptor_M_K(a_grid_desc_mraw_kraw);
    }

    template <typename BLayout, GemmSpecialization GemmSpec>
    __host__ __device__ static auto
    MakeBGridDescriptor_N_K(index_t KRaw, index_t NRaw, index_t StrideB)
    {
        constexpr auto matrix_padder =
            ck::tensor_operation::device::MatrixPadder<GemmSpec, index_t, index_t, index_t>{
                MPerBlock, NPerBlock, KPerBlock};

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

        return matrix_padder.PadBDescriptor_N_K(b_grid_desc_nraw_kraw);
    }

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

#if defined(__HIPCC_RTC__) || defined(CK_CODE_GEN_RTC)
    template <typename DsLayout, GemmSpecialization GemmSpec>
    __host__ __device__ static auto
    MakeDsGridDescriptor_M_N(const ck::Array<index_t, NumDTensor>& MRaws,
                             const ck::Array<index_t, NumDTensor>& NRaws,
                             const ck::Array<index_t, NumDTensor>& DsStride)
#else
    template <typename DsLayout, GemmSpecialization GemmSpec>
    __host__ __device__ static auto
    MakeDsGridDescriptor_M_N(const std::array<index_t, NumDTensor>& MRaws,
                             const std::array<index_t, NumDTensor>& NRaws,
                             const std::array<index_t, NumDTensor>& DsStride)
#endif

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
              typename AGridDesc_AK0_M_AK1,
              typename BGridDesc_BK0_N_BK1,
              typename DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
              typename EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
              typename Block2ETileMap,
              typename EGridDesc_M_N_Direct = Tuple<>>
    __device__ static void Run(const ADataType* __restrict__ p_a_grid,
                               const BDataType* __restrict__ p_b_grid,
                               DsGridPointer p_ds_grid,
                               EDataType* __restrict__ p_e_grid,
                               void* __restrict__ p_shared,
                               const AElementwiseOperation& a_element_op,
                               const BElementwiseOperation& b_element_op,
                               const CDEElementwiseOperation& cde_element_op,
                               const AGridDesc_AK0_M_AK1& a_grid_desc_ak0_m_ak1,
                               const BGridDesc_BK0_N_BK1& b_grid_desc_bk0_n_bk1,
                               const DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock&
                                   ds_grid_desc_mblock_mperblock_nblock_nperblock,
                               const EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock&
                                   e_grid_desc_mblock_mperblock_nblock_nperblock,
                               const Block2ETileMap& block_2_etile_map,
                               const index_t k_batch                              = 1,
                               const index_t k_idx                                = 0,
                               const EGridDesc_M_N_Direct& e_grid_desc_m_n_direct = {})
    {
        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc_ak0_m_ak1.GetElementSpaceSize());

        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_grid_desc_bk0_n_bk1.GetElementSpaceSize());

        // divide block work by [M, N]
        const auto block_work_idx =
            block_2_etile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        if(!block_2_etile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(e_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I0),
                          e_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I2))))
        {
            return;
        }

        const index_t num_ak0_per_block =
            __builtin_amdgcn_readfirstlane(a_grid_desc_ak0_m_ak1.GetLength(I0) / k_batch);
        const index_t num_bk0_per_block =
            __builtin_amdgcn_readfirstlane(b_grid_desc_bk0_n_bk1.GetLength(I0) / k_batch);
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
                                                Sequence<AK0PerBlock, MPerBlock, AK1>,
                                                ABlockTransferThreadClusterLengths_AK0_M_AK1,
                                                ABlockTransferThreadClusterArrangeOrder,
                                                ADataType,
                                                AComputeDataType,
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
                make_multi_index(num_ak0_per_block * k_idx, m_block_data_idx_on_grid, 0),
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
                                                Sequence<BK0PerBlock, NPerBlock, BK1>,
                                                BBlockTransferThreadClusterLengths_BK0_N_BK1,
                                                BBlockTransferThreadClusterArrangeOrder,
                                                BDataType,
                                                BComputeDataType,
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
                make_multi_index(num_bk0_per_block * k_idx, n_block_data_idx_on_grid, 0),
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
            (((is_same<AComputeDataType, half_t>::value ||
               is_same<AComputeDataType, bhalf_t>::value) &&
              lcm_AK1_BK1 <= 4) ||
             (is_same<AComputeDataType, int8_t>::value && lcm_AK1_BK1 <= 8) ||
             ((is_same<AComputeDataType, f8_t>::value || is_same<AComputeDataType, bf8_t>::value) &&
#if defined(__gfx125__)
              lcm_AK1_BK1 < 128))
#else
              lcm_AK1_BK1 < 32))
#endif
                ? true
                : false;
        constexpr auto is_scale_mfma = false;
        constexpr index_t KPack      = math::max(lcm_AK1_BK1,
                                            MfmaSelector<AComputeDataType_,
                                                              MPerXdl,
                                                              NPerXdl,
                                                              BComputeDataType_,
                                                              is_single_rate_mfma,
                                                              is_scale_mfma>::selected_mfma.k_per_blk);
        auto blockwise_gemm          = BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_Selector<
                     BlockSize,
                     AComputeDataType,
                     BComputeDataType,
                     AccDataType,
                     decltype(a_block_desc_ak0_m_ak1),
                     decltype(b_block_desc_bk0_n_bk1),
                     MPerXdl,
                     NPerXdl,
                     MXdlPerWave,
                     NXdlPerWave,
                     KPack,
                     LoopSched,
                     AComputeDataType_,
                     BComputeDataType_>();

        auto c_thread_buf = blockwise_gemm.GetCThreadBuffer();

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);

        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<AComputeDataType*>(p_shared), a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        auto b_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<BComputeDataType*>(p_shared) + a_block_space_size_aligned,
            b_block_desc_bk0_n_bk1.GetElementSpaceSize());

        constexpr auto a_block_slice_copy_step = make_multi_index(KPerBlock / AK1, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(KPerBlock / BK1, 0, 0);

        // gridwise GEMM pipeline
        const auto gridwise_gemm_pipeline =
            GridwiseGemmPipeline_Selector<PipelineVer, NumGemmKPrefetchStage, LoopSched>();

        const index_t num_k_block_main_loop = __builtin_amdgcn_readfirstlane(
            (a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2)) /
            (KPerBlock * k_batch));

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

        // Use RunEpilogueNoShuffle from gridwise_common (the base class) for
        // direct VGPR-to-global output when conditions are met:
        // (1) no D tensors, (2) scalar-per-vector is 1, (3) PassThrough element-wise op.
        if constexpr(NumDTensor == 0 && CDEShuffleBlockTransferScalarPerVector_NPerBlock == 1 &&
                     is_same_v<CDEElementwiseOperation,
                               tensor_operation::element_wise::PassThrough>)
        {
            const auto e_grid_desc_m_n = [&]() {
                if constexpr(!is_same_v<EGridDesc_M_N_Direct, Tuple<>>)
                {
                    return e_grid_desc_m_n_direct;
                }
                else
                {
                    return transform_tensor_descriptor(
                        e_grid_desc_mblock_mperblock_nblock_nperblock,
                        make_tuple(make_merge_transform(make_tuple(
                                       e_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I0),
                                       Number<MPerBlock>{})),
                                   make_merge_transform(make_tuple(
                                       e_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I2),
                                       Number<NPerBlock>{}))),
                        make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}),
                        make_tuple(Sequence<0>{}, Sequence<1>{}));
                }
            }();

            const auto c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
                blockwise_gemm.MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(e_grid_desc_m_n);

            Base::template RunEpilogueNoShuffle<EGlobalMemoryDataOperation,
                                                false,
                                                Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                                7>(blockwise_gemm,
                                                   c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                                                   c_thread_buf,
                                                   block_work_idx[I0],
                                                   block_work_idx[I1],
                                                   p_e_grid,
                                                   cde_element_op);
        }
        else
        {
            Base::template RunMultiDEpilogue<EGlobalMemoryDataOperation, false, false, true>(
                blockwise_gemm,
                ds_grid_desc_mblock_mperblock_nblock_nperblock,
                e_grid_desc_mblock_mperblock_nblock_nperblock,
                c_thread_buf,
                block_work_idx[I0],
                block_work_idx[I1],
                p_shared,
                p_ds_grid,
                p_e_grid,
                cde_element_op);
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
                               const AElementwiseOperation& a_element_op,
                               const BElementwiseOperation& b_element_op,
                               const CDEElementwiseOperation& cde_element_op,
                               const index_t M,
                               const index_t N,
                               const index_t K,
                               const index_t StrideA,
                               const index_t StrideB,
#if defined(__HIPCC_RTC__) || defined(CK_CODE_GEN_RTC)
                               const ck::Array<index_t, NumDTensor> StrideDs,
#else
                               const std::array<index_t, NumDTensor> StrideDs,
#endif
                               const index_t StrideE,
                               const Block2ETileMap& block_2_etile_map)
    {
        const auto p_a_grid = reinterpret_cast<const ADataType*>(p_a_grid_);
        const auto p_b_grid = reinterpret_cast<const BDataType*>(p_b_grid_);
        const auto p_e_grid = reinterpret_cast<EDataType*>(p_e_grid_);

        // tensor descriptors for problem definiton
        const auto a_grid_desc_m_k = MakeAGridDescriptor_M_K<ALayout, GemmSpec>(M, K, StrideA);
        const auto b_grid_desc_n_k = MakeBGridDescriptor_N_K<BLayout, GemmSpec>(K, N, StrideB);

        using DsGridDesc_M_N =
            remove_cvref_t<decltype(MakeDsGridDescriptor_M_N<DsLayout, GemmSpec>({}, {}, {}))>;

        DsGridDesc_M_N ds_grid_desc_m_n;

        static_for<0, NumDTensor, 1>{}([&](auto j) {
            using DLayout = remove_cvref_t<tuple_element_t<j.value, DsLayout>>;

            ds_grid_desc_m_n(j) = MakeEGridDescriptor_M_N<DLayout, GemmSpec>(M, N, StrideDs[j]);
        });

        const auto e_grid_desc_m_n = MakeEGridDescriptor_M_N<ELayout, GemmSpec>(M, N, StrideE);

        // tensor descriptors for block/thread-wise copy
        const auto a_grid_desc_ak0_m_ak1 = MakeDefaultAGridDescriptor_AK0_M_AK1(a_grid_desc_m_k);

        const auto b_grid_desc_bk0_n_bk1 = MakeDefaultBGridDescriptor_BK0_N_BK1(b_grid_desc_n_k);

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

        Run<HasMainKBlockLoop, EGlobalMemoryDataOperation>(
            p_a_grid,
            p_b_grid,
            p_ds_grid,
            p_e_grid,
            p_shared,
            a_element_op,
            b_element_op,
            cde_element_op,
            a_grid_desc_ak0_m_ak1,
            b_grid_desc_bk0_n_bk1,
            ds_grid_desc_mblock_mperblock_nblock_nperblock,
            e_grid_desc_mblock_mperblock_nblock_nperblock,
            block_2_etile_map);
    }

    template <bool HasMainKBlockLoop,
              InMemoryDataOperationEnum EGlobalMemoryDataOperation,
              typename AGridDesc_MK,
              typename BGridDesc_NK,
              typename DsGridDesc_MN,
              typename EGridDesc_MN,
              typename Block2ETileMap>
    __device__ static void Run(const void* __restrict__ p_a_grid_,
                               const void* __restrict__ p_b_grid_,
                               DsGridPointer p_ds_grid,
                               void* __restrict__ p_e_grid_,
                               void* __restrict__ p_shared,
                               const AElementwiseOperation& a_element_op,
                               const BElementwiseOperation& b_element_op,
                               const CDEElementwiseOperation& cde_element_op,
                               const AGridDesc_MK& a_grid_desc_m_k,
                               const BGridDesc_NK& b_grid_desc_n_k,
                               const DsGridDesc_MN& ds_grid_desc_m_n,
                               const EGridDesc_MN& e_grid_desc_m_n,
                               const Block2ETileMap& block_2_etile_map)
    {
        const auto p_a_grid = reinterpret_cast<const ADataType*>(p_a_grid_);
        const auto p_b_grid = reinterpret_cast<const BDataType*>(p_b_grid_);
        const auto p_e_grid = reinterpret_cast<EDataType*>(p_e_grid_);

        // tensor descriptors for block/thread-wise copy
        const auto a_grid_desc_ak0_m_ak1 = MakeDefaultAGridDescriptor_AK0_M_AK1(a_grid_desc_m_k);
        const auto b_grid_desc_bk0_n_bk1 = MakeDefaultBGridDescriptor_BK0_N_BK1(b_grid_desc_n_k);

        using DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock =
            remove_cvref_t<decltype(MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                DsGridDesc_MN{}))>;

        DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock ds_grid_desc_mblock_mperblock_nblock_nperblock;

        static_for<0, NumDTensor, 1>{}([&](auto j) {
            ds_grid_desc_mblock_mperblock_nblock_nperblock(j) =
                MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(ds_grid_desc_m_n[j]);
        });

        const auto e_grid_desc_mblock_mperblock_nblock_nperblock =
            MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(e_grid_desc_m_n);

        Run<HasMainKBlockLoop, EGlobalMemoryDataOperation>(
            p_a_grid,
            p_b_grid,
            p_ds_grid,
            p_e_grid,
            p_shared,
            a_element_op,
            b_element_op,
            cde_element_op,
            a_grid_desc_ak0_m_ak1,
            b_grid_desc_bk0_n_bk1,
            ds_grid_desc_mblock_mperblock_nblock_nperblock,
            e_grid_desc_mblock_mperblock_nblock_nperblock,
            block_2_etile_map);
    }
};

} // namespace ck
