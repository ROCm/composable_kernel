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
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_selector.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_direct_load.hpp"

#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_common.hpp"

#define DEBUG_LOG 0

#if __clang_major__ >= 23
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"
#endif
namespace ck {

// Currently we do not have a elegant way to put single lds buffer & double lds buffer pipe in same
// kernel function Blockers:
// 1. Two separted declaration of __shared__ pointer is the key to make sure data access operate on
// two lds chunks.
// 2. Occupied __shared__ won't release until whole shader end, a.k.a AB and C may not use same lds
// buffer when we declare __shared__ inside blkgemmpipe
template <typename GridwiseGemm,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(GridwiseGemm::MaxBlockSize, MinimumOccupancy)
#endif
    // __attribute__((amdgpu_waves_per_eu(1, 1)))
    kernel_gemm_xdl_cshuffle_v3_multi_d(typename GridwiseGemm::Argument karg)
{
#if defined(__gfx9__) || defined(__gfx11__) || defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<CGlobalMemoryDataOperation>())
    {
        __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte(get_device_arch())];

        auto splitk_batch_offset = typename GridwiseGemm::SplitKBatchOffset(karg, blockIdx.z);

        GridwiseGemm::template Run<HasMainKBlockLoop, CGlobalMemoryDataOperation, TailNum>(
            karg.p_a_grid + splitk_batch_offset.a_k_split_offset,
            karg.p_b_grid + splitk_batch_offset.b_k_split_offset,
            karg.p_ds_grid,
            karg.p_c_grid,
            p_shared,
            karg,
            karg.a_element_op,
            karg.b_element_op,
            karg.c_element_op);
    }
#else
    ignore = karg;
#endif // end of if (defined(__gfx9__))
}

template <typename GridwiseGemm,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(GridwiseGemm::MaxBlockSize, MinimumOccupancy)
#endif
    // __attribute__((amdgpu_waves_per_eu(1, 1)))
    kernel_gemm_xdl_cshuffle_v3_multi_d_2lds(typename GridwiseGemm::Argument karg)
{
#if defined(__gfx9__) || defined(__gfx11__) || defined(__gfx12__)
    if constexpr(GridwiseGemm::template IsValidCompilationParameter<CGlobalMemoryDataOperation>())
    {
        // Pass two lds pointer is the key to tell compiler that ds_read/write
        // operate on different lds chunk at same time without order dependecy
        __shared__ char p_shared_0[GridwiseGemm::GetSharedMemoryNumberOfByte(get_device_arch())];
        __shared__ char p_shared_1[GridwiseGemm::GetSharedMemoryNumberOfByte(get_device_arch())];

        auto splitk_batch_offset = typename GridwiseGemm::SplitKBatchOffset(karg, blockIdx.z);

        GridwiseGemm::template Run_2Lds<HasMainKBlockLoop, CGlobalMemoryDataOperation, TailNum>(
            karg.p_a_grid + splitk_batch_offset.a_k_split_offset,
            karg.p_b_grid + splitk_batch_offset.b_k_split_offset,
            karg.p_ds_grid,
            karg.p_c_grid,
            p_shared_0,
            p_shared_1,
            karg,
            karg.a_element_op,
            karg.b_element_op,
            karg.c_element_op);
    }
#else
    ignore = karg;
#endif // end of if (defined(__gfx9__))
}

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          tensor_operation::device::GemmSpecialization GemmSpec,
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
          index_t ABlockLdsExtraMCustom,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          index_t BBlockLdsExtraNCustom,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          typename CDEShuffleBlockTransferScalarPerVectors,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v4,
          typename ComputeTypeA                       = CDataType,
          typename ComputeTypeB                       = ComputeTypeA,
          typename LDSTypeA                           = ADataType,
          typename LDSTypeB                           = BDataType,
          bool DoElementwiseBeforeCShuffle            = false,
          bool DirectLoad                             = false,
          bool LargeTensors                           = false>
struct GridwiseGemmMultiD_xdl_cshuffle_v3
    : public GridwiseGemm_xdl_cshuffle_base<
          ALayout,
          BLayout,
          CLayout,
          LDSTypeA,
          LDSTypeB,
          AccDataType,
          CShuffleDataType,
          DsDataType,
          CDataType,
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
          ABlockLdsExtraMCustom,
          BBlockTransferThreadClusterLengths_BK0_N_BK1,
          BBlockTransferThreadClusterArrangeOrder,
          BBlockTransferSrcAccessOrder,
          BBlockTransferSrcVectorDim,
          BBlockTransferSrcScalarPerVector,
          BBlockTransferDstScalarPerVector_BK1,
          BThreadTransferSrcResetCoordinateAfterRun,
          BBlockLdsExtraNCustom,
          CShuffleMXdlPerWavePerShuffle,
          CShuffleNXdlPerWavePerShuffle,
          CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          CDEShuffleBlockTransferScalarPerVectors,
          ComputeTypeA,
          ComputeTypeB,
          BlkGemmPipelineVer == BlockGemmPipelineVersion::v4,
          DirectLoad,
          false, // IsMxGemm (base default)
          LargeTensors>
{
    static_assert((is_same_v<AElementwiseOperation, tensor_operation::element_wise::PassThrough> &&
                   is_same_v<BElementwiseOperation, tensor_operation::element_wise::PassThrough>) ||
                  !DirectLoad);

    using IndexType = conditional_t<LargeTensors, long_index_t, index_t>;

    using Base = GridwiseGemm_xdl_cshuffle_base<
        ALayout,
        BLayout,
        CLayout,
        LDSTypeA,
        LDSTypeB,
        AccDataType,
        CShuffleDataType,
        DsDataType,
        CDataType,
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
        ABlockLdsExtraMCustom,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsExtraNCustom,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEShuffleBlockTransferScalarPerVectors,
        ComputeTypeA,
        ComputeTypeB,
        BlkGemmPipelineVer == BlockGemmPipelineVersion::v4,
        DirectLoad,
        false, // IsMxGemm (base default)
        LargeTensors>;

    using Base::AK0Number;
    using Base::AK1Number;
    using Base::BK0Number;
    using Base::BK1Number;
    using Base::I0;
    using Base::I1;
    using Base::I2;
    using ThisThreadBlock = typename Base::ThisThreadBlock;
    using Base::CShuffleBlockTransferScalarPerVector_NPerBlock;
    using Base::NumDTensor;

    // K1 should be Number<...>

    static constexpr bool DirectLoadEnabled = DirectLoad;

    static constexpr auto MakeDsGridPointer()
    {
        return generate_tuple(
            [&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                return static_cast<const DDataType*>(nullptr);
            },
            Number<NumDTensor>{});
    }

    using DsGridPointer = decltype(MakeDsGridPointer());

    static constexpr auto lcm_AK1_BK1 = math::lcm(AK1Number, BK1Number);
    static constexpr bool is_single_rate_mfma =
        (((is_same<ComputeTypeA, half_t>::value || is_same<ComputeTypeA, bhalf_t>::value) &&
          lcm_AK1_BK1 <= 4) ||
         (is_same<ComputeTypeA, int8_t>::value && lcm_AK1_BK1 <= 8) ||
         // gfx950 double rate mfma16x16 require at least 128 KPerBlock to consume
         ((is_same<ComputeTypeA, f8_t>::value || is_same<ComputeTypeA, bf8_t>::value) &&
          KPerBlock < 128 && MPerXdl == 16))
            ? true
            : false;
    static constexpr auto is_scale_mfma = false;
    static constexpr index_t KPack =
        math::max(lcm_AK1_BK1,
                  MfmaSelector<ComputeTypeA,
                               MPerXdl,
                               NPerXdl,
                               ComputeTypeB,
                               is_single_rate_mfma,
                               is_scale_mfma>::selected_mfma.k_per_blk);

    __host__ static auto CalculateGridSize(IndexType M, IndexType N, IndexType KBatch)
    {
        return std::make_tuple(Block2CTileMapDefault::CalculateGridSize(M, N), 1, KBatch);
    }

    __host__ __device__ static IndexType CalculateMPadded(IndexType M)
    {
        return math::integer_least_multiple(M, MPerBlock);
    }

    __host__ __device__ static IndexType CalculateNPadded(IndexType N)
    {
        return math::integer_least_multiple(N, NPerBlock);
    }

    __host__ __device__ static IndexType CalculateKPadded(IndexType K)
    {
        return math::integer_divide_ceil(K, KPerBlock) * KPerBlock;
    }

    __host__ __device__ static IndexType CalculateAK0Padded(IndexType K, IndexType K_Batch = 1)
    {
        auto K_t = K_Batch * KPerBlock;
        return (K + K_t - 1) / K_t * (KPerBlock / AK1Value);
    }

    __host__ __device__ static IndexType CalculateBK0Padded(IndexType K, IndexType K_Batch = 1)
    {
        auto K_t = K_Batch * KPerBlock;
        return (K + K_t - 1) / K_t * (KPerBlock / BK1Value);
    }

    __host__ __device__ static IndexType CalculateKPadded(IndexType K, IndexType K_Batch = 1)
    {
        auto K_t = K_Batch * KPerBlock;
        return (K + K_t - 1) / K_t * KPerBlock;
    }

    __host__ __device__ static IndexType CalculateKRead(IndexType K, IndexType K_Batch = 1)
    {
        constexpr auto KReadVec = math::lcm(AK1Number, BK1Number);
        auto K_t                = K_Batch * KReadVec;
        return (K + K_t - 1) / K_t * KReadVec;
    }

    __host__ __device__ static IndexType CalculateMBlock(IndexType M)
    {
        return math::integer_divide_ceil(M, MPerBlock);
    }

    __host__ __device__ static IndexType CalculateNBlock(IndexType N)
    {
        return math::integer_divide_ceil(N, NPerBlock);
    }

    template <typename GridDesc_K0_MN_K1_T, index_t K0Number, index_t K1Value>
    __host__ __device__ static auto TransformGrid(GridDesc_K0_MN_K1_T& desc)
    {

        if constexpr(!DirectLoad)
        {
            return desc;
        }
        else
        {
            const index_t K  = desc.GetLength(I0) * desc.GetLength(I2);
            const index_t MN = desc.GetLength(I1);

            const auto desc_unmerged = transform_tensor_descriptor(
                desc,
                make_tuple(make_unmerge_transform(make_tuple(K / KPerBlock, K0Number)),
                           make_pass_through_transform(MN),
                           make_pass_through_transform(K1Value)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}));

            const auto desc_permuted = transform_tensor_descriptor(
                desc_unmerged,
                make_tuple(make_pass_through_transform(K / KPerBlock),
                           make_xor_with_modulo_transform(make_tuple(MN, K0Number)),
                           make_pass_through_transform(K1Value)),
                make_tuple(Sequence<0>{}, Sequence<2, 1>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<2, 1>{}, Sequence<3>{}));

            return transform_tensor_descriptor(
                desc_permuted,
                make_tuple(
                    make_merge_transform_v3_division_mod(make_tuple(K / KPerBlock, K0Number)),
                    make_pass_through_transform(MN),
                    make_pass_through_transform(K1Value)),
                make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
        }
    }

    template <index_t MNXdlPerWave, index_t MNWaves, index_t MNPerXdl, typename TileDesc_K0_MN_K1>
    __host__ __device__ static constexpr auto MakeGemmMmaTileDescriptor(const TileDesc_K0_MN_K1&)
    {
        constexpr index_t K0 = TileDesc_K0_MN_K1{}.GetLength(Number<0>{});
        constexpr index_t K1 = TileDesc_K0_MN_K1{}.GetLength(Number<2>{});

        if constexpr(!DirectLoad)
        {
            return transform_tensor_descriptor(
                TileDesc_K0_MN_K1{},
                make_tuple(
                    make_merge_transform_v3_division_mod(make_tuple(Number<K0>{}, Number<K1>{})),
                    make_unmerge_transform(
                        make_tuple(Number<MNXdlPerWave>{}, Number<MNWaves>{}, Number<MNPerXdl>{}))),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
                make_tuple(Sequence<3>{}, Sequence<0, 1, 2>{}));
        }
        else
        {
            constexpr index_t MN = TileDesc_K0_MN_K1{}.GetLength(Number<1>{});

            constexpr auto desc = transform_tensor_descriptor(
                TileDesc_K0_MN_K1{},
                make_tuple(make_xor_with_modulo_transform(make_tuple(Number<MN>{}, Number<K0>{})),
                           make_pass_through_transform(Number<K1>{})),
                make_tuple(Sequence<1, 0>{}, Sequence<2>{}),
                make_tuple(Sequence<1, 0>{}, Sequence<2>{}));

            return transform_tensor_descriptor(
                desc,
                make_tuple(
                    make_merge_transform_v3_division_mod(make_tuple(Number<K0>{}, Number<K1>{})),
                    make_unmerge_transform(
                        make_tuple(Number<MNXdlPerWave>{}, Number<MNWaves>{}, Number<MNPerXdl>{}))),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
                make_tuple(Sequence<3>{}, Sequence<0, 1, 2>{}));
        }
    }

    __host__ __device__ static auto MakeAGridDescriptor_AK0_M_AK1(
        index_t M, index_t MPad, index_t K, index_t KPad, index_t StrideA, index_t AK0)
    {
        const auto a_grid_desc_mraw_kraw = [&]() {
            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(StrideA, I1));
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(I1, StrideA));
            }
        }();

        using GemmSpecialization = tensor_operation::device::GemmSpecialization;

        if constexpr(GemmSpec == GemmSpecialization::MKPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad both M and K
            const auto a_grid_desc_m_k =
                transform_tensor_descriptor(a_grid_desc_mraw_kraw,
                                            make_tuple(make_right_pad_transform(M, MPad - M),
                                                       make_right_pad_transform(K, KPad - K)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto a_grid_desc_ak0_m_ak1 = transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(AK0, AK1Value)),
                           make_pass_through_transform(MPad)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::MPadding ||
                          GemmSpec == GemmSpecialization::MNPadding)
        {
            // pad M, but not K
            const auto a_grid_desc_ak0_m_ak1 = transform_tensor_descriptor(
                a_grid_desc_mraw_kraw,
                make_tuple(make_unmerge_transform(make_tuple(AK0, AK1Value)),
                           make_right_pad_transform(M, MPad - M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::KPadding ||
                          GemmSpec == GemmSpecialization::NKPadding)
        {
            // pad K, but not M
            const auto a_grid_desc_m_k = transform_tensor_descriptor(
                a_grid_desc_mraw_kraw,
                make_tuple(make_pass_through_transform(M), make_right_pad_transform(K, KPad - K)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto a_grid_desc_ak0_m_ak1 = transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(AK0, AK1Value)),
                           make_pass_through_transform(M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
        else
        {
            // not pad M or K
            const auto a_grid_desc_ak0_m_ak1 = transform_tensor_descriptor(
                a_grid_desc_mraw_kraw,
                make_tuple(make_unmerge_transform(make_tuple(AK0, AK1Value)),
                           make_pass_through_transform(M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return a_grid_desc_ak0_m_ak1;
        }
    }

    __host__ __device__ static auto MakeBGridDescriptor_BK0_N_BK1(
        index_t K, index_t KPad, index_t N, index_t NPad, index_t StrideB, index_t BK0)
    {
        const auto b_grid_desc_nraw_kraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(N, K), make_tuple(I1, StrideB));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(N, K), make_tuple(StrideB, I1));
            }
        }();

        using GemmSpecialization = tensor_operation::device::GemmSpecialization;

        if constexpr(GemmSpec == GemmSpecialization::NKPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // pad both N and K
            const auto b_grid_desc_n_k =
                transform_tensor_descriptor(b_grid_desc_nraw_kraw,
                                            make_tuple(make_right_pad_transform(N, NPad - N),
                                                       make_right_pad_transform(K, KPad - K)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto b_grid_desc_bk0_n_bk1 = transform_tensor_descriptor(
                b_grid_desc_n_k,
                make_tuple(make_unmerge_transform(make_tuple(BK0, BK1Value)),
                           make_pass_through_transform(NPad)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::NPadding ||
                          GemmSpec == GemmSpecialization::MNPadding)
        {
            // pad N, but not K
            const auto b_grid_desc_bk0_n_bk1 = transform_tensor_descriptor(
                b_grid_desc_nraw_kraw,
                make_tuple(make_unmerge_transform(make_tuple(BK0, BK1Value)),
                           make_right_pad_transform(N, NPad - N)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
        else if constexpr(GemmSpec == GemmSpecialization::KPadding ||
                          GemmSpec == GemmSpecialization::MKPadding)
        {
            // pad K, but not N
            const auto b_grid_desc_n_k = transform_tensor_descriptor(
                b_grid_desc_nraw_kraw,
                make_tuple(make_pass_through_transform(N), make_right_pad_transform(K, KPad - K)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto b_grid_desc_bk0_n_bk1 = transform_tensor_descriptor(
                b_grid_desc_n_k,
                make_tuple(make_unmerge_transform(make_tuple(BK0, BK1Value)),
                           make_pass_through_transform(N)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
        else
        {
            // not pad N or K
            const auto b_grid_desc_bk0_n_bk1 = transform_tensor_descriptor(
                b_grid_desc_nraw_kraw,
                make_tuple(make_unmerge_transform(make_tuple(BK0, BK1Value)),
                           make_pass_through_transform(N)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return b_grid_desc_bk0_n_bk1;
        }
    }

    template <typename ABlockDesc_AK0_M_AK1>
    __host__ __device__ static constexpr auto
    MakeAMmaTileDescriptor_M0_M1_M2_K(const ABlockDesc_AK0_M_AK1&)
    {
        constexpr index_t MWaves = MPerBlock / (MXdlPerWave * MPerXdl);

        return MakeGemmMmaTileDescriptor<MXdlPerWave, MWaves, MPerXdl>(ABlockDesc_AK0_M_AK1{});
    }

    template <typename BBlockDesc_BK0_N_BK1>
    __host__ __device__ static constexpr auto
    MakeBMmaTileDescriptor_N0_N1_N2_K(const BBlockDesc_BK0_N_BK1&)
    {
        constexpr index_t NWaves = NPerBlock / (NXdlPerWave * NPerXdl);

        return MakeGemmMmaTileDescriptor<NXdlPerWave, NWaves, NPerXdl>(BBlockDesc_BK0_N_BK1{});
    }

    template <typename ELayout>
    __host__ __device__ static auto
    MakeCGridDescriptor_M_N(index_t M, index_t MPad, index_t N, index_t NPad, index_t StrideC)
    {
        const auto c_grid_desc_mraw_nraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ELayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(StrideC, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ELayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(I1, StrideC));
            }
            else
            {
                static_assert(false,
                              "The layout configuration is not supported! "
                              "Only support Row & Col major.");
            }
        }();

        // pad M and N
        return transform_tensor_descriptor(c_grid_desc_mraw_nraw,
                                           make_tuple(make_right_pad_transform(M, MPad - M),
                                                      make_right_pad_transform(N, NPad - N)),
                                           make_tuple(Sequence<0>{}, Sequence<1>{}),
                                           make_tuple(Sequence<0>{}, Sequence<1>{}));
    }

    __host__ __device__ static auto MakeDsGridDescriptor_M_N(
        index_t M, index_t MPad, index_t N, index_t NPad, std::array<index_t, NumDTensor> StrideDs)
    {
        return generate_tuple(
            [&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;
                return MakeCGridDescriptor_M_N<DLayout>(M, MPad, N, NPad, StrideDs[i]);
            },
            Number<NumDTensor>{});
    }

    template <typename DsGridDesc>
    __device__ static constexpr auto MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
        const DsGridDesc& ds_grid_desc_m_n, index_t MBlock, index_t NBlock)
    {
        return generate_tuple(
            [&](auto i) {
                return MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                    ds_grid_desc_m_n[i], MBlock, NBlock);
            },
            Number<NumDTensor>{});
    }

    struct Problem
    {
        __host__ __device__ Problem() = default;
        __host__ __device__ Problem(IndexType M_,
                                    IndexType N_,
                                    IndexType K_,
                                    IndexType StrideA_,
                                    IndexType StrideB_,
                                    std::array<IndexType, NumDTensor> StrideDs_,
                                    IndexType StrideC_,
                                    IndexType KBatch_)
            : M{M_},
              N{N_},
              K{K_},
              StrideA{StrideA_},
              StrideB{StrideB_},
              StrideDs{StrideDs_},
              StrideC{StrideC_},
              KBatch{KBatch_},
              MPadded{CalculateMPadded(M_)},
              NPadded{CalculateNPadded(N_)},
              KRead{CalculateKRead(K_, KBatch_)},
              KPadded{CalculateKPadded(K_, KBatch_)},
              AK0{CalculateAK0Padded(K_, KBatch_)},
              BK0{CalculateBK0Padded(K_, KBatch_)},
              MBlock{CalculateMBlock(M_)},
              NBlock{CalculateNBlock(N_)}
        {
        }

        __host__ void Print() const
        {
            std::cout << "problem {" << "M:" << M << ", " << "N:" << N << ", " << "K:" << K << ", "
                      << "SA:" << StrideA << ", " << "SB:" << StrideB << ", " << "SC:" << StrideC
                      << ", " << "MP:" << MPadded << ", " << "NP:" << NPadded << ", "
                      << "KRead:" << KRead << ", " << "KP:" << KPadded << ", " << "AK0:" << AK0
                      << ", " << "BK0:" << BK0 << ", " << "MBlock: " << MBlock << ", "
                      << "NBlock: " << NBlock << "}" << std::endl;
        }

        IndexType M;
        IndexType N;
        IndexType K;
        IndexType StrideA;
        IndexType StrideB;
        std::array<IndexType, NumDTensor> StrideDs;
        IndexType StrideC;
        IndexType KBatch;
        IndexType MPadded;
        IndexType NPadded;
        IndexType KRead;
        IndexType KPadded;
        IndexType AK0;
        IndexType BK0;
        IndexType MBlock;
        IndexType NBlock;
    };

    // Argument
    struct Argument : public tensor_operation::device::BaseArgument, public Problem
    {
        __host__ Argument() = default;
        __host__ Argument(const ADataType* p_a_grid_,
                          const BDataType* p_b_grid_,
                          std::array<const void*, NumDTensor> p_ds_grid_,
                          CDataType* p_c_grid_,
                          IndexType M_,
                          IndexType N_,
                          IndexType K_,
                          IndexType StrideA_,
                          IndexType StrideB_,
                          std::array<IndexType, NumDTensor> StrideDs_,
                          IndexType StrideC_,
                          IndexType k_batch_,
                          AElementwiseOperation a_element_op_,
                          BElementwiseOperation b_element_op_,
                          CElementwiseOperation c_element_op_)
            : Problem{M_, N_, K_, StrideA_, StrideB_, StrideDs_, StrideC_, k_batch_},
              p_a_grid{p_a_grid_},
              p_b_grid{p_b_grid_},
              p_ds_grid{},
              p_c_grid{p_c_grid_},
              a_element_op{a_element_op_},
              b_element_op{b_element_op_},
              c_element_op{c_element_op_}
        {

            // populate pointer, desc for Ds
            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DDataType_ = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                // D pointer
                p_ds_grid(i) = static_cast<const DDataType_*>(p_ds_grid_[i]);
            });
        }

        const ADataType* p_a_grid;
        const BDataType* p_b_grid;
        DsGridPointer p_ds_grid;
        CDataType* p_c_grid;

        AElementwiseOperation a_element_op;
        BElementwiseOperation b_element_op;
        CElementwiseOperation c_element_op;
    };

    struct SplitKBatchOffset
    {
        __device__ SplitKBatchOffset(Argument& karg, index_t k_id)
        {
            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                a_k_split_offset = k_id * karg.KRead;
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                a_k_split_offset = k_id * karg.KRead * karg.StrideA;
            }

            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, BLayout>)
            {
                b_k_split_offset = k_id * karg.KRead * karg.StrideB;
            }
            else if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, BLayout>)
            {
                b_k_split_offset = k_id * karg.KRead;
            }

            if(k_id < karg.KBatch - 1)
            {
                karg.K = karg.KRead;
            }
            else
            {
                karg.K = karg.K - karg.KRead * (karg.KBatch - 1);
            }
        }

        index_t a_k_split_offset;
        index_t b_k_split_offset;
    };

    template <typename DeviceArch>
    __device__ __host__ static constexpr auto
    GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1(DeviceArch)
    {
        if constexpr(is_same_v<DeviceArch, gfx950_t>)
        {
            if constexpr(DirectLoad == false)
            {
                // Force use padded layout on gfx950 to reduce bank conflicts
                constexpr index_t ABlockLdsExtraM = 1;
                return make_naive_tensor_descriptor(
                    make_tuple(AK0Number, Number<MPerBlock>{}, AK1Number),
                    make_tuple(Number<MPerBlock + ABlockLdsExtraM>{} * AK1Number, AK1Number, I1));
            }
            else
            {
                return Base::GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1(DeviceArch{});
            }
        }
        else
        {
            return Base::GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1(DeviceArch{});
        }
    }

    template <typename DeviceArch>
    __device__ __host__ static constexpr auto
    GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1(DeviceArch)
    {
        if constexpr(is_same_v<DeviceArch, gfx950_t>)
        {
            if constexpr(DirectLoad == false)
            {
                constexpr index_t BBlockLdsExtraN = 1;
                return make_naive_tensor_descriptor(
                    make_tuple(BK0Number, Number<NPerBlock>{}, BK1Number),
                    make_tuple(Number<NPerBlock + BBlockLdsExtraN>{} * BK1Number, BK1Number, I1));
            }
            else
            {
                return Base::GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1(DeviceArch{});
            }
        }
        else
        {
            return Base::GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1(DeviceArch{});
        }
    }

    using BlockwiseGemmPipe = remove_cvref_t<
        decltype(BlockGemmPipeline_Selector<
                 BlkGemmPipelineVer,
                 BlkGemmPipeSched,
                 BlockSize,
                 LDSTypeA,
                 LDSTypeB,
                 ComputeTypeA,
                 AccDataType,
                 decltype(GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1(get_device_arch())),
                 decltype(GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1(get_device_arch())),
                 decltype(MakeAMmaTileDescriptor_M0_M1_M2_K(
                     GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1(get_device_arch()))),
                 decltype(MakeBMmaTileDescriptor_N0_N1_N2_K(
                     GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1(get_device_arch()))),
                 ABlockTransferSrcScalarPerVector,
                 BBlockTransferSrcScalarPerVector,
                 MPerBlock,
                 NPerBlock,
                 KPerBlock,
                 MPerXdl,
                 NPerXdl,
                 MXdlPerWave,
                 NXdlPerWave,
                 KPack,
                 DirectLoad>())>;

    template <typename DeviceArch>
    __device__ __host__ static constexpr index_t GetSharedMemoryNumberOfByte(DeviceArch)
    {
        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_desc_ak0_m_ak1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1(DeviceArch{});
        constexpr auto b_block_desc_bk0_n_bk1 =
            GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1(DeviceArch{});

        // lds max alignment
        constexpr auto max_lds_align = math::lcm(AK1Number, BK1Number);

        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_space_size_aligned = math::integer_least_multiple(
            b_block_desc_bk0_n_bk1.GetElementSpaceSize(), max_lds_align);

        // LDS allocation for C shuffle in LDS
        constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            Base::GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(DeviceArch{});

        constexpr auto c_block_size =
            c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize();

        return math::max((a_block_space_size_aligned * sizeof(LDSTypeA) +
                          b_block_space_size_aligned * sizeof(LDSTypeB)),
                         c_block_size * sizeof(CShuffleDataType));
    }

    __host__ static index_t GetSharedMemoryNumberOfByteOnHost()
    {
#if !defined(__HIPCC_RTC__) || !defined(CK_CODE_GEN_RTC)
        if(is_gfx125_supported())
        {
            return GetSharedMemoryNumberOfByte(gfx125_t{});
        }
        else if(ck::get_device_name() == "gfx950")
        {
            return GetSharedMemoryNumberOfByte(gfx950_t{});
        }
        else
#endif
        {
            return GetSharedMemoryNumberOfByte(gfx_invalid_t{});
        }
    }

    template <bool IsGfx11>
    static constexpr index_t GetEstimateVgprCount()
    {
        constexpr index_t MWave    = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t NWave    = NPerBlock / (NXdlPerWave * NPerXdl);
        constexpr index_t WaveSize = BlockSize / (MWave * NWave);

        // VGPR used in LDS loading and WMMA
        constexpr index_t BaseInputVgprCount =
            MPerBlock * KPerBlock / MWave / WaveSize * sizeof(ComputeTypeA) / sizeof(uint32_t) +
            NPerBlock * KPerBlock / NWave / WaveSize * sizeof(ComputeTypeB) / sizeof(uint32_t);
        // WMMA input is duplicated in GFX11
        constexpr index_t InputVgprCount = IsGfx11 ? BaseInputVgprCount * 2 : BaseInputVgprCount;
        // VGPR used in Accumulator
        constexpr index_t AccVgprCount =
            MPerBlock * NPerBlock / BlockSize * sizeof(AccDataType) / sizeof(uint32_t);

        if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
        {
            return InputVgprCount + AccVgprCount;
        }
        else if constexpr((BlkGemmPipelineVer == BlockGemmPipelineVersion::v2) ||
                          (BlkGemmPipelineVer == BlockGemmPipelineVersion::v3) ||
                          (BlkGemmPipelineVer == BlockGemmPipelineVersion::v5))
        {
            return 2 * InputVgprCount + AccVgprCount;
        }
        else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v4)
        {
            return 3 * InputVgprCount + AccVgprCount;
        }
        else
        {
            // invalid pipeline version
            static_assert(0);
        }
    }
    template <
        InMemoryDataOperationEnum CGlobalMemoryDataOperation_ = InMemoryDataOperationEnum::Set>
    __device__ static bool constexpr IsValidCompilationParameter()
    {
        enum struct Arch : bool
        {
#if defined(__gfx950__)
            is_gfx950_build = true,
#else
            is_gfx950_build = false,
#endif
        };

        // skip building the instances with K1>=32 on pre-gfx950
        if constexpr((static_cast<bool>(Arch::is_gfx950_build) == false) &&
                     (AK1Number >= 32 || BK1Number >= 32))
        {
            return false;
        }

        constexpr bool valid = ck::tensor_operation::device::IsValidGemmCompilationParameter<
            BlockSize,
            MPerBlock,
            NPerBlock,
            MPerXdl,
            NPerXdl,
            MXdlPerWave,
            NXdlPerWave,
            CDataType,
            CGlobalMemoryDataOperation_>();
        if constexpr(!valid)
        {
            return false;
        }

        using MfmaInst = MfmaSelector<ComputeTypeA,
                                      MPerXdl,
                                      NPerXdl,
                                      ComputeTypeB,
                                      is_single_rate_mfma,
                                      is_scale_mfma>;

        constexpr index_t KPerThread =
            KPerBlock / (MfmaInst::GetKPerXdlops() / MfmaInst::GetK1PerXdlops());
        if constexpr(KPerThread % KPack != 0)
        {
            return false;
        }

        if constexpr(NXdlPerWave % CShuffleNXdlPerWavePerShuffle != 0)
        {
            return false;
        }

        constexpr index_t LdsBufferCount =
            BlkGemmPipelineVer == BlockGemmPipelineVersion::v4 ? 2 : 1;
        if constexpr(GetSharedMemoryNumberOfByte(get_device_arch()) * LdsBufferCount >
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
        return true;
    }

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    __host__ static constexpr bool CheckValidity(const Argument& karg)
    {
        static_assert((MPerBlock % (MPerXdl * MXdlPerWave) == 0) &&
                          (NPerBlock % (NXdlPerWave * NPerXdl)) == 0,
                      "Invalid tuning param!");

        if constexpr(NXdlPerWave % CShuffleNXdlPerWavePerShuffle != 0)
        {
            return false;
        }
        constexpr index_t ldsBufferCount =
            BlkGemmPipelineVer == BlockGemmPipelineVersion::v4 ? 2 : 1;
        if(GetSharedMemoryNumberOfByteOnHost() * ldsBufferCount > get_lds_size())
        {
            return false;
        }
        if(!is_xdl_wmma_k_supported<ComputeTypeA, KPerBlock>())
        {
            return false;
        }
        if constexpr(!(GemmSpec == tensor_operation::device::GemmSpecialization::MPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding) &&
                     !(is_same<tensor_layout::gemm::RowMajor, ALayout>::value))
        {
            if(!(karg.M % MPerBlock == 0))
            {
#if DEBUG_LOG
                std::cout << "Arg M value is not a multiple of MPerBlock! M: " << karg.M << " "
                          << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                          << std::endl;

#endif // DEBUG_LOG
                return false;
            }
        }

        if constexpr(!(GemmSpec == tensor_operation::device::GemmSpecialization::NPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::NKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding) &&
                     (is_same<tensor_layout::gemm::RowMajor, BLayout>::value))
        {
            if(!(karg.N % NPerBlock == 0))
            {
#if DEBUG_LOG
                std::cout << "Arg N value is not a multiple of NPerBlock! N: " << karg.N << " "
                          << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                          << std::endl;

#endif // DEBUG_LOG
                return false;
            }
        }

        if constexpr(!(GemmSpec == tensor_operation::device::GemmSpecialization::KPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::NKPadding ||
                       GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding))
        {

            auto K_t = karg.KBatch * KPerBlock;
            if(!(karg.K % K_t == 0))
            {
#if DEBUG_LOG
                std::cout << "Arg K value is not a multiple of K_Batch * K0PerBlock * K1! K: "
                          << karg.K << " " << __FILE__ << ":" << __LINE__
                          << ", in function: " << __func__ << std::endl;

#endif // DEBUG_LOG
                return false;
            }
        }
        else
        {
            constexpr auto KReadVec = math::lcm(AK1Number, BK1Number);
            auto K_t                = karg.KBatch * KReadVec;
            auto KReadPadSplited    = math::integer_divide_ceil(karg.K, K_t) * KReadVec;
            if((KReadPadSplited * (karg.KBatch - 1)) >= karg.K)
            {
                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
        {
            if(karg.K % ABlockTransferSrcScalarPerVector != 0)
            {
#if DEBUG_LOG
                std::cout << "Arg K (" << karg.K
                          << ") value is not a multiple of ABlockTransferSrcScalarPerVector ("
                          << ABlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                          << __LINE__ << ", in function: " << __func__ << std::endl;

#endif // DEBUG_LOG
                return false;
            }
        }
        else
        {
            if(karg.M % ABlockTransferSrcScalarPerVector != 0)
            {
#if DEBUG_LOG
                std::cout << "Arg M (" << karg.M
                          << ") value is not a multiple of ABlockTransferSrcScalarPerVector ("
                          << ABlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                          << __LINE__ << ", in function: " << __func__ << std::endl;

#endif // DEBUG_LOG
                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
        {
            if(karg.N % BBlockTransferSrcScalarPerVector != 0)
            {
#if DEBUG_LOG
                std::cout << "Arg N (" << karg.N
                          << ") value is not a multiple of BBlockTransferSrcScalarPerVector ("
                          << BBlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                          << __LINE__ << ", in function: " << __func__ << std::endl;

#endif // DEBUG_LOG
                return false;
            }
        }
        else
        {
            if(karg.K % BBlockTransferSrcScalarPerVector != 0)
            {
#if DEBUG_LOG
                std::cout << "Arg K (" << karg.K
                          << ") value is not a multiple of BBlockTransferSrcScalarPerVector ("
                          << BBlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                          << __LINE__ << ", in function: " << __func__ << std::endl;

#endif // DEBUG_LOG
                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
        {
            if(karg.N % CShuffleBlockTransferScalarPerVector_NPerBlock != 0)
            {
#if DEBUG_LOG
                std::cout << "Arg N (" << karg.N
                          << ") value is not a multiple of "
                             "CShuffleBlockTransferScalarPerVector_NPerBlock ("
                          << CShuffleBlockTransferScalarPerVector_NPerBlock << " )! " << __FILE__
                          << ":" << __LINE__ << ", in function: " << __func__ << std::endl;

#endif // DEBUG_LOG
                return false;
            }
        }
        else
        {
            if(karg.M % CShuffleBlockTransferScalarPerVector_NPerBlock != 0)
            {
#if DEBUG_LOG
                std::cout << "Arg M (" << karg.M
                          << ") value is not a multiple of "
                             "CShuffleBlockTransferScalarPerVector_NPerBlock ("
                          << CShuffleBlockTransferScalarPerVector_NPerBlock << " )! " << __FILE__
                          << ":" << __LINE__ << ", in function: " << __func__ << std::endl;

#endif // DEBUG_LOG
                return false;
            }
        }

        // check gridwise gemm pipeline
        const auto num_k_loop = karg.AK0 / (KPerBlock / AK1Value);

        if constexpr(BlkGemmPipelineVer != BlockGemmPipelineVersion::v1)
        {
            if(num_k_loop <= BlockwiseGemmPipe::PrefetchStages)
            {
                return false;
            }
        }

        if constexpr(!LargeTensors)
        {
            constexpr long_index_t TwoGB = (long_index_t{1} << 31);
            if(!(karg.M * karg.K * sizeof(ADataType) <= TwoGB &&
                 karg.N * karg.K * sizeof(BDataType) <= TwoGB &&
                 karg.M * karg.N * sizeof(CDataType) <= TwoGB))
            {
                return false;
            }
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

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        return true;
    }

    __host__ __device__ static constexpr bool CalculateHasMainKBlockLoop(index_t K)
    {
        const index_t num_loop = K / KPerBlock;

        return BlockwiseGemmPipe::BlockHasHotloop(num_loop);
    }

    __host__ __device__ static constexpr TailNumber CalculateKBlockLoopTailNum(index_t K)
    {
        const index_t num_loop = K / KPerBlock;

        return BlockwiseGemmPipe::BlockLoopTailNum(num_loop);
    }

    template <typename CGridDesc>
    __device__ static constexpr auto MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
        const CGridDesc& c_grid_desc_m_n, index_t MBlock, index_t NBlock)
    {
        const auto c_grid_desc_mblock_mperblock_nblock_nperblock = transform_tensor_descriptor(
            c_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{})),
                       make_unmerge_transform(make_tuple(NBlock, Number<NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

        return c_grid_desc_mblock_mperblock_nblock_nperblock;
    }

    // return block_id to C matrix tile idx (m0, n0) mapping
    // if arch = gfx942
    using Block2CTileMapDefault =
        BlockToCTileMap_Grouped_M00_N0_M01Adapt<8, MPerBlock, NPerBlock, IndexType>;

    template <bool HasMainKBlockLoop,
              InMemoryDataOperationEnum CGlobalMemoryDataOperation,
              TailNumber TailNum = TailNumber::Odd>
    __device__ static void Run(const ADataType* __restrict__ p_a_grid,
                               const BDataType* __restrict__ p_b_grid,
                               DsGridPointer& p_ds_grid,
                               CDataType* __restrict__ p_c_grid,
                               void* __restrict__ p_shared,
                               const Problem& problem,
                               AElementwiseOperation a_element_op,
                               BElementwiseOperation b_element_op,
                               CElementwiseOperation c_element_op)
    {
        const auto block_2_ctile_map = Block2CTileMapDefault{problem.M, problem.N, 4};
        Run<Block2CTileMapDefault, HasMainKBlockLoop, CGlobalMemoryDataOperation, TailNum>(
            p_a_grid,
            p_b_grid,
            p_ds_grid,
            p_c_grid,
            p_shared,
            problem,
            a_element_op,
            b_element_op,
            c_element_op,
            block_2_ctile_map);
    }

    template <typename Block2CTileMap,
              bool HasMainKBlockLoop,
              InMemoryDataOperationEnum CGlobalMemoryDataOperation,
              TailNumber TailNum = TailNumber::Odd>
    __device__ static void Run(const ADataType* __restrict__ p_a_grid,
                               const BDataType* __restrict__ p_b_grid,
                               DsGridPointer& p_ds_grid,
                               CDataType* __restrict__ p_c_grid,
                               void* __restrict__ p_shared,
                               const Problem& problem,
                               AElementwiseOperation a_element_op,
                               BElementwiseOperation b_element_op,
                               CElementwiseOperation c_element_op,
                               const Block2CTileMap& block_2_ctile_map)
    {
        const auto a_grid_desc_ak0_m_ak1 = MakeAGridDescriptor_AK0_M_AK1(
            problem.M, problem.MPadded, problem.K, problem.KPadded, problem.StrideA, problem.AK0);
        const auto b_grid_desc_bk0_n_bk1 = MakeBGridDescriptor_BK0_N_BK1(
            problem.K, problem.KPadded, problem.N, problem.NPadded, problem.StrideB, problem.BK0);

        const auto c_grid_desc_m_n = MakeCGridDescriptor_M_N<CLayout>(
            problem.M, problem.MPadded, problem.N, problem.NPadded, problem.StrideC);
        const auto ds_grid_desc_m_n = MakeDsGridDescriptor_M_N(
            problem.M, problem.MPadded, problem.N, problem.NPadded, problem.StrideDs);

        Run<HasMainKBlockLoop, CGlobalMemoryDataOperation, TailNum, Block2CTileMap>(
            p_a_grid,
            p_b_grid,
            p_ds_grid,
            p_c_grid,
            p_shared,
            problem,
            a_element_op,
            b_element_op,
            c_element_op,
            block_2_ctile_map,
            a_grid_desc_ak0_m_ak1,
            b_grid_desc_bk0_n_bk1,
            ds_grid_desc_m_n,
            c_grid_desc_m_n);
    }

    template <bool HasMainKBlockLoop,
              InMemoryDataOperationEnum CGlobalMemoryDataOperation,
              TailNumber TailNum,
              typename Block2CTileMap,
              typename AGridDesc_AK0_M_K1,
              typename BGridDesc_BK0_N_K1,
              typename DsGridDesc_M_N,
              typename CGridDesc_M_N>
    __device__ static void Run(const ADataType* __restrict__ p_a_grid,
                               const BDataType* __restrict__ p_b_grid,
                               DsGridPointer& p_ds_grid,
                               CDataType* __restrict__ p_c_grid,
                               void* __restrict__ p_shared,
                               const Problem& problem,
                               [[maybe_unused]] AElementwiseOperation a_element_op,
                               [[maybe_unused]] BElementwiseOperation b_element_op,
                               CElementwiseOperation c_element_op,
                               const Block2CTileMap& block_2_ctile_map,
                               const AGridDesc_AK0_M_K1& a_grid_desc_ak0_m_ak1,
                               const BGridDesc_BK0_N_K1& b_grid_desc_bk0_n_bk1,
                               const DsGridDesc_M_N& ds_grid_desc_m_n,
                               const CGridDesc_M_N& c_grid_desc_m_n)
    {

        const auto a_grid_buf =
            make_dynamic_buffer<AddressSpaceEnum::Global,
                                AmdBufferCoherenceEnum::DefaultCoherence,
                                IndexType>(p_a_grid, a_grid_desc_ak0_m_ak1.GetElementSpaceSize());
        const auto b_grid_buf =
            make_dynamic_buffer<AddressSpaceEnum::Global,
                                AmdBufferCoherenceEnum::DefaultCoherence,
                                IndexType>(p_b_grid, b_grid_desc_bk0_n_bk1.GetElementSpaceSize());

        const auto c_grid_desc_mblock_mperblock_nblock_nperblock =
            MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                c_grid_desc_m_n, problem.MBlock, problem.NBlock);

        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        if(!block_2_ctile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I0),
                          c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I2))))
        {
            return;
        }

        const index_t block_m_id = __builtin_amdgcn_readfirstlane(block_work_idx[I0]);
        const index_t block_n_id = __builtin_amdgcn_readfirstlane(block_work_idx[I1]);

        // HACK: this force m/n_block_data_idx_on_grid into SGPR
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_m_id * MPerBlock);

        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_n_id * NPerBlock);

        // lds max alignment
        constexpr auto max_lds_align = math::lcm(AK1Number, BK1Number);

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_ak0_m_ak1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1(get_device_arch());

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_bk0_n_bk1 =
            GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1(get_device_arch());

        auto get_a_blockwise_copy = [&]() {
            if constexpr(DirectLoad)
            {
                return ThreadGroupTensorSliceTransfer_DirectLoad<
                    ThisThreadBlock,
                    Sequence<AK0Number, MPerBlock, AK1Number>,
                    ABlockTransferThreadClusterLengths_AK0_M_AK1,
                    ABlockTransferThreadClusterArrangeOrder,
                    ADataType,
                    ADataType,
                    decltype(a_grid_desc_ak0_m_ak1),
                    decltype(a_block_desc_ak0_m_ak1),
                    ABlockTransferSrcAccessOrder,
                    ABlockTransferSrcVectorDim,
                    2,
                    ABlockTransferSrcScalarPerVector>(
                    a_grid_desc_ak0_m_ak1,
                    make_multi_index(0, m_block_data_idx_on_grid, 0),
                    a_block_desc_ak0_m_ak1,
                    make_multi_index(0, 0, 0));
            }
            else
            {
                return ThreadGroupTensorSliceTransfer_v4r1<
                    ThisThreadBlock,
                    AElementwiseOperation,
                    ck::tensor_operation::element_wise::PassThrough,
                    InMemoryDataOperationEnum::Set,
                    Sequence<AK0Number, MPerBlock, AK1Number>,
                    ABlockTransferThreadClusterLengths_AK0_M_AK1,
                    ABlockTransferThreadClusterArrangeOrder,
                    ADataType,
                    LDSTypeA,
                    decltype(a_grid_desc_ak0_m_ak1),
                    decltype(a_block_desc_ak0_m_ak1),
                    ABlockTransferSrcAccessOrder,
                    Sequence<0, 1, 2>,
                    ABlockTransferSrcVectorDim,
                    2,
                    ABlockTransferSrcScalarPerVector,
                    ABlockTransferDstScalarPerVector_AK1,
                    1,
                    1,
                    AThreadTransferSrcResetCoordinateAfterRun,
                    true,
                    BlockwiseGemmPipe::GlobalBufferNum,
                    IndexType>(a_grid_desc_ak0_m_ak1,
                               make_multi_index(0, m_block_data_idx_on_grid, 0),
                               a_element_op,
                               a_block_desc_ak0_m_ak1,
                               make_multi_index(0, 0, 0),
                               ck::tensor_operation::element_wise::PassThrough{});
            }
        };

        // B matrix blockwise copy
        auto get_b_blockwise_copy = [&]() {
            if constexpr(DirectLoad)
            {
                return ThreadGroupTensorSliceTransfer_DirectLoad<
                    ThisThreadBlock,
                    Sequence<BK0Number, NPerBlock, BK1Number>,
                    BBlockTransferThreadClusterLengths_BK0_N_BK1,
                    BBlockTransferThreadClusterArrangeOrder,
                    BDataType,
                    BDataType,
                    decltype(b_grid_desc_bk0_n_bk1),
                    decltype(b_block_desc_bk0_n_bk1),
                    BBlockTransferSrcAccessOrder,
                    BBlockTransferSrcVectorDim,
                    2,
                    BBlockTransferSrcScalarPerVector>(
                    b_grid_desc_bk0_n_bk1,
                    make_multi_index(0, n_block_data_idx_on_grid, 0),
                    b_block_desc_bk0_n_bk1,
                    make_multi_index(0, 0, 0));
            }
            else
            {
                return ThreadGroupTensorSliceTransfer_v4r1<
                    ThisThreadBlock,
                    BElementwiseOperation,
                    ck::tensor_operation::element_wise::PassThrough,
                    InMemoryDataOperationEnum::Set,
                    Sequence<BK0Number, NPerBlock, BK1Number>,
                    BBlockTransferThreadClusterLengths_BK0_N_BK1,
                    BBlockTransferThreadClusterArrangeOrder,
                    BDataType,
                    LDSTypeB,
                    decltype(b_grid_desc_bk0_n_bk1),
                    decltype(b_block_desc_bk0_n_bk1),
                    BBlockTransferSrcAccessOrder,
                    Sequence<0, 1, 2>,
                    BBlockTransferSrcVectorDim,
                    2,
                    BBlockTransferSrcScalarPerVector,
                    BBlockTransferDstScalarPerVector_BK1,
                    1,
                    1,
                    BThreadTransferSrcResetCoordinateAfterRun,
                    true,
                    BlockwiseGemmPipe::GlobalBufferNum,
                    IndexType>(b_grid_desc_bk0_n_bk1,
                               make_multi_index(0, n_block_data_idx_on_grid, 0),
                               b_element_op,
                               b_block_desc_bk0_n_bk1,
                               make_multi_index(0, 0, 0),
                               ck::tensor_operation::element_wise::PassThrough{});
            }
        };

        auto a_blockwise_copy = get_a_blockwise_copy();
        auto b_blockwise_copy = get_b_blockwise_copy();

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);

        // Cast after lds
        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<LDSTypeA*>(p_shared), a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        auto b_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<LDSTypeB*>(p_shared) +
                a_block_space_size_aligned * sizeof(LDSTypeA) / sizeof(LDSTypeB),
            b_block_desc_bk0_n_bk1.GetElementSpaceSize());

        constexpr auto a_block_slice_copy_step = make_multi_index(KPerBlock / AK1Number, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(KPerBlock / BK1Number, 0, 0);

        // Blockwise GEMM pipeline
        static_assert(std::is_default_constructible_v<BlockwiseGemmPipe>);
        auto blockwise_gemm_pipeline = BlockwiseGemmPipe{};
        auto c_thread_buf            = blockwise_gemm_pipeline.GetCThreadBuffer();

        const index_t num_k_block_main_loop = __builtin_amdgcn_readfirstlane(
            (a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2)) /
            KPerBlock);

        blockwise_gemm_pipeline.template Run<HasMainKBlockLoop, TailNum>(a_grid_desc_ak0_m_ak1,
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
                                                                         c_thread_buf,
                                                                         num_k_block_main_loop);

        if constexpr(LargeTensors)
        {
            static_assert(NumDTensor == 0, "Not implemented");
            Base::template RunEpilogue<CGlobalMemoryDataOperation, false, false>(
                blockwise_gemm_pipeline,
                c_grid_desc_mblock_mperblock_nblock_nperblock,
                c_thread_buf,
                block_m_id,
                block_n_id,
                p_shared,
                p_c_grid,
                c_element_op);
        }
        else
        {
            // shuffle C and write out
            const auto ds_grid_desc_mblock_mperblock_nblock_nperblock =
                MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                    ds_grid_desc_m_n, problem.MBlock, problem.NBlock);
            Base::template RunMultiDEpilogue<CGlobalMemoryDataOperation,
                                             DoElementwiseBeforeCShuffle,
                                             false,
                                             false>(blockwise_gemm_pipeline,
                                                    ds_grid_desc_mblock_mperblock_nblock_nperblock,
                                                    c_grid_desc_mblock_mperblock_nblock_nperblock,
                                                    c_thread_buf,
                                                    block_m_id,
                                                    block_n_id,
                                                    p_shared,
                                                    p_ds_grid,
                                                    p_c_grid,
                                                    c_element_op);
        }
    }

    template <bool HasMainKBlockLoop,
              InMemoryDataOperationEnum CGlobalMemoryDataOperation,
              TailNumber TailNum = TailNumber::Odd>
    __device__ static void Run_2Lds(const ADataType* __restrict__ p_a_grid,
                                    const BDataType* __restrict__ p_b_grid,
                                    DsGridPointer& p_ds_grid,
                                    CDataType* __restrict__ p_c_grid,
                                    void* __restrict__ p_shared_0,
                                    void* __restrict__ p_shared_1,
                                    const Problem& problem,
                                    AElementwiseOperation a_element_op,
                                    BElementwiseOperation b_element_op,
                                    CElementwiseOperation c_element_op)
    {
        // divide block work by [M, N]
        const auto block_2_ctile_map = Block2CTileMapDefault{problem.M, problem.N, 4};
        Run_2Lds<Block2CTileMapDefault, HasMainKBlockLoop, CGlobalMemoryDataOperation, TailNum>(
            p_a_grid,
            p_b_grid,
            p_ds_grid,
            p_c_grid,
            p_shared_0,
            p_shared_1,
            problem,
            a_element_op,
            b_element_op,
            c_element_op,
            block_2_ctile_map);
    }

    template <typename Block2CTileMap,
              bool HasMainKBlockLoop,
              InMemoryDataOperationEnum CGlobalMemoryDataOperation,
              TailNumber TailNum = TailNumber::Odd>
    __device__ static void Run_2Lds(const ADataType* __restrict__ p_a_grid,
                                    const BDataType* __restrict__ p_b_grid,
                                    DsGridPointer& p_ds_grid,
                                    CDataType* __restrict__ p_c_grid,
                                    void* __restrict__ p_shared_0,
                                    void* __restrict__ p_shared_1,
                                    const Problem& problem,
                                    AElementwiseOperation a_element_op,
                                    BElementwiseOperation b_element_op,
                                    CElementwiseOperation c_element_op,
                                    const Block2CTileMap& block_2_ctile_map)
    {
        const auto a_grid_desc_ak0_m_ak1 = MakeAGridDescriptor_AK0_M_AK1(
            problem.M, problem.MPadded, problem.K, problem.KPadded, problem.StrideA, problem.AK0);
        const auto b_grid_desc_bk0_n_bk1 = MakeBGridDescriptor_BK0_N_BK1(
            problem.K, problem.KPadded, problem.N, problem.NPadded, problem.StrideB, problem.BK0);

        const auto c_grid_desc_m_n = MakeCGridDescriptor_M_N<CLayout>(
            problem.M, problem.MPadded, problem.N, problem.NPadded, problem.StrideC);
        const auto ds_grid_desc_m_n = MakeDsGridDescriptor_M_N(
            problem.M, problem.MPadded, problem.N, problem.NPadded, problem.StrideDs);

        Run_2Lds<HasMainKBlockLoop, CGlobalMemoryDataOperation, TailNum>(p_a_grid,
                                                                         p_b_grid,
                                                                         p_ds_grid,
                                                                         p_c_grid,
                                                                         p_shared_0,
                                                                         p_shared_1,
                                                                         problem,
                                                                         a_element_op,
                                                                         b_element_op,
                                                                         c_element_op,
                                                                         block_2_ctile_map,
                                                                         a_grid_desc_ak0_m_ak1,
                                                                         b_grid_desc_bk0_n_bk1,
                                                                         ds_grid_desc_m_n,
                                                                         c_grid_desc_m_n);
    }

    template <bool HasMainKBlockLoop,
              InMemoryDataOperationEnum CGlobalMemoryDataOperation,
              TailNumber TailNum,
              typename Block2CTileMap,
              typename AGridDesc_AK0_M_K1,
              typename BGridDesc_BK0_N_K1,
              typename DsGridDesc_M_N,
              typename CGridDesc_M_N>
    __device__ static void Run_2Lds(const ADataType* __restrict__ p_a_grid,
                                    const BDataType* __restrict__ p_b_grid,
                                    DsGridPointer& p_ds_grid,
                                    CDataType* __restrict__ p_c_grid,
                                    void* __restrict__ p_shared_0,
                                    void* __restrict__ p_shared_1,
                                    const Problem& problem,
                                    [[maybe_unused]] AElementwiseOperation a_element_op,
                                    [[maybe_unused]] BElementwiseOperation b_element_op,
                                    CElementwiseOperation c_element_op,
                                    const Block2CTileMap& block_2_ctile_map,
                                    const AGridDesc_AK0_M_K1& a_grid_desc_ak0_m_ak1,
                                    const BGridDesc_BK0_N_K1& b_grid_desc_bk0_n_bk1,
                                    const DsGridDesc_M_N& ds_grid_desc_m_n,
                                    const CGridDesc_M_N& c_grid_desc_m_n)
    {

        const auto c_grid_desc_mblock_mperblock_nblock_nperblock =
            MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                c_grid_desc_m_n, problem.MBlock, problem.NBlock);

        const auto a_grid_buf =
            make_dynamic_buffer<AddressSpaceEnum::Global,
                                AmdBufferCoherenceEnum::DefaultCoherence,
                                IndexType>(p_a_grid, a_grid_desc_ak0_m_ak1.GetElementSpaceSize());
        const auto b_grid_buf =
            make_dynamic_buffer<AddressSpaceEnum::Global,
                                AmdBufferCoherenceEnum::DefaultCoherence,
                                IndexType>(p_b_grid, b_grid_desc_bk0_n_bk1.GetElementSpaceSize());

        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        if(!block_2_ctile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I0),
                          c_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I2))))
        {
            return;
        }

        const index_t block_m_id = __builtin_amdgcn_readfirstlane(block_work_idx[I0]);
        const index_t block_n_id = __builtin_amdgcn_readfirstlane(block_work_idx[I1]);

        // HACK: this force m/n_block_data_idx_on_grid into SGPR
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_m_id * MPerBlock);

        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_n_id * NPerBlock);

        // lds max alignment
        constexpr auto max_lds_align = math::lcm(AK1Number, BK1Number);

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_ak0_m_ak1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1(get_device_arch());

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_bk0_n_bk1 =
            GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1(get_device_arch());

        auto get_a_blockwise_copy = [&]() {
            if constexpr(DirectLoad)
            {
                return ThreadGroupTensorSliceTransfer_DirectLoad<
                    ThisThreadBlock,
                    Sequence<AK0Number, MPerBlock, AK1Number>,
                    ABlockTransferThreadClusterLengths_AK0_M_AK1,
                    ABlockTransferThreadClusterArrangeOrder,
                    ADataType,
                    ADataType,
                    decltype(a_grid_desc_ak0_m_ak1),
                    decltype(a_block_desc_ak0_m_ak1),
                    ABlockTransferSrcAccessOrder,
                    ABlockTransferSrcVectorDim,
                    2,
                    ABlockTransferSrcScalarPerVector>(
                    a_grid_desc_ak0_m_ak1,
                    make_multi_index(0, m_block_data_idx_on_grid, 0),
                    a_block_desc_ak0_m_ak1,
                    make_multi_index(0, 0, 0));
            }
            else
            {
                return ThreadGroupTensorSliceTransfer_v4r1<
                    ThisThreadBlock,
                    AElementwiseOperation,
                    ck::tensor_operation::element_wise::PassThrough,
                    InMemoryDataOperationEnum::Set,
                    Sequence<AK0Number, MPerBlock, AK1Number>,
                    ABlockTransferThreadClusterLengths_AK0_M_AK1,
                    ABlockTransferThreadClusterArrangeOrder,
                    ADataType,
                    LDSTypeA,
                    decltype(a_grid_desc_ak0_m_ak1),
                    decltype(a_block_desc_ak0_m_ak1),
                    ABlockTransferSrcAccessOrder,
                    Sequence<0, 1, 2>,
                    ABlockTransferSrcVectorDim,
                    2,
                    ABlockTransferSrcScalarPerVector,
                    ABlockTransferDstScalarPerVector_AK1,
                    1,
                    1,
                    AThreadTransferSrcResetCoordinateAfterRun,
                    true,
                    BlockwiseGemmPipe::GlobalBufferNum,
                    IndexType>(a_grid_desc_ak0_m_ak1,
                               make_multi_index(0, m_block_data_idx_on_grid, 0),
                               a_element_op,
                               a_block_desc_ak0_m_ak1,
                               make_multi_index(0, 0, 0),
                               ck::tensor_operation::element_wise::PassThrough{});
            }
        };

        // B matrix blockwise copy
        auto get_b_blockwise_copy = [&]() {
            if constexpr(DirectLoad)
            {
                return ThreadGroupTensorSliceTransfer_DirectLoad<
                    ThisThreadBlock,
                    Sequence<BK0Number, NPerBlock, BK1Number>,
                    BBlockTransferThreadClusterLengths_BK0_N_BK1,
                    BBlockTransferThreadClusterArrangeOrder,
                    BDataType,
                    BDataType,
                    decltype(b_grid_desc_bk0_n_bk1),
                    decltype(b_block_desc_bk0_n_bk1),
                    BBlockTransferSrcAccessOrder,
                    BBlockTransferSrcVectorDim,
                    2,
                    BBlockTransferSrcScalarPerVector>(
                    b_grid_desc_bk0_n_bk1,
                    make_multi_index(0, n_block_data_idx_on_grid, 0),
                    b_block_desc_bk0_n_bk1,
                    make_multi_index(0, 0, 0));
            }
            else
            {
                return ThreadGroupTensorSliceTransfer_v4r1<
                    ThisThreadBlock,
                    BElementwiseOperation,
                    ck::tensor_operation::element_wise::PassThrough,
                    InMemoryDataOperationEnum::Set,
                    Sequence<BK0Number, NPerBlock, BK1Number>,
                    BBlockTransferThreadClusterLengths_BK0_N_BK1,
                    BBlockTransferThreadClusterArrangeOrder,
                    BDataType,
                    LDSTypeB,
                    decltype(b_grid_desc_bk0_n_bk1),
                    decltype(b_block_desc_bk0_n_bk1),
                    BBlockTransferSrcAccessOrder,
                    Sequence<0, 1, 2>,
                    BBlockTransferSrcVectorDim,
                    2,
                    BBlockTransferSrcScalarPerVector,
                    BBlockTransferDstScalarPerVector_BK1,
                    1,
                    1,
                    BThreadTransferSrcResetCoordinateAfterRun,
                    true,
                    BlockwiseGemmPipe::GlobalBufferNum,
                    IndexType>(b_grid_desc_bk0_n_bk1,
                               make_multi_index(0, n_block_data_idx_on_grid, 0),
                               b_element_op,
                               b_block_desc_bk0_n_bk1,
                               make_multi_index(0, 0, 0),
                               ck::tensor_operation::element_wise::PassThrough{});
            }
        };

        auto a_blockwise_copy = get_a_blockwise_copy();
        auto b_blockwise_copy = get_b_blockwise_copy();

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);

        auto a_block_buf_ping = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<LDSTypeA*>(p_shared_0), a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        auto b_block_buf_ping = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<LDSTypeB*>(p_shared_0) +
                a_block_space_size_aligned * sizeof(LDSTypeA) / sizeof(LDSTypeB),
            b_block_desc_bk0_n_bk1.GetElementSpaceSize());

        auto a_block_buf_pong = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<LDSTypeA*>(p_shared_1), a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        auto b_block_buf_pong = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<LDSTypeB*>(p_shared_1) +
                a_block_space_size_aligned * sizeof(LDSTypeA) / sizeof(LDSTypeB),
            b_block_desc_bk0_n_bk1.GetElementSpaceSize());

        auto a_block_bufs = make_tuple(a_block_buf_ping, a_block_buf_pong);
        auto b_block_bufs = make_tuple(b_block_buf_ping, b_block_buf_pong);

        constexpr auto a_block_slice_copy_step = make_multi_index(KPerBlock / AK1Number, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(KPerBlock / BK1Number, 0, 0);

        // Blockwise GEMM pipeline
        static_assert(std::is_default_constructible_v<BlockwiseGemmPipe>);
        auto blockwise_gemm_pipeline = BlockwiseGemmPipe{};
        auto c_thread_buf            = blockwise_gemm_pipeline.GetCThreadBuffer();

        const index_t num_k_block_main_loop = __builtin_amdgcn_readfirstlane(
            (a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2)) /
            KPerBlock);

        blockwise_gemm_pipeline.template Run<HasMainKBlockLoop, TailNum>(a_grid_desc_ak0_m_ak1,
                                                                         a_block_desc_ak0_m_ak1,
                                                                         a_blockwise_copy,
                                                                         a_grid_buf,
                                                                         a_block_bufs,
                                                                         a_block_slice_copy_step,
                                                                         b_grid_desc_bk0_n_bk1,
                                                                         b_block_desc_bk0_n_bk1,
                                                                         b_blockwise_copy,
                                                                         b_grid_buf,
                                                                         b_block_bufs,
                                                                         b_block_slice_copy_step,
                                                                         c_thread_buf,
                                                                         num_k_block_main_loop);
        if constexpr(LargeTensors)
        {
            static_assert(NumDTensor == 0, "Not implemented");
            Base::template RunEpilogue<CGlobalMemoryDataOperation, false, false>(
                blockwise_gemm_pipeline,
                c_grid_desc_mblock_mperblock_nblock_nperblock,
                c_thread_buf,
                block_m_id,
                block_n_id,
                p_shared_0,
                p_c_grid,
                c_element_op);
        }
        else
        {
            // shuffle C and write out
            const auto ds_grid_desc_mblock_mperblock_nblock_nperblock =
                MakeDsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                    ds_grid_desc_m_n, problem.MBlock, problem.NBlock);
            Base::template RunMultiDEpilogue<CGlobalMemoryDataOperation,
                                             DoElementwiseBeforeCShuffle,
                                             false,
                                             false>(blockwise_gemm_pipeline,
                                                    ds_grid_desc_mblock_mperblock_nblock_nperblock,
                                                    c_grid_desc_mblock_mperblock_nblock_nperblock,
                                                    c_thread_buf,
                                                    block_m_id,
                                                    block_n_id,
                                                    p_shared_0,
                                                    p_ds_grid,
                                                    p_c_grid,
                                                    c_element_op);
        }
    }
};

} // namespace ck

#if __clang_major__ >= 23
#pragma clang diagnostic pop
#endif
