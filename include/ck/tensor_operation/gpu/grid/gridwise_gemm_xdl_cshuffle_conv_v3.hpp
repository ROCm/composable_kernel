// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
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

#if __clang_major__ >= 23
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"
#endif
namespace ck {

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CShuffleDataType,
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
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v4,
          typename ComputeTypeA                       = CDataType,
          typename ComputeTypeB                       = ComputeTypeA,
          bool DirectLoad                             = false,
          bool ALdsScalarLoadToVgpr                   = false,
          bool BLdsScalarLoadToVgpr                   = false,
          bool LargeTensors                           = false>
struct GridwiseGemm_xdl_cshuffle_conv_v3
    : public GridwiseGemm_xdl_cshuffle_base<
          ALayout,
          BLayout,
          CLayout,
          ADataType,
          BDataType,
          AccDataType,
          CShuffleDataType,
          Tuple<>,
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
          Sequence<CShuffleBlockTransferScalarPerVector_NPerBlock>,
          ComputeTypeA,
          ComputeTypeB,
          false, // ForceNaiveLayout
          false, // DirectLoad (base default)
          false, // IsMxGemm  (base default)
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
        ADataType,
        BDataType,
        AccDataType,
        CShuffleDataType,
        Tuple<>,
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
        Sequence<CShuffleBlockTransferScalarPerVector_NPerBlock>,
        ComputeTypeA,
        ComputeTypeB,
        false, // ForceNaiveLayout
        false, // DirectLoad (base default)
        false, // IsMxGemm  (base default)
        LargeTensors>;

    using Base::AK0Number;
    using Base::AK1Number;
    using Base::BK0Number;
    using Base::BK1Number;
    using Base::I0;
    using Base::I1;
    using Base::I2;
    using ThisThreadBlock = typename Base::ThisThreadBlock;

    static constexpr bool DirectLoadEnabled = DirectLoad;

    static constexpr auto lcm_AK1_BK1 = math::lcm(AK1Number, BK1Number);
    static constexpr bool is_single_rate_mfma =
        (((is_same<ComputeTypeA, half_t>::value || is_same<ComputeTypeA, bhalf_t>::value) &&
          lcm_AK1_BK1 <= 4) ||
         (is_same<ComputeTypeA, int8_t>::value && lcm_AK1_BK1 <= 8) ||
         ((is_same<ComputeTypeA, f8_t>::value || is_same<ComputeTypeA, bf8_t>::value) &&
#if defined(__gfx125__)
          lcm_AK1_BK1 < 128))
#else
          lcm_AK1_BK1 < 32))
#endif
            ? true
            : false;
    static constexpr auto is_scale_mfma = false;
    static constexpr index_t KPack =
        math::max(lcm_AK1_BK1,
                  MfmaSelector<ComputeTypeA,
                               MPerXdl,
                               NPerXdl,
                               ComputeTypeA,
                               is_single_rate_mfma,
                               is_scale_mfma>::selected_mfma.k_per_blk);

    __host__ static auto CalculateGridSize(IndexType M, IndexType N, index_t KBatch, index_t Batch)
    {
        return std::make_tuple(Block2CTileMap::CalculateGridSize(M, N), KBatch, Batch);
    }

    __host__ static IndexType CalculateMPadded(IndexType M)
    {
        return math::integer_least_multiple(M, MPerBlock);
    }

    __host__ static IndexType CalculateNPadded(IndexType N)
    {
        return math::integer_least_multiple(N, NPerBlock);
    }

    __host__ static IndexType CalculateKPadded(IndexType K)
    {
        return math::integer_divide_ceil(K, KPerBlock) * KPerBlock;
    }

    __host__ static IndexType CalculateAK0Padded(IndexType K, IndexType K_Batch = 1)
    {
        auto K_t = K_Batch * KPerBlock;
        return (K + K_t - 1) / K_t * (KPerBlock / AK1Value);
    }

    __host__ static IndexType CalculateBK0Padded(IndexType K, IndexType K_Batch = 1)
    {
        auto K_t = K_Batch * KPerBlock;
        return (K + K_t - 1) / K_t * (KPerBlock / BK1Value);
    }

    __host__ static IndexType CalculateKPadded(IndexType K, IndexType K_Batch = 1)
    {
        auto K_t = K_Batch * KPerBlock;
        return (K + K_t - 1) / K_t * KPerBlock;
    }

    __host__ static IndexType CalculateKRead(IndexType K, IndexType K_Batch = 1)
    {
        constexpr auto KReadVec = math::lcm(AK1Number, BK1Number);
        auto K_t                = K_Batch * KReadVec;
        return (K + K_t - 1) / K_t * KReadVec;
    }

    __host__ static IndexType CalculateMBlock(IndexType M)
    {
        return math::integer_divide_ceil(M, static_cast<IndexType>(MPerBlock));
    }

    __host__ static IndexType CalculateNBlock(IndexType N)
    {
        return math::integer_divide_ceil(N, static_cast<IndexType>(NPerBlock));
    }

    template <typename GridDesc_K0_MN_K1_T, index_t K0Number, index_t K1Value>
    __host__ __device__ static auto TransformGrid(const GridDesc_K0_MN_K1_T& desc)
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

    template <index_t MNXdlPerWave,
              index_t MNWaves,
              index_t MNPerXdl,
              bool IsKContinous,
              typename TileDesc_K0_MN_K1>
    __host__ __device__ static constexpr auto MakeGemmMmaTileDescriptor(const TileDesc_K0_MN_K1&)
    {
        if constexpr(DirectLoad && IsKContinous)
        {
            constexpr index_t K0 = TileDesc_K0_MN_K1{}.GetLength(Number<0>{});
            constexpr index_t K1 = TileDesc_K0_MN_K1{}.GetLength(Number<2>{});

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
        else
        {
            constexpr index_t K0 = TileDesc_K0_MN_K1{}.GetLength(Number<0>{});
            constexpr index_t K1 = TileDesc_K0_MN_K1{}.GetLength(Number<2>{});

            return transform_tensor_descriptor(
                TileDesc_K0_MN_K1{},
                make_tuple(
                    make_merge_transform_v3_division_mod(make_tuple(Number<K0>{}, Number<K1>{})),
                    make_unmerge_transform(
                        make_tuple(Number<MNXdlPerWave>{}, Number<MNWaves>{}, Number<MNPerXdl>{}))),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
                make_tuple(Sequence<3>{}, Sequence<0, 1, 2>{}));
        }
    }

    template <typename ABlockDesc_AK0_M_AK1>
    __host__ __device__ static constexpr auto
    MakeAMmaTileDescriptor_M0_M1_M2_K(const ABlockDesc_AK0_M_AK1&)
    {
        constexpr index_t MWaves = MPerBlock / (MXdlPerWave * MPerXdl);

        return MakeGemmMmaTileDescriptor<MXdlPerWave,
                                         MWaves,
                                         MPerXdl,
                                         is_same<tensor_layout::gemm::RowMajor, ALayout>::value>(
            ABlockDesc_AK0_M_AK1{});
    }

    template <typename BBlockDesc_BK0_N_BK1>
    __host__ __device__ static constexpr auto
    MakeBMmaTileDescriptor_N0_N1_N2_K(const BBlockDesc_BK0_N_BK1&)
    {
        constexpr index_t NWaves = NPerBlock / (NXdlPerWave * NPerXdl);

        return MakeGemmMmaTileDescriptor<NXdlPerWave,
                                         NWaves,
                                         NPerXdl,
                                         is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value>(
            BBlockDesc_BK0_N_BK1{});
    }

    struct Problem
    {
        __host__ Problem(IndexType M_,
                         IndexType N_,
                         IndexType K_,
                         IndexType StrideA_,
                         IndexType StrideB_,
                         IndexType StrideC_,
                         IndexType KBatch_)
            : M{M_},
              N{N_},
              K{K_},
              StrideA{StrideA_},
              StrideB{StrideB_},
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
        __host__ Argument(const ADataType* p_a_grid_,
                          const BDataType* p_b_grid_,
                          CDataType* p_c_grid_,
                          IndexType M_,
                          IndexType N_,
                          IndexType K_,
                          IndexType StrideA_,
                          IndexType StrideB_,
                          IndexType StrideC_,
                          IndexType k_batch_)
            : Problem{M_, N_, K_, StrideA_, StrideB_, StrideC_, k_batch_},
              p_a_grid{p_a_grid_},
              p_b_grid{p_b_grid_},
              p_c_grid{p_c_grid_}
        {
        }

        const ADataType* p_a_grid;
        const BDataType* p_b_grid;
        CDataType* p_c_grid;
    };

    template <typename DeviceArch>
    __device__ static constexpr auto GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1(DeviceArch)
    {
        if constexpr(DirectLoad)
        {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(AK0Number, Number<MPerBlock>{}, AK1Number),
                    make_tuple(AK1Number, Number<KPerBlock>{}, I1));
            }
            else
            {
                return make_naive_tensor_descriptor(
                    make_tuple(AK0Number, Number<MPerBlock>{}, AK1Number),
                    make_tuple(Number<MPerBlock * AK1Number>{}, I1, Number<MPerBlock>{}));
            }
        }
        else if constexpr(is_same_v<DeviceArch, gfx950_t>)
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

    template <typename DeviceArch>
    __device__ static constexpr auto GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1(DeviceArch)
    {
        if constexpr(DirectLoad)
        {
            if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(BK0Number, Number<NPerBlock>{}, BK1Number),
                    make_tuple(BK1Number, Number<KPerBlock>{}, I1));
            }
            else
            {
                return make_naive_tensor_descriptor(
                    make_tuple(BK0Number, Number<NPerBlock>{}, BK1Number),
                    make_tuple(Number<NPerBlock * BK1Number>{}, I1, Number<NPerBlock>{}));
            }
        }
        else if constexpr(is_same_v<DeviceArch, gfx950_t>)
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

    IS_VALID_COMPILATION_PARAMETER_IMPL(CDataType)

    // Disable vector load from lds to vgpr for direct load (backward weight store with continous M
    // or N dimension)
    // static constexpr bool LdsScalarLoadToVgpr = DirectLoad;
    using BlockwiseGemmPipe = remove_cvref_t<
        decltype(BlockGemmPipeline_Selector<
                 BlkGemmPipelineVer,
                 BlkGemmPipeSched,
                 BlockSize,
                 ADataType,
                 BDataType,
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
                 DirectLoad,
                 false, // TransposeC
                 false, // UseDataCachePrefetch
                 ALdsScalarLoadToVgpr,
                 BLdsScalarLoadToVgpr>())>;

    template <typename DeviceArch>
    __device__ static constexpr index_t GetSharedMemoryNumberOfByte(DeviceArch)
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

        return math::max((a_block_space_size_aligned * sizeof(ADataType) +
                          b_block_space_size_aligned * sizeof(BDataType)),
                         c_block_size * sizeof(CShuffleDataType));
    }

    __host__ static constexpr bool CalculateHasMainKBlockLoop(index_t K)
    {
        const index_t num_loop = K / KPerBlock;

        return BlockwiseGemmPipe::BlockHasHotloop(num_loop);
    }

    __host__ static constexpr TailNumber CalculateKBlockLoopTailNum(index_t K)
    {
        const index_t num_loop = K / KPerBlock;

        return BlockwiseGemmPipe::BlockLoopTailNum(num_loop);
    }

    template <index_t N>
    using NumberType =
        std::conditional_t<std::is_same_v<IndexType, index_t>, Number<N>, LongNumber<N>>;

    template <typename CGridDesc>
    __host__ __device__ static constexpr auto MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
        const CGridDesc& c_grid_desc_m_n, IndexType MBlock, IndexType NBlock)
    {
        const auto c_grid_desc_mblock_mperblock_nblock_nperblock = transform_tensor_descriptor(
            c_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, NumberType<MPerBlock>{})),
                       make_unmerge_transform(make_tuple(NBlock, NumberType<NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

        return c_grid_desc_mblock_mperblock_nblock_nperblock;
    }

    // return block_id to C matrix tile idx (m0, n0) mapping
    // if arch = gfx942
    using Block2CTileMap =
        BlockToCTileMap_Grouped_M00_N0_M01Adapt<8, MPerBlock, NPerBlock, IndexType>;

    template <typename AGridDesc_AK0_M_K1,
              typename BGridDesc_BK0_N_K1,
              typename CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
              bool HasMainKBlockLoop,
              InMemoryDataOperationEnum CGlobalMemoryDataOperation,
              TailNumber TailNum    = TailNumber::Odd,
              bool SplitKOffsetHack = false>
    __device__ static void Run(const ADataType* p_a_grid,
                               const BDataType* p_b_grid,
                               CDataType* p_c_grid,
                               void* p_shared,
                               const Problem& problem,
                               const AGridDesc_AK0_M_K1& a_grid_desc_ak0_m_ak1,
                               const BGridDesc_BK0_N_K1& b_grid_desc_bk0_n_bk1,
                               const CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock&
                                   c_grid_desc_mblock_mperblock_nblock_nperblock,
                               const index_t k_id        = 0,
                               const index_t k_batch     = 1,
                               const index_t block_idx_x = static_cast<index_t>(blockIdx.x))
    {
        const long_index_t a_space_size_divisor = SplitKOffsetHack ? k_batch : 1;
        const long_index_t b_space_size_divisor = SplitKOffsetHack ? k_batch : 1;

        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global,
                                                    AmdBufferCoherenceEnum::DefaultCoherence,
                                                    IndexType>(
            p_a_grid, a_grid_desc_ak0_m_ak1.GetElementSpaceSize() / a_space_size_divisor);
        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global,
                                                    AmdBufferCoherenceEnum::DefaultCoherence,
                                                    IndexType>(
            p_b_grid, b_grid_desc_bk0_n_bk1.GetElementSpaceSize() / b_space_size_divisor);

        const AElementwiseOperation a_element_op{};
        const BElementwiseOperation b_element_op{};
        const CElementwiseOperation c_element_op{};

        // divide block work by [M, N]
        const auto block_2_ctile_map = Block2CTileMap{problem.M, problem.N, 4};

        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(block_idx_x));

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
                return ThreadGroupTensorSliceTransfer_DirectLoad < ThisThreadBlock,
                       Sequence<AK0Number, MPerBlock, AK1Number>,
                       ABlockTransferThreadClusterLengths_AK0_M_AK1,
                       ABlockTransferThreadClusterArrangeOrder, ADataType, ADataType,
                       decltype(a_grid_desc_ak0_m_ak1), decltype(a_block_desc_ak0_m_ak1),
                       ABlockTransferSrcAccessOrder, ABlockTransferSrcVectorDim,
                       is_same<tensor_layout::gemm::RowMajor, ALayout>::value ? 2 : 1,
                       ABlockTransferSrcScalarPerVector >
                           (a_grid_desc_ak0_m_ak1,
                            make_multi_index(
                                SplitKOffsetHack ? 0 : k_id, m_block_data_idx_on_grid, 0),
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
                    ADataType,
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
                    IndexType>(
                    a_grid_desc_ak0_m_ak1,
                    make_multi_index(SplitKOffsetHack ? 0 : k_id, m_block_data_idx_on_grid, 0),
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
                return ThreadGroupTensorSliceTransfer_DirectLoad < ThisThreadBlock,
                       Sequence<BK0Number, NPerBlock, BK1Number>,
                       BBlockTransferThreadClusterLengths_BK0_N_BK1,
                       BBlockTransferThreadClusterArrangeOrder, BDataType, BDataType,
                       decltype(b_grid_desc_bk0_n_bk1), decltype(b_block_desc_bk0_n_bk1),
                       BBlockTransferSrcAccessOrder, BBlockTransferSrcVectorDim,
                       is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value ? 2 : 1,
                       BBlockTransferSrcScalarPerVector >
                           (b_grid_desc_bk0_n_bk1,
                            make_multi_index(
                                SplitKOffsetHack ? 0 : k_id, n_block_data_idx_on_grid, 0),
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
                    BDataType,
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
                    IndexType>(
                    b_grid_desc_bk0_n_bk1,
                    make_multi_index(SplitKOffsetHack ? 0 : k_id, n_block_data_idx_on_grid, 0),
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
            static_cast<ADataType*>(p_shared), a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        auto b_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<BDataType*>(p_shared) +
                a_block_space_size_aligned * sizeof(ADataType) / sizeof(BDataType),
            b_block_desc_bk0_n_bk1.GetElementSpaceSize());

        constexpr auto a_block_slice_copy_step = make_multi_index(KPerBlock / AK1Number, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(KPerBlock / BK1Number, 0, 0);

        // Blockwise GEMM pipeline
        static_assert(std::is_default_constructible_v<BlockwiseGemmPipe>);
        auto blockwise_gemm_pipeline = BlockwiseGemmPipe{};
        auto c_thread_buf            = blockwise_gemm_pipeline.GetCThreadBuffer();

        const index_t num_k_block_main_loop = __builtin_amdgcn_readfirstlane(
            (a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2)) /
            (KPerBlock * problem.KBatch));

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

        // shuffle C and write out
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

    template <typename AGridDesc_AK0_M_K1,
              typename BGridDesc_BK0_N_K1,
              typename CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
              bool HasMainKBlockLoop,
              InMemoryDataOperationEnum CGlobalMemoryDataOperation,
              TailNumber TailNum    = TailNumber::Odd,
              bool SplitKOffsetHack = false>
    __device__ static void Run_2Lds(const ADataType* p_a_grid,
                                    const BDataType* p_b_grid,
                                    CDataType* p_c_grid,
                                    void* p_shared_0,
                                    void* p_shared_1,
                                    const Problem& problem,
                                    const AGridDesc_AK0_M_K1& a_grid_desc_ak0_m_ak1,
                                    const BGridDesc_BK0_N_K1& b_grid_desc_bk0_n_bk1,
                                    const CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock&
                                        c_grid_desc_mblock_mperblock_nblock_nperblock,
                                    const index_t k_id        = 0,
                                    const index_t k_batch     = 1,
                                    const index_t block_idx_x = static_cast<index_t>(blockIdx.x))
    {
        const long_index_t a_space_size_divisor = SplitKOffsetHack ? k_batch : 1;
        const long_index_t b_space_size_divisor = SplitKOffsetHack ? k_batch : 1;

        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global,
                                                    AmdBufferCoherenceEnum::DefaultCoherence,
                                                    IndexType>(
            p_a_grid, a_grid_desc_ak0_m_ak1.GetElementSpaceSize() / a_space_size_divisor);
        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global,
                                                    AmdBufferCoherenceEnum::DefaultCoherence,
                                                    IndexType>(
            p_b_grid, b_grid_desc_bk0_n_bk1.GetElementSpaceSize() / b_space_size_divisor);

        const AElementwiseOperation a_element_op{};
        const BElementwiseOperation b_element_op{};
        const CElementwiseOperation c_element_op{};

        // divide block work by [M, N]
        const auto block_2_ctile_map = Block2CTileMap{problem.M, problem.N, 4};

        const auto block_work_idx = block_2_ctile_map.CalculateBottomIndex(
            make_multi_index(static_cast<index_t>(block_idx_x)));

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
                return ThreadGroupTensorSliceTransfer_DirectLoad < ThisThreadBlock,
                       Sequence<AK0Number, MPerBlock, AK1Number>,
                       ABlockTransferThreadClusterLengths_AK0_M_AK1,
                       ABlockTransferThreadClusterArrangeOrder, ADataType, ADataType,
                       decltype(a_grid_desc_ak0_m_ak1), decltype(a_block_desc_ak0_m_ak1),
                       ABlockTransferSrcAccessOrder, ABlockTransferSrcVectorDim,
                       is_same<tensor_layout::gemm::RowMajor, ALayout>::value ? 2 : 1,
                       ABlockTransferSrcScalarPerVector >
                           (a_grid_desc_ak0_m_ak1,
                            make_multi_index(
                                SplitKOffsetHack ? 0 : k_id, m_block_data_idx_on_grid, 0),
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
                    ADataType,
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
                    IndexType>(
                    a_grid_desc_ak0_m_ak1,
                    make_multi_index(SplitKOffsetHack ? 0 : k_id, m_block_data_idx_on_grid, 0),
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
                return ThreadGroupTensorSliceTransfer_DirectLoad < ThisThreadBlock,
                       Sequence<BK0Number, NPerBlock, BK1Number>,
                       BBlockTransferThreadClusterLengths_BK0_N_BK1,
                       BBlockTransferThreadClusterArrangeOrder, BDataType, BDataType,
                       decltype(b_grid_desc_bk0_n_bk1), decltype(b_block_desc_bk0_n_bk1),
                       BBlockTransferSrcAccessOrder, BBlockTransferSrcVectorDim,
                       is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value ? 2 : 1,
                       BBlockTransferSrcScalarPerVector >
                           (b_grid_desc_bk0_n_bk1,
                            make_multi_index(
                                SplitKOffsetHack ? 0 : k_id, n_block_data_idx_on_grid, 0),
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
                    BDataType,
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
                    IndexType>(
                    b_grid_desc_bk0_n_bk1,
                    make_multi_index(SplitKOffsetHack ? 0 : k_id, n_block_data_idx_on_grid, 0),
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
            static_cast<ADataType*>(p_shared_0), a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        auto b_block_buf_ping = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<BDataType*>(p_shared_0) +
                a_block_space_size_aligned * sizeof(ADataType) / sizeof(BDataType),
            b_block_desc_bk0_n_bk1.GetElementSpaceSize());

        auto a_block_buf_pong = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<ADataType*>(p_shared_1), a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        auto b_block_buf_pong = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<BDataType*>(p_shared_1) +
                a_block_space_size_aligned * sizeof(ADataType) / sizeof(BDataType),
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
            (KPerBlock * problem.KBatch));

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

        // shuffle C and write out
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
};

} // namespace ck

#if __clang_major__ >= 23
#pragma clang diagnostic pop
#endif
