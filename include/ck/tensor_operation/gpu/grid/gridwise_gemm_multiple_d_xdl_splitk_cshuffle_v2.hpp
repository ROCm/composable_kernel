// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"

#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v6r1.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/utility/reduction_functions_accumulate.hpp"

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
          typename ComputeType,
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename EDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          InMemoryDataOperationEnum EGlobalMemoryDataOperation,
          tensor_operation::device::GemmSpecialization GemmSpec,
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
          PipelineVersion PipelineVer>
class GridwiseGemmMultipleD_xdl_splitk_cshuffle_v2
{
    static constexpr index_t NumDTensor = DsDataType::Size();

    using GemmSpecialization = ck::tensor_operation::device::GemmSpecialization;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};

    static constexpr auto AK1         = Number<AK1Value>{};
    static constexpr auto BK1         = Number<BK1Value>{};
    static constexpr auto AK0PerBlock = Number<KPerBlock / AK1Value>{};
    static constexpr auto BK0PerBlock = Number<KPerBlock / BK1Value>{};

    static constexpr index_t KPack = math::max(
        math::lcm(AK1, BK1), MfmaSelector<ComputeType, MPerXdl, NPerXdl>::selected_mfma.k_per_blk);

    using ThisThreadBlock  = ThisThreadBlock<BlockSize>;
    using GridwiseGemmPipe = remove_cvref_t<
        decltype(GridwiseGemmPipeline_Selector<PipelineVer, NumGemmKPrefetchStage, LoopSched>())>;

    public:
    using AccType       = AccDataType;
    using CShuffleDataT = CShuffleDataType;

    __host__ __device__ static auto CalculateMPadded(index_t M)
    {
        return math::integer_least_multiple(M, MPerBlock);
    }

    __host__ __device__ static auto CalculateNPadded(index_t N)
    {
        return math::integer_least_multiple(N, NPerBlock);
    }

    __host__ __device__ static auto CalculateKPadded(index_t K)
    {
        return math::integer_least_multiple(K, KPerBlock);
    }

    __host__ __device__ static constexpr auto GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1()
    {
        // A matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(AK0PerBlock, Number<MPerBlock>{}, AK1),
            make_tuple(Number<MPerBlock + ABlockLdsExtraM>{} * AK1, AK1, I1));
    }

    __host__ __device__ static constexpr auto GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1()
    {
        // B matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(BK0PerBlock, Number<NPerBlock>{}, BK1),
            make_tuple(Number<NPerBlock + BBlockLdsExtraN>{} * BK1, BK1, I1));
    }

    __host__ __device__ static auto
    MakeAGridDescriptor_AK0_M_AK1(index_t M, index_t K, index_t StrideA)
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

        const auto KPad = CalculateKPadded(K);

        const auto a_grid_desc_m_kpad = transform_tensor_descriptor(
            a_grid_desc_m_k,
            make_tuple(make_pass_through_transform(M), make_right_pad_transform(K, KPad - K)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        const auto AK0 = KPad / AK1;

        if constexpr(GemmSpec == GemmSpecialization::MPadding ||
                     GemmSpec == GemmSpecialization::MNPadding ||
                     GemmSpec == GemmSpecialization::MKPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            const auto MPad = CalculateMPadded(M);
            return transform_tensor_descriptor(
                a_grid_desc_m_kpad,
                make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                           make_right_pad_transform(M, MPad - M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                a_grid_desc_m_kpad,
                make_tuple(make_unmerge_transform(make_tuple(AK0, AK1)),
                           make_pass_through_transform(M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
    }

    __host__ __device__ static auto
    MakeBGridDescriptor_BK0_N_BK1(index_t K, index_t N, index_t StrideB)
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
        const auto KPad = CalculateKPadded(K);

        const auto b_grid_desc_kpad_n = transform_tensor_descriptor(
            b_grid_desc_k_n,
            make_tuple(make_right_pad_transform(K, KPad - K), make_pass_through_transform(N)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        const auto BK0 = KPad / BK1;

        if constexpr(GemmSpec == GemmSpecialization::NPadding ||
                     GemmSpec == GemmSpecialization::MNPadding ||
                     GemmSpec == GemmSpecialization::NKPadding ||
                     GemmSpec == GemmSpecialization::MNKPadding)
        {
            // const auto PadN = (NPerBlock - N % NPerBlock) % NPerBlock;
            return transform_tensor_descriptor(
                b_grid_desc_kpad_n,
                make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                           make_right_pad_transform(N, NPad - N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                b_grid_desc_kpad_n,
                make_tuple(make_unmerge_transform(make_tuple(BK0, BK1)),
                           make_pass_through_transform(N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
    }

    private:
    using AGridDesc_AK0_M_AK1 = remove_cvref_t<decltype(MakeAGridDescriptor_AK0_M_AK1(1, 1, 1))>;
    using BGridDesc_BK0_N_BK1 = remove_cvref_t<decltype(MakeBGridDescriptor_BK0_N_BK1(1, 1, 1))>;

    using ABlockDesc_AK0PerB_MPerB_AK1 =
        remove_cvref_t<decltype(GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1())>;
    using BBlockDesc_BK0PerB_NPerB_BK1 =
        remove_cvref_t<decltype(GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1())>;

    public:
    __host__ __device__ static constexpr auto
    GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock()
    {
        constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t NWave = NPerBlock / (NXdlPerWave * NPerXdl);

        constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            make_naive_tensor_descriptor_packed(
                make_tuple(I1,
                           Number<CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl>{},
                           I1,
                           Number<CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>{}));

        return c_shuffle_block_desc_mblock_mperblock_nblock_nperblock;
    }

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

    using DsGridPointer = decltype(MakeDsGridPointer());

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

        // LDS allocation for C shuffle in LDS
        constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock();

        constexpr auto c_block_size =
            c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize();

        return math::max((a_block_space_size_aligned + b_block_space_size_aligned) *
                             sizeof(ComputeType),
                         c_block_size * sizeof(CShuffleDataType));
    }

    // M0 - MBlock
    // M1 - MPerBlock
    // N0 - NBlock
    // N1 - N repeats
    // N2 - NVecSize * cluster length
    template <typename EGridDesc_M_N>
    __host__ __device__ static constexpr auto
    MakeEGridDescriptor_M0M1_N0N1N2(const EGridDesc_M_N& e_grid_desc_m_n)
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

        constexpr auto cluster_length_reduce             = GetClusterLengthReduction_M0_N0N1();
        constexpr auto workspace_thread_desc_m0m1_n0n1n2 = MakeReductionThreadDesc_M0M1_N0N1N2();

        const auto e_grid_desc_m0m1_n0n1n2 = transform_tensor_descriptor(
            e_grid_desc_mblock_mperblock_nblock_nperblock,
            make_tuple(
                make_pass_through_transform(
                    e_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I0)),
                make_pass_through_transform(
                    e_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I1)),
                make_pass_through_transform(
                    e_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I2)),
                make_unmerge_transform(make_tuple(workspace_thread_desc_m0m1_n0n1n2.GetLength(I3),
                                                  workspace_thread_desc_m0m1_n0n1n2.GetLength(I4) *
                                                      cluster_length_reduce.At(I2)))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3, 4>{}));

        return e_grid_desc_m0m1_n0n1n2;
    }

    // Ds desc for source in blockwise copy
    template <typename DsGridDesc_M_N>
    __host__ __device__ static constexpr auto
    MakeDsGridDescriptor_M0M1_N0N1N2(const DsGridDesc_M_N& ds_grid_desc_m_n)
    {
        return generate_tuple(
            [&](auto i) { return MakeEGridDescriptor_M0M1_N0N1N2(ds_grid_desc_m_n[i]); },
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

    __host__ __device__ static constexpr bool
    CheckValidity(const index_t M,
                  const index_t N,
                  const index_t K,
                  const index_t StrideA,
                  const index_t StrideB,
                  [[maybe_unused]] const std::array<index_t, NumDTensor> StrideDs,
                  const index_t StrideE,
                  const index_t KBatch)
    {
        const auto a_grid_desc_ak0_m_ak1 = MakeAGridDescriptor_AK0_M_AK1(M, K, StrideA);
        const auto b_grid_desc_bk0_n_bk1 = MakeBGridDescriptor_BK0_N_BK1(K, N, StrideB);
        const auto e_grid_desc_m_n       = MakeEGridDescriptor_M_N<ELayout>(M, N, StrideE);

        const auto IsMPadded = []() -> bool {
            if constexpr(GemmSpec == tensor_operation::device::GemmSpecialization::MPadding ||
                         GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding ||
                         GemmSpec == tensor_operation::device::GemmSpecialization::MKPadding ||
                         GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding)
                return true;
            else
                return false;
        };

        const auto IsNPadded = []() -> bool {
            if constexpr(GemmSpec == tensor_operation::device::GemmSpecialization::NPadding ||
                         GemmSpec == tensor_operation::device::GemmSpecialization::MNPadding ||
                         GemmSpec == tensor_operation::device::GemmSpecialization::NKPadding ||
                         GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding)
                return true;
            else
                return false;
        };

        const auto IsKPadded = []() -> bool {
            if constexpr(GemmSpec == tensor_operation::device::GemmSpecialization::KPadding ||
                         GemmSpec == tensor_operation::device::GemmSpecialization::MKPadding ||
                         GemmSpec == tensor_operation::device::GemmSpecialization::NKPadding ||
                         GemmSpec == tensor_operation::device::GemmSpecialization::MNKPadding)
                return true;
            else
                return false;
        };

        if constexpr(!IsMPadded())
        {
            if(!(M % MPerBlock == 0))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg M value is not a multiple of MPerBlock! M: " << M << " "
                              << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                              << std::endl;
                }
                return false;
            }
        }

        if constexpr(!IsNPadded())
        {
            if(!(N % NPerBlock == 0))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg N value is not a multiple of NPerBlock! N: " << N << " "
                              << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                              << std::endl;
                }

                return false;
            }
        }

        if constexpr(!IsKPadded())
        {
            if(!(K % KPerBlock == 0))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg K value is not a multiple of ! KPerBlock: " << K << " "
                              << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                              << std::endl;
                }
                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
        {
            if(K % ABlockTransferSrcScalarPerVector != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg K (" << K
                              << ") value is not a multiple of ABlockTransferSrcScalarPerVector ("
                              << ABlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }
        else
        {
            if(M % ABlockTransferSrcScalarPerVector != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg M (" << M
                              << ") value is not a multiple of ABlockTransferSrcScalarPerVector ("
                              << ABlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
        {
            if(N % BBlockTransferSrcScalarPerVector != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg N (" << N
                              << ") value is not a multiple of BBlockTransferSrcScalarPerVector ("
                              << BBlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }
        else
        {
            if(K % BBlockTransferSrcScalarPerVector != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg K (" << K
                              << ") value is not a multiple of BBlockTransferSrcScalarPerVector ("
                              << BBlockTransferSrcScalarPerVector << " )! " << __FILE__ << ":"
                              << __LINE__ << ", in function: " << __func__ << std::endl;
                }
                return false;
            }
        }

        if constexpr(is_same<tensor_layout::gemm::RowMajor, ELayout>::value)
        {
            if(N % CDEShuffleBlockTransferScalarPerVector_NPerBlock != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg N (" << N
                              << ") value is not a multiple of "
                                 "CDEShuffleBlockTransferScalarPerVector_NPerBlock ("
                              << CDEShuffleBlockTransferScalarPerVector_NPerBlock << " )! "
                              << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                              << std::endl;
                }
                return false;
            }
        }
        else
        {
            if(M % CDEShuffleBlockTransferScalarPerVector_NPerBlock != 0)
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    std::cout << "Arg M (" << M
                              << ") value is not a multiple of "
                                 "CDEShuffleBlockTransferScalarPerVector_NPerBlock ("
                              << CDEShuffleBlockTransferScalarPerVector_NPerBlock << " )! "
                              << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                              << std::endl;
                }
                return false;
            }
        }

        const auto k_batch_size =
            (a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2)) / KBatch;

        if(k_batch_size < KPerBlock)
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "The k-batch size (" << k_batch_size
                          << ") value is less than KPerBlock!\n"
                          << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                          << std::endl;
            }
            return false;
        }

        if(k_batch_size % KPerBlock != 0)
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "The k-batch size (" << k_batch_size
                          << ") value is not a multiple of KPerBlock!\n"
                          << __FILE__ << ":" << __LINE__ << ", in function: " << __func__
                          << std::endl;
            }
            return false;
        }

        // check gridwise gemm pipeline
        // This does not take into account that each WGP can run multiple kbatch tiles
        // However that information is dynamic at kernel run-time.
        // So this condition may be too restrictive.
        const auto num_k_loop =
            (a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2)) /
            (KPerBlock * KBatch);

        if(!GridwiseGemmPipe::IsSupported(num_k_loop))
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                std::cout << "The number of k loops (" << num_k_loop
                          << ") value is not supported by GridwiseGemm Pipeline."
                          << " AK0Padded: " << a_grid_desc_ak0_m_ak1.GetLength(I0) << __FILE__
                          << ":" << __LINE__ << ", in function: " << __func__ << std::endl;
            }
            return false;
        }

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        // check tensor size: cannot be larger than 2GB each
        constexpr long_index_t TwoGB = (long_index_t{1} << 31);

        if(!(a_grid_desc_ak0_m_ak1.GetElementSpaceSize() * sizeof(ADataType) <= TwoGB &&
             b_grid_desc_bk0_n_bk1.GetElementSpaceSize() * sizeof(BDataType) <= TwoGB &&
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

    template <typename TensorDataLayout>
    __host__ __device__ static auto
    MakeEGridDescriptor_M_N(index_t MRaw, index_t NRaw, index_t StrideE)
    {
        constexpr auto matrix_padder =
            ck::tensor_operation::device::MatrixPadder<GemmSpec, index_t, index_t, index_t>{
                MPerBlock, NPerBlock, KPerBlock};
        const auto e_grid_desc_mraw_nraw = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, TensorDataLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(StrideE, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, TensorDataLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(MRaw, NRaw),
                                                    make_tuple(I1, StrideE));
            }
        }();

        return matrix_padder.PadCDescriptor_M_N(e_grid_desc_mraw_nraw);
    }

    __host__ __device__ static auto
    MakeDsGridDescriptor_M_N(const std::array<index_t, NumDTensor>& MRaws,
                             const std::array<index_t, NumDTensor>& NRaws,
                             const std::array<index_t, NumDTensor>& DsStride)
    {
        return generate_tuple(
            [&](auto i) {
                using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

                return MakeEGridDescriptor_M_N<DLayout>(MRaws[i], NRaws[i], DsStride[i]);
            },
            Number<NumDTensor>{});
    }

    __host__ __device__ static auto
    MakeWorkspaceGridDesc_GridSize_MPerBlock_I1_NPerBlock(index_t grid_size)
    {
        return make_naive_tensor_descriptor(
            make_tuple(grid_size, MPerBlock, I1.value, NPerBlock),
            make_tuple(MPerBlock * NPerBlock, NPerBlock, NPerBlock, I1.value));
    }

    __device__ __host__ static constexpr auto GetMPerBlock() { return MPerBlock; }
    __device__ __host__ static constexpr auto GetNPerBlock() { return NPerBlock; }
    __device__ __host__ static constexpr auto GetMXdlPerWave() { return MXdlPerWave; }
    __device__ __host__ static constexpr auto GetNXdlPerWave() { return NXdlPerWave; }

    __device__ static constexpr auto GetCThreadBufferVectorSize()
    {
        using BlockwiseGemmT =
            remove_cvref_t<decltype(BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_Selector<
                                    BlockSize,
                                    ComputeType,
                                    ComputeType,
                                    AccDataType,
                                    decltype(GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1()),
                                    decltype(GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1()),
                                    MPerXdl,
                                    NPerXdl,
                                    MXdlPerWave,
                                    NXdlPerWave,
                                    KPack,
                                    LoopSched>())>;
        return BlockwiseGemmT::xdlops_gemm.GetRegSizePerXdlops();
    }

    template <typename Block2ETileMap, typename CThreadBuf>
    __device__ static void RunGEMM(const ADataType* __restrict__ p_a_grid,
                                   const BDataType* __restrict__ p_b_grid,
                                   void* __restrict__ p_shared,
                                   const AElementwiseOperation& a_element_op,
                                   const BElementwiseOperation& b_element_op,
                                   const AGridDesc_AK0_M_AK1& a_grid_desc_ak0_m_ak1,
                                   const BGridDesc_BK0_N_BK1& b_grid_desc_bk0_n_bk1,
                                   const Block2ETileMap& block_2_etile_map,
                                   CThreadBuf& c_thread_buf,
                                   const index_t k_batch,
                                   const index_t next_k_tiles)
    {
        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc_ak0_m_ak1.GetElementSpaceSize());

        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_grid_desc_bk0_n_bk1.GetElementSpaceSize());

        // divide block work by [M, N, K]
        const auto block_work_idx = block_2_etile_map.GetBottomIndex();

        const index_t kbatch_id = __builtin_amdgcn_readfirstlane(block_work_idx[I2]);
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * MPerBlock);
        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * NPerBlock);

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_ak0_m_ak1 = GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();
        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_bk0_n_bk1 = GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        using ABlockwiseCopy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                AElementwiseOperation,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<AK0PerBlock, MPerBlock, AK1>,
                                                ABlockTransferThreadClusterLengths_AK0_M_AK1,
                                                ABlockTransferThreadClusterArrangeOrder,
                                                ADataType,
                                                ComputeType,
                                                AGridDesc_AK0_M_AK1,
                                                ABlockDesc_AK0PerB_MPerB_AK1,
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
                                                NumGemmKPrefetchStage>;

        using BBlockwiseCopy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                BElementwiseOperation,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<BK0PerBlock, NPerBlock, BK1>,
                                                BBlockTransferThreadClusterLengths_BK0_N_BK1,
                                                BBlockTransferThreadClusterArrangeOrder,
                                                BDataType,
                                                ComputeType,
                                                BGridDesc_BK0_N_BK1,
                                                BBlockDesc_BK0PerB_NPerB_BK1,
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
                                                NumGemmKPrefetchStage>;

        const index_t num_k_tiles_per_batch =
            (a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2)) /
            (KPerBlock * k_batch);
        const index_t ak0_start_idx = kbatch_id * num_k_tiles_per_batch * AK0PerBlock.value;
        const index_t bk0_start_idx = kbatch_id * num_k_tiles_per_batch * BK0PerBlock.value;

        // A matrix blockwise copy
        auto a_blockwise_copy =
            ABlockwiseCopy(a_grid_desc_ak0_m_ak1,
                           make_multi_index(ak0_start_idx, m_block_data_idx_on_grid, 0),
                           a_element_op,
                           a_block_desc_ak0_m_ak1,
                           make_multi_index(0, 0, 0),
                           ck::tensor_operation::element_wise::PassThrough{});

        // B matrix blockwise copy
        auto b_blockwise_copy =
            BBlockwiseCopy(b_grid_desc_bk0_n_bk1,
                           make_multi_index(bk0_start_idx, n_block_data_idx_on_grid, 0),
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
        // auto& c_thread_buf = blockwise_gemm_.GetCThreadBuffer();

        // lds max alignment
        constexpr auto max_lds_align = math::lcm(AK1, BK1);

        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);

        auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<ComputeType*>(p_shared), a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        auto b_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<ComputeType*>(p_shared) + a_block_space_size_aligned,
            b_block_desc_bk0_n_bk1.GetElementSpaceSize());

        constexpr auto a_block_slice_copy_step = make_multi_index(AK0PerBlock.value, 0, 0);
        constexpr auto b_block_slice_copy_step = make_multi_index(BK0PerBlock.value, 0, 0);

        // gridwise GEMM pipeline
        const auto gridwise_gemm_pipeline = GridwiseGemmPipe();
        const index_t num_k_block_main_loop =
            __builtin_amdgcn_readfirstlane(next_k_tiles * num_k_tiles_per_batch);
        const bool has_k_block_main_loop =
            gridwise_gemm_pipeline.CalculateHasMainLoop(num_k_block_main_loop);

        auto blockwise_gemm = BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_Selector<
            BlockSize,
            ComputeType,
            ComputeType,
            AccDataType,
            decltype(GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1()),
            decltype(GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1()),
            MPerXdl,
            NPerXdl,
            MXdlPerWave,
            NXdlPerWave,
            KPack,
            LoopSched>();

        if(has_k_block_main_loop)
        {
            gridwise_gemm_pipeline.template Run<true>(a_grid_desc_ak0_m_ak1,
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
        }
        else
        {
            gridwise_gemm_pipeline.template Run<false>(a_grid_desc_ak0_m_ak1,
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
        }
    }

    template <typename Block2ETileMap, typename CThreadBuf>
    __device__ static void RunGEMM(const void* __restrict__ p_a_grid_,
                                   const void* __restrict__ p_b_grid_,
                                   void* __restrict__ p_shared,
                                   const AElementwiseOperation& a_element_op,
                                   const BElementwiseOperation& b_element_op,
                                   const index_t M,
                                   const index_t N,
                                   const index_t K,
                                   const index_t stride_a,
                                   const index_t stride_b,
                                   const index_t k_batch,
                                   const Block2ETileMap& block_2_etile_map,
                                   CThreadBuf& c_thread_buf,
                                   const index_t next_k_tiles)
    {
        const auto p_a_grid = reinterpret_cast<const ADataType*>(p_a_grid_);
        const auto p_b_grid = reinterpret_cast<const BDataType*>(p_b_grid_);

        // tensor descriptors for block/thread-wise copy
        const auto a_grid_desc_ak0_m_ak1 = MakeAGridDescriptor_AK0_M_AK1(M, K, stride_a);
        const auto b_grid_desc_bk0_n_bk1 = MakeBGridDescriptor_BK0_N_BK1(K, N, stride_b);

        RunGEMM(p_a_grid,
                p_b_grid,
                p_shared,
                a_element_op,
                b_element_op,
                a_grid_desc_ak0_m_ak1,
                b_grid_desc_bk0_n_bk1,
                block_2_etile_map,
                c_thread_buf,
                k_batch,
                next_k_tiles);
    }

    template <typename CThreadBuf>
    __device__ static void StorePartials(void* __restrict__ p_workspace,
                                         void* __restrict__ p_shared,
                                         const CThreadBuf& c_thread_buf)
    {
        using BlockwiseGemmT =
            remove_cvref_t<decltype(BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_Selector<
                                    BlockSize,
                                    ComputeType,
                                    ComputeType,
                                    AccDataType,
                                    decltype(GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1()),
                                    decltype(GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1()),
                                    MPerXdl,
                                    NPerXdl,
                                    MXdlPerWave,
                                    NXdlPerWave,
                                    KPack,
                                    LoopSched>())>;

        static_assert(MXdlPerWave % CShuffleMXdlPerWavePerShuffle == 0 &&
                          NXdlPerWave % CShuffleNXdlPerWavePerShuffle == 0,
                      "MXdlPerWave % CShuffleMXdlPerWavePerShuffle != 0 or "
                      "NXdlPerWave % CShuffleNXdlPerWavePerShuffle != 0,");

        constexpr auto c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp =
            BlockwiseGemmT::GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

        constexpr auto M0 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I0);
        constexpr auto N0 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I1);
        constexpr auto M1 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I2);
        constexpr auto N1 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I3);
        constexpr auto M2 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I4);
        constexpr auto M3 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I5);
        constexpr auto M4 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I6);
        constexpr auto N2 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I7);

        constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t NWave = NPerBlock / (NXdlPerWave * NPerXdl);

        constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock();

        auto c_shuffle_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<CShuffleDataType*>(p_shared),
            c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

        constexpr auto c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2 = transform_tensor_descriptor(
            c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
            make_tuple(make_freeze_transform(I0),
                       make_unmerge_transform(make_tuple(
                           Number<CShuffleMXdlPerWavePerShuffle>{}, // M0 (MXdlPerWave) per shuffle
                           M1,                                      // M1 = MWave
                           M2,                                      // M2 * M3 * M4 = MPerXdl
                           M3,
                           M4)),
                       make_freeze_transform(I0),
                       make_unmerge_transform(make_tuple(
                           Number<CShuffleNXdlPerWavePerShuffle>{}, // N0 (NXdlPerWave) per shuffle
                           N1,                                      // N1 = NWave
                           N2))),                                   // N2 = NPerXdl
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<>{}, Sequence<0, 2, 4, 5, 6>{}, Sequence<>{}, Sequence<1, 3, 7>{}));

        // calculate origin of thread output tensor on global memory
        //     blockwise GEMM c matrix starting index
        constexpr auto c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
            BlockwiseGemmT::GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

        const auto c_thread_mtx_on_block =
            BlockwiseGemmT::CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

        const index_t m_thread_data_on_block = c_thread_mtx_on_block[I0];
        const index_t n_thread_data_on_block = c_thread_mtx_on_block[I1];

        const auto m_thread_data_on_block_to_m0_m1_m2_m3_m4_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(M0, M1, M2, M3, M4))),
                make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                make_tuple(Sequence<0>{}));

        const auto m_thread_data_on_block_idx =
            m_thread_data_on_block_to_m0_m1_m2_m3_m4_adaptor.CalculateBottomIndex(
                make_multi_index(m_thread_data_on_block));

        const auto n_thread_data_on_block_to_n0_n1_n2_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(N0, N1, N2))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));

        const auto n_thread_data_on_block_idx =
            n_thread_data_on_block_to_n0_n1_n2_adaptor.CalculateBottomIndex(
                make_multi_index(n_thread_data_on_block));

        // shuffle: threadwise copy C from VGPR to LDS
        auto c_thread_copy_vgpr_to_lds =
            ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                               CShuffleDataType,
                                               decltype(c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2),
                                               decltype(c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2),
                                               ck::tensor_operation::element_wise::PassThrough,
                                               Sequence<CShuffleMXdlPerWavePerShuffle,
                                                        CShuffleNXdlPerWavePerShuffle,
                                                        I1,
                                                        I1,
                                                        M2,
                                                        I1,
                                                        M4,
                                                        I1>,
                                               Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                               7,
                                               1,
                                               InMemoryDataOperationEnum::Set,
                                               1,
                                               true>{
                c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                make_multi_index(0,
                                 0,
                                 m_thread_data_on_block_idx[I1],
                                 n_thread_data_on_block_idx[I1],
                                 m_thread_data_on_block_idx[I2],
                                 m_thread_data_on_block_idx[I3],
                                 m_thread_data_on_block_idx[I4],
                                 n_thread_data_on_block_idx[I2]),
                ck::tensor_operation::element_wise::PassThrough{}};

        // M0 = grid_size
        // M1 = MPerBlock
        // N0 = 1
        // N1 = NPerBlock
        const auto workspace_grid_desc_m0_m1_n0_n1 =
            MakeWorkspaceGridDesc_GridSize_MPerBlock_I1_NPerBlock(get_grid_size());
        auto p_workspace_grid = reinterpret_cast<AccDataType*>(p_workspace);
        auto w_grid_buf =
#if(defined(__gfx908__) || defined(__gfx90a__))
            make_dynamic_buffer<AddressSpaceEnum::Global, AmdBufferCoherenceEnum::GLC>(
#elif defined(__gfx94__)
            make_dynamic_buffer<AddressSpaceEnum::Global, AmdBufferCoherenceEnum::SYSTEM_NT0>(
#else // for host
            make_dynamic_buffer<AddressSpaceEnum::Global, AmdBufferCoherenceEnum::DefaultCoherence>(
#endif
                p_workspace_grid, workspace_grid_desc_m0_m1_n0_n1.GetElementSpaceSize());

        // shuffle: blockwise copy C from LDS to workspace
        auto c_shuffle_block_copy_lds_to_global = ThreadGroupTensorSliceTransfer_v6r1<
            ThisThreadBlock,                                 // ThreadGroup
            ck::tensor_operation::element_wise::PassThrough, // ElementwiseOperation,
            InMemoryDataOperationEnum::Set,                  // DstInMemOp,
            Sequence<1,
                     CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                     1,
                     CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>, // BlockSliceLengths,
            CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
            Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
            CShuffleDataType,     // typename SrcData,
            CShuffleDataType,     // typename DstData,
            decltype(c_shuffle_block_desc_mblock_mperblock_nblock_nperblock),
            decltype(workspace_grid_desc_m0_m1_n0_n1),
            Sequence<0, 1, 2, 3>,                             // typename DimAccessOrder,
            3,                                                // index_t VectorDim,
            CDEShuffleBlockTransferScalarPerVector_NPerBlock, // index_t ScalarPerVector,
            true,  // bool ThreadTransferSrcResetCoordinateAfterRun,
            false> // bool ThreadTransferDstResetCoordinateAfterRun>
            {c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
             make_multi_index(0, 0, 0, 0),
             workspace_grid_desc_m0_m1_n0_n1,
             make_multi_index(static_cast<index_t>(blockIdx.x), 0, 0, 0),
             ck::tensor_operation::element_wise::PassThrough{}};

        // space filling curve for threadwise C in VGPR
        constexpr auto sfc_c_vgpr =
            SpaceFillingCurve<Sequence<MXdlPerWave, NXdlPerWave, 1, 1, M2, 1, M4, 1>,
                              Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                              Sequence<CShuffleMXdlPerWavePerShuffle,
                                       CShuffleNXdlPerWavePerShuffle,
                                       1,
                                       1,
                                       M2,
                                       1,
                                       M4,
                                       1>>{};

        // space filling curve for shuffled blockwise W in global mem
        constexpr auto sfc_w_global =
            SpaceFillingCurve<Sequence<1, MPerBlock, 1, NPerBlock>,
                              Sequence<0, 2, 1, 3>,
                              Sequence<1,
                                       CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                                       1,
                                       CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>>{};

        constexpr index_t num_access = sfc_c_vgpr.GetNumOfAccess();
        static_assert(num_access == sfc_w_global.GetNumOfAccess(), "wrong!");

        static_for<0, num_access, 1>{}([&](auto access_id) {
            // make sure it's safe to write to LDS
            block_sync_lds();

            // each thread write its data from VGPR to LDS
            c_thread_copy_vgpr_to_lds.Run(c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                                          sfc_c_vgpr.GetIndexTupleOfNumber(access_id),
                                          c_thread_buf,
                                          c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                                          c_shuffle_block_buf);

            // make sure it's safe to read from LDS
            block_sync_lds();

            // each block copy its data from LDS to global
            c_shuffle_block_copy_lds_to_global.Run(
                c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                c_shuffle_block_buf,
                workspace_grid_desc_m0_m1_n0_n1,
                w_grid_buf);

            if constexpr(access_id < num_access - 1)
            {
                constexpr auto w_global_step = sfc_w_global.GetForwardStep(access_id);
                // move on C
                c_shuffle_block_copy_lds_to_global.MoveDstSliceWindow(
                    workspace_grid_desc_m0_m1_n0_n1, w_global_step);
            }
        });
    }

    __device__ static constexpr auto GetClusterLengthReduction_M0_N0N1()
    {
        return Sequence<CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::At(I1),
                        I1.value,
                        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::At(I3)>{};
    }

    // M0 - 1
    // M1 - M elements per thread
    // N0 - 1
    // N1 - N repeats per thread
    // N2 - Vector load/store size
    __device__ static constexpr auto MakeReductionThreadDesc_M0M1_N0N1N2()
    {
        constexpr auto cluster_lengths = GetClusterLengthReduction_M0_N0N1();

        constexpr auto N2 = Number<CDEShuffleBlockTransferScalarPerVector_NPerBlock>{};
        constexpr auto N1 = Number<NPerBlock>{} / (Number<cluster_lengths.At(I2)>{} * N2);
        constexpr auto M1 = math::integer_divide_ceil(Number<MPerBlock>{}, cluster_lengths.At(I0));

        static_assert(
            Number<M1>{} * cluster_lengths.At(I0) == Number<MPerBlock>{},
            "Invalid ReductionThreadDesc M0M1_N0N1N2! M1 * cluster_length[0] have to be grater "
            "or equal to MPerBlock.");
        static_assert(Number<N1>{} * Number<N2>{} * cluster_lengths.At(I2) == Number<NPerBlock>{},
                      "Invalid ReductionThreadDesc M0M1_N0N1N2! N1 * N2 * cluster_length[2] have "
                      "to be grater or equal to NPerBlock.");

        return make_naive_tensor_descriptor_packed(make_tuple(I1, Number<M1>{}, I1, N1, N2));
    }

    template <typename AccumulationBuffer>
    __device__ static void AccumulatePartials(void* __restrict__ p_workspace,
                                              AccumulationBuffer& acc_buff,
                                              uint32_t reduce_count)
    {
        constexpr auto cluster_length_reduce = GetClusterLengthReduction_M0_N0N1();
        constexpr auto reduce_cluster_desc   = make_cluster_descriptor(cluster_length_reduce);

        static_assert(ThisThreadBlock::GetNumOfThread() >= reduce_cluster_desc.GetElementSize(),
                      "Error! ThisThreadBlock::GetNumOfThread() too small");

        if(ThisThreadBlock::GetThreadId() >= reduce_cluster_desc.GetElementSize())
        {
            return;
        }

        const auto reduce_thread_cluster_idx =
            reduce_cluster_desc.CalculateBottomIndex(make_multi_index(get_thread_local_1d_id()));
        const auto thread_m_cluster_id  = reduce_thread_cluster_idx[I0];
        const auto thread_n0_cluster_id = reduce_thread_cluster_idx[I1]; // Should be I0
        const auto thread_n1_cluster_id = reduce_thread_cluster_idx[I2];

        constexpr auto workspace_thread_desc_m0m1_n0n1n2 = MakeReductionThreadDesc_M0M1_N0N1N2();
        const auto workspace_grid_desc_m0m1_n0n1 =
            MakeWorkspaceGridDesc_GridSize_MPerBlock_I1_NPerBlock(get_grid_size());

        const auto workspace_grid_desc_m0m1_n0n1n2 = transform_tensor_descriptor(
            workspace_grid_desc_m0m1_n0n1,
            make_tuple(
                make_pass_through_transform(workspace_grid_desc_m0m1_n0n1.GetLength(I0)),
                make_pass_through_transform(workspace_grid_desc_m0m1_n0n1.GetLength(I1)),
                make_pass_through_transform(workspace_grid_desc_m0m1_n0n1.GetLength(I2)),
                make_unmerge_transform(make_tuple(workspace_thread_desc_m0m1_n0n1n2.GetLength(I3),
                                                  workspace_thread_desc_m0m1_n0n1n2.GetLength(I4) *
                                                      cluster_length_reduce.At(I2)))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3, 4>{}));

        StaticBuffer<AddressSpaceEnum::Vgpr,
                     CShuffleDataType,
                     workspace_thread_desc_m0m1_n0n1n2.GetElementSpaceSize(),
                     true>
            partial_acc_buf{};

        auto p_workspace_grid = reinterpret_cast<CShuffleDataType*>(p_workspace);
        auto w_grid_buf =
#if(defined(__gfx908__) || defined(__gfx90a__))
            make_dynamic_buffer<AddressSpaceEnum::Global, AmdBufferCoherenceEnum::GLC>(
#elif defined(__gfx94__)
            make_dynamic_buffer<AddressSpaceEnum::Global, AmdBufferCoherenceEnum::SYSTEM_NT0>(
#else // for host
            make_dynamic_buffer<AddressSpaceEnum::Global, AmdBufferCoherenceEnum::DefaultCoherence>(
#endif
                p_workspace_grid, workspace_grid_desc_m0m1_n0n1n2.GetElementSpaceSize());

        auto acc_load = ThreadwiseTensorSliceTransfer_v2<
            CShuffleDataType,                                         // SrcData,
            CShuffleDataType,                                         // DstData,
            decltype(workspace_grid_desc_m0m1_n0n1n2),                // SrcDesc,
            decltype(workspace_thread_desc_m0m1_n0n1n2),              // DstDesc,
            decltype(workspace_thread_desc_m0m1_n0n1n2.GetLengths()), // SliceLengths,
            Sequence<0, 1, 2, 3, 4>,                                  // DimAccessOrder,
            4,                                                        // SrcVectorDim,
            CDEShuffleBlockTransferScalarPerVector_NPerBlock,         // SrcScalarPerVector,
            1,                                                        // SrcScalarStrideInVector,
            false                                                     // SrcResetCoordinateAfterRun,
            >{workspace_grid_desc_m0m1_n0n1n2,
              // We do not need to read this workgroup partial results since they're
              // already in c_thread_buff
              //   make_multi_index((static_cast<index_t>(blockIdx.x) + 1),
              // We want to have a thread raked access pattern
              make_multi_index(
                  static_cast<index_t>(blockIdx.x),
                  thread_m_cluster_id * workspace_thread_desc_m0m1_n0n1n2.GetLength(I1),
                  I0,
                  thread_n0_cluster_id,
                  thread_n1_cluster_id * workspace_thread_desc_m0m1_n0n1n2.GetLength(I4))};

        using Accumulation = ck::detail::
            AccumulateWithNanCheck<false /*PropagateNan*/, reduce::Add, CShuffleDataType>;
        constexpr auto partial_acc_load_step = make_multi_index(I1, I0, I0, I0, I0);

        // TODO: We do not need to read this workgroup partial results since they're
        // already in c_thread_buff
        for(uint32_t i_t = 0; i_t < reduce_count; ++i_t)
        {
            partial_acc_buf.Clear();
            acc_load.Run(workspace_grid_desc_m0m1_n0n1n2,
                         w_grid_buf,
                         workspace_thread_desc_m0m1_n0n1n2,
                         make_tuple(I0, I0, I0, I0, I0),
                         partial_acc_buf);

            static_for<0, workspace_thread_desc_m0m1_n0n1n2.GetElementSpaceSize(), 1>{}(
                [&](auto i_vec) {
                    Accumulation::Calculate(acc_buff(i_vec), partial_acc_buf[i_vec]);
                });

            acc_load.MoveSrcSliceWindow(workspace_grid_desc_m0m1_n0n1n2, partial_acc_load_step);
        }
    }

    template <typename Block2ETileMap, typename AccumulationBuffer>
    __device__ static void RunWrite(DsGridPointer p_ds_grid,
                                    EDataType* __restrict__ p_e_grid,
                                    AccumulationBuffer& acc_buff,
                                    const index_t M,
                                    const index_t N,
                                    const std::array<index_t, NumDTensor> StrideDs,
                                    const index_t StrideE,
                                    const CDEElementwiseOperation& cde_element_op,
                                    const Block2ETileMap& block_2_etile_map)
    {
        using DsGridDesc_M_N = remove_cvref_t<decltype(MakeDsGridDescriptor_M_N({}, {}, {}))>;
        using DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock =
            remove_cvref_t<decltype(MakeDsGridDescriptor_M0M1_N0N1N2(DsGridDesc_M_N{}))>;

        constexpr index_t ScalarPerVector = CDEShuffleBlockTransferScalarPerVector_NPerBlock;

        DsGridDesc_M_N ds_grid_desc_m_n;
        DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock ds_grid_desc_m0m1_n0n1n2;

        const auto e_grid_desc_m_n         = MakeEGridDescriptor_M_N<ELayout>(M, N, StrideE);
        const auto e_grid_desc_m0m1_n0n1n2 = MakeEGridDescriptor_M0M1_N0N1N2(e_grid_desc_m_n);
        auto e_grid_buf                    = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_e_grid, e_grid_desc_m0m1_n0n1n2.GetElementSpaceSize());

        static_for<0, NumDTensor, 1>{}([&](auto j) {
            using DLayout       = remove_cvref_t<tuple_element_t<j.value, DsLayout>>;
            ds_grid_desc_m_n(j) = MakeEGridDescriptor_M_N<DLayout>(M, N, StrideDs[j]);
        });

        static_for<0, NumDTensor, 1>{}([&](auto j) {
            ds_grid_desc_m0m1_n0n1n2(j) = MakeEGridDescriptor_M0M1_N0N1N2(ds_grid_desc_m_n[j]);
        });

        // TODO: on MI300 we could use NonTemporal load, MI200 streaming mode?
        const auto ds_grid_buf = generate_tuple(
            [&](auto i) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_ds_grid[i], ds_grid_desc_m0m1_n0n1n2[i].GetElementSpaceSize());
            },
            Number<NumDTensor>{});

        auto ds_thread_buf = generate_tuple(
            [&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
                return StaticBuffer<AddressSpaceEnum::Vgpr, DDataType, ScalarPerVector, true>{};
            },
            Number<NumDTensor>{});

        constexpr auto d_vgpr_buf_desc = make_naive_tensor_descriptor_packed(
            make_tuple(I1, I1, I1, I1, Number<ScalarPerVector>{}));

        // divide block work by [M, N, K]
        const auto block_work_idx = block_2_etile_map.GetBottomIndex();

        constexpr auto cluster_length_reduce = GetClusterLengthReduction_M0_N0N1();
        constexpr auto reduce_cluster_desc   = make_cluster_descriptor(cluster_length_reduce);

        // TODO similar assertion
        // static_assert(
        //     is_same<SliceLengths, decltype(thread_slice_lengths * ThreadClusterLengths{})>{},
        //     "wrong! threads should be mapped to cover entire slicing window");

        static_assert(ThisThreadBlock::GetNumOfThread() >= reduce_cluster_desc.GetElementSize(),
                      "Error! ThisThreadBlock::GetNumOfThread() too small");

        if(ThisThreadBlock::GetThreadId() >= reduce_cluster_desc.GetElementSize())
        {
            return;
        }

        const auto reduce_thread_cluster_idx =
            reduce_cluster_desc.CalculateBottomIndex(make_multi_index(get_thread_local_1d_id()));
        const auto thread_m_cluster_id  = reduce_thread_cluster_idx[I0];
        const auto thread_n0_cluster_id = reduce_thread_cluster_idx[I1]; // Should be I0
        const auto thread_n1_cluster_id = reduce_thread_cluster_idx[I2];

        constexpr auto workspace_thread_desc_m0m1_n0n1n2 = MakeReductionThreadDesc_M0M1_N0N1N2();

        auto ds_grid_load = generate_tuple(
            [&](auto i) {
                using DDataType    = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
                using SliceLengths = Sequence<I1, I1, I1, I1, ScalarPerVector>;
                return ThreadwiseTensorSliceTransfer_v2<DDataType,
                                                        DDataType,
                                                        decltype(ds_grid_desc_m0m1_n0n1n2[i]),
                                                        decltype(d_vgpr_buf_desc),
                                                        SliceLengths,
                                                        Sequence<0, 1, 2, 3, 4>,
                                                        4,
                                                        ScalarPerVector,
                                                        1,
                                                        false>{
                    ds_grid_desc_m0m1_n0n1n2[i],
                    make_multi_index(
                        block_work_idx[I0],
                        thread_m_cluster_id * workspace_thread_desc_m0m1_n0n1n2.GetLength(I1),
                        block_work_idx[I1],
                        thread_n0_cluster_id,
                        thread_n1_cluster_id * workspace_thread_desc_m0m1_n0n1n2.GetLength(I4))};
            },
            Number<NumDTensor>{});

        // Each thread writes consecutive M rows and strided N columns
        auto e_grid_store =
            ThreadwiseTensorSliceTransfer_v1r3<EDataType,
                                               EDataType,
                                               decltype(workspace_thread_desc_m0m1_n0n1n2),
                                               decltype(e_grid_desc_m0m1_n0n1n2),
                                               ck::tensor_operation::element_wise::PassThrough,
                                               Sequence<I1, I1, I1, I1, ScalarPerVector>,
                                               Sequence<0, 1, 2, 3, 4>,
                                               4,
                                               ScalarPerVector,
                                               EGlobalMemoryDataOperation,
                                               1,
                                               false>{
                e_grid_desc_m0m1_n0n1n2,
                make_multi_index(
                    block_work_idx[I0],
                    thread_m_cluster_id * workspace_thread_desc_m0m1_n0n1n2.GetLength(I1),
                    block_work_idx[I1],
                    thread_n0_cluster_id,
                    thread_n1_cluster_id * workspace_thread_desc_m0m1_n0n1n2.GetLength(I4)),
                ck::tensor_operation::element_wise::PassThrough{}};

        constexpr auto MIter   = workspace_thread_desc_m0m1_n0n1n2.GetLength(I1);
        constexpr auto NIter   = workspace_thread_desc_m0m1_n0n1n2.GetLength(I3);
        constexpr auto n1_step = I1;

        constexpr auto d_grid_M1_fwd_step = make_multi_index(I0, I1, I0, I0, I0);
        constexpr auto d_grid_N1_fwd_step = make_multi_index(I0, I0, I0, n1_step, I0);
        constexpr auto d_grid_N1_bwd_step =
            make_multi_index(I0, I0, I0, -1 * n1_step * (NIter - 1), I0);

        constexpr auto thr_buf_N1_offset = Number<ScalarPerVector>{};
        constexpr auto thr_buf_M1_offset = NIter * thr_buf_N1_offset;

        static_for<0, MIter, 1>{}([&](auto m_idx) {
            static_for<0, NIter, 1>{}([&](auto n_idx) {
                // load multiple Ds:
                static_for<0, NumDTensor, 1>{}([&](auto d_idx) {
                    ds_grid_load(d_idx).Run(ds_grid_desc_m0m1_n0n1n2[d_idx],
                                            ds_grid_buf[d_idx],
                                            d_vgpr_buf_desc,
                                            make_tuple(I0, I0, I0, I0, I0),
                                            ds_thread_buf(d_idx));
                });

                constexpr auto acc_buf_offset =
                    m_idx * thr_buf_M1_offset + n_idx * thr_buf_N1_offset;

                // apply pointwise function
                static_for<0, ScalarPerVector, 1>{}([&](auto I) {
                    // get reference to src data
                    const auto src_data_ds_refs = generate_tie(
                        // return type should be lvalue
                        [&](auto iSrc) -> const auto& { return ds_thread_buf[iSrc][I]; },
                        Number<NumDTensor>{});

                    const auto src_data_refs = concat_tuple_of_reference(
                        tie(acc_buff[acc_buf_offset + I]), src_data_ds_refs);
                    // apply pointwise function
                    // pointwise function signature:
                    // element_op_(dst_data_refs[I0],
                    //             dst_data_refs[I1],
                    //             ...,
                    //             src_data_refs[I0],
                    //             src_data_refs[I1],
                    //             ...)
                    unpack2(cde_element_op, tie(acc_buff(acc_buf_offset + I)), src_data_refs);
                });

                e_grid_store.Run(workspace_thread_desc_m0m1_n0n1n2,
                                 make_tuple(I0, m_idx, I0, n_idx, I0),
                                 acc_buff,
                                 e_grid_desc_m0m1_n0n1n2,
                                 e_grid_buf);

                if constexpr(NIter != 1)
                {
                    if constexpr(n_idx != (NIter - 1))
                    {
                        static_for<0, NumDTensor, 1>{}([&](auto d_idx) {
                            ds_grid_load(d_idx).MoveSrcSliceWindow(ds_grid_desc_m0m1_n0n1n2(d_idx),
                                                                   d_grid_N1_fwd_step);
                        });
                        e_grid_store.MoveDstSliceWindow(e_grid_desc_m0m1_n0n1n2,
                                                        d_grid_N1_fwd_step);
                    }
                    else
                    {
                        static_for<0, NumDTensor, 1>{}([&](auto d_idx) {
                            ds_grid_load(d_idx).MoveSrcSliceWindow(ds_grid_desc_m0m1_n0n1n2(d_idx),
                                                                   d_grid_N1_bwd_step);
                        });
                        e_grid_store.MoveDstSliceWindow(e_grid_desc_m0m1_n0n1n2,
                                                        d_grid_N1_bwd_step);
                    }
                }
            }); // NIter
            {
                static_for<0, NumDTensor, 1>{}([&](auto d_idx) {
                    ds_grid_load(d_idx).MoveSrcSliceWindow(ds_grid_desc_m0m1_n0n1n2(d_idx),
                                                           d_grid_M1_fwd_step);
                });

                e_grid_store.MoveDstSliceWindow(e_grid_desc_m0m1_n0n1n2, d_grid_M1_fwd_step);
            }
        }); // MIter
    }
};

} // namespace ck
