// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/philox_rand.hpp"
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
#include "ck/tensor_operation/gpu/block/blockwise_softmax.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_dropout.hpp"

namespace ck {

template <typename InputDataType,
          typename OutputDataType,
          typename ZDataType,
          typename GemmDataType,
          typename FloatGemmAcc,
          typename FloatCShuffle,
          typename FloatLSE,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename SElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CElementwiseOperation,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          typename QGridDesc_K0_M_K1,
          typename KGridDesc_K0_N_K1,
          typename KGridDesc_N_K,
          typename ZGridDesc_M_N,
          typename VGridDesc_O0_N_O1,
          typename YGridDesc_M_O,
          typename LSEGridDesc_M,
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
          index_t Gemm2NXdlPerWave,
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
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          LoopScheduler LoopSched,
          bool PadN,
          bool MaskOutUpperTriangle,
          bool Deterministic,
          PipelineVersion PipelineVer = PipelineVersion::v1>
struct GridwiseBatchedMultiheadAttentionBackward_Xdl_CShuffle_V1
{
    static_assert(LoopSched == LoopScheduler::Default,
                  "Non-default loop scheduler is currently not supported");

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};
    static constexpr auto I8 = Number<8>{};
    static constexpr auto I9 = Number<9>{};

    static constexpr auto WaveSize = 64;

    // K1 should be Number<...>
    // Gemm0
    static constexpr auto AK0 = Number<KPerBlock / AK1Value>{};
    static constexpr auto BK0 = Number<KPerBlock / BK1Value>{};
    static constexpr auto AK1 = Number<AK1Value>{};
    static constexpr auto BK1 = Number<BK1Value>{};

    static constexpr auto Gemm0MWaves = MPerBlock / (MPerXdl * MXdlPerWave);
    static constexpr auto Gemm0NWaves = NPerBlock / (NPerXdl * NXdlPerWave);

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    // C desc for source in blockwise copy
    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5(const ZGridDesc_M_N& z_grid_desc_m_n)
    {
        const auto M = z_grid_desc_m_n.GetLength(I0);
        const auto N = z_grid_desc_m_n.GetLength(I1);

        constexpr auto mfma = MfmaSelector<GemmDataType, MPerXdl, NPerXdl>::selected_mfma;
        constexpr auto N3   = mfma.num_groups_per_blk;
        constexpr auto N4   = mfma.num_input_blks;
        constexpr auto N5   = mfma.group_size;
        return transform_tensor_descriptor(
            z_grid_desc_m_n,
            make_tuple(make_unmerge_transform(
                           make_tuple(M / MPerBlock, MXdlPerWave, Gemm0MWaves, MPerXdl)),
                       make_unmerge_transform(
                           make_tuple(N / NPerBlock, NXdlPerWave, Gemm0NWaves, N3, N4, N5))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2, 4, 6>{}, Sequence<1, 3, 5, 7, 8, 9>{}));
    }

    __device__ static auto GetGemm0WaveIdx()
    {
        const index_t thread_id = get_thread_local_1d_id();

        constexpr auto threadid_to_wave_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(Gemm0MWaves, Gemm0NWaves, WaveSize))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));

        return threadid_to_wave_idx_adaptor.CalculateBottomIndex(make_multi_index(thread_id));
    }

    __device__ static auto GetGemm0WaveMNIdx(const index_t thread_id)
    {
        constexpr auto wave_threadid_to_mn_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(WaveSize / MPerXdl, MPerXdl))),
            make_tuple(Sequence<0, 1>{}),
            make_tuple(Sequence<0>{}));

        return wave_threadid_to_mn_idx_adaptor.CalculateBottomIndex(make_multi_index(thread_id));
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

    template <typename AccThreadDesc_M0_N0_M1_N1_M2_M3_M4_N2>
    __host__ __device__ static constexpr auto GetA1SrcThreadDescriptor_AK0PerBlock_MPerBlock_AK1(
        const AccThreadDesc_M0_N0_M1_N1_M2_M3_M4_N2& acc_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2)
    {
        // acc_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2 to a_src_thread_desc_k0_m_k1
        // m0_m1_m2_m3 -> k0
        // n0_n1_n2 -> m
        // m4 -> k1
        // NOTE: had to use merge_v3 or will spit out compilation errors
        const auto m0 = acc_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I0);
        const auto n0 = acc_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I1);
        const auto m1 = acc_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I2);
        const auto n1 = acc_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I3);
        const auto m2 = acc_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I4);
        const auto m3 = acc_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I5);
        const auto m4 = acc_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I6);
        const auto n2 = acc_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I7);

        return transform_tensor_descriptor(
            acc_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2,
            make_tuple(make_merge_transform_v3_division_mod(make_tuple(m0, m1, m2, m3)),
                       make_merge_transform_v3_division_mod(make_tuple(n0, n1, n2)),
                       make_pass_through_transform(m4)),
            make_tuple(Sequence<0, 2, 4, 5>{}, Sequence<1, 3, 7>{}, Sequence<6>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
    }

    __host__ __device__ static constexpr auto
    GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock()
    {
        constexpr index_t MWave = Gemm0NWaves;
        constexpr index_t NWave = Gemm0MWaves;

        constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            make_naive_tensor_descriptor_packed(
                make_tuple(I1,
                           Number<CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl>{},
                           I1,
                           Number<CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>{}));

        return c_shuffle_block_desc_mblock_mperblock_nblock_nperblock;
    }

    template <typename Gemm2Param>
    __host__ __device__ static constexpr auto GetA2BlockDescriptor_K0_M_K1()
    {
        return make_naive_tensor_descriptor(
            make_tuple(Number<Gemm2Param::A_K0>{},
                       Number<Gemm2Param::Gemm2_M>{},
                       Number<Gemm2Param::A_K1>{}),
            make_tuple(Number<Gemm2Param::Gemm2_M + Gemm2Param::A_LdsPad>{} *
                           Number<Gemm2Param::A_K1>{},
                       Number<Gemm2Param::A_K1>{},
                       I1));
    }

    __host__ __device__ static constexpr bool
    CheckValidity(const QGridDesc_K0_M_K1& q_grid_desc_k0_m_k1,
                  const KGridDesc_K0_N_K1& k_grid_desc_k0_n_k1,
                  const VGridDesc_O0_N_O1& v_grid_desc_o0_n_o1,
                  const YGridDesc_M_O& y_grid_desc_m_o)
    {
        static_assert((MPerBlock % (MPerXdl * MXdlPerWave) == 0) &&
                          (NPerBlock % (NXdlPerWave * NPerXdl)) == 0,
                      "Invalid tuning param!");

        const auto M      = q_grid_desc_k0_m_k1.GetLength(I1);
        const auto N      = k_grid_desc_k0_n_k1.GetLength(I1);
        const auto K      = q_grid_desc_k0_m_k1.GetLength(I0) * q_grid_desc_k0_m_k1.GetLength(I2);
        const auto Gemm1N = v_grid_desc_o0_n_o1.GetLength(I0) * v_grid_desc_o0_n_o1.GetLength(I2);

        // This assumption reduces implemention complexity by categorizing 6 separate GEMMs into 3
        // types of GEMM operations, therefore some code body can be reused accordingly
        // P_MNK / dP_MNO Gemm (Gemm0 rcr)
        // Y_MON / dQ_MKN Gemm (Gemm1 rrr)
        // dV_NOM / dK_NKM Gemm (Gemm2 crr)
        if(Gemm1N != K)
        {
            std::cerr << "SizeK must be equal to SizeO (equal attention head size)" << '\n';
            return false;
        }

        if(!(M == y_grid_desc_m_o.GetLength(I0) && Gemm1N == y_grid_desc_m_o.GetLength(I1)))
        {
            return false;
        }

        if(!(M % MPerBlock == 0 && N % NPerBlock == 0 && K % KPerBlock == 0 &&
             Gemm1N % Gemm1NPerBlock == 0))
        {
            return false;
        }

        // check gemm1 gridwise gemm pipeline
        if(!(NPerBlock % Gemm1KPerBlock == 0))
        {
            return false;
        }

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        return true;
    }

    __host__ __device__ static constexpr auto
    MakeYGridDescriptor_MBlock_MPerBlock_OBlock_OPerBlock(const YGridDesc_M_O& y_grid_desc_m_o)
    {
        const auto M = y_grid_desc_m_o.GetLength(I0);
        const auto O = y_grid_desc_m_o.GetLength(I1);

        const auto MBlock = M / MPerBlock;
        const auto OBlock = O / Gemm1NPerBlock;

        const auto y_grid_desc_mblock_mperblock_oblock_operblock = transform_tensor_descriptor(
            y_grid_desc_m_o,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{})),
                       make_unmerge_transform(make_tuple(OBlock, Number<Gemm1NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

        return y_grid_desc_mblock_mperblock_oblock_operblock;
    }

    template <typename SrcBlockwiseGemm>
    __host__ __device__ static constexpr auto
    MakeLSEGridDescriptor_MB_M0_M1_M2_M3_M4(const LSEGridDesc_M& lse_grid_desc_m)
    {
        const index_t M      = lse_grid_desc_m.GetLength(I0);
        const index_t MBlock = M / MPerBlock;
        constexpr auto SrcBlockDesc_M0_N0_M1_N1_M2_M3_M4_N2 =
            SrcBlockwiseGemm::GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();
        // M0 MXdlPerWave, M1 MWave, M2 num_groups_per_blk, M3 num_input_blks, M4 group_size
        const auto M0 = SrcBlockDesc_M0_N0_M1_N1_M2_M3_M4_N2.GetLength(I0);
        const auto M1 = SrcBlockDesc_M0_N0_M1_N1_M2_M3_M4_N2.GetLength(I2);
        const auto M2 = SrcBlockDesc_M0_N0_M1_N1_M2_M3_M4_N2.GetLength(I4);
        const auto M3 = SrcBlockDesc_M0_N0_M1_N1_M2_M3_M4_N2.GetLength(I5);
        const auto M4 = SrcBlockDesc_M0_N0_M1_N1_M2_M3_M4_N2.GetLength(I6);

        const auto lse_grid_desc_mb_m0_m1_m2_m3_m4 = transform_tensor_descriptor(
            lse_grid_desc_m,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, M0, M1, M2, M3, M4))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2, 3, 4, 5>{}));

        return lse_grid_desc_mb_m0_m1_m2_m3_m4;
    }

    __device__ static auto MakeKGradGridDesc_NBlock_NPerBlock_OBlock_OPerBlock(
        const KGridDesc_K0_N_K1& k_grid_desc_k0_n_k1)
    {
        const auto O0 = k_grid_desc_k0_n_k1.GetLength(I0);
        const auto N  = k_grid_desc_k0_n_k1.GetLength(I1);
        const auto O1 = k_grid_desc_k0_n_k1.GetLength(I2);
        const auto O  = O0 * O1;

        const auto NBlock = N / NPerBlock;
        const auto OBlock = O / Gemm1NPerBlock;

        const auto k_grid_desc_n_o = transform_tensor_descriptor(
            k_grid_desc_k0_n_k1,
            make_tuple(make_pass_through_transform(N),
                       make_merge_transform_v3_division_mod(make_tuple(O0, O1))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return transform_tensor_descriptor(
            k_grid_desc_n_o,
            make_tuple(make_unmerge_transform(make_tuple(NBlock, Number<NPerBlock>{})),
                       make_unmerge_transform(make_tuple(OBlock, Number<Gemm1NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));
    }

    __device__ static auto MakeVGradGridDesc_NBlock_NPerBlock_OBlock_OPerBlock(
        const VGridDesc_O0_N_O1& v_grid_desc_o0_n_o1)
    {
        const auto O0 = v_grid_desc_o0_n_o1.GetLength(I0);
        const auto N  = v_grid_desc_o0_n_o1.GetLength(I1);
        const auto O1 = v_grid_desc_o0_n_o1.GetLength(I2);
        const auto O  = O0 * O1;

        const auto NBlock = N / NPerBlock;
        const auto OBlock = O / Gemm1NPerBlock;

        const auto v_grid_desc_n_o = transform_tensor_descriptor(
            v_grid_desc_o0_n_o1,
            make_tuple(make_pass_through_transform(N),
                       make_merge_transform_v3_division_mod(make_tuple(O0, O1))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return transform_tensor_descriptor(
            v_grid_desc_n_o,
            make_tuple(make_unmerge_transform(make_tuple(NBlock, Number<NPerBlock>{})),
                       make_unmerge_transform(make_tuple(OBlock, Number<Gemm1NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));
    }

    __device__ static auto MakeQGradGridDesc_M_K(const QGridDesc_K0_M_K1& q_grid_desc_k0_m_k1)
    {
        const auto K_K0 = q_grid_desc_k0_m_k1.GetLength(I0);
        const auto M    = q_grid_desc_k0_m_k1.GetLength(I1);
        const auto K_K1 = q_grid_desc_k0_m_k1.GetLength(I2);

        return transform_tensor_descriptor(
            q_grid_desc_k0_m_k1,
            make_tuple(make_pass_through_transform(M),
                       make_merge_transform_v3_division_mod(make_tuple(K_K0, K_K1))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));
    }

    // return block_id to C matrix tile idx (m0, n0) mapping
    __host__ __device__ static constexpr auto
    MakeDefaultBlock2CTileMap(const KGridDesc_N_K& k_grid_desc_n_k)
    {
        return BlockToCTileMap_M00_N0_M01Adapt<NPerBlock, KPerBlock, KGridDesc_N_K>(
            k_grid_desc_n_k);
    }

    using YGridDescriptor_MBlock_MPerBlock_OBlock_OPerBlock = remove_cvref_t<decltype(
        MakeYGridDescriptor_MBlock_MPerBlock_OBlock_OPerBlock(YGridDesc_M_O{}))>;

    using DefaultBlock2CTileMap =
        remove_cvref_t<decltype(MakeDefaultBlock2CTileMap(KGridDesc_N_K{}))>;

    using ZGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5 = remove_cvref_t<decltype(
        MakeCGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5(ZGridDesc_M_N{}))>;

    // Q / K / V / dY
    struct GemmBlockwiseCopy
    {
        // Q matrix in LDS memory, dst of blockwise copy
        static constexpr auto q_block_desc_k0_m_k1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();

        // K matrix in LDS memory, dst of blockwise copy
        static constexpr auto k_block_desc_k0_n_k1 =
            GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        // V matrix in LDS memory, dst of blockwise copy
        static constexpr auto v_block_desc_k0_n_k1 =
            GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        // dY matrix in LDS memory, dst of blockwise copy
        static constexpr auto ygrad_block_desc_k0_m_k1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();

        template <typename GridDesc_K0_M_K1>
        using QBlockwiseCopy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                tensor_operation::element_wise::PassThrough,
                                                tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<AK0, MPerBlock, AK1>,
                                                ABlockTransferThreadClusterLengths_AK0_M_AK1,
                                                ABlockTransferThreadClusterArrangeOrder,
                                                InputDataType,
                                                GemmDataType,
                                                GridDesc_K0_M_K1,
                                                decltype(q_block_desc_k0_m_k1),
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
                                                NumGemmKPrefetchStage>;

        template <typename GridDesc_K0_N_K1>
        using KBlockwiseCopy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                tensor_operation::element_wise::PassThrough,
                                                tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<BK0, NPerBlock, BK1>,
                                                BBlockTransferThreadClusterLengths_BK0_N_BK1,
                                                BBlockTransferThreadClusterArrangeOrder,
                                                InputDataType,
                                                GemmDataType,
                                                GridDesc_K0_N_K1,
                                                decltype(k_block_desc_k0_n_k1),
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
                                                NumGemmKPrefetchStage>;

        template <typename GridDesc_K0_N_K1>
        using VBlockwiseCopy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                tensor_operation::element_wise::PassThrough,
                                                tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<BK0, NPerBlock, BK1>,
                                                BBlockTransferThreadClusterLengths_BK0_N_BK1,
                                                BBlockTransferThreadClusterArrangeOrder,
                                                InputDataType,
                                                GemmDataType,
                                                GridDesc_K0_N_K1,
                                                decltype(v_block_desc_k0_n_k1),
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
                                                NumGemmKPrefetchStage>;

        template <typename GridDesc_K0_M_K1>
        using YGradBlockwiseCopy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                tensor_operation::element_wise::PassThrough,
                                                tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<AK0, MPerBlock, AK1>,
                                                ABlockTransferThreadClusterLengths_AK0_M_AK1,
                                                ABlockTransferThreadClusterArrangeOrder,
                                                InputDataType,
                                                GemmDataType,
                                                GridDesc_K0_M_K1,
                                                decltype(ygrad_block_desc_k0_m_k1),
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
                                                NumGemmKPrefetchStage>;

        static constexpr auto gemm_tile_q_block_slice_copy_step = make_multi_index(0, MPerBlock, 0);
        static constexpr auto gemm_tile_ygrad_block_slice_copy_step =
            make_multi_index(0, MPerBlock, 0);
    };

    // S / dP Gemm (type 1 rcc)
    struct Gemm0
    {
        // A matrix in LDS memory, dst of blockwise copy
        static constexpr auto a_block_desc_ak0_m_ak1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();

        // B matrix in LDS memory, dst of blockwise copy
        static constexpr auto b_block_desc_bk0_n_bk1 =
            GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

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

        static constexpr index_t KPack =
            math::max(math::lcm(AK1, BK1),
                      MfmaSelector<GemmDataType, MPerXdl, NPerXdl>::selected_mfma.k_per_blk);

        // Blockwise gemm with transposed XDL output
        using BlockwiseGemm = BlockwiseGemmXdlops_v2<
            BlockSize,
            GemmDataType,
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
            KPack>;
    };

    // dV / dK Gemm (type 2 rrr)
    template <typename ASrcThreadDesc_M0_N0_M1_N1_M2_M3_M4_N2,
              typename ASrcBlockDesc_M0_N0_M1_N1_M2_M3_M4_N2>
    struct Gemm1
    {
        private:
        static constexpr auto m0 = ASrcThreadDesc_M0_N0_M1_N1_M2_M3_M4_N2{}.GetLength(I0);
        static constexpr auto n0 = ASrcThreadDesc_M0_N0_M1_N1_M2_M3_M4_N2{}.GetLength(I1);
        static constexpr auto m1 = ASrcThreadDesc_M0_N0_M1_N1_M2_M3_M4_N2{}.GetLength(I2);
        static constexpr auto n1 = ASrcThreadDesc_M0_N0_M1_N1_M2_M3_M4_N2{}.GetLength(I3);
        static constexpr auto m2 = ASrcThreadDesc_M0_N0_M1_N1_M2_M3_M4_N2{}.GetLength(I4);
        static constexpr auto m3 = ASrcThreadDesc_M0_N0_M1_N1_M2_M3_M4_N2{}.GetLength(I5);
        static constexpr auto m4 = ASrcThreadDesc_M0_N0_M1_N1_M2_M3_M4_N2{}.GetLength(I6);
        static constexpr auto n2 = ASrcThreadDesc_M0_N0_M1_N1_M2_M3_M4_N2{}.GetLength(I7);

        // M2 num_groups_per_blk, M3 num_input_blks, M4 group_size
        static constexpr auto M3 = ASrcBlockDesc_M0_N0_M1_N1_M2_M3_M4_N2{}.GetLength(I5);

        public:
        static constexpr auto AThreadSliceLength_K0 = Number<Gemm1KPerBlock / m4 / M3>{};
        static constexpr auto AThreadSliceLength_M  = Number<n0 * n1 * n2>{};
        static constexpr auto AThreadSliceLength_K1 = Number<m4>{};

        // A source matrix layout in AccVGPR
        static constexpr auto a_src_thread_desc_k0_m_k1 =
            GetA1SrcThreadDescriptor_AK0PerBlock_MPerBlock_AK1(
                ASrcThreadDesc_M0_N0_M1_N1_M2_M3_M4_N2{});

        // A matrix in VGPR memory, dst of AccVGPR-to-VGPR copy
        static constexpr auto a_thread_desc_k0_m_k1 = make_naive_tensor_descriptor_packed(
            make_tuple(AThreadSliceLength_K0, AThreadSliceLength_M, AThreadSliceLength_K1));

        // B matrix in LDS memory, dst of blockwise copy
        static constexpr auto b_block_desc_bn0_k_bn1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();

        template <typename ABlockDesc_AK0_M_AK1>
        __host__ __device__ static constexpr auto
        MakeGemm1AMmaTileDescriptor_M0_M1_M2_K(const ABlockDesc_AK0_M_AK1&)
        {
            return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<NXdlPerWave, 1, 1>(
                ABlockDesc_AK0_M_AK1{});
        }

        template <typename BBlockDesc_BK0_N_BK1>
        __host__ __device__ static constexpr auto
        MakeGemm1BMmaTileDescriptor_N0_N1_N2_K(const BBlockDesc_BK0_N_BK1&)
        {
            return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<Gemm1NXdlPerWave, 1, 1>(
                BBlockDesc_BK0_N_BK1{});
        }

        static constexpr auto ASrcScalarPerVector = m4;

        using AThreadSliceLengths_K0_M_K1 = decltype(a_thread_desc_k0_m_k1.GetLengths());

        template <typename ElementwiseOp = tensor_operation::element_wise::PassThrough>
        using ABlockwiseCopy =
            ThreadwiseTensorSliceTransfer_StaticToStatic<FloatGemmAcc,
                                                         GemmDataType,
                                                         decltype(a_src_thread_desc_k0_m_k1),
                                                         decltype(a_thread_desc_k0_m_k1),
                                                         ElementwiseOp,
                                                         AThreadSliceLengths_K0_M_K1,
                                                         Sequence<1, 0, 2>,
                                                         2,
                                                         ASrcScalarPerVector>;

        // for a_block_slice_copy_step to be able to address static buffers, it MUST be a
        // tuple-based container as well as containing ONLY integral constants
        static constexpr auto a_block_slice_copy_step = make_tuple(AThreadSliceLength_K0, I0, I0);

        // selected_mfma.group_size or B1K1 <= Gemm1KPack <= selected_mfma.group_size
        // selected_mfma.k_per_blk <= Gemm1KPack
        //
        // Following similar rationale behind Gemm0KPack, let Gemm1KPack be the lowest common
        // multiples of A1K1 (predetermined by selected_mfma.group_size) and B1K1. But in this case
        // Gemm1KPack can't be higher than A1K1 itself because A1 matrix is distributed in VGPRs
        // with 'group_size' amount of contiguous elements. Having Gemm1KPack greater than A1K1 will
        // cause mismatch in summation index for example c[0:7] = a1[[0:3, 8:11]] * b1[0:7].
        // therefore we may just as well assign Gemm1KPack = group_size
        static constexpr index_t GemmKPack =
            MfmaSelector<GemmDataType, MPerXdl, NPerXdl>::selected_mfma.group_size;

        static constexpr index_t GemmMWave   = Gemm0NWaves;                  // 4        // 4
        static constexpr index_t GemmNWave   = Gemm0MWaves;                  // 1        // 1
        static constexpr index_t GemmMRepeat = NXdlPerWave;                  // 1        // 1
        static constexpr index_t GemmNRepeat = Gemm1NXdlPerWave;             // 1        // 2
        static constexpr index_t GemmKLoop   = MPerBlock / Gemm1KPerBlock;   // 128/32=4 // 64/32=2
        static constexpr index_t B_K3        = GemmKPack;                    // 4        // 4
        static constexpr index_t B_K2        = M3;                           // 2        // 2
        static constexpr index_t B_K1        = Gemm1KPerBlock / B_K2 / B_K3; // 4        // 4
        static constexpr index_t B_K0        = GemmKLoop;                    // 4        // 2

        __host__ __device__ static constexpr auto MakeBBlockDesc_N0_N1_N2_K0_K1_K2_K3()
        {
            const auto N0_ = b_block_desc_bn0_k_bn1.GetLength(I0);
            const auto K_  = b_block_desc_bn0_k_bn1.GetLength(I1);
            const auto N1_ = b_block_desc_bn0_k_bn1.GetLength(I2);

            constexpr auto b_block_desc_n_k = transform_tensor_descriptor( //(32, 128) //(64, 64)
                b_block_desc_bn0_k_bn1,
                make_tuple(
                    make_merge_transform_v3_division_mod(make_tuple(N0_, N1_)), //(4, 8)   //(8, 8)
                    make_pass_through_transform(K_)),                           // 128     // 64
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return transform_tensor_descriptor(
                b_block_desc_n_k,
                make_tuple(make_unmerge_transform(make_tuple(
                               GemmNRepeat, GemmNWave, NPerXdl)), //(1, 1, 32)   //(2, 1, 32)
                           make_unmerge_transform(
                               make_tuple(B_K0, B_K1, B_K2, B_K3))), //(4, 4, 2, 4) //(2, 4, 2, 4)
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 2>{}, Sequence<3, 4, 5, 6>{}));
        }

        static constexpr auto b_block_desc_n0_n1_n2_k0_k1_k2_k3 =
            MakeBBlockDesc_N0_N1_N2_K0_K1_K2_K3();

        using BThreadSlice_N0_N1_N2_K0_K1_K2_K3 = Sequence<GemmNRepeat, 1, 1, 1, B_K1, 1, B_K3>;

        static constexpr auto b_thread_desc_n0_n1_n2_k0_k1_k2_k3 =
            make_naive_tensor_descriptor_packed(
                make_tuple(Number<GemmNRepeat>{}, I1, I1, I1, Number<B_K1>{}, I1, Number<B_K3>{}));

        __host__ __device__ static constexpr auto MakeBThreadDesc_K0_N_K1()
        {
            constexpr auto b_thread_desc_n_k = transform_tensor_descriptor(
                b_thread_desc_n0_n1_n2_k0_k1_k2_k3,
                make_tuple(
                    make_merge_transform_v3_division_mod(make_tuple(Number<GemmNRepeat>{}, I1, I1)),
                    make_merge_transform_v3_division_mod(
                        make_tuple(I1, Number<B_K1>{}, I1, Number<B_K3>{}))),
                make_tuple(Sequence<0, 1, 2>{}, Sequence<3, 4, 5, 6>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return transform_tensor_descriptor(
                b_thread_desc_n_k,
                make_tuple(make_pass_through_transform(Number<GemmNRepeat>{}),
                           make_unmerge_transform(make_tuple(Number<B_K1>{}, Number<B_K3>{}))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<1>{}, Sequence<0, 2>{}));
        }

        static constexpr auto b_thread_desc_k0_n_k1 = MakeBThreadDesc_K0_N_K1();

        using BBlockwiseCopy =
            ThreadwiseTensorSliceTransfer_v2<GemmDataType,
                                             GemmDataType,
                                             decltype(b_block_desc_n0_n1_n2_k0_k1_k2_k3),
                                             decltype(b_thread_desc_n0_n1_n2_k0_k1_k2_k3),
                                             BThreadSlice_N0_N1_N2_K0_K1_K2_K3,
                                             Sequence<0, 1, 2, 3, 4, 5, 6>,
                                             6,
                                             1,
                                             1,
                                             true>;

        static constexpr auto b_block_slice_copy_step = make_multi_index(0, 0, 0, 1, 0, 0, 0);
        static constexpr auto b_block_reset_copy_step = make_multi_index(0, 0, 0, -B_K0, 0, 0, 0);

        using BlockwiseGemm = BlockwiseGemmXdlops_v2<
            BlockSize,
            GemmDataType,
            FloatGemmAcc,
            decltype(a_thread_desc_k0_m_k1),
            decltype(b_thread_desc_k0_n_k1),
            decltype(MakeGemm1AMmaTileDescriptor_M0_M1_M2_K(a_thread_desc_k0_m_k1)),
            decltype(MakeGemm1BMmaTileDescriptor_N0_N1_N2_K(b_thread_desc_k0_n_k1)),
            NPerBlock,
            Gemm1NPerBlock,
            Gemm1KPerBlock,
            MPerXdl,
            NPerXdl,
            NXdlPerWave,
            Gemm1NXdlPerWave,
            GemmKPack,
            true,      // TransposeC
            GemmKPack, // AMmaKStride
            GemmKPack>;
    };

    // dQ Gemm (type 3 crr)
    // Describes tuning parameter for C2_m_n = A2_m_k * B2_k_n
    template <index_t Sum_K_ = NPerXdl * 2>
    struct Gemm2Params_
    {
        static constexpr index_t Gemm2_M = MPerBlock;      // 128  // 64
        static constexpr index_t Gemm2_K = NPerBlock;      // 128  // 128
        static constexpr index_t Gemm2_N = Gemm1NPerBlock; // 32   // 64
        static constexpr index_t Sum_K   = Sum_K_;

        static constexpr index_t A_K1     = 8; // dS will be row-major
        static constexpr index_t A_K0     = Sum_K / A_K1;
        static constexpr index_t A_LdsPad = 0; // how many multiples of K1 per M * K1 elements

        static_assert(Sum_K % NPerXdl == 0, "");

        static constexpr index_t GemmNWave   = Gemm2_N / Gemm2NXdlPerWave / NPerXdl;    // 1 // 2
        static constexpr index_t GemmMWave   = BlockSize / get_warp_size() / GemmNWave; // 4 // 2
        static constexpr index_t GemmNRepeat = Gemm2NXdlPerWave;                        // 1 // 1
        static constexpr index_t GemmMRepeat = Gemm2_M / GemmMWave / MPerXdl;           // 1 // 1
        static constexpr index_t GemmKLoop   = Gemm2_K / Sum_K;                         // 2 // 2
        static constexpr index_t GemmKPack =
            math::max(A_K1, MfmaSelector<GemmDataType, MPerXdl, NPerXdl>::selected_mfma.k_per_blk);
        static constexpr index_t B_K3 = GemmKPack; // 8
        static constexpr index_t B_K2 =
            XdlopsGemm<GemmDataType, MPerXdl, NPerXdl, GemmKPack, false>{}.K0PerXdlops; // 2
        static constexpr index_t B_K1 = Sum_K / B_K2 / B_K3;                            // 4
        static constexpr index_t B_K0 = GemmKLoop;                                      // 2

        __host__ __device__ static constexpr auto GetABlockSliceLengths_M0_K0_M1_K1_M2_K2()
        {
            // perform manual unmerge: m -> m_repeat, m_waves, m_per_xdl
            constexpr index_t k  = Gemm2Params::Sum_K - 1;
            constexpr index_t k2 = k % NPerXdl;
            constexpr index_t k1 = k / NPerXdl % Gemm0NWaves;
            constexpr index_t k0 = k / NPerXdl / Gemm0NWaves % NXdlPerWave;

            // perform manual unmerge: n -> n_repeat, n_waves, n_per_xdl
            constexpr index_t m  = Gemm2Params::Gemm2_M - 1;
            constexpr index_t m2 = m % MPerXdl;
            constexpr index_t m1 = m / MPerXdl % Gemm0MWaves;
            constexpr index_t m0 = m / MPerXdl / Gemm0MWaves % MXdlPerWave;

            // assume 256 decomposed into 2 x 4 x 32
            // 1d idx ( 32 - 1) -> 3d idx 0, 0, 31 -> 3d dim 1 x 1 x 32
            // 1d idx (256 - 1) -> 3d idx 1, 3, 31 -> 3d dim 2 x 4 x 32
            return Sequence<m0, k0, m1, k1, m2, k2>{} + Sequence<1, 1, 1, 1, 1, 1>{};
        }

        __host__ __device__ static constexpr auto GetABlockSliceLengths_M0_K0_M1_K1()
        {
            return generate_sequence_v2(
                [](auto I) { return GetABlockSliceLengths_M0_K0_M1_K1_M2_K2().At(I); },
                Number<4>{});
        }

        using ABlockSliceLengths_M0_K0_M1_K1 =
            decltype(GetABlockSliceLengths_M0_K0_M1_K1()); //(2, 1, 1, 2) //(4, 1, 1, 2)
    };
    using Gemm2Params = Gemm2Params_<>; // tune later

    // dQ Gemm (type 3 crr)
    template <typename Gemm2Params, typename ASrcBlockwiseGemm>
    struct Gemm2
    {
        private:
        static constexpr auto a_src_block_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
            ASrcBlockwiseGemm::GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();
        static constexpr auto M0 = a_src_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I0); // repeat
        static constexpr auto N0 = a_src_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I1);
        static constexpr auto M1 = a_src_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I2); // wave
        static constexpr auto N1 = a_src_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I3);
        static constexpr auto M2 = a_src_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I4); // xdl
        static constexpr auto M3 = a_src_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I5);
        static constexpr auto M4 = a_src_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I6);
        static constexpr auto N2 = a_src_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I7);

        public:
        // A source matrix layout in VGPR, src of VGPR-to-LDS copy
        static constexpr auto a_src_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
            ASrcBlockwiseGemm::GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

        // A matrix in LDS memory, dst of blockwise copy
        static constexpr auto a_block_desc_k0_m_k1 = GetA2BlockDescriptor_K0_M_K1<Gemm2Params>();

        // // B matrix in LDS memory, dst of blockwise copy
        static constexpr auto b_block_desc_n0_k_n1 =
            GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        template <typename ABlockDesc_K0_M_K1>
        __host__ __device__ static constexpr auto
        MakeGemm2AMmaTileDescriptor_M0_M1_M2_K(const ABlockDesc_K0_M_K1&)
        {
            return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<Gemm2Params::GemmMRepeat,
                                                           Gemm2Params::GemmMWave,
                                                           MPerXdl>(ABlockDesc_K0_M_K1{});
        }

        template <typename BBlockDesc_K0_N_K1>
        __host__ __device__ static constexpr auto
        MakeGemm2BMmaTileDescriptor_N0_N1_N2_K(const BBlockDesc_K0_N_K1&)
        {
            return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<Gemm2Params::GemmNRepeat, 1, 1>(
                BBlockDesc_K0_N_K1{});
        }

        __host__ __device__ static constexpr auto MakeABlockDesc_M0_K0_M1_K1_M2_M3_M4_K2()
        {
            const auto K0_ = a_block_desc_k0_m_k1.GetLength(I0);
            const auto M_  = a_block_desc_k0_m_k1.GetLength(I1);
            const auto K1_ = a_block_desc_k0_m_k1.GetLength(I2);

            const auto a_block_desc_m_k = transform_tensor_descriptor(
                a_block_desc_k0_m_k1,
                make_tuple(make_merge_transform_v3_division_mod(make_tuple(K0_, K1_)),
                           make_pass_through_transform(M_)),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
                make_tuple(Sequence<1>{}, Sequence<0>{}));

            // HACK: for unmerge transform, the length of highest dim is irrelevant so we put dummy
            // variable I1 there
            return transform_tensor_descriptor(
                a_block_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(I1, M1, M2, M3, M4)),
                           make_unmerge_transform(make_tuple(I1, N1, N2))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2, 4, 5, 6>{}, Sequence<1, 3, 7>{}));
        }

        // Note: we will perform sub-workgroup VGPR-to-LDS copy to save LDS space, therefore the
        // destination coordinate can overlap between wavefronts in a workgroup as seen in the mod
        // operation before returning the values
        __host__ __device__ static auto MakeAThreadOriginOnBlock_M0_K0_M1_K1_M2_M3_M4_K2()
        {
            const auto a_thread_origin_on_block_idx =
                ASrcBlockwiseGemm::CalculateCThreadOriginDataIndex8D(I0, I0, I0, I0);

            constexpr auto a_block_slice_lengths_m0_k0_m1_k1 =
                typename Gemm2Params::ABlockSliceLengths_M0_K0_M1_K1{}; // mrepeat, nrepeat,
                                                                        // mwaves, nwaves,

            return make_tuple(
                a_thread_origin_on_block_idx[I0],                                         // mrepeat
                a_thread_origin_on_block_idx[I1],                                         // nrepeat
                a_thread_origin_on_block_idx[I2] % a_block_slice_lengths_m0_k0_m1_k1[I2], // mwave
                a_thread_origin_on_block_idx[I3] % a_block_slice_lengths_m0_k0_m1_k1[I3], // nwave
                a_thread_origin_on_block_idx[I4],                                         // xdlops
                a_thread_origin_on_block_idx[I5],
                a_thread_origin_on_block_idx[I6],
                a_thread_origin_on_block_idx[I7]);
        }

        static constexpr auto a_block_desc_m0_k0_m1_k1_m2_m3_m4_k2 =
            MakeABlockDesc_M0_K0_M1_K1_M2_M3_M4_K2();

        using ASrcBlockSliceWindowIterator =
            SpaceFillingCurve<Sequence<M0, N0, M1, N1>,
                              Sequence<0, 1, 2, 3>,
                              typename Gemm2Params::ABlockSliceLengths_M0_K0_M1_K1,
                              false>;

        template <typename ElementwiseOp = tensor_operation::element_wise::PassThrough>
        using ABlockwiseCopy = ThreadwiseTensorSliceTransfer_v1r3<
            FloatGemmAcc,
            GemmDataType,
            decltype(a_src_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2),
            decltype(a_block_desc_m0_k0_m1_k1_m2_m3_m4_k2),
            ElementwiseOp,
            Sequence<Gemm2Params::ABlockSliceLengths_M0_K0_M1_K1::At(I0), // ThreadSliceLengths
                     Gemm2Params::ABlockSliceLengths_M0_K0_M1_K1::At(I1),
                     I1,
                     I1,
                     M2,
                     I1,
                     M4,
                     I1>,
            Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
            7, // DstVectorDim
            1, // DstScalarPerVector
            InMemoryDataOperationEnum::Set,
            1, // DstScalarStrideInVector
            true>;

        __host__ __device__ static constexpr auto MakeBBlockDesc_N0_N1_N2_K0_K1_K2_K3()
        {
            const auto N0_ = b_block_desc_n0_k_n1.GetLength(I0);
            const auto K_  = b_block_desc_n0_k_n1.GetLength(I1);
            const auto N1_ = b_block_desc_n0_k_n1.GetLength(I2);

            constexpr auto b_block_desc_n_k = transform_tensor_descriptor( //(32, 128)  //(64, 128)
                b_block_desc_n0_k_n1,
                make_tuple(make_merge_transform_v3_division_mod(
                               make_tuple(N0_, N1_)),        //(4, 8)     //(8, 8)
                           make_pass_through_transform(K_)), // 128       // 128
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return transform_tensor_descriptor(
                b_block_desc_n_k,
                make_tuple(
                    make_unmerge_transform(make_tuple(Gemm2Params::GemmNRepeat,
                                                      Gemm2Params::GemmNWave,
                                                      NPerXdl)), //(1, 1, 32)     //(1, 2, 32)
                    make_unmerge_transform(
                        make_tuple(Gemm2Params::B_K0,
                                   Gemm2Params::B_K1,
                                   Gemm2Params::B_K2,
                                   Gemm2Params::B_K3))), //(2, 4, 2, 8)   //(2, 4, 2, 8)
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 2>{}, Sequence<3, 4, 5, 6>{}));
        }

        static constexpr auto b_block_desc_n0_n1_n2_k0_k1_k2_k3 =
            MakeBBlockDesc_N0_N1_N2_K0_K1_K2_K3();

        using BThreadSlice_N0_N1_N2_K0_K1_K2_K3 =
            Sequence<Gemm2Params::GemmNRepeat, 1, 1, 1, Gemm2Params::B_K1, 1, Gemm2Params::B_K3>;

        static constexpr auto b_thread_desc_n0_n1_n2_k0_k1_k2_k3 =
            make_naive_tensor_descriptor_packed(make_tuple(Number<Gemm2Params::GemmNRepeat>{},
                                                           I1,
                                                           I1,
                                                           I1,
                                                           Number<Gemm2Params::B_K1>{},
                                                           I1,
                                                           Number<Gemm2Params::B_K3>{}));

        __host__ __device__ static constexpr auto MakeBThreadDesc_K0_N_K1()
        {
            constexpr auto b_thread_desc_n_k = transform_tensor_descriptor(
                b_thread_desc_n0_n1_n2_k0_k1_k2_k3,
                make_tuple(make_merge_transform_v3_division_mod(
                               make_tuple(Number<Gemm2Params::GemmNRepeat>{}, I1, I1)),
                           make_merge_transform_v3_division_mod(make_tuple(
                               I1, Number<Gemm2Params::B_K1>{}, I1, Number<Gemm2Params::B_K3>{}))),
                make_tuple(Sequence<0, 1, 2>{}, Sequence<3, 4, 5, 6>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return transform_tensor_descriptor(
                b_thread_desc_n_k,
                make_tuple(make_pass_through_transform(Number<Gemm2Params::GemmNRepeat>{}),
                           make_unmerge_transform(make_tuple(Number<Gemm2Params::B_K1>{},
                                                             Number<Gemm2Params::B_K3>{}))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<1>{}, Sequence<0, 2>{}));
        }

        static constexpr auto b_thread_desc_k0_n_k1 = MakeBThreadDesc_K0_N_K1();

        using BBlockwiseCopy =
            ThreadwiseTensorSliceTransfer_v2<GemmDataType,
                                             GemmDataType,
                                             decltype(b_block_desc_n0_n1_n2_k0_k1_k2_k3),
                                             decltype(b_thread_desc_n0_n1_n2_k0_k1_k2_k3),
                                             BThreadSlice_N0_N1_N2_K0_K1_K2_K3,
                                             Sequence<0, 1, 2, 3, 4, 5, 6>,
                                             6,
                                             1,
                                             1,
                                             true>;

        static constexpr auto b_block_slice_copy_step = make_multi_index(0, 0, 0, 1, 0, 0, 0);
        static constexpr auto b_block_reset_copy_step =
            make_multi_index(0, 0, 0, -Gemm2Params::B_K0, 0, 0, 0);

        using BlockwiseGemm = BlockwiseGemmXdlops_v2<
            BlockSize,
            GemmDataType,
            FloatGemmAcc,
            decltype(a_block_desc_k0_m_k1),
            decltype(b_thread_desc_k0_n_k1),
            decltype(MakeGemm2AMmaTileDescriptor_M0_M1_M2_K(a_block_desc_k0_m_k1)),
            decltype(MakeGemm2BMmaTileDescriptor_N0_N1_N2_K(b_thread_desc_k0_n_k1)),
            MPerBlock,
            Gemm1NPerBlock,
            Gemm2Params::Sum_K,
            MPerXdl,
            NPerXdl,
            Gemm2Params::GemmMRepeat,
            Gemm2Params::GemmNRepeat,
            Gemm2Params::GemmKPack,
            true, // TransposeC
            Gemm2Params::GemmKPack *
                XdlopsGemm<GemmDataType, MPerXdl, NPerXdl, Gemm2Params::GemmKPack, false>{}
                    .K0PerXdlops,
            Gemm2Params::GemmKPack>;

        static constexpr auto c_block_slice_copy_step =
            make_multi_index(Gemm2Params::GemmMRepeat, 0, 0, 0, 0, 0, 0, 0);

        template <typename CGradDesc_M_N>
        __host__ __device__ static auto
        MakeCGridDesc_M0_N0_M1_N1_M2_N2_N3_N4(const CGradDesc_M_N& c_grid_desc_m_n)
        {
            // HACK: for unmerge transform, the length of highest dim is irrelevant so we put dummy
            // variable I1 there
            const auto c_grid_desc_m0_n0_m1_n1_m2_n2 = transform_tensor_descriptor(
                c_grid_desc_m_n,
                make_tuple(make_unmerge_transform(make_tuple(I1, Gemm2Params::GemmMWave, MPerXdl)),
                           make_unmerge_transform(make_tuple(I1, Gemm2Params::GemmNWave, NPerXdl))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}));

            const auto c_grid_desc_m0_n0_m1_n1_m2_n2_n3_n4 =
                BlockwiseGemm{}.xdlops_gemm.MakeCDescriptor_M0_N0_M1_N1_M2_N2_N3_N4(
                    c_grid_desc_m0_n0_m1_n1_m2_n2);

            return c_grid_desc_m0_n0_m1_n1_m2_n2_n3_n4;
        }

        static constexpr auto c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4 =
            BlockwiseGemm::GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();

        __host__ __device__ static auto GetCThreadOriginOnBlock_M0_N0_M1_N1_M2_N2_N3_N4()
        {
            return to_multi_index(BlockwiseGemm::CalculateCThreadOriginDataIndex8D(I0, I0, I0, I0));
        }

        template <typename CGridDesc_M0_N0_M1_N1_M2_N2_N3_N4,
                  typename ElementwiseOp = tensor_operation::element_wise::PassThrough>
        using CBlockwiseCopy = ThreadwiseTensorSliceTransfer_v1r3<
            FloatGemmAcc,
            OutputDataType,
            decltype(c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4),
            CGridDesc_M0_N0_M1_N1_M2_N2_N3_N4,
            ElementwiseOp,                                                // CElementwiseOperation
            decltype(c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetLengths()), // SliceLengths
            Sequence<0, 1, 2, 3, 4, 5, 6, 7>,                             // AccessOrder
            7,                                                            // VectorDim
            2,                                                            // ScalarPerVector
            InMemoryDataOperationEnum::AtomicAdd, // GlobalMemoryDataOperation
            1,                                    // DstScalarStrideInVector
            true>;
    };

    template <index_t BlockSize_, index_t BlockSliceLength_M_, index_t BlockSliceLength_O_>
    struct YDotYGrad_M_O_
    {
        static constexpr index_t SrcScalarPerVector = 16 / sizeof(InputDataType);
        static constexpr auto ThreadClusterLength_O =
            Number<BlockSliceLength_O_ / SrcScalarPerVector>{};
        static constexpr auto ThreadClusterLength_M = Number<BlockSize_ / ThreadClusterLength_O>{};
        static constexpr auto ThreadSliceLength_O   = Number<SrcScalarPerVector>{};
        static constexpr auto ThreadSliceLength_M =
            Number<BlockSliceLength_M_ * ThreadClusterLength_O / BlockSize_>{};

        // dY matrix in LDS memory, dst of blockwise copy
        static constexpr auto ygrad_block_desc_o0_m_o1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();

        __host__ __device__ static constexpr auto MakeYGradBlockDesc_M_O()
        {
            const auto O0_ = ygrad_block_desc_o0_m_o1.GetLength(I0);
            const auto M_  = ygrad_block_desc_o0_m_o1.GetLength(I1);
            const auto O1_ = ygrad_block_desc_o0_m_o1.GetLength(I2);

            static_assert(O0_ * O1_ == BlockSliceLength_O_, "");
            static_assert(M_ == BlockSliceLength_M_, "");

            return transform_tensor_descriptor( //(128, 64)
                ygrad_block_desc_o0_m_o1,
                make_tuple(make_merge_transform_v3_division_mod(make_tuple(O0_, O1_)), //(8, 8)
                           make_pass_through_transform(M_)),                           // 128
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
                make_tuple(Sequence<1>{}, Sequence<0>{}));
        }

        static constexpr auto ygrad_block_desc_m_o = MakeYGradBlockDesc_M_O();

        static_assert(ThreadClusterLength_O * ThreadSliceLength_O == BlockSliceLength_O_, "");
        static_assert(ThreadClusterLength_M * ThreadSliceLength_M == BlockSliceLength_M_, "");

        using SrcBufType = StaticBuffer<AddressSpaceEnum::Vgpr,
                                        FloatGemmAcc,
                                        ThreadSliceLength_M * ThreadSliceLength_O,
                                        true>;

        using DstBufType =
            StaticBuffer<AddressSpaceEnum::Vgpr, FloatGemmAcc, ThreadSliceLength_M, true>;
    };
    using YDotYGrad_M_O = YDotYGrad_M_O_<BlockSize, MPerBlock, Gemm1NPerBlock>;

    struct SharedMemTrait
    {
        // // LDS allocation for A and B: be careful of alignment
        static constexpr auto q_block_desc_k0_m_k1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();
        static constexpr auto k_block_desc_k0_n_k1 =
            GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();
        static constexpr auto v_block_desc_k0_n_k1 =
            GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();
        static constexpr auto ygrad_block_desc_k0_m_k1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();
        static constexpr auto p_slash_sgrad_block_desc_k0_m_k1 =
            GetA2BlockDescriptor_K0_M_K1<Gemm2Params>();

        static constexpr auto max_lds_align = Number<16 / sizeof(GemmDataType)>{};

        static constexpr auto q_block_space_size_aligned =
            math::integer_least_multiple(q_block_desc_k0_m_k1.GetElementSpaceSize(), max_lds_align);
        static constexpr auto k_block_space_size_aligned =
            math::integer_least_multiple(k_block_desc_k0_n_k1.GetElementSpaceSize(), max_lds_align);
        static constexpr auto v_block_space_size_aligned =
            math::integer_least_multiple(v_block_desc_k0_n_k1.GetElementSpaceSize(), max_lds_align);
        static constexpr auto ygrad_block_space_size_aligned = math::integer_least_multiple(
            ygrad_block_desc_k0_m_k1.GetElementSpaceSize(), max_lds_align);
        static constexpr auto p_slash_sgrad_block_space_size_aligned = math::integer_least_multiple(
            p_slash_sgrad_block_desc_k0_m_k1.GetElementSpaceSize(), max_lds_align);

        static constexpr auto k_block_space_offset = 0;
        static constexpr auto v_block_space_offset = k_block_space_size_aligned.value;
        static constexpr auto ygrad_block_space_offset =
            k_block_space_size_aligned.value + v_block_space_size_aligned.value;
        static constexpr auto q_block_space_offset = k_block_space_size_aligned.value +
                                                     v_block_space_size_aligned.value +
                                                     ygrad_block_space_size_aligned.value;
        static constexpr auto p_slash_sgrad_block_space_offset =
            k_block_space_size_aligned.value + v_block_space_size_aligned.value +
            ygrad_block_space_size_aligned.value + q_block_space_size_aligned.value;

        // LDS allocation for reduction
        static constexpr index_t reduction_space_size_aligned =
            math::integer_least_multiple(BlockSize, max_lds_align);

        static constexpr auto reduction_space_offset =
            (k_block_space_size_aligned.value + v_block_space_size_aligned.value +
             ygrad_block_space_size_aligned.value + q_block_space_size_aligned.value) *
            sizeof(GemmDataType) / sizeof(FloatGemmAcc);

        // LDS allocation for C shuffle in LDS
        static constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock();
        static constexpr auto c_block_space_size =
            c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize();
    };

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        const index_t p_slash_sgrad_bytes_end =
            (SharedMemTrait::p_slash_sgrad_block_space_offset +
             SharedMemTrait::p_slash_sgrad_block_space_size_aligned) *
            sizeof(GemmDataType);
        const index_t softmax_bytes_end = (SharedMemTrait::reduction_space_offset +
                                           SharedMemTrait::reduction_space_size_aligned) *
                                          sizeof(FloatGemmAcc);
        const index_t c_block_bytes_end =
            SharedMemTrait::c_block_space_size * sizeof(FloatCShuffle);

        return math::max(p_slash_sgrad_bytes_end, softmax_bytes_end, c_block_bytes_end);
    }

    template <bool HasMainKBlockLoop,
              typename Block2CTileMap,
              typename C0MatrixMask,
              typename YGradGridDesc_O0_M_O1>
    __device__ static void Run(const InputDataType* __restrict__ p_q_grid,
                               const InputDataType* __restrict__ p_k_grid,
                               ZDataType* __restrict__ p_z_grid,
                               const InputDataType* __restrict__ p_v_grid,
                               const InputDataType* __restrict__ p_y_grid,
                               const FloatLSE* __restrict__ p_lse_grid,
                               const InputDataType* __restrict__ p_ygrad_grid,
                               OutputDataType* __restrict__ p_qgrad_grid,
                               OutputDataType* __restrict__ p_kgrad_grid,
                               OutputDataType* __restrict__ p_vgrad_grid,
                               void* __restrict__ p_shared,
                               const AElementwiseOperation& a_element_op,
                               const BElementwiseOperation& b_element_op,
                               const SElementwiseOperation& s_element_op,
                               const B1ElementwiseOperation& b1_element_op,
                               const CElementwiseOperation& c_element_op,
                               const QGridDesc_K0_M_K1& q_grid_desc_k0_m_k1,
                               const KGridDesc_K0_N_K1& k_grid_desc_k0_n_k1,
                               const ZGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5&
                                   z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5,
                               const VGridDesc_O0_N_O1& v_grid_desc_o0_n_o1,
                               const YGridDescriptor_MBlock_MPerBlock_OBlock_OPerBlock&
                                   y_grid_desc_mblock_mperblock_oblock_operblock,
                               const LSEGridDesc_M& lse_grid_desc_m,
                               const YGradGridDesc_O0_M_O1& ygrad_grid_desc_o0_m_o1,
                               const Block2CTileMap& block_2_ctile_map,
                               const C0MatrixMask& c0_matrix_mask,
                               const float p_drop,
                               ck::philox& ph,
                               const index_t block_idx_n)
    {
        const FloatGemmAcc p_dropout  = type_convert<FloatGemmAcc>(1.0f - p_drop);
        const FloatGemmAcc rp_dropout = type_convert<FloatGemmAcc>(1.0f / p_dropout);
        const ushort p_dropout_in_16bits =
            __builtin_amdgcn_readfirstlane(std::floor(p_dropout * 65535.0));
        const tensor_operation::element_wise::Scale scale_rp_dropout(s_element_op.Value() *
                                                                     rp_dropout);

        const auto q_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_q_grid, q_grid_desc_k0_m_k1.GetElementSpaceSize());
        const auto k_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_k_grid, k_grid_desc_k0_n_k1.GetElementSpaceSize());
        const auto v_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_v_grid, v_grid_desc_o0_n_o1.GetElementSpaceSize());
        const auto y_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_y_grid, y_grid_desc_mblock_mperblock_oblock_operblock.GetElementSpaceSize());
        const auto lse_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_lse_grid, lse_grid_desc_m.GetElementSpaceSize());
        const auto ygrad_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_ygrad_grid, ygrad_grid_desc_o0_m_o1.GetElementSpaceSize());
        auto vgrad_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_vgrad_grid, v_grid_desc_o0_n_o1.GetElementSpaceSize());
        auto qgrad_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_qgrad_grid, q_grid_desc_k0_m_k1.GetElementSpaceSize());
        auto kgrad_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_kgrad_grid, k_grid_desc_k0_n_k1.GetElementSpaceSize());

        // divide block work by [N, K]
        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        const index_t block_work_idx_n = Deterministic ? block_idx_n : block_work_idx[I0];

        // HACK: this force n_block_data_idx_on_grid into SGPR
        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx_n * NPerBlock);

        // 6 GEMM operations are categorized into 3 buckets. SizeK == SizeO == head_dim
        // S_MNK / dP_MNO Gemm (Gemm0 rcr)
        // Y_MON / dQ_MKN Gemm (Gemm1 rrr)
        // dV_NOM / dK_NKM Gemm (Gemm2 crr)

        // LDS allocation for Q / K / V / dY
        auto q_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<GemmDataType*>(p_shared) + SharedMemTrait::q_block_space_offset,
            GemmBlockwiseCopy::q_block_desc_k0_m_k1.GetElementSpaceSize());

        auto k_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<GemmDataType*>(p_shared) + SharedMemTrait::k_block_space_offset,
            GemmBlockwiseCopy::k_block_desc_k0_n_k1.GetElementSpaceSize());

        auto v_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<GemmDataType*>(p_shared) + SharedMemTrait::v_block_space_offset,
            GemmBlockwiseCopy::v_block_desc_k0_n_k1.GetElementSpaceSize());

        auto ygrad_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<GemmDataType*>(p_shared) + SharedMemTrait::ygrad_block_space_offset,
            GemmBlockwiseCopy::ygrad_block_desc_k0_m_k1.GetElementSpaceSize());

        // Q matrix blockwise copy
        auto gemm_tile_q_blockwise_copy =
            typename GemmBlockwiseCopy::template QBlockwiseCopy<decltype(q_grid_desc_k0_m_k1)>(
                q_grid_desc_k0_m_k1,
                make_multi_index(0, 0, 0), // will loop over GemmM dimension
                a_element_op,
                GemmBlockwiseCopy::q_block_desc_k0_m_k1,
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});

        // K matrix blockwise copy
        auto gemm_tile_k_blockwise_copy =
            typename GemmBlockwiseCopy::template KBlockwiseCopy<decltype(k_grid_desc_k0_n_k1)>(
                k_grid_desc_k0_n_k1,
                make_multi_index(0, n_block_data_idx_on_grid, 0),
                b_element_op,
                GemmBlockwiseCopy::k_block_desc_k0_n_k1,
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});

        // V matrix blockwise copy
        auto gemm_tile_v_blockwise_copy =
            typename GemmBlockwiseCopy::template VBlockwiseCopy<decltype(v_grid_desc_o0_n_o1)>(
                v_grid_desc_o0_n_o1,
                make_multi_index(0, n_block_data_idx_on_grid, 0),
                b1_element_op,
                GemmBlockwiseCopy::v_block_desc_k0_n_k1,
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});

        // dY matrix blockwise copy
        auto gemm_tile_ygrad_blockwise_copy =
            typename GemmBlockwiseCopy::template YGradBlockwiseCopy<decltype(
                ygrad_grid_desc_o0_m_o1)>(
                ygrad_grid_desc_o0_m_o1,
                make_multi_index(0, 0, 0), // will loop over GemmM dimension
                a_element_op,
                GemmBlockwiseCopy::ygrad_block_desc_k0_m_k1,
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});

        //
        // set up S / dP Gemm (type 1 rcr)
        //

        // S: blockwise gemm
        auto s_blockwise_gemm = typename Gemm0::BlockwiseGemm{}; // TransposeC

        auto s_slash_p_thread_buf = s_blockwise_gemm.GetCThreadBuffer();

        // dP: blockwise gemm
        // we need separate blockwise gemm object because we need separate thread buffer
        auto pgrad_blockwise_gemm = typename Gemm0::BlockwiseGemm{};

        auto pgrad_thread_buf = pgrad_blockwise_gemm.GetCThreadBuffer();

        //
        // set up dV / dK Gemm (type 2 rrr)
        //
        using Gemm1 =
            Gemm1<decltype(s_blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2()),
                  decltype(s_blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2())>;

        // Gemm1: VGPR allocation for A and B
        auto gemm1_a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, GemmDataType>(
            Gemm1::a_thread_desc_k0_m_k1.GetElementSpaceSize());

        auto gemm1_b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, GemmDataType>(
            Gemm1::b_thread_desc_n0_n1_n2_k0_k1_k2_k3.GetElementSpaceSize());

        // dV: A matrix blockwise copy
        auto vgrad_gemm_tile_p_blockwise_copy =
            typename Gemm1::template ABlockwiseCopy<tensor_operation::element_wise::Relu>{
                tensor_operation::element_wise::Relu{}}; // relu(P-dropped)

        // dV: blockwise gemm
        auto vgrad_blockwise_gemm =
            typename Gemm1::BlockwiseGemm{make_tuple(0, 0, 0, 0), make_tuple(0, 0, 0, 0)};

        // dV: B matrix blockwise copy
        auto ygrad_thread_origin = vgrad_blockwise_gemm.CalculateBThreadOriginDataIndex();

        // dV: B matrix LDS-to-VGPR blockwise copy
        auto vgrad_gemm_tile_ygrad_blockwise_copy = typename Gemm1::BBlockwiseCopy{
            Gemm1::b_block_desc_n0_n1_n2_k0_k1_k2_k3,
            make_multi_index(0,                                          // nrepeat
                             ygrad_thread_origin[I1],                    // nwave
                             ygrad_thread_origin[I2],                    // nperxdl
                             0,                                          // k0
                             0,                                          // k1
                             ygrad_thread_origin[I3] / Gemm1::GemmKPack, // k2
                             0)};

        auto vgrad_thread_buf = vgrad_blockwise_gemm.GetCThreadBuffer();

        // dV: transform input and output tensor descriptors
        auto vgrad_grid_desc_nblock_nperblock_oblock_operblock =
            MakeVGradGridDesc_NBlock_NPerBlock_OBlock_OPerBlock(v_grid_desc_o0_n_o1);

        // dK: A matrix blockwise copy
        auto kgrad_gemm_tile_sgrad_blockwise_copy =
            typename Gemm1::template ABlockwiseCopy<tensor_operation::element_wise::PassThrough>{
                tensor_operation::element_wise::PassThrough{}};

        // dK: blockwise gemm
        auto kgrad_blockwise_gemm =
            typename Gemm1::BlockwiseGemm{make_tuple(0, 0, 0, 0), make_tuple(0, 0, 0, 0)};

        // dK: B matrix blockwise copy
        auto q_thread_origin = kgrad_blockwise_gemm.CalculateBThreadOriginDataIndex();

        // dK: B matrix LDS-to-VGPR blockwise copy
        auto kgrad_gemm_tile_q_blockwise_copy = typename Gemm1::BBlockwiseCopy{
            Gemm1::b_block_desc_n0_n1_n2_k0_k1_k2_k3,
            make_multi_index(0,                                      // nrepeat
                             q_thread_origin[I1],                    // nwave
                             q_thread_origin[I2],                    // nperxdl
                             0,                                      // k0
                             0,                                      // k1
                             q_thread_origin[I3] / Gemm1::GemmKPack, // k2
                             0)};

        auto kgrad_thread_buf = kgrad_blockwise_gemm.GetCThreadBuffer();

        // dK: transform input and output tensor descriptors
        auto kgrad_grid_desc_nblock_nperblock_oblock_operblock =
            MakeKGradGridDesc_NBlock_NPerBlock_OBlock_OPerBlock(k_grid_desc_k0_n_k1);

        //
        // set up dQ Gemm (type 3 crr)
        //
        using Gemm2 = Gemm2<Gemm2Params, decltype(pgrad_blockwise_gemm)>;

        // Gemm2: LDS allocation for A and B: be careful of alignment
        auto gemm2_a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<GemmDataType*>(p_shared) + SharedMemTrait::p_slash_sgrad_block_space_offset,
            Gemm2::a_block_desc_k0_m_k1.GetElementSpaceSize());

        auto gemm2_b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, GemmDataType>(
            Gemm2::b_thread_desc_n0_n1_n2_k0_k1_k2_k3.GetElementSpaceSize());

        // dQ: A matrix VGPR-to-LDS blockwise copy
        auto qgrad_gemm_tile_sgrad_thread_copy_vgpr_to_lds =
            typename Gemm2::template ABlockwiseCopy<tensor_operation::element_wise::PassThrough>{
                Gemm2::a_block_desc_m0_k0_m1_k1_m2_m3_m4_k2,
                Gemm2::MakeAThreadOriginOnBlock_M0_K0_M1_K1_M2_M3_M4_K2(),
                tensor_operation::element_wise::PassThrough{}};

        // dQ: blockwise gemm
        auto qgrad_blockwise_gemm = typename Gemm2::BlockwiseGemm{};
        qgrad_blockwise_gemm.SetBBlockStartWindow(make_tuple(0, 0, 0, 0));

        auto k_thread_origin = qgrad_blockwise_gemm.CalculateBThreadOriginDataIndex();

        // dQ: B matrix LDS-to-VGPR blockwise copy
        auto qgrad_gemm_tile_k_blockwise_copy = typename Gemm2::BBlockwiseCopy{
            Gemm2::b_block_desc_n0_n1_n2_k0_k1_k2_k3,
            make_multi_index(0,                                            // nrepeat
                             k_thread_origin[I1],                          // nwave
                             k_thread_origin[I2],                          // nperxdl
                             0,                                            // k0
                             0,                                            // k1
                             k_thread_origin[I3] / Gemm2Params::GemmKPack, // k2
                             0)};                                          // k3

        auto qgrad_thread_buf = qgrad_blockwise_gemm.GetCThreadBuffer();

        // dQ: transform output tensor descriptors
        const auto qgrad_grid_desc_m_k = MakeQGradGridDesc_M_K(q_grid_desc_k0_m_k1);
        const auto qgrad_grid_desc_m0_o0_m1_o1_m2_o2_o3_o4 =
            Gemm2::MakeCGridDesc_M0_N0_M1_N1_M2_N2_N3_N4(qgrad_grid_desc_m_k);

        // dQ: C VGPR-to-global copy
        const auto qgrad_thread_origin_on_grid_m0_o0_m1_o1_m2_o2_o3_o4 =
            Gemm2::GetCThreadOriginOnBlock_M0_N0_M1_N1_M2_N2_N3_N4();

        auto qgrad_thread_copy_vgpr_to_global = typename Gemm2::template CBlockwiseCopy<
            decltype(qgrad_grid_desc_m0_o0_m1_o1_m2_o2_o3_o4),
            decltype(scale_rp_dropout)>(qgrad_grid_desc_m0_o0_m1_o1_m2_o2_o3_o4,
                                        qgrad_thread_origin_on_grid_m0_o0_m1_o1_m2_o2_o3_o4,
                                        scale_rp_dropout);

        //
        // Blockwise softmax
        //
        // get acc0 8D thread cluster
        constexpr auto thread_cluster_m0_n0_m1_n1_m2_m3_m4_n2 =
            s_blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2().GetLengths() /
            s_blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2().GetLengths();
        constexpr auto tm0 = thread_cluster_m0_n0_m1_n1_m2_m3_m4_n2.At(I0);
        constexpr auto tn0 = thread_cluster_m0_n0_m1_n1_m2_m3_m4_n2.At(I1);
        constexpr auto tm1 = thread_cluster_m0_n0_m1_n1_m2_m3_m4_n2.At(I2);
        constexpr auto tn1 = thread_cluster_m0_n0_m1_n1_m2_m3_m4_n2.At(I3);
        constexpr auto tm2 = thread_cluster_m0_n0_m1_n1_m2_m3_m4_n2.At(I4);
        constexpr auto tm3 = thread_cluster_m0_n0_m1_n1_m2_m3_m4_n2.At(I5);
        constexpr auto tm4 = thread_cluster_m0_n0_m1_n1_m2_m3_m4_n2.At(I6);
        constexpr auto tn2 = thread_cluster_m0_n0_m1_n1_m2_m3_m4_n2.At(I7);

        // get acc0 thread map
        constexpr auto n0_m_n1_to_m_n_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(tn0 * tn1, tn2)),
                       make_pass_through_transform(I1)),
            make_tuple(Sequence<1>{}, Sequence<0>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        constexpr auto threadid_to_n0_m_n1_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(
                make_merge_transform(make_tuple(tn0 * tn1, tm0 * tm1 * tm2 * tm3 * tm4, tn2))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));
        const auto threadid_to_m_n_thread_cluster_adaptor =
            chain_tensor_adaptors(n0_m_n1_to_m_n_adaptor, threadid_to_n0_m_n1_adaptor);

        // get acc0 2D thread cluster & 2D thread slice
        constexpr auto thread_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
            s_blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();
        constexpr auto m0 = thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I0);
        constexpr auto n0 = thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I1);
        constexpr auto m1 = thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I2);
        constexpr auto n1 = thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I3);
        constexpr auto m2 = thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I4);
        constexpr auto m3 = thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I5);
        constexpr auto m4 = thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I6);
        constexpr auto n2 = thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I7);

        constexpr auto thread_cluster_desc_m_n = make_naive_tensor_descriptor_packed(
            make_tuple(tm0 * tm1 * tm2 * tm3 * tm4, tn0 * tn1 * tn2));
        constexpr auto thread_slice_desc_m_n =
            make_naive_tensor_descriptor_packed(make_tuple(m0 * m1 * m2 * m3 * m4, n0 * n1 * n2));

        auto blockwise_softmax = BlockwiseSoftmax<BlockSize,
                                                  FloatGemmAcc,
                                                  decltype(threadid_to_m_n_thread_cluster_adaptor),
                                                  decltype(thread_cluster_desc_m_n),
                                                  decltype(thread_slice_desc_m_n)>{};

        auto blockwise_dropout = BlockwiseDropout<FloatGemmAcc, decltype(thread_slice_desc_m_n)>{
            p_dropout_in_16bits, rp_dropout};

        auto lse_grid_desc_mb_m0_m1_m2_m3_m4 =
            MakeLSEGridDescriptor_MB_M0_M1_M2_M3_M4<decltype(s_blockwise_gemm)>(lse_grid_desc_m);

        constexpr auto lse_thread_desc_mb_m0_m1_m2_m3_m4 =
            make_naive_tensor_descriptor_packed(make_tuple(I1, m0, m1, m2, m3, m4));

        auto lse_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatLSE>(
            lse_thread_desc_mb_m0_m1_m2_m3_m4.GetElementSpaceSize());

        auto acc0_thread_origin = s_blockwise_gemm.CalculateCThreadOriginDataIndex8D(
            Number<0>{}, Number<0>{}, Number<0>{}, Number<0>{});

        auto lse_thread_copy_global_to_vgpr =
            ThreadwiseTensorSliceTransfer_v2<FloatLSE,
                                             FloatLSE,
                                             decltype(lse_grid_desc_mb_m0_m1_m2_m3_m4),
                                             decltype(lse_thread_desc_mb_m0_m1_m2_m3_m4),
                                             Sequence<1, m0, m1, m2, m3, m4>,
                                             Sequence<0, 1, 2, 3, 4, 5>,
                                             5,
                                             1,
                                             1,
                                             true /* ResetCoordAfterRun */>{
                lse_grid_desc_mb_m0_m1_m2_m3_m4,
                make_multi_index(0,                      // mblock
                                 acc0_thread_origin[I0], // mrepeat
                                 acc0_thread_origin[I2], // mwave
                                 acc0_thread_origin[I4], // mperxdl
                                 acc0_thread_origin[I5],
                                 acc0_thread_origin[I6])};

        //
        // z vgpr copy to global
        //
        // z matrix threadwise desc
        constexpr auto z_thread_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5 =
            make_naive_tensor_descriptor_packed(make_tuple(I1,   // MBlockId
                                                           I1,   // NBlockID
                                                           m0,   // MRepeat
                                                           I1,   // NRepeat
                                                           m1,   // MWaveId
                                                           n1,   // NWaveId
                                                           m2,   // MPerXdl
                                                           m3,   // NGroupNum
                                                           m4,   // NInputNum
                                                           n2)); // registerNum

        StaticBuffer<AddressSpaceEnum::Vgpr,
                     unsigned short,
                     z_thread_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5.GetElementSpaceSize(),
                     true>
            z_tenor_buffer;
        z_tenor_buffer.Clear();
        // z matrix global desc
        /*const auto M = q_grid_desc_k0_m_k1.GetLength(I1);
        const auto N = k_grid_desc_k0_n_k1.GetLength(I1);

        auto z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5 =
            MakeZGridDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5(M, N);*/

        auto z_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_z_grid, z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5.GetElementSpaceSize());

        const auto wave_id     = GetGemm0WaveIdx();
        const auto wave_m_n_id = GetGemm0WaveMNIdx(wave_id[I2]); // I2: 0~63

        auto z_thread_copy_vgpr_to_global = ThreadwiseTensorSliceTransfer_v1r3<
            ushort,
            ZDataType,
            decltype(z_thread_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5),
            decltype(z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5),
            tensor_operation::element_wise::PassThrough,
            Sequence<I1, // MBlockId
                     I1, // NBlockID
                     m0, // MRepeat
                     I1, // NRepeat
                     m1, // MWaveId
                     n1, // NWaveId
                     m2, // MPerXdl
                     m3, // NGroupNum
                     m4, // NInputNum
                     n2>,
            Sequence<0, 1, 2, 3, 4, 5, 6, 7, 8, 9>,
            9, // DstVectorDim
            1, // DstScalarPerVector
            InMemoryDataOperationEnum::Set,
            1, // DstScalarStrideInVector
            true>{z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5,
                  make_multi_index(block_work_idx_n, // MBlockId
                                   0,                // NBlockId
                                   0,                // mrepeat
                                   0,                // nrepeat
                                   wave_id[I0],      // MWaveId
                                   wave_id[I1],      // NWaveId
                                   wave_m_n_id[I1],  // MPerXdl
                                   0,                // group
                                   wave_m_n_id[I0],  // NInputIndex
                                   0),
                  tensor_operation::element_wise::PassThrough{}};

        //
        // set up Y dot dY
        //

        // m0, n0 are m/n repeat per wave
        // m1, n1 are number of waves
        constexpr auto p_block_lengths =
            s_blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2().GetLengths();
        constexpr auto P_M0 = p_block_lengths[I0]; // repeats
        constexpr auto P_M1 = p_block_lengths[I2]; // waves
        constexpr auto P_M2 = p_block_lengths[I4]; // xdl
        constexpr auto P_M3 = p_block_lengths[I5];
        constexpr auto P_M4 = p_block_lengths[I6];

        constexpr auto y_thread_desc_m0_m1_o0_o1 = make_naive_tensor_descriptor_packed(make_tuple(
            I1, YDotYGrad_M_O::ThreadSliceLength_M, I1, YDotYGrad_M_O::ThreadSliceLength_O));
        constexpr auto ygrad_thread_desc_m_o     = make_naive_tensor_descriptor_packed(
            make_tuple(YDotYGrad_M_O::ThreadSliceLength_M, YDotYGrad_M_O::ThreadSliceLength_O));
        constexpr auto y_thread_cluster_desc =
            make_cluster_descriptor(Sequence<I1,
                                             YDotYGrad_M_O::ThreadClusterLength_M,
                                             I1,
                                             YDotYGrad_M_O::ThreadClusterLength_O>{},
                                    Sequence<0, 1, 2, 3>{});
        const auto y_thread_cluster_idx =
            y_thread_cluster_desc.CalculateBottomIndex(make_multi_index(get_thread_local_1d_id()));

        constexpr auto ygrad_thread_cluster_desc = make_cluster_descriptor(
            Sequence<YDotYGrad_M_O::ThreadClusterLength_M, YDotYGrad_M_O::ThreadClusterLength_O>{},
            Sequence<0, 1>{});
        const auto ygrad_thread_cluster_idx = ygrad_thread_cluster_desc.CalculateBottomIndex(
            make_multi_index(get_thread_local_1d_id()));

        const auto y_thread_data_on_block_idx =
            y_thread_cluster_idx * y_thread_desc_m0_m1_o0_o1.GetLengths();
        const auto ygrad_thread_data_on_block_idx =
            ygrad_thread_cluster_idx * ygrad_thread_desc_m_o.GetLengths();
        const auto y_thread_data_on_grid_idx = y_thread_data_on_block_idx;

        // performs for y
        auto y_threadwise_copy = ThreadwiseTensorSliceTransfer_v2<
            InputDataType,
            FloatGemmAcc,
            YGridDescriptor_MBlock_MPerBlock_OBlock_OPerBlock,
            decltype(y_thread_desc_m0_m1_o0_o1),
            decltype(y_thread_desc_m0_m1_o0_o1.GetLengths()),
            Sequence<0, 1, 2, 3>,
            3,                                 // SrcVectorDim
            YDotYGrad_M_O::SrcScalarPerVector, // SrcScalarPerVector
            1,                                 // SrcScalarStrideInVector
            true /* ResetCoordAfterRun */>(y_grid_desc_mblock_mperblock_oblock_operblock,
                                           y_thread_data_on_grid_idx);

        // performs for ygrad
        auto ygrad_threadwise_copy = ThreadwiseTensorSliceTransfer_v2<
            GemmDataType,
            FloatGemmAcc,
            decltype(YDotYGrad_M_O::ygrad_block_desc_m_o),
            decltype(ygrad_thread_desc_m_o),
            decltype(ygrad_thread_desc_m_o.GetLengths()),
            Sequence<0, 1>,
            1,                                 // SrcVectorDim
            YDotYGrad_M_O::SrcScalarPerVector, // SrcScalarPerVector
            1,                                 // SrcScalarStrideInVector
            true /* ResetCoordAfterRun */>(YDotYGrad_M_O::ygrad_block_desc_m_o,
                                           ygrad_thread_data_on_block_idx);

        auto y_thread_buf                 = typename YDotYGrad_M_O::SrcBufType{};
        auto ygrad_thread_buf             = typename YDotYGrad_M_O::SrcBufType{};
        auto y_dot_ygrad_thread_accum_buf = typename YDotYGrad_M_O::DstBufType{};
        auto y_dot_ygrad_block_accum_buf  = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<FloatGemmAcc*>(p_shared) + SharedMemTrait::reduction_space_offset,
            MPerBlock);

        constexpr auto y_dot_ygrad_block_desc_mb_m0_m1_m2_m3_m4 =
            make_naive_tensor_descriptor_packed(make_tuple(I1, P_M0, P_M1, P_M2, P_M3, P_M4));
        constexpr auto y_dot_ygrad_thread_desc_mb_m0_m1_m2_m3_m4 =
            lse_thread_desc_mb_m0_m1_m2_m3_m4; // reuse LSE thread descriptor because
                                               // per-thread LSE data and y_dot_ygrad is
                                               // tiled the same way

        auto y_dot_ygrad_thread_copy_lds_to_vgpr =
            ThreadwiseTensorSliceTransfer_v2<FloatGemmAcc,
                                             FloatGemmAcc,
                                             decltype(y_dot_ygrad_block_desc_mb_m0_m1_m2_m3_m4),
                                             decltype(y_dot_ygrad_thread_desc_mb_m0_m1_m2_m3_m4),
                                             Sequence<1, m0, m1, m2, m3, m4>,
                                             Sequence<0, 1, 2, 3, 4, 5>,
                                             5,
                                             1,
                                             1,
                                             true /* ResetCoordAfterRun */>{
                y_dot_ygrad_block_desc_mb_m0_m1_m2_m3_m4,
                make_multi_index(I0,                     // mblock
                                 acc0_thread_origin[I0], // mrepeat
                                 acc0_thread_origin[I2], // mwave
                                 acc0_thread_origin[I4], // mperxdl
                                 acc0_thread_origin[I5],
                                 acc0_thread_origin[I6])};

        auto y_dot_ygrad_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatGemmAcc>(
            y_dot_ygrad_thread_desc_mb_m0_m1_m2_m3_m4.GetElementSpaceSize());

        const index_t num_gemm0_m_block_outer_loop = q_grid_desc_k0_m_k1.GetLength(I1) / MPerBlock;
        constexpr index_t num_gemm1_k_block_inner_loop = MPerBlock / Gemm1KPerBlock;

        if constexpr(Deterministic)
        {
            block_sync_lds();
        }

        // Initialize dK&dV
        kgrad_thread_buf.Clear();
        vgrad_thread_buf.Clear();

        // load k
        gemm_tile_k_blockwise_copy.Run(k_grid_desc_k0_n_k1,
                                       k_grid_buf,
                                       GemmBlockwiseCopy::k_block_desc_k0_n_k1,
                                       k_block_buf,
                                       I0);

        // load v
        gemm_tile_v_blockwise_copy.Run(v_grid_desc_o0_n_o1,
                                       v_grid_buf,
                                       GemmBlockwiseCopy::v_block_desc_k0_n_k1,
                                       v_block_buf,
                                       I0);

        // gemm0 M loop
        index_t gemm0_m_block_outer_index = 0;
        do
        {
            auto m_block_data_idx_on_grid =
                __builtin_amdgcn_readfirstlane(gemm0_m_block_outer_index * MPerBlock);
            if(c0_matrix_mask.IsTileSkippable(
                   m_block_data_idx_on_grid, n_block_data_idx_on_grid, MPerBlock, NPerBlock))
            {
                // move slice window
                gemm_tile_q_blockwise_copy.MoveSrcSliceWindow(
                    q_grid_desc_k0_m_k1,
                    GemmBlockwiseCopy::gemm_tile_q_block_slice_copy_step); // step M
                gemm_tile_ygrad_blockwise_copy.MoveSrcSliceWindow(
                    ygrad_grid_desc_o0_m_o1,
                    GemmBlockwiseCopy::gemm_tile_ygrad_block_slice_copy_step); // step M
                qgrad_thread_copy_vgpr_to_global.MoveDstSliceWindow(
                    qgrad_grid_desc_m0_o0_m1_o1_m2_o2_o3_o4,
                    Gemm2::c_block_slice_copy_step); // step M
                lse_thread_copy_global_to_vgpr.MoveSrcSliceWindow(
                    lse_grid_desc_mb_m0_m1_m2_m3_m4, make_multi_index(1, 0, 0, 0, 0, 0));
                y_threadwise_copy.MoveSrcSliceWindow(y_grid_desc_mblock_mperblock_oblock_operblock,
                                                     make_multi_index(1, 0, 0, 0));
                continue;
            }

            // load ygrad
            gemm_tile_ygrad_blockwise_copy.Run(ygrad_grid_desc_o0_m_o1,
                                               ygrad_grid_buf,
                                               GemmBlockwiseCopy::ygrad_block_desc_k0_m_k1,
                                               ygrad_block_buf,
                                               I0);

            block_sync_lds();

            //
            // calculate Y dot dY
            //

            // clear accum buffers
            y_dot_ygrad_thread_accum_buf.Clear();
            y_dot_ygrad_block_accum_buf.Clear();

            y_threadwise_copy.Run(y_grid_desc_mblock_mperblock_oblock_operblock,
                                  y_grid_buf,
                                  y_thread_desc_m0_m1_o0_o1,
                                  make_tuple(I0, I0, I0, I0),
                                  y_thread_buf);
            ygrad_threadwise_copy.Run(YDotYGrad_M_O::ygrad_block_desc_m_o,
                                      ygrad_block_buf,
                                      ygrad_thread_desc_m_o,
                                      make_tuple(I0, I0),
                                      ygrad_thread_buf);

            static_for<0, YDotYGrad_M_O::ThreadSliceLength_M, 1>{}([&](auto iM) {
                static_for<0, YDotYGrad_M_O::ThreadSliceLength_O, 1>{}([&](auto iO) {
                    constexpr auto y_offset =
                        y_thread_desc_m0_m1_o0_o1.CalculateOffset(make_multi_index(I0, iM, I0, iO));
                    constexpr auto ygrad_offset =
                        ygrad_thread_desc_m_o.CalculateOffset(make_multi_index(iM, iO));
                    y_dot_ygrad_thread_accum_buf(iM) +=
                        y_thread_buf[Number<y_offset>{}] * ygrad_thread_buf[Number<ygrad_offset>{}];
                });
            });

            // blockwise reduction using atomic_add
            block_sync_lds();
            static_for<0, YDotYGrad_M_O::ThreadSliceLength_M, 1>{}([&](auto iM) {
                const auto idx_on_block = y_thread_data_on_block_idx[I1] + iM;
                y_dot_ygrad_block_accum_buf.AtomicAdd(idx_on_block,
                                                      true,
                                                      y_dot_ygrad_thread_accum_buf[iM] *
                                                          p_dropout); // p_dropoutD1
            });
            block_sync_lds();

            // distribute y_dot_ygrad to threads; LDS accum buffer can be safely reused after
            // barrier
            y_dot_ygrad_thread_copy_lds_to_vgpr.Run(y_dot_ygrad_block_desc_mb_m0_m1_m2_m3_m4,
                                                    y_dot_ygrad_block_accum_buf,
                                                    y_dot_ygrad_thread_desc_mb_m0_m1_m2_m3_m4,
                                                    make_tuple(I0, I0, I0, I0, I0, I0),
                                                    y_dot_ygrad_thread_buf);

            lse_thread_copy_global_to_vgpr.Run(lse_grid_desc_mb_m0_m1_m2_m3_m4,
                                               lse_grid_buf,
                                               lse_thread_desc_mb_m0_m1_m2_m3_m4,
                                               make_tuple(I0, I0, I0, I0, I0, I0),
                                               lse_thread_buf);

            // gemm dP
            // dP = dY * V^T
            pgrad_thread_buf.Clear();
            pgrad_blockwise_gemm.Run(ygrad_block_buf, v_block_buf, pgrad_thread_buf);

            // gemm S
            // S = Q * K^T
            s_slash_p_thread_buf.Clear();
            gemm_tile_q_blockwise_copy.Run(q_grid_desc_k0_m_k1,
                                           q_grid_buf,
                                           GemmBlockwiseCopy::q_block_desc_k0_m_k1,
                                           q_block_buf,
                                           I0);
            block_sync_lds();
            s_blockwise_gemm.Run(q_block_buf, k_block_buf, s_slash_p_thread_buf);

            // do MNK padding or upper triangular masking
            if constexpr(MaskOutUpperTriangle || PadN)
            {
                // 8d thread_desc in thread scope
                constexpr auto c_thread_lengths =
                    s_blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2().GetLengths();

                // 8d block_desc in block scope
                constexpr auto c_block_lengths =
                    s_blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2().GetLengths();

                constexpr auto M0 = c_block_lengths[I0];
                constexpr auto N0 = c_block_lengths[I1];
                constexpr auto M1 = c_block_lengths[I2];
                constexpr auto N1 = c_block_lengths[I3];
                constexpr auto M2 = c_block_lengths[I4];
                constexpr auto M3 = c_block_lengths[I5];
                constexpr auto M4 = c_block_lengths[I6];
                constexpr auto N2 = c_block_lengths[I7];

                // works like multi-dimension static_for (static_ford), but provides both the linear
                // index as well as n-d index
                using Acc0TileIterator = SpaceFillingCurve<
                    decltype(c_thread_lengths),
                    typename arithmetic_sequence_gen<0, c_thread_lengths.Size(), 1>::type,
                    typename uniform_sequence_gen<c_thread_lengths.Size(), 1>::type,
                    false>; // SnakeCurved

                constexpr auto block_idx_to_m_n_adaptor = make_single_stage_tensor_adaptor(
                    make_tuple(make_unmerge_transform(make_tuple(M0, M1, M2, M3, M4)),
                               make_unmerge_transform(make_tuple(N0, N1, N2))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                    make_tuple(Sequence<0, 2, 4, 5, 6>{}, Sequence<1, 3, 7>{}));

                static_for<0, Acc0TileIterator::GetNumOfAccess(), 1>{}([&](auto i) {
                    auto acc0_thread_idx = Acc0TileIterator::GetIndex(i) + acc0_thread_origin;
                    auto m_local =
                        block_idx_to_m_n_adaptor.CalculateBottomIndex(acc0_thread_idx)[I0];
                    auto n_local =
                        block_idx_to_m_n_adaptor.CalculateBottomIndex(acc0_thread_idx)[I1];
                    auto m_global = m_local + m_block_data_idx_on_grid;
                    auto n_global = n_local + n_block_data_idx_on_grid;
                    if(c0_matrix_mask.IsMaskedElement(m_global, n_global))
                    {
                        s_slash_p_thread_buf(i) = -ck::NumericLimits<float>::Infinity();
                    }
                    else
                    {
                        s_element_op(s_slash_p_thread_buf(i), s_slash_p_thread_buf[i]);
                    }
                    // bool masked_flag = c0_matrix_mask.IsMaskedElement(m_global, n_global);
                    // s_element_op(s_slash_p_thread_buf(i),
                    //              masked_flag ? -ck::NumericLimits<float>::Infinity()
                    //                          : s_slash_p_thread_buf[i]);
                });
            }
            else
            {
                static_for<0, s_slash_p_thread_buf.Size(), 1>{}([&](auto i) {
                    s_element_op(s_slash_p_thread_buf(i), s_slash_p_thread_buf[i]);
                });
            }

            block_sync_lds(); // wait for lds read in gemm0 blockwise gemm

            // P_i: = softmax(scalar * S_i:)
            // scaling is already performed in the preceding statements with s_element_op
            blockwise_softmax.RunWithPreCalcStats(s_slash_p_thread_buf, lse_thread_buf);

            // save z to global
            if(p_z_grid)
            {
                // P_dropped
                static_for<0, n0, 1>{}([&](auto i) {
                    blockwise_dropout.template ApplyDropout<decltype(s_slash_p_thread_buf),
                                                            decltype(z_tenor_buffer),
                                                            true,
                                                            decltype(n0),
                                                            decltype(i)>(
                        s_slash_p_thread_buf, ph, z_tenor_buffer);

                    z_thread_copy_vgpr_to_global.Run(
                        z_thread_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5,
                        make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0, I0),
                        z_tenor_buffer,
                        z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5,
                        z_grid_buf);
                    z_thread_copy_vgpr_to_global.MoveDstSliceWindow(
                        z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5,
                        make_multi_index(0, 0, 0, 1, 0, 0, 0, 0, 0, 0));
                });
                z_thread_copy_vgpr_to_global.MoveDstSliceWindow(
                    z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5,
                    make_multi_index(0, 0, 0, -n0.value, 0, 0, 0, 0, 0, 0));
            }
            else
            {
                ignore = z_grid_buf;
                // P_dropped
                blockwise_dropout.template ApplyDropout<decltype(s_slash_p_thread_buf), true>(
                    s_slash_p_thread_buf, ph);
            }

            block_sync_lds(); // wait for gemm1 LDS read

            // dS = P * (dP - Y_dot_dY)
            auto& sgrad_thread_buf = pgrad_thread_buf;
            constexpr auto pgrad_thread_tile_iterator =
                pgrad_blockwise_gemm.MakeCThreadTileIterator();
            constexpr auto pgrad_thread_idx_to_m_n_adaptor =
                pgrad_blockwise_gemm.MakeCThreadIndexAdaptor8DTo2D();
            static_for<0, pgrad_thread_tile_iterator.GetNumOfAccess(), 1>{}([&](auto i) {
                constexpr auto pgrad_thread_idx = pgrad_thread_tile_iterator.GetIndex(i);
                constexpr auto m =
                    pgrad_thread_idx_to_m_n_adaptor.CalculateBottomIndex(pgrad_thread_idx)[I0];
                // dS and P has same thread buf layout
                if(s_slash_p_thread_buf[i] >= 0)
                {
                    sgrad_thread_buf(i) =
                        s_slash_p_thread_buf[i] *
                        (pgrad_thread_buf[i] - y_dot_ygrad_thread_buf[Number<m>{}]);
                }
                else
                {
                    sgrad_thread_buf(i) =
                        s_slash_p_thread_buf[i] * y_dot_ygrad_thread_buf[Number<m>{}];
                }
                // bool undropped_flag = s_slash_p_thread_buf[i] >= 0;
                // sgrad_thread_buf(i) =
                //     s_slash_p_thread_buf[i] *
                //     (undropped_flag ? (pgrad_thread_buf[i] - y_dot_ygrad_thread_buf[Number<m>{}])
                //                     : y_dot_ygrad_thread_buf[Number<m>{}]);
            });

            // gemm dV
            // dV = P_drop^T * dY
            {
                // TODO: explore using dynamic buffer for a1 thread buffer
                // For a1_blockwise_copy, the goal is to satisfy pipeline requirements RunRead(),
                // RunWrite(), and MoveSliceWindow(). But it is impossible to implement given that
                // the A1 source buffer is static buffer holding the output of first GEMM and
                // requires constexpr offset by design. Therefore, we pass tensor coordinate offset
                // explicitly in Run() below.

                // main body
                static_for<0, num_gemm1_k_block_inner_loop, 1>{}([&](auto i) {
                    vgrad_gemm_tile_p_blockwise_copy.Run(Gemm1::a_src_thread_desc_k0_m_k1,
                                                         Gemm1::a_block_slice_copy_step * i,
                                                         s_slash_p_thread_buf,
                                                         Gemm1::a_thread_desc_k0_m_k1,
                                                         make_tuple(I0, I0, I0),
                                                         gemm1_a_thread_buf);

                    vgrad_gemm_tile_ygrad_blockwise_copy.Run(
                        Gemm1::b_block_desc_n0_n1_n2_k0_k1_k2_k3,
                        ygrad_block_buf,
                        Gemm1::b_thread_desc_n0_n1_n2_k0_k1_k2_k3,
                        make_tuple(I0, I0, I0, I0, I0, I0, I0),
                        gemm1_b_thread_buf);

                    vgrad_gemm_tile_ygrad_blockwise_copy.MoveSrcSliceWindow(
                        Gemm1::b_block_desc_n0_n1_n2_k0_k1_k2_k3, Gemm1::b_block_slice_copy_step);

                    block_sync_lds();

                    vgrad_blockwise_gemm.Run(
                        gemm1_a_thread_buf, gemm1_b_thread_buf, vgrad_thread_buf);

                    // block_sync_lds();
                });
            } // end gemm dV

            // gemm dK
            // dK = scalar * dS^T * Q
            {
                // TODO: explore using dynamic buffer for a1 thread buffer
                // For a1_blockwise_copy, the goal is to satisfy pipeline requirements RunRead(),
                // RunWrite(), and MoveSliceWindow(). But it is impossible to implement given that
                // the A1 source buffer is static buffer holding the output of first GEMM and
                // requires constexpr offset by design. Therefore, we pass tensor coordinate offset
                // explicitly in Run() below.

                // main body
                static_for<0, num_gemm1_k_block_inner_loop, 1>{}([&](auto i) {
                    kgrad_gemm_tile_sgrad_blockwise_copy.Run(Gemm1::a_src_thread_desc_k0_m_k1,
                                                             Gemm1::a_block_slice_copy_step * i,
                                                             sgrad_thread_buf,
                                                             Gemm1::a_thread_desc_k0_m_k1,
                                                             make_tuple(I0, I0, I0),
                                                             gemm1_a_thread_buf);

                    kgrad_gemm_tile_q_blockwise_copy.Run(Gemm1::b_block_desc_n0_n1_n2_k0_k1_k2_k3,
                                                         q_block_buf,
                                                         Gemm1::b_thread_desc_n0_n1_n2_k0_k1_k2_k3,
                                                         make_tuple(I0, I0, I0, I0, I0, I0, I0),
                                                         gemm1_b_thread_buf);

                    kgrad_gemm_tile_q_blockwise_copy.MoveSrcSliceWindow(
                        Gemm1::b_block_desc_n0_n1_n2_k0_k1_k2_k3, Gemm1::b_block_slice_copy_step);

                    block_sync_lds();

                    kgrad_blockwise_gemm.Run(
                        gemm1_a_thread_buf, gemm1_b_thread_buf, kgrad_thread_buf);

                    // block_sync_lds();
                });
            } // end gemm dK

            SubThreadBlock<BlockSize> gemm2_a_copy_subgroup(s_blockwise_gemm.GetWaveIdx()[I0],
                                                            s_blockwise_gemm.GetWaveIdx()[I1]);

            constexpr index_t num_gemm2_loop = NPerBlock / Gemm2Params::Sum_K;
            static_assert(Gemm2::ASrcBlockSliceWindowIterator::GetNumOfAccess() == num_gemm2_loop,
                          "");

            // TODO: tune gemm2 pipeline
            // gemm dQ
            // dQ = scalar * dS * K
            qgrad_thread_buf.Clear();
            static_for<0, num_gemm2_loop, 1>{}([&](auto gemm2_loop_idx) { // gemm dQ
                // load VGrad Gemm A
                const auto p_slice_idx =
                    Gemm2::ASrcBlockSliceWindowIterator::GetIndexTupleOfNumber(gemm2_loop_idx);
                constexpr auto mwave_range = make_tuple(
                    p_slice_idx[I2],
                    p_slice_idx[I2] + Gemm2Params::ABlockSliceLengths_M0_K0_M1_K1::At(I2));
                constexpr auto nwave_range = make_tuple(
                    p_slice_idx[I3],
                    p_slice_idx[I3] + Gemm2Params::ABlockSliceLengths_M0_K0_M1_K1::At(I3));

                if(gemm2_a_copy_subgroup.IsBelong(mwave_range, nwave_range))
                {
                    qgrad_gemm_tile_sgrad_thread_copy_vgpr_to_lds.Run(
                        Gemm2::a_src_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                        make_tuple(p_slice_idx[I0], p_slice_idx[I1], I0, I0, I0, I0, I0, I0),
                        sgrad_thread_buf,
                        Gemm2::a_block_desc_m0_k0_m1_k1_m2_m3_m4_k2,
                        gemm2_a_block_buf);
                }

                // block_sync_lds(); // sync before write

                qgrad_gemm_tile_k_blockwise_copy.Run(Gemm2::b_block_desc_n0_n1_n2_k0_k1_k2_k3,
                                                     k_block_buf,
                                                     Gemm2::b_thread_desc_n0_n1_n2_k0_k1_k2_k3,
                                                     make_tuple(I0, I0, I0, I0, I0, I0, I0),
                                                     gemm2_b_thread_buf);

                qgrad_gemm_tile_k_blockwise_copy.MoveSrcSliceWindow(
                    Gemm2::b_block_desc_n0_n1_n2_k0_k1_k2_k3, Gemm2::b_block_slice_copy_step);

                block_sync_lds(); // sync before read
                qgrad_blockwise_gemm.Run(gemm2_a_block_buf, gemm2_b_thread_buf, qgrad_thread_buf);

            }); // end gemm dQ
            // atomic_add dQ
            qgrad_thread_copy_vgpr_to_global.Run(Gemm2::c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                                                 make_tuple(I0, I0, I0, I0, I0, I0, I0, I0),
                                                 qgrad_thread_buf,
                                                 qgrad_grid_desc_m0_o0_m1_o1_m2_o2_o3_o4,
                                                 qgrad_grid_buf);

            // move slice window
            gemm_tile_q_blockwise_copy.MoveSrcSliceWindow(
                q_grid_desc_k0_m_k1,
                GemmBlockwiseCopy::gemm_tile_q_block_slice_copy_step); // step M
            gemm_tile_ygrad_blockwise_copy.MoveSrcSliceWindow(
                ygrad_grid_desc_o0_m_o1,
                GemmBlockwiseCopy::gemm_tile_ygrad_block_slice_copy_step); // step M
            vgrad_gemm_tile_ygrad_blockwise_copy.MoveSrcSliceWindow(
                Gemm1::b_block_desc_n0_n1_n2_k0_k1_k2_k3,
                Gemm1::b_block_reset_copy_step); // rewind M
            qgrad_gemm_tile_k_blockwise_copy.MoveSrcSliceWindow(
                Gemm2::b_block_desc_n0_n1_n2_k0_k1_k2_k3,
                Gemm2::b_block_reset_copy_step); // rewind K
            kgrad_gemm_tile_q_blockwise_copy.MoveSrcSliceWindow(
                Gemm1::b_block_desc_n0_n1_n2_k0_k1_k2_k3,
                Gemm1::b_block_reset_copy_step); // rewind M
            qgrad_thread_copy_vgpr_to_global.MoveDstSliceWindow(
                qgrad_grid_desc_m0_o0_m1_o1_m2_o2_o3_o4, Gemm2::c_block_slice_copy_step); // step M
            z_thread_copy_vgpr_to_global.MoveDstSliceWindow(
                z_grid_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5,
                make_multi_index(0, 1, 0, 0, 0, 0, 0, 0, 0, 0));
            lse_thread_copy_global_to_vgpr.MoveSrcSliceWindow(lse_grid_desc_mb_m0_m1_m2_m3_m4,
                                                              make_multi_index(1, 0, 0, 0, 0, 0));
            y_threadwise_copy.MoveSrcSliceWindow(y_grid_desc_mblock_mperblock_oblock_operblock,
                                                 make_multi_index(1, 0, 0, 0));

        } while(++gemm0_m_block_outer_index < num_gemm0_m_block_outer_loop); // end j loop

        // shuffle dK&dV and write
        {
            static_assert(Gemm1::GemmMRepeat % CShuffleMXdlPerWavePerShuffle == 0 &&
                              Gemm1::GemmNRepeat % CShuffleNXdlPerWavePerShuffle == 0,
                          "wrong!");

            constexpr index_t MWave = Gemm1::GemmMWave;
            constexpr index_t NWave = Gemm1::GemmNWave;

            // TODO: hacky, fix it!
            // thread desc same with kgrad_blockwise_gemm
            constexpr auto c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4 =
                vgrad_blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();

            // TODO: hacky, fix it!
            // c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp is only used to get lengths
            // block desc same with kgrad_blockwise_gemm
            constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp =
                vgrad_blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();

            constexpr auto M0 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I0);
            constexpr auto N0 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I1);
            constexpr auto M1 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I2);
            constexpr auto N1 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I3);
            constexpr auto M2 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I4);
            constexpr auto N2 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I5);
            constexpr auto N3 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I6);
            constexpr auto N4 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I7);

            constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
                GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock();

            auto c_shuffle_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                static_cast<FloatCShuffle*>(p_shared),
                c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

            constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4 = transform_tensor_descriptor(
                c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                make_tuple(
                    make_freeze_transform(I0),
                    make_unmerge_transform(make_tuple(
                        Number<CShuffleMXdlPerWavePerShuffle>{}, // M0 (MXdlPerWave) per shuffle
                        M1,                                      // M1 = MWave
                        M2)),                                    // M2 = MPerXdl
                    make_freeze_transform(I0),
                    make_unmerge_transform(make_tuple(
                        Number<CShuffleNXdlPerWavePerShuffle>{}, // N0 (NXdlPerWave) per shuffle
                        N1,                                      // N1 = NWave
                        N2,                                      // N2 * N3 * N4 = NPerXdl
                        N3,
                        N4))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(
                    Sequence<>{}, Sequence<0, 2, 4>{}, Sequence<>{}, Sequence<1, 3, 5, 6, 7>{}));

            // calculate origin of thread output tensor on global memory
            //     blockwise GEMM c matrix starting index
            // index same with kgrad_blockwise_gemm
            const auto c_thread_mtx_on_block =
                vgrad_blockwise_gemm.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

            const index_t m_thread_data_on_block = c_thread_mtx_on_block[I0];
            const index_t n_thread_data_on_block = c_thread_mtx_on_block[I1];

            const auto m_thread_data_on_block_to_m0_m1_m2_adaptor =
                make_single_stage_tensor_adaptor(
                    make_tuple(make_merge_transform(make_tuple(M0, M1, M2))),
                    make_tuple(Sequence<0, 1, 2>{}),
                    make_tuple(Sequence<0>{}));

            const auto m_thread_data_on_block_idx =
                m_thread_data_on_block_to_m0_m1_m2_adaptor.CalculateBottomIndex(
                    make_multi_index(m_thread_data_on_block));

            const auto n_thread_data_on_block_to_n0_n1_n2_n3_n4_adaptor =
                make_single_stage_tensor_adaptor(
                    make_tuple(make_merge_transform(make_tuple(N0, N1, N2, N3, N4))),
                    make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                    make_tuple(Sequence<0>{}));

            const auto n_thread_data_on_block_idx =
                n_thread_data_on_block_to_n0_n1_n2_n3_n4_adaptor.CalculateBottomIndex(
                    make_multi_index(n_thread_data_on_block));

            // shuffle: threadwise copy C from VGPR to LDS
            auto vgrad_thread_copy_vgpr_to_lds =
                ThreadwiseTensorSliceTransfer_v1r3<FloatGemmAcc,
                                                   FloatCShuffle,
                                                   decltype(c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4),
                                                   decltype(c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4),
                                                   SElementwiseOperation,
                                                   Sequence<CShuffleMXdlPerWavePerShuffle,
                                                            CShuffleNXdlPerWavePerShuffle,
                                                            I1,
                                                            I1,
                                                            I1,
                                                            N2,
                                                            I1,
                                                            N4>,
                                                   Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                                   7,
                                                   1,
                                                   InMemoryDataOperationEnum::Set,
                                                   1,
                                                   true>{
                    c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                    make_multi_index(0,
                                     0,
                                     m_thread_data_on_block_idx[I1],
                                     n_thread_data_on_block_idx[I1],
                                     m_thread_data_on_block_idx[I2],
                                     n_thread_data_on_block_idx[I2],
                                     n_thread_data_on_block_idx[I3],
                                     n_thread_data_on_block_idx[I4]),
                    tensor_operation::element_wise::Scale{rp_dropout}};

            // shuffle: blockwise copy C from LDS to global
            auto vgrad_shuffle_block_copy_lds_to_global = ThreadGroupTensorSliceTransfer_v6r1<
                ThisThreadBlock,            // ThreadGroup
                CElementwiseOperation,      // ElementwiseOperation,
                CGlobalMemoryDataOperation, // DstInMemOp,
                Sequence<1,
                         CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                         1,
                         CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>, // BlockSliceLengths,
                CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
                FloatCShuffle,        // typename SrcData,
                OutputDataType,       // typename DstData,
                decltype(c_shuffle_block_desc_mblock_mperblock_nblock_nperblock),
                decltype(vgrad_grid_desc_nblock_nperblock_oblock_operblock),
                Sequence<0, 1, 2, 3>,                           // typename DimAccessOrder,
                3,                                              // index_t VectorDim,
                CShuffleBlockTransferScalarPerVector_NPerBlock, // index_t ScalarPerVector,
                true,  // bool ThreadTransferSrcResetCoordinateAfterRun,
                false> // bool ThreadTransferDstResetCoordinateAfterRun>
                {c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                 make_multi_index(0, 0, 0, 0),
                 vgrad_grid_desc_nblock_nperblock_oblock_operblock,
                 make_multi_index(block_work_idx_n, 0, block_work_idx[I1], 0),
                 c_element_op};

            // shuffle: threadwise copy C from VGPR to LDS
            auto kgrad_thread_copy_vgpr_to_lds =
                ThreadwiseTensorSliceTransfer_v1r3<FloatGemmAcc,
                                                   FloatCShuffle,
                                                   decltype(c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4),
                                                   decltype(c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4),
                                                   SElementwiseOperation,
                                                   Sequence<CShuffleMXdlPerWavePerShuffle,
                                                            CShuffleNXdlPerWavePerShuffle,
                                                            I1,
                                                            I1,
                                                            I1,
                                                            N2,
                                                            I1,
                                                            N4>,
                                                   Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                                   7,
                                                   1,
                                                   InMemoryDataOperationEnum::Set,
                                                   1,
                                                   true>{
                    c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                    make_multi_index(0,
                                     0,
                                     m_thread_data_on_block_idx[I1],
                                     n_thread_data_on_block_idx[I1],
                                     m_thread_data_on_block_idx[I2],
                                     n_thread_data_on_block_idx[I2],
                                     n_thread_data_on_block_idx[I3],
                                     n_thread_data_on_block_idx[I4]),
                    scale_rp_dropout};

            // shuffle: blockwise copy C from LDS to global
            auto kgrad_shuffle_block_copy_lds_to_global = ThreadGroupTensorSliceTransfer_v6r1<
                ThisThreadBlock,            // ThreadGroup
                CElementwiseOperation,      // ElementwiseOperation,
                CGlobalMemoryDataOperation, // DstInMemOp,
                Sequence<1,
                         CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                         1,
                         CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>, // BlockSliceLengths,
                CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
                FloatCShuffle,        // typename SrcData,
                OutputDataType,       // typename DstData,
                decltype(c_shuffle_block_desc_mblock_mperblock_nblock_nperblock),
                decltype(kgrad_grid_desc_nblock_nperblock_oblock_operblock),
                Sequence<0, 1, 2, 3>,                           // typename DimAccessOrder,
                3,                                              // index_t VectorDim,
                CShuffleBlockTransferScalarPerVector_NPerBlock, // index_t ScalarPerVector,
                true,  // bool ThreadTransferSrcResetCoordinateAfterRun,
                false> // bool ThreadTransferDstResetCoordinateAfterRun>
                {c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                 make_multi_index(0, 0, 0, 0),
                 kgrad_grid_desc_nblock_nperblock_oblock_operblock,
                 make_multi_index(block_work_idx_n, 0, block_work_idx[I1], 0),
                 c_element_op};

            // space filling curve for threadwise C in VGPR
            constexpr auto sfc_c_vgpr = SpaceFillingCurve<
                Sequence<Gemm1::GemmMRepeat, Gemm1::GemmNRepeat, 1, 1, 1, N2, 1, N4>,
                Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                Sequence<CShuffleMXdlPerWavePerShuffle,
                         CShuffleNXdlPerWavePerShuffle,
                         1,
                         1,
                         1,
                         N2,
                         1,
                         N4>>{};

            // space filling curve for shuffled blockwise C in global mem
            constexpr auto sfc_c_global =
                SpaceFillingCurve<Sequence<1, NPerBlock, 1, Gemm1NPerBlock>,
                                  Sequence<0, 2, 1, 3>,
                                  Sequence<1,
                                           CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                                           1,
                                           CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>>{};

            constexpr index_t num_access = sfc_c_vgpr.GetNumOfAccess();

            static_assert(num_access == sfc_c_global.GetNumOfAccess(), "wrong!");

            // dK
            static_for<0, num_access, 1>{}([&](auto access_id) {
                // make sure it's safe to write to LDS
                block_sync_lds();

                // each thread write its data from VGPR to LDS
                kgrad_thread_copy_vgpr_to_lds.Run(c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                                                  sfc_c_vgpr.GetIndexTupleOfNumber(access_id),
                                                  kgrad_thread_buf,
                                                  c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                                                  c_shuffle_block_buf);

                // make sure it's safe to read from LDS
                block_sync_lds();

                // each block copy its data from LDS to global
                kgrad_shuffle_block_copy_lds_to_global.Run(
                    c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                    c_shuffle_block_buf,
                    kgrad_grid_desc_nblock_nperblock_oblock_operblock,
                    kgrad_grid_buf);

                if constexpr(access_id < num_access - 1)
                {
                    constexpr auto c_global_step = sfc_c_global.GetForwardStep(access_id);

                    // move on C
                    kgrad_shuffle_block_copy_lds_to_global.MoveDstSliceWindow(
                        kgrad_grid_desc_nblock_nperblock_oblock_operblock, c_global_step);
                }
            });

            // dV
            static_for<0, num_access, 1>{}([&](auto access_id) {
                // make sure it's safe to write to LDS
                block_sync_lds();

                // each thread write its data from VGPR to LDS
                vgrad_thread_copy_vgpr_to_lds.Run(c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                                                  sfc_c_vgpr.GetIndexTupleOfNumber(access_id),
                                                  vgrad_thread_buf,
                                                  c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                                                  c_shuffle_block_buf);

                // make sure it's safe to read from LDS
                block_sync_lds();

                // each block copy its data from LDS to global
                vgrad_shuffle_block_copy_lds_to_global.Run(
                    c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                    c_shuffle_block_buf,
                    vgrad_grid_desc_nblock_nperblock_oblock_operblock,
                    vgrad_grid_buf);

                if constexpr(access_id < num_access - 1)
                {
                    constexpr auto c_global_step = sfc_c_global.GetForwardStep(access_id);

                    // move on C
                    vgrad_shuffle_block_copy_lds_to_global.MoveDstSliceWindow(
                        vgrad_grid_desc_nblock_nperblock_oblock_operblock, c_global_step);
                }
            });
        }
    }
};

} // namespace ck
