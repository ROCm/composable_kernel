// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

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
#include "ck/tensor_operation/gpu/block/blockwise_softmax.hpp"

namespace ck {

template <typename DataType,
          typename FloatGemmAcc,
          typename FloatCShuffle,
          typename FloatLSE,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename AccElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CElementwiseOperation,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          typename AGridDesc_AK0_M_AK1,
          typename BGridDesc_BK0_N_BK1,
          typename VGridDesc_N0_O_N1,
          typename CGridDesc_M_N,
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
          bool PadN,
          bool MaskOutUpperTriangle,
          PipelineVersion PipelineVer = PipelineVersion::v1>
struct GridwiseBatchedGemmSoftmaxGemm_Xdl_CShuffle
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

    // K1 should be Number<...>
    // Gemm0
    static constexpr auto AK0 = Number<KPerBlock / AK1Value>{};
    static constexpr auto BK0 = Number<KPerBlock / BK1Value>{};
    static constexpr auto AK1 = Number<AK1Value>{};
    static constexpr auto BK1 = Number<BK1Value>{};

    static constexpr auto Gemm0MWaves = MPerBlock / (MPerXdl * MXdlPerWave);
    static constexpr auto Gemm0NWaves = NPerBlock / (NPerXdl * NXdlPerWave);

    // Gemm1
    static constexpr auto B1K0 = Number<Gemm1KPerBlock / B1K1Value>{};
    static constexpr auto B1K1 = Number<B1K1Value>{};

    // VGrad Gemm
    template <index_t Sum_M_ = MPerXdl * 2>
    struct VGradGemmTile_N_O_M_
    {
        static constexpr index_t Free0_N = NPerBlock;
        static constexpr index_t Free1_O = Gemm1NPerBlock;
        static constexpr index_t Sum_M   = Sum_M_;

        static constexpr index_t P_M1     = 8; // P will be row-major
        static constexpr index_t P_M0     = Sum_M / P_M1;
        static constexpr index_t P_LdsPad = 0; // how many multiples of M1 per N * M1 elements

        static constexpr index_t YGrad_M1     = 2; // dY assumed row-major, typically =2 for fp16
        static constexpr index_t YGrad_M0     = Sum_M / YGrad_M1;
        static constexpr index_t YGrad_LdsPad = 0; // how many multiples of M1 per N * M1 elements

        static_assert(Sum_M % MPerXdl == 0, "");

        static constexpr index_t YGrad_SrcVectorDim       = 1; // Free1_O dimension
        static constexpr index_t YGrad_SrcScalarPerVector = 4;

        static constexpr index_t GemmNWave   = 2;
        static constexpr index_t GemmOWave   = BlockSize / get_warp_size() / GemmNWave;
        static constexpr index_t GemmNRepeat = Free0_N / GemmNWave / MPerXdl;
        static constexpr index_t GemmORepeat = Free1_O / GemmOWave / NPerXdl;
        static constexpr index_t GemmMPack =
            math::max(math::lcm(P_M1, YGrad_M1),
                      MfmaSelector<DataType, MPerXdl, NPerXdl>::selected_mfma.k_per_blk);

        using YGrad_BlockSliceLengths = Sequence<YGrad_M0, Free1_O, YGrad_M1>;
        using YGrad_ThreadClusterLengths =
            Sequence<BlockSize / (Free1_O / YGrad_SrcScalarPerVector),
                     Free1_O / YGrad_SrcScalarPerVector,
                     1>;
        using YGrad_ThreadClusterArrangeOrder = Sequence<0, 2, 1>;

        __host__ __device__ static constexpr auto GetPBlockDescriptor_M0_N_M1()
        {
            constexpr index_t P_M0 = Sum_M / P_M1;
            return make_naive_tensor_descriptor(
                make_tuple(Number<P_M0>{}, Number<Free0_N>{}, Number<P_M1>{}),
                make_tuple(Number<Free0_N + P_LdsPad>{} * Number<P_M1>{}, Number<P_M1>{}, I1));
        }
        __host__ __device__ static constexpr auto GetYGradBlockDescriptor_M0_O_M1()
        {
            constexpr index_t YGrad_M0 = Sum_M / YGrad_M1;
            return make_naive_tensor_descriptor(
                make_tuple(Number<YGrad_M0>{}, Number<Free1_O>{}, Number<YGrad_M1>{}),
                make_tuple(
                    Number<Free1_O + YGrad_LdsPad>{} * Number<YGrad_M1>{}, Number<YGrad_M1>{}, I1));
        }

        __host__ __device__ static constexpr auto GetPBlockSliceLengths_M0_N0_M1_N1_M2_N2()
        {
            // perform manual unmerge: m -> m_repeat, m_waves, m_per_xdl
            constexpr index_t m  = Sum_M - 1;
            constexpr index_t m2 = m % MPerXdl;
            constexpr index_t m1 = m / MPerXdl % Gemm0MWaves;
            constexpr index_t m0 = m / MPerXdl / Gemm0MWaves % MXdlPerWave;

            // perform manual unmerge: n -> n_repeat, n_waves, n_per_xdl
            constexpr index_t n  = Free0_N - 1;
            constexpr index_t n2 = n % NPerXdl;
            constexpr index_t n1 = n / NPerXdl % Gemm0NWaves;
            constexpr index_t n0 = n / NPerXdl / Gemm0NWaves % NXdlPerWave;

            // assume 256 decomposed into 2 x 4 x 32
            // 1d idx ( 32 - 1) -> 3d idx 0, 0, 31 -> 3d dim 1 x 1 x 32
            // 1d idx (256 - 1) -> 3d idx 1, 3, 31 -> 3d dim 2 x 4 x 32
            return Sequence<m0, n0, m1, n1, m2, n2>{} + Sequence<1, 1, 1, 1, 1, 1>{};
        }

        __host__ __device__ static constexpr auto GetPBlockSliceLengths_M0_N0_M1_N1()
        {
            return generate_sequence_v2(
                [](auto I) { return GetPBlockSliceLengths_M0_N0_M1_N1_M2_N2().At(I); },
                Number<4>{});
        }
    };

    using VGradGemmTile_N_O_M = VGradGemmTile_N_O_M_<>; // tune later

    template <index_t BlockSize_, index_t BlockSliceLength_M_, index_t BlockSliceLength_O_>
    struct YDotYGrad_M_O_
    {
        static constexpr index_t SrcScalarPerVetor = 16 / sizeof(DataType);
        static constexpr auto ThreadClusterLength_O =
            Number<BlockSliceLength_O_ / SrcScalarPerVetor>{};
        static constexpr auto ThreadClusterLength_M = Number<BlockSize_ / ThreadClusterLength_O>{};
        static constexpr auto ThreadSliceLength_O   = Number<SrcScalarPerVetor>{};
        static constexpr auto ThreadSliceLength_M =
            Number<BlockSliceLength_M_ * ThreadClusterLength_O / BlockSize_>{};

        static_assert(ThreadClusterLength_O * ThreadSliceLength_O == BlockSliceLength_O_, "");
        static_assert(ThreadClusterLength_M * ThreadSliceLength_M == BlockSliceLength_M_, "");

        using SrcBufType = StaticBuffer<AddressSpaceEnum::Vgpr,
                                        DataType,
                                        ThreadSliceLength_M * ThreadSliceLength_O,
                                        true>;

        using DstBufType =
            StaticBuffer<AddressSpaceEnum::Vgpr, FloatGemmAcc, ThreadSliceLength_M, true>;
    };
    using YDotYGrad_M_O = YDotYGrad_M_O_<BlockSize, MPerBlock, Gemm1NPerBlock>;

    // QGrad Gemm
    // KGrad Gemm

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    using GridwiseGemmPipe = remove_cvref_t<decltype(
        GridwiseGemmPipeline_Selector<PipelineVer, NumGemmKPrefetchStage>())>;

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
    __host__ __device__ static constexpr auto
    MakeGemm1BMmaTileDescriptor_N0_N1_N2_K(const BBlockDesc_BK0_N_BK1&)
    {
        constexpr index_t Gemm1NWaves = Gemm1NPerBlock / (Gemm1NXdlPerWave * NPerXdl);
        return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<Gemm1NXdlPerWave, Gemm1NWaves, NPerXdl>(
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

    __host__ __device__ static constexpr auto GetB1BlockDescriptor_BK0PerBlock_NPerBlock_BK1()
    {
        // B1 matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(
            make_tuple(B1K0, Number<Gemm1NPerBlock>{}, B1K1),
            make_tuple(Number<Gemm1NPerBlock + B1BlockLdsExtraN>{} * B1K1, B1K1, I1));
    }

    __host__ __device__ static constexpr auto
    GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock()
    {
        constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t NWave = Gemm1NPerBlock / (Gemm1NXdlPerWave * NPerXdl);

        constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            make_naive_tensor_descriptor_packed(
                make_tuple(I1,
                           Number<CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl>{},
                           I1,
                           Number<CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>{}));

        return c_shuffle_block_desc_mblock_mperblock_nblock_nperblock;
    }

    __host__ __device__ static constexpr auto
    GetPBlockDescriptor_NBlock_NPerBlock_MBlock_MPerBlock()
    {
        constexpr auto ptrans_block_desc = make_naive_tensor_descriptor_packed(make_tuple(
            I1, Number<VGradGemmTile_N_O_M::Free0_N>{}, I1, Number<VGradGemmTile_N_O_M::Sum_M>{}));

        return ptrans_block_desc;
    }

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        const index_t gemm0_bytes_end = (SharedMemTrait::a_block_space_size_aligned +
                                         SharedMemTrait::b_block_space_size_aligned) *
                                        sizeof(DataType);
        const index_t gemm1_bytes_end =
            (SharedMemTrait::b1_block_space_offset + SharedMemTrait::b1_block_space_size_aligned) *
            sizeof(DataType);
        const index_t vgrad_gemm_bytes_end = (SharedMemTrait::p_block_space_size_aligned +
                                              SharedMemTrait::ygrad_block_space_size_aligned) *
                                             sizeof(DataType);

        const index_t softmax_bytes_end = (SharedMemTrait::reduction_space_offset +
                                           SharedMemTrait::reduction_space_size_aligned) *
                                          sizeof(FloatGemmAcc);
        const index_t c_block_bytes_end =
            SharedMemTrait::c_block_space_size * sizeof(FloatCShuffle);

        return math::max(gemm0_bytes_end,
                         gemm1_bytes_end,
                         vgrad_gemm_bytes_end,
                         softmax_bytes_end,
                         c_block_bytes_end);
    }

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    template <typename Block2CTileMap>
    __host__ __device__ static constexpr bool
    CheckValidity(const AGridDesc_AK0_M_AK1& a_grid_desc_ak0_m_ak1,
                  const BGridDesc_BK0_N_BK1& b_grid_desc_bk0_n_bk1,
                  const VGridDesc_N0_O_N1& v_grid_desc_n0_o_n1,
                  const CGridDesc_M_N& c_grid_desc_m_n,
                  const Block2CTileMap& block_2_ctile_map)
    {
        static_assert((MPerBlock % (MPerXdl * MXdlPerWave) == 0) &&
                          (NPerBlock % (NXdlPerWave * NPerXdl)) == 0,
                      "Invalid tuning param!");

        const auto M = a_grid_desc_ak0_m_ak1.GetLength(I1);
        const auto N = b_grid_desc_bk0_n_bk1.GetLength(I1);
        const auto K = a_grid_desc_ak0_m_ak1.GetLength(I0) * a_grid_desc_ak0_m_ak1.GetLength(I2);
        const auto Gemm1N = v_grid_desc_n0_o_n1.GetLength(I1);

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

        const auto y_grid_desc_mblock_mperblock_oblock_operblock = transform_tensor_descriptor(
            c_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{})),
                       make_unmerge_transform(make_tuple(NBlock, Number<Gemm1NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

        return y_grid_desc_mblock_mperblock_oblock_operblock;
    }

    __host__ __device__ static constexpr auto
    MakeLSEGridDescriptor_MBlock_MRepeat_NWave_MPerXdl(const LSEGridDesc_M& lse_grid_desc_m)
    {
        const index_t M         = lse_grid_desc_m.GetLength(I0);
        const index_t MBlock    = M / MPerBlock;
        constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);

        const auto lse_grid_desc_mblock_mrepeat_mwave_mperxdl = transform_tensor_descriptor(
            lse_grid_desc_m,
            make_tuple(make_unmerge_transform(
                make_tuple(MBlock, Number<MXdlPerWave>{}, MWave, Number<MPerXdl>{}))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2, 3>{}));

        return lse_grid_desc_mblock_mrepeat_mwave_mperxdl;
    }

    // return block_id to C matrix tile idx (m0, n0) mapping
    __host__ __device__ static constexpr auto
    MakeDefaultBlock2CTileMap(const CGridDesc_M_N& c_grid_desc_m_n)
    {
        return BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, Gemm1NPerBlock, CGridDesc_M_N>(
            c_grid_desc_m_n);
    }

    using YGridDescriptor_MBlock_MPerBlock_OBlock_OPerBlock = remove_cvref_t<decltype(
        MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(CGridDesc_M_N{}))>;

    using DefaultBlock2CTileMap =
        remove_cvref_t<decltype(MakeDefaultBlock2CTileMap(CGridDesc_M_N{}))>;

    // PGrad Gemm has the same layout as P Gemm (A row-major B col-major)
    struct PGradGemmTile_M_N_O
    {
        private:
        static constexpr auto ygrad_block_desc_o0_m_o1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();
        static constexpr auto v_block_desc_o0_n_o1 =
            GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();
        static constexpr index_t KPack = math::max(
            math::lcm(AK1, BK1), MfmaSelector<DataType, MPerXdl, NPerXdl>::selected_mfma.k_per_blk);

        public:
        using BlockwiseGemm = BlockwiseGemmXdlops_v2<
            BlockSize,
            DataType,
            FloatGemmAcc,
            decltype(ygrad_block_desc_o0_m_o1),
            decltype(v_block_desc_o0_n_o1),
            decltype(MakeGemm0AMmaTileDescriptor_M0_M1_M2_K(ygrad_block_desc_o0_m_o1)),
            decltype(MakeGemm0BMmaTileDescriptor_N0_N1_N2_K(v_block_desc_o0_n_o1)),
            MPerBlock,
            NPerBlock,
            KPerBlock,
            MPerXdl,
            NPerXdl,
            MXdlPerWave,
            NXdlPerWave,
            KPack,
            true>;

        // Should have made all input tensors 2D and transform them into appropriate 3D form in
        // kernel to make things more concise - if we can get the compiler to behave
        template <typename YGradGridDesc_M0_O_M1_>
        __device__ static const auto
        MakeYGradGridDesc_O0_M_O1(const YGradGridDesc_M0_O_M1_& ygrad_grid_desc_m0_o_m1)
        {
            const auto M0 = ygrad_grid_desc_m0_o_m1.GetLength(I0);
            const auto O  = ygrad_grid_desc_m0_o_m1.GetLength(I1);
            const auto M1 = ygrad_grid_desc_m0_o_m1.GetLength(I2);

            constexpr auto Y_O1 = AK1;
            const auto Y_O0 = O / Y_O1;

            const auto ygrad_grid_desc_o0_m_o1 = transform_tensor_descriptor(
                ygrad_grid_desc_m0_o_m1,
                make_tuple(make_unmerge_transform(make_tuple(Y_O0, Y_O1)),
                           make_merge_transform_v3_division_mod(make_tuple(M0, M1))),
                make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return ygrad_grid_desc_o0_m_o1;
        }

        template <typename VGridDesc_N0_O_N1_>
        __device__ static const auto
        MakeVGridDesc_O0_N_O1(const VGridDesc_N0_O_N1_& v_grid_desc_n0_o_n1)
        {
            const auto N0 = v_grid_desc_n0_o_n1.GetLength(I0);
            const auto O  = v_grid_desc_n0_o_n1.GetLength(I1);
            const auto N1 = v_grid_desc_n0_o_n1.GetLength(I2);

            constexpr auto V_O1 = BK1;
            const auto V_O0 = O / V_O1;

            const auto v_grid_desc_o0_n_o1 = transform_tensor_descriptor(
                v_grid_desc_n0_o_n1,
                make_tuple(make_unmerge_transform(make_tuple(V_O0, V_O1)),
                           make_merge_transform_v3_division_mod(make_tuple(N0, N1))),
                make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            return v_grid_desc_o0_n_o1;
        }
    };

    struct SharedMemTrait
    {
        // LDS allocation for A and B: be careful of alignment
        static constexpr auto a_block_desc_ak0_m_ak1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();
        static constexpr auto b_block_desc_bk0_n_bk1 =
            GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();
        static constexpr auto b1_block_desc_bk0_n_bk1 =
            GetB1BlockDescriptor_BK0PerBlock_NPerBlock_BK1();
        static constexpr auto p_block_desc_m0_n_m1 =
            VGradGemmTile_N_O_M::GetPBlockDescriptor_M0_N_M1();
        static constexpr auto ygrad_block_desc_m0_o_m1 =
            VGradGemmTile_N_O_M::GetYGradBlockDescriptor_M0_O_M1();

        static constexpr auto max_lds_align = Number<16 / sizeof(DataType)>{};

        static constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);
        static constexpr auto b_block_space_size_aligned = math::integer_least_multiple(
            b_block_desc_bk0_n_bk1.GetElementSpaceSize(), max_lds_align);
        static constexpr auto b1_block_space_size_aligned = math::integer_least_multiple(
            b1_block_desc_bk0_n_bk1.GetElementSpaceSize(), max_lds_align);
        static constexpr auto p_block_space_size_aligned =
            math::integer_least_multiple(p_block_desc_m0_n_m1.GetElementSpaceSize(), max_lds_align);
        static constexpr auto ygrad_block_space_size_aligned = math::integer_least_multiple(
            ygrad_block_desc_m0_o_m1.GetElementSpaceSize(), max_lds_align);

        static constexpr auto a_block_space_offset     = 0;
        static constexpr auto b_block_space_offset     = a_block_space_size_aligned.value;
        static constexpr auto b1_block_space_offset    = 0;
        static constexpr auto p_block_space_offset     = 0;
        static constexpr auto ygrad_block_space_offset = p_block_space_size_aligned.value;

        // LDS allocation for reduction
        static constexpr index_t reduction_space_size_aligned =
            math::integer_least_multiple(BlockSize, max_lds_align);

        static constexpr auto reduction_space_offset = 0;

        // LDS allocation for C shuffle in LDS
        static constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock();
        static constexpr auto c_block_space_size =
            c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize();
    };

    template <bool HasMainKBlockLoop,
              typename Block2CTileMap,
              typename C0MatrixMask,
              typename VGradGridDescriptor_N_O,
              typename YGradGridDesc_M0_O_M1>
    __device__ static void Run(const DataType* __restrict__ p_a_grid,
                               const DataType* __restrict__ p_b_grid,
                               const DataType* __restrict__ p_v_grid,
                               const DataType* __restrict__ p_y_grid,
                               const FloatLSE* __restrict__ p_lse_grid,
                               const DataType* __restrict__ p_ygrad_grid,
                               DataType* __restrict__ p_qgrad_grid,
                               DataType* __restrict__ p_kgrad_grid,
                               DataType* __restrict__ p_vgrad_grid,
                               void* __restrict__ p_shared,
                               const AElementwiseOperation& a_element_op,
                               const BElementwiseOperation& b_element_op,
                               const AccElementwiseOperation& acc_element_op,
                               const B1ElementwiseOperation& b1_element_op,
                               const CElementwiseOperation& c_element_op,
                               const AGridDesc_AK0_M_AK1& a_grid_desc_ak0_m_ak1,
                               const BGridDesc_BK0_N_BK1& b_grid_desc_bk0_n_bk1,
                               const VGridDesc_N0_O_N1& v_grid_desc_n0_o_n1,
                               const YGridDescriptor_MBlock_MPerBlock_OBlock_OPerBlock&
                                   y_grid_desc_mblock_mperblock_oblock_operblock,
                               const LSEGridDesc_M& lse_grid_desc_m,
                               const VGradGridDescriptor_N_O& vgrad_grid_desc_n_o,
                               const YGradGridDesc_M0_O_M1& ygrad_grid_desc_m0_o_m1,
                               const Block2CTileMap& block_2_ctile_map,
                               const C0MatrixMask& c0_matrix_mask)
    {
        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc_ak0_m_ak1.GetElementSpaceSize());
        const auto b_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_grid, b_grid_desc_bk0_n_bk1.GetElementSpaceSize());
        const auto v_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_v_grid, v_grid_desc_n0_o_n1.GetElementSpaceSize());
        const auto y_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_y_grid, y_grid_desc_mblock_mperblock_oblock_operblock.GetElementSpaceSize());
        auto lse_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_lse_grid, lse_grid_desc_m.GetElementSpaceSize());
        const auto ygrad_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_ygrad_grid, ygrad_grid_desc_m0_o_m1.GetElementSpaceSize());
        auto vgrad_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_vgrad_grid, vgrad_grid_desc_n_o.GetElementSpaceSize());

        // divide block work by [M, O]
        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        if(!block_2_ctile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(y_grid_desc_mblock_mperblock_oblock_operblock.GetLength(I0),
                          y_grid_desc_mblock_mperblock_oblock_operblock.GetLength(I2))))
        {
            return;
        }

        // HACK: this force m/gemm1_n_block_data_idx_on_grid into SGPR
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * MPerBlock);

        const index_t gemm1_n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I1] * Gemm1NPerBlock);

        // A matrix in LDS memory, dst of blockwise copy
        constexpr auto a_block_desc_ak0_m_ak1 = GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();

        // B matrix in LDS memory, dst of blockwise copy
        constexpr auto b_block_desc_bk0_n_bk1 = GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

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
                                                DataType,
                                                DataType,
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
                                                DataType,
                                                DataType,
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
        constexpr index_t KPack = math::max(
            math::lcm(AK1, BK1), MfmaSelector<DataType, MPerXdl, NPerXdl>::selected_mfma.k_per_blk);

        auto blockwise_gemm = BlockwiseGemmXdlops_v2<
            BlockSize,
            DataType,
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
            static_cast<DataType*>(p_shared) + SharedMemTrait::a_block_space_offset,
            a_block_desc_ak0_m_ak1.GetElementSpaceSize());

        auto b_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<DataType*>(p_shared) + SharedMemTrait::b_block_space_offset,
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

        constexpr auto A1ThreadSliceK0        = A1ThreadSlice_K0_M_K1[I0];
        constexpr auto A1ThreadSliceM         = A1ThreadSlice_K0_M_K1[I1];
        constexpr auto A1ThreadSliceK1        = A1ThreadSlice_K0_M_K1[I2];
        constexpr auto a1_thread_desc_k0_m_k1 = make_naive_tensor_descriptor(
            A1ThreadSlice_K0_M_K1,
            make_tuple(A1ThreadSliceM * A1ThreadSliceK1, A1ThreadSliceK1, I1));

        // B1 matrix in LDS memory, dst of blockwise copy
        constexpr auto b1_block_desc_bk0_n_bk1 = GetB1BlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        // A1 matrix blockwise copy
        auto a1_blockwise_copy = ThreadwiseTensorSliceTransfer_StaticToStatic<
            FloatGemmAcc,
            DataType,
            decltype(acc_thread_desc_k0_m_k1),
            decltype(a1_thread_desc_k0_m_k1),
            tensor_operation::element_wise::PassThrough,
            Sequence<A1ThreadSliceK0, A1ThreadSliceM, A1ThreadSliceK1>,
            Sequence<1, 0, 2>,
            2,
            n4>{tensor_operation::element_wise::PassThrough{}};

        // B1 matrix blockwise copy
        auto b1_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                BElementwiseOperation,
                                                tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<B1K0, Gemm1NPerBlock, B1K1>,
                                                B1BlockTransferThreadClusterLengths_BK0_N_BK1,
                                                B1BlockTransferThreadClusterArrangeOrder,
                                                DataType,
                                                DataType,
                                                decltype(v_grid_desc_n0_o_n1),
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
                v_grid_desc_n0_o_n1,
                make_multi_index(0, gemm1_n_block_data_idx_on_grid, 0),
                b1_element_op,
                b1_block_desc_bk0_n_bk1,
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});

        auto a1_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, DataType>(
            a1_thread_desc_k0_m_k1.GetElementSpaceSize());

        // reuse LDS space for gemm0's b_block_buf
        auto b1_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<DataType*>(p_shared) + SharedMemTrait::b1_block_space_offset,
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
        constexpr index_t Gemm1KPack =
            MfmaSelector<DataType, MPerXdl, NPerXdl>::selected_mfma.group_size;

        auto gemm1_blockwise_gemm = BlockwiseGemmXdlops_v2<
            BlockSize,
            DataType,
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
            true,       // TransposeC
            Gemm1KPack, // AMmaKStride
            Gemm1KPack * XdlopsGemm<DataType, MPerXdl, NPerXdl, Gemm1KPack, false>{}.K0PerXdlops>{
            // BMmaKStride
            make_tuple(0, 0, 0, 0)}; // A_origin

        auto acc1_thread_buf = gemm1_blockwise_gemm.GetCThreadBuffer();

        //
        // Blockwise softmax
        //
        auto workspace_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<FloatGemmAcc*>(p_shared) + SharedMemTrait::reduction_space_offset,
            SharedMemTrait::reduction_space_size_aligned);

        // get acc0 8D thread cluster
        constexpr auto thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4 =
            blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLengths() /
            blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLengths();
        constexpr auto tm0 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I0);
        constexpr auto tn0 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I1);
        constexpr auto tm1 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I2);
        constexpr auto tn1 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I3);
        constexpr auto tm2 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I4);
        constexpr auto tn2 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I5);
        constexpr auto tn3 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I6);
        constexpr auto tn4 = thread_cluster_m0_n0_m1_n1_m2_n2_n3_n4.At(I7);

        // get acc0 thread map
        constexpr auto m0_n_m1_to_m_n_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(tm0 * tm1, tm2)),
                       make_pass_through_transform(I1)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        constexpr auto threadid_to_m0_n_m1_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(
                make_merge_transform(make_tuple(tm0 * tm1, tn0 * tn1 * tn2 * tn3 * tn4, tm2))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));
        const auto threadid_to_m_n_thread_cluster_adaptor =
            chain_tensor_adaptors(m0_n_m1_to_m_n_adaptor, threadid_to_m0_n_m1_adaptor);

        // get acc0 2D thread cluster & 2D thread slice
        constexpr auto thread_cluster_desc_m_n = make_naive_tensor_descriptor_packed(
            make_tuple(tm0 * tm1 * tm2, tn0 * tn1 * tn2 * tn3 * tn4));
        constexpr auto thread_slice_desc_m_n =
            make_naive_tensor_descriptor_packed(make_tuple(m0 * m1 * m2, n0 * n1 * n2 * n3 * n4));

        auto blockwise_softmax = BlockwiseSoftmax<BlockSize,
                                                  FloatGemmAcc,
                                                  decltype(threadid_to_m_n_thread_cluster_adaptor),
                                                  decltype(thread_cluster_desc_m_n),
                                                  decltype(thread_slice_desc_m_n)>{};

        const index_t num_gemm1_k_block_outer_loop =
            b_grid_desc_bk0_n_bk1.GetLength(I1) / NPerBlock;
        constexpr index_t num_gemm1_k_block_inner_loop = NPerBlock / Gemm1KPerBlock;

        // Initialize C
        StaticBuffer<AddressSpaceEnum::Vgpr, FloatGemmAcc, acc1_thread_buf.Size(), true>
            c_thread_buf;
        c_thread_buf.Clear();

        // Initialize running sum and max of exponentiating row vectors
        using SoftmaxBuf = typename decltype(blockwise_softmax)::BufferType;
        SoftmaxBuf running_sum, running_sum_new, running_max, running_max_new;
        running_sum     = 0;
        running_sum_new = 0;
        running_max     = NumericLimits<FloatGemmAcc>::Lowest();
        running_max_new = NumericLimits<FloatGemmAcc>::Lowest();

        auto lse_grid_desc_mblock_mrepeat_mwave_mperxdl =
            MakeLSEGridDescriptor_MBlock_MRepeat_NWave_MPerXdl(lse_grid_desc_m);

        constexpr auto lse_thread_desc_mblock_mrepeat_mwave_mperxdl =
            make_naive_tensor_descriptor_packed(make_tuple(I1, m0, m1, m2));

        auto lse_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatLSE>(
            lse_thread_desc_mblock_mrepeat_mwave_mperxdl.GetElementSpaceSize());

        auto acc0_thread_origin = blockwise_gemm.CalculateCThreadOriginDataIndex8D(
            Number<0>{}, Number<0>{}, Number<0>{}, Number<0>{});

        auto lse_thread_copy_global_to_vgpr =
            ThreadwiseTensorSliceTransfer_v2<FloatLSE,
                                             FloatLSE,
                                             decltype(lse_grid_desc_mblock_mrepeat_mwave_mperxdl),
                                             decltype(lse_thread_desc_mblock_mrepeat_mwave_mperxdl),
                                             Sequence<1, m0, m1, m2>,
                                             Sequence<0, 1, 2, 3>,
                                             3,
                                             m2,
                                             1,
                                             false>{
                lse_grid_desc_mblock_mrepeat_mwave_mperxdl,
                make_multi_index(block_work_idx[I0],       // mblock
                                 acc0_thread_origin[I0],   // mrepeat
                                 acc0_thread_origin[I2],   // mwave
                                 acc0_thread_origin[I4])}; // mperxdl

        //
        // dV
        //

        // P vgpr to lds: writes vgprs of a subgroup to LDS and transform into m0_n_m1
        // m0, n0 are m/n repeat per wave
        // m1, n1 are number of waves
        constexpr auto p_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4 =
            blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();

        constexpr auto p_block_desc_m0_n_m1 = VGradGemmTile_N_O_M::GetPBlockDescriptor_M0_N_M1();

        constexpr auto p_block_lengths =
            blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLengths();
        constexpr auto P_M0 = p_block_lengths[I0]; // repeats
        constexpr auto P_N0 = p_block_lengths[I1];
        constexpr auto P_M1 = p_block_lengths[I2]; // waves
        constexpr auto P_N1 = p_block_lengths[I3];
        constexpr auto P_M2 = p_block_lengths[I4]; // xdl
        constexpr auto P_N2 = p_block_lengths[I5];
        constexpr auto P_N3 = p_block_lengths[I6];
        constexpr auto P_N4 = p_block_lengths[I7];

        constexpr auto p_block_desc_m0_n0_m1_n1_m2_n2_n3_n4 = [&]() constexpr
        {
            constexpr auto p_block_desc_m_n = transform_tensor_descriptor(
                p_block_desc_m0_n_m1,
                make_tuple(make_merge_transform_v3_division_mod(
                               make_tuple(VGradGemmTile_N_O_M::P_M0, VGradGemmTile_N_O_M::P_M1)),
                           make_pass_through_transform(VGradGemmTile_N_O_M::Free0_N)),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            // HACK: for unmerge transform, the length of highest dim is irrelevant so we put dummy
            // variable I1 there
            return transform_tensor_descriptor(
                p_block_desc_m_n,
                make_tuple(make_unmerge_transform(make_tuple(I1, P_M1, P_M2)),
                           make_unmerge_transform(make_tuple(I1, P_N1, P_N2, P_N3, P_N4))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5, 6, 7>{}));
        }
        ();

        const auto p_thread_origin_nd_idx_on_block = [&]() {
            const auto c_thread_mtx_on_block =
                blockwise_gemm.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

            const index_t m_thread_data_on_block = c_thread_mtx_on_block[I0];
            const index_t n_thread_data_on_block = c_thread_mtx_on_block[I1];

            const auto m_thread_data_on_block_to_m0_m1_m2_adaptor =
                make_single_stage_tensor_adaptor(
                    make_tuple(make_merge_transform(make_tuple(P_M0, P_M1, P_M2))),
                    make_tuple(Sequence<0, 1, 2>{}),
                    make_tuple(Sequence<0>{}));

            const auto m_thread_data_on_block_idx =
                m_thread_data_on_block_to_m0_m1_m2_adaptor.CalculateBottomIndex(
                    make_multi_index(m_thread_data_on_block));

            const auto n_thread_data_on_block_to_n0_n1_n2_n3_n4_adaptor =
                make_single_stage_tensor_adaptor(
                    make_tuple(make_merge_transform(make_tuple(P_N0, P_N1, P_N2, P_N3, P_N4))),
                    make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                    make_tuple(Sequence<0>{}));

            const auto n_thread_data_on_block_idx =
                n_thread_data_on_block_to_n0_n1_n2_n3_n4_adaptor.CalculateBottomIndex(
                    make_multi_index(n_thread_data_on_block));

            return make_tuple(m_thread_data_on_block_idx[I0], // mrepeat
                              n_thread_data_on_block_idx[I0], // nrepeat
                              m_thread_data_on_block_idx[I1], // mwave
                              n_thread_data_on_block_idx[I1], // nwave
                              m_thread_data_on_block_idx[I2], // xdlops
                              n_thread_data_on_block_idx[I2],
                              n_thread_data_on_block_idx[I3],
                              n_thread_data_on_block_idx[I4]);
        }();

        constexpr auto p_block_slice_lengths_m0_n0_m1_n1 =
            VGradGemmTile_N_O_M::GetPBlockSliceLengths_M0_N0_M1_N1(); // mrepeat, nrepeat,
                                                                      // mwaves, nwaves,

        // how to properly perform copy for a sub-workgroup?
        auto p_thread_copy_vgpr_to_lds = ThreadwiseTensorSliceTransfer_v1r3<
            FloatGemmAcc,
            DataType,
            decltype(p_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4),
            decltype(p_block_desc_m0_n0_m1_n1_m2_n2_n3_n4),
            tensor_operation::element_wise::PassThrough,
            Sequence<p_block_slice_lengths_m0_n0_m1_n1[I0], // ThreadSliceLengths
                     p_block_slice_lengths_m0_n0_m1_n1[I1],
                     I1,
                     I1,
                     I1,
                     P_N2,
                     I1,
                     P_N4>,
            Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
            7, // DstVectorDim
            1, // DstScalarPerVector
            InMemoryDataOperationEnum::Set,
            1, // DstScalarStrideInVector
            true>{p_block_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                  make_multi_index(
                      p_thread_origin_nd_idx_on_block[I0],
                      p_thread_origin_nd_idx_on_block[I1],
                      p_thread_origin_nd_idx_on_block[I2] % p_block_slice_lengths_m0_n0_m1_n1[I2],
                      p_thread_origin_nd_idx_on_block[I3] % p_block_slice_lengths_m0_n0_m1_n1[I3],
                      p_thread_origin_nd_idx_on_block[I4],
                      p_thread_origin_nd_idx_on_block[I5],
                      p_thread_origin_nd_idx_on_block[I6],
                      p_thread_origin_nd_idx_on_block[I7]),
                  tensor_operation::element_wise::PassThrough{}};

        // Sequence<p_block_slice_lengths_m0_n0_m1_n1[I0],
        //          p_block_slice_lengths_m0_n0_m1_n1[I1],
        //          I1,
        //          I1,
        //          I1,
        //          P_N2,
        //          I1,
        //          P_N4>{}
        //     .foo();
        // 1, 4, 1, 1, 1, 4, 1, 4

        constexpr auto sfc_p_m0_n0_m1_n1_m2_n2 =
            SpaceFillingCurve<Sequence<P_M0, P_N0, P_M1, P_N1>,
                              Sequence<0, 1, 2, 3>,
                              decltype(p_block_slice_lengths_m0_n0_m1_n1),
                              false>{};

        constexpr auto ygrad_block_desc_m0_o_m1 =
            VGradGemmTile_N_O_M::GetYGradBlockDescriptor_M0_O_M1();

        auto ygrad_blockwise_copy = ThreadGroupTensorSliceTransfer_v4r1<
            ThisThreadBlock,
            tensor_operation::element_wise::PassThrough,
            tensor_operation::element_wise::PassThrough,
            InMemoryDataOperationEnum::Set,
            typename VGradGemmTile_N_O_M::YGrad_BlockSliceLengths,
            typename VGradGemmTile_N_O_M::YGrad_ThreadClusterLengths,
            typename VGradGemmTile_N_O_M::YGrad_ThreadClusterArrangeOrder,
            DataType,
            DataType,
            decltype(ygrad_grid_desc_m0_o_m1),
            decltype(ygrad_block_desc_m0_o_m1),
            typename VGradGemmTile_N_O_M::YGrad_ThreadClusterArrangeOrder, // access order == thread
                                                                           // order
            Sequence<1, 0, 2>,
            VGradGemmTile_N_O_M::YGrad_SrcVectorDim,
            2, // DstVectorDim
            VGradGemmTile_N_O_M::YGrad_SrcScalarPerVector,
            VGradGemmTile_N_O_M::YGrad_M1,
            1,
            1,
            true,
            true,
            1>(ygrad_grid_desc_m0_o_m1,
               make_multi_index(m_block_data_idx_on_grid / VGradGemmTile_N_O_M::YGrad_M1,
                                gemm1_n_block_data_idx_on_grid,
                                0),
               tensor_operation::element_wise::PassThrough{},
               ygrad_block_desc_m0_o_m1,
               make_multi_index(0, 0, 0),
               tensor_operation::element_wise::PassThrough{});

        auto p_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<DataType*>(p_shared) + SharedMemTrait::p_block_space_offset,
            p_block_desc_m0_n_m1.GetElementSpaceSize());
        auto ygrad_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<DataType*>(p_shared) + SharedMemTrait::ygrad_block_space_offset,
            ygrad_block_desc_m0_o_m1.GetElementSpaceSize());

        auto vgrad_blockwise_gemm =
            BlockwiseGemmXdlops_k0mk1_k0nk1_m0n0m1n1m2m3m4n2_v1<BlockSize,
                                                                DataType,
                                                                FloatGemmAcc,
                                                                decltype(p_block_desc_m0_n_m1),
                                                                decltype(ygrad_block_desc_m0_o_m1),
                                                                MPerXdl,
                                                                NPerXdl,
                                                                VGradGemmTile_N_O_M::GemmNRepeat,
                                                                VGradGemmTile_N_O_M::GemmORepeat,
                                                                VGradGemmTile_N_O_M::GemmMPack,
                                                                true>{}; // TranspossC

        auto vgrad_acc_thread_buf = vgrad_blockwise_gemm.GetCThreadBuffer();

        // HACK: for unmerge transform, the length of highest dim is irrelevant so we put dummy
        // variable I1 there
        const auto vgrad_grid_desc_n0_o0_n1_o1_n2_o2 = transform_tensor_descriptor(
            vgrad_grid_desc_n_o,
            make_tuple(
                make_unmerge_transform(make_tuple(I1, VGradGemmTile_N_O_M::GemmNWave, MPerXdl)),
                make_unmerge_transform(make_tuple(I1, VGradGemmTile_N_O_M::GemmOWave, NPerXdl))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}));

        constexpr auto vgrad_thread_desc_n0_o0_n1_o1_n2_o2_o3_o4 =
            vgrad_blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();

        const auto vgrad_grid_desc_n0_o0_n1_o1_n2_o2_o3_o4 =
            vgrad_blockwise_gemm.xdlops_gemm.MakeCDescriptor_M0_N0_M1_N1_M2_N2_N3_N4(
                vgrad_grid_desc_n0_o0_n1_o1_n2_o2);

        const auto vgrad_thread_mtx_on_block_n_o =
            vgrad_blockwise_gemm.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

        constexpr auto vgrad_block_desc_n0_o0_n1_o1_n2_o2_o3_o4 =
            decltype(vgrad_blockwise_gemm)::GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();
        constexpr auto VGrad_N0 = vgrad_block_desc_n0_o0_n1_o1_n2_o2_o3_o4.GetLength(I0);
        constexpr auto VGrad_O0 = vgrad_block_desc_n0_o0_n1_o1_n2_o2_o3_o4.GetLength(I1);
        constexpr auto VGrad_N1 = vgrad_block_desc_n0_o0_n1_o1_n2_o2_o3_o4.GetLength(I2);
        constexpr auto VGrad_O1 = vgrad_block_desc_n0_o0_n1_o1_n2_o2_o3_o4.GetLength(I3);
        constexpr auto VGrad_N2 = vgrad_block_desc_n0_o0_n1_o1_n2_o2_o3_o4.GetLength(I4);
        constexpr auto VGrad_O2 = vgrad_block_desc_n0_o0_n1_o1_n2_o2_o3_o4.GetLength(I5);
        constexpr auto VGrad_O3 = vgrad_block_desc_n0_o0_n1_o1_n2_o2_o3_o4.GetLength(I6);
        constexpr auto VGrad_O4 = vgrad_block_desc_n0_o0_n1_o1_n2_o2_o3_o4.GetLength(I7);

        const index_t n_thread_data_idx_on_grid = vgrad_thread_mtx_on_block_n_o[I0];

        const index_t o_thread_data_idx_on_grid =
            vgrad_thread_mtx_on_block_n_o[I1] + gemm1_n_block_data_idx_on_grid;

        const auto n_thread_data_on_grid_to_n0_n1_n2_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(VGrad_N0, VGrad_N1, VGrad_N2))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));

        const auto n_thread_data_nd_idx_on_grid =
            n_thread_data_on_grid_to_n0_n1_n2_adaptor.CalculateBottomIndex(
                make_multi_index(n_thread_data_idx_on_grid));

        const auto o_thread_data_on_grid_to_o0_o1_o2_o3_o4_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(
                    make_tuple(VGrad_O0, VGrad_O1, VGrad_O2, VGrad_O3, VGrad_O4))),
                make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                make_tuple(Sequence<0>{}));

        const auto o_thread_data_nd_idx_on_grid =
            o_thread_data_on_grid_to_o0_o1_o2_o3_o4_adaptor.CalculateBottomIndex(
                make_multi_index(o_thread_data_idx_on_grid));

        auto vgrad_thread_copy_vgpr_to_global = ThreadwiseTensorSliceTransfer_v1r3<
            FloatGemmAcc,
            DataType,
            decltype(vgrad_thread_desc_n0_o0_n1_o1_n2_o2_o3_o4),
            decltype(vgrad_grid_desc_n0_o0_n1_o1_n2_o2_o3_o4),
            tensor_operation::element_wise::PassThrough, // CElementwiseOperation
            decltype(vgrad_thread_desc_n0_o0_n1_o1_n2_o2_o3_o4.GetLengths()), // SliceLengths
            Sequence<0, 1, 2, 3, 4, 5, 6, 7>,                                 // AccessOrder
            7,                                                                // VectorDim
            2,                                                                // ScalarPerVector
            InMemoryDataOperationEnum::AtomicAdd, // GlobalMemoryDataOperation
            1,                                    // DstScalarStrideInVector
            true>(vgrad_grid_desc_n0_o0_n1_o1_n2_o2_o3_o4,
                  make_multi_index(n_thread_data_nd_idx_on_grid[I0],
                                   o_thread_data_nd_idx_on_grid[I0],
                                   n_thread_data_nd_idx_on_grid[I1],
                                   o_thread_data_nd_idx_on_grid[I1],
                                   n_thread_data_nd_idx_on_grid[I2],
                                   o_thread_data_nd_idx_on_grid[I2],
                                   o_thread_data_nd_idx_on_grid[I3],
                                   o_thread_data_nd_idx_on_grid[I4]),
                  tensor_operation::element_wise::PassThrough{});

#if 0
        if(hipThreadIdx_x % 32 < 4)
        {
            printf("wid %zd tid %zd _n0_o0_n1_o1_n2_o2_o3_o4 %d %d %d %d %d %d %d %d\n",
                   hipBlockIdx_x,
                   hipThreadIdx_x,
                   n_thread_data_nd_idx_on_grid[I0],
                   o_thread_data_nd_idx_on_grid[I0],
                   n_thread_data_nd_idx_on_grid[I1],
                   o_thread_data_nd_idx_on_grid[I1],
                   n_thread_data_nd_idx_on_grid[I2],
                   o_thread_data_nd_idx_on_grid[I2],
                   o_thread_data_nd_idx_on_grid[I3],
                   o_thread_data_nd_idx_on_grid[I4]);
        }
#endif
        // p_thread_slice_copy_step will be in for loop
        constexpr auto ygrad_block_slice_copy_step =
            make_multi_index(VGradGemmTile_N_O_M::YGrad_M0, 0, 0);
        constexpr auto ygrad_block_reset_copy_step =
            make_multi_index(-MPerBlock / VGradGemmTile_N_O_M::YGrad_M1, 0, 0);

        // vgrad gemm output tile
        const auto vgrad_block_slice_copy_step =
            make_multi_index(VGradGemmTile_N_O_M::GemmNRepeat, 0, 0, 0, 0, 0, 0, 0);
#if 0
        if(hipThreadIdx_x == 0)
        {
            printf("bid %zd, n_grid = %d, o_grid = %d, step N0 = %d\n",
                   hipBlockIdx_x,
                   n_thread_data_idx_on_grid,
                   o_thread_data_idx_on_grid,
                   n_thread_data_on_grid_to_n0_n1_n2_adaptor.CalculateBottomIndex(
                       make_multi_index(NPerBlock))[I0]);
        }
#endif

        //
        // dP
        //
        constexpr auto y_thread_desc_m0_m1_o0_o1 = make_naive_tensor_descriptor_packed(make_tuple(
            I1, YDotYGrad_M_O::ThreadSliceLength_M, I1, YDotYGrad_M_O::ThreadSliceLength_O));
        constexpr auto y_thread_cluster_desc =
            make_cluster_descriptor(Sequence<I1,
                                             YDotYGrad_M_O::ThreadClusterLength_M,
                                             I1,
                                             YDotYGrad_M_O::ThreadClusterLength_O>{},
                                    Sequence<0, 1, 2, 3>{});
        const auto y_thread_cluster_idx =
            y_thread_cluster_desc.CalculateBottomIndex(make_multi_index(get_thread_local_1d_id()));

        const auto y_thread_data_on_block_idx =
            y_thread_cluster_idx * y_thread_desc_m0_m1_o0_o1.GetLengths();
        const auto y_thread_data_on_grid_idx =
            make_multi_index(
                block_work_idx[I0], I0, I0 /* all WGs start from o_block_idx = 0 */, I0) +
            y_thread_data_on_block_idx;

        // performs double duty for both y and ygrad
        auto yygrad_threadwise_copy =
            ThreadwiseTensorSliceTransfer_v2<DataType,
                                             DataType,
                                             YGridDescriptor_MBlock_MPerBlock_OBlock_OPerBlock,
                                             decltype(y_thread_desc_m0_m1_o0_o1),
                                             decltype(y_thread_desc_m0_m1_o0_o1.GetLengths()),
                                             Sequence<0, 1, 2, 3>,
                                             3,                                // SrcVectorDim
                                             YDotYGrad_M_O::SrcScalarPerVetor, // SrcScalarPerVector
                                             1, // SrcScalarStrideInVector
                                             true /* ResetCoordAfterRun */,
                                             true /* InvalidElementAsNaN */>(
                y_grid_desc_mblock_mperblock_oblock_operblock, y_thread_data_on_grid_idx);

        auto y_thread_buf                 = typename YDotYGrad_M_O::SrcBufType{};
        auto ygrad_thread_buf             = typename YDotYGrad_M_O::SrcBufType{};
        auto y_dot_ygrad_thread_accum_buf = typename YDotYGrad_M_O::DstBufType{};
        auto y_dot_ygrad_block_accum_buf  = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<FloatGemmAcc*>(p_shared), MPerBlock);

        constexpr auto y_dot_ygrad_block_desc_mblock_mrepeat_mwave_mperxdl =
            make_naive_tensor_descriptor(make_tuple(I1, P_M0, P_M1, P_M2),
                                         make_tuple(P_M0 * P_M1 * P_M2, P_M1 * P_M2, P_M2, I1));
        constexpr auto y_dot_ygrad_thread_desc_mblock_mrepeat_mwave_mperxdl =
            lse_thread_desc_mblock_mrepeat_mwave_mperxdl; // reuse LSE thread descriptor because
                                                          // per-thread LSE data and y_dot_ygrad is
                                                          // tiled the same way

        // TODO ANT: dP Gemm can reuse first blockwise gemm and pipeline
        const auto ygrad_grid_desc_o0_m_o1 =
            PGradGemmTile_M_N_O::MakeYGradGridDesc_O0_M_O1(ygrad_grid_desc_m0_o_m1);
        const auto v_grid_desc_o0_n_o1 =
            PGradGemmTile_M_N_O::MakeVGridDesc_O0_N_O1(v_grid_desc_n0_o_n1);

        // A matrix blockwise copy
        auto pgrad_gemm_tile_ygrad_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                tensor_operation::element_wise::PassThrough,
                                                tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<AK0, MPerBlock, AK1>,
                                                ABlockTransferThreadClusterLengths_AK0_M_AK1,
                                                ABlockTransferThreadClusterArrangeOrder,
                                                DataType,
                                                DataType,
                                                decltype(ygrad_grid_desc_o0_m_o1),
                                                decltype(a_block_desc_ak0_m_ak1), // reuse block buf
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
                ygrad_grid_desc_o0_m_o1,
                make_multi_index(0, m_block_data_idx_on_grid, 0),
                tensor_operation::element_wise::PassThrough{},
                a_block_desc_ak0_m_ak1,
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});

        // B matrix blockwise copy
        auto pgrad_gemm_tile_v_blockwise_copy =
            ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                                tensor_operation::element_wise::PassThrough,
                                                tensor_operation::element_wise::PassThrough,
                                                InMemoryDataOperationEnum::Set,
                                                Sequence<BK0, NPerBlock, BK1>,
                                                BBlockTransferThreadClusterLengths_BK0_N_BK1,
                                                BBlockTransferThreadClusterArrangeOrder,
                                                DataType,
                                                DataType,
                                                decltype(v_grid_desc_o0_n_o1),
                                                decltype(b_block_desc_bk0_n_bk1), // reuse block buf
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
                v_grid_desc_o0_n_o1,
                make_multi_index(0, 0, 0), // will loop over GemmN dimension
                tensor_operation::element_wise::PassThrough{},
                b_block_desc_bk0_n_bk1,
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});

        auto pgrad_blockwise_gemm = typename PGradGemmTile_M_N_O::BlockwiseGemm{};
        auto pgrad_acc_thread_buf = pgrad_blockwise_gemm.GetCThreadBuffer();
        const auto pgrad_gemm_tile_ygrad_block_reset_copy_step =
            make_multi_index(-ygrad_grid_desc_o0_m_o1.GetLength(I0), 0, 0);
        const auto pgrad_gemm_tile_v_block_reset_copy_step =
            make_multi_index(-v_grid_desc_o0_n_o1.GetLength(I0), NPerBlock, 0);

        const index_t num_o_block_main_loop = __builtin_amdgcn_readfirstlane(
            (ygrad_grid_desc_o0_m_o1.GetLength(I0) * ygrad_grid_desc_o0_m_o1.GetLength(I2)) /
            KPerBlock);

        auto y_dot_ygrad_thread_copy_lds_to_vgpr = ThreadwiseTensorSliceTransfer_v2<
            FloatGemmAcc,
            FloatGemmAcc,
            decltype(y_dot_ygrad_block_desc_mblock_mrepeat_mwave_mperxdl),
            decltype(y_dot_ygrad_thread_desc_mblock_mrepeat_mwave_mperxdl),
            Sequence<1, m0, m1, m2>,
            Sequence<0, 1, 2, 3>,
            3,
            m2,
            1,
            false>{y_dot_ygrad_block_desc_mblock_mrepeat_mwave_mperxdl,
                   make_multi_index(I0,                       // mblock
                                    acc0_thread_origin[I0],   // mrepeat
                                    acc0_thread_origin[I2],   // mwave
                                    acc0_thread_origin[I4])}; // mperxdl

        // clear accum buffers
        y_dot_ygrad_thread_accum_buf.Clear();
        y_dot_ygrad_block_accum_buf.Clear();
#if 0
        if(hipThreadIdx_x == 0 && hipBlockIdx_x == 0) printf("lds before accum\n");
        if(hipBlockIdx_x == 0)
        {
            debug::print_shared(y_dot_ygrad_block_accum_buf.p_data_, MPerBlock);
        }
#endif

        auto y_dot_ygrad_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatGemmAcc>(
            y_dot_ygrad_thread_desc_mblock_mrepeat_mwave_mperxdl.GetElementSpaceSize());

        //
        // calculate y dot ygrad
        //
        index_t oblock_idx = 0;
        do
        {
            yygrad_threadwise_copy.Run(y_grid_desc_mblock_mperblock_oblock_operblock,
                                       y_grid_buf,
                                       y_thread_desc_m0_m1_o0_o1,
                                       make_tuple(I0, I0, I0, I0),
                                       y_thread_buf);
            yygrad_threadwise_copy.Run(y_grid_desc_mblock_mperblock_oblock_operblock,
                                       ygrad_grid_buf,
                                       y_thread_desc_m0_m1_o0_o1,
                                       make_tuple(I0, I0, I0, I0),
                                       ygrad_thread_buf);

            static_for<0, YDotYGrad_M_O::ThreadSliceLength_M, 1>{}([&](auto iM) {
                static_for<0, YDotYGrad_M_O::ThreadSliceLength_O, 1>{}([&](auto iO) {
                    constexpr auto offset =
                        y_thread_desc_m0_m1_o0_o1.CalculateOffset(make_multi_index(I0, iM, I0, iO));
                    y_dot_ygrad_thread_accum_buf(iM) +=
                        y_thread_buf[Number<offset>{}] * ygrad_thread_buf[Number<offset>{}];
                });
            });

#if 0
            if (hipThreadIdx_x % 32 < 4 && hipBlockIdx_x == 0)
            {
                printf("bid %zd tid %zd, oblock_idx %d, y_thread_buf[0:3] = %f %f %f %f,  ygrad_thread_buf[0:3] = %f %f %f %f\n",
                       hipBlockIdx_x,
                       hipThreadIdx_x,
                       oblock_idx,
                       (float)y_thread_buf[I0],
                       (float)y_thread_buf[I1],
                       (float)y_thread_buf[I2],
                       (float)y_thread_buf[I3],
                       (float)ygrad_thread_buf[I0],
                       (float)ygrad_thread_buf[I1],
                       (float)ygrad_thread_buf[I2],
                       (float)ygrad_thread_buf[I3]);
            }
#endif
            yygrad_threadwise_copy.MoveSrcSliceWindow(y_grid_desc_mblock_mperblock_oblock_operblock,
                                                      make_multi_index(0, 0, 1, 0));

            oblock_idx++;
        } while(oblock_idx < y_grid_desc_mblock_mperblock_oblock_operblock.GetLength(I2));

        // blockwise reduction using atomic_add
        block_sync_lds();
        static_for<0, YDotYGrad_M_O::ThreadSliceLength_M, 1>{}([&](auto iM) {
            const auto idx_on_block = y_thread_data_on_block_idx[I1] + iM;
            y_dot_ygrad_block_accum_buf.AtomicAdd(
                idx_on_block, true, y_dot_ygrad_thread_accum_buf[iM]);
        });
        block_sync_lds();

#if 1
        if(hipThreadIdx_x == 0 && hipBlockIdx_x == 0) printf("lds after accum\n");
        if(hipBlockIdx_x == 0)
        {
            debug::print_shared(y_dot_ygrad_block_accum_buf.p_data_, MPerBlock);
        }
#endif

        // distribute y_dot_ygrad to threads; LDS accum buffer can be safely accessed after barrier
        y_dot_ygrad_thread_copy_lds_to_vgpr.Run(
            y_dot_ygrad_block_desc_mblock_mrepeat_mwave_mperxdl,
            y_dot_ygrad_block_accum_buf,
            y_dot_ygrad_thread_desc_mblock_mrepeat_mwave_mperxdl,
            make_tuple(I0, I0, I0, I0),
            y_dot_ygrad_thread_buf);

#if 0
        if(hipBlockIdx_x < 4 && hipThreadIdx_x % 32 < 4)
        {
            printf("bid %zd tid %zd, y_m0_m1_o0_o1 = %d, %d, %d, %d\n",
                   hipBlockIdx_x,
                   hipThreadIdx_x,
                   y_thread_data_on_grid_idx[I0],
                   y_thread_data_on_grid_idx[I1],
                   y_thread_data_on_grid_idx[I2],
                   y_thread_data_on_grid_idx[I3]);
        }
#endif

        lse_thread_copy_global_to_vgpr.Run(lse_grid_desc_mblock_mrepeat_mwave_mperxdl,
                                           lse_grid_buf,
                                           lse_thread_desc_mblock_mrepeat_mwave_mperxdl,
                                           make_tuple(I0, I0, I0, I0),
                                           lse_thread_buf);

        // gemm1 K loop
        index_t gemm1_k_block_outer_index = 0;
        do
        {
            auto n_block_data_idx_on_grid =
                __builtin_amdgcn_readfirstlane(gemm1_k_block_outer_index * NPerBlock);
            if(c0_matrix_mask.IsTileSkippable(
                   m_block_data_idx_on_grid, n_block_data_idx_on_grid, MPerBlock, NPerBlock))
            {
                continue;
            }
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

            // do MNK padding or upper triangular masking
            if constexpr(MaskOutUpperTriangle || PadN)
            {
                // 8d thread_desc in thread scope
                constexpr auto c_thread_lengths =
                    blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLengths();

                // 8d block_desc in block scope
                constexpr auto c_block_lengths =
                    blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4().GetLengths();

                constexpr auto M0 = c_block_lengths[I0];
                constexpr auto N0 = c_block_lengths[I1];
                constexpr auto M1 = c_block_lengths[I2];
                constexpr auto N1 = c_block_lengths[I3];
                constexpr auto M2 = c_block_lengths[I4];
                constexpr auto N2 = c_block_lengths[I5];
                constexpr auto N3 = c_block_lengths[I6];
                constexpr auto N4 = c_block_lengths[I7];

                // works like multi-dimension static_for (static_ford), but provides both the linear
                // index as well as n-d index
                using Acc0TileIterator = SpaceFillingCurve<
                    decltype(c_thread_lengths),
                    typename arithmetic_sequence_gen<0, c_thread_lengths.Size(), 1>::type,
                    typename uniform_sequence_gen<c_thread_lengths.Size(), 1>::type,
                    false>; // SnakeCurved

                auto acc0_thread_origin = blockwise_gemm.CalculateCThreadOriginDataIndex8D(
                    Number<0>{}, Number<0>{}, Number<0>{}, Number<0>{});

                constexpr auto block_idx_to_m_n_adaptor = make_single_stage_tensor_adaptor(
                    make_tuple(make_unmerge_transform(make_tuple(M0, M1, M2)),
                               make_unmerge_transform(make_tuple(N0, N1, N2, N3, N4))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                    make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5, 6, 7>{}));

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
                        acc_thread_buf(i) = -ck::NumericLimits<float>::Infinity();
                    }
                    else
                    {
                        acc_element_op(acc_thread_buf(i), acc_thread_buf[i]);
                    }
                });
            }
            else
            {
                static_for<0, acc_thread_buf.Size(), 1>{}(
                    [&](auto i) { acc_element_op(acc_thread_buf(i), acc_thread_buf[i]); });
            }

            block_sync_lds(); // wait for lds read in gemm0 blockwise gemm

#if 0
            if (hipBlockIdx_x == 0 && hipThreadIdx_x % 32 < 4)
            {
                printf("tid %zd, S[0:3] = %f, %f, %f, %f\n",
                       hipThreadIdx_x,
                       acc_thread_buf[I0],
                       acc_thread_buf[I1],
                       acc_thread_buf[I2],
                       acc_thread_buf[I3]);
            }
#endif

            // P_i: = softmax(S_i:)
            blockwise_softmax.RunWithPreCalcStats(acc_thread_buf, lse_thread_buf);

#if 0
            if (hipBlockIdx_x == 0 && hipThreadIdx_x % 32 < 4)
            {
                printf("tid %zd, P[0:3] = %f, %f, %f, %f\n",
                       hipThreadIdx_x,
                       acc_thread_buf[I0],
                       acc_thread_buf[I1],
                       acc_thread_buf[I2],
                       acc_thread_buf[I3]);
            }
#endif

            block_sync_lds(); // wait for gemm1 LDS read

            SubThreadBlock<BlockSize> p_thread_copy_subgroup(blockwise_gemm.GetWaveIdx()[I0],
                                                             blockwise_gemm.GetWaveIdx()[I1]);

            constexpr index_t num_vgrad_gemm_loop = MPerBlock / VGradGemmTile_N_O_M::Sum_M;
            static_assert(sfc_p_m0_n0_m1_n1_m2_n2.GetNumOfAccess() == num_vgrad_gemm_loop, "");

            vgrad_acc_thread_buf.Clear();

            // TODO ANT: single buffer prefetch pipeline
            static_for<0, num_vgrad_gemm_loop, 1>{}([&](auto vgrad_gemm_loop_idx) { // gemm dV
                // load VGrad Gemm B
                ygrad_blockwise_copy.RunRead(ygrad_grid_desc_m0_o_m1, ygrad_grid_buf);

                // load VGrad Gemm A
                const auto p_nd_idx =
                    sfc_p_m0_n0_m1_n1_m2_n2.GetIndexTupleOfNumber(vgrad_gemm_loop_idx);
                constexpr auto mwave_range =
                    make_tuple(p_nd_idx[I2], p_nd_idx[I2] + p_block_slice_lengths_m0_n0_m1_n1[I2]);
                constexpr auto nwave_range =
                    make_tuple(p_nd_idx[I3], p_nd_idx[I3] + p_block_slice_lengths_m0_n0_m1_n1[I3]);
#if 0
                if(hipThreadIdx_x % 64 == 0)
                {
                    printf(
                        "VGrad P vgrad_gemm_loop_idx %d, wave_id = %d, mrepeat, nrepeat, mwave, "
                        "nwave = %d, %d, %d, %d, active %d\n",
                        vgrad_gemm_loop_idx.value,
                        (int)hipThreadIdx_x / 64,
                        p_nd_idx[I0].value,
                        p_nd_idx[I1].value,
                        p_nd_idx[I2].value,
                        p_nd_idx[I3].value,
                        p_thread_copy_subgroup.IsBelong(mwave_range, nwave_range));
                }
#endif
                if(p_thread_copy_subgroup.IsBelong(mwave_range, nwave_range))
                {
                    p_thread_copy_vgpr_to_lds.Run(
                        p_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                        make_tuple(p_nd_idx[I0], p_nd_idx[I1], I0, I0, I0, I0, I0, I0),
                        acc_thread_buf,
                        p_block_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                        p_block_buf);
                }

                // ygrad slice window is moved with MoveSrcSliceWindow() since it is dynamic buffer
                // p slice window is moved by loop index
                ygrad_blockwise_copy.MoveSrcSliceWindow(ygrad_grid_desc_m0_o_m1,
                                                        ygrad_block_slice_copy_step);

                block_sync_lds(); // sync before write
                ygrad_blockwise_copy.RunWrite(ygrad_block_desc_m0_o_m1, ygrad_block_buf);
#if 0
                if (hipBlockIdx_x == 0)
                {
                    debug::print_shared(
                        p_block_buf.p_data_,
                        index_t(p_block_desc_m0_n0_m1_n1_m2_n2_n3_n4.GetElementSpaceSize()));
                }
#endif
#if 0
                if (hipBlockIdx_x == 0)
                {
                    debug::print_shared(ygrad_block_buf.p_data_,
                                        index_t(ygrad_block_desc_m0_o_m1.GetElementSpaceSize()));
                }
#endif

                block_sync_lds(); // sync before read
                vgrad_blockwise_gemm.Run(p_block_buf, ygrad_block_buf, vgrad_acc_thread_buf);

#if 0
                if(hipBlockIdx_x == 0 && hipThreadIdx_x % 32 < 4)
                {
                    printf("outer %d inner %d tid %zd, dV[0:3] = %f, %f, %f, %f\n",
                           gemm1_k_block_outer_index,
                           vgrad_gemm_loop_idx.value,
                           hipThreadIdx_x,
                           vgrad_acc_thread_buf[I0],
                           vgrad_acc_thread_buf[I1],
                           vgrad_acc_thread_buf[I2],
                           vgrad_acc_thread_buf[I3]);
                }
#endif
            }); // end gemm dV

            // atomic_add dV
            vgrad_thread_copy_vgpr_to_global.Run(vgrad_thread_desc_n0_o0_n1_o1_n2_o2_o3_o4,
                                                 make_tuple(I0, I0, I0, I0, I0, I0, I0, I0),
                                                 vgrad_acc_thread_buf,
                                                 vgrad_grid_desc_n0_o0_n1_o1_n2_o2_o3_o4,
                                                 vgrad_grid_buf);

            // gemm dP
            // assume size K == size O so has main block loop
            block_sync_lds();
            gridwise_gemm_pipeline.template Run<HasMainKBlockLoop>(
                ygrad_grid_desc_o0_m_o1,
                a_block_desc_ak0_m_ak1, // reuse
                pgrad_gemm_tile_ygrad_blockwise_copy,
                ygrad_grid_buf,
                a_block_buf,             // reuse
                a_block_slice_copy_step, // reuse
                v_grid_desc_o0_n_o1,
                b_block_desc_bk0_n_bk1, // reuse
                pgrad_gemm_tile_v_blockwise_copy,
                v_grid_buf,
                b_block_buf,             // reuse
                b_block_slice_copy_step, // reuse
                pgrad_blockwise_gemm,
                pgrad_acc_thread_buf,
                num_o_block_main_loop);
#if 0
            if (hipBlockIdx_x == 0 && hipThreadIdx_x % 32 < 4)
            {
                printf("j loop idx %d, tid %zd, dP[0:3] = %f, %f, %f, %f\n",
                       gemm1_k_block_outer_index,
                       hipThreadIdx_x,
                       pgrad_acc_thread_buf[I0],
                       pgrad_acc_thread_buf[I1],
                       pgrad_acc_thread_buf[I2],
                       pgrad_acc_thread_buf[I3]);
            }
#endif
            // move slice window
            a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc_ak0_m_ak1,
                                                a_block_reset_copy_step); // rewind K
            b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc_bk0_n_bk1,
                                                b_block_reset_copy_step); // rewind K and step N
            ygrad_blockwise_copy.MoveSrcSliceWindow(ygrad_grid_desc_m0_o_m1,
                                                    ygrad_block_reset_copy_step); // rewind M
            vgrad_thread_copy_vgpr_to_global.MoveDstSliceWindow(
                vgrad_grid_desc_n0_o0_n1_o1_n2_o2_o3_o4, vgrad_block_slice_copy_step); // step N
            pgrad_gemm_tile_ygrad_blockwise_copy.MoveSrcSliceWindow(
                ygrad_grid_desc_o0_m_o1, pgrad_gemm_tile_ygrad_block_reset_copy_step); // rewind O
            pgrad_gemm_tile_v_blockwise_copy.MoveSrcSliceWindow(
                v_grid_desc_o0_n_o1,
                pgrad_gemm_tile_v_block_reset_copy_step); // rewind O and step N

        } while(++gemm1_k_block_outer_index < num_gemm1_k_block_outer_loop); // end j loop

// TODO ANT:
// shuffle dQ and write
#if 0
        {
            static_assert(MXdlPerWave % CShuffleMXdlPerWavePerShuffle == 0 &&
                              Gemm1NXdlPerWave % CShuffleNXdlPerWavePerShuffle == 0,
                          "wrong!");

            constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);
            constexpr index_t NWave = Gemm1NPerBlock / (Gemm1NXdlPerWave * NPerXdl);

            // TODO: hacky, fix it!
            constexpr auto c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4 =
                gemm1_blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();

            // TODO: hacky, fix it!
            // c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp is only used to get lengths
            constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp =
                gemm1_blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();

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
            const auto c_thread_mtx_on_block =
                gemm1_blockwise_gemm.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

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
            auto c_thread_copy_vgpr_to_lds =
                ThreadwiseTensorSliceTransfer_v1r3<FloatGemmAcc,
                                                   FloatCShuffle,
                                                   decltype(c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4),
                                                   decltype(c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4),
                                                   tensor_operation::element_wise::PassThrough,
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
                    tensor_operation::element_wise::PassThrough{}};

            // shuffle: blockwise copy C from LDS to global
            auto c_shuffle_block_copy_lds_to_global = ThreadGroupTensorSliceTransfer_v6r1<
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
                DataType,             // typename DstData,
                decltype(c_shuffle_block_desc_mblock_mperblock_nblock_nperblock),
                decltype(c_grid_desc_mblock_mperblock_nblock_nperblock),
                Sequence<0, 1, 2, 3>,                           // typename DimAccessOrder,
                3,                                              // index_t VectorDim,
                CShuffleBlockTransferScalarPerVector_NPerBlock, // index_t ScalarPerVector,
                true,  // bool ThreadTransferSrcResetCoordinateAfterRun,
                false> // bool ThreadTransferDstResetCoordinateAfterRun>
                {c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                 make_multi_index(0, 0, 0, 0),
                 c_grid_desc_mblock_mperblock_nblock_nperblock,
                 make_multi_index(block_work_idx[I0], 0, block_work_idx[I1], 0),
                 c_element_op};

            // space filling curve for threadwise C in VGPR
            constexpr auto sfc_c_vgpr =
                SpaceFillingCurve<Sequence<MXdlPerWave, Gemm1NXdlPerWave, 1, 1, 1, N2, 1, N4>,
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
                SpaceFillingCurve<Sequence<1, MPerBlock, 1, Gemm1NPerBlock>,
                                  Sequence<0, 2, 1, 3>,
                                  Sequence<1,
                                           CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                                           1,
                                           CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>>{};

            constexpr index_t num_access = sfc_c_vgpr.GetNumOfAccess();

            static_assert(num_access == sfc_c_global.GetNumOfAccess(), "wrong!");

            static_for<0, num_access, 1>{}([&](auto access_id) {
                // make sure it's safe to write to LDS
                block_sync_lds();

                // each thread write its data from VGPR to LDS
                c_thread_copy_vgpr_to_lds.Run(c_thread_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                                              sfc_c_vgpr.GetIndexTupleOfNumber(access_id),
                                              c_thread_buf,
                                              c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4,
                                              c_shuffle_block_buf);

                // make sure it's safe to read from LDS
                block_sync_lds();

                // each block copy its data from LDS to global
                c_shuffle_block_copy_lds_to_global.Run(
                    c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                    c_shuffle_block_buf,
                    c_grid_desc_mblock_mperblock_nblock_nperblock,
                    c_grid_buf);

                if constexpr(access_id < num_access - 1)
                {
                    constexpr auto c_global_step = sfc_c_global.GetForwardStep(access_id);

                    // move on C
                    c_shuffle_block_copy_lds_to_global.MoveDstSliceWindow(
                        c_grid_desc_mblock_mperblock_nblock_nperblock, c_global_step);
                }
            });
        }
#endif
    }
};

} // namespace ck
