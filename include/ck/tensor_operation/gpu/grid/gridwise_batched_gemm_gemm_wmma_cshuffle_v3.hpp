// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdarg>
#include "ck/utility/env.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmma_selector.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_wmma.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

// Gemm0: AccOp(A [M x K] x B0 [K x L], D0) = Acc [M x L]
// Gemm1: CDEOp1(Acc [M x L] x B1 [L x N], D1) = E [M x N]
template <typename ADataType,
          typename B0DataType,
          typename D0sDataType,
          typename Acc0DataType,
          typename B1DataType,
          typename D1sDataType,
          typename Acc1DataType,
          typename CShuffleDataType,
          typename E1DataType,
          typename AElementwiseOperation,
          typename B0ElementwiseOperation,
          typename AccElementwiseOperation,
          typename B1ElementwiseOperation,
          typename CDEElementwiseOperation,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          typename AGridDesc,
          typename B0GridDesc,
          typename D0sGridDesc,
          typename B1GridDesc,
          typename D1sGridDesc,
          typename E1GridDesc,
          index_t MPerBlock,
          index_t LPerBlock,
          index_t KPerBlock,
          index_t AK1Value,
          index_t BK1Value,
          index_t NPerBlock,
          index_t LTilePerBlock,
          index_t L1Value,
          index_t MPerWmma,
          index_t LPerWmma,
          index_t NPerWmma,
          index_t MRepeat,
          index_t LRepeat,
          index_t NRepeat,
          index_t BlockSize,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_K1,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          bool ABlockLdsExtraM,
          typename B0BlockTransferThreadClusterLengths_K0_L_K1,
          typename B0BlockTransferThreadClusterArrangeOrder,
          typename B0BlockTransferSrcAccessOrder,
          index_t B0BlockTransferSrcVectorDim,
          index_t B0BlockTransferSrcScalarPerVector,
          index_t B0BlockTransferDstScalarPerVector_K1,
          bool B0ThreadTransferSrcResetCoordinateAfterRun,
          bool B0BlockLdsExtraL,
          index_t CDE0BlockTransferSrcScalarPerVector,
          typename B1BlockTransferThreadClusterLengths_L0_N_L1,
          typename B1BlockTransferThreadClusterArrangeOrder,
          typename B1BlockTransferSrcAccessOrder,
          index_t B1BlockTransferSrcVectorDim,
          index_t B1BlockTransferSrcScalarPerVector,
          index_t B1BlockTransferDstScalarPerVector_L1,
          bool B1ThreadTransferSrcResetCoordinateAfterRun,
          bool B1BlockLdsExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CDE1ShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDE1ShuffleBlockTransferScalarPerVector_NPerBlock,
          bool PadN,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1>
struct GridwiseBatchedGemmGemm_wmma_cshuffle_v3
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};

    static constexpr index_t NumD0Tensor = D0sDataType::Size();
    static constexpr index_t NumD1Tensor = D1sDataType::Size();

    static constexpr auto AK0 = Number<KPerBlock / AK1Value>{};
    static constexpr auto AK1 = Number<AK1Value>{};
    static constexpr auto BK0 = Number<KPerBlock / BK1Value>{};
    static constexpr auto BK1 = Number<BK1Value>{};

    static constexpr auto L0PerBlock = LTilePerBlock / L1Value;
    static constexpr auto AL0 = Number<L0PerBlock / 2>{}; // TODO: Where does this 2 come from?
    static constexpr auto AL1 = Number<L1Value>{};
    static constexpr auto BL0 = Number<L0PerBlock>{};
    static constexpr auto BL1 = Number<L1Value>{};

    static constexpr auto MWaves    = MPerBlock / (MRepeat * MPerWmma);
    static constexpr auto LWaves    = LPerBlock / (LRepeat * LPerWmma);
    static constexpr auto NWaves    = NPerBlock / (NRepeat * NPerWmma);
    static constexpr auto WaveSize0 = BlockSize / (MWaves * LWaves);
    static constexpr auto WaveSize1 = BlockSize / (MWaves * NWaves);
    static constexpr auto WaveSize  = WaveSize0;

    static_assert(
        WaveSize0 == 32 || WaveSize0 == 64,
        "Misconfigured wave parameters: BlockSize / (MWaves * LWaves) != 32/64 threads per wave");
    static_assert(
        WaveSize1 == 32 || WaveSize1 == 64,
        "Misconfigured wave parameters: BlockSize / (MWaves * NWaves) != 32/64 threads per wave");

    static constexpr index_t KPerWmmaBlk =
        WmmaSelector<ADataType, B0DataType, Acc0DataType, MPerWmma, LPerWmma>::selected_wmma
            .k_per_blk;

    static constexpr index_t KInnerA = ck::math::integer_divide_ceil(AK1Value, KPerWmmaBlk);

    static constexpr index_t KInnerB = ck::math::integer_divide_ceil(BK1Value, KPerWmmaBlk);

    static constexpr index_t KInner = ck::math::min(KInnerA, KInnerB);

    static constexpr index_t KPack =
        KInner *
        WmmaSelector<ADataType, B0DataType, Acc0DataType, MPerWmma, LPerWmma>::selected_wmma
            .k_per_wmma;

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    __host__ __device__ static constexpr auto MakeABlockDescriptor()
    {
        constexpr auto a_block_desc = [&]() {
            // K0->M->K1 Per Block
            constexpr auto K0PerBlock    = KPerBlock / AK1;
            constexpr auto max_lds_align = AK1;

            if constexpr(ABlockLdsExtraM)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(Number<K0PerBlock>{}, Number<MPerBlock>{}, AK1),
                    make_tuple(Number<MPerBlock + 1>{} * AK1, AK1, I1));
            }
            else
            {
                return make_naive_tensor_descriptor_aligned(
                    make_tuple(Number<K0PerBlock>{}, Number<MPerBlock>{}, AK1), max_lds_align);
            }
        }();

        return a_block_desc;
    }

    __host__ __device__ static constexpr auto MakeB0BlockDescriptor()
    {
        constexpr auto b0_block_desc = [&]() {
            // K0->L->BK1 Per Block
            constexpr auto K0PerBlock    = KPerBlock / BK1;
            constexpr auto max_lds_align = BK1;

            if constexpr(B0BlockLdsExtraL)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(Number<K0PerBlock>{}, Number<LPerBlock>{}, BK1),
                    make_tuple(Number<LPerBlock + 1>{} * BK1, BK1, I1));
            }
            else
            {
                return make_naive_tensor_descriptor_aligned(
                    make_tuple(Number<K0PerBlock>{}, Number<LPerBlock>{}, BK1), max_lds_align);
            }
        }();

        return b0_block_desc;
    }

    __host__ __device__ static constexpr auto MakeB1BlockDescriptor()
    {
        constexpr auto b1_block_desc = [&]() {
            // L0->N->BL1 Per Block
            constexpr auto max_lds_align = BL1;

            if constexpr(B1BlockLdsExtraN)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(Number<L0PerBlock>{}, Number<NPerBlock>{}, BL1),
                    make_tuple(Number<NPerBlock + 1>{} * BL1, BL1, I1));
            }
            else
            {
                return make_naive_tensor_descriptor_aligned(
                    make_tuple(Number<L0PerBlock>{}, Number<NPerBlock>{}, BL1), max_lds_align);
            }
        }();

        return b1_block_desc;
    }

    __host__ __device__ static constexpr auto MakeABlockSliceCopyStep()
    {
        constexpr auto a_block_copy_step = [&]() { return make_multi_index(AK0, 0, 0); }();
        return a_block_copy_step;
    }

    __host__ __device__ static constexpr auto MakeB0BlockSliceCopyStep()
    {
        constexpr auto b0_block_copy_step = [&]() { return make_multi_index(BK0, 0, 0); }();
        return b0_block_copy_step;
    }

    __host__ __device__ static constexpr auto MakeB1BlockSliceCopyStep()
    {
        constexpr auto b1_block_copy_step = [&]() { return make_multi_index(L0PerBlock, 0, 0); }();
        return b1_block_copy_step;
    }

    // ck::Tuple<const D0DataType1*, const D0DataType2*, ...>
    static constexpr auto MakeD0sGridPointer()
    {
        return generate_tuple(
            [&](auto i) {
                using D0iDataType = remove_cvref_t<tuple_element_t<i.value, D0sDataType>>;

                return static_cast<const D0iDataType*>(nullptr);
            },
            Number<NumD0Tensor>{});
    }

    // ck::Tuple<const D1DataType1*, const D1DataType2*, ...>
    static constexpr auto MakeD1sGridPointer()
    {
        return generate_tuple(
            [&](auto i) {
                using D1iDataType = remove_cvref_t<tuple_element_t<i.value, D1sDataType>>;

                return static_cast<const D1iDataType*>(nullptr);
            },
            Number<NumD1Tensor>{});
    }

    __device__ static auto GetGemm0WaveIdx()
    {
        const index_t thread_id = get_thread_local_1d_id();

        constexpr auto threadid_to_wave_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(MWaves, LWaves, WaveSize))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));

        return threadid_to_wave_idx_adaptor.CalculateBottomIndex(make_multi_index(thread_id));
    }

    __device__ static auto GetGemm0WaveMNIdx(const index_t thread_id)
    {
        constexpr auto wave_threadid_to_mn_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(WaveSize / LPerWmma, LPerWmma))),
            make_tuple(Sequence<0, 1>{}),
            make_tuple(Sequence<0>{}));

        return wave_threadid_to_mn_idx_adaptor.CalculateBottomIndex(make_multi_index(thread_id));
    }

    template <index_t MNRepeat, index_t MNWaves, index_t MNPerWmma, typename BlockDesc>
    __host__ __device__ static constexpr auto MakeWmmaTileDescriptor(const BlockDesc&)
    {
        // K0_MN_K1 -> K0_MNRepeat_MNWaves_KRow_MNPerWmma_K1
        constexpr auto K0 = BlockDesc{}.GetLength(I0);
        constexpr auto K1 = BlockDesc{}.GetLength(I2);
#ifdef __gfx12__
        constexpr auto KRow = I2;
#else
        constexpr auto KRow = I1;
#endif

        if constexpr(KInner > 1)
        {
            // KPack = KInner * KPerWmma
            // K1 = KInner * KPerWmmaBlk
            // Each thread loads multiple tiles with one instruction
            // 1 - MNRepeat - K0 / KRow - MNWaves - KRow - MNPerWmma - K1
            return transform_tensor_descriptor(
                BlockDesc{},
                make_tuple(
                    make_unmerge_transform(make_tuple(Number<K0 / (KRow)>{}, KRow, Number<1>{})),
                    make_unmerge_transform(
                        make_tuple(Number<MNRepeat>{}, Number<MNWaves>{}, Number<MNPerWmma>{})),
                    make_pass_through_transform(Number<K1>{})),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<2, 4, 0>{}, Sequence<1, 3, 5>{}, Sequence<6>{}));
        }
        else
        {
            // KPack = KPerWmma (KInner == 1)
            if constexpr(K1 <= KPerWmmaBlk)
            {
                // K1 <= single tile (KPerWmmaBlk)
                // Each thread will load KPerWmmaBlk for the WMMA instruction
                // Since K1 <= single tile, K0 is unmerged first over KPack / KRow / K1
                // (rest of the single WMMA tile for single thread) and then over KRow
                // (rest of the single WMMA tile for single wave)
                // KPack / KRow / K1 - MNRepeat - K0 / KRow - MNWaves - KRow - MNPerWmma - K1
                return transform_tensor_descriptor(
                    BlockDesc{},
                    make_tuple(make_unmerge_transform(make_tuple(
                                   Number<K0 / (KPack / K1)>{}, KRow, Number<KPack / KRow / K1>{})),
                               make_unmerge_transform(make_tuple(
                                   Number<MNRepeat>{}, Number<MNWaves>{}, Number<MNPerWmma>{})),
                               make_pass_through_transform(Number<K1>{})),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<2, 4, 0>{}, Sequence<1, 3, 5>{}, Sequence<6>{}));
            }
            else
            {
                // K1 > single tile (KPerWmmaBlk)
                // Each thread will load KPerWmmaBlk for the WMMA instruction
                // Since K1 > single tile, each thread loads KPerWmmaBlk and the next
                // KPerWmmaBlk chunk is loaded by a different thread in the same wave (WMMA layout).
                // This layout is needed to support for example AK1 > single tile and
                // BK1 <= single tile in the same gemm
                // KPack / KPerWmmaBlk / KRow - MNRepeat - K0 / KRow - MNWaves - KRow - MNPerWmma -
                // K1
                constexpr auto desc1 = transform_tensor_descriptor(
                    BlockDesc{},
                    make_tuple(
                        make_pass_through_transform(Number<K0>{}),
                        make_unmerge_transform(
                            make_tuple(Number<MNRepeat>{}, Number<MNWaves>{}, Number<MNPerWmma>{})),
                        make_unmerge_transform(make_tuple(Number<K1 / KPack>{},
                                                          Number<KPack / KPerWmmaBlk / KRow>{},
                                                          Number<KRow>{},
                                                          Number<KPerWmmaBlk>{}))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<2>{}, Sequence<1, 4, 6>{}, Sequence<3, 0, 5, 7>{}));

                return transform_tensor_descriptor(
                    desc1,
                    make_tuple(make_pass_through_transform(Number<KPack / KPerWmmaBlk / KRow>{}),
                               make_pass_through_transform(Number<MNRepeat>{}),
                               make_merge_transform(make_tuple(Number<K0>{}, Number<K1 / KPack>{})),
                               make_pass_through_transform(Number<MNWaves>{}),
                               make_pass_through_transform(Number<KRow>{}),
                               make_pass_through_transform(Number<MNPerWmma>{}),
                               make_pass_through_transform(Number<KPerWmmaBlk>{})),
                    make_tuple(Sequence<0>{},
                               Sequence<1>{},
                               Sequence<2, 3>{},
                               Sequence<4>{},
                               Sequence<5>{},
                               Sequence<6>{},
                               Sequence<7>{}),
                    make_tuple(Sequence<0>{},
                               Sequence<1>{},
                               Sequence<2>{},
                               Sequence<3>{},
                               Sequence<4>{},
                               Sequence<5>{},
                               Sequence<6>{}));
            }
        }
    }

    template <typename ABlockDesc_>
    __host__ __device__ static constexpr auto MakeAWaveDescriptor(const ABlockDesc_&)
    {
        return MakeWmmaTileDescriptor<MRepeat, MWaves, MPerWmma>(ABlockDesc_{});
    }

    template <typename B0BlockDesc_>
    __host__ __device__ static constexpr auto MakeB0WaveDescriptor(const B0BlockDesc_&)
    {
        return MakeWmmaTileDescriptor<LRepeat, LWaves, LPerWmma>(B0BlockDesc_{});
    }

    template <typename A1BlockDesc_AL0_M_AL1>
    __host__ __device__ static constexpr auto
    MakeA1WaveDescriptor_L0_M0_M1_M2_L1(const A1BlockDesc_AL0_M_AL1&)
    {
        constexpr index_t A_L0 = A1BlockDesc_AL0_M_AL1{}.GetLength(I0);
        constexpr index_t A_L1 = A1BlockDesc_AL0_M_AL1{}.GetLength(I2);
        constexpr auto A_LRow  = I1;
        return transform_tensor_descriptor(
            A1BlockDesc_AL0_M_AL1{},
            make_tuple(make_unmerge_transform(make_tuple(Number<A_L0>{}, A_LRow)),
                       make_unmerge_transform(make_tuple(Number<MRepeat>{}, I1, I1)),
                       make_pass_through_transform(Number<A_L1>{})),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0, 3>{}, Sequence<1, 2, 4>{}, Sequence<5>{}));
    }

    template <typename B1BlockDesc_>
    __host__ __device__ static constexpr auto MakeB1WaveDescriptor(const B1BlockDesc_&)
    {

        constexpr auto b1_wave_desc = [&]() {
            // BL0_N_BL1 -> BL0_NRepeat_Nwaves_NPerWmma_BL1
            constexpr auto B_L0 = B1BlockDesc_{}.GetLength(I0);
            constexpr auto B_L1 = B1BlockDesc_{}.GetLength(I2);
#ifdef __gfx12__
            constexpr auto B_LRow = I2;
#else
            constexpr auto B_LRow = I1;
#endif
            return transform_tensor_descriptor(
                B1BlockDesc_{},
                make_tuple(make_unmerge_transform(make_tuple(Number<B_L0 / B_LRow>{}, B_LRow)),
                           make_unmerge_transform(
                               make_tuple(Number<NRepeat>{}, Number<NWaves>{}, Number<NPerWmma>{})),
                           make_pass_through_transform(Number<B_L1>{})),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0, 3>{}, Sequence<1, 2, 4>{}, Sequence<5>{}));
        }();

        return b1_wave_desc;
    }

    __host__ __device__ static constexpr auto
    // *Caution Here repeat is shuffle repeat
    GetCShuffleBlockDescriptor_MShRepeat_MPerShRepeat_NShRepeat_NPerShRepeat()
    {
        constexpr auto c1_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat =
            make_naive_tensor_descriptor_packed(
                make_tuple(I1,
                           Number<CShuffleMRepeatPerShuffle * MWaves * MPerWmma>{},
                           I1,
                           Number<CShuffleNRepeatPerShuffle * NWaves * NPerWmma>{}));

        return c1_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat;
    }

    __host__ __device__ static constexpr index_t GetSharedMemoryNumberOfByte()
    {
        // LDS allocation for A and B: be careful of alignment
        const index_t gemm0_bytes_end =
            (SharedMemTrait::a_block_space_size_aligned * sizeof(ADataType) +
             SharedMemTrait::b0_block_space_size_aligned * sizeof(B0DataType));

        const index_t gemm1_bytes_end =
            (SharedMemTrait::b1_block_space_offset +
             SharedMemTrait::b1_block_space_size_aligned * sizeof(B1DataType));

        const index_t acc0_bytes_end =
            SharedMemTrait::reduction_space_offset +
            SharedMemTrait::reduction_space_size_aligned * sizeof(Acc0DataType);

        const index_t c_block_bytes_end =
            SharedMemTrait::c_block_space_size * sizeof(CShuffleDataType);

        return math::max(gemm0_bytes_end, gemm1_bytes_end, acc0_bytes_end, c_block_bytes_end);
    }

    // Blockwise gemm pipeline for gemm0, this replaces the old GridwiseGemmPipe +
    // BlockwiseGemmWMMA. The latter had two enableLDS bools which we don't
    // have anymore, the new pipelines ALWAYS use lds. Furthermore the original BlockwiseGemmWMMA
    // used TransposeC = true which we still need to make the operation work.
    using BlockwiseGemmPipe =
        remove_cvref_t<decltype(BlockGemmPipeline_Selector<
                                BlkGemmPipelineVer,
                                BlkGemmPipeSched,
                                BlockSize,
                                ADataType,
                                B0DataType,
                                // TODO: Check if these compute types should always be
                                //  equal to data type.
                                ADataType,  // ComputeTypeA
                                B0DataType, // ComputeTypeB
                                Acc0DataType,
                                decltype(MakeAWaveDescriptor(MakeABlockDescriptor())),
                                decltype(MakeB0WaveDescriptor(MakeB0BlockDescriptor())),
                                ABlockTransferSrcScalarPerVector,
                                B0BlockTransferSrcScalarPerVector,
                                MPerBlock,
                                LPerBlock,
                                KPerBlock,
                                MPerWmma,
                                LPerWmma,
                                MRepeat,
                                LRepeat,
                                KPack,
                                KInner,
                                true>())>; // TransposeC (must be true to work), C' = B' x A'

    // block_id to matrix tile idx (m0, n0) mapping is controlled by {M01, N01}
    template <typename Block2ETileMap>
    __host__ __device__ static constexpr bool CheckValidity(const AGridDesc& a_grid_desc,
                                                            const B0GridDesc& b0_grid_desc,
                                                            const D0sGridDesc& d0s_grid_desc,
                                                            const B1GridDesc& b1_grid_desc,
                                                            const D1sGridDesc& d1s_grid_desc,
                                                            const E1GridDesc& c_grid_desc_m_n,
                                                            const Block2ETileMap& block_2_etile_map)
    {
        // Print lambda with env check and printf() style formmating.
        const char* curFunc = __func__;
        auto print          = [&curFunc](const char* format, ...) -> void {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-nonliteral"
#endif
                va_list args;
                va_start(args, format);
                std::vfprintf(stdout, format, args);
                va_end(args);

#ifdef __clang__
#pragma clang diagnostic pop
#endif
                std::cout << "In file: " << __FILE__ << ", function: " << curFunc << "\n";
            }
        };

        static_assert(MPerBlock % (MPerWmma * MRepeat) == 0 &&
                          LPerBlock % (LPerWmma * LRepeat) == 0 &&
                          NPerBlock % (NPerWmma * NRepeat) == 0,
                      "Invalid tuning param!");

        const auto M = a_grid_desc.GetLength(I1);
        const auto L = b0_grid_desc.GetLength(I1);
        const auto K = a_grid_desc.GetLength(I0) * a_grid_desc.GetLength(I2);
        const auto N = b1_grid_desc.GetLength(I1);

        if(!(M == c_grid_desc_m_n.GetLength(I0) && N == c_grid_desc_m_n.GetLength(I1)))
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                print("GridwiseOp: M/N Length err, A_M/N = %d, %d | C_M/N = %d, %d\n",
                      M,
                      N,
                      c_grid_desc_m_n.GetLength(I0),
                      c_grid_desc_m_n.GetLength(I1));
            }
            return false;
        }

        bool d0s_desc_valid = true;
        static_for<0, NumD0Tensor, 1>{}([&](auto i) {
            if(!(M == d0s_grid_desc[i].GetLength(I0) && L == d0s_grid_desc[i].GetLength(I1)))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    print("GridwiseOp: M/L Length err, A_M/B0_L = %d, %d | D0s_M/N = %d, %d\n",
                          M,
                          L,
                          d0s_grid_desc[i].GetLength(I0),
                          d0s_grid_desc[i].GetLength(I1));
                }

                d0s_desc_valid = false;
            }
        });

        bool d1s_desc_valid = true;
        static_for<0, NumD1Tensor, 1>{}([&](auto i) {
            if(!(M == d1s_grid_desc[i].GetLength(I0) && N == d1s_grid_desc[i].GetLength(I1)))
            {
                if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
                {
                    print("GridwiseOp: M/N Length err, A_M/N = %d, %d | D1s_M/N = %d, %d\n",
                          M,
                          N,
                          d1s_grid_desc[i].GetLength(I0),
                          d1s_grid_desc[i].GetLength(I1));
                }
                d1s_desc_valid = false;
            }
        });

        if(!(d0s_desc_valid && d1s_desc_valid))
        {
            return false;
        }

        if(!(M % MPerBlock == 0 && L % LPerBlock == 0 && K % KPerBlock == 0 && N % NPerBlock == 0))
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                print("GridwiseOp: M/L/K/N Division err, M/L/K/N = %d, %d, %d, %d | "
                      "M/L/K/NPerBlock = "
                      "%d, %d, %d, %d\n",
                      M,
                      L,
                      K,
                      N,
                      MPerBlock,
                      LPerBlock,
                      KPerBlock,
                      NPerBlock);
            }
            return false;
        }

        // check gemm1 gridwise gemm pipeline
        if(!(LPerBlock % LTilePerBlock == 0))
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                print("GridwiseOp: inner loop division, L/LTilePerblock: %d, %d\n",
                      LPerBlock,
                      LTilePerBlock);
            }
            return false;
        }

        if(!block_2_etile_map.CheckValidity(c_grid_desc_m_n))
        {
            if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
            {
                print("GridwiseOp: invalid block_2_etile_map\n");
            }
            return false;
        }

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        return true;
    }

    __host__ __device__ static constexpr bool CalculateHasMainKBlockLoop(index_t K)
    {
        const index_t num_loop = math::integer_divide_ceil(K, KPerBlock);
        return BlockwiseGemmPipe::BlockHasHotloop(num_loop);
    }

    __host__ __device__ static constexpr TailNumber CalculateKBlockLoopTailNum(index_t K)
    {
        const index_t num_loop = math::integer_divide_ceil(K, KPerBlock);
        return BlockwiseGemmPipe::BlockLoopTailNum(num_loop);
    }

    __host__ __device__ static constexpr auto
    MakeE1GridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(const E1GridDesc& e_grid_desc_m_n)
    {
        const auto M = e_grid_desc_m_n.GetLength(I0);
        const auto N = e_grid_desc_m_n.GetLength(I1);

        const auto MBlock = M / MPerBlock;
        const auto NBlock = N / NPerBlock;

        const auto e1_grid_desc_mblock_mperblock_nblock_nperblock = transform_tensor_descriptor(
            e_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{})),
                       make_unmerge_transform(make_tuple(NBlock, Number<NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

        return e1_grid_desc_mblock_mperblock_nblock_nperblock;
    }

    // D0 desc for source in blockwise copy
    template <typename D0GridDesc_M_N>
    __host__ __device__ static constexpr auto
    MakeD0GridDescriptor_MRepeat_MWave_MThreadPerSubGroup_NRepeat_NWave_NSubGroup_NAccVgprs(
        const D0GridDesc_M_N& d0_grid_desc_m_n)
    {
        const auto M = d0_grid_desc_m_n.GetLength(I0);
        const auto N = d0_grid_desc_m_n.GetLength(I1);

        constexpr auto wmma =
            WmmaSelector<ADataType, B0DataType, Acc0DataType, MPerWmma, LPerWmma>::selected_wmma;

        return transform_tensor_descriptor(
            d0_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(M / MPerBlock, MRepeat, MWaves, MPerWmma)),
                       make_unmerge_transform(make_tuple(N / LPerBlock,
                                                         LRepeat,
                                                         LWaves,
                                                         WaveSize / LPerWmma,
                                                         wmma.num_acc_vgprs_per_wave))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2, 3, 4>{}, Sequence<1, 5, 6, 7, 8>{}));
    }

    // D0s desc for source in blockwise copy
    __host__ __device__ static constexpr auto
    MakeD0sGridDescriptor_MRepeat_MWave_MThreadPerSubGroup_NRepeat_NWave_NSubGroup_NAccVgprs(
        const D0sGridDesc& ds_grid_desc_m_n)
    {
        return generate_tuple(
            [&](auto i) {
                return MakeD0GridDescriptor_MRepeat_MWave_MThreadPerSubGroup_NRepeat_NWave_NSubGroup_NAccVgprs(
                    ds_grid_desc_m_n[i]);
            },
            Number<NumD0Tensor>{});
    }
    // Ds desc for source in blockwise copy
    template <typename DsGridDescriptor_M_N>
    __host__ __device__ static constexpr auto
    MakeD1sGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
        const DsGridDescriptor_M_N& ds_grid_desc_m_n)
    {
        return generate_tuple(
            [&](auto i) {
                return MakeE1GridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(ds_grid_desc_m_n[i]);
            },
            Number<NumD1Tensor>{});
    }

    // return block_id to E matrix tile idx (m0, n0) mapping
    __host__ __device__ static constexpr auto MakeDefaultBlock2ETileMap(
        const E1GridDesc& c_grid_desc_m_n, index_t /* M01 */, index_t /* N01 */)
    {
        return BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock, E1GridDesc>(c_grid_desc_m_n);
    }

    using E1GridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock =
        remove_cvref_t<decltype(MakeE1GridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            E1GridDesc{}))>;

    using D0sGridDescriptor_MRepeat_MWave_MThreadPerSubGroup_NRepeat_NWave_NSubGroup_NAccVgprs =
        remove_cvref_t<
            decltype(MakeD0sGridDescriptor_MRepeat_MWave_MThreadPerSubGroup_NRepeat_NWave_NSubGroup_NAccVgprs(
                D0sGridDesc{}))>;

    using D1sGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock =
        remove_cvref_t<decltype(MakeD1sGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            D1sGridDesc{}))>;
    using DefaultBlock2ETileMap =
        remove_cvref_t<decltype(MakeDefaultBlock2ETileMap(E1GridDesc{}, 1, 1))>;

    struct SharedMemTrait
    {
        // LDS allocation for A and B: be careful of alignment
        static constexpr auto max_lds_align = math::lcm(math::lcm(AK1, BK1), BL1);

        static constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            MakeABlockDescriptor().GetElementSpaceSize(), max_lds_align);
        static constexpr auto b0_block_space_size_aligned = math::integer_least_multiple(
            MakeB0BlockDescriptor().GetElementSpaceSize(), max_lds_align);
        static constexpr auto b1_block_space_size_aligned = math::integer_least_multiple(
            MakeB1BlockDescriptor().GetElementSpaceSize(), max_lds_align);

        static constexpr auto a_block_space_offset  = 0;
        static constexpr auto b0_block_space_offset = a_block_space_size_aligned;
        static constexpr auto b1_block_space_offset = 0;

        // LDS allocation for reduction
        // Feature to add, IntraThread Reduction
        static constexpr index_t reduction_space_size_aligned =
            math::integer_least_multiple(BlockSize, max_lds_align);

        static constexpr auto reduction_space_offset = 0;

        // LDS allocation for C shuffle in LDS
        static constexpr auto c_block_space_size =
            GetCShuffleBlockDescriptor_MShRepeat_MPerShRepeat_NShRepeat_NPerShRepeat()
                .GetElementSpaceSize();
    };

    using D0sGridPointer = decltype(MakeD0sGridPointer());
    using D1sGridPointer = decltype(MakeD1sGridPointer());

    template <bool HasMainKBlockLoop,
              TailNumber TailNum,
              typename Block2ETileMap = DefaultBlock2ETileMap>
    __device__ static void Run(const ADataType* __restrict__ p_a_grid,
                               const B0DataType* __restrict__ p_b0_grid,
                               D0sGridPointer p_d0s_grid,
                               const B1DataType* __restrict__ p_b1_grid,
                               D1sGridPointer p_d1s_grid,
                               E1DataType* __restrict__ p_e1_grid,
                               void* __restrict__ p_shared,
                               const AGridDesc& a_grid_desc,
                               const B0GridDesc& b0_grid_desc,
                               const D0sGridDesc& d0s_grid_desc,
                               const B1GridDesc& b1_grid_desc,
                               const D1sGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock&
                                   d1s_grid_desc_mblock_mperblock_nblock_nperblock,
                               const E1GridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock&
                                   e1_grid_desc_mblock_mperblock_nblock_nperblock,
                               const AElementwiseOperation& a_element_op,
                               const B0ElementwiseOperation& b0_element_op,
                               const AccElementwiseOperation& acc_element_op,
                               const B1ElementwiseOperation& b1_element_op,
                               const CDEElementwiseOperation& c_element_op,
                               const Block2ETileMap& block_2_etile_map)
    {
        // clang-format off
/*******************************************************************************/
// Memory buffer zone.
        const auto d0s_griddesc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs =
            MakeD0sGridDescriptor_MRepeat_MWave_MThreadPerSubGroup_NRepeat_NWave_NSubGroup_NAccVgprs(d0s_grid_desc);
        const auto a_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_grid, a_grid_desc.GetElementSpaceSize());
        const auto b0_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b0_grid, b0_grid_desc.GetElementSpaceSize());
        const auto b1_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b1_grid, b1_grid_desc.GetElementSpaceSize());
        auto e1_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_e1_grid, e1_grid_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());
        const auto d0s_grid_buf = generate_tuple(
            [&](auto i) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_d0s_grid[i],
                    d0s_griddesc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs[i].GetElementSpaceSize());
            },
            Number<NumD0Tensor>{});
        const auto d1s_grid_buf = generate_tuple(
            [&](auto i) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_d1s_grid[i],
                    d1s_grid_desc_mblock_mperblock_nblock_nperblock[i].GetElementSpaceSize());
            },
            Number<NumD1Tensor>{});

/*******************************************************************************/
// BlockIdx.x -> [BlockId.m, BlockId.n]
        const auto block_work_idx = block_2_etile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));
        if(!block_2_etile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(e1_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I0),
                          e1_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I2))))
        { return; }

        // Store BlockId into SGPR
        const index_t m_block_data_idx_on_grid = __builtin_amdgcn_readfirstlane(block_work_idx[I0] * MPerBlock);
        const index_t n_block_data_idx_on_grid = __builtin_amdgcn_readfirstlane(block_work_idx[I1] * NPerBlock);

/*******************************************************************************/
// set up Gemm0
/*******************************************************************************/

/*******************************************************************************/
// BlockLevel, A/B Matrix ThreadMapping in LDS, As Destination of BlockWise_Copy
        constexpr auto a_block_desc  = MakeABlockDescriptor();
        constexpr auto b0_block_desc = MakeB0BlockDescriptor();

        auto a_block_trait = [&](){
            // A matrix blockwise copy
            constexpr auto AK0PerBlock = KPerBlock/ AK1;
            auto a_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                static_cast<ADataType*>(p_shared) + SharedMemTrait::a_block_space_offset, 
                SharedMemTrait::a_block_space_size_aligned);

            auto a_blockwise_copy =
                ThreadGroupTensorSliceTransfer_v4r1< ThisThreadBlock,
/* typename SrcElementwiseOperation,              */     AElementwiseOperation,
/* typename DstElementwiseOperation,              */     ck::tensor_operation::element_wise::PassThrough,
/* InMemoryDataOperationEnum DstInMemOp,          */     InMemoryDataOperationEnum::Set,
/* typename BlockSliceLengths,                    */     Sequence<AK0PerBlock, MPerBlock, AK1>,
/* typename ThreadClusterLengths,                 */     ABlockTransferThreadClusterLengths_K0_M_K1,
/* typename ThreadClusterArrangeOrder,            */     ABlockTransferThreadClusterArrangeOrder,
/* typename SrcData,                              */     ADataType,
/* typename DstData,                              */     ADataType,
/* typename SrcDesc,                              */     decltype(a_grid_desc),
/* typename DstDesc,                              */     decltype(a_block_desc),
/* typename SrcDimAccessOrder,                    */     ABlockTransferSrcAccessOrder,
/* typename DstDimAccessOrder,                    */     Sequence<0, 1, 2>,
/* index_t SrcVectorDim,                          */     ABlockTransferSrcVectorDim,
/* index_t DstVectorDim,                          */     2,
/* index_t SrcScalarPerVector,                    */     ABlockTransferSrcScalarPerVector,
/* index_t DstScalarPerVector,                    */     ABlockTransferDstScalarPerVector_K1,
/* index_t SrcScalarStrideInVector,               */     1,
/* index_t DstScalarStrideInVector,               */     1,
/* bool ThreadTransferSrcResetCoordinateAfterRun, */     AThreadTransferSrcResetCoordinateAfterRun,
/* bool ThreadTransferDstResetCoordinateAfterRun, */     true,
                                                        BlockwiseGemmPipe::GlobalBufferNum>( 
            a_grid_desc,
            make_multi_index(0, m_block_data_idx_on_grid, 0),
            a_element_op,
            a_block_desc,
            make_multi_index(0, 0, 0),
            ck::tensor_operation::element_wise::PassThrough{});

            return make_tuple(a_block_buf, a_blockwise_copy);
        };
        
        auto b0_block_trait = [&](){
            auto b0_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                static_cast<B0DataType*>(p_shared) + SharedMemTrait::b0_block_space_offset, 
                SharedMemTrait::b0_block_space_size_aligned);

            auto b0_blockwise_copy =
                ThreadGroupTensorSliceTransfer_v4r1<ThisThreadBlock,
                                            B0ElementwiseOperation,
                                            ck::tensor_operation::element_wise::PassThrough,
                                            InMemoryDataOperationEnum::Set,
                                            Sequence<BK0, LPerBlock, BK1>,
                                            B0BlockTransferThreadClusterLengths_K0_L_K1,
                                            B0BlockTransferThreadClusterArrangeOrder,
                                            B0DataType,
                                            B0DataType,
                                            decltype(b0_grid_desc),
                                            decltype(b0_block_desc),
                                            B0BlockTransferSrcAccessOrder,
                                            Sequence<0, 1, 2>,
                                            B0BlockTransferSrcVectorDim,
                                            2,
                                            B0BlockTransferSrcScalarPerVector,
                                            B0BlockTransferDstScalarPerVector_K1,
                                            1,
                                            1,
                                            B0ThreadTransferSrcResetCoordinateAfterRun,
                                            true,
                                            BlockwiseGemmPipe::GlobalBufferNum>(
            b0_grid_desc,
            make_multi_index(0, 0, 0),
            b0_element_op,
            b0_block_desc,
            make_multi_index(0, 0, 0),
            ck::tensor_operation::element_wise::PassThrough{});
            
            return make_tuple(b0_block_buf, b0_blockwise_copy);
        };

        auto a_block_buf       = a_block_trait()[I0];
        auto a_blockwise_copy  = a_block_trait()[I1];
        
        auto b0_block_buf       = b0_block_trait()[I0];
        auto b0_blockwise_copy  = b0_block_trait()[I1];

/*******************************************************************************/
        // Gemm0
        // Blockwise GEMM0 pipeline
        static_assert(std::is_default_constructible_v<BlockwiseGemmPipe>);
        auto blockwise_gemm0_pipeline = BlockwiseGemmPipe{};
        auto acc0_thread_buf          = blockwise_gemm0_pipeline.GetCThreadBuffer();

        // Note that we are using the transposeC version of GetCThreadDescriptor.
        constexpr auto acc0_thread_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs = 
            blockwise_gemm0_pipeline.GetCThreadDescriptor_MRepeat_MWave_MThreadPerSubGroup_NRepeat_NWave_NSubGroup_NAccVgprs();
        
        constexpr auto mrepeat            = acc0_thread_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs.GetLength(I0);
        constexpr auto mwave              = acc0_thread_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs.GetLength(I1);
        constexpr auto mthreadpersubgroup = acc0_thread_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs.GetLength(I2);
        constexpr auto lrepeat            = acc0_thread_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs.GetLength(I3);
        constexpr auto lwave              = acc0_thread_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs.GetLength(I4);
        constexpr auto lsubgroup          = acc0_thread_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs.GetLength(I5);
        constexpr auto laccvgprs          = acc0_thread_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs.GetLength(I6);

        // d0 matrix threadwise copy
        constexpr auto d0_thread_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs =
            make_naive_tensor_descriptor_packed(make_tuple(
                                                           I1,              // MBlockId
                                                           I1,              // NBlockID
                                                           mrepeat,
                                                           mwave,
                                                           mthreadpersubgroup,
                                                           lrepeat,
                                                           lwave,
                                                           lsubgroup,
                                                           laccvgprs));

        auto d0s_thread_buf = generate_tuple(
            [&](auto i) {
                using D0iDataType = remove_cvref_t<tuple_element_t<i.value, D0sDataType>>;
                return StaticBuffer<
                    AddressSpaceEnum::Vgpr,
                    D0iDataType,
                    d0_thread_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs.GetElementSpaceSize(),
                    true>{};
            },
            Number<NumD0Tensor>{});

        const auto wave_id     = GetGemm0WaveIdx(); // I0: MWaves, I1: LWaves, I2: WaveSize
        const auto wave_m_n_id = GetGemm0WaveMNIdx(wave_id[I2]); // I0: WaveSize / LPerWmma, I1: LPerWmma

        static_assert(CDE0BlockTransferSrcScalarPerVector <= laccvgprs,
                      "vector load must be not greater than n4");
        static_assert(laccvgprs % CDE0BlockTransferSrcScalarPerVector == 0);

        auto d0s_threadwise_copy = generate_tuple(
            [&](auto i) {
                using D0iDataType = remove_cvref_t<tuple_element_t<i.value, D0sDataType>>;
                return ThreadwiseTensorSliceTransfer_v2<
                    D0iDataType,
                    D0iDataType,
                    decltype(d0s_griddesc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs[i]),
                    decltype(d0_thread_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs),
                    Sequence<I1, // MBlockId
                             I1, // NBlockID
                             mrepeat,
                             mwave,
                             mthreadpersubgroup,
                             lrepeat,
                             lwave,
                             lsubgroup,
                             laccvgprs>,
                    Sequence<0, 1, 2, 3, 4, 5, 6, 7, 8>,
                    8, // NOTE: XDL has this exposed as CDE0BlockTransferSrcVectorDim. 
                       // But as the grid descriptor is built internally, the parameter doesn't really make sense to configure per instance
                    CDE0BlockTransferSrcScalarPerVector,
                    1,
                    false>(d0s_griddesc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs[i],
                           make_multi_index(block_work_idx[I0],  // MBlockId
                                            0,                      // NBlockId 
                                            0,                      // mrepeat
                                            wave_id[I0],            // mwave
                                            wave_m_n_id[I1],        // mthreadpersubgroup
                                            0,                      // nrepeat
                                            wave_id[I1],            // nwave
                                            wave_m_n_id[I0],        // nsubgroup
                                            0));                    // register number
            },
            Number<NumD0Tensor>{});
        
        constexpr auto acc0_thread_desc_l0perblock_mperblock_l1 = transform_tensor_descriptor(
            acc0_thread_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs,
            make_tuple(make_merge_transform_v3_division_mod(make_tuple(lrepeat, lwave, lsubgroup)),
                       make_merge_transform_v3_division_mod(make_tuple(mrepeat, mwave, mthreadpersubgroup)),
                       make_pass_through_transform(laccvgprs)),
            make_tuple(Sequence<3, 4, 5>{}, Sequence<0, 1, 2>{}, Sequence<6>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

/*******************************************************************************/        
        // Shift Per SUB_K
        constexpr auto a_block_slice_copy_step = MakeABlockSliceCopyStep();
        constexpr auto b0_block_slice_copy_step = MakeB0BlockSliceCopyStep();

        const auto a_block_reset_copy_step = [&](){
            return make_multi_index(-a_grid_desc.GetLength(I0), 0, 0);
        }();

        const auto b0_block_reset_copy_step = [&](){
            return make_multi_index(-b0_grid_desc.GetLength(I0), LPerBlock, 0);
        }();

        const auto K = [&](){
            return a_grid_desc.GetLength(I0) * a_grid_desc.GetLength(I2);
        }();

        const index_t KBlockMainLoop = __builtin_amdgcn_readfirstlane(K / KPerBlock);
        
/*******************************************************************************/
// set up Gemm1
/*******************************************************************************/
        // Acc0 thread buffer -> A1 thread buffer -> blockwise gemm
        // A1 matrix in VGPR
        constexpr auto A1ThreadSlice_L0PerBlock_MPerBlock_L1 = make_tuple(
            Number<AL0 * AL1 / laccvgprs>{}, 
            Number<mrepeat * mwave * mthreadpersubgroup>{}, 
            Number<laccvgprs>{});

        constexpr auto A1ThreadSliceL0PerBlock  = A1ThreadSlice_L0PerBlock_MPerBlock_L1[I0];
        constexpr auto A1ThreadSliceMPerBlock   = A1ThreadSlice_L0PerBlock_MPerBlock_L1[I1];
        constexpr auto A1ThreadSliceL1          = A1ThreadSlice_L0PerBlock_MPerBlock_L1[I2];

        constexpr auto a1_thread_desc_l0perblock_mperblock_l1 = make_naive_tensor_descriptor(
            make_tuple(A1ThreadSliceL0PerBlock, A1ThreadSliceMPerBlock, A1ThreadSliceL1),
            make_tuple(A1ThreadSliceMPerBlock * A1ThreadSliceL1, A1ThreadSliceL1, I1));

        // A1 matrix blockwise copy
        auto a1_blockwise_copy = ThreadwiseTensorSliceTransfer_StaticToStatic<
            Acc0DataType,
            ADataType,
            decltype(acc0_thread_desc_l0perblock_mperblock_l1),
            decltype(a1_thread_desc_l0perblock_mperblock_l1),
            tensor_operation::element_wise::PassThrough,
            Sequence<A1ThreadSliceL0PerBlock, A1ThreadSliceMPerBlock, A1ThreadSliceL1>,
            Sequence<0, 1, 2>,
            2,
            laccvgprs>{tensor_operation::element_wise::PassThrough{}};
   
        auto a1_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ADataType>(
            a1_thread_desc_l0perblock_mperblock_l1.GetElementSpaceSize());       
            
        constexpr auto b1_block_desc = MakeB1BlockDescriptor();

        auto b1_block_trait = [&](){
            auto b1_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                static_cast<B1DataType*>(p_shared) + SharedMemTrait::b1_block_space_offset, 
                SharedMemTrait::b1_block_space_size_aligned);

            auto b1_blockwise_copy = ThreadGroupTensorSliceTransfer_v4r1<
                                                    ThisThreadBlock,
/* typename SrcElementwiseOperation,              */ B1ElementwiseOperation,
/* typename DstElementwiseOperation,              */ tensor_operation::element_wise::PassThrough,
/* InMemoryDataOperationEnum DstInMemOp,          */ InMemoryDataOperationEnum::Set,
/* typename BlockSliceLengths,                    */ Sequence<BL0, NPerBlock, BL1>,
/* typename ThreadClusterLengths,                 */ B1BlockTransferThreadClusterLengths_L0_N_L1,
/* typename ThreadClusterArrangeOrder,            */ B1BlockTransferThreadClusterArrangeOrder,
/* typename SrcData,                              */ B1DataType,
/* typename DstData,                              */ B1DataType,
/* typename SrcDesc,                              */ decltype(b1_grid_desc),
/* typename DstDesc,                              */ decltype(b1_block_desc),
/* typename SrcDimAccessOrder,                    */ B1BlockTransferSrcAccessOrder,
/* typename DstDimAccessOrder,                    */ Sequence<1, 0, 2>,
/* index_t SrcVectorDim,                          */ B1BlockTransferSrcVectorDim,
/* index_t DstVectorDim,                          */ 2,
/* index_t SrcScalarPerVector,                    */ B1BlockTransferSrcScalarPerVector,
/* index_t DstScalarPerVector,                    */ B1BlockTransferDstScalarPerVector_L1,
/* index_t SrcScalarStrideInVector,               */ 1,
/* index_t DstScalarStrideInVector,               */ 1,
/* bool ThreadTransferSrcResetCoordinateAfterRun, */ B1ThreadTransferSrcResetCoordinateAfterRun,
/* bool ThreadTransferDstResetCoordinateAfterRun, */ true, // DstResetCoord
                                                     1>( // Used to be NumGemmKPrefetchStage, never tested / used for != 1
                b1_grid_desc,
                make_multi_index(0, n_block_data_idx_on_grid, 0),
                b1_element_op,
                b1_block_desc,
                make_multi_index(0, 0, 0),
                tensor_operation::element_wise::PassThrough{});
                    
            return make_tuple(b1_block_buf, b1_blockwise_copy);
        };

        auto b1_block_buf       = b1_block_trait()[I0];
        auto b1_blockwise_copy  = b1_block_trait()[I1];

        constexpr auto b1_block_slice_copy_step = MakeB1BlockSliceCopyStep();

        auto blockwise_gemm1 =
            BlockwiseGemmWMMA<BlockSize,
                              ADataType,
                              B1DataType,
                              Acc1DataType,
                              decltype(MakeA1WaveDescriptor_L0_M0_M1_M2_L1(a1_thread_desc_l0perblock_mperblock_l1)),
                              decltype(MakeB1WaveDescriptor(b1_block_desc)),
                              MPerBlock,
                              NPerBlock,
                              LTilePerBlock,
                              MPerWmma,
                              NPerWmma,
                              MRepeat,
                              NRepeat,
                              KPack,
                              false, // Acc1EnableLds
                              true,  // B1EnableLds
                              true>{make_tuple(0, 0, 0, 0, 0, 0)};

        auto acc1_thread_buf = blockwise_gemm1.GetCThreadBuffer();

        const auto L = [&](){
            return b0_grid_desc.GetLength(I1);
        }();

        const index_t num_gemm1_l_block_outer_loop = L / LPerBlock;
        constexpr index_t num_gemm1_l_block_inner_loop = LPerBlock / LTilePerBlock;

        // Initialize C
        StaticBuffer<AddressSpaceEnum::Vgpr, Acc1DataType, acc1_thread_buf.Size(), true> c_thread_buf;
        c_thread_buf.Clear();

        // Empty BScale struct for the blockwise pipeline.
        using ABScale       = typename BlockwiseGemmPipe::Empty;
        auto a_scale_struct = ABScale{};
        auto b_scale_struct = ABScale{};

/*******************************************************************************/
        // 
        // Kernel Main Stage
        //
        index_t gemm1_l_block_outer_index = 0;
        // Outer loop, along GEMM_L
        // Inner loop, along GEMM_K
        do {
            blockwise_gemm0_pipeline.template Run<HasMainKBlockLoop, TailNum>(a_grid_desc, 
                                                                              a_block_desc, 
                                                                              a_blockwise_copy, 
                                                                              a_grid_buf,
                                                                              a_block_buf, 
                                                                              a_block_slice_copy_step,
                                                                              b0_grid_desc,
                                                                              b0_block_desc,
                                                                              b0_blockwise_copy,
                                                                              b0_grid_buf,
                                                                              b0_block_buf,
                                                                              b0_block_slice_copy_step,
                                                                              acc0_thread_buf,
                                                                              a_scale_struct,
                                                                              b_scale_struct,
                                                                              KBlockMainLoop,
                                                                              1); // num_k_block_per_scale
            // multiple d
            if constexpr(NumD0Tensor)
            {
                constexpr auto d0s_thread_buf_size = d0_thread_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs.GetElementSpaceSize();

                static_for<0, NumD0Tensor, 1>{}([&](auto i) {
                    d0s_threadwise_copy(i).Run(d0s_griddesc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs[i],
                                               d0s_grid_buf[i],
                                               d0_thread_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs,
                                               make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0),
                                               d0s_thread_buf(i));
                });
                static_for<0, d0s_thread_buf_size, 1>{}([&](auto i) {
                    // get reference to src data
                    const auto src_data_refs = generate_tie(
                        // return type should be lvalue
                        [&](auto iSrc) -> const auto& { return d0s_thread_buf[iSrc][i]; },
                        Number<NumD0Tensor>{});

                    // get reference to dst data
                    auto dst_data_refs = generate_tie(
                        // return type should be lvalue
                        [&](auto) -> auto& { return acc0_thread_buf(i); },
                        Number<2>{});

                    unpack2(acc_element_op, dst_data_refs, src_data_refs);
                });
                static_for<0, NumD0Tensor, 1>{}([&](auto i) {
                    d0s_threadwise_copy(i).MoveSrcSliceWindow(
                        d0s_griddesc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs[i],
                        make_multi_index(0, 1, 0, 0, 0, 0, 0, 0, 0));
                });
            }
            else
            {
                static_for<0, acc0_thread_buf.Size(), 1>{}(
                        [&](auto i) { acc_element_op(acc0_thread_buf(i), acc0_thread_buf[i]); });
            }

            block_sync_lds();

            // gemm1
            {
                // TODO: explore using dynamic buffer for a1 thread buffer
                // For a1_blockwise_copy, the goal is to satisfy pipeline requirements RunRead(),
                // RunWrite(), and MoveSliceWindow(). But it is impossible to implement given that
                // the A1 source buffer is static buffer holding the output of first GEMM and
                // requires constexpr offset by design. Therefore, we pass tensor coordinate offset
                // explicitly in Run() below.

                // Initialize acc1
                acc1_thread_buf.Clear();

                // preload data into LDS
                b1_blockwise_copy.RunRead(b1_grid_desc, b1_grid_buf);

                b1_blockwise_copy.MoveSrcSliceWindow(b1_grid_desc,
                                                     b1_block_slice_copy_step);

                block_sync_lds(); // wait for reduction LDS read

                b1_blockwise_copy.RunWrite(b1_block_desc, b1_block_buf);

                // main body
                if constexpr(num_gemm1_l_block_inner_loop > 1)
                {
                    static_for<0, num_gemm1_l_block_inner_loop - 1, 1>{}([&](auto i) {
                        // Data cast from Acc0DataType to ADataType happens here
                        a1_blockwise_copy.Run(acc0_thread_desc_l0perblock_mperblock_l1,
                                              make_tuple(Number<i * A1ThreadSliceL0PerBlock>{}, I0, I0),
                                              acc0_thread_buf,
                                              a1_thread_desc_l0perblock_mperblock_l1,
                                              make_tuple(I0, I0, I0),
                                              a1_thread_buf);

                        b1_blockwise_copy.RunRead(b1_grid_desc, b1_grid_buf);

                        block_sync_lds();

                        blockwise_gemm1.Run(a1_thread_buf, b1_block_buf, acc1_thread_buf);

                        block_sync_lds();

                        b1_blockwise_copy.MoveSrcSliceWindow(b1_grid_desc,
                                                             b1_block_slice_copy_step);

                        b1_blockwise_copy.RunWrite(b1_block_desc, b1_block_buf);
                    });
                }
                // tail
                {
                    a1_blockwise_copy.Run(
                        acc0_thread_desc_l0perblock_mperblock_l1,
                        make_tuple(
                            Number<(num_gemm1_l_block_inner_loop - 1) * A1ThreadSliceL0PerBlock>{}, I0, I0),
                        acc0_thread_buf,
                        a1_thread_desc_l0perblock_mperblock_l1,
                        make_tuple(I0, I0, I0),
                        a1_thread_buf);

                    block_sync_lds();
            
                    blockwise_gemm1.Run(a1_thread_buf, b1_block_buf, acc1_thread_buf);
                }
            } // end gemm1

            constexpr auto c1_thread_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs =
                blockwise_gemm1.GetCThreadDescriptor_MRepeat_MWave_MThreadPerSubGroup_NRepeat_NWave_NSubGroup_NAccVgprs();
            constexpr auto c_mrepeat            = c1_thread_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs.GetLength(I0);
            constexpr auto c_mwave              = c1_thread_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs.GetLength(I1);
            constexpr auto c_mthreadpersubgroup = c1_thread_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs.GetLength(I2);
            constexpr auto c_nrepeat            = c1_thread_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs.GetLength(I3);
            constexpr auto c_nwave              = c1_thread_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs.GetLength(I4);
            constexpr auto c_nsubgroup          = c1_thread_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs.GetLength(I5);
            constexpr auto c_naccvgprs          = c1_thread_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs.GetLength(I6);
            
            constexpr auto c_thread_slice_desc_m_n = make_naive_tensor_descriptor_packed(
                make_tuple(c_mrepeat * c_mwave * c_mthreadpersubgroup, 
                           c_nrepeat * c_nwave * c_nsubgroup * c_naccvgprs));
            constexpr auto c_thread_buf_slice_m = c_thread_slice_desc_m_n.GetLength(I0);
            constexpr auto c_thread_buf_slice_n = c_thread_slice_desc_m_n.GetLength(I1);

            static_ford<Sequence<c_thread_buf_slice_m, c_thread_buf_slice_n>>{}([&](auto ii) {
                constexpr auto iM = Number<ii[Number<0>{}]>{};
                constexpr auto iN = Number<ii[Number<1>{}]>{};
                auto I = Number<c_thread_slice_desc_m_n.CalculateOffset(make_tuple(iM, iN))>{};
                Acc1DataType acc1  = acc1_thread_buf[I]; // P*V
                Acc1DataType c     = c_thread_buf[I];    // O
                Acc1DataType c_new = c + acc1; // Simply add results since we are no longer using softmax.

                c_thread_buf(I) = c_new; // O_new
            });

            a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc,
                                                a_block_reset_copy_step); // rewind K
            b0_blockwise_copy.MoveSrcSliceWindow(b0_grid_desc,
                                                b0_block_reset_copy_step); // rewind K and step N

            block_sync_lds(); // wait for gemm1 LDS read
        }while(++gemm1_l_block_outer_index < num_gemm1_l_block_outer_loop);
/*******************************************************************************/
        // write out to C, implement shuffle
        {
            constexpr auto c1_thread_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs =  
            blockwise_gemm1.GetCThreadDescriptor_MRepeat_MWave_MThreadPerSubGroup_NRepeat_NWave_NSubGroup_NAccVgprs();

            // This API Provide All dimension (size) you need
            constexpr auto c1_block_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs_tmp =
                blockwise_gemm1.GetCBlockDescriptor_MRepeat_MWave_MThreadPerSubGroup_NRepeat_NWave_NSubGroup_NAccVgprs();

            constexpr auto MWave              = c1_block_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs_tmp.GetLength(I1);
            constexpr auto MThreadPerSubGroup = c1_block_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs_tmp.GetLength(I2);
            constexpr auto NWave              = c1_block_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs_tmp.GetLength(I4);
            constexpr auto NSubGroup          = c1_block_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs_tmp.GetLength(I5);
            constexpr auto NAccVgprs          = c1_block_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs_tmp.GetLength(I6);

            // LDS descriptor, shuffle and write out in MRepeat x NRepeat times
            constexpr auto c1_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat =
                GetCShuffleBlockDescriptor_MShRepeat_MPerShRepeat_NShRepeat_NPerShRepeat();

            auto c1_shuffle_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
                static_cast<CShuffleDataType*>(p_shared),
                c1_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat.GetElementSpaceSize());

            constexpr auto c_block_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs = transform_tensor_descriptor(
                c1_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat,
                make_tuple(
                    make_freeze_transform(I0),
                    make_unmerge_transform(make_tuple(
                        Number<CShuffleMRepeatPerShuffle>{}, // MRepeat per shuffle repeat
                        MWave,                               // MWave
                        MThreadPerSubGroup                   // MThreadPerSubGroup = MPerWmma
                        )),
                    make_freeze_transform(I0),
                    make_unmerge_transform(make_tuple(
                        Number<CShuffleNRepeatPerShuffle>{}, // NRepeat per shuffle repeat
                        NWave,                               // NWave
                        NSubGroup,
                        NAccVgprs))),                        // NSubGroup * NAccVgprs = NPerWmma
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<>{}, Sequence<0, 1, 2>{}, Sequence<>{}, Sequence<3, 4, 5, 6>{}));

            // calculate origin of thread output tensor on global memory
            //     blockwise GEMM c matrix starting index
            const auto c_thread_mtx_on_block = blockwise_gemm1.CalculateCThreadOriginDataIndex(I0, I0);

            const index_t m_thread_data_on_block = c_thread_mtx_on_block[I0];
            const index_t n_thread_data_on_block = c_thread_mtx_on_block[I1];

            const auto m_thread_data_on_block_to_mrepeat_mwave_mthreadpersubgroup_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(MRepeat, MWave, MThreadPerSubGroup))),
                make_tuple(Sequence<0, 1, 2>{}),
                make_tuple(Sequence<0>{}));

            const auto n_thread_data_on_block_to_nrepeat_nwave_nsubgroup_naccvgprs_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(NRepeat, NWave, NSubGroup, NAccVgprs))),
                make_tuple(Sequence<0, 1, 2, 3>{}),
                make_tuple(Sequence<0>{}));
            
            const auto m_thread_data_on_block_idx = m_thread_data_on_block_to_mrepeat_mwave_mthreadpersubgroup_adaptor.CalculateBottomIndex(
                make_multi_index(m_thread_data_on_block));
            
            const auto n_thread_data_on_block_idx = n_thread_data_on_block_to_nrepeat_nwave_nsubgroup_naccvgprs_adaptor.CalculateBottomIndex(
                make_multi_index(n_thread_data_on_block));

            // shuffle: threadwise copy C from VGPR to LDS
            auto c1_thread_copy_vgpr_to_lds =
                ThreadwiseTensorSliceTransfer_v1r3<Acc1DataType,
                                                   CShuffleDataType,
                                                   decltype(c1_thread_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs),
                                                   decltype(c_block_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs),
                                                   ck::tensor_operation::element_wise::PassThrough,
                                                   Sequence<CShuffleMRepeatPerShuffle,
                                                            I1,
                                                            I1,
                                                            CShuffleNRepeatPerShuffle,
                                                            I1,
                                                            I1,
                                                            NAccVgprs>,
                                                   Sequence<0, 1, 2, 3, 4, 5, 6>,
                                                   6,
                                                   8, // vector write pixel
                                                   InMemoryDataOperationEnum::Set,
                                                   1,
                                                   true>{
                    c_block_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs,
                    make_multi_index(0,
                                     m_thread_data_on_block_idx[I1],
                                     m_thread_data_on_block_idx[I2],
                                     0,
                                     n_thread_data_on_block_idx[I1],
                                     n_thread_data_on_block_idx[I2],
                                     n_thread_data_on_block_idx[I3]),
                    ck::tensor_operation::element_wise::PassThrough{}};
                    
                    
            // tuple of reference to C/Ds tensor descriptors
            const auto e1_d1s_desc_refs = concat_tuple_of_reference(
                tie(c1_shuffle_block_desc_mshrepeat_mpershrepeat_nshrepeat_npershrepeat),
                generate_tie([&](auto i) -> const auto& // return type should be reference
                             { return d1s_grid_desc_mblock_mperblock_nblock_nperblock[i]; },
                             Number<NumD1Tensor>{}));

            // tuple of reference to C/Ds tensor descriptors
            const auto c1_d1s_buf_refs = concat_tuple_of_reference(
                tie(c1_shuffle_block_buf),
                generate_tie([&](auto i) -> const auto& // return type should be reference
                             { return d1s_grid_buf[i]; },
                             Number<NumD1Tensor>{}));

            // tuple of starting index of C/Ds blockwise copy
            const auto idx_c1_d1s_block_begin = container_concat(
                make_tuple(make_multi_index(0, 0, 0, 0)),
                generate_tuple(
                    [&](auto) {
                        return make_multi_index(block_work_idx[I0], 0, block_work_idx[I1], 0);
                    },
                    Number<NumD1Tensor>{}));


            // shuffle: blockwise copy C from LDS to global
            auto cde1_shuffle_block_copy_lds_to_global = ThreadGroupTensorSliceTransfer_v7<
                ThisThreadBlock,
                decltype(container_concat(make_tuple(CShuffleDataType{}), D1sDataType{})),
                Tuple<E1DataType>,
                decltype(e1_d1s_desc_refs),
                decltype(tie(e1_grid_desc_mblock_mperblock_nblock_nperblock)),
                CDEElementwiseOperation,
                Sequence<static_cast<index_t>(CGlobalMemoryDataOperation)>, // FIXME: make Sequence
                                                                             // support arbitray
                                                                             // type
                
                Sequence<1,
                         CShuffleMRepeatPerShuffle * MWave * MPerWmma,
                         1,
                         CShuffleNRepeatPerShuffle * NWave * NPerWmma>, // BlockSliceLengths,
                CDE1ShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
                Sequence<0, 1, 2, 3>, // typename DimAccessOrder,
                3,                    // index_t VectorDim,
                CDE1ShuffleBlockTransferScalarPerVector_NPerBlock,
                sequence_merge_t<
                    Sequence<true>,
                    uniform_sequence_gen_t<NumD1Tensor,
                                           false>>, // ThreadTransferSrcResetCoordinateAfterRunFlags
                Sequence<false>>                    // ThreadTransferDstResetCoordinateAfterRunFlags
                {e1_d1s_desc_refs,
                 idx_c1_d1s_block_begin,
                 tie(e1_grid_desc_mblock_mperblock_nblock_nperblock),
                 make_tuple(make_multi_index(block_work_idx[I0], 0, block_work_idx[I1], 0)),
                 c_element_op};


            // space filling curve for local reg & global memory
            // space filling curve for threadwise C in VGPR
            constexpr auto sfc_c1_vgpr =
                SpaceFillingCurve<Sequence<MRepeat, 1, 1, NRepeat, 1, 1, NAccVgprs>,
                                  Sequence<0, 1, 2, 3, 4, 5, 6>,
                                  Sequence<CShuffleMRepeatPerShuffle,
                                           1,
                                           1,
                                           CShuffleNRepeatPerShuffle,
                                           1,
                                           1,
                                           NAccVgprs>>{};

            // space filling curve for shuffled blockwise C in global mem
            constexpr auto sfc_e1_global =
                SpaceFillingCurve<Sequence<1, MPerBlock, 1, NPerBlock>,
                                  Sequence<0, 2, 1, 3>,
                                  Sequence<1,
                                           CShuffleMRepeatPerShuffle * MWave * MPerWmma,
                                           1,
                                           CShuffleNRepeatPerShuffle * NWave * NPerWmma>>{};

            constexpr index_t num_access = sfc_c1_vgpr.GetNumOfAccess();

            static_assert(num_access == sfc_e1_global.GetNumOfAccess(), "wrong!");

            static_for<0, num_access, 1>{}([&](auto access_id) {
                // make sure it's safe to write to LDS
                block_sync_lds();

                // each thread write its data from VGPR to LDS
                c1_thread_copy_vgpr_to_lds.Run(c1_thread_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs,
                                              sfc_c1_vgpr.GetIndexTupleOfNumber(access_id),
                                              c_thread_buf,
                                              c_block_desc_mrepeat_mwave_mthreadpersubgroup_nrepeat_nwave_nsubgroup_naccvgprs,
                                              c1_shuffle_block_buf);

                // make sure it's safe to read from LDS
                block_sync_lds();

                // each block copy its data from LDS to global
                cde1_shuffle_block_copy_lds_to_global.Run(
                    e1_d1s_desc_refs,
                    c1_d1s_buf_refs,
                    tie(e1_grid_desc_mblock_mperblock_nblock_nperblock),
                    tie(e1_grid_buf));

                if constexpr(access_id < num_access - 1)
                {
                    constexpr auto e1_global_step = sfc_e1_global.GetForwardStep(access_id);

                    // move on D1s
                    static_for<0, NumD1Tensor, 1>{}([&](auto i) {
                        cde1_shuffle_block_copy_lds_to_global.MoveSrcSliceWindow(
                            e1_d1s_desc_refs, i + I1, e1_global_step);
                    });

                    // move on C
                    cde1_shuffle_block_copy_lds_to_global.MoveDstSliceWindow(
                        tie(e1_grid_desc_mblock_mperblock_nblock_nperblock), I0, e1_global_step);
                }
            });
        }
        // clang-format on
    }
};

} // namespace ck
