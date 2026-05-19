// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_eight_waves_base.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp"
#include "ck_tile/ops/gemm_mx/pipeline/gemm_pipeline_ag_bg_cr_comp_async_eight_waves_policy.hpp"

namespace ck_tile {

/**
 * @brief Compute optimized pipeline version async for 8 waves
 *
 * This pipeline introduces asynchronous load from global memory to LDS,
 * skipping the intermediate loading into pipeline registers.
 */
template <typename Problem, typename Policy = MXGemmPipelineAgBgCrCompAsyncEightWavesPolicy>
struct MXGemmPipelineAgBgCrCompAsyncEightWaves : public BaseGemmPipelineAgBgCrCompV3<Problem>
{
    using Base             = BaseGemmPipelineAgBgCrCompV3<Problem>;
    using PipelineImplBase = GemmPipelineAgBgCrEightWavesImplBase<Problem, Policy>;

    using AsDataType     = remove_cvref_t<typename Problem::AsDataTypeTuple>;
    using BsDataType     = remove_cvref_t<typename Problem::BsDataTypeTuple>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    using AsLayout = remove_cvref_t<typename Problem::AsLayoutTuple>;
    using BsLayout = remove_cvref_t<typename Problem::BsLayoutTuple>;
    using CLayout  = remove_cvref_t<typename Problem::CLayout>;

    using AElementWise = remove_cvref_t<typename Problem::AElementWise>;
    using BElementWise = remove_cvref_t<typename Problem::BElementWise>;

    using ALayout = remove_cvref_t<std::tuple_element_t<0, AsLayout>>;
    using BLayout = remove_cvref_t<std::tuple_element_t<0, BsLayout>>;

    using ADataType = remove_cvref_t<std::tuple_element_t<0, AsDataType>>;
    using BDataType = remove_cvref_t<std::tuple_element_t<0, BsDataType>>;

    static_assert(!std::is_same_v<BDataType, pk_int4_t>, "Not implemented");

    static constexpr index_t APackedSize = ck_tile::numeric_traits<ADataType>::PackedSize;
    static constexpr index_t BPackedSize = ck_tile::numeric_traits<BDataType>::PackedSize;

    using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;
    using WarpGemm  = typename BlockGemm::WarpGemm;

    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};

    static constexpr index_t BlockSize = Problem::kBlockSize;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    static constexpr index_t MWarps = BlockGemmShape::BlockWarps::at(I0);
    static constexpr index_t NWarps = BlockGemmShape::BlockWarps::at(I1);
    static constexpr index_t KWarps = BlockGemmShape::BlockWarps::at(I2);

    static constexpr index_t kflatKPerBlock = BlockGemmShape::flatKPerBlock;

    static constexpr index_t MIterPerWarp = MPerBlock / (MWarps * WarpGemm::kM);
    static constexpr index_t NIterPerWarp = NPerBlock / (NWarps * WarpGemm::kN);
    static constexpr index_t KIterPerWarp = KPerBlock / (KWarps * WarpGemm::kK);

    static constexpr bool Async = true;

    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeA()
    {
        return Policy::template GetVectorSizeA<Problem>();
    }
    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeB()
    {
        return Policy::template GetVectorSizeB<Problem>();
    }

    static constexpr index_t GetSmemPackA() { return Policy::template GetSmemPackA<Problem>(); }
    static constexpr index_t GetSmemPackB() { return Policy::template GetSmemPackB<Problem>(); }

    static constexpr index_t NumWaveGroups = Problem::NumWaveGroups;
    static constexpr index_t Preshuffle    = Problem::Preshuffle;

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr bool DoubleSmemBuffer = Problem::DoubleSmemBuffer;

    static constexpr auto Scheduler = Problem::Scheduler;

    [[nodiscard]] CK_TILE_HOST static const std::string GetPipelineName()
    {
        // clang-format off
        return "COMPUTE_ASYNC_EIGHT_WAVES";
        // clang-format on
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(I0);
        constexpr index_t WaveNumN = BlockGemmShape::BlockWarps::at(I1);
        return concat('_', "pipeline_AgBgCrCompAsyncEightWaves", 
                      concat('x', MPerBlock, NPerBlock, KPerBlock),  BlockSize,
                      concat('x', GetVectorSizeA(), GetVectorSizeB()),
                      concat('x', WaveNumM, WaveNumN),
                      concat('x', kPadM, kPadN, kPadK),
                      Problem::GetName());
        // clang-format on
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    static constexpr index_t MFMA_INST = MIterPerWarp * NIterPerWarp * KIterPerWarp;

    // Scales are packed so odd numbers of iterations greater than 1 are not supported
    static_assert((MIterPerWarp == 1) || (MIterPerWarp % 2 == 0));
    static_assert((NIterPerWarp == 1) || (NIterPerWarp % 2 == 0));
    static_assert((KIterPerWarp == 1) || (KIterPerWarp % 2 == 0));

    template <GemmPipelineScheduler Scheduler>
    struct PipelineImpl : public PipelineImplBase
    {
    };

    template <>
    struct PipelineImpl<GemmPipelineScheduler::Intrawave> : public PipelineImplBase
    {
        using Base = PipelineImplBase;

        template <bool HasHotLoop,
                  TailNumber TailNum,
                  typename AsDramBlockWindowTmp,
                  typename BsDramBlockWindowTmp,
                  typename AElementFunction,
                  typename BElementFunction,
                  typename ScaleADramBlockWindowTmp,
                  typename ScaleBDramBlockWindowTmp,
                  typename std::enable_if_t<!is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                                !is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                            bool>* = nullptr>
        CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                       const AElementFunction& a_element_func,
                                       const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                       const BElementFunction& b_element_func,
                                       const ScaleADramBlockWindowTmp& scale_a_window,
                                       const ScaleBDramBlockWindowTmp& scale_b_window,
                                       index_t num_loop,
                                       void* __restrict__ p_smem) const
        {
            // TODO: A/B elementwise functions currently not supported
            ignore = a_element_func;
            ignore = b_element_func;

            // ------
            // Checks
            // ------
            static_assert(
                std::is_same_v<ADataType,
                               remove_cvref_t<typename AsDramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BDataType,
                                   remove_cvref_t<typename BsDramBlockWindowTmp::DataType>>,
                "A/B Dram block window should have the same data type as appropriate "
                "([A|B]DataType) defined in Problem definition!");

            static_assert(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>, "Wrong!");
            static_assert(std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>, "Wrong!");

            static_assert((MPerBlock == AsDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                           KPerBlock == AsDramBlockWindowTmp{}.get_window_lengths()[I1]),
                          "A block window has incorrect lengths for defined ALayout!");
            static_assert(Preshuffle //
                              ? (NWarps == BsDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                                 kflatKPerBlock == BsDramBlockWindowTmp{}.get_window_lengths()[I1])
                              : (NPerBlock == BsDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                                 KPerBlock == BsDramBlockWindowTmp{}.get_window_lengths()[I1]),
                          "B block window has incorrect lengths for defined BLayout!");

            // ------------------
            // Hot loop scheduler
            // ------------------
            auto hot_loop_scheduler = [&]() {
                __builtin_amdgcn_sched_group_barrier(0x008, MIterPerWarp, 0); // MFMA
                s_waitcnt_lgkm<4>();
                __builtin_amdgcn_sched_group_barrier(0x004, 1, 0); // lgkmcnt / SALU
                static_for<0, MFMA_INST - MIterPerWarp, 1>{}([&](auto) {
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                });
                __builtin_amdgcn_sched_barrier(0);
            };

            // -------
            // Compute
            // -------
            return Base::template Run_<HasHotLoop, TailNum>(p_smem,
                                                            num_loop,
                                                            a_dram_block_window_tmp,
                                                            b_dram_block_window_tmp,
                                                            scale_a_window,
                                                            scale_b_window,
                                                            hot_loop_scheduler);
        }
    };

    template <typename AsDramBlockWindowTmp,
              typename BsDramBlockWindowTmp,
              typename AElementFunction,
              typename BElementFunction,
              typename ScaleADramBlockWindowTmp,
              typename ScaleBDramBlockWindowTmp,
              typename std::enable_if_t<!is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                            !is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                   const AElementFunction& a_element_func,
                                   const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const BElementFunction& b_element_func,
                                   const ScaleADramBlockWindowTmp& scale_a_window,
                                   const ScaleBDramBlockWindowTmp& scale_b_window,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        const bool has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto tail_number  = Base::GetBlockLoopTailNum(num_loop);
        const auto RunPipeline  = [&](auto hot_loop_, auto tail_num_) {
            return PipelineImpl<Scheduler>{}.template operator()<hot_loop_.value, tail_num_.value>(
                a_dram_block_window_tmp,
                a_element_func,
                b_dram_block_window_tmp,
                b_element_func,
                scale_a_window,
                scale_b_window,
                num_loop,
                p_smem);
        };

        return Base::TailHandler(RunPipeline, has_hot_loop, tail_number);
    }

    template <typename AsDramBlockWindowTmp,
              typename BsDramBlockWindowTmp,
              typename ScaleADramBlockWindowTmp,
              typename ScaleBDramBlockWindowTmp,
              typename std::enable_if_t<!is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                            !is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const ScaleADramBlockWindowTmp& scale_a_window,
                                   const ScaleBDramBlockWindowTmp& scale_b_window,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        const bool has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto tail_number  = Base::GetBlockLoopTailNum(num_loop);
        const auto RunPipeline  = [&](auto hot_loop_, auto tail_num_) {
            return PipelineImpl<Scheduler>{}.template operator()<hot_loop_.value, tail_num_.value>(
                a_dram_block_window_tmp,
                identity{},
                b_dram_block_window_tmp,
                identity{},
                scale_a_window,
                scale_b_window,
                num_loop,
                p_smem);
        };

        return Base::TailHandler(RunPipeline, has_hot_loop, tail_number);
    }
};

} // namespace ck_tile
