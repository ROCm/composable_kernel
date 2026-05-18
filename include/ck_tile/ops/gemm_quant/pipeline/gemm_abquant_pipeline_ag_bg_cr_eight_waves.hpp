// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <sstream>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_eight_waves_base.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_abquant_pipeline_ag_bg_cr_eight_waves_policy.hpp"
#include "ck_tile/host/concat.hpp"

namespace ck_tile {

// Compute optimized pipeline
// GlobalPrefetchStages: 2
// LocalPreFillStages: 1
// LocalPreFetchStages: 1
// LocalSharedMemoryBuffer: 1

template <typename Problem, typename Policy = GemmABQuantPipelineAgBgCrAsyncPolicy>
struct ABQuantGemmPipelineAgBgCrEightWaves : public BaseGemmPipelineAgBgCrCompV3<Problem>
{
    using Base             = BaseGemmPipelineAgBgCrCompV3<Problem>;
    using PipelineImplBase = GemmPipelineAgBgCrEightWavesImplBase<Problem, Policy>;
#if defined(__gfx950__)
    static constexpr bool kIsAvailable = true;
#else
    static constexpr bool kIsAvailable = false;
#endif

    using ADataType       = remove_cvref_t<typename Problem::ADataType>;
    using AQDataType      = remove_cvref_t<typename Problem::AQDataType>;
    using BDataType       = remove_cvref_t<typename Problem::BDataType>;
    using BQDataType      = remove_cvref_t<typename Problem::BQDataType>;
    using CDataType       = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape  = remove_cvref_t<typename Problem::BlockGemmShape>;
    using AQuantGroupSize = remove_cvref_t<typename Problem::AQuantGroupSize>;
    using BQuantGroupSize = remove_cvref_t<typename Problem::BQuantGroupSize>;
    // BDataType gets converted from PkInt4 during loading
    using OverrideBDataType =
        std::conditional_t<std::is_same_v<BDataType, pk_int4_t>, ADataType, BDataType>;

    static_assert(BQuantGroupSize::kM == 1, "only N/K blocks for BQuant kernel!");
    static_assert(AQuantGroupSize::kN == 1, "only M/K blocks for AQuant kernel!");
    static_assert(AQuantGroupSize::kM == 1, "no block M for AQuant kernel supported yet!");
    static_assert(AQuantGroupSize::kK == BQuantGroupSize::kK,
                  "AQuantGroupSize::kK should be equal to BQuantGroupSize::kK");

    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};

    static constexpr index_t APackedSize =
        ck_tile::numeric_traits<remove_cvref_t<ADataType>>::PackedSize;
    static constexpr index_t BPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<BDataType>>::PackedSize;

    static constexpr index_t AQPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<AQDataType>>::PackedSize;

    static constexpr index_t BQPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<BQDataType>>::PackedSize;

    using ALayout  = remove_cvref_t<typename Problem::ALayout>;
    using AQLayout = remove_cvref_t<typename Problem::AQLayout>;
    using BLayout  = remove_cvref_t<typename Problem::BLayout>;
    using BQLayout = remove_cvref_t<typename Problem::BQLayout>;
    using CLayout  = remove_cvref_t<typename Problem::CLayout>;

    using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;
    using WarpGemm  = typename BlockGemm::WarpGemm;

    static constexpr index_t BlockSize   = Problem::kBlockSize;
    static constexpr index_t MPerBlock   = BlockGemmShape::kM;
    static constexpr index_t NPerBlock   = BlockGemmShape::kN;
    static constexpr index_t KPerBlock   = BlockGemmShape::kK;
    static constexpr index_t KPerBlockAQ = KPerBlock / AQuantGroupSize::kK;
    static constexpr index_t NPerBlockBQ = NPerBlock / BQuantGroupSize::kN;
    static constexpr index_t KPerBlockBQ = KPerBlock / BQuantGroupSize::kK;
    static constexpr index_t MWarps      = BlockGemmShape::BlockWarps::at(I0);
    static constexpr index_t NWarps      = BlockGemmShape::BlockWarps::at(I1);
    static constexpr index_t KWarps      = BlockGemmShape::BlockWarps::at(I2);

    static constexpr index_t MIterPerWarp = MPerBlock / (MWarps * WarpGemm::kM);
    static constexpr index_t NIterPerWarp = NPerBlock / (NWarps * WarpGemm::kN);
    static constexpr index_t KIterPerWarp = KPerBlock / (KWarps * WarpGemm::kK);

    static constexpr index_t GetVectorSizeA() { return Policy::template GetVectorSizeA<Problem>(); }
    static constexpr index_t GetVectorSizeB() { return Policy::template GetVectorSizeB<Problem>(); }
    static constexpr index_t GetVectorSizeAQ()
    {
        return Policy::template GetVectorSizeAQ<Problem>();
    }
    static constexpr index_t GetVectorSizeBQ()
    {
        return Policy::template GetVectorSizeBQ<Problem>();
    }

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr bool DoubleSmemBuffer = Problem::DoubleSmemBuffer;
    static constexpr bool PreshuffleB      = Problem::PreshuffleB;
    static constexpr bool PreshuffleQuant  = Problem::Traits::PreshuffleQuant;

    static constexpr index_t kflatKPerBlock = BlockGemmShape::flatKPerBlock;
    static constexpr index_t flatKPerWarp   = BlockGemmShape::flatKPerWarp;
    static constexpr index_t flatNPerWarp   = BlockGemmShape::flatNPerWarp;
    static constexpr index_t WarpTileN      = BlockGemmShape::WarpTile::at(I1);
    static constexpr auto Scheduler         = Problem::Scheduler;

    static_assert(Scheduler == GemmPipelineScheduler::Intrawave, "Only Intrawave supported!");
    static_assert(!PipelineImplBase::is_a_load_tr, "Transposed A not supported!");
    static_assert(!PipelineImplBase::is_b_load_tr, "Transposed B not supported!");

    using Base::PrefetchStages;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "ABQuantGemmPipelineAgBgCrEightWaves",
                      concat('x', MPerBlock, NPerBlock, KPerBlock),
                      Problem::kBlockSize,
                      concat('x', MWarps, NWarps),
                      concat('x', WarpGemm::kM, WarpGemm::kN, WarpGemm::kK),
                      concat('x', kPadM, kPadN, kPadK), AQuantGroupSize::GetName(), BQuantGroupSize::GetName());
        // clang-format on
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        // We are not storing the original packed type in LDS, so we need to multiply the smem size
        // by the packed size.
        constexpr index_t smem_size_a = Policy::template GetSmemSizeA<Problem>() * APackedSize;
        constexpr index_t smem_size_b = Policy::template GetSmemSizeB<Problem>() * BPackedSize;

        return 2 * (smem_size_a + smem_size_b);
    }

    CK_TILE_HOST static std::string Print() { return "ABQuantGemmPipelineAgBgCrEightWaves\n"; }

    static constexpr index_t MFMA_INST = MIterPerWarp * NIterPerWarp * KIterPerWarp;

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
                  typename ADramBlockWindowTmp,
                  typename BDramBlockWindowTmp,
                  typename AQDramBlockWindowTmp,
                  typename BQDramBlockWindowTmp>
        CK_TILE_DEVICE auto operator()(void* __restrict__ p_smem,
                                       const ADramBlockWindowTmp& a_dram_window_tmp,
                                       const BDramBlockWindowTmp& b_dram_window_tmp,
                                       const AQDramBlockWindowTmp& aq_dram_window_tmp,
                                       const BQDramBlockWindowTmp& bq_dram_window_tmp,
                                       index_t num_loop) const
        {
            // ------
            // Checks
            // ------
            static_assert(
                std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BDataType,
                                   remove_cvref_t<typename BDramBlockWindowTmp::DataType>> &&
                    std::is_same_v<AQDataType,
                                   remove_cvref_t<typename AQDramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BQDataType,
                                   remove_cvref_t<typename BQDramBlockWindowTmp::DataType>>,
                "A/B/AQ/BQ Dram block window should have the same data type as appropriate "
                "([A|B|AQ|BQ]DataType) defined in Problem definition!");

            static_assert(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>, "Wrong!");
            static_assert(std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>, "Wrong!");
            static_assert(std::is_same_v<BQLayout, tensor_layout::gemm::ColumnMajor>, "Wrong!");

            static_assert((MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0] &&
                           KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1]),
                          "A block window has incorrect lengths for defined ALayout!");
            static_assert(PreshuffleB //
                              ? (NWarps == BDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                                 kflatKPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1])
                              : (NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                                 KPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1]),
                          "B block window has incorrect lengths for defined BLayout!");
            static_assert((NPerBlockBQ == BQDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                           KPerBlockBQ == BQDramBlockWindowTmp{}.get_window_lengths()[I1]),
                          "Bq block window has incorrect lengths for defined BqLayout!");

            // ------------------
            // Hot loop scheduler
            // ------------------
            auto hot_loop_scheduler = [&]() {
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
                __builtin_amdgcn_sched_group_barrier(0x002, MIterPerWarp, 0);
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
                s_waitcnt_lgkm<4>();
                __builtin_amdgcn_sched_group_barrier(0x004, 1, 0); // lgkmcnt
                static_for<0, MFMA_INST - 3, 1>{}([&](auto) {
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
                    __builtin_amdgcn_sched_group_barrier(0x002, 4, 0);
                });
                __builtin_amdgcn_sched_group_barrier(0x002, 12, 0);

                __builtin_amdgcn_sched_barrier(0);
            };

            // -------
            // Compute
            // -------
            return Base::template Run_<Problem::HasHotLoop, Problem::TailNum>(p_smem,
                                                                              num_loop,
                                                                              a_dram_window_tmp,
                                                                              b_dram_window_tmp,
                                                                              aq_dram_window_tmp,
                                                                              bq_dram_window_tmp,
                                                                              hot_loop_scheduler);
        }
    };

    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename AQDramBlockWindowTmp,
              typename BQDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const AQDramBlockWindowTmp& aq_dram_block_window_tmp,
                                   const BQDramBlockWindowTmp& bq_dram_block_window_tmp,
                                   index_t num_loop,
                                   void* p_smem,
                                   index_t /*m*/ = 0,
                                   index_t /*n*/ = 0) const
    {
#ifdef __gfx950__
        return PipelineImpl<Scheduler>{}.template operator()<Problem::HasHotLoop, Problem::TailNum>(
            p_smem,
            a_dram_block_window_tmp,
            b_dram_block_window_tmp,
            aq_dram_block_window_tmp,
            bq_dram_block_window_tmp,
            num_loop);
#else
        ignore = a_dram_block_window_tmp;
        ignore = b_dram_block_window_tmp;
        ignore = aq_dram_block_window_tmp;
        ignore = bq_dram_block_window_tmp;
        ignore = num_loop;
        ignore = p_smem;
        return BlockGemm::MakeCBlockTile(); // do nothing on non-gfx950 targets
#endif
    }

    /// @brief Runtime pipeline dispatch operator for grouped GEMM kernels.
    ///
    /// This operator is used by grouped GEMM kernels where pipeline parameters
    /// (has_hot_loop, num_loop, tail_number) are calculated on the device side
    /// at runtime, not on the host side during compilation. This is necessary
    /// because different GEMM problems in the group may have different K dimensions,
    /// requiring different pipeline configurations that cannot be determined at
    /// compile time.
    ///
    /// @param a_dram_block_window_tmp Block window for A tensor in DRAM
    /// @param b_dram_block_window_tmp Block window for B tensor in DRAM
    /// @param aq_dram_block_window_tmp Block window for AQ (quantization scale) tensor in DRAM
    /// @param bq_dram_block_window_tmp Block window for BQ (quantization scale) tensor in DRAM
    /// @param num_loop Number of main loop iterations (calculated on device)
    /// @param has_hot_loop Whether the pipeline has a hot loop (calculated on device)
    /// @param tail_number Type of tail handling required (calculated on device)
    /// @param p_smem Pointer to shared memory
    /// @return Accumulated result tile in registers
    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename AQDramBlockWindowTmp,
              typename BQDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const AQDramBlockWindowTmp& aq_dram_block_window_tmp,
                                   const BQDramBlockWindowTmp& bq_dram_block_window_tmp,
                                   index_t num_loop,
                                   bool has_hot_loop,
                                   TailNumber tail_number,
                                   void* p_smem,
                                   index_t /*m*/ = 0,
                                   index_t /*n*/ = 0) const
    {
#ifdef __gfx950__
        const auto RunPipeline = [&](auto has_hot_loop_, auto tail_number_) {
            constexpr bool hot_loop = has_hot_loop_.value;
            constexpr auto tail_num = tail_number_.value;

            return PipelineImpl<Scheduler>{}.template operator()<hot_loop, tail_num>(
                p_smem,
                a_dram_block_window_tmp,
                b_dram_block_window_tmp,
                aq_dram_block_window_tmp,
                bq_dram_block_window_tmp,
                num_loop);
        };
        return Base::TailHandler(RunPipeline, has_hot_loop, tail_number);
#else
        ignore = a_dram_block_window_tmp;
        ignore = b_dram_block_window_tmp;
        ignore = aq_dram_block_window_tmp;
        ignore = bq_dram_block_window_tmp;
        ignore = num_loop;
        ignore = has_hot_loop;
        ignore = tail_number;
        ignore = p_smem;
        return BlockGemm::MakeCBlockTile(); // do nothing on non-gfx950 targets
#endif
    }
};

} // namespace ck_tile
