// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"

namespace ck_tile {

/// @brief Wavelet GEMM pipeline: 2-way wave specialization for load/math overlap.
///
/// Splits the workgroup into load waves and math waves:
///   - Math threads [0, MathBlockSize): LDS reads + MFMA accumulation only
///   - Load threads [MathBlockSize, LaunchBlockSize): global loads + LDS writes only
///
/// Load threads use a remapped partition index ({warp_id - NumMathWarps, lane_id})
/// so they appear as virtual threads [0, LoadBlockSize) to tile distributions.
/// This allows using standard Problem-sized distributions (MathBlockSize threads)
/// without modifying the core tile distribution infrastructure.
///
/// The pipeline returns c_block_tile. Math threads hold the real accumulated result;
/// load threads return a zero-initialized c_block_tile.
///
/// @tparam Problem  The GEMM pipeline problem. Problem::kBlockSize == MathBlockSize
///                  (the standard NumWarps * warp_size for BlockGemm compatibility).
/// @tparam Policy   The universal pipeline policy.
/// @tparam NumLoadWaves_  Number of additional load waves (default 4).
template <typename Problem,
          typename Policy       = UniversalGemmPipelineAgBgCrPolicy,
          index_t NumLoadWaves_ = 4>
struct GemmPipelineAgBgCrWavelet
{
    using PipelineImplBase = GemmPipelineAgBgCrImplBase<Problem, Policy>;

    using AsDataType = remove_cvref_t<typename Problem::AsDataTypeTuple>;
    using BsDataType = remove_cvref_t<typename Problem::BsDataTypeTuple>;
    using CDataType  = remove_cvref_t<typename Problem::CDataType>;

    using AElementWise   = remove_cvref_t<typename Problem::AElementWise>;
    using BElementWise   = remove_cvref_t<typename Problem::BElementWise>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    using AsLayout = remove_cvref_t<typename Problem::AsLayoutTuple>;
    using BsLayout = remove_cvref_t<typename Problem::BsLayoutTuple>;
    using CLayout  = remove_cvref_t<typename Problem::CLayout>;

    using ALayout = remove_cvref_t<std::tuple_element_t<0, AsLayout>>;
    using BLayout = remove_cvref_t<std::tuple_element_t<0, BsLayout>>;

    using ADataType = remove_cvref_t<std::tuple_element_t<0, AsDataType>>;
    using BDataType = remove_cvref_t<std::tuple_element_t<0, BsDataType>>;

    using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;
    using I0        = number<0>;
    using I1        = number<1>;

    // --- Thread group sizes ---
    // MathBlockSize: the standard block size for BlockGemm (NumWarps * warp_size).
    // This is Problem::kBlockSize.
    static constexpr index_t MathBlockSize = Problem::kBlockSize;

    // LoadBlockSize: extra threads dedicated to loading.
    static constexpr index_t NumLoadWaves  = NumLoadWaves_;
    static constexpr index_t LoadBlockSize = NumLoadWaves * get_warp_size();

    // LaunchBlockSize: total threads launched per workgroup.
    static constexpr index_t LaunchBlockSize = MathBlockSize + LoadBlockSize;

    static constexpr bool LargeTensors = Problem::LargeTensors;

    // BlockSize exposed to the kernel for BlockGemm compatibility.
    // The kernel uses this for MFMA wave-to-tile mapping.
    static constexpr index_t BlockSize = MathBlockSize;

    // Standard pipeline interface members (required by builder instance traits).
    static constexpr auto Scheduler        = Problem::Scheduler;
    static constexpr bool DoubleSmemBuffer = false;
    static constexpr index_t NumWaveGroups = 1;

    // Wavelet performs no weight preshuffle. Exposed so kernels that route
    // through UniversalGemmKernel (e.g. bwd_weight) can gate their
    // `if constexpr(GemmPipeline::Preshuffle)` branches, matching every other
    // pipeline which defines `Preshuffle = Problem::Preshuffle;`.
    static constexpr index_t Preshuffle = Problem::Preshuffle;

    // --- Wavelet traits ---
    static constexpr bool IsWavelet = true;

    CK_TILE_DEVICE static bool IsMathWave() { return get_thread_local_1d_id() < MathBlockSize; }

    // --- Pipeline traits (matching standard pipeline interface) ---
    static constexpr index_t PrefetchStages  = 1;
    static constexpr index_t PrefillStages   = 1;
    static constexpr index_t GlobalBufferNum = 1;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr auto is_a_load_tr_v = bool_constant<PipelineImplBase::is_a_load_tr>{};
    static constexpr auto is_b_load_tr_v = bool_constant<PipelineImplBase::is_b_load_tr>{};

    static constexpr bool Async               = false;
    static constexpr bool UsePersistentKernel = false;

    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeA()
    {
        return Policy::template GetVectorSizeA<Problem, IsWave32Host>();
    }
    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeB()
    {
        return Policy::template GetVectorSizeB<Problem, IsWave32Host>();
    }
    static constexpr index_t GetVectorSizeC() { return Policy::template GetVectorSizeC<Problem>(); }

    [[nodiscard]] CK_TILE_HOST static const std::string GetPipelineName() { return "WAVELET"; }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(I0{});
        constexpr index_t WaveNumN = BlockGemmShape::BlockWarps::at(I1{});
        return concat('_',
                      "pipeline_AgBgCrWavelet",
                      concat('x', MPerBlock, NPerBlock, KPerBlock),
                      BlockSize,
                      concat('x', GetVectorSizeA(), GetVectorSizeB(), GetVectorSizeC()),
                      concat('x', WaveNumM, WaveNumN),
                      concat('x', NumLoadWaves),
                      Problem::GetName());
    }

    CK_TILE_HOST_DEVICE static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > 1;
    }

    CK_TILE_HOST_DEVICE static constexpr TailNumber GetBlockLoopTailNum(index_t /*num_loop*/)
    {
        return TailNumber::Odd;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    static constexpr index_t NumMathWarps = MathBlockSize / get_warp_size();

    static constexpr index_t APackedSize =
        ck_tile::numeric_traits<remove_cvref_t<ADataType>>::PackedSize;
    static constexpr index_t BPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<BDataType>>::PackedSize;

    static constexpr index_t GetSmemPackA() { return Policy::template GetSmemPackA<Problem>(); }
    static constexpr index_t GetSmemPackB() { return Policy::template GetSmemPackB<Problem>(); }

    // LoadProblem: same as Problem but with kBlockSize = LoadBlockSize.
    // Used for DRAM tile distributions so that LoadBlockSize threads (which may
    // differ from MathBlockSize) cover the full tile cooperatively.
    struct LoadProblem : Problem
    {
        static constexpr index_t kBlockSize = LoadBlockSize;
    };

    // --- Main pipeline operator ---
    template <typename ADramBlockWindowTmp, typename BDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        // ----------------------------------------------------------------
        // Thread role.
        //
        // VGPR pressure: the body is split into two TOP-LEVEL, mutually
        // exclusive branches by role. Each branch constructs ONLY its own
        // windows / tiles / accumulator, so the register allocator can map the
        // two roles' (disjoint) live ranges onto the same physical VGPRs -- the
        // kernel's single per-wave VGPR count approaches max(load, math) rather
        // than load + math. Constructing any role-specific object before the
        // branch would force every wave to carry it and defeat this.
        //
        // BARRIER INVARIANT (critical -- mismatch deadlocks the workgroup):
        // both branches execute exactly 1 + 2*(num_loop-1) block_sync_lds()
        // calls, in the same order. AMD s_barrier pairs the Nth barrier each
        // wave reaches, so identical counts (not identical PCs) are required.
        // ----------------------------------------------------------------
        const bool is_math = IsMathWave();

        // LDS views over shared smem -- cheap (descriptors), used by both roles.
        auto&& [a_lds_block, b_lds_block] = PipelineImplBase{}.GetABLdsTensorViews(p_smem);

        constexpr bool is_a_col_major = std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
        constexpr bool is_b_row_major = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;

        // LDS window distributions (compile-time, no registers).
        constexpr auto a_lds_load_tile_distr =
            make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
        constexpr auto b_lds_load_tile_distr =
            make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());

        // Accumulator type -- used so both branches return the same type. Load
        // waves return a default-constructed (dead) tile that RunGemm discards.
        using CBlockTile = decltype(BlockGemm().MakeCBlockTile());

        if(is_math)
        {
            // ============================================================
            // MATH WAVES: LDS read + MFMA accumulate. No DRAM/load state.
            // ============================================================
            auto [a_copy_lds_window_unused, a_lds_gemm_window] =
                PipelineImplBase{}.MakeALdsWindows(a_lds_block, a_lds_load_tile_distr);
            auto [b_copy_lds_window_unused, b_lds_gemm_window] =
                PipelineImplBase{}.MakeBLdsWindows(b_lds_block, b_lds_load_tile_distr);
            (void)a_copy_lds_window_unused;
            (void)b_copy_lds_window_unused;

            auto block_gemm   = BlockGemm();
            auto c_block_tile = block_gemm.MakeCBlockTile();
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            // Prologue barrier (#0): wait for load waves to fill LDS iter 0.
            block_sync_lds();

            index_t i = 0;
            while(i < num_loop - 1)
            {
                // MFMA on current LDS data (concurrent with load waves' DRAM fetch).
                block_gemm.LocalPrefetch(
                    a_lds_gemm_window, b_lds_gemm_window, is_a_load_tr_v, is_b_load_tr_v);
                block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);

                block_sync_lds(); // barrier A: math done reading LDS
                block_sync_lds(); // barrier B: load done writing next LDS (math idle between)

                ++i;
            }

            // Tail: last iteration's MFMA.
            block_gemm.LocalPrefetch(
                a_lds_gemm_window, b_lds_gemm_window, is_a_load_tr_v, is_b_load_tr_v);
            block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);

            return c_block_tile;
        }
        else
        {
            // ============================================================
            // LOAD WAVES: DRAM read + LDS write. No accumulator/MFMA state.
            // ============================================================
            // Remap warp ID so load threads appear as virtual threads
            // [0, LoadBlockSize) to the LoadProblem tile distributions.
            const auto load_partition =
                array<index_t, 2>{get_warp_id() - NumMathWarps, get_lane_id()};

            using YPerTileA =
                std::conditional_t<is_a_col_major, number<KPerBlock>, number<MPerBlock>>;
            using XPerTileA =
                std::conditional_t<is_a_col_major, number<MPerBlock>, number<KPerBlock>>;
            auto a_copy_dram_window =
                make_tile_window(a_dram_block_window_tmp.get_bottom_tensor_view(),
                                 make_tuple(YPerTileA{}, XPerTileA{}),
                                 a_dram_block_window_tmp.get_window_origin(),
                                 Policy::template MakeADramTileDistribution<LoadProblem>(),
                                 load_partition);

            using YPerTileB =
                std::conditional_t<is_b_row_major, number<KPerBlock>, number<NPerBlock>>;
            using XPerTileB =
                std::conditional_t<is_b_row_major, number<NPerBlock>, number<KPerBlock>>;
            auto b_copy_dram_window =
                make_tile_window(b_dram_block_window_tmp.get_bottom_tensor_view(),
                                 make_tuple(YPerTileB{}, XPerTileB{}),
                                 b_dram_block_window_tmp.get_window_origin(),
                                 Policy::template MakeBDramTileDistribution<LoadProblem>(),
                                 load_partition);

            auto [a_copy_lds_window, a_lds_gemm_window_unused] =
                PipelineImplBase{}.MakeALdsWindows(a_lds_block, a_lds_load_tile_distr);
            auto [b_copy_lds_window, b_lds_gemm_window_unused] =
                PipelineImplBase{}.MakeBLdsWindows(b_lds_block, b_lds_load_tile_distr);
            (void)a_lds_gemm_window_unused;
            (void)b_lds_gemm_window_unused;

            auto a_block_tile = decltype(load_tile(a_copy_dram_window)){};
            auto b_block_tile = decltype(load_tile(b_copy_dram_window)){};

            using ADramTileWindowStep = typename ADramBlockWindowTmp::BottomTensorIndex;
            using BDramTileWindowStep = typename BDramBlockWindowTmp::BottomTensorIndex;
            constexpr ADramTileWindowStep a_dram_step =
                is_a_col_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
            constexpr BDramTileWindowStep b_dram_step =
                is_b_row_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);

            // store_tile helper with optional in-register transpose (for
            // architectures without hardware transpose-load, e.g. gfx942).
            auto store_a_to_lds = [&]() {
                if constexpr(is_a_col_major && !PipelineImplBase::is_a_load_tr)
                {
                    auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                        Policy::template MakeShuffledARegTileDistribution<LoadProblem>());
                    transpose_tile2d(a_shuffle_tmp, a_block_tile);
                    store_tile(a_copy_lds_window, a_shuffle_tmp, load_partition);
                }
                else
                {
                    store_tile(a_copy_lds_window, a_block_tile, load_partition);
                }
            };
            auto store_b_to_lds = [&]() {
                if constexpr(is_b_row_major && !PipelineImplBase::is_b_load_tr)
                {
                    auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                        Policy::template MakeShuffledBRegTileDistribution<LoadProblem>());
                    transpose_tile2d(b_shuffle_tmp, b_block_tile);
                    store_tile(b_copy_lds_window, b_shuffle_tmp, load_partition);
                }
                else
                {
                    store_tile(b_copy_lds_window, b_block_tile, load_partition);
                }
            };

            // Prologue: fetch iteration 0 from DRAM -> LDS.
            a_block_tile = load_tile(a_copy_dram_window);
            move_tile_window(a_copy_dram_window, a_dram_step);
            b_block_tile = load_tile(b_copy_dram_window);
            move_tile_window(b_copy_dram_window, b_dram_step);
            store_a_to_lds();
            store_b_to_lds();

            // Prologue barrier (#0): LDS iter 0 ready for math waves.
            block_sync_lds();

            index_t i = 0;
            while(i < num_loop - 1)
            {
                // Fetch next iteration from DRAM (concurrent with math MFMA).
                a_block_tile = load_tile(a_copy_dram_window);
                move_tile_window(a_copy_dram_window, a_dram_step);
                b_block_tile = load_tile(b_copy_dram_window);
                move_tile_window(b_copy_dram_window, b_dram_step);

                block_sync_lds(); // barrier A: math done reading current LDS

                // Write next iteration's data to LDS.
                store_a_to_lds();
                store_b_to_lds();

                block_sync_lds(); // barrier B: LDS populated for next iteration

                ++i;
            }

            // Load waves hold no result; return a dead tile (discarded by RunGemm).
            return CBlockTile{};
        }
    }
};

} // namespace ck_tile
