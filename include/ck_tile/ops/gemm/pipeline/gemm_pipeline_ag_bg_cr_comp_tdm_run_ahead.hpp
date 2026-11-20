// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_tdm_run_ahead_policy.hpp"

namespace ck_tile {

/**
 * @brief TDM GEMM pipeline whose math waves fill an LDS ring ahead of themselves (gfx1250).
 *
 * Every wave multiplies: BlockGemm maps the C tile onto all of them exactly as in any other
 * pipeline. Two of them also stream one operand each into a ring of NumSlots LDS buffers, one
 * TENSOR_LOAD_TO_LDS per K step, RunAhead steps ahead of the step being multiplied:
 *
 *     wave 0 --TDM A--> +--------+--------+--------+--------+
 *                       | slot 0 | slot 1 | slot 2 | slot 3 | --ds_read--> all waves --> C tile
 *     wave 1 --TDM B--> +--------+--------+--------+--------+
 *
 * Each K step k is one workgroup barrier. Before it, a filling wave retires its fill for step k
 * and every wave has drained its LDS reads of step k - 1. After it, slot k is complete for
 * everyone and the slot step k + RunAhead reuses was last read at step k + RunAhead - NumSlots,
 * which is at most k - 1, so the fills for step k + RunAhead are issued before the math of
 * step k starts.
 *
 * Why there are no dedicated producer waves, as in GemmPipelineAgBgCrCompTDMProducerConsumer:
 * a kernel has a single VGPR count, so a producer wave sharing a SIMD with a math wave gets the
 * math wave's allocation. The tiles this pipeline is for keep a 128x128 C tile per wave
 * (512 accumulator VGPRs plus double-buffered fragments), one wave per SIMD; a second wave on
 * any SIMD would not fit in its 1024 VGPRs. Issuing a fill is a handful of scalar instructions,
 * so the math waves can afford to do it themselves.
 *
 * Why it does not drain per step, as GemmPipelineAgBgCrCompTDMV2 and similar designs do: with
 * one wave per SIMD nothing hides a fill's latency except issuing it early, which needs more
 * than two buffers and a retire that waits only for the oldest fill.
 *
 * The handshake is an ordinary workgroup barrier, which every wave crosses once per step, so
 * there is no role asymmetry and no named barrier: none of the ring protocol of named_barrier.hpp
 * applies.
 *
 * Other targets compile an inert fallback that returns zeros, so that multi-target builds still
 * compile; UniversalGemmKernel::IsSupportedArgument rejects those devices.
 */
template <typename Problem, typename Policy = GemmPipelineAgBgCrCompTDMRunAheadPolicy<>>
struct GemmPipelineAgBgCrCompTDMRunAhead
{
    using PipelineImplBase = GemmPipelineAgBgCrImplBase<Problem, Policy>;

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

    using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    static constexpr index_t NumSlots = Policy::NumSlots;
    static constexpr index_t RunAhead = Policy::RunAhead;

    static constexpr index_t NumWaves = BlockGemmShape::NumWarps;
    /** The wave that streams A, and the one that streams B. One wave does both when the block
     *  has a single wave. */
    static constexpr index_t FillerWaveA = 0;
    static constexpr index_t FillerWaveB = NumWaves > 1 ? 1 : 0;
    /// Tensor ops a filling wave issues per K step: one per operand it streams.
    static constexpr index_t FillsPerStep = FillerWaveA == FillerWaveB ? 2 : 1;

    static constexpr index_t BlockSize = Problem::kBlockSize;

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr index_t NumWaveGroups    = Problem::NumWaveGroups;
    static constexpr index_t Preshuffle       = Problem::Preshuffle;
    static constexpr bool UsePersistentKernel = Problem::Traits::UsePersistentKernel;

    // TDM vectorises internally and handles ragged extents, so launch checks do not apply.
    static constexpr bool skipCheckValidLaunchParams = true;
    // Read by the kernels that host the TDM pipelines. The NumSlots ring always multi-buffers
    // LDS (Problem::DoubleSmemBuffer is not consulted), and cluster launch is rejected below.
    static constexpr bool DoubleSmemBuffer = true;
    static constexpr bool UseClusterLaunch = false;
    // Validated only in the plain GEMM kernel, persistent or not: every kernel that derives from or
    // wraps UniversalGemmKernel (Stream-K, MX, batched, grouped, multi-D, multi-ABD, batched
    // contraction) rejects it at compile time (UniversalGemmKernel::kIsGemmKernelOnlyPipeline).
    static constexpr bool kGemmKernelOnly = true;

    static constexpr index_t KSubTileNum = Policy::template GetPipelineSubTileNum<Problem>().value;

    static constexpr auto is_a_load_tr_v = bool_constant<PipelineImplBase::is_a_load_tr>{};
    static constexpr auto is_b_load_tr_v = bool_constant<PipelineImplBase::is_b_load_tr>{};

    static_assert(NumWaveGroups == 1, "wave groups are not supported");
    // NumWaves and kBlockSize count every BlockWarps wave, but BlockGemm's A/B/C distributions
    // map the wave id over MWarps x NWarps only and nothing reduces partial sums across K waves.
    // A wave id past MWarps x NWarps would get an M warp index of MWarps or more: it would read A
    // past its LDS slot and own C rows outside the block tile. It would still reach every
    // StepBarrier, so the result is silent corruption, not a hang.
    static_assert(BlockGemmShape::BlockWarps::at(number<2>{}) == 1,
                  "BlockWarps[2] (K waves) must be 1: BlockGemm distributes waves over M and N "
                  "only and the pipeline has no K-wave reduction");
    static_assert(!Preshuffle, "preshuffled B is not supported");
    static_assert(!kPadM && !kPadN && !kPadK,
                  "padding is not supported: TDM handles ragged extents itself, and padded "
                  "descriptor lengths would carry its transfers past the tensor");
    static_assert(!Policy::template isClusterLaunch<Problem>(),
                  "cluster multicast is not supported: a peer's transfer into this workgroup's "
                  "LDS is invisible to the tensor counter a filling wave retires on");
    static_assert(Policy::DataCachePrefetchA == DataCachePrefetchKind::None &&
                      Policy::DataCachePrefetchB == DataCachePrefetchKind::None,
                  "data cache prefetch is not supported");
    static_assert(!std::is_same_v<BDataType, pk_int4_t>, "Not implemented");

    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeA()
    {
        return 1;
    }
    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeB()
    {
        return 1;
    }
    static constexpr index_t GetVectorSizeC() { return 1; }

    static constexpr index_t GetSmemPackA() { return Policy::template GetSmemPackA<Problem>(); }
    static constexpr index_t GetSmemPackB() { return Policy::template GetSmemPackB<Problem>(); }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return NumSlots * Policy::template GetSmemSize<Problem>();
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "pipeline_AgBgCrCompTDMRunAhead",
                      concat('x', MPerBlock, NPerBlock, KPerBlock), BlockSize,
                      concat('x', BlockGemmShape::BlockWarps::at(number<0>{}),
                                  BlockGemmShape::BlockWarps::at(number<1>{})),
                      concat('x', kPadM, kPadN, kPadK),
                      concat('x', NumSlots, RunAhead));
        // clang-format on
    }

    template <typename AsDramBlockWindowTmp,
              typename AElementFunction,
              typename BsDramBlockWindowTmp,
              typename BElementFunction>
    CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                   const AElementFunction& a_element_func,
                                   const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const BElementFunction& b_element_func,
                                   index_t num_loop,
                                   void* __restrict__ p_smem) const
    {
        static_assert(1 == std::tuple_size_v<AsDramBlockWindowTmp> &&
                          1 == std::tuple_size_v<BsDramBlockWindowTmp>,
                      "one A and one B tensor");
        using ADramBlockWindowTmp = remove_cvref_t<std::tuple_element_t<0, AsDramBlockWindowTmp>>;
        using BDramBlockWindowTmp = remove_cvref_t<std::tuple_element_t<0, BsDramBlockWindowTmp>>;

        static_assert(std::is_same_v<AElementFunction, element_wise::PassThrough> &&
                          std::is_same_v<BElementFunction, element_wise::PassThrough>,
                      "TDM moves global memory to LDS unmodified; no element-wise function");
        static_assert(
            std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                std::is_same_v<BDataType, remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
            "Data Type conflict on A and B matrix input data type.");

        constexpr bool is_a_col_major = std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
        constexpr bool is_b_row_major = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;
        static_assert(is_a_col_major
                          ? (KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                             MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<1>{}])
                          : (MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                             KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<1>{}]),
                      "A block window has incorrect lengths for defined ALayout!");
        static_assert(is_b_row_major
                          ? (KPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                             NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[number<1>{}])
                          : (NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                             KPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[number<1>{}]),
                      "B block window has incorrect lengths for defined BLayout!");

        ignore = a_element_func;
        ignore = b_element_func;

#if defined(__gfx125__)
        const PipelineImplBase impl{};
        const auto lds_views    = impl.template GetABLdsTensorViews<NumSlots>(p_smem);
        const auto& a_lds_views = lds_views.at(number<0>{});
        const auto& b_lds_views = lds_views.at(number<1>{});

        // Wave-uniform: every branch on it below is taken or skipped by a whole wave.
        const index_t warp_id = get_warp_id();
        const bool fills_a    = warp_id == FillerWaveA;
        const bool fills_b    = warp_id == FillerWaveB;

        auto a_filler = MakeFiller<true>(impl, a_dram_block_window_tmp[number<0>{}], a_lds_views);
        auto b_filler = MakeFiller<false>(impl, b_dram_block_window_tmp[number<0>{}], b_lds_views);

        // Issues the next unfilled step's transfers into `slot`. Steps are filled strictly in
        // order: each call advances the DRAM windows by one K block.
        auto fill = [&](auto slot) {
            if(fills_a)
                a_filler(slot);
            if(fills_b)
                b_filler(slot);
        };

        // Leaves at most `ahead` steps of this wave's fills in flight. TENSORcnt counts tensor
        // ops, not steps, and retires them in issue order, so "at most ahead * FillsPerStep
        // outstanding" means "every step up to the oldest unretired one has landed". The only
        // tensor ops a filling wave has outstanding in the K loop are its own fills: data cache
        // prefetch and cluster multicast are rejected above, and on a persistent kernel the
        // previous tile's TdmEpilogue retires its tensor_store (s_wait_tensorcnt<0>) before the
        // next tile primes. The immediate must be a constant; `ahead` is wave-uniform. `ahead` is
        // in [0, RunAhead) by construction; anything else fails closed (waits for every fill)
        // rather than open (waits for none and publishes an unlanded slot).
        auto retire = [&](index_t ahead) {
            static_for<0, RunAhead, 1>{}([&](auto n) {
                if(ahead == n.value || (n.value == 0 && (ahead < 0 || ahead >= RunAhead)))
                    s_wait_tensorcnt<n.value * FillsPerStep>();
            });
        };

        // One read window per slot; every step leaves its slot's window where it found it.
        auto a_windows = generate_tuple(
            [&](auto slot) {
                return impl.MakeALdsWindows(a_lds_views[slot], ALdsLoadTileDistr{}).at(number<1>{});
            },
            number<NumSlots>{});
        auto b_windows = generate_tuple(
            [&](auto slot) {
                return impl.MakeBLdsWindows(b_lds_views[slot], BLdsLoadTileDistr{}).at(number<1>{});
            },
            number<NumSlots>{});

        // K sub-tiles double-buffer through registers: one is read while the other multiplies.
        using ALdsTile = decltype(make_static_distributed_tensor<ADataType>(ALdsLoadTileDistr{}));
        using BLdsTile = decltype(make_static_distributed_tensor<BDataType>(BLdsLoadTileDistr{}));
        ALdsTile a_tiles[2];
        BLdsTile b_tiles[2];

        auto block_gemm   = BlockGemm();
        auto c_block_tile = block_gemm.MakeCBlockTile();
        clear_tile(c_block_tile);

        if constexpr(KSubTileNum % 2 == 0)
        {
            // Step k + 1's first sub-tile goes to register buffer 0 while step k's last
            // sub-tile, in buffer (KSubTileNum - 1) % 2 == 1, multiplies.
            // Issues the LDS loads of `slot`'s first K sub-tile into register buffer 0.
            auto prefetch_first = [&](auto slot) {
                block_gemm.template LocalPrefetch<WindowSlideMode::Move>(a_tiles[0],
                                                                         b_tiles[0],
                                                                         a_windows[slot],
                                                                         b_windows[slot],
                                                                         is_a_load_tr_v,
                                                                         is_b_load_tr_v);
            };

            // Makes step `step` (in `slot`) readable by every wave and refills the slot it frees.
            // Every wave must already have ISSUED all of its LDS reads of step - 1: the drain below
            // completes them, so after the barrier the slot step + RunAhead reuses (last read at
            // step + RunAhead - NumSlots <= step - 1) is free.
            auto handoff = [&](auto slot, index_t step) {
                if(fills_a || fills_b)
                    retire(max(min(RunAhead - 1, num_loop - 1 - step), 0));
                s_wait_dscnt<0>();
                StepBarrier();
                // Every wave's first loads go out before a filling wave's descriptor arithmetic.
                prefetch_first(slot);
                if(step + RunAhead < num_loop)
                    fill(number<(slot.value + RunAhead) % NumSlots>{});
            };

            // Prime: the first RunAhead steps go out before any barrier.
            static_for<0, RunAhead, 1>{}([&](auto s) {
                if(s.value < num_loop)
                    fill(s);
            });

            if(num_loop > 0)
            {
                handoff(number<0>{}, 0);
            }

            // Unrolled by NumSlots so every slot index is a compile-time constant; step k always
            // uses slot k % NumSlots because the ring starts at slot 0. Step k's first sub-tile is
            // already in flight on entry; step k + 1's is issued before step k's last WMMAs.
            index_t k = 0;
            while(k < num_loop)
            {
                static_for<0, NumSlots, 1>{}([&](auto slot) {
                    if(k < num_loop)
                    {
                        auto& a_window = a_windows[slot];
                        auto& b_window = b_windows[slot];
                        static_for<0, KSubTileNum - 1, 1>{}([&](auto i) {
                            constexpr index_t kNext = (i.value + 1) % 2;
                            constexpr auto kSlideTo = (i.value + 1 == KSubTileNum - 1)
                                                          ? WindowSlideMode::Reset
                                                          : WindowSlideMode::Move;
                            block_gemm.template LocalPrefetch<kSlideTo>(a_tiles[kNext],
                                                                        b_tiles[kNext],
                                                                        a_window,
                                                                        b_window,
                                                                        is_a_load_tr_v,
                                                                        is_b_load_tr_v);
                            __builtin_amdgcn_sched_barrier(0);
                            block_gemm(c_block_tile, a_tiles[i.value % 2], b_tiles[i.value % 2]);
                        });

                        // Every LDS read of step k is issued: hand off step k + 1 and pin its first
                        // sub-tile's loads ahead of step k's last WMMAs.
                        // Unconditional on purpose: after the last step this is a harmless read of
                        // an idle slot, and a branch here would merge the waitcnt state of a path
                        // with step k's last loads still pending, draining the prefetch early.
                        constexpr auto next = number<(slot.value + 1) % NumSlots>{};
                        handoff(next, k + 1);
                        __builtin_amdgcn_sched_barrier(0);
                        block_gemm(c_block_tile,
                                   a_tiles[(KSubTileNum - 1) % 2],
                                   b_tiles[(KSubTileNum - 1) % 2]);
                        ++k;
                    }
                });
            }
        }
        else
        {
            // The math of one K step, done by every wave: multiply the slot, then retire its reads.
            auto consume = [&](auto slot) {
                auto& a_window = a_windows[slot];
                auto& b_window = b_windows[slot];

                block_gemm.template LocalPrefetch<KSubTileNum == 1 ? WindowSlideMode::Stay
                                                                   : WindowSlideMode::Move>(
                    a_tiles[0], b_tiles[0], a_window, b_window, is_a_load_tr_v, is_b_load_tr_v);
                static_for<0, KSubTileNum - 1, 1>{}([&](auto i) {
                    constexpr index_t kNext = (i.value + 1) % 2;
                    constexpr auto kSlideTo = (i.value + 1 == KSubTileNum - 1)
                                                  ? WindowSlideMode::Reset
                                                  : WindowSlideMode::Move;
                    block_gemm.template LocalPrefetch<kSlideTo>(a_tiles[kNext],
                                                                b_tiles[kNext],
                                                                a_window,
                                                                b_window,
                                                                is_a_load_tr_v,
                                                                is_b_load_tr_v);
                    // Pin the next sub-tile's loads ahead of this sub-tile's WMMAs. Unpinned, the
                    // scheduler sinks every fragment's loads to just before its first use, and the
                    // waitcnt pass then drains the whole LDS queue before almost every WMMA; with
                    // one wave per SIMD there is no sibling to hide those drains behind.
                    __builtin_amdgcn_sched_barrier(0);
                    block_gemm(c_block_tile, a_tiles[i.value % 2], b_tiles[i.value % 2]);
                });
                block_gemm(
                    c_block_tile, a_tiles[(KSubTileNum - 1) % 2], b_tiles[(KSubTileNum - 1) % 2]);

                // The next StepBarrier may hand this slot to a fill: retire its reads first.
                s_wait_dscnt<0>();
            };

            // Prime: the first RunAhead steps go out before any barrier.
            static_for<0, RunAhead, 1>{}([&](auto s) {
                if(s.value < num_loop)
                    fill(s);
            });

            // Unrolled by NumSlots so every slot index is a compile-time constant; step k always
            // uses slot k % NumSlots because the ring starts at slot 0.
            index_t k = 0;
            while(k < num_loop)
            {
                static_for<0, NumSlots, 1>{}([&](auto slot) {
                    if(k < num_loop)
                    {
                        if(fills_a || fills_b)
                            retire(min(RunAhead - 1, num_loop - 1 - k));

                        StepBarrier();

                        if(k + RunAhead < num_loop)
                            fill(number<(slot.value + RunAhead) % NumSlots>{});

                        consume(slot);
                        ++k;
                    }
                });
            }
        }

        // The epilogue stages C through the same LDS: nobody may still be reading the ring.
        StepBarrier();

        return c_block_tile;
#else
        ignore = a_dram_block_window_tmp;
        ignore = b_dram_block_window_tmp;
        ignore = num_loop;
        ignore = p_smem;
        return decltype(BlockGemm::MakeCBlockTile()){};
#endif
    }

    private:
    // A fenced workgroup barrier, built like __syncthreads(). The release fence makes the barrier
    // wait for this wave's own LDS reads of the slot about to be reused; the acquire fence keeps
    // the compiler from hoisting the next slot's LDS reads above the barrier. Neither covers a
    // tensor transfer, which is why a filling wave retires its fills before arriving.
    CK_TILE_DEVICE static void StepBarrier()
    {
#if defined(__gfx125__)
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
        __builtin_amdgcn_s_barrier_signal(-1);
        __builtin_amdgcn_s_barrier_wait(-1);
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
#endif
    }

    // One operand's fill: the DRAM window it walks and the LDS copy window of every slot.
    // Calling it with a slot issues one TENSOR_LOAD_TO_LDS (NumCoord == 1 on the non-gather
    // path, tile_window.hpp) and advances the DRAM window by one K block.
    template <bool IsA, typename DramBlockWindowTmp, typename LdsViews>
    CK_TILE_DEVICE static auto MakeFiller(const PipelineImplBase& impl,
                                          const DramBlockWindowTmp& dram_block_window_tmp,
                                          const LdsViews& lds_views)
    {
        auto dram_window = [&]() {
            if constexpr(IsA)
                return impl.CopyADramWindow(dram_block_window_tmp);
            else
                return impl.CopyBDramWindow(dram_block_window_tmp);
        }();

        // Only the copy windows are used: TDM needs no per-thread distribution in LDS.
        auto lds_windows = generate_tuple(
            [&](auto slot) {
                if constexpr(IsA)
                    return impl.MakeALdsWindows(lds_views[slot], ALdsLoadTileDistr{})
                        .at(number<0>{});
                else
                    return impl.MakeBLdsWindows(lds_views[slot], BLdsLoadTileDistr{})
                        .at(number<0>{});
            },
            number<NumSlots>{});

        // Col-major A and row-major B windows are (K, M|N): K is their outer, strided index.
        constexpr bool k_is_outer = IsA ? std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>
                                        : std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;
        using DramStep            = typename decltype(dram_window)::BottomTensorIndex;
        constexpr DramStep k_step =
            k_is_outer ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);

        return [impl,
                dram_window,
                lds_windows,
                k_step,
                tdm_config = Policy::template MakeTdmConfig<Problem, IsA>()](auto slot) mutable {
            impl.GlobalPrefetchTDM(tdm_config, lds_windows[slot], dram_window, k_step);
        };
    }

    using ALdsLoadTileDistr =
        decltype(make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode()));
    using BLdsLoadTileDistr =
        decltype(make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode()));
};

} // namespace ck_tile
