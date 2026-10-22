// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_tdm_producer_consumer_policy.hpp"

namespace ck_tile {

/**
 * @brief TDM GEMM pipeline whose loader waves run ahead of its compute waves, synchronised over
 * hardware named barriers (gfx1250).
 *
 * The workgroup is split by role. Waves [0, NumConsumerWaves) are consumers: BlockGemm maps the
 * C tile onto them exactly as in any other pipeline, and they only read LDS and multiply. The
 * next two waves are producers, one per operand, and only issue that operand's TDM transfers.
 * Between them sits a ring of NumSlots LDS buffers:
 *
 *     producer A --TDM--> +--------+--------+--------+
 *                         | slot 0 | slot 1 | slot 2 | --ds_read--> consumers --> C tile
 *     producer B --TDM--> +--------+--------+--------+
 *
 * A producer waits only until the slot it is about to overwrite has been drained, and the
 * consumers only until the slot they are about to read has been filled, so the loaders run up
 * to NumSlots K steps ahead instead of meeting the math at a workgroup barrier every step, as
 * the symmetric TDM pipelines do. The handshake itself lives in named_barrier.hpp; this pipeline
 * supplies only what a step does on each side.
 *
 * The kernel owns the barriers: it declares `using barrier_pipeline = BarrierPipeline;`, passes
 * the token of BarrierPipeline::init<Kernel>() to operator(), launches LaunchBlockSize threads
 * and runs the epilogue on the consumer waves only. UniversalGemmKernel does all of this for any
 * pipeline exposing these members. The producers return an empty accumulator.
 *
 * Other targets compile an inert fallback that returns zeros, so that multi-target builds still
 * compile; UniversalGemmKernel::IsSupportedArgument rejects those devices.
 */
template <typename Problem, typename Policy = GemmPipelineAgBgCrCompTDMProducerConsumerPolicy<>>
struct GemmPipelineAgBgCrCompTDMProducerConsumer
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

    static constexpr index_t NumSlots   = Policy::NumSlots;
    static constexpr index_t PublishLag = Policy::PublishLag;

    /// BlockGemm maps the C tile onto these waves; the producers are launched in addition.
    static constexpr index_t NumConsumerWaves = BlockGemmShape::NumWarps;
    /// One producer per operand: wave NumConsumerWaves loads A, the wave after it loads B.
    static constexpr index_t NumProducerWaves = 2;
    // The role dispatch in operator() maps producer wave P to operand P, so this pipeline
    // supports exactly one loader per operand. Raising this alone duplicates an operand's
    // producer and leaves a role unplayed: the duplicated role's FREE barrier over-signals
    // (corrupting the next generation) while the orphaned one never completes -- silent wrong
    // results plus a hang, with no fault. It needs the operand split (kOperand = P % 2) and the
    // per-operand tile partition that goes with it. The policy's barrier-budget note assumes
    // this value too.
    static_assert(NumProducerWaves == 2,
                  "the role dispatch in operator() maps producer wave P to operand P: this "
                  "pipeline supports exactly one loader per operand");

    /// The named barriers the kernel must arm for this pipeline.
    using BarrierPipeline =
        named_barrier_pipeline<ring_spec<NumSlots, NumProducerWaves, NumConsumerWaves>>;

    // Wavelet interface: BlockGemm and the epilogue see BlockSize; the kernel launches more.
    static constexpr index_t BlockSize = Problem::kBlockSize;
    static constexpr index_t LaunchBlockSize =
        (NumConsumerWaves + NumProducerWaves) * get_warp_size();
    // The producer waves launch on top of BlockGemm's, so a BlockWarps shape that fits an
    // ordinary pipeline can overflow here, and nothing downstream rejects it: BlockSize() only
    // converts units (it halves on wave32), it never validates, so the sole symptom is an opaque
    // launch failure naming neither this pipeline nor the BlockWarps that caused it.
    //
    // Asserted in the DEVICE pass only. get_warp_size() reports the wave size of the CURRENT
    // COMPILER PASS: 32 here, but 64 in the host pass, where amdgcn_target_wave_size_id::HOST is
    // defined as 64. So the host pass sees twice the real workgroup and would reject valid
    // shapes -- [4, 4, 1] is 18 waves = 576 threads, which the host pass scores as 1152. Only
    // the device pass carries the real width. Spelled as an implication rather than an #if to
    // match the pattern arch.hpp uses for its own target sanity checks.
    static_assert(!ck_tile::core::amdgcn_compiler_target_state::CK_TILE_DEVICE_COMPILE ||
                      LaunchBlockSize <= 1024,
                  "workgroup exceeds the hardware thread limit: reduce BlockWarps, since the "
                  "producer waves are launched in addition to BlockGemm's consumer waves");
    static constexpr bool IsWavelet = true;

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr index_t NumWaveGroups    = Problem::NumWaveGroups;
    static constexpr index_t Preshuffle       = Problem::Preshuffle;
    static constexpr bool UsePersistentKernel = Problem::Traits::UsePersistentKernel;

    // TDM vectorises internally and handles ragged extents, so launch checks do not apply.
    static constexpr bool skipCheckValidLaunchParams = true;

    static constexpr index_t KSubTileNum = Policy::template GetPipelineSubTileNum<Problem>().value;

    static constexpr auto is_a_load_tr_v = bool_constant<PipelineImplBase::is_a_load_tr>{};
    static constexpr auto is_b_load_tr_v = bool_constant<PipelineImplBase::is_b_load_tr>{};

    static_assert(NumWaveGroups == 1, "the producer/consumer split replaces wave groups");
    static_assert(!Preshuffle, "preshuffled B is not supported");
    static_assert(!kPadM && !kPadN && !kPadK,
                  "padding is not supported: TDM handles ragged extents itself, and padded "
                  "descriptor lengths would carry its transfers past the tensor");
    static_assert(!Policy::template isClusterLaunch<Problem>(),
                  "cluster multicast is not supported: a peer's transfer into this workgroup's "
                  "LDS is invisible to the tensor counter a producer publishes on");
    static_assert(Policy::DataCachePrefetchA == DataCachePrefetchKind::None &&
                      Policy::DataCachePrefetchB == DataCachePrefetchKind::None,
                  "data cache prefetch is not supported");
    static_assert(!std::is_same_v<BDataType, pk_int4_t>, "Not implemented");

    CK_TILE_DEVICE static bool IsMathWave() { return get_warp_id() < NumConsumerWaves; }

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
        // The ring triple ends in NumProducerWaves because only BlockWarps[0] and [1] are
        // emitted below, so consumers-plus-producers cannot be recovered from the rest of the
        // name. Forward protection, not a live collision: tile_engine has no NumSlots/PublishLag
        // config axis today, so every generated comp_tdm_pc instance ships one ring shape. If
        // that changes, the ring triple belongs in _format_trait_combo_string -- the one string
        // that reaches both the CMake target name and the generated header path -- and not in
        // _format_kernel_name, which neither of those two derives from.
        // clang-format off
        return concat('_', "pipeline_AgBgCrCompTDMProducerConsumer",
                      concat('x', MPerBlock, NPerBlock, KPerBlock), LaunchBlockSize,
                      concat('x', BlockGemmShape::BlockWarps::at(number<0>{}),
                                  BlockGemmShape::BlockWarps::at(number<1>{})),
                      concat('x', kPadM, kPadN, kPadK),
                      concat('x', NumSlots, PublishLag, NumProducerWaves));
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
                                   void* __restrict__ p_smem,
                                   typename BarrierPipeline::token barriers) const
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

        using CBlockTile = decltype(BlockGemm::MakeCBlockTile());

        if constexpr(BarrierPipeline::kIsSupported)
        {
            const auto lds_views =
                PipelineImplBase{}.template GetABLdsTensorViews<NumSlots>(p_smem);

            // Wave-uniform: every ring call below executes once per wave, whatever EXEC holds.
            const index_t warp_id = get_warp_id();
            if(warp_id < NumConsumerWaves)
            {
                return RunConsumers(
                    lds_views.at(number<0>{}), lds_views.at(number<1>{}), num_loop, barriers);
            }
            // Exactly one wave plays each producer<P>. The tests below are on disjoint warp_id
            // values and cover [NumConsumerWaves, NumConsumerWaves + NumProducerWaves)
            // exhaustively, with no fallthrough: a wave launched past that range does nothing
            // rather than becoming a second producer<NumProducerWaves - 1>.
            //
            // Why that matters, in signal counts: a DUPLICATED role over-signals its FREE[p][s]
            // -- NumConsumerWaves releases plus a join+signal from each of the two waves, i.e.
            // kFreeMemberCount + 1 per step -- so it completes a generation early and the two
            // waves drift out of phase until one waits on a generation nobody completes. The
            // ORPHANED role's FREE[p][s] is merely inert, since nobody blocks on it. DATA[s]
            // stays at exactly kDataMemberCount throughout, so the first observable is not the
            // hang but silent wrong results: the orphaned operand is never transferred and the
            // consumers multiply stale LDS. Debug accordingly.
            // See the "Producer<P> is a role, not an identity" note in named_barrier.hpp.
            static_for<0, NumProducerWaves, 1>{}([&](auto p) {
                if(warp_id == NumConsumerWaves + p.value)
                {
                    // kProducer is the ring role; kOperand is which matrix it streams. They
                    // coincide while NumProducerWaves == 2 and separate when the operand split
                    // lands, so they are named apart to keep that change local to here. The two
                    // arms are exhaustive because the NumProducerWaves == 2 assert above bounds
                    // kOperand to {0, 1}; that assert fires first, and names the cause more
                    // precisely, if the count is ever raised without the operand split.
                    constexpr index_t kProducer = p.value;
                    constexpr index_t kOperand  = p.value;
                    if constexpr(kOperand == 0)
                    {
                        RunProducer<kProducer, 0>(a_dram_block_window_tmp[number<0>{}],
                                                  lds_views.at(number<0>{}),
                                                  num_loop,
                                                  barriers);
                    }
                    else
                    {
                        RunProducer<kProducer, 1>(b_dram_block_window_tmp[number<0>{}],
                                                  lds_views.at(number<1>{}),
                                                  num_loop,
                                                  barriers);
                    }
                }
            });
            return CBlockTile{};
        }
        else
        {
            ignore = a_dram_block_window_tmp;
            ignore = b_dram_block_window_tmp;
            ignore = num_loop;
            ignore = p_smem;
            ignore = barriers;
            return CBlockTile{};
        }
    }

    private:
    template <bool IsA>
    CK_TILE_DEVICE static TDMConfig MakeTdmConfig()
    {
        constexpr auto padding = Policy::template GetLdsPaddingConfig<Problem, IsA>();

        TDMConfig config;
        config.pad_enable              = padding[number<0>{}];
        config.pad_config.pad_amount   = padding[number<1>{}];
        config.pad_config.pad_interval = padding[number<2>{}];
        // workgroup_mask stays 0: a non-zero mask makes the transfer wait for cluster peers.
        return config;
    }

    // One producer wave: streams operand Operand (0 = A, 1 = B) into the ring, one TDM per step.
    // Producer is the ring ROLE this wave plays; Operand is which matrix it streams. The two
    // coincide while NumProducerWaves == 2, but they are distinct concepts -- see "Producer<P>
    // is a role, not an identity" in named_barrier.hpp -- and keeping them apart confines the
    // operand split (several producers cooperating on one matrix) to the dispatch in operator().
    template <index_t Producer, index_t Operand, typename DramBlockWindowTmp, typename LdsViews>
    CK_TILE_DEVICE static void RunProducer(const DramBlockWindowTmp& dram_block_window_tmp,
                                           const LdsViews& lds_views,
                                           index_t num_loop,
                                           typename BarrierPipeline::token barriers)
    {
        constexpr bool IsA = (Operand == 0);
        const PipelineImplBase impl{};

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

        const TDMConfig tdm_config = MakeTdmConfig<IsA>();

        using Ring = typename BarrierPipeline::template ring<0>;
        Ring::template producer<Producer>::template run<PublishLag>(
            barriers,
            num_loop,
            [&](auto slot, index_t) {
                impl.GlobalPrefetchTDM(tdm_config, lds_windows[slot], dram_window, k_step);
            },
            // retire(N) must leave at most N of THIS WAVE's fills in flight, and this is the
            // only thing that covers them. s_wait_tensorcnt counts tensor ops, not fills, so the
            // identity rests on three facts, none of them local:
            //   1. one fill == one TENSOR_LOAD_TO_LDS. GlobalPrefetchTDM issues NumCoord of them
            //      on the non-gather path, and tile_window.hpp pins NumCoord == 1. Passing a
            //      gather index view here takes the NumIterations path and breaks this.
            //   2. a producer wave fills one operand only, so every tensor op it has in flight
            //      is one of its own fills.
            //   3. TENSORcnt retires in issue order, so "at most N outstanding" means
            //      "fills 0..k-N have landed".
            // Break any of the three and the producer publishes a slot whose transfer has not
            // landed: silent wrong results, no fault, no hang.
            [](auto in_flight) { s_wait_tensorcnt<decltype(in_flight)::value>(); });
    }

    // The consumer waves: multiply each slot as it fills, and hand it back once read.
    template <typename ALdsViews, typename BLdsViews>
    CK_TILE_DEVICE static auto RunConsumers(const ALdsViews& a_lds_views,
                                            const BLdsViews& b_lds_views,
                                            index_t num_loop,
                                            typename BarrierPipeline::token barriers)
    {
        const PipelineImplBase impl{};

        // One window per slot; every step leaves its slot's window where it found it.
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

        using Ring = typename BarrierPipeline::template ring<0>;
        Ring::consumer::run(barriers, num_loop, [&](auto slot, index_t) {
            auto& a_window = a_windows[slot];
            auto& b_window = b_windows[slot];

            block_gemm.template LocalPrefetch<KSubTileNum == 1 ? WindowSlideMode::Stay
                                                               : WindowSlideMode::Move>(
                a_tiles[0], b_tiles[0], a_window, b_window, is_a_load_tr_v, is_b_load_tr_v);
            static_for<0, KSubTileNum - 1, 1>{}([&](auto i) {
                constexpr index_t kNext = (i.value + 1) % 2;
                constexpr auto kSlideTo = (i.value + 1 == KSubTileNum - 1) ? WindowSlideMode::Reset
                                                                           : WindowSlideMode::Move;
                block_gemm.template LocalPrefetch<kSlideTo>(a_tiles[kNext],
                                                            b_tiles[kNext],
                                                            a_window,
                                                            b_window,
                                                            is_a_load_tr_v,
                                                            is_b_load_tr_v);
                block_gemm(c_block_tile, a_tiles[i.value % 2], b_tiles[i.value % 2]);
            });
            block_gemm(
                c_block_tile, a_tiles[(KSubTileNum - 1) % 2], b_tiles[(KSubTileNum - 1) % 2]);

            // The ring releases the slot the moment this returns, so retire its reads first.
            s_wait_dscnt<0>();
        });
        return c_block_tile;
    }

    using ALdsLoadTileDistr =
        decltype(make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode()));
    using BLdsLoadTileDistr =
        decltype(make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode()));
};

} // namespace ck_tile
