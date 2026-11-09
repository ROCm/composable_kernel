// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_fmha_lse.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_pingpong_policy.hpp"

namespace ck_tile {

/**
 * @brief How qr_tdm_pingpong schedules the LDS reads of the main-loop gemm operands
 *        (K for the QK gemm, V for the PV gemm). The code generator picks one per head
 *        dimension; both default to off, the read-ahead of kReadAheadWmma.
 *
 * @tparam PrefetchNextSliceMinRows_ Tiles of at least this many query rows (64 or 128;
 *                            0: none) prefetch the next slice: while the WMMAs of one kK0
 *                            (QK) or kK1 (PV) slice run, the reads of the next slice are
 *                            already in flight; each slice is its own scheduling region,
 *                            so the compiler cannot pull a WMMA ahead onto a read issued
 *                            just before it. Costs a second slice of B fragments in VGPRs,
 *                            which can drop a smaller tile's occupancy.
 * @tparam ReadVBeforeSoftmaxMinRows_ Tiles of at least this many query rows (0: none) with
 *                            two V buffers, where V_i is in LDS from the loop head on, read
 *                            its first kK1 slice right after the QK gemm, so the softmax
 *                            covers the read instead of the PV gemm waiting on it. Costs
 *                            that slice's B fragments in VGPRs across the softmax.
 */
template <index_t PrefetchNextSliceMinRows_ = 0, index_t ReadVBeforeSoftmaxMinRows_ = 0>
struct QrTdmPingPongLdsReadSchedule
{
    static constexpr index_t kPrefetchNextSliceMinRows  = PrefetchNextSliceMinRows_;
    static constexpr index_t kReadVBeforeSoftmaxMinRows = ReadVBeforeSoftmaxMinRows_;
};

// qr_tdm with a whole-tile K ping/pong, for gfx1250 fp16/bf16: trip i reads K buffer
// (i & 1) while the TDM fills the other with tile i + 1. V streams through one buffer
// under the QK gemm, or, where is_qr_tdm_pingpong_double_v_v holds, through two with
// V_{i+1} issued a whole trip ahead. It shares qr_tdm's problem, gemms and
// register layouts; only the LDS layout (see BlockFmhaPipelineQRKSVSTdmPingPongPolicy)
// and the main loop differ.
template <typename Problem_,
          typename Policy_          = BlockFmhaPipelineQRKSVSTdmPingPongPolicy,
          typename LdsReadSchedule_ = QrTdmPingPongLdsReadSchedule<>>
struct BlockFmhaPipelineQRKSVSTdmPingPong : BlockFmhaPipelineQRKSVSTdm<Problem_, Policy_>
{
    using Base            = BlockFmhaPipelineQRKSVSTdm<Problem_, Policy_>;
    using LdsReadSchedule = remove_cvref_t<LdsReadSchedule_>;

    using typename Base::BlockFmhaShape;
    using typename Base::FmhaMask;
    using typename Base::KDataType;
    using typename Base::LSEDataType;
    using typename Base::PDataType;
    using typename Base::Policy;
    using typename Base::Problem;
    using typename Base::QDataType;
    using typename Base::RandValOutputDataType;
    using typename Base::SaccDataType;
    using typename Base::SMPLComputeDataType;
    using typename Base::VDataType;

    using Base::BiasEnum;
    using Base::I0;
    using Base::I1;
    using Base::kHasLogitsSoftCap;
    using Base::kHasSink;
    using Base::kHasUnevenSplits;
    using Base::kK0;
    using Base::kK1;
    using Base::kM0;
    using Base::kN0;
    using Base::kN1;
    using Base::kNWarp;
    using Base::kPadSeqLenK;
    using Base::kQKHeaddim;
    using Base::kQuantized;
    using Base::kStoreLSE;
    using Base::kSubQKHeaddim;

    using Base::LoadVTile;
    using Base::ReduceRowSync;
    using Base::VReadLengths;
    using Base::VSeqOffset;

    // softmax, or GDPA (see qr_tdm): P = act(scale_s * S), no row max/sum, no normalization
    using Base::Activation;
    using Base::ApplyActivation;
    using Base::kIsSoftmax;
    using Base::kMaskedScore;

    static constexpr const char* name = "qr_tdm_pingpong";

    // the kernel paths qr_tdm takes by name
    static constexpr bool kUsesUntransposedVKernelPath = true;
    static constexpr bool kUsesTdmAffineDramPath       = true;

    static constexpr bool kKLoadOnce = true;

    static_assert(detail::is_qr_tdm_pingpong_qkv_v<Problem> && !kQuantized,
                  "qr_tdm_pingpong: fp16/bf16 Q/K/V without quantization scales");
    static_assert(!Problem::kUseDoubleKVLdsBuffer && !Problem::kProgressiveDsLoadK,
                  "qr_tdm_pingpong has its own K/V buffering");

    static constexpr bool kDoubleV = detail::is_qr_tdm_pingpong_double_v_v<Problem>;

    // B-operand read-ahead of the main-loop gemms, in WMMAs. Where the waves of a SIMD
    // cannot cover each other's LDS read latency, the reads run three B fragments ahead:
    // the d128 128-row tile and the d256/d512 64-row tiles, capped at three (two at d512)
    // waves per SIMD by their O accumulators (gfx1250: d128 x 5252 keys +8%, d256/d512
    // +5..14%), and the masked 128-row tiles, whose per-tile mask work stretches the trip
    // (d64 causal x 1859 keys: +6%). The other tiles keep one fragment: deeper read-ahead
    // costs them VGPRs they need for occupancy.
    static constexpr index_t kReadAheadWmma =
        (BlockFmhaShape::kM0 >= 128 && (kSubQKHeaddim >= 128 || FmhaMask::IsMasking)) ||
                (BlockFmhaShape::kM0 >= 64 && kSubQKHeaddim >= 256)
            ? 6
            : 2;
    static_assert(kSubQKHeaddim == kQKHeaddim, "qr_tdm_pingpong stages whole K rows");
    // BlockDropout::Run walks P in the QK gemm's output layout
    static_assert(!Base::kHasDropout || kNWarp == 1,
                  "qr_tdm_pingpong: dropout needs the QK gemm warps on M");

    // The row sum of P: on the matrix unit when one gemm_1 warp tile of columns spans a
    // row, i.e. the gemm_1 warps are all on M; the short-query tile spends them on N and
    // takes an fp32 sum. GDPA has no row sum, so its main loop schedules no row-sum
    // WMMAs (kRowSumWmma).
    static constexpr bool kRowSumOnMatrixUnit =
        BlockFmhaShape::Gemm1BlockWarps::at(I1) == 1 && !Base::kHasDropout && kIsSoftmax;
    // Without a per-tile row max (whole rows in one warp, the tiles with the matrix-unit
    // row sum): P is taken against the running base m_base and its fp32 row sum, which l
    // and the LSE need anyway, bounds every p. A tile whose sum stays at most
    // 2^kRebaseThreshold per thread needs no row max (v_max3 and a cross-lane sync ahead
    // of the exps). The rare tile over it, or with a row that has no base yet, goes the
    // row max way, the former taking S again from the K tile still in LDS. l and o_acc
    // then sum the same fp32 P with and without the LSE, and the LSE store is all the
    // LSE adds. Each thread keeps its part of l; the lanes are synced once, after the
    // main loop.
    static constexpr bool kRowMaxBySum = kRowSumOnMatrixUnit && kNWarp == 1;
    // A mask whose rows start their keys in later tiles (a window) leaves rows without a
    // base for most of a row's few tiles: those tiles take the row max anyway, and the
    // check only costs (gfx1250 d128 window 127 x 2048 keys: +3%); the row max then stays.
    static constexpr bool kMaskStartsRowsLate = []() {
        if constexpr(!FmhaMask::IsMasking)
            return false;
        else if constexpr(requires { FmhaMask::IsLocal; })
            return FmhaMask::IsLocal;
        else
            return true;
    }();
    // A bias keeps the row max as well: its per-score work leaves the check little to save,
    // and the fallback's second bias pass took the d160 ALiBi tile over 512 VGPRs.
    static constexpr bool kSkipRowMax =
        kRowMaxBySum && !kMaskStartsRowsLate && BiasEnum == BlockAttentionBiasEnum::NO_BIAS;
    // The short-query tile keeps the whole-gemm read-ahead: its slices are a single
    // WMMA row. Without the row max the 128-row tiles take the slice prefetch at two
    // waves per SIMD, K two slices ahead (gfx1250 d128 512 x 5252 keys, with the row
    // max: 1.77 ms, without, three waves: 1.88 ms, two waves with K two ahead: 1.75 ms).
    static constexpr bool kPrefetchNextSlice =
        BlockFmhaShape::kM0 >= 64 &&
        ((kSkipRowMax && BlockFmhaShape::kM0 >= 128) ||
         (LdsReadSchedule::kPrefetchNextSliceMinRows > 0 &&
          BlockFmhaShape::kM0 >= LdsReadSchedule::kPrefetchNextSliceMinRows));
    static constexpr bool kKReadsTwoSlicesAhead =
        kPrefetchNextSlice && kSkipRowMax && kQKHeaddim / kK0 >= 3;
    static constexpr bool kReadVBeforeSoftmax =
        LdsReadSchedule::kReadVBeforeSoftmaxMinRows > 0 &&
        BlockFmhaShape::kM0 >= LdsReadSchedule::kReadVBeforeSoftmaxMinRows&& kDoubleV;
    // P goes to 16 bits right after the exps, so the fp32 P is gone before the bound
    // check; only the slice prefetch schedule with the check converts under the PV WMMAs
    // (gfx1250 d96 window 1024 keys: early 332 VGPRs, three waves; late 341, two).
    static constexpr bool kEarlyPConvert = kRowMaxBySum && (!kSkipRowMax || !kPrefetchNextSlice);
    // Without the row max the 128-row tiles of d32..d96 keep the waves per SIMD they have
    // with it (the bound check and its fallback otherwise leave the compiler a few VGPRs
    // over: d64 512 x 5252 keys three waves 1.045 ms, four 1.000 ms; d96 causal 1024 keys
    // two waves +6%, d96 window with the LSE two waves +5%); d128 and up take two waves
    // with the slice prefetch. The masked d32 tile spills at four (and keeps its waves
    // without).
    static constexpr index_t kBlockPerCu = []() {
        if constexpr(kSkipRowMax && BlockFmhaShape::kM0 >= 128 && kQKHeaddim <= 64 &&
                     (kQKHeaddim > 32 || !FmhaMask::IsMasking))
            return index_t{4};
        else if constexpr(kRowMaxBySum && BlockFmhaShape::kM0 >= 128 && kQKHeaddim <= 96)
            return index_t{3};
        else
            return Base::kBlockPerCu;
    }();

    // Whether a row's elements sit in the thread buffer as neighbouring pairs (2k, 2k+1).
    template <typename PComputeTensor, typename Idx0, typename Ids>
    CK_TILE_HOST_DEVICE static constexpr bool RowPairsAdjacent(Idx0, Ids)
    {
        constexpr index_t kN = Ids::size();
        array<index_t, kN> o{};
        static_for<0, kN, 1>{}([&](auto i) {
            constexpr auto y_idx =
                PComputeTensor::get_tile_distribution().get_y_indices_from_distributed_indices(
                    make_tuple(Idx0{}, Ids{}.at(number<i>{})));
            o(i) = typename PComputeTensor::ThreadTensorDesc{}.calculate_offset(y_idx);
        });
        for(index_t i = 1; i < kN; ++i)
            for(index_t j = i; j > 0 && o[j - 1] > o[j]; --j)
            {
                const index_t t = o[j];
                o(j)            = o[j - 1];
                o(j - 1)        = t;
            }
        for(index_t k = 0; k < kN / 2; ++k)
            if(o[2 * k] % 2 != 0 || o[2 * k + 1] != o[2 * k] + 1)
                return false;
        return true;
    }

    // Row sum of P as a pairwise tree per thread, then (kSyncLanes) the cross-lane sync.
    // The serial fold of block_tile_reduce is a dependent v_add_f32 chain the
    // scheduler interleaves with the PV WMMAs, paying v_nop hazards for it; the
    // tree has depth log2(n) and pairs operands for v_pk_add_f32.
    template <bool kSyncLanes = true, typename PComputeTensor>
    CK_TILE_DEVICE static auto BlockRowSum(const PComputeTensor& p_compute)
    {
        auto rowsum = make_static_distributed_tensor<SMPLComputeDataType>(
            make_static_tile_distribution(detail::make_reduce_tile_distribution_encoding(
                PComputeTensor::get_tile_distribution().get_static_tile_distribution_encoding(),
                sequence<1>{})));

        constexpr auto spans = PComputeTensor::get_distributed_spans();
        // unpack the whole row span so its values are addressable by ordinal
        constexpr auto row_lengths = typename decltype(spans[I1])::Impl{};

        sweep_tile_span(spans[I0], [&](auto idx0) {
            sweep_tile_uspan(
                spans[I1],
                [&](auto... idx1) {
                    constexpr index_t kN = sizeof...(idx1);
                    static_assert(is_power_of_two_integer(kN), "row span must be a power of 2");

                    constexpr auto ids = make_tuple(remove_cvref_t<decltype(idx1)>{}...);

                    if constexpr(!kSyncLanes && kN >= 4 &&
                                 RowPairsAdjacent<PComputeTensor>(idx0, ids))
                    {
                        // the row's thread-buffer offsets in ascending order, summed as
                        // register pairs (v_pk_add_f32 of neighbouring registers)
                        constexpr auto offs = [&]() {
                            array<index_t, kN> o{};
                            static_for<0, kN, 1>{}([&](auto i) {
                                constexpr auto y_idx =
                                    PComputeTensor::get_tile_distribution()
                                        .get_y_indices_from_distributed_indices(
                                            make_tuple(idx0, ids.at(number<i>{})));
                                o(i) = typename PComputeTensor::ThreadTensorDesc{}.calculate_offset(
                                    y_idx);
                            });
                            for(index_t i = 1; i < kN; ++i)
                                for(index_t j = i; j > 0 && o[j - 1] > o[j]; --j)
                                {
                                    const index_t t = o[j];
                                    o(j)            = o[j - 1];
                                    o(j - 1)        = t;
                                }
                            return o;
                        }();
                        using f2        = ext_vector_t<float, 2>;
                        const auto& buf = p_compute.get_thread_buffer();
                        array<f2, kN / 4> w;
                        static_for<0, kN / 4, 1>{}([&](auto k) {
                            const f2 a{buf[number<offs[2 * k]>{}], buf[number<offs[2 * k + 1]>{}]};
                            const f2 b{buf[number<offs[2 * k + kN / 2]>{}],
                                       buf[number<offs[2 * k + 1 + kN / 2]>{}]};
                            w(k) = a + b;
                        });
                        static_for<0, integer_log2_floor(kN / 4), 1>{}([&](auto stage) {
                            constexpr index_t half = (kN / 4) >> (stage + 1);
                            static_for<0, half, 1>{}(
                                [&](auto k) { w(k) = w[k] + w[k + number<half>{}]; });
                        });
                        // opaque per row: the vectorizer would otherwise pair the two
                        // rows' trees, and with them the rows' exps and S (~20 v_mov)
                        auto row = w[number<0>{}][0] + w[number<0>{}][1];
                        asm("" : "+v"(row));
                        rowsum(make_tuple(idx0)) = row;
                        return;
                    }
                    array<SMPLComputeDataType, kN / 2> v;
                    static_for<0, kN / 2, 1>{}([&](auto i) {
                        constexpr auto ia = make_tuple(idx0, ids.at(number<i>{}));
                        constexpr auto ib = make_tuple(idx0, ids.at(number<i + kN / 2>{}));
                        v(i)              = p_compute[ia] + p_compute[ib];
                    });
                    static_for<0, integer_log2_floor(kN / 2), 1>{}([&](auto stage) {
                        constexpr index_t half = (kN / 2) >> (stage + 1);
                        static_for<0, half, 1>{}(
                            [&](auto i) { v(i) = v[i] + v[i + number<half>{}]; });
                    });

                    auto row = v[number<0>{}];
                    if constexpr(!kSyncLanes)
                        asm("" : "+v"(row));
                    rowsum(make_tuple(idx0)) = row;
                },
                row_lengths);
        });

        if constexpr(kSyncLanes)
            ReduceRowSync(rowsum, [](auto e0, auto e1) { return e0 + e1; });
        return rowsum;
    }

    // l += rowsum(P) on the matrix unit (see GetRowSumBlockGemm), from the 16-bit P that
    // gemm_1 consumes. The result comes back replicated across a warp tile's columns.
    template <typename LTensor, typename PTile>
    CK_TILE_DEVICE static void AccumulateRowSumOnMatrixUnit(LTensor& l, const PTile& p_tile)
    {
        constexpr auto gemm_rowsum = Policy::template GetRowSumBlockGemm<Problem>();
        using RowSumGemm           = remove_cvref_t<decltype(gemm_rowsum)>;

        // constant B operand of ones
        auto ones = make_static_distributed_tensor<PDataType>(
            make_static_tile_distribution(RowSumGemm::MakeBBlockDistributionEncode()));
        static_for<0, decltype(ones)::get_thread_buffer_size(), 1>{}(
            [&](auto i) { ones.get_thread_buffer()(i) = type_convert<PDataType>(1.0f); });

        auto l_acc = RowSumGemm::MakeCBlockTile();
        clear_tile(l_acc);

        static_for<0, kN0 / kK1, 1>{}([&](auto i_k) {
            gemm_rowsum(
                l_acc,
                get_slice_tile(p_tile, sequence<0, i_k * kK1>{}, sequence<kM0, (i_k + 1) * kK1>{}),
                ones);
        });

        constexpr auto l_spans = decltype(l_acc)::get_distributed_spans();
        sweep_tile_span(l_spans[I0], [&](auto idx0) {
            SMPLComputeDataType row{0};
            // every column carries the same row sum; the sweep only names a valid column
            sweep_tile_span(l_spans[I1], [&](auto idx1) { row = l_acc[make_tuple(idx0, idx1)]; });
            l(make_tuple(idx0)) += row;
        });
    }

    // The sink, mask/bias and LSE/O-finalization steps of qr_tdm's bodies, as
    // functions: the main loop below keeps only its own TDM scheduling and softmax
    // rescaling. qr_tdm keeps its inline copies; moving its bodies onto calls computes
    // the same values but changes register allocation of its kernels at the VGPR limit.

    // Scheduling groups that issue the WMMAs' B-operand reads kAheadWmma WMMAs ahead: a
    // prelude of kAheadWmma WMMAs' reads, then one WMMA and the reads of the WMMA
    // kAheadWmma later, alternating. A B fragment takes two ds reads and is shared by the
    // warp's MIterPerWarp WMMAs (one K iteration per warp gemm). kLeadingWmma WMMAs
    // without reads of their own (the matrix-unit row sum) that share the scheduling
    // region go first.
    template <index_t kMIterPerWarp,
              index_t kNumWmma,
              index_t kLeadingWmma = 0,
              index_t kAheadWmma   = 2>
    CK_TILE_DEVICE static void SchedReadsAheadOfWmma()
    {
        static_assert(kMIterPerWarp == 1 || kMIterPerWarp == 2,
                      "read groups are sized for one or two MIters per B fragment");
        static_assert(kNumWmma >= 2);
        constexpr index_t kReadsPerWmma = 2 / kMIterPerWarp;
        constexpr index_t kAhead        = kAheadWmma < kNumWmma ? kAheadWmma : kNumWmma;
        if constexpr(kLeadingWmma > 0)
            __builtin_amdgcn_sched_group_barrier(0x008, kLeadingWmma, 0);       // WMMA
        __builtin_amdgcn_sched_group_barrier(0x100, kAhead * kReadsPerWmma, 0); // DS read
        static_for<0, kNumWmma - kAhead, 1>{}([&](auto) {
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);             // WMMA
            __builtin_amdgcn_sched_group_barrier(0x100, kReadsPerWmma, 0); // DS read
        });
        __builtin_amdgcn_sched_group_barrier(0x008, kAhead, 0); // WMMA
    }

    // One slice region of kPrefetchNextSlice: each WMMA of this slice follows the reads
    // of its share of the next slice's B fragments. The first region also holds the
    // reads of its own slice, issued ahead of everything. The closing sched_barrier
    // keeps the WMMAs of later slices out of the region.
    template <index_t kMIterPerWarp, index_t kNumWmma, bool kFirstSlice>
    CK_TILE_DEVICE static void SchedSliceUnderNextSliceReads()
    {
        static_assert(kMIterPerWarp == 1 || kMIterPerWarp == 2,
                      "read groups are sized for one or two MIters per B fragment");
        constexpr index_t kReadsPerWmma = 2 / kMIterPerWarp;
        if constexpr(kFirstSlice)
            __builtin_amdgcn_sched_group_barrier(0x100, kNumWmma * kReadsPerWmma, 0); // DS read
        static_for<0, kNumWmma, 1>{}([&](auto) {
            __builtin_amdgcn_sched_group_barrier(0x100, kReadsPerWmma, 0); // DS read
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);             // WMMA
        });
        __builtin_amdgcn_sched_barrier(0);
    }

    template <typename PaddingConfig>
    CK_TILE_DEVICE static TDMConfig MakeTdmConfig()
    {
        using Raw = detail::EncodedTdmPadding<PaddingConfig>;
        TDMConfig config;
        config.pad_enable              = Raw::kEnabled;
        config.pad_config.pad_amount   = Raw::kPadAmount;
        config.pad_config.pad_interval = Raw::kPadInterval;
        return config;
    }

    // A finite sink's score in the row max's domain.
    CK_TILE_DEVICE static SMPLComputeDataType SinkRowMax(float sink_v, float scale_s)
    {
#if CK_TILE_FMHA_FWD_FAST_EXP2
        if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI ||
                     BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS || kHasLogitsSoftCap)
            return sink_v * scale_s * C_LOG2E;
        else
            return sink_v * C_LOG2E;
#else
        ignore = scale_s;
        return sink_v;
#endif
    }

    // Initial row max and row sum: a finite sink pre-seeds them.
    template <typename RowTile>
    CK_TILE_DEVICE static void InitRowMaxSum(RowTile& m, RowTile& l, float sink_v, float scale_s)
    {
        if(__builtin_isinf_sign(sink_v) >= 0)
        {
            set_tile(m, SinkRowMax(sink_v, scale_s));
            set_tile(l, SMPLComputeDataType{1.0f});
        }
        else
        {
            set_tile(m, -numeric<SMPLComputeDataType>::infinity());
            clear_tile(l);
        }
    }

    // Sink-aware K tile range of the block's rows: (sink_seq_end, seqlen_k_start,
    // seqlen_k_end). For non-sink, sink_seq_end is always 0.
    template <typename QOrigin>
    CK_TILE_DEVICE static auto GetKTileRange(const FmhaMask& mask, const QOrigin& q_origin)
    {
        if constexpr(kHasSink)
            return mask.GetSinkTileRangeAlongX(q_origin.at(I0), number<kM0>{}, number<kN0>{});
        else
        {
            auto [start, end] =
                mask.GetTileRangeAlongX(q_origin.at(I0), number<kM0>{}, number<kN0>{});
            return ck_tile::make_tuple(0, start, end);
        }
    }

    // LSE of a block with no K tile to visit: the sink alone, or -inf.
    template <typename LSEaccDramBlockWindowTmp, typename RowTile>
    CK_TILE_DEVICE static void StoreLseOfEmptyRows(LSEaccDramBlockWindowTmp& lse_acc_dram_window,
                                                   const RowTile& m,
                                                   float sink_v,
                                                   float scale_s,
                                                   bool lse_log2)
    {
        auto lse_acc = make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());
        if(__builtin_isinf_sign(sink_v) >= 0)
            set_tile(lse_acc,
                     SMPLComputeDataType{fmha_lse_from_natural(sink_v * scale_s, lse_log2)});
        else
            set_tile(lse_acc, -numeric<SMPLComputeDataType>::infinity());
        store_tile(lse_acc_dram_window, lse_acc);
    }

    // STAGE 2 of the QK result: scale_s and the bias (elementwise or ALiBi), or the tanh
    // logits soft cap, for the K tile starting at k_start. Without either, scale_s is
    // applied in the exponent. position_encoding is taken by value; Alibi::update does not
    // change it.
    template <typename SaccTile,
              typename BiasDramWindow,
              typename QOrigin,
              typename PositionEncoding>
    CK_TILE_DEVICE static void ApplyScaleAndBias(SaccTile& s_acc,
                                                 const BiasDramWindow& bias_dram_window,
                                                 const QOrigin& q_origin,
                                                 index_t k_start,
                                                 float scale_s,
                                                 PositionEncoding position_encoding,
                                                 float logits_soft_cap,
                                                 float logits_soft_cap_rcp)
    {
        constexpr bool kHasBias = BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                  BiasEnum == BlockAttentionBiasEnum::ALIBI;
        // scale_s, then the soft cap, then the bias (the reference's order); without
        // either, scale_s is left to the exponent
        if constexpr(kHasLogitsSoftCap)
        {
            // tanh only: BlockFmhaPipelineQRKSVSTdm refuses another soft cap form
            detail::qr_tdm_apply_logits_soft_cap(
                s_acc, scale_s, logits_soft_cap, logits_soft_cap_rcp);
        }
        else if constexpr(kHasBias)
        {
            tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, s_acc);
        }
        else
        {
            ignore = scale_s;
            ignore = logits_soft_cap;
            ignore = logits_soft_cap_rcp;
        }

        if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
        {
            const auto bias_tile = load_tile(bias_dram_window);
            tile_elementwise_inout(
                [](auto& x, const auto& y) {
#if !CK_TILE_FMHA_FWD_FAST_EXP2
                    x += type_convert<SaccDataType>(y);
#else
                    x += log2e_v<SaccDataType> * type_convert<SaccDataType>(y);
#endif
                },
                s_acc,
                bias_tile);
        }
        else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
        {
            constexpr auto s_spans = SaccTile::get_distributed_spans();
            sweep_tile_span(s_spans[I0], [&](auto idx0) {
                sweep_tile_span(s_spans[I1], [&](auto idx1) {
                    const auto tile_idx = get_x_indices_from_distributed_indices(
                        s_acc.get_tile_distribution(), make_tuple(idx0, idx1));
                    const auto row         = q_origin.at(I0) + tile_idx.at(I0);
                    const auto col         = k_start + tile_idx.at(I1);
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    position_encoding.update(s_acc(i_j_idx), row, col);
                });
            });
        }
        if constexpr(BiasEnum != BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            ignore = bias_dram_window;
        if constexpr(BiasEnum != BlockAttentionBiasEnum::ALIBI)
        {
            ignore = q_origin;
            ignore = k_start;
            ignore = position_encoding;
        }
    }

    // -inf on the scores the K tile at k_start must not see: past seqlen_k_end on
    // the last tile (uneven splits; kClampTail lets a body drop the check when its
    // tile range already guarantees it) and outside the mask on edge tiles.
    template <typename SaccTile, typename QOrigin, bool kClampTail = true>
    CK_TILE_DEVICE static void MaskScores(SaccTile& s_acc,
                                          const FmhaMask& mask,
                                          const QOrigin& q_origin,
                                          index_t k_start,
                                          bool is_last_tile,
                                          index_t seqlen_k_end,
                                          bool_constant<kClampTail> = {})
    {
        if constexpr(kHasUnevenSplits && kClampTail)
        {
            if(is_last_tile)
            {
                set_tile_if(
                    s_acc, type_convert<SMPLComputeDataType>(kMaskedScore), [&](auto tile_idx) {
                        return seqlen_k_end <= k_start + tile_idx.at(I1);
                    });
            }
        }
        else
        {
            ignore = is_last_tile;
            ignore = seqlen_k_end;
        }

        if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
        {
            if(NeedsScoreMask(mask, q_origin.at(I0), k_start))
            {
                set_tile_if(
                    s_acc, type_convert<SMPLComputeDataType>(kMaskedScore), [&](auto tile_idx) {
                        const auto row = q_origin.at(I0) + tile_idx.at(I0);
                        const auto col = k_start + tile_idx.at(I1);
                        if constexpr(kHasSink)
                            return mask.IsOutOfSinkBound(row, col);
                        else if constexpr(kBandMask)
                            return mask.IsOutOfBoundUnsigned(row, col); // one compare per score
                        else
                            return mask.IsOutOfBound(row, col);
                    });
            }
        }
    }

    // A causal or sliding-window mask without sink tokens: each query row sees one contiguous
    // key range, so a tile's masked scores sit in its top-right and bottom-left corners.
    static constexpr bool kBandMask =
        !kHasSink && std::is_same_v<FmhaMask, SimplifiedGenericAttentionMask<true>>;

    // Whether the K tile at k_start has scores to mask. Only rows below seqlen_q are stored, so
    // a band mask judges the tile by the corners of its valid rows: IsEdgeTile also counts the
    // padding rows of a partial last query tile, which under a bottom-right causal mask makes
    // every K tile of that query tile an edge tile.
    CK_TILE_DEVICE static bool NeedsScoreMask(const FmhaMask& mask, index_t q_row0, index_t k_start)
    {
        if constexpr(kBandMask)
        {
            const index_t last_row = min(q_row0 + kM0, mask.GetYTotal()) - 1;
            return mask.IsOutOfBound(q_row0, k_start + kN0 - 1) ||
                   mask.IsOutOfBound(last_row, k_start);
        }
        else
            return mask.IsEdgeTile(q_row0, k_start, number<kM0>{}, number<kN0>{});
    }

    // LSE from the final row max m (log2 domain) and row sum l.
    template <typename LSEaccDramBlockWindowTmp, typename RowTile>
    CK_TILE_DEVICE static void StoreLse(LSEaccDramBlockWindowTmp& lse_acc_dram_window,
                                        const RowTile& m,
                                        const RowTile& l,
                                        float scale_s,
                                        bool lse_log2)
    {
        auto lse_acc = make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

        constexpr auto lse_acc_spans = decltype(lse_acc)::get_distributed_spans();
        sweep_tile_span(lse_acc_spans[I0], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                         BiasEnum == BlockAttentionBiasEnum::ALIBI || kHasLogitsSoftCap)
                lse_acc(i_idx) = fmha_lse(m[i_idx], l[i_idx], lse_log2);
            else
                lse_acc(i_idx) = fmha_lse(m[i_idx] * scale_s, l[i_idx], lse_log2);
        });

        store_tile(lse_acc_dram_window, lse_acc);
    }

    // O / l (a fully masked row, l == 0, gives 0).
    template <typename OaccTile, typename RowTile>
    CK_TILE_DEVICE static void NormalizeO(OaccTile& o_acc, const RowTile& l)
    {
        constexpr auto o_spans = OaccTile::get_distributed_spans();
        sweep_tile_span(o_spans[I0], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            const auto tmp       = [&]() {
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             FmhaMask::IsMasking)
                {
                    return l[i_idx] == 0.f ? 0.f : 1 / l[i_idx];
                }
                else
                    return 1 / l[i_idx];
            }();
            sweep_tile_span(o_spans[I1], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                o_acc(i_j_idx) *= tmp;
            });
        });
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename LSEaccDramBlockWindowTmp,
              typename PositionEncoding,
              typename RandValDramBlockWindowTmp = std::nullptr_t,
              typename Dropout                   = NullBlockDropout>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,
               const KDramBlockWindowTmp& k_dram_block_window_tmp,
               const VDramBlockWindowTmp& v_dram_block_window_tmp,
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp,
               LSEaccDramBlockWindowTmp& lse_acc_dram_window_tmp,
               FmhaMask mask,
               PositionEncoding position_encoding,
               float scale_s,
               void* smem_arena,
               float sink_v,
               bool lse_log2                                           = false,
               float logits_soft_cap                                   = 0.f,
               float logits_soft_cap_rcp                               = 0.f,
               RandValDramBlockWindowTmp randval_dram_block_window_tmp = {},
               Dropout dropout                                         = {}) const
    {
        using Layout = typename Policy::template LdsArenaLayout<Problem>;

        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

        // Q, K and V all go through the TDM (box-major LDS write); a row-major V is read
        // back with ds_load_tr (transposed), a column-major one with plain LDS loads (see
        // LoadVTile). V dram window keeps (kN1, kK1) shape.
        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kSubQKHeaddim == QDramBlockWindowTmp{}.get_window_lengths()[I1] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kK0 == KDramBlockWindowTmp{}.get_window_lengths()[I1] &&
                          kN1 == VDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kK1 == VDramBlockWindowTmp{}.get_window_lengths()[I1] &&
                          kM0 == BiasDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kN0 == BiasDramBlockWindowTmp{}.get_window_lengths()[I1],
                      "wrong!");
        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetPVBlockGemm<Problem>();

        using SaccBlockTileType = decltype(gemm_0.MakeCBlockTile());
        auto s_acc              = SaccBlockTileType{};

        // reduction function for softmax
        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };

        using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());

        auto o_acc = OaccBlockTileType{};

        // infer Sacc, S, P, M, L, Oacc type
        using SBlockTileType = decltype(cast_tile<SMPLComputeDataType>(o_acc));

        using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
            SBlockTileType{}, sequence<1>{}, f_max, SMPLComputeDataType{0}));

        // init M, L (sink-aware: when sink_v is finite, pre-seed m/l). l normalises O; on
        // the matrix unit it sums the 16-bit P that gemm_1 consumes. Without the row max
        // (kRowMaxBySum) it is each thread's part of the fp32 sum, without the sink's
        // seed, synced and seeded after the main loop.
        auto m = MLBlockTileType{};
        auto l = MLBlockTileType{};
        // row max o_acc and l are normalised to; trails m by up to
        // kRebaseThreshold (see the lazy rebasing in the main loop)
        auto m_base = MLBlockTileType{};

        clear_tile(o_acc);
        InitRowMaxSum(m, l, sink_v, scale_s);
        m_base = m;
        if constexpr(kRowMaxBySum)
            clear_tile(l);

        const auto q_origin = q_dram_block_window_tmp.get_window_origin();

        const auto tile_range_result      = GetKTileRange(mask, q_origin);
        const auto sink_seq_end           = tile_range_result.get(ck_tile::number<0>{});
        const auto logical_seqlen_k_start = tile_range_result.get(ck_tile::number<1>{});
        const auto logical_seqlen_k_end   = tile_range_result.get(ck_tile::number<2>{});

        const auto num_sink_loop = integer_divide_ceil(sink_seq_end, kN0);

        // check early exit if no work to do
        if constexpr(FmhaMask::IsMasking || kPadSeqLenK || kHasUnevenSplits)
        {
            const index_t logical_num_total_loop =
                integer_divide_ceil(logical_seqlen_k_end - logical_seqlen_k_start, kN0) +
                num_sink_loop;
            if(logical_num_total_loop <= 0)
            {
                if constexpr(kStoreLSE)
                    StoreLseOfEmptyRows(lse_acc_dram_window_tmp, m, sink_v, scale_s, lse_log2);

                // Note: here occ are all cleard, return it
                // Note: q loaded but no fence, ignore it.
                return o_acc;
            }
        }

        // TDM configs for Q / K / V; the LDS padding comes from the policy.
        TDMConfig tdm_config_q =
            MakeTdmConfig<typename Policy::template LdsPaddingConfigQ<Problem>>();
        TDMConfig tdm_config_k =
            MakeTdmConfig<typename Policy::template LdsPaddingConfigK<Problem>>();
        TDMConfig tdm_config_v =
            MakeTdmConfig<typename Policy::template LdsPaddingConfigV<Problem>>();

        // Q tile in LDS
        auto q_dram_window = make_tile_window(
            q_dram_block_window_tmp, Policy::template MakeQDramTileDistribution<Problem>());

        // Q sits on V's region, so K's transfer can be issued right behind Q's.
        // Q is read once, before the loop; the loop-head barrier drains those
        // reads before the first V transfer overwrites the region.
        QDataType* const q_smem_ptr =
            reinterpret_cast<QDataType*>(static_cast<char*>(smem_arena) + Layout::kQOffset);

        auto q_lds_view = make_tensor_view<address_space_enum::lds>(
            q_smem_ptr, Policy::template MakeQLdsBlockDescriptor<Problem>());

        auto q_lds_store_window = make_tile_window(
            q_lds_view, Policy::template MakeQLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        auto q_lds_read_window =
            make_tile_window(q_lds_view,
                             Policy::template MakeQLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0},
                             Policy::template MakeQRegTileDistribution<Problem>());

        load_tile_tdm(tdm_config_q, q_lds_store_window, q_dram_window);

        // K tile in LDS
        // For sink: kv_load_start is 0 when there are sink tokens (we start
        // reading from seq position 0), otherwise seqlen_k_start.
        const auto kv_load_start =
            (sink_seq_end == 0 && logical_seqlen_k_start > 0) ? logical_seqlen_k_start : 0;
        const index_t physical_seqlen_k_start = logical_seqlen_k_start;
        const index_t physical_seqlen_k_end   = logical_seqlen_k_end;

        // Bias tile window (ELEMENTWISE_BIAS or ALIBI -- null window for NO_BIAS)
        const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();
        auto bias_dram_window =
            make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                             bias_dram_block_window_tmp.get_window_lengths(),
                             {bias_origin.at(number<0>{}), kv_load_start},
                             gemm_0.MakeCBlockTile().get_tile_distribution());

        // K is staged whole-tile (one TDM transfer per iteration) and double
        // buffered: the next tile goes into the buffer this trip does not read.
        auto k_dram_window =
            make_tile_window(k_dram_block_window_tmp,
                             {kv_load_start, 0},
                             Policy::template MakeKDramTileDistribution<Problem, true>());

        // Both K buffers share one descriptor; the ping/pong only swaps the
        // windows' base pointer.
        KDataType* const k_smem_buf0 =
            reinterpret_cast<KDataType*>(static_cast<char*>(smem_arena) + Layout::kK0Offset);
        KDataType* const k_smem_buf1 =
            reinterpret_cast<KDataType*>(static_cast<char*>(smem_arena) + Layout::kK1Offset);

        auto k_lds_view = make_tensor_view<address_space_enum::lds>(
            k_smem_buf0, Policy::template MakeKLdsBlockDescriptor<Problem>());

        auto k_lds_write_window = make_tile_window(
            k_lds_view, Policy::template MakeKLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});
        auto k_lds_read_window =
            make_tile_window(k_lds_view,
                             make_tuple(number<kN0>{}, number<kK0>{}),
                             {0, 0},
                             Policy::template MakeKRegTileDistribution<Problem>());

        // issued while Q is still in flight
        load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);

        // S tile in LDS, only used with the gemm_0 warps on N
        auto s_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<SaccDataType*>(static_cast<char*>(smem_arena) + Layout::kSOffset),
            Policy::template MakeSLdsBlockDescriptor<Problem>());
        [[maybe_unused]] auto s_write_lds_window = make_tile_window(
            s_lds, Policy::template MakeSLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});
        [[maybe_unused]] auto s_read_lds_window =
            make_tile_window(s_lds,
                             Policy::template MakeSLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0},
                             Policy::template MakeSRegTileDistribution<Problem>());

        // V tile in LDS
        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp,
                             VSeqOffset(kv_load_start),
                             Policy::template MakeVDramTileDistribution<Problem>());

        // written by the TDM, read back through the same descriptor (LoadVTile: ds_load_tr
        // for a row-major V, plain LDS loads for a column-major one)
        VDataType* const v_smem_buf0 =
            reinterpret_cast<VDataType*>(static_cast<char*>(smem_arena) + Layout::kV0Offset);
        [[maybe_unused]] VDataType* const v_smem_buf1 =
            reinterpret_cast<VDataType*>(static_cast<char*>(smem_arena) + Layout::kV1Offset);
        auto v_lds_view = make_tensor_view<address_space_enum::lds>(
            v_smem_buf0, Policy::template MakeVLdsBlockDescriptor<Problem>());
        auto v_lds_write_window = make_tile_window(
            v_lds_view, Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        auto v_lds_read_window =
            make_tile_window(v_lds_view,
                             VReadLengths(),
                             {0, 0},
                             Policy::template MakeVRegTileDistribution<Problem>());

        const index_t num_total_loop =
            integer_divide_ceil(physical_seqlen_k_end - physical_seqlen_k_start, kN0) +
            num_sink_loop;

        // retire Q only; K (and V_0), issued after it, stay in flight
        if constexpr(kDoubleV && !Layout::kQOnBothV)
        {
            // V0 does not overlap Q, which sits on V1
            load_tile_tdm(tdm_config_v, v_lds_write_window, v_dram_window);
            move_tile_window(v_dram_window, VSeqOffset(kN0));
            s_wait_tensorcnt_barrier<2>();
        }
        else
            s_wait_tensorcnt_barrier<1>();
        auto q_tile = load_tile(q_lds_read_window);
        if constexpr(Layout::kQOnBothV)
        {
            // Q spans both V buffers: V0 goes out once every wave holds its Q
            block_sync_lds();
            load_tile_tdm(tdm_config_v, v_lds_write_window, v_dram_window);
            move_tile_window(v_dram_window, VSeqOffset(kN0));
        }

        // dropout's randval window walks the keys with the loop (BlockDropout::Run moves it)
        [[maybe_unused]] auto randval_dram_window = [&]() {
            if constexpr(Base::kHasDropout)
                return dropout.template MakeRandvalDramWindow<decltype(gemm_0)>(
                    randval_dram_block_window_tmp, kv_load_start);
            else
            {
                ignore = randval_dram_block_window_tmp;
                return nullptr;
            }
        }();

        index_t i_total_loops      = 0;
        constexpr index_t k0_loops = kQKHeaddim / kK0;
        constexpr index_t k1_loops = kN0 / kK1;

        static_assert(1 <= k0_loops);
        static_assert(1 <= k1_loops);

        do
        {
            // trip i reads K buffer (i & 1) and stages trip i+1 into the other
            const bool k_buf_is_even     = ((i_total_loops & 1) == 0);
            KDataType* const k_read_ptr  = k_buf_is_even ? k_smem_buf0 : k_smem_buf1;
            KDataType* const k_write_ptr = k_buf_is_even ? k_smem_buf1 : k_smem_buf0;

            // One barrier for two hazards: the V transfer below overwrites what
            // the last trip's gemm_1 (or, on the first trip, the Q load) read,
            // and gemm_0 needs this trip's K. The only TDM in flight here is
            // K_i (and, with double V, V_i), so the count is <0>; dscnt 0 also
            // retires the Q reads.
            s_wait_tensorcnt_barrier<0, 0>();

            // K is whole in LDS; the kK0 slices are walked by the read window. With
            // slice prefetch the first slice is read before this trip's TDM issue,
            // whose descriptor setup then covers the read latency (the transfers
            // write the other buffers).
            k_lds_read_window.set_bottom_tensor_view_data_ptr(k_read_ptr);
            auto k_tile = decltype(load_tile(k_lds_read_window)){};
            if constexpr(kPrefetchNextSlice)
                k_tile = load_tile(k_lds_read_window);

            const bool has_next_tile = i_total_loops + 1 < num_total_loop;
            if constexpr(kDoubleV)
            {
                // V_{i+1} goes into the buffer the last trip's gemm_1 read. V runs a
                // tile ahead, so it takes the sink->normal jump one trip early.
                if constexpr(kHasSink)
                {
                    if(i_total_loops == num_sink_loop - 1)
                        move_tile_window(v_dram_window,
                                         VSeqOffset(physical_seqlen_k_start - sink_seq_end));
                }
                if(has_next_tile)
                {
                    v_lds_write_window.set_bottom_tensor_view_data_ptr(k_buf_is_even ? v_smem_buf1
                                                                                     : v_smem_buf0);
                    load_tile_tdm(tdm_config_v, v_lds_write_window, v_dram_window);
                    move_tile_window(v_dram_window, VSeqOffset(kN0));
                }
                v_lds_read_window.set_bottom_tensor_view_data_ptr(k_buf_is_even ? v_smem_buf0
                                                                                : v_smem_buf1);
            }
            else
            {
                load_tile_tdm(tdm_config_v, v_lds_write_window, v_dram_window);
                move_tile_window(v_dram_window, VSeqOffset(kN0));
            }

            // Next K, into the buffer the last trip's gemm_0 read: the barrier above
            // retired those reads, so it goes out with V, a whole trip ahead of its
            // gemm_0. The last trip has no next tile and issues nothing.
            if constexpr(kHasSink)
            {
                if(i_total_loops == num_sink_loop - 1)
                    move_tile_window(k_dram_window, {physical_seqlen_k_start - sink_seq_end, 0});
            }
            move_tile_window(k_dram_window, {kN0, 0});
            if(has_next_tile)
            {
                k_lds_write_window.set_bottom_tensor_view_data_ptr(k_write_ptr);
                load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);
            }

            // STAGE 1, QK gemm
            clear_tile(s_acc); // initialize C

            // V stays in flight under this gemm until the wait before gemm_1.
            if constexpr(!kPrefetchNextSlice)
                k_tile = load_tile(k_lds_read_window);

            if constexpr(kKReadsTwoSlicesAhead)
            {
                // slice i's WMMAs run under the reads of slice i + 2; k_tile holds
                // slice 0, read at the loop head
                using G0 = remove_cvref_t<decltype(gemm_0)>;
                move_tile_window(k_lds_read_window, {0, kK0});
                auto k_tile_1 = load_tile(k_lds_read_window);
                static_for<0, k0_loops, 1>{}([&](auto i_k0) {
                    const auto q_slice = get_slice_tile(
                        q_tile, sequence<0, i_k0 * kK0>{}, sequence<kM0, (i_k0 + 1) * kK0>{});
                    if constexpr(i_k0 + 2 < k0_loops)
                    {
                        move_tile_window(k_lds_read_window, {0, kK0});
                        auto k_tile_2 = load_tile(k_lds_read_window);
                        gemm_0(s_acc, q_slice, k_tile);
                        SchedSliceUnderNextSliceReads<G0::MIterPerWarp,
                                                      G0::MIterPerWarp * G0::NIterPerWarp,
                                                      i_k0.value == 0>();
                        k_tile   = k_tile_1;
                        k_tile_1 = k_tile_2;
                    }
                    else
                    {
                        gemm_0(s_acc, q_slice, k_tile);
                        if constexpr(i_k0 + 2 == k0_loops)
                        {
                            __builtin_amdgcn_sched_barrier(0);
                            k_tile = k_tile_1;
                        }
                    }
                });
                move_tile_window(k_lds_read_window, {0, -kK0 * (k0_loops - 1)});
            }
            else
            {
                if constexpr(1 < k0_loops)
                {
                    static_for<0, k0_loops - 1, 1>{}([&](auto i_k0) {
                        // loop over along the [K]ey head dimension
                        move_tile_window(k_lds_read_window, {0, kK0});
                        auto k_tile_switch = load_tile(k_lds_read_window);

                        gemm_0(s_acc,
                               get_slice_tile(q_tile,
                                              sequence<0, i_k0 * kK0>{},
                                              sequence<kM0, (i_k0 + 1) * kK0>{}),
                               k_tile);
                        if constexpr(kPrefetchNextSlice)
                        {
                            using G0 = remove_cvref_t<decltype(gemm_0)>;
                            SchedSliceUnderNextSliceReads<G0::MIterPerWarp,
                                                          G0::MIterPerWarp * G0::NIterPerWarp,
                                                          i_k0.value == 0>();
                        }

                        k_tile = k_tile_switch;
                    });
                    // move back to the origin
                    move_tile_window(k_lds_read_window, {0, -kK0 * (k0_loops - 1)});
                }

                gemm_0(s_acc,
                       get_slice_tile(q_tile,
                                      sequence<0, (k0_loops - 1) * kK0>{},
                                      sequence<kM0, k0_loops * kK0>{}),
                       k_tile);
            }

            // Software-pipeline the K reads ahead of their consumer (kReadAheadWmma);
            // left alone the scheduler waits on dscnt before every WMMA.
            using Gemm0T = remove_cvref_t<decltype(gemm_0)>;
            static_assert(Gemm0T::WarpGemm::kK == kK0, "one K iteration per QK warp gemm");
            if constexpr(!kPrefetchNextSlice)
                SchedReadsAheadOfWmma<Gemm0T::MIterPerWarp,
                                      Gemm0T::MIterPerWarp * Gemm0T::NIterPerWarp * k0_loops,
                                      0,
                                      kReadAheadWmma>();

            // V_i's first slice (see kReadVBeforeSoftmax);
            // the sched_barrier keeps the reads ahead of the softmax.
            auto v_tile = decltype(LoadVTile(v_lds_read_window)){};
            if constexpr(kReadVBeforeSoftmax)
            {
                v_tile = LoadVTile(v_lds_read_window);
                __builtin_amdgcn_sched_barrier(0);
            }

            // Sink-aware: in the sink phase tiles start at 0, in the normal phase at
            // physical_seqlen_k_start.
            const index_t kv_tile_start =
                (num_sink_loop > i_total_loops)
                    ? kv_load_start + i_total_loops * kN0
                    : physical_seqlen_k_start + (i_total_loops - num_sink_loop) * kN0;

            [[maybe_unused]] const auto bias_window_of_tile = bias_dram_window;
            // STAGE 2: scale_s, add bias, mask
            ApplyScaleAndBias(s_acc,
                              bias_dram_window,
                              q_origin,
                              kv_tile_start,
                              scale_s,
                              position_encoding,
                              logits_soft_cap,
                              logits_soft_cap_rcp);
            MaskScores(s_acc,
                       mask,
                       q_origin,
                       kv_tile_start,
                       i_total_loops == num_total_loop - 1,
                       physical_seqlen_k_end);

            // Sink->normal window jump: at the boundary between sink region
            // and normal region, jump the V (single V) and bias dram windows forward;
            // K jumped with its issue at the loop head.
            if constexpr(kHasSink)
            {
                if(i_total_loops == num_sink_loop - 1)
                {
                    if constexpr(!kDoubleV)
                        move_tile_window(v_dram_window,
                                         VSeqOffset(physical_seqlen_k_start - sink_seq_end));
                    move_tile_window(bias_dram_window, {0, physical_seqlen_k_start - sink_seq_end});
                }
            }

            move_tile_window(bias_dram_window, {0, kN0});

            // S{j}. With the gemm_0 warps on N each warp holds a column slice, so S goes
            // through LDS into the row-complete layout gemm_1's A operand takes.
            const auto s_new = [&]() {
                if constexpr(kNWarp > 1)
                {
                    store_tile(s_write_lds_window, cast_tile<SMPLComputeDataType>(s_acc));
                    block_sync_lds();
                    return load_tile(s_read_lds_window);
                }
                else
                {
                    return cast_tile<SMPLComputeDataType>(s_acc);
                }
            }();

            const auto update_row_max = [&](const auto& s_in) {
                auto m_local = block_tile_reduce<SMPLComputeDataType>(
                    s_in,
                    sequence<1>{},
                    f_max,
                    -numeric<SMPLComputeDataType>::infinity()); // m_local = rowmax(S{j})
                ReduceRowSync(m_local, f_max);
                tile_elementwise_inout(
                    [](auto& e0, auto e1) { e0 = max(e0, e1); }, m, m_local); // m{j}
            };

            auto p_compute = make_static_distributed_tensor<SMPLComputeDataType>(
                s_new.get_tile_distribution()); // Pcompute{j}

            static const auto get_validated_m = [](SMPLComputeDataType raw_m) {
                /// NOTICE: bias might be materialized mask including -inf values, need
                /// consideration
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             FmhaMask::IsMasking)
                {
                    return raw_m == -numeric<SMPLComputeDataType>::infinity()
                               ? type_convert<SMPLComputeDataType>(0.f)
                               : raw_m;
                }
                else
                {
                    return raw_m;
                }
            };

            // Exponent that p (and hence o_acc / l) is normalised against.
            const auto get_row_exp = [&](SMPLComputeDataType raw_m) {
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             BiasEnum == BlockAttentionBiasEnum::ALIBI || kHasLogitsSoftCap)
                {
                    return get_validated_m(raw_m);
                }
                else
                {
                    return scale_s * get_validated_m(raw_m);
                }
            };

            constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();

            // Lazy rebasing: o_acc and l stay normalised to m_base, and are
            // rescaled only when a row max outgrows it by more than
            // kRebaseThreshold (p then stays below 2^kRebaseThreshold). The gap
            // uses the raw maxima: a row whose m_base is -inf always rebases, a
            // fully masked row (NaN gap) never does.
            constexpr auto kRebaseThreshold = SMPLComputeDataType{8}; // 256x
            const auto row_gap              = [&](auto i_idx) {
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             BiasEnum == BlockAttentionBiasEnum::ALIBI || kHasLogitsSoftCap)
                    return m[i_idx] - m_base[i_idx];
                else
                    return scale_s * (m[i_idx] - m_base[i_idx]);
            };
            const auto rebase_rows = [&]() {
                bool need_rebase = false;
                sweep_tile_span(o_spans[I0], [&](auto idx0) {
                    need_rebase |= row_gap(make_tuple(idx0)) > kRebaseThreshold;
                });
                if(need_rebase)
                {
                    sweep_tile_span(o_spans[I0], [&](auto idx0) {
                        constexpr auto i_idx = make_tuple(idx0);
                        const auto gap       = row_gap(i_idx);
                        if(gap > 0)
                        {
                            const auto tmp = exp2(-gap);
                            l(i_idx) *= tmp;
                            m_base(i_idx) = m[i_idx];
                            sweep_tile_span(o_spans[I1], [&](auto idx1) {
                                o_acc(make_tuple(idx0, idx1)) *= tmp;
                            });
                        }
                    });
                }
            };

            constexpr auto p_spans = decltype(p_compute)::get_distributed_spans();
            const auto compute_p   = [&](const auto& s_in) {
                sweep_tile_span(p_spans[I0], [&](auto idx0) {
                    constexpr auto i_idx = make_tuple(idx0);
                    auto row_max         = get_row_exp(m_base[i_idx]);
                    sweep_tile_span(p_spans[I1], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                     BiasEnum == BlockAttentionBiasEnum::ALIBI)
                        {
                            p_compute(i_j_idx) = exp2(s_in[i_j_idx] - row_max);
                        }
                        else
                        {
                            if constexpr(kHasLogitsSoftCap)
                            {
                                p_compute(i_j_idx) = exp2(s_in[i_j_idx] - row_max);
                            }
                            else
                            {
                                p_compute(i_j_idx) = exp2(scale_s * s_in[i_j_idx] - row_max);
                            }
                        }
                    });
                });
            };

            [[maybe_unused]] int32_t p_scale_early = 0;
            using PTileEarly =
                decltype(Base::template MakePForGemm1<decltype(gemm_1)>(p_compute, p_scale_early));
            [[maybe_unused]] PTileEarly p_tile_early{};
            // kEarlyPConvert: P in 16 bits before the bound check
            [[maybe_unused]] auto make_p_tile = [&]() {
                if constexpr(kRowMaxBySum)
                    return Base::template MakePForGemm1<decltype(gemm_1)>(p_compute, p_scale_early);
                else
                    return 0;
            };
            if constexpr(!kIsSoftmax)
                ApplyActivation(s_new, p_compute, scale_s);
            else if constexpr(kRowMaxBySum)
            {
                // P against the running base (see kRowMaxBySum)
                // a row without a key yet has no base (-inf): its tile takes the row max
                bool no_base = false;
                sweep_tile_span(o_spans[I0], [&](auto idx0) {
                    no_base |=
                        m_base[make_tuple(idx0)] == -numeric<SMPLComputeDataType>::infinity();
                });
                const bool row_max_tile = !kSkipRowMax || __builtin_amdgcn_ballot_w32(no_base) != 0;
                if(row_max_tile)
                {
                    update_row_max(s_new);
                    rebase_rows();
                }
                compute_p(s_new);
                if constexpr(kEarlyPConvert)
                    p_tile_early = make_p_tile();
                auto part = BlockRowSum<false>(p_compute);
                bool redo = false;
                if(!row_max_tile)
                    sweep_tile_span(o_spans[I0], [&](auto idx0) {
                        redo |= !(part[make_tuple(idx0)] <=
                                  SMPLComputeDataType{256} /* 2^kRebaseThreshold */);
                    });
                if(kSkipRowMax && __builtin_expect(__builtin_amdgcn_ballot_w32(redo) != 0, 0))
                {
                    // a p over 2^kRebaseThreshold, or one that does not fit the 16-bit P:
                    // S again (K_i is still in LDS, the TDM stages K_{i+1} into the other
                    // buffer), then the row max way
                    __builtin_amdgcn_sched_barrier(0);
                    // S again into P's registers (same layout, P is retaken from it)
                    clear_tile(p_compute);
                    static_for<0, k0_loops, 1>{}([&](auto i_k0) {
                        gemm_0(p_compute,
                               get_slice_tile(q_tile,
                                              sequence<0, i_k0 * kK0>{},
                                              sequence<kM0, (i_k0 + 1) * kK0>{}),
                               (k_tile = load_tile(k_lds_read_window), k_tile));
                        move_tile_window(k_lds_read_window, {0, kK0});
                        __builtin_amdgcn_sched_barrier(0);
                    });
                    move_tile_window(k_lds_read_window, {0, -kK0 * k0_loops});
                    auto bias_window_2 = bias_window_of_tile;
                    ApplyScaleAndBias(p_compute,
                                      bias_window_2,
                                      q_origin,
                                      kv_tile_start,
                                      scale_s,
                                      position_encoding,
                                      logits_soft_cap,
                                      logits_soft_cap_rcp);
                    MaskScores(p_compute,
                               mask,
                               q_origin,
                               kv_tile_start,
                               i_total_loops == num_total_loop - 1,
                               physical_seqlen_k_end);
                    update_row_max(p_compute);
                    rebase_rows();
                    compute_p(p_compute);
                    if constexpr(kEarlyPConvert)
                        p_tile_early = make_p_tile();
                    part = BlockRowSum<false>(p_compute);
                }
                sweep_tile_span(o_spans[I0], [&](auto idx0) {
                    constexpr auto i_idx = make_tuple(idx0);
                    l(i_idx) += part[i_idx];
                });
            }
            else
            {
                update_row_max(s_new);
                rebase_rows();
                compute_p(s_new);
            }

            // MakePForGemm1 reorders P's (MIter, KIter) blocks within a thread, which
            // lines up with gemm_1's A operand only with the warps on M or one MIter.
            static_assert(BlockFmhaShape::Gemm1BlockWarps::at(I1) == 1 ||
                          remove_cvref_t<decltype(gemm_1)>::MIterPerWarp == 1);
            if constexpr(Base::kHasDropout)
            {
                // l{j} sums the undropped P; dropout then scales the kept P by 1/(1-p)
                const auto rowsum_p = BlockRowSum(p_compute);
                sweep_tile_span(o_spans[I0], [&](auto idx0) {
                    constexpr auto i_idx = make_tuple(idx0);
                    l(i_idx) += rowsum_p[i_idx];
                });
                if constexpr(kHasSink)
                {
                    // past the sink tiles the keys jump to physical_seqlen_k_start
                    if(num_sink_loop > 0 && i_total_loops == num_sink_loop)
                        move_tile_window(randval_dram_window,
                                         {0, physical_seqlen_k_start - sink_seq_end});
                }
                // the randval shuffle's own LDS, outside the arena
                __shared__ char
                    dropout_lds[BlockDropout::MakeRandValLdsBlockDescriptor<decltype(gemm_0)>()
                                    .get_element_space_size()];
                dropout.template Run<decltype(gemm_0), SMPLComputeDataType, RandValOutputDataType>(
                    dropout_lds, kv_tile_start, p_compute, randval_dram_window);
            }
            [[maybe_unused]] int32_t p_scale = 0;
            auto p_tile                      = [&]() {
                if constexpr(kRowMaxBySum && kEarlyPConvert)
                    return p_tile_early;
                else
                    return Base::template MakePForGemm1<decltype(gemm_1)>(p_compute, p_scale);
            }();

            // l{j}; o_acc is already in p's m_base frame.
            if constexpr(kRowSumOnMatrixUnit && !kRowMaxBySum)
                AccumulateRowSumOnMatrixUnit(l, p_tile);
            if constexpr(!kRowSumOnMatrixUnit && kIsSoftmax && !Base::kHasDropout)
            {
                const auto rowsum_p = BlockRowSum(p_compute); // rowsum(Pcompute{j})
                sweep_tile_span(o_spans[I0], [&](auto idx0) {
                    constexpr auto i_idx = make_tuple(idx0);
                    l(i_idx) += rowsum_p[i_idx];
                });
            }

            // In flight: V, then the next K. <1> retires V and leaves K running
            // under the PV gemm; on the last trip only V is in flight. With double
            // V, V_i already landed before the loop-head barrier.
            if constexpr(!kDoubleV)
            {
                if(has_next_tile)
                    s_wait_tensorcnt_barrier<1>();
                else
                    s_wait_tensorcnt_barrier<0>();
            }

            if constexpr(!kReadVBeforeSoftmax)
                v_tile = LoadVTile(v_lds_read_window);

            if constexpr(1 < k1_loops)
            {
                static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
                    if constexpr(kPrefetchNextSlice)
                    {
                        // the next slice's reads go out before this slice's WMMAs
                        move_tile_window(v_lds_read_window, VSeqOffset(kK1));
                        auto v_tile_next = LoadVTile(v_lds_read_window);
                        gemm_1(o_acc,
                               get_slice_tile(p_tile,
                                              sequence<0, i_k1 * kK1>{},
                                              sequence<kM0, (i_k1 + 1) * kK1>{}),
                               v_tile);
                        using G1 = remove_cvref_t<decltype(gemm_1)>;
                        SchedSliceUnderNextSliceReads<G1::MIterPerWarp,
                                                      G1::MIterPerWarp * G1::NIterPerWarp,
                                                      i_k1.value == 0>();
                        v_tile = v_tile_next;
                    }
                    else
                    {
                        gemm_1(o_acc,
                               get_slice_tile(p_tile,
                                              sequence<0, i_k1 * kK1>{},
                                              sequence<kM0, (i_k1 + 1) * kK1>{}),
                               v_tile);

                        // loop over along the [V]alue Sequence length
                        move_tile_window(v_lds_read_window, VSeqOffset(kK1));
                        v_tile = LoadVTile(v_lds_read_window);
                    }
                });
                // move back to the origin
                move_tile_window(v_lds_read_window, VSeqOffset(-kK1 * (k1_loops - 1)));
            }

            gemm_1(o_acc,
                   get_slice_tile(p_tile,
                                  sequence<0, (k1_loops - 1) * kK1>{},
                                  sequence<kM0, k1_loops * kK1>{}),
                   v_tile);

            // V reads run ahead as for K above. Without the wait before gemm_1
            // (double V) the row-sum WMMAs share its scheduling region; placing them
            // first measured faster for d64 dense grids only (gfx1250: +4% at 128 x 1859
            // keys, -1..5% on d64 masked and d128 grids), so the others leave them free.
            using Gemm1T = remove_cvref_t<decltype(gemm_1)>;
            static_assert(Gemm1T::WarpGemm::kK == kK1, "one K iteration per PV warp gemm");
            constexpr index_t kRowSumWmma = kDoubleV && kRowSumOnMatrixUnit && !kRowMaxBySum &&
                                                    !FmhaMask::IsMasking && kSubQKHeaddim <= 64
                                                ? k1_loops * Gemm1T::MIterPerWarp
                                                : 0;
            if constexpr(!kPrefetchNextSlice)
                SchedReadsAheadOfWmma<Gemm1T::MIterPerWarp,
                                      Gemm1T::MIterPerWarp * Gemm1T::NIterPerWarp * k1_loops,
                                      kRowSumWmma,
                                      kReadAheadWmma>();

        } while(++i_total_loops < num_total_loop);

        if constexpr(kRowMaxBySum)
        {
            // the lanes' parts of the row, then the sink's seed in the m_base frame
            ReduceRowSync(l, [](auto e0, auto e1) { return e0 + e1; });
            if constexpr(kHasSink)
            {
                if(__builtin_isinf_sign(sink_v) >= 0)
                {
                    const auto sink_m      = SinkRowMax(sink_v, scale_s);
                    constexpr auto l_spans = decltype(l)::get_distributed_spans();
                    sweep_tile_span(l_spans[I0], [&](auto idx0) {
                        constexpr auto i_idx = make_tuple(idx0);
                        if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                     BiasEnum == BlockAttentionBiasEnum::ALIBI || kHasLogitsSoftCap)
                            l(i_idx) += exp2(sink_m - m_base[i_idx]);
                        else
                            l(i_idx) += exp2(scale_s * (sink_m - m_base[i_idx]));
                    });
                }
            }
        }
        if constexpr(kStoreLSE)
        {
            // the fp32 sum, normalised to m_base, validated as in the loop (a fully
            // masked row keeps m_base at -inf)
            auto m_lse = m_base;
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                         FmhaMask::IsMasking)
            {
                tile_elementwise_inout(
                    [](auto& x) {
                        if(x == -numeric<SMPLComputeDataType>::infinity())
                            x = type_convert<SMPLComputeDataType>(0.f);
                    },
                    m_lse);
            }
            StoreLse(lse_acc_dram_window_tmp, m_lse, l, scale_s, lse_log2);
        }

        if constexpr(kIsSoftmax)
            NormalizeO(o_acc, l);
        return o_acc;
    }

    // The kernel's other LDS argument order (see FmhaFwdKernel).
    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename LSEaccDramBlockWindowTmp,
              typename PositionEncoding,
              typename RandValDramBlockWindowTmp = std::nullptr_t,
              typename Dropout                   = NullBlockDropout>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,
               const KDramBlockWindowTmp& k_dram_block_window_tmp,
               const VDramBlockWindowTmp& v_dram_block_window_tmp,
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp,
               LSEaccDramBlockWindowTmp& lse_acc_dram_window_tmp,
               FmhaMask mask,
               PositionEncoding position_encoding,
               float scale_s,
               float sink_v,
               void* smem_arena,
               bool lse_log2                                           = false,
               float logits_soft_cap                                   = 0.f,
               float logits_soft_cap_rcp                               = 0.f,
               RandValDramBlockWindowTmp randval_dram_block_window_tmp = {},
               Dropout dropout                                         = {}) const
    {
        return (*this)(q_dram_block_window_tmp,
                       k_dram_block_window_tmp,
                       v_dram_block_window_tmp,
                       bias_dram_block_window_tmp,
                       lse_acc_dram_window_tmp,
                       mask,
                       position_encoding,
                       scale_s,
                       smem_arena,
                       sink_v,
                       lse_log2,
                       logits_soft_cap,
                       logits_soft_cap_rcp,
                       randval_dram_block_window_tmp,
                       dropout);
    }
};

} // namespace ck_tile
