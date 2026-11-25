// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_fmha_lse.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_pingpong_policy.hpp"
#include "ck_tile/ops/fmha/block/block_fmha_quant_scales.hpp"

namespace ck_tile {

namespace detail {
// Whether a mask's rows can start their keys in later tiles: its IsLocal (a window) where it
// has one, else true (a generic mask may). A C++17 detection idiom: ck_tile still builds as
// C++17.
template <typename Mask, typename = void>
struct qr_tdm_mask_starts_rows_late : std::true_type
{
};
template <typename Mask>
struct qr_tdm_mask_starts_rows_late<Mask, std::void_t<decltype(Mask::IsLocal)>>
    : std::bool_constant<Mask::IsLocal>
{
};
template <typename Mask>
inline constexpr bool qr_tdm_mask_starts_rows_late_v = qr_tdm_mask_starts_rows_late<Mask>::value;

// A masking GenericAttentionMask or SimplifiedGenericAttentionMask: a band of the diagonals
// col - row, bounded on the right by x and, with a left window (kLeft), on the left by y.
template <typename Mask>
struct qr_tdm_band_mask : std::false_type
{
    static constexpr bool kLeft = false;
};
template <bool IsLocal>
struct qr_tdm_band_mask<GenericAttentionMask<true, IsLocal>> : std::true_type
{
    static constexpr bool kLeft = IsLocal;
};
template <>
struct qr_tdm_band_mask<SimplifiedGenericAttentionMask<true>> : std::true_type
{
    static constexpr bool kLeft = true;
};
} // namespace detail

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

// qr_tdm with a whole-tile K ping/pong, for gfx1250 fp16/bf16 and fp8 (fp8 P): trip i reads K
// buffer (i & 1) while the TDM fills the other with tile i + 1 (the decode tiles hold one K
// buffer, refilled once the QK gemm has read it). V streams through one buffer under
// the QK gemm, or, where is_qr_tdm_pingpong_double_v_v holds, through two with V_{i+1} issued a
// whole trip ahead. It shares qr_tdm's problem, gemms and register layouts; only the LDS layout
// (see BlockFmhaPipelineQRKSVSTdmPingPongPolicy) and the main loop differ.
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
    using Base::kStoreLSE;
    using Base::kSubQKHeaddim;

    using Base::LoadVTile;
    using Base::ReduceRowSync;
    using Base::VReadLengths;
    using Base::VSeqOffset;

    static constexpr const char* name = "qr_tdm_pingpong";

    // the kernel paths qr_tdm takes by name
    static constexpr bool kUsesUntransposedVKernelPath = true;
    static constexpr bool kUsesTdmAffineDramPath       = true;

    static constexpr bool kKLoadOnce = true;
    // Square causal grids visit the longest query tiles of all batches and heads first.
    static constexpr bool kUsesGlobalCausalOrder = true;

    static_assert(detail::is_qr_tdm_pingpong_qkv_v<Problem>,
                  "qr_tdm_pingpong: fp16/bf16, or fp8 Q/K/V and P");
    // fp8: no scales, per-tensor/per-head, block or MX scales (see BlockFmhaQuantScales)
    static_assert(Base::QScaleEnum == BlockAttentionQuantScaleEnum::NO_SCALE ||
                      ((Base::QScaleEnum == BlockAttentionQuantScaleEnum::PERTENSOR ||
                        Base::QScaleEnum == BlockAttentionQuantScaleEnum::PERHEAD ||
                        Base::QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE ||
                        Base::QScaleEnum == BlockAttentionQuantScaleEnum::MX) &&
                       std::is_same_v<PDataType, fp8_t>),
                  "qr_tdm_pingpong: quantization scales are per-tensor, per-head, per-block or "
                  "MX, on fp8");
    // The quantization scales (BlockFmhaQuantScales): what each scale kind does on the scores
    // and on O, and the trips' K/V scales staged in LDS.
    using QkGemm         = remove_cvref_t<decltype(Policy::template GetQKBlockGemm<Problem>())>;
    using PvGemm         = remove_cvref_t<decltype(Policy::template GetPVBlockGemm<Problem>())>;
    using QuantScales    = BlockFmhaQuantScales<Problem, QkGemm, PvGemm>;
    using QuantScaleArgs = typename QuantScales::Args;
    // block k/v descales: a block of keys spans whole PV slices (IsSupportedArgument)
    static constexpr index_t kKVScaleAlign = QuantScales::kVBlockKeys;

    // An fp8 P (e4m3) is taken 2^kPExp2Offset times its softmax value: p, below 2^0.75 by
    // the rebase threshold, lands under 2^0.75 * 2^kPExp2Offset = 431, within e4m3's largest
    // finite 448, with its normal range reaching 2^-6 / 2^kPExp2Offset. l sums the same
    // scaled p, so O / l cancels the factor; the LSE and a sink's seed take it out. The
    // smallest p round to zero or to e4m3's subnormal steps, which drops a long row's tail
    // and sharpens its softmax: on -init=3 (d128, 2048 x 16384) 2^7 left O 0.22% high, past
    // the reference's systematic-gain bound, 2^8 halves that (qr_tdm_fp8_sched's 256).
    // Dropout does not scale an fp8 P by 1/(1 - p_drop) (the kernel puts it on O's factor),
    // so the kept p stays in that range.
    static constexpr bool kFp8P           = std::is_same_v<PDataType, fp8_t>;
    static constexpr index_t kPExp2Offset = kFp8P ? 8 : 0;
    // ds reads per B fragment of a main-loop WMMA (16 columns of kK, over 32 lanes):
    // ds_load_b128 for K and a column-major V; a row-major V goes through the transposed
    // read, ds_load_tr16_b128 at 16 bits, ds_load_tr8_b64 at fp8.
    static constexpr index_t kKReadsPerFragment =
        BlockFmhaShape::Gemm0WarpTile::at(number<2>{}) * 16 * sizeof(KDataType) / 32 / 16;
    static constexpr index_t kVReadsPerFragment =
        BlockFmhaShape::Gemm1WarpTile::at(number<2>{}) * 16 * sizeof(VDataType) / 32 /
        ((!Base::kVColMajor && sizeof(VDataType) == 1) ? 8 : 16);
    static_assert(!Problem::kUseDoubleKVLdsBuffer && !Problem::kProgressiveDsLoadK,
                  "qr_tdm_pingpong has its own K/V buffering");

    using Layout                   = typename Policy::template LdsArenaLayout<Problem>;
    static constexpr bool kDoubleV = Layout::kDoubleV;
    // one K buffer (Layout::kSingleK): K_{i+1} is issued past trip i's QK gemm
    static constexpr bool kSingleK = Layout::kSingleK;
    static_assert(!kSingleK || !kDoubleV, "one K buffer takes the single-V trip schedule");
    // the staged scales' LDS (BlockFmhaQuantScales) beside the arena, two blocks a CU
    static_assert(Layout::kArenaBytes + QuantScales::kTripBuffers * QuantScales::kTripDwords *
                                            static_cast<index_t>(sizeof(int32_t)) <=
                      Layout::kArenaBudget,
                  "qr_tdm_pingpong: the arena and the staged scales take two blocks a CU");

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
    // the block gemms give each warp kN / warps whole warp tiles: a remainder would go
    // uncomputed
    static_assert(BlockFmhaShape::kN0 % (BlockFmhaShape::Gemm0BlockWarps::at(I1) *
                                         BlockFmhaShape::Gemm0WarpTile::at(I1)) ==
                          0 &&
                      BlockFmhaShape::kN1 % (BlockFmhaShape::Gemm1BlockWarps::at(I1) *
                                             BlockFmhaShape::Gemm1WarpTile::at(I1)) ==
                          0,
                  "qr_tdm_pingpong: the warps on N need whole warp tiles of K rows and V columns");
    // BlockDropout::Run walks P in the QK gemm's output layout
    static_assert(!Base::kHasDropout || kNWarp == 1,
                  "qr_tdm_pingpong: dropout needs the QK gemm warps on M");

    // The short-query (decode) tile spends its four warps on the trip's keys and V's
    // columns, kKeyWarps x (4 / kKeyWarps) (flash decoding inside the block): gemm_0's warps
    // on N take whole key ranges (warp w the w-th of kKeyWarps, repeated for each column
    // range), gemm_1's warps on K the same ones and on N V's columns, and each warp runs its
    // own softmax and holds its own row maxima, row sums and O over its keys, merged once
    // after the main loop (MergeKeySplit). A trip exchanges nothing between the warps.
    // Where not even two warps' keys fit (d512: kKeyWarps 1), gemm_0's warps on N take the
    // trip's keys and exchange S through LDS, each warp then softmaxing whole rows for its V
    // columns (kExchangeS): on a 512-byte Q/K row, cheaper than each warp's own QK gemm over
    // all the keys (MX d512 decode: -4..-26%).
    static constexpr index_t kKeyWarps = BlockFmhaShape::Gemm1BlockWarps::at(number<2>{});
    static constexpr bool kKeySplit    = kKeyWarps > 1;
    static constexpr bool kExchangeS   = kNWarp > 1 && !kKeySplit;
    // P and the scales take a warp's keys as one contiguous range of the trip, which the PV
    // slices split only where there is one slice a trip
    static_assert(!kKeySplit || kK1 == kN0, "qr_tdm_pingpong: the key split takes one PV slice");
    static_assert((kKeySplit ? kNWarp == kKeyWarps : true) &&
                      BlockFmhaShape::Gemm0BlockWarps::at(I0) ==
                          BlockFmhaShape::Gemm1BlockWarps::at(I0),
                  "qr_tdm_pingpong: gemm_0's warps on N are gemm_1's on K (the key split)");
    // Whole rows in each warp (over its keys), no dropout: l and o_acc sum the same fp32 P
    // with and without the LSE, and the LSE store is all the LSE adds. Each thread keeps its
    // part of l; the lanes are synced once, after the main loop.
    static constexpr bool kRowMaxBySum = !kExchangeS && !Base::kHasDropout;
    // An fp8 P is rounded for gemm_1, and l sums the rounded values gemm_1 takes, as
    // qr_tdm_fp8_sched does (on the VALU), here on the matrix unit: P times a column of ones,
    // whole rows that need no lane sync. The VALU sum loads the softmax's VALU (even at eight
    // a v_cvt_scale_pk8_f32_fp8: fp8 d128 dense 4096 x 16384 -12%). The
    // unrounded sum would count what e4m3 flushes (p below 2^-16 of the row max) and bias
    // long, flat rows low.
    // With dropout l sums the undropped rounded P (dropout only zeroes an fp8 P, which
    // commutes with the rounding).
    static constexpr bool kRowSumOnMatrixUnit = kFp8P;
    // The LSE is log sum exp of the scores: it takes the row sum of the unrounded P, kept
    // beside l when l sums the rounded one (as qr_tdm_fp8_sched keeps both).
    static constexpr bool kLseRowSum = kFp8P && kStoreLSE;
    // A mask whose rows start their keys in later tiles (a window).
    static constexpr bool kMaskStartsRowsLate = []() {
        if constexpr(!FmhaMask::IsMasking)
            return false;
        else
            return detail::qr_tdm_mask_starts_rows_late_v<FmhaMask>;
    }();
    // A tile whose softmax has no per-score bias and whose rows all start at the first key
    // tile (no window): the deeper read schedules below (slice prefetch, K two slices
    // ahead, four waves) apply to it. Every tile takes the row max (see the rebasing in
    // the main loop).
    static constexpr bool kPlainSoftmaxTile =
        kRowMaxBySum && !kMaskStartsRowsLate && BiasEnum == BlockAttentionBiasEnum::NO_BIAS;
    // The short-query tile keeps the whole-gemm read-ahead: its slices are a single
    // WMMA row. The plain 128-row tiles take the slice prefetch at two waves per SIMD,
    // K two slices ahead.
    // MX tiles on 128-deep PV warp gemms (the scaled WMMA's K; kMxWidePv) hold twice the P
    // and V registers of the 64-key trips. Up to d64 they run three waves a SIMD, which
    // leaves no registers for the read-ahead below (the next slice, V before the softmax):
    // MX d64 128-row three waves 296 VGPRs, with either read-ahead 336 and 23..39 spilled; two
    // waves with both 378 (gfx1250 MX d64: dense 1024^2 +7%, causal 4096^2 +6%, 4096^2 +3%).
    static constexpr bool kMxScales = Base::QScaleEnum == BlockAttentionQuantScaleEnum::MX;
    static constexpr bool kMxWidePv =
        kMxScales && BlockFmhaShape::kM0 >= 64 && BlockFmhaShape::kK1 >= 128;
    static constexpr bool kThreeWavesNoReadAhead = kMxWidePv && kQKHeaddim <= 64;
    static constexpr bool kPrefetchNextSlice =
        !kThreeWavesNoReadAhead && BlockFmhaShape::kM0 >= 64 &&
        ((kPlainSoftmaxTile && BlockFmhaShape::kM0 >= 128) ||
         (LdsReadSchedule::kPrefetchNextSliceMinRows > 0 &&
          BlockFmhaShape::kM0 >= LdsReadSchedule::kPrefetchNextSliceMinRows));
    // Main-loop trips split into unmasked interior ones and masked edge ones (see run_trip),
    // for the masks: without one the only edge trip is a partial last tile, cheaper as a
    // branch than as a second copy of the trip (fp8 d128 128-key trips: 512 -> 414 VGPRs).
    // Without sink tokens, whose tiles lead the key range, every mask here is a band.
    static constexpr bool kSplitEdgeTrips = !kHasSink && FmhaMask::IsMasking;
    static constexpr bool kKReadsTwoSlicesAhead =
        kPrefetchNextSlice && kPlainSoftmaxTile && kQKHeaddim / kK0 >= 3;
    static constexpr bool kReadVBeforeSoftmax =
        !kThreeWavesNoReadAhead && LdsReadSchedule::kReadVBeforeSoftmaxMinRows > 0 &&
        BlockFmhaShape::kM0 >= LdsReadSchedule::kReadVBeforeSoftmaxMinRows&& kDoubleV;
    // Split-off edge trips (the score mask) read V after the softmax: the mask's indices would
    // otherwise share the registers of V's early slice (fp8 d128 causal: 405 VGPRs, two
    // waves). Unsplit trips keep the early read.
    template <bool kEdgeTrip>
    static constexpr bool kReadVBeforeSoftmaxOn =
        kReadVBeforeSoftmax && !(kEdgeTrip && kSplitEdgeTrips);
    // P goes to PDataType right after the exps; the plain tiles with the slice prefetch
    // convert under the PV WMMAs (gfx1250 d96 window 1024 keys: early 332 VGPRs, three
    // waves; late 341, two).
    static constexpr bool kEarlyPConvert =
        kRowMaxBySum && (!kPlainSoftmaxTile || !kPrefetchNextSlice);
    // The plain 128-row tiles of d33..d64 (and the unmasked d32 one) take four waves per
    // SIMD (d64 512 x 5252 keys: three waves 1.045 ms, four 1.000 ms), the other 128-row
    // tiles of d32..d96 three (d96 causal 1024 keys two waves +6%, d96 window with the LSE
    // two waves +5%); d128 and up take two waves with the slice prefetch.
    // MX on 128-deep PV warp gemms (kMxWidePv): three waves to d64 without the read-ahead
    // (kThreeWavesNoReadAhead), two to d128.
    // The decode tile's registers are budgeted for two blocks a CU (see the codegen's decode
    // tile), qr_tdm's three to d64.
    static constexpr index_t kBlockPerCu = []() {
        if constexpr(BlockFmhaShape::kM0 <= 16)
            return max(index_t{2}, Base::kBlockPerCu);
        else if constexpr(kMxWidePv && kQKHeaddim <= 128)
            return kThreeWavesNoReadAhead ? index_t{3} : index_t{2};
        else if constexpr(kPlainSoftmaxTile && BlockFmhaShape::kM0 >= 128 && kQKHeaddim <= 64 &&
                          (kQKHeaddim > 32 || !FmhaMask::IsMasking))
            return index_t{4};
        else if constexpr(kRowMaxBySum && BlockFmhaShape::kM0 >= 128 && kQKHeaddim <= 96)
            return index_t{3};
        else if constexpr(kFp8P && BlockFmhaShape::kM0 >= 128 && BlockFmhaShape::kN0 <= 64 &&
                          kQKHeaddim <= 128)
            return index_t{3};
        // past d128 the 128-row tiles run two blocks too, the pipeline's premise: qr_tdm's
        // one would let them past 512 VGPRs (fp8 d256 128-row: 518, one wave a SIMD, -20%);
        // MX's 128-deep PV slices take them far past it (d256 754: one)
        else if constexpr(BlockFmhaShape::kM0 >= 128 && kQKHeaddim > 128 && kK1 < 128)
            return index_t{2};
        else
            return Base::kBlockPerCu;
    }();

    // A row's thread-buffer offsets (where the tile distribution puts its elements), in
    // ascending order.
    template <typename PComputeTensor, typename Idx0, typename Ids>
    CK_TILE_HOST_DEVICE static constexpr auto SortedRowOffsets(Idx0, Ids)
    {
        constexpr auto offsets = generate_sequence_v2(
            [](auto i) {
                constexpr auto y_idx =
                    PComputeTensor::get_tile_distribution().get_y_indices_from_distributed_indices(
                        make_tuple(Idx0{}, Ids{}.at(i)));
                return number<typename PComputeTensor::ThreadTensorDesc{}.calculate_offset(
                    y_idx)>{};
            },
            number<Ids::size()>{});
        return typename sequence_sort<remove_cvref_t<decltype(offsets)>, less<index_t>>::type{};
    }

    // Whether a row's elements sit in the thread buffer as neighbouring pairs (2k, 2k+1).
    template <typename PComputeTensor, typename Idx0, typename Ids>
    CK_TILE_HOST_DEVICE static constexpr bool RowPairsAdjacent(Idx0, Ids)
    {
        constexpr index_t kN = Ids::size();
        constexpr auto o     = SortedRowOffsets<PComputeTensor>(Idx0{}, Ids{});
        bool adjacent        = true;
        static_for<0, kN / 2, 1>{}([&](auto k) {
            adjacent &=
                o[number<2 * k>{}] % 2 == 0 && o[number<2 * k + 1>{}] == o[number<2 * k>{}] + 1;
        });
        return adjacent;
    }

    // P in PDataType. An fp8 P converts eight values per v_cvt_scalef32_pk8_fp8_f32, which
    // also lets the exps write P in place: for the two-value convert, which takes any register
    // pair, the compiler streamed every exp through one temporary, a pk_fma -> exp -> nop
    // chain (fp8 d128 128-key trips, 2048 x 16384: +20%).
    template <typename PComputeTensor>
    CK_TILE_DEVICE static auto CastP(const PComputeTensor& p_compute)
    {
        constexpr index_t kBuf = PComputeTensor::get_thread_buffer_size();
        if constexpr(kFp8P && kBuf % 8 == 0)
        {
            auto p = make_static_distributed_tensor<PDataType>(p_compute.get_tile_distribution());
            static_for<0, kBuf / 8, 1>{}([&](auto i) {
                p.get_thread_buffer().template set_as<fp8x8_t>(
                    i,
                    fp32x8_to_fp8x8(p_compute.get_thread_buffer().template get_as<fp32x8_t>(i),
                                    1.0f));
            });
            return p;
        }
        else
            return cast_tile<PDataType>(p_compute);
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
                        // the row summed as register pairs (v_pk_add_f32 of
                        // neighbouring registers): pair j of the row is the thread-buffer
                        // elements offs[2j], offs[2j + 1] (built from the two elements:
                        // reading them through thread_buffer::get_as moved the schedule)
                        constexpr auto offs = SortedRowOffsets<PComputeTensor>(idx0, ids);
                        const auto& buf     = p_compute.get_thread_buffer();
                        const auto pair     = [&](auto j) {
                            return fp32x2_t{buf[number<offs[number<2 * j>{}]>{}],
                                            buf[number<offs[number<2 * j + 1>{}]>{}]};
                        };
                        array<fp32x2_t, kN / 4> w;
                        static_for<0, kN / 4, 1>{}(
                            [&](auto k) { w(k) = pair(k) + pair(number<k + kN / 4>{}); });
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

    // l += rowsum(P) on the matrix unit (see GetRowSumBlockGemm), from the P that gemm_1
    // consumes: P times a column of ones, whole rows replicated across a warp tile's
    // columns.
    template <typename LTensor, typename PTile>
    CK_TILE_DEVICE static void AccumulateRowSumOnMatrixUnit(LTensor& l, const PTile& p_tile)
    {
        constexpr auto gemm_rowsum = Policy::template GetRowSumBlockGemm<Problem>();
        using RowSumGemm           = remove_cvref_t<decltype(gemm_rowsum)>;

        // constant B operand of ones
        auto ones = make_static_distributed_tensor<PDataType>(
            make_static_tile_distribution(RowSumGemm::MakeBBlockDistributionEncode()));
        set_tile(ones, type_convert<PDataType>(1.0f));

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

    // Each SIMD runs a wave of every resident block, with the blocks out of phase: one in
    // its gemms while another runs its softmax. The waves in their gemms take priority, so
    // their WMMAs issue back to back and the softmax's VALU work fills the gaps between them
    // (gfx1250, bf16: PR table geomean +3.3%, causal d32/d160 +15..27%; fp8 d128 bottom-right
    // causal 1024 x 8192 +8%). A priority switch bounds the compiler's scheduling regions,
    // which costs registers: it rides on V-before-softmax, already on only where the tile's
    // registers have room (off it spills: bf16 d96 causal -30%, d512 -25%). Long dense trips
    // (128 keys, no mask) run without it, measured a loss in every placement tried (fp8 d128
    // dense 1024 x 8192 -4%, 4096 x 16384 -6%; bf16 d32 dense -2%). MX on 128-deep PV warp
    // gemms (kMxWidePv) take it on every trip: their two or three waves a SIMD contend for
    // the WMMAs and the exps without it (gfx1250 MX 32 x 8 heads: d64 +3%, d128 +1..2%).
    static constexpr bool kLongDenseTrip = BlockFmhaShape::kN0 >= 128 && !FmhaMask::IsMasking;
    template <bool kEdgeTrip>
    static constexpr bool kGemmPriority =
        kMxWidePv || (kReadVBeforeSoftmaxOn<kEdgeTrip> && !kLongDenseTrip);

    // Marks the start (kGemm) or end of a gemm phase of a trip: see kGemmPriority.
    template <bool kGemm, bool kEdgeTrip>
    CK_TILE_DEVICE static void SetGemmPriority()
    {
        if constexpr(kGemmPriority<kEdgeTrip>)
            __builtin_amdgcn_s_setprio(kGemm ? 1 : 0);
    }

    // The sink, mask/bias and LSE/O-finalization steps of qr_tdm's bodies, as
    // functions: the main loop below keeps only its own TDM scheduling and softmax
    // rescaling. qr_tdm keeps its inline copies; moving its bodies onto calls computes
    // the same values but changes register allocation of its kernels at the VGPR limit.

    // Scheduling groups that issue the WMMAs' B-operand reads kAheadWmma WMMAs ahead: a
    // prelude of kAheadWmma WMMAs' reads, then one WMMA and the reads of the WMMA
    // kAheadWmma later, alternating. A B fragment takes kReadsPerFragment ds reads and is
    // shared by the warp's MIterPerWarp WMMAs (one K iteration per warp gemm).
    template <index_t kMIterPerWarp,
              index_t kNumWmma,
              index_t kAheadWmma        = 2,
              index_t kReadsPerFragment = 2>
    CK_TILE_DEVICE static void SchedReadsAheadOfWmma()
    {
        static_assert(kMIterPerWarp == 1 || kMIterPerWarp == 2,
                      "read groups are sized for one or two MIters per B fragment");
        // a single WMMA (fp8 short-query tiles: one 64-deep slice a warp) has no read ahead
        if constexpr(kNumWmma < 2)
            return;
        constexpr index_t kReadsPerWmma = kReadsPerFragment / kMIterPerWarp;
        constexpr index_t kAhead        = kAheadWmma < kNumWmma ? kAheadWmma : kNumWmma;
        __builtin_amdgcn_sched_group_barrier(
            LLVMSchedGroupMask::DS_READ, kAhead * kReadsPerWmma, 0);
        static_for<0, kNumWmma - kAhead, 1>{}([&](auto) {
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::MFMA, 1, 0);
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::DS_READ, kReadsPerWmma, 0);
        });
        __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::MFMA, kAhead, 0);
    }

    // One slice region of kPrefetchNextSlice: each WMMA of this slice follows the reads
    // of its share of the next slice's B fragments. The first region also holds the
    // reads of its own slice, issued ahead of everything. The closing sched_barrier
    // keeps the WMMAs of later slices out of the region.
    template <index_t kMIterPerWarp,
              index_t kNumWmma,
              bool kFirstSlice,
              index_t kReadsPerFragment = 2>
    CK_TILE_DEVICE static void SchedSliceUnderNextSliceReads()
    {
        static_assert(kMIterPerWarp == 1 || kMIterPerWarp == 2,
                      "read groups are sized for one or two MIters per B fragment");
        constexpr index_t kReadsPerWmma = kReadsPerFragment / kMIterPerWarp;
        if constexpr(kFirstSlice)
            __builtin_amdgcn_sched_group_barrier(
                LLVMSchedGroupMask::DS_READ, kNumWmma * kReadsPerWmma, 0);
        static_for<0, kNumWmma, 1>{}([&](auto) {
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::DS_READ, kReadsPerWmma, 0);
            __builtin_amdgcn_sched_group_barrier(LLVMSchedGroupMask::MFMA, 1, 0);
        });
        __builtin_amdgcn_sched_barrier(0);
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

    // Initial row max: the scores' alone (a sink logit joins the row sums after the main
    // loop, see finish_row_sums, so an fp8 P is never taken against a larger sink).
    template <typename RowTile>
    CK_TILE_DEVICE static void InitRowMax(RowTile& m)
    {
        set_tile(m, -numeric<SMPLComputeDataType>::infinity());
    }

    // Sink-aware K tile range of the block's rows: (sink_seq_end, seqlen_k_start,
    // seqlen_k_end). For non-sink, sink_seq_end is always 0.
    // Split-KV: split i_split of num_splits takes the next equal share of the range's key tiles,
    // so only the last split ends off a tile boundary (empty when the tiles run out).
    template <typename QOrigin>
    CK_TILE_DEVICE static auto GetKTileRange(const FmhaMask& mask,
                                             const QOrigin& q_origin,
                                             index_t num_splits = 1,
                                             index_t i_split    = 0)
    {
        // split i_split of num_splits: the next equal run of whole key tiles of [start, end)
        const auto split = [&](index_t& start, index_t& end) {
            if(num_splits > 1)
            {
                const index_t tiles_per_split =
                    integer_divide_ceil(integer_divide_ceil(end - start, kN0), num_splits);
                const index_t split_start = start + i_split * tiles_per_split * kN0;
                end                       = min(end, split_start + tiles_per_split * kN0);
                start                     = min(split_start, end);
            }
        };
        if constexpr(kHasSink)
        {
            // split-KV: the sink tiles go to split 0, the window's tiles are shared out
            auto [sink_seq_end, start, end] =
                mask.GetSinkTileRangeAlongX(q_origin.at(I0), number<kM0>{}, number<kN0>{});
            split(start, end);
            return ck_tile::make_tuple(i_split == 0 ? sink_seq_end : index_t{0}, start, end);
        }
        else
        {
            auto [start, end] =
                mask.GetTileRangeAlongX(q_origin.at(I0), number<kM0>{}, number<kN0>{});
            split(start, end);
            return ck_tile::make_tuple(0, start, end);
        }
    }

    // LSE of a block with no K tile to visit: the sink alone, or -inf. Merged rows (packed
    // GQA) take their own head's sink logit (sink_v only marks that there is one); row0 is
    // the tile's first row.
    template <typename LSEaccDramBlockWindowTmp, typename RowTile>
    CK_TILE_DEVICE static void StoreLseOfEmptyRows(LSEaccDramBlockWindowTmp& lse_acc_dram_window,
                                                   const RowTile& m,
                                                   float sink_v,
                                                   float scale_s,
                                                   const QuantScaleArgs& args,
                                                   index_t row0,
                                                   bool lse_log2)
    {
        auto lse_acc = make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());
        if(__builtin_isinf_sign(sink_v) < 0)
            set_tile(lse_acc, -numeric<SMPLComputeDataType>::infinity());
        else if constexpr(Problem::kMergeNumHeadGroupsSeqLenQ)
        {
            constexpr auto spans = decltype(lse_acc)::get_distributed_spans();
            sweep_tile_span(spans[I0], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                const index_t row    = min(row0 + get_x_indices_from_distributed_indices(
                                                   lse_acc.get_tile_distribution(), i_idx)
                                                   .at(I0),
                                        args.rows - 1);
                lse_acc(i_idx) = fmha_lse_from_natural(
                    args.sink != nullptr ? args.sink[row / args.rows_per_head] : sink_v * scale_s,
                    lse_log2);
            });
        }
        else
            set_tile(lse_acc,
                     SMPLComputeDataType{fmha_lse_from_natural(sink_v * scale_s, lse_log2)});
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
            // No per-element OOB branch: padded scores are masked to -inf, padded rows are not
            // stored, and the buffer range check returns 0 past the end.
            const auto bias_tile =
                load_tile(bias_dram_window, number<-1>{}, bool_constant<false>{});
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
    // tile range already guarantees it) and outside the mask on edge tiles. Each
    // branch takes k_start through an opaque copy: the two share the scores' key
    // indices, which the compiler would otherwise compute ahead of both branches, on
    // every trip (d128 fp8 causal: 60 VALU a trip, 10% of its softmax).
    CK_TILE_DEVICE static index_t OpaqueIndex(index_t v)
    {
        asm volatile("" : "+s"(v));
        return v;
    }
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
                const index_t k_tail = OpaqueIndex(k_start);
                set_tile_if(s_acc, -numeric<SMPLComputeDataType>::infinity(), [&](auto tile_idx) {
                    return seqlen_k_end <= k_tail + tile_idx.at(I1);
                });
            }
        }
        ignore = is_last_tile;
        ignore = seqlen_k_end;

        if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
        {
            if(NeedsScoreMask(mask, q_origin.at(I0), k_start))
            {
                const index_t k0 = OpaqueIndex(k_start);
                // Merged rows under a mask are the heads' single queries (seqlen_q 1; the mask
                // has one row): every row takes row 0's key range.
                if constexpr(Problem::kMergeNumHeadGroupsSeqLenQ && FmhaMask::IsMasking)
                {
                    set_tile_if(
                        s_acc, -numeric<SMPLComputeDataType>::infinity(), [&](auto tile_idx) {
                            const auto col = k0 + tile_idx.at(I1);
                            if constexpr(kHasSink)
                                return mask.IsOutOfSinkBound(0, col);
                            else
                                return mask.IsOutOfBound(0, col);
                        });
                    return;
                }
                const index_t q0 = OpaqueIndex(q_origin.at(I0));
                if constexpr(detail::qr_tdm_band_mask<FmhaMask>::value && !kHasSink)
                {
                    MaskBand(s_acc, q0 + mask.GetWindowX() - k0, q0 - mask.GetWindowY() + 1 - k0);
                    // The band's right edge is min(row + x, seqlen_k): rows whose diagonal
                    // passes seqlen_k (a top-left mask with more query rows than keys, a
                    // right window, packed GQA's merged rows) end at seqlen_k, past which the
                    // tile that crosses it holds padding keys.
                    const index_t seqlen_k = mask.GetXTotal();
                    if(seqlen_k - k_start < kN0 && q0 + kM0 - 1 + mask.GetWindowX() > seqlen_k)
                        set_tile_if(
                            s_acc, -numeric<SMPLComputeDataType>::infinity(), [&](auto tile_idx) {
                                return seqlen_k <= k0 + tile_idx.at(I1);
                            });
                    return;
                }
                set_tile_if(s_acc, -numeric<SMPLComputeDataType>::infinity(), [&](auto tile_idx) {
                    const auto row = q0 + tile_idx.at(I0);
                    const auto col = k0 + tile_idx.at(I1);
                    if constexpr(kHasSink)
                        return mask.IsOutOfSinkBound(row, col);
                    else
                        return mask.IsOutOfBound(row, col);
                });
            }
        }
    }

    // -inf on the scores (row r, column c of the tile) out of the band: c - r >= diag_end
    // (right edge; columns past seqlen_k: see MaskScores) or, with a left window,
    // c - r < diag_start. Rows past seqlen_q are not stored and are left as they are. A
    // score's diagonal c - r is linear in the lane and the register (tile distributions
    // unmerge the lane and register indices into rows and columns): a lane part, the same for
    // all of a thread's scores, plus a constant per register. The lane part less the bound
    // is taken once a trip, opaque, so a score costs a compare against its constant and a
    // select; left visible, the compiler hoists every score's lane part plus constant out of
    // the loop, a VGPR each (fp8 d128 causal: spills).
    template <typename SaccTile>
    CK_TILE_DEVICE static void MaskBand(SaccTile& s_acc, index_t diag_end, index_t diag_start)
    {
        constexpr bool kLeft = detail::qr_tdm_band_mask<FmhaMask>::kLeft;
        using Dstr           = remove_cvref_t<decltype(s_acc.get_tile_distribution())>;
        constexpr auto spans = SaccTile::get_distributed_spans();
        const auto partition = get_partition_index(Dstr{});
        using Partition      = remove_cvref_t<decltype(partition)>;
        constexpr Partition origin{};
        const auto diagonal = [](auto x) { return x.at(I1) - x.at(I0); };
        // the lane part, from any score
        index_t lane_part = 0;
        sweep_tile_span(spans[I0], [&](auto idx0) {
            sweep_tile_span(spans[I1], [&](auto idx1) {
                constexpr auto idx = make_tuple(idx0, idx1);
                constexpr index_t c =
                    diagonal(get_x_indices_from_distributed_indices(Dstr{}, idx, origin));
                lane_part =
                    diagonal(get_x_indices_from_distributed_indices(Dstr{}, idx, partition)) - c;
            });
        });
        // right edge only: c - r >= diag_end; with a left window, one unsigned compare of
        // c - r - diag_start against the band's width
        index_t lane_rel = lane_part - (kLeft ? diag_start : diag_end);
        asm volatile("" : "+v"(lane_rel));
        [[maybe_unused]] const uint32_t width =
            diag_start < diag_end ? uint32_t(diag_end) - uint32_t(diag_start) : 0u;
        sweep_tile_span(spans[I0], [&](auto idx0) {
            sweep_tile_span(spans[I1], [&](auto idx1) {
                constexpr auto idx = make_tuple(idx0, idx1);
                constexpr index_t c =
                    diagonal(get_x_indices_from_distributed_indices(Dstr{}, idx, origin));
                bool out;
                if constexpr(kLeft)
                    out = uint32_t(lane_rel + c) >= width;
                else
                    out = lane_rel >= -c;
                if(out)
                    s_acc(idx) = -numeric<SMPLComputeDataType>::infinity();
            });
        });
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

    // The key split's merge (kKeySplit): each warp's row maxima m_base, row sums l (and
    // l_lse) and o_acc cover its own keys; the block's are M = max over the warps of
    // m_base and the sums of l and O, each warp's scaled by 2^(m_base - M) (0 for a warp
    // whose rows saw no score). Through LDS, once after the main loop; every warp ends up
    // with the merged values of its V columns (o_acc is replicated over the key warps).
    template <typename RowTile, typename OAccTile>
    CK_TILE_DEVICE static void MergeKeySplit(RowTile& m_base,
                                             RowTile& l,
                                             RowTile& l_lse,
                                             OAccTile& o_acc,
                                             void* smem_arena,
                                             float scale_s,
                                             float o_unit)
    {
        constexpr index_t kRowStats = kLseRowSum ? 3 : 2;
        constexpr index_t kRowBytes = kKeyWarps * kRowStats * kM0 * sizeof(float);
        // O rows padded by 16 bytes: a lane's 16-byte accesses walk the rows, which at a
        // power-of-two stride all fall in the same banks (d128: the merge took ~15k cycles)
        constexpr index_t kOStride = kN1 + 4;
        constexpr index_t kOBytes  = kKeyWarps * kM0 * kOStride * sizeof(float);
        static_assert(kRowBytes + kOBytes <= Layout::kArenaBytes,
                      "qr_tdm_pingpong: the key split's merge needs every warp's O in LDS");
        float* const rows  = static_cast<float*>(smem_arena);
        float* const o_lds = reinterpret_cast<float*>(static_cast<char*>(smem_arena) + kRowBytes);
        // the warp's key range (the warps of a key range split V's columns, see o_acc's
        // layout: warp ids are (V columns, keys), keys innermost)
        const index_t warp = get_warp_id() % kKeyWarps;

        const auto o_view = make_tensor_view<address_space_enum::lds>(
            o_lds,
            make_naive_tensor_descriptor(make_tuple(number<kKeyWarps * kM0>{}, number<kN1>{}),
                                         make_tuple(number<kOStride>{}, number<1>{}),
                                         number<4>{},
                                         number<1>{}));
        const auto o_window = [&](index_t w) {
            return make_tile_window(o_view,
                                    make_tuple(number<kM0>{}, number<kN1>{}),
                                    {w * kM0, 0},
                                    o_acc.get_tile_distribution());
        };
        constexpr auto o_spans = OAccTile::get_distributed_spans();
        // the tile row of a row span (any column names it)
        const auto row_of = [&](auto idx0) {
            index_t row = 0;
            sweep_tile_span(o_spans[I1], [&](auto idx1) {
                row = get_x_indices_from_distributed_indices(OAccTile::get_tile_distribution(),
                                                             make_tuple(idx0, idx1))
                          .at(I0);
            });
            return row;
        };
        const auto stat = [&](index_t w, index_t k, index_t row) -> float& {
            return rows[(w * kRowStats + k) * kM0 + row];
        };

        // the main loop's last LDS reads and transfers are done before the arena is reused
        s_wait_tensorcnt_barrier<0, 0>();
        sweep_tile_span(o_spans[I0], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            const index_t row    = row_of(idx0);
            stat(warp, 0, row)   = m_base[i_idx];
            stat(warp, 1, row)   = l[i_idx];
            if constexpr(kLseRowSum)
                stat(warp, 2, row) = l_lse[i_idx];
        });
        auto o_mine = o_window(warp);
        // this warp's o_acc in plain units (block v descales hold it in its last block's)
        if(o_unit != 1.f)
            tile_elementwise_inout([o_unit](auto& x) { x *= o_unit; }, o_acc);
        store_tile(o_mine, o_acc);
        block_sync_lds();

        const auto to_exp2 = [&](auto d) {
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                         BiasEnum == BlockAttentionBiasEnum::ALIBI || kHasLogitsSoftCap)
                return d;
            else
                return scale_s * d;
        };
        clear_tile(o_acc);
        static_for<0, kKeyWarps, 1>{}([&](auto w) {
            auto o_w_window = o_window(w);
            const auto o_w  = load_tile(o_w_window);
            sweep_tile_span(o_spans[I0], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                const index_t row    = row_of(idx0);
                // M, the rows' maximum over the warps
                auto big_m = stat(0, 0, row);
                static_for<1, kKeyWarps, 1>{}([&](auto v) { big_m = max(big_m, stat(v, 0, row)); });
                // a warp whose keys of a row were all masked (or past seqlen_k) adds nothing:
                // its sums and O may be NaN there (no row max to take p against), so they are
                // skipped, not scaled by 0
                const auto m_w    = stat(w, 0, row);
                const bool scored = m_w != -numeric<SMPLComputeDataType>::infinity();
                const auto factor = scored ? exp2(to_exp2(m_w - big_m)) : SMPLComputeDataType{0};
                if constexpr(decltype(w)::value == 0)
                {
                    l(i_idx) = 0;
                    if constexpr(kLseRowSum)
                        l_lse(i_idx) = 0;
                }
                l(i_idx) += scored ? stat(w, 1, row) * factor : SMPLComputeDataType{0};
                if constexpr(kLseRowSum)
                    l_lse(i_idx) += scored ? stat(w, 2, row) * factor : SMPLComputeDataType{0};
                if constexpr(decltype(w)::value == kKeyWarps - 1)
                    m_base(i_idx) = big_m;
                sweep_tile_span(o_spans[I1], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    o_acc(i_j_idx) += scored ? o_w[i_j_idx] * factor : SMPLComputeDataType{0};
                });
            });
        });
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
            // ln(2^m * l) in fp32, or with lse_log2 log2(2^m * l): one v_log_f32 and an fma
            // (the division by the double C_LOG2E took the f64 divide sequence per row). l
            // carries an fp8 P's 2^kPExp2Offset, taken off in the log2 domain.
            const float to_base = lse_log2 ? 1.f : log2e_rcp_v<float>;
            const auto log2_l   = log2(l[i_idx]) - kPExp2Offset;
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                         BiasEnum == BlockAttentionBiasEnum::ALIBI || kHasLogitsSoftCap)
                lse_acc(i_idx) = (m[i_idx] + log2_l) * to_base;
            else
                lse_acc(i_idx) = (m[i_idx] * scale_s + log2_l) * to_base;
        });

        store_tile(lse_acc_dram_window, lse_acc);
    }

    // O * v_descale / l (a fully masked row, l == 0, gives 0).
    template <typename OaccTile, typename RowTile>
    CK_TILE_DEVICE static void NormalizeO(OaccTile& o_acc, const RowTile& l, float v_descale)
    {
        constexpr auto o_spans = OaccTile::get_distributed_spans();
        sweep_tile_span(o_spans[I0], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            const auto tmp       = [&]() {
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             FmhaMask::IsMasking)
                {
                    return l[i_idx] == 0.f ? 0.f : v_descale / l[i_idx];
                }
                else
                    return v_descale / l[i_idx];
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
               Dropout dropout                                         = {},
               QuantScaleArgs quant_scale_args                         = {},
               index_t num_splits                                      = 1,
               index_t i_split                                         = 0) const
    {
        static_assert(Base::kIsSoftmax,
                      "qr_tdm_pingpong: softmax only (GDPA runs on qr_tdm_gdpa_qloop)");

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
        QuantScales quant_scales;
        quant_scales.args = quant_scale_args;

        // reduction function for softmax
        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };

        using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());

        auto o_acc = OaccBlockTileType{};

        // infer Sacc, S, P, M, L, Oacc type
        using SBlockTileType = decltype(cast_tile<SMPLComputeDataType>(o_acc));

        using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
            SBlockTileType{}, sequence<1>{}, f_max, SMPLComputeDataType{0}));

        // init M, L. m starts at -inf (a sink joins after the loop). l normalises O: the row sum of
        // P (of the rounded P for an fp8 one); l_lse, kept for an fp8 LSE, the unrounded one. Both
        // start at 0, are rescaled with o_acc, and take the sink's seed after the main loop
        // (finish_row_sums); with kRowMaxBySum l holds each lane's part until then, l_lse always
        // does.
        auto m = MLBlockTileType{};
        auto l = MLBlockTileType{};
        // row max o_acc and l are normalised to (m as of the last rebase, see the main loop)
        auto m_base = MLBlockTileType{};

        clear_tile(o_acc);
        InitRowMax(m);
        m_base = m;
        clear_tile(l);
        [[maybe_unused]] auto l_lse = MLBlockTileType{};
        if constexpr(kLseRowSum)
            clear_tile(l_lse);

        const auto q_origin = q_dram_block_window_tmp.get_window_origin();

        const auto tile_range_result      = GetKTileRange(mask, q_origin, num_splits, i_split);
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
                    StoreLseOfEmptyRows(lse_acc_dram_window_tmp,
                                        m,
                                        sink_v,
                                        scale_s,
                                        quant_scale_args,
                                        q_origin.at(I0),
                                        lse_log2);

                // Note: here occ are all cleard, return it
                // Note: q loaded but no fence, ignore it.
                return o_acc;
            }
        }

        // TDM configs for Q / K / V; the LDS padding comes from the policy.
        TDMConfig tdm_config_q =
            detail::make_tdm_config<typename Policy::template LdsPaddingConfigQ<Problem>>();
        TDMConfig tdm_config_k =
            detail::make_tdm_config<typename Policy::template LdsPaddingConfigK<Problem>>();
        TDMConfig tdm_config_v =
            detail::make_tdm_config<typename Policy::template LdsPaddingConfigV<Problem>>();

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
        const index_t num_total_loop =
            integer_divide_ceil(physical_seqlen_k_end - physical_seqlen_k_start, kN0) +
            num_sink_loop;

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

        // S tile in LDS, only used where gemm_0's warps on N exchange S (kExchangeS)
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
        quant_scales.template Begin<decltype(q_tile),
                                    decltype(load_tile(k_lds_read_window)),
                                    decltype(LoadVTile(v_lds_read_window))>(q_origin.at(I0));
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

        index_t i_total_loops = 0;
        // the first key of trip i: in the sink phase tiles start at 0, in the normal phase at
        // physical_seqlen_k_start
        const auto trip_start = [&](index_t i) {
            return num_sink_loop > i ? kv_load_start + i * kN0
                                     : physical_seqlen_k_start + (i - num_sink_loop) * kN0;
        };
        // the scales of the first two trips (see BlockFmhaQuantScales)
        if constexpr(QuantScales::kStagesTrips)
        {
            for(index_t i = 0; i < min(num_total_loop, index_t{2}); ++i)
                quant_scales.IssueTrip(trip_start(i), i);
        }
        constexpr index_t k0_loops = kQKHeaddim / kK0;
        constexpr index_t k1_loops = kN0 / kK1;

        static_assert(1 <= k0_loops);
        static_assert(1 <= k1_loops);

        // One trip of the main loop. Only edge trips (kEdgeTrip) carry the score mask: a
        // branch for it splits every trip's schedule in two (d128 fp8 causal 4096^2: masking
        // the diagonal tiles inside every trip's body cost 12% end to end).
        const auto run_trip = [&](auto edge_trip) {
            constexpr bool kEdgeTrip = decltype(edge_trip)::value;
            // trip i reads K buffer (i & 1) and stages trip i+1 into the other
            const bool k_buf_is_even     = ((i_total_loops & 1) == 0);
            KDataType* const k_read_ptr  = k_buf_is_even ? k_smem_buf0 : k_smem_buf1;
            KDataType* const k_write_ptr = k_buf_is_even ? k_smem_buf1 : k_smem_buf0;
            const bool has_next_tile     = i_total_loops + 1 < num_total_loop;

            const auto qk_gemm = [&](auto i_k0, const auto& q_slice, const auto& k_slice) {
                quant_scales.Qk(gemm_0, s_acc, q_slice, k_slice, i_k0, i_total_loops);
            };

            // One barrier for two hazards: the V transfer below overwrites what the last
            // trip's gemm_1 (or, on the first trip, the Q load) read, and gemm_0 needs this
            // trip's K. The only TDM in flight here is K_i (and, with double V, V_i), so the
            // count is <0>; dscnt 0 also retires the Q reads.
            // (and this trip's staged scales, which the barrier makes visible)
            QuantScales::WaitTrip(i_total_loops + 1 >= num_total_loop);
            s_wait_tensorcnt_barrier<0, 0>();
            quant_scales.ReadTrip(i_total_loops, trip_start(i_total_loops));

            // K is whole in LDS; the kK0 slices are walked by the read window. With
            // slice prefetch the first slice is read before this trip's TDM issue,
            // whose descriptor setup then covers the read latency (the transfers
            // write the other buffers).
            k_lds_read_window.set_bottom_tensor_view_data_ptr(k_read_ptr);
            auto k_tile = decltype(load_tile(k_lds_read_window)){};
            if constexpr(kPrefetchNextSlice)
                k_tile = load_tile(k_lds_read_window);

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
            // gemm_0. The last trip has no next tile and issues nothing. With one K buffer
            // it waits for this trip's gemm_0 instead.
            const auto issue_next_k = [&]() {
                if constexpr(kHasSink)
                {
                    if(i_total_loops == num_sink_loop - 1)
                        move_tile_window(k_dram_window,
                                         {physical_seqlen_k_start - sink_seq_end, 0});
                }
                move_tile_window(k_dram_window, {kN0, 0});
                if(has_next_tile)
                {
                    k_lds_write_window.set_bottom_tensor_view_data_ptr(k_write_ptr);
                    load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);
                }
            };
            if constexpr(!kSingleK)
                issue_next_k();

            // the trip's scale reads landed (under the transfers' issue); trip i + 2's scales
            // go out, into the buffer trip i - 1 read
            quant_scales.PinTrip();
            if constexpr(QuantScales::kStagesTrips)
            {
                if(i_total_loops + 2 < num_total_loop)
                    quant_scales.IssueTrip(trip_start(i_total_loops + 2), i_total_loops + 2);
            }

            // STAGE 1, QK gemm
            SetGemmPriority<true, kEdgeTrip>();
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
                        qk_gemm(i_k0, q_slice, k_tile);
                        SchedSliceUnderNextSliceReads<G0::MIterPerWarp,
                                                      G0::MIterPerWarp * G0::NIterPerWarp,
                                                      i_k0.value == 0,
                                                      kKReadsPerFragment>();
                        k_tile   = k_tile_1;
                        k_tile_1 = k_tile_2;
                    }
                    else
                    {
                        qk_gemm(i_k0, q_slice, k_tile);
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

                        qk_gemm(i_k0,
                                get_slice_tile(q_tile,
                                               sequence<0, i_k0 * kK0>{},
                                               sequence<kM0, (i_k0 + 1) * kK0>{}),
                                k_tile);
                        if constexpr(kPrefetchNextSlice)
                        {
                            using G0 = remove_cvref_t<decltype(gemm_0)>;
                            SchedSliceUnderNextSliceReads<G0::MIterPerWarp,
                                                          G0::MIterPerWarp * G0::NIterPerWarp,
                                                          i_k0.value == 0,
                                                          kKReadsPerFragment>();
                        }

                        k_tile = k_tile_switch;
                    });
                    // move back to the origin
                    move_tile_window(k_lds_read_window, {0, -kK0 * (k0_loops - 1)});
                }

                qk_gemm(number<k0_loops - 1>{},
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
                                      kReadAheadWmma,
                                      kKReadsPerFragment>();

            SetGemmPriority<false, kEdgeTrip>();
            if constexpr(kSingleK)
            {
                // every wave has read K_i: its buffer takes K_{i+1} (the last trip has none)
                if(has_next_tile)
                    block_sync_lds();
                issue_next_k();
            }
            // V_i's first slice (see kReadVBeforeSoftmax);
            // the sched_barrier keeps the reads ahead of the softmax.
            auto v_tile = decltype(LoadVTile(v_lds_read_window)){};
            if constexpr(kReadVBeforeSoftmaxOn<kEdgeTrip>)
            {
                v_tile = LoadVTile(v_lds_read_window);
                __builtin_amdgcn_sched_barrier(0);
            }

            // (past the QK gemm: held across it, it took the d128 fp8 128-row tile past its
            // VGPRs, 59 spilled)
            const index_t kv_tile_start = trip_start(i_total_loops);
            // the rows' q descales and, with block scales, the columns' k descales
            quant_scales.ScaleScores(s_acc, i_total_loops);

            // STAGE 2: scale_s, add bias, mask
            ApplyScaleAndBias(s_acc,
                              bias_dram_window,
                              q_origin,
                              kv_tile_start,
                              scale_s,
                              position_encoding,
                              logits_soft_cap,
                              logits_soft_cap_rcp);
            if constexpr(kEdgeTrip)
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

            // S{j}: each warp's rows over its keys (kKeySplit), or, exchanged through LDS
            // (kExchangeS), over all of the trip's
            const auto s_new = [&]() {
                if constexpr(kExchangeS)
                {
                    store_tile(s_write_lds_window, cast_tile<SMPLComputeDataType>(s_acc));
                    block_sync_lds();
                    return load_tile(s_read_lds_window);
                }
                else
                    return cast_tile<SMPLComputeDataType>(s_acc);
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

            // Exponent that p (and hence o_acc / l) is normalised against; an fp8 P sits
            // kPExp2Offset above it.
            const auto get_row_exp = [&](SMPLComputeDataType raw_m) {
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             BiasEnum == BlockAttentionBiasEnum::ALIBI || kHasLogitsSoftCap)
                {
                    return get_validated_m(raw_m) - kPExp2Offset;
                }
                else
                {
                    return scale_s * get_validated_m(raw_m) - kPExp2Offset;
                }
            };

            constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();

            // o_acc and l stay normalised to m_base, the row max as of the last rebase,
            // and are rescaled when a tile raises a row max by more than a factor of 2^0.75
            // (kRebaseThreshold, log2): p stays below 2^0.75 (an fp8 P's range, see
            // kPExp2Offset), and a row's largest p is 1 (exact in the 16-bit P) unless its
            // max rose less than that, when the old largest still weighs over half of it. A
            // base trailing the row max by up to 2^8 (skipping the row max on most tiles)
            // left the largest p anywhere up to 256, rounded in the 16-bit P: on bf16
            // (-init=0, 1024 x 1024, 100 seeds) up to 21% of the runs past the reference's
            // tolerance at d512, where a threshold of 1 stays with qr_tdm and the exact max
            // (one run, at the tolerance). Rebasing on every raise took the causal d64..d128
            // tiles 2-3% longer. The gap uses the raw maxima: a row whose m_base is -inf
            // always rebases, a fully masked row (NaN gap) never does.
            const auto row_gap = [&](auto i_idx) {
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             BiasEnum == BlockAttentionBiasEnum::ALIBI || kHasLogitsSoftCap)
                    return m[i_idx] - m_base[i_idx];
                else
                    return scale_s * (m[i_idx] - m_base[i_idx]);
            };
            constexpr auto kRebaseThreshold = SMPLComputeDataType{0.75f};
            const auto rebase_rows          = [&]() {
                bool need_rebase = false;
                sweep_tile_span(o_spans[I0], [&](auto idx0) {
                    need_rebase |= row_gap(make_tuple(idx0)) > kRebaseThreshold;
                });
                if(need_rebase)
                {
                    sweep_tile_span(o_spans[I0], [&](auto idx0) {
                        constexpr auto i_idx = make_tuple(idx0);
                        const auto gap       = row_gap(i_idx);
                        if(gap > kRebaseThreshold)
                        {
                            const auto tmp = exp2(-gap);
                            l(i_idx) *= tmp;
                            if constexpr(kLseRowSum)
                                l_lse(i_idx) *= tmp;
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
                // scale_s in one VGPR, which v_pk_fma_f32 broadcasts: from its SGPR the
                // compiler built four equal SGPR pairs, spilled to VGPR lanes and read back
                // every trip (fp8 d128 128-key trips, 2048 x 16384: +3%)
                auto scale_v = scale_s;
                asm volatile("" : "+v"(scale_v));
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
                                p_compute(i_j_idx) = exp2(scale_v * s_in[i_j_idx] - row_max);
                            }
                        }
                    });
                });
            };

            // P for gemm_1's A operand, rounded to PDataType (an fp8 P already carries its
            // scale, kPExp2Offset) and repacked (Base::RepackPForGemm1); l sums the rounded
            // values of an fp8 P (see kRowSumOnMatrixUnit).
            using G1    = remove_cvref_t<decltype(gemm_1)>;
            using PTile = decltype(Base::template RepackPForGemm1<G1>(CastP(p_compute)));
            [[maybe_unused]] PTile p_tile_early{};
            update_row_max(s_new);
            rebase_rows();
            compute_p(s_new);
            const auto make_p_tile = [&]() {
                return Base::template RepackPForGemm1<G1>(CastP(p_compute));
            };
            if constexpr(kRowMaxBySum)
            {
                // kEarlyPConvert: P in PDataType right after the exps
                if constexpr(kEarlyPConvert)
                    p_tile_early = make_p_tile();
                if constexpr(!kRowSumOnMatrixUnit)
                {
                    const auto part = BlockRowSum<false>(p_compute);
                    sweep_tile_span(o_spans[I0], [&](auto idx0) {
                        constexpr auto i_idx = make_tuple(idx0);
                        l(i_idx) += part[i_idx];
                    });
                }
            }

            // MakePForGemm1 reorders P's (MIter, KIter) blocks within a thread, which
            // lines up with gemm_1's A operand only with the warps on M or one MIter.
            static_assert(BlockFmhaShape::Gemm1BlockWarps::at(I1) == 1 ||
                          remove_cvref_t<decltype(gemm_1)>::MIterPerWarp == 1);
            // l (on the matrix unit, the rounded P gemm_1 takes) and an fp8 LSE's unrounded row
            // sum, of the undropped P
            const auto accumulate_row_sums = [&](const auto& p_rounded) {
                if constexpr(kRowSumOnMatrixUnit)
                    AccumulateRowSumOnMatrixUnit(l, p_rounded);
                if constexpr(kLseRowSum)
                {
                    const auto rowsum_p = BlockRowSum<false>(p_compute);
                    sweep_tile_span(o_spans[I0], [&](auto idx0) {
                        constexpr auto i_idx = make_tuple(idx0);
                        l_lse(i_idx) += rowsum_p[i_idx];
                    });
                }
            };
            if constexpr(Base::kHasDropout)
            {
                // l{j} sums the undropped P; dropout then scales the kept P by 1/(1-p) (an
                // fp8 P's 1/(1-p) is on O, see the kernel)
                if constexpr(kRowSumOnMatrixUnit)
                    accumulate_row_sums(make_p_tile());
                else
                {
                    const auto rowsum_p = BlockRowSum(p_compute);
                    sweep_tile_span(o_spans[I0], [&](auto idx0) {
                        constexpr auto i_idx = make_tuple(idx0);
                        l(i_idx) += rowsum_p[i_idx];
                    });
                }
                if constexpr(kHasSink)
                {
                    // past the sink tiles the keys jump to physical_seqlen_k_start
                    if(num_sink_loop > 0 && i_total_loops == num_sink_loop)
                        move_tile_window(randval_dram_window,
                                         {0, physical_seqlen_k_start - sink_seq_end});
                }
                // the randval shuffle's LDS, the arena's last region
                static_assert(BlockDropout::MakeRandValLdsBlockDescriptor<decltype(gemm_0)>()
                                  .get_element_space_size() <= Layout::kRandValBytes);
                dropout.template Run<decltype(gemm_0), SMPLComputeDataType, RandValOutputDataType>(
                    static_cast<char*>(smem_arena) + Layout::kRandValOffset,
                    kv_tile_start,
                    p_compute,
                    randval_dram_window);
            }
            auto p_tile = [&]() {
                if constexpr(kRowMaxBySum && kEarlyPConvert)
                    return p_tile_early;
                else
                    return make_p_tile();
            }();
            if constexpr(!Base::kHasDropout)
                accumulate_row_sums(p_tile);

            // l{j}; o_acc is already in p's m_base frame.
            if constexpr(!kRowMaxBySum && !Base::kHasDropout && !kRowSumOnMatrixUnit)
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

            if constexpr(!kReadVBeforeSoftmaxOn<kEdgeTrip>)
                v_tile = LoadVTile(v_lds_read_window);
            SetGemmPriority<true, kEdgeTrip>();

            // One kK1 slice of the PV gemm.
            const auto gemm_1_slice = [&](auto i_k1, const auto& v_slice) {
                quant_scales.Pv(gemm_1,
                                o_acc,
                                get_slice_tile(p_tile,
                                               sequence<0, i_k1 * kK1>{},
                                               sequence<kM0, (i_k1 + 1) * kK1>{}),
                                v_slice,
                                i_k1,
                                i_total_loops);
            };
            using Gemm1T = remove_cvref_t<decltype(gemm_1)>;
            // (the key split's one slice a trip takes KIterPerWarp warp gemms on each warp)
            constexpr index_t kWmmaPerPvSlice =
                Gemm1T::MIterPerWarp * Gemm1T::NIterPerWarp * Gemm1T::KIterPerWarp;

            if constexpr(1 < k1_loops)
            {
                static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
                    if constexpr(kPrefetchNextSlice)
                    {
                        // the next slice's reads go out before this slice's WMMAs
                        move_tile_window(v_lds_read_window, VSeqOffset(kK1));
                        auto v_tile_next = LoadVTile(v_lds_read_window);
                        gemm_1_slice(i_k1, v_tile);
                        SchedSliceUnderNextSliceReads<Gemm1T::MIterPerWarp,
                                                      kWmmaPerPvSlice,
                                                      i_k1.value == 0,
                                                      kVReadsPerFragment>();
                        v_tile = v_tile_next;
                    }
                    else
                    {
                        gemm_1_slice(i_k1, v_tile);

                        // loop over along the [V]alue Sequence length
                        move_tile_window(v_lds_read_window, VSeqOffset(kK1));
                        v_tile = LoadVTile(v_lds_read_window);
                    }
                });
                // move back to the origin
                move_tile_window(v_lds_read_window, VSeqOffset(-kK1 * (k1_loops - 1)));
            }

            gemm_1_slice(number<k1_loops - 1>{}, v_tile);

            // V reads run ahead as for K above.
            static_assert(Gemm1T::WarpGemm::kK * Gemm1T::KIterPerWarp * Gemm1T::KWarp == kK1,
                          "a PV slice is whole warp gemms on every warp on K");
            if constexpr(!kPrefetchNextSlice)
                SchedReadsAheadOfWmma<Gemm1T::MIterPerWarp,
                                      kWmmaPerPvSlice * k1_loops,
                                      kReadAheadWmma,
                                      kVReadsPerFragment>();
            SetGemmPriority<false, kEdgeTrip>();
        };

        if constexpr(kSplitEdgeTrips)
        {
            // A band mask leaves the masked tiles at the two ends of a block's key range: the
            // leading ones (a window's left edge), the trailing ones (the diagonal, a window's
            // right edge, a partial last tile); the interior trips run unmasked.
            const auto is_edge_trip = [&](index_t i) {
                const index_t k_start = physical_seqlen_k_start + i * kN0;
                bool edge             = i == num_total_loop - 1 &&
                            (physical_seqlen_k_end - physical_seqlen_k_start) % kN0 != 0;
                if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
                    edge = edge || NeedsScoreMask(mask, q_origin.at(I0), k_start);
                return edge;
            };
            index_t num_trailing_edge = 0;
            while(num_trailing_edge < num_total_loop &&
                  is_edge_trip(num_total_loop - 1 - num_trailing_edge))
                ++num_trailing_edge;
            index_t num_leading_edge = 0;
            while(num_leading_edge < num_total_loop - num_trailing_edge &&
                  is_edge_trip(num_leading_edge))
                ++num_leading_edge;
            for(; i_total_loops < num_leading_edge; ++i_total_loops)
                run_trip(bool_constant<true>{});
            for(; i_total_loops < num_total_loop - num_trailing_edge; ++i_total_loops)
                run_trip(bool_constant<false>{});
            for(; i_total_loops < num_total_loop; ++i_total_loops)
                run_trip(bool_constant<true>{});
        }
        else
        {
            for(; i_total_loops < num_total_loop; ++i_total_loops)
                run_trip(bool_constant<true>{});
        }

        // finish_row_sums: the lanes' parts of the row sums (whole rows on the matrix unit or
        // already synced each trip), then a finite sink_v's (sink_ptr) seed: a last rebase to
        // M = max(m_base, sink) takes o_acc and the sums to M's frame (a row without scores,
        // m_base -inf, to 0), and the sink adds 2^(sink - M) there, at most 2^kPExp2Offset
        if constexpr(kRowMaxBySum && !kRowSumOnMatrixUnit)
            ReduceRowSync(l, [](auto e0, auto e1) { return e0 + e1; });
        if constexpr(kLseRowSum)
            ReduceRowSync(l_lse, [](auto e0, auto e1) { return e0 + e1; });
        if constexpr(kKeySplit)
            MergeKeySplit(m_base, l, l_lse, o_acc, smem_arena, scale_s, quant_scales.TakeOUnit());
        if(__builtin_isinf_sign(sink_v) >= 0)
        {
            constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
            sweep_tile_span(o_spans[I0], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                // merged rows: their own head's sink logit
                const auto sink_m = [&]() {
                    if constexpr(Problem::kMergeNumHeadGroupsSeqLenQ)
                    {
                        if(quant_scales.args.sink != nullptr)
                        {
                            index_t row = 0;
                            sweep_tile_span(o_spans[I1], [&](auto idx1) {
                                row = get_x_indices_from_distributed_indices(
                                          decltype(o_acc)::get_tile_distribution(),
                                          make_tuple(idx0, idx1))
                                          .at(I0);
                            });
                            row = min(row + q_dram_block_window_tmp.get_window_origin().at(I0),
                                      quant_scales.args.rows - 1);
                            return SinkRowMax(
                                quant_scales.args.sink[row / quant_scales.args.rows_per_head] /
                                    scale_s,
                                scale_s);
                        }
                    }
                    return SinkRowMax(sink_v, scale_s);
                }();
                const auto big_m   = max(m_base[i_idx], sink_m);
                const auto to_exp2 = [&](auto d) {
                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                 BiasEnum == BlockAttentionBiasEnum::ALIBI || kHasLogitsSoftCap)
                        return d;
                    else
                        return scale_s * d;
                };
                const auto factor = exp2(to_exp2(m_base[i_idx] - big_m));
                const auto seed   = exp2(to_exp2(sink_m - big_m) + kPExp2Offset);
                m_base(i_idx)     = big_m;
                l(i_idx)          = l[i_idx] * factor + seed;
                if constexpr(kLseRowSum)
                    l_lse(i_idx) = l_lse[i_idx] * factor + seed;
                sweep_tile_span(o_spans[I1],
                                [&](auto idx1) { o_acc(make_tuple(idx0, idx1)) *= factor; });
            });
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
            if constexpr(kLseRowSum)
                StoreLse(lse_acc_dram_window_tmp, m_lse, l_lse, scale_s, lse_log2);
            else
                StoreLse(lse_acc_dram_window_tmp, m_lse, l, scale_s, lse_log2);
        }

        NormalizeO(o_acc, l, quant_scales.O());
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
               Dropout dropout                                         = {},
               QuantScaleArgs quant_scale_args                         = {},
               index_t num_splits                                      = 1,
               index_t i_split                                         = 0) const
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
                       dropout,
                       quant_scale_args,
                       num_splits,
                       i_split);
    }
};

} // namespace ck_tile
