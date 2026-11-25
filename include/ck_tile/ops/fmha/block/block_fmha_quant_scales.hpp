// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp"

namespace ck_tile {

// The quantization scales of a forward attention block, one component for every input width
// and scale kind, as BlockDropout is for dropout (used by qr_tdm_pingpong): the pipeline calls
// it at fixed points of its prologue and trip, and the kinds with nothing to do there compile
// to nothing.
//
//   kind                      scores (QK)                       O (PV)
//   none                      -                                 -
//   per-tensor / per-head     q*k in scale_s (kernel)           v on O (Args::o)
//   block                     q per row, k per column           v per 32 keys, o_acc rescaled
//   merged rows, per-head     q per row (its head's)            v on O
//   MX                        Q/K E8M0 WMMA scale operands      V's E8M0 WMMA scale operands
//
// A row's q descale (block, merged per-head) and a merged row's sink logit are its head's and
// query's: tile row r is query r % rows_per_head of head r / rows_per_head (merged query
// heads, packed GQA; else one head), and rows past `rows` take the last one's.
//
// The K/V scales of a trip (block: a descale per key, and per 32 keys; MX: four
// E8M0 bytes per key and QK slice, and per V column and 128 keys) are staged in LDS, three
// buffers in turn:
// the block copies trip i + 2's with async global->LDS loads in trip i, once trip i - 1 has
// read the buffer and trip i its own (IssueTrip), and the barrier of trip i + 2 makes them
// visible (WaitTrip before it). Two trips cover the copies' global latency; one did not
// (ATT, d128 128-row MX: its wait took 3% of the cycles). The copies go out once the trip
// has read its own scales (ReadTrip, PinTrip): read at the gemms instead, each read's dscnt wait
// also retired the K/V fragment reads ahead of it, and the reads could not move above copies
// into the same LDS (ATT, d128 128-row: s_wait_dscnt 19% of the cycles). A global load per
// trip, the other way to get them, stalled the gemm on its latency.
template <typename Problem, typename QkGemm, typename PvGemm>
struct BlockFmhaQuantScales
{
    using Shape                        = typename Problem::BlockFmhaShape;
    using SaccTile                     = decltype(std::declval<QkGemm>().MakeCBlockTile());
    static constexpr auto QScaleEnum   = Problem::QScaleEnum;
    static constexpr index_t kN0       = Shape::kN0;
    static constexpr index_t kK0       = Shape::kK0;
    static constexpr index_t kN1       = Shape::kN1;
    static constexpr index_t kK1       = Shape::kK1;
    static constexpr index_t kQkSlices = Shape::kQKHeaddim / kK0;
    static constexpr index_t kThreads  = Shape::NumWarps * get_warp_size();
    static constexpr bool kMx          = QScaleEnum == BlockAttentionQuantScaleEnum::MX;
    static constexpr bool kBlock       = QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE;
    // rows with their own q descales (scale_s holds none then)
    static constexpr bool kRowQDescale =
        kBlock || (Problem::kMergeNumHeadGroupsSeqLenQ &&
                   QScaleEnum == BlockAttentionQuantScaleEnum::PERHEAD);
    // MX: an E8M0 scale per 32 elements along each gemm's K, a lane's int32 the bytes of its
    // row's (column's) slice: QK slices and PV warp gemms of 128 or 64 (two bytes, the WMMA
    // zero-padded to 128); P takes 1.0.
    static constexpr index_t kPvK = PvGemm::WarpGemm::kK;
    static_assert(!kMx || ((kK0 == 128 || kK0 == 64) && (kPvK == 128 || kPvK == 64)),
                  "MX takes 64- or 128-deep QK slices and PV warp gemms");
    // MX V scale operands per PV warp gemm's K of a trip: kVChunks in the trip,
    // kVChunksPerWarp read by each warp (all of them, or with gemm_1's warps on K, see
    // qr_tdm_pingpong's key split, its own range's). They are staged a dword (128 keys) at
    // a time, kVDwords a trip from the trip's 128-key boundary (trips start at multiples of
    // kN0), and a 64-deep chunk takes its half.
    static constexpr index_t kKeyWarps = PvGemm::KWarp;
    static constexpr index_t kVChunks  = kMx ? kN0 / kPvK : 1;
    static constexpr index_t kVDwords  = kMx ? max(kN0 / 128, index_t{1}) : 1;
    static_assert(kK1 % kKeyWarps == 0 && (!kMx || kVChunks % kKeyWarps == 0));
    static_assert(!kMx || kPvK == 128 || (kKeyWarps == 1 && (kN0 == 64 || kN0 % 128 == 0)));
    static constexpr index_t kVChunksPerWarp = kVChunks / kKeyWarps;
    // block v descales: one per 32 keys (k/v blocks of whole 32-key chunks), kVBlockChunks in a
    // warp's part of a PV slice (with the key split each warp takes 1 / kKeyWarps of it)
    static constexpr index_t kVBlockKeys   = 32;
    static constexpr index_t kVBlockSlots  = kN0 / kVBlockKeys;
    static constexpr index_t kVBlockChunks = kK1 / kKeyWarps / kVBlockKeys;
    static_assert(!kBlock || (kK1 / kKeyWarps) % kVBlockKeys == 0);
    static constexpr index_t kQkScaleBytes = kK0 / 32;
    static constexpr int32_t kE8m0One      = 0x7f7f7f7f;

    // LDS dwords of one trip's staged K and V scales
    static constexpr index_t kTripKDwords = kMx ? kN0 * kQkSlices : kBlock ? kN0 : 0;
    static constexpr index_t kTripVDwords = kMx ? kN1 * kVDwords : kBlock ? kVBlockSlots : 0;
    static constexpr index_t kTripDwords  = kTripKDwords + kTripVDwords;
    static constexpr bool kStagesTrips    = kTripDwords > 0;
    // trips in flight: the one read, the next, and the one whose copies go out
    static constexpr index_t kTripBuffers = 3;

    // What the kernel hands over. Pointers are this workgroup's (head, batch offsets applied).
    struct Args
    {
        // the tile's rows (see above) and the merged rows' sink logits, per head
        index_t rows_per_head = 1;
        index_t rows          = 1;
        const float* sink     = nullptr;
        // O's factor: per-tensor/per-head v descale (and dropout's 1 / (1 - p_drop))
        float o = 1.f;
        // block / merged per-head: float q descales at q[head * q_head_stride + query / q_block];
        // MX: E8M0 rows at q[head * q_head_stride + query * stride_q]
        const void* q         = nullptr;
        index_t q_head_stride = 0;
        index_t q_block       = 1;
        // block: float k/v descales per kv_block keys; MX: E8M0 rows, k[key * stride_k + d / 32]
        // and v[column * stride_v + key / 32]. keys: seqlen_k, past which the last key's stand in.
        const void* k    = nullptr;
        const void* v    = nullptr;
        index_t kv_block = 1;
        index_t keys     = 1;
        // MX row strides and lengths: ceil(hdim_q / 32) bytes of a Q/K row (a padded head dim's
        // zero columns take its last byte's scale: a finite one, no E8M0 NaN from past the row),
        // hdim_v V rows
        index_t stride_q = 0;
        index_t stride_k = 0;
        index_t stride_v = 0;
        index_t qk_bytes = 1;
        index_t cols_v   = 1;
    };

    // The row (dim 0) of each of a tile's dim-0 spans this lane holds, in sweep order: a WMMA
    // operand's lane holds row lane % 16 of each 16-row band, which its scale operand describes.
    template <index_t N, typename Tile>
    CK_TILE_DEVICE static array<index_t, N> SpanRows()
    {
        array<index_t, N> rows{};
        index_t i            = 0;
        constexpr auto spans = Tile::get_distributed_spans();
        sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
            index_t r = 0;
            sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                r = get_x_indices_from_distributed_indices(Tile::get_tile_distribution(),
                                                           make_tuple(idx0, idx1))
                        .at(number<0>{});
            });
            rows(i++) = r;
        });
        return rows;
    }

    Args args;

    // ---- prologue -------------------------------------------------------------------------

    // The rows' q descales (kRowQDescale) and MX Q scale operands; row0: the tile's first row.
    // KTile, VTile: the gemms' B operand tiles (their lanes' K rows and V columns).
    template <typename QTile, typename KTile, typename VTile>
    CK_TILE_DEVICE void Begin(index_t row0)
    {
        if constexpr(kRowQDescale)
        {
            if(args.q != nullptr)
            {
                const auto q =
                    cast_pointer_to_constant_address_space(static_cast<const float*>(args.q));
                constexpr auto spans = SaccTile::get_distributed_spans();
                sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
                    // the row of idx0 (any of its columns names it)
                    index_t r = 0;
                    sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                        r = get_x_indices_from_distributed_indices(
                                SaccTile::get_tile_distribution(), make_tuple(idx0, idx1))
                                .at(number<0>{});
                    });
                    const index_t row        = min(row0 + r, args.rows - 1);
                    q_row_(make_tuple(idx0)) = q[(row / args.rows_per_head) * args.q_head_stride +
                                                 (row % args.rows_per_head) / args.q_block];
                });
            }
            else
                set_tile(q_row_, 1.f);
        }
        if constexpr(kMx)
        {
            const auto rows  = SpanRows<QkGemm::MIterPerWarp, QTile>();
            const auto* q    = static_cast<const uint8_t*>(args.q);
            const bool words = QkWords(q, args.stride_q | args.q_head_stride);
            static_for<0, QkGemm::MIterPerWarp, 1>{}([&](auto i_m) {
                const index_t row          = min(row0 + rows[i_m], args.rows - 1);
                const uint8_t* const q_row = q + (row / args.rows_per_head) * args.q_head_stride +
                                             (row % args.rows_per_head) * args.stride_q;
                static_for<0, kQkSlices, 1>{}([&](auto sl) {
                    mx_q_(number<i_m * kQkSlices + sl>{}) =
                        words ? *reinterpret_cast<const int32_t*>(q_row + sl * kQkScaleBytes)
                              : Bytes(q_row, sl * kQkScaleBytes, args.qk_bytes);
                });
            });
        }
        if constexpr(kStagesTrips)
        {
            __shared__ int32_t trip_lds[kTripBuffers * kTripDwords];
            lds_    = trip_lds;
            k_rows_ = SpanRows<QkGemm::NIterPerWarp, KTile>();
            v_cols_ = SpanRows<PvGemm::NIterPerWarp, VTile>();
            if constexpr(kMx)
            {
                k_words_ = QkWords(static_cast<const uint8_t*>(args.k), args.stride_k);
                v_words_ = Words(static_cast<const uint8_t*>(args.v), args.stride_v);
                // this thread's rows of the copies, fixed across trips
                const index_t t = get_thread_id();
                static_for<0, kKSlots, 1>{}(
                    [&](auto i) { k_lane_(i) = (i * kThreads + t) * args.stride_k; });
                static_for<0, kVSlots, 1>{}([&](auto i) {
                    v_lane_(i) = min(i * kThreads + t, args.cols_v - 1) * args.stride_v;
                });
            }
        }
    }

    // ---- trips ----------------------------------------------------------------------------

    // Copies trip `trip`'s K/V scales (its first key kv_start) into its LDS buffer.
    CK_TILE_DEVICE void IssueTrip([[maybe_unused]] index_t kv_start, [[maybe_unused]] index_t trip)
    {
        if constexpr(kStagesTrips)
        {
            int32_t* const k_dst = lds_ + (trip % kTripBuffers) * kTripDwords;
            int32_t* const v_dst = k_dst + kTripKDwords;
            if constexpr(kMx)
            {
                const auto* k = static_cast<const uint8_t*>(args.k);
                const auto* v = static_cast<const uint8_t*>(args.v);
                // V's dwords from the trip's 128-key boundary
                const index_t v_start = kv_start & ~127;
                // a whole trip of aligned rows: dwords at a trip offset (uniform) from each
                // thread's rows
                if(k_words_ && v_words_ && v_start + kVDwords * 128 <= args.keys)
                {
                    const uint8_t* const k_trip = k + kv_start * args.stride_k;
                    const uint8_t* const v_trip = v + v_start / 32;
                    const index_t t             = get_thread_id();
                    static_for<0, kKSlots, 1>{}([&](auto i) {
                        if(kN0 % kThreads == 0 || i * kThreads + t < kN0)
                            static_for<0, kQkSlices, 1>{}([&](auto sl) {
                                CopyDword(k_trip + k_lane_[i] + sl * kQkScaleBytes,
                                          k_dst + sl * kN0 + i * kThreads + t);
                            });
                    });
                    static_for<0, kVSlots, 1>{}([&](auto i) {
                        if(kN1 % kThreads == 0 || i * kThreads + t < kN1)
                            static_for<0, kVDwords, 1>{}([&](auto sl) {
                                CopyDword(v_trip + v_lane_[i] + sl * (128 / 32),
                                          v_dst + sl * kN1 + i * kThreads + t);
                            });
                    });
                }
                else
                {
                    ForEachSlot<kN0>([&](index_t key) {
                        const uint8_t* const k_row =
                            k + min(kv_start + key, args.keys - 1) * args.stride_k;
                        static_for<0, kQkSlices, 1>{}([&](auto sl) {
                            CopyDword(k_row,
                                      sl * kQkScaleBytes,
                                      args.qk_bytes,
                                      k_words_,
                                      k_dst + sl * kN0 + key);
                        });
                    });
                    // whole dwords where the trip's keys fill them (the last trip's may not)
                    const bool v_words    = v_words_ && v_start + kVDwords * 128 <= args.keys;
                    const index_t v_bytes = integer_divide_ceil(args.keys, 32);
                    ForEachSlot<kN1>([&](index_t col) {
                        const uint8_t* const v_row = v + min(col, args.cols_v - 1) * args.stride_v;
                        static_for<0, kVDwords, 1>{}([&](auto sl) {
                            CopyDword(v_row,
                                      (v_start + sl * 128) / 32,
                                      v_bytes,
                                      v_words,
                                      v_dst + sl * kN1 + col);
                        });
                    });
                }
            }
            else // block: the descale of each key's block and of each PV slice's
            {
                const auto* k = static_cast<const float*>(args.k);
                const auto* v = static_cast<const float*>(args.v);
                ForEachSlot<kN0>([&](index_t key) {
                    CopyDword(k + min(kv_start + key, args.keys - 1) / args.kv_block, k_dst + key);
                });
                ForEachSlot<kVBlockSlots>([&](index_t sl) {
                    CopyDword(v + min(kv_start + sl * kVBlockKeys, args.keys - 1) / args.kv_block,
                              v_dst + sl);
                });
            }
#if CK_TILE_HAS_ASYNCMARK
            // the wait in WaitTrip counts marks: an unmarked copy would not be waited on
            asyncmark();
#endif
        }
    }

    // Before the barrier of a trip: its copies landed (the barrier shares them); the next
    // trip's, issued a trip later, may still be in flight. The last trip has no next one in
    // flight, so it waits for every copy (with <1> its own stayed outstanding, read early:
    // intermittent errors on a single trip, MX packed GQA d64, 77 keys).
    CK_TILE_DEVICE static void WaitTrip(bool last_trip)
    {
        if constexpr(kStagesTrips)
        {
            if(last_trip)
                s_wait_asynccnt<0>();
            else
                s_wait_asynccnt<1>();
        }
        else
            ignore = last_trip;
    }

    // Past a trip's barrier: its MX scale operands, read into VGPRs (ReadTrip), landed
    // before its gemms (PinTrip: left alone the compiler sinks each read to its WMMA). The
    // next copies go out after PinTrip, so no read waits on them.
    CK_TILE_DEVICE void ReadTrip([[maybe_unused]] index_t trip, [[maybe_unused]] index_t kv_start)
    {
        if constexpr(kMx)
        {
            const int32_t* const k = Trip(trip);
            const int32_t* const v = k + kTripKDwords;
            static_for<0, QkGemm::NIterPerWarp, 1>{}([&](auto n) {
                static_for<0, kQkSlices, 1>{}([&](auto sl) {
                    mx_k_(number<n * kQkSlices + sl>{}) = k[sl * kN0 + k_rows_[n]];
                });
            });
            const index_t chunk0 =
                kKeyWarps > 1 ? (get_warp_id() % kKeyWarps) * kVChunksPerWarp : 0;
            static_for<0, PvGemm::NIterPerWarp, 1>{}([&](auto n) {
                static_for<0, kVChunksPerWarp, 1>{}([&](auto c) {
                    // the chunk's first key, from the trip's 128-key boundary
                    const index_t key = (kv_start & 127) + (chunk0 + c) * kPvK;
                    const int32_t w   = v[(key / 128) * kN1 + v_cols_[n]];
                    // a 64-deep chunk: its two bytes, the upper (zero operands') 1.0
                    mx_v_(number<n * kVChunksPerWarp + c>{}) =
                        kPvK == 128
                            ? w
                            : static_cast<int32_t>(
                                  ((static_cast<uint32_t>(w) >> (key & 64 ? 16 : 0)) & 0xffffu) |
                                  0x7f7f0000u);
                });
            });
        }
    }
    CK_TILE_DEVICE void PinTrip()
    {
        if constexpr(kMx)
        {
            static_for<0, QkGemm::NIterPerWarp * kQkSlices, 1>{}([&](auto i) {
                asm volatile("" : "+v"(mx_k_(i)));
            });
            static_for<0, PvGemm::NIterPerWarp * kVChunksPerWarp, 1>{}([&](auto i) {
                asm volatile("" : "+v"(mx_v_(i)));
            });
        }
    }

    // gemm_0 on QK slice i_k0 of trip `trip`.
    template <typename Gemm, typename SAcc, typename QSlice, typename KSlice, typename IK0>
    CK_TILE_DEVICE void Qk(const Gemm& gemm,
                           SAcc& s_acc,
                           const QSlice& q_slice,
                           const KSlice& k_slice,
                           IK0,
                           [[maybe_unused]] index_t trip) const
    {
        if constexpr(kMx)
        {
            gemm(
                s_acc,
                q_slice,
                k_slice,
                [&](auto i_m, auto) { return mx_q_[i_m * kQkSlices + IK0::value]; },
                [&](auto i_n, auto) { return mx_k_[i_n * kQkSlices + IK0::value]; });
        }
        else
            gemm(s_acc, q_slice, k_slice);
    }

    // The scores' row q descales and, with block scales, column k descales of trip `trip`.
    CK_TILE_DEVICE void ScaleScores([[maybe_unused]] SaccTile& s_acc,
                                    [[maybe_unused]] index_t trip) const
    {
        if constexpr(kRowQDescale)
        {
            constexpr auto spans = SaccTile::get_distributed_spans();
            // each column's k descale (any row of the column names its key)
            [[maybe_unused]] array<float, kN0> k_col{};
            if constexpr(kBlock)
            {
                const auto* k = reinterpret_cast<const float*>(Trip(trip));
                index_t j     = 0;
                sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
                    if(j > 0)
                        return;
                    sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                        k_col(j++) =
                            k[get_x_indices_from_distributed_indices(
                                  SaccTile::get_tile_distribution(), make_tuple(idx0, idx1))
                                  .at(number<1>{})];
                    });
                });
            }
            sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
                const float f = q_row_[make_tuple(idx0)];
                index_t j     = 0;
                sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                    if constexpr(kBlock)
                        s_acc(make_tuple(idx0, idx1)) *= f * k_col[j++];
                    else
                        s_acc(make_tuple(idx0, idx1)) *= f;
                });
            });
        }
    }

    // gemm_1 on PV slice i_k1 of trip `trip`. With block scales o_acc is held in units of the
    // current v block descale (O takes it at the end, see O()); the slice's block takes it
    // over, and a zero descale's keys add nothing (their P stays in l). A slice whose 32-key
    // chunks lie in blocks of different descales takes a gemm a chunk, P zeroed past it.
    template <typename Gemm, typename OAcc, typename PSlice, typename VSlice, typename IK1>
    CK_TILE_DEVICE void Pv(const Gemm& gemm,
                           OAcc& o_acc,
                           const PSlice& p_slice,
                           const VSlice& v_slice,
                           IK1,
                           [[maybe_unused]] index_t trip)
    {
        if constexpr(kBlock)
        {
            const index_t warp_k = kKeyWarps > 1 ? index_t(get_warp_id()) % kKeyWarps : 0;
            const float* const v = reinterpret_cast<const float*>(Trip(trip) + kTripKDwords) +
                                   (IK1::value * kKeyWarps + warp_k) * kVBlockChunks;
            // descales are staged as stored: a block's are bitwise equal
            const auto same = [](float a, float b) {
                return bit_cast<uint32_t>(a) == bit_cast<uint32_t>(b);
            };
            const auto block_gemm = [&](float v_block, const auto& p) {
                if(v_block == 0.f)
                    return;
                if(!same(v_block, v_block_))
                {
                    const float ratio = v_block_ / v_block;
                    tile_elementwise_inout([ratio](auto& x) { x *= ratio; }, o_acc);
                    v_block_ = v_block;
                }
                gemm(o_acc, p, v_slice);
            };
            bool uniform = true;
            static_for<1, kVBlockChunks, 1>{}([&](auto c) { uniform &= same(v[c], v[0]); });
            if(uniform)
                block_gemm(v[0], p_slice);
            else
                static_for<0, kVBlockChunks, 1>{}(
                    [&](auto c) { block_gemm(v[c], KeepChunk(p_slice, c)); });
            return;
        }
        if constexpr(kMx)
        {
            gemm(
                o_acc,
                p_slice,
                v_slice,
                [](auto, auto) { return kE8m0One; },
                [&](auto i_n, auto i_k) {
                    return mx_v_[i_n * kVChunksPerWarp + IK1::value * PvGemm::KIterPerWarp + i_k];
                });
        }
        else
            gemm(o_acc, p_slice, v_slice);
    }

    // P with the keys of the warp's 32-key chunk c alone
    template <typename PSlice, typename IC>
    CK_TILE_DEVICE static PSlice KeepChunk(PSlice p, IC)
    {
        constexpr auto spans = PSlice::get_distributed_spans();
        sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                const index_t key = get_x_indices_from_distributed_indices(
                                        PSlice::get_tile_distribution(), make_tuple(idx0, idx1))
                                        .at(number<1>{});
                if((key / kVBlockKeys) % kVBlockChunks != IC::value)
                    p(make_tuple(idx0, idx1)) = type_convert<typename PSlice::DataType>(0.f);
            });
        });
        return p;
    }

    // ---- end ------------------------------------------------------------------------------

    // O's factor
    CK_TILE_DEVICE float O() const { return kBlock ? args.o * v_block_ : args.o; }
    // o_acc's unit (the v block descale it is held in), taken out of O()'s factor: the key
    // split merges the warps' o_acc, each in its own unit, in plain units.
    CK_TILE_DEVICE float TakeOUnit()
    {
        const float unit = kBlock ? v_block_ : 1.f;
        v_block_         = 1.f;
        return unit;
    }

    private:
    // the copies' thread slots (rows of K, columns of V)
    static constexpr index_t kKSlots = integer_divide_ceil(kN0, kThreads);
    static constexpr index_t kVSlots = integer_divide_ceil(kN1, kThreads);

    // a value per score row
    using RowTile = decltype(make_static_distributed_tensor<float>(
        make_static_tile_distribution(detail::make_reduce_tile_distribution_encoding(
            SaccTile::get_tile_distribution().get_static_tile_distribution_encoding(),
            sequence<1>{}))));

    // f(i) for i in [0, N), the block's threads taking one each in turn
    template <index_t N, typename F>
    CK_TILE_DEVICE static void ForEachSlot(const F& f)
    {
        const index_t t = get_thread_id();
        static_for<0, integer_least_multiple(N, kThreads), kThreads>{}([&](auto i0) {
            if(N % kThreads == 0 || i0 + t < N)
                f(i0 + t);
        });
    }

    CK_TILE_DEVICE const int32_t* Trip(index_t trip) const
    {
        return lds_ + (trip % kTripBuffers) * kTripDwords;
    }

    CK_TILE_DEVICE static bool Words(const uint8_t* p, index_t stride)
    {
        return ((reinterpret_cast<uintptr_t>(p) | static_cast<uintptr_t>(stride)) & 3) == 0;
    }
    // Q/K scale rows read a dword a slice: whole slices of four bytes in word-aligned rows
    // that hold them (K64 slices take two bytes, read bytewise)
    CK_TILE_DEVICE bool QkWords(const uint8_t* p, index_t stride) const
    {
        return kQkScaleBytes % 4 == 0 && Words(p, stride) &&
               kQkSlices * kQkScaleBytes <= args.qk_bytes;
    }

    // row[b0, b0 + 4), the bytes past the row's last (row_bytes) taking that last one's
    CK_TILE_DEVICE static int32_t Bytes(const uint8_t* row, index_t b0, index_t row_bytes)
    {
        int32_t s = 0;
        static_for<0, 4, 1>{}(
            [&](auto b) { s |= static_cast<int32_t>(row[min(b0 + b, row_bytes - 1)]) << (8 * b); });
        return s;
    }

    using gint_t  = __attribute__((address_space(1))) int32_t;
    using gchar_t = __attribute__((address_space(1))) char;
    using lint_t  = __attribute__((address_space(3))) int32_t;
    using lchar_t = __attribute__((address_space(3))) char;
    CK_TILE_DEVICE static lint_t* ToLds(int32_t* p)
    {
        return reinterpret_cast<lint_t*>(
            static_cast<uintptr_t>(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p))));
    }
    template <typename T>
    CK_TILE_DEVICE static gchar_t* ToGlobal(const T* p)
    {
        return __builtin_bit_cast(gchar_t*, const_cast<T*>(p));
    }
    // an async dword copy
    template <typename T>
    CK_TILE_DEVICE static void CopyDword(const T* src, int32_t* dst)
    {
        __builtin_amdgcn_global_load_async_to_lds_b32(
            __builtin_bit_cast(gint_t*, ToGlobal(src)), ToLds(dst), 0, 0);
    }
    // dst[0, 4) <- row[b0, b0 + 4): a dword, or bytes clamped to the row's last
    CK_TILE_DEVICE static void
    CopyDword(const uint8_t* row, index_t b0, index_t row_bytes, bool word, int32_t* dst)
    {
        if(word)
            CopyDword(row + b0, dst);
        else
            static_for<0, 4, 1>{}([&](auto b) {
                __builtin_amdgcn_global_load_async_to_lds_b8(
                    ToGlobal(row + min(b0 + b, row_bytes - 1)),
                    reinterpret_cast<lchar_t*>(ToLds(dst)) + b,
                    0,
                    0);
            });
    }

    // per-lane state
    [[maybe_unused]] RowTile q_row_{};
    [[maybe_unused]] array<int32_t, kMx ? QkGemm::MIterPerWarp * kQkSlices : 1> mx_q_{};
    [[maybe_unused]] array<int32_t, kMx ? QkGemm::NIterPerWarp * kQkSlices : 1> mx_k_{};
    [[maybe_unused]] array<int32_t, kMx ? PvGemm::NIterPerWarp * kVChunksPerWarp : 1> mx_v_{};
    [[maybe_unused]] array<index_t, kMx ? kKSlots : 1> k_lane_{};
    [[maybe_unused]] array<index_t, kMx ? kVSlots : 1> v_lane_{};
    [[maybe_unused]] array<index_t, kStagesTrips ? QkGemm::NIterPerWarp : 1> k_rows_{};
    [[maybe_unused]] array<index_t, kStagesTrips ? PvGemm::NIterPerWarp : 1> v_cols_{};
    [[maybe_unused]] int32_t* lds_  = nullptr;
    [[maybe_unused]] bool k_words_  = false;
    [[maybe_unused]] bool v_words_  = false;
    [[maybe_unused]] float v_block_ = 1.f;
};

} // namespace ck_tile
