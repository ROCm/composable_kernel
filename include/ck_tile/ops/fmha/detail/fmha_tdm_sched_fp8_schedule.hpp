// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_fp8_config.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_schedule.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_schedule_types.hpp"

namespace ck_tile {

// QK reads as in the 16-bit schedule (last QK stage: V); the PV stage packs its next-K reads.
template <typename Geometry>
struct FmhaTdmSchedFp8EvenlyInterleavedReadSchedule
    : FmhaTdmSchedEvenlyInterleavedReadSchedule<Geometry>
{
    using Base = FmhaTdmSchedEvenlyInterleavedReadSchedule<Geometry>;
    using Base::kNumPvStages;
    using Base::kPvWmmasPerStage;
    using typename Base::Kind;

    // PV next-K reads: second half packed into the last rows, a few WMMAs after its V reuse.
    struct PvKRows
    {
        static constexpr index_t kLoads = Geometry::kKSuLoadCount;
        static constexpr index_t kHalf  = kLoads / 2;
        // Tail: the last five rows; at D64/D192 it starts after the first half instead.
        static constexpr index_t kTailRows = 5;
        static constexpr index_t kTailFirstRow =
            kHalf <= kPvWmmasPerStage - kTailRows ? kPvWmmasPerStage - kTailRows : kHalf;
        // Two reads per tail row, or as many as D192's twelve-read half needs over four rows.
        static constexpr index_t kMaxLoadsPerRow =
            max(index_t{2},
                (kLoads - kHalf + kPvWmmasPerStage - kTailFirstRow - 1) /
                    (kPvWmmasPerStage - kTailFirstRow));
        struct Row
        {
            index_t accesses[kMaxLoadsPerRow]{};
            index_t size = 0;
        };
        Row rows[kPvWmmasPerStage]{};
        static_assert(kLoads - kHalf <= kMaxLoadsPerRow * (kPvWmmasPerStage - kTailFirstRow));

        CK_TILE_HOST_DEVICE constexpr PvKRows()
        {
            for(index_t a = 0; a < kHalf; ++a)
                rows[a].accesses[rows[a].size++] = a;
            constexpr index_t tail_rows = kPvWmmasPerStage - kTailFirstRow;
            for(index_t a = kHalf; a < kLoads; ++a)
            {
                const index_t r = kTailFirstRow + (a - kHalf) * tail_rows / (kLoads - kHalf);
                rows[r].accesses[rows[r].size++] = a;
            }
        }
    };

    template <index_t Stage>
    CK_TILE_HOST_DEVICE static constexpr auto MakePvRows()
    {
        static_assert(Stage >= 0 && Stage < kNumPvStages);
        if constexpr(Stage == kNumPvStages - 1)
            return PvKRows{};
        else
            return FmhaTdmSchedActiveRows<kPvWmmasPerStage, Geometry::kVStageLoadCount>{};
    }

    // The first PV row reads before its WMMA, so its load waits only on the P conversions.
    template <index_t Stage, index_t Wmma>
    CK_TILE_HOST_DEVICE static constexpr bool PvReadsFirst()
    {
        return Stage == kNumPvStages - 1 && Wmma == 0;
    }

    // Redefined because the base visitor reads the base MakePvRows.
    template <index_t Stage, index_t Wmma, bool SecondHalf, typename Visitor>
    CK_TILE_HOST_DEVICE static constexpr void VisitPvRowHalf(Visitor&& visitor)
    {
        static_assert(Wmma >= 0 && Wmma < kPvWmmasPerStage);
        constexpr auto row  = MakePvRows<Stage>().rows[Wmma];
        constexpr auto kind = Stage == kNumPvStages - 1 ? Kind::KRead : Kind::VRead;
        static_for<SecondHalf ? row.size / 2 : 0, SecondHalf ? row.size : row.size / 2, 1>{}(
            [&](auto i) {
                visitor(std::integral_constant<Kind, kind>{}, number<row.accesses[i]>{});
            });
    }
};

// D256: the last QK stage reads the first V chunks; PV streams the rest kLead chunks ahead.
template <typename Geometry>
struct FmhaTdmSchedFp8D256StreamedVReadSchedule
{
    using Kind                                = FmhaTdmSchedLoadKind;
    static constexpr index_t kNumQkStages     = Geometry::kQkStages;
    static constexpr index_t kNumPvStages     = Geometry::kPvStages;
    static constexpr index_t kQkWmmasPerStage = Geometry::kQkWmmasPerStage;
    static constexpr index_t kPvWmmasPerStage = Geometry::kPvWmmasPerStage;
    static constexpr index_t kChunkLoads      = Geometry::kVStageLoadCount / Geometry::kPvNIter;
    static constexpr index_t kMIter           = Geometry::kPvMIter;
    // PV row r runs V chunk r / kMIter and reads chunk r / kMIter + kLead to hide LDS latency.
    static constexpr index_t kLead = 6;
    // Next-K group 0 reads go one per row from here, behind the V stream.
    static constexpr index_t kKFirstRow = 16;
    static_assert(kNumPvStages == 1 && kPvWmmasPerStage == Geometry::kPvNIter * kMIter);

    template <index_t Stage>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQkRows()
    {
        static_assert(Stage >= 0 && Stage < kNumQkStages);
        constexpr index_t loads =
            Stage == kNumQkStages - 1 ? kLead * kChunkLoads : Geometry::kKSuLoadCount;
        return FmhaTdmSchedActiveRows<kQkWmmasPerStage, loads>{};
    }

    struct PvRows
    {
        static constexpr index_t kMaxLoadsPerRow = 8;
        struct Row
        {
            index_t accesses[kMaxLoadsPerRow]{};
            bool is_k[kMaxLoadsPerRow]{};
            index_t size = 0;
        };
        Row rows[kPvWmmasPerStage]{};

        CK_TILE_HOST_DEVICE constexpr PvRows()
        {
            constexpr index_t v_loads = Geometry::kVStageLoadCount - kLead * kChunkLoads;
            constexpr index_t v_rows  = kPvWmmasPerStage - kLead * kMIter;
            for(index_t a = 0; a < v_loads; ++a)
            {
                Row& r               = rows[a * v_rows / v_loads];
                r.accesses[r.size++] = kLead * kChunkLoads + a;
            }
            constexpr index_t k_loads = Geometry::kKSuLoadCount;
            constexpr index_t k_rows  = kPvWmmasPerStage - kKFirstRow;
            for(index_t a = 0; a < k_loads; ++a)
            {
                Row& r               = rows[kKFirstRow + a * k_rows / k_loads];
                r.is_k[r.size]       = true;
                r.accesses[r.size++] = a;
            }
        }
    };

    template <index_t Stage, index_t Wmma>
    CK_TILE_HOST_DEVICE static constexpr bool PvReadsFirst()
    {
        return false;
    }

    template <index_t Stage, index_t Wmma, bool SecondHalf, typename Visitor>
    CK_TILE_HOST_DEVICE static constexpr void VisitQkRowHalf(Visitor&& visitor)
    {
        static_assert(Wmma >= 0 && Wmma < kQkWmmasPerStage);
        constexpr auto row  = MakeQkRows<Stage>().rows[Wmma];
        constexpr auto kind = Stage == kNumQkStages - 1 ? Kind::VRead : Kind::KRead;
        static_for<SecondHalf ? row.size / 2 : 0, SecondHalf ? row.size : row.size / 2, 1>{}(
            [&](auto i) {
                visitor(std::integral_constant<Kind, kind>{}, number<row.accesses[i]>{});
            });
    }

    template <index_t Stage, index_t Wmma, bool SecondHalf, typename Visitor>
    CK_TILE_HOST_DEVICE static constexpr void VisitPvRowHalf(Visitor&& visitor)
    {
        static_assert(Wmma >= 0 && Wmma < kPvWmmasPerStage);
        constexpr auto row = PvRows{}.rows[Wmma];
        static_for<SecondHalf ? row.size / 2 : 0, SecondHalf ? row.size : row.size / 2, 1>{}(
            [&](auto i) {
                visitor(std::integral_constant < Kind,
                        row.is_k[i] ? Kind::KRead : Kind::VRead > {},
                        number<row.accesses[i]>{});
            });
    }
};

// The 16-bit WMMA-first placement, except that the read schedule may put PV row reads first.
template <typename ReadPlacement_>
struct FmhaTdmSchedFp8WmmaFirstPlacement : FmhaTdmSchedWmmaFirstPlacement<ReadPlacement_>
{
    using Base          = FmhaTdmSchedWmmaFirstPlacement<ReadPlacement_>;
    using ReadPlacement = ReadPlacement_;
    using Point         = FmhaTdmSchedSchedulePoint;

    template <typename Mode,
              index_t Stage,
              index_t Wmma,
              typename WmmaEmitter,
              typename ReadEmitter,
              typename FenceEmitter,
              typename RescaleEmitter>
    CK_TILE_HOST_DEVICE static constexpr void RunPvRow(WmmaEmitter& emit_wmma,
                                                       ReadEmitter& emit_read,
                                                       FenceEmitter& emit_fence,
                                                       RescaleEmitter& emit_rescale)
    {
        static_assert(!Mode::SelectedPolicy::kLeadFirstLoad && !Mode::kStreamOutputRescale);
        if constexpr(ReadPlacement::template PvReadsFirst<Stage, Wmma>())
        {
            ReadPlacement::template VisitPvRowHalf<Stage, Wmma, false>(emit_read);
            ReadPlacement::template VisitPvRowHalf<Stage, Wmma, true>(emit_read);
            emit_fence(number<Stage>{},
                       number<Wmma>{},
                       std::integral_constant<Point, Point::AfterTokens>{});
            emit_wmma(number<Stage>{}, number<Wmma>{});
            // A full barrier after the WMMA as well keeps this read-first row ahead of later code.
            emit_fence(number<Stage>{},
                       number<Wmma>{},
                       std::integral_constant<Point, Point::AfterTokens>{});
        }
        else
            Base::template RunPvRow<Mode, Stage, Wmma>(
                emit_wmma, emit_read, emit_fence, emit_rescale);
    }
};

template <typename Geometry>
using FmhaTdmSchedFp8DefaultScheduleFor =
    FmhaTdmSchedSequentialTileSchedule<FmhaTdmSchedFp8WmmaFirstPlacement<
        std::conditional_t<Geometry::kHeadDimQK == 256,
                           FmhaTdmSchedFp8D256StreamedVReadSchedule<Geometry>,
                           FmhaTdmSchedFp8EvenlyInterleavedReadSchedule<Geometry>>>>;

} // namespace ck_tile
