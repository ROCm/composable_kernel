// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "fmha_test_common.hpp"
#include "gtest/gtest.h"

#include "ck_tile/host/device_memory.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <ostream>
#include <vector>

namespace {
using ck_tile::index_t;

enum class TraceOp
{
    Wmma,
    KRead,
    VRead,
    Fence,
    Rescale,
};

struct PlacementEvent
{
    TraceOp op;
    bool pv;
    int stage;
    int row;
    int half;
    int ordinal;

    bool operator==(const PlacementEvent&) const = default;
};

std::ostream& operator<<(std::ostream& out, const PlacementEvent& event)
{
    return out << "op=" << static_cast<int>(event.op) << " pv=" << event.pv
               << " stage=" << event.stage << " row=" << event.row << " half=" << event.half
               << " ordinal=" << event.ordinal;
}

// Each callback below comes from the executor used by the production pipeline. Row halves are
// observed at the fences; the expected trace never invokes a production read-row visitor.
template <typename Mode>
std::vector<PlacementEvent> ProductionD192Trace()
{
    using Geometry = typename Mode::Geometry;
    using Schedule = typename Mode::SelectedPolicy::Schedule;
    using Executor = ck_tile::FmhaTdmSchedPlacementExecutor<Schedule, Mode>;
    using Kind     = ck_tile::FmhaTdmSchedLoadKind;
    using Point    = ck_tile::FmhaTdmSchedSchedulePoint;
    static_assert(Geometry::kHeadDimQK == 192);
    std::vector<PlacementEvent> trace;

    ck_tile::static_for<0, 2, 1>{}([&](auto phase) {
        constexpr bool pv = decltype(phase)::value != 0;
        ck_tile::static_for<0, Geometry::kN / 32, 1>{}([&](auto stage) {
            constexpr int s = decltype(stage)::value;
            int row         = -1;
            int half        = -1;
            auto rescale    = [&](auto ordinal) {
                trace.push_back({TraceOp::Rescale, pv, s, row, -1, decltype(ordinal)::value});
            };
            auto prelude = [&](auto ordinal) {
                constexpr bool k_read =
                    pv ? s + 1 == Geometry::kPvStages : s + 1 != Geometry::kQkStages;
                trace.push_back({k_read ? TraceOp::KRead : TraceOp::VRead,
                                 pv,
                                 s,
                                 -1,
                                 -1,
                                 decltype(ordinal)::value});
            };
            if constexpr(pv)
            {
                Schedule::template RunPvStagePrelude<Mode, s>(rescale);
                Schedule::template RunPvReadPrelude<Mode, s>(prelude);
            }
            else
                Schedule::template RunQkReadPrelude<Mode, s>(prelude);

            auto wmma = [&](auto emitted_stage, auto ordinal) {
                row  = decltype(ordinal)::value;
                half = -1;
                trace.push_back({TraceOp::Wmma, pv, decltype(emitted_stage)::value, row, -1, row});
            };
            auto read = [&](auto kind, auto ordinal) {
                trace.push_back(
                    {decltype(kind)::value == Kind::KRead ? TraceOp::KRead : TraceOp::VRead,
                     pv,
                     s,
                     row,
                     half,
                     decltype(ordinal)::value});
            };
            auto fence = [&](auto emitted_stage, auto ordinal, auto point) {
                trace.push_back({TraceOp::Fence,
                                 pv,
                                 decltype(emitted_stage)::value,
                                 decltype(ordinal)::value,
                                 -1,
                                 static_cast<int>(decltype(point)::value)});
                half = decltype(point)::value == Point::AfterWmma            ? 0
                       : decltype(point)::value == Point::BetweenTokenHalves ? 1
                                                                             : -1;
            };
            if constexpr(pv)
                Executor::template ExecutePvStage<s>(wmma, read, fence, rescale);
            else
                Executor::template ExecuteQkStage<s>(wmma, read, fence);
        });
    });
    return trace;
}

struct ExpectedReadRow
{
    int first;
    int count;
    int first_half_count;
};

// The BF16/N128 layout is a compact fixture of the historical read positions. Only the active
// K/V positions and midpoint side are retained: no legacy tokens, executor, or current table.
constexpr ExpectedReadRow ExpectedD192Bf16N128ReadRow(bool pv, int stage, int row)
{
    if(stage == 3)
    {
        if(row >= 8)
            return {0, 0, 0};
        return pv ? ExpectedReadRow{3 * row, 3, 2} : ExpectedReadRow{2 * row, 2, 2};
    }
    const int first_row = stage == 2 ? 0 : 1;
    if(row < first_row || row >= first_row + 8)
        return {0, 0, 0};
    if(pv)
        return {2 * (row - first_row), 2, stage == 1 && row == 7 ? 1 : 2};
    return {3 * (row - first_row), 3, 2};
}

static_assert(ExpectedD192Bf16N128ReadRow(false, 0, 1).first == 0);
static_assert(ExpectedD192Bf16N128ReadRow(false, 2, 7).first == 21);
static_assert(ExpectedD192Bf16N128ReadRow(true, 1, 7).first == 12 &&
              ExpectedD192Bf16N128ReadRow(true, 1, 7).first_half_count == 1);
static_assert(ExpectedD192Bf16N128ReadRow(true, 3, 7).first == 21);

template <typename Problem>
std::vector<PlacementEvent> ExpectedD192Trace()
{
    using Point           = ck_tile::FmhaTdmSchedSchedulePoint;
    constexpr int stages  = Problem::BlockFmhaShape::kN0 / 32;
    constexpr int qk_rows = 2 * 2 * (192 / 32);
    constexpr int pv_rows = 2 * (Problem::BlockFmhaShape::kN1 / 16);
    constexpr int k_reads = 192 / 8;
    constexpr int v_reads = Problem::BlockFmhaShape::kN1 / 8;
    constexpr bool historical =
        std::is_same_v<typename Problem::QDataType, ck_tile::bf16_t> && stages == 4;
    std::vector<PlacementEvent> trace;
    for(bool pv : {false, true})
        for(int stage = 0; stage < stages; ++stage)
        {
            if(historical && pv && stage == 0)
                for(int ordinal = 0; ordinal < 2; ++ordinal)
                    trace.push_back({TraceOp::Rescale, pv, stage, -1, -1, ordinal});
            const int rows  = pv ? pv_rows : qk_rows;
            const bool k    = pv ? stage + 1 == stages : stage + 1 != stages;
            const int loads = k ? k_reads : v_reads;
            for(int row = 0; row < rows; ++row)
            {
                trace.push_back({TraceOp::Wmma, pv, stage, row, -1, row});
                if(historical && pv && stage == 0 && row + 2 < pv_rows)
                    trace.push_back({TraceOp::Rescale, pv, stage, row, -1, row + 2});
                trace.push_back(
                    {TraceOp::Fence, pv, stage, row, -1, static_cast<int>(Point::AfterWmma)});
                // Every proportional row owns [floor(row*loads/rows), floor((row+1)*loads/rows)).
                const int first  = row * loads / rows;
                const int count  = (row + 1) * loads / rows - first;
                const auto reads = historical ? ExpectedD192Bf16N128ReadRow(pv, stage, row)
                                              : ExpectedReadRow{first, count, count / 2};
                for(int half = 0; half < 2; ++half)
                {
                    const int begin = half == 0 ? 0 : reads.first_half_count;
                    const int end   = half == 0 ? reads.first_half_count : reads.count;
                    for(int access = begin; access < end; ++access)
                        trace.push_back({k ? TraceOp::KRead : TraceOp::VRead,
                                         pv,
                                         stage,
                                         row,
                                         half,
                                         reads.first + access});
                    trace.push_back({TraceOp::Fence,
                                     pv,
                                     stage,
                                     row,
                                     -1,
                                     static_cast<int>(half == 0 ? Point::BetweenTokenHalves
                                                                : Point::AfterTokens)});
                }
            }
        }
    return trace;
}

bool IsRead(const PlacementEvent& event)
{
    return event.op == TraceOp::KRead || event.op == TraceOp::VRead;
}

bool PermuteReadsWithinOneHalf(std::vector<PlacementEvent>& trace)
{
    for(std::size_t i = 1; i < trace.size(); ++i)
        if(IsRead(trace[i - 1]) && IsRead(trace[i]) && trace[i - 1].pv == trace[i].pv &&
           trace[i - 1].stage == trace[i].stage && trace[i - 1].row == trace[i].row &&
           trace[i - 1].half == trace[i].half)
        {
            std::swap(trace[i - 1].ordinal, trace[i].ordinal);
            return true;
        }
    // Proportional placements have at most one read per row half. Exchange ordinals between
    // two rows of the same stage/half instead, preserving every event's phase and row position.
    for(std::size_t i = 0; i < trace.size(); ++i)
        for(std::size_t j = i + 1; j < trace.size(); ++j)
            if(IsRead(trace[i]) && trace[i].op == trace[j].op && trace[i].pv == trace[j].pv &&
               trace[i].stage == trace[j].stage && trace[i].half == trace[j].half &&
               trace[i].ordinal != trace[j].ordinal)
            {
                std::swap(trace[i].ordinal, trace[j].ordinal);
                return true;
            }
    return false;
}

// The host compiler defaults to wave64. An explicitly named gfx125 WMMA layout therefore
// provides host descriptor coverage, while only the separate device probe tests dispatch.
template <typename Config>
struct Gfx125TraitScoreLayout
{
    using Problem     = typename Config::Problem;
    using Geometry    = typename Config::Mode::Geometry;
    using WarpGemm    = ck_tile::WarpGemmImpl<ck_tile::WarpGemmAttributeWmma<
           ck_tile::WarpGemmAttributeWmmaImpl<typename Geometry::QkWmma>,
           true,
           ck_tile::WGAttrNumAccessEnum::Default,
           ck_tile::WGAttrNumAccessEnum::Default>>;
    using GemmProblem = ck_tile::BlockGemmProblem<
        typename Problem::QDataType,
        typename Problem::KDataType,
        float,
        4 * 32,
        ck_tile::TileGemmShape<ck_tile::sequence<128, Problem::BlockFmhaShape::kN0, 32>,
                               typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                               typename Problem::BlockFmhaShape::Gemm0WarpTile>>;
    using GemmPolicy = ck_tile::BlockGemmARegBRegCRegV2CustomPolicy<
        typename Problem::QDataType,
        typename Problem::KDataType,
        float,
        typename Problem::BlockFmhaShape::Gemm0BlockWarps,
        WarpGemm,
        ck_tile::GemmLoopOrder::MNK>;
    using BlockGemm = ck_tile::BlockGemmARegBRegCRegV2<GemmProblem, GemmPolicy>;
    using ScoreTile = decltype(BlockGemm::MakeCBlockTile());
};

template <typename Mapping, typename ScoreTile>
CK_TILE_HOST_DEVICE constexpr bool CheckScoreDescriptor()
{
    constexpr auto descriptor = typename ScoreTile::ThreadTensorDesc{};
    constexpr auto lengths    = descriptor.get_lengths();
    static_assert(lengths[ck_tile::number<0>{}] == Mapping::kNumMIter);
    static_assert(lengths[ck_tile::number<1>{}] == Mapping::kNumNIter);
    static_assert(lengths[ck_tile::number<2>{}] == 1);
    static_assert(lengths[ck_tile::number<3>{}] == Mapping::kElementsPerWmma);
    static_assert(ScoreTile::get_thread_buffer_size() == Mapping::kThreadBufferSize);
    bool valid = true;
    // Start with CK's physical per-thread descriptor coordinates and compare every offset
    // against the production mapping, independently of its Decode/ValidateBijection helpers.
    for(int m = 0; m < 2; ++m)
        for(int n = 0; n < Mapping::kNumNIter; ++n)
            for(int element = 0; element < 8; ++element)
            {
                const int msb  = 2 * m + n % 2;
                const int pair = 4 * (n / 2) + element / 2;
                valid &= descriptor.calculate_offset(ck_tile::make_tuple(m, n, 0, element)) ==
                         Mapping::GetThreadBufferOffset(msb, pair, element % 2);
            }
    return valid;
}

CK_TILE_HOST_DEVICE constexpr std::uint32_t ScoreBits(int offset, int lane, bool replacement)
{
    // Preserve zeros, infinities, subnormals, and quiet-NaN payloads bit for bit. Other slots
    // carry distinct finite values so swaps are visible even when a whole tile is roundtripped.
    switch(offset)
    {
    case 0: return replacement ? 0x80000000u : 0x00000000u;
    case 1: return replacement ? 0x00000000u : 0x80000000u;
    case 2: return replacement ? 0xff800000u : 0x7f800000u;
    case 3: return replacement ? 0x7f800000u : 0xff800000u;
    case 4: return replacement ? 0x807fffffu : 0x00000001u;
    case 5: return replacement ? 0x00000001u : 0x807fffffu;
    case 6:
        return (replacement ? 0xffc00000u : 0x7fc00000u) |
               static_cast<std::uint32_t>(1 + lane * 256 + offset);
    default:
        return (replacement ? 0xc0000000u : 0x3f000000u) |
               static_cast<std::uint32_t>(1 + lane * 256 + offset);
    }
}

template <typename Mapping, typename ScoreTile>
CK_TILE_HOST_DEVICE bool CheckScorePairRoundtrip(int lane)
{
    ScoreTile score{};
    constexpr auto descriptor = typename ScoreTile::ThreadTensorDesc{};
    for(int offset = 0; offset < ScoreTile::get_thread_buffer_size(); ++offset)
        score.get_thread_buffer()[offset] =
            ck_tile::bit_cast<float>(ScoreBits(offset, lane, false));
    bool valid = true;
    ck_tile::static_ford<ck_tile::sequence<4, Mapping::kPairsPerMsb>>{}([&](auto indices) {
        constexpr int msb  = indices[ck_tile::number<0>{}];
        constexpr int pair = indices[ck_tile::number<1>{}];
        // CK descriptor, rather than Mapping::GetThreadBufferOffset, selects the expected pair.
        constexpr int offset = descriptor.calculate_offset(
            ck_tile::make_tuple(msb / 2, 2 * (pair / 4) + msb % 2, 0, 2 * (pair % 4)));
        const auto loaded = Mapping::template LoadPair<msb, pair>(score);
        valid &= ck_tile::bit_cast<std::uint32_t>(loaded[0]) == ScoreBits(offset, lane, false);
        valid &= ck_tile::bit_cast<std::uint32_t>(loaded[1]) == ScoreBits(offset + 1, lane, false);
        const typename Mapping::Pair replacement{
            ck_tile::bit_cast<float>(ScoreBits(offset, lane, true)),
            ck_tile::bit_cast<float>(ScoreBits(offset + 1, lane, true))};
        Mapping::template StorePair<msb, pair>(score, replacement);
        const auto roundtrip = Mapping::template LoadPair<msb, pair>(score);
        valid &= ck_tile::bit_cast<std::uint32_t>(roundtrip[0]) == ScoreBits(offset, lane, true);
        valid &=
            ck_tile::bit_cast<std::uint32_t>(roundtrip[1]) == ScoreBits(offset + 1, lane, true);
    });
    for(int offset = 0; offset < ScoreTile::get_thread_buffer_size(); ++offset)
        valid &= ck_tile::bit_cast<std::uint32_t>(score.get_thread_buffer()[offset]) ==
                 ScoreBits(offset, lane, true);
    return valid;
}

template <typename Config>
__global__ void ProductionScoreMappingProbe(std::uint32_t* result)
{
    auto* lane_result = result + 3 * threadIdx.x;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
    using Problem   = typename Config::Problem;
    using Policy    = typename Config::Policy;
    using Mapping   = typename Policy::ScoreMapping;
    using BlockGemm = ck_tile::remove_cvref_t<decltype(Policy::template GetQKBlockGemm<Problem>())>;
    using ScoreTile = decltype(BlockGemm::MakeCBlockTile());
    static_assert(ck_tile::get_warp_size() == 32);
    static_assert(Problem::kBlockSize == 128);
    static_assert(std::is_same_v<typename BlockGemm::WarpGemm,
                                 typename Gfx125TraitScoreLayout<Config>::WarpGemm>);
    static_assert(std::is_same_v<ScoreTile, typename Gfx125TraitScoreLayout<Config>::ScoreTile>);
    static_assert(CheckScoreDescriptor<Mapping, ScoreTile>());
    lane_result[0] = ck_tile::get_warp_size();
    lane_result[1] = CheckScoreDescriptor<Mapping, ScoreTile>() ? 1 : 0;
    lane_result[2] = CheckScorePairRoundtrip<Mapping, ScoreTile>(threadIdx.x) ? 1 : 0;
#else
    // A non-gfx125 device pass must never look like a successful physical wave32 check.
    lane_result[0] = 0;
    lane_result[1] = 0;
    lane_result[2] = 0;
#endif
}

template <typename Policy>
constexpr bool CheckTileOrder()
{
    int phase    = 0;
    bool valid   = true;
    auto begin   = [&] { valid &= phase++ == 0; };
    auto qk      = [&] { valid &= phase++ == 1; };
    auto score   = [&] { valid &= phase++ == 2; };
    auto softmax = [&] {
        valid &= phase++ == 3;
        return index_t{17};
    };
    auto next_k  = [&] { valid &= phase++ == 4; };
    auto pv      = [&](index_t p_tile) { valid &= phase++ == 5 && p_tile == 17; };
    auto advance = [&] { valid &= phase++ == 6; };
    auto ops     = ck_tile::MakeFmhaTdmSchedTileOps(begin, qk, score, softmax, next_k, pv, advance);
    Policy::Schedule::Run(ops);
    return valid && phase == 7;
}

template <typename Mode>
constexpr bool CheckSplitSoftmax()
{
    using Schedule = typename Mode::SelectedPolicy::Schedule;
    int phase      = 0;
    int part01     = 0;
    int part2      = 0;
    int wait_v     = 0;
    int rescale[Mode::SelectedPolicy::OutputFragments::kNumFragments]{};
    bool valid = true;
    Schedule::template RunSplitSoftmax<Mode>(
        [&] {
            valid &= phase++ == 0;
            ++part01;
        },
        [&] {
            valid &= part01 == 1;
            ++wait_v;
            ++phase;
        },
        [&](auto ordinal) {
            valid &= part01 == 1 && part2 == static_cast<int>(Mode::kPostSoftmaxFragmentRescale);
            ++rescale[decltype(ordinal)::value];
            ++phase;
        },
        [&](auto) {
            valid &= part01 == 1;
            ++part2;
            ++phase;
        });
    valid &= part01 == 1 && part2 == 1 && wait_v == static_cast<int>(Mode::kDeferTensorReady);
    for(int count : rescale)
        valid &= count ==
                 static_cast<int>(Mode::kEarlyOutputRescale || Mode::kPostSoftmaxFragmentRescale);
    return valid;
}

template <typename Mode>
constexpr bool CheckRows()
{
    using Schedule = typename Mode::SelectedPolicy::Schedule;
    using Geometry = typename Mode::Geometry;
    using Executor = ck_tile::FmhaTdmSchedPlacementExecutor<Schedule, Mode>;
    using Kind     = ck_tile::FmhaTdmSchedLoadKind;
    int qk_wmma[Geometry::kQkStages][Geometry::kQkWmmasPerStage]{};
    int pv_wmma[Geometry::kPvStages][Geometry::kPvWmmasPerStage]{};
    int qk_k_read[Geometry::kQkStages][Geometry::kKSuLoadCount]{};
    int qk_v_read[Geometry::kVStageLoadCount]{};
    int pv_v_read[Geometry::kPvStages][Geometry::kVStageLoadCount]{};
    int pv_k_read[Geometry::kKSuLoadCount]{};
    int qk_fence[Geometry::kQkStages][Geometry::kQkWmmasPerStage]{};
    int pv_fence[Geometry::kPvStages][Geometry::kPvWmmasPerStage]{};
    int rescale[Mode::SelectedPolicy::OutputFragments::kNumFragments]{};
    bool valid = true;

    ck_tile::static_for<0, Geometry::kQkStages, 1>{}([&](auto stage) {
        constexpr int s = decltype(stage)::value;
        Schedule::template RunQkReadPrelude<Mode, s>([&](auto ordinal) {
            if constexpr(s < Geometry::kQkStages - 1)
                ++qk_k_read[s][decltype(ordinal)::value];
            else
                ++qk_v_read[decltype(ordinal)::value];
        });
        auto wmma = [&](auto, auto ordinal) { ++qk_wmma[s][decltype(ordinal)::value]; };
        auto read = [&](auto kind, auto ordinal) {
            if constexpr(decltype(kind)::value == Kind::KRead)
                ++qk_k_read[s][decltype(ordinal)::value];
            else
                ++qk_v_read[decltype(ordinal)::value];
        };
        auto fence = [&](auto, auto ordinal, auto) { ++qk_fence[s][decltype(ordinal)::value]; };
        Executor::template ExecuteQkStage<s>(wmma, read, fence);
    });

    ck_tile::static_for<0, Geometry::kPvStages, 1>{}([&](auto stage) {
        constexpr int s       = decltype(stage)::value;
        auto rescale_fragment = [&](auto ordinal) { ++rescale[decltype(ordinal)::value]; };
        Schedule::template RunPvStagePrelude<Mode, s>(rescale_fragment);
        Schedule::template RunPvReadPrelude<Mode, s>([&](auto ordinal) {
            if constexpr(s < Geometry::kPvStages - 1)
                ++pv_v_read[s][decltype(ordinal)::value];
            else
                ++pv_k_read[decltype(ordinal)::value];
        });
        auto wmma = [&](auto, auto ordinal) { ++pv_wmma[s][decltype(ordinal)::value]; };
        auto read = [&](auto kind, auto ordinal) {
            if constexpr(decltype(kind)::value == Kind::VRead)
                ++pv_v_read[s][decltype(ordinal)::value];
            else
                ++pv_k_read[decltype(ordinal)::value];
        };
        auto fence = [&](auto, auto ordinal, auto) { ++pv_fence[s][decltype(ordinal)::value]; };
        Executor::template ExecutePvStage<s>(wmma, read, fence, rescale_fragment);
    });

    for(int s = 0; s < Geometry::kQkStages; ++s)
    {
        for(int w = 0; w < Geometry::kQkWmmasPerStage; ++w)
            valid &= qk_wmma[s][w] == 1 && qk_fence[s][w] == 3;
        if(s < Geometry::kQkStages - 1)
            for(int r = 0; r < Geometry::kKSuLoadCount; ++r)
                valid &= qk_k_read[s][r] == 1;
    }
    for(int s = 0; s < Geometry::kPvStages; ++s)
    {
        for(int w = 0; w < Geometry::kPvWmmasPerStage; ++w)
            valid &= pv_wmma[s][w] == 1 && pv_fence[s][w] == 3;
        if(s < Geometry::kPvStages - 1)
            for(int r = 0; r < Geometry::kVStageLoadCount; ++r)
                valid &= pv_v_read[s][r] == 1;
    }
    for(int r = 0; r < Geometry::kVStageLoadCount; ++r)
        valid &= qk_v_read[r] == 1;
    for(int r = 0; r < Geometry::kKSuLoadCount; ++r)
        valid &= pv_k_read[r] == 1;
    for(int count : rescale)
        valid &= count == static_cast<int>(Mode::kStreamOutputRescale);
    return valid;
}

template <typename Mode>
constexpr bool CheckPlacementOrder()
{
    using Schedule            = typename Mode::SelectedPolicy::Schedule;
    using Geometry            = typename Mode::Geometry;
    using Executor            = ck_tile::FmhaTdmSchedPlacementExecutor<Schedule, Mode>;
    using Point               = ck_tile::FmhaTdmSchedSchedulePoint;
    bool valid                = true;
    constexpr bool read_first = Geometry::kHeadDimQK == 64;

    ck_tile::static_for<0, Geometry::kQkStages, 1>{}([&](auto stage) {
        constexpr int s = decltype(stage)::value;
        int row         = 0;
        int phase       = 0;
        int prelude     = 0;
        Schedule::template RunQkReadPrelude<Mode, s>([&](auto ordinal) {
            valid &= row == 0 && phase == 0 && decltype(ordinal)::value == 0;
            ++prelude;
        });
        valid &= prelude == static_cast<int>(Mode::SelectedPolicy::kLeadFirstLoad &&
                                             (s == 0 || s == Geometry::kQkStages - 1));
        auto wmma = [&](auto emitted_stage, auto ordinal) {
            valid &= decltype(emitted_stage)::value == s && decltype(ordinal)::value == row &&
                     phase == (read_first ? 3 : 0);
            phase = 1;
        };
        auto read = [&](auto, auto) {
            valid &= phase == (read_first ? 0 : 2) || phase == (read_first ? 2 : 3);
        };
        auto fence = [&](auto emitted_stage, auto ordinal, auto point) {
            valid &= decltype(emitted_stage)::value == s && decltype(ordinal)::value == row;
            if constexpr(decltype(point)::value == Point::AfterWmma)
            {
                valid &= phase == 1;
                if constexpr(read_first)
                {
                    phase = 0;
                    ++row;
                }
                else
                    phase = 2;
            }
            else if constexpr(decltype(point)::value == Point::BetweenTokenHalves)
            {
                valid &= phase == (read_first ? 0 : 2);
                phase = read_first ? 2 : 3;
            }
            else
            {
                valid &= phase == (read_first ? 2 : 3);
                if constexpr(read_first)
                    phase = 3;
                else
                {
                    phase = 0;
                    ++row;
                }
            }
        };
        Executor::template ExecuteQkStage<s>(wmma, read, fence);
        valid &= row == Geometry::kQkWmmasPerStage && phase == 0;
    });

    ck_tile::static_for<0, Geometry::kPvStages, 1>{}([&](auto stage) {
        constexpr int s = decltype(stage)::value;
        int row         = 0;
        int phase       = 0;
        int prelude     = 0;
        int rescaled    = 0;
        Schedule::template RunPvStagePrelude<Mode, s>([&](auto ordinal) {
            valid &= row == 0 && phase == 0 && decltype(ordinal)::value == rescaled;
            ++rescaled;
        });
        Schedule::template RunPvReadPrelude<Mode, s>([&](auto ordinal) {
            valid &= row == 0 && phase == 0 && decltype(ordinal)::value == 0;
            ++prelude;
        });
        valid &= prelude == static_cast<int>(Mode::SelectedPolicy::kLeadFirstLoad &&
                                             (s == 0 || s == Geometry::kPvStages - 1));
        valid &= rescaled == static_cast<int>(Mode::kStreamOutputRescale && s == 0) * 2;
        auto wmma = [&](auto emitted_stage, auto ordinal) {
            valid &= decltype(emitted_stage)::value == s && decltype(ordinal)::value == row &&
                     phase == (read_first ? 3 : 0);
            phase = 1;
        };
        auto read = [&](auto, auto) {
            valid &= phase == (read_first ? 0 : 2) || phase == (read_first ? 2 : 3);
        };
        auto fence = [&](auto emitted_stage, auto ordinal, auto point) {
            valid &= decltype(emitted_stage)::value == s && decltype(ordinal)::value == row;
            if constexpr(decltype(point)::value == Point::AfterWmma)
            {
                valid &= phase == 1;
                if constexpr(read_first)
                {
                    phase = 0;
                    ++row;
                }
                else
                    phase = 2;
            }
            else if constexpr(decltype(point)::value == Point::BetweenTokenHalves)
            {
                valid &= phase == (read_first ? 0 : 2);
                phase = read_first ? 2 : 3;
            }
            else
            {
                valid &= phase == (read_first ? 2 : 3);
                if constexpr(read_first)
                    phase = 3;
                else
                {
                    phase = 0;
                    ++row;
                }
            }
        };
        auto rescale = [&](auto ordinal) {
            valid &= Mode::kStreamOutputRescale && s == 0 && phase == 1 &&
                     decltype(ordinal)::value == row + 2;
            ++rescaled;
        };
        Executor::template ExecutePvStage<s>(wmma, read, fence, rescale);
        valid &= row == Geometry::kPvWmmasPerStage && phase == 0;
        valid &= rescaled == static_cast<int>(Mode::kStreamOutputRescale && s == 0) *
                                 Geometry::kPvWmmasPerStage;
    });
    return valid;
}

template <typename Problem_>
struct ScheduleConfig
{
    using Problem = Problem_;
    using Policy  = ck_tile::FmhaTdmSchedPolicyFor<Problem>;
    using Mode    = ck_tile::FmhaTdmSchedMode<Problem, Policy>;
};
using Configs = ::testing::Types<
    ScheduleConfig<
        typename ck_tile::test::Model<64, false, false, true, ck_tile::bf16_t, 64, 64>::Problem>,
    ScheduleConfig<
        typename ck_tile::test::Model<64, false, true, true, ck_tile::half_t, 64, 64>::Problem>,
    ScheduleConfig<typename ck_tile::test::Model<128>::Problem>,
    ScheduleConfig<typename ck_tile::test::Model<128, true, true, true, ck_tile::half_t>::Problem>,
    ScheduleConfig<
        typename ck_tile::test::Model<192, false, false, true, ck_tile::bf16_t, 64>::Problem>,
    ScheduleConfig<
        typename ck_tile::test::Model<192, true, true, true, ck_tile::half_t, 64>::Problem>,
    ScheduleConfig<typename ck_tile::test::Model<192, false, false, true>::Problem>,
    ScheduleConfig<typename ck_tile::test::Model<192, true, true, true, ck_tile::half_t>::Problem>>;

template <typename T>
class QrTdmSchedSchedule : public ::testing::Test
{
};
TYPED_TEST_SUITE(QrTdmSchedSchedule, Configs);

TYPED_TEST(QrTdmSchedSchedule, TilePhaseOrder)
{
    EXPECT_TRUE(CheckTileOrder<typename TypeParam::Policy>());
}
TYPED_TEST(QrTdmSchedSchedule, SoftmaxAndRescaleOwnership)
{
    EXPECT_TRUE(CheckSplitSoftmax<typename TypeParam::Mode>());
}
TYPED_TEST(QrTdmSchedSchedule, EveryReadWmmaAndFenceExecutesOnce)
{
    EXPECT_TRUE(CheckRows<typename TypeParam::Mode>());
}
TYPED_TEST(QrTdmSchedSchedule, RowPlacementAndFenceOrder)
{
    EXPECT_TRUE(CheckPlacementOrder<typename TypeParam::Mode>());
}

using D192Configs = ::testing::Types<
    ScheduleConfig<
        typename ck_tile::test::Model<192, false, false, true, ck_tile::bf16_t, 64>::Problem>,
    ScheduleConfig<
        typename ck_tile::test::Model<192, true, true, true, ck_tile::half_t, 64>::Problem>,
    ScheduleConfig<typename ck_tile::test::Model<192, false, false, true>::Problem>,
    ScheduleConfig<typename ck_tile::test::Model<192, true, true, true, ck_tile::half_t>::Problem>>;

template <typename T>
class QrTdmSchedD192Placement : public ::testing::Test
{
};
TYPED_TEST_SUITE(QrTdmSchedD192Placement, D192Configs);

TYPED_TEST(QrTdmSchedD192Placement, ProductionReadRowHalfAndAccessTrace)
{
    const auto actual   = ProductionD192Trace<typename TypeParam::Mode>();
    const auto expected = ExpectedD192Trace<typename TypeParam::Problem>();
    ASSERT_EQ(actual.size(), expected.size());
    for(std::size_t i = 0; i < expected.size(); ++i)
        EXPECT_EQ(actual[i], expected[i]) << "callback index=" << i;
}

TYPED_TEST(QrTdmSchedD192Placement, ReadTraceRejectsSameHalfPermutation)
{
    const auto actual   = ProductionD192Trace<typename TypeParam::Mode>();
    const auto expected = ExpectedD192Trace<typename TypeParam::Problem>();
    ASSERT_EQ(actual, expected);
    auto permuted = actual;
    ASSERT_TRUE(PermuteReadsWithinOneHalf(permuted));
    EXPECT_NE(permuted, expected);

    // The mutation preserves all callback counts and every non-read position; only exact
    // row/access identity within the same stage/half distinguishes it from production.
    ASSERT_EQ(permuted.size(), actual.size());
    int changed = 0;
    for(std::size_t i = 0; i < actual.size(); ++i)
    {
        EXPECT_EQ(permuted[i].op, actual[i].op);
        EXPECT_EQ(permuted[i].pv, actual[i].pv);
        EXPECT_EQ(permuted[i].stage, actual[i].stage);
        EXPECT_EQ(permuted[i].row, actual[i].row);
        EXPECT_EQ(permuted[i].half, actual[i].half);
        if(!IsRead(actual[i]))
            EXPECT_EQ(permuted[i], actual[i]);
        changed += permuted[i] == actual[i] ? 0 : 1;
    }
    EXPECT_EQ(changed, 2);
    auto sorted_actual   = actual;
    auto sorted_permuted = permuted;
    // The original count-only tests index reads by stage/kind/access, not row or half.
    for(auto* events : {&sorted_actual, &sorted_permuted})
        for(auto& event : *events)
            if(IsRead(event))
                event.row = event.half = 0;
    auto less = [](const PlacementEvent& a, const PlacementEvent& b) {
        return std::array<int, 6>{static_cast<int>(a.op), a.pv, a.stage, a.row, a.half, a.ordinal} <
               std::array<int, 6>{static_cast<int>(b.op), b.pv, b.stage, b.row, b.half, b.ordinal};
    };
    std::sort(sorted_actual.begin(), sorted_actual.end(), less);
    std::sort(sorted_permuted.begin(), sorted_permuted.end(), less);
    EXPECT_EQ(sorted_actual, sorted_permuted);
}

TYPED_TEST(QrTdmSchedSchedule, HostGfx125TraitScoreDescriptorAndBitwisePairs)
{
    using Mapping   = typename TypeParam::Policy::ScoreMapping;
    using ScoreTile = typename Gfx125TraitScoreLayout<TypeParam>::ScoreTile;
    // This check deliberately fixes the WMMA traits and block to physical wave32. It is
    // host-side descriptor/accessor coverage, not evidence that host wave64 dispatch matches.
    static_assert(CheckScoreDescriptor<Mapping, ScoreTile>());
    EXPECT_TRUE((CheckScoreDescriptor<Mapping, ScoreTile>()));
    for(int lane = 0; lane < 128; ++lane)
        EXPECT_TRUE((CheckScorePairRoundtrip<Mapping, ScoreTile>(lane))) << "lane=" << lane;
}

template <typename T>
class QrTdmSchedScoreDevice : public ck_tile::test::Gfx125FmhaTest
{
};
TYPED_TEST_SUITE(QrTdmSchedScoreDevice, Configs);

TYPED_TEST(QrTdmSchedScoreDevice, ProductionWave32DescriptorAndBitwisePairs)
{
    constexpr int lanes = 4 * 32;
    std::vector<std::uint32_t> result(3 * lanes, 0xffffffffu);
    ck_tile::DeviceMem device_result(result.size() * sizeof(std::uint32_t));
    device_result.ToDevice(result.data());
    ProductionScoreMappingProbe<TypeParam>
        <<<dim3(1), dim3(lanes)>>>(static_cast<std::uint32_t*>(device_result.GetDeviceBuffer()));
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    device_result.FromDevice(result.data());
    for(int lane = 0; lane < lanes; ++lane)
    {
        EXPECT_EQ(result[3 * lane], 32u) << "physical wave-size stamp, lane=" << lane;
        EXPECT_EQ(result[3 * lane + 1], 1u) << "production CK descriptor, lane=" << lane;
        EXPECT_EQ(result[3 * lane + 2], 1u)
            << "bitwise LoadPair/StorePair roundtrip, lane=" << lane;
    }
}

} // namespace
