// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_16bit_traits.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_8bit_traits.hpp"

namespace ck_tile {

namespace detail {

template <typename Geometry, typename Tails, index_t... Stages>
CK_TILE_HOST_DEVICE constexpr bool fmha_tdm_sched_valid_qk_tails(sequence<Stages...>)
{
    return ((Tails::at(number<Stages>{}) == 0 ||
             Tails::at(number<Stages>{}) == (Stages + 1 == Geometry::kQkStages
                                                 ? Geometry::kVStageLoadCount
                                                 : Geometry::kKSuLoadCount)) &&
            ...);
}

template <typename Geometry, typename Tails, index_t... Stages>
CK_TILE_HOST_DEVICE constexpr bool fmha_tdm_sched_valid_pv_tails(sequence<Stages...>)
{
    return ((Tails::at(number<Stages>{}) == 0 ||
             Tails::at(number<Stages>{}) == (Stages + 1 == Geometry::kPvStages
                                                 ? Geometry::kKSuLoadCount
                                                 : Geometry::kVStageLoadCount)) &&
            ...);
}

} // namespace detail

// P repack (MakePForGemm1): gemm_0 produces S with the transposed C layout
// (WarpGemmDispatcher<..., TransposeC = true>), gemm_1 consumes P as its A operand. P can be
// re-packed without crossing lanes iff each gemm_1 A tile is exactly kAK0PerLane adjacent
// gemm_0 C tiles in the same lanes and per-lane order:
//   C^T: row = lane % kCNLane, col = (lane / kCNLane) * kCM1PerLane + [0, kCM1PerLane)
//   A:   row = lane % kAMLane, k   = k0 * kABKLane * kAK1PerLane
//                                    + (lane / kAMLane) * kAK1PerLane + [0, kAK1PerLane)
template <typename QkWmma, typename PvWmma>
struct FmhaTdmSchedPRepackLayout
{
    // Same row -> lane mapping, one 16-row block per instruction.
    static constexpr bool kRowLanes = QkWmma::kCNBlock == 1 && PvWmma::kAMBlock == 1 &&
                                      QkWmma::kCNLane == PvWmma::kAMLane &&
                                      QkWmma::kM == PvWmma::kM;
    // Same split of the reduction columns across lane groups, same contiguous run per lane.
    static constexpr bool kColumnLanes =
        QkWmma::kCMLane == PvWmma::kABKLane && QkWmma::kCM1PerLane == PvWmma::kAK1PerLane;
    // One C^T tile holds exactly one run per lane, i.e. one A k0 step ...
    static constexpr bool kOneRunPerTile = QkWmma::kCMBlock == 1 && QkWmma::kCM0PerLane == 1 &&
                                           QkWmma::kN == QkWmma::kCMLane * QkWmma::kCM1PerLane;
    // ... and one A tile is kAK0PerLane adjacent C^T tiles.
    static constexpr bool kTilesPerA = PvWmma::kK == PvWmma::kAK0PerLane * QkWmma::kN;
    // Lane = group * rows + row on both sides; A stores k0-major, k1-minor per lane.
    static constexpr bool kLaneEncoding =
        std::is_same_v<typename QkWmma::kCTPs2RHssMajor, sequence<2, 1>> &&
        std::is_same_v<typename QkWmma::kCTPs2RHssMinor, sequence<2, 1>> &&
        std::is_same_v<typename PvWmma::kABPs2RHssMajor, sequence<2, 1>> &&
        std::is_same_v<typename PvWmma::kABPs2RHssMinor, sequence<1, 1>> &&
        std::is_same_v<typename PvWmma::kABYs2RHsMajor, sequence<1, 2, 2>> &&
        std::is_same_v<typename PvWmma::kABYs2RHsMinor, sequence<0, 0, 2>>;

    static constexpr bool kInThread =
        kRowLanes && kColumnLanes && kOneRunPerTile && kTilesPerA && kLaneEncoding;
};

// Per-wave work decomposition of one qr_tdm_sched KV tile, derived from the block tile and the
// gfx125 WMMA instructions that the block GEMMs dispatch to.
//
// The instruction traits are named explicitly (WmmaTraits<gfx125_t, ...>) rather than through
// WarpGemmDispatcher: the dispatcher selects its gfx125 WMMA entries under __gfx125__, so the
// host pass would otherwise see MFMA shapes and derive different counts than the device pass.
// The pipeline checks on the device pass that its dispatched warp GEMMs use these same traits.
template <typename Q,
          typename K,
          typename V,
          typename P,
          index_t M,
          index_t N,
          index_t Dv,
          index_t HeadDimQK,
          index_t PvStageK,
          typename BlockWarps,
          typename QkWarpTile,
          typename PvWarpTile,
          index_t QkSuColumns>
struct FmhaTdmSchedGeometry
{
    using QDataType = Q;
    using KDataType = K;
    using VDataType = V;
    using PDataType = P;

    // Both GEMMs accumulate in fp32. gemm_0 computes S = Q * K^T, gemm_1 computes O += P * V.
    using QkWmma = WmmaTraits<gfx125_t,
                              Q,
                              K,
                              float,
                              QkWarpTile::at(number<0>{}),
                              QkWarpTile::at(number<1>{}),
                              QkWarpTile::at(number<2>{})>;
    using PvWmma = WmmaTraits<gfx125_t,
                              P,
                              V,
                              float,
                              PvWarpTile::at(number<0>{}),
                              PvWarpTile::at(number<1>{}),
                              PvWarpTile::at(number<2>{})>;

    // Block tile.
    static constexpr index_t kM         = M;
    static constexpr index_t kN         = N;
    static constexpr index_t kDv        = Dv;
    static constexpr index_t kHeadDimQK = HeadDimQK;

    // Waves split only the query rows; every wave owns all kN columns of its rows.
    static_assert(BlockWarps::at(number<1>{}) == 1 && BlockWarps::at(number<2>{}) == 1,
                  "qr_tdm_sched splits the query rows across waves only");
    static constexpr index_t kWaves            = BlockWarps::at(number<0>{});
    static constexpr index_t kWaveSize         = QkWmma::kAMLane * QkWmma::kABKLane;
    static constexpr index_t kQueryRowsPerWave = kM / kWaves;

    // WMMA instruction shapes.
    static constexpr index_t kQkWmmaM = QkWmma::kM;
    static constexpr index_t kQkWmmaN = QkWmma::kN;
    static constexpr index_t kQkWmmaK = QkWmma::kK;
    static constexpr index_t kPvWmmaM = PvWmma::kM;
    static constexpr index_t kPvWmmaN = PvWmma::kN;
    static constexpr index_t kPvWmmaK = PvWmma::kK;

    // Stage split. QK walks the KV tile in kQkSuColumns-wide column groups; PV walks it in
    // kPvStageK-deep reduction slices (the block tile's kK1).
    static constexpr index_t kQkSuColumns = QkSuColumns;
    static constexpr index_t kQkStages    = kN / kQkSuColumns;
    static constexpr index_t kPvStageK    = PvStageK;
    static constexpr index_t kPvStages    = kN / kPvStageK;

    // Per-wave WMMA iterations of one stage.
    static constexpr index_t kQkMIter      = kQueryRowsPerWave / kQkWmmaM;
    static constexpr index_t kQkSuNIter    = kQkSuColumns / kQkWmmaN;
    static constexpr index_t kQkKIter      = kHeadDimQK / kQkWmmaK;
    static constexpr index_t kPvMIter      = kQueryRowsPerWave / kPvWmmaM;
    static constexpr index_t kPvNIter      = kDv / kPvWmmaN;
    static constexpr index_t kPvStageKIter = kPvStageK / kPvWmmaK;

    static constexpr index_t kQkWmmasPerStage      = kQkMIter * kQkSuNIter * kQkKIter;
    static constexpr index_t kPvWmmasPerStage      = kPvMIter * kPvNIter * kPvStageKIter;
    static constexpr index_t kQkWmmasPerWaveKvTile = kQkStages * kQkWmmasPerStage;
    static constexpr index_t kPvWmmasPerWaveKvTile = kPvStages * kPvWmmasPerStage;

    // LDS reads per stage and lane: each wave reads the whole B operand of its stage (K for QK,
    // V for PV) with one 16-byte access per event (ds_load_b128 / ds_load_tr16_b128).
    static constexpr index_t kLdsReadBytes    = 16;
    static constexpr index_t kQkBPerLane      = QkWmma::kBK0PerLane * QkWmma::kBK1PerLane;
    static constexpr index_t kPvBPerLane      = PvWmma::kBK0PerLane * PvWmma::kBK1PerLane;
    static constexpr index_t kKSuLaneBytes    = kQkSuNIter * kQkKIter * kQkBPerLane * sizeof(K);
    static constexpr index_t kVStageLaneBytes = kPvNIter * kPvStageKIter * kPvBPerLane * sizeof(V);
    static constexpr index_t kKSuLoadCount    = kKSuLaneBytes / kLdsReadBytes;
    static constexpr index_t kVStageLoadCount = kVStageLaneBytes / kLdsReadBytes;

    static_assert(kWaves > 0 && kM % (kWaves * kQkWmmaM) == 0 && kM % (kWaves * kPvWmmaM) == 0,
                  "query rows must split evenly into waves and WMMA row tiles");
    static_assert(kN % kQkSuColumns == 0 && kQkSuColumns % kQkWmmaN == 0,
                  "QK column groups must tile the KV tile with whole WMMA column tiles");
    static_assert(kHeadDimQK % kQkWmmaK == 0, "QK head dim must be whole WMMA K steps");
    static_assert(kN % kPvStageK == 0 && kPvStageK % kPvWmmaK == 0,
                  "PV stages must tile the KV tile with whole WMMA K steps");
    static_assert(kDv % kPvWmmaN == 0, "V head dim must be whole WMMA column tiles");
    static_assert(kKSuLaneBytes % kLdsReadBytes == 0 && kVStageLaneBytes % kLdsReadBytes == 0,
                  "per-lane stage operands must be whole 16-byte LDS reads");

    using PRepackLayout = FmhaTdmSchedPRepackLayout<QkWmma, PvWmma>;
};

// Default QK column group: two WMMA column tiles per stage.
template <typename QkWarpTile>
inline constexpr index_t kFmhaTdmSchedDefaultQkSuColumns = 2 * QkWarpTile::at(number<1>{});

template <typename Problem,
          index_t QkSuColumns = kFmhaTdmSchedDefaultQkSuColumns<
              typename remove_cvref_t<typename Problem::BlockFmhaShape>::Gemm0WarpTile>>
using FmhaTdmSchedGeometryFor =
    FmhaTdmSchedGeometry<remove_cvref_t<typename Problem::QDataType>,
                         remove_cvref_t<typename Problem::KDataType>,
                         remove_cvref_t<typename Problem::VDataType>,
                         remove_cvref_t<typename Problem::PDataType>,
                         Problem::BlockFmhaShape::kM0,
                         Problem::BlockFmhaShape::kN0,
                         Problem::BlockFmhaShape::kN1,
                         Problem::BlockFmhaShape::kQKHeaddim,
                         Problem::BlockFmhaShape::kK1,
                         sequence<Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<0>{}),
                                  Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<1>{}),
                                  Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<2>{})>,
                         sequence<Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{}),
                                  Problem::BlockFmhaShape::Gemm0WarpTile::at(number<1>{}),
                                  Problem::BlockFmhaShape::Gemm0WarpTile::at(number<2>{})>,
                         sequence<Problem::BlockFmhaShape::Gemm1WarpTile::at(number<0>{}),
                                  Problem::BlockFmhaShape::Gemm1WarpTile::at(number<1>{}),
                                  Problem::BlockFmhaShape::Gemm1WarpTile::at(number<2>{})>,
                         QkSuColumns>;

// Native tiles use four wave32 waves and 16x16x32 WMMA.
template <typename T, index_t HeadDimQK, index_t N = 128, index_t Dv = 128>
using FmhaTdmSchedNativeGeometry =
    FmhaTdmSchedGeometry<T,
                         T,
                         T,
                         T,
                         128,
                         N,
                         Dv,
                         HeadDimQK,
                         32,
                         sequence<4, 1, 1>,
                         sequence<16, 16, 32>,
                         sequence<16, 16, 32>,
                         kFmhaTdmSchedDefaultQkSuColumns<sequence<16, 16, 32>>>;

// This concrete geometry selects the validated BF16 D192 N128 placement.
using FmhaTdmSchedBf16D192N128Geometry = FmhaTdmSchedNativeGeometry<bf16_t, 192, 128>;
using FmhaTdmSchedBf16D192N64Geometry  = FmhaTdmSchedNativeGeometry<bf16_t, 192, 64>;
using FmhaTdmSchedD128Geometry         = FmhaTdmSchedNativeGeometry<bf16_t, 128>;

// Geometries the hand-written pieces (LDS layout, split-softmax mapping, output fragments,
// schedule tables) are implemented for. Geometry itself only derives;
// the Policy requires this.
template <typename Geometry>
struct FmhaTdmSchedSupportedGeometry
{
    using Q = typename Geometry::QDataType;
    static_assert((std::is_same_v<Q, bf16_t> || std::is_same_v<Q, half_t>) &&
                      std::is_same_v<typename Geometry::KDataType, Q> &&
                      std::is_same_v<typename Geometry::VDataType, Q> &&
                      std::is_same_v<typename Geometry::PDataType, Q>,
                  "qr_tdm_sched requires homogeneous native BF16 or FP16 Q/K/V/P");
    static_assert(Geometry::kM == 128 &&
                      ((Geometry::kHeadDimQK == 64 && Geometry::kDv == 64 && Geometry::kN == 64) ||
                       (Geometry::kHeadDimQK == 128 && Geometry::kDv == 128 &&
                        Geometry::kN == 128) ||
                       (Geometry::kHeadDimQK == 192 && Geometry::kDv == 128 &&
                        (Geometry::kN == 64 || Geometry::kN == 128))),
                  "qr_tdm_sched supports D64/V64 N64, D128/V128 N128, or D192/V128 N64/N128");
    static_assert(Geometry::kWaves == 4 && Geometry::kWaveSize == 32,
                  "qr_tdm_sched requires four wave32 waves");
    static_assert(Geometry::kQkWmmaK == 32 && Geometry::kPvWmmaK == 32,
                  "qr_tdm_sched supports only native K32 WMMA geometry");
    static_assert(Geometry::kQkStages == Geometry::kN / 32 &&
                      Geometry::kPvStages == Geometry::kN / 32,
                  "qr_tdm_sched requires one QK and PV stage per 32 KV rows");
    static_assert(Geometry::PRepackLayout::kInThread,
                  "gemm_0 C^T and gemm_1 A lane layouts differ; P cannot be re-packed in-thread");
    static constexpr bool value = true;
};

template <typename Geometry>
struct FmhaTdmSchedLayout
{
    static_assert(FmhaTdmSchedSupportedGeometry<Geometry>::value);
    static constexpr index_t kKPhysicalStride = Geometry::kHeadDimQK + 8;
    static constexpr index_t kVPhysicalStride = Geometry::kDv + 16;
    // TDM fields encode interval_bytes = 4 * 2^(interval + 1) and
    // pad_bytes = 4 * (amount + 1). Native 16-bit V64/V128 rows have 128/256
    // bytes, so intervals 4/5 insert 32 bytes at each row boundary (16 elements).
    static constexpr index_t kVPadAmount   = 7;
    static constexpr index_t kVPadInterval = integer_log2_floor(Geometry::kDv / 2) - 1;
};

template <index_t KWait,
          index_t VWait,
          typename QkTails,
          typename PvTails,
          bool TailDrain,
          index_t BlockPerCu,
          typename Geometry = FmhaTdmSchedBf16D192N128Geometry>
struct FmhaTdmSchedTuning
{
    using QkTailDsCounts = QkTails;
    using PvTailDsCounts = PvTails;

    static constexpr index_t kKWait      = KWait;
    static constexpr index_t kVWait      = VWait;
    static constexpr bool kTailDrain     = TailDrain;
    static constexpr index_t kBlockPerCu = BlockPerCu;
    static constexpr index_t kQkStages   = QkTails::size();
    static constexpr index_t kPvStages   = PvTails::size();

    static_assert(kKWait == 0 || kKWait == 1, "K wait count must be zero or one");
    static_assert(kVWait == 0 || kVWait == 1, "V wait count must be zero or one");
    static_assert(kQkStages == Geometry::kQkStages && kPvStages == Geometry::kPvStages,
                  "QK and PV tail sequences must match the geometry");

    template <index_t Stage>
    CK_TILE_HOST_DEVICE static constexpr index_t GetQkTailDsCount()
    {
        static_assert(Stage >= 0 && Stage < kQkStages);
        return QkTails::at(number<Stage>{});
    }

    template <index_t Stage>
    CK_TILE_HOST_DEVICE static constexpr index_t GetPvTailDsCount()
    {
        static_assert(Stage >= 0 && Stage < kPvStages);
        return PvTails::at(number<Stage>{});
    }

    static_assert(
        detail::fmha_tdm_sched_valid_qk_tails<Geometry, QkTails>(make_index_sequence<kQkStages>{}),
        "QK tail waits must match the geometry producer budgets");
    static_assert(
        detail::fmha_tdm_sched_valid_pv_tails<Geometry, PvTails>(make_index_sequence<kPvStages>{}),
        "PV tail waits must match the geometry producer budgets");
    static_assert(kBlockPerCu > 0, "BlockPerCu must be positive");
};

template <typename Geometry>
struct FmhaTdmSchedQkTailValue
{
    template <index_t Stage>
    CK_TILE_HOST_DEVICE constexpr index_t operator()(number<Stage>) const
    {
        return Stage + 1 == Geometry::kQkStages ? Geometry::kVStageLoadCount
                                                : Geometry::kKSuLoadCount;
    }
};

template <typename Geometry>
struct FmhaTdmSchedPvTailValue
{
    template <index_t Stage>
    CK_TILE_HOST_DEVICE constexpr index_t operator()(number<Stage>) const
    {
        return Stage + 1 == Geometry::kPvStages ? Geometry::kKSuLoadCount
                                                : Geometry::kVStageLoadCount;
    }
};

template <typename Geometry, index_t BlockPerCu = 1>
using FmhaTdmSchedDerivedTuning = FmhaTdmSchedTuning<
    1,
    1,
    typename sequence_gen<Geometry::kQkStages, FmhaTdmSchedQkTailValue<Geometry>>::type,
    typename sequence_gen<Geometry::kPvStages, FmhaTdmSchedPvTailValue<Geometry>>::type,
    true,
    BlockPerCu,
    Geometry>;

} // namespace ck_tile
