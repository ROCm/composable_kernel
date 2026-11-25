// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <string>
#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_mx_asmem_bsmem_creg_tdm_scheduled.hpp"
#include "ck_tile/ops/gemm/warp/warp_wmma_gemm.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_traits.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_tdm_default_policy.hpp"

namespace ck_tile {

/// Why a Problem and policy do not build, or an M x N x K problem does not run, in the order the
/// checks run: the problem's shape, the types and tile, the policy's knobs, the cluster, the LDS
/// layout. Ok when it builds.
enum class MxTdmScheduledReason : int
{
    Ok,
    ShapePositive,
    KScaleBlock,
    OutDtype,
    TileMN,
    Dtype,
    DtypeWidth,
    TileK,
    WaveGroup,
    Mma,
    TileAtoms,
    ScaleSpan,
    KShort,
    KLong,
    KTailFull,
    SplitBRead,
    Gl2Distance,
    ClusterDim,
    ClusterSize,
    ClusterGrid,
    LdsPad,
    Accumulators,
    InterleaveRepeat,
    InterleaveBaselineSplit,
    InterleaveCapacity,
    LdsCapacity
};

/// The short code of a reason, as tile_engine's mx_tdm_scheduled.py validator reports it.
CK_TILE_HOST_DEVICE constexpr const char* mx_tdm_scheduled_reason_code(MxTdmScheduledReason r)
{
    switch(r)
    {
    case MxTdmScheduledReason::Ok: return "ok";
    case MxTdmScheduledReason::ShapePositive: return "shape_positive";
    case MxTdmScheduledReason::KScaleBlock: return "k_scale_block";
    case MxTdmScheduledReason::OutDtype: return "out_dtype";
    case MxTdmScheduledReason::TileMN: return "tile_mn";
    case MxTdmScheduledReason::Dtype: return "dtype";
    case MxTdmScheduledReason::DtypeWidth: return "dtype_width";
    case MxTdmScheduledReason::TileK: return "tile_k";
    case MxTdmScheduledReason::WaveGroup: return "wave_group";
    case MxTdmScheduledReason::Mma: return "mma";
    case MxTdmScheduledReason::TileAtoms: return "tile_atoms";
    case MxTdmScheduledReason::ScaleSpan: return "scale_span";
    case MxTdmScheduledReason::KShort: return "k_short";
    case MxTdmScheduledReason::KLong: return "k_long";
    case MxTdmScheduledReason::KTailFull: return "k_tail_full";
    case MxTdmScheduledReason::SplitBRead: return "split_b_read";
    case MxTdmScheduledReason::Gl2Distance: return "gl2_distance";
    case MxTdmScheduledReason::ClusterDim: return "cluster_dim";
    case MxTdmScheduledReason::ClusterSize: return "cluster_size";
    case MxTdmScheduledReason::ClusterGrid: return "cluster_grid";
    case MxTdmScheduledReason::LdsPad: return "lds_pad";
    case MxTdmScheduledReason::Accumulators: return "accumulators";
    case MxTdmScheduledReason::InterleaveRepeat: return "interleave_repeat";
    case MxTdmScheduledReason::InterleaveBaselineSplit: return "interleave_baseline_split";
    case MxTdmScheduledReason::InterleaveCapacity: return "interleave_capacity";
    case MxTdmScheduledReason::LdsCapacity: return "lds_capacity";
    }
    return "unknown";
}

/// The message of a reason.
CK_TILE_HOST_DEVICE constexpr const char* mx_tdm_scheduled_reason_message(MxTdmScheduledReason r)
{
    switch(r)
    {
    case MxTdmScheduledReason::Ok: return "ok";
    case MxTdmScheduledReason::ShapePositive: return "M, N and K must be positive";
    case MxTdmScheduledReason::KScaleBlock:
        return "K must be a multiple of 32: each E8M0 scale covers 32 K values";
    case MxTdmScheduledReason::OutDtype: return "out_dtype must be one of fp8e4m3, fp16, bf16";
    case MxTdmScheduledReason::TileMN: return "tile_m and tile_n must be in 16, 32, 64, 128, 256";
    case MxTdmScheduledReason::Dtype: return "a_dtype and b_dtype must be in fp4, fp8, bf8";
    case MxTdmScheduledReason::DtypeWidth:
        return "a_dtype and b_dtype must have the same width (no FP4 x 8-bit)";
    case MxTdmScheduledReason::TileK:
        return "tile_k must be in 128, 256, 512, 1024 for fp4 and 128, 256 for 8-bit types";
    case MxTdmScheduledReason::WaveGroup: return "wave_group must be 2x2, 4x1 or 1x4 (4 waves)";
    case MxTdmScheduledReason::Mma: return "mma must be 32x16 or 16x16 for fp4 and 16x16 for 8-bit";
    case MxTdmScheduledReason::TileAtoms: return "the tile must hold whole WMMA atoms per wave";
    case MxTdmScheduledReason::ScaleSpan:
        return "A and B scale bytes must each be at most 2^31 (i32 offsets)";
    case MxTdmScheduledReason::KShort: return "K must hold at least one full tile";
    case MxTdmScheduledReason::KLong:
        return "K plus the K the loads look ahead (max(gl2_prefetch_distance, 2) + 1 tiles) must "
               "fit in i32";
    case MxTdmScheduledReason::KTailFull:
        return "K mod tile_k must be at most tile_k - 128: no instance of this tile_k runs a "
               "longer K tail";
    case MxTdmScheduledReason::SplitBRead:
        return "SplitBLdsRead needs PrefetchLdsRead and an even N wave repeat";
    case MxTdmScheduledReason::Gl2Distance:
        return "gl2_prefetch_distance must be 0 (off) or 2..8 tiles";
    case MxTdmScheduledReason::ClusterDim: return "cluster_dim must be a positive (x, y, z) triple";
    case MxTdmScheduledReason::ClusterSize:
        return "cluster_dim must be 2D with at most 16 workgroups";
    case MxTdmScheduledReason::ClusterGrid: return "the grid must be divisible by cluster_dim";
    case MxTdmScheduledReason::LdsPad: return "an LDS pad does not encode as TDM padding";
    case MxTdmScheduledReason::Accumulators: return "the accumulators per lane exceed 1024 VGPRs";
    case MxTdmScheduledReason::InterleaveRepeat:
        return "InterleaveLdsRegions needs the split operand's wave repeat to cover its wave's "
               "rows";
    case MxTdmScheduledReason::InterleaveBaselineSplit:
        return "InterleaveLdsRegions: the baseline layout already splits B's components across "
               "64 KiB segments";
    case MxTdmScheduledReason::InterleaveCapacity:
        return "InterleaveLdsRegions: the aligned layout needs more LDS than fits";
    case MxTdmScheduledReason::LdsCapacity: return "LDS bytes exceed the 327680 B of gfx1250";
    }
    return "unknown";
}

/**
 * @brief The GEMM traits of a MxGemmPipelineAgBgCrCompTDMScheduled instance: RCR
 * TileGemmUniversalTraits plus the code paths that depend on the problem's M, N and K.
 *
 * EdgeM / EdgeN (kPadM / kPadN): the last M (N) tile is partial, its TDM rows clipped and its D
 * vectors masked. WideA / WideB / WideD (LargeTensors if any): A's, B's or D's byte offsets pass
 * 2^31 and are i64. TailSteps: the K steps of a last, partial tile when tile_k does not divide K
 * (0 without one). The edge and wide paths are supersets of the ones they replace (an edge kernel
 * runs whole tiles, a wide one narrow offsets), so one instance serves every problem its traits
 * cover; TailSteps must be the problem's. The parity of the whole-tile count is the pipeline's
 * TailNumber, chosen at run time.
 * MxTdmScheduledSelectTraits gives the values an M x N x K problem needs.
 */
template <bool EdgeM_        = false,
          bool EdgeN_        = false,
          bool WideA_        = false,
          bool WideB_        = false,
          bool WideD_        = false,
          index_t TailSteps_ = 0>
struct MxTdmScheduledGemmTraits : TileGemmUniversalTraits<EdgeM_,
                                                          EdgeN_,
                                                          false,
                                                          true,
                                                          tensor_layout::gemm::RowMajor,
                                                          tensor_layout::gemm::ColumnMajor,
                                                          tensor_layout::gemm::RowMajor,
                                                          false,
                                                          false,
                                                          false,
                                                          1,
                                                          false,
                                                          16,
                                                          DataCachePrefetchKind::None,
                                                          DataCachePrefetchKind::None,
                                                          false,
                                                          WideA_ || WideB_ || WideD_>
{
    static constexpr bool kEdgeM        = EdgeM_;
    static constexpr bool kEdgeN        = EdgeN_;
    static constexpr bool kWideA        = WideA_;
    static constexpr bool kWideB        = WideB_;
    static constexpr bool kWideD        = WideD_;
    static constexpr index_t kTailSteps = TailSteps_;
    static constexpr bool kHasTail      = TailSteps_ != 0;
};

/// The MxTdmScheduledGemmTraits values of one M x N x K problem, as host values.
struct MxTdmScheduledShapeTraits
{
    bool edge_m;
    bool edge_n;
    bool wide_a;
    bool wide_b;
    bool wide_d;
    index_t tail_steps;
};

/**
 * @brief The MxTdmScheduledGemmTraits an M x N x K problem needs from an instance of Problem's
 *        tile and types.
 */
template <typename Problem>
CK_TILE_HOST_DEVICE constexpr MxTdmScheduledShapeTraits
MxTdmScheduledSelectTraits(long_index_t M, long_index_t N, long_index_t K)
{
    using Shape                 = typename Problem::BlockGemmShape;
    constexpr long_index_t span = 1ll << 31;
    constexpr index_t k_per_byte =
        std::is_same_v<remove_cvref_t<typename Problem::ADataType>, pk_fp4_t> ? 2 : 1;
    const long_index_t span_m   = integer_least_multiple(M, Shape::kM);
    const long_index_t span_n   = integer_least_multiple(N, Shape::kN);
    const long_index_t packed_k = K / k_per_byte;
    return {M % Shape::kM != 0,
            N % Shape::kN != 0,
            span_m * packed_k > span,
            span_n * packed_k > span,
            span_m * span_n > span,
            static_cast<index_t>(integer_divide_ceil(K % Shape::kK, 128))};
}

/**
 * @brief Problem of MxGemmPipelineAgBgCrCompTDMScheduled: E = A @ B^T.
 *
 * A is M x K and B is N x K, both K-contiguous (RCR); E is M x N row-major. A and B are
 * pk_fp4_t, fp8_t or bf8_t of the same width, with one E8M0 scale per 32 K values in the K-major
 * layout [ceil(K / 128)][M or N][4] (see mx_scale_to_kmajor in host/mx_processing.hpp). ODataType
 * (E) is fp8_t (saturating), half_t or bf16_t.
 *
 * BlockGemmShape is a TileGemmShape: block tile (tile_m, tile_n, tile_k), block warps 2x2x1,
 * 4x1x1 or 1x4x1 and warp tile 32x16x128 (FP4) or 16x16x128: the scaled WMMA atom. A
 * ClusterTileGemmShape adds the cluster (kclusterM M tiles by kclusterN N tiles, kclusterK 1):
 * A tiles are multicast along N, B tiles along M. A TileGemmShape is no cluster launch.
 *
 * An MxGemmPipelineProblem with float accumulators, PassThrough element functions and E8M0
 * scales per 32 K values, plus ODataType; kBlockSize follows its convention (warps x
 * get_warp_size()). Traits is an MxTdmScheduledGemmTraits. Scheduler must be Intrawave: the
 * pipeline is one explicitly scheduled loop in which every wave interleaves its LDS reads and
 * WMMAs.
 *
 * TailNum is the K loop's variant, as in CK's GemmPipelineProblem: TailNumber::One (an odd count
 * of whole K tiles) or Two (even), picked on the host by the pipeline's TailHandler, one kernel
 * per variant; TailNumber::Full runs any count, its odd tile chosen at run time.
 */
template <typename ADataType_,
          typename BDataType_,
          typename ODataType_,
          typename BlockGemmShape_,
          typename Traits_                 = MxTdmScheduledGemmTraits<>,
          GemmPipelineScheduler Scheduler_ = GemmPipelineScheduler::Intrawave,
          TailNumber TailNum_              = TailNumber::Full>
struct MxGemmTdmScheduledProblem
    : MxGemmPipelineProblem<ADataType_, BDataType_, float, BlockGemmShape_, Traits_, Scheduler_>
{
    using ODataType   = remove_cvref_t<ODataType_>;
    using AccDataType = float;

    static constexpr auto TailNum = TailNum_;
    static_assert(TailNum == TailNumber::One || TailNum == TailNumber::Two ||
                      TailNum == TailNumber::Full,
                  "comp_tdm_scheduled: TailNum is TailNumber::One, Two or Full");
    /// This Problem with TailNum T.
    template <TailNumber T>
    using WithTailNum = MxGemmTdmScheduledProblem<ADataType_,
                                                  BDataType_,
                                                  ODataType_,
                                                  BlockGemmShape_,
                                                  Traits_,
                                                  Scheduler_,
                                                  T>;

    static_assert(std::is_same_v<typename MxGemmTdmScheduledProblem::ALayout,
                                 tensor_layout::gemm::RowMajor> &&
                      std::is_same_v<typename MxGemmTdmScheduledProblem::BLayout,
                                     tensor_layout::gemm::ColumnMajor> &&
                      std::is_same_v<typename MxGemmTdmScheduledProblem::CLayout,
                                     tensor_layout::gemm::RowMajor>,
                  "comp_tdm_scheduled: A and B are K-contiguous and C row-major (RCR)");
};

namespace mx_tdm_scheduled {

/// Instantiating this with a reason other than Ok fails, and the compiler names the reason.
template <MxTdmScheduledReason R>
struct InvalidTuning;
template <>
struct InvalidTuning<MxTdmScheduledReason::Ok>
{
    static constexpr bool value = true;
};

} // namespace mx_tdm_scheduled

/**
 * @brief Policy of MxGemmPipelineAgBgCrCompTDMScheduled: the tuning knobs that are not part of the
 * Problem's shape, and everything derived from them and the Problem at compile time: the build
 * rules (GetReason), the K step's reads and waits, and the LDS layout.
 *
 * @tparam Gl2PrefetchDistance  Each tile prefetches into GL2 the tile this many K tiles ahead;
 *                              0 is off, else 2..8.
 * @tparam PrefetchLdsRead      K step s + 1's fragments are read from LDS while K step s's WMMAs
 *                              run. Off: each K step reads, waits for its reads, then computes.
 * @tparam SplitBLdsRead        B's fragments are read in two halves with a wait for each, and
 *                              the WMMAs run B-outer, so the first half's WMMAs start early.
 *                              Needs PrefetchLdsRead and an even N wave repeat.
 * @tparam InterleaveLdsRegions Place the A and B components in different 64 KiB LDS segments
 *                              (GetInterleavePlacement). Off: the regions follow one another.
 */
template <index_t Gl2PrefetchDistance = 4,
          bool PrefetchLdsRead        = true,
          bool SplitBLdsRead          = true,
          bool InterleaveLdsRegions   = true>
struct MxGemmPipelineAgBgCrCompTDMScheduledPolicy
{
    static constexpr index_t kGl2PrefetchDistance = Gl2PrefetchDistance;
    static constexpr bool kPrefetchLdsRead        = PrefetchLdsRead;
    static constexpr bool kSplitBLdsRead          = SplitBLdsRead;
    static constexpr bool kInterleaveLdsRegions   = InterleaveLdsRegions;

    /// K of the scaled WMMA atom, the warp tile's K.
    static constexpr index_t kAtomK = 128;
    /// gfx1250 runs wave32; get_warp_size() is the compiler pass's wave size, 64 in the host pass.
    static constexpr index_t kWaveSize = 32;
    /// A lane's K chunk of a row and K step; the lane groups (lane / 16) read side by side.
    static constexpr index_t kChunkBytes = 16;
    static constexpr index_t kLaneGroups = 2;
    /// The GL2 prefetch's granule.
    static constexpr index_t kGl2LineBytes = 256;
    /// LDS of a gfx1250 workgroup; get_smem_capacity() reports 64 KiB for gfx1250.
    static constexpr index_t kLdsCapacityBytes = 327680;
    /// The largest s_wait_dscnt immediate.
    static constexpr index_t kDscntMax = 63;
    /// VGPRs per lane, which bound the accumulators.
    static constexpr index_t kVgprs = 1024;

    /// Whether T is an A or B type: pk_fp4_t, fp8_t or bf8_t.
    template <typename T>
    CK_TILE_HOST_DEVICE static constexpr bool IsMxDataType()
    {
        return std::is_same_v<T, pk_fp4_t> || std::is_same_v<T, fp8_t> || std::is_same_v<T, bf8_t>;
    }
    /// Bits of an A or B element.
    template <typename T>
    CK_TILE_HOST_DEVICE static constexpr index_t GetElemBits()
    {
        return std::is_same_v<T, pk_fp4_t> ? 4 : 8;
    }
    /// Whether ODataType is fp8_t, half_t or bf16_t.
    template <typename T>
    CK_TILE_HOST_DEVICE static constexpr bool IsOutDataType()
    {
        return std::is_same_v<T, fp8_t> || std::is_same_v<T, half_t> || std::is_same_v<T, bf16_t>;
    }

    /// The WMMA atom: 32x16 (FP4 only) or 16x16 rows by columns, from the warp tile. Any other
    /// warp tile reads as 16x16 and is rejected as "mma".
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool IsMma32()
    {
        using WarpTile = typename Problem::BlockGemmShape::WarpTile;
        return WarpTile::at(number<0>{}) == 32 && WarpTile::at(number<1>{}) == 16;
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetAtomM()
    {
        return IsMma32<Problem>() ? 32 : 16;
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetAtomN()
    {
        return 16;
    }
    template <typename Problem, bool IsA>
    CK_TILE_HOST_DEVICE static constexpr index_t GetAtom()
    {
        return IsA ? GetAtomM<Problem>() : GetAtomN<Problem>();
    }

    /// Waves along M and N; 0 unless the wave group is one wave deep in K on 128-wide atoms
    /// (rejected as "wave_group").
    template <typename Problem, bool IsA>
    CK_TILE_HOST_DEVICE static constexpr index_t GetWarps()
    {
        using Shape = typename Problem::BlockGemmShape;
        using Warps = typename Shape::BlockWarps;
        constexpr bool k_ok =
            Warps::at(number<2>{}) == 1 && Shape::WarpTile::at(number<2>{}) == kAtomK;
        return k_ok ? Warps::at(number < IsA ? 0 : 1 > {}) : 0;
    }

    /// WMMA atoms per wave along M and N.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetWaveRepeatM()
    {
        return Problem::BlockGemmShape::kM / (GetWarps<Problem, true>() * GetAtomM<Problem>());
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetWaveRepeatN()
    {
        return Problem::BlockGemmShape::kN / (GetWarps<Problem, false>() * GetAtomN<Problem>());
    }
    template <typename Problem, bool IsA>
    CK_TILE_HOST_DEVICE static constexpr index_t GetRepeats()
    {
        return IsA ? GetWaveRepeatM<Problem>() : GetWaveRepeatN<Problem>();
    }

    /// Threads of the device pass: the waves times 32.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetBlockSize()
    {
        return kWaveSize * Problem::BlockGemmShape::NumWarps;
    }

    /// 128-wide K steps per K tile.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetKSteps()
    {
        return Problem::BlockGemmShape::kK / kAtomK;
    }

    /// Floats of one WMMA C fragment per lane.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetCFragLen()
    {
        return IsMma32<Problem>() ? 16 : 8;
    }

    /// K values per byte of A and B (2 for FP4).
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetKPerByte()
    {
        return 8 / GetElemBits<typename Problem::ADataType>();
    }

    /// Bytes of a 128-wide K step in an A or B row.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetStepBytes()
    {
        return kAtomK / GetKPerByte<Problem>();
    }

    /// 16-byte LDS reads per lane per A or B row fragment, and the byte offset of read i; lane
    /// group g reads kChunkBytes * g further.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetChunksPerLane()
    {
        return GetStepBytes<Problem>() / (kLaneGroups * kChunkBytes);
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetChunkOffset(index_t i)
    {
        return i * kLaneGroups * kChunkBytes;
    }

    /// Whether the next K step's fragments are read while this one's WMMAs run.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool GetPrefetchLdsRead()
    {
        return kPrefetchLdsRead;
    }

    /// Whether the B fragments are read in two halves.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool GetSplitBLdsRead()
    {
        return kPrefetchLdsRead && kSplitBLdsRead;
    }

    /// LDS reads of a K step, clamped to the 6-bit s_wait_dscnt immediate. Above the clamp (FP8
    /// and BF8 tiles with 16 repeats, or 8 x 8) each wait also drains the oldest reads of the
    /// next K step before the WMMAs: still correct, but less read latency is hidden. The waits
    /// are only correct while GetLdsReads is at most the ds_loads a K step issues.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetReadsPerStep()
    {
        return min_i(GetLdsReads<Problem>(), kDscntMax);
    }

    /// Reads of the other buffer left in flight at the cluster late wait, clamped.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetLatePendingReads()
    {
        return kPrefetchLdsRead ? GetReadsPerStep<Problem>() : 0;
    }

    /// The first compile-time rule the Problem and this policy break, or Ok. The rules that need
    /// M, N and K are the pipeline's GetUnsupportedReason.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr MxTdmScheduledReason GetReason()
    {
        using R                  = MxTdmScheduledReason;
        using Shape              = typename Problem::BlockGemmShape;
        using WarpTile           = typename Shape::WarpTile;
        using ADataType          = remove_cvref_t<typename Problem::ADataType>;
        using BDataType          = remove_cvref_t<typename Problem::BDataType>;
        constexpr index_t tile_m = Shape::kM;
        constexpr index_t tile_n = Shape::kN;
        constexpr index_t tile_k = Shape::kK;
        if(!IsOutDataType<typename Problem::ODataType>())
            return R::OutDtype;
        if(!IsValidTileMN(tile_m) || !IsValidTileMN(tile_n))
            return R::TileMN;
        if(!IsMxDataType<ADataType>() || !IsMxDataType<BDataType>())
            return R::Dtype;
        if(GetElemBits<BDataType>() != GetElemBits<ADataType>())
            return R::DtypeWidth;
        constexpr bool fp4 = std::is_same_v<ADataType, pk_fp4_t>;
        if(!(tile_k == 128 || tile_k == 256 || (fp4 && (tile_k == 512 || tile_k == 1024))))
            return R::TileK;
        constexpr index_t warp_m = GetWarps<Problem, true>();
        constexpr index_t warp_n = GetWarps<Problem, false>();
        if(!((warp_m == 2 && warp_n == 2) || (warp_m == 4 && warp_n == 1) ||
             (warp_m == 1 && warp_n == 4)))
            return R::WaveGroup;
        constexpr index_t wt_m = WarpTile::at(number<0>{});
        constexpr index_t wt_n = WarpTile::at(number<1>{});
        if(!((wt_m == 16 && wt_n == 16) || (fp4 && wt_m == 32 && wt_n == 16)))
            return R::Mma;
        if(tile_m % (warp_m * GetAtomM<Problem>()) || tile_n % (warp_n * GetAtomN<Problem>()))
            return R::TileAtoms;
        const index_t rm = GetWaveRepeatM<Problem>();
        const index_t rn = GetWaveRepeatN<Problem>();
        if(kSplitBLdsRead && (!kPrefetchLdsRead || rn % 2))
            return R::SplitBRead;
        if(kGl2PrefetchDistance != 0 && !(2 <= kGl2PrefetchDistance && kGl2PrefetchDistance <= 8))
            return R::Gl2Distance;
        if(Shape::kclusterN < 1 || Shape::kclusterM < 1 || Shape::kclusterK < 1)
            return R::ClusterDim;
        if(Shape::kclusterK != 1 || Shape::kclusterN * Shape::kclusterM > 16)
            return R::ClusterSize;
        const DerivedPads pads  = GetDerivedLdsPads<Problem>();
        const index_t row_bytes = tile_k / GetKPerByte<Problem>();
        struct PadCheck
        {
            LdsPad pad;
            index_t rows;
            index_t chunk;
            bool data;
        };
        const PadCheck checks[4] = {{pads.a, tile_m / 2, 16, true},
                                    {pads.b, tile_n / 2, 16, true},
                                    {pads.a_scale, tile_m / 2, min_i(16, 4 * rm), false},
                                    {pads.b_scale, tile_n / 2, min_i(16, 4 * rn), false}};
        for(const PadCheck& c : checks)
        {
            const R reason = GetPadReason(c.pad, c.data ? row_bytes : 0, c.rows, c.chunk);
            if(reason != R::Ok)
                return reason;
            if(c.data && c.pad.on() && c.pad.block_bytes <= kTdmPadPeriodMax &&
               c.pad.block_bytes % row_bytes)
                return R::LdsPad;
        }
        if(rm * rn * GetCFragLen<Problem>() > kVgprs)
            return R::Accumulators;
        const LdsLayout layout = GetLdsLayout<Problem>();
        if(layout.reason != R::Ok)
            return layout.reason;
        if(layout.lds_bytes > kLdsCapacityBytes)
            return R::LdsCapacity;
        return R::Ok;
    }

    /// nullptr if the Problem and this policy build, else the reason code (see
    /// mx_tdm_scheduled_reason_code). The rules that need M, N and K are the pipeline's
    /// GetUnsupportedReason.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr const char* Validate()
    {
        constexpr MxTdmScheduledReason r = GetReason<Problem>();
        return r == MxTdmScheduledReason::Ok ? nullptr : mx_tdm_scheduled_reason_code(r);
    }

    /// Bytes of LDS: both buffers.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return GetLdsLayout<Problem>().lds_bytes;
    }

    /// Bytes between the two LDS buffers.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsBufferStride()
    {
        return GetLdsLayout<Problem>().buffer_stride;
    }

    /// Bytes of an A or B row of one K tile in LDS, padding excluded.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsRowBytes()
    {
        return GetLdsLayout<Problem>().row_bytes;
    }

    /// The LDS placement of A's or B's data or scales in one buffer: two TDM components (one per
    /// wave pair) of `rows` rows, component c at base + c * stride, padded by pad_bytes after every
    /// block_bytes; a data load in iterate mode (a pad period above 1024 B) writes iterate_rows
    /// rows per iteration, iterate_stride apart.
    template <typename Problem, bool IsA, bool IsScale = false>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsOperand()
    {
        constexpr LdsLayout l = GetLdsLayout<Problem>();
        if constexpr(IsScale)
            return IsA ? l.a_scale : l.b_scale;
        else
            return IsA ? l.a : l.b;
    }

    /// Whether the TDM pads A's or B's data or scales in LDS: a data load in iterate mode is
    /// unpadded (its rows land at iterate_stride instead).
    template <typename Problem, bool IsA, bool IsScale = false>
    CK_TILE_HOST_DEVICE static constexpr bool GetLdsPadEnable()
    {
        constexpr LdsOperand op = GetLdsOperand<Problem, IsA, IsScale>();
        return (IsScale || op.iterate_rows == 0) && op.pad_bytes > 0;
    }

    /// The TDM LDS padding of A's or B's data or scales as the default TDM policy gives it:
    /// (pad_enable, pad_amount, pad_interval), zeros without padding.
    template <typename Problem, bool IsA, bool IsScale = false>
    CK_TILE_HOST_DEVICE static constexpr auto GetLdsPaddingConfig()
    {
        constexpr LdsOperand op = GetLdsOperand<Problem, IsA, IsScale>();
        constexpr bool enable   = GetLdsPadEnable<Problem, IsA, IsScale>();
        constexpr TDMLdsPaddingConfig padding =
            enable ? make_tdm_lds_padding_config(op.block_bytes, op.pad_bytes)
                   : TDMLdsPaddingConfig{};
        return make_tuple(number<enable>{},
                          number<static_cast<index_t>(padding.pad_amount)>{},
                          number<static_cast<index_t>(padding.pad_interval)>{});
    }

    /// A's or B's data in LDS as the TDM loads write it, over [tile row, K] in elements of the
    /// operand: component c's op.rows rows start op.stride bytes after component c - 1's. The
    /// offset is from the operand's base.
    template <typename Problem, bool IsA>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsTdmBlockDescriptor()
    {
        constexpr auto runs       = MakeLdsRunsDescriptor<Problem, IsA, true>();
        constexpr auto lengths    = runs.get_lengths();
        constexpr index_t comps   = lengths[number<0>{}];
        constexpr index_t outer   = lengths[number<1>{}];
        constexpr index_t inner   = lengths[number<2>{}];
        constexpr index_t k_elems = lengths[number<3>{}];
        return transform_tensor_descriptor(
            runs,
            make_tuple(
                make_merge_transform(make_tuple(number<comps>{}, number<outer>{}, number<inner>{})),
                make_pass_through_transform(number<k_elems>{})),
            make_tuple(sequence<0, 1, 2>{}, sequence<3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    /// The DRAM distribution of one TDM component load of A or B: one box of the component's
    /// rows (an iterate-mode block of them) by the K tile, the same for every wave; a wave's
    /// component is its window's origin. With IsScale, of A's or B's K-major scales: the tile's
    /// K steps by the component's rows.
    template <typename Problem, bool IsA, bool IsScale = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeDramTdmTileDistribution()
    {
        constexpr LdsOperand op    = GetLdsOperand<Problem, IsA, IsScale>();
        constexpr index_t box_rows = op.iterate_rows > 0 ? op.iterate_rows : op.rows;
        constexpr index_t len0     = IsScale ? GetKSteps<Problem>() : box_rows;
        constexpr index_t len1     = IsScale ? op.rows : Problem::BlockGemmShape::kK;
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<Problem::BlockGemmShape::NumWarps>,
                                       tuple<sequence<len0>, sequence<len1>>,
                                       tuple<sequence<0>>,
                                       tuple<sequence<0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{},
            bool_constant<true>{});
    }

    /// A's or B's data in LDS as a descriptor over [tile row in fragment order, K word]. Row t =
    /// (wave * repeats + r) * atom + l is tile row wave * atom * repeats + l * repeats + r (V1's
    /// PackMNIter order), so one WMMA fragment is atom consecutive rows; the offset, in 4-byte
    /// words, is GetLdsRowByteOffset's less the operand's base.
    template <typename Problem, bool IsA>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsBlockDescriptor()
    {
        constexpr LdsOperand op     = GetLdsOperand<Problem, IsA>();
        constexpr index_t row_bytes = GetLdsRowBytes<Problem>();
        constexpr index_t warps     = GetWarps<Problem, IsA>();
        constexpr index_t atom      = GetAtom<Problem, IsA>();
        constexpr index_t repeats   = GetRepeats<Problem, IsA>();
        constexpr index_t rows      = warps * atom * repeats;
        constexpr index_t words     = row_bytes / 4;
        constexpr auto naive        = MakeLdsRunsDescriptor<Problem, IsA>();
        constexpr index_t inner     = naive.get_lengths()[number<2>{}];
        constexpr auto by_row       = transform_tensor_descriptor(
            naive,
            make_tuple(make_merge_transform(make_tuple(
                           number<rows / op.rows>{}, number<op.rows / inner>{}, number<inner>{})),
                       make_pass_through_transform(number<words>{})),
            make_tuple(sequence<0, 1, 2>{}, sequence<3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
        constexpr auto by_wave = transform_tensor_descriptor(
            by_row,
            make_tuple(make_unmerge_transform(
                           make_tuple(number<warps>{}, number<atom>{}, number<repeats>{})),
                       make_pass_through_transform(number<words>{})),
            make_tuple(sequence<0>{}, sequence<1>{}),
            make_tuple(sequence<0, 1, 2>{}, sequence<3>{}));
        return transform_tensor_descriptor(
            by_wave,
            make_tuple(make_merge_transform(
                           make_tuple(number<warps>{}, number<repeats>{}, number<atom>{})),
                       make_pass_through_transform(number<words>{})),
            make_tuple(sequence<0, 2, 1>{}, sequence<3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    /// A's or B's data in LDS as runs of rows, over [component, run, run row, K]: K in 4-byte
    /// words, or with InElements in elements of the operand.
    template <typename Problem, bool IsA, bool InElements = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsRunsDescriptor()
    {
        constexpr LdsOperand op     = GetLdsOperand<Problem, IsA>();
        constexpr index_t row_bytes = GetLdsRowBytes<Problem>();
        constexpr index_t warps     = GetWarps<Problem, IsA>();
        constexpr index_t atom      = GetAtom<Problem, IsA>();
        constexpr index_t repeats   = GetRepeats<Problem, IsA>();
        constexpr index_t rows      = warps * atom * repeats;
        // A component's rows split into (outer, inner) runs: iterate-mode blocks, padded blocks
        // of whole rows, or one run (rows padded within, or unpadded).
        constexpr bool iterate         = op.iterate_rows > 0;
        constexpr bool padded          = !iterate && op.pad_bytes > 0;
        constexpr bool block_pad       = padded && op.block_bytes >= row_bytes;
        constexpr index_t inner        = iterate     ? op.iterate_rows
                                         : block_pad ? op.block_bytes / row_bytes
                                                     : op.rows;
        constexpr index_t outer_stride = iterate     ? op.iterate_stride
                                         : block_pad ? op.block_bytes + op.pad_bytes
                                                     : 0;
        constexpr index_t row_stride   = padded && !block_pad
                                             ? row_bytes + row_bytes / op.block_bytes * op.pad_bytes
                                             : row_bytes;
        static_assert(rows % op.rows == 0 && op.rows % inner == 0 && row_bytes % 4 == 0 &&
                          op.stride % 4 == 0 && outer_stride % 4 == 0 && row_stride % 4 == 0,
                      "comp_tdm_scheduled: the LDS layout is not a whole number of words per run");
        constexpr auto unit = [](index_t bytes) {
            return InElements ? bytes * GetKPerByte<Problem>() : bytes / 4;
        };
        return make_naive_tensor_descriptor(make_tuple(number<rows / op.rows>{},
                                                       number<op.rows / inner>{},
                                                       number<inner>{},
                                                       number<unit(row_bytes)>{}),
                                            make_tuple(number<unit(op.stride)>{},
                                                       number<unit(outer_stride)>{},
                                                       number<unit(row_stride)>{},
                                                       number<1>{}),
                                            number < InElements ? 1 : 4 > {},
                                            number<1>{});
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        return MakeLdsBlockDescriptor<Problem, true>();
    }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        return MakeLdsBlockDescriptor<Problem, false>();
    }

    /// 16-row halves of an A or B WMMA fragment: 2 for A of the 32x16 atom, else 1.
    template <typename Problem, bool IsA>
    CK_TILE_HOST_DEVICE static constexpr index_t GetFragmentHalves()
    {
        return IsA ? GetAtomM<Problem>() / 16 : 1;
    }

    /// One wave's read of an A or B fragment of one K step, in words: lane l reads row l % 16 of
    /// each half and, per 32-byte chunk, the 16 bytes l / 16 selects.
    ///   rows: (half, l % 16)   K words: (chunk, l / 16, 4)
    template <typename Problem, bool IsA>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsReadDistribution()
    {
        constexpr index_t chunks = GetChunksPerLane<Problem>();
        if constexpr(GetFragmentHalves<Problem, IsA>() == 2)
            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<>,
                                           tuple<sequence<2, 16>, sequence<chunks, 2, 4>>,
                                           tuple<sequence<2, 1>>,
                                           tuple<sequence<1, 1>>,
                                           sequence<1, 2, 2>,
                                           sequence<0, 0, 2>>{});
        else
            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<>,
                                           tuple<sequence<16>, sequence<chunks, 2, 4>>,
                                           tuple<sequence<2, 1>>,
                                           tuple<sequence<1, 0>>,
                                           sequence<2, 2>,
                                           sequence<0, 2>>{});
    }

    /// Word offset of fragment-order row t of A's or B's data from the operand's base, as
    /// MakeLdsBlockDescriptor maps it.
    template <typename Problem, bool IsA>
    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsRowOffset(index_t t)
    {
        constexpr LdsOperand op   = GetLdsOperand<Problem, IsA>();
        constexpr index_t atom    = GetAtom<Problem, IsA>();
        constexpr index_t repeats = GetRepeats<Problem, IsA>();
        const index_t wave        = t / (repeats * atom);
        const index_t r           = t / atom % repeats;
        const index_t l           = t % atom;
        const index_t row         = wave * atom * repeats + l * repeats + r;
        return (GetLdsRowByteOffset<index_t>(op, GetLdsRowBytes<Problem>(), row) - op.base) / 4;
    }

    /// Whether a fragment read can take its repeat and half offsets as constants: for every
    /// lane's first row, moving by a repeat r and a half h adds the offset of (r, h) from row 0.
    /// Also spot-checks MakeLdsBlockDescriptor against GetLdsRowByteOffset (a full check is
    /// past the constexpr step limit).
    template <typename Problem, bool IsA>
    CK_TILE_HOST_DEVICE static constexpr bool IsLdsReadLinear()
    {
        constexpr index_t warps   = GetWarps<Problem, IsA>();
        constexpr index_t atom    = GetAtom<Problem, IsA>();
        constexpr index_t repeats = GetRepeats<Problem, IsA>();
        constexpr index_t halves  = GetFragmentHalves<Problem, IsA>();
        constexpr index_t rows    = warps * repeats * atom;
        constexpr auto desc       = MakeLdsBlockDescriptor<Problem, IsA>();
        const auto offset         = [](index_t t) { return GetLdsRowOffset<Problem, IsA>(t); };
        for(const index_t t : {index_t{1}, atom - 1, atom, rows / 2 + 1, rows - 1})
            if(desc.calculate_offset(make_multi_index(t, 0)) != offset(t))
                return false;
        const index_t wave_offset = offset(repeats * atom);
        for(index_t wave = 0; wave < warps; ++wave)
            for(index_t l = 0; l < 16; ++l)
            {
                const index_t t0 = wave * repeats * atom + l;
                const index_t o0 = offset(t0);
                if(o0 != wave * wave_offset + offset(l))
                    return false;
                for(index_t r = 0; r < repeats; ++r)
                    for(index_t h = 0; h < halves; ++h)
                        if(offset(t0 + r * atom + h * 16) != o0 + offset(r * atom) + offset(h * 16))
                            return false;
            }
        return true;
    }

    /// A's or B's E8M0 scales in LDS as the TDM loads write them, over [K step, tile row] in
    /// 4-byte packs: component c's op.rows rows of every step start op.stride bytes after
    /// component c - 1's, each step's rows follow the last step's, padded per block.
    template <typename Problem, bool IsA>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsScaleTdmBlockDescriptor()
    {
        constexpr LdsOperand op   = GetLdsOperand<Problem, IsA, true>();
        constexpr index_t warps   = GetWarps<Problem, IsA>();
        constexpr index_t atom    = GetAtom<Problem, IsA>();
        constexpr index_t repeats = GetRepeats<Problem, IsA>();
        constexpr index_t comps   = warps * atom * repeats / op.rows;
        constexpr index_t steps   = GetKSteps<Problem>();
        constexpr index_t run     = steps * op.rows;
        constexpr index_t block   = op.pad_bytes > 0 ? op.block_bytes / 4 : run;
        static_assert(
            warps * atom * repeats % op.rows == 0 && run % block == 0 && op.stride % 4 == 0 &&
                (op.block_bytes + op.pad_bytes) % 4 == 0,
            "comp_tdm_scheduled: the scale layout is not a whole number of padded blocks");
        constexpr auto naive = make_naive_tensor_descriptor(
            make_tuple(number<comps>{}, number<run / block>{}, number<block>{}),
            make_tuple(number<op.stride / 4>{},
                       number<(op.block_bytes + op.pad_bytes) / 4>{},
                       number<1>{}),
            number<4>{},
            number<1>{});
        constexpr auto linear = transform_tensor_descriptor(
            naive,
            make_tuple(make_pass_through_transform(number<comps>{}),
                       make_merge_transform(make_tuple(number<run / block>{}, number<block>{}))),
            make_tuple(sequence<0>{}, sequence<1, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
        constexpr auto by_step = transform_tensor_descriptor(
            linear,
            make_tuple(make_pass_through_transform(number<comps>{}),
                       make_unmerge_transform(make_tuple(number<steps>{}, number<op.rows>{}))),
            make_tuple(sequence<0>{}, sequence<1>{}),
            make_tuple(sequence<1>{}, sequence<0, 2>{}));
        return transform_tensor_descriptor(
            by_step,
            make_tuple(make_pass_through_transform(number<steps>{}),
                       make_merge_transform(make_tuple(number<comps>{}, number<op.rows>{}))),
            make_tuple(sequence<0>{}, sequence<1, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    /// A's or B's E8M0 scales in LDS as words, top dimensions (K step, wave * atom + lane row,
    /// repeat): MakeLdsScaleTdmBlockDescriptor with the tile row split per repeat.
    template <typename Problem, bool IsA>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsScaleDescriptor()
    {
        constexpr index_t warps   = GetWarps<Problem, IsA>();
        constexpr index_t atom    = GetAtom<Problem, IsA>();
        constexpr index_t repeats = GetRepeats<Problem, IsA>();
        constexpr index_t steps   = GetKSteps<Problem>();
        return transform_tensor_descriptor(
            MakeLdsScaleTdmBlockDescriptor<Problem, IsA>(),
            make_tuple(
                make_pass_through_transform(number<steps>{}),
                make_unmerge_transform(make_tuple(number<warps * atom>{}, number<repeats>{}))),
            make_tuple(sequence<0>{}, sequence<1>{}),
            make_tuple(sequence<0>{}, sequence<1, 2>{}));
    }

    /// Lanes reading distinct scale rows: all 32 for A of the 32x16 atom, else 16 (lanes l and
    /// l + 16 read the same row).
    template <typename Problem, bool IsA>
    CK_TILE_HOST_DEVICE static constexpr index_t GetScaleLanes()
    {
        return IsA && GetAtomM<Problem>() == 32 ? 32 : 16;
    }

    /// One wave's read of A's or B's scales of one K step: lane l reads its row's words of all
    /// repeats.   (K step: 1, rows: lanes, repeats)
    template <typename Problem, bool IsA>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsScaleReadDistribution()
    {
        constexpr index_t repeats = GetRepeats<Problem, IsA>();
        if constexpr(GetScaleLanes<Problem, IsA>() == 32)
            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<>,
                                           tuple<sequence<1>, sequence<32>, sequence<repeats>>,
                                           tuple<sequence<2>>,
                                           tuple<sequence<0>>,
                                           sequence<1, 3>,
                                           sequence<0, 0>>{});
        else
            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<2>,
                                           tuple<sequence<1>, sequence<16>, sequence<repeats>>,
                                           tuple<sequence<0, 2>>,
                                           tuple<sequence<0, 0>>,
                                           sequence<1, 3>,
                                           sequence<0, 0>>{});
    }

    /// Whether a scale read can take its K step as a constant word offset and its repeats as
    /// consecutive words, checked against GetLdsScaleByteOffset. Also spot-checks
    /// MakeLdsScaleDescriptor.
    template <typename Problem, bool IsA>
    CK_TILE_HOST_DEVICE static constexpr bool IsLdsScaleReadLinear()
    {
        constexpr index_t warps   = GetWarps<Problem, IsA>();
        constexpr index_t atom    = GetAtom<Problem, IsA>();
        constexpr index_t repeats = GetRepeats<Problem, IsA>();
        constexpr index_t steps   = GetKSteps<Problem>();
        constexpr auto desc       = MakeLdsScaleDescriptor<Problem, IsA>();
        const auto offset         = [](index_t step, index_t row) {
            constexpr LdsOperand op = GetLdsOperand<Problem, IsA, true>();
            return (GetLdsScaleByteOffset<index_t>(op, step, row) - op.base) / 4;
        };
        const index_t step_words = offset(1, 0);
        for(const index_t t : {index_t{1}, atom - 1, atom, warps * atom - 1})
            for(const index_t step : {index_t{0}, steps - 1})
                if(desc.calculate_offset(make_multi_index(step, t, repeats - 1)) !=
                   offset(step, t * repeats + repeats - 1))
                    return false;
        for(index_t t = 0; t < warps * atom; ++t)
            for(index_t step = 0; step < steps; ++step)
                for(index_t r = 0; r < repeats; ++r)
                    if(offset(step, t * repeats + r) !=
                       offset(0, t * repeats) + step * step_words + r)
                        return false;
        return true;
    }

    /// The cluster workgroups that share this one's A (kM: same M tile) or B (kN: same N tile)
    /// and so receive its TDM loads; 0 without a cluster. As the default TDM policy's, but grid
    /// x walks N tiles and y M tiles.
    template <MultiCastDirection Direction, typename Problem>
    CK_TILE_DEVICE static uint16_t GetTDMWorkgroupMask(dim3 block_id_in_cluster)
    {
        static_assert(Direction != MultiCastDirection::kMN,
                      "comp_tdm_scheduled multicasts A along M and B along N");
        using Shape   = typename Problem::BlockGemmShape;
        uint32_t mask = 0;
        if constexpr(!HasCluster<Problem>())
            ignore = block_id_in_cluster;
        else if constexpr(Direction == MultiCastDirection::kM)
            mask = cluster_mask_same_y(Shape::kclusterN, block_id_in_cluster.y);
        else
            mask = cluster_mask_same_x(Shape::kclusterN, Shape::kclusterM, block_id_in_cluster.x);
        // A cluster has at most 16 workgroups; without this the 16-bit mask costs an s_and.
        __builtin_assume(mask <= 0xFFFFu);
        return static_cast<uint16_t>(mask);
    }

    /// The C block tile: per thread the WMMA C fragments of its M and N repeats, (rm, rn) outer
    /// and the fragment slot inner. A lane's M repeats are adjacent rows and its N repeats
    /// adjacent columns, so one slot across the N repeats is one vector store of D.
    ///   M: (wave_m, slot / 8, lane / 16, slot % 8, rm)   N: (wave_n, lane % 16, rn)
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeCBlockTileDistribution()
    {
        constexpr index_t warp_m   = GetWarps<Problem, true>();
        constexpr index_t warp_n   = GetWarps<Problem, false>();
        constexpr index_t atom_m   = GetAtomM<Problem>();
        constexpr index_t atom_n   = GetAtomN<Problem>();
        constexpr index_t repeat_m = GetWaveRepeatM<Problem>();
        constexpr index_t repeat_n = GetWaveRepeatN<Problem>();
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<warp_m, atom_m / 16, 2, 8, repeat_m>,
                                             sequence<warp_n, atom_n, repeat_n>>,
                                       tuple<sequence<1, 2>, sequence<1, 2>>,
                                       tuple<sequence<0, 0>, sequence<2, 1>>,
                                       sequence<1, 2, 1, 1>,
                                       sequence<4, 2, 1, 3>>{});
    }

    /// The block GEMM: BlockGemmARegBRegCRegV1 with PackMNIter (M and N in the fragment order of
    /// MakeLdsBlockDescriptor) and one K step per sub-tile, reading this policy's LDS layout. The
    /// warp GEMM's operands take one access per 16-byte chunk, as MakeLdsReadDistribution reads
    /// them. The warp GEMM is the gfx1250 WMMA WarpGemmDispatcher picks on the device, built from
    /// its traits because the dispatcher has no gfx1250 16x16x128 WMMA in the host pass (which also
    /// instantiates the pipeline) nor named WMMAs with USE_NEW_UNIFIED_FRAMEWORK.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using Shape               = typename Problem::BlockGemmShape;
        using ADataType           = typename Problem::ADataType;
        using BDataType           = typename Problem::BDataType;
        constexpr auto num_access = static_cast<WGAttrNumAccessEnum>(GetChunksPerLane<Problem>());
        using Wmma                = WmmaTraits<gfx125_t,
                                               ADataType,
                                               BDataType,
                                               float,
                                               GetAtomM<Problem>(),
                                               GetAtomN<Problem>(),
                                               kAtomK>;
        using Attr =
            WarpGemmAttributeWmma<WarpGemmAttributeWmmaImpl<Wmma>, false, num_access, num_access>;
        using WarpGemm = WarpGemmImpl<Attr>;
        using V1Policy = BlockGemmARegBRegCRegV1CustomPolicy<ADataType,
                                                             BDataType,
                                                             float,
                                                             typename Shape::BlockWarps,
                                                             WarpGemm,
                                                             GetKSteps<Problem>(),
                                                             true>;
        return BlockGemmMxAsmemBsmemCRegTDMScheduled<Problem,
                                                     MxGemmPipelineAgBgCrCompTDMScheduledPolicy,
                                                     V1Policy>{};
    }

    /// Whether the kernel is launched with a cluster: (x, y, z) = (kclusterN, kclusterM, 1).
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr bool HasCluster()
    {
        using Shape = typename Problem::BlockGemmShape;
        return !(Shape::kclusterN == 1 && Shape::kclusterM == 1);
    }

    /**
     * @brief The tuning part of a kernel name: the cluster, then every value that is not its
     *        default, each with a leading "_": the out dtype, the wave group, pf<N> (a GL2
     *        prefetch distance other than 4), lp0 (PrefetchLdsRead off), hplrnone (SplitBLdsRead
     *        off), lsi0 (InterleaveLdsRegions off) and mma16 (the 16x16 atom for FP4).
     *
     * @param with_shape_parts  Include the out dtype, wave group and mma16, which a CK kernel name
     *                          already carries in its tile and type fields.
     */
    template <typename Problem>
    CK_TILE_HOST static std::string GetTuningName(bool with_shape_parts = false)
    {
        using Shape          = typename Problem::BlockGemmShape;
        using ODataType      = typename Problem::ODataType;
        constexpr index_t wm = GetWarps<Problem, true>();
        constexpr index_t wn = GetWarps<Problem, false>();
        std::string name =
            "_c" + std::to_string(Shape::kclusterN) + "x" + std::to_string(Shape::kclusterM);
        if(with_shape_parts && !std::is_same_v<ODataType, fp8_t>)
            name += std::is_same_v<ODataType, half_t>   ? "_fp16"
                    : std::is_same_v<ODataType, bf16_t> ? "_bf16"
                                                        : "_unknown";
        if(with_shape_parts && !(wm == 2 && wn == 2))
            name += "_wg" + std::to_string(wm) + "x" + std::to_string(wn);
        if(kGl2PrefetchDistance != 4)
            name += "_pf" + std::to_string(kGl2PrefetchDistance);
        if(!kPrefetchLdsRead)
            name += "_lp0";
        if(!kSplitBLdsRead)
            name += "_hplrnone";
        if(!kInterleaveLdsRegions)
            name += "_lsi0";
        if(with_shape_parts && !IsMma32<Problem>())
            name += "_mma16";
        return name;
    }

    private:
    // The LDS layout. A and B rows are padded by 16 B per bank cycle-aligned block, so the rows a
    // K step reads start in different banks; the scales take 16 B per 256 B when their reads
    // would otherwise conflict; regions start at 2048-bit boundaries; with InterleaveLdsRegions
    // the components are placed by 64 KiB segment (GetInterleavePlacement).
    static constexpr index_t kDataPadBytes    = 16;
    static constexpr index_t kBankCycleBytes  = 256;
    static constexpr index_t kMxsBlockBytes   = 256;
    static constexpr index_t kMxsPadBytes     = 16;
    static constexpr index_t kRegionAlignBits = 2048;
    static constexpr index_t kSegmentBytes    = 1 << 16;
    static constexpr index_t kTdmPadPeriodMax = 1024;
    static constexpr index_t kTdmPadMaxBytes  = 128 * 4;
    static constexpr index_t kLdsPadStepBytes = 8;

    CK_TILE_HOST_DEVICE static constexpr bool is_pow2(long_index_t v)
    {
        return v > 0 && (v & (v - 1)) == 0;
    }
    CK_TILE_HOST_DEVICE static constexpr long_index_t round_up(long_index_t v, long_index_t m)
    {
        return (v + m - 1) / m * m;
    }
    CK_TILE_HOST_DEVICE static constexpr long_index_t ceil_div(long_index_t v, long_index_t d)
    {
        return (v + d - 1) / d;
    }
    /// The smallest power of two at least v (1 for v <= 1).
    CK_TILE_HOST_DEVICE static constexpr long_index_t pow2_ceil(long_index_t v)
    {
        long_index_t p = 1;
        while(p < v)
            p <<= 1;
        return p;
    }
    CK_TILE_HOST_DEVICE static constexpr index_t min_i(index_t a, index_t b)
    {
        return a < b ? a : b;
    }
    CK_TILE_HOST_DEVICE static constexpr index_t max_i(index_t a, index_t b)
    {
        return a > b ? a : b;
    }

    CK_TILE_HOST_DEVICE static constexpr bool IsValidTileMN(index_t t)
    {
        return t == 16 || t == 32 || t == 64 || t == 128 || t == 256;
    }

    /// LDS reads of one K step: scales, then one per K chunk of each A row and B fragment.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetLdsReads()
    {
        const index_t rm = GetWaveRepeatM<Problem>();
        const index_t rn = GetWaveRepeatN<Problem>();
        return max_i(1, rm / 4) + max_i(1, rn / 4) +
               GetChunksPerLane<Problem>() * (GetAtomM<Problem>() / 16 * rm + rn);
    }

    /// pad_bytes of LDS padding after every block_bytes written.
    struct LdsPad
    {
        index_t pad_bytes   = 0;
        index_t block_bytes = 0;

        CK_TILE_HOST_DEVICE constexpr bool on() const { return pad_bytes > 0; }
    };

    /// One operand's place in an LDS buffer (see GetLdsOperand).
    struct LdsOperand
    {
        index_t rows           = 0;
        index_t base           = 0;
        index_t stride         = 0;
        index_t pad_bytes      = 0;
        index_t block_bytes    = 0;
        index_t iterate_rows   = 0;
        index_t iterate_stride = 0;
    };

    /// The two LDS buffers. Buffer 1 starts buffer_stride after buffer 0; buffer_bytes is where
    /// buffer 0's last region ends. reason is not Ok when InterleaveLdsRegions does not apply.
    struct LdsLayout
    {
        MxTdmScheduledReason reason = MxTdmScheduledReason::Ok;
        index_t row_bytes           = 0;
        index_t k_steps             = 0;
        LdsOperand a{};
        LdsOperand b{};
        LdsOperand a_scale{};
        LdsOperand b_scale{};
        index_t buffer_bytes  = 0;
        index_t buffer_stride = 0;
        index_t lds_bytes     = 0;
    };

    /// The bytes after which an A or B component takes kDataPadBytes: the K row of `repeat`
    /// interleaved rows (the operand's wave repeat, see MakeLdsBlockDescriptor), rounded up to a
    /// bank cycle, so the rows a K step reads start in different banks.
    CK_TILE_HOST_DEVICE static constexpr index_t GetDataPadBlock(index_t k_bytes, index_t repeat)
    {
        index_t block = static_cast<index_t>(round_up(k_bytes, kBankCycleBytes));
        if(k_bytes * repeat > kBankCycleBytes)
            block = static_cast<index_t>(round_up(k_bytes * repeat, kBankCycleBytes));
        return block;
    }

    /// The scale padding of an operand with wave repeat `repeat`: a K step of its scales is
    /// 4 * repeat bytes per row group. When that is an odd multiple of 16 B (repeat 4, 12, ...),
    /// 16 B follow every 256 B; else no padding.
    CK_TILE_HOST_DEVICE static constexpr LdsPad GetScalePad(index_t repeat)
    {
        const index_t per_block = (kAtomK / 32) * repeat;
        if(repeat < 4 || per_block % 16 || (per_block / 16) & 1)
            return {};
        return {kMxsPadBytes, kMxsBlockBytes};
    }

    struct DerivedPads
    {
        LdsPad a{};
        LdsPad b{};
        LdsPad a_scale{};
        LdsPad b_scale{};
    };

    /// The LDS padding of A, B and their scales, from the tile's K bytes and the wave repeats.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr DerivedPads GetDerivedLdsPads()
    {
        const index_t k_bytes = Problem::BlockGemmShape::kK / GetKPerByte<Problem>();
        const index_t rm      = GetWaveRepeatM<Problem>();
        const index_t rn      = GetWaveRepeatN<Problem>();
        return {{kDataPadBytes, GetDataPadBlock(k_bytes, rm)},
                {kDataPadBytes, GetDataPadBlock(k_bytes, rn)},
                GetScalePad(rm),
                GetScalePad(rn)};
    }

    struct DataComponent
    {
        index_t bytes          = 0;
        index_t iterate_rows   = 0;
        index_t iterate_stride = 0;
    };

    CK_TILE_HOST_DEVICE static constexpr DataComponent
    GetDataComponent(index_t rows, index_t row_bytes, LdsPad pad)
    {
        if(pad.on() && pad.block_bytes > kTdmPadPeriodMax)
        {
            const index_t iterate_rows = pad.block_bytes / row_bytes;
            const index_t stride       = pad.block_bytes + pad.pad_bytes;
            return {
                static_cast<index_t>(ceil_div(rows, iterate_rows)) * stride, iterate_rows, stride};
        }
        index_t size = rows * row_bytes;
        if(pad.on())
            size += size / pad.block_bytes * pad.pad_bytes;
        return {size, 0, 0};
    }

    /// Bytes of an operand's scale region (both components), padded as one tensor.
    CK_TILE_HOST_DEVICE static constexpr index_t
    GetScaleRegionBytes(index_t rows, index_t steps, LdsPad pad)
    {
        const index_t size = 2 * rows * steps * 4;
        return pad.on() ? size / pad.block_bytes * (pad.block_bytes + pad.pad_bytes) : size;
    }

    /// Bases and component strides of A and B and where the scales start. An aligned layout
    /// reaches span (scales included), past the baseline span.
    struct Placement
    {
        MxTdmScheduledReason reason = MxTdmScheduledReason::Ok;
        index_t a_base              = 0;
        index_t a_stride            = 0;
        index_t b_base              = 0;
        index_t b_stride            = 0;
        index_t scale_base          = 0;
        bool aligned                = false;
        index_t span                = 0;
    };

    /// first at 0 with component stride `stride`, the other after first's component 0 with
    /// `second_stride`.
    CK_TILE_HOST_DEVICE static constexpr Placement Place(bool first_is_b,
                                                         index_t comp_first,
                                                         index_t stride,
                                                         index_t second_stride,
                                                         index_t scale_base,
                                                         bool aligned = false,
                                                         index_t span = 0)
    {
        Placement p{};
        if(first_is_b)
        {
            p.b_base   = 0;
            p.b_stride = stride;
            p.a_base   = comp_first;
            p.a_stride = second_stride;
        }
        else
        {
            p.a_base   = 0;
            p.a_stride = stride;
            p.b_base   = comp_first;
            p.b_stride = second_stride;
        }
        p.scale_base = scale_base;
        p.aligned    = aligned;
        p.span       = span;
        return p;
    }

    /// The placement of InterleaveLdsRegions: the A and B components a K step reads together go
    /// to different 64 KiB LDS segments, so the two segments serve them in parallel. An operand
    /// is split only when its wave repeat covers its wave's rows (InterleaveRepeat otherwise);
    /// `scales` is the aligned size of both scale regions.
    ///
    /// 2x2: A's components go to different segments, as tight [B0][A0][B1][A1], or
    /// segment-aligned when B0 A0 fit one segment. With the 32x16 atom an A read would cross a
    /// component, so A stays whole: [B0][A0 A1][B1], aligned when that fits one segment. 4x1 and
    /// 1x4: only the operand spanning four waves splits, around the other one.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr Placement GetInterleavePlacement(
        index_t comp_a, index_t comp_b, index_t data_a, index_t data_b, index_t scales)
    {
        constexpr index_t seg    = kSegmentBytes;
        constexpr index_t tile_m = Problem::BlockGemmShape::kM;
        constexpr index_t tile_n = Problem::BlockGemmShape::kN;
        const index_t warp_m     = GetWarps<Problem, true>();
        const index_t warp_n     = GetWarps<Problem, false>();
        const index_t lanes      = min_i(GetAtomM<Problem>(), GetAtomN<Problem>());
        const auto coarse        = [&](bool is_b) {
            const index_t tile   = is_b ? tile_n : tile_m;
            const index_t waves  = is_b ? warp_n : warp_m;
            const index_t repeat = is_b ? GetWaveRepeatN<Problem>() : GetWaveRepeatM<Problem>();
            return lanes * repeat >= tile / waves;
        };
        Placement fail{};
        if(warp_m == 1 || warp_n == 1)
        {
            const bool active_b      = warp_n > 1;
            const index_t comp_act   = active_b ? comp_b : comp_a;
            const index_t comp_share = active_b ? comp_a : comp_b;
            const index_t data_act   = active_b ? data_b : data_a;
            if(!coarse(active_b))
            {
                fail.reason = MxTdmScheduledReason::InterleaveRepeat;
                return fail;
            }
            if(active_b && data_b <= seg && comp_b >= seg)
            {
                fail.reason = MxTdmScheduledReason::InterleaveBaselineSplit;
                return fail;
            }
            const index_t stride = comp_act + 2 * comp_share;
            if(stride / seg > (data_act - 1) / seg)
                return Place(active_b, comp_act, stride, comp_share, stride + comp_act);
            const index_t pre  = static_cast<index_t>(round_up(stride, seg));
            const index_t span = pre + comp_act;
            return Place(active_b, comp_act, pre, comp_share, span, true, span + scales);
        }
        if(!coarse(true))
        {
            fail.reason = MxTdmScheduledReason::InterleaveRepeat;
            return fail;
        }
        const index_t a_cols  = GetAtomM<Problem>() * warp_m * GetWaveRepeatM<Problem>();
        const bool a_readable = coarse(false) || (a_cols > 0 && (tile_m / 2) % a_cols == 0);
        if(!a_readable)
        {
            const index_t stride = comp_b + 2 * comp_a;
            if(stride >= seg)
                return Place(true, comp_b, stride, comp_a, stride + comp_b);
            const index_t pre  = static_cast<index_t>(round_up(stride, seg));
            const index_t span = pre + comp_b;
            return Place(true, comp_b, pre, comp_a, span, true, span + scales);
        }
        const index_t pair = comp_a + comp_b;
        if(pair < seg)
        {
            const index_t span = seg + pair;
            return Place(true, comp_b, seg, seg, span, true, span + scales);
        }
        return Place(true, comp_b, pair, pair, 2 * pair);
    }

    /// The LDS layout of a Problem whose tile, types and wave group are valid. Baseline
    /// (InterleaveLdsRegions off) is [B0 B1][B scales][A scales][A0 A1]; InterleaveLdsRegions
    /// places the components so the two LDS read ports read different 64 KiB segments.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr LdsLayout GetLdsLayout()
    {
        const DerivedPads pads  = GetDerivedLdsPads<Problem>();
        const index_t row_bytes = Problem::BlockGemmShape::kK / GetKPerByte<Problem>();
        const index_t steps     = GetKSteps<Problem>();
        const index_t a_rows    = Problem::BlockGemmShape::kM / 2;
        const index_t b_rows    = Problem::BlockGemmShape::kN / 2;
        const DataComponent a   = GetDataComponent(a_rows, row_bytes, pads.a);
        const DataComponent b   = GetDataComponent(b_rows, row_bytes, pads.b);

        const auto scale_component = [&](index_t rows, LdsPad pad) {
            const index_t size = steps * rows * 4;
            return size + (pad.on() ? size / pad.block_bytes * pad.pad_bytes : 0);
        };

        const index_t align = kRegionAlignBits / GetElemBits<typename Problem::ADataType>();
        const index_t b_scale_region =
            static_cast<index_t>(round_up(GetScaleRegionBytes(b_rows, steps, pads.b_scale), align));
        const index_t a_scale_region =
            static_cast<index_t>(round_up(GetScaleRegionBytes(a_rows, steps, pads.a_scale), align));
        const index_t b_end = static_cast<index_t>(round_up(2 * b.bytes, align));
        Placement baseline{};
        baseline.a_base             = b_end + b_scale_region + a_scale_region;
        baseline.a_stride           = a.bytes;
        baseline.b_base             = 0;
        baseline.b_stride           = b.bytes;
        baseline.scale_base         = b_end;
        const index_t baseline_span = baseline.a_base + 2 * a.bytes;
        const index_t natural_stride =
            baseline.a_base + static_cast<index_t>(round_up(2 * a.bytes, align));

        LdsLayout out{};
        Placement placement = baseline;
        if(kInterleaveLdsRegions)
        {
            placement = GetInterleavePlacement<Problem>(a.bytes,
                                                        b.bytes,
                                                        a_rows * row_bytes,
                                                        b_rows * row_bytes,
                                                        b_scale_region + a_scale_region);
            if(placement.reason != MxTdmScheduledReason::Ok)
            {
                out.reason = placement.reason;
                return out;
            }
        }
        index_t buffer_stride = 0;
        index_t lds_bytes     = 0;
        if(placement.aligned)
        {
            buffer_stride = static_cast<index_t>(
                pow2_ceil(natural_stride > placement.span ? natural_stride : placement.span));
            if(2ll * buffer_stride > kLdsCapacityBytes)
            {
                out.reason = MxTdmScheduledReason::InterleaveCapacity;
                return out;
            }
            lds_bytes = max_i(buffer_stride + baseline_span, buffer_stride + placement.span);
        }
        else
        {
            buffer_stride = natural_stride;
            if(natural_stride + pow2_ceil(natural_stride) <= kLdsCapacityBytes)
                buffer_stride = static_cast<index_t>(pow2_ceil(natural_stride));
            lds_bytes = buffer_stride + baseline_span;
        }
        const index_t b_scale_base = placement.scale_base;
        const index_t a_scale_base = b_scale_base + b_scale_region;
        const index_t a_scale_comp = scale_component(a_rows, pads.a_scale);
        const index_t b_scale_comp = scale_component(b_rows, pads.b_scale);

        out.row_bytes = row_bytes;
        out.k_steps   = steps;
        out.a         = {a_rows,
                         placement.a_base,
                         placement.a_stride,
                         pads.a.pad_bytes,
                         pads.a.block_bytes,
                         a.iterate_rows,
                         a.iterate_stride};
        out.b         = {b_rows,
                         placement.b_base,
                         placement.b_stride,
                         pads.b.pad_bytes,
                         pads.b.block_bytes,
                         b.iterate_rows,
                         b.iterate_stride};
        out.a_scale   = {a_rows,
                         a_scale_base,
                         a_scale_comp,
                         pads.a_scale.pad_bytes,
                         pads.a_scale.block_bytes,
                         0,
                         0};
        out.b_scale   = {b_rows,
                         b_scale_base,
                         b_scale_comp,
                         pads.b_scale.pad_bytes,
                         pads.b_scale.block_bytes,
                         0,
                         0};
        out.buffer_bytes =
            max_i(max_i(placement.a_base + placement.a_stride + a.bytes,
                        placement.b_base + placement.b_stride + b.bytes),
                  max_i(a_scale_base + 2 * a_scale_comp, b_scale_base + 2 * b_scale_comp));
        out.buffer_stride = buffer_stride;
        out.lds_bytes     = lds_bytes;
        return out;
    }

    /// LDS byte offset, in one buffer, of row `row` (0..tile rows) of a data operand.
    template <typename T>
    CK_TILE_HOST_DEVICE static constexpr T
    GetLdsRowByteOffset(const LdsOperand& op, index_t row_bytes, T row)
    {
        const T component = row / op.rows;
        const T local     = row % op.rows;
        T linear          = 0;
        if(op.iterate_rows)
        {
            linear =
                local / op.iterate_rows * op.iterate_stride + local % op.iterate_rows * row_bytes;
        }
        else
        {
            linear = local * row_bytes;
            if(op.pad_bytes > 0)
                linear += linear / op.block_bytes * op.pad_bytes;
        }
        return op.base + component * op.stride + linear;
    }

    /// LDS byte offset of the four E8M0 scales of row `row` at K step `step`.
    template <typename T>
    CK_TILE_HOST_DEVICE static constexpr T
    GetLdsScaleByteOffset(const LdsOperand& op, T step, T row)
    {
        T linear = step * (op.rows * 4) + row % op.rows * 4;
        if(op.pad_bytes > 0)
            linear += linear / op.block_bytes * op.pad_bytes;
        return op.base + row / op.rows * op.stride + linear;
    }

    CK_TILE_HOST_DEVICE static constexpr MxTdmScheduledReason
    GetPadReason(LdsPad pad, index_t row_bytes, index_t rows, index_t chunk)
    {
        if(pad.pad_bytes < 0 || pad.block_bytes < 0)
            return MxTdmScheduledReason::LdsPad;
        if(!pad.on())
            return MxTdmScheduledReason::Ok;
        if(pad.pad_bytes % kLdsPadStepBytes || pad.pad_bytes > kTdmPadMaxBytes)
            return MxTdmScheduledReason::LdsPad;
        const index_t block = pad.block_bytes;
        if(block <= kTdmPadPeriodMax)
        {
            if(block < 8 || !is_pow2(block))
                return MxTdmScheduledReason::LdsPad;
        }
        else if(row_bytes == 0)
            return MxTdmScheduledReason::LdsPad;
        else if(block % row_bytes || rows % (block / row_bytes))
            return MxTdmScheduledReason::LdsPad;
        if(block % chunk)
            return MxTdmScheduledReason::LdsPad;
        return MxTdmScheduledReason::Ok;
    }
};

} // namespace ck_tile
