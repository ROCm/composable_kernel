// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp"
#include "ck_tile/ops/fmha/block/block_masking.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_sched_fp8_policy.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_output.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_executor.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_fp8_schedule.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_softmax.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

// Same intrinsic as __builtin_amdgcn_s_buffer_load_f32, which older clang (e.g. ROCm 10.0) lacks.
CK_TILE_DEVICE_EXTERN __attribute__((noconvergent)) float llvm_amdgcn_s_buffer_load_fp32(
    int32x4_t srsrc, index_t soffset, index_t glc_slc) __asm("llvm.amdgcn.s.buffer.load.f32");

// Uses the clang builtin when present, else the intrinsic declaration (e.g. ROCm 10.0).
CK_TILE_DEVICE float amd_s_buffer_load_fp32(int32x4_t rsrc, index_t offset)
{
#if __has_builtin(__builtin_amdgcn_s_buffer_load_f32)
    return __builtin_amdgcn_s_buffer_load_f32(cast_to_amdgpu_buffer_rsrc_t(rsrc), offset, 0);
#else
    return llvm_amdgcn_s_buffer_load_fp32(rsrc, offset, 0);
#endif
}

// Hand-scheduled gfx125 TDM pipeline for FP8 D64/D128/D192/D256 (K64 QK WMMA at D64/D192).
template <typename Problem_,
          typename Policy_        = FmhaTdmSchedFp8PolicyFor<Problem_>,
          bool EnableVirtualSink_ = true>
struct BlockFmhaPipelineQRKSVSTdmSchedFp8
{
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};

    using Problem               = remove_cvref_t<Problem_>;
    using Policy                = remove_cvref_t<Policy_>;
    using QDataType             = remove_cvref_t<typename Problem::QDataType>;
    using KDataType             = remove_cvref_t<typename Problem::KDataType>;
    using VDataType             = remove_cvref_t<typename Problem::VDataType>;
    using SaccDataType          = remove_cvref_t<typename Problem::SaccDataType>;
    using SMPLComputeDataType   = remove_cvref_t<typename Problem::SMPLComputeDataType>;
    using BiasDataType          = remove_cvref_t<typename Problem::BiasDataType>;
    using RandValOutputDataType = remove_cvref_t<typename Problem::RandValOutputDataType>;
    using LSEDataType           = remove_cvref_t<typename Problem::LSEDataType>;
    using PDataType             = remove_cvref_t<typename Problem::PDataType>;
    using OaccDataType          = remove_cvref_t<typename Problem::OaccDataType>;
    using ODataType             = remove_cvref_t<typename Problem::ODataType>;
    using AttentionVariant      = remove_cvref_t<typename Problem::AttentionVariant>;
    using FmhaMask              = remove_cvref_t<typename Problem::FmhaMask>;
    using SplitSoftmax          = typename Policy::ScoreSoftmax;
    using Schedule              = typename Policy::Schedule;
    using Mode                  = FmhaTdmSchedMode<Problem, Policy>;
    static_assert(Mode::kIsLegal, "qr_tdm_fp8_sched schedule geometry is incompatible");

    using BlockFmhaShape             = remove_cvref_t<typename Problem::BlockFmhaShape>;
    using VLayout                    = remove_cvref_t<typename BlockFmhaShape::VLayout>;
    static constexpr bool kQLoadOnce = true; // if q_tile load whole block length (hdim) at once
    static_assert(kQLoadOnce == Policy::QLoadOnce);
    // K TDM tiles always span the whole head dim.
    static constexpr bool kKLoadOnce = true;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kM0           = BlockFmhaShape::kM0;
    static constexpr index_t kN0           = BlockFmhaShape::kN0;
    static constexpr index_t kK0           = BlockFmhaShape::kK0;
    static constexpr index_t kN1           = BlockFmhaShape::kN1;
    static constexpr index_t kK1           = BlockFmhaShape::kK1;
    static constexpr index_t kQKHeaddim    = BlockFmhaShape::kQKHeaddim;
    static constexpr index_t kSubQKHeaddim = BlockFmhaShape::kSubQKHeaddim;
    static constexpr index_t kNWarp        = BlockFmhaShape::Gemm0BlockWarps::at(I1);

    static constexpr bool kIsGroupMode       = Problem::kIsGroupMode;
    static constexpr bool kPadSeqLenQ        = Problem::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK        = Problem::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ       = Problem::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV       = Problem::kPadHeadDimV;
    static constexpr bool kHasLogitsSoftCap  = Problem::kHasLogitsSoftCap;
    static constexpr bool kHasDropout        = Problem::kHasDropout;
    static constexpr auto BiasEnum           = Problem::BiasEnum;
    static constexpr bool kStoreLSE          = Problem::kStoreLSE;
    static constexpr bool kHasSink           = Problem::kHasSink;
    static constexpr bool kEnableVirtualSink = EnableVirtualSink_;
    static constexpr auto QScaleEnum         = Problem::QScaleEnum;
    // Query rows fold the heads of one KV head: row = g * seqlen_q + s.
    static constexpr bool kMergeHeads = Problem::kMergeNumHeadGroupsSeqLenQ;

    static_assert(CK_TILE_FMHA_FWD_FAST_EXP2,
                  "qr_tdm_fp8_sched: every softmax site calls exp2, so log2(e) must be folded "
                  "into scale_s");
    static_assert(BiasEnum == BlockAttentionBiasEnum::NO_BIAS ||
                      BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS,
                  "qr_tdm_fp8_sched implements elementwise bias only, not alibi");
    static_assert(!kHasLogitsSoftCap, "qr_tdm_fp8_sched does not implement logits soft cap");
    static_assert(!kHasDropout, "qr_tdm_fp8_sched does not support dropout");
    static_assert(QScaleEnum == BlockAttentionQuantScaleEnum::NO_SCALE ||
                      QScaleEnum == BlockAttentionQuantScaleEnum::PERTENSOR ||
                      QScaleEnum == BlockAttentionQuantScaleEnum::PERHEAD ||
                      QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE ||
                      QScaleEnum == BlockAttentionQuantScaleEnum::MX,
                  "qr_tdm_fp8_sched: unsupported quantization granularity");
    static_assert(!kMergeHeads || (!kIsGroupMode && BiasEnum == BlockAttentionBiasEnum::NO_BIAS),
                  "merged GQA rows need batch mode and no bias");
    static constexpr bool kQuantized  = QScaleEnum != BlockAttentionQuantScaleEnum::NO_SCALE;
    static constexpr bool kBlockScale = QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE;
    static constexpr bool kMx         = QScaleEnum == BlockAttentionQuantScaleEnum::MX;
    // Merged PERHEAD/BLOCKSCALE rows fold their own head's q descale into the softmax scale.
    static constexpr bool kRowScale =
        kMergeHeads && (QScaleEnum == BlockAttentionQuantScaleEnum::PERHEAD || kBlockScale);
    // Quantized modes cast P to fp8 against this fixed scale and undo it in the epilogue.
    static constexpr float kPScale = 256.0f;
    // Normalize O by the row sum of the fp8 P gemm_1 consumed, so a common rounding gain cancels.
    static constexpr bool kNormalizeQuantizedP = kQuantized && !kMx;
    // Without LSE or sink nothing needs the unquantized row sum, so l sums q(P) directly.
    static constexpr bool kQuantizedRowSum =
        kNormalizeQuantizedP && !kStoreLSE && !kHasSink && !kEnableVirtualSink;

    static constexpr index_t kAlignmentQ    = Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK    = Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV    = Policy::template GetAlignmentV<Problem>();
    static constexpr index_t kAlignmentO    = Policy::template GetAlignmentO<Problem>();
    static constexpr index_t kAlignmentOacc = Policy::template GetAlignmentO<Problem>();
    static constexpr index_t kAlignmentBias =
        kPadSeqLenK ? 1 : Policy::template GetAlignmentBias<Problem>();
    static constexpr index_t kAlignmentRandVal =
        kPadSeqLenK ? 1 : Policy::template GetAlignmentRandVal<Problem>();

    static_assert(Policy::template IsSupportedProblem<Problem>(),
                  "qr_tdm_fp8_sched requires an exact supported FP8 geometry");

    // IsSupportedProblem above pins Geometry == FmhaTdmSchedGeometryFor<Problem>.
    using Geometry = typename Policy::Geometry;
    static_assert(Geometry::kM == kM0 && Geometry::kN == kN0 && Geometry::kDv == kN1);
    // D128/D256 blockscale folds the fp32 K descale into the scale of each QK stage's scores.
    static constexpr bool kKDescaleInSoftmax =
        kBlockScale && Geometry::kQkWmmaK == 128 && BiasEnum == BlockAttentionBiasEnum::NO_BIAS;
    static constexpr bool kKDescaleWithScaleS =
        kBlockScale && Geometry::kQkWmmaK == 128 && BiasEnum != BlockAttentionBiasEnum::NO_BIAS;
    // D256 streams V inside PV and pairs PV rows with V chunks (PvOrdinal).
    static constexpr bool kD256 = Geometry::kHeadDimQK == 256;
    // Segmented LDS layout: odd wavefronts read the K/V halves placed in another LDS segment.
    static constexpr bool kLdsSegSplit = Policy::kLdsSegLayout;
    static_assert(!kLdsSegSplit || kFmhaTdmSchedFp8LdsSegLayout<Problem>,
                  "the segmented LDS layout supports dense D256 problems only");

    // D256 PV row r pairs V chunk r / 2 with P block r % 2; others run fragment r.
    CK_TILE_HOST_DEVICE static constexpr index_t PvOrdinal(index_t row)
    {
        if constexpr(!kD256)
            return row;
        constexpr index_t kNumN = Policy::OutputFragments::kNumN;
        const index_t chunk     = row / 2;
        return ((row % 2) * 2 + chunk % 2) * kNumN + chunk / 2;
    }

    static constexpr const char* name = "qr_tdm_fp8_sched";

    static constexpr bool kUsesUntransposedVKernelPath = true;
    static constexpr bool kUsesTdmAffineDramPath       = true;
    static constexpr bool kUsesFixedSegmentedLdsArena  = true;
    static constexpr bool kUsesGlobalCausalOrder       = true;
    static constexpr index_t kBlockPerCu               = Policy::kBlockPerCu;
    // Merged D192 at two workgroups per CU cannot afford IsOutOfBoundUnsigned's per-row VGPRs.
    static constexpr bool kUnpinnedMaskWidth =
        kMergeHeads && Geometry::kHeadDimQK == 192 && kBlockPerCu > 1;

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
    using QKBlockGemm = remove_cvref_t<decltype(Policy::template GetQKBlockGemm<Problem>())>;
    using PVBlockGemm = remove_cvref_t<decltype(Policy::template GetPVBlockGemm<Problem>())>;
    using ScoreTile   = decltype(QKBlockGemm::MakeCBlockTile());
    static_assert(ScoreTile::get_thread_buffer_size() == SplitSoftmax::Mapping::kThreadBufferSize);
    // Both WarpGemm implementations expose the physical instruction shape here.
    using QkMmaShape = typename QKBlockGemm::WarpGemm::WarpGemmAttribute::Impl;
    using PvMmaShape = typename PVBlockGemm::WarpGemm::WarpGemmAttribute::Impl;
    static_assert(QkMmaShape::kM == Geometry::QkWmma::kM &&
                      QkMmaShape::kN == Geometry::QkWmma::kN &&
                      QkMmaShape::kK == Geometry::QkWmma::kK,
                  "gemm_0 dispatched a different physical WMMA shape");
    static_assert(PvMmaShape::kM == Geometry::PvWmma::kM &&
                      PvMmaShape::kN == Geometry::PvWmma::kN &&
                      PvMmaShape::kK == Geometry::PvWmma::kK,
                  "gemm_1 dispatched a different physical WMMA shape");
#endif

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        static_assert(Policy::template GetSmemSize<Problem>() == Policy::GetLdsArenaSize());
        return Policy::GetLdsArenaSize();
    }

    // O is written by the pipeline through a dedicated LDS staging tile and per-wavefront TDM.
    static constexpr bool kStoresOutputViaTdm = true;
    // 16 B row pad: the 16 rows of one ds_store_b128 then start on distinct bank quads.
    static constexpr index_t kOLdsRowPitch = kN1 + 16 / sizeof(ODataType);
    // Own staging; an epilogue class reusing the K/V LDS would need a K/V drain barrier.
    static constexpr index_t kOLdsBytes = kM0 * kOLdsRowPitch * sizeof(ODataType);
    static constexpr index_t kOBandRows = Geometry::PvWmma::kM;

    // Row of lane 0 of wavefront `wave` in M iteration `band` of a block GEMM C tile (Y0 = M).
    template <typename CDstr>
    CK_TILE_HOST_DEVICE static constexpr index_t GetBandRowOrigin(index_t wave, index_t band)
    {
        constexpr index_t kNDimP = CDstr::get_num_of_dimension_p();
        array<index_t, kNDimP + CDstr::NDimY> ps_ys{};
        ps_ys(0)      = wave;
        ps_ys(kNDimP) = band;
        return make_tensor_adaptor_coordinate(CDstr{}.get_ps_ys_to_xs_adaptor(), ps_ys)
            .get_bottom_index()[number<0>{}];
    }

    // True if wavefront w owns rows [(b * kWaves + w) * BandRows, +BandRows) of M iteration b.
    template <typename CDstr, index_t NumBands, index_t BandRows>
    CK_TILE_HOST_DEVICE static constexpr bool IsWaveInterleavedBands()
    {
        for(index_t b = 0; b < NumBands; ++b)
            for(index_t w = 0; w < Geometry::kWaves; ++w)
                if(GetBandRowOrigin<CDstr>(w, b) != (b * Geometry::kWaves + w) * BandRows)
                    return false;
        return true;
    }

    // o_ptr/rows_left/cols_left are relative to the tile origin (i_m0, i_n1).
    template <typename OAccTile>
    CK_TILE_DEVICE static void StoreOutput(const OAccTile& o_acc_tile,
                                           ODataType* o_ptr,
                                           index_t rows_left,
                                           index_t cols_left,
                                           index_t stride_o,
                                           void* smem_o)
    {
        static_assert(16 % sizeof(ODataType) == 0, "O LDS rows are padded by one 16 B vector");
        static_assert(kM0 == Geometry::kPvMIter * Geometry::kWaves * kOBandRows,
                      "each wavefront must own kPvMIter bands of kOBandRows rows");
        static_assert(IsWaveInterleavedBands<typename OAccTile::StaticTileDistribution,
                                             Geometry::kPvMIter,
                                             kOBandRows>(),
                      "the barrier-free O store needs each wavefront to read back only its rows");

        auto* smem = static_cast<ODataType*>(smem_o);
        auto o_lds = make_tensor_view<address_space_enum::lds>(
            smem,
            make_naive_tensor_descriptor(make_tuple(number<kM0>{}, number<kN1>{}),
                                         make_tuple(number<kOLdsRowPitch>{}, number<1>{}),
                                         number<16 / sizeof(ODataType)>{},
                                         number<1>{}));
        auto o_lds_window = make_tile_window(o_lds,
                                             make_tuple(number<kM0>{}, number<kN1>{}),
                                             {0, 0},
                                             o_acc_tile.get_tile_distribution());
        store_tile(o_lds_window, cast_tile<ODataType>(o_acc_tile));
        // Each TDM store below only reads rows this wavefront wrote, so no workgroup barrier.
        s_wait_dscnt<0>();

        // TDM moves sizeof(ODataType) elements, so o_ptr and stride_o need only element alignment.
        const index_t wave = get_warp_id();
        static_for<0, Geometry::kPvMIter, 1>{}([&](auto band) {
            const index_t row0 =
                (static_cast<index_t>(band) * Geometry::kWaves + wave) * kOBandRows;
            const index_t rows = min(rows_left - row0, kOBandRows);
            if(rows > 0)
            {
                // Box spans the pad; relies on the TDM store dropping columns past tensor_dim0.
                const uint32_t dims[2]    = {static_cast<uint32_t>(cols_left),
                                             static_cast<uint32_t>(rows)};
                const uint64_t strides[2] = {static_cast<uint64_t>(stride_o), 0};
                constexpr uint16_t box[2] = {kOLdsRowPitch, kOBandRows};
                const auto desc           = createTDMDescriptor<ODataType, 2>(
                    o_ptr + static_cast<long_index_t>(row0) * stride_o,
                    smem + row0 * kOLdsRowPitch,
                    dims,
                    strides,
                    box,
                    TDMConfig{});
                amd_tdm_store(desc);
            }
        });
        s_wait_tensorcnt<0>();
    }

    // A scale operand packs kScaleBytes E8M0 bytes; a gemm_1 byte covers kPScaleGranularity K.
    static constexpr index_t kScaleBytes        = 4;
    static constexpr index_t kPScaleGranularity = kK1 / kScaleBytes;
    // K descale is one scalar per QK stage, V descale one E8M0 byte per kPScaleGranularity rows.
    static constexpr index_t kKVScaleAlign = Geometry::kQkSuColumns;
    // K and V descales of one KV tile, one per kKVScaleAlign-wide group; both share the index.
    static constexpr index_t kKVScaleGroups = kN0 / kKVScaleAlign;
    static_assert(!kBlockScale ||
                  (kPScaleGranularity == kKVScaleAlign && kN0 % kKVScaleAlign == 0 &&
                   Geometry::kQkStages == kKVScaleGroups));
    using KVDescales = array<float, kKVScaleGroups>;

    // Neutral scale operand, four E8M0 1.0 bytes; a zeroed one decodes to 2^-127.
    static constexpr int32_t kE8m0One = 0x7f7f7f7f;

    // MX: one E8M0 per 32 elements along each GEMM's K dimension; strides are in scales.
    struct MxDescales
    {
        const e8m0_t* q_ptr = nullptr;
        const e8m0_t* k_ptr = nullptr;
        const e8m0_t* v_ptr = nullptr;
        index_t stride_q    = 0;
        index_t stride_k    = 0;
        index_t stride_v    = 0;
        index_t seqlen_q    = 1;
    };
    static constexpr index_t kMxGranularity = 32;
    // E8M0 bytes of one Q/K scale row, and of one QK WMMA K step (K64 steps: two).
    static constexpr index_t kMxQkScaleBytes   = kQKHeaddim / kMxGranularity;
    static constexpr index_t kMxStepScaleBytes = Geometry::kQkWmmaK / kMxGranularity;
    // One scale dword per QK WMMA K step; D192/D256 keep each step's Q/K scales in its own plane.
    static constexpr index_t kMxQkScaleDwords   = Geometry::kQkKIter;
    static constexpr index_t kMxKScalesPerPlane = Geometry::kQkStages * Geometry::kQkSuNIter;
    static_assert(!kMx || ((kMxStepScaleBytes == 2 || kMxStepScaleBytes == kScaleBytes) &&
                           kMxQkScaleBytes == kMxStepScaleBytes * kMxQkScaleDwords &&
                           kPScaleGranularity == kMxGranularity),
                  "MX needs whole K64/K128 QK steps and a 32-element P/V scale block");
    // K64 steps pad the two scale bytes of their zero K half to 1.0.
    static constexpr int32_t kMxQkPadScale = kMxStepScaleBytes == 2 ? 0x7f7f0000 : 0;
    // Pointer and stride alignment of the Q/K scale rows read by LoadMxRowScale.
    static constexpr index_t kMxQkScaleAlign = min(kMxStepScaleBytes, kScaleBytes);
    // V scale rows each thread copies: two at D256; K rows: kN / kBlockSize (one-wavefront: four).
    static constexpr index_t kMxVRowsPerThread = max(Geometry::kDv / kBlockSize, index_t{1});
    static constexpr index_t kMxKRowsPerThread = Geometry::kN / kBlockSize;

    // Merged GQA rows: query position row % seqlen_q, head row / seqlen_q of num_groups.
    struct MergedHeads
    {
        index_t seqlen_q      = 1;
        index_t num_groups    = 1;
        const float* sink_ptr = nullptr;
        // Per-head q descales of the group, folded as (scale_s * q_descale) * k_descale.
        const float* q_descale_ptr    = nullptr;
        index_t q_descale_head_stride = 0;
        float scale_s                 = 1.0f;
        float k_descale               = 1.0f;
    };

    // Tile row that this lane holds at distributed index idx of a row tensor.
    template <typename Tensor, typename Idx>
    CK_TILE_DEVICE static index_t LaneRowOf(Idx idx)
    {
        constexpr auto dstr  = typename Tensor::StaticTileDistribution{};
        using PartitionIndex = remove_cvref_t<decltype(get_partition_index(dstr))>;
        const auto lane_x    = make_tensor_adaptor_coordinate(
                                dstr.get_ps_ys_to_xs_adaptor(),
                                container_concat(get_partition_index(dstr),
                                                 array<index_t, dstr.get_num_of_dimension_y()>{}))
                                .get_bottom_index();
        return (lane_x + get_x_indices_from_distributed_indices(dstr, idx, PartitionIndex{}))
            .at(number<0>{});
    }

    // Query position of each lane row (row tensor layout); merged rows of one head share one.
    template <typename RowTensor>
    CK_TILE_DEVICE static auto MergedRowPositions(const MergedHeads& merged, index_t row0)
    {
        auto pos =
            make_static_distributed_tensor<index_t>(typename RowTensor::StaticTileDistribution{});
        sweep_tile_span(decltype(pos)::get_distributed_spans()[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            const index_t row    = row0 + LaneRowOf<RowTensor>(i_idx);
            pos(i_idx)           = merged.seqlen_q == 1 ? 0 : row % merged.seqlen_q;
        });
        return pos;
    }

    // Sets tensor elements where pred(row position, col0 + column) holds; positions per lane row.
    template <typename Tensor, typename RowPositions, typename DataType, typename Predicate>
    CK_TILE_DEVICE static void SetTileIfRowPos(
        Tensor& tensor, DataType value, const RowPositions& pos, index_t col0, Predicate pred)
    {
        constexpr auto dstr  = typename Tensor::StaticTileDistribution{};
        constexpr auto spans = Tensor::get_distributed_spans();
        using PartitionIndex = remove_cvref_t<decltype(get_partition_index(dstr))>;
        const auto lane_x    = make_tensor_adaptor_coordinate(
                                dstr.get_ps_ys_to_xs_adaptor(),
                                container_concat(get_partition_index(dstr),
                                                 array<index_t, dstr.get_num_of_dimension_y()>{}))
                                .get_bottom_index();
        sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
            const index_t row = pos[make_tuple(idx0)];
            sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                constexpr auto idx = make_tuple(idx0, idx1);
                const index_t col =
                    col0 + lane_x.at(number<1>{}) +
                    get_x_indices_from_distributed_indices(dstr, idx, PartitionIndex{})
                        .at(number<1>{});
                if(pred(row, col))
                    tensor(idx) = value;
            });
        });
    }

    // Softmax scale of a merged row's head, folded as the kernel folds it for one head.
    CK_TILE_DEVICE static float MergedRowScale(const MergedHeads& merged, index_t row)
    {
        const index_t g       = min(row / merged.seqlen_q, merged.num_groups - 1);
        const float q_descale = merged.q_descale_ptr[g * merged.q_descale_head_stride];
        if constexpr(kBlockScale)
            return merged.scale_s * q_descale;
        else
            return merged.scale_s * q_descale * merged.k_descale;
    }

    // Sink logit of a merged row's head, in the units of the unmerged per-head sink_v * scale_s.
    CK_TILE_DEVICE static float
    MergedSinkLogit(const MergedHeads& merged, index_t row, float scale_s)
    {
        // row is tile-local: merged query heads always fit one M tile, so it is the query row.
        const index_t g = min(row / merged.seqlen_q, merged.num_groups - 1);
        // Rounds like the unmerged path, which passes sink / scale_s and multiplies by scale_s.
        return (merged.sink_ptr[g] / scale_s) * scale_s;
    }

    // Compiler-only memory fence: keeps LDS reads below a barrier that publishes async writes.
    CK_TILE_DEVICE static void CompilerMemoryFence() { asm volatile("" ::: "memory"); }

    // From an SGPR the hardware reads only byte 0 of the scale operand and broadcasts it.
    CK_TILE_DEVICE static int32_t PinToVgpr(int32_t v)
    {
        asm volatile("" : "+v"(v));
        return v;
    }

    // Blocks past the last KV block read 0 from the bounds check; only masked P/S use them.
    CK_TILE_DEVICE static int32x4_t
    MakeKVDescaleResource(const float* ptr, index_t kv_last, index_t block_scale_size_kv)
    {
        const index_t num_blocks = kv_last / block_scale_size_kv + 1;
        return make_wave_buffer_resource(ptr, num_blocks * sizeof(float), std::true_type{});
    }

    CK_TILE_DEVICE static float
    LoadKVDescale(const int32x4_t& res, index_t kv, index_t block_scale_size_kv)
    {
        return amd_s_buffer_load_fp32(
            res, kv / block_scale_size_kv * static_cast<index_t>(sizeof(float)));
    }

    CK_TILE_DEVICE static void LoadKVDescalesAt(const int32x4_t& res,
                                                index_t kv_tile_start,
                                                index_t block_scale_size_kv,
                                                KVDescales& descales)
    {
        static_for<0, kKVScaleGroups, 1>{}([&](auto g) {
            descales(g) =
                LoadKVDescale(res, kv_tile_start + g * kKVScaleAlign, block_scale_size_kv);
        });
    }

    // Walks the KV groups in order, so each descale offset comes without a runtime division.
    struct KVScaleCursor
    {
        index_t offset;           // byte offset of the current group's descale
        index_t group_in_block;   // index of the current group within its descale block
        index_t groups_per_block; // block_scale_size_kv / kKVScaleAlign

        KVScaleCursor() = default;
        CK_TILE_DEVICE KVScaleCursor(index_t kv, index_t block_scale_size_kv)
        {
            const auto group  = static_cast<uint32_t>(kv) / kKVScaleAlign;
            groups_per_block  = block_scale_size_kv / kKVScaleAlign;
            const auto blocks = static_cast<uint32_t>(groups_per_block);
            offset            = static_cast<index_t>(group / blocks * sizeof(float));
            group_in_block    = static_cast<index_t>(group % blocks);
        }

        CK_TILE_DEVICE index_t Next()
        {
            const index_t current = offset;
            const bool wrap       = group_in_block + 1 == groups_per_block;
            group_in_block        = wrap ? 0 : group_in_block + 1;
            offset += wrap ? static_cast<index_t>(sizeof(float)) : 0;
            return current;
        }
    };

    using KVScaleOffsets = array<index_t, kKVScaleGroups>;

    CK_TILE_DEVICE static void AdvanceKVScaleCursor(KVScaleCursor& cursor, KVScaleOffsets& offsets)
    {
        static_for<0, kKVScaleGroups, 1>{}([&](auto g) { offsets(g) = cursor.Next(); });
    }

    CK_TILE_DEVICE static void LoadKVDescalesFromOffsets(const int32x4_t& k_res,
                                                         const int32x4_t& v_res,
                                                         const KVScaleOffsets& offsets,
                                                         KVDescales& k_descales,
                                                         KVDescales& v_descales)
    {
        static_for<0, kKVScaleGroups, 1>{}([&](auto g) {
            k_descales(g) = amd_s_buffer_load_fp32(k_res, offsets[g]);
            v_descales(g) = amd_s_buffer_load_fp32(v_res, offsets[g]);
        });
    }

    template <index_t Stage>
    CK_TILE_DEVICE static int32_t PackVScale(const KVDescales& v_descales, number<Stage>)
    {
        Packed4Scale_E8M0 packed;
        packed.data() = 0;
        static_for<0, kScaleBytes, 1>{}([&](auto i) {
            packed.pack_scale(e8m0_t(v_descales[number<Stage * kScaleBytes + i>{}]), i);
        });
        return PinToVgpr(static_cast<int32_t>(packed.data()));
    }

    // Pin after all loads are issued; a pin per load serializes the round trips.
    template <index_t N>
    CK_TILE_DEVICE static void PinEachToVgpr(array<int32_t, N>& v)
    {
        static_for<0, N, 1>{}([&](auto i) { v(i) = PinToVgpr(v[i]); });
    }

    // set_tile_if with X split into one lane base plus a constant per element; pred may take both.
    template <typename Tensor, typename DataType, typename Predicate>
    CK_TILE_DEVICE static void SetTileIfLaneBase(Tensor& tensor, DataType value, Predicate pred)
    {
        constexpr auto dstr  = typename Tensor::StaticTileDistribution{};
        constexpr auto spans = Tensor::get_distributed_spans();
        using PartitionIndex = remove_cvref_t<decltype(get_partition_index(dstr))>;
        const auto lane_x    = make_tensor_adaptor_coordinate(
                                dstr.get_ps_ys_to_xs_adaptor(),
                                container_concat(get_partition_index(dstr),
                                                 array<index_t, dstr.get_num_of_dimension_y()>{}))
                                .get_bottom_index();
        sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                constexpr auto idx = make_tuple(idx0, idx1);
                if constexpr(std::is_invocable_v<Predicate, decltype(lane_x), decltype(lane_x)>)
                {
                    const auto cx =
                        get_x_indices_from_distributed_indices(dstr, idx, PartitionIndex{});
                    if(pred(lane_x + cx, cx))
                        tensor(idx) = value;
                }
                else
                {
                    const auto x = lane_x + get_x_indices_from_distributed_indices(
                                                dstr, idx, PartitionIndex{});
                    if(pred(x))
                        tensor(idx) = value;
                }
            });
        });
    }

    // (row, col) of the lane's element whose distributed index is zero.
    template <typename Tensor>
    CK_TILE_DEVICE static auto TileLaneOrigin()
    {
        constexpr auto dstr = typename Tensor::StaticTileDistribution{};
        return make_tensor_adaptor_coordinate(
                   dstr.get_ps_ys_to_xs_adaptor(),
                   container_concat(get_partition_index(dstr),
                                    array<index_t, dstr.get_num_of_dimension_y()>{}))
            .get_bottom_index();
    }

    // Sets the elements whose lane-independent (row, col) part passes pred.
    template <typename Tensor, typename DataType, typename Predicate>
    CK_TILE_DEVICE static void SetTileIfConstPart(Tensor& tensor, DataType value, Predicate pred)
    {
        constexpr auto dstr  = typename Tensor::StaticTileDistribution{};
        constexpr auto spans = Tensor::get_distributed_spans();
        using PartitionIndex = remove_cvref_t<decltype(get_partition_index(dstr))>;
        sweep_tile_span(spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(spans[number<1>{}], [&](auto idx1) {
                constexpr auto idx = make_tuple(idx0, idx1);
                const auto x = get_x_indices_from_distributed_indices(dstr, idx, PartitionIndex{});
                if(pred(x.at(number<0>{}), x.at(number<1>{})))
                    tensor(idx) = value;
            });
        });
    }

    // Byte i of a WMMA scale operand covers K elements [32i, 32i + 32) of the lane's row.
    CK_TILE_DEVICE static int32_t LoadMxRowScale(const e8m0_t* ptr, index_t row, index_t stride)
    {
        if constexpr(kMxStepScaleBytes >= kScaleBytes)
            return *reinterpret_cast<const int32_t*>(ptr + row * stride);
        else
            return *reinterpret_cast<const uint16_t*>(ptr + row * stride) | kMxQkPadScale;
    }

    // K64-step K scale dwords hold two copied bytes; replace the upper two by the 1.0 pad scale.
    template <index_t N>
    CK_TILE_DEVICE static void PadMxKScales(array<int32_t, N>& scales)
    {
        if constexpr(kMxStepScaleBytes == 2)
            static_for<0, N, 1>{}(
                [&](auto i) { scales(i) = (scales[i] & 0xffff) | kMxQkPadScale; });
    }

    // MX scales of a KV tile in LDS: kN K dwords per plane, then the V dwords (first kDv used).
    static constexpr index_t kMxKDwords       = Geometry::kN * kMxQkScaleDwords;
    static constexpr index_t kMxLdsTileDwords = kMxKDwords + kBlockSize * kMxVRowsPerThread;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
    // Static LDS of one workgroup: arena, O staging, and MX or row scales.
    static constexpr index_t kStaticLdsBytes =
        Policy::GetLdsArenaSize() + kOLdsBytes +
        (kMx ? 2 * kMxLdsTileDwords * static_cast<index_t>(sizeof(int32_t)) : 0) +
        (kRowScale ? 2 * kBlockSize * static_cast<index_t>(sizeof(float)) : 0);
    static constexpr index_t kLdsWorkgroupsPerCu =
        static_cast<index_t>(get_lds_size()) / kStaticLdsBytes;
    // Four wavefronts: kBlockPerCu workgroups fit. One: LDS caps residency below the VGPR bound.
    static_assert(Geometry::kWaves == 1
                      ? kLdsWorkgroupsPerCu >= 1 && kLdsWorkgroupsPerCu <= 4 * kBlockPerCu
                      : kLdsWorkgroupsPerCu >= kBlockPerCu);
#endif
    CK_TILE_DEVICE static int32_t* MxScaleLds(index_t tile)
    {
        __shared__ int32_t buf[2 * kMxLdsTileDwords];
        return buf + (tile & 1) * kMxLdsTileDwords;
    }

    // Merged row softmax scales live in LDS, one pair per lane, so they hold no VGPRs in the loop.
    using RowScalePair = float __attribute__((ext_vector_type(2)));
    CK_TILE_DEVICE static RowScalePair* RowScaleLds()
    {
        __shared__ RowScalePair buf[kBlockSize];
        return buf + get_thread_id();
    }

    // The fence keeps each read where it is used instead of hoisted out of the loop.
    CK_TILE_DEVICE static RowScalePair LoadRowScales()
    {
        CompilerMemoryFence();
        return *RowScaleLds();
    }

    // core to_global is declared only for gfx125 device passes; this header parses in all passes.
    template <typename T>
    CK_TILE_DEVICE static T* ToGlobalPtr(const void* p)
    {
        return __builtin_bit_cast(T*, const_cast<void*>(p));
    }

    // Per-thread scale copy offsets that stay fixed across KV tiles.
    struct MxCopyOffsets
    {
        index_t k;        // offset of this thread's K scale row from the tile's first row
        index_t v;        // offset of this thread's V scale row
        index_t fast_end; // tiles starting below this are full and dword aligned
    };

    CK_TILE_DEVICE static MxCopyOffsets
    MakeMxCopyOffsets(const MxDescales& mx, index_t kv_load_start, index_t kv_last)
    {
        const index_t t = get_thread_id();
        // Tiles advance by kN0 = kScaleBytes V blocks, so the V dword alignment never changes.
        static_assert(kN0 == kScaleBytes * kMxGranularity);
        const bool aligned =
            kv_load_start % kMxGranularity == 0 && mx.stride_v % kScaleBytes == 0 &&
            (reinterpret_cast<uintptr_t>(mx.v_ptr) + kv_load_start / kMxGranularity) %
                    kScaleBytes ==
                0;
        // Threads past kDv copy the last V row into spare dwords, which keeps the copy branch-free.
        const index_t v_row = Geometry::kDv == kBlockSize ? t : min(t, Geometry::kDv - 1);
        // Pinned so the compiler does not fold them back into per-tile multiplies (D128 only).
        const auto pin = [](index_t v) {
            return kMxQkScaleBytes == kScaleBytes ? PinToVgpr(v) : v;
        };
        return {pin(t * mx.stride_k),
                pin(v_row * mx.stride_v),
                // kv_base < fast_end iff [kv_base, kv_base + kN0) ends by seqlen_k = kv_last + 1.
                aligned ? kv_last + 2 - kN0 : 0};
    }

    // Full tiles copy one K and one V dword per thread; edge tiles clamp K and copy V bytewise.
    template <bool kAllFast = false>
    CK_TILE_DEVICE static void IssueMxScalesToLds(const MxDescales& mx,
                                                  const MxCopyOffsets& off,
                                                  int32_t* lds,
                                                  index_t kv_base,
                                                  index_t kv_last)
    {
        static_assert(Geometry::kN == kBlockSize * kMxKRowsPerThread &&
                      Geometry::kDv <= kBlockSize * kMxVRowsPerThread);
        using gint_t  = __attribute__((address_space(1))) int32_t;
        using gchar_t = __attribute__((address_space(1))) char;
        using lchar_t = __attribute__((address_space(3))) char;
        auto* k_dst   = fmha_tdm_sched_fp8_lds_ptr<int32_t>(fmha_tdm_sched_fp8_lds_offset(lds)) +
                      get_thread_id();
        auto* v_dst = k_dst + kMxKDwords;
        // Unsigned adds keep each lane offset one 32-bit value next to the scalar base.
        const auto lane = [](index_t tile, index_t lane_off) {
            return static_cast<index_t>(static_cast<uint32_t>(tile) +
                                        static_cast<uint32_t>(lane_off));
        };
        // A K64 step has two scale bytes, copied bytewise; PadMxKScales fills the rest.
        const auto copy_k = [&](const e8m0_t* src, auto* dst) {
            if constexpr(kMxQkScaleBytes == kScaleBytes)
                __builtin_amdgcn_global_load_async_to_lds_b32(ToGlobalPtr<gint_t>(src), dst, 0, 0);
            else if constexpr(kMxStepScaleBytes == kScaleBytes)
                static_for<0, kMxQkScaleDwords, 1>{}([&](auto j) {
                    __builtin_amdgcn_global_load_async_to_lds_b32(
                        ToGlobalPtr<gint_t>(src + j * kScaleBytes), dst + j * Geometry::kN, 0, 0);
                });
            else
                static_for<0, kMxQkScaleDwords, 1>{}([&](auto j) {
                    static_for<0, kMxStepScaleBytes, 1>{}([&](auto i) {
                        __builtin_amdgcn_global_load_async_to_lds_b8(
                            ToGlobalPtr<gchar_t>(src + j * kMxStepScaleBytes + i),
                            reinterpret_cast<lchar_t*>(dst + j * Geometry::kN) + i,
                            0,
                            0);
                    });
                });
        };
        const index_t k_off = lane(kv_base * mx.stride_k, off.k);
        // K row r of this thread is tile row t + r * kBlockSize.
        const auto k_row_off = [&](auto r) { return k_off + r * kBlockSize * mx.stride_k; };
        if(kAllFast || __builtin_expect(kv_base < off.fast_end, 1))
        {
            static_for<0, kMxKRowsPerThread, 1>{}(
                [&](auto r) { copy_k(mx.k_ptr + k_row_off(r), k_dst + r * kBlockSize); });
            __builtin_amdgcn_global_load_async_to_lds_b32(
                ToGlobalPtr<gint_t>(mx.v_ptr + lane(kv_base / kMxGranularity, off.v)), v_dst, 0, 0);
            static_for<1, kMxVRowsPerThread, 1>{}([&](auto r) {
                __builtin_amdgcn_global_load_async_to_lds_b32(
                    ToGlobalPtr<gint_t>(mx.v_ptr + lane(kv_base / kMxGranularity,
                                                        off.v + r * kBlockSize * mx.stride_v)),
                    v_dst + r * kBlockSize,
                    0,
                    0);
            });
        }
        else
        {
            static_for<0, kMxKRowsPerThread, 1>{}([&](auto r) {
                copy_k(mx.k_ptr + min(k_row_off(r), kv_last * mx.stride_k), k_dst + r * kBlockSize);
            });
            static_for<0, kMxVRowsPerThread, 1>{}([&](auto r) {
                static_for<0, kScaleBytes, 1>{}([&](auto i) {
                    const index_t blk = min(kv_base + i * kMxGranularity, kv_last) / kMxGranularity;
                    __builtin_amdgcn_global_load_async_to_lds_b8(
                        ToGlobalPtr<gchar_t>(mx.v_ptr +
                                             lane(blk, off.v + r * kBlockSize * mx.stride_v)),
                        reinterpret_cast<lchar_t*>(v_dst + r * kBlockSize) + i,
                        0,
                        0);
                });
            });
        }
    }

    // Lane l reads the scales of WMMA row/column i * 16 + l % 16; p points at its lane's first.
    template <typename LanePtr, index_t N>
    CK_TILE_DEVICE static void ReadMxScalesFromLds(LanePtr p, array<int32_t, N>& scales)
    {
        static_for<0, N, 1>{}([&](auto i) { scales(i) = p[i * Geometry::QkWmma::kN]; });
    }

    // LDS pointer from a byte address computed (and pinned) ahead of the barrier.
    CK_TILE_DEVICE static auto MxScaleLanePtr(int32_t lds_byte)
    {
        return fmha_tdm_sched_fp8_lds_ptr<const int32_t>(static_cast<uint32_t>(lds_byte));
    }

    CK_TILE_DEVICE static int32_t LdsByteAddress(const int32_t* p)
    {
        return static_cast<int32_t>(fmha_tdm_sched_fp8_lds_offset(p));
    }

    // TDM instructions one load_tile_tdm issues per wavefront: one per precomputed coordinate.
    template <typename Window>
    static constexpr index_t kTdmInstsPerLoad = -1;
    template <typename B, typename L, typename D, index_t NumCoord>
    static constexpr index_t
        kTdmInstsPerLoad<tile_window_with_static_distribution<B, L, D, NumCoord>> = NumCoord;

    // Recomputes the clamped row stride in SGPRs at each issue instead of a VGPR v_readfirstlane.
    template <typename LdsWindow, typename DramWindow>
    CK_TILE_DEVICE static void
    LoadTileTdmInLoop(const TDMConfig& config, LdsWindow& lds_window, const DramWindow& dram)
    {
        dram.tdm_load_to_lds(config,
                             lds_window,
                             make_null_tile_window(dram.get_window_lengths()),
                             number<-1>{},
                             bool_constant<false>{},
                             bool_constant<true>{});
    }

    // Unscaled fp8x8 as fp32; a literal scale is uniform, so one convert serves every lane.
    CK_TILE_DEVICE static fp32x8_t Dequantize(const fp8x8_t& q)
    {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
        return __builtin_amdgcn_cvt_scale_pk8_f32_fp8(bit_cast<uint32x2_t>(q), 0x7f7f7f7fu, 0);
#else
        return fp8x8_to_fp32x8(q, 1.0f);
#endif
    }

    // Rescales row_sum like l and adds the lane rows' sum of the fp8 P that gemm_1 consumes.
    template <typename Gemm1, typename PTile, typename RowTile>
    CK_TILE_DEVICE static void
    AccumulateQuantizedMass(const PTile& p_tile, RowTile& row_sum, float scale_m0, float scale_m1)
    {
        constexpr index_t kVec  = 8;
        constexpr index_t kPMI  = Geometry::kPvMIter;
        constexpr index_t kBuf  = PTile::get_thread_buffer_size();
        constexpr index_t kPKI  = kN0 / Gemm1::WarpGemm::kK;
        constexpr index_t kPSub = kBuf / (kPMI * kPKI);
        static_assert(kPMI == 2 && kPSub % kVec == 0);
        array<fp32x8_t, 4> acc{};
        static_for<0, kBuf / kVec, 1>{}([&](auto i) {
            constexpr index_t row = (decltype(i)::value * kVec / kPSub) % kPMI;
            constexpr index_t a   = row * 2 + decltype(i)::value % 2;
            acc(number<a>{}) += Dequantize(p_tile.get_thread_buffer().template get_as<fp8x8_t>(i));
        });
        auto& lq = row_sum.get_thread_buffer();
        static_for<0, 2, 1>{}([&](auto row) {
            const fp32x8_t v = acc[number<row * 2>{}] + acc[number<row * 2 + 1>{}];
            float s          = ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]));
            s += SplitSoftmax::PermuteLaneX16(s);
            lq(row) = __builtin_fmaf(decltype(row)::value == 0 ? scale_m0 : scale_m1, lq[row], s);
        });
    }

    // Re-pack gemm_0 C (M-outer) into gemm_1 A (K-outer) in-thread; lanes already align.
    template <typename Gemm1, typename PComputeTensor>
    CK_TILE_DEVICE static auto MakePForGemm1(const PComputeTensor& p_compute)
    {
        static_assert(Geometry::PRepackLayout::kInThread &&
                          Gemm1::MIterPerWarp == Geometry::kPvMIter,
                      "P re-pack requires gemm_0 C^T and gemm_1 A to share lanes");
        auto p_tile = make_static_distributed_tensor<PDataType>(
            Policy::template MakePRegTileDistribution<Problem>());
        const auto p_src = [&] {
            if constexpr(kQuantized)
            {
                // The scaled convert divides by its power-of-two scale, so this stores 256 * P.
                auto t =
                    make_static_distributed_tensor<PDataType>(p_compute.get_tile_distribution());
                constexpr index_t kVec = 8;
                static_assert(decltype(t)::get_thread_buffer_size() % kVec == 0);
                static_for<0, decltype(t)::get_thread_buffer_size() / kVec, 1>{}([&](auto i) {
                    fp32x8_t v;
                    static_for<0, kVec, 1>{}([&](auto e) {
                        v[e()] = p_compute.get_thread_buffer()[number<i * kVec + e>{}];
                    });
                    t.get_thread_buffer().template set_as<fp8x8_t>(
                        i, fp32x8_to_fp8x8(v, 1.0f / kPScale));
                });
                return t;
            }
            else // NO_SCALE skips kPScale, so small P lose precision to e4m3 subnormals.
                return cast_tile<PDataType>(p_compute);
        }();
        constexpr index_t kPBuf = decltype(p_src)::get_thread_buffer_size();
        constexpr index_t kPMI  = Gemm1::MIterPerWarp;
        constexpr index_t kPKI  = kN0 / Gemm1::WarpGemm::kK;
        constexpr index_t kPSub = kPBuf / (kPMI * kPKI);
        using p_bulk_t          = array<PDataType, kPSub>;
        static_for<0, kPMI, 1>{}([&](auto mi) {
            static_for<0, kPKI, 1>{}([&](auto ki) {
                p_tile.get_thread_buffer().template set_as<p_bulk_t>(
                    number<ki * kPMI + mi>{},
                    p_src.get_thread_buffer().template get_as<p_bulk_t>(number<mi * kPKI + ki>{}));
            });
        });
        return p_tile;
    }

    // Q, K and V reach the fixed LDS arena by TDM; K and V are double buffered.
    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename LSEaccDramBlockWindowTmp,
              typename PositionEncoding>
    CK_TILE_HOST_DEVICE auto
    run(const QDramBlockWindowTmp& __restrict__ q_dram_block_window_tmp,       // M0*K0 tile
        const KDramBlockWindowTmp& __restrict__ k_dram_block_window_tmp,       // N0*K0 tile
        const VDramBlockWindowTmp& __restrict__ v_dram_block_window_tmp,       // N1*K1 tile
        const BiasDramBlockWindowTmp& __restrict__ bias_dram_block_window_tmp, // M0*N0 tile
        LSEaccDramBlockWindowTmp& __restrict__ lse_acc_dram_window_tmp,        // M0*1 tile
        FmhaMask mask,
        [[maybe_unused]] PositionEncoding position_encoding,
        float scale_s,
        void* __restrict__ smem_ptrk0,
        void* __restrict__ smem_ptrk1,
        void* __restrict__ smem_ptrv0,
        void* __restrict__ smem_ptrv1,
        float sink_v,
        [[maybe_unused]] const float* k_descale_ptr,
        [[maybe_unused]] const float* v_descale_ptr,
        [[maybe_unused]] index_t block_scale_size_kv,
        [[maybe_unused]] float v_descale,
        [[maybe_unused]] const MxDescales& mx,
        [[maybe_unused]] const MergedHeads& merged) const
    {
        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

        // Q, K, and V use TDM global-to-LDS transfers; V is read with transposed LDS loads.
        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kSubQKHeaddim == QDramBlockWindowTmp{}.get_window_lengths()[I1] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kK0 == KDramBlockWindowTmp{}.get_window_lengths()[I1] &&
                          kN1 == VDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kK1 == VDramBlockWindowTmp{}.get_window_lengths()[I1] &&
                          kM0 == BiasDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kN0 == BiasDramBlockWindowTmp{}.get_window_lengths()[I1],
                      "wrong!");
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetPVBlockGemm<Problem>();

        using SaccBlockTileType = decltype(gemm_0.MakeCBlockTile());
        auto s_acc              = SaccBlockTileType{};
        // Band masks need lane-linear rows; masked merged tiles are not, so they check per element.
        constexpr bool kBandMask =
            std::is_same_v<FmhaMask, SimplifiedGenericAttentionMask<FmhaMask::IsMasking>> &&
            !Policy::template UseCountdownLoop<Problem>() && !(kMergeHeads && FmhaMask::IsMasking);
        [[maybe_unused]] const auto s_lane_origin = [] {
            if constexpr(kBandMask)
                return TileLaneOrigin<SaccBlockTileType>();
            else
                return 0;
        }();

        const auto f_max        = [](auto e0, auto e1) { return max(e0, e1); };
        using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());

        using PvBlockGemm   = remove_cvref_t<decltype(gemm_1)>;
        using PvWarpGemm    = typename PvBlockGemm::WarpGemm;
        using PvCWarpTensor = typename PvWarpGemm::CWarpTensor;
        static_assert(PvBlockGemm::KIterPerWarp == 1 && PvBlockGemm::MIterPerWarp == 2 &&
                      PvBlockGemm::NIterPerWarp == Geometry::kPvNIter);
        static_assert(std::is_same_v<OaccDataType, float>);
        static_assert(std::is_same_v<remove_cvref_t<typename PvCWarpTensor::DataType>, float>);
        static_assert(std::is_same_v<typename Policy::OutputFragments::Fragment, fp32x8_t>);
        static_assert(PvCWarpTensor::get_thread_buffer_size() ==
                      Policy::OutputFragments::kElementsPerFragment);
        static_assert(Policy::OutputFragments::kNumDmsb == 2 * PvBlockGemm::MIterPerWarp);
        static_assert(Policy::OutputFragments::kNumN == PvBlockGemm::NIterPerWarp / 2);
        static_assert(OaccBlockTileType::get_thread_buffer_size() ==
                      Policy::OutputFragments::Mapping::kThreadBufferSize);
        if constexpr(Geometry::kN == Geometry::kDv)
            static_assert(std::is_same_v<typename SaccBlockTileType::StaticTileDistribution,
                                         typename OaccBlockTileType::StaticTileDistribution>,
                          "equal-width score/output distributions must preserve the fragment "
                          "reconstruction map");

        auto o_acc = Policy::OutputFragments::MakeZero();

        using SBlockTileType = decltype(cast_tile<SMPLComputeDataType>(OaccBlockTileType{}));

        using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
            SBlockTileType{}, sequence<1>{}, f_max, SMPLComputeDataType{0}));

        auto m = MLBlockTileType{};
        auto l = MLBlockTileType{};
        // Row sum of kPScale * q(P) beside the exact l, for instances that also need the exact l.
        [[maybe_unused]] auto l_quant = MLBlockTileType{};
        if constexpr(kNormalizeQuantizedP)
            clear_tile(l_quant);

        // A virtual sink has V=0; merge it only at normalization so it cannot shift P.
        set_tile(m, -numeric<SMPLComputeDataType>::infinity());
        clear_tile(l);

        const auto q_origin = q_dram_block_window_tmp.get_window_origin();

        // Softmax scale of each lane row: scale_s, or the row head's own for merged descales.
        [[maybe_unused]] auto row_scale =
            make_static_distributed_tensor<SMPLComputeDataType>(m.get_tile_distribution());
        if constexpr(kRowScale)
        {
            sweep_tile_span(decltype(row_scale)::get_distributed_spans()[I0], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                row_scale(i_idx) =
                    MergedRowScale(merged, q_origin.at(I0) + LaneRowOf<decltype(m)>(i_idx));
            });
            *RowScaleLds() =
                RowScalePair{row_scale.get_thread_buffer()[I0], row_scale.get_thread_buffer()[I1]};
        }
        const auto scale_of = [&](auto i_idx) {
            if constexpr(kRowScale)
                return row_scale[i_idx];
            else
                return scale_s;
        };

        const auto tile_range_result = [&]() {
            if constexpr(kHasSink)
                return mask.GetSinkTileRangeAlongX(q_origin.at(I0), number<kM0>{}, number<kN0>{});
            else
            {
                auto [start, end] =
                    mask.GetTileRangeAlongX(q_origin.at(I0), number<kM0>{}, number<kN0>{});
                return ck_tile::make_tuple(0, start, end);
            }
        }();
        const auto sink_seq_end           = tile_range_result.get(ck_tile::number<0>{});
        const auto logical_seqlen_k_start = tile_range_result.get(ck_tile::number<1>{});
        const auto logical_seqlen_k_end   = tile_range_result.get(ck_tile::number<2>{});

        const auto num_sink_loop = integer_divide_ceil(sink_seq_end, kN0);

        // No KV tile to visit: write the empty-row LSE and a zero O.
        if(const index_t logical_num_total_loop =
               integer_divide_ceil(logical_seqlen_k_end - logical_seqlen_k_start, kN0) +
               num_sink_loop;
           logical_num_total_loop <= 0)
        {
            if constexpr(kStoreLSE)
            {
                auto lse_acc =
                    make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

                if(__builtin_isinf_sign(sink_v) >= 0)
                {
                    if constexpr(kMergeHeads)
                        sweep_tile_span(
                            decltype(lse_acc)::get_distributed_spans()[I0], [&](auto idx0) {
                                constexpr auto i_idx = make_tuple(idx0);
                                lse_acc(i_idx)       = MergedSinkLogit(
                                    merged, LaneRowOf<decltype(m)>(i_idx), scale_of(i_idx));
                            });
                    else
                        set_tile(lse_acc, SMPLComputeDataType{sink_v * scale_s});
                }
                else
                {
                    set_tile(lse_acc, -numeric<SMPLComputeDataType>::infinity());
                }

                store_tile(lse_acc_dram_window_tmp, lse_acc);
            }

            // O is still zero here and no TDM transfer has been issued yet.
            return Policy::OutputFragments::template Reconstruct<OaccBlockTileType>(o_acc);
        }

        // TDM configs for Q/K/V; LDS padding comes from the policy, no cluster multicast.
        TDMConfig tdm_config_q;
        TDMConfig tdm_config_k;
        TDMConfig tdm_config_v;
        {
            constexpr auto LdsPaddingConfigQ     = Policy::template GetLdsPaddingConfigQ<Problem>();
            tdm_config_q.pad_enable              = LdsPaddingConfigQ[I0];
            tdm_config_q.pad_config.pad_amount   = LdsPaddingConfigQ[I1];
            tdm_config_q.pad_config.pad_interval = LdsPaddingConfigQ[number<2>{}];

            constexpr auto LdsPaddingConfigK     = Policy::template GetLdsPaddingConfigK<Problem>();
            tdm_config_k.pad_enable              = LdsPaddingConfigK[I0];
            tdm_config_k.pad_config.pad_amount   = LdsPaddingConfigK[I1];
            tdm_config_k.pad_config.pad_interval = LdsPaddingConfigK[number<2>{}];

            constexpr auto LdsPaddingConfigV     = Policy::template GetLdsPaddingConfigV<Problem>();
            tdm_config_v.pad_enable              = LdsPaddingConfigV[I0];
            tdm_config_v.pad_config.pad_amount   = LdsPaddingConfigV[I1];
            tdm_config_v.pad_config.pad_interval = LdsPaddingConfigV[number<2>{}];
        }

        // Q tile in LDS: staged in V1, which the iteration-0 V prefetch overwrites first.
        auto q_dram_window = make_tile_window(
            q_dram_block_window_tmp, Policy::template MakeQDramTileDistribution<Problem>());

        auto q_lds_view = make_tensor_view<address_space_enum::lds>(
            static_cast<QDataType*>(smem_ptrv1),
            Policy::template MakeQLdsBlockDescriptor<Problem>());

        auto q_lds_store_window = make_tile_window(
            q_lds_view, Policy::template MakeQLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        auto q_lds_read_window =
            make_tile_window(q_lds_view,
                             Policy::template MakeQLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0},
                             Policy::template MakeQRegTileDistribution<Problem>());

        // Lane l carries the scales of WMMA row/column l % 16 (both K halves agree).
        [[maybe_unused]] const index_t mx_lane = get_lane_id() % Geometry::QkWmma::kM;

        // K tile in LDS (sink-aware start)
        const auto kv_load_start =
            (sink_seq_end == 0 && logical_seqlen_k_start > 0) ? logical_seqlen_k_start : 0;

        // Bias tile window
        const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();
        auto bias_dram_window =
            make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                             bias_dram_block_window_tmp.get_window_lengths(),
                             {bias_origin.at(number<0>{}), kv_load_start},
                             gemm_0.MakeCBlockTile().get_tile_distribution());
        // The next KV tile's bias is in flight while the current tile is scored.
        [[maybe_unused]] auto bias_next = [&] {
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                return load_tile(bias_dram_window, number<-1>{}, bool_constant<false>{});
            else
                return 0;
        }();

        auto k_dram_window =
            make_tile_window(k_dram_block_window_tmp,
                             {kv_load_start, 0},
                             Policy::template MakeKDramTileDistribution<Problem, true>());

        auto k_lds_view = make_tensor_view<address_space_enum::lds>(
            static_cast<KDataType* __restrict__>(smem_ptrk0),
            Policy::template MakeKLdsBlockDescriptor<Problem, true>());

        auto k_lds_write_window = make_tile_window(
            k_lds_view,
            Policy::template MakeKLdsBlockDescriptor<Problem, true>().get_lengths(),
            {0, 0});

        // Read window of one QK stage's K group over the K buffer at k_lds_ptr.
        const auto make_k_su_read_window = [&](KDataType* __restrict__ k_lds_ptr) {
            auto window =
                make_tile_window(k_lds_view,
                                 make_tuple(number<Geometry::kQkSuColumns>{}, number<kQKHeaddim>{}),
                                 {0, 0},
                                 Policy::template MakeKSuRegTileDistribution<Problem>());
            window.set_bottom_tensor_view_data_ptr(k_lds_ptr);
            return window;
        };

        // V tile in LDS (sink-aware start)
        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp,
                             {kv_load_start, 0},
                             Policy::template MakeVDramTileDistribution<Problem>());

        auto v_lds_view = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<VDataType* __restrict__>(static_cast<char*>(smem_ptrv0)),
            Policy::template MakeVLdsBlockDescriptor<Problem>());

        auto v_lds_write_window = make_tile_window(
            v_lds_view, Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        auto v_lds_read_window =
            make_tile_window(v_lds_view,
                             make_tuple(number<kK1>{}, number<kN1>{}),
                             {0, 0},
                             Policy::template MakeVRegTileDistribution<Problem>());

        const index_t num_total_loop =
            integer_divide_ceil(logical_seqlen_k_end - logical_seqlen_k_start, kN0) + num_sink_loop;
        // The tile range rounds its end up to kN0, so bound the scale index by seqlen_k.
        [[maybe_unused]] const index_t kv_last = mask.GetXTotal() - 1;

        constexpr bool kUseCountdownLoop = Policy::template UseCountdownLoop<Problem>();
        bool even_buffer                 = true;
        index_t i_total_loops            = 0;
        index_t remaining_loops          = num_total_loop;
        constexpr index_t k1_loops       = kN0 / kK1;

        // Only D256 MX folds each QK stage into a lane row max as it lands; others use one tree.
        constexpr bool kProgressiveMax = kD256 && kUseCountdownLoop && kMx;

        static_assert(1 <= k1_loops);
        constexpr index_t kKHalfBytes = Geometry::kN / 2 * Policy::kKPhysicalStride;
        constexpr index_t kVHalfBytes = Geometry::kN / 2 * Policy::kVPhysicalStride;
        // Per-wavefront segment byte offsets; only the segmented layout reads them.
        [[maybe_unused]] index_t seg_write = 0;
        [[maybe_unused]] index_t seg_k_lo  = 0;
        [[maybe_unused]] index_t seg_k_hi  = 0;
        [[maybe_unused]] index_t seg_v_lo  = 0;
        [[maybe_unused]] index_t seg_v_hi  = 0;
        [[maybe_unused]] index_t seg_col   = 0;
        [[maybe_unused]] const auto k_seg  = [](void* p, index_t d) {
            return reinterpret_cast<KDataType* __restrict__>(static_cast<char*>(p) + d);
        };
        [[maybe_unused]] const auto v_seg = [](void* p, index_t d) {
            return reinterpret_cast<VDataType* __restrict__>(static_cast<char*>(p) + d);
        };
        if constexpr(kLdsSegSplit)
        {
            constexpr index_t kSegShift = Policy::kLdsSegShift;
            const index_t seg_wave      = get_warp_id();
            const bool seg_odd          = seg_wave & 1;
            seg_write                   = seg_wave >= 2 ? kSegShift : 0;
            seg_k_lo                    = seg_odd ? kKHalfBytes + kSegShift : 0;
            seg_k_hi                    = seg_odd ? -kKHalfBytes : kSegShift;
            seg_v_lo                    = seg_odd ? kVHalfBytes + kSegShift : 0;
            seg_v_hi                    = seg_odd ? -kVHalfBytes : kSegShift;
            seg_col                     = seg_odd ? kN0 / 2 : 0;
            k_lds_write_window.set_bottom_tensor_view_data_ptr(k_seg(smem_ptrk0, seg_write));
            v_lds_write_window.set_bottom_tensor_view_data_ptr(v_seg(smem_ptrv0, seg_write));
        }
        // Fixed TDM prologue, issued back to back: Q, current K/V, and the look-ahead K1.
        load_tile_tdm(tdm_config_q, q_lds_store_window, q_dram_window);
        load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);
        load_tile_tdm(tdm_config_v, v_lds_write_window, v_dram_window);
        move_tile_window(k_dram_window, {kN0, 0});
        // K1 is consumed at iteration 1. A one-tile prefix must jump before K1 loads.
        if constexpr(kHasSink)
        {
            if(num_sink_loop == 1 && num_total_loop > 1)
                move_tile_window(k_dram_window, {logical_seqlen_k_start - sink_seq_end, 0});
        }
        if constexpr(kLdsSegSplit)
            k_lds_write_window.set_bottom_tensor_view_data_ptr(k_seg(smem_ptrk1, seg_write));
        else
            k_lds_write_window.set_bottom_tensor_view_data_ptr(
                static_cast<KDataType* __restrict__>(smem_ptrk1));
        load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);

        // Reads still in flight at the K and V buffer-reuse barriers.
        constexpr index_t k_lds_insts = kFmhaTdmSchedFp8DsWait<Geometry::kKSuLoadCount>;
        constexpr index_t v_lds_insts = kFmhaTdmSchedFp8DsWait<Geometry::kVStageLoadCount>;
        static_assert(v_lds_read_window.get_num_of_access() == Geometry::kVStageLoadCount);

        // The counts below assume one TDM instruction per wavefront per load_tile_tdm.
        static_assert(kTdmInstsPerLoad<decltype(q_dram_window)> == 1 &&
                          kTdmInstsPerLoad<decltype(k_dram_window)> == 1 &&
                          kTdmInstsPerLoad<decltype(v_dram_window)> == 1,
                      "prologue tensorcnt counts assume a single TDM coordinate per window");
        // V0 is waited in Begin, which the sequential schedule runs before Qk reads it.
        static_assert(
            std::is_same_v<Schedule,
                           FmhaTdmSchedSequentialTileSchedule<typename Schedule::LocalPlacement>>,
            "the iteration-0 V0 wait in Begin assumes the sequential tile schedule");
        // Prologue TDM transfers issued after Q (K0, V0, K1) and after K0 (V0, K1).
        constexpr index_t kTdmIssuedAfterQ  = 3;
        constexpr index_t kTdmIssuedAfterK0 = 2;
        // Begin of iteration 0 sees V0, K1 in flight; later Begins see at most the next K.
        constexpr index_t kTdmIssuedAfterV0 = 1;

        [[maybe_unused]] array<int32_t, Geometry::kQkMIter * kMxQkScaleDwords> q_scales;
        if constexpr(kMx)
        {
            static_assert(Geometry::QkWmma::kM == Geometry::QkWmma::kN &&
                          Geometry::PvWmma::kN == Geometry::QkWmma::kN);
            static_assert(IsWaveInterleavedBands<typename SaccBlockTileType::StaticTileDistribution,
                                                 Geometry::kQkMIter,
                                                 Geometry::QkWmma::kM>(),
                          "MX Q scale rows assume wave-interleaved score row bands");
            static_for<0, Geometry::kQkMIter, 1>{}([&](auto m_iter) {
                const index_t row =
                    q_origin.at(I0) +
                    (m_iter * Geometry::kWaves + get_warp_id()) * Geometry::QkWmma::kM + mx_lane;
                static_for<0, kMxQkScaleDwords, 1>{}([&](auto j) {
                    q_scales(number<j * Geometry::kQkMIter + m_iter>{}) = LoadMxRowScale(
                        mx.q_ptr + j * kMxStepScaleBytes, min(row, mx.seqlen_q - 1), mx.stride_q);
                });
            });
        }
        // MX scales of tile j + 1 land in LDS during iteration j; K is read one iteration early.
        [[maybe_unused]] array<int32_t, kMxKScalesPerPlane * kMxQkScaleDwords> k_scales;
        [[maybe_unused]] array<int32_t, Geometry::kPvNIter> v_scales;
        [[maybe_unused]] MxCopyOffsets mx_copy;
        if constexpr(kMx)
        {
            mx_copy = MakeMxCopyOffsets(mx, kv_load_start, kv_last);
            IssueMxScalesToLds(mx, mx_copy, MxScaleLds(0), kv_load_start, kv_last);
            PinEachToVgpr(q_scales);
        }
        // MX scale loads above do not touch Q's LDS, so they overlap the Q transfer.
        s_wait_tensorcnt_barrier<kTdmIssuedAfterQ>();
        auto q_tile = load_tile(q_lds_read_window);

        [[maybe_unused]] int32x4_t k_descale_res;
        [[maybe_unused]] int32x4_t v_descale_res;
        // Descales are loaded one iteration ahead; kv_scale_origin is the tile they belong to.
        [[maybe_unused]] KVDescales k_descales;
        [[maybe_unused]] KVDescales v_descales;
        [[maybe_unused]] index_t kv_scale_origin = kv_load_start;
        // Without sink tokens the tiles are contiguous, so a cursor replaces the division.
        [[maybe_unused]] KVScaleCursor kv_scale_cursor;
        if constexpr(kBlockScale && !kHasSink)
            kv_scale_cursor = KVScaleCursor(kv_load_start, block_scale_size_kv);
        [[maybe_unused]] KVScaleOffsets kv_scale_offsets;
        if constexpr(kBlockScale)
        {
            k_descale_res = MakeKVDescaleResource(k_descale_ptr, kv_last, block_scale_size_kv);
            v_descale_res = MakeKVDescaleResource(v_descale_ptr, kv_last, block_scale_size_kv);
            if constexpr(kHasSink)
            {
                LoadKVDescalesAt(k_descale_res, kv_scale_origin, block_scale_size_kv, k_descales);
                LoadKVDescalesAt(v_descale_res, kv_scale_origin, block_scale_size_kv, v_descales);
            }
            else
            {
                AdvanceKVScaleCursor(kv_scale_cursor, kv_scale_offsets);
                LoadKVDescalesFromOffsets(
                    k_descale_res, v_descale_res, kv_scale_offsets, k_descales, v_descales);
            }
        }

        // dscnt 0 + barrier retires every wavefront's Q reads before the iteration-0 V1 write.
        __builtin_amdgcn_sched_barrier(0);
        if constexpr(kMx)
            s_wait_asynccnt<0>();
        s_wait_tensorcnt_barrier<kTdmIssuedAfterK0, 0>();
        if constexpr(kMx)
        {
            CompilerMemoryFence();
            ReadMxScalesFromLds(MxScaleLds(0) + mx_lane, k_scales);
            PadMxKScales(k_scales);
        }
        __builtin_amdgcn_sched_barrier(0);
        auto initial_k_lds_su_read_window =
            make_tile_window(k_lds_view,
                             make_tuple(number<Geometry::kQkSuColumns>{}, number<kQKHeaddim>{}),
                             {0, 0},
                             Policy::template MakeKSuRegTileDistribution<Problem>());
        if constexpr(kLdsSegSplit)
            initial_k_lds_su_read_window.set_bottom_tensor_view_data_ptr(
                k_seg(smem_ptrk0, seg_k_lo));
        else
            initial_k_lds_su_read_window.set_bottom_tensor_view_data_ptr(
                static_cast<KDataType* __restrict__>(smem_ptrk0));
        auto k_su0_tile = load_tile(initial_k_lds_su_read_window);

        __builtin_amdgcn_sched_barrier(0);

        auto mainloop = [&](KDataType* __restrict__ k_lds_cur_ptr,
                            KDataType* __restrict__ k_lds_next_ptr,
                            VDataType* __restrict__ v_lds_next_ptr,
                            VDataType* __restrict__ v_lds_cur_ptr,
                            auto mx_all_fast) {
            decltype(load_tile_transpose(v_lds_read_window)) v_tile;
            [[maybe_unused]] array<float, 2> qk_lane_max;
            [[maybe_unused]] float pv_output_scale_m0 = 1.0f;
            [[maybe_unused]] float pv_output_scale_m1 = 1.0f;
            [[maybe_unused]] int32_t v_scale          = kE8m0One;
            [[maybe_unused]] index_t kv_tile_start    = 0;
            [[maybe_unused]] const index_t mx_loop =
                kUseCountdownLoop ? num_total_loop - remaining_loops : i_total_loops;
            [[maybe_unused]] array<int32_t, k1_loops> v_scales_bs;
            // LDS byte addresses of this lane's next-tile K scales and this tile's V scales.
            [[maybe_unused]] int32_t mx_k_read = 0;
            [[maybe_unused]] int32_t mx_v_read = 0;
            v_lds_read_window.set_bottom_tensor_view_data_ptr(v_lds_cur_ptr);
            // LDS read bases of this iteration: K groups 1.., the V tile, then next K group 0.
            constexpr index_t kKBases    = Geometry::kQkSuNIter;
            constexpr index_t kVBase     = (Geometry::kQkStages - 1) * kKBases;
            constexpr index_t kVBases    = kLdsSegSplit ? 2 : 1;
            constexpr index_t kVSplit    = kLdsSegSplit ? kVHalfBytes : 0;
            constexpr index_t kNextKBase = kVBase + kVBases;
            array<int32_t, kNextKBase + 1> lds_bases;
            auto pinned = [&](const auto& window, auto first, auto num_bases) {
                return FmhaTdmSchedFp8PinnedWindow<remove_cvref_t<decltype(window)>,
                                                   decltype(num_bases)::value>{
                    window, &lds_bases[decltype(first)::value]};
            };
            auto pinned_v = [&] {
                return FmhaTdmSchedFp8PinnedWindow<remove_cvref_t<decltype(v_lds_read_window)>,
                                                   kVBases,
                                                   kVSplit>{v_lds_read_window, &lds_bases[kVBase]};
            };

            auto run_fragment_pv_stage = [&](auto stage,
                                             const auto& a_block_tensor,
                                             const auto& b_block_tensor,
                                             auto& next_b_block_tensor,
                                             const auto& b_lds_window) {
                using BlockGemm   = remove_cvref_t<decltype(gemm_1)>;
                using WarpGemm    = typename BlockGemm::WarpGemm;
                using AWarpDstr   = typename WarpGemm::AWarpDstr;
                using BWarpDstr   = typename WarpGemm::BWarpDstr;
                using AWarpTensor = typename WarpGemm::AWarpTensor;
                using BWarpTensor = typename WarpGemm::BWarpTensor;
                using CWarpTensor = typename WarpGemm::CWarpTensor;
                constexpr auto a_warp_y_lengths =
                    to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
                constexpr auto b_warp_y_lengths =
                    to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
                constexpr auto a_warp_y_index_zeros = uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
                constexpr auto b_warp_y_index_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};

                if constexpr(kBlockScale)
                    v_scale = v_scales_bs[stage];
                static_assert(!kMx || Geometry::kPvStages == 1, "MX V scales cover one PV stage");

                auto emit_wmma = [&](auto, auto wmma) {
                    constexpr index_t ordinal     = PvOrdinal(decltype(wmma)::value);
                    constexpr index_t d_msb       = ordinal / Policy::OutputFragments::kNumN;
                    constexpr index_t n           = ordinal % Policy::OutputFragments::kNumN;
                    constexpr index_t m_iter      = d_msb / 2;
                    constexpr index_t v_msb       = d_msb % 2;
                    constexpr index_t full_n_iter = n * 2 + v_msb;

                    AWarpTensor a_warp_tensor;
                    a_warp_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<0, m_iter>{}, a_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                    BWarpTensor b_warp_tensor;
                    b_warp_tensor.get_thread_buffer() = b_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<0, full_n_iter>{}, b_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

                    CWarpTensor c_warp_tensor;
                    c_warp_tensor.get_thread_buffer().template set_as<fp32x8_t>(
                        number<0>{}, o_acc.at(number<ordinal>{}));
                    // LDS-read MX scales are already VGPRs; a pin would add a wait per scale.
                    if constexpr(kMx)
                        WarpGemm{}(c_warp_tensor,
                                   a_warp_tensor,
                                   b_warp_tensor,
                                   kE8m0One,
                                   v_scales[full_n_iter]);
                    else if constexpr(kBlockScale)
                        WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor, kE8m0One, v_scale);
                    else
                        WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
                    o_acc.at(number<ordinal>{}) =
                        c_warp_tensor.get_thread_buffer().template get_as<fp32x8_t>(number<0>{});
                };

                // Dword next-K stores drop QK byte permutes; MX and blockscale keep bytes for size.
                if constexpr(kD256 && kLdsSegSplit)
                    Policy::template RunPvStageWithWmma<decltype(stage)::value,
                                                        Mode,
                                                        !kMx && !kBlockScale>(
                        next_b_block_tensor, b_lds_window, v_tile, pinned_v(), emit_wmma);
                else if constexpr(kD256)
                    Policy::template RunPvStageWithWmma<decltype(stage)::value,
                                                        Mode,
                                                        !kMx && !kBlockScale>(
                        next_b_block_tensor,
                        b_lds_window,
                        v_tile,
                        pinned(v_lds_read_window, number<kVBase>{}, number<1>{}),
                        emit_wmma);
                else
                {
                    typename Policy::NoStreamedV no_v;
                    Policy::template RunPvStageWithWmma<decltype(stage)::value,
                                                        Mode,
                                                        !kMx && !kBlockScale>(
                        next_b_block_tensor, b_lds_window, no_v, no_v, emit_wmma);
                }
            };

            // Wave priority by phase: PV > QK > softmax exps > softmax max.
            constexpr int16_t kQkPrio = 2, kSoftmaxMaxPrio = 0, kSoftmaxExpPrio = 1, kPvPrio = 3;
            auto run_pv = [&](const auto& p_tile) {
                __builtin_amdgcn_s_setprio(kPvPrio);
                if constexpr(kBlockScale)
                {
                    static_for<0, k1_loops, 1>{}(
                        [&](auto i_k1) { v_scales_bs(i_k1) = PackVScale(v_descales, i_k1); });
                    // Both descale sets are consumed; fetch the next tile's.
                    if constexpr(kHasSink)
                    {
                        const index_t next = i_total_loops + 1;
                        kv_scale_origin    = next < num_sink_loop ? kN0 * next + kv_load_start
                                                                  : kN0 * (next - num_sink_loop) +
                                                                     logical_seqlen_k_start;
                        LoadKVDescalesAt(
                            k_descale_res, kv_scale_origin, block_scale_size_kv, k_descales);
                        LoadKVDescalesAt(
                            v_descale_res, kv_scale_origin, block_scale_size_kv, v_descales);
                    }
                    else
                        LoadKVDescalesFromOffsets(
                            k_descale_res, v_descale_res, kv_scale_offsets, k_descales, v_descales);
                }
                if constexpr(1 < k1_loops)
                {
                    static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
                        move_tile_window(v_lds_read_window, {kK1, 0});
                        decltype(v_tile) v_tile_switch;
                        auto p_tile_stage = get_slice_tile(
                            p_tile, sequence<0, i_k1 * kK1>{}, sequence<kM0, (i_k1 + 1) * kK1>{});
                        run_fragment_pv_stage(
                            i_k1, p_tile_stage, v_tile, v_tile_switch, v_lds_read_window);
                        v_tile = v_tile_switch;
                    });
                    move_tile_window(v_lds_read_window, {-kK1 * (k1_loops - 1), 0});
                }

                auto p_tile_last = get_slice_tile(
                    p_tile, sequence<0, (k1_loops - 1) * kK1>{}, sequence<kM0, k1_loops * kK1>{});
                auto k_lds_su0_read_window = make_k_su_read_window(k_lds_next_ptr);
                run_fragment_pv_stage(
                    number<Geometry::kPvStages - 1>{},
                    p_tile_last,
                    v_tile,
                    k_su0_tile,
                    pinned(k_lds_su0_read_window, number<kNextKBase>{}, number<1>{}));
                if constexpr(kNormalizeQuantizedP)
                    AccumulateQuantizedMass<decltype(gemm_1)>(p_tile,
                                                              kQuantizedRowSum ? l : l_quant,
                                                              pv_output_scale_m0,
                                                              pv_output_scale_m1);
            };

            auto begin_iteration = [&]() {
                // Fixed boundary: start the V TDM transfer once preceding reads drain.
                s_wait_tensorcnt<kTdmIssuedAfterV0>();
                block_sync_lds<k_lds_insts>();
                if constexpr(kMx)
                {
                    // DS reads retire in order: the wait above retired every read of this buffer.
                    CompilerMemoryFence();
                    // Without sink tokens the KV tiles are contiguous from kv_load_start.
                    kv_tile_start           = kv_load_start + mx_loop * kN0;
                    index_t next_tile_start = kv_tile_start + kN0;
                    if constexpr(kHasSink)
                    {
                        // Tiles past the sink prefix jump to the window start.
                        const index_t window_base = logical_seqlen_k_start - sink_seq_end;
                        if(mx_loop >= num_sink_loop)
                            kv_tile_start = window_base + mx_loop * kN0;
                        if(mx_loop + 1 >= num_sink_loop)
                            next_tile_start = window_base + (mx_loop + 1) * kN0;
                    }
                    const bool has_next = kUseCountdownLoop ? remaining_loops > 1
                                                            : i_total_loops + 1 < num_total_loop;
                    // The last tile reloads its own scales so the copy stays in bounds.
                    IssueMxScalesToLds<decltype(mx_all_fast)::value>(mx,
                                                                     mx_copy,
                                                                     MxScaleLds(mx_loop + 1),
                                                                     has_next ? next_tile_start
                                                                              : kv_tile_start,
                                                                     kv_last);
                    // Address the post-softmax scale reads now so they need no fresh VALU there.
                    mx_k_read = PinToVgpr(LdsByteAddress(MxScaleLds(mx_loop + 1) + mx_lane));
                    mx_v_read =
                        PinToVgpr(LdsByteAddress(MxScaleLds(mx_loop) + kMxKDwords + mx_lane));
                }
                // V is prefetched one iteration before its consumer.
                if constexpr(kHasSink)
                {
                    if(i_total_loops == num_sink_loop - 1 && i_total_loops + 1 < num_total_loop)
                        move_tile_window(v_dram_window, {logical_seqlen_k_start - sink_seq_end, 0});
                }
                move_tile_window(v_dram_window, {kN0, 0});
                if constexpr(kLdsSegSplit)
                    v_lds_write_window.set_bottom_tensor_view_data_ptr(
                        v_seg(v_lds_next_ptr, seg_write));
                else
                    v_lds_write_window.set_bottom_tensor_view_data_ptr(v_lds_next_ptr);
                LoadTileTdmInLoop(tdm_config_v, v_lds_write_window, v_dram_window);
            };

            auto run_qk = [&] {
                __builtin_amdgcn_s_setprio(kQkPrio);
                clear_tile(s_acc);
                constexpr auto gemm_0_su  = Policy::template GetQKBlockGemmSu<Problem>();
                auto k_lds_su_read_window = make_k_su_read_window(k_lds_cur_ptr);
                // One address wait here, after the barrier, instead of one per K group.
                static_for<0, Geometry::kQkStages - 1, 1>{}([&](auto i_su) {
                    auto window = k_lds_su_read_window;
                    move_tile_window(window, {(i_su + 1) * Geometry::kQkSuColumns, 0});
                    FmhaTdmSchedFp8PinnedWindow<decltype(window), kKBases>::ComputeBases(
                        window, &lds_bases(number<i_su * kKBases>{}));
                    if constexpr(kLdsSegSplit)
                        static_for<0, kKBases, 1>{}([&](auto j) {
                            lds_bases(number<i_su * kKBases + j>{}) +=
                                decltype(i_su)::value + 1 < Geometry::kQkStages / 2 ? seg_k_lo
                                                                                    : seg_k_hi;
                        });
                });
                FmhaTdmSchedFp8PinnedWindow<decltype(v_lds_read_window), kVBases, kVSplit>::
                    ComputeBases(v_lds_read_window, &lds_bases(number<kVBase>{}));
                auto k_su0_window = k_lds_su_read_window;
                k_su0_window.set_bottom_tensor_view_data_ptr(k_lds_next_ptr);
                FmhaTdmSchedFp8PinnedWindow<decltype(k_su0_window), 1>::ComputeBases(
                    k_su0_window, &lds_bases(number<kNextKBase>{}));
                if constexpr(kLdsSegSplit)
                {
                    lds_bases(number<kVBase>{}) += seg_v_lo;
                    lds_bases(number<kVBase + 1>{}) += seg_v_hi;
                    lds_bases(number<kNextKBase>{}) += seg_k_lo;
                }
                PinEachToVgpr(lds_bases);
                auto s_acc_su  = gemm_0_su.MakeCBlockTile();
                auto k_su_tile = k_su0_tile;
                decltype(k_su_tile) k_su_tile_next;
                // Blockscale keeps the per-stage K read order; other modes issue group 1 early.
                constexpr bool kQkLead = !kBlockScale;
                // Group 1's first half goes out before the first WMMA, a full stage ahead of use.
                if constexpr(kQkLead)
                {
                    static_for<0, Geometry::kKSuLoadCount / 2, 1>{}([&](auto a) {
                        Policy::KLoad::template LoadInstruction<decltype(a)::value>(
                            k_su_tile_next,
                            pinned(k_lds_su_read_window, number<0>{}, number<kKBases>{}));
                    });
                    __builtin_amdgcn_sched_barrier(0);
                }
                // WMMA scale operands of QK stage `stage`, used by MX only.
                auto qk_a_scale = [&](auto) {
                    if constexpr(kMx)
                        return [&](auto m_iter, auto k_iter) {
                            return q_scales[number<k_iter * Geometry::kQkMIter + m_iter>{}];
                        };
                    else
                        return typename Policy::NoScale{};
                };
                auto qk_b_scale = [&](auto stage) {
                    if constexpr(kMx)
                        return [&](auto n_iter, auto k_iter) {
                            return k_scales[number<k_iter * kMxKScalesPerPlane +
                                                   decltype(stage)::value * Geometry::kQkSuNIter +
                                                   decltype(n_iter)::value>{}];
                        };
                    else
                        return typename Policy::NoScale{};
                };
                auto run_qk_lead_stage = [&](auto stage,
                                             auto& next_tile,
                                             const auto& window,
                                             auto& lead_tile,
                                             const auto& lead_window) {
                    Policy::template RunQkScheduledLeadStage<decltype(stage)::value, Mode>(
                        gemm_0_su,
                        s_acc_su,
                        q_tile,
                        k_su_tile,
                        next_tile,
                        window,
                        lead_tile,
                        lead_window,
                        qk_a_scale(stage),
                        qk_b_scale(stage));
                };
                auto run_qk_stage = [&](auto stage, auto& next_tile, const auto& window) {
                    Policy::template RunQkScheduledStage<decltype(stage)::value, Mode>(
                        gemm_0_su,
                        s_acc_su,
                        q_tile,
                        k_su_tile,
                        next_tile,
                        window,
                        qk_a_scale(stage),
                        qk_b_scale(stage));
                };
                static_for<0, Geometry::kQkStages, 1>{}([&](auto i_su) {
                    // Re-issued mid-QK so the extra setprio also splits the QK schedule.
                    if constexpr(decltype(i_su)::value == Geometry::kQkStages / 2)
                        __builtin_amdgcn_s_setprio(kQkPrio);
                    clear_tile(s_acc_su);
                    if constexpr(!kQkLead && decltype(i_su)::value < Geometry::kQkStages - 1)
                    {
                        run_qk_stage(i_su,
                                     k_su_tile_next,
                                     pinned(k_lds_su_read_window,
                                            number<i_su * kKBases>{},
                                            number<kKBases>{}));
                        k_su_tile = k_su_tile_next;
                    }
                    else if constexpr(decltype(i_su)::value < Geometry::kQkStages - 2)
                    {
                        decltype(k_su_tile) k_su_tile_lead;
                        run_qk_lead_stage(i_su,
                                          k_su_tile_next,
                                          pinned(k_lds_su_read_window,
                                                 number<i_su * kKBases>{},
                                                 number<kKBases>{}),
                                          k_su_tile_lead,
                                          pinned(k_lds_su_read_window,
                                                 number<(i_su + 1) * kKBases>{},
                                                 number<kKBases>{}));
                        k_su_tile      = k_su_tile_next;
                        k_su_tile_next = k_su_tile_lead;
                    }
                    else if constexpr(decltype(i_su)::value == Geometry::kQkStages - 2)
                    {
                        typename Policy::NoLeadGroup no_lead;
                        run_qk_lead_stage(i_su,
                                          k_su_tile_next,
                                          pinned(k_lds_su_read_window,
                                                 number<i_su * kKBases>{},
                                                 number<kKBases>{}),
                                          no_lead,
                                          no_lead);
                        k_su_tile = k_su_tile_next;
                    }
                    else
                    {
                        if constexpr(kLdsSegSplit)
                            run_qk_stage(i_su, v_tile, pinned_v());
                        else
                            run_qk_stage(i_su,
                                         v_tile,
                                         pinned(v_lds_read_window, number<kVBase>{}, number<1>{}));
                    }
                    if constexpr(kKDescaleWithScaleS)
                        tile_elementwise_inout(
                            [kd = k_descales[i_su] * scale_s](auto& x) { x *= kd; }, s_acc_su);
                    else if constexpr(kBlockScale && !kKDescaleInSoftmax)
                        tile_elementwise_inout([kd = k_descales[i_su]](auto& x) { x *= kd; },
                                               s_acc_su);
                    set_slice_tile(
                        s_acc,
                        s_acc_su,
                        sequence<0, decltype(i_su)::value * Geometry::kQkSuColumns>{},
                        sequence<kM0, (decltype(i_su)::value + 1) * Geometry::kQkSuColumns>{});
                    if constexpr(kProgressiveMax && decltype(i_su)::value < Geometry::kQkStages - 1)
                        Policy::template StageLaneMax<decltype(i_su)::value,
                                                      (decltype(i_su)::value > 0)>(s_acc,
                                                                                   qk_lane_max);
                });
            };

            auto process_score = [&]() {
                // QK scores are ready for bias, masking, and softmax.
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                {
                    if constexpr(!kKDescaleWithScaleS)
                        tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, s_acc);
                    const auto bias_tile = bias_next;
                    tile_elementwise_inout(
                        [](auto& x, const auto& y) {
                            x += log2e_v<SaccDataType> * type_convert<SaccDataType>(y);
                        },
                        s_acc,
                        bias_tile);
                }

                constexpr bool kDeferTensorReady = Mode::kDeferTensorReady;
                if constexpr(!kDeferTensorReady)
                    s_wait_tensorcnt_barrier<Policy::kVPrefetchTensorCount>();

                const auto is_masked_out = [&](index_t row, index_t col) {
                    if constexpr(kHasSink)
                        return mask.IsOutOfSinkBound(row, col);
                    else if constexpr(std::is_same_v<FmhaMask,
                                                     SimplifiedGenericAttentionMask<true>>)
                    {
                        if constexpr(kUnpinnedMaskWidth)
                        {
                            const index_t x_start = -mask.GetWindowY() + row + 1;
                            const index_t x_end   = min(row + mask.GetWindowX(), mask.GetXTotal());
                            const uint32_t width  = row < mask.GetYTotal() && x_start < x_end
                                                        ? uint32_t(x_end) - uint32_t(x_start)
                                                        : 0u;
                            return uint32_t(col) - uint32_t(x_start) >= width;
                        }
                        else
                            return mask.IsOutOfBoundUnsigned(row, col);
                    }
                    else
                        return mask.IsOutOfBound(row, col);
                };
                // One predicate pass keeps D64 sink edge tiles from holding tail and pad indices.
                constexpr bool kFusedEdgePredicate = Geometry::kHeadDimQK == 64 &&
                                                     !FmhaMask::IsMasking &&
                                                     BiasEnum == BlockAttentionBiasEnum::NO_BIAS;

                // Sink-aware k_origin
                const auto k_origin = [&]() {
                    if constexpr(kUseCountdownLoop)
                    {
                        // V's window already points one prefetched tile ahead.
                        return make_tuple(v_dram_window.get_window_origin().at(I0) - kN0, 0);
                    }
                    const bool in_sink_phase = (num_sink_loop > i_total_loops);
                    if(in_sink_phase)
                        return make_tuple(kN0 * i_total_loops + kv_load_start, 0);
                    else
                        return make_tuple(
                            kN0 * (i_total_loops - num_sink_loop) + logical_seqlen_k_start, 0);
                }();
                if constexpr(kUseCountdownLoop)
                {
                    const bool needs_tail_predicate = k_origin.at(I0) + kN0 > logical_seqlen_k_end;
                    const bool need_split_check     = remaining_loops == 1 && needs_tail_predicate;
                    bool need_mask_check            = false;
                    if constexpr(kPadSeqLenK)
                    {
                        need_mask_check = mask.IsEdgeTile(
                            q_origin.at(I0), k_origin.at(I0), number<kM0>{}, number<kN0>{});
                    }

                    if(__builtin_expect(need_split_check || need_mask_check, false))
                    {
                        if constexpr(kMergeHeads)
                            SetTileIfRowPos(
                                s_acc,
                                -numeric<SMPLComputeDataType>::infinity(),
                                MergedRowPositions<decltype(m)>(merged, q_origin.at(I0)),
                                k_origin.at(I0),
                                [&](index_t row, index_t col) {
                                    return (need_split_check && logical_seqlen_k_end <= col) ||
                                           (need_mask_check && mask.IsOutOfBound(row, col));
                                });
                        else if constexpr(!kLdsSegSplit)
                            SetTileIfLaneBase(
                                s_acc,
                                -numeric<SMPLComputeDataType>::infinity(),
                                [&](auto tile_idx) {
                                    const auto row = q_origin.at(I0) + tile_idx.at(I0);
                                    const auto col = k_origin.at(I0) + tile_idx.at(I1);
                                    return (need_split_check && logical_seqlen_k_end <= col) ||
                                           (need_mask_check && mask.IsOutOfBound(row, col));
                                });
                        else
                            SetTileIfLaneBase(
                                s_acc,
                                -numeric<SMPLComputeDataType>::infinity(),
                                [&](auto tile_idx, auto const_idx) {
                                    // The constant column part alone picks the K half.
                                    const auto row = q_origin.at(I0) + tile_idx.at(I0);
                                    const auto col =
                                        k_origin.at(I0) + tile_idx.at(I1) +
                                        (const_idx.at(I1) < kN0 / 2 ? seg_col : -seg_col);
                                    return (need_split_check && logical_seqlen_k_end <= col) ||
                                           (need_mask_check && mask.IsOutOfBound(row, col));
                                });
                        if constexpr(kProgressiveMax)
                            static_for<0, Geometry::kQkStages - 1, 1>{}([&](auto su) {
                                Policy::template StageLaneMax<decltype(su)::value,
                                                              (decltype(su)::value > 0)>(
                                    s_acc, qk_lane_max);
                            });
                    }
                }
                else if constexpr(kFusedEdgePredicate)
                {
                    const bool needs_tail_predicate = k_origin.at(I0) + kN0 > logical_seqlen_k_end;
                    const bool need_split_check =
                        i_total_loops == (num_total_loop - 1) && needs_tail_predicate;
                    bool need_mask_check = false;
                    if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
                    {
                        need_mask_check = mask.IsEdgeTile(
                            q_origin.at(I0), k_origin.at(I0), number<kM0>{}, number<kN0>{});
                    }

                    if(__builtin_expect(need_split_check || need_mask_check, false))
                    {
                        SetTileIfLaneBase(
                            s_acc, -numeric<SMPLComputeDataType>::infinity(), [&](auto tile_idx) {
                                const auto row = q_origin.at(I0) + tile_idx.at(I0);
                                const auto col = k_origin.at(I0) + tile_idx.at(I1);
                                return (need_split_check && logical_seqlen_k_end <= col) ||
                                       (need_mask_check && is_masked_out(row, col));
                            });
                    }
                }
                else if constexpr(kBandMask)
                {
                    const auto ninf                     = -numeric<SMPLComputeDataType>::infinity();
                    const index_t q0                    = q_origin.at(I0);
                    const index_t k0                    = k_origin.at(I0);
                    const index_t lr                    = s_lane_origin.at(I0);
                    const index_t lc                    = s_lane_origin.at(I1);
                    [[maybe_unused]] const index_t diag = k0 - q0;
                    // Sink keys of in-range rows escape every bound but the right one.
                    [[maybe_unused]] const index_t ts =
                        (mask.GetWindowY() < mask.GetYTotal() ? mask.GetSink() : 0) - 1 - k0 - lc;
                    [[maybe_unused]] const index_t tr = mask.GetXTotal() - 1 - q0 - lr;
                    auto mask_if                      = [&](auto out) {
                        if constexpr(kHasSink && FmhaMask::IsMasking)
                            SetTileIfConstPart(s_acc, ninf, [&](index_t r, index_t c) {
                                return out(r, c) && !(c <= ts && r <= tr);
                            });
                        else
                            SetTileIfConstPart(s_acc, ninf, out);
                    };
                    // Also covers the tile-aligned split end, which only adds out-of-window keys.
                    if(k0 + kN0 > mask.GetXTotal())
                    {
                        const index_t t = mask.GetXTotal() - 1 - k0 - lc;
                        mask_if([&](index_t, index_t c) { return c > t; });
                    }
                    if constexpr(FmhaMask::IsMasking)
                    {
                        if(diag + (kN0 - 1) >= mask.GetWindowX())
                        {
                            const index_t t = mask.GetWindowX() - 1 - diag - (lc - lr);
                            SetTileIfConstPart(
                                s_acc, ninf, [&](index_t r, index_t c) { return c - r > t; });
                        }
                        if(diag - (kM0 - 1) <= -mask.GetWindowY())
                        {
                            const index_t t = 1 - mask.GetWindowY() - diag - (lc - lr);
                            mask_if([&](index_t r, index_t c) { return c - r < t; });
                        }
                        if(q0 + kM0 > mask.GetYTotal())
                        {
                            const index_t t = mask.GetYTotal() - 1 - q0 - lr;
                            mask_if([&](index_t r, index_t) { return r > t; });
                        }
                    }
                }
                else
                {
                    const bool needs_tail_predicate = k_origin.at(I0) + kN0 > logical_seqlen_k_end;
                    const bool is_last_loop         = i_total_loops == (num_total_loop - 1);
                    if(is_last_loop && needs_tail_predicate)
                    {
                        set_tile_if(
                            s_acc,
                            -numeric<SMPLComputeDataType>::infinity(),
                            [&, logical_seqlen_k_end_ = logical_seqlen_k_end](auto tile_idx) {
                                const auto col = k_origin.at(I0) + tile_idx.at(I1);
                                return logical_seqlen_k_end_ <= col;
                            });
                    }

                    if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
                    {
                        bool need_perpixel_check;
                        if constexpr(kMergeHeads &&
                                     std::is_same_v<FmhaMask, SimplifiedGenericAttentionMask<true>>)
                        {
                            // Merged rows are query positions below seqlen_q, so no row is padding.
                            const index_t k0 = k_origin.at(I0);
                            need_perpixel_check =
                                k0 + kN0 > min(mask.GetWindowX(), mask.GetXTotal()) ||
                                k0 + mask.GetWindowY() < mask.GetYTotal();
                        }
                        else
                            need_perpixel_check = mask.IsEdgeTile(
                                q_origin.at(I0), k_origin.at(I0), number<kM0>{}, number<kN0>{});
                        if(need_perpixel_check)
                        {
                            if constexpr(kMergeHeads)
                                SetTileIfRowPos(
                                    s_acc,
                                    -numeric<SMPLComputeDataType>::infinity(),
                                    MergedRowPositions<decltype(m)>(merged, q_origin.at(I0)),
                                    k_origin.at(I0),
                                    is_masked_out);
                            else
                                set_tile_if(s_acc,
                                            -numeric<SMPLComputeDataType>::infinity(),
                                            [&](auto tile_idx) {
                                                const auto row = q_origin.at(I0) + tile_idx.at(I0);
                                                const auto col = k_origin.at(I0) + tile_idx.at(I1);
                                                return is_masked_out(row, col);
                                            });
                        }
                    }
                }

                // Sink->normal window jump
                if constexpr(kHasSink)
                {
                    if(i_total_loops == num_sink_loop - 1)
                    {
                        move_tile_window(bias_dram_window,
                                         {0, logical_seqlen_k_start - sink_seq_end});
                    }
                }
                move_tile_window(bias_dram_window, {0, kN0});
                // Unguarded loads issue together; out-of-range keys are masked to -inf.
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                    bias_next = load_tile(bias_dram_window, number<-1>{}, bool_constant<false>{});
            };

            auto run_softmax = [&] {
                __builtin_amdgcn_s_setprio(kSoftmaxMaxPrio);
                if constexpr(kBlockScale && !kHasSink)
                    AdvanceKVScaleCursor(kv_scale_cursor, kv_scale_offsets);
                auto s_new = cast_tile<SMPLComputeDataType>(s_acc);

                auto p_tile = [&]() {
                    static_assert(kNWarp == 1);
                    static_assert(std::is_same_v<SaccDataType, float> &&
                                  std::is_same_v<SMPLComputeDataType, float>);
                    float delta_m0;
                    float delta_m1;
                    // The bias path already scaled S into the log2 domain.
                    const auto softmax_scale = [&]() {
                        if constexpr(kRowScale)
                        {
                            const RowScalePair scales = LoadRowScales();
                            return array<float, 2>{scales[0], scales[1]};
                        }
                        else if constexpr(BiasEnum == BlockAttentionBiasEnum::NO_BIAS)
                            return scale_s;
                        else
                            return 1.0f;
                    }();
                    const auto col_scale = [&]() {
                        if constexpr(kKDescaleInSoftmax)
                        {
                            // Floor: out-of-range blocks read 0, and -inf * 0 is NaN.
                            KVDescales scales;
                            static_for<0, kKVScaleGroups, 1>{}(
                                [&](auto g) { scales(g) = max(k_descales[g], 0x1p-60f); });
                            return scales;
                        }
                        else
                            return typename Policy::NoColumnScale{};
                    }();
                    Schedule::template RunSplitSoftmax<Mode>(
                        [&]() {
                            if constexpr(kProgressiveMax)
                                Policy::template RunSplitSoftmaxPart01FromPartial<Problem>(
                                    s_new, m, softmax_scale, delta_m0, delta_m1, qk_lane_max);
                            else
                                Policy::template RunSplitSoftmaxPart01<Problem>(
                                    s_new, m, softmax_scale, delta_m0, delta_m1, col_scale);
                        },
                        [&]() { s_wait_tensorcnt_barrier<Policy::kVPrefetchTensorCount>(); },
                        // Part 2 below rescales the output inside its exp stream.
                        [](auto) {},
                        [&](auto) {
                            __builtin_amdgcn_s_setprio(kSoftmaxExpPrio);
                            Policy::template RunSplitSoftmaxPart2AndGetScale<Problem,
                                                                             !kQuantizedRowSum>(
                                s_new,
                                m,
                                l,
                                softmax_scale,
                                delta_m0,
                                delta_m1,
                                pv_output_scale_m0,
                                pv_output_scale_m1,
                                o_acc,
                                col_scale);
                        });
                    return MakePForGemm1<decltype(gemm_1)>(s_new);
                }();
                return p_tile;
            };

            auto load_next_k = [&]() {
                // Fixed boundary: the next K TDM transfer follows softmax.
                if constexpr(kMx)
                    s_wait_asynccnt<0>();
                block_sync_lds<v_lds_insts>();
                if constexpr(kMx)
                {
                    CompilerMemoryFence();
                    ReadMxScalesFromLds(MxScaleLanePtr(mx_k_read), k_scales);
                    PadMxKScales(k_scales);
                    ReadMxScalesFromLds(MxScaleLanePtr(mx_v_read), v_scales);
                }
                // The K transfer here feeds iteration i+2, beyond the already prefetched K1.
                if constexpr(kHasSink)
                {
                    if(i_total_loops == num_sink_loop - 2 && i_total_loops + 2 < num_total_loop)
                        move_tile_window(k_dram_window, {logical_seqlen_k_start - sink_seq_end, 0});
                }
                move_tile_window(k_dram_window, {kN0, 0});
                if constexpr(kLdsSegSplit)
                    k_lds_write_window.set_bottom_tensor_view_data_ptr(
                        k_seg(k_lds_cur_ptr, seg_write));
                else
                    k_lds_write_window.set_bottom_tensor_view_data_ptr(k_lds_cur_ptr);
                LoadTileTdmInLoop(tdm_config_k, k_lds_write_window, k_dram_window);
            };

            auto advance_iteration = [&]() {
                // Fixed boundary: wait for the next K tile before advancing.
                s_wait_tensorcnt_barrier<Policy::kKPrefetchTensorCount>();
            };

            auto tile_ops = MakeFmhaTdmSchedTileOps(begin_iteration,
                                                    run_qk,
                                                    process_score,
                                                    run_softmax,
                                                    load_next_k,
                                                    run_pv,
                                                    advance_iteration);
            Schedule::Run(tile_ops);
        }; // mainloop

        // Dense D256 MX runs a loop copy without the edge-tile scale path when every tile is full.
        if constexpr(kMx && kUseCountdownLoop && kD256)
        {
            auto run_loop = [&](auto mx_all_fast) {
                do
                {
                    bool is_even_loop   = kUseCountdownLoop ? even_buffer : i_total_loops % 2 == 0;
                    auto k_lds_cur_ptr  = is_even_loop
                                              ? static_cast<KDataType* __restrict__>(smem_ptrk0)
                                              : static_cast<KDataType* __restrict__>(smem_ptrk1);
                    auto k_lds_next_ptr = is_even_loop
                                              ? static_cast<KDataType* __restrict__>(smem_ptrk1)
                                              : static_cast<KDataType* __restrict__>(smem_ptrk0);
                    auto v_lds_next_ptr = is_even_loop
                                              ? static_cast<VDataType* __restrict__>(smem_ptrv1)
                                              : static_cast<VDataType* __restrict__>(smem_ptrv0);
                    auto v_lds_cur_ptr  = is_even_loop
                                              ? static_cast<VDataType* __restrict__>(smem_ptrv0)
                                              : static_cast<VDataType* __restrict__>(smem_ptrv1);
                    mainloop(
                        k_lds_cur_ptr, k_lds_next_ptr, v_lds_next_ptr, v_lds_cur_ptr, mx_all_fast);
                    if constexpr(kUseCountdownLoop)
                    {
                        even_buffer = !even_buffer;
                        --remaining_loops;
                    }
                    else
                    {
                        i_total_loops++;
                    }
                } while(kUseCountdownLoop ? remaining_loops > 0 : i_total_loops < num_total_loop);
            };
            if(kv_load_start + (num_total_loop - 1) * kN0 < mx_copy.fast_end)
                run_loop(std::true_type{});
            else
                run_loop(std::false_type{});
        }
        else
        {
            do
            {
                bool is_even_loop   = kUseCountdownLoop ? even_buffer : i_total_loops % 2 == 0;
                auto k_lds_cur_ptr  = is_even_loop
                                          ? static_cast<KDataType* __restrict__>(smem_ptrk0)
                                          : static_cast<KDataType* __restrict__>(smem_ptrk1);
                auto k_lds_next_ptr = is_even_loop
                                          ? static_cast<KDataType* __restrict__>(smem_ptrk1)
                                          : static_cast<KDataType* __restrict__>(smem_ptrk0);
                auto v_lds_next_ptr = is_even_loop
                                          ? static_cast<VDataType* __restrict__>(smem_ptrv1)
                                          : static_cast<VDataType* __restrict__>(smem_ptrv0);
                auto v_lds_cur_ptr  = is_even_loop
                                          ? static_cast<VDataType* __restrict__>(smem_ptrv0)
                                          : static_cast<VDataType* __restrict__>(smem_ptrv1);
                mainloop(k_lds_cur_ptr,
                         k_lds_next_ptr,
                         v_lds_next_ptr,
                         v_lds_cur_ptr,
                         std::false_type{});
                if constexpr(kUseCountdownLoop)
                {
                    even_buffer = !even_buffer;
                    --remaining_loops;
                }
                else
                {
                    i_total_loops++;
                }
            } while(kUseCountdownLoop ? remaining_loops > 0 : i_total_loops < num_total_loop);
        }
        // Wave priority stays at kPvPrio; only the epilogue runs from here.

        // Reconstruct the CK tensor only at the final normalization boundary.
        auto output = Policy::OutputFragments::template Reconstruct<OaccBlockTileType>(o_acc);
        constexpr auto o_spans = decltype(output)::get_distributed_spans();

        if constexpr(kQuantized)
        {
            // Undo the fixed P scale; v_descale is the scalar descale, or 1 for blockscale and MX.
            sweep_tile_span(o_spans[I0], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                const float o_scale  = [&]() {
                    if constexpr(kQuantizedRowSum)
                        return v_descale;
                    else if constexpr(kNormalizeQuantizedP)
                        return l_quant[i_idx] == 0.f ? 0.f : v_descale * l[i_idx] / l_quant[i_idx];
                    else
                        return v_descale * (1.0f / kPScale);
                }();
                sweep_tile_span(o_spans[I1],
                                [&](auto idx1) { output(make_tuple(idx0, idx1)) *= o_scale; });
            });
        }

        if constexpr(kRowScale)
        {
            const RowScalePair scales         = LoadRowScales();
            row_scale.get_thread_buffer()(I0) = scales[0];
            row_scale.get_thread_buffer()(I1) = scales[1];
        }

        auto finalize_real_tokens = [&]() {
            if constexpr(kStoreLSE)
            {
                auto lse_acc =
                    make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

                constexpr auto lse_acc_spans = decltype(lse_acc)::get_distributed_spans();
                sweep_tile_span(lse_acc_spans[I0], [&, m_ = m, l_ = l](auto idx0) {
                    constexpr auto i_idx = make_tuple(idx0);
                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                    {
                        lse_acc(i_idx) = m_[i_idx] / C_LOG2E + log(l_[i_idx]);
                    }
                    else
                    {
                        lse_acc(i_idx) = m_[i_idx] * scale_of(i_idx) / C_LOG2E + log(l_[i_idx]);
                    }
                });

                store_tile(lse_acc_dram_window_tmp, lse_acc);
            }

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
                    output(i_j_idx) *= tmp;
                });
            });
        };

        if constexpr(kEnableVirtualSink)
        {
            if(__builtin_isinf_sign(sink_v) != -1)
            {
                auto lse_acc =
                    make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());
                const float sink_natural = sink_v * scale_s;
                const float sink_log2    = sink_natural * C_LOG2E;
                sweep_tile_span(o_spans[I0], [&](auto idx0) {
                    constexpr auto i_idx         = make_tuple(idx0);
                    const float row_sink_natural = [&] {
                        if constexpr(kMergeHeads)
                            return MergedSinkLogit(
                                merged, LaneRowOf<decltype(m)>(i_idx), scale_of(i_idx));
                        else
                            return sink_natural;
                    }();
                    const float row_sink_log2 =
                        kMergeHeads ? row_sink_natural * C_LOG2E : sink_log2;
                    float output_scale;
                    float merged_lse;
                    if(__builtin_isnan(row_sink_natural))
                    {
                        output_scale = row_sink_natural;
                        merged_lse   = row_sink_natural;
                    }
                    else if(l[i_idx] == 0.f || __builtin_isinf_sign(row_sink_natural) > 0)
                    {
                        output_scale = 0.f;
                        merged_lse   = row_sink_natural;
                    }
                    else
                    {
                        // The bias path keeps scale_s inside m.
                        const float real_log2 = BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS
                                                    ? m[i_idx]
                                                    : m[i_idx] * scale_of(i_idx);
                        const float merged_max  = max(real_log2, row_sink_log2);
                        const float real_weight = SplitSoftmax::Exp2(real_log2 - merged_max);
                        const float sink_weight = SplitSoftmax::Exp2(row_sink_log2 - merged_max);
                        const float merged_sum = __builtin_fmaf(l[i_idx], real_weight, sink_weight);
                        output_scale           = real_weight / merged_sum;
                        merged_lse             = merged_max / C_LOG2E + log(merged_sum);
                    }
                    if constexpr(kStoreLSE)
                        lse_acc(i_idx) = merged_lse;
                    sweep_tile_span(o_spans[I1], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        // Explicit zeros also cover empty rows and an infinite sink.
                        output(i_j_idx) =
                            output_scale == 0.f ? 0.f : output(i_j_idx) * output_scale;
                    });
                });
                if constexpr(kStoreLSE)
                    store_tile(lse_acc_dram_window_tmp, lse_acc);
            }
            else
            {
                finalize_real_tokens();
            }
        }
        else
        {
            finalize_real_tokens();
        }

        Schedule::template RunEpilogue<Mode>(
            [&]() {
                // A partial prefetch wait can leave the terminal look-ahead TDM in flight.
                s_wait_tensorcnt<0>();
            },
            [&]() {
                // The final PV stage's next-K reads have no consumer in the last iteration.
                s_wait_dscnt<0>();
            });

        return output;
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename LSEaccDramBlockWindowTmp,
              typename PositionEncoding>
    CK_TILE_HOST_DEVICE auto operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,
                                        const KDramBlockWindowTmp& k_dram_block_window_tmp,
                                        const VDramBlockWindowTmp& v_dram_block_window_tmp,
                                        const BiasDramBlockWindowTmp& bias_dram_block_window_tmp,
                                        LSEaccDramBlockWindowTmp& lse_acc_dram_window_tmp,
                                        FmhaMask mask,
                                        PositionEncoding position_encoding,
                                        float scale_s,
                                        float sink_v,
                                        void* smem_ptrk0,
                                        void* smem_ptrk1,
                                        void* smem_ptrv0,
                                        void* smem_ptrv1,
                                        const float* k_descale_ptr  = nullptr,
                                        const float* v_descale_ptr  = nullptr,
                                        index_t block_scale_size_kv = 1,
                                        float v_descale             = 1.0f,
                                        const MxDescales& mx        = {},
                                        const MergedHeads& merged   = {}) const
    {
        return run(q_dram_block_window_tmp,
                   k_dram_block_window_tmp,
                   v_dram_block_window_tmp,
                   bias_dram_block_window_tmp,
                   lse_acc_dram_window_tmp,
                   mask,
                   position_encoding,
                   scale_s,
                   smem_ptrk0,
                   smem_ptrk1,
                   smem_ptrv0,
                   smem_ptrv1,
                   sink_v,
                   k_descale_ptr,
                   v_descale_ptr,
                   block_scale_size_kv,
                   v_descale,
                   mx,
                   merged);
    }
};

} // namespace ck_tile
