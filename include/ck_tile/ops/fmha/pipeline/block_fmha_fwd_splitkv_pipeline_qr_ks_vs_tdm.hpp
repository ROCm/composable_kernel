// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_attention_quant_scale_enum.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

// This pipeline is qkv all located in LDS, targeting gfx1250
template <typename Problem_, typename Policy_ = BlockFmhaPipelineQRKSVSTdmDefaultPolicy>

// One V buffer and two K buffers in the LDS arena: split-kv problems hardwire
// kUseDoubleKVLdsBuffer to false, which selects the single-V layout.
// The sink logit is folded in exactly once, by the first split with work.
// fp8/bf8 is not supported here.
struct BlockFmhaFwdSplitKVPipelineQRKSVSTdm
{
    static constexpr bool kUsesLdsArena = true;

    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};

    using Problem             = remove_cvref_t<Problem_>;
    using Policy              = remove_cvref_t<Policy_>;
    using QDataType           = remove_cvref_t<typename Problem::QDataType>;
    using KDataType           = remove_cvref_t<typename Problem::KDataType>;
    using VDataType           = remove_cvref_t<typename Problem::VDataType>;
    using SaccDataType        = remove_cvref_t<typename Problem::SaccDataType>;
    using SMPLComputeDataType = remove_cvref_t<typename Problem::SMPLComputeDataType>;
    using BiasDataType        = remove_cvref_t<typename Problem::BiasDataType>;
    using LSEDataType         = remove_cvref_t<typename Problem::LSEDataType>;
    using PDataType           = remove_cvref_t<typename Problem::PDataType>;
    using OaccDataType        = remove_cvref_t<typename Problem::OaccDataType>;
    using ODataType           = remove_cvref_t<typename Problem::ODataType>;
    using AttentionVariant    = remove_cvref_t<typename Problem::AttentionVariant>;
    using FmhaMask            = remove_cvref_t<typename Problem::FmhaMask>;

    using BlockFmhaShape = remove_cvref_t<typename Problem::BlockFmhaShape>;
    using VLayout        = remove_cvref_t<typename BlockFmhaShape::VLayout>;
    // V is read untransposed as (seqlen_k, hdim_v); codegen emits only row-major V.
    static_assert(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>,
                  "qr_tdm split-kv supports only row-major V");
    static constexpr bool kQLoadOnce = true; // if q_tile load whole block length (hdim) at once
    static_assert(kQLoadOnce == Policy::QLoadOnce);
    // Each LDS buffer holds a whole (kN0 x kSubQKHeaddim) K tile, which the
    // progressive reload walks slice by slice.
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
    // Softmax reads S straight from the gemm0 accumulators and MakePForGemm1
    // relies on aligned lanes; codegen pairs qr_tdm only with one N warp.
    static_assert(kNWarp == 1, "qr_tdm split-kv supports only one warp along N");

    // 512 is the last head dim this pipeline can stage: its arena needs a whole
    // 320 KiB of LDS, i.e. one workgroup per CU. See QrTdmLdsArenaLayout's
    // kLdsOccupancyTarget for the budget it switches to at that size.
    static_assert(kSubQKHeaddim <= 512, "hdim bigger than 512 is not suitable for this pipeline!");

    static constexpr bool kIsGroupMode = Problem::kIsGroupMode;
    static constexpr bool kPadSeqLenQ  = Problem::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK  = Problem::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ =
        Problem::kPadHeadDimQ; // support multiple of vector(like 8x)
    static constexpr bool kPadHeadDimV =
        Problem::kPadHeadDimV; // support multiple of vector(like 8x)

    static constexpr bool kHasLogitsSoftCap = Problem::kHasLogitsSoftCap;
    static constexpr auto BiasEnum          = Problem::BiasEnum;
    static constexpr bool kStoreLSE         = Problem::kStoreLSE;
    static constexpr bool kHasSink          = Problem::kHasSink;
    static constexpr bool kIsPagedKV        = Problem::kIsPagedKV;
    // Paged walks start on a page-friendly boundary that can precede the split,
    // so the head erase below is needed there whatever the split bounds look
    // like. Everywhere else this follows the problem.
    static constexpr bool kHasUnevenSplits = kIsPagedKV || Problem::kHasUnevenSplits;

    // FmhaFwdSplitKVKernel has no name-based fallback (unlike fmha_fwd_kernel.hpp),
    // so both must be declared: V is stepped by {kN0, 0}, which needs the
    // untransposed (seqlen, hdim_v) view, and the affine flag asks for the
    // unpadded K/V views this file's TDM loads expect.
    static constexpr bool kUsesUntransposedVKernelPath = true;
    static constexpr bool kUsesTdmAffineDramPath       = true;

    // The single-V arena below is the only layout this pipeline implements.
    static_assert(!Problem::kUseDoubleKVLdsBuffer,
                  "split-kv qr_tdm implements only the single-V LDS layout");

    // Reload K from LDS in place during the QK sweep. Independent of the
    // problem's kProgressiveDsLoadK, which configures the double-buffer path.
    // Gated on kN0 == 64 for the reload's static_asserts.
    static constexpr bool kReloadKProgressively = (kN0 == 64);

    // Every softmax site here calls exp2 directly; there is no exp() path to fall back to.
    static_assert(CK_TILE_FMHA_FWD_FAST_EXP2,
                  "qr_tdm pipeline: FAST_EXP2=0 has no code path here - log2(e) is never folded "
                  "into scale_s, so the kernel would return 2^logit instead of e^logit");

    // LogitsTransform leaves the capped score with scale_s and log2(e) already
    // folded in, which is the frame the bias branches hand downstream. Both at
    // once would scale twice, so they cannot share an instance.
    static_assert(!kHasLogitsSoftCap || BiasEnum == BlockAttentionBiasEnum::NO_BIAS,
                  "qr_tdm split-kv folds scale_s into the capped score, so logits soft cap and "
                  "bias cannot share the same instance");

    static_assert(!is_any_of<VDataType, fp8_t, bf8_t>::value,
                  "qr_tdm split-kv does not support fp8/bf8");

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
    static constexpr index_t kAlignmentQ = Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK = Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV = Policy::template GetAlignmentV<Problem>();

    static constexpr index_t kAlignmentO = Policy::template GetAlignmentO<Problem>();
    // Read by fmha_fwd_splitkv_kernel.hpp when it builds the o_acc dram window.
    // Unlike the generic pipeline this keeps the full vector length under
    // kPadHeadDimV, which holds because dcheck/dvcheck admit only hdim_v % 8 == 0.
    static constexpr index_t kAlignmentOacc = Policy::template GetAlignmentO<Problem>();

    static constexpr index_t kAlignmentBias =
        kPadSeqLenK ? 1 : Policy::template GetAlignmentBias<Problem>();

    static constexpr index_t kBlockPerCu = []() {
        if constexpr(Problem::kBlockPerCu != -1)
            return Problem::kBlockPerCu;
        else
        {
            if constexpr(kQKHeaddim <= 32)
            {
                return 2;
            }
            else if constexpr(kQKHeaddim <= 64)
            {
                return 3;
            }
            else if constexpr(kQKHeaddim <= 128)
            {
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS || kM0 >= 256)
                    return 1;
                else
                    return 2;
            }
            else if constexpr(kQKHeaddim <= 256)
            {
                return 1;
            }
            else
            {
                return 1;
            }
        }
    }();

    static constexpr const char* name = "qr_tdm";

    // Progressive K reload: pull the next four K accesses back into k_tile once
    // the WMMA sweep has consumed the fragments they overwrite. Run from the
    // gemm0 after-warp callback.
    template <typename Window, typename Tile, index_t NIter>
    CK_TILE_DEVICE static void
    ReloadProgressiveKFragment(Window& k_lds_read_window, Tile& k_tile, number<NIter>)
    {
        // Allow VALU/SALU/VMEM/DS-write/TRANS/LDSDMA to cross while
        // keeping MFMA/WMMA and DS-read ordered around the reload.
        constexpr unsigned kProgressiveDsLoadSchedMask =
            0x002 | 0x004 | 0x010 | 0x020 | 0x040 | 0x200 | 0x400 | 0x800;
        __builtin_amdgcn_sched_barrier(kProgressiveDsLoadSchedMask);
        constexpr index_t begin = 2 * (NIter - 1);
        k_lds_read_window.template load_access_range<begin, begin + 4>(k_tile);
        __builtin_amdgcn_sched_barrier(kProgressiveDsLoadSchedMask);
    }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        using Layout = typename Policy::template LdsArenaLayout<Problem>;
        return Layout::kArenaBytes;
    }

    // Re-pack gemm_0 C into gemm_1 A: C is M-outer (MIter,KIter), A is K-outer.
    // Lanes already align (NWarp==1), so this is an in-thread block reorder;
    // identity when MIterPerWarp==1.
    template <typename Gemm1, typename PComputeTensor>
    CK_TILE_DEVICE static auto MakePForGemm1(const PComputeTensor& p_compute)
    {
        constexpr index_t kPMI = Gemm1::MIterPerWarp;
        constexpr index_t kPKI = kN0 / Gemm1::WarpGemm::kK;

        auto p_tile = make_static_distributed_tensor<PDataType>(
            Policy::template MakePRegTileDistribution<Problem>());

        const auto p_src        = cast_tile<PDataType>(p_compute);
        constexpr index_t kPBuf = decltype(p_src)::get_thread_buffer_size();
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

    // Single V, double K LDS buffer path, copied from
    // BlockFmhaPipelineQRKSVSTdm::run_single_v_double_k_lds_buffer plus the
    // split-kv additions (page navigator, split bounds, sink folding, LSE-acc
    // output). Keep these identical to the source so a fix reaches both:
    // - TDM issue order: K(cur) one iteration ahead, then V, then K(next), with
    //   s_wait_tensorcnt_barrier<2> before gemm0, <1> before gemm1 and <0>
    //   after the loop.
    // - K-buffer parity: the loop runs unrolled by two so kCurBuf/kNextBuf are
    //   compile-time, and iteration 0 consumes buffer 0.
    // - Progressive-reload shape: ReloadProgressiveKFragment (sched mask and
    //   access range) and its trigger (last mIter, kIter 0, odd nIter).
    //
    // K and V arrive as window lengths plus a page block navigator rather than
    // ready-made windows: the navigator is what resolves a global seqlen_k
    // coordinate to a page and a page-local origin. For non-paged problems it is
    // a TrivialPageBlockNavigator whose make_tile_window / move_tile_window /
    // prefetch_table_id degenerate to the plain tile window calls, so both paths
    // share one body.
    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowLengths,
              typename KPageBlockNavigator,
              typename VDramBlockWindowLengths,
              typename VPageBlockNavigator,
              typename BiasDramBlockWindowTmp,
              typename LSEaccDramBlockWindowTmp,
              typename PositionEncoding,
              typename AttentionVariantParams,
              typename BlockIndices>
    CK_TILE_HOST_DEVICE auto run_single_v_double_k_lds_buffer(
        const QDramBlockWindowTmp& q_dram_block_window_tmp,         // M0*K0 tile
        const KDramBlockWindowLengths& k_dram_block_window_lengths, // N0*K0 tile
        const KPageBlockNavigator& k_page_block_navigator,
        const VDramBlockWindowLengths& v_dram_block_window_lengths, // N1*K1 tile
        const VPageBlockNavigator& v_page_block_navigator,
        const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
        LSEaccDramBlockWindowTmp& lse_acc_dram_window_tmp,        // M0*1 tile
        index_t num_splits,
        index_t i_split,
        FmhaMask mask,
        PositionEncoding position_encoding,
        float scale_s,
        const AttentionVariant& variant,
        const AttentionVariantParams& variant_params,
        const BlockIndices& block_indices,
        index_t kv_l2p_offset,
        void* smem_arena,
        float sink_v) const
    {
        using Layout = typename Policy::template LdsArenaLayout<Problem>;
        auto* smem_ptrq =
            reinterpret_cast<QDataType*>(static_cast<char*>(smem_arena) + Layout::kQOffset);
        auto* smem_ptrk =
            reinterpret_cast<KDataType*>(static_cast<char*>(smem_arena) + Layout::kK0Offset);
        auto* smem_ptrk1 =
            reinterpret_cast<KDataType*>(static_cast<char*>(smem_arena) + Layout::kK1Offset);
        auto* smem_ptrv =
            reinterpret_cast<VDataType*>(static_cast<char*>(smem_arena) + Layout::kV0Offset);
        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>>,
            "wrong!");

        // Q/K/V all reach LDS through load_tile_tdm; V is then read back with
        // ds_load_tr. V dram window keeps (kN1, kK1) shape.
        static_assert(
            kM0 == QDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                kSubQKHeaddim == QDramBlockWindowTmp{}.get_window_lengths()[I1] &&
                kN0 == KDramBlockWindowLengths{}[I0] && kK0 == KDramBlockWindowLengths{}[I1] &&
                kN1 == VDramBlockWindowLengths{}[I0] && kK1 == VDramBlockWindowLengths{}[I1] &&
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
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());

        auto o_acc = OaccBlockTileType{};

        // infer Sacc, S, P, M, L, Oacc type
        using SBlockTileType = decltype(cast_tile<SMPLComputeDataType>(o_acc));

        using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
            SBlockTileType{}, sequence<1>{}, f_max, SMPLComputeDataType{0}));

        // init M, L (sink-aware: when sink_v is finite, pre-seed m/l)
        auto m = MLBlockTileType{};
        auto l = MLBlockTileType{};

        clear_tile(o_acc);
        // Split-kv: the sink logit has to be counted exactly once across all the
        // splits, so only the first split that has work seeds m/l with it. That is
        // not always split 0 -- an earlier split can come out empty under a mask --
        // hence the fixup after the early-exit check below.
        const auto init_ml = [&](bool take_sink) {
            if(take_sink)
            {
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI ||
                             BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             kHasLogitsSoftCap)
                    set_tile(m, sink_v * scale_s * C_LOG2E);
                else
                    set_tile(m, sink_v * C_LOG2E);
                set_tile(l, SMPLComputeDataType{1.0f});
            }
            else
            {
                set_tile(m, -numeric<SMPLComputeDataType>::infinity());
                clear_tile(l);
            }
        };
        init_ml(__builtin_isinf_sign(sink_v) >= 0 && i_split == 0);

        const auto q_origin = q_dram_block_window_tmp.get_window_origin();

        // Sink-aware tile range: GetSinkTileRangeAlongX returns
        // (sink_seq_end, seqlen_k_start, seqlen_k_end). For non-sink,
        // sink_seq_end is always 0.
        //
        // Split-kv: the split-aware overloads intersect that range with this
        // split's slice of the key sequence. The slice bound is
        // ceil(seqlen_k / num_splits) and is therefore not kN0 aligned, so the
        // last tile of a split can reach past it -- kHasUnevenSplits erases that
        // tail from s_acc in the main loop.
        const auto tile_range_result = [&mask, &q_origin, num_splits, i_split]() {
            if constexpr(kHasSink)
                return mask.GetSinkTileRangeAlongX(
                    q_origin.at(number<0>{}), number<kM0>{}, number<kN0>{}, num_splits, i_split);
            else
            {
                auto [start, end] = mask.GetTileRangeAlongX(
                    q_origin.at(I0), number<kM0>{}, number<kN0>{}, num_splits, i_split);
                return ck_tile::make_tuple(0, start, end);
            }
        }();
        const auto sink_seq_end           = tile_range_result.get(ck_tile::number<0>{});
        const auto logical_seqlen_k_start = tile_range_result.get(ck_tile::number<1>{});
        const auto logical_seqlen_k_end   = tile_range_result.get(ck_tile::number<2>{});

        const auto num_sink_loop = integer_divide_ceil(sink_seq_end, kN0);

        // check early exit if no work to do
        if constexpr(FmhaMask::IsMasking || kPadSeqLenK || kHasUnevenSplits)
        {
            // Sink loops are deliberately not counted: a split with no ordinary KV
            // work exits through the closed form below, which decides sink ownership.
            // Looping over sink tiles instead gives -inf LSE when the rows are fully
            // masked (SplitKvSinkDroppedSink).
            const index_t logical_num_total_loop =
                integer_divide_ceil(logical_seqlen_k_end - logical_seqlen_k_start, kN0);
            if(logical_num_total_loop <= 0)
            {
                if constexpr(kStoreLSE)
                {
                    auto lse_acc =
                        make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

                    // The sink logit must enter the combined softmax denominator
                    // exactly once. Split 0 owns it only while no split has work
                    // of its own: any split with work seeds m/l with the sink
                    // itself, and publishing a sink-bearing lse_acc here as well
                    // would count exp(sink) twice.
                    bool keeps_sink = (__builtin_isinf_sign(sink_v) >= 0) && i_split == 0;
                    if(keeps_sink && 1 < num_splits)
                    {
                        // Ask for the ordinary range over the whole key axis. The
                        // sink-aware overload returns (sink_offset, start, end),
                        // and sink_offset answers ownership, not occupancy.
                        auto [full_start, full_end] = mask.GetTileRangeAlongX(
                            q_origin.at(number<0>{}), number<kM0>{}, number<kN0>{}, 1, 0);
                        keeps_sink = (full_end <= full_start);
                    }
                    if(keeps_sink)
                    {
                        set_tile(lse_acc, SMPLComputeDataType{sink_v * scale_s});
                    }
                    else
                    {
                        set_tile(lse_acc, -numeric<SMPLComputeDataType>::infinity());
                    }

                    store_tile(lse_acc_dram_window_tmp, lse_acc);
                }

                // o_acc is still cleared, and Q has not been loaded yet, so
                // there is nothing to fence before returning.
                return o_acc;
            }
        }

        // Reaching here means this split has work. It owns the sink logit when
        // the split before it was empty, which the seeding above could not know.
        if(i_split > 0)
        {
            auto [prev_start, prev_end] = mask.GetTileRangeAlongX(
                q_origin.at(I0), number<kM0>{}, number<kN0>{}, num_splits, i_split - 1);
            init_ml(__builtin_isinf_sign(sink_v) >= 0 && prev_start >= prev_end);
        }

        // ---------------------------------------------------------------------
        // TDM configs for Q / K / V
        // pad_enable + pad_amount + pad_interval are compile-time, sourced from
        // the policy. workgroup_mask defaults to 0 (no cluster multicast).
        // V uses load_tile_tdm (single-box plain layout) -- same TDM machinery
        // as Q / K, with V dram dist switched to trivial tile-major and the
        // V LDS read view kept plain row-major (matches the write view).
        // ---------------------------------------------------------------------
        TDMConfig tdm_config_q;
        TDMConfig tdm_config_k;
        TDMConfig tdm_config_v;
        {
            using QRaw =
                detail::EncodedTdmPadding<typename Policy::template LdsPaddingConfigQ<Problem>>;
            tdm_config_q.pad_enable              = QRaw::kEnabled;
            tdm_config_q.pad_config.pad_amount   = QRaw::kPadAmount;
            tdm_config_q.pad_config.pad_interval = QRaw::kPadInterval;

            using KRaw =
                detail::EncodedTdmPadding<typename Policy::template LdsPaddingConfigK<Problem>>;
            tdm_config_k.pad_enable              = KRaw::kEnabled;
            tdm_config_k.pad_config.pad_amount   = KRaw::kPadAmount;
            tdm_config_k.pad_config.pad_interval = KRaw::kPadInterval;

            using VRaw =
                detail::EncodedTdmPadding<typename Policy::template LdsPaddingConfigV<Problem>>;
            tdm_config_v.pad_enable              = VRaw::kEnabled;
            tdm_config_v.pad_config.pad_amount   = VRaw::kPadAmount;
            tdm_config_v.pad_config.pad_interval = VRaw::kPadInterval;
        }

        // Q tile in LDS
        auto q_dram_window = make_tile_window(
            q_dram_block_window_tmp, Policy::template MakeQDramTileDistribution<Problem>());

        // Q LDS writer (TDM) and reader share plain row-major desc; TDM
        // box-major write cannot produce XOR'd layout, so no swizzle here.
        auto q_lds_write_view = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<QDataType*>(smem_ptrq),
            Policy::template MakeQLdsBlockDescriptor<Problem>());

        auto q_lds_read_view = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<QDataType*>(smem_ptrq),
            Policy::template MakeQLdsBlockDescriptor<Problem>());

        auto q_lds_store_window =
            make_tile_window(q_lds_write_view,
                             Policy::template MakeQLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0});

        auto q_lds_read_window =
            make_tile_window(q_lds_read_view,
                             Policy::template MakeQLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0},
                             Policy::template MakeQRegTileDistribution<Problem>());

        load_tile_tdm(tdm_config_q, q_lds_store_window, q_dram_window);

        // K tile in LDS
        // For sink: kv_load_start is 0 when there are sink tokens (we start
        // reading from seq position 0), otherwise seqlen_k_start.
        // kv_l2p_offset shifts logical seqlen_k coordinates into the physical
        // ones the page table is indexed by. It is 0 for every non-paged problem.
        const index_t physical_seqlen_k_start = logical_seqlen_k_start + kv_l2p_offset;
        const index_t physical_seqlen_k_end   = logical_seqlen_k_end + kv_l2p_offset;

        // Paged-kv needs each tile inside one page, so the walk starts on a kN0
        // boundary (assumes page_block_size % kN0 == 0, a caller contract). That can
        // precede physical_seqlen_k_start; the main loop erases the head from s_acc.
        const index_t aligned_physical_seqlen_k_start =
            kIsPagedKV ? kN0 * integer_divide_floor(physical_seqlen_k_start, kN0)
                       : physical_seqlen_k_start;
        const auto kv_load_start = (sink_seq_end == 0 && aligned_physical_seqlen_k_start > 0)
                                       ? aligned_physical_seqlen_k_start
                                       : 0;

        // Bias tile window (ELEMENTWISE_BIAS or ALIBI -- null window for NO_BIAS).
        // Bias is indexed logically, so outside the sink range the origin has to
        // come back out of the physical frame.
        const auto bias_origin      = bias_dram_block_window_tmp.get_window_origin();
        const index_t bias_n_offset = [&]() {
            if constexpr(kHasSink)
                return kv_load_start;
            else
                return logical_seqlen_k_start -
                       (physical_seqlen_k_start - aligned_physical_seqlen_k_start);
        }();
        auto bias_dram_window =
            make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                             bias_dram_block_window_tmp.get_window_lengths(),
                             {bias_origin.at(number<0>{}), bias_n_offset},
                             gemm_0.MakeCBlockTile().get_tile_distribution());

        // K is staged a whole (kN0 x kSubQKHeaddim) tile at a time, into two
        // ping-pong buffers: the tile consumed this iteration and the next
        // iteration's tile, prefetched one iteration ahead.
        // kv_load_start is a global seqlen_k coordinate; the navigator resolves
        // it to a page and a page-local origin (identity when not paged).
        auto [i_page_block_k, k_dram_window] = k_page_block_navigator.make_tile_window(
            k_dram_block_window_lengths,
            {kv_load_start, 0},
            Policy::template MakeKDramTileDistribution<Problem, true>());

        // K LDS writer (TDM) and reader share plain row-major desc; see Q
        // comment above for the no-swizzle rationale.
        auto k_lds_write_view = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<KDataType*>(smem_ptrk),
            Policy::template MakeKLdsBlockDescriptor<Problem, true>());
        auto k_lds_read_view = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<KDataType*>(smem_ptrk),
            Policy::template MakeKLdsBlockDescriptor<Problem, true>());

        auto k_lds_write_window = make_tile_window(
            k_lds_write_view,
            Policy::template MakeKLdsBlockDescriptor<Problem, true>().get_lengths(),
            {0, 0});
        auto k_lds_read_window =
            make_tile_window(k_lds_read_view,
                             make_tuple(number<kN0>{}, number<kK0>{}),
                             {0, 0},
                             Policy::template MakeKRegTileDistribution<Problem>());

        // V tile in LDS: loaded via load_tile_tdm (same TDM machinery as Q/K),
        // with V's DRAM dist switched to trivial tile-major and the V LDS read
        // view kept plain row-major (matches the write view).
        // V is untransposed here, so seqlen_k is dim 0 for the navigator too
        // (the kernel builds it with VirtualDim 0 for this pipeline).
        auto [i_page_block_v, v_dram_window] = v_page_block_navigator.make_tile_window(
            v_dram_block_window_lengths,
            {kv_load_start, 0},
            Policy::template MakeVDramTileDistribution<Problem>());

        auto v_lds_write_view = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<VDataType*>(smem_ptrv),
            Policy::template MakeVLdsBlockDescriptor<Problem>());
        // V LDS read view uses the same plain row-major desc as the write
        // view (Xor=false). This matches the TDM box-major writer (single
        // plain box per V tile) and lets the existing MakeVRegTileDistribution
        // outer-dist + QuadInputEncoding suffix (TransposedDstrEncode) drive
        // ds_load_tr_b128 with per-lane VOFFSETs that satisfy the WMMA B
        // operand expected pattern.
        auto v_lds_read_view = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<VDataType*>(smem_ptrv),
            Policy::template MakeVLdsBlockDescriptor<Problem>());
        auto v_lds_write_window =
            make_tile_window(v_lds_write_view,
                             Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0});

        auto v_lds_read_window =
            make_tile_window(v_lds_read_view,
                             make_tuple(number<kK1>{}, number<kN1>{}),
                             {0, 0},
                             Policy::template MakeVRegTileDistribution<Problem>());

        s_wait_tensorcnt_barrier<0>();
        auto q_tile = load_tile(q_lds_read_window);

        const index_t num_total_loop =
            integer_divide_ceil(physical_seqlen_k_end - aligned_physical_seqlen_k_start, kN0) +
            num_sink_loop;

        index_t i_total_loops      = 0;
        constexpr index_t k0_loops = kQKHeaddim / kK0;
        constexpr index_t k1_loops = kN0 / kK1;

        static_assert(1 <= k0_loops);
        static_assert(1 <= k1_loops);

        // K ping-pong: each buffer holds a whole (kN0 x kSubQKHeaddim) K tile.
        // Iteration i consumes buffer (i % 2) while the next tile streams into
        // the other one, so the K load latency is hidden behind a full
        // gemm0 + softmax + gemm1 instead of a single gemm.
        //
        // The buffer is picked at compile time on purpose. Selecting it with a
        // runtime index turns the LDS base into a per-iteration s_cselect, and
        // here that was enough to lose the compiler's LDS address-space
        // inference: the K reads came out as generic flat_load instead of
        // ds_load, and since a flat access bumps loadcnt as well as dscnt,
        // every v_wmma ended up gated by a conservative s_wait_loadcnt_dscnt
        // 0x0, which costs far more than the ping-pong saves. Hence the
        // mainloop is unrolled by two and each half names its K buffer as a
        // constant.
        const auto k_lds_ptr = [&](auto i_buf) {
            if constexpr(decltype(i_buf)::value == 0)
                return smem_ptrk;
            else
                return smem_ptrk1;
        };

        // Every K/V DRAM advance goes through the navigator, which also swaps
        // the page pointer when paged; for a trivial navigator this folds to a
        // plain move_tile_window.
        auto advance_k = [&](index_t step) {
            const auto next_id = amd_wave_read_first_lane(
                k_page_block_navigator.prefetch_table_id(i_page_block_k, k_dram_window, {step, 0}));
            i_page_block_k = k_page_block_navigator.move_tile_window(
                i_page_block_k, k_dram_window, {step, 0}, next_id);
        };
        auto advance_v = [&](index_t step) {
            const auto next_id = amd_wave_read_first_lane(
                v_page_block_navigator.prefetch_table_id(i_page_block_v, v_dram_window, {step, 0}));
            i_page_block_v = v_page_block_navigator.move_tile_window(
                i_page_block_v, v_dram_window, {step, 0}, next_id);
        };

        // Prologue: issue K for the first KV tile (q_tile has already been read
        // out of LDS above, so reusing the Q region for K0 is safe), then point
        // the K dram window at the next tile -- the in-loop load prefetches the
        // *next* tile. k_lds_write_window is still bound to buffer 0, so
        // iteration 0 consumes buffer 0 and therefore runs at parity 0.
        block_sync_lds();
        load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);
        advance_k(num_total_loop > 1 ? kN0 : 0);

        // One KV-tile iteration. i_parity is the compile-time LDS buffer
        // parity: the iteration consumes buffer i_parity and prefetches the
        // next tile into buffer (1 - i_parity).
        auto kv_iteration = [&](auto i_parity) {
            constexpr index_t kCurBuf  = decltype(i_parity)::value;
            constexpr index_t kNextBuf = 1 - kCurBuf;
            // Tile origin: kv_load_start in the sink phase, the aligned split start after it.
            const index_t kv_tile_start =
                (num_sink_loop > i_total_loops)
                    ? kv_load_start + i_total_loops * kN0
                    : aligned_physical_seqlen_k_start + (i_total_loops - num_sink_loop) * kN0;

            // This barrier covers both reuses below: V is single-buffered (the
            // previous iteration's gemm1 must be done reading it) and buffer
            // kNextBuf is the one the previous iteration's gemm0 consumed.
            block_sync_lds();

            // Issue order is [K(cur) (issued one iteration ago), V, K(next)] and
            // TDM retires in order, hence the waits below:
            //   wait<2> -> K(cur) landed, V and K(next) still in flight
            //   wait<1> -> V landed, K(next) still in flight
            load_tile_tdm(tdm_config_v, v_lds_write_window, v_dram_window); // prefetch load v tile

            // move V tile windows
            advance_v(kN0);

            // The K load below prefetches the NEXT iteration's tile, so the
            // sink->normal jump has to be applied one iteration earlier than it
            // is for V/bias (which are handled at the bottom of the loop).
            //
            // On the final iteration there is no next tile. The prefetch is
            // still issued unconditionally, because the wait counts above are
            // compile-time constants: they only name K(cur) and V correctly
            // while exactly three TDM loads are in flight on every iteration.
            // Skipping the last prefetch would shift them by one, and gemm0
            // would read K before it landed. The window is parked on the last
            // valid tile rather than advanced past the end of K, so the extra
            // load is a redundant re-read into a buffer nobody consumes.
            if constexpr(kHasSink)
            {
                if(i_total_loops == num_sink_loop - 1 && i_total_loops + 1 < num_total_loop)
                {
                    advance_k(aligned_physical_seqlen_k_start - sink_seq_end);
                }
            }
            k_lds_write_window.set_bottom_tensor_view_data_ptr(k_lds_ptr(number<kNextBuf>{}));
            load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);
            advance_k(i_total_loops + 2 < num_total_loop ? kN0 : 0);

            // STAGE 1, QK gemm
            clear_tile(s_acc); // initialize C

            s_wait_tensorcnt_barrier<2>();
            k_lds_read_window.set_bottom_tensor_view_data_ptr(k_lds_ptr(number<kCurBuf>{}));
            if constexpr(kReloadKProgressively)
            {
                // Reload k_tile in place, in groups of four ds_loads dropped
                // into the WMMA sweep right after the fragments they overwrite
                // died. Needs the whole K tile in LDS, which this path stages.
                static_assert(std::is_same_v<Policy, BlockFmhaPipelineQRKSVSTdmDefaultPolicy>);
                using Gemm0  = remove_cvref_t<decltype(gemm_0)>;
                using Window = remove_cvref_t<decltype(k_lds_read_window)>;
                static_assert(Gemm0::MIterPerWarp == 1 || Gemm0::MIterPerWarp == 2);
                static_assert(Gemm0::NIterPerWarp == 4 && Gemm0::KIterPerWarp == 1);
                static_assert(Window::Traits::NumAccess == 8);
                static_assert(Window::Traits::ScalarPerVector == 8);
                static_assert(sizeof(typename Window::Traits::vector_t) == 16);

                auto k_tile = load_tile(k_lds_read_window);
                static_assert(remove_cvref_t<decltype(k_tile)>::get_thread_buffer_size() == 64);

                static_for<0, k0_loops - 1, 1>{}([&](auto i_k0) {
                    move_tile_window(k_lds_read_window, {0, kK0});
                    gemm_0.template RunWithAfterWarp<true>(
                        s_acc,
                        get_slice_tile(
                            q_tile, sequence<0, i_k0 * kK0>{}, sequence<kM0, (i_k0 + 1) * kK0>{}),
                        k_tile,
                        [&k_lds_read_window, &k_tile](auto mIter, auto nIter, auto kIter) {
                            if constexpr(mIter == Gemm0::MIterPerWarp - 1 && kIter == 0 &&
                                         decltype(nIter)::value % 2 == 1)
                            {
                                ReloadProgressiveKFragment(k_lds_read_window, k_tile, nIter);
                            }
                        });
                });
                // Keep the reload-overlap traversal on the final slice too, even
                // though it has no reload, so the schedule stays uniform.
                gemm_0.template RunWithAfterWarp<true>(
                    s_acc,
                    get_slice_tile(q_tile,
                                   sequence<0, (k0_loops - 1) * kK0>{},
                                   sequence<kM0, k0_loops * kK0>{}),
                    k_tile,
                    [](auto, auto, auto) {});
            }
            else
            {
                static_for<0, k0_loops, 1>{}([&](auto i_k0) {
                    auto k_tile = load_tile(k_lds_read_window);
                    gemm_0(s_acc,
                           get_slice_tile(q_tile,
                                          sequence<0, i_k0 * kK0>{},
                                          sequence<kM0, (i_k0 + 1) * kK0>{}),
                           k_tile);
                    if constexpr(i_k0 + 1 < k0_loops)
                        move_tile_window(k_lds_read_window, {0, kK0});
                });
            }
            // rewind the LDS read window to the tile origin for the next iteration
            if constexpr(1 < k0_loops)
                move_tile_window(k_lds_read_window, {0, -kK0 * (k0_loops - 1)});

            // STAGE 2: scale_s, add bias
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                // Pre-scale s_acc by scale_s before adding bias
                tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, s_acc);
                const auto bias_tile = load_tile(bias_dram_window);
                tile_elementwise_inout(
                    [](auto& x, const auto& y) {
                        x += log2e_v<SaccDataType> * type_convert<SaccDataType>(y);
                    },
                    s_acc,
                    bias_tile);
            }
            else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
            {
                const auto current_k_origin = make_tuple(kv_tile_start, 0);
                constexpr auto s_spans      = decltype(s_acc)::get_distributed_spans();
                sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(s_spans[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            s_acc.get_tile_distribution(), make_tuple(idx0, idx1));
                        const auto row = q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                        const auto col = current_k_origin.at(I0) + tile_idx.at(number<1>{});
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        s_acc(i_j_idx) *= scale_s;
                        // position_encoding accepts only logical coordinates
                        position_encoding.update(s_acc(i_j_idx), row, col - kv_l2p_offset);
                    });
                });
            }
            else if constexpr(kHasLogitsSoftCap)
            {
                // No scale_s here: under FAST_EXP2 LogitsSoftCapParams folds both
                // scale_s and log2(e) into its own constants, so this one call
                // leaves s_acc in the same frame the bias branches above do.
                tile_elementwise_inout(
                    [&](auto& x) {
                        x = variant.LogitsTransform(variant_params,
                                                    variant.QueryTransform(variant_params, x),
                                                    block_indices.batch_idx,
                                                    block_indices.qo_head_idx,
                                                    block_indices.kv_head_idx);
                    },
                    s_acc);
            }

            const auto k_origin = make_tuple(kv_tile_start, 0);

            if constexpr(kHasUnevenSplits)
            {
                // The walk's ends are not kN0 aligned: erase the last tile's tail and,
                // under paged-kv, the first tile's head (see aligned_physical_seqlen_k_start).
                // Sink tiles are exempt from the head check: they start at column 0.
                const bool is_last_tile = (i_total_loops == (num_total_loop - 1));
                const bool is_sink_tile = (num_sink_loop > i_total_loops);
                const bool check_head =
                    kIsPagedKV && !is_sink_tile && (i_total_loops == num_sink_loop);
                if(is_last_tile || check_head)
                {
                    set_tile_if(s_acc,
                                -numeric<SMPLComputeDataType>::infinity(),
                                [&,
                                 physical_seqlen_k_start_ = physical_seqlen_k_start,
                                 physical_seqlen_k_end_   = physical_seqlen_k_end,
                                 check_head_              = check_head,
                                 is_last_tile_            = is_last_tile](auto tile_idx) {
                                    const auto col = k_origin.at(I0) + tile_idx.at(I1);

                                    return (check_head_ && col < physical_seqlen_k_start_) ||
                                           (is_last_tile_ && physical_seqlen_k_end_ <= col);
                                });
                }
            }

            if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
            {
                // mask accepts only logical coordinates, do conversion here
                bool need_perpixel_check = mask.IsEdgeTile(
                    q_origin.at(I0), k_origin.at(I0) - kv_l2p_offset, number<kM0>{}, number<kN0>{});
                if(need_perpixel_check)
                {
                    set_tile_if(
                        s_acc, -numeric<SMPLComputeDataType>::infinity(), [&](auto tile_idx) {
                            const auto row = q_origin.at(I0) + tile_idx.at(I0);
                            const auto col = k_origin.at(I0) + tile_idx.at(I1) - kv_l2p_offset;
                            if constexpr(kHasSink)
                                return mask.IsOutOfSinkBound(row, col);
                            else
                                return mask.IsOutOfBound(row, col);
                        });
                }
            }

            // Sink->normal window jump for V/bias, which are consumed in the
            // iteration that issues them. K runs one iteration ahead and was
            // already jumped before its prefetch at the top of the loop.
            if constexpr(kHasSink)
            {
                if(i_total_loops == num_sink_loop - 1)
                {
                    advance_v(aligned_physical_seqlen_k_start - sink_seq_end);
                    // Bias is indexed logically, so its landing column drops
                    // kv_l2p_offset. That puts it on the same column the
                    // non-sink branch computes for bias_n_offset.
                    move_tile_window(
                        bias_dram_window,
                        {0, aligned_physical_seqlen_k_start - sink_seq_end - kv_l2p_offset});
                }
            }

            // K's dram window is advanced right after its prefetch at the top of
            // the loop, so only bias needs advancing here.
            move_tile_window(bias_dram_window, {0, kN0});

            // Gemm1
            auto s_new = cast_tile<SMPLComputeDataType>(s_acc); // S{j}

            auto m_local = block_tile_reduce<SMPLComputeDataType>(
                s_new,
                sequence<1>{},
                f_max,
                -numeric<SMPLComputeDataType>::infinity()); // m_local = rowmax(S{j})
            block_tile_reduce_sync(m_local, f_max, bool_constant<false>{});

            const auto m_old = m; // m{j-1}
            tile_elementwise_inout(
                [](auto& e0, auto e1, auto e2) { e0 = max(e1, e2); }, m, m_old, m_local); // m{j}

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

            constexpr auto p_spans = decltype(p_compute)::get_distributed_spans();
            sweep_tile_span(p_spans[I0], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                auto row_max         = scale_s * get_validated_m(m[i_idx]);
                sweep_tile_span(p_spans[I1], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                 BiasEnum == BlockAttentionBiasEnum::ALIBI)
                    {
                        p_compute(i_j_idx) = exp2(s_new[i_j_idx] - get_validated_m(m[i_idx]));
                    }
                    else
                    {
                        if constexpr(kHasLogitsSoftCap)
                        {
                            p_compute(i_j_idx) = exp2(s_new[i_j_idx] - get_validated_m(m[i_idx]));
                        }
                        else
                        {
                            p_compute(i_j_idx) = exp2(scale_s * s_new[i_j_idx] - row_max);
                        }
                    }
                });
            });

            auto rowsum_p = block_tile_reduce<SMPLComputeDataType>(
                p_compute, sequence<1>{}, f_sum, SMPLComputeDataType{0}); // rowsum(Pcompute{j})

            block_tile_reduce_sync(rowsum_p, f_sum, bool_constant<false>{});

            auto p_tile = MakePForGemm1<decltype(gemm_1)>(p_compute);

            // l{j}, Oacc{j}
            constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
            sweep_tile_span(o_spans[I0], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                const auto tmp       = [&]() {
                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                 BiasEnum == BlockAttentionBiasEnum::ALIBI)
                    {
                        return exp2(m_old[i_idx] - get_validated_m(m[i_idx]));
                    }
                    else
                    {
                        if constexpr(kHasLogitsSoftCap)
                        {
                            return exp2(m_old[i_idx] - get_validated_m(m[i_idx]));
                        }
                        else
                        {
                            auto row_max = scale_s * get_validated_m(m[i_idx]);
                            return exp2(scale_s * m_old[i_idx] - row_max);
                        }
                    }
                }();
                l(i_idx) = tmp * l[i_idx] + rowsum_p[i_idx];
                sweep_tile_span(o_spans[I1], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    o_acc(i_j_idx) *= tmp;
                });
            });

            // V is on the tensorcnt counter (load_tile_tdm). Wait for V's TDM
            // write to commit before ds_load_tr reads it; the next iteration's
            // K prefetch is still outstanding, hence <1> rather than <0>.
            s_wait_tensorcnt_barrier<1>();

            auto v_tile = load_tile_transpose(v_lds_read_window);

            if constexpr(1 < k1_loops)
            {
                static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
                    gemm_1(o_acc,
                           get_slice_tile(p_tile,
                                          sequence<0, i_k1 * kK1>{},
                                          sequence<kM0, (i_k1 + 1) * kK1>{}),
                           v_tile);

                    // loop over along the [V]alue Sequence length
                    move_tile_window(v_lds_read_window, {kK1, 0});
                    v_tile = load_tile_transpose(v_lds_read_window);
                });
                // move back to the origin
                move_tile_window(v_lds_read_window, {-kK1 * (k1_loops - 1), 0});
            }

            gemm_1(o_acc,
                   get_slice_tile(p_tile,
                                  sequence<0, (k1_loops - 1) * kK1>{},
                                  sequence<kM0, k1_loops * kK1>{}),
                   v_tile);
        };

        // Unrolled by two so that each half addresses its K LDS buffer with a
        // compile-time constant (see the k_lds_ptr comment above). Written as
        // an early-exit do/while rather than "pairs + tail" so the body is
        // instantiated exactly twice, not three times. num_total_loop >= 1 is
        // guaranteed by the caller, and i_total_loops is even whenever the
        // parity-0 half runs.
        do
        {
            kv_iteration(number<0>{});
            if(++i_total_loops >= num_total_loop)
                break;
            kv_iteration(number<1>{});
        } while(++i_total_loops < num_total_loop);

        // The last iteration issued a K prefetch that nothing consumes; drain it
        // before leaving, so the LDS region this pipeline was handed is not
        // still a TDM destination once the caller reuses it.
        s_wait_tensorcnt_barrier<0>();

        if constexpr(kStoreLSE)
        {
            // store lse acc
            auto lse_acc = make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

            constexpr auto lse_acc_spans = decltype(lse_acc)::get_distributed_spans();
            sweep_tile_span(lse_acc_spans[I0], [&, m_ = m, l_ = l](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             BiasEnum == BlockAttentionBiasEnum::ALIBI)
                {
                    lse_acc(i_idx) = m_[i_idx] / C_LOG2E + log(l_[i_idx]);
                }
                else
                {
                    if constexpr(kHasLogitsSoftCap)
                    {
                        lse_acc(i_idx) = m_[i_idx] / C_LOG2E + log(l_[i_idx]);
                    }
                    else
                    {
                        lse_acc(i_idx) = m_[i_idx] * scale_s / C_LOG2E + log(l_[i_idx]);
                    }
                }
            });

            store_tile(lse_acc_dram_window_tmp, lse_acc);
        }

        // finally, O
        constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();

        sweep_tile_span(o_spans[I0], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            auto tmp             = [&]() {
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

        return o_acc;
    }

    // Split-kv entry point. The parameter list matches the non-fp8 operator() of
    // BlockFmhaFwdSplitKVPipelineQRKSVS, so the kernel calls both pipelines
    // through the same expression.
    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowLengths,
              typename KPageBlockNavigator,
              typename VDramBlockWindowLengths,
              typename VPageBlockNavigator,
              typename BiasDramBlockWindowTmp,
              typename LSEaccDramBlockWindowTmp,
              typename PositionEncoding,
              typename AttentionVariantParams,
              typename BlockIndices>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,         // M0*K0 tile
               const KDramBlockWindowLengths& k_dram_block_window_lengths, // N0*K0 tile
               const KPageBlockNavigator& k_page_block_navigator,
               const VDramBlockWindowLengths& v_dram_block_window_lengths, // N1*K1 tile
               const VPageBlockNavigator& v_page_block_navigator,
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
               LSEaccDramBlockWindowTmp& lse_acc_dram_block_window_tmp,  // M0*1 tile
               index_t num_splits,
               index_t i_split,
               FmhaMask mask,
               PositionEncoding position_encoding,
               float scale_s,
               const AttentionVariant& variant,
               const AttentionVariantParams& variant_params,
               const BlockIndices& block_indices,
               index_t kv_l2p_offset, // logical-to-physical offset of seqlen_k coordinate
               void* smem_ptr,
               float sink_v) const
    {
        // The navigators are forwarded as-is: the callee builds its K/V windows
        // through them, so a page turn and a plain window step are the same call.
        return run_single_v_double_k_lds_buffer(q_dram_block_window_tmp,
                                                k_dram_block_window_lengths,
                                                k_page_block_navigator,
                                                v_dram_block_window_lengths,
                                                v_page_block_navigator,
                                                bias_dram_block_window_tmp,
                                                lse_acc_dram_block_window_tmp,
                                                num_splits,
                                                i_split,
                                                mask,
                                                position_encoding,
                                                scale_s,
                                                variant,
                                                variant_params,
                                                block_indices,
                                                kv_l2p_offset,
                                                smem_ptr,
                                                sink_v);
    }
};

} // namespace ck_tile
