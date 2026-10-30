// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_pingpong_policy.hpp"

namespace ck_tile {

// qr_tdm with a single V LDS buffer and a whole-tile K ping/pong, for gfx1250 fp16/bf16:
// trip i reads K buffer (i & 1) while the TDM fills the other with tile i + 1, and V
// streams through one buffer under the QK gemm. It shares qr_tdm's problem, gemms and
// register layouts; only the LDS layout (see BlockFmhaPipelineQRKSVSTdmPingPongPolicy)
// and the main loop differ.
template <typename Problem_, typename Policy_ = BlockFmhaPipelineQRKSVSTdmPingPongPolicy>
struct BlockFmhaPipelineQRKSVSTdmPingPong : BlockFmhaPipelineQRKSVSTdm<Problem_, Policy_>
{
    using Base = BlockFmhaPipelineQRKSVSTdm<Problem_, Policy_>;

    using typename Base::BlockFmhaShape;
    using typename Base::FmhaMask;
    using typename Base::KDataType;
    using typename Base::LSEDataType;
    using typename Base::PDataType;
    using typename Base::Policy;
    using typename Base::Problem;
    using typename Base::QDataType;
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

    using Base::ReduceRowSync;

    static constexpr const char* name = "qr_tdm_pingpong";

    // the kernel paths qr_tdm takes by name
    static constexpr bool kUsesUntransposedVKernelPath = true;
    static constexpr bool kUsesTdmAffineDramPath       = true;

    static constexpr bool kKLoadOnce = true;

    static_assert(detail::is_qr_tdm_pingpong_qkv_v<Problem> && !kQuantized,
                  "qr_tdm_pingpong: fp16/bf16 Q/K/V without quantization scales");
    static_assert(!Problem::kUseDoubleKVLdsBuffer && !Problem::kProgressiveDsLoadK,
                  "qr_tdm_pingpong has its own K/V buffering");
    static_assert(kSubQKHeaddim == kQKHeaddim, "qr_tdm_pingpong stages whole K rows");
    static_assert(kNWarp == 1, "qr_tdm_pingpong keeps S in registers: gemm_0 warps all on M");

    // Row sum of P as a pairwise tree per thread, then the cross-lane sync. The
    // serial fold of block_tile_reduce is a dependent v_add_f32 chain the
    // scheduler interleaves with the PV WMMAs, paying v_nop hazards for it; the
    // tree has depth log2(n) and pairs operands for v_pk_add_f32.
    template <typename PComputeTensor>
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

                    rowsum(make_tuple(idx0)) = v[number<0>{}];
                },
                row_lengths);
        });

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

    // Initial row max and row sum: a finite sink pre-seeds them.
    template <typename RowTile>
    CK_TILE_DEVICE static void InitRowMaxSum(RowTile& m, RowTile& l, float sink_v, float scale_s)
    {
        if(__builtin_isinf_sign(sink_v) >= 0)
        {
#if CK_TILE_FMHA_FWD_FAST_EXP2
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI ||
                         BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS || kHasLogitsSoftCap)
                set_tile(m, sink_v * scale_s * C_LOG2E);
            else
                set_tile(m, sink_v * C_LOG2E);
#else
            ignore = scale_s;
            set_tile(m, sink_v);
#endif
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
                                                   float scale_s)
    {
        auto lse_acc = make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());
        if(__builtin_isinf_sign(sink_v) >= 0)
            set_tile(lse_acc, SMPLComputeDataType{sink_v * scale_s});
        else
            set_tile(lse_acc, -numeric<SMPLComputeDataType>::infinity());
        store_tile(lse_acc_dram_window, lse_acc);
    }

    // STAGE 2 of the QK result: scale_s and the bias (elementwise or ALiBi) for the
    // K tile starting at k_start. Without bias, scale_s is applied in the exponent.
    // position_encoding is taken by value; Alibi::update does not change it.
    template <typename SaccTile,
              typename BiasDramWindow,
              typename QOrigin,
              typename PositionEncoding>
    CK_TILE_DEVICE static void ApplyScaleAndBias(SaccTile& s_acc,
                                                 const BiasDramWindow& bias_dram_window,
                                                 const QOrigin& q_origin,
                                                 index_t k_start,
                                                 float scale_s,
                                                 PositionEncoding position_encoding)
    {
        if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
        {
            // Pre-scale s_acc by scale_s before adding bias
            tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, s_acc);
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
                    s_acc(i_j_idx) *= scale_s;
                    position_encoding.update(s_acc(i_j_idx), row, col);
                });
            });
        }
        else
        {
            ignore = bias_dram_window;
            ignore = q_origin;
            ignore = k_start;
            ignore = scale_s;
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
                set_tile_if(s_acc, -numeric<SMPLComputeDataType>::infinity(), [&](auto tile_idx) {
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
            if(mask.IsEdgeTile(q_origin.at(I0), k_start, number<kM0>{}, number<kN0>{}))
            {
                set_tile_if(s_acc, -numeric<SMPLComputeDataType>::infinity(), [&](auto tile_idx) {
                    const auto row = q_origin.at(I0) + tile_idx.at(I0);
                    const auto col = k_start + tile_idx.at(I1);
                    if constexpr(kHasSink)
                        return mask.IsOutOfSinkBound(row, col);
                    else
                        return mask.IsOutOfBound(row, col);
                });
            }
        }
    }

    // LSE from the final row max m (log2 domain) and row sum l.
    template <typename LSEaccDramBlockWindowTmp, typename RowTile>
    CK_TILE_DEVICE static void StoreLse(LSEaccDramBlockWindowTmp& lse_acc_dram_window,
                                        const RowTile& m,
                                        const RowTile& l,
                                        float scale_s)
    {
        auto lse_acc = make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

        constexpr auto lse_acc_spans = decltype(lse_acc)::get_distributed_spans();
        sweep_tile_span(lse_acc_spans[I0], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                         BiasEnum == BlockAttentionBiasEnum::ALIBI || kHasLogitsSoftCap)
                lse_acc(i_idx) = m[i_idx] / C_LOG2E + log(l[i_idx]);
            else
                lse_acc(i_idx) = m[i_idx] * scale_s / C_LOG2E + log(l[i_idx]);
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
              typename PositionEncoding>
    CK_TILE_HOST_DEVICE auto operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,
                                        const KDramBlockWindowTmp& k_dram_block_window_tmp,
                                        const VDramBlockWindowTmp& v_dram_block_window_tmp,
                                        const BiasDramBlockWindowTmp& bias_dram_block_window_tmp,
                                        LSEaccDramBlockWindowTmp& lse_acc_dram_window_tmp,
                                        FmhaMask mask,
                                        PositionEncoding position_encoding,
                                        float scale_s,
                                        void* smem_arena,
                                        float sink_v) const
    {
        using Layout = typename Policy::template LdsArenaLayout<Problem>;

        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

        // Q, K and V all go through the TDM (box-major LDS write); V is read
        // back with ds_load_tr (transposed). V dram window keeps (kN1, kK1) shape.
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

        // init M, L (sink-aware: when sink_v is finite, pre-seed m/l). l sums the 16-bit
        // P that gemm_1 consumes and normalises O; the LSE needs the fp32 sum, l_lse.
        auto m                      = MLBlockTileType{};
        auto l                      = MLBlockTileType{};
        [[maybe_unused]] auto l_lse = MLBlockTileType{};
        // row max o_acc and l are normalised to; trails m by up to
        // kRebaseThreshold (see the lazy rebasing in the main loop)
        auto m_base = MLBlockTileType{};

        clear_tile(o_acc);
        InitRowMaxSum(m, l, sink_v, scale_s);
        m_base = m;
        if constexpr(kStoreLSE)
            l_lse = l;

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
                    StoreLseOfEmptyRows(lse_acc_dram_window_tmp, m, sink_v, scale_s);

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

        // V tile in LDS
        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp,
                             {kv_load_start, 0},
                             Policy::template MakeVDramTileDistribution<Problem>());

        // written by the TDM, read back with ds_load_tr through the same row-major descriptor
        auto v_lds_view = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<VDataType*>(static_cast<char*>(smem_arena) + Layout::kV0Offset),
            Policy::template MakeVLdsBlockDescriptor<Problem>());
        auto v_lds_write_window = make_tile_window(
            v_lds_view, Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        auto v_lds_read_window =
            make_tile_window(v_lds_view,
                             make_tuple(number<kK1>{}, number<kN1>{}),
                             {0, 0},
                             Policy::template MakeVRegTileDistribution<Problem>());

        // retire Q only; K, issued after it, stays in flight
        s_wait_tensorcnt_barrier<1>();
        auto q_tile = load_tile(q_lds_read_window);

        const index_t num_total_loop =
            integer_divide_ceil(physical_seqlen_k_end - physical_seqlen_k_start, kN0) +
            num_sink_loop;

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
            // K_i, so the count is <0>; dscnt 0 also retires the Q reads.
            s_wait_tensorcnt_barrier<0, 0>();

            load_tile_tdm(tdm_config_v, v_lds_write_window, v_dram_window);
            move_tile_window(v_dram_window, {kN0, 0});

            // STAGE 1, QK gemm
            clear_tile(s_acc); // initialize C

            // K is whole in LDS; the kK0 slices are walked by the read window.
            // V stays in flight under this gemm until the wait before gemm_1.
            k_lds_read_window.set_bottom_tensor_view_data_ptr(k_read_ptr);
            auto k_tile = load_tile(k_lds_read_window);

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

            // Two K reads per WMMA keep an LDS read in flight without a deep
            // register buffer; left alone the scheduler waits on dscnt before
            // every WMMA.
            static_for<0, k0_loops, 1>{}([&](auto) {
                __builtin_amdgcn_sched_group_barrier(0x100, 2, 0); // DS read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // WMMA
            });

            // Sink-aware: in the sink phase tiles start at 0, in the normal phase at
            // physical_seqlen_k_start.
            const index_t kv_tile_start =
                (num_sink_loop > i_total_loops)
                    ? kv_load_start + i_total_loops * kN0
                    : physical_seqlen_k_start + (i_total_loops - num_sink_loop) * kN0;

            // STAGE 2: scale_s, add bias, mask
            ApplyScaleAndBias(
                s_acc, bias_dram_window, q_origin, kv_tile_start, scale_s, position_encoding);
            MaskScores(s_acc,
                       mask,
                       q_origin,
                       kv_tile_start,
                       i_total_loops == num_total_loop - 1,
                       physical_seqlen_k_end);

            // Sink->normal window jump: at the boundary between sink region
            // and normal region, jump K/V/bias dram windows forward.
            if constexpr(kHasSink)
            {
                if(i_total_loops == num_sink_loop - 1)
                {
                    move_tile_window(k_dram_window, {physical_seqlen_k_start - sink_seq_end, 0});
                    move_tile_window(v_dram_window, {physical_seqlen_k_start - sink_seq_end, 0});
                    move_tile_window(bias_dram_window, {0, physical_seqlen_k_start - sink_seq_end});
                }
            }

            // move K and bias tile windows after current status checked
            // prefetch next-tile along [K]ey sequence length dimension
            move_tile_window(k_dram_window, {kN0, 0});
            move_tile_window(bias_dram_window, {0, kN0});

            // Next K, into the buffer nobody reads this trip, so no barrier.
            // Issued on the last trip too: the wait before gemm_1 counts on it
            // being behind V on the counter.
            k_lds_write_window.set_bottom_tensor_view_data_ptr(k_write_ptr);
            load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);

            const auto s_new = cast_tile<SMPLComputeDataType>(s_acc); // S{j}

            auto m_local = block_tile_reduce<SMPLComputeDataType>(
                s_new,
                sequence<1>{},
                f_max,
                -numeric<SMPLComputeDataType>::infinity()); // m_local = rowmax(S{j})
            ReduceRowSync(m_local, f_max);

            tile_elementwise_inout([](auto& e0, auto e1) { e0 = max(e0, e1); }, m, m_local); // m{j}

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
                        if constexpr(kStoreLSE)
                            l_lse(i_idx) *= tmp;
                        m_base(i_idx) = m[i_idx];
                        sweep_tile_span(o_spans[I1],
                                        [&](auto idx1) { o_acc(make_tuple(idx0, idx1)) *= tmp; });
                    }
                });
            }

            constexpr auto p_spans = decltype(p_compute)::get_distributed_spans();
            sweep_tile_span(p_spans[I0], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                auto row_max         = get_row_exp(m_base[i_idx]);
                sweep_tile_span(p_spans[I1], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                 BiasEnum == BlockAttentionBiasEnum::ALIBI)
                    {
                        p_compute(i_j_idx) = exp2(s_new[i_j_idx] - row_max);
                    }
                    else
                    {
                        if constexpr(kHasLogitsSoftCap)
                        {
                            p_compute(i_j_idx) = exp2(s_new[i_j_idx] - row_max);
                        }
                        else
                        {
                            p_compute(i_j_idx) = exp2(scale_s * s_new[i_j_idx] - row_max);
                        }
                    }
                });
            });

            [[maybe_unused]] int32_t p_scale = 0;
            auto p_tile = Base::template MakePForGemm1<decltype(gemm_1)>(p_compute, p_scale);

            // l{j}; o_acc is already in p's m_base frame.
            AccumulateRowSumOnMatrixUnit(l, p_tile);
            if constexpr(kStoreLSE)
            {
                const auto rowsum_p = BlockRowSum(p_compute); // rowsum(Pcompute{j})
                sweep_tile_span(o_spans[I0], [&](auto idx0) {
                    constexpr auto i_idx = make_tuple(idx0);
                    l_lse(i_idx) += rowsum_p[i_idx];
                });
            }

            // In flight: V, then the next K. <1> retires V and leaves K running
            // under the PV gemm.
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

            // two V reads per WMMA, as for K above
            using Gemm1T = remove_cvref_t<decltype(gemm_1)>;
            static_for<0, Gemm1T::MIterPerWarp * Gemm1T::NIterPerWarp * k1_loops, 1>{}([&](auto) {
                __builtin_amdgcn_sched_group_barrier(0x100, 2, 0); // DS read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // WMMA
            });

        } while(++i_total_loops < num_total_loop);

        // retire the K tile issued past the last one consumed
        s_wait_tensorcnt<0>();

        if constexpr(kStoreLSE)
        {
            // l_lse is normalised to m_base, validated as in the loop (a fully
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
            StoreLse(lse_acc_dram_window_tmp, m_lse, l_lse, scale_s);
        }

        NormalizeO(o_acc, l);
        return o_acc;
    }

    // The kernel's other LDS argument order (see FmhaFwdKernel).
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
                                        void* smem_arena) const
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
                       sink_v);
    }
};

} // namespace ck_tile
