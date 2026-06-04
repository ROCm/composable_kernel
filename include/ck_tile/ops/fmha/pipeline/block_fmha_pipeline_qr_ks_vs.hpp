// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/block/block_dropout.hpp"
#include "ck_tile/ops/fmha/block/cast_tile_mx.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_default_policy.hpp"
#include "ck_tile/ops/gemm/warp/warp_wmma_gemm_gfx11_utils.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

// This pipeline is qkv all located in LDS
template <typename Problem_,
          typename Policy_         = BlockFmhaPipelineQRKSVSDefaultPolicy,
          bool PaddedVecLoadStore_ = false>
struct BlockFmhaPipelineQRKSVS
{
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
    using QScaleDataType        = remove_cvref_t<typename Problem::QScaleDataType>;
    using KScaleDataType        = remove_cvref_t<typename Problem::KScaleDataType>;
    using VScaleDataType        = remove_cvref_t<typename Problem::VScaleDataType>;
    using PScaleDataType        = remove_cvref_t<typename Problem::PScaleDataType>;
    using AttentionVariant      = remove_cvref_t<typename Problem::AttentionVariant>;
    using FmhaMask              = remove_cvref_t<typename Problem::FmhaMask>;

    template <typename T>
    using has_partial_k_support = decltype(T::kSupportsPartialK);
    template <typename T>
    using has_partial_n_support = decltype(T::kSupportsPartialN);

    using BlockFmhaShape             = remove_cvref_t<typename Problem::BlockFmhaShape>;
    using VLayout                    = remove_cvref_t<typename BlockFmhaShape::VLayout>;
    static constexpr bool kQLoadOnce = true; // if q_tile load whole block length (hdim) at once
    static_assert(kQLoadOnce == Policy::QLoadOnce);

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kM0           = BlockFmhaShape::kM0;
    static constexpr index_t kN0           = BlockFmhaShape::kN0;
    static constexpr index_t kK0           = BlockFmhaShape::kK0;
    static constexpr index_t kN1           = BlockFmhaShape::kN1;
    static constexpr index_t kK1           = BlockFmhaShape::kK1;
    static constexpr index_t kQKHeaddim    = BlockFmhaShape::kQKHeaddim;
    static constexpr index_t kSubQKHeaddim = BlockFmhaShape::kSubQKHeaddim;

    static_assert(kSubQKHeaddim <= 256, "hdim bigger than 256 is not suitable for this pipeline!");

    static constexpr bool kIsGroupMode        = Problem::kIsGroupMode;
    static constexpr bool kPadSeqLenQ         = Problem::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK         = Problem::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ        = Problem::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV        = Problem::kPadHeadDimV;
    static constexpr bool kHasLogitsSoftCap   = Problem::kHasLogitsSoftCap;
    static constexpr auto BiasEnum            = Problem::BiasEnum;
    static constexpr bool kStoreLSE           = Problem::kStoreLSE;
    static constexpr bool kHasDropout         = Problem::kHasDropout;
    static constexpr auto QScaleEnum          = Problem::QScaleEnum;
    static constexpr bool kHasSink            = Problem::kHasSink;
    static constexpr bool kPaddedVecLoadStore = PaddedVecLoadStore_;
    static constexpr bool kUseHdimTailArgs    = kPadHeadDimQ || kPadHeadDimV;

    static constexpr ck_tile::index_t kQKScaleGranularity = Problem::kQKScaleGranularity;
    static constexpr ck_tile::index_t kVScaleGranularity  = Problem::kVScaleGranularity;

    // For BLOCKSCALE: shift value for exp2(x + shift) to scale P to [0, 2^shift]
    static constexpr float OCP_FP8_SHIFT  = 8.0f;
    static constexpr float FNUZ_FP8_SHIFT = 7.0f;

    static constexpr uint32_t DS_READ = 0x100; // Barrier for DS (data share) read
    static constexpr uint32_t MFMA    = 0x008; // Barrier for MFMA (matrix multiply-accumulate)

    static_assert((CK_TILE_FMHA_FWD_FAST_EXP2 &&
                   (kHasLogitsSoftCap && Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS ||
                    !kHasLogitsSoftCap)) ||
                  (!CK_TILE_FMHA_FWD_FAST_EXP2 && !kHasLogitsSoftCap));
    static_assert(!kPaddedVecLoadStore || (kPadHeadDimQ && kPadHeadDimV),
                  "padded vector load/store fast path only applies to padded head-dim kernels");

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
    static constexpr index_t kAlignmentQ = (kPadHeadDimQ && !kPaddedVecLoadStore)
                                               ? numeric_traits<QDataType>::PackedSize
                                               : Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK = (kPadHeadDimQ && !kPaddedVecLoadStore)
                                               ? numeric_traits<KDataType>::PackedSize
                                               : Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV = []() {
        if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            return (kPadHeadDimV && !kPaddedVecLoadStore)
                       ? 1
                       : Policy::template GetAlignmentV<Problem>();
        else
            return kPadSeqLenK ? numeric_traits<VDataType>::PackedSize
                               : Policy::template GetAlignmentV<Problem>();
    }();

    static constexpr index_t kAlignmentO =
        (kPadHeadDimV && !kPaddedVecLoadStore) ? 1 : Policy::template GetAlignmentO<Problem>();
    static constexpr index_t kAlignmentBias =
        kPadSeqLenK ? 1 : Policy::template GetAlignmentBias<Problem>();
    static constexpr index_t kAlignmentRandVal =
        kPadSeqLenK ? 1 : Policy::template GetAlignmentRandVal<Problem>();

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
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
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

    static constexpr const char* name = "qr";

    using DropoutType = std::conditional_t<kHasDropout, BlockDropout, NullBlockDropout>;

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename RandValDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename QElementFunction,
              typename KElementFunction,
              typename VElementFunction,
              typename BiasElementFunction,
              typename LSEElementFunction,
              typename SAccElementFunction,
              typename PComputeElementFunction,
              typename OAccElementFunction,
              typename PositionEncoding,
              typename AttentionVariantParams,
              typename BlockIndices,
              typename QScaleDramBlockWindowTmp,
              typename KScaleDramBlockWindowTmp,
              typename VScaleDramBlockWindowTmp>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp, // M0*K0 tile
               const QElementFunction& q_element_func,
               const KDramBlockWindowTmp& k_dram_block_window_tmp, // N0*K0 tile
               const KElementFunction& k_element_func,
               const VDramBlockWindowTmp& v_dram_block_window_tmp, // N1*K1 tile
               const VElementFunction& v_element_func,
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
               const BiasElementFunction& bias_element_func,
               RandValDramBlockWindowTmp& randval_dram_block_window_tmp,
               LSEDramBlockWindowTmp& lse_dram_window_tmp, // M0*1 tile
               const LSEElementFunction& lse_element_func,
               const SAccElementFunction& s_acc_element_func,
               const PComputeElementFunction& p_compute_element_func,
               const OAccElementFunction& o_acc_element_func,
               FmhaMask mask,
               PositionEncoding position_encoding,
               float scale_s,
               const AttentionVariant& variant,
               const AttentionVariantParams& variant_params,
               const BlockIndices& block_indices,
               void* smem_ptr,
               DropoutType& dropout,
               const float* k_descale_ptr,
               const float* v_descale_ptr,
               const index_t block_scale_size_kv,
               const QScaleDramBlockWindowTmp&
                   q_scale_dram_block_window_tmp, // M0*(K0/kQKScaleGranularity) tile
               const KScaleDramBlockWindowTmp&
                   k_scale_dram_block_window_tmp, // N0*(K0/kQKScaleGranularity) tile
               const VScaleDramBlockWindowTmp&
                   v_scale_dram_block_window_tmp, // N1*(K1/kVScaleGranularity) tile
               const float sink_v,
               const index_t valid_k0_loops,
               const index_t valid_last_k0_length,
               const index_t valid_n1_length) const
    {
        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kSubQKHeaddim ==
                              QDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kK0 == KDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kN1 == VDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kK1 == VDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kM0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::MX)
        {
            static_assert(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::ColumnMajor>);

            static_assert(
                std::is_same_v<QScaleDataType,
                               remove_cvref_t<typename QScaleDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KScaleDataType,
                               remove_cvref_t<typename KScaleDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VScaleDataType,
                               remove_cvref_t<typename VScaleDramBlockWindowTmp::DataType>>);
            static_assert(kM0 == QScaleDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kSubQKHeaddim ==
                              QScaleDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] *
                                  kQKScaleGranularity &&
                          kN0 == KScaleDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kK0 == KScaleDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] *
                                     kQKScaleGranularity &&
                          kN1 == VScaleDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kK1 == VScaleDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] *
                                     kVScaleGranularity);
        }

        // K tile in LDS
        KDataType* k_lds_ptr = static_cast<KDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQ<Problem>()));
        auto k_lds           = make_tensor_view<address_space_enum::lds>(
            k_lds_ptr, Policy::template MakeKLdsBlockDescriptor<Problem>());
        auto k_lds_window =
            make_tile_window(k_lds, make_tuple(number<kN0>{}, number<kK0>{}), {0, 0});

        // V tile in LDS
        auto v_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<VDataType*>(smem_ptr),
            Policy::template MakeVLdsBlockDescriptor<Problem>());
        auto v_lds_window = make_tile_window(
            v_lds, Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        // Block GEMM
        constexpr auto gemm_0                      = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1                      = Policy::template GetKVBlockGemm<Problem>();
        using BlockGemm0                           = remove_cvref_t<decltype(gemm_0)>;
        using BlockGemm1                           = remove_cvref_t<decltype(gemm_1)>;
        constexpr bool kBlockGemm0SupportsPartialK = [] {
            if constexpr(ck_tile::is_detected<has_partial_k_support, BlockGemm0>::value)
                return static_cast<bool>(BlockGemm0::kSupportsPartialK);
            else
                return false;
        }();
        constexpr bool kBlockGemm1SupportsPartialN = [] {
            if constexpr(ck_tile::is_detected<has_partial_n_support, BlockGemm1>::value)
                return static_cast<bool>(BlockGemm1::kSupportsPartialN);
            else
                return false;
        }();

        constexpr auto gemm_0_config =
            BlockGemm0::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using Gemm0WarpGemm           = remove_cvref_t<decltype(gemm_0_config.template at<0>())>;
        constexpr index_t kGemm0WarpK = Gemm0WarpGemm::kK;
        constexpr index_t kGemm0KItersPerBlock = kK0 / kGemm0WarpK;
        constexpr bool kUsePartialKForGemm0Tail =
            kPadHeadDimQ && kBlockGemm0SupportsPartialK && (kGemm0KItersPerBlock > 1);

        auto q_dram_window = make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
                                              q_dram_block_window_tmp.get_window_lengths(),
                                              q_dram_block_window_tmp.get_window_origin(),
                                              Policy::template MakeQRegTileDistribution<Problem>());

        auto q = load_tile(q_dram_window);

        using SaccBlockTileType = decltype(gemm_0.MakeCBlockTile());
        auto s_acc              = SaccBlockTileType{};

        // reduction function for softmax
        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        // infer Sacc, S, P, M, L, Oacc type
        using SBlockTileType = decltype(cast_tile<SMPLComputeDataType>(s_acc));

        using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
            SBlockTileType{}, sequence<1>{}, f_max, SMPLComputeDataType{0}));

        using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());

        // init Oacc, M, L
        auto o_acc = OaccBlockTileType{};
        auto m     = MLBlockTileType{};
        auto l     = MLBlockTileType{};

        clear_tile(o_acc);
        if(__builtin_isinf_sign(sink_v) >= 0)
        {
#if CK_TILE_FMHA_FWD_FAST_EXP2
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI ||
                         BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                set_tile(m, sink_v * scale_s * C_LOG2E);
            else
                set_tile(m, sink_v * C_LOG2E);
#else
            set_tile(m, sink_v);
#endif
            set_tile(l, SMPLComputeDataType{1.0f});
        }
        else
        {
            set_tile(m, -numeric<SMPLComputeDataType>::infinity());
            clear_tile(l);
        }
        const auto q_origin = q_dram_window.get_window_origin();

        const auto tile_range_result = [&mask, &q_origin]() {
            if constexpr(kHasSink)
                return mask.GetSinkTileRangeAlongX(
                    q_origin.at(number<0>{}), number<kM0>{}, number<kN0>{});
            else
            {
                auto [start, end] =
                    mask.GetTileRangeAlongX(q_origin.at(number<0>{}), number<kM0>{}, number<kN0>{});
                return ck_tile::make_tuple(0, start, end);
            }
        }();
        const auto sink_seq_end   = tile_range_result.get(ck_tile::number<0>{});
        const auto seqlen_k_start = tile_range_result.get(ck_tile::number<1>{});
        const auto seqlen_k_end   = tile_range_result.get(ck_tile::number<2>{});

        const auto kv_load_start = (sink_seq_end == 0 && seqlen_k_start > 0) ? seqlen_k_start : 0;
        const auto num_sink_loop = integer_divide_ceil(sink_seq_end, kN0);
        const auto num_total_loop =
            integer_divide_ceil(seqlen_k_end - seqlen_k_start, kN0) + num_sink_loop;

        // check early exit if no work to do
        if constexpr(FmhaMask::IsMasking || kPadSeqLenK)
        {
            if(num_total_loop <= 0)
            {
                if constexpr(kStoreLSE)
                {
                    auto lse =
                        make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

                    if(__builtin_isinf_sign(sink_v) >= 0)
                    {
                        set_tile(lse, SMPLComputeDataType{sink_v * scale_s});
                    }
                    else
                    {
                        set_tile(lse, -numeric<SMPLComputeDataType>::infinity());
                    }

                    store_tile(lse_dram_window_tmp, tile_elementwise_in(lse_element_func, lse));
                }

                // Note: here occ are all cleard, return it
                // Note: q loaded but no fence, ignore it.
                return o_acc;
            }
        }

        auto k_dram_block_window =
            make_tile_window(k_dram_block_window_tmp.get_bottom_tensor_view(),
                             k_dram_block_window_tmp.get_window_lengths(),
                             {kv_load_start, 0});

        const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();
        auto bias_dram_window =
            make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                             bias_dram_block_window_tmp.get_window_lengths(),
                             {bias_origin.at(number<0>{}), kv_load_start}, // M/N
                             Policy::template MakeBiasDramTileDistribution<decltype(gemm_0)>());

        auto randval_dram_window = dropout.template MakeRandvalDramWindow<decltype(gemm_0)>(
            randval_dram_block_window_tmp, kv_load_start);

        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp.get_bottom_tensor_view(),
                             v_dram_block_window_tmp.get_window_lengths(),
                             {0, kv_load_start}, // TODO: hdim split?
                             Policy::template MakeVDramTileDistribution<Problem>());

        auto q_tile = tile_elementwise_in(q_element_func, q);

        auto q_scale = [&] {
            if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::MX)
            {
                auto q_scale_dram_window =
                    make_tile_window(q_scale_dram_block_window_tmp.get_bottom_tensor_view(),
                                     q_scale_dram_block_window_tmp.get_window_lengths(),
                                     q_scale_dram_block_window_tmp.get_window_origin(),
                                     Policy::template MakeQScaleRegTileDistribution<Problem>());
                return load_tile(q_scale_dram_window);
            }
            else
            {
                return null_tensor{};
            }
        }();
        auto k_scale_dram_block_window = [&] {
            if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::MX)
            {
                return make_tile_window(k_scale_dram_block_window_tmp.get_bottom_tensor_view(),
                                        k_scale_dram_block_window_tmp.get_window_lengths(),
                                        {seqlen_k_start, 0});
            }
            else
            {
                return make_null_tile_window(make_tuple());
            }
        }();
        auto v_scale_dram_window = [&] {
            if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::MX)
            {
                return make_tile_window(v_scale_dram_block_window_tmp.get_bottom_tensor_view(),
                                        v_scale_dram_block_window_tmp.get_window_lengths(),
                                        {0, seqlen_k_start / kVScaleGranularity},
                                        Policy::template MakeVScaleRegTileDistribution<Problem>());
            }
            else
            {
                return make_null_tile_window(make_tuple());
            }
        }();

        // prefetch K tile
        index_t i_total_loops      = 0;
        constexpr index_t k0_loops = kQKHeaddim / kK0;
        constexpr index_t k1_loops = kN0 / kK1;
        // Number of k0 iterations prefetched ahead of the current compute iteration.
        // The skip decision must be made this many iterations before the last k0 loop.
        constexpr index_t kK0PrefetchDepth = 2;
        const index_t gemm0_tail_k_iters   = [&]() {
            if constexpr(kUsePartialKForGemm0Tail)
            {
                return ck_tile::integer_divide_ceil(valid_last_k0_length, kGemm0WarpK);
            }
            return static_cast<index_t>(kGemm0KItersPerBlock);
        }();
        const bool skip_last_k0_loop = [&]() {
            if constexpr(kPadHeadDimQ)
            {
                return valid_k0_loops == (k0_loops - 1);
            }
            return false;
        }();
        // Use compile-time conditional for group barrier sequence
        // (No runtime lambda selection)
        auto schedule_gemm_0 = [] {
            constexpr auto WarpGemmConfig =
                BlockGemm0::Policy::template GetWarpGemmMWarpNWarp<Problem>();
            using WarpGemm0 = remove_cvref_t<decltype(WarpGemmConfig.template at<0>())>;
            constexpr index_t Gemm0MWarp   = WarpGemmConfig.template at<1>();
            constexpr index_t Gemm0NWarp   = WarpGemmConfig.template at<2>();
            constexpr index_t WarpGemm0M   = WarpGemm0::WarpGemmAttribute::Impl::kM;
            constexpr index_t WarpGemm0N   = WarpGemm0::WarpGemmAttribute::Impl::kN;
            constexpr index_t WarpGemm0K   = WarpGemm0::WarpGemmAttribute::Impl::kK;
            constexpr index_t NumMfmaInsts = (kM0 / WarpGemm0M) * (kN0 / WarpGemm0N) *
                                             (kK0 / WarpGemm0K) / (Gemm0MWarp * Gemm0NWarp);
            if constexpr(get_warp_size() == 64 && kQKHeaddim == 256)
            {
                if constexpr(NumMfmaInsts % 8 == 0)
                {
                    static_for<0, NumMfmaInsts / 8, 1>{}([&](auto) {
                        __builtin_amdgcn_sched_group_barrier(DS_READ, 2, 0); // DS read
                        __builtin_amdgcn_sched_group_barrier(MFMA, 2, 0);    // MFMA
                        __builtin_amdgcn_sched_group_barrier(DS_READ, 1, 0); // DS read
                        __builtin_amdgcn_sched_group_barrier(MFMA, 2, 0);    // MFMA
                        __builtin_amdgcn_sched_group_barrier(DS_READ, 1, 0); // DS read
                        __builtin_amdgcn_sched_group_barrier(MFMA, 4, 0);    // MFMA
                    });
                }
            }
        };

        static_assert(kK0PrefetchDepth <= k0_loops);
        static_assert(1 <= k1_loops);
        do
        {
            float k_descale = 1.0f;
            if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE)
            {
                // K and V share the same seqlen_k position within a block
                const index_t kv_idx = (kv_load_start + i_total_loops * kN0) / block_scale_size_kv;
                k_descale            = k_descale_ptr[kv_idx];
            }
            // STAGE 1, QK gemm
            auto k_dram_window = make_tile_window(
                k_dram_block_window.get_bottom_tensor_view(),
                k_dram_block_window.get_window_lengths(),
                k_dram_block_window.get_window_origin(),
                Policy::template MakeKDramTileDistribution<Problem>()); // K DRAM tile window for
                                                                        // load
            auto k_scale_dram_window = [&] {
                if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::MX)
                {
                    return make_tile_window(
                        k_scale_dram_block_window.get_bottom_tensor_view(),
                        k_scale_dram_block_window.get_window_lengths(),
                        k_scale_dram_block_window.get_window_origin(),
                        Policy::template MakeKScaleRegTileDistribution<Problem>());
                }
                else
                {
                    return make_null_tile_window(make_tuple());
                }
            }();
            auto load_k_scale_block_tile = [&] {
                if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::MX)
                {
                    auto t = load_tile(k_scale_dram_window);
                    move_tile_window(k_scale_dram_window, {0, kK0 / kQKScaleGranularity});
                    return t;
                }
                else
                {
                    return make_null_tile_window(make_tuple());
                }
            };

            auto k_block_tile = load_tile(k_dram_window);
            {
                move_tile_window(k_dram_window, {0, kK0});
                clear_tile(s_acc); // initialize C
                store_tile(k_lds_window, tile_elementwise_in(k_element_func, k_block_tile));
                k_block_tile = load_tile(k_dram_window);
            }
            auto k_scale_block_tile = load_k_scale_block_tile();

            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                __builtin_amdgcn_sched_barrier(
                    0); // prevent from messing up the order of global loads
            }
            const auto bias_tile = load_tile(bias_dram_window); // load bias tile
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                __builtin_amdgcn_sched_barrier(
                    0); // prevent from messing up the order of global loads
            }

            auto run_gemm_0 = [&](auto i_k0) {
                if constexpr(kUsePartialKForGemm0Tail)
                {
                    if(static_cast<index_t>(i_k0.value) == (valid_k0_loops - 1) &&
                       gemm0_tail_k_iters < kGemm0KItersPerBlock)
                    {
                        static_for<1, kGemm0KItersPerBlock, 1>{}([&](auto i_tail_k_iter) {
                            constexpr index_t kTailKIters = i_tail_k_iter;
                            constexpr index_t kTailK0     = kTailKIters * kGemm0WarpK;

                            if(gemm0_tail_k_iters == kTailKIters)
                            {
                                using Gemm0TailProblem = BlockGemmProblem<
                                    QDataType,
                                    KDataType,
                                    SaccDataType,
                                    Problem::kNumGemm0Warps * get_warp_size(),
                                    TileGemmShape<
                                        sequence<kM0, kN0, kTailK0>,
                                        typename BlockFmhaShape::Gemm0BlockWarps,
                                        sequence<BlockFmhaShape::Gemm0WarpTile::at(number<0>{}),
                                                 BlockFmhaShape::Gemm0WarpTile::at(number<1>{}),
                                                 kGemm0WarpK>>>;
                                constexpr auto gemm_0_tail =
                                    BlockGemmARegBSmemCRegV2<Gemm0TailProblem,
                                                             typename BlockGemm0::Policy>{};

                                auto q_slice =
                                    get_slice_tile(q_tile,
                                                   sequence<0, i_k0 * kK0>{},
                                                   sequence<kM0, i_k0 * kK0 + kTailK0>{});
                                auto k_tail_window = make_tile_window(
                                    k_lds, make_tuple(number<kN0>{}, number<kTailK0>{}), {0, 0});

                                gemm_0_tail(s_acc, q_slice, k_tail_window);
                            }
                        });
                        return;
                    }
                }

                auto q_slice = get_slice_tile(
                    q_tile, sequence<0, i_k0 * kK0>{}, sequence<kM0, (i_k0 + 1) * kK0>{});
                if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::MX)
                {
                    auto q_scale_slice =
                        get_slice_tile(q_scale,
                                       sequence<0, i_k0*(kK0 / kQKScaleGranularity)>{},
                                       sequence<kM0, (i_k0 + 1) * (kK0 / kQKScaleGranularity)>{});
                    gemm_0(s_acc, q_slice, q_scale_slice, k_lds_window, k_scale_block_tile);
                }
                else
                {
                    gemm_0(s_acc, q_slice, k_lds_window);
                    schedule_gemm_0();
                }
            };

            if constexpr(k0_loops > kK0PrefetchDepth)
            {
                static_for<0, k0_loops - kK0PrefetchDepth, 1>{}([&](auto i_k0) {
                    block_sync_lds();
                    run_gemm_0(number<i_k0>{});
                    block_sync_lds();
                    if constexpr(kPadHeadDimQ && i_k0 == (k0_loops - 1 - kK0PrefetchDepth))
                    {
                        if(!skip_last_k0_loop)
                        {
                            move_tile_window(k_dram_window, {0, kK0});
                        }

                        store_tile(
                            k_lds_window,
                            tile_elementwise_in(k_element_func, k_block_tile)); // LDS write i + 1

                        if(!skip_last_k0_loop)
                        {
                            k_block_tile = load_tile(k_dram_window); // global read i + 2
                        }
                    }
                    else
                    {
                        move_tile_window(k_dram_window, {0, kK0});

                        store_tile(
                            k_lds_window,
                            tile_elementwise_in(k_element_func, k_block_tile)); // LDS write i + 1
                        k_block_tile = load_tile(k_dram_window);                // global read i + 2
                    }
                    k_scale_block_tile = load_k_scale_block_tile();
                });
            }

            auto v_prefetch = decltype(load_tile(v_dram_window)){};
            enum class VPrefetchPoint
            {
                BeforeGemm0Tail,
                AfterGemm0Tail,
                AfterSoftmax
            };

#if defined(__gfx11__) || defined(__gfx12__)
            constexpr auto kVPrefetch =
                kPadHeadDimV ? VPrefetchPoint::AfterSoftmax : VPrefetchPoint::AfterGemm0Tail;
#else
            constexpr auto kVPrefetch = VPrefetchPoint::BeforeGemm0Tail;
#endif
            if constexpr(kVPrefetch == VPrefetchPoint::BeforeGemm0Tail)
            {
                load_tile(v_prefetch, v_dram_window); // prefetch load v tile
            }
            { // tail
                block_sync_lds();
                run_gemm_0(number<k0_loops - kK0PrefetchDepth>{});
                if(!skip_last_k0_loop)
                {
                    block_sync_lds();

                    store_tile(k_lds_window, tile_elementwise_in(k_element_func, k_block_tile));

                    k_scale_block_tile = load_k_scale_block_tile();

                    block_sync_lds();

                    run_gemm_0(number<k0_loops - 1>{});
                }
            }
            if constexpr(kVPrefetch == VPrefetchPoint::AfterGemm0Tail)
            {
                load_tile(v_prefetch, v_dram_window);
            }
            // dequant
            auto s_acc_element_func_ = [&s_acc_element_func, k_descale]() {
                if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE)
                {
                    return s_acc_element_func * k_descale;
                }
                else
                    return s_acc_element_func;
            }();

            // STAGE 2, scale_s, add bias, mask, softmax
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                s_acc = tile_elementwise_in(s_acc_element_func_, s_acc);
                tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, s_acc);
                tile_elementwise_inout(
                    [&](auto& x, const auto& y) {
#if !CK_TILE_FMHA_FWD_FAST_EXP2
                        x += type_convert<SaccDataType>(bias_element_func(y));
#else
                        x += log2e_v<SaccDataType> *
                             type_convert<SaccDataType>(bias_element_func(y));
#endif
                    },
                    s_acc,
                    bias_tile);
            }
            else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
            {
                const auto k_origin    = k_dram_block_window.get_window_origin();
                constexpr auto s_spans = decltype(s_acc)::get_distributed_spans();
                s_acc                  = tile_elementwise_in(s_acc_element_func_, s_acc);
                sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(s_spans[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            s_acc.get_tile_distribution(), make_tuple(idx0, idx1));

                        const auto row = q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                        const auto col = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);

                        s_acc(i_j_idx) *= scale_s;
                        position_encoding.update(s_acc(i_j_idx), row, col);
                    });
                });
            }
            else
            {
                s_acc = tile_elementwise_in(s_acc_element_func_, s_acc);
                if constexpr(kHasLogitsSoftCap)
                {
                    auto apply_logits_transform =
                        [&variant, &variant_params, &block_indices](auto& x) {
                            x = variant.LogitsTransform(variant_params,
                                                        variant.QueryTransform(variant_params, x),
                                                        block_indices.batch_idx,
                                                        block_indices.qo_head_idx,
                                                        block_indices.kv_head_idx);
                        };
#if !CK_TILE_FMHA_FWD_FAST_EXP2
                    tile_elementwise_inout(apply_logits_transform, s_acc);
#else
                    tile_elementwise_inout(apply_logits_transform, s_acc);
#endif
                }
                else
                {
#if !CK_TILE_FMHA_FWD_FAST_EXP2
                    tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, s_acc);
#endif
                }
            }
            if constexpr(kHasSink)
            {
                if(i_total_loops == 0)
                    move_tile_window(bias_dram_window, {0, seqlen_k_start - sink_seq_end});
            }
            move_tile_window(bias_dram_window, {0, kN0});
            if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
            {
                const auto k_origin      = k_dram_block_window.get_window_origin();
                bool need_perpixel_check = mask.IsEdgeTile(q_origin.at(number<0>{}),
                                                           k_origin.at(number<0>{}),
                                                           number<kM0>{},
                                                           number<kN0>{});
                if(need_perpixel_check)
                {
                    auto apply_mask = [&](auto&& mask_func) {
                        set_tile_if(
                            s_acc, -numeric<SMPLComputeDataType>::infinity(), [&](auto tile_idx) {
                                const auto row =
                                    q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                                const auto col =
                                    k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                                return !mask_func(variant_params,
                                                  block_indices.batch_idx,
                                                  row,
                                                  col,
                                                  block_indices.qo_head_idx,
                                                  block_indices.kv_head_idx);
                            });
                    };

                    if constexpr(kHasSink)
                    {
                        apply_mask([&](auto&&... args) {
                            return variant.LogitsSinkMask(std::forward<decltype(args)>(args)...);
                        });
                    }
                    else
                    {
                        apply_mask([&](auto&&... args) {
                            return variant.LogitsMask(std::forward<decltype(args)>(args)...);
                        });
                    }
                }
            }

            const auto s = cast_tile<SMPLComputeDataType>(s_acc); // S{j}
            auto m_local = block_tile_reduce<SMPLComputeDataType>(
                s,
                sequence<1>{},
                f_max,
                -numeric<SMPLComputeDataType>::infinity()); // m_local = rowmax(S{j})
            block_tile_reduce_sync(m_local, f_max, bool_constant<false>{});

            const auto m_old = m; // m{j-1}
            tile_elementwise_inout(
                [](auto& e0, auto e1, auto e2) { e0 = max(e1, e2); }, m, m_old, m_local); // m{j}

            auto p_compute = make_static_distributed_tensor<SMPLComputeDataType>(
                s.get_tile_distribution()); // Pcompute{j}

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

            // Conditional rescaling: skip o_acc rescale when correction factor
            // exp2(acc_scale_log2) is negligible (< exp2(-8) ~= 0.004, below BF16
            // precision). Adapted from FlashAttention-4 (Tri Dao, 2025).
            // Eliminates 70-90% of rescale operations in practice.
            //
            // For skip rows we stabilize P with m_old (the previous max) instead of
            // the new max m_j, so P is computed directly in the m_{j-1} frame and no
            // post-correction sweep is needed. For rescale rows we use m_j as usual.
            // FP8 quant modes (PERTENSOR/BLOCKSCALE/etc.) cast P to FP8 after
            // softmax. In the skip branch P is computed with m_old, so P can
            // exceed the FP8 representable range and saturate, corrupting the
            // P*V GEMM. Disable skip for all FP8 paths (threshold 0).
            static constexpr SMPLComputeDataType kRescaleThreshold =
                type_convert<SMPLComputeDataType>(
                    QScaleEnum == BlockAttentionQuantScaleEnum::NO_SCALE ? 8.0f : 0.0f);

            // Per-row stabilizer: m_old for skip rows, m_j for rescale rows.
            auto m_stab =
                make_static_distributed_tensor<SMPLComputeDataType>(m.get_tile_distribution());
            // Per-row rescale factor (exp2 of acc_scale_log2); only valid when
            // needs_rescale[i] is true.
            auto rescale_factor =
                make_static_distributed_tensor<SMPLComputeDataType>(m.get_tile_distribution());
            auto needs_rescale = make_static_distributed_tensor<bool>(m.get_tile_distribution());
            set_tile(needs_rescale, false);

            constexpr auto m_spans = decltype(m)::get_distributed_spans();
            sweep_tile_span(m_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                const auto acc_scale_log2 = [&]() {
                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                 BiasEnum == BlockAttentionBiasEnum::ALIBI)
                    {
                        return m_old[i_idx] - get_validated_m(m[i_idx]);
                    }
                    else
                    {
                        if constexpr(kHasLogitsSoftCap)
                        {
                            return m_old[i_idx] - get_validated_m(m[i_idx]);
                        }
                        else
                        {
                            auto row_max = scale_s * get_validated_m(m[i_idx]);
                            return scale_s * m_old[i_idx] - row_max;
                        }
                    }
                }();

                const bool need_rescale =
                    (acc_scale_log2 < type_convert<SMPLComputeDataType>(-kRescaleThreshold));

                if(need_rescale)
                {
                    rescale_factor(i_idx) = exp2(acc_scale_log2);
                    m_stab(i_idx)         = m[i_idx];
                    needs_rescale(i_idx)  = true;
                }
                else
                {
                    // Skip branch: stabilize P with m_old so P is already in
                    // m_{j-1} frame; restore m to m_old for downstream iterations.
                    m_stab(i_idx) = m_old[i_idx];
                    m(i_idx)      = m_old[i_idx];
                }
#else
                const auto diff = m_old[i_idx] - get_validated_m(m[i_idx]);
                const bool need_rescale =
                    (diff < type_convert<SMPLComputeDataType>(-kRescaleThreshold));

                if(need_rescale)
                {
                    rescale_factor(i_idx) = exp(diff);
                    m_stab(i_idx)         = m[i_idx];
                    needs_rescale(i_idx)  = true;
                }
                else
                {
                    m_stab(i_idx) = m_old[i_idx];
                    m(i_idx)      = m_old[i_idx];
                }
#endif
            });

            constexpr auto p_spans = decltype(p_compute)::get_distributed_spans();
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                // For BLOCKSCALE: precompute (m - shift) once per row
                // Bias/Alibi/SoftCap: exp2(s - m + shift) = exp2(s - (m - shift))
                // else: exp2(scale_s*s - scale_s*m + shift) = exp2(scale_s*s - (scale_s*m - shift))
                auto validated_m = get_validated_m(m_stab[i_idx]);
                auto row_max     = scale_s * validated_m;
                if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE)
                {
#if CK_TILE_USE_OCP_FP8
                    validated_m -= OCP_FP8_SHIFT; // for Bias/Alibi/SoftCap
                    row_max -= OCP_FP8_SHIFT;     // for else branch
#else
                    validated_m -= FNUZ_FP8_SHIFT;
                    row_max -= FNUZ_FP8_SHIFT;
#endif
                }
#endif
                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                 BiasEnum == BlockAttentionBiasEnum::ALIBI)
                    {
                        p_compute(i_j_idx) = exp2(s[i_j_idx] - validated_m);
                    }
                    else
                    {
                        if constexpr(kHasLogitsSoftCap)
                        {
                            p_compute(i_j_idx) = exp2(s[i_j_idx] - validated_m);
                        }
                        else
                        {
                            p_compute(i_j_idx) = exp2(scale_s * s[i_j_idx] - row_max);
                        }
                    }
#else
                    p_compute(i_j_idx)     = exp(s[i_j_idx] - get_validated_m(m_stab[i_idx]));
#endif
                });
            });

            auto rowsum_p = block_tile_reduce<SMPLComputeDataType>(
                p_compute, sequence<1>{}, f_sum, SMPLComputeDataType{0}); // rowsum(Pcompute{j})

            block_tile_reduce_sync(rowsum_p, f_sum, bool_constant<false>{});
            // l{j}, Oacc{j}
            constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
            sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                if(needs_rescale[i_idx])
                {
                    const auto tmp = rescale_factor[i_idx];
                    l(i_idx)       = tmp * l[i_idx] + rowsum_p[i_idx];
                    sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        o_acc(i_j_idx) *= tmp;
                    });
                }
                else
                {
                    // Skip: P already in m_{j-1} frame, no o_acc rescale needed.
                    l(i_idx) = l[i_idx] + rowsum_p[i_idx];
                }
            });

            if constexpr(kHasDropout)
            {
                block_sync_lds();
                auto randval_ptr = reinterpret_cast<char*>(smem_ptr);

                index_t seq_offset = [&]() {
                    if constexpr(!kHasSink)
                        return seqlen_k_start + i_total_loops * kN0;

                    const bool in_sink_phase = (num_sink_loop > i_total_loops);
                    if(i_total_loops == num_sink_loop)
                        move_tile_window(randval_dram_window, {0, seqlen_k_start - sink_seq_end});

                    return in_sink_phase ? (kv_load_start + i_total_loops * kN0)
                                         : (seqlen_k_start + (i_total_loops - num_sink_loop) * kN0);
                }();

                dropout.template Run<decltype(gemm_0), SMPLComputeDataType, RandValOutputDataType>(
                    randval_ptr, seq_offset, p_compute, randval_dram_window);
            }

            if constexpr(kVPrefetch == VPrefetchPoint::AfterSoftmax)
            {
                load_tile(v_prefetch, v_dram_window);
            }

            block_sync_lds();
            if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                auto v_shuffle_tmp = make_static_distributed_tensor<VDataType>(
                    Policy::template MakeShuffledVRegBlockDescriptor<Problem>());
                shuffle_tile(v_shuffle_tmp, v_prefetch);
                store_tile(
                    v_lds_window,
                    tile_elementwise_in(v_element_func, v_shuffle_tmp)); // store the prefetch
            }
            else
            {
                store_tile(v_lds_window,
                           tile_elementwise_in(v_element_func, v_prefetch)); // store the prefetch
            }

            move_tile_window(v_dram_window, {0, kK1});

            auto load_v_scale_block_tile = [&] {
                if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::MX)
                {
                    auto t = load_tile(v_scale_dram_window);
                    move_tile_window(v_scale_dram_window, {0, kK1 / kVScaleGranularity});
                    return t;
                }
                else
                {
                    return make_null_tile_window(make_tuple());
                }
            };
            auto v_scale_block_tile = load_v_scale_block_tile();

            float v_descale = 1.0f;
            if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE)
            {
                // K and V share the same seqlen_k position within a block
                const index_t kv_idx = (kv_load_start + i_total_loops * kN0) / block_scale_size_kv;
                v_descale            = v_descale_ptr[kv_idx];
            }

            const auto p_p_scale = [&] {
                if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::MX)
                {
                    auto p_result = make_static_distributed_tensor<PDataType>(
                        p_compute.get_tile_distribution());
                    auto p_scale_result = make_static_distributed_tensor<PScaleDataType>(
                        Policy::template MakePScaleRegTileDistribution<Problem>());

                    constexpr auto config =
                        decltype(gemm_1)::Policy::template GetWarpGemmMWarpNWarp<Problem>();
                    using WG = remove_cvref_t<decltype(config.template at<0>())>;

                    cast_tile_mx<kVScaleGranularity, WG::WarpGemmAttribute::Impl::kAMLane>(
                        p_result, p_scale_result, p_compute);

                    return make_tuple(p_result, p_scale_result);
                }
                else
                {
#if defined(__gfx11__)
                    auto p_result = make_static_distributed_tensor<PDataType>(
                        decltype(gemm_1)::template MakeABlockTileDistribution<kM0, kN0>());
                    PermuteWarpGemmCToA(p_result,
                                        cast_tile<PDataType>(tile_elementwise_in(
                                            p_compute_element_func, p_compute)));
#else
                    const auto p_result = cast_tile<PDataType>(
                        tile_elementwise_in(p_compute_element_func, p_compute));
#endif
                    return make_tuple(p_result, null_tensor{});
                }
            }();
            const auto p       = p_p_scale[number<0>{}];
            const auto p_scale = p_p_scale[number<1>{}];

            // STAGE 3, KV gemm
            auto o_acc0 = decltype(o_acc){};
            clear_tile(o_acc0);

            constexpr auto gemm_1_config =
                BlockGemm1::Policy::template GetWarpGemmMWarpNWarp<Problem>();
            using Gemm1WarpGemm = remove_cvref_t<decltype(gemm_1_config.template at<0>())>;
            constexpr index_t kGemm1NWarp    = gemm_1_config.template at<2>();
            constexpr index_t kGemm1NPerIter = kGemm1NWarp * Gemm1WarpGemm::kN;
            const index_t valid_n_iters      = [&]() {
                if constexpr(kPadHeadDimV && kBlockGemm1SupportsPartialN)
                {
                    return ck_tile::integer_divide_ceil(valid_n1_length, kGemm1NPerIter);
                }
                return static_cast<index_t>(0);
            }();

            auto run_gemm_1_impl =
                [&](auto& o_acc_tensor, const auto& p_slice, const auto&... gemm_1_args) {
                    if constexpr(kPadHeadDimV && kBlockGemm1SupportsPartialN)
                    {
                        gemm_1(o_acc_tensor, p_slice, gemm_1_args..., valid_n_iters);
                    }
                    else
                    {
                        gemm_1(o_acc_tensor, p_slice, gemm_1_args...);
                    }
                };

            auto run_gemm_1 = [&](auto i_k1) {
                auto p_slice =
                    get_slice_tile(p, sequence<0, i_k1 * kK1>{}, sequence<kM0, (i_k1 + 1) * kK1>{});
                if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::MX)
                {
                    auto p_scale_slice =
                        get_slice_tile(p_scale,
                                       sequence<0, i_k1*(kK1 / kVScaleGranularity)>{},
                                       sequence<kM0, (i_k1 + 1) * (kK1 / kVScaleGranularity)>{});
                    run_gemm_1_impl(
                        o_acc, p_slice, p_scale_slice, v_lds_window, v_scale_block_tile);
                }
                else
                {
                    if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE)
                    {
                        run_gemm_1_impl(o_acc0, p_slice, v_lds_window);
                    }
                    else
                    {
                        run_gemm_1_impl(o_acc, p_slice, v_lds_window);
                    }
                }
            };

            if constexpr(k1_loops > 1)
            {
                static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
                    const auto v = load_tile(v_dram_window); // load next v
                    block_sync_lds();
                    run_gemm_1(number<i_k1>{});
                    block_sync_lds();
                    if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
                    {
                        auto v_shuffle_tmp = make_static_distributed_tensor<VDataType>(
                            Policy::template MakeShuffledVRegBlockDescriptor<Problem>());
                        shuffle_tile(v_shuffle_tmp, v);
                        store_tile(v_lds_window,
                                   tile_elementwise_in(v_element_func,
                                                       v_shuffle_tmp)); // store the prefetch
                    }
                    else
                    {
                        store_tile(v_lds_window,
                                   tile_elementwise_in(v_element_func, v)); // store next v
                    }
                    move_tile_window(v_dram_window, {0, kK1});
                    v_scale_block_tile = load_v_scale_block_tile();
                });
            }
            // move K tile windows
            if constexpr(kHasSink)
            {
                if(i_total_loops == 0)
                {
                    move_tile_window(k_dram_block_window, {seqlen_k_start - sink_seq_end, 0});
                    move_tile_window(v_dram_window, {0, seqlen_k_start - sink_seq_end});
                }
            }
            move_tile_window(k_dram_block_window, {kN0, 0});
            if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::MX)
            {
                move_tile_window(k_scale_dram_block_window, {kN0, 0});
            }
            // tail
            {
                block_sync_lds();
                run_gemm_1(number<k1_loops - 1>{});
                block_sync_lds();
            }
            if constexpr(QScaleEnum == BlockAttentionQuantScaleEnum::BLOCKSCALE)
            {
                tile_elementwise_inout(
                    [&v_descale](auto& o, auto& o0) { o += o0 * v_descale; }, o_acc, o_acc0);
            }
        } while(++i_total_loops < num_total_loop);

        // store lse
        if constexpr(kStoreLSE)
        {
            auto lse = make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

            constexpr auto lse_spans = decltype(lse)::get_distributed_spans();
            sweep_tile_span(lse_spans[number<0>{}], [&, m_ = m, l_ = l](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                // In the masked biased case, the entire row can be suppressed and the accumulated
                // softmax denominator becomes zero; treat it as log(0) = -inf to avoid NaNs.
                if(l_[i_idx] == 0.0f)
                {
                    lse(i_idx) = -numeric<LSEDataType>::infinity();
                }
                else
                {
#if CK_TILE_FMHA_FWD_FAST_EXP2
                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                 BiasEnum == BlockAttentionBiasEnum::ALIBI)
                    {
                        lse(i_idx) = m_[i_idx] / C_LOG2E + log(l_[i_idx]);
                    }
                    else
                    {
                        if constexpr(kHasLogitsSoftCap)
                        {
                            lse(i_idx) = m_[i_idx] / C_LOG2E + log(l_[i_idx]);
                        }
                        else
                        {
                            lse(i_idx) = m_[i_idx] * scale_s / C_LOG2E + log(l_[i_idx]);
                        }
                    }
#else
                    lse(i_idx) = m_[i_idx] + log(l_[i_idx]);
#endif
                }
            });

            store_tile(lse_dram_window_tmp, tile_elementwise_in(lse_element_func, lse));
        }

        // finally, O
        constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();

        sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            const auto tmp       = [&]() {
                // When bias carries -inf masks the denominator can be zero; guard the normalization
                // so we do not divide by zero after a fully masked row.
                if constexpr(FmhaMask::IsMasking ||
                             BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                {
                    return l[i_idx] == 0.f ? 0.f : 1 / l[i_idx];
                }
                else
                    return 1 / l[i_idx];
            }();
            sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                o_acc(i_j_idx) *= tmp;
            });
        });

        o_acc = tile_elementwise_in(o_acc_element_func, o_acc);

        return o_acc;
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename RandValDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename QElementFunction,
              typename KElementFunction,
              typename VElementFunction,
              typename BiasElementFunction,
              typename LSEElementFunction,
              typename SAccElementFunction,
              typename PComputeElementFunction,
              typename OAccElementFunction,
              typename PositionEncoding,
              typename AttentionVariantParams,
              typename BlockIndices,
              typename QScaleDramBlockWindowTmp,
              typename KScaleDramBlockWindowTmp,
              typename VScaleDramBlockWindowTmp>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp, // M0*K0 tile
               const QElementFunction& q_element_func,
               const KDramBlockWindowTmp& k_dram_block_window_tmp, // N0*K0 tile
               const KElementFunction& k_element_func,
               const VDramBlockWindowTmp& v_dram_block_window_tmp, // N1*K1 tile
               const VElementFunction& v_element_func,
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
               const BiasElementFunction& bias_element_func,
               RandValDramBlockWindowTmp& randval_dram_block_window_tmp,
               LSEDramBlockWindowTmp& lse_dram_window_tmp, // M0*1 tile
               const LSEElementFunction& lse_element_func,
               const SAccElementFunction& s_acc_element_func,
               const PComputeElementFunction& p_compute_element_func,
               const OAccElementFunction& o_acc_element_func,
               FmhaMask mask,
               PositionEncoding position_encoding,
               float scale_s,
               const AttentionVariant& variant,
               const AttentionVariantParams& variant_params,
               const BlockIndices& block_indices,
               void* smem_ptr,
               DropoutType& dropout,
               const float* k_descale_ptr,
               const float* v_descale_ptr,
               const index_t block_scale_size_kv,
               const QScaleDramBlockWindowTmp&
                   q_scale_dram_block_window_tmp, // M0*(K0/kQKScaleGranularity) tile
               const KScaleDramBlockWindowTmp&
                   k_scale_dram_block_window_tmp, // N0*(K0/kQKScaleGranularity) tile
               const VScaleDramBlockWindowTmp&
                   v_scale_dram_block_window_tmp, // N1*(K1/kVScaleGranularity) tile
               const float sink_v) const
    {
        return operator()(q_dram_block_window_tmp,
                          q_element_func,
                          k_dram_block_window_tmp,
                          k_element_func,
                          v_dram_block_window_tmp,
                          v_element_func,
                          bias_dram_block_window_tmp,
                          bias_element_func,
                          randval_dram_block_window_tmp,
                          lse_dram_window_tmp,
                          lse_element_func,
                          s_acc_element_func,
                          p_compute_element_func,
                          o_acc_element_func,
                          mask,
                          position_encoding,
                          scale_s,
                          variant,
                          variant_params,
                          block_indices,
                          smem_ptr,
                          dropout,
                          k_descale_ptr,
                          v_descale_ptr,
                          block_scale_size_kv,
                          q_scale_dram_block_window_tmp,
                          k_scale_dram_block_window_tmp,
                          v_scale_dram_block_window_tmp,
                          sink_v,
                          kQKHeaddim / kK0,
                          kK0,
                          kN1);
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename RandValDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename PositionEncoding,
              typename AttentionVariantParams,
              typename BlockIndices>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,       // M0*K0 tile
               const KDramBlockWindowTmp& k_dram_block_window_tmp,       // N0*K0 tile
               const VDramBlockWindowTmp& v_dram_block_window_tmp,       // N1*K1 tile
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
               RandValDramBlockWindowTmp& randval_dram_block_window_tmp, // M0*N0 tile
               LSEDramBlockWindowTmp& lse_dram_block_window_tmp,         // M0*1 tile
               FmhaMask mask,
               PositionEncoding position_encoding,
               float scale_s,
               const AttentionVariant& variant,
               const AttentionVariantParams& variant_params,
               const BlockIndices& block_indices,
               void* smem_ptr,
               DropoutType& dropout,
               const float sink_v,
               const index_t valid_k0_loops,
               const index_t valid_last_k0_length,
               const index_t valid_n1_length) const
    {
        return operator()(q_dram_block_window_tmp,
                          identity{},
                          k_dram_block_window_tmp,
                          identity{},
                          v_dram_block_window_tmp,
                          identity{},
                          bias_dram_block_window_tmp,
                          identity{},
                          randval_dram_block_window_tmp,
                          lse_dram_block_window_tmp,
                          identity{},
                          identity{},
                          identity{},
                          identity{},
                          mask,
                          position_encoding,
                          scale_s,
                          variant,
                          variant_params,
                          block_indices,
                          smem_ptr,
                          dropout,
                          nullptr,
                          nullptr,
                          1,
                          make_null_tile_window(make_tuple()),
                          make_null_tile_window(make_tuple()),
                          make_null_tile_window(make_tuple()),
                          sink_v,
                          valid_k0_loops,
                          valid_last_k0_length,
                          valid_n1_length);
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename RandValDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename PositionEncoding,
              typename AttentionVariantParams,
              typename BlockIndices>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,       // M0*K0 tile
               const KDramBlockWindowTmp& k_dram_block_window_tmp,       // N0*K0 tile
               const VDramBlockWindowTmp& v_dram_block_window_tmp,       // N1*K1 tile
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
               RandValDramBlockWindowTmp& randval_dram_block_window_tmp, // M0*N0 tile
               LSEDramBlockWindowTmp& lse_dram_block_window_tmp,         // M0*1 tile
               FmhaMask mask,
               PositionEncoding position_encoding,
               float scale_s,
               const AttentionVariant& variant,
               const AttentionVariantParams& variant_params,
               const BlockIndices& block_indices,
               void* smem_ptr,
               DropoutType& dropout,
               const float sink_v) const
    {
        return operator()(q_dram_block_window_tmp,
                          k_dram_block_window_tmp,
                          v_dram_block_window_tmp,
                          bias_dram_block_window_tmp,
                          randval_dram_block_window_tmp,
                          lse_dram_block_window_tmp,
                          mask,
                          position_encoding,
                          scale_s,
                          variant,
                          variant_params,
                          block_indices,
                          smem_ptr,
                          dropout,
                          sink_v,
                          kQKHeaddim / kK0,
                          kK0,
                          kN1);
    }
};

template <typename Problem_, typename Policy_ = BlockFmhaPipelineQRKSVSDefaultPolicy>
using BlockFmhaPipelineQRKSVSHpad = BlockFmhaPipelineQRKSVS<Problem_, Policy_, true>;

} // namespace ck_tile
