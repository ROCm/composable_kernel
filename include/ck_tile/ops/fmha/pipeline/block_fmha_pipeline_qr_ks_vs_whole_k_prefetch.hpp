// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_whole_k_prefetch_default_policy.hpp"
#include "ck_tile/ops/fmha/block/block_dropout.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

template <typename Problem_, typename Policy_ = BlockFmhaPipelineQRKSVSWholeKPrefetchDefaultPolicy>
struct BlockFmhaPipelineQRKSVSWholeKPrefetch
{
    using Problem               = remove_cvref_t<Problem_>;
    using Policy                = remove_cvref_t<Policy_>;
    using QDataType             = remove_cvref_t<typename Problem::QDataType>;
    using KDataType             = remove_cvref_t<typename Problem::KDataType>;
    using VDataType             = remove_cvref_t<typename Problem::VDataType>;
    using SaccDataType          = remove_cvref_t<typename Problem::SaccDataType>;
    using CompDataType          = remove_cvref_t<typename Problem::SMPLComputeDataType>;
    using BiasDataType          = remove_cvref_t<typename Problem::BiasDataType>;
    using RandValOutputDataType = remove_cvref_t<typename Problem::RandValOutputDataType>;
    using LSEDataType           = remove_cvref_t<typename Problem::LSEDataType>;
    using PDataType             = remove_cvref_t<typename Problem::PDataType>;
    using OaccDataType          = remove_cvref_t<typename Problem::OaccDataType>;
    using ODataType             = remove_cvref_t<typename Problem::ODataType>;
    using FmhaMask              = remove_cvref_t<typename Problem::FmhaMask>;
    using AttentionVariant      = remove_cvref_t<typename Problem::AttentionVariant>;

    using BlockFmhaShape             = remove_cvref_t<typename Problem::BlockFmhaShape>;
    using VLayout                    = remove_cvref_t<typename BlockFmhaShape::VLayout>;
    static constexpr bool kQLoadOnce = true;
    static_assert(kQLoadOnce == Policy::QLoadOnce);
    static_assert(!Problem::kUseTrLoad, "This pipeline does not use trload!");
    static_assert(sizeof(KDataType) == sizeof(VDataType) &&
                      alignof(KDataType) == alignof(VDataType),
                  "K and V share the same LDS region; their element types must have identical "
                  "size and alignment.");

    static constexpr bool kUseN0Loop       = true;
    static constexpr bool kIgnoreFastExp2  = true;
    static constexpr bool kIsNaiveHDimLoad = true;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kM0 = BlockFmhaShape::kM0;
    static constexpr index_t kN0 = BlockFmhaShape::kN0;
    static constexpr index_t kN0Sub =
        BlockFmhaShape::kK0; // subdivision of kN0 used in N0-loop, same value as kK0
    static constexpr index_t kN1           = BlockFmhaShape::kN1;
    static constexpr index_t kK1           = BlockFmhaShape::kK1;
    static constexpr index_t kQKHeaddim    = BlockFmhaShape::kQKHeaddim;
    static constexpr index_t kSubQKHeaddim = BlockFmhaShape::kSubQKHeaddim;

    static_assert(kSubQKHeaddim <= 256, "hdim bigger than 256 is not suitable for this pipeline!");

    static constexpr bool kIsGroupMode      = Problem::kIsGroupMode;
    static constexpr bool kPadSeqLenQ       = Problem::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK       = Problem::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ      = Problem::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV      = Problem::kPadHeadDimV;
    static constexpr auto BiasEnum          = Problem::BiasEnum;
    static constexpr bool kStoreLSE         = Problem::kStoreLSE;
    static constexpr bool kHasDropout       = Problem::kHasDropout;
    static constexpr bool kHasLogitsSoftCap = Problem::kHasLogitsSoftCap;

    // since this pipeline is only used by the inference path of xformers, the Dropout function is
    // not well tested with the pipeline, so here we have Dropout disabled
    static_assert(kHasDropout == false, "Dropout is not supported by this pipeline at present!");

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
    static constexpr index_t kAlignmentQ =
        kPadHeadDimQ ? 1 : Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK =
        kPadHeadDimQ ? 1 : Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV =
        Problem::kPadHeadDimV ? 1 : Policy::template GetAlignmentV<Problem>();

    static constexpr index_t kAlignmentO =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentO<Problem>();
    static constexpr index_t kAlignmentBias =
        kPadSeqLenK ? 1 : Policy::template GetAlignmentBias<Problem>();

    static constexpr index_t kBlockPerCu = []() {
        if constexpr(Problem::kBlockPerCu != -1)
            return Problem::kBlockPerCu;
        else
        {
            if constexpr(kQKHeaddim == 32)
            {
                return 2;
            }
            else if constexpr(kQKHeaddim == 64)
            {
                return 2;
            }
            else if constexpr(kQKHeaddim == 96 || kQKHeaddim == 128)
            {
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                    return 1;
                else
                    return 2;
            }
            else if constexpr(kQKHeaddim == 256)
            {
                return 1;
            }
            else
            {
                return 1;
            };
        }
    }();

    static constexpr const char* name = "qr_async";

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
              typename BlockIndices>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp, // M0*kQKHeaddim tile
               const QElementFunction& q_element_func,
               const KDramBlockWindowTmp& k_dram_block_window_tmp, // N0*kQKHeaddim tile
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
               const AttentionVariant& /* unused */,
               const AttentionVariantParams& /* unused */,
               const BlockIndices& /* unused */,
               void* smem_ptr,
               DropoutType& dropout) const
    {
        // xformers path does not require the pipeline to output random values for host
        // verification, since a separate kernel is used to generate random values
        ignore = randval_dram_block_window_tmp;

        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0Sub == KDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kQKHeaddim == KDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kN1 == VDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kK1 == VDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kM0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        constexpr auto I0 = number<0>{};

        constexpr index_t n0_loops = kN0 / kN0Sub;
        constexpr index_t k1_loops = kN0 / kK1;

        // usually kN0 is 128,  kN0Sub/kK1 is 32/16
        static_assert(n0_loops >= 2, "n0_loops >= 2 required to use this pipeline");
        static_assert(k1_loops >= 2, "k1_loops >= 2 required to use this pipeline");

        constexpr auto NumKVLdsBuffers = Policy::template GetNumKVLdsBuffers<Problem>();

        constexpr index_t NumPrefetchV = Policy::template GetNumPrefetchV<Problem>();
        static_assert(n0_loops >= NumPrefetchV, "Check failed!");
        static_assert(k1_loops >= NumPrefetchV, "Check failed!");

        constexpr bool kPreloadWholeNextIterationK =
            Policy::template IsPreloadWholeNextIterationK<Problem>();

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetKVBlockGemm<Problem>();

        // SaccBlockTile size is [kM0, kK1]
        // PcompBlockTile size is [kM0, kN0]
        using SaccBlockTileType        = decltype(gemm_0.template MakeCBlockTile<kM0, kN0Sub>());
        using CombineSaccBlockTileType = decltype(gemm_0.template MakeCBlockTile<kM0, kN0>());
        using PcompBlockTileType = decltype(cast_tile<CompDataType>(CombineSaccBlockTileType{}));

        SaccBlockTileType sacc_tile;
        PcompBlockTileType pcomp_tile;

        // reduction function for softmax
        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        using MLBlockTileType = decltype(block_tile_reduce<CompDataType>(
            PcompBlockTileType{}, sequence<1>{}, f_max, CompDataType{0}));

        auto m = MLBlockTileType{};
        auto l = MLBlockTileType{};

        using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());
        OaccBlockTileType o_acc;

        auto q_dram_window = make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
                                              make_tuple(number<kM0>{}, number<kQKHeaddim>{}),
                                              q_dram_block_window_tmp.get_window_origin(),
                                              Policy::template MakeQRegTileDistribution<Problem>());

        const auto q_origin = q_dram_window.get_window_origin();
        const auto [seqlen_k_start, seqlen_k_end] =
            mask.GetTileRangeAlongX(q_origin.at(number<0>{}), number<kM0>{}, number<kN0>{});

        if(seqlen_k_end <= seqlen_k_start)
        {
            clear_tile(o_acc);
            o_acc = tile_elementwise_in(o_acc_element_func, o_acc);
            return o_acc;
        };

        auto k_dram_window =
            make_tile_window(k_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kN0Sub>{}, number<kQKHeaddim>{}),
                             {seqlen_k_start, 0},
                             Policy::template MakeKDramTileDistribution<Problem>());

        using k_tile_type = decltype(load_tile(k_dram_window));

        // only prefetch two k tiles to save vgprs consumption
        auto k_tiles = [&]() {
            if constexpr(kPreloadWholeNextIterationK)
                return statically_indexed_array<k_tile_type, n0_loops>{};
            else
                return statically_indexed_array<k_tile_type, 1>{};
        }();

        k_tiles[I0] = load_tile(k_dram_window);
        move_tile_window(k_dram_window, {kN0Sub, 0});

        auto q_tile = load_tile(q_dram_window);

        __builtin_amdgcn_sched_barrier(0x00000001);

        // provide partition_index for LDS tile window with so that warp_id is in vgpr
        array<index_t, 2> partition_index{get_warp_id<false>(), get_lane_id()};

        // K tile in LDS
        KDataType* k_lds_ptr = static_cast<KDataType*>(smem_ptr);
        auto k_lds           = make_tensor_view<address_space_enum::lds>(
            k_lds_ptr, Policy::template MakeKLdsBlockDescriptor<Problem>());
        auto k_lds_window = make_tile_window(
            k_lds, Policy::template MakeKLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        using k_lds_window_type = decltype(get_slice_tile(
            k_lds_window, sequence<0, 0>{}, sequence<kN0Sub, kQKHeaddim>{}));

        statically_indexed_array<k_lds_window_type, NumKVLdsBuffers> k_lds_windows;

        static_for<0, NumKVLdsBuffers, 1>{}([&](auto i_buf) {
            k_lds_windows[i_buf] = get_slice_tile(k_lds_window,
                                                  sequence<i_buf * kN0Sub, 0>{},
                                                  sequence<(i_buf + 1) * kN0Sub, kQKHeaddim>{});
        });

        // V tile in LDS
        auto v_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<VDataType*>(smem_ptr),
            Policy::template MakeVLdsBlockDescriptor<Problem>());
        auto v_lds_window = make_tile_window(
            v_lds, Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        using v_lds_window_type =
            decltype(get_slice_tile(v_lds_window, sequence<0, 0>{}, sequence<kN1, kK1>{}));

        statically_indexed_array<v_lds_window_type, NumKVLdsBuffers> v_lds_windows;

        static_for<0, NumKVLdsBuffers, 1>{}([&](auto i_buf) {
            v_lds_windows[i_buf] = get_slice_tile(
                v_lds_window, sequence<i_buf * kN1, 0>{}, sequence<(i_buf + 1) * kN1, kK1>{});
        });

        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kN1>{}, number<kK1>{}),
                             {0, seqlen_k_start},
                             Policy::template MakeVDramTileDistribution<Problem>());

        const auto f_exp = [&](CompDataType x) {
            if constexpr(std::is_same_v<CompDataType, float>)
            {
                return __expf(x);
            }
            else
            {
                return exp(x);
            }
        };

        clear_tile(o_acc);
        set_tile(m, -numeric<CompDataType>::infinity());
        clear_tile(l);

        const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();
        auto bias_dram_window =
            make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kM0>{}, number<kN0>{}),
                             {bias_origin.at(number<0>{}), seqlen_k_start},
                             Policy::template MakeBiasDramTileDistribution<Problem>());

        // assuming no random values need be saved, this is true when the pipeline is called from
        // xformers, since we have a separate kernel to generated random values
        auto null_randval_window = [&]() {
            if constexpr(kHasDropout)
            {
                // need to pass a null_randval_dram and tile window to the BlockDropout operator to
                // make it works
                const auto null_randval_dram = [&]() {
                    const auto null_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                        static_cast<uint8_t*>(nullptr),
                        make_tuple(1, 1),
                        make_tuple(1, 1),
                        number<1>{},
                        number<1>{});

                    return pad_tensor_view(null_dram_naive,
                                           make_tuple(number<1>{}, number<1>{}),
                                           sequence<true, true>{});
                }();

                return make_tile_window(
                    null_randval_dram, make_tuple(number<1>{}, number<1>{}), {0, 0});
            }
            else
                return make_null_tile_window(make_tuple(number<1>{}, number<1>{}));
        }();

        q_tile = tile_elementwise_in(q_element_func, q_tile);

        auto seqlen_k_curr = seqlen_k_start;

        using v_tile_type = decltype(load_tile(v_dram_window));

        statically_indexed_array<v_tile_type, k1_loops> v_tiles;

        do
        {
            // STAGE 1, Gemm_0 ( S = Q@K )
            if constexpr(kPreloadWholeNextIterationK) // used when kM0 = 64
            {
                if(seqlen_k_curr == seqlen_k_start) // at first iteration
                {
                    if(seqlen_k_curr < seqlen_k_end - kN0) // not the last iteration
                    {
                        static_for<0, n0_loops, 1>{}([&](auto i_n0) {
                            store_tile(k_lds_windows[number<i_n0 % NumKVLdsBuffers>{}],
                                       tile_elementwise_in(k_element_func, k_tiles[number<i_n0>{}]),
                                       partition_index);

                            if constexpr(i_n0 < n0_loops - 1)
                            {
                                k_tiles[number<i_n0 + 1>{}] = load_tile(k_dram_window);
                                move_tile_window(k_dram_window, {kN0Sub, 0});
                            };

                            if constexpr(i_n0 == n0_loops - 1)
                            {
                                v_tiles[I0] = load_tile(v_dram_window);
                                move_tile_window(v_dram_window, {0, kK1});

                                // prefetch all k_tiles for next iteration
                                static_for<0, n0_loops, 1>{}([&](auto ii_n0) {
                                    k_tiles[number<ii_n0>{}] = load_tile(k_dram_window);
                                    move_tile_window(k_dram_window, {kN0Sub, 0});
                                });
                            };

                            block_sync_lds();
                            gemm_0(
                                sacc_tile, q_tile, k_lds_windows[number<i_n0 % NumKVLdsBuffers>{}]);

                            sacc_tile     = tile_elementwise_in(s_acc_element_func, sacc_tile);
                            auto tmp_tile = cast_tile<CompDataType>(sacc_tile);
                            set_slice_tile(pcomp_tile,
                                           tmp_tile,
                                           sequence<0, i_n0 * kN0Sub>{},
                                           sequence<kM0, (i_n0 + 1) * kN0Sub>{});
                        });
                    }
                    else // the iteration is also the last iteration
                    {
                        static_for<0, n0_loops, 1>{}([&](auto i_n0) {
                            store_tile(k_lds_windows[number<i_n0 % NumKVLdsBuffers>{}],
                                       tile_elementwise_in(k_element_func, k_tiles[number<i_n0>{}]),
                                       partition_index);

                            if constexpr(i_n0 < n0_loops - 1)
                            {
                                k_tiles[number<i_n0 + 1>{}] = load_tile(k_dram_window);
                                move_tile_window(k_dram_window, {kN0Sub, 0});
                            };

                            if constexpr(i_n0 == n0_loops - 1)
                            {
                                v_tiles[I0] = load_tile(v_dram_window);
                                move_tile_window(v_dram_window, {0, kK1});
                            };

                            block_sync_lds();
                            gemm_0(
                                sacc_tile, q_tile, k_lds_windows[number<i_n0 % NumKVLdsBuffers>{}]);

                            sacc_tile     = tile_elementwise_in(s_acc_element_func, sacc_tile);
                            auto tmp_tile = cast_tile<CompDataType>(sacc_tile);
                            set_slice_tile(pcomp_tile,
                                           tmp_tile,
                                           sequence<0, i_n0 * kN0Sub>{},
                                           sequence<kM0, (i_n0 + 1) * kN0Sub>{});
                        });
                    };
                }
                else // at intermediate and last iteration
                {
                    if(seqlen_k_curr < seqlen_k_end - kN0) // intermediate iteration
                    {
                        static_for<0, n0_loops, 1>{}([&](auto i_n0) {
                            store_tile(k_lds_windows[number<i_n0 % NumKVLdsBuffers>{}],
                                       tile_elementwise_in(k_element_func, k_tiles[number<i_n0>{}]),
                                       partition_index);

                            if constexpr(i_n0 == 0)
                            {
                                v_tiles[I0] = load_tile(v_dram_window);
                                move_tile_window(v_dram_window, {0, kK1});
                            };

                            // prefetch k_tile for next iteration
                            k_tiles[i_n0] = load_tile(k_dram_window);
                            move_tile_window(k_dram_window, {kN0Sub, 0});

                            block_sync_lds();
                            gemm_0(
                                sacc_tile, q_tile, k_lds_windows[number<i_n0 % NumKVLdsBuffers>{}]);

                            sacc_tile     = tile_elementwise_in(s_acc_element_func, sacc_tile);
                            auto tmp_tile = cast_tile<CompDataType>(sacc_tile);
                            set_slice_tile(pcomp_tile,
                                           tmp_tile,
                                           sequence<0, i_n0 * kN0Sub>{},
                                           sequence<kM0, (i_n0 + 1) * kN0Sub>{});
                        });
                    }
                    else // last iteration
                    {
                        static_for<0, n0_loops, 1>{}([&](auto i_n0) {
                            store_tile(k_lds_windows[number<i_n0 % NumKVLdsBuffers>{}],
                                       tile_elementwise_in(k_element_func, k_tiles[number<i_n0>{}]),
                                       partition_index);

                            if constexpr(i_n0 == 0)
                            {
                                v_tiles[I0] = load_tile(v_dram_window);
                                move_tile_window(v_dram_window, {0, kK1});
                            };

                            block_sync_lds();
                            gemm_0(
                                sacc_tile, q_tile, k_lds_windows[number<i_n0 % NumKVLdsBuffers>{}]);

                            sacc_tile     = tile_elementwise_in(s_acc_element_func, sacc_tile);
                            auto tmp_tile = cast_tile<CompDataType>(sacc_tile);
                            set_slice_tile(pcomp_tile,
                                           tmp_tile,
                                           sequence<0, i_n0 * kN0Sub>{},
                                           sequence<kM0, (i_n0 + 1) * kN0Sub>{});
                        });
                    };
                }
            }
            else // only preload one unroll of K for next iteration, used when kM0=128
            {
                static_for<0, n0_loops, 1>{}([&](auto i_n0) {
                    store_tile(k_lds_windows[number<i_n0 % NumKVLdsBuffers>{}],
                               tile_elementwise_in(k_element_func, k_tiles[I0]),
                               partition_index);

                    __builtin_amdgcn_sched_barrier(0x00000001);

                    if constexpr(i_n0 < n0_loops - 1)
                    {
                        k_tiles[I0] = load_tile(k_dram_window);
                        move_tile_window(k_dram_window, {kN0Sub, 0});
                    };

                    if constexpr(i_n0 == n0_loops - 1)
                    {
                        v_tiles[I0] = load_tile(v_dram_window);
                        move_tile_window(v_dram_window, {0, kK1});
                    };

                    __builtin_amdgcn_sched_barrier(0x00000001);

                    block_sync_lds();

                    gemm_0(sacc_tile, q_tile, k_lds_windows[number<i_n0 % NumKVLdsBuffers>{}]);

                    sacc_tile     = tile_elementwise_in(s_acc_element_func, sacc_tile);
                    auto tmp_tile = cast_tile<CompDataType>(sacc_tile);
                    set_slice_tile(pcomp_tile,
                                   tmp_tile,
                                   sequence<0, i_n0 * kN0Sub>{},
                                   sequence<kM0, (i_n0 + 1) * kN0Sub>{});
                });
            }

            __builtin_amdgcn_sched_barrier(0x00000001);

            const auto bias_tile = load_tile(bias_dram_window); // load bias tile

            // STAGE 2, scale_s, add bias, mask, softmax
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, pcomp_tile);

                tile_elementwise_inout(
                    [&](auto& x, const auto y) {
                        x += type_convert<CompDataType>(bias_element_func(y));
                    },
                    pcomp_tile,
                    bias_tile);
            }
            else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
            {
                constexpr auto pcomp_spans = decltype(pcomp_tile)::get_distributed_spans();
                sweep_tile_span(pcomp_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(pcomp_spans[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            pcomp_tile.get_tile_distribution(), make_tuple(idx0, idx1));

                        const auto row = q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                        const auto col = seqlen_k_curr + tile_idx.at(number<1>{});
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);

                        pcomp_tile(i_j_idx) *= scale_s;
                        position_encoding.update(pcomp_tile(i_j_idx), row, col);
                    });
                });
            }
            else
            {
                tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, pcomp_tile);
            }

            move_tile_window(bias_dram_window, {0, kN0});

            if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
            {
                bool need_perpixel_check = mask.IsEdgeTile(
                    q_origin.at(number<0>{}), seqlen_k_curr, number<kM0>{}, number<kN0>{});
                if(need_perpixel_check)
                {
                    set_tile_if(pcomp_tile, -numeric<CompDataType>::infinity(), [&](auto tile_idx) {
                        const auto row = q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                        const auto col = seqlen_k_curr + tile_idx.at(number<1>{});
                        return mask.IsOutOfBound(row, col);
                    });
                }
            }

            __builtin_amdgcn_sched_barrier(0x00000001);

            const auto m_old = m;

            block_tile_reduce(m, pcomp_tile, sequence<1>{}, f_max);
            block_tile_reduce_sync(m, f_max, bool_constant<false>{});

            __builtin_amdgcn_sched_barrier(0);

            auto v_shuffled_tile = make_static_distributed_tensor<VDataType>(
                Policy::template MakeShuffledVRegTileDistribution<Problem>());
            shuffle_tile(v_shuffled_tile, tile_elementwise_in(v_element_func, v_tiles[I0]));

            // check whether first V-LdsBufer overlap with last K-LdsBuffer,
            // this does not occur when k1_loops == 2 and NumKVLdsBuffers == 4
            if constexpr((n0_loops - 1) % NumKVLdsBuffers == 2 % NumKVLdsBuffers)
            {
                __builtin_amdgcn_s_barrier();
            };

            store_tile(
                v_lds_windows[number<2 % NumKVLdsBuffers>{}], v_shuffled_tile, partition_index);

            __builtin_amdgcn_sched_barrier(0x00000001);

            static_for<1, NumPrefetchV, 1>{}([&](auto i_k1) {
                v_tiles[i_k1] = load_tile(v_dram_window);
                move_tile_window(v_dram_window, {0, kK1});
            });

            __builtin_amdgcn_sched_barrier(0);

            constexpr auto p_spans = decltype(pcomp_tile)::get_distributed_spans();
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);

                if(m[i_idx] == -numeric<CompDataType>::infinity())
                {
                    sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        pcomp_tile(i_j_idx)    = type_convert<CompDataType>(0.0f);
                    });
                }
                else
                {
                    sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        pcomp_tile(i_j_idx)    = f_exp(pcomp_tile[i_j_idx] - m[i_idx]);
                    });
                }
            });

            auto rowsum_p =
                block_tile_reduce<CompDataType>(pcomp_tile, sequence<1>{}, f_sum, CompDataType{0});

            block_tile_reduce_sync(rowsum_p, f_sum, bool_constant<false>{});

            // adjust o_acc[] according to the update between m and m_old
            constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
            sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);

                if(m[i_idx] == -numeric<CompDataType>::infinity())
                {
                    l(i_idx) = rowsum_p[i_idx];
                }
                else
                {
                    const auto tmp = f_exp(m_old[i_idx] - m[i_idx]);
                    l(i_idx)       = tmp * l[i_idx] + rowsum_p[i_idx];
                    sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        o_acc(i_j_idx) *= tmp;
                    });
                }
            });

            __builtin_amdgcn_sched_barrier(0x00000001);

            if constexpr(kHasDropout)
            {
                auto randval_lds_ptr =
                    reinterpret_cast<char*>(smem_ptr) + Policy::template GetSmemSizeKV<Problem>();

                dropout.template Run<decltype(gemm_0), CompDataType, uint8_t>(
                    randval_lds_ptr, seqlen_k_curr, pcomp_tile, null_randval_window);
            }

            seqlen_k_curr += kN0;

            __builtin_amdgcn_sched_barrier(0x00000001);

            auto p = cast_tile<PDataType>(tile_elementwise_in(p_compute_element_func, pcomp_tile));

            __builtin_amdgcn_sched_barrier(0x00000001);

            // STAGE 3, Gemm_1 ( O = P@V )
            static_for<0, k1_loops, 1>{}([&](auto i_k1) {
                if constexpr(i_k1 < k1_loops - NumPrefetchV)
                {
                    v_tiles[number<i_k1 % NumPrefetchV>{}] = load_tile(v_dram_window);
                    move_tile_window(v_dram_window, {0, kK1});
                };

                if constexpr(i_k1 == k1_loops - NumPrefetchV)
                {
                    if constexpr(!kPreloadWholeNextIterationK)
                    {
                        if(seqlen_k_curr < seqlen_k_end)
                        {
                            k_tiles[I0] = load_tile(k_dram_window);
                            move_tile_window(k_dram_window, {kN0Sub, 0});
                        };
                    }
                };

                block_sync_lds();
                gemm_1(
                    o_acc,
                    get_slice_tile(p, sequence<0, i_k1 * kK1>{}, sequence<kM0, (i_k1 + 1) * kK1>{}),
                    v_lds_windows[number<(i_k1 + 2) % NumKVLdsBuffers>{}]);

                if constexpr(i_k1 < k1_loops - 1)
                {
                    shuffle_tile(v_shuffled_tile,
                                 tile_elementwise_in(v_element_func,
                                                     v_tiles[number<(i_k1 + 1) % NumPrefetchV>{}]));
                    store_tile(v_lds_windows[number<(i_k1 + 3) % NumKVLdsBuffers>{}],
                               v_shuffled_tile,
                               partition_index);
                };
            });

            // check whether last V-LdsBuffer overlap with first K-LdsBuffer,
            // this does not occur when k1_loops == 2 and NumKVLdsBuffers == 4
            if constexpr((k1_loops - 1 + 2) % NumKVLdsBuffers == 0)
            {
                __builtin_amdgcn_s_barrier();
            };
        } while(seqlen_k_curr < seqlen_k_end);

        // store lse
        if constexpr(kStoreLSE)
        {
            auto lse = make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

            constexpr auto lse_spans = decltype(lse)::get_distributed_spans();
            sweep_tile_span(lse_spans[number<0>{}], [&, m_ = m, l_ = l](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                lse(i_idx)           = m_[i_idx] + log(l_[i_idx]);
            });

            store_tile(lse_dram_window_tmp, tile_elementwise_in(lse_element_func, lse));
        }

        // finally, O
        constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();

        sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);

                if(m[i_idx] == -numeric<CompDataType>::infinity())
                    o_acc(i_j_idx) = 0.0f;
                else
                    o_acc(i_j_idx) *= 1.0f / l[i_idx];
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
        ignore = sink_v;

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
                          dropout);
    }
};

} // namespace ck_tile
