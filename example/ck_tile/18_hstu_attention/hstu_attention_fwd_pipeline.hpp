// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_dropout.hpp"

#include "hstu_attention_fwd_pipeline_default_policy.hpp"

namespace ck_tile {

template <typename Problem_, typename Policy_ = HstuAttentionFwdPipelineQRKSVSDefaultPolicy>
struct HstuAttentionFwdPipelineQRKSVS
{
    using Problem         = remove_cvref_t<Problem_>;
    using Policy          = remove_cvref_t<Policy_>;
    using QKVDataType     = remove_cvref_t<typename Problem::InOutDataType>;
    using GemmAccDataType = remove_cvref_t<typename Problem::GemmAccDataType>;
    using CompDataType    = remove_cvref_t<typename Problem::CompDataType>;
    using BiasDataType    = remove_cvref_t<typename Problem::BiasDataType>;
    using PDataType       = remove_cvref_t<typename Problem::InOutDataType>;
    using ODataType       = remove_cvref_t<typename Problem::InOutDataType>;
    using HstuMask        = remove_cvref_t<typename Problem::HstuMask>;

    using HstuAttentionTileShape     = remove_cvref_t<typename Problem::HstuAttentionTileShape>;
    using VLayout                    = remove_cvref_t<typename HstuAttentionTileShape::VLayout>;
    static constexpr bool kQLoadOnce = true;
    static_assert(kQLoadOnce == Policy::QLoadOnce);

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kM0           = HstuAttentionTileShape::kM0;
    static constexpr index_t kN0           = HstuAttentionTileShape::kN0;
    static constexpr index_t kK0           = HstuAttentionTileShape::kK0;
    static constexpr index_t kN1           = HstuAttentionTileShape::kN1;
    static constexpr index_t kK1           = HstuAttentionTileShape::kK1;
    static constexpr index_t kQKHeaddim    = HstuAttentionTileShape::kQKHeaddim;
    static constexpr index_t kSubQKHeaddim = HstuAttentionTileShape::kSubQKHeaddim;

    static_assert(kSubQKHeaddim <= 256, "hdim bigger than 256 is not suitable for this pipeline!");

    static constexpr bool kIsJagged   = Problem::kIsJagged;
    static constexpr auto kHasBias    = Problem::kHasBias;
    static constexpr bool kHasDropout = Problem::kHasDropout;

    static constexpr bool kPadSeqLenQ   = Problem::Traits::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK   = Problem::Traits::kPadSeqLenK;
    static constexpr bool kPadHeadDimQK = Problem::Traits::kPadHeadDimQK;
    static constexpr bool kPadHeadDimV =
        (kQKHeaddim < kSubQKHeaddim) ? 1 : Problem::Traits::kPadHeadDimV;

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
    static constexpr index_t kAlignmentQ =
        kPadHeadDimQK ? 1 : Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK =
        kPadHeadDimQK ? 1 : Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV = []() {
        if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            return Problem::Traits::kPadHeadDimV ? 1 : Policy::template GetAlignmentV<Problem>();
        else
            return kPadSeqLenK ? 1 : Policy::template GetAlignmentV<Problem>();
    }();

    static constexpr index_t kAlignmentO =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentO<Problem>();
    static constexpr index_t kAlignmentBias =
        kPadSeqLenK ? 1 : Policy::template GetAlignmentBias<Problem>();

    static constexpr index_t kBlockPerCu = []() {
        if constexpr(Problem::Traits::kBlockPerCu != -1)
            return Problem::Traits::kBlockPerCu;
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
                if constexpr(kHasBias)
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

    static constexpr const char* name = "qr_hstu";

    using DropoutType = std::conditional_t<kHasDropout, BlockDropout, NullBlockDropout>;

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename QElementFunction,
              typename KElementFunction,
              typename VElementFunction,
              typename BiasElementFunction,
              typename SAccElementFunction,
              typename PComputeElementFunction,
              typename OAccElementFunction>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp, // M0*kSubQKHeaddim tile
               const QElementFunction& q_element_func,
               const KDramBlockWindowTmp& k_dram_block_window_tmp, // N0*kSubQKHeaddim tile
               const KElementFunction& k_element_func,
               const VDramBlockWindowTmp& v_dram_block_window_tmp, // N1*K1 tile
               const VElementFunction& v_element_func,
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
               const BiasElementFunction& bias_element_func,
               const SAccElementFunction& s_acc_element_func,
               const PComputeElementFunction& p_compute_element_func,
               const OAccElementFunction& o_acc_element_func,
               HstuMask mask,
               float scale_s,
               void* smem_ptr,
               DropoutType& dropout) const
    {
        ignore = q_element_func;
        ignore = k_element_func;

        static_assert(
            std::is_same_v<QKVDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<QKVDataType,
                               remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<QKVDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kK0 == KDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kN1 == VDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kK1 == VDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kM0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        constexpr auto I0 = number<0>{};

        constexpr index_t k0_loops = kQKHeaddim / kK0;
        constexpr index_t k1_loops = kN0 / kK1;
        static_assert(2 <= k0_loops);
        static_assert(2 <= k1_loops);

        constexpr auto NumKLdsBuffers = Policy::template GetNumKLdsBuffers<Problem>();
        constexpr auto NumVLdsBuffers = Policy::template GetNumVLdsBuffers<Problem>();
        constexpr auto NumPrefetchV   = Policy::template GetNumPrefetchV<Problem>();

        static_assert(NumKLdsBuffers >= 2);
        static_assert(NumPrefetchV >= 2);

        auto q_dram_window = make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
                                              q_dram_block_window_tmp.get_window_lengths(),
                                              q_dram_block_window_tmp.get_window_origin(),
                                              Policy::template MakeQRegTileDistribution<Problem>());

        const auto q_origin = q_dram_window.get_window_origin();
        const auto [seqlen_k_start, seqlen_k_end] =
            mask.GetTileRangeAlongX(q_origin.at(number<0>{}), number<kM0>{}, number<kN0>{});

        auto k_dram_block_window =
            make_tile_window(k_dram_block_window_tmp.get_bottom_tensor_view(),
                             k_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_k_start, 0});

        auto k_dram_window =
            make_tile_window(k_dram_block_window.get_bottom_tensor_view(),
                             k_dram_block_window.get_window_lengths(),
                             k_dram_block_window.get_window_origin(),
                             Policy::template MakeKDramTileDistribution<Problem>());

        auto k_tile = load_tile(k_dram_window);
        move_tile_window(k_dram_window, {0, kK0});

        auto q_tile = load_tile(q_dram_window);

        __builtin_amdgcn_sched_barrier(0);

        // K tile in LDS
        QKVDataType* k_lds_ptr = static_cast<QKVDataType*>(smem_ptr);
        auto k_lds             = make_tensor_view<address_space_enum::lds>(
            k_lds_ptr, Policy::template MakeKLdsBlockDescriptor<Problem>());
        auto k_lds_window = make_tile_window(
            k_lds, Policy::template MakeKLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        using k_lds_window_type =
            decltype(get_slice_tile(k_lds_window, sequence<0, 0>{}, sequence<kN0, kK0>{}));

        statically_indexed_array<k_lds_window_type, NumKLdsBuffers> k_lds_windows;

        static_for<0, NumKLdsBuffers, 1>{}([&](auto i_buf) {
            k_lds_windows[i_buf] = get_slice_tile(
                k_lds_window, sequence<i_buf * kN0, 0>{}, sequence<(i_buf + 1) * kN0, kK0>{});
        });

        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp.get_bottom_tensor_view(),
                             v_dram_block_window_tmp.get_window_lengths(),
                             {0, seqlen_k_start}, // TODO: hdim split?
                             Policy::template MakeVDramTileDistribution<Problem>());
        // V tile in LDS
        auto v_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<QKVDataType*>(static_cast<char*>(smem_ptr) +
                                           Policy::template GetExclusiveKLdsBytes<Problem>()),
            Policy::template MakeVLdsBlockDescriptor<Problem>());
        auto v_lds_window = make_tile_window(
            v_lds, Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        using v_tile_type = decltype(load_tile(v_dram_window));

        statically_indexed_array<v_tile_type, NumPrefetchV> v_tiles;

        using v_lds_window_type =
            decltype(get_slice_tile(v_lds_window, sequence<0, 0>{}, sequence<kN1, kK1>{}));

        statically_indexed_array<v_lds_window_type, NumVLdsBuffers> v_lds_windows;

        static_for<0, NumVLdsBuffers, 1>{}([&](auto i_buf) {
            v_lds_windows[i_buf] = get_slice_tile(
                v_lds_window, sequence<i_buf * kN1, 0>{}, sequence<(i_buf + 1) * kN1, kK1>{});
        });

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetKVBlockGemm<Problem>();

        using SaccBlockTileType = decltype(gemm_0.MakeCBlockTile());
        auto s_acc              = SaccBlockTileType{};

        // reduction function for softmax
        const auto f_silu = [](CompDataType x) {
            auto one = ck_tile::type_convert<CompDataType>(1.0f);

            auto sigmod_val = one / (one + exp(-x));

            return sigmod_val * x;
        };

        using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());

        // init Oacc, M, L
        auto o_acc = OaccBlockTileType{};

        clear_tile(o_acc);

        const auto num_loops = integer_divide_ceil(seqlen_k_end - seqlen_k_start, kN0);

        // check early exit if no work to do
        if constexpr(HstuMask::IsMasking || kPadSeqLenK)
        {
            if(num_loops <= 0)
            {
                return o_acc;
            }
        }

        const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();
        auto bias_dram_window =
            make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                             bias_dram_block_window_tmp.get_window_lengths(),
                             {bias_origin.at(number<0>{}), seqlen_k_start}, // M/N
                             Policy::template MakeBiasDramTileDistribution<decltype(gemm_0)>());

        auto null_randval_window = [&]() {
            if constexpr(kHasDropout)
            {
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

        index_t i_loop = 0;

        do
        {
            static_for<0, k0_loops - 1, 1>{}([&](auto i_k0) {
                store_tile(k_lds_windows[number<i_k0 % NumKLdsBuffers>{}],
                           tile_elementwise_in(k_element_func, k_tile));
                if constexpr(i_k0 == 0)
                    clear_tile(s_acc);

                k_tile = load_tile(k_dram_window);
                if constexpr(i_k0 < k0_loops - 2)
                    move_tile_window(k_dram_window, {0, kK0});

                block_sync_lds();
                // execute current unroll of gemm_0
                gemm_0(s_acc,
                       get_slice_tile(
                           q_tile, sequence<0, i_k0 * kK0>{}, sequence<kM0, (i_k0 + 1) * kK0>{}),
                       k_lds_windows[number<i_k0 % NumKLdsBuffers>{}]);
            });

            store_tile(k_lds_windows[number<(k0_loops - 1) % NumKLdsBuffers>{}],
                       tile_elementwise_in(k_element_func, k_tile));

            // prefetch first v_tile
            v_tiles[I0] = load_tile(v_dram_window);
            move_tile_window(v_dram_window, {0, kK1});

            block_sync_lds();
            gemm_0(s_acc,
                   get_slice_tile(q_tile,
                                  sequence<0, (k0_loops - 1) * kK0>{},
                                  sequence<kM0, k0_loops * kK0>{}),
                   k_lds_windows[number<(k0_loops - 1) % NumKLdsBuffers>{}]);

            __builtin_amdgcn_sched_barrier(0);

            const auto bias_tile = load_tile(bias_dram_window); // load bias tile

            static_for<1, NumPrefetchV, 1>{}([&](auto i_buf) {
                v_tiles[i_buf] = load_tile(v_dram_window);
                move_tile_window(v_dram_window, {0, kK1});
            });

            // STAGE 2, scale_s, add bias, mask, siLU
            if constexpr(kHasBias)
            {
                s_acc = tile_elementwise_in(s_acc_element_func, s_acc);
                tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, s_acc);
                tile_elementwise_inout(
                    [&](auto& x, const auto& y) {
                        x += type_convert<GemmAccDataType>(bias_element_func(y));
                    },
                    s_acc,
                    bias_tile);
            }
            else
            {
                s_acc = tile_elementwise_in(s_acc_element_func, s_acc);
                tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, s_acc);
            }

            move_tile_window(bias_dram_window, {0, kN0});
            if constexpr(HstuMask::IsMasking)
            {
                const auto k_origin = k_dram_block_window.get_window_origin();
                set_tile_if(s_acc, type_convert<GemmAccDataType>(0), [&](auto tile_idx) {
                    const auto row = q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                    const auto col = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                    return !mask.IsTokenPairInsideMask(row, col);
                });
            }
            else if constexpr(kPadSeqLenK)
            {
                const auto k_origin = k_dram_block_window.get_window_origin();
                set_tile_if(s_acc, type_convert<GemmAccDataType>(0), [&](auto tile_idx) {
                    if(q_origin.at(number<0>{}) + kM0 <= mask.max_uih_len && i_loop < num_loops - 1)
                        return false;
                    const auto row = q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                    const auto col = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                    return !mask.IsTokenPairInsideMask(row, col);
                });
            }
            else
            {
                if(q_origin.at(number<0>{}) + kM0 > mask.max_uih_len)
                {
                    const auto k_origin = k_dram_block_window.get_window_origin();
                    set_tile_if(s_acc, type_convert<GemmAccDataType>(0), [&](auto tile_idx) {
                        const auto row = q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                        const auto col = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                        return !mask.IsTokenPairInsideMask(row, col);
                    });
                };
            };

            auto s = cast_tile<CompDataType>(s_acc); // S{j}

            s = tile_elementwise_in(f_silu, s);

            if constexpr(kHasDropout)
            {
                auto randval_lds_ptr =
                    reinterpret_cast<char*>(smem_ptr) + Policy::template GetSmemSizeK<Problem>();

                dropout.template Run<decltype(gemm_0), CompDataType, uint8_t>(
                    randval_lds_ptr, seqlen_k_start + i_loop * kN0, s, null_randval_window);
            }

            __builtin_amdgcn_sched_barrier(0x7f);

            if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                auto v_shuffle_tmp = make_static_distributed_tensor<QKVDataType>(
                    Policy::template MakeShuffledVRegBlockDescriptor<Problem>());
                shuffle_tile(v_shuffle_tmp, v_tiles[I0]);

                // ensure gemm_0 has finished access of k-Lds for all warps
                if constexpr(Policy::template IsFirstVLdsBufferOverlapLastKLdsBuffer<Problem>())
                    __builtin_amdgcn_s_barrier();

                store_tile(
                    v_lds_windows[I0],
                    tile_elementwise_in(v_element_func, v_shuffle_tmp)); // store the prefetch
            }
            else
            {
                // ensure gemm_0 has finished access of k-Lds for all warps
                if constexpr(Policy::template IsFirstVLdsBufferOverlapLastKLdsBuffer<Problem>())
                    __builtin_amdgcn_s_barrier();

                store_tile(v_lds_windows[I0],
                           tile_elementwise_in(v_element_func, v_tiles[I0])); // store the prefetch
            }

            __builtin_amdgcn_sched_barrier(0);

            const auto p = cast_tile<PDataType>(tile_elementwise_in(p_compute_element_func, s));

            move_tile_window(k_dram_window, {kN0, -(k0_loops - 1) * kK0});
            k_tile = load_tile(k_dram_window);
            move_tile_window(k_dram_window, {0, kK0});

            __builtin_amdgcn_sched_barrier(0);

            static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
                if constexpr(i_k1 < k1_loops - NumPrefetchV)
                    v_tiles[number<i_k1 % NumPrefetchV>{}] = load_tile(v_dram_window);

                block_sync_lds();
                gemm_1(
                    o_acc,
                    get_slice_tile(p, sequence<0, i_k1 * kK1>{}, sequence<kM0, (i_k1 + 1) * kK1>{}),
                    v_lds_windows[number<i_k1 % NumVLdsBuffers>{}]);

                if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
                {
                    auto v_shuffle_tmp = make_static_distributed_tensor<QKVDataType>(
                        Policy::template MakeShuffledVRegBlockDescriptor<Problem>());
                    shuffle_tile(v_shuffle_tmp, v_tiles[number<(i_k1 + 1) % NumPrefetchV>{}]);
                    store_tile(v_lds_windows[number<(i_k1 + 1) % NumVLdsBuffers>{}],
                               tile_elementwise_in(v_element_func, v_shuffle_tmp));
                }
                else
                {
                    store_tile(v_lds_windows[number<(i_k1 + 1) % NumVLdsBuffers>{}],
                               tile_elementwise_in(v_element_func,
                                                   v_tiles[number<(i_k1 + 1) % NumPrefetchV>{}]));
                }

                if constexpr(i_k1 < k1_loops - NumPrefetchV)
                    move_tile_window(v_dram_window, {0, kK1});
            });

            // move K tile windows
            move_tile_window(k_dram_block_window, {kN0, 0});

            block_sync_lds();
            gemm_1(o_acc,
                   get_slice_tile(p, sequence<0, (k1_loops - 1) * kK1>{}, sequence<kM0, kN0>{}),
                   v_lds_windows[number<(k1_loops - 1) % NumVLdsBuffers>{}]);

            if constexpr(Policy::template IsFirstKLdsBufferOverlapLastVLdsBuffer<Problem>())
            {
                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_barrier();
            };
        } while(++i_loop < num_loops);

        o_acc = tile_elementwise_in(o_acc_element_func, o_acc);

        return o_acc;
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,       // M0*K0 tile
               const KDramBlockWindowTmp& k_dram_block_window_tmp,       // N0*K0 tile
               const VDramBlockWindowTmp& v_dram_block_window_tmp,       // N1*K1 tile
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
               HstuMask mask,
               float scale_s,
               void* smem_ptr,
               DropoutType& dropout) const
    {
        return operator()(q_dram_block_window_tmp,
                          identity{},
                          k_dram_block_window_tmp,
                          identity{},
                          v_dram_block_window_tmp,
                          identity{},
                          bias_dram_block_window_tmp,
                          identity{},
                          identity{},
                          identity{},
                          identity{},
                          mask,
                          scale_s,
                          smem_ptr,
                          dropout);
    }
};

} // namespace ck_tile
