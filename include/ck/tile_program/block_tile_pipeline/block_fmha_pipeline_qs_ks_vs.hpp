// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/load_tile.hpp"
#include "ck/tile_program/tile/store_tile.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/tile/tile_gemm_shape.hpp"
#include "ck/tile_program/tile/slice_tile.hpp"
#include "ck/tile_program/warp_tile/warp_gemm.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qs_ks_vs_default_policy.hpp"
#include "ck/tile_program/block_tile/block_reduce.hpp"
#include "ck/tile_program/tile/shuffle_distributed_tensor.hpp"

namespace ck {
namespace tile_program {
namespace block {

// This pipeline is qkv all located in LDS
template <typename Problem, typename Policy = BlockFmhaPipelineQSKSVSDefaultPolicy>
struct BlockFmhaPipelineQSKSVS
{
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
    using FmhaMask            = remove_cvref_t<typename Problem::FmhaMask>;

    using BlockFmhaShape             = remove_cvref_t<typename Problem::BlockFmhaShape>;
    using VLayout                    = remove_cvref_t<typename BlockFmhaShape::VLayout>;
    static constexpr bool kQLoadOnce = false;
    static_assert(kQLoadOnce == Policy::QLoadOnce);
    static constexpr bool kIsFp8 = Problem::kIsFp8;

    static constexpr index_t kBlockPerCu = Problem::kBlockPerCu;
    static constexpr index_t kBlockSize  = Problem::kBlockSize;

    static constexpr index_t kM0            = BlockFmhaShape::kM0;
    static constexpr index_t kN0            = BlockFmhaShape::kN0;
    static constexpr index_t kK0            = BlockFmhaShape::kK0;
    static constexpr index_t kN1            = BlockFmhaShape::kN1;
    static constexpr index_t kK1            = BlockFmhaShape::kK1;
    static constexpr index_t kK0BlockLength = BlockFmhaShape::kK0BlockLength;

    static constexpr bool kIsGroupMode = Problem::kIsGroupMode;
    static constexpr bool kPadSeqLenQ  = Problem::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK  = Problem::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ = Problem::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV = Problem::kPadHeadDimV;
    static constexpr bool kHasBias     = Problem::kHasBias;
    static constexpr bool kStoreLSE    = Problem::kStoreLSE;

    __host__ __device__ static constexpr ck::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    __host__ __device__ static constexpr ck::index_t GetSmemSizeQ()
    {
        return Policy::template GetSmemSizeQ<Problem>();
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename QElementFunction,
              typename KElementFunction,
              typename VElementFunction,
              typename BiasElementFunction,
              typename LSEElementFunction>
    __host__ __device__ auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp, // M0*K0 tile
               const QElementFunction& q_element_func,
               const KDramBlockWindowTmp& k_dram_block_window_tmp, // N0*K0 tile
               const KElementFunction& k_element_func,
               const VDramBlockWindowTmp& v_dram_block_window_tmp, // N1*K1 tile
               const VElementFunction& v_element_func,
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
               const BiasElementFunction& bias_element_func,
               LSEDramBlockWindowTmp& lse_dram_window_tmp, // M0*1 tile
               const LSEElementFunction& lse_element_func,
               FmhaMask mask,
               float scale,
               void* smem_ptr) const
    {
        static_assert(
            is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kN0 == KDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kK0 == KDramBlockWindowTmp{}.GetWindowLengths()[Number<1>{}] &&
                          kN1 == VDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kK1 == VDramBlockWindowTmp{}.GetWindowLengths()[Number<1>{}] &&
                          kM0 == BiasDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kN0 == BiasDramBlockWindowTmp{}.GetWindowLengths()[Number<1>{}],
                      "wrong!");

        // Q tile in LDS
        auto q_lds = make_tensor_view<AddressSpaceEnum::Lds>(
            reinterpret_cast<QDataType*>(smem_ptr),
            Policy::template MakeQLdsBlockDescriptor<Problem>());
        auto q_lds_window =
            make_tile_window(q_lds, make_tuple(Number<kM0>{}, Number<kK0>{}), {0, 0});

        // K tile in LDS
        KDataType* k_lds_ptr = static_cast<KDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQ<Problem>()));
        auto k_lds           = make_tensor_view<AddressSpaceEnum::Lds>(
            k_lds_ptr, Policy::template MakeKLdsBlockDescriptor<Problem>());
        auto k_lds_window =
            make_tile_window(k_lds, make_tuple(Number<kN0>{}, Number<kK0>{}), {0, 0});

        // V tile in LDS
        auto v_lds = make_tensor_view<AddressSpaceEnum::Lds>(
            reinterpret_cast<VDataType*>(smem_ptr),
            Policy::template MakeVLdsBlockDescriptor<Problem>());
        auto v_lds_window = make_tile_window(
            v_lds, Policy::template MakeVLdsBlockDescriptor<Problem>().GetLengths(), {0, 0});

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetKVBlockGemm<Problem>();

        using SaccBlockTileType = decltype(gemm_0.MakeCBlockTile());
        auto s_acc              = SaccBlockTileType{};

        // reduction function for softmax
        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        // infer Sacc, S, P, M, L, Oacc type
        using SBlockTileType = decltype(cast_tile<SMPLComputeDataType>(s_acc));

        using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
            SBlockTileType{}, Sequence<1>{}, f_max, SMPLComputeDataType{0}));

        using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());

        // init Oacc, M, L
        auto o_acc = OaccBlockTileType{};
        auto m     = MLBlockTileType{};
        auto l     = MLBlockTileType{};

        clear_tile(o_acc);
        set_tile(m, -NumericLimits<SMPLComputeDataType>::Infinity());
        clear_tile(l);

        const auto q_origin = q_dram_block_window_tmp.GetWindowOrigin();
        const auto [seqlen_k_start, seqlen_k_end] =
            mask.GetTileRangeAlongX(q_origin.At(Number<0>{}), Number<kM0>{}, Number<kN0>{});

        const auto num_total_loop = math::integer_divide_ceil(seqlen_k_end - seqlen_k_start, kN0);

        // check early exit if there is no work to do. also avoid fullfilling unwanted sentinel
        // values (-INF).
        if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
        {
            if(num_total_loop <= 0)
            {
                if constexpr(kStoreLSE)
                {
                    auto lse = make_static_distributed_tensor<LSEDataType>(m.GetTileDistribution());

                    set_tile(lse, -NumericLimits<SMPLComputeDataType>::Infinity());

                    store_tile(lse_dram_window_tmp, tile_elementwise_in(lse_element_func, lse));
                }

                // Note: here occ are all cleard, return it
                // Note: q loaded but no fence, ignore it.
                return o_acc;
            }
        }

        auto k_dram_block_window = make_tile_window(k_dram_block_window_tmp.GetBottomTensorView(),
                                                    k_dram_block_window_tmp.GetWindowLengths(),
                                                    {seqlen_k_start, 0});

        const auto bias_origin = bias_dram_block_window_tmp.GetWindowOrigin();
        auto bias_dram_window  = make_tile_window(
            bias_dram_block_window_tmp.GetBottomTensorView(),
            bias_dram_block_window_tmp.GetWindowLengths(),
            {bias_origin.At(Number<0>{}), seqlen_k_start}, // M/N
            Policy::template MakeBiasDramTileDistribution<Problem, decltype(gemm_0)>());

        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp.GetBottomTensorView(),
                             v_dram_block_window_tmp.GetWindowLengths(),
                             {0, seqlen_k_start}, // TODO: hdim split?
                             Policy::template MakeVDramTileDistribution<Problem>());

        // prefetch K tile
        index_t i_total_loops      = 0;
        constexpr index_t k0_loops = kK0BlockLength / kK0;
        constexpr index_t k1_loops = kN0 / kK1;

        static_assert(2 <= k0_loops);
        static_assert(1 <= k1_loops);
        do
        {
            // STAGE 1, QK gemm
            auto q_dram_window =
                make_tile_window(q_dram_block_window_tmp.GetBottomTensorView(),
                                 q_dram_block_window_tmp.GetWindowLengths(),
                                 q_dram_block_window_tmp.GetWindowOrigin(),
                                 Policy::template MakeQDramTileDistribution<Problem>());

            auto k_dram_window =
                make_tile_window(k_dram_block_window.GetBottomTensorView(),
                                 k_dram_block_window.GetWindowLengths(),
                                 k_dram_block_window.GetWindowOrigin(),
                                 Policy::template MakeKDramTileDistribution<Problem>());

            auto q_block_tile = load_tile(q_dram_window);
            auto k_block_tile = load_tile(k_dram_window);
            {
                move_tile_window(q_dram_window, {0, kK0});
                move_tile_window(k_dram_window, {0, kK0});

                clear_tile(s_acc); // Initialize C

                store_tile(q_lds_window, tile_elementwise_in(q_element_func, q_block_tile));
                q_block_tile = load_tile(q_dram_window);

                store_tile(k_lds_window, tile_elementwise_in(k_element_func, k_block_tile));
                k_block_tile = load_tile(k_dram_window);
            }

            if constexpr(kHasBias)
            {
                __builtin_amdgcn_sched_barrier(
                    0); // prevent from messing up the order of global loads
            }
            const auto bias_tile = load_tile(bias_dram_window); // load bias tile
            if constexpr(kHasBias)
            {
                __builtin_amdgcn_sched_barrier(
                    0); // prevent from messing up the order of global loads
            }

            if constexpr(k0_loops > 2)
            {
                static_for<0, k0_loops - 2, 1>{}([&](auto) {
                    block_sync_lds();
                    gemm_0(s_acc, q_lds_window, k_lds_window);
                    block_sync_lds();

                    move_tile_window(q_dram_window, {0, kK0});
                    move_tile_window(k_dram_window, {0, kK0});

                    store_tile(
                        q_lds_window,
                        tile_elementwise_in(q_element_func, q_block_tile)); // LDS write i + 1
                    q_block_tile = load_tile(q_dram_window);                // global read i + 2

                    store_tile(
                        k_lds_window,
                        tile_elementwise_in(k_element_func, k_block_tile)); // LDS write i + 1
                    k_block_tile = load_tile(k_dram_window);                // global read i + 2
                });
            }

            const auto v_prefetch = load_tile(v_dram_window); // prefetch load v tile
            {                                                 // tail
                block_sync_lds();
                gemm_0(s_acc, q_lds_window, k_lds_window);
                block_sync_lds();

                store_tile(q_lds_window, tile_elementwise_in(q_element_func, q_block_tile));
                store_tile(k_lds_window, tile_elementwise_in(k_element_func, k_block_tile));
                block_sync_lds();

                gemm_0(s_acc, q_lds_window, k_lds_window);
            }

            // STAGE 2, scale, add bias, mask, softmax
            if constexpr(kHasBias)
            {
                tile_elementwise_inout(
                    [&](auto& x, const auto& y) {
#if !CK_FMHA_FWD_FAST_EXP2
                        x = scale * x + type_convert<SaccDataType>(bias_element_func(y));
#else
                        x = scale * x + math::log2e_v<SaccDataType> *
                                            type_convert<SaccDataType>(bias_element_func(y));
#endif
                    },
                    s_acc,
                    bias_tile);
            }
            else
            {
#if !CK_FMHA_FWD_FAST_EXP2
                tile_elementwise_inout([&scale](auto& x) { x = x * scale; }, s_acc);
#endif
            }
            move_tile_window(bias_dram_window, {0, kN0});
            if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
            {
                const auto k_origin      = k_dram_block_window.GetWindowOrigin();
                bool need_perpixel_check = mask.IsEdgeTile(q_origin.At(Number<0>{}),
                                                           k_origin.At(Number<0>{}),
                                                           Number<kM0>{},
                                                           Number<kN0>{});
                if(need_perpixel_check)
                {
                    set_tile_if(
                        s_acc, -NumericLimits<SMPLComputeDataType>::Infinity(), [&](auto tile_idx) {
                            const auto row = q_origin.At(Number<0>{}) + tile_idx.At(Number<0>{});
                            const auto col = k_origin.At(Number<0>{}) + tile_idx.At(Number<1>{});
                            return mask.IsOutOfBound(row, col);
                        });
                }
            }

            const auto s = cast_tile<SMPLComputeDataType>(s_acc); // S{j}
            auto m_local = block_tile_reduce<SMPLComputeDataType>(
                s,
                Sequence<1>{},
                f_max,
                -NumericLimits<SMPLComputeDataType>::Infinity()); // m_local = rowmax(S{j})
            block_tile_reduce_sync(m_local, f_max, bool_constant<false>{});

            const auto m_old = m; // m{j-1}
            tile_elementwise_inout(
                [](auto& e0, auto e1, auto e2) { e0 = max(e1, e2); }, m, m_old, m_local); // m{j}

            auto p_compute = make_static_distributed_tensor<SMPLComputeDataType>(
                s.GetTileDistribution()); // Pcompute{j}

            static const auto get_validated_m = [](SMPLComputeDataType raw_m) {
                /// NOTICE: bias might be materialized mask including -inf values, need
                /// consideration
                if constexpr(kHasBias || FmhaMask::IsMasking)
                {
                    return raw_m == -NumericLimits<SMPLComputeDataType>::Infinity()
                               ? type_convert<SMPLComputeDataType>(0.f)
                               : raw_m;
                }
                else
                {
                    return raw_m;
                }
            };

            constexpr auto p_spans = decltype(p_compute)::GetDistributedSpans();
            sweep_tile_span(p_spans[Number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
#if CK_FMHA_FWD_FAST_EXP2
                auto row_max = scale * get_validated_m(m[i_idx]);
#endif
                sweep_tile_span(p_spans[Number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
#if CK_FMHA_FWD_FAST_EXP2
                    if constexpr(kHasBias)
                    {
                        p_compute(i_j_idx) = math::exp2(s[i_j_idx] - get_validated_m(m[i_idx]));
                    }
                    else
                    {
                        p_compute(i_j_idx) = math::exp2(scale * s[i_j_idx] - row_max);
                    }
#else
                    p_compute(i_j_idx)     = math::exp(s[i_j_idx] - get_validated_m(m[i_idx]));
#endif
                });
            });

            auto rowsum_p = block_tile_reduce<SMPLComputeDataType>(
                p_compute, Sequence<1>{}, f_sum, SMPLComputeDataType{0}); // rowsum(Pcompute{j})

            block_tile_reduce_sync(rowsum_p, f_sum, bool_constant<false>{});
            // l{j}, Oacc{j}
            constexpr auto o_spans = decltype(o_acc)::GetDistributedSpans();
            sweep_tile_span(o_spans[Number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
#if CK_FMHA_FWD_FAST_EXP2
                const auto tmp = [&]() {
                    if constexpr(kHasBias)
                    {
                        return math::exp2(m_old[i_idx] - get_validated_m(m[i_idx]));
                    }
                    else
                    {
                        auto row_max = scale * get_validated_m(m[i_idx]);
                        return math::exp2(scale * m_old[i_idx] - row_max);
                    }
                }();
#else
                const auto tmp       = math::exp(m_old[i_idx] - get_validated_m(m[i_idx]));
#endif
                l(i_idx) = tmp * l[i_idx] + rowsum_p[i_idx];
                sweep_tile_span(o_spans[Number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    // FIXME: this use different equation from FA v2 paper,
                    // but produce correc result.
                    // Is the equation wrong?
                    o_acc(i_j_idx) *= tmp;
                });
            });

            block_sync_lds();
            if constexpr(ck::is_same_v<VLayout, ck::tensor_layout::gemm::RowMajor>)
            {
                auto v_shuffle_tmp = make_static_distributed_tensor<VDataType>(
                    Policy::template MakeShuffledVRegBlockDescriptor<Problem>());
                shuffle_distributed_tensor(v_shuffle_tmp, v_prefetch);
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

            const auto p = cast_tile<PDataType>(p_compute);

            // STAGE 3, KV gemm
            if constexpr(k1_loops > 1)
            {
                static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
                    const auto v = load_tile(v_dram_window); // load next v
                    block_sync_lds();
                    gemm_1(o_acc,
                           get_slice_tile(
                               p, Sequence<0, i_k1 * kK1>{}, Sequence<kM0, (i_k1 + 1) * kK1>{}),
                           v_lds_window);
                    block_sync_lds();
                    if constexpr(ck::is_same_v<VLayout, ck::tensor_layout::gemm::RowMajor>)
                    {
                        auto v_shuffle_tmp = make_static_distributed_tensor<VDataType>(
                            Policy::template MakeShuffledVRegBlockDescriptor<Problem>());
                        shuffle_distributed_tensor(v_shuffle_tmp, v);
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
                });
            }
            // move K tile windows
            move_tile_window(k_dram_block_window, {kN0, 0});
            // tail
            {
                block_sync_lds();
                gemm_1(o_acc,
                       get_slice_tile(p, Sequence<0, (k1_loops - 1) * kK1>{}, Sequence<kM0, kN0>{}),
                       v_lds_window);
                block_sync_lds();
            }
        } while(++i_total_loops < num_total_loop);

        // store lse
        if constexpr(kStoreLSE)
        {
            auto lse = make_static_distributed_tensor<LSEDataType>(m.GetTileDistribution());

            constexpr auto lse_spans = decltype(lse)::GetDistributedSpans();
            sweep_tile_span(lse_spans[Number<0>{}], [&, m_ = m, l_ = l](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
#if CK_FMHA_FWD_FAST_EXP2
                if constexpr(kHasBias)
                {
                    lse(i_idx) = m_[i_idx] / C_LOG2E + math::log(l_[i_idx]);
                }
                else
                {
                    lse(i_idx) = m_[i_idx] * scale / C_LOG2E + math::log(l_[i_idx]);
                }
#else
                lse(i_idx) = m_[i_idx] + math::log(l_[i_idx]);
#endif
            });

            store_tile(lse_dram_window_tmp, tile_elementwise_in(lse_element_func, lse));
        }

        // finally, O
        constexpr auto o_spans = decltype(o_acc)::GetDistributedSpans();

        sweep_tile_span(o_spans[Number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            const auto tmp       = [&]() {
                if constexpr(FmhaMask::IsMasking)
                {
                    return l[i_idx] == 0.f ? 0.f : 1 / l[i_idx];
                }
                else
                    return 1 / l[i_idx];
            }();
            sweep_tile_span(o_spans[Number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                o_acc(i_j_idx) *= tmp;
            });
        });

        return o_acc;
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp>
    __host__ __device__ auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,       // M0*K0 tile
               const KDramBlockWindowTmp& k_dram_block_window_tmp,       // N0*K0 tile
               const VDramBlockWindowTmp& v_dram_block_window_tmp,       // N1*K1 tile
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
               LSEDramBlockWindowTmp& lse_dram_block_window_tmp,         // M0*1 tile
               FmhaMask mask,
               float scale,
               void* smem_ptr) const
    {
        return operator()(q_dram_block_window_tmp,
                          identity{},
                          k_dram_block_window_tmp,
                          identity{},
                          v_dram_block_window_tmp,
                          identity{},
                          bias_dram_block_window_tmp,
                          identity{},
                          lse_dram_block_window_tmp,
                          identity{},
                          mask,
                          scale,
                          smem_ptr);
    }
};

} // namespace block
} // namespace tile_program
} // namespace ck
