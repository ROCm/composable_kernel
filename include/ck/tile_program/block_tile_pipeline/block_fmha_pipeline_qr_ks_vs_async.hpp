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
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qr_ks_vs_async_default_policy.hpp"
#include "ck/tile_program/block_tile/block_reduce.hpp"
#include "ck/tile_program/tile/shuffle_distributed_tensor.hpp"

namespace ck {
namespace tile_program {
namespace block {

// a variation of qr/ks/vs, where we use async copy to load k (potentially v in the future)
template <typename Problem, typename Policy = BlockFmhaPipelineQRKSVSAsyncDefaultPolicy>
struct BlockFmhaPipelineQRKSVSAsync
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
    static constexpr bool kQLoadOnce = true; // if q_tile load whole block length (hdim) at once
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

#if CK_FMHA_FWD_FAST_EXP2
    static constexpr auto R_LOG2E = 1.0 / math::log2e_v<SaccDataType>;
#endif

    __host__ __device__ static constexpr ck::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
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
               const KElementFunction& /*k_element_func*/,
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

        constexpr auto LdsSeq = Policy::template GetLdsBufferSequence<Problem>();

        // K tile in LDS
        auto k_lds_ptr   = reinterpret_cast<KDataType*>(smem_ptr);
        auto k_lds_store = generate_tuple(
            [&](auto i_buf) {
                return make_tile_window(
                    make_tensor_view<AddressSpaceEnum::Lds>(
                        k_lds_ptr, Policy::template MakeKLdsStoreBlockDescriptor<Problem>(i_buf)),
                    Policy::template MakeKLdsStoreBlockDescriptor<Problem>(i_buf).GetLengths(),
                    {0, 0, 0});
            },
            Number<Policy::NumPrefetchK>{});

#if K_LDS_LOAD_USE_OFFSET_TRANSFORM
        auto k_lds_load = generate_tuple(
            [&](auto i_buf) {
                return make_tile_window(
                    make_tensor_view<AddressSpaceEnum::Lds>(
                        k_lds_ptr, Policy::template MakeKLdsLoadBlockDescriptor<Problem>(i_buf)),
                    Policy::template MakeKLdsLoadBlockDescriptor<Problem>(i_buf).GetLengths(),
                    {0, 0});
            },
            Number<Policy::NumPrefetchK>{});
#else
        auto k_lds_Load_view = make_tensor_view<AddressSpaceEnum::Lds>(
            k_lds_ptr, Policy::template MakeKLdsLoadBlockDescriptor<Problem>());

        auto k_lds_load =
            make_tile_window(k_lds_Load_view,
                             Policy::template MakeKLdsLoadBlockDescriptor<Problem>().GetLengths(),
                             {0, 0});
#endif

        // V tile in LDS
        auto v_lds = make_tensor_view<AddressSpaceEnum::Lds>(
            reinterpret_cast<VDataType*>(smem_ptr),
            Policy::template MakeVLdsBlockDescriptor<Problem>());
        auto v_lds_window = make_tile_window(
            v_lds, Policy::template MakeVLdsBlockDescriptor<Problem>().GetLengths(), {0, 0});

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetKVBlockGemm<Problem>();

        auto q_dram_window = make_tile_window(
            q_dram_block_window_tmp.GetBottomTensorView(),
            q_dram_block_window_tmp.GetWindowLengths(),
            q_dram_block_window_tmp.GetWindowOrigin(),
            Policy::template MakeQDramTileDistribution<Problem, decltype(gemm_0)>());

        // TODO: we use async Copy for K, which is inline asm
        // a side effect is we have to use inline asm for q as well
        auto q = load_tile_raw(q_dram_window);
        __builtin_amdgcn_sched_barrier(0);

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

        __builtin_amdgcn_sched_barrier(0);
        const auto q_origin = q_dram_window.GetWindowOrigin();
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

            if constexpr(FmhaMask::IsMasking)
                __builtin_amdgcn_sched_barrier(0); // make sure sched_barrier(0) for this check
        }

        auto k_dram_block_window = make_tile_window(k_dram_block_window_tmp.GetBottomTensorView(),
                                                    k_dram_block_window_tmp.GetWindowLengths(),
                                                    {seqlen_k_start, 0});

        auto k_dram_window = make_tile_window(
            k_dram_block_window.GetBottomTensorView(),
            k_dram_block_window.GetWindowLengths(),
            k_dram_block_window.GetWindowOrigin(),
            Policy::template MakeKDramTileDistribution<Problem>()); // K DRAM tile window for
                                                                    // load
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
        async_load_tile_raw(k_lds_store(LdsSeq.At(Number<0>{})), k_dram_window);
        move_tile_window(k_dram_window, {0, kK0});
        __builtin_amdgcn_sched_barrier(0);

        buffer_load_fence(k_dram_window.GetNumAccess());

        auto q_tile = tile_elementwise_in(q_element_func, q);
        __builtin_amdgcn_sched_barrier(0);

        index_t i_total_loops      = 0;
        constexpr index_t k0_loops = kK0BlockLength / kK0;
        constexpr index_t k1_loops = kN0 / kK1;

        static_assert(1 <= k0_loops);
        static_assert(1 <= k1_loops);
        // main loop
        do
        {
            // STAGE 1, QK gemm
            clear_tile(s_acc); // Initialize C
            if constexpr(k0_loops > 1)
            {
                static_for<0, k0_loops - 1, 1>{}([&](auto i_k0) {
                    async_load_tile_raw(k_lds_store(Number<LdsSeq.At(Number<i_k0 + 1>{})>{}),
                                        k_dram_window);
                    if constexpr(i_k0 < k0_loops - 1)
                        move_tile_window(k_dram_window, {0, kK0});

                    async_load_fence(k_dram_window.GetNumAccess());
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
                    gemm_0(s_acc,
                           get_slice_tile(q_tile,
                                          Sequence<0, i_k0 * kK0>{},
                                          Sequence<kM0, (i_k0 + 1) * kK0>{}),
#if K_LDS_LOAD_USE_OFFSET_TRANSFORM
                           k_lds_load[Number<LdsSeq.At(Number<i_k0>{})>{}]);

#else
                           get_slice_tile(k_lds_load,
                                          Sequence<(LdsSeq.At(Number<i_k0>{})) * kN0, 0>{},
                                          Sequence<(LdsSeq.At(Number<i_k0>{}) + 1) * kN0, kK0>{}));
#endif
                });
            }

            // TODO: this to fix a bug when loop smaller than 2,
            // the following fence/barrier will be scheduled inside 1st loop
            if constexpr(k0_loops <= 2)
                __builtin_amdgcn_sched_barrier(0);

            async_load_fence();
            __builtin_amdgcn_s_barrier();

            const auto bias_tile = load_tile(bias_dram_window); // load bias tile
            auto v_buf           = load_tile(v_dram_window);
            __builtin_amdgcn_sched_barrier(0);
            { // tail
                gemm_0(s_acc,
                       get_slice_tile(q_tile,
                                      Sequence<0, (k0_loops - 1) * kK0>{},
                                      Sequence<kM0, k0_loops * kK0>{}),
#if K_LDS_LOAD_USE_OFFSET_TRANSFORM
                       k_lds_load[Number<LdsSeq.At(Number<k0_loops - 1>{})>{}]);

#else
                       get_slice_tile(
                           k_lds_load,
                           Sequence<(LdsSeq.At(Number<k0_loops - 1>{})) * kN0, 0>{},
                           Sequence<(LdsSeq.At(Number<k0_loops - 1>{}) + 1) * kN0, kK0>{}));
#endif
            }
            __builtin_amdgcn_sched_barrier(1);

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

            __builtin_amdgcn_sched_barrier(0x7F);
            // store & prefetch next v, after the max reduction
            if constexpr(ck::is_same_v<VLayout, ck::tensor_layout::gemm::RowMajor>)
            {
                auto v_shuffle_tmp = make_static_distributed_tensor<VDataType>(
                    Policy::template MakeShuffledVRegBlockDescriptor<Problem>());
                shuffle_distributed_tensor(v_shuffle_tmp, v_buf);

                auto v_lds_window_tmp =
                    get_slice_tile(v_lds_window,
                                   Sequence<(LdsSeq.At(Number<k0_loops>{})) * kN1, 0>{},
                                   Sequence<(LdsSeq.At(Number<k0_loops>{}) + 1) * kN1, kK1>{});

                store_tile(
                    v_lds_window_tmp,
                    tile_elementwise_in(v_element_func, v_shuffle_tmp)); // store the prefetch
            }
            else
            {
                store_tile(v_lds_window,
                           tile_elementwise_in(v_element_func, v_buf)); // store the prefetch
            }

            if constexpr(k1_loops > 1)
            {
                move_tile_window(
                    v_dram_window,
                    {0, kK1}); // will have scratch if move this right after load_tile(v_dram)...
                v_buf = load_tile(v_dram_window); // load next v_buf
            }
            __builtin_amdgcn_sched_barrier(0);

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
                const auto tmp = math::exp(m_old[i_idx] - get_validated_m(m[i_idx]));
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

            const auto p = cast_tile<PDataType>(p_compute);

            // STAGE 3, KV gemm
            if constexpr(k1_loops > 1)
            {
                static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
                    if constexpr(i_k1 != 0 && i_k1 < k1_loops - 1)
                    {
                        v_buf = load_tile(v_dram_window); // load next v_buf
                    }
                    block_sync_lds();
                    gemm_1(o_acc,
                           get_slice_tile(
                               p, Sequence<0, i_k1 * kK1>{}, Sequence<kM0, (i_k1 + 1) * kK1>{}),
                           get_slice_tile(
                               v_lds_window,
                               Sequence<(LdsSeq.At(Number<k0_loops + i_k1>{})) * kN1, 0>{},
                               Sequence<(LdsSeq.At(Number<k0_loops + i_k1>{}) + 1) * kN1, kK1>{}));

                    if constexpr(ck::is_same_v<VLayout, ck::tensor_layout::gemm::RowMajor>)
                    {
                        auto v_shuffle_tmp = make_static_distributed_tensor<VDataType>(
                            Policy::template MakeShuffledVRegBlockDescriptor<Problem>());
                        shuffle_distributed_tensor(v_shuffle_tmp, v_buf);
                        auto v_lds_window_tmp = get_slice_tile(
                            v_lds_window,
                            Sequence<(LdsSeq.At(Number<k0_loops + i_k1 + 1>{})) * kN1, 0>{},
                            Sequence<(LdsSeq.At(Number<k0_loops + i_k1 + 1>{}) + 1) * kN1, kK1>{});
                        store_tile(v_lds_window_tmp,
                                   tile_elementwise_in(v_element_func,
                                                       v_shuffle_tmp)); // store the prefetch
                    }
                    else
                    {
                        store_tile(v_lds_window,
                                   tile_elementwise_in(v_element_func, v_buf)); // store next v_buf
                    }
                    if constexpr(i_k1 < k1_loops - 1)
                        move_tile_window(v_dram_window, {0, kK1});
                });
            }
            i_total_loops++;
            if(i_total_loops < num_total_loop)
            {
                // move K tile windows
                move_tile_window(k_dram_block_window, {kN0, 0});
                k_dram_window =
                    make_tile_window(k_dram_block_window.GetBottomTensorView(),
                                     k_dram_block_window.GetWindowLengths(),
                                     k_dram_block_window.GetWindowOrigin(),
                                     Policy::template MakeKDramTileDistribution<Problem>());

                if constexpr(k1_loops >= 2 &&
                             LdsSeq.At(Number<0>{}) == LdsSeq.At(Number<k0_loops + k1_loops - 2>{}))
                    __builtin_amdgcn_s_barrier();
                async_load_tile_raw(k_lds_store(LdsSeq.At(Number<0>{})), k_dram_window);
                move_tile_window(k_dram_window, {0, kK0});
            }
            // tail
            {
                block_sync_lds();
                gemm_1(
                    o_acc,
                    get_slice_tile(p, Sequence<0, (k1_loops - 1) * kK1>{}, Sequence<kM0, kN0>{}),
                    get_slice_tile(
                        v_lds_window,
                        Sequence<(LdsSeq.At(Number<k0_loops + k1_loops - 1>{})) * kN1, 0>{},
                        Sequence<(LdsSeq.At(Number<k0_loops + k1_loops - 1>{}) + 1) * kN1, kK1>{}));
            }
        } while(i_total_loops < num_total_loop);

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
                    lse(i_idx) = m_[i_idx] * R_LOG2E + math::log(l_[i_idx]);
                }
                else
                {
                    lse(i_idx) = m_[i_idx] * scale * R_LOG2E + math::log(l_[i_idx]);
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
