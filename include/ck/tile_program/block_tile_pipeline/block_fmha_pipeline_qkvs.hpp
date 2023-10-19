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
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qkvs_default_policy.hpp"
#include "ck/tile_program/block_tile/block_reduce.hpp"

namespace ck {
namespace tile_program {
namespace block {

// This pipeline is qkv all located in LDS
template <typename Problem, typename Policy = BlockFmhaPipelineQKVSDefaultPolicy>
struct BlockFmhaPipelineQKVS
{
    using QDataType           = remove_cvref_t<typename Problem::QDataType>;
    using KDataType           = remove_cvref_t<typename Problem::KDataType>;
    using VDataType           = remove_cvref_t<typename Problem::VDataType>;
    using SaccDataType        = remove_cvref_t<typename Problem::SaccDataType>;
    using SMPLComputeDataType = remove_cvref_t<typename Problem::SMPLComputeDataType>;
    using PDataType           = remove_cvref_t<typename Problem::PDataType>;
    using OaccDataType        = remove_cvref_t<typename Problem::OaccDataType>;
    using ODataType           = remove_cvref_t<typename Problem::ODataType>;

    using BlockFmhaShape = remove_cvref_t<typename Problem::BlockFmhaShape>;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kM0 = BlockFmhaShape::kM0;
    static constexpr index_t kN0 = BlockFmhaShape::kN0;
    static constexpr index_t kK0 = BlockFmhaShape::kK0;
    static constexpr index_t kN1 = BlockFmhaShape::kN1;
    static constexpr index_t kK1 = BlockFmhaShape::kK1;

    __host__ __device__ static constexpr ck::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename QElementFunction,
              typename KElementFunction,
              typename VElementFunction>
    __host__ __device__ auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp, // M0*K0 tile
               const QElementFunction& q_element_func,
               const KDramBlockWindowTmp& k_dram_block_window_tmp, // N0*K0 tile
               const KElementFunction& k_element_func,
               const VDramBlockWindowTmp& v_dram_block_window_tmp, // N1*K1 tile
               const VElementFunction& v_element_func,
               index_t num_total_loop,
               index_t num_sub_loop_qk,
               void* smem_ptr) const
    {
        static_assert(
            is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kK0 == QDramBlockWindowTmp{}.GetWindowLengths()[Number<1>{}] &&
                          kN0 == KDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kK0 == KDramBlockWindowTmp{}.GetWindowLengths()[Number<1>{}] &&
                          kN1 == VDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kK1 == VDramBlockWindowTmp{}.GetWindowLengths()[Number<1>{}],
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
        auto v_lds_window =
            make_tile_window(v_lds, make_tuple(Number<kN1>{}, Number<kK1>{}), {0, 0});

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetKVBlockGemm<Problem>();

        auto s_acc = decltype(gemm_0(q_lds_window, k_lds_window)){};

        // reduction function for softmax
        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        // infer Sacc, S, P, M, L, Oacc type
        using SBlockTileType =
            decltype(tile_elementwise_in(type_convert<SMPLComputeDataType, SaccDataType>, s_acc));

        using PBlockTileType =
            decltype(tile_elementwise_in(type_convert<PDataType, SaccDataType>, s_acc));

        using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
            SBlockTileType{}, Sequence<1>{}, f_max, SMPLComputeDataType{0}));

        using OaccBlockTileType = decltype(gemm_1(
            get_slice_tile(PBlockTileType{}, Sequence<0, 0>{}, Sequence<kM0, kK1>{}),
            v_lds_window));

        // init Oacc, M, L
        auto o_acc = OaccBlockTileType{};
        auto m     = MLBlockTileType{};
        auto l     = MLBlockTileType{};

        tile_elementwise_inout([](auto& e) { e = 0; }, o_acc);
        tile_elementwise_inout([](auto& e) { e = NumericLimits<SMPLComputeDataType>::Lowest(); },
                               m);
        tile_elementwise_inout([](auto& e) { e = 0; }, l);

        auto k_dram_block_window = k_dram_block_window_tmp;
        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp.GetBottomTensorView(),
                             v_dram_block_window_tmp.GetWindowLengths(),
                             v_dram_block_window_tmp.GetWindowOrigin(),
                             Policy::template MakeVDramTileDistribution<Problem>());

        index_t i_total_loops = 0;
        do
        {
            // STAGE 1, QK gemm
            auto q_dram_window = make_tile_window(
                q_dram_block_window_tmp.GetBottomTensorView(),
                q_dram_block_window_tmp.GetWindowLengths(),
                q_dram_block_window_tmp.GetWindowOrigin(),
                Policy::template MakeQDramTileDistribution<Problem>()); // Q DRAM tile window for
                                                                        // load

            auto k_dram_window = make_tile_window(
                k_dram_block_window.GetBottomTensorView(),
                k_dram_block_window.GetWindowLengths(),
                k_dram_block_window.GetWindowOrigin(),
                Policy::template MakeKDramTileDistribution<Problem>()); // K DRAM tile window for
                                                                        // load

            auto q_block_tile = load_tile(q_dram_window); // prefetch, global read 0
            auto k_block_tile = load_tile(k_dram_window);
            {
                move_tile_window(q_dram_window, {0, kK0}); // move to 1
                move_tile_window(k_dram_window, {0, kK0});

                tile_elementwise_inout([](auto& c) { c = 0; }, s_acc); // Initialize C

                store_tile(q_lds_window,
                           tile_elementwise_in(q_element_func, q_block_tile)); // LDS write 0
                q_block_tile = load_tile(q_dram_window);                       // global read 1
                store_tile(k_lds_window,
                           tile_elementwise_in(k_element_func, k_block_tile)); // LDS write 0
                k_block_tile = load_tile(k_dram_window);                       // global read 1
            }

            index_t i_k0_loops = num_sub_loop_qk - 2;
            do
            {
                block_sync_lds();
                gemm_0(s_acc, q_lds_window, k_lds_window); // GEMM i
                block_sync_lds();

                move_tile_window(q_dram_window, {0, kK0}); // move to i + 2
                move_tile_window(k_dram_window, {0, kK0});

                store_tile(q_lds_window,
                           tile_elementwise_in(q_element_func, q_block_tile)); // LDS write i + 1
                q_block_tile = load_tile(q_dram_window);                       // global read i + 2
                store_tile(k_lds_window,
                           tile_elementwise_in(k_element_func, k_block_tile)); // LDS write i + 1
                k_block_tile = load_tile(k_dram_window);                       // global read i + 2

                i_k0_loops--;
            } while(i_k0_loops > 0);

            const auto v_prefetch = load_tile(v_dram_window); // prefetch load v tile

            { // tail
                block_sync_lds();
                gemm_0(s_acc, q_lds_window, k_lds_window); // GEMM num_loop - 2
                block_sync_lds();

                store_tile(
                    q_lds_window,
                    tile_elementwise_in(q_element_func, q_block_tile)); // LDS write num_loop - 1
                store_tile(k_lds_window, tile_elementwise_in(k_element_func, k_block_tile));
                block_sync_lds();

                gemm_0(s_acc, q_lds_window, k_lds_window); // GEMM num_loop - 1
            }

            // STAGE 2, scale softmax
            const auto s =
                tile_elementwise_in(type_convert<SMPLComputeDataType, SaccDataType>, s_acc); // S{j}
            auto m_local = block_tile_reduce<SMPLComputeDataType>(
                s,
                Sequence<1>{},
                f_max,
                NumericLimits<SMPLComputeDataType>::Lowest()); // m_local = rowmax(S{j})
            block_tile_reduce_sync(m_local, f_max);

            const auto m_old = m; // m{j-1}
            tile_elementwise_inout(
                [](auto& e0, auto e1, auto e2) { e0 = max(e1, e2); }, m, m_old, m_local); // m{j}

            auto p_compute = make_static_distributed_tensor<SMPLComputeDataType>(
                s.GetTileDistribution()); // Pcompute{j}

            constexpr auto p_spans = decltype(p_compute)::GetDistributedSpans();
            sweep_tile_span(p_spans[Number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                sweep_tile_span(p_spans[Number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    p_compute(i_j_idx)     = math::exp(s[i_j_idx] - m[i_idx]);
                });
            });

            auto rowsum_p = block_tile_reduce<SMPLComputeDataType>(
                p_compute, Sequence<1>{}, f_sum, SMPLComputeDataType{0}); // rowsum(Pcompute{j})

            block_tile_reduce_sync(rowsum_p, f_sum);
            // l{j}, Oacc{j}
            sweep_tile_span(p_spans[Number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                const auto tmp       = math::exp(m_old[i_idx] - m[i_idx]);
                l(i_idx)             = tmp * l[i_idx] + rowsum_p[i_idx];
                sweep_tile_span(p_spans[Number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    // FIXME: this use different equation from FA v2 paper,
                    // but produce correc result.
                    // Is the equation wrong?
                    o_acc(i_j_idx) *= tmp;
                });
            });

            block_sync_lds();
            store_tile(v_lds_window,
                       tile_elementwise_in(v_element_func, v_prefetch)); // store the prefetch
            move_tile_window(v_dram_window, {0, kK1});

            const auto p =
                tile_elementwise_in(type_convert<PDataType, SMPLComputeDataType>, p_compute);

            // STAGE 3, KV gemm
            constexpr index_t k1_loops = kN0 / kK1;
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
                    store_tile(v_lds_window,
                               tile_elementwise_in(v_element_func, v)); // store next v
                    move_tile_window(v_dram_window, {0, kK1});
                });
            }
            // tail
            {
                block_sync_lds();
                gemm_1(o_acc,
                       get_slice_tile(p, Sequence<0, (k1_loops - 1) * kK1>{}, Sequence<kM0, kN0>{}),
                       v_lds_window);
                block_sync_lds();
            }
            // move K tile windows
            move_tile_window(k_dram_block_window, {kN0, 0});

            i_total_loops++;
        } while(i_total_loops < num_total_loop);

        // finally, O
        constexpr auto o_spans = decltype(o_acc)::GetDistributedSpans();

        sweep_tile_span(o_spans[Number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            const auto tmp       = 1 / l[i_idx];
            sweep_tile_span(o_spans[Number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                o_acc(i_j_idx) *= tmp;
            });
        });

        return o_acc;
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp>
    __host__ __device__ auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp, // M0*K0 tile
               const KDramBlockWindowTmp& k_dram_block_window_tmp, // N0*K0 tile
               const VDramBlockWindowTmp& v_dram_block_window_tmp, // N1*K1 tile
               index_t num_total_loop,
               index_t num_sub_loop_qk,
               void* smem_ptr) const
    {
        return operator()(
            q_dram_block_window_tmp,
            [](const QDataType& x) { return x; },
            k_dram_block_window_tmp,
            [](const KDataType& x) { return x; },
            v_dram_block_window_tmp,
            [](const VDataType& x) { return x; },
            num_total_loop,
            num_sub_loop_qk,
            smem_ptr);
    }
};

} // namespace block
} // namespace tile_program
} // namespace ck
