// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_pingpong_policy.hpp"

namespace ck_tile {

namespace detail {

// [KV buffer 0] ... [KV buffer kNumKVBuffers - 1]. One tile of the shared K/V
// tensor per buffer serves as K (QK gemm) and as V (PV gemm, transposed reads). Q sits on
// the last buffer: it is read before the main loop, and that buffer's first write follows
// the loop-head barrier. The pipeline keeps S in registers and merges its warps in a K/V
// buffer.
template <typename Problem, typename QPadding, typename KVPadding, index_t kNumKVBuffers>
struct QrTdmKVSharedLdsArenaLayout
{
    using Shape = typename Problem::BlockFmhaShape;

    static constexpr index_t kArenaAlignment  = 256;
    static constexpr index_t kRegionAlignment = 256;

    static constexpr index_t kQBytes =
        make_qr_tdm_pingpong_lds_descriptor<typename Problem::QDataType,
                                            Shape::kM0,
                                            Shape::kSubQKHeaddim,
                                            QPadding>()
            .get_element_space_size() *
        sizeof(typename Problem::QDataType);
    static constexpr index_t kKVBytes =
        make_qr_tdm_pingpong_lds_descriptor<typename Problem::KDataType,
                                            Shape::kN0,
                                            Shape::kSubQKHeaddim,
                                            KVPadding>()
            .get_element_space_size() *
        sizeof(typename Problem::KDataType);

    static_assert(kNumKVBuffers >= 2);
    // KV buffer i at i * kKVStride
    static constexpr index_t kKVStride = integer_least_multiple(kKVBytes, kRegionAlignment);
    static constexpr index_t kQOffset  = (kNumKVBuffers - 1) * kKVStride;
    static constexpr index_t kArenaBytes =
        integer_least_multiple(kNumKVBuffers * kKVStride, kArenaAlignment);

    static_assert(kQBytes <= kKVBytes && kArenaBytes <= 128 * 1024);
};

} // namespace detail

// qr_tdm_kvshared: the qr_tdm_pingpong policy for K and V read from one tensor. Both
// gemms read the same LDS tile, so K and V share one descriptor and one row padding,
// K's 16 B per row (V's 32 B costs d128 one workgroup per WGP, 5-7% slower at short
// seqlen_k, and d256 1%). Two K/V buffers keep one tile in flight under the gemms; a third
// costs more in resident workgroups than it adds in loads in flight (d256 [1024, 200]:
// 28% slower).
struct BlockFmhaPipelineQRKSVSTdmKVSharedPolicy : BlockFmhaPipelineQRKSVSTdmPingPongPolicy
{
    static constexpr index_t kNumKVBuffers = 2;

    template <typename Problem>
    using LdsPaddingConfigQ = typename detail::QrTdmPingPongPadding<Problem>::Q;

    template <typename Problem>
    using LdsPaddingConfigKV = typename detail::QrTdmPingPongPadding<Problem>::K;

    template <typename Problem>
    using LdsPaddingConfigK = LdsPaddingConfigKV<Problem>;

    template <typename Problem>
    using LdsPaddingConfigV = LdsPaddingConfigKV<Problem>;

    template <typename Problem>
    using LdsArenaLayout = detail::QrTdmKVSharedLdsArenaLayout<Problem,
                                                               LdsPaddingConfigQ<Problem>,
                                                               LdsPaddingConfigKV<Problem>,
                                                               kNumKVBuffers>;

    // The whole K/V tile: the QK gemm walks its kK0 column slices, the PV gemm its kK1
    // row slices.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKVLdsBlockDescriptor()
    {
        static_assert(Problem::BlockFmhaShape::kSubQKHeaddim == Problem::BlockFmhaShape::kN1 &&
                          std::is_same_v<typename Problem::KDataType, typename Problem::VDataType>,
                      "shared K/V: one head dim and one data type");
        return detail::make_qr_tdm_pingpong_lds_descriptor<typename Problem::KDataType,
                                                           Problem::BlockFmhaShape::kN0,
                                                           Problem::BlockFmhaShape::kSubQKHeaddim,
                                                           LdsPaddingConfigKV<Problem>>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        return MakeKVLdsBlockDescriptor<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor()
    {
        return MakeKVLdsBlockDescriptor<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return LdsArenaLayout<Problem>::kArenaBytes;
    }
};

} // namespace ck_tile
