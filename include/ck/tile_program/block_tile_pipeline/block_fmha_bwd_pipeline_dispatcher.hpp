// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_bwd_pipeline_v9.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_bwd_pipeline_v10.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_bwd_pipeline_v13.hpp"

namespace ck {
namespace tile_program {
namespace block {

template <typename LoadStrategy_, typename Problem_>
struct BlockFmhaBwdPipelineDispatcher
{
    using LoadStrategy = ck::remove_cvref_t<LoadStrategy_>;
    using Problem      = ck::remove_cvref_t<Problem_>;

    static constexpr bool kQLoadOnce =
        LoadStrategy::At(Number<0>{}); // if q load whole block length (qkhdim) to LDS at once
    static constexpr bool kQTLoadOnce =
        LoadStrategy::At(Number<1>{}); // if q^t load whole block length (qkhdim) to LDS at once
    static constexpr bool kKLoadOnce =
        LoadStrategy::At(Number<2>{}); // if k load whole block length (qkhdim) to LDS at once
    static constexpr bool kKTLoadOnce =
        LoadStrategy::At(Number<3>{}); // if k^t load whole block length (qkhdim) to LDS at once
    static constexpr bool kVLoadOnce =
        LoadStrategy::At(Number<4>{}); // if v load whole block length (vhdim) to Vgprs at once
    static constexpr bool kOGradLoadOnce =
        LoadStrategy::At(Number<5>{}); // if do load whole block length (vhdim) to LDS at once
    static constexpr bool kOGradTLoadOnce =
        LoadStrategy::At(Number<6>{}); // if do^t load whole block length (vhdim) to LDS at once

    template <bool QLoadOnce,
              bool QTLoadOnce,
              bool KLoadOnce,
              bool KTLoadOnce,
              bool VLoadOnce,
              bool OGradLoadOnce,
              bool OGradTLoadOnce>
    struct BlockPipelineDispatcher;

    // clang-format off
    // #########################################| QLoadOnce| QTLoadOnce| KLoadOnce| KTLoadOnce| VLoadOnce| OGradLoadOnce| OGradTLoadOnce|
    // template<> struct BlockPipelineDispatcher<      true,       true,      true,       true,      true,          true,           true> { using Type = BlockFmhaBwdPipelineV1<Problem>; };
    // template<> struct BlockPipelineDispatcher<      true,       true,      true,       true,      true,          true,          false> { using Type = BlockFmhaBwdPipelineV2<Problem>; };
    // template<> struct BlockPipelineDispatcher<      true,      false,      true,       true,      true,          true,           true> { using Type = BlockFmhaBwdPipelineV3<Problem>; };
    // template<> struct BlockPipelineDispatcher<      true,      false,      true,       true,      true,          true,          false> { using Type = BlockFmhaBwdPipelineV4<Problem>; };
    // template<> struct BlockPipelineDispatcher<      true,       true,      true,       true,      true,         false,          false> { using Type = BlockFmhaBwdPipelineV5<Problem>; };
    // template<> struct BlockPipelineDispatcher<     false,      false,      true,       true,      true,          true,           true> { using Type = BlockFmhaBwdPipelineV6<Problem>; };
    // template<> struct BlockPipelineDispatcher<      true,      false,      true,       true,      true,         false,          false> { using Type = BlockFmhaBwdPipelineV7<Problem>; };
    // template<> struct BlockPipelineDispatcher<     false,      false,      true,       true,      true,          true,          false> { using Type = BlockFmhaBwdPipelineV8<Problem>; };
       template<> struct BlockPipelineDispatcher<     false,      false,      true,       true,      true,         false,          false> { using Type = BlockFmhaBwdPipelineV9<Problem>; };
       template<> struct BlockPipelineDispatcher<      true,      false,      true,      false,      true,          true,          false> { using Type = BlockFmhaBwdPipelineV10<Problem>; };
    // template<> struct BlockPipelineDispatcher<      true,      false,      true,      false,      true,         false,          false> { using Type = BlockFmhaBwdPipelineV11<Problem>; };
    // template<> struct BlockPipelineDispatcher<     false,      false,      true,      false,      true,          true,          false> { using Type = BlockFmhaBwdPipelineV12<Problem>; };
       template<> struct BlockPipelineDispatcher<     false,      false,      true,      false,      true,         false,          false> { using Type = BlockFmhaBwdPipelineV13<Problem>; };
    // template<> struct BlockPipelineDispatcher<     false,      false,     false,      false,      true,         false,          false> { using Type = BlockFmhaBwdPipelineV14<Problem>; };
    // template<> struct BlockPipelineDispatcher<     false,      false,     false,      false,     false,         false,          false> { using Type = BlockFmhaBwdPipelineV15<Problem>; };
    // clang-format on

    using BlockPipeline = typename BlockPipelineDispatcher<kQLoadOnce,
                                                           kQTLoadOnce,
                                                           kKLoadOnce,
                                                           kKTLoadOnce,
                                                           kVLoadOnce,
                                                           kOGradLoadOnce,
                                                           kOGradTLoadOnce>::Type;
};

} // namespace block
} // namespace tile_program
} // namespace ck
