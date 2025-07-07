#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_impl.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma.hpp"

namespace ck_tile {

template <bool kTransLdA = false, bool kTransLdB = false, bool kTransC = false>
using WarpGemmWmma_f32_16x16x16_f16_f16_gfx11 =
    WarpGemmImpl<WarpGemmAtrributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x16_f16_f16_gfx11,
                                       kTransLdA,
                                       kTransLdB,
                                       kTransC>>;

template <bool kTransLdA = false, bool kTransLdB = false, bool kTransC = false>
using WarpGemmWmma_f32_16x16x16_bf16_bf16_gfx11 =
    WarpGemmImpl<WarpGemmAtrributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x16_bf16_bf16_gfx11,
                                       kTransLdA,
                                       kTransLdB,
                                       kTransC>>;

template <bool kTransLdA = false, bool kTransLdB = false, bool kTransC = false>
using WarpGemmWmma_i32_16x16x16_i8_i8_gfx11 =
    WarpGemmImpl<WarpGemmAtrributeWmma<WarpGemmAttributeWmmaImpl_i32_16x16x16_i8_i8_gfx11,
                                       kTransLdA,
                                       kTransLdB,
                                       kTransC>>;

template <bool kTransLdA = false, bool kTransLdB = false, bool kTransC = false>
using WarpGemmWmma_f32_16x16x16_f16_f16_gfx12 =
    WarpGemmImpl<WarpGemmAtrributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x16_f16_f16_gfx12,
                                       kTransLdA,
                                       kTransLdB,
                                       kTransC>>;

template <bool kTransLdA = false, bool kTransLdB = false, bool kTransC = false>
using WarpGemmWmma_f32_16x16x16_bf16_bf16_gfx12 =
    WarpGemmImpl<WarpGemmAtrributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x16_bf16_bf16_gfx12,
                                       kTransLdA,
                                       kTransLdB,
                                       kTransC>>;

template <bool kTransLdA = false, bool kTransLdB = false, bool kTransC = false>
using WarpGemmWmma_f32_16x16x16_f8_f8_gfx12 =
    WarpGemmImpl<WarpGemmAtrributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x16_f8_f8_gfx12,
                                       kTransLdA,
                                       kTransLdB,
                                       kTransC>>;

template <bool kTransLdA = false, bool kTransLdB = false, bool kTransC = false>
using WarpGemmWmma_f32_16x16x16_bf8_bf8_gfx12 =
    WarpGemmImpl<WarpGemmAtrributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x16_bf8_bf8_gfx12,
                                       kTransLdA,
                                       kTransLdB,
                                       kTransC>>;

template <bool kTransLdA = false, bool kTransLdB = false, bool kTransC = false>
using WarpGemmWmma_f32_16x16x16_f8_bf8_gfx12 =
    WarpGemmImpl<WarpGemmAtrributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x16_f8_bf8_gfx12,
                                       kTransLdA,
                                       kTransLdB,
                                       kTransC>>;

template <bool kTransLdA = false, bool kTransLdB = false, bool kTransC = false>
using WarpGemmWmma_f32_16x16x16_bf8_f8_gfx12 =
    WarpGemmImpl<WarpGemmAtrributeWmma<WarpGemmAttributeWmmaImpl_f32_16x16x16_bf8_f8_gfx12,
                                       kTransLdA,
                                       kTransLdB,
                                       kTransC>>;

} // namespace ck_tile
