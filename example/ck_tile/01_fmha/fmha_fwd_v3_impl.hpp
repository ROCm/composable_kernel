// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <utility>

#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
#include "ck_tile/ops/fmha/block/block_masking.hpp"
#include "ck_tile/ops/fmha/kernel/fmha_fwd_v3_kernel.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_problem.hpp"
#include "ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp"
#include "ck_tile/ops/fmha/pipeline/tile_fmha_traits.hpp"

#include "fmha_fwd.hpp"
#include "fmha_fwd_v3.hpp"
#include "mask.hpp"

#define INST_FMHA_FWD_V3_DISPATCH(kernel_traits)                                             \
    template <>                                                                              \
    float fmha_fwd_<kernel_traits, ck_tile::gfx950_t>(const ck_tile::stream_config& config,  \
                                                      fmha_fwd_args args)                    \
    {                                                                                        \
        using kernel        = typename ck_tile::get_fmha_fwd_v3_kernel<kernel_traits>::type; \
        auto [kargs, grids] = fmha_fwd_v3_create_kargs_and_grids<kernel>(args);              \
        const dim3 blocks   = kernel::BlockSize();                                           \
        constexpr ck_tile::index_t kBlockPerCu = kernel::kBlockPerCu;                        \
        return ck_tile::launch_kernel(config,                                                \
                                      ck_tile::make_kernel<kBlockPerCu, ck_tile::gfx950_t>(  \
                                          kernel{}, grids, blocks, 0, kargs));               \
    }

namespace ck_tile {

template <typename DataType, bool kIsGroupMode, bool kIsMasking>
using fmha_fwd_v3_kernel_traits =
    fmha_fwd_traits_<128,
                     DataType,
                     kIsGroupMode,
                     256,
                     32,
                     128,
                     128,
                     32,
                     128,
                     true,
                     ck_tile::BlockFmhaPipelineEnum::QRKSVS_ASYNC_TRLOAD_V3,
                     false,
                     ck_tile::GenericAttentionMask<kIsMasking, /*IsLocal=*/false>,
                     ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                     false,
                     false,
                     false,
                     true,
                     true,
                     false,
                     false,
                     true,
                     false>;

template <typename KernelTraits>
struct get_fmha_fwd_v3_kernel
{
    using fmha_dtype                   = KernelTraits::DataType;
    static constexpr bool kIsGroupMode = KernelTraits::kIsGroupMode;

    //                                    M0   N0  K0   N1   K1
    using fmha_block_tile      = sequence<KernelTraits::kM0,
                                          KernelTraits::kN0,
                                          KernelTraits::kK0,
                                          KernelTraits::kN1,
                                          KernelTraits::kK1,
                                          KernelTraits::kK0BlockLength>;
    using fmha_warp_gemm_shape = sequence<32, 32, 16>;
    using fmha_block_warps     = sequence<8, 1, 1>;

    using fmha_shape = TileFmhaShape<fmha_block_tile,
                                     fmha_block_warps,
                                     fmha_warp_gemm_shape,
                                     fmha_block_warps,
                                     fmha_warp_gemm_shape,
                                     KernelTraits::kIsVLayoutRowMajor>;

    using fmha_traits = ck_tile::TileFmhaTraits<KernelTraits::kPadSK,
                                                KernelTraits::kPadSK,
                                                KernelTraits::kPadD,
                                                KernelTraits::kPadDv,
                                                false,
                                                ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                                                false,
                                                false,
                                                false,
                                                false,
                                                -1,
                                                false>;

    using fmha_variant = ck_tile::ComposedAttention<false * ck_tile::LOGITS_SOFT_CAP>;

    using fmha_mask = KernelTraits::FmhaMask;

    using fmha_pipeline_problem =
        BlockFmhaPipelineProblem<typename FmhaFwdTypeConfig<fmha_dtype>::QDataType,
                                 typename FmhaFwdTypeConfig<fmha_dtype>::KDataType,
                                 typename FmhaFwdTypeConfig<fmha_dtype>::VDataType,
                                 typename FmhaFwdTypeConfig<fmha_dtype>::SaccDataType,
                                 typename FmhaFwdTypeConfig<fmha_dtype>::SMPLComputeDataType,
                                 typename FmhaFwdTypeConfig<fmha_dtype>::BiasDataType,
                                 typename FmhaFwdTypeConfig<fmha_dtype>::RandValOutputDataType,
                                 typename FmhaFwdTypeConfig<fmha_dtype>::LSEDataType,
                                 typename FmhaFwdTypeConfig<fmha_dtype>::PDataType,
                                 typename FmhaFwdTypeConfig<fmha_dtype>::OaccDataType,
                                 typename FmhaFwdTypeConfig<fmha_dtype>::ODataType,
                                 fmha_shape,
                                 kIsGroupMode,
                                 fmha_variant,
                                 fmha_mask,
                                 true,
                                 fmha_traits>;

    using fmha_pipeline = BlockFmhaFwdV3Pipeline<fmha_pipeline_problem>;

    using epilogue = Default2DEpilogue<
        Default2DEpilogueProblem<typename FmhaFwdTypeConfig<fmha_dtype>::OaccDataType,
                                 typename FmhaFwdTypeConfig<fmha_dtype>::ODataType,
                                 true, // kPadM
                                 true, // kPadM
                                 true  // UseRawStore
                                 >>;

    using type = FmhaFwdV3Kernel<fmha_pipeline, epilogue>;
};

} // namespace ck_tile
