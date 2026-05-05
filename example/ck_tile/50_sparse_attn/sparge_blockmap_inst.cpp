// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Hand-written template instantiation for SpargeBlockMapKernel (fp16, D=128).

#include "sparge_blockmap_trek.hpp"
#include "ck_tile/ops/fmha/block/variants.hpp"

#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <iostream>

// ============================================================================
// Type configuration for block map kernel (reuses FmhaSparseFwdTypeConfig)
// ============================================================================

// fp16: D=128, kM0=64, kN0=128
using bmap_fp16_block_tile = ck_tile::sequence<64, 128, 128, 128, 128, 128>;
//                                              kM0 kN0  kK0  kN1  kK1  kQKHeaddim(D)

using bmap_fp16_shape =
    ck_tile::TileFmhaShape<bmap_fp16_block_tile,
                           ck_tile::sequence<4, 1, 1>,    // Gemm0BlockWarps
                           ck_tile::sequence<16, 16, 16>, // Gemm0WarpTile (unused by blockmap, but
                                                          // needed by shape)
                           ck_tile::sequence<4, 1, 1>,    // Gemm1BlockWarps
                           ck_tile::sequence<16, 16, 16>, // Gemm1WarpTile
                           true>;                         // VLayout row-major

using bmap_fp16_trait = ck_tile::TileFmhaTraits<true,  // kPadSeqLenQ
                                                true,  // kPadSeqLenK
                                                true,  // kPadHeadDimQ
                                                true,  // kPadHeadDimV
                                                false, // kHasLogitsSoftCap
                                                ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                                                false, // kStoreLSE
                                                false, // kHasDropout
                                                false, // kHasRandVal
                                                ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
                                                -1,     // kBlockPerCu
                                                false>; // kIsVRowMajorSkip

using bmap_fp16_variant = ck_tile::ComposedAttention<0, CK_TILE_FMHA_FWD_FAST_EXP2>;
using bmap_fp16_mask    = ck_tile::GenericAttentionMask<false>;

using bmap_fp16_problem = ck_tile::BlockFmhaPipelineProblem<ck_tile::half_t, // QDataType
                                                            ck_tile::half_t, // KDataType
                                                            ck_tile::half_t, // VDataType
                                                            float,           // SaccDataType
                                                            float,           // SMPLComputeDataType
                                                            ck_tile::half_t, // BiasDataType
                                                            uint8_t, // RandValOutputDataType
                                                            float,   // LSEDataType
                                                            ck_tile::half_t, // PDataType
                                                            float,           // OaccDataType
                                                            ck_tile::half_t, // ODataType
                                                            bmap_fp16_shape,
                                                            false, // kIsGroupMode
                                                            bmap_fp16_variant,
                                                            bmap_fp16_mask,
                                                            false, // kUseTrLoad
                                                            bmap_fp16_trait>;

using bmap_fp16_pipeline = ck_tile::SpargeBlockMapPipeline<bmap_fp16_problem>;
using bmap_fp16_kernel   = ck_tile::SpargeBlockMapKernel<bmap_fp16_pipeline>;

using kstats_fp16_pipeline = ck_tile::SpargeKStatsPipeline<bmap_fp16_problem>;
using kstats_fp16_kernel   = ck_tile::SpargeKStatsKernel<kstats_fp16_pipeline>;

// ============================================================================
// bf16: D=128, kM0=64, kN0=128
// ============================================================================

using bmap_bf16_block_tile = ck_tile::sequence<64, 128, 128, 128, 128, 128>;

using bmap_bf16_shape =
    ck_tile::TileFmhaShape<bmap_bf16_block_tile,
                           ck_tile::sequence<4, 1, 1>,
                           ck_tile::sequence<16, 16, 16>,
                           ck_tile::sequence<4, 1, 1>,
                           ck_tile::sequence<16, 16, 16>,
                           true>;

using bmap_bf16_trait = ck_tile::TileFmhaTraits<true,  // kPadSeqLenQ
                                                true,  // kPadSeqLenK
                                                true,  // kPadHeadDimQ
                                                true,  // kPadHeadDimV
                                                false, // kHasLogitsSoftCap
                                                ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                                                false, // kStoreLSE
                                                false, // kHasDropout
                                                false, // kHasRandVal
                                                ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
                                                -1,
                                                false>;

using bmap_bf16_variant = ck_tile::ComposedAttention<0, CK_TILE_FMHA_FWD_FAST_EXP2>;
using bmap_bf16_mask    = ck_tile::GenericAttentionMask<false>;

using bmap_bf16_problem = ck_tile::BlockFmhaPipelineProblem<ck_tile::bf16_t,  // QDataType
                                                            ck_tile::bf16_t,  // KDataType
                                                            ck_tile::bf16_t,  // VDataType
                                                            float,            // SaccDataType
                                                            float,            // SMPLComputeDataType
                                                            ck_tile::bf16_t,  // BiasDataType
                                                            uint8_t,          // RandValOutputDataType
                                                            float,            // LSEDataType
                                                            ck_tile::bf16_t,  // PDataType
                                                            float,            // OaccDataType
                                                            ck_tile::bf16_t,  // ODataType
                                                            bmap_bf16_shape,
                                                            false, // kIsGroupMode
                                                            bmap_bf16_variant,
                                                            bmap_bf16_mask,
                                                            false, // kUseTrLoad
                                                            bmap_bf16_trait>;

using bmap_bf16_pipeline = ck_tile::SpargeBlockMapPipeline<bmap_bf16_problem>;
using bmap_bf16_kernel   = ck_tile::SpargeBlockMapKernel<bmap_bf16_pipeline>;

using kstats_bf16_pipeline = ck_tile::SpargeKStatsPipeline<bmap_bf16_problem>;
using kstats_bf16_kernel   = ck_tile::SpargeKStatsKernel<kstats_bf16_pipeline>;

// ============================================================================
// Internal K-stat workspace (R20): process-lifetime lazy hipMalloc, sized
// to the largest (batch, nhead_k, N_k, D) seen so far. Caller API unchanged.
// ============================================================================

namespace {

struct KStatsWorkspace
{
    void* pooled_k_dev = nullptr; // [batch, nhead_k, N_k, D] fp32
    void* sim_k_dev    = nullptr; // [batch, nhead_k, N_k] uint8
    size_t pooled_k_bytes = 0;
    size_t sim_k_bytes    = 0;

    void ensure(int batch, int nhead_k, int N_k, int D)
    {
        const size_t need_p = static_cast<size_t>(batch) * nhead_k * N_k * D * sizeof(float);
        const size_t need_s = static_cast<size_t>(batch) * nhead_k * N_k * sizeof(uint8_t);
        if(need_p > pooled_k_bytes)
        {
            if(pooled_k_dev != nullptr) (void)hipFree(pooled_k_dev);
            (void)hipMalloc(&pooled_k_dev, need_p);
            pooled_k_bytes = need_p;
        }
        if(need_s > sim_k_bytes)
        {
            if(sim_k_dev != nullptr) (void)hipFree(sim_k_dev);
            (void)hipMalloc(&sim_k_dev, need_s);
            sim_k_bytes = need_s;
        }
    }
};

KStatsWorkspace& g_kstats_ws()
{
    static KStatsWorkspace ws;
    return ws;
}

template <typename KStatsKernel, typename BlockMapKernel>
void launch_kstats_then_blockmap(sparge_blockmap_args args, const ck_tile::stream_config& s)
{
    const int N_k = ck_tile::integer_divide_ceil(args.seqlen_k, BlockMapKernel::kN0);
    const int D   = BlockMapKernel::D;
    auto& ws      = g_kstats_ws();
    ws.ensure(args.batch, args.nhead_k, N_k, D);

    // Stage 1: K stats
    {
        auto [kargs, grids] =
            sparge_kstats_create_kargs_and_grids<KStatsKernel>(args, ws.pooled_k_dev, ws.sim_k_dev);
        const dim3 blocks                      = KStatsKernel::BlockSize();
        constexpr ck_tile::index_t kBlockPerCu = KStatsKernel::kBlockPerCu;
        ck_tile::make_kernel<kBlockPerCu>(KStatsKernel{}, grids, blocks, 0, kargs)(
            ck_tile::stream_config{s.stream_id_});
    }
    // Stage 2: block_map (reads ws)
    {
        auto [kargs, grids] = sparge_blockmap_create_kargs_and_grids<BlockMapKernel>(
            args, ws.pooled_k_dev, ws.sim_k_dev);
        const dim3 blocks                      = BlockMapKernel::BlockSize();
        constexpr ck_tile::index_t kBlockPerCu = BlockMapKernel::kBlockPerCu;
        ck_tile::make_kernel<kBlockPerCu>(BlockMapKernel{}, grids, blocks, 0, kargs)(
            ck_tile::stream_config{s.stream_id_});
    }
}

} // namespace

// ============================================================================
// Dispatch
// ============================================================================

float sparge_blockmap_fwd(sparge_blockmap_traits traits,
                          sparge_blockmap_args args,
                          const ck_tile::stream_config& s)
{
    if(traits.data_type == "fp16" && traits.hdim_q == 128)
    {
        if(s.log_level_ > 0)
            std::cout << ", sparge_blockmap_fp16_d128" << std::flush;
        return ck_tile::launch_kernel(s, [=](const ck_tile::stream_config& s_) {
            launch_kstats_then_blockmap<kstats_fp16_kernel, bmap_fp16_kernel>(args, s_);
        });
    }

    if(traits.data_type == "bf16" && traits.hdim_q == 128)
    {
        if(s.log_level_ > 0)
            std::cout << ", sparge_blockmap_bf16_d128" << std::flush;
        return ck_tile::launch_kernel(s, [=](const ck_tile::stream_config& s_) {
            launch_kstats_then_blockmap<kstats_bf16_kernel, bmap_bf16_kernel>(args, s_);
        });
    }

    if(s.log_level_ > 0)
        std::cerr << "sparge_blockmap_fwd: unsupported config (data_type=" << traits.data_type
                  << ", hdim_q=" << traits.hdim_q << ")" << std::endl;
    return -1.f;
}

// ============================================================================
// Oneshot version: launches kernel without timing wrapper
// ============================================================================

void sparge_blockmap_fwd_oneshot(sparge_blockmap_traits traits,
                                 sparge_blockmap_args args,
                                 const ck_tile::stream_config& s)
{
    if(traits.data_type == "fp16" && traits.hdim_q == 128)
    {
        launch_kstats_then_blockmap<kstats_fp16_kernel, bmap_fp16_kernel>(args, s);
        return;
    }

    if(traits.data_type == "bf16" && traits.hdim_q == 128)
    {
        launch_kstats_then_blockmap<kstats_bf16_kernel, bmap_bf16_kernel>(args, s);
        return;
    }

    std::cerr << "sparge_blockmap_fwd_oneshot: unsupported config (data_type=" << traits.data_type
              << ", hdim_q=" << traits.hdim_q << ")" << std::endl;
}

// ============================================================================
// Combined functions: blockmap + attention timed together via launch_kernel
// ============================================================================

float sparge_jenga_fwd(sparge_blockmap_traits bmap_t, sparge_blockmap_args bmap_a,
                       fmha_jenga_fwd_traits attn_t, fmha_jenga_fwd_args attn_a,
                       const ck_tile::stream_config& s)
{
    if(s.log_level_ > 0)
        std::cout << ", sparge_blockmap_" << bmap_t.data_type << "_d" << bmap_t.hdim_q
                  << ", fmha_jenga_fwd_" << attn_t.data_type << "_d" << attn_t.hdim_q
                  << std::flush;

    return ck_tile::launch_kernel(
        s,
        [=](const ck_tile::stream_config& s_) {
            sparge_blockmap_fwd_oneshot(bmap_t, bmap_a, s_);
        },
        [=](const ck_tile::stream_config& s_) {
            fmha_jenga_fwd_oneshot(attn_t, attn_a, s_);
        });
}

float sparge_vsa_fwd_combined(sparge_blockmap_traits bmap_t, sparge_blockmap_args bmap_a,
                              fmha_vsa_fwd_traits attn_t, fmha_vsa_fwd_args attn_a,
                              const ck_tile::stream_config& s)
{
    if(s.log_level_ > 0)
        std::cout << ", sparge_blockmap_" << bmap_t.data_type << "_d" << bmap_t.hdim_q
                  << ", fmha_vsa_fwd_" << attn_t.data_type << "_d" << attn_t.hdim_q
                  << std::flush;

    return ck_tile::launch_kernel(
        s,
        [=](const ck_tile::stream_config& s_) {
            sparge_blockmap_fwd_oneshot(bmap_t, bmap_a, s_);
        },
        [=](const ck_tile::stream_config& s_) {
            fmha_vsa_fwd_oneshot(attn_t, attn_a, s_);
        });
}
