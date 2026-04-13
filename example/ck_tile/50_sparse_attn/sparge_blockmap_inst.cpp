// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Hand-written template instantiation for SpargeBlockMapKernel (fp16, D=128).

#include "sparge_blockmap_trek.hpp"
#include "ck_tile/ops/fmha/block/variants.hpp"

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

// ============================================================================
// Dispatch
// ============================================================================

float sparge_blockmap_fwd(sparge_blockmap_traits traits,
                          sparge_blockmap_args args,
                          const ck_tile::stream_config& s)
{
    if(traits.data_type == "fp16" && traits.hdim_q == 128)
    {
        using k_ = bmap_fp16_kernel;
        if(s.log_level_ > 0)
            std::cout << ", sparge_blockmap_fp16_d128" << std::flush;
        auto [kargs, grids]                    = sparge_blockmap_create_kargs_and_grids<k_>(args);
        const dim3 blocks                      = k_::BlockSize();
        constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
        return ck_tile::launch_kernel(
            s, ck_tile::make_kernel<kBlockPerCu>(k_{}, grids, blocks, 0, kargs));
    }

    if(s.log_level_ > 0)
        std::cerr << "sparge_blockmap_fwd: unsupported config (data_type=" << traits.data_type
                  << ", hdim_q=" << traits.hdim_q << ")" << std::endl;
    return -1.f;
}
