// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_tdm_default_policy.hpp"

namespace ck_tile {

/// Policy of GemmPipelineAgBgCrCompTDMV3. Derived from, never edits, the shared TDM default
/// policy (wave-specialized form, as used by comp_tdm_v2), so that v3-only knobs do not change
/// any other pipeline.
struct GemmPipelineAgBgCrCompTDMV3Policy : public GemmPipelineAgBgCrCompTDMDefaultPolicy<true>
{
    using DefaultBase = GemmPipelineAgBgCrCompTDMDefaultPolicy<true>;

    /// LDS stage p is filled by wave 2p (A) and 2p+1 (B) when true; by waves 0 (A) and 1 (B)
    /// for both stages when false.
    static constexpr bool kAlternateIssuers = true;

    /// true: one split workgroup barrier per K tile (workgroup_barrier_arrive/wait). Per K
    /// sub-tile, B is read whole and stays in registers and A is read one warp-GEMM row at a time,
    /// right before that row's WMMAs; the arrive follows the read of the last row of the last
    /// sub-tile, so that row's WMMAs overlap the barrier. One TDM fill per LDS stage is in flight.
    /// Not used with data cache prefetch (DataCachePrefetchA/B), which keeps the joint form.
    /// false: two joint barriers per K tile, with the refill issued after the stage's WMMAs (up
    /// to two fills in flight).
    static constexpr bool kUseSplitBarrier = true;

    /// Split barrier only. true: every wave waits for TENSORcnt 0 before it arrives (a no-op on
    /// the waves that issued no fill). false: only the waves that issued the fill being
    /// published wait, inside their wave-uniform issuer branch.
    static constexpr bool kTensorWaitAllWaves = true;

    /// Where the next K tile's TDM fill is issued (split barrier only). Both keep one fill per
    /// LDS stage in flight.
    /// TopOfNextTile: every wave waits at the top of the K tile; then the issuers of the LDS
    /// stage every wave finished reading before its previous arrive refill it.
    /// AfterArrive: the issuers of the stage just read wait right after their own arrive, refill
    /// that stage, then run their overlapped WMMAs; every other wave waits at the top of the next
    /// K tile. The fill starts earlier, and the issuers give up their overlap window.
    enum class RefillPoint
    {
        TopOfNextTile,
        AfterArrive
    };
    static constexpr RefillPoint kRefillPoint = RefillPoint::TopOfNextTile;

    /// Workgroups per CU the register budget assumes (the launch's kBlockPerCu).
    static constexpr index_t kBlockPerCu = 1;

    /// Per-wave VGPR budget: the 1024-entry gfx1250 file divided by the waves sharing a SIMD
    /// (4 SIMDs per CU). The shared picker uses get_max_vgpr_count() = 1024 whatever the wave
    /// count, which is 4x too much for 16 waves.
    template <typename Problem>
    CK_TILE_DEVICE static constexpr index_t GetVgprBudget()
    {
        constexpr index_t waves_per_simd =
            integer_divide_ceil(Problem::kBlockSize / get_warp_size(), 4) * kBlockPerCu;
        return min(get_max_vgpr_count(), get_max_vgpr_count() / waves_per_simd);
    }

    /// Number of K sub-tiles per block K-tile (1, 2 or 4), as in the default policy but
    /// against GetVgprBudget().
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetPipelineSubTileNum()
    {
        constexpr index_t KPerBlock        = Problem::BlockGemmShape::kK;
        constexpr index_t KPerTile         = Problem::BlockGemmShape::WarpTile::at(DefaultBase::I2);
        constexpr index_t max_sub_tile_num = KPerBlock / KPerTile;

        constexpr auto estimated_vgpr    = DefaultBase::template GetEstimatedVgprCount<Problem>();
        constexpr index_t acc_vgpr_num   = estimated_vgpr.at(number<0>{});
        constexpr index_t input_vgpr_num = estimated_vgpr.at(number<1>{});

        constexpr index_t vgpr_budget  = GetVgprBudget<Problem>();
        constexpr index_t sub_tile_num = ((input_vgpr_num + acc_vgpr_num) <= vgpr_budget)       ? 1
                                         : ((input_vgpr_num / 2 + acc_vgpr_num) <= vgpr_budget) ? 2
                                                                                                : 4;

        return number<min(sub_tile_num, max_sub_tile_num)>{};
    }

    /// Estimated per-wave VGPRs of the v3 main loop: the accumulators plus one A and one B
    /// register tile of a single K sub-tile. v3 reads each K sub-tile (k-atom) from LDS right
    /// before its WMMAs and keeps no second register tile, so unlike GetEstimatedVgprCount there
    /// is no double-buffer factor. Addresses and descriptors are not counted.
    template <typename Problem>
    CK_TILE_DEVICE static constexpr index_t GetEstimatedFootprintVgpr()
    {
        using ADataType = remove_cvref_t<typename Problem::ADataType>;
        using BDataType = remove_cvref_t<typename Problem::BDataType>;
        using CDataType = remove_cvref_t<typename Problem::CDataType>;

        constexpr index_t MPerBlock    = Problem::BlockGemmShape::kM;
        constexpr index_t NPerBlock    = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock    = Problem::BlockGemmShape::kK;
        constexpr index_t MWarps       = Problem::BlockGemmShape::BlockWarps::at(DefaultBase::I0);
        constexpr index_t NWarps       = Problem::BlockGemmShape::BlockWarps::at(DefaultBase::I1);
        constexpr index_t BytesPerVGPR = 4;
        constexpr index_t KPerSubTile  = KPerBlock / GetPipelineSubTileNum<Problem>().value;

        constexpr index_t acc_vgpr =
            sizeof(CDataType) * MPerBlock * NPerBlock / Problem::kBlockSize / BytesPerVGPR;
        constexpr index_t a_vgpr = sizeof(ADataType) * MPerBlock * KPerSubTile /
                                   numeric_traits<ADataType>::PackedSize / MWarps /
                                   get_warp_size() / BytesPerVGPR;
        constexpr index_t b_vgpr = sizeof(BDataType) * NPerBlock * KPerSubTile /
                                   numeric_traits<BDataType>::PackedSize / NWarps /
                                   get_warp_size() / BytesPerVGPR;
        return acc_vgpr + a_vgpr + b_vgpr;
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetBlockGemm()
    {
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;

        // same as GemmPipelineAgBgCrCompTDMDefaultPolicy::GetBlockGemm, but with the v3 sub-tile
        // count (the base version calls its own GetPipelineSubTileNum)
        constexpr auto pipeline_tune_params = GetPipelineSubTileNum<Problem>();
        constexpr index_t sub_tile_num      = pipeline_tune_params.value;
        constexpr index_t kScaleBlockSize   = detail::scale_block_size_or_default<Problem>::value;

#if defined(__gfx125__)
        // Compute WGAttrNumAccess for a single operand (A or B).
        //
        // For normal types:
        //   vec_size       = instruction K-pack (kAKPack / kBKPack)
        //   total_elements = K_warp_tile / 2  (2 lanes share K)
        //
        // For packed types (f4/f6) with transpose load:
        //   vec_size       = PackedSize * elements_per_vgpr  (one VGPR worth of packed data)
        //   total_elements = instruction K-pack              (kAKPack / kBKPack)
        constexpr auto compute_num_access = []<typename DataType>(bool is_load_tr,
                                                                  index_t instr_kpack,
                                                                  index_t k_warp_tile) constexpr {
            constexpr index_t packed_size   = numeric_traits<DataType>::PackedSize;
            constexpr index_t bits_per_elem = sizeof(DataType) * 8 / packed_size;
            // in gfx1250 always use double vgpr for fp4 and fp8 in tr load
            constexpr index_t elems_per_dvgpr = 64 / bits_per_elem;

            const bool is_packed =
                is_load_tr && ((packed_size > 1) || ((bits_per_elem == 8) && (instr_kpack > 8)));
            const auto vec_size       = is_packed ? elems_per_dvgpr : instr_kpack;
            const auto total_elements = is_packed ? instr_kpack : (k_warp_tile / 2);
            const auto ratio          = total_elements / vec_size;

            // Map the ratio to WGAttrNumAccessEnum;
            // is_packed selects between Packed* and non-Packed variants.
            switch(ratio)
            {
            case 1:
                return is_packed ? WGAttrNumAccessEnum::PackedSingle : WGAttrNumAccessEnum::Single;
            case 2:
                return is_packed ? WGAttrNumAccessEnum::PackedDouble : WGAttrNumAccessEnum::Double;
            case 4: return is_packed ? WGAttrNumAccessEnum::PackedQuad : WGAttrNumAccessEnum::Quad;
            case 8: return is_packed ? WGAttrNumAccessEnum::PackedOcta : WGAttrNumAccessEnum::Octa;
            default: return WGAttrNumAccessEnum::Invalid;
            }
        };

        // Probe the default warp gemm to get instruction K-pack sizes.
        // For scale16, probe the scale16 type directly; otherwise use the standard dispatcher.
        using WarpGemmProbe      = WarpGemmDispatcher<typename Problem::ADataType,
                                                      typename Problem::BDataType,
                                                      typename Problem::CDataType,
                                                      WarpTile::at(DefaultBase::I0),
                                                      WarpTile::at(DefaultBase::I1),
                                                      WarpTile::at(DefaultBase::I2),
                                                      Problem::TransposeC,
                                                      false,
                                                      false,
                                                      WGAttrNumAccessEnum::Default,
                                                      WGAttrNumAccessEnum::Default,
                                                      (kScaleBlockSize == 16)>;
        constexpr index_t k_warp = WarpTile::at(DefaultBase::I2);

        constexpr auto a_wg_attr_num_access =
            compute_num_access.template operator()<typename Problem::ADataType>(
                DefaultBase::template is_a_load_tr<Problem>, WarpGemmProbe::kAKPack, k_warp);

        constexpr auto b_wg_attr_num_access =
            compute_num_access.template operator()<typename Problem::BDataType>(
                DefaultBase::template is_b_load_tr<Problem>, WarpGemmProbe::kBKPack, k_warp);
#else
        constexpr auto a_wg_attr_num_access = WGAttrNumAccessEnum::Default;
        constexpr auto b_wg_attr_num_access = WGAttrNumAccessEnum::Default;
#endif
        using WarpGemm = WarpGemmDispatcher<typename Problem::ADataType,
                                            typename Problem::BDataType,
                                            typename Problem::CDataType, // AccDataType
                                            WarpTile::at(DefaultBase::I0),
                                            WarpTile::at(DefaultBase::I1),
                                            WarpTile::at(DefaultBase::I2),
                                            Problem::TransposeC,
                                            false,
                                            false,
                                            a_wg_attr_num_access,
                                            b_wg_attr_num_access,
                                            (kScaleBlockSize == 16)>;

        using BlockGemmPolicy = BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::ADataType,
                                                                    typename Problem::BDataType,
                                                                    typename Problem::CDataType,
                                                                    BlockWarps,
                                                                    WarpGemm,
                                                                    sub_tile_num>;

        return BlockGemmARegBRegCRegV1<Problem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
