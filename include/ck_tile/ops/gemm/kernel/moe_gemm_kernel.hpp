// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/utility/literals.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include "ck_tile/host.hpp"

namespace ck_tile {

struct MoeGemmHostArgs : public ck_tile::GemmHostArgs
{
    ck_tile::index_t NumTokens;
    ck_tile::index_t TopK;
    CK_TILE_HOST MoeGemmHostArgs() noexcept = default;
    CK_TILE_HOST MoeGemmHostArgs(const void* a_ptr_,
                                 const void* b_ptr_,
                                 void* c_ptr_,
                                 ck_tile::index_t NumTokens_,
                                 ck_tile::index_t TopK_,
                                 ck_tile::index_t M_,
                                 ck_tile::index_t N_,
                                 ck_tile::index_t K_,
                                 ck_tile::index_t stride_A_,
                                 ck_tile::index_t stride_B_,
                                 ck_tile::index_t stride_C_)
        : GemmHostArgs(a_ptr_, b_ptr_, c_ptr_, KBatch, M_, N_, K_, stride_A_, stride_B_, stride_C_)
              NumTokens(NumTokens_),
          TopK(TopK_),

    {
    }

    private:
    static constexpr index_t KBatch = 1;
};

template <typename TilePartitioner_, typename GemmPipeline_, typename EpiloguePipeline_>
struct MoeGemmKernel : public GemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>
{
    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline     = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;
    using ALayout          = remove_cvref_t<typename GemmPipeline::ALayout>;
    using BLayout          = remove_cvref_t<typename GemmPipeline::BLayout>;
    using CLayout          = remove_cvref_t<typename GemmPipeline::CLayout>;

    using ADataType = remove_cvref_t<typename GemmPipeline::ADataType>;
    using BDataType = remove_cvref_t<typename GemmPipeline::BDataType>;
    using CDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    using OffsetTile1DPartitioner = OffsettedTile1DPartitioner<TilePartitioner>;
    using Base                    = GemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>;
    using GemmKernelArgs          = typename Base::GemmKernelArgs;

    static constexpr index_t KernelBlockSize = GemmPipeline::BlockSize;

    struct MoeGemmKernelArgs : public GemmKernelArgs
    {
        const ck_tile::index_t* p_sorted_token_ids;
        const ck_tile::index_t* p_sorted_expert_ids;
        const ck_tile::index_t* p_max_token_id;
        ck_tile::index_t NumTokens;
        ck_tile::index_t TopK;

        CK_TILE_HOST MoeGemmKernelArgs() noexcept = default;
        CK_TILE_HOST MoeGemmKernelArgs(const ck_tile::index_t* p_sorted_token_ids_,
                                       const ck_tile::index_t* p_sorted_expert_ids_,
                                       const ck_tile::index_t* p_max_token_id_,
                                       const void* a_ptr_,
                                       const void* b_ptr_,
                                       void* c_ptr_,
                                       ck_tile::index_t NumTokens_,
                                       ck_tile::index_t TopK_,
                                       ck_tile::index_t M_,
                                       ck_tile::index_t N_,
                                       ck_tile::index_t K_,
                                       ck_tile::index_t stride_A_,
                                       ck_tile::index_t stride_B_,
                                       ck_tile::index_t stride_C_)
            : GemmKernelArgs{a_ptr_,
                             b_ptr_,
                             c_ptr_,
                             KBatch,
                             M_,
                             N_,
                             K_,
                             stride_A_,
                             stride_B_,
                             stride_C_},
              NumTokens(NumTokens_),
              TopK(TopK_),
              p_sorted_token_ids(p_sorted_token_ids_),
              p_sorted_expert_ids(p_sorted_expert_ids_),
              p_max_token_id(p_max_token_id_)
        {
        }

        CK_TILE_HOST static constexpr MoeGemmKernelArgs
        MakeKernelArgs(const MoeGemmHostArgs& hostArgs,
                       const ck_tile::index_t* p_sorted_token_ids_,
                       const ck_tile::index_t p_sorted_expert_ids_,
                       const ck_tile::index_t p_max_token_id_)
        {
            return MoeGemmKernelArgs{p_sorted_token_ids_,
                                     p_sorted_expert_ids_,
                                     p_max_token_id_,
                                     hostArgs.a_ptr,
                                     hostArgs.b_ptr,
                                     hostArgs.c_ptr,
                                     hostArgs.NumTokens,
                                     hostArgs.TopK,
                                     hostArgs.M,
                                     hostArgs.N,
                                     hostArgs.K,
                                     hostArgs.stride_A,
                                     hostArgs.stride_B,
                                     hostArgs.stride_C,
                                     hostArgs.k_batch};
        }
    };

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        using P_ = GemmPipeline;

        return concat('_', "moe_gemm", gemm_prec_str<ADataType, BDataType>,
                      concat('x', P_::kMPerBlock, P_::kNPerBlock, P_::kKPerBlock),
                      concat('x', P_::GetVectorSizeA(), P_::GetVectorSizeB(), P_::GetVectorSizeC()),
                      concat('x', P_::kPadM, P_::kPadN, P_::kPadK));
        // clang-format on
    }

    __host__ static constexpr auto BlockSize() -> dim3 { return dim3(KernelBlockSize); }

    __host__ static constexpr auto GridSize(index_t M, index_t N, index_t KBatch)
    {
        // TODO: remove assertion
        assert(KBatch == 1) return Base::GridSize(M, N, KBatch);
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetSmemSize() -> index_t
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void Run(const MoeGemmKernelArgs& kargs) const
    {
        const auto [iM, iN] = OffsetTile1DPartitioner::GetOffsetedTileIndex(
            kargs.block_start, kargs.group_karg.M, kargs.group_karg.N);

        const index_t i_m = __builtin_amdgcn_readfirstlane(iM * TilePartitioner::MPerBlock);
        const index_t i_n = __builtin_amdgcn_readfirstlane(iN * TilePartitioner::NPerBlock);

        const typename Base::SplitKBatchOffset splitk_batch_offset(kargs.group_karg, blockIdx.z);

        const ADataType* a_ptr = static_cast<const ADataType*>(kargs.group_karg.a_ptr);
        const BDataType* b_ptr = static_cast<const BDataType*>(kargs.group_karg.b_ptr);
        CDataType* c_ptr       = static_cast<CDataType*>(kargs.group_karg.c_ptr);

        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        this->RunGemm(
            a_ptr, b_ptr, c_ptr, smem_ptr, kargs.group_karg, splitk_batch_offset, i_m, i_n);
    }

    CK_TILE_DEVICE void operator()(const MoeGemmKernelArgs gemm_desc) const
    {

        // TODO: the branch without swizzle
        const index_t max_token_id = __builtin_amdgcn_readfirstlane(gemm_desc.p_max_token_id[0]);
        const index_t block_id     = ck_tile::get_block_1d_id();

        // TODO: check the block id caculation
        const auto [expert_blk_id, _] =
            OffsetTile1DPartitioner::GetOffsetedTileIndex(0, gemm_desc.M, gemm_desc.N);

        if(expert_blk_id * MPerBlock >= max_token_id)
            return;

        const index_t NBlocks        = gemm_desc.N / TilePartitioner::NPerBlock;
        const index_t expert_id      = gemm_desc.p_sorted_expert_ids[iM];
        const index_t prefix_blk_m   = gemm_desc.p_max_token_id[1 + expert_id];
        const index_t blk_cnt_of_eid = gemm_desc.p_max_token_id[2 + expert_id];

        const index_t block_start = prefix_blk_m * NBlocks;

        const index_t ecnt           = blk_cnt_of_eid - prefix_blk_m;
        const index_t expert_swizzle = ecnt > 0 ? ecnt : 1;
        // index_t block_end = block_start + blk_cnt_of_eid * NBlocks;

        const index_t block_id_start_in_expert = block_id - block_start;
        const index_t im = __builtin_amdgcn_readfirstlane(prefix_blk_m + block_id_start_in_expert /
                                                                             8 % expert_swizzle);
        const index_t in = __builtin_amdgcn_readfirstlane(
            block_id_start_in_expert % 8 + block_id_start_in_expert / (8 * expert_swizzle) * 8)
            // const auto gemm_desc_ptr = reinterpret_cast<const GemmTransKernelArg*>(
            //     cast_pointer_to_generic_address_space(gemm_descs_const));

            Run(gemm_desc_ptr[group_id]);
    }
};

} // namespace ck_tile
