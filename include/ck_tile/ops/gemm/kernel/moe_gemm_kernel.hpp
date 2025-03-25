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
    const ck_tile::index_t* p_sorted_token_ids;
    const ck_tile::index_t* p_sorted_expert_ids;
    const ck_tile::index_t* p_max_token_id;

    // TODO: add kbatch for splitk
    CK_TILE_HOST MoeGemmHostArgs() noexcept = default;
    CK_TILE_HOST MoeGemmHostArgs(const ck_tile::index_t* p_sorted_token_ids_,
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
        : GemmHostArgs(a_ptr_, b_ptr_, c_ptr_, 1, M_, N_, K_, stride_A_, stride_B_, stride_C_),
          NumTokens(NumTokens_),
          TopK(TopK_),
          p_sorted_token_ids(p_sorted_token_ids_),
          p_sorted_expert_ids(p_sorted_expert_ids_),
          p_max_token_id(p_max_token_id_)
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
                                       ck_tile::index_t stride_C_,
                                       ck_tile::index_t KBatch)
            : GemmKernelArgs{a_ptr_,
                             b_ptr_,
                             c_ptr_,
                             M_,
                             N_,
                             K_,
                             stride_A_,
                             stride_B_,
                             stride_C_,
                             KBatch},
              p_sorted_token_ids(p_sorted_token_ids_),
              p_sorted_expert_ids(p_sorted_expert_ids_),
              p_max_token_id(p_max_token_id_),
              NumTokens(NumTokens_),
              TopK(TopK_)
        {
        }

        CK_TILE_HOST static constexpr MoeGemmKernelArgs
        MakeKernelArgs(const MoeGemmHostArgs& hostArgs)
        {
            printf("in moe gemm kernel args!");
            return MoeGemmKernelArgs{hostArgs.p_sorted_token_ids,
                                     hostArgs.p_sorted_expert_ids,
                                     hostArgs.p_max_token_id,
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
                                     1
                                     /*hostArgs.k_batch*/};
        }
    };

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        using P_ = GemmPipeline;
        return concat('_', "moe_gemm", gemm_prec_str<ADataType, BDataType>,
                      concat('x', P_::MPerBlock, P_::NPerBlock, P_::KPerBlock),
                      concat('x', P_::GetVectorSizeA(), P_::GetVectorSizeB(), P_::GetVectorSizeC()),
                      concat('x', P_::kPadM, P_::kPadN, P_::kPadK));
        // clang-format on
    }

    __host__ static constexpr auto BlockSize() -> dim3 { return dim3(KernelBlockSize); }

    __host__ static constexpr auto GridSize(index_t M, index_t N, index_t KBatch)
    {
        // TODO: remove assertion
        assert(KBatch == 1);
        return Base::GridSize(M, N, KBatch);
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetSmemSize() -> index_t
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    template <typename AView>
    CK_TILE_DEVICE static auto GetATransformGemmView(const AView& view, const index_t token_id)
    {
        if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            return transform_tensor_view(
                view,
                make_tuple(make_indexing_transform(
                               view.get_tensor_descriptor().get_length(number<0>()), token_id),
                           make_pass_through_transform(
                               view.get_tensor_descriptor().get_length(number<1>()))),
                make_tuple(sequence<0>{}, sequence<1>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        else
            return transform_tensor_view(
                view,
                make_tuple(make_pass_through_transform(
                               view.get_tensor_descriptor().get_length(number<0>())),
                           make_indexing_transform(
                               view.get_tensor_descriptor().get_length(number<1>()), token_id)),
                make_tuple(sequence<0>{}, sequence<1>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
    }

    template <typename CView>
    CK_TILE_DEVICE static auto GetCTransformGemmView(const CView& view, const index_t token_id)
    {
        if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, CLayout>)
            return transform_tensor_view(
                view,
                make_tuple(make_indexing_transform(
                               view.get_tensor_descriptor().get_length(number<0>()), token_id),
                           make_pass_through_transform(
                               view.get_tensor_descriptor().get_length(number<1>()))),
                make_tuple(sequence<0>{}, sequence<1>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        else
            return transform_tensor_view(
                view,
                make_tuple(make_pass_through_transform(
                               view.get_tensor_descriptor().get_length(number<0>())),
                           make_indexing_transform(
                               view.get_tensor_descriptor().get_length(number<1>()), token_id)),
                make_tuple(sequence<0>{}, sequence<1>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
    }

    template <typename PadView>
    CK_TILE_DEVICE static auto TransformGemmPadViews(const PadView& views, const index_t token_id)
    {
        auto a_pad_view = views.at(number<0>());
        auto b_pad_view = views.at(number<1>());
        auto c_pad_view = views.at(number<2>());

        const auto a_gather_view = GetATransformGemmView(a_pad_view, token_id);
        // TODO： Caculate expert offset of the buf in B.

        // const auto c_scatter_view = GetCTransformGemmView(c_pad_view, token_id);
        // if (token_id){}
        return make_tuple(a_gather_view, b_pad_view, c_pad_view);
    }

    template <typename PadView>
    CK_TILE_DEVICE static auto
    MakeGemmTileWindows(const PadView& views, const index_t i_m, const index_t i_n)
    {
        const auto& a_pad_view = views.at(number<0>{});
        const auto& b_pad_view = views.at(number<1>{});
        const auto& c_pad_view = views.at(number<2>{});
        if(i_m) {}
        const auto& a_block_window = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return make_tile_window(a_pad_view,
                                        make_tuple(number<TilePartitioner::MPerBlock>{},
                                                   number<TilePartitioner::KPerBlock>{}),
                                        {0, 0});
            }
            else
            {
                return make_tile_window(a_pad_view,
                                        make_tuple(number<TilePartitioner::KPerBlock>{},
                                                   number<TilePartitioner::MPerBlock>{}),
                                        {0, 0});
            }
        }();

        const auto& b_block_window = [&]() {
            if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>)
            {
                return make_tile_window(b_pad_view,
                                        make_tuple(number<TilePartitioner::NPerBlock>{},
                                                   number<TilePartitioner::KPerBlock>{}),
                                        {i_n, 0});
            }
            else
            {
                return make_tile_window(b_pad_view,
                                        make_tuple(number<TilePartitioner::KPerBlock>{},
                                                   number<TilePartitioner::NPerBlock>{}),
                                        {0, i_n});
            }
        }();

        auto c_block_window = make_tile_window(
            c_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {0, i_n});

        return make_tuple(a_block_window, b_block_window, c_block_window);
    }

    CK_TILE_DEVICE void operator()(const MoeGemmKernelArgs gemm_desc) const
    {
        // TODO: implement C scatter store accordring to expert_id
        // TODO: the branch without swizzle
        const index_t max_token_id = __builtin_amdgcn_readfirstlane(gemm_desc.p_max_token_id[0]);
        const index_t block_id     = ck_tile::get_block_1d_id();

        // TODO: check the block id caculation
        const auto [expert_blk_id, _] =
            OffsetTile1DPartitioner::GetOffsetedTileIndex(0, gemm_desc.M, gemm_desc.N);

        if(expert_blk_id * TilePartitioner::MPerBlock >= max_token_id)
            return;

        const index_t NBlocks        = gemm_desc.N / TilePartitioner::NPerBlock;
        const index_t expert_id      = gemm_desc.p_sorted_expert_ids[expert_blk_id];
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
            block_id_start_in_expert % 8 + block_id_start_in_expert / (8 * expert_swizzle) * 8);

        const auto a_coord         = GemmPipeline::GetACoord(); // 2d thread offset, [i_row, i_col]
        const auto sorted_token_id = a_coord[number<0>{}] + im * TilePartitioner::MPerBlock;

        // constexpr auto AMRepeat = GemmPipeline::GetAMRepeat();

        // ck_tile::statically_indexed_array<ck_tile::index_t, AMRepeat> gather_offset;
        // static_for<0, AMRepeat, 1>{}([&](auto thr_offset_m){
        //     const index_t fused_token = gemm_desc.p_sorted_token_ids[sorted_token_id +
        //     thr_offset_m]; gather_offset(thr_offset_m) = fused_token & 0xffffff;
        // });

        const index_t fused_token = gemm_desc.p_sorted_token_ids[sorted_token_id];
        // printf("a_coord[number<0>{}]: %d \n",a_coord[number<0>{}]);

        // TODO: token_id should include topk offset depends on ffn1 or ffn2
        const index_t token_id = fused_token & 0xffffff;

        // const index_t expert_stride = __builtin_amdgcn_readfirstlane(problem.N * problem.K);

        const typename Base::SplitKBatchOffset splitk_batch_offset(gemm_desc);
        // options
        const ADataType* a_ptr =
            static_cast<const ADataType*>(gemm_desc.a_ptr) + splitk_batch_offset.a_k_split_offset;
        const BDataType* b_ptr =
            static_cast<const BDataType*>(gemm_desc.b_ptr) + splitk_batch_offset.b_k_split_offset;
        CDataType* c_ptr = static_cast<CDataType*>(gemm_desc.c_ptr);

        const auto& gemm_tensor_views_tuple =
            Base::MakeGemmTensorViews(a_ptr, b_ptr, c_ptr, gemm_desc, splitk_batch_offset);
        const auto& gemm_pad_views    = Base::MakeGemmPadViews(gemm_tensor_views_tuple);
        const auto& transformed_views = TransformGemmPadViews(gemm_pad_views, token_id);
        auto gemm_tile_windows        = MakeGemmTileWindows(
            transformed_views, im * TilePartitioner::MPerBlock, in * TilePartitioner::NPerBlock);
        const index_t num_loop =
            __builtin_amdgcn_readfirstlane(TilePartitioner::GetLoopNum(gemm_desc.K));

        // printf("num_loop: %d", num_loop);

        static_assert(GemmPipeline::DoubleSmemBuffer == true,
                      "For now, only support doublesmembuffer");

        __shared__ char smem_ptr_0[GetSmemSize()];
        __shared__ char smem_ptr_1[GetSmemSize()];
        // Run GEMM cooperatively by whole workgroup.
        const auto& a_block_window = gemm_tile_windows.at(number<0>{});
        const auto& b_block_window = gemm_tile_windows.at(number<1>{});

        const auto& c_block_tile = GemmPipeline{}.template operator()(
            a_block_window, b_block_window, num_loop, smem_ptr_0, smem_ptr_1);

        // Run Epilogue Pipeline
        auto& c_block_window = gemm_tile_windows.at(number<2>{});

        EpiloguePipeline{}.template operator()<decltype(c_block_window), decltype(c_block_tile)>(
            c_block_window,
            c_block_tile,
            smem_ptr_0,
            gemm_desc.p_sorted_token_ids,
            im * TilePartitioner::MPerBlock);
    }
};

} // namespace ck_tile
