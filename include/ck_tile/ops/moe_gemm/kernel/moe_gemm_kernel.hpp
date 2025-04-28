// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/utility/literals.hpp"
#include "ck_tile/ops/flatmm/kernel/flatmm_kernel.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp"
#include "ck_tile/host.hpp"

// #define disable_tile_gs

namespace ck_tile {

struct MoeGemmHostArgs : public ck_tile::FlatmmHostArgs
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
                                 const void* b_shuffle_ptr_,
                                 void* c_ptr_,
                                 ck_tile::index_t NumTokens_,
                                 ck_tile::index_t TopK_,
                                 ck_tile::index_t k_batch_,
                                 ck_tile::index_t M_,
                                 ck_tile::index_t N_,
                                 ck_tile::index_t K_,
                                 ck_tile::index_t stride_A_,
                                 ck_tile::index_t stride_B_,
                                 ck_tile::index_t stride_C_)
        : FlatmmHostArgs(a_ptr_, b_shuffle_ptr_, c_ptr_, k_batch_, M_, N_, K_, stride_A_, stride_B_, stride_C_),
          NumTokens(NumTokens_),
          TopK(TopK_),
          p_sorted_token_ids(p_sorted_token_ids_),
          p_sorted_expert_ids(p_sorted_expert_ids_),
          p_max_token_id(p_max_token_id_)
    {
    }
    // TODO: why kBatch?
    // private:
    // static constexpr index_t KBatch = 1;
};

template <typename TilePartitioner_,
          typename FlatmmPipeline_,
          typename EpiloguePipeline_>
struct MoeGemmKernel
{
    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using FlatmmPipeline   = remove_cvref_t<FlatmmPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;
    using ALayout          = remove_cvref_t<typename FlatmmPipeline::ALayout>;
    using BLayout          = remove_cvref_t<typename FlatmmPipeline::BLayout>;
    using CLayout          = remove_cvref_t<typename FlatmmPipeline::CLayout>;
    using BlockGemmShape =
        remove_cvref_t<typename FlatmmPipeline::BlockGemmShape>; // TileFlatmmShape

    static constexpr bool IsInputGemm = FlatmmPipeline::IsInputGemm;

    using ADataType = remove_cvref_t<typename FlatmmPipeline::ADataType>;
    using BDataType = remove_cvref_t<typename FlatmmPipeline::BDataType>;
    using CDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    using OffsetTile1DPartitioner = OffsettedTile1DPartitioner<TilePartitioner>;
    static constexpr index_t KernelBlockSize = FlatmmPipeline::BlockSize;

    static constexpr auto I0 = number<0>();
    static constexpr auto I1 = number<1>();
    static constexpr auto I2 = number<2>();

    struct MoeGemmKernelArgs
    {
        const ck_tile::index_t* p_sorted_token_ids;
        const ck_tile::index_t* p_sorted_expert_ids;
        const ck_tile::index_t* p_max_token_id;
        const void* p_a_ptr;
        const void* p_b_shuffle_ptr;
        void* p_c_ptr;
        ck_tile::index_t NumTokens;
        ck_tile::index_t TopK;
        ck_tile::index_t M;
        ck_tile::index_t N;
        ck_tile::index_t K;
        ck_tile::index_t stride_A;
        ck_tile::index_t stride_B;
        ck_tile::index_t stride_C;
        ck_tile::index_t k_batch;
        //
        // CK_TILE_HOST MoeGemmKernelArgs() noexcept = default;
        // CK_TILE_HOST MoeGemmKernelArgs(const ck_tile::index_t* p_sorted_token_ids_,
        //                                const ck_tile::index_t* p_sorted_expert_ids_,
        //                                const ck_tile::index_t* p_max_token_id_,
        //                                const void* a_ptr_,
        //                                const void* b_shuffle_ptr_,
        //                                void* c_ptr_,
        //                                ck_tile::index_t NumTokens_,
        //                                ck_tile::index_t TopK_,
        //                                ck_tile::index_t M_,
        //                                ck_tile::index_t N_,
        //                                ck_tile::index_t K_,
        //                                ck_tile::index_t stride_A_,
        //                                ck_tile::index_t stride_B_,
        //                                ck_tile::index_t stride_C_,
        //                                ck_tile::index_t KBatch) :
        //       p_sorted_token_ids(p_sorted_token_ids_),
        //       p_sorted_expert_ids(p_sorted_expert_ids_),
        //       p_max_token_id(p_max_token_id_),
        //       p_a_ptr(a_ptr_),
        //       p_b_shuffle_ptr(b_shuffle_ptr_),
        //       p_c_ptr(c_ptr_),
        //       NumTokens(NumTokens_),
        //       TopK(TopK_),
        //       M(M_),
        //       N(N_),
        //       K(K_),
        //       stride_A(stride_A_),
        //       stride_B(stride_B_),
        //       stride_C(stride_C_),
        //       k_batch(KBatch)
        // {
        // }

    };

    CK_TILE_HOST static constexpr MoeGemmKernelArgs MakeKernelArgs(const MoeGemmHostArgs& hostArgs)
    {
		printf("in moe gemm kernel args! \n");
        return MoeGemmKernelArgs{hostArgs.p_sorted_token_ids,
                                 hostArgs.p_sorted_expert_ids,
                                 hostArgs.p_max_token_id,
                                 hostArgs.a_ptr,
                                 hostArgs.b_shuffle_ptr,
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

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        using P_ = FlatmmPipeline;
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
        assert(KBatch == 1);
        return dim3(TilePartitioner::GridSize(M, N), 1, KBatch);
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetSmemSize() -> index_t
    {
        return max(FlatmmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    struct SplitKBatchOffset
    {
        __device__ SplitKBatchOffset(const MoeGemmKernelArgs& kargs,
                                     const std::size_t k_id = blockIdx.z)
        {
            constexpr auto K1   = TilePartitioner::BlockGemmShape::WarpTile::at(number<2>{});
            const index_t K_t   = __builtin_amdgcn_readfirstlane(kargs.k_batch * K1);
            const index_t KRead = __builtin_amdgcn_readfirstlane((kargs.K + K_t - 1) / K_t * K1);

            if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                a_k_split_offset = __builtin_amdgcn_readfirstlane(k_id * KRead);
            }
            else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                a_k_split_offset = __builtin_amdgcn_readfirstlane(k_id * KRead * kargs.stride_A);
            }

            if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, BLayout>)
            {
                b_k_split_offset = __builtin_amdgcn_readfirstlane(k_id * KRead * kargs.stride_B);
            }
            else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, BLayout>)
            {
                b_k_split_offset = __builtin_amdgcn_readfirstlane(k_id * KRead);
            }

            if(k_id < static_cast<uint32_t>(kargs.k_batch - 1))
            {
                splitted_k = __builtin_amdgcn_readfirstlane(KRead);
            }
            else
            {
                splitted_k = __builtin_amdgcn_readfirstlane(kargs.K - KRead * (kargs.k_batch - 1));
            }
        }

        index_t a_k_split_offset;
        index_t b_k_split_offset;
        index_t splitted_k;
    };

    template <memory_operation_enum DstInMemOp = memory_operation_enum::set>
    CK_TILE_DEVICE static auto MakeGemmTensorViews(const ADataType* a_ptr,
                                                   const BDataType* b_flat_ptr,
                                                   CDataType* c_ptr,
                                                   const MoeGemmKernelArgs& kargs,
                                                   const SplitKBatchOffset& splitk_batch_offset)
    {
        // static_assert(!TilePartitioner::BlockGemmShape::PermuteA, "Not implemented!");
        const auto& a_tensor_view = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_ptr,
                    make_tuple(IsInputGemm ? kargs.NumTokens : kargs.NumTokens * kargs.TopK,
                               splitk_batch_offset.splitted_k),
                    make_tuple(kargs.stride_A, 1),
                    number<FlatmmPipeline::GetVectorSizeA()>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_ptr,
                    make_tuple(splitk_batch_offset.splitted_k,
                               IsInputGemm ? kargs.NumTokens : kargs.NumTokens * kargs.TopK),
                    make_tuple(kargs.stride_A, 1),
                    number<FlatmmPipeline::GetVectorSizeA()>{},
                    number<1>{});
            }
        }();

        index_t kFlatK = FlatmmPipeline::flatKPerWarp * (splitk_batch_offset.splitted_k /
                                                         BlockGemmShape::WarpTile::at(number<2>{}));
        index_t kFlatN = kargs.N * kargs.K / kFlatK;
        const auto& b_flat_tensor_view = [&]() {
            return make_naive_tensor_view<address_space_enum::global>(
                b_flat_ptr,
                make_tuple(kFlatN, kFlatK),
                make_tuple(kFlatK, 1),
                number<FlatmmPipeline::GetVectorSizeB()>{},
                number<1>{});
        }();


        // const auto& b_tensor_view = [&]() {
        //     if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
        //     {
        //         if constexpr(TilePartitioner::BlockGemmShape::PermuteB)
        //         {
        //             constexpr index_t K1          = FlatmmPipeline::GetSmemPackB();
        //             const index_t K0              = splitk_batch_offset.splitted_k / K1;
        //             constexpr index_t VectorSizeB = std::min(K1, FlatmmPipeline::GetVectorSizeB());
        //             const auto b_k0_n_k1_desc =
        //                 make_naive_tensor_descriptor(make_tuple(K0, kargs.N, K1),
        //                                              make_tuple(kargs.N * K1, K1, I1),
        //                                              number<VectorSizeB>{},
        //                                              number<1>{});
        //             const auto b_n_k_desc = transform_tensor_descriptor(
        //                 b_k0_n_k1_desc,
        //                 make_tuple(make_merge_transform(make_tuple(K0, K1)),
        //                            make_pass_through_transform(kargs.N)),
        //                 make_tuple(sequence<0, 2>{}, sequence<1>{}),
        //                 make_tuple(sequence<0>{}, sequence<1>{}));
        //             return make_tensor_view<address_space_enum::global>(b_ptr, b_n_k_desc);
        //         }
        //         else
        //         {
        //             return make_naive_tensor_view<address_space_enum::global>(
        //                 b_ptr,
        //                 make_tuple(splitk_batch_offset.splitted_k, kargs.N),
        //                 make_tuple(kargs.stride_B, 1),
        //                 number<FlatmmPipeline::GetVectorSizeB()>{},
        //                 number<1>{});
        //         }
        //     }
        //     else
        //     {
        //         if constexpr(TilePartitioner::BlockGemmShape::PermuteB)
        //         {
        //             constexpr index_t K1          = FlatmmPipeline::GetSmemPackB();
        //             const index_t K0              = splitk_batch_offset.splitted_k / K1;
        //             constexpr index_t VectorSizeB = std::min(K1, FlatmmPipeline::GetVectorSizeB());
        //             const auto b_k0_n_k1_desc =
        //                 make_naive_tensor_descriptor(make_tuple(K0, kargs.N, K1),
        //                                              make_tuple(kargs.N * K1, K1, I1),
        //                                              number<VectorSizeB>{},
        //                                              number<1>{});
        //             const auto b_n_k_desc = transform_tensor_descriptor(
        //                 b_k0_n_k1_desc,
        //                 make_tuple(make_merge_transform(make_tuple(K0, K1)),
        //                            make_pass_through_transform(kargs.N)),
        //                 make_tuple(sequence<0, 2>{}, sequence<1>{}),
        //                 make_tuple(sequence<1>{}, sequence<0>{}));
        //             return make_tensor_view<address_space_enum::global>(b_ptr, b_n_k_desc);
        //         }
        //         else
        //         {
        //             return make_naive_tensor_view<address_space_enum::global>(
        //                 b_ptr,
        //                 make_tuple(kargs.N, splitk_batch_offset.splitted_k),
        //                 make_tuple(kargs.stride_B, 1),
        //                 number<FlatmmPipeline::GetVectorSizeB()>{},
        //                 number<1>{});
        //         }
        //     }
        // }();

        // TODO: enable vector write for C in ColMajor
        const auto& c_tensor_view = [&]() {
            if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    c_ptr,
                    make_tuple(IsInputGemm ? kargs.NumTokens * kargs.TopK : kargs.NumTokens,
                               kargs.N),
                    make_tuple(kargs.stride_C, 1),
                    number<EpiloguePipeline::GetVectorSizeC()>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    c_ptr,
                    make_tuple(IsInputGemm ? kargs.NumTokens * kargs.TopK : kargs.NumToken,
                               kargs.N),
                    make_tuple(1, kargs.stride_C),
                    number<1>{},
                    number<1>{});
            }
        }();

        return make_tuple(a_tensor_view, b_flat_tensor_view, c_tensor_view);
    }

    template <typename TensorView>
    CK_TILE_DEVICE static auto MakeGemmPadViews(const TensorView& views)
    {
        const auto& a_pad_view = [&]() {
            const auto& a_tensor_view = views.at(I0);
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(a_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::KPerBlock>{}),
                                       sequence<false, FlatmmPipeline::kPadK>{});
            }
            else
            {
                return pad_tensor_view(a_tensor_view,
                                       make_tuple(number<TilePartitioner::KPerBlock>{},
                                                  number<TilePartitioner::MPerBlock>{}),
                                       sequence<false, FlatmmPipeline::kPadM>{});
            }
        }();

        const auto& b_flat_tensor_view = views.at(I1);

        // TODO vector write in for C in ColMajor
        const auto& c_pad_view = [&]() {
            const auto& c_tensor_view = views.at(I2);
            if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(c_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<false, FlatmmPipeline::kPadN>{});
            }
            else
            {
                return pad_tensor_view(c_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<FlatmmPipeline::kPadM, false>{});
            }
        }();

        return make_tuple(a_pad_view, b_flat_tensor_view, c_pad_view);
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

    // template <typename CView>
    // CK_TILE_DEVICE static auto GetCTransformGemmView(const CView& view, const index_t token_id)
    // {
    //     if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, CLayout>)
    //         return transform_tensor_view(
    //             view,
    //             make_tuple(make_indexing_transform(
    //                            view.get_tensor_descriptor().get_length(number<0>()), token_id),
    //                        make_pass_through_transform(
    //                            view.get_tensor_descriptor().get_length(number<1>()))),
    //             make_tuple(sequence<0>{}, sequence<1>{}),
    //             make_tuple(sequence<0>{}, sequence<1>{}));
    //     else
    //         return transform_tensor_view(
    //             view,
    //             make_tuple(make_pass_through_transform(
    //                            view.get_tensor_descriptor().get_length(number<0>())),
    //                        make_indexing_transform(
    //                            view.get_tensor_descriptor().get_length(number<1>()), token_id)),
    //             make_tuple(sequence<0>{}, sequence<1>{}),
    //             make_tuple(sequence<0>{}, sequence<1>{}));
    // }

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
        (void)i_m;

        const auto& a_pad_view      = views.at(number<0>{});
        const auto& b_flat_pad_view = views.at(number<1>{});
        const auto& c_pad_view      = views.at(number<2>{});
        // if(i_m) {}
 
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

        const auto& b_flat_block_window =
            make_tile_window(b_flat_pad_view,
                             make_tuple(number<FlatmmPipeline::flatNPerWarp>{},
                                        number<FlatmmPipeline::flatKPerWarp>{}),
                             {static_cast<int>(i_n / BlockGemmShape::WarpTile::at(I1)), 0});

        auto c_block_window = make_tile_window(
            c_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            // {i_m, i_n});
            {0, i_n});

        return make_tuple(a_block_window, b_flat_block_window, c_block_window);
    }

    template <bool IsInputGemm = true>
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

        // printf("expert_blk_id: %d, expert_id: %d \n",expert_blk_id, expert_id);

        // expert_id = expert_blk_id;

        const index_t block_start = prefix_blk_m * NBlocks;

        const index_t ecnt           = blk_cnt_of_eid - prefix_blk_m;
        const index_t expert_swizzle = ecnt > 0 ? ecnt : 1;
        // index_t block_end = block_start + blk_cnt_of_eid * NBlocks;

        const index_t block_id_start_in_expert = block_id - block_start;
        const index_t im = __builtin_amdgcn_readfirstlane(prefix_blk_m + block_id_start_in_expert /
                                                                             8 % expert_swizzle);
        const index_t in = __builtin_amdgcn_readfirstlane(
            block_id_start_in_expert % 8 + block_id_start_in_expert / (8 * expert_swizzle) * 8);

        const auto a_coord         = FlatmmPipeline::GetACoord(); // 2d thread offset, [i_row, i_col]
#ifdef disable_tile_gs
        const auto sorted_token_id = a_coord[number<0>{}] + im * TilePartitioner::MPerBlock;
        const index_t fused_token = gemm_desc.p_sorted_token_ids[sorted_token_id];

        // TODO: token_id should include topk offset depends on ffn1 or ffn2
        constexpr index_t token_id_mask = 0xffffff;
        index_t token_id = fused_token & token_id_mask;
        if constexpr(!IsInputGemm)
        {
            constexpr index_t token_id_offset = 24;
            token_id = token_id * gemm_desc.TopK + (fused_token >> token_id_offset);
        }
#else
		constexpr ck_tile::index_t MRepeat = FlatmmPipeline::GetAMRepeat();
		statically_indexed_array<ck_tile::index_t, MRepeat> a_offsets;

        constexpr index_t token_id_mask = 0xffffff;
        constexpr index_t token_id_offset = 24;

        // constexpr auto kMWave = TilePartitioner::BlockGemmShape::BlockWarps::at(I0);
        // constexpr auto kNWave = TilePartitioner::BlockGemmShape::BlockWarps::at(I1);
        // const index_t iMWarp = get_warp_id() / kNWave;
		static_for<0, MRepeat, 1>{}([&](auto m0) {
            // const auto sorted_token_id = a_coord[I0] + im * TilePartitioner::MPerBlock +
            //     iMWarp * TilePartitioner::MPerBlock / kMWave +
            //     m0 * TilePartitioner::MPerBlock / kMWave / MRepeat;
            const auto sorted_token_id = a_coord[I0] + im * TilePartitioner::MPerBlock +
                m0 * TilePartitioner::MPerBlock / MRepeat;
            const index_t fused_token = gemm_desc.p_sorted_token_ids[sorted_token_id];

            // TODO: token_id should include topk offset depends on ffn1 or ffn2
            index_t gather_token_id  = fused_token & token_id_mask;
            if constexpr(!IsInputGemm)
            {
                gather_token_id = gather_token_id * gemm_desc.TopK + (fused_token >> token_id_offset);
            }
			a_offsets[m0] = gather_token_id * gemm_desc.stride_A;
        });
#endif

        const index_t expert_stride = __builtin_amdgcn_readfirstlane(gemm_desc.N * gemm_desc.K);

        const SplitKBatchOffset splitk_batch_offset(gemm_desc);
        // options
        const ADataType* a_ptr =
            static_cast<const ADataType*>(gemm_desc.p_a_ptr) + splitk_batch_offset.a_k_split_offset;
        const BDataType* b_shuffle_ptr = static_cast<const BDataType*>(gemm_desc.p_b_shuffle_ptr) +
             splitk_batch_offset.b_k_split_offset + expert_stride * expert_id;
        CDataType* c_ptr = static_cast<CDataType*>(gemm_desc.p_c_ptr);

        const auto& gemm_tensor_views_tuple =
            MakeGemmTensorViews(a_ptr, b_shuffle_ptr, c_ptr, gemm_desc, splitk_batch_offset);
        const auto& gemm_pad_views    = MakeGemmPadViews(gemm_tensor_views_tuple);

#ifdef disable_tile_gs
        const auto& transformed_views = TransformGemmPadViews(gemm_pad_views, token_id);
        auto gemm_tile_windows        = MakeGemmTileWindows(
            transformed_views, im * TilePartitioner::MPerBlock, in * TilePartitioner::NPerBlock);
#else
        auto gemm_tile_windows        = MakeGemmTileWindows(
            gemm_pad_views, im * TilePartitioner::MPerBlock, in * TilePartitioner::NPerBlock);
#endif

        const index_t num_loop =
            __builtin_amdgcn_readfirstlane(TilePartitioner::GetLoopNum(gemm_desc.K));

        // printf("num_loop: %d", num_loop);

        // static_assert(FlatmmPipeline::DoubleSmemBuffer == true,
        //               "For now, only support doublesmembuffer");

        __shared__ char smem_ptr_0[GetSmemSize()];
        // __shared__ char smem_ptr_1[GetSmemSize()];
        // Run GEMM cooperatively by whole workgroup.
        const auto& a_block_window = gemm_tile_windows.at(number<0>{});
        const auto& b_block_window = gemm_tile_windows.at(number<1>{});

#ifdef disable_tile_gs
        const auto& c_block_tile = FlatmmPipeline{}.template operator()(
            a_block_window, b_block_window, num_loop, smem_ptr_0);
#else
        auto a_gather_block_tile = ck_tile::make_tile_scatter_gather(
            a_block_window.get_bottom_tensor_view(),
            a_block_window.get_window_lengths(),
            a_block_window.get_window_origin(),
            FlatmmPipeline::GetADramTileDistribution(),
            a_offsets); // K DRAM tile window for
        const auto& c_block_tile = FlatmmPipeline{}.template operator()(
            a_gather_block_tile, b_block_window, num_loop, smem_ptr_0);
#endif

        // Run Epilogue Pipeline
        auto& c_block_window = gemm_tile_windows.at(number<2>{});

        EpiloguePipeline{}.template operator()<decltype(c_block_window),
                                               decltype(c_block_tile)>(
            c_block_window,
            c_block_tile,
            smem_ptr_0,
            gemm_desc.p_sorted_token_ids,
            im * TilePartitioner::MPerBlock,
            gemm_desc.TopK,
            gemm_desc.stride_C);
    }
};

} // namespace ck_tile
