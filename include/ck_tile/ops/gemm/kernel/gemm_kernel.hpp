// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"

namespace ck_tile {

template <typename TilePartitioner_, typename GemmPipeline_, typename EpiloguePipeline_>
struct GemmKernel
{
    using TilePartitioner                    = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline                       = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline                   = remove_cvref_t<EpiloguePipeline_>;
    using ALayout                            = remove_cvref_t<typename GemmPipeline::ALayout>;
    using BLayout                            = remove_cvref_t<typename GemmPipeline::BLayout>;
    using CLayout                            = remove_cvref_t<typename GemmPipeline::CLayout>;
    static constexpr index_t KernelBlockSize = GemmPipeline::BlockSize;

    using ADataType = remove_cvref_t<typename GemmPipeline::ADataType>;
    using BDataType = remove_cvref_t<typename GemmPipeline::BDataType>;
    // using CAccDataType = remove_cvref_t<typename GemmPipeline::CDataType>;
    using CDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    __host__ static constexpr auto GridSize(index_t M, index_t N, index_t KBatch)
    {
        return TilePartitioner::GridSize(M, N, KBatch);
    }

    __host__ static constexpr auto BlockSize() { return dim3(KernelBlockSize); }

    struct GemmCommonKargs
    {
        const void* a_ptr;
        const void* b_ptr;
        void* c_ptr;
        index_t M;
        index_t N;
        index_t K;
        index_t stride_A;
        index_t stride_B;
        index_t stride_C;
        index_t KBatch;
    };

    CK_TILE_HOST static constexpr GemmCommonKargs MakeKargs(const void* a_ptr,
                                                            const void* b_ptr,
                                                            void* c_ptr,
                                                            index_t M,
                                                            index_t N,
                                                            index_t K,
                                                            index_t stride_A,
                                                            index_t stride_B,
                                                            index_t stride_C,
                                                            index_t KBatch)
    {
        return GemmCommonKargs{a_ptr, b_ptr, c_ptr, M, N, K, stride_A, stride_B, stride_C, KBatch};
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    struct SplitKBatchOffset
    {
        __device__ SplitKBatchOffset(GemmCommonKargs& kargs)
        {
            constexpr auto K1   = TilePartitioner::BlockGemmShape::WarpTile::at(number<2>{});
            const index_t K_t   = kargs.KBatch * K1;
            const index_t KRead = (kargs.K + K_t - 1) / K_t * K1;

            if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                a_k_split_offset = blockIdx.z * KRead;
            }
            else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                a_k_split_offset = blockIdx.z * KRead * kargs.stride_A;
            }

            if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, BLayout>)
            {
                b_k_split_offset = blockIdx.z * KRead * kargs.stride_B;
            }
            else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, BLayout>)
            {
                b_k_split_offset = blockIdx.z * KRead;
            }

            if(blockIdx.z < static_cast<uint32_t>(kargs.KBatch - 1))
            {
                SplittedK = KRead;
            }
            else
            {
                SplittedK = kargs.K - KRead * (kargs.KBatch - 1);
            }
        }

        index_t a_k_split_offset;
        index_t b_k_split_offset;
        index_t SplittedK;
    };

    CK_TILE_HOST static bool IsSupportedArgument(const GemmCommonKargs& kargs)
    {
        constexpr bool is_output_c_reg_transposed =
            EpiloguePipeline::IsOutputTransposed() != GemmPipeline::IsTransposeC();
        if constexpr(!((GemmPipeline::VectorSizeC % 2 == 0 &&
                        std::is_same_v<CLayout, tensor_layout::gemm::RowMajor> &&
                        is_output_c_reg_transposed) ||
                       !(std::is_same_v<CDataType, half_t> ||
                         std::is_same_v<CDataType, bfloat16_t>)))
        {
            if(kargs.KBatch != 1)
            {
                return false;
            }
        }

        if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.K % TilePartitioner::kK != 0 && GemmPipeline::kPadK == false)
            {
                return false;
            }
            if(kargs.K % GemmPipeline::VectorSizeA != 0)
            {
                return false;
            }
        }
        else
        {
            if(kargs.M % TilePartitioner::kM != 0 && GemmPipeline::kPadM == false)
            {
                return false;
            }
            if(kargs.M % GemmPipeline::VectorSizeA != 0)
            {
                return false;
            }
        }

        if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.N % TilePartitioner::kN != 0 && GemmPipeline::kPadN == false)
            {
                return false;
            }
            if(kargs.N % GemmPipeline::VectorSizeB != 0)
            {
                return false;
            }
        }
        else
        {
            if(kargs.K % TilePartitioner::kK != 0 && GemmPipeline::kPadK == false)
            {
                return false;
            }
            if(kargs.K % GemmPipeline::VectorSizeB != 0)
            {
                return false;
            }
        }

        if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.N % TilePartitioner::kN != 0 && GemmPipeline::kPadN == false)
            {
                return false;
            }
            if(kargs.N % GemmPipeline::VectorSizeC != 0)
            {
                return false;
            }
        }
        else
        {
            if(kargs.M % TilePartitioner::kM != 0 && GemmPipeline::kPadM == false)
            {
                return false;
            }
            if(kargs.M % GemmPipeline::VectorSizeC != 0)
            {
                return false;
            }
        }
        return true;
    }

    CK_TILE_DEVICE void operator()(GemmCommonKargs kargs) const
    {
        const auto [i_m, i_n] = TilePartitioner{}();
        const SplitKBatchOffset splitk_batch_offset(kargs);
        // options
        const ADataType* a_start =
            static_cast<const ADataType*>(kargs.a_ptr) + splitk_batch_offset.a_k_split_offset;
        const BDataType* b_start =
            static_cast<const BDataType*>(kargs.b_ptr) + splitk_batch_offset.b_k_split_offset;
        // Convert pointers to tensor views
        auto a_tensor_view = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_start,
                    make_tuple(kargs.M, splitk_batch_offset.SplittedK),
                    make_tuple(kargs.stride_A, 1),
                    number<GemmPipeline::VectorSizeA>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_start,
                    make_tuple(kargs.M, splitk_batch_offset.SplittedK),
                    make_tuple(1, kargs.stride_A),
                    number<1>{},
                    number<1>{});
            }
        }();

        auto b_tensor_view = [&]() {
            if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    b_start,
                    make_tuple(kargs.N, splitk_batch_offset.SplittedK),
                    make_tuple(1, kargs.stride_B),
                    number<1>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    b_start,
                    make_tuple(kargs.N, splitk_batch_offset.SplittedK),
                    make_tuple(kargs.stride_B, 1),
                    number<GemmPipeline::VectorSizeB>{},
                    number<1>{});
            }
        }();

        auto a_pad_view = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(
                    a_tensor_view,
                    make_tuple(number<TilePartitioner::kM>{}, number<TilePartitioner::kK>{}),
                    sequence<false, GemmPipeline::kPadK>{});
            }
            else
            {
                return pad_tensor_view(
                    a_tensor_view,
                    make_tuple(number<TilePartitioner::kM>{}, number<TilePartitioner::kK>{}),
                    sequence<GemmPipeline::kPadM, false>{});
            }
        }();
        // clang-format on

        auto a_block_window = make_tile_window(
            a_pad_view,
            make_tuple(number<TilePartitioner::kM>{}, number<TilePartitioner::kK>{}),
            {i_m, 0});

        auto b_pad_view = [&]() {
            if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>)
            {
                return pad_tensor_view(
                    b_tensor_view,
                    make_tuple(number<TilePartitioner::kN>{}, number<TilePartitioner::kK>{}),
                    sequence<false, GemmPipeline::kPadK>{});
            }
            else
            {
                return pad_tensor_view(
                    b_tensor_view,
                    make_tuple(number<TilePartitioner::kN>{}, number<TilePartitioner::kK>{}),
                    sequence<GemmPipeline::kPadN, false>{});
            }
        }();

        auto b_block_window = make_tile_window(
            b_pad_view,
            make_tuple(number<TilePartitioner::kN>{}, number<TilePartitioner::kK>{}),
            {i_n, 0});

        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        const index_t num_loop = TilePartitioner::GetLoopNum(splitk_batch_offset.SplittedK);

        // Run GEMM cooperatively by whole wokrgroup.
        auto c_block_tile =
            GemmPipeline{}.template operator()(a_block_window, b_block_window, num_loop, smem_ptr);

        CDataType* c_start = static_cast<CDataType*>(kargs.c_ptr);
        if(kargs.KBatch == 1)
        {
            auto c_tensor_view = [&]() {
                if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
                {
                    return make_naive_tensor_view<address_space_enum::global>(
                        c_start,
                        make_tuple(kargs.M, kargs.N),
                        make_tuple(kargs.stride_C, 1),
                        number<GemmPipeline::VectorSizeC>{},
                        number<1>{});
                }
                else
                {
                    return make_naive_tensor_view<address_space_enum::global>(
                        c_start,
                        make_tuple(kargs.M, kargs.N),
                        make_tuple(1, kargs.stride_C),
                        number<1>{},
                        number<1>{});
                }
            }();

            auto c_pad_view = [&]() {
                if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
                {
                    return pad_tensor_view(
                        c_tensor_view,
                        make_tuple(number<TilePartitioner::kM>{}, number<TilePartitioner::kN>{}),
                        sequence<false, GemmPipeline::kPadN>{});
                }
                else
                {
                    return pad_tensor_view(
                        c_tensor_view,
                        make_tuple(number<TilePartitioner::kM>{}, number<TilePartitioner::kN>{}),
                        sequence<GemmPipeline::kPadM, false>{});
                }
            }();
            auto CBlockWindow_pad = make_tile_window(
                c_pad_view,
                make_tuple(number<TilePartitioner::kM>{}, number<TilePartitioner::kN>{}),
                {i_m, i_n});
            EpiloguePipeline{}(CBlockWindow_pad, c_block_tile);
        }
        else
        {
            auto c_tensor_view = [&]() {
                if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
                {
                    return make_naive_tensor_view<address_space_enum::global,
                                                  memory_operation_enum::atomic_add>(
                        c_start,
                        make_tuple(kargs.M, kargs.N),
                        make_tuple(kargs.stride_C, 1),
                        number<GemmPipeline::VectorSizeC>{},
                        number<1>{});
                }
                else
                {
                    return make_naive_tensor_view<address_space_enum::global,
                                                  memory_operation_enum::atomic_add>(
                        c_start,
                        make_tuple(kargs.M, kargs.N),
                        make_tuple(1, kargs.stride_C),
                        number<1>{},
                        number<1>{});
                }
            }();

            auto c_pad_view = [&]() {
                if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
                {
                    return pad_tensor_view(
                        c_tensor_view,
                        make_tuple(number<TilePartitioner::kM>{}, number<TilePartitioner::kN>{}),
                        sequence<false, GemmPipeline::kPadN>{});
                }
                else
                {
                    return pad_tensor_view(
                        c_tensor_view,
                        make_tuple(number<TilePartitioner::kM>{}, number<TilePartitioner::kN>{}),
                        sequence<GemmPipeline::kPadM, false>{});
                }
            }();
            auto CBlockWindow_pad = make_tile_window(
                c_pad_view,
                make_tuple(number<TilePartitioner::kM>{}, number<TilePartitioner::kN>{}),
                {i_m, i_n});
            // Logical xor
            constexpr bool is_output_c_reg_transposed =
                EpiloguePipeline::IsOutputTransposed() != GemmPipeline::IsTransposeC();
            if constexpr((GemmPipeline::VectorSizeC % 2 == 0 &&
                          std::is_same_v<CLayout, tensor_layout::gemm::RowMajor> &&
                          is_output_c_reg_transposed) ||
                         !(std::is_same_v<CDataType, half_t> ||
                           std::is_same_v<CDataType, bfloat16_t>))
            {
                EpiloguePipeline{}
                    .template operator()<decltype(CBlockWindow_pad),
                                         decltype(c_block_tile),
                                         memory_operation_enum::atomic_add>(CBlockWindow_pad,
                                                                            c_block_tile);
            }
        }
    }
};

} // namespace ck_tile
