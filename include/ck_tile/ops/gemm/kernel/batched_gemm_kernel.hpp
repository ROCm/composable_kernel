// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

namespace ck_tile {

struct BatchedGemmHostArgs
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
    index_t batch_stride_A;
    index_t batch_stride_B;
    index_t batch_stride_C;
    index_t batch_count;
};

template <typename TilePartitioner_, typename GemmPipeline_, typename EpiloguePipeline_>
struct BatchedGemmKernel
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
    using CDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    struct BatchedGemmKargs
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
        index_t batch_stride_A;
        index_t batch_stride_B;
        index_t batch_stride_C;
        index_t batch_count;
    };

    using Kargs = BatchedGemmKargs;
    using Hargs = BatchedGemmHostArgs;

    __host__ static constexpr auto GridSize(const Hargs& h)
    {
        return TilePartitioner::GridSize(h.M, h.N, h.batch_count);
    }

    __host__ static constexpr auto BlockSize() { return dim3(KernelBlockSize); }

    CK_TILE_HOST static constexpr BatchedGemmKargs MakeKargs(const Hargs& h)
    {
        Kargs k;
        k.a_ptr          = h.a_ptr;
        k.b_ptr          = h.b_ptr;
        k.c_ptr          = h.c_ptr;
        k.M              = h.M;
        k.N              = h.N;
        k.K              = h.K;
        k.stride_A       = h.stride_A;
        k.stride_B       = h.stride_B;
        k.stride_C       = h.stride_C;
        k.batch_stride_A = h.batch_stride_A;
        k.batch_stride_B = h.batch_stride_B;
        k.batch_stride_C = h.batch_stride_C;
        k.batch_count    = h.batch_count;
        return k;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const auto [i_m, i_n] = TilePartitioner{}();
        const auto i_batch    = __builtin_amdgcn_readfirstlane(blockIdx.z);

        //  options
        const auto batch_stride_A = __builtin_amdgcn_readfirstlane(kargs.batch_stride_A);
        const auto batch_offset_A = __builtin_amdgcn_readfirstlane(i_batch * batch_stride_A);
        const ADataType* a_start  = static_cast<const ADataType*>(kargs.a_ptr);

        const auto batch_stride_B = __builtin_amdgcn_readfirstlane(kargs.batch_stride_B);
        const auto batch_offset_B = __builtin_amdgcn_readfirstlane(i_batch * batch_stride_B);
        const BDataType* b_start  = static_cast<const BDataType*>(kargs.b_ptr);

        // Convert pointers to tensor views
        auto a_tensor_view = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_start + batch_offset_A,
                    make_tuple(kargs.M, kargs.K),
                    make_tuple(kargs.stride_A, 1),
                    number<GemmPipeline::VectorSizeA>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_start + batch_offset_A,
                    make_tuple(kargs.M, kargs.K),
                    make_tuple(1, kargs.stride_A),
                    number<1>{},
                    number<1>{});
            }
        }();

        auto b_tensor_view = [&]() {
            if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    b_start + batch_offset_B,
                    make_tuple(kargs.N, kargs.K),
                    make_tuple(1, kargs.stride_B),
                    number<1>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    b_start + batch_offset_B,
                    make_tuple(kargs.N, kargs.K),
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
        // clang-format on

        auto b_block_window = make_tile_window(
            b_pad_view,
            make_tuple(number<TilePartitioner::kN>{}, number<TilePartitioner::kK>{}),
            {i_n, 0});

        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        const index_t num_loop = TilePartitioner::GetLoopNum(kargs.K);

        // Run GEMM cooperatively by whole wokrgroup.
        auto c_block_tile =
            GemmPipeline{}.template operator()(a_block_window, b_block_window, num_loop, smem_ptr);

        const auto batch_stride_C = __builtin_amdgcn_readfirstlane(kargs.batch_stride_C);
        const auto batch_offset_C = __builtin_amdgcn_readfirstlane(i_batch * batch_stride_C);
        CDataType* c_start        = static_cast<CDataType*>(kargs.c_ptr);
        auto c_tensor_view        = [&]() {
            if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    c_start + batch_offset_C,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(kargs.stride_C, 1),
                    number<GemmPipeline::VectorSizeC>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    c_start + batch_offset_C,
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
        auto c_block_window = make_tile_window(
            c_pad_view,
            make_tuple(number<TilePartitioner::kM>{}, number<TilePartitioner::kN>{}),
            {i_m, i_n});

        EpiloguePipeline{}(c_block_window, c_block_tile);
    }
};

} // namespace ck_tile
