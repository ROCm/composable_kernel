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
    };

    CK_TILE_HOST static constexpr GemmCommonKargs MakeKargs(const void* a_ptr,
                                                            const void* b_ptr,
                                                            void* c_ptr,
                                                            index_t M,
                                                            index_t N,
                                                            index_t K,
                                                            index_t stride_A,
                                                            index_t stride_B,
                                                            index_t stride_C)
    {
        return GemmCommonKargs{a_ptr, b_ptr, c_ptr, M, N, K, stride_A, stride_B, stride_C};
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void operator()(GemmCommonKargs kargs) const
    {
        const auto [i_m, i_n] = TilePartitioner{}();
        // options
        const ADataType* a_start = static_cast<const ADataType*>(kargs.a_ptr);
        const BDataType* b_start = static_cast<const BDataType*>(kargs.b_ptr);
        // Convert pointers to tensor views
        auto a_tensor_view = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_start,
                    make_tuple(kargs.M, kargs.K),
                    make_tuple(kargs.stride_A, 1),
                    number<GemmPipeline::VectorSizeA>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_start,
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
                    b_start,
                    make_tuple(kargs.N, kargs.K),
                    make_tuple(1, kargs.stride_B),
                    number<1>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    b_start,
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

        CDataType* c_start = static_cast<CDataType*>(kargs.c_ptr);
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
        
        using CSubTileDistr = decltype(GemmPipeline::MakeCBlockSubTile());
        CSubTileDistr c_sub_tile;
        // printf("!!!!!!!!!!!!!!!!!!!!");
        // c_sub_tile.get_tile_distribution().print();
        // if (threadIdx.x==0) {
        //     printf("!!!!!!!!!!!!!!!!!!!!~~~ %d %d\n", c_block_tile.get_tile_distribution().get_num_of_dimension_y(), c_sub_tile.get_tile_distribution().get_num_of_dimension_y());
        //     // c_block_tile.get_tile_distribution().print();
        //     constexpr auto c_sub_y_index_zeros = uniform_sequence_gen_t<c_sub_tile.get_tile_distribution().get_num_of_dimension_y(), 0>{};
        //     constexpr auto c_sub_y_lengths = to_sequence(c_sub_tile.get_tile_distribution().get_ys_to_d_descriptor().get_lengths());
        //     c_sub_y_index_zeros.print();
        //     c_sub_y_lengths.print();
        // }
        
        // auto c_sub_y_index_zeros = uniform_sequence_gen_t<c_sub_tile.get_tile_distribution().get_num_of_dimension_y(), 0>{};
        // auto c_sub_y_lengths = to_sequence(c_sub_tile.get_tile_distribution().get_ys_to_d_descriptor().get_lengths());
        // if (threadIdx.x == 0) {
        //     c_sub_y_index_zeros.print();
        //         printf("\n");
        //     c_sub_y_lengths.print();
        //         printf("\n");
        //     printf("%d %d\n", GemmPipeline::NumCSubTile(), c_sub_tile.get_tile_distribution().get_num_of_dimension_y());
        // }
        // auto tbuf = c_block_tile.get_thread_buffer();
        // for (index_t i = 0; i < tbuf.size(); i++) {
        //     if (threadIdx.x<16) {
        //         tbuf.set_as(i, float(threadIdx.x * 100 + i));
        //     } else {
                
        //         tbuf.set_as(i, float(threadIdx.x));
        //     }
        // }
        // c_block_tile.get_thread_buffer() = tbuf;
        // if (threadIdx.x ==0) {
        //     constexpr auto span_2d = decltype(c_block_tile)::get_distributed_spans();
        //     sweep_tile_span(span_2d[number<0>{}], [&](auto idx0) {
        //         sweep_tile_span(span_2d[number<1>{}], [&](auto idx1) {
        //             constexpr auto i_j_idx = make_tuple(idx0, idx1);
        //             printf("%f,", type_convert<float>(c_block_tile(i_j_idx)));
        //         });
        //         printf("\n");
        //     });
        // }
        
        static_for<0, GemmPipeline::NumCSubTile(), 1>{}([&](auto i_m0) 
        {
            constexpr auto c_sub_y_index_zeros = uniform_sequence_gen_t<c_sub_tile.get_tile_distribution().get_num_of_dimension_y(), 0>{};
            constexpr auto c_sub_y_lengths = to_sequence(c_sub_tile.get_tile_distribution().get_ys_to_d_descriptor().get_lengths());
            c_sub_tile.get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
                                                merge_sequences(sequence<i_m0>{}, c_sub_y_index_zeros),
                                                merge_sequences(sequence<1>{}, c_sub_y_lengths));
                                                
            EpiloguePipeline{}(CBlockWindow_pad, c_sub_tile, smem_ptr);
            move_tile_window(CBlockWindow_pad, {TilePartitioner::kM / GemmPipeline::NumCSubTile(), 0});
            
        });
    }
};

} // namespace ck_tile
