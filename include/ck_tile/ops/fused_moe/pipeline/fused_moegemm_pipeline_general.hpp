// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/fused_moe/pipeline/fused_moegemm_pipeline_general_policy.hpp"

namespace ck_tile {

/*
This pipeline deal with a gemm(actually 2 gemm) with one very small(token), one very big(weight)
we need to design the pipeline such that all waves along gemm-N dim (gemm-m only 1 wave)

    <----- gemm-N ------>
    +----+----+----+----+
    | w0 | w1 | w2 | w3 | gemm-m
    +----+----+----+----+
*/
template <typename Problem_, typename Policy_ = FusedMoeGemmPipelineGeneralPolicy>
struct FusedMoeGemmPipeline_General
{
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    using BlockShape = typename Problem::BlockShape; // this is FusedMoeGemmShape

    using ADataType            = typename Problem::ADataType;
    using GDataType            = typename Problem::GDataType;
    using DDataType            = typename Problem::DDataType;
    using AccDataType          = typename Problem::AccDataType;
    using ODataType            = typename Problem::ODataType;
    using AScaleDataType       = typename Problem::AScaleDataType;
    using GScaleDataType       = typename Problem::GScaleDataType;
    using DScaleDataType       = typename Problem::DScaleDataType;
    using YSmoothScaleDataType = typename Problem::YSmoothScaleDataType;
    using TopkWeightDataType   = typename Problem::TopkWeightDataType;
    using IndexDataType        = typename Problem::IndexDataType;
    using YDataType            = typename Problem::YDataType;

    using Traits = typename Problem::Traits;

    static constexpr bool IsGateOnly          = Traits::IsGateOnly;
    static constexpr bool UseSmoothQuant      = Traits::UseSmoothQuant;
    static constexpr bool PadHiddenSize       = Traits::PadHiddenSize;
    static constexpr bool PadIntermediateSize = Traits::PadIntermediateSize;

    static constexpr index_t kAlignmentA = Policy::template GetAlignment_A<Problem>();
    static constexpr index_t kAlignmentG = Policy::template GetAlignment_G<Problem>();
    static constexpr index_t kAlignmentD = Policy::template GetAlignment_D<Problem>();
    static constexpr index_t kAlignmentO = Policy::template GetAlignment_O<Problem>();

    static constexpr index_t SLD_A = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::SLD_A);
    static constexpr index_t GLD_A = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::GLD_A);
    static constexpr index_t GLD_B = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::GLD_B);
    static constexpr index_t GST_O = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::GST_O);

    static constexpr index_t kBlockPerCu = []() {
        if constexpr(Problem::kBlockPerCu != -1)
            return Problem::kBlockPerCu;
        else
        {
            // minimize occupancy
            return 2;
        }
    }();

    static constexpr const char* name = "flatmm_gl";

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeA()
    {
        // matrix a or tokens smem
        return Policy::template GetSmemSize_A<Problem>();
    }
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        // matrix a or tokens smem
        constexpr index_t smem_mat_a = GetSmemSizeA();
        constexpr index_t smem_mat_d = Policy::template GetSmemSize_G<Problem>();
        // shuffle C matrix
        constexpr index_t smem_bridge = Policy::template GetSmemSize_Bridge<Problem>();

        constexpr index_t smem_mat_o = BlockShape::Block_N1 * BlockShape::Block_K1 * sizeof(float);

        return max(smem_mat_a + smem_mat_d, smem_bridge, smem_mat_o);
        // return Policy::template GetSmemSize<Problem>();
    }

    template <typename T>
    CK_TILE_HOST_DEVICE static void
    PrintMem(T& tensor, const char* pstr, unsigned int threadid = 0, unsigned int blockid = 0)
    {
        constexpr auto spans = T::get_distributed_spans();
        int counter          = 0;
        sweep_tile_span(spans[number<0>{}], [&](auto idxn) {
            sweep_tile_span(spans[number<1>{}], [&](auto idxk) {
                constexpr auto i_j_idx = make_tuple(idxn, idxk);
                const auto tile_idx =
                    get_x_indices_from_distributed_indices(tensor.get_tile_distribution(), i_j_idx);
                if(threadIdx.x == threadid && blockIdx.x == 0 && blockIdx.y == blockid &&
                   blockIdx.z == 0)
                {
                    const auto row = tile_idx.at(number<0>{});
                    const auto col = tile_idx.at(number<1>{});
                    printf("in %s row is %d , col is %d, counter is %d, value is: %f"
                           " \n",
                           pstr,
                           row,
                           col,
                           counter,
                           ck_tile::type_convert<float>(tensor(i_j_idx)));
                    counter = counter + 1;
                }
            });
        });
    }
    template <typename AWindow,
              typename GWindow,
              typename DWindow,
              typename OWindow,
              typename CWindow,
              typename WWindow>
    CK_TILE_DEVICE auto operator()(const AWindow& a_window_,
                                   const GWindow& g_window_,
                                   const DWindow& d_window_,
                                   const WWindow& w_window_,
                                   OWindow& o_window_,
                                   CK_TILE_LDS_ADDR void* smem,
                                   index_t hidden_size,
                                   index_t /*intermediate_size*/,
                                   CWindow& /*c_window_*/)
    {
        CK_TILE_LDS_ADDR ADataType* smem_a = reinterpret_cast<CK_TILE_LDS_ADDR ADataType*>(smem);
        CK_TILE_LDS_ADDR GDataType* smem_g = reinterpret_cast<CK_TILE_LDS_ADDR GDataType*>(
            smem_a + GetSmemSizeA() / sizeof(ADataType));

        auto a_lds_view = make_tensor_view<address_space_enum::lds>(
            smem_a, Policy::template MakeLdsBlockDesc_A<Problem>());
        auto a_lds_win = make_tile_window(
            a_lds_view,
            make_tuple(number<BlockShape::Block_M0>{}, number<BlockShape::Block_K0>{}),
            {0, 0});

        auto g_lds_view = make_tensor_view<address_space_enum::lds>(
            smem_g, Policy::template MakeLdsBlockDesc_G<Problem>());
        auto g_lds_win = make_tile_window(
            g_lds_view,
            make_tuple(number<BlockShape::Block_N0>{}, number<BlockShape::Block_K0>{}),
            {0, 0});

        auto a_global_to_dram_window = make_tile_window(
            a_window_.get_bottom_tensor_view(),
            make_tuple(number<BlockShape::Block_M0>{}, number<BlockShape::Block_K0>{}),
            a_window_.get_window_origin(),
            Policy::template MakeGlobalTileDistribution_A<Problem>());

        // load g to register
        auto g_global_to_dram_window = make_tile_window(
            g_window_.get_bottom_tensor_view(),
            make_tuple(number<BlockShape::Block_N0>{}, number<BlockShape::Block_K0>{}),
            g_window_.get_window_origin(),
            Policy::template MakeGlobalTileDistribution_G<Problem>());

#if 0
        PrintMem(g_dram_block, "G", 0);
#endif
        // gemm0(gate)
        constexpr auto gemm_0   = Policy::template GetBlockGemm0<Problem>();
        using SaccBlockTileType = decltype(gemm_0.MakeCBlockTile());
        auto s_acc              = SaccBlockTileType{};

        auto a_dram_block = load_tile(a_global_to_dram_window);
        auto g_dram_block = load_tile(g_global_to_dram_window);
        // block_sync_load_raw();
        // save tokens to lds
        store_tile(a_lds_win, a_dram_block);
        store_tile(g_lds_win, g_dram_block);

#if 0
        PrintMem(a_dram_block,"A", 0);
#endif
        clear_tile(s_acc); // initialize C
        constexpr index_t kK0  = BlockShape::Block_K0;
        const index_t k0_loops = ck_tile::integer_divide_ceil(hidden_size, kK0);
        index_t iCounter0      = k0_loops - 1;
        while(iCounter0 >= 0)
        {
            if(iCounter0 > 0)
            {
                move_tile_window(a_global_to_dram_window, {0, kK0});
                move_tile_window(g_global_to_dram_window, {0, kK0});

                a_dram_block = load_tile(a_global_to_dram_window);
                g_dram_block = load_tile(g_global_to_dram_window);
            }

            block_sync_lds();
            gemm_0(s_acc, a_lds_win, g_lds_win);
            // gemm_0(s_acc, a_lds_win, g_dram_block);

            block_sync_lds();

            if(iCounter0 > 0)
            {
                store_tile(a_lds_win, a_dram_block);
                store_tile(g_lds_win, g_dram_block);
            }

            iCounter0--;
        }
        // tail
        // {
        //     block_sync_lds();
        //     // gemm_0(s_acc, a_lds_win, g_dram_block);
        //     gemm_0(s_acc, a_lds_win, g_lds_win);
        //     block_sync_lds();
        // }

#if 0
        PrintMem(s_acc, "S", 0);
#endif
        // relu
        const auto activation = ck_tile::element_wise::Gelu{};
        tile_elementwise_inout(activation, s_acc, s_acc);
        // cast data to YDataType
        auto y_pre = cast_tile<YDataType>(s_acc);

#if 0
        PrintMem(y_pre, "Y_pre", 0);
#endif
        // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
        // {
        //     block_sync_lds();
        //     store_tile(c_window_, y_pre);
        // }
        // save to lds
        CK_TILE_LDS_ADDR ADataType* smem_y = reinterpret_cast<CK_TILE_LDS_ADDR YDataType*>(smem);
        auto bridge_lds_view               = make_tensor_view<address_space_enum::lds>(
            smem_y, Policy::template MakeBridgeLdsBlockDesc<Problem>());
        auto bridge_slds_win =
            make_tile_window(bridge_lds_view,
                             Policy::template MakeBridgeLdsBlockDesc<Problem>().get_lengths(),
                             {0, 0});
        store_tile(bridge_slds_win, y_pre);
        block_sync_lds();

        // gemm down
        constexpr auto gemm_1   = Policy::template GetBlockGemm1<Problem>();
        using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());
        auto o_acc              = OaccBlockTileType{};

        constexpr auto w_dstr =
            make_static_tile_distribution(detail::make_reduce_tile_distribution_encoding(
                s_acc.get_tile_distribution().get_static_tile_distribution_encoding(),
                sequence<1>{}));
        auto w_global_to_dram_window = make_tile_window(w_window_.get_bottom_tensor_view(),
                                                        make_tuple(number<BlockShape::Block_M0>{}),
                                                        w_window_.get_window_origin(),
                                                        w_dstr);
        auto w                       = load_tile(w_global_to_dram_window);
        float weight                 = type_convert<float>(w.get_thread_buffer()[0]);
#if 0
        constexpr index_t w_buffer_size = decltype(w)::get_thread_buffer_size();
        if(threadIdx.x == 1 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
        {
            for(int i = 0; i < w_buffer_size; i++)
            {
                printf("\n len: %d, w[%d]: %f weight: %f", w_buffer_size, i, type_convert<float>(w.get_thread_buffer()[i]), topk_weight);
            }
        }
#endif
        // y data
        auto bridge_llds_win =
            make_tile_window(bridge_lds_view,
                             Policy::template MakeBridgeLdsBlockDesc<Problem>().get_lengths(),
                             {0, 0},
                             Policy::template MakeYTileDistribution<Problem>());
        auto y = load_tile(bridge_llds_win);
        block_sync_lds();

#if 0
        PrintMem(y,"Y",0);
        //PrintMem(y,"Y",32);
        if(threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
        {
            for(int i = 0; i < 16; i++)
            {
                printf("\n smem_a[%d]: %f ", i, type_convert<float>(smem_a[i]));
            }
        }
        //store_tile(c_window_, y);
#endif
        // d data
        auto d_global_to_dram_window = make_tile_window(
            d_window_.get_bottom_tensor_view(),
            make_tuple(number<BlockShape::Block_N0>{}, number<BlockShape::Block_K0>{}),
            d_window_.get_window_origin(),
            Policy::template MakeGlobalTileDistribution_D<Problem>());
        auto d = load_tile(d_global_to_dram_window);
#if 0
        PrintMem(d,"D",0);
#endif
        // add to LDS
        CK_TILE_LDS_ADDR float* smem_o = reinterpret_cast<CK_TILE_LDS_ADDR float*>(smem);
        auto o_lds_view                = make_tensor_view<address_space_enum::lds>(
            smem_o, Policy::template MakeLdsBlockDesc_O<Problem>());
        auto o_alds_win = make_tile_window(
            o_lds_view,
            make_tuple(number<BlockShape::Block_K1>{}, number<BlockShape::Block_N1>{}),
            {0, 0});
        auto o_olds_win = make_tile_window(
            o_lds_view,
            make_tuple(number<BlockShape::Block_M1>{}, number<BlockShape::Block_N1>{}),
            {0, 0},
            Policy::template MakeGlobalTileDistribution_O<Problem>());

        auto save_o = [&]() {
            // if(blockIdx.x == 0 && (blockIdx.y == 0 || blockIdx.y == 1) && blockIdx.z == 0)
            {
                if(threadIdx.x < 64)
                {
                    auto o0                              = load_tile(o_olds_win);
                    constexpr index_t thread_buffer_size = decltype(o0)::get_thread_buffer_size();
                    static_for<1, BlockShape::Repeat_K1, 1>{}([&](auto) {
                        move_tile_window(o_olds_win, {32, 0});
                        auto o1 = load_tile(o_olds_win);
                        static_for<0, thread_buffer_size, 1>{}([&](auto i) {
                            o0.get_thread_buffer()(i) =
                                type_convert<float>(type_convert<float>(o0.get_thread_buffer()[i]) +
                                                    type_convert<float>(o1.get_thread_buffer()[i]));
                        });
                    });
                    auto o = cast_tile<ODataType>(o0);
                    update_tile(o_window_, o);
                    // restore pos
                    move_tile_window(o_olds_win,
                                     {-BlockShape::Block_M1 * (BlockShape::Repeat_K1 - 1), 0});
                }
            }
        };
        constexpr index_t kN1  = BlockShape::Block_N1;
        const index_t n1_loops = ck_tile::integer_divide_ceil(hidden_size, kN1);
        index_t iCounter1      = n1_loops - 1;
        while(iCounter1 > 0)
        {
            clear_tile(o_acc);
            block_sync_lds_direct_load();
            gemm_1(o_acc, y, d);

            move_tile_window(d_global_to_dram_window, {kN1, 0});
            d = load_tile(d_global_to_dram_window);

            // move out window and save data
            tile_elementwise_inout([&weight](auto& x) { x = x * type_convert<float>(weight); },
                                   o_acc);
            // auto o = cast_tile<ODataType>(o_acc);
            store_tile(o_alds_win, o_acc);
            block_sync_lds();
            save_o();

            move_tile_window(o_window_, {0, kN1});

            iCounter1--;
        }
        // tail
        {
            clear_tile(o_acc);
            block_sync_lds_direct_load();
            gemm_1(o_acc, y, d);

            // block_sync_lds();
            tile_elementwise_inout([&weight](auto& x) { x = x * type_convert<float>(weight); },
                                   o_acc);
            // auto o = cast_tile<ODataType>(o_acc);
            store_tile(o_alds_win, o_acc);
            block_sync_lds();
            save_o();
            // store_tile(o_window_, o);
#if 0
        PrintMem(o,"O");
#endif
        }
        // store_tile(o_window_, a_dram_block);
    }
};

} // namespace ck_tile
