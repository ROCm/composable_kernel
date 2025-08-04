// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "block_gemm_pipeline_agmem_bgmem_creg_default_policy.hpp"

#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/reduce.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"

namespace ck_tile {

//  A Tile Window: global memory
//  B Tile Window: global memory
//  C Distributed tensor: register
template <typename Problem, typename Policy = ck_tile::BlockGemmPipelineAGmemBGmemCRegDefaultPolicy>
struct BlockGemmSoftmaxTopkPipelineAGmemBGmemCReg
{
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using WeightType     = remove_cvref_t<typename Problem::WeightType>;
    using IndexType      = remove_cvref_t<typename Problem::IndexType>;
    using ComputeDataType= float;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    static constexpr index_t kBlockSize = Problem::kBlockSize;
    static constexpr index_t topk = Problem::topk;

    static constexpr index_t kMPerBlock = BlockGemmShape::kM;
    static constexpr index_t kNPerBlock = BlockGemmShape::kN;
    static constexpr index_t kKPerBlock = BlockGemmShape::kK;

    using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;

    struct ArgmaxPacket
    {
        WeightType arg;
        IndexType value;
    };

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetStaticLdsSize()
    {
        return integer_divide_ceil(
                   sizeof(ADataType) *
                       Policy::template MakeALdsBlockDescriptor<Problem>().get_element_space_size(),
                   16) *
                   16 +
               sizeof(BDataType) *
                   Policy::template MakeBLdsBlockDescriptor<Problem>().get_element_space_size();
    }

#if defined(ENABLE_INSTRUCTION_SCH)
    static constexpr index_t kPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<ADataType>>::PackedSize;

    static constexpr index_t GetVectorSizeA() { return Policy::template GetVectorSizeA<Problem>(); }
    static constexpr index_t GetVectorSizeB() { return Policy::template GetVectorSizeB<Problem>(); }

    static constexpr index_t GetSmemPack() { return Policy::template GetSmemPack<Problem>(); }

    static constexpr bool HasHotLoop = Problem::HasHotLoop;

    CK_TILE_DEVICE static constexpr auto HotLoopScheduler()
    {
        constexpr index_t MPerXDL = BlockGemm::WarpGemm::kM;
        constexpr index_t NPerXDL = BlockGemm::WarpGemm::kN;
        constexpr index_t KPerXDL = BlockGemm::WarpGemm::WarpGemmAttribute::Impl::kK;

        constexpr index_t WaveSize = 64;
        constexpr index_t WaveNumM = BlockGemm::MWarp;
        constexpr index_t WaveNumN = BlockGemm::NWarp;

        constexpr index_t AB_LDS_RW_Width = GetSmemPack();

        constexpr index_t A_Buffer_Load_Inst_Num =
            kMPerBlock * kKPerBlock / (kBlockSize * GetVectorSizeA());
        constexpr index_t B_Buffer_Load_Inst_Num =
            kNPerBlock * kKPerBlock / (kBlockSize * GetVectorSizeB());

        constexpr index_t A_LDS_Write_Inst_Num =
            kMPerBlock * kKPerBlock / (kBlockSize * AB_LDS_RW_Width);
        constexpr index_t B_LDS_Write_Inst_Num =
            kNPerBlock * kKPerBlock / (kBlockSize * AB_LDS_RW_Width);

        constexpr index_t A_LDS_Read_Inst_Num =
            WaveNumN * kMPerBlock * kKPerBlock / (kBlockSize * AB_LDS_RW_Width);
        constexpr index_t B_LDS_Read_Inst_Num =
            WaveNumM * kNPerBlock * kKPerBlock / (kBlockSize * AB_LDS_RW_Width);

        constexpr index_t C_MFMA_Inst_Num = kMPerBlock * kNPerBlock * kKPerBlock /
                                            (kBlockSize / WaveSize) / (MPerXDL * NPerXDL * KPerXDL);

        // A/B split schedule
        // compiler is likely to use ds_read2 when instruction width smaller than 16bytes
        constexpr auto num_ds_read_inst_a = AB_LDS_RW_Width * sizeof(ADataType) / kPackedSize == 16
                                                ? A_LDS_Read_Inst_Num
                                                : A_LDS_Read_Inst_Num / 2;
        constexpr auto num_ds_read_inst_b = AB_LDS_RW_Width * sizeof(BDataType) / kPackedSize == 16
                                                ? B_LDS_Read_Inst_Num
                                                : B_LDS_Read_Inst_Num / 2;

        constexpr auto num_ds_write_inst_a = A_LDS_Write_Inst_Num;
        constexpr auto num_ds_write_inst_b = B_LDS_Write_Inst_Num;

        constexpr auto num_buffer_load_inst_a = A_Buffer_Load_Inst_Num;
        constexpr auto num_buffer_load_inst_b = B_Buffer_Load_Inst_Num;

        constexpr auto num_mfma_inst = C_MFMA_Inst_Num;

        constexpr auto mfma_cycle = NPerXDL == 16 ? 16 : 32;
        constexpr auto ds_read_a_issue_cycle =
            AB_LDS_RW_Width * sizeof(ADataType) / kPackedSize == 16 ? 8 : 4;
        constexpr auto ds_read_b_issue_cycle =
            AB_LDS_RW_Width * sizeof(BDataType) / kPackedSize == 16 ? 8 : 4;
        constexpr auto ds_read_a_mfma_rate =
            (mfma_cycle - 4 + 2 * ds_read_a_issue_cycle - 1) / (2 * ds_read_a_issue_cycle);
        constexpr auto ds_read_b_mfma_rate =
            (mfma_cycle - 4 + 2 * ds_read_b_issue_cycle - 1) / (2 * ds_read_b_issue_cycle);

        constexpr auto num_dsread_a_mfma =
            (num_ds_read_inst_a + ds_read_a_mfma_rate - 1) / ds_read_a_mfma_rate;
        constexpr auto num_dsread_b_mfma =
            (num_ds_read_inst_b + ds_read_b_mfma_rate - 1) / ds_read_b_mfma_rate;

        // stage 1
        // Separate this part?
        // constexpr auto num_mfma_per_ds_read = sizeof(CDataType) / sizeof(ADataType) >
        //                                               sizeof(CDataType) /
        //                                               sizeof(BDataType)
        //                                           ? sizeof(CDataType) /
        //                                           sizeof(ADataType) : sizeof(CDataType)
        //                                           / sizeof(BDataType);
        constexpr auto num_mfma_stage1 = num_mfma_inst - (num_dsread_a_mfma + num_dsread_b_mfma);
        constexpr auto num_mfma_per_issue =
            num_mfma_stage1 / (num_buffer_load_inst_a + num_buffer_load_inst_b);
        constexpr auto num_dswrite_per_issue_a = num_ds_write_inst_a / num_buffer_load_inst_a;
        constexpr auto num_dswrite_per_issue_b = num_ds_write_inst_b / num_buffer_load_inst_b;
        constexpr auto num_mfma_per_dswrite_a =
            (num_mfma_per_issue - num_dswrite_per_issue_a * 2 >= 1) ? 2 : 1;
        constexpr auto num_mfma_per_dswrite_b =
            (num_mfma_per_issue - num_dswrite_per_issue_b * 2 >= 1) ? 2 : 1;

        static_for<0, num_buffer_load_inst_a, 1>{}([&](auto i) {
            ignore = i;
            static_for<0, num_dswrite_per_issue_a, 1>{}([&](auto idswrite) {
                ignore = idswrite;
                __builtin_amdgcn_sched_group_barrier(0x200, 1, 0);                      // DS write
                __builtin_amdgcn_sched_group_barrier(0x008, num_mfma_per_dswrite_a, 0); // MFMA
            });
            __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
            __builtin_amdgcn_sched_group_barrier(0x008,
                                                 num_mfma_per_issue - num_mfma_per_dswrite_a *
                                                                          num_dswrite_per_issue_a,
                                                 0); // MFMA
        });
        static_for<0, num_buffer_load_inst_b, 1>{}([&](auto i) {
            ignore = i;
            static_for<0, num_dswrite_per_issue_b, 1>{}([&](auto idswrite) {
                ignore = idswrite;
                __builtin_amdgcn_sched_group_barrier(0x200, 1, 0);                      // DS write
                __builtin_amdgcn_sched_group_barrier(0x008, num_mfma_per_dswrite_b, 0); // MFMA
            });
            __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
            __builtin_amdgcn_sched_group_barrier(0x008,
                                                 num_mfma_per_issue - num_mfma_per_dswrite_b *
                                                                          num_dswrite_per_issue_b,
                                                 0); // MFMA
        });

        // stage 2
        static_for<0, num_dsread_a_mfma, 1>{}([&](auto i) {
            if constexpr((num_ds_read_inst_a - (i + 1) * ds_read_a_mfma_rate) >=
                         ds_read_a_mfma_rate)
            {
                __builtin_amdgcn_sched_group_barrier(0x100, ds_read_a_mfma_rate, 0); // DS read
            }
            else
            {
                __builtin_amdgcn_sched_group_barrier(0x100,
                                                     num_ds_read_inst_a - (num_dsread_a_mfma - 1) *
                                                                              ds_read_a_mfma_rate,
                                                     0); // DS read
            }
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
        });

        static_for<0, num_dsread_b_mfma, 1>{}([&](auto i) {
            if constexpr((num_ds_read_inst_b - (i + 1) * ds_read_b_mfma_rate) >=
                         ds_read_b_mfma_rate)
            {
                __builtin_amdgcn_sched_group_barrier(0x100, ds_read_b_mfma_rate, 0); // DS read
            }
            else
            {
                __builtin_amdgcn_sched_group_barrier(0x100,
                                                     num_ds_read_inst_b - (num_dsread_b_mfma - 1) *
                                                                              ds_read_b_mfma_rate,
                                                     0); // DS read
            }
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
        });
    }
#endif

    template <typename ADramBlockWindowTmp, typename BDramBlockWindowTmp, typename ValueWindow, typename IndexWindow>
    CK_TILE_HOST_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                        const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                        const ValueWindow& value_dram_block_window,
                                        const IndexWindow& index_dram_block_window,
                                        index_t num_loop,
                                        void* p_smem) const
    {
        static_assert(
            std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                std::is_same_v<BDataType, remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kMPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kNPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kKPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        // -----------------------------------------------------------------------------------------
        // Definitions of all needed tiles

        // A tile in LDS
        ADataType* p_a_lds = static_cast<ADataType*>(p_smem);

        constexpr auto a_lds_block_desc = Policy::template MakeALdsBlockDescriptor<Problem>();

        auto a_lds_block = make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_block_desc);

        constexpr index_t a_lds_block_space_size_aligned =
            integer_divide_ceil(sizeof(ADataType) * a_lds_block_desc.get_element_space_size(), 16) *
            16;

        // B tile in LDS
        BDataType* p_b_lds = static_cast<BDataType*>(
            static_cast<void*>(static_cast<char*>(p_smem) + a_lds_block_space_size_aligned));

        constexpr auto b_lds_block_desc = Policy::template MakeBLdsBlockDescriptor<Problem>();

        auto b_lds_block = make_tensor_view<address_space_enum::lds>(p_b_lds, b_lds_block_desc);

        // A DRAM tile window for load
        auto a_copy_dram_window =
            make_tile_window(a_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                             a_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeADramTileDistribution<Problem>());

        // A LDS tile window for store
        auto a_copy_lds_window =
            make_tile_window(a_lds_block,
                             make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                             {0, 0},
                             a_copy_dram_window.get_tile_distribution());

        // B DRAM tile window for load
        auto b_copy_dram_window =
            make_tile_window(b_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
                             b_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeBDramTileDistribution<Problem>());

        // B LDS tile window for store
        auto b_copy_lds_window =
            make_tile_window(b_lds_block,
                             make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
                             {0, 0},
                             b_copy_dram_window.get_tile_distribution());

#if defined(ENABLE_PREFETCH)
        // A LDS tile for block GEMM
        auto a_lds_gemm_window = make_tile_window(
            a_lds_block,
            make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
            {0, 0},
            make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode()));

        // B LDS tile for block GEMM
        auto b_lds_gemm_window = make_tile_window(
            b_lds_block,
            make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
            {0, 0},
            make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode()));
#else
        // A LDS tile for block GEMM
        auto a_lds_gemm_window = make_tile_window(
            a_lds_block, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {0, 0});

        // B LDS tile for block GEMM
        auto b_lds_gemm_window = make_tile_window(
            b_lds_block, make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}), {0, 0});
#endif

        // Block GEMM
        auto block_gemm = BlockGemm();

        // Acc register tile
        auto c_block_tile = decltype(block_gemm(a_lds_gemm_window, b_lds_gemm_window)){};

        using ABlockTileDistr = decltype(a_copy_dram_window.get_tile_distribution());
        using BBlockTileDistr = decltype(b_copy_dram_window.get_tile_distribution());

        using ABlockTile = decltype(make_static_distributed_tensor<ADataType>(ABlockTileDistr{}));
        using BBlockTile = decltype(make_static_distributed_tensor<BDataType>(BBlockTileDistr{}));

        ABlockTile a_block_tile;
        BBlockTile b_block_tile;

        using ValueBlockTileDistr = decltype(value_dram_block_window.get_tile_distribution());
        using IndexBlockTileDistr = decltype(index_dram_block_window.get_tile_distribution());

        using ValueBlockTile = decltype(make_static_distributed_tensor<WeightType>(ValueBlockTileDistr{}));
        using IndexBlockTile = decltype(make_static_distributed_tensor<IndexType>(IndexBlockTileDistr{}));

        ValueBlockTile value_block_tile;
        IndexBlockTile index_block_tile;

        using ADramTileWindowStep = typename ADramBlockWindowTmp::BottomTensorIndex;
        using BDramTileWindowStep = typename BDramBlockWindowTmp::BottomTensorIndex;
        constexpr ADramTileWindowStep a_dram_tile_window_step = make_array(0, kKPerBlock);
        constexpr BDramTileWindowStep b_dram_tile_window_step = make_array(0, kKPerBlock);

        // -------------------------------------------------------------------------------------
        // Gemm pipeline start

        // Initialize C
        tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

#if defined(ENABLE_PREFETCH)
#pragma message("global prefetch")
        // Prefetch
        // Global read 0
        a_block_tile = load_tile(a_copy_dram_window);
        b_block_tile = load_tile(b_copy_dram_window);

        if(num_loop > 1)
        {
            move_tile_window(a_copy_dram_window, a_dram_tile_window_step);
            move_tile_window(b_copy_dram_window, b_dram_tile_window_step);

            // LDS write 0
            store_tile(a_copy_lds_window, a_block_tile);
            store_tile(b_copy_lds_window, b_block_tile);

            // Global read 1
            a_block_tile = load_tile(a_copy_dram_window);
            b_block_tile = load_tile(b_copy_dram_window);
            move_tile_window(a_copy_dram_window, a_dram_tile_window_step);
            move_tile_window(b_copy_dram_window, b_dram_tile_window_step);

            block_sync_lds();

            // Prefetch from LDS to warp register in block gemm
            block_gemm.LocalPrefetch(a_lds_gemm_window, b_lds_gemm_window);
        }

        __builtin_amdgcn_sched_barrier(0);

        // Main body
        if(num_loop > 2)
        {
            index_t iCounter = 0;
            do
            {
                block_sync_lds();

                // LDS write 1
                store_tile(a_copy_lds_window, a_block_tile);
                store_tile(b_copy_lds_window, b_block_tile);

                // Global read 2
                a_block_tile = load_tile(a_copy_dram_window);
                b_block_tile = load_tile(b_copy_dram_window);
                move_tile_window(a_copy_dram_window, a_dram_tile_window_step);
                move_tile_window(b_copy_dram_window, b_dram_tile_window_step);

                block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);

                block_sync_lds();

                // Prefetch from LDS to warp register in block gemm
                block_gemm.LocalPrefetch(a_lds_gemm_window, b_lds_gemm_window);

#if defined(ENABLE_INSTRUCTION_SCH)
                HotLoopScheduler();
#endif

                __builtin_amdgcn_sched_barrier(0);

                iCounter += 1;
            } while(iCounter < (num_loop - 2));
        }

        // Tail
        if(num_loop > 1)
        {
            block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
            block_sync_lds();
        }
        store_tile(a_copy_lds_window, a_block_tile);
        store_tile(b_copy_lds_window, b_block_tile);
        block_sync_lds();
        block_gemm.LocalPrefetch(a_lds_gemm_window, b_lds_gemm_window);
        block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
#else
        // non-prefetch
        index_t iCounter = num_loop;

        while(iCounter > 0)
        {
            a_block_tile = load_tile(a_copy_dram_window);
            b_block_tile = load_tile(b_copy_dram_window);
            move_tile_window(a_copy_dram_window, a_dram_tile_window_step);
            move_tile_window(b_copy_dram_window, b_dram_tile_window_step);
            store_tile(a_copy_lds_window, a_block_tile);
            store_tile(b_copy_lds_window, b_block_tile);

            block_sync_lds();
            block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
            block_sync_lds();

            iCounter--;
        }
#endif
        // apply softmax for c_block_tile
        // reduction function for softmax
        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        // m_local = rowmax(c_block_tile)
        auto m_local = block_tile_reduce<ComputeDataType>(
            c_block_tile, sequence<1>{}, f_max, std::numeric_limits<ComputeDataType>::lowest());
        
        block_tile_reduce_sync(m_local, f_max);

        // Pcompute{j} = sum(exp(x - m_local))
        auto p_compute =
            make_static_distributed_tensor<ComputeDataType>(c_block_tile.get_tile_distribution());

        constexpr auto p_spans = decltype(p_compute)::get_distributed_spans();

        sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);

            sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);

                p_compute(i_j_idx) = exp(c_block_tile[i_j_idx] - m_local[i_idx]);
            });
        });

        // rowsum for p_compute{i, j}
        auto rowsum_p = block_tile_reduce<ComputeDataType>(
            p_compute, sequence<1>{}, f_sum, ComputeDataType{0});

        block_tile_reduce_sync(rowsum_p, f_sum);

        // softmax = p_compute{i, j} / rowsum_p
        sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);

            sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);

                p_compute(i_j_idx) = p_compute[i_j_idx] / rowsum_p[i_idx];
            });
        });

        // apply topk for softmax output
        auto x_tmp = p_compute;
        // constexpr auto dst_dist = BlockGemmPipelineAGmemBGmemCRegDefaultPolicy::MakeOutputDistribution();

        // argmax for topk
        const auto f_argmax = [](ArgmaxPacket e0, ArgmaxPacket e1) {
            return e0.arg > e1.arg ? e0 : e1;
        };

        for(index_t i_k = 0; i_k < topk; i_k++)
        {
            constexpr auto span_2d = decltype(p_compute)::get_distributed_spans();
            auto packet            = [&]() {
                auto tmp = make_static_distributed_tensor<ArgmaxPacket>(p_compute.get_tile_distribution());

                sweep_tile_span(span_2d[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(span_2d[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            tmp.get_tile_distribution(), make_tuple(idx0, idx1));
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        ArgmaxPacket t;
                        t.arg        = x_tmp(i_j_idx); // !!! we reference p_compute here
                        t.value      = tile_idx.at(number<1>{});
                        tmp(i_j_idx) = t;
                    });
                });
                return tmp;
            }();

            auto argmax_init = ArgmaxPacket{-numeric<WeightType>::infinity(), 0};
            auto r = block_tile_reduce<ArgmaxPacket>(packet, sequence<1>{}, f_argmax, argmax_init);
            block_tile_reduce_xor_sync(r, f_argmax);

            // auto value_block_tile = make_static_distributed_tensor<WeightType>(dst_dist);
            // auto index_block_tile = make_static_distributed_tensor<IndexType>(dst_dist);

            // Initialize value_block_tile and index_block_tile
            tile_elementwise_inout([](auto& value) { value = 0; }, value_block_tile);
            tile_elementwise_inout([](auto& index) { index = 0; }, index_block_tile);

            sweep_tile_span(span_2d[number<0>{}], [&](auto idx0) {
                sweep_tile_span(span_2d[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    ArgmaxPacket tmp       = r(i_j_idx);
                    value_block_tile(i_j_idx)             = tmp.arg;
                    index_block_tile(i_j_idx)             = tmp.value;
                });
            });

            // update value
            sweep_tile_span(span_2d[number<0>{}], [&](auto idx0) {
                sweep_tile_span(span_2d[number<1>{}], [&](auto idx1) {
                    const auto tile_idx = get_x_indices_from_distributed_indices(
                        p_compute.get_tile_distribution(), make_tuple(idx0, idx1));
                    auto col_id = tile_idx.at(number<1>{});

                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    x_tmp(i_j_idx) = (col_id == r(i_j_idx).value) ? -numeric<WeightType>::infinity()
                                                                    : x_tmp(i_j_idx);
                });
            });
        }
        // // cast to WeightType and apply CElementFunction
        // value_block_tile = tile_elementwise_in(
        //     [&](const auto& value) { return c_element_func(type_convert<WeightType>(value)); },
        //     value_block_tile);

        store_tile(value_dram_block_window, value_block_tile);
        store_tile(index_dram_block_window, index_block_tile);
    }
};

} // namespace ck_tile
