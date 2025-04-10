// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "block_gemm_pipeline_agmem_bgmem_creg_default_policy.hpp"

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"

namespace ck_tile {

//  A Tile Window: global memory
//  B Tile Window: global memory
//  C Distributed tensor: register
template <typename Problem, typename Policy = ck_tile::BlockGemmPipelineAGmemBGmemCRegDefaultPolicy>
struct BlockGemmPipelineAGmemBGmemCReg
{
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kMPerBlock = BlockGemmShape::kM;
    static constexpr index_t kNPerBlock = BlockGemmShape::kN;
    static constexpr index_t kKPerBlock = BlockGemmShape::kK;

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
    using PipelineImplBase = GemmPipelineAgBgCrImplBase<Problem, Policy>;

    static constexpr index_t APackedSize =
        ck_tile::numeric_traits<remove_cvref_t<ADataType>>::PackedSize;
    static constexpr index_t BPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<BDataType>>::PackedSize;

    using ALayout = remove_cvref_t<typename Problem::ALayout>;
    using BLayout = remove_cvref_t<typename Problem::BLayout>;
    using CLayout = remove_cvref_t<typename Problem::CLayout>;

    using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;
    using I0        = number<0>;
    using I1        = number<1>;
    using I2        = number<2>;

    static constexpr index_t BlockSize = Problem::kBlockSize;
    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    static constexpr index_t GetVectorSizeA() { return Policy::template GetVectorSizeA<Problem>(); }
    static constexpr index_t GetVectorSizeB() { return Policy::template GetVectorSizeB<Problem>(); }

    static constexpr index_t GetSmemPackA() { return Policy::template GetSmemPackA<Problem>(); }
    static constexpr index_t GetSmemPackB() { return Policy::template GetSmemPackB<Problem>(); }

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr bool DoubleSmemBuffer = Problem::DoubleSmemBuffer;

    static constexpr bool HasHotLoop = Problem::HasHotLoop;
    static constexpr auto TailNum    = Problem::TailNum;
    static constexpr auto Scheduler  = Problem::Scheduler;

    template <GemmPipelineScheduler Scheduler>
    struct PipelineImpl : public PipelineImplBase
    {
    };

    template <>
    struct PipelineImpl<GemmPipelineScheduler::Intrawave> : public PipelineImplBase
    {
        using Base = PipelineImplBase;

        CK_TILE_DEVICE static constexpr auto HotLoopScheduler()
        {
            constexpr index_t MPerXDL = BlockGemm::WarpGemm::kM;
            constexpr index_t NPerXDL = BlockGemm::WarpGemm::kN;
            constexpr index_t KPerXDL = BlockGemm::WarpGemm::WarpGemmAttribute::Impl::kK;

            constexpr index_t WaveSize = 64;
            constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(I0{});
            constexpr index_t WaveNumN = BlockGemmShape::BlockWarps::at(I1{});

            // Below should be equal to AK1|BK1
            constexpr index_t A_LDS_Read_Width = GetSmemPackA();
            constexpr index_t B_LDS_Read_Width = GetSmemPackB();

            constexpr index_t A_LDS_Write_Width = GetSmemPackA();
            constexpr index_t B_LDS_Write_Width = GetSmemPackB();

            constexpr index_t A_Buffer_Load_Inst_Num =
                MPerBlock * KPerBlock / (BlockSize * GetVectorSizeA());
            constexpr index_t B_Buffer_Load_Inst_Num =
                NPerBlock * KPerBlock / (BlockSize * GetVectorSizeB());

            constexpr index_t A_LDS_Write_Inst_Num =
                MPerBlock * KPerBlock / (BlockSize * A_LDS_Write_Width);
            constexpr index_t B_LDS_Write_Inst_Num =
                NPerBlock * KPerBlock / (BlockSize * B_LDS_Write_Width);

            constexpr index_t A_LDS_Read_Inst_Num =
                WaveNumN * MPerBlock * KPerBlock / (BlockSize * A_LDS_Read_Width);
            constexpr index_t B_LDS_Read_Inst_Num =
                WaveNumM * NPerBlock * KPerBlock / (BlockSize * B_LDS_Read_Width);

            constexpr index_t C_MFMA_Inst_Num = MPerBlock * NPerBlock * KPerBlock /
                                                (BlockSize / WaveSize) /
                                                (MPerXDL * NPerXDL * KPerXDL);

            // A/B split schedule
            // compiler is likely to use ds_read2 when instruction width smaller than 16bytes
            constexpr auto num_ds_read_inst_a =
                A_LDS_Read_Width * sizeof(ADataType) / APackedSize == 16 ? A_LDS_Read_Inst_Num :
                                                                           A_LDS_Read_Inst_Num / 2;
            constexpr auto num_ds_read_inst_b =
                B_LDS_Read_Width * sizeof(BDataType) / BPackedSize == 16 ? B_LDS_Read_Inst_Num :
                                                                           B_LDS_Read_Inst_Num / 2;

            constexpr auto num_ds_write_inst_a = A_LDS_Write_Inst_Num;
            constexpr auto num_ds_write_inst_b = B_LDS_Write_Inst_Num;

            constexpr auto num_buffer_load_inst_a = A_Buffer_Load_Inst_Num;
            constexpr auto num_buffer_load_inst_b = B_Buffer_Load_Inst_Num;

            constexpr auto num_mfma_inst = C_MFMA_Inst_Num;

            constexpr auto mfma_cycle = NPerXDL == 16 ? 16 : 32;
            constexpr auto ds_read_a_issue_cycle =
                A_LDS_Read_Width * sizeof(ADataType) / APackedSize == 16 ? 8 : 4;
            constexpr auto ds_read_b_issue_cycle =
                B_LDS_Read_Width * sizeof(BDataType) / BPackedSize == 16 ? 8 : 4;
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
            // constexpr auto num_mfma_per_ds_read = sizeof(ComputeDataType) / sizeof(ADataType) >
            //                                               sizeof(ComputeDataType) /
            //                                               sizeof(BDataType)
            //                                           ? sizeof(ComputeDataType) /
            //                                           sizeof(ADataType) : sizeof(ComputeDataType)
            //                                           / sizeof(BDataType);
            constexpr auto num_mfma_stage1 =
                num_mfma_inst - (num_dsread_a_mfma + num_dsread_b_mfma);
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
                    __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                    __builtin_amdgcn_sched_group_barrier(0x008, num_mfma_per_dswrite_a, 0); // MFMA
                });
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(0x008,
                    num_mfma_per_issue - num_mfma_per_dswrite_a * num_dswrite_per_issue_a, 0); // MFMA
            });
            static_for<0, num_buffer_load_inst_b, 1>{}([&](auto i) {
                ignore = i;
                static_for<0, num_dswrite_per_issue_b, 1>{}([&](auto idswrite) {
                    ignore = idswrite;
                    __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                    __builtin_amdgcn_sched_group_barrier(0x008, num_mfma_per_dswrite_b, 0); // MFMA
                });
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(0x008,
                    num_mfma_per_issue - num_mfma_per_dswrite_b * num_dswrite_per_issue_b, 0); // MFMA
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
                    __builtin_amdgcn_sched_group_barrier(
                        0x100,
                        num_ds_read_inst_a - (num_dsread_a_mfma - 1) * ds_read_a_mfma_rate,
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
                    __builtin_amdgcn_sched_group_barrier(
                        0x100,
                        num_ds_read_inst_b - (num_dsread_b_mfma - 1) * ds_read_b_mfma_rate,
                        0); // DS read
                }
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });
        }

        template <bool HasHotLoop,
                    TailNumber TailNum,
                    typename ADramBlockWindowTmp,
                    typename BDramBlockWindowTmp,
                    typename AElementFunction,
                    typename BElementFunction>
        CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                        const AElementFunction& a_element_func,
                                        const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                        const BElementFunction& b_element_func,
                                        index_t num_loop,
                                        void* p_smem) const
        {
            static_assert(
                std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BDataType,
                                    remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
                "A/B Dram block window should have the same data type as appropriate "
                "([A|B]DataType) defined in Problem definition!");

            // ------------------------------------------------------------------------------------
            // Definitions of all needed tiles

            // A/B tiles in LDS
            auto&& [a_lds_block, b_lds_block] = Base::GetABLdsTensorViews(p_smem);

            // Tile distribution for load from lds
            constexpr auto a_lds_load_tile_distr =
                make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
            constexpr auto b_lds_load_tile_distr =
                make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());

            // A DRAM tile window for load
            // A LDS tile window for store
            // A LDS tile for block GEMM
            auto&& [a_copy_dram_window, a_copy_lds_window, a_lds_gemm_window] =
                Base::GetAWindows(a_dram_block_window_tmp, a_lds_block, a_lds_load_tile_distr);

            // B DRAM tile window for load
            // B LDS tile window for store
            // B LDS tile for block GEMM
            auto&& [b_copy_dram_window, b_copy_lds_window, b_lds_gemm_window] =
                Base::GetBWindows(b_dram_block_window_tmp, b_lds_block, b_lds_load_tile_distr);

            // Block GEMM
            auto block_gemm   = BlockGemm();
            auto c_block_tile = block_gemm.MakeCBlockTile();

            using ABlockTileDistr = decltype(a_copy_dram_window.get_tile_distribution());
            using BBlockTileDistr = decltype(b_copy_dram_window.get_tile_distribution());

            using ABlockTile =
                decltype(make_static_distributed_tensor<ADataType>(ABlockTileDistr{}));
            using BBlockTile =
                decltype(make_static_distributed_tensor<BDataType>(BBlockTileDistr{}));

            ABlockTile a_block_tile;
            BBlockTile b_block_tile;

            using ADramTileWindowStep = typename ADramBlockWindowTmp::BottomTensorIndex;
            using BDramTileWindowStep = typename BDramBlockWindowTmp::BottomTensorIndex;

            constexpr ADramTileWindowStep a_dram_tile_window_step = make_array(0, KPerBlock);
            constexpr BDramTileWindowStep b_dram_tile_window_step = make_array(0, KPerBlock);

            // -----------------------------------------------------------------------------------------
            // Gemm pipeline start

            // prefetch
            // global read 0
            Base::GlobalPrefetch(a_block_tile, a_copy_dram_window, a_dram_tile_window_step);
            Base::GlobalPrefetch(b_block_tile, b_copy_dram_window, b_dram_tile_window_step);

            // initialize C
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            // LDS write 0
            Base::LocalPrefill(a_copy_lds_window, a_block_tile, a_element_func);
            Base::LocalPrefill(b_copy_lds_window, b_block_tile, b_element_func);

            Base::GlobalPrefetch(a_block_tile, a_copy_dram_window, a_dram_tile_window_step);
            Base::GlobalPrefetch(b_block_tile, b_copy_dram_window, b_dram_tile_window_step);

            block_sync_lds();
            block_gemm.LocalPrefetch(a_lds_gemm_window, b_lds_gemm_window);

            __builtin_amdgcn_sched_barrier(0);

            // main body
            if constexpr(HasHotLoop)
            {
                index_t i = 0;
                do
                {
                    block_sync_lds();
                    Base::LocalPrefill(a_copy_lds_window, a_block_tile, a_element_func);
                    Base::LocalPrefill(b_copy_lds_window, b_block_tile, b_element_func);

                    Base::GlobalPrefetch(a_block_tile, a_copy_dram_window, a_dram_tile_window_step);
                    Base::GlobalPrefetch(b_block_tile, b_copy_dram_window, b_dram_tile_window_step);

                    block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);

                    block_sync_lds();

                    block_gemm.LocalPrefetch(a_lds_gemm_window, b_lds_gemm_window);
                    HotLoopScheduler();
                    __builtin_amdgcn_sched_barrier(0);

                    i += 1;
                } while(i < (num_loop - 1));
            }
            // tail
            if constexpr((TailNum == TailNumber::Full) || (TailNum == TailNumber::Odd))
            {
                // Leak last MFMA block to epilogue region, cover the potential lds-shuffle
                // latency
                block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
            }
            else
            {
                block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
                block_sync_lds();
                Base::LocalPrefill(a_copy_lds_window, a_block_tile, a_element_func);
                Base::LocalPrefill(b_copy_lds_window, b_block_tile, b_element_func);
                block_sync_lds();
                block_gemm.LocalPrefetch(a_lds_gemm_window, b_lds_gemm_window);
                block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
            }
            // __builtin_amdgcn_sched_barrier(0);
            return c_block_tile;
        }
    };

    template <typename ADramBlockWindowTmp,
                typename BDramBlockWindowTmp,
                typename AElementFunction,
                typename BElementFunction>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                    const AElementFunction& a_element_func,
                                    const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                    const BElementFunction& b_element_func,
                                    index_t num_loop,
                                    void* p_smem) const
    {
        return PipelineImpl<Scheduler>{}.template operator()<HasHotLoop, TailNum>(
            a_dram_block_window_tmp,
            a_element_func,
            b_dram_block_window_tmp,
            b_element_func,
            num_loop,
            p_smem);
    }

    template <typename ADramBlockWindowTmp, typename BDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                    const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                    index_t num_loop,
                                    void* p_smem) const
    {
        return PipelineImpl<Scheduler>{}.template operator()<HasHotLoop, TailNum>(
            a_dram_block_window_tmp,
            [](const ADataType& a) { return a; },
            b_dram_block_window_tmp,
            [](const BDataType& b) { return b; },
            num_loop,
            p_smem);
    }

#else

    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp>
    CK_TILE_HOST_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                        const BDramBlockWindowTmp& b_dram_block_window_tmp,
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

        // A tile in LDS
        ADataType* p_a_lds = static_cast<ADataType*>(p_smem);

        constexpr auto a_lds_block_desc = Policy::template MakeALdsBlockDescriptor<Problem>();

        auto a_lds_block = make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_block_desc);

        constexpr index_t a_lds_block_space_size_aligned =
            integer_divide_ceil(sizeof(ADataType) * a_lds_block_desc.get_element_space_size(),
                                      16) * 16;

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

        // A LDS tile for block GEMM
        auto a_lds_gemm_window = make_tile_window(
            a_lds_block, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {0, 0});

        // B LDS tile for block GEMM
        auto b_lds_gemm_window = make_tile_window(
            b_lds_block, make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}), {0, 0});

        // Block GEMM
        constexpr auto block_gemm = Policy::template GetBlockGemm<Problem>();

        // Acc register tile
        auto c_block_tile = decltype(block_gemm(a_lds_gemm_window, b_lds_gemm_window)){};

#if defined(ENABLE_PREFETCH)
        // prefetch
        // global read 0
        auto a_block_tile = load_tile(a_copy_dram_window);
        auto b_block_tile = load_tile(b_copy_dram_window);

        {
            // move to 1
            move_tile_window(a_copy_dram_window, {0, kKPerBlock});
            move_tile_window(b_copy_dram_window, {0, kKPerBlock});

            // Initialize C
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            // LDS write 0
            store_tile(a_copy_lds_window, a_block_tile);
            // global read 1
            a_block_tile = load_tile(a_copy_dram_window);

            // LDS write 0
            store_tile(b_copy_lds_window, b_block_tile);
            // global read 1
            b_block_tile = load_tile(b_copy_dram_window);
        }

        index_t iCounter = num_loop - 2;

        do
        {
            block_sync_lds();

            // GEMM i
            block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);

            block_sync_lds();

            // move to i + 2
            move_tile_window(a_copy_dram_window, {0, kKPerBlock});
            move_tile_window(b_copy_dram_window, {0, kKPerBlock});

            // LDS write i + 1
            store_tile(a_copy_lds_window, a_block_tile);
            // global read i + 2
            a_block_tile = load_tile(a_copy_dram_window);

            // LDS write i + 1
            store_tile(b_copy_lds_window, b_block_tile);
            // global read i + 2
            b_block_tile = load_tile(b_copy_dram_window);

            iCounter--;

        } while(iCounter > 0);

        // tail
        {
            block_sync_lds();

            // GEMM num_loop - 2
            block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);

            block_sync_lds();

            // LDS write num_loop - 1
            store_tile(a_copy_lds_window, a_block_tile);

            store_tile(b_copy_lds_window, b_block_tile);

            block_sync_lds();

            // GEMM num_loop - 1
            block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
        }
#else
        // non-prefetch
        auto a_block_tile = load_tile(a_copy_dram_window);
        auto b_block_tile = load_tile(b_copy_dram_window);
        store_tile(a_copy_lds_window, a_block_tile);
        store_tile(b_copy_lds_window, b_block_tile);

        block_sync_lds();
        block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
        block_sync_lds();

        index_t iCounter = num_loop - 1;

        while (iCounter > 0)
        {
            move_tile_window(a_copy_dram_window, {0, kKPerBlock});
            move_tile_window(b_copy_dram_window, {0, kKPerBlock});

            a_block_tile = load_tile(a_copy_dram_window);
            b_block_tile = load_tile(b_copy_dram_window);
            store_tile(a_copy_lds_window, a_block_tile);
            store_tile(b_copy_lds_window, b_block_tile);

            block_sync_lds();
            block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
            block_sync_lds();

            iCounter--;
        }
#endif
        return c_block_tile;
    }

#endif

};

} // namespace ck_tile
