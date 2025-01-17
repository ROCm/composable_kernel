// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp"

namespace ck_tile {

template <typename Problem, typename Policy = GemmPipelineAGmemBGmemCRegV1DefaultPolicy>
struct GemmPipelineAgBgCrCompV4 : public BaseGemmPipelineAgBgCrCompV3<Problem>
{
    using Base             = BaseGemmPipelineAgBgCrCompV3<Problem>;
    using PipelineImplBase = GemmPipelineAgBgCrImplBase<Problem, Policy>;

    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

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

    static constexpr index_t VectorSizeA = Problem::VectorSizeA;
    static constexpr index_t VectorSizeB = Problem::VectorSizeB;
    static constexpr index_t VectorSizeC = Problem::VectorSizeC;

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr bool isDoubleSmemBuffer = Problem::isDoubleSmemBuffer;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    CK_TILE_HOST_DEVICE static constexpr auto IsTransposeC() { return Policy::IsTransposeC(); }

    template <GemmPipelineScheduler Scheduler>
    struct PipelineImpl : public PipelineImplBase
    {
    };

    template <>
    struct PipelineImpl<GemmPipelineScheduler::Intrawave> : public PipelineImplBase
    {
        CK_TILE_DEVICE static constexpr auto HotLoopScheduler()
        {
            constexpr index_t MPerXDL = BlockGemmShape::WarpTile::at(I0{});
            constexpr index_t NPerXDL = BlockGemmShape::WarpTile::at(I1{});
            constexpr index_t KPerXDL = BlockGemmShape::WarpTile::at(I2{});

            constexpr index_t WaveSize = 64;
            constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(I0{});
            constexpr index_t WaveNumN = BlockGemmShape::BlockWarps::at(I1{});

            constexpr index_t A_LDS_Read_Width = KPerXDL;
            constexpr index_t B_LDS_Read_Width = KPerXDL;

            constexpr index_t A_Buffer_Load_Inst_Num =
                MPerBlock * KPerBlock / (BlockSize * VectorSizeA);
            constexpr index_t B_Buffer_Load_Inst_Num =
                NPerBlock * KPerBlock / (BlockSize * VectorSizeB);

            constexpr index_t A_LDS_Write_Inst_Num = MPerBlock * KPerBlock / (BlockSize * KPerXDL);
            constexpr index_t B_LDS_Write_Inst_Num = NPerBlock * KPerBlock / (BlockSize * KPerXDL);

            constexpr index_t A_LDS_Read_Inst_Num =
                WaveNumN * MPerBlock * KPerBlock / (BlockSize * KPerXDL);
            constexpr index_t B_LDS_Read_Inst_Num =
                WaveNumM * MPerBlock * KPerBlock / (BlockSize * KPerXDL);

            constexpr index_t C_MFMA_Inst_Num = MPerBlock * NPerBlock * KPerBlock /
                                                (BlockSize / WaveSize) /
                                                (MPerXDL * NPerXDL * KPerXDL);

            constexpr auto num_ds_read_inst_a = A_LDS_Read_Width * sizeof(ADataType) == 16
                                                    ? A_LDS_Read_Inst_Num
                                                    : A_LDS_Read_Inst_Num / 2;
            constexpr auto num_ds_read_inst_b = B_LDS_Read_Width * sizeof(BDataType) == 16
                                                    ? B_LDS_Read_Inst_Num
                                                    : B_LDS_Read_Inst_Num / 2;

            constexpr auto num_ds_read_inst = num_ds_read_inst_a + num_ds_read_inst_b;

            constexpr auto num_ds_write_inst = A_LDS_Write_Inst_Num + B_LDS_Write_Inst_Num;

            constexpr auto num_buffer_load_inst = A_Buffer_Load_Inst_Num + B_Buffer_Load_Inst_Num;

            static_for<0, num_buffer_load_inst, 1>{}([&](auto i) {
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA : 1
                __builtin_amdgcn_sched_group_barrier(
                    0x100, num_ds_read_inst / num_buffer_load_inst, 0); // DS read : 2
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);      // MFMA: 1
                __builtin_amdgcn_sched_group_barrier(
                    0x200, num_ds_write_inst / num_buffer_load_inst, 0); // DS write : 1
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);       // MFMA : 1
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0);       // VMEM read :1
                __builtin_amdgcn_sched_group_barrier(
                    0x008, C_MFMA_Inst_Num / num_buffer_load_inst - 3, 0); // MFMA : 5
            });
            __builtin_amdgcn_sched_barrier(0);
        }

        template <typename ADramBlockWindowTmp,
                  typename BDramBlockWindowTmp,
                  typename AElementFunction,
                  typename BElementFunction>
        CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                       const AElementFunction& a_element_func,
                                       const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                       const BElementFunction& b_element_func,
                                       index_t num_loop,
                                       void* __restrict__ p_smem_0,
                                       void* __restrict__ p_smem_1)
        {
            static_assert(
                std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BDataType,
                                   remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
                "wrong!");

            static_assert(MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                              NPerBlock ==
                                  BDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                              KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                          "wrong!");

            ////////////// global window & register /////////////////
            // A DRAM tile window for load
            auto a_copy_dram_window =
                make_tile_window_linear(a_dram_block_window_tmp.get_bottom_tensor_view(),
                                        make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                                        a_dram_block_window_tmp.get_window_origin(),
                                        Policy::template MakeADramTileDistribution<Problem>());

            // B DRAM tile window for load
            auto b_copy_dram_window =
                make_tile_window_linear(b_dram_block_window_tmp.get_bottom_tensor_view(),
                                        make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                                        b_dram_block_window_tmp.get_window_origin(),
                                        Policy::template MakeBDramTileDistribution<Problem>());

            // A register tile for global load
            constexpr auto ABlockTileDistr = a_copy_dram_window.get_tile_distribution();
            constexpr auto BBlockTileDistr = b_copy_dram_window.get_tile_distribution();
            using ABlockTile = decltype(make_static_distributed_tensor<ADataType>(ABlockTileDistr));
            using BBlockTile = decltype(make_static_distributed_tensor<BDataType>(BBlockTileDistr));
            ABlockTile a_global_load_tile;
            BBlockTile b_global_load_tile;

            // global prefetch 0
            // global read 0
            Base::GlobalPrefetch(a_global_load_tile, a_copy_dram_window);
            Base::GlobalPrefetch(b_global_load_tile, b_copy_dram_window);
            ////////////// LDS desc, window & register /////////////////
            auto&& [a_lds_block0, b_lds_block0] = Base::GetABLdsTensorViews(p_smem_0);
            auto&& [a_lds_block1, b_lds_block1] = Base::GetABLdsTensorViews(p_smem_1);

            auto a_copy_lds_window0 =
                make_tile_window(a_lds_block0,
                                 make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                                 {0, 0},
                                 ABlockTileDistr);

            auto a_copy_lds_window1 =
                make_tile_window(a_lds_block1,
                                 make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                                 {0, 0},
                                 ABlockTileDistr);

            auto b_copy_lds_window0 =
                make_tile_window(b_lds_block0,
                                 make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
                                 {0, 0},
                                 BBlockTileDistr);

            auto b_copy_lds_window1 =
                make_tile_window(b_lds_block1,
                                 make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
                                 {0, 0},
                                 BBlockTileDistr);

            // Block GEMM
            auto block_gemm   = BlockGemm();
            auto c_block_tile = block_gemm.MakeCBlockTile();

            // initialize C
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            // LDS write 0
            Base::LocalPrefill(a_copy_lds_window0, a_global_load_tile, a_element_func);
            Base::LocalPrefill(b_copy_lds_window0, b_global_load_tile, b_element_func);
            // global read 1
            Base::GlobalPrefetch(a_global_load_tile, a_copy_dram_window);
            Base::GlobalPrefetch(b_global_load_tile, b_copy_dram_window);

            block_sync_lds();
            block_gemm.LocalPrefetch();

            
        }
    };
};
} // namespace ck_tile
