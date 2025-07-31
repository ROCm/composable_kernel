// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_v1_policy.hpp"

namespace ck_tile {

template <typename Problem>
struct BaseWeightPreshufflePipelineAGmemBGmemCRegV1
{
    static constexpr index_t PrefetchStages   = 1;
    static constexpr index_t PrefillStages    = 1;
    static constexpr index_t GlobalBufferNum  = 1;
    static constexpr bool UsePersistentKernel = Problem::Traits::UsePersistentKernel;

    CK_TILE_HOST_DEVICE static constexpr auto TransposeC() { return Problem::TransposeC; }

    CK_TILE_HOST_DEVICE static constexpr bool BlockHasHotloop(index_t) { return true; }

    CK_TILE_HOST_DEVICE static constexpr TailNumber GetBlockLoopTailNum(index_t)
    {
        return TailNumber::Empty;
    }

    template <typename RunFunction>
    CK_TILE_HOST_DEVICE static auto TailHandler(const RunFunction& run_func, bool, TailNumber)
    {
        return run_func(bool_constant<true>{}, integral_constant<TailNumber, TailNumber::Empty>{});
    }
};

template <typename Problem, typename PipelinePolicy = UniversalWeightPreshufflePipelineAgBgCrPolicy>
struct WeightPreshufflePipelineAGmemBGmemCRegV1
    : public BaseWeightPreshufflePipelineAGmemBGmemCRegV1<Problem>
{
    using Base           = BaseWeightPreshufflePipelineAGmemBGmemCRegV1<Problem>;
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    using ALayout = remove_cvref_t<typename Problem::ALayout>;
    using BLayout = remove_cvref_t<typename Problem::BLayout>;
    using CLayout = remove_cvref_t<typename Problem::CLayout>;

    using BlockWeightPreshuffle =
        remove_cvref_t<decltype(PipelinePolicy::template GetBlockWeightPreshuffle<Problem>())>;

    static constexpr index_t BlockSize = Problem::kBlockSize;

    static constexpr index_t kMPerBlock = BlockGemmShape::kM;
    static constexpr index_t kNPerBlock = BlockGemmShape::kN;
    static constexpr index_t kKPerBlock = BlockGemmShape::kK;

    static constexpr index_t flatKPerWarp = BlockGemmShape::flatKPerWarp;
    static constexpr index_t flatNPerWarp = BlockGemmShape::flatNPerWarp;

    static constexpr index_t GetVectorSizeA()
    {
        return PipelinePolicy::template GetVectorSizeA<Problem>();
    }
    static constexpr index_t GetVectorSizeB()
    {
        return PipelinePolicy::template GetVectorSizeB<Problem>();
    }

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr index_t kLdsAlignmentInBytes = 16;
    static constexpr index_t NumWaveGroups        = Problem::NumWaveGroups;

    static constexpr auto I0 = number<0>();
    static constexpr auto I1 = number<1>();
    static constexpr auto I2 = number<2>();

    using BlockTile  = remove_cvref_t<typename BlockGemmShape::BlockTile>;
    using BlockWarps = remove_cvref_t<typename BlockGemmShape::BlockWarps>;
    using WarpTile   = remove_cvref_t<typename BlockGemmShape::WarpTile>;

    static constexpr bool DoubleSmemBuffer = Problem::DoubleSmemBuffer;
    static constexpr bool Preshuffle       = Problem::Preshuffle;
    using Base::UsePersistentKernel;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "pipeline_AGmemBGmemCRegV1", 
                      concat('x', kMPerBlock, kNPerBlock, kKPerBlock,  BlockSize),
                      concat('x', GetVectorSizeA(), GetVectorSizeB()),
                      concat('x', kPadM, kPadN, kPadK));
        // clang-format on
    }

    CK_TILE_HOST_DEVICE static constexpr auto TransposeC() { return Problem::TransposeC; }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return PipelinePolicy::template GetSmemSize<Problem>();
    }

    CK_TILE_HOST_DEVICE static constexpr auto HotLoopScheduler()
    {
        constexpr auto config =
            BlockWeightPreshuffle::BlockPolicy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        constexpr index_t KIterPerWarp = kKPerBlock / WG::kK;
        constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WG::kN);

        constexpr index_t KPerLoad               = Problem::VectorLoadSize / sizeof(ADataType);
        constexpr index_t A_Buffer_Load_Inst_Num = kMPerBlock * kKPerBlock / BlockSize / KPerLoad;
        constexpr index_t A_LDS_Read_Inst_Num    = MIterPerWarp * KIterPerWarp;
        constexpr index_t B_Buffer_Load_Inst_Num = NIterPerWarp * KIterPerWarp;

        if constexpr(WG::kM == 16 && WG::kN == 16)
        {
            static_for<0, A_Buffer_Load_Inst_Num, 1>{}([&](auto i) {
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });
            static_for<0, A_LDS_Read_Inst_Num - A_Buffer_Load_Inst_Num, 1>{}([&](auto i) {
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                __builtin_amdgcn_sched_group_barrier(0x008, 3, 0); // MFMA
            });
            static_for<0, B_Buffer_Load_Inst_Num, 1>{}([&](auto i) {
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
            });
            static_for<0, A_Buffer_Load_Inst_Num, 1>{}([&](auto i) {
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                __builtin_amdgcn_sched_group_barrier(0x008, 4, 0); // MFMA
            });
        }
        else if constexpr(WG::kM == 32 && WG::kN == 32 &&
                          (A_LDS_Read_Inst_Num / 2 >
                           A_Buffer_Load_Inst_Num + B_Buffer_Load_Inst_Num))
        {
            static_for<0,
                       A_LDS_Read_Inst_Num / 2 - A_Buffer_Load_Inst_Num - B_Buffer_Load_Inst_Num,
                       1>{}([&](auto i) {
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });
            static_for<0, A_Buffer_Load_Inst_Num, 1>{}([&](auto i) {
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });
            static_for<0, A_LDS_Read_Inst_Num / 2, 1>{}([&](auto i) {
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });
            static_for<0, B_Buffer_Load_Inst_Num, 1>{}([&](auto i) {
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });
            static_for<0, A_Buffer_Load_Inst_Num, 1>{}([&](auto i) {
                ignore = i;
                __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                __builtin_amdgcn_sched_group_barrier(0x008, 3, 0); // MFMA
            });
            __builtin_amdgcn_sched_group_barrier(0x008, 4, 0); // MFMA
        }
    }

    template <typename ADramBlockWindowTmp, typename BFlatBlockWindowTmp, typename AElementFunction>
    CK_TILE_HOST_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                        const AElementFunction& a_element_func,
                                        const BFlatBlockWindowTmp& b_flat_dram_block_window_tmp,
                                        index_t num_loop,
                                        void* p_smem) const
    {
        static_assert(
            std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                std::is_same_v<BDataType, remove_cvref_t<typename BFlatBlockWindowTmp::DataType>>,
            "A/B Dram block window should have the same data type as appropriate "
            "([A|B]DataType) defined in Problem definition!");

        constexpr bool is_a_col_major = std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;

        static_assert(is_a_col_major
                          ? (kKPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0] &&
                             kMPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1])
                          : (kMPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0] &&
                             kKPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1]),
                      "A block window has incorrect lengths for defined ALayout!");

        constexpr auto config =
            BlockWeightPreshuffle::BlockPolicy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WG::kN);
        constexpr index_t KIterPerWarp = kKPerBlock / WG::kK;

        constexpr index_t KFlatPerBlockPerIter = flatKPerWarp;
        constexpr index_t NFlatPerBlockPerIter = flatNPerWarp;

        constexpr index_t MPerBlockPerIter = kMPerBlock / MIterPerWarp;
        constexpr index_t KPerBlockPerIter = kKPerBlock / KIterPerWarp;

        const index_t iMWarp = get_warp_id() / NWarp;

        // A tile in LDS
        ADataType* p_a_lds = static_cast<ADataType*>(p_smem);

        constexpr auto a_lds_block_desc =
            PipelinePolicy::template MakeALdsBlockDescriptor<Problem>();

        auto a_lds_block = make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_block_desc);

        // A DRAM tile window for load
        auto a_copy_dram_window =
            make_tile_window(a_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                             a_dram_block_window_tmp.get_window_origin(),
                             PipelinePolicy::template MakeADramTileDistribution<Problem>());

        // A LDS tile window for store
        auto a_copy_lds_window = make_tile_window(
            a_lds_block, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {0, 0});

        // A LDS tile for block GEMM
        auto a_lds_gemm_window = make_tile_window(
            a_lds_block, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {0, 0});

        auto a_warp_window_tmp = make_tile_window(
            a_lds_gemm_window.get_bottom_tensor_view(),
            make_tuple(number<WG::kM>{}, number<WG::kK>{}),
            a_lds_gemm_window.get_window_origin() + multi_index<2>{iMWarp * WG::kM, 0},
            make_static_tile_distribution(typename WG::AWarpDstrEncoding{}));

        statically_indexed_array<
            statically_indexed_array<decltype(a_warp_window_tmp), KIterPerWarp>,
            MIterPerWarp>
            a_warp_windows;
        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                a_warp_windows(mIter)(kIter) = a_warp_window_tmp;

                move_tile_window(a_warp_windows(mIter)(kIter),
                                 {mIter * MPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });

        // Block GEMM
        auto block_flatmm = BlockWeightPreshuffle();

        // B flat DRAM window for load
        auto b_flat_distribution =
            PipelinePolicy::template MakeBFlatDramTileDistribution<Problem>();
        auto b_flat_dram_window = // tile_window_with_static_distribution
            make_tile_window(
                b_flat_dram_block_window_tmp.get_bottom_tensor_view(), // from kernel gemm_pad_views
                make_tuple(number<flatNPerWarp>{}, number<flatKPerWarp>{}),
                b_flat_dram_block_window_tmp.get_window_origin(),
                b_flat_distribution);

        // Acc register tile
        auto c_block_tile = block_flatmm.MakeCBlockTile();

        // prefetch
        // global read 0
        auto a_block_tile = load_tile(a_copy_dram_window);

        statically_indexed_array<
            statically_indexed_array<decltype(b_flat_dram_window), KIterPerWarp>,
            NIterPerWarp>
            b_flat_dram_windows;

        statically_indexed_array<
            statically_indexed_array<decltype(load_tile(b_flat_dram_window)), KIterPerWarp>,
            NIterPerWarp>
            b_warp_tensor;

        statically_indexed_array<
            statically_indexed_array<decltype(load_tile(b_flat_dram_window)), KIterPerWarp>,
            NIterPerWarp>
            b_warp_tensor_2;

        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                b_flat_dram_windows(nIter)(kIter) = b_flat_dram_window;

                move_tile_window(b_flat_dram_windows(nIter)(kIter),
                                 {nIter * NFlatPerBlockPerIter, kIter * KFlatPerBlockPerIter});

                b_warp_tensor(nIter)(kIter) = load_tile(b_flat_dram_windows(nIter)(kIter));
            });
        });

        {
            // move to 1
            move_tile_window(a_copy_dram_window, {0, kKPerBlock});

            // move to next flat K
            move_tile_window(b_flat_dram_window, {0, BlockGemmShape::flatKPerBlock});

            // initialize C
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            // LDS write 0
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>)
            {
                auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                    PipelinePolicy::template MakeShuffledARegBlockDistribution<Problem>());
                shuffle_tile(a_shuffle_tmp, a_block_tile);
                const auto a_block_tile_tmp = tile_elementwise_in(a_element_func, a_shuffle_tmp);
                store_tile(a_copy_lds_window, a_block_tile_tmp);
            }
            else
            {
                store_tile(a_copy_lds_window, tile_elementwise_in(a_element_func, a_block_tile));
            }
            block_sync_lds();
        }

        index_t iCounter = num_loop / 2 - 1;
        while(iCounter > 0)
        {
            // global read i + 1
            a_block_tile = load_tile(a_copy_dram_window);

            // GEMM i
            block_flatmm(c_block_tile, a_warp_windows, b_warp_tensor);

            block_sync_lds();

            static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                    b_flat_dram_windows(nIter)(kIter) = b_flat_dram_window;

                    move_tile_window(b_flat_dram_windows(nIter)(kIter),
                                     {nIter * NFlatPerBlockPerIter, kIter * KFlatPerBlockPerIter});

                    b_warp_tensor_2(nIter)(kIter) = load_tile(b_flat_dram_windows(nIter)(kIter));
                });
            });

            // move to i + 2
            move_tile_window(a_copy_dram_window, {0, kKPerBlock});

            // move to next flat K
            move_tile_window(b_flat_dram_window, {0, BlockGemmShape::flatKPerBlock});

            // LDS write i + 1
            auto a_block_tile_tmp = tile_elementwise_in(a_element_func, a_block_tile);
            store_tile(a_copy_lds_window, a_block_tile_tmp);
            HotLoopScheduler();
            block_sync_lds();

            // iCounter--;

            // global read i + 1
            a_block_tile = load_tile(a_copy_dram_window);

            // GEMM i
            block_flatmm(c_block_tile, a_warp_windows, b_warp_tensor_2);

            block_sync_lds();

            static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                    b_flat_dram_windows(nIter)(kIter) = b_flat_dram_window;

                    move_tile_window(b_flat_dram_windows(nIter)(kIter),
                                     {nIter * NFlatPerBlockPerIter, kIter * KFlatPerBlockPerIter});

                    b_warp_tensor(nIter)(kIter) = load_tile(b_flat_dram_windows(nIter)(kIter));
                });
            });

            // move to i + 2
            move_tile_window(a_copy_dram_window, {0, kKPerBlock});

            // move to next flat K
            move_tile_window(b_flat_dram_window, {0, BlockGemmShape::flatKPerBlock});

            // LDS write i + 1
            a_block_tile_tmp = tile_elementwise_in(a_element_func, a_block_tile);
            store_tile(a_copy_lds_window, a_block_tile_tmp);

            HotLoopScheduler();
            block_sync_lds();

            iCounter--;
        }

        // tail
        {
            // global read i + 1
            a_block_tile = load_tile(a_copy_dram_window);

            // GEMM i
            block_flatmm(c_block_tile, a_warp_windows, b_warp_tensor);

            block_sync_lds();

            static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                    b_flat_dram_windows(nIter)(kIter) = b_flat_dram_window;

                    move_tile_window(b_flat_dram_windows(nIter)(kIter),
                                     {nIter * NFlatPerBlockPerIter, kIter * KFlatPerBlockPerIter});

                    b_warp_tensor_2(nIter)(kIter) = load_tile(b_flat_dram_windows(nIter)(kIter));
                });
            });

            // move to i + 2
            // move_tile_window(a_copy_dram_window, {0, kKPerBlock});

            // LDS write i + 1
            const auto a_block_tile_tmp = tile_elementwise_in(a_element_func, a_block_tile);
            store_tile(a_copy_lds_window, a_block_tile_tmp);

            // move to next flat K
            // move_tile_window(b_flat_dram_window, {0, BlockGemmShape::flatKPerBlock});

            HotLoopScheduler();
            block_sync_lds();

            // GEMM num_loop - 1
            block_flatmm(c_block_tile, a_warp_windows, b_warp_tensor_2);
        }

        return c_block_tile;
    }

    template <typename ADramBlockWindowTmp, typename BFlatBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BFlatBlockWindowTmp& b_flat_dram_block_window_tmp,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        return operator()(
            a_dram_block_window_tmp,
            [](const ADataType & a) { return a; },
            b_flat_dram_block_window_tmp,
            num_loop,
            p_smem);
    }
};

} // namespace ck_tile
