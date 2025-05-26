// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// #define BLOCKWISE_LOADSTORE
#include "ck_tile/core.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp"

namespace ck_tile {

template <typename Problem, typename PipelinePolicy = UniversalFlatmmPipelineAgBgCrPolicy>
struct FlatmmPipelineAGmemBGmemCRegV1
{
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>; // TileFlatmmShape

    using ALayout = remove_cvref_t<typename Problem::ALayout>;
    using BLayout = remove_cvref_t<typename Problem::BLayout>;
    using CLayout = remove_cvref_t<typename Problem::CLayout>;

    using BlockFlatmm =
        remove_cvref_t<decltype(PipelinePolicy::template GetBlockFlatmm<Problem>())>;

    static constexpr index_t BlockSize = Problem::kBlockSize;

    static constexpr index_t kMPerBlock = BlockGemmShape::kM;
    static constexpr index_t kNPerBlock = BlockGemmShape::kN;
    static constexpr index_t kKPerBlock = BlockGemmShape::kK;

    static constexpr index_t flatKPerWarp = BlockGemmShape::flatKPerWarp;
    static constexpr index_t flatNPerWarp = BlockGemmShape::flatNPerWarp;

    static constexpr index_t GetVectorSizeA() { return Problem::VectorSizeA; }
    static constexpr index_t GetVectorSizeB() { return Problem::VectorSizeB; }
    static constexpr index_t GetVectorSizeC() { return Problem::VectorSizeC; }

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr index_t kLdsAlignmentInBytes = 16;

    static constexpr auto I0   = number<0>();
    static constexpr auto I1   = number<1>();
    static constexpr auto I2   = number<2>();
    static constexpr auto idxM = I0;
    static constexpr auto idxN = I1;
    static constexpr auto idxK = I2;
    using BlockTile            = remove_cvref_t<typename BlockGemmShape::BlockTile>;
    using BlockWarps           = remove_cvref_t<typename BlockGemmShape::BlockWarps>;
    using WarpTile             = remove_cvref_t<typename BlockGemmShape::WarpTile>;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "pipeline_AGmemBGmemCRegV1", 
                      concat('x', kMPerBlock, kNPerBlock, kKPerBlock,  BlockSize),
                      concat('x', GetVectorSizeA(), GetVectorSizeB(), GetVectorSizeC()),
                      concat('x', kPadM, kPadN, kPadK));
        // clang-format on
    }

    // For the basic gemm pipelien DoubleSmemBuffer set to be false naturally.
    static constexpr bool DoubleSmemBuffer = false;

    CK_TILE_HOST_DEVICE static constexpr auto TransposeC() { return Problem::TransposeC; }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return PipelinePolicy::template GetSmemSize<Problem>();
    }

    CK_TILE_HOST_DEVICE static constexpr auto HotLoopScheduler()
    {
#if 0
#if defined(USING_MFMA_16x16x32) && defined(ENABLE_FP8) || defined(USING_MFMA_32x32x16)
        constexpr auto config = BlockFlatmm::BlockPolicy::template GetWarpGemmMWarpNWarp<Problem>();

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
#endif
#if defined(USING_MFMA_16x16x32) && defined(ENABLE_FP8)
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

#elif defined(USING_MFMA_32x32x16)
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
#endif
#endif
        // static_for<0, 4, 1>{}([&](auto i) {
        //     ignore = i;
        //     __builtin_amdgcn_sched_group_barrier(0x020, 2, 0); // VMEM read
        //     __builtin_amdgcn_sched_group_barrier(0x100, 4, 0); // DS read
        //     __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
        // });

        // static_for<0, 4, 1>{}([&](auto i) {
        //     ignore = i;
        //     __builtin_amdgcn_sched_group_barrier(0x100, 4, 0); // DS read
        //     __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
        //     //__builtin_amdgcn_sched_group_barrier(0x020, 2, 0); // VMEM read
        // });

        // static_for<0, 16, 1>{}([&](auto i) {
        //     ignore = i;
        //     __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
        //     __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
        // });
        // static_for<0, 8, 1>{}([&](auto i) {
        //     ignore = i;
        //     __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
        //     __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
        // });

        static_for<0, 7, 1>{}([&](auto i) {
            ignore = i;
            __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x020, 2, 0); // VMEM read
            __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x100, 2, 0); // DS read
        });
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
        __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
        __builtin_amdgcn_sched_group_barrier(0x020, 2, 0); // VMEM read
        __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0); // DS read

        static_for<0, 7, 1>{}([&](auto i) {
            ignore = i;
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
            __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x100, 2, 0); // DS read
        });
        __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
        __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
        __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0); // DS read

        __builtin_amdgcn_sched_barrier(0);
    }


    CK_TILE_HOST_DEVICE static constexpr auto TailHotLoopScheduler()
    {
        static_for<0, 8, 1>{}([&](auto i) {
            ignore = i;
            __builtin_amdgcn_sched_group_barrier(0x100, 4, 0); // DS read
            __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
        });

        static_for<0, 24, 1>{}([&](auto i) {
            ignore = i;
            __builtin_amdgcn_sched_group_barrier(0x008, 2, 0); // MFMA
        });
    }

    template <typename ADramBlockWindowTmp, typename BFlatBlockWindowTmp, typename AElementFunction>
    CK_TILE_HOST_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                        const AElementFunction& a_element_func,
                                        const BFlatBlockWindowTmp& b_flat_dram_block_window_tmp,
                                        index_t num_loop,
                                        void* p_smem) const
    {
        static_assert(
            std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kMPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<0>{}],
                      "wrong!");
        static_assert(kKPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        constexpr auto config = BlockFlatmm::BlockPolicy::template GetWarpGemmMWarpNWarp<Problem>();

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

        constexpr index_t K1 = 16 / sizeof(ADataType);
        constexpr index_t ACopyLoadNum = kMPerBlock * kKPerBlock / BlockSize / K1;
        constexpr index_t AcopyPerLoadM = kMPerBlock / ACopyLoadNum;

        const index_t iMWarp = get_warp_id() / NWarp;

        using CWarpDstr   = typename WG::CWarpDstr;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        __builtin_amdgcn_sched_barrier(0);
        
        // A tile in LDS
        ADataType* p_a_lds = static_cast<ADataType*>(p_smem);

        constexpr auto a_lds_block_desc =
            PipelinePolicy::template MakeALdsBlockDescriptor<Problem>();

        auto a_lds_block = make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_block_desc);
        auto a_lds_block_2 = make_tensor_view<address_space_enum::lds>(p_a_lds+PipelinePolicy::template GetSmemSizeA<Problem>()/sizeof(typename Problem::ADataType), a_lds_block_desc);

        // A DRAM tile window for load
        #ifdef BLOCKWISE_LOADSTORE
        auto a_copy_dram_window =
            make_tile_window(a_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                             a_dram_block_window_tmp.get_window_origin(),
                             PipelinePolicy::template MakeADramTileDistribution<Problem>());

        auto a_copy_lds_window =
            make_tile_window(a_lds_block,
                             make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                             {0, 0},
                             PipelinePolicy::template MakeADramTileDistribution<Problem>());

        auto a_copy_lds_window_2 =
            make_tile_window(a_lds_block_2,
                            make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                            {0, 0},
                            PipelinePolicy::template MakeADramTileDistribution<Problem>());
        #else
        auto a_copy_dram_window_tmp =
            make_tile_window(a_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<AcopyPerLoadM>{}, number<kKPerBlock>{}),
                             a_dram_block_window_tmp.get_window_origin(),
                             PipelinePolicy::template MakeADramDistribution<Problem>());

        statically_indexed_array<decltype(a_copy_dram_window_tmp), ACopyLoadNum> a_copy_dram_window;
        static_for<0, ACopyLoadNum, 1>{}([&](auto AIter) {
            a_copy_dram_window(AIter) = a_copy_dram_window_tmp;
            move_tile_window(a_copy_dram_window(AIter), {AIter * AcopyPerLoadM, 0});
        });

        auto a_copy_lds_window_tmp = make_tile_window(
            a_lds_block,
            make_tuple(number<AcopyPerLoadM>{}, number<kKPerBlock>{}),
            {0, 0},
            PipelinePolicy::template MakeADramDistribution<Problem>()
        );

        statically_indexed_array<decltype(a_copy_lds_window_tmp), ACopyLoadNum> a_copy_lds_window;
        static_for<0, ACopyLoadNum, 1>{}([&](auto AIter) {
            a_copy_lds_window(AIter) = a_copy_lds_window_tmp;
            move_tile_window(a_copy_lds_window(AIter), {AIter * AcopyPerLoadM, 0});
        });

        auto a_copy_lds_window_tmp_2 = make_tile_window(
            a_lds_block_2,
            make_tuple(number<AcopyPerLoadM>{}, number<kKPerBlock>{}),
            {0, 0},
            PipelinePolicy::template MakeADramDistribution<Problem>()
        );

        statically_indexed_array<decltype(a_copy_lds_window_tmp_2), ACopyLoadNum> a_copy_lds_window_2;
        static_for<0, ACopyLoadNum, 1>{}([&](auto AIter) {
            a_copy_lds_window_2(AIter) = a_copy_lds_window_tmp_2;
            move_tile_window(a_copy_lds_window_2(AIter), {AIter * AcopyPerLoadM, 0});
        });
        #endif

        // A LDS tile for block GEMM
        // auto a_lds_gemm_window = make_tile_window(
        //     a_lds_block, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {0, 0});

        // ping-pong window for A LDS
        auto a_warp_window_tmp = make_tile_window(
            a_lds_block,
            make_tuple(number<WG::kM>{}, number<WG::kK>{}),
            {iMWarp * WG::kM, 0},
            make_static_tile_distribution(typename WG::AWarpDstrEncoding{}));

        auto a_warp_window_tmp_2 = make_tile_window(
            a_lds_block_2,
            make_tuple(number<WG::kM>{}, number<WG::kK>{}),
            {iMWarp * WG::kM, 0},
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

        // static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
        //     static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
        //         a_warp_windows(mIter)(number<kIter+KIterPerWarp>{}) = a_warp_window_tmp_2;

        //         move_tile_window(a_warp_windows(mIter)(number<kIter+KIterPerWarp>{}),
        //                             {mIter * MPerBlockPerIter, kIter * KPerBlockPerIter});
        //     });
        // });

        statically_indexed_array<
            statically_indexed_array<decltype(a_warp_window_tmp_2), KIterPerWarp>,
            MIterPerWarp>
            a_warp_windows_2;
        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                a_warp_windows_2(mIter)(kIter) = a_warp_window_tmp_2;

                move_tile_window(a_warp_windows_2(mIter)(kIter),
                                 {mIter * MPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });

        // Block GEMM
        auto block_flatmm = BlockFlatmm();
        // Acc register tile
        auto c_block_tile = block_flatmm.MakeCBlockTile();

        // B flat DRAM window for load
        auto b_flat_distribution =
            PipelinePolicy::template MakeBFlatDramTileDistribution<Problem>();
        auto b_flat_dram_window = // tile_window_with_static_distribution
            make_tile_window(
                b_flat_dram_block_window_tmp.get_bottom_tensor_view(), // from kernel gemm_pad_views
                make_tuple(number<flatNPerWarp>{}, number<flatKPerWarp>{}),
                b_flat_dram_block_window_tmp.get_window_origin(),
                b_flat_distribution);

        // pingpong buffer for B
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


        // Prefetch A0
        #ifdef BLOCKWISE_LOADSTORE
        auto a_block_tile = load_tile(a_copy_dram_window);
        // move A window to next k
        move_tile_window(a_copy_dram_window, {0, kKPerBlock});
        #else
        statically_indexed_array<decltype(load_tile(a_copy_dram_window(number<0>{}))), ACopyLoadNum> a_block_tile;
        static_for<0, ACopyLoadNum, 1>{}([&](auto AIter) {
            a_block_tile(AIter) = load_tile(a_copy_dram_window(AIter));
            move_tile_window(a_copy_dram_window(AIter), {0, kKPerBlock});
        });
        #endif

        // prefetch B
        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                b_flat_dram_windows(nIter)(kIter) = b_flat_dram_window;

                move_tile_window(b_flat_dram_windows(nIter)(kIter),
                                {nIter * NFlatPerBlockPerIter, kIter * KFlatPerBlockPerIter});

                b_warp_tensor(nIter)(kIter) = load_tile(b_flat_dram_windows(nIter)(kIter));
            });
        });
        // move B window to next flat K
        move_tile_window(b_flat_dram_window, {0, BlockGemmShape::flatKPerBlock});

        // Prefill A0
        // if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>)
        // {
        //     auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
        //         PipelinePolicy::template MakeShuffledARegBlockDistribution<Problem>());
        //     shuffle_tile(a_shuffle_tmp, a_block_tile);
        //     const auto a_block_tile_tmp = tile_elementwise_in(a_element_func, a_shuffle_tmp);
        //     store_tile(a_copy_lds_window, a_block_tile_tmp);
        // }
        // else
        // {
        //     store_tile(a_copy_lds_window, tile_elementwise_in(a_element_func, a_block_tile));
        // }
        #ifdef BLOCKWISE_LOADSTORE
        auto a_block_tile_tmp = tile_elementwise_in(a_element_func, a_block_tile);
        store_tile(a_copy_lds_window, a_block_tile_tmp);
        #else
        static_for<0, ACopyLoadNum, 1>{}([&](auto AIter) {
            store_tile(a_copy_lds_window(AIter), tile_elementwise_in(a_element_func, a_block_tile(AIter)));
        });
        #endif
        __builtin_amdgcn_sched_barrier(0);

        // Prefetch A1
        #ifdef BLOCKWISE_LOADSTORE
        a_block_tile = load_tile(a_copy_dram_window);
        // move A window to next k
        move_tile_window(a_copy_dram_window, {0, kKPerBlock});
        #else
        static_for<0, ACopyLoadNum, 1>{}([&](auto AIter) {
            a_block_tile(AIter) = load_tile(a_copy_dram_window(AIter));
            move_tile_window(a_copy_dram_window(AIter), {0, kKPerBlock});
        });
        #endif

        // initialize C
        tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

        block_sync_lds();

        // preload A00,A10 from lds
        statically_indexed_array<decltype(load_tile(a_warp_windows(number<0>{})(number<0>{}))), 2> a_warp_tensor;
        static_for<0, 2, 1>{}([&](auto mIter) {
            a_warp_tensor(mIter) = load_tile(a_warp_windows(mIter)(number<0>{}));
        });
        __builtin_amdgcn_sched_barrier(0);

        // if(threadIdx.x==0){
        //     for(int i=0;i<a_block_tile.get_thread_buffer_size();i++) {
        //         printf("dteng--A buffer load: idx.x=%u, ablocktile=%f, buffer size=%d\n", threadIdx.x, type_convert<float>(a_block_tile.thread_buf_(i)),a_block_tile.get_thread_buffer_size());
        //     }
        // }
        // for(int i=0;i<a_warp_tensor(number<0>{}).get_thread_buffer_size();i++) {
        //     printf("dteng--A lds load 00: idx.x=%u, awarptensor=%f, buffer size=%d\n", threadIdx.x, type_convert<float>(a_warp_tensor(number<0>{}).thread_buf_(i)),a_warp_tensor(number<0>{}).get_thread_buffer_size());
        // }


        index_t iCounter = num_loop / 2 - 1;
        while(iCounter > 0)
            {
                #ifdef BLOCKWISE_LOADSTORE
                // prefetch B(2i+1)
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                        b_flat_dram_windows(nIter)(kIter) = b_flat_dram_window;

                        move_tile_window(b_flat_dram_windows(nIter)(kIter),
                                        {nIter * NFlatPerBlockPerIter, kIter * KFlatPerBlockPerIter});

                        b_warp_tensor_2(nIter)(kIter) = load_tile(b_flat_dram_windows(nIter)(kIter));
                    });
                });

                // Prefill A(2i+1)
                a_block_tile_tmp = tile_elementwise_in(a_element_func, a_block_tile);
                store_tile(a_copy_lds_window_2, a_block_tile_tmp);

                // Prefetch A(2i+2)
                a_block_tile = load_tile(a_copy_dram_window);
                // move A window to next k
                move_tile_window(a_copy_dram_window, {0, kKPerBlock});
                #endif
                
                // GEMM 2i
                static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                    static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                        constexpr auto AwarpIter = mIter % 2;
                        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                            // read C warp tensor from C block tensor
                            CWarpTensor c_warp_tensor;
        
                            c_warp_tensor.get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
                                merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                                merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));
        
                            // warp GEMM
                            WG{}(c_warp_tensor, a_warp_tensor(number<AwarpIter>{}), b_warp_tensor(nIter)(kIter));
        
                            // write C warp tensor into C block tensor
                            c_block_tile.set_y_sliced_thread_data(
                                merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                                merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                                c_warp_tensor.get_thread_buffer());

                            #ifndef BLOCKWISE_LOADSTORE
                            // prefetch B(2i+1)
                            constexpr auto curMNIter = mIter * NIterPerWarp + nIter;
                            constexpr auto BloadGap = MIterPerWarp / 2;
                            if constexpr((curMNIter < NIterPerWarp * BloadGap) && ((curMNIter % BloadGap)==1))
                            {
                                constexpr auto BnIter = curMNIter / BloadGap;
                                constexpr auto BkIter = kIter;
                                b_flat_dram_windows(number<BnIter>{})(number<BkIter>{}) = b_flat_dram_window;
                                move_tile_window(b_flat_dram_windows(number<BnIter>{})(number<BkIter>{}),
                                                {BnIter * NFlatPerBlockPerIter, BkIter * KFlatPerBlockPerIter});
                                b_warp_tensor_2(number<BnIter>{})(number<BkIter>{}) = load_tile(b_flat_dram_windows(number<BnIter>{})(number<BkIter>{}));
                            }
                            // Prefill A(2i+1)
                            constexpr auto ACopyLoadNumPerK = ACopyLoadNum / KIterPerWarp;
                            constexpr auto AStoreIter = mIter - (MIterPerWarp - 1 - ACopyLoadNumPerK);
                            if constexpr((AStoreIter >= 0) && (AStoreIter < ACopyLoadNumPerK) && ((nIter % NIterPerWarp)==0))
                            {                            
                                static_assert(AStoreIter+kIter*ACopyLoadNumPerK==0);
                                // store_tile(a_copy_lds_window_2(number<AStoreIter+kIter*ACopyLoadNumPerK>{}), tile_elementwise_in(a_element_func, a_block_tile(number<AStoreIter+kIter*ACopyLoadNumPerK>{})));
                            }
                            // Prefetch A(2i+2)
                            constexpr auto ALoadIter = mIter - (MIterPerWarp - 1 - ACopyLoadNumPerK) - 1;
                            if constexpr((ALoadIter >= 0) && (ALoadIter < ACopyLoadNumPerK) && ((nIter % NIterPerWarp)==(NIterPerWarp-2)))
                            {
                                //a_block_tile(number<ALoadIter+kIter*ACopyLoadNumPerK>{}) = load_tile(a_copy_dram_window(number<ALoadIter+kIter*ACopyLoadNumPerK>{}));
                                //move_tile_window(a_copy_dram_window(number<ALoadIter+kIter*ACopyLoadNumPerK>{}), {0, kKPerBlock});
                            }
                            #endif

                            //barrier
                            if constexpr((kIter == KIterPerWarp - 1) && (mIter == (MIterPerWarp - 2)) && (nIter == (NIterPerWarp-2)))
                            {
                                block_sync_lds();
                            }
                            __builtin_amdgcn_sched_barrier(0x7F6);
                        });
                        // preload next A from lds
                        if constexpr((kIter != KIterPerWarp - 1) || (mIter < (MIterPerWarp - 2)))
                        {
                            constexpr auto AmIter    = (mIter + 2) % MIterPerWarp;
                            constexpr auto AkIter    = (kIter + (mIter + 2) / MIterPerWarp);
                            a_warp_tensor(number<AwarpIter>{}) = load_tile(a_warp_windows(number<AmIter>{})(number<AkIter>{}));
                        }
                    });
                });
                //block_flatmm(c_block_tile, a_warp_windows, b_warp_tensor);

                // move B window to next flat K
                move_tile_window(b_flat_dram_window, {0, BlockGemmShape::flatKPerBlock});

                static_for<0, 2, 1>{}([&](auto mIter) {
                    a_warp_tensor(mIter) = load_tile(a_warp_windows_2(mIter)(number<0>{}));
                });
                HotLoopScheduler();
                
                //Next K
     
                // prefetch B(2i+2)
                #ifdef BLOCKWISE_LOADSTORE
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                        b_flat_dram_windows(nIter)(kIter) = b_flat_dram_window;

                        move_tile_window(b_flat_dram_windows(nIter)(kIter),
                                        {nIter * NFlatPerBlockPerIter, kIter * KFlatPerBlockPerIter});

                        b_warp_tensor(nIter)(kIter) = load_tile(b_flat_dram_windows(nIter)(kIter));
                    });
                });
                #endif
                                
                // Prefill A(2i+2)
                #ifdef BLOCKWISE_LOADSTORE
                a_block_tile_tmp = tile_elementwise_in(a_element_func, a_block_tile);
                store_tile(a_copy_lds_window, a_block_tile_tmp);
                #else
                static_for<0, ACopyLoadNum, 1>{}([&](auto AIter) {
                    store_tile(a_copy_lds_window(AIter), tile_elementwise_in(a_element_func, a_block_tile(AIter)));
                });
                #endif

                // Prefetch A(2i+3)
                #ifdef BLOCKWISE_LOADSTORE
                a_block_tile = load_tile(a_copy_dram_window);
                // move A window to next k
                move_tile_window(a_copy_dram_window, {0, kKPerBlock});
                #else
                static_for<0, ACopyLoadNum, 1>{}([&](auto AIter) {
                    a_block_tile(AIter) = load_tile(a_copy_dram_window(AIter));
                    move_tile_window(a_copy_dram_window(AIter), {0, kKPerBlock});
                });
                #endif

                // GEMM 2i+1
                static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                    static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                        // // read A warp tensor from A block window
                        // const auto a_warp_tensor = load_tile(a_warp_windows(mIter)(number<kIter+KIterPerWarp>{}));
                        constexpr auto AwarpIter = mIter % 2;
                        // a_warp_tensor(number<AwarpIter>{}) = load_tile(a_warp_windows_2(number<mIter>{})(number<kIter>{}));
                        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                            // read C warp tensor from C block tensor
                            CWarpTensor c_warp_tensor;
                            c_warp_tensor.get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
                                merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                                merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));
        
                            // warp GEMM
                            WG{}(c_warp_tensor, a_warp_tensor(number<AwarpIter>{}), b_warp_tensor_2(nIter)(kIter));
        
                            // write C warp tensor into C block tensor
                            c_block_tile.set_y_sliced_thread_data(
                                merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                                merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                                c_warp_tensor.get_thread_buffer());
                            
                            #ifndef BLOCKWISE_LOADSTORE
                            // prefetch B(2i+2)
                            constexpr auto curMNIter = mIter * NIterPerWarp + nIter;
                            constexpr auto BloadGap = MIterPerWarp / 2;
                            if constexpr((curMNIter < NIterPerWarp * BloadGap) && ((curMNIter % BloadGap)==1))
                            {
                                constexpr auto BnIter = curMNIter / BloadGap;
                                constexpr auto BkIter = kIter;
                                b_flat_dram_windows(number<BnIter>{})(number<BkIter>{}) = b_flat_dram_window;
                                move_tile_window(b_flat_dram_windows(number<BnIter>{})(number<BkIter>{}),
                                                {BnIter * NFlatPerBlockPerIter, BkIter * KFlatPerBlockPerIter});
                                b_warp_tensor(number<BnIter>{})(number<BkIter>{}) = load_tile(b_flat_dram_windows(number<BnIter>{})(number<BkIter>{}));
                            }
                            #endif

                            //barrier
                            if constexpr((kIter == KIterPerWarp - 1) && (mIter == (MIterPerWarp - 2)) && (nIter == (NIterPerWarp-2)))
                            {
                                block_sync_lds();
                            }
                            __builtin_amdgcn_sched_barrier(0x7F6);
                        });
                        // preload next A from lds
                        if constexpr((kIter!=KIterPerWarp-1)||(mIter<(MIterPerWarp-2)))
                        {
                            constexpr auto AmIter    = (mIter + 2) % MIterPerWarp;
                            constexpr auto AkIter    = (kIter + (mIter + 2) / MIterPerWarp);
                            a_warp_tensor(number<AwarpIter>{}) = load_tile(a_warp_windows_2(number<AmIter>{})(number<AkIter>{}));
                        }
                    });
                });            
                // block_flatmm(c_block_tile, a_warp_windows, b_warp_tensor_2);

                // move B window to next flat K
                move_tile_window(b_flat_dram_window, {0, BlockGemmShape::flatKPerBlock});

                static_for<0, 2, 1>{}([&](auto mIter) {
                    a_warp_tensor(mIter) = load_tile(a_warp_windows(mIter)(number<0>{}));
                });
                HotLoopScheduler();

                iCounter--;
            }

        // tail
        {
            // __builtin_amdgcn_sched_barrier(0);

            // GEMM loopK-1
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    // // read A warp tensor from A block window
                    // const auto a_warp_tensor = load_tile(a_warp_windows(mIter)(kIter));
                    constexpr auto AwarpIter = mIter % 2;
                    // a_warp_tensor(number<AwarpIter>{}) = load_tile(a_warp_windows(number<mIter>{})(number<kIter>{}));
                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // read C warp tensor from C block tensor
                        CWarpTensor c_warp_tensor;
    
                        c_warp_tensor.get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));
    
                        // warp GEMM
                        WG{}(c_warp_tensor, a_warp_tensor(number<AwarpIter>{}), b_warp_tensor(nIter)(kIter));
    
                        // write C warp tensor into C block tensor
                        c_block_tile.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());
                        
                        #ifndef BLOCKWISE_LOADSTORE
                        // prefetch B(loopK)
                        constexpr auto curMNIter = mIter * NIterPerWarp + nIter;
                        constexpr auto BloadGap = MIterPerWarp / 2;
                        if constexpr((curMNIter < NIterPerWarp * BloadGap) && ((curMNIter % BloadGap)==1))
                        {
                            constexpr auto BnIter = curMNIter / BloadGap;
                            constexpr auto BkIter = kIter;
                            b_flat_dram_windows(number<BnIter>{})(number<BkIter>{}) = b_flat_dram_window;
                            move_tile_window(b_flat_dram_windows(number<BnIter>{})(number<BkIter>{}),
                                            {BnIter * NFlatPerBlockPerIter, BkIter * KFlatPerBlockPerIter});
                            b_warp_tensor_2(number<BnIter>{})(number<BkIter>{}) = load_tile(b_flat_dram_windows(number<BnIter>{})(number<BkIter>{}));
                        }
                        #endif
                        __builtin_amdgcn_sched_barrier(0x7F6);
                    });
                    // preload next A from lds
                    if constexpr((kIter!=KIterPerWarp-1)||(mIter<(MIterPerWarp-2)))
                    {
                        constexpr auto AmIter    = (mIter + 2) % MIterPerWarp;
                        constexpr auto AkIter    = (kIter + (mIter + 2) / MIterPerWarp);
                        a_warp_tensor(number<AwarpIter>{}) = load_tile(a_warp_windows(number<AmIter>{})(number<AkIter>{}));
                    }
                });
            });   
            //block_flatmm(c_block_tile, a_warp_windows, b_warp_tensor);

            // prefetch B(loopK)
            #ifdef BLOCKWISE_LOADSTORE
            static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                    b_flat_dram_windows(nIter)(kIter) = b_flat_dram_window;

                    move_tile_window(b_flat_dram_windows(nIter)(kIter),
                                     {nIter * NFlatPerBlockPerIter, kIter * KFlatPerBlockPerIter});

                    b_warp_tensor_2(nIter)(kIter) = load_tile(b_flat_dram_windows(nIter)(kIter));
                });
            });
            #endif

            // Prefill A(loopK)
            #ifdef BLOCKWISE_LOADSTORE
            a_block_tile_tmp = tile_elementwise_in(a_element_func, a_block_tile);
            store_tile(a_copy_lds_window_2, a_block_tile_tmp);
            #else
            static_for<0, ACopyLoadNum, 1>{}([&](auto AIter) {
                store_tile(a_copy_lds_window_2(AIter), tile_elementwise_in(a_element_func, a_block_tile(AIter)));
            });
            #endif

            // HotLoopScheduler();
            block_sync_lds();

            static_for<0, 2, 1>{}([&](auto mIter) {
                a_warp_tensor(mIter) = load_tile(a_warp_windows_2(mIter)(number<0>{}));
            });

            // __builtin_amdgcn_sched_barrier(0);
            
            // GEMM loopK
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    // // read A warp tensor from A block window
                    // const auto a_warp_tensor = load_tile(a_warp_windows(mIter)(number<kIter+KIterPerWarp>{}));
                    constexpr auto AwarpIter = mIter % 2;
                    // a_warp_tensor(number<AwarpIter>{}) = load_tile(a_warp_windows_2(number<mIter>{})(number<kIter>{}));
                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // read C warp tensor from C block tensor
                        CWarpTensor c_warp_tensor;
    
                        c_warp_tensor.get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));
    
                        // warp GEMM
                        WG{}(c_warp_tensor, a_warp_tensor(number<AwarpIter>{}), b_warp_tensor_2(nIter)(kIter));
    
                        // write C warp tensor into C block tensor
                        c_block_tile.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());
                        __builtin_amdgcn_sched_barrier(0x7F6);
                    });
                    if constexpr((kIter!=KIterPerWarp-1)||(mIter<(MIterPerWarp-2)))
                    {
                        constexpr auto AmIter    = (mIter + 2) % MIterPerWarp;
                        constexpr auto AkIter    = (kIter + (mIter + 2) / MIterPerWarp);
                        a_warp_tensor(number<AwarpIter>{}) = load_tile(a_warp_windows_2(number<AmIter>{})(number<AkIter>{}));
                    }
                });
            });
            //block_flatmm(c_block_tile, a_warp_windows, b_warp_tensor_2);

            // TailHotLoopScheduler();
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
            [](const ADataType& a) { return a; },
            b_flat_dram_block_window_tmp,
            num_loop,
            p_smem);
    }
};

} // namespace ck_tile
