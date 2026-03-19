// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp"
#include "ck_tile/host/concat.hpp"

namespace ck_tile {

template <typename Problem>
struct BaseGemmPipelineAGmemBGmemCRegV1
{
    static constexpr index_t PrefetchStages   = 1;
    static constexpr index_t PrefillStages    = 1;
    static constexpr index_t GlobalBufferNum  = 1;
    static constexpr bool UsePersistentKernel = false;

    CK_TILE_HOST_DEVICE static constexpr auto TransposeC() { return Problem::TransposeC; }

    CK_TILE_HOST_DEVICE static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    CK_TILE_HOST_DEVICE static constexpr TailNumber GetBlockLoopTailNum(index_t)
    {
        return TailNumber::Empty;
    }

    template <typename RunFunction>
    CK_TILE_HOST_DEVICE static auto TailHandler(const RunFunction& run_func, bool has_hot_loop)
    {
        // Use amd_wave_read_first_lane to avoid higher resource usage.
        // It forces to store these values in SGPR.
        // Compiler cannot deduce if one path is used for all threads
        const bool has_hot_loop_first_lane = amd_wave_read_first_lane(has_hot_loop);

        if(has_hot_loop_first_lane)
        {
            return run_func(ck_tile::bool_constant<true>{});
        }
        else
        {
            return run_func(ck_tile::bool_constant<false>{});
        }
    }
};

//  A Tile Window: global memory
//  B Tile Window: global memory
//  C Distributed tensor: register
template <typename Problem, typename Policy = UniversalGemmPipelineAgBgCrPolicy>
struct GemmPipelineAGmemBGmemCRegV1 : public BaseGemmPipelineAGmemBGmemCRegV1<Problem>
{
    using Base             = BaseGemmPipelineAGmemBGmemCRegV1<Problem>;
    using PipelineImplBase = GemmPipelineAgBgCrImplBase<Problem, Policy>;

    using AsDataType = remove_cvref_t<typename Problem::AsDataTypeTuple>;
    using BsDataType = remove_cvref_t<typename Problem::BsDataTypeTuple>;
    using CDataType  = remove_cvref_t<typename Problem::CDataType>;

    using AElementWise   = remove_cvref_t<typename Problem::AElementWise>;
    using BElementWise   = remove_cvref_t<typename Problem::BElementWise>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    using AsLayout = remove_cvref_t<typename Problem::AsLayoutTuple>;
    using BsLayout = remove_cvref_t<typename Problem::BsLayoutTuple>;
    using CLayout  = remove_cvref_t<typename Problem::CLayout>;

    using ALayout = remove_cvref_t<std::tuple_element_t<0, AsLayout>>;
    using BLayout = remove_cvref_t<std::tuple_element_t<0, BsLayout>>;

    using ADataType = remove_cvref_t<std::tuple_element_t<0, AsDataType>>;
    using BDataType = remove_cvref_t<std::tuple_element_t<0, BsDataType>>;

    using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;

    using I0 = number<0>;
    using I1 = number<1>;
    using I2 = number<2>;

    static constexpr index_t BlockSize = Problem::kBlockSize;

    static constexpr index_t kMPerBlock = BlockGemmShape::kM;
    static constexpr index_t kNPerBlock = BlockGemmShape::kN;
    static constexpr index_t kKPerBlock = BlockGemmShape::kK;

    static constexpr bool Async = false;

    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeA()
    {
        return Policy::template GetVectorSizeA<Problem, IsWave32Host>();
    }
    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeB()
    {
        return Policy::template GetVectorSizeB<Problem, IsWave32Host>();
    }
    static constexpr index_t GetVectorSizeC() { return Policy::template GetVectorSizeC<Problem>(); }

    static constexpr index_t GetSmemPackA() { return Policy::template GetSmemPackA<Problem>(); }
    static constexpr index_t GetSmemPackB() { return Policy::template GetSmemPackB<Problem>(); }

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr bool Preshuffle = Problem::Preshuffle;

    static constexpr auto Scheduler = Problem::Scheduler;

    static constexpr auto is_a_load_tr_v = bool_constant<PipelineImplBase::is_a_load_tr>{};
    static constexpr auto is_b_load_tr_v = bool_constant<PipelineImplBase::is_b_load_tr>{};

    static constexpr index_t NumWaveGroups = Problem::NumWaveGroups;

    static constexpr index_t kLdsAlignmentInBytes = 16;

    [[nodiscard]] CK_TILE_HOST static const std::string GetPipelineName()
    {
        // clang-format off
        return "BASIC_V1";
        // clang-format on
    }

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
        return Policy::template GetSmemSize<Problem>();
    }

    template <GemmPipelineScheduler Scheduler>
    struct PipelineImpl : public PipelineImplBase
    {
    };

    template <>
    struct PipelineImpl<GemmPipelineScheduler::Intrawave> : public PipelineImplBase
    {
        using Base = PipelineImplBase;

        template <bool HasHotLoop,
                  typename AsDramBlockWindowTmp,
                  typename BsDramBlockWindowTmp,
                  typename AElementFunction,
                  typename BElementFunction,
                  typename std::enable_if_t<is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                                is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                            bool>* = nullptr>
        CK_TILE_HOST_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                            const AElementFunction& a_element_func,
                                            const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                            const BElementFunction& b_element_func,
                                            index_t num_loop,
                                            void* p_smem) const
        {
            using ADramBlockWindowTmp =
                remove_cvref_t<std::tuple_element_t<number<0>{}, AsDramBlockWindowTmp>>;
            using BDramBlockWindowTmp =
                remove_cvref_t<std::tuple_element_t<number<0>{}, BsDramBlockWindowTmp>>;

            static_assert(
                std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BDataType,
                                   remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
                "wrong!");

            constexpr bool is_a_col_major =
                std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
            constexpr bool is_b_row_major = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;

            static_assert(is_a_col_major
                              ? (kKPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 kMPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}])
                              : (kMPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 kKPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}]),
                          "A block window has incorrect lengths for defined ALayout!");
            static_assert(is_b_row_major
                              ? (kKPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 kNPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1{}])
                              : (kNPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 kKPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1{}]),
                          "B block window has incorrect lengths for defined BLayout!");
            // A tile in LDS
            ADataType* p_a_lds = static_cast<ADataType*>(p_smem);

            constexpr auto a_lds_block_desc = Policy::template MakeALdsBlockDescriptor<Problem>();

            auto a_lds_block = make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_block_desc);

            constexpr index_t a_lds_block_space_size_aligned =
                integer_divide_ceil(sizeof(ADataType) * a_lds_block_desc.get_element_space_size(),
                                    kLdsAlignmentInBytes) *
                kLdsAlignmentInBytes;

            // B tile in LDS
            BDataType* p_b_lds = static_cast<BDataType*>(
                static_cast<void*>(static_cast<char*>(p_smem) + a_lds_block_space_size_aligned));

            constexpr auto b_lds_block_desc = Policy::template MakeBLdsBlockDescriptor<Problem>();

            auto b_lds_block = make_tensor_view<address_space_enum::lds>(p_b_lds, b_lds_block_desc);

            // Tile distribution for load from lds
            constexpr auto a_lds_load_tile_distr =
                make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
            constexpr auto b_lds_load_tile_distr =
                make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());

            // A DRAM tile window for load
            // A LDS tile window for store
            // A LDS tile for block GEMM
            auto&& [as_copy_dram_window, a_copy_lds_window, a_lds_gemm_window] =
                Base::GetAWindows(a_dram_block_window_tmp, a_lds_block, a_lds_load_tile_distr);

            // B DRAM tile window for load
            // B LDS tile window for store
            // B LDS tile for block GEMM
            auto&& [bs_copy_dram_window, b_copy_lds_window, b_lds_gemm_window] =
                Base::GetBWindows(b_dram_block_window_tmp, b_lds_block, b_lds_load_tile_distr);

            using ADramTileWindowStep = typename ADramBlockWindowTmp::BottomTensorIndex;
            using BDramTileWindowStep = typename BDramBlockWindowTmp::BottomTensorIndex;

            constexpr ADramTileWindowStep a_dram_tile_window_step =
                is_a_col_major ? make_array(kKPerBlock, 0) : make_array(0, kKPerBlock);
            constexpr BDramTileWindowStep b_dram_tile_window_step =
                is_b_row_major ? make_array(kKPerBlock, 0) : make_array(0, kKPerBlock);

            // Block GEMM
            auto block_gemm = BlockGemm();

            // Acc register tile
            auto c_block_tile = block_gemm.MakeCBlockTile();

            // prefetch
            // global read 0
            // Load tile — during value loading, an elementwise function is executed for each A0,
            // A1, … AN. The values A0, A1, … AN are read by the same thread.
            auto elementwise_As_res =
                load_tile_with_elementwise(as_copy_dram_window, a_element_func);

            // Load tile — during value loading, an elementwise function is executed for each B0,
            // B1, … BN. The values B0, B1, … BN are read by the same thread.
            auto elementwise_Bs_res =
                load_tile_with_elementwise(bs_copy_dram_window, b_element_func);

            {
                // move to 1
                // Move each A — the enhanced function move_tile_window is executed, which takes a
                // tuple as input.
                move_tile_window(as_copy_dram_window, a_dram_tile_window_step);
                // Move each B — the enhanced function move_tile_window is executed, which takes a
                // tuple as input.
                move_tile_window(bs_copy_dram_window, b_dram_tile_window_step);

                // initialize C
                tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

                // LDS write 0
                if constexpr(is_a_col_major && !is_a_load_tr_v())
                {
                    auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                        Policy::template MakeShuffledARegTileDistribution<Problem>());
                    transpose_tile2d(a_shuffle_tmp, elementwise_As_res);
                    Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp);
                }
                else
                {
                    Base::LocalPrefill(a_copy_lds_window, elementwise_As_res);
                }
                if constexpr(is_b_row_major && !is_b_load_tr_v())
                {
                    auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                        Policy::template MakeShuffledBRegTileDistribution<Problem>());
                    transpose_tile2d(b_shuffle_tmp, elementwise_Bs_res);
                    Base::LocalPrefill(b_copy_lds_window, b_shuffle_tmp);
                }
                else
                {
                    Base::LocalPrefill(b_copy_lds_window, elementwise_Bs_res);
                }
            }

            if constexpr(HasHotLoop)
            {
                index_t iCounter = num_loop - 1;
                while(iCounter > 0)
                {
                    // global read i + 1
                    elementwise_As_res =
                        load_tile_with_elementwise(as_copy_dram_window, a_element_func);
                    elementwise_Bs_res =
                        load_tile_with_elementwise(bs_copy_dram_window, b_element_func);
                    block_sync_lds();

                    block_gemm.LocalPrefetch(
                        a_lds_gemm_window, b_lds_gemm_window, is_a_load_tr_v, is_b_load_tr_v);

                    // GEMM i
                    block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);

                    block_sync_lds();

                    // move to i + 2
                    move_tile_window(as_copy_dram_window, a_dram_tile_window_step);
                    move_tile_window(bs_copy_dram_window, b_dram_tile_window_step);

                    // LDS write i + 1
                    if constexpr(is_a_col_major && !is_a_load_tr_v())
                    {
                        auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                            Policy::template MakeShuffledARegTileDistribution<Problem>());
                        transpose_tile2d(a_shuffle_tmp, elementwise_As_res);
                        Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp);
                    }
                    else
                    {
                        Base::LocalPrefill(a_copy_lds_window, elementwise_As_res);
                    }
                    if constexpr(is_b_row_major && !is_b_load_tr_v())
                    {
                        auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                            Policy::template MakeShuffledBRegTileDistribution<Problem>());
                        transpose_tile2d(b_shuffle_tmp, elementwise_Bs_res);
                        Base::LocalPrefill(b_copy_lds_window, b_shuffle_tmp);
                    }
                    else
                    {
                        Base::LocalPrefill(b_copy_lds_window, elementwise_Bs_res);
                    }

                    iCounter--;
                }
            }

            // tail
            {
                block_sync_lds();
                block_gemm.LocalPrefetch(
                    a_lds_gemm_window, b_lds_gemm_window, is_a_load_tr_v, is_b_load_tr_v);
                // GEMM num_loop - 1
                block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
            }

            return c_block_tile;
        }
    };

    template <>
    struct PipelineImpl<GemmPipelineScheduler::Interwave> : public PipelineImplBase
    {
        using Base = PipelineImplBase;

        template <bool HasHotLoop,
                  typename AsDramBlockWindowTmp,
                  typename BsDramBlockWindowTmp,
                  typename AElementFunction,
                  typename BElementFunction,
                  typename std::enable_if_t<is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                                is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                            bool>* = nullptr>
        CK_TILE_HOST_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                            const AElementFunction& a_element_func,
                                            const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                            const BElementFunction& b_element_func,
                                            index_t num_loop,
                                            void* p_smem) const
        {
            using ADramBlockWindowTmp =
                remove_cvref_t<std::tuple_element_t<number<0>{}, AsDramBlockWindowTmp>>;
            using BDramBlockWindowTmp =
                remove_cvref_t<std::tuple_element_t<number<0>{}, BsDramBlockWindowTmp>>;

            static_assert(
                std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BDataType,
                                   remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
                "wrong!");

            constexpr bool is_a_col_major =
                std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
            constexpr bool is_b_row_major = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;

            static_assert(is_a_col_major
                              ? (kKPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 kMPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}])
                              : (kMPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 kKPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}]),
                          "A block window has incorrect lengths for defined ALayout!");
            static_assert(is_b_row_major
                              ? (kKPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 kNPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1{}])
                              : (kNPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 kKPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1{}]),
                          "B block window has incorrect lengths for defined BLayout!");
            // A tile in LDS
            ADataType* p_a_lds = static_cast<ADataType*>(p_smem);

            constexpr auto a_lds_block_desc = Policy::template MakeALdsBlockDescriptor<Problem>();

            auto a_lds_block = make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_block_desc);

            constexpr index_t a_lds_block_space_size_aligned =
                integer_divide_ceil(sizeof(ADataType) * a_lds_block_desc.get_element_space_size(),
                                    kLdsAlignmentInBytes) *
                kLdsAlignmentInBytes;

            // B tile in LDS
            BDataType* p_b_lds = static_cast<BDataType*>(
                static_cast<void*>(static_cast<char*>(p_smem) + a_lds_block_space_size_aligned));

            constexpr auto b_lds_block_desc = Policy::template MakeBLdsBlockDescriptor<Problem>();

            auto b_lds_block = make_tensor_view<address_space_enum::lds>(p_b_lds, b_lds_block_desc);

            // // Tile distribution for load from lds
            constexpr auto a_lds_load_tile_distr =
                make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
            constexpr auto b_lds_load_tile_distr =
                make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());

            // A DRAM tile window for load
            // A LDS tile window for store
            // A LDS tile for block GEMM
            auto&& [as_copy_dram_window, a_copy_lds_window, a_lds_gemm_window] =
                Base::GetAWindows(a_dram_block_window_tmp, a_lds_block, a_lds_load_tile_distr);

            // B DRAM tile window for load
            // B LDS tile window for store
            // B LDS tile for block GEMM
            auto&& [bs_copy_dram_window, b_copy_lds_window, b_lds_gemm_window] =
                Base::GetBWindows(b_dram_block_window_tmp, b_lds_block, b_lds_load_tile_distr);

            using ADramTileWindowStep = typename ADramBlockWindowTmp::BottomTensorIndex;
            using BDramTileWindowStep = typename BDramBlockWindowTmp::BottomTensorIndex;

            constexpr ADramTileWindowStep a_dram_tile_window_step =
                is_a_col_major ? make_array(kKPerBlock, 0) : make_array(0, kKPerBlock);
            constexpr BDramTileWindowStep b_dram_tile_window_step =
                is_b_row_major ? make_array(kKPerBlock, 0) : make_array(0, kKPerBlock);

            // Block GEMM
            auto block_gemm = BlockGemm();

            // Acc register tile
            auto c_block_tile = block_gemm.MakeCBlockTile();

            // prefetch
            // global read 0
            // Load tile — during value loading, an elementwise function is executed for each A0,
            // A1, … AN. The values A0, A1, … AN are read by the same thread.
            auto elementwise_As_res =
                load_tile_with_elementwise(as_copy_dram_window, a_element_func);

            // Load tile — during value loading, an elementwise function is executed for each B0,
            // B1, … BN. The values B0, B1, … BN are read by the same thread.
            auto elementwise_Bs_res =
                load_tile_with_elementwise(bs_copy_dram_window, b_element_func);

            {
                // move to 1
                // Move each A — the enhanced function move_tile_window is executed, which takes a
                // tuple as input.
                move_tile_window(as_copy_dram_window, a_dram_tile_window_step);
                // Move each B — the enhanced function move_tile_window is executed, which takes a
                // tuple as input.
                move_tile_window(bs_copy_dram_window, b_dram_tile_window_step);

                // initialize C
                tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

                // LDS write 0
                if constexpr(is_a_col_major && !is_a_load_tr_v())
                {
                    auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                        Policy::template MakeShuffledARegTileDistribution<Problem>());
                    transpose_tile2d(a_shuffle_tmp, elementwise_As_res);
                    Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp);
                }
                else
                {
                    Base::LocalPrefill(a_copy_lds_window, elementwise_As_res);
                }
                if constexpr(is_b_row_major && !is_b_load_tr_v())
                {
                    auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                        Policy::template MakeShuffledBRegTileDistribution<Problem>());
                    transpose_tile2d(b_shuffle_tmp, elementwise_Bs_res);
                    Base::LocalPrefill(b_copy_lds_window, b_shuffle_tmp);
                }
                else
                {
                    Base::LocalPrefill(b_copy_lds_window, elementwise_Bs_res);
                }
            }

            if constexpr(HasHotLoop)
            {
                index_t iCounter = num_loop - 1;
                while(iCounter > 0)
                {
                    // global read i + 1
                    elementwise_As_res =
                        load_tile_with_elementwise(as_copy_dram_window, a_element_func);
                    block_sync_lds();
                    elementwise_Bs_res =
                        load_tile_with_elementwise(bs_copy_dram_window, b_element_func);

                    // GEMM i
                    block_gemm(c_block_tile,
                               a_lds_gemm_window,
                               b_lds_gemm_window,
                               is_a_load_tr_v,
                               is_b_load_tr_v);

                    // move to i + 2
                    move_tile_window(as_copy_dram_window, a_dram_tile_window_step);
                    move_tile_window(bs_copy_dram_window, b_dram_tile_window_step);

                    // LDS write i + 1
                    if constexpr(is_a_col_major && !is_a_load_tr_v())
                    {
                        auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                            Policy::template MakeShuffledARegTileDistribution<Problem>());
                        transpose_tile2d(a_shuffle_tmp, elementwise_As_res);
                        Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp);
                    }
                    else
                    {
                        Base::LocalPrefill(a_copy_lds_window, elementwise_As_res);
                    }
                    if constexpr(is_b_row_major && !is_b_load_tr_v())
                    {
                        auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                            Policy::template MakeShuffledBRegTileDistribution<Problem>());
                        transpose_tile2d(b_shuffle_tmp, elementwise_Bs_res);
                        Base::LocalPrefill(b_copy_lds_window, b_shuffle_tmp);
                    }
                    else
                    {
                        Base::LocalPrefill(b_copy_lds_window, elementwise_Bs_res);
                    }

                    iCounter--;
                }
            }

            // tail
            {
                block_sync_lds();
                // GEMM num_loop - 1
                block_gemm(c_block_tile,
                           a_lds_gemm_window,
                           b_lds_gemm_window,
                           is_a_load_tr_v,
                           is_b_load_tr_v);
            }

            return c_block_tile;
        }
    };

    template <typename AsDramBlockWindowTmp,
              typename BsDramBlockWindowTmp,
              typename std::enable_if_t<is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                            is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        const bool has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto RunPipeline  = [&](auto hot_loop_) {
            constexpr bool hot_loop = hot_loop_.value;
            return PipelineImpl<Scheduler>{}.template operator()<hot_loop>(
                a_dram_block_window_tmp,
                element_wise::PassThrough{},
                b_dram_block_window_tmp,
                element_wise::PassThrough{},
                num_loop,
                p_smem);
        };
        return Base::TailHandler(RunPipeline, has_hot_loop);
    }

    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename std::enable_if_t<!is_detected<is_tuple, ADramBlockWindowTmp>::value &&
                                            !is_detected<is_tuple, BDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        return operator()(ck_tile::make_tuple(a_dram_block_window_tmp),
                          ck_tile::make_tuple(b_dram_block_window_tmp),
                          num_loop,
                          p_smem);
    }

    template <typename AsDramBlockWindowTmp,
              typename BsDramBlockWindowTmp,
              typename AElementFunction,
              typename BElementFunction,
              typename std::enable_if_t<is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                            is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_HOST_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                        const AElementFunction& a_element_func,
                                        const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                        const BElementFunction& b_element_func,
                                        index_t num_loop,
                                        void* p_smem) const
    {
        const bool has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto RunPipeline  = [&](auto hot_loop_) {
            constexpr bool hot_loop = hot_loop_.value;
            return PipelineImpl<Scheduler>{}.template operator()<hot_loop>(a_dram_block_window_tmp,
                                                                           a_element_func,
                                                                           b_dram_block_window_tmp,
                                                                           b_element_func,
                                                                           num_loop,
                                                                           p_smem);
        };
        return Base::TailHandler(RunPipeline, has_hot_loop);
    }
};

} // namespace ck_tile
