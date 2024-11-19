// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"

namespace ck_tile {

//  A Tile Window: global memory
//  B Tile Window: global memory
//  C Distributed tensor: register
template <typename Problem>
struct BaseGemmPipelineAgBgCrMem
{
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    static constexpr index_t BlockSize = Problem::kBlockSize;
    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    // TODO: Is this 32K value gfx9 arch specific?
    static constexpr index_t MinMemInFlyBytes = 32768;

    static constexpr index_t WgpPerCU =
        (4 * get_warp_size() / BlockSize) >= 1 ? 4 * get_warp_size() / BlockSize : 1;
    static constexpr index_t FullMemBandPrefetchStages = integer_divide_ceil(
        MinMemInFlyBytes / WgpPerCU,
        (MPerBlock * sizeof(ADataType) + NPerBlock * sizeof(BDataType)) * KPerBlock);
    static constexpr index_t PrefetchStages =
        FullMemBandPrefetchStages >= 2
            ? FullMemBandPrefetchStages <= 8 ? FullMemBandPrefetchStages : 8
            : 2;

    static constexpr index_t LocalPrefillStages = 1;
    static constexpr index_t GlobalBufferNum    = PrefetchStages;

    CK_TILE_HOST static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    CK_TILE_HOST static constexpr TailNumber GetBlockLoopTailNum(index_t num_loop)
    {
        if(num_loop % PrefetchStages == 1)
        {
            return TailNumber::One;
        }
        else if(num_loop % PrefetchStages == 2)
        {
            return TailNumber::Two;
        }
        else if(num_loop % PrefetchStages == 3)
        {
            return TailNumber::Three;
        }
        else if(num_loop % PrefetchStages == 4)
        {
            return TailNumber::Four;
        }
        else if(num_loop % PrefetchStages == 5)
        {
            return TailNumber::Five;
        }
        else if(num_loop % PrefetchStages == 6)
        {
            return TailNumber::Six;
        }
        else if(num_loop % PrefetchStages == 7)
        {
            return TailNumber::Seven;
        }
        else
        {
            return TailNumber::Full;
        }
    }
};

// Maximum Global Memory throughput pipeline with >=32KB data in fly
// GlobalPrefetchStages: >=2
// LocalPreFillStages: 1
// LocalPreFetchStages: 0
// LocalSharedMemoryBuffer: 1
template <typename Problem, typename Policy = GemmPipelineAGmemBGmemCRegV1DefaultPolicy>
struct GemmPipelineAgBgCrMem : public BaseGemmPipelineAgBgCrMem<Problem>
{
    using Base = BaseGemmPipelineAgBgCrMem<Problem>;

    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    using ALayout = remove_cvref_t<typename Problem::ALayout>;
    using BLayout = remove_cvref_t<typename Problem::BLayout>;
    using CLayout = remove_cvref_t<typename Problem::CLayout>;

    using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;
    using I0        = number<0>;

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

    // Where is the right place for HasHotLoop and TailNum ???
    static constexpr bool HasHotLoop = Problem::HasHotLoop;
    static constexpr auto TailNum    = Problem::TailNum;
    static constexpr auto Scheduler  = Problem::Scheduler;

    using Base::PrefetchStages;

    CK_TILE_HOST_DEVICE constexpr index_t GetStaticLdsSize()
    {
        return integer_divide_ceil(
                   sizeof(ADataType) *
                       Policy::template MakeALdsBlockDescriptor<Problem>().get_element_space_size(),
                   16) *
                   16 +
               sizeof(BDataType) *
                   Policy::template MakeBLdsBlockDescriptor<Problem>().get_element_space_size();
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <GemmPipelineScheduler Scheduler>
    struct PipelineImpl
    {
    };

    template <>
    struct PipelineImpl<GemmPipelineScheduler::Intrawave>
    {
        template <typename DstBlockTile, typename SrcTileWindow>
        CK_TILE_DEVICE void GlobalPrefetch(DstBlockTile& dst_block_tile,
                                           SrcTileWindow& dram_tile_window) const
        {
            load_tile(dst_block_tile, dram_tile_window);
            move_tile_window(dram_tile_window, {0, KPerBlock});
        }

        template <typename DstTileWindow, typename SrcBlockTile, typename ElementFunction>
        CK_TILE_DEVICE void LocalPrefill(DstTileWindow& lds_tile_window,
                                         const SrcBlockTile& src_block_tile,
                                         const ElementFunction& element_func) const
        {
            const auto block_tile_tmp = tile_elementwise_in(element_func, src_block_tile);
            store_tile(lds_tile_window, block_tile_tmp);
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

            static_assert(MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                              NPerBlock ==
                                  BDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                              KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                          "A/B block window appropriate sizes must be equal to MPerBlock/NPerblock"
                          " or KPerBlock!");

            // ------------------------------------------------------------------------------------
            // Definitions of all needed tiles

            // A tile in LDS
            ADataType* p_a_lds              = static_cast<ADataType*>(p_smem);
            constexpr auto a_lds_block_desc = Policy::template MakeALdsBlockDescriptor<Problem>();
            auto a_lds_block = make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_block_desc);

            // TODO: LDS alignment should come from Policy!
            constexpr index_t a_lds_block_space_size_aligned =
                integer_divide_ceil(sizeof(ADataType) * a_lds_block_desc.get_element_space_size(),
                                    16) *
                16;

            // B tile in LDS
            BDataType* p_b_lds = static_cast<BDataType*>(
                static_cast<void*>(static_cast<char*>(p_smem) + a_lds_block_space_size_aligned));
            constexpr auto b_lds_block_desc = Policy::template MakeBLdsBlockDescriptor<Problem>();
            auto b_lds_block = make_tensor_view<address_space_enum::lds>(p_b_lds, b_lds_block_desc);

            // A DRAM tile window for load
            auto a_copy_dram_window =
                make_tile_window(a_dram_block_window_tmp.get_bottom_tensor_view(),
                                 make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                                 a_dram_block_window_tmp.get_window_origin(),
                                 Policy::template MakeADramTileDistribution<Problem>());

            // A LDS tile window for store
            auto a_copy_lds_window =
                make_tile_window(a_lds_block,
                                 make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                                 {0, 0},
                                 a_copy_dram_window.get_tile_distribution());
            // B DRAM tile window for load
            auto b_copy_dram_window =
                make_tile_window(b_dram_block_window_tmp.get_bottom_tensor_view(),
                                 make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                                 b_dram_block_window_tmp.get_window_origin(),
                                 Policy::template MakeBDramTileDistribution<Problem>());

            // B LDS tile window for store
            auto b_copy_lds_window =
                make_tile_window(b_lds_block,
                                 make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                                 {0, 0},
                                 b_copy_dram_window.get_tile_distribution());

            // A LDS tile for block GEMM
            auto a_lds_gemm_window = make_tile_window(
                a_lds_block, make_tuple(number<MPerBlock>{}, number<KPerBlock>{}), {0, 0});
            // B LDS tile for block GEMM
            auto b_lds_gemm_window = make_tile_window(
                b_lds_block, make_tuple(number<NPerBlock>{}, number<KPerBlock>{}), {0, 0});

            // Block GEMM
            constexpr auto block_gemm = BlockGemm();
            auto c_block_tile         = block_gemm.MakeCBlockTile();

            using ABlockTileDistr = decltype(a_copy_dram_window.get_tile_distribution());
            using BBlockTileDistr = decltype(b_copy_dram_window.get_tile_distribution());

            using ABlockTile =
                decltype(make_static_distributed_tensor<ADataType>(ABlockTileDistr{}));
            using BBlockTile =
                decltype(make_static_distributed_tensor<BDataType>(BBlockTileDistr{}));

            tuple_array<ABlockTile, PrefetchStages> a_block_tiles;
            tuple_array<BBlockTile, PrefetchStages> b_block_tiles;

            // -----------------------------------------------------------------------------------------
            // Gemm pipeline start

            // prefetch
            // global read 0
            GlobalPrefetch(a_block_tiles.get(I0{}), a_copy_dram_window);
            GlobalPrefetch(b_block_tiles.get(I0{}), b_copy_dram_window);

            // initialize C
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            // LDS write 0
            LocalPrefill(a_copy_lds_window, a_block_tiles.get(I0{}), a_element_func);
            LocalPrefill(b_copy_lds_window, b_block_tiles.get(I0{}), b_element_func);

            // Global prefetch [1, PrefetchStages]
            static_for<1, PrefetchStages, 1>{}([&](auto prefetch_idx) {
                GlobalPrefetch(a_block_tiles.get(number<prefetch_idx>{}), a_copy_dram_window);
                GlobalPrefetch(b_block_tiles.get(number<prefetch_idx>{}), b_copy_dram_window);
            });

            // main body
            if constexpr(HasHotLoop)
            {
                index_t i = 0;
                do
                {
                    static_for<0, PrefetchStages, 1>{}([&](auto prefetch_idx) {
                        block_sync_lds();
                        // block_gemm.LocalPrefetch();
                        block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);

                        block_sync_lds();

                        LocalPrefill(
                            a_copy_lds_window,
                            a_block_tiles.get(number<(prefetch_idx + 1) % PrefetchStages>{}),
                            a_element_func);
                        LocalPrefill(
                            b_copy_lds_window,
                            b_block_tiles.get(number<(prefetch_idx + 1) % PrefetchStages>{}),
                            b_element_func);

                        GlobalPrefetch(a_block_tiles.get(number<prefetch_idx>{}),
                                       a_copy_dram_window);
                        GlobalPrefetch(b_block_tiles.get(number<prefetch_idx>{}),
                                       b_copy_dram_window);
                    });

                    i += PrefetchStages;
                } while(i < (num_loop - PrefetchStages));
            }

            auto HotLoopTail = [&](auto tail_num) {
                static_for<1, tail_num, 1>{}([&](auto prefetch_idx) {
                    block_sync_lds();

                    // block_gemm.LocalPrefetch();
                    block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);

                    block_sync_lds();
                    LocalPrefill(a_copy_lds_window,
                                 a_block_tiles.get(number<prefetch_idx>{}),
                                 a_element_func);
                    LocalPrefill(b_copy_lds_window,
                                 b_block_tiles.get(number<prefetch_idx>{}),
                                 b_element_func);
                });

                block_sync_lds();
                // block_gemm.LocalPrefetch();
                block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
            };

            if constexpr(TailNum == TailNumber::One)
            {
                block_sync_lds();
                // block_gemm.LocalPrefetch();
                block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
            }
            else if constexpr(TailNum == TailNumber::Two)
            {
                HotLoopTail(number<2>{});
            }
            else if constexpr(TailNum == TailNumber::Three)
            {
                HotLoopTail(number<3>{});
            }
            else if constexpr(TailNum == TailNumber::Four)
            {
                HotLoopTail(number<4>{});
            }
            else if constexpr(TailNum == TailNumber::Five)
            {
                HotLoopTail(number<5>{});
            }
            else if constexpr(TailNum == TailNumber::Six)
            {
                HotLoopTail(number<6>{});
            }
            else if constexpr(TailNum == TailNumber::Seven)
            {
                HotLoopTail(number<7>{});
            }
            else if constexpr(TailNum == TailNumber::Full)
            {
                HotLoopTail(number<PrefetchStages>{});
            }

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
};

} // namespace ck_tile
