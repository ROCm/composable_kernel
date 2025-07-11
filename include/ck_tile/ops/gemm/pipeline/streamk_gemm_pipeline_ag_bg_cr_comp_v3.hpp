// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"

namespace ck_tile {

/**
 * @brief Stream-K compatible GEMM pipeline that supports partial accumulation
 * 
 * This extends the existing CompV3 pipeline to support Stream-K's partial tile computation
 * and atomic accumulation for load balancing across workgroups.
 */
template <typename Problem, typename Policy = UniversalGemmPipelineAgBgCrPolicy>
struct StreamKGemmPipelineAgBgCrCompV3 : public GemmPipelineAgBgCrCompV3<Problem, Policy>
{
    using Base = GemmPipelineAgBgCrCompV3<Problem, Policy>;
    using PipelineImplBase = GemmPipelineAgBgCrImplBase<Problem, Policy>;

    using ADataType      = typename Base::ADataType;
    using BDataType      = typename Base::BDataType;
    using CDataType      = typename Base::CDataType;
    using BlockGemmShape = typename Base::BlockGemmShape;

    using ALayout = typename Base::ALayout;
    using BLayout = typename Base::BLayout;
    using CLayout = typename Base::CLayout;

    using BlockGemm = typename Base::BlockGemm;
    using I0 = typename Base::I0;
    using I1 = typename Base::I1;
    using I2 = typename Base::I2;

    // Stream-K specific constants
    static constexpr bool SupportsPartialTiles = true;
    static constexpr bool SupportsAtomicAccumulation = true;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        return concat('_', "streamk_pipeline_AgBgCrCompV3", Base::GetName());
    }

    /**
     * @brief Extended pipeline implementation with Stream-K support
     */
    template <GemmPipelineScheduler Scheduler>
    struct StreamKPipelineImpl : public Base::template PipelineImpl<Scheduler>
    {
        using BaseImpl = typename Base::template PipelineImpl<Scheduler>;

        /**
         * @brief Run pipeline with partial K-range support for Stream-K
         */
        template <bool HasHotLoop,
                  TailNumber TailNum,
                  typename ADramBlockWindowTmp,
                  typename BDramBlockWindowTmp,
                  typename AElementFunction,
                  typename BElementFunction>
        CK_TILE_DEVICE auto RunPartialK(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                        const AElementFunction& a_element_func,
                                        const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                        const BElementFunction& b_element_func,
                                        index_t k_start,
                                        index_t k_end,
                                        void* p_smem) const
        {
            // Calculate effective K range for this partial computation
            const index_t k_length = k_end - k_start;
            const index_t num_loop = (k_length + BlockGemmShape::kK - 1) / BlockGemmShape::kK;

            // Create windowed views for the partial K range
            auto a_partial_window = a_dram_block_window_tmp;
            auto b_partial_window = b_dram_block_window_tmp;

            // Adjust windows to start at k_start
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>) {
                move_tile_window(a_partial_window, {0, k_start});
            } else {
                move_tile_window(a_partial_window, {k_start, 0});
            }

            if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>) {
                move_tile_window(b_partial_window, {k_start, 0});
            } else {
                move_tile_window(b_partial_window, {0, k_start});
            }

            // Run the base pipeline implementation with adjusted parameters
            return BaseImpl::template operator()<HasHotLoop, TailNum>(
                a_partial_window,
                a_element_func,
                b_partial_window,
                b_element_func,
                num_loop,
                p_smem);
        }

        /**
         * @brief Run pipeline with multiple K-slices for Stream-K work decomposition
         */
        template <bool HasHotLoop,
                  TailNumber TailNum,
                  typename ADramBlockWindowTmp,
                  typename BDramBlockWindowTmp,
                  typename AElementFunction,
                  typename BElementFunction>
        CK_TILE_DEVICE auto RunMultipleKSlices(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                               const AElementFunction& a_element_func,
                                               const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                               const BElementFunction& b_element_func,
                                               index_t work_start,
                                               index_t work_end,
                                               index_t num_tile_m,
                                               index_t num_tile_n,
                                               index_t num_tile_k,
                                               void* p_smem) const
        {
            // Initialize accumulation tile
            auto block_gemm = BlockGemm();
            auto c_accum_tile = block_gemm.MakeCBlockTile();
            tile_elementwise_inout([](auto& c) { c = 0; }, c_accum_tile);

            // Process each work unit assigned to this block
            for (index_t work_idx = work_start; work_idx < work_end; ++work_idx) {
                // Convert work index to tile coordinates
                const index_t tiles_per_k_slice = num_tile_m * num_tile_n;
                const index_t k_slice = work_idx / tiles_per_k_slice;
                const index_t tile_idx = work_idx % tiles_per_k_slice;
                
                const index_t tile_m = tile_idx / num_tile_n;
                const index_t tile_n = tile_idx % num_tile_n;

                // Calculate K range for this slice
                const index_t k_start = k_slice * BlockGemmShape::kK;
                const index_t k_end = (k_slice + 1) * BlockGemmShape::kK;

                // Process this K-slice
                auto partial_result = RunPartialK<HasHotLoop, TailNum>(
                    a_dram_block_window_tmp, a_element_func,
                    b_dram_block_window_tmp, b_element_func,
                    k_start, k_end, p_smem);

                // Accumulate results
                constexpr auto c_spans = decltype(c_accum_tile)::get_distributed_spans();
                sweep_tile_span(c_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(c_spans[number<1>{}], [&](auto idx1) {
                        auto idx = make_tuple(idx0, idx1);
                        c_accum_tile(idx) += partial_result(idx);
                    });
                });
            }

            return c_accum_tile;
        }

        /**
         * @brief Atomic accumulation for partial results
         */
        template <typename CBlockTile, typename CBlockWindow>
        CK_TILE_DEVICE void AtomicAccumulate(CBlockWindow& c_block_window, 
                                            const CBlockTile& c_block_tile) const
        {
            // Get the distributed spans for the tile
            constexpr auto c_spans = CBlockTile::get_distributed_spans();
            
            // Perform atomic accumulation for each element
            sweep_tile_span(c_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(c_spans[number<1>{}], [&](auto idx1) {
                    auto idx = make_tuple(idx0, idx1);
                    auto& dst_ref = c_block_window[idx];
                    const auto src_val = c_block_tile[idx];
                    
                    if constexpr(std::is_same_v<CDataType, float>) {
                        atomicAdd(&dst_ref, src_val);
                    } else if constexpr(std::is_same_v<CDataType, half>) {
                        // Use half precision atomic add (if supported)
                        #if __HIP_DEVICE_COMPILE__
                        atomicAdd(reinterpret_cast<__half*>(&dst_ref), 
                                static_cast<__half>(src_val));
                        #endif
                    } else if constexpr(std::is_same_v<CDataType, bf16_t>) {
                        // Use bfloat16 atomic add (if supported)
                        #if __HIP_DEVICE_COMPILE__
                        atomicAdd(reinterpret_cast<__hip_bfloat16*>(&dst_ref), 
                                static_cast<__hip_bfloat16>(src_val));
                        #endif
                    } else {
                        // Fallback for other data types using compare-and-swap
                        AtomicAddFallback(&dst_ref, src_val);
                    }
                });
            });
        }

        /**
         * @brief Fallback atomic accumulation using compare-and-swap
         */
        template <typename T>
        CK_TILE_DEVICE void AtomicAddFallback(T* address, T val) const
        {
            T old = *address;
            T assumed;
            do {
                assumed = old;
                old = atomicCAS(address, assumed, assumed + val);
            } while (assumed != old);
        }

        /**
         * @brief Store results with optional atomic accumulation
         */
        template <typename CBlockTile, typename CBlockWindow>
        CK_TILE_DEVICE void StoreResults(CBlockWindow& c_block_window,
                                        const CBlockTile& c_block_tile,
                                        bool use_atomic_add = false) const
        {
            if (use_atomic_add) {
                AtomicAccumulate(c_block_window, c_block_tile);
            } else {
                store_tile(c_block_window, c_block_tile);
            }
        }
    };

    /**
     * @brief Stream-K aware pipeline execution
     */
    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename AElementFunction,
              typename BElementFunction>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const AElementFunction& a_element_func,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const BElementFunction& b_element_func,
                                   index_t num_loop,
                                   void* p_smem,
                                   bool is_partial_tile = false) const
    {
        if (is_partial_tile) {
            // Use Stream-K implementation for partial tiles
            return StreamKPipelineImpl<Base::Scheduler>{}.template RunPartialK<Base::HasHotLoop, Base::TailNum>(
                a_dram_block_window_tmp,
                a_element_func,
                b_dram_block_window_tmp,
                b_element_func,
                0,
                num_loop * BlockGemmShape::kK,
                p_smem);
        } else {
            // Use standard implementation for full tiles
            return Base::operator()(a_dram_block_window_tmp,
                                  a_element_func,
                                  b_dram_block_window_tmp,
                                  b_element_func,
                                  num_loop,
                                  p_smem);
        }
    }

    /**
     * @brief Extended operator for K-range specification
     */
    template <typename ADramBlockWindowTmp, typename BDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   index_t k_start,
                                   index_t k_end,
                                   void* p_smem) const
    {
        constexpr auto PassThrough = [](const auto& x) { return x; };
        return StreamKPipelineImpl<Base::Scheduler>{}.template RunPartialK<Base::HasHotLoop, Base::TailNum>(
            a_dram_block_window_tmp,
            PassThrough,
            b_dram_block_window_tmp,
            PassThrough,
            k_start,
            k_end,
            p_smem);
    }

    /**
     * @brief Stream-K work decomposition operator
     */
    template <typename ADramBlockWindowTmp, typename BDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   index_t work_start,
                                   index_t work_end,
                                   index_t num_tile_m,
                                   index_t num_tile_n,
                                   index_t num_tile_k,
                                   void* p_smem) const
    {
        constexpr auto PassThrough = [](const auto& x) { return x; };
        return StreamKPipelineImpl<Base::Scheduler>{}.template RunMultipleKSlices<Base::HasHotLoop, Base::TailNum>(
            a_dram_block_window_tmp,
            PassThrough,
            b_dram_block_window_tmp,
            PassThrough,
            work_start,
            work_end,
            num_tile_m,
            num_tile_n,
            num_tile_k,
            p_smem);
    }
};

/**
 * @brief Stream-K problem traits with additional Stream-K specific parameters
 */
template <bool kPadM_,
          bool kPadN_,
          bool kPadK_,
          bool DoubleSmemBuffer_,
          typename ALayout_,
          typename BLayout_,
          typename CLayout_,
          bool TransposeC_            = false,
          bool UseStructuredSparsity_ = false,
          bool UsePersistentKernel_   = false,
          index_t NumWaveGroups_      = 1,
          index_t StreamKFactor_      = 4>  // New: Stream-K splitting factor
struct StreamKTileGemmTraits
{
    static constexpr bool kPadM = kPadM_;
    static constexpr bool kPadN = kPadN_;
    static constexpr bool kPadK = kPadK_;

    static constexpr bool DoubleSmemBuffer = DoubleSmemBuffer_;

    using ALayout = ALayout_;
    using BLayout = BLayout_;
    using CLayout = CLayout_;

    static constexpr bool TransposeC            = TransposeC_;
    static constexpr bool UseStructuredSparsity = UseStructuredSparsity_;
    static constexpr bool UsePersistentKernel   = UsePersistentKernel_;
    static constexpr index_t NumWaveGroups      = NumWaveGroups_;
    
    // Stream-K specific traits
    static constexpr index_t StreamKFactor      = StreamKFactor_;
    static constexpr bool EnableStreamK         = true;
    static constexpr bool RequireAtomicAdd      = true;
};

/**
 * @brief Stream-K epilogue pipeline with atomic accumulation support
 */
template <typename Problem, typename Policy>
struct StreamKEpiloguePipeline
{
    using CDataType = typename Problem::CDataType;
    
    static constexpr auto MemoryOperation = memory_operation_enum::atomic_add;
    
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 0; }
    
    template <typename CBlockWindow, typename CBlockTile, typename DBlockWindow>
    CK_TILE_DEVICE void operator()(CBlockWindow& c_block_window,
                                   const CBlockTile& c_block_tile,
                                   const DBlockWindow& d_block_window,
                                   void* smem_ptr,
                                   bool use_atomic = true) const
    {
        if (use_atomic) {
            // Atomic accumulation for Stream-K partial results
            constexpr auto c_spans = CBlockTile::get_distributed_spans();
            
            sweep_tile_span(c_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(c_spans[number<1>{}], [&](auto idx1) {
                    auto idx = make_tuple(idx0, idx1);
                    auto& dst_ref = c_block_window[idx];
                    const auto src_val = c_block_tile[idx];
                    
                    if constexpr(std::is_same_v<CDataType, float>) {
                        atomicAdd(&dst_ref, src_val);
                    } else if constexpr(std::is_same_v<CDataType, half>) {
                        #if __HIP_DEVICE_COMPILE__
                        atomicAdd(reinterpret_cast<__half*>(&dst_ref), 
                                static_cast<__half>(src_val));
                        #endif
                    } else if constexpr(std::is_same_v<CDataType, bf16_t>) {
                        #if __HIP_DEVICE_COMPILE__
                        atomicAdd(reinterpret_cast<__hip_bfloat16*>(&dst_ref), 
                                static_cast<__hip_bfloat16>(src_val));
                        #endif
                    }
                });
            });
        } else {
            // Standard store operation
            store_tile(c_block_window, c_block_tile);
        }
    }
};

/**
 * @brief Host-side Stream-K configuration helper
 */
struct StreamKConfig
{
    CK_TILE_HOST static auto CalculateOptimalBlocks(index_t M, 
                                                    index_t N, 
                                                    index_t K,
                                                    index_t MPerBlock,
                                                    index_t NPerBlock, 
                                                    index_t KPerBlock,
                                                    index_t available_sms) -> tuple<index_t, index_t, index_t, bool>
    {
        const index_t num_tile_m = (M + MPerBlock - 1) / MPerBlock;
        const index_t num_tile_n = (N + NPerBlock - 1) / NPerBlock;
        const index_t num_tile_k = (K + KPerBlock - 1) / KPerBlock;
        
        const index_t total_output_tiles = num_tile_m * num_tile_n;
        const index_t total_work_units = total_output_tiles * num_tile_k;
        
        // Decide whether to use Stream-K
        const bool use_stream_k = (num_tile_k > 2) && 
                                 (total_work_units > total_output_tiles * 2) &&
                                 (total_work_units > 64);
        
        if (!use_stream_k) {
            return make_tuple(total_output_tiles, index_t(0), index_t(0), false);
        }
        
        // Calculate optimal number of blocks for Stream-K
        const index_t max_useful_blocks = min(total_work_units, available_sms * 4);
        const index_t optimal_blocks = min(max_useful_blocks, total_work_units);
        
        const index_t work_per_block = total_work_units / optimal_blocks;
        const index_t big_blocks = total_work_units % optimal_blocks;
        
        return make_tuple(optimal_blocks, work_per_block, big_blocks, true);
    }
    
    CK_TILE_HOST static bool ShouldUseStreamK(index_t M, 
                                              index_t N, 
                                              index_t K,
                                              index_t MPerBlock,
                                              index_t NPerBlock, 
                                              index_t KPerBlock)
    {
        const auto [blocks, work_per_block, big_blocks, use_streamk] = 
            CalculateOptimalBlocks(M, N, K, MPerBlock, NPerBlock, KPerBlock, 1024);
        return use_streamk;
    }
};

} // namespace ck_tile