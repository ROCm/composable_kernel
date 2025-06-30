// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"

namespace ck_tile {

// Global static barrier array - initialized once at program startup
// constexpr index_t MAX_STATIC_TILES = 4;  // Adjust based on max grid size
// __device__ static uint32_t global_static_barriers[MAX_STATIC_TILES] = {0};  // Initialize to zeros

template <typename ADataType_,
          typename BDataType_,
          typename AccDataType_,
          typename ODataType_,
          typename CLayout_,
          index_t kBlockSize_,
          index_t kM_,
          index_t kN_,
          index_t kMWave_,
          index_t kNWave_,
          index_t kMPerXdl_,
          index_t kNPerXdl_,
          index_t kKPerXdl_,
          bool isCTransposed_,
          memory_operation_enum MemoryOperation_,
          bool Zeroing_ = false>
struct CShuffleEpilogueProblem
{
    using ADataType                                        = remove_cvref_t<ADataType_>;
    using BDataType                                        = remove_cvref_t<BDataType_>;
    using AccDataType                                      = remove_cvref_t<AccDataType_>;
    using ODataType                                        = remove_cvref_t<ODataType_>;
    using CLayout                                          = remove_cvref_t<CLayout_>;
    static constexpr index_t kBlockSize                    = kBlockSize_;
    static constexpr index_t kMPerBlock                    = kM_;
    static constexpr index_t kNPerBlock                    = kN_;
    static constexpr index_t kMWave                        = kMWave_;
    static constexpr index_t kNWave                        = kNWave_;
    static constexpr index_t kMPerXdl                      = kMPerXdl_;
    static constexpr index_t kNPerXdl                      = kNPerXdl_;
    static constexpr index_t kKPerXdl                      = kKPerXdl_;
    static constexpr index_t isCTransposed                 = isCTransposed_;
    static constexpr memory_operation_enum MemoryOperation = MemoryOperation_;
    static constexpr bool EnableZeroing = Zeroing_;
};

template <typename Problem_, typename Policy_ = void>
struct CShuffleEpilogue
{
    using Problem     = remove_cvref_t<Problem_>;
    using ADataType   = remove_cvref_t<typename Problem::ADataType>;
    using BDataType   = remove_cvref_t<typename Problem::BDataType>;
    using AccDataType = remove_cvref_t<typename Problem::AccDataType>;
    using ODataType   = remove_cvref_t<typename Problem::ODataType>;
    // Used for weight-only quantization kernel, B would be dequantized to the same data type as A
    using BTypeToUse =
        std::conditional_t<std::is_same_v<BDataType, pk_int4_t>, ADataType, BDataType>;
    using CLayout = remove_cvref_t<typename Problem::CLayout>;
    static constexpr memory_operation_enum MemoryOperation = Problem::MemoryOperation;
    static constexpr index_t kBlockSize                    = Problem::kBlockSize;
    static constexpr index_t kMPerBlock                    = Problem::kMPerBlock;
    static constexpr index_t kNPerBlock                    = Problem::kNPerBlock;
    static constexpr index_t kMWave                        = Problem::kMWave;
    static constexpr index_t kNWave                        = Problem::kNWave;
    static constexpr index_t kMPerXdl                      = Problem::kMPerXdl;
    static constexpr index_t kNPerXdl                      = Problem::kNPerXdl;
    static constexpr index_t kKPerXdl                      = Problem::kKPerXdl;
    static constexpr index_t isCTransposed                 = Problem::isCTransposed;
    static constexpr index_t kMPerIteration                = kMPerXdl * kMWave;
    static constexpr index_t kNPerIteration                = kNPerXdl * kNWave;

    using WG = WarpGemmMfmaDispatcher<ADataType,
                                      BTypeToUse,
                                      AccDataType,
                                      kMPerXdl,
                                      kNPerXdl,
                                      kKPerXdl,
                                      isCTransposed>;

    using CWarpDstr   = typename WG::CWarpDstr;
    using CWarpTensor = typename WG::CWarpTensor;

    /**
     * @brief Get the vector store size for C tensor.
     *
     * @note The vector store size for output C tensor would depend on multiple factors
     *       like its data layout and warp gemm C transposition. In general it would
     *       be the number of consecutive elements in contiguous C dimension hold by
     *       single thread.
     *
     * @return The vector store size for C tensor.
     */
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeC()
    {
        constexpr index_t MaxVectorStoreSize = 16;
        if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
        {
            return std::min(static_cast<int>(kNPerIteration),
                            static_cast<int>(MaxVectorStoreSize / sizeof(ODataType)));
        }
        else if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::ColumnMajor>)
        {
            return std::min(static_cast<int>(kMPerIteration),
                            static_cast<int>(MaxVectorStoreSize / sizeof(ODataType)));
        }
        else
        {
            static_assert(false, "Unsupported CLayout!");
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsBlockDescriptor()
    {
        // N is contiguous dimension
        if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
        {
            return make_naive_tensor_descriptor(
                make_tuple(number<kMWave * kMPerXdl>{}, number<kNWave * kNPerXdl>{}),
                make_tuple(number<kNWave * kNPerXdl>{}, number<1>{}));
        }
        // M is contiguous dimension
        else if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::ColumnMajor>)
        {
            return make_naive_tensor_descriptor(
                make_tuple(number<kMWave * kMPerXdl>{}, number<kNWave * kNPerXdl>{}),
                make_tuple(number<1>{}, number<kMWave * kMPerXdl>{}));
        }
        else
        {
            static_assert(false, "Unsupported CLayout!");
        }
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return kMWave * kNWave * kMPerXdl * kNPerXdl * sizeof(ODataType);
    }


    template <typename ODramWindow, typename OAccTile>
    CK_TILE_DEVICE auto operator()(ODramWindow& out_dram_window, 
                                const OAccTile& o_acc_tile, 
                                void* p_smem,
                                uint32_t* cleared_c_tile_barrier = nullptr,
                                uint32_t* updated_batches_barrier = nullptr,
                                uint64_t* epilogue_timing = nullptr)
    {
        // Timing variables for detailed epilogue analysis
        
        const index_t iMWarp = get_warp_id() / kNWave;
        const index_t iNWarp = get_warp_id() - iMWarp * kNWave;

        constexpr auto lds_block_desc = MakeLdsBlockDescriptor<Problem>();
        auto o_lds_block              = make_tensor_view<address_space_enum::lds>(
            static_cast<ODataType*>(p_smem), lds_block_desc);
        auto in_lds_window =
            make_tile_window(o_lds_block,
                             make_tuple(number<kMPerXdl>{}, number<kNPerXdl>{}),
                             {number<kMPerXdl>{} * iMWarp, number<kNPerXdl>{} * iNWarp});
        auto out_lds_window =
            make_tile_window(o_lds_block,
                             make_tuple(number<kMWave * kMPerXdl>{}, number<kNWave * kNPerXdl>{}),
                             {0, 0});
        
        
        using SFC                    = space_filling_curve<sequence<kMPerBlock, kNPerBlock>,
                                        sequence<0, 1>,
                                        sequence<kMPerXdl * kMWave, kNPerXdl * kNWave>>;
        constexpr index_t num_access = SFC::get_num_of_access();

        using TileEncodingPattern =
            TileDistributionEncodingPattern2D<kBlockSize,
                                              kMPerIteration,
                                              kNPerIteration,
                                              GetVectorSizeC(),
                                              tile_distribution_pattern::thread_raked>;
        constexpr auto dram_tile_distribution = TileEncodingPattern::Make2DStaticTileDistribution();

        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};
        
        
        CWarpTensor c_warp_in_tensor;
        static_for<0, num_access, 1>{}([&](auto iAccess) {
           
            
            constexpr auto idx_y_start = SFC::get_index(iAccess);

            constexpr auto mIter = number<idx_y_start.at(number<0>{}) / (kMPerXdl * kMWave)>{};
            constexpr auto nIter = number<idx_y_start.at(number<1>{}) / (kNPerXdl * kNWave)>{};

            c_warp_in_tensor.get_thread_buffer() = o_acc_tile.get_y_sliced_thread_data(
                merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

            const auto c_warp_in_tensor_casted = cast_tile<ODataType>(c_warp_in_tensor);
            
           
            block_sync_lds();

           
            store_tile(in_lds_window, c_warp_in_tensor_casted);

           
            block_sync_lds();

           
            const auto c_out_tensor =
                load_tile(make_tile_window(out_lds_window, dram_tile_distribution));

            
            if constexpr(Problem::EnableZeroing)
            {
                uint64_t barrier_wait_start = 0, barrier_wait_end = 0;
                uint64_t atomic_add_start = 0, atomic_add_end = 0;
                uint64_t barrier_update_start = 0, barrier_update_end = 0;

                // Wait for C tile to be zeroed before first access
                if constexpr(iAccess == 0) {
                    if(cleared_c_tile_barrier != nullptr) {
                        if(threadIdx.x == 0 && epilogue_timing != nullptr) {
                            barrier_wait_start = __builtin_amdgcn_s_memrealtime();
                        }
                        if(threadIdx.x == 0) {
                            while( __atomic_load_n(&cleared_c_tile_barrier[blockIdx.x], __ATOMIC_ACQUIRE)== 0) {
                                __builtin_amdgcn_s_sleep(1);
                            }
                        }
                        __syncthreads();
                        if(threadIdx.x == 0 && epilogue_timing != nullptr) {
                            barrier_wait_end = __builtin_amdgcn_s_memrealtime();
                        }
                    }
                }

           
                if(threadIdx.x == 0 && epilogue_timing != nullptr) {
                    atomic_add_start = __builtin_amdgcn_s_memrealtime();
                }
    
                
                // All k-batches use atomic add (since C is already zeroed)
                update_tile(out_dram_window, c_out_tensor);

                if(threadIdx.x == 0 && epilogue_timing != nullptr) {
                    atomic_add_end = __builtin_amdgcn_s_memrealtime();
                }

                
                // After last access, increment completed batches counter
                if constexpr(iAccess == num_access - 1) {
                    
                    if(updated_batches_barrier != nullptr && threadIdx.x == 0) {
                        if(epilogue_timing != nullptr) {
                            barrier_update_start = __builtin_amdgcn_s_memrealtime();
                        }

                        uint32_t completed = __atomic_fetch_add(&updated_batches_barrier[blockIdx.x], 1, __ATOMIC_RELEASE) + 1;
                        // If all k-batches completed, reset for next tile
                        if(completed >= static_cast<uint32_t>(gridDim.z)) {
                            __atomic_store_n(&cleared_c_tile_barrier[blockIdx.x], 0, __ATOMIC_RELEASE);
                            __atomic_store_n(&updated_batches_barrier[blockIdx.x], 0, __ATOMIC_RELEASE);
                            // printf("Barriers Reset. tile:%u \n", blockIdx.x);
                        }

                        if(epilogue_timing != nullptr) {
                            barrier_update_end = __builtin_amdgcn_s_memrealtime();
                        }
                    }
                }

                // Store barrier timing (only for block 4, like your other timing)
                if(threadIdx.x == 0 && epilogue_timing != nullptr && blockIdx.x == 4) {
                    const double freq_mhz = 25.0;
                    
                    // Store timing for different barrier operations
                    if constexpr(iAccess == 0) {
                        // Barrier wait timing
                        const double barrier_wait_us = (barrier_wait_end - barrier_wait_start) / freq_mhz;
                        epilogue_timing[0] = barrier_wait_us * 1000; 
                    }
                    
                    // Atomic add timing 
                    const double atomic_add_us = (atomic_add_end - atomic_add_start) / freq_mhz;
                    epilogue_timing[1] += atomic_add_us * 1000; 
                    
                    if constexpr(iAccess == num_access - 1) {
                        // Barrier update timing
                        const double barrier_update_us = (barrier_update_end - barrier_update_start) / freq_mhz;
                        epilogue_timing[2] = barrier_update_us * 1000; 
                    }
                }
            }
            else
            {
                if constexpr(MemoryOperation == memory_operation_enum::set)
                {
                    store_tile(out_dram_window, c_out_tensor);
                }
                else
                {
                    update_tile(out_dram_window, c_out_tensor);
                }
            }

           
            if constexpr(iAccess != num_access - 1)
            {
                constexpr auto step = SFC::get_forward_step(iAccess);
                move_tile_window(out_dram_window, {step.at(number<0>{}), step.at(number<1>{})});
            }
        });

       
    }
};
} // namespace ck_tile
