// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

/// @brief Execution mode policy - determines how epilogue stages are executed
enum class execution_mode_enum
{
    sequential, ///< Execute stages one after another 
    in_loop     ///< Execute stages within access loop 
};

/// @brief Synchronization requirement policy
enum class sync_policy_enum
{
    none,         ///< No synchronization barriers needed
    before_stage, ///< Insert barrier before each stage
    after_stage   ///< Insert barrier after each stage  
};



/// @brief Main policy template that combines execution and synchronization policies
template <execution_mode_enum ExecutionMode_ = execution_mode_enum::sequential,
          sync_policy_enum SyncPolicy_ = sync_policy_enum::none>
struct EpiloguePolicy
{
    static constexpr execution_mode_enum ExecutionMode = ExecutionMode_;
    static constexpr sync_policy_enum SyncPolicy = SyncPolicy_;

    /// @brief Get execution mode
    template <typename Problem>
    static constexpr execution_mode_enum GetExecutionMode()
    {
        return ExecutionMode;
    }

    /// @brief Check if barriers are required based on sync policy
    template <typename Problem>  
    static constexpr bool RequiresBarrier()
    {
        return (SyncPolicy == sync_policy_enum::before_stage) || 
               (SyncPolicy == sync_policy_enum::after_stage);
    }
};


/// @brief Policy-aware execution context that holds runtime parameters
template <typename Problem, typename Policy>
struct EpilogueExecutionContext
{
    using Problem_ = Problem;
    using Policy_ = Policy;
    
    static constexpr execution_mode_enum execution_mode = Policy::ExecutionMode;
    static constexpr bool requires_barrier = Policy::template RequiresBarrier<Problem>();
    
    // Runtime parameters can be stored here
    void* smem_ptr = nullptr;
    index_t current_sfc_step = 0;
    
    /// @brief Insert barrier if policy requires it
    CK_TILE_DEVICE static void InsertBarrierIfNeeded()
    {
        if constexpr(requires_barrier)
        {
            __syncthreads();
        }
    }
    
    /// @brief Get current execution mode for runtime decisions
    CK_TILE_DEVICE static constexpr execution_mode_enum GetExecutionMode()
    {
        return execution_mode;
    }
    
};

} // namespace ck_tile
