// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/epilogue/chainer/epilogue_policy.hpp"

namespace ck_tile {

/// @brief Policy based multiple epilogue chainer for executing operations sequentially with shared runtime-context and execution-context
///
/// This class provides a framework for executing a sequence of epilogue operations
/// with initialization epilogue, tuple of main epilogues and some flexibility on execution policies.
/// Each stage can share context data efficiently without re-computation.
template <typename InitEpilogue, typename MainEpilogueTuple, typename Policy>
class EpilogueChainer
{
    static_assert(MainEpilogueTuple::size() >= 1,
                  "EpilogueChainer requires at least 1 main epilogue");

    private:
    static constexpr index_t NMainEpilogues = MainEpilogueTuple::size();

    template <index_t I>
    using MainEpilogueAt = typename std::tuple_element<I, MainEpilogueTuple>::type;

    /// @brief Compute maximum shared memory requirement across all stages
    static constexpr index_t ComputeMaxSmemSize()
    {
        index_t max_size = InitEpilogue::GetSmemSize();

        static_for<0, NMainEpilogues, 1>{}([&](auto I) {
            using Epilogue = MainEpilogueAt<I>;
            max_size       = max(max_size, Epilogue::GetSmemSize());
        });

        return max_size;
    }

    /// @brief Execute stages sequentially (one pass)
    template <typename ODramWindow, typename OAccTile, typename DsDramWindows, typename ExecutionContext>
    CK_TILE_DEVICE static auto ExecuteSequential(ODramWindow& out_dram_window,
                                                 const OAccTile& o_acc_tile,
                                                 const DsDramWindows& ds_dram_windows,
                                                 ExecutionContext& exec_context)
    {
        // Execute initialization stage
        auto context = InitEpilogue{}(out_dram_window, o_acc_tile, ds_dram_windows, exec_context.smem_ptr);

        if constexpr(Policy::SyncPolicy == sync_policy_enum::after_stage)
        {
            ExecutionContext::InsertBarrierIfNeeded();
        }

        // Execute main stages sequentially
        static_for<0, NMainEpilogues, 1>{}([&](auto I) {

            if constexpr(Policy::SyncPolicy == sync_policy_enum::before_stage)
            {
                ExecutionContext::InsertBarrierIfNeeded();
            }

            using Epilogue = MainEpilogueAt<I>;
            Epilogue{}(out_dram_window, o_acc_tile, ds_dram_windows, exec_context.smem_ptr, context);

            if constexpr(Policy::SyncPolicy == sync_policy_enum::after_stage)
            {
                ExecutionContext::InsertBarrierIfNeeded();
            }
        });   
    }

    /// @brief Execute stages within access loop
    template <typename ODramWindow, typename OAccTile, typename DsDramWindows, typename ExecutionContext>
    CK_TILE_DEVICE static auto ExecuteInLoop(ODramWindow& out_dram_window,
                                             const OAccTile& o_acc_tile,
                                             const DsDramWindows& ds_dram_windows,
                                             ExecutionContext& exec_context)
    {
        // Execute initialization stage once
        auto context = InitEpilogue{}(out_dram_window, o_acc_tile, ds_dram_windows, exec_context.smem_ptr);

        if constexpr(Policy::SyncPolicy == sync_policy_enum::after_stage)
        {
            ExecutionContext::InsertBarrierIfNeeded();
        }

        // Execute main stages within access loop
        constexpr index_t num_access = SelectEpilogue::SFC::get_num_of_access();
        static_for<0, num_access, 1>{}([&](auto iAccess) {
            exec_context.current_sfc_step = iAccess;

            static_for<0, NMainEpilogues, 1>{}([&](auto I) {

                if constexpr(Policy::SyncPolicy == sync_policy_enum::before_stage)
                {
                    ExecutionContext::InsertBarrierIfNeeded();
                }

                using Epilogue = MainEpilogueAt<I>;
                Epilogue{}(out_dram_window, o_acc_tile, ds_dram_windows, exec_context.smem_ptr, iAccess, context);

                if constexpr(Policy::SyncPolicy == sync_policy_enum::after_stage)
                {
                    ExecutionContext::InsertBarrierIfNeeded();
                }
            });
        });

    }

    /// @brief Execute with parameterized arguments (sequential)
    template <typename ODramWindow,
              typename OAccTile,
              typename DsDramWindows,
              typename ExecutionContext,
              typename InitArgs,
              typename MainArgsTuple>
    CK_TILE_DEVICE static auto ExecuteSequential(ODramWindow& out_dram_window,
                                                 const OAccTile& o_acc_tile,
                                                 const DsDramWindows& ds_dram_windows,
                                                 ExecutionContext& exec_context,
                                                 const InitArgs& init_args,
                                                 const MainArgsTuple& main_args_tuple)
    {
        // Execute initialization with arguments
        auto context = ck_tile::apply(
            [&](auto&&... unpacked_args) {
                return InitEpilogue{}(out_dram_window,
                                      o_acc_tile,
                                      ds_dram_windows,
                                      exec_context.smem_ptr,
                                      std::forward<decltype(unpacked_args)>(unpacked_args)...);
            },
            init_args);

        if constexpr(Policy::SyncPolicy == sync_policy_enum::after_stage)
        {
            ExecutionContext::InsertBarrierIfNeeded();
        }

        // Execute main stages with their respective arguments
        static_for<0, NMainEpilogues, 1>{}([&](auto I) {

            if constexpr(Policy::SyncPolicy == sync_policy_enum::before_stage)
            {
                ExecutionContext::InsertBarrierIfNeeded();
            }

            using Epilogue   = MainEpilogueAt<I>;
            const auto& args = main_args_tuple.template get<I>();
            ck_tile::apply(
                [&](auto&&... unpacked_args) {
                    Epilogue{}(out_dram_window,
                               o_acc_tile,
                               ds_dram_windows,
                               exec_context.smem_ptr,
                               context,
                               std::forward<decltype(unpacked_args)>(unpacked_args)...);
                },
                args);

            if constexpr(Policy::SyncPolicy == sync_policy_enum::after_stage)
            {
                ExecutionContext::InsertBarrierIfNeeded();
            }
        });
    }

    /// @brief Execute with parameterized arguments (in loop)
    template <typename ODramWindow,
              typename OAccTile,
              typename DsDramWindows,
              typename ExecutionContext,
              typename InitArgs,
              typename MainArgsTuple>
    CK_TILE_DEVICE static auto ExecuteInLoop(ODramWindow& out_dram_window,
                                             const OAccTile& o_acc_tile,
                                             const DsDramWindows& ds_dram_windows,
                                             ExecutionContext& exec_context,
                                             const InitArgs& init_args,
                                             const MainArgsTuple& main_args_tuple)
    {
        // Execute initialization with arguments
        auto context = ck_tile::apply(
            [&](auto&&... unpacked_args) {
                return InitEpilogue{}(out_dram_window,
                                      o_acc_tile,
                                      ds_dram_windows,
                                      exec_context.smem_ptr,
                                      std::forward<decltype(unpacked_args)>(unpacked_args)...);
            },
            init_args);

        if constexpr(Policy::SyncPolicy == sync_policy_enum::after_stage)
        {
            ExecutionContext::InsertBarrierIfNeeded();
        }

        // Execute main stages for each access with arguments
        constexpr index_t num_access = SelectEpilogue::SFC::get_num_of_access();
        static_for<0, num_access, 1>{}([&](auto iAccess) {
            exec_context.current_sfc_step = iAccess;

            static_for<0, NMainEpilogues, 1>{}([&](auto I) {
                if constexpr(Policy::SyncPolicy == sync_policy_enum::before_stage)
                {
                    ExecutionContext::InsertBarrierIfNeeded();
                }

                using Epilogue   = MainEpilogueAt<I>;
                const auto& args = main_args_tuple.template get<I>();
                ck_tile::apply(
                    [&](auto&&... unpacked_args) {
                        Epilogue{}(out_dram_window,
                                   o_acc_tile,
                                   ds_dram_windows,
                                   exec_context.smem_ptr,
                                   iAccess,
                                   context,
                                   std::forward<decltype(unpacked_args)>(unpacked_args)...);
                    },
                    args);
                if constexpr(Policy::SyncPolicy == sync_policy_enum::after_stage)
                {
                    ExecutionContext::InsertBarrierIfNeeded();
                }
            });
        });
    }

    public:

    using SelectEpilogue = InitEpilogue;
    using Problem        = typename SelectEpilogue::Problem;
    using ODataType      = typename SelectEpilogue::ODataType;
    using DsDataType     = typename SelectEpilogue::DsDataType;
    using DsLayout       = typename SelectEpilogue::DsLayout;
    using AccDataType    = typename SelectEpilogue::AccDataType;
    using Policy_ = Policy;
    using ExecutionContext = EpilogueExecutionContext<Problem, Policy>;

    static constexpr auto MemoryOperation = SelectEpilogue::MemoryOperation;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return ComputeMaxSmemSize(); }

    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeC()
    {
        return SelectEpilogue::GetVectorSizeC();
    }

    template <index_t I>
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeD(number<I> idx)
    {
        return SelectEpilogue::GetVectorSizeD(idx);
    }

    CK_TILE_DEVICE static constexpr auto MakeLdsDistributionEncode()
    {
        return SelectEpilogue::MakeLdsDistributionEncode();
    }

    // ========================================
    // Main execution interfaces
    // ========================================

    /// @brief Simple execution without arguments
    template <typename ODramWindow, typename OAccTile, typename DsDramWindows>
    CK_TILE_DEVICE auto operator()(ODramWindow& out_dram_window,
                                   const OAccTile& o_acc_tile,
                                   const DsDramWindows& ds_dram_windows,
                                   void* p_smem) const -> void
    {
        ExecutionContext exec_context;
        exec_context.smem_ptr = p_smem;

        if constexpr(ExecutionContext::execution_mode == execution_mode_enum::sequential)
        {
            ExecuteSequential(out_dram_window, o_acc_tile, ds_dram_windows, exec_context);
        }
        else // in_loop
        {
            ExecuteInLoop(out_dram_window, o_acc_tile, ds_dram_windows, exec_context);
        }
    }



    /// @brief Execution with arguments
    template <typename ODramWindow,
              typename OAccTile,
              typename DsDramWindows,
              typename InitArgs,
              typename MainArgsTuple>
    CK_TILE_DEVICE auto operator()(ODramWindow& out_dram_window,
                                   const OAccTile& o_acc_tile,
                                   const DsDramWindows& ds_dram_windows,
                                   void* p_smem,
                                   const InitArgs& init_args,
                                   const MainArgsTuple& main_args) const -> void
    {
        ExecutionContext exec_context;
        exec_context.smem_ptr = p_smem;

        if constexpr(ExecutionContext::execution_mode == execution_mode_enum::sequential)
        {
            ExecuteSequential(out_dram_window, o_acc_tile, ds_dram_windows, 
                                    exec_context, init_args, main_args);
        }
        else // in_loop
        {
            ExecuteInLoop(out_dram_window, o_acc_tile, ds_dram_windows, 
                                exec_context, init_args, main_args);
        }
    }
};

} // namespace ck_tile
