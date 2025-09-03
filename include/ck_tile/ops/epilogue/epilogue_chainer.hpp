// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"

namespace ck_tile {

/// @brief Chains multiple epilogue operations sequentially with shared context
///
/// This class provides a framework for executing a sequence of epilogue operations
/// with initialization, main processing, and optional finalization stages.
/// Each stage can share context data efficiently without re-computation.
template <typename InitEpilogue, typename MainEpilogueTuple, typename FinalEpilogue = void>
class EpilogueChainer
{
    static_assert(MainEpilogueTuple::size() >= 1,
                  "EpilogueChainer requires at least 1 main epilogue");

    private:
    static constexpr index_t NMainEpilogues = MainEpilogueTuple::size();
    static constexpr bool HasFinalEpilogue  = !std::is_same_v<FinalEpilogue, void>;

    template <index_t I>
    using MainEpilogueAt = typename std::tuple_element<I, MainEpilogueTuple>::type;

    static constexpr index_t ComputeMaxSmemSize()
    {
        index_t max_size = InitEpilogue::GetSmemSize();

        static_for<0, NMainEpilogues, 1>{}([&](auto I) {
            using Epilogue = MainEpilogueAt<I>;
            max_size       = max(max_size, Epilogue::GetSmemSize());
        });

        if constexpr(HasFinalEpilogue)
        {
            max_size = max(max_size, FinalEpilogue::GetSmemSize());
        }

        return max_size;
    }

    template <typename ODramWindow, typename OAccTile, typename DsDramWindows>
    CK_TILE_DEVICE static auto ExecuteSequential(ODramWindow& out_dram_window,
                                                 const OAccTile& o_acc_tile,
                                                 const DsDramWindows& ds_dram_windows,
                                                 void* p_smem)
    {
        auto context = InitEpilogue{}(out_dram_window, o_acc_tile, ds_dram_windows, p_smem);

        static_for<0, NMainEpilogues, 1>{}([&](auto I) {
            using Epilogue = MainEpilogueAt<I>;
            Epilogue{}(out_dram_window, o_acc_tile, ds_dram_windows, p_smem, context);
        });

        if constexpr(HasFinalEpilogue)
        {
            FinalEpilogue{}(out_dram_window, o_acc_tile, ds_dram_windows, p_smem, context);
        }
    }

    template <typename ODramWindow, typename OAccTile, typename DsDramWindows>
    CK_TILE_DEVICE static auto ExecuteInLoop(ODramWindow& out_dram_window,
                                             const OAccTile& o_acc_tile,
                                             const DsDramWindows& ds_dram_windows,
                                             void* p_smem)
    {
        auto context = InitEpilogue{}(out_dram_window, o_acc_tile, ds_dram_windows, p_smem);

        constexpr index_t num_access = SelectEpilogue::SFC::get_num_of_access();
        static_for<0, num_access, 1>{}([&](auto iAccess) {
            static_for<0, NMainEpilogues, 1>{}([&](auto I) {
                using Epilogue = MainEpilogueAt<I>;
                Epilogue{}(out_dram_window, o_acc_tile, ds_dram_windows, p_smem, iAccess, context);
            });
        });

        if constexpr(HasFinalEpilogue)
        {
            FinalEpilogue{}(out_dram_window, o_acc_tile, ds_dram_windows, p_smem, context);
        }
    }

    template <typename ODramWindow,
              typename OAccTile,
              typename DsDramWindows,
              typename InitArgs,
              typename MainArgsTuple,
              typename FinalArgs>
    CK_TILE_DEVICE static auto ExecuteSequential(ODramWindow& out_dram_window,
                                                 const OAccTile& o_acc_tile,
                                                 const DsDramWindows& ds_dram_windows,
                                                 void* p_smem,
                                                 const InitArgs& init_args,
                                                 const MainArgsTuple& main_args_tuple,
                                                 const FinalArgs& final_args)
    {
        auto context = ck_tile::apply(
            [&](auto&&... unpacked_args) {
                return InitEpilogue{}(out_dram_window,
                                      o_acc_tile,
                                      ds_dram_windows,
                                      p_smem,
                                      std::forward<decltype(unpacked_args)>(unpacked_args)...);
            },
            init_args);

        static_for<0, NMainEpilogues, 1>{}([&](auto I) {
            using Epilogue   = MainEpilogueAt<I>;
            const auto& args = main_args_tuple.template get<I>();
            ck_tile::apply(
                [&](auto&&... unpacked_args) {
                    Epilogue{}(out_dram_window,
                               o_acc_tile,
                               ds_dram_windows,
                               p_smem,
                               context,
                               std::forward<decltype(unpacked_args)>(unpacked_args)...);
                },
                args);
        });

        if constexpr(HasFinalEpilogue)
        {
            ck_tile::apply(
                [&](auto&&... unpacked_args) {
                    FinalEpilogue{}(out_dram_window,
                                    o_acc_tile,
                                    ds_dram_windows,
                                    p_smem,
                                    context,
                                    std::forward<decltype(unpacked_args)>(unpacked_args)...);
                },
                final_args);
        }
    }

    template <typename ODramWindow,
              typename OAccTile,
              typename DsDramWindows,
              typename InitArgs,
              typename MainArgsTuple,
              typename FinalArgs>
    CK_TILE_DEVICE static auto ExecuteInLoop(ODramWindow& out_dram_window,
                                             const OAccTile& o_acc_tile,
                                             const DsDramWindows& ds_dram_windows,
                                             void* p_smem,
                                             const InitArgs& init_args,
                                             const MainArgsTuple& main_args_tuple,
                                             const FinalArgs& final_args)
    {
        auto context = ck_tile::apply(
            [&](auto&&... unpacked_args) {
                return InitEpilogue{}(out_dram_window,
                                      o_acc_tile,
                                      ds_dram_windows,
                                      p_smem,
                                      std::forward<decltype(unpacked_args)>(unpacked_args)...);
            },
            init_args);

        constexpr index_t num_access = SelectEpilogue::SFC::get_num_of_access();
        static_for<0, num_access, 1>{}([&](auto iAccess) {
            static_for<0, NMainEpilogues, 1>{}([&](auto I) {
                using Epilogue   = MainEpilogueAt<I>;
                const auto& args = main_args_tuple.template get<I>();
                ck_tile::apply(
                    [&](auto&&... unpacked_args) {
                        Epilogue{}(out_dram_window,
                                   o_acc_tile,
                                   ds_dram_windows,
                                   p_smem,
                                   iAccess,
                                   context,
                                   std::forward<decltype(unpacked_args)>(unpacked_args)...);
                    },
                    args);
            });
        });

        if constexpr(HasFinalEpilogue)
        {
            ck_tile::apply(
                [&](auto&&... unpacked_args) {
                    FinalEpilogue{}(out_dram_window,
                                    o_acc_tile,
                                    ds_dram_windows,
                                    p_smem,
                                    context,
                                    std::forward<decltype(unpacked_args)>(unpacked_args)...);
                },
                final_args);
        }
    }

    public:
    using SelectEpilogue = InitEpilogue;
    using Problem        = typename SelectEpilogue::Problem;
    using ODataType      = typename SelectEpilogue::ODataType;
    using DsDataType     = typename SelectEpilogue::DsDataType;
    using DsLayout       = typename SelectEpilogue::DsLayout;
    using AccDataType    = typename SelectEpilogue::AccDataType;

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

    template <typename ODramWindow, typename OAccTile, typename DsDramWindows>
    CK_TILE_DEVICE auto operator()(ODramWindow& out_dram_window,
                                   const OAccTile& o_acc_tile,
                                   const DsDramWindows& ds_dram_windows,
                                   void* p_smem) const -> void
    {
        ExecuteSequential(out_dram_window, o_acc_tile, ds_dram_windows, p_smem);
    }

    template <typename ODramWindow, typename OAccTile, typename DsDramWindows>
    CK_TILE_DEVICE auto operator()(ODramWindow& out_dram_window,
                                   const OAccTile& o_acc_tile,
                                   const DsDramWindows& ds_dram_windows,
                                   void* p_smem,
                                   std::true_type /*loop*/) const -> void
    {
        ExecuteInLoop(out_dram_window, o_acc_tile, ds_dram_windows, p_smem);
    }

    template <typename ODramWindow,
              typename OAccTile,
              typename DsDramWindows,
              typename InitArgs,
              typename MainArgsTuple,
              typename FinalArgs>
    CK_TILE_DEVICE auto operator()(ODramWindow& out_dram_window,
                                   const OAccTile& o_acc_tile,
                                   const DsDramWindows& ds_dram_windows,
                                   void* p_smem,
                                   const InitArgs& init_args,
                                   const MainArgsTuple& main_args,
                                   const FinalArgs& final_args) const -> void
    {
        ExecuteSequential(
            out_dram_window, o_acc_tile, ds_dram_windows, p_smem, init_args, main_args, final_args);
    }

    template <typename ODramWindow,
              typename OAccTile,
              typename DsDramWindows,
              typename InitArgs,
              typename MainArgsTuple,
              typename FinalArgs>
    CK_TILE_DEVICE auto operator()(ODramWindow& out_dram_window,
                                   const OAccTile& o_acc_tile,
                                   const DsDramWindows& ds_dram_windows,
                                   void* p_smem,
                                   const InitArgs& init_args,
                                   const MainArgsTuple& main_args,
                                   const FinalArgs& final_args,
                                   std::true_type /*loop*/) const -> void
    {
        ExecuteInLoop(
            out_dram_window, o_acc_tile, ds_dram_windows, p_smem, init_args, main_args, final_args);
    }
};

} // namespace ck_tile
