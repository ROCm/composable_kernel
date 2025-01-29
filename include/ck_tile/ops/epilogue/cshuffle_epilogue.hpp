// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"

namespace ck_tile {

template <typename AccDataType_,
          typename ODataType_,
          index_t kBlockSize_,
          index_t kM_,
          index_t kN_,
          index_t kMWave_,
          index_t kNWave_,
          index_t kMPerXdl_,
          index_t kNPerXdl_,
          index_t kKPerXdl_,
          bool isCTransposed_>
struct CShuffleEpilogueProblem
{
    using AccDataType                      = remove_cvref_t<AccDataType_>;
    using ODataType                        = remove_cvref_t<ODataType_>;
    static constexpr index_t kBlockSize    = kBlockSize_;
    static constexpr index_t kMPerBlock    = kM_;
    static constexpr index_t kNPerBlock    = kN_;
    static constexpr index_t kMWave        = kMWave_;
    static constexpr index_t kNWave        = kNWave_;
    static constexpr index_t kMPerXdl      = kMPerXdl_;
    static constexpr index_t kNPerXdl      = kNPerXdl_;
    static constexpr index_t kKPerXdl      = kKPerXdl_;
    static constexpr index_t isCTransposed = isCTransposed_;
};

template <typename Problem_, typename Policy_ = void>
struct CShuffleEpilogue
{
    using Problem                          = remove_cvref_t<Problem_>;
    using AccDataType                      = remove_cvref_t<typename Problem::AccDataType>;
    using ODataType                        = remove_cvref_t<typename Problem::ODataType>;
    static constexpr bool UseRawStore      = Problem::UseRawStore;
    static constexpr index_t kBlockSize    = Problem::kBlockSize;
    static constexpr index_t kMPerBlock    = Problem::kMPerBlock;
    static constexpr index_t kNPerBlock    = Problem::kNPerBlock;
    static constexpr index_t kMWave        = Problem::kMWave;
    static constexpr index_t kNWave        = Problem::kNWave;
    static constexpr index_t kMPerXdl      = Problem::kMPerXdl;
    static constexpr index_t kNPerXdl      = Problem::kNPerXdl;
    static constexpr index_t kKPerXdl      = Problem::kKPerXdl;
    static constexpr index_t isCTransposed = Problem::isCTransposed;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsBlockDescriptor()
    {
        return make_naive_tensor_descriptor(
            make_tuple(number<kMWave * kMPerXdl>{}, number<kNWave * kNPerXdl>{}),
            make_tuple(number<kNWave * kNPerXdl>{}, number<1>{}));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeDramTileDistribution()
    {
        constexpr index_t MRepeat = Problem::kMPerBlock / (Problem::kMPerXdl * Problem::kMWave);
        constexpr index_t NRepeat = Problem::kNPerBlock / (Problem::kNPerXdl * Problem::kNWave);
        constexpr index_t kMPerIteration = Problem::kMPerBlock / MRepeat;
        constexpr index_t kNPerIteration = Problem::kNPerBlock / NRepeat;

        constexpr index_t N1 = 16 / sizeof(ODataType);
        constexpr index_t N0 = kNPerIteration / N1;
        constexpr index_t M2 = get_warp_size() / N0;
        // coalesce reading for each blocks
        if constexpr(get_warp_size() % (M2 * N0) == 0)
        {
            constexpr index_t M1 = kBlockSize / get_warp_size();
            static_assert(M2 != 0, "M2 is zero, which will lead to a division by zero error.");
            static_assert(M1 != 0, "M1 is zero, which will lead to a division by zero error.");
            constexpr index_t M0 = kMPerIteration / (M2 * M1);
            static_assert(M0 * M1 * M2 == kMPerIteration,
                          "Incorrect M0, M2, M1 configuration! "
                          "M0, M1, M2 must cover whole kMPerIteration!");

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<M0, M1, M2>, sequence<N0, N1>>,
                                           tuple<sequence<1>, sequence<1, 2>>,
                                           tuple<sequence<1>, sequence<2, 0>>,
                                           sequence<1, 2>,
                                           sequence<0, 1>>{});
        }
        else
        {
            constexpr index_t M0 = kBlockSize / get_warp_size();
            constexpr index_t M1 = kMPerIteration / (M2 * M0);
            static_assert(M0 * M1 * M2 == kMPerIteration,
                          "Incorrect M0, M1, M2 configuration! "
                          "M0, M1, M2 must cover whole kMPerIteration!");
            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<M0, M1, M2>, sequence<N0, N1>>,
                                           tuple<sequence<1>, sequence<1, 2>>,
                                           tuple<sequence<0>, sequence<2, 0>>,
                                           sequence<1, 2>,
                                           sequence<1, 1>>{});
        }
    }

    using WG = WarpGemmMfmaDispatcher<ODataType,
                                      ODataType,
                                      AccDataType,
                                      kMPerXdl,
                                      kNPerXdl,
                                      kKPerXdl,
                                      isCTransposed>;

    using CWarpDstr   = typename WG::CWarpDstr;
    using CWarpTensor = typename WG::CWarpTensor;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return kMWave * kNWave * kMPerXdl * kNPerXdl * sizeof(ODataType);
    }

    CK_TILE_HOST_DEVICE static constexpr bool IsOutputTransposed() { return true; }

    template <typename ODramWindow,
              typename OAccTile,
              memory_operation_enum out_memory_data_op = memory_operation_enum::set>
    CK_TILE_DEVICE auto
    operator()(ODramWindow& out_dram_window, const OAccTile& o_acc_tile, void* p_smem)
    {

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
                load_tile(make_tile_window(out_lds_window, MakeDramTileDistribution<Problem>()));

            move_tile_window(out_dram_window,
                             {idx_y_start.at(number<0>{}), idx_y_start.at(number<1>{})});
            if constexpr(out_memory_data_op == memory_operation_enum::set)
            {
                store_tile(out_dram_window, c_out_tensor);
            }
            else
            {
                update_tile(out_dram_window, c_out_tensor);
            }
            move_tile_window(out_dram_window,
                             {-(idx_y_start.at(number<0>{})), -(idx_y_start.at(number<1>{}))});
        });
    }
};
} // namespace ck_tile
