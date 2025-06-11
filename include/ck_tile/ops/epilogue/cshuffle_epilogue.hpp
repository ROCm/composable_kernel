// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"

namespace ck_tile {

template <typename ADataType_,
          typename BDataType_,
          typename AccDataType_,
          typename ODataType_,
          typename CLayout_,
          index_t kBlockSize_,
          index_t kM_,
          index_t kN_,
          index_t MWave_,
          index_t NWave_,
          index_t MPerXdl_,
          index_t NPerXdl_,
          index_t KPerXdl_,
          bool isCTransposed_,
          memory_operation_enum MemoryOperation_>
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
    static constexpr index_t MWave                         = MWave_;
    static constexpr index_t NWave                         = NWave_;
    static constexpr index_t MPerXdl                       = MPerXdl_;
    static constexpr index_t NPerXdl                       = NPerXdl_;
    static constexpr index_t KPerXdl                       = KPerXdl_;
    static constexpr index_t isCTransposed                 = isCTransposed_;
    static constexpr memory_operation_enum MemoryOperation = MemoryOperation_;
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
    static constexpr index_t MWave                         = Problem::MWave;
    static constexpr index_t NWave                         = Problem::NWave;
    static constexpr index_t MPerXdl                       = Problem::MPerXdl;
    static constexpr index_t NPerXdl                       = Problem::NPerXdl;
    static constexpr index_t KPerXdl                       = Problem::KPerXdl;
    static constexpr index_t isCTransposed                 = Problem::isCTransposed;
    static constexpr index_t MPerIteration                 = MPerXdl * MWave;
    static constexpr index_t NPerIteration                 = NPerXdl * NWave;

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
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeC()
    {
        constexpr index_t max_vector_size = 16;
        if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
        {
            return std::min(static_cast<int>(NPerIteration),
                            static_cast<int>(max_vector_size / sizeof(ODataType)));
        }
        else if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::ColumnMajor>)
        {
            return std::min(static_cast<int>(MPerIteration),
                            static_cast<int>(max_vector_size / sizeof(ODataType)));
        }
        else
        {
            static_assert(false, "Unsupported CLayout!");
        }
    }

    /**
     * @brief Shuffle tile configuration parameters
     *
     * @details These parameters control the number of XDL tiles processed per wave in each shuffle
     * iteration:
     * - NumMXdlPerWavePerShuffle: Number of XDL tiles in M dimension processed per wave
     * - NumNXdlPerWavePerShuffle: Number of XDL tiles in N dimension processed per wave
     */
    static constexpr auto shuffle_tile_tuple = [] {
        constexpr index_t elem_per_thread = MPerXdl * NPerXdl / get_warp_size();
        if constexpr(elem_per_thread >= GetVectorSizeC())
        {
            return std::make_tuple(1, 1);
        }
        else
        {
            constexpr index_t num_xdl_shuffles = GetVectorSizeC() / elem_per_thread;
            if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
            {
                static_assert((kMPerBlock % (MPerXdl * MWave) == 0) &&
                                  (kMPerBlock % num_xdl_shuffles == 0),
                              "kMPerBlock must be divisible by MPerXdl*MWave and "
                              "num_xdl_shuffles for CShuffleEpilogue");
                return std::make_tuple(min(num_xdl_shuffles, kMPerBlock / (MPerXdl * MWave)), 1);
            }
            else
            {
                static_assert((kNPerBlock % (NPerXdl * NWave) == 0) &&
                                  (kNPerBlock % num_xdl_shuffles == 0),
                              "kNPerBlock must be divisible by NPerXdl*NWave and "
                              "num_xdl_shuffles for CShuffleEpilogue");
                return std::make_tuple(1, min(num_xdl_shuffles, kNPerBlock / (NPerXdl * NWave)));
            }
        }
    }();
    static constexpr index_t NumMXdlPerWavePerShuffle = std::get<0>(shuffle_tile_tuple);
    static constexpr index_t NumNXdlPerWavePerShuffle = std::get<1>(shuffle_tile_tuple);

    static constexpr auto MNPerIterationShuffle = [] {
        constexpr index_t m_val = MPerXdl * MWave * NumMXdlPerWavePerShuffle;
        constexpr index_t n_val = NPerXdl * NWave * NumNXdlPerWavePerShuffle;
        if constexpr(kMPerBlock % m_val != 0 || kNPerBlock % n_val != 0)
            return std::make_tuple(MPerXdl * MWave, NPerXdl * NWave);
        else
            return std::make_tuple(m_val, n_val);
    }();
    static constexpr index_t MPerIterationShuffle = std::get<0>(MNPerIterationShuffle);
    static constexpr index_t NPerIterationShuffle = std::get<1>(MNPerIterationShuffle);
    using WG                                      = WarpGemmMfmaDispatcher<ADataType,
                                      BTypeToUse,
                                      AccDataType,
                                      MPerXdl,
                                      NPerXdl,
                                      KPerXdl,
                                      isCTransposed>;

    using CWarpDstr   = typename WG::CWarpDstr;
    using CWarpTensor = typename WG::CWarpTensor;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsBlockDescriptor()
    {
        // N is contiguous dimension
        if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
        {
            return make_naive_tensor_descriptor(
                make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
                make_tuple(number<NPerIterationShuffle>{}, number<1>{}));
        }
        // M is contiguous dimension
        else if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::ColumnMajor>)
        {
            return make_naive_tensor_descriptor(
                make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
                make_tuple(number<1>{}, number<MPerIterationShuffle>{}));
        }
        else
        {
            static_assert(false, "Unsupported CLayout!");
        }
    }

    CK_TILE_DEVICE static constexpr auto MakeLdsDistributionEncode()
    {
        constexpr auto block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<NumMXdlPerWavePerShuffle, MWave>,
                                             sequence<NumNXdlPerWavePerShuffle, NWave>>,
                                       tuple<sequence<1, 2>>,
                                       tuple<sequence<1, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};
        constexpr auto block_dstr_encoding = detail::make_embed_tile_distribution_encoding(
            block_outer_dstr_encoding, typename CWarpDstr::DstrEncode{});

        return block_dstr_encoding;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return MPerIterationShuffle * NPerIterationShuffle * sizeof(ODataType);
    }

    template <typename ODramWindow, typename OAccTile>
    CK_TILE_DEVICE auto
    operator()(ODramWindow& out_dram_window, const OAccTile& o_acc_tile, void* p_smem)
    {
        constexpr auto LdsTileDistr = make_static_tile_distribution(MakeLdsDistributionEncode());

        auto lds_tile = make_static_distributed_tensor<AccDataType>(LdsTileDistr);

        constexpr auto lds_block_desc = MakeLdsBlockDescriptor<Problem>();
        auto o_lds_block              = make_tensor_view<address_space_enum::lds>(
            static_cast<ODataType*>(p_smem), lds_block_desc);

        auto in_lds_window = make_tile_window(
            o_lds_block,
            make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
            {0, 0},
            LdsTileDistr);

        auto out_lds_window = make_tile_window(
            o_lds_block,
            make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
            {0, 0});

        using SFC                    = space_filling_curve<sequence<kMPerBlock, kNPerBlock>,
                                        sequence<0, 1>,
                                        sequence<MPerIterationShuffle, NPerIterationShuffle>>;
        constexpr index_t num_access = SFC::get_num_of_access();

        static_assert(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>,
                      "Currently, the CShuffle Epilogue only supports the Row Major Output layout");

        using TileEncodingPattern =
            TileDistributionEncodingPattern2D<kBlockSize,
                                              MPerIterationShuffle,
                                              NPerIterationShuffle,
                                              GetVectorSizeC(),
                                              tile_distribution_pattern::thread_raked>;
        constexpr auto dram_tile_distribution = TileEncodingPattern::Make2DStaticTileDistribution();

        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        static_for<0, num_access, 1>{}([&](auto iAccess) {
            block_sync_lds();
            constexpr auto idx_y_start = SFC::get_index(iAccess);

            constexpr auto mIter = number<idx_y_start.at(number<0>{}) / (MPerIterationShuffle)>{};
            constexpr auto nIter = number<idx_y_start.at(number<1>{}) / (NPerIterationShuffle)>{};

            lds_tile.get_thread_buffer() = o_acc_tile.get_y_sliced_thread_data(
                merge_sequences(
                    sequence<mIter * NumMXdlPerWavePerShuffle, nIter * NumNXdlPerWavePerShuffle>{},
                    c_warp_y_index_zeros),
                merge_sequences(sequence<NumMXdlPerWavePerShuffle, NumNXdlPerWavePerShuffle>{},
                                c_warp_y_lengths));

            const auto c_warptile_in_tensor_casted = cast_tile<ODataType>(lds_tile);

            store_tile(in_lds_window, c_warptile_in_tensor_casted);
            block_sync_lds();

            const auto c_out_tensor =
                load_tile(make_tile_window(out_lds_window, dram_tile_distribution));

            if constexpr(MemoryOperation == memory_operation_enum::set)
            {
                store_tile(out_dram_window, c_out_tensor);
            }
            else
            {
                update_tile(out_dram_window, c_out_tensor);
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
