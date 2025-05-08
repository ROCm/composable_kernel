// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"

namespace ck_tile {

template <typename ADataType_,
          typename BDataType_,
          typename DsDataType_,
          typename AccDataType_,
          typename ODataType_,
          typename DsLayout_,
          typename ELayout_,
          typename CDElementwise_,
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
    using ADataType                        = remove_cvref_t<ADataType_>;
    using BDataType                        = remove_cvref_t<BDataType_>;
    using AccDataType                      = remove_cvref_t<AccDataType_>;
    using ODataType                        = remove_cvref_t<ODataType_>;
    using DsDataType                       = remove_cvref_t<DsDataType_>;
    using DsLayout                         = remove_cvref_t<DsLayout_>;
    using ELayout                          = remove_cvref_t<ELayout_>;
    using CDElementwise                    = remove_cvref_t<CDElementwise_>;
    static constexpr index_t kBlockSize    = kBlockSize_;
    static constexpr index_t kMPerBlock    = kM_;
    static constexpr index_t kNPerBlock    = kN_;
    static constexpr index_t kMWave        = kMWave_;
    static constexpr index_t kNWave        = kNWave_;
    static constexpr index_t kMPerXdl      = kMPerXdl_;
    static constexpr index_t kNPerXdl      = kNPerXdl_;
    static constexpr index_t kKPerXdl      = kKPerXdl_;
    static constexpr index_t isCTransposed = isCTransposed_;
    static constexpr index_t NumDTensor    = DsDataType::size();

    static_assert(NumDTensor == DsDataType::size(),
                  "The size of DsDataType and DsLayout should be the same");
};

template <typename Problem_, typename Policy_ = void>
struct CShuffleEpilogue
{
    using Problem     = remove_cvref_t<Problem_>;
    using ADataType   = remove_cvref_t<typename Problem::ADataType>;
    using BDataType   = remove_cvref_t<typename Problem::BDataType>;
    using AccDataType = remove_cvref_t<typename Problem::AccDataType>;
    using ODataType   = remove_cvref_t<typename Problem::ODataType>;
    using DsDataType  = remove_cvref_t<typename Problem::DsDataType>;
    using DsLayout    = remove_cvref_t<typename Problem::DsLayout>;
    using BTypeToUse =
        std::conditional_t<std::is_same_v<BDataType, pk_int4_t>, ODataType, BDataType>;
    using ELayout                           = remove_cvref_t<typename Problem::ELayout>;
    using CDElementwise                     = remove_cvref_t<typename Problem::CDElementwise>;
    static constexpr index_t kBlockSize     = Problem::kBlockSize;
    static constexpr index_t kMPerBlock     = Problem::kMPerBlock;
    static constexpr index_t kNPerBlock     = Problem::kNPerBlock;
    static constexpr index_t kMWave         = Problem::kMWave;
    static constexpr index_t kNWave         = Problem::kNWave;
    static constexpr index_t kMPerXdl       = Problem::kMPerXdl;
    static constexpr index_t kNPerXdl       = Problem::kNPerXdl;
    static constexpr index_t kKPerXdl       = Problem::kKPerXdl;
    static constexpr index_t isCTransposed  = Problem::isCTransposed;
    static constexpr index_t kMPerIteration = kMPerXdl * kMWave;
    static constexpr index_t kNPerIteration = kNPerXdl * kNWave;
    static constexpr index_t NumDTensor     = Problem::NumDTensor;

    static constexpr index_t MaxVectorStoreSize = 16;

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
        return MaxVectorStoreSize / sizeof(ODataType);
    }

    /**
     * @brief Get the vector store size for Di tensor.
     *
     * @return The vector store size for Di tensor.
     */
    template <index_t I>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeD(number<I> index)
    {
        using DiDataType = remove_cvref_t<std::tuple_element_t<index.value, DsDataType>>;
        return MaxVectorStoreSize / sizeof(DiDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsBlockDescriptor()
    {
        // N is contiguous dimension
        if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
        {
            return make_naive_tensor_descriptor(
                make_tuple(number<kMWave * kMPerXdl>{}, number<kNWave * kNPerXdl>{}),
                make_tuple(number<kNWave * kNPerXdl>{}, number<1>{}));
        }
        // M is contiguous dimension
        else if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::ColumnMajor>)
        {
            return make_naive_tensor_descriptor(
                make_tuple(number<kMWave * kMPerXdl>{}, number<kNWave * kNPerXdl>{}),
                make_tuple(number<1>{}, number<kMWave * kMPerXdl>{}));
        }
        else
        {
            static_assert(false, "Unsupported ELayout!");
        }
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return kMWave * kNWave * kMPerXdl * kNPerXdl * sizeof(ODataType);
    }

    template <typename ODramWindow,
              typename OAccTile,
              typename DsDramWindows,
              memory_operation_enum out_memory_data_op = memory_operation_enum::set>
    CK_TILE_DEVICE auto operator()(ODramWindow& out_dram_window,
                                   const OAccTile& o_acc_tile,
                                   const DsDramWindows& ds_dram_window,
                                   void* p_smem)
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

        using TileEncodingPattern =
            TileDistributionEncodingPattern2D<kBlockSize,
                                              kMPerIteration,
                                              kNPerIteration,
                                              GetVectorSizeC(),
                                              tile_distribution_pattern::thread_raked>;
        constexpr auto dram_tile_distribution = TileEncodingPattern::Make2DStaticTileDistribution();

        auto d_dram_windows = generate_tuple(
            [&](auto idx) { return make_tile_window(ds_dram_window[idx], dram_tile_distribution); },
            number<NumDTensor>{});

        using elemenet_wise_output_t =
            decltype(load_tile(make_tile_window(out_lds_window, dram_tile_distribution)));
        elemenet_wise_output_t elemenet_wise_output;

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

            const auto ds_tensor = generate_tuple(
                [&](auto idx) { return load_tile(d_dram_windows[idx]); }, number<NumDTensor>{});

            const auto c_ds_tiles = concat_tuple_of_reference(
                tie(elemenet_wise_output, c_out_tensor),
                generate_tie(
                    [&](auto i) -> const auto& { return ds_tensor[i]; }, number<NumDTensor>{}));

            tile_elementwise_in_out_unpack_tuple(typename Problem::CDElementwise{}, c_ds_tiles);

            if constexpr(out_memory_data_op == memory_operation_enum::set)
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

                static_for<0, NumDTensor, 1>{}([&](auto idx) {
                    move_tile_window(d_dram_windows[idx],
                                     {step.at(number<0>{}), step.at(number<1>{})});
                });
            }
        });
    }
};
} // namespace ck_tile
