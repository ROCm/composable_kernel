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
          index_t kMWave_,
          index_t kNWave_,
          index_t kMPerXdl_,
          index_t kNPerXdl_,
          index_t kKPerXdl_,
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
    static constexpr index_t kMWave                        = kMWave_;
    static constexpr index_t kNWave                        = kNWave_;
    static constexpr index_t kMPerXdl                      = kMPerXdl_;
    static constexpr index_t kNPerXdl                      = kNPerXdl_;
    static constexpr index_t kKPerXdl                      = kKPerXdl_;
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
        return MaxVectorStoreSize / sizeof(ODataType);
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

    CK_TILE_HOST_DEVICE static constexpr auto GetCDisrtribution()
    {
        using TileEncodingPattern =
            TileDistributionEncodingPattern2D<kBlockSize,
                                              kMPerIteration,
                                              kNPerIteration,
                                              GetVectorSizeC(),
                                              tile_distribution_pattern::thread_raked>;
        return TileEncodingPattern::Make2DStaticTileDistribution();
    }

    template <typename ODramWindow,
              typename OAccTile,
              bool IsInputGemm                         = true,
              memory_operation_enum out_memory_data_op = memory_operation_enum::set>
    CK_TILE_DEVICE auto operator()(ODramWindow& out_dram_window,
                                   const OAccTile& o_acc_tile,
                                   void* p_smem,
                                   const index_t* p_sorted_tokens_id,
                                   index_t token_pos,
                                   index_t TopK,
                                   index_t stride_C)
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
        // auto coord = dram_tile_distribution.calculate_index();

        // const auto& view = out_dram_window.get_bottom_tensor_view();

        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};
        auto c_coord                        = dram_tile_distribution.calculate_index();

        // printf("c_coord[0]:%d \n", c_coord[0]);

        CWarpTensor c_warp_in_tensor;
        static_for<0, num_access, 1>{}([&](auto iAccess) {
            constexpr auto idx_y_start = SFC::get_index(iAccess);

            // auto idx_m =  number<idx_y_start.at(number<0>{})>{} + 0;
            // printf("idx_y_start:%d \n", idx_m);
            constexpr auto MPerAcess = kMPerXdl * kMWave;
            constexpr auto mIter = number<idx_y_start.at(number<0>{}) / (MPerAcess)>{};
            using CDstrEncode = typename decltype(dram_tile_distribution)::DstrEncode;
            constexpr ck_tile::index_t MRepeat = CDstrEncode::hs_lengthss_[number<0>{}][number<0>{}];

            statically_indexed_array<index_t, MRepeat> offsets;

            static_for<0, MRepeat, 1>{}([&](auto m0) {
                auto token_id    = token_pos + m0 + c_coord[0] + mIter * kMPerXdl * kMWave;
                // auto token_id    = token_pos + c_coord[0] + mIter * MPerAcess + MPerAcess / MRepeat * m0.value;
                auto fused_token = p_sorted_tokens_id[token_id];

                index_t scatter_token_id = fused_token & 0xffffff;
                if constexpr(IsInputGemm)
                {
                    scatter_token_id = scatter_token_id * TopK + (fused_token >> 24);
                }

                offsets[m0] = scatter_token_id * stride_C; // Problem::kN_;
            });
            // printf("c_coord[number<0>{}]: %d \n", coord[number<0>{}]);
            // printf("mIter: %d", mIter+0);

            constexpr auto nIter = number<idx_y_start.at(number<1>{}) / (kNPerXdl * kNWave)>{};

            // printf("mIter, nIter(%d, %d) \n", mIter+0, nIter+0);

            c_warp_in_tensor.get_thread_buffer() = o_acc_tile.get_y_sliced_thread_data(
                merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

            const auto c_warp_in_tensor_casted = cast_tile<ODataType>(c_warp_in_tensor);

            block_sync_lds();
            store_tile(in_lds_window, c_warp_in_tensor_casted);
            block_sync_lds();

            const auto c_out_tensor =
                load_tile(make_tile_window(out_lds_window, dram_tile_distribution));

            if constexpr(out_memory_data_op == memory_operation_enum::set)
            {
                auto tile_window = make_tile_scatter_gather(out_dram_window.get_bottom_tensor_view(),
                                    out_dram_window.get_window_lengths(),
                                    out_dram_window.get_window_origin(),
                                    dram_tile_distribution,
                                    offsets);
                tile_window.store(c_out_tensor);
                // store_tile(out_dram_window, c_out_tensor, offsets);
            }
            else
            {
                update_tile(out_dram_window, c_out_tensor);
            }
            if constexpr(iAccess != num_access - 1)
            {
                constexpr auto step = SFC::get_forward_step(iAccess);
                move_tile_window(out_dram_window, {0, step.at(number<1>{})});
                // printf("step.at(number<0>{}), step.at(number<1>{}):,%d, %d",
                // step.at(number<0>{})+0, step.at(number<1>{})+0);
            }
        });
    }

    template <typename ODramWindow, typename OAccTile>
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
