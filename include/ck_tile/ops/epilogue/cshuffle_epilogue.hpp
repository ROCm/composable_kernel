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
          index_t kM_,
          index_t kN_,
          index_t MWave_,
          index_t NWave_,
          index_t MPerXdl_,
          index_t NPerXdl_,
          index_t KPerXdl_,
          bool isCTransposed_,
          memory_operation_enum MemoryOperation_,
          index_t NumGroupsToMerge_ = 1,
          index_t kNumWaveGroups_ = 1,
          bool FixedVectorSize_   = false,
          index_t VectorSizeC_    = 1>
struct CShuffleEpilogueProblem
{
    using ADataType                                        = remove_cvref_t<ADataType_>;
    using BDataType                                        = remove_cvref_t<BDataType_>;
    using AccDataType                                      = remove_cvref_t<AccDataType_>;
    using ODataType                                        = remove_cvref_t<ODataType_>;
    using DsDataType                                       = remove_cvref_t<DsDataType_>;
    using DsLayout                                         = remove_cvref_t<DsLayout_>;
    using ELayout                                          = remove_cvref_t<ELayout_>;
    using CDElementwise                                    = remove_cvref_t<CDElementwise_>;
    static constexpr index_t kBlockSize                    = MWave_ * NWave_ * get_warp_size();
    static constexpr index_t kMPerBlock                    = kM_;
    static constexpr index_t kNPerBlock                    = kN_;
    static constexpr index_t MWave                         = MWave_;
    static constexpr index_t NWave                         = NWave_;
    static constexpr index_t MPerXdl                       = MPerXdl_;
    static constexpr index_t NPerXdl                       = NPerXdl_;
    static constexpr index_t KPerXdl                       = KPerXdl_;
    static constexpr index_t isCTransposed                 = isCTransposed_;
    static constexpr memory_operation_enum MemoryOperation = MemoryOperation_;
    static constexpr bool FixedVectorSize                  = FixedVectorSize_;
    static constexpr index_t VectorSizeC                   = VectorSizeC_;
    static constexpr index_t kNumWaveGroups                = kNumWaveGroups_;
    static constexpr index_t NumDTensor                    = DsDataType::size();
    static constexpr index_t NumGroupsToMerge              = NumGroupsToMerge_;

    static_assert(NumDTensor == DsLayout::size(),
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
    using ATypeToUse =
        std::conditional_t<std::is_same_v<ADataType, pk_int4_t>, BDataType, ADataType>;
    // Used for weight-only quantization kernel, B would be dequantized to the same data type as A
    using BTypeToUse =
        std::conditional_t<std::is_same_v<BDataType, pk_int4_t>, ADataType, BDataType>;
    using ELayout       = remove_cvref_t<typename Problem::ELayout>;
    using CDElementwise = remove_cvref_t<typename Problem::CDElementwise>;
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
    static constexpr bool FixedVectorSize                  = Problem::FixedVectorSize;
    static constexpr index_t VectorSizeC                   = Problem::VectorSizeC;
    static constexpr index_t MPerIteration                 = MPerXdl * MWave;
    static constexpr index_t NPerIteration                 = NPerXdl * NWave;
    static constexpr index_t NumDTensor                    = Problem::NumDTensor;
    static constexpr index_t NumGroupsToMerge              = Problem::NumGroupsToMerge;

    static_assert(NumDTensor == DsLayout::size(),
                  "The size of DsDataType and DsLayout should be the same");
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
        if constexpr(FixedVectorSize)
        {
            return VectorSizeC;
        }
        constexpr index_t max_vector_size = 16;
        if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
        {
            return std::min(static_cast<int>(NPerIteration),
                            static_cast<int>(max_vector_size / sizeof(ODataType)));
        }
        else if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::ColumnMajor>)
        {
            return std::min(static_cast<int>(MPerIteration),
                            static_cast<int>(max_vector_size / sizeof(ODataType)));
        }
        else
        {
            static_assert(false, "Unsupported ELayout!");
        }
    }

    /**
     * @brief Get the vector store size for Di tensor.
     *
     * @return The vector store size for Di tensor.
     */
    template <index_t I>
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeD(number<I> index)
    {
        constexpr index_t max_vector_size = 16;
        using DiDataType = remove_cvref_t<std::tuple_element_t<index.value, DsDataType>>;
        using DiLayout   = remove_cvref_t<std::tuple_element_t<index.value, DsLayout>>;
        if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
        {
            return std::min(static_cast<int>(NPerIteration),
                            static_cast<int>(max_vector_size / sizeof(DiDataType)));
        }
        else if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::ColumnMajor>)
        {
            return std::min(static_cast<int>(MPerIteration),
                            static_cast<int>(max_vector_size / sizeof(DiDataType)));
        }
        else
        {
            static_assert(false, "Unsupported DLayout!");
        }
        return max_vector_size / sizeof(DiDataType);
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
            if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
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

    using WG = WarpGemmDispatcher<ATypeToUse,
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
        if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
        {
            return make_naive_tensor_descriptor(
                make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
                make_tuple(number<NPerIterationShuffle>{}, number<1>{}));
        }
        // M is contiguous dimension
        else if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::ColumnMajor>)
        {
            return make_naive_tensor_descriptor(
                make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
                make_tuple(number<1>{}, number<MPerIterationShuffle>{}));
        }
        else
        {
            static_assert(false, "Unsupported ELayout!");
        }
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsDistributionEncode()
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
        if constexpr(NumGroupsToMerge > 1)
        {
            return kMPerBlock * kNPerBlock * sizeof(ODataType);
        }
        else
        {
            return MPerIterationShuffle * NPerIterationShuffle * sizeof(ODataType);
        }
    }

    template <typename ODramWindow, typename OAccTile, typename DsDramWindows>
    CK_TILE_DEVICE auto operator()(ODramWindow& out_dram_window,
                                   const OAccTile& o_acc_tile,
                                   const DsDramWindows& ds_dram_windows,
                                   void* p_smem)

    {
        if constexpr (NumGroupsToMerge > 1)
        {
            // When NumGroupsToMerge > 1, we want to write out only the diagonal blocks.
            // Hence, we configure the shuffle such that it iterates one merge group block at a time.
            return merged_op(out_dram_window, o_acc_tile, ds_dram_windows, p_smem);
        }
        else
        {
            // When NumGroupsToMerge == 1, we want to write out all the blocks.
            return unmerged_op(out_dram_window, o_acc_tile, ds_dram_windows, p_smem);
        }
    }

    template <typename DataType, typename StaticTileDistribution>
    CK_TILE_DEVICE void print_tensor_matrix_format(
        const static_distributed_tensor<DataType, StaticTileDistribution>& tensor,
        const char* /*name = "tensor_matrix"*/)
    {
        const auto spans = tensor.get_distributed_spans();
        //static_assert(spans.size() == 2, "This function is for 2D tensors only");
        
        const auto dim0_span = spans[number<0>{}];
        const auto dim1_span = spans[number<1>{}];
        
        //printf("%s matrix format (tid %u):\n", name, threadIdx.x);
        
        sweep_tile_span(dim0_span, [&](auto row) {
            printf("  ");
            sweep_tile_span(dim1_span, [&](auto col) {
                constexpr auto distributed_indices = make_tuple(row, col);
                const auto value = tensor[distributed_indices];
                printf("tid %u: %.7f\n", threadIdx.x, static_cast<float>(value));
            });
            //printf("\n");
        });
        //printf("\n");
    }

    template <typename ODramWindow, typename OAccTile, typename DsDramWindows>
    CK_TILE_DEVICE auto merged_op(ODramWindow& out_dram_window,
                                   const OAccTile& o_acc_tile,
                                   const DsDramWindows& ds_dram_windows,
                                   void* p_smem)
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

        using SFC = space_filling_curve<sequence<kMPerBlock, kNPerBlock>,
                                    sequence<0, 1>,
                                    sequence<MPerIterationShuffle, NPerIterationShuffle>>;

        constexpr index_t num_access = SFC::get_num_of_access();

        static_assert(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>,
                    "Currently, the CShuffle Epilogue only supports the Row Major Output layout");

        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        // Store full data to LDS.
        block_sync_lds();
        static_for<0, num_access, 1>{}([&](auto iAccess) {
            
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
            
            if constexpr(iAccess != num_access - 1)
            {
                constexpr auto step = SFC::get_forward_step(iAccess);

                move_tile_window(in_lds_window, {step.at(number<0>{}), step.at(number<1>{})});
            }
        });
        block_sync_lds();

        constexpr index_t Gs = NumGroupsToMerge;
        constexpr index_t MPerGroup = kMPerBlock / Gs;
        constexpr index_t NPerGroup = kNPerBlock / Gs;

        // Tile enconding for a single group (diagonal block in LDS)
        constexpr auto dram_tile_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<1, 1, MPerGroup, 1>, 
                sequence<1, 1, NPerGroup, 1>>,
            tuple<sequence<1,2>, sequence<1,2>>,
            tuple<sequence<1,1>, sequence<2,2>>, 
            sequence<1, 1, 2, 2>, 
            sequence<0, 3, 0, 3>>{};
        constexpr auto dram_tile_distribution = make_static_tile_distribution(dram_tile_encoding);

        // The LDS data has the following 4D layout
        // linear_index = c + Gs * m + Gs * MPerGroup * n + Gs * MPerGroup * NPerGroup * r
        // for 4D coordinates (r,c,m,n) where (r,c) is the group index and (m,n) is the index within the group.
        // We pick-up only the diagonal blocks where r == c.
        // For each block, the tile distribution and the tensor descriptors are the same.
        // The only thing that changes is the p_smem offset.
        constexpr auto lds_block_desc_2d = make_naive_tensor_descriptor(
            make_tuple(number<MPerGroup>{}, number<NPerGroup>{}),
            make_tuple(number<Gs>{}, number<Gs * MPerGroup>{}));

        // Loop over the groups (diagonal blocks in LDS)
        static_for<0, Gs, 1>{}([&](auto g) {
            block_sync_lds();

            constexpr index_t group_offset = g * (1 + Gs* MPerGroup * NPerGroup);
            auto lds_view = make_tensor_view<address_space_enum::lds>(
                         static_cast<ODataType*>(p_smem) + group_offset, lds_block_desc_2d);

            auto d_dram_windows = generate_tuple(
                [&](auto idx) {
                    return make_tile_window(ds_dram_windows[idx], dram_tile_distribution);
                },
                number<NumDTensor>{});

            const auto lds_window = make_tile_window(
                lds_view,
                make_tuple(number<MPerGroup>{}, number<NPerGroup>{}),
                {0, 0},
                dram_tile_distribution);

            // Load static_distributed_tensor from LDS.
            auto c_out_tensor = load_tile(lds_window);

            // DEBUG: Print out the c_out_tensor contents for debugging
            // print_tensor_matrix_format(c_out_tensor, "c_out_tensor");
            // __syncthreads();

            const auto ds_tensor = generate_tuple(
                [&](auto idx) { return load_tile(d_dram_windows[idx]); }, number<NumDTensor>{});

            const auto c_ds_tiles = concat_tuple_of_reference(
                tie(c_out_tensor, c_out_tensor),
                generate_tie([&](auto idx) -> const auto& { return ds_tensor[idx]; },
                            number<NumDTensor>{}));

            tile_elementwise_inout_unpack(typename Problem::CDElementwise{}, c_ds_tiles);

            if constexpr(MemoryOperation == memory_operation_enum::set)
            {
                store_tile(out_dram_window, c_out_tensor);
            }
            else
            {
                update_tile(out_dram_window, c_out_tensor);
            }

            // Move the output window to the next group position.
            if constexpr(g != Gs - 1)
            {
                move_tile_window(out_dram_window, {number<MPerGroup>{}, 0});

                static_for<0, NumDTensor, 1>{}([&](auto idx) {
                    move_tile_window(d_dram_windows[idx], {number<MPerGroup>{}, 0});
                });
            }
        });


        //---------------------------------------------------------------------


        // // 4D tensor view of LDS memory
        // // with (g_i, g_j, i, j) where (g_i, g_j) is the group index
        // // and (i, j) is the index within the group.
        // constexpr auto lds_desc_4d = make_naive_tensor_descriptor(
        //     make_tuple(number<Gs>{}, number<Gs>{}, number<MPerGroup>{}, number<NPerGroup>{}),
        //     make_tuple(number<Gs * MPerGroup * NPerGroup>{}, number<1>{}, number<Gs>{}, number<Gs * MPerGroup>{}));

        // // We must merge (r,m) and (c,n) dimensions together to make a 2D tensor descriptor.
        // constexpr auto lds_desc = transform_tensor_descriptor(
        //     lds_desc_4d, 
        //     make_tuple(
        //         make_merge_transform(make_tuple(Gs, MPerGroup)),
        //         make_merge_transform(make_tuple(Gs, NPerGroup))
        //     ),
        //     make_tuple(sequence<0, 2>{}, sequence<1, 3>{}), 
        //     make_tuple(sequence<0>{}, sequence<1>{})
        // );

        // auto lds_view = make_tensor_view<address_space_enum::lds>(
        //                  static_cast<ODataType*>(p_smem), lds_desc);

        // constexpr auto dram_tile_encoding = tile_distribution_encoding<
        //     sequence<>,
        //     tuple<sequence<1, Gs, MPerGroup, 1>, 
        //         sequence<1, Gs, NPerGroup, 1>>,
        //     tuple<sequence<1,2>, sequence<1,2>>,
        //     tuple<sequence<1,1>, sequence<2,2>>, 
        //     sequence<1, 1, 2, 2>, 
        //     sequence<0, 3, 0, 3>>{};

        // constexpr auto dram_tile_distribution = make_static_tile_distribution(dram_tile_encoding);

        // auto d_dram_windows = generate_tuple(
        //     [&](auto idx) {
        //         return make_tile_window(ds_dram_windows[idx], dram_tile_distribution);
        //     },
        //     number<NumDTensor>{});

        // // Calculate which block in the Gs x Gs space we are located at.
        // const auto x_space_coord = dram_tile_distribution.calculate_index();
        // const index_t m_block = x_space_coord[0] / MPerGroup;
        // const index_t n_block = x_space_coord[1] / NPerGroup;

        // const auto current_lds_window = make_tile_window(
        //         lds_view,
        //         make_tuple(number<Gs * MPerGroup>{}, number<Gs * NPerGroup>{}),
        //         {0, 0},
        //         dram_tile_distribution);

        // // Copy only the diagonal blocks.
        // if (m_block == n_block)
        // {
        //     // Load static_distributed_tensor from LDS.
        //     auto c_out_tensor = load_tile(current_lds_window);

        //     // DEBUG: Print out the c_out_tensor contents for debugging
        //     print_tensor_matrix_format(c_out_tensor, "c_out_tensor");
        //     __syncthreads();

        //     // TODO: We must move the d_dram_windows to the correct group position.
        //     const auto ds_tensor = generate_tuple(
        //         [&](auto idx) { return load_tile(d_dram_windows[idx]); }, number<NumDTensor>{});

        //     const auto c_ds_tiles = concat_tuple_of_reference(
        //         tie(c_out_tensor, c_out_tensor),
        //         generate_tie([&](auto idx) -> const auto& { return ds_tensor[idx]; },
        //                     number<NumDTensor>{}));

        //     tile_elementwise_inout_unpack(typename Problem::CDElementwise{}, c_ds_tiles);

        //     // Move the output window to the correct position.
        //     //printf("m_block: %d, n_block: %d \n", m_block, n_block);
        //     //auto out_window = make_tile_window(out_dram_window, dram_tile_distribution);
        //     //move_tile_window(out_window, {m_block * MPerGroup, 0});

        //     if constexpr(MemoryOperation == memory_operation_enum::set)
        //     {
        //         store_tile(out_window, c_out_tensor);
        //     }
        //     else
        //     {
        //         update_tile(out_window, c_out_tensor);
        //     }
        // }
    }

    template <typename ODramWindow, typename OAccTile, typename DsDramWindows>
    CK_TILE_DEVICE auto unmerged_op(ODramWindow& out_dram_window,
                                   const OAccTile& o_acc_tile,
                                   const DsDramWindows& ds_dram_windows,
                                   void* p_smem)
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

        using SFC = space_filling_curve<sequence<kMPerBlock, kNPerBlock>,
                                    sequence<0, 1>,
                                    sequence<MPerIterationShuffle, NPerIterationShuffle>>;

        constexpr index_t num_access = SFC::get_num_of_access();

        static_assert(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>,
                    "Currently, the CShuffle Epilogue only supports the Row Major Output layout");

        using TileEncodingPattern = tile_distribution_encoding_pattern_2d<kBlockSize,
                                                    MPerIterationShuffle,
                                                    NPerIterationShuffle,
                                                    GetVectorSizeC(),
                                                    tile_distribution_pattern::sparse_row,
                                                    Problem::kNumWaveGroups>;
        constexpr auto dram_tile_distribution =
            TileEncodingPattern::make_2d_static_tile_distribution();

        auto d_dram_windows = generate_tuple(
            [&](auto idx) {
                return make_tile_window(ds_dram_windows[idx], dram_tile_distribution);
            },
            number<NumDTensor>{});

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

            auto c_out_tensor = load_tile(make_tile_window(out_lds_window, dram_tile_distribution));

            const auto ds_tensor = generate_tuple(
                [&](auto idx) { return load_tile(d_dram_windows[idx]); }, number<NumDTensor>{});

            const auto c_ds_tiles = concat_tuple_of_reference(
                tie(c_out_tensor, c_out_tensor),
                generate_tie([&](auto idx) -> const auto& { return ds_tensor[idx]; },
                            number<NumDTensor>{}));

            tile_elementwise_inout_unpack(typename Problem::CDElementwise{}, c_ds_tiles);

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

                static_for<0, NumDTensor, 1>{}([&](auto idx) {
                    move_tile_window(d_dram_windows[idx],
                                    {step.at(number<0>{}), step.at(number<1>{})});
                });
            }
        });
    }
};
} // namespace ck_tile
