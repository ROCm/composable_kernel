// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/host/concat.hpp"

namespace ck_tile {
namespace reboot {

/// @brief The Stream K GEMM kernel host arguments.
///
/// @par Overview
///      This structure is passed to @ref StreamKKernel "StreamKKernel" when creating the kernel
///      arguments object. It contains all necessary information required to build proper kernel
///      arguments and launch the kernel on GPU. This structure defines the GEMM problem
///      configuration by stating all required information like M,N,K sizes and respective strides.
struct StreamKHostArgs : public ck_tile::UniversalGemmHostArgs<>
{
    CK_TILE_HOST explicit StreamKHostArgs(const void* a_ptr_,
                                          const void* b_ptr_,
                                          void* c_ptr_,
                                          index_t M_,
                                          index_t N_,
                                          index_t K_,
                                          index_t stride_A_,
                                          index_t stride_B_,
                                          index_t stride_C_,
                                          StreamKReductionStrategy reduction_strategy_)
        : UniversalGemmHostArgs<>({a_ptr_},
                                  {b_ptr_},
                                  {/*ds_ptr*/},
                                  c_ptr_,
                                  /*k_batch_ =*/1,
                                  M_,
                                  N_,
                                  K_,
                                  {stride_A_},
                                  {stride_B_},
                                  {/*stride_Ds_*/},
                                  stride_C_),
          reduction_strategy{reduction_strategy_}
    {
    }

    ck_tile::StreamKReductionStrategy reduction_strategy;
};

/// @brief The Stream K GEMM kernel class.
///
/// @par Overview
///      This class is responsible for the Stream-K kernel, making use of UniversalGemm.
//	 The main kernel functions are the operator() functions. There is one for Persistent
//	 and one for Non-Persistent data parallel sections of the Stream-K algorithm.
//
//	 Both the Non-Persistent and Persistent kernels make use of `BaseGemm()` and
//	 `StreamKGemm()`. `BaseGemm()` computes offsets into the A,B,C tensors, then calls
//	 `RunGemm()` which runs the GEMM pipeline and epilogue. `StreamKGemm()` performs the
//	 main Stream-K algorithm. Each iteration of the Stream-K loop calls `BaseGemm()`.
template <typename TilePartitioner_, typename GemmPipeline_, typename EpiloguePipeline_>
struct StreamKKernel
{
    /// @brief Inject the UniversalGemmKernel base class to support execution of all necessary
    /// functions.
    using UniversalGemmKernel =
        UniversalGemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>;

    static constexpr index_t kBlockSize = UniversalGemmKernel::kBlockSize;
    static constexpr bool PersistentDP  = UniversalGemmKernel::PersistentKernel;

    using TilePartitioner  = TilePartitioner_;
    using GemmPipeline     = GemmPipeline_;
    using EpiloguePipeline = EpiloguePipeline_;

    static_assert(
        TilePartitioner::PERSISTENT == PersistentDP,
        "Persistent flag from TilePartitioner must match Persistent flag from UniversalGemm.");

    /// @brief  Specify the layout configurations for A, B, and C
    using ALayout = typename GemmPipeline::ALayout;
    using BLayout = typename GemmPipeline::BLayout;
    using CLayout = typename GemmPipeline::CLayout;

    /// @brief  Specify the data type configurations for A, B, and C
    using ADataType   = typename GemmPipeline::ADataType;
    using BDataType   = typename GemmPipeline::BDataType;
    using CDataType   = typename EpiloguePipeline::ODataType;
    using AccDataType = typename EpiloguePipeline::AccDataType;

    template <typename T>
    static constexpr bool is_tuple_v = is_detected<is_tuple, T>::value;

    /// @brief  ALayout and ADataType are expected to be scalars, not a tuple.
    static_assert(!is_tuple_v<ALayout> && !is_tuple_v<ADataType>,
                  "ALayout and ADataType must be scalars.");

    /// @brief  BLayout and BDataType are expected to be scalars, not a tuple.
    static_assert(!is_tuple_v<BLayout> && !is_tuple_v<BDataType>,
                  "BLayout and BDataType must be scalars.");

    /// @brief  CLayout and CDataType are expected to be scalars, not a tuple.
    static_assert(!is_tuple_v<CLayout> && !is_tuple_v<CDataType>,
                  "CLayout and CDataType must be scalars.");

    struct StreamKKernelArgs : ck_tile::UniversalGemmKernelArgs<>
    {
        StreamKKernelArgs(const StreamKHostArgs& host_args, index_t grid)
            : UniversalGemmKernelArgs{host_args.as_ptr,
                                      host_args.bs_ptr,
                                      host_args.ds_ptr,
                                      host_args.e_ptr,
                                      host_args.M,
                                      host_args.N,
                                      host_args.K,
                                      host_args.stride_As,
                                      host_args.stride_Bs,
                                      host_args.stride_Ds,
                                      host_args.stride_E,
                                      host_args.k_batch},
              reduction_strategy{host_args.reduction_strategy},
              // The workspace pointer is set to nullptr because we must first
              // instantiate the TilePartitioner to get the necessary size
              workspace_ptr{nullptr},
              tile_partitioner{TilePartitioner{host_args.M, host_args.N, host_args.K, grid}}

        {
        }

        /// @brief  The strategy used by work groups to compute final results in C tensor.
        StreamKReductionStrategy reduction_strategy;
        /// @brief  A pointer to a buffer in device memory for accumulating partial via reduction
        /// strategy.
        void* workspace_ptr;
        /// @brief  An instance of the TilePartioner class for assisting with mapping workgroups to
        /// the C tensor.
        TilePartitioner tile_partitioner;
    };

    using KernelArgs = StreamKKernelArgs;
    using Kernel     = StreamKKernel<TilePartitioner, GemmPipeline, EpiloguePipeline>;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        using P_ = GemmPipeline;
        using WarpTile = typename P_::BlockGemmShape::WarpTile;

        return concat('_', "streamk", gemm_prec_str<ADataType, BDataType>(),
                      concat('x', P_::MPerBlock, P_::NPerBlock, P_::KPerBlock),
                      concat('x', WarpTile::at(number<0>{}), WarpTile::at(number<1>{}), WarpTile::at(number<2>{})),
                      concat('x', P_::GetVectorSizeA(), P_::GetVectorSizeB(), P_::GetVectorSizeC()),
                      concat('x', P_::kPadM, P_::kPadN, P_::kPadK));
        // clang-format on
    }

    /// @brief Compute the grid size for the Stream K kernel using the tile_partitioner.
    /// @return The grid size.
    CK_TILE_HOST static auto GridSize(const TilePartitioner& tile_partitioner) -> dim3
    {
        return tile_partitioner.grid_size();
    }

    /// @brief Get the maximum occupancy grid size for the persistent kernel on the current device.
    /// @return The maximum occupancy grid size.
    /// @note This function queries the maximum occupancy of the kernel using
    /// `hipOccupancyMaxActiveBlocksPerMultiprocessor`.
    CK_TILE_HOST static auto MaxOccupancyGridSize(const stream_config& s) -> dim3
    {
        return UniversalGemmKernel::MaxOccupancyGridSize(s);
    }

    CK_TILE_HOST static constexpr auto BlockSize() -> dim3
    {
        return UniversalGemmKernel::BlockSize();
    }

    /// @brief Constructs kernel arguments for the Stream-K kernel.
    /// @param host_args Stream-K host arguments.
    /// @param num_cu Number of compute units (CUs). The default is the number of CUs on the device.
    /// The caller may select their own to assist with test reproducibility, etc.
    /// @param occupancy The maximum number of active blocks per CU for this kernel. The caller may
    /// select their own to assist with test reproducibility, etc.
    /// @return The kernel arguments for Stream-K.
    CK_TILE_HOST static StreamKKernelArgs MakeKernelArgs(const StreamKHostArgs& host_args,
                                                         int num_cu    = NumCU(),
                                                         int occupancy = Occupancy())
    {
        const index_t grid = num_cu * occupancy;

        return StreamKKernelArgs{host_args, grid};
    }

    template <bool UseDefaultScheduler = true>
    CK_TILE_DEVICE static void
    RunGemm(const std::array<const ADataType*, UniversalGemmKernel::NumATensor>& as_ptr,
            const std::array<const BDataType*, UniversalGemmKernel::NumBTensor>& bs_ptr,
            const std::array<const void*, UniversalGemmKernel::NumDTensor>& ds_ptr,
            CDataType* c_ptr,
            void* smem_ptr_0,
            const typename UniversalGemmKernel::KernelArgs& kargs,
            const index_t num_loop,
            const index_t block_idx_m,
            const index_t block_idx_n,
            const index_t k_size)
    {
        // Create Gemm tensor views, pad views and tile windows
        const auto& gemm_tensor_views_tuple =
            UniversalGemmKernel::template MakeGemmTensorViews<EpiloguePipeline::MemoryOperation>(
                as_ptr, bs_ptr, ds_ptr, c_ptr, kargs, k_size);

        const auto& gemm_pad_views = UniversalGemmKernel::MakeGemmPadViews(gemm_tensor_views_tuple);
        auto gemm_tile_windows =
            UniversalGemmKernel::MakeGemmTileWindows(gemm_pad_views, block_idx_m, block_idx_n);

        // Run GEMM cooperatively by whole workgroup.
        const auto& as_block_window = gemm_tile_windows.at(UniversalGemmKernel::I0);
        const auto& bs_block_window = gemm_tile_windows.at(UniversalGemmKernel::I1);
        const auto& ds_block_window = gemm_tile_windows.at(UniversalGemmKernel::I2);

        // Since num_loop can vary per WG and per iteration of the Stream-K while loop, we compute
        // has_hot_loop and tail_num here. This is a similar pattern used by grouped GEMM. In this
        // case, we call the GemmPipeline's operator() function that takes both has_hot_loop and
        // tail_num.
        const bool has_hot_loop   = GemmPipeline::BlockHasHotloop(num_loop);
        const TailNumber tail_num = GemmPipeline::GetBlockLoopTailNum(num_loop);

        const auto& c_block_tile = GemmPipeline{}(as_block_window[UniversalGemmKernel::I0],
                                                  bs_block_window[UniversalGemmKernel::I0],
                                                  num_loop,
                                                  has_hot_loop,
                                                  tail_num,
                                                  smem_ptr_0);

        if(UseDefaultScheduler || (get_warp_id() == 0))
        {
            // Run Epilogue Pipeline
            auto& c_block_window = gemm_tile_windows.at(UniversalGemmKernel::I3);

            EpiloguePipeline{}(c_block_window, c_block_tile, ds_block_window, smem_ptr_0);
        }
    }

    CK_TILE_HOST static bool IsSupportedArgument(const StreamKKernelArgs& kargs)
    {
        return UniversalGemmKernel::IsSupportedArgument(kargs);
    }

    /// @brief Computes the buffer size needed to store accumulation results for Stream K.
    /// @return The buffer size needed.
    CK_TILE_HOST static uint32_t GetWorkSpaceSize(const StreamKKernelArgs& kargs)
    {
        return kargs.tile_partitioner.get_workspace_size(sizeof(AccDataType));
    }

    /// @brief Sets the kargs' current workspace_ptr to the given workspace_ptr.
    /// @note Assumes that the given workspace_ptr points to allocated device memory.
    CK_TILE_HOST static void SetWorkSpacePointer(StreamKKernelArgs& kargs, void* workspace_ptr)
    {
        kargs.workspace_ptr = workspace_ptr;
    }

    /// @brief Computes offsets into A, B, and C tensors then runs the GEMM pipeline and epilogue.
    /// @param kargs Stream-K kernel arguments.
    /// @param tile_idx The 1D tile index in the C tensor for this workgroup.
    /// @param num_loop The number of iterations (at the macro tile level) in the K dimension this
    /// workgroup will perform in the C tile.
    /// @param i_k_a The K offset in the A tensor.
    /// @param i_k_b The K offset in the B tensor.
    /// @param k_size The portion of the K dimension this workgroup processes in the assigned
    /// `tile_idx`.
    /// @param smem_ptr_0 Pointer to LDS.
    CK_TILE_DEVICE void BaseGemm(StreamKKernelArgs& kargs,
                                 index_t tile_idx,
                                 index_t num_loop,
                                 index_t i_k_a,
                                 index_t i_k_b,
                                 index_t k_size,
                                 void* smem_ptr_0) const
    {
        const auto c_macro_tile_idx = kargs.tile_partitioner.get_output_tile_index(tile_idx);
        index_t i_m = c_macro_tile_idx[UniversalGemmKernel::I0] * TilePartitioner::MPerBlock;
        index_t i_n = c_macro_tile_idx[UniversalGemmKernel::I1] * TilePartitioner::NPerBlock;

        const ADataType* a_ptr = static_cast<const ADataType*>(kargs.as_ptr[0]) + i_k_a;
        const BDataType* b_ptr = static_cast<const BDataType*>(kargs.bs_ptr[0]) + i_k_b;
        CDataType* c_ptr       = static_cast<CDataType*>(kargs.e_ptr);

        // Run the GEMM pipeline and Epilogue.
        RunGemm(
            {a_ptr}, {b_ptr}, {/*ds_ptr*/}, c_ptr, smem_ptr_0, kargs, num_loop, i_m, i_n, k_size);
    }

    /// @brief Signals that the current thread block (CTA) has completed storing its partial
    /// results.
    /// @param kargs Kernel arguments, including the workspace pointer.
    /// @param cta_idx The index of the current thread block (CTA).
    /// @note This function utilizes a workgroup barrier to set a synchronization flag for the given
    /// CTA index.
    CK_TILE_DEVICE void SignalStorePartialDone(const StreamKKernelArgs& kargs,
                                               index_t cta_idx) const
    {
        auto sk_flags_ptr = static_cast<uint32_t*>(kargs.workspace_ptr);
        workgroup_barrier sk_flags(sk_flags_ptr);
        sk_flags.wait_set(0, 1, cta_idx);
    }

    /// @brief Waits for the thread block (cta_idx) to complete storing its partial results.
    /// @param kargs Kernel arguments, including the workspace pointer.
    /// @param cta_idx The index of the thread block (CTA).
    /// @note This function utilizes a workgroup barrier to wait for the synchronization flag to be
    /// set by the given CTA index.
    CK_TILE_DEVICE void WaitStorePartialDone(const StreamKKernelArgs& kargs, index_t cta_idx) const
    {
        auto sk_flags_ptr = static_cast<uint32_t*>(kargs.workspace_ptr);
        workgroup_barrier sk_flags(sk_flags_ptr);
        sk_flags.wait_eq(1, cta_idx);
    }

    /// @brief Adds the values of a block tile to an output block tile.
    /// @param in_out_block_tile The output block tile to which values are added.
    /// @param in_block_tile The input block tile whose values are added.
    /// @note This function iterates over the distributed spans of the block tiles and updates the
    /// output block tile with accumulated values.
    template <typename OAccTile>
    CK_TILE_DEVICE void AddBlockTile(OAccTile& in_out_block_tile,
                                     const OAccTile& in_block_tile) const
    {
        using BlockType        = remove_cvref_t<decltype(in_out_block_tile)>;
        constexpr auto o_spans = BlockType::get_distributed_spans();
        sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                constexpr auto idx     = make_tuple(idx0, idx1);
                in_out_block_tile(idx) = in_out_block_tile[idx] + in_block_tile[idx];
            });
        });
    }

    /// @brief Loads a partial block tile from the workspace buffer.
    /// @param kargs Kernel arguments, including the workspace pointer.
    /// @param cta_idx The index of the thread block (CTA).
    /// @param c_block_tile_dist The tile distribution for the block.
    /// @return The loaded partial block tile.
    /// @note This function calculates the buffer pointer and uses the tile distribution for loading
    /// the partial block tile.
    template <typename DataType, typename OAccTileDist>
    CK_TILE_DEVICE auto LoadPartial(const StreamKKernelArgs& kargs,
                                    index_t cta_idx,
                                    const OAccTileDist& c_block_tile_dist) const
    {
        const auto c_block_tile_buffer_size =
            TilePartitioner::MPerBlock * TilePartitioner::NPerBlock * sizeof(DataType);
        void* partial_buffer_ptr = static_cast<char*>(kargs.workspace_ptr) +
                                   kargs.tile_partitioner.get_flags_buffer_size() +
                                   cta_idx * c_block_tile_buffer_size;

        const auto& partial_tensor_view = make_naive_tensor_view<address_space_enum::global>(
            static_cast<DataType*>(partial_buffer_ptr),
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            make_tuple(TilePartitioner::NPerBlock, 1),
            number<GemmPipeline::GetVectorSizeC()>{},
            number<1>{});

        auto partial_tile_window = make_tile_window(
            partial_tensor_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {0, 0},
            c_block_tile_dist);

        return load_tile(partial_tile_window);
    }

    /// @brief Stores a partial block tile to the workspace buffer.
    /// @param kargs Kernel arguments, including the workspace pointer.
    /// @param cta_idx The index of the thread block (CTA).
    /// @param c_block_tile The block tile to be stored.
    /// @note This function calculates the buffer pointer and uses the tile window for storing the
    /// partial block tile.
    template <typename OAccTile>
    CK_TILE_DEVICE void StorePartial(const StreamKKernelArgs& kargs,
                                     index_t cta_idx,
                                     const OAccTile& c_block_tile) const
    {
        const auto c_block_tile_buffer_size = TilePartitioner::MPerBlock *
                                              TilePartitioner::NPerBlock *
                                              sizeof(typename OAccTile::DataType);
        void* partial_buffer_ptr = static_cast<char*>(kargs.workspace_ptr) +
                                   kargs.tile_partitioner.get_flags_buffer_size() +
                                   cta_idx * c_block_tile_buffer_size;

        const auto& partial_tensor_view = make_naive_tensor_view<address_space_enum::global>(
            static_cast<typename OAccTile::DataType*>(partial_buffer_ptr),
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            make_tuple(TilePartitioner::NPerBlock, 1),
            number<GemmPipeline::GetVectorSizeC()>{},
            number<1>{});

        auto partial_tile_window = make_tile_window(
            partial_tensor_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {0, 0});

        store_tile(partial_tile_window, c_block_tile);
    }

    /// @brief Runs the main Stream-K algorithm.
    /// @param kargs Stream-K kernel arguments.
    /// @param cta_idx The current Stream-K workgroup's index.
    /// @param smem_ptr_0 Pointer to LDS.
    /// @note It is assumed that the first Stream-K workgroup has a `cta_idx` of zero. If a
    /// non-persistent data-parallel (DP) section is used, then a Stream-K workgroup's `cta_idx`
    /// should be something like `blockIdx.x` minus number of DP workgroups.
    CK_TILE_DEVICE void
    StreamKGemm(StreamKKernelArgs& kargs, index_t cta_idx, void* smem_ptr_0) const
    {
        index_t iter_start, iter_end;
        kargs.tile_partitioner.get_iter_boundaries(iter_start, iter_end, cta_idx);

        while(iter_start < iter_end)
        {
            // Get the 1D tile index in the C tensor that this workgroup will work in for this
            // iteration of the loop.
            index_t tile_idx =
                amd_wave_read_first_lane(kargs.tile_partitioner.get_tile_index(iter_start));

            // Get the start and end boundaries for the current tile.
            index_t tile_iter_start, tile_iter_end;
            kargs.tile_partitioner.get_tile_boundaries(tile_iter_start, tile_iter_end, tile_idx);

            // Get the start and end iteration within the current tile for the workgroup.
            index_t local_iter_start = amd_wave_read_first_lane(
                kargs.tile_partitioner.get_local_iter(iter_start, tile_iter_start));
            index_t local_iter_end =
                amd_wave_read_first_lane(kargs.tile_partitioner.get_local_iter_end(
                    tile_iter_start, iter_end, tile_iter_end));

            // Get the iteration length.
            index_t num_loop_sk = local_iter_end - local_iter_start;

            // Determine the total size along the K dimension the workgroup is using in this
            // iteration (used to construct tensor views).
            index_t k_size = num_loop_sk * TilePartitioner::KPerBlock;

            // Get the K offsets for the A and B tensors
            auto [i_k_a, i_k_b] = GetKOffsets<ALayout, BLayout>(
                local_iter_start, kargs.stride_As[0], kargs.stride_Bs[0]);

            if constexpr(TilePartitioner::ReductionStrategy == StreamKReductionStrategy::Atomic)
            {
                BaseGemm(kargs, tile_idx, num_loop_sk, i_k_a, i_k_b, k_size, smem_ptr_0);
            }
            else
            {
                const auto c_macro_tile_idx =
                    kargs.tile_partitioner.get_output_tile_index(tile_idx);
                index_t i_m =
                    c_macro_tile_idx[UniversalGemmKernel::I0] * TilePartitioner::MPerBlock;
                index_t i_n =
                    c_macro_tile_idx[UniversalGemmKernel::I1] * TilePartitioner::NPerBlock;

                const ADataType* a_ptr = static_cast<const ADataType*>(kargs.as_ptr[0]) + i_k_a;
                const BDataType* b_ptr = static_cast<const BDataType*>(kargs.bs_ptr[0]) + i_k_b;
                CDataType* c_ptr       = static_cast<CDataType*>(kargs.e_ptr);

                // Create Gemm tensor views, pad views and tile windows
                const auto& gemm_tensor_views_tuple =
                    UniversalGemmKernel::template MakeGemmTensorViews<
                        EpiloguePipeline::MemoryOperation>(
                        {a_ptr}, {b_ptr}, {/*ds_ptr*/}, c_ptr, kargs, k_size);

                const auto& gemm_pad_views =
                    UniversalGemmKernel::MakeGemmPadViews(gemm_tensor_views_tuple);
                auto gemm_tile_windows =
                    UniversalGemmKernel::MakeGemmTileWindows(gemm_pad_views, i_m, i_n);

                // Run GEMM cooperatively by whole workgroup.
                const auto& as_block_window = gemm_tile_windows.at(UniversalGemmKernel::I0);
                const auto& bs_block_window = gemm_tile_windows.at(UniversalGemmKernel::I1);
                const auto& ds_block_window = gemm_tile_windows.at(UniversalGemmKernel::I2);

                // Since num_loop can vary per WG and per iteration of the Stream-K while loop,
                // we compute has_hot_loop and tail_num here. This is a similar pattern used by
                // grouped GEMM. In this case, we call the GemmPipeline's operator() function
                // that takes both has_hot_loop and tail_num.
                const bool has_hot_loop   = GemmPipeline::BlockHasHotloop(num_loop_sk);
                const TailNumber tail_num = GemmPipeline::GetBlockLoopTailNum(num_loop_sk);

                const auto& c_block_tile = GemmPipeline{}(as_block_window[UniversalGemmKernel::I0],
                                                          bs_block_window[UniversalGemmKernel::I0],
                                                          num_loop_sk,
                                                          has_hot_loop,
                                                          tail_num,
                                                          smem_ptr_0);

                auto tile_started = iter_start == tile_iter_start;
                auto tile_ended   = iter_end >= tile_iter_end;
                if(!tile_started)
                {
                    StorePartial(kargs, cta_idx, c_block_tile);
                    // Ensure device-wide visibility of partial results stored in global memory
                    // before signaling completion. __threadfence() guarantees that all global
                    // memory writes by this thread are visible to other threads on the device.
                    __threadfence(); // send signal when the store is done
                    SignalStorePartialDone(kargs, cta_idx);
                }
                else
                {
                    auto accum_block_tile = c_block_tile;
                    if(!tile_ended)
                    {
                        const index_t iter_per_tile = kargs.tile_partitioner.get_iters_per_tile();
                        const index_t iter_per_cta  = kargs.tile_partitioner.get_iters_per_sk_cta();
                        const index_t extra_iters   = kargs.tile_partitioner.get_extra_iters();
                        int accum_iters             = local_iter_end - local_iter_start;
                        int next_cta                = cta_idx + 1;

                        while(accum_iters < iter_per_tile)
                        {
                            WaitStorePartialDone(kargs, next_cta);

                            using BlockType = remove_cvref_t<decltype(c_block_tile)>;
                            AddBlockTile(
                                accum_block_tile,
                                LoadPartial<typename BlockType::DataType>(
                                    kargs, next_cta, c_block_tile.get_tile_distribution()));

                            accum_iters += iter_per_cta + (next_cta < extra_iters);
                            ++next_cta;
                        }
                    }

                    auto& c_block_window = gemm_tile_windows.at(UniversalGemmKernel::I3);
                    EpiloguePipeline{}(
                        c_block_window, accum_block_tile, ds_block_window, smem_ptr_0);
                }
            }

            // Prepare for next Stream-K loop iteration.
            iter_start = tile_iter_end;
            block_sync_lds();
        }
    }

    /// @brief Entry point for the Stream-K Kernel with non-persistent DP.
    ///
    /// @par Overview
    ///     For the Non-Persistent kernel, each data parallel workgroup will
    ///     compute the results for their assigned macro-tile by calling `BaseGemm()`.
    ///     The Stream-K workgroups will do their assigned work by calling
    ///     `StreamKGemm()`, which calls `BaseGemm()` in the Stream-K loop.
    template <bool U = PersistentDP>
    CK_TILE_DEVICE typename std::enable_if_t<!U> operator()(StreamKKernelArgs kargs) const
    {
        // Allocate LDS
        __shared__ char smem_ptr_0[UniversalGemmKernel::GetSmemSize()];

        index_t block_idx   = ck_tile::get_block_1d_id();
        index_t dp_num_loop = kargs.tile_partitioner.get_iters_per_tile();
        index_t dp_ctas     = kargs.tile_partitioner.get_dp_ctas();
        bool is_dp_ctas     = block_idx < kargs.tile_partitioner.get_dp_ctas();

        // Check if at the data parallel section
        if(is_dp_ctas)
        {
            BaseGemm(kargs, block_idx, dp_num_loop, 0, 0, kargs.K, smem_ptr_0);
        }
        else
        {
            // Stream-K
            StreamKGemm(kargs, block_idx - dp_ctas, smem_ptr_0);
        }
    }

    /// @brief Entry point for the Stream-K Kernel with persistent DP.
    ///
    /// @par Overview
    ///     For the Persistent kernel, each workgroup will first compute their
    ///     assigned data-parallel tiles. Each data parallel tile will be computed
    ///     by calling `BaseGemm()`. Then the workgroups will proceed with the
    ///     Stream-K portion by calling `StreamKGemm()`, which calls `BaseGemm()`
    ///     in the Stream-K loop.
    template <bool U = PersistentDP>
    CK_TILE_DEVICE typename std::enable_if_t<U> operator()(StreamKKernelArgs kargs) const
    {
        // Allocate LDS
        __shared__ char smem_ptr_0[UniversalGemmKernel::GetSmemSize()];

        index_t block_idx   = ck_tile::get_block_1d_id();
        index_t dp_num_loop = kargs.tile_partitioner.get_iters_per_tile();

        // Data-parallel section
        for(index_t tile_idx = block_idx; tile_idx < kargs.tile_partitioner.get_dp_tiles();
            tile_idx += kargs.tile_partitioner.get_grid())
        {
            BaseGemm(kargs, tile_idx, dp_num_loop, 0, 0, kargs.K, smem_ptr_0);
        }

        // Stream-K section
        StreamKGemm(kargs, block_idx, smem_ptr_0);
    }

    private:
    /// @brief Computes the K offsets in the A and B tensors given iter_offset, where iter_offset is
    /// the starting macro tile index in the K dimension for the workgroup.
    /// @return A tuple containing the offsets into the A and B tensors accounting for the layouts
    /// of A and B.
    /// @note The default case is that A is assumed to be row major and B is assumed to be column
    /// major.
    template <typename ALayout, typename BLayout>
    CK_TILE_DEVICE static tuple<index_t, index_t>
    GetKOffsets(index_t iter_offset, index_t stride_a, index_t stride_b)
    {
        index_t stride_offset_a;
        index_t stride_offset_b;
        if constexpr(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::ColumnMajor>)
        {
            stride_offset_a = stride_a;
        }
        else
        {
            stride_offset_a = 1;
        }

        if constexpr(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            stride_offset_b = stride_b;
        }
        else
        {
            stride_offset_b = 1;
        }

        index_t base_offset = iter_offset * TilePartitioner::KPerBlock;

        return make_tuple(base_offset * stride_offset_a, base_offset * stride_offset_b);
    }

    CK_TILE_HOST static int NumCU()
    {
        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        hip_check_error(hipGetDevice(&dev));
        hip_check_error(hipGetDeviceProperties(&dev_prop, dev));
        int num_cu = dev_prop.multiProcessorCount;

        return num_cu;
    }

    /// @brief Computes the occupancy (i.e. maximum number of active blocks per CU) for the kernel
    /// @return The occupancy
    /// @note This function queries the maximum occupancy of the kernel using
    /// `hipOccupancyMaxActiveBlocksPerMultiprocessor`.
    CK_TILE_HOST static int Occupancy()
    {
        int occupancy;

        // Since occupancy of 1 is valid for stream k, we set min_num_block_per_cu to 1
        constexpr int min_block_per_cu = 1;
        const auto kernel              = kentry<min_block_per_cu, Kernel, KernelArgs>;

        hip_check_error(
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, kBlockSize, 0));

        return occupancy;
    }
};
} // namespace reboot

/// @brief The Stream K GEMM kernel host arguments.
///
/// @par Overview
///      This structure is passed to @ref StreamKKernel "StreamKKernel" when creating the kernel
///      arguments object. It contains all necessary information required to build proper kernel
///      arguments and launch the kernel on GPU. This structure defines the GEMM problem
///      configuration by stating all required information like M,N,K sizes and respective strides.
struct StreamKHostArgs : public ck_tile::UniversalGemmHostArgs<>
{
    CK_TILE_HOST explicit StreamKHostArgs(const void* a_ptr_,
                                          const void* b_ptr_,
                                          void* c_ptr_,
                                          index_t M_,
                                          index_t N_,
                                          index_t K_,
                                          index_t stride_A_,
                                          index_t stride_B_,
                                          index_t stride_C_,
                                          StreamKReductionStrategy reduction_strategy_,
                                          uint32_t num_sk_blocks_ = 0xffffffff)
        : UniversalGemmHostArgs<>({a_ptr_},
                                  {b_ptr_},
                                  {/*ds_ptr*/},
                                  c_ptr_,
                                  /*k_batch_ =*/1,
                                  M_,
                                  N_,
                                  K_,
                                  {stride_A_},
                                  {stride_B_},
                                  {/*stride_Ds_*/},
                                  stride_C_),
          reduction_strategy{reduction_strategy_},
          num_sk_blocks{num_sk_blocks_}
    {
    }

    ck_tile::StreamKReductionStrategy reduction_strategy;
    uint32_t num_sk_blocks;
};

template <typename TilePartitioner_, typename GemmPipeline_, typename EpiloguePipeline_>
struct StreamKKernel
{
    /// @brief Inject the UniversalGemmKernel base class to support execution of all necessary
    /// functions.
    using UniversalGemmKernel =
        UniversalGemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>;

    static constexpr index_t kBlockSize = UniversalGemmKernel::kBlockSize;

    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline     = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;

    /// @brief  Specify the layout configurations for A, B, and C
    using ALayout = remove_cvref_t<typename GemmPipeline::ALayout>;
    using BLayout = remove_cvref_t<typename GemmPipeline::BLayout>;
    using CLayout = remove_cvref_t<typename GemmPipeline::CLayout>;

    /// @brief  Specify the data type configurations for A, B, and C
    using ADataType = remove_cvref_t<typename GemmPipeline::ADataType>;
    using BDataType = remove_cvref_t<typename GemmPipeline::BDataType>;
    using CDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    /// @brief  ALayout and ADataType are expected to be scalars, not a tuple.
    static_assert(!is_detected<is_tuple, ALayout>::value &&
                      !is_detected<is_tuple, ADataType>::value,
                  "ALayout and ADataType must be scalars.");

    /// @brief  BLayout and BDataType are expected to be scalars, not a tuple.
    static_assert(!is_detected<is_tuple, BLayout>::value &&
                      !is_detected<is_tuple, BDataType>::value,
                  "BLayout and BDataType must be scalars.");

    /// @brief  CLayout and CDataType are expected to be scalars, not a tuple.
    static_assert(!is_detected<is_tuple, CLayout>::value &&
                      !is_detected<is_tuple, CDataType>::value,
                  "CLayout and CDataType must be scalars.");

    struct StreamKKernelArgs : ck_tile::UniversalGemmKernelArgs<>
    {
        /// @brief  The strategy used by work groups to compute final results in C tensor.
        StreamKReductionStrategy reduction_strategy;
        /// @brief  The number of stream k blocks.
        uint32_t num_sk_blocks;
        /// @brief  A pointer to a buffer in device memory for accumulating partial via reduction
        /// strategy.
        void* workspace_ptr;
        /// @brief  An instance of the TilePartioner class for assisting with mapping workgroups to
        /// the C tensor.
        TilePartitioner tile_partitioner;
    };

    using KernelArgs = StreamKKernelArgs;
    using Kernel     = StreamKKernel<TilePartitioner, GemmPipeline, EpiloguePipeline>;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        using P_ = GemmPipeline;
        using WarpTile = typename P_::BlockGemmShape::WarpTile;

        return concat('_', "streamk", gemm_prec_str<ADataType, BDataType>(),
                      concat('x', P_::MPerBlock, P_::NPerBlock, P_::KPerBlock),
                      concat('x', WarpTile::at(number<0>{}), WarpTile::at(number<1>{}), WarpTile::at(number<2>{})),
                      concat('x', P_::GetVectorSizeA(), P_::GetVectorSizeB(), P_::GetVectorSizeC()),
                      concat('x', P_::kPadM, P_::kPadN, P_::kPadK));
        // clang-format on
    }

    /// @brief Compute the grid size for the Stream K kernel using the tile_partitioner.
    /// @return The grid size.
    CK_TILE_HOST static auto GridSize(const TilePartitioner& tile_partitioner) -> dim3
    {
        return tile_partitioner.GridSize();
    }

    /// @brief Get the maximum occupancy grid size for the persistent kernel on the current device.
    /// @return The maximum occupancy grid size.
    /// @note This function queries the maximum occupancy of the kernel using
    /// `hipOccupancyMaxActiveBlocksPerMultiprocessor`.
    CK_TILE_HOST static auto MaxOccupancyGridSize(const stream_config& s) -> dim3
    {
        return UniversalGemmKernel::MaxOccupancyGridSize(s);
    }

    CK_TILE_HOST static constexpr auto BlockSize() -> dim3
    {
        return UniversalGemmKernel::BlockSize();
    }

    /// @brief Constructs kernel arguments for the Stream-K kernel.
    /// @param host_args Stream-K host arguments.
    /// @param num_cu Number of compute units (CUs). The default is the number of CUs on the device.
    /// The caller may select their own to assist with test reproducibility, etc.
    /// @param occupancy The maximum number of active blocks per CU for this kernel. The caller may
    /// select their own to assist with test reproducibility, etc.
    /// @return The kernel arguments for Stream-K.
    CK_TILE_HOST static StreamKKernelArgs MakeKernelArgs(const StreamKHostArgs& host_args,
                                                         int num_cu    = NumCU(),
                                                         int occupancy = Occupancy())
    {
        return StreamKKernelArgs{{host_args.as_ptr,
                                  host_args.bs_ptr,
                                  host_args.ds_ptr,
                                  host_args.e_ptr,
                                  host_args.M,
                                  host_args.N,
                                  host_args.K,
                                  host_args.stride_As,
                                  host_args.stride_Bs,
                                  host_args.stride_Ds,
                                  host_args.stride_E,
                                  host_args.k_batch},
                                 host_args.reduction_strategy,
                                 host_args.num_sk_blocks,
                                 // The workspace pointer is set to nullptr because we must first
                                 // instantiate the TilePartitioner to get the necessary size
                                 /*workspace_ptr =*/nullptr,
                                 TilePartitioner{static_cast<uint32_t>(host_args.M),
                                                 static_cast<uint32_t>(host_args.N),
                                                 static_cast<uint32_t>(host_args.K),
                                                 static_cast<uint32_t>(num_cu),
                                                 static_cast<uint32_t>(occupancy),
                                                 host_args.num_sk_blocks}};
    }

    template <bool UseDefaultScheduler = true>
    CK_TILE_DEVICE static void
    RunGemm(const std::array<const ADataType*, UniversalGemmKernel::NumATensor>& as_ptr,
            const std::array<const BDataType*, UniversalGemmKernel::NumBTensor>& bs_ptr,
            const std::array<const void*, UniversalGemmKernel::NumDTensor>& ds_ptr,
            CDataType* c_ptr,
            void* smem_ptr_0,
            const typename UniversalGemmKernel::KernelArgs& kargs,
            const index_t num_loop,
            const index_t block_idx_m,
            const index_t block_idx_n,
            const index_t k_size)
    {
        // Create Gemm tensor views, pad views and tile windows
        const auto& gemm_tensor_views_tuple =
            UniversalGemmKernel::template MakeGemmTensorViews<EpiloguePipeline::MemoryOperation>(
                as_ptr, bs_ptr, ds_ptr, c_ptr, kargs, k_size);

        const auto& gemm_pad_views = UniversalGemmKernel::MakeGemmPadViews(gemm_tensor_views_tuple);
        auto gemm_tile_windows =
            UniversalGemmKernel::MakeGemmTileWindows(gemm_pad_views, block_idx_m, block_idx_n);

        // Run GEMM cooperatively by whole workgroup.
        const auto& as_block_window = gemm_tile_windows.at(UniversalGemmKernel::I0);
        const auto& bs_block_window = gemm_tile_windows.at(UniversalGemmKernel::I1);
        const auto& ds_block_window = gemm_tile_windows.at(UniversalGemmKernel::I2);

        // Since num_loop can vary per WG and per iteration of the Stream-K while loop, we compute
        // has_hot_loop and tail_num here. This is a similar pattern used by grouped GEMM. In this
        // case, we call the GemmPipeline's operator() function that takes both has_hot_loop and
        // tail_num.
        const bool has_hot_loop   = GemmPipeline::BlockHasHotloop(num_loop);
        const TailNumber tail_num = GemmPipeline::GetBlockLoopTailNum(num_loop);

        const auto& c_block_tile = GemmPipeline{}(as_block_window[UniversalGemmKernel::I0],
                                                  bs_block_window[UniversalGemmKernel::I0],
                                                  num_loop,
                                                  has_hot_loop,
                                                  tail_num,
                                                  smem_ptr_0);

        if(UseDefaultScheduler || (get_warp_id() == 0))
        {
            // Run Epilogue Pipeline
            auto& c_block_window = gemm_tile_windows.at(UniversalGemmKernel::I3);

            EpiloguePipeline{}(c_block_window, c_block_tile, ds_block_window, smem_ptr_0);
        }
    }

    CK_TILE_HOST static bool IsSupportedArgument(const StreamKKernelArgs& kargs)
    {
        if(kargs.reduction_strategy == StreamKReductionStrategy::Reduction)
        {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR("CK Tile Stream-K only supports the atomic reduction strategy.");
            }
            return false;
        }
        return UniversalGemmKernel::IsSupportedArgument(kargs);
    }

    /// @brief Computes the buffer size needed to store accumulation results for Stream K.
    /// @return The buffer size needed.
    CK_TILE_HOST static uint32_t GetWorkSpaceSize(const StreamKKernelArgs& kargs)
    {
        // For reduction, we need to determine the amount of device space for acculumation
        // results and semaphores.
        if(kargs.reduction_strategy == ck_tile::StreamKReductionStrategy::Reduction)
        {
            return kargs.tile_partitioner.GetWorkSpaceSize(sizeof(CDataType));
        }

        // Otherwise, no additional space is needed since blocks atomically store their results.
        return 0;
    }

    /// @brief Sets the kargs' current workspace_ptr to the given workspace_ptr.
    /// @note Assumes that the given workspace_ptr points to allocated device memory.
    CK_TILE_HOST static void SetWorkSpacePointer(StreamKKernelArgs& kargs, void* workspace_ptr)
    {
        kargs.workspace_ptr = workspace_ptr;
    }

    /// @brief Entry point for the Stream-K Kernel, performing the main Stream-K loop.
    CK_TILE_DEVICE void operator()(StreamKKernelArgs kargs) const
    {
        // Allocate LDS
        __shared__ char smem_ptr_0[UniversalGemmKernel::GetSmemSize()];

        uint32_t block_idx = ck_tile::get_block_1d_id();

        bool is_padding_block =
            amd_wave_read_first_lane(block_idx >= kargs.tile_partitioner.sk_num_blocks &&
                                     block_idx < kargs.tile_partitioner.dp_start_block_idx);

        // Padding blocks make it such that the DP blocks are aligned with the number of CUs; they
        // should not partake in the GEMM
        if(is_padding_block)
            return;

        // Determine the K offset of the first and final macro tile in the A and B tensors along the
        // K dimension.
        uint32_t iter_start, iter_end;
        kargs.tile_partitioner.GetBlockItr(block_idx, iter_start, iter_end);

        // Main Stream-K loop
        while(true)
        {
            // Determine the number of macro tiles in A and B this WG is resposible for in the
            // current C macro tile.
            uint32_t current_iter_length = amd_wave_read_first_lane(
                kargs.tile_partitioner.GetCurrentIterLength(iter_start, iter_end));

            // Determine the 1D tile_idx and the iter_offset for this WG.
            // The tile_idx is the 1D macro tile index in the C tensor.
            // The iter_offset is the starting macro tile index in the K dimension for the WG in the
            // current iteration of the while loop.
            uint32_t tile_idx, iter_offset;
            kargs.tile_partitioner.GetTileIdxWithOffset(iter_start, tile_idx, iter_offset);

            // Get the 2D tile index in the C tensor for this WG using the 1D index (i.e. tile_idx)
            auto spatial_idx = kargs.tile_partitioner.GetOutputTileIndex(tile_idx);

            // Get the offsets in A, B, C tensors.
            index_t i_m         = static_cast<index_t>(spatial_idx[UniversalGemmKernel::I0] *
                                               TilePartitioner::MPerBlock);
            index_t i_n         = static_cast<index_t>(spatial_idx[UniversalGemmKernel::I1] *
                                               TilePartitioner::NPerBlock);
            auto [i_k_a, i_k_b] = GetKOffsets<ALayout, BLayout>(
                static_cast<index_t>(iter_offset), kargs.stride_As[0], kargs.stride_Bs[0]);

            // Determine the total size along the K dimension the WG is using in this iteration
            // (used to construct tensor views).
            index_t k_size = static_cast<index_t>(current_iter_length * TilePartitioner::KPerBlock);

            // Update pointer offsets for A, B, and C.
            const ADataType* a_ptr = static_cast<const ADataType*>(kargs.as_ptr[0]) + i_k_a;
            const BDataType* b_ptr = static_cast<const BDataType*>(kargs.bs_ptr[0]) + i_k_b;
            CDataType* c_ptr       = static_cast<CDataType*>(kargs.e_ptr);

            // Run the GEMM pipeline and Epilogue.
            RunGemm({a_ptr},
                    {b_ptr},
                    {/*ds_ptr*/},
                    c_ptr,
                    smem_ptr_0,
                    kargs,
                    current_iter_length,
                    i_m,
                    i_n,
                    k_size);

            // Prepare for next Stream-K loop iteration.
            iter_start += current_iter_length;
            if(iter_end <= iter_start)
                break;
            block_sync_lds();
        }
    }

    private:
    /// @brief Computes the K offsets in the A and B tensors given iter_offset, where iter_offset is
    /// the starting macro tile index in the K dimension for the workgroup.
    /// @return A tuple containing the offsets into the A and B tensors accounting for the layouts
    /// of A and B.
    /// @note The default case is that A is assumed to be row major and B is assumed to be column
    /// major.
    template <typename ALayout, typename BLayout>
    CK_TILE_DEVICE static tuple<index_t, index_t>
    GetKOffsets(index_t iter_offset, index_t stride_a, index_t stride_b)
    {
        index_t stride_offset_a;
        index_t stride_offset_b;
        if constexpr(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::ColumnMajor>)
        {
            stride_offset_a = stride_a;
        }
        else
        {
            stride_offset_a = 1;
        }

        if constexpr(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            stride_offset_b = stride_b;
        }
        else
        {
            stride_offset_b = 1;
        }

        index_t base_offset = iter_offset * TilePartitioner::KPerBlock;

        return make_tuple(base_offset * stride_offset_a, base_offset * stride_offset_b);
    }

    CK_TILE_HOST static int NumCU()
    {
        hipDeviceProp_t dev_prop;
        hipDevice_t dev;
        hip_check_error(hipGetDevice(&dev));
        hip_check_error(hipGetDeviceProperties(&dev_prop, dev));
        int num_cu = dev_prop.multiProcessorCount;

        return num_cu;
    }

    /// @brief Computes the occupancy (i.e. maximum number of active blocks per CU) for the kernel
    /// @return The occupancy
    /// @note This function queries the maximum occupancy of the kernel using
    /// `hipOccupancyMaxActiveBlocksPerMultiprocessor`.
    CK_TILE_HOST static int Occupancy()
    {
        int occupancy;

        // Since occupancy of 1 is valid for stream k, we set min_num_block_per_cu to 1
        constexpr int min_block_per_cu = 1;
        const auto kernel              = kentry<min_block_per_cu, Kernel, KernelArgs>;

        hip_check_error(
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, kBlockSize, 0));

        return occupancy;
    }
};

} // namespace ck_tile
