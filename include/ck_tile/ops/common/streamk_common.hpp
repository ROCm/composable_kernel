// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/kernel/streamk_gemm/streamk_gemm_coherency.hpp"

namespace ck_tile {
enum StreamKReductionStrategy : uint32_t
{
    Atomic = 0u,
    Linear = 1u,
    Tree   = 2u
};

/// @brief StreamK reduction helpers: partial store/load, flag signaling, and tile accumulation.
///        Shared by StreamK GEMM and StreamK conv bwd weight kernels.
template <typename TilePartitioner_, typename GemmPipeline_, typename KernelArgs_>
struct StreamKReductionOps
{
    using TilePartitioner = remove_cvref_t<TilePartitioner_>;
    using BlockGemm       = typename GemmPipeline_::BlockGemm;
    using WarpGemm        = typename BlockGemm::WarpGemm;
    using BlockGemmShape  = typename GemmPipeline_::BlockGemmShape;

    /**
     *@brief Signals that the current thread block(CTA) has completed storing its partial
     * results.
     * @param kargs Kernel arguments, including the workspace pointer.
     * @param cta_idx The index of the current thread block (CTA).
     * @note This function utilizes a scalar store to write to the flags buffer.
     */
    CK_TILE_DEVICE void SignalStorePartialDone(const KernelArgs_& kargs, index_t cta_idx) const
    {
        auto* sk_flags_ptr = static_cast<index_t*>(kargs.workspace_ptr);
        index_t offset     = cta_idx * sizeof(index_t);

        // Depending on the architecture, the GLC flag will bypass the appropriate
        // cache level(s) to ensure the write is visible to other workgroups. See the
        // appropriate ISA for details about the GLC modifier.
        asm volatile("s_store_dword %0, %1, %2 glc\n\t"
                     "s_waitcnt lgkmcnt(0)" // Wait for the store to complete
                     :
                     : "s"(1), "s"(sk_flags_ptr), "s"(offset)
                     : "memory");
    }

    /**
     * @brief Waits for the thread block (cta_idx) to complete storing its partial results.
     * @param kargs Kernel arguments, including the workspace pointer.
     * @param cta_idx The index of the thread block (CTA).
     * @note This function utilizes a scalar load to read from the flags
     * buffer.
     */
    CK_TILE_DEVICE void WaitStorePartialDone(const KernelArgs_& kargs, index_t cta_idx) const
    {
        auto* sk_flags_ptr = static_cast<index_t*>(kargs.workspace_ptr);
        index_t result;
        index_t offset = cta_idx * sizeof(index_t);

        do
        {
            // Depending on the architecture, the GLC flag will bypass the
            // appropriate cache level(s) to avoid reading stale flags. See the
            // appropriate ISA for details about the GLC modifier.
            asm volatile("s_load_dword %0, %1, %2 glc\n\t"
                         "s_waitcnt lgkmcnt(0)" // Wait for the load to complete
                         : "=s"(result)
                         : "s"(sk_flags_ptr), "s"(offset)
                         : "memory");
        } while(result != 1);
    }

    /**
     * @brief Adds the values of a block tile to an output block tile.
     * @param in_out_block_tile The output block tile to which values are added.
     * @param in_block_tile The input block tile whose values are added.
     * @note This function iterates over the distributed spans of the block tiles and updates
     * the output block tile with accumulated values.
     */
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

    /**
     * @brief Loads a partial block tile from the workspace buffer.
     * @param kargs Kernel arguments, including the workspace pointer.
     * @param cta_idx The index of the thread block (CTA).
     * @param c_block_tile_dist The tile distribution for the block.
     * @return The loaded partial block tile.
     * @note This function calculates the buffer pointer and uses the tile distribution for
     * loading the partial block tile.
     */
    template <typename DataType, typename OAccTileDist>
    CK_TILE_DEVICE auto LoadPartial(const KernelArgs_& kargs,
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
            number<GetVectorSizePartials()>{},
            number<1>{});

        auto partial_tile_window = make_tile_window(
            partial_tensor_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {0, 0},
            MakePartialsDistribution());

        auto partials_tile = load_tile(partial_tile_window);

        // Since the partials distribution is not the same as the C block distribution, we must
        // describe the contents in the partials tile with the C block distribution.
        // Note: The data assigned to threads does not change between distributions.
        auto partials_tile_with_c_distr = make_static_distributed_tensor<DataType>(
            c_block_tile_dist, partials_tile.get_thread_buffer());

        return partials_tile_with_c_distr;
    }

    /**
     * @brief Returns the vector size to be used for reading from and writing to partials.
     * @return The vector size
     */
    CK_TILE_DEVICE static constexpr index_t GetVectorSizePartials()
    {
        // We use kCM1PerLane from the C register layout of the warp GEMM which corresponds to the
        // maximum vector width
        return WarpGemm::WarpGemmAttribute::Impl::kCM1PerLane;
    }

    /**
     * @brief Returns distribution used for reading from and writing to partials.
     * @return The distribution.
     * @note This will result in optimized reads from and writes to partials when C is row major.
     * Additional functionality should be added to ensure optimized accesses to partials when C is
     * column major. Since the C-Shuffle epilogue only supports C as row major, this is not a
     * current limitation.
     */
    CK_TILE_DEVICE static constexpr auto MakePartialsDistribution()
    {
        // Create the encoding to describe waves within a block
        constexpr index_t m_warp = BlockGemmShape::BlockWarps::at(number<0>{});
        constexpr index_t n_warp = BlockGemmShape::BlockWarps::at(number<1>{});

        constexpr index_t m_iter_per_warp = TilePartitioner::MPerBlock / (m_warp * WarpGemm::kM);
        constexpr index_t n_iter_per_warp = TilePartitioner::NPerBlock / (n_warp * WarpGemm::kN);

        constexpr auto partials_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<m_iter_per_warp, m_warp>, sequence<n_iter_per_warp, n_warp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        // Create the encoding to describe threads within a wave
        constexpr index_t vector_size         = GetVectorSizePartials();
        constexpr index_t m_warp_repeat       = WarpGemm::WarpGemmAttribute::Impl::kCM0PerLane;
        constexpr index_t warp_tile_n_threads = WarpGemm::kN / vector_size;
        constexpr index_t warp_tile_m_threads = get_warp_size() / warp_tile_n_threads;

        // This inner encoding ensures that contiguous threads perform vectorized writes along the
        // same row in C.
        constexpr auto partials_inner_dstr_encoding =
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<m_warp_repeat, warp_tile_m_threads>,
                                             sequence<warp_tile_n_threads, vector_size>>,
                                       tuple<sequence<1, 2>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{};

        // Combine the outer and inner encoding
        constexpr auto partials_dstr_encode = detail::make_embed_tile_distribution_encoding(
            partials_outer_dstr_encoding, partials_inner_dstr_encoding);

        return make_static_tile_distribution(partials_dstr_encode);
    }

    /**
     * @brief Stores a partial block tile to the workspace buffer.
     * @param kargs Kernel arguments, including the workspace pointer.
     * @param cta_idx The index of the thread block (CTA).
     * @param c_block_tile The block tile to be stored.
     * @note This function calculates the buffer pointer and uses the tile window for storing
     * the partial block tile.
     */
    template <typename OAccTile>
    CK_TILE_DEVICE void
    StorePartial(const KernelArgs_& kargs, index_t cta_idx, const OAccTile& c_block_tile) const
    {
        const auto c_block_tile_buffer_size = TilePartitioner::MPerBlock *
                                              TilePartitioner::NPerBlock *
                                              sizeof(typename OAccTile::DataType);
        void* partial_buffer_ptr = static_cast<char*>(kargs.workspace_ptr) +
                                   kargs.tile_partitioner.get_flags_buffer_size() +
                                   cta_idx * c_block_tile_buffer_size;

        const auto& partial_tensor_view = make_naive_tensor_view<
            address_space_enum::global,
            memory_operation_enum::set,
            StreamKCoherency<decltype(core::arch::get_compiler_target())>::BUFFER_COHERENCE>(
            static_cast<typename OAccTile::DataType*>(partial_buffer_ptr),
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            make_tuple(TilePartitioner::NPerBlock, 1),
            number<GetVectorSizePartials()>{},
            number<1>{});

        auto partial_tile_window = make_tile_window(
            partial_tensor_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {0, 0},
            MakePartialsDistribution());

        // Since the C block distribution is not the same as the partials distribution, we must
        // describe the contents in the c_block_tile with the partials distribution.
        // Note: The data assigned to threads does not change between distributions.
        auto c_with_partials_dist = make_static_distributed_tensor<typename OAccTile::DataType>(
            MakePartialsDistribution(), c_block_tile.get_thread_buffer());

        store_tile(partial_tile_window, c_with_partials_dist);
        // Wait for all vector stores for this wavefront to complete
        s_waitcnt</*vmcnt*/ 0, waitcnt_arg::kMaxExpCnt, waitcnt_arg::kMaxLgkmCnt>();
        // Wait for all wavefronts in this workgroup to arrive here before continuing
        __builtin_amdgcn_s_barrier();
    }
};

/// @brief StreamK data-parallel (DP) dispatch: handles persistent vs non-persistent DP,
///        then delegates to the Stream-K loop. Shared by GEMM and Conv StreamK kernels.
///
/// Non-persistent: launches dp_ctas + sk_ctas workgroups. DP workgroups each process
///   one full tile; SK workgroups share the remaining tiles' K-iterations.
/// Persistent: launches num_cu * occupancy workgroups. Each loops over DP tiles
///   (round-robin), then proceeds to SK work.
///
/// @tparam TilePartitioner_ Partitioner type (persistent or non-persistent specialization).
/// @param tile_partitioner The partitioner instance from kernel args.
/// @param dp_tile_func     Callable(index_t tile_idx) - processes one full DP tile.
/// @param sk_func          Callable(index_t sk_cta_idx) - runs the StreamK loop for this CTA.
template <typename TilePartitioner_, typename DPTileFunc, typename SKFunc>
CK_TILE_DEVICE void StreamKDispatch(const TilePartitioner_& tile_partitioner,
                                    DPTileFunc dp_tile_func,
                                    SKFunc sk_func,
                                    index_t block_idx)
{
    if constexpr(TilePartitioner_::PERSISTENT)
    {
        // Persistent: each workgroup loops over multiple DP tiles, then does SK work
        for(index_t tile_idx = block_idx; tile_idx < tile_partitioner.get_dp_tiles();
            tile_idx += tile_partitioner.get_max_active_wgs())
        {
            dp_tile_func(tile_idx);
            block_sync_lds();
        }
        sk_func(block_idx);
    }
    else
    {
        // Non-persistent: dedicated DP workgroups, then dedicated SK workgroups
        const index_t dp_ctas = tile_partitioner.get_dp_ctas();
        if(block_idx < dp_ctas)
            dp_tile_func(block_idx);
        else
            sk_func(block_idx - dp_ctas);
    }
}

} // namespace ck_tile
