// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/kernel/streamk_gemm/streamk_gemm_coherency.hpp"
#include "ck_tile/ops/gemm/kernel/wavelet_utils.hpp"

// Test-only bound on WaitStorePartialDone's spin. Linear and tree reduction deadlock if the
// workgroup they wait on is never resident; with a non-zero limit each spinning wave executes
// s_trap 2 after that many polls (wave halted, queue put in error state, process aborted), which
// tears the queue down instead of hanging the device. Waves parked at a workgroup or named
// barrier do not poll and are not covered. 0 (the default) spins forever.
#ifndef CK_TILE_STREAMK_WAIT_SPIN_LIMIT
#define CK_TILE_STREAMK_WAIT_SPIN_LIMIT 0
#endif

namespace ck_tile {

/// CK_TILE_STREAMK_WAIT_SPIN_LIMIT as an index_t. Brace initialisation rejects values polls cannot
/// reach (e.g. 4294967296, 1e6) at compile time. Set it target-wide: each TU bakes in its own
/// value.
inline constexpr index_t kStreamKWaitSpinLimit{CK_TILE_STREAMK_WAIT_SPIN_LIMIT};
static_assert(kStreamKWaitSpinLimit >= 0,
              "CK_TILE_STREAMK_WAIT_SPIN_LIMIT must be >= 0 (0 spins forever)");
enum StreamKReductionStrategy : uint32_t
{
    Atomic = 0u,
    Linear = 1u,
    Tree   = 2u
};

/// @brief Lower-case name of a Stream-K reduction strategy, as used in kernel names.
CK_TILE_HOST_DEVICE constexpr const char*
streamk_reduction_strategy_name(StreamKReductionStrategy strategy)
{
    switch(strategy)
    {
    case StreamKReductionStrategy::Atomic: return "atomic";
    case StreamKReductionStrategy::Linear: return "linear";
    case StreamKReductionStrategy::Tree: return "tree";
    }
    return "unknown";
}

/// @brief StreamK reduction helpers: partial store/load, flag signaling, and tile accumulation.
///        Shared by StreamK GEMM and StreamK conv bwd weight kernels.
template <typename TilePartitioner_, typename GemmPipeline_, typename KernelArgs_>
struct StreamKReductionOps
{
    using TilePartitioner = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline    = remove_cvref_t<GemmPipeline_>;
    using BlockGemm       = typename GemmPipeline_::BlockGemm;
    using WarpGemm        = typename BlockGemm::WarpGemm;
    using BlockGemmShape  = typename GemmPipeline_::BlockGemmShape;
    using CompilerTarget  = decltype(core::arch::get_compiler_target());

    // Helper: all supported architectures for specialized versions
    template <typename CompilerTarget_>
    static constexpr bool IsStreamKReductionSupportedArch()
    {
        return core::arch::is_target_id_any_of<CompilerTarget_,
                                               core::arch::amdgcn_target_id::GFX90A,
                                               core::arch::amdgcn_target_id::GFX942,
                                               core::arch::amdgcn_target_id::GFX950,
                                               core::arch::amdgcn_target_id::GFX1200,
                                               core::arch::amdgcn_target_id::GFX1201,
                                               core::arch::amdgcn_target_id::GFX1250,
                                               core::arch::amdgcn_target_id::GFX12_GENERIC>() ||
               core::arch::is_target_family_gfx11<CompilerTarget_>() ||
               core::arch::is_target_family_gfx117<CompilerTarget_>();
    }

    // Counts one unanswered poll of WaitStorePartialDone and traps (noreturn) once
    // kStreamKWaitSpinLimit polls have gone unanswered, so it only ever returns false; the bool
    // exists so it can sit in the loop condition after `result != 1` and count failed polls only.
    // Compiles to nothing at the default limit of 0.
    CK_TILE_DEVICE static bool SpinLimitReached([[maybe_unused]] index_t& polls)
    {
        if constexpr(kStreamKWaitSpinLimit > 0)
        {
            if(++polls >= kStreamKWaitSpinLimit)
            {
                __builtin_trap();
            }
        }
        return false;
    }

    /**
     *@brief Signals that the current thread block(CTA) has completed storing its partial
     * results.
     * @param kargs Kernel arguments, including the workspace pointer.
     * @param cta_idx The index of the current thread block (CTA).
     * @note This function utilizes a scalar store to write to the flags buffer.
     */
    template <typename CompilerTarget_ = CompilerTarget>
    CK_TILE_DEVICE core::arch::enable_if_target_id_t<CompilerTarget_,
                                                     core::arch::amdgcn_target_id::GFX90A,
                                                     core::arch::amdgcn_target_id::GFX942,
                                                     core::arch::amdgcn_target_id::GFX950>
    SignalStorePartialDone(const KernelArgs_& kargs, index_t cta_idx) const
    {
        // s_store_dword needs a wave-uniform (SGPR) address; kargs-by-ref can leave the pointer
        // in a VGPR (instantiation-dependent), which the assembler rejects. Pull the (uniform)
        // operands into SGPRs first.
        auto* sk_flags_ptr = reinterpret_cast<index_t*>(
            amd_wave_read_first_lane(reinterpret_cast<uintptr_t>(kargs.workspace_ptr)));
        index_t offset = amd_wave_read_first_lane(cta_idx) * sizeof(index_t);

        // Depending on the architecture, the GLC flag will bypass the appropriate
        // cache level(s) to ensure the write is visible to other workgroups. See the
        // appropriate ISA for details about the GLC modifier.
        asm volatile("s_store_dword %0, %1, %2 glc\n\t"
                     "s_waitcnt lgkmcnt(0)" // Wait for the store to complete
                     :
                     : "s"(1), "s"(sk_flags_ptr), "s"(offset)
                     : "memory");
    }

    template <typename CompilerTarget_ = CompilerTarget>
    CK_TILE_DEVICE std::enable_if_t<
        core::arch::is_target_id_any_of<CompilerTarget_,
                                        core::arch::amdgcn_target_id::GFX1200,
                                        core::arch::amdgcn_target_id::GFX1201,
                                        core::arch::amdgcn_target_id::GFX1250,
                                        core::arch::amdgcn_target_id::GFX12_GENERIC>() ||
        core::arch::is_target_family_gfx11<CompilerTarget_>() ||
        core::arch::is_target_family_gfx117<CompilerTarget_>()>
    SignalStorePartialDone(const KernelArgs_& kargs, index_t cta_idx) const
    {
        auto* sk_flags_ptr = static_cast<index_t*>(kargs.workspace_ptr);
        index_t offset     = cta_idx * sizeof(index_t);
        // The flag store uses raw builtins in both buffer-addressing modes.
        __amdgpu_buffer_rsrc_t buffer_rsc = __builtin_amdgcn_make_buffer_rsrc(
            sk_flags_ptr,
            0,
            sizeof(index_t) * kargs.tile_partitioner.get_sk_ctas(),
            CK_TILE_BUFFER_RESOURCE_3RD_DWORD);

        if(threadIdx.x == 0)
        {
            __builtin_amdgcn_raw_buffer_store_b32(
                1,
                buffer_rsc,
                offset,
                0,
                static_cast<int>(StreamKCoherency<CompilerTarget_>::BUFFER_COHERENCE));
        }
    }

    template <typename CompilerTarget_ = CompilerTarget>
    CK_TILE_DEVICE std::enable_if_t<!IsStreamKReductionSupportedArch<CompilerTarget_>()>
    SignalStorePartialDone([[maybe_unused]] const KernelArgs_& kargs,
                           [[maybe_unused]] index_t cta_idx) const
    {
        static_assert(IsStreamKReductionSupportedArch<CompilerTarget_>(),
                      "SignalStorePartialDone not implemented for this architecture.");
    }

    /**
     * @brief Waits for the thread block (cta_idx) to complete storing its partial results.
     * @param kargs Kernel arguments, including the workspace pointer.
     * @param cta_idx The index of the thread block (CTA).
     * @note This function utilizes a scalar load to read from the flags
     * buffer.
     */
    template <typename CompilerTarget_ = CompilerTarget>
    CK_TILE_DEVICE core::arch::enable_if_target_id_t<CompilerTarget_,
                                                     core::arch::amdgcn_target_id::GFX90A,
                                                     core::arch::amdgcn_target_id::GFX942,
                                                     core::arch::amdgcn_target_id::GFX950>
    WaitStorePartialDone(const KernelArgs_& kargs, index_t cta_idx) const
    {
        // s_load_dword needs a wave-uniform (SGPR) address (see SignalStorePartialDone).
        auto* sk_flags_ptr = reinterpret_cast<index_t*>(
            amd_wave_read_first_lane(reinterpret_cast<uintptr_t>(kargs.workspace_ptr)));
        index_t result;
        index_t offset = amd_wave_read_first_lane(cta_idx) * sizeof(index_t);
        index_t polls  = 0;

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
        } while(result != 1 && !SpinLimitReached(polls));
    }

    template <typename CompilerTarget_ = CompilerTarget>
    CK_TILE_DEVICE core::arch::enable_if_target_id_t<CompilerTarget_,
                                                     core::arch::amdgcn_target_id::GFX1200,
                                                     core::arch::amdgcn_target_id::GFX1201,
                                                     core::arch::amdgcn_target_id::GFX1250,
                                                     core::arch::amdgcn_target_id::GFX12_GENERIC>
    WaitStorePartialDone(const KernelArgs_& kargs, index_t cta_idx) const
    {
        // The scalar load requires uniform SGPR operands even when kargs is passed by reference.
        auto* sk_flags_ptr = reinterpret_cast<index_t*>(
            amd_wave_read_first_lane(reinterpret_cast<uintptr_t>(kargs.workspace_ptr)));
        index_t result;
        index_t offset = amd_wave_read_first_lane(cta_idx) * sizeof(index_t);
        index_t polls  = 0;
        do
        {
            asm volatile("s_load_b32 %0, %1, %2 scope:SCOPE_DEV\n\t"
                         "s_wait_kmcnt 0" // Wait for the load to complete
                         : "=s"(result)
                         : "s"(sk_flags_ptr), "s"(offset)
                         : "memory");
        } while(result != 1 && !SpinLimitReached(polls));
    }

    template <typename CompilerTarget_ = CompilerTarget>
    CK_TILE_DEVICE std::enable_if_t<core::arch::is_target_family_gfx11<CompilerTarget_>() ||
                                    core::arch::is_target_family_gfx117<CompilerTarget_>()>
    WaitStorePartialDone(const KernelArgs_& kargs, index_t cta_idx) const
    {
        auto* sk_flags_ptr = static_cast<index_t*>(kargs.workspace_ptr);
        index_t result;
        index_t offset = cta_idx * sizeof(index_t);
        index_t polls  = 0;
        do
        {
            asm volatile("s_load_b32 %0, %1, %2 glc dlc\n\t"
                         "s_waitcnt lgkmcnt(0)" // Wait for the load to complete
                         : "=s"(result)
                         : "s"(sk_flags_ptr), "s"(offset)
                         : "memory");
        } while(result != 1 && !SpinLimitReached(polls));
    }

    template <typename CompilerTarget_ = CompilerTarget>
    CK_TILE_DEVICE std::enable_if_t<!IsStreamKReductionSupportedArch<CompilerTarget_>()>
    WaitStorePartialDone([[maybe_unused]] const KernelArgs_& kargs,
                         [[maybe_unused]] index_t cta_idx) const
    {
        static_assert(IsStreamKReductionSupportedArch<CompilerTarget_>(),
                      "WaitStorePartialDone not implemented for this architecture.");
    }

    /**
     * @brief Adds the partial tile stored by @p cta_idx into @p accum_block_tile.
     * @note Only the math waves of a wavelet pipeline hold an accumulator, so only they load; the
     * load waves keep their (unused) tile. Issues no barrier, so the branch is safe.
     * @note The partial is loaded and added in row slices of at most MAX_PARTIAL_SLICE_ELEMS
     * values per lane. Loading it whole would hold a second accumulator-sized tile in VGPRs and,
     * for large tiles, raise the kernel's VGPR count (and cut its occupancy) on every launch.
     */
    template <typename OAccTile>
    CK_TILE_DEVICE void
    AccumulatePartial(const KernelArgs_& kargs, index_t cta_idx, OAccTile& accum_block_tile) const
    {
        if(IsMathWave<GemmPipeline>())
        {
            using DataType                     = typename OAccTile::DataType;
            constexpr index_t m_iter_per_slice = GetPartialMIterPerSlice<OAccTile>();
            constexpr index_t num_slices       = PARTIAL_M_ITER_PER_WARP / m_iter_per_slice;
            constexpr index_t slice_rows       = TilePartitioner::MPerBlock / num_slices;
            constexpr index_t slice_lane_elems = OAccTile::get_thread_buffer_size() / num_slices;
            static_assert(OAccTile::get_thread_buffer_size() % num_slices == 0,
                          "the accumulator must split evenly into the partial's M slices");

            // Partial results cross workgroup boundaries; see StreamKCoherency::LOAD_COHERENCE.
            auto partial_tile_window = make_tile_window(
                MakePartialTensorView<DataType, StreamKCoherency<CompilerTarget>::LOAD_COHERENCE>(
                    kargs, cta_idx),
                make_tuple(number<slice_rows>{}, number<TilePartitioner::NPerBlock>{}),
                {0, 0},
                MakePartialsDistribution<m_iter_per_slice>());

            // The M iteration is the outermost Y dimension of both the partials and the C block
            // distribution, so a slice of rows is a contiguous run of each lane's accumulator.
            // Add in whole warp-GEMM C fragments (the vector each WMMA/MFMA writes): scalar adds
            // let the SLP vectorizer pair them into <2 x float> values that cut across those
            // fragments, and the register allocator then copies the accumulator into the
            // pipeline's fragment registers instead of reusing them (~40 VGPRs at 256x128,
            // ~1500 spills at 256x256 on gfx1250).
            constexpr index_t frag_elems = WarpGemm::CWarpTensor::get_thread_buffer_size();
            static_assert(slice_lane_elems % frag_elems == 0,
                          "a partial slice must hold whole warp-GEMM C fragments");
            using FragVec                 = ext_vector_t<DataType, frag_elems>;
            constexpr index_t slice_frags = slice_lane_elems / frag_elems;

            auto& accum = accum_block_tile.get_thread_buffer();
            static_for<0, num_slices, 1>{}([&](auto slice) {
                if constexpr(slice > 0)
                {
                    // Keep this slice's loads behind the previous slice's adds, so only one
                    // slice is live at a time. Scheduling only: ordering against the flag poll
                    // in WaitStorePartialDone comes from that asm's "memory" clobber.
                    __builtin_amdgcn_sched_barrier(0);
                    move_tile_window(partial_tile_window, {slice_rows, 0});
                }
                const auto partial = load_tile(partial_tile_window);
                // Each lane's slice must be exactly the next slice_lane_elems accumulator values;
                // a C distribution whose M iteration is not the outermost Y would break this.
                static_assert(remove_cvref_t<decltype(partial)>::get_thread_buffer_size() ==
                                  slice_lane_elems,
                              "partial slice and accumulator slice hold different value counts");
                static_for<0, slice_frags, 1>{}([&](auto f) {
                    constexpr auto dst = number<slice * slice_frags + f>{};
                    accum.template set_as<FragVec>(
                        dst,
                        accum.template get_as<FragVec>(dst) +
                            partial.get_thread_buffer().template get_as<FragVec>(f));
                });
            });
        }
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
        // MakePartialsDistribution maps the BlockWarps (math) waves only. The load waves of a
        // wavelet pipeline hold no partial tile, so they skip the store but still join the
        // barrier below.
        if(IsMathWave<GemmPipeline>())
        {
            StoreAndFencePartial(kargs, cta_idx, c_block_tile);
        }
        // Wait for all wavefronts in this workgroup to arrive here before continuing
        __builtin_amdgcn_s_barrier();
    }

    private:
    // Upper bound on the partial values per lane that AccumulatePartial holds at once.
    static constexpr index_t MAX_PARTIAL_SLICE_ELEMS = 32;

    // M iterations of each math wave over the tile (the outer M dimension of the partials).
    static constexpr index_t PARTIAL_M_ITER_PER_WARP =
        TilePartitioner::MPerBlock / (BlockGemmShape::BlockWarps::at(number<0>{}) * WarpGemm::kM);

    // Largest divisor of the M iterations whose slice holds at most MAX_PARTIAL_SLICE_ELEMS
    // values per lane; 1 when even a single M iteration holds more.
    template <typename OAccTile>
    CK_TILE_DEVICE static constexpr index_t GetPartialMIterPerSlice()
    {
        constexpr index_t m_iter         = PARTIAL_M_ITER_PER_WARP;
        constexpr index_t elems_per_iter = OAccTile::get_thread_buffer_size() / m_iter;
        index_t per_slice                = 1;
        for(index_t i = 1; i <= m_iter; ++i)
        {
            if(m_iter % i == 0 && i * elems_per_iter <= MAX_PARTIAL_SLICE_ELEMS)
            {
                per_slice = i;
            }
        }
        return per_slice;
    }

    // The partial tile cta_idx stores, as a row-major MPerBlock x NPerBlock global view.
    template <typename DataType, amd_buffer_coherence_enum Coherence>
    CK_TILE_DEVICE static auto MakePartialTensorView(const KernelArgs_& kargs, index_t cta_idx)
    {
        constexpr auto c_block_tile_buffer_size =
            TilePartitioner::MPerBlock * TilePartitioner::NPerBlock * sizeof(DataType);
        void* partial_buffer_ptr = static_cast<char*>(kargs.workspace_ptr) +
                                   kargs.tile_partitioner.get_flags_buffer_size() +
                                   cta_idx * c_block_tile_buffer_size;

        return make_naive_tensor_view<address_space_enum::global,
                                      memory_operation_enum::set,
                                      Coherence>(
            static_cast<DataType*>(partial_buffer_ptr),
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            make_tuple(TilePartitioner::NPerBlock, 1),
            number<GetVectorSizePartials()>{},
            number<1>{});
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
     * @tparam MIterPerWarp M iterations covered: all of them for the whole tile, fewer for a
     * slice of MIterPerWarp * m_warp * WarpGemm::kM rows.
     */
    template <index_t MIterPerWarp = PARTIAL_M_ITER_PER_WARP>
    CK_TILE_DEVICE static constexpr auto MakePartialsDistribution()
    {
        // Create the encoding to describe waves within a block
        constexpr index_t m_warp = BlockGemmShape::BlockWarps::at(number<0>{});
        constexpr index_t n_warp = BlockGemmShape::BlockWarps::at(number<1>{});

        constexpr index_t m_iter_per_warp = MIterPerWarp;
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

    // Stores this wave's share of the partial tile and waits for the stores to complete.
    template <typename OAccTile>
    CK_TILE_DEVICE void StoreAndFencePartial(const KernelArgs_& kargs,
                                             index_t cta_idx,
                                             const OAccTile& c_block_tile) const
    {
        auto partial_tile_window = make_tile_window(
            MakePartialTensorView<typename OAccTile::DataType,
                                  StreamKCoherency<CompilerTarget>::BUFFER_COHERENCE>(kargs,
                                                                                      cta_idx),
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
        if constexpr(core::arch::is_target_family_any_of<
                         CompilerTarget,
                         core::arch::amdgcn_target_family_id::GFX12,
                         core::arch::amdgcn_target_family_id::GFX1250>())
        {
            // gfx12 has separate load and store counters. Wait for the partial stores
            // before publishing the completion flag in either buffer-addressing mode.
            buffer_store_fence();
        }
        else
        {
            s_waitcnt</*vmcnt*/ 0, waitcnt_arg::kMaxExpCnt, waitcnt_arg::kMaxLgkmCnt>();
        }
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
