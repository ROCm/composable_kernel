// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>

#include "ck_tile/core/arch/named_barrier.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include "ck_tile/ops/gemm/kernel/wavelet_utils.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/common/utils.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"
#include "ck_tile/ops/epilogue/tdm_epilogue.hpp"
#include "streamk_gemm_coherency.hpp"

namespace ck_tile {

/**
 * @brief The Stream K GEMM kernel host arguments.
 *
 * @par Overview
 *      This structure is passed to @ref StreamKKernel "StreamKKernel" when creating the kernel
 *      arguments object. It contains all necessary information required to build proper kernel
 *      arguments and launch the kernel on GPU. This structure defines the GEMM problem
 *      configuration by stating all required information like M,N,K sizes and respective strides.
 */
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
                                          index_t stride_C_)
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
                                  stride_C_)
    {
    }
};

/**
 * @brief The Stream K GEMM kernel class.
 *
 * @par Overview
 *      This class is responsible for the Stream-K kernel, making use of UniversalGemm.
 *	 The main kernel functions are the operator() functions. There is one for Persistent
 *	 and one for Non-Persistent data parallel sections of the Stream-K algorithm.
 *
 *	 Both the Non-Persistent and Persistent kernels make use of `BaseGemm()` and
 *	 `StreamKGemm()`. `BaseGemm()` computes offsets into the A,B,C tensors, then calls
 *	 `RunGemm()` which runs the GEMM pipeline and epilogue. `StreamKGemm()` performs the
 *	 main Stream-K algorithm. Each iteration of the Stream-K loop calls `BaseGemm()`.
 */
template <typename TilePartitioner_, typename GemmPipeline_, typename EpiloguePipeline_>
struct StreamKKernel
{
    /**
     *@brief Inject the UniversalGemmKernel base class to support execution of all necessary
     *functions.
     */
    using UniversalGemmKernel =
        UniversalGemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>;

    static constexpr index_t kBlockSize = UniversalGemmKernel::kBlockSize;
    static constexpr bool PersistentDP  = UniversalGemmKernel::PersistentKernel;

    using TilePartitioner  = TilePartitioner_;
    using GemmPipeline     = GemmPipeline_;
    using EpiloguePipeline = EpiloguePipeline_;
    using WarpGemm         = typename GemmPipeline::BlockGemm::WarpGemm;
    using BlockGemmShape   = typename GemmPipeline::BlockGemmShape;

    static_assert(
        TilePartitioner::PERSISTENT == PersistentDP,
        "Persistent flag from TilePartitioner must match Persistent flag from UniversalGemm.");

    /// The named barriers this kernel arms for GemmPipeline; void when it uses none. Declared here
    /// because the launched kernel, not the wrapped UniversalGemmKernel, owns the barrier array.
    using barrier_pipeline = typename UniversalGemmKernel::barrier_pipeline;
    /// Stands in for the barrier token when the pipeline uses no named barriers.
    using NoBarriers = typename UniversalGemmKernel::NoBarriers;
    /// Token proving the named barriers are armed; NoBarriers without them.
    using BarrierToken = decltype(UniversalGemmKernel::InitBarriers());

    // Atomic reduction adds every workgroup's partial tile into C. An epilogue whose store always
    // overwrites (TdmEpilogue) would keep only the last partial, and only traps at run time.
    static_assert(TilePartitioner::ReductionStrategy != StreamKReductionStrategy::Atomic ||
                      epilogue_can_accumulate_v<EpiloguePipeline>,
                  "StreamKReductionStrategy::Atomic adds every partial tile into C, so "
                  "EpiloguePipeline must accumulate (e.g. CShuffleEpilogue); TdmEpilogue "
                  "overwrites C");

    /**
     * @brief Workgroups of this kernel that fit on one WGP by named barriers alone; unbounded
     * without them.
     * @note hipOccupancy* does not model the named barrier pool; Occupancy() applies this cap.
     */
    static constexpr index_t kMaxResidentPerWgpByBarriers = []() {
        if constexpr(std::is_void_v<barrier_pipeline>)
            return std::numeric_limits<index_t>::max();
        else
            return barrier_pipeline::kMaxResidentPerWgp;
    }();
    static_assert(kMaxResidentPerWgpByBarriers >= 1,
                  "the pipeline's named barrier groups exceed one WGP's pool; Occupancy() would "
                  "return 0");

    /**
     * @brief  Specify the layout configurations for A, B, and C
     */
    using ALayout = typename GemmPipeline::ALayout;
    using BLayout = typename GemmPipeline::BLayout;
    using CLayout = typename GemmPipeline::CLayout;

    /**
     * @brief  Specify the data type configurations for A, B, and C
     */
    using ADataType   = typename GemmPipeline::ADataType;
    using BDataType   = typename GemmPipeline::BDataType;
    using CDataType   = typename EpiloguePipeline::ODataType;
    using AccDataType = typename EpiloguePipeline::AccDataType;

    template <typename T>
    static constexpr bool is_tuple_v = is_detected<is_tuple, T>::value;
    /**
     *@brief ALayout and ADataType are expected to be scalars, not a tuple.
     */
    static_assert(!is_tuple_v<ALayout> && !is_tuple_v<ADataType>,
                  "ALayout and ADataType must be scalars.");

    /**
     *@brief BLayout and BDataType are expected to be scalars, not a tuple.
     */
    static_assert(!is_tuple_v<BLayout> && !is_tuple_v<BDataType>,
                  "BLayout and BDataType must be scalars.");

    /**
     *@brief CLayout and CDataType are expected to be scalars, not a tuple.
     */
    static_assert(!is_tuple_v<CLayout> && !is_tuple_v<CDataType>,
                  "CLayout and CDataType must be scalars.");

    /**
     *@brief Stream-K gives every workgroup its own K range, so the TDM pipelines run without
     *cluster multicast.
     */
    static constexpr bool kClusterLaunch =
        BlockGemmShape::kclusterM * BlockGemmShape::kclusterN * BlockGemmShape::kclusterK > 1;
    static_assert(!UniversalGemmKernel::kIsTdmPipeline || !kClusterLaunch,
                  "TDM Stream-K GEMM: cluster launch unsupported (per-workgroup K ranges)");

    /**
     *@brief True for comp_tdm, comp_tdm_v2, and comp_tdm_v3: the TDM pipelines other than the
     *wavelet comp_tdm_pc. V1 and V2 can return with their last, clipped TDM prefetch still writing
     *LDS (V3 drains before returning). CShuffleEpilogue and TdmEpilogue wait for it; StreamKGemm
     *waits for it after a partial tile, which runs no epilogue. C must be row-major:
     *CShuffleEpilogue supports nothing else, and TdmEpilogue's column-major store is not validated
     *with these pipelines.
     */
    static constexpr bool kIsCompTdmPipeline =
        UniversalGemmKernel::kIsTdmPipeline && !is_wavelet_pipeline<GemmPipeline>;
    static_assert(!kIsCompTdmPipeline || is_tdm_epilogue_v<EpiloguePipeline> ||
                      is_specialization_of<EpiloguePipeline, CShuffleEpilogue>::value,
                  "Stream-K with comp_tdm, comp_tdm_v2, or comp_tdm_v3 requires CShuffleEpilogue "
                  "or TdmEpilogue (they wait for the pipeline's last TDM prefetch).");
    static_assert(!kIsCompTdmPipeline || std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>,
                  "Stream-K with comp_tdm, comp_tdm_v2, or comp_tdm_v3 supports row-major C only.");

    struct StreamKKernelArgs : ck_tile::UniversalGemmKernelArgs<>
    {
        /**
         * @param num_cu CUs the max_active_wgs workgroups run on; 0 treats every workgroup slot
         * as a CU (see StreamKTilePartitionerBase).
         */
        StreamKKernelArgs(const StreamKHostArgs& host_args,
                          index_t max_active_wgs,
                          int num_xccs_  = 1,
                          index_t num_cu = 0)
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
              // The workspace pointer is set to nullptr because we must first
              // instantiate the TilePartitioner to get the necessary size
              workspace_ptr{nullptr},
              tile_partitioner{TilePartitioner{host_args.M,
                                               host_args.N,
                                               host_args.K,
                                               max_active_wgs,
                                               num_cu > 0 ? num_cu : max_active_wgs}},
              num_xccs{num_xccs_}

        {
        }
        /**
         * @brief  A pointer to a buffer in device memory for accumulating partial via reduction
         * strategy.
         */
        void* workspace_ptr;
        /**
         * @brief  An instance of the TilePartioner class for assisting with mapping workgroups to
         * the C tensor.
         */
        TilePartitioner tile_partitioner;
        /**
         * @brief  An int for the number of xcds available on a given device for remapping the block
         * indices to be contiguous.
         */
        int num_xccs;
    };

    using KernelArgs = StreamKKernelArgs;
    using Kernel     = StreamKKernel<TilePartitioner, GemmPipeline, EpiloguePipeline>;
    using StreamKOps = StreamKReductionOps<TilePartitioner, GemmPipeline, StreamKKernelArgs>;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        using P_ = GemmPipeline;
        using WarpTile = typename BlockGemmShape::WarpTile;

        return concat('_', "streamk", gemm_prec_str<ADataType, BDataType>(),
                      concat('x', P_::MPerBlock, P_::NPerBlock, P_::KPerBlock),
                      concat('x', WarpTile::at(number<0>{}), WarpTile::at(number<1>{}), WarpTile::at(number<2>{})),
                      concat('x', P_::GetVectorSizeA(), P_::GetVectorSizeB(), P_::GetVectorSizeC()),
                      concat('x', P_::kPadM, P_::kPadN, P_::kPadK),
                      streamk_reduction_strategy_name(TilePartitioner::ReductionStrategy),
                      PersistentDP ? "persistent" : "nonpersistent",
                      P_::GetName());
        // clang-format on
    }

    /**
     * @brief Compute the grid size for the Stream K kernel using the tile_partitioner.
     * @return The grid size.
     */
    CK_TILE_HOST static auto GridSize(const TilePartitioner& tile_partitioner) -> dim3
    {
        return tile_partitioner.grid_size();
    }

    /**
     * @brief Get the maximum occupancy grid size for the persistent kernel on the current device.
     * @return The maximum occupancy grid size.
     * @note This function queries the maximum occupancy of the kernel using
     * `hipOccupancyMaxActiveBlocksPerMultiprocessor`.
     */
    CK_TILE_HOST static auto MaxOccupancyGridSize(const stream_config& s) -> dim3
    {
        // Occupancy() measures this kernel; UniversalGemmKernel's would measure its own entry.
        return dim3(get_available_compute_units(s) * Occupancy(), 1, 1);
    }

    CK_TILE_HOST static constexpr auto BlockSize() -> dim3
    {
        return UniversalGemmKernel::BlockSize();
    }

    /**
     * @brief Constructs kernel arguments for the Stream-K kernel.
     * @param host_args Stream-K host arguments.
     * @param num_cu Number of compute units (CUs). The default is the number of CUs on the device.
     * The caller may select their own to assist with test reproducibility, etc.
     * @param occupancy The maximum number of active blocks per CU for this kernel. The caller may
     * select their own to assist with test reproducibility, etc.
     * @return The kernel arguments for Stream-K.
     */
    CK_TILE_HOST static StreamKKernelArgs MakeKernelArgs(const StreamKHostArgs& host_args,
                                                         int num_cu    = NumCU(),
                                                         int occupancy = Occupancy())
    {
        const index_t max_active_wgs = num_cu * occupancy;
        const int num_xccs           = get_num_xccs();
        return StreamKKernelArgs{host_args, max_active_wgs, num_xccs, num_cu};
    }

    /**
     * @brief Runs the GEMM pipeline and the epilogue for one C tile.
     * @tparam UseDefaultScheduler If false, only warp 0 runs the epilogue; wavelet pipelines
     * require true.
     * @tparam CMemOp How the epilogue writes C: `set` for a tile this workgroup computes alone.
     * @tparam Token Must be BarrierToken; deduced from @p barriers.
     * @param barriers The token from InitBarriers(); may be omitted without named barriers.
     */
    template <bool UseDefaultScheduler     = true,
              memory_operation_enum CMemOp = TilePartitioner::MemoryOperation,
              typename Token               = NoBarriers>
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
            const index_t k_size,
            const Token barriers = {})
    {
        static_assert(std::is_same_v<Token, BarrierToken>,
                      "pass RunGemm the token from InitBarriers(); only pipelines without named "
                      "barriers may omit it");
        // Only the math waves of a wavelet pipeline may run the epilogue, and the load waves must
        // mirror its barriers; restricting it to warp 0 would deadlock them.
        static_assert(UseDefaultScheduler || !is_wavelet_pipeline<GemmPipeline>,
                      "RunGemm<UseDefaultScheduler = false> runs the epilogue on warp 0 only; a "
                      "wavelet GemmPipeline needs it on every math wave and the barrier stub on "
                      "every load wave");

        // Create block windows using specialized methods
        const auto& as_block_window =
            UniversalGemmKernel::MakeABlockWindows(as_ptr, kargs, k_size, block_idx_m);
        const auto& bs_block_window =
            UniversalGemmKernel::MakeBBlockWindows(bs_ptr, kargs, k_size, block_idx_n);
        const auto& ds_block_window =
            UniversalGemmKernel::MakeDBlockWindows(ds_ptr, kargs, block_idx_m, block_idx_n);

        // Run GEMM cooperatively by whole workgroup.
        const auto& c_block_tile =
            RunTilePipeline(as_block_window, bs_block_window, num_loop, smem_ptr_0, barriers);

        if(UseDefaultScheduler || (get_warp_id() == 0))
        {
            RunTileEpilogue<CMemOp>(
                c_ptr, kargs, block_idx_m, block_idx_n, c_block_tile, ds_block_window, smem_ptr_0);
        }
    }

    CK_TILE_HOST static bool IsSupportedArgument(const StreamKKernelArgs& kargs)
    {
        if constexpr(UniversalGemmKernel::kIsTdmPipeline)
        {
            // Revalidate the actual k_batch (UniversalGemmKernel substitutes 1 for a
            // split-K-capable TDM pipeline with a non-TDM epilogue, but Stream-K splits K itself)
            // and the device (UniversalGemmKernel checks it only for named-barrier pipelines; off
            // gfx125 the TDM instructions compile to no-ops). UniversalGemmKernel checks the rest,
            // including the C tail with a non-TDM epilogue.
            if(!UniversalGemmKernel::IsTdmArgumentSupported("Stream-K GEMM",
                                                            kargs.k_batch,
                                                            kargs.K,
                                                            kargs.as_ptr[0],
                                                            kargs.bs_ptr[0],
                                                            kargs.e_ptr) ||
               !UniversalGemmKernel::IsTdmDeviceSupported("Stream-K GEMM"))
            {
                return false;
            }
        }
        // Linear and tree reduction spin until another Stream-K workgroup has stored its
        // partial tile. A waiter only waits on higher sk_cta indices, and every workgroup it
        // waits on stores before waiting itself, so with in-order dispatch the waits resolve
        // with few workgroups resident (linear: two; tree: about log2 of the workgroups per
        // tile). sk_ctas <= device capacity is a conservative bound that does not rely on
        // dispatch order. It is device-wide, not per XCC (remap_xcd places non-persistent
        // Stream-K workgroups on the last XCCs), so it does not by itself make them all
        // co-resident. hipOccupancy* ignores the named-barrier pool, so the caller's
        // max_active_wgs is checked against the capped Occupancy().
        if constexpr(!std::is_void_v<barrier_pipeline> &&
                     TilePartitioner::ReductionStrategy != StreamKReductionStrategy::Atomic)
        {
            if(kargs.tile_partitioner.get_sk_ctas() > NumCU() * Occupancy())
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("Stream-K linear/tree reduction: sk_ctas exceeds NumCU() * "
                                  "Occupancy(), the conservative residency bound for workgroups "
                                  "that wait on each other's partial tiles!");
                }
                return false;
            }
        }
        return UniversalGemmKernel::IsSupportedArgument(kargs);
    }

    /**
     * @brief Computes the buffer size needed to store accumulation results for Stream K.
     * @return The buffer size needed.
     */
    CK_TILE_HOST static uint32_t GetWorkSpaceSize(const StreamKKernelArgs& kargs)
    {
        return kargs.tile_partitioner.get_workspace_size(sizeof(AccDataType));
    }
    /**
     *@brief Sets the kargs' current workspace_ptr to the given workspace_ptr.
     * @note Assumes that the given workspace_ptr points to allocated device memory.
     */
    CK_TILE_HOST static void SetWorkSpacePointer(StreamKKernelArgs& kargs, void* workspace_ptr)
    {
        kargs.workspace_ptr = workspace_ptr;
    }

    /**
     * @brief Computes offsets into A, B, and C tensors then runs the GEMM pipeline and epilogue.
     * @param kargs Stream-K kernel arguments.
     * @param tile_idx The 1D tile index in the C tensor for this workgroup.
     * @param num_loop The number of iterations (at the macro tile level) in the K dimension this
     * workgroup will perform in the C tile.
     * @param i_k_a The K offset in the A tensor.
     * @param i_k_b The K offset in the B tensor.
     * @param k_size The portion of the K dimension this workgroup processes in the assigned
     * `tile_idx`.
     * @param smem_ptr_0 Pointer to LDS.
     * @param barriers The token from InitBarriers().
     * @tparam CMemOp How the epilogue writes C (see RunGemm).
     */
    template <memory_operation_enum CMemOp>
    CK_TILE_DEVICE void BaseGemm(StreamKKernelArgs& kargs,
                                 index_t tile_idx,
                                 index_t num_loop,
                                 index_t i_k_a,
                                 index_t i_k_b,
                                 index_t k_size,
                                 void* smem_ptr_0,
                                 const BarrierToken barriers) const
    {
        const auto c_macro_tile_idx = kargs.tile_partitioner.get_output_tile_index(tile_idx);
        index_t i_m = c_macro_tile_idx[UniversalGemmKernel::I0] * TilePartitioner::MPerBlock;
        index_t i_n = c_macro_tile_idx[UniversalGemmKernel::I1] * TilePartitioner::NPerBlock;

        const ADataType* a_ptr = static_cast<const ADataType*>(kargs.as_ptr[0]) + i_k_a;
        const BDataType* b_ptr = static_cast<const BDataType*>(kargs.bs_ptr[0]) + i_k_b;
        CDataType* c_ptr       = static_cast<CDataType*>(kargs.e_ptr);

        // Run the GEMM pipeline and Epilogue.
        RunGemm<true, CMemOp>({a_ptr},
                              {b_ptr},
                              {/*ds_ptr*/},
                              c_ptr,
                              smem_ptr_0,
                              kargs,
                              num_loop,
                              i_m,
                              i_n,
                              k_size,
                              barriers);
    }

    /**
     * @brief Runs the main Stream-K algorithm.
     * @param kargs Stream-K kernel arguments.
     * @param cta_idx The current Stream-K workgroup's index.
     * @param smem_ptr_0 Pointer to LDS.
     * @param barriers The token from InitBarriers().
     * @note It is assumed that the first Stream-K workgroup has a `cta_idx` of zero. If a
     * non-persistent data-parallel (DP) section is used, then a Stream-K workgroup's `cta_idx`
     * *should be something like `blockIdx.x` minus number of DP workgroups.
     */
    CK_TILE_DEVICE void StreamKGemm(StreamKKernelArgs& kargs,
                                    index_t cta_idx,
                                    void* smem_ptr_0,
                                    const BarrierToken barriers) const
    {
        const StreamKOps sk_ops{};
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
            index_t k_size = amd_wave_read_first_lane(
                kargs.tile_partitioner.get_k_size(num_loop_sk, local_iter_end));

            // Get the K offsets for the A and B tensors
            auto [i_k_a, i_k_b] = GetKOffsets<ALayout, BLayout>(
                local_iter_start, kargs.stride_As[0], kargs.stride_Bs[0]);

            const auto c_macro_tile_idx = kargs.tile_partitioner.get_output_tile_index(tile_idx);
            index_t i_m = c_macro_tile_idx[UniversalGemmKernel::I0] * TilePartitioner::MPerBlock;
            index_t i_n = c_macro_tile_idx[UniversalGemmKernel::I1] * TilePartitioner::NPerBlock;

            const ADataType* a_ptr = static_cast<const ADataType*>(kargs.as_ptr[0]) + i_k_a;
            const BDataType* b_ptr = static_cast<const BDataType*>(kargs.bs_ptr[0]) + i_k_b;
            CDataType* c_ptr       = static_cast<CDataType*>(kargs.e_ptr);

            // Create block windows using specialized methods
            const auto& as_block_window =
                UniversalGemmKernel::MakeABlockWindows({a_ptr}, kargs, k_size, i_m);
            const auto& bs_block_window =
                UniversalGemmKernel::MakeBBlockWindows({b_ptr}, kargs, k_size, i_n);
            const auto& ds_block_window =
                UniversalGemmKernel::MakeDBlockWindows({/*ds_ptr*/}, kargs, i_m, i_n);

            // Run GEMM cooperatively by whole workgroup. Every strategy shares this one pipeline
            // instance; only what follows depends on how the tile is reduced.
            const auto& c_block_tile = RunTilePipeline(
                as_block_window, bs_block_window, num_loop_sk, smem_ptr_0, barriers);

            if constexpr(TilePartitioner::ReductionStrategy == StreamKReductionStrategy::Atomic)
            {
                // A workgroup that covers the whole tile is its only writer: a plain store skips
                // the read-modify-write of an atomic add (a CAS loop for bf16 on some targets).
                // Branch on the epilogue only, so the pipeline is not inlined once per CMemOp.
                if(num_loop_sk == kargs.tile_partitioner.get_iters_per_tile())
                {
                    RunTileEpilogue<memory_operation_enum::set>(
                        c_ptr, kargs, i_m, i_n, c_block_tile, ds_block_window, smem_ptr_0);
                }
                else
                {
                    RunTileEpilogue<TilePartitioner::MemoryOperation>(
                        c_ptr, kargs, i_m, i_n, c_block_tile, ds_block_window, smem_ptr_0);
                }
            }
            else if constexpr(TilePartitioner::ReductionStrategy ==
                                  StreamKReductionStrategy::Linear ||
                              TilePartitioner::ReductionStrategy == StreamKReductionStrategy::Tree)
            {

                auto tile_started = iter_start == tile_iter_start;
                auto tile_ended   = iter_end >= tile_iter_end;

                if constexpr(TilePartitioner::ReductionStrategy == StreamKReductionStrategy::Linear)
                {
                    if(!tile_started)
                    {
                        sk_ops.StorePartial(kargs, cta_idx, c_block_tile);
                        sk_ops.SignalStorePartialDone(kargs, cta_idx);
                    }
                    else
                    {
                        auto accum_block_tile = c_block_tile;
                        if(!tile_ended)
                        {
                            const index_t iter_per_tile =
                                kargs.tile_partitioner.get_iters_per_tile();
                            const index_t iter_per_cta =
                                kargs.tile_partitioner.get_iters_per_sk_cta();
                            const index_t extra_iters = kargs.tile_partitioner.get_extra_iters();
                            int accum_iters           = local_iter_end - local_iter_start;
                            int next_cta              = cta_idx + 1;

                            while(accum_iters < iter_per_tile)
                            {
                                // Only the math waves read the partial; the load waves wait
                                // for them at the next workgroup barrier instead of polling.
                                if(IsMathWave<GemmPipeline>())
                                {
                                    sk_ops.WaitStorePartialDone(kargs, next_cta);
                                }

                                sk_ops.AccumulatePartial(kargs, next_cta, accum_block_tile);

                                accum_iters += iter_per_cta + (next_cta < extra_iters);
                                ++next_cta;
                            }
                        }

                        RunTileEpilogue<TilePartitioner::MemoryOperation>(
                            c_ptr, kargs, i_m, i_n, accum_block_tile, ds_block_window, smem_ptr_0);
                    }
                }
                else // Tree Reduction
                {
                    auto accum_block_tile      = c_block_tile;
                    index_t tile_local_cta_idx = amd_wave_read_first_lane(
                        kargs.tile_partitioner.get_tile_local_cta_index(tile_iter_start, cta_idx));

                    for(index_t stride = 1;; stride <<= 1)
                    {
                        const index_t partner_cta_idx = amd_wave_read_first_lane(cta_idx + stride);
                        const index_t partner_start_iter = amd_wave_read_first_lane(
                            kargs.tile_partitioner.get_start_iter(partner_cta_idx));
                        const bool partner_in_tile =
                            amd_wave_read_first_lane(partner_start_iter < tile_iter_end);

                        // If the partner of the workgroup who started the tile is not in this tile,
                        // then the work for this tile is done; the epilogue below stores it.
                        if(tile_started && !partner_in_tile)
                        {
                            break;
                        }

                        // It's this workgroup's turn to write to partials. All workgroups, except
                        // the workgroup who starts the tile, will write to partials, after which
                        // they have no more work to do for this tile.
                        if(tile_local_cta_idx % (stride << 1) != 0)
                        {
                            sk_ops.StorePartial(kargs, cta_idx, accum_block_tile);
                            sk_ops.SignalStorePartialDone(kargs, cta_idx);
                            break;
                        }

                        // It's this workgroup's turn to read from partials. If its partner is in
                        // the tile then it can read from partials and accumulate results.
                        if(partner_in_tile)
                        {
                            if(IsMathWave<GemmPipeline>())
                            {
                                sk_ops.WaitStorePartialDone(kargs, partner_cta_idx);
                            }
                            sk_ops.AccumulatePartial(kargs, partner_cta_idx, accum_block_tile);
                        }
                    }

                    // The epilogue stays out of the loop: inside it, the loop-carried accumulator
                    // fed both the partial store and the epilogue and cost ~34 extra VGPRs.
                    if(tile_started)
                    {
                        RunTileEpilogue<TilePartitioner::MemoryOperation>(
                            c_ptr, kargs, i_m, i_n, accum_block_tile, ds_block_window, smem_ptr_0);
                    }
                }
            }
            else
            {
                static_assert(
                    false, "An implementation does not exist for the chosen reduction strategy.");
            }

            // A workgroup that only stores a partial tile runs no epilogue, so wait for the
            // pipeline's last TDM prefetch here, before the next tile refills the same LDS.
            if constexpr(kIsCompTdmPipeline)
            {
                s_wait_tensorcnt<0>();
            }

            // Prepare for next Stream-K loop iteration.
            iter_start = tile_iter_end;
            // Load-bearing for wavelet pipelines: the load waves leave RunBarrierStub at the
            // epilogue's last barrier while the math waves are still reading the CShuffle tile
            // from LDS, and the next tile's first TDM fill targets the same LDS. StreamKDispatch's
            // barrier after each persistent DP tile plays the same role.
            block_sync_lds();
        }
    }

    /// @brief Collective, once per launch: each tile leaves the barriers at rest for the next.
    CK_TILE_DEVICE static BarrierToken InitBarriers()
    {
        if constexpr(std::is_void_v<barrier_pipeline>)
        {
            return BarrierToken{};
        }
        else
        {
            return barrier_pipeline::template init<StreamKKernel>();
        }
    }

    /**
     * @brief Entry point for the Stream-K kernel.
     *
     * @par Overview
     *     Uses StreamKDispatch to handle both persistent and non-persistent DP sections.
     *     Non-persistent: dedicated DP workgroups process full tiles, then dedicated SK
     *     workgroups share remaining K-iterations.
     *     Persistent: each workgroup loops over DP tiles (round-robin), then proceeds
     *     to SK work.
     */
    CK_TILE_DEVICE void operator()(StreamKKernelArgs kargs) const
    {
        __shared__ char smem_ptr_0[UniversalGemmKernel::GetSmemSize()];
        index_t block_idx         = ck_tile::get_block_1d_id();
        index_t grid_size         = kargs.tile_partitioner.grid_size().x;
        const index_t dp_num_loop = kargs.tile_partitioner.get_iters_per_tile();

        block_idx = kargs.tile_partitioner.remap_xcd(block_idx, grid_size, kargs.num_xccs);

        const auto barriers = InitBarriers();

        StreamKDispatch(
            kargs.tile_partitioner,
            // A DP tile has a single writer under every reduction strategy.
            [&](index_t tile_idx) {
                BaseGemm<memory_operation_enum::set>(
                    kargs, tile_idx, dp_num_loop, 0, 0, kargs.K, smem_ptr_0, barriers);
            },
            [&](index_t sk_cta_idx) { StreamKGemm(kargs, sk_cta_idx, smem_ptr_0, barriers); },
            block_idx);
    }

    private:
    // Runs the GEMM pipeline over one C tile. num_loop varies per workgroup and per Stream-K
    // iteration, so the pipeline resolves its hot loop and tail at run time.
    template <typename AsBlockWindow, typename BsBlockWindow>
    CK_TILE_DEVICE static auto RunTilePipeline(const AsBlockWindow& as_block_window,
                                               const BsBlockWindow& bs_block_window,
                                               index_t num_loop,
                                               void* smem_ptr_0,
                                               const BarrierToken barriers)
    {
        return RunGemmPipeline<GemmPipeline>(as_block_window,
                                             typename UniversalGemmKernel::AElementWise{},
                                             bs_block_window,
                                             typename UniversalGemmKernel::BElementWise{},
                                             num_loop,
                                             smem_ptr_0,
                                             barriers);
    }

    // Writes one C tile through the epilogue with CMemOp. For a wavelet pipeline only the math
    // waves store; the load waves run the epilogue's barrier stub.
    template <memory_operation_enum CMemOp, typename CBlockTile, typename DsBlockWindow>
    CK_TILE_DEVICE static void
    RunTileEpilogue(CDataType* c_ptr,
                    const typename UniversalGemmKernel::KernelArgs& kargs,
                    index_t i_m,
                    index_t i_n,
                    const CBlockTile& c_block_tile,
                    const DsBlockWindow& ds_block_window,
                    void* smem_ptr_0)
    {
        RunWaveletAwareEpilogue<GemmPipeline, EpiloguePipeline>([&]() {
            auto c_block_window =
                UniversalGemmKernel::template MakeCBlockWindows<CMemOp>(c_ptr, kargs, i_m, i_n);
            EpiloguePipeline{}(c_block_window, c_block_tile, ds_block_window, smem_ptr_0);
        });
    }

    /**
     * @brief Computes the K offsets in the A and B tensors given iter_offset, where iter_offset
     * is the starting macro tile index in the K dimension for the workgroup.
     * @return A tuple containing the offsets into the A and B tensors accounting for the
     * layouts of A and B.
     * @note The default case is that A is assumed to be row major and B is assumed to be column
     * major.
     */
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
        ck_tile::hip_check_error(hipGetDevice(&dev));
        ck_tile::hip_check_error(hipGetDeviceProperties(&dev_prop, dev));
        int num_cu = dev_prop.multiProcessorCount;

        return num_cu;
    }

    /**
     * @brief Computes the occupancy (i.e. maximum number of active blocks per CU) for the
     * kernel
     * @return The occupancy
     * @note This function queries the maximum occupancy of the kernel using
     * `hipOccupancyMaxActiveBlocksPerMultiprocessor`.
     */
    CK_TILE_HOST static int Occupancy()
    {
        int occupancy;

        // Since occupancy of 1 is valid for stream k, we set min_num_block_per_cu to 1
        constexpr int min_block_per_cu = 1;
        const auto kernel              = kentry<min_block_per_cu, Kernel, KernelArgs>;

        ck_tile::hip_check_error(
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, BlockSize().x, 0));

        // In WGP mode a gfx12 multiprocessor is a WGP, the unit the named barrier pool is per.
        return min(max(occupancy, 1), kMaxResidentPerWgpByBarriers);
    }
};

} // namespace ck_tile
