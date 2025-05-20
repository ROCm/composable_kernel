// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/utility/literals.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/host/stream_utils.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include "ck_tile/host.hpp"

#include <hip/hip_runtime.h>

namespace ck_tile {

struct GemmTransKernelArg
{
    GemmKernelArgs group_karg;
    ck_tile::index_t block_start;
    ck_tile::index_t block_end;

    GemmTransKernelArg() = default;
    GemmTransKernelArg(GemmKernelArgs&& karg, index_t bl_start, index_t bl_end)
        : group_karg{karg}, block_start{bl_start}, block_end{bl_end}
    {
    }

    GemmTransKernelArg(GemmKernelArgs&& karg) : group_karg{karg}, block_start{0}, block_end{0} {}
};

template <typename TilePartitioner_, typename GemmPipeline_, typename EpiloguePipeline_>
struct GroupedGemmKernel : public GemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>
{
    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline     = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;
    using ALayout          = remove_cvref_t<typename GemmPipeline::ALayout>;
    using BLayout          = remove_cvref_t<typename GemmPipeline::BLayout>;
    using CLayout          = remove_cvref_t<typename GemmPipeline::CLayout>;

    using ADataType = remove_cvref_t<typename GemmPipeline::ADataType>;
    using BDataType = remove_cvref_t<typename GemmPipeline::BDataType>;
    using CDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    using OffsetTile1DPartitioner = OffsettedTile1DPartitioner<TilePartitioner>;
    using Base                    = GemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>;
    using Kernel = GroupedGemmKernel<TilePartitioner, GemmPipeline, EpiloguePipeline>;

    static constexpr index_t KernelBlockSize  = GemmPipeline::BlockSize;
    static constexpr bool UsePersistentKernel = GemmPipeline::UsePersistentKernel;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        using P_ = GemmPipeline;

        return concat('_', "gemm_grouped", gemm_prec_str<ADataType, BDataType>,
                      concat('x', P_::MPerBlock, P_::NPerBlock, P_::KPerBlock),
                      concat('x', P_::GetVectorSizeA(), P_::GetVectorSizeB(), P_::GetVectorSizeC()),
                      concat('x', P_::kPadM, P_::kPadN, P_::kPadK),
                      (UsePersistentKernel ? "Persistent" : "NonPersistent"));
        // clang-format on
    }

    CK_TILE_HOST static auto GetWorkSpaceSize(const std::vector<GemmHostArgs>& gemm_descs)
        -> std::size_t
    {
        return gemm_descs.size() * sizeof(GemmTransKernelArg);
    }

    CK_TILE_HOST static auto GetWorkSpaceSize(index_t group_count) -> std::size_t
    {
        return group_count * sizeof(GemmTransKernelArg);
    }

    CK_TILE_HOST static constexpr auto BlockSize() -> dim3 { return dim3(KernelBlockSize); }

    /**
     * @brief Get the maximum occupancy grid size for the persistent kernel on the current device.
     * @return The maximum occupancy grid size.
     * @note This function queries the maximum occupancy of the kernel using
     *       `hipOccupancyMaxActiveBlocksPerMultiprocessor`.
     */
    CK_TILE_HOST static auto MaxOccupancyGridSize(const stream_config& s) -> dim3
    {
        using ConstantPointer = const void CK_CONSTANT_ADDRESS_SPACE*;
        const auto kernel     = kentry<KernelBlockSize, 1, Kernel, ConstantPointer, index_t>;
        int occupancy;
        HIP_CHECK_ERROR(
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, KernelBlockSize, 0));
        const int grid_size = get_available_compute_units(s) * occupancy;
        return dim3(grid_size, 1, 1);
    }

    CK_TILE_HOST static constexpr auto GridSize(const std::vector<GemmHostArgs>& gemm_descs)
    {
        index_t grid_size = 0;
        for(const auto& it_desc : gemm_descs)
        {
            const auto local_grid_size = TilePartitioner::GridSize(it_desc.M, it_desc.N);
            grid_size += local_grid_size * it_desc.k_batch;
        }
        return dim3(grid_size, 1, 1);
    }

    CK_TILE_HOST static auto MakeKargs(const std::vector<GemmHostArgs>& gemm_descs)
        -> std::vector<GemmTransKernelArg>
    {
        std::vector<GemmTransKernelArg> gemm_kernel_args_;
        index_t group_count = ck_tile::type_convert<ck_tile::index_t>(gemm_descs.size());
        index_t grid_size   = 0;
        gemm_kernel_args_.reserve(group_count);

        for(std::size_t i = 0; i < gemm_descs.size(); ++i)
        {
            const index_t M = gemm_descs[i].M;
            const index_t N = gemm_descs[i].N;
            const index_t K = gemm_descs[i].K;

            if(M == 0 || N == 0 || K == 0)
            {
                continue;
            }

            const index_t stride_a = gemm_descs[i].stride_A;
            const index_t stride_b = gemm_descs[i].stride_B;
            const index_t stride_c = gemm_descs[i].stride_C;

            const index_t grid_size_grp = TilePartitioner::GridSize(M, N) * gemm_descs[i].k_batch;

            const index_t block_start = grid_size;
            const index_t block_end   = grid_size + grid_size_grp;

            grid_size += grid_size_grp;

            auto karg = GemmKernelArgs{type_convert<const ADataType*>(gemm_descs[i].a_ptr),
                                       type_convert<const BDataType*>(gemm_descs[i].b_ptr),
                                       type_convert<CDataType*>(gemm_descs[i].c_ptr),
                                       M,
                                       N,
                                       K,
                                       stride_a,
                                       stride_b,
                                       stride_c,
                                       gemm_descs[i].k_batch};

            gemm_kernel_args_.emplace_back(std::move(karg), block_start, block_end);
        }

        return gemm_kernel_args_;
    }

    CK_TILE_HOST static bool IsSupportedArgument(const std::vector<GemmTransKernelArg>& kargs)
    {
        for(const auto& karg : kargs)
        {
            if(!Base::IsSupportedArgument(karg.group_karg))
            {
                return false;
            }
        }
        return true;
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetSmemSize() -> index_t
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void Run(const GemmTransKernelArg& kargs,
                            const tuple<index_t, index_t>& block_idx_2d,
                            const index_t block_idx_z) const
    {
        Run(kargs.group_karg, block_idx_2d, block_idx_z);
    }

    CK_TILE_DEVICE void Run(const GemmKernelArgs& kargs,
                            const tuple<index_t, index_t>& block_idx_2d,
                            const index_t block_idx_z) const
    {
        const auto [iM, iN] = block_idx_2d;

        const index_t i_m = __builtin_amdgcn_readfirstlane(iM * TilePartitioner::MPerBlock);
        const index_t i_n = __builtin_amdgcn_readfirstlane(iN * TilePartitioner::NPerBlock);

        const typename Base::SplitKBatchOffset splitk_batch_offset(kargs, block_idx_z);

        const ADataType* a_ptr =
            static_cast<const ADataType*>(kargs.a_ptr) + splitk_batch_offset.a_k_split_offset;
        const BDataType* b_ptr =
            static_cast<const BDataType*>(kargs.b_ptr) + splitk_batch_offset.b_k_split_offset;
        CDataType* c_ptr = static_cast<CDataType*>(kargs.c_ptr);

        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        if constexpr(UsePersistentKernel)
        {
            RunGemmWithPipelineSelection(
                a_ptr, b_ptr, c_ptr, smem_ptr, kargs, splitk_batch_offset, i_m, i_n);
        }
        else
        {
            this->RunGemm(a_ptr, b_ptr, c_ptr, smem_ptr, kargs, splitk_batch_offset, i_m, i_n);
        }
    }

    /**
     * @brief Runs single GEMM problem cooperatively by whole workgroup.
     *
     * @note The GEMM pipeline is selected in-kernel based on the number of K-loops
     *       and the tail-number. This is needed for the persistent tile-loop when
     *       we didn't have access to the K dimension on the host.
     *
     * @param a_ptr input A pointer
     * @param b_ptr input B pointer
     * @param c_ptr output C pointer
     * @param smem_ptr_0 The start memory pointer of the shared memory block.
     * @param kargs GEMM kernel arguments
     * @param splitk_batch_offset splitk_batch_offset Utility structure used to calculate k batch.
     * @param block_idx_m The GEMM's output M dimension tile index processed by this workgroup.
     * @param block_idx_n The GEMM's output N dimension tile index processed by this workgroup.
     *
     */
    CK_TILE_DEVICE static void
    RunGemmWithPipelineSelection(const ADataType* a_ptr,
                                 const BDataType* b_ptr,
                                 CDataType* c_ptr,
                                 void* smem_ptr_0,
                                 const GemmKernelArgs& kargs,
                                 const typename Base::SplitKBatchOffset& splitk_batch_offset,
                                 const index_t block_idx_m,
                                 const index_t block_idx_n)
    {
        // Create Gemm tensor views, pad views and tile windows
        const auto& gemm_tensor_views_tuple =
            Base::template MakeGemmTensorViews<EpiloguePipeline::MemoryOperation>(
                a_ptr, b_ptr, c_ptr, kargs, splitk_batch_offset);

        const auto& gemm_pad_views = Base::MakeGemmPadViews(gemm_tensor_views_tuple);
        auto gemm_tile_windows =
            Base::MakeGemmTileWindows(gemm_pad_views, block_idx_m, block_idx_n);
        const auto& a_block_window = gemm_tile_windows.at(Base::I0);
        const auto& b_block_window = gemm_tile_windows.at(Base::I1);

        // Get hot-loop and tail configuration
        const index_t num_loop = __builtin_amdgcn_readfirstlane(
            TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k));
        const bool has_hot_loop   = GemmPipeline::BlockHasHotloop(num_loop);
        const TailNumber tail_num = GemmPipeline::GetBlockLoopTailNum(num_loop);

        const auto RunEpilogue = [&](auto& c_block_tile) {
            // Run Epilogue Pipeline
            auto& c_block_window = gemm_tile_windows.at(Base::I2);
            EpiloguePipeline{}
                .template operator()<decltype(c_block_window), decltype(c_block_tile)>(
                    c_block_window, c_block_tile, smem_ptr_0);
        };

        if constexpr(is_specialization_of<GemmPipeline, GemmPipelineAgBgCrCompV3>::value)
        {
            // Run the specific implementation with hotloop+tailnum config
            using PipelineImpl =
                typename GemmPipeline::template PipelineImpl<GemmPipeline::Scheduler>;
            const auto PassThrough = [](const auto& a) { return a; };
            if(has_hot_loop && tail_num == TailNumber::Full)
            {
                const auto& c_block_tile =
                    PipelineImpl{}.template operator()<true, TailNumber::Full>(a_block_window,
                                                                               PassThrough,
                                                                               b_block_window,
                                                                               PassThrough,
                                                                               num_loop,
                                                                               smem_ptr_0);
                RunEpilogue(c_block_tile);
            }
            else if(has_hot_loop && tail_num == TailNumber::Odd)
            {
                const auto& c_block_tile =
                    PipelineImpl{}.template operator()<true, TailNumber::Odd>(a_block_window,
                                                                              PassThrough,
                                                                              b_block_window,
                                                                              PassThrough,
                                                                              num_loop,
                                                                              smem_ptr_0);
                RunEpilogue(c_block_tile);
            }
            else if(has_hot_loop && tail_num == TailNumber::Even)
            {
                const auto& c_block_tile =
                    PipelineImpl{}.template operator()<true, TailNumber::Even>(a_block_window,
                                                                               PassThrough,
                                                                               b_block_window,
                                                                               PassThrough,
                                                                               num_loop,
                                                                               smem_ptr_0);
                RunEpilogue(c_block_tile);
            }
        }
        else
        {
            ignore = a_block_window;
            ignore = b_block_window;
            static_assert(false, "GemmPipeline specialization not supported!");
        }
    }

    CK_TILE_DEVICE index_t FindGroupId(const GemmTransKernelArg* gemm_desc_ptr,
                                       index_t block_id,
                                       index_t group_count) const
    {
        index_t left     = 0;
        index_t right    = group_count;
        index_t group_id = index_t((left + right) >> 1);

        while((!(block_id >= gemm_desc_ptr[group_id].block_start &&
                 block_id < gemm_desc_ptr[group_id].block_end)) &&
              left <= right)
        {
            if(block_id < gemm_desc_ptr[group_id].block_start)
            {
                right = group_id;
            }
            else
            {
                left = group_id;
            }
            group_id = index_t((left + right) >> 1);
        }

        return group_id;
    }

    // For non-persistent kernels
    template <bool U = UsePersistentKernel, typename = std::enable_if_t<!U>>
    CK_TILE_DEVICE void operator()(const void CK_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
                                   index_t group_count) const
    {
        const index_t block_id   = ck_tile::get_block_1d_id();
        const auto gemm_desc_ptr = reinterpret_cast<const GemmTransKernelArg*>(
            cast_pointer_to_generic_address_space(gemm_descs_const));

        const index_t group_id  = FindGroupId(gemm_desc_ptr, block_id, group_count);
        const auto& kargs       = gemm_desc_ptr[group_id];
        const auto grid_size_2d = TilePartitioner::GridSize(kargs.group_karg.M, kargs.group_karg.N);
        const auto block_idx_2d = OffsetTile1DPartitioner::GetOffsetedTileIndex(
            0,
            kargs.group_karg.M,
            kargs.group_karg.N,
            (block_id - kargs.block_start) % grid_size_2d);
        Run(kargs, block_idx_2d, (block_id - kargs.block_start) / grid_size_2d);
    }

    // For persistent kernels
    template <bool U   = UsePersistentKernel,
              typename = std::enable_if_t<U>,
              typename = void> // extra template parameter to avoid redefinition
    CK_TILE_DEVICE void operator()(const void CK_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
                                   const index_t group_count) const
    {
        const index_t grid_size  = ck_tile::get_grid_size();
        const auto gemm_desc_ptr = reinterpret_cast<const GemmTransKernelArg*>(
            cast_pointer_to_generic_address_space(gemm_descs_const));
        index_t block_id      = ck_tile::get_block_1d_id(); // initial block_id
        index_t cum_grid_size = 0;
        for(index_t group_id = 0; group_id < group_count; ++group_id)
        {
            const auto& kargs      = gemm_desc_ptr[group_id].group_karg;
            const auto& k_batch    = kargs.k_batch;
            const auto block_start = cum_grid_size;
            cum_grid_size += TilePartitioner::GridSize(kargs.M, kargs.N) * k_batch;
            while(block_id < cum_grid_size)
            {
                const auto grid_size_2d = TilePartitioner::GridSize(kargs.M, kargs.N);
                const auto block_idx_2d = OffsetTile1DPartitioner::GetOffsetedTileIndex(
                    0, kargs.M, kargs.N, (block_id - block_start) % grid_size_2d);
                Run(kargs, block_idx_2d, (block_id - block_start) / grid_size_2d);
                block_id = block_id + grid_size; // advance to next block
                // NOTE: this check is redundant but helps the compiler avoid spilling some VGPR
                if(block_id >= cum_grid_size)
                {
                    break; // exit the loop if all blocks are processed
                }
            }
        }
    }
};

} // namespace ck_tile
