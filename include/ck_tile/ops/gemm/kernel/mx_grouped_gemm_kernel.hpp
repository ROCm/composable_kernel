// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/host/stream_utils.hpp"
#include "ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp"
#include "ck_tile/ops/gemm/kernel/mx_gemm_kernel.hpp"
#include "ck_tile/host.hpp"

#include <hip/hip_runtime.h>

namespace ck_tile {

/// @brief Host args for MX Grouped GEMM - extends GroupedGemmHostArgs with A/B scale pointers.
template <index_t NumDTensor = 0>
struct MxGroupedGemmHostArgs : public GroupedGemmHostArgs<NumDTensor>
{
    CK_TILE_HOST explicit MxGroupedGemmHostArgs(const void* a_ptr_,
                                                const void* a_scale_ptr_,
                                                const void* b_ptr_,
                                                const void* b_scale_ptr_,
                                                const std::array<const void*, NumDTensor>& ds_ptr_,
                                                void* e_ptr_,
                                                index_t k_batch_,
                                                index_t M_,
                                                index_t N_,
                                                index_t K_,
                                                index_t stride_A_,
                                                index_t stride_B_,
                                                const std::array<index_t, NumDTensor>& stride_Ds_,
                                                index_t stride_E_)
        : GroupedGemmHostArgs<NumDTensor>(a_ptr_,
                                          b_ptr_,
                                          ds_ptr_,
                                          e_ptr_,
                                          k_batch_,
                                          M_,
                                          N_,
                                          K_,
                                          stride_A_,
                                          stride_B_,
                                          stride_Ds_,
                                          stride_E_),
          a_scale_ptr(a_scale_ptr_),
          b_scale_ptr(b_scale_ptr_)
    {
    }

    const void* a_scale_ptr;
    const void* b_scale_ptr;
};

/// @brief Per-group device kernel args: wraps MxGemmKernelArgs + block range [block_start,
/// block_end).
template <index_t NumDTensor = 0>
struct MxGemmTransKernelArg
{
    MxGemmKernelArgs<1, 1, NumDTensor> group_karg;
    ck_tile::index_t block_start;
    ck_tile::index_t block_end;

    MxGemmTransKernelArg() = delete;

    MxGemmTransKernelArg(MxGemmKernelArgs<1, 1, NumDTensor>&& karg,
                         index_t bl_start,
                         index_t bl_end)
        : group_karg{std::move(karg)}, block_start{bl_start}, block_end{bl_end}
    {
    }

    explicit MxGemmTransKernelArg(MxGemmKernelArgs<1, 1, NumDTensor>&& karg)
        : group_karg{std::move(karg)}, block_start{0}, block_end{0}
    {
    }
};

/// @brief MX Grouped GEMM kernel.
///
/// @par Overview
///      Combines the multi-group dispatch logic of GroupedGemmKernel with the MX microscaling
///      support of MxGemmKernel. Each group gets its own A/B data pointers and A/B scale
///      pointers. The kernel dispatches each workgroup to the correct GEMM group using a
///      binary-search (non-persistent) or a wave-front tile loop (persistent), then delegates
///      the actual computation to MxGemmKernel::RunGemm which builds the scale windows and
///      calls the MX pipeline.
template <typename TilePartitioner_, typename GemmPipeline_, typename EpiloguePipeline_>
struct MxGroupedGemmKernel
{
    /// @brief MxGemmKernel provides scale window creation and RunGemm with scale support.
    using Base = MxGemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>;

    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline     = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;

    using ADataType  = remove_cvref_t<typename Base::ADataType>;
    using BDataType  = remove_cvref_t<typename Base::BDataType>;
    using CDataType  = remove_cvref_t<typename Base::EDataType>;
    using DsDataType = remove_cvref_t<typename EpiloguePipeline::DsDataType>;

    static constexpr index_t NumDTensor_ = DsDataType::size();

    using OffsetTile1DPartitioner = OffsettedTile1DPartitioner<TilePartitioner>;
    using Kernel = MxGroupedGemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>;

    static constexpr index_t kBlockSize = GemmPipeline::BlockSize;
    static constexpr bool UsePersistentKernel =
        false; // hardcoded, pipeline does not support it now

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        using P_ = GemmPipeline;
        return concat('_', "mx_gemm_grouped", gemm_prec_str<ADataType, BDataType>(),
                      concat('x', P_::MPerBlock, P_::NPerBlock, P_::KPerBlock),
                      concat('x', P_::GetVectorSizeA(), P_::GetVectorSizeB(), P_::GetVectorSizeC()),
                      concat('x', P_::kPadM, P_::kPadN, P_::kPadK),
                      (UsePersistentKernel ? "Persistent" : "NonPersistent"),
                      (NumDTensor_ == 2 ? "MultiD" : "NoMultiD"),
                      (GemmPipeline::DoubleSmemBuffer ? "DoubleSmemBuffer" : "SingleSmemBuffer"));
        // clang-format on
    }

    CK_TILE_HOST static auto GetWorkSpaceSize(index_t group_count) -> std::size_t
    {
        return group_count * sizeof(MxGemmTransKernelArg<NumDTensor_>);
    }

    CK_TILE_HOST static auto BlockSize() -> dim3
    {
        if(is_wave32())
        {
            return dim3(kBlockSize / 2);
        }
        else
        {
            return dim3(kBlockSize); // untested branching
        }
    }

    CK_TILE_HOST static auto
    GridSize(const std::vector<MxGemmTransKernelArg<NumDTensor_>>& kargs) -> dim3
    {
        if(kargs.empty())
            return dim3(0, 1, 1);
        return dim3(kargs.back().block_end, 1, 1);
    }

    /// @brief Convert host descriptors into per-group device kernel args.
    ///
    ///        For each group, builds a MxGemmKernelArgs (which extends UniversalGemmKernelArgs
    ///        with as_scale_ptr / bs_scale_ptr) and pairs it with the block range
    ///        [block_start, block_end) that this group owns in the flat 1-D grid.
    CK_TILE_HOST static auto
    MakeKargs(const std::vector<MxGroupedGemmHostArgs<NumDTensor_>>& gemm_descs)
        -> std::vector<MxGemmTransKernelArg<NumDTensor_>>
    {
        std::vector<MxGemmTransKernelArg<NumDTensor_>> gemm_kernel_args_;
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
            const index_t stride_e = gemm_descs[i].stride_E;
            auto stride_ds         = gemm_descs[i].stride_Ds;

            const index_t grid_size_grp = TilePartitioner::GridSize(M, N) * gemm_descs[i].k_batch;
            const index_t block_start   = grid_size;
            const index_t block_end     = grid_size + grid_size_grp;
            grid_size += grid_size_grp;

            // Build MxGemmKernelArgs: base UniversalGemmKernelArgs + MX scale pointers.
            // The nested braces initialise the UniversalGemmKernelArgs base sub-object,
            // followed by as_scale_ptr and bs_scale_ptr from MxGemmKernelArgs.
            auto karg = MxGemmKernelArgs<1, 1, NumDTensor_>{
                {// UniversalGemmKernelArgs base
                 {type_convert<const ADataType*>(gemm_descs[i].a_ptr)},
                 {type_convert<const BDataType*>(gemm_descs[i].b_ptr)},
                 gemm_descs[i].ds_ptr,
                 type_convert<CDataType*>(gemm_descs[i].e_ptr),
                 M,
                 N,
                 K,
                 {stride_a},
                 {stride_b},
                 stride_ds,
                 stride_e,
                 gemm_descs[i].k_batch},
                // MxGemmKernelArgs extensions
                {gemm_descs[i].a_scale_ptr},
                {gemm_descs[i].b_scale_ptr}};

            gemm_kernel_args_.emplace_back(std::move(karg), block_start, block_end);
        }

        return gemm_kernel_args_;
    }

    CK_TILE_HOST static bool
    IsSupportedArgument(const std::vector<MxGemmTransKernelArg<NumDTensor_>>& kargs)
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

    /// @brief Run a single GEMM group cooperatively by the whole workgroup.
    ///
    ///        Extracts the data pointers with split-K offsets applied and delegates
    ///        to MxGemmKernel::RunGemm, which builds the scale block windows and
    ///        invokes the MX pipeline (e.g. GemmPipelineAgBgCrCompTDMV1).
    CK_TILE_DEVICE void Run(const MxGemmKernelArgs<1, 1, NumDTensor_>& kargs,
                            const tuple<index_t, index_t>& block_idx_2d,
                            const index_t block_idx_z) const
    {
        const auto [iM, iN] = block_idx_2d;

        const index_t i_m = amd_wave_read_first_lane(iM * TilePartitioner::MPerBlock);
        const index_t i_n = amd_wave_read_first_lane(iN * TilePartitioner::NPerBlock);

        const typename Base::SplitKBatchOffset splitk_batch_offset(kargs, block_idx_z);

        // Apply split-K offset to the main data pointers (scale pointers are K-stationary).
        const ADataType* a_ptr = static_cast<const ADataType*>(kargs.as_ptr[0]) +
                                 splitk_batch_offset.as_k_split_offset[0];
        const BDataType* b_ptr = static_cast<const BDataType*>(kargs.bs_ptr[0]) +
                                 splitk_batch_offset.bs_k_split_offset[0];
        CDataType* c_ptr = static_cast<CDataType*>(kargs.e_ptr);

        __shared__ char smem_ptr[GetSmemSize()];

        // MxGemmKernel::RunGemm builds scale windows from kargs.as_scale_ptr /
        // kargs.bs_scale_ptr and passes them to the MX pipeline.
        Base::RunGemm(
            {a_ptr}, {b_ptr}, kargs.ds_ptr, c_ptr, smem_ptr, kargs, splitk_batch_offset, i_m, i_n);
    }

    /// @brief Binary search: find which group owns block_id.
    CK_TILE_DEVICE index_t FindGroupId(const MxGemmTransKernelArg<NumDTensor_>* gemm_desc_ptr,
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

    /// @brief Non-persistent kernel entry point.
    ///        Each workgroup binary-searches for its group and runs exactly one tile.
    template <bool U = UsePersistentKernel, typename = std::enable_if_t<!U>>
    CK_TILE_DEVICE void operator()(const void CK_TILE_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
                                   index_t group_count) const
    {
        const index_t block_id   = ck_tile::get_block_1d_id();
        const auto gemm_desc_ptr = reinterpret_cast<const MxGemmTransKernelArg<NumDTensor_>*>(
            cast_pointer_to_generic_address_space(gemm_descs_const));

        const index_t group_id = FindGroupId(gemm_desc_ptr, block_id, group_count);
        const auto& kargs      = gemm_desc_ptr[group_id];

        const auto grid_size_2d = TilePartitioner::GridSize(kargs.group_karg.M, kargs.group_karg.N);
        const auto block_idx_2d = OffsetTile1DPartitioner::GetOffsetedTileIndex(
            0,
            kargs.group_karg.M,
            kargs.group_karg.N,
            (block_id - kargs.block_start) % grid_size_2d);
        Run(kargs.group_karg, block_idx_2d, (block_id - kargs.block_start) / grid_size_2d);
    }
};

} // namespace ck_tile
