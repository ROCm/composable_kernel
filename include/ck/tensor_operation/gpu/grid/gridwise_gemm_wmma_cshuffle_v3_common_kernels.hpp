// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/grid/epilogue_type.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/utility/scheduler_enum.hpp"

namespace ck {

template <typename GridwiseGemm,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum EGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
    kernel_gemm_wmma_cshuffle_v3(typename GridwiseGemm::Argument karg)
{
#if(defined(__gfx11__) || defined(__gfx12__))
#if defined(__gfx11__)
    // gfx11 does not support *_atomic_pk_add_f16/bf16 instructions
    using e_data_type = remove_cvref_t<remove_pointer_t<decltype(karg.p_e_grid)>>;
    if constexpr(!(EGlobalMemoryDataOperation == InMemoryDataOperationEnum::AtomicAdd &&
                   (std::is_same_v<e_data_type, ck::half_t> ||
                    std::is_same_v<e_data_type, ck::bhalf_t>)))
    {
#endif
        constexpr auto epilogue_type =
            GridwiseGemm::IsBWaveTransferApplicable && GridwiseGemm::UseDirectStore
                ? EpilogueType::DirectStore
                : EpilogueType::CShuffle;
        using SelectedEpilogue = get_epilogue_t<epilogue_type, GridwiseGemm>;

        constexpr index_t LDS_size =
            GridwiseGemm::template GetSharedMemoryNumberOfByte<SelectedEpilogue>();
        __shared__ char p_shared[LDS_size];

        auto splitk_batch_offset = typename GridwiseGemm::SplitKBatchOffset(karg, blockIdx.z);

        auto epilogue_args = SelectedEpilogue{};

        GridwiseGemm::template Run<HasMainKBlockLoop, EGlobalMemoryDataOperation, TailNum>(
            p_shared, splitk_batch_offset, karg, epilogue_args);

#if defined(__gfx11__)
    }
#endif
#else
    ignore = karg;
#endif
}

template <typename GridwiseGemm,
          typename ComputePtrOffsetOfStridedBatch,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          bool IsBScaled           = false,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
    kernel_batched_gemm_wmma_cshuffle_v3(
        typename GridwiseGemm::Argument karg, // This works for now but it actually receives a
                                              // DeviceBatchedGemm_Wmma_CShuffleV3::Argument
                                              // argument through implicit conversion to base class!
        const ComputePtrOffsetOfStridedBatch compute_ptr_offset_of_batch)
{
#if(defined(__gfx11__) || defined(__gfx12__))
#if defined(__gfx11__)
    // gfx11 does not support *_atomic_pk_add_f16/bf16 instructions
    using c_data_type = remove_cvref_t<remove_pointer_t<decltype(karg.p_e_grid)>>;
    if constexpr(!(CGlobalMemoryDataOperation == InMemoryDataOperationEnum::AtomicAdd &&
                   (std::is_same_v<c_data_type, ck::half_t> ||
                    std::is_same_v<c_data_type, ck::bhalf_t>)))
    {
#endif
        // The normal approach to batching would be to increase the grid size by just stretching out
        // the grid Z dimension (which is the outermost dimension), but this depends on lower level
        // functions not directly using the Z dimension for other calculations. As it turns out, k
        // batching does rely directly on blockIdx.Z through SplitKBatchOffset. Therefore, for now
        // we will use the grid Y dimension for batching. This may be a bit fragile.
        const index_t g_idx = amd_wave_read_first_lane(blockIdx.y);

        const long_index_t a_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx));
        const long_index_t b_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx));
        const long_index_t c_batch_offset =
            amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetCPtrOffset(g_idx));

        constexpr auto epilogue_type =
            GridwiseGemm::IsBWaveTransferApplicable && GridwiseGemm::UseDirectStore
                ? EpilogueType::DirectStore
                : EpilogueType::CShuffle;
        using SelectedEpilogue = get_epilogue_t<epilogue_type, GridwiseGemm>;

        constexpr index_t LDS_size =
            GridwiseGemm::template GetSharedMemoryNumberOfByte<SelectedEpilogue>();

        __shared__ char p_shared[LDS_size];

        auto splitk_batch_offset = typename GridwiseGemm::SplitKBatchOffset(karg, blockIdx.z);

        // shift A matrices pointer for splitk
        typename GridwiseGemm::AsGridPointer p_as_grid_shift;
        static_for<0, GridwiseGemm::NumATensor, 1>{}([&](auto i) {
            using ADataType_ =
                remove_cvref_t<tuple_element_t<i.value, typename GridwiseGemm::AsDataType_>>;
            p_as_grid_shift(i) = static_cast<const ADataType_*>(karg.p_as_grid[i]) +
                                 splitk_batch_offset.a_k_split_offset[i] + a_batch_offset;
        });

        // shift B matrices pointer for splitk
        typename GridwiseGemm::BsGridPointer p_bs_grid_shift;
        static_for<0, GridwiseGemm::NumBTensor, 1>{}([&](auto i) {
            using BDataType_ =
                remove_cvref_t<tuple_element_t<i.value, typename GridwiseGemm::BsDataType_>>;
            p_bs_grid_shift(i) = static_cast<const BDataType_*>(karg.p_bs_grid[i]) +
                                 splitk_batch_offset.b_k_split_offset[i] + b_batch_offset;
        });

        auto epilogue_args = SelectedEpilogue{};

        if constexpr(IsBScaled)
        {
            const long_index_t b_scale_batch_offset =
                amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetScaleBPtrOffset(g_idx));

            GridwiseGemm::template Run<HasMainKBlockLoop, CGlobalMemoryDataOperation, TailNum>(
                p_as_grid_shift,
                p_bs_grid_shift,
                karg.p_ds_grid,
                karg.p_e_grid + splitk_batch_offset.c_reduce_offset + c_batch_offset,
                karg.p_a_scale_grid,
                karg.p_b_scale_grid + b_scale_batch_offset +
                    splitk_batch_offset.scale_b_k_split_offset,
                p_shared,
                karg,
                karg.a_element_op,
                karg.b_element_op,
                karg.cde_element_op,
                epilogue_args);
        }
        else
        {
            GridwiseGemm::template Run<HasMainKBlockLoop, CGlobalMemoryDataOperation, TailNum>(
                p_as_grid_shift,
                p_bs_grid_shift,
                karg.p_ds_grid,
                karg.p_e_grid + splitk_batch_offset.c_reduce_offset + c_batch_offset,
                p_shared,
                karg,
                karg.a_element_op,
                karg.b_element_op,
                karg.cde_element_op,
                epilogue_args);
        }
#if defined(__gfx11__)
    }
#endif
#else
    ignore = karg;
    ignore = compute_ptr_offset_of_batch;
#endif
}

template <typename GridwiseGemm,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum EGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
    kernel_gemm_b_preshuffle_wmma_cshuffle_v3(typename GridwiseGemm::Argument karg)
{
#if(defined(__gfx11__) || defined(__gfx12__))
#if defined(__gfx11__)
    // gfx11 does not support *_atomic_pk_add_f16/bf16 instructions
    using e_data_type = remove_cvref_t<remove_pointer_t<decltype(karg.p_e_grid)>>;
    if constexpr(!(EGlobalMemoryDataOperation == InMemoryDataOperationEnum::AtomicAdd &&
                   (std::is_same_v<e_data_type, ck::half_t> ||
                    std::is_same_v<e_data_type, ck::bhalf_t>)))
    {
#endif

        using SelectedEpilogue = get_epilogue_t<EpilogueType::CShuffle, GridwiseGemm>;

        constexpr index_t LDS_size =
            GridwiseGemm::template GetSharedMemoryNumberOfByte<SelectedEpilogue>();
        __shared__ char p_shared[LDS_size];

        auto splitk_batch_offset = typename GridwiseGemm::SplitKBatchOffset(karg, blockIdx.z);

        const index_t num_k_per_block = math::integer_divide_ceil(karg.K, GridwiseGemm::KPack);
        const index_t k_id            = blockIdx.z * num_k_per_block;

        auto epilogue_args = SelectedEpilogue{};

        GridwiseGemm::template Run<HasMainKBlockLoop, EGlobalMemoryDataOperation, TailNum>(
            p_shared,
            splitk_batch_offset,
            karg,
            epilogue_args,
            0, /* A_k_id == 0 (we shift the pointer for splitk) */
            k_id);

#if defined(__gfx11__)
    }
#endif
#else
    ignore = karg;
#endif
}

} // namespace ck
