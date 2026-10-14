// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/scheduler_enum.hpp"

namespace ck {

namespace impl {
template <typename T>
using is_tuple = decltype(ck::declval<T&>().IsTuple());

template <typename T, typename = void>
struct is_buffer_tuple
{
    static constexpr bool value = false;
};

template <typename T>
struct is_buffer_tuple<T, void_t<is_tuple<T>>>
{
    static constexpr bool value = true;
};
}; // namespace impl

template <typename TileLoadThreadGroup, index_t NumGemmKPrefetchStage>
struct GridwiseGemmLoadWave;

// 1-stage prefetch
template <typename TileLoadThreadGroup>
struct GridwiseGemmLoadWave<TileLoadThreadGroup, 1>
{
    __host__ __device__ static constexpr bool IsSupported(index_t /* num_loop */)
    {
        // TODO: improve applicability
        return true;
    }

    template <bool HasMainLoop,
              typename AGridDesc,
              typename ABlockDesc,
              typename ABlockTransfer,
              typename AGridBuffer,
              typename ABlockBuffer,
              typename ABlockTransferStep,
              typename BGridDesc,
              typename BBlockDesc,
              typename BBlockTransfer,
              typename BGridBuffer,
              typename BBlockBuffer,
              typename BBlockTransferStep,
              typename std::enable_if_t<!impl::is_buffer_tuple<ABlockBuffer>::value &&
                                            !impl::is_buffer_tuple<BBlockBuffer>::value,
                                        bool>* = nullptr>
    static __device__ void RunLoadWavePipeline(const AGridDesc& a_grid_desc,
                                               const ABlockDesc& a_block_desc,
                                               ABlockTransfer& a_blockwise_copy,
                                               const AGridBuffer& a_grid_buf,
                                               ABlockBuffer& a_block_buf,
                                               const ABlockTransferStep& a_block_copy_step,
                                               const BGridDesc& b_grid_desc,
                                               const BBlockDesc& b_block_desc,
                                               BBlockTransfer& b_blockwise_copy,
                                               const BGridBuffer& b_grid_buf,
                                               BBlockBuffer& b_block_buf,
                                               const BBlockTransferStep& b_block_copy_step,
                                               index_t num_loop)
    {
        // sched_group_barrier hints force VMEM reads (0x20) to issue before
        // address-advance VALU (0x02), so loads are in-flight during computation.

        // global read 0: issue VMEM loads, then VALU for address advance
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        __builtin_amdgcn_sched_group_barrier(0x20, 4, 0); // prioritize VMEM reads
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);
        __builtin_amdgcn_sched_group_barrier(0x20, 4, 0); // prioritize VMEM reads

        // move to 1 (pure VALU)
        __builtin_amdgcn_sched_group_barrier(0x02, 8, 0); // batch VALU
        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // LDS write 0
        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                // sync for Load threads()
                block_sync_lds();

                // global read i + 1: front-load VMEM reads
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
                __builtin_amdgcn_sched_group_barrier(0x20, 4, 0); // VMEM reads
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);
                __builtin_amdgcn_sched_group_barrier(0x20, 4, 0); // VMEM reads

                // move to i + 2 (pure VALU, overlaps with VMEM latency)
                __builtin_amdgcn_sched_group_barrier(0x02, 8, 0); // batch VALU
                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                // sync with math threads()
                block_sync_lds();

                // LDS write i+1
                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();
            // GEMM num_loop - 1
        }
    }

    template <bool HasMainLoop,
              TailNumber TailNum,
              typename AGridDesc,
              typename ABlockDesc,
              typename ABlockTransfer,
              typename AGridBuffer,
              typename ABlockBuffer,
              typename ABlockTransferStep,
              typename BGridDesc,
              typename BBlockDesc,
              typename BBlockTransfer,
              typename BGridBuffer,
              typename BBlockBuffer,
              typename BBlockTransferStep,
              typename std::enable_if_t<impl::is_buffer_tuple<ABlockBuffer>::value &&
                                            impl::is_buffer_tuple<BBlockBuffer>::value,
                                        bool>* = nullptr>
    static __device__ void RunLoadWavePipeline(const AGridDesc& a_grid_desc,
                                               const ABlockDesc& a_block_desc,
                                               ABlockTransfer& a_blockwise_copy,
                                               const AGridBuffer& a_grid_buf,
                                               ABlockBuffer& a_block_buf,
                                               const ABlockTransferStep& a_block_copy_step,
                                               const BGridDesc& b_grid_desc,
                                               const BBlockDesc& b_block_desc,
                                               BBlockTransfer& b_blockwise_copy,
                                               const BGridBuffer& b_grid_buf,
                                               BBlockBuffer& b_block_buf,
                                               const BBlockTransferStep& b_block_copy_step,
                                               index_t num_loop)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        // Split src descriptor index VALU and loading VMEM instructions
        // We want to issue VMEM instruction as soon as possible so we pre-compute the
        // indices of the src in the previous step and in the current step we load
        // immediately and then pre-compute indices for the next step.

        __builtin_amdgcn_sched_barrier(0);
        a_blockwise_copy.PrecomputeIdx(a_grid_desc);
        b_blockwise_copy.PrecomputeIdx(b_grid_desc);

        // First prefetch buffer A0, B0
        __builtin_amdgcn_sched_barrier(0);
        b_blockwise_copy.Load(b_grid_buf, b_block_desc, b_block_buf.At(I0));
        a_blockwise_copy.Load(a_grid_buf, a_block_desc, a_block_buf.At(I0));
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_sched_barrier(0);
        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
        a_blockwise_copy.PrecomputeIdx(a_grid_desc);
        b_blockwise_copy.PrecomputeIdx(b_grid_desc);
        __builtin_amdgcn_sched_barrier(0);

        wait_dscnt();
        block_sync_lds_direct_load();

        // Second prefetch buffer A1, B1
        __builtin_amdgcn_sched_barrier(0);
        a_blockwise_copy.Load(a_grid_buf, a_block_desc, a_block_buf.At(I1));
        b_blockwise_copy.Load(b_grid_buf, b_block_desc, b_block_buf.At(I1));
        __builtin_amdgcn_sched_barrier(0);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
        a_blockwise_copy.PrecomputeIdx(a_grid_desc);
        b_blockwise_copy.PrecomputeIdx(b_grid_desc);
        __builtin_amdgcn_sched_barrier(0);

        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            // Note: In case of odd num_loop, the last iteration is doing an extra load of A1 and
            // B1. We force MNKPadding, so all lanes will issue ds_store and fill LDS with 0s. The
            // math waves will not use this data and go directly to the epilogue which is using LDS
            // from A0 and B0, so this is safe and it allows to simplify the code and the kernel
            // invoker.
            do
            {
                wait_dscnt();
                block_sync_lds_direct_load();

                // Load A0, B0. Precompute indices for A1, B1
                __builtin_amdgcn_sched_barrier(0);
                b_blockwise_copy.Load(b_grid_buf, b_block_desc, b_block_buf.At(I0));
                a_blockwise_copy.Load(a_grid_buf, a_block_desc, a_block_buf.At(I0));
                __builtin_amdgcn_sched_barrier(0);

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
                a_blockwise_copy.PrecomputeIdx(a_grid_desc);
                b_blockwise_copy.PrecomputeIdx(b_grid_desc);
                __builtin_amdgcn_sched_barrier(0);

                wait_dscnt();
                block_sync_lds_direct_load();

                // Load A1, B1. Precompute indices for A0, B0
                __builtin_amdgcn_sched_barrier(0);
                a_blockwise_copy.Load(a_grid_buf, a_block_desc, a_block_buf.At(I1));
                b_blockwise_copy.Load(b_grid_buf, b_block_desc, b_block_buf.At(I1));
                __builtin_amdgcn_sched_barrier(0);

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
                a_blockwise_copy.PrecomputeIdx(a_grid_desc);
                b_blockwise_copy.PrecomputeIdx(b_grid_desc);
                __builtin_amdgcn_sched_barrier(0);

                i += 2;
            } while(i < (num_loop - 2));
        }

        // tail
        if constexpr(TailNum == TailNumber::Even)
        {
            // Wait for last load of A1, B1 (to be processed by math waves)
            wait_dscnt();
            block_sync_lds_direct_load();
        }
    }
};

template <typename TileMathThreadGroup, index_t NumGemmKPrefetchStage>
struct GridwiseGemmMathWave;
// 1- stage prefetch
template <typename TileMathThreadGroup>
struct GridwiseGemmMathWave<TileMathThreadGroup, 1>
{

    __host__ __device__ static constexpr bool IsSupported(index_t /* num_loop */) { return true; }

    template <bool HasMainLoop,
              typename ABlockBuffer,
              typename BBlockBuffer,
              typename BlockwiseGemm,
              typename CThreadBuffer,
              typename std::enable_if_t<!impl::is_buffer_tuple<ABlockBuffer>::value &&
                                            !impl::is_buffer_tuple<BBlockBuffer>::value,
                                        bool>* = nullptr>
    static __device__ void RunMathWavePipeline(ABlockBuffer& a_block_buf,
                                               BBlockBuffer& b_block_buf,
                                               const BlockwiseGemm& block_gemm,
                                               CThreadBuffer& c_thread_buf,
                                               index_t num_loop)
    {
        // Initialize C
        c_thread_buf.Clear();

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                block_sync_lds();

                // GEMM i
                block_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                block_sync_lds();
                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();

            // GEMM num_loop - 1
            block_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
        }
    }

    template <bool HasMainLoop,
              TailNumber TailNum,
              typename ABlockBuffer,
              typename BBlockBuffer,
              typename BlockwiseGemm,
              typename CThreadBuffer,
              typename std::enable_if_t<impl::is_buffer_tuple<ABlockBuffer>::value &&
                                            impl::is_buffer_tuple<BBlockBuffer>::value,
                                        bool>* = nullptr>
    static __device__ void RunMathWavePipeline(ABlockBuffer& a_block_buf,
                                               BBlockBuffer& b_block_buf,
                                               const BlockwiseGemm& block_gemm,
                                               CThreadBuffer& c_thread_buf,
                                               index_t num_loop)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        // Initialize C
        c_thread_buf.Clear();

        block_sync_lds();

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                __builtin_amdgcn_sched_barrier(0);
                // GEMM A0, B0
                block_gemm.Run(a_block_buf.At(I0), b_block_buf.At(I0), c_thread_buf);

                block_sync_lds();

                __builtin_amdgcn_sched_barrier(0);

                // GEMM A1, B1
                block_gemm.Run(a_block_buf.At(I1), b_block_buf.At(I1), c_thread_buf);

                block_sync_lds();

                __builtin_amdgcn_sched_barrier(0);
                i += 2;
            } while(i < (num_loop - 2));
        }

        // tail
        if constexpr(TailNum == TailNumber::Odd)
        {
            // GEMM A0, B0
            block_gemm.Run(a_block_buf.At(I0), b_block_buf.At(I0), c_thread_buf);
        }
        else if constexpr(TailNum == TailNumber::Even)
        {
            // GEMM A0, B0
            block_gemm.Run(a_block_buf.At(I0), b_block_buf.At(I0), c_thread_buf);

            block_sync_lds();

            // GEMM A1, B1
            block_gemm.Run(a_block_buf.At(I1), b_block_buf.At(I1), c_thread_buf);
        }
    }
};

} // namespace ck
