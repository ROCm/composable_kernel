// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/loop_scheduler.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"

namespace ck {

template <index_t NumPrefetch, bool AEnableLds, bool BEnableLds>
struct GridwiseGemmPipeline_v1;

// 1-stage prefetch
template <>
struct GridwiseGemmPipeline_v1<1, true, true>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    __host__ __device__ static constexpr bool IsSupported(index_t /* num_loop */) { return true; }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 1;
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
              typename BlockwiseGemm,
              typename CThreadBuffer>
    __device__ static void Run(const AGridDesc& a_grid_desc,
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
                               const BlockwiseGemm& blockwise_gemm,
                               CThreadBuffer& c_thread_buf,
                               index_t num_loop)
    {
        // preload data into LDS
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);

                block_sync_lds();

                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                block_sync_lds();

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();

            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
        }
    }
};

// 2-stage prefetch
template <>
struct GridwiseGemmPipeline_v1<2, true, true>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    __host__ __device__ static constexpr bool IsSupported(index_t)
    {
        return true;
    }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 2;
    }

    __host__ __device__ static constexpr bool CalculateIsOddLoop(index_t num_loop)
    {
        return (num_loop % 2) == 1;
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
              typename BlockwiseGemm,
              typename CThreadBuffer>
    static __device__ void Run(const AGridDesc& a_grid_desc,
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
                               const BlockwiseGemm& blockwise_gemm,
                               CThreadBuffer& c_thread_buf,
                               index_t num_loop)
    {
        // preload data into LDS
        {
            // Read 0
            a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I0);
            b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, I0);

            // Move
            a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
            b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

            // Read 1
            a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I1);
            b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, I1);
        }

        // Initialize C
        c_thread_buf.Clear();

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                // Move
                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                // Write i
                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, I0);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, I0);

                // Read i+2
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I0);
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, I0);

                // Sync
                block_sync_lds();

                // Gemm i
                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                // Sync
                block_sync_lds();

                // Move
                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                // Write i+1
                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, I1);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, I1);

                // Read i+3
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I1);
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, I1);

                // Sync
                block_sync_lds();

                // Gemm i+1
                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                // Sync
                block_sync_lds();

                i += 2;
            } while(i < (num_loop - 2));
        }

        // tail
        {
            // Write num_loop - 2
            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, I0);
            b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, I0);

            // Sync
            block_sync_lds();

            // Gemm num_loop - 2
            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

            // Sync
            block_sync_lds();

            // Write num_loop - 1
            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, I1);
            b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, I1);

            // Sync
            block_sync_lds();

            // Gemm num_loop - 1
            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
        }
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
              typename BlockwiseGemm,
              typename CThreadBuffer>
    static __device__ void Run(const AGridDesc& a_grid_desc,
                               const ABlockDesc& a_block_desc,
                               ABlockTransfer& a_blockwise_copy,
                               const AGridBuffer& a_grid_buf,
                               ABlockBuffer& a_block_buf_0,
                               ABlockBuffer& a_block_buf_1,
                               const ABlockTransferStep& a_block_copy_step,
                               const BGridDesc& b_grid_desc,
                               const BBlockDesc& b_block_desc,
                               BBlockTransfer& b_blockwise_copy,
                               const BGridBuffer& b_grid_buf,
                               BBlockBuffer& b_block_buf_0,
                               BBlockBuffer& b_block_buf_1,
                               const BBlockTransferStep& b_block_copy_step,
                               const BlockwiseGemm& blockwise_gemm,
                               CThreadBuffer& c_thread_buf,
                               index_t num_loop)
    {
        // We have two thread scratches indexed by I0, I1 for double buffering.
        // Optimized with IGLP scheduling for better instruction-level parallelism

        // Scheduling barrier to mark pipeline start
        __builtin_amdgcn_sched_barrier(0);

        // Prologue - load data into buffer 0
        // Read from global mem to registers (I0 scratch)
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I0);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, I0);

        // Move source slice window for next read (I1 scratch)
        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        // Write from I0 registers to LDS buffer 0
        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf_0, I0);
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf_0, I0);

        // Main body
        if constexpr(HasMainLoop)
        {
            index_t i = 1;

            do
            {
                // Read from global mem to registers (I1 scratch)
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I1);
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, I1);

                // Move source slice window for next read (I0 scratch)
                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                // Sync LDS to ensure buffer 0 is ready
                __builtin_amdgcn_sched_barrier(0);
                block_sync_lds();
                __builtin_amdgcn_sched_barrier(0);

                // Run GEMM on buffer 0 while buffer 1 is loading
                blockwise_gemm.Run(a_block_buf_0, b_block_buf_0, c_thread_buf);

                // Write from registers (I1 scratch) to LDS buffer 1
                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf_1, I1);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf_1, I1);

                // Read from global mem to registers (I0 scratch)
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I0);
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, I0);

                // Move source slice window for next read (I1 scratch)
                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                // Sync LDS to ensure buffer 1 is ready
                __builtin_amdgcn_sched_barrier(0);
                block_sync_lds();
                __builtin_amdgcn_sched_barrier(0);

                // Run GEMM on buffer 1 while buffer 0 is loading
                blockwise_gemm.Run(a_block_buf_1, b_block_buf_1, c_thread_buf);

                // Write from registers (I0 scratch) to LDS buffer 0
                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf_0, I0);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf_0, I0);

                i += 2;
            } while(i <= (num_loop - 2));
        }

        // tail - handle remaining iterations
        if (num_loop % 2 == 0)
        {
            // Even number of loops: need to process 2 more iterations
            // Read from global mem to registers (I1 scratch)
            a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I1);
            b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, I1);

            // Sync LDS to ensure buffer 0 is ready
            __builtin_amdgcn_sched_barrier(0);
            block_sync_lds();
            __builtin_amdgcn_sched_barrier(0);

            // Run GEMM on buffer 0
            blockwise_gemm.Run(a_block_buf_0, b_block_buf_0, c_thread_buf);

            // Write from registers (I1 scratch) to LDS buffer 1
            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf_1, I1);
            b_blockwise_copy.RunWrite(b_block_desc, b_block_buf_1, I1);

            // Sync LDS to ensure buffer 1 is ready
            __builtin_amdgcn_sched_barrier(0);
            block_sync_lds();
            __builtin_amdgcn_sched_barrier(0);

            // Run GEMM on buffer 1
            blockwise_gemm.Run(a_block_buf_1, b_block_buf_1, c_thread_buf);
        }
        else
        {
            // Odd number of loops: need to process 1 more iteration
            // Sync LDS to ensure buffer 0 is ready
            __builtin_amdgcn_sched_barrier(0);
            block_sync_lds();
            __builtin_amdgcn_sched_barrier(0);

            // Run GEMM on buffer 0
            blockwise_gemm.Run(a_block_buf_0, b_block_buf_0, c_thread_buf);
        }

        // Final barrier to complete pipeline
        __builtin_amdgcn_sched_barrier(0);
    }
};

template <>
struct GridwiseGemmPipeline_v1<1, false, true>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    __host__ __device__ static constexpr bool IsSupported(index_t /* num_loop */) { return true; }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 1;
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
              typename BlockwiseGemm,
              typename CThreadBuffer>
    __device__ static void Run(const AGridDesc& a_grid_desc,
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
                               const BlockwiseGemm& blockwise_gemm,
                               CThreadBuffer& c_thread_buf,
                               index_t num_loop)
    {
        constexpr auto a_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0);
        auto a_block_buf_switch           = a_block_buf;

        // preload data into LDS
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);
        a_blockwise_copy.Run(
            a_grid_desc, a_grid_buf, a_block_desc, a_block_origin_idx, a_block_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                a_blockwise_copy.Run(
                    a_grid_desc, a_grid_buf, a_block_desc, a_block_origin_idx, a_block_buf_switch);

                block_sync_lds();

                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                block_sync_lds();

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

                a_block_buf = a_block_buf_switch;
                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();

            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

            block_sync_lds();
        }
    }
};

template <>
struct GridwiseGemmPipeline_v1<1, true, false>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    __host__ __device__ static constexpr bool IsSupported(index_t /* num_loop */) { return true; }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 1;
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
              typename BlockwiseGemm,
              typename CThreadBuffer>
    __device__ static void Run(const AGridDesc& a_grid_desc,
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
                               const BlockwiseGemm& blockwise_gemm,
                               CThreadBuffer& c_thread_buf,
                               index_t num_loop)
    {
        constexpr auto b_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0);
        auto b_block_buf_switch           = b_block_buf;

        // preload data into LDS
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.Run(
            b_grid_desc, b_grid_buf, b_block_desc, b_block_origin_idx, b_block_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                b_blockwise_copy.Run(
                    b_grid_desc, b_grid_buf, b_block_desc, b_block_origin_idx, b_block_buf_switch);

                block_sync_lds();

                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);

                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                block_sync_lds();

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);

                b_block_buf = b_block_buf_switch;
                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();

            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

            block_sync_lds();
        }
    }
};

template <>
struct GridwiseGemmPipeline_v1<1, false, false>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    __host__ __device__ static constexpr bool IsSupported(index_t /* num_loop */) { return true; }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 1;
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
              typename BlockwiseGemm,
              typename CThreadBuffer>
    __device__ static void Run(const AGridDesc& a_grid_desc,
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
                               const BlockwiseGemm& blockwise_gemm,
                               CThreadBuffer& c_thread_buf,
                               index_t num_loop)
    {
        constexpr auto b_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0);
        constexpr auto a_block_origin_idx = make_tuple(I0, I0, I0, I0, I0, I0, I0);
        auto b_block_buf_switch           = b_block_buf;
        auto a_block_buf_switch           = a_block_buf;

        // preload data into LDS
        a_blockwise_copy.Run(
            a_grid_desc, a_grid_buf, a_block_desc, a_block_origin_idx, a_block_buf);
        b_blockwise_copy.Run(
            b_grid_desc, b_grid_buf, b_block_desc, b_block_origin_idx, b_block_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                a_blockwise_copy.Run(
                    a_grid_desc, a_grid_buf, a_block_desc, a_block_origin_idx, a_block_buf_switch);
                b_blockwise_copy.Run(
                    b_grid_desc, b_grid_buf, b_block_desc, b_block_origin_idx, b_block_buf_switch);

                block_sync_lds();

                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                block_sync_lds();

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                a_block_buf = a_block_buf_switch;
                b_block_buf = b_block_buf_switch;
                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();

            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

            block_sync_lds();
        }
    }
};

template <index_t NumPrefetch, bool AEnableLds, bool BEnableLds>
struct GridwiseGemmPipeline_v1_WeightOnly;

template <>
struct GridwiseGemmPipeline_v1_WeightOnly<1, true, true>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    __host__ __device__ static constexpr bool IsSupported(index_t /* num_loop */) { return true; }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 1;
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
              typename ScaleGridDesc,
              typename ScaleGridBuffer,
              typename BlockwiseGemm,
              typename CThreadBuffer>
    __device__ static void Run(const AGridDesc& a_grid_desc,
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
                               const ScaleGridDesc& scale_grid_desc,
                               const ScaleGridBuffer& scale_grid_buf,
                               const BlockwiseGemm& blockwise_gemm,
                               CThreadBuffer& c_thread_buf,
                               index_t num_loop)
    {
        // Global Prefetch Stage 1
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);
        // Scale read once
        b_blockwise_copy.RunScaleRead(scale_grid_desc, scale_grid_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
        // Dequantization fused in blockwise_copy
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);

                block_sync_lds();

                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                block_sync_lds();

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();

            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
        }
    }
};

template <index_t NumPrefetch>
struct GridwiseGemmPipelineInterwave_v1;

template <>
struct GridwiseGemmPipelineInterwave_v1<1>
{
    __host__ __device__ static constexpr bool IsSupported(index_t /* num_loop */) { return true; }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 1;
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
              typename BlockwiseGemm,
              typename CThreadBuffer>
    static __device__ void Run(const AGridDesc& a_grid_desc,
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
                               const BlockwiseGemm& blockwise_gemm,
                               CThreadBuffer& c_thread_buf,
                               index_t num_loop)
    {
        // preload data into LDS
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);

                block_sync_lds();

                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                // block_sync_lds(); // moved into blockwise_gemm

                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();

            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
        }
    }
};

// Note: 2 stage prefetch not optimized for inter-wave loop scheduler
template <>
struct GridwiseGemmPipelineInterwave_v1<2> : public GridwiseGemmPipeline_v1<2, true, true>
{
};

// TODO: deprecate as GridwiseGemmPipeline_Selector covers the functionality
template <index_t NumPrefetch, LoopScheduler LoopSched>
constexpr auto GridwiseGemmPipeline_v1_Selector()
{
    if constexpr(LoopSched == LoopScheduler::Default)
    {
        return GridwiseGemmPipeline_v1<NumPrefetch, true, true>{};
    }
    else if constexpr(LoopSched == LoopScheduler::Interwave)
    {
        return GridwiseGemmPipelineInterwave_v1<NumPrefetch>{};
    }
}

} // namespace ck
