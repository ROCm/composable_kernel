// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"

namespace ck {

template <typename ABBlockTransferThreadGroup,
          typename BlockGemmThreadGroup,
          index_t NumGemmKPrefetchStage>
struct GridwiseGemmPipelineProducerConsumer;

// 1-stage prefetch
template <typename ABBlockTransferThreadGroup, typename BlockGemmThreadGroup>
struct GridwiseGemmPipelineProducerConsumer<ABBlockTransferThreadGroup, BlockGemmThreadGroup, 1>
{
    __host__ __device__ static constexpr bool IsSupported(index_t num_loop)
    {
        // TODO: improve applicability
        return num_loop % 2 == 0;
    }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop / 2 > 1;
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
              typename BBlockTransferStep>
    static __device__ void RunABBlockTransferPipeline(const AGridDesc& a_grid_desc,
                                                      const ABlockDesc& a_block_desc,
                                                      ABlockTransfer& a_block_copy,
                                                      const AGridBuffer& a_grid_buf,
                                                      ABlockBuffer& a_block_buf,
                                                      const ABlockTransferStep& a_block_copy_step,
                                                      const BGridDesc& b_grid_desc,
                                                      const BBlockDesc& b_block_desc,
                                                      BBlockTransfer& b_block_copy,
                                                      const BGridBuffer& b_grid_buf,
                                                      BBlockBuffer& b_block_buf,
                                                      const BBlockTransferStep& b_block_copy_step,
                                                      index_t num_loop)
    {
        // global read 0
        a_block_copy.RunRead(a_grid_desc, a_grid_buf);
        b_block_copy.RunRead(b_grid_desc, b_grid_buf);

        // move to 1
        a_block_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_block_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // LDS write 0
        a_block_copy.RunWrite(a_block_desc, a_block_buf);
        // global Read 1
        a_block_copy.RunRead(a_grid_desc, a_grid_buf);

        // LDS write 0
        b_block_copy.RunWrite(b_block_desc, b_block_buf);
        // global Read 1
        b_block_copy.RunRead(b_grid_desc, b_grid_buf);

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                block_sync_lds();

                // GEMM i

                block_sync_lds();

                // move to i + 2
                a_block_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_block_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                // LDS write i + 1
                a_block_copy.RunWrite(a_block_desc, a_block_buf);
                // global read i + 2
                a_block_copy.RunRead(a_grid_desc, a_grid_buf);

                // LDS write i + 1
                b_block_copy.RunWrite(b_block_desc, b_block_buf);
                // global read i + 2
                b_block_copy.RunRead(b_grid_desc, b_grid_buf);

                ++i;
            } while(i < (num_loop - 2));
        }

        // tail
        {
            block_sync_lds();

            // GEMM num_loop - 2

            block_sync_lds();

            // LDS write num_loop - 1
            a_block_copy.RunWrite(a_block_desc, a_block_buf);
            b_block_copy.RunWrite(b_block_desc, b_block_buf);

            block_sync_lds();

            // GEMM num_loop - 1
        }
    }

    template <bool HasMainLoop,
              typename ABlockBuffer,
              typename BBlockBuffer,
              typename BlockwiseGemm,
              typename CThreadBuffer>
    static __device__ void RunBlockGemmPipeline(ABlockBuffer& a_block_buf,
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

                // move to i + 2

                // LDS write i + 1
                // global read i + 2

                // LDS write i + 1
                // global read i + 2

                ++i;
            } while(i < (num_loop - 2));
        }

        // tail
        {
            block_sync_lds();

            // GEMM num_loop - 2
            block_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

            block_sync_lds();

            // LDS write num_loop - 1

            block_sync_lds();

            // GEMM num_loop - 1
            block_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
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
                               ABlockTransfer& a_block_copy,
                               const AGridBuffer& a_grid_buf,
                               ABlockBuffer& a_block_buf,
                               const ABlockTransferStep& a_block_copy_step,
                               const BGridDesc& b_grid_desc,
                               const BBlockDesc& b_block_desc,
                               BBlockTransfer& b_block_copy,
                               const BGridBuffer& b_grid_buf,
                               BBlockBuffer& b_block_buf,
                               const BBlockTransferStep& b_block_copy_step,
                               const BlockwiseGemm& block_gemm,
                               CThreadBuffer& c_thread_buf,
                               index_t num_loop)
    {
        if(ABBlockTransferThreadGroup::IsBelong())
        {
            RunABBlockTransferPipeline<HasMainLoop>(a_grid_desc,
                                                    a_block_desc,
                                                    a_block_copy,
                                                    a_grid_buf,
                                                    a_block_buf,
                                                    a_block_copy_step,
                                                    b_grid_desc,
                                                    b_block_desc,
                                                    b_block_copy,
                                                    b_grid_buf,
                                                    b_block_buf,
                                                    b_block_copy_step,
                                                    num_loop);
        }
        else if(BlockGemmThreadGroup::IsBelong())
        {
            RunBlockGemmPipeline<HasMainLoop>(
                a_block_buf, b_block_buf, block_gemm, c_thread_buf, num_loop);
        }
    }
};

} // namespace ck
