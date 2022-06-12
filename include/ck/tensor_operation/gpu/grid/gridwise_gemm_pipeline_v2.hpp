#pragma once
#include "common_header.hpp"

namespace ck {

// N-stage prefetch
template <index_t NumPrefetch>
struct GridwiseGemmPipeline_v2;

// 1-stage prefetch
template <>
struct GridwiseGemmPipeline_v2<1>
{
    __host__ __device__ static constexpr bool IsSupported(index_t num_loop)
    {
        // TODO: improve applicability
        return num_loop > 2;
    }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return (num_loop / 2) > 1;
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
        // global read 0
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        // move to 1
        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        // LDS write 0
        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
        // global Read 1
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);

        // LDS write 0
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);
        // global Read 1
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            {
                block_sync_lds();

                // GEMM i
                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                block_sync_lds();

                // move to i + 2
                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                // LDS write i + 1
                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
                // global read i + 2
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);

                // LDS write i + 1
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);
                // global read i + 2
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                ++i;
            } while(i < (num_loop - 2));
        }

        // tail
        {
            block_sync_lds();

            // GEMM num_loop - 2
            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

            block_sync_lds();

            // LDS write num_loop - 1
            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
            b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

            block_sync_lds();

            // GEMM num_loop - 1
            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
        }
    }
};


// 2-stage prefetch
template <>
struct GridwiseGemmPipeline_v2<2>
{

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    __host__ __device__ static constexpr bool IsSupported(index_t num_loop)
    {
        // TODO: improve applicability
        return num_loop > 2;
    }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 2;
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
        // global read 0
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I0);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, I0);

        // move to 1
        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        // global read 1
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I1);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, I1);

        // Initialize C
        c_thread_buf.Clear();

        index_t i = 0;

        // main body
        if constexpr(HasMainLoop)
        {
            do
            {
                // move to i + 2
                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                // LDS write i
                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, I0);
                // global Read i + 2
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I0);

                // LDS write i
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, I0);
                // global Read i + 2
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, I0);

                block_sync_lds();

                // GEMM i
                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                block_sync_lds();

                // move to i + 3
                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                // LDS write i + 1
                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, I1);
                // global read i + 3
                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, I1);

                // LDS write i + 1
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, I1);
                // global read i + 3
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, I1);

                block_sync_lds();

                // GEMM i + 1
                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                block_sync_lds();

                i += 2;
            } while(i < (num_loop - 2));
        }

        // tail
        if (i > num_loop - 2)
        {
            // LDS write num_loop - 1
            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, I0);
            b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, I0);

            block_sync_lds();

            // GEMM num_loop - 1
            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
        }

        // tail
        else if (i == num_loop - 2)
        {
            // Write num_loop - 2
            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, I0);
            b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, I0);

            block_sync_lds();

            // GEMM num_loop - 2
            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

            block_sync_lds();

            // LDS write num_loop - 1
            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, I1);
            b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, I1);

            block_sync_lds();

            // GEMM num_loop - 1
            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
        }
        
    }
};


} // namespace ck
