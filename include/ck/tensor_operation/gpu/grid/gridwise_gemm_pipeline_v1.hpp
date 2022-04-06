#ifndef CK_GRIDWISE_GEMM_PIPELINE_V1_HPP
#define CK_GRIDWISE_GEMM_PIPELINE_V1_HPP

#include "common_header.hpp"

namespace ck {

template <typename AGridDesc,
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
          typename CThreadBuffer,
          index_t NumPrefetch,
          bool HasMainLoop>
struct GridwiseGemmPipeline_v1;

// 1-stage prefetch
template <typename AGridDesc,
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
          typename CThreadBuffer,
          bool HasMainLoop>
struct GridwiseGemmPipeline_v1<AGridDesc,
                               ABlockDesc,
                               ABlockTransfer,
                               AGridBuffer,
                               ABlockBuffer,
                               ABlockTransferStep,
                               BGridDesc,
                               BBlockDesc,
                               BBlockTransfer,
                               BGridBuffer,
                               BBlockBuffer,
                               BBlockTransferStep,
                               BlockwiseGemm,
                               CThreadBuffer,
                               1,
                               HasMainLoop>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

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
#if 0
        // preload data into LDS
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

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
                a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);

                block_sync_lds();

                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                block_sync_lds();

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
#elif 0
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
#elif 1
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
        // FIXME: HasMainLoop = (num_loop) > 2
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
#endif
    }
};

// 2-stage prefetch
template <typename AGridDesc,
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
          typename CThreadBuffer,
          bool HasMainLoop>
struct GridwiseGemmPipeline_v1<AGridDesc,
                               ABlockDesc,
                               ABlockTransfer,
                               AGridBuffer,
                               ABlockBuffer,
                               ABlockTransferStep,
                               BGridDesc,
                               BBlockDesc,
                               BBlockTransfer,
                               BGridBuffer,
                               BBlockBuffer,
                               BBlockTransferStep,
                               BlockwiseGemm,
                               CThreadBuffer,
                               2,
                               HasMainLoop>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

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
};

} // namespace ck
#endif
