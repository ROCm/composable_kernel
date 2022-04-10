#pragma once

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
struct GridwiseGemmPipeline_v2;

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
struct GridwiseGemmPipeline_v2<AGridDesc,
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

    static __device__ void RunProducer(const AGridDesc& a_grid_desc,
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
        // global read 0
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        // move to 1
        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

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

            block_sync_lds();

            // LDS write num_loop - 1
            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf);
            b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

            block_sync_lds();

            // GEMM num_loop - 1
        }
    }

    static __device__ void RunConsumer(ABlockBuffer& a_block_buf,
                                       BBlockBuffer& b_block_buf,
                                       const BlockwiseGemm& blockwise_gemm,
                                       CThreadBuffer& c_thread_buf,
                                       index_t num_loop)
    {
        // Initialize C
        c_thread_buf.Clear();

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
            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

            block_sync_lds();

            // LDS write num_loop - 1

            block_sync_lds();

            // GEMM num_loop - 1
            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);
        }
    }
};

} // namespace ck
