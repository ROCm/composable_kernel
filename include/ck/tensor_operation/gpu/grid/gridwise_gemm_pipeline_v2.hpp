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

// 3-stage prefetch
template <>
struct GridwiseGemmPipeline_v2<3>
{

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};

    __host__ __device__ static constexpr bool IsSupported(index_t num_loop)
    {
        // TODO: improve applicability
        return num_loop > 3;
    }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 3;
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
        static_for<0, 3, 1>{}([&](auto i_pre){
            // global read i_pre
            a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, Number<i_pre>{});
            b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, Number<i_pre>{});

            // move to i_pre + 1
            a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
            b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
        });
        
        // Initialize C
        c_thread_buf.Clear();

        index_t i = 0;

        // main body
        if constexpr(HasMainLoop)
        {
            do
            {
                static_for<0, 3, 1>{}([&](auto i_main){

                    // LDS write i_main
                    a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, Number<i_main>{});
                    // global Read i_main + 3
                    a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, Number<i_main>{});

                    // LDS write i_main
                    b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, Number<i_main>{});
                    // global Read i_main + 3
                    b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, Number<i_main>{});

                    // move to i_main + 3
                    a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                    b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                    block_sync_lds();

                    // GEMM i_main
                    blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                    block_sync_lds();

                });
                
                i += 3;
            } while(i < (num_loop - 3));
        }

        // tail
        if (i == num_loop - 3)
        {
            static_for<0, I3, 1>{}([&](auto i_res){

                // Write num_loop - 3
                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, Number<i_res>{});
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, Number<i_res>{});

                block_sync_lds();

                // GEMM num_loop - 3
                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                block_sync_lds();
            });
        }

        // tail
        else if (i == num_loop - 2)
        {
            static_for<0, I2, 1>{}([&](auto i_res){

                // Write num_loop
                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, Number<i_res>{});
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, Number<i_res>{});

                block_sync_lds();

                // GEMM num_loop
                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                block_sync_lds();
            });
        }

        // tail
        else if (i == num_loop - 1)
        {
            static_for<0, I1, 1>{}([&](auto i_res){

                // Write num_loop
                a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, Number<i_res>{});
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, Number<i_res>{});

                block_sync_lds();

                // GEMM num_loop
                blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                block_sync_lds();
            });
        }
        
    }
};

// 4-stage prefetch
template <>
struct GridwiseGemmPipeline_v2<4>
{

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};

    __host__ __device__ static constexpr bool IsSupported(index_t num_loop)
    {
        // TODO: improve applicability
        return num_loop % 4 == 0;
    }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop / 4 > 1;
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
        static_for<0, 4, 1>{}([&](auto i_pre){
            // global read i_pre
            a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, Number<i_pre>{});
            b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, Number<i_pre>{});

            s_nop();

            // move to i_pre + 1
            a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
            b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

            s_nop();
        });
        // Initialize C
        c_thread_buf.Clear();

        index_t i = 0;

        // main body
        if constexpr(HasMainLoop)
        {
            do
            {
                static_for<0, 4, 1>{}([&](auto i_main){

                    // LDS write i_main
                    a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, Number<i_main>{});
                    // global Read i_main + 3
                    a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf, Number<i_main>{});

                    // LDS write i_main
                    b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, Number<i_main>{});
                    // global Read i_main + 3
                    b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf, Number<i_main>{});

                    // move to i_main + 3
                    a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                    b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

                    block_sync_lds();
                    s_nop();

                    // GEMM i_main
                    blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

                    block_sync_lds();
                    s_nop();

                });
                
                i += 4;
            } while(i < (num_loop - 4));
        }

        // tail
        static_for<0, I4, 1>{}([&](auto i_res){

            // Write num_loop - 3
            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf, Number<i_res>{});
            b_blockwise_copy.RunWrite(b_block_desc, b_block_buf, Number<i_res>{});

            block_sync_lds();
            s_nop();

            // GEMM num_loop - 3
            blockwise_gemm.Run(a_block_buf, b_block_buf, c_thread_buf);

            block_sync_lds();
            s_nop();
        });
        
    }
};

} // namespace ck
