// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/loop_scheduler.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"

namespace ck {

template <index_t NumPrefetch, bool AEnableLds, bool BEnableLds>
struct GridwiseGemmPipeline_v1_nchw;

// 1-stage prefetch
template <>
struct GridwiseGemmPipeline_v1_nchw<1, true, true>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    __host__ __device__ static constexpr bool IsSupported(index_t /* num_loop */) { return true; }

    __host__ __device__ static constexpr bool CalculateHasMainLoop(index_t num_loop)
    {
        return num_loop > 1;
    }

    template <bool HasMainLoop,
              typename AGridDescNCHW,
              typename ABlock1DescNCHWSlice,
              typename ABlock1DescNHOWOCYX,
              typename ABlock2DescK0MK1,
              typename ABlockTransferGlobalIntoLds1,
              typename ABlockTransferLds1IntoLds2,
              typename AGridBuffer,
              typename ABlock1Buffer,
              typename ABlock2Buffer,
              typename ABlockTransferStep,
              typename BGridDesc,
              typename BBlockDesc,
              typename BBlockTransfer,
              typename BGridBuffer,
              typename BBlockBuffer,
              typename BBlockTransferStep,
              typename BlockwiseGemm,
              typename CThreadBuffer>
    __device__ static void Run([[maybe_unused]] const AGridDescNCHW& a_grid_desc,
                               [[maybe_unused]] const ABlock1DescNCHWSlice& a_block1_desc_nchw_slice,
                               [[maybe_unused]] const ABlock1DescNHOWOCYX& a_block1_desc_nhowo_cyx,
                               [[maybe_unused]] const ABlock2DescK0MK1& a_block2_desc_ak0_m_k1,
                               [[maybe_unused]] ABlockTransferGlobalIntoLds1& a_blockwise_global_to_lds1_copy,
                               [[maybe_unused]] ABlockTransferLds1IntoLds2& a_blockwise_lds1_to_lds2_copy,
                               [[maybe_unused]] const AGridBuffer& a_grid_buf,
                               [[maybe_unused]] ABlock1Buffer& a_block1_buf,
                               [[maybe_unused]] ABlock2Buffer& a_block2_buf,
                               [[maybe_unused]] const ABlockTransferStep& a_block_copy_step,
                               [[maybe_unused]] const BGridDesc& b_grid_desc,
                               [[maybe_unused]] const BBlockDesc& b_block_desc,
                               [[maybe_unused]] BBlockTransfer& b_blockwise_copy,
                               [[maybe_unused]] const BGridBuffer& b_grid_buf,
                               [[maybe_unused]] BBlockBuffer& b_block_buf,
                               [[maybe_unused]] const BBlockTransferStep& b_block_copy_step,
                               [[maybe_unused]] const BlockwiseGemm& blockwise_gemm,
                               [[maybe_unused]] CThreadBuffer& c_thread_buf,
                               [[maybe_unused]] index_t num_loop)
    {
        //preload data into LDS1
        a_blockwise_global_to_lds1_copy.RunRead(a_grid_desc, a_grid_buf);
        b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);

        a_blockwise_global_to_lds1_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
        //block_sync_lds();
        a_blockwise_global_to_lds1_copy.RunWrite(a_block1_desc_nchw_slice, a_block1_buf);
        b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);

        //
        c_thread_buf.Clear();

        // LDS1 -> LDS2 broadcast
        block_sync_lds();
        a_blockwise_lds1_to_lds2_copy.RunRead(a_block1_desc_nhowo_cyx, a_block1_buf);//a_block2_buf); ?? jaki bufor
        a_blockwise_lds1_to_lds2_copy.RunWrite(a_block2_desc_ak0_m_k1, a_block2_buf);
        //block_sync_lds();

        // if(threadIdx.x == 0) {
        //     for(int c=0; c<4; ++c) {
        //         for(int h=0; h<2;++h) {
        //             for(int w=0; w<64;++w) {
        //                 printf("Ag[%d][%d][%d]=%f\n", c, h, w, static_cast<float>(a_grid_buf[c * 64 * 2 + h * 64 + w]));
        //             }         
        //         }
        //     }

        //        // A[3][2][33]=0.000000

        //     for(int c=0; c<8; ++c) {
        //         for(int h=0; h<4;++h) {
        //             for(int w=0; w<64;++w) {
        //                 printf("A[%d][%d][%d]=%f\n", c, h, w, static_cast<float>(a_block1_buf[c * 64 * 4 + h * 64 + w]));
        //             }         
        //         }
        //     }

        //     for(int m=0; m<128; ++m) {
        //         for(int k0=0; k0<8; ++k0) {
        //             for(int k1=0;k1<8;++k1) {
        //                 printf("A[%d][%d]=%f\n", m, k0*8+k1, static_cast<float>(a_block2_buf[k0*8*128 + m*8 + k1]));
                    
        //                 int k = k0*8+k1;
        //                 int x = k%3;
        //                 int y = (k/3)%3;
        //                 int c = k/(9);

        //                 // A[m][k] = A[c, y+m/64,x+m%64] z oryginalnego obrazka

        //                 // if(x+m%64 > 0 && x+m%64 < 127) {
        //                 //     auto lds1 = static_cast<float>(a_block1_buf[c * 64 * 4 + (m/64 + y) * 64 + m%64 + x - 1]);
        //                 //     auto lds2 = static_cast<float>(a_block2_buf[k0*8*128 + m*8 + k1]);
        //                 //     printf("lds2[%d][%d]:%f lds1[%d][%d][%d]:%f %s\n", m, k0*8+k1, lds2, c, y + m/64, m%64 + x - 1, lds1, (lds1 > lds2)? "diff" : "");

        //                 // }

        //                     // if(lds1 > lds2) {
        //                     //     printf("diff lds1[%d][%d][%d]:%f lds2[%d][%d]:%f\n", c, y + m/64, m%64 + x, lds1, m, k0*8+k1 - 1, lds2);
        //                     // }
        //             }
        //         }
        //     }
        // }

        // // Initialize C
        


        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;

            do
            { // ask bartek how to structure this pipeline
                a_blockwise_global_to_lds1_copy.RunRead(a_grid_desc, a_grid_buf);  // A Global -> VGPR
                b_blockwise_copy.RunRead(b_grid_desc, b_grid_buf);                 // B Global -> VGPR
                //

                a_blockwise_global_to_lds1_copy.RunWrite(a_block1_desc_nchw_slice, a_block1_buf);  // A VGPR -> LDS1
                blockwise_gemm.Run(a_block2_buf, b_block_buf, c_thread_buf);

                block_sync_lds();
                a_blockwise_lds1_to_lds2_copy.RunRead(a_block1_desc_nhowo_cyx, a_block1_buf); // A LDS1 -> LDS2

                a_blockwise_global_to_lds1_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
                //block_sync_lds();

                a_blockwise_lds1_to_lds2_copy.RunWrite(a_block2_desc_ak0_m_k1, a_block2_buf);
                b_blockwise_copy.RunWrite(b_block_desc, b_block_buf);
                //block_sync_lds();

                ++i;
            } while(i < (num_loop - 1));
        }

        // tail
        {
            block_sync_lds();
            blockwise_gemm.Run(a_block2_buf, b_block_buf, c_thread_buf);
        }
    }
};

// TODO: deprecate as GridwiseGemmPipeline_Selector covers the functionality
template <index_t NumPrefetch, LoopScheduler LoopSched>
constexpr auto GridwiseGemmPipeline_v1_nchw_Selector()
{
    return GridwiseGemmPipeline_v1<NumPrefetch, true, true>{};

}

}  // namespace ck
