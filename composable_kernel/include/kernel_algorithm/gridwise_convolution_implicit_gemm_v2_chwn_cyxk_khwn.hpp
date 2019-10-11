#ifndef CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V2_CHWN_CYXK_KHWN
#define CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V2_CHWN_CYXK_KHWN

#include "common_header.hpp"
#include "ConstantTensorDescriptor_deprecated.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "blockwise_4d_tensor_op.hpp"
#include "blockwise_2d_tensor_op.hpp"
#include "blockwise_gemm.hpp"

namespace ck {

// define B = flatten(N, Hi, Wi)
template <index_t GridSize,
          index_t BlockSize,
          class Float,
          class InGlobalDesc,
          class WeiGlobalDesc,
          class OutGlobalDesc,
          index_t BPerBlock,
          index_t KPerBlock,
          index_t CPerBlock,
          index_t BPerThread,
          index_t KPerThread,
          index_t GemmMPerThreadSubC,
          index_t GemmNPerThreadSubC,
          index_t GemmMLevel0Cluster,
          index_t GemmNLevel0Cluster,
          index_t GemmMLevel1Cluster,
          index_t GemmNLevel1Cluster,
          index_t GemmKPerThreadLoop,
          index_t GemmDataPerReadA,
          index_t GemmDataPerReadB,
          index_t InBlockCopyThreadPerDim0,
          index_t InBlockCopyThreadPerDim1,
          index_t WeiBlockCopyThreadPerDim0,
          index_t WeiBlockCopyThreadPerDim1,
          index_t InBlockCopyDataPerRead,
          index_t WeiBlockCopyDataPerRead,
          index_t OutThreadCopyDataPerWrite>
struct GridwiseConvolutionImplicitGemm_v2_chwn_cyxk_khwn
{
    __device__ void Run(const Float* const __restrict__ p_in_global,
                        const Float* const __restrict__ p_wei_global,
                        Float* const __restrict__ p_out_global) const
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr auto in_chwn_global_desc  = InGlobalDesc{};
        constexpr auto wei_cyxk_global_desc = WeiGlobalDesc{};
        constexpr auto out_khwn_global_desc = OutGlobalDesc{};

        constexpr index_t C  = in_chwn_global_desc.GetLength(I0);
        constexpr index_t Hi = in_chwn_global_desc.GetLength(I1);
        constexpr index_t Wi = in_chwn_global_desc.GetLength(I2);
        constexpr index_t N  = in_chwn_global_desc.GetLength(I3);

        constexpr index_t K  = out_khwn_global_desc.GetLength(I0);
        constexpr index_t Ho = out_khwn_global_desc.GetLength(I1);
        constexpr index_t Wo = out_khwn_global_desc.GetLength(I2);

        constexpr index_t Y = wei_cyxk_global_desc.GetLength(I1);
        constexpr index_t X = wei_cyxk_global_desc.GetLength(I2);

        constexpr index_t B          = N * Hi * Wi;
        constexpr index_t BGhostRead = (Y - 1) * Wi + (X - 1);

        // divide block work by 2d: [K, B]
        constexpr index_t KBlockWork = (K + KPerBlock - 1) / KPerBlock;
        constexpr index_t BBlockWork = (B + BPerBlock - 1) / BPerBlock;

        const index_t k_block_work_id = get_block_1d_id() / BBlockWork;
        const index_t b_block_work_id = get_block_1d_id() - k_block_work_id * BBlockWork;

        const index_t k_block_data_begin = k_block_work_id * KPerBlock;
        const index_t b_block_data_begin = b_block_work_id * BPerBlock;

        // flattend (2d) tensor view of gridwise input
        constexpr auto in_cb_global_desc  = make_ConstantTensorDescriptor(Sequence<C, B>{});
        constexpr auto wei_ek_global_desc = make_ConstantTensorDescriptor(Sequence<C * Y * X, K>{});

        // tensor view of blockwise input and weight
        //   be careful of alignment
        constexpr auto in_cb_block_desc = make_ConstantTensorDescriptor_aligned(
            Sequence<CPerBlock, BPerBlock + BGhostRead>{}, Number<InBlockCopyDataPerRead>{});

        constexpr auto wei_ek_block_desc = make_ConstantTensorDescriptor_aligned(
            Sequence<CPerBlock * Y * X, KPerBlock>{}, Number<WeiBlockCopyDataPerRead>{});

        constexpr auto wei_cyxk_block_desc = make_ConstantTensorDescriptor_aligned(
            Sequence<CPerBlock, Y, X, KPerBlock>{}, Number<WeiBlockCopyDataPerRead>{});

        // tensor view of threadwise output in register
        constexpr auto out_kb_thread_desc =
            make_ConstantTensorDescriptor(Sequence<KPerThread, BPerThread>{});

// blockwise in copy
//   formmat is [CPerBlock,BPerBlock + BGhostRead]
#if 0
        const auto blockwise_in_copy =
            Blockwise2dTensorCopy1<BlockSize,
                                   Float,
                                   decltype(in_cb_global_desc),
                                   decltype(in_cb_block_desc),
                                   decltype(in_cb_block_desc.GetLengths())>{};
#elif 0
        const auto blockwise_in_copy =
            Blockwise2dTensorCopy2<BlockSize,
                                   Float,
                                   decltype(in_cb_global_desc),
                                   decltype(in_cb_block_desc),
                                   decltype(in_cb_block_desc.GetLengths()),
                                   InBlockCopyThreadPerDim0,
                                   InBlockCopyThreadPerDim1>{};
#elif 1
        const auto blockwise_in_copy =
            Blockwise2dTensorCopy3<BlockSize,
                                   Float,
                                   decltype(in_cb_global_desc),
                                   decltype(in_cb_block_desc),
                                   decltype(in_cb_block_desc.GetLengths()),
                                   InBlockCopyDataPerRead>{};
#endif

// blockwise wei copy
//   format is [CPerBlock*Y*X,KPerBlock]
#if 0
        const auto blockwise_wei_copy =
            Blockwise2dTensorCopy1<BlockSize,
                                   Float,
                                   decltype(wei_ek_global_desc),
                                   decltype(wei_ek_block_desc),
                                   decltype(wei_ek_block_desc.GetLengths())>{};
#elif 0
        const auto blockwise_wei_copy =
            Blockwise2dTensorCopy2<BlockSize,
                                   Float,
                                   decltype(wei_ek_global_desc),
                                   decltype(wei_ek_block_desc),
                                   decltype(wei_ek_block_desc.GetLengths()),
                                   WeiBlockCopyThreadPerDim0,
                                   WeiBlockCopyThreadPerDim1>{};
#elif 1
        const auto blockwise_wei_copy =
            Blockwise2dTensorCopy3<BlockSize,
                                   Float,
                                   decltype(wei_ek_global_desc),
                                   decltype(wei_ek_block_desc),
                                   decltype(wei_ek_block_desc.GetLengths()),
                                   WeiBlockCopyDataPerRead>{};
#endif

        // a series of blockwise GEMM
        // c_mtx += transpose(a_mtx) * b_mtx
        //   a_mtx and b_mtx saved in LDS, c_mtx saved in register
        //   a_mtx[C,K] is a sub-matrix of wei_block[C,Y,X,K]
        //   b_mtx[C,B] is a subset of in_block[C,B + BGhostRead]
        //   c_mtx[K,B] is out_block[K,B]
        constexpr auto a_cxk_block_mtx_desc = make_ConstantMatrixDescriptor(
            Number<CPerBlock>{}, Number<KPerBlock>{}, Number<wei_cyxk_block_desc.GetStride(I0)>{});

        constexpr auto b_cxb_block_mtx_desc = make_ConstantMatrixDescriptor(
            Number<CPerBlock>{}, Number<BPerBlock>{}, Number<in_cb_block_desc.GetStride(I0)>{});

        constexpr auto c_kxb_thread_mtx_desc =
            make_ConstantMatrixDescriptor(Number<KPerThread>{}, Number<BPerThread>{});

        const auto blockwise_gemm =
            BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_v2<BlockSize,
                                                                    decltype(a_cxk_block_mtx_desc),
                                                                    decltype(b_cxb_block_mtx_desc),
                                                                    decltype(c_kxb_thread_mtx_desc),
                                                                    GemmMPerThreadSubC,
                                                                    GemmNPerThreadSubC,
                                                                    GemmMLevel0Cluster,
                                                                    GemmNLevel0Cluster,
                                                                    GemmMLevel1Cluster,
                                                                    GemmNLevel1Cluster,
                                                                    GemmKPerThreadLoop,
                                                                    GemmDataPerReadA,
                                                                    GemmDataPerReadB>{};

        // LDS: be careful of alignment
        constexpr index_t max_align =
            math::lcm(index_t(4), InBlockCopyDataPerRead, WeiBlockCopyDataPerRead);

        constexpr index_t in_block_space = in_cb_block_desc.GetElementSpace(Number<max_align>{});

        constexpr index_t wei_block_space =
            wei_cyxk_block_desc.GetElementSpace(Number<max_align>{});

        __shared__ Float p_in_block[in_block_space];
        __shared__ Float p_wei_block[wei_block_space];

        const Float* p_in_global_block_offset =
            p_in_global + in_cb_global_desc.GetOffsetFromMultiIndex(0, b_block_data_begin);

        const Float* p_wei_global_block_offset =
            p_wei_global +
            wei_cyxk_global_desc.GetOffsetFromMultiIndex(0, 0, 0, k_block_data_begin);

        // register
        Float p_out_thread[out_kb_thread_desc.GetElementSpace()];

        // set threadwise output to 0
        threadwise_matrix_set_zero(c_kxb_thread_mtx_desc, p_out_thread);

        for(index_t c_block_data_begin = 0; c_block_data_begin < C; c_block_data_begin += CPerBlock,
                    p_in_global_block_offset += CPerBlock * in_cb_global_desc.GetStride(I0),
                    p_wei_global_block_offset += CPerBlock * wei_cyxk_global_desc.GetStride(I0),
                    __syncthreads())
        {
            // load data
            Float p_in_register_buffer[blockwise_in_copy.GetRegisterBufferSize()];
            Float p_wei_register_buffer[blockwise_wei_copy.GetRegisterBufferSize()];

            blockwise_in_copy.RunLoadRegisterBuffer(p_in_global_block_offset, p_in_register_buffer);
            blockwise_wei_copy.RunLoadRegisterBuffer(p_wei_global_block_offset,
                                                     p_wei_register_buffer);

            blockwise_in_copy.RunStoreRegisterBuffer(p_in_register_buffer, p_in_block);
            blockwise_wei_copy.RunStoreRegisterBuffer(p_wei_register_buffer, p_wei_block);

            __syncthreads();

            // compute on current data
            //   a series of GEMM
            for(index_t y = 0; y < Y; ++y)
            {
                for(index_t x = 0; x < X; ++x)
                {
#if 1
                    blockwise_gemm.Run
#elif 0
                    blockwise_gemm.Run_RegisterDoubleBuffer
#elif 1
                    blockwise_gemm.Run_amd_asm
#endif
                        (p_wei_block + wei_cyxk_block_desc.GetOffsetFromMultiIndex(0, y, x, 0),
                         p_in_block + y * Wi + x,
                         p_out_thread);
                }
            }
        }

        // output: register to global mem,
        const auto c_thread_mtx_begin =
            blockwise_gemm.GetBeginOfThreadMatrixC(get_thread_local_1d_id());

        const index_t k_thread_data_begin = k_block_data_begin + c_thread_mtx_begin.row;
        const index_t b_thread_data_begin = b_block_data_begin + c_thread_mtx_begin.col;

        for(index_t k = 0; k < out_kb_thread_desc.GetLength(I0); ++k)
        {
            for(index_t b = 0; b < out_kb_thread_desc.GetLength(I1); ++b)
            {
                const auto c_thread_mtx_distance =
                    blockwise_gemm.GetDistanceFromBeginOfThreadMatrixC(k, b);

                index_t k_data = k_thread_data_begin + c_thread_mtx_distance.row;
                index_t b_data = b_thread_data_begin + c_thread_mtx_distance.col;

                index_t h_data = b_data / (Wi * N);
                index_t itmp   = b_data - h_data * (Wi * N);
                index_t w_data = itmp / N;
                index_t n_data = itmp - w_data * N;

                if(n_data < N && h_data < Ho && w_data < Wo)
                {
                    p_out_global[out_khwn_global_desc.GetOffsetFromMultiIndex(
                        k_data, h_data, w_data, n_data)] =
                        p_out_thread[out_kb_thread_desc.GetOffsetFromMultiIndex(k, b)];
                }
            }
        }
    }
};

} // namespace ck
#endif
