#ifndef CK_GRIDWISE_CONVOLUTION_DIRECT_V2_NCHW_KCYX_NKHW
#define CK_GRIDWISE_CONVOLUTION_DIRECT_V2_NCHW_KCYX_NKHW

#include "common_header.hpp"
#include "ConstantTensorDescriptor_deprecated.hpp"
#include "blockwise_2d_tensor_op.hpp"
#include "blockwise_4d_tensor_op.hpp"
#include "threadwise_tensor_slice_copy.hpp"
#include "threadwise_direct_convolution.hpp"

namespace ck {

template <index_t GridSize,
          index_t BlockSize,
          class Float,
          class InGlobalDesc,
          class WeiGlobalDesc,
          class OutGlobalDesc,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t CPerBlock,
          index_t HoPerBlock,
          index_t WoPerBlock,
          index_t NPerThread,
          index_t KPerThread,
          index_t CPerThread,
          index_t HoPerThread,
          index_t WoPerThread,
          index_t InBlockCopyDataPerRead,
          index_t WeiBlockCopyDataPerRead>
struct GridwiseConvolutionDirect_v2_nchw_kcyx_nkhw
{
    __device__ void Run(const Float* const __restrict__ p_in_global,
                        const Float* const __restrict__ p_wei_global,
                        Float* const __restrict__ p_out_global)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr auto in_nchw_global_desc  = InGlobalDesc{};
        constexpr auto wei_kcyx_global_desc = WeiGlobalDesc{};
        constexpr auto out_nkhw_global_desc = OutGlobalDesc{};

        constexpr index_t N = in_nchw_global_desc.GetLength(I0);
        constexpr index_t K = wei_kcyx_global_desc.GetLength(I0);
        constexpr index_t C = wei_kcyx_global_desc.GetLength(I1);
        constexpr index_t Y = wei_kcyx_global_desc.GetLength(I2);
        constexpr index_t X = wei_kcyx_global_desc.GetLength(I3);

        constexpr auto wei_ke_global_desc = make_ConstantTensorDescriptor_packed(
            Sequence<K, C * Y * X>{}); // 2d view of wei for blockwise copy

        constexpr index_t HiPerBlock = HoPerBlock + Y - 1;
        constexpr index_t WiPerBlock = WoPerBlock + X - 1;

        constexpr auto in_nchw_block_desc = make_ConstantTensorDescriptor_aligned(
            Sequence<NPerBlock, CPerBlock, HiPerBlock, WiPerBlock>{},
            Number<InBlockCopyDataPerRead>{});

        constexpr auto wei_ke_block_desc = make_ConstantTensorDescriptor_aligned(
            Sequence<KPerBlock, CPerBlock * Y * X>{},
            Number<WeiBlockCopyDataPerRead>{}); // 2d view of wei for blockwise copy

        constexpr auto wei_kcyx_block_desc =
            make_ConstantTensorDescriptor(Sequence<KPerBlock, CPerBlock, Y, X>{},
                                          Sequence<wei_ke_block_desc.GetStride(I0), Y * X, X, 1>{});

        // shared mem
        constexpr index_t in_block_element_size =
            in_nchw_block_desc.GetElementSpace(Number<InBlockCopyDataPerRead>{});
        constexpr index_t wei_block_element_size =
            wei_kcyx_block_desc.GetElementSpace(Number<WeiBlockCopyDataPerRead>{});

        constexpr index_t max_align = InBlockCopyDataPerRead > WeiBlockCopyDataPerRead
                                          ? InBlockCopyDataPerRead
                                          : WeiBlockCopyDataPerRead;

        __shared__ Float
            p_in_block[max_align * ((in_block_element_size + max_align - 1) / max_align)];
        __shared__ Float
            p_wei_block[max_align * ((wei_block_element_size + max_align - 1) / max_align)];

        // threadwise tensors
        constexpr index_t HiPerThread = HoPerThread + Y - 1;
        constexpr index_t WiPerThread = WoPerThread + X - 1;

        constexpr auto in_nchw_thread_block_desc = make_ConstantTensorDescriptor(
            Sequence<NPerThread, CPerThread, HiPerThread, WiPerThread>{},
            in_nchw_block_desc.GetStrides());

        constexpr auto wei_kcyx_thread_block_desc = make_ConstantTensorDescriptor(
            Sequence<KPerThread, CPerThread, Y, X>{}, wei_kcyx_block_desc.GetStrides());

        constexpr auto out_nkhw_thread_desc = get_convolution_output_default_4d_tensor_descriptor(
            in_nchw_thread_block_desc, wei_kcyx_thread_block_desc);

        // register
        Float p_out_thread[out_nkhw_thread_desc.GetElementSpace()];

        // divide block work
        constexpr index_t NBlockWork =
            (out_nkhw_global_desc.GetLength(I0) + NPerBlock - 1) / NPerBlock;
        constexpr index_t KBlockWork =
            (out_nkhw_global_desc.GetLength(I1) + KPerBlock - 1) / KPerBlock;
        constexpr index_t HBlockWork =
            (out_nkhw_global_desc.GetLength(I2) + HoPerBlock - 1) / HoPerBlock;
        constexpr index_t WBlockWork =
            (out_nkhw_global_desc.GetLength(I3) + WoPerBlock - 1) / WoPerBlock;

        const index_t block_id = blockIdx.x;

        index_t itmp                  = block_id;
        const index_t n_block_work_id = itmp / (KBlockWork * HBlockWork * WBlockWork);
        itmp -= n_block_work_id * (KBlockWork * HBlockWork * WBlockWork);
        const index_t k_block_work_id = itmp / (HBlockWork * WBlockWork);
        itmp -= k_block_work_id * (HBlockWork * WBlockWork);
        const index_t h_block_work_id = itmp / WBlockWork;
        const index_t w_block_work_id = itmp - h_block_work_id * WBlockWork;

        const index_t n_block_data_begin  = n_block_work_id * NPerBlock;
        const index_t k_block_data_begin  = k_block_work_id * KPerBlock;
        const index_t ho_block_data_begin = h_block_work_id * HoPerBlock;
        const index_t wo_block_data_begin = w_block_work_id * WoPerBlock;

        const index_t hi_block_data_begin = ho_block_data_begin; // minus padding
        const index_t wi_block_data_begin = wo_block_data_begin; // minus padding

        // divide thread work
        constexpr index_t NThreadWork = (NPerBlock + NPerThread - 1) / NPerThread;
        constexpr index_t KThreadWork = (KPerBlock + KPerThread - 1) / KPerThread;
        constexpr index_t HThreadWork = (HoPerBlock + HoPerThread - 1) / HoPerThread;
        constexpr index_t WThreadWork = (WoPerBlock + WoPerThread - 1) / WoPerThread;

        const index_t thread_id = get_thread_local_1d_id();

        itmp                           = thread_id;
        const index_t n_thread_work_id = itmp / (KThreadWork * HThreadWork * WThreadWork);
        itmp -= n_thread_work_id * (KThreadWork * HThreadWork * WThreadWork);
        const index_t k_thread_work_id = itmp / (HThreadWork * WThreadWork);
        itmp -= k_thread_work_id * (HThreadWork * WThreadWork);
        const index_t h_thread_work_id = itmp / WThreadWork;
        const index_t w_thread_work_id = itmp - h_thread_work_id * WThreadWork;

        const index_t n_thread_data_begin  = n_thread_work_id * NPerThread;
        const index_t k_thread_data_begin  = k_thread_work_id * KPerThread;
        const index_t ho_thread_data_begin = h_thread_work_id * HoPerThread;
        const index_t wo_thread_data_begin = w_thread_work_id * WoPerThread;

        const index_t hi_thread_data_begin = ho_thread_data_begin;
        const index_t wi_thread_data_begin = wo_thread_data_begin;

        constexpr auto blockwise_in_copy =
            Blockwise4dTensorCopy1<BlockSize,
                                   Float,
                                   decltype(in_nchw_global_desc),
                                   decltype(in_nchw_block_desc),
                                   decltype(in_nchw_block_desc.GetLengths()),
                                   InBlockCopyDataPerRead>{};

#if 0
    constexpr auto blockwise_wei_copy =
        Blockwise4dTensorCopy1<BlockSize,
                               Float,
                               decltype(wei_kcyx_global_desc),
                               decltype(wei_kcyx_block_desc),
                               decltype(wei_kcyx_block_desc.GetLengths()),
                               1>{};
#elif 1
        const auto blockwise_wei_copy =
            Blockwise2dTensorCopy3<BlockSize,
                                   Float,
                                   decltype(wei_ke_global_desc),
                                   decltype(wei_ke_block_desc),
                                   decltype(wei_ke_block_desc.GetLengths()),
                                   WeiBlockCopyDataPerRead>({0, 0}, {0, 0});
#endif

        // set threadwise output tensor to 0
        threadwise_4d_tensor_set_zero(out_nkhw_thread_desc, p_out_thread);

        for(index_t c_block_data_begin = 0; c_block_data_begin < C;
            c_block_data_begin += CPerBlock, __syncthreads())
        {
            // copy input tensor to LDS
            blockwise_in_copy.Run(
                p_in_global +
                    in_nchw_global_desc.GetOffsetFromMultiIndex(n_block_data_begin,
                                                                c_block_data_begin,
                                                                hi_block_data_begin,
                                                                wi_block_data_begin),
                p_in_block);

            // copy weight tensor to LDS
            blockwise_wei_copy.Run(p_wei_global +
                                       wei_kcyx_global_desc.GetOffsetFromMultiIndex(
                                           k_block_data_begin, c_block_data_begin, 0, 0),
                                   p_wei_block);

            __syncthreads();

            for(index_t c_thread_data = 0; c_thread_data < CPerBlock; c_thread_data += CPerThread)
            {
// threadwise convolution
#if 1
                threadwise_direct_convolution_2(
                    in_nchw_thread_block_desc,
                    p_in_block +
                        in_nchw_block_desc.GetOffsetFromMultiIndex(n_thread_data_begin,
                                                                   c_thread_data,
                                                                   hi_thread_data_begin,
                                                                   wi_thread_data_begin),
                    wei_kcyx_thread_block_desc,
                    p_wei_block +
                        wei_kcyx_block_desc.GetOffsetFromMultiIndex(
                            k_thread_data_begin, c_thread_data, 0, 0),
                    out_nkhw_thread_desc,
                    p_out_thread);
#elif 0
                threadwise_direct_convolution_3(
                    in_nchw_thread_block_desc,
                    p_in_block +
                        in_nchw_block_desc.GetOffsetFromMultiIndex(n_thread_data_begin,
                                                                   c_thread_data,
                                                                   hi_thread_data_begin,
                                                                   wi_thread_data_begin),
                    wei_kcyx_thread_block_desc,
                    p_wei_block +
                        wei_kcyx_block_desc.GetOffsetFromMultiIndex(
                            k_thread_data_begin, c_thread_data, 0, 0),
                    out_nkhw_thread_desc,
                    p_out_thread);
#endif
            }
        }

        // copy output tensor from register to global mem
        threadwise_tensor_slice_copy(out_nkhw_thread_desc,
                                     p_out_thread,
                                     out_nkhw_global_desc,
                                     p_out_global +
                                         out_nkhw_global_desc.GetOffsetFromMultiIndex(
                                             n_block_data_begin + n_thread_data_begin,
                                             k_block_data_begin + k_thread_data_begin,
                                             ho_block_data_begin + ho_thread_data_begin,
                                             wo_block_data_begin + wo_thread_data_begin),
                                     out_nkhw_thread_desc.GetLengths(),
                                     Number<1>{});
    }
};

} // namespace ck
#endif
