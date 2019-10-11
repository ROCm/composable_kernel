#ifndef CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V1R3_CHWN_CYXK_KHWN_PADDED_HPP
#define CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V1R3_CHWN_CYXK_KHWN_PADDED_HPP

#include "common_header.hpp"
#include "ConstantTensorDescriptor_deprecated.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "blockwise_generic_tensor_slice_copy.hpp"
#include "threadwise_generic_tensor_slice_copy.hpp"
#include "blockwise_batched_gemm.hpp"

namespace ck {

template <index_t GridSize,
          index_t BlockSize,
          class Float,
          class InGlobalDesc,
          class WeiGlobalDesc,
          class OutGlobalDesc,
          class LeftPads,
          class RightPads,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t CPerBlock,
          index_t HoPerBlock,
          index_t WoPerBlock,
          index_t NPerThread,
          index_t KPerThread,
          index_t HoPerThread,
          index_t WoPerThread,
          index_t GemmMPerThreadSubC,
          index_t GemmNPerThreadSubC,
          index_t GemmMLevel0Cluster,
          index_t GemmNLevel0Cluster,
          index_t GemmMLevel1Cluster,
          index_t GemmNLevel1Cluster,
          index_t GemmKPerThreadLoop,
          index_t GemmDataPerReadA,
          index_t GemmDataPerReadB,
          class InBlockCopySubLengths_CHWN,
          class InBlockCopyClusterLengths_CHWN,
          index_t InBlockCopyDataPerAccess_N,
          class WeiBlockCopySubLengths_CK,
          class WeiBlockCopyClusterLengths_CK,
          index_t WeiBlockCopyDataPerAccess_K,
          index_t OutThreadCopyDataPerAccess_N>
struct GridwiseConvolutionImplicitGemm_v1r3_chwn_cyxk_khwn_padded
{
    __device__ void Run(const Float* const __restrict__ p_in_global,
                        const Float* const __restrict__ p_wei_global,
                        Float* const __restrict__ p_out_global) const
    {
        static constexpr auto I0 = Number<0>{};
        static constexpr auto I1 = Number<1>{};
        static constexpr auto I2 = Number<2>{};
        static constexpr auto I3 = Number<3>{};

        static constexpr auto True  = integral_constant<bool, true>{};
        static constexpr auto False = integral_constant<bool, false>{};

        // be careful of this assertion
        static_assert(
            NPerBlock % NPerThread == 0 &&
                ((GemmNPerThreadSubC <= NPerBlock && NPerBlock % GemmNPerThreadSubC == 0) ||
                 (GemmNPerThreadSubC >= NPerBlock && NPerThread == NPerBlock &&
                  GemmNPerThreadSubC % NPerThread == 0)),
            "wrong!");

        constexpr auto in_c_h_w_n_global_desc_old  = InGlobalDesc{};
        constexpr auto wei_c_y_x_k_global_desc_old = WeiGlobalDesc{};
        constexpr auto out_k_h_w_n_global_desc_old = OutGlobalDesc{};

        constexpr auto in_c_h_w_n_global_desc = make_native_tensor_descriptor(
            in_c_h_w_n_global_desc_old.GetLengths(), in_c_h_w_n_global_desc_old.GetStrides());

        constexpr auto wei_c_y_x_k_global_desc = make_native_tensor_descriptor(
            wei_c_y_x_k_global_desc_old.GetLengths(), wei_c_y_x_k_global_desc_old.GetStrides());

        constexpr auto out_k_h_w_n_global_desc = make_native_tensor_descriptor(
            out_k_h_w_n_global_desc_old.GetLengths(), out_k_h_w_n_global_desc_old.GetStrides());

        constexpr index_t C  = in_c_h_w_n_global_desc.GetLength(I0);
        constexpr index_t Hi = in_c_h_w_n_global_desc.GetLength(I1);
        constexpr index_t Wi = in_c_h_w_n_global_desc.GetLength(I2);

        constexpr index_t K  = out_k_h_w_n_global_desc.GetLength(I0);
        constexpr index_t Ho = out_k_h_w_n_global_desc.GetLength(I1);
        constexpr index_t Wo = out_k_h_w_n_global_desc.GetLength(I2);
        constexpr index_t N  = out_k_h_w_n_global_desc.GetLength(I3);

        constexpr index_t Y = wei_c_y_x_k_global_desc.GetLength(I1);
        constexpr index_t X = wei_c_y_x_k_global_desc.GetLength(I2);

        // divide block work: [K, Ho, Wo, N]
        static_assert(N % NPerBlock == 0 && K % KPerBlock == 0 && C % CPerBlock == 0 &&
                          Ho % HoPerBlock == 0 && Wo % WoPerBlock == 0,
                      "wrong! cannot evenly divide work for workgroup ");

        constexpr index_t KBlockWork = math::integer_divide_ceil(K, KPerBlock);
        constexpr index_t HBlockWork = math::integer_divide_ceil(Ho, HoPerBlock);
        constexpr index_t WBlockWork = math::integer_divide_ceil(Wo, WoPerBlock);
        constexpr index_t NBlockWork = math::integer_divide_ceil(N, NPerBlock);

        constexpr auto block_work_desc = make_ConstantTensorDescriptor_packed(
            Sequence<KBlockWork, HBlockWork, WBlockWork, NBlockWork>{});

        const auto block_work_multi_id =
            block_work_desc.GetMultiIndexFrom1dIndex(get_block_1d_id());

        const index_t k_block_data_begin  = block_work_multi_id[0] * KPerBlock;
        const index_t ho_block_data_begin = block_work_multi_id[1] * HoPerBlock;
        const index_t wo_block_data_begin = block_work_multi_id[2] * WoPerBlock;
        const index_t n_block_data_begin  = block_work_multi_id[3] * NPerBlock;

        const index_t hp_block_data_begin = ho_block_data_begin;
        const index_t wp_block_data_begin = wo_block_data_begin;

        // input global tensor view
        constexpr auto in_c_hp_wp_n_global_desc = transform_tensor_descriptor(
            in_c_h_w_n_global_desc,
            make_tuple(
                PassThrough<C>{}, Pad<Sequence<Hi, Wi>, LeftPads, RightPads>{}, PassThrough<N>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

        // LDS tensor view
        //   be careful of alignment
        constexpr index_t max_align = math::lcm(InBlockCopyDataPerAccess_N,
                                                WeiBlockCopyDataPerAccess_K,
                                                GemmDataPerReadA,
                                                GemmDataPerReadB);

        constexpr auto in_c_h_w_n_block_desc_old = make_ConstantTensorDescriptor_aligned(
            Sequence<CPerBlock, HoPerBlock, WoPerBlock, NPerBlock>{}, Number<max_align>{});

        // hack
        constexpr auto in_c_h_w_n_block_desc = make_native_tensor_descriptor(
            in_c_h_w_n_block_desc_old.GetLengths(), in_c_h_w_n_block_desc_old.GetStrides());

        // this check is ad-hoc
        // TODO: need to properly implement tensor descriptor with alignment
        static_assert(in_c_h_w_n_block_desc.GetStride(I1) % GemmDataPerReadB == 0,
                      "GemmDataPerReadB alignment requirement is not meet");

        constexpr auto wei_c_1_1_k_block_desc_old = make_ConstantTensorDescriptor_aligned(
            Sequence<CPerBlock, 1, 1, KPerBlock>{}, Number<max_align>{});

        constexpr auto wei_c_1_1_k_block_desc = make_native_tensor_descriptor(
            wei_c_1_1_k_block_desc_old.GetLengths(), wei_c_1_1_k_block_desc_old.GetStrides());

        // LDS: be careful of alignment
        constexpr index_t in_block_space  = in_c_h_w_n_block_desc_old.GetElementSpace();
        constexpr index_t wei_block_space = wei_c_1_1_k_block_desc_old.GetElementSpace();

        __shared__ Float p_in_block[in_block_space];
        __shared__ Float p_wei_block[wei_block_space];

        // tensor view of threadwise output in register
        constexpr auto out_k_h_w_n_thread_desc_old = make_ConstantTensorDescriptor_packed(
            Sequence<KPerThread, HoPerThread, WoPerThread, NPerThread>{});

        constexpr auto out_k_h_w_n_thread_desc = make_native_tensor_descriptor(
            out_k_h_w_n_thread_desc_old.GetLengths(), out_k_h_w_n_thread_desc_old.GetStrides());

        // blockwise input copy
        //   format is [C, Hi, Wi, N]
        auto blockwise_in_copy =
            BlockwiseGenericTensorSliceCopy_v4<BlockSize,
                                               decltype(in_c_hp_wp_n_global_desc),
                                               decltype(in_c_h_w_n_block_desc),
                                               decltype(in_c_h_w_n_block_desc.GetLengths()),
                                               InBlockCopySubLengths_CHWN,
                                               InBlockCopyClusterLengths_CHWN,
                                               Sequence<0, 1, 2, 3>,
                                               Sequence<0, 1, 2, 3>,
                                               Sequence<0, 1, 2, 3>,
                                               3,
                                               3,
                                               InBlockCopyDataPerAccess_N,
                                               InBlockCopyDataPerAccess_N>(
                {0, hp_block_data_begin, wp_block_data_begin, n_block_data_begin}, {0, 0, 0, 0});

        // blockwise wei copy
        //   format is [CPerBlock, KPerBlock]
        using WeiBlockCopySubLengths_CYXK =
            Sequence<WeiBlockCopySubLengths_CK::At(0), 1, 1, WeiBlockCopySubLengths_CK::At(1)>;
        using WeiBlockCopyClusterLengths_CYXK = Sequence<WeiBlockCopyClusterLengths_CK::At(0),
                                                         1,
                                                         1,
                                                         WeiBlockCopyClusterLengths_CK::At(1)>;

        auto blockwise_wei_copy =
            BlockwiseGenericTensorSliceCopy_v4<BlockSize,
                                               decltype(wei_c_y_x_k_global_desc),
                                               decltype(wei_c_1_1_k_block_desc),
                                               decltype(wei_c_1_1_k_block_desc.GetLengths()),
                                               WeiBlockCopySubLengths_CYXK,
                                               WeiBlockCopyClusterLengths_CYXK,
                                               Sequence<0, 1, 2, 3>,
                                               Sequence<0, 1, 2, 3>,
                                               Sequence<0, 1, 2, 3>,
                                               3,
                                               3,
                                               WeiBlockCopyDataPerAccess_K,
                                               WeiBlockCopyDataPerAccess_K>(
                {0, 0, 0, k_block_data_begin}, {0, 0, 0, 0});

        // a series of blockwise batched GEMM
        // C_matrix += transpose(A_matrix) * B_matrix
        //   A_matrix and B_matrix saved in LDS, C_matrix saved in register
        //   A_matrix[C,K] is a sub-matrix of wei_block[C,K]
        //   B_matrix[C,Wo*N] is a sub-matrix of in_block[C,Hi,Wi,N]
        //   C_matrix[K,Wo*N] is a sub-matrix of out_block[K,Ho,Wo,N]
        constexpr auto a_c_k_block_mtx_desc =
            make_ConstantMatrixDescriptor(Number<CPerBlock>{},
                                          Number<KPerBlock>{},
                                          Number<wei_c_1_1_k_block_desc.GetStride(I0)>{});

        constexpr auto b_c_wn_block_mtx_desc =
            make_ConstantMatrixDescriptor(Number<CPerBlock>{},
                                          Number<WoPerBlock * NPerBlock>{},
                                          Number<in_c_h_w_n_block_desc.GetStride(I0)>{});

        constexpr auto c_k_wn_thread_mtx_desc =
            make_ConstantMatrixDescriptor(Number<KPerThread>{},
                                          Number<WoPerThread * NPerThread>{},
                                          Number<out_k_h_w_n_thread_desc.GetStride(I0)>{});

        const auto blockwise_batch_gemm =
            BlockwiseBatchGemmBlockABlockBThreadCTransANormalBNormalC_V2<
                BlockSize,
                decltype(a_c_k_block_mtx_desc),
                decltype(b_c_wn_block_mtx_desc),
                decltype(c_k_wn_thread_mtx_desc),
                0,
                in_c_h_w_n_block_desc.GetStride(I1),
                out_k_h_w_n_thread_desc.GetStride(I1),
                HoPerBlock,
                GemmMPerThreadSubC,
                GemmNPerThreadSubC,
                GemmMLevel0Cluster,
                GemmNLevel0Cluster,
                GemmMLevel1Cluster,
                GemmNLevel1Cluster,
                GemmKPerThreadLoop,
                HoPerThread,
                GemmDataPerReadA,
                GemmDataPerReadB>{};

        // register
        // C++ lambda doesn't capture array, use pointer instead
        Float p_out_thread_data[out_k_h_w_n_thread_desc_old.GetElementSpace()];
        Float* const p_out_thread = p_out_thread_data;

        // set threadwise output tensor to 0
        threadwise_matrix_set_zero(c_k_wn_thread_mtx_desc, p_out_thread);

        for(index_t y = 0; y < Y; ++y)
        {
            for(index_t x = 0; x < X; ++x)
            {
                for(index_t c_block_data_begin = 0; c_block_data_begin < C;
                    c_block_data_begin += CPerBlock)
                {
                    blockwise_in_copy.Run(p_in_global, p_in_block);
                    blockwise_wei_copy.Run(p_wei_global, p_wei_block);

                    __syncthreads();

                    blockwise_batch_gemm.Run(p_wei_block, p_in_block, p_out_thread);

                    __syncthreads();

                    // move along C
                    blockwise_in_copy.MoveSrcSliceWindow(make_multi_index(CPerBlock, 0, 0, 0),
                                                         True);
                    blockwise_wei_copy.MoveSrcSliceWindow(make_multi_index(CPerBlock, 0, 0, 0),
                                                          True);
                }

                // reset C
                blockwise_in_copy.MoveSrcSliceWindow(make_multi_index(C, 0, 0, 0), False);
                blockwise_wei_copy.MoveSrcSliceWindow(make_multi_index(C, 0, 0, 0), False);

                // move aling X
                blockwise_in_copy.MoveSrcSliceWindow(make_multi_index(0, 0, 1, 0), True);
                blockwise_wei_copy.MoveSrcSliceWindow(make_multi_index(0, 0, 1, 0), True);
            }

            // reset X
            blockwise_in_copy.MoveSrcSliceWindow(make_multi_index(0, 0, X, 0), False);
            blockwise_wei_copy.MoveSrcSliceWindow(make_multi_index(0, 0, X, 0), False);

            // move along Y
            blockwise_in_copy.MoveSrcSliceWindow(make_multi_index(0, 1, 0, 0), True);
            blockwise_wei_copy.MoveSrcSliceWindow(make_multi_index(0, 1, 0, 0), True);
        }

        // output: register to global mem
        const auto c_thread_mtx_begin =
            blockwise_batch_gemm.GetBeginOfThreadMatrixC(get_thread_local_1d_id());

        const index_t k_thread_data_begin  = c_thread_mtx_begin.row;
        const index_t ho_thread_data_begin = c_thread_mtx_begin.batch;
        const index_t wo_thread_data_begin = c_thread_mtx_begin.col / NPerBlock;
        const index_t n_thread_data_begin  = c_thread_mtx_begin.col % NPerBlock;

        static_if<GemmNPerThreadSubC <= NPerBlock>{}([&](auto fwd) {
            // fwd do nothing but perfect forwarding.
            // Using this trick to make this lambda a generic lambda, so it won't be compiled until
            // being instantiated here
            static_assert(
                (fwd(GemmNPerThreadSubC) <= NPerBlock && NPerBlock % GemmNPerThreadSubC == 0),
                "wrong!");

            // output is a 10d tensor
            constexpr index_t N2 = GemmNPerThreadSubC;
            constexpr index_t N1 = NPerBlock / N2;

            constexpr index_t W2 =
                (GemmNLevel0Cluster * GemmNLevel1Cluster) / fwd(NPerBlock / GemmNPerThreadSubC);
            constexpr index_t W1 = WoPerBlock / W2;

            constexpr index_t K2 = GemmMPerThreadSubC;
            constexpr index_t K1 = KPerBlock / KPerThread;

            constexpr auto out_10d_global_desc_old = fwd(out_k_h_w_n_global_desc_old)
                                                         .Fold(I3, Number<N1>{}, Number<N2>{})
                                                         .Fold(I2, Number<W1>{}, Number<W2>{})
                                                         .Fold(I0, Number<K1>{}, Number<K2>{});

            constexpr auto out_10d_global_desc = make_native_tensor_descriptor(
                out_10d_global_desc_old.GetLengths(), out_10d_global_desc_old.GetStrides());

            constexpr auto out_10d_thread_desc_old = fwd(out_k_h_w_n_thread_desc_old)
                                                         .Fold(I3, Number<1>{}, Number<N2>{})
                                                         .Fold(I2, Number<W1>{}, Number<1>{})
                                                         .Fold(I0, Number<1>{}, Number<K2>{});

            constexpr auto out_10d_thread_desc = make_native_tensor_descriptor(
                out_10d_thread_desc_old.GetLengths(), out_10d_thread_desc_old.GetStrides());

            Float* p_out_thread_on_global =
                p_out_global +
                out_k_h_w_n_global_desc.CalculateOffset({k_block_data_begin + k_thread_data_begin,
                                                         ho_block_data_begin + ho_thread_data_begin,
                                                         wo_block_data_begin + wo_thread_data_begin,
                                                         n_block_data_begin + n_thread_data_begin});

            ThreadwiseGenericTensorSliceCopy_v4r2<decltype(out_10d_thread_desc),
                                                  decltype(out_10d_global_desc),
                                                  decltype(out_10d_thread_desc.GetLengths()),
                                                  arithmetic_sequence_gen<0, 10, 1>::type,
                                                  9,
                                                  OutThreadCopyDataPerAccess_N,
                                                  OutThreadCopyDataPerAccess_N>(
                make_zero_array<index_t, 10>(), make_zero_array<index_t, 10>())
                .Run(p_out_thread, p_out_thread_on_global);
        }).Else([&](auto fwd) {
            static_assert(fwd(GemmNPerThreadSubC) >= NPerBlock && NPerThread == NPerBlock &&
                              GemmNPerThreadSubC % NPerThread == 0,
                          "wrong!");

            // output is a 10d tensor
            constexpr index_t N1 = NPerBlock;

            constexpr index_t W3 = GemmNPerThreadSubC / NPerBlock;
            constexpr index_t W2 = GemmNLevel0Cluster * GemmNLevel1Cluster;
            constexpr index_t W1 = WoPerBlock / fwd(W2 * W3);

            constexpr index_t K2 = GemmMPerThreadSubC;
            constexpr index_t K1 = KPerBlock / KPerThread;

            constexpr auto out_10d_global_desc_old =
                fwd(out_k_h_w_n_global_desc_old)
                    .Fold(I3, Number<N1>{})
                    .Fold(I2, Number<W1>{}, Number<W2>{}, Number<W3>{})
                    .Fold(I0, Number<K1>{}, Number<K2>{});

            constexpr auto out_10d_global_desc = make_native_tensor_descriptor(
                out_10d_global_desc_old.GetLengths(), out_10d_global_desc_old.GetStrides());

            constexpr auto out_10d_thread_desc_old =
                fwd(out_k_h_w_n_thread_desc_old)
                    .Fold(I3, Number<N1>{})
                    .Fold(I2, Number<W1>{}, Number<1>{}, Number<W3>{})
                    .Fold(I0, Number<1>{}, Number<K2>{});

            constexpr auto out_10d_thread_desc = make_native_tensor_descriptor(
                out_10d_thread_desc_old.GetLengths(0), out_10d_thread_desc_old.GetStrides());

            Float* p_out_thread_on_global =
                p_out_global +
                out_k_h_w_n_global_desc.CalculateOffset({k_block_data_begin + k_thread_data_begin,
                                                         ho_block_data_begin + ho_thread_data_begin,
                                                         wo_block_data_begin + wo_thread_data_begin,
                                                         n_block_data_begin + n_thread_data_begin});

            ThreadwiseGenericTensorSliceCopy_v4r2<decltype(out_10d_thread_desc),
                                                  decltype(out_10d_global_desc),
                                                  decltype(out_10d_thread_desc.GetLengths()),
                                                  arithmetic_sequence_gen<0, 10, 1>::type,
                                                  9,
                                                  OutThreadCopyDataPerAccess_N,
                                                  OutThreadCopyDataPerAccess_N>(
                make_zero_array<index_t, 10>(), make_zero_array<index_t, 10>())
                .Run(p_out_thread, p_out_thread_on_global);
        });
    }
};

} // namespace ck
#endif
