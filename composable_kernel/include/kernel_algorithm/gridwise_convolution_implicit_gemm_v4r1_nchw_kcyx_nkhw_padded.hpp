#ifndef CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4R1_NCHW_KCYX_NKHW_PADDED_HPP
#define CK_GRIDWISE_CONVOLUTION_IMPLICIT_GEMM_V4R1_NCHW_KCYX_NKHW_PADDED_HPP

#include "common_header.hpp"
#include "ConstantTensorDescriptor.hpp"
#include "ConstantMergedTensorDescriptor.hpp"
#include "ConstantMatrixDescriptor.hpp"
#include "blockwise_generic_tensor_slice_copy.hpp"
#include "blockwise_gemm.hpp"
#include "threadwise_generic_tensor_slice_copy.hpp"

namespace ck {

// define B = merge(N0, Ho, Wo)
template <index_t GridSize,
          index_t BlockSize,
          typename Float,
          typename InGlobalDesc,
          typename WeiGlobalDesc,
          typename OutGlobalDesc,
          typename ConvStrides,
          typename ConvDilations,
          typename LeftPads,
          typename RightPads,
          index_t BPerBlock,
          index_t KPerBlock,
          index_t EPerBlock,
          index_t GemmNRepeat,
          index_t GemmMPerThreadSubC,
          index_t GemmNPerThreadSubC,
          index_t GemmMLevel0Cluster,
          index_t GemmNLevel0Cluster,
          index_t GemmMLevel1Cluster,
          index_t GemmNLevel1Cluster,
          index_t GemmKPerThreadLoop,
          index_t GemmDataPerReadA,
          index_t GemmDataPerReadB,
          typename InBlockCopySubLengths_E_N1_B_N2,
          typename InBlockCopyClusterLengths_E_N1_B_N2,
          typename InBlockCopyThreadClusterArrangeOrder,
          typename InBlockCopySrcAccessOrder,
          typename InBlockCopyDstAccessOrder,
          index_t InBlockCopySrcDataPerRead_B,
          index_t InBlockCopyDstDataPerWrite_N2,
          typename WeiBlockCopySubLengths_E_K,
          typename WeiBlockCopyClusterLengths_E_K,
          typename WeiBlockCopyThreadClusterArrangeOrder,
          typename WeiBlockCopySrcAccessOrder,
          typename WeiBlockCopyDstAccessOrder,
          index_t WeiBlockCopySrcDataPerRead_E,
          index_t WeiBlockCopyDstDataPerWrite_K>
struct GridwiseConvolutionImplicitGemm_v4r1_nchw_kcyx_nkhw_padded
{
#if 1
    __device__ void Run(const Float* const __restrict__ p_in_global,
                        const Float* const __restrict__ p_wei_global,
                        Float* const __restrict__ p_out_global) const
    {
        // this is a mess
        // TODO: find more elegent way of specifying (or calculating) performance parameters
        constexpr index_t N1 = GemmNRepeat;
        constexpr index_t N2 = GemmNPerThreadSubC;

        static_assert((N1 * N2 * BPerBlock) %
                              (GemmNPerThreadSubC * GemmNLevel0Cluster * GemmNLevel1Cluster) ==
                          0,
                      "wrong!");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr auto True = integral_constant<bool, true>{};

        constexpr auto in_n_c_hi_wi_global_desc =
            make_native_tensor_descriptor(InGlobalDesc::GetLengths(), InGlobalDesc::GetStrides());
        constexpr auto wei_k_c_y_x_global_desc =
            make_native_tensor_descriptor(WeiGlobalDesc::GetLengths(), WeiGlobalDesc::GetStrides());
        constexpr auto out_n_k_ho_wo_global_desc =
            make_native_tensor_descriptor(OutGlobalDesc::GetLengths(), OutGlobalDesc::GetStrides());

        constexpr index_t N  = in_n_c_hi_wi_global_desc.GetLength(I0);
        constexpr index_t C  = in_n_c_hi_wi_global_desc.GetLength(I1);
        constexpr index_t Hi = in_n_c_hi_wi_global_desc.GetLength(I2);
        constexpr index_t Wi = in_n_c_hi_wi_global_desc.GetLength(I3);

        constexpr index_t K  = out_n_k_ho_wo_global_desc.GetLength(I1);
        constexpr index_t Ho = out_n_k_ho_wo_global_desc.GetLength(I2);
        constexpr index_t Wo = out_n_k_ho_wo_global_desc.GetLength(I3);

        constexpr index_t Y = wei_k_c_y_x_global_desc.GetLength(I2);
        constexpr index_t X = wei_k_c_y_x_global_desc.GetLength(I3);

        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];

        static_assert(N % (N1 * N2) == 0, "wrong! cannot divice N evenly among thread");

        constexpr index_t N0 = N / (N1 * N2);

        constexpr index_t B = N0 * Ho * Wo;

        constexpr index_t E = C * Y * X;

        // sanity-check for vectorized memory load
        static_assert((Wo == 1 || (ConvStrideW == 1 || InBlockCopySrcDataPerRead_B == 1)) &&
                          (X == 1 || ConvDilationW % InBlockCopySrcDataPerRead_B == 0),
                      "wrong! aligment requirement for vectorized global load of input tensor will "
                      "be violated");

        // divide block work by [K, B]
        static_assert(K % KPerBlock == 0 && B % BPerBlock == 0 && E % EPerBlock == 0,
                      "wrong! cannot divide work evenly among block");

        constexpr index_t KBlockWork = K / KPerBlock;
        constexpr index_t BBlockWork = B / BPerBlock;

        constexpr auto block_work_desc =
            make_ConstantTensorDescriptor_packed(Sequence<KBlockWork, BBlockWork>{});

        const auto block_work_multi_id =
            block_work_desc.GetMultiIndexFrom1dIndex(get_block_1d_id());

        const index_t k_block_data_on_global = block_work_multi_id[0] * KPerBlock;
        const index_t b_block_data_on_global = block_work_multi_id[1] * BPerBlock;

        // input tensor
        //     global memory
        constexpr auto in_n_c_hip_wip_global_desc = transform_tensor_descriptor(
            in_n_c_hi_wi_global_desc,
            make_tuple(
                PassThrough<N>{}, PassThrough<C>{}, Pad<Sequence<Hi, Wi>, LeftPads, RightPads>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}));

        constexpr auto in_n0_n1_n2_c_y_ho_x_wo_global_desc = transform_tensor_descriptor(
            in_n_c_hip_wip_global_desc,
            make_tuple(Unmerge<Sequence<N0, N1, N2>>{},
                       PassThrough<C>{},
                       Embed<Sequence<Y, Ho>, Sequence<ConvDilationH, ConvStrideH, 0>>{},
                       Embed<Sequence<X, Wo>, Sequence<ConvDilationW, ConvStrideW, 0>>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}, Sequence<4, 5>{}, Sequence<6, 7>{}));

        constexpr auto in_e_n1_b_n2_global_desc = transform_tensor_descriptor(
            in_n0_n1_n2_c_y_ho_x_wo_global_desc,
            make_tuple(Merge<Sequence<C, Y, X>>{},
                       PassThrough<N1>{},
                       Merge<Sequence<N0, Ho, Wo>>{},
                       PassThrough<N2>{}),
            make_tuple(Sequence<3, 4, 6>{}, Sequence<1>{}, Sequence<0, 5, 7>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        //     memory layout descriptor in LDS [E, N1, B, N2], dst of blockwise copy
        //     be careful of LDS alignment
        constexpr auto in_e_n1_b_n2_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<EPerBlock, N1, BPerBlock, N2>{}, Number<InBlockCopyDstDataPerWrite_N2>{});

        //     this check is ad-hoc
        //     TODO: need to properly implement tensor descriptor with multiple alignment
        //     requirements
        static_assert(in_e_n1_b_n2_block_desc.GetStride(I1) % GemmDataPerReadB == 0,
                      "GemmDataPerReadB alignment requirement is not satisfied");

        // input blockwise copy
        //     slice a merged tensor, reorder and copy to a normal tensor
        //     this copy operator already has blockwise offset built-in
        auto blockwise_in_copy =
            BlockwiseGenericTensorSliceCopy_v4<BlockSize,
                                               decltype(in_e_n1_b_n2_global_desc),
                                               decltype(in_e_n1_b_n2_block_desc),
                                               decltype(in_e_n1_b_n2_block_desc.GetLengths()),
                                               InBlockCopySubLengths_E_N1_B_N2,
                                               InBlockCopyClusterLengths_E_N1_B_N2,
                                               InBlockCopyThreadClusterArrangeOrder,
                                               InBlockCopySrcAccessOrder,
                                               InBlockCopyDstAccessOrder,
                                               2,
                                               3,
                                               InBlockCopySrcDataPerRead_B,
                                               InBlockCopyDstDataPerWrite_N2>(
                {0, 0, b_block_data_on_global, 0}, {0, 0, 0, 0});

#if 0
        // weight tensor
        //     tensor descriptor in device memory, src of blockwise copy
        constexpr auto wei_e_k_global_desc =
            transform_tensor_descriptor(wei_k_c_y_x_global_desc,
                                        make_tuple(Merge<Sequence<C, Y, X>>{}, PassThrough<K>{}),
                                        make_tuple(Sequence<1, 2, 3>{}, Sequence<0>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));
#else // hack
        constexpr auto wei_e_k_global_desc_old =
            WeiGlobalDesc::Unfold(I1, I3).ReorderGivenNew2Old(Sequence<1, 0>{});

        constexpr auto wei_e_k_global_desc = make_native_tensor_descriptor(
            wei_e_k_global_desc_old.GetLengths(), wei_e_k_global_desc_old.GetStrides());
#endif

        //     tensor descriptor in LDS, dst of blockwise copy
        //     be careful of LDS alignment
        constexpr auto wei_e_k_block_desc = make_native_tensor_descriptor_aligned(
            Sequence<EPerBlock, KPerBlock>{},
            Number<math::lcm(WeiBlockCopyDstDataPerWrite_K, GemmDataPerReadA)>{});

        //     this check is ad-hoc
        //     TODO: need to properly implement tensor descriptor with multiple alignment
        //     requirements
        static_assert(wei_e_k_block_desc.GetStride(I0) % GemmDataPerReadA == 0,
                      "GemmDataPerReadA alignment requirement is not satisfied");

        // operator for blockwise copy of weight into LDS
        //     slice a tensor, and copy it into another tensor
        //     this copy operator already have blockwise offset built-in
        auto blockwise_wei_copy =
            BlockwiseGenericTensorSliceCopy_v4<BlockSize,
                                               decltype(wei_e_k_global_desc),
                                               decltype(wei_e_k_block_desc),
                                               decltype(wei_e_k_block_desc.GetLengths()),
                                               WeiBlockCopySubLengths_E_K,
                                               WeiBlockCopyClusterLengths_E_K,
                                               WeiBlockCopyThreadClusterArrangeOrder,
                                               WeiBlockCopySrcAccessOrder,
                                               WeiBlockCopyDstAccessOrder,
                                               0,
                                               1,
                                               WeiBlockCopySrcDataPerRead_E,
                                               WeiBlockCopyDstDataPerWrite_K>(
                {0, k_block_data_on_global}, {0, 0});

        // GEMM definition
        // c_mtx += transpose(a_mtx) * b_mtx
        //     a_mtx[EPerBlock, KPerBlock] is in LDS
        //     b_mtx[EPerBlocl, N1 * BPerBlock * N2] is in LDS
        //     c_mtx[KPerBlock, N1 * BPerBlock * N2] is distributed among threads, and saved in
        //     register
        constexpr auto a_e_k_block_mtx_desc = make_ConstantMatrixDescriptor(wei_e_k_block_desc);

        constexpr auto b_e_n1bn2_block_mtx_desc = make_ConstantMatrixDescriptor(
            in_e_n1_b_n2_block_desc.GetLength(I0),
            in_e_n1_b_n2_block_desc.GetLength(I1) * in_e_n1_b_n2_block_desc.GetLength(I2) *
                in_e_n1_b_n2_block_desc.GetLength(I3),
            in_e_n1_b_n2_block_desc.GetStride(I0));

        // sanity check
        static_assert(KPerBlock % (GemmMPerThreadSubC * GemmMLevel0Cluster * GemmMLevel1Cluster) ==
                          0,
                      "wrong!");

        constexpr index_t GemmMRepeat =
            KPerBlock / (GemmMPerThreadSubC * GemmMLevel0Cluster * GemmMLevel1Cluster);

        // c_thread_mtx definition: this is a mess
        // TODO:: more elegent way of defining c_thread_mtx
        constexpr auto c_k0k2_n1n2_thread_mtx_desc = make_ConstantMatrixDescriptor_packed(
            Number<GemmMRepeat * GemmMPerThreadSubC>{}, Number<N1 * N2>{});

        const auto blockwise_gemm = BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_v2<
            BlockSize,
            decltype(a_e_k_block_mtx_desc),
            decltype(b_e_n1bn2_block_mtx_desc),
            decltype(c_k0k2_n1n2_thread_mtx_desc),
            GemmMPerThreadSubC,
            GemmNPerThreadSubC,
            GemmMLevel0Cluster,
            GemmNLevel0Cluster,
            GemmMLevel1Cluster,
            GemmNLevel1Cluster,
            GemmKPerThreadLoop,
            GemmDataPerReadA,
            GemmDataPerReadB>{};

        // LDS allocation for input and weight: be careful of alignment
        constexpr index_t max_align = math::lcm(InBlockCopyDstDataPerWrite_N2,
                                                WeiBlockCopyDstDataPerWrite_K,
                                                GemmDataPerReadA,
                                                GemmDataPerReadB);

        constexpr index_t in_block_space =
            math::integer_least_multiple(in_e_n1_b_n2_block_desc.GetElementSpace(), max_align);

        constexpr index_t wei_block_space =
            math::integer_least_multiple(wei_e_k_block_desc.GetElementSpace(), max_align);

        __shared__ Float p_in_block[in_block_space];
        __shared__ Float p_wei_block[wei_block_space];

        // register allocation for output
        Float p_out_thread[c_k0k2_n1n2_thread_mtx_desc.GetElementSpace()];

        // zero out threadwise output
        threadwise_matrix_set_zero(c_k0k2_n1n2_thread_mtx_desc, p_out_thread);

        // do work
        for(index_t e = 0; e < E; e += EPerBlock)
        {
            blockwise_in_copy.Run(p_in_global, p_in_block);
            blockwise_wei_copy.Run(p_wei_global, p_wei_block);

            __syncthreads();

            blockwise_gemm.Run(p_wei_block, p_in_block, p_out_thread);

            __syncthreads();

            blockwise_in_copy.MoveSrcSliceWindow(make_multi_index(EPerBlock, 0, 0, 0), True);
            blockwise_wei_copy.MoveSrcSliceWindow(make_multi_index(EPerBlock, 0), True);
        }

        // copy output: register to global memory
        {
            constexpr index_t K2 = GemmMPerThreadSubC;
            constexpr index_t K1 = GemmMLevel0Cluster * GemmMLevel1Cluster;

            static_assert(K % (K1 * K2) == 0, "wrong!");

            // define tensor descriptor for threadwise copy
            //     output memory layout descriptor in register
            constexpr auto out_k0_k1_k2_n1_n0_ho_wo_n2_thread_desc =
                make_native_tensor_descriptor_packed(
                    Sequence<KPerBlock / (K1 * K2), 1, K2, N1, 1, 1, 1, N2>{});

            //     output tensor descriptor in register, src of threadwise copy
            constexpr auto out_n0_n1_n2_k0_k1_k2_ho_wo_thread_desc =
                reorder_tensor_descriptor_given_upper2lower(out_k0_k1_k2_n1_n0_ho_wo_n2_thread_desc,
                                                            Sequence<4, 3, 7, 0, 1, 2, 5, 6>{});

            //     output memory layout descriptor in device memory, dst of threadwise copy
            constexpr auto out_n0_n1_n2_k0_k1_k2_ho_wo_global_desc = transform_tensor_descriptor(
                out_n_k_ho_wo_global_desc,
                make_tuple(Unmerge<Sequence<N / (N1 * N2), N1, N2>>{},
                           Unmerge<Sequence<K / (K1 * K2), K1, K2>>{},
                           PassThrough<Ho>{},
                           PassThrough<Wo>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0, 1, 2>{}, Sequence<3, 4, 5>{}, Sequence<6>{}, Sequence<7>{}));

            // calculate origin of thread output tensor on global memory
            //     blockwise GEMM c matrix starting index
            const auto c_thread_mtx_on_block =
                blockwise_gemm.GetBeginOfThreadMatrixC(get_thread_local_1d_id());

            const index_t k_thread_data_on_global =
                k_block_data_on_global + c_thread_mtx_on_block.row;

            const index_t b_thread_data_on_global =
                b_block_data_on_global + c_thread_mtx_on_block.col / N2;

            //     output merged global tensor descriptor, for calculating origin of thread tensor
            //     in global memory
            constexpr auto out_n0_n1_n2_k_ho_wo_global_desc = transform_tensor_descriptor(
                out_n_k_ho_wo_global_desc,
                make_tuple(Unmerge<Sequence<N / (N1 * N2), N1, N2>>{},
                           PassThrough<K>{},
                           PassThrough<Ho>{},
                           PassThrough<Wo>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}, Sequence<4>{}, Sequence<5>{}));

            constexpr auto out_k_n1_b_n2_global_desc = transform_tensor_descriptor(
                out_n0_n1_n2_k_ho_wo_global_desc,
                make_tuple(PassThrough<K>{},
                           PassThrough<N1>{},
                           Merge<Sequence<N0, Ho, Wo>>{},
                           PassThrough<N2>{}),
                make_tuple(Sequence<3>{}, Sequence<1>{}, Sequence<0, 4, 5>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

            //     origin of dst in device memory
            Float* p_out_thread_on_global =
                p_out_global +
                out_k_n1_b_n2_global_desc.CalculateOffset(
                    {k_thread_data_on_global, 0, b_thread_data_on_global, 0});

            ThreadwiseGenericTensorSliceCopy_v4r2<
                decltype(out_n0_n1_n2_k0_k1_k2_ho_wo_thread_desc),
                decltype(out_n0_n1_n2_k0_k1_k2_ho_wo_global_desc),
                decltype(out_n0_n1_n2_k0_k1_k2_ho_wo_thread_desc.GetLengths()),
                arithmetic_sequence_gen<0, 8, 1>::type,
                7,
                1,
                1>({0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0})
                .Run(p_out_thread, p_out_thread_on_global);
        }
    }
#else
    __device__ void Run(const Float* const __restrict__ p_in_global,
                        const Float* const __restrict__ p_wei_global,
                        Float* const __restrict__ p_out_global) const
    {
        // this is a mess
        // TODO: find more elegent way of specifying (or calculating) performance parameters
        constexpr index_t N1 = GemmNRepeat;
        constexpr index_t N2 = GemmNPerThreadSubC;

        static_assert((N1 * N2 * BPerBlock) %
                              (GemmNPerThreadSubC * GemmNLevel0Cluster * GemmNLevel1Cluster) ==
                          0,
                      "wrong!");

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I5 = Number<5>{};

        constexpr auto True = integral_constant<bool, true>{};

        constexpr auto in_n_c_h_w_global_desc =
            make_native_tensor_descriptor(InGlobalDesc::GetLengths(), InGlobalDesc::GetStrides());
        constexpr auto wei_k_c_y_x_global_desc =
            make_native_tensor_descriptor(WeiGlobalDesc::GetLengths(), WeiGlobalDesc::GetStrides());
        constexpr auto out_n_k_h_w_global_desc =
            make_native_tensor_descriptor(OutGlobalDesc::GetLengths(), OutGlobalDesc::GetStrides());

        constexpr index_t N  = in_n_c_h_w_global_desc.GetLength(I0);
        constexpr index_t C  = in_n_c_h_w_global_desc.GetLength(I1);
        constexpr index_t Hi = in_n_c_h_w_global_desc.GetLength(I2);
        constexpr index_t Wi = in_n_c_h_w_global_desc.GetLength(I3);

        constexpr index_t K  = out_n_k_h_w_global_desc.GetLength(I1);
        constexpr index_t Ho = out_n_k_h_w_global_desc.GetLength(I2);
        constexpr index_t Wo = out_n_k_h_w_global_desc.GetLength(I3);

        constexpr index_t Y = wei_k_c_y_x_global_desc.GetLength(I2);
        constexpr index_t X = wei_k_c_y_x_global_desc.GetLength(I3);

        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];

        static_assert(N % (N1 * N2) == 0, "wrong! cannot divice N evenly among thread");

        constexpr index_t N0 = N / (N1 * N2);

        constexpr index_t B = N0 * Ho * Wo;

        constexpr index_t E = C * Y * X;

        // sanity-check for vectorized memory load
        static_assert(ConvStrideW == 1 || InBlockCopySrcDataPerRead_B == 1,
                      "wrong! global vector load of input tensor is wrong");

        static_assert((X == 1 || ConvDilationW % InBlockCopySrcDataPerRead_B == 0),
                      "wrong! aligment requirement for vectorized global load of input tensor will "
                      "be violated");

        // input
        constexpr auto in_n_c_hi_wi_global_desc =
            make_native_tensor_descriptor(InGlobalDesc::GetLengths(), InGlobalDesc::GetStrides());

        constexpr auto in_n_c_hip_wip_global_desc = transform_tensor_descriptor(
            in_n_c_hi_wi_global_desc,
            make_tuple(
                PassThrough<N>{}, PassThrough<C>{}, Pad<Sequence<Hi, Wi>, LeftPads, RightPads>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}));

        constexpr auto in_n0_n1_n2_c_y_ho_x_wo_global_desc = transform_tensor_descriptor(
            in_n_c_hip_wip_global_desc,
            make_tuple(Unmerge<Sequence<N0, N1, N2>>{},
                       PassThrough<C>{},
                       Embed<Sequence<Y, Ho>, Sequence<ConvDilationH, ConvStrideH, 0>>{},
                       Embed<Sequence<X, Wo>, Sequence<ConvDilationW, ConvStrideW, 0>>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}, Sequence<4, 5>{}, Sequence<6, 7>{}));

        constexpr auto in_e_n1_b_n2_global_desc = transform_tensor_descriptor(
            in_n0_n1_n2_c_y_ho_x_wo_global_desc,
            make_tuple(Merge<Sequence<C, Y, X>>{},
                       PassThrough<N1>{},
                       Merge<Sequence<N0, Ho, Wo>>{},
                       PassThrough<N2>{}),
            make_tuple(Sequence<3, 4, 6>{}, Sequence<1>{}, Sequence<0, 5, 7>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        // weight
        constexpr auto wei_e_k_global_desc =
            transform_tensor_descriptor(wei_k_c_y_x_global_desc,
                                        make_tuple(Merge<Sequence<C, Y, X>>{}, PassThrough<K>{}),
                                        make_tuple(Sequence<1, 2, 3>{}, Sequence<0>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

#if 0
        if(get_thread_local_1d_id() == 0 && get_block_1d_id() == 0)
        {
            print_tensor_descriptor("in_n_c_hi_wi_global_desc: ", in_n_c_hi_wi_global_desc);
            print_tensor_descriptor("in_n_c_hip_wip_global_desc: ", in_n_c_hip_wip_global_desc);
            print_tensor_descriptor("in_n0_n1_n2_c_y_ho_x_wo_global_desc: ",
                                    in_n0_n1_n2_c_y_ho_x_wo_global_desc);
            print_tensor_descriptor("in_e_n1_b_n2_global_desc: ", in_e_n1_b_n2_global_desc);

            auto coord3 = make_tensor_coordinate_v2(in_e_n1_b_n2_global_desc, {1, 1, 1, 1});

            auto idx3 = coord3.GetIndex();
            auto idx2 = coord3.GetLowerCoordinate().GetIndex();
            auto idx1 = coord3.GetLowerCoordinate().GetLowerCoordinate().GetIndex();
            auto idx0 =
                coord3.GetLowerCoordinate().GetLowerCoordinate().GetLowerCoordinate().GetIndex();

            print_array("idx3: ", idx3);
            print_array("idx2: ", idx2);
            print_array("idx1: ", idx1);
            print_array("idx0: ", idx0);
        }
#else
        index_t itmp    = get_block_1d_id() + get_thread_local_1d_id();
        auto wei_coord1 = make_tensor_coordinate_v2(wei_e_k_global_desc, {itmp, itmp + 1});

        auto step_sizes = make_multi_index(EPerBlock, 0);

        wei_coord1 += step_sizes;

        p_out_global[0] = wei_coord1.GetLowerCoordinate().GetIndex()[0];
        p_out_global[1] = wei_coord1.GetLowerCoordinate().GetIndex()[1];
        p_out_global[2] = wei_coord1.GetLowerCoordinate().GetIndex()[2];
        p_out_global[3] = wei_coord1.GetLowerCoordinate().GetIndex()[3];
#endif
    }
#endif
};

} // namespace ck
#endif
