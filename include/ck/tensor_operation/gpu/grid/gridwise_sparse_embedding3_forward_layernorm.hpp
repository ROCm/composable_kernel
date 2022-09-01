// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/get_id.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/reduction_common.hpp"
#include "ck/utility/reduction_operator.hpp"
#include "ck/utility/reduction_functions_accumulate.hpp"
#include "ck/tensor_operation/gpu/block/reduction_functions_blockwise.hpp"
#include "ck/tensor_operation/gpu/thread/reduction_functions_threadwise.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_welford.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

template <
          typename EmbType,
          typename IndexType,
          typename GammaDataType,
          typename BetaDataType,
          typename AccDataType,
          typename OutType,
          typename EmbeddingAccessOrder,
          typename EmbeddingScalarVector,
          typename OutputScalarVector,
          typename OutGridDesc,
          typename GammaGridDesc,
          typename BetaGridDesc
          ck::index_t BlockSize,
          ck::index_t DimClusterSize,
          ck::index_t RowClusterSize,
          ck::index_t DimPerBlock,   // Row x Dim, along Dim
          ck::index_t RowPerBlock,   // Row x Dim, along Row
          ck::index_t DimVectorSize,
          ck::index_t RowVectorSize
          >
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
    __global__ void kernel_sparse_embedding3_forward_layernorm(
                    OutType * p_out,
                    const EmbType* p_emb_a,
                    const EmbType* p_emb_b,
                    const EmbType* p_emb_c,
                    const IndexType* p_index_a,
                    const IndexType* p_index_b,
                    const IndexType* p_index_c,
                    const GammaDataType* p_gamma,
                    const BetaDataType* p_beta,
                    const OutGridDesc out_grid_desc,
                    const GammaGridDesc gamma_grid_desc,
                    const BetaGridDesc beta_grid_desc,
                    const ck::index_t num_row
                )
    {
        GridwiseSparseEmbedding3ForwardLayernorm<
                EmbType,
                IndexType,
                GammaDataType,
                BetaDataType,
                AccDataType,
                OutType,
                EmbeddingAccessOrder,
                EmbeddingScalarVector,
                OutputScalarVector,
                OutGridDesc,
                GammaGridDesc,
                BetaGridDesc,
                BlockSize,
                DimClusterSize,
                RowClusterSize,
                DimPerBlock,
                RowPerBlock,
                DimVectorSize,
                RowVectorSize>::Run(
            p_out,
            p_emb_a,
            p_emb_b,
            p_emb_c,
            p_index_a,
            p_index_b,
            p_index_c,
            p_gamma,
            p_beta,
            out_grid_desc,
            gamma_grid_desc,
            beta_grid_desc,
            num_row
        );
    }

template <typename EmbType,
          typename IndexType,
          typename GammaDataType,
          typename BetaDataType,
          typename AccDataType,
          typename OutType,
          typename OutGridDesc,
          typename GammaGridDesc,
          typename BetaGridDesc,
          ck::index_t BlockSize,
          ck::index_t DimClusterSize,
          ck::index_t RowClusterSize,
          ck::index_t DimPerBlock,      // Row x Dim, along Dim
          ck::index_t RowPerBlock,      // Row x Dim, along Row
          ck::index_t DimVectorSize,    // this is actually not vector, but number of registers
          ck::index_t RowVectorSize,
          ck::index_t EmbeddingDim
        >
struct GridwiseSparseEmbedding3ForwardLayernorm {
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static_assert(BlockSize == RowClusterSize * DimClusterSize, "Invalid cluster distribution within block");
    static_assert(DimPerBlock % (DimClusterSize * 1) == 0, "");
    static_assert(RowPerBlock % (RowClusterSize * RowVectorSize) == 0, "");

    constexpr auto DimSubBlocks = DimPerBlock / (DimClusterSize * DimVectorSize);
    constexpr auto RowSubBlocks = RowPerBlock / (RowClusterSize * RowVectorSize);

    static_assert((DimPerBlock & DimSubBlocks == 0) && (RowPerBlock & RowSubBlocks == 0), "");
    static constexpr DimPerSubBlock = DimPerBlock / DimSubBlocks;
    static constexpr RowPerSubBlock = RowPerBlock / RowSubBlocks;

    using ThreadBufferDesc = decltype(make_naive_tensor_descriptor_packed(
            make_tuple(Number<DimSubBlocks * DimVectorSize>{}, Number<RowSubBlocks * RowVectorSize>{})));

    using ThreadSubBufferDesc = decltype(make_naive_tensor_descriptor_packed(
            make_tuple(Number<DimVectorSize>{}, Number<RowVectorSize>{})));

    // using ThreadClusterLengths_R_D = Sequence<ThreadClusterSize_R, ThreadClusterSize_D>;
    // static constexpr auto thread_cluster_desc =
    //     make_cluster_descriptor(ThreadClusterLengths_R_D{});

    using ThreadReduceSrcDesc_R_D = decltype(make_naive_tensor_descriptor_packed(
        make_tuple(Number<DimVectorSize>{}, Number<RowVectorSize>{})));
    using ThreadReduceDstDesc_D =
        decltype(make_naive_tensor_descriptor_packed(make_tuple(Number<RowVectorSize>{})));

    using ThreadwiseWelford =
        ThreadwiseWelford<AccDataType, ThreadReduceSrcDesc_R_D, ThreadReduceDstDesc_D>;

    using ThreadwiseRowDesc = 
            decltype(make_naive_tensor_descriptor_packed(make_tuple(Number<DimVectorSize>{})));


    __device__ static void
    Run(OutType * p_out,
        const EmbType* p_emb_a,
        const EmbType* p_emb_b,
        const EmbType* p_emb_c,
        const IndexType* p_index_a,
        const IndexType* p_index_b,
        const IndexType* p_index_c,
        const GammaDataType* p_gamma,
        const BetaDataType* p_beta,
        const OutGridDesc out_grid_desc,
        const GammaGridDesc gamma_grid_desc,
        const BetaGridDesc beta_grid_desc,
        const ck::index_t num_row)
    {
        const index_t thread_local_id = get_thread_local_1d_id();
        const index_t block_global_id = get_block_1d_id();

        const auto thread_cluster_idx =
            thread_cluster_desc.CalculateBottomIndex(make_multi_index(thread_local_id));

        const auto thread_r_cluster_id = thread_cluster_idx[I0];
        const auto thread_d_cluster_id = thread_cluster_idx[I1];
        
        const auto index_length = out_grid_desc.GetLength(I0);
        const auto emb_dim = out_grid_desc.GetLength(I1);

        const auto out_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_emb_a, out_grid_desc.GetElementSpaceSize());

        const auto index_start = block_global_id * DimPerBlock;

        auto threadwise_welford = ThreadwiseWelford();

        constexpr auto thread_buf_size = DimSubBlocks * DimVectorSize * RowSubBlocks * RowVectorSize;
        constexpr auto thread_buf_desc = make_naive_tensor_descriptor_packed(make_tuple(DimSubBlocks, DimVectorSize, RowSubBlocks, RowVectorSize));

        StaticBuffer<AddressSpaceEnum::Vgpr, EmbType, thread_buf_size, true> in_thread_buf_a;
        StaticBuffer<AddressSpaceEnum::Vgpr, EmbType, thread_buf_size, true> in_thread_buf_b;
        StaticBuffer<AddressSpaceEnum::Vgpr, EmbType, thread_buf_size, true> in_thread_buf_c;

        StaticBuffer<AddressSpaceEnum::Sgpr, IndexType, DimPerBlock, true> index_buf_a;
        StaticBuffer<AddressSpaceEnum::Sgpr, IndexType, DimPerBlock, true> index_buf_b;
        StaticBuffer<AddressSpaceEnum::Sgpr, IndexType, DimPerBlock, true> index_buf_c;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, thread_buf_size, true>
            acc_thread_buf;
        
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, thread_buf_size, true>
            out_thread_buf;
        
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, RowVectorSize, true> gamma_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, RowVectorSize, true> beta_thread_buf;

        StaticBuffer<AddressSpaceEnum::Vgpr, AccDataType, RowVectorSize, true> mean_thread_buf;

        auto threadwise_gamma_load =
            ThreadwiseTensorSliceTransfer_v2<GammaDataType,
                                             AccDataType,
                                             GammaGridDesc,
                                             ThreadwiseRowDesc,
                                             Sequence<DimVectorSize>,
                                             Sequence<0>,
                                             0,
                                             RowVectorSize,
                                             1,
                                             true>(
                gamma_grid_desc, make_multi_index(thread_d_cluster_id * KThreadSliceSize));

        auto threadwise_beta_load = ThreadwiseTensorSliceTransfer_v2<BetaDataType,
                                                                     AccDataType,
                                                                     BetaGridDesc,
                                                                     ThreadwiseRowDesc,
                                                                     Sequence<DimVectorSize>,
                                                                     Sequence<0>,
                                                                     0,
                                                                     RowVectorSize,
                                                                     1,
                                                                     true>(
            beta_grid_desc, make_multi_index(thread_d_cluster_id * KThreadSliceSize));

        auto threadwise_out_store =
            ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                               OutDataType,
                                               decltype(thread_buffer_desc_m_k),
                                               GridDesc_M_K,
                                               AccElementwiseOperation,
                                               ThreadBufferLengths_M_K,
                                               ThreadBufferDimAccessOrder,
                                               YDstVectorDim,
                                               YDstVectorSize,
                                               InMemoryDataOperationEnum::Set,
                                               1,
                                               true>(
                y_grid_desc_m_k,
                make_multi_index(block_global_id * M_BlockTileSize +
                                     thread_m_cluster_id * MThreadSliceSize,
                                 thread_k_cluster_id * KThreadSliceSize),
                acc_elementwise_op);
        static_for<0, DimPerBlock, 1>{}([&](auto i_idx){
            // prefer use s_load
            index_buf_a[i_idx] = p_index_a[index_start + i_idx];
            index_buf_b[i_idx] = p_index_b[index_start + i_idx];
            index_buf_c[i_idx] = p_index_c[index_start + i_idx];
        });

        auto load_current_sub_row = [&](index_t i_dim_sub_, index_t i_row_sub_){
            vector_type_maker_t<EmbType, RowVectorSize> emb_vector_a;
            vector_type_maker_t<EmbType, RowVectorSize> emb_vector_b;
            vector_type_maker_t<EmbType, RowVectorSize> emb_vector_c;

            using src_vector_t = typename decltype(emb_vector_a)::type;
            static_for<0, DimVectorSize, 1>{}([&](auto i_dim_vec) {
                index_t current_dim = i_dim_sub_ * DimPerSubBlock + i_dim_vec;
                IndexType index_a = index_buf_a[current_dim];
                IndexType index_b = index_buf_b[current_dim];
                IndexType index_c = index_buf_c[current_dim];

                index_t thread_offset = thread_local_id + i_row_sub_ * BlockSize * sizeof(EmbType) * RowVectorSize;

                int32x4_t emb_res_a = amdgcn_make_buffer_resource(p_emb_a + index_a * EmbeddingDim);
                int32x4_t emb_res_b = amdgcn_make_buffer_resource(p_emb_b + index_b * EmbeddingDim);
                int32x4_t emb_res_c = amdgcn_make_buffer_resource(p_emb_c + index_c * EmbeddingDim);
                emb_vector_a.template AsType<src_vector_t>()(I0) = amd_buffer_load_impl<EmbType, RowVectorSize>(emb_res_a, thread_offset, 0);
                emb_vector_b.template AsType<src_vector_t>()(I0) = amd_buffer_load_impl<EmbType, RowVectorSize>(emb_res_b, thread_offset, 0);
                emb_vector_c.template AsType<src_vector_t>()(I0) = amd_buffer_load_impl<EmbType, RowVectorSize>(emb_res_c, thread_offset, 0);

                static_for<0, RowVectorSize, 1>{}([&](auto i_row_vec){
                    constexpr auto register_offset = thread_buf_desc.CalculateOffset(make_tuple(i_dim_sub_, i_dim_vec, i_row_sub_, i_row_vec));
                    in_thread_buf_a(Number<register_offset>{}) = emb_vector_a.template AsType<EmbType>()[Number<i_row_vec>{}];
                    in_thread_buf_b(Number<register_offset>{}) = emb_vector_b.template AsType<EmbType>()[Number<i_row_vec>{}];
                    in_thread_buf_c(Number<register_offset>{}) = emb_vector_c.template AsType<EmbType>()[Number<i_row_vec>{}];
                });
            });
        };

        auto accumulate_current_sub_row = [&](index_t i_dim_sub_, index_t i_row_sub_){
            static_for<0, DimVectorSize, 1>{}([&](auto i_dim_vec) {
                static_for<0, RowVectorSize, 1>{}([&](auto i_row_vec) {
                    constexpr auto register_offset = thread_buf_desc.CalculateOffset(make_tuple(i_dim_sub_, i_dim_vec, i_row_sub_, i_row_vec));
                    AccDataType va = ck::type_convert<AccDataType>(in_thread_buf_a(Number<register_offset>{}));
                    AccDataType vb = ck::type_convert<AccDataType>(in_thread_buf_b(Number<register_offset>{}));
                    AccDataType vc = ck::type_convert<AccDataType>(in_thread_buf_c(Number<register_offset>{}));

                    acc_thread_buf(Number<register_offset>{}) = va + vb + vc;
                });
            });
        };

        auto threadwise_welford_sub_row = [&](index_t i_dim_sub_, index_t i_row_sub_){
            static_for<0, DimVectorSize, 1>{}([&](auto i_dim_vec) {
                static_for<0, RowVectorSize, 1>{}([&](auto i_row_vec) {
                    constexpr auto register_offset = thread_buf_desc.CalculateOffset(make_tuple(i_dim_sub_, i_dim_vec, i_row_sub_, i_row_vec));
                    AccDataType va = ck::type_convert<AccDataType>(in_thread_buf_a(Number<register_offset>{}));
                    AccDataType vb = ck::type_convert<AccDataType>(in_thread_buf_b(Number<register_offset>{}));
                    AccDataType vc = ck::type_convert<AccDataType>(in_thread_buf_c(Number<register_offset>{}));

                    acc_thread_buf(Number<register_offset>{}) = va + vb + vc;
                });
            });
        };

        static_for<0, DimSubBlocks, 1>{}([&](auto i_dim_sub){
            if constexpr(RowSubBlocks % 2 == 0)
            {
                load_current_sub_row(i_dim_sub, 0);
                static_for<0, (RowSubBlocks / 2 - 1), 1>{}([&](auto i_row) {
                    load_current_sub_row(i_dim_sub, 2 * i_row + 1);
                    accumulate_current_sub_row(i_dim_sub, 2 * i_row);

                    load_current_sub_row(i_dim_sub, 2 * i_row + 2);
                    accumulate_current_sub_row(i_dim_sub, 2 * i_row + 1);

                    // thread-wise welford

                });

                load_current_sub_row(i_dim_sub, RowSubBlocks - 1);
                accumulate_current_sub_row(i_dim_sub, RowSubBlocks - 2);
                accumulate_current_sub_row(i_dim_sub, RowSubBlocks - 1);

                // thread-wise welford

                // blockwise welford

                // store
            }
            else if constexpr(RowSubBlocks % 3 == 0)
            {

            }
        });
    }
};

}
