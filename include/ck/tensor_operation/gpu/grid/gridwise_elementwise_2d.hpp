#pragma once

#include "cluster_descriptor.hpp"
#include "data_type.hpp"
#include "element_wise_operation.hpp"
#include "threadwise_tensor_slice_transfer.hpp"

namespace ck {

template <typename GridwiseEltwise,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename GridDesc_M_N,
          typename ElementwiseFunctor>
__global__ void kernel_elementwise_2d(const ADataType* __restrict__ p_a_global,
                                      const BDataType* __restrict__ p_b_global,
                                      CDataType* __restrict__ p_c_global,
                                      const GridDesc_M_N a_grid_desc_m_k,
                                      const GridDesc_M_N b_grid_desc_m_k,
                                      const GridDesc_M_N c_grid_desc_m_k,
                                      const ElementwiseFunctor functor)
{
    GridwiseEltwise::Run(p_a_global,
                         p_b_global,
                         p_c_global,
                         a_grid_desc_m_k,
                         b_grid_desc_m_k,
                         c_grid_desc_m_k,
                         functor);
}

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename GridDesc_M_N,
          typename ElementwiseFunctor,
          index_t MThreadPerBlock,
          index_t NThreadPerBlock,
          index_t MThreadTileSize,
          index_t NThreadTileSize,
          index_t AThreadTransferSrcVectorDim,
          index_t AThreadTransferSrcScalarPerVector,
          index_t BThreadTransferSrcVectorDim,
          index_t BThreadTransferSrcScalarPerVector,
          index_t CThreadTransferSrcScalarPerVector>
struct GridwiseElementwise_2D
{
    static constexpr auto thread_buf_desc_M_N = make_naive_tensor_descriptor_packed(
        make_tuple(Number<MThreadTileSize>{}, Number<NThreadTileSize>{}));

    using PassThrough       = tensor_operation::element_wise::PassThrough;
    using ThreadBufDesc_M_N = decltype(thread_buf_desc_M_N);

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    static constexpr int M_BlockTileSize = MThreadPerBlock * MThreadTileSize;
    static constexpr int N_BlockTileSize = NThreadPerBlock * NThreadTileSize;

    static __device__ __host__ auto CalculateElementwiseIndex(const GridDesc_M_N& grid_desc_m_n)
    {
        const index_t thread_id = get_thread_local_1d_id();
        const index_t block_id  = get_block_1d_id();

        const index_t M              = grid_desc_m_n.GetLength(I0);
        const index_t gridSize_m     = M / M_BlockTileSize;
        const index_t block_2d_idx_m = block_id % gridSize_m;
        const index_t block_2d_idx_n = block_id / gridSize_m;

        constexpr auto thread_desc =
            make_cluster_descriptor(Sequence<MThreadPerBlock, NThreadPerBlock>{}, Sequence<1, 0>{});

        const auto thread_2d_idx = thread_desc.CalculateBottomIndex(make_multi_index(thread_id));

        return make_multi_index(
            block_2d_idx_m * M_BlockTileSize + thread_2d_idx[I0] * MThreadTileSize,
            block_2d_idx_n * N_BlockTileSize + thread_2d_idx[I1] * NThreadTileSize);
    }

    __device__ static void Run(const ADataType* __restrict__ p_a_global,
                               const BDataType* __restrict__ p_b_global,
                               CDataType* __restrict__ p_c_global,
                               const GridDesc_M_N a_grid_desc_m_n,
                               const GridDesc_M_N b_grid_desc_m_n,
                               const GridDesc_M_N c_grid_desc_m_n,
                               const ElementwiseFunctor functor)
    {
        const auto a_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_global, a_grid_desc_m_n.GetElementSpaceSize());
        const auto b_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_global, b_grid_desc_m_n.GetElementSpaceSize());
        auto c_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_global, c_grid_desc_m_n.GetElementSpaceSize());

        StaticBuffer<AddressSpaceEnum::Vgpr, ADataType, MThreadTileSize * NThreadTileSize, true>
            a_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, BDataType, MThreadTileSize * NThreadTileSize, true>
            b_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, CDataType, MThreadTileSize * NThreadTileSize, true>
            c_thread_buf;

        const auto a_global_load_offset = CalculateElementwiseIndex(a_grid_desc_m_n);
        const auto b_global_load_offset = CalculateElementwiseIndex(b_grid_desc_m_n);

        auto a_global_load = ThreadwiseTensorSliceTransfer_v2<
            ADataType,
            ADataType,
            GridDesc_M_N,
            decltype(thread_buf_desc_M_N),
            Sequence<MThreadTileSize, NThreadTileSize>, // SliceLengths
            Sequence<0, 1>,                             // DimAccessOrder
            AThreadTransferSrcVectorDim,
            AThreadTransferSrcScalarPerVector,
            1, // SrcScalarStrideInVector
            false>{a_grid_desc_m_n, a_global_load_offset};

        auto b_global_load = ThreadwiseTensorSliceTransfer_v2<
            BDataType,
            BDataType,
            GridDesc_M_N,
            decltype(thread_buf_desc_M_N),
            Sequence<MThreadTileSize, NThreadTileSize>, // SliceLengths
            Sequence<0, 1>,                             // DimAccessOrder
            BThreadTransferSrcVectorDim,
            BThreadTransferSrcScalarPerVector,
            1, // SrcScalarStrideInVector
            false>{b_grid_desc_m_n, b_global_load_offset};

        a_global_load.Run(
            a_grid_desc_m_n, a_global_buf, thread_buf_desc_M_N, make_tuple(I0, I0), a_thread_buf);

        b_global_load.Run(
            b_grid_desc_m_n, b_global_buf, thread_buf_desc_M_N, make_tuple(I0, I0), b_thread_buf);

        static_for<0, MThreadTileSize, 1>{}([&](auto m) {
            static_for<0, NThreadTileSize, 1>{}([&](auto n) {
                constexpr auto offset = thread_buf_desc_M_N.CalculateOffset(make_tuple(m, n));
                functor(c_thread_buf(Number<offset>{}),
                        a_thread_buf(Number<offset>{}),
                        b_thread_buf(Number<offset>{}));
            });
        });

        // TODO - global write
        const auto c_global_write_offset = CalculateElementwiseIndex(c_grid_desc_m_n);
        auto c_global_write              = ThreadwiseTensorSliceTransfer_v1r3<
            CDataType,
            CDataType,
            decltype(thread_buf_desc_M_N),
            GridDesc_M_N,
            PassThrough,
            Sequence<MThreadTileSize, NThreadTileSize>, // SliceLengths
            Sequence<0, 1>,                             // DimAccessOrder
            1,                                          // DstVectorDim
            CThreadTransferSrcScalarPerVector,          // DstScalarPerVector
            InMemoryDataOperationEnum::Set,             // DstInMemOp
            1,                                          // DstScalarStrideInVector
            false>{c_grid_desc_m_n, c_global_write_offset, PassThrough{}};

        c_global_write.Run(
            thread_buf_desc_M_N, make_tuple(I0, I0), c_thread_buf, c_grid_desc_m_n, c_global_buf);
    }
};

} // namespace ck
