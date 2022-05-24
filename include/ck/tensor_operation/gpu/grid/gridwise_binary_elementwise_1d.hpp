#pragma once

#include "cluster_descriptor.hpp"
#include "data_type.hpp"
#include "element_wise_operation.hpp"
#include "threadwise_tensor_slice_transfer.hpp"

namespace ck {

template <typename GridwiseBinEltwise,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename GridDesc_M0,
          typename ElementwiseFunctor>
__global__ void kernel_binary_elementwise_1d(const ADataType* __restrict__ p_a_global,
                                             const BDataType* __restrict__ p_b_global,
                                             CDataType* __restrict__ p_c_global,
                                             const GridDesc_M0 a_grid_desc_m0,
                                             const GridDesc_M0 b_grid_desc_m0,
                                             const GridDesc_M0 c_grid_desc_m0,
                                             const ElementwiseFunctor functor)
{
    GridwiseBinEltwise::Run(p_a_global,
                            p_b_global,
                            p_c_global,
                            a_grid_desc_m0,
                            b_grid_desc_m0,
                            c_grid_desc_m0,
                            functor);
}

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ComputeDataType,
          typename GridDesc_M0,
          typename ElementwiseFunctor,
          index_t M0PerThread,
          index_t AScalarPerVector = M0PerThread,
          index_t BScalarPerVector = M0PerThread>
struct GridwiseBinaryElementwise_1D
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto thread_desc_m0 =
        make_naive_tensor_descriptor_packed(make_tuple(Number<M0PerThread>{}));

    using PassThrough = tensor_operation::element_wise::PassThrough;

    static __device__ auto CalculateElementwiseIndex()
    {
        const index_t global_thread_id = get_thread_global_1d_id();
        return make_multi_index(global_thread_id * M0PerThread);
    }

    __device__ static void Run(const ADataType* __restrict__ p_a_global,
                               const BDataType* __restrict__ p_b_global,
                               CDataType* __restrict__ p_c_global,
                               const GridDesc_M0 a_grid_desc_m0,
                               const GridDesc_M0 b_grid_desc_m0,
                               const GridDesc_M0 c_grid_desc_m0,
                               const ElementwiseFunctor functor)
    {
        const auto a_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_global, a_grid_desc_m0.GetElementSpaceSize());
        const auto b_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_global, b_grid_desc_m0.GetElementSpaceSize());
        auto c_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_global, c_grid_desc_m0.GetElementSpaceSize());

        StaticBuffer<AddressSpaceEnum::Vgpr, ComputeDataType, M0PerThread, true> a_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, ComputeDataType, M0PerThread, true> b_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, ComputeDataType, M0PerThread, true> c_thread_buf;

        const auto thread_store_global_offset = CalculateElementwiseIndex();

        auto a_global_load =
            ThreadwiseTensorSliceTransfer_v2<ADataType,
                                             ComputeDataType,
                                             GridDesc_M0,
                                             decltype(thread_desc_m0),
                                             Sequence<M0PerThread>, // SliceLengths
                                             Sequence<0>,           // DimAccessOrder
                                             0,                     // SrcVectorDim
                                             AScalarPerVector,
                                             1, // SrcScalarStrideInVector
                                             false>{a_grid_desc_m0, thread_store_global_offset};

        auto b_global_load =
            ThreadwiseTensorSliceTransfer_v2<BDataType,
                                             ComputeDataType,
                                             GridDesc_M0,
                                             decltype(thread_desc_m0),
                                             Sequence<M0PerThread>, // SliceLengths
                                             Sequence<0>,           // DimAccessOrder
                                             0,                     // SrcVectorDim
                                             BScalarPerVector,
                                             1, // SrcScalarStrideInVector
                                             false>{b_grid_desc_m0, thread_store_global_offset};

        auto c_global_write =
            ThreadwiseTensorSliceTransfer_v1r3<ComputeDataType,
                                               CDataType,
                                               decltype(thread_desc_m0),
                                               GridDesc_M0,
                                               PassThrough,
                                               Sequence<M0PerThread>, // SliceLengths
                                               Sequence<0>,           // DimAccessOrder
                                               0,                     // DstVectorDim
                                               M0PerThread,
                                               InMemoryDataOperationEnum::Set,
                                               1, // DstScalarStrideInVector
                                               false>{
                c_grid_desc_m0, thread_store_global_offset, PassThrough{}};

        const index_t blockSize    = get_block_size();
        const index_t blockPerGrid = get_grid_size();
        const auto m0              = c_grid_desc_m0.GetLength(I0);
        const index_t loop_step    = blockPerGrid * blockSize * M0PerThread;
        const auto loop_step_index = make_multi_index(loop_step);

        index_t num_iter = m0 / (loop_step);
        do
        {
            // read and process M0PerThread elements
            a_global_load.Run(
                a_grid_desc_m0, a_global_buf, thread_desc_m0, make_tuple(I0), a_thread_buf);

            b_global_load.Run(
                b_grid_desc_m0, b_global_buf, thread_desc_m0, make_tuple(I0), b_thread_buf);

            static_for<0, M0PerThread, 1>{}([&](auto m) {
                constexpr auto offset = thread_desc_m0.CalculateOffset(make_tuple(m));
                functor(c_thread_buf(Number<offset>{}),
                        a_thread_buf(Number<offset>{}),
                        b_thread_buf(Number<offset>{}));
            });

            c_global_write.Run(thread_desc_m0,
                               make_tuple(I0), // SrcSliceOriginIdx
                               c_thread_buf,
                               c_grid_desc_m0,
                               c_global_buf);

            a_global_load.MoveSrcSliceWindow(a_grid_desc_m0, loop_step_index);
            b_global_load.MoveSrcSliceWindow(b_grid_desc_m0, loop_step_index);
            c_global_write.MoveDstSliceWindow(c_grid_desc_m0, loop_step_index);
        } while(--num_iter);
    }
};

} // namespace ck
