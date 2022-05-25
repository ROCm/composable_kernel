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
          typename AGridDesc_M,
          typename BGridDesc_M,
          typename CGridDesc_M,
          typename ElementwiseFunctor>
__global__ void kernel_binary_elementwise_1d(const ADataType* __restrict__ p_a_global,
                                             const BDataType* __restrict__ p_b_global,
                                             CDataType* __restrict__ p_c_global,
                                             const AGridDesc_M a_grid_desc_m,
                                             const BGridDesc_M b_grid_desc_m,
                                             const CGridDesc_M c_grid_desc_m,
                                             const ElementwiseFunctor functor)
{
    GridwiseBinEltwise::Run(
        p_a_global, p_b_global, p_c_global, a_grid_desc_m, b_grid_desc_m, c_grid_desc_m, functor);
}

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ComputeDataType,
          typename AGridDesc_M,
          typename BGridDesc_M,
          typename CGridDesc_M,
          typename ElementwiseFunctor,
          index_t MPerThread,
          index_t AScalarPerVector,
          index_t BScalarPerVector,
          index_t CScalarPerVector>
struct GridwiseBinaryElementwise_1D
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto thread_desc_m =
        make_naive_tensor_descriptor_packed(make_tuple(Number<MPerThread>{}));

    using PassThrough = tensor_operation::element_wise::PassThrough;

    static __device__ auto CalculateElementwiseIndex()
    {
        const index_t global_thread_id = get_thread_global_1d_id();
        return make_multi_index(global_thread_id * MPerThread);
    }

    __device__ static void Run(const ADataType* __restrict__ p_a_global,
                               const BDataType* __restrict__ p_b_global,
                               CDataType* __restrict__ p_c_global,
                               const AGridDesc_M a_grid_desc_m,
                               const BGridDesc_M b_grid_desc_m,
                               const CGridDesc_M c_grid_desc_m,
                               const ElementwiseFunctor functor)
    {
        const auto a_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_a_global, a_grid_desc_m.GetElementSpaceSize());
        const auto b_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_b_global, b_grid_desc_m.GetElementSpaceSize());
        auto c_global_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_global, c_grid_desc_m.GetElementSpaceSize());

        StaticBuffer<AddressSpaceEnum::Vgpr, ComputeDataType, MPerThread, true> a_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, ComputeDataType, MPerThread, true> b_thread_buf;
        StaticBuffer<AddressSpaceEnum::Vgpr, ComputeDataType, MPerThread, true> c_thread_buf;

        const auto thread_store_global_offset = CalculateElementwiseIndex();

        auto a_global_load =
            ThreadwiseTensorSliceTransfer_v2<ADataType,
                                             ComputeDataType,
                                             AGridDesc_M,
                                             decltype(thread_desc_m),
                                             Sequence<MPerThread>, // SliceLengths
                                             Sequence<0>,          // DimAccessOrder
                                             0,                    // SrcVectorDim
                                             AScalarPerVector,     // ScalarPerVector
                                             1,                    // SrcScalarStrideInVector
                                             false>{a_grid_desc_m, thread_store_global_offset};

        auto b_global_load =
            ThreadwiseTensorSliceTransfer_v2<BDataType,
                                             ComputeDataType,
                                             BGridDesc_M,
                                             decltype(thread_desc_m),
                                             Sequence<MPerThread>, // SliceLengths
                                             Sequence<0>,          // DimAccessOrder
                                             0,                    // SrcVectorDim
                                             BScalarPerVector,     // ScalarPerVector
                                             1,                    // SrcScalarStrideInVector
                                             false>{b_grid_desc_m, thread_store_global_offset};

        auto c_global_write =
            ThreadwiseTensorSliceTransfer_v1r3<ComputeDataType,
                                               CDataType,
                                               decltype(thread_desc_m),
                                               CGridDesc_M,
                                               PassThrough,
                                               Sequence<MPerThread>, // SliceLengths
                                               Sequence<0>,          // DimAccessOrder
                                               0,                    // DstVectorDim
                                               CScalarPerVector,     // ScalarPerVector
                                               InMemoryDataOperationEnum::Set,
                                               1, // DstScalarStrideInVector
                                               false>{
                c_grid_desc_m, thread_store_global_offset, PassThrough{}};

        const index_t blockSize    = get_block_size();
        const index_t blockPerGrid = get_grid_size();
        const auto M               = c_grid_desc_m.GetLength(I0);
        const index_t loop_step    = blockPerGrid * blockSize * MPerThread;
        const auto loop_step_index = make_multi_index(loop_step);

        index_t num_iter = M / (loop_step);
        do
        {
            // read and process MPerThread elements
            a_global_load.Run(
                a_grid_desc_m, a_global_buf, thread_desc_m, make_tuple(I0), a_thread_buf);

            b_global_load.Run(
                b_grid_desc_m, b_global_buf, thread_desc_m, make_tuple(I0), b_thread_buf);

            static_for<0, MPerThread, 1>{}([&](auto m) {
                constexpr auto offset = thread_desc_m.CalculateOffset(make_tuple(m));
                functor(c_thread_buf(Number<offset>{}),
                        a_thread_buf(Number<offset>{}),
                        b_thread_buf(Number<offset>{}));
            });

            c_global_write.Run(thread_desc_m,
                               make_tuple(I0), // SrcSliceOriginIdx
                               c_thread_buf,
                               c_grid_desc_m,
                               c_global_buf);

            a_global_load.MoveSrcSliceWindow(a_grid_desc_m, loop_step_index);
            b_global_load.MoveSrcSliceWindow(b_grid_desc_m, loop_step_index);
            c_global_write.MoveDstSliceWindow(c_grid_desc_m, loop_step_index);
        } while(--num_iter);
    }
};

} // namespace ck
