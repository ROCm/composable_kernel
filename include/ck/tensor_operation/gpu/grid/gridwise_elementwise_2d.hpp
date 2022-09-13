// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

template <typename GridwiseElementwise2dFunctor,
          typename InGrid2dDescTuple,
          typename OutGrid2dDescTuple,
          typename InDataTypePointerTuple,
          typename OutDataTypePointerTuple,
          typename ElementwiseOperation>
__global__ void kernel_elementwise_2d(const InGrid2dDescTuple in_grid_2d_desc_tuple,
                                      const OutGrid2dDescTuple out_grid_2d_desc_tuple,
                                      const InDataTypePointerTuple p_in_global_tuple,
                                      const OutDataTypePointerTuple p_out_global_tuple,
                                      const ElementwiseOperation elementwise_op)
{
    GridwiseElementwise2dFunctor::Run(in_grid_2d_desc_tuple,
                                      out_grid_2d_desc_tuple,
                                      p_in_global_tuple,
                                      p_out_global_tuple,
                                      elementwise_op);
}

template <typename InGrid2dDescTuple,
          typename OutGrid2dDescTuple,
          typename InDataTypePointerTuple,
          typename OutDataTypePointerTuple,
          typename ElementwiseOperation,
          index_t MPerThread,
	  index_t NPerThread,
          typename InScalarPerVectorSeq,
          typename OutScalarPerVectorSeq>
struct GridwiseElementwise_1D
{
    static constexpr index_t NumInput  = InDataTypePointerTuple::Size();
    static constexpr index_t NumOutput = OutDataTypePointerTuple::Size();

    static_assert(NumInput == InScalarPerVectorSeq::Size() &&
                      NumOutput == OutScalarPerVectorSeq::Size() &&
                      NumInput == InGrid2dDescTuple::Size() &&
                      NumOutput == OutGrid2dDescTuple::Size(),
                  "Tuple size is inconsistent with the number of in/out!");

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    static constexpr auto thread_buffer_desc_mn =
        make_naive_tensor_descriptor_packed(make_tuple(Number<MPerThread>{}, Number<NPerThread>{}));

    using PassThroughOp = tensor_operation::element_wise::PassThrough;

    __device__ static void Run(const InGrid2dDescTuple in_grid_2d_desc_tuple,
                               const OutGrid2dDescTuple out_grid_2d_desc_tuple,
                               const InDataTypePointerTuple p_in_global_tuple,
                               const OutDataTypePointerTuple p_out_global_tuple,
                               const ElementwiseOperation elementwise_op)
    {
        auto in_thread_buf_tuple = generate_tuple(
            [&](auto I) {
                using DataTypePointer = remove_cvref_t<decltype(InDataTypePointerTuple{}[I])>;
                using DataType        = remove_cv_t<remove_pointer_t<DataTypePointer>>;

                return StaticBuffer<AddressSpaceEnum::Vgpr, DataType, MPerThread*NPerThread, true>{};
            },
            Number<NumInput>{});

        auto out_thread_buf_tuple = generate_tuple(
            [&](auto I) {
                using DataTypePointer = remove_cvref_t<decltype(OutDataTypePointerTuple{}[I])>;
                using DataType        = remove_pointer_t<DataTypePointer>;

                return StaticBuffer<AddressSpaceEnum::Vgpr, DataType, MPerThread*NPerThread, true>{};
            },
            Number<NumOutput>{});

        auto in_global_buf_tuple = generate_tuple(
            [&](auto I) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_in_global_tuple[I], in_grid_2d_desc_tuple[I].GetElementSpaceSize());
            },
            Number<NumInput>{});

        auto out_global_buf_tuple = generate_tuple(
            [&](auto I) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_out_global_tuple[I], out_grid_2d_desc_tuple[I].GetElementSpaceSize());
            },
            Number<NumOutput>{});
 

        const index_t blockSize    = get_block_size();
        const index_t blockPerGrid = get_grid_size();
        const auto M               = in_grid_2d_desc_tuple[I0].GetLength(I0);
	const auto N               = in_grid_2d_desc_tuple[I1].GetLength(I1);
        const index_t loop_step    = blockPerGrid * blockSize * MPerThread*NPerThread;
        const auto loop_step_index = make_multi_index(loop_step);

	const index_t thread_global_id = thread_buffer_desc_mn.CalculateBottomIndex(make_multi_index(get_thread_global_1d_id));
	const auto thread_global_offset = make_multi_index(thread_global_id * MPerThread*NPerThread);

        auto in_global_load_tuple = generate_tuple(
            [&](auto I) {
                using DataTypePointer = remove_cvref_t<decltype(InDataTypePointerTuple{}[I])>;
                using DataType        = remove_cv_t<remove_pointer_t<DataTypePointer>>;

                return ThreadwiseTensorSliceTransfer_v2<DataType,
                                                        DataType,
                                                        decltype(in_grid_2d_desc_tuple[I]),
                                                        decltype(thread_buffer_desc_mn),
                                                        Sequence<MPerThread, NPerThread>, // SliceLengths
                                                        Sequence<0,1>,          // DimAccessOrder
                                                        1,                    // SrcVectorDim
                                                        InScalarPerVectorSeq::At(
                                                            I), // ScalarPerVector
                                                        1,      // SrcScalarStrideInVector
                                                        false>{in_grid_2d_desc_tuple[I],
                                                               thread_global_offset};
            },
            Number<NumInput>{});

        auto out_global_store_tuple = generate_tuple(
            [&](auto I) {
                using DataTypePointer = remove_cvref_t<decltype(OutDataTypePointerTuple{}[I])>;
                using DataType        = remove_pointer_t<DataTypePointer>;

                return ThreadwiseTensorSliceTransfer_v1r3<DataType,
                                                          DataType,
                                                          decltype(thread_buffer_desc_mn),
                                                          decltype(out_grid_2d_desc_tuple[I]),
                                                          PassThroughOp,
                                                          Sequence<MPerThread,NPerThread>, // SliceLengths
                                                          Sequence<1,0>,          // DimAccessOrder
                                                          0,                    // SrcVectorDim
                                                          OutScalarPerVectorSeq::At(I),
                                                          InMemoryDataOperationEnum::Set,
                                                          1,
                                                          false>(
                    out_grid_2d_desc_tuple[I], thread_global_offset, PassThroughOp{});
            },
            Number<NumOutput>{});

        index_t num_iter_m = M / (loop_step);
	index_t num_iter_n = N / (loop_step);
        do
        {
		do 
		{

            		static_for<0, NumInput, 1>{}([&](auto I) {
                		in_global_load_tuple(I).Run(in_grid_2d_desc_tuple[I],
                                            		in_global_buf_tuple[I],
                                            		thread_buffer_desc_mn,
                                            		make_tuple(I0,I1),
                                            		in_thread_buf_tuple(I));

                		in_global_load_tuple(I).MoveSrcSliceWindow(in_grid_2d_desc_tuple[I],
                                                           		loop_step_index);
            		});

            		static_for<0, MPerThread, 1>{}([&](auto iM) {
                		// get reference to in data
                		const auto in_data_refs = generate_tie(
                    		// return type should be lvalue
                    		[&](auto I) -> const auto& { return in_thread_buf_tuple(I)(iM); },
                    		Number<NumInput>{});

                		// get reference to dst data
                		auto out_data_refs = generate_tie(
                    		// return type should be lvalue
                    		[&](auto I) -> auto& { return out_thread_buf_tuple(I)(iM); },
                    		Number<NumOutput>{});

                		unpack2(elementwise_op, out_data_refs, in_data_refs);
            		});

            		static_for<0, NumOutput, 1>{}([&](auto I) {
                		out_global_store_tuple(I).Run(thread_buffer_desc_mn,
                                		         make_tuple(I0,I1),
                                              		out_thread_buf_tuple[I],
                                              		out_grid_2d_desc_tuple[I],
                                              		out_global_buf_tuple(I));

                		out_global_store_tuple(I).MoveDstSliceWindow(out_grid_2d_desc_tuple[I],
                                                             		loop_step_index);
            		});
		}(while --num_iter_n);
        } while(--num_iter_m);
    }
};

} // namespace ck
