// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

// TODO: Write the description.
template <typename ThreadGroup,
          typename BlockSliceLengths,
          typename ThreadClusterLengths,
          typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          index_t SrcVectorDim,
          index_t DstVectorDim,
          index_t ScalarPerVector,
          index_t NumLdsBuffers = 1>
struct ThreadGroupTensorSliceTransfer_DirectLoad
{
    static constexpr index_t nDim = remove_reference_t<SrcDesc>::GetNumOfDimension();
    using Index                   = MultiIndex<nDim>;

    using SrcCoord = decltype(make_tensor_coordinate(SrcDesc{}, Index{}));
    using DstCoord = decltype(make_tensor_coordinate(DstDesc{}, Index{}));

    using SrcCoordStep = decltype(make_tensor_coordinate_step(SrcDesc{}, Index{}));
    using DstCoordStep = decltype(make_tensor_coordinate_step(DstDesc{}, Index{}));

    static constexpr auto I0 = Number<0>{};

    static constexpr auto block_slice_lengths    = BlockSliceLengths{};
    static constexpr auto thread_cluster_lengths = ThreadClusterLengths{};

    static constexpr auto thread_single_load_size = generate_sequence(
        detail::lambda_scalar_per_access<DstVectorDim, ScalarPerVector>{}, Number<nDim>{});
    // After a load, each thread moves by `thread_steps` instead of loading the next elements.
    // It makes whole wavefront load contiguous memory, what is required for direct loads.
    static constexpr auto thread_steps         = thread_cluster_lengths * thread_single_load_size;
    static constexpr auto thread_slice_lengths = block_slice_lengths / thread_steps;

    __device__ constexpr ThreadGroupTensorSliceTransfer_DirectLoad(
        const SrcDesc& src_desc,
        const Index& src_block_slice_origin,
        const DstDesc& dst_desc,
        const Index& dst_block_slice_origin)

    {
        static_assert(NumLdsBuffers == 1,
                      "Direct load transfer does not support multiple LDS buffers.");

        static_assert(ck::is_same_v<SrcData, DstData>,
                      "Direct load transfer does not support datatypes conversion. Source and "
                      "destination data types must be the same.");

        static_assert(
            DstVectorDim == nDim - 1,
            "Direct load transfer requires the destination vector dimension to be the last one.");

        static_assert(ScalarPerVector == 1 || SrcVectorDim == DstVectorDim,
                      "When loading more than one element per thread at once, the contiguous "
                      "dimension must be the same between source and destination.");

        constexpr auto dword_bytes           = 4;
        constexpr auto bytes_per_thread_load = ScalarPerVector * sizeof(SrcData);
        static_assert(bytes_per_thread_load == dword_bytes,
                      "Direct load transfer requires each thread to load exactly a single "
                      "DWORD of data.");

        static_assert(nDim == remove_cvref_t<SrcDesc>::GetNumOfDimension() &&
                          nDim == remove_cvref_t<DstDesc>::GetNumOfDimension() &&
                          nDim == ThreadClusterLengths::Size(),
                      "Inconsistent number of dimensions across lengths and descriptors.");

        static_assert(ThreadGroup::GetNumOfThread() >= thread_cluster_desc_.GetElementSize(),
                      "The number of threads cannot be less than the number of elements in "
                      "thread cluster lengths.");

        const auto thread_cluster_idx =
            thread_cluster_desc_.CalculateBottomIndex(make_multi_index(ThreadGroup::GetThreadId()));

        const auto thread_data_idx_begin = thread_cluster_idx * thread_single_load_size;

        SetSrcSliceOrigin(src_desc, src_block_slice_origin + thread_data_idx_begin);
        SetDstSliceOrigin(dst_desc, dst_block_slice_origin + thread_data_idx_begin);
    }

    __device__ void SetSrcSliceOrigin(const SrcDesc& src_desc, const Index& src_slice_origin_idx)
    {
        src_coord_        = make_tensor_coordinate(src_desc, src_slice_origin_idx);
        src_slice_origin_ = src_slice_origin_idx;
    }

    __device__ void SetDstSliceOrigin(const DstDesc& dst_desc, const Index& dst_slice_origin_idx)
    {
        dst_coord_        = make_tensor_coordinate(dst_desc, dst_slice_origin_idx);
        dst_slice_origin_ = dst_slice_origin_idx;
    }

    __device__ void ResetDstSliceWindow(const DstDesc& dst_desc)
    {
        dst_coord_ = make_tensor_coordinate(dst_desc, dst_slice_origin_);
    }

    template <typename SrcBuffer, typename DstBuffer>
    __device__ void Run(const SrcDesc& src_desc,
                        const SrcBuffer& src_buf,
                        const DstDesc& dst_desc,
                        DstBuffer& dst_buf)
    {
        static_assert(SrcBuffer::GetAddressSpace() == AddressSpaceEnum::Global,
                      "Source data must come from a global memory buffer.");
        static_assert(DstBuffer::GetAddressSpace() == AddressSpaceEnum::Lds,
                      "Destination data must be stored in an LDS memory buffer.");

        static_assert(
            ck::is_same_v<remove_cvref_t<typename SrcBuffer::type>, remove_cvref_t<SrcData>>,
            "SrcBuffer and SrcData data types must be consistent.");
        static_assert(
            ck::is_same_v<remove_cvref_t<typename DstBuffer::type>, remove_cvref_t<DstData>>,
            "DstBuffer and DstData data types must be consistent.");

        constexpr auto dst_access_lengths = thread_slice_lengths;

        const auto dst_forward_steps  = generate_steps(dst_desc, 1);
        const auto dst_backward_steps = generate_steps(dst_desc, -1);
        const auto src_forward_steps  = generate_steps(src_desc, 1);
        const auto src_backward_steps = generate_steps(src_desc, -1);

        // Loop over the destination block and copy data.
        static_ford<decltype(dst_access_lengths)>{}([&](auto ordered_dst_access_idx) {
            const auto src_offset = src_coord_.GetOffset();
            const auto dst_offset = dst_coord_.GetOffset();

            // Check if src data is not in the logic padding area.
            const bool is_src_valid =
                coordinate_has_valid_offset_assuming_visible_index_is_valid(src_desc, src_coord_);

            src_buf.template CopyTo<remove_cvref_t<decltype(dst_buf)>, ScalarPerVector>(
                dst_buf, src_offset, dst_offset, is_src_valid);

            constexpr auto move_on_dim = [&]() constexpr
            {
                StaticallyIndexedArray<bool, nDim> move_on_dim_;

                static_for<0, nDim, 1>{}([&](auto i) {
                    move_on_dim_(i) = ordered_dst_access_idx[i] < dst_access_lengths[i] - 1;

                    static_for<i + 1, nDim, 1>{}([&](auto j) {
                        move_on_dim_(i) &= ordered_dst_access_idx[j] == dst_access_lengths[j] - 1;
                    });
                });

                return move_on_dim_;
            }
            ();

            // Decide whether to move forward or backward.
            constexpr auto forward_sweep = [&]() {
                StaticallyIndexedArray<bool, nDim> forward_sweep_;

                forward_sweep_(I0) = true;

                static_for<1, nDim, 1>{}([&](auto i) {
                    index_t tmp = ordered_dst_access_idx[I0];

                    static_for<1, i, 1>{}([&](auto j) {
                        tmp = tmp * dst_access_lengths[j] + ordered_dst_access_idx[j];
                    });

                    forward_sweep_(i) = tmp % 2 == 0;
                });

                return forward_sweep_;
            }();

            static_for<0, nDim, 1>{}([&](auto i) {
                if constexpr(move_on_dim[i])
                {
                    if constexpr(forward_sweep[i])
                    {
                        move_tensor_coordinate(dst_desc, dst_coord_, dst_forward_steps[i]);
                        move_tensor_coordinate(src_desc, src_coord_, src_forward_steps[i]);
                    }
                    else
                    {
                        move_tensor_coordinate(dst_desc, dst_coord_, dst_backward_steps[i]);
                        move_tensor_coordinate(src_desc, src_coord_, src_backward_steps[i]);
                    }
                }
            });
        });

        // Reset the destination slice since the entire buffer has been already filled.
        ResetDstSliceWindow(dst_desc);
    }

    __device__ void MoveSrcSliceWindow(const SrcDesc& src_desc, const Index& step)
    {
        src_slice_origin_ = src_slice_origin_ + step;
        src_coord_        = make_tensor_coordinate(src_desc, src_slice_origin_);
    }

    template <typename DescType>
    __device__ auto generate_steps(const DescType& desc, int sign)
    {
        return generate_tuple(
            [&](auto i) {
                Index step_idx;

                static_for<0, nDim, 1>{}([&](auto j) {
                    step_idx(j) = (i.value == j.value) ? sign * thread_steps[i] : 0;
                });

                return make_tensor_coordinate_step(desc, step_idx);
            },
            Number<nDim>{});
    }

    private:
    static constexpr auto thread_cluster_desc_ = make_cluster_descriptor(ThreadClusterLengths{});

    SrcCoord src_coord_;
    DstCoord dst_coord_;
    Index src_slice_origin_;
    Index dst_slice_origin_;
};

} // namespace ck
