#ifndef CK_THREADWISE_DYNAMIC_TENSOR_SLICE_TRANSFER_HPP
#define CK_THREADWISE_DYNAMIC_TENSOR_SLICE_TRANSFER_HPP

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"

namespace ck {

// TODO: How to fix this? It uses an struct instead of lambda because lambda
// doesn't have constructor
template <index_t VectorDim, index_t ScalarPerVector>
struct lambda_scalar_per_access
{
    __host__ __device__ constexpr auto operator()(index_t i) const
    {
        return (i == VectorDim) ? ScalarPerVector : 1;
    }
};

template <index_t VectorDim>
struct lambda_scalar_step_in_vector
{
    __host__ __device__ constexpr auto operator()(index_t i) const
    {
        return (i == VectorDim) ? 1 : 0;
    }
};

// this version is less likely to have scratch memory issue, due to:
//   1. It does not keep reference to tensor descriptor
//   2. It does not construct new tensor coordinate for this->Run()
// Assume src_slice_origin_idx is 0
// TODO: support non-zero src_slice_oring_idx
template <typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename SliceLengths,
          typename DimAccessOrder,
          index_t DstVectorDim,
          index_t DstScalarPerVector,
          AddressSpace SrcAddressSpace,
          AddressSpace DstAddressSpace,
          InMemoryDataOperation DstInMemOp,
          index_t DstScalarStrideInVector,
          bool DstResetCoordinateAfterRun,
          typename std::enable_if<SrcDesc::IsKnownAtCompileTime(), bool>::type = false>
struct ThreadwiseDynamicTensorSliceTransfer_v1r3
{
    static constexpr index_t nDim = SliceLengths::Size();

    using Index = MultiIndex<nDim>;

    using DstCoord = decltype(make_dynamic_tensor_coordinate(DstDesc{}, Index{}));

    using DstCoordIterator = decltype(make_dynamic_tensor_coordinate_iterator(DstDesc{}, Index{}));

    __device__ constexpr ThreadwiseDynamicTensorSliceTransfer_v1r3(
        const DstDesc& dst_desc, const Index& dst_slice_origin_idx)
        : dst_slice_origin_coord_(make_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_idx))
    {
        static_assert(SrcDesc::IsKnownAtCompileTime(),
                      "wrong! SrcDesc need to known at compile-time");
    }

    __device__ void SetDstSliceOrigin(const DstDesc& dst_desc, const Index& dst_slice_origin_idx)
    {
        dst_slice_origin_coord_ = make_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_idx);
    }

    template <typename SrcSliceOriginIdx, typename DstIteratorHacks>
    __device__ void Run(const SrcDesc&,
                        const SrcSliceOriginIdx&,
                        const SrcData* p_src,
                        const DstDesc& dst_desc,
                        DstData* p_dst,
                        const DstIteratorHacks& dst_iterator_hacks)
    {
        static_assert(SrcDesc::IsKnownAtCompileTime(),
                      "wrong! SrcDesc need to known at compile-time");

        static_assert(
            is_known_at_compile_time<remove_cv_t<remove_reference_t<SrcSliceOriginIdx>>>::value,
            "wrong! SrcSliceOrigin need to known at compile-time");

        // SrcDesc and src_slice_origin_idx are known at compile-time
        constexpr auto src_desc             = remove_cv_t<remove_reference_t<SrcDesc>>{};
        constexpr auto src_slice_origin_idx = SrcSliceOriginIdx{};

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        // scalar per access on each dim
        // TODO: don't use lambda_scalar_per_access
        constexpr auto dst_scalar_per_access = generate_sequence(
            lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        constexpr auto dst_scalar_step_in_vector =
            generate_sequence(lambda_scalar_step_in_vector<DstVectorDim>{}, Number<nDim>{});

        constexpr auto access_lengths = SliceLengths{} / dst_scalar_per_access;

        constexpr auto dim_access_order = DimAccessOrder{};

        constexpr auto ordered_access_lengths =
            container_reorder_given_new2old(access_lengths, dim_access_order);

        // make forward iterators
        const auto dst_forward_iterators = generate_tuple(
            [&](auto i) {
                Index forward_step;

                static_for<0, nDim, 1>{}([&](auto j) {
                    forward_step(j) = (i.value == j.value) ? dst_scalar_per_access[i] : 0;
                });

                return make_dynamic_tensor_coordinate_iterator(
                    dst_desc, forward_step, dst_iterator_hacks[I0][i]);
            },
            Number<nDim>{});

        // make backward iterators
        const auto dst_backward_iterators = generate_tuple(
            [&](auto i) {
                Index backward_step;

                static_for<0, nDim, 1>{}([&](auto j) {
                    backward_step(j) = (i.value == j.value) ? -dst_scalar_per_access[i] : 0;
                });

                return make_dynamic_tensor_coordinate_iterator(
                    dst_desc, backward_step, dst_iterator_hacks[I1][i]);
            },
            Number<nDim>{});

        // loop over tensor and copy
        static_ford<decltype(ordered_access_lengths)>{}([&](auto ordered_access_idx) {
            // judge move forward or move backward
            constexpr auto forward_sweep = [&]() {
                StaticallyIndexedArray<bool, nDim> forward_sweep;

                forward_sweep(I0) = true;

                static_for<1, nDim, 1>{}([&](auto i) {
                    index_t tmp = ordered_access_idx[I0];

                    static_for<0, i, 1>{}([&](auto j) {
                        tmp = tmp * ordered_access_lengths[j] + ordered_access_idx[j];
                    });

                    forward_sweep(i) = tmp % 2 == 0;
                });

                return forward_sweep;
            }();

            // calculate dst data index
            constexpr auto dst_data_idx = [&]() {
                Index ordered_idx;

                static_for<0, nDim, 1>{}([&](auto i) {
                    ordered_idx(i) = forward_sweep[i]
                                         ? ordered_access_idx[i]
                                         : ordered_access_lengths[i] - 1 - ordered_access_idx[i];
                });

                auto dst_data_idx = container_reorder_given_old2new(ordered_idx, dim_access_order) *
                                    dst_scalar_per_access;

                return dst_data_idx;
            }();

            // copy data
            typename vector_type_maker<DstData, DstScalarPerVector>::type dst_vector;

            using dst_vector_t =
                typename vector_type_maker<DstData, DstScalarPerVector>::type::type;

            static_for<0, DstScalarPerVector, 1>{}([&](auto i) {
                constexpr index_t src_offset =
                    src_desc.CalculateOffset(to_multi_index(src_slice_origin_idx) + dst_data_idx +
                                             i * dst_scalar_step_in_vector);

                dst_vector.template AsType<DstData>()(i) =
                    type_convert<DstData>{}(p_src[Number<src_offset>{}]);
            });

            const bool is_dst_valid = coordinate_has_valid_offset_assuming_visible_index_is_valid(
                dst_desc, dst_slice_origin_coord_);

            if constexpr(SrcAddressSpace == AddressSpace::Vgpr &&
                         DstAddressSpace == AddressSpace::Global)
            {
#if CK_USE_AMD_BUFFER_ADDRESSING
                amd_buffer_store_v2<DstData, DstScalarPerVector>(
                    dst_vector.template AsType<dst_vector_t>()(Number<0>{}),
                    p_dst,
                    dst_slice_origin_coord_.GetOffset(),
                    is_dst_valid,
                    dst_desc.GetElementSpaceSize());
#else
                if(is_dst_valid)
                {
                    *reinterpret_cast<dst_vector_t*>(
                        &(p_dst[dst_slice_origin_coord_.GetOffset()])) =
                        dst_vector.template AsType<dst_vector_t>()[Number<0>{}];
                }
#endif
            }
            else
            {
                if(is_dst_valid)
                {
                    *reinterpret_cast<dst_vector_t*>(
                        &(p_dst[dst_slice_origin_coord_.GetOffset()])) =
                        dst_vector.template AsType<dst_vector_t>()[Number<0>{}];
                }
            }

            constexpr auto move_on_dim = [&]() constexpr
            {
                StaticallyIndexedArray<bool, nDim> move_on_dim;

                static_for<0, nDim, 1>{}([&](auto i) {
                    move_on_dim(i) = ordered_access_idx[i] < ordered_access_lengths[i] - 1;

                    static_for<i + 1, nDim, 1>{}([&](auto j) {
                        move_on_dim(i) &= ordered_access_idx[j] == ordered_access_lengths[j] - 1;
                    });
                });

                return move_on_dim;
            }
            ();

            // move
            static_for<0, nDim, 1>{}([&](auto i) {
                if constexpr(move_on_dim[i])
                {
                    if constexpr(forward_sweep[i])
                    {
                        move_dynamic_tensor_coordinate(dst_desc,
                                                       dst_slice_origin_coord_,
                                                       dst_forward_iterators[dim_access_order[i]]);
                    }
                    else
                    {
                        move_dynamic_tensor_coordinate(dst_desc,
                                                       dst_slice_origin_coord_,
                                                       dst_backward_iterators[dim_access_order[i]]);
                    }
                }
            });
        });

        // move dst coordinate back to slice origin (or not)
        if constexpr(DstResetCoordinateAfterRun)
        {
            const auto dst_reset_iterator =
                make_dynamic_tensor_coordinate_iterator(dst_desc, GetDstCoordinateResetStep());

            move_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_coord_, dst_reset_iterator);
        }
    }

    __device__ void Run(const SrcData* p_src, const DstDesc& dst_desc, DstData* p_dst)
    {
        constexpr index_t ntransform_dst = DstDesc::GetNumOfTransform();

        constexpr auto zeros = typename uniform_sequence_gen<ntransform_dst, 0>::type{};

        constexpr auto dst_iterator_hacks =
            make_tuple(generate_tuple([&](auto) { return zeros; }, Number<nDim>{}),
                       generate_tuple([&](auto) { return zeros; }, Number<nDim>{}));

        Run(p_src, dst_desc, p_dst, dst_iterator_hacks);
    }

    __device__ static constexpr auto GetDstCoordinateResetStep()
    {
        constexpr auto I0 = Number<0>{};

        // scalar per access on each dim
        // TODO: don't use lambda_scalar_per_access
        constexpr auto dst_scalar_per_access = generate_sequence(
            lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        constexpr auto access_lengths = SliceLengths{} / dst_scalar_per_access;

        constexpr auto dim_access_order = DimAccessOrder{};

        constexpr auto ordered_access_lengths =
            container_reorder_given_new2old(access_lengths, dim_access_order);

        // judge move forward or move backward during the last iteration
        constexpr auto forward_sweep = [&]() {
            StaticallyIndexedArray<bool, nDim> forward_sweep;

            forward_sweep(I0) = true;

            static_for<1, nDim, 1>{}([&](auto i) {
                index_t tmp = ordered_access_lengths[I0] - 1;

                static_for<0, i, 1>{}([&](auto j) {
                    tmp = tmp * ordered_access_lengths[j] + ordered_access_lengths[j] - 1;
                });

                forward_sweep(i) = tmp % 2 == 0;
            });

            return forward_sweep;
        }();

        // calculate dst data index after last iteration in Run(), if it has not being reset by
        // RunWrite()
        constexpr auto dst_data_idx = [&]() {
            Index ordered_idx;

            static_for<0, nDim, 1>{}([&](auto i) {
                ordered_idx(i) = forward_sweep[i] ? ordered_access_lengths[i] - 1 : 0;
            });

            auto dst_data_idx = container_reorder_given_old2new(ordered_idx, dim_access_order) *
                                dst_scalar_per_access;

            return dst_data_idx;
        }();

        //
        constexpr auto reset_dst_data_step = [&]() {
            Index reset_dst_data_step;

            static_for<0, nDim, 1>{}([&](auto i) { reset_dst_data_step(i) = -dst_data_idx[i]; });

            return reset_dst_data_step;
        }();

        return reset_dst_data_step;
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveDstSliceWindow(const DstDesc& dst_desc,
                                       const Index& dst_slice_origin_step_idx)
    {
        // if dst coord was not reset by Run(), then need to adjust the step here
        const auto adjusted_step_idx =
            DstResetCoordinateAfterRun ? dst_slice_origin_step_idx
                                       : dst_slice_origin_step_idx + GetDstCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step =
            make_dynamic_tensor_coordinate_iterator(dst_desc, adjusted_step_idx);

        move_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_coord_, adjusted_step);
    }

    private:
    DstCoord dst_slice_origin_coord_;
}; // namespace ck

// this version is less likely to have scratch memory issue, due to:
//   1. It does not keep reference to tensor descriptor
//   2. It does not construct new tensor coordinate for this->Run()
// Assume dst_slice_origin_idx is 0
template <typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename SliceLengths,
          typename DimAccessOrder,
          index_t SrcVectorDim,
          index_t SrcScalarPerVector,
          AddressSpace SrcAddressSpace,
          AddressSpace DstAddressSpace,
          index_t SrcScalarStrideInVector,
          bool SrcResetCoordinateAfterRun,
          typename std::enable_if<DstDesc::IsKnownAtCompileTime(), bool>::type = false>
struct ThreadwiseDynamicTensorSliceTransfer_v2
{
    static constexpr index_t nDim = SliceLengths::Size();

    using Index = MultiIndex<nDim>;

    using SrcCoord = decltype(make_dynamic_tensor_coordinate(SrcDesc{}, Index{}));

    using SrcCoordIterator = decltype(make_dynamic_tensor_coordinate_iterator(SrcDesc{}, Index{}));

    __device__ constexpr ThreadwiseDynamicTensorSliceTransfer_v2(const SrcDesc& src_desc,
                                                                 const Index& src_slice_origin_idx)
        : src_slice_origin_coord_(make_dynamic_tensor_coordinate(src_desc, src_slice_origin_idx))
    {
        static_assert(DstDesc::IsKnownAtCompileTime(),
                      "wrong! SrcDesc need to known at compile-time");
    }

    __device__ void SetDstSliceOrigin(const SrcDesc& src_desc, const Index& src_slice_origin_idx)
    {
        src_slice_origin_coord_ = make_dynamic_tensor_coordinate(src_desc, src_slice_origin_idx);
    }

    template <typename DstSliceOriginIdx, typename SrcIteratorHacks>
    __device__ void Run(const SrcDesc& src_desc,
                        const SrcData* p_src,
                        const DstDesc&,
                        const DstSliceOriginIdx&,
                        DstData* p_dst,
                        const SrcIteratorHacks& src_iterator_hacks)
    {
        static_assert(DstDesc::IsKnownAtCompileTime(),
                      "wrong! DstDesc need to known at compile-time");

        static_assert(
            is_known_at_compile_time<remove_cv_t<remove_reference_t<DstSliceOriginIdx>>>::value,
            "wrong! DstSliceOrigin need to known at compile-time");

        // DstDesc and dst_slice_origin_idx are known at compile-time
        constexpr auto dst_desc             = remove_cv_t<remove_reference_t<DstDesc>>{};
        constexpr auto dst_slice_origin_idx = DstSliceOriginIdx{};

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        // scalar per access on each dim
        // TODO: don't use lambda_scalar_per_access
        constexpr auto src_scalar_per_access = generate_sequence(
            lambda_scalar_per_access<SrcVectorDim, SrcScalarPerVector>{}, Number<nDim>{});

        constexpr auto src_scalar_step_in_vector =
            generate_sequence(lambda_scalar_step_in_vector<SrcVectorDim>{}, Number<nDim>{});

        constexpr auto access_lengths = SliceLengths{} / src_scalar_per_access;

        constexpr auto dim_access_order = DimAccessOrder{};

        constexpr auto ordered_access_lengths =
            container_reorder_given_new2old(access_lengths, dim_access_order);

        // make forward iterators
        const auto src_forward_iterators = generate_tuple(
            [&](auto i) {
                Index forward_step;

                static_for<0, nDim, 1>{}([&](auto j) {
                    forward_step(j) = (i.value == j.value) ? src_scalar_per_access[i] : 0;
                });

                return make_dynamic_tensor_coordinate_iterator(
                    src_desc, forward_step, src_iterator_hacks[I0][i]);
            },
            Number<nDim>{});

        // make backward iterators
        const auto src_backward_iterators = generate_tuple(
            [&](auto i) {
                Index backward_step;

                static_for<0, nDim, 1>{}([&](auto j) {
                    backward_step(j) = (i.value == j.value) ? -src_scalar_per_access[i] : 0;
                });

                return make_dynamic_tensor_coordinate_iterator(
                    src_desc, backward_step, src_iterator_hacks[I1][i]);
            },
            Number<nDim>{});

        // loop over tensor and copy
        static_ford<decltype(ordered_access_lengths)>{}([&](auto ordered_access_idx) {
            // judge move forward or move backward
            constexpr auto forward_sweep = [&]() {
                StaticallyIndexedArray<bool, nDim> forward_sweep;

                forward_sweep(I0) = true;

                static_for<1, nDim, 1>{}([&](auto i) {
                    index_t tmp = ordered_access_idx[I0];

                    static_for<0, i, 1>{}([&](auto j) {
                        tmp = tmp * ordered_access_lengths[j] + ordered_access_idx[j];
                    });

                    forward_sweep(i) = tmp % 2 == 0;
                });

                return forward_sweep;
            }();

            // calculate src data index
            constexpr auto src_data_idx = [&]() {
                Index ordered_idx;

                static_for<0, nDim, 1>{}([&](auto i) {
                    ordered_idx(i) = forward_sweep[i]
                                         ? ordered_access_idx[i]
                                         : ordered_access_lengths[i] - 1 - ordered_access_idx[i];
                });

                auto src_data_idx = container_reorder_given_old2new(ordered_idx, dim_access_order) *
                                    src_scalar_per_access;

                return src_data_idx;
            }();

            // copy data
            static_assert(DstAddressSpace == AddressSpace::Vgpr, "wrong! hardcode for vgpr dst");

            typename vector_type_maker<SrcData, SrcScalarPerVector>::type src_vector;

            using src_vector_t =
                typename vector_type_maker<SrcData, SrcScalarPerVector>::type::type;

            const bool is_src_valid = coordinate_has_valid_offset_assuming_visible_index_is_valid(
                src_desc, src_slice_origin_coord_);

            if constexpr(SrcAddressSpace == AddressSpace::Global)
            {
#if CK_USE_AMD_BUFFER_ADDRESSING
                src_vector.template AsType<src_vector_t>()(Number<0>{}) =
                    amd_buffer_load_v2<SrcData, SrcScalarPerVector>(
                        p_src,
                        src_slice_origin_coord_.GetOffset(),
                        is_src_valid,
                        src_desc.GetElementSpaceSize());
#else
                src_vector.template AsType<src_vector_t>()(Number<0>{}) =
                    is_src_valid ? *reinterpret_cast<const src_vector_t*>(
                                       &p_src[src_slice_origin_coord_.GetOffset()])
                                 : src_vector_t{0};
#endif
            }
            else
            {
                src_vector.template AsType<src_vector_t>()(Number<0>{}) =
                    is_src_valid ? *reinterpret_cast<const src_vector_t*>(
                                       &p_src[src_slice_origin_coord_.GetOffset()])
                                 : src_vector_t{0};
            }

            static_for<0, SrcScalarPerVector, 1>{}([&](auto i) {
                constexpr index_t dst_offset =
                    dst_desc.CalculateOffset(to_multi_index(dst_slice_origin_idx) + src_data_idx +
                                             i * src_scalar_step_in_vector);

                p_dst[Number<dst_offset>{}] = src_vector.template AsType<SrcData>()[i];
            });

            constexpr auto move_on_dim = [&]() constexpr
            {
                StaticallyIndexedArray<bool, nDim> move_on_dim;

                static_for<0, nDim, 1>{}([&](auto i) {
                    move_on_dim(i) = ordered_access_idx[i] < ordered_access_lengths[i] - 1;

                    static_for<i + 1, nDim, 1>{}([&](auto j) {
                        move_on_dim(i) &= ordered_access_idx[j] == ordered_access_lengths[j] - 1;
                    });
                });

                return move_on_dim;
            }
            ();

            // move
            static_for<0, nDim, 1>{}([&](auto i) {
                if constexpr(move_on_dim[i])
                {
                    if constexpr(forward_sweep[i])
                    {
                        move_dynamic_tensor_coordinate(src_desc,
                                                       src_slice_origin_coord_,
                                                       src_forward_iterators[dim_access_order[i]]);
                    }
                    else
                    {
                        move_dynamic_tensor_coordinate(src_desc,
                                                       src_slice_origin_coord_,
                                                       src_backward_iterators[dim_access_order[i]]);
                    }
                }
            });
        });

        // move src coordinate back to slice origin (or not)
        if constexpr(SrcResetCoordinateAfterRun)
        {
            const auto src_reset_iterator =
                make_dynamic_tensor_coordinate_iterator(src_desc, GetSrcCoordinateResetStep());

            move_dynamic_tensor_coordinate(src_desc, src_slice_origin_coord_, src_reset_iterator);
        }
    }

    __device__ void Run(const SrcDesc& src_desc, const SrcData* p_src, DstData* p_dst)
    {
        constexpr index_t ntransform_src = SrcDesc::GetNumOfTransform();

        constexpr auto zeros = typename uniform_sequence_gen<ntransform_src, 0>::type{};

        constexpr auto src_iterator_hacks =
            make_tuple(generate_tuple([&](auto) { return zeros; }, Number<nDim>{}),
                       generate_tuple([&](auto) { return zeros; }, Number<nDim>{}));

        Run(src_desc, p_src, p_dst, src_iterator_hacks);
    }

    __device__ static constexpr auto GetSrcCoordinateResetStep()
    {
        constexpr auto I0 = Number<0>{};

        // scalar per access on each dim
        // TODO: don't use lambda_scalar_per_access
        constexpr auto src_scalar_per_access = generate_sequence(
            lambda_scalar_per_access<SrcVectorDim, SrcScalarPerVector>{}, Number<nDim>{});

        constexpr auto access_lengths = SliceLengths{} / src_scalar_per_access;

        constexpr auto dim_access_order = DimAccessOrder{};

        constexpr auto ordered_access_lengths =
            container_reorder_given_new2old(access_lengths, dim_access_order);

        // judge move forward or move backward during the last iteration
        constexpr auto forward_sweep = [&]() {
            StaticallyIndexedArray<bool, nDim> forward_sweep;

            forward_sweep(I0) = true;

            static_for<1, nDim, 1>{}([&](auto i) {
                index_t tmp = ordered_access_lengths[I0] - 1;

                static_for<0, i, 1>{}([&](auto j) {
                    tmp = tmp * ordered_access_lengths[j] + ordered_access_lengths[j] - 1;
                });

                forward_sweep(i) = tmp % 2 == 0;
            });

            return forward_sweep;
        }();

        // calculate src data index after last iteration in Run(), if it has not being reset by
        // RunWrite()
        constexpr auto src_data_idx = [&]() {
            Index ordered_idx;

            static_for<0, nDim, 1>{}([&](auto i) {
                ordered_idx(i) = forward_sweep[i] ? ordered_access_lengths[i] - 1 : 0;
            });

            auto src_data_idx = container_reorder_given_old2new(ordered_idx, dim_access_order) *
                                src_scalar_per_access;

            return src_data_idx;
        }();

        //
        constexpr auto reset_src_data_step = [&]() {
            Index reset_src_data_step;

            static_for<0, nDim, 1>{}([&](auto i) { reset_src_data_step(i) = -src_data_idx[i]; });

            return reset_src_data_step;
        }();

        return reset_src_data_step;
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveSrcSliceWindow(const SrcDesc& src_desc,
                                       const Index& src_slice_origin_step_idx)
    {
        // if src coord was not reset by Run(), then need to adjust the step here
        const auto adjusted_step_idx =
            SrcResetCoordinateAfterRun ? src_slice_origin_step_idx
                                       : src_slice_origin_step_idx + GetSrcCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step =
            make_dynamic_tensor_coordinate_iterator(src_desc, adjusted_step_idx);

        move_dynamic_tensor_coordinate(src_desc, src_slice_origin_coord_, adjusted_step);
    }

    private:
    SrcCoord src_slice_origin_coord_;
}; // namespace ck

// this version does following things to avoid "alloca" in LLVM-IR, which would cause scratch memory
// and sometimes useless instructions
// 1. It does not keep reference to tensor descriptor
// 2. It does not construct new tensor coordinate for this->Run()
// 3. It does not use pointer for VGPR thread buffer
// 4. It calculate offset for thread buffer directly, instead of moving the coordinate
template <typename SliceLengths,
          InMemoryDataOperation DstInMemOp,
          typename SrcData,
          typename DstData,
          typename SrcDesc,
          typename DstDesc,
          typename SrcDimAccessOrder,
          typename DstDimAccessOrder,
          index_t SrcVectorDim,
          index_t DstVectorDim,
          index_t SrcScalarPerVector,
          index_t DstScalarPerVector,
          index_t SrcScalarStrideInVector,
          index_t DstScalarStrideInVector,
          AddressSpace SrcAddressSpace,
          AddressSpace DstAddressSpace,
          bool SrcResetCoordinateAfterRun, // control whether to move back src coordinate after each
                                           // RunRead(),  will be fused with MoveSrcSliceWindow to
                                           // save addr computation
          bool DstResetCoordinateAfterRun> // control whether to move back dst coordinate after each
                                           // RunWrite(),  will be fused with MoveDstSliceWindow to
                                           // save addr computation
struct ThreadwiseDynamicTensorSliceTransfer_v3
{
    static constexpr index_t nDim = SliceLengths::Size();
    using Index                   = MultiIndex<nDim>;

    using SrcCoord = decltype(make_dynamic_tensor_coordinate(SrcDesc{}, Index{}));
    using DstCoord = decltype(make_dynamic_tensor_coordinate(DstDesc{}, Index{}));

    using SrcCoordIterator = decltype(make_dynamic_tensor_coordinate_iterator(SrcDesc{}, Index{}));
    using DstCoordIterator = decltype(make_dynamic_tensor_coordinate_iterator(DstDesc{}, Index{}));

    __device__ constexpr ThreadwiseDynamicTensorSliceTransfer_v3(const SrcDesc& src_desc,
                                                                 const Index& src_slice_origin,
                                                                 const DstDesc& dst_desc,
                                                                 const Index& dst_slice_origin)
        : src_slice_origin_coord_(make_dynamic_tensor_coordinate(src_desc, src_slice_origin)),
          dst_slice_origin_coord_(make_dynamic_tensor_coordinate(dst_desc, dst_slice_origin))
    {
        static_assert(SrcAddressSpace == AddressSpace::Global or
                          SrcAddressSpace == AddressSpace::Lds,
                      "wrong!");
        static_assert(DstAddressSpace == AddressSpace::Global or
                          DstAddressSpace == AddressSpace::Lds,
                      "wrong!");
    }

    __device__ void SetSrcSliceOrigin(const SrcDesc& src_desc, const Index& src_slice_origin_idx)
    {
        src_slice_origin_coord_ = make_dynamic_tensor_coordinate(src_desc, src_slice_origin_idx);
    }

    __device__ void SetDstSliceOrigin(const DstDesc& dst_desc, const Index& dst_slice_origin_idx)
    {
        dst_slice_origin_coord_ = make_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_idx);
    }

    template <typename SrcIteratorHacks>
    __device__ void RunRead(const SrcDesc& src_desc,
                            const SrcData* p_src,
                            const SrcIteratorHacks& src_iterator_hacks)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        // scalar per access on each dim
        // TODO: don't use lambda_scalar_per_access
        constexpr auto src_scalar_per_access = generate_sequence(
            lambda_scalar_per_access<SrcVectorDim, SrcScalarPerVector>{}, Number<nDim>{});

        constexpr auto src_scalar_step_in_vector =
            generate_sequence(lambda_scalar_step_in_vector<SrcVectorDim>{}, Number<nDim>{});

        constexpr auto src_access_lengths = SliceLengths{} / src_scalar_per_access;

        constexpr auto src_dim_access_order = SrcDimAccessOrder{};

        constexpr auto ordered_src_access_lengths =
            container_reorder_given_new2old(src_access_lengths, src_dim_access_order);

        // make forward iterators
        const auto src_forward_iterators = generate_tuple(
            [&](auto i) {
                Index forward_step;

                static_for<0, nDim, 1>{}([&](auto j) {
                    forward_step(j) = (i.value == j.value) ? src_scalar_per_access[i] : 0;
                });

                return make_dynamic_tensor_coordinate_iterator(
                    src_desc, forward_step, src_iterator_hacks[I0][i]);
            },
            Number<nDim>{});

        // make backward iterators
        const auto src_backward_iterators = generate_tuple(
            [&](auto i) {
                Index backward_step;

                static_for<0, nDim, 1>{}([&](auto j) {
                    backward_step(j) = (i.value == j.value) ? -src_scalar_per_access[i] : 0;
                });

                return make_dynamic_tensor_coordinate_iterator(
                    src_desc, backward_step, src_iterator_hacks[I1][i]);
            },
            Number<nDim>{});

        // loop over tensor and copy
        static_ford<decltype(ordered_src_access_lengths)>{}([&](auto ordered_src_access_idx) {
            // judge move forward or move backward
            constexpr auto forward_sweep = [&]() {
                StaticallyIndexedArray<bool, nDim> forward_sweep;

                forward_sweep(I0) = true;

                static_for<1, nDim, 1>{}([&](auto i) {
                    index_t tmp = ordered_src_access_idx[I0];

                    static_for<0, i, 1>{}([&](auto j) {
                        tmp = tmp * ordered_src_access_lengths[j] + ordered_src_access_idx[j];
                    });

                    forward_sweep(i) = tmp % 2 == 0;
                });

                return forward_sweep;
            }();

            // calculate src data index
            constexpr auto src_data_idx = [&]() {
                Index ordered_idx;

                static_for<0, nDim, 1>{}([&](auto i) {
                    ordered_idx(i) = forward_sweep[i] ? ordered_src_access_idx[i]
                                                      : ordered_src_access_lengths[i] - 1 -
                                                            ordered_src_access_idx[i];
                });

                auto src_data_idx =
                    container_reorder_given_old2new(ordered_idx, src_dim_access_order) *
                    src_scalar_per_access;

                return src_data_idx;
            }();

            // copy data
            typename vector_type_maker<SrcData, SrcScalarPerVector>::type src_vector;

            using src_vector_t =
                typename vector_type_maker<SrcData, SrcScalarPerVector>::type::type;

            const bool is_src_valid = coordinate_has_valid_offset_assuming_visible_index_is_valid(
                src_desc, src_slice_origin_coord_);

            if constexpr(SrcAddressSpace == AddressSpace::Global)
            {
#if CK_USE_AMD_BUFFER_ADDRESSING
                src_vector.template AsType<src_vector_t>()(Number<0>{}) =
                    amd_buffer_load_v2<SrcData, SrcScalarPerVector>(
                        p_src,
                        src_slice_origin_coord_.GetOffset(),
                        is_src_valid,
                        src_desc.GetElementSpaceSize());
#else
                src_vector.template AsType<src_vector_t>()(Number<0>{}) =
                    is_src_valid ? *reinterpret_cast<const src_vector_t*>(
                                       &p_src[src_slice_origin_coord_.GetOffset()])
                                 : src_vector_t{0};
#endif
            }
            else
            {
                src_vector.template AsType<src_vector_t>()(Number<0>{}) =
                    is_src_valid ? *reinterpret_cast<const src_vector_t*>(
                                       &p_src[src_slice_origin_coord_.GetOffset()])
                                 : src_vector_t{0};
            }

            static_for<0, SrcScalarPerVector, 1>{}([&](auto i) {
                constexpr index_t buffer_offset =
                    buffer_desc_.CalculateOffset(src_data_idx + i * src_scalar_step_in_vector);

                buffer_(Number<buffer_offset>{}) = src_vector.template AsType<SrcData>()[i];
            });

            constexpr auto move_on_dim = [&]() constexpr
            {
                StaticallyIndexedArray<bool, nDim> move_on_dim;

                static_for<0, nDim, 1>{}([&](auto i) {
                    move_on_dim(i) = ordered_src_access_idx[i] < ordered_src_access_lengths[i] - 1;

                    static_for<i + 1, nDim, 1>{}([&](auto j) {
                        move_on_dim(i) &=
                            ordered_src_access_idx[j] == ordered_src_access_lengths[j] - 1;
                    });
                });

                return move_on_dim;
            }
            ();

            // move
            static_for<0, nDim, 1>{}([&](auto i) {
                if constexpr(move_on_dim[i])
                {
                    if constexpr(forward_sweep[i])
                    {
                        move_dynamic_tensor_coordinate(
                            src_desc,
                            src_slice_origin_coord_,
                            src_forward_iterators[src_dim_access_order[i]]);
                    }
                    else
                    {
                        move_dynamic_tensor_coordinate(
                            src_desc,
                            src_slice_origin_coord_,
                            src_backward_iterators[src_dim_access_order[i]]);
                    }
                }
            });
        });

        // move src coordinate back to slice origin (or not)
        if constexpr(SrcResetCoordinateAfterRun)
        {
            const auto src_reset_iterator =
                make_dynamic_tensor_coordinate_iterator(src_desc, GetSrcCoordinateResetStep());

            move_dynamic_tensor_coordinate(src_desc, src_slice_origin_coord_, src_reset_iterator);
        }
    }

    template <typename DstIteratorHacks>
    __device__ void
    RunWrite(const DstDesc& dst_desc, DstData* p_dst, const DstIteratorHacks& dst_iterator_hacks)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        // src scalar per access on each dim
        // TODO: don't use this
        constexpr auto dst_scalar_per_access = generate_sequence(
            lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        constexpr auto dst_scalar_step_in_vector =
            generate_sequence(lambda_scalar_step_in_vector<DstVectorDim>{}, Number<nDim>{});

        constexpr auto dst_access_lengths = SliceLengths{} / dst_scalar_per_access;

        constexpr auto dst_dim_access_order = DstDimAccessOrder{};

        constexpr auto ordered_dst_access_lengths =
            container_reorder_given_new2old(dst_access_lengths, dst_dim_access_order);

        // make forward iterators
        const auto dst_forward_iterators = generate_tuple(
            [&](auto i) {
                Index forward_step;

                static_for<0, nDim, 1>{}([&](auto j) {
                    forward_step(j) = (i.value == j.value) ? dst_scalar_per_access[i] : 0;
                });

                const auto forward_iterator = make_dynamic_tensor_coordinate_iterator(
                    dst_desc, forward_step, dst_iterator_hacks[I0][i]);

                return forward_iterator;
            },
            Number<nDim>{});

        // make backward iterators
        const auto dst_backward_iterators = generate_tuple(
            [&](auto i) {
                Index backward_step;

                static_for<0, nDim, 1>{}([&](auto j) {
                    backward_step(j) = (i.value == j.value) ? -dst_scalar_per_access[i] : 0;
                });

                const auto backward_iterator = make_dynamic_tensor_coordinate_iterator(
                    dst_desc, backward_step, dst_iterator_hacks[I1][i]);

                return backward_iterator;
            },
            Number<nDim>{});

        // loop over tensor and copy
        static_ford<decltype(ordered_dst_access_lengths)>{}([&](auto ordered_dst_access_idx) {
            // judge move forward or move backward
            constexpr auto forward_sweep = [&]() {
                StaticallyIndexedArray<bool, nDim> forward_sweep;

                forward_sweep(I0) = true;

                static_for<1, nDim, 1>{}([&](auto i) {
                    index_t tmp = ordered_dst_access_idx[I0];

                    static_for<0, i, 1>{}([&](auto j) {
                        tmp = tmp * ordered_dst_access_lengths[j] + ordered_dst_access_idx[j];
                    });

                    forward_sweep(i) = tmp % 2 == 0;
                });

                return forward_sweep;
            }();

            // calculate dst data index
            constexpr auto dst_data_idx = [&]() {
                Index ordered_idx;

                static_for<0, nDim, 1>{}([&](auto i) {
                    ordered_idx(i) = forward_sweep[i] ? ordered_dst_access_idx[i]
                                                      : ordered_dst_access_lengths[i] - 1 -
                                                            ordered_dst_access_idx[i];
                });

                auto dst_data_idx =
                    container_reorder_given_old2new(ordered_idx, dst_dim_access_order) *
                    dst_scalar_per_access;

                return dst_data_idx;
            }();

            // copy data
            // hardcoding for ds_write
            // TODO refactor transfer_data() to encapsulate this
            static_assert(DstAddressSpace == AddressSpace::Lds &&
                              DstInMemOp == InMemoryDataOperation::Set,
                          "wrong! hardcoded for ds_write");

            typename vector_type_maker<DstData, DstScalarPerVector>::type dst_vector;

            static_for<0, DstScalarPerVector, 1>{}([&](auto i) {
                constexpr index_t buffer_offset =
                    buffer_desc_.CalculateOffset(dst_data_idx + i * dst_scalar_step_in_vector);

                dst_vector.template AsType<DstData>()(i) = buffer_[Number<buffer_offset>{}];
            });

            using DstVectorType =
                typename vector_type_maker<DstData, DstScalarPerVector>::type::type;

            *reinterpret_cast<DstVectorType*>(p_dst + dst_slice_origin_coord_.GetOffset()) =
                dst_vector.template AsType<DstVectorType>()[Number<0>{}];

            constexpr auto move_on_dim = [&]() constexpr
            {
                StaticallyIndexedArray<bool, nDim> move_on_dim;

                static_for<0, nDim, 1>{}([&](auto i) {
                    move_on_dim(i) = ordered_dst_access_idx[i] < ordered_dst_access_lengths[i] - 1;

                    static_for<i + 1, nDim, 1>{}([&](auto j) {
                        move_on_dim(i) &=
                            ordered_dst_access_idx[j] == ordered_dst_access_lengths[j] - 1;
                    });
                });

                return move_on_dim;
            }
            ();

            // move
            static_for<0, nDim, 1>{}([&](auto i) {
                if constexpr(move_on_dim[i])
                {
                    if constexpr(forward_sweep[i])
                    {
                        move_dynamic_tensor_coordinate(
                            dst_desc,
                            dst_slice_origin_coord_,
                            dst_forward_iterators[dst_dim_access_order[i]]);
                    }
                    else
                    {
                        move_dynamic_tensor_coordinate(
                            dst_desc,
                            dst_slice_origin_coord_,
                            dst_backward_iterators[dst_dim_access_order[i]]);
                    }
                }
            });
        });

        // move dst coordinate back to slice origin (or not)
        if constexpr(DstResetCoordinateAfterRun)
        {
            const auto dst_reset_iterator =
                make_dynamic_tensor_coordinate_iterator(dst_desc, GetDstCoordinateResetStep());

            move_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_coord_, dst_reset_iterator);
        }
    }

    __device__ void RunRead(const SrcDesc& src_desc, const SrcData* p_src)
    {
        constexpr index_t ntransform_src = SrcDesc::GetNumOfTransform();

        constexpr auto zeros = typename uniform_sequence_gen<ntransform_src, 0>::type{};

        constexpr auto src_iterator_hacks =
            make_tuple(generate_tuple([&](auto) { return zeros; }, Number<nDim>{}),
                       generate_tuple([&](auto) { return zeros; }, Number<nDim>{}));

        RunRead(src_desc, p_src, src_iterator_hacks);
    }

    __device__ void RunWrite(const DstDesc& dst_desc, DstData* p_dst)
    {
        constexpr index_t ntransform_dst = DstDesc::GetNumOfTransform();

        constexpr auto zeros = typename uniform_sequence_gen<ntransform_dst, 0>::type{};

        constexpr auto dst_iterator_hacks =
            make_tuple(generate_tuple([&](auto) { return zeros; }, Number<nDim>{}),
                       generate_tuple([&](auto) { return zeros; }, Number<nDim>{}));

        RunWrite(dst_desc, p_dst, dst_iterator_hacks);
    }

    __device__ static constexpr auto GetSrcCoordinateResetStep()
    {
        constexpr auto I0 = Number<0>{};

        // scalar per access on each dim
        // TODO: don't use lambda_scalar_per_access
        constexpr auto src_scalar_per_access = generate_sequence(
            lambda_scalar_per_access<SrcVectorDim, SrcScalarPerVector>{}, Number<nDim>{});

        constexpr auto src_access_lengths = SliceLengths{} / src_scalar_per_access;

        constexpr auto src_dim_access_order = SrcDimAccessOrder{};

        constexpr auto ordered_src_access_lengths =
            container_reorder_given_new2old(src_access_lengths, src_dim_access_order);

        // judge move forward or move backward during the last iteration
        constexpr auto forward_sweep = [&]() {
            StaticallyIndexedArray<bool, nDim> forward_sweep;

            forward_sweep(I0) = true;

            static_for<1, nDim, 1>{}([&](auto i) {
                index_t tmp = ordered_src_access_lengths[I0] - 1;

                static_for<0, i, 1>{}([&](auto j) {
                    tmp = tmp * ordered_src_access_lengths[j] + ordered_src_access_lengths[j] - 1;
                });

                forward_sweep(i) = tmp % 2 == 0;
            });

            return forward_sweep;
        }();

        // calculate src data index after last iteration in RunRead(), if it has not being reset by
        // RunRead()
        constexpr auto src_data_idx = [&]() {
            Index ordered_idx;

            static_for<0, nDim, 1>{}([&](auto i) {
                ordered_idx(i) = forward_sweep[i] ? ordered_src_access_lengths[i] - 1 : 0;
            });

            auto src_data_idx = container_reorder_given_old2new(ordered_idx, src_dim_access_order) *
                                src_scalar_per_access;

            return src_data_idx;
        }();

        //
        constexpr auto reset_src_data_step = [&]() {
            Index reset_src_data_step;

            static_for<0, nDim, 1>{}([&](auto i) { reset_src_data_step(i) = -src_data_idx[i]; });

            return reset_src_data_step;
        }();

        return reset_src_data_step;
    }

    __device__ static constexpr auto GetDstCoordinateResetStep()
    {
        constexpr auto I0 = Number<0>{};

        // scalar per access on each dim
        // TODO: don't use lambda_scalar_per_access
        constexpr auto dst_scalar_per_access = generate_sequence(
            lambda_scalar_per_access<DstVectorDim, DstScalarPerVector>{}, Number<nDim>{});

        constexpr auto dst_access_lengths = SliceLengths{} / dst_scalar_per_access;

        constexpr auto dst_dim_access_order = DstDimAccessOrder{};

        constexpr auto ordered_dst_access_lengths =
            container_reorder_given_new2old(dst_access_lengths, dst_dim_access_order);

        // judge move forward or move backward during the last iteration
        constexpr auto forward_sweep = [&]() {
            StaticallyIndexedArray<bool, nDim> forward_sweep;

            forward_sweep(I0) = true;

            static_for<1, nDim, 1>{}([&](auto i) {
                index_t tmp = ordered_dst_access_lengths[I0] - 1;

                static_for<0, i, 1>{}([&](auto j) {
                    tmp = tmp * ordered_dst_access_lengths[j] + ordered_dst_access_lengths[j] - 1;
                });

                forward_sweep(i) = tmp % 2 == 0;
            });

            return forward_sweep;
        }();

        // calculate dst data index after last iteration in RunWrite(), if it has not being reset by
        // RunWrite()
        constexpr auto dst_data_idx = [&]() {
            Index ordered_idx;

            static_for<0, nDim, 1>{}([&](auto i) {
                ordered_idx(i) = forward_sweep[i] ? ordered_dst_access_lengths[i] - 1 : 0;
            });

            auto dst_data_idx = container_reorder_given_old2new(ordered_idx, dst_dim_access_order) *
                                dst_scalar_per_access;

            return dst_data_idx;
        }();

        //
        constexpr auto reset_dst_data_step = [&]() {
            Index reset_dst_data_step;

            static_for<0, nDim, 1>{}([&](auto i) { reset_dst_data_step(i) = -dst_data_idx[i]; });

            return reset_dst_data_step;
        }();

        return reset_dst_data_step;
    }

    // src_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveSrcSliceWindow(const SrcDesc& src_desc,
                                       const Index& src_slice_origin_step_idx)
    {
        // if src coord was not reset by RunRead(), then need to adjust the step here
        const auto adjusted_step_idx =
            SrcResetCoordinateAfterRun ? src_slice_origin_step_idx
                                       : src_slice_origin_step_idx + GetSrcCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step =
            make_dynamic_tensor_coordinate_iterator(src_desc, adjusted_step_idx);

        move_dynamic_tensor_coordinate(src_desc, src_slice_origin_coord_, adjusted_step);
    }

    // src_slice_origin_step_idx need to be known at compile-time, for performance reason
    template <typename SrcMoveSliceWindowIteratorHack>
    __device__ void
    MoveSrcSliceWindow(const SrcDesc& src_desc,
                       const Index& src_slice_origin_step_idx,
                       const SrcMoveSliceWindowIteratorHack& src_move_slice_window_iterator_hack)
    {
        // if src coord was not reset by RunRead(), then need to adjust the step here
        const auto adjusted_step_idx =
            SrcResetCoordinateAfterRun ? src_slice_origin_step_idx
                                       : src_slice_origin_step_idx + GetSrcCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step = make_dynamic_tensor_coordinate_iterator(
            src_desc, adjusted_step_idx, src_move_slice_window_iterator_hack);

        move_dynamic_tensor_coordinate(src_desc, src_slice_origin_coord_, adjusted_step);
    }

    // dst_slice_origin_step_idx need to be known at compile-time, for performance reason
    __device__ void MoveDstSliceWindow(const DstDesc& dst_desc,
                                       const Index& dst_slice_origin_step_idx)
    {
        // if dst coord was not reset by RunWrite(), then need to adjust the step here
        const auto adjusted_step_idx =
            DstResetCoordinateAfterRun ? dst_slice_origin_step_idx
                                       : dst_slice_origin_step_idx + GetDstCoordinateResetStep();

        // is it OK to construct a new step every time?
        const auto adjusted_step =
            make_dynamic_tensor_coordinate_iterator(dst_desc, adjusted_step_idx);

        move_dynamic_tensor_coordinate(dst_desc, dst_slice_origin_coord_, adjusted_step);
    }

    private:
    static constexpr auto buffer_desc_ =
        make_dynamic_naive_tensor_descriptor_packed_v2(sequence_to_tuple_of_number(SliceLengths{}));

    static constexpr auto buffer_size_ = buffer_desc_.GetElementSpaceSize();

    StaticallyIndexedArray<SrcData, buffer_size_> buffer_;

    SrcCoord src_slice_origin_coord_;
    DstCoord dst_slice_origin_coord_;
};

} // namespace ck
#endif
