// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_space_filling_curve.hpp"

namespace ck {

/**
 * @file threadwise_tensor_slice_transfer_util.hpp
 * @brief Shared helper class hierarchy for threadwise tensor slice transfer variants.
 *
 * Provides a three-tier inheritance structure:
 *
 * - @ref ThreadwiseTransferHelper_Base -- generic coordinate/descriptor utilities
 *   - @ref ThreadwiseTransferHelper_Serpentine -- serpentine (snake/zigzag) traversal
 *   - @ref ThreadwiseTransferHelper_SFC -- SpaceFillingCurve traversal
 */

namespace detail {

/** @brief Functor returning ScalarPerVector for dimension VectorDim, 1 otherwise. */
template <index_t VectorDim, index_t ScalarPerVector>
struct lambda_scalar_per_access
{
    __host__ __device__ constexpr auto operator()(index_t i) const
    {
        return (i == VectorDim) ? ScalarPerVector : 1;
    }
};

/** @brief Functor returning 1 for dimension VectorDim, 0 otherwise. */
template <index_t VectorDim>
struct lambda_scalar_step_in_vector
{
    __host__ __device__ constexpr auto operator()(index_t i) const
    {
        return (i == VectorDim) ? 1 : 0;
    }
};

/**
 * @brief Functor computing scalar-per-access for combined src/dst vector dimensions.
 * Returns lcm when both src and dst share the same vector dimension.
 */
template <index_t SrcVectorDim,
          index_t SrcScalarPerVector,
          index_t DstVectorDim,
          index_t DstScalarPerVector>
struct lambda_scalar_per_access_for_src_and_dst
{
    __host__ __device__ constexpr auto operator()(index_t i) const
    {
        if(i == SrcVectorDim && i == DstVectorDim)
        {
            return math::lcm(SrcScalarPerVector, DstScalarPerVector);
        }
        else if(i == SrcVectorDim)
        {
            return SrcScalarPerVector;
        }
        else if(i == DstVectorDim)
        {
            return DstScalarPerVector;
        }
        else
        {
            return 1;
        }
    }
};

} // namespace detail

/**
 * @brief Base helper with methods shared by all threadwise transfer variants.
 *
 * Both ThreadwiseTransferHelper_Serpentine and ThreadwiseTransferHelper_SFC
 * inherit from this class. Contains generic coordinate stepping, thread scratch
 * descriptor construction, and compile-time index constants.
 */
struct ThreadwiseTransferHelper_Base
{
    /**
     * @name Compile-time index constants
     * @{
     */
    static constexpr auto I0  = Number<0>{};
    static constexpr auto I1  = Number<1>{};
    static constexpr auto I2  = Number<2>{};
    static constexpr auto I3  = Number<3>{};
    static constexpr auto I4  = Number<4>{};
    static constexpr auto I5  = Number<5>{};
    static constexpr auto I6  = Number<6>{};
    static constexpr auto I7  = Number<7>{};
    static constexpr auto I8  = Number<8>{};
    static constexpr auto I10 = Number<10>{};
    static constexpr auto I12 = Number<12>{};
    static constexpr auto I13 = Number<13>{};
    static constexpr auto I14 = Number<14>{};
    static constexpr auto I16 = Number<16>{};
    /** @} */

    /**
     * @brief Move the slice window by a step, optionally fusing coordinate reset.
     *
     * If the coordinate was not reset after RunRead/RunWrite, the reset step is
     * added to the movement step to avoid a separate coordinate adjustment.
     *
     * @tparam ResetCoordinateAfterRun  Whether the coordinate was already reset.
     * @param desc                      Tensor descriptor.
     * @param coord                     Tensor coordinate to move (modified in place).
     * @param slice_origin_step_idx     Step index for the slice window movement.
     * @param get_reset_step            Callable returning the coordinate reset step.
     */
    template <typename Desc,
              typename Coord,
              bool ResetCoordinateAfterRun,
              typename StepIdx,
              typename GetCoordinateResetStepFunc>
    __host__ __device__ static void MoveSliceWindow(const Desc& desc,
                                                    Coord& coord,
                                                    const StepIdx& slice_origin_step_idx,
                                                    GetCoordinateResetStepFunc get_reset_step)
    {
        const auto adjusted_step_idx = ResetCoordinateAfterRun
                                           ? slice_origin_step_idx
                                           : slice_origin_step_idx + get_reset_step();

        const auto adjusted_step = make_tensor_coordinate_step(desc, adjusted_step_idx);

        move_tensor_coordinate(desc, coord, adjusted_step);
    }

    /**
     * @brief Build the thread-local scratch tensor descriptor.
     *
     * Creates a transformed tensor descriptor where the vector dimension is merged
     * with an additional dimension of size ScalarPerVector, enabling vector-typed
     * access to the scratch buffer.
     *
     * @tparam SliceLengths      Compile-time sequence of per-dimension slice lengths.
     * @tparam VectorDim         Which dimension is vectorized.
     * @tparam ScalarPerVector_  Number of scalars per vector load/store.
     * @return Transformed tensor descriptor for the thread scratch buffer.
     */
    template <typename SliceLengths, index_t VectorDim, index_t ScalarPerVector_>
    __host__ __device__ static constexpr auto ComputeThreadScratchDescriptor()
    {
        constexpr index_t nDim           = SliceLengths::Size();
        constexpr auto scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<VectorDim, ScalarPerVector_>{}, Number<nDim>{});

        constexpr auto access_lengths = SliceLengths{} / scalar_per_access;

        constexpr auto access_lengths_and_vector_length = container_push_back(
            sequence_to_tuple_of_number(access_lengths), Number<ScalarPerVector_>{});

        constexpr auto desc0 =
            make_naive_tensor_descriptor_packed(access_lengths_and_vector_length);

        constexpr auto transforms = generate_tuple(
            [&](auto i) {
                if constexpr(i == VectorDim)
                {
                    return make_merge_transform_v3_division_mod(
                        make_tuple(access_lengths_and_vector_length[i],
                                   access_lengths_and_vector_length[Number<nDim>{}]));
                }
                else
                {
                    return make_pass_through_transform(access_lengths_and_vector_length[i]);
                }
            },
            Number<nDim>{});

        constexpr auto low_dim_idss = generate_tuple(
            [&](auto i) {
                if constexpr(i == VectorDim)
                {
                    return Sequence<i.value, nDim>{};
                }
                else
                {
                    return Sequence<i.value>{};
                }
            },
            Number<nDim>{});

        constexpr auto up_dim_idss = generate_identity_sequences<nDim>();

        return transform_tensor_descriptor(desc0, transforms, low_dim_idss, up_dim_idss);
    }

    /**
     * @brief Compute forward (+1) coordinate steps for each dimension.
     *
     * Returns a tuple of nDim coordinate steps, where step[i] moves by
     * +scalar_per_access[i] in dimension i and 0 in all other dimensions.
     *
     * @param desc   Tensor descriptor.
     * @param scalar_per_access  Per-dimension access widths (Sequence type).
     */
    template <typename Desc, typename ScalarPerAccess>
    __host__ __device__ static constexpr auto
    ComputeForwardSteps(const Desc& desc, const ScalarPerAccess& scalar_per_access)
    {
        constexpr index_t nDim = ScalarPerAccess::Size();
        return generate_tuple(
            [&](auto i) {
                MultiIndex<nDim> step_idx;

                static_for<0, nDim, 1>{}(
                    [&](auto j) { step_idx(j) = (i.value == j.value) ? scalar_per_access[i] : 0; });

                return make_tensor_coordinate_step(desc, step_idx);
            },
            Number<nDim>{});
    }

    /**
     * @brief Compute backward (-1) coordinate steps for each dimension.
     *
     * Same as ComputeForwardSteps but with negated step values.
     *
     * @param desc   Tensor descriptor.
     * @param scalar_per_access  Per-dimension access widths (Sequence type).
     */
    template <typename Desc, typename ScalarPerAccess>
    __host__ __device__ static constexpr auto
    ComputeBackwardSteps(const Desc& desc, const ScalarPerAccess& scalar_per_access)
    {
        constexpr index_t nDim = ScalarPerAccess::Size();
        return generate_tuple(
            [&](auto i) {
                MultiIndex<nDim> step_idx;

                static_for<0, nDim, 1>{}([&](auto j) {
                    step_idx(j) = (i.value == j.value) ? -scalar_per_access[i] : 0;
                });

                return make_tensor_coordinate_step(desc, step_idx);
            },
            Number<nDim>{});
    }

    /**
     * @brief Create a tuple of default-constructed vector containers, one per data type.
     *
     * @tparam DataTypes       Tuple of data types (e.g., Tuple<fp16_t, fp16_t>).
     * @tparam ScalarPerVector Number of scalars per vector.
     * @return Tuple of vector_type_maker_t<DataType, ScalarPerVector> instances.
     */
    template <typename DataTypes, index_t ScalarPerVector>
    __host__ __device__ static auto MakeVectorContainerTuple()
    {
        auto data_types = DataTypes{};

        constexpr index_t num = data_types.Size();

        return generate_tuple(
            [&](auto i) {
                using DataType = remove_cvref_t<decltype(data_types[i])>;

                return vector_type_maker_t<DataType, ScalarPerVector>{};
            },
            Number<num>{});
    }
};

/**
 * @brief Serpentine (snake/zigzag) traversal helper.
 *
 * Provides methods for computing serpentine sweep directions, dimension movement
 * decisions, and coordinate reset steps used by the v3r1 family of transfer classes.
 *
 * Used by: ThreadwiseTensorSliceTransfer_v3r1, v3r2, v3r1_gather, v3r1_dequant.
 */
struct ThreadwiseTransferHelper_Serpentine : ThreadwiseTransferHelper_Base
{
    /**
     * @brief Binary decomposition of vector widths 0-16 into power-of-2 sub-load sizes.
     * Index N gives the sequence of sub-load widths whose sum equals N.
     * E.g. index 7 -> Sequence<I4, I2, I1> means loads of width 4, 2, 1.
     */
    using VectorSizeLookupTable = Tuple<Sequence<>,
                                        Sequence<I1>,
                                        Sequence<I2>,
                                        Sequence<I2, I1>,
                                        Sequence<I4>,
                                        Sequence<I4, I1>,
                                        Sequence<I4, I2>,
                                        Sequence<I4, I2, I1>,
                                        Sequence<I8>,
                                        Sequence<I8, I1>,
                                        Sequence<I8, I2>,
                                        Sequence<I8, I2, I1>,
                                        Sequence<I8, I4>,
                                        Sequence<I8, I4, I1>,
                                        Sequence<I8, I4, I2>,
                                        Sequence<I8, I4, I2, I1>,
                                        Sequence<I16>>;

    /**
     * @brief Starting offsets for each sub-load in VectorSizeLookupTable.
     * E.g. index 7 -> Sequence<I0, I4, I6> means offsets 0, 4, 6.
     */
    using VectorOffsetsLookupTable = Tuple<Sequence<>,
                                           Sequence<I0>,
                                           Sequence<I0>,
                                           Sequence<I0, I2>,
                                           Sequence<I0>,
                                           Sequence<I0, I4>,
                                           Sequence<I0, I4>,
                                           Sequence<I0, I4, I6>,
                                           Sequence<I0>,
                                           Sequence<I0, I8>,
                                           Sequence<I0, I8>,
                                           Sequence<I0, I8, I10>,
                                           Sequence<I0, I8>,
                                           Sequence<I0, I8, I12>,
                                           Sequence<I0, I8, I12>,
                                           Sequence<I0, I8, I12, I14>,
                                           Sequence<I0>>;

    /**
     * @brief Compute serpentine sweep direction for each dimension.
     *
     * Determines whether each dimension should be traversed forward or backward
     * based on the current position in the ordered access grid, implementing
     * a zigzag (serpentine) traversal pattern.
     *
     * @param ordered_access_idx      Current position in the ordered access grid.
     * @param ordered_access_lengths  Size of the ordered access grid per dimension.
     * @return Array of booleans: true = forward, false = backward.
     */
    template <typename OrderedAccessIdx, typename OrderedAccessLengths>
    __host__ __device__ static constexpr auto
    ComputeForwardSweep(const OrderedAccessIdx& ordered_access_idx,
                        const OrderedAccessLengths& ordered_access_lengths)
    {
        constexpr index_t nDim = OrderedAccessLengths::Size();
        static_assert(OrderedAccessIdx::Size() == nDim,
                      "ordered_access_idx and ordered_access_lengths must have same nDim");
        StaticallyIndexedArray_v2<bool, nDim> forward_sweep_;

        forward_sweep_(I0) = true;

        static_for<1, nDim, 1>{}([&](auto i) {
            index_t tmp = ordered_access_idx[I0];

            static_for<1, i, 1>{}(
                [&](auto j) { tmp = tmp * ordered_access_lengths[j] + ordered_access_idx[j]; });

            forward_sweep_(i) = tmp % 2 == 0;
        });

        return forward_sweep_;
    }

    /**
     * @brief Determine which dimensions need coordinate movement at a given iteration.
     *
     * A dimension moves when it hasn't reached its end and all higher-priority
     * (faster-varying) dimensions have completed their ranges.
     *
     * @param ordered_access_idx      Current position in the ordered access grid.
     * @param ordered_access_lengths  Size of the ordered access grid per dimension.
     * @return Array of booleans: true = move coordinate on this dimension.
     */
    template <typename OrderedAccessIdx, typename OrderedAccessLengths>
    __host__ __device__ static constexpr auto
    ComputeMoveOnDim(const OrderedAccessIdx& ordered_access_idx,
                     const OrderedAccessLengths& ordered_access_lengths)
    {
        constexpr index_t nDim = OrderedAccessLengths::Size();
        static_assert(OrderedAccessIdx::Size() == nDim,
                      "ordered_access_idx and ordered_access_lengths must have same nDim");
        StaticallyIndexedArray_v2<bool, nDim> move_on_dim_;

        static_for<0, nDim, 1>{}([&](auto i) {
            move_on_dim_(i) = ordered_access_idx[i] < ordered_access_lengths[i] - 1;

            static_for<i + 1, nDim, 1>{}([&](auto j) {
                move_on_dim_(i) &= ordered_access_idx[j] == ordered_access_lengths[j] - 1;
            });
        });

        return move_on_dim_;
    }

    /**
     * @brief Convert ordered access index to natural dimension order and apply scaling.
     *
     * @param ordered_access_idx      Current position in the ordered access grid.
     * @param ordered_access_lengths  Size of the ordered access grid per dimension.
     * @param forward_sweep           Per-dimension sweep direction.
     * @param dim_access_order        Mapping from ordered to natural dimension indices.
     * @param scalar_per_access       Per-dimension access widths.
     * @return MultiIndex in natural dimension order, scaled by scalar_per_access.
     */
    template <typename OrderedAccessIdx,
              typename OrderedAccessLengths,
              typename ForwardSweep,
              typename DimAccessOrder,
              typename ScalarPerAccess>
    __host__ __device__ static constexpr auto
    ComputeDataIndex(const OrderedAccessIdx& ordered_access_idx,
                     const OrderedAccessLengths& ordered_access_lengths,
                     const ForwardSweep& forward_sweep,
                     const DimAccessOrder& dim_access_order,
                     const ScalarPerAccess& scalar_per_access)
    {
        constexpr index_t nDim = ScalarPerAccess::Size();
        static_assert(OrderedAccessIdx::Size() == nDim,
                      "all arguments to ComputeDataIndex must have same nDim");
        MultiIndex<nDim> ordered_idx;

        static_for<0, nDim, 1>{}([&](auto i) {
            ordered_idx(i) = forward_sweep[i]
                                 ? ordered_access_idx[i]
                                 : ordered_access_lengths[i] - 1 - ordered_access_idx[i];
        });

        return container_reorder_given_old2new(ordered_idx, dim_access_order) * scalar_per_access;
    }

    /**
     * @brief Compute the coordinate step needed to return to the origin after traversal.
     *
     * Determines where the coordinate ends up after a full serpentine traversal,
     * then returns the negated position as the reset step.
     *
     * @tparam SliceLengths      Compile-time sequence of per-dimension slice lengths.
     * @tparam VectorDim         Which dimension is vectorized.
     * @tparam ScalarPerVector_  Number of scalars per vector load/store.
     * @tparam DimAccessOrder    Compile-time sequence mapping ordered to natural dims.
     * @return MultiIndex representing the step to reset the coordinate to the origin.
     */
    template <typename SliceLengths,
              index_t VectorDim,
              index_t ScalarPerVector_,
              typename DimAccessOrder>
    __host__ __device__ static constexpr auto ComputeCoordinateResetStep()
    {
        constexpr index_t nDim = SliceLengths::Size();
        static_assert(DimAccessOrder::Size() == nDim,
                      "SliceLengths and DimAccessOrder must have same nDim");
        constexpr auto scalar_per_access = generate_sequence(
            detail::lambda_scalar_per_access<VectorDim, ScalarPerVector_>{}, Number<nDim>{});

        constexpr auto access_lengths = SliceLengths{} / scalar_per_access;

        constexpr auto dim_access_order = DimAccessOrder{};

        constexpr auto ordered_access_lengths =
            container_reorder_given_new2old(access_lengths, dim_access_order);

        constexpr auto ordered_access_lengths_minus_1 = generate_tuple(
            [&](auto i) { return Number<ordered_access_lengths.At(i) - 1>{}; }, Number<nDim>{});
        constexpr auto forward_sweep =
            ComputeForwardSweep(ordered_access_lengths_minus_1, ordered_access_lengths);

        constexpr auto reset_step = [&]() {
            MultiIndex<nDim> ordered_idx;
            static_for<0, nDim, 1>{}([&](auto i) {
                ordered_idx(i) = forward_sweep[i] ? ordered_access_lengths[i] - 1 : 0;
            });

            auto data_idx =
                container_reorder_given_old2new(ordered_idx, dim_access_order) * scalar_per_access;

            MultiIndex<nDim> step;
            static_for<0, nDim, 1>{}([&](auto i) { step(i) = -data_idx[i]; });
            return step;
        }();

        return reset_step;
    }
};

/**
 * @brief SpaceFillingCurve traversal helper.
 *
 * Provides coordinate reset computation using SpaceFillingCurve's GetStepBetween
 * method, which computes the step from the last access position back to the origin.
 *
 * Used by: ThreadwiseTensorSliceTransfer v6r1, v6r1r2, v6r2, v6r3, v7r2, v7r3,
 *          v7r3_scatter.
 */
struct ThreadwiseTransferHelper_SFC : ThreadwiseTransferHelper_Base
{
    /**
     * @brief Compute the coordinate reset step using SpaceFillingCurve traversal.
     *
     * @tparam SliceLengths     Compile-time sequence of per-dimension slice lengths.
     * @tparam DimAccessOrder   Compile-time sequence defining dimension access order.
     * @tparam ScalarPerAccess  Compile-time sequence of per-dimension access widths.
     * @return MultiIndex representing the step from last access position to origin.
     */
    template <typename SliceLengths, typename DimAccessOrder, typename ScalarPerAccess>
    __host__ __device__ static constexpr auto ComputeSFCCoordinateResetStep()
    {
        using SFC = SpaceFillingCurve<SliceLengths, DimAccessOrder, remove_cv_t<ScalarPerAccess>>;

        constexpr auto num_access = SFC::GetNumOfAccess();
        if constexpr(num_access == 0)
        {
            return typename SFC::Index{};
        }
        else
        {
            return SFC::GetStepBetween(Number<num_access - 1>{}, Number<0>{});
        }
    }
};

} // namespace ck
